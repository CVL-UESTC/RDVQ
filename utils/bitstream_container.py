"""Compact RDVQ bitstream container helpers.

The container stores only actual rANS payload bytes plus minimal per-image and
per-record metadata. Model weights, codebook, entropy settings, and dataset
identity are assumed to be agreed by encoder and decoder configuration.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

MAGIC = b"RDVQBN2\0"
LEAN_MAGIC = b"RDVQBN3\0"
HEADER_STRUCT = struct.Struct("<BBHII")
RECORD_STRUCT = struct.Struct("<BIIII")
LEAN_STRUCT = struct.Struct("<BIII")
BACKEND_IDS = {"tensor": 1, "compressai": 2, "mixed": 3, "raw": 3, "empty": 0}
BACKEND_NAMES = {value: key for key, value in BACKEND_IDS.items()}


@dataclass(frozen=True)
class EncodedBinRecord:
    """One patch/image payload record inside an RDVQ compact bin."""

    patch_index: int
    backend: str
    compressai: bytes = b""
    tensor_top: bytes = b""
    tensor_residual: bytes = b""

    @property
    def payload_bytes(self) -> int:
        return len(self.compressai) + len(self.tensor_top) + len(self.tensor_residual)


def encoded_stream_to_bin_record(encoded_stream, patch_index=0) -> EncodedBinRecord:
    """Collect raw rANS payloads from an EncodedEntropyStream.

    The stats dict keeps EncodedEntropyStream hidden under
    _encoded_entropy_stream while rate metrics are accumulated. This function
    is the boundary where that runtime object becomes serializable bytes.
    """
    return EncodedBinRecord(
        patch_index=int(patch_index),
        backend=str(encoded_stream.backend),
        compressai=b"" if encoded_stream.compressai_stream is None else bytes(encoded_stream.compressai_stream),
        tensor_top=b"" if encoded_stream.tensor_top_stream is None else bytes(encoded_stream.tensor_top_stream),
        tensor_residual=b"" if encoded_stream.tensor_residual_stream is None else bytes(encoded_stream.tensor_residual_stream),
    )


def attach_bin_stream(coding_stats, patch_index=0):
    """Move hidden encoded stream metadata into compact-bin records."""
    encoded_stream = coding_stats.pop("_encoded_entropy_stream", None)
    if encoded_stream is None:
        coding_stats["_bin_streams"] = []
        return coding_stats
    coding_stats["_bin_streams"] = [encoded_stream_to_bin_record(encoded_stream, patch_index)]
    return coding_stats


def collect_bin_streams(target, stats):
    """Append per-patch compact-bin records from one stats dict into another."""
    streams = stats.get("_bin_streams") or []
    if streams:
        target.setdefault("_bin_streams", []).extend(streams)


def _as_record(record) -> EncodedBinRecord:
    if isinstance(record, EncodedBinRecord):
        return record
    return EncodedBinRecord(
        patch_index=int(record["patch_index"]),
        backend=str(record["backend"]),
        compressai=bytes(record.get("compressai", b"")),
        tensor_top=bytes(record.get("tensor_top", b"")),
        tensor_residual=bytes(record.get("tensor_residual", b"")),
    )


def _can_use_lean_tensor_container(records, split_image: bool) -> bool:
    """Use the lean format only when sender/receiver can infer all codec state."""

    if split_image or len(records) != 1:
        return False
    record = records[0]
    return record.backend == "tensor" and not record.compressai


def _write_lean_tensor_bin(path, record: EncodedBinRecord, *, transfer_slices, original_shape):
    """Write a single-image tensor rANS bin with only negotiated metadata.

    Layout:
      magic[8] = b"RDVQBN3\0"
      transfer_slices:uint8
      original_h:uint32, original_w:uint32
      tensor_top_len:uint32
      payload blobs: tensor_top, tensor_residual.

    Backend, model, top-k, precision, and entropy mode are assumed to be shared
    by encoder and decoder configuration.
    """

    transfer = int(transfer_slices)
    if not 0 <= transfer <= 255:
        raise ValueError(f"transfer_slices must fit uint8 in lean compact bin, got {transfer}")
    if len(record.tensor_top) > 0xFFFFFFFF:
        raise ValueError("tensor_top stream is too large for lean compact bin")

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    original_h = int(original_shape[-2])
    original_w = int(original_shape[-1])
    with out_path.open("wb") as f:
        f.write(LEAN_MAGIC)
        f.write(LEAN_STRUCT.pack(transfer, original_h, original_w, len(record.tensor_top)))
        f.write(record.tensor_top)
        f.write(record.tensor_residual)
    return out_path


def write_rdvq_bin(path, records, *, transfer_slices, original_shape, split_image=False):
    """Write an RDVQ bitstream container.

    Single-record tensor payloads use the lean RDVQBN3 layout above; split or
    mixed-backend payloads fall back to the v2 layout below.

    RDVQBN2 layout, little-endian:
      magic[8] = b"RDVQBN2\0"
      flags:uint8  bit0=split image, bit1=has compressai,
                   bit2=has tensor top, bit3=has tensor residual
      transfer_slices:uint8
      record_count:uint16
      original_h:uint32, original_w:uint32
      repeated records:
        backend:uint8  1=tensor, 2=compressai, 3=mixed/raw
        patch_index:uint32
        compressai_len:uint32, tensor_top_len:uint32, tensor_residual_len:uint32
      payload blobs in record order: compressai, tensor_top, tensor_residual.
    """
    # records may come from one whole image or many split patches. Normalize
    # them first so the writer can choose a lean single-record layout or the
    # general multi-record layout.
    records = [_as_record(record) for record in records]
    if not records:
        return None
    if _can_use_lean_tensor_container(records, split_image):
        return _write_lean_tensor_bin(path, records[0], transfer_slices=transfer_slices, original_shape=original_shape)
    if len(records) > 65535:
        raise ValueError(f"too many bitstream records for compact bin: {len(records)}")

    transfer = int(transfer_slices)
    if not 0 <= transfer <= 255:
        raise ValueError(f"transfer_slices must fit uint8 in compact bin, got {transfer}")

    flags = 1 if split_image else 0
    for record in records:
        flags |= 2 if record.compressai else 0
        flags |= 4 if record.tensor_top else 0
        flags |= 8 if record.tensor_residual else 0
        for name, payload in (
            ("compressai", record.compressai),
            ("tensor_top", record.tensor_top),
            ("tensor_residual", record.tensor_residual),
        ):
            if len(payload) > 0xFFFFFFFF:
                raise ValueError(f"{name} stream is too large for compact bin")

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    original_h = int(original_shape[-2])
    original_w = int(original_shape[-1])

    with out_path.open("wb") as f:
        f.write(MAGIC)
        f.write(HEADER_STRUCT.pack(flags, transfer, len(records), original_h, original_w))
        for record in records:
            f.write(RECORD_STRUCT.pack(
                BACKEND_IDS.get(record.backend, 3),
                int(record.patch_index),
                len(record.compressai),
                len(record.tensor_top),
                len(record.tensor_residual),
            ))
        for record in records:
            f.write(record.compressai)
            f.write(record.tensor_top)
            f.write(record.tensor_residual)
    return out_path


def save_image_bitstream_bin(output_dir, img_name, image_stats, *, transfer_slices, original_shape, split_image=False):
    """Save one image's compact bitstream under ``<output_dir>/bin``."""
    records = image_stats.get("_bin_streams") or []
    if not records:
        return None
    out_path = Path(output_dir) / "bin" / f"{Path(img_name).stem}.bin"
    return write_rdvq_bin(
        out_path,
        records,
        transfer_slices=transfer_slices,
        original_shape=original_shape,
        split_image=split_image,
    )


def read_rdvq_bin(path):
    """Read a compact RDVQ bitstream container for inspection/validation."""
    data = Path(path).read_bytes()
    if len(data) < len(MAGIC) + HEADER_STRUCT.size:
        raise ValueError("file is too small to be an RDVQ bitstream")
    if data[: len(LEAN_MAGIC)] == LEAN_MAGIC:
        offset = len(LEAN_MAGIC)
        if offset + LEAN_STRUCT.size > len(data):
            raise ValueError("truncated RDVQ lean header")
        transfer_slices, original_h, original_w, top_len = LEAN_STRUCT.unpack_from(data, offset)
        offset += LEAN_STRUCT.size
        end_top = offset + top_len
        if end_top > len(data):
            raise ValueError("truncated RDVQ lean tensor top stream")
        record = EncodedBinRecord(
            patch_index=0,
            backend="tensor",
            tensor_top=data[offset:end_top],
            tensor_residual=data[end_top:],
        )
        return {
            "format": "RDVQBN3",
            "flags": 1 if record.tensor_residual else 0,
            "split_image": False,
            "transfer_slices": transfer_slices,
            "original_shape": (original_h, original_w),
            "records": [record],
            "payload_bytes": record.payload_bytes,
            "container_bytes": len(data),
        }
    if data[: len(MAGIC)] != MAGIC:
        raise ValueError("invalid RDVQ bitstream magic")

    offset = len(MAGIC)
    flags, transfer_slices, record_count, original_h, original_w = HEADER_STRUCT.unpack_from(data, offset)
    offset += HEADER_STRUCT.size

    descriptors = []
    for _ in range(record_count):
        if offset + RECORD_STRUCT.size > len(data):
            raise ValueError("truncated RDVQ record table")
        backend_id, patch_index, comp_len, top_len, residual_len = RECORD_STRUCT.unpack_from(data, offset)
        offset += RECORD_STRUCT.size
        descriptors.append((backend_id, patch_index, comp_len, top_len, residual_len))

    records = []
    for backend_id, patch_index, comp_len, top_len, residual_len in descriptors:
        end_comp = offset + comp_len
        end_top = end_comp + top_len
        end_residual = end_top + residual_len
        if end_residual > len(data):
            raise ValueError("truncated RDVQ payload data")
        records.append(EncodedBinRecord(
            patch_index=patch_index,
            backend=BACKEND_NAMES.get(backend_id, "mixed"),
            compressai=data[offset:end_comp],
            tensor_top=data[end_comp:end_top],
            tensor_residual=data[end_top:end_residual],
        ))
        offset = end_residual

    if offset != len(data):
        raise ValueError("RDVQ bitstream has trailing bytes")

    return {
        "format": "RDVQBN2",
        "flags": flags,
        "split_image": bool(flags & 1),
        "transfer_slices": transfer_slices,
        "original_shape": (original_h, original_w),
        "records": records,
        "payload_bytes": sum(record.payload_bytes for record in records),
        "container_bytes": len(data),
    }


def bitstream_size_report(path):
    """Return compact container and payload size information."""
    info = read_rdvq_bin(path)
    return {
        "path": str(path),
        "format": info.get("format", "unknown"),
        "records": len(info["records"]),
        "payload_bytes": info["payload_bytes"],
        "container_bytes": info["container_bytes"],
        "overhead_bytes": info["container_bytes"] - info["payload_bytes"],
    }
