"""Entropy coding data structures — deferred rANS payload and stream metadata.

These are shared between the AR predictor (which builds per-slice packets) and
the image-level stream merge / verify layer (which encodes, decodes, and checks
the merged byte streams).
"""

import os
from dataclasses import dataclass

import torch


# ── Tensor-native top-k/escape payload ──────────────────────────────────


@dataclass
class TensorEntropyPayload:
    """Tensor-native top-k/escape payload for image-level stream merging."""

    top_symbols: torch.Tensor
    top_cdfs: torch.Tensor
    residual_symbols: torch.Tensor
    residual_cdfs: torch.Tensor | None
    topk: int
    precision: int = 16
    top_indices: torch.Tensor | None = None
    residual_non_top_indices: torch.Tensor | None = None


# ── Encoded bitstream metadata ──────────────────────────────────────────


@dataclass
class EncodedEntropyStream:
    """Encoded image-level entropy payload plus verification metadata.

    Phase 2 keeps packet metadata so the existing correctness verifier can
    decode the merged streams without changing the external bit accounting API.
    A future decoder-side pass should rebuild CDFs from logits instead of using
    these encoder-side packets.
    """

    payload_bits: int
    backend: str
    raw_streams: tuple[bytes, ...] = ()
    compressai_stream: bytes | None = None
    tensor_top_stream: bytes | None = None
    tensor_residual_stream: bytes | None = None
    list_packets: tuple = ()
    tensor_packets: tuple = ()
    packet_count: int = 0
    symbol_count: int = 0


# ── Deferred rANS payload ───────────────────────────────────────────────


class EntropyPacket:
    """Deferred rANS payload for image-level stream merging."""

    __slots__ = (
        "symbols",
        "indexes",
        "cdfs",
        "cdf_lengths",
        "offsets",
        "tensor_payload",
        "coding_backend",
        "decoded_template",
        "coding_mask",
        "code_selected_mask",
        "slice_idx",
        "restore_topk",
    )

    def __init__(
        self,
        symbols=None,
        indexes=None,
        cdfs=None,
        cdf_lengths=None,
        offsets=None,
        *,
        tensor_payload=None,
        coding_backend="compressai",
        decoded_template=None,
        coding_mask=None,
        code_selected_mask=None,
        slice_idx=None,
        restore_topk=None,
    ):
        self.symbols = symbols
        self.indexes = indexes
        self.cdfs = cdfs
        self.cdf_lengths = cdf_lengths
        self.offsets = offsets
        self.tensor_payload = tensor_payload
        self.coding_backend = coding_backend
        self.decoded_template = decoded_template
        self.coding_mask = coding_mask
        self.code_selected_mask = code_selected_mask
        self.slice_idx = slice_idx
        self.restore_topk = restore_topk

    def __len__(self):
        return 0


# ── Packet inspection helpers ───────────────────────────────────────────


def is_entropy_packet(value):
    return isinstance(value, EntropyPacket)


def _has_list_payload(packet):
    return packet.symbols is not None and len(packet.symbols) > 0


def _has_tensor_payload(packet):
    payload = packet.tensor_payload
    return payload is not None and (payload.top_symbols.numel() > 0 or payload.residual_symbols.numel() > 0)


def _normalize_streams(streams):
    if streams is None:
        return []
    if isinstance(streams, (bytes, bytearray, EntropyPacket)):
        return [streams]
    return list(streams)


def _split_entropy_streams(streams):
    raw_streams = []
    list_packets = []
    tensor_packets = []
    for stream in _normalize_streams(streams):
        if stream is None:
            continue
        if is_entropy_packet(stream):
            if stream.coding_backend == "tensor" and _has_tensor_payload(stream):
                tensor_packets.append(stream)
            elif _has_list_payload(stream):
                list_packets.append(stream)
            continue
        raw_streams.append(bytes(stream))
    return raw_streams, list_packets, tensor_packets


def _tensor_payload_tensors(packets):
    payloads = [packet.tensor_payload for packet in packets if _has_tensor_payload(packet)]
    if not payloads:
        empty = torch.empty(0, dtype=torch.int32)
        return 16, empty, None, empty, None

    precision = payloads[0].precision
    if any(payload.precision != precision for payload in payloads):
        raise ValueError("All tensor rANS payloads must use the same precision")

    top_payloads = [payload for payload in payloads if payload.top_symbols.numel() > 0]
    if top_payloads:
        top_symbols = torch.cat([payload.top_symbols for payload in top_payloads], dim=0)
        top_cdfs = torch.cat([payload.top_cdfs for payload in top_payloads], dim=0)
    else:
        top_symbols = torch.empty(0, dtype=torch.int32)
        top_cdfs = None

    residual_payloads = [payload for payload in payloads if payload.residual_symbols.numel() > 0]
    if residual_payloads:
        residual_symbols = torch.cat([payload.residual_symbols for payload in residual_payloads], dim=0)
        residual_cdfs = torch.cat([payload.residual_cdfs for payload in residual_payloads], dim=0)
    else:
        residual_symbols = torch.empty(0, dtype=torch.int32)
        residual_cdfs = None

    return precision, top_symbols, top_cdfs, residual_symbols, residual_cdfs


def _stream_list_bits(streams):
    if isinstance(streams, list):
        if any(is_entropy_packet(stream) for stream in streams):
            return None
        return sum(len(stream) * 8 for stream in streams)
    if is_entropy_packet(streams):
        return None
    return len(streams) * 8
