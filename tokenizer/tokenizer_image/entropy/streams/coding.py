"""Image-level entropy coding: encode, verify, decode, and stream merge.

Public entry points used by real bitstream inference:
  - ``encode_entropy_packets`` / ``verify_entropy_packets``
  - ``merged_entropy_stream_bits`` (compatibility wrapper)
  - ``decode_entropy_stream_to_token_slices``
"""

import os

import torch
from compressai.ans import RansDecoder

from ..codecs import CompressAIEntropyCodec, TensorRansEntropyCodec
from ..streams.packet import _split_entropy_streams
from ..symbols.probability import pmf_to_quantized_cdf
from ..utils.profiling import _env_flag, _profile_add, _profile_tic, _profile_toc
from ..symbols.symbol_mapping import (
    map_list_topk_decoded_to_codebook,
    map_tensor_topk_decoded_to_codebook,
)


# ── CompressAI list-path encode / verify ────────────────────────────────


def encode_compressai_entropy_packets(packets, profile=None) -> tuple[bytes, int, int]:
    """Encode CompressAI/list packets into one merged stream."""

    return CompressAIEntropyCodec().encode_packets(packets, profile=profile)


def verify_compressai_entropy_packets(stream, packets, profile=None) -> None:
    """Decode a merged CompressAI stream against encoder-side packets."""

    CompressAIEntropyCodec().verify_packets(stream, packets, profile=profile)


# ── Tensor rANS encode / verify ─────────────────────────────────────────


def encode_tensor_entropy_packets(packets, profile=None) -> tuple[bytes, bytes, int, int]:
    """Encode tensor-native top/residual packets into split merged streams."""

    return TensorRansEntropyCodec().encode_packets(packets, profile=profile)


def verify_tensor_entropy_packets(top_stream, residual_stream, packets, profile=None) -> None:
    """Decode tensor streams against encoder-side packets for correctness."""

    TensorRansEntropyCodec().verify_packets(top_stream, residual_stream, packets, profile=profile)


# ── High-level encode / verify ──────────────────────────────────────────


def encode_entropy_packets(streams, profile=None):
    """Encode all deferred entropy packets and return stream metadata."""

    from ..streams.packet import EncodedEntropyStream

    raw_streams, list_packets, tensor_packets = _split_entropy_streams(streams)
    raw_bits = sum(len(stream) * 8 for stream in raw_streams)

    compressai_stream, compressai_bits, compressai_symbols = encode_compressai_entropy_packets(list_packets, profile=profile)
    tensor_top_stream, tensor_residual_stream, tensor_bits, tensor_symbols = encode_tensor_entropy_packets(tensor_packets, profile=profile)

    payload_bits = raw_bits + compressai_bits + tensor_bits
    packet_count = len(list_packets) + len(tensor_packets)
    symbol_count = compressai_symbols + tensor_symbols
    if list_packets and tensor_packets:
        backend = "mixed"
    elif tensor_packets:
        backend = "tensor"
    elif list_packets:
        backend = "compressai"
    elif raw_streams:
        backend = "raw"
    else:
        backend = "empty"

    _profile_add(profile, "entropy.encoded_packet_groups", int(packet_count > 0))
    return EncodedEntropyStream(
        payload_bits=payload_bits,
        backend=backend,
        raw_streams=tuple(raw_streams),
        compressai_stream=compressai_stream or None,
        tensor_top_stream=tensor_top_stream or None,
        tensor_residual_stream=tensor_residual_stream or None,
        list_packets=tuple(list_packets),
        tensor_packets=tuple(tensor_packets),
        packet_count=packet_count,
        symbol_count=symbol_count,
    )


def verify_entropy_packets(encoded_stream, profile=None) -> None:
    """Verify merged entropy streams using encoder-side packet metadata."""

    if encoded_stream.compressai_stream is not None:
        verify_compressai_entropy_packets(encoded_stream.compressai_stream, encoded_stream.list_packets, profile=profile)
    if encoded_stream.tensor_packets:
        verify_tensor_entropy_packets(
            encoded_stream.tensor_top_stream or b"",
            encoded_stream.tensor_residual_stream or b"",
            encoded_stream.tensor_packets,
            profile=profile,
        )


# ── Bitstream → token slice decoding ────────────────────────────────────


def _restore_packet_values(packet, decoded_values):
    rec = packet.decoded_template.clone()
    if packet.coding_mask is None or packet.code_selected_mask is None:
        return rec
    selected = rec[packet.coding_mask].reshape(-1)
    code_mask = packet.code_selected_mask.to(dtype=torch.bool)
    selected[code_mask] = decoded_values.to(dtype=selected.dtype)
    rec[packet.coding_mask] = selected
    if packet.decoded_template is not None and not torch.equal(rec, packet.decoded_template):
        raise AssertionError("Bitstream-decoded codebook indices do not match encoder-side reconstruction")
    return rec


def _map_topk_decoded_to_codebook(packet, top_decoded, residual_decoded):
    return map_tensor_topk_decoded_to_codebook(packet, top_decoded, residual_decoded)

def _map_list_decoded_to_codebook(packet, decoded):
    return map_list_topk_decoded_to_codebook(packet, decoded)

def decode_entropy_stream_to_token_slices(encoded_stream, profile=None):
    """Decode rANS bytes back to codebook-index slices for reconstruction.

    This function consumes the real encoded byte streams.  It still uses packet
    CDF/mapping metadata produced during the encoder pass; a fully independent
    decoder should rebuild the same metadata from model logits and bitstream
    headers.
    """
    from compressai.ans import RansDecoder as _RansDecoder

    decoded_slices = []

    if encoded_stream.compressai_stream is not None and encoded_stream.list_packets:
        t = _profile_tic(profile)
        decoder = _RansDecoder()
        decoder.set_stream(encoded_stream.compressai_stream)
        for packet in encoded_stream.list_packets:
            decoded = decoder.decode_stream(packet.indexes, packet.cdfs, packet.cdf_lengths, packet.offsets)
            values = _map_list_decoded_to_codebook(packet, decoded)
            decoded_slices.append((packet.slice_idx, _restore_packet_values(packet, values)))
        _profile_toc(profile, "stream_merge.rans_decode_tokens", t)

    if encoded_stream.tensor_packets:
        top_decoded, residual_decoded = TensorRansEntropyCodec().decode_payloads(
            encoded_stream.tensor_top_stream or b"",
            encoded_stream.tensor_residual_stream or b"",
            encoded_stream.tensor_packets,
            profile=profile,
            token_decode=True,
        )

        top_cursor = 0
        residual_cursor = 0
        for packet in encoded_stream.tensor_packets:
            payload = packet.tensor_payload
            n_top = int(payload.top_symbols.numel())
            n_res = int(payload.residual_symbols.numel())
            packet_top = top_decoded[top_cursor : top_cursor + n_top]
            packet_residual = residual_decoded[residual_cursor : residual_cursor + n_res]
            values = _map_topk_decoded_to_codebook(packet, packet_top, packet_residual)
            decoded_slices.append((packet.slice_idx, _restore_packet_values(packet, values)))
            top_cursor += n_top
            residual_cursor += n_res

    _profile_add(profile, "entropy.bitstream_decoded_slices", len(decoded_slices))
    _profile_add(profile, "entropy.bitstream_decoded_tokens", sum(int(tensor.numel()) for _, tensor in decoded_slices))
    return decoded_slices


# ── Compatibility wrapper + legacy ──────────────────────────────────────


def merged_entropy_stream_bits(streams, profile=None, verify=None):
    """Compatibility wrapper: encode packets, optionally verify, return bits."""

    if verify is None:
        verify = _env_flag("RDVQ_STREAM_MERGE_VERIFY", True)
    encoded_stream = encode_entropy_packets(streams, profile=profile)
    if verify:
        verify_entropy_packets(encoded_stream, profile=profile)
    return encoded_stream.payload_bits


