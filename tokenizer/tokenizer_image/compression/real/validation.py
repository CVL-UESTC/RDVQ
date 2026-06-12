"""Result packing and byte-exact validation helpers for real generation."""

from __future__ import annotations

import torch

from tokenizer.tokenizer_image.entropy import encode_entropy_packets


def _pack_result(all_tokens, bits_all, streams_all, stats, return_stats):
    if return_stats:
        return all_tokens, bits_all, streams_all, stats
    return all_tokens, bits_all, streams_all


def _extract_result(result, return_stats):
    if return_stats:
        all_tokens, bits_all, streams_all, stats = result
    else:
        all_tokens, bits_all, streams_all = result
        stats = {}
    return all_tokens, bits_all, streams_all, stats


def _encoded_stream_signature(streams):
    encoded = encode_entropy_packets(streams, profile=None)
    return (
        encoded.backend,
        encoded.payload_bits,
        encoded.compressai_stream or b"",
        encoded.tensor_top_stream or b"",
        encoded.tensor_residual_stream or b"",
        encoded.packet_count,
        encoded.symbol_count,
    )


def _compare_generation_results(full_result, seq_result, return_stats, atol_bits=1e-2):
    """Compare full-forward and sequential AR outputs before trusting fast mode."""
    full_tokens, full_bits, full_streams, full_stats = _extract_result(full_result, return_stats)
    seq_tokens, seq_bits, seq_streams, seq_stats = _extract_result(seq_result, return_stats)
    if not torch.equal(full_tokens, seq_tokens):
        return False, "decoded token buffers differ"
    for key in ("payload_bits", "entropy_packet_count", "entropy_symbol_count"):
        if int(full_stats.get(key, -1)) != int(seq_stats.get(key, -1)):
            return False, f"{key} differs: full={full_stats.get(key)}, seq={seq_stats.get(key)}"
    if _encoded_stream_signature(full_streams) != _encoded_stream_signature(seq_streams):
        return False, "merged entropy stream bytes differ"
    if abs(float(full_bits) - float(seq_bits)) > atol_bits:
        return False, f"estimated bits differ: full={float(full_bits):.8f}, seq={float(seq_bits):.8f}"
    return True, "ok"
