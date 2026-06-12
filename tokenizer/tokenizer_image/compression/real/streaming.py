"""Entropy stream helpers for real AR generation."""

from __future__ import annotations

import torch

from tokenizer.tokenizer_image.entropy import (
    decode_entropy_stream_to_token_slices,
    encode_entropy_packets,
    merged_entropy_stream_bits,
    verify_entropy_packets,
)

from .profiling import _env_flag


def _should_transmit(input_pos, transfer_slices):
    return int(input_pos.reshape(-1)[0].item()) <= transfer_slices - 1


def _current_valid_mask(valid_token_mask, mask_all, input_pos):
    if valid_token_mask is None:
        return None
    pos = mask_all == input_pos
    return valid_token_mask[:, pos].bool()


def _fill_invalid_tokens(tokens, valid_mask, padding_token):
    if valid_mask is None:
        return tokens
    fill = torch.full_like(tokens, int(padding_token))
    return torch.where(valid_mask.to(tokens.device), tokens, fill)


def _append_streams(streams_all, bitstreams):
    if bitstreams is None:
        return
    if isinstance(bitstreams, list):
        streams_all.extend(bitstreams)
    else:
        streams_all.append(bitstreams)


def _encode_streams(streams, profile=None, verify=None):
    if verify is None:
        verify = _env_flag("RDVQ_STREAM_MERGE_VERIFY", True)
    encoded_stream = encode_entropy_packets(streams, profile=profile)
    if verify:
        verify_entropy_packets(encoded_stream, profile=profile)
    return encoded_stream


def _stream_bits(streams, profile=None):
    return merged_entropy_stream_bits(streams, profile=profile)


def _apply_bitstream_decoded_slices(all_tokens, encoded_stream, mask_all, profile=None):
    if not _env_flag("RDVQ_USE_BITSTREAM_DECODED_TOKENS", True):
        return all_tokens, 0, 0
    decoded_slices = decode_entropy_stream_to_token_slices(encoded_stream, profile=profile)
    decoded_token_count = 0
    for slice_idx, decoded_slice in decoded_slices:
        if slice_idx is None:
            continue
        positions = mask_all.to(all_tokens.device) == int(slice_idx)
        decoded_slice = decoded_slice.to(device=all_tokens.device, dtype=all_tokens.dtype)
        if decoded_slice.shape != all_tokens[:, positions].shape:
            raise AssertionError(
                f"decoded slice shape mismatch at slice {slice_idx}: "
                f"decoded={tuple(decoded_slice.shape)}, target={tuple(all_tokens[:, positions].shape)}"
            )
        all_tokens[:, positions] = decoded_slice
        decoded_token_count += int(decoded_slice.numel())
    return all_tokens, len(decoded_slices), decoded_token_count


def verify_restore_from_encoder_packets(all_tokens, encoded_stream, mask_all, profile=None):
    """Restore tokens from encoder-built packets for legacy verification paths.

    This is not an independent causal decoder: it consumes an encoded stream
    together with encoder-side packet metadata. The explicit name keeps legacy
    fast-path verification separate from the default causal real codec.
    """

    return _apply_bitstream_decoded_slices(all_tokens, encoded_stream, mask_all, profile=profile)
