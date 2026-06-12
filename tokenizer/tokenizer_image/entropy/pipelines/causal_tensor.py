"""Causal tensor-rANS compression/decompression pipeline.

This module is now the model-level orchestration layer. It keeps the causal AR
encoder/decoder loops readable and delegates tensor-rANS details to
``TopkTensorRansCodec``. The decoder rebuilds logits from already recovered
history, consumes real entropy bytes for transmitted prefix slices, and samples
untransmitted suffix slices locally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import time

import torch

from tokenizer.tokenizer_image.compression.real.sampling import sample
from tokenizer.tokenizer_image.compression.real.streaming import _fill_invalid_tokens
from tokenizer.tokenizer_image.entropy.codecs.topk_tensor_rans import (
    EncodedTopkTensorStreams,
    TopkTensorRansCodec,
    TopkTensorRansConfig,
)
from tokenizer.tokenizer_image.entropy.streams.packet import EncodedEntropyStream
from tokenizer.tokenizer_image.entropy.pipelines.causal_ar_loop import (
    CausalARState,
    max_new_tokens,
    num_ar_slices,
    valid_token_mask_from_padding,
)
from tokenizer.tokenizer_image.entropy.utils.profiling import _profile_tic, _profile_toc




def _sync_device(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _coarse_tic(device):
    _sync_device(device)
    return time.perf_counter()


def _coarse_toc(start, device):
    _sync_device(device)
    return time.perf_counter() - start

# Backward-friendly name for older notes/imports. New code should use
# EncodedTopkTensorStreams from topk_tensor_rans.py.
CausalChunkedStreams = EncodedTopkTensorStreams


@dataclass
class CausalEncoderOutput:
    """Encoder-side payload and accounting for transmitted prefix slices."""

    streams: EncodedTopkTensorStreams
    estimated_bits: float
    valid_token_count: int
    padded_token_count: int
    skipped_padded_token_count: int
    transmitted_token_count: int
    num_slices: int
    transfer_count: int


@dataclass
class CausalCompressionOutput:
    """Encoder payload plus independent decoder reconstruction result."""

    decoded_indices: torch.Tensor
    estimated_bits: float
    payload_bits: int
    encoded_stream: EncodedEntropyStream
    stats: dict[str, Any]


def _entropy_for_targets(model, logits, targets_current, valid_current, padding_token, profile=None):
    """Estimate likelihood bits for transmitted target tokens."""

    t = _profile_tic(profile, logits)
    if valid_current is not None:
        fill = torch.full_like(targets_current, int(padding_token))
        targets_for_entropy = torch.where(valid_current.to(targets_current.device), targets_current, fill)
    else:
        targets_for_entropy = targets_current
    entropy = model.cross_entropy_log2(logits.reshape(-1, logits.size(-1)), targets_for_entropy.reshape(-1), reduction="none")
    entropy = entropy.view(targets_current.shape[0], -1)
    if valid_current is not None:
        entropy = entropy * valid_current.to(entropy.device, dtype=entropy.dtype)
    _profile_toc(profile, "causal.encoder_entropy_loss", t, logits)
    return float(entropy.sum().item())


def _sample_suffix_slice(logits, valid_current, padding_token, profile=None, **sampling_kwargs):
    """Generate an untransmitted slice on the decoder side."""

    t = _profile_tic(profile, logits)
    sampled, _ = sample(logits, **sampling_kwargs)
    _profile_toc(profile, "causal.decoder_suffix_sample", t, logits)
    return _fill_invalid_tokens(sampled, valid_current, padding_token).long()


@torch.no_grad()
def encode_transmitted_prefix(
    model,
    *,
    shape_list,
    gt_indices,
    transfer_slices: int,
    batch_size: int,
    valid_token_mask,
    codebook,
    padding_token: int,
    topk: int,
    precision: int,
    profile=None,
) -> CausalEncoderOutput:
    """Run encoder-side AR forward for transmitted slices and write rANS streams."""

    total_tokens = max_new_tokens(shape_list)
    total_slices = num_ar_slices(shape_list, int(getattr(model, "num_ar_per_scale", 4)))
    transfer_count = min(max(int(transfer_slices), 0), total_slices)
    rans_codec = TopkTensorRansCodec(TopkTensorRansConfig(topk=int(topk), precision=int(precision), padding_token=int(padding_token)))
    ar_state = CausalARState(
        model=model,
        shape_list=shape_list,
        batch_size=batch_size,
        codebook=codebook,
        padding_token=padding_token,
        valid_token_mask=valid_token_mask,
        profile=profile,
    )

    estimated_bits = 0.0
    ar_state.setup()
    mask_all = ar_state.mask_all
    try:
        for slice_idx in range(transfer_count):
            # Model step: build logits from encoder-side causal history.
            logits = ar_state.logits_for_slice(slice_idx)
            targets_current = ar_state.target_slice(gt_indices, slice_idx)
            valid_current = ar_state.valid_mask_for_slice(slice_idx)

            estimated_bits += _entropy_for_targets(model, logits, targets_current, valid_current, padding_token, profile=profile)

            # Entropy step: append one transmitted slice to the rANS black box.
            history_tokens = rans_codec.encode_slice(logits, targets_current, valid_current, profile=profile)

            # History step: ground-truth transmitted tokens become context for the next AR slice.
            ar_state.write_slice(slice_idx, history_tokens)
    finally:
        ar_state.destroy()

    streams = rans_codec.finish_encoding(profile=profile)
    valid_token_count = int(valid_token_mask.sum().item()) if valid_token_mask is not None else batch_size * total_tokens
    padded_token_count = 0 if valid_token_mask is None else int((~valid_token_mask).sum().item())
    skipped_padded = 0
    if valid_token_mask is not None and transfer_count > 0:
        skipped_padded = int((~valid_token_mask[:, mask_all < transfer_count]).sum().item())
    transmitted_token_count = int(batch_size * int((mask_all < transfer_count).sum().item()))
    if valid_token_mask is not None:
        transmitted_token_count = int(valid_token_mask[:, mask_all < transfer_count].sum().item())

    return CausalEncoderOutput(
        streams=streams,
        estimated_bits=float(estimated_bits),
        valid_token_count=int(valid_token_count),
        padded_token_count=int(padded_token_count),
        skipped_padded_token_count=int(skipped_padded),
        transmitted_token_count=int(transmitted_token_count),
        num_slices=int(total_slices),
        transfer_count=int(transfer_count),
    )


@torch.no_grad()
def decode_all_slices(
    model,
    *,
    shape_list,
    streams: EncodedTopkTensorStreams,
    transfer_slices: int,
    batch_size: int,
    valid_token_mask,
    codebook,
    padding_token: int,
    topk: int,
    precision: int,
    profile=None,
    **sampling_kwargs,
):
    """Run independent decoder-side AR forward and consume rANS streams."""

    total_slices = num_ar_slices(shape_list, int(getattr(model, "num_ar_per_scale", 4)))
    transfer_count = min(max(int(transfer_slices), 0), total_slices)
    rans_codec = TopkTensorRansCodec(TopkTensorRansConfig(topk=int(topk), precision=int(precision), padding_token=int(padding_token)))
    rans_codec.begin_decoding(streams)
    ar_state = CausalARState(
        model=model,
        shape_list=shape_list,
        batch_size=batch_size,
        codebook=codebook,
        padding_token=padding_token,
        valid_token_mask=valid_token_mask,
        profile=profile,
    )

    decoded_token_count = 0
    ar_state.setup()
    try:
        for slice_idx in range(total_slices):
            # Model step: decoder rebuilds logits from its own recovered history.
            logits = ar_state.logits_for_slice(slice_idx)
            valid_current = ar_state.valid_mask_for_slice(slice_idx)

            if slice_idx < transfer_count:
                # Entropy step: decode transmitted symbols from real rANS bytes.
                current = rans_codec.decode_slice(logits, valid_current, profile=profile)
                decoded_token_count += int(current.numel() if valid_current is None else valid_current.sum().item())
            else:
                # Generation step: untransmitted suffix slices are sampled locally.
                current = _sample_suffix_slice(logits, valid_current, padding_token, profile=profile, **sampling_kwargs)

            # History step: decoded/generated tokens become context for the next AR slice.
            ar_state.write_slice(slice_idx, current)
        decoded_indices = ar_state.all_indices()
    finally:
        ar_state.destroy()

    return decoded_indices, decoded_token_count


@torch.no_grad()
def causal_tensor_compress_and_decompress(
    model,
    *,
    shape_list,
    gt_indices,
    transfer_slices: int,
    batch_size: int,
    mask_padded=None,
    codebook=None,
    padding_token: int = 0,
    topk: int = 1024,
    precision: int = 16,
    profile=None,
    **sampling_kwargs,
) -> CausalCompressionOutput:
    """Encode transmitted prefix slices and independently decode all slices."""

    if codebook is None:
        raise ValueError("causal tensor compression requires codebook embeddings")
    device = model.start_token.device
    valid_token_mask = valid_token_mask_from_padding(mask_padded, device)

    t = _profile_tic(profile, device)
    coarse_t = _coarse_tic(device)
    encoded = encode_transmitted_prefix(
        model,
        shape_list=shape_list,
        gt_indices=gt_indices,
        transfer_slices=transfer_slices,
        batch_size=batch_size,
        valid_token_mask=valid_token_mask,
        codebook=codebook,
        padding_token=padding_token,
        topk=topk,
        precision=precision,
        profile=profile,
    )
    causal_encoder_time = _coarse_toc(coarse_t, device)
    _profile_toc(profile, "causal.encoder_total", t, device)

    streams = encoded.streams
    payload_bits = int(streams.payload_bits)
    encoded_stream = EncodedEntropyStream(
        payload_bits=payload_bits,
        backend="tensor",
        tensor_top_stream=streams.top_stream or None,
        tensor_residual_stream=streams.residual_stream or None,
        packet_count=int(encoded.transfer_count),
        symbol_count=int(streams.symbol_count),
    )

    t = _profile_tic(profile, device)
    coarse_t = _coarse_tic(device)
    decoded_indices, decoded_token_count = decode_all_slices(
        model,
        shape_list=shape_list,
        streams=streams,
        transfer_slices=transfer_slices,
        batch_size=batch_size,
        valid_token_mask=valid_token_mask,
        codebook=codebook,
        padding_token=padding_token,
        topk=topk,
        precision=precision,
        profile=profile,
        **sampling_kwargs,
    )
    causal_decoder_time = _coarse_toc(coarse_t, device)
    _profile_toc(profile, "causal.decoder_total", t, device)

    stats = {
        "estimated_bits": float(encoded.estimated_bits),
        "payload_bits": int(payload_bits),
        "payload_bits_raw_rans": int(payload_bits),
        "causal_stream_header_bits": 0,
        "entropy_packet_count": int(encoded.transfer_count),
        "entropy_symbol_count": int(streams.symbol_count),
        "entropy_stream_backend": "tensor",
        "_encoded_entropy_stream": encoded_stream,
        "decoder_token_source": "independent_causal_decoder",
        "bitstream_decoded_slice_count": int(encoded.transfer_count),
        "bitstream_decoded_token_count": int(decoded_token_count),
        "valid_token_count": int(encoded.valid_token_count),
        "padded_token_count": int(encoded.padded_token_count),
        "skipped_padded_token_count": int(encoded.skipped_padded_token_count),
        "transmitted_token_count": int(encoded.transmitted_token_count),
        "causal_total_slices": int(encoded.num_slices),
        "causal_transfer_slices": int(encoded.transfer_count),
        "causal_encoder_time": float(causal_encoder_time),
        "causal_decoder_time": float(causal_decoder_time),
    }
    return CausalCompressionOutput(
        decoded_indices=decoded_indices,
        estimated_bits=float(encoded.estimated_bits),
        payload_bits=int(payload_bits),
        encoded_stream=encoded_stream,
        stats=stats,
    )
