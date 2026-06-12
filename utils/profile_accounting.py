"""Profiling and timing accounting helpers for RDVQ inference."""

from __future__ import annotations

import time

import torch


def profile_add(profile, key, value):
    """Accumulate a numeric counter when profiling is enabled."""
    if profile is not None:
        profile[key] = profile.get(key, 0.0) + float(value)


def profile_sync(device):
    """Synchronize CUDA timing boundaries; CPU devices are no-ops."""
    if device is None:
        return
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def profile_tic(profile, device=None):
    """Start a profile interval if profiling is enabled."""
    if profile is None:
        return None
    profile_sync(device)
    return time.perf_counter()


def profile_toc(profile, key, start, device=None):
    """Stop a profile interval and accumulate it under ``key``."""
    if profile is None or start is None:
        return
    profile_sync(device)
    profile_add(profile, key, time.perf_counter() - start)


def sum_profile(profile, keys):
    if not profile:
        return 0.0
    return sum(float(profile.get(key, 0.0)) for key in keys)


def build_real_timing_summary(profile, image_count, legacy_enc_time, legacy_dec_time):
    """Build explicit codec timing fields from fine-grained profile counters.

    The legacy fields are kept for backward comparison. When decoder replay is
    enabled, decoder-side entropy-model timing is reported separately from VQ
    decode and rANS token decode.
    """
    summary = {
        "average_enc_time_legacy": float(legacy_enc_time),
        "average_dec_time_legacy": float(legacy_dec_time),
        "timing_split_mode": "profile_accounting_v1",
    }
    if not profile or image_count <= 0:
        return summary

    entropy_encode_keys = (
        "tensor_rans.top_encode",
        "tensor_rans.residual_encode",
        "stream_merge.rans_encode_flush",
        "entropy.rans_encode",
        "entropy.rans_flush",
    )
    entropy_decode_verify_keys = (
        "tensor_rans.top_decode",
        "tensor_rans.residual_decode",
        "stream_merge.rans_decode",
        "entropy.rans_decode",
    )
    entropy_decode_keys = (
        "tensor_rans.top_decode_tokens",
        "tensor_rans.residual_decode_tokens",
        "stream_merge.rans_decode_tokens",
    )

    vq_enc = float(profile.get("real.vq_encode", 0.0)) / image_count
    generate_total = float(profile.get("real.generate_total", 0.0)) / image_count
    vq_dec = float(profile.get("real.vq_decode", 0.0)) / image_count
    causal_encoder = float(profile.get("causal.encoder_total", 0.0)) / image_count
    causal_decoder = float(profile.get("causal.decoder_total", 0.0)) / image_count
    if causal_encoder > 0 or causal_decoder > 0:
        summary.update({
            "timing_split_mode": "profile_accounting_v3_independent_causal",
            "average_causal_entropy_encoder_time": causal_encoder,
            "average_causal_entropy_decoder_time": causal_decoder,
            "average_vq_enc_time": vq_enc,
            "average_vq_dec_time": vq_dec,
            "average_real_enc_time": vq_enc + causal_encoder,
            "average_real_dec_time": causal_decoder + vq_dec,
            "average_real_dec_time_verify": None,
            "average_total_codec_time": vq_enc + causal_encoder + causal_decoder + vq_dec,
        })
        return summary
    entropy_encode = sum_profile(profile, entropy_encode_keys) / image_count
    entropy_decode_verify = sum_profile(profile, entropy_decode_verify_keys) / image_count
    entropy_decode = sum_profile(profile, entropy_decode_keys) / image_count
    entropy_model_dec_total = float(profile.get("real.entropy_model_decode", 0.0))
    entropy_model_dec = entropy_model_dec_total / image_count if entropy_model_dec_total > 0 else None
    entropy_model_enc = max(generate_total - entropy_encode - entropy_decode - entropy_decode_verify - (entropy_model_dec or 0.0), 0.0)
    replay_no_entropy_decode = None if entropy_model_dec is None else entropy_model_dec + vq_dec
    replay_with_entropy_decode = None if entropy_model_dec is None else entropy_model_dec + entropy_decode + vq_dec
    if entropy_model_dec is not None:
        summary["timing_split_mode"] = "profile_accounting_v2_decoder_replay"

    summary.update({
        "average_vq_enc_time": vq_enc,
        "average_entropy_model_enc_time": entropy_model_enc,
        "average_entropy_encode_time": entropy_encode,
        "average_entropy_decode_verify_time": entropy_decode_verify,
        "average_entropy_model_dec_time": entropy_model_dec,
        "average_entropy_decode_time": entropy_decode,
        "average_vq_dec_time": vq_dec,
        "average_real_enc_time": vq_enc + entropy_model_enc + entropy_encode,
        "average_real_dec_time_verify": entropy_decode_verify + vq_dec,
        "average_real_dec_time_replay_no_entropy_decode": replay_no_entropy_decode,
        "average_real_dec_time_replay_with_entropy_decode": replay_with_entropy_decode,
        "average_real_dec_time": None,
        "average_total_codec_time": vq_enc + entropy_model_enc + entropy_encode + entropy_decode + entropy_decode_verify + (entropy_model_dec or 0.0) + vq_dec,
    })
    return summary


# Backward-friendly alias used by older notes/scripts.
build_timing_summary = build_real_timing_summary
