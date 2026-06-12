#!/usr/bin/env python3
"""Summarize RDVQ inference metrics/profile JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

COUNTER_KEYS = {
    "ar.calls",
    "ar.decoder_replay_calls",
    "ar.full_forward_validation_fail",
    "ar.full_forward_validation_pass",
    "ar.full_forward_calls",
    "entropy.calls",
    "entropy.bitstream_decoded_tokens",
    "entropy.bitstream_decoded_slices",
    "entropy.deferred_packets",
    "entropy.deferred_packet_symbols",
    "entropy.deterministic_candidates",
    "entropy.deterministic_checked_steps",
    "entropy.deterministic_exceptions",
    "entropy.deterministic_fallback_steps",
    "entropy.deterministic_symbols",
    "entropy.deterministic_zero_stream_steps",
    "entropy.escape_symbols",
    "entropy.encoded_packet_groups",
    "entropy.fast_cdf_calls",
    "entropy.fast_cdf_fallbacks",
    "entropy.scalar_cdf_calls",
    "entropy.scalar_cdf_fallbacks",
    "entropy.symbols",
    "entropy.topk_escape_calls",
    "entropy.topk_value",
    "stream_merge.packets",
    "stream_merge.payload_bits",
    "stream_merge.symbols",
    "entropy.tensor_packets",
    "tensor_rans.top_symbols",
    "tensor_rans.residual_symbols",
    "tensor_rans.payload_bits",
}

TIME_PREFIXES = ("real.", "ar.", "entropy.", "stream_merge.", "tensor_rans.", "generate.")


def _fmt(value):
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _is_time_key(key: str) -> bool:
    return key.startswith(TIME_PREFIXES) and key not in COUNTER_KEYS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_json", type=Path)
    parser.add_argument("--top", type=int, default=30, help="Number of time entries to print")
    args = parser.parse_args()

    data = json.loads(args.metrics_json.read_text())
    profile = data.get("profile_seconds", {}) or {}
    image_count = int(data.get("image_count") or 1)
    average_dec_total = float(data.get("average_dec_time") or 0.0) * image_count

    cd_bpp = data.get("cd_bpp")
    real_bpp = data.get("cd_bpp_real", data.get("real_bpp_payload"))
    overhead = None
    if cd_bpp not in (None, 0) and real_bpp is not None:
        overhead = (float(real_bpp) / float(cd_bpp) - 1.0) * 100.0

    print(f"Metrics JSON: {args.metrics_json}")
    print(f"image_count: {image_count}")
    for key in [
        "average_time",
        "average_enc_time",
        "average_dec_time",
        "average_enc_time_legacy",
        "average_dec_time_legacy",
        "timing_split_mode",
        "average_vq_enc_time",
        "average_entropy_model_enc_time",
        "average_entropy_encode_time",
        "average_entropy_decode_verify_time",
        "average_entropy_model_dec_time",
        "average_entropy_decode_time",
        "average_vq_dec_time",
        "average_real_enc_time",
        "average_real_dec_time_verify",
        "average_real_dec_time_replay_no_entropy_decode",
        "average_real_dec_time_replay_with_entropy_decode",
        "average_real_dec_time",
        "average_total_codec_time",
        "cd_bpp",
        "cd_bpp_real",
        "cd_bpp_real_with_header",
        "entropy_coder",
        "entropy_topk",
        "rans_backend",
        "allow_scalar_fallback",
        "fast_cdf_fallbacks",
        "scalar_cdf_fallbacks",
        "tensor_packets",
        "deterministic_threshold",
        "stream_merge",
        "stream_merge_verify",
        "encoder_full_forward",
        "encoder_full_forward_validate",
        "decoder_timing_pass",
        "use_bitstream_decoded_tokens",
        "decoder_token_source",
        "bitstream_decoded_slice_count",
        "bitstream_decoded_token_count",
        "valid_token_count",
        "transmitted_token_count",
        "entropy_packet_count",
        "entropy_symbol_count",
        "padded_token_count",
        "skipped_padded_token_count",
    ]:
        if key in data:
            print(f"{key}: {_fmt(data[key])}")
    if overhead is not None:
        print(f"real_payload_overhead_percent: {overhead:.4f}")

    time_items = sorted(
        ((key, float(value)) for key, value in profile.items() if _is_time_key(key)),
        key=lambda item: item[1],
        reverse=True,
    )
    print("\nTime profile:")
    for key, value in time_items[: args.top]:
        per_image = value / image_count
        pct_dec = (value / average_dec_total * 100.0) if average_dec_total > 0 else 0.0
        print(f"{key:42s} total={value:10.6f}s per_img={per_image:9.6f}s pct_dec={pct_dec:7.2f}%")

    counter_items = sorted(
        ((key, value) for key, value in profile.items() if key in COUNTER_KEYS),
        key=lambda item: item[0],
    )
    if counter_items:
        print("\nCounters:")
        for key, value in counter_items:
            print(f"{key:42s} {value}")


if __name__ == "__main__":
    main()
