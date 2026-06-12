"""Rate and compact-bitstream accounting for real codec inference."""

from __future__ import annotations

from utils.bitstream_container import attach_bin_stream


def build_real_patch_stats(codec_result, image_tensor, *, header_bits_mode: str = "none"):
    """Build bpp values and serializable stats for one processed image/patch tensor."""

    coding_pixels = image_tensor.shape[0] * image_tensor.shape[2] * image_tensor.shape[3]
    coding_stats = dict(codec_result.stats)
    coding_stats.setdefault("estimated_bits", float(codec_result.estimated_bits))
    coding_stats.setdefault("payload_bits", int(codec_result.payload_bits))
    bpp = coding_stats["estimated_bits"] / coding_pixels
    bpp_real = coding_stats["payload_bits"] / coding_pixels
    coding_stats["coding_pixels"] = int(coding_pixels)
    coding_stats["header_bits"] = 32 if header_bits_mode == "patch" else 0
    coding_stats["real_bits_with_header"] = coding_stats["payload_bits"] + coding_stats["header_bits"]
    coding_stats = attach_bin_stream(coding_stats)
    return bpp, bpp_real, coding_stats
