"""Legacy RDVQ real compression pipeline boundary.

This module keeps the historical fast/causal mode dispatcher for benchmark
and compatibility use. The default public real-bitstream path now uses
``simple_real_codec.SimpleRealCodec`` instead of this env-driven dispatcher.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import torch

from autoregressive.models.generate_single_stage_real import generate
from tokenizer.tokenizer_image.entropy.pipelines import causal_tensor_compress_and_decompress


_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def _env_choice(name: str, default: str) -> str:
    value = os.environ.get(name, default)
    return str(value).strip().lower()


def _env_flag(name: str, default: bool) -> bool:
    default_value = "1" if default else "0"
    value = _env_choice(name, default_value)
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    return default


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return int(default)
    return int(value)


@dataclass(frozen=True)
class RealCompressionConfig:
    """Resolved user-facing and low-level real compression options."""

    requested_mode: str
    pipeline_mode: str
    backend: str
    entropy_coder: str
    topk: int
    stream_merge: bool
    use_bitstream_decoded_tokens: bool
    decoder_kind: str


@dataclass
class RealCompressionPipeline:
    """Objects needed to run the selected real compression pipeline."""

    ar_model: torch.nn.Module
    codebook: torch.Tensor
    padding_token: int
    config: RealCompressionConfig


@dataclass
class RealCompressionResult:
    """Named result returned by the real compression pipeline."""

    decoded_indices: torch.Tensor
    estimated_bits: float
    payload_bits: int
    real_bits_with_header: int
    streams: list[Any] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, float] = field(default_factory=dict)
    mode: str = "fast_bulk_encoder"
    backend: str = "compressai"
    decoder_kind: str = "verify_restore"


def _resolve_pipeline_mode(requested_mode: str) -> str:
    mode = str(requested_mode or "fast").strip().lower()
    if mode in {"fast", "fast_bulk", "fast_bulk_encoder"}:
        return "fast_bulk_encoder"
    if mode in {"causal", "causal_codec"}:
        return "causal_codec"
    if mode == "validate":
        raise NotImplementedError("REAL_CODEC_MODE=validate is planned but not implemented in this phase")
    raise ValueError(f"unsupported REAL_CODEC_MODE: {requested_mode!r}")


def build_real_compression_pipeline(args, ar_model, codebook, padding_token) -> RealCompressionPipeline:
    """Build the real compression pipeline from CLI args and environment options.

    The fast mode keeps the current encoder-side bulk tensor rANS path while
    exposing it as a compression pipeline rather than a test runtime.
    """

    # Resolve user-facing env knobs once and pass an immutable config down the
    # call chain. This keeps process_one_image independent of the concrete
    # entropy implementation selected below.
    requested_mode = _env_choice("REAL_CODEC_MODE", "fast")
    pipeline_mode = _resolve_pipeline_mode(requested_mode)
    use_decoded = _env_flag("RDVQ_USE_BITSTREAM_DECODED_TOKENS", True)
    decoder_kind = "independent_causal" if pipeline_mode == "causal_codec" else ("verify_restore" if use_decoded else "encoder_simulation")
    config = RealCompressionConfig(
        requested_mode=requested_mode,
        pipeline_mode=pipeline_mode,
        backend=_env_choice("RDVQ_RANS_BACKEND", "compressai"),
        entropy_coder=_env_choice("RDVQ_ENTROPY_CODER", "full"),
        topk=_env_int("RDVQ_TOPK", 0),
        stream_merge=_env_flag("RDVQ_STREAM_MERGE", False),
        use_bitstream_decoded_tokens=use_decoded,
        decoder_kind=decoder_kind,
    )
    return RealCompressionPipeline(
        ar_model=ar_model,
        codebook=codebook,
        padding_token=int(padding_token),
        config=config,
    )


@torch.no_grad()
def compress_and_decompress_tokens(
    pipeline: RealCompressionPipeline,
    *,
    shape_list,
    gt_indices: torch.Tensor,
    transfer_slices: int,
    batch_size: int,
    mask_padded=None,
    temperature: float = 1.0,
    top_k: int = -1,
    top_p: float = 1.0,
    profile=None,
) -> RealCompressionResult:
    """Run real entropy coding and return decoded indices plus bitstream stats."""

    if pipeline.config.pipeline_mode == "causal_codec":
        # Strict real-codec path: the encoder writes only transmitted prefix
        # slices into tensor-rANS streams, and the decoder rebuilds AR logits
        # from previously recovered tokens before consuming each slice.
        if pipeline.config.backend != "tensor":
            raise ValueError("REAL_CODEC_MODE=causal currently requires RDVQ_RANS_BACKEND=tensor")
        if pipeline.config.entropy_coder not in {"topk", "topk_escape"}:
            raise ValueError("REAL_CODEC_MODE=causal currently requires RDVQ_ENTROPY_CODER=topk_escape")
        topk = int(pipeline.config.topk or 1024)
        causal_result = causal_tensor_compress_and_decompress(
            pipeline.ar_model,
            shape_list=shape_list,
            gt_indices=gt_indices,
            transfer_slices=transfer_slices,
            batch_size=batch_size,
            mask_padded=mask_padded,
            codebook=pipeline.codebook,
            padding_token=pipeline.padding_token,
            topk=topk,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            sample_logits=True,
            profile=profile,
        )
        stats = dict(causal_result.stats or {})
        stats["real_compression_mode"] = pipeline.config.requested_mode
        stats["real_codec_mode"] = pipeline.config.requested_mode
        stats["entropy_pipeline_mode"] = pipeline.config.pipeline_mode
        stats["entropy_backend"] = pipeline.config.backend
        stats["decoder_kind"] = pipeline.config.decoder_kind
        stats["use_bitstream_decoded_tokens"] = 1
        return RealCompressionResult(
            decoded_indices=causal_result.decoded_indices,
            estimated_bits=float(causal_result.estimated_bits),
            payload_bits=int(causal_result.payload_bits),
            real_bits_with_header=int(causal_result.payload_bits),
            streams=[],
            stats=stats,
            timing={},
            mode=pipeline.config.pipeline_mode,
            backend=pipeline.config.backend,
            decoder_kind=pipeline.config.decoder_kind,
        )

    if pipeline.config.pipeline_mode != "fast_bulk_encoder":
        raise NotImplementedError(f"unsupported pipeline mode: {pipeline.config.pipeline_mode}")

    # Fast path: reuse the existing AR generation runtime. It can either step
    # through slices sequentially or use a teacher-forced full-forward shortcut
    # when the run is full-transfer and validation allows it.
    decoded_indices, estimated_bits, streams, stats = generate(
        pipeline.ar_model,
        shape_list=shape_list,
        gt_indices=gt_indices,
        transfer_slices=transfer_slices,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        sample_logits=True,
        Bs=batch_size,
        mask_padded=mask_padded,
        codebook=pipeline.codebook,
        padding_token=pipeline.padding_token,
        return_stats=True,
        profile=profile,
    )

    stats = dict(stats or {})
    stats.setdefault("estimated_bits", float(estimated_bits))
    stats.setdefault("payload_bits", 0)
    stats["real_compression_mode"] = pipeline.config.requested_mode
    stats["real_codec_mode"] = pipeline.config.requested_mode
    stats["entropy_pipeline_mode"] = pipeline.config.pipeline_mode
    stats["entropy_backend"] = pipeline.config.backend
    stats["decoder_kind"] = pipeline.config.decoder_kind
    stats["use_bitstream_decoded_tokens"] = int(pipeline.config.use_bitstream_decoded_tokens)

    payload_bits = int(stats.get("payload_bits", 0))
    return RealCompressionResult(
        decoded_indices=decoded_indices,
        estimated_bits=float(stats.get("estimated_bits", estimated_bits)),
        payload_bits=payload_bits,
        real_bits_with_header=payload_bits,
        streams=list(streams or []),
        stats=stats,
        timing={},
        mode=pipeline.config.pipeline_mode,
        backend=pipeline.config.backend,
        decoder_kind=pipeline.config.decoder_kind,
    )
