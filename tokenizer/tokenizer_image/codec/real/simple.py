"""Simplified real codec wrapper for public real-bitstream evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from tokenizer.tokenizer_image.codec.latent_io import EncodedLatents
from tokenizer.tokenizer_image.codec.real.causal_pipeline import causal_tensor_compress_and_decompress


@dataclass(frozen=True)
class SimpleRealCodecConfig:
    """Fixed public real-codec configuration.

    Only transfer_slices and topk are intended as public compression controls.
    The codec itself is fixed to causal top-k tensor rANS.
    """

    transfer_slices: int = 28
    topk: int = 1024
    precision: int = 16
    temperature: float = 1.0
    sample_top_k: int = -1
    top_p: float = 1.0


@dataclass
class SimpleRealCodecResult:
    """Result of actual entropy encode followed by actual entropy decode."""

    decoded_indices: torch.Tensor
    estimated_bits: float
    payload_bits: int
    stats: dict[str, Any] = field(default_factory=dict)


class SimpleRealCodec:
    """Causal top-k tensor-rANS codec used by the simplified real path."""

    real_codec_mode = "causal"
    pipeline_mode = "causal_topk_tensor"
    backend = "tensor"
    entropy_coder = "topk_escape"
    decoder_kind = "independent_causal"
    decoder_token_source = "independent_causal_decoder"
    stream_layout = "merged_top_residual"

    def __init__(self, ar_model, codebook, padding_token: int, config: SimpleRealCodecConfig | None = None):
        self.ar_model = ar_model
        self.codebook = codebook
        self.padding_token = int(padding_token)
        self.config = config or SimpleRealCodecConfig()
        self.last_mask_all = None

    @property
    def mask_all(self):
        transformer_mask = getattr(getattr(self.ar_model, "transformer", None), "mask_all", None)
        return transformer_mask if transformer_mask is not None else self.last_mask_all

    def roundtrip(
        self,
        latents: EncodedLatents,
        *,
        transfer_slices: int | None = None,
        topk: int | None = None,
        temperature: float | None = None,
        sample_top_k: int | None = None,
        top_p: float | None = None,
        profile=None,
    ) -> SimpleRealCodecResult:
        """Encode transmitted prefix slices to bytes, then decode from bytes."""

        cfg = self.config
        transfer = cfg.transfer_slices if transfer_slices is None else int(transfer_slices)
        entropy_topk = cfg.topk if topk is None else int(topk)
        temp = cfg.temperature if temperature is None else float(temperature)
        sample_k = cfg.sample_top_k if sample_top_k is None else int(sample_top_k)
        nucleus_p = cfg.top_p if top_p is None else float(top_p)

        causal_result = causal_tensor_compress_and_decompress(
            self.ar_model,
            shape_list=latents.shape_list,
            gt_indices=latents.indices,
            transfer_slices=transfer,
            batch_size=latents.batch_patches,
            mask_padded=latents.mask_padded,
            codebook=self.codebook,
            padding_token=self.padding_token,
            topk=entropy_topk,
            precision=cfg.precision,
            temperature=temp,
            top_k=sample_k,
            top_p=nucleus_p,
            sample_logits=True,
            profile=profile,
        )
        self.last_mask_all = getattr(getattr(self.ar_model, "transformer", None), "mask_all", None)

        stats = dict(causal_result.stats or {})
        stats.update({
            "real_compression_mode": self.real_codec_mode,
            "real_codec_mode": self.real_codec_mode,
            "entropy_pipeline_mode": self.pipeline_mode,
            "entropy_backend": self.backend,
            "entropy_coder": self.entropy_coder,
            "entropy_topk": int(entropy_topk),
            "decoder_kind": self.decoder_kind,
            "decoder_token_source": self.decoder_token_source,
            "stream_layout": self.stream_layout,
            "use_bitstream_decoded_tokens": 1,
            "actual_entropy_roundtrip": 1,
        })
        return SimpleRealCodecResult(
            decoded_indices=causal_result.decoded_indices,
            estimated_bits=float(causal_result.estimated_bits),
            payload_bits=int(causal_result.payload_bits),
            stats=stats,
        )
