"""Entropy payload structures for the public causal tensor-rANS codec."""

from dataclasses import dataclass

import torch


@dataclass
class TensorEntropyPayload:
    """Tensor-native top-k/escape payload before byte encoding."""

    top_symbols: torch.Tensor
    top_cdfs: torch.Tensor
    residual_symbols: torch.Tensor
    residual_cdfs: torch.Tensor | None
    topk: int
    precision: int = 16
    top_indices: torch.Tensor | None = None
    residual_non_top_indices: torch.Tensor | None = None


@dataclass
class EncodedEntropyStream:
    """Encoded image-level tensor-rANS payload metadata."""

    payload_bits: int
    backend: str
    compressai_stream: bytes | None = None
    tensor_top_stream: bytes | None = None
    tensor_residual_stream: bytes | None = None
    packet_count: int = 0
    symbol_count: int = 0
