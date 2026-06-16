"""Tensor-native CPU rANS wrapper for RDVQ real entropy coding."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_EXT = None


def get_tensor_rans_ext():
    """Build/load the local tensor rANS extension lazily."""

    global _EXT
    if _EXT is not None:
        return _EXT

    root = Path(__file__).resolve().parent
    csrc = root / "csrc"
    sources = [str(csrc / "tensor_rans.cpp"), str(csrc / "tensor_rans_bindings.cpp")]
    _EXT = load(
        name="rdvq_tensor_rans",
        sources=sources,
        extra_include_paths=[str(csrc)],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=os.environ.get("RDVQ_BUILD_VERBOSE", "0").strip().lower() in {"1", "true", "yes", "on"},
    )
    return _EXT


def _cpu_int32_contiguous(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dtype != torch.int32:
        tensor = tensor.to(torch.int32)
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()
    return tensor.contiguous()


def encode_indexed(symbols: torch.Tensor, cdfs: torch.Tensor, precision: int = 16) -> bytes:
    """Encode ``symbols`` using one fixed-width CDF row per symbol."""

    symbols = _cpu_int32_contiguous(symbols, "symbols")
    cdfs = _cpu_int32_contiguous(cdfs, "cdfs")
    return get_tensor_rans_ext().encode_indexed_cdf(symbols, cdfs, int(precision))


def decode_indexed(stream: bytes, cdfs: torch.Tensor, precision: int = 16) -> torch.Tensor:
    """Decode a fixed-width CDF stream into an int32 CPU tensor."""

    cdfs = _cpu_int32_contiguous(cdfs, "cdfs")
    return get_tensor_rans_ext().decode_indexed_cdf(stream, cdfs, int(precision))


class IndexedRansDecoder:
    """Stateful row-wise CDF decoder for chunked causal rANS decoding."""

    def __init__(self, stream: bytes, precision: int = 16):
        self.precision = int(precision)
        self._decoder = get_tensor_rans_ext().IndexedRansDecoder()
        self._decoder.set_stream(bytes(stream))

    def decode_chunk(self, cdfs: torch.Tensor) -> torch.Tensor:
        cdfs = _cpu_int32_contiguous(cdfs, "cdfs")
        return self._decoder.decode_chunk(cdfs, self.precision)
