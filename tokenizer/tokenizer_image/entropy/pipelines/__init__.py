"""Entropy-coding pipelines for RDVQ real compression."""

from .causal_tensor import causal_tensor_compress_and_decompress

__all__ = ["causal_tensor_compress_and_decompress"]
