"""Backend interface for entropy-codec implementations."""

from __future__ import annotations

from typing import Protocol


class EntropyCodec(Protocol):
    """Minimal interface shared by concrete entropy backends."""

    backend: str

    def encode_packets(self, packets, profile=None):
        """Encode prepared packets into backend byte stream(s)."""

    def verify_packets(self, *args, profile=None):
        """Decode backend stream(s) against packet metadata for verification."""
