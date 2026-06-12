"""Tensor-native rANS backend wrapper."""

from __future__ import annotations

import torch

from tokenizer.tokenizer_image.entropy.streams.packet import _tensor_payload_tensors
from tokenizer.tokenizer_image.entropy.utils.profiling import _profile_add, _profile_tic, _profile_toc
from tokenizer.tokenizer_image.entropy.native.tensor_rans import decode_indexed, encode_indexed


class TensorRansEntropyCodec:
    """Encode and decode top-k/escape tensor payloads with tensor rANS."""

    backend = "tensor"

    def encode_packets(self, packets, profile=None) -> tuple[bytes, bytes, int, int]:
        precision, top_symbols, top_cdfs, residual_symbols, residual_cdfs = _tensor_payload_tensors(packets)
        if top_symbols.numel() == 0 and residual_symbols.numel() == 0:
            return b"", b"", 0, 0

        if top_symbols.numel() > 0:
            t = _profile_tic(profile)
            top_stream = encode_indexed(top_symbols, top_cdfs, precision=precision)
            _profile_toc(profile, "tensor_rans.top_encode", t)
        else:
            top_stream = b""

        if residual_symbols.numel() > 0:
            t = _profile_tic(profile)
            residual_stream = encode_indexed(residual_symbols, residual_cdfs, precision=precision)
            _profile_toc(profile, "tensor_rans.residual_encode", t)
        else:
            residual_stream = b""

        bits = (len(top_stream) + len(residual_stream)) * 8
        symbol_count = int(top_symbols.numel() + residual_symbols.numel())
        _profile_add(profile, "tensor_rans.top_symbols", int(top_symbols.numel()))
        _profile_add(profile, "tensor_rans.residual_symbols", int(residual_symbols.numel()))
        _profile_add(profile, "tensor_rans.payload_bits", bits)
        return top_stream, residual_stream, bits, symbol_count

    def decode_payloads(self, top_stream, residual_stream, packets, profile=None, *, token_decode=False):
        precision, top_symbols, top_cdfs, residual_symbols, residual_cdfs = _tensor_payload_tensors(packets)
        top_key = "tensor_rans.top_decode_tokens" if token_decode else "tensor_rans.top_decode"
        residual_key = "tensor_rans.residual_decode_tokens" if token_decode else "tensor_rans.residual_decode"

        if top_symbols.numel() > 0:
            t = _profile_tic(profile)
            top_decoded = decode_indexed(top_stream or b"", top_cdfs, precision=precision)
            if not torch.equal(top_decoded, top_symbols):
                suffix = " while restoring tokens" if token_decode else ""
                raise AssertionError(f"Tensor rANS top stream decode mismatch{suffix}!")
            _profile_toc(profile, top_key, t)
        else:
            top_decoded = torch.empty(0, dtype=torch.int32)

        if residual_symbols.numel() > 0:
            t = _profile_tic(profile)
            residual_decoded = decode_indexed(residual_stream or b"", residual_cdfs, precision=precision)
            if not torch.equal(residual_decoded, residual_symbols):
                suffix = " while restoring tokens" if token_decode else ""
                raise AssertionError(f"Tensor rANS residual stream decode mismatch{suffix}!")
            _profile_toc(profile, residual_key, t)
        else:
            residual_decoded = torch.empty(0, dtype=torch.int32)

        return top_decoded, residual_decoded

    def verify_packets(self, top_stream, residual_stream, packets, profile=None) -> None:
        self.decode_payloads(top_stream, residual_stream, packets, profile=profile, token_decode=False)
