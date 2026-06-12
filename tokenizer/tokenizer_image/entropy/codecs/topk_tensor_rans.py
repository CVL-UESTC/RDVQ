"""Top-k/escape tensor-rANS codec for causal RDVQ decoding.

This module is intentionally the rANS black box used by the causal real codec.
The AR pipeline supplies logits and target/valid tensors; this module owns the
entropy-coding details: PMF/CDF construction, top-k rank mapping, escape
residual mapping, tensor-rANS byte encoding, and tensor-rANS byte decoding.

Why top-k/escape:
- The RDVQ codebook can contain thousands of entries.
- Building and coding a full CDF for every token is expensive.
- Most probability mass is usually concentrated in the top-k candidates.

How encoding works:
1. Convert AR logits to a PMF.
2. Select the top-k most likely codebook indices for each token.
3. If the target index is in top-k, encode its top-k rank.
4. Otherwise encode an ESCAPE symbol in the top stream and encode the target's
   residual rank among non-top-k indices in the residual stream.
5. Merge all top-rank symbols into one tensor-rANS stream.
6. Merge all residual escape symbols into one tensor-rANS stream.

How decoding works:
1. The decoder recomputes logits from its own decoded AR history.
2. It rebuilds the same top-k PMF/CDF for the current slice.
3. It decodes top-rank symbols from the top stream.
4. For ESCAPE symbols, it rebuilds the residual CDF and decodes residual ranks.
5. It maps ranks back to codebook indices and writes them into AR history.

Important:
- The decoder must not use encoder-side logits or encoder-side CDFs.
- The byte streams are real entropy-coded payloads, not simulated tokens.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tokenizer.tokenizer_image.entropy.streams.packet import TensorEntropyPayload
from tokenizer.tokenizer_image.entropy.symbols.probability import logits_to_pmf, quantize_pmf_to_cdf
from tokenizer.tokenizer_image.entropy.utils.profiling import _profile_add, _profile_tic, _profile_toc
from tokenizer.tokenizer_image.entropy.symbols.symbol_mapping import build_topk_escape_coding
from tokenizer.tokenizer_image.entropy.native.tensor_rans import IndexedRansDecoder, encode_indexed


@dataclass(frozen=True)
class TopkTensorRansConfig:
    """Configuration for causal top-k tensor-rANS coding."""

    topk: int = 1024
    precision: int = 16
    padding_token: int = 0


@dataclass
class EncodedTopkTensorStreams:
    """Merged tensor-rANS streams for transmitted prefix slices."""

    top_stream: bytes
    residual_stream: bytes
    top_symbols: int
    residual_symbols: int

    @property
    def payload_bits(self) -> int:
        return (len(self.top_stream) + len(self.residual_stream)) * 8

    @property
    def symbol_count(self) -> int:
        return int(self.top_symbols) + int(self.residual_symbols)


class TopkTensorRansCodec:
    """Black-box top-k/escape tensor-rANS codec.

    The codec is stateful within one image/patch roundtrip: encoder calls append
    per-slice tensor payloads, ``finish_encoding`` merges them into byte streams,
    and decoder calls consume those streams slice by slice.
    """

    def __init__(self, config: TopkTensorRansConfig):
        self.config = config
        self._payloads: list[TensorEntropyPayload] = []
        self._top_decoder: IndexedRansDecoder | None = None
        self._residual_decoder: IndexedRansDecoder | None = None

    def encode_slice(self, logits, targets, valid_mask, profile=None) -> torch.Tensor:
        """Prepare one transmitted slice and append its tensor-rANS payload.

        Returns the token tensor that the encoder should write into causal
        history for later AR slices. Invalid padded positions are filled with
        ``padding_token`` and are not entropy-coded.
        """

        device = logits.device
        pmf = logits_to_pmf(logits, profile=profile)
        _, _, alphabet = pmf.shape
        topk = int(self.config.topk)
        if not 0 < topk < int(alphabet):
            raise ValueError(f"causal tensor path requires 0 < topk < alphabet, got topk={topk}, alphabet={alphabet}")

        coding_mask = torch.ones_like(targets, dtype=torch.bool, device=device) if valid_mask is None else valid_mask.to(device=device, dtype=torch.bool)
        history_tokens = torch.full_like(targets, int(self.config.padding_token))
        if int(coding_mask.sum().item()) == 0:
            return history_tokens

        pmf_selected = pmf[coding_mask].reshape(-1, alphabet).contiguous()
        sym_code = targets.to(device=device, dtype=torch.long)[coding_mask].reshape(-1)
        topk_coding = build_topk_escape_coding(
            pmf_selected,
            sym_code,
            topk,
            int(self.config.precision),
            use_tensor_rans=True,
            profile=profile,
            device_ref=logits,
        )
        if topk_coding.tensor_payload is not None:
            self._payloads.append(topk_coding.tensor_payload)

        history_tokens[coding_mask] = sym_code.to(dtype=targets.dtype)
        return history_tokens

    def finish_encoding(self, profile=None) -> EncodedTopkTensorStreams:
        """Merge all pending transmitted slices into top/residual byte streams."""

        top_payloads = [payload for payload in self._payloads if payload.top_symbols.numel() > 0]
        if top_payloads:
            top_symbols = torch.cat([payload.top_symbols for payload in top_payloads], dim=0)
            top_cdfs = torch.cat([payload.top_cdfs for payload in top_payloads], dim=0)
            t = _profile_tic(profile)
            top_stream = encode_indexed(top_symbols, top_cdfs, precision=int(self.config.precision))
            _profile_toc(profile, "tensor_rans.top_encode", t)
        else:
            top_symbols = torch.empty(0, dtype=torch.int32)
            top_stream = b""

        residual_payloads = [payload for payload in self._payloads if payload.residual_symbols.numel() > 0]
        if residual_payloads:
            residual_symbols = torch.cat([payload.residual_symbols for payload in residual_payloads], dim=0)
            residual_cdfs = torch.cat([payload.residual_cdfs for payload in residual_payloads], dim=0)
            t = _profile_tic(profile)
            residual_stream = encode_indexed(residual_symbols, residual_cdfs, precision=int(self.config.precision))
            _profile_toc(profile, "tensor_rans.residual_encode", t)
        else:
            residual_symbols = torch.empty(0, dtype=torch.int32)
            residual_stream = b""

        bits = (len(top_stream) + len(residual_stream)) * 8
        _profile_add(profile, "tensor_rans.top_symbols", int(top_symbols.numel()))
        _profile_add(profile, "tensor_rans.residual_symbols", int(residual_symbols.numel()))
        _profile_add(profile, "tensor_rans.payload_bits", bits)
        return EncodedTopkTensorStreams(
            top_stream=top_stream,
            residual_stream=residual_stream,
            top_symbols=int(top_symbols.numel()),
            residual_symbols=int(residual_symbols.numel()),
        )

    def begin_decoding(self, streams: EncodedTopkTensorStreams):
        """Attach encoded byte streams to stateful tensor-rANS decoders."""

        self._top_decoder = IndexedRansDecoder(streams.top_stream, precision=int(self.config.precision)) if streams.top_stream else None
        self._residual_decoder = IndexedRansDecoder(streams.residual_stream, precision=int(self.config.precision)) if streams.residual_stream else None

    def decode_slice(self, logits, valid_mask, profile=None) -> torch.Tensor:
        """Decode one transmitted slice using decoder-rebuilt logits/CDFs."""

        device = logits.device
        pmf = logits_to_pmf(logits, profile=profile)
        B, N, alphabet = pmf.shape
        topk = int(self.config.topk)
        if not 0 < topk < int(alphabet):
            raise ValueError(f"causal tensor path requires 0 < topk < alphabet, got topk={topk}, alphabet={alphabet}")

        coding_mask = torch.ones((B, N), dtype=torch.bool, device=device) if valid_mask is None else valid_mask.to(device=device, dtype=torch.bool)
        decoded_slice = torch.full((B, N), int(self.config.padding_token), dtype=torch.long, device=device)
        transmitted = int(coding_mask.sum().item())
        if transmitted == 0:
            return decoded_slice
        if self._top_decoder is None:
            raise AssertionError("top rANS decoder is missing for a transmitted slice")

        pmf_selected = pmf[coding_mask].reshape(-1, alphabet).contiguous()

        t = _profile_tic(profile, logits)
        top_values, top_indices = torch.topk(pmf_selected, topk, dim=-1, largest=True, sorted=True)
        escape_mass = (1.0 - top_values.sum(dim=-1, keepdim=True)).clamp_min(torch.finfo(top_values.dtype).tiny)
        top_pmf = torch.cat((top_values, escape_mass), dim=-1)
        _profile_toc(profile, "causal.decoder_topk_select", t, logits)

        top_cdf = quantize_pmf_to_cdf(top_pmf, int(self.config.precision), profile=profile, key="causal.decoder_top_cdf", device_ref=logits)
        t = _profile_tic(profile)
        top_decoded = self._top_decoder.decode_chunk(top_cdf.cpu().contiguous()).to(device=device, dtype=torch.long)
        _profile_toc(profile, "causal.decoder_top_ans", t)
        if int(top_decoded.numel()) != transmitted:
            raise AssertionError(f"decoded top symbol count mismatch: got {top_decoded.numel()}, expected {transmitted}")

        values = torch.empty(transmitted, dtype=torch.long, device=device)
        decoded_in_top = top_decoded < topk
        rows = torch.arange(transmitted, device=device)
        if bool(decoded_in_top.any().item()):
            values[decoded_in_top] = top_indices[rows[decoded_in_top], top_decoded[decoded_in_top]]

        escape_mask = ~decoded_in_top
        escape_count = int(escape_mask.sum().item())
        if escape_count > 0:
            if self._residual_decoder is None:
                raise AssertionError("residual rANS decoder is missing but escape symbols were decoded")
            t = _profile_tic(profile, logits)
            pmf_escape = pmf_selected[escape_mask]
            top_idx_escape = top_indices[escape_mask]
            residual_mask = torch.ones((escape_count, alphabet), dtype=torch.bool, device=device)
            residual_mask.scatter_(1, top_idx_escape, False)
            residual_pmf = pmf_escape[residual_mask].reshape(escape_count, alphabet - topk)
            all_indices = torch.arange(alphabet, device=device, dtype=torch.long).expand(escape_count, alphabet)
            residual_non_top_indices = all_indices[residual_mask].reshape(escape_count, alphabet - topk)
            _profile_toc(profile, "causal.decoder_residual_prepare", t, logits)

            residual_cdf = quantize_pmf_to_cdf(residual_pmf, int(self.config.precision), profile=profile, key="causal.decoder_residual_cdf", device_ref=logits)
            t = _profile_tic(profile)
            residual_decoded = self._residual_decoder.decode_chunk(residual_cdf.cpu().contiguous()).to(device=device, dtype=torch.long)
            _profile_toc(profile, "causal.decoder_residual_ans", t)
            if int(residual_decoded.numel()) != escape_count:
                raise AssertionError(f"decoded residual symbol count mismatch: got {residual_decoded.numel()}, expected {escape_count}")
            residual_rows = torch.arange(escape_count, device=device)
            values[escape_mask] = residual_non_top_indices[residual_rows, residual_decoded]

        decoded_slice[coding_mask] = values
        return decoded_slice
