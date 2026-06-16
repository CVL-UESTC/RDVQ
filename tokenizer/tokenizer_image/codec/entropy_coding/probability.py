"""Probability and CDF helpers for RDVQ entropy coding."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from compressai._CXX import pmf_to_quantized_cdf as _pmf_to_quantized_cdf

from tokenizer.tokenizer_image.codec.entropy_coding.native.fast_cdf import (
    batch_pmf_to_quantized_cdf,
    cdf_to_compressai_lists,
)

from .profiling import _profile_add, _profile_tic, _profile_toc
from .specs import DeterministicSelection


def logits_to_pmf(logits: torch.Tensor, profile=None) -> torch.Tensor:
    """Convert logits to float32 PMF with the historical profile key."""

    t = _profile_tic(profile, logits)
    pmf = F.softmax(logits, dim=-1)
    _profile_toc(profile, "entropy.softmax", t, logits)
    return pmf


def validate_symbol_range(ind: torch.Tensor, alphabet_size: int) -> None:
    """Check that codebook indices fit the current entropy alphabet."""

    if torch.any((ind < 0) | (ind >= alphabet_size)):
        raise ValueError(
            f"entropy symbols must be in [0, {alphabet_size}), "
            f"got min={int(ind.min())}, max={int(ind.max())}"
        )


def apply_deterministic_selection(
    pmf_selected_all: torch.Tensor,
    sym_selected_all: torch.Tensor,
    rec_selected: torch.Tensor,
    code_selected_mask: torch.Tensor,
    threshold: float,
    dtype: torch.dtype,
    profile=None,
    device_ref=None,
) -> DeterministicSelection:
    """Remove symbols the decoder can recover as argmax from visible logits.

    If any high-confidence position disagrees with the target, the whole step
    falls back to entropy coding to avoid hiding exceptions.
    """

    t = _profile_tic(profile, device_ref)
    max_prob, argmax_idx = torch.max(pmf_selected_all, dim=-1)
    deterministic_mask = max_prob >= float(threshold)
    deterministic_count = int(deterministic_mask.sum().item())
    _profile_add(profile, "entropy.deterministic_candidates", deterministic_count)
    _profile_add(profile, "entropy.deterministic_checked_steps", 1)

    if deterministic_count > 0:
        deterministic_exception_mask = deterministic_mask & (argmax_idx != sym_selected_all)
        deterministic_exception_count = int(deterministic_exception_mask.sum().item())
        _profile_add(profile, "entropy.deterministic_exceptions", deterministic_exception_count)
        if deterministic_exception_count == 0:
            rec_selected[deterministic_mask] = argmax_idx[deterministic_mask].to(dtype)
            code_selected_mask = ~deterministic_mask
            _profile_add(profile, "entropy.deterministic_symbols", deterministic_count)
            if not bool(code_selected_mask.any().item()):
                _profile_add(profile, "entropy.deterministic_zero_stream_steps", 1)
                _profile_toc(profile, "entropy.deterministic_select", t, device_ref)
                return DeterministicSelection(rec_selected, code_selected_mask, zero_stream=True)
        else:
            _profile_add(profile, "entropy.deterministic_fallback_steps", 1)

    _profile_toc(profile, "entropy.deterministic_select", t, device_ref)
    return DeterministicSelection(rec_selected, code_selected_mask, zero_stream=False)


def quantize_pmf_to_cdf(pmf: torch.Tensor, precision: int, profile=None, key: str = "entropy.cdf_quantize_batch", device_ref=None) -> torch.Tensor:
    """Batch-quantize PMF rows into int32 CDF rows."""

    t = _profile_tic(profile, device_ref)
    cdf = batch_pmf_to_quantized_cdf(pmf, precision)
    elapsed = _profile_toc(profile, key, t, device_ref)
    _profile_add(profile, "entropy.cdf_build_and_lists", elapsed)
    return cdf


def cdf_to_lists_profiled(cdf: torch.Tensor, profile=None, key: str = "entropy.cdf_to_lists", device_ref=None):
    """Convert CDF tensor rows to CompressAI lists with accounting."""

    t = _profile_tic(profile, device_ref)
    cdf_list, cdf_len_list = cdf_to_compressai_lists(cdf)
    elapsed = _profile_toc(profile, key, t, device_ref)
    _profile_add(profile, "entropy.cdf_build_and_lists", elapsed)
    return cdf_list, cdf_len_list


def build_full_cdf_lists(pmf_selected: torch.Tensor, precision: int, chunk_rows: int, profile=None, device_ref=None):
    """Build full-alphabet CompressAI CDF lists using batched chunks."""

    if chunk_rows <= 0:
        chunk_rows = pmf_selected.shape[0]
    cdf_list = []
    cdf_len_list = []
    for start in range(0, pmf_selected.shape[0], chunk_rows):
        pmf_chunk = pmf_selected[start : start + chunk_rows]
        cdf_tensor = quantize_pmf_to_cdf(pmf_chunk, precision, profile=profile, key="entropy.cdf_quantize_batch", device_ref=device_ref)
        cdf_chunk, cdf_len_chunk = cdf_to_lists_profiled(cdf_tensor, profile=profile, key="entropy.cdf_to_lists", device_ref=device_ref)
        cdf_list.extend(cdf_chunk)
        cdf_len_list.extend(cdf_len_chunk)
    return cdf_list, cdf_len_list


def build_scalar_cdf_lists(pmf_cpu: torch.Tensor, precision: int, profile=None):
    """Legacy scalar CompressAI CDF path kept only as an explicit fallback."""

    cdf_list = []
    cdf_len_list = []
    total_elements = pmf_cpu.shape[0]
    t = _profile_tic(profile)
    for i in range(total_elements):
        quantized_cdf = _pmf_to_quantized_cdf(pmf_cpu[i].tolist(), precision)
        cdf_list.append(quantized_cdf)
        cdf_len_list.append(len(quantized_cdf))
    elapsed = _profile_toc(profile, "entropy.cdf_build_scalar", t)
    _profile_add(profile, "entropy.cdf_build_and_lists", elapsed)
    return cdf_list, cdf_len_list


@torch.no_grad()
def pmf_to_quantized_cdf(pmf: torch.Tensor, precision: int = 16) -> torch.Tensor:
    """Compatibility wrapper matching the historical scalar helper."""

    cdf = _pmf_to_quantized_cdf(pmf.tolist(), precision)
    return torch.IntTensor(cdf)
