"""Codebook-index <-> entropy-symbol mapping helpers."""

from __future__ import annotations

import torch

from .packet import TensorEntropyPayload
from .probability import cdf_to_lists_profiled, quantize_pmf_to_cdf
from .profiling import _profile_add, _profile_tic, _profile_toc
from .specs import TopKEscapeCoding


def build_topk_escape_coding(
    pmf_selected: torch.Tensor,
    sym_code: torch.Tensor,
    topk: int,
    precision: int,
    *,
    use_tensor_rans: bool,
    profile=None,
    device_ref=None,
) -> TopKEscapeCoding:
    """Prepare top-k/escape symbols for CompressAI or tensor rANS.

    The first stream encodes rank-in-top-k plus an escape symbol.  Escaped
    positions get a second residual rank over non-top-k indices.
    """

    device = pmf_selected.device
    t = _profile_tic(profile, device_ref)
    sym_cpu = sym_code.to(torch.int32).cpu()
    _profile_toc(profile, "entropy.gpu_to_cpu", t, device_ref)

    t = _profile_tic(profile, device_ref)
    top_values, top_indices = torch.topk(pmf_selected, topk, dim=-1, largest=True, sorted=True)
    target_matches = top_indices == sym_code.unsqueeze(-1)
    in_top = target_matches.any(dim=-1)
    top_rank = torch.argmax(target_matches.to(torch.int64), dim=-1)
    escape_code = torch.full_like(top_rank, topk)
    top_symbols = torch.where(in_top, top_rank, escape_code).to(torch.int32)
    escape_mass = (1.0 - top_values.sum(dim=-1, keepdim=True)).clamp_min(torch.finfo(top_values.dtype).tiny)
    top_pmf = torch.cat((top_values, escape_mass), dim=-1)
    escape_mask = ~in_top
    escape_count = int(escape_mask.sum().item())
    _profile_toc(profile, "entropy.topk_select", t, device_ref)
    _profile_add(profile, "entropy.topk_escape_calls", 1)
    _profile_add(profile, "entropy.topk_value", topk)
    _profile_add(profile, "entropy.escape_symbols", escape_count)

    top_cdf = quantize_pmf_to_cdf(top_pmf, precision, profile=profile, key="entropy.topk_cdf_quantize", device_ref=device_ref)
    cdf_list = []
    cdf_len_list = []
    if not use_tensor_rans:
        top_cdf_list, top_cdf_len_list = cdf_to_lists_profiled(top_cdf, profile=profile, key="entropy.topk_cdf_to_lists", device_ref=device_ref)
    else:
        top_cdf_list, top_cdf_len_list = None, None

    residual_rank = torch.empty(0, dtype=torch.int32, device=device)
    residual_cdf = None
    residual_cdf_list = []
    residual_cdf_len_list = []
    residual_non_top_indices = None
    if escape_count > 0:
        t = _profile_tic(profile, device_ref)
        pmf_escape = pmf_selected[escape_mask]
        top_idx_escape = top_indices[escape_mask]
        target_escape = sym_code[escape_mask]
        residual_mask = torch.ones((escape_count, pmf_selected.shape[-1]), dtype=torch.bool, device=device)
        residual_mask.scatter_(1, top_idx_escape, False)
        residual_pmf = pmf_escape[residual_mask].reshape(escape_count, pmf_selected.shape[-1] - topk)
        all_indices = torch.arange(pmf_selected.shape[-1], device=device, dtype=torch.long).expand(escape_count, pmf_selected.shape[-1])
        residual_non_top_indices = all_indices[residual_mask].reshape(escape_count, pmf_selected.shape[-1] - topk)
        residual_matches = residual_non_top_indices == target_escape.unsqueeze(-1)
        if not bool(residual_matches.any(dim=-1).all().item()):
            raise RuntimeError("top-k escape target was not found in residual alphabet")
        residual_rank = torch.argmax(residual_matches.to(torch.int64), dim=-1).to(torch.int32)
        _profile_toc(profile, "entropy.escape_prepare", t, device_ref)

        residual_cdf = quantize_pmf_to_cdf(residual_pmf, precision, profile=profile, key="entropy.escape_cdf_quantize", device_ref=device_ref)
        if not use_tensor_rans:
            residual_cdf_list, residual_cdf_len_list = cdf_to_lists_profiled(residual_cdf, profile=profile, key="entropy.escape_cdf_to_lists", device_ref=device_ref)

    tensor_payload = None
    sym_list = None
    idx_list = None
    offset_list = None
    if use_tensor_rans:
        t = _profile_tic(profile, device_ref)
        tensor_payload = TensorEntropyPayload(
            top_symbols=top_symbols.cpu().contiguous(),
            top_cdfs=top_cdf.cpu().contiguous(),
            residual_symbols=residual_rank.cpu().contiguous(),
            residual_cdfs=None if residual_cdf is None else residual_cdf.cpu().contiguous(),
            topk=topk,
            precision=precision,
            top_indices=top_indices.to(torch.int32).cpu().contiguous(),
            residual_non_top_indices=None if residual_non_top_indices is None else residual_non_top_indices.to(torch.int32).cpu().contiguous(),
        )
        _profile_toc(profile, "entropy.tensor_payload_to_cpu", t, device_ref)
        _profile_add(profile, "entropy.tensor_packets", 1)
    else:
        t = _profile_tic(profile)
        top_sym_list = top_symbols.cpu().tolist()
        residual_sym_list = residual_rank.cpu().tolist()
        escape_mask_list = escape_mask.cpu().tolist()
        residual_cursor = 0
        sym_list = []
        for i, code in enumerate(top_sym_list):
            sym_list.append(int(code))
            cdf_list.append(top_cdf_list[i])
            cdf_len_list.append(top_cdf_len_list[i])
            if escape_mask_list[i]:
                sym_list.append(int(residual_sym_list[residual_cursor]))
                cdf_list.append(residual_cdf_list[residual_cursor])
                cdf_len_list.append(residual_cdf_len_list[residual_cursor])
                residual_cursor += 1
        idx_list = list(range(len(sym_list)))
        offset_list = [0] * len(sym_list)
        elapsed = _profile_toc(profile, "entropy.topk_list_prepare", t)
        _profile_add(profile, "entropy.cdf_build_and_lists", elapsed)

    restore_topk = {
        "topk": int(topk),
        "top_indices": top_indices,
        "residual_non_top_indices": residual_non_top_indices,
        "total_elements": int(sym_code.numel()),
    }
    return TopKEscapeCoding(
        sym_cpu=sym_cpu,
        sym_list=sym_list,
        idx_list=idx_list,
        cdf_list=cdf_list,
        cdf_len_list=cdf_len_list,
        offset_list=offset_list,
        tensor_payload=tensor_payload,
        restore_topk=restore_topk,
        escape_count=escape_count,
    )


def decode_topk_escape_symbol_list(decoded, restore_topk, *, dtype=None, device=None):
    """Map decoded top-k/escape ranks back to codebook indices."""

    topk = int(restore_topk["topk"])
    total_elements = int(restore_topk["total_elements"])
    top_dec = []
    residual_dec = []
    pos = 0
    for _ in range(total_elements):
        code = int(decoded[pos])
        pos += 1
        top_dec.append(code)
        if code == topk:
            residual_dec.append(int(decoded[pos]))
            pos += 1
    if pos != len(decoded):
        raise AssertionError("top-k escape decode consumed an unexpected number of symbols")

    if device is None:
        device = restore_topk["top_indices"].device
    if dtype is None:
        dtype = torch.long
    top_dec_tensor = torch.tensor(top_dec, dtype=torch.long, device=device)
    decoded_values = torch.empty(total_elements, dtype=dtype, device=device)
    top_indices = restore_topk["top_indices"].to(device=device)
    rows = torch.arange(total_elements, device=device)
    decoded_in_top = top_dec_tensor < topk
    if bool(decoded_in_top.any().item()):
        decoded_values[decoded_in_top] = top_indices[rows[decoded_in_top], top_dec_tensor[decoded_in_top]].to(dtype)
    if residual_dec:
        residual_non_top_indices = restore_topk["residual_non_top_indices"]
        if residual_non_top_indices is None or residual_non_top_indices.shape[0] != len(residual_dec):
            raise AssertionError("top-k escape residual count mismatch")
        residual_non_top_indices = residual_non_top_indices.to(device=device)
        residual_rank_tensor = torch.tensor(residual_dec, dtype=torch.long, device=device)
        residual_rows = torch.arange(len(residual_dec), device=device)
        decoded_values[~decoded_in_top] = residual_non_top_indices[residual_rows, residual_rank_tensor].to(dtype)
    return decoded_values


def map_tensor_topk_decoded_to_codebook(packet, top_decoded, residual_decoded):
    """Map tensor-rANS top/residual decoded symbols back to codebook indices."""

    payload = packet.tensor_payload
    top_decoded = top_decoded.to(torch.long)
    topk = int(payload.topk)
    total = int(top_decoded.numel())
    values = torch.empty(total, dtype=torch.int64)
    decoded_in_top = top_decoded < topk

    if payload.top_indices is None:
        raise RuntimeError("tensor top-k payload is missing top_indices metadata")
    top_indices = payload.top_indices.to(torch.long)
    rows = torch.arange(total, dtype=torch.long)
    if bool(decoded_in_top.any().item()):
        values[decoded_in_top] = top_indices[rows[decoded_in_top], top_decoded[decoded_in_top]]

    if bool((~decoded_in_top).any().item()):
        residual_non_top_indices = payload.residual_non_top_indices
        if residual_non_top_indices is None:
            raise RuntimeError("tensor top-k payload has escapes but no residual index metadata")
        residual_non_top_indices = residual_non_top_indices.to(torch.long)
        residual_rows = torch.arange(int(residual_decoded.numel()), dtype=torch.long)
        values[~decoded_in_top] = residual_non_top_indices[residual_rows, residual_decoded.to(torch.long)]
    return values


def map_list_topk_decoded_to_codebook(packet, decoded):
    """Map CompressAI top-k/escape decoded list symbols back to codebook indices."""

    if packet.restore_topk is None:
        return torch.tensor(decoded, dtype=torch.long)
    return decode_topk_escape_symbol_list(decoded, packet.restore_topk, dtype=torch.long, device=torch.device("cpu"))
