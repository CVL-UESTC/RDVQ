"""Fast batched PMF-to-CDF helpers for real entropy coding.

CompressAI exposes only a scalar ``pmf_to_quantized_cdf`` binding.  The real
RDVQ path needs one CDF per transmitted token, so building those CDFs one row at
a time creates a large Python/C++ call overhead.  This module keeps the same
basic rANS contract (strictly positive integer PMF, final CDF value 2**precision)
and builds a whole batch with tensor operations.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F


def _validate_pmf(pmf: torch.Tensor, precision: int) -> tuple[int, int, int]:
    if pmf.ndim != 2:
        raise ValueError(f"pmf must be 2-D [N, K], got shape {tuple(pmf.shape)}")
    if precision <= 0 or precision > 30:
        raise ValueError(f"precision must be in [1, 30], got {precision}")
    total = 1 << precision
    rows, symbols = pmf.shape
    if symbols <= 0:
        raise ValueError("pmf must contain at least one symbol")
    if symbols > total:
        raise ValueError(
            f"cannot build a positive integer PMF with {symbols} symbols "
            f"and only {total} quantization slots"
        )
    if rows == 0:
        raise ValueError("pmf must contain at least one row")
    if not torch.isfinite(pmf).all():
        raise ValueError("pmf contains NaN or Inf values")
    if (pmf < 0).any():
        raise ValueError("pmf contains negative values")
    return rows, symbols, total


@torch.no_grad()
def batch_pmf_to_quantized_cdf(
    pmf: torch.Tensor,
    precision: int = 16,
    *,
    normalize: bool = True,
) -> torch.Tensor:
    """Convert a batch of PMFs to rANS-compatible quantized CDFs.

    Args:
        pmf: Probability tensor of shape ``[N, K]``. CPU and CUDA tensors are
            both accepted.
        precision: Entropy coder precision. The final CDF value is
            ``2 ** precision``.
        normalize: Renormalize rows before quantization. This is enabled by
            default as a safety boundary for tiny softmax drift.

    Returns:
        An ``int32`` tensor of shape ``[N, K + 1]`` on the same device as
        ``pmf``. Each row starts at 0, ends at ``2 ** precision``, and has
        strictly positive symbol frequencies.
    """

    _, symbols, total = _validate_pmf(pmf, precision)

    work = pmf.to(dtype=torch.float64)
    if normalize:
        row_sum = work.sum(dim=-1, keepdim=True)
        if (row_sum <= 0).any():
            raise ValueError("pmf contains a row with non-positive mass")
        work = work / row_sum

    # Assign one frequency slot to every symbol first, then distribute the
    # remaining mass by largest remainders. This is slightly smoother than the
    # old round-and-repair path, but it cannot fail for sharp distributions as
    # long as ``symbols <= 2**precision``.
    remaining = total - symbols
    scaled = work * remaining
    base = torch.floor(scaled).to(torch.int64)
    pmf_int = base + 1

    deficit = total - pmf_int.sum(dim=-1)
    if torch.any(deficit < 0):
        raise ValueError("quantized PMF exceeded precision budget")
    if torch.any(deficit > 0):
        remainders = scaled - base.to(scaled.dtype)
        max_deficit = int(deficit.max().item())
        ranked = torch.argsort(remainders, dim=-1, descending=True)[:, :max_deficit]
        add_mask = torch.arange(max_deficit, device=pmf.device).unsqueeze(0) < deficit.unsqueeze(1)
        pmf_int.scatter_add_(1, ranked, add_mask.to(torch.int64))

    cdf = F.pad(torch.cumsum(pmf_int, dim=-1), (1, 0), mode="constant", value=0)
    cdf[:, -1] = total
    return cdf.to(torch.int32)


def _cdf_to_lists_numpy(cdf_cpu: torch.Tensor) -> list[list[int]]:
    return cdf_cpu.numpy().tolist()


def cdf_to_compressai_lists_compact(cdf: torch.Tensor) -> tuple[list[list[int]], list[int]]:
    """Return CompressAI CDF lists while reusing identical rows.

    The rANS API accepts a Python list per symbol.  Some low-entropy slices can
    produce repeated quantized CDF rows; reusing the same list object avoids
    rebuilding those rows.  This keeps the stream format unchanged.
    """

    if cdf.ndim != 2:
        raise ValueError(f"cdf must be 2-D [N, K + 1], got shape {tuple(cdf.shape)}")
    if cdf.dtype != torch.int32:
        cdf = cdf.to(torch.int32)
    cdf_cpu = cdf.cpu().contiguous()
    cdf_array = cdf_cpu.numpy()
    cache: dict[bytes, list[int]] = {}
    cdf_list: list[list[int]] = []
    for row in cdf_array:
        key = row.tobytes()
        cached = cache.get(key)
        if cached is None:
            cached = row.tolist()
            cache[key] = cached
        cdf_list.append(cached)
    cdf_len_list = [cdf_cpu.shape[1]] * cdf_cpu.shape[0]
    return cdf_list, cdf_len_list


@torch.no_grad()
def cdf_to_compressai_lists(cdf: torch.Tensor) -> tuple[list[list[int]], list[int]]:
    """Return CompressAI ``cdf`` and ``cdf_lengths`` lists from a CDF tensor.

    ``RDVQ_CDF_LIST_MODE`` controls the conversion backend:
    ``numpy`` (default) uses NumPy's list conversion, ``torch`` uses
    ``Tensor.tolist()``, and ``compact`` reuses identical row objects.
    """

    if cdf.ndim != 2:
        raise ValueError(f"cdf must be 2-D [N, K + 1], got shape {tuple(cdf.shape)}")
    if cdf.dtype != torch.int32:
        cdf = cdf.to(torch.int32)
    cdf_cpu = cdf.cpu().contiguous()
    mode = os.environ.get("RDVQ_CDF_LIST_MODE", "numpy").strip().lower()
    if mode == "torch":
        cdf_list = cdf_cpu.tolist()
    elif mode == "compact":
        return cdf_to_compressai_lists_compact(cdf_cpu)
    else:
        cdf_list = _cdf_to_lists_numpy(cdf_cpu)
    cdf_len_list = [cdf_cpu.shape[1]] * cdf_cpu.shape[0]
    return cdf_list, cdf_len_list
