"""Shared data objects for RDVQ entropy-coding helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class DeterministicSelection:
    """Result of removing decoder-recoverable high-confidence symbols."""

    rec_selected: torch.Tensor
    code_selected_mask: torch.Tensor
    zero_stream: bool = False


@dataclass
class TopKEscapeCoding:
    """Prepared top-k/escape symbols, CDFs, and restore metadata."""

    sym_cpu: torch.Tensor
    sym_list: list[int] | None
    idx_list: list[int] | None
    cdf_list: list[list[int]] | None
    cdf_len_list: list[int] | None
    offset_list: list[int] | None
    tensor_payload: Any | None
    restore_topk: dict[str, Any]
    escape_count: int
