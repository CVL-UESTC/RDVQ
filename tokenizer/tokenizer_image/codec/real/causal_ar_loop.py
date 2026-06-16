"""Causal AR model state used by the real entropy codec pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def num_ar_slices(shape_list, k: int) -> int:
    """Return the number of AR slices implied by the multi-scale shape list."""

    return sum([int(k) * 2 ** i for i in range(len(shape_list))])


def max_new_tokens(shape_list) -> int:
    """Return the flattened token length across all latent scales."""

    return sum([pn[0] * pn[1] for pn in shape_list])


def valid_token_mask_from_padding(mask_padded, device):
    """Convert VQ padding mask to a boolean valid-token mask."""

    return None if mask_padded is None else (mask_padded.to(device) == 0)


@dataclass
class CausalARState:
    """Mutable AR encoding/decoding state for one image or patch batch.

    This class owns model-side state only: transformer caches, the flattened
    token history, valid-token masks, and AR logits. It intentionally knows
    nothing about rANS bytes, CDF rows, or top-k/escape residual streams.
    """

    model: object
    shape_list: list
    batch_size: int
    codebook: torch.Tensor
    padding_token: int
    valid_token_mask: torch.Tensor | None
    profile: dict | None = None

    def setup(self):
        """Build transformer caches and initialize causal token history."""

        self.device = self.model.start_token.device
        self.max_tokens = max_new_tokens(self.shape_list)
        self.cond = torch.zeros(self.batch_size, dtype=torch.int, device=self.device).view(self.batch_size, 1)
        self.all_tokens = torch.full(
            (self.batch_size, self.max_tokens),
            int(self.padding_token),
            dtype=torch.long,
            device=self.device,
        )
        with torch.device(self.device):
            self.model.transformer.setup_caches(
                max_batch_size=self.batch_size,
                shape_list=self.shape_list,
                dtype=self.model.start_token.dtype,
            )
        self.mask_all = self.model.transformer.mask_all
        self.cur_token = None
        return self

    def destroy(self):
        """Destroy transformer caches owned by this state."""

        self.model.transformer.destroy_caches()

    def logits_for_slice(self, slice_idx: int):
        """Run one causal AR forward step for the current slice."""

        input_pos = torch.tensor([int(slice_idx)], device=self.device, dtype=torch.int)
        if int(slice_idx) == 0:
            logits, _ = self.model.ar_one_step(
                None,
                self.cond,
                input_pos,
                targets=None,
                codebook=self.codebook,
                entropy_coding=False,
                profile=self.profile,
            )
        else:
            logits, _ = self.model.ar_one_step(
                self.cur_token,
                cond_idx=None,
                input_pos=input_pos,
                all_tokens=self.all_tokens,
                targets=None,
                codebook=self.codebook,
                entropy_coding=False,
                profile=self.profile,
            )
        return logits

    def slice_positions(self, slice_idx: int):
        """Return flattened-token positions for one AR slice."""

        return self.mask_all == int(slice_idx)

    def valid_mask_for_slice(self, slice_idx: int):
        """Return valid-token mask for one AR slice."""

        if self.valid_token_mask is None:
            return None
        return self.valid_token_mask[:, self.slice_positions(slice_idx)].bool()

    def target_slice(self, gt_indices, slice_idx: int):
        """Gather ground-truth indices for one AR slice."""

        return gt_indices.to(device=self.device, dtype=torch.long)[:, self.slice_positions(slice_idx)]

    def write_slice(self, slice_idx: int, tokens):
        """Write tokens into causal history for later AR slices."""

        positions = self.slice_positions(slice_idx)
        self.all_tokens[:, positions] = tokens.to(device=self.device, dtype=torch.long)
        self.cur_token = self.all_tokens[:, positions].reshape(-1, 1)

    def all_indices(self):
        """Return the full flattened token history."""

        return self.all_tokens
