"""VQ_AR_Predictor — the active RDVQ autoregressive entropy model.

This is the multi-scale masked-Transformer predictor that estimates
conditional entropy during training and provides AR logits to the real codec at
inference time.
"""

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

from autoregressive.models.mask_generation import (
    generate_MS_ck_mask,
    generate_num_k_mask,
    rerank_input,
)
from tokenizer.tokenizer_image.models.ar_transformer import MS_input_transformer

from tokenizer.tokenizer_image.models.compression_model import CompressionModel
from utils.profile_accounting import (
    profile_add as _profile_add,
    profile_tic as _profile_tic,
    profile_toc as _profile_toc,
)


logger = logging.getLogger(__name__)


class VQ_AR_Predictor(CompressionModel):
    def __init__(self, in_channels=8, V_size=16384, d_model=512, nhead=8, num_layers=12, temperature=1., use_distance=False, l2_norm=False, num_ar_per_scale=4, use_start_token=True, use_patch_ck_ar = True):
        super().__init__()
        self.V_size = V_size
        self.in_channels = in_channels
        self.d_model = d_model
        self.temperature = temperature
        self.use_distance = use_distance
        self.l2_norm=l2_norm
        self.num_ar_per_scale = num_ar_per_scale
        self.use_start_token = use_start_token
        logger.debug("Using ar num: %s", num_ar_per_scale)
        logger.debug("Using start token: %s", use_start_token)

        # start token (embedding大小和in_channels保持一致)
        if self.use_start_token:
            self.start_token = nn.Parameter(torch.zeros(1, 1, d_model))
        else:
            self.start_token = None

        self.input_proj = nn.Linear(in_channels, d_model)
        self.use_patch_ck_ar = use_patch_ck_ar
        # causal_mask, mask_all_training = self.generate_MS_ck_mask([(4,4), (8,8), (16,16)], device="cuda")
        _, mask_all_training = generate_MS_ck_mask([(4,4), (8,8), (16,16)], device="cuda", use_patch_ck_ar=self.use_patch_ck_ar, k=self.num_ar_per_scale)

        self.transformer = MS_input_transformer(d_model=d_model, nhead=nhead, num_layers=num_layers, vocab_size=V_size, use_tok_embedding=False, use_patch_ck_ar=self.use_patch_ck_ar)

        self.mask_all = mask_all_training

    def entropy_code_ans(self, logits: torch.Tensor, ind: torch.Tensor, coding_mask=None, fill_value=0, profile=None, packet_position=None):
        """Removed legacy fast/debug entropy coder.

        Real bitstream coding now lives behind ``SimpleRealCodec`` so the model
        module does not depend on rANS internals.
        """
        return self._entropy_code_real_ans(
            logits,
            ind,
            coding_mask=coding_mask,
            fill_value=fill_value,
            profile=profile,
            packet_position=packet_position,
        )

    def encoding_decoding_compressai(self, logits: torch.Tensor, ind: torch.Tensor, coding_mask=None, fill_value=0, profile=None, packet_position=None):
        """Removed legacy CompressAI-named fast/debug entropy coder."""
        return self._entropy_code_real_ans(
            logits,
            ind,
            coding_mask=coding_mask,
            fill_value=fill_value,
            profile=profile,
            packet_position=packet_position,
        )

    def ar_one_step(
        self,
        idx=None,
        cond_idx=None,
        input_pos=None,
        targets=None,
        all_tokens=None,
        codebook=None,
        entropy_coding=False,
        valid_token_mask=None,
        padding_token=0,
        profile=None,
    ):
        ## bs: B*nh*nw, L: length of sequence
        if idx is None and cond_idx is None:
            raise ValueError("ar_one_step requires either idx or cond_idx")
        device = self.start_token.device

        t = _profile_tic(profile, device)
        if cond_idx is not None:
            token_embeddings = self.start_token.expand(cond_idx.shape[0], -1, -1)
            decoded_embeddings = None
        else:
            token_embeddings = codebook[idx]  # [bs, L, ch_in]
            decoded_embeddings = codebook[all_tokens]  # [bs, L, ch_in]
            token_embeddings = self.input_proj(token_embeddings)  # [bs, L, d_model]
            decoded_embeddings = self.input_proj(decoded_embeddings)  # [bs, L, d_model]
        _profile_toc(profile, "ar.embedding_project", t, device)

        if self.transformer.config.use_MS_ck_ar:
            t = _profile_tic(profile, device)
            token_embeddings = self.transformer.generate_query_w_last_token(token_embeddings, decoded_embeddings, input_pos)
            _profile_toc(profile, "ar.query_generation", t, device)

        Bs, L = token_embeddings.shape[0:2]
        t = _profile_tic(profile, device)
        logits = self.transformer.ar_one_step(token_embeddings, input_pos)  # [B, L, V_size]
        _profile_toc(profile, "ar.transformer", t, device)
        _profile_add(profile, "ar.calls", 1)

        entropy = None
        current_valid_mask = None
        targets_current = None
        if targets is not None:
            t = _profile_tic(profile, device)
            mask_all = self.transformer.mask_all
            curr_pos = mask_all == input_pos
            targets_current = targets.to(self.start_token.device)[:, curr_pos]
            if valid_token_mask is not None:
                current_valid_mask = valid_token_mask.to(self.start_token.device, dtype=torch.bool)[:, curr_pos]
                fill = torch.full_like(targets_current, int(padding_token))
                targets_current = torch.where(current_valid_mask, targets_current, fill)

            entropy = self.cross_entropy_log2(logits.view(-1, logits.size(-1)), targets_current.view(-1), reduction="none")
            entropy = entropy.view(Bs, -1)
            if current_valid_mask is not None:
                entropy = entropy * current_valid_mask.to(entropy.dtype)
            _profile_toc(profile, "ar.entropy_loss", t, device)

        if entropy_coding:
            t = _profile_tic(profile, device)
            bitstreams, ind = self.entropy_code_ans(
                logits,
                targets_current,
                coding_mask=current_valid_mask,
                fill_value=padding_token,
                profile=profile,
                packet_position=int(input_pos.reshape(-1)[0].item()) if input_pos is not None else None,
            )
            _profile_toc(profile, "ar.entropy_coding_call", t, device)
            return logits, entropy, bitstreams, ind
        else:
            return logits, entropy

    def full_forward_logits(self, quant, updated_shape=None):
        """Teacher-forced full-sequence logits for encoder-side coding.

        The query construction is identical to ``forward()`` but omits the
        cross-entropy calculation so real coding can build entropy packets
        slice-by-slice from the logits.  Causal masking still prevents each
        position from attending to future latent tokens.
        """
        B, L, _ = quant.shape
        start = self.start_token.expand(B, -1, -1)
        x = self.input_proj(quant)

        if not self.use_patch_ck_ar:
            x = torch.cat([start, x[:, :-1, :]], dim=1)

        device = x.device
        if self.training and updated_shape[-1] == (16, 16):
            mask_all = self.mask_all
        else:
            _, mask_all = generate_MS_ck_mask(
                input_shape=updated_shape,
                device=device,
                use_patch_ck_ar=self.use_patch_ck_ar,
                k=self.num_ar_per_scale,
            )

        if self.use_patch_ck_ar:
            if updated_shape is None:
                raise ValueError("updated_shape is required when use_patch_ck_ar=True")
            query = rerank_input(x, start, updated_shape, mask_all, k=self.num_ar_per_scale)
        else:
            query = x

        if self.training:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                query = query.to(torch.bfloat16)
                logits = self.transformer(ranked_input=query, shape_list=updated_shape)
        else:
            logits = self.transformer(ranked_input=query, shape_list=updated_shape)
        return logits.view(B, L, self.V_size)

    def forward(self, quant, gt_index=None, updated_shape=None):
        """
        self.use_patch_ck_ar : True (MS CK AR), False (Pixel AR)
        quant: [B, L, C]
        return: [B, L, V_size]
        """

        B, L, _ = quant.shape
        logits = self.full_forward_logits(quant, updated_shape=updated_shape)
        target_probs = gt_index.view(-1) if gt_index.dtype == torch.long else gt_index.view(-1, self.V_size)
        entropy = self.cross_entropy_log2(logits.view(-1, self.V_size), target_probs)
        entropy = entropy.view(B, L)

        return entropy, logits
