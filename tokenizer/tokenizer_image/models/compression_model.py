"""CompressionModel base class and historical neural entropy helpers.

This module contains model-side Gaussian mixture / slice-prediction utilities
and the standalone RMSNorm. Actual real bitstream coding lives under
``tokenizer.tokenizer_image.codec``.
"""

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

logger = logging.getLogger(__name__)


# ── Standalone helpers ──────────────────────────────────────────────────


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


# ── CompressionModel ────────────────────────────────────────────────────


class CompressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.masks = {}
        self.res_slice = None
        self.spatial_adaptor = None
        self.scale_predictor = None
        self.mean_predictor = None

    def quant(self, x):
        return torch.round(x)

    def get_curr_q(self, q_scale, q_basic, q_index=None):
        q_scale = q_scale[q_index]
        return q_basic * q_scale

    def prob_round(self, x):
        if self.training:
            half = float(0.5)
            noise = torch.empty_like(x).uniform_(-half, half)
            x = x + noise
        else:
            x = torch.round(x)
        return x

    def closest_factors(self, n: int):
        assert n % 4 == 0, "n must be divided by 4"
        root = int(math.isqrt(n))  # sqrt(n) 向下取整
        for a in range(root, 0, -1):
            if n % a == 0:
                b = n // a
                return a, b

    @staticmethod
    def probs_to_bits(probs):
        bits = -1.0 * torch.log(probs + 1e-5) / math.log(2.0)
        bits = torch.clamp_min(bits, 0)
        return bits

    def get_y_mixture_gaussian_bits(self, y, sigma, weight, scaling_factor=1):
        """
        if gaussian_dim==1:
            y,sigma,weight: (B,H, W,num_gaussian)
        if gaussian_dim==2:
            y:(B,H,W,num_gaussian*gaussian_dim)
            sigma:(sigma_x,sigma_y):(B,H,W,num_gaussian*gaussian_dim)
            weight: (B,H, W,num_gaussian)
        """
        if self.gaussian_dim==1:
            y = y / scaling_factor
            mu = torch.zeros_like(y)
            sigma = sigma.clamp(1e-5, 1e10)
            gaussian = torch.distributions.normal.Normal(mu, sigma)
            probs = gaussian.cdf(y + 0.5/scaling_factor) - gaussian.cdf(y - 0.5/scaling_factor) #(B, H, W, num_gaussian)
        elif self.gaussian_dim == 2:
            # 拆成 x 和 y 两部分
            y_x, y_y = y       # (B,H,W,num_gaussian)
            y_x = y_x/ scaling_factor
            y_y = y_y/ scaling_factor
            sigma_x, sigma_y = sigma
            sigma_x = sigma_x.clamp(1e-5, 1e10)
            sigma_y = sigma_y.clamp(1e-5, 1e10)

            mu_x = torch.zeros_like(y_x)
            mu_y = torch.zeros_like(y_y)

            gaussian_x = torch.distributions.Normal(mu_x, sigma_x)
            gaussian_y = torch.distributions.Normal(mu_y, sigma_y)

            probs_x = gaussian_x.cdf(y_x + 0.5/scaling_factor) - gaussian_x.cdf(y_x - 0.5/scaling_factor)
            probs_y = gaussian_y.cdf(y_y + 0.5/scaling_factor) - gaussian_y.cdf(y_y - 0.5/scaling_factor)

            # 独立 -> 联合概率 = 乘积
            probs = probs_x * probs_y   # (B,H,W,num_gaussian)

        else:
            raise NotImplementedError("Only 1D and 2D Gaussian supported")

        assert (torch.abs(weight.sum(dim=-1)-1) < 1e-4).all() ## weight should be summed equal to 1
        probs = torch.sum(probs*weight, dim=-1)
        probs = probs.clamp(1e-8, 1)
        entropy = -torch.log2(probs)
        if not self.training:
            entropy = entropy.clamp_min(0)
        return entropy

    def get_y_gaussian_bits(self, y, sigma, scaling_factor=1):
        y = y / scaling_factor
        mu = torch.zeros_like(sigma)
        sigma = sigma.clamp(1e-5, 1e10)
        gaussian = torch.distributions.normal.Normal(mu, sigma)
        probs = gaussian.cdf(y + 0.5/scaling_factor) - gaussian.cdf(y - 0.5/scaling_factor)
        probs = probs.clamp(1e-8, 1)
        entropy = -torch.log2(probs)
        return entropy

    def get_y_laplace_bits(self, y, sigma, scaling_factor=1):
        y = y /scaling_factor
        mu = torch.zeros_like(sigma)
        sigma = sigma.clamp(1e-5, 1e10)
        gaussian = torch.distributions.laplace.Laplace(mu, sigma)
        probs = gaussian.cdf(y + 0.5/scaling_factor) - gaussian.cdf(y - 0.5/scaling_factor)
        probs = probs.clamp(1e-8, 1)
        entropy = -torch.log2(probs)
        return entropy

    def log_probs_with_top_k_top_p_(self, logits_BlV: torch.Tensor, top_k: int = 0, top_p: float = 0.0, num_samples_portion=0.1) -> torch.Tensor:  # return log(probs)
        B, l, V = logits_BlV.shape
        num_samples_escape = int(num_samples_portion*l)
        if num_samples_escape>0:
            logits_wo_process = logits_BlV[:,:num_samples_escape,:]
            logits_BlV = logits_BlV[:,num_samples_escape:,:]
        if top_k > 0:
            idx_to_remove = logits_BlV < logits_BlV.topk(top_k, largest=True, sorted=False, dim=-1)[0].amin(dim=-1, keepdim=True)
            logits_BlV.masked_fill_(idx_to_remove, -torch.inf)
        if top_p > 0:
            sorted_logits, sorted_idx = logits_BlV.sort(dim=-1, descending=False)
            sorted_idx_to_remove = sorted_logits.softmax(dim=-1).cumsum_(dim=-1) <= (1 - top_p)
            sorted_idx_to_remove[..., -1:] = False
            logits_BlV.masked_fill_(sorted_idx_to_remove.scatter(sorted_idx.ndim - 1, sorted_idx, sorted_idx_to_remove), -torch.inf)
        logits = torch.cat((logits_wo_process,logits_BlV), dim=1).view(B*l, V)
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs+1e-6)/torch.log(torch.tensor(2.0))
        return log_probs

    def softmax(self, input, dim=-1, temp=None):
        t = temp if temp is not None else self.temperature
        probs = F.softmax(input/t, dim=dim)
        return probs

    def cross_entropy_log2(self, input, target, reduction="none"):
        """
        log2 cross entropy loss
        """

        log_probs = F.log_softmax(input, dim=-1) / math.log(2.0)

        if target.dim() == 1 or target.shape == input.shape:
            if target.dim() == 1:
                loss = F.nll_loss(log_probs, target, reduction="none")

            else:
                if not torch.allclose(target.sum(dim=1), torch.tensor(1.0), atol=1e-3):
                    target = F.softmax(target, dim=1)

                loss = -torch.sum(target * log_probs, dim=1)

        else:
            raise ValueError(f"incompatible dimensions for input and target: input shape {input.shape}, target shape {target.shape}")

        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss

    def process_mask(self, mask, n_h, n_w, h, w):
        ## return mask: 1, h, w
        mask_full = mask.repeat(n_h, n_w)
        mask_full = mask_full[:h, :w]  # Crop to the desired size
        return mask_full.unsqueeze(0)

    def generate_checkboard_mask(self, h, w, dtype, device):
        curr_mask_str = f"{h}x{w}"
        if curr_mask_str not in self.masks:
            n_h = (h+1)//2
            n_w = (w+1)//2
            m1_ = torch.tensor(((1, 0), (0, 0)), dtype=dtype, device=device)
            m2_ = torch.tensor(((0, 1), (0, 0)), dtype=dtype, device=device)
            m3_ = torch.tensor(((0, 0), (1, 0)), dtype=dtype, device=device)
            m4_ = torch.tensor(((0, 0), (0, 1)), dtype=dtype, device=device)
            m1 = self.process_mask(m1_, n_h, n_w, h, w)
            m2 = self.process_mask(m2_, n_h, n_w, h, w)
            m3 = self.process_mask(m3_, n_h, n_w, h, w)
            m4 = self.process_mask(m4_, n_h, n_w, h, w)
            self.masks[curr_mask_str] = [m1, m2, m3, m4]
            return [m1, m2, m3, m4]

        return self.masks[curr_mask_str]

    def process_with_mask(self, y, scales, means, mask):
        scales_hat = scales * mask
        means_hat = means * mask

        y_res = (y - means_hat) * mask
        y_q = self.ste_round(y_res)
        y_hat = y_q + means_hat

        return y_res, y_q, y_hat, scales_hat

    def slice_prediction(self, param, curr_prediction):
        """
        Divide curr_prediction feature into four parts along the spatial dimension, and autoregressively predict the next part based on the previous parts.

        The first slice of is conditioned on param, the second slice is conditioned on param and 1st part of curr_prediction, and so on.
        """
        b, c, h, w = curr_prediction.shape
        dtype = curr_prediction.dtype
        device = curr_prediction.device
        m1, m2, m3, m4 = self.generate_checkboard_mask(h, w, dtype, device)
        params1 = param * m1
        hat_so_far = curr_prediction * m1
        params_so_far = params1

        params2 = self.res_slice(torch.cat([hat_so_far, param], dim=1)) * m2
        hat_so_far = hat_so_far + curr_prediction * m2
        params_so_far = params_so_far + params2

        params3 = self.res_slice(torch.cat([hat_so_far, param], dim=1)) * m3
        hat_so_far = hat_so_far + curr_prediction * m3
        params_so_far = params_so_far + params3

        params4 = self.res_slice(torch.cat([hat_so_far, param], dim=1)) * m4
        params_so_far = params_so_far + params4

        return params_so_far

    def process_with_mask_wo_quant(self, y, scales, means, mask):
        scales_hat = scales * mask
        means_hat = means * mask

        y_res = (y - means_hat) * mask
        y_q = y_res
        y_hat = y_q + means_hat

        return y_res, y_q, y_hat, scales_hat

    def masked_idx_quantize(self, x, mask, return_soft_idx=False):
        """
        x: [B, C, H, W]
        mask: [B, 1, H, W] (0/1)
        只在 mask=1 的位置查码表
        """
        B, C, H, W = x.shape
        mask = repeat(mask, "c h w -> b c h w", b=B)

        mask_flat = mask.view(B, -1)  # [B, HW]
        x_flat = x.permute(0, 2, 3, 1).reshape(B, -1, C).contiguous()  # [B, HW, C]

        # 取出 mask=1 的位置
        active_idx = mask_flat.nonzero(as_tuple=False)  # [N, 2], (batch_id, position_id)
        x_active = x_flat[active_idx[:, 0], active_idx[:, 1]]  # [N, C]

        # ---- 调用原始 quantize ----
        logger.debug("Gaussian dim: %s", self.gaussian_dim)
        z_q, losses, info, soft_idx = self.idx_quantize(x_active, return_soft_idx=return_soft_idx, gaussian_dim=self.gaussian_dim)

        z_q = z_q.squeeze(0).squeeze(-1).permute(1, 0)  # [N, C]

        # ---- scatter 回原始空间 ----
        z_q_full = torch.zeros_like(x_flat)
        z_q_full[active_idx[:, 0], active_idx[:, 1]] = z_q
        z_q_full = z_q_full.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        indices_full = torch.zeros(B, H*W).long().to(z_q.device)
        indices_full[active_idx[:,0], active_idx[:, 1]] = info[-1]
        indices_full = indices_full.view(B, H, W).contiguous()

        soft_idx_full = None
        if return_soft_idx and soft_idx is not None:
            soft_idx_full = torch.zeros(B, H * W, device=x.device, dtype=soft_idx.dtype)
            soft_idx_full[active_idx[:, 0], active_idx[:, 1]] = soft_idx
            soft_idx_full = soft_idx_full.view(B, H, W)  # [B, H, W]

        return z_q_full, losses, indices_full, soft_idx_full

    def build_2d_sincos_position_embedding(self, H, W, d_model, device, normalize=False):
        """
        构造 2D 正余弦位置编码
        返回: [1, H*W, d_model]
        - normalize=True 时，使用归一化坐标 (0~1)，保证多尺度可对齐
        - normalize=False 时，使用原始坐标 (0~H-1, 0~W-1)
        """
        if d_model % 4 != 0:
            raise ValueError("d_model 必须能被4整除，才能均分到 x/y 的 sin/cos 上")

        # 位置坐标
        y_embed = torch.arange(H, dtype=torch.float32, device=device)
        x_embed = torch.arange(W, dtype=torch.float32, device=device)

        # 归一化坐标，保证不同分辨率可对齐
        if normalize:
            y_embed = y_embed / (H - 1) if H > 1 else y_embed
            x_embed = x_embed / (W - 1) if W > 1 else x_embed

        # 网格
        yy, xx = torch.meshgrid(y_embed, x_embed, indexing="ij")  # [H, W]

        # 频率尺度
        dim_t = torch.arange(d_model // 4, dtype=torch.float32, device=device)
        dim_t = 10000 ** (2 * (dim_t // 2) / (d_model // 2))  # [d_model/4]

        # H方向
        pos_y = yy[:, :, None] / dim_t  # [H, W, d_model/4]
        pos_x = xx[:, :, None] / dim_t

        pos_y = torch.stack((pos_y.sin(), pos_y.cos()), dim=-1)  # [H, W, d_model/4, 2]
        pos_x = torch.stack((pos_x.sin(), pos_x.cos()), dim=-1)

        pos_y = pos_y.flatten(-2)  # [H, W, d_model/2]
        pos_x = pos_x.flatten(-2)

        pos = torch.cat((pos_y, pos_x), dim=-1)  # [H, W, d_model]
        pos = pos.reshape(1, H * W, d_model)     # [1, L, d_model]
        return pos

    def build_2d_rope_position_embedding(self, H, W, d_model, device, base=10000, normalize=False, scale_factor=1000.0):
        """
        新增参数 scale_factor: 归一化后的坐标会乘以这个因子
        """
        if d_model % 2 != 0:
            raise ValueError("d_model 必须能被2整除")

        y_embed = torch.arange(H, dtype=torch.float32, device=device)
        x_embed = torch.arange(W, dtype=torch.float32, device=device)

        if normalize:
            y_embed = y_embed / (H - 1) if H > 1 else y_embed
            x_embed = x_embed / (W - 1) if W > 1 else x_embed
            # 应用缩放因子
            y_embed = y_embed * scale_factor
            x_embed = x_embed * scale_factor

        yy, xx = torch.meshgrid(y_embed, x_embed, indexing="ij")

        dim_t = torch.arange(d_model // 2, dtype=torch.float32, device=device)
        dim_t = base ** (2 * (dim_t // 2) / (d_model // 2))

        pos_y = yy[:, :, None] / dim_t
        pos_x = xx[:, :, None] / dim_t

        sin_y = torch.sin(pos_y)
        cos_y = torch.cos(pos_y)
        sin_x = torch.sin(pos_x)
        cos_x = torch.cos(pos_x)

        sin = torch.stack((sin_y, sin_x), dim=-1).flatten(start_dim=-2)
        cos = torch.stack((cos_y, cos_x), dim=-1).flatten(start_dim=-2)

        sin = sin.reshape(1, H * W, d_model)
        cos = cos.reshape(1, H * W, d_model)

        return sin, cos

    def apply_rotary_emb(self, xq, xk, sin, cos):
        """
        应用旋转位置编码到 Query 和 Key 上
        xq, xk: [batch_size, seq_len, num_heads, head_dim]
        sin, cos: [1, seq_len, head_dim] (通过 reshape 或 expand 匹配 xq/xk 的 head_dim)
        """
        # 旋转一半的维度，另一半不变
        xq_roped = xq * cos + self.rotate_half(xq) * sin
        xk_roped = xk * cos + self.rotate_half(xk) * sin
        return xq_roped, xk_roped

    def rotate_half(self, x):
        """
        将输入张量的后一半维度旋转，实现 [-x2, x1] 的效果
        x: [..., d]
        """
        d = x.shape[-1]
        x1 = x[..., : d//2]
        x2 = x[..., d//2 :]
        return torch.cat((-x2, x1), dim=-1)

    def slice_prediction_mean_scale(self, means_formal, scales_formal, curr_prediction):
        """
        Divide curr_prediction feature into four parts along the spatial dimension,
        and autoregressively predict the next part based on the previous parts.
        """
        b, c, h, w = curr_prediction.shape
        dtype = curr_prediction.dtype
        device = curr_prediction.device
        mask_list = self.generate_checkboard_mask(h, w, dtype, device)

        means, scales = means_formal, scales_formal
        total_vq_loss, total_commit_loss, total_cb_loss, total_usage = 0, 0, 0, 0
        total_soft_idx = torch.zeros(b, h, w, device=device)  # 聚合 soft_idx
        total_indices = torch.zeros(b, h, w, device=device).long()

        length_mask = len(mask_list)
        for i in range(length_mask):
            means_i = means * mask_list[i]
            scales_i = scales * mask_list[i]
            y_res_i = (curr_prediction - means_i) * mask_list[i]
            y_q, (vq_loss, commit_loss, cb_loss, usage), indices_full, soft_idx_full = \
                self.masked_idx_quantize(y_res_i, mask_list[i], return_soft_idx=True)

            total_vq_loss += vq_loss
            total_commit_loss += commit_loss
            total_cb_loss += cb_loss
            total_usage += usage
            total_indices += indices_full

            if soft_idx_full is not None:
                total_soft_idx += soft_idx_full  # 聚合 soft_idx

            if i == 0:
                means_so_far = means_i
                scales_so_far = scales_i
                y_res_so_far = y_res_i
                y_q_res_so_far = y_q
            else:
                means_so_far += means_i
                scales_so_far += scales_i
                y_res_so_far += y_res_i
                y_q_res_so_far += y_q
            if i<length_mask-1:
                means = self.mean_predictor(
                    self.spatial_adaptor_mean[i](torch.cat((means_formal, means_so_far, y_q_res_so_far.detach()+means_so_far), dim=1))
                )
                scales = self.scale_predictor(
                    self.spatial_adaptor_scale[i](torch.cat((scales_formal, scales_so_far, y_q_res_so_far.detach()+means_so_far), dim=1))
                )

        return [y_q_res_so_far, y_res_so_far , scales_so_far, means_so_far], \
            [total_vq_loss/length_mask, total_commit_loss/length_mask, 0, total_usage/length_mask], \
            [[None, None, total_indices], total_soft_idx]

    def chunk_distance_cal(self, h, code_book, h_norm, codebook_norm):
        """
        Calculate the distance in chunks to avoid OOM.
        h: (B, N, C), code_book: (V, C)
        h_norm: (B, N, 1), codebook_norm: (1, 1, V)
        """
        d_chunks = []
        for i in range(0, self.V_size, self.chunk_size):
            code_chunk = code_book[i:i+self.chunk_size, :]  # (V_chunk, C)
            inner_chunk = torch.einsum('bnc,vc->bnv', h, code_chunk)
            d_chunk = h_norm + codebook_norm[:, :, i:i+self.chunk_size] - 2 * inner_chunk
            d_chunks.append(d_chunk)
        return torch.cat(d_chunks, dim=2).contiguous()

    def predict_soft_probs(self, input, beta = 1):
        code_book = self.idx_quantize.embedding.weight.data
        B, C, H, W = input.shape
        input = input.permute(0, 2, 3, 1).contiguous().view(B, -1, C)
        if self.l2_norm:
            input = F.normalize(input, p=2, dim=-1)
            code_book = F.normalize(code_book, p=2, dim=-1)  # code_book: (C, V)

        # Step 2: ||code_book||^2 -> (1, 1, V)
        codebook_norm = torch.sum(code_book ** 2, dim=1)  # (V,)
        codebook_norm = codebook_norm.view(1, 1, self.V_size)       # (1, 1, V)

        h_norm = torch.sum(input ** 2, dim=2, keepdim=True)  # (B, N, 1)
        # # Step 4: Combine
        # d = h_norm + codebook_norm - 2 * inner_product  # (B, N, V)
        d = self.chunk_distance_cal(input, code_book, h_norm, codebook_norm)  # (B, N, V)
        probs = F.softmax(-beta * d, dim=-1) #(B, N, V)
        # soft_idx_pred = torch.sum(probs * indices, dim=-1)   # soft expectation, (B,N)
        # soft_idx_pred = soft_idx_pred.view(B,1, H, W)
        return probs.view(B, H, W, -1)

    def js_divergence(self, p, q, eps=1e-8, bidirection=True, reduction="batchmean"):
        """
        p, q: [B, K] 概率分布 (已经 softmax 过)
        eps: 避免 log(0)
        bidirection: True 用标准 JS 散度, False 用 KL(p||q)
        reduction: "none", "mean", "batchmean"
        """
        if bidirection:
            m = 0.5 * (p + q)
            js = 0.5 * torch.sum(p * (torch.log(p + eps) - torch.log(m + eps)), dim=-1) + \
                0.5 * torch.sum(q * (torch.log(q + eps) - torch.log(m + eps)), dim=-1)
            if reduction == "mean":
                return js.mean()
            elif reduction == "batchmean":
                return js.sum() / p.size(0)
            else:
                return js  # [B]
        else:
            # KL(P || Q) 注意 input 要是 log Q
            log_q = torch.log(q + eps)
            kl = F.kl_div(input=log_q, target=p, reduction=reduction)
            return kl

    def slice_prediction_woAR_quantize(self, means_formal, y):
        """
        Divide curr_prediction feature into four parts along the spatial dimension,
        and autoregressively predict the next part based on the previous parts.
        """
        y_q_restored, emb_loss_res, info_res, additional_return = self.idx_quantize(y, return_soft_idx=True, return_cons_loss=True, gaussian_dim=self.gaussian_dim)
        soft_idx, cons_loss, probs_gt = additional_return
        y_q_in = y_q_restored
        b, c, h, w = y_q_in.shape
        dtype = y_q_in.dtype
        device = y_q_in.device
        mask_list = self.generate_checkboard_mask(h, w, dtype, device)

        means = means_formal

        length_mask = len(mask_list)
        for i in range(length_mask):
            means_i = means * mask_list[i]
            y_q_i = y_q_in * mask_list[i]

            if i == 0:
                means_so_far = means_i
                y_q_so_far = y_q_i
            else:
                means_so_far += means_i
                y_q_so_far += y_q_i
            if i<length_mask-1:
                means = self.feature_extractor(
                    self.spatial_adaptor[i](torch.cat((means_formal, means_so_far, y_q_so_far+means_so_far), dim=1))
                )

        probs = self.predict_soft_probs(means_so_far) #(B, H, W, V)
        B,H,W,V = probs.shape
        JsKl_loss = self.js_divergence(probs_gt, probs.view(B*H*W, -1), bidirection=False)

        if self.gaussian_dim==1:
            indices = torch.arange(self.V_size, device=device, dtype=dtype) #(V)
            means_final  = self.mean_predictor(probs*indices) #(B, H, W, num_gaussion)
            scales_final = self.scale_predictor(probs) #(B, H, W, num_gaussion)
        else:
            height = width = int(sqrt(V))
            indices_x = torch.arange(height, device=device, dtype=dtype) #(V)
            indices_y = torch.arange(width, device=device, dtype=dtype) #(V)
            probs_new = probs.view(B,H,W,height,width)
            probs_x = probs_new.sum(dim=-1) #(B, H, W, height)
            probs_y = probs_new.sum(dim=-2)
            means_final_x = self.mean_predictor(probs_x*indices_x) #(B, H, W, num_gau*gau_dim)
            means_final_y = self.mean_predictor(probs_y*indices_y)
            scales_finel_x = self.scale_predictor(probs_x)
            scales_finel_y = self.scale_predictor(probs_y)
            means_final = (means_final_x, means_final_y)
            scales_final = (scales_finel_x, scales_finel_y)

        if self.num_gaussian>1:
            weights = self.weight_predictor(probs) #(B, H, W, num_gaussion)
            weights = F.softmax(weights, dim=-1)
        else:
            weights = torch.ones_like(B,H,W,1) #(B, H, W, 1)
        if self.gaussian_dim == 1:
            soft_idx = soft_idx.view(b, h, w, 1)
        else:
            soft_idx_x = soft_idx[0].view(b, h, w, 1)
            soft_idx_y = soft_idx[1].view(b, h, w, 1)
            soft_idx = (soft_idx_x, soft_idx_y)
        return [means_final, scales_final, weights], [y_q_restored, emb_loss_res, info_res, soft_idx, cons_loss, JsKl_loss]

    @torch.no_grad()
    def _entropy_code_real_ans(self, *args, **kwargs):
        """Removed legacy fast/debug entropy path.

        Actual real bitstream coding is implemented by
        ``tokenizer.tokenizer_image.codec.real.SimpleRealCodec`` so model files
        do not depend on rANS or stream-merge internals.
        """

        raise NotImplementedError(
            "The legacy model.entropy_code_ans fast/debug path was removed. "
            "Use tokenizer.tokenizer_image.codec.real.SimpleRealCodec for real bitstream coding."
        )
