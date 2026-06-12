import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt, log
from tokenizer.tokenizer_image.entropy import VQ_AR_Predictor


def compute_entropy_loss(affinity, loss_type="softmax", temperature=0.01, Cd_entropy_estnet=None, quant=None, training=False, updated_num_patches=None, get_cd_loss = False, mask_padded=None):
    flat_affinity = affinity.reshape(-1, affinity.shape[-1])
    flat_affinity /= temperature
    probs = F.softmax(flat_affinity, dim=-1)
    log_probs = F.log_softmax(flat_affinity + 1e-5, dim=-1)/log(2.0)
    if loss_type == "softmax":
        target_probs = probs
    else:
        raise ValueError("Entropy loss {} not supported".format(loss_type))
    sample_entropy = - torch.mean(torch.sum(target_probs * log_probs, dim=-1))

    # target_probs = target_probs.view(b,h,w,-1)
    avg_probs = torch.mean(target_probs, dim=0)

    avg_entropy = - torch.mean(torch.sum(avg_probs * torch.log(avg_probs + 1e-6)/log(2.0), dim=-1))

    ont_hot_gt = F.one_hot(flat_affinity.argmax(dim=-1), num_classes=affinity.shape[-1])
    # print(f"ont_hot_gt.shape:{ont_hot_gt.shape}, affinity.shape:{affinity.shape}, log_probs.shape:{log_probs.shape}")
    one_hot_sample_entropy = - torch.mean(torch.sum(ont_hot_gt * log_probs, dim=-1))

    if Cd_entropy_estnet is not None and get_cd_loss:
        # print(f"z_q.shape{quant.shape}, net:{Cd_entropy_estnet}")
        if training:
            gt_index = probs
        else:
            gt_index = affinity.argmax(dim=-1)
        cd_entropy, logits = Cd_entropy_estnet(quant=quant, gt_index=gt_index, updated_shape=updated_num_patches)
       
        if isinstance(cd_entropy, torch.Tensor):
            num_points = 1
            for dim in cd_entropy.shape:
                num_points = num_points * dim
            if mask_padded is not None:
                cd_entropy = cd_entropy * (1-mask_padded)
            cd_entropy = cd_entropy.sum()
        elif isinstance(cd_entropy, dict):
            num_points = sum(pn[0]*pn[1] for pn in updated_num_patches)
            cd_entropy = sum(item for _, item in cd_entropy.items())
    else:
        cd_entropy = 0.
        num_points = 1
        logits = None
        
    loss = cd_entropy/num_points
    return loss, sample_entropy, avg_entropy, one_hot_sample_entropy, cd_entropy, logits
    
class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta, entropy_loss_ratio, l2_norm, show_usage, init_larger_uniform=False, use_SIMVQ=False, use_predictor=False):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.l2_norm = l2_norm
        self.show_usage = show_usage
        self.use_SIMVQ = use_SIMVQ
        self.entropy_loss_ratio = entropy_loss_ratio
        print(f"Initial codebook with size:{n_e}, dim:{e_dim}, init with larger uniform:{init_larger_uniform}, use SIMVQ:{use_SIMVQ}")

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        if use_predictor:
            self.condition_entropy_small = VQ_AR_Predictor(in_channels=e_dim, V_size=n_e, d_model=384, nhead=8, num_layers=6, temperature=1., num_ar_per_scale=4)
        else:
            self.condition_entropy_small = None
        if not init_larger_uniform:
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        else:
            scales = torch.linspace(1e-8, 2 / n_e, n_e).unsqueeze(1)  # [n_e, 1]
            init_weights = torch.empty(n_e, e_dim).uniform_(-1.0, 1.0) * scales

            self.embedding.weight.data.copy_(init_weights)

        if self.use_SIMVQ:
            self.embedding_proj = nn.Linear(self.e_dim, self.e_dim)

        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        if self.l2_norm:
            self.embedding.weight.data = F.normalize(self.embedding.weight.data, p=2, dim=-1)
        if self.show_usage:
            self.register_buffer("codebook_used", nn.Parameter(torch.zeros(65536)))

    
    def chunk_distance_cal_flatten(self, z_flattened, embedding, chunk_size=1024):
        """
        Args:
            z_flattened: (B*N, D)  # 展平后的输入特征
            embedding: (V, D)      # codebook
            chunk_size: 每次处理的 codebook 块大小
        Returns:
            d: (B*N, V)  # 距离矩阵
        """
        # ||z||^2 部分
        # print("Type of z_flattened: ",type(z_flattened))
        z_norm = torch.sum(z_flattened ** 2, dim=1, keepdim=True)  # (B*N, 1)

        # ||e||^2 部分
        codebook_norm = torch.sum(embedding ** 2, dim=1, keepdim=True).t()  # (1, V)

        d_chunks = []
        V_size = embedding.shape[0]
        for i in range(0, V_size, chunk_size):
            e_chunk = embedding[i:i+chunk_size]  # (V_chunk, D)
            # 内积部分
            inner = torch.matmul(z_flattened, e_chunk.t())  # (B*N, V_chunk)
            # 完整距离
            d_chunk = z_norm + codebook_norm[:, i:i+chunk_size] - 2 * inner
            d_chunks.append(d_chunk)

        return torch.cat(d_chunks, dim=1).contiguous()  # (B*N, V)

    def soft_argmin(self, d, beta=1., gaussian_dim=1):
        # d: [B, N]
        B, N = d.shape
        if self.training:
            # d: [B, N]
            probs = F.softmax(-beta * d, dim=-1)   # softmin
            if gaussian_dim == 1:
                indices = torch.arange(d.size(-1), device=d.device, dtype=d.dtype)
                soft = torch.sum(probs * indices, dim=-1)   # soft expectation

                # hard:
                #hard_idx = torch.argmin(d, dim=-1)
                hard = hard_idx.to(d.dtype)

                # straight-through: forward = hard, backward = soft
                out = hard.detach() + soft - soft.detach()
                indices = soft.long()
            elif gaussian_dim == 2:
                assert sqrt(self.n_e)%1==0 
                height = width = int(sqrt(self.n_e))
                probs = probs.view(B, height, width)
                probs_x = torch.sum(probs, dim=2) #(B, height)
                probs_y = torch.sum(probs, dim=1) #(B, width)
                indices_x = torch.arange(height, device=d.device, dtype=d.dtype)
                indices_y = torch.arange(width, device=d.device, dtype=d.dtype)
                soft_x = torch.sum(probs_x*indices_x, dim=1) #(B)
                soft_y = torch.sum(probs_y*indices_y, dim=1) #(B)
                hard_idx = torch.argmin(d, dim=-1) #(B)
                hard_x = hard_idx//width
                hard_y = hard_idx%width
                #out_x = hard_x.detach() + soft_x - soft_x.detach()
                #out_y = hard_y.detach() + soft_y - soft_y.detach()
                #out = (out_x, out_y)
                indices = (soft_x*width+soft_y).long()
                out = (soft_x, soft_y)
            else:
                raise ValueError("Only support gaussian dim 1 or 2")
            return out, probs.view(B,-1), indices
        else:
            indices = torch.argmin(d, dim=-1)
            probs = F.softmax(-beta * d, dim=-1)
            # indices = torch.sum(probs * torch.arange(d.size(-1), device=d.device, dtype=d.dtype), dim=-1).long()
            if gaussian_dim == 1:
                return indices, probs, indices
            else:
                height = width = int(sqrt(self.n_e))
                return (indices//width, indices%width), probs, indices
    

    def build_window_weights(self, assign_ids, K, k, sigma=None, mode="1d"):
        """
        assign_ids: [B] 每个样本的 gt index
        K: codebook size
        k: 窗口半径 (1d距离或2d半径)
        sigma: 衰减参数，默认取 k/2
        mode: "1d" 或 "2d"
        return: [B, K] float 权重 (0~1)
        """
        B = assign_ids.size(0)
        device = assign_ids.device

        if sigma is None:
            sigma = k / 2

        if mode == "1d":
            # [1, K]
            all_idx = torch.arange(K, device=device).unsqueeze(0)  
            assign_ids = assign_ids.unsqueeze(1)  # [B, 1]
            distance = torch.abs(all_idx - assign_ids).float()  # [B, K]

        elif mode == "2d":
            side = int(K**0.5)
            assert side * side == K, "K 必须是完全平方数才能 reshape 为 2D"

            # 生成所有位置坐标 [K, 2]
            xx, yy = torch.meshgrid(
                torch.arange(side, device=device),
                torch.arange(side, device=device),
                indexing="ij"
            )
            coords = torch.stack([xx, yy], dim=-1).view(-1, 2)  # [K, 2]

            # gt 的二维坐标 [B, 2]
            gt_xy = torch.stack([assign_ids // side, assign_ids % side], dim=-1).float()  # [B, 2]

            # 计算二维欧式距离 [B, K]
            distance = torch.cdist(gt_xy, coords.float())  

        else:
            raise ValueError("mode must be '1d' or '2d'")

        # 高斯衰减权重
        weights = torch.exp(- (distance ** 2) / (2 * sigma ** 2))

        # 窗口外截断为 0
        weights = weights * (distance <= k)

        return weights  # [B, K]

    def contrastive_window_loss(self, z_normalized, cb_normalized, assign_ids, k=20, temperature=0.07, gaussian_dim=1):
        """
        z_normalized: [B, D]
        cb_normalized: [K, D]
        assign_ids: [B]  每个 z 的 ground-truth codebook index
        k: 窗口大小（gt 左右各 k 个当作正样本）
        """
        B, D = z_normalized.shape
        K, _ = cb_normalized.shape

        # 相似度 [B, K]
        sim = torch.matmul(z_normalized, cb_normalized.T) / temperature

        # 构造正样本 mask
        mode = "1d" if gaussian_dim==1 else "2d"
        weights = self.build_window_weights(assign_ids=assign_ids, K=K, k=k, mode=mode)  # [B, K]
        

        # InfoNCE 多正样本版本
        exp_sim = torch.exp(sim)
        pos_sum = (exp_sim * weights).sum(dim=1)  # 正样本总和
        denom   = exp_sim.sum(dim=1)           # 所有候选

        loss = -torch.log(pos_sum / (denom + 1e-8) + 1e-8).mean()
        return loss
    
    def forward(self, z, return_soft_idx=False, return_cons_loss=False, gaussian_dim=1):
        # reshape z -> (batch, height, width, channel) and flatten
        if z.dim() == 2:  # [N, C]
            z = z.unsqueeze(0).permute(0, 2, 1).unsqueeze(-1)  # [1, C, N, 1]
        z = torch.einsum('b c h w -> b h w c', z).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        if self.l2_norm:
            z = F.normalize(z, p=2, dim=-1)
            z_flattened = F.normalize(z_flattened, p=2, dim=-1)
            
            if self.use_SIMVQ:
                embedding = F.normalize(self.embedding_proj(self.embedding.weight), p=2, dim=-1)
            else:
                embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
        else:
            embedding = self.embedding.weight
            if self.use_SIMVQ:
                embedding = self.embedding_proj(embedding)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(embedding**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, torch.einsum('n d -> d n', embedding))
        
        # d = self.chunk_distance_cal_flatten(z_flattened, embedding)
        additional_return = [None, None, None]
        if return_soft_idx:
            soft_idx, probs, indices = self.soft_argmin(d, gaussian_dim=gaussian_dim)
            # min_encoding_indices = torch.argmin(d, dim=-1)
            # print("soft_idx",type(soft_idx), soft_idx[0].shape, soft_idx[1].shape)
            min_encoding_indices = indices
            additional_return[0] = soft_idx
            additional_return[2] = probs
        else:
            min_encoding_indices = torch.argmin(d, dim=1)

        detach_input=False
        if return_cons_loss:
            if self.l2_norm:
                z_normalized = z_flattened
                cb_normalized = embedding
            else:
                z_normalized = F.normalize(z_flattened, p=2, dim=-1)
                cb_normalized = F.normalize(self.embedding.weight, p=2, dim=-1)
            if detach_input:
                z_normalized = z_normalized.detach()
                # cb_normalized = cb_normalized.detach()
            cnt = 0
            chunk_size=64
            loss_cons = 0
            additional_return[1] = loss_cons

        z_q = embedding[min_encoding_indices].view(z.shape)

        perplexity = None
        min_encodings = None
        vq_loss = None
        commit_loss = None
        entropy_loss = None
        codebook_usage = 0

        if self.show_usage and self.training:
            cur_len = min_encoding_indices.shape[0]
            self.codebook_used[:-cur_len] = self.codebook_used[cur_len:].clone()
            self.codebook_used[-cur_len:] = min_encoding_indices
            codebook_usage = len(torch.unique(self.codebook_used)) / self.n_e

        # compute loss for embedding
        if self.training:
            vq_loss = torch.mean((z_q - z.detach()) ** 2) 
            commit_loss = self.beta * torch.mean((z_q.detach() - z) ** 2) 

            # preserve gradients
            z_q = z + (z_q - z).detach()
            # reshape back to match original input shape
            z_q = torch.einsum('b h w c -> b c h w', z_q)
            gt_total_entropy_loss, sample_entropy, avg_entropy, one_hot_sample_entropy, cd_enteropy = compute_entropy_loss(-d, shape=z.shape, Cd_entropy_estnet=self.condition_entropy_small, quant = z_q, codebook=embedding, training=self.training)
            entropy_loss = self.entropy_loss_ratio * gt_total_entropy_loss
        else:
            # gt_total_entropy_loss, sample_entropy, avg_entropy, one_hot_sample_entropy, cd_enteropy = 0, 0, 0, 0, 0
            # reshape back to match original input shape
            z_q = torch.einsum('b h w c -> b c h w', z_q)
            gt_total_entropy_loss, sample_entropy, avg_entropy, one_hot_sample_entropy, cd_enteropy = compute_entropy_loss(-d, shape=z.shape, Cd_entropy_estnet=self.condition_entropy_small, quant = z_q, codebook=embedding, training=self.training)

        if return_soft_idx or return_cons_loss:
             return z_q, [vq_loss, commit_loss, entropy_loss, codebook_usage, sample_entropy, avg_entropy, one_hot_sample_entropy], [perplexity, min_encodings, min_encoding_indices],  additional_return
        else:
            return z_q, [vq_loss, commit_loss, entropy_loss, codebook_usage, sample_entropy, avg_entropy, one_hot_sample_entropy, cd_enteropy], (perplexity, min_encodings, min_encoding_indices.view(-1, *z_q.shape[2:]))



    def get_codebook_entry(self, indices, shape=None, channel_first=True):
        # shape = (batch, channel, height, width) if channel_first else (batch, height, width, channel)
        if self.l2_norm:
            embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
        else:
            embedding = self.embedding.weight
        z_q = embedding[indices]  # (b*h*w, c)

        if shape is not None:
            if channel_first:
                z_q = z_q.reshape(shape[0], shape[2], shape[3], shape[1])
                # reshape back to match original input shape
                z_q = z_q.permute(0, 3, 1, 2).contiguous()
            else:
                z_q = z_q.view(shape)
        return z_q


class VectorQuantizer_MS_input(VectorQuantizer):
    def __init__(self, n_e, e_dim, beta, entropy_loss_ratio, l2_norm, show_usage, init_larger_uniform=False, use_SIMVQ=False, num_ar_per_scale=4, use_predictor=False, num_layers=6, use_patch_ck_ar=False, temp=0.01):
        super().__init__(n_e, e_dim, beta, entropy_loss_ratio, l2_norm, show_usage, init_larger_uniform=init_larger_uniform, use_SIMVQ=use_SIMVQ, use_predictor=use_predictor)
        self.num_ar_per_scale = num_ar_per_scale
        self.use_patch_ck_ar = use_patch_ck_ar
        self.temp = temp
        print(f"VQ_MS_input: use_predictor:{use_predictor}, num_ar_per_scale:{num_ar_per_scale}, use_patch_ck_ar:{use_patch_ck_ar}, temp:{temp}")
        if use_predictor:
            # self.condition_entropy_small = VQ_AR_Predictor(in_channels=e_dim, V_size=n_e, d_model=640, nhead=8, num_layers=num_layers, ff_dim=1280, temperature=1., num_ar_per_scale=num_ar_per_scale, use_my_transformer=use_my_transformer, l2_norm=False)
            self.condition_entropy_small = VQ_AR_Predictor(in_channels=e_dim, V_size=n_e, d_model=num_layers*64, nhead=8, num_layers=num_layers, temperature=1., num_ar_per_scale=num_ar_per_scale, l2_norm=False, use_patch_ck_ar = self.use_patch_ck_ar)
        else:
            self.condition_entropy_small = None
        
        self.cnt = 0
        self.use_predictor = use_predictor
    
    def forward(self, input_list, enc_wo_l2norm=False, mask_padded=None):
        # reshape z -> (batch, height, width, channel) and flatten
        input_shape = []
        input_all = []
        L = 0 
        for i, z in enumerate(input_list):
            # print(f"i:{i}:", z.shape)
            assert z.ndim == 4
            b,c,h,w = z.shape
            input_shape.append((h, w))
            z = torch.einsum('b c h w -> b h w c', z).contiguous()
            input_all.append(z.view(b,-1, c))
            L+=h*w
        input_all = torch.cat(input_all, dim=1) # (b, L, c)
        B, L, C = input_all.shape
        z_flattened = input_all.view(-1, self.e_dim) # (b*L, C)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        embedding_wo_norm = self.embedding.weight
        if self.l2_norm:
            input_all = F.normalize(input_all, p=2, dim=-1)
            z_flattened = F.normalize(z_flattened, p=2, dim=-1)
            
            if self.use_SIMVQ:
                embedding = F.normalize(self.embedding_proj(self.embedding.weight), p=2, dim=-1)
            else:
                embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
        else:
            embedding = self.embedding.weight
            if self.use_SIMVQ:
                embedding = self.embedding_proj(embedding)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(embedding**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, torch.einsum('n d -> d n', embedding))
        # d = self.chunk_distance_cal_flatten(z_flattened, e
        # mbedding)
        min_encoding_indices = torch.argmin(d, dim=1).view(b, L) #(B*L)
        # print("min_encoding_indices:",min_encoding_indices.shape)

        # print(f"min_encoding_indices:",min_encoding_indices.view(-1)[:30])
        if enc_wo_l2norm:
            z_q = embedding_wo_norm[min_encoding_indices].view(b, L, c)
        else:
            z_q = embedding[min_encoding_indices].view(b, L, c)

        perplexity = None
        min_encodings = None
        vq_loss = None
        commit_loss = None
        entropy_loss = None
        codebook_usage = 0

        if self.show_usage and self.training:
            cur_len = b*L
            self.codebook_used[:-cur_len] = self.codebook_used[cur_len:].clone()
            self.codebook_used[-cur_len:] = min_encoding_indices.view(-1)
            codebook_usage = len(torch.unique(self.codebook_used)) / self.n_e

        # compute loss for embedding
        vq_loss = torch.mean((z_q - input_all.detach()) ** 2) 
        commit_loss = self.beta * torch.mean((z_q.detach() - input_all) ** 2) 

        # preserve gradients
        if self.training:
            z_q = input_all + (z_q - input_all).detach()
        # reshape back to match original input shape
        # z_q = torch.einsum('b h w c -> b c h w', z_q)
        with torch.amp.autocast("cuda", enabled=False):
            gt_total_entropy_loss, sample_entropy, avg_entropy, one_hot_sample_entropy, cd_entropy, logits = compute_entropy_loss(-d, Cd_entropy_estnet=self.condition_entropy_small, quant = z_q, training=self.training, updated_num_patches=input_shape, get_cd_loss=self.use_predictor, mask_padded=mask_padded, temperature=self.temp)
            entropy_loss = self.entropy_loss_ratio * gt_total_entropy_loss

        quant_all = []
        start = 0

        # self.embedding_noemed = embedding
        # self.z_q = z_q #embedding[min_encoding_indices]
        # self.indices = min_encoding_indices

        z_q = torch.einsum('b l c -> b c l', z_q)
        
        for pn in input_shape:
            end = start + pn[0]*pn[1]
            quant_all.append(z_q[:, :,start:end].view(b, c, pn[0], pn[1]))
            start = end
        return quant_all, [vq_loss, commit_loss, entropy_loss, codebook_usage, sample_entropy, avg_entropy, one_hot_sample_entropy, cd_entropy], (logits, perplexity, min_encodings, min_encoding_indices)
