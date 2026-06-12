# Modified from:
#   VQGAN:    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/transformer/mingpt.py
#   DiT:      https://github.com/facebookresearch/DiT/blob/main/models.py  
#   nanoGPT:  https://github.com/karpathy/nanoGPT/blob/master/model.py
#   llama:    https://github.com/facebookresearch/llama/blob/main/llama/model.py
#   gpt-fast: https://github.com/pytorch-labs/gpt-fast/blob/main/model.py
#   PixArt:   https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
from dataclasses import dataclass
from typing import Optional, List
import torch._dynamo as dynamo

import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.drop_path import DropPath
from autoregressive.models.mask_generation import generate_MS_ck_mask, rerank_input
import math

def find_multiple(n: int, k: int):
    if n % k == 0:
        return n
    return n + k - (n % k)

def cross_entropy_log2(input, target, reduction="mean"):
        """
        log2 cross entropy loss
        """
        # if not self.training:
        #     log_probs = self.log_probs_with_top_k_top_p_(input.view(origin_shape), top_k=3600, top_p=0.95, num_samples_portion=0.1)
        #     print("log probs:",log_probs)
        #     print(log_probs.shape, origin_shape)
        # else:
        # probs = F.softmax(input/self.temperature, dim=-1)
        # log_probs = torch.log(probs + 1e-8) / torch.log(torch.tensor(2.0))  # 换底公式
        log_probs = F.log_softmax(input, dim=-1) / math.log(2.0)
        # print("log probs:",log_probs)
        
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
        
@dataclass
class ModelArgs:
    dim: int = 4096
    n_layer: int = 32
    n_head: int = 32
    n_kv_head: Optional[int] = None
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    rope_base: float = 1000
    norm_eps: float = 1e-5
    initializer_range: float = 0.02
    
    token_dropout_p: float = 0.1
    attn_dropout_p: float = 0.0
    resid_dropout_p: float = 0.1
    ffn_dropout_p: float = 0.1
    drop_path_rate: float = 0.0

    num_classes: int = 1000
    caption_dim: int = 2048
    class_dropout_prob: float = 0.1
    model_type: str = 'c2i'

    vocab_size: int = 16384
    cls_token_num: int = 1
    block_size: int = 256
    max_batch_size: int = 32
    max_seq_len: int = 2048
    use_MS_ck_ar: bool = False 
    cos_attn: bool = True
    use_alibi_bias: bool = False


#################################################################################
#                      Embedding Layers for Class Labels                        #
#################################################################################
class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels).unsqueeze(1)
        return embeddings


#################################################################################
#                      Embedding Layers for Text Feature                        #
#################################################################################
class CaptionEmbedder(nn.Module):
    """
    Embeds text caption into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, in_channels, hidden_size, uncond_prob, token_num=120):
        super().__init__()
        self.cap_proj = MLP(in_features=in_channels, hidden_features=hidden_size, out_features=hidden_size)
        self.register_buffer("uncond_embedding", nn.Parameter(torch.randn(token_num, in_channels) / in_channels ** 0.5))
        self.uncond_prob = uncond_prob

    def token_drop(self, caption, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(caption.shape[0], device=caption.device) < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        caption = torch.where(drop_ids[:, None, None], self.uncond_embedding, caption)
        return caption

    def forward(self, caption, train, force_drop_ids=None):
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            caption = self.token_drop(caption, force_drop_ids)
        embeddings = self.cap_proj(caption)
        return embeddings


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


#################################################################################
#                                  GPT Model                                    #
#################################################################################
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


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        hidden_dim = 4 * config.dim
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if config.ffn_dim_multiplier is not None:
            hidden_dim = int(config.ffn_dim_multiplier * hidden_dim)
        hidden_dim = find_multiple(hidden_dim, config.multiple_of)

        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)
        self.ffn_dropout = nn.Dropout(config.ffn_dropout_p)

    def forward(self, x):
        return self.ffn_dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_head, head_dim, dtype):
        super().__init__()
        cache_shape = (max_batch_size, n_head, max_seq_length, head_dim)
        # print(f"cache shape:", cache_shape)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        # assert input_pos.shape[1] == k_val.shape[2]
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0
        self.dim = config.dim
        self.head_dim = config.dim // config.n_head
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head if config.n_kv_head is not None else config.n_head
        total_kv_dim = (self.n_head + 2 * self.n_kv_head) * self.head_dim

        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_kv_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None
        self.cos_attn = config.cos_attn

        if self.cos_attn:
            self.scale = 1
            size = (1, self.n_head, 1, 1)
            # size: 11H1 or 1H11
            self.scale_mul_1H11 = nn.Parameter(torch.full(size=size, fill_value=4.0).log(), requires_grad=True)
            self.max_scale_mul = torch.log(torch.tensor(100)).item()
        else:
            self.scale = 1 / math.sqrt(self.head_dim) / self.tau

        # regularization
        self.attn_dropout_p = config.attn_dropout_p
        self.resid_dropout = nn.Dropout(config.resid_dropout_p)

    def attention_forward(self, q, k, v, mask):
        return F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=mask, 
            is_causal=True if mask is None else False,
            dropout_p=self.attn_dropout_p if self.training else 0,
            scale=self.scale
        )
    
    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor = None, 
        input_pos: Optional[torch.Tensor] = None, 
        mask: Optional[torch.Tensor] = None,
        alibi_bias: Optional[torch.Tensor] = None
    ):
        bsz, seqlen, _ = x.shape
        kv_size = self.n_kv_head * self.head_dim
        xq, xk, xv = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        xq = apply_rotary_emb(xq, freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis)

        xq, xk, xv = map(lambda x: x.transpose(1, 2), (xq, xk, xv))

        if self.kv_cache is not None:
            keys, values = self.kv_cache.update(input_pos, xk, xv)
        else:
            keys, values = xk, xv
        keys = keys.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        values = values.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        
        # print("Attention xq:", xq.shape, "keys:", keys.shape, "values:", values.shape)
        # attn = xq@keys.transpose(-2,-1)/math.sqrt(self.head_dim)
        # print(attn.min(), attn.max(), attn.mean(), attn.abs().mean())

        q, k, v = xq, keys, values
        if self.cos_attn:   # always True
            scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp() # 11H1 (flash), or 1H11 (not flash)
            q = F.normalize(q, dim=-1, eps=1e-12).mul(scale_mul).contiguous()   # fp32
            k = F.normalize(k, dim=-1, eps=1e-12).contiguous()                  # fp32
            v = v.contiguous()                                                  # bf16
        else:   # be contiguous, to make kernel happy
            q = q.contiguous()      # bf16
            k = k.contiguous()      # bf16
            v = v.contiguous()      # bf16

        if alibi_bias is not None:
            mask = mask.unsqueeze(0).unsqueeze(0) + alibi_bias.unsqueeze(0)
        mask = mask.to(q.device)
            
        # output = F.scaled_dot_product_attention(
        #     q, k, v, 
        #     attn_mask=mask, 
        #     is_causal=True if mask is None else False, # is_causal=False is for KV cache
        #     dropout_p=self.attn_dropout_p if self.training else 0,
        #     scale=self.scale
        # )
        output = self.attention_forward(q, k, v, mask)

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        output = self.resid_dropout(self.wo(output))
        return output


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs, drop_path: float):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, start_pos: int, mask: Optional[torch.Tensor] = None, alibi_bias=None):
        h = x + self.drop_path(self.attention(self.attention_norm(x), freqs_cis, start_pos, mask, alibi_bias=alibi_bias))
        out = h + self.drop_path(self.feed_forward(self.ffn_norm(h)))
        return out


class Transformer(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layer = config.n_layer
        self.block_size = config.block_size
        self.num_classes = config.num_classes
        self.model_type = config.model_type
        self.cls_token_num = config.cls_token_num
        if self.model_type == 'c2i':
            self.cls_embedding = LabelEmbedder(config.num_classes, config.dim, config.class_dropout_prob)
        elif self.model_type == 't2i':
            self.cls_embedding = CaptionEmbedder(config.caption_dim, config.dim, config.class_dropout_prob)
        else:
            raise Exception("please check model type")
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.tok_dropout = nn.Dropout(config.token_dropout_p)

        # transformer blocks
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.n_layer)]
        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layer):
            self.layers.append(TransformerBlock(config, dpr[layer_id]))

        # output layer
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # 2d rotary pos embedding
        # grid_size = int(self.block_size ** 0.5)
        # assert grid_size * grid_size == self.block_size
        # self.freqs_cis = precompute_freqs_cis_2d(grid_size, self.config.dim // self.config.n_head, self.config.rope_base, self.cls_token_num)

        self.head_dim = self.config.dim // self.config.n_head
        print(f"Using MS_ck_ar?:", config.use_MS_ck_ar)
        if not config.use_MS_ck_ar:
            #mask_all_training = torch.arange(4*4+8*8+16*16)
            shape_list = [(1,1), (4,4), (8,8), (16,16)]
        else:
            shape_list = [(4,4), (8,8), (16,16)]
        self.build_mask_and_freq_cis(shape_list)

        ##Alibi positional bias
        
        # self.register_buffer("alibi_slopes", torch.tensor(slopes).unsqueeze(1).unsqueeze(1))

        # attn_bias, mask_all_training = generate_MS_ck_mask(input_shape=shape_list, device="cpu", use_patch_ck_ar=self.config.use_MS_ck_ar)
        # # print(f"mask_all_training.shape:",mask_all_training.shape)
        # self.freqs_cis = precompute_freqs_MS_2d(shape_list=shape_list, head_dim=self.head_dim, base=self.config.rope_base, mask_all=mask_all_training, use_patch_ck_ar=self.config.use_MS_ck_ar)
        # self.attn_bias = attn_bias
        # self.mask_all = mask_all_training
        # print(attn_bias, mask_all_training)
        
        # KVCache
        self.max_batch_size = -1
        self.max_seq_length = -1

        self.initialize_weights()

    def initialize_weights(self):        
        # Initialize nn.Linear and nn.Embedding
        self.apply(self._init_weights)

        # Zero-out output layers:
        nn.init.constant_(self.output.weight, 0)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def build_mask_and_freq_cis(self, shape_list, k=4):
        device = next(self.parameters()).device
        attn_bias, mask_all_training = generate_MS_ck_mask(input_shape=shape_list, device=device, use_patch_ck_ar=self.config.use_MS_ck_ar, k=k)
        # print(f"mask_all_training.shape:",mask_all_training.shape)
        self.freqs_cis = precompute_freqs_MS_2d(shape_list=shape_list, head_dim=self.head_dim, base=self.config.rope_base, mask_all=mask_all_training, use_patch_ck_ar=self.config.use_MS_ck_ar)
        self.attn_bias = attn_bias.cpu()
        self.mask_all = mask_all_training
        start_idx = [0] + [k*2**i for i in range(len(shape_list)-1)]
        self.start_idx = [sum(start_idx[:i+1]) for i in range(len(start_idx))]
        self.each_scale_num = [0] + [pn[0]*pn[1] for pn in shape_list]

        del attn_bias, mask_all_training
        self.freqs_cis = self.freqs_cis.to(device)
        # self.attn_bias = self.attn_bias.to(device)
        self.mask_all = self.mask_all.to(device)
        if self.config.use_alibi_bias:
            self.alibi_bias = self.build_alibi_for_all_slices(shape_list, device=device).cpu()
        else:
            self.alibi_bias = None

    def build_alibi_for_all_slices(self, shape_list, device):
        ## extract bias of largest scale
        H, W = shape_list[-1]
        slopes = [2 ** (-8 * i / self.config.n_head) for i in range(self.config.n_head)]
        slopes = torch.tensor(slopes).unsqueeze(1).unsqueeze(1).to(device)
        y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        coords = torch.stack([x, y], dim=-1).to(device).permute(2, 0, 1).unsqueeze(0).float()  # (H, W, 2)
        ## interpolate to other scales
        alibi_bias_all = [coords]
        for i in range(len(shape_list)-1):
            h, w = shape_list[i]
            coords = F.interpolate(alibi_bias_all[-1], size=(h, w), mode='bilinear')[0]
            alibi_bias_all.insert(0, coords)
        for i in range(len(alibi_bias_all)):
            alibi_bias_all[i] = alibi_bias_all[i].squeeze(0).permute(1, 2, 0).reshape(-1, 2)
        alibi_bias_all = torch.cat(alibi_bias_all, dim=0)
        diff = alibi_bias_all[:, None, :] - alibi_bias_all[None, :, :]  # (L, L, 2)
        dist = diff.abs().sum(-1).float() #(L,L)
        alibi_bias = -slopes * dist.unsqueeze(0) #(H, L, L)
        return alibi_bias.contiguous()
    
    def setup_caches(self, max_batch_size=1, shape_list=[(4,4), (8,8), (16,16)], dtype=torch.float32):
        # if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
        #     return
        # max_seq_length = find_multiple(max_seq_length, 8)
        max_seq_length = sum([pn[0]*pn[1] for pn in shape_list])
        self.shape_list = shape_list
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        self.build_mask_and_freq_cis(shape_list)
        for b in self.layers:
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_head, self.head_dim, dtype)

    def destroy_caches(self):
        for b in self.layers:
            b.attention.kv_cache = None

    def forward(
        self, 
        idx: torch.Tensor, 
        cond_idx: torch.Tensor,  # cond_idx_or_embed, [B, 1]
        input_pos:  Optional[torch.Tensor] = None, 
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
        shape_list = None
    ):
        device = next(self.parameters()).device
        ## decide to utilize existing mask, freq_cis or dynamic generation
        if self.training and (shape_list[-1]==(16,16)):
            mask = self.attn_bias.to(device)
            # freqs_cis = self.freqs_cis[:token_embeddings.shape[1]]
            freqs_cis = self.freqs_cis.to(device)
            mask_all = self.mask_all
            if self.config.use_alibi_bias:
                alibi_bias = self.alibi_bias.to(device=device)
            else:
                alibi_bias = None
        else:
            assert shape_list is not None
            mask, mask_all_training = generate_MS_ck_mask(input_shape=shape_list, device=device, use_patch_ck_ar=self.config.use_MS_ck_ar)
            freqs_cis = precompute_freqs_MS_2d(shape_list=shape_list, head_dim=self.head_dim, base=self.config.rope_base, mask_all=mask_all_training, use_patch_ck_ar=self.config.use_MS_ck_ar).to(device)
            mask_all = mask_all_training
            if self.config.use_alibi_bias:
                alibi_bias = self.build_alibi_for_all_slices(shape_list, device=device)
            else:
                alibi_bias = None

        if idx is not None and cond_idx is not None: # training or naive inference
            cond_embeddings = self.cls_embedding(cond_idx, train=self.training)[:,:self.cls_token_num]
            token_embeddings = self.tok_embeddings(idx)
            if not self.config.use_MS_ck_ar: ## Pixel-wise AR
                token_embeddings = torch.cat((cond_embeddings, token_embeddings), dim=1)
            else: # MS CK based AR
                token_embeddings = rerank_input(input=token_embeddings, cond_embedding=cond_embeddings, scale_shape=shape_list, mask_all=mask_all)
                self.query = token_embeddings # only for debug
            h = self.tok_dropout(token_embeddings)
            self.freqs_cis = self.freqs_cis.to(device)
        else:
            if cond_idx is not None: # prefill in inference
                token_embeddings = self.cls_embedding(cond_idx, train=self.training)[:,:self.cls_token_num]
            else: # decode_n_tokens(kv cache) in inference
                token_embeddings = self.tok_embeddings(idx)
            
            bs = token_embeddings.shape[0]
            mask = self.causal_mask[:bs, None, input_pos]
            h = self.tok_dropout(token_embeddings)

        # if self.training or keep_training_cis:
        #     freqs_cis = self.freqs_cis[:token_embeddings.shape[1]]
        # else:
        #     freqs_cis = self.freqs_cis[input_pos]
        # transformer blocks
        for layer in self.layers:
            h = layer(h, freqs_cis, input_pos, mask=mask, alibi_bias=alibi_bias)
        
        # output layers
        h = self.norm(h)
        logits = self.output(h).float()

        
        if self.training:
            logits = logits[:, self.cls_token_num - 1:].contiguous()
            # print(f"logits.shape in training:", logits.shape)

        # if we are given some desired targets also calculate the loss
        loss = None
        if valid is not None:
            loss_all = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')
            valid_all = valid[:,None].repeat(1, targets.shape[1]).view(-1)
            loss = (loss_all * valid_all).sum() / max(valid_all.sum(), 1)
        elif targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            # loss = cross_entropy_log2(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
    
    def generate_query_w_last_token(self, last_token, decoded_embeddings, pos):
        shape_list = self.shape_list
        if pos == 0: ##The first slice of first scale
            num_start= sum(self.mask_all == pos)
            query = last_token.expand(-1, num_start, -1)
            return query
        elif pos in self.start_idx: ## The first slice of other scales
            B, _, C = decoded_embeddings.shape
            idx = self.start_idx.index(pos)
            prev_h, prev_w = shape_list[idx-1]
            curr_h, curr_w = shape_list[idx]
            
            ## get the positions of last token in the last scale and current scale in mask_all
            idx_range_last_scale = torch.arange(self.each_scale_num[idx], device=last_token.device) + sum(self.each_scale_num[:idx])
            idx_range_curr_scale = torch.arange(self.each_scale_num[idx+1], device=last_token.device) + sum(self.each_scale_num[:idx+1])

            feature_last_scale = decoded_embeddings[:, idx_range_last_scale].view(B, prev_h, prev_w, C).permute(0, 3, 1, 2).contiguous()
            up_feature = F.interpolate(feature_last_scale, size=(curr_h, curr_w), mode='bicubic', align_corners=False).contiguous()
            up_feature = up_feature.permute(0, 2, 3, 1).contiguous().view(B, curr_h*curr_w, C)

            mask_all_curr_scale = self.mask_all[idx_range_curr_scale]

            curr_pos = mask_all_curr_scale == pos
            query = up_feature[:, curr_pos].contiguous()
            return query
    
        else: ## other slices
            positions = (self.mask_all == pos)
            #print('positions:', positions)
            pos_idx = torch.nonzero(positions, as_tuple=False).view(-1)
            prev_positions = (self.mask_all == pos-1)
            prev_idx = torch.nonzero(prev_positions, as_tuple=False).view(-1)
            if prev_idx.numel() == 0:
                        raise Exception("The numel of last slice should be 1/4 of total indices in this scale!")
            if prev_idx.numel() != pos_idx.numel():
                num_diff = abs(prev_idx.numel() - pos_idx.numel())
                if num_diff/pos_idx > 0.05:
                    print("Warning!!! the neraby slice in the save scale has numel difference of:", num_diff/pos_idx)
                rep_times = (pos_idx.numel() + prev_idx.numel() - 1) // prev_idx.numel()
                prev_idx = prev_idx.repeat(rep_times)[:pos_idx.numel()]
            query = decoded_embeddings[:, prev_idx, :]
            return query


    def ar_one_step(
        self, 
        idx: torch.Tensor, 
        cond_idx: torch.Tensor,  # cond_idx_or_embed, [B, 1]
        input_pos:  Optional[torch.Tensor] = None,  #the slice idx
        targets: Optional[torch.Tensor] = None,
        all_tokens=None
    ):
        device = next(self.parameters()).device

        assert idx is None or cond_idx is None

        if cond_idx is not None: # prefill in inference
            cond_idx = cond_idx.squeeze(1)
            token_embeddings = self.cls_embedding(cond_idx, train=self.training)[:,:self.cls_token_num]
            decoded_embeddings = None
        else: # decode_n_tokens(kv cache) in inference
            token_embeddings = self.tok_embeddings(idx)
            all_tokens = all_tokens.to(device)
            decoded_embeddings = self.tok_embeddings(all_tokens) #(B, s_L, C)
        if self.config.use_MS_ck_ar:
            token_embeddings = self.generate_query_w_last_token(token_embeddings, decoded_embeddings, pos=input_pos)
        
        curr_pos = input_pos == self.mask_all # find the corresponding slice
        
        mask = self.attn_bias.to(device) ##(slice_num, L)
        mask = mask[curr_pos] ##(S,L)
        h = self.tok_dropout(token_embeddings)
        freqs_cis = self.freqs_cis[curr_pos].to(device)

        alibi_bias = self.alibi_bias[:, curr_pos.to("cpu")].to(device=device) #(H, S, L)

        # transformer blocks
        for layer in self.layers:
            h = layer(h, freqs_cis, curr_pos, mask, alibi_bias=alibi_bias)

        # output layers
        h = self.norm(h)
        logits = self.output(h).float()

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            targets = targets.to(device)
            targets = targets[:, curr_pos]
            # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss = cross_entropy_log2(logits.view(-1, logits.size(-1)), targets.view(-1), reduction="sum")
        else:
            loss=1e6

        return logits, loss

    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        return list(self.layers)



#################################################################################
#                      Rotary Positional Embedding Functions                    #
#################################################################################
# https://github.com/pytorch-labs/gpt-fast/blob/main/model.py 

def precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 10000, cls_token_num=120):
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs) # (seq_len, head_dim // 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1) # (cls_token_num+seq_len, head_dim // 2, 2)
    cond_cache = torch.cat([torch.zeros(cls_token_num, n_elem // 2, 2), cache]) # (cls_token_num+seq_len, head_dim // 2, 2)
    return cond_cache 


def precompute_freqs_cis_2d(grid_size: int, n_elem: int, base: int = 10000, cls_token_num=120):
    # split the dimension into half, one for x and one for y
    half_dim = n_elem // 2
    freqs = 1.0 / (base ** (torch.arange(0, half_dim, 2)[: (half_dim // 2)].float() / half_dim))
    t = torch.arange(grid_size, device=freqs.device)
    freqs = torch.outer(t, freqs) # (grid_size, head_dim // 2)
    freqs_grid = torch.concat([
        freqs[:, None, :].expand(-1, grid_size, -1),
        freqs[None, :, :].expand(grid_size, -1, -1),
    ], dim=-1)  # (grid_size, grid_size, head_dim // 2)
    cache_grid = torch.stack([torch.cos(freqs_grid), torch.sin(freqs_grid)], dim=-1) # (grid_size, grid_size, head_dim // 2, 2)
    cache = cache_grid.flatten(0, 1)
    cond_cache = torch.cat([torch.zeros(cls_token_num, n_elem // 2, 2), cache]) # (cls_token_num+grid_size**2, head_dim // 2, 2)
    return cond_cache 

# def build_2d_rope(grid_h, grid_w, head_dim, base=10000, cls_token_num=0, device="cuda"):
#     half_dim = head_dim // 2
    
#     freq_seq = 1.0 / (base ** (torch.arange(0, half_dim, 2, device=device).float() / half_dim))
    
#     # normalize and scaling
#     y = torch.arange(grid_h, device=device) / grid_h * 1000
#     x = torch.arange(grid_w, device=device) / grid_w * 1000 

#     freqs_y = torch.outer(y, freq_seq)   # [grid_h, half_dim//2]
#     freqs_x = torch.outer(x, freq_seq)   # [grid_w, half_dim//2]

#     freqs_y = freqs_y[:, None, :].expand(-1, grid_w, -1)
#     freqs_x = freqs_x[None, :, :].expand(grid_h, -1, -1)

#     freqs = torch.cat([freqs_y, freqs_x], dim=-1)   # [grid_h, grid_w, head_dim//2]
#     cache = torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1) # [H,W,head_dim//2,2]

#     cache = cache.flatten(0,1)  # [H*W, head_dim//2, 2]

#     if cls_token_num > 0:
#         cls_cache = torch.zeros(cls_token_num, head_dim//2, 2, device=device)
#         cache = torch.cat([cls_cache, cache], dim=0)

#     return cache
@dynamo.disable
def build_2d_rope(
    grid_h, grid_w, head_dim,
    base=1000, cls_token_num=0, device="cuda", train_shape=(16,16),
    scaling="ntk"
):
    train_grid_h, train_grid_w=train_shape
    half_dim = head_dim // 2  

    inv_freq = 1.0 / (base ** (torch.arange(0, half_dim, 2, device=device).float() / half_dim))  # length = half_dim//2

    if scaling != "none":
        alpha_h = max(1.0, float(grid_h) / float(train_grid_h))
        alpha_w = max(1.0, float(grid_w) / float(train_grid_w))
        alpha = max(alpha_h, alpha_w)
        if scaling == "linear":
            inv_freq = inv_freq / alpha
        elif scaling == "ntk":
            inv_freq = inv_freq / (alpha ** 1.5)

    y = torch.arange(grid_h, device=device)
    x = torch.arange(grid_w, device=device)

    freqs_y = torch.outer(y, inv_freq)    # [grid_h, half_dim//2]
    freqs_x = torch.outer(x, inv_freq)    # [grid_w, half_dim//2]

    freqs_y = freqs_y[:, None, :].expand(-1, grid_w, -1)  # [grid_h, grid_w, half_dim//2]
    freqs_x = freqs_x[None, :, :].expand(grid_h, -1, -1)  # [grid_h, grid_w, half_dim//2]

    freqs = torch.cat([freqs_y, freqs_x], dim=-1)  # [grid_h, grid_w, half_dim]
    # final cache shape: [H*W, head_dim//2, 2] (cos, sin)
    cache = torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)  # [H,W,half_dim,2]
    cache = cache.flatten(0, 1)  # [H*W, half_dim, 2]

    if cls_token_num > 0:
        cls_cache = torch.zeros(cls_token_num, head_dim // 2, 2, device=device)
        cache = torch.cat([cls_cache, cache], dim=0)

    return cache

@dynamo.disable
def precompute_freqs_MS_2d(shape_list, head_dim, base=5000, device="cuda", mask_all: torch.Tensor=None, use_patch_ck_ar=False):
    cache_all = []
    cls_token_num = 0
    train_shape_list = [(4,4), (8,8), (16,16)]
    # scale_factor = shape_list[-1][0] / 16.0
    for i, pn in enumerate(shape_list):
        cache = build_2d_rope(pn[0], pn[1], head_dim, base=base, cls_token_num=cls_token_num, device=device, train_shape= train_shape_list[i], scaling="ntk")
        cache_all.append(cache)
    cache_all = torch.cat(cache_all, dim=0)
    if not use_patch_ck_ar:
        cache_all = cache_all[:-1,...]
    pos_start = mask_all == 0
    num_pos_start = pos_start.sum()
    if num_pos_start > 0:
        cache_all[pos_start,...] = torch.zeros(num_pos_start, head_dim//2, 2, device=device)
    return cache_all

@dynamo.disable
def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    # x: (bs, seq_len, n_head, head_dim)
    # freqs_cis (seq_len, head_dim // 2, 2)
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2) # (bs, seq_len, n_head, head_dim//2, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2) # (1, seq_len, 1, head_dim//2, 2)
    x_out2 = torch.stack([
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
    ], dim=-1)
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)



#################################################################################
#                                GPT Configs                                    #
#################################################################################
### text-conditional
def GPT_7B(**kwargs):
    return Transformer(ModelArgs(n_layer=32, n_head=32, dim=4096, **kwargs)) # 6.6B

def GPT_3B(**kwargs):
    return Transformer(ModelArgs(n_layer=24, n_head=32, dim=3200, **kwargs)) # 3.1B

def GPT_1B(**kwargs):
    return Transformer(ModelArgs(n_layer=22, n_head=32, dim=2048, **kwargs)) # 1.2B

### class-conditional
def GPT_XXXL(**kwargs):
    return Transformer(ModelArgs(n_layer=48, n_head=40, dim=2560, **kwargs)) # 3.9B

def GPT_XXL(**kwargs):
    return Transformer(ModelArgs(n_layer=48, n_head=24, dim=1536, **kwargs)) # 1.4B

def GPT_XL(**kwargs):
    return Transformer(ModelArgs(n_layer=36, n_head=20, dim=1280, **kwargs)) # 775M

def GPT_L(**kwargs):
    return Transformer(ModelArgs(n_layer=24, n_head=16, dim=1024, **kwargs)) # 343M

def GPT_B(**kwargs):
    return Transformer(ModelArgs(n_layer=12, n_head=12, dim=768, **kwargs)) # 111M
        

GPT_models = {
    'GPT-B': GPT_B, 'GPT-L': GPT_L, 'GPT-XL': GPT_XL, 'GPT-XXL': GPT_XXL, 'GPT-XXXL': GPT_XXXL,
    'GPT-1B': GPT_1B, 'GPT-3B': GPT_3B, 'GPT-7B': GPT_7B, 
}