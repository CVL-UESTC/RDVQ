# Modified from:
#   VQGAN:    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/transformer/mingpt.py
#   DiT:      https://github.com/facebookresearch/DiT/blob/main/models.py  
#   nanoGPT:  https://github.com/karpathy/nanoGPT/blob/master/model.py
#   llama:    https://github.com/facebookresearch/llama/blob/main/llama/model.py
#   gpt-fast: https://github.com/pytorch-labs/gpt-fast/blob/main/model.py
#   PixArt:   https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
from dataclasses import dataclass
from typing import Optional, List


import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.drop_path import DropPath
from autoregressive.models.mask_generation import generate_MS_ck_mask, rerank_input
from autoregressive.models.gpt import LabelEmbedder, CaptionEmbedder, RMSNorm, KVCache, TransformerBlock


def find_multiple(n: int, k: int):
    if n % k == 0:
        return n
    return n + k - (n % k)

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

    use_patch_ck_ar: bool = True
    use_MS_ck_ar: bool=True
    cos_attn: bool=True
    use_alibi_bias: bool = False


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
        grid_size = int(self.block_size ** 0.5)
        assert grid_size * grid_size == self.block_size
        self.freqs_cis = precompute_freqs_cis_2d(grid_size, self.config.dim // self.config.n_head, self.config.rope_base, self.cls_token_num)
        
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

    def setup_caches(self, max_batch_size, max_seq_length, dtype):
        # if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
        #     return
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        for b in self.layers:
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_head, head_dim, dtype)

        causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool))
        self.causal_mask = causal_mask.unsqueeze(0).repeat(self.max_batch_size, 1, 1)
        grid_size = int(self.config.block_size ** 0.5)
        assert grid_size * grid_size == self.block_size
        self.freqs_cis = precompute_freqs_cis_2d(grid_size, self.config.dim // self.config.n_head, self.config.rope_base, self.cls_token_num)

    def forward(
        self, 
        idx: torch.Tensor, 
        cond_idx: torch.Tensor,  # cond_idx_or_embed
        input_pos:  Optional[torch.Tensor] = None, 
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
        keep_training_cis=False
    ):
        if idx is not None and cond_idx is not None: # training or naive inference
            cond_embeddings = self.cls_embedding(cond_idx, train=self.training)[:,:self.cls_token_num]
            token_embeddings = self.tok_embeddings(idx)
            token_embeddings = torch.cat((cond_embeddings, token_embeddings), dim=1)
            h = self.tok_dropout(token_embeddings)
            self.freqs_cis = self.freqs_cis.to(h.device)
        else:
            if cond_idx is not None: # prefill in inference
                token_embeddings = self.cls_embedding(cond_idx, train=self.training)[:,:self.cls_token_num]
            else: # decode_n_tokens(kv cache) in inference
                token_embeddings = self.tok_embeddings(idx)
            
            bs = token_embeddings.shape[0]
            mask = self.causal_mask[:bs, None, input_pos]
            h = self.tok_dropout(token_embeddings)
            self.freqs_cis = self.freqs_cis
        
        if self.training or keep_training_cis:
            freqs_cis = self.freqs_cis[:token_embeddings.shape[1]]
        else:
            freqs_cis = self.freqs_cis[input_pos]
        # transformer blocks
        for layer in self.layers:
            h = layer(h, freqs_cis, input_pos, mask)
        
        # output layers
        h = self.norm(h)
        logits = self.output(h).float()
        
        if self.training:
            logits = logits[:, self.cls_token_num - 1:].contiguous()

        # if we are given some desired targets also calculate the loss
        loss = None
        if valid is not None:
            loss_all = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')
            valid_all = valid[:,None].repeat(1, targets.shape[1]).view(-1)
            loss = (loss_all * valid_all).sum() / max(valid_all.sum(), 1)
        elif targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss


    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        return list(self.layers)


class Transformer_MS_input(nn.Module):
    def __init__(self, config: ModelArgs, use_tok_embedding=True):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layer = config.n_layer
        self.block_size = config.block_size
        self.num_classes = config.num_classes
        self.model_type = config.model_type
        self.cls_token_num = config.cls_token_num
        # if self.model_type == 'c2i':
        #     self.cls_embedding = LabelEmbedder(config.num_classes, config.dim, config.class_dropout_prob)
        # elif self.model_type == 't2i':
        #     self.cls_embedding = CaptionEmbedder(config.caption_dim, config.dim, config.class_dropout_prob)
        # else:
        #     raise Exception("please check model type")
        if use_tok_embedding:
            self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        else:
            self.tok_embeddings = None
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
        if self.config.use_MS_ck_ar:
            shape_list = [(4,4), (8,8), (16,16)]
        else:
            shape_list = [(1,1), (4,4), (8,8), (16,16)]

        self.build_mask_and_freq_cis(shape_list)
        # attn_bias, mask_all_training = generate_MS_ck_mask(input_shape=shape_list, device="cpu", use_patch_ck_ar=self.config.use_patch_ck_ar)
        # self.freqs_cis = precompute_freqs_MS_2d(shape_list=shape_list, head_dim=self.head_dim, base=self.config.rope_base, mask_all=mask_all_training, use_patch_ck_ar=self.config.use_patch_ck_ar)
        
        # self.attn_bias = attn_bias
        # self.mask_all = mask_all_training

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
        attn_bias, mask_all_training = generate_MS_ck_mask(input_shape=shape_list, device="cpu", use_patch_ck_ar=self.config.use_MS_ck_ar, k=k)
        # print(f"mask_all_training.shape:",mask_all_training.shape)
        self.freqs_cis = precompute_freqs_MS_2d(shape_list=shape_list, head_dim=self.head_dim, base=self.config.rope_base, mask_all=mask_all_training, use_patch_ck_ar=self.config.use_MS_ck_ar)
        self.attn_bias = attn_bias
        self.mask_all = mask_all_training
        start_idx = [0] + [k*2**i for i in range(len(shape_list)-1)]
        self.start_idx = [sum(start_idx[:i+1]) for i in range(len(start_idx))]
        self.each_scale_num = [0] + [pn[0]*pn[1] for pn in shape_list]

        self.freqs_cis = self.freqs_cis.to(device)
        self.attn_bias = self.attn_bias.to(device)
        self.mask_all = self.mask_all.to(device)

        # self.alibi_bias = self.build_alibi_for_all_slices(shape_list, device="cpu")
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
        ranked_input,
        shape_list = None,
        input_pos:  Optional[torch.Tensor] = None, 
    ):
        h = self.tok_dropout(ranked_input)

        device = h.device
        if self.training and (shape_list[-1]==(16,16)):
            mask = self.attn_bias.to(device)
            # freqs_cis = self.freqs_cis[:token_embeddings.shape[1]]
            freqs_cis = self.freqs_cis.to(device)
            if self.config.use_alibi_bias:
                alibi_bias = self.alibi_bias.to(device=device)
            else:
                alibi_bias = None
        else:
            assert shape_list is not None
            mask, mask_all_training = generate_MS_ck_mask(input_shape=shape_list, device="cpu", use_patch_ck_ar=self.config.use_MS_ck_ar)
            mask_all_training = mask_all_training.to(device)
            freqs_cis = precompute_freqs_MS_2d(shape_list=shape_list, head_dim=self.head_dim, base=self.config.rope_base, mask_all=mask_all_training, use_patch_ck_ar=self.config.use_MS_ck_ar).to(device)
            alibi_bias = self.build_alibi_for_all_slices(shape_list, device=device) if self.config.use_alibi_bias else None
            # alibi_bias = alibi_bias.cpu()
            
            
        freqs_cis = freqs_cis.to(device)

        for layer in self.layers:
            h = layer(h, freqs_cis, input_pos, mask=mask, alibi_bias=alibi_bias)
        
        # output layers
        h = self.norm(h)
        logits = self.output(h).float()

        return logits

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
        token_embeddings: torch.Tensor, 
        input_pos:  Optional[torch.Tensor] = None,  #the slice idx
    ):
        device = next(self.parameters()).device

        # assert idx is None or cond_idx is None

        # if cond_idx is not None: # prefill in inference
        #     cond_idx = cond_idx.squeeze(1)
        #     token_embeddings = self.cls_embedding(cond_idx, train=self.training)[:,:self.cls_token_num]
        #     decoded_embeddings = None
        # else: # decode_n_tokens(kv cache) in inference
        #     token_embeddings = self.tok_embeddings(idx)
        #     all_tokens = all_tokens.to(device)
        #     decoded_embeddings = self.tok_embeddings(all_tokens) #(B, s_L, C)
        # if self.config.use_MS_ck_ar:
        #     token_embeddings = self.generate_query_w_last_token(token_embeddings, decoded_embeddings, pos=input_pos)
        
        curr_pos = input_pos == self.mask_all # find the corresponding slice
        
        mask = self.attn_bias.to(device) ##(slice_num, L)
        mask = mask[curr_pos] ##(S,L)
        h = self.tok_dropout(token_embeddings)
        freqs_cis = self.freqs_cis[curr_pos].to(device)

        if self.config.use_alibi_bias:
            alibi_bias = self.alibi_bias[curr_pos.to("cpu")].to(device=device) #(H, S, L)
        else:
            alibi_bias = None

        # transformer blocks
        for layer in self.layers:
            h = layer(h, freqs_cis, curr_pos, mask, alibi_bias=alibi_bias)

        # output layers
        h = self.norm(h)
        logits = self.output(h).float()

        return logits
    
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


def precompute_freqs_cis_2d(grid_size: int, n_elem: int, base: int = 10000, cls_token_num=1):
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

def build_2d_rope(grid_h, grid_w, head_dim, base=1000, cls_token_num=0, device="cuda"):
    half_dim = head_dim // 2
    freq_seq = 1.0 / (base ** (torch.arange(0, half_dim, 2, device=device).float() / half_dim))
    
    # normalize and scaling
    y = torch.arange(grid_h, device=device) / grid_h * 1000
    x = torch.arange(grid_w, device=device) / grid_w * 1000 

    freqs_y = torch.outer(y, freq_seq)   # [grid_h, half_dim//2]
    freqs_x = torch.outer(x, freq_seq)   # [grid_w, half_dim//2]

    freqs_y = freqs_y[:, None, :].expand(-1, grid_w, -1)
    freqs_x = freqs_x[None, :, :].expand(grid_h, -1, -1)

    freqs = torch.cat([freqs_y, freqs_x], dim=-1)   # [grid_h, grid_w, head_dim//2]
    cache = torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1) # [H,W,head_dim//2,2]

    cache = cache.flatten(0,1)  # [H*W, head_dim//2, 2]

    if cls_token_num > 0:
        cls_cache = torch.zeros(cls_token_num, head_dim//2, 2, device=device)
        cache = torch.cat([cls_cache, cache], dim=0)

    return cache

def precompute_freqs_MS_2d(shape_list, head_dim, base=5000, device="cuda", mask_all: torch.Tensor=None, use_patch_ck_ar=False):
    cache_all = []
    cls_token_num = 0
    for i, pn in enumerate(shape_list):
        cache = build_2d_rope(pn[0], pn[1], head_dim, base=base, cls_token_num=cls_token_num, device=device)
        cache_all.append(cache)
    cache_all = torch.cat(cache_all, dim=0)
    if not use_patch_ck_ar:
        cache_all = cache_all[:-1,...]
    pos_start = mask_all == 0
    num_pos_start = pos_start.sum()
    if num_pos_start > 0:
        cache_all[pos_start,...] = torch.zeros(num_pos_start, head_dim//2, 2, device=device)
    return cache_all


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

def MS_input_transformer(**kwargs):
    return Transformer_MS_input(config=ModelArgs(n_layer=kwargs["num_layers"], n_head=kwargs["nhead"], dim=kwargs["d_model"], vocab_size=kwargs["vocab_size"], use_patch_ck_ar = kwargs["use_patch_ck_ar"]), use_tok_embedding=kwargs["use_tok_embedding"])        

GPT_models = {
    'GPT-B': GPT_B, 'GPT-L': GPT_L, 'GPT-XL': GPT_XL, 'GPT-XXL': GPT_XXL, 'GPT-XXXL': GPT_XXXL,
    'GPT-1B': GPT_1B, 'GPT-3B': GPT_3B, 'GPT-7B': GPT_7B, 
}