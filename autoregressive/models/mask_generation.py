import math
import torch
import copy
import torch.nn.functional as F
import torch._dynamo as dynamo

@dynamo.disable
def process_mask(mask, n_h, n_w, h, w):
        ## return mask: 1, h, w
        mask_full = mask.repeat(n_h, n_w)
        mask_full = mask_full[:h, :w]  # Crop to the desired size
        return mask_full

@dynamo.disable
def closest_factors(n: int):
    assert n % 4 == 0, "n must be divided by 4"
    root = int(math.isqrt(n))  # sqrt(n) 向下取整
    for a in range(root, 0, -1):
        if n % a == 0:
            b = n // a
            return a, b
        
@dynamo.disable
def generate_num_k_mask(h, w, device, k=9):
    length = int(math.isqrt(k))
    if length * length != k:
        #raise ValueError(f"k must be a perfect square, got k={k}")
        s_h, s_w = closest_factors(k)
    else:
        s_h, s_w = length, length
    # print(f"k:{k}, s_h:{s_h}, s_w:{s_w}")
    mask_unit = torch.arange(k, device=device).view(s_h, s_w)
    n_h = h // s_h + 1
    n_w = w // s_w + 1
    mask = mask_unit.repeat(n_h, n_w)[:h, :w]
    return mask.flatten().long()

@dynamo.disable
def generate_MS_ck_mask(input_shape, device, use_patch_ck_ar=True, k=4):
    ## Generating MS checkboard AR mask when use_patch_ck_ar, else generating pixel wise autoregressive mask
    mask_list = []
    # if not use_patch_ck_ar:
        # start_token = torch.tensor([0], dtype=torch.long, device=device)
        # mask_list.append(start_token)
    idx_so_far = 0
    
    for i, (h, w) in enumerate(input_shape):
        num_ar_i = k*2**i
        if use_patch_ck_ar:
            m = generate_num_k_mask(h, w, device, k=num_ar_i) + idx_so_far
            idx_so_far += num_ar_i
        else:
            #m = torch.zeros(h * w, dtype=torch.long, device=device) + i + 1
            m = torch.arange(idx_so_far, idx_so_far+h*w, device= device)
            idx_so_far = idx_so_far+h*w
        mask_list.append(m)
    mask_all = torch.cat(mask_list, dim=0).view(-1)  # long
    # print("mask_all.shape:",mask_all.shape)
    # if not use_patch_ck_ar:
    #     mask_all = mask_all[:-1]
    # print("mask_all.shape:",mask_all.shape, "use_patch_ck_ar:", use_patch_ck_ar)
    n = mask_all.numel()
    col = mask_all.view(n, 1)
    row = mask_all.view(1, n)
    allowed = (col > row)   # 或改成 (col >= row) 视语义决定
    start_col = col==0
    start_row = row==0
    allowed_start = start_col & start_row
    allowed = allowed | allowed_start
    attn_bias = torch.zeros(n, n, device=device, dtype=torch.float32)
    attn_bias.masked_fill_(~allowed, float('-inf'))
    return attn_bias, mask_all


def rerank_input(input, cond_embedding, scale_shape, mask_all, k=4):
    ## This function is utilized to rearanged input according mask all
    ## input: [B, L, C]
    ## cond_embedding: [B, 1, C]
    ## mask_all: [L]
    B, L, C = input.shape

    device = input.device

    assert mask_all.ndim==1
    mask_all = mask_all.to(device).long()

    # scale_shape = [(1, 1)] + copy.deepcopy(scale_shape)
    raw_scale_slices = []
    idx = 0
    for (h, w) in scale_shape:
        end = idx + h * w
        raw = input[:, idx:end, :]   # B, h*w, C
        raw_scale_slices.append(raw)
        idx = end

    # 上一尺度上采样到下一尺度
    each_scale_features = []
    for i in range(len(raw_scale_slices) - 1):
        prev = raw_scale_slices[i]
        prev_h, prev_w = scale_shape[i]
        next_h, next_w = scale_shape[i + 1]
        # print(f"i:{i}, prev.shape:", prev.shape, f"B:{B}, scale_i:{scale_shape[i]}, scale_i+1:{scale_shape[i+1]}")
        prev_reshaped = prev.view(B, prev_h, prev_w, C).permute(0, 3, 1, 2).contiguous()
        up = F.interpolate(prev_reshaped, size=(next_h, next_w), mode='bicubic', align_corners=False).contiguous()
        up_flat = up.permute(0, 2, 3, 1).contiguous().view(B, next_h * next_w, C)
        each_scale_features.append(up_flat)

    query_features = torch.zeros_like(input, device=device, dtype=input.dtype)
    # query_features[:, 0, :] = st

    num_scales = len(scale_shape)

    idx_so_far = 0
    for scale_i in range(num_scales):
        num_ar_scale = k*2**scale_i
        h, w = scale_shape[scale_i]
        base_mask = generate_num_k_mask(h, w, device, k=num_ar_scale) if (k != -1) else torch.zeros(h*w, dtype=torch.long, device=device)
        for slice_idx in range(num_ar_scale if k!=-1 else 1):
            # global_value = slice_idx + k * scale_i if k!=-1 else scale_i
            global_value = slice_idx + idx_so_far
            positions = (mask_all == global_value)
            pos_idx = torch.nonzero(positions, as_tuple=False).view(-1)
            if slice_idx == 0:
                if scale_i == 0:
                    # st_batch = st.unsqueeze(1).expand(B, pos_idx.numel(), C)
                    # print("cond_embedding.shape:",cond_embedding.shape)
                    st_batch = cond_embedding.expand(B, pos_idx.numel(), C)
                    query_features[:, pos_idx, :] = st_batch
                else:
                    sel = (base_mask == 0)
                    last_scale_input = each_scale_features[scale_i - 1][:, sel, :]
                    query_features[:, pos_idx, :] = last_scale_input
                # print("slice_idx 0, pos_idx numel:", pos_idx.numel())
                # print(f"scale shape:{scale_shape[scale_i]}, scale_i:{scale_i}, num_ar_scale:{num_ar_scale}, idx_so_far:{idx_so_far}, global_value:{global_value}, pos_idx numel:{pos_idx.numel()}")
            else:
                prev_global = (slice_idx - 1) + idx_so_far
                prev_positions = (mask_all == prev_global)
                prev_idx = torch.nonzero(prev_positions, as_tuple=False).view(-1)

                if len(prev_idx) == 0:
                    raise Exception("The numel of last slice should be 1/4 of total indices in this scale!")
                # repeat prev_idx to match pos_idx if needed
                if len(prev_idx) != len(pos_idx):
                    num_diff = (len(prev_idx) - len(pos_idx))
                    if num_diff/max(len(pos_idx), len(prev_idx)) > 0.5:
                        print("Warning!!! the neraby slice in the save scale has numel difference of:", num_diff/max(len(pos_idx), len(prev_idx)))
                    rep_times = (len(pos_idx) + len(prev_idx) - 1) // len(prev_idx)
                    prev_idx = prev_idx.repeat(rep_times)[:len(pos_idx)]
                query_features[:, pos_idx, :] = input[:, prev_idx, :]
        idx_so_far += num_ar_scale
    return query_features.contiguous()