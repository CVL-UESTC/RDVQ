import torch
import torch.nn as nn
import torch.nn.functional as F


# this file only provides the 2 modules used in VQVAE
__all__ = ['Encoder', 'Decoder',]


"""
References: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/diffusionmodules/model.py
"""
# swish
def nonlinearity(x):
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample2x(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode='nearest'))


class Downsample2x(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)
    
    def forward(self, x):
        return self.conv(F.pad(x, pad=(0, 1, 0, 1), mode='constant', value=0))


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, dropout): # conv_shortcut=False,  # conv_shortcut: always False in VAE
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        
        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout) if dropout > 1e-6 else nn.Identity()
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.nin_shortcut = nn.Identity()
    
    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x), inplace=True))
        h = self.conv2(self.dropout(F.silu(self.norm2(h), inplace=True)))
        return self.nin_shortcut(x) + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.C = in_channels
        
        self.norm = Normalize(in_channels)
        self.qkv = torch.nn.Conv2d(in_channels, 3*in_channels, kernel_size=1, stride=1, padding=0)
        self.w_ratio = int(in_channels) ** (-0.5)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        qkv = self.qkv(self.norm(x))
        B, _, H, W = qkv.shape  # should be B,3C,H,W
        C = self.C
        q, k, v = qkv.reshape(B, 3, C, H, W).unbind(1)
        
        # compute attention
        q = q.view(B, C, H * W).contiguous()
        q = q.permute(0, 2, 1).contiguous()     # B,HW,C
        k = k.view(B, C, H * W).contiguous()    # B,C,HW
        w = torch.bmm(q, k).mul_(self.w_ratio)  # B,HW,HW    w[B,i,j]=sum_c q[B,i,C]k[B,C,j]
        w = F.softmax(w, dim=2)
        
        # attend to values
        v = v.view(B, C, H * W).contiguous()
        w = w.permute(0, 2, 1).contiguous()  # B,HW,HW (first HW of k, second of q)
        h = torch.bmm(v, w)  # B, C,HW (HW of q) h[B,C,j] = sum_i v[B,C,i] w[B,i,j]
        h = h.view(B, C, H, W).contiguous()
        
        return x + self.proj_out(h)

# class WindowedAttnBlock(nn.Module):
#     """
#     带有分窗功能的注意力模块。
#     当输入特征图尺寸大于window_size时，执行重叠的分窗注意力。
#     否则，执行标准的全局注意力。
#     """
#     def __init__(self, in_channels, window_size):
#         super().__init__()
#         self.C = in_channels
#         self.window_size = window_size
        
#         self.norm = Normalize(in_channels)
#         self.qkv = nn.Conv2d(in_channels, 3 * in_channels, kernel_size=1, stride=1, padding=0)
#         self.w_ratio = int(in_channels) ** (-0.5)
#         self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

#     def _global_attention(self, x):
#         """ 原始的全局注意力实现 """
#         B, C, H, W = x.shape
#         qkv = self.qkv(self.norm(x))
#         q, k, v = qkv.reshape(B, 3, C, H, W).unbind(1)
        
#         # 计算注意力权重
#         q = q.view(B, C, H * W).contiguous().permute(0, 2, 1)  # B, HW, C
#         k = k.view(B, C, H * W).contiguous()                     # B, C, HW
#         w = torch.bmm(q, k).mul_(self.w_ratio)                   # B, HW, HW
#         w = F.softmax(w, dim=2)
        
#         # 应用注意力到values
#         v = v.view(B, C, H * W).contiguous()
#         w = w.permute(0, 2, 1).contiguous()
#         h = torch.bmm(v, w)                                     # B, C, HW
#         h = h.view(B, C, H, W).contiguous()
        
#         return self.proj_out(h)

#     def forward(self, input):
#         B, C, H, W = input.shape
#         # 如果特征图小于或等于窗口大小，则执行全局注意力
#         if self.window_size is None or (H <= self.window_size and W <= self.window_size):
#             attn_out = self._global_attention(input)
#             return input + attn_out

#         # --- 分窗注意力逻辑 ---
#         ws = self.window_size
#         # 为了重叠1格，stride为window_size - 1
#         stride = ws - 2
        
#         # 1. 计算所有窗口的起始坐标，确保覆盖整个特征图
#         y_starts = []
#         x_starts = []
        
#         # 计算y方向的起始位置
#         y = 0
#         while y < H:
#             y_starts.append(y)
#             y += stride
#             # 如果下一个窗口会超出边界，调整最后一个窗口的位置
#             if y + ws > H and y_starts[-1] + ws != H:
#                 y_starts.append(max(0, H - ws))
#                 break
        
#         # 计算x方向的起始位置  
#         x = 0
#         while x < W:
#             x_starts.append(x)
#             x += stride
#             # 如果下一个窗口会超出边界，调整最后一个窗口的位置
#             if x + ws > W and x_starts[-1] + ws != W:
#                 x_starts.append(max(0, W - ws))
#                 break
        
#         # 去重并排序
#         y_starts = sorted(list(set(y_starts)))
#         x_starts = sorted(list(set(x_starts)))
        
#         # 2. 提取所有重叠的窗口
#         patches = []
#         valid_coords = []  # 记录有效的坐标对
        
#         for y in y_starts:
#             for x in x_starts:
#                 # 确保窗口不会超出边界
#                 if y + ws <= H and x + ws <= W:
#                     patch = input[:, :, y:y+ws, x:x+ws]
#                     patches.append(patch)
#                     valid_coords.append((y, x))
        
#         # 如果没有有效的窗口，回退到全局注意力
#         if not patches:
#             attn_out = self._global_attention(input)
#             return input + attn_out
        
#         # 将patches列表堆叠并重塑为一个大的批次
#         # patches: [num_windows, B, C, ws, ws]
#         patches = torch.stack(patches, dim=0)
#         num_windows = patches.shape[0]
        
#         # patches_reshaped: [B*num_windows, C, ws, ws]
#         patches_reshaped = patches.permute(1, 0, 2, 3, 4).reshape(-1, C, ws, ws)
        
#         # 3. 对所有窗口并行执行注意力计算
#         attn_out_reshaped = self._global_attention(patches_reshaped)
        
#         # 4. 将结果重塑回来
#         # attn_out_windows: [B, num_windows, C, ws, ws]
#         attn_out_windows = attn_out_reshaped.reshape(B, num_windows, C, ws, ws)
        
#         # 5. 重建输出特征图（使用平均法处理重叠区域）
#         result = torch.zeros_like(input)
#         counts = torch.zeros_like(input)
        
#         for window_idx, (y, x) in enumerate(valid_coords):
#             result[:, :, y:y+ws, x:x+ws] += attn_out_windows[:, window_idx, :, :, :]
#             counts[:, :, y:y+ws, x:x+ws] += 1.0
        
#         # 避免除以零
#         final_h = result / (counts + 1e-8)

#         return input + final_h


def make_attn(in_channels, using_sa=True, window_size=None):
    return AttnBlock(in_channels) if using_sa else nn.Identity()

# def make_attn(in_channels, using_sa=True, window_size=None):
#     return WindowedAttnBlock(in_channels, window_size) if using_sa else nn.Identity()


class Encoder(nn.Module):
    def __init__(
        self, *, ch=128, ch_mult=(1, 2, 4, 8), num_res_blocks=2,
        dropout=0.0, in_channels=3,
        z_channels, double_z=False, using_sa=True, using_mid_sa=True, scale=["s1"], enc_residual=True
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.downsample_ratio = 2 ** (self.num_resolutions - 1)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels

        self.enc_residual = enc_residual
        self.scales = scale
        assert len(self.scales)<self.num_resolutions+2
        self.scales_id = [self.num_resolutions-i-1 for i in range(len(self.scales))] # [self.num_resolutions-1: s1, self.num_resolutions-1: s2, ...]

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)
        
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                # if i_level == self.num_resolutions - 1 and using_sa:
                if i_level in self.scales_id and using_sa:
                    attn.append(make_attn(block_in, using_sa=True, window_size=256//2**i_level))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample2x(block_in)
            self.down.append(down)
        
        # middle
        if scale == ["s1"]:
            self.mid = nn.Module()
            self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
            self.mid.attn_1 = make_attn(block_in, using_sa=using_mid_sa)
            self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
            # end
            self.norm_out = Normalize(block_in)
            self.conv_out = torch.nn.Conv2d(block_in, (2 * z_channels if double_z else z_channels), kernel_size=3, stride=1, padding=1)
        else:
            self.mid = nn.ModuleDict()
            self.norm_out = nn.ModuleDict()
            self.conv_out = nn.ModuleDict()
            ## All scales should have the same channels
            for i, s in enumerate(self.scales):
                self.mid[s] = nn.Module()
                self.mid[s].block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
                self.mid[s].attn_1 = make_attn(block_in, using_sa=using_mid_sa, window_size=4*2**i)
                self.mid[s].block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
                # end
                self.norm_out[s] = Normalize(block_in)
                self.conv_out[s] = torch.nn.Conv2d(block_in, (2 * z_channels if double_z else z_channels), kernel_size=3, stride=1, padding=1)
        
    
    def forward(self, x):

        # downsampling
        h = self.conv_in(x)
        h_dict_in = {}
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level in self.scales_id:
                idx = self.scales_id.index(i_level)
                h_dict_in[self.scales[idx]] = h
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)
        
        # middle
        if len(self.scales)==1:
            h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h)))
            
            # end
            h = self.conv_out(F.silu(self.norm_out(h), inplace=True))
        else:
            h_dict = {}
            for s in self.scales:
                h_mid = self.mid[s].block_2(self.mid[s].attn_1(self.mid[s].block_1(h_dict_in[s])))
                # end
                h_dict[s] = self.conv_out[s](F.silu(self.norm_out[s](h_mid), inplace=True))
            if self.enc_residual:
                for i, s in enumerate(self.scales):
                    if i > 0:
                        for j in range(0, i):
                            up_scale_factor = 2**(i-j)
                            h_dict[s] = h_dict[s] - F.interpolate(h_dict[self.scales[j]], scale_factor=up_scale_factor, mode="bicubic", align_corners=False)
        if len(self.scales) == 1:
            return h
        else:
            return h_dict

class Decoder(nn.Module):
    def __init__(
        self, *, ch=128, ch_mult=(1, 2, 4, 8), num_res_blocks=2,
        dropout=0.0, in_channels=3,  # in_channels: raw img channels
        z_channels, using_sa=True, using_mid_sa=True, scale=["s1"]
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels

        self.scales = scale
        self.scales_id = [self.num_resolutions-i-1 for i in range(len(self.scales))] # [self.num_resolutions-1: s1, self.num_resolutions-1: s2, ...]
        
        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        
        
        # middle
        if len(self.scales)==1:
            self.mid = nn.Module()
            self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
            self.mid.attn_1 = make_attn(block_in, using_sa=using_mid_sa)
            self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
            # z to block_in
            self.conv_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)
        else:
            self.mid = nn.ModuleDict()
            self.conv_in = nn.ModuleDict()
            ## All scales should have the same channels
            for i, s in enumerate(self.scales):
                self.mid[s] = nn.Module()
                self.mid[s].block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
                self.mid[s].attn_1 = make_attn(block_in, using_sa=using_mid_sa, window_size=4*2**i)
                self.mid[s].block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
                # z to block_in
                self.conv_in[s] = torch.nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)
        
        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                if i_level in self.scales_id and using_sa:
                    attn.append(make_attn(block_in, using_sa=True, window_size=256//2**i_level))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample2x(block_in)
            self.up.insert(0, up)  # prepend to get consistent order
        
        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, in_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, z, return_fused_feature = False):
        # z to block_in
        # middle
        if len(self.scales)==1:
            h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(self.conv_in(z))))
        else:
            h_dict = {}
            for s in self.scales:
                h_dict[s] = self.mid[s].block_2(self.mid[s].attn_1(self.mid[s].block_1(self.conv_in[s](z[s]))))
            h=0
        
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            if i_level in self.scales_id and len(self.scales)>1:
                idx = self.scales_id.index(i_level)
                h = h + h_dict[self.scales[idx]]
                if return_fused_feature and idx == len(self.scales_id)-1:
                    return h
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        
        # end
        h = self.conv_out(F.silu(self.norm_out(h), inplace=True))
        return h
