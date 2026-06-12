# Modified from:
#   taming-transformers: https://github.com/CompVis/taming-transformers
#   maskgit: https://github.com/google-research/maskgit
import logging
from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from tokenizer.tokenizer_image.models.autoencoder import Encoder as AREncoder, Decoder as ARDecoder
from tokenizer.tokenizer_image.models.quantizer import VectorQuantizer_MS_input


logger = logging.getLogger(__name__)


@dataclass
class ModelArgs:
    codebook_size: int = 16384
    codebook_embed_dim: int = 8
    codebook_l2_norm: bool = True
    codebook_show_usage: bool = True
    commit_loss_beta: float = 0.25
    entropy_loss_ratio: float = 0.0
    enc_residual: bool = False
    num_res_quant: int = 0

    use_predictor: bool = False
    wo_attn:bool = True
    
    encoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    decoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    z_channels: int = 256
    dropout_p: float = 0.0
    scale: List[str] = field(default_factory=lambda: ["s"])
    scale_sample: bool = False
    use_hyper: bool = False

    tau: float = 0.01
    num_codebooks: int = 1

    patch_size: int=256


def pad_right_bottom(x, p):
    """
    return
        x_padded
        padding_info: (left, right, top, bottom)
        mask: (1, 1, h, w), padded set to 1, original set to 0
    """
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p

    padding_left = 0
    padding_top = 0
    padding_right = new_w - w
    padding_bottom = new_h - h

    
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )

    mask = torch.zeros((1, 1, new_h, new_w), dtype=torch.float32, device=x.device)
    if padding_right > 0:
        mask[:, :, :, -padding_right:] = 1
    if padding_bottom > 0:
        mask[:, :, -padding_bottom:, :] = 1

    return x_padded, (padding_left, padding_right, padding_top, padding_bottom), mask


def crop_right_bottom(x, padding):
    # Undo pad_right_bottom after patch-wise latent processing; padding is only
    # added to the right and bottom edges.
    if any(p != 0 for p in padding):
        x = x[:, :, :-padding[3] if padding[3] > 0 else None,
                   :-padding[1] if padding[1] > 0 else None]
    return x

def split_to_patches(x, s):
    # Flatten a spatial grid of latent windows into the batch dimension so the
    # quantizer/AR model can process fixed-size windows uniformly.
    B, C, H, W = x.shape
    # print("split_to_patches input shape:", x.shape, "patch size:", s)
    if H % s != 0 or W % s != 0:
        raise ValueError("H and W must be divisible by patch size s")

    patches = rearrange(
        x, 'b c (nh s1) (nw s2) -> (b nh nw) c s1 s2', s1=s, s2=s
    )
    return patches

def merge_patches(patches, B, H, W, s):
    """
    Input shape: [B*(H/s)*(W/s), C, s, s]
    Output shape: [B, C, H, W]
    """
    nh, nw = H // s, W // s
    x = rearrange(
        patches, '(b nh nw) c s1 s2 -> b c (nh s1) (nw s2)',
        b=B, nh=nh, nw=nw, s1=s, s2=s
    )
    return x


class VQModelW_MS_qaunt_Once_Flex(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        logger.debug("Model config: %s", config)
        self.scale = config.scale
        self.AR_infer=False
        logger.debug("Using multi-scale VQ model with scale: %s", config.scale)
        using_attn = not config.wo_attn
        logger.debug("Using attention in en-decoder transformation: %s", using_attn)
        ch = 128
        self.input_size = config.patch_size
        logger.debug("Patch size for AR transformer: %s", self.input_size)
        self.encoder = AREncoder(ch=ch, ch_mult=config.encoder_ch_mult, z_channels=config.z_channels, dropout=config.dropout_p, scale=config.scale, enc_residual=config.enc_residual, using_mid_sa=using_attn, using_sa=using_attn)
        self.decoder = ARDecoder(ch=ch, ch_mult=config.decoder_ch_mult, z_channels=config.z_channels, dropout=config.dropout_p, scale=config.scale, using_mid_sa=using_attn, using_sa=using_attn)
        self.num_res_quant = config.num_res_quant
        if self.num_res_quant > 0:
            logger.debug("Using residual quantization with num_res_quant = %s", self.num_res_quant)

        self.config = config
        self.emb_loss_cnt=0

        self.quantize = VectorQuantizer_MS_input(config.codebook_size, config.codebook_embed_dim, 
                                        config.commit_loss_beta, config.entropy_loss_ratio,
                                        config.codebook_l2_norm, config.codebook_show_usage, use_SIMVQ=False, use_predictor=config.use_predictor, num_ar_per_scale=4, num_layers=12, use_patch_ck_ar=True, temp=config.tau)

        self.quant_conv_list = nn.ModuleDict()
        self.post_quant_conv_list = nn.ModuleDict()


        for s in self.scale:
            self.quant_conv_list[s] = nn.Conv2d(config.z_channels, config.codebook_embed_dim, 1)
            self.post_quant_conv_list[s] = nn.Conv2d(config.codebook_embed_dim, config.z_channels, 1)

    def encode(self, x, return_latent_shape=False, enc_wo_l2norm=False, merge=True):
        # Encoder entry used by real inference: pixels -> multi-scale features
        # -> quantizer token ids. return_latent_shape=True additionally returns
        # patch/shape metadata needed to rebuild features after entropy coding.
        B, _, H, W = x.shape
        # if (H, W) != (256,256) and not self.training: ## when testing, x is [1, C, H, W], need to be padded to 256x256
        #     x_patched, padding = self.patcher.patch(x, target_h=256, target_w=256, overlapping=64) # s1:1, s2:2, s3:3
        #     h_dict = self.encoder(x_patched)
        #     additional_padded_h = padding[0]//64
        #     additional_padded_w = padding[1]//64
        #     for i, s in enumerate(self.scale):
        #         h_dict[s] = self.patcher.unpatch(h_dict[s], overlapping=64, additional_padded_h=additional_padded_h, additional_padded_w=additional_padded_w, scale_idx = i)

        # else:
        # Backbone encoder may return one feature map or a dict of multi-scale
        # maps. Normalize to the multi-scale dict used by the quantizer.
        h_dict = self.encoder(x)
        if not isinstance(h_dict, dict):
            h_dict = {
                "s1": h_dict
            }

        quant_dict, info_dict  = {}, {}

        info_dict["unquantized"] = h_dict.copy()

        for i, s in enumerate(self.scale):
            # Project each encoder scale to codebook embedding dimension before
            # splitting into fixed AR windows.
            quant_conv_layer = self.quant_conv_list[s]

            h_dict[s] = quant_conv_layer(h_dict[s])

        ## pad to fit ar transformer, and expand into windows
        # Non-256-aligned images are padded per latent scale and split into
        # windows. mask_padded marks invalid padded tokens so entropy coding and
        # metrics do not count them as real image content.
        if (H % self.input_size != 0 or W % self.input_size != 0) and (self.config.use_predictor):
            mask_padded = []
            pad_infos = []
            padded_shapes = []
            for i, s in enumerate(self.scale):
                scale_size = self.input_size//2**(6-i)
                padded_feature, pad_info, pad_mask = pad_right_bottom(h_dict[s], scale_size)
                h_dict[s] = split_to_patches(padded_feature, scale_size) #[B*nh*nw, C, scale_size, scale_size]
                # mask_padded.append(pad_mask.view(-1))
                mask_padded.append(split_to_patches(pad_mask, scale_size).contiguous().view(-1, scale_size*scale_size)) #[1*nh*nw, scale_size* scale_size]
                pad_infos.append(pad_info)
                padded_shapes.append(padded_feature.shape[2:])
            mask_padded = torch.cat(mask_padded, dim=1).repeat(B,1) #[B*nh*nw, L], L=336
        else:
            mask_padded = None
            pad_infos = None
            padded_shapes = []
            for i, s in enumerate(self.scale):
                padded_shapes.append(h_dict[s].shape[2:])
                h_dict[s] = split_to_patches(h_dict[s], self.input_size//2**(6-i)) #[B*nh*nw, C, scale_size, scale_size]

        # Quantizer returns quantized latent tensors and info[-1], the flattened
        # codebook indices consumed by the AR entropy model.
        quant, emb_loss, info = self.quantize(list(h_dict.values()),enc_wo_l2norm=enc_wo_l2norm, mask_padded=mask_padded)

        ##  Reshape back to origin shape and Crop padding
        # Normal reconstruction can merge windows immediately. Real bitstream
        # inference calls encode(..., merge=False), keeping windowed metadata so
        # decoded indices can be restored after entropy coding.
        if merge and (H!=self.input_size or W!=self.input_size):
            for i, s in enumerate(self.scale):
                scale_size = self.input_size//2**(6-i)
                merged_feature = merge_patches(quant[i], B, padded_shapes[i][0], padded_shapes[i][1], scale_size)
                if pad_infos is not None:
                    quant[i] = crop_right_bottom(merged_feature, pad_infos[i])
                else:
                    quant[i] = merged_feature
           

        for i, s in enumerate(self.scale):
            quant_dict[f"quant_{s}"] = quant[i]
        info_dict["info"] = info
        
        # return quant_dict, emb_loss, info_dict

        if not return_latent_shape:
            return quant_dict, emb_loss, info_dict
        else:
            # shape_list stores each scale's window shape after splitting;
            # process_one_image uses it to map the flat decoded token sequence
            # back to quant_s1/quant_s2/... feature tensors.
            shape_list = []
            for s in self.scale:
                shape_list.append(h_dict[s].shape[2:])
            bs_nh_nw = quant_dict["quant_s1"].shape[0]
            return info_dict, [shape_list, mask_padded, pad_infos, padded_shapes, bs_nh_nw]
        
    def decode(self, quant_dict):
        quant_dict_dec = {k:quant_dict[f"quant_{k}"].clone() for k in self.scale}
        for s in self.scale:
            post_quant_conv_layer = self.post_quant_conv_list[s]
            if self.num_res_quant > 0:
                for m in range(self.num_res_quant):
                    quant_dict_dec[s] = quant_dict_dec[s] + quant_dict[f"quant_{s}_res_{m}"]
            quant_dict_dec[s] = post_quant_conv_layer(quant_dict_dec[s])

        dec = self.decoder(quant_dict_dec)
        return dec

    def forward(self, input):
        B,C,H,W = input.shape
        quant_dict, diff, info_dict = self.encode(input)
        diff[-1] = diff[-1]/(B*H*W)
        dec = self.decode(quant_dict)
        # diff.append(quant_dict)
        diff.append({**quant_dict, **info_dict["unquantized"]})
        return dec, diff


#################################################################################
#                              VQ Model Configs                                 #
#################################################################################
def VQ_16_32_64_quant_once(**kwargs):
    return VQModelW_MS_qaunt_Once_Flex(
        ModelArgs(
            encoder_ch_mult=[1, 1, 2, 2, 4, 4, 4],
            decoder_ch_mult=[1, 1, 2, 2, 4, 4, 4],
            scale=["s1", "s2", "s3"],
            **kwargs,
        )
    )


VQ_models = {
    "VQ-16-32-64_quant_once": VQ_16_32_64_quant_once,
}
