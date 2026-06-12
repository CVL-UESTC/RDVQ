"""VQ latent helpers for simplified real-bitstream inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tokenizer.tokenizer_image.models.vq_model import crop_right_bottom, merge_patches
from utils.profile_accounting import profile_tic, profile_toc


@dataclass
class EncodedLatents:
    """VQ encoder outputs needed by the real codec and VQ decoder."""

    indices: Any
    shape_list: list
    mask_padded: Any
    pad_infos: Any
    padded_shapes: Any
    batch_patches: int


    @property
    def valid_mask(self):
        return None if self.mask_padded is None else self.mask_padded == 0


def encode_pixels_to_latents(vq_model, imgs, profile=None, device=None) -> EncodedLatents:
    """Encode pixels to flattened VQ indices plus shape metadata."""

    t_vq_encode = profile_tic(profile, device)
    info_dict, infos = vq_model.encode(imgs, return_latent_shape=True, merge=False)
    profile_toc(profile, "real.vq_encode", t_vq_encode, device)
    shape_list, mask_padded, pad_infos, padded_shapes, batch_patches = infos
    return EncodedLatents(
        indices=info_dict["info"][-1],
        shape_list=shape_list,
        mask_padded=mask_padded,
        pad_infos=pad_infos,
        padded_shapes=padded_shapes,
        batch_patches=batch_patches,
    )


def restore_indices_to_multiscale_features(
    vq_model,
    decoded_indices,
    latents: EncodedLatents,
    *,
    codebook,
    transfer_slices: int,
    zero_padding: bool = False,
    ar_mask_all=None,
    image_hw=None,
    patch_batch: int = 1,
    profile=None,
    device=None,
):
    """Map decoded indices back to the multi-scale latent dict for VQ decode."""

    t_restore = profile_tic(profile, device)
    if zero_padding:
        if ar_mask_all is None:
            raise ValueError("zero_padding restore requires the AR slice mask")
        zero_pad_positions = ar_mask_all.to(decoded_indices.device) > int(transfer_slices) - 1
        quant_feature = codebook[decoded_indices]
        quant_feature[:, zero_pad_positions, :] = 0
        quant_feature = quant_feature.permute(0, 2, 1).contiguous()
    else:
        quant_feature = codebook[decoded_indices].permute(0, 2, 1).contiguous()
    profile_toc(profile, "real.embedding_restore", t_restore, device)

    t_features = profile_tic(profile, device)
    quant_fea_dict = {}
    start = 0
    b, c, _ = quant_feature.shape
    for i, pn in enumerate(latents.shape_list):
        end = start + pn[0] * pn[1]
        s = vq_model.scale[i]
        quant_fea_dict[f"quant_{s}"] = quant_feature[:, :, start:end].view(b, c, pn[0], pn[1])
        start = end
    profile_toc(profile, "real.tokens_to_multiscale_features", t_features, device)

    t_unpatch = profile_tic(profile, device)
    if latents.mask_padded is not None:
        for i, s in enumerate(vq_model.scale):
            scale_size = vq_model.input_size // 2 ** (6 - i)
            merged_feature = merge_patches(
                quant_fea_dict[f"quant_{s}"],
                patch_batch,
                latents.padded_shapes[i][0],
                latents.padded_shapes[i][1],
                scale_size,
            )
            quant_fea_dict[f"quant_{s}"] = crop_right_bottom(merged_feature, latents.pad_infos[i])
    else:
        if image_hw is None:
            raise ValueError("image_hw is required when latents were not padded")
        H, W = image_hw
        for i, s in enumerate(vq_model.scale):
            h = H // 2 ** (6 - i)
            w = W // 2 ** (6 - i)
            scale_size = vq_model.input_size // 2 ** (6 - i)
            quant_fea_dict[f"quant_{s}"] = merge_patches(
                quant_fea_dict[f"quant_{s}"],
                patch_batch,
                h,
                w,
                scale_size,
            )
    profile_toc(profile, "real.latent_unpatch_crop", t_unpatch, device)
    return quant_fea_dict
