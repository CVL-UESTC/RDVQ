import json
import os
from pathlib import Path

import torch
from torchvision import transforms

from tokenizer.tokenizer_image.models.vq_model import VQ_models

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
DEFAULT_METRICS = "bpp,lpips,dists,musiq,clipiqa,niqe,psnr,msssim"


def list_images(images_dir, max_images=0):
    image_paths = []
    for file_name in sorted(os.listdir(images_dir)):
        if Path(file_name).suffix.lower() in IMAGE_EXTENSIONS:
            image_paths.append(file_name)
    if max_images > 0:
        image_paths = image_paths[:max_images]
    if not image_paths:
        raise ValueError(f"No image files found in {images_dir}")
    return image_paths


def parse_metrics(metrics):
    return [name.strip() for name in metrics.split(",") if name.strip()]


def build_test_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])


def load_vq_model(args, *, force_predictor=None, log_prefix="Restored from"):
    use_predictor = args.use_predictor if force_predictor is None else force_predictor
    model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim,
        use_predictor=use_predictor,
        wo_attn=True,
        patch_size=args.patch_size,
    )

    ckpt_path = args.ckpt_path
    if ckpt_path:
        if not args.load_official:
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model"]
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"{log_prefix} {ckpt_path}")
            print(f"Missing keys: {missing}")
            print(f"Unexpected keys: {unexpected}")
            del state_dict
        else:
            model.init_from_ckpt()

    return model.eval()


def resolve_output_base(output_dir, ckpt_path):
    if output_dir:
        return output_dir
    if ckpt_path is not None:
        ckpt_name = os.path.basename(ckpt_path)
        ckpt_stem = os.path.splitext(ckpt_name)[0]
        return os.path.join(os.path.dirname(ckpt_path), ckpt_stem)
    return os.path.join(".", "outputs", "Debug")


def join_output_parts(base_dir, *parts):
    clean_parts = [part for part in parts if part]
    return os.path.join(base_dir, *clean_parts) if clean_parts else base_dir


def scalarize(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().float().mean().cpu().item()
    elif hasattr(value, "item"):
        value = value.item()
    return float(value)


def get_zero_padding_code_index(vq_model):
    """Return the deterministic code index used for zero-valued latent padding."""
    quantizer = vq_model.quantize
    embedding = quantizer.embedding.weight
    zero = torch.zeros((1, quantizer.e_dim), dtype=embedding.dtype, device=embedding.device)

    if quantizer.l2_norm:
        zero = torch.nn.functional.normalize(zero, p=2, dim=-1)
        if getattr(quantizer, "use_SIMVQ", False):
            embedding = torch.nn.functional.normalize(quantizer.embedding_proj(embedding), p=2, dim=-1)
        else:
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
    elif getattr(quantizer, "use_SIMVQ", False):
        embedding = quantizer.embedding_proj(embedding)

    distances = torch.sum(zero ** 2, dim=1, keepdim=True) + torch.sum(embedding ** 2, dim=1) - 2 * zero @ embedding.t()
    return int(torch.argmin(distances, dim=1).item())


def metric_averages(metrics):
    averages = {}
    for name, values in metrics.rets.items():
        if values:
            averages[name] = scalarize(sum(values) / len(values))
    return averages


def write_metrics_json(output_dir, summary):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    json_path = Path(output_dir) / "metrics.json"
    with json_path.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"Metrics JSON: {json_path}")
    return json_path
