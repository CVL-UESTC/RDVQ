import torch
import torch.distributed as dist
from copy import deepcopy
from bitsandbytes.optim import AdamW8bit
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint

from tokenizer.tokenizer_image.models.vq_model import VQ_models
from tokenizer.tokenizer_image.training.losses.vq_loss import VQLoss
from tokenizer.tokenizer_image.training.optim.muon import MuonWithAuxAdam
from utils.ema import update_ema, requires_grad


# ──────────────────────────────────────────────
#  Helper utilities
# ──────────────────────────────────────────────


def seperate_muon(param_list, n_dim=3):
    """Split parameters into muon (>= n_dim) and adam (< n_dim) groups."""
    head_param = param_list[0:2]
    muon_weights = [p for p in param_list[2:] if p.ndim >= n_dim]
    adam_weights = [p for p in param_list[2:] if p.ndim < n_dim]
    adam_weights = adam_weights + head_param
    return muon_weights, adam_weights


def setup_muon(param_list, lr, n_dim=2):
    """Create a MuonWithAuxAdam optimizer with two param groups."""
    muon_params, adamw_params = seperate_muon(param_list, n_dim=n_dim)
    param_groups = [
        dict(params=muon_params, use_muon=True, lr=lr * 5, weight_decay=0.01),
        dict(params=adamw_params, use_muon=False, lr=lr,
             betas=(0.9, 0.95), weight_decay=0.01),
    ]
    return MuonWithAuxAdam(param_groups)


def forward_with_checkpoint(model, x, dtype="bf16"):
    """Run model forward with gradient checkpointing."""
    def custom_forward(x):
        with autocast(dtype=dtype):
            return model(x)
    return checkpoint(custom_forward, x)


# ──────────────────────────────────────────────
#  Model, loss, optimizer, dataset construction
# ──────────────────────────────────────────────


def build_model(args):
    """Construct the VQ model from args and print parameter info.

    Returns (vq_model, ema) — ema is None unless args.ema is set.
    """
    from tokenizer.tokenizer_image.training.train_utils import build_training_loader as _bld  # noqa: F401

    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim,
        commit_loss_beta=args.commit_loss_beta,
        entropy_loss_ratio=args.entropy_loss_ratio,
        dropout_p=args.dropout_p,
        scale_sample=args.scale_sample,
        use_predictor=args.use_predictor,
        wo_attn=args.wo_attn,
        tau=args.tau,
    )
    print(f"VQ Model Parameters: {sum(p.numel() for p in vq_model.parameters()):,}")

    ema = None
    if args.ema:
        ema = deepcopy(vq_model)
        requires_grad(ema, False)
        print(f"VQ Model EMA Parameters: {sum(p.numel() for p in ema.parameters()):,}")

    return vq_model, ema


def build_loss(args, device):
    """Build the VQLoss with discriminator."""
    vq_loss = VQLoss(
        disc_start=args.disc_start,
        disc_weight=args.disc_weight,
        disc_type=args.disc_type,
        disc_loss=args.disc_loss,
        gen_adv_loss=args.gen_loss,
        image_size=args.image_size,
        perceptual_weight=args.perceptual_weight,
        reconstruction_weight=args.reconstruction_weight,
        reconstruction_loss=args.reconstruction_loss,
        codebook_weight=args.codebook_weight,
        entropy_weight=args.entropy_loss_ratio,
        using_gan_GAN_loss=(not args.freeze_decoder),
        use_clip_loss=args.use_clip_loss,
        pretrain_entropy=args.pretrain_entropy,
    ).to(device)
    print(f"Discriminator Parameters: "
          f"{sum(p.numel() for p in vq_loss.discriminator.parameters()):,}")
    return vq_loss


def apply_freeze_policy(args, vq_model, vq_loss):
    """Freeze model parts according to flags, in-place."""
    if args.pretrain_entropy:
        print("Pretrain Entropy estimation network, freeze all parameters except Entropy_est")
        for n, p in vq_model.named_parameters():
            p.requires_grad = (".condition_entropy_small" in n)
        for p in vq_loss.discriminator.parameters():
            p.requires_grad = False

    if args.freeze_codebook:
        print("Freeze codebook parameters")
        for p in vq_model.quantize.embedding.parameters():
            p.requires_grad = False

    if args.freeze_encoder:
        print("Freeze encoder parameters")
        for p in vq_model.encoder.parameters():
            p.requires_grad = False

    if args.freeze_decoder:
        print("Freeze decoder parameters")
        for p in vq_model.decoder.parameters():
            p.requires_grad = False
        for p in vq_model.post_quant_conv_list.parameters():
            p.requires_grad = False
        for p in vq_loss.discriminator.parameters():
            p.requires_grad = False

    if args.freeze_entropy:
        print("Freeze entropy parameters")
        for n, p in vq_model.named_parameters():
            if "Entropy_est" in n:
                p.requires_grad = False


def build_optimizer(model_params, args):
    """Create the generator optimizer (adam or muon)."""
    if args.optimizer == "adam":
        print(f"using 8bit optimizer: {args.adam_8bit}")
        if args.adam_8bit:
            return AdamW8bit(model_params, lr=args.lr, betas=(args.beta1, args.beta2))
        else:
            return torch.optim.Adam(model_params, lr=args.lr, betas=(args.beta1, args.beta2))
    elif args.optimizer == "muon":
        return setup_muon(list(model_params), lr=args.lr, n_dim=3)
    else:
        raise NotImplementedError(f"Optimizer {args.optimizer} not implemented")


def build_discriminator_optimizer(disc_model, args):
    """Create the discriminator optimizer."""
    if args.optimizer == "adam":
        return torch.optim.Adam(disc_model.parameters(), lr=args.disc_lr,
                                betas=(args.beta1, args.beta2))
    elif args.optimizer == "muon":
        return setup_muon(list(disc_model.parameters()), lr=args.disc_lr, n_dim=3)
    else:
        raise NotImplementedError(f"Optimizer {args.optimizer} not implemented")
