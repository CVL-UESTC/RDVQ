# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/train.py
#   nanoGPT: https://github.com/karpathy/nanoGPT/blob/master/model.py
import os
import sys
import time
import bisect
import random
import warnings
from copy import deepcopy

import numpy as np
import torch
import torch.distributed as dist
from bitsandbytes.optim import AdamW8bit
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint
from torchvision import transforms
from torchvision.utils import make_grid

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
os.chdir(project_root)
sys.path.insert(0, project_root)

# ── Subpackage imports ──
from tokenizer.tokenizer_image.training.build import (
    apply_freeze_policy,
    build_discriminator_optimizer,
    build_loss,
    build_model,
    build_optimizer,
    forward_with_checkpoint,
    seperate_muon,
    setup_muon,
)
from utils.ema import update_ema
from tokenizer.tokenizer_image.training.visualization import (
    detach_input,
    get_entropy_weight_portion,
    visualize_results,
)
from tokenizer.tokenizer_image.training.train_utils import (
    build_training_loader,
    create_checkpoint_payload,
    create_experiment,
    remove_stale_checkpoint,
    save_checkpoint_file,
)
from tokenizer.tokenizer_image.models.vq_model import VQ_models
from tokenizer.tokenizer_image.training.optim.scheduler import LrWdScheduler

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "8")
os.environ["MKL_NUM_THREADS"] = os.environ.get("MKL_NUM_THREADS", "8")
os.environ["OPENBLAS_NUM_THREADS"] = os.environ.get("OPENBLAS_NUM_THREADS", "8")
torch.set_num_threads(8)

warnings.filterwarnings("ignore")

#################################################################################
#                                  Training Loop                                #
#################################################################################


def build_mrs_resolution_list(max_resolution):
    """Return the 512-to-max MRS resolution ladder used during high-res tuning."""
    if max_resolution < 512:
        raise ValueError("--MRS_tuning requires --image-size >= 512.")
    resolutions = [
        int(value.item())
        for value in torch.arange(512, max_resolution + 1, 256).int()
    ]
    if resolutions[-1] != max_resolution:
        resolutions.append(int(max_resolution))
    return resolutions


def set_mrs_crop_transform(target_h, target_w):
    """Resize the shorter side before cropping so large MRS crops are valid."""
    shorter = max(int(target_h), int(target_w))
    return transforms.Compose([
        transforms.Resize(shorter, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomCrop((int(target_h), int(target_w))),
    ])


def main(args):
    """
    Trains a new model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    from utils.distributed import init_distributed_mode
    init_distributed_mode(args)
    assert args.global_batch_size % dist.get_world_size() == 0, \
        f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    use_entropy_model = False
    print(f"Codebook Size:{args.codebook_size}, Codebook Embed Dim:{args.codebook_embed_dim}")

    # Setup an experiment folder:
    logger, record, tensorboard_logger, checkpoint_dir = create_experiment(
        args, rank, use_entropy_model=use_entropy_model
    )

    # training args
    logger.info(f"{args}")

    # training env
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    logger.info(f"Using scale sample: {args.scale_sample}")
    if rank == 0:
        print("Use codebook_embed_dim:", args.codebook_embed_dim)

    # ── Build model & EMA ──────────────────────────────────────────────────
    vq_model, ema_model = build_model(args)
    # vq_model is not yet moved to device — do it after EMA copy
    ckpt = args.vq_ckpt

    if args.ema and ema_model is not None:
        ema_model = ema_model.to(device)

    vq_model = vq_model.to(device)

    if "AR_entropy" in args.vq_model:
        args.lr = args.lr * args.global_batch_size / 256
    print("Learning rate of generator:", args.lr, "Disc:", args.disc_lr)

    # ── Build loss ─────────────────────────────────────────────────────────
    vq_loss = build_loss(args, device)
    logger.info(f"Discriminator Parameters: {sum(p.numel() for p in vq_loss.discriminator.parameters()):,}")

    # ── Apply freeze policy ────────────────────────────────────────────────
    apply_freeze_policy(args, vq_model, vq_loss)

    # ── Build dataset ──────────────────────────────────────────────────────
    dataset, sampler, loader = build_training_loader(args, dist.get_world_size(), rank)
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # ── Explicit multi-resolution schedule for high-res MRS tuning ─────────
    mrs_enabled = bool(args.MRS_tuning)
    if mrs_enabled:
        largest_patch = int(args.image_size)
        start_flag = True
        width_list = build_mrs_resolution_list(largest_patch)
        height_list = list(width_list)
        C2F_iters = [5e3 * (i + 1) for i in range(len(width_list) - 1)]
        if args.no_reso_warm:
            C2F_iters = [1 * (i + 1) for i in range(len(width_list) - 1)]
        crop_transform = set_mrs_crop_transform(512, 512)
        base_batchsize = args.global_batch_size // dist.get_world_size()
        curr_bs = base_batchsize
        curr_resolution = (512, 512)
        largest_idx = 0
        resolution_change_freq = 5
        logger.info(
            "MRS tuning enabled: resolutions=%s, no_reso_warm=%s",
            width_list, args.no_reso_warm,
        )

    # ── Load checkpoint ────────────────────────────────────────────────────
    if args.vq_ckpt:
        checkpoint = torch.load(args.vq_ckpt, map_location="cpu", weights_only=False)
        if not args.load_official:
            vq_model.load_state_dict(checkpoint["model"],
                                     strict=not args.not_load_strict)
        else:
            vq_model.init_from_ckpt()

    # ── Compile ────────────────────────────────────────────────────────────
    if args.compile:
        logger.info("compiling the model... (may take several minutes)")
        vq_model = torch.compile(vq_model, mode="default")

    # ── DDP wrap ──────────────────────────────────────────────────────────
    find_unused_parameters = False
    vq_model = DDP(vq_model.to(device), device_ids=[args.gpu],
                   find_unused_parameters=find_unused_parameters)
    vq_model.train()

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer = build_optimizer(vq_model.parameters(), args)
    scheduler = None

    # ── Resume training state ─────────────────────────────────────────────
    if args.vq_ckpt:
        if args.ema and ema_model is not None and (ckpt == args.vq_ckpt) and hasattr(checkpoint, "ema"):
            ema_model.load_state_dict(checkpoint["ema"])
        if not args.not_load_strict and (ckpt == args.vq_ckpt) and not args.adam_8bit:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if not args.finetune and ckpt == args.vq_ckpt:
            train_steps = (checkpoint["steps"] if "steps" in checkpoint
                           else int(args.vq_ckpt.split("/")[-1].split(".")[0]))
            start_epoch = int(train_steps / int(len(dataset) / args.global_batch_size))
        else:
            train_steps = 0
            start_epoch = 0
        logger.info(f"Resume training from checkpoint: {args.vq_ckpt}")
        logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")
    else:
        train_steps = 0
        start_epoch = 0
        if args.ema:
            update_ema(ema_model, vq_model, decay=0)

    if args.ema:
        ema_model.eval()

    # ── Discriminator & disc optimizer ────────────────────────────────────
    if not args.pretrain_entropy and not args.freeze_decoder:
        vq_loss = DDP(vq_loss.to(device), device_ids=[args.gpu])
        vq_loss.module.discriminator.train()
        disc_model = vq_loss.module.discriminator
    else:
        for p in vq_loss.parameters():
            p.requires_grad = False
        vq_loss.eval()
        disc_model = vq_loss.discriminator

    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision == "fp16"))
    scaler_disc = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision == "fp16"))

    optimizer_disc = build_discriminator_optimizer(disc_model, args)

    if args.vq_ckpt and not args.reset_disc:
        disc_model.load_state_dict(checkpoint["discriminator"])
        optimizer_disc.load_state_dict(checkpoint["optimizer_disc"])
        del checkpoint

    scales = vq_model.module.scale
    if rank == 0:
        print("VQ model scales:", scales)

    ptdtype = {"none": torch.float32, "bf16": torch.bfloat16,
               "fp16": torch.float16}[args.mixed_precision]

    # ── Monitoring variables ──────────────────────────────────────────────
    log_steps = 0
    running_loss = 0
    start_time = time.time()

    warm_steps = 2e4
    ori_steps = train_steps
    entropy_warm_iters_ori = [0, 2e4, 2e4]
    entropy_warm_iters = [entropy_warm_iters_ori[i] + ori_steps
                          for i in range(len(entropy_warm_iters_ori))]
    best_loss = 1e3

    if mrs_enabled:
        C2F_iters = [i + train_steps for i in C2F_iters]

    logger.info(f"Training for {args.epochs} epochs...")

    # ════════════════════════════════════════════════════════════════════
    #                         Training loop
    # ════════════════════════════════════════════════════════════════════
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            imgs = x.to(device, non_blocking=True)

            # ── MRS crop update: sample a resolution from the active ladder ──
            if mrs_enabled:
                if train_steps % resolution_change_freq == 0:
                    largest_idx = bisect.bisect_right(C2F_iters, train_steps)
                    largest_idx = min(largest_idx, len(width_list) - 1)
                    if rank == 0:
                        idx_x = np.random.randint(largest_idx + 1)
                        idx_y = np.random.randint(largest_idx + 1)
                        # The 2048px recipe follows the reference implementation:
                        # avoid full 2048x2048 crops and sample 2048x1024 or
                        # 1024x2048 instead to keep the fine-tune memory bounded.
                        if (largest_patch >= 2048 and
                                width_list[idx_x] * height_list[idx_y] > 2048 * 1024):
                            large_idx = width_list.index(largest_patch)
                            small_idx = min(
                                range(len(height_list)),
                                key=lambda i: abs(height_list[i] - 1024),
                            )
                            if random.choice([True, False]):
                                idx_x = small_idx
                                idx_y = large_idx
                            else:
                                idx_x = large_idx
                                idx_y = small_idx
                        new_res_idx = torch.tensor([idx_x, idx_y],
                                                   dtype=torch.int, device="cuda")
                    else:
                        new_res_idx = torch.zeros(2, dtype=torch.int, device="cuda")
                    torch.distributed.broadcast(new_res_idx, src=0)
                    current_res_idx = new_res_idx.cpu().tolist()
                    if start_flag:
                        current_res_idx = [0, 0]
                        start_flag = False
                    curr_resolution = (width_list[current_res_idx[0]],
                                       height_list[current_res_idx[1]])
                    curr_bs = max(int(base_batchsize * (256 * 256)
                                      / (curr_resolution[0] * curr_resolution[1])), 1)
                    crop_transform = set_mrs_crop_transform(
                        curr_resolution[1], curr_resolution[0])
                    if train_steps % 500 == 0:
                        print("curr res idx:", current_res_idx)
                        print("largest idx:", largest_idx)
                        logger.info(f"Iters {train_steps}, Resolution: {curr_resolution}, "
                                    f"Batch size: {curr_bs}")
                imgs = crop_transform(imgs)[:curr_bs]

            # ── Generator forward ─────────────────────────────────────────
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(dtype=ptdtype):
                if use_entropy_model:
                    entropy_opt_index = (
                        bisect.bisect_right(entropy_warm_iters, train_steps)
                        if args.warmup else None
                    )
                    recons_imgs, codebook_loss, entropy_params = vq_model(
                        imgs, entropy_opt_index=entropy_opt_index)
                    entropy_loss = 0
                    opt_loss = 0
                    for i, key in enumerate(scales):
                        if f"entropy_loss_{key}" in entropy_params.keys():
                            if args.pretrain_entropy or not args.warmup:
                                weight_portion_i = 1.0
                            else:
                                weight_portion_i = get_entropy_weight_portion(
                                    train_steps,
                                    entropy_warm_iters[len(scales) - i - 1],
                                    warm_steps)
                            entropy_loss = entropy_loss + \
                                weight_portion_i * (entropy_params[f"entropy_loss_{key}"])
                            opt_loss = opt_loss + entropy_params[f"opt_loss_{key}"]
                else:
                    if args.use_checkpoint:
                        recons_imgs, codebook_loss = forward_with_checkpoint(
                            vq_model, imgs, dtype=ptdtype)
                    else:
                        recons_imgs, codebook_loss = vq_model(imgs)
                    entropy_params = None

                # ── Entropy loss weight warmup ────────────────────────────
                if not args.warmup:
                    if hasattr(vq_loss, "module"):
                        vq_loss.module.gt_entropy_loss_weight = args.entropy_loss_ratio
                    else:
                        vq_loss.gt_entropy_loss_weight = args.entropy_loss_ratio
                else:
                    weight_portion_i_gt = get_entropy_weight_portion(
                        train_steps, entropy_warm_iters[0], warm_steps)
                    assert args.entropy_loss_ratio >= 0 and args.entropy_loss_ratio_init >= 0
                    if args.entropy_loss_ratio > 0 and args.entropy_loss_ratio_init > 0:
                        assert args.entropy_loss_ratio_init <= args.entropy_loss_ratio
                        portion = args.entropy_loss_ratio_init / args.entropy_loss_ratio
                        weight_portion_i_gt = weight_portion_i_gt * (1 - portion) + portion
                    gt_entropy_loss_weight = weight_portion_i_gt * args.entropy_loss_ratio
                    if hasattr(vq_loss, "module"):
                        vq_loss.module.gt_entropy_loss_weight = gt_entropy_loss_weight
                    else:
                        vq_loss.gt_entropy_loss_weight = gt_entropy_loss_weight

                # ── Compute generator loss ────────────────────────────────
                loss_gen, loss_dict_gen = vq_loss(
                    codebook_loss, imgs, recons_imgs, optimizer_idx=0,
                    global_step=train_steps + 1, last_layer=None,
                    logger=logger, log_every=args.log_every,
                    entropy_loss=codebook_loss[2],
                    bpp=codebook_loss[7],
                )
                if use_entropy_model:
                    loss_dict_gen["opt_loss_rate"] = opt_loss

            # ── Backward / step generator ──────────────────────────────
            scaler.scale(loss_gen).backward()
            if args.max_grad_norm != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(vq_model.parameters(),
                                               args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            if args.ema:
                update_ema(ema_model,
                           vq_model.module._orig_mod if args.compile
                           else vq_model.module)
            if scheduler is not None:
                cur_lr, cur_wd = scheduler.step()

            # ── Discriminator step ────────────────────────────────────────
            if args.freeze_decoder or args.pretrain_entropy:
                if record is not None:
                    record.update({**loss_dict_gen,
                                   **{"train_loss": loss_gen.item()}})
            else:
                optimizer_disc.zero_grad()
                with torch.cuda.amp.autocast(dtype=ptdtype):
                    loss_disc, loss_dict_disc = vq_loss(
                        None, imgs.detach(), recons_imgs.detach(),
                        optimizer_idx=1,
                        global_step=train_steps + 1,
                        logger=logger, log_every=args.log_every,
                        entropy_loss=None,
                    )
                scaler_disc.scale(loss_disc).backward()
                if args.max_grad_norm != 0.0:
                    scaler_disc.unscale_(optimizer_disc)
                    torch.nn.utils.clip_grad_norm_(
                        vq_loss.module.discriminator.parameters(),
                        args.max_grad_norm)
                scaler_disc.step(optimizer_disc)
                scaler_disc.update()

                running_loss += loss_gen.item() + loss_disc.item()
                if record is not None:
                    record.update({**loss_dict_gen, **loss_dict_disc,
                                   **{"train_loss": loss_gen.item() + loss_disc.item(),
                                      "entropy_weight":
                                      vq_loss.module.gt_entropy_loss_weight}})

            log_steps += 1
            train_steps += 1
            vq_model.module.global_steps = train_steps

            # ── Logging / checkpoint ──────────────────────────────────────
            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, "
                            f"Train Steps/Sec: {steps_per_sec:.2f}")
                running_loss = 0
                log_steps = 0
                if record is not None:
                    avr_results = record.get()
                    visualize_results(avr_results, tensorboard_logger,
                                      train_steps, imgs, recons_imgs,
                                      codebook_loss[-1],
                                      entropy_params["entropy"]
                                      if entropy_params is not None else None)
                    record.reset()
                    if avr_results["save_loss"] < best_loss:
                        best_loss = avr_results["save_loss"]
                        if not args.no_local_save:
                            ckpt_payload = create_checkpoint_payload(
                                args, vq_model, vq_loss, optimizer,
                                optimizer_disc, train_steps,
                                ema_model if args.ema else None)
                            ckpt_path = save_checkpoint_file(
                                ckpt_payload, checkpoint_dir, "best.pt")
                            logger.info(f"Saved best model checkpoint to {ckpt_path}")
                start_time = time.time()

            # ── Periodic checkpoint ───────────────────────────────────────
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0 and not args.no_local_save:
                    ckpt_payload = create_checkpoint_payload(
                        args, vq_model, vq_loss, optimizer,
                        optimizer_disc, train_steps,
                        ema_model if args.ema else None)
                    ckpt_path = save_checkpoint_file(
                        ckpt_payload, checkpoint_dir, f"{train_steps:07d}.pt")
                    logger.info(f"Saved checkpoint to {ckpt_path}")
                    remove_stale_checkpoint(checkpoint_dir, train_steps,
                                            args.ckpt_every)
                dist.barrier()

            # ── Early stop ────────────────────────────────────────────────
            if train_steps >= args.max_steps + 10:
                logger.info(f"Reached max training steps {args.max_steps}, "
                            f"stopping training.")
                dist.barrier()
                print(f"Rank {dist.get_rank()} reached max_steps="
                      f"{args.max_steps}, exiting...")
                dist.destroy_process_group()
                exit(0)

    vq_model.eval()
    logger.info("Done!")
    dist.destroy_process_group()


if __name__ == "__main__":
    from tokenizer.tokenizer_image.training.config import create_args_parser
    parser = create_args_parser()
    args = parser.parse_args()
    main(args)
