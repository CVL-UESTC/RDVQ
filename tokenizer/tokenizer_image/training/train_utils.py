import os
from glob import glob

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from dataset.augmentation import random_crop_arr
from dataset.build import build_dataset
from utils.logger import Static, create_logger


def create_experiment(args, rank, use_entropy_model=False):
    if rank != 0:
        return create_logger(None), None, None, None

    if args.scale_sample:
        args.results_dir = f"{args.results_dir}/scale_sample"
    os.makedirs(args.results_dir, exist_ok=True)

    experiment_index = len(glob(f"{args.results_dir}/*"))
    model_string_name = args.vq_model.replace("/", "-")
    experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
    checkpoint_dir = f"{experiment_dir}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger = create_logger(experiment_dir)
    logger.info(f"Experiment directory created at {experiment_dir}")
    tensorboard_logger = SummaryWriter(log_dir=f"{experiment_dir}/tensorboard", flush_secs=30)

    entropy_list = ["entropy_loss"] if use_entropy_model else []
    bpp_list = ["bpp"] if use_entropy_model else []
    opt_loss_list = ["opt_loss_rate"] if use_entropy_model else []
    record = Static([
        "rec_loss",
        "p_loss",
        "generator_adv_loss",
        "codebook_loss0",
        "codebook_loss1",
        "discriminator_adv_loss",
        "disc_weight",
        "train_loss",
        "save_loss",
        "Gt_entropy_loss",
        "Gt_sample_entropy",
        "Gt_avg_entropy",
        "Gt_onehot_sample_entropy",
        "Cd_entropy",
    ] + entropy_list + bpp_list + opt_loss_list)
    return logger, record, tensorboard_logger, checkpoint_dir


def build_training_transform(args):
    return transforms.Compose([
        transforms.Lambda(lambda pil_image: random_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])


def build_training_loader(args, world_size, rank):
    dataset = build_dataset(args, transform=build_training_transform(args))
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.global_seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // world_size),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    return dataset, sampler, loader


def get_model_state_dict(vq_model, compile_enabled=False):
    model = vq_model.module
    if compile_enabled:
        model = model._orig_mod
    return model.state_dict()


def get_discriminator_state_dict(vq_loss):
    discriminator = vq_loss.module.discriminator if hasattr(vq_loss, "module") else vq_loss.discriminator
    return discriminator.state_dict()


def create_checkpoint_payload(args, vq_model, vq_loss, optimizer, optimizer_disc, train_steps, ema=None):
    checkpoint = {
        "model": get_model_state_dict(vq_model, compile_enabled=args.compile),
        "optimizer": optimizer.state_dict(),
        "discriminator": get_discriminator_state_dict(vq_loss),
        "optimizer_disc": optimizer_disc.state_dict(),
        "steps": train_steps,
        "args": args,
    }
    if ema is not None:
        checkpoint["ema"] = ema.state_dict()
    return checkpoint


def save_checkpoint_file(checkpoint, checkpoint_dir, file_name):
    checkpoint_path = os.path.join(checkpoint_dir, file_name)
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def remove_stale_checkpoint(checkpoint_dir, train_steps, ckpt_every, keep_last=2):
    stale_step = train_steps - keep_last * ckpt_every
    stale_path = os.path.join(checkpoint_dir, f"{stale_step:07d}.pt")
    if os.path.exists(stale_path):
        os.remove(stale_path)
