import argparse

from tokenizer.tokenizer_image.models.vq_model import VQ_models


def create_args_parser():
    """Build the argument parser for RDVQ training."""
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--data-face-path", type=str, default=None,
                        help="face datasets to improve vq model")
    parser.add_argument("--dataset", type=str, default="openimage",
                        choices=["openimage"])

    # Model
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()),
                        default="VQ-16-32-64_quant_once")
    parser.add_argument("--vq-ckpt", type=str, default=None,
                        help="ckpt path for resume training")
    parser.add_argument("--finetune", action="store_true",
                        help="finetune a pre-trained vq model")
    parser.add_argument("--load-official", action="store_true", default=False)
    parser.add_argument("--not-load-strict", action="store_true", default=False)
    parser.add_argument("--compile", action="store_true", default=False)
    parser.add_argument("--dropout-p", type=float, default=0.0)
    parser.add_argument("--use-predictor", action="store_true", default=False)
    parser.add_argument("--wo-attn", action="store_true", default=False)

    # Codebook / VQ
    parser.add_argument("--codebook-size", type=int, default=16384,
                        help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8,
                        help="codebook dimension for vector quantization")
    parser.add_argument("--codebook-l2-norm", action="store_true", default=True,
                        help="l2 norm codebook")
    parser.add_argument("--codebook-weight", type=float, default=1.0,
                        help="codebook loss weight for vector quantization")
    parser.add_argument("--commit-loss-beta", type=float, default=0.25,
                        help="commit loss beta in codebook loss")

    # Entropy / rate
    parser.add_argument("--entropy-loss-ratio", type=float, default=0.0,
                        help="entropy loss ratio in codebook loss")
    parser.add_argument("--entropy-loss-ratio-init", type=float, default=0.0,
                        help="entropy loss ratio initial value for warmup")
    parser.add_argument("--entropy-weight", type=float, default=1.0,
                        help="The entropy weight for entropy loss")
    parser.add_argument("--pretrain-entropy", action="store_true", default=False)
    parser.add_argument("--entropy-cb-pretrain", action="store_true", default=False)
    parser.add_argument("--warmup", action="store_true", default=False)
    parser.add_argument("--tau", type=float, default=0.01)

    # Image
    parser.add_argument("--image-size", type=int,
                        choices=[256, 384, 512, 640, 768, 1024, 2048],
                        default=256)
    parser.add_argument("--scale-sample", action="store_true", default=False)
    parser.add_argument("--MRS_tuning", action="store_true", default=False,
                        help="enable multi-resolution tuning from 512 to image-size")
    parser.add_argument("--no-reso-warm", action="store_true", default=False,
                        help="skip the coarse-to-fine warmup for MRS tuning")

    # Loss weights
    parser.add_argument("--reconstruction-weight", type=float, default=1.0,
                        help="reconstruction loss weight of image pixel")
    parser.add_argument("--reconstruction-loss", type=str, default="l2",
                        help="reconstruction loss type of image pixel")
    parser.add_argument("--perceptual-weight", type=float, default=1.0,
                        help="perceptual loss weight of LPIPS")
    parser.add_argument("--disc-weight", type=float, default=0.1,
                        help="discriminator loss weight for gan training")
    parser.add_argument("--disc-start", type=int, default=25000,
                        help="iteration to start discriminator training and loss")
    parser.add_argument("--disc-type", type=str,
                        choices=["patchgan", "stylegan"], default="patchgan",
                        help="discriminator type")
    parser.add_argument("--disc-loss", type=str,
                        choices=["hinge", "vanilla", "non-saturating"],
                        default="hinge", help="discriminator loss")
    parser.add_argument("--gen-loss", type=str,
                        choices=["hinge", "non-saturating"], default="hinge",
                        help="generator loss for gan training")
    parser.add_argument("--use-clip-loss", action="store_true", default=False)

    # Optimizer
    parser.add_argument("--optimizer", type=str,
                        choices=["adam", "muon"], default="adam")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--disc-lr", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.9,
                        help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--beta2", type=float, default=0.95,
                        help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--weight-decay", type=float, default=5e-2,
                        help="Weight decay to use.")
    parser.add_argument("--max-grad-norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--adam-8bit", action="store_true", default=False)

    # Training schedule
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--max_steps", type=int, default=int(7e6),
                        help="max training steps, used for early stopping")
    parser.add_argument("--global-batch-size", type=int, default=4)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str,
                        default="bf16", choices=["none", "fp16", "bf16"])

    # Freeze / fine-tune controls
    parser.add_argument("--freeze-codebook", action="store_true", default=False)
    parser.add_argument("--freeze-encoder", action="store_true", default=False)
    parser.add_argument("--freeze-decoder", action="store_true", default=False)
    parser.add_argument("--freeze-entropy", action="store_true", default=False)

    # Checkpoint / resume
    parser.add_argument("--no-local-save", action="store_true",
                        help="no save checkpoints to local path for limited disk volume")
    parser.add_argument("--results-dir", type=str, default="results_tokenizer_image")
    parser.add_argument("--ema", action="store_true",
                        help="whether using ema training")
    parser.add_argument("--reset-disc", action="store_true", default=False)
    parser.add_argument("--use-checkpoint", action="store_true", default=False)

    return parser
