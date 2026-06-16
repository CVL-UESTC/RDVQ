"""Real RDVQ bitstream evaluation with AR entropy coding."""

import argparse
import copy
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from cleanfid import fid
import torchvision.transforms as T
from tqdm import tqdm

from tokenizer.tokenizer_image.codec.latent_io import encode_pixels_to_latents, restore_indices_to_multiscale_features
from tokenizer.tokenizer_image.codec.real import SimpleRealCodec, SimpleRealCodecConfig
from evaluations.utils.evaluator import ImagePatcher, crop, packed_metrics, pad
from evaluations.utils.img_divider import split_image
from utils.bitstream_container import collect_bin_streams, save_image_bitstream_bin
from utils.real_codec_stats import build_real_patch_stats
from utils.profile_accounting import build_real_timing_summary, profile_tic, profile_toc
from utils.inference_common import (
    DEFAULT_METRICS,
    VQ_models,
    build_test_transform,
    get_zero_padding_code_index,
    join_output_parts,
    list_images,
    load_vq_model,
    metric_averages,
    parse_metrics,
    resolve_output_base,
    scalarize,
    write_metrics_json,
)


# -----------------------------------------------------------------------------
# Model and argument setup
# -----------------------------------------------------------------------------

def load_model(args):
    vq_model = load_vq_model(args, force_predictor=True, log_prefix="Restored VQ model from")
    ar_model = copy.deepcopy(vq_model.quantize.condition_entropy_small)

    # Keep the copied AR model for real coding and skip predictor calls inside predictor.
    vq_model.quantize.use_predictor = False
    vq_model.quantize.condition_entropy_small = None


    return vq_model.eval(), ar_model.eval()

        


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)

    # Dataset and output.
    parser.add_argument('-i', '--images_dir', type=str, default='../dataset/Kodak', help='Path to the image directory')
    parser.add_argument('-o', '--output_dir', type=str, default='', help='Directory where outputs are saved')
    parser.add_argument('--fid_test', type=str, default='', help='FID/KID dataset name; empty disables FID')
    parser.add_argument('--fid-ref-dir', type=str, default='', help='Reference tile directory for FID/KID when --fid_test is set')
    parser.add_argument('--dataset_name', type=str, default='', help='Dataset label used in the output path')
    parser.add_argument('--max-images', type=int, default=0, help='Only process the first N images when N > 0')
    parser.add_argument('--metrics', type=str, default=DEFAULT_METRICS, help='Comma-separated metrics to compute')

    # Runtime switches.
    parser.add_argument('--cuda', action=argparse.BooleanOptionalAction, default=torch.cuda.is_available(), help='Use CUDA when available')
    parser.add_argument('--save_img', action=argparse.BooleanOptionalAction, default=True, help='Save reconstructed images')
    parser.add_argument('--save_bin', action=argparse.BooleanOptionalAction, default=True, help='Save actual rANS payload streams as .bin files')
    parser.add_argument('--precision', type=str, default='none', choices=['none', 'fp16', 'bf16'])
    parser.add_argument('--profile-real', action='store_true', help='Collect fine-grained real coding timing breakdown')
    parser.add_argument('--verbose', action='store_true', help='Print per-image timing and split details')

    # Model defaults used by the released RDVQ checkpoints.
    parser.add_argument('--vq-model', type=str, choices=list(VQ_models.keys()), default='VQ-16-32-64_quant_once')
    parser.add_argument('--ckpt-path', type=str, default='', help='Path to the checkpoint to load')
    parser.add_argument('--codebook-size', type=int, default=4096, help='Codebook size for vector quantization')
    parser.add_argument('--codebook-embed-dim', type=int, default=32, help='Codebook embedding dimension')
    parser.add_argument('--load-strict', action=argparse.BooleanOptionalAction, default=True, help='Load the checkpoint strictly when supported')
    parser.add_argument('--load-official', action='store_true', default=False)
    parser.add_argument('--use-MS-ck-ar', action=argparse.BooleanOptionalAction, default=True)

    # Real bitstream and large-image processing.
    parser.add_argument('--transfer-slices', type=int, default=28)
    parser.add_argument('--entropy-topk', type=int, default=1024, help='K for fixed top-k/escape tensor-rANS entropy coding')
    parser.add_argument('--zero-padding', action='store_true', default=False)
    parser.add_argument('--top-k', type=int, default=-1, help='top-k value to sample with')
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature value to sample with')
    parser.add_argument('--top-p', type=float, default=1.0, help='top-p value to sample with')
    parser.add_argument('--pad-multiple', type=int, default=64, help='Pad input images to this multiple')
    parser.add_argument('--patch-size', type=int, default=256, help='Patch/window size for real bitstream inference')
    parser.add_argument('--patch-stride', type=int, default=256, help='Patch stride for large-image real bitstream inference')
    parser.add_argument('--patch-pad-multiple', type=int, default=128, help='Pad edge patches to this multiple')
    parser.add_argument('--split-threshold-pixels', type=int, default=2048 * 2048, help='Split images larger than this many padded pixels')
    parser.add_argument('--header-bits-mode', type=str, default='none', choices=['none', 'image', 'patch'], help='How to account for shape/crop header bits')
    parser.add_argument('--split-mode', type=str, default='nonoverlap', choices=['nonoverlap', 'overlap'], help='Large-image split accounting mode')

    # Optional GT tile generation for FID reference sets.
    parser.add_argument('--generate-gt-teles', action='store_true', help='Generate GT tiles for FID reference')
    return parser


# -----------------------------------------------------------------------------
# FID/KID reference-tile helpers
# -----------------------------------------------------------------------------

def resolve_fid_ref_dir(args):
    """Resolve the FID/KID reference tile directory without local defaults."""

    if not args.fid_test:
        return ""
    if args.fid_ref_dir:
        return args.fid_ref_dir
    env_ref_dir = os.environ.get("FID_REF_DIR", "").strip()
    if env_ref_dir:
        return env_ref_dir
    env_ref_root = os.environ.get("FID_REF_ROOT", "").strip()
    if env_ref_root:
        return str(Path(env_ref_root) / f"{args.fid_test}_256teles")
    raise ValueError(
        "FID/KID is enabled but no reference tile directory was provided. "
        "Set --fid-ref-dir, FID_REF_DIR, FID_REF_ROOT, or disable FID with DISABLE_FID=1."
    )



def fid_ref_has_tiles(ref_dir):
    if not ref_dir:
        return False
    ref_path = Path(ref_dir)
    if not ref_path.is_dir():
        return False
    image_exts = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
    return any(path.is_file() and path.suffix.lower() in image_exts for path in ref_path.iterdir())



# -----------------------------------------------------------------------------
# Simplified real codec image/patch stage
# -----------------------------------------------------------------------------

def _coarse_tic_ms(device):
    timing_device = torch.device(device)
    if timing_device.type == "cuda":
        torch.cuda.synchronize(timing_device)
    return time.perf_counter()


def _coarse_toc_ms(start, device):
    timing_device = torch.device(device)
    if timing_device.type == "cuda":
        torch.cuda.synchronize(timing_device)
    return (time.perf_counter() - start) * 1e3


def decode_features_to_image(vq_model, quant_fea_dict, profile, device):
    """Run the VQ decoder from restored multi-scale latent features."""

    t_vq_decode = profile_tic(profile, device)
    recons_imgs = vq_model.decode(quant_fea_dict)
    profile_toc(profile, "real.vq_decode", t_vq_decode, device)
    return recons_imgs


def process_one_image(vq_model, real_codec, imgs, args, device, Bs=1, profile=None):
    """Process one image/patch tensor with the fixed causal real codec.

    The public path is intentionally small: VQ encode, actual entropy
    roundtrip, latent restore, VQ decode, and rate accounting. Detailed VQ
    patch bookkeeping and rANS internals live behind helper modules.
    """

    global total_enc_time, total_dec_time, total_real_enc_time, total_real_dec_time
    global total_causal_enc_time, total_causal_dec_time, total_vq_dec_time, total_latent_restore_time
    H, W = imgs.shape[-2:]
    timing_device = torch.device(device)
    use_cuda_timing = timing_device.type == "cuda"
    if use_cuda_timing:
        enc_start_event = torch.cuda.Event(enable_timing=True)
        enc_end_event = torch.cuda.Event(enable_timing=True)
        enc_start_event.record()
    else:
        enc_start_time = time.perf_counter()

    latents = encode_pixels_to_latents(vq_model, imgs, profile=profile, device=device)

    if use_cuda_timing:
        enc_end_event.record()
        torch.cuda.synchronize(timing_device)
        enc_time = enc_start_event.elapsed_time(enc_end_event)
    else:
        enc_time = (time.perf_counter() - enc_start_time) * 1e3

    if use_cuda_timing:
        dec_start_event = torch.cuda.Event(enable_timing=True)
        dec_end_event = torch.cuda.Event(enable_timing=True)
        dec_start_event.record()
    else:
        dec_start_time = time.perf_counter()

    codec_result = real_codec.roundtrip(
        latents,
        transfer_slices=args.transfer_slices,
        topk=args.entropy_topk,
        temperature=args.temperature,
        sample_top_k=args.top_k,
        top_p=args.top_p,
        profile=profile,
    )
    causal_encoder_time = float(codec_result.stats.get("causal_encoder_time", 0.0)) * 1e3
    causal_decoder_time = float(codec_result.stats.get("causal_decoder_time", 0.0)) * 1e3

    restore_start = _coarse_tic_ms(device)
    quant_fea_dict = restore_indices_to_multiscale_features(
        vq_model,
        codec_result.decoded_indices,
        latents,
        codebook=real_codec.codebook,
        transfer_slices=args.transfer_slices,
        zero_padding=args.zero_padding,
        ar_mask_all=real_codec.mask_all,
        image_hw=(H, W),
        patch_batch=Bs,
        profile=profile,
        device=device,
    )
    latent_restore_time = _coarse_toc_ms(restore_start, device)

    vq_decode_start = _coarse_tic_ms(device)
    recons_imgs = decode_features_to_image(vq_model, quant_fea_dict, profile, device)
    vq_decode_time = _coarse_toc_ms(vq_decode_start, device)

    if use_cuda_timing:
        dec_end_event.record()
        torch.cuda.synchronize(timing_device)
        dec_time = dec_start_event.elapsed_time(dec_end_event)
    else:
        dec_time = (time.perf_counter() - dec_start_time) * 1e3

    total_enc_time += enc_time
    total_dec_time += dec_time
    total_causal_enc_time += causal_encoder_time
    total_causal_dec_time += causal_decoder_time
    total_vq_dec_time += vq_decode_time
    total_latent_restore_time += latent_restore_time
    total_real_enc_time += enc_time + causal_encoder_time
    total_real_dec_time += causal_decoder_time + vq_decode_time

    bpp, bpp_real, coding_stats = build_real_patch_stats(
        codec_result,
        imgs,
        header_bits_mode=args.header_bits_mode,
    )
    return recons_imgs, bpp, bpp_real, coding_stats


# Accumulate patch-level bitstream counters into image-level or run-level totals.
def add_coding_stats(total, stats):
    for key in [
        "estimated_bits",
        "payload_bits",
        "header_bits",
        "real_bits_with_header",
        "valid_token_count",
        "padded_token_count",
        "skipped_padded_token_count",
        "transmitted_token_count",
        "coding_pixels",
        "entropy_packet_count",
        "entropy_symbol_count",
        "bitstream_decoded_slice_count",
        "bitstream_decoded_token_count",
        "payload_bits_raw_rans",
        "causal_stream_header_bits",
        "causal_total_slices",
        "causal_transfer_slices",
    ]:
        total[key] = total.get(key, 0) + stats.get(key, 0)


def print_real_codec_config(opt, real_codec):
    rate_mode = "real_full_transfer" if opt.transfer_slices == 28 else "real_partial_transfer"
    print("Real codec config:")
    print(f"  real_codec_mode: {real_codec.real_codec_mode}")
    print(f"  entropy_pipeline_mode: {real_codec.pipeline_mode}")
    print(f"  entropy_backend: {real_codec.backend}")
    print(f"  entropy_coder: {real_codec.entropy_coder}")
    print(f"  entropy_topk: {opt.entropy_topk}")
    print(f"  decoder_kind: {real_codec.decoder_kind}")
    print(f"  decoder_token_source: {real_codec.decoder_token_source}")
    print(f"  transfer_slices: {opt.transfer_slices}")
    print(f"  rate_mode: {rate_mode}")
    print(f"  stream_layout: {real_codec.stream_layout}")
    print("  mode_note: fixed causal top-k tensor-rANS roundtrip; decoded tokens come from actual entropy bytes.")


# -----------------------------------------------------------------------------
# Main evaluation entry
# -----------------------------------------------------------------------------

def main():
    parser = get_parser()
    opt = parser.parse_args()
    # Stage 1: resolve runtime config, input list, checkpoint, and output path.
    pad_multiple = opt.pad_multiple
    print("Opt.vq_model:", opt.vq_model)
    global total_enc_time, total_dec_time, total_real_enc_time, total_real_dec_time
    global total_causal_enc_time, total_causal_dec_time, total_vq_dec_time, total_latent_restore_time
    total_enc_time = 0
    total_dec_time = 0
    total_real_enc_time = 0
    total_real_dec_time = 0
    total_causal_enc_time = 0
    total_causal_dec_time = 0
    total_vq_dec_time = 0
    total_latent_restore_time = 0

    image_list = list_images(opt.images_dir, opt.max_images)
    print(f"Found {len(image_list)} images in {opt.images_dir}")

    device = 'cuda' if opt.cuda else 'cpu'
    device_metrics = device
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[opt.precision]

    vq_model, ar_model = load_model(opt)
    vq_model.to(device)
    ar_model.to(device, dtype=precision)

    output_base = resolve_output_base(opt.output_dir, opt.ckpt_path)
    real_parts = ["Real"]
    if opt.zero_padding:
        real_parts.append("Zero_padding")
    real_parts.append(f"transfer_slices_{opt.transfer_slices}")
    dataset_part = opt.dataset_name or opt.fid_test
    opt.output_dir = join_output_parts(output_base, *real_parts, dataset_part)
    print("Save path:", opt.output_dir)

    # model = torch.compile(model, dynamic=True)
    vq_parameters = sum(p.numel() for p in vq_model.parameters())
    print('(VQ model) number of params (M): %.2f' % (vq_parameters / 1.e6))

    ar_parameters = sum(p.numel() for p in ar_model.parameters())
    print('(AR model) number of params (M): %.2f' % (ar_parameters / 1.e6))

    ## AR model is consists in VQ model
    print("Total number of params (M): %.2f" % ((vq_parameters+ar_parameters) / 1.e6))

    ## measure flops

    Path(opt.output_dir).mkdir(parents=True, exist_ok=True)

    rec_output_dir = Path(opt.output_dir) / 'reconstructed'
    rec_output_dir.mkdir(parents=True, exist_ok=True)

    total_time = 0
    cd_bpp_sum = 0
    cd_bpp_sum_real = 0
    cd_bpp_sum_real_with_header = 0
    coding_totals = {}
    profile_totals = {} if opt.profile_real else None
    metric_names = parse_metrics(opt.metrics)
    metrics = packed_metrics(metric_names, device=device_metrics)

    test_transform = build_test_transform()
    fid_ref_dir = resolve_fid_ref_dir(opt)
    generate_gt_teles = bool(opt.fid_test and (opt.generate_gt_teles or not fid_ref_has_tiles(fid_ref_dir)))
    if generate_gt_teles:
        print(f"Generating FID reference tiles in: {fid_ref_dir}")
    elif opt.fid_test:
        print(f"Using FID reference tiles from: {fid_ref_dir}")
    
    padding_token = get_zero_padding_code_index(vq_model)

    embedding = vq_model.quantize.embedding.weight
    if vq_model.config.codebook_l2_norm:
        embedding = F.normalize(embedding, p=2, dim=-1)
    real_codec = SimpleRealCodec(
        ar_model,
        embedding,
        padding_token,
        SimpleRealCodecConfig(
            transfer_slices=opt.transfer_slices,
            topk=opt.entropy_topk,
            temperature=opt.temperature,
            sample_top_k=opt.top_k,
            top_p=opt.top_p,
        ),
    )
    print_real_codec_config(opt, real_codec)

    # image_list = image_list[:20]

    # Stage 2: process each image through the real bitstream pipeline.
    for img_name in tqdm(image_list):
        img_path  = os.path.join(opt.images_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        # x = T.ToTensor()(img).unsqueeze(0).to(device)
        x = test_transform(img).to(device).unsqueeze(0)
        
        # x = torch.zeros((1, 3, 2048, 1024), device=device)
       
        num_pixels_ori = x.shape[2] * x.shape[3]
        x_padded, padding = pad(x, pad_multiple)

        split_img = x_padded.shape[2] * x_padded.shape[3] > opt.split_threshold_pixels
        image_coding_stats = {}
        with torch.no_grad():
            if opt.cuda:
                torch.cuda.synchronize()
            s = time.time()
            if opt.verbose:
                print("image:", img_name)
            # Large padded images are coded as patches; bpp is accumulated
            # back in original-image pixel units.
            if split_img:
                cd_bpp = 0
                cd_bpp_real = 0
                image_patcher = ImagePatcher(patch_size=opt.patch_size, stride=opt.patch_stride, pad_multiple=opt.patch_pad_multiple)
                main_patches, edge_patches = image_patcher.to_patches(x_padded) # [B, C, H, W], list of [B, C, h, w]
                ## process main patches
                main_patches_hat, bpp_i, bpp_real_i, stats_i = process_one_image(vq_model, real_codec, main_patches, opt, device, Bs=main_patches.shape[0], profile=profile_totals)
                add_coding_stats(image_coding_stats, stats_i)
                collect_bin_streams(image_coding_stats, stats_i)
                main_patch_pixels = main_patches.shape[2]*main_patches.shape[3]*main_patches.shape[0]
                cd_bpp += bpp_i*main_patch_pixels/num_pixels_ori
                cd_bpp_real += bpp_real_i*main_patch_pixels/num_pixels_ori
                ## process edge patches
                edge_patches_hat =[]
                for edge_patch in edge_patches:
                    edge_patch_hat, bpp_i, bpp_real_i, stats_i = process_one_image(vq_model, real_codec, edge_patch, opt, device, Bs=1, profile=profile_totals)
                    add_coding_stats(image_coding_stats, stats_i)
                    collect_bin_streams(image_coding_stats, stats_i)
                    edge_patch_pixels = edge_patch.shape[2]*edge_patch.shape[3]*edge_patch.shape[0]
                    cd_bpp += bpp_i*edge_patch_pixels/num_pixels_ori
                    cd_bpp_real += bpp_real_i*edge_patch_pixels/num_pixels_ori
                    edge_patches_hat.append(edge_patch_hat)
                ## combine patches
                x_hat = image_patcher.reconstruct(main_patches_hat, edge_patches_hat)
            else:
                num_pixels_input = x_padded.shape[2] * x_padded.shape[3]
                # with torch.autocast(device_type='cuda', dtype=precision):
                #     with torch.no_grad():
                x_hat, bpp_i, bpp_real, stats_i = process_one_image(vq_model, real_codec, x_padded, opt, device, profile=profile_totals)
                add_coding_stats(image_coding_stats, stats_i)
                collect_bin_streams(image_coding_stats, stats_i)
                cd_bpp = bpp_i*num_pixels_input/num_pixels_ori
                cd_bpp_real = bpp_real*num_pixels_input/num_pixels_ori

            if opt.header_bits_mode == "image":
                image_coding_stats["header_bits"] = image_coding_stats.get("header_bits", 0) + 32
            image_coding_stats["real_bits_with_header"] = image_coding_stats.get("payload_bits", 0) + image_coding_stats.get("header_bits", 0)
            if opt.save_bin:
                bin_path = save_image_bitstream_bin(opt.output_dir, img_name, image_coding_stats, transfer_slices=opt.transfer_slices, original_shape=x.shape, split_image=split_img)
                if opt.verbose and bin_path is not None:
                    print(f"Saved bin: {bin_path}")
            cd_bpp = image_coding_stats.get("estimated_bits", 0.0) / num_pixels_ori
            cd_bpp_real = image_coding_stats.get("payload_bits", 0.0) / num_pixels_ori
            cd_bpp_real_with_header = image_coding_stats.get("real_bits_with_header", 0.0) / num_pixels_ori
            add_coding_stats(coding_totals, image_coding_stats)
            cd_bpp_sum += cd_bpp
            cd_bpp_sum_real += cd_bpp_real
            cd_bpp_sum_real_with_header += cd_bpp_real_with_header
            if opt.cuda:
                torch.cuda.synchronize()
            e = time.time()
            total_time += e - s
            x_hat = crop(x_hat, padding)
            x_hat = (x_hat+1)/2
            x_hat = x_hat.clamp(0, 1)
            if opt.verbose:
                print(f"Time: {e - s}")
            if "bpp" in metric_names:
                metrics.update("bpp", cd_bpp_real)

            title = f"{img_name}"
            x= (x+1)/2 ## transform back to (0,1)
            title += metrics(x.to(device_metrics), x_hat.to(device_metrics))
        # Artifacts and FID tiles are generated after core codec timing.
        if opt.save_img:
            output = T.ToPILImage()(x_hat.squeeze(0))
            os.makedirs(f'{opt.output_dir}/x_hat/', exist_ok=True)
            output.save(f'{opt.output_dir}/x_hat/{title}.png', 'png')
            ###re compute generated FID
            # output = Image.open(f'{opt.output_dir}/x_hat/{title}.png').convert('RGB')
            if opt.fid_test:
                count=0
                tile_size = 256
                os.makedirs(f'{opt.output_dir}/tiles/', exist_ok=True)
                count = split_image(output, img_name, f'{opt.output_dir}/tiles/', 0, 0, tile_size)
                count = split_image(output, img_name, f'{opt.output_dir}/tiles/', tile_size // 2, tile_size // 2, tile_size, count)
                
            ###Create GT splits
            ###Generate GT splits
            if generate_gt_teles:
                count=0
                tile_size = 256
                input = img #T.ToPILImage()(x.squeeze(0))
                gt_split_dir = fid_ref_dir
                os.makedirs(gt_split_dir, exist_ok=True)
                # _ = split_image(input, img_name, f'{gt_split_dir}/', -padding[0], -padding[2], tile_size)
                count = split_image(input, img_name, f'{gt_split_dir}/', 0, 0, tile_size)
                count = split_image(input, img_name, f'{gt_split_dir}/', tile_size // 2, tile_size // 2, tile_size, count=count)
    
    # Stage 3: summarize rates, timing fields, optional FID/KID, and JSON output.
    cd_bpp_avg = cd_bpp_sum/len(image_list)
    cd_bpp_avg_real = cd_bpp_sum_real/len(image_list)
    cd_bpp_avg_real_with_header = cd_bpp_sum_real_with_header/len(image_list)
    score_fid = None
    score_kid = None
    title = ""
    if opt.fid_test:
        gt_split_dir = fid_ref_dir
        ##Testwith clean-fid
        score_fid = fid.compute_fid(f'{opt.output_dir}/tiles/', gt_split_dir)
        score_kid = fid.compute_kid(f'{opt.output_dir}/tiles/', gt_split_dir)

        title += f'_fid:{score_fid:.4f}_kid:{score_kid}'
        print(title)
    
    title += metrics.show()
    print("CD bpp:", cd_bpp_avg)
    title += f"_cd_bpp:{cd_bpp_avg}"
    print("CD bpp real payload:", cd_bpp_avg_real)
    title += f"_cd_bpp_real:{cd_bpp_avg_real}"
    print("CD bpp real with header:", cd_bpp_avg_real_with_header)
    total_time = total_time / len(image_list)
    total_enc_time = total_enc_time / len(image_list)
    total_dec_time = total_dec_time / len(image_list)
    total_real_enc_time = total_real_enc_time / len(image_list)
    total_real_dec_time = total_real_dec_time / len(image_list)
    total_causal_enc_time = total_causal_enc_time / len(image_list)
    total_causal_dec_time = total_causal_dec_time / len(image_list)
    total_vq_dec_time = total_vq_dec_time / len(image_list)
    total_latent_restore_time = total_latent_restore_time / len(image_list)
    summary = {
        "image_count": len(image_list),
        "cd_bpp": scalarize(cd_bpp_avg),
        "cd_bpp_real": scalarize(cd_bpp_avg_real),
        "cd_bpp_real_with_header": scalarize(cd_bpp_avg_real_with_header),
        "estimated_bits_nonpad": scalarize(coding_totals.get("estimated_bits", 0.0)),
        "estimated_bpp_aligned": scalarize(cd_bpp_avg),
        "real_payload_bits_nonpad": int(coding_totals.get("payload_bits", 0)),
        "real_bpp_payload": scalarize(cd_bpp_avg_real),
        "header_bits": int(coding_totals.get("header_bits", 0)),
        "real_bpp_with_header": scalarize(cd_bpp_avg_real_with_header),
        "valid_token_count": int(coding_totals.get("valid_token_count", 0)),
        "padded_token_count": int(coding_totals.get("padded_token_count", 0)),
        "skipped_padded_token_count": int(coding_totals.get("skipped_padded_token_count", 0)),
        "transmitted_token_count": int(coding_totals.get("transmitted_token_count", 0)),
        "entropy_packet_count": int(coding_totals.get("entropy_packet_count", 0)),
        "entropy_symbol_count": int(coding_totals.get("entropy_symbol_count", 0)),
        "bitstream_decoded_slice_count": int(coding_totals.get("bitstream_decoded_slice_count", 0)),
        "bitstream_decoded_token_count": int(coding_totals.get("bitstream_decoded_token_count", 0)),
        "real_compression_mode": real_codec.real_codec_mode,
        "real_codec_mode": real_codec.real_codec_mode,
        "entropy_pipeline_mode": real_codec.pipeline_mode,
        "entropy_backend": real_codec.backend,
        "decoder_kind": real_codec.decoder_kind,
        "decoder_token_source": real_codec.decoder_token_source,
        "payload_bits_raw_rans": int(coding_totals.get("payload_bits_raw_rans", 0)),
        "causal_stream_header_bits": int(coding_totals.get("causal_stream_header_bits", 0)),
        "causal_total_slices": int(coding_totals.get("causal_total_slices", 0)),
        "causal_transfer_slices": int(coding_totals.get("causal_transfer_slices", 0)),
        "zero_padding_code_index": int(padding_token),
        "header_bits_mode": opt.header_bits_mode,
        "split_mode": opt.split_mode,
        "patch_size": int(opt.patch_size),
        "patch_stride": int(opt.patch_stride),
        "rate_mode": "real_full_transfer" if opt.transfer_slices == 28 else "real_partial_transfer",
        "entropy_coder": real_codec.entropy_coder,
        "entropy_topk": int(opt.entropy_topk),
        "rans_backend": real_codec.backend,
        "stream_layout": real_codec.stream_layout,
        "actual_entropy_roundtrip": 1,
        "average_time": float(total_time),
        "average_enc_time": float(total_real_enc_time / 1e3),
        "average_dec_time": float(total_real_dec_time / 1e3),
        "average_real_enc_time": float(total_real_enc_time / 1e3),
        "average_real_dec_time": float(total_real_dec_time / 1e3),
        "average_enc_time_legacy": float(total_enc_time / 1e3),
        "average_dec_time_legacy": float(total_dec_time / 1e3),
        "average_vq_enc_time": float(total_enc_time / 1e3),
        "average_vq_dec_time": float(total_vq_dec_time / 1e3),
        "average_causal_entropy_encoder_time": float(total_causal_enc_time / 1e3),
        "average_causal_entropy_decoder_time": float(total_causal_dec_time / 1e3),
        "average_latent_restore_time": float(total_latent_restore_time / 1e3),
        "average_total_codec_time": float((total_real_enc_time + total_real_dec_time) / 1e3),
        "transfer_slices": int(opt.transfer_slices),
        "metrics": metric_averages(metrics),
    }
    timing_summary = build_real_timing_summary(profile_totals, len(image_list), total_enc_time / 1e3, total_dec_time / 1e3)
    summary.update(timing_summary)
    # Public timing names use the standard codec definition. Legacy boundaries
    # are retained below for comparison with older logs.
    summary["average_enc_time"] = float(total_real_enc_time / 1e3)
    summary["average_dec_time"] = float(total_real_dec_time / 1e3)
    summary["average_real_enc_time"] = float(total_real_enc_time / 1e3)
    summary["average_real_dec_time"] = float(total_real_dec_time / 1e3)
    summary["average_vq_enc_time"] = float(total_enc_time / 1e3)
    summary["average_vq_dec_time"] = float(total_vq_dec_time / 1e3)
    summary["average_causal_entropy_encoder_time"] = float(total_causal_enc_time / 1e3)
    summary["average_causal_entropy_decoder_time"] = float(total_causal_dec_time / 1e3)
    summary["average_latent_restore_time"] = float(total_latent_restore_time / 1e3)
    summary["average_total_codec_time"] = float((total_real_enc_time + total_real_dec_time) / 1e3)

    print(f'average_time: {total_time:.4f} s')
    print(f'average_enc_time: {summary["average_enc_time"]:.4f} s')
    print(f'average_dec_time: {summary["average_dec_time"]:.4f} s')
    print(f'average_vq_enc_time: {summary["average_vq_enc_time"]:.4f} s')
    print(f'average_causal_entropy_encoder_time: {summary["average_causal_entropy_encoder_time"]:.4f} s')
    print(f'average_causal_entropy_decoder_time: {summary["average_causal_entropy_decoder_time"]:.4f} s')
    print(f'average_vq_dec_time: {summary["average_vq_dec_time"]:.4f} s')
    print(f'average_latent_restore_time: {summary["average_latent_restore_time"]:.4f} s')
    print(f'average_enc_time_legacy: {summary["average_enc_time_legacy"]:.4f} s')
    print(f'average_dec_time_legacy: {summary["average_dec_time_legacy"]:.4f} s')

    if profile_totals is not None:
        profile_seconds = {key: float(value) for key, value in sorted(profile_totals.items())}
        summary["profile_seconds"] = profile_seconds
        summary["fast_cdf_fallbacks"] = int(profile_totals.get("entropy.fast_cdf_fallbacks", 0))
        summary["scalar_cdf_fallbacks"] = int(profile_totals.get("entropy.scalar_cdf_fallbacks", 0))
        summary["tensor_packets"] = int(profile_totals.get("entropy.tensor_packets", 0))
    if score_fid is not None:
        summary["fid"] = float(score_fid)
    if score_kid is not None:
        summary["kid"] = scalarize(score_kid)
    write_metrics_json(opt.output_dir, summary)
    os.makedirs(f'{opt.output_dir}/{title}', exist_ok=True)

if __name__ == '__main__':
    main()
