"""Forward RDVQ evaluation using likelihood-estimated rate."""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from cleanfid import fid
import torchvision.transforms as T
from tqdm import tqdm

from evaluations.utils.evaluator import crop, packed_metrics, pad
from evaluations.utils.img_divider import split_image
from utils.inference_common import (
    DEFAULT_METRICS,
    VQ_models,
    build_test_transform,
    join_output_parts,
    list_images,
    load_vq_model,
    metric_averages,
    parse_metrics,
    resolve_output_base,
    scalarize,
    write_metrics_json,
)

class ImageSpliterTh:
    def __init__(self, im, pch_size, stride, sf=1, extra_bs=1):
        '''
        Input:
            im: n x c x h x w, torch tensor, float, low-resolution image in SR
            pch_size, stride: patch setting
            sf: scale factor in image super-resolution
            pch_bs: aggregate pchs to processing, only used when inputing single image
        '''
        assert stride <= pch_size
        self.stride = stride
        self.pch_size = pch_size
        self.sf = sf
        self.extra_bs = extra_bs

        bs, chn, height, width= im.shape
        self.true_bs = bs

        self.height_starts_list = self.extract_starts(height)
        self.width_starts_list = self.extract_starts(width)
        self.starts_list = []
        for ii in self.height_starts_list:
            for jj in self.width_starts_list:
                self.starts_list.append([ii, jj])

        self.length = self.__len__()
        self.count_pchs = 0

        self.im_ori = im
        self.im_res = torch.zeros([bs, chn, height*sf, width*sf], dtype=im.dtype, device=im.device)
        self.pixel_count = torch.zeros([bs, chn, height*sf, width*sf], dtype=im.dtype, device=im.device)

    def extract_starts(self, length):
        if length <= self.pch_size:
            starts = [0,]
        else:
            starts = list(range(0, length, self.stride))
            for ii in range(len(starts)):
                if starts[ii] + self.pch_size > length:
                    starts[ii] = length - self.pch_size
            starts = sorted(set(starts), key=starts.index)
        return starts

    def __len__(self):
        return len(self.height_starts_list) * len(self.width_starts_list)

    def __iter__(self):
        return self

    def __next__(self):
        if self.count_pchs < self.length:
            index_infos = []
            current_starts_list = self.starts_list[self.count_pchs:self.count_pchs+self.extra_bs]
            for ii, (h_start, w_start) in enumerate(current_starts_list):
                w_end = w_start + self.pch_size
                h_end = h_start + self.pch_size
                current_pch = self.im_ori[:, :, h_start:h_end, w_start:w_end]
                if ii == 0:
                    pch =  current_pch
                else:
                    pch = torch.cat([pch, current_pch], dim=0)

                h_start *= self.sf
                h_end *= self.sf
                w_start *= self.sf
                w_end *= self.sf
                index_infos.append([h_start, h_end, w_start, w_end])

            self.count_pchs += len(current_starts_list)
        else:
            raise StopIteration()

        return pch, index_infos

    def update(self, pch_res, index_infos):
        '''
        Input:
            pch_res: (n*extra_bs) x c x pch_size x pch_size, float
            index_infos: [(h_start, h_end, w_start, w_end),]
        '''
        assert pch_res.shape[0] % self.true_bs == 0
        pch_list = torch.split(pch_res, self.true_bs, dim=0)
        assert len(pch_list) == len(index_infos)
        for ii, (h_start, h_end, w_start, w_end) in enumerate(index_infos):
            current_pch = pch_list[ii]
            self.im_res[:, :, h_start:h_end, w_start:w_end] +=  current_pch
            self.pixel_count[:, :, h_start:h_end, w_start:w_end] += 1

    def gather(self):
        assert torch.all(self.pixel_count != 0)
        return self.im_res.div(self.pixel_count)


def load_model(args):
    return load_vq_model(args, log_prefix="Restored from")

        
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
    parser.add_argument('--verbose', action='store_true', help='Print per-image timing and split details')

    # Model defaults used by the released RDVQ checkpoints.
    parser.add_argument('--vq-model', type=str, choices=list(VQ_models.keys()), default='VQ-16-32-64_quant_once')
    parser.add_argument('--codebook-size', type=int, default=4096, help='Codebook size for vector quantization')
    parser.add_argument('--codebook-embed-dim', type=int, default=32, help='Codebook embedding dimension')
    parser.add_argument('--ckpt-path', type=str, default='', help='Path to the checkpoint to load')
    parser.add_argument('--load-strict', action=argparse.BooleanOptionalAction, default=True, help='Load the checkpoint strictly when supported')
    parser.add_argument('--load-official', action='store_true', default=False)
    parser.add_argument('--use-predictor', action=argparse.BooleanOptionalAction, default=True)

    # Large-image processing.
    parser.add_argument('--pad-multiple', type=int, default=64, help='Pad input images to this multiple')
    parser.add_argument('--patch-size', type=int, default=256, help='Patch/window size for large-image inference')
    parser.add_argument('--patch-stride', type=int, default=256, help='Patch stride for large-image inference')
    parser.add_argument('--split-threshold-pixels', type=int, default=2048 * 2048, help='Split images larger than this many padded pixels')

    # Optional GT tile generation for FID reference sets.
    parser.add_argument('--generate-gt-teles', action='store_true', help='Generate GT tiles for FID reference')
    return parser



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

def _tqdm_out():
    """Write tqdm directly to the terminal TTY when available,
    so the progress bar stays animated even when stdout/stderr is piped."""
    try:
        return open('/dev/tty', 'w')
    except OSError:
        return sys.stderr


def main():
    parser = get_parser()
    opt = parser.parse_args()
    # 1. Resolve runtime config and load the checkpoint.
    pad_multiple = opt.pad_multiple
    print("Opt.vq_model:", opt.vq_model)
    image_list = list_images(opt.images_dir, opt.max_images)
    print(f"Found {len(image_list)} images in {opt.images_dir}")

    device = 'cuda' if opt.cuda else 'cpu'
    device_metrics = device

    model = load_model(opt).to(device)

    output_base = resolve_output_base(opt.output_dir, opt.ckpt_path)
    dataset_part = opt.dataset_name or opt.fid_test
    opt.output_dir = join_output_parts(output_base, "forward", dataset_part)
    print("Save path:", opt.output_dir)

    # model = torch.compile(model, dynamic=True)
    # if hasattr(model, "config"):
    #     num_res_quant = model.config.num_res_quant
    
    n_parameters = sum(p.numel() for p in model.parameters())
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    Path(opt.output_dir).mkdir(parents=True, exist_ok=True)

    rec_output_dir = Path(opt.output_dir) / 'reconstructed'
    rec_output_dir.mkdir(parents=True, exist_ok=True)

    total_time = 0
    cd_bpp_sum = 0
    metric_names = parse_metrics(opt.metrics)
    metrics = packed_metrics(metric_names, device=device_metrics)

    test_transform = build_test_transform()
    fid_ref_dir = resolve_fid_ref_dir(opt)
    generate_gt_teles = bool(opt.fid_test and (opt.generate_gt_teles or not fid_ref_has_tiles(fid_ref_dir)))
    if generate_gt_teles:
        print(f"Generating FID reference tiles in: {fid_ref_dir}")
    elif opt.fid_test:
        print(f"Using FID reference tiles from: {fid_ref_dir}")
    
    # 2. Run image reconstruction and metric collection.
    for img_name in tqdm(image_list, file=_tqdm_out(), mininterval=0.05):
        img_path  = os.path.join(opt.images_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        # x = T.ToTensor()(img).unsqueeze(0).to(device)
        x = test_transform(img).to(device).unsqueeze(0)
       
        num_pixels_ori = x.shape[2] * x.shape[3]
        x_padded, padding = pad(x, pad_multiple)

        split_img = x_padded.shape[2] * x_padded.shape[3] > opt.split_threshold_pixels
        with torch.no_grad():
            if opt.cuda:
                torch.cuda.synchronize()
            s = time.time()
            if opt.verbose:
                print("image:", img_name)
            if split_img:
                cd_bpp = 0
                img_spliter = ImageSpliterTh(x_padded, pch_size=opt.patch_size, stride=opt.patch_stride, sf=1, extra_bs=1)
                for patch, index_infos in img_spliter:
                    num_pixels_input = patch.shape[2] * patch.shape[3]
                    patch_hat, emb_loss = model(patch)
                    cd_bpp += (emb_loss[-2])*num_pixels_input/num_pixels_ori
                    img_spliter.update(patch_hat, index_infos)
                x_hat = img_spliter.gather().contiguous()
            else:
                num_pixels_input = x_padded.shape[2] * x_padded.shape[3]
                x_hat, emb_loss = model(x_padded)
                cd_bpp = (emb_loss[-2])*num_pixels_input/num_pixels_ori
            cd_bpp_sum += cd_bpp
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
                metrics.update("bpp", cd_bpp)

            title = f"{img_name}"
            x= (x+1)/2 ## transform back to (0,1)
            title += metrics(x.to(device_metrics), x_hat.to(device_metrics))
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

                #count = split_image(output, img_name, f'{opt.output_dir}/tiles/', -padding[0], -padding[2], tile_size)
                
            ###Create GT splits
            ###Generate GT splits
            if generate_gt_teles:
                count=0
                tile_size = 256
                input = img #T.ToPILImage()(x.squeeze(0))
                gt_split_dir = fid_ref_dir
                # gt_split_dir = f"Path_to_save_teles/{opt.fid_test}_256teles"
                # _ = split_image(input, img_name, f'{gt_split_dir}/', -padding[0], -padding[2], tile_size)
                count = split_image(input, img_name, f'{gt_split_dir}/', 0, 0, tile_size)
                count = split_image(input, img_name, f'{gt_split_dir}/', tile_size // 2, tile_size // 2, tile_size, count=count)
    
    # 3. Summarize metrics and write machine-readable results.
    cd_bpp_avg = cd_bpp_sum/len(image_list)
    score_fid = None
    score_kid = None
    title = ""
    if opt.fid_test:
        # if opt.fid_test == "div2k":
        # elif opt.fid_test == "clic":
        # else:
        #     raise NotImplementedError
        gt_tele_dir = fid_ref_dir
        ##Testwith clean-fid
        score_fid = fid.compute_fid(f'{opt.output_dir}/tiles/', gt_tele_dir)
        score_kid = fid.compute_kid(f'{opt.output_dir}/tiles/', gt_tele_dir)

        title += f'_fid:{score_fid:.4f}_kid:{score_kid}'
        print(title)
    
    title += metrics.show()
    print("CD bpp:", cd_bpp_avg)
    title += f"_cd_bpp:{cd_bpp_avg}"
    total_time = total_time / len(image_list)
    title += f"_avg_time:{total_time:.4f}"
    print(f'average_time: {total_time:.4f} s')
    summary = {
        "image_count": len(image_list),
        "cd_bpp": scalarize(cd_bpp_avg),
        "average_time": float(total_time),
        "metrics": metric_averages(metrics),
    }
    if score_fid is not None:
        summary["fid"] = float(score_fid)
    if score_kid is not None:
        summary["kid"] = scalarize(score_kid)
    write_metrics_json(opt.output_dir, summary)
    os.makedirs(f'{opt.output_dir}/{title}', exist_ok=True)
    # if args.real:

if __name__ == '__main__':
    main()
