import os
import torch
from torchvision import transforms
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from PIL import Image
import pyiqa
import math
import csv
import copy
# import util.misc as misc
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity

from einops import rearrange


class ImagePatcher:
    def __init__(self, patch_size=64, stride=32, pad_multiple=64):
        self.patch_size = patch_size
        self.stride = stride
        self.pad_multiple = pad_multiple
        self._meta = None

    def _pad_to_multiple(self, patch):
        _, _, h, w = patch.shape
        pad_h = (self.pad_multiple - h % self.pad_multiple) % self.pad_multiple
        pad_w = (self.pad_multiple - w % self.pad_multiple) % self.pad_multiple
        if pad_h > 0 or pad_w > 0:
            patch = F.pad(patch, (0, pad_w, 0, pad_h), mode='constant', value=0)
        return patch, pad_h, pad_w

    def to_patches(self, x):
        assert x.dim() == 4 and x.size(0) == 1
        B, C, H, W = x.shape
        w, s = self.patch_size, self.stride

        # 计算主体部分
        valid_h = ((H - w) // s) * s + w
        valid_w = ((W - w) // s) * s + w
        
        # 计算边缘部分
        edge_h = H - valid_h
        edge_w = W - valid_w
        
        # 计算边缘patch的起始位置（与主体部分保持相同的重叠策略）
        # 使用 w - s 而不是 s，以保持与主体部分相同的重叠量
        overlap = w - s
        edge_start_h = max(0, valid_h - overlap) if edge_h > 0 else valid_h
        edge_start_w = max(0, valid_w - overlap) if edge_w > 0 else valid_w

        # unfold 主体部分
        core = x[:, :, :valid_h, :valid_w]
        patches = core.unfold(2, w, s).unfold(3, w, s)
        n_h, n_w = patches.shape[2:4]
        patches = rearrange(patches, "b c nh nw ph pw -> (b nh nw) c ph pw")
        main_patches = patches.contiguous()

        edge_patches = []
        edge_positions = []

        # 下边缘 - 从edge_start_h开始，高度为w，宽度从0到valid_w
        if edge_h > 0:
            # 计算下边缘需要多少个patch
            n_edge_w = ((valid_w - w) // s) + 1 if valid_w >= w else 1
            for j in range(n_edge_w):
                x0 = j * s
                # 确保不越界
                x0 = min(x0, valid_w - w)
                y0 = edge_start_h
                patch = x[:, :, y0:y0 + w, x0:x0 + w]
                patch, pad_h, pad_w = self._pad_to_multiple(patch)
                edge_patches.append(patch)
                edge_positions.append(('bottom', y0, x0, w, w))

        # 右边缘 - 从edge_start_w开始，宽度为w，高度从0到valid_h
        if edge_w > 0:
            # 计算右边缘需要多少个patch
            n_edge_h = ((valid_h - w) // s) + 1 if valid_h >= w else 1
            for i in range(n_edge_h):
                y0 = i * s
                # 确保不越界
                y0 = min(y0, valid_h - w)
                x0 = edge_start_w
                patch = x[:, :, y0:y0 + w, x0:x0 + w]
                patch, pad_h, pad_w = self._pad_to_multiple(patch)
                edge_patches.append(patch)
                edge_positions.append(('right', y0, x0, w, w))

        # 右下角 - 从(edge_start_h, edge_start_w)开始
        if edge_h > 0 and edge_w > 0:
            y0 = edge_start_h
            x0 = edge_start_w
            patch = x[:, :, y0:y0 + w, x0:x0 + w]
            patch, pad_h, pad_w = self._pad_to_multiple(patch)
            edge_patches.append(patch)
            edge_positions.append(('corner', y0, x0, w, w))

        self._meta = dict(H=H, W=W, stride=s, patch_size=w, n_h=n_h, n_w=n_w,
                          valid_h=valid_h, valid_w=valid_w, edge_positions=edge_positions)
        return main_patches, edge_patches

    def reconstruct(self, main_patches, edge_patches):
        meta = self._meta
        H, W = meta['H'], meta['W']
        s, w = meta['stride'], meta['patch_size']
        n_h, n_w = meta['n_h'], meta['n_w']
        valid_h, valid_w = meta['valid_h'], meta['valid_w']
        edge_positions = meta['edge_positions']

        img = torch.zeros((1, 3, H, W), device=main_patches.device)
        weight = torch.zeros((1, 1, H, W), device=main_patches.device)

        main_patches = rearrange(main_patches, '(b nh nw) c ph pw -> b c nh nw ph pw', nh=n_h, nw=n_w)

        # >>> 只保留 stride 区域 <<<
        trim = (w - s) // 2
        y1, y2 = trim, trim + s
        x1, x2 = trim, trim + s

        for i in range(n_h):
            for j in range(n_w):
                y0, x0 = i * s, j * s
                patch_crop = main_patches[:, :, i, j, y1:y2, x1:x2]
                img[:, :, y0:y0 + s, x0:x0 + s] += patch_crop
                weight[:, :, y0:y0 + s, x0:x0 + s] += 1

        # 边缘部分的处理要特殊一点
        for patch, pos_info in zip(edge_patches, edge_positions):
            pos_type, y0, x0, orig_h, orig_w = pos_info
            actual_h = min(orig_h, H - y0)
            actual_w = min(orig_w, W - x0)
            patch_data = patch[:, :, :actual_h, :actual_w]
            img[:, :, y0:y0+actual_h, x0:x0+actual_w] += patch_data
            weight[:, :, y0:y0+actual_h, x0:x0+actual_w] += 1

        img /= torch.clamp(weight, min=1.0)
        return img
    
def pad(x, p):
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )
    return x_padded, (padding_left, padding_right, padding_top, padding_bottom)

def crop(x, padding):
    return F.pad(
        x,
        (-padding[0], -padding[1], -padding[2], -padding[3]),
    )

def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def compute_msssim(a, b):
    return -10 * math.log10(1-ms_ssim(a, b, data_range=1.).item())

class compute_psnr_msssim:
    def __init__(self, metric):
        assert metric in ['psnr', 'msssim']
        self.metric = metric
        
    def __call__(self, a, b):
        ret = torch.tensor(compute_psnr(a, b) if self.metric == 'psnr' else compute_msssim(a, b))
        return ret

def compute_bpp(out_net, shape=None):
    size = shape if shape else out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
              for likelihoods in out_net['likelihoods'].values()).item()

class packed_metrics:
    def __init__(self, metrics: list, device='cpu'):
        self.non_ref = ['musiq', 'clipiqa', 'niqe']
        self.compute = {}
        self.rets = {}
        self.last_ret = {}
        list_metrics = pyiqa.list_models()
        for metric in metrics:
            self.rets[metric] = []
            self.last_ret[metric] = None
            if metric in ['psnr', 'msssim']: 
                self.compute[metric] = compute_psnr_msssim(metric)
            elif metric == "lpips":
                print("lpips init")
                self.compute[metric] = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)
            elif metric in list_metrics:
                self.compute[metric] = pyiqa.create_metric(metric, device=device)

    def update(self, metric, value):
        self.rets[metric].append(value)
        if self.last_ret[metric] is not None:
            print(f'{metric} has been updated twice since the last display.')
        self.last_ret[metric] = value

    def show_last(self):
        title = ''
        for name, ret in self.last_ret.items():
            if ret is not None:
                print(f'{name}: {ret:.4f}')
                title += f'_{name}:{ret:.4f}'
                self.last_ret[name] = None
            else:
                print(f'{name} has not been updated since the last display.')
        return title

    def __call__(self, target, pred, show_last=True):
        for name in self.rets:
            if name in self.compute:
                if name in self.non_ref:
                    ret = self.compute[name](pred).item()
                else:
                    ret = self.compute[name](target, pred).item()
                self.rets[name].append(ret)
                if self.last_ret[name] is not None:
                    print(f'{name} has been updated twice since the last display.')
                self.last_ret[name] = ret
            
        if show_last:
            return self.show_last()

    def show(self, detail=False):
        title = ''
        for name, ret in self.rets.items():
            if detail: print(f'{name}({len(ret)} in total):', ret)
            print(f'average_{name}: {sum(ret) / len(ret):.4f}')
            title += f'_{name}:{sum(ret) / len(ret):.4f}'
        return title

class save_testret:
    def __init__(self, save_path, args, vae, em):
        self.save_path = save_path
        self.vae = vae
        self.em = em
        self.path = '/home/hanminghao/dataset/OI4train_kodak4test/test'
        self.data_path = save_path + '/data.csv'
        self.p = 128
        self.img_list = []
        for file in os.listdir(self.path):
            if file[-3:] in ["jpg", "png", "peg"]:
                self.img_list.append(file)

        self.iqa_metric = pyiqa.create_metric('lpips', device='cuda')
        self.normalizer = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        self.test_scales = sorted(args.test_scales)
        self.scale_range = sorted(args.scale_range)

    def __call__(self, iteration, model, ema_params, use_ema=True):
        print("start online testing")
        model.eval()
        if use_ema:
            model_state_dict = copy.deepcopy(model.state_dict())
            ema_state_dict = copy.deepcopy(model.state_dict())
            for i, (name, _value) in enumerate(model.named_parameters()):
                assert name in ema_state_dict
                ema_state_dict[name] = ema_params[i]
            print("Switch to ema")
            model.load_state_dict(ema_state_dict)

        for idx, scale in enumerate(self.test_scales):
            PSNR = 0
            Bit_rate = 0
            MS_SSIM = 0
            count = 0
            lpips = 0
            for img_name in self.img_list:
                img_path = os.path.join(self.path, img_name)
                img = transforms.ToTensor()(Image.open(img_path).convert('RGB')).cuda()
                x = img.unsqueeze(0)
                x_padded, padding = pad(x, self.p)
                x_padded = self.normalizer(x_padded)
                count += 1
                with torch.no_grad():
                    posterior = self.vae.encode(x_padded)
                    z = posterior.sample().mul_(0.2325)
                    out_net = self.em(z, custom_scale=scale)
                    z_hat = model.sample(out_net, scale)
                    out_net['x_hat'] = self.vae.decode(z_hat / 0.2325)
                    out_net['x_hat'] = (out_net['x_hat'] + 1) / 2
                    out_net['x_hat'].clamp_(0, 1)
                    out_net["x_hat"] = crop(out_net["x_hat"], padding)
                lpips += self.iqa_metric(x, out_net["x_hat"]).item()
                PSNR += compute_psnr(x, out_net["x_hat"])
                MS_SSIM += compute_msssim(x, out_net["x_hat"])
                Bit_rate += compute_bpp(out_net)
            PSNR = PSNR / count
            MS_SSIM = MS_SSIM / count
            Bit_rate = Bit_rate / count
            lpips = lpips / count
            if misc.get_rank() == 0:
                with open(self.data_path, 'a') as csvfile:
                    fieldnames = ['iter', 'psnr', 'lpips', 'ssim', 'bpp', 'scale']
                    writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
                    writer.writerow({'iter':str(iteration), 'psnr':str(PSNR), 'lpips':str(lpips), 'ssim':str(MS_SSIM), 'bpp':str(Bit_rate), 'scale':str(scale)})
        
        if use_ema:
            print("Switch back from ema")
            model.load_state_dict(model_state_dict)
        model.train()
        print("resume training")


# def save_testret(epoch, iteration, save_path, args, model, vae, em):
#     print("start online testing")
#     model.eval()
#     path = '/home/hanminghao/dataset/OI4train_kodak4test/test'
#     data_path = save_path + '/data.csv'
#     p = 128
#     img_list = []
#     for file in os.listdir(path):
#         if file[-3:] in ["jpg", "png", "peg"]:
#             img_list.append(file)

#     iqa_metric = pyiqa.create_metric('lpips', device='cuda')
#     normalizer = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

#     scale_range = sorted(args.scale_range)

#     for idx, scale in enumerate(scale_range):
#         PSNR = 0
#         Bit_rate = 0
#         MS_SSIM = 0
#         count = 0
#         lpips = 0
#         for img_name in img_list:
#             img_path = os.path.join(path, img_name)
#             img = transforms.ToTensor()(Image.open(img_path).convert('RGB')).cuda()
#             x = img.unsqueeze(0)
#             x_padded, padding = pad(x, p)
#             x_padded = normalizer(x_padded)
#             count += 1
#             with torch.no_grad():
#                 posterior = vae.encode(x_padded)
#                 z = posterior.sample().mul_(0.2325)
#                 out_net = em(z, custom_scale=scale)
#                 z_hat = model.sample(out_net['y_hat'], scale_range, scale)
#                 out_net['x_hat'] = vae.decode(z_hat / 0.2325)
#                 out_net['x_hat'] = (out_net['x_hat'] + 1) / 2
#                 out_net['x_hat'].clamp_(0, 1)
#                 out_net["x_hat"] = crop(out_net["x_hat"], padding)
#             lpips += iqa_metric(x, out_net["x_hat"]).item()
#             PSNR += compute_psnr(x, out_net["x_hat"])
#             MS_SSIM += compute_msssim(x, out_net["x_hat"])
#             Bit_rate += compute_bpp(out_net)
#         PSNR = PSNR / count
#         MS_SSIM = MS_SSIM / count
#         Bit_rate = Bit_rate / count
#         lpips = lpips / count
#         with open(data_path, 'a') as csvfile:
#             fieldnames = ['iter', 'lr', 'psnr', 'lpips', 'ssim', 'bpp', 'scale']
#             writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
#             writer.writerow({'iter':str(iteration), 'lr':str(args.learning_rate), 'psnr':str(PSNR), 'lpips':str(lpips), 'ssim':str(MS_SSIM), 'bpp':str(Bit_rate), 'scale':str(scale)})
#     model.train()
#     print("resume training")

def save_testret_forem(epoch, save_path, args, vae, em):
    print("start online testing")
    em.eval()
    path = args.dataset + '/test/'
    data_path = save_path + '/data.csv'
    p = 128
    img_list = []
    for file in os.listdir(path):
        if file[-3:] in ["jpg", "png", "peg"]:
            img_list.append(file)

    iqa_metric = pyiqa.create_metric('lpips', device='cuda')

    lmbda_range = em.lmbda if args.stage == 'variable' else [em.lmbda[0]]

    for idx, lmbda in enumerate(lmbda_range):
        PSNR = 0
        Bit_rate = 0
        MS_SSIM = 0
        count = 0
        lpips = 0
        for img_name in img_list:
            img_path = os.path.join(path, img_name)
            img = transforms.ToTensor()(Image.open(img_path).convert('RGB')).cuda()
            x = img.unsqueeze(0)
            x_padded, padding = pad(x, p)
            normalizer = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            x_padded = normalizer(x_padded)
            count += 1
            with torch.no_grad():
                posterior = vae.encode(x_padded)
                z = posterior.sample().mul_(0.2325)
                out_net = em(z, idx)
                out_net['x_hat'] = vae.decode(out_net['y_hat'] / 0.2325)
                out_net['x_hat'] = (out_net['x_hat'] + 1) / 2
                out_net['x_hat'].clamp_(0, 1)
                out_net["x_hat"] = crop(out_net["x_hat"], padding)
            lpips += iqa_metric(x, out_net["x_hat"]).item()
            PSNR += compute_psnr(x, out_net["x_hat"])
            MS_SSIM += compute_msssim(x, out_net["x_hat"])
            Bit_rate += compute_bpp(out_net)
        PSNR = PSNR / count
        MS_SSIM = MS_SSIM / count
        Bit_rate = Bit_rate / count
        lpips = lpips / count
        with open(data_path, 'a') as csvfile:
            fieldnames = ['epoch', 'lr', 'psnr', 'lpips', 'ssim', 'bpp', 'lmbda', 'q_scale']
            writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
            writer.writerow({'epoch':str(epoch), 'lr':str(args.learning_rate), 'psnr':str(PSNR), 'lpips':str(lpips), 'ssim':str(MS_SSIM), 'bpp':str(Bit_rate), 'lmbda':str(lmbda), 'q_scale':str(em.q_scale[idx].item())})
    em.train()
    print("resume training")