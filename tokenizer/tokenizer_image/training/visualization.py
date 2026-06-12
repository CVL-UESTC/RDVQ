import torch
from torchvision.utils import make_grid


def visualize_results(avr_results, tensorboard_logger, train_steps,
                      imgs, recons_imgs, quant_features, entropy_features=None):
    """Log scalar metrics and periodically save image grids to tensorboard."""
    for key in avr_results.keys():
        tensorboard_logger.add_scalar(key, avr_results[key], train_steps)
    vis_imgs = 4
    if train_steps % 2000 == 0:
        imgs = (imgs + 1) / 2
        recons_imgs = (recons_imgs + 1) / 2
        imgs = imgs.detach().cpu()
        recons_imgs = recons_imgs.detach().cpu()
        grid_input = make_grid(imgs[:vis_imgs], nrow=vis_imgs,
                               normalize=True, scale_each=True)
        grid_recons = make_grid(recons_imgs[:vis_imgs], nrow=vis_imgs,
                                normalize=True, scale_each=True)
        tensorboard_logger.add_image("input_imgs", grid_input, train_steps)
        tensorboard_logger.add_image("recons_imgs", grid_recons, train_steps)
        for key in quant_features.keys():
            if isinstance(quant_features[key], torch.Tensor) and quant_features[key].ndim > 2:
                feat = quant_features[key].mean(dim=1).detach().cpu()
                grid_quant = make_grid(feat[:vis_imgs].unsqueeze(1), nrow=vis_imgs,
                                       normalize=True, scale_each=True)
                tensorboard_logger.add_image(f"{key}", grid_quant, train_steps)
        if entropy_features is not None:
            for key in entropy_features.keys():
                print("Visualize:", key)
                if isinstance(entropy_features[key], torch.Tensor) and entropy_features[key].ndim > 2:
                    feat = entropy_features[key].detach().cpu()
                    grid_entropy = make_grid(feat[:vis_imgs].unsqueeze(1), nrow=vis_imgs,
                                             normalize=True, scale_each=True)
                    tensorboard_logger.add_image(f"{key}", grid_entropy, train_steps)


def get_entropy_weight_portion(train_steps, ori_steps, warm_steps):
    """Linear warmup of entropy loss weight from 0 to 1."""
    if train_steps < ori_steps:
        return 0.
    elif train_steps < ori_steps + warm_steps:
        return (train_steps - ori_steps) / warm_steps + 1e-4
    else:
        return 1.


def detach_input(input):
    """Recursively detach tensors in nested structures."""
    if input is not None:
        if isinstance(input, torch.Tensor):
            input.detach()
        elif isinstance(input, dict):
            for item in input.values():
                detach_input(item)
        elif isinstance(input, list):
            for item in input:
                detach_input(item)
