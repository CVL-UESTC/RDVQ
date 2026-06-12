from dataset.openimage import build_openimage


def build_dataset(args, **kwargs):
    if args.dataset == 'openimage':
        return build_openimage(args, **kwargs)
    raise ValueError(f'dataset {args.dataset} is not supported')
