import logging
import torch.distributed as dist
import torch

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


class AverageMeter:
    def __init__(self) -> None:
        super().__init__()
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Static:
  def __init__(self, keys):
    self.items = {key:AverageMeter() for key in keys}

  def update(self, items):
    for key in items.keys():
      if not hasattr(self.items, key):
         self.items[key] = AverageMeter()
      if isinstance(items[key], torch.Tensor):
        self.items[key].update(items[key].item())
      else:
        self.items[key].update(items[key])

  def get(self):
      result = {}
      for key in self.items.keys():
        result[key] = self.items[key].avg
      return result

  def reset(self):
    for key in self.items.keys():
      self.items[key].reset()