import math
from torch.optim import Optimizer

class LrWdScheduler:
    def __init__(self, optimizer: Optimizer, sche_type: str, peak_lr: float, wd: float, wd_end: float,
                 max_it: int, wp_it: int, wp0=0.005, wpe=0.001):
        self.optimizer = optimizer
        self.sche_type = sche_type
        self.peak_lr = peak_lr
        self.wd = wd
        self.wd_end = wd_end
        self.max_it = max_it
        self.wp_it = wp_it
        self.wp0 = wp0
        self.wpe = wpe
        self.cur_it = 0

    def step(self):
        cur_it = self.cur_it
        wp_it = self.wp_it

        # ---- learning rate ----
        if cur_it < wp_it:
            cur_lr = self.wp0 + (1 - self.wp0) * cur_it / wp_it
        else:
            pasd = (cur_it - wp_it) / (self.max_it - 1 - wp_it)  # [0, 1]
            rest = 1 - pasd
            if self.sche_type == 'cos':
                cur_lr = self.wpe + (1 - self.wpe) * (0.5 + 0.5 * math.cos(math.pi * pasd))
            elif self.sche_type == 'lin':
                T = 0.15
                max_rest = 1 - T
                if pasd < T:
                    cur_lr = 1
                else:
                    cur_lr = self.wpe + (1 - self.wpe) * rest / max_rest
            elif self.sche_type == 'lin0':
                cur_lr = self.wpe + (1 - self.wpe) * rest
            else:
                raise NotImplementedError

        cur_lr *= self.peak_lr

        # ---- weight decay ----
        pasd = cur_it / (self.max_it - 1)
        cur_wd = self.wd_end + (self.wd - self.wd_end) * (0.5 + 0.5 * math.cos(math.pi * pasd))

        # ---- apply to optimizer ----
        for pg in self.optimizer.param_groups:
            pg['lr'] = cur_lr * pg.get('lr_sc', 1.0)
            if pg.get('wd_sc', 1.0) > 0:
                pg['weight_decay'] = cur_wd * pg['wd_sc']

        self.cur_it += 1
        return cur_lr, cur_wd