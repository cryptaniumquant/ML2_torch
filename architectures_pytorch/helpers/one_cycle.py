"""
OneCycle Learning Rate Scheduler
Dan Mezhiborsky - @dmezh
See:
https://github.com/dmezh/convmixer-tf
https://github.com/tmp-iclr/convmixer/issues/11#issuecomment-951947395
"""

import math
from torch.optim.lr_scheduler import _LRScheduler

class OneCycleLRScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer,
        max_lr,
        total_steps,
        pct_start=0.4,
        pct_end=0.8,
        div_factor=20.0,
        last_epoch=-1,
        verbose=False
    ):
        """
        OneCycle learning rate scheduler with a warmup phase followed by cosine annealing.
        
        Args:
            optimizer: PyTorch optimizer
            max_lr: Maximum learning rate
            total_steps: Total number of training steps
            pct_start: Percentage of total steps to reach max_lr
            pct_end: Percentage of total steps to start final decay
            div_factor: Factor to divide max_lr by at the start
            last_epoch: The index of last epoch
            verbose: If True, prints a message to stdout for each update
        """
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.start_step = int(total_steps * pct_start)
        self.end_step = int(total_steps * pct_end)
        self.initial_lr = max_lr / div_factor
        self.final_lr = max_lr / (div_factor * 20.0)
        super(OneCycleLRScheduler, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                        "please use `get_last_lr()`.", UserWarning)

        step = self.last_epoch
        
        if step >= self.total_steps:
            return [self.final_lr for _ in self.base_lrs]
            
        if step < self.start_step:
            # Linear warmup
            pct = step / self.start_step
            return [self.initial_lr + (self.max_lr - self.initial_lr) * pct 
                   for _ in self.base_lrs]
        elif step < self.end_step:
            # Cosine annealing to final_lr
            pct = (step - self.start_step) / (self.end_step - self.start_step)
            cos_out = math.cos(math.pi * pct) * 0.5 + 0.5
            return [self.final_lr + (self.max_lr - self.final_lr) * cos_out 
                   for _ in self.base_lrs]
        else:
            # Linear decay to zero
            pct = (step - self.end_step) / (self.total_steps - self.end_step)
            return [self.final_lr * (1 - pct) for _ in self.base_lrs]
