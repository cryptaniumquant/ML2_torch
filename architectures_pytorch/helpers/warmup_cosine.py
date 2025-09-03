import math
from torch.optim.lr_scheduler import _LRScheduler

class WarmUpCosineScheduler(_LRScheduler):
    def __init__(
        self, 
        optimizer,
        total_steps,
        warmup_steps,
        warmup_learning_rate=0.0,
        last_epoch=-1,
        verbose=False
    ):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.warmup_learning_rate = warmup_learning_rate
        super(WarmUpCosineScheduler, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                        "please use `get_last_lr()`.", UserWarning)

        step = self.last_epoch

        if step > self.total_steps:
            return [0.0 for _ in self.base_lrs]

        if step < self.warmup_steps:
            # Linear warmup
            scale = step / max(1, self.warmup_steps)
            return [self.warmup_learning_rate + (base_lr - self.warmup_learning_rate) * scale
                    for base_lr in self.base_lrs]
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [base_lr * scale for base_lr in self.base_lrs]
