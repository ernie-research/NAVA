import torch

class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, max_steps, eta_min=0.0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        if step < self.warmup_steps:
            scale = step / float(self.warmup_steps)
        else:
            progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            scale = self.eta_min + 0.5 * (1 - self.eta_min) * (1 + torch.cos(torch.tensor(progress * 3.141592653589793)))
        return [base_lr * scale for base_lr in self.base_lrs]
