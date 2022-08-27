import torch
from torch import nn, optim
import numpy as np
from matplotlib import pyplot as plt

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor

class InvSqrtScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, d_model, warmup_steps) -> None:
        self.opt = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.const = 1 / np.sqrt(self.d_model)
        self.steps = 0
        super().__init__(optimizer)
    
    def get_lr(self) -> float:
        self.steps += 1
        arg1 = 1 / np.sqrt(self.steps)
        arg2 = self.steps * (self.warmup_steps ** -1.5)
        lr_factor = self.const * min(arg1, arg2)
        return [base_lr * lr_factor for base_lr in self.base_lrs]


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)