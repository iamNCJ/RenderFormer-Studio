from functools import partial
import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_cosine_steps: int, num_cycles: float, eta: float=0.01
):
    current_step = current_step % (num_warmup_steps + num_cosine_steps)
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_cosine_steps))
    return max(0.0, 0.5 * (1.0 + eta + math.cos(math.pi * float(num_cycles) * 2.0 * progress) * (1.0 - eta)))


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_cosine_steps: int, num_cycles: float = 0.5, eta: float = 0.01, last_epoch: int = -1
):
    """
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_cosine_steps=num_cosine_steps,
        eta=eta,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)
