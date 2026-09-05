"""Shared training boilerplate.

The RF1 and RF2 runners each spend ~150 lines on identical setup before their
version-specific training step. This module collects those blocks so each
runner can focus on the actual step
(preprocessing, augmentation, model forward, loss).

Nothing here changes training math. Each helper is a literal extraction of
code that previously lived inline in 2-4 of the scripts. Behavior
parameterized by ``config.trainer_config`` / ``config.model_config`` knobs the
old code already read.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional, Tuple

import imageio
import lpips as lpips_pkg
import numpy as np
import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin, ProjectConfiguration
from einops import rearrange
from pytorch_msssim import MS_SSIM
from torchmetrics.regression import SymmetricMeanAbsolutePercentageError

from renderformer.training.checkpointing import auto_resume, maybe_save_ckpt
from renderformer.training.gradient_accumulation import (
    validate_gradient_accumulation_steps,
)
from renderformer.training.scheduler import get_cosine_schedule_with_warmup


def setup_accelerator(config) -> Tuple[Accelerator, Path]:
    """Create output_dir, the project config, the (optional) DeepSpeed plugin,
    the Accelerator, and call init_trackers (wandb).

    Returns ``(accelerator, output_dir)``. The training script still wires
    ``model / optimizer / scheduler`` through ``accelerator.prepare()`` itself.
    """
    grad_accum_steps = validate_gradient_accumulation_steps(
        config.trainer_config.gradient_accumulation_steps
    )

    output_dir = Path(config.trainer_config.log_dir, config.trainer_config.exp_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    project_config = ProjectConfiguration(
        project_dir=str(output_dir),
        logging_dir=str(output_dir),
        automatic_checkpoint_naming=False,
    )

    deepspeed_plugin = None
    if config.trainer_config.use_deepspeed:
        if config.trainer_config.deepspeed_config_path is None:
            deepspeed_plugin = DeepSpeedPlugin(
                hf_ds_config={
                    "zero_optimization": {
                        "stage": 2,
                        "overlap_comm": True,
                        "contiguous_gradients": True,
                        "reduce_bucket_size": config.model_config.latent_dim * config.model_config.latent_dim,
                    },
                    "bf16": {"enabled": "auto"},
                    "gradient_clipping": 1.0,
                    "train_micro_batch_size_per_gpu": config.trainer_config.batch_size,
                    "gradient_accumulation_steps": grad_accum_steps,
                }
            )
        else:
            deepspeed_plugin = DeepSpeedPlugin(config_file=config.trainer_config.deepspeed_config_path)
        print('DeepSpeed enabled with ZeRO-2 optimization')

    accelerator = Accelerator(
        log_with='wandb',
        project_config=project_config,
        mixed_precision='bf16',
        deepspeed_plugin=deepspeed_plugin,
        gradient_accumulation_steps=grad_accum_steps,
        step_scheduler_with_optimizer=False,
    )
    init_kwargs = {
        'wandb': {
            'name': config.trainer_config.exp_id,
            'id': config.trainer_config.exp_id,
            'resume': 'allow',
            'config': config.to_dict(),
            'settings': wandb.Settings(code_dir='.'),
        }
    }
    accelerator.init_trackers(
        config.trainer_config.project_id,
        config=config.to_dict(),
        init_kwargs=init_kwargs,
    )
    return accelerator, output_dir


def build_metrics(device, channel: int = 3):
    """Construct LPIPS / MS_SSIM / SMAPE criteria on ``device``.

    ``channel`` is 3 for RGB, 4 when the model includes an alpha channel.
    """
    lpips_criterion = lpips_pkg.LPIPS(net='vgg').to(device)
    ms_ssim_criterion = MS_SSIM(data_range=1., size_average=True, channel=channel).to(device)
    smape_criterion = SymmetricMeanAbsolutePercentageError().to(device)
    return lpips_criterion, ms_ssim_criterion, smape_criterion


def build_optimizer_and_scheduler(params: Iterable[torch.Tensor], config) -> Tuple[torch.optim.Optimizer, object]:
    """AdamW + cosine-with-warmup. Honors ``trainer_config.load_optimizer_state``."""
    optimizer = torch.optim.AdamW(
        params,
        lr=config.trainer_config.lr,
        weight_decay=config.trainer_config.weight_decay,
    )
    if config.trainer_config.load_optimizer_state is not None:
        optimizer.load_state_dict(
            torch.load(config.trainer_config.load_optimizer_state, weights_only=True)
        )
        print(f'Loaded optimizer state from {config.trainer_config.load_optimizer_state}')
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.trainer_config.num_warmup_steps,
        num_cosine_steps=config.trainer_config.num_cosine_steps,
    )
    return optimizer, scheduler


def _pick_rows(total: int) -> int:
    """Largest divisor of ``total`` that is <= 4. Falls back to 1 for total < 4."""
    return next(r for r in (4, 3, 2, 1) if r <= total and total % r == 0)


def log_image_mosaic_chw(
    accelerator,
    output_dir: Path,
    gt_chw: torch.Tensor,
    pred_chw: torch.Tensor,
    rays_d: Optional[torch.Tensor],
    bs: int,
    nv: int,
    global_step: int,
    learn_ldr: bool,
) -> None:
    """Log a (gt, pred[, ray_map]) mosaic to wandb + write PNGs to ``output_dir/img``.

    ``gt_chw`` and ``pred_chw`` are ``(B*V, 3, H, W)`` tensors at this point. ROWS
    is the largest divisor of ``bs * nv`` that is <= 4, so the mosaic always
    fits exactly without dropping samples even on small batches.
    """
    total = bs * nv
    rows = _pick_rows(total)
    n2 = total // rows

    vis_pred = rearrange(
        pred_chw.detach().float().cpu().numpy(),
        '(n1 n2) c h w -> (n1 h) (n2 w) c', n1=rows, n2=n2,
    )
    if not learn_ldr:
        vis_pred = np.clip(np.power(10., vis_pred) - 1., 0., 1.)
    vis_pred = (vis_pred * 255).clip(0, 255).astype('uint8')

    vis_gt = rearrange(
        gt_chw.detach().float().cpu().numpy(),
        '(n1 n2) c h w -> (n1 h) (n2 w) c', n1=rows, n2=n2,
    )
    vis_gt = (vis_gt * 255).clip(0, 255).astype('uint8')

    images = [
        wandb.Image(vis_gt, caption='gt_albedo', file_type='jpg'),
        wandb.Image(vis_pred, caption='pred_albedo', file_type='jpg'),
    ]
    if rays_d is not None:
        vis_rays = rearrange(
            rays_d.detach().float().cpu().view(-1, *rays_d.shape[2:]).numpy(),
            '(n1 n2) h w c -> (n1 h) (n2 w) c', n1=rows, n2=n2,
        )
        vis_rays = (vis_rays * 127.5 + 127.5).clip(0, 255).astype('uint8')
        images.append(wandb.Image(vis_rays, caption='ray_map', file_type='jpg'))

    accelerator.trackers[0].log({'train/images': images}, step=global_step)
    os.makedirs(output_dir / 'img', exist_ok=True)
    imageio.v3.imwrite(output_dir / 'img' / f'gt_albedo_{global_step}.png', vis_gt)
    imageio.v3.imwrite(output_dir / 'img' / f'pred_albedo_{global_step}.png', vis_pred)
