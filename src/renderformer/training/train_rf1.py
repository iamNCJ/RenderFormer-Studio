"""RF1 training runner.

Use :mod:`renderformer.training.cli` as the public command-line entrypoint. This module
keeps the RF1 loop independent from RF2 while exposing an in-process
``main()`` for the dispatcher and source-checkout wrapper.
"""

import accelerate
from accelerate.utils import AutocastKwargs, set_seed

import os
from pathlib import Path
import yaml
import math

import torch
from torch import nn
from tqdm import tqdm
from einops import rearrange
import numpy as np
import imageio
import wandb
import lpips
from pytorch_msssim import MS_SSIM
from torchmetrics.regression import SymmetricMeanAbsolutePercentageError

import tyro
from renderformer.training.system_config import SystemConfig
from pathlib import Path

from renderformer.models.aug import aug_triangles_and_camera, trans_to_cam_coord
from renderformer.models.ray_generator import RayGenerator
from renderformer.models.tri_utils import compute_triangle_area
from renderformer.ops.apply_kernels import apply_kernels
from renderformer.data.loaders.v1 import TriangleRenderDALIDataset
from renderformer.training.cli_helper import preload_config_file_from_args
from renderformer.training.gradient_accumulation import (
    validate_v1_gradient_accumulation_steps,
)
from renderformer.training.rank_helper import get_total_cpus
from renderformer.training.train_harness import (
    auto_resume,
    build_metrics,
    build_optimizer_and_scheduler,
    log_image_mosaic_chw,
    maybe_save_ckpt,
    setup_accelerator,
)
from renderformer.training.model_loading import load_training_model


def main() -> None:
    # enable tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # load config
    config_from_file = preload_config_file_from_args(SystemConfig)
    config: SystemConfig = tyro.cli(SystemConfig, default=config_from_file)
    validate_v1_gradient_accumulation_steps(
        config.trainer_config.gradient_accumulation_steps
    )
    print(config)
    assert config.model_config.decoding_method == 'view_transformer'

    accelerator, output_dir = setup_accelerator(config)

    model = load_training_model(
        config.model_config,
        config.trainer_config.load_weights,
        strict=config.trainer_config.load_weights_strict,
        revision=config.trainer_config.load_weights_revision,
        cache_dir=config.trainer_config.load_weights_cache_dir,
        local_files_only=config.trainer_config.load_weights_local_files_only,
        subfolder=config.trainer_config.load_weights_subfolder,
    )
    model = apply_kernels(model, use_rms_norm=True, use_swiglu=True, use_rope=True, verbose=False)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'Number of parameters: {num_params / 1e6:.2f}M')
    model.transformer.grad_checkpointing = config.trainer_config.gradient_checkpointing
    print(f'Set gradient checkpointing for view-indep transformers: {config.trainer_config.gradient_checkpointing}')
    model.view_transformer.transformer.grad_checkpointing = config.trainer_config.gradient_checkpointing_view_tf
    print(f'Set gradient checkpointing for view-dep transformers: {config.trainer_config.gradient_checkpointing_view_tf}')

    parameters_to_optimize = []
    if config.trainer_config.load_weights is not None:
        for name, param in model.named_parameters():
            if name.startswith('view_transformer'):
                parameters_to_optimize.append(param)
            else:
                if config.model_config.freeze_radiosity_transformer_if_initialized:
                    param.requires_grad = False
                else:
                    parameters_to_optimize.append(param)
    else:
        parameters_to_optimize = model.parameters()

    optimizer, scheduler = build_optimizer_and_scheduler(parameters_to_optimize, config)

    # Accelerate handles both regular and DeepSpeed cases automatically
    model, optimizer, scheduler = accelerator.prepare(
        model, optimizer, scheduler
    )
    dataloader = TriangleRenderDALIDataset(
        config.dataset_config,
        batch_size=config.trainer_config.batch_size,
        num_view_limit=config.trainer_config.num_view_limit,
        num_workers=config.trainer_config.num_workers,
        num_threads=max(get_total_cpus() // (accelerator.num_processes * config.trainer_config.num_workers), 4),
        # num_threads=config.trainer_config.num_workers * 4,
        device_id=accelerator.device.index,
        single_value_texture=config.model_config.texture_encode_patch_size == 1  # for single value reflectance
    )

    # seed everything
    set_seed(config.trainer_config.seed + accelerator.process_index)
    print(
        f'Using seed {config.trainer_config.seed + accelerator.process_index}')

    start_step = auto_resume(
        accelerator,
        output_dir,
        config.trainer_config.load_ckpt,
        enabled=config.trainer_config.auto_resume,
    )

    # training loop
    iterator = iter(dataloader)
    model.train()
    device = accelerator.device
    ray_generator = RayGenerator().to(device)
    disable_autocast_kwargs = AutocastKwargs(enabled=False)
    lpips_criterion, ms_ssim_criterion, smape_criterion = build_metrics(
        device, channel=4 if config.model_config.include_alpha else 3,
    )
    pbar = tqdm(range(start_step, config.trainer_config.total_steps),
                disable=not accelerator.is_main_process, dynamic_ncols=True, initial=start_step)

    if accelerator.mixed_precision == 'fp16':
        dtype = torch.float16
        view_tf_32model = False
    elif accelerator.mixed_precision == 'bf16':
        dtype = torch.bfloat16
        view_tf_32model = False
    else:
        dtype = torch.float32
        view_tf_32model = False
    if config.trainer_config.override_view_tf32_mode:
        view_tf_32model = True
    print(f'Using dtype {dtype} for transformer training')
    # print(accelerator.scaler)

    # some constants for log encoding
    log_bias = 0.05
    log_compensation = np.log10(0.5 + log_bias)

    # main training loop
    num_skip_steps = 0
    for global_step in pbar:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            batch = next(iterator)

        # extract batch
        mvp = batch['mvp']
        tris = batch['triangles']
        valid_mask = batch['mask']
        textures = batch['texture'].type(dtype)
        vns = batch['vn']
        gt_imgs = batch['img'].type(dtype)
        c2w = batch['c2w']
        fov = batch['fov'][..., None] / 180. * torch.pi
        bs, nv = gt_imgs.shape[0], gt_imgs.shape[1]

        # preprocess textures
        with torch.no_grad(), accelerator.autocast(disable_autocast_kwargs):
            # all_diffuse 0:3, all_specular 3:6, all_roughness 6, all_normal 7:10, all_irradiance 10:13
            if config.trainer_config.emission_area_encoding:
                areas = compute_triangle_area(tris)
                textures[:, :, -3:] = textures[:, :, -3:] * areas[..., None, None, None]
            if not config.trainer_config.learn_ldr:  # log encode lighting
                textures[:, :, -3:] = torch.log10(textures[:, :, -3:] + 1.)
            if config.model_config.log_roughness_encoding:
                textures[:, :, 6] = torch.log10(textures[:, :, 6] + log_bias) - log_compensation

        # generate ray maps, disable amp
        render_resolution = config.trainer_config.img_resolution
        with torch.no_grad(), accelerator.autocast(disable_autocast_kwargs):
            # augment triangles for view-independent transformer
            if config.trainer_config.use_aug:
                tris_for_model, c2w_for_model, vns_for_model = aug_triangles_and_camera(c2w, tris, vns, rot_scale=config.trainer_config.aug_rot_scale, trans_scale=config.trainer_config.aug_trans_scale)
            else:
                tris_for_model = tris
                c2w_for_model = c2w
                vns_for_model = vns
            # transform to camera coordinate for view-dependent transformer
            if config.trainer_config.turn_to_cam_coord:
                # c2w: [B, V, 4, 4] -> [B * V, 4, 4]
                # tris: [B, N, 3, 3] -> [B * V, N, 3, 3]
                tris_for_view_tf, c2w_for_view_tf, _ = trans_to_cam_coord(
                    c2w.view(-1, 4, 4),
                    tris.repeat_interleave(nv, dim=0),
                )
                # c2w_for_view_tf: [B * V, 4, 4] -> [B, V, 4, 4]
                # tris_for_view_tf: [B * V, N, 3, 3] -> [B, V, N, 3, 3]
                c2w_for_view_tf = c2w_for_view_tf.view(bs, nv, 4, 4)
                tris_for_view_tf = tris_for_view_tf.view(bs, nv, -1, 3, 3)
            else:
                tris_for_view_tf = tris_for_model.unsqueeze(1).expand(-1, nv, -1, -1, -1)
                c2w_for_view_tf = c2w_for_model
            # form rays
            rays_o, rays_d = ray_generator(c2w_for_view_tf, fov, render_resolution)

            patch_size = config.trainer_config.crop_size
            if patch_size is not None and patch_size < render_resolution:
                # Force crop_x, crop_y to be multiples of 8 and within range
                max_crop = render_resolution - patch_size
                # Find largest multiple of 8 not greater than max_crop
                max_crop_8 = (max_crop // 8) * 8
                crop_x = torch.randint(0, max_crop_8 + 1, (bs, nv), device=device)
                crop_y = torch.randint(0, max_crop_8 + 1, (bs, nv), device=device)
                crop_x = (crop_x // 8) * 8
                crop_y = (crop_y // 8) * 8
                # # additional sampling on the boundary
                # boundary_margin = 4
                # crop_x, crop_y = torch.randint(-boundary_margin, render_resolution - patch_size + boundary_margin, (2, bs, nv), device=device)
                # crop_x = torch.clamp(crop_x, 0, render_resolution - patch_size)
                # crop_y = torch.clamp(crop_y, 0, render_resolution - patch_size)
                patch_idx = torch.arange(patch_size, device=device)
                y_idx = crop_y[:, :, None] + patch_idx[None, :]  # [N, patch_size]
                x_idx = crop_x[:, :, None] + patch_idx[None, :]  # [N, patch_size]
                # Crop along height
                gt_imgs = torch.gather(
                    gt_imgs, 2, y_idx[:, :, :, None, None].expand(-1, -1, patch_size, gt_imgs.shape[3], gt_imgs.shape[4])
                )
                rays_d = torch.gather(
                    rays_d, 2, y_idx[:, :, :, None, None].expand(-1, -1, patch_size, rays_d.shape[3], rays_d.shape[4])
                )
                # Crop along width
                gt_imgs = torch.gather(
                    gt_imgs, 3, x_idx[:, :, None, :, None].expand(-1, -1, patch_size, patch_size, gt_imgs.shape[4])
                )
                rays_d = torch.gather(
                    rays_d, 3, x_idx[:, :, None, :, None].expand(-1, -1, patch_size, patch_size, rays_d.shape[4])
                )

        # view-independent transformer
        rendered_imgs = model(
            tris_for_model.type(dtype).reshape(bs, -1, 9),
            textures,
            mvp[:, 0].type(dtype).reshape(bs, 16) * 0.,  # disable view token
            valid_mask,  # things you want is True
            vns_for_model.type(dtype).reshape(bs, -1, 9),
            rays_o=rays_o,
            rays_d=rays_d,
            tri_vpos_view_tf=tris_for_view_tf.reshape(bs, nv, -1, 9),
            tf32_view_tf=view_tf_32model,
        )  # [N, V, 3/4, H, W]

        gt_imgs = gt_imgs.permute(0, 1, 4, 2, 3)  # [N, V, H, W, 3/4] -> [N, V, 3/4, H, W]
        if not config.model_config.include_alpha:
            gt_imgs = gt_imgs[:, :, :3]

        if config.trainer_config.clamped_hdr:
            gt_imgs = gt_imgs.clamp(0, config.trainer_config.clamped_hdr_max)

        # compute loss
        with accelerator.autocast():
            # reshape to [N * V, 3/4, H, W] to directly reuse existing loss functions
            gt_imgs = gt_imgs.reshape(-1, *gt_imgs.shape[2:])
            rendered_imgs = rendered_imgs.reshape(-1, *rendered_imgs.shape[2:])
            if config.trainer_config.learn_ldr:
                gt_imgs = gt_imgs.clamp(0, 1)  # to ldr
                # gt_imgs = torch.pow(gt_imgs, 1/2.2)  # gamma correction
                loss_l1 = nn.functional.l1_loss(rendered_imgs, gt_imgs)
                loss_l2 = nn.functional.mse_loss(rendered_imgs, gt_imgs)
                loss_smape = smape_criterion(rendered_imgs, gt_imgs)
                if config.model_config.include_alpha:
                    loss_fg = nn.functional.mse_loss(rendered_imgs[:, :3] * gt_imgs[:, 3:], gt_imgs[:, :3] * gt_imgs[:, 3:])
                else:
                    loss_fg = 0.
                ms_ssim_eval = ms_ssim_criterion(
                    rendered_imgs,
                    gt_imgs,
                )
                lpips_loss = lpips_criterion(
                    rendered_imgs[:, :3],
                    gt_imgs[:, :3],
                    normalize=True
                ).mean()
                with torch.no_grad():
                    psnr = -10.0 * torch.log10(loss_l2)
                loss = loss_l1 * config.trainer_config.loss_l1_weight + \
                    loss_l2 * config.trainer_config.loss_l2_weight + \
                    (1. - ms_ssim_eval) * config.trainer_config.loss_ms_ssim_weight + \
                    lpips_loss * config.trainer_config.loss_lpips_weight + \
                    loss_fg * config.trainer_config.loss_foreground_weight + \
                    loss_smape * config.trainer_config.loss_smape_weight
            else:
                log_rgb_gt = torch.log10(gt_imgs[:, :3] + 1.)
                if config.trainer_config.linear_loss_on_proxy:
                    linear_rgb_pred = rendered_imgs[:, :3] / math.log10(2.)
                    linear_rgb_gt = log_rgb_gt / math.log10(2.)
                else:
                    linear_rgb_pred = torch.pow(10., rendered_imgs[:, :3]) - 1.
                    linear_rgb_gt = gt_imgs[:, :3]
                linear_rgb_pred_clamp_nchw = linear_rgb_pred.clamp(0., 1.)
                linear_rgb_gt_clamp_nchw = linear_rgb_gt.clamp(0., 1.)
                ms_ssim_eval = ms_ssim_criterion(
                    linear_rgb_pred_clamp_nchw,
                    linear_rgb_gt_clamp_nchw,
                )
                lpips_loss = lpips_criterion(
                    linear_rgb_pred_clamp_nchw,
                    linear_rgb_gt_clamp_nchw,
                    normalize=True
                ).mean()
                with torch.no_grad():
                    psnr = -10.0 * torch.log10(torch.nn.functional.mse_loss(
                        (torch.pow(10., rendered_imgs[:, :3]) - 1.).clamp(0., 1.),
                        gt_imgs[:, :3].clamp(0., 1.)
                    ))
                if config.trainer_config.loss_on_linear:
                    loss_l1 = nn.functional.l1_loss(linear_rgb_pred, linear_rgb_gt)
                    loss_l2 = nn.functional.mse_loss(linear_rgb_pred, linear_rgb_gt)
                    loss_smape = smape_criterion(linear_rgb_pred, linear_rgb_gt)
                else:
                    loss_l1 = nn.functional.l1_loss(rendered_imgs[:, :3], log_rgb_gt)
                    loss_l2 = nn.functional.mse_loss(rendered_imgs[:, :3], log_rgb_gt)
                    loss_smape = smape_criterion(rendered_imgs[:, :3], log_rgb_gt)
                loss = loss_l1 * config.trainer_config.loss_l1_weight + \
                    loss_l2 * config.trainer_config.loss_l2_weight + \
                    (1. - ms_ssim_eval) * config.trainer_config.loss_ms_ssim_weight + \
                    lpips_loss * config.trainer_config.loss_lpips_weight + \
                    loss_smape * config.trainer_config.loss_smape_weight

        # backprop
        accelerator.backward(loss)
        
        # Accelerate handles both regular and DeepSpeed cases automatically
        total_norm = torch.tensor(0.)
        if accelerator.sync_gradients:
            total_norm = accelerator.clip_grad_norm_(model.parameters(), 1.)
            # total_norm = total_norm.item()

        # optimizer.step()
        if type(total_norm) == torch.Tensor and total_norm.isnan():
            print(f'Skipping step {global_step} due to gradient NaN')
            num_skip_steps += 1
            if accelerator.scaler is not None:
                accelerator.scaler.update()
            print(f'Skipped step {global_step} due to gradient NaN, total skipped steps: {num_skip_steps}')
        else:
            optimizer.step()
        # optimizer.step()
        # if global_step < config.trainer_config.num_warmup_steps or total_norm < 5.0:  # skip gradient explosion but allow for the warmup stage
        #     optimizer.step()
        # else:
        #     num_skip_steps += 1
        #     if accelerator.scaler is not None:
        #         accelerator.scaler.update()
        #     print(f'Skipped step {global_step} due to gradient explosion, total skipped steps: {num_skip_steps}')
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        if global_step % config.trainer_config.log_loss_interval == 0 and accelerator.is_main_process:
            # log loss
            loss_item = loss.item()
            lr = optimizer.param_groups[0]['lr']
            # print(pbar.format_dict)
            accelerator.log({
                'train/loss': loss_item,
                'train/lr': lr,
                'train/ms_ssim': ms_ssim_eval.item(),
                'train/loss_l1': loss_l1.item(),
                'train/loss_l2': loss_l2.item(),
                'train/loss_lpips': lpips_loss.item(),
                'train/loss_smape': loss_smape.item(),
                'train/psnr': psnr.item(),
                'train/samples': bs * accelerator.num_processes * global_step * nv,
                'train/total_norm': total_norm,
                'train/skip_steps': num_skip_steps,
                'train/rate': pbar.format_dict['rate'] if pbar.format_dict['rate'] is not None else 0,
            }, step=global_step)

        pbar.set_postfix({
            'global_step': global_step,
            'loss': loss.item(),
        })

        if global_step % config.trainer_config.log_img_interval == 0 and accelerator.is_main_process:
            log_image_mosaic_chw(
                accelerator, output_dir,
                gt_chw=gt_imgs[:, :3], pred_chw=rendered_imgs[:, :3], rays_d=rays_d,
                bs=bs, nv=nv, global_step=global_step,
                learn_ldr=config.trainer_config.learn_ldr,
            )

        maybe_save_ckpt(accelerator, output_dir, config, global_step)


if __name__ == "__main__":
    main()
