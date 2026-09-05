"""RF2 multi-resolution training runner (env + vol + varlen + multi-res).

Mixes H5 samples from multiple source resolutions in one batch by cropping at
load time and generating rays in the source-image coordinate system.

Differences from the prior single-resolution env+vol path (now retired):

* Uses :class:`renderformer.data.loaders.multires.TriangleRenderDALIDataset`
  which (a) samples weighted-random across multiple H5 path lists and (b) random-crops the
  GT image at load time, emitting ``crop_origin`` and ``source_resolution`` per view.
* Reads ``trainer_config.crop_size`` and forwards it to the dataset config so crops are
  consistent across both ends.
* Calls ``RayGenerator`` with ``crop_origin`` / ``crop_size`` / ``source_res`` so the
  rays are generated for the cropped patch in the source-image coordinate system. This
  drops the GPU-side patch-crop block of the base script (no longer needed).
"""

import accelerate
from accelerate.utils import AutocastKwargs, set_seed

import os
import dataclasses
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

from renderformer.models.aug import (
    aug_triangles_and_camera_and_env_and_vol,
    trans_to_cam_coord,
)
from renderformer.models.ray_generator import RayGenerator
from renderformer.data.envmap import latlong_ray_direction
from renderformer.models.tri_utils import compute_triangle_area
from renderformer.ops.apply_kernels import apply_kernels
from renderformer.data.loaders.multires import (
    TriangleRenderDALIDataset,
)
from renderformer.training.cli_helper import preload_config_file_from_args
from renderformer.training.gradient_accumulation import (
    validate_gradient_accumulation_steps,
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

    # Single source of truth for crop_size: trainer_config.crop_size overrides
    # dataset_config.crop_size if explicitly set, so we don't have to remember
    # to set it in two places.
    if config.trainer_config.crop_size is not None:
        if config.dataset_config.crop_size != config.trainer_config.crop_size:
            print(
                f"[multires] overriding dataset_config.crop_size "
                f"({config.dataset_config.crop_size}) -> trainer_config.crop_size "
                f"({config.trainer_config.crop_size})"
            )
            config.dataset_config = dataclasses.replace(
                config.dataset_config, crop_size=config.trainer_config.crop_size,
            )
    crop_size = config.dataset_config.crop_size
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

    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    dataloader = TriangleRenderDALIDataset(
        config.dataset_config,
        batch_size=config.trainer_config.batch_size,
        num_view_limit=config.trainer_config.num_view_limit,
        num_workers=config.trainer_config.num_workers,
        num_threads=max(get_total_cpus() // (accelerator.num_processes * config.trainer_config.num_workers), 4),
        device_id=accelerator.device.index,
        single_value_texture=config.model_config.texture_encode_patch_size == 1,
    )

    set_seed(config.trainer_config.seed + accelerator.process_index)
    print(f'Using seed {config.trainer_config.seed + accelerator.process_index}')

    start_step = auto_resume(
        accelerator,
        output_dir,
        config.trainer_config.load_ckpt,
        enabled=config.trainer_config.auto_resume,
    )

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

    log_bias = 0.05
    log_compensation = np.log10(0.5 + log_bias)

    env_vae = None
    is_env_video_vae = False
    if config.model_config.use_env_lighting:
        assert config.model_config.envmap_vae_model_id is not None
        from diffusers import AutoencoderKL, AutoencoderKLQwenImage
        if config.model_config.envmap_vae_model_id == 'Qwen/Qwen-Image':
            env_vae = AutoencoderKLQwenImage.from_pretrained(
                config.model_config.envmap_vae_model_id,
                torch_dtype=dtype,
                subfolder="vae",
            ).to(device)
            is_env_video_vae = True
        else:
            env_vae = AutoencoderKL.from_pretrained(
                config.model_config.envmap_vae_model_id,
                torch_dtype=dtype,
                subfolder="vae",
            ).to(device)
        env_vae.eval()
        env_vae.requires_grad_(False)

    num_skip_steps = 0
    grad_accum_steps = validate_gradient_accumulation_steps(
        config.trainer_config.gradient_accumulation_steps
    )
    for global_step in pbar:
        for _ in range(grad_accum_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                batch = next(iterator)

            with accelerator.accumulate(model):
                mvp = batch['mvp']
                tris = batch['triangles']
                valid_mask = batch['mask']
                textures = batch['texture'].type(dtype)
                vns = batch['vn']
                gt_imgs = batch['img'].type(dtype)
                c2w = batch['c2w']
                fov = batch['fov'][..., None] / 180. * torch.pi
                light_strength = batch['light_strength'].type(dtype).clamp(min=0.)
                env_map = batch.get('env_map', None)
                volume_density = batch['volume_density'].type(dtype)
                volume_position = batch['volume_position'].type(torch.float32)
                volume_rotation = batch['volume_rotation'].type(dtype)
                volume_scale = batch['volume_scale'].type(dtype)
                volume_scattering = batch['volume_scattering'].type(dtype)
                volume_absorption = batch['volume_absorption'].type(dtype)
                volume_mask = batch['volume_mask']
                crop_origin = batch['crop_origin']           # [B, V, 2] in source pixel coords
                source_resolution = batch['source_resolution']  # [B, V]
                bs, nv = gt_imgs.shape[0], gt_imgs.shape[1]

                # preprocess textures
                with torch.no_grad(), accelerator.autocast(disable_autocast_kwargs):
                    if config.trainer_config.emission_area_encoding:
                        areas = compute_triangle_area(tris)
                        textures[:, :, -3:] = textures[:, :, -3:] * areas[..., None, None, None]
                    if not config.trainer_config.learn_ldr:
                        light_strength = torch.log10(light_strength + 1.)
                    if config.model_config.log_roughness_encoding:
                        textures[:, :, 6] = torch.log10(textures[:, :, 6] + log_bias) - log_compensation

                # augment + generate ray maps for the cropped patch in source-pixel coords
                with torch.no_grad(), accelerator.autocast(disable_autocast_kwargs):
                    if config.trainer_config.use_aug:
                        env_map_input = env_map if (env_map is not None and config.model_config.use_env_lighting) else None
                        tris_for_model, c2w_for_model, vns_for_model, env_map_rot, volume_position, volume_rotation = aug_triangles_and_camera_and_env_and_vol(
                            c2w, tris, vns, env_map_input,
                            volume_position, volume_rotation, volume_mask,
                            rot_scale=config.trainer_config.aug_rot_scale,
                            trans_scale=config.trainer_config.aug_trans_scale,
                        )
                        if env_map_input is None:
                            env_map_rot = env_map
                    else:
                        tris_for_model = tris
                        c2w_for_model = c2w
                        vns_for_model = vns
                        env_map_rot = env_map

                    if config.trainer_config.turn_to_cam_coord:
                        tris_for_view_tf, c2w_for_view_tf, _ = trans_to_cam_coord(
                            c2w.view(-1, 4, 4),
                            tris.repeat_interleave(nv, dim=0),
                        )
                        c2w_for_view_tf = c2w_for_view_tf.view(bs, nv, 4, 4)
                        tris_for_view_tf = tris_for_view_tf.view(bs, nv, -1, 3, 3)
                    else:
                        tris_for_view_tf = tris_for_model.unsqueeze(1).expand(-1, nv, -1, -1, -1)
                        c2w_for_view_tf = c2w_for_model

                    # Generate rays directly at the cropped patch in source-pixel coords.
                    rays_o, rays_d = ray_generator(
                        c2w_for_view_tf, fov,
                        crop_origin=crop_origin,
                        crop_size=crop_size,
                        source_res=source_resolution,
                    )

                # Process environment map if available
                env_latent_ldr = None
                env_latent_hdr = None
                env_lighting_strength = None
                env_ray_dir = None
                if env_map_rot is not None and config.model_config.use_env_lighting:
                    with torch.no_grad(), accelerator.autocast(disable_autocast_kwargs):
                        if env_map_rot.dim() == 4 and env_map_rot.shape[1] == 3:
                            env_map_hwc = env_map_rot.permute(0, 2, 3, 1)
                        else:
                            env_map_hwc = env_map_rot

                        clamped_env = torch.clamp(env_map_hwc, 0., 1.)
                        log_env = torch.log10(env_map_hwc + 1.)
                        log_env_max = log_env.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0].max(dim=-3, keepdim=True)[0]
                        normalized_env = log_env / (log_env_max + 1e-8)
                        env_lighting_strength = log_env_max.flatten()

                        clamped_env_chw = clamped_env.permute(0, 3, 1, 2)
                        normalized_env_chw = normalized_env.permute(0, 3, 1, 2)
                        env_imgs = torch.cat([clamped_env_chw, normalized_env_chw], dim=0)
                        env_imgs = env_imgs * 2. - 1.

                        if is_env_video_vae:
                            env_imgs_vae = env_imgs[:, :, None]
                        else:
                            env_imgs_vae = env_imgs
                        latent_env = env_vae.encode(env_imgs_vae.to(dtype)).latent_dist.mean
                        if is_env_video_vae:
                            latent_env = latent_env[:, :, 0]

                        env_latent_ldr, env_latent_hdr = latent_env.chunk(2, dim=0)
                        env_latent_ldr = env_latent_ldr.to(dtype)
                        env_latent_hdr = env_latent_hdr.to(dtype)
                        env_lighting_strength = env_lighting_strength.to(dtype)

                        env_ray_dir = latlong_ray_direction(256, 512, device=env_map_rot.device)
                        env_ray_dir = env_ray_dir.unsqueeze(0).expand(bs, -1, -1, -1)
                        env_ray_dir = env_ray_dir.to(dtype)

                rendered_imgs = model(
                    tris_for_model.type(dtype).reshape(bs, -1, 9),
                    textures,
                    mvp[:, 0].type(dtype).reshape(bs, 16) * 0.,
                    valid_mask,
                    vns_for_model.type(dtype).reshape(bs, -1, 9),
                    light_strength=light_strength,
                    env_latent_ldr=env_latent_ldr,
                    env_latent_hdr=env_latent_hdr,
                    env_lighting_strength=env_lighting_strength,
                    env_ray_dir=env_ray_dir,
                    rays_o=rays_o,
                    rays_d=rays_d,
                    tri_vpos_view_tf=tris_for_view_tf.reshape(bs, nv, -1, 9),
                    tf32_view_tf=view_tf_32model,
                    use_packed_sequence=config.model_config.use_volumes,
                    volume_density=volume_density,
                    volume_position=volume_position,
                    volume_rotation=volume_rotation,
                    volume_scale=volume_scale,
                    volume_scattering=volume_scattering,
                    volume_absorption=volume_absorption,
                    volume_mask=volume_mask,
                )

                gt_imgs = gt_imgs.permute(0, 1, 4, 2, 3)
                if not config.model_config.include_alpha:
                    gt_imgs = gt_imgs[:, :, :3]

                if config.trainer_config.clamped_hdr:
                    gt_imgs = gt_imgs.clamp(0, config.trainer_config.clamped_hdr_max)

                with accelerator.autocast():
                    gt_imgs = gt_imgs.reshape(-1, *gt_imgs.shape[2:])
                    rendered_imgs = rendered_imgs.reshape(-1, *rendered_imgs.shape[2:])
                    if config.trainer_config.learn_ldr:
                        gt_imgs = gt_imgs.clamp(0, 1)
                        loss_l1 = nn.functional.l1_loss(rendered_imgs, gt_imgs)
                        loss_l2 = nn.functional.mse_loss(rendered_imgs, gt_imgs)
                        loss_smape = smape_criterion(rendered_imgs, gt_imgs)
                        if config.model_config.include_alpha:
                            loss_fg = nn.functional.mse_loss(rendered_imgs[:, :3] * gt_imgs[:, 3:], gt_imgs[:, :3] * gt_imgs[:, 3:])
                        else:
                            loss_fg = 0.
                        ms_ssim_eval = ms_ssim_criterion(rendered_imgs, gt_imgs)
                        lpips_loss = lpips_criterion(rendered_imgs[:, :3], gt_imgs[:, :3], normalize=True).mean()
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
                        ms_ssim_eval = ms_ssim_criterion(linear_rgb_pred_clamp_nchw, linear_rgb_gt_clamp_nchw)
                        lpips_loss = lpips_criterion(linear_rgb_pred_clamp_nchw, linear_rgb_gt_clamp_nchw, normalize=True).mean()
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

                accelerator.backward(loss)

                total_norm = torch.tensor(0.)
                if accelerator.sync_gradients:
                    total_norm = accelerator.clip_grad_norm_(model.parameters(), 1.)

                if type(total_norm) == torch.Tensor and total_norm.isnan():
                    print(f'Skipping step {global_step} due to gradient NaN')
                    num_skip_steps += 1
                    if accelerator.scaler is not None:
                        accelerator.scaler.update()
                    print(f'Skipped step {global_step} due to gradient NaN, total skipped steps: {num_skip_steps}')
                else:
                    optimizer.step()
                # Step the LR scheduler once per *optimizer* step, not once
                # per micro-batch. This loop runs `grad_accum_steps`
                # micro-batches per `global_step`, and the harness builds the
                # Accelerator with step_scheduler_with_optimizer=False (so the
                # AcceleratedScheduler does NOT auto-skip non-sync steps).
                # Without this gate the cosine schedule advances grad_accum x
                # too fast and, because trainer/scheduler.py wraps current_step
                # with `% (num_warmup+num_cosine)`, the whole warmup+cosine
                # cycle repeats grad_accum times over a run (e.g. 2x at
                # grad_accum=2). sync_gradients is True exactly on the last
                # micro-batch of each accumulation group.
                if accelerator.sync_gradients:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        if global_step % config.trainer_config.log_loss_interval == 0 and accelerator.is_main_process:
            loss_item = loss.item()
            lr = optimizer.param_groups[0]['lr']
            accelerator.log({
                'train/loss': loss_item,
                'train/lr': lr,
                'train/ms_ssim': ms_ssim_eval.item(),
                'train/loss_l1': loss_l1.item(),
                'train/loss_l2': loss_l2.item(),
                'train/loss_lpips': lpips_loss.item(),
                'train/loss_smape': loss_smape.item(),
                'train/psnr': psnr.item(),
                'train/samples': bs * accelerator.num_processes * global_step * nv * grad_accum_steps,
                'train/total_norm': total_norm,
                'train/skip_steps': num_skip_steps,
            }, step=global_step)

        pbar.set_postfix({'global_step': global_step, 'loss': loss.item(), 'psnr': psnr.item()})

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
