import dataclasses
from typing import Literal, Union, List


@dataclasses.dataclass(frozen=True)
class TrainerConfig:
    # seed
    seed: int = 3407

    # dataloader
    batch_size: int = 8
    num_view_limit: int = 2
    num_workers: int = 24
    img_resolution: int = 256
    crop_size: int | None = None
    use_aug: bool = True
    aug_rot_scale: Union[float, List[float]] = 1.0
    aug_trans_scale: float = 0.25
    turn_to_cam_coord: bool = False

    # logging and checkpointing
    log_dir: str = 'logs'
    project_id: str = 'renderformer-v1.5'
    exp_id: str = 'TBD'
    log_loss_interval: int = 20
    log_img_interval: int = 500
    save_interval: int = 500
    test_interval: int = 4000
    total_steps: int = 1_300_000

    # gradient computation
    gradient_checkpointing: bool = False
    gradient_checkpointing_view_tf: bool = False
    # Optimizer-step accumulation is implemented by the RF2 runner. The RF1
    # runner validates that this remains at the historical value of 1.
    gradient_accumulation_steps: int = 1

    # optimizer
    lr: float = 1e-4
    loss_l1_weight: float = 1.0
    loss_l2_weight: float = 0.0
    loss_foreground_weight: float = 1.0
    loss_lpips_weight: float = 0.01
    loss_ms_ssim_weight: float = 0.0
    loss_smape_weight: float = 0.0
    learn_ldr: bool = False
    loss_on_linear: bool = False
    linear_loss_on_proxy: bool = True
    emission_area_encoding: bool = False
    clamped_hdr: bool = False
    clamped_hdr_max: float = 9.0
    num_warmup_steps: int = 8_000
    num_cosine_steps: int = 1_200_000
    weight_decay: float = 0.0

    # Full Accelerate state (model + optimizer + scheduler) versus a standalone
    # RenderFormerModel checkpoint used to initialize a fresh training run.
    auto_resume: bool = False
    load_ckpt: str | None = None
    load_weights: str | None = None
    # Historical curricula intentionally permit partial cross-architecture
    # warm starts. Set True for same-architecture component initialization.
    load_weights_strict: bool = False
    load_weights_revision: str | None = None
    load_weights_cache_dir: str | None = None
    load_weights_local_files_only: bool = False
    load_weights_subfolder: str | None = None
    load_optimizer_state: str | None = None
    finetune_mode: Literal["full", "last_layer", "last_layer_with_token", "proj_only"] = "full"

    override_view_tf32_mode: bool = False

    # DeepSpeed configuration (via Accelerate)
    use_deepspeed: bool = False
    deepspeed_config_path: str | None = None

    def __post_init__(self) -> None:
        resume_sources = [
            name
            for name, enabled in (
                ("auto_resume", self.auto_resume),
                ("load_ckpt", self.load_ckpt is not None),
            )
            if enabled
        ]
        component_sources = [
            name
            for name, value in (
                ("load_weights", self.load_weights),
                ("load_optimizer_state", self.load_optimizer_state),
            )
            if value is not None
        ]
        if len(resume_sources) > 1:
            raise ValueError(
                "auto_resume and load_ckpt are mutually exclusive full-state "
                "resume modes"
            )
        if resume_sources and component_sources:
            raise ValueError(
                f"{resume_sources[0]} is a full-state resume and cannot be combined with "
                + ", ".join(component_sources)
            )
        weight_options = {
            "load_weights_revision": self.load_weights_revision,
            "load_weights_cache_dir": self.load_weights_cache_dir,
            "load_weights_local_files_only": self.load_weights_local_files_only,
            "load_weights_subfolder": self.load_weights_subfolder,
        }
        configured_options = [
            name for name, value in weight_options.items() if value not in {None, False}
        ]
        if self.load_weights is None and configured_options:
            raise ValueError(
                "load_weights must be set when using " + ", ".join(configured_options)
            )
