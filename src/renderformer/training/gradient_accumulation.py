"""Training-entrypoint gradient-accumulation boundaries."""

from __future__ import annotations


def validate_gradient_accumulation_steps(steps: int) -> int:
    """Return a valid positive accumulation count; never silently clamp it."""

    if steps <= 0:
        raise ValueError(
            "trainer_config.gradient_accumulation_steps must be a positive "
            f"integer (received {steps})"
        )
    return steps


def validate_v1_gradient_accumulation_steps(steps: int) -> None:
    """Fail before setup when a V1 loop requests unsupported accumulation.

    The RF1 runner counts one dataloader batch as one optimizer/global step.
    Unlike ``renderformer.training.train_rf2``, it has no optimizer-step outer loop around
    multiple Accelerate accumulation micro-batches. Silently accepting another
    value would therefore advance the optimizer, scheduler, logs, and
    checkpoints with inconsistent meanings.
    """

    validate_gradient_accumulation_steps(steps)
    if steps != 1:
        raise ValueError(
            "The RF1 training runner requires "
            "trainer_config.gradient_accumulation_steps == 1; only "
            "the RF2 runner implements gradient accumulation with optimizer-step "
            f"global_step semantics (received {steps})"
        )
