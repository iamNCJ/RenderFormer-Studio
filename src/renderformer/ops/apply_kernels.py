import torch
from .rms_norm import LigerRMSNorm
from .swiglu import LigerSwiGLUMLP
from .rope import liger_rotary_pos_emb
from renderformer.models.attention import FeedForwardSwiGLU, MultiHeadAttention


def apply_kernels(
    module: torch.nn.Module,
    use_rms_norm: bool = True,
    use_swiglu: bool = True,
    use_rope: bool = True,
    verbose: bool = True,
):
    """
    Apply the Liger kernels to the given module.

    Args:
        module: The module to apply the kernel to.
        use_rms_norm: Whether to use the LigerRMSNorm kernel.
        use_swiglu: Whether to use the LigerSwiGLUMLP kernel.
        use_rope: Whether to use the LigerRoPE kernel.
        verbose: Whether to print out the names of the modules that the kernel was applied to.

    Returns:
        The module with the kernel applied.
    """

    if use_rms_norm:
        swap_dict = {}
        module_dict = dict(module.named_modules())
        for old_module_name, old_module in module.named_modules():
            if isinstance(old_module, torch.nn.RMSNorm):
                parent_name = '.'.join(old_module_name.split('.')[:-1])
                child_name = old_module_name.split('.')[-1]
                parent_module = module
                if parent_name:
                    parent_module = module_dict[parent_name]
                swap_dict[old_module_name] = (LigerRMSNorm.from_torch_module(old_module), parent_module, parent_name, child_name)
        for old_module_name, (new_module, parent_module, parent_name, child_name) in swap_dict.items():
            setattr(parent_module, child_name, new_module)
            if verbose:
                print(f"Applied LigerRMSNorm to {parent_name} attr {child_name}")

    if use_swiglu:
        swap_dict = {}
        module_dict = dict(module.named_modules())
        for old_module_name, old_module in module.named_modules():
            if isinstance(old_module, FeedForwardSwiGLU) and LigerSwiGLUMLP.can_be_applied(old_module):
                parent_name = '.'.join(old_module_name.split('.')[:-1])
                child_name = old_module_name.split('.')[-1]
                parent_module = module
                if parent_name:
                    parent_module = module_dict[parent_name]
                swap_dict[old_module_name] = (LigerSwiGLUMLP.from_torch_module(old_module), parent_module, parent_name, child_name)
        for old_module_name, (new_module, parent_module, parent_name, child_name) in swap_dict.items():
            setattr(parent_module, child_name, new_module)
            if verbose:
                print(f"Applied LigerSwiGLUMLP to {parent_name} attr {child_name}")

    if use_rope:
        for old_module_name, old_module in module.named_modules():
            if isinstance(old_module, MultiHeadAttention):
                old_module.apply_rope_cossin = liger_rotary_pos_emb # type: ignore
                # _bind_method_to_module(old_module, "apply_rope_cossin", liger_rotary_pos_emb)
                if verbose:
                    print(f"Applied LigerRoPE to {old_module_name} attr apply_rope_cossin")

    return module
