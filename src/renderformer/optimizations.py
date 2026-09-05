"""Validated optional inference fusion for the shared model."""

from __future__ import annotations

import torch


def apply_inference_optimizations(model, level: str = "compile", *, verbose: bool = True):
    if level == "none":
        return model
    if level != "compile":
        raise ValueError(f"unknown optimization level: {level}")
    if verbose:
        print("[opt] compiling encoder and view transformer", flush=True)
    model.transformer = torch.compile(model.transformer, dynamic=True, fullgraph=False)
    model.view_transformer.transformer = torch.compile(
        model.view_transformer.transformer,
        dynamic=True,
        fullgraph=False,
    )
    if hasattr(model.view_transformer, "out_dpt"):
        model.view_transformer.out_dpt = torch.compile(
            model.view_transformer.out_dpt,
            dynamic=False,
            fullgraph=False,
        )
    if verbose:
        print("[opt] compile enabled; the first inputs include tracing warm-up", flush=True)
    return model
