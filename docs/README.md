# RenderFormer documentation

The public documentation is split by workflow. Run commands from the
repository root unless a guide says otherwise.

## Environment setup

Use [environment setup](environment-setup/README.md) for the supported Conda
environments, pip requirements, CUDA/FlashAttention installation, and
dependency-only Docker images. The source tree is mounted at runtime and is
never copied into an image.

Related references:

- [Conda environment manifests](../environments/README.md)
- [Dockerfiles and pinned requirements](../docker/)
- [Blender add-ons for bpy 4.5.10](../tools/blender_addon/README.md)

## Inference

Use the [inference guide](inference/README.md) for
`RenderFormerPipeline`, `RenderFormerV2Pipeline`, Hugging Face loading, input
H5 preparation, validation, CLI usage, batching, and output formats.

Runnable scenes and launchers are kept separately:

- [RF1/V1 examples](../examples/rf1/README.md)
- [RF2/V2 examples](../examples/rf2/README.md)

## Training

Use the [training recipes](training/README.md) for the V1 two-stage curriculum,
the V2 eight-stage curriculum, and the standalone material-autoencoder recipe.
Each renderer stage binds a semantic data YAML, a data-generation shell script,
a model YAML, and a training shell script; Material AE training consumes an
explicit EXR manifest and records its physical-input preprocessing contract.

Related references:

- [Mixed-resolution training](training/mixed-resolution.md)
- [Objaverse, renderer, and material-sphere data-generation receipts](../scripts/data/README.md)
- [Training launchers](../scripts/train/README.md)
- [Configuration ownership](../configs/README.md)
- [Data package and schemas](../src/renderformer/data/README.md)

## Citation and licensing

Please cite both papers listed in the repository
[README](../README.md#citation). Machine-readable metadata is in
[`CITATION.cff`](../CITATION.cff), and third-party terms are summarized in
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).
