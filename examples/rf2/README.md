# RF2 examples

This directory contains the RenderFormer V2 paper scenes that are not already
covered by the RF1 examples. The scene inventory is below, and asset credits
are in [`PAPER_SCENE_ATTRIBUTIONS.md`](PAPER_SCENE_ATTRIBUTIONS.md).

Generated H5 files are intentionally not stored in Git. The generation
launcher starts from a bundled prepared frame (plus any explicitly requested
external asset), renders the raw scene data, encodes its materials, and
validates the result. The inference launcher consumes that processed H5 in the
separate CUDA environment.

## Scene inventory

| Scene ID | Paper scene | Default output | Tone mapper | Source status |
| --- | --- | ---: | --- | --- |
| `transparent-torus` | [Transparent Torus](reference/paper/transparent-torus.jpg) | 512 | `none` | bundled |
| `environment-lit-spheres` | [Environment Lit Spheres](reference/paper/environment-lit-spheres.jpg) | 512 | `none` | geometry/material bundled; local environment map required |
| `smoky-bunny` | [Smoky Bunny](reference/paper/smoky-bunny.jpg) | 512 | `none` | geometry/config bundled; local VDB required |
| `three-teapots` | [Three Teapots](reference/paper/three-teapots.jpg) | 512 | `pbr_neutral` | bundled |
| `displacement-mapped-cornell-cube` | [Displacement Mapped Cornell Cube](reference/paper/displacement-mapped-cornell-cube.jpg) | 512 | `none` | bundled |
| `spaceship-in-smoke` | [Spaceship in Smoke](reference/paper/spaceship-in-smoke.jpg) | 2048 | `agx` | config bundled; local spaceship split and VDB required |
| `dinner-scene` | [Dinner Scene](reference/paper/dinner-scene.jpg) | 2048 | `none` | licensed local prepared frame required |
| `cube-pile` | [Cube Pile](reference/paper/cube-pile.jpg) | 512 | `none` | bundled |
| `dragon` | [Dragon](reference/paper/dragon.jpg) | 2048 | `agx` | bundled; `RESOLUTION=512` is also supported |
| `bedroom` | [Bedroom](reference/paper/bedroom.jpg) | 2048 | `agx` | room bundled; local wall base color required |
| `living-room` | [Living Room](reference/paper/living-room.jpg) | 2048 | `agx` | bundled |
| `living-room-daylight` | [Living Room with Daylight](reference/paper/living-room-daylight.jpg) | 2048 | `none` | room bundled; local HDRI required |

Reference JPEGs identify the scene and paper appearance. They are not inputs
to the model and are not regenerated inference outputs.

## Install the two environments

Data generation and model inference intentionally use separate environments.
Generation uses Python 3.11, OpenVDB 13, and the pinned PyPI Blender wheel; do
not use a separately installed Blender application or its embedded Python.

```bash
conda env create -f environments/datagen.yml
conda activate renderformer-datagen

# Linux/NVIDIA. See environments/README.md for CPU/macOS alternatives.
python -m pip install \
  torch==2.7.1+cu126 torchvision==0.22.1+cu126 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install bpy==4.5.10 --index-url https://pypi.org/simple
python -m pip install -r docker/requirements-datagen.txt
python -m pip install --no-deps -e .
```

The VDB examples also require the OpenVDB 13 Python binding supplied by
`environments/datagen.yml`. Install the supported CUDA inference environment
separately:

```bash
conda env create -f environments/cuda.yml
conda activate renderformer-cuda
python -m pip install -r docker/requirements-cuda.txt
MAX_JOBS=4 python -m pip install \
  flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
python -m pip install --no-deps -e .
```

See the [environment setup guide](../../docs/environment-setup/README.md) for
the dependency-only container workflow. Keeping the environments separate
avoids mixing Blender/OpenVDB's Python 3.11 and NumPy 1.x constraints with the
Python 3.10 CUDA training and inference stack.

## Generate and render the Cornell-box example

Generate the processed H5 in `renderformer-datagen`:

```bash
conda activate renderformer-datagen
bash examples/rf2/generate-paper-scene.sh \
  displacement-mapped-cornell-cube \
  /tmp/renderformer-rf2-displacement-mapped-cornell-cube
```

Then switch to `renderformer-cuda` and run inference on that exact artifact:

```bash
conda activate renderformer-cuda
INPUT_H5=/tmp/renderformer-rf2-displacement-mapped-cornell-cube/processed/scene.h5 \
  bash examples/rf2/render-paper-scene.sh \
  displacement-mapped-cornell-cube
```

The default inference output directory is
`/tmp/renderformer-rf2-displacement-mapped-cornell-cube/inference`. The
launcher selects `transformer_512` from the
`RenderFormer/renderformer-v2` bundle for this scene. It never starts Blender
or data generation. Authenticate with Hugging Face if the bundle is still
access-protected during release staging.

If `INPUT_H5` is unset, the launcher reads
`${DATA_OUTPUT_DIR}/processed/scene.h5`; `DATA_OUTPUT_DIR` defaults to
`/tmp/renderformer-rf2-${SCENE_ID}`. This equivalent invocation is convenient
when generation used that layout:

```bash
conda activate renderformer-cuda
DATA_OUTPUT_DIR=/tmp/renderformer-rf2-displacement-mapped-cornell-cube \
OUTPUT_DIR=/tmp/my-rf2-cbox-results \
  bash examples/rf2/render-paper-scene.sh \
  displacement-mapped-cornell-cube
```

Common inference overrides are `INPUT_H5`, `DATA_OUTPUT_DIR`, `CHECKPOINT`,
`CHECKPOINT_SUBFOLDER`, `RESOLUTION`, `DEVICE`, `PRECISION`,
`EXECUTION_PROFILE`, `TONE_MAPPER`, `PYTHON_BIN`, and `OUTPUT_DIR`. The official
bundle selects `transformer_512` or `transformer_2048` from `RESOLUTION` when
`CHECKPOINT_SUBFOLDER` is unset. `TONE_MAPPER` accepts `none`, `agx`, `filmic`,
or `pbr_neutral`. When it is unset, the launcher uses the per-scene display
transform in the inventory above; setting it explicitly overrides that choice.
Tone mapping changes the display PNG only, while the EXR remains linear HDR. A
missing processed H5 fails immediately with the generation command needed to
create it.

## Use the Python pipeline

Generate a processed input in the datagen environment:

```bash
conda activate renderformer-datagen
bash examples/rf2/generate-paper-scene.sh \
  displacement-mapped-cornell-cube /tmp/renderformer-rf2-cbox-data
```

Switch to the CUDA environment before loading the official model:

```bash
conda activate renderformer-cuda
```

```python
from renderformer import RenderFormerV2Pipeline

pipeline = RenderFormerV2Pipeline.from_pretrained(
    "RenderFormer/renderformer-v2",
    transformer_subfolder="transformer_512",
    resolution=512,
    device="cuda",
)
result = pipeline(
    "/tmp/renderformer-rf2-cbox-data/processed/scene.h5",
    resolution=512,
    precision="fp16",
)
hdr = result.hdr
```

See the [inference guide](../../docs/inference/README.md) for pipeline loading,
H5 validation, execution profiles, and output formats.

## Generate a scene without inference

Run the data pipeline directly when you want to inspect or reuse its H5:

```bash
conda activate renderformer-datagen
bash examples/rf2/generate-paper-scene.sh \
  three-teapots /tmp/renderformer-rf2-three-teapots
```

The launcher writes:

```text
/tmp/renderformer-rf2-three-teapots/
  raw/v2_training_raw.h5
  processed/scene.h5
```

It uses the settings expected by the released V2 models:

- the per-scene default resolution listed above;
- 4,096 Cycles samples per pixel;
- 256-pixel texture crop and target resolution;
- 32 x 32 per-triangle material samples;
- the `v2_training_raw` export profile; and
- the Qwen-Image VAE's repository-default revision.

Set `POSTPROCESS=0` to stop after raw generation. The supported generation
overrides are `RESOLUTION`, `SPP`, `POSTPROCESS_DEVICE`, `QWEN_MODEL_ID`,
`QWEN_REVISION`, and `PYTHON_BIN`. Leave `QWEN_REVISION` unset for the
repository default, or set it to an explicit branch, tag, or commit.

The processed H5 is a generated artifact. Keep it in a data/output directory,
not in the source checkout.

## External assets

Some paper assets cannot be redistributed. The launcher requires an explicit
local file or directory path for each one and checks that the path has the
expected type before generation.

| Scene | Environment variable | Required local asset |
| --- | --- | --- |
| Environment Lit Spheres | `RF2_ENV_SPHERES_MAP` | Grace Cathedral lat-long EXR |
| Smoky Bunny | `RF2_SMOKY_BUNNY_VDB` | processed bunny-smoke VDB |
| Spaceship in Smoke | `RF2_SPACESHIP_SPLIT` | prepared `split/` directory |
| Spaceship in Smoke | `RF2_SPACESHIP_VDB` | processed fluid VDB |
| Dinner Scene | `RF2_DINNER_FRAME` | complete prepared-frame directory |
| Bedroom | `RF2_BEDROOM_WALL_BASECOLOR` | wall base-color PNG |
| Living Room with Daylight | `RF2_DAYLIGHT_ENV_MAP` | paper HDRI EXR |

Example:

```bash
conda activate renderformer-datagen
RF2_SMOKY_BUNNY_VDB=/licensed/assets/fluid_data_0001_preprocessed.vdb \
  bash examples/rf2/generate-paper-scene.sh \
  smoky-bunny /tmp/renderformer-rf2-smoky-bunny

conda activate renderformer-cuda
INPUT_H5=/tmp/renderformer-rf2-smoky-bunny/processed/scene.h5 \
  bash examples/rf2/render-paper-scene.sh smoky-bunny
```

Supplying a path does not grant redistribution permission or prove that the
asset is byte-identical to the paper input. A result made with a different
lawfully acquired asset should not be described as the original paper input.

## Prepared-frame safety

Paper-scene OBJ files are already in world space. The `--prepared-frame` data
path therefore validates that all mesh/material paths remain inside the frame,
rejects missing or private absolute paths, stages inputs atomically, renders
the untouched frame, and applies mesh postprocessing only to a separate model
input copy. It also preserves the exported camera list.

This prevents accidental double transforms and keeps interrupted generation
from publishing a partial H5. The scene generator refuses a stale non-empty
output directory; select a new directory or move the old results first.

## Attribution

Asset licenses and required credits are listed in
[`PAPER_SCENE_ATTRIBUTIONS.md`](PAPER_SCENE_ATTRIBUTIONS.md). Cite
both RenderFormer papers as listed in the root [README](../../README.md#citation),
plus any applicable source assets.
