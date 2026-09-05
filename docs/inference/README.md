# RenderFormer inference

RenderFormer exposes one trainable transformer component and two inference
pipelines:

| API | Use it for | What it owns |
| --- | --- | --- |
| `RenderFormerModel` | Training, fine-tuning, or direct checkpoint inspection | Only the shared RenderFormer transformer |
| `RenderFormerPipeline` | V1/RF1 inference | Transformer, RF1 preprocessing, cameras/rays, and HDR decoding |
| `RenderFormerV2Pipeline` | V2 inference | Transformer plus the texture and environment encoders required by the V2 checkpoint |

There is only one transformer implementation. The pipelines wrap the same
`RenderFormerModel` used by the training entrypoints; V1 inference is not a
second copy of the architecture.

The examples import pipeline classes from the package root. Importing them
from `renderformer.pipelines` is equivalent and is useful when inspecting the
inference layer explicitly.

## Installation

Python 3.10 and 3.11 are supported. For a regular editable inference install:

```bash
python -m pip install -e ".[inference]"
```

Legacy JSON-to-RF1 composition also uses the local mesh utilities. The
`inference` extra already includes the default mesh dependency; a data-only
installation can instead add `geometry`:

```bash
python -m pip install -e ".[geometry]"
```

For the supported CUDA stack, create the Conda environment and install the
shared pinned requirements plus FlashAttention:

```bash
conda env create -f environments/cuda.yml
conda activate renderformer-cuda
python -m pip install -r docker/requirements-cuda.txt
MAX_JOBS=4 python -m pip install \
  flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
python -m pip install --no-deps -e .
```

FlashAttention is required whenever V2 resolves
`use_packed_sequence=True`. The default V2 `release` profile is unpacked, while
the explicit trainer-config `checkpoint` diagnostic can enable packing for the
current volume-enabled curriculum checkpoints. PyTorch SDPA does not implement
that packed path: if FlashAttention is unavailable, or
`USE_SDPA=1`/`ATTN_IMPL=sdpa` selected SDPA before RenderFormer was imported,
the pipeline fails before model execution. V1 and the default V2 release path
can use SDPA.

The repository Docker images contain dependencies only. Mount this checkout at
runtime rather than baking the source into an image; see
[environment setup guide](../environment-setup/README.md).

V2 Blender/OpenVDB generation uses the separate `renderformer-datagen`
environment documented below. This separation is intentional: create the
processed H5 there, then activate `renderformer-cuda` for the pipeline or CLI.

## Pipeline API

### V1: `RenderFormerPipeline`

The two public V1 model IDs are `microsoft/renderformer-v1-base` and
`microsoft/renderformer-v1.1-swin-large`. By default, `from_pretrained` follows
the selected Hugging Face repository's default revision. Pass `revision=` only
when a particular branch, tag, or commit is required.

```python
from renderformer import RenderFormerPipeline

pipeline = RenderFormerPipeline.from_pretrained(
    "microsoft/renderformer-v1.1-swin-large",
    device="cuda",
)

scene = pipeline.load_h5("scene.h5")
output = pipeline.render_scene(
    scene,
    resolution=512,
    precision="fp16",
)

# Linear HDR RGB, shaped (batch=1, views, height, width, channels=3).
hdr = output.hdr
# Un-decoded model output, shaped (batch=1, views, channels=3, height, width).
raw = output.raw
```

For a single H5, the pipeline-style shortcut performs the same validated load
and render:

```python
output = pipeline("scene.h5", resolution=512, precision="fp16")
```

Use `first_view_only=True` to render only view zero. To batch RF1 scenes with a
shared triangle padding length, use `render_v1_scenes`:

```python
scenes = [pipeline.load_h5(path) for path in h5_paths]
outputs = pipeline.render_v1_scenes(
    scenes,
    resolution=512,
    precision="fp16",
    padding_length=4192,
)
```

Every scene in one call must have the same number of selected views. Scenes
that do not share a view count should be rendered separately. The historical
tensor-level `render(...)`/`__call__(...)` contract remains available, and
`RenderFormerRenderingPipeline` is an alias for `RenderFormerPipeline`, but
new code should prefer validated `SceneData` objects and `render_scene`.

The separately published historical RF1 video bundle requires an explicit
compatibility boundary because it predates the canonical float32 producer
contract. Load those files with:

```python
scene = pipeline.load_h5(
    "video-frame.h5",
    allow_legacy_rf1_dtypes=True,
)

# The path shortcut accepts the same explicit option.
output = pipeline(
    "video-frame.h5",
    allow_legacy_rf1_dtypes=True,
    resolution=512,
    precision="fp16",
)
```

This option is only for `renderformer/renderformer-video-data`. It accepts the
strict RF1 signature plus the bundle's two exact historical signatures:

| `triangles` | `texture` | `vn` | `c2w` | `fov` |
| --- | --- | --- | --- | --- |
| float64 | float16 | float64 | float32 | float32 |
| float32 | float16 | float32 | float32 | float32 |

Accepted legacy arrays are converted to float32 in memory, as they were by the
original V1 inference loader. Files are not rewritten. Integer arrays, mixed
geometry/normal dtypes, and every other dtype combination still fail.
The public
[`examples/rf1/video-sequences.json`](../../examples/rf1/video-sequences.json)
inventory is the source of truth for sequence order, archive/input mapping,
frame counts, and tone mapping. `download-video-data.sh` writes an ordered
`frames.manifest` for each listed archive, and `render-videos.sh` consumes
those manifests with the compatibility flag already enabled.

### V2: `RenderFormerV2Pipeline`

V2 has a separate class because its preprocessing can include material texture
latents, environment-map latents, light strengths, and participating-media
volumes:

```python
import torch

from renderformer import RenderFormerV2Pipeline

pipeline = RenderFormerV2Pipeline.from_pretrained(
    "RenderFormer/renderformer-v2",
    transformer_subfolder="transformer_512",
    resolution=512,
    device="cuda",
    environment_encoder_dtype=torch.float16,
    texture_encoder_dtype=torch.float32,
)

output = pipeline(
    "/datasets/renderformer/processed/scene.h5",
    resolution=512,
    precision="fp16",
)
hdr = output.hdr
```

The two release transformers and the four material components share one
componentized Hugging Face repository. A repository revision therefore selects
the complete V2 component set atomically:

| Repository | Transformer subfolder | Native output |
| --- | --- | ---: |
| `RenderFormer/renderformer-v2` | `transformer_512` | 512 |
| `RenderFormer/renderformer-v2` | `transformer_2048` | 2048 |

Use native-512 with `resolution=512` for normal V2 examples. Use native-2048
with `resolution=2048` for high-resolution output. These are the complete V2
release pair. Authenticate with Hugging Face whenever the bundle is
access-protected during release staging. Omit `revision` to follow its default
revision, or pass one explicit branch, tag, or commit to select every bundled
component from the same repository state.

`precision` on the render call selects transformer autocast. Auxiliary encoder
dtypes are independent. Under the default `release` profile, the environment
encoder and geometry use render precision while the texture encoder uses fp32;
raw texture and light preprocessing is performed in fp32 before the model
inputs are cast. The explicit `checkpoint` diagnostic uses the same dtype and
input-preparation policy, but reads its view-transformer and packing switches
from the trainer config. The CLI instead uses fp32 for both encoders and fp32
geometry under `legacy_v2`, matching the retired raw-inference path. Under
`legacy_template_v2`, the environment encoder follows render precision, the
pre-encoded texture path does not instantiate a texture encoder, and geometry
remains fp32. `legacy_v2` also preserves the retired path's unpadded
explicit-volume length and casts every volume field to render precision; the
other profiles retain their checkpoint-sized padded representation. The
`torch_dtype` argument is a convenience override for both encoders;
`environment_encoder_dtype` and `texture_encoder_dtype` are preferable when
reproducing a recorded run. None of these three options casts the transformer
checkpoint, which is strictly loaded before the pipeline is returned.

V2 component ownership is intentionally explicit:

| Component | Source and selection | Pipeline behavior |
| --- | --- | --- |
| Transformer | The primary `source` and optional `revision` | Loaded strictly, switched to eval mode, frozen, exposed as both `.model` and `.transformer` |
| Environment encoder | `model_config.envmap_vae_model_id` when environment lighting is enabled | Loaded from Diffusers and exposed as `.environment_encoder` |
| Texture encoder | Qwen for 64/80-channel checkpoints; DC-AE for 128/160-channel checkpoints | Encodes raw per-triangle material maps and is exposed as `.texture_encoder` |

The pipeline encoders above are separate from the historical 9-D material
models exposed by `renderformer.models.material`. `MaterialEncoder`,
`MaterialDecoder`, and `MaterialAutoencoder` operate on 3 x 256 x 256
network-space material images: the encoder returns a 9-D `tanh` latent and the
decoder returns a sigmoid-bounded network-space image. The release checkpoint
uses alpha-premultiplied, finite, nonnegative linear RGB with the explicit
`log10(1 + RGB)` transform and `10**network_RGB - 1` inverse. The same module
exports the three BRDF-parameter mapper MLP classes used during V2 data
generation.

The material autoencoder and all three mapper artifacts occupy subfolders in
the same `RenderFormer/renderformer-v2` bundle. The autoencoder loads without
trainer state; its component-level `preprocessor_config.json` records the
physical-input contract:

```python
import torch

from renderformer.models.material import MaterialAutoencoder

material_ae = MaterialAutoencoder.from_pretrained(
    "RenderFormer/renderformer-v2",
    subfolder="material_autoencoder",
    strict=True,
).eval()

# Input is alpha-premultiplied, nonnegative linear RGB.
linear_rgb = torch.rand(1, 3, 256, 256)
latent = material_ae.encode_hdr(linear_rgb)
reconstructed_linear_rgb = material_ae.decode_hdr(latent)
```

Use `encode` and `decode` only when tensors are already in network space.
`encode_hdr` and `decode_hdr` apply the audited release transform without
changing the checkpoint-compatible architecture.

The mapper subfolders are `diffspec_mapper`, `metallic_mapper`, and
`metallic_transmission_mapper`. V2 data generation selects them automatically
from `RenderFormer/renderformer-v2`; environment variables can still override
an individual mapper with a standalone local directory or alternative model
ID. Direct callers use the same repository-plus-subfolder API as the material
autoencoder.

```python
from renderformer.models.material import DiffuseSpecularToLatent

mapper = DiffuseSpecularToLatent.from_pretrained(
    "RenderFormer/renderformer-v2",
    subfolder="diffspec_mapper",
    strict=True,
).eval()
```

See the
[data package guide](../../src/renderformer/data/README.md#material-latent-models)
for the network-space contract and legacy local-checkpoint compatibility.

Auxiliary encoders also follow their repository-default revisions. `cache_dir`
and `local_files_only` apply to both the primary checkpoint and auxiliary
downloads. Advanced callers can use `component_revisions` to request a
specific auxiliary branch, tag, or commit without changing the transformer
revision:

```python
pipeline = RenderFormerV2Pipeline.from_pretrained(
    "RenderFormer/renderformer-v2",
    transformer_subfolder="transformer_512",
    resolution=512,
    device="cuda",
    cache_dir="/models/huggingface",
    local_files_only=True,
    component_revisions={
        "Qwen/Qwen-Image": "my-tested-tag",
    },
)
```

By default, an environment encoder declared by the checkpoint is loaded
eagerly. The texture encoder is always deferred until an input contains a raw
texture, so a pre-encoded H5 does not allocate a second large model. Set
`load_auxiliary_models=False` to make the environment encoder lazy as well.
This is useful for inspecting a pipeline without allocating encoder weights;
it does not disable a component that a later input requires.

Advanced callers can provide preloaded encoder objects. The pipeline then owns
their inference lifecycle: it switches them to eval mode, disables gradients,
and moves them with the pipeline. It also casts each injected module to the
resolved role dtype and verifies that the module exposes a callable `encode`
method and floating-point parameters. Reusing one object for both roles
requires identical dtype and image/video layout settings.

```python
import torch
from diffusers import AutoencoderKLQwenImage

from renderformer import RenderFormerV2Pipeline

qwen_vae = AutoencoderKLQwenImage.from_pretrained(
    "Qwen/Qwen-Image",
    subfolder="vae",
    torch_dtype=torch.float16,
)

pipeline = RenderFormerV2Pipeline.from_pretrained(
    "RenderFormer/renderformer-v2",
    transformer_subfolder="transformer_512",
    resolution=512,
    device="cuda",
    environment_encoder_dtype=torch.float16,
    texture_encoder_dtype=torch.float16,
    load_auxiliary_models=False,
    environment_encoder=qwen_vae,
    texture_encoder=qwen_vae,
    environment_encoder_is_video=True,
    texture_encoder_is_video=True,
)
```

The two `*_is_video` flags describe an injected encoder's input convention.
Qwen's VAE is video-style and receives an added temporal dimension; DC-AE is
image-style. The pipeline recognizes the built-in Diffusers classes, but a
custom or wrapped encoder should set these flags explicitly rather than rely
on class-name detection.

Automatic components are reused only when model ID, revision, subfolder,
loader type, and dtype all match. Under `release` and `checkpoint`, the CLI
uses render precision for the environment encoder and fp32 for the texture
encoder, so its Qwen roles are separate at fp16/bf16 render precision. Set both
encoder dtypes alike only when sharing one compatible encoder instance is
intentional.
A pre-encoded texture is accepted when its channel count exactly matches the
checkpoint. For a 64/128-channel checkpoint, raw `(N, >=12, H, W)` material
maps are encoded in triangle batches. An 80/160-channel checkpoint additionally
requires either the historical 15-channel form (zero heightmap) or at least 16
channels (heightmap at channel 15). A checkpoint with an unrecognized learned
texture width fails explicitly and asks for a pre-encoded tensor.

`pipeline.component_info` returns serializable provenance for both loaded and
lazy components, including source model, revision, subfolder, and dtype. Once
loaded, it also reports the concrete class and whether an object was reused:

```python
for role, component in pipeline.component_info.items():
    print(role, component)
```

For 64/128-channel checkpoints, raw channels 0–11 form four RGB encoder
groups. An 80/160-channel checkpoint adds the heightmap group from raw channel
15. Historical 15-channel inputs have no heightmap channel and are interpreted
as a zero heightmap; current raw exports normally contain all 16 channels.

### V2 execution profiles

New inference should use the default `release` profile. It enables
view-transformer TF32, disables packed sequences, encodes a missing environment
as black, and keeps a missing volume fully masked at the checkpoint's padded
length. Geometry and the environment encoder follow render precision; the
texture encoder remains fp32.

The `checkpoint` profile is available when inspecting settings serialized by
a training checkpoint. The compatibility-only `legacy_v2` and
`legacy_template_v2` profiles remain accepted for old inputs, but should not
be selected for new inference.

The CLI equivalents are `--execution-profile`,
`--[no-]view-transformer-tf32`, and `--[no-]packed-sequence`. These V2-only
options are rejected for a V1 checkpoint. When packing is enabled,
FlashAttention is mandatory; the pipeline fails early if the active attention
backend cannot execute packed sequences.

### Device movement and compilation

Both pipelines support `.to(device)`. Moving a CPU-created pipeline to CUDA
also applies the repository's local fused kernels when they were requested:

```python
pipeline = RenderFormerPipeline.from_pretrained(
    "microsoft/renderformer-v1-base",
    device="cpu",
)
pipeline.to("cuda")
```

`pipeline.optimize("compile")` applies the maintained `torch.compile`
optimizations. Its first call includes compilation warm-up; compare a
representative output with `optimize("none")` before deploying it.

## Loading the transformer for training

Training and fine-tuning should load `RenderFormerModel` directly. Unlike a
pipeline, `from_pretrained` deliberately leaves the module in training mode
with gradients enabled and does not construct or freeze any auxiliary encoder:

```python
import torch

from renderformer import RenderFormerModel

model, loading_info = RenderFormerModel.from_pretrained(
    "RenderFormer/renderformer-v2",
    subfolder="transformer_512",
    device="cuda",
    strict=True,
    output_loading_info=True,
)

assert model.training
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
```

For a public V1 transformer:

```python
model = RenderFormerModel.from_pretrained(
    "microsoft/renderformer-v1-base",
)
```

A local `.safetensors` file can be loaded when its config is supplied or a
sibling `model_config.json`, `config.json`, or `config.yaml` exists.
`save_pretrained()` writes the first of these. In a component-style bundle,
each transformer keeps its `model.safetensors` and checkpoint config together
inside its subfolder:

```python
model = RenderFormerModel.from_pretrained(
    "/checkpoints/renderformer-bundle",
    subfolder="transformer_512",
)

# Architecture + model.safetensors only; no optimizer or pipeline components.
model.save_pretrained("/checkpoints/exported-transformer")
```

Pipelines use the more explicit name `transformer_subfolder` for the same
layout:

```python
pipeline = RenderFormerV2Pipeline.from_pretrained(
    "/checkpoints/renderformer-bundle",
    transformer_subfolder="transformer_512",
    resolution=512,
    device="cuda",
)
```

Strict loading is the default. `ignore_mismatched_sizes` is available for an
intentional architecture change. Optimizer, scheduler, EMA, and distributed
trainer state are outside `RenderFormerModel`; the training entrypoints own
those states separately, as Diffusers-style component loading does.

The maintained RF1 and RF2 runners, selected through `renderformer train`,
call `renderformer.training.model_loading.load_training_model`. Their
`trainer_config.load_weights` value initializes only `RenderFormerModel`, logs
missing/unexpected/mismatched keys, and starts a fresh optimizer, scheduler,
and step counter. Direct use of `load_training_model` is strict by default.
Strict training loads cannot use a shape-mismatch allowlist.
`trainer_config.load_ckpt` selects an explicit
Accelerate-state directory; `trainer_config.auto_resume` (default `false`)
selects the latest state under the same run directory. These full-state modes
are mutually exclusive and cannot be combined with model/optimizer component
loading. Do not substitute one for the other when continuing an interrupted
run.

## Unified command-line inference

`renderformer infer` and `python -m renderformer infer` invoke the same CLI. Checkpoint
schema detection selects the V1 or V2 pipeline; `--variant` is an optional
assertion, not a way to reinterpret a checkpoint.

### One H5 file

```bash
renderformer infer \
  --input /data/rf1/scene.h5 \
  --checkpoint microsoft/renderformer-v1.1-swin-large \
  --device cuda \
  --precision fp16 \
  --resolution 512 \
  --output-dir /results/scene
```

### A newline manifest

Blank lines and lines beginning with `#` are ignored. Relative entries are
resolved relative to the manifest itself.

```text
# scenes.txt
scenes/0001.h5
scenes/0002.h5
```

```bash
renderformer infer \
  --input /data/rf1/scenes.txt \
  --checkpoint microsoft/renderformer-v1-base \
  --batch-size 4 \
  --padding-length 4192 \
  --num-workers 4 \
  --output-dir /results/rf1
```

Use unique H5 basenames in a manifest: output names are derived from the
basename. Duplicate stems, including case-only and Unicode-normalization
variants that collide on common filesystems, are rejected before checkpoint
loading. Supplying `--output-dir` explicitly also makes a manifest run easier
to reproduce. Before loading the checkpoint, the CLI opens and closes every
candidate H5 once so a corrupt late entry fails before partial rendering. It
then loads, validates, renders, and releases payloads in ordered windows of at
most `--batch-size` scenes; it does not retain every H5 in a manifest.
`--num-workers` is capped at that window size and one worker pool is reused
across the run. Use `--batch-size 1` for large or heterogeneous video frames.
`--save-video` still retains the much smaller uint8 output frames until it
writes the final MP4.

### A directory

Directory collection is non-recursive. Prefer a manifest for a fixed input
collection or dataset split.

```bash
renderformer infer \
  --input /data/rf1/scenes \
  --checkpoint microsoft/renderformer-v1-base \
  --batch-size 4 \
  --padding-length 4192 \
  --output-dir /results/rf1
```

### V2

This example uses the selected native-512 Hugging Face release model:

```bash
conda activate renderformer-cuda
renderformer infer \
  --input /data/renderformer-v2/processed.txt \
  --checkpoint RenderFormer/renderformer-v2 \
  --checkpoint-subfolder transformer_512 \
  --variant v2 \
  --device cuda \
  --precision fp16 \
  --output-dir /results/renderformer-v2
```

The repository includes source scenes and launchers for 12 curated RF2 paper
examples. Generated H5 files stay in the selected data/output directory rather
than in Git. Generate the Cornell-box H5 in the datagen environment:

```bash
conda activate renderformer-datagen
bash examples/rf2/generate-paper-scene.sh \
  displacement-mapped-cornell-cube \
  /tmp/renderformer-rf2-cbox
```

Then switch to the CUDA environment and infer the processed H5:

```bash
conda activate renderformer-cuda
INPUT_H5=/tmp/renderformer-rf2-cbox/processed/scene.h5 \
  bash examples/rf2/render-paper-scene.sh \
  displacement-mapped-cornell-cube
```

The inference launcher uses the default resolution in the RF2 scene inventory
and selects the matching native-512 or native-2048 model. It never invokes
Blender. Set `INPUT_H5` to an explicit processed artifact, or set
`DATA_OUTPUT_DIR` and let the launcher derive `processed/scene.h5` beneath it.
Use `RESOLUTION` to override output size or `CHECKPOINT` for a local directory
or another Hub model. Set `TONE_MAPPER` to `none` (the default), `agx`,
`filmic`, or `pbr_neutral` to control the display PNG without changing the
linear HDR EXR. See the
[RF2 example guide](../../examples/rf2/README.md) for scene assets, generation
requirements, and H5 preparation flow.
Environment maps and volume counts can differ per scene, so V2 scenes are
rendered one at a time even when `--batch-size` is greater than one. V1 uses
real padded scene batches when view counts match.

### Validation before a heavy run

`--validate-only` resolves and strictly loads the checkpoint, opens every H5,
and enforces both loader-level schema checks and checkpoint-specific input
constraints without running the model or loading V2 auxiliary encoders. It
uses the same bounded, ordered load windows as inference:

```bash
renderformer infer \
  --input /data/rf1/scenes.txt \
  --checkpoint microsoft/renderformer-v1-base \
  --variant v1 \
  --validate-only \
  --num-workers 4
```

Run this over every candidate input before inference. Re-run it after copying,
merging, or converting a dataset; a valid source directory does not guarantee
that a partially written destination is valid. An HDF5 error such as
`bad object header version number` normally indicates an interrupted or
out-of-memory write and the affected scene should be regenerated. For very
large meshes, especially the 128k tier, use `--num-workers 1` while loading or
validating. `renderformer data convert` processes one H5 per invocation; a
batch launcher should likewise run only one large conversion process at a
time.

For V2, this includes learned texture width/patch compatibility, required
separate light strength, final environment-map shape and encoder declaration,
volume shape/count versus checkpoint padding, resolved execution settings, and
packed-attention backend availability. It does not execute an auxiliary
encoder or the transformer, so exercise at least one representative scene with
the final checkpoint and execution profile before a long run.

After a full run, compare the EXR count with the sum of selected views in the
validated H5 files. If it is lower than expected, recheck for corrupted inputs
and mixed view counts instead of assuming every scene contributes four views.

### Main CLI controls

| Option | Meaning |
| --- | --- |
| `--revision` | Optional primary Hugging Face branch, tag, or commit. Omit it to use the repository default. |
| `--checkpoint-subfolder` | Transformer component inside a bundle. The official V2 repository selects `transformer_512` below 2048 output resolution and `transformer_2048` at 2048 or above when this option is omitted. |
| `--cache-dir`, `--force-download`, `--local-files-only` | Control primary and V2 auxiliary model resolution. |
| `--precision {auto,fp16,bf16,fp32}` | Transformer autocast and the `release`/`checkpoint`/`legacy_template_v2` environment-encoder dtype. `auto` is fp16 on CUDA and fp32 elsewhere. The texture encoder is fp32; `legacy_v2` also forces the environment encoder to fp32. Low-precision rendering is CUDA-only. |
| `--allow-legacy-rf1-dtypes` | Explicitly normalize only the two dtype signatures in the historical public RF1 video bundle. Rejected for V2; strict RF1 loading remains the default. |
| `--batch-size` | Maximum number of H5 scenes resident in one ordered load window, and actual padded batching for compatible V1 scenes. V2 remains per-scene after each bounded load. |
| `--padding-length` | V1 triangle padding length. It must be at least the largest scene in the batch. |
| `--num-workers` | Reused spawned processes for H5 loading, capped at `--batch-size`. Zero loads in the main process. |
| `--first-view-only` | Render only view zero from each scene. |
| `--execution-profile {release,checkpoint,legacy_v2,legacy_template_v2}` | V2 defaults to `release`; `checkpoint` reads trainer settings, while the legacy values are compatibility-only. |
| `--[no-]view-transformer-tf32` | Override whether the V2 view transformer uses TF32. |
| `--[no-]packed-sequence` | Override the V2 profile's packed-sequence choice. |
| `--tone-mapper` | PNG display conversion: `none`, `agx`, `filmic`, or `pbr_neutral`. EXR remains linear HDR. |
| `--no-fused-kernels` | Disable the local CUDA kernel replacements. |
| `--optimize compile` | Enable the optional compiled inference path. |
| `--save-video` | Assemble PNG frames in input/view order into `video.mp4`. |

Recommended V1 Base settings by triangle tier are:

| Tier | `--padding-length` | `--batch-size` |
| --- | ---: | ---: |
| 4k | 4192 | 4 |
| 8k | 8192 | 2 |
| 32k | 32768 | 1 |
| 64k | 65536 | 1 |
| 128k | 131072 | 1 |

Use fp16 unless investigating numerical behavior. V1 Base was trained on
scenes with at most about 4k triangles; quality degradation at 8k and above is
expected even when memory settings are sufficient.

### Outputs

For input `scene.h5`, each rendered view produces:

```text
scene_view_0.exr   # float32 linear HDR
scene_view_0.png   # selected display/tone-map path
scene_view_1.exr
scene_view_1.png
...
inference_manifest.json
```

The manifest records the checkpoint source, component subfolder, and requested
revision (`null` when the repository default was used), detected model
generation, precision, resolution, tone mapper, batching settings, every
rendered EXR/PNG output, and the optional top-level `video` path. For RF1 it
also records the
requested `allow_legacy_rf1_dtypes` setting. When legacy RF1 normalization
occurs, each scene's `input_preprocessing.rf1_dtype_normalization` records the
exact source dtypes, normalized keys, float32 target, and compatibility
profile. For V2 it also records the resolved execution settings, component
model/revision/subfolder/dtype provenance, and per-scene preprocessing sources
such as raw versus pre-encoded texture, transformed versus processed
environment, and `h5`/`texture`/`missing` light strength. For a single H5
without `--output-dir`, output defaults to a
`renderformer_outputs/` sibling directory. For directory or manifest jobs,
specify an output directory explicitly in release scripts.

Each scene `input` in the manifest is stored as a resolved absolute H5 path;
each `outputs[].exr`/`png` name is relative to the manifest directory. This
preserves the identities needed by downstream validation without depending on
legacy output filename conventions.

For V1, `execution_settings.view_transformer_tf32` records the actual
precision-derived pipeline choice: `true` for fp16/bf16 and `false` for fp32.

## Preparing V1/RF1 inputs

### Legacy scene JSON to RF1 H5

The working-tree JSON examples can be composed without Blender or Cycles:

```bash
renderformer data compose-rf1 \
  examples/rf1/cbox.json \
  --output /tmp/renderformer-rf1/cbox.h5

renderformer data validate-h5 \
  --input /tmp/renderformer-rf1/cbox.h5 \
  --format v1

renderformer infer \
  --input /tmp/renderformer-rf1/cbox.h5 \
  --checkpoint microsoft/renderformer-v1.1-swin-large \
  --output-dir /tmp/renderformer-rf1/rendered
```

The bundled example meshes have passed the release provenance review. See the
[per-file manifest](../../examples/rf1/ASSET_PROVENANCE.json) and
[human-readable attributions](../../examples/rf1/ASSET_ATTRIBUTIONS.md).
Gallery scenes retain their listed third-party terms, including Stanford's
research/non-commercial restriction.

Mesh paths in a legacy JSON file are resolved relative to that JSON file. The
composer applies the historical object transforms, normals, material values,
per-triangle/per-shading-group random diffuse behavior, and camera look-at
conversion using CPU mesh operations. It does not render reference images.

### RF1 scene-distribution constraints

RF1 consumes one token per triangle; it is not a conventional rasterizer and
is not invariant to arbitrary retriangulation. Do not use a coarse mesh whose
triangles cover large regions of the rendered view. Each large face contributes
only one geometry token, so an under-tessellated object or receiver can lack the
spatial granularity the model saw during training and produce warped
silhouettes, lighting, or shadows. Remesh first, then use QSlim to obtain a
watertight mesh with approximately uniform, geometrically meaningful faces.
Do not simplify below the point where large faces replace important silhouette
or material-boundary detail. There is no single world-space maximum edge
length: the relevant quantity is the triangle's projected footprint at the
requested camera views. The supplied 1188- and 3968-face preparation tiers are
better starting points than a hand-authored primitive with only a few faces.

Stay close to the public RF1 training range when preparing a new scene:

- keep the scene bounding box near `[-0.5, 0.5]` on each axis;
- place the camera about `1.5` to `2.0` units from the scene center and use a
  field of view between 30 and 60 degrees;
- use the canonical triangular light mesh, with light distance and strength
  patterned after the bundled RF1 examples; and
- keep V1 Base inputs at or below about 4k triangles. Extrapolation may work,
  but is not guaranteed.

For every new public example, render the same scene with Blender/Cycles and
compare camera projection, silhouettes, occlusion, and shadow placement before
calling inference validated. A valid H5, finite EXR values, or a non-constant
PNG proves only that the I/O path ran successfully; it does not prove visual
correctness.

By default it also reproduces the historical float16 material-value
quantization, then stores the result as the required float32 RF1 tensor. The
`--no-legacy-texture-quantization` switch intentionally changes those values;
do not use it for legacy-checkpoint parity. `pymeshlab` is imported only if a scene
sets `remesh: true`.

See [the RF1 examples](../../examples/rf1/README.md) for the scene schema and
batch scripts.

### Exact RF1 H5 contract

Every inference H5 must contain all five keys below, with exactly these dtypes
and shapes:

| Key | dtype | Shape | Meaning |
| --- | --- | --- | --- |
| `triangles` | float32 | `(N, 3, 3)` | Three world-space vertices for each triangle |
| `texture` | float32 | `(N, 13, 32, 32)` | RF1 material/light texture |
| `vn` | float32 | `(N, 3, 3)` | Per-vertex normals aligned with `triangles` |
| `c2w` | float32 | `(V, 4, 4)` | Camera-to-world matrices |
| `fov` | float32 | `(V,)` | One field of view in degrees per camera |

`N` and `V` must both be nonzero. The texture channels are diffuse RGB,
specular RGB, roughness, normal XYZ, and emissive/irradiance RGB. The RF1
lower-triangular texture mask is part of the encoding. An optional `mvp` or
ground-truth `img` may be present, but inference does not require either.

The pipeline does not silently cast a malformed RF1 file into compliance. A
float16 `texture`, a missing normal array, or a mismatched view count fails
before model execution by default. The sole compatibility exception is the
explicit `allow_legacy_rf1_dtypes=True` API option or
`--allow-legacy-rf1-dtypes` CLI flag for the published historical RF1 video
bundle. It accepts only the two signatures documented above, converts them in
memory, and records the conversion in the inference manifest. The strict
`renderformer data validate-h5 --format v1` producer validator is unchanged.

## Preparing modern V2 inputs

The maintained V2 path has two data stages: Blender/Cycles generation writes a
raw H5, then postprocess converts it into the representation consumed by
training and inference.

Generation requires the Python 3.11/OpenVDB 13 data environment. Install the
pinned `bpy==4.5.10` wheel from PyPI and the datagen requirements after Conda
creates the environment:

```bash
conda env create -f environments/datagen.yml
conda activate renderformer-datagen
# Linux/NVIDIA only; use the CPU command in environments/README.md elsewhere.
python -m pip install \
  torch==2.7.1+cu126 torchvision==0.22.1+cu126 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install bpy==4.5.10 --index-url https://pypi.org/simple
python -m pip install -r docker/requirements-datagen.txt
python -m pip install --no-deps -e .
```

First validate the external object, material, and environment collections
described in [the external asset contract](../../external/README.md). Then pass
the resolved lists and roots as runtime overrides:

```bash
renderformer data generate \
  --template src/renderformer/data/templates/full-new-volume/4k-env-volume/corner-single-object.jsonc \
  --object-list /datasets/renderformer-assets/objaverse/lists/objects-3968.txt \
  --object-root /datasets/renderformer-assets/objaverse/objects \
  --texture-list /datasets/renderformer/textures-matsynth.txt \
  --texture-root /datasets/renderformer/textures/matsynth \
  --env-map-list /datasets/renderformer/envmaps-filtered-channel-shuffle-blank.txt \
  --env-map-root /datasets/renderformer/envmaps/filtered-channel-shuffle-512 \
  --profile src/renderformer/data/profiles/v2_training_raw.yaml \
  --output-dir /tmp/renderformer-v2/raw \
  --num-views 4 \
  --resolution 512 \
  --spp 4096 \
  --texture-crop-res 256 \
  --texture-target-res 256 \
  --texture-size 32
```

This profile writes `/tmp/renderformer-v2/raw/v2_training_raw.h5`. The three
BRDF mappers load from their default Hugging Face model IDs;
optional local or alternate model overrides are documented in the external
asset contract. Prepare the non-blank six-permutation HDRI collection with
[`renderformer data shuffle-envmaps`](../../external/README.md#prepare-the-v2-channel-shuffled-environment-maps),
then explicitly construct and record any required blank mixture. The exact
historical blank/non-blank ratio and list order are unavailable. Strict
`no_env` recipes continue to use the blank-only collection.

For the canonical training and repeated-inference path, postprocess the raw H5
before feeding it to the current SSS checkpoints:

```bash
renderformer data convert \
  --input /tmp/renderformer-v2/raw/v2_training_raw.h5 \
  --output /tmp/renderformer-v2/processed/scene.h5 \
  --texture-encoder qwen_vae \
  --model-id Qwen/Qwen-Image \
  --device cuda \
  --torch-dtype float32 \
  --triangle-batch-size 256

renderformer data validate-h5 \
  --input /tmp/renderformer-v2/processed/scene.h5 \
  --format v2
```

Keep that artifact path, switch to the CUDA environment, and run inference:

```bash
conda activate renderformer-cuda
renderformer infer \
  --input /tmp/renderformer-v2/processed/scene.h5 \
  --checkpoint RenderFormer/renderformer-v2 \
  --checkpoint-subfolder transformer_512 \
  --variant v2 \
  --device cuda \
  --precision fp16 \
  --resolution 512 \
  --output-dir /tmp/renderformer-v2/inference
```

Postprocess performs all three training-time transformations: learned material
texture encoding, environment-map rotation/strength application, and
`volume_data` to `volume_density` conversion. The inference loader rejects raw
`volume_data`; this prevents a raw generation artifact from being mistaken for
a model-ready file.

A raw export without `volume_data` can be inferred directly when needed. The
loader applies its channel-last environment map's rotation and strength, and
the pipeline encodes its 15/16-channel material texture. Preprocessing first is
still recommended for training and repeated inference because it freezes and
amortizes the learned texture encoding. A raw volume export must always be
postprocessed.

The maintained Qwen and DC-AE model IDs follow their repository-default
revisions. Use `--revision` only when a specific branch, tag, or commit belongs
in the data receipt. The generated manifest records the requested revision; a
model name alone is not a byte-level data lock.

A V2 H5 contains geometry/cameras plus either an encoded texture whose channel
count and patch size match the checkpoint (unless its config explicitly enables
texture interpolation) or a raw texture that the pipeline can encode. Raw
environment maps are HWC RGB(A) with optional rotation and
strength fields; processed maps are CHW RGB. The encoder input is required to
resolve to `(3, 256, 512)`. If a checkpoint separates `light_strength`, a raw
15/16-channel texture can supply it, while a pre-encoded H5 must contain the
explicit dataset. Processed volumes use
`volume_density`, `volume_position`, `volume_rotation`, `volume_scale`,
`volume_scattering_scale`, and `volume_absorption_scale` with a shared first
dimension. See [renderformer.data](../../src/renderformer/data/README.md) for the full
raw/export/postprocess schema.

## Model version selection

The inference APIs and commands above are the complete public inference
surface. Official model IDs use their Hugging Face repository's default
revision. The pipeline and CLI accept an explicit `revision` for the primary
transformer, and `component_revisions` provides the equivalent opt-in control
for auxiliary encoders. There is no package-level model-ID-to-commit registry.
