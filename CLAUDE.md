# RenderFormer Studio repository guide

This is the canonical coding-agent guide for the repository. `AGENTS.md` is a
compatibility symlink to this file so the instructions have one source of
truth.

## Working rules

- Parallelize independent work. Use processes rather than threads for
  CPU-bound batches; threads remain appropriate for streaming subprocess I/O.
- Drive batches from explicit JSON/JSONL/database metadata rather than
  discovering work with filesystem traversal. Record IDs, paths, and status in
  the metadata.
- Size worker pools for the workload and machine. Print progress with
  `flush=True` and include a worker prefix such as `[W0]` when outputs can
  interleave.
- Fail fast. Add exception handling only when there is a defined, actionable
  recovery path.
- Support Python 3.10 and 3.11. Use the active environment for CPU work; use
  `environments/cuda.yml` plus the pinned pip and FlashAttention receipts for
  CUDA, and `environments/datagen.yml` plus PyPI `bpy==4.5.10` for data
  generation. Never hardcode a machine-specific environment prefix.

## Purpose and public surface

RenderFormer Studio contains the data, model, training, inference, and
evaluation code for two model generations:

- V1/RF1: the released diffuse/specular renderer, including the public Base
  and Large checkpoints.
- V2/RF2: heterogeneous primitives, environment lighting, volumes,
  displacement, SVBRDF material latents, and mixed-resolution training.

There is one installable Python package, `renderformer`, under
`src/renderformer/`. The distribution is named `renderformer-studio`, and it
installs one executable with three subcommands:

```bash
renderformer infer --help
renderformer train --help
renderformer data --help
```

The equivalent module form is `python -m renderformer <command> ...`. It works
after an editable install, or with `PYTHONPATH=src` in a source-only
environment. Do not reintroduce repository-root inference or training
wrappers.

## Repository map

```text
src/renderformer/
  pipelines/    RenderFormerPipeline and RenderFormerV2Pipeline
  modeling.py   reusable pretrained/save adapter for RenderFormerModel
  models/       shared numerical model implementation
  data/         V1/V2 generation, H5 schemas, postprocess, and loaders
  training/     independent RF1/RF2/material runners and shared utilities
  ops/          maintained CUDA/Triton operator adapters
  cli/          unified dispatcher and inference command
configs/
  model/v1/     V1 architecture YAML
  model/v2/     V2 architecture YAML
  model/material/  material-autoencoder architecture YAML
  data/v1/      stage-numbered V1 generation YAML
  data/v2/      stage-numbered V2 generation YAML
  data/material/  material-sphere EXR generation YAML
scripts/data/   portable generation recipes and data utilities
scripts/train/  portable curriculum-stage launchers
examples/       runnable RF1/RF2 scene sources and asset provenance
docs/           environment, inference, and training guides
```

Renderer training and inference use the same `RenderFormerModel`. The
inference pipelines assemble version-specific preprocessing and auxiliary
encoders around that component. RF1, RF2, and the material autoencoder retain
separate training loops because their data and optimization flows differ;
`renderformer train` is their shared dispatcher, not a combined numerical
loop.

## Data boundaries

All maintained data code lives under `renderformer.data`:

- `renderformer.data.exporters.v1` owns the strict RF1 13-channel encoding.
- `renderformer.data.exporters.v2` owns the raw V2 export.
- `renderformer.data.textures` owns learned material postprocessing.
- `renderformer.data.h5` owns atomic H5 writing and schema validation.
- `renderformer.data.loaders.v1` owns RF1 PyTorch/DALI loading.
- `renderformer.data.loaders.multires` owns weighted RF2 multiresolution,
  environment, and volume loading.
- `renderformer.data.geometry` owns the reusable local remesh and UV helpers.
- `renderformer.data.material_exr` owns the strict manifest-driven EXR dataset,
  and `renderformer.data.material_preprocessing` owns its invertible
  `log10_1p` transform.
- `renderformer.data.material_spheres` owns deterministic planning, explicit
  render shards, verified resume, and finalization of the three-family EXR
  corpus. Rendering runs only in the active PyPI `bpy==4.5.10` Python.
- `renderformer.models.material` owns the historical material autoencoder and
  three BRDF-to-9-D-latent mapper architectures. The two V2 transformers and
  four material components share the `RenderFormer/renderformer-v2`
  repository; its subfolder layout is listed in the root README. Default mapper
  repository/subfolder pairs live with the Blender material definitions. One
  repository revision selects all six components atomically, and the default
  revision is used unless the caller explicitly requests one. The autoencoder
  release contract is alpha-premultiplied, finite, nonnegative linear RGB
  transformed by `log10(1 + RGB)` and inverted by `10**network_RGB - 1`.

Keep `import renderformer.data` lightweight. Blender, DALI, Diffusers, and
other heavy optional backends must remain behind explicit or lazy imports.
Format conversion belongs in export/postprocess, not in a training loader.

RF1 and RF2 H5 schemas are different. Never infer one from the other. The
exact RF1 contract below is part of the published checkpoint interface.

### RF1 compatibility contract

Every RF1 input H5 must contain:

| Key | dtype | Shape |
| --- | --- | --- |
| `triangles` | float32 | `(num_triangles, 3, 3)` |
| `texture` | float32 | `(num_triangles, 13, 32, 32)` |
| `vn` | float32 | `(num_triangles, 3, 3)` |
| `c2w` | float32 | `(num_views, 4, 4)` |
| `fov` | float32 | `(num_views,)` |

This remains the canonical producer and default inference contract. The only
exception is the separately published historical
`renderformer/renderformer-video-data` bundle. Its launcher explicitly uses
`--allow-legacy-rf1-dtypes`, which accepts only its two observed signatures:
float16 `texture` with float32 `triangles`/`vn`, or float16 `texture` with
float64 `triangles`/`vn`; `c2w` and `fov` remain float32 in both cases. The
loader converts those arrays to float32 in memory and records the original
dtypes. Do not enable this flag for generated or third-party inputs, do not
weaken `validate-h5 --format v1`, and do not broaden the accepted signatures.

Recommended V1 padding length and batch size pairs are 4192/4 for 4k, 8192/2
for 8k, 32768/1 for 32k, 65536/1 for 64k, and 131072/1 for 128k. Use FP16
unless debugging numerical behavior. The V1 Base model was trained on at most
4k triangles, so degradation above that range is expected.

RenderFormer consumes triangle tokens and is not invariant to arbitrary
retriangulation. Do not feed it a coarse mesh whose triangles occupy large
projected image regions: a handful of large faces does not provide the token
granularity seen during training and can produce warped silhouettes and
shadows. Remesh first, then use QSlim to obtain a watertight mesh with roughly
uniform, geometrically meaningful faces. Do not simplify below the point where
large faces replace important silhouette or material-boundary detail. There is
no universal world-space maximum edge length because camera distance and scene
scale determine the projected footprint.

Historical RF1 evaluation outputs use
`{date}_rf1_compat_{scene_type}_rf1_trans_{scene_id}_view_{view_id}.exr` and
GT directories use
`{date}_rf1_compat_{scene_type}_512/{scene_id:04d}_rendered_scene.h5`. Preserve
the date prefix when reproducing those layouts.

Open every candidate H5 with `h5py.File(..., "r")` before a heavy inference
run, rerun conversion for corrupt files, and validate a merged directory again
after merging. `Unable to synchronously open file (bad object header version
number)` usually means an interrupted or OOM-partial write. Use one conversion
worker for very large meshes, especially 128k. If `rf2_h5_dir` has no matching
scene, conversion retains every camera, which can produce 8 views instead of a
filtered 4. If the EXR count is below the selected-view total, check for corrupt
H5 inputs and mixed view counts.

## Training stages

A reproducible curriculum stage has four layers:

1. a stage-numbered data YAML under `configs/data/{v1,v2}/`;
2. its generation launcher under `scripts/data/recipes/{v1,v2}/`;
3. an architecture-only YAML under `configs/model/{v1,v2}/`; and
4. a training launcher under `scripts/train/recipes/{v1,v2}/` that binds the
   generated `paths.txt`, model config, predecessor weights, and trainer
   arguments.

Do not duplicate a model YAML when only data or trainer settings change. Use
[`docs/training/README.md`](docs/training/README.md) as the readable stage
matrix and the shell launchers as the executable source of truth. Public
launcher names and default run IDs are semantic and date-free; historical run
identifiers belong only in retained source-group provenance, never in public
launcher names.

The material autoencoder is independent of the renderer curriculum. Its
architecture YAML is `configs/model/material/autoencoder.yaml`, its input is an
explicit JSONL manifest with required training and optional validation rows, and
`scripts/train/material_autoencoder.sh` is the portable launcher. New
checkpoints must record the preprocessing mode and manifest hash.

## Common checks

Use Python 3.10 or 3.11 from the active environment. From the repository root:

```bash
renderformer data validate-h5 --input /path/to/scene.h5 --format v1
renderformer infer --input /path/to/scene.h5 --checkpoint MODEL --validate-only
```

Use the supported CUDA environment for DALI, FlashAttention, Triton, and GPU
parity checks:

```bash
conda env create -f environments/cuda.yml
conda activate renderformer-cuda
python -m pip install -r docker/requirements-cuda.txt
MAX_JOBS=4 python -m pip install \
  flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
python -m pip install --no-deps -e .
```

Important environment switches include `USE_SDPA=1` for the PyTorch attention
fallback, `BLENDER_BACKEND` for the Cycles backend, and
`OPENCV_IO_ENABLE_OPENEXR=1` when OpenCV reads or writes EXR. Do not document
machine-specific environment prefixes, credentials, or private storage paths.

## Compatibility and failure rules

- Preserve the RF1 exporter byte contract documented below and in
  [`src/renderformer/data/README.md`](src/renderformer/data/README.md).
- V1 templates require white lighting and mono-specular material sampling;
  treat those settings as part of the published checkpoint contract.
- New public examples must be visually compared with a Blender/Cycles reference.
  H5 validity, finite EXR values, and non-constant pixels prove file integrity,
  not rendering correctness.
- The V2 manifest is curriculum-scoped. Final capability comes from the full
  1k to 64k progression, not a single template group.
- For large H5 conversion, especially 128k meshes, use one worker and validate
  every candidate file before and after copying or merging.
- Use process workers for CPU-bound batch work and explicit JSON/JSONL
  manifests instead of directory traversal.
- Fail fast unless an error has a defined, actionable recovery path.

## Documentation routes

- [`README.md`](README.md): installation and concise inference/training entry.
- [`docs/README.md`](docs/README.md): task-oriented documentation index.
- [`docs/inference/README.md`](docs/inference/README.md): pipeline API, H5 preparation, and
  inference workflows.
- [`src/renderformer/data/README.md`](src/renderformer/data/README.md): data
  architecture, schemas, profiles, and programmatic API.
- [`external/README.md`](external/README.md): external collection and BRDF
  mapper contract.
