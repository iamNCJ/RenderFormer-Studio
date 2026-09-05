# Training recipes

This guide describes the public RenderFormer Studio curricula. It is organized
by conceptual data stage: generate one stage, train on that stage, then pass
the resulting model component to the next stage. All commands are plain Bash
and work without AML or Kubernetes.

The executable source of truth is:

```text
configs/data/<version>/<stage>.yaml
scripts/data/recipes/<version>/<stage>.sh
configs/model/<version>/<model>.yaml
scripts/train/recipes/<version>/<stage>.sh
```

Before generating stages, build the three object-list inputs (the 1188- and
3968-face-cap lists plus the explicit special-object list) from cataloged
Objaverse source meshes with the
[Objaverse data receipt](../../scripts/data/README.md#prepare-the-objaverse-mesh-collection).
Those lists are generated artifacts with face-cap, watertightness, UV, and hash
metadata; they are not opaque files shipped under `external/`.

V1 exposes two data stages for each released model family. V2 exposes eight
data stages for one 200M sliding/sink/summary model. A model YAML is reused
whenever architecture is unchanged.

## What a stage means

A stage number names a data and scale transition, not an internal submission
or a checkpoint step. Its data YAML defines scene templates, render settings,
topology/environment corrections, and postprocessing. Its training shell binds
that YAML to one architecture config and to the loader/trainer settings used at
that scale.

The generation launcher always writes a new `OUTPUT_DIR/paths.txt`. The
training launcher consumes that exact list through either:

- `DATASET_LIST=/absolute/path/to/paths.txt`; or
- `DATASET_ROOT/<version>/<stage>/paths.txt` by convention.

Source-group metadata inside data YAML documents the recipe inputs. Public
stage launchers never select a dated artifact or private dataset path.

## Quick start

Create the supported CUDA environment first, then install the CUDA
requirements and FlashAttention as described in the main README.

Set shared roots:

```bash
export DATASET_ROOT=/datasets/renderformer
export OUTPUT_ROOT=/runs/renderformer
export ASSET_MANIFEST=/datasets/renderformer/assets.json
export WANDB_MODE=offline
```

Generate V2 stage 1 into its conventional directory:

```bash
OUTPUT_DIR="${DATASET_ROOT}/v2/stage_01_1k_no_env_no_volume" \
NUM_SAMPLES=1000 SEED=3407 \
  scripts/data/recipes/v2/stage_01_1k_no_env_no_volume.sh
```

Validate and launch its matching training stage:

```bash
scripts/train/recipes/v2/stage_01_1k_no_env_no_volume.sh --validate
NPROC_PER_NODE=8 \
  scripts/train/recipes/v2/stage_01_1k_no_env_no_volume.sh
```

For a custom generation output, bypass the conventional layout explicitly:

```bash
DATASET_LIST=/datasets/custom/stage-one/paths.txt \
  scripts/train/recipes/v2/stage_01_1k_no_env_no_volume.sh --validate
```

The launcher still binds the stage's data YAML and model YAML; `DATASET_LIST`
only chooses the generated list location.

## V1 curriculum

V1 has exactly two data stages:

1. nominal 1k-token scenes rendered at 256 resolution, with loader padding
   1572; and
2. nominal 4k-token scenes rendered at 512 resolution, with loader padding
   4192.

Both portable data stages apply the mesh-aware background-topology correction
from the beginning. The correction is a release-wide data policy, not a later
fine-tuning stage.

### V1 data and launcher mapping

| Stage | Data YAML | Generate | Base training | Large training |
| ---: | --- | --- | --- | --- |
| 1 | [`stage_01_1k_256res.yaml`](../../configs/data/v1/stage_01_1k_256res.yaml) | [`stage_01_1k_256res.sh`](../../scripts/data/recipes/v1/stage_01_1k_256res.sh) | [`base_full_attention_stage_01_1k_256res.sh`](../../scripts/train/recipes/v1/base_full_attention_stage_01_1k_256res.sh) | [`large_swin_stage_01_1k_256res.sh`](../../scripts/train/recipes/v1/large_swin_stage_01_1k_256res.sh) |
| 2 | [`stage_02_4k_512res.yaml`](../../configs/data/v1/stage_02_4k_512res.yaml) | [`stage_02_4k_512res.sh`](../../scripts/data/recipes/v1/stage_02_4k_512res.sh) | [`base_full_attention_stage_02_4k_512res.sh`](../../scripts/train/recipes/v1/base_full_attention_stage_02_4k_512res.sh) | [`large_swin_stage_02_4k_512res.sh`](../../scripts/train/recipes/v1/large_swin_stage_02_4k_512res.sh) |

Base uses
[`configs/model/v1/200m_full_attention.yaml`](../../configs/model/v1/200m_full_attention.yaml)
for both stages. Large uses
[`configs/model/v1/500m_swin.yaml`](../../configs/model/v1/500m_swin.yaml)
for both stages.

### V1 Base settings

| Stage | Padding | Image | Batch/rank | LR | Warmup / cosine / total | Gradient checkpointing |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 1572 | 256 | 16 | `1e-4` | `8k / 1.2M / 1.3M` | view transformer |
| 2 | 4192 | 512 | 8 | `5e-5` | `8k / 92k / 100k` | image and view transformers |

Both stages use weight decay `0.01`, augmentation, camera-coordinate
conversion, two training views, and LPIPS weight `0.05`.

### V1 Large settings

| Stage | Padding | Image | Batch/rank | LR | Warmup / cosine / total | Gradient checkpointing |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 1572 | 256 | 4 | `1e-4` | `8k / 592k / 600k` | none |
| 2 | 4192 | 512 | 6 | `5e-5` | `4k / 46k / 50k` | image and view transformers |

The remaining common settings match V1 Base.

## V2 curriculum

All eight stages use one final architecture config:
[`configs/model/v2/200m_sliding_sink_summary.yaml`](../../configs/model/v2/200m_sliding_sink_summary.yaml).
The public curriculum therefore has no architecture conversion step and every
cross-stage initialization is strict.

The checkpoint graph is:

```text
stage 1 → stage 2 → stage 3 → stage 4 → stage 5 → stage 6
        → stage 7 (64k, 512 source/output)
        → stage 8 (64k, 2048 source with 512 training crops)
```

Stage 7 first establishes 64k training at native 512 resolution. Stage 8 then
warm-starts from the stage 7 model component and switches to native-2048 source
renders while retaining 512-pixel training crops.

### V2 data and launcher mapping

| Stage | Semantic purpose | Data YAML | Generate / train |
| ---: | --- | --- | --- |
| 1 | 1k, no environment, no volume | [`stage_01_1k_no_env_no_volume.yaml`](../../configs/data/v2/stage_01_1k_no_env_no_volume.yaml) | [`generate`](../../scripts/data/recipes/v2/stage_01_1k_no_env_no_volume.sh) / [`train`](../../scripts/train/recipes/v2/stage_01_1k_no_env_no_volume.sh) |
| 2 | 1k texture/no-texture mixture | [`stage_02_1k_no_env_no_volume_texture_mix.yaml`](../../configs/data/v2/stage_02_1k_no_env_no_volume_texture_mix.yaml) | [`generate`](../../scripts/data/recipes/v2/stage_02_1k_no_env_no_volume_texture_mix.sh) / [`train`](../../scripts/train/recipes/v2/stage_02_1k_no_env_no_volume_texture_mix.sh) |
| 3 | 1k environment and volume mixture | [`stage_03_1k_env_volume_mix.yaml`](../../configs/data/v2/stage_03_1k_env_volume_mix.yaml) | [`generate`](../../scripts/data/recipes/v2/stage_03_1k_env_volume_mix.sh) / [`train`](../../scripts/train/recipes/v2/stage_03_1k_env_volume_mix.sh) |
| 4 | 4k environment and volume | [`stage_04_4k_env_volume.yaml`](../../configs/data/v2/stage_04_4k_env_volume.yaml) | [`generate`](../../scripts/data/recipes/v2/stage_04_4k_env_volume.sh) / [`train`](../../scripts/train/recipes/v2/stage_04_4k_env_volume.sh) |
| 5 | 16k environment and volume | [`stage_05_16k_env_volume.yaml`](../../configs/data/v2/stage_05_16k_env_volume.yaml) | [`generate`](../../scripts/data/recipes/v2/stage_05_16k_env_volume.sh) / [`train`](../../scripts/train/recipes/v2/stage_05_16k_env_volume.sh) |
| 6 | 16k refraction; environment:no-environment = 1:3 | [`stage_06_16k_refraction_env_mix.yaml`](../../configs/data/v2/stage_06_16k_refraction_env_mix.yaml) | [`generate`](../../scripts/data/recipes/v2/stage_06_16k_refraction_env_mix.sh) / [`train`](../../scripts/train/recipes/v2/stage_06_16k_refraction_env_mix.sh) |
| 7 | 64k with native-512 source/output training | [`stage_07_64k_env_volume_512_source.yaml`](../../configs/data/v2/stage_07_64k_env_volume_512_source.yaml) | [`generate`](../../scripts/data/recipes/v2/stage_07_64k_env_volume_512_source.sh) / [`train`](../../scripts/train/recipes/v2/stage_07_64k_env_volume_512_source.sh) |
| 8 | 64k with native-2048 source renders and 512 training crops | [`stage_08_64k_env_volume_2048_source.yaml`](../../configs/data/v2/stage_08_64k_env_volume_2048_source.yaml) | [`generate`](../../scripts/data/recipes/v2/stage_08_64k_env_volume_2048_source.sh) / [`train`](../../scripts/train/recipes/v2/stage_08_64k_env_volume_2048_source.sh) |

The 1:3 ratio in stage 6 is encoded in template weights, not as a runtime
loader override. Every V2 data YAML applies the six offline RGB environment-map
channel permutations. This is also a release-wide policy rather than a
separate stage; blank maps remain blank under permutation.

### V2 settings

`pad/vol` is triangle/volume padding. `image/crop` is the source-render
resolution and training crop.

| Stage | pad/vol | image/crop | Batch/rank | LR | Gradient checkpointing | Predecessor |
| ---: | --- | --- | ---: | ---: | --- | --- |
| 1 | `1536 / 2` | `256 / 256` | 4 | `1e-4` | none | scratch |
| 2 | `1536 / 2` | `256 / 256` | 8 | `1e-4` | none | stage 1 |
| 3 | `1536 / 512` | `256 / 256` | 16 | `1e-4` | none | stage 2 |
| 4 | `4096 / 1024` | `512 / 512` | 8 | `7.5e-5` | image transformer | stage 3 |
| 5 | `16384 / 1024` | `512 / 512` | 8 | `1e-4` | image and view transformers | stage 4 |
| 6 | `16384 / 1024` | `512 / 512` | 8 | `5e-5` | image and view transformers | stage 5 |
| 7 | `65536 / 1024` | `512 / 512` | 8 | `1.25e-5` | image and view transformers | stage 6 |
| 8 | `65536 / 1024` | `2048 / 512` | 8 | `1.25e-5` | image and view transformers | stage 7 |

Every V2 stage uses weight decay `0.01`, augmentation, no camera-coordinate
conversion, two training views, LPIPS weight `0.05`, TF32 view-transformer
override, and a `1k / 298k / 300k` warmup/cosine/total schedule. Stages 7 and
8 save checkpoints and log images every 100 steps. These are launcher defaults;
record any override with the resulting run metadata.

## Material autoencoder recipe

The 9-D material autoencoder is a standalone training path, not a renderer
curriculum stage. Its checkpoint-compatible architecture is fixed by
[`configs/model/material/autoencoder.yaml`](../../configs/model/material/autoencoder.yaml),
and [`scripts/train/material_autoencoder.sh`](../../scripts/train/material_autoencoder.sh)
binds the data and optimization receipt.

Data is an explicit JSONL manifest. Every nonblank row contains exactly an
`id`, EXR `path`, and `split` (`train` or `validation`). Relative paths resolve
from the manifest directory. IDs and resolved paths must be unique, so a file
cannot appear in both splits. Training rows are required; validation rows are
optional. The loader reads no unlisted file and never retries a failed sample.
The public [material-sphere data receipt](../../scripts/data/README.md#generate-the-material-autoencoder-exrs)
creates this manifest from the three recovered material families with an
immutable seeded plan and explicit render shards.

The physical input contract is finite, nonnegative linear RGB. RGBA inputs are
alpha-premultiplied by default, then transformed to network space as:

```text
network_RGB = log10(1 + linear_RGB)
linear_RGB  = 10**network_RGB - 1
```

Validate the complete manifest without constructing the model, then train:

```bash
MANIFEST=/datasets/material/exr-manifest.jsonl \
OUTPUT_DIR=/runs/material-autoencoder \
  scripts/train/material_autoencoder.sh --validate

MANIFEST=/datasets/material/exr-manifest.jsonl \
OUTPUT_DIR=/runs/material-autoencoder \
NPROC_PER_NODE=8 \
  scripts/train/material_autoencoder.sh
```

The public defaults are batch size 64 per rank, 50 epochs, AdamW at `1e-4`, L1
reconstruction, and latent smoothness weight 2 with Gaussian latent noise
standard deviation `2/255`. Set `SMOOTHNESS_WEIGHT=0` to train reconstruction
only. DDP is enabled only when `NPROC_PER_NODE` or `NNODES` exceeds one; W&B is
imported only when `WANDB=1`.

`RESUME=/path/to/checkpoint.pt` restores model, optimizer, scheduler, epoch,
and global step under an identical recorded contract. `LOAD_MODEL` strictly
loads only model weights for a fresh optimizer. The original standalone legacy
`.pt` predates embedded preprocessing metadata, so warm-starting that file
additionally requires `ALLOW_MISSING_PREPROCESSING_METADATA=1`; this is an
explicit acknowledgment, not preprocessing auto-detection. The official Hugging
Face component includes `preprocessor_config.json` and needs no acknowledgment:

```bash
MANIFEST=/datasets/material/exr-manifest.jsonl \
OUTPUT_DIR=/runs/material-autoencoder-finetune \
LOAD_MODEL=RenderFormer/renderformer-v2 \
LOAD_MODEL_SUBFOLDER=material_autoencoder \
  scripts/train/material_autoencoder.sh
```

Set `LOAD_MODEL_REVISION` as well when the complete V2 component set must come
from a particular branch, tag, or commit.

New runs write the manifest hash, preprocessing mode, alpha policy, objective,
and architecture into every resumable `.pt` checkpoint. They also export
`best_model/` and `final_model/` as model-only `config.json` +
`model.safetensors` + `preprocessor_config.json` bundles that load directly
with `MaterialAutoencoder.from_pretrained`.

For provenance, the selected historical run saw 60,110 principled-metallic,
60,000 diffuse/specular, and 100,000 principled-transmission directories. The
extra 110 principled samples have no recovered task manifest and appear to be
retained exploratory renders, so newly generated IDs cannot match them. The
220,110 total at batch size 64 explains 3440 steps per epoch. The run requested
50 epochs, learning rate `1e-4`, latent size 9, 256-pixel images, smoothness
weight 2, and 128 validation samples per dataset; its zero-based epoch 44 /
step 154800 checkpoint represents 45 completed epochs.

That run was a model-only fine-tune from a predecessor whose checkpoint is
unavailable, and its historical validation selections were also reachable
through training data. Therefore it is not an exact from-scratch or
clean-holdout reproduction recipe. The public generator retains the recovered
family counts but assigns 384 unique validation paths, leaving 219,726 disjoint
training paths. This clean manifest is recommended, and its metrics are not
presented as directly comparable to the leaked historical validation.

## Stage initialization and resume

The launchers distinguish model-only continuation from full-state resume.

For model-only continuation, a non-first stage resolves its predecessor from:

```text
OUTPUT_ROOT/<stable-predecessor-run-id>/checkpoints/latest_checkpoint.json
```

The JSON pointer names the checkpoint directory. This avoids filesystem scans
and hard-coded checkpoint step numbers. The selected directory must contain
`model.safetensors`. The launcher passes it as `load_weights` together with
`load_weights_strict=true`; optimizer, scheduler, and global step start fresh.

If a predecessor used a custom `RUN_ID`, provide an explicit model component:

```bash
INIT_WEIGHTS=/runs/custom-stage/checkpoints/ckpt_STEP \
DATASET_LIST=/datasets/custom-next-stage/paths.txt \
  scripts/train/recipes/v2/stage_04_4k_env_volume.sh --validate
```

For an interrupted stage, use the same `RUN_ID` and full-state resume:

```bash
RESUME=1 scripts/train/recipes/v2/stage_04_4k_env_volume.sh
```

`RESUME=1` requires the current run's `latest_checkpoint.json`, adds
`auto_resume`, and does not load predecessor weights.

## Inspect before launching

Each recipe supports four non-running actions:

```bash
scripts/train/recipes/v2/stage_06_16k_refraction_env_mix.sh --describe
scripts/train/recipes/v2/stage_06_16k_refraction_env_mix.sh --print
scripts/train/recipes/v2/stage_06_16k_refraction_env_mix.sh --print0
scripts/train/recipes/v2/stage_06_16k_refraction_env_mix.sh --validate
```

- `--describe` prints stable model/data/DAG metadata.
- `--print` emits a shell-escaped `torchrun` command and permits placeholder
  paths.
- `--print0` emits exact command argv separated by NUL bytes.
- `--validate` checks the data list, model/data config, initialization, and
  existing-run state without starting workers.

The launcher protects model, data, output, run identity, and strict
initialization arguments from trailing overrides. Boolean Tyro flags do not
take `true` or `false`; use their positive or `no-` spelling.

## Distributed execution

The public installed command is `renderformer train`, while curriculum scripts
invoke its module form through `torchrun`:

```bash
python -m torch.distributed.run \
  --standalone --nproc-per-node 8 \
  --module renderformer.training.cli rf2 \
  --config_file configs/model/v2/200m_sliding_sink_summary.yaml \
  [TYRO_OPTIONS]
```

For multiple nodes, launch the same stage on each node with a unique rank:

```bash
NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.10 MASTER_PORT=29500 \
NPROC_PER_NODE=8 scripts/train/recipes/v2/stage_06_16k_refraction_env_mix.sh
```

The second node uses `NODE_RANK=1`. `OMP_NUM_THREADS` defaults to 12.
`WANDB_MODE=offline` is recommended for unattended work.

## Release curriculum policy

These recipes define the canonical portable curriculum for the source
release. V1 applies the background-topology correction in both stages. V2 uses
the final SSS architecture and environment-map channel shuffling from stage 1;
stage 7 establishes the 64k native-512 regime, and stage 8 warm-starts from it
with native-2048 source renders.

Dataset roots, concrete file lists, sample budgets, seeds, world size, and
output locations are deployment inputs. Record them alongside each run when
exact experiment repetition is required. Published model IDs follow their
Hugging Face repositories' default revisions. A stage that requires a specific
upstream branch, tag, or commit must state it explicitly in that stage's data
or trainer configuration.
