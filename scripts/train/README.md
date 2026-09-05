# Curriculum training launchers

This directory is the public training entrypoint for RenderFormer Studio. Each
shell launcher has the same semantic stage name as one data YAML and one data
generation launcher. There are no cloud job definitions and no date-coded run
names.

The release surface contains:

- four V1 launchers: Base and Large, each with the same two data stages;
- eight V2 launchers for the consolidated 200M sliding/sink/summary curriculum;
  and
- [`material_autoencoder.sh`](material_autoencoder.sh), a standalone strict-EXR
  manifest recipe using [`autoencoder.yaml`](../../configs/model/material/autoencoder.yaml).

Model YAML describes architecture. A new model YAML is not copied for every
data stage: both V1 lineages reuse one model YAML each, and all eight V2 stages
reuse `configs/model/v2/200m_sliding_sink_summary.yaml`.

## Stage contract

A training launcher binds these defaults:

- `DATA_CONFIG`: the matching file under `configs/data/`;
- `MODEL_CONFIG`: the shared architecture-only YAML;
- the RF1 or RF2 training runner;
- padding, crop, optimizer, and checkpointing settings; and
- a stable, date-free run ID and predecessor edge.

Trailing Tyro arguments may override tuning defaults such as batch size or
gradient accumulation. Model/data/output/run identity, initialization, and
strict-loading invariants must instead use the documented environment
variables and cannot be replaced by trailing arguments.

The launcher consumes the `paths.txt` emitted by its matching data launcher.
Use either of these input contracts:

```bash
export DATASET_LIST=/datasets/renderformer/custom-output/paths.txt
```

or place each generation output in the conventional stage directory and set
only `DATASET_ROOT`:

```text
DATASET_ROOT/
  v1/stage_01_1k_256res/paths.txt
  v1/stage_02_4k_512res/paths.txt
  v2/stage_01_1k_no_env_no_volume/paths.txt
  ...
```

`DATASET_LIST` takes precedence. Historical artifact keys in the data YAMLs
are provenance records; public training launchers do not select them.

## Generate and train one stage

The data output directory and the training stage name intentionally match:

```bash
export DATASET_ROOT=/datasets/renderformer
export OUTPUT_ROOT=/runs/renderformer
export WANDB_MODE=offline

OUTPUT_DIR="${DATASET_ROOT}/v2/stage_01_1k_no_env_no_volume" \
NUM_SAMPLES=1000 SEED=3407 \
ASSET_MANIFEST=/datasets/renderformer/assets.json \
  scripts/data/recipes/v2/stage_01_1k_no_env_no_volume.sh

scripts/train/recipes/v2/stage_01_1k_no_env_no_volume.sh --validate
NPROC_PER_NODE=8 \
  scripts/train/recipes/v2/stage_01_1k_no_env_no_volume.sh
```

Use `--describe` to inspect the binding, `--print` for a shell-escaped command,
`--print0` for exact NUL-separated argv, and `--validate` before allocating
workers.

Additional arguments are appended as Tyro overrides:

```bash
scripts/train/recipes/v2/stage_04_4k_env_volume.sh \
  --trainer-config.batch-size 4 \
  --trainer-config.gradient-accumulation-steps 2
```

Gradient accumulation is V2-only. V1 launchers reject that override because
the RF1 loop requires an accumulation value of one.

## Initialization between stages

Stage 1 starts from scratch by default. Every continuation has a stable
predecessor. If the predecessor used its default run ID, the launcher reads:

```text
OUTPUT_ROOT/<predecessor-run-id>/checkpoints/latest_checkpoint.json
```

and loads the named checkpoint directory. It reads this metadata directly; it
does not guess a step number or scan checkpoint directories. If a predecessor
used a custom run ID, provide the model component explicitly:

```bash
INIT_WEIGHTS=/runs/experiment/checkpoints/ckpt_STEP \
  scripts/train/recipes/v2/stage_05_16k_env_volume.sh --validate
```

All cross-stage loads add `--trainer-config.load-weights-strict`. They load
model parameters only and start a new optimizer, scheduler, and step counter.
This is different from resuming an interrupted stage.

For a same-stage resume, keep the same `RUN_ID` and set `RESUME=1`:

```bash
RESUME=1 scripts/train/recipes/v2/stage_05_16k_env_volume.sh
```

Resume uses the current run's `latest_checkpoint.json`, restores full
Accelerate state, and ignores cross-stage initialization.

## Curriculum inventory

| Lineage | Stage | Data generation | Training | Model YAML | Predecessor |
| --- | ---: | --- | --- | --- | --- |
| V1 Base | 1 | [`stage_01_1k_256res.sh`](../data/recipes/v1/stage_01_1k_256res.sh) | [`base_full_attention_stage_01_1k_256res.sh`](recipes/v1/base_full_attention_stage_01_1k_256res.sh) | [`200m_full_attention.yaml`](../../configs/model/v1/200m_full_attention.yaml) | scratch |
| V1 Base | 2 | [`stage_02_4k_512res.sh`](../data/recipes/v1/stage_02_4k_512res.sh) | [`base_full_attention_stage_02_4k_512res.sh`](recipes/v1/base_full_attention_stage_02_4k_512res.sh) | same | V1 Base stage 1 |
| V1 Large | 1 | [`stage_01_1k_256res.sh`](../data/recipes/v1/stage_01_1k_256res.sh) | [`large_swin_stage_01_1k_256res.sh`](recipes/v1/large_swin_stage_01_1k_256res.sh) | [`500m_swin.yaml`](../../configs/model/v1/500m_swin.yaml) | scratch |
| V1 Large | 2 | [`stage_02_4k_512res.sh`](../data/recipes/v1/stage_02_4k_512res.sh) | [`large_swin_stage_02_4k_512res.sh`](recipes/v1/large_swin_stage_02_4k_512res.sh) | same | V1 Large stage 1 |
| V2 200M SSS | 1 | [`stage_01_1k_no_env_no_volume.sh`](../data/recipes/v2/stage_01_1k_no_env_no_volume.sh) | [`stage_01_1k_no_env_no_volume.sh`](recipes/v2/stage_01_1k_no_env_no_volume.sh) | [`200m_sliding_sink_summary.yaml`](../../configs/model/v2/200m_sliding_sink_summary.yaml) | scratch |
| V2 200M SSS | 2 | [`stage_02_1k_no_env_no_volume_texture_mix.sh`](../data/recipes/v2/stage_02_1k_no_env_no_volume_texture_mix.sh) | [`stage_02_1k_no_env_no_volume_texture_mix.sh`](recipes/v2/stage_02_1k_no_env_no_volume_texture_mix.sh) | same | stage 1 |
| V2 200M SSS | 3 | [`stage_03_1k_env_volume_mix.sh`](../data/recipes/v2/stage_03_1k_env_volume_mix.sh) | [`stage_03_1k_env_volume_mix.sh`](recipes/v2/stage_03_1k_env_volume_mix.sh) | same | stage 2 |
| V2 200M SSS | 4 | [`stage_04_4k_env_volume.sh`](../data/recipes/v2/stage_04_4k_env_volume.sh) | [`stage_04_4k_env_volume.sh`](recipes/v2/stage_04_4k_env_volume.sh) | same | stage 3 |
| V2 200M SSS | 5 | [`stage_05_16k_env_volume.sh`](../data/recipes/v2/stage_05_16k_env_volume.sh) | [`stage_05_16k_env_volume.sh`](recipes/v2/stage_05_16k_env_volume.sh) | same | stage 4 |
| V2 200M SSS | 6 | [`stage_06_16k_refraction_env_mix.sh`](../data/recipes/v2/stage_06_16k_refraction_env_mix.sh) | [`stage_06_16k_refraction_env_mix.sh`](recipes/v2/stage_06_16k_refraction_env_mix.sh) | same | stage 5 |
| V2 200M SSS | 7 | [`stage_07_64k_env_volume_512_source.sh`](../data/recipes/v2/stage_07_64k_env_volume_512_source.sh) | [`stage_07_64k_env_volume_512_source.sh`](recipes/v2/stage_07_64k_env_volume_512_source.sh) | same | stage 6 |
| V2 200M SSS | 8 | [`stage_08_64k_env_volume_2048_source.sh`](../data/recipes/v2/stage_08_64k_env_volume_2048_source.sh) | [`stage_08_64k_env_volume_2048_source.sh`](recipes/v2/stage_08_64k_env_volume_2048_source.sh) | same | stage 7 |

V2 stages 7 and 8 are sequential: stage 7 trains at native 512 resolution,
then stage 8 model-only warm-starts from stage 7 for native-2048 source renders
with 512-pixel crops.

## Material autoencoder

The material autoencoder is not a renderer curriculum stage. Its manifest has
one JSON object per line with exactly `id`, `path`, and `split`; at least one
`train` row is required and `validation` rows are optional. Validate every EXR,
then launch:

```bash
MANIFEST=/datasets/material/exr-manifest.jsonl \
OUTPUT_DIR=/runs/material-autoencoder \
  scripts/train/material_autoencoder.sh --validate

MANIFEST=/datasets/material/exr-manifest.jsonl \
OUTPUT_DIR=/runs/material-autoencoder \
NPROC_PER_NODE=8 \
  scripts/train/material_autoencoder.sh
```

The launcher explicitly records `log10_1p`, uses L1 reconstruction plus the
historical latent-smoothness term, and supports model-only initialization or
full-state resume. Each run keeps resumable `.pt` checkpoints and exports
model-only `best_model/` and `final_model/` directories for
`MaterialAutoencoder.from_pretrained`. See the
[material training receipt](../../docs/training/README.md#material-autoencoder-recipe).

## Distributed launch

Each launcher invokes `python -m torch.distributed.run`. Single-node launch is
the default. For multiple nodes, run the same stage on every node:

```bash
NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.10 MASTER_PORT=29500 \
NPROC_PER_NODE=8 scripts/train/recipes/v2/stage_06_16k_refraction_env_mix.sh
```

Use `NODE_RANK=1` on the second node. `batch-size` is per rank. V2 effective
global batch is `batch-size × NNODES × NPROC_PER_NODE ×
gradient-accumulation-steps`; V1 omits the final factor.

## Reproduction boundary

These scripts are a consolidated release curriculum, not a byte-exact replay
of internal job history. The historical V2 path used unpublished tex64-to-tex32
and full-to-SSS conversion weights and several sibling experiments, so mapping
those submissions onto eight sequential public stages would invent checkpoint
edges. The public curriculum instead trains the final SSS architecture from
stage 1 and uses strict same-architecture continuation throughout.

The retained stage hyperparameters are the closest defensible representatives
for each public data stage. Unknown historical sample budgets, seeds, exact
list ordering, world sizes, and private scheduler history remain unknown. See
[`docs/training/README.md`](../../docs/training/README.md) for the full
release-stage table and the retained source-group metadata behind each stage.

## Runtime requirements

Use Python 3.10 or 3.11. V1 requires DALI and the V1 CUDA/Triton kernels. V2
also requires FlashAttention. Create `environments/cuda.yml`, install
`docker/requirements-cuda.txt`, and install the pinned FlashAttention wheel as
described in the repository README. Set `WANDB_MODE=offline` or `disabled` for
unattended runs.
