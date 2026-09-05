# Portable data recipes

This directory contains the canonical Objaverse preparation receipt, 10
portable renderer scene-generation launchers, and the standalone material-sphere
EXR launcher. YAML owns each generation distribution; shell owns only local
execution. No Azure queue, table, identity, mount, AML, or Kubernetes resource
is required.

```text
scripts/data/recipes/v1/   2 V1 scale-stage primitives
scripts/data/recipes/v2/   8 V2 200M scale/data stages
scripts/data/material_spheres.sh   material-autoencoder EXRs
```

Each wrapper sets exactly one `DATA_RECIPE` and delegates to `_common.sh`,
which invokes `renderformer data generate-batch`.

Within V1 and V2 separately, `stage_XX` is a contiguous scale/data-curriculum
number. It is independent of the training-receipt `SXX` number because runs can
reuse a generated partition and several dated source partitions can belong to
one scale stage. V1 has exactly two such stages: nominal 1k tokens at 256
resolution and nominal 4k tokens at 512 resolution. Their actual padded tensor
lengths are 1572 and 4192. Dated artifact paths remain as historical
provenance.

## Generate the material-autoencoder EXRs

[`material_spheres.sh`](material_spheres.sh) reproduces the recovered sphere,
camera, material-distribution, and render settings for all three material
families. The selected checkpoint saw directory counts of 60,110 principled
metallic, 60,000 diffuse/specular, and 100,000 principled transmission samples;
these are the YAML counts. The smaller 6000/1500/2500 constants in the old
scripts were per-process loop caps, not final corpus sizes. The launcher runs
under the active Python with the pinned PyPI `bpy==4.5.10`; it never invokes a
Blender application executable or discovers GPUs.

Supply the exact 2048 x 1024 monochrome lat-long EXR used for lighting. It was
formed by taking the arithmetic mean of the Uffizi probe's RGB channels and
repeating the mean into three channels; its SHA-256 is fixed in the YAML as
`b1da0a8259daeba85ecc338065aecf169e3e24ffa9dc2feda073a4440ac4faff`.
The binary has no recoverable source/license receipt and is not redistributed
here. To use another environment map, copy the YAML, change the recorded hash,
and treat the result as a new compatible dataset rather than a reproduction.
Create one immutable plan before starting workers:

```bash
export OUTPUT_DIR=/datasets/renderformer/material-spheres
export ENVIRONMENT_MAP=/datasets/lighting/uffizi-probe-mono.exr
export SEED=3407
export NUM_SHARDS=8

scripts/data/material_spheres.sh --validate
scripts/data/material_spheres.sh --plan
```

Launch exactly one process for every zero-based shard index. CPU is the
portable default:

```bash
SHARD_INDEX=0 scripts/data/material_spheres.sh --render
SHARD_INDEX=1 scripts/data/material_spheres.sh --render
# Continue through SHARD_INDEX=7, potentially on separate machines that share
# OUTPUT_DIR and use the exact same plan inputs.
```

For a GPU worker, select the backend and index explicitly; the launcher does
not inspect `nvidia-smi` or free memory:

```bash
CUDA_VISIBLE_DEVICES=0 DEVICE=cuda DEVICE_INDEX=0 SHARD_INDEX=0 \
  scripts/data/material_spheres.sh --render
```

After all shards finish, verify every EXR and publish the training manifest:

```bash
scripts/data/material_spheres.sh --finalize
```

The output contract is:

```text
material-spheres/
  material-sphere-plan.json
  tasks.jsonl
  samples/<family>/<deterministic-id>/render.exr
  samples/<family>/<deterministic-id>/metadata.json
  records/<deterministic-id>.json
  shards/shard-<index>.json
  material-training.jsonl
  material-sphere-receipt.json
```

The root seed deterministically fixes IDs, parameters, clean train/validation
assignments, and `global_index % NUM_SHARDS`. A repeated shard skips a sample
only after matching its task metadata, opening its 256 x 256 RGB EXR, checking
finite nonnegative values, and matching its SHA-256 record. Changed plans and
corrupt or replaced outputs fail. Finalization reopens every planned EXR and
writes only the exact `id`, relative `path`, and `split` fields accepted by the
Material AE loader. Rendered RGBA is alpha-premultiplied and stored as linear
RGB; training then applies `log10(1 + RGB)`.

## Prepare the Objaverse mesh collection

Scene generation consumes relative mesh-path lists, historically called object
ID lists. They are generated data artifacts, not files to copy into
`external/` by hand. The canonical public receipt is
[`prepare_objaverse.sh`](prepare_objaverse.sh). It accepts only an explicit
JSONL source catalog and never discovers files by walking an Objaverse checkout.

Acquire the source objects from [Objaverse](https://objaverse.allenai.org/).
This repository does not download or redistribute those meshes. Export one
JSON object per local mesh, using its stable Objaverse UID as `id`:

```json
{"id":"00000000-0000-0000-0000-000000000001","path":"glbs/00000000-0000-0000-0000-000000000001.glb","special":false}
{"id":"00000000-0000-0000-0000-000000000002","path":"glbs/00000000-0000-0000-0000-000000000002.glb","special":true}
```

Relative paths resolve against the JSONL file. IDs must match
`[A-Za-z0-9][A-Za-z0-9._-]*` and must be unique. `special=true` is an explicit
public selection for the V2 refraction stage; at least one row must carry it.
The historical
`special-mesh-full.txt` selection was not recoverable. Do not label a new choice
as the paper's exact object subset. If stage 6 will not be generated,
`ALLOW_EMPTY_SPECIAL=1` permits an empty compatibility list.

Create the Python 3.11 datagen environment first. Its pinned pip receipt now
contains `libigl==2.5.1`, `scikit-image==0.25.2`, `bpy-helper==0.0.13`, and the
PyPI `bpy==4.5.10` installation described in
[`environments/README.md`](../../environments/README.md). Then run:

```bash
INPUT_MANIFEST=/datasets/objaverse/sources.jsonl \
OUTPUT_DIR=/datasets/renderformer-assets/objaverse \
NUM_WORKERS=4 \
  scripts/data/prepare_objaverse.sh
```

The fixed public receipt applies, independently to every source:

1. center and normalize to radius `0.45`;
2. reconstruct a closed surface on a `256`-cell voxel grid and require it to
   be watertight;
3. run `igl.qslim` independently at maximum face counts 1188 and 3968;
4. require each simplified topology to remain watertight;
5. apply Blender cube-project UV unwrapping; and
6. require an indexed UV map, finite nonempty geometry, the face cap, and the
   watertight topology after ignoring UV/normal seam duplication.

`1188` and `3968` are maximum face counts, not list lengths. For an object that
already has fewer faces, the corresponding tier remains below the cap.

The output is a self-contained, resumable collection:

```text
objaverse/
  objaverse-plan.json
  objaverse-receipt.json
  processed-objects.jsonl
  records/<uid>.json
  objects/<uid>/.renderformer-record.json
  objects/<uid>/mesh_1188.obj
  objects/<uid>/mesh_3968.obj
  lists/objects-1188.txt
  lists/objects-3968.txt
  lists/objects-special.txt
```

Every list row is a unique, sorted path relative to the `objaverse/objects`
root and includes `.obj`. `objects-special.txt` contains the 3968-face outputs
for only the input rows marked `special=true`.

The plan fingerprints the input JSONL, parameters, and exact canonical
dependency versions. It rejects a different NumPy, SciPy, trimesh, libigl,
scikit-image, `bpy`, or `bpy-helper` build before processing. Per-object records
fingerprint the source and both outputs. The hidden record is published
atomically with its two meshes; `records/<uid>.json` is its collection-level
copy. If a run is interrupted, repeat the identical command: verified
completed objects are skipped, including an object interrupted before that copy
was written. Changed source bytes, manifest content, parameters, dependencies,
or outputs fail instead of silently entering a list. Final list hashes and the
sorted per-object record stream are stored in `objaverse-receipt.json`.

`renderformer-objaverse-watertight-qslim-v1` is the versioned algorithm ID in
the plan. Any output-affecting implementation change must increment that ID so
one resumed collection cannot mix two implementations.

This public pipeline preserves the recovered watertight/QSlim/UV and face-cap
contracts. It does not claim byte-equivalence with the former internal workers,
whose adaptive SDF trials, random target counts, train/test split seed, and
special-object selection were not fully recorded.

Map the three generated lists to the collection root with the local manifest
format documented in [`external/README.md`](../../external/README.md), then run
`renderformer data validate-assets --asset-manifest ...` before generating a
stage.

## Inspect and validate

Recipe inspection does not import Blender or require external assets:

```bash
scripts/data/recipes/v2/stage_04_4k_env_volume.sh --describe
scripts/data/recipes/v2/stage_04_4k_env_volume.sh --print
```

Validate packaged templates and, when supplied, the external asset manifest:

```bash
ASSET_MANIFEST=/datasets/renderformer/assets.json \
  scripts/data/recipes/v2/stage_04_4k_env_volume.sh --validate
```

Set `SKIP_ASSET_ENTRY_CHECK=1` only when validating asset-manifest metadata and
list locations without stat-ing every external list entry.

## Deterministic planning

Historical sample counts, seeds, and list order were not recorded. Every new
run must choose `NUM_SAMPLES` and `SEED` explicitly. A dry run writes a
deterministic `task_manifest.json` plus streaming `tasks.jsonl` without Blender
generation:

```bash
OUTPUT_DIR=/tmp/renderformer-stage04-plan \
NUM_SAMPLES=1000 \
SEED=3407 \
DRY_RUN=1 \
  scripts/data/recipes/v2/stage_04_4k_env_volume.sh
```

The manifest records recipe/profile/template hashes and, when used, the exact
external asset manifest plus every metadata-selected collection-list path and
SHA-256. `tasks.jsonl` records the selected template, task seed, and output name
in task-index order without building a million-row JSON object in memory. Task
seeds use a root-seed-keyed 32-bit permutation, so up to `2**32` task indices
cannot collide; template sampling has its own RNG stream. This determinism
applies to the new run; seed 3407 is an example and is not claimed to be the
missing historical seed.

## Generate a real partition

Real generation requires a schema-v1 external asset manifest. It uses process
workers for scene generation and consumes results as workers finish instead of
collecting them with `Pool.map`. V2 postprocessing begins as raw scenes arrive,
and one learned encoder instance is reused for the entire run.

```bash
export ASSET_MANIFEST=/datasets/renderformer/assets.json

OUTPUT_DIR=/datasets/renderformer/stage_04_4k_env_volume \
NUM_SAMPLES=100000 \
SEED=123456 \
NUM_WORKERS=8 \
  scripts/data/recipes/v2/stage_04_4k_env_volume.sh
```

The output directory receives:

- `task_manifest.json`, the immutable plan and input fingerprints;
- `tasks.jsonl`, the streaming task-index record referenced by the manifest;
- `run_state.json`, atomically updated progress for resume diagnostics;
- one final H5 per generated task; V2 also retains one raw H5 per task until
  you remove it after the corresponding final H5 has been verified;
- `paths.txt`, containing absolute paths to completed H5 files; and
- `run_manifest.json`, which records the recipe, count, task-plan and input
  fingerprints, output kind, and the explicit fact that historical byte
  equivalence is false.

Budget V2 disk space for both raw and final H5 files during generation. Once a
run is complete, raw V2 H5 files may be deleted to reclaim space; a later
resume accepts a verified final output without requiring its raw intermediate.

## Resume and failure rules

Re-run the same launcher with the same `OUTPUT_DIR`, recipe, `NUM_SAMPLES`,
`SEED`, mode, and assets after an interruption. The runner validates the task
JSONL digest and every existing H5's schema and embedded task identity. Valid
final outputs are skipped; valid V2 raw H5s are postprocessed without being
rendered again. `paths.txt` is always emitted in stable task-index order even
though process workers finish out of order.

Resume is deliberately strict. A changed recipe/profile/template, edited
asset manifest or collection list, changed seed/count/mode, corrupt H5, or H5
from another task stops immediately. Remove or relocate the explicitly named
bad output only after reviewing the error; the runner never silently mixes
inputs or labels a new run as a historical artifact.

For a plain `paths` artifact, set `OUTPUT_DIR` to the parent directory recorded
by the artifact so the new `paths.txt` is directly consumable by its training
stage. The primitive launcher always writes `paths.txt`; it does not label a
new list as a historical `paths_patched`, `paths_merge_tex`, `paths_merged`, or
`merged` artifact. Those keys preserve historical patch/merge contracts and
their known source provenance. Reconstruct a new combined list explicitly when
using those distributions, and record its new seed/order rather than claiming
that it is the missing internal list.

V1 recipes export the RF1 13-channel schema directly and automatically apply
their required `background_topology_debias` policy. Every V2 recipe enables
`env_map_channel_shuffle` from stage 1, then exports raw scenes and applies the
texture/environment/volume postprocess pinned in the data YAML. Channel shuffle
means the six offline RGB permutations of every filtered HDRI; it is not file,
task, or loader shuffling. A strict blank EXR remains blank, so `no_env`
recipes do not acquire environment lighting. The Qwen VAE revision and dtype
are therefore recipe data rather than hidden shell defaults.

## Train from the generated partition

The V2 4k partition above feeds public training stage 4. Generate it into the
conventional stage directory, then run the matching training launcher:

```bash
export DATASET_ROOT=/datasets/renderformer
export OUTPUT_ROOT=/runs/renderformer
export WANDB_MODE=offline

OUTPUT_DIR="${DATASET_ROOT}/v2/stage_04_4k_env_volume" \
NUM_SAMPLES=1000 SEED=3407 ASSET_MANIFEST=/datasets/renderformer/assets.json \
  scripts/data/recipes/v2/stage_04_4k_env_volume.sh

scripts/train/recipes/v2/stage_04_4k_env_volume.sh --validate
NPROC_PER_NODE=8 scripts/train/recipes/v2/stage_04_4k_env_volume.sh
```

The training launcher resolves:

```text
DATA_CONFIG=configs/data/v2/stage_04_4k_env_volume.yaml
DATASET_ROOT/v2/stage_04_4k_env_volume/paths.txt
MODEL_CONFIG=configs/model/v2/200m_sliding_sink_summary.yaml
```

It resolves stage 3 through that run's `latest_checkpoint.json`, or accepts an
explicit `INIT_WEIGHTS`, before launching `torchrun` with strict model loading.

## Primitive recipes and retained artifacts

| Family | Data YAML | Generation shell | Retained artifact keys |
| --- | --- | --- | --- |
| V1 | [`stage_01_1k_256res.yaml`](../../configs/data/v1/stage_01_1k_256res.yaml) | [`stage_01_1k_256res.sh`](recipes/v1/stage_01_1k_256res.sh) | `historical_250112_paths_patched`, `historical_250418_paths` |
| V1 | [`stage_02_4k_512res.yaml`](../../configs/data/v1/stage_02_4k_512res.yaml) | [`stage_02_4k_512res.sh`](recipes/v1/stage_02_4k_512res.sh) | `historical_250416_paths`, `historical_250428_merged`, `historical_250510_merged` |
| V2 | [`stage_01_1k_no_env_no_volume.yaml`](../../configs/data/v2/stage_01_1k_no_env_no_volume.yaml) | [`stage_01_1k_no_env_no_volume.sh`](recipes/v2/stage_01_1k_no_env_no_volume.sh) | `paths` |
| V2 | [`stage_02_1k_no_env_no_volume_texture_mix.yaml`](../../configs/data/v2/stage_02_1k_no_env_no_volume_texture_mix.yaml) | [`stage_02_1k_no_env_no_volume_texture_mix.sh`](recipes/v2/stage_02_1k_no_env_no_volume_texture_mix.sh) | `paths`, `paths_merge_tex` |
| V2 | [`stage_03_1k_env_volume_mix.yaml`](../../configs/data/v2/stage_03_1k_env_volume_mix.yaml) | [`stage_03_1k_env_volume_mix.sh`](recipes/v2/stage_03_1k_env_volume_mix.sh) | `paths_merged` |
| V2 | [`stage_04_4k_env_volume.yaml`](../../configs/data/v2/stage_04_4k_env_volume.yaml) | [`stage_04_4k_env_volume.sh`](recipes/v2/stage_04_4k_env_volume.sh) | `paths` |
| V2 | [`stage_05_16k_env_volume.yaml`](../../configs/data/v2/stage_05_16k_env_volume.yaml) | [`stage_05_16k_env_volume.sh`](recipes/v2/stage_05_16k_env_volume.sh) | `historical_260120_paths`, `historical_260206_paths` |
| V2 | [`stage_06_16k_refraction_env_mix.yaml`](../../configs/data/v2/stage_06_16k_refraction_env_mix.yaml) | [`stage_06_16k_refraction_env_mix.sh`](recipes/v2/stage_06_16k_refraction_env_mix.sh) | `historical_260212_merged`, `historical_260216_merged` |
| V2 | [`stage_07_64k_env_volume_512_source.yaml`](../../configs/data/v2/stage_07_64k_env_volume_512_source.yaml) | [`stage_07_64k_env_volume_512_source.sh`](recipes/v2/stage_07_64k_env_volume_512_source.sh) | `paths` |
| V2 | [`stage_08_64k_env_volume_2048_source.yaml`](../../configs/data/v2/stage_08_64k_env_volume_2048_source.yaml) | [`stage_08_64k_env_volume_2048_source.sh`](recipes/v2/stage_08_64k_env_volume_2048_source.sh) | `paths` |

There are 10 generation recipes and 16 retained artifact keys. The two
V1 scale stages expose five historically consumed lists. V2 has eight
conceptual stages and eight executable recipes. Its release inventory contains
nine semantic source-template groups and 84 templates, each at a unique
packaged path. Stage 2 exposes two retained artifacts, and stage 5 exposes both
the earlier dated `historical_260120_paths` list and the folded
`historical_260206_paths` list.
Those two historical groups share the deduplicated `16k-env-volume` physical
templates in the release. The source-only
`250428_oxl_high_res/paths.txt` is recorded through merged-artifact provenance,
not exposed as another retained training artifact.

Stage 6 is one public refraction mixture. Its 260212 templates use non-blank
shuffled environment maps and retain internal weights summing to 16. Its
260216 stronger-glass templates use a blank environment and retain their
internal proportions after a threefold scale, summing to 48. This yields a
1:3 environment-to-blank template-weight ratio, and both groups disable
volumes. Historically, training used the 260212 distribution first and then
continued with 260216 under the same experiment ID; the public mixture makes
both distributions available throughout this stage. Running the public wrapper
writes that new mixture to `OUTPUT_DIR/paths.txt`. The two named
`historical_*_merged` artifacts remain exact references to the old separate
lists and are not regenerated by the mixed recipe.

## Merged artifacts and the reproduction boundary

The following V1 row counts survive in the historical trace:

| Artifact | 250416 rows | 250428 rows | 250510 rows | Total |
| --- | ---: | ---: | ---: | ---: |
| `250428_oxl_high_res/merged.txt` | 2,014,579 | 6,986,617 | 0 | 9,001,196 |
| stage 2 `historical_250510_merged` → `250510_fix_template_bias/merged.txt` | 1,343,338 | 4,656,662 | 4,576,280 | 10,576,280 |

Those counts are provenance, not a promise that one new invocation can
recreate the original rows. The repository does not contain every primitive
sample budget, historical random-seed stream, exact list ordering/filter, or
external queue record. The launchers therefore provide strict,
distribution-compatible generation with an auditable new task plan. They do
not claim byte-exact reconstruction of the internal datasets.

Both V1 recipe YAMLs enable the strict background-topology correction. A
portable run therefore applies the corrected mesh-orientation and placement
policy from data stage 1, whereas the surviving historical candidate lineage
introduced it late. Stage 2 retains the old 250510 list, its distinct
14-template/weight-42 sampling distribution, and its merge composition while
referencing the corresponding retained 250428 template files. This dated
partition is provenance inside the 4k/512 stage, not a separate fix stage.

See [`../../configs/README.md`](../../configs/README.md) for config ownership
and [`../../docs/training/README.md`](../../docs/training/README.md) for the
public stage bindings and checkpoint graph.
