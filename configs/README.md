# Configuration ownership

Release configuration is split by responsibility:

- `model/` defines reusable model structure only;
- `data/` defines stage-specific generation and postprocessing; and
- `scripts/train/recipes/` binds one model YAML and one data stage to loader,
  optimizer, schedule, and checkpoint settings.

This replaces the old per-run curriculum YAML format, which mixed model
structure, private dataset paths, trainer flags, and infrastructure settings.

## Model configs

Four architecture files form the public training surface:

| Model YAML | Public use |
| --- | --- |
| [`model/v1/200m_full_attention.yaml`](model/v1/200m_full_attention.yaml) | V1 Base stages 1–2 |
| [`model/v1/500m_swin.yaml`](model/v1/500m_swin.yaml) | V1 Large stages 1–2 |
| [`model/v2/200m_sliding_sink_summary.yaml`](model/v2/200m_sliding_sink_summary.yaml) | all eight V2 200M stages |
| [`model/material/autoencoder.yaml`](model/material/autoencoder.yaml) | standalone 9-D material autoencoder |

A model YAML is reused when only data, padding, crop, or trainer settings
change. V2 therefore does not carry a separate precursor architecture in the
public curriculum. The consolidated release trains the final
sliding/sink/summary structure from stage 1 and strict-loads it between stages.

Each renderer model file contains exactly one `model_config` mapping. The
material YAML is a direct, strict architecture mapping because its standalone
runner does not consume the renderer configuration schema. Dataset and trainer
fields do not belong in either form.

## Data configs

The ten renderer YAML files under `data/` define two V1 and eight V2 generation
stages. The standalone
[`data/material/material_spheres.yaml`](data/material/material_spheres.yaml)
defines the three-family EXR corpus used by the material autoencoder. Renderer
stage files record:

- a stable semantic `recipe_id` and export profile;
- scene-template paths and sampling weights;
- render resolution, views, texture settings, and corrections;
- V2 material/environment postprocessing; and
- retained artifact metadata and explicitly unknown historical fields.

Cloud queues, identities, mounted paths, and worker allocations are not part
of these configs.

The material recipe fixes the 256-pixel sphere/camera/render setup, the three
historical parameter distributions and sample counts, 128 clean validation
samples per family, alpha premultiplication, and `log10_1p`. Its environment
map, root seed, output directory, and shard count are explicit runtime inputs;
none is a machine-specific path in the YAML.

### V1

| Stage YAML | Generation launcher | Training launchers |
| --- | --- | --- |
| [`stage_01_1k_256res.yaml`](data/v1/stage_01_1k_256res.yaml) | [`stage_01_1k_256res.sh`](../scripts/data/recipes/v1/stage_01_1k_256res.sh) | Base and Large stage 1 |
| [`stage_02_4k_512res.yaml`](data/v1/stage_02_4k_512res.yaml) | [`stage_02_4k_512res.sh`](../scripts/data/recipes/v1/stage_02_4k_512res.sh) | Base and Large stage 2 |

The rounded 1k/4k labels are curriculum bands; training padding is 1572/4192.
Both stages enable the recovered mesh-aware background-topology correction
from the beginning.

### V2

| Stage YAML | Generation launcher | Purpose |
| --- | --- | --- |
| [`stage_01_1k_no_env_no_volume.yaml`](data/v2/stage_01_1k_no_env_no_volume.yaml) | [`stage_01_1k_no_env_no_volume.sh`](../scripts/data/recipes/v2/stage_01_1k_no_env_no_volume.sh) | 1k, no environment or volume |
| [`stage_02_1k_no_env_no_volume_texture_mix.yaml`](data/v2/stage_02_1k_no_env_no_volume_texture_mix.yaml) | [`stage_02_1k_no_env_no_volume_texture_mix.sh`](../scripts/data/recipes/v2/stage_02_1k_no_env_no_volume_texture_mix.sh) | add the texture/no-texture mix |
| [`stage_03_1k_env_volume_mix.yaml`](data/v2/stage_03_1k_env_volume_mix.yaml) | [`stage_03_1k_env_volume_mix.sh`](../scripts/data/recipes/v2/stage_03_1k_env_volume_mix.sh) | add environment maps and volumes |
| [`stage_04_4k_env_volume.yaml`](data/v2/stage_04_4k_env_volume.yaml) | [`stage_04_4k_env_volume.sh`](../scripts/data/recipes/v2/stage_04_4k_env_volume.sh) | scale to 4k |
| [`stage_05_16k_env_volume.yaml`](data/v2/stage_05_16k_env_volume.yaml) | [`stage_05_16k_env_volume.sh`](../scripts/data/recipes/v2/stage_05_16k_env_volume.sh) | scale to 16k |
| [`stage_06_16k_refraction_env_mix.yaml`](data/v2/stage_06_16k_refraction_env_mix.yaml) | [`stage_06_16k_refraction_env_mix.sh`](../scripts/data/recipes/v2/stage_06_16k_refraction_env_mix.sh) | refraction with 1:3 environment/no-environment weights |
| [`stage_07_64k_env_volume_512_source.yaml`](data/v2/stage_07_64k_env_volume_512_source.yaml) | [`stage_07_64k_env_volume_512_source.sh`](../scripts/data/recipes/v2/stage_07_64k_env_volume_512_source.sh) | 64k native-512 source/output stage |
| [`stage_08_64k_env_volume_2048_source.yaml`](data/v2/stage_08_64k_env_volume_2048_source.yaml) | [`stage_08_64k_env_volume_2048_source.sh`](../scripts/data/recipes/v2/stage_08_64k_env_volume_2048_source.sh) | 64k native-2048 source continuation |

Environment-map RGB channel permutation is enabled in every V2 YAML. It is a
release-wide policy, not a late curriculum stage. Blank environment maps remain
blank under permutation.

## Runtime output versus retained artifact metadata

Every generation launcher writes a new `OUTPUT_DIR/paths.txt` plus task and run
metadata. Public training launchers consume that file directly via
`DATASET_LIST`, or through the conventional
`DATASET_ROOT/<version>/<stage>/paths.txt` layout.

The `artifacts` mappings in data YAML retain old list identities and known row
counts for provenance. They are not promises that a new generation run can
reconstruct a private byte stream. Exact historical sample budgets, random
seeds, list ordering, and some source filters were not recovered.

The training manifests retain the historical source-group metadata needed to
interpret those fields. Use
[`docs/training/README.md`](../docs/training/README.md) for the date-free
public curricula.
