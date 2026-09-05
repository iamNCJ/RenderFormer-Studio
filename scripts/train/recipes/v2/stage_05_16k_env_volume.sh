#!/usr/bin/env bash
# V2 release curriculum: 16k-token environment/volume stage.

set -euo pipefail

_RECIPE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${_RECIPE_DIR}/../../_common.sh"

RECIPE_KEY=v2/stage_05_16k_env_volume
FAMILY=v2-200m-sss
STAGE_ORDER=5
STAGE_NAME=stage_05_16k_env_volume
DEFAULT_RUN_ID=v2_stage_05_16k_env_volume
PREDECESSOR_RUN_ID=v2_stage_04_4k_env_volume
TRAINING_VERSION=rf2
MODEL_CONFIG=configs/model/v2/200m_sliding_sink_summary.yaml
DATA_CONFIG=configs/data/v2/stage_05_16k_env_volume.yaml

DATASET_ARGS=(
  --dataset-config.padding-length 16384
  --dataset-config.volume-padding-length 1024
)

TRAINER_ARGS=(
  --trainer-config.weight-decay 0.01
  --trainer-config.use-aug
  --trainer-config.no-turn-to-cam-coord
  --trainer-config.batch-size 8
  --trainer-config.num-view-limit 2
  --trainer-config.img-resolution 512
  --trainer-config.crop-size 512
  --trainer-config.lr 0.0001
  --trainer-config.loss-lpips-weight 0.05
  --trainer-config.num-warmup-steps 1000
  --trainer-config.num-cosine-steps 298000
  --trainer-config.total-steps 300000
  --trainer-config.override-view-tf32-mode
  --trainer-config.gradient-checkpointing
  --trainer-config.gradient-checkpointing-view-tf
)

run_training_recipe "$@"
