#!/usr/bin/env bash
# V1 Large Swin release curriculum: 1k-token, 256-resolution stage.

set -euo pipefail

_RECIPE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${_RECIPE_DIR}/../../_common.sh"

RECIPE_KEY=v1/large_swin_stage_01_1k_256res
FAMILY=v1-large-swin
STAGE_ORDER=1
STAGE_NAME=stage_01_1k_256res
DEFAULT_RUN_ID=v1_large_swin_stage_01_1k_256res
PREDECESSOR_RUN_ID=''
TRAINING_VERSION=rf1
MODEL_CONFIG=configs/model/v1/500m_swin.yaml
DATA_CONFIG=configs/data/v1/stage_01_1k_256res.yaml

DATASET_ARGS=(
  --dataset-config.padding-length 1572
)

TRAINER_ARGS=(
  --trainer-config.weight-decay 0.01
  --trainer-config.use-aug
  --trainer-config.turn-to-cam-coord
  --trainer-config.batch-size 4
  --trainer-config.num-view-limit 2
  --trainer-config.lr 0.0001
  --trainer-config.loss-lpips-weight 0.05
  --trainer-config.num-warmup-steps 8000
  --trainer-config.num-cosine-steps 592000
  --trainer-config.total-steps 600000
)

run_training_recipe "$@"
