#!/usr/bin/env bash
# V1 Base full-attention release curriculum: 4k-token, 512-resolution stage.

set -euo pipefail

_RECIPE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${_RECIPE_DIR}/../../_common.sh"

RECIPE_KEY=v1/base_full_attention_stage_02_4k_512res
FAMILY=v1-base-full-attention
STAGE_ORDER=2
STAGE_NAME=stage_02_4k_512res
DEFAULT_RUN_ID=v1_base_full_attention_stage_02_4k_512res
PREDECESSOR_RUN_ID=v1_base_full_attention_stage_01_1k_256res
TRAINING_VERSION=rf1
MODEL_CONFIG=configs/model/v1/200m_full_attention.yaml
DATA_CONFIG=configs/data/v1/stage_02_4k_512res.yaml

DATASET_ARGS=(
  --dataset-config.padding-length 4192
)

TRAINER_ARGS=(
  --trainer-config.weight-decay 0.01
  --trainer-config.use-aug
  --trainer-config.turn-to-cam-coord
  --trainer-config.batch-size 8
  --trainer-config.num-view-limit 2
  --trainer-config.img-resolution 512
  --trainer-config.lr 5e-05
  --trainer-config.loss-lpips-weight 0.05
  --trainer-config.num-warmup-steps 8000
  --trainer-config.num-cosine-steps 92000
  --trainer-config.total-steps 100000
  --trainer-config.gradient-checkpointing
  --trainer-config.gradient-checkpointing-view-tf
)

run_training_recipe "$@"
