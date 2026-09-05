#!/usr/bin/env bash
# V2 release curriculum: 64k native-512 source and output after stage 6.

set -euo pipefail

_RECIPE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${_RECIPE_DIR}/../../_common.sh"

RECIPE_KEY=v2/stage_07_64k_env_volume_512_source
FAMILY=v2-200m-sss
STAGE_ORDER=7
STAGE_NAME=stage_07_64k_env_volume_512_source
DEFAULT_RUN_ID=v2_stage_07_64k_env_volume_512_source
PREDECESSOR_RUN_ID=v2_stage_06_16k_refraction_env_mix
TRAINING_VERSION=rf2
MODEL_CONFIG=configs/model/v2/200m_sliding_sink_summary.yaml
DATA_CONFIG=configs/data/v2/stage_07_64k_env_volume_512_source.yaml

DATASET_ARGS=(
  --dataset-config.padding-length 65536
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
  --trainer-config.lr 1.25e-05
  --trainer-config.loss-lpips-weight 0.05
  --trainer-config.num-warmup-steps 1000
  --trainer-config.num-cosine-steps 298000
  --trainer-config.total-steps 300000
  --trainer-config.override-view-tf32-mode
  --trainer-config.gradient-checkpointing
  --trainer-config.gradient-checkpointing-view-tf
  --trainer-config.save-interval 100
  --trainer-config.log-img-interval 100
)

run_training_recipe "$@"
