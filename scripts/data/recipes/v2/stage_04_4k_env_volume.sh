#!/usr/bin/env bash
# Portable V2 data recipe: stage_04_4k_env_volume

set -euo pipefail

_RECIPE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${_RECIPE_DIR}/../../_common.sh"

DATA_RECIPE=configs/data/v2/stage_04_4k_env_volume.yaml

run_data_recipe "$@"
