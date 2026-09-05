#!/usr/bin/env bash
# Portable V2 data recipe: stage_07_64k_env_volume_512_source

set -euo pipefail

_RECIPE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${_RECIPE_DIR}/../../_common.sh"

DATA_RECIPE=configs/data/v2/stage_07_64k_env_volume_512_source.yaml

run_data_recipe "$@"
