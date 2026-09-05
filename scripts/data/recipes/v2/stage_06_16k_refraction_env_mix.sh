#!/usr/bin/env bash
# Portable V2 data recipe: stage_06_16k_refraction_env_mix

set -euo pipefail

_RECIPE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${_RECIPE_DIR}/../../_common.sh"

DATA_RECIPE=configs/data/v2/stage_06_16k_refraction_env_mix.yaml

run_data_recipe "$@"
