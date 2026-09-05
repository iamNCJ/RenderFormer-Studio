#!/usr/bin/env bash
# Portable V1 data recipe: stage_02_4k_512res

set -euo pipefail

_RECIPE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${_RECIPE_DIR}/../../_common.sh"

DATA_RECIPE=configs/data/v1/stage_02_4k_512res.yaml

run_data_recipe "$@"
