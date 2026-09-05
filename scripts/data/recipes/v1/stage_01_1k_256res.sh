#!/usr/bin/env bash
# Portable V1 data recipe: stage_01_1k_256res

set -euo pipefail

_RECIPE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${_RECIPE_DIR}/../../_common.sh"

DATA_RECIPE=configs/data/v1/stage_01_1k_256res.yaml

run_data_recipe "$@"
