#!/usr/bin/env bash
# Canonical public receipt for Objaverse mesh preparation and object-list output.

set -euo pipefail

_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${_SCRIPT_DIR}/../.." && pwd)"

if [[ -z "${INPUT_MANIFEST:-}" ]]; then
  echo "error: INPUT_MANIFEST must name the explicit Objaverse source JSONL" >&2
  exit 2
fi
if [[ -z "${OUTPUT_DIR:-}" ]]; then
  echo "error: OUTPUT_DIR must name the prepared collection directory" >&2
  exit 2
fi
if [[ ! "${NUM_WORKERS:-1}" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: NUM_WORKERS must be a positive integer" >&2
  exit 2
fi
if [[ "${ALLOW_EMPTY_SPECIAL:-0}" != "0" && "${ALLOW_EMPTY_SPECIAL:-0}" != "1" ]]; then
  echo "error: ALLOW_EMPTY_SPECIAL must be 0 or 1" >&2
  exit 2
fi

python_bin="${PYTHON_BIN:-python}"
command=(
  "${python_bin}" -m renderformer data prepare-objaverse
  --input-manifest "${INPUT_MANIFEST}"
  --output-dir "${OUTPUT_DIR}"
  --num-workers "${NUM_WORKERS:-1}"
  --watertight-backend voxel
  --voxel-resolution 256
  --normalize-radius 0.45
)
if [[ "${ALLOW_EMPTY_SPECIAL:-0}" == "1" ]]; then
  command+=(--allow-empty-special)
fi

cd -- "${REPO_ROOT}"
exec "${command[@]}"
