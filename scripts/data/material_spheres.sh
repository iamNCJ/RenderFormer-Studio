#!/usr/bin/env bash
# Plan, render, and finalize the manifest-driven material-sphere EXR dataset.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CALLER_CWD="$(pwd -P)"

usage() {
  cat <<'EOF'
Usage: scripts/data/material_spheres.sh ACTION

Actions (choose exactly one):
  --validate    Validate and describe the YAML; does not import bpy.
  --plan        Write/verify the deterministic plan and tasks.jsonl.
  --render      Render the explicit SHARD_INDEX.
  --finalize    Verify all EXRs and write material-training.jsonl.

Runtime environment for --plan, --render, and --finalize:
  OUTPUT_DIR        Dataset directory (required)
  ENVIRONMENT_MAP   Lat-long EXR path (required)
  SEED              Non-negative plan seed (required)
  NUM_SHARDS        Fixed shard count recorded in the plan (required)

Additional environment for --render:
  SHARD_INDEX       Zero-based shard index (required)
  DEVICE            cpu, cuda, optix, hip, metal, or oneapi (default: cpu)
  DEVICE_INDEX      Required zero-based backend index for non-CPU DEVICE

Optional environment:
  CONFIG            Recipe YAML (default: configs/data/material/material_spheres.yaml)
  PYTHON_BIN        Active Python containing PyPI bpy==4.5.10 (default: python)

Run --plan once, launch exactly one process for every shard index, then run
--finalize. Repeating an identical shard verifies and skips completed samples.
No Blender executable or GPU-discovery command is used.
EOF
}

absolute_from_caller() {
  local path="$1"
  if [[ "${path}" == /* ]]; then
    printf '%s' "${path}"
  else
    printf '%s/%s' "${CALLER_CWD}" "${path}"
  fi
}

if (($# != 1)); then
  usage >&2
  exit 2
fi
case "$1" in
  --validate) action=validate ;;
  --plan) action=plan ;;
  --render) action=render ;;
  --finalize) action=finalize ;;
  --help|-h) usage; exit 0 ;;
  *) echo "error: unknown action: $1" >&2; usage >&2; exit 2 ;;
esac

python_bin="${PYTHON_BIN:-python}"
config="${CONFIG:-${REPO_ROOT}/configs/data/material/material_spheres.yaml}"
config="$(absolute_from_caller "${config}")"
command=(
  "${python_bin}" -m renderformer data material-spheres
  --recipe "${config}"
)

if [[ "${action}" == validate ]]; then
  command+=(--validate)
else
  : "${OUTPUT_DIR:?OUTPUT_DIR is required}"
  : "${ENVIRONMENT_MAP:?ENVIRONMENT_MAP is required}"
  : "${SEED:?SEED is required}"
  : "${NUM_SHARDS:?NUM_SHARDS is required}"
  if [[ ! "${SEED}" =~ ^[0-9]+$ ]]; then
    echo "error: SEED must be a non-negative integer" >&2
    exit 2
  fi
  if [[ ! "${NUM_SHARDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: NUM_SHARDS must be a positive integer" >&2
    exit 2
  fi
  command+=(
    --output-dir "$(absolute_from_caller "${OUTPUT_DIR}")"
    --environment-map "$(absolute_from_caller "${ENVIRONMENT_MAP}")"
    --seed "${SEED}"
    --num-shards "${NUM_SHARDS}"
  )
  if [[ "${action}" == plan ]]; then
    command+=(--plan)
  elif [[ "${action}" == finalize ]]; then
    command+=(--finalize)
  else
    : "${SHARD_INDEX:?SHARD_INDEX is required for --render}"
    if [[ ! "${SHARD_INDEX}" =~ ^[0-9]+$ ]]; then
      echo "error: SHARD_INDEX must be a non-negative integer" >&2
      exit 2
    fi
    device="${DEVICE:-cpu}"
    command+=(--render-shard "${SHARD_INDEX}" --device "${device}")
    if [[ "${device}" != cpu ]]; then
      : "${DEVICE_INDEX:?DEVICE_INDEX is required for a non-CPU DEVICE}"
      if [[ ! "${DEVICE_INDEX}" =~ ^[0-9]+$ ]]; then
        echo "error: DEVICE_INDEX must be a non-negative integer" >&2
        exit 2
      fi
      command+=(--device-index "${DEVICE_INDEX}")
    elif [[ -n "${DEVICE_INDEX:-}" ]]; then
      echo "error: DEVICE_INDEX is valid only for a non-CPU DEVICE" >&2
      exit 2
    fi
  fi
fi

cd -- "${REPO_ROOT}"
exec "${command[@]}"
