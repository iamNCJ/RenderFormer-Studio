#!/usr/bin/env bash
# Shared local launcher for strict, portable data-generation recipes.

set -euo pipefail

_DATA_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${_DATA_SCRIPT_DIR}/../.." && pwd)"

_data_recipe_usage() {
  cat <<EOF
Usage: $(basename -- "$0") [--describe | --validate | --print] [--] [BATCH_OVERRIDES...]

Runtime environment:
  OUTPUT_DIR        Directory for a new/resumed seeded primitive generation
                    (task_manifest.json, tasks.jsonl, run_state.json, paths.txt, H5s)
  NUM_SAMPLES       Explicit number of newly generated scenes (required; historical value is unknown)
  SEED              Explicit non-negative task-plan seed (required; historical value is unknown)
  ASSET_MANIFEST    Schema-v1 external asset manifest (required for real generation)
  NUM_WORKERS       Scene-generation process count (default: 1)
  PYTHON_BIN        Python executable (default: python)
  DRY_RUN=1         Write only the deterministic task manifest; no Blender generation
  SKIP_ASSET_ENTRY_CHECK=1
                    Check asset metadata/list locations without stat'ing every list entry

Historical sample count, seed, and list order were not recorded. Every real
run must provide NUM_SAMPLES and SEED and does not claim byte-exact equivalence.
OUTPUT_DIR/paths.txt lists that new primitive output; it does not materialize
the historical artifact keys (patched/merged lists) recorded in the YAML.
Repeating the exact invocation resumes verified outputs. Changed plans, input
fingerprints, corrupt H5s, and mismatched H5 task identities fail explicitly.
EOF
}

_require_nonempty() {
  local name="$1"
  local value="$2"
  if [[ -z "${value}" ]]; then
    echo "error: ${name} must be set" >&2
    return 2
  fi
}

_require_positive_integer() {
  local name="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: ${name} must be a positive integer" >&2
    return 2
  fi
}

_require_nonnegative_integer() {
  local name="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
    echo "error: ${name} must be a non-negative integer" >&2
    return 2
  fi
}

_print_data_command() {
  local -a command=("$@")
  printf 'cd %q\n' "${REPO_ROOT}"
  printf '%q ' "${command[@]}"
  printf '\n'
}

run_data_recipe() {
  local action="run"
  local -a extra_args=()
  while (($#)); do
    case "$1" in
      --describe|--validate|--print)
        if [[ "${action}" != "run" ]]; then
          echo "error: choose only one data-recipe action" >&2
          return 2
        fi
        action="${1#--}"
        shift
        ;;
      --help|-h)
        _data_recipe_usage
        return 0
        ;;
      --)
        shift
        extra_args+=("$@")
        break
        ;;
      *)
        extra_args+=("$1")
        shift
        ;;
    esac
  done

  if [[ -z "${DATA_RECIPE:-}" ]]; then
    echo "error: recipe wrapper did not set DATA_RECIPE" >&2
    return 2
  fi
  local recipe_path="${REPO_ROOT}/${DATA_RECIPE}"
  local python_bin="${PYTHON_BIN:-python}"
  local -a base_command=(
    "${python_bin}" -m renderformer data generate-batch
    --recipe "${recipe_path}"
  )
  cd -- "${REPO_ROOT}"

  if [[ "${action}" == "describe" ]]; then
    "${base_command[@]}" --describe
    return 0
  fi

  local asset_manifest="${ASSET_MANIFEST:-}"
  local skip_entry_check="${SKIP_ASSET_ENTRY_CHECK:-0}"
  if [[ "${skip_entry_check}" != "0" && "${skip_entry_check}" != "1" ]]; then
    echo "error: SKIP_ASSET_ENTRY_CHECK must be 0 or 1" >&2
    return 2
  fi
  local -a asset_args=()
  if [[ -n "${asset_manifest}" ]]; then
    asset_args+=(--asset-manifest "${asset_manifest}")
  elif [[ "${action}" == "print" ]]; then
    asset_args+=(--asset-manifest /path/to/assets.json)
  fi
  if [[ "${skip_entry_check}" == "1" ]]; then
    asset_args+=(--skip-asset-entry-check)
  fi
  if [[ "${action}" == "validate" ]]; then
    "${base_command[@]}" --validate \
      ${asset_args[@]+"${asset_args[@]}"} \
      ${extra_args[@]+"${extra_args[@]}"}
    return 0
  fi

  local output_dir="${OUTPUT_DIR:-}"
  local num_samples="${NUM_SAMPLES:-}"
  local seed="${SEED:-}"
  local num_workers="${NUM_WORKERS:-1}"
  local dry_run="${DRY_RUN:-0}"
  if [[ "${action}" == "print" ]]; then
    output_dir="${output_dir:-/path/to/new-dataset}"
    num_samples="${num_samples:-1000}"
    seed="${seed:-3407}"
  else
    _require_nonempty OUTPUT_DIR "${output_dir}"
    _require_nonempty NUM_SAMPLES "${num_samples}"
    _require_nonempty SEED "${seed}"
  fi
  _require_positive_integer NUM_SAMPLES "${num_samples}"
  _require_nonnegative_integer SEED "${seed}"
  _require_positive_integer NUM_WORKERS "${num_workers}"
  if [[ "${dry_run}" != "0" && "${dry_run}" != "1" ]]; then
    echo "error: DRY_RUN must be 0 or 1" >&2
    return 2
  fi
  if [[ "${action}" != "print" && "${dry_run}" != "1" ]]; then
    _require_nonempty ASSET_MANIFEST "${asset_manifest}"
  fi

  local -a command=(
    "${base_command[@]}"
    --output-dir "${output_dir}"
    --num-samples "${num_samples}"
    --seed "${seed}"
    --num-workers "${num_workers}"
    ${asset_args[@]+"${asset_args[@]}"}
  )
  if [[ "${dry_run}" == "1" ]]; then
    command+=(--dry-run)
  fi
  command+=(${extra_args[@]+"${extra_args[@]}"})

  if [[ "${action}" == "print" ]]; then
    _print_data_command "${command[@]}"
    return 0
  fi
  "${command[@]}"
}
