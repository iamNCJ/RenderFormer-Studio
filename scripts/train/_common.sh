#!/usr/bin/env bash
# Shared local launcher for the public V1/V2 curriculum stages.

set -euo pipefail

_TRAIN_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${_TRAIN_SCRIPT_DIR}/../.." && pwd)"
CALLER_CWD="$(pwd -P)"

_recipe_usage() {
  cat <<EOF
Usage: $(basename -- "$0") [--print | --print0 | --describe | --validate] [--] [TYRO_OVERRIDES...]

Data:
  DATASET_ROOT       Root containing <version>/<stage>/paths.txt
  DATASET_LIST       Explicit paths.txt from the matching data stage; overrides
                     DATASET_ROOT (one of DATASET_ROOT/DATASET_LIST is required)

Initialization and output:
  OUTPUT_ROOT        Root for logs and checkpoints (required)
  RUN_ID             Override this stage's stable, date-free default run ID
  INIT_WEIGHTS       Explicit model.safetensors or component directory. For a
                     continuation stage this overrides automatic predecessor
                     lookup through latest_checkpoint.json.
  RESUME=1           Resume this same run from OUTPUT_ROOT/RUN_ID/checkpoints

Distributed runtime:
  NPROC_PER_NODE     Local worker/GPU count for torchrun (default: 1)
  NNODES             Number of nodes (default: 1)
  NODE_RANK          This node's zero-based rank (required when NNODES > 1)
  MASTER_ADDR        Rank-zero host/IP (required when NNODES > 1)
  MASTER_PORT        Rendezvous port (default: 29500)
  OMP_NUM_THREADS    CPU threads per worker (default: 12)
  PYTHON_BIN         Python executable (default: python)
  WANDB_MODE         Use offline/disabled for unattended runs; online needs login

Extra Tyro arguments are appended last. Stage identity, dataset location,
model config, output location, initialization, and strict loading cannot be
overridden through extra arguments.
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

_absolute_from_caller() {
  local path="$1"
  if [[ "${path}" == /* ]]; then
    printf '%s' "${path}"
  else
    printf '%s/%s' "${CALLER_CWD}" "${path}"
  fi
}

_validate_recipe_definition() {
  if [[ "${TRAINING_VERSION}" != "rf1" && "${TRAINING_VERSION}" != "rf2" ]]; then
    echo "error: TRAINING_VERSION must be rf1 or rf2" >&2
    return 2
  fi
  local data_family="v${TRAINING_VERSION#rf}"
  local expected_data_config="configs/data/${data_family}/${STAGE_NAME}.yaml"
  local expected_data_launcher="scripts/data/recipes/${data_family}/${STAGE_NAME}.sh"
  if [[ "${DATA_CONFIG}" != "${expected_data_config}" ]]; then
    echo "error: ${RECIPE_KEY} must bind ${expected_data_config}, got ${DATA_CONFIG}" >&2
    return 2
  fi
  if [[ ! -f "${REPO_ROOT}/${DATA_CONFIG}" ]]; then
    echo "error: data config not found: ${REPO_ROOT}/${DATA_CONFIG}" >&2
    return 2
  fi
  if [[ ! -x "${REPO_ROOT}/${expected_data_launcher}" ]]; then
    echo "error: data launcher not found or not executable: ${REPO_ROOT}/${expected_data_launcher}" >&2
    return 2
  fi
  if [[ ! -f "${REPO_ROOT}/${MODEL_CONFIG}" ]]; then
    echo "error: model config not found: ${REPO_ROOT}/${MODEL_CONFIG}" >&2
    return 2
  fi
  if [[ -z "${DEFAULT_RUN_ID}" || "${DEFAULT_RUN_ID}" == */* ]]; then
    echo "error: DEFAULT_RUN_ID must be one path component" >&2
    return 2
  fi
  if [[ -n "${PREDECESSOR_RUN_ID}" && "${PREDECESSOR_RUN_ID}" == */* ]]; then
    echo "error: PREDECESSOR_RUN_ID must be empty or one path component" >&2
    return 2
  fi
}

_resolve_predecessor_weights() {
  local python_bin="$1"
  local output_root="$2"
  local predecessor_run_id="$3"
  local pointer="${output_root}/${predecessor_run_id}/checkpoints/latest_checkpoint.json"

  if [[ ! -f "${pointer}" ]]; then
    echo "error: predecessor checkpoint metadata not found: ${pointer}" >&2
    echo "Set INIT_WEIGHTS to an explicit model checkpoint if the predecessor used a custom RUN_ID." >&2
    return 2
  fi

  "${python_bin}" -c '
import json
from pathlib import Path
import re
import sys

pointer = Path(sys.argv[1])
payload = json.loads(pointer.read_text(encoding="utf-8"))
checkpoint = payload.get("checkpoint") if isinstance(payload, dict) else None
if not isinstance(checkpoint, str) or re.fullmatch(r"ckpt_[0-9]+", checkpoint) is None:
    raise ValueError(f"invalid latest checkpoint metadata: {pointer}")
checkpoint_dir = pointer.parent / checkpoint
weights = checkpoint_dir / "model.safetensors"
if not weights.is_file():
    raise FileNotFoundError(f"model component not found: {weights}")
print(checkpoint_dir)
' "${pointer}"
}

_validate_recipe_paths() {
  local dataset_file="$1"
  local init_weights="$2"
  local output_root="$3"
  local run_id="$4"
  local resume="$5"

  if [[ ! -f "${REPO_ROOT}/src/renderformer/training/cli.py" ]]; then
    echo "error: training CLI not found: ${REPO_ROOT}/src/renderformer/training/cli.py" >&2
    return 2
  fi
  if [[ ! -f "${dataset_file}" ]]; then
    echo "error: dataset list not found: ${dataset_file}" >&2
    return 2
  fi
  if [[ -n "${init_weights}" ]]; then
    if [[ -d "${init_weights}" ]]; then
      if [[ ! -f "${init_weights}/model.safetensors" ]]; then
        echo "error: initialization directory has no model.safetensors: ${init_weights}" >&2
        return 2
      fi
    elif [[ ! -f "${init_weights}" || "$(basename -- "${init_weights}")" != "model.safetensors" ]]; then
      echo "error: INIT_WEIGHTS must be model.safetensors or a component directory: ${init_weights}" >&2
      return 2
    fi
  fi

  local checkpoint_dir="${output_root}/${run_id}/checkpoints"
  local pointer="${checkpoint_dir}/latest_checkpoint.json"
  if [[ "${resume}" == "1" ]]; then
    if [[ ! -f "${pointer}" ]]; then
      echo "error: RESUME=1 but checkpoint metadata is missing: ${pointer}" >&2
      return 2
    fi
  elif [[ -f "${pointer}" ]]; then
    echo "error: ${pointer} already records a checkpoint" >&2
    echo "Set RESUME=1 to resume this run, or choose a different RUN_ID." >&2
    return 2
  fi
}

_print_command() {
  local -a command=("$@")
  printf 'cd %q\n' "${REPO_ROOT}"
  printf 'OMP_NUM_THREADS=%q ' "${OMP_NUM_THREADS:-12}"
  printf '%q ' "${command[@]}"
  printf '\n'
}

run_training_recipe() {
  local action="run"
  local -a extra_args=()

  while (($#)); do
    case "$1" in
      --print|--print0|--describe|--validate)
        if [[ "${action}" != "run" ]]; then
          echo "error: choose only one recipe action" >&2
          return 2
        fi
        action="${1#--}"
        shift
        ;;
      --help|-h)
        _recipe_usage
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

  _validate_recipe_definition

  local data_family="v${TRAINING_VERSION#rf}"
  local data_launcher="scripts/data/recipes/${data_family}/${STAGE_NAME}.sh"
  local dataset_relative_path="${data_family}/${STAGE_NAME}/paths.txt"
  if [[ "${action}" == "describe" ]]; then
    printf 'recipe\t%s\n' "${RECIPE_KEY}"
    printf 'family\t%s\n' "${FAMILY}"
    printf 'order\t%s\n' "${STAGE_ORDER}"
    printf 'stage_name\t%s\n' "${STAGE_NAME}"
    printf 'training_cli\t%s\n' 'renderformer.training.cli'
    printf 'training_version\t%s\n' "${TRAINING_VERSION}"
    printf 'model_config\t%s\n' "${MODEL_CONFIG}"
    printf 'data_config\t%s\n' "${DATA_CONFIG}"
    printf 'data_launcher\t%s\n' "${data_launcher}"
    printf 'dataset_default\t%s\n' "${dataset_relative_path}"
    printf 'default_run_id\t%s\n' "${DEFAULT_RUN_ID}"
    printf 'predecessor_run_id\t%s\n' "${PREDECESSOR_RUN_ID}"
    printf 'initialization\t%s\n' "$([[ -n "${PREDECESSOR_RUN_ID}" ]] && printf predecessor || printf scratch)"
    return 0
  fi

  local python_bin="${PYTHON_BIN:-python}"
  if [[ "${python_bin}" == */* ]]; then
    python_bin="$(_absolute_from_caller "${python_bin}")"
  fi
  local dataset_root="${DATASET_ROOT:-}"
  local dataset_file="${DATASET_LIST:-}"
  local output_root="${OUTPUT_ROOT:-}"
  local run_id="${RUN_ID:-${DEFAULT_RUN_ID}}"
  local resume="${RESUME:-0}"
  local init_weights="${INIT_WEIGHTS:-}"

  if [[ "${resume}" != "0" && "${resume}" != "1" ]]; then
    echo "error: RESUME must be 0 or 1" >&2
    return 2
  fi
  if [[ "${run_id}" == */* || "${run_id}" == "." || "${run_id}" == ".." ]]; then
    echo "error: RUN_ID must be one path component: ${run_id}" >&2
    return 2
  fi

  local extra_arg
  for extra_arg in ${extra_args[@]+"${extra_args[@]}"}; do
    if [[ "${FAMILY}" == v1-* ]]; then
      case "${extra_arg}" in
        --trainer-config.gradient-accumulation-steps|\
        --trainer-config.gradient-accumulation-steps=*|\
        --trainer-config.gradient_accumulation_steps|\
        --trainer-config.gradient_accumulation_steps=*)
          echo "error: V1 stages require trainer_config.gradient_accumulation_steps == 1; gradient-accumulation overrides are supported only by V2" >&2
          return 2
          ;;
      esac
    fi
    case "${extra_arg}" in
      --config_file|--config-file|\
      --dataset-config.h5-folder-path|--dataset_config.h5_folder_path|\
      --trainer-config.log-dir|--trainer-config.exp-id|\
      --trainer-config.load-weights*|--trainer-config.no-load-weights*|\
      --trainer-config.load-ckpt*|--trainer-config.load-optimizer-state*|\
      --trainer-config.auto-resume*|--trainer-config.no-auto-resume*|\
      --trainer_config.log_dir|--trainer_config.exp_id|\
      --trainer_config.load_weights*|--trainer_config.no_load_weights*|\
      --trainer_config.load_ckpt*|--trainer_config.load_optimizer_state*|\
      --trainer_config.auto_resume*|--trainer_config.no_auto_resume*|\
      --model-config.*|--model_config.*|\
      --config_file=*|--config-file=*|\
      --dataset-config.h5-folder-path=*|--dataset_config.h5_folder_path=*|\
      --trainer-config.log-dir=*|--trainer-config.exp-id=*|\
      --trainer_config.log_dir=*|--trainer_config.exp_id=*)
        echo "error: override ${extra_arg} through DATASET_LIST, OUTPUT_ROOT, RUN_ID, or INIT_WEIGHTS" >&2
        return 2
        ;;
    esac
  done

  if [[ -n "${dataset_root}" ]]; then
    dataset_root="$(_absolute_from_caller "${dataset_root}")"
  fi
  if [[ -n "${dataset_file}" ]]; then
    dataset_file="$(_absolute_from_caller "${dataset_file}")"
  fi
  if [[ -n "${output_root}" ]]; then
    output_root="$(_absolute_from_caller "${output_root}")"
  fi
  if [[ -n "${init_weights}" ]]; then
    init_weights="$(_absolute_from_caller "${init_weights}")"
  fi

  if [[ "${action}" == "print" || "${action}" == "print0" ]]; then
    if [[ -z "${dataset_file}" ]]; then
      dataset_root="${dataset_root:-/path/to/datasets}"
      dataset_file="${dataset_root}/${dataset_relative_path}"
    fi
    output_root="${output_root:-/path/to/outputs}"
    if [[ "${resume}" != "1" && -z "${init_weights}" && -n "${PREDECESSOR_RUN_ID}" ]]; then
      init_weights="${output_root}/${PREDECESSOR_RUN_ID}/checkpoints/ckpt_STEP"
    fi
  else
    if [[ -z "${dataset_file}" ]]; then
      _require_nonempty DATASET_ROOT "${dataset_root}"
      dataset_file="${dataset_root}/${dataset_relative_path}"
    fi
    _require_nonempty OUTPUT_ROOT "${output_root}"
    if [[ "${resume}" != "1" && -z "${init_weights}" && -n "${PREDECESSOR_RUN_ID}" ]]; then
      init_weights="$(_resolve_predecessor_weights \
        "${python_bin}" "${output_root}" "${PREDECESSOR_RUN_ID}")" || return
    fi
  fi

  local -a train_args=(
    --config_file "${MODEL_CONFIG}"
    --dataset-config.h5-folder-path "${dataset_file}"
  )
  train_args+=(${DATASET_ARGS[@]+"${DATASET_ARGS[@]}"})
  train_args+=(
    --trainer-config.log-dir "${output_root}"
    --trainer-config.exp-id "${run_id}"
  )
  train_args+=(${TRAINER_ARGS[@]+"${TRAINER_ARGS[@]}"})
  if [[ "${resume}" == "1" ]]; then
    train_args+=(--trainer-config.auto-resume)
  elif [[ -n "${init_weights}" ]]; then
    train_args+=(
      --trainer-config.load-weights "${init_weights}"
      --trainer-config.load-weights-strict
    )
  fi
  train_args+=(${extra_args[@]+"${extra_args[@]}"})

  local nproc_per_node="${NPROC_PER_NODE:-1}"
  local nnodes="${NNODES:-1}"
  local -a torchrun_args=(--nproc-per-node "${nproc_per_node}")
  if [[ "${nnodes}" == "1" ]]; then
    torchrun_args=(--standalone "${torchrun_args[@]}")
  else
    local node_rank="${NODE_RANK:-}"
    local master_addr="${MASTER_ADDR:-}"
    local master_port="${MASTER_PORT:-29500}"
    _require_nonempty NODE_RANK "${node_rank}"
    _require_nonempty MASTER_ADDR "${master_addr}"
    torchrun_args=(
      --nnodes "${nnodes}"
      --node-rank "${node_rank}"
      --master-addr "${master_addr}"
      --master-port "${master_port}"
      "${torchrun_args[@]}"
    )
  fi
  local -a command=(
    "${python_bin}" -m torch.distributed.run
    "${torchrun_args[@]}"
    --module renderformer.training.cli
    "${TRAINING_VERSION}"
    "${train_args[@]}"
  )

  if [[ "${action}" == "print" ]]; then
    _print_command "${command[@]}"
    return 0
  fi
  if [[ "${action}" == "print0" ]]; then
    printf '%s\0' "${command[@]}"
    return 0
  fi

  _validate_recipe_paths \
    "${dataset_file}" "${init_weights}" "${output_root}" "${run_id}" "${resume}"
  if [[ "${action}" == "validate" ]]; then
    printf 'stage %s is ready\n' "${RECIPE_KEY}"
    return 0
  fi

  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"
  cd -- "${REPO_ROOT}"
  printf 'Launching %s (%s stage %s) as run %s\n' \
    "${RECIPE_KEY}" "${FAMILY}" "${STAGE_ORDER}" "${run_id}"
  exec "${command[@]}"
}
