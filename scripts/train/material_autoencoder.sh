#!/usr/bin/env bash
# Portable material-autoencoder recipe with an explicit HDR network-space contract.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CALLER_CWD="$(pwd -P)"

usage() {
  cat <<'EOF'
Usage: scripts/train/material_autoencoder.sh [--print | --validate] [--] [EXTRA_ARGS...]

Required environment variables:
  MANIFEST             JSONL manifest with explicit EXR rows (train required;
                       validation optional)
  OUTPUT_DIR           Run directory for checkpoints and resolved contract

Optional recipe variables:
  PYTHON_BIN           Python executable (default: python)
  PREPROCESSING_MODE   Must be log10_1p (default: log10_1p)
  BATCH_SIZE           Per-rank batch size (default: 64)
  EPOCHS               Total epoch count (default: 50)
  LEARNING_RATE        AdamW learning rate (default: 1e-4)
  NUM_WORKERS          Loader workers per rank (default: 8)
  SMOOTHNESS_WEIGHT    Historical latent smoothness weight (default: 2)
  SAVE_EVERY_EPOCHS    Periodic checkpoint interval (default: 5)
  SEED                 Random seed (default: 42)
  RESUME               Full checkpoint from this runner
  LOAD_MODEL           Model-only checkpoint/path/Hub ID for a fresh run
  LOAD_MODEL_REVISION  Optional Hub revision for LOAD_MODEL
  LOAD_MODEL_SUBFOLDER Optional component subfolder inside LOAD_MODEL
  ALLOW_MISSING_PREPROCESSING_METADATA=1
                       Acknowledge standalone legacy .pt files without metadata
  NPROC_PER_NODE       torchrun workers (default: 1)
  NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT
                       Optional multi-node torchrun settings
  WANDB=1              Enable W&B (disabled by default)
  WANDB_PROJECT        Project name
  WANDB_NAME           Optional run name

The public recipe always passes --preprocessing-mode log10_1p explicitly.
--validate reads and transforms every EXR without constructing the model.
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

action=run
extra_args=()
while (($#)); do
  case "$1" in
    --print|--validate)
      if [[ "${action}" != run ]]; then
        echo "error: choose only one of --print or --validate" >&2
        exit 2
      fi
      action="${1#--}"
      shift
      ;;
    --help|-h)
      usage
      exit 0
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

python_bin="${PYTHON_BIN:-python}"
manifest="${MANIFEST:-}"
output_dir="${OUTPUT_DIR:-}"
preprocessing_mode="${PREPROCESSING_MODE:-log10_1p}"
if [[ "${preprocessing_mode}" != log10_1p ]]; then
  echo "error: PREPROCESSING_MODE must be log10_1p" >&2
  exit 2
fi
if [[ -n "${manifest}" ]]; then
  manifest="$(absolute_from_caller "${manifest}")"
fi
if [[ -n "${output_dir}" ]]; then
  output_dir="$(absolute_from_caller "${output_dir}")"
fi

if [[ "${action}" == print ]]; then
  manifest="${manifest:-/path/to/material-exr-manifest.jsonl}"
  output_dir="${output_dir:-/path/to/material-autoencoder-run}"
else
  if [[ -z "${manifest}" || ! -f "${manifest}" ]]; then
    echo "error: MANIFEST must name an existing JSONL file: ${manifest:-<unset>}" >&2
    exit 2
  fi
  if [[ -z "${output_dir}" ]]; then
    echo "error: OUTPUT_DIR must be set" >&2
    exit 2
  fi
fi

resume="${RESUME:-}"
load_model="${LOAD_MODEL:-}"
if [[ -n "${resume}" && -n "${load_model}" ]]; then
  echo "error: RESUME and LOAD_MODEL are mutually exclusive" >&2
  exit 2
fi
if [[ -n "${resume}" && "${resume}" != /* ]]; then
  resume="$(absolute_from_caller "${resume}")"
fi
if [[ -n "${load_model}" ]]; then
  if [[ "${load_model}" == /* ]]; then
    if [[ "${action}" != print && ! -e "${load_model}" ]]; then
      echo "error: absolute LOAD_MODEL path does not exist: ${load_model}" >&2
      exit 2
    fi
  elif [[ -e "${CALLER_CWD}/${load_model}" ]]; then
    load_model="$(absolute_from_caller "${load_model}")"
  fi
fi

train_args=(
  --manifest "${manifest}"
  --model-config "${REPO_ROOT}/configs/model/material/autoencoder.yaml"
  --output-dir "${output_dir}"
  --preprocessing-mode "${preprocessing_mode}"
  --batch-size "${BATCH_SIZE:-64}"
  --epochs "${EPOCHS:-50}"
  --learning-rate "${LEARNING_RATE:-1e-4}"
  --weight-decay "${WEIGHT_DECAY:-0.01}"
  --num-workers "${NUM_WORKERS:-8}"
  --smoothness-weight "${SMOOTHNESS_WEIGHT:-2.0}"
  --smoothness-noise-std "${SMOOTHNESS_NOISE_STD:-0.00784313725490196}"
  --seed "${SEED:-42}"
  --log-interval "${LOG_INTERVAL:-100}"
  --save-every-epochs "${SAVE_EVERY_EPOCHS:-5}"
  --device "${DEVICE:-cuda}"
)
if [[ "${action}" == validate ]]; then
  train_args+=(--validate-only)
fi
if [[ -n "${resume}" ]]; then
  train_args+=(--resume "${resume}")
fi
if [[ -n "${load_model}" ]]; then
  train_args+=(--load-model "${load_model}")
  if [[ -n "${LOAD_MODEL_REVISION:-}" ]]; then
    train_args+=(--load-model-revision "${LOAD_MODEL_REVISION}")
  fi
  if [[ -n "${LOAD_MODEL_SUBFOLDER:-}" ]]; then
    train_args+=(--load-model-subfolder "${LOAD_MODEL_SUBFOLDER}")
  fi
  if [[ "${ALLOW_MISSING_PREPROCESSING_METADATA:-0}" == 1 ]]; then
    train_args+=(--allow-missing-preprocessing-metadata)
  fi
fi
if [[ "${WANDB:-0}" == 1 ]]; then
  train_args+=(--wandb --wandb-project "${WANDB_PROJECT:-renderformer-material-autoencoder}")
  if [[ -n "${WANDB_NAME:-}" ]]; then
    train_args+=(--wandb-name "${WANDB_NAME}")
  fi
fi
if ((${#extra_args[@]})); then
  train_args+=("${extra_args[@]}")
fi

nproc_per_node="${NPROC_PER_NODE:-1}"
nnodes="${NNODES:-1}"
if [[ "${action}" == validate ]]; then
  nproc_per_node=1
  nnodes=1
fi
if ((nproc_per_node > 1 || nnodes > 1)); then
  torchrun_args=(--nproc-per-node "${nproc_per_node}" --nnodes "${nnodes}")
  if [[ "${nnodes}" == 1 ]]; then
    torchrun_args=(--standalone "${torchrun_args[@]}")
  else
    : "${NODE_RANK:?NODE_RANK is required when NNODES > 1}"
    : "${MASTER_ADDR:?MASTER_ADDR is required when NNODES > 1}"
    torchrun_args+=(
      --node-rank "${NODE_RANK}"
      --master-addr "${MASTER_ADDR}"
      --master-port "${MASTER_PORT:-29500}"
    )
  fi
  command=(
    "${python_bin}" -m torch.distributed.run "${torchrun_args[@]}"
    --module renderformer.training.cli material --ddp "${train_args[@]}"
  )
else
  command=("${python_bin}" -m renderformer.training.cli material "${train_args[@]}")
fi

if [[ "${action}" == print ]]; then
  printf 'cd %q\n' "${REPO_ROOT}"
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

cd -- "${REPO_ROOT}"
exec "${command[@]}"
