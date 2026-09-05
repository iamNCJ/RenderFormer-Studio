#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

SCENE_ID="${1:-}"
OUTPUT_DIR="${2:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESOLUTION="${RESOLUTION:-}"
SPP="${SPP:-4096}"
POSTPROCESS="${POSTPROCESS:-1}"
POSTPROCESS_DEVICE="${POSTPROCESS_DEVICE:-cuda}"
QWEN_MODEL_ID="${QWEN_MODEL_ID:-Qwen/Qwen-Image}"
QWEN_REVISION="${QWEN_REVISION:-}"

usage() {
  cat >&2 <<'EOF'
usage: bash examples/rf2/generate-paper-scene.sh SCENE_ID [OUTPUT_DIR]

SCENE_ID is one of:
  transparent-torus, environment-lit-spheres, smoky-bunny,
  three-teapots, displacement-mapped-cornell-cube,
  spaceship-in-smoke, dinner-scene, cube-pile, dragon, bedroom,
  living-room, living-room-daylight

External inputs that cannot be redistributed are listed in README.md. The
launcher prints the required environment variable when one is missing.
EOF
}

if [[ -z "${SCENE_ID}" ]]; then
  usage
  exit 2
fi

case "${SCENE_ID}" in
  transparent-torus|environment-lit-spheres|smoky-bunny|three-teapots|\
  displacement-mapped-cornell-cube|cube-pile)
    DEFAULT_RESOLUTION=512
    ;;
  spaceship-in-smoke|dinner-scene|dragon|bedroom|living-room|\
  living-room-daylight)
    DEFAULT_RESOLUTION=2048
    ;;
  *)
    echo "error: unknown RF2 paper scene: ${SCENE_ID}" >&2
    usage
    exit 2
    ;;
esac

if [[ -z "${RESOLUTION}" ]]; then
  RESOLUTION="${DEFAULT_RESOLUTION}"
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="/tmp/renderformer-rf2-${SCENE_ID}"
fi

FRAME_DIR="${SCRIPT_DIR}/scenes/${SCENE_ID}/frame"
SOURCE_FRAME="${FRAME_DIR}"
TEMP_ROOT=""

cleanup() {
  if [[ -n "${TEMP_ROOT}" && -d "${TEMP_ROOT}" ]]; then
    rm -rf -- "${TEMP_ROOT}"
  fi
}
trap cleanup EXIT

require_file() {
  local variable_name="$1"
  local value="$2"
  if [[ -z "${value}" || ! -f "${value}" ]]; then
    echo "error: ${variable_name} must point to the required local file" >&2
    exit 2
  fi
}

require_directory() {
  local variable_name="$1"
  local value="$2"
  if [[ -z "${value}" || ! -d "${value}" ]]; then
    echo "error: ${variable_name} must point to the required local directory" >&2
    exit 2
  fi
}

stage_bundled_frame() {
  TEMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/renderformer-rf2-paper.XXXXXX")"
  SOURCE_FRAME="${TEMP_ROOT}/frame"
  mkdir -p "${SOURCE_FRAME}"
  cp -R "${FRAME_DIR}/." "${SOURCE_FRAME}/"
}

case "${SCENE_ID}" in
  environment-lit-spheres)
    require_file RF2_ENV_SPHERES_MAP "${RF2_ENV_SPHERES_MAP:-}"
    stage_bundled_frame
    mkdir -p "${SOURCE_FRAME}/external"
    cp "${RF2_ENV_SPHERES_MAP}" "${SOURCE_FRAME}/external/grace_probe_latlong.exr"
    ;;
  smoky-bunny)
    require_file RF2_SMOKY_BUNNY_VDB "${RF2_SMOKY_BUNNY_VDB:-}"
    stage_bundled_frame
    mkdir -p "${SOURCE_FRAME}/external"
    cp "${RF2_SMOKY_BUNNY_VDB}" "${SOURCE_FRAME}/external/fluid_data_0001_preprocessed.vdb"
    ;;
  spaceship-in-smoke)
    require_directory RF2_SPACESHIP_SPLIT "${RF2_SPACESHIP_SPLIT:-}"
    require_file RF2_SPACESHIP_VDB "${RF2_SPACESHIP_VDB:-}"
    stage_bundled_frame
    cp -R "${RF2_SPACESHIP_SPLIT}" "${SOURCE_FRAME}/split"
    mkdir -p "${SOURCE_FRAME}/external"
    cp "${RF2_SPACESHIP_VDB}" "${SOURCE_FRAME}/external/fluid_data_0175_preprocessed.vdb"
    ;;
  dinner-scene)
    require_directory RF2_DINNER_FRAME "${RF2_DINNER_FRAME:-}"
    SOURCE_FRAME="${RF2_DINNER_FRAME}"
    echo "warning: Dinner frame is the 29,992-triangle candidate; the paper reports 29,276 tokens." >&2
    ;;
  bedroom)
    require_file RF2_BEDROOM_WALL_BASECOLOR "${RF2_BEDROOM_WALL_BASECOLOR:-}"
    stage_bundled_frame
    cp "${RF2_BEDROOM_WALL_BASECOLOR}" "${SOURCE_FRAME}/split/wall0/basecolor.png"
    cp "${RF2_BEDROOM_WALL_BASECOLOR}" "${SOURCE_FRAME}/split/wall1/basecolor.png"
    ;;
  living-room-daylight)
    require_file RF2_DAYLIGHT_ENV_MAP "${RF2_DAYLIGHT_ENV_MAP:-}"
    stage_bundled_frame
    mkdir -p "${SOURCE_FRAME}/external"
    cp "${RF2_DAYLIGHT_ENV_MAP}" "${SOURCE_FRAME}/external/20060807_wells6_hd.exr"
    ;;
esac

RAW_DIR="${OUTPUT_DIR}/raw"
PROCESSED_DIR="${OUTPUT_DIR}/processed"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" -m renderformer data generate \
  --prepared-frame "${SOURCE_FRAME}" \
  --profile src/renderformer/data/profiles/v2_training_raw.yaml \
  --output-dir "${RAW_DIR}" \
  --resolution "${RESOLUTION}" \
  --spp "${SPP}" \
  --texture-crop-res 256 \
  --texture-target-res 256 \
  --texture-size 32

RAW_H5="${RAW_DIR}/v2_training_raw.h5"
"${PYTHON_BIN}" -m renderformer data validate-h5 \
  --input "${RAW_H5}" \
  --format v2

if [[ "${POSTPROCESS}" == "0" ]]; then
  echo "raw RF2 paper scene: ${RAW_H5}"
  exit 0
fi

mkdir -p "${PROCESSED_DIR}"
PROCESSED_H5="${PROCESSED_DIR}/scene.h5"
CONVERT_ARGS=(
  --input "${RAW_H5}"
  --output "${PROCESSED_H5}"
  --texture-encoder qwen_vae
  --model-id "${QWEN_MODEL_ID}"
)
if [[ -n "${QWEN_REVISION}" ]]; then
  CONVERT_ARGS+=(--revision "${QWEN_REVISION}")
fi
CONVERT_ARGS+=(
  --device "${POSTPROCESS_DEVICE}"
  --torch-dtype float32
  --triangle-batch-size 256
)
"${PYTHON_BIN}" -m renderformer data convert "${CONVERT_ARGS[@]}"

"${PYTHON_BIN}" -m renderformer data validate-h5 \
  --input "${PROCESSED_H5}" \
  --format v2

echo "postprocessed RF2 paper scene: ${PROCESSED_H5}"
