#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

SCENE_ID="${1:-}"
CHECKPOINT="${CHECKPOINT:-}"
CHECKPOINT_SUBFOLDER="${CHECKPOINT_SUBFOLDER:-}"
INPUT_H5="${INPUT_H5:-}"
DATA_OUTPUT_DIR="${DATA_OUTPUT_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
PRECISION="${PRECISION:-fp16}"
RESOLUTION="${RESOLUTION:-}"
EXECUTION_PROFILE="${EXECUTION_PROFILE:-release}"
TONE_MAPPER="${TONE_MAPPER:-}"

if [[ -z "${SCENE_ID}" ]]; then
  echo "usage: bash examples/rf2/render-paper-scene.sh SCENE_ID" >&2
  echo "set INPUT_H5, or generate the scene under DATA_OUTPUT_DIR first" >&2
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
    exit 2
    ;;
esac

if [[ -z "${RESOLUTION}" ]]; then
  RESOLUTION="${DEFAULT_RESOLUTION}"
fi

if [[ -z "${TONE_MAPPER}" ]]; then
  case "${SCENE_ID}" in
    bedroom|dragon|spaceship-in-smoke|living-room)
      TONE_MAPPER=agx
      ;;
    three-teapots)
      TONE_MAPPER=pbr_neutral
      ;;
    *)
      TONE_MAPPER=none
      ;;
  esac
fi

if [[ -z "${CHECKPOINT}" ]]; then
  CHECKPOINT="RenderFormer/renderformer-v2"
fi

if [[ -z "${DATA_OUTPUT_DIR}" ]]; then
  DATA_OUTPUT_DIR="/tmp/renderformer-rf2-${SCENE_ID}"
fi
if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${DATA_OUTPUT_DIR}/inference"
fi

if [[ -z "${INPUT_H5}" ]]; then
  INPUT_H5="${DATA_OUTPUT_DIR}/processed/scene.h5"
fi
if [[ ! -f "${INPUT_H5}" ]]; then
  cat >&2 <<EOF
error: processed RF2 input not found: ${INPUT_H5}

Generate it first in the renderformer-datagen environment:
  bash examples/rf2/generate-paper-scene.sh "${SCENE_ID}" "${DATA_OUTPUT_DIR}"

Then activate renderformer-cuda and rerun this inference launcher. Set
INPUT_H5 explicitly when the processed H5 is stored elsewhere.
EOF
  exit 2
fi

cd "${REPO_ROOT}"
"${PYTHON_BIN}" -m renderformer data validate-h5 \
  --input "${INPUT_H5}" \
  --format v2

INFER_ARGS=(
  --input "${INPUT_H5}"
  --checkpoint "${CHECKPOINT}"
  --variant v2
  --execution-profile "${EXECUTION_PROFILE}"
  --device "${DEVICE}"
  --precision "${PRECISION}"
  --resolution "${RESOLUTION}"
  --tone-mapper "${TONE_MAPPER}"
  --first-view-only
  --output-dir "${OUTPUT_DIR}"
)
if [[ -n "${CHECKPOINT_SUBFOLDER}" ]]; then
  INFER_ARGS+=(--checkpoint-subfolder "${CHECKPOINT_SUBFOLDER}")
fi
"${PYTHON_BIN}" -m renderformer infer "${INFER_ARGS[@]}"
