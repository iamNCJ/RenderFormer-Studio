#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-source-only}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

case "${MODE}" in
  source-only)
    OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/dist/source-only}"
    build_args=(--mode source-only --output-dir "${OUTPUT_DIR}")
    ;;
  release)
    OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/dist/release}"
    build_args=(--mode release --output-dir "${OUTPUT_DIR}")
    if [[ -n "${WHEELHOUSE:-}" ]]; then
      build_args+=(--wheelhouse "${WHEELHOUSE}")
    fi
    ;;
  *)
    echo "Usage: $0 [source-only|release]" >&2
    exit 2
    ;;
esac

"${PYTHON_BIN}" "${SCRIPT_DIR}/build_release.py" "${build_args[@]}"
