#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
RENDERFORMER_VIDEO_DATA_PATH="${RENDERFORMER_VIDEO_DATA_PATH:-./video-data}"
MODEL_ID="${MODEL_ID:-microsoft/renderformer-v1.1-swin-large}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/output/videos}"
SEQUENCE_METADATA="$SCRIPT_DIR/video-sequences.json"

render_video() {
  local input_manifest="$1"
  local output_dir="$2"
  local tone_mapper="${3:-none}"
  "${PYTHON_BIN}" -m renderformer infer \
    --input "${input_manifest}" \
    --checkpoint "${MODEL_ID}" \
    --allow-legacy-rf1-dtypes \
    --batch-size 1 \
    --output-dir "${output_dir}" \
    --tone-mapper "${tone_mapper}" \
    --save-video
}

emit_sequences() {
  "$PYTHON_BIN" - "$SEQUENCE_METADATA" <<'PY'
import json
import re
import sys
from pathlib import PurePosixPath

metadata = json.load(open(sys.argv[1], encoding="utf-8"))
if metadata.get("schema_version") != 1:
    raise SystemExit("video-sequences.json has an unsupported schema_version")
sequences = metadata.get("sequences")
if not isinstance(sequences, list) or not sequences:
    raise SystemExit("video-sequences.json must define a nonempty sequences list")
required = {"id", "archive", "input", "expected_frames", "tone_mapper"}
seen = {key: set() for key in ("id", "archive", "input")}
for sequence in sequences:
    if not isinstance(sequence, dict) or set(sequence) != required:
        raise SystemExit(f"invalid video sequence record: {sequence!r}")
    for key in ("id", "archive", "input", "tone_mapper"):
        value = sequence[key]
        if not isinstance(value, str) or not value or "\t" in value or "\n" in value:
            raise SystemExit(f"invalid {key} in video sequence record: {sequence!r}")
    if re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", sequence["id"]) is None:
        raise SystemExit(f"invalid sequence id: {sequence['id']!r}")
    for key in ("archive", "input"):
        path = PurePosixPath(sequence[key])
        if path.is_absolute() or ".." in path.parts:
            raise SystemExit(f"{key} must be a safe relative path: {sequence[key]!r}")
    expected_frames = sequence["expected_frames"]
    if isinstance(expected_frames, bool) or not isinstance(expected_frames, int) or expected_frames <= 0:
        raise SystemExit(f"invalid expected_frames in video sequence record: {sequence!r}")
    if sequence["tone_mapper"] not in {"none", "agx", "filmic", "pbr_neutral"}:
        raise SystemExit(f"invalid tone_mapper in video sequence record: {sequence!r}")
    for key in seen:
        if sequence[key] in seen[key]:
            raise SystemExit(f"duplicate {key} in video-sequences.json: {sequence[key]!r}")
        seen[key].add(sequence[key])
    print(
        sequence["id"],
        sequence["input"],
        sequence["expected_frames"],
        sequence["tone_mapper"],
        sep="\t",
    )
PY
}

emit_sequences | while IFS=$'\t' read -r sequence_id input expected_frames tone_mapper; do
  input_manifest="$RENDERFORMER_VIDEO_DATA_PATH/$input/frames.manifest"
  if [ ! -f "$input_manifest" ]; then
    echo "Missing ordered frame manifest for $sequence_id: $input_manifest" >&2
    echo "Run examples/rf1/download-video-data.sh first." >&2
    exit 1
  fi
  actual_frames=$("$PYTHON_BIN" - "$input_manifest" <<'PY'
import sys
from pathlib import Path

lines = [
    line.strip()
    for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.lstrip().startswith("#")
]
print(len(lines))
PY
  )
  if [ "$actual_frames" -ne "$expected_frames" ]; then
    echo "$sequence_id manifest has $actual_frames frames; expected $expected_frames" >&2
    exit 1
  fi
  render_video "$input_manifest" "$OUTPUT_ROOT/$sequence_id" "$tone_mapper"
done
