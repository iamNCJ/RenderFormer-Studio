#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
RENDERFORMER_VIDEO_DATA_PATH="${RENDERFORMER_VIDEO_DATA_PATH:-video-data}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEQUENCE_METADATA="$SCRIPT_DIR/video-sequences.json"

DATASET_ID=$("$PYTHON_BIN" - "$SEQUENCE_METADATA" <<'PY'
import json
import sys

metadata = json.load(open(sys.argv[1], encoding="utf-8"))
if metadata.get("schema_version") != 1:
    raise SystemExit("video-sequences.json has an unsupported schema_version")
dataset = metadata.get("dataset")
if not isinstance(dataset, str) or not dataset:
    raise SystemExit("video-sequences.json must define a nonempty dataset")
print(dataset)
PY
)

hf download \
  --repo-type dataset \
  "$DATASET_ID" \
  --local-dir "$RENDERFORMER_VIDEO_DATA_PATH"

# Check if video-data directory exists
if [ ! -d "$RENDERFORMER_VIDEO_DATA_PATH" ]; then
    echo "Error: $RENDERFORMER_VIDEO_DATA_PATH directory not found!"
    exit 1
fi

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
        sequence["archive"],
        sequence["input"],
        sequence["expected_frames"],
        sequence["tone_mapper"],
        sep="\t",
    )
PY
}

process_sequence() {
  local sequence_id="$1"
  local archive="$2"
  local input="$3"
  local expected_frames="$4"
  local archive_path="$RENDERFORMER_VIDEO_DATA_PATH/$archive"
  local input_path="$RENDERFORMER_VIDEO_DATA_PATH/$input"

  if [ ! -f "$archive_path" ]; then
    echo "Missing archive for $sequence_id: $archive_path" >&2
    return 1
  fi

  "$PYTHON_BIN" - "$archive_path" "$expected_frames" <<'PY'
import stat
import sys
from pathlib import PurePosixPath
from zipfile import ZipFile

archive_path, expected_raw = sys.argv[1:]
with ZipFile(archive_path) as archive:
    entries = archive.infolist()
h5_members = []
for entry in entries:
    member = entry.filename
    path = PurePosixPath(member)
    if path.is_absolute() or ".." in path.parts or len(path.parts) != 1:
        raise SystemExit(f"unsafe or nested archive member in {archive_path}: {member!r}")
    mode = (entry.external_attr >> 16) & 0o170000
    if stat.S_ISLNK(mode):
        raise SystemExit(f"symbolic-link archive member in {archive_path}: {member!r}")
    if entry.is_dir():
        raise SystemExit(f"unexpected directory archive member in {archive_path}: {member!r}")
    if path.suffix.lower() != ".h5":
        raise SystemExit(f"unexpected non-H5 archive member in {archive_path}: {member!r}")
    h5_members.append(member)
if len(h5_members) != int(expected_raw):
    raise SystemExit(
        f"{archive_path} contains {len(h5_members)} H5 frames; expected {expected_raw}"
    )
if len(set(h5_members)) != len(h5_members):
    raise SystemExit(f"duplicate H5 members in {archive_path}")
PY

  mkdir -p "$input_path"
  unzip -oq "$archive_path" -d "$input_path"

  "$PYTHON_BIN" - "$archive_path" "$input_path" "$expected_frames" <<'PY'
import os
import sys
from pathlib import Path
from zipfile import ZipFile

archive_path = Path(sys.argv[1])
input_path = Path(sys.argv[2])
expected_frames = int(sys.argv[3])
with ZipFile(archive_path) as archive:
    frames = sorted(
        item.filename
        for item in archive.infolist()
        if not item.is_dir() and Path(item.filename).suffix.lower() == ".h5"
    )
if len(frames) != expected_frames:
    raise SystemExit(
        f"{archive_path} contains {len(frames)} H5 frames; expected {expected_frames}"
    )
missing = [frame for frame in frames if not (input_path / frame).is_file()]
if missing:
    raise SystemExit(f"extraction is missing H5 frames from {archive_path}: {missing[:5]}")
manifest = input_path / "frames.manifest"
temporary = input_path / ".frames.manifest.tmp"
temporary.write_text("\n".join(frames) + "\n", encoding="utf-8")
os.replace(temporary, manifest)
print(f"wrote {manifest} with {len(frames)} ordered frames", flush=True)
PY
}

emit_sequences | while IFS=$'\t' read -r sequence_id archive input expected_frames _tone_mapper; do
  echo "Extracting $sequence_id from $archive"
  process_sequence "$sequence_id" "$archive" "$input" "$expected_frames"
done

echo "All zip files have been processed!"
