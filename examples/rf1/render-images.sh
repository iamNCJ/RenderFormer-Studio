#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_ID="${MODEL_ID:-microsoft/renderformer-v1.1-swin-large}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/output}"

render_scene() {
  local scene_name="$1"
  local tone_mapper="${2:-none}"
  local scene_output="$OUTPUT_ROOT/$scene_name"
  mkdir -p "$scene_output"
  renderformer data compose-rf1 \
    "$scene_name.json" \
    --output "$scene_output/$scene_name.h5"
  renderformer infer \
    --input "$scene_output/$scene_name.h5" \
    --checkpoint "$MODEL_ID" \
    --output-dir "$scene_output/rendered" \
    --tone-mapper "$tone_mapper"
}

render_scene cbox
render_scene tree
render_scene constant-width
render_scene compose-scene
render_scene renderformer-logo
render_scene cbox-lucy
render_scene cbox-bunny
render_scene cbox-teapot
render_scene shader-ball agx
render_scene veach-mis pbr_neutral
render_scene horse-and-heart agx
render_scene fox-in-the-wild agx
render_scene crystals agx
