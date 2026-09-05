# RF1 scene examples

This directory contains legacy RenderFormer V1 scene JSON files and the small
OBJ files used for compatibility testing. The maintained workflow is:

```text
legacy scene JSON + relative OBJ files
  -> renderformer data compose-rf1
  -> strict RF1 H5
  -> RenderFormerPipeline / renderformer infer
  -> linear HDR EXR + display PNG
```

The old standalone scene processor has been removed. Composition now uses the
shared `renderformer.data` exporters, and inference uses the same transformer
implementation as training.

## Asset provenance and licenses

The legacy example meshes have passed the release provenance review. Every
bundled OBJ file is byte-for-byte identical to the corresponding file in the
pinned official RenderFormer repository. Their per-file SHA-256 hashes,
official-repository blob hashes, source mapping, authors, licenses, change
notices, and redistribution decisions are recorded in
[`ASSET_PROVENANCE.json`](ASSET_PROVENANCE.json). Human-readable credits are in
[`ASSET_ATTRIBUTIONS.md`](ASSET_ATTRIBUTIONS.md).

Do not treat every mesh as MIT or Creative Commons. The Sketchfab-derived
assets are CC BY 4.0; Spot, constant-width, and Veach MIS are CC0; Stanford's
Bunny and Lucy retain research/non-commercial terms; and the Cornell, Utah,
and Mitsuba sources retain the repository-specific terms described in the
attribution file.

## Install

From the repository root:

```bash
python -m pip install -e ".[inference]"
```

The composer is CPU-only and does not invoke Blender or Cycles. It uses
`pymeshlab` only when a JSON object explicitly enables `remesh`. Install the
geometry dependencies for such a scene:

```bash
python -m pip install -e ".[inference,geometry]"
```

## Render one scene

Commands may be run from the repository root:

```bash
renderformer data compose-rf1 \
  examples/rf1/cbox.json \
  --output /tmp/renderformer-rf1/cbox.h5

renderformer infer \
  --input /tmp/renderformer-rf1/cbox.h5 \
  --checkpoint microsoft/renderformer-v1.1-swin-large \
  --precision fp16 \
  --output-dir /tmp/renderformer-rf1/cbox-rendered
```

On a non-CUDA device, use `--precision fp32`. The public V1 model ID follows
the Hugging Face repository's default revision unless `--revision` is passed.

The result contains `cbox_view_<view>.exr`, a linear HDR image, and
`cbox_view_<view>.png`, a display image. Use `--tone-mapper agx`, `filmic`, or
`pbr_neutral` when a scene needs highlight compression in its PNG. Tone mapping
never changes the EXR.

## Render all static examples

`render-images.sh` composes and renders every maintained attributed static
gallery scene:

```bash
bash examples/rf1/render-images.sh
```

Override its checkpoint and output directory without editing the script:

```bash
MODEL_ID=microsoft/renderformer-v1-base \
OUTPUT_ROOT=/tmp/renderformer-rf1/examples \
bash examples/rf1/render-images.sh
```

Each scene is written under `OUTPUT_ROOT/<scene>/`; its H5 and rendered output
remain separate.

## Scene JSON contract

A scene contains top-level `scene_name`, `version`, `objects`, and `cameras`
fields. `objects` is a mapping, and `cameras` contains one or more camera
objects. Each scene object supplies:

- `mesh_path`, resolved relative to the scene JSON;
- `transform.translation`, `rotation` in degrees, `scale`, and `normalize`;
- `material.diffuse`, `material.specular`, `material.roughness`,
  `material.emissive`, and `material.smooth_shading`;
- optional deterministic random diffuse controls under `material`;
- optional `remesh` and `remesh_target_face_num`.

Each camera supplies `position`, `look_at`, `up`, and a field of view in
degrees. `compose-rf1` validates this schema strictly: misspelled or extra
fields fail instead of being ignored.

Do not use under-tessellated objects or receivers whose triangles cover large
regions of the rendered view. Remesh first, then use QSlim to produce
watertight, approximately uniform faces without simplifying away silhouette or
material-boundary detail. See
[RF1 scene-distribution constraints](../../docs/inference/README.md#rf1-scene-distribution-constraints)
before authoring a new scene.

The composer reproduces the legacy transform order, shading split, seeded
random diffuse behavior, camera conversion, RF1 texture mask, and historical
material-value quantization. It stores the final required datasets as
float32. Pass `--no-legacy-texture-quantization` only when intentionally
creating a non-legacy numeric variant.

The composed H5 has this exact inference contract:

| Key | dtype | Shape |
| --- | --- | --- |
| `triangles` | float32 | `(N, 3, 3)` |
| `texture` | float32 | `(N, 13, 32, 32)` |
| `vn` | float32 | `(N, 3, 3)` |
| `c2w` | float32 | `(V, 4, 4)` |
| `fov` | float32 | `(V,)` |

Validate a custom scene before inference:

```bash
renderformer data validate-h5 --input scene.h5 --format v1
renderformer infer \
  --input scene.h5 \
  --checkpoint microsoft/renderformer-v1-base \
  --variant v1 \
  --validate-only
```

Composition encodes cameras and scene attributes but does not render a Blender
reference image. Generate and archive ground truth separately when computing
metrics.

## Video examples

The optional public `renderformer/renderformer-video-data` H5 bundle can be
downloaded with the Hugging Face CLI and `unzip` through:

```bash
bash examples/rf1/download-video-data.sh
```

Set `RENDERFORMER_VIDEO_DATA_PATH` to choose a different destination. Then
render every maintained sequence:

```bash
RENDERFORMER_VIDEO_DATA_PATH=/datasets/renderformer-video-data \
OUTPUT_ROOT=/tmp/renderformer-rf1/videos \
bash examples/rf1/render-videos.sh
```

The published video bundle predates the canonical float32 RF1 producer
contract. Its textures are float16; depending on the sequence, its geometry
and normals are either both float32 or both float64. `render-videos.sh`
therefore passes the narrowly scoped `--allow-legacy-rf1-dtypes` compatibility
flag. The loader accepts only those two known historical signatures and casts
them to float32, matching the original V1 inference loader. Do not use the flag
to bypass validation for newly generated or third-party RF1 inputs.

[`video-sequences.json`](video-sequences.json) is the single ordered inventory
for all 15 sequences: it records each archive, extracted input directory,
expected frame count, and tone mapper. The download script extracts only those
listed archives and writes an ordered `frames.manifest` beside each sequence's
H5 files. The render script consumes those manifests rather than discovering
frames with a directory glob. The CLI loads and releases one frame at a time
with the script's default `--batch-size 1`, so sequence length does not make
all H5 scene tensors resident at once. Each sequence produces per-view EXR/PNG
files, `video.mp4`, and `inference_manifest.json`.

## More information

The full [inference guide](../../docs/inference/README.md) documents the Python
pipeline APIs, checkpoint locks, batching recommendations, V2 components,
metric protocols, and current reproduction gaps. For academic attribution of
this repository, cite both papers listed in the root
[README](../../README.md#citation). Asset attribution and academic paper
citation remain separate obligations.
