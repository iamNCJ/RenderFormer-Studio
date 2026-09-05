# RenderFormer Blender extensions

This directory contains two separate Blender extensions. They have different package,
operator, panel, and property identifiers, so both can be installed and enabled in the
same Blender profile.

| Source package | Manifest ID | Blender sidebar | Output |
| --- | --- | --- | --- |
| `renderformer_v1_tools/` | `renderformer_v1_tools` | RenderFormer V1 | Direct RF1 HDF5 |
| `renderformer_v2_tools/` | `renderformer_v2_tools` | RenderFormer V2 | Portable prepared-frame folder |

The old combined `renderformer_tools` extension has been removed. Because Blender
tracks extensions by manifest ID, installing the two new IDs does not upgrade or remove
an existing combined extension. Before upgrading, open **Preferences > Get
Extensions > Installed**, disable and remove **Renderformer Tools**
(`renderformer_tools`), then install the two new archives. Old `rf_*` scene/object
settings are not migrated to the new `rf1_*` and `rf2_*` properties. Do not install
the parent `tools/blender_addon/` directory as one extension.

## Supported Blender versions

The supported build and validation runtime is the PyPI `bpy==4.5.10` wheel under
Python 3.11. Other Blender point releases and Blender 5.x are not supported.
Dependency wheels must use the matching CPython 3.11 interpreter and platform tags.
Use exactly Blender/bpy 4.5.10 for all supported export and data-generation workflows.

## Published source and optional local archives

RenderFormer Studio publishes the two checked-in package directories as source; the
project release does not attach prebuilt add-on ZIPs. Their manifests intentionally do
not name generated wheel files. Both source packages can be installed, imported, and
enabled without third-party modules; an export that needs a missing module stops with
an actionable dependency error.

Source-only export dependencies are:

- V1: `numpy`, `h5py`, `networkx`, and `trimesh`; **Include Cycles Reference** also
  needs `simple-exr` and `OpenEXR`.
- V2: `numpy`, `imageio`, and `Pillow`.

Exact versions live in `release_dependencies.toml`. To use a source-only archive,
install those pins alongside `bpy==4.5.10` in the Python 3.11 environment that runs
the extension.

The release builder downloads compatible CPython 3.11 wheels, copies them into a
temporary package, injects `platforms` and `wheels` into the temporary manifest, and
records filenames and SHA-256 hashes in `WHEEL_INVENTORY.json`. It never writes wheels
into either checked-in source package. A release ZIP is self-contained for its one
declared platform; it must not be renamed or advertised for another platform.

## Build and install

Build clearly labeled dependency-free source archives in a Python 3.11 environment
with the pinned PyPI `bpy==4.5.10` wheel:

```bash
python3.11 -m pip install "bpy==4.5.10"
PYTHON_BIN=python3.11 \
OUTPUT_DIR=/tmp/renderformer-blender-source \
bash tools/blender_addon/build.sh source-only
```

This produces `renderformer_v1_tools-source-only.zip` and
`renderformer_v2_tools-source-only.zip`. They are local installation conveniences,
not prebuilt project-release artifacts or dependency bundles.

Build wheel-bearing release archives on the target platform with the same Python and
PyPI bpy pin:

```bash
python3.11 -m pip install "bpy==4.5.10"
PYTHON_BIN=python3.11 \
OUTPUT_DIR=/tmp/renderformer-blender-release \
bash tools/blender_addon/build.sh release
```

By default the release build obtains the exact pins from the configured Python package
index. For an offline/reviewed build, pre-populate a wheel directory and add
`WHEELHOUSE=/path/to/wheelhouse`; the builder then passes `--no-index` and fails if a
locked compatible wheel is unavailable. `SOURCE_DATE_EPOCH` may override the fixed
reproducible outer-archive timestamp.

The output names include the platform, for example
`renderformer_v1_tools-1.1.0-macos-arm64.zip`. In Blender, open **Preferences > Get
Extensions**, choose **Install from Disk**, and select both release ZIPs. Each archive
has a distinct ID, so both can be enabled together. The add-on source is MIT-licensed;
bundled wheels retain their upstream licenses, which are summarized in each package's
`THIRD_PARTY_NOTICES.md` and preserved inside the wheels.

## V1: direct RF1 HDF5

The V1 exporter writes the five required RF1 datasets:

- `triangles`: float32 `(N, 3, 3)`;
- `vn`: float32 `(N, 3, 3)`;
- `texture`: float32 `(N, 13, 32, 32)` with the RF1 triangular mask;
- `c2w`: float32 `(V, 4, 4)` (one view per exported frame);
- `fov`: float32 `(V,)`.

The 13 texture channels are diffuse RGB, specular RGB, roughness, tangent-space
normal RGB, and irradiance RGB. An optional Cycles reference is stored as `img`.

RF1 cannot encode UV/image textures, metallic or transmission materials,
displacement, environment textures, Blender light/volume objects, or multiple material
slots. The exporter reports these cases instead of silently changing the material.
Use an emissive mesh for a light. Vertex colors are supported only on already
triangulated geometry and cannot be combined with topology/normal recomputation.

Validate an export with the main repository:

```bash
renderformer data validate-h5 --input /path/to/scene.h5 --format v1
```

## V2: prepared-frame export

The V2 extension intentionally does not write a model-ready V2 HDF5 file. For each
frame it writes this portable intermediate representation:

```text
frame_000001/
  scene_config.json
  env_map.exr                 # only when an environment texture is connected
  split/
    object_name.obj
    object_name/              # for spatially varying materials
      basecolor.png           # or diffuse.png + specular.png
      metallic.png            # metallic workflow only
      roughness.png
      normal.png
      height.png              # only when displacement is enabled
```

The exporter materializes constant fallback maps when only part of a shader is
textured, so the folder has every file required by `validate_prepared_frame()`. It
supports Principled and `SpecularRoughnessBSDF` materials, homogeneous transmission,
UV texture maps, image-based displacement, an environment texture, emissive meshes,
and one camera. It does not export Blender LIGHT/VOLUME objects, measured BRDFs,
spatially varying emission, multiple materials on one mesh, or per-triangle vertex
colors.

To prevent stale meshes or maps from an older export from masquerading as current
output, the exporter refuses to write into an existing non-empty `frame_NNNNNN`
directory. Choose another output root or explicitly remove the obsolete frame after
reviewing it.

Complete the pipeline outside Blender, where RenderFormer can download the
official BRDF mappers, Qwen VAE, and V2 checkpoint. Local mapper
directories remain optional overrides:

```bash
renderformer data generate \
  --prepared-frame /path/to/frame_000001 \
  --profile src/renderformer/data/profiles/v2_training_raw.yaml \
  --output-dir /tmp/renderformer-v2/raw \
  --resolution 256 \
  --spp 4096 \
  --texture-crop-res 256 \
  --texture-target-res 256 \
  --texture-size 32

renderformer data convert \
  --input /tmp/renderformer-v2/raw/v2_training_raw.h5 \
  --output /tmp/renderformer-v2/processed/scene.h5 \
  --texture-encoder qwen_vae \
  --model-id Qwen/Qwen-Image \
  --device cuda \
  --torch-dtype float32 \
  --triangle-batch-size 256

renderformer data validate-h5 \
  --input /tmp/renderformer-v2/processed/scene.h5 \
  --format v2

renderformer infer \
  --input /tmp/renderformer-v2/processed/scene.h5 \
  --checkpoint RenderFormer/renderformer-v2 \
  --checkpoint-subfolder transformer_512 \
  --variant v2 \
  --device cuda \
  --precision fp16 \
  --output-dir /tmp/renderformer-v2/rendered
```

The generation step produces the raw V2 H5. Conversion uses the Qwen model's
repository-default revision and performs the learned texture encoding plus
environment/volume postprocessing. Pass `--revision` to opt into a particular
branch, tag, or commit.
Validation checks the generic V2 schema; inference then applies the stricter
selected-checkpoint contract through the public-default `release` profile. The
transformer is loaded from the componentized V2 bundle. Its repository revision
selects both native transformers, the material autoencoder, and all three
mapper MLPs atomically; authenticate with Hugging Face whenever the bundle is
access-protected during release staging.
Learned BRDF mapper and Qwen weights are deliberately not bundled in the
Blender extension; the data pipeline downloads them from the Hub as needed.

## Source validation

The published source packages were validated together in one clean Python 3.11
process using the PyPI `bpy==4.5.10` wheel, without Blender.app or a `blender`
executable. Both extensions registered simultaneously with disjoint operators,
panels, and properties. A real V1 export passed the repository's strict RF1 H5
validator, and a real V2 image-textured Principled material export passed
`validate_prepared_frame()`.

Locally built wheel-bearing archives remain platform-specific. Anyone distributing
one must repeat the two real export checks on that target platform; the source release
does not imply that unbuilt binary dependency bundles were tested on every platform.
