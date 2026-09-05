# `renderformer.data`

`renderformer.data` is the data subpackage of RenderFormer Studio. From the
same install as the model, pipelines, and training code, it produces both V1
(RF1) and V2 H5 training/evaluation samples. A YAML export profile selects the
output format without creating a second data package.

The training side of the repo remains a thin H5 loader: anything format-specific
(channel layouts, masking conventions, env-map rotation, etc.) belongs in this
package or in the trained model itself, never in the loader.

This README covers:

- [Design principles](#design-principles)
- [Package layout](#package-layout)
- [End-to-end data flow](#end-to-end-data-flow)
- [Schemas](#schemas)
- [Export profiles](#export-profiles)
- [Capabilities and validation](#capabilities-and-validation)
- [Texture encoding conventions](#texture-encoding-conventions)
- [Material latent models](#material-latent-models)
- [Postprocess](#postprocess)
- [Training loaders](#training-loaders)
- [Environment-map utilities](#environment-map-utilities)
- [Mesh and UV utilities](#mesh-and-uv-utilities)
- [Templates and manifests](#templates-and-manifests)
- [CLI reference](#cli-reference)
- [Programmatic API](#programmatic-api)
- [Adding a new export format](#adding-a-new-export-format)

---

## Design principles

1. **One datagen, many formats.** The scene builder produces a canonical,
   format-agnostic representation. Exporters consume that representation and
   each owns its channel layout, masking, and spatial encoding. The generation
   dispatcher selects an exporter; format-specific encoding stays inside
   `exporters/v1.py` or `exporters/v2.py`.
2. **Profile YAML is the source of truth.** What a format supports
   (`allow_env_map`, `allow_heightmap`, `allowed_material_types`, …) is
   declared in `profiles/<name>.yaml`. The validator reads those flags; it
   does not encode per-format rules in Python.
3. **Material semantics are separate from material encoding.**
   `MaterialType` and `SpatialVariation` describe what the scene contains;
   `ExportProfile.material_encoding` and `texture_export` describe how the
   exporter writes it. v1 disallows e.g. uv_texture by listing only
   `[uniform, per_triangle]` in `allowed_spatial_variation`; v1 doesn't have
   to re-implement that check.
4. **Lightweight import.** Top-level `renderformer.data` imports should not
   pull in Blender (`bpy`), DALI, diffusers, or torch. Heavy backends sit
   under `renderformer.data.blender.*` / `renderformer.data.textures.*` and
   are lazy-imported.
5. **Postprocess is part of the datagen chain.** The on-disk H5 schema the
   v2 model and DALI loader consume is what `postprocess_v2_h5` produces,
   not what the export step writes. Texture VAE encoding, env_map rotation,
   and `volume_data → volume_density` reshape all happen there.

---

## Package layout

```
src/renderformer/data/
├── __init__.py             # import-light public H5 scene API
├── envmap.py               # torch-only lat-long directions, sampling, and rotation
├── envmap_assets.py        # offline six-permutation HDRI collection builder
├── blender/                # bpy-heavy scene generation & rendering
│   ├── pipeline.py             # `generate_prepared_scene[_from_template]`
│   ├── scene_builder.py        # `build_prepared_scene` (format-agnostic)
│   ├── renderer.py             # Cycles render + view filtering
│   ├── mesh.py                 # Scene/mesh synthesis from template
│   ├── templates.py            # Load packaged template JSONCs + manifests
│   ├── template_dataclass.py   # Template schema dataclasses
│   ├── generated_config.py     # GeneratedConfig / MaterialConfig dataclasses
│   ├── material_types.py       # Material type string constants + supports_svbrdf
│   ├── texture_sample_fn.py    # UV-sampling for SVBRDF materials
│   ├── brdf_mapper.py          # BRDF -> 9-d latent learned head
│   ├── brdf_utils.py           # map_brdf_to_latent + helpers
│   └── volume_utils.py         # VDB I/O + micro-grid compaction
├── capabilities/
│   └── validator.py            # Profile-driven scene validation
├── cli/
│   └── main.py                 # implementation behind `renderformer data`
├── exporters/                  # Format-specific H5 writers
│   ├── common.py                   # PreparedScene / ObjectPreparedData / RenderResults
│   ├── v1.py                       # RF1 13-channel + NDC normal + mask
│   └── v2.py                       # latent + normal[-1,1] + emissive [+ heightmap]
├── h5/                         # Canonical atomic H5 I/O and schema validation
│   ├── io.py                       # Atomic H5 writer
│   └── validate.py                 # H5 schema check used by `validate-h5`
├── geometry/                   # Local mesh normalization/remesh/slice + lazy UV backend
├── loaders/                    # Shared training-time H5 input package
│   ├── config.py                   # YAML/CLI-compatible V1+V2 config
│   ├── common.py                   # File lists, parsing, padding, and crops
│   ├── v1.py                       # RF1 PyTorch + optional DALI loader
│   └── multires.py                 # V2 weighted multires + env/volume loader
├── pipelines/
│   └── generate.py             # `run_exports` orchestration of multi-profile jobs
├── profiles/
│   ├── v1_training_rf1.yaml    # RF1 release profile
│   └── v2_training_raw.yaml    # V2 raw training-data profile
├── schemas/
│   ├── export_profile.py       # ExportProfile + enums
│   ├── io.py                       # Profile / scene loader helpers
│   ├── material.py                 # MaterialType / MaterialSource / SpatialVariation / MaterialSemantic
│   └── scene.py                    # ObjectSemantic / SceneSemantic
├── templates/
│   ├── training_v1_final_manifest.json         # 85 logical v1 entries (5 datasets, 71 files)
│   ├── training_v2_200m_curriculum_manifest.json  # 84 v2 templates (default)
│   ├── v1/                                     # packaged v1 template JSONCs
│   ├── full-new-volume/                        # packaged v2 template JSONCs
│   ├── meshes/, bbox/, lighting/, camera/, special-objects/  # shared assets
└── textures/
    ├── encoders.py             # TextureEncoder protocol + Passthrough / QwenVAE / DC-AE
    └── postprocess.py          # `postprocess_v2_h5` (texture VAE + env rot + volume reshape)
```

---

## End-to-end data flow

```
template JSONC                profile YAML
      │                            │
      ▼                            │
generate_prepared_scene_from_template  (renderformer.data.blender.pipeline)
      │
      ├── generate_scene_from_template (mesh.py)
      │       └── writes scene.obj + split/<obj>.obj per scene_config sample
      ├── scene_to_img (renderer.py)
      │       └── Cycles render, returns per-view (rgb, c2w, depth, …)
      └── build_prepared_scene (scene_builder.py)
              └── per-object data via _collect_objects:
                     • triangles, vn
                     • face_diffuse from mesh.visual.face_colors
                     • full MaterialConfig
                     • svbrdf_texture (only for UV-textured materials)
                     plus render_results + extras (env_map, volume_*)
      ▼
PreparedScene { objects: list[ObjectPreparedData], render_results, extra }
      │
      ▼
run_exports (pipelines/generate.py)
   ├── validate_scene_for_export   (capabilities/validator.py)
   │       └── checks profile.allow_*, profile.allowed_*; raises on policy=error
   └── per profile:
         export_v1_h5  ── 13-ch + NDC normal + lower-tri mask
         export_v2_h5  ── latent(9) + normal(3) + emissive(3) [+ heightmap(1)]
      ▼
raw H5 on disk (v1_training_rf1.h5 / v2_training_raw.h5)
      │
      ▼ (v2 only, when feeding the SSS model)
postprocess_v2_h5  (textures/postprocess.py)
   ├── Qwen-VAE texture encode  (N, 16, H, W) → (N, 80, h, w)
   ├── env_map rotation + strength → CHW (3, H, W)
   └── volume_data (N, 4, 4, 4) → volume_density (N, 64)
      ▼
training-ready H5 → DALI loader / shared RenderFormer pipeline
```

---

## Training loaders

The maintained loaders live in `renderformer.data.loaders` and share one
`TriangleRenderH5DatasetConfig`, so the existing YAML and Tyro field names stay
compatible:

- `loaders.v1` — RF1 PyTorch fallback and DALI training loader.
- `loaders.multires` — V2 weighted file lists, env maps, volumes, aligned
  random crops, `crop_origin`, and `source_resolution`.

```python
from renderformer.data.loaders import TriangleRenderH5DatasetConfig
from renderformer.data.loaders.v1 import TriangleRenderH5Dataset
from renderformer.data.loaders.multires import TriangleRenderH5MultiresDataset

config = TriangleRenderH5DatasetConfig(
    h5_folder_path="training_paths.txt",
    padding_length=4192,
    shuffle_files=False,
)
sample = TriangleRenderH5Dataset(config)[0]
```

New jobs should provide newline-delimited H5 list files. Directories of H5s
remain accepted for CLI compatibility, but explicit list metadata is easier to
reproduce and does not require recursive discovery. V2 supports multiple lists
through `h5_lists` and `h5_list_weights`.

DALI is optional at import time. Constructing a DALI dataset without it raises
an actionable `ImportError`; the PyTorch datasets remain usable on CPU hosts.
Samples that exceed triangle or volume padding fail explicitly instead of being
silently truncated. The full-resolution V2 compatibility dataset does not crop;
only the multires dataset used by the RF2 training runner emits crop metadata.

---

## Environment-map utilities

`renderformer.data.envmap` is the single PyTorch implementation of the Z-up
lat-long convention used by training, augmentation, postprocess, and inference.
It exposes `latlong_ray_direction`, `view_direction_to_latlong`,
`sample_env_map`, and `rotate_env_map`. Batched NCHW/NHWC and unbatched HWC
layouts are supported, and rotations accept either one matrix or one per batch
item. The module is imported explicitly so `import renderformer.data` itself
does not load PyTorch.

V2's channel shuffle is an offline asset transformation, not a tensor or
loader augmentation. `renderformer.data.envmap_assets` creates the six fixed
RGB permutations (`RGB`, `RBG`, `GRB`, `GBR`, `BRG`, `BGR`) for every EXR stem
in an explicit input list. Its CLI atomically publishes a self-contained
non-blank collection with a deterministic six-times-longer list and a JSON
receipt containing source/output hashes, paths, counts, permutations, and
dtype metadata. It never discovers inputs by walking a directory. See
[`shuffle-envmaps`](#shuffle-envmaps-build-the-v2-non-blank-hdri-collection)
and the [external asset contract](../../../external/README.md#prepare-the-v2-channel-shuffled-environment-maps).

All current V2 data recipes enable this correction from stage 1. A `no_env`
recipe still uses the strict blank EXR collection, so the policy does not add
environment illumination. The exact historical blank/non-blank mixing ratio
and list order for the shuffled-plus-blank collection are unavailable; a new
mixture must be constructed explicitly and recorded as a new asset receipt.

---

## Mesh and UV utilities

`renderformer.data.geometry` contains the reusable part of the former
Azure-coupled remesh and UV workers. `voxel_remesh` provides the portable
watertight path through trimesh and scikit-image; `sdf_remesh_cuda` retains the
optional historical CUDA implementation. `simplify_mesh(..., backend="igl")`
calls `igl.qslim` explicitly. `unwrap_uv` imports `bpy_helper` and the pinned
PyPI `bpy==4.5.10` only when UV unwrapping is requested.

Both RenderFormer generations operate on triangle tokens. Avoid coarse meshes
whose triangles occupy large projected image regions: overly large faces do not
provide the spatial token granularity represented in the training data and can
produce warped silhouettes, lighting, and shadows. Remesh before simplifying,
then use QSlim to produce a watertight mesh with approximately uniform,
geometrically meaningful faces. Do not simplify below the point where large
faces replace required silhouette, material-boundary, displacement, or volume
detail. Because projection depends on scene scale and camera distance, the
repository does not prescribe a misleading universal maximum edge length; use
the supplied 1188- and 3968-face preparation tiers as practical starting
points.

For one file:

```bash
python scripts/data/process_mesh.py remesh input.glb output.obj \
  --backend voxel --voxel-resolution 256 \
  --target-faces 3968 --simplify-backend igl
python scripts/data/process_mesh.py unwrap-uv output.obj output_uv.obj --uv-method smart_project
```

For batches, pass `--manifest jobs.json --num-workers N`. The manifest is
explicit metadata rather than a directory scan:

```json
{"jobs": [{"input": "meshes/a.glb", "output": "prepared/a.obj"}]}
```

Use `--backend cuda-sdf` only when the optional CUDA SDF dependencies are
installed. Batch failures are surfaced immediately; the tool never copies an
unprocessed input as a silent fallback.

The maintained collection-level interface is:

```bash
renderformer data prepare-objaverse \
  --input-manifest /datasets/objaverse/sources.jsonl \
  --output-dir /datasets/renderformer-assets/objaverse \
  --num-workers 4
```

It fans each metadata-declared source into independently QSlimmed 1188- and
3968-face-cap OBJ files, applies cube-project UVs, validates topology and UVs,
and derives the three object-list outputs from per-object records. It also
writes source/output hashes, a dependency-and-parameter plan, a sorted JSONL
object inventory, and a final receipt. Repeating the exact command resumes only
after validating completed records. It never scans a source directory.

The canonical fixed-parameter shell receipt, input JSONL schema, output tree,
three-list contract, explicit `special=true` selection, and historical
reproduction boundary are documented in
[`scripts/data/README.md`](../../../scripts/data/README.md#prepare-the-objaverse-mesh-collection).

---

## Schemas

### `MaterialType`, `MaterialSource`, and `SpatialVariation`

`schemas/material.py` defines what a scene contains — independent of how it's
encoded. Source / variation describe *where* material data lives, not what
channel layout to use:

```python
class MaterialType(str, Enum):
    DIFFUSE_SPECULAR = "diffuse_specular"
    METALLIC_ROUGHNESS = "metallic_roughness"
    METALLIC_ROUGHNESS_TRANSMISSION = "metallic_roughness_transmission"
    MEASURED_BRDF = "measured_brdf"

class MaterialSource(str, Enum):
    HOMOGENEOUS = "homogeneous"
    PROCEDURAL = "procedural"
    SVBRDF_TEXTURE = "svbrdf_texture"

class SpatialVariation(str, Enum):
    UNIFORM = "uniform"           # constant across the object
    PER_TRIANGLE = "per_triangle" # face-varying (e.g. rand_tri_diffuse_seed)
    UV_TEXTURE = "uv_texture"     # sampled from UV-mapped texture maps
```

`MaterialSemantic` ties these together plus optional `texture_map_path` and
`use_heightmap`. `ObjectSemantic` adds mesh path + scene flags, `SceneSemantic`
adds top-level `has_env_map` / `has_volume`.

### `ExportProfile` (`schemas/export_profile.py`)

```yaml
format: v1
schema_version: 1
unsupported_policy: error          # error | warn | drop
allowed_material_types: [diffuse_specular]
allowed_spatial_variation: [uniform, per_triangle]
allow_env_map: false
allow_volume: false
allow_heightmap: false
material_encoding:
  format: rf_v1_diffuse_specular_13ch
texture_export:
  mode: raw_only                    # raw_only | encoded_only | raw_and_encoded
  encode_timing: inline             # inline | postprocess
  spatial_encoding: constant_1x1_expanded_to_32
  texture_encoder: {}               # encoder config (qwen_vae / dc_ae / raw)
render_passes: { img: true }
```

`from_dict` infers the boolean capability flags from the older
`env_map.enabled` / `volume.enabled` sections if `allow_env_map` /
`allow_volume` are missing, so legacy profile yamls keep parsing.

### `PreparedScene` and `ObjectPreparedData` (`exporters/common.py`)

The canonical, format-agnostic representation the exporters consume:

```python
@dataclass(frozen=True)
class ObjectPreparedData:
    obj_key: str
    triangles: np.ndarray           # (N_i, 3, 3) float32
    vn: np.ndarray                  # (N_i, 3, 3) float32
    face_diffuse: np.ndarray        # (N_i, 3) [0, 1] — Blender per-face colour
    material: MaterialConfig        # original generated_config.MaterialConfig
    svbrdf_texture: np.ndarray | None  # (N_i, 15 or 16, H, W) only for SVBRDF

@dataclass(frozen=True)
class PreparedScene:
    scene: SceneSemantic
    objects: list[ObjectPreparedData]
    render_results: RenderResults
    output_stem: str = "scene"
    extra: dict[str, Any] = field(default_factory=dict)  # env_map, volume_*, …
```

`RenderResults` holds the per-view `c2w` / `fov` / `mvp` / `img` and optional
render passes (`diffuse_img`, `glossy_img`, …).

---

## Export profiles

Two profiles ship in `profiles/`:

| Profile | Format | Notes |
| --- | --- | --- |
| `v1_training_rf1.yaml` | v1 | RF1 13-channel; mono-specular + white-light scenes only; no env_map / volume / heightmap |
| `v2_training_raw.yaml` | v2 | Diffuse-specular / metallic-roughness / measured BRDF; allow_env_map / allow_volume / allow_heightmap = true; texture postprocess via Qwen VAE |

v2 currently lists `allowed_spatial_variation: [uniform, uv_texture]` — the
exporter still encodes the homogeneous BRDF as a single (9,) latent per object,
so `per_triangle` (rand_tri_diffuse_seed) won't survive into the H5. Re-enable
once a per-face latent path lands.

To add another profile, drop a YAML in `profiles/` and pass it via
`renderformer data generate --profile <path/to/profile.yaml> …`. Multiple
`--profile` flags can be given to write multiple formats from one scene.

---

## Capabilities and validation

`capabilities/validator.py:validate_scene_for_export(scene, profile)` is
pure profile-driven:

- `scene.has_volume` and `profile.allow_volume = False` → `<scene>: volume` issue
- `scene.has_env_map` and `profile.allow_env_map = False` → `<scene>: env_map` issue
- `material.type not in profile.allowed_material_types` → per-object issue
- `material.spatial_variation not in profile.allowed_spatial_variation` → per-object issue
- `material.use_heightmap and not profile.allow_heightmap` → per-object issue

If `profile.unsupported_policy is ERROR`, any issue raises
`UnsupportedFeatureError(issues)`. With `WARN` / `DROP` the issues are
returned for the caller to handle.

No format-specific branches live in the validator. To forbid a feature for a
new format, list / unlist it in that format's profile YAML.

---

## Texture encoding conventions

This is where the format-specific encoding lives, all inside the exporter.

### V1 (`exporters/v1.py`): RF1 13-channel layout

Per face: `[diffuse(3), specular(3), roughness(1), normal(3), irradiance(3)]`.

| Channels | Source |
| --- | --- |
| 0-2 (diffuse) | `obj.face_diffuse` — per-face from `mesh.visual.face_colors / 255`. Captures `rand_tri_diffuse_seed` per-triangle randomness. |
| 3-5 (specular) | `material.diffuse_specular_params['specular']`, broadcast per object. |
| 6 (roughness) | `material.diffuse_specular_params['roughness']`, broadcast. |
| 7-9 (normal, NDC) | `(0.5, 0.5, 1.0)` — flat identity in `[0, 1]` NDC, matching RF1's pre-stored normal map convention. |
| 10-12 (irradiance) | `material.emissive` per object; non-light objects supply `[0, 0, 0]`. |

After concat across objects the exporter applies the RF1 mask
`x + y <= 32` on the 32×32 spatial grid — cells in the upper-right half are
zeroed. The result is `(N_total, 13, 32, 32)` float32.

This is the historical RF1 checkpoint contract. Treat its channel order,
masking rule, dtype, and layout as immutable unless performing an intentional
format migration with a corresponding checkpoint transition.

### V2 (`exporters/v2.py`): SSS latent layout

Per face: `[latent(9), normal(3), emissive(3) [+ heightmap(1)]]`.

| Channels | Source |
| --- | --- |
| 0-8 (latent) | For homogeneous objects: `map_brdf_to_latent(material)`. For SVBRDF: from `obj.svbrdf_texture` produced by `texture_sample` at datagen time. |
| 9-11 (normal, linear) | `(0, 0, 1)` for homogeneous; per-pixel sampled normal in `[-1, 1]` for SVBRDF. |
| 12-14 (emissive) | `material.emissive`, broadcast per face. Also surfaced separately as `light_strength` in the H5 (`raw_texture[:, 12:15, 0, 0]`). |
| 15 (heightmap) | Only present when `profile.allow_heightmap` is true. Zero for homogeneous. |

The v2 exporter writes raw `(N, 15 or 16, 32, 32)` texture, and `light_strength`.
Whether to run a learned texture encoder afterwards is controlled by
`profile.texture_export.mode`:

- `raw_only`: write only raw — let `postprocess_v2_h5` encode later (current v2
  yaml).
- `encoded_only`: run the encoder inline and write only the encoded texture.
- `raw_and_encoded`: write both, with the encoded one as the primary `texture`
  dataset and the raw as `texture_raw`.

`postprocess_v2_h5` is what the V2 generation and training data pipeline uses.

---

## Material latent models

The canonical historical material models live in
`renderformer.models.material`, outside the Blender package. They require a
PyTorch workflow extra such as `blender`, `textures`, `inference`, or
`training`; the lightweight base data install does not load them:

- `MaterialEncoder`, `MaterialDecoder`, and `MaterialAutoencoder` implement the
  fixed 3-channel, 256 x 256 material-image architecture. The encoder produces
  a 9-D `tanh` latent, and the decoder produces a sigmoid-bounded network-space
  image. The release checkpoint contract is alpha-premultiplied, nonnegative
  linear RGB transformed with `log10(1 + RGB)`. Its inverse is
  `10**network_RGB - 1`.
- `DiffuseSpecularToLatent`, `PrincipledBRDFToLatent`, and
  `PrincipledBRDFToLatentWithTransmission` are the three BRDF-parameter MLPs
  used by the V2 exporter. Each produces the same 9-D `tanh` latent.

```python
import torch

from renderformer.models.material import MaterialAutoencoder

model = MaterialAutoencoder.from_pretrained(
    "RenderFormer/renderformer-v2",
    subfolder="material_autoencoder",
    strict=True,
).eval()

# `physical_rgb` is finite, nonnegative linear RGB. Composite EXR alpha first.
physical_rgb = torch.rand(1, 3, 256, 256)
with torch.inference_mode():
    latent = model.encode_hdr(physical_rgb)
    reconstructed_rgb = model.decode_hdr(latent)
```

`encode` and `decode` remain raw network-space methods for checkpoint-compatible
applications. `encode_hdr` and `decode_hdr` apply the physical-space contract
without adding parameters or changing state-dict keys. The selected training
run and its data path establish `log10_1p` as the release contract, and the
published Hub component records it in
`material_autoencoder/preprocessor_config.json`. Legacy
checkpoint helpers remain available for a wrapped `model_state_dict` and load
the complete model, encoder-only component, or decoder-only component strictly.

Material-autoencoder training uses an explicit JSONL manifest rather than
walking a directory. Each nonblank row contains exactly `id`, `path`, and
`split`; paths may be relative to the manifest, `train` is required, and
`validation` is optional:

```json
{"id":"material-000001","path":"exr/material-000001.exr","split":"train"}
{"id":"material-000002","path":"exr/material-000002.exr","split":"validation"}
```

The public generator creates that manifest from a deterministic plan covering
the principled-metallic, diffuse/specular, and principled-transmission sphere
families:

```bash
renderformer data material-spheres \
  --recipe configs/data/material/material_spheres.yaml \
  --validate
```

Planning, explicit shard execution, output verification, and finalization are
documented in the
[material-sphere data receipt](../../../scripts/data/README.md#generate-the-material-autoencoder-exrs).

IDs and resolved paths must be unique. Every EXR must be floating-point,
finite, 256 x 256 RGB or RGBA, and nonnegative after optional alpha
premultiplication. A bad row raises with its ID and path; the loader never
silently substitutes another sample. Validate a manifest before allocating
GPUs:

```bash
MANIFEST=/datasets/material/exr-manifest.jsonl \
OUTPUT_DIR=/runs/material-autoencoder \
  scripts/train/material_autoencoder.sh --validate
```

See the [material training receipt](../../../docs/training/README.md#material-autoencoder-recipe)
for the objective, initialization, checkpoint, and distributed-run contracts.

The official component layout is listed in the root README. The V2
transformers, material autoencoder, and mapper MLPs share one
`RenderFormer/renderformer-v2` revision, so an explicit branch, tag, or commit
selects their complete compatible set atomically. The bundle follows its
default revision when callers omit one. Authenticate whenever the repository
is access-protected during release staging.

These 9-D material models are distinct from the Qwen/DC-AE texture encoders
below, which compress per-triangle texture maps into the wider transformer
input representation.

---

## Postprocess

`textures/postprocess.py:postprocess_v2_h5(input, output, texture_encoder_config, keep_raw)`
is the bridge between a raw renderformer.data v2 export and the on-disk H5
the SSS model + DALI loader consume. It performs three transforms:

1. **Texture VAE encode.** `(N, 16, 32, 32)` raw → `(N, 80, 4, 4)` latents via
   `QwenTextureVAEEncoder` (5 groups of 3 channels each → 5 × 16 = 80
   channels). The Qwen video VAE downsamples 8× spatially, so 32→4.
2. **env_map rotation + strength.** Pops `env_map_rotation` (Euler XYZ
   degrees) and `env_map_strength` from the H5, rotates the lat-long
   env_map by the inverse rotation (so the model sees lighting in the
   orientation Blender rendered the GT with), multiplies by `strength`,
   and writes the result back as CHW `(3, H, W)`.
3. **Volume reshape.** `volume_data: (N, 4, 4, 4)` → `volume_density: (N, 64)`.
   Drops the legacy `volume_indices` / `voxel_indices` helper keys.

Use the CLI wrapper:

```bash
renderformer data convert \
    --input v2_raw.h5 \
    --output v2_training.h5 \
    --texture-encoder qwen_vae \
    --triangle-batch-size 256
```

…or call `postprocess_v2_h5` directly from Python.

V1 does *not* need postprocess — its profile sets
`texture_export.encode_timing: inline` and the v1 exporter applies the RF1
mask and 13-channel layout directly at export time.

---

## Templates and manifests

Templates are JSONC files describing how to generate a scene: which meshes can
be sampled, lighting parameters, camera distributions, material distributions.
They live under `templates/`:

- `templates/v1/<dataset>/*.jsonc` — packaged v1 templates with the dataset
  partition baked into the path (the same legacy `templates/foo.jsonc` can
  appear under multiple data-partition namespaces because legacy paths were
  reused across commits with different contents).
- `templates/full-new-volume/<group>/*.jsonc` — packaged v2 templates.
- Shared assets: `meshes/`, `bbox/`, `lighting/`, `camera/`, `special-objects/`.

Large mesh, texture, and environment-map collections are not bundled. Their
template fields use portable placeholders under `external/`, with compatibility
list names retained from the original sampling contracts (for example
`objects-1188.txt`, `objects-3968.txt`, and
`envmaps-filtered-channel-shuffle-blank.txt`). Supply the real locations at
runtime; no workstation-specific absolute paths are embedded in the package.
The complete list/root contract, MatSynth licensing boundary, local asset
manifest schema, and self-contained V1 check are documented in
[`external/README.md`](../../../external/README.md). Packaged `external/...` paths
resolve relative to the `src/renderformer/data/` package root, not the repository
root.

Manifests list which templates ship with each release profile:

| Manifest | Templates | Description |
| --- | ---: | --- |
| `training_v1_final_manifest.json` | 85 logical / 71 paths | Five historical source partitions underlying two V1 scale stages; 250510 entries share retained 250428 files |
| `training_v2_200m_curriculum_manifest.json` | 84 | **Default for `load_training_template_manifest()`** — nine semantic V2 template groups used across eight active stages |

Loader API in `blender/templates.py`:

```python
from renderformer.data.blender.templates import (
    apply_background_topology_debias, # copy + correct a V1 room template
    iter_training_template_paths,
    iter_training_template_entries,
    load_training_template_manifest,
    load_scene_template,           # parse one JSONC -> SceneTemplate dataclass
    resolve_template_asset_paths,  # rewrite relative paths against the package root
    override_template_asset_paths, # copy + apply runtime list/root locations
    packaged_template_root,
)

# default = training_v2_200m_curriculum
paths = iter_training_template_paths()

# or a specific profile
paths = iter_training_template_paths("training_v1_final_manifest.json")
```

The V1 manifest contains five dated historical source partitions, not five
curriculum stages. The public curriculum groups them into a nominal 1k/256
stage with training padding 1572 and a nominal 4k/512 stage with padding 4192.
The May 2025 `250510` partition belongs to the second stage and keeps its
historical 14-entry, weight-42 sampling and merged-list receipt, but references
the corresponding retained `250428` files. The
`apply_background_topology_debias` policy applies from stage 1 and reproduces
the historical background-render edits without storing a duplicate
fix-topology template directory.

The V2 manifest is a de-duplicated public inventory. Its nine semantic template
groups feed eight active stages. Stage 6 mixes the refraction-plus-non-blank-
environment group at total template weight 16 with the stronger-glass-plus-
blank-environment group at weight 48, giving the documented 1:3 ratio. Stage 7
then trains 64k scenes at 512 resolution, and stage 8 continues from it at 2048
resolution.

The V1 manifest retains its source-partition metadata because those partitions
are part of the RF1 compatibility receipt. The V2 manifest exposes only the
semantic public inventory.

---

## CLI reference

Use `renderformer data <subcommand> ...` after installation, or the equivalent
`python -m renderformer data <subcommand> ...` module form. A source-only
environment must put `src` on `PYTHONPATH`, as the dependency-only Docker
images do.

### `generate`: build a scene and export to one or more formats

```bash
renderformer data generate \
    --template src/renderformer/data/templates/full-new-volume/16k-env-volume/corner-single-object.jsonc \
    --object-list /datasets/renderformer-assets/objaverse/lists/objects-3968.txt \
    --object-root /datasets/renderformer-assets/objaverse/objects \
    --texture-list /datasets/renderformer/textures.txt \
    --texture-root /datasets/renderformer/textures \
    --env-map-list /datasets/renderformer/envmaps.txt \
    --env-map-root /datasets/renderformer/envmaps \
    --profile src/renderformer/data/profiles/v2_training_raw.yaml \
    --output-dir /tmp/out \
    --resolution 512 --spp 4096 \
    --texture-crop-res 256 --texture-target-res 256 --texture-size 32
```

Important flags:

- exactly one of `--scene-config <path.json>`, `--template <path.jsonc>`, or
  `--prepared-frame <frame-directory>`. A prepared frame contains
  `scene_config.json` plus `split/` meshes already exported in world space.
- `--object-list` / `--object-root`, `--texture-list` / `--texture-root`, and
  `--env-map-list` / `--env-map-root` replace the packaged `external/`
  placeholders. These flags are valid only with `--template`; relative
  override paths are resolved from the current working directory.
- `--profile` (repeatable) — one H5 per profile. For example, pass
  `--profile src/renderformer/data/profiles/v1_training_rf1.yaml` and
  `--profile src/renderformer/data/profiles/v2_training_raw.yaml` to write
  both `v1_training_rf1.h5` and `v2_training_raw.h5`
- `--num-views` for template generation only. Prepared frames carry their
  exported camera list, so combining this flag with `--prepared-frame` is an
  error rather than a silently ignored override.
- `--background-topology-debias` applies the consolidated V1 room correction
  policy and is valid only with `--template`. V1 batch recipes enable it
  automatically from data stage 1; V2 recipes explicitly disable it. The
  raised-floor special case is intentionally classified only for canonical
  packaged V1 `more-tri` plane/wall paths; copied or renamed external templates
  still receive mesh-orientation rules but not that path-specific repair.
- `--min-views` requires at least N valid Cycles views; the caller is
  responsible for retrying with a fresh seed on failure.
- `--resolution`, `--spp` (Cycles GT settings)
- `--texture-size 32` for v1. The exported RF1 H5 contract requires exactly
  `(num_triangles, 13, 32, 32)` texture samples; any later texture collapse is
  an internal model-loader detail. The v2 path pads or truncates to 15 or 16
  channels based on `profile.allow_heightmap`.
- `--save-images` to dump per-view PNGs alongside the H5

Prepared frames are validated as self-contained inputs before Blender runs.
Paths must stay within the frame directory, mesh paths must match
`split/<object-key>.obj`, and every required material component must exist.
Generation writes to a hidden sibling staging directory, removes that exact
staging directory after failure, and only publishes it to the requested empty
output path after all exports succeed. It preserves the historical H5-only
smooth-normal pass without changing the render/source OBJ files. The RF2
paper-scene launcher is a complete example:

```bash
bash examples/rf2/generate-paper-scene.sh transparent-torus /tmp/rf2-torus
```

Surface-only prepared scenes can use the pip `blender` extra. Volume scenes
also need Python OpenVDB bindings, which are intentionally not advertised as a
portable pip dependency. Use the Python 3.11 Conda environment in
`environments/datagen.yml` or the dependency-only `docker/datagen.Dockerfile`
environment for those scenes.

### `shuffle-envmaps`: build the V2 non-blank HDRI collection

Input-list entries are safe relative stems without `.exr`. The command reads
only those entries and writes six float32 EXRs per source. The output list is
globally sorted by relative stem, matching the recoverable historical
non-blank list ordering:

```bash
renderformer data shuffle-envmaps \
    --input-list /datasets/renderformer/envmaps-filtered.txt \
    --input-root /datasets/renderformer/envmaps/filtered-512 \
    --output-root /datasets/renderformer/envmaps/filtered-channel-shuffle-512 \
    --num-workers 16
```

The output root is published only after every generated EXR has been decoded
and checked. By default it contains
`envmaps-filtered-channel-shuffle.txt`, `channel-shuffle-manifest.json`, and
the generated EXRs. Use `--output-list-name` or `--manifest-name` to change
the two filenames inside that root. The default process count is
`min(CPU count, 32, source count)`.

An existing output root always fails the run. Choose a new output path and
switch consumers only after validation. A sibling lock rejects concurrent
publishers, and the completed staging directory is published with one rename,
so consumers never see a mixed or partial collection. The command never moves,
replaces, or deletes an existing collection. The receipt hashes the
snapshotted input list, every source and output
EXR, the emitted list, and aggregate source/output inventories. Point the
asset manifest's non-blank list path at the list inside this output root and
its root path at the output root itself.

This utility produces the non-blank six-permutation list only. The historical
`envmaps-filtered-channel-shuffle-blank.txt` contents, blank/non-blank ratio,
and ordering were not recovered. If a recipe needs that collection, construct
a separate mixed root in which every non-blank and blank stem resolves, then
retain the exact list plus hashes or an equivalent receipt. Do not claim that
a new mixture reconstructs the historical one. Strict `no_env` recipes should
use `envmaps-blank.txt`, not a mixed list.

### `validate-assets`: audit template asset locations

With no template selection, this validates the de-duplicated union of all final
V1/V2 manifests. Pass a local schema-v1 asset manifest, as described in
[`external/README.md`](../../../external/README.md), to map portable template
placeholders to your local collections:

```bash
renderformer data validate-assets \
    --asset-manifest /path/to/my-renderformer-assets.json
```

For a quick packaged-only check, no external collections are needed:

```bash
renderformer data validate-assets \
    --template src/renderformer/data/templates/v1/250418_256_res/plane-no-object.jsonc
```

`--template-root` applies the same explicit relative-path root used by
`generate`; otherwise each template root is inferred. `--skip-entry-check`
checks list/root availability without checking each listed file. Missing assets
produce a nonzero exit code and name the exact list, root, or entry that failed.

### `validate-scene`: check a scene config against an export profile

```bash
renderformer data validate-scene --scene scene.json --profile profile.yaml
```

Loads `scene.json` and `profile.yaml`, runs the same validator the exporter
uses, and prints any issues (or raises on `unsupported_policy: error`).

### `validate-h5`: check that an H5 matches its expected schema

```bash
renderformer data validate-h5 --input scene.h5 --format v1
renderformer data validate-h5 --input scene.h5 --format v2
```

This is a shallow base-schema check selected by the explicit `--format` flag.
It verifies the required geometry, texture, camera, and V2 `mvp` datasets,
including their basic shapes and dtypes. It does not read `export_metadata` or
validate V2 light, environment, and volume tensors. Use it as an early CI gate,
then run `renderformer infer --validate-only` with the intended checkpoint to
exercise the model loader's complete input contract before a large inference
run.

### `convert`: postprocess an existing V2 H5

```bash
renderformer data convert \
    --input v2_raw.h5 --output v2_training.h5 \
    --texture-encoder qwen_vae --triangle-batch-size 256
```

Runs `postprocess_v2_h5`. The encoder choice is required so a release workflow
cannot silently select a different encoding backend. `--keep-raw` keeps the raw
16-channel texture under `texture_raw` alongside the encoded one.

To exercise a generated H5 with a checkpoint, switch back to the public
inference surface:

```bash
renderformer infer \
    --input scene.h5 \
    --checkpoint ./checkpoints/v2 \
    --variant v2
```

The data command does not maintain a second inference wrapper.

---

## Programmatic API

Most callers only need the scene builder, a packaged export profile, and the
export dispatcher. This real V2 example assumes Blender is available and that
the external collections described in
[`external/README.md`](../../../external/README.md) have been configured. BRDF
mappers default to the official Hugging Face model IDs:

```python
from pathlib import Path

from renderformer.data.blender.pipeline import generate_prepared_scene_from_template
from renderformer.data.pipelines.generate import ExportJob, run_exports
from renderformer.data.schemas.io import load_export_profile

prepared = generate_prepared_scene_from_template(
    template_path="src/renderformer/data/templates/full-new-volume/16k-env-volume/corner-single-object.jsonc",
    output_dir="/tmp/out",
    object_list_path="/datasets/renderformer-assets/objaverse/lists/objects-3968.txt",
    object_root="/datasets/renderformer-assets/objaverse/objects",
    texture_list_path="/datasets/renderformer/textures.txt",
    texture_root="/datasets/renderformer/textures",
    env_map_list_path="/datasets/renderformer/envmaps.txt",
    env_map_root="/datasets/renderformer/envmaps",
    num_views=4,
    resolution=512,
    spp=4096,
    texture_size=32,
)

v2_profile = load_export_profile(
    "src/renderformer/data/profiles/v2_training_raw.yaml"
)
run_exports(
    prepared,
    [
        ExportJob(
            "v2_training_raw",
            v2_profile,
            Path("/tmp/out/scene_v2_raw.h5"),
        ),
    ],
)
```

The full-volume template above is paired with the packaged V2 profile because
that profile explicitly permits volume and environment inputs. For an RF1
export, choose an RF1-compatible surface-only template and load
`profiles/v1_training_rf1.yaml` instead of combining incompatible capabilities
by hand.

For headless integrations, construct a `PreparedScene` directly and pass it to
the exporter without starting Blender.

Mesh preparation is also available directly:

```python
from renderformer.data.geometry import remesh_file, unwrap_uv

remesh_file("input.glb", "mesh.obj", target_faces=4000)
unwrap_uv("mesh.obj", "mesh_uv.obj", method="smart_project")
```

---

## Adding a new export format

1. Add a `MaterialType` / `SpatialVariation` entry in `schemas/material.py`
   if your format needs a new semantic concept (rare — usually existing
   ones suffice).
2. Drop a `profiles/<name>.yaml`. Set `allowed_material_types`,
   `allowed_spatial_variation`, `allow_env_map` / `allow_volume` /
   `allow_heightmap`, and any `material_encoding` / `texture_export`
   knobs your exporter reads.
3. Add an exporter under `exporters/<name>.py` exposing
   `export_<name>_h5(prepared, profile, output_path) -> ExportResult`. It
   should:
   - Call `validate_scene_for_export(prepared.scene, profile)` first.
   - Loop over `prepared.objects` and build whatever channel layout the
     target model expects.
   - Concatenate across objects, write via `atomic_write_h5`, and put a
     `"format"` key in `export_metadata` for provenance. Callers still select
     the schema explicitly with `validate-h5 --format`.
4. Wire it into `pipelines/generate.py:run_exports` by switching on
   `profile.format`. (Currently a small `if/elif` block; extend it.)
5. If your format needs a learned texture encoder, add an encoder under
   `textures/encoders.py` and register it in `build_texture_encoder`.
   Encoders implement the `TextureEncoder` protocol
   (`encode(texture, triangle_batch_size) -> np.ndarray`) and lazy-import
   torch / diffusers so the package's top-level import stays light.
6. Document the schema contract and validate a generated sample with
   `renderformer data validate-h5` before exposing the new format.
