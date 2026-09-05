# External asset contract

RenderFormer ships scene templates and the small fixed meshes under
`src/renderformer/data/templates/`. It does **not** redistribute the large object,
material, or environment-map collections used to train V1 and V2. No asset
downloader is included.

This directory contains documentation only. It is not the default runtime
asset directory. For a packaged template,
`templates.py` resolves relative paths against the installed
`renderformer.data` package directory:

```text
templates/meshes/plane.obj  -> <data-package-root>/templates/meshes/plane.obj
external/objects-1188.txt   -> <data-package-root>/external/objects-1188.txt
```

In an editable source checkout, `<data-package-root>` is this repository's
`src/renderformer/data/` directory. More generally, path inference finds the
nearest ancestor directory named `templates` and uses its parent as the root;
if there is no such ancestor, it uses the JSONC file's directory. Generation
can replace that inference with `--template-root`. Prefer runtime overrides
instead of copying large data into an installed package.

## Objaverse source boundary

The three `external/objects-*.txt` paths in scene templates are placeholders for
generated data artifacts. Do not populate them manually or copy lists into the
Python package. Acquire the original meshes from
[Objaverse](https://objaverse.allenai.org/) under their per-object terms, retain
the project website's existing asset credits, and give the local paths and
stable UIDs to the canonical
[Objaverse data receipt](../scripts/data/README.md#prepare-the-objaverse-mesh-collection).

That receipt creates watertight meshes, libigl QSlim tiers, cube-project UVs,
per-object hashes, and the two face-cap lists. The face-cap numbers 1188 and
3968 are not collection sizes. The refraction-only
`objects-special.txt` is derived solely from source rows explicitly marked
`special=true`; the old private selection is not claimed or reconstructed.

Each generated list row is a relative mesh path including `.obj`, resolved as
`<root>/<entry>`, where `<root>` is the receipt's `objaverse/objects`
directory. The two V1
lists under `templates/special-objects/` are unrelated small package assets and
require no external collection. A template whose object count has `max: 0`
does not open its retained object-list placeholder.

All V2 templates carry the following MatSynth-derived material placeholders:

| Template list placeholder | Root placeholder |
| --- | --- |
| `external/textures-matsynth.txt` | `external/textures/matsynth` |

The seven `1k-no-env-no-volume-homogeneous` templates explicitly assign zero
probability to both SVBRDF types. Runtime loading and validation therefore do
not open their retained texture-list placeholder. V1 final templates have no
texture-list or texture-root fields.

A texture-list entry is a directory relative to the root. Each selected
directory must contain:

- diffuse/specular mode: `diffuse.png`, `specular.png`, `roughness.png`,
  `normal.png`
- metallic/roughness mode: `basecolor.png` or `base_color.png`, plus
  `metallic.png`, `roughness.png`, `normal.png`
- optional displacement: `height.png` (generation falls back when absent)

MatSynth data is not bundled and this repository does not grant rights to it.
Acquire it from the MatSynth project or another authorized source, review the
terms for the exact release you use, and prepare the list and map directories
yourself. Do not assume that this repository's code license covers third-party
materials.

Every V2 template also uses one of these environment-map collections. List
entries are relative stems **without** `.exr`; runtime resolves each entry as
`<root>/<entry>.exr`.

| Template list placeholder | Root placeholder | Used by final profiles |
| --- | --- | --- |
| `external/envmaps-blank.txt` | `external/envmaps/blank-512` | V2-200M |
| `external/envmaps-filtered-channel-shuffle-blank.txt` | `external/envmaps/filtered-channel-shuffle-512` | V2-200M |
| `external/envmaps-filtered-channel-shuffle.txt` | `external/envmaps/filtered-channel-shuffle-512` | V2-200M |

The old plain `envmaps-filtered-blank.txt` placeholder is not part of the
release contract because no retained template uses it. The `no-env` V2 groups
still use a strict blank EXR collection; they are not free of environment-map
files. V1 final templates have no environment-map list or root. V2 volumes
have no list or asset root: when enabled,
they are generated procedurally from each template's `volume_generation_params`
and written to the generation output/cache. The Blender/OpenVDB runtime is
still required for volume generation. V1 final templates have no volume block.

## Prepare the V2 channel-shuffled environment maps

Every current V2 data recipe applies the environment-map channel-shuffle
policy from stage 1. This is an offline six-way RGB permutation, not file-list,
task, or training-loader shuffling. Start from an explicit metadata list whose
entries are relative EXR stems without `.exr`:

```bash
renderformer data shuffle-envmaps \
  --input-list /datasets/renderformer/envmaps-filtered.txt \
  --input-root /datasets/renderformer/envmaps/filtered-512 \
  --output-root /datasets/renderformer/envmaps/filtered-channel-shuffle-512 \
  --num-workers 16
```

For each input, the command writes float32 `_RGB`, `_RBG`, `_GRB`, `_GBR`,
`_BRG`, and `_BGR` EXRs. It atomically publishes one self-contained output
root only after decoding and checking every generated EXR. That root contains
the images plus `envmaps-filtered-channel-shuffle.txt` and
`channel-shuffle-manifest.json` by default. The six-times-longer list is
globally sorted by relative stem, matching the recoverable historical
non-blank ordering. Point the asset manifest's non-blank `list_path` at this
file and its `root_path` at the containing output root.

Use `--output-list-name` and `--manifest-name` to choose other filenames
inside the output root. The default worker count is
`min(CPU count, 32, source count)`. The receipt records the snapshotted input
list hash, per-file and aggregate source/output hashes, output-list hash,
paths, shapes, dtypes, counts, and permutations. The command reads only the
metadata list; it does not scan the root or download HDRIs. An existing output
root always fails; choose a new output path and switch consumers only after
validation. A sibling lock rejects concurrent publishers. The completed
staging directory is published with one rename, so consumers never see a mixed
or partial collection. The command never moves, replaces, or deletes an
existing collection.

This command deliberately produces only the non-blank shuffled collection.
The exact blank/non-blank ratio and list ordering of the historical
`envmaps-filtered-channel-shuffle-blank.txt` were not recovered. To use the
mixed placeholder, construct a separate mixed root in which all generated and
blank stems resolve, then retain its exact list plus hashes or an equivalent
receipt. That new mixture is distribution-compatible only; do not claim it
reproduces the historical list. A strict `no_env` recipe uses
`envmaps-blank.txt`; permuting a blank EXR is a no-op and never introduces
environment illumination.

## BRDF mapper weights (separate from collection assets)

V2 generation/export also maps material parameters into the model's 9-D BRDF
latent. Those weights are not bundled with the source tree and are intentionally
outside the object/texture/environment asset manifest. With no overrides, the
data pipeline downloads these components from the componentized
`RenderFormer/renderformer-v2` bundle:

| Mapper | Repository | Component subfolder |
| --- | --- | --- |
| Diffuse/specular | `RenderFormer/renderformer-v2` | `diffspec_mapper` |
| Metallic/roughness | `RenderFormer/renderformer-v2` | `metallic_mapper` |
| Metallic/roughness/transmission | `RenderFormer/renderformer-v2` | `metallic_transmission_mapper` |

To use local or alternative mapper weights, configure individual
directories/model IDs:

```bash
export RENDERFORMER_DIFFSPEC_MAPPER=/path/or/model-id/for/diffuse-specular
export RENDERFORMER_METALLIC_MAPPER=/path/or/model-id/for/metallic-roughness
export RENDERFORMER_TRANSMISSION_MAPPER=/path/or/model-id/for/transmission
```

Alternatively, set `RENDERFORMER_BRDF_MAPPER_ROOT` to a directory containing
`diffspec_mapper/`, `metallic_mapper/`, and `metallic_transmission_mapper/`.
The validator does not check these values because a valid value may be a remote
model ID rather than a local path. V1's final exporter encodes its homogeneous
diffuse/specular parameters directly and does not require these mapper weights.

The canonical mapper classes are exported from `renderformer.models.material`:

| Directory | Class | Parameters mapped to the 9-D latent |
| --- | --- | --- |
| `diffspec_mapper/` | `DiffuseSpecularToLatent` | diffuse RGB, specular RGB, roughness |
| `metallic_mapper/` | `PrincipledBRDFToLatent` | base-color RGB, metallic, roughness |
| `metallic_transmission_mapper/` | `PrincipledBRDFToLatentWithTransmission` | base-color RGB, metallic, roughness, IOR, transmission |

The same module contains the historical 3 x 256 x 256 material
encoder/decoder/autoencoder. Its latent is 9-D with `tanh`, and its sigmoid
decoder returns a bounded network-space image. The release checkpoint consumes
alpha-premultiplied, finite, nonnegative linear RGB transformed as
`log10(1 + RGB)`; the physical-space inverse is `10**network_RGB - 1`.
`MaterialAutoencoder.encode_hdr` and `decode_hdr` implement this contract.
Strict loaders for the full, encoder-only, and decoder-only historical
checkpoints are available.

The two V2 transformers, selected autoencoder, and three mapper artifacts share
one Hugging Face revision, so a branch, tag, or commit identifies the complete
component set atomically. Callers that require a fixed revision can load one
explicitly or point the mapper environment variables at a local snapshot. The
autoencoder loads directly from `RenderFormer/renderformer-v2` with
`subfolder="material_autoencoder"` and
`MaterialAutoencoder.from_pretrained(..., strict=True)`. Its component-level
`preprocessor_config.json` records `log10_1p`; the published safetensors contain
only model weights, while historical optimizer, scheduler, and training-history
state is excluded. These models remain separate from collection assets and are
not bundled in this source tree. Authenticate with Hugging Face whenever the
V2 bundle is access-protected during release staging.

## Validate a local asset layout

Create a local schema-v1 JSON manifest that maps each `external/...` placeholder
listed in this guide to its `kind`, metadata `list_path`, and data `root_path`.
Keep machine-specific paths outside the source checkout, then validate the
union of all final templates:

```json
{
  "schema_version": 1,
  "collections": {
    "external/objects-1188.txt": {
      "kind": "object",
      "list_path": "objaverse/lists/objects-1188.txt",
      "root_path": "objaverse/objects"
    }
  }
}
```

Relative paths resolve from the manifest's parent directory. Add entries with
the same three fields for `objects-3968.txt`, `objects-special.txt`, the
material list, and each environment-map list used by the selected stage. The
validator reports every required placeholder that is still missing.

```bash
renderformer data validate-assets \
  --asset-manifest /path/to/my-renderformer-assets.json
```

The validator reads the explicit metadata lists rather than scanning asset
directories. It checks fixed packaged meshes, required lists and roots, object
files, EXR naming, and material map files. Use `--skip-entry-check` for a quick
list/root-only check. Use `--template path/to/template.jsonc` to validate one
template, `--template-root` to mirror generation's explicit path root, or
repeat `--training-manifest` to select release manifests.

The asset manifest is only a portable mapping for validation. During generation
the explicit `--object-list` / `--object-root`, `--texture-list` /
`--texture-root`, and `--env-map-list` / `--env-map-root` flags override the
placeholders for the selected template.

## Self-contained V1 check

`plane-no-object.jsonc` samples zero external objects, has no texture list,
environment map, or volume, and uses only fixed package meshes. First verify the
asset boundary:

```bash
renderformer data validate-assets \
  --template src/renderformer/data/templates/v1/250418_256_res/plane-no-object.jsonc
```

Then run a small Blender generation check (the `blender` optional dependencies
and a working Cycles installation are still required):

```bash
renderformer data generate \
  --template src/renderformer/data/templates/v1/250418_256_res/plane-no-object.jsonc \
  --profile src/renderformer/data/profiles/v1_training_rf1.yaml \
  --output-dir /tmp/renderformer-v1-self-contained \
  --num-views 1 --min-views 1 \
  --resolution 256 --spp 64 --texture-size 32
```

For a check that includes a sampled object while remaining self-contained, use
`src/renderformer/data/templates/v1/250416_high_res_fixed_bsdf_dev/corner-single-object-special.jsonc`.
