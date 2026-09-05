# Mixed-resolution training

Train one model on H5 datasets rendered at different image resolutions
(e.g. 256, 512, 2048) in the same batch, with a configurable per-list
sampling ratio. Each sample becomes a fixed `(crop_size, crop_size)`
patch of its source image, and the model receives rays computed in the
source's own pixel coordinate system.

## Why the multiresolution path exists

The retired single-resolution environment-and-volume path did three things
that the RF2 runner keeps as compatibility defaults:

* Generated rays in the model at `trainer_config.img_resolution`.
* Cropped on the GPU after rays were formed.
* Read a single `dataset_config.h5_folder_path`.

This breaks the moment two samples in the same batch come from H5 files
rendered at different resolutions: focal length, principal point, and
ray angular content all depend on the source resolution.

The multires variant fixes three things:

1. **`RayGenerator` accepts per-sample focal info.** New
   `crop_origin / crop_size / source_res` arguments let the model
   generate rays for an arbitrary patch in a virtual `source_res x source_res`
   grid, with focal derived from the per-sample `source_res`. The legacy
   call path (only `c2w / fov / img_res`) is unchanged and remains covered by
   an exact backward-compatibility test.
2. **The dataloader does the crop and emits the metadata.** Crop happens
   at H5-load time so the batch has consistent `(crop_size, crop_size, C)`
   shape regardless of how big the source image is. The dataloader emits
   `crop_origin` and `source_resolution` per view.
3. **Weighted multi-list sampling.** `h5_lists: List[str]` and
   `h5_list_weights: List[float]` replace the single `h5_folder_path`.
   Each entry is a directory of `*.h5` or a text file of paths. The
   sampler picks a list weighted-randomly, then a file uniformly from
   it.

Augmentation is **not** changed: `aug_triangles_and_camera_and_env_and_vol`
still rotates `c2w` in place, and rays are recomputed from the augmented
`c2w` on the next call — exactly as before.

## File map

| Component | File |
|---|---|
| Ray generator (extended) | `src/renderformer/models/ray_generator.py` |
| Multires DALI dataloader | `src/renderformer/data/loaders/multires.py` |
| Shared dataset config | `src/renderformer/data/loaders/config.py::TriangleRenderH5DatasetConfig` |
| Training entrypoint | `renderformer train rf2` (`renderformer.training.train_rf2` runner) |

The multires fields are part of the shared `TriangleRenderH5DatasetConfig`,
so single-resolution train scripts that ignore them keep working unchanged;
this is additive.

## Dataset config

The multires-specific fields live on the shared
`TriangleRenderH5DatasetConfig` (in `src/renderformer/data/loaders/config.py`):

```python
from renderformer.data.loaders import TriangleRenderH5DatasetConfig

dataset_config = TriangleRenderH5DatasetConfig(
    # Use h5_folder_path="/datasets/single/paths.txt" for one list instead.
    h5_lists=["/datasets/4k/paths.txt", "/datasets/16k/paths.txt"],
    h5_list_weights=[0.5, 0.5],       # normalized by the loader
    crop_size=256,                    # GT crops are (crop_size, crop_size)
    crop_alignment=8,                 # crop origin snaps to this multiple
    padding_length=5102,
    volume_padding_length=512,
    img_key="img",
    shuffle_files=True,
)
```

For the multires script, either `h5_lists` (preferred) or `h5_folder_path`
must be set; the former takes precedence. Single-resolution dataloaders
read only `h5_folder_path` and ignore the rest.

Use newline-delimited file-list metadata for reproducible published runs.
Directory scanning remains supported only for compatibility with old CLI
commands.

`trainer_config.crop_size` overrides `dataset_config.crop_size` at
startup so there is one source of truth — set it in `trainer_config` if
you change it via CLI.

## Per-sample fields in the batch

The DALI batch dict adds two new fields on top of the env+vol baseline:

| Key | Shape | Meaning |
|---|---|---|
| `crop_origin` | `(B, V, 2)` float32 | `(y0, x0)` in source-pixel coords, snapped to `crop_alignment` |
| `source_resolution` | `(B, V)` float32 | Side of the source square image (assumed `H == W`) |

The training script passes these directly into `RayGenerator`:

```python
rays_o, rays_d = ray_generator(
    c2w_for_view_tf, fov,
    crop_origin=crop_origin,
    crop_size=crop_size,
    source_res=source_resolution,
)
```

## Launching a run

Use the release model YAML as the architecture base and load a transformer
checkpoint explicitly. Dataset and trainer options remain command-line
overrides:

```bash
export DATASET_ROOT=/path/to/multires/manifests
export OUTPUT_ROOT=/path/to/training/runs
export CHECKPOINT_DIR=/path/to/checkpoint

WANDB_MODE=disabled \
python -m torch.distributed.run \
  --standalone --nproc-per-node 1 \
  --module renderformer.training.cli rf2 \
  --config_file configs/model/v2/200m_sliding_sink_summary.yaml \
  --dataset-config.h5-lists \
      "${DATASET_ROOT}/paths_2048res.txt" \
      "${DATASET_ROOT}/paths_512res.txt" \
  --dataset-config.h5-list-weights 1.0 1.0 \
  --dataset-config.crop-size 256 \
  --dataset-config.padding-length 5102 \
  --dataset-config.volume-padding-length 1024 \
  --trainer-config.crop-size 256 \
  --trainer-config.batch-size 8 \
  --trainer-config.num-view-limit 2 \
  --trainer-config.exp-id my-multires-run \
  --trainer-config.log-dir "${OUTPUT_ROOT}" \
  --trainer-config.load-weights "${CHECKPOINT_DIR}/model.safetensors"
```

Notes:

* `--dataset-config.h5-lists A B C` and
  `--dataset-config.h5-list-weights 1 2 0.5` take N space-separated args
  — any number of lists is supported.
* Use `--trainer-config.no-gradient-checkpointing` to disable
  checkpointing if it was enabled in the base YAML.
* If a sample's triangle or volume count exceeds its configured padding,
  the dataloader fails with a field-specific error. Increase the matching
  padding length or pre-filter the manifest before starting the run.

## How "mixed resolution" actually flows

```
H5 file (img: VxHxWxC, H=W=source_res, varies per file)
  │
  ▼
dataloader random-crops img -> (V, crop_size, crop_size, C)
                     emits   crop_origin (V, 2) in source pixels
                     emits   source_resolution = H
  │
  ▼
batch: img stacked to (B, V, K, K, C) — same shape regardless of source res
       crop_origin (B, V, 2), source_resolution (B, V), c2w (B, V, 4, 4), fov (B, V, 1)
  │
  ▼
aug_*  rotates c2w_aug, triangles, vns, env_map, volumes (unchanged)
  │
  ▼
RayGenerator(c2w_aug, fov,
             crop_origin, crop_size=K, source_res=src_res)
   focal_x = focal_y = source_res / 2 / tan(fov/2)        # per-sample
   pixel center = (crop_origin + 0.5 + arange(K)) in source pixels
   rays_d in camera frame, rotated by c2w_aug[:3,:3]
  │
  ▼
model forward: rays_d has shape (B, V, K, K, 3), gt img matches
```

The geometric compatibility contract reduces to one identity: generating rays
with `(source_res=R, crop_origin=(y, x), crop_size=K)` is numerically equivalent
within `1e-6` to slicing a full `R x R` ray map produced by the legacy path at
`[y:y+K, x:x+K]`.

## Validation

Use newline-delimited H5 manifests for reproducible runs and validate every H5
against the selected checkpoint contract before allocating GPUs. Confirm crop
geometry, mixed-list weights, bounds, and pixel preservation on a small local
batch before starting a distributed run.

## Limitations / gotchas

* **Square images only.** `source_res` is a scalar per sample; the
  dataloader rejects a sample unless `H == W`.
* **`crop_size` ≤ smallest source.** If any list contains files with
  `source_res < crop_size`, the dataloader will error per sample.
* **64k-triangle lists need bigger `padding_length`.** If a sample contains
  more triangles than the configured padding, the loader raises `ValueError`
  and stops instead of silently skipping that sample. Validate the list and
  increase `padding_length` before training.
* **Model's positional encoding range matters.** A ckpt trained at
  `crop_size=512` (`K/8 = 64` patch tokens per side) can run at
  `crop_size=256` (`K/8 = 32`) only because Swin / RoPE here happen to
  generalize over the patch grid. If you change `patch_size`, this no
  longer holds — always sanity-check via the mosaic demo first.
