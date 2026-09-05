# Environment-map reflection shapes

This optional local pipeline generates synthetic meshes with the PyPI
`bpy==4.5.10` wheel, prepares their topology, and adds UV coordinates. It runs
through the active Python interpreter and never invokes a separately installed
Blender application. It has no cloud queue, storage-account, or internal mount
dependency.

## Shape types

The generator supports ellipsoids, superellipsoids, metaballs, Platonic solids,
convex hulls, tori, and organic blobs. Generation writes `all_shapes.json`; the
processing step uses that metadata file instead of scanning the directory.

## Generate and process

Run from the repository root:

```bash
python scripts/data/env_map_shapes/pipeline.py \
  --input-dir ./shape-source \
  --output-dir ./shape-output \
  --generate \
  --num-shapes 100 \
  --target-faces 2000 \
  --uv-method sphere_project \
  --num-workers 4
```

To generate only, invoke the bpy-backed script with the same Python
environment:

```bash
python scripts/data/env_map_shapes/create_convex_shapes.py \
  --output_dir ./shape-source \
  --num_shapes 100 \
  --seed 42 \
  --shape_types all
```

For a large generation-only batch, the process-based launcher uses that same
Python executable for every isolated worker:

```bash
python scripts/data/env_map_shapes/generate_parallel.py \
  --output-dir ./shape-source \
  --num-shapes 10000 \
  --num-workers 8
```

Both launchers fail before generation unless the active environment contains
the PyPI `bpy==4.5.10` distribution and its 4.5.10 runtime. Create the datagen
environment as described in
[`docs/environment-setup/README.md`](../../../docs/environment-setup/README.md)
before running them.

The default `simplify` remesh backend is local and CPU-oriented. The historical
watertight CUDA path is available explicitly with `--remesh-backend cuda-sdf`;
it requires `torchcumesh2sdf` and `diso`. UV unwrapping requires the pinned
`bpy` and `bpy_helper` packages. Simplification uses libigl when installed,
otherwise trimesh's optional `fast-simplification` backend.

Each processed shape contains:

```text
{shape_id}/
├── original.obj
├── remeshed.obj
├── final.obj
└── metadata.json
```

The output root also contains `processed_shapes.json`, which records the exact
paths produced. Any invalid mesh or missing optional backend fails immediately;
the pipeline does not silently substitute unprocessed geometry.
