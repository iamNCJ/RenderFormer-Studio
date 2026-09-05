# Conda environments

The environment YAMLs install only the interpreter and compiled system
packages that Conda is responsible for. Pinned Python dependency sets stay in
`docker/requirements-cuda.txt`, `docker/requirements-datagen.txt`, and the
explicit PyTorch, bpy, and FlashAttention commands below, so local environments
and dependency-only Docker images share the same receipts.

Run these commands from the repository root. Either `conda` or a compatible
`mamba` command can create the environments.

Keep the two environments separate. PyPI `bpy==4.5.10` requires NumPy 1.x and
the datagen receipt therefore pins OpenCV 4.11, while the CUDA inference and
training receipt uses NumPy 2.x and OpenCV 4.12. Do not install the `blender`
and `inference` extras into one environment; use the explicit receipt for the
workflow you are running.

## CUDA training and inference

The supported CUDA environment uses Python 3.10 and CUDA 12.6. Create it, then
install the pinned Python dependencies and FlashAttention explicitly:

```bash
conda env create -f environments/cuda.yml
conda activate renderformer-cuda
python -m pip install -r docker/requirements-cuda.txt
MAX_JOBS=4 python -m pip install \
  flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
python -m pip install --no-deps -e .
python -m pip check
```

Use this environment for GPU training, V1 CUDA/Triton kernels, DALI loaders,
and packed-sequence V2 inference. The CUDA environment targets Linux with an
NVIDIA driver compatible with CUDA 12.6.

## Data generation and Blender add-ons

The data-generation environment uses Python 3.11 and Conda's OpenVDB 13
Python binding. Blender's Python runtime is the pinned `bpy==4.5.10` wheel
from PyPI; an installed Blender.app is neither required nor used.

```bash
conda env create -f environments/datagen.yml
conda activate renderformer-datagen
python -m pip install \
  torch==2.7.1+cu126 torchvision==0.22.1+cu126 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install bpy==4.5.10 --index-url https://pypi.org/simple
python -m pip install -r docker/requirements-datagen.txt
python -m pip install --no-deps -e .
PIP_CHECK_OUTPUT="$(python -m pip check 2>&1 || true)"
test "$PIP_CHECK_OUTPUT" = "bpy 4.5.10 is not supported on this platform"
python - <<'PY'
from importlib import metadata
from bpy_helper.io import create_compositing_nodes, render_with_compositing_nodes
from bpy_helper.material import (
    create_emissive_material,
    create_full_principled_bsdf_material,
)
import bpy

assert metadata.version("bpy") == "4.5.10"
assert metadata.version("bpy-helper") == "0.0.13"
assert bpy.app.version[:3] == (4, 5, 10)
PY
```

The CUDA PyTorch command is appropriate for GPU data generation on Linux. On
a CPU-only development machine, replace that command with:

```bash
python -m pip install \
  torch==2.7.1 torchvision==0.22.1 \
  --index-url https://pypi.org/simple
```

Keep Python 3.11, `bpy==4.5.10`, `bpy-helper==0.0.13`, and the remaining
requirements unchanged. Version 0.0.13 is the first `bpy-helper` release that
contains the complete material and compositor API used by the V2 generator.

The official 4.5.10 files on PyPI have CPython 3.11 filenames and
`Requires-Python`, but their embedded `WHEEL` metadata incorrectly says
`cp39-cp39`. Consequently, current `pip check` reports
`bpy 4.5.10 is not supported on this platform` even though pip selected the
CPython 3.11 artifact and the module imports successfully. The explicit
distribution/runtime assertion above is the release gate for this upstream
metadata issue; do not generalize it into ignoring other dependency errors.

To update an existing environment after a YAML change, run
`conda env update --prune -f environments/<name>.yml`, reactivate it, and then
rerun the corresponding explicit pip commands above.
