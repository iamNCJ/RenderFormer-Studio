# Environment setup

RenderFormer uses two intentionally separate Conda environments:

- `renderformer-cuda` (Python 3.10) for CUDA inference and training; and
- `renderformer-datagen` (Python 3.11) for scene generation, OpenVDB, and the
  PyPI `bpy==4.5.10` runtime.

Run all commands from the repository root. The complete platform notes and
update procedure are in [`environments/README.md`](../../environments/README.md).

## CUDA inference and training

```bash
conda env create -f environments/cuda.yml
conda activate renderformer-cuda
python -m pip install -r docker/requirements-cuda.txt
MAX_JOBS=4 python -m pip install \
  flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
python -m pip install --no-deps -e .
python -m pip check
```

This environment targets Linux with an NVIDIA driver compatible with CUDA
12.6. Use it for training, V1 CUDA/Triton kernels, DALI, and any V2 execution
profile that enables packed sequences.

## Data generation

```bash
conda env create -f environments/datagen.yml
conda activate renderformer-datagen
python -m pip install \
  torch==2.7.1+cu126 torchvision==0.22.1+cu126 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install bpy==4.5.10 --index-url https://pypi.org/simple
python -m pip install -r docker/requirements-datagen.txt
python -m pip install --no-deps -e .
python -c "import bpy; assert bpy.app.version[:3] == (4, 5, 10)"
```

Use the CPU PyTorch installation documented in
[`environments/README.md`](../../environments/README.md) on a machine without
CUDA. An installed Blender application is neither required nor used.

## Dependency-only Docker images

The Docker images contain operating-system packages, CUDA/Python runtimes, and
third-party Python dependencies. They deliberately do **not** contain a copy of
RenderFormer and do not install this repository as a Python package. Supply the
source tree when the container starts and run the package as a module from that
tree.

This separation keeps an image reusable across source revisions and makes the
code/image boundary explicit instead of hiding source in an image layer. It
also means the `renderformer` console script is not installed in a fresh image.
Use `python -m renderformer <command> ...`; the image sets
`PYTHONPATH=/workspace/renderformer/src` for this source-mounted form.

### Build the images

The root image is the CUDA 12.6, Python 3.10 training/inference environment.
The data-generation image uses Python 3.11, installs `bpy==4.5.10` from PyPI,
and resolves OpenVDB 13 from the same Conda manifest used for local setup.

```bash
docker build -t renderformer-env .
docker build --platform linux/amd64 \
  -f docker/datagen.Dockerfile -t renderformer-datagen-env .
```

The PyPI 4.5.10 release provides a Linux x86-64 wheel but no Linux ARM64
wheel, so the datagen image is intentionally a Linux AMD64 artifact. On an ARM
workstation the command above uses container emulation; it does not claim a
native Linux ARM64 add-on build.

Both builds consume only dependency manifests. The datagen image additionally
consumes `environments/datagen.yml`; it does not compile a second, divergent
OpenVDB stack. The local Conda environments use the same Python/CUDA/OpenVDB
baselines, then consume the same pip requirement files. When a pin changes,
update and validate the Docker image and the corresponding Conda plus pip setup
together.

### Local bind mount

Mount source read-only, and keep caches and outputs on separate writable
mounts. `PYTHONPATH=/workspace/renderformer/src` and the cache locations are
already set by the images.

```bash
mkdir -p outputs

docker run --rm -it --gpus all --ipc=host \
  --mount type=bind,src="$(pwd)",dst=/workspace/renderformer,readonly \
  --mount type=bind,src="$(pwd)/outputs",dst=/outputs \
  --mount type=volume,src=renderformer-cache,dst=/cache \
  renderformer-env \
  python -m renderformer infer --help
```

Use the data-generation environment in the same way:

```bash
docker run --rm -it --gpus all \
  --mount type=bind,src="$(pwd)",dst=/workspace/renderformer,readonly \
  --mount type=bind,src="$(pwd)/outputs",dst=/outputs \
  --mount type=volume,src=renderformer-datagen-cache,dst=/cache \
  renderformer-datagen-env \
  python -m renderformer data --help
```

Use explicit writable mounts for datasets, checkpoints, and training outputs;
do not make the source mount writable merely to store run artifacts.

### Other runtimes

The release does not ship AML or Kubernetes infrastructure. If an external
orchestrator is used, preserve the same boundary as the local example: start
the environment image, provide an exported source tree at
`/workspace/renderformer`, and keep checkpoints, caches, datasets, and outputs
on separate mounts. Do not copy `.git` or bake a source revision into a derived
environment image.

### Validate the image boundary

The source directory is empty in a freshly built environment image:

```bash
docker run --rm renderformer-env /bin/bash -ceu \
  'test ! -e /workspace/renderformer/pyproject.toml; python -m pip check'

docker run --rm renderformer-datagen-env /bin/bash -ceu \
  'test ! -e /workspace/renderformer/pyproject.toml; python -c "import importlib.metadata, bpy, openvdb; assert importlib.metadata.version(\"bpy\") == \"4.5.10\"; assert bpy.app.version[:3] == (4, 5, 10)"'
```

After mounting source, validate the intended workflow. CUDA validation should
exercise a real GPU operation and the packed `flash-attn` path.
Data-generation validation should include a tiny Cycles render and H5 schema
check rather than relying on imports alone.

The datagen command deliberately uses an import/runtime check instead of
`pip check`. The official PyPI 4.5.10 wheels target CPython 3.11 in their
filenames and `Requires-Python`, but contain an incorrect `cp39-cp39` internal
tag, which makes current pip emit a false platform-compatibility error.
