# Blender/OpenVDB environment image for RenderFormer data generation.
#
# The repository is supplied at runtime. Only environment manifests enter this
# image; application source is never copied or installed here.
FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04@sha256:4cf7f8137bdeeb099b1f2de126e505aa1f01b6e4471d13faf93727a9bf83d539

ARG MINIFORGE_VERSION=25.3.1-0
ARG MINIFORGE_SHA256=376b160ed8130820db0ab0f3826ac1fc85923647f75c1b8231166e3d559ab768
ARG BLENDER_VERSION=4.5.10

ENV DEBIAN_FRONTEND=noninteractive \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CONDA_DIR=/opt/conda \
    CONDA_ENV=/opt/conda/envs/renderformer-datagen \
    PYTHONPATH=/workspace/renderformer/src \
    HF_HOME=/cache/huggingface \
    TORCH_EXTENSIONS_DIR=/cache/torch-extensions \
    TRITON_CACHE_DIR=/cache/triton \
    MPLCONFIGDIR=/cache/matplotlib \
    XDG_CACHE_HOME=/cache/xdg

ENV PATH=${CONDA_ENV}/bin:${CONDA_DIR}/bin:${PATH}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libegl1 \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libx11-6 \
        libxcursor1 \
        libxext6 \
        libxi6 \
        libxinerama1 \
        libxkbcommon-x11-0 \
        libxrandr2 \
        libxrender1 \
        libxxf86vm1 \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSLo /tmp/miniforge.sh \
        "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh" \
    && echo "${MINIFORGE_SHA256}  /tmp/miniforge.sh" | sha256sum --check --strict \
    && bash /tmp/miniforge.sh -b -p "${CONDA_DIR}" \
    && rm -f /tmp/miniforge.sh

# These are dependency manifests, not RenderFormer application source. Conda
# owns Python/OpenVDB and pip owns the pinned Python workflow dependencies.
COPY environments/datagen.yml /tmp/environment.yml
COPY docker/requirements-datagen.txt /tmp/requirements-datagen.txt

RUN conda env create --file /tmp/environment.yml \
    && conda clean --all --yes \
    && python -m pip install --upgrade \
        pip==25.1.1 setuptools==80.9.0 wheel==0.45.1 \
    && python -m pip install \
        torch==2.7.1+cu126 torchvision==0.22.1+cu126 \
        --index-url https://download.pytorch.org/whl/cu126 \
    && python -m pip install \
        "bpy==${BLENDER_VERSION}" \
        --index-url https://pypi.org/simple \
    && python -m pip install -r /tmp/requirements-datagen.txt \
    && rm -f /tmp/environment.yml /tmp/requirements-datagen.txt

# Prefer the Conda C++ runtime as well as its executables.  This keeps PyPI bpy
# and Conda OpenVDB loadable in either import order inside one Python process.
ENV LD_LIBRARY_PATH=${CONDA_ENV}/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

RUN mkdir -p \
        /cache/huggingface \
        /cache/torch-extensions \
        /cache/triton \
        /cache/matplotlib \
        /cache/xdg \
        /workspace/renderformer \
    && chmod -R 1777 /cache \
    && chmod 0777 /workspace/renderformer \
    && command -v tar >/dev/null \
    && PIP_CHECK_OUTPUT="$(python -m pip check 2>&1 || true)" \
    && { [ "${PIP_CHECK_OUTPUT}" = "No broken requirements found." ] \
        || [ "${PIP_CHECK_OUTPUT}" = "bpy 4.5.10 is not supported on this platform" ]; } \
    && python -c "import importlib.metadata, sys, bpy, torch; assert sys.version_info[:2] == (3, 11); assert importlib.metadata.version('bpy') == '4.5.10'; assert importlib.metadata.version('bpy-helper') == '0.0.13'; assert bpy.app.version[:3] == (4, 5, 10); assert torch.__version__.startswith('2.7.1'); assert torch.version.cuda.startswith('12.6')" \
    && python -c "from bpy_helper.io import create_compositing_nodes, render_with_compositing_nodes; from bpy_helper.material import create_emissive_material, create_full_principled_bsdf_material" \
    && python -c "import bpy, cv2, diffusers, h5py, imageio, openvdb, pymeshlab, scipy, simple_exr, torchvision, trimesh"

WORKDIR /workspace/renderformer

CMD ["/bin/bash"]
