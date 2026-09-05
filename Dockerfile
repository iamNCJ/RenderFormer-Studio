# RenderFormer CUDA environment image.
#
# This image intentionally contains dependencies only. Application source is
# bind-mounted or copied into /workspace/renderformer when a container starts;
# keeping source out of the image makes one environment usable for any commit.
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04@sha256:b3e7fba84d169f46939f00c25be7d016f712a8d651f4756d6a55e693d84d94f2

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/renderformer-env \
    PATH=/opt/renderformer-env/bin:$PATH \
    PYTHONPATH=/workspace/renderformer/src \
    HF_HOME=/cache/huggingface \
    TORCH_EXTENSIONS_DIR=/cache/torch-extensions \
    TRITON_CACHE_DIR=/cache/triton \
    MPLCONFIGDIR=/cache/matplotlib \
    XDG_CACHE_HOME=/cache/xdg \
    NUMBA_CACHE_DIR=/cache/numba

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ffmpeg \
        git \
        libegl1 \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        tar \
        unzip \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv "$VIRTUAL_ENV" \
    && mkdir -p \
        /cache/huggingface \
        /cache/torch-extensions \
        /cache/triton \
        /cache/matplotlib \
        /cache/xdg \
        /cache/numba \
        /workspace/renderformer \
    && chmod -R 1777 /cache \
    && chmod 0777 /workspace/renderformer

# Only the dependency lock enters the image. In particular, this Dockerfile
# must not COPY project packages or install RenderFormer itself.
COPY docker/requirements-cuda.txt /tmp/requirements-cuda.txt

# A CUDA vendor wheel has a nonstandard platform tag that crashes the final
# platform scan in `pip check`. Run pip's dependency-consistency checker
# directly so missing and conflicting requirements remain fatal.
RUN python -m pip install --upgrade \
        pip==25.1.1 setuptools==80.9.0 wheel==0.45.1 \
    && python -m pip install -r /tmp/requirements-cuda.txt \
    && MAX_JOBS=4 python -m pip install \
        flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir \
    && python -c "from pip._internal.operations.check import check_package_set, create_package_set_from_installed; packages, parsing_problems = create_package_set_from_installed(); missing, conflicting = check_package_set(packages); assert not parsing_problems, 'invalid installed requirement metadata'; assert not missing, f'missing requirements: {missing}'; assert not conflicting, f'conflicting requirements: {conflicting}'" \
    && rm -f /tmp/requirements-cuda.txt

RUN python -c "import sys, torch; assert sys.version_info[:2] == (3, 10); assert torch.__version__.startswith('2.7.1'); assert torch.version.cuda.startswith('12.6')" \
    && python -c "import importlib.metadata, accelerate, deepspeed, diffusers, flash_attn, nvidia.dali, simple_ocio, triton; assert importlib.metadata.version('flash-attn') == '2.7.4.post1'"

WORKDIR /workspace/renderformer

CMD ["/bin/bash"]
