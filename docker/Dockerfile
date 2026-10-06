# runtime (not devel) is enough: torch/flash-attn/natten are all prebuilt
# wheels that bundle their CUDA libs, and triton JITs with its own ptxas.
# Host requirement: NVIDIA driver >= 580 (CUDA 13) to run the cu130 wheels.
FROM nvidia/cuda:13.0.3-runtime-ubuntu24.04

LABEL authors="jaret"

# Set noninteractive to avoid timezone prompts
ENV DEBIAN_FRONTEND=noninteractive

# ref https://en.wikipedia.org/wiki/CUDA
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0 10.0 12.0"

# Install dependencies
RUN apt-get update && apt-get install --no-install-recommends -y \
    git \
    curl \
    build-essential \
    cmake \
    wget \
    python3.12 \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python3-wheel \
    python3-venv \
    ffmpeg \
    tmux \
    htop \
    nvtop \
    python3-opencv \
    openssh-client \
    openssh-server \
    openssl \
    rsync \
    unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install nodejs
WORKDIR /tmp
RUN curl -sL https://deb.nodesource.com/setup_23.x -o nodesource_setup.sh && \
    bash nodesource_setup.sh && \
    apt-get update && \
    apt-get install -y nodejs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Set aliases for python and pip
RUN ln -s /usr/bin/python3 /usr/bin/python

# install pytorch before cache bust to avoid redownloading pytorch
# (versions must match manager/spec.py — the AI Toolkit Manager's linux spec)
RUN pip install --no-cache-dir torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130 --break-system-packages

WORKDIR /app/ai-toolkit

# ---------------------------------------------------------------------------- #
# Dependency layers come BEFORE the source clone so they are only rebuilt (and
# only need to be re-pulled by servers) when the dependency manifests change,
# not on every code change.
# ---------------------------------------------------------------------------- #

# Install Python dependencies (only re-runs when the requirements files change)
COPY requirements.txt requirements_base.txt /app/ai-toolkit/
COPY docker/runtime-constraints.txt /app/runtime-constraints.txt
ENV PIP_CONSTRAINT=/app/runtime-constraints.txt
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# Ensure the GPU runtime wins over InsightFace's CPU onnxruntime dependency.
RUN pip install --no-cache-dir --break-system-packages --force-reinstall --no-deps onnxruntime-gpu==1.26.0 && \
    python -c "import nvidia, glob, os; print('\n'.join(sorted({d for b in nvidia.__path__ for d in glob.glob(os.path.join(b, '*', 'lib'))})))" > /etc/ld.so.conf.d/nvidia-pip-libs.conf && \
    ldconfig && \
    python -c "import onnxruntime as o; p=o.get_available_providers(); assert 'CUDAExecutionProvider' in p, p"

# Accelerators, matching the manager's linux cu130 spec (manager/spec.py):
# flash-attn 2.8.3 (prebuilt for torch 2.13 / cu130 / cp312), NATTEN 0.21.7,
# and torchcodec 0.15. Installed AFTER requirements with -U so they override
# any older pins in there (same order the manager uses).
RUN pip install --no-cache-dir --break-system-packages -U \
    torchcodec==0.15.0 \
    natten==0.21.7+torch2130cu130 --find-links https://whl.natten.org \
    https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.47/flash_attn-2.8.3+cu130torch2.13-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl && \
    python -c "import flash_attn, natten, torchcodec; print('accelerators OK:', flash_attn.__version__, natten.__version__, torchcodec.__version__)"

# Install Node dependencies (only re-runs when package.json / package-lock.json change)
COPY ui/package.json ui/package-lock.json /app/ai-toolkit/ui/
RUN cd /app/ai-toolkit/ui && npm ci

# ---------------------------------------------------------------------------- #
# Build this fork from the local build context, retaining cached dependencies.
# ---------------------------------------------------------------------------- #
COPY . /app/ai-toolkit/

# Fail the image build on runtime drift or broken native-H3 training imports,
# before paying for a GPU Pod. Real CUDA forward/backward is checked on the Pod.
RUN python -c "import torch, torchvision, torchaudio; assert torch.__version__.split('+')[0] == '2.13.0', torch.__version__; assert torch.version.cuda.startswith('13.'), torch.version.cuda; from extensions_built_in.diffusion_models.minimax_h3.minimax_h3 import MinimaxH3Model; from toolkit.util.convrot_quant import ConvRotInt8Quantizer; print('Blackwell runtime and H3 imports OK:', torch.__version__, torch.version.cuda)"

# Build UI (re-runs on code change, but reuses the cached node_modules above).
# update_db runs first because it does `prisma generate`, which creates the
# @prisma/client types the TS build needs. In the old layout generate happened
# as a side effect of npm install seeing the schema; now the source arrives
# after npm ci, so run it explicitly before the build.
RUN cd /app/ai-toolkit/ui && \
    npm run update_db && \
    npm run build

# Expose port (assuming the application runs on port 3000)
EXPOSE 8675

WORKDIR /

COPY docker/start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]
