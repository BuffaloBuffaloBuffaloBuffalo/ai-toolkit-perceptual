FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

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
RUN pip install --no-cache-dir torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128 --break-system-packages

WORKDIR /app/ai-toolkit

# Install Python deps first so source changes don't bust the pip cache layer
COPY requirements.txt /app/ai-toolkit/requirements.txt
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt && \
    pip install setuptools==69.5.1 --no-cache-dir --break-system-packages

# insightface hard-depends on the CPU-only `onnxruntime`, which installs into the
# same package dir as `onnxruntime-gpu` and shadows its native library — leaving
# onnxruntime with no CUDAExecutionProvider, so InsightFace face-embedding caching
# silently runs on CPU (~4s/image vs ~12ms on an H100; days vs minutes on a large
# dataset). Force the GPU build's libraries to win, then register torch's bundled
# CUDA/cuDNN libs (the nvidia-*-cu12 wheels under nvidia/*/lib, which aren't in
# ldconfig) so the CUDA EP can load libcudnn.so.9 at runtime. The final assert
# fails the build if the GPU provider isn't active.
RUN pip install --no-cache-dir --break-system-packages --force-reinstall --no-deps onnxruntime-gpu==1.26.0 && \
    python -c "import nvidia, glob, os; print('\n'.join(sorted({d for b in nvidia.__path__ for d in glob.glob(os.path.join(b, '*', 'lib'))})))" > /etc/ld.so.conf.d/nvidia-pip-libs.conf && \
    ldconfig && \
    python -c "import onnxruntime as o; p=o.get_available_providers(); assert 'CUDAExecutionProvider' in p, p; print('onnxruntime providers OK:', p)"

# Install Node deps (still cache-friendly if package files don't change)
COPY ui/package.json /app/ai-toolkit/ui/
COPY ui/package-lock.json /app/ai-toolkit/ui/
WORKDIR /app/ai-toolkit/ui
RUN npm install

# Now copy the full source tree (fork code) and build the UI.
# update_db runs `prisma generate` to produce @prisma/client types — must run
# before `next build` / `tsc` which depend on those types.
COPY . /app/ai-toolkit/
RUN npm run update_db && npm run build

# Expose port (assuming the application runs on port 3000)
EXPOSE 8675

WORKDIR /

COPY docker/start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]