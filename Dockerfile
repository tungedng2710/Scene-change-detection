# syntax=docker/dockerfile:1
###############################################################################
# 1. CUDA + Ubuntu base (already contains the libraries your GPU code needs)
###############################################################################
FROM nvidia/cuda:12.5.0-runtime-ubuntu22.04 AS base
ENV DEBIAN_FRONTEND=noninteractive
LABEL maintainer="you@example.com"

###############################################################################
# 2. OS packages for Python 3.11 + build tools
###############################################################################
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-distutils \
        build-essential git curl \
        libgl1 libglib2.0-0 libsm6 && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python && \
    apt-get clean && rm -rf /var/lib/apt/lists/*


###############################################################################
# 3. Isolate Python deps to maximise Docker cache hits
###############################################################################
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Optional: grab a CUDA wheel of PyTorch if you need it
# RUN pip install --no-cache-dir torch==2.3.0+cu118 torchvision==0.18.0+cu118 \
#     --extra-index-url https://download.pytorch.org/whl/cu118

###############################################################################
# 4. Git-based packages
###############################################################################
RUN pip install --no-cache-dir -U --no-deps --force-reinstall \
      git+https://github.com/Z-Zheng/pytorch-change-models \
      git+https://github.com/Z-Zheng/ever.git

###############################################################################
# 5. Copy the rest of your source *after* deps are installed
###############################################################################
COPY . .
###############################################################################
# 6. Entrypoint
###############################################################################
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7866"]