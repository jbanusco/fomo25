# Base: NGC PyTorch image (matches CSCS Alps ngc-pytorch+25.06.sqsh)
# Includes: CUDA, cuDNN, Python, PyTorch, torchvision — all pre-configured
FROM nvcr.io/nvidia/pytorch:25.06-py3

# Maintainer
LABEL maintainer="Jaume Banus <jaume.banus-cobo@chuv.ch>"

RUN chmod 1777 /tmp && apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FOMO25_WEIGHTS_PATH=/usr/src/weights/fomo25_mmunetvae_pretrained.ckpt

# Set work directory and copy project
WORKDIR /usr/src/
COPY . /usr/src/

# Install project dependencies (PyTorch already in base image)
# Satisfy PyTorch's 1.x requirement while keeping dependencies happy
RUN pip install --no-cache-dir "numpy>=1.23,<2.0" && \
    pip install --no-cache-dir -e ".[dev,test]"
#RUN pip install --no-cache-dir -e ".[dev,test]"

# Verify PyTorch/CUDA installation
RUN python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}'); print(f'NCCL available: {torch.distributed.is_nccl_available()}')"

# Verify weights exist (non-fatal warning if missing)
RUN test -f /usr/src/weights/fomo25_mmunetvae_pretrained.ckpt || echo "Warning: Pretrained weights not found"
