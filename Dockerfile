# Base image with GPU support (CUDA 12.2)
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11 and system dependencies
RUN apt-get update && apt-get install --no-install-recommends -y \
    python3.11 python3.11-dev curl \
    wget unzip git ca-certificates \
    zstd libxrender1 libxxf86vm1 libxfixes3 libxi6 libxkbcommon0 \
    libgl1 libglib2.0-0 \
 && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Install pip explicitly
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Set working directory
WORKDIR /app

# Copy requirements file and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir \
    opencv-python \
    accelerate \
    trimesh \
    open-clip-torch \
    imageio && \
    pip install --no-cache-dir git+https://github.com/openai/CLIP.git && \
    python3 -c "import trimesh, open_clip, imageio; print('Dependency check OK: trimesh/open_clip/imageio')"

# Download and extract Blender dataset
RUN mkdir -p /blender && \
    cd /blender && \
    wget https://huggingface.co/datasets/tiange/Cap3D/resolve/main/misc/blender.zip && \
    unzip blender.zip && \
    rm blender.zip

# Stage-5 grasping demo (grasping/): PyBullet physics sim + rtree (trimesh ray/AABB).
# Late layer so it doesn't invalidate the heavy base layers on rebuild.
RUN pip install --no-cache-dir pybullet rtree

# Default command: keep the container alive; actual work is started via
# `docker compose run oscar python3 ...`
CMD ["sleep", "infinity"]


