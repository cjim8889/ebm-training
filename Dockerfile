# Start from the NVIDIA CUDA runtime image (includes cuDNN)
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

# Set non-interactive frontend for package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install essential system packages, git-lfs, and uv
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates git && \
    # Install git-lfs using packagecloud script
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install -y git-lfs && \
    # Clean up apt lists
    rm -rf /var/lib/apt/lists/* && \
    # Install uv
    curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH (default install location for root)
ENV PATH="/root/.local/bin:${PATH}"

# Set the working directory
WORKDIR /app

# Clone the repository and switch branch, skipping LFS smudge during clone
# RUN git clone https://github.com/cjim8889/ebm-training.git . && \
#     git checkout jax-corrected

# Install LFS hooks and explicitly pull LFS files
RUN git lfs install --skip-smudge && \
    git clone https://github.com/cjim8889/ebm-training.git . && \
    git checkout jax-corrected

# Install Python and dependencies using uv (uses pyproject.toml from cloned repo)
# --system-site-packages installs them into the Python managed by uv
RUN uv sync

# (Optional: Define a default command if needed, e.g., CMD ["python", "main.py"])