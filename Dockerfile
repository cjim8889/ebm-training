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

# Copy only pyproject.toml first to leverage Docker cache
# Note: This assumes pyproject.toml is sufficient for `uv sync` before cloning the full repo.
# If the full repo is needed for `uv sync`, this COPY and the RUN uv sync might need adjustment
# depending on the project structure in the repo.
COPY pyproject.toml .

# Install Python and dependencies using uv
# --system-site-packages installs them into the Python managed by uv
RUN uv sync

# Clone the repository, switch branch, and pull LFS files
RUN git clone https://github.com/cjim8889/ebm-training . && \
    git checkout jax-corrected && \
    git lfs install && \
    git lfs pull

# (Optional: Define a default command if needed, e.g., CMD ["python", "main.py"])