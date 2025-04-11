# Start from the NVIDIA CUDA runtime image (includes cuDNN)
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

# Set non-interactive frontend for package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install essential system packages and uv
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates git && \
    rm -rf /var/lib/apt/lists/* && \
    curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH (default install location for root)
ENV PATH="/root/.cargo/bin:${PATH}"

# Set the working directory
WORKDIR /app

# Copy only pyproject.toml first to leverage Docker cache
COPY pyproject.toml .

# Install Python and dependencies using uv
# --system-site-packages installs them into the Python managed by uv
RUN uv sync --system-site-packages

# Copy the rest of the application code
COPY . .

# (Optional: Define a default command if needed, e.g., CMD ["python", "main.py"])