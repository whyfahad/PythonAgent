FROM python:3.10-slim

# Set environment variables
ENV TORCH_HOME=/root/.cache/torch

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch (adjust the CUDA version as needed)
RUN pip install torch==2.6.0+cu124 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu124

# Install Fairseq2 compatible with the installed PyTorch
RUN pip install fairseq2 --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/pt2.6.0/cu124

# Install SONAR
RUN pip install sonar-space

# Set the working directory
WORKDIR /app

# Copy your application code
COPY . /app
