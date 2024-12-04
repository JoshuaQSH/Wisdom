# Base image with CUDA support
FROM nvidia/cuda:12.0.1-base-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.local/bin:$PATH"

# Install required packages
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch (matching your version)
RUN pip3 install torch==2.0.1+cu117 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# Install other dependencies
COPY requirements.txt /tmp/
RUN pip3 install -r /tmp/requirements.txt

# Clone your repository and switch to the branch
RUN git clone -b captum https://github.com/JoshuaQSH/torch-deepimportance.git /app
WORKDIR /app

# Entry point (e.g., interactive shell)
CMD ["bash"]