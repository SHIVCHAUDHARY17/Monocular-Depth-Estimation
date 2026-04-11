FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU PyTorch first (lighter for Docker)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt .
RUN pip install opencv-python-headless numpy PyYAML timm transformers onnx onnxruntime pillow ultralytics

# Copy project files
COPY src/ ./src/
COPY configs/ ./configs/
COPY run_depth.py .
COPY conftest.py .

# Default command runs depth on image
CMD ["python", "run_depth.py", "--mode", "image"]