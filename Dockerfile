FROM python:3.9-slim

# Install system dependencies in one layer (includes git for cloning repos)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-ara \
    libgl1 \
    libglib2.0-0 \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first for better layer caching
COPY requirements.txt .

# Install dependencies with pip cache (includes gdown for Google Drive downloads)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gdown

# Clone gitignored repos that contain model weights (not in git, must fetch at build time)
# 1. urdu-text-detection: contains yolov8m_UrduDoc.pt (YOLOv8 Urdu text line detector)
RUN git lfs install && \
    git clone --depth 1 https://github.com/abdur75648/urdu-text-detection.git urdu-text-detection

# 2. UTRNet: recognition model code (modules/, read.py, etc.)
RUN git clone --depth 1 https://github.com/abdur75648/UTRNet-High-Resolution-Urdu-Text-Recognition.git \
    UTRNet-High-Resolution-Urdu-Text-Recognition

# 3. Download UTRNet-Large pretrained weights from Google Drive
RUN mkdir -p saved_models/UTRNet-Large && \
    gdown "1xXG7vsSePBw4vtapIEdPWEZ-qrbR9Q9K" -O saved_models/UTRNet-Large/best_norm_ED.pth

# Copy application code
COPY *.py ./

# Copy TrOCR model (tracked in git as submodule/directory)
COPY trocr-base-handwritten/ ./trocr-base-handwritten/

# Create directories
RUN mkdir -p inputs outputs temp_uploads

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run the production app (app.py has model preloading, ThreadPoolExecutor)
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
