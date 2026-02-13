FROM python:3.9-slim

# Install system dependencies in one layer (curl for downloading YOLOv8 text model)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-ara \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only necessary files
COPY requirements.txt .

# Install dependencies with pip cache
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY *.py ./

# Download YOLOv8 UrduDoc text-detection model (region detection works on Railway; same as local)
# Source: https://github.com/abdur75648/urdu-text-detection/releases
RUN mkdir -p urdu-text-detection saved_models inputs outputs temp_uploads && \
    curl -sL -o urdu-text-detection/yolov8m_UrduDoc.pt \
    "https://github.com/abdur75648/urdu-text-detection/releases/download/v1.0.0/yolov8m_UrduDoc.pt"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run the application
CMD uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
