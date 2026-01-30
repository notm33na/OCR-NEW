# Bilingual OCR API - Railway / Docker (Python 3.9, CPU-only)
FROM python:3.9-slim

# Tesseract with Arabic and English language packs
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-ara \
    tesseract-ocr-eng \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps from requirements (no PyTorch yet)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# PyTorch CPU (Linux wheels)
RUN pip install --no-cache-dir torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# Application code
COPY *.py .
COPY *.md .

# Directories for models and I/O (mount or copy at runtime if needed)
RUN mkdir -p saved_models/UTRNet-Large inputs outputs

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
