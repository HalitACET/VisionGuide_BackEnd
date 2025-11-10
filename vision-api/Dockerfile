# ======================================================
# VisionGuide AI - FastAPI Object Detection API
# Production Dockerfile (CPU)
# ======================================================

FROM python:3.10-slim

# ----------------------------
# 1. Sistem Gereksinimleri
# ----------------------------
WORKDIR /app
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 ffmpeg wget curl \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------
# 2. Python Gereksinimleri
# ----------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ----------------------------
# 3. Uygulama Dosyalarını Kopyala
# ----------------------------
COPY . .

# ----------------------------
# 4. Ortam Değişkenleri
# ----------------------------
ENV MODEL_URL=https://tfhub.dev/tensorflow/efficientdet/d0/1
ENV MODEL_INPUT_SIZE=512
ENV VISION_API_KEY="default_key"
ENV PYTHONUNBUFFERED=1

# ----------------------------
# 5. Başlangıç Komutu
# ----------------------------
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

