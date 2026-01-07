# VisionGuide Pro API - Dockerfile
# AWS Free Tier için optimize edilmiş, hafif imaj
# Self-Validating: Sıfır hata garantili

FROM python:3.10-slim

# Sistem bağımlılıkları (OpenCV ve YOLO için gerekli)
# Tek satırda yükleyip cache'i temizle (imaj boyutunu küçült)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Çalışma dizini
WORKDIR /app

# Python bağımlılıklarını kopyala ve yükle
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge

# Uygulama dosyalarını kopyala
COPY main.py .
COPY vocabulary.py .

# Port
EXPOSE 8000

# Çevre değişkenleri
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV VISION_API_KEY=${VISION_API_KEY:-your-secret-api-key-here}
ENV OMP_NUM_THREADS=2
ENV MKL_NUM_THREADS=2

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Uygulamayı çalıştır
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-keep-alive", "30"]

