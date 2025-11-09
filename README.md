````markdown
# 🧠 VisionGuide AI  
**FastAPI Tabanlı Görüntü Tanıma ve Nesne Algılama API'si**

Bu proje, TensorFlow Hub tabanlı bir makine öğrenimi modelini (örneğin *EfficientDet D0*) kullanarak görüntülerdeki nesneleri tespit eden bir **FastAPI** servisini içerir.  
Android uygulamanız bu API'ye Base64 formatında görüntü gönderir, model nesneleri algılar ve JSON formatında koordinatlar, etiketler ve olasılık değerlerini döndürür.

---

## 🚀 Özellikler
✅ FastAPI tabanlı yüksek performanslı REST API  
✅ TensorFlow Hub ile nesne tespiti (EfficientDet)  
✅ Base64 veya Data URI formatında görüntü girişi desteği  
✅ Piksel koordinatlı bounding box sonuçları  
✅ Threshold ayarlanabilirliği  
✅ Basit API Key doğrulaması (isteğe bağlı)  
✅ Docker ve AWS Fargate uyumlu yapı  

---

## 📂 Proje Dizini
```bash
VisionGuide-LocalTest/
├─ main.py                 # FastAPI uygulaması
├─ requirements.txt        # Gereksinim dosyası
├─ Dockerfile              # Docker imajı oluşturmak için
├─ venv/                   # Sanal ortam (lokalde)
└─ README.md               # Bu dosya
````

---

## 🧩 Gereksinimler

| Yazılım    | Versiyon       |
| ---------- | -------------- |
| Python     | 3.9 veya üzeri |
| TensorFlow | 2.x            |
| FastAPI    | 0.95+          |
| Uvicorn    | 0.20+          |

---

## ⚙️ Kurulum (Yerel)

1. **Projeyi klonla**

   ```bash
   git clone https://github.com/<kullanıcı_adı>/VisionGuide-AI.git
   cd VisionGuide-AI
   ```

2. **Sanal ortam oluştur ve aktif et**

   ```bash
   python -m venv venv
   # macOS / Linux
   source venv/bin/activate
   # Windows PowerShell
   .\venv\Scripts\Activate.ps1
   ```

3. **Gereksinimleri yükle**

   ```bash
   pip install -r requirements.txt
   ```

4. **(İsteğe bağlı) Ortam değişkenlerini ayarla**

   ```bash
   export VISION_API_KEY="benim_gizli_keyim"
   export VISION_JWT_SECRET="jwt_secret"
   ```

5. **Sunucuyu başlat**

   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

Sunucu çalıştığında şu şekilde bir çıktı görürsünüz:

```
Uvicorn running on http://0.0.0.0:8000
Application startup complete.
```

---

## 🌐 API Uç Noktaları

### 🔹 1. Health Check

**GET /health**

```bash
curl http://localhost:8000/health
```

**Yanıt:**

```json
{"status": "ok", "model_loaded": true}
```

---

### 🔹 2. Nesne Tespiti

**POST /detect**

**İstek Gövdesi (JSON):**

```json
{
  "image": "<BASE64_STRING_OR_DATA_URI>"
}
```

**Header (isteğe bağlı):**

```
X-API-KEY: benim_gizli_keyim
```

**Örnek Curl:**

```bash
b64=$(base64 -w 0 test.jpg)
curl -X POST "http://localhost:8000/detect?threshold=0.5" \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: benim_gizli_keyim" \
  -d "{\"image\":\"$b64\"}"
```

**Yanıt:**

```json
{
  "detections": [
    {
      "label": "bardak",
      "score": 0.87,
      "box": [0.12, 0.34, 0.45, 0.60],
      "box_pixels": [34, 50, 150, 220]
    }
  ]
}
```

---

## 📱 Android Entegrasyonu (Özet)

* Görüntüyü `Bitmap` olarak al → JPEG/PNG’e çevir → Base64 encode et (`Base64.NO_WRAP`)
* Retrofit veya OkHttp ile `POST /detect` çağrısı gönder
* `box_pixels` değerleriyle overlay çiz

---

## 🐳 Docker ile Çalıştırma

### 1. İmaj oluştur

```bash
docker build -t visionguide:latest .
```

### 2. Konteyner başlat

```bash
docker run -p 8000:8000 \
  -e VISION_API_KEY=benim_gizli_keyim \
  visionguide:latest
```

---

## ☁️ AWS Dağıtımı (Özet)

### Seçenek 1: **AWS ECS Fargate**

1. Docker imajını AWS ECR’e push et
2. ECS Service oluştur (CPU-only veya GPU instance)
3. `VISION_API_KEY` gibi değerleri **AWS Secrets Manager** veya **SSM Parameter Store** ile yönetin
4. ALB (Application Load Balancer) + HTTPS (ACM sertifikası) ekleyin

### Seçenek 2: **EC2 GPU Instance**

* TensorFlow GPU sürümü kullanın (`tensorflow-gpu`)
* NVIDIA sürücüleri ve `nvidia-docker` gereklidir

---

## 🧠 Sık Karşılaşılan Sorunlar

| Sorun              | Çözüm                                                             |
| ------------------ | ----------------------------------------------------------------- |
| `MemoryError`      | TensorFlow model boyutunu küçültün veya RAM artırın               |
| `libGL` hatası     | Dockerfile’a `libsm6 libxext6` paketlerini ekleyin                |
| Yavaş yanıt        | `--reload` devre dışı bırakın, GPU veya daha küçük model deneyin  |
| `Model not loaded` | `GET /health` çağrısı ile doğrulayın, model yüklenememiş olabilir |

---

## 🧾 Önerilen `requirements.txt`

```
fastapi
uvicorn[standard]
tensorflow
tensorflow-hub
pillow
opencv-python-headless
numpy
python-multipart
pyjwt
```

---

## 🧱 Örnek `Dockerfile`

```dockerfile
FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y libsm6 libxext6 ffmpeg && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 🛡️ Güvenlik Notları

* Üretim ortamında HTTPS (AWS ACM + ALB) kullanın
* API anahtarlarını açık şekilde tutmayın, **Secrets Manager** veya `.env` dosyası kullanın
* Gerekirse JWT veya OAuth 2.0 ile kimlik doğrulama ekleyin

---

## 💬 Katkıda Bulunma

1. Bu repo'yu forklayın
2. Yeni bir branch oluşturun (`feature/yenilik`)
3. Değişikliklerinizi commit edin
4. Pull request gönderin 🎯

---

> 💡 **Not:** README, hem yerel geliştirme hem de AWS dağıtım sürecini kapsayacak şekilde optimize edilmiştir.
> Docker, ECS ve Android entegrasyon detayları için ek dokümantasyon yakında eklenecektir.
