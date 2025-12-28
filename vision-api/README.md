# VisionGuide Pro API

Görme engellilere yönelik bulut tabanlı nesne tanıma API'si. AWS Free Tier için optimize edilmiş, 500 kelimelik Türkçe-İngilizce sözlük ile 3 farklı mod (Ev, Sokak, Market/Ofis) destekler.

## Özellikler

- 🚀 **Yüksek Performans**: YOLOv8n-world (Nano) modeli ile hızlı tespit
- 💾 **Bellek Verimli**: 400MB altında RAM kullanımı (AWS t2.micro uyumlu)
- 🌍 **500 Kelime**: Kapsamlı Türkçe-İngilizce nesne sözlüğü
- 🎯 **3 Mod**: Ev, Sokak, Market/Ofis modları
- 🔒 **Güvenli**: API key tabanlı kimlik doğrulama
- 🐳 **Docker Ready**: Tek komutla deploy

## Hızlı Başlangıç

### Docker ile Çalıştırma

```bash
# Docker imajını oluştur
docker build -t vision-api .

# Container'ı çalıştır
docker run -d \
  -p 8000:8000 \
  -e VISION_API_KEY=your-secret-api-key-here \
  --name vision-api \
  vision-api
```

### Yerel Geliştirme

```bash
# Bağımlılıkları yükle
pip install -r requirements.txt

# API'yi çalıştır
uvicorn main:app --host 0.0.0.0 --port 8000
```

API dokümantasyonu: http://localhost:8000/docs

## API Kullanımı

### Endpoint: `/detect`

**Method**: POST  
**Headers**: 
- `X-API-KEY`: API anahtarınız

**Request Body**:
```json
{
  "image": "base64_encoded_image_string",
  "mode": "E"  // E: Ev, S: Sokak, M: Market/Ofis
}
```

**Response**:
```json
{
  "success": true,
  "mode": "E",
  "mode_name": "Ev Modu",
  "detections": [
    {
      "label": "Sandalye",
      "label_en": "chair",
      "score": 0.95,
      "box": {
        "x1": 0.1,
        "y1": 0.2,
        "x2": 0.5,
        "y2": 0.8
      },
      "box_pixels": {
        "x1": 64,
        "y1": 96,
        "x2": 320,
        "y2": 384
      }
    }
  ],
  "image_width": 640,
  "image_height": 480,
  "processing_time_ms": 125.5
}
```

## Modlar

### E - Ev Modu
~170 nesne kategorisi: Mutfak gereçleri, mobilya, elektronik, vb.

### S - Sokak Modu
~170 nesne kategorisi: Trafik işaretleri, araçlar, binalar, vb.

### M - Market/Ofis Modu
~160 nesne kategorisi: Ofis malzemeleri, market ürünleri, teknolojik cihazlar, vb.

## Güvenlik

API key'i environment variable olarak ayarlayın:

```bash
export VISION_API_KEY=your-secret-api-key-here
```

## AWS Deployment

### EC2 Instance (t2.micro)

```bash
# EC2 instance'a bağlan
ssh -i your-key.pem ubuntu@your-ec2-ip

# Docker yükle
sudo apt-get update
sudo apt-get install docker.io -y
sudo usermod -aG docker ubuntu

# Projeyi klonla
git clone your-repo
cd Vision-webcam

# Docker imajını oluştur ve çalıştır
docker build -t vision-api .
docker run -d \
  -p 8000:8000 \
  -e VISION_API_KEY=your-secret-key \
  --restart unless-stopped \
  --name vision-api \
  vision-api
```

### Nginx Reverse Proxy (Opsiyonel)

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Performans

- **Model Boyutu**: ~6MB (yolov8n-world.pt)
- **RAM Kullanımı**: ~350-400MB (model + inference)
- **İşlem Süresi**: ~100-200ms (640x480 görüntü)
- **Eşzamanlı İstek**: 2 worker thread

## Hata Kodları

- `200`: Başarılı
- `400`: Geçersiz istek (bozuk görüntü, eksik parametre)
- `401`: Yetkilendirme hatası (geçersiz API key)
- `500`: Sunucu hatası
- `503`: Servis kullanılamıyor (model yüklenemedi)

## Test Senaryoları

API aşağıdaki durumları otomatik olarak işler:

✅ **Geçersiz Veri**: Boş base64, hatalı JSON, eksik mode  
✅ **Bellek Yönetimi**: Her istek sonrası otomatik temizlik  
✅ **Model Yükleme**: Retry mekanizması (3 deneme)  
✅ **Doğruluk Kontrolü**: Bounding box'lar görüntü sınırları içinde

## Android Entegrasyonu

Detaylı Android Kotlin entegrasyon rehberi için: [ANDROID_KOTLIN_INTEGRATION.md](ANDROID_KOTLIN_INTEGRATION.md)

## Lisans

MIT License

## İletişim

Sorularınız için issue açabilirsiniz.

