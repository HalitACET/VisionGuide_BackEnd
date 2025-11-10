# VisionGuide Backend Monorepo

Bu depo, VisionGuide projesinin tüm backend servislerini içerir.

## Projeler

Bu monorepo iki ana servisten oluşmaktadır:

### 1. Yapay Zeka Servisi (`/vision-api`)

* **Teknoloji:** Python, FastAPI, TensorFlow
* **Amaç:** Android uygulamasından gelen görüntülere nesne tespiti (object detection) uygular ve sonuçları JSON olarak döndürür.
* **Detaylar:** Bu servisin nasıl çalıştırılacağına dair detaylı bilgi için lütfen `vision-api/README.md` dosyasına bakın.

### 2. Forum Servisi (`/forum-api`)

* **Teknoloji:** Java, Spring Boot, PostgreSQL
* **Amaç:** Kullanıcıların gönderi (post) ve yorum (comment) oluşturabildiği "Sürekli Etkileşim" (Topluluk Forumu) API'sini sağlar.
* **Detaylar:** Bu servisin nasıl çalıştırılacağına dair detaylı bilgi için (eğer varsa) `forum-api/README.md` dosyasına bakın.
