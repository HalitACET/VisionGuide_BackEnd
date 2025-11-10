# Forum API

## Proje Tanımı
Forum API, topluluk kullanıcılarının gönderi ve yorum paylaşabildiği, REST mimarisi ile tasarlanmış ölçeklenebilir bir içerik yönetim servisidir. Uygulama gönderileri sayfalı olarak listeleyebilir, tekil gönderileri yorumlarıyla birlikte dönebilir ve üretim ortamında JWT tabanlı güvenliği destekler. Performans odaklı sorgular, DTO katmanı ve bellek dostu veri eşlemeleri sayesinde düşük gecikme ve kaynak kullanımı hedeflenmiştir.

## Kurulum ve Gereksinimler
- İşletim sistemi: macOS, Linux veya Windows 10+
- Java Geliştirme Kiti (JDK) 17
- Gradle Wrapper (repo ile birlikte gelir, ayrıca kurulum gerekmez)
- Docker (opsiyonel, PostgreSQL’i hızlıca ayağa kaldırmak için)
- PostgreSQL 14+ (varsayılan kurulum) veya uyumlu bir sunucu
- İnternet erişimi (bağımlılık indirmeleri için)

Güvenlik katmanını etkinleştirmek için aşağıdaki konfigürasyonlardan en az birini sağlamanız gerekir:
- `spring.security.oauth2.resourceserver.jwt.issuer-uri`
- `spring.security.oauth2.resourceserver.jwt.jwk-set-uri`

Bu değerler set edilmediğinde API geliştirme amaçlı açık modda çalışır, tüm istekleri kimlik doğrulaması olmadan kabul eder.

## Kurulum Talimatları
1. Depoyu klonlayın:
   ```bash
   git clone https://github.com/<kullanici-adiniz>/forum-api.git
   cd forum-api
   ```
2. (Opsiyonel) PostgreSQL konteyneri çalıştırın:
   ```bash
   docker run --name forum-postgres \
     -e POSTGRES_USER=postgres \
     -e POSTGRES_PASSWORD=postgres \
     -e POSTGRES_DB=visionguide_forum_db \
     -p 5432:5432 \
     -d postgres:15
   ```
3. Uygulama yapılandırmasını güncelleyin:
   - `src/main/resources/application-local.properties` dosyasındaki `spring.datasource.*` değerlerini ortamınıza göre düzenleyin veya
   - Sistem değişkenleri / `application.properties` üzerinden gerekli bağlantı bilgilerini sağlayın.
4. Bağımlılıkları indirin ve projeyi derleyin:
   ```bash
   ./gradlew build
   ```

## Kullanım Talimatları
### Uygulamayı Başlatma
Yerel ortamda Spring Boot uygulamasını çalıştırın:
```bash
./gradlew bootRun --args='--spring.profiles.active=local'
```
Varsayılan olarak servis `http://localhost:8080` adresinde ayağa kalkar.

### Başlıca Uç Noktalar
- `GET /api/forum/posts?page=0&size=10`
  - Gönderileri sayfalı olarak listeler.
  - Dönen `PostSummaryResponseDTO` nesneleri: `id`, `title`, `authorId`, `createdAt`, `excerpt`, `imageUrl`.
  - Sistem, `size` değerini 1 ile 50 arasına otomatik olarak sınırlar, sıralama tarih alanı üzerinden yapılır.

- `GET /api/forum/posts/{id}`
  - İlgili gönderiyi tüm ayrıntıları ve kronolojik yorum listesiyle (`CommentResponseDTO`) döner.
  - `PostDetailResponseDTO` içinde gönderi başlığı, içerik, görsel bilgisi ve yorumlar yer alır.

Kimlik doğrulama konfigüre edildiğinde, POST/PUT/DELETE istekleri JWT taşımayan istemciler için 401/403 dönecek şekilde otomatik olarak korunur.

### Örnek İstek
```bash
curl "http://localhost:8080/api/forum/posts?page=0&size=5"
```

## Testler
Projede, H2 bellek içi veritabanında çalışan Spring Boot tabanlı entegrasyon testleri yer alır. Testler, DTO eşlemelerini, sıralama ve özet verisi üretimini doğrular.

- Tüm testleri çalıştırmak:
  ```bash
  ./gradlew test
  ```
- Başarı kriterleri:
  - `ForumApiApplicationTests` Spring context’inin sorunsuz yüklendiğini doğrular.
  - `ForumServiceIntegrationTest` sayfalama sonuçlarını, özet içeriklerin kesilmesini ve yorumların kronolojik olarak döndüğünü test eder.
  - Testler yeşil tamamlandığında raporlar `build/reports/tests/test/index.html` altında üretilecektir.

## Katkı Sağlama
Projeye katkıda bulunmak isterseniz:
1. Yeni bir dal açın:
   ```bash
   git checkout -b ozellik/kisa-aciklama
   ```
2. Yapacağınız değişiklikleri küçük, odaklı commit’ler halinde ekleyin.
3. `./gradlew test` komutunu çalıştırarak regresyonları önleyin.
4. Açıklayıcı bir Pull Request hazırlayarak değişikliklerinizi paylaşın.

Katkı süreçlerinde kod biçimlendirme, testlerin çalıştırılması ve güvenlik yapılandırmalarının dokümante edilmesi beklenir. Sorularınız için GitHub Issues üzerinden başlık açabilirsiniz.

