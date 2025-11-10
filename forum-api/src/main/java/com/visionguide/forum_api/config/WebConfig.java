package com.visionguide.forum_api.config;

import org.springframework.context.annotation.Configuration;
import org.springframework.lang.NonNull;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

@Configuration
public class WebConfig implements WebMvcConfigurer {

    // 'application.properties' dosyasından alınacak olan izinli kaynakların listesi
    // Örnek: "http://localhost:3000,https://visionguide-app.com"
    // Şimdilik "*" (herkese izin ver) olarak ayarlayabiliriz.
    // private String[] allowedOrigins = {"*"};

    @Override
    public void addCorsMappings(@NonNull CorsRegistry registry) {
        registry.addMapping("/api/**") // API'nizdeki hangi yollar için geçerli? (Tümü için /api/**)

                // Geliştirme (development) için en esnek ayar:
                .allowedOrigins("*") // Herkese izin ver (güvensiz, sadece geliştirme için!)

                // Üretim (production) için daha güvenli ayar:
                // .allowedOrigins(allowedOrigins)

                .allowedMethods("GET", "POST", "PUT", "DELETE", "OPTIONS") // İzin verilen HTTP metodları
                .allowedHeaders("*") // Gelen tüm başlıklara (header) izin ver
                .allowCredentials(false) // Kimlik bilgisi (cookie vb.) gerektirmez
                .maxAge(3600); // Bu ayarların ne kadar süre önbellekte tutulacağı (saniye)
    }
}