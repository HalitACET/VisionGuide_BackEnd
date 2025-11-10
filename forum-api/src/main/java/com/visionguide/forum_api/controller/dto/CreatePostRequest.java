package com.visionguide.forum_api.controller.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;

// Android'den yeni bir gönderi oluşturmak için gelen JSON.
public class CreatePostRequest {

    @NotBlank(message = "Başlık boş olamaz")
    @Size(min = 3, max = 100)
    private String title;

    @NotBlank(message = "İçerik boş olamaz")
    private String content;

    // GÜVENLİK NOTU: 'authorId' alanı buradan KASTEN kaldırıldı.
    // Kullanıcının "Ben başkasıyım" diyerek gönderi atmasını engellemeliyiz.
    // 'authorId', JSON'dan değil, güvenlik katmanından (JWT token) alınmalıdır.

    // Erişilebilirlik alanları opsiyoneldir
    private String imageUrl;
    private String imageAltText;

    // ... (Getter/Setter'ları buraya ekleyin)
    public String getTitle() { return title; }
    public void setTitle(String title) { this.title = title; }
    public String getContent() { return content; }
    public void setContent(String content) { this.content = content; }
    public String getImageUrl() { return imageUrl; }
    public void setImageUrl(String imageUrl) { this.imageUrl = imageUrl; }
    public String getImageAltText() { return imageAltText; }
    public void setImageAltText(String imageAltText) { this.imageAltText = imageAltText; }
}