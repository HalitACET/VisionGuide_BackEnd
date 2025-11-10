package com.visionguide.forum_api.controller.dto;

import jakarta.validation.constraints.NotBlank;

// Android'den yeni bir yorum eklemek için gelen JSON.
public class CreateCommentRequest {

    @NotBlank(message = "Yorum içeriği boş olamaz")
    private String content;

    // 'authorId' burada da yok. Güvenlik katmanından alınacak.

    // ... (Getter/Setter)
    public String getContent() { return content; }
    public void setContent(String content) { this.content = content; }
}