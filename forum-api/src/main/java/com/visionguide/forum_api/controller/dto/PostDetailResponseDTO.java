package com.visionguide.forum_api.controller.dto;

import java.time.Instant;
import java.util.List;

/**
 * Tek bir forum gönderisinin detaylı görünümü için DTO.
 * Gönderi içeriği + resim bilgisi + yorum listesini içerir.
 */
public class PostDetailResponseDTO {
    private Long id;
    private String title;
    private String content;
    private String authorId;
    private Instant createdAt;
    private String imageUrl;
    private String imageAltText;
    private List<CommentResponseDTO> comments; // Yorumlar detay görünümünde yer alır.

    // --- Constructors ---
    public PostDetailResponseDTO() {
    }

    public PostDetailResponseDTO(Long id, String title, String content, String authorId,
                                 Instant createdAt, String imageUrl, String imageAltText,
                                 List<CommentResponseDTO> comments) {
        this.id = id;
        this.title = title;
        this.content = content;
        this.authorId = authorId;
        this.createdAt = createdAt;
        this.imageUrl = imageUrl;
        this.imageAltText = imageAltText;
        this.comments = comments;
    }

    // --- Getters & Setters ---
    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public String getContent() {
        return content;
    }

    public void setContent(String content) {
        this.content = content;
    }

    public String getAuthorId() {
        return authorId;
    }

    public void setAuthorId(String authorId) {
        this.authorId = authorId;
    }

    public Instant getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(Instant createdAt) {
        this.createdAt = createdAt;
    }

    public String getImageUrl() {
        return imageUrl;
    }

    public void setImageUrl(String imageUrl) {
        this.imageUrl = imageUrl;
    }

    public String getImageAltText() {
        return imageAltText;
    }

    public void setImageAltText(String imageAltText) {
        this.imageAltText = imageAltText;
    }

    public List<CommentResponseDTO> getComments() {
        return comments;
    }

    public void setComments(List<CommentResponseDTO> comments) {
        this.comments = comments;
    }
}
