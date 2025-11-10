package com.visionguide.forum_api.controller.dto;

import java.time.Instant;

/**
 * Forum gönderilerinin özet görünümü (yorumlar hariç) için DTO.
 * Listeleme işlemlerinde N+1 sorununu önlemek amacıyla kullanılır.
 */
public class PostSummaryResponseDTO {
    private Long id;
    private String title;
    private String authorId;
    private Instant createdAt;
    private String excerpt;
    private String imageUrl;

    // --- Constructors ---
    public PostSummaryResponseDTO() {
    }

    public PostSummaryResponseDTO(Long id, String title, String authorId, Instant createdAt, String excerpt, String imageUrl) {
        this.id = id;
        this.title = title;
        this.authorId = authorId;
        this.createdAt = createdAt;
        this.excerpt = excerpt;
        this.imageUrl = imageUrl;
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

    public String getExcerpt() {
        return excerpt;
    }

    public void setExcerpt(String excerpt) {
        this.excerpt = excerpt;
    }

    public String getImageUrl() {
        return imageUrl;
    }

    public void setImageUrl(String imageUrl) {
        this.imageUrl = imageUrl;
    }
}
