package com.visionguide.forum_api.controller.dto;

import java.time.Instant;

/**
 * Yorumları Android istemcisine DTO olarak döndürmek için kullanılır.
 */
public class CommentResponseDTO {
    private Long id;
    private String content;
    private String authorId;
    private Instant createdAt;

    // --- Constructors ---
    public CommentResponseDTO() {
    }

    public CommentResponseDTO(Long id, String content, String authorId, Instant createdAt) {
        this.id = id;
        this.content = content;
        this.authorId = authorId;
        this.createdAt = createdAt;
    }

    // --- Getters & Setters ---
    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
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
}
