package com.visionguide.forum_api.model;

import com.fasterxml.jackson.annotation.JsonIgnore;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.FetchType;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.Lob;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.Table;
import org.hibernate.annotations.CreationTimestamp;
import java.time.Instant;

@Entity
@Table(name = "comments")
public class Comment {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Lob
    @Column(nullable = false, columnDefinition = "TEXT")
    private String content;

    @Column(nullable = false, updatable = false)
    private String authorId; // Yorumu yapan kullanıcının kimliği

    @CreationTimestamp
    @Column(nullable = false, updatable = false)
    private Instant createdAt;

    // --- İlişki (Relationship) ---
    // Bir yorum, sadece bir gönderiye aittir.
    @ManyToOne(fetch = FetchType.LAZY) // Yorumu çekerken gönderiyi (Post) tekrar çekme
    @JoinColumn(name = "post_id", nullable = false)
    @JsonIgnore // Bu, JSON'a dönüştürülürken sonsuz döngüyü engeller
    private ForumPost post;

    // --- Getter/Setter (ve Constructor) ---
    public Comment() {}

    // ... (Gerekli tüm getter ve setter'ları buraya ekleyin)

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getContent() { return content; }
    public void setContent(String content) { this.content = content; }
    public String getAuthorId() { return authorId; }
    public void setAuthorId(String authorId) { this.authorId = authorId; }
    public Instant getCreatedAt() { return createdAt; }
    public void setCreatedAt(Instant createdAt) { this.createdAt = createdAt; }
    public ForumPost getPost() { return post; }
    public void setPost(ForumPost post) { this.post = post; }
}