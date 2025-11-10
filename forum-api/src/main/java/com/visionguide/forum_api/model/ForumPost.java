package com.visionguide.forum_api.model;

import jakarta.persistence.CascadeType;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Lob;
import jakarta.persistence.OneToMany;
import jakarta.persistence.Table;
import org.hibernate.annotations.BatchSize;
import org.hibernate.annotations.CreationTimestamp;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

@Entity
@Table(name = "forum_posts")
public class ForumPost {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false, length = 255)
    private String title;

    @Lob // "Large Object" - Bu alanın büyük metin (TEXT) içereceğini belirtir
    @Column(nullable = false, columnDefinition = "TEXT")
    private String content;

    // Cognito'dan (veya güvenlik bağlamından) gelecek olan kullanıcının kimliği.
    @Column(nullable = false, updatable = false)
    private String authorId;

    @CreationTimestamp
    @Column(nullable = false, updatable = false)
    private Instant createdAt;

    // --- Erişilebilirlik İsteri ---
    @Column(nullable = true)
    private String imageUrl;

    @Column(nullable = true)
    private String imageAltText; // Resim varsa bu alan dolu olmalı

    // --- İlişki (Relationship) ---
    // Bir gönderinin birden fazla yorumu olabilir.
    // "mappedBy": Comment sınıfındaki 'post' alanı ile ilişkiyi yönetir.
    // "cascade": Bir gönderi silinirse, ona bağlı tüm yorumları da siler.
    @BatchSize(size = 50)
    @OneToMany(mappedBy = "post", cascade = CascadeType.ALL, orphanRemoval = true)
    @jakarta.persistence.OrderBy("createdAt ASC")
    private List<Comment> comments = new ArrayList<>();

    // --- Getter/Setter (ve Constructor) ---
    public ForumPost() {}

    // ... (Gerekli tüm getter ve setter'ları buraya ekleyin)
    // (IDE'niz (IntelliJ) bunları sizin için otomatik oluşturabilir)

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getTitle() { return title; }
    public void setTitle(String title) { this.title = title; }
    public String getContent() { return content; }
    public void setContent(String content) { this.content = content; }
    public String getAuthorId() { return authorId; }
    public void setAuthorId(String authorId) { this.authorId = authorId; }
    public Instant getCreatedAt() { return createdAt; }
    public void setCreatedAt(Instant createdAt) { this.createdAt = createdAt; }
    public String getImageUrl() { return imageUrl; }
    public void setImageUrl(String imageUrl) { this.imageUrl = imageUrl; }
    public String getImageAltText() { return imageAltText; }
    public void setImageAltText(String imageAltText) { this.imageAltText = imageAltText; }
    public List<Comment> getComments() { return comments; }
    public void setComments(List<Comment> comments) { this.comments = comments; }
}
