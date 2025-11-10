package com.visionguide.forum_api.repository;

import com.visionguide.forum_api.model.Comment;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import java.util.List;

@Repository
public interface CommentRepository extends JpaRepository<Comment, Long> {
    // Belirli bir gönderiye ait tüm yorumları getirmek için
    // Spring Data JPA, bu metodun adından sorguyu otomatik olarak oluşturur.
    List<Comment> findByPostIdOrderByCreatedAtAsc(Long postId);
}