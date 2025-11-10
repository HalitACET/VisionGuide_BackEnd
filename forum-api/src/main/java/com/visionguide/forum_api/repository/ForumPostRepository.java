package com.visionguide.forum_api.repository;

import com.visionguide.forum_api.controller.dto.PostSummaryResponseDTO;
import com.visionguide.forum_api.model.ForumPost;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.util.Optional;

public interface ForumPostRepository extends JpaRepository<ForumPost, Long> {

    @Query("""
            SELECT new com.visionguide.forum_api.controller.dto.PostSummaryResponseDTO(
                p.id,
                p.title,
                p.authorId,
                p.createdAt,
                CASE WHEN LENGTH(CAST(p.content AS string)) > 300
                     THEN CONCAT(SUBSTRING(CAST(p.content AS string), 1, 300), '...')
                     ELSE CAST(p.content AS string)
                END,
                p.imageUrl
            )
            FROM ForumPost p
            """)
    Page<PostSummaryResponseDTO> findAllSummaries(Pageable pageable);

    @Query("SELECT DISTINCT p FROM ForumPost p LEFT JOIN FETCH p.comments WHERE p.id = :id")
    Optional<ForumPost> findByIdWithComments(@Param("id") Long id);
}