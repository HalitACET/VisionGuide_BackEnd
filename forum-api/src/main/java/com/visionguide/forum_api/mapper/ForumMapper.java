package com.visionguide.forum_api.mapper;

import com.visionguide.forum_api.controller.dto.CommentResponseDTO;
import com.visionguide.forum_api.controller.dto.PostDetailResponseDTO;
import com.visionguide.forum_api.controller.dto.PostSummaryResponseDTO;
import com.visionguide.forum_api.model.Comment;
import com.visionguide.forum_api.model.ForumPost;
import org.springframework.stereotype.Component;

import java.util.stream.Collectors;

@Component
public class ForumMapper {

    public CommentResponseDTO toCommentDTO(Comment comment) {
        return new CommentResponseDTO(
                comment.getId(),
                comment.getContent(),
                comment.getAuthorId(),
                comment.getCreatedAt()
        );
    }

    public PostSummaryResponseDTO toPostSummaryDTO(ForumPost post) {
        return new PostSummaryResponseDTO(
                post.getId(),
                post.getTitle(),
                post.getAuthorId(),
                post.getCreatedAt(),
                buildExcerpt(post.getContent()),
                post.getImageUrl()
        );
    }

    public PostDetailResponseDTO toPostDetailDTO(ForumPost post) {
        PostDetailResponseDTO dto = new PostDetailResponseDTO();
        dto.setId(post.getId());
        dto.setTitle(post.getTitle());
        dto.setContent(post.getContent());
        dto.setAuthorId(post.getAuthorId());
        dto.setCreatedAt(post.getCreatedAt());
        dto.setImageUrl(post.getImageUrl());
        dto.setImageAltText(post.getImageAltText());
        dto.setComments(
                post.getComments().stream()
                        .map(this::toCommentDTO)
                        .collect(Collectors.toList())
        );
        return dto;
    }

    private String buildExcerpt(String content) {
        if (content == null || content.isBlank()) {
            return "";
        }
        int limit = Math.min(content.length(), 300);
        String excerpt = content.substring(0, limit);
        if (limit < content.length()) {
            return excerpt + "...";
        }
        return excerpt;
    }
}

