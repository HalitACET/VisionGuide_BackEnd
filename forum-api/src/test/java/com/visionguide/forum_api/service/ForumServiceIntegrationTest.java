package com.visionguide.forum_api.service;

import com.visionguide.forum_api.controller.dto.CommentResponseDTO;
import com.visionguide.forum_api.controller.dto.PostDetailResponseDTO;
import com.visionguide.forum_api.controller.dto.PostSummaryResponseDTO;
import com.visionguide.forum_api.model.Comment;
import com.visionguide.forum_api.model.ForumPost;
import com.visionguide.forum_api.repository.ForumPostRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Sort;
import org.springframework.test.context.ActiveProfiles;

import java.time.Instant;
import static org.assertj.core.api.Assertions.assertThat;

@SpringBootTest
@ActiveProfiles("test")
class ForumServiceIntegrationTest {

    @Autowired
    private ForumPostRepository forumPostRepository;

    @Autowired
    private ForumService forumService;

    @BeforeEach
    void setUp() {
        forumPostRepository.deleteAll();
    }

    @Test
    void getAllPosts_returnsPagedSummariesWithExcerpt() {
        ForumPost post = persistPostWithComments(1, 2);

        Page<PostSummaryResponseDTO> page = forumService.getAllPosts(PageRequest.of(0, 5, Sort.by(Sort.Direction.DESC, "createdAt")));

        assertThat(page.getTotalElements()).isEqualTo(1);
        PostSummaryResponseDTO summary = page.getContent().get(0);
        assertThat(summary.getId()).isEqualTo(post.getId());
        assertThat(summary.getExcerpt())
                .hasSizeLessThanOrEqualTo(303)
                .endsWith("...");
        assertThat(summary.getImageUrl()).isEqualTo("https://example.com/image.png");
    }

    @Test
    void getPostByIdWithComments_returnsDetailedDtoWithSortedComments() {
        ForumPost post = persistPostWithComments(1, 3);

        PostDetailResponseDTO detail = forumService.getPostByIdWithComments(post.getId())
                .orElseThrow();

        assertThat(detail.getId()).isEqualTo(post.getId());
        assertThat(detail.getComments()).hasSize(3);
        assertThat(detail.getComments())
                .extracting(CommentResponseDTO::getCreatedAt)
                .isSorted();
    }

    private ForumPost persistPostWithComments(int postIndex, int commentCount) {
        ForumPost post = new ForumPost();
        post.setTitle("Sample Post " + postIndex);
        post.setContent("A".repeat(500));
        post.setAuthorId("author-" + postIndex);
        post.setCreatedAt(Instant.parse("2024-01-01T00:00:00Z").plusSeconds(postIndex));
        post.setImageUrl("https://example.com/image.png");
        post.setImageAltText("Sample alt text");

        for (int i = 0; i < commentCount; i++) {
            Comment comment = new Comment();
            comment.setContent("Comment " + i);
            comment.setAuthorId("commenter-" + i);
            comment.setCreatedAt(Instant.parse("2024-01-01T00:00:00Z").plusSeconds(i));
            comment.setPost(post);
            post.getComments().add(comment);
        }

        return forumPostRepository.saveAndFlush(post);
    }
}

