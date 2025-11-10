package com.visionguide.forum_api.controller;

import com.visionguide.forum_api.controller.dto.PostDetailResponseDTO;
import com.visionguide.forum_api.controller.dto.PostSummaryResponseDTO;
import com.visionguide.forum_api.service.ForumService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.Optional;

@RestController
@RequestMapping("/api/forum")
public class ForumPostController {

    private static final int MAX_PAGE_SIZE = 50;

    private final ForumService forumService;

    @Autowired
    public ForumPostController(ForumService forumService) {
        this.forumService = forumService;
    }

    @GetMapping("/posts")
    public ResponseEntity<Page<PostSummaryResponseDTO>> getAllPosts(
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "10") int size) {
        int safePage = Math.max(page, 0);
        int safeSize = Math.max(1, Math.min(size, MAX_PAGE_SIZE));
        Pageable pageable = PageRequest.of(safePage, safeSize, Sort.by(Sort.Direction.DESC, "createdAt"));
        Page<PostSummaryResponseDTO> result = forumService.getAllPosts(pageable);
        return ResponseEntity.ok(result);
    }

    @GetMapping("/posts/{id}")
    public ResponseEntity<PostDetailResponseDTO> getPostById(@PathVariable Long id) {
        Optional<PostDetailResponseDTO> opt = forumService.getPostByIdWithComments(id);
        return opt.map(ResponseEntity::ok).orElseGet(() -> ResponseEntity.notFound().build());
    }
}