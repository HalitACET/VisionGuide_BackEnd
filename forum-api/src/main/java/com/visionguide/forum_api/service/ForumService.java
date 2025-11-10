package com.visionguide.forum_api.service;

import com.visionguide.forum_api.controller.dto.PostDetailResponseDTO;
import com.visionguide.forum_api.controller.dto.PostSummaryResponseDTO;
import com.visionguide.forum_api.mapper.ForumMapper;
import com.visionguide.forum_api.repository.ForumPostRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.Optional;

@Service
public class ForumService {

    private final ForumPostRepository postRepository;
    private final ForumMapper forumMapper;

    @Autowired
    public ForumService(ForumPostRepository postRepository, ForumMapper forumMapper) {
        this.postRepository = postRepository;
        this.forumMapper = forumMapper;
    }

    @Transactional(readOnly = true)
    public Page<PostSummaryResponseDTO> getAllPosts(Pageable pageable) {
        return postRepository.findAllSummaries(pageable);
    }

    @Transactional(readOnly = true)
    public Optional<PostDetailResponseDTO> getPostByIdWithComments(Long id) {
        return postRepository.findByIdWithComments(id).map(forumMapper::toPostDetailDTO);
    }

// Diğer CRUD metodları projedeki mevcut implementasyonunuza göre eklenebilir
}