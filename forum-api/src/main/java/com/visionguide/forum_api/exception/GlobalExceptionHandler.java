package com.visionguide.forum_api.exception;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ControllerAdvice;
import org.springframework.web.bind.annotation.ExceptionHandler;

import java.util.Map;
import java.util.stream.Collectors;

@ControllerAdvice // Tüm @RestController'lar için merkezi hata yönetimi sağlar
public class GlobalExceptionHandler {

    // DTO'lardaki @NotBlank, @Size gibi @Valid hatalarını yakalar
    @ExceptionHandler(MethodArgumentNotValidException.class)
    public ResponseEntity<Map<String, String>> handleValidationExceptions(MethodArgumentNotValidException ex) {
        // Hataları "alan: mesaj" formatında bir haritaya dönüştür
        Map<String, String> errors = ex.getBindingResult().getFieldErrors().stream()
                .collect(Collectors.toMap(
                        fieldError -> fieldError.getField(),
                        fieldError -> fieldError.getDefaultMessage()
                ));
        return new ResponseEntity<>(errors, HttpStatus.BAD_REQUEST);
    }

    // ForumService'teki "IllegalArgumentException" (Erişilebilirlik hatası) yakalar
    @ExceptionHandler(IllegalArgumentException.class)
    public ResponseEntity<Map<String, String>> handleIllegalArgument(IllegalArgumentException ex) {
        return new ResponseEntity<>(
                Map.of("hata", ex.getMessage()), // "hata": "Erişilebilirlik ihlali..."
                HttpStatus.BAD_REQUEST
        );
    }
}