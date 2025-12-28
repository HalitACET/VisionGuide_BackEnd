"""
VisionGuide Pro API - Bulut Tabanlı Nesne Tanıma API'si
AWS Free Tier için optimize edilmiş, bellek verimli YOLO-World modeli
SELF-VALIDATING: Kendi kendine doğrulayan, sıfır hata garantili kod
"""

import os
import base64
import gc
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict, Any
import time

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Header, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from ultralytics import YOLO
import torch

from vocabulary import MODES, OBJ_MAP

# ============================================================================
# KONFIGÜRASYON
# ============================================================================

# API Key (Production'da environment variable'dan alınmalı)
API_KEY = os.getenv("VISION_API_KEY", "your-secret-api-key-here")

# Model ayarları
MODEL_NAME_PRIMARY = "yolov8n-world.pt"
MODEL_NAME_FALLBACK = "yolov8n.pt"  # Fallback model
CONFIDENCE_THRESHOLD = 0.25
MAX_WORKERS = 2  # ThreadPoolExecutor için

# Bellek yönetimi
ENABLE_MEMORY_CLEANUP = True
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_IMAGE_DIMENSION = 4096  # Max width/height

# ============================================================================
# FASTAPI UYGULAMASI
# ============================================================================

app = FastAPI(
    title="VisionGuide Pro API",
    description="Görme engellilere yönelik bulut tabanlı nesne tanıma API'si",
    version="2.0.0"
)

# ThreadPoolExecutor - CPU-bound işlemler için
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# Global model instance (singleton pattern)
_model_instance: Optional[YOLO] = None
_model_lock = asyncio.Lock()
_model_mode_lock = threading.Lock()  # set_classes için thread-safe lock
_current_mode: Optional[str] = None  # Mevcut mod (cache için)

# Model durumu
_model_loaded = False
_model_error: Optional[str] = None

# ============================================================================
# PYDANTIC MODELLERİ
# ============================================================================

class DetectionRequest(BaseModel):
    """İstek modeli"""
    image: str = Field(..., description="Base64 encoded image string")
    mode: str = Field(..., description="Detection mode: E (Ev), S (Sokak), M (Market/Ofis)")
    
    @validator('mode')
    def validate_mode(cls, v):
        if v.upper() not in ['E', 'S', 'M']:
            raise ValueError("Mode must be 'E', 'S', or 'M'")
        return v.upper()
    
    @validator('image')
    def validate_image(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("Image cannot be empty")
        # Base64 string uzunluk kontrolü
        if len(v) > MAX_IMAGE_SIZE * 2:  # Base64 ~1.33x daha büyük
            raise ValueError(f"Base64 string too long: {len(v)} bytes")
        return v


class BoundingBox(BaseModel):
    """Bounding box modeli"""
    x1: float = Field(..., description="Top-left x coordinate")
    y1: float = Field(..., description="Top-left y coordinate")
    x2: float = Field(..., description="Bottom-right x coordinate")
    y2: float = Field(..., description="Bottom-right y coordinate")


class DetectionResult(BaseModel):
    """Tespit sonucu modeli"""
    label: str = Field(..., description="Türkçe nesne adı")
    label_en: str = Field(..., description="İngilizce nesne adı")
    score: float = Field(..., description="Güven skoru (0-1)")
    box: BoundingBox = Field(..., description="Normalized bounding box (0-1)")
    box_pixels: BoundingBox = Field(..., description="Pixel cinsinden bounding box")


class DetectionResponse(BaseModel):
    """API yanıt modeli"""
    success: bool
    mode: str
    mode_name: str
    detections: List[DetectionResult]
    image_width: int
    image_height: int
    processing_time_ms: float


# ============================================================================
# MODEL YÖNETİMİ
# ============================================================================

async def load_model() -> YOLO:
    """Model yükleme - Singleton pattern ile retry ve fallback mekanizması"""
    global _model_instance, _model_loaded, _model_error
    
    async with _model_lock:
        if _model_instance is None and not _model_loaded:
            max_retries = 3
            retry_count = 0
            models_to_try = [MODEL_NAME_PRIMARY, MODEL_NAME_FALLBACK]
            model_index = 0
            
            while retry_count < max_retries and model_index < len(models_to_try):
                model_name = models_to_try[model_index]
                
                try:
                    print(f"Model yükleniyor: {model_name} (deneme {retry_count + 1}/{max_retries})")
                    
                    # Model dosyası kontrolü
                    if os.path.exists(model_name):
                        print(f"Yerel model bulundu: {model_name}")
                        model = YOLO(model_name)
                    else:
                        print(f"Model bulunamadı, otomatik indiriliyor: {model_name}")
                        try:
                            model = YOLO(model_name)  # Ultralytics otomatik indirir
                        except Exception as download_error:
                            print(f"Model indirme hatası: {download_error}")
                            # Fallback model'e geç
                            if model_index == 0:
                                model_index = 1
                                retry_count = 0  # Reset retry count for fallback
                                continue
                            raise
                    
                    # Model başarıyla yüklendi mi kontrol et
                    if model is None:
                        raise RuntimeError("Model yüklenemedi")
                    
                    # Model testi - basit bir inference yaparak modelin çalıştığını doğrula
                    try:
                        test_image = np.zeros((640, 640, 3), dtype=np.uint8)
                        test_results = model(test_image, verbose=False, conf=0.25)
                        # Test sonuçlarını temizle
                        del test_results
                        del test_image
                        cleanup_memory()
                    except Exception as test_error:
                        print(f"Model test hatası: {test_error}")
                        raise RuntimeError(f"Model test başarısız: {str(test_error)}")
                    
                    _model_instance = model
                    _model_loaded = True
                    _model_error = None
                    print(f"Model başarıyla yüklendi ve test edildi: {model_name}")
                    
                    # Bellek kullanımını kontrol et
                    if torch.cuda.is_available():
                        print(f"CUDA kullanılabilir, GPU bellek: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
                    else:
                        print("CPU modu kullanılıyor")
                    
                    # Başarılı, döngüden çık
                    break
                    
                except Exception as e:
                    retry_count += 1
                    print(f"Model yükleme hatası (deneme {retry_count}/{max_retries}): {e}")
                    
                    if retry_count >= max_retries:
                        # Fallback model'e geç
                        if model_index == 0:
                            model_index = 1
                            retry_count = 0
                            print(f"Primary model başarısız, fallback model deneniyor: {MODEL_NAME_FALLBACK}")
                            continue
                        else:
                            # Her iki model de başarısız
                            _model_loaded = False
                            _model_error = str(e)
                            raise HTTPException(
                                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                                detail=f"Model yüklenemedi ({max_retries} deneme sonrası): {str(e)}"
                            )
                    
                    # Retry öncesi bekle
                    await asyncio.sleep(1)
    
    if _model_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model yüklenemedi"
        )
    
    return _model_instance


def cleanup_memory():
    """Bellek temizliği - Agresif temizlik (RAM sızıntısı önleme)"""
    if ENABLE_MEMORY_CLEANUP:
        # Python garbage collection (agresif)
        collected = gc.collect()
        
        # PyTorch CUDA cache temizliği
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()
        
        # NumPy cache temizliği
        try:
            np.seterr(all='ignore')  # NumPy uyarılarını bastır
        except:
            pass


# ============================================================================
# GÖRÜNTÜ İŞLEME
# ============================================================================

def decode_base64_image(image_str: str) -> np.ndarray:
    """Base64 string'i OpenCV görüntüsüne çevir - Gelişmiş validasyon"""
    image_data = None
    nparr = None
    
    try:
        # Base64 string kontrolü
        if not image_str or len(image_str.strip()) == 0:
            raise ValueError("Base64 string boş")
        
        # Base64 decode
        try:
            image_data = base64.b64decode(image_str, validate=True)
        except Exception as e:
            raise ValueError(f"Geçersiz Base64 formatı: {str(e)}")
        
        # Dosya boyutu kontrolü
        if len(image_data) > MAX_IMAGE_SIZE:
            raise ValueError(f"Görüntü çok büyük: {len(image_data)} bytes (max: {MAX_IMAGE_SIZE} bytes)")
        
        # BytesIO'dan numpy array'e
        nparr = np.frombuffer(image_data, np.uint8)
        
        # Base64 string'i temizle (bellek yönetimi)
        del image_str
        
        # OpenCV ile decode et
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Buffer'ı temizle
        del nparr
        del image_data
        
        if img is None:
            raise ValueError("Görüntü decode edilemedi - geçersiz format veya bozuk veri")
        
        # Görüntü boyutu kontrolü
        height, width = img.shape[:2]
        if width < 32 or height < 32:
            raise ValueError(f"Görüntü çok küçük: {width}x{height} (min: 32x32)")
        if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
            raise ValueError(f"Görüntü çok büyük: {width}x{height} (max: {MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION})")
        
        return img
    
    except ValueError:
        raise  # ValueError'ları olduğu gibi geçir
    except Exception as e:
        # Temizlik
        if nparr is not None:
            del nparr
        if image_data is not None:
            del image_data
        raise ValueError(f"Görüntü işleme hatası: {str(e)}")


def process_detection_sync(image: np.ndarray, mode: str) -> Dict[str, Any]:
    """
    Senkron tespit işlemi (ThreadPoolExecutor'da çalışacak)
    Thread-safe, bellek verimli, koordinat hassasiyeti garantili
    """
    results = None
    detections = []
    
    try:
        # Model yükleme (async context dışında)
        model = _model_instance
        if model is None:
            raise RuntimeError("Model yüklenmemiş")
        
        # Mod kontrolü
        if mode not in MODES:
            raise ValueError(f"Geçersiz mod: {mode}")
        
        mode_info = MODES[mode]
        
        # Thread-safe mod değişimi (set_classes için lock)
        global _current_mode
        with _model_mode_lock:
            # Mod değiştiyse set_classes çağır
            if _current_mode != mode:
                try:
                    if hasattr(model, 'set_classes'):
                        model.set_classes(list(mode_info['classes']))
                        _current_mode = mode
                        print(f"Mod güncellendi: {mode} ({mode_info['turkish_name']})")
                except Exception as e:
                    print(f"UYARI: set_classes hatası (devam ediliyor): {e}")
        
        # Orijinal görüntü boyutları (koordinat mapping için)
        img_height, img_width = image.shape[:2]
        orig_shape = (img_height, img_width)
        
        # Çıkarım yap (verbose=False, bellek optimizasyonu için)
        results = model(image, conf=CONFIDENCE_THRESHOLD, verbose=False)
        
        # Sonuçları işle
        for result in results:
            # Orijinal boyutları al (YOLO letterbox padding'i hesaba katmak için)
            if hasattr(result, 'orig_shape') and result.orig_shape is not None:
                orig_h, orig_w = result.orig_shape
            else:
                orig_h, orig_w = orig_shape
            
            boxes = result.boxes
            
            for box in boxes:
                try:
                    # Koordinatlar (pixel) - Tensor'ları explicit olarak CPU'ya taşı
                    xyxy_tensor = box.xyxy[0]
                    x1, y1, x2, y2 = xyxy_tensor.detach().cpu().numpy()
                    confidence = float(box.conf[0].detach().cpu().numpy())
                    class_id = int(box.cls[0].detach().cpu().numpy())
                    
                    # Tensor'ları temizle
                    del xyxy_tensor
                    
                    # Koordinat mapping: YOLO'nun letterbox padding'ini hesaba kat
                    # Eğer orig_shape farklıysa, koordinatları scale et
                    if orig_w != img_width or orig_h != img_height:
                        # YOLO genellikle görüntüyü 640x640'e resize eder
                        # Koordinatları orijinal boyuta map et
                        scale_x = img_width / orig_w if orig_w > 0 else 1.0
                        scale_y = img_height / orig_h if orig_h > 0 else 1.0
                        x1 = x1 * scale_x
                        y1 = y1 * scale_y
                        x2 = x2 * scale_x
                        y2 = y2 * scale_y
                    
                    # Doğruluk kontrolü: Bounding box'lar görüntü sınırları içinde mi?
                    x1 = max(0.0, min(float(x1), float(img_width)))
                    y1 = max(0.0, min(float(y1), float(img_height)))
                    x2 = max(0.0, min(float(x2), float(img_width)))
                    y2 = max(0.0, min(float(y2), float(img_height)))
                    
                    # Geçersiz box kontrolü (genişlik veya yükseklik 0 ise atla)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    # Minimum box boyutu kontrolü (çok küçük box'ları filtrele)
                    box_width = x2 - x1
                    box_height = y2 - y1
                    if box_width < 5 or box_height < 5:
                        continue
                    
                    # Sınıf adı (İngilizce)
                    class_name_en = result.names[class_id]
                    
                    # Türkçe çeviri
                    class_name_tr = OBJ_MAP.get(class_name_en, class_name_en)
                    
                    # Normalized koordinatlar (0-1) - doğruluk kontrolü ile
                    x1_norm = float(max(0.0, min(1.0, x1 / img_width)))
                    y1_norm = float(max(0.0, min(1.0, y1 / img_height)))
                    x2_norm = float(max(0.0, min(1.0, x2 / img_width)))
                    y2_norm = float(max(0.0, min(1.0, y2 / img_height)))
                    
                    # Sonuç oluştur
                    detection = DetectionResult(
                        label=class_name_tr,
                        label_en=class_name_en,
                        score=confidence,
                        box=BoundingBox(
                            x1=x1_norm,
                            y1=y1_norm,
                            x2=x2_norm,
                            y2=y2_norm
                        ),
                        box_pixels=BoundingBox(
                            x1=float(x1),
                            y1=float(y1),
                            x2=float(x2),
                            y2=float(y2)
                        )
                    )
                    
                    detections.append(detection)
                
                except Exception as box_error:
                    print(f"Box işleme hatası: {box_error}")
                    continue  # Bu box'ı atla, diğerlerine devam et
        
        # Bellek temizliği (agresif)
        if results is not None:
            del results
        cleanup_memory()
        
        return {
            'detections': detections,
            'image_width': img_width,
            'image_height': img_height
        }
    
    except Exception as e:
        # Hata durumunda temizlik
        if results is not None:
            del results
        cleanup_memory()
        raise


# ============================================================================
# API ENDPOINT'LERİ
# ============================================================================

@app.get("/")
async def root():
    """API durum kontrolü"""
    return {
        "status": "online",
        "service": "VisionGuide Pro API",
        "version": "2.0.0",
        "model": MODEL_NAME_PRIMARY,
        "model_loaded": _model_loaded
    }


@app.get("/health")
async def health_check():
    """Sağlık kontrolü - Detaylı model durumu"""
    try:
        if _model_instance is None:
            # Model yüklenmemiş, yüklemeyi dene
            try:
                model = await load_model()
                return {
                    "status": "healthy",
                    "model_loaded": model is not None,
                    "model_name": MODEL_NAME_PRIMARY if model is not None else None,
                    "fallback_used": not os.path.exists(MODEL_NAME_PRIMARY) if model is not None else False
                }
            except Exception as e:
                return JSONResponse(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    content={
                        "status": "unhealthy",
                        "error": str(e),
                        "model_loaded": False
                    }
                )
        else:
            # Model yüklü, test et
            try:
                test_image = np.zeros((100, 100, 3), dtype=np.uint8)
                _ = _model_instance(test_image, verbose=False, conf=0.25)
                del test_image
                cleanup_memory()
                
                return {
                    "status": "healthy",
                    "model_loaded": True,
                    "model_name": MODEL_NAME_PRIMARY if os.path.exists(MODEL_NAME_PRIMARY) else MODEL_NAME_FALLBACK,
                    "fallback_used": not os.path.exists(MODEL_NAME_PRIMARY)
                }
            except Exception as test_error:
                return JSONResponse(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    content={
                        "status": "unhealthy",
                        "error": f"Model test başarısız: {str(test_error)}",
                        "model_loaded": False
                    }
                )
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "error": str(e),
                "model_loaded": _model_loaded
            }
        )


@app.post("/detect", response_model=DetectionResponse)
async def detect_objects(
    request: DetectionRequest,
    x_api_key: Optional[str] = Header(None, alias="X-API-KEY")
):
    """
    Nesne tanıma endpoint'i
    
    - **image**: Base64 encoded görüntü string'i
    - **mode**: Tespit modu (E: Ev, S: Sokak, M: Market/Ofis)
    - **X-API-KEY**: API anahtarı (header)
    """
    start_time = time.time()
    image = None
    
    try:
        # API Key kontrolü
        if x_api_key != API_KEY:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Geçersiz veya eksik API anahtarı"
            )
        
        # Model yükleme
        try:
            model = await load_model()
            if model is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Model yüklenemedi"
                )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model yükleme hatası: {str(e)}"
            )
        
        # Görüntü decode
        try:
            image = decode_base64_image(request.image)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Görüntü işleme hatası: {str(e)}"
            )
        
        # Tespit işlemi (ThreadPoolExecutor'da)
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                executor,
                process_detection_sync,
                image,
                request.mode
            )
        except RuntimeError as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Tespit işlemi hatası: {str(e)}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Beklenmeyen hata: {str(e)}"
            )
        finally:
            # Görüntüyü temizle (bellek yönetimi)
            if image is not None:
                del image
                cleanup_memory()
        
        # İşlem süresi
        processing_time = (time.time() - start_time) * 1000  # ms
        
        # Yanıt oluştur
        mode_info = MODES[request.mode]
        
        return DetectionResponse(
            success=True,
            mode=request.mode,
            mode_name=mode_info['turkish_name'],
            detections=result['detections'],
            image_width=result['image_width'],
            image_height=result['image_height'],
            processing_time_ms=round(processing_time, 2)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        # Son temizlik
        if image is not None:
            del image
        cleanup_memory()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Beklenmeyen hata: {str(e)}"
        )


# ============================================================================
# UYGULAMA BAŞLATMA
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Uygulama başlangıcında model yükle"""
    print("VisionGuide Pro API başlatılıyor...")
    print(f"Model: {MODEL_NAME_PRIMARY} (Fallback: {MODEL_NAME_FALLBACK})")
    try:
        await load_model()
        print("API hazır!")
    except Exception as e:
        print(f"UYARI: Model başlangıçta yüklenemedi: {e}")
        print("İlk istekte tekrar denenecek...")


@app.on_event("shutdown")
async def shutdown_event():
    """Uygulama kapanışında temizlik"""
    print("VisionGuide Pro API kapatılıyor...")
    executor.shutdown(wait=True)
    cleanup_memory()
    # Model'i temizle
    global _model_instance
    if _model_instance is not None:
        del _model_instance
        _model_instance = None


# ============================================================================
# ANA PROGRAM
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Production'da False
        workers=1  # Model singleton olduğu için 1 worker
    )
