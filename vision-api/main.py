import io, base64, logging, asyncio, os, time, uuid
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Header, Depends, Request
from pydantic import BaseModel
from PIL import Image

import numpy as np
import cv2
import tensorflow as tf
import tensorflow_hub as hub

# -------- CONFIG ----------
MODEL_HANDLE = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"

MODEL_INPUT_SIZE = 512
MAX_PARALLEL_INFERENCES = 2
MAX_IMAGE_BYTES = 10 * 1024 * 1024
DEFAULT_SCORE_THRESHOLD = 0.5
# Optional API key (set as env var) for simple auth:
API_KEY = os.environ.get("VISION_API_KEY")  # örn: export VISION_API_KEY="mysecret"
JWT_SECRET = os.environ.get("VISION_JWT_SECRET")  # eğer JWT kullanacaksanız
# --------------------------

# Detaylı logging yapılandırması
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Request ID için custom filter
class RequestIDFilter(logging.Filter):
    def filter(self, record):
        record.request_id = getattr(record, 'request_id', 'SYSTEM')
        return True

logger = logging.getLogger("visionguide_api")
logger.addFilter(RequestIDFilter())

COCO_LABELS = {  # kısaltılmış listede gerektiği kadar genişletin
    1: 'kisi', 2: 'bisiklet', 3: 'araba', 4: 'motosiklet', 5: 'ucak', 6: 'otobus',
    7: 'tren', 8: 'kamyon', 9: 'tekne', 10: 'trafik lambasi', 11: 'yangin muslugu',
    13: 'dur isareti', 14: 'parkmetre', 15: 'bank', 16: 'kus', 17: 'kedi',
    18: 'kopek', 19: 'at', 20: 'koyun', 21: 'inek', 22: 'fil', 23: 'ayi',
    24: 'zebra', 25: 'zurafa', 27: 'sirt cantasi', 28: 'semsiye', 31: 'el cantasi',
    32: 'kravat', 33: 'bavul', 34: 'frizbi', 35: 'kayak', 36: 'snowboard',
    37: 'spor topu', 38: 'ucurtma', 39: 'beyzbol sopasi', 40: 'beyzbol eldiveni',
    41: 'kaykay', 42: 'sorf tahtasi', 43: 'tenis raketi', 44: 'sise',
    46: 'sarap kadehi', 47: 'bardak', 48: 'catal', 49: 'bicak', 50: 'kasik'
}

# Pydantic modeller
class ImageUploadRequest(BaseModel):
    image: str

class DetectionItem(BaseModel):
    label: str
    score: float
    box: List[float]         # normalized [ymin, xmin, ymax, xmax]
    box_pixels: List[int]    # pixel coords [x1, y1, x2, y2]

class DetectionResponse(BaseModel):
    detections: List[DetectionItem]

app = FastAPI(title="VisionGuide AI API")

executor = ThreadPoolExecutor(max_workers=MAX_PARALLEL_INFERENCES)
inference_semaphore = asyncio.Semaphore(MAX_PARALLEL_INFERENCES)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    # Request bilgilerini logla
    logger.info(
        f"📍 İstek başladı - Method: {request.method}, Path: {request.url.path}, "
        f"Client: {request.client.host if request.client else 'unknown'}, "
        f"Query: {dict(request.query_params)}",
        extra={"request_id": request_id}
    )
    
    # Request ID'yi state'e ekle
    request.state.request_id = request_id
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Response bilgilerini logla
        logger.info(
            f"✅ İstek tamamlandı - Status: {response.status_code}, "
            f"Süre: {process_time:.3f}s, "
            f"Path: {request.url.path}",
            extra={"request_id": request_id}
        )
        
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(process_time)
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(
            f"❌ İstek hatası - Path: {request.url.path}, "
            f"Hata: {str(e)}, Süre: {process_time:.3f}s",
            extra={"request_id": request_id},
            exc_info=True
        )
        raise

# Yardımcılar
def _clean_base64(b64: str) -> str:
    if b64.startswith("data:"):
        parts = b64.split(",", 1)
        if len(parts) == 2:
            return parts[1]
    return b64

def base64_to_pil(base64_string: str, request_id: str = "SYSTEM") -> Image.Image:
    b64 = _clean_base64(base64_string)
    decode_start = time.time()
    try:
        image_bytes = base64.b64decode(b64)
        decode_time = time.time() - decode_start
        image_size_kb = len(image_bytes) / 1024
        logger.info(
            f"🖼️ Base64 decode başarılı - Boyut: {image_size_kb:.2f} KB, "
            f"Süre: {decode_time:.3f}s",
            extra={"request_id": request_id}
        )
    except Exception as e:
        logger.error(
            f"❌ Base64 decode hatası - Hata: {str(e)}",
            extra={"request_id": request_id},
            exc_info=True
        )
        raise ValueError("Base64 decode failed") from e
    if len(image_bytes) > MAX_IMAGE_BYTES:
        logger.warning(
            f"⚠️ Görüntü çok büyük - Boyut: {len(image_bytes)} bytes, "
            f"Limit: {MAX_IMAGE_BYTES} bytes",
            extra={"request_id": request_id}
        )
        raise ValueError("Image too large")
    image = Image.open(io.BytesIO(image_bytes))
    orig_format = image.format
    orig_size = image.size
    if image.mode != "RGB":
        logger.debug(
            f"🔄 Görüntü modu değiştiriliyor - Eski: {image.mode}, Yeni: RGB",
            extra={"request_id": request_id}
        )
        image = image.convert("RGB")
    logger.info(
        f"📸 Görüntü yüklendi - Format: {orig_format}, "
        f"Boyut: {orig_size[0]}x{orig_size[1]}, Mode: RGB",
        extra={"request_id": request_id}
    )
    return image

def pil_to_resized_uint8_array(image_pil: Image.Image, size: int) -> np.ndarray:
    arr = np.array(image_pil)  # RGB
    resized = cv2.resize(arr, (size, size), interpolation=cv2.INTER_LINEAR)
    return resized.astype(np.uint8)

def ensure_numpy(x):
    try:
        if hasattr(x, "numpy"):
            return x.numpy()
        else:
            return np.array(x)
    except Exception:
        return np.array(x)

# Inference (blocking)
def run_inference_sync(model, image_pil: Image.Image, score_threshold: float = DEFAULT_SCORE_THRESHOLD, request_id: str = "SYSTEM"):
    inference_start = time.time()
    orig_w, orig_h = image_pil.size  # PIL gives (width, height)
    
    logger.info(
        f"🔄 Görüntü ön işleme başlıyor - Orijinal boyut: {orig_w}x{orig_h}, "
        f"Model boyutu: {MODEL_INPUT_SIZE}x{MODEL_INPUT_SIZE}, "
        f"Threshold: {score_threshold}",
        extra={"request_id": request_id}
    )
    
    resize_start = time.time()
    resized = pil_to_resized_uint8_array(image_pil, MODEL_INPUT_SIZE)
    resize_time = time.time() - resize_start
    logger.debug(
        f"🖼️ Görüntü yeniden boyutlandırıldı - Süre: {resize_time:.3f}s",
        extra={"request_id": request_id}
    )
    
    tensor_start = time.time()
    image_tensor = tf.convert_to_tensor(resized, dtype=tf.uint8)
    image_tensor = tf.expand_dims(image_tensor, 0)
    tensor_time = time.time() - tensor_start
    logger.debug(
        f"🔢 Tensor oluşturuldu - Süre: {tensor_time:.3f}s, Shape: {image_tensor.shape}",
        extra={"request_id": request_id}
    )

    model_start = time.time()
    detections = model(image_tensor)
    model_time = time.time() - model_start
    logger.info(
        f"🤖 Model inference tamamlandı - Süre: {model_time:.3f}s",
        extra={"request_id": request_id}
    )

    process_start = time.time()
    boxes = ensure_numpy(detections.get("detection_boxes", detections.get("boxes")))
    scores = ensure_numpy(detections.get("detection_scores", detections.get("scores")))
    classes = ensure_numpy(detections.get("detection_classes", detections.get("classes")))
    classes = classes.astype(int)

    boxes = boxes[0]; scores = scores[0]; classes = classes[0]
    total_detections = len(scores)
    logger.debug(
        f"📊 Ham detection sonuçları - Toplam tespit: {total_detections}",
        extra={"request_id": request_id}
    )

    items = []
    detection_num = 0
    for i in range(len(scores)):
        if float(scores[i]) >= score_threshold:
            detection_num += 1
            ymin, xmin, ymax, xmax = boxes[i]  # normalized
            # pixel hesaplama image'in ORİJİNAL boyutuna göre
            x1 = int(xmin * orig_w)
            y1 = int(ymin * orig_h)
            x2 = int(xmax * orig_w)
            y2 = int(ymax * orig_h)
            label = COCO_LABELS.get(int(classes[i]), "Bilinmeyen")
            score = float(scores[i])
            
            # Her tespit edilen nesne için detaylı log
            logger.info(
                f"🎯 Tespit #{detection_num} - Nesne: {label}, "
                f"Güven: {score:.2%}, "
                f"Konum: ({x1}, {y1}) - ({x2}, {y2}), "
                f"Boyut: {x2-x1}x{y2-y1} px",
                extra={"request_id": request_id}
            )
            
            items.append({
                "label": label,
                "score": score,
                "box": [float(ymin), float(xmin), float(ymax), float(xmax)],
                "box_pixels": [x1, y1, x2, y2]
            })
    
    process_time = time.time() - process_start
    inference_time = time.time() - inference_start
    
    # Detection sonuçlarını özetle
    if items:
        label_counts = {}
        total_confidence = 0.0
        for item in items:
            label = item["label"]
            label_counts[label] = label_counts.get(label, 0) + 1
            total_confidence += item["score"]
        avg_confidence = (total_confidence / len(items)) * 100
        label_summary = ", ".join([f"{label}: {count}" for label, count in label_counts.items()])
        logger.info(
            f"✅ Detection tamamlandı - Toplam tespit: {len(items)}, "
            f"Ortalama güven: {avg_confidence:.1f}%, "
            f"Toplam süre: {inference_time:.3f}s (Model: {model_time:.3f}s, "
            f"İşleme: {process_time:.3f}s) - Özet: {label_summary}",
            extra={"request_id": request_id}
        )
    else:
        logger.warning(
            f"⚠️ Hiç tespit yapılamadı - Threshold: {score_threshold}, "
            f"Toplam süre: {inference_time:.3f}s",
            extra={"request_id": request_id}
        )
    
    return items

# Model yükleme
@app.on_event("startup")
def load_model():
    startup_start = time.time()
    logger.info("🚀 VisionGuide API başlatılıyor...", extra={"request_id": "SYSTEM"})
    logger.info(f"📋 Log seviyesi: {LOG_LEVEL}", extra={"request_id": "SYSTEM"})
    
    # GPU kontrolü
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            logger.info(f"🎮 {len(gpus)} GPU bulundu", extra={"request_id": "SYSTEM"})
            for i, gpu in enumerate(gpus):
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info(f"  GPU {i}: Memory growth etkinleştirildi", extra={"request_id": "SYSTEM"})
                except Exception as e:
                    logger.warning(f"  GPU {i}: Memory growth ayarlanamadı - {str(e)}", extra={"request_id": "SYSTEM"})
        else:
            logger.info("💻 GPU bulunamadı, CPU kullanılacak", extra={"request_id": "SYSTEM"})
    except Exception as e:
        logger.warning(f"⚠️ GPU kontrolü başarısız: {str(e)}", extra={"request_id": "SYSTEM"})
    
    # TensorFlow versiyonu
    logger.info(f"📦 TensorFlow versiyonu: {tf.__version__}", extra={"request_id": "SYSTEM"})
    
    # Model yükleme
    logger.info(
        f"📥 Model yükleniyor... Model URL: {MODEL_HANDLE}, "
        f"Input Size: {MODEL_INPUT_SIZE}x{MODEL_INPUT_SIZE}",
        extra={"request_id": "SYSTEM"}
    )
    model_load_start = time.time()
    try:
        model = hub.load(MODEL_HANDLE)
        model_load_time = time.time() - model_load_start
        app.state.model = model
        logger.info(
            f"✅ Model başarıyla yüklendi - Süre: {model_load_time:.2f}s, "
            f"Model: {MODEL_HANDLE}",
            extra={"request_id": "SYSTEM"}
        )
    except Exception as e:
        model_load_time = time.time() - model_load_start
        logger.error(
            f"❌ Model yüklenirken hata oluştu - Süre: {model_load_time:.2f}s, "
            f"Hata: {str(e)}",
            extra={"request_id": "SYSTEM"},
            exc_info=True
        )
        app.state.model = None
    
    startup_time = time.time() - startup_start
    logger.info(
        f"🎉 VisionGuide API başlatma tamamlandı - Toplam süre: {startup_time:.2f}s, "
        f"Model durumu: {'Yüklü' if app.state.model is not None else 'Yüklenemedi'}",
        extra={"request_id": "SYSTEM"}
    )

@app.on_event("shutdown")
def shutdown_event():
    logger.info("🛑 VisionGuide API kapatılıyor...", extra={"request_id": "SYSTEM"})
    executor.shutdown(wait=False)
    logger.info("✅ VisionGuide API kapatıldı", extra={"request_id": "SYSTEM"})

# Basit API-key bağımlılığı (opsiyonel)
def require_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    if API_KEY:
        if x_api_key != API_KEY:
            logger.warning("🔒 API key doğrulama başarısız", extra={"request_id": "AUTH"})
            raise HTTPException(status_code=401, detail="Invalid API key")
        else:
            logger.debug("🔓 API key doğrulama başarılı", extra={"request_id": "AUTH"})
    else:
        logger.debug("⚠️ API key kontrolü devre dışı (development mode)", extra={"request_id": "AUTH"})

@app.get("/health")
async def health(request: Request):
    request_id = getattr(request.state, "request_id", "SYSTEM")
    model_loaded = app.state.model is not None
    logger.info(
        f"🏥 Health check - Model durumu: {'Yüklü' if model_loaded else 'Yüklenemedi'}",
        extra={"request_id": request_id}
    )
    return {"status": "ok", "model_loaded": model_loaded}

@app.post("/detect", response_model=DetectionResponse, dependencies=[Depends(require_api_key)])
async def detect_objects(
    request_body: ImageUploadRequest, 
    request: Request,
    threshold: Optional[float] = Query(DEFAULT_SCORE_THRESHOLD, ge=0.0, le=1.0)
):
    request_id = getattr(request.state, "request_id", str(uuid.uuid4())[:8])
    
    logger.info(
        f"🔍 Detection isteği alındı - Threshold: {threshold}",
        extra={"request_id": request_id}
    )
    
    detect_start = time.time()
    model = getattr(app.state, "model", None)
    if model is None:
        logger.error(
            f"❌ Model kullanılamıyor - Model durumu: None",
            extra={"request_id": request_id}
        )
        raise HTTPException(status_code=503, detail="Model not available")

    # Base64 string uzunluğunu logla (güvenlik için tam içeriği değil)
    base64_length = len(request_body.image)
    logger.debug(
        f"📥 Base64 string alındı - Uzunluk: {base64_length} karakter",
        extra={"request_id": request_id}
    )

    try:
        image_pil = base64_to_pil(request_body.image, request_id)
    except ValueError as e:
        logger.error(
            f"❌ Görüntü yükleme hatası - Hata: {str(e)}",
            extra={"request_id": request_id},
            exc_info=True
        )
        raise HTTPException(status_code=400, detail=str(e))

    try:
        logger.debug(
            f"⏳ Inference semaphore bekleniyor (max: {MAX_PARALLEL_INFERENCES})",
            extra={"request_id": request_id}
        )
        semaphore_wait_start = time.time()
        async with inference_semaphore:
            semaphore_acquired_time = time.time()
            wait_time = semaphore_acquired_time - semaphore_wait_start
            if wait_time > 0.1:
                logger.info(
                    f"⏱️ Semaphore bekleme süresi: {wait_time:.3f}s",
                    extra={"request_id": request_id}
                )
            
            logger.info(
                f"🚀 Inference başlatılıyor - Threshold: {threshold}",
                extra={"request_id": request_id}
            )
            loop = asyncio.get_event_loop()
            items = await loop.run_in_executor(
                executor, 
                run_inference_sync, 
                model, 
                image_pil, 
                float(threshold),
                request_id
            )
    except Exception as e:
        logger.error(
            f"❌ Inference hatası - Hata: {str(e)}, Süre: {time.time() - detect_start:.3f}s",
            extra={"request_id": request_id},
            exc_info=True
        )
        raise HTTPException(status_code=500, detail="Inference failed")

    detections = [DetectionItem(**i) for i in items]
    total_time = time.time() - detect_start
    
    logger.info(
        f"🎯 Detection isteği tamamlandı - Toplam süre: {total_time:.3f}s, "
        f"Tespit sayısı: {len(detections)}, Threshold: {threshold}",
        extra={"request_id": request_id}
    )
    
    return DetectionResponse(detections=detections)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
