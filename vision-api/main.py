import io, base64, logging, asyncio, os
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Header, Depends
from pydantic import BaseModel
from PIL import Image

import numpy as np
import cv2
import tensorflow as tf
import tensorflow_hub as hub

# -------- CONFIG ----------
MODEL_HANDLE = "https://tfhub.dev/tensorflow/efficientdet/d0/1"
MODEL_INPUT_SIZE = 512
MAX_PARALLEL_INFERENCES = 2
MAX_IMAGE_BYTES = 10 * 1024 * 1024
DEFAULT_SCORE_THRESHOLD = 0.5
# Optional API key (set as env var) for simple auth:
API_KEY = os.environ.get("VISION_API_KEY")  # örn: export VISION_API_KEY="mysecret"
JWT_SECRET = os.environ.get("VISION_JWT_SECRET")  # eğer JWT kullanacaksanız
# --------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("visionguide_api")

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

# Yardımcılar
def _clean_base64(b64: str) -> str:
    if b64.startswith("data:"):
        parts = b64.split(",", 1)
        if len(parts) == 2:
            return parts[1]
    return b64

def base64_to_pil(base64_string: str) -> Image.Image:
    b64 = _clean_base64(base64_string)
    try:
        image_bytes = base64.b64decode(b64)
    except Exception as e:
        raise ValueError("Base64 decode failed") from e
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError("Image too large")
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")
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
def run_inference_sync(model, image_pil: Image.Image, score_threshold: float = DEFAULT_SCORE_THRESHOLD):
    orig_w, orig_h = image_pil.size  # PIL gives (width, height)
    resized = pil_to_resized_uint8_array(image_pil, MODEL_INPUT_SIZE)
    image_tensor = tf.convert_to_tensor(resized, dtype=tf.uint8)
    image_tensor = tf.expand_dims(image_tensor, 0)

    detections = model(image_tensor)

    boxes = ensure_numpy(detections.get("detection_boxes", detections.get("boxes")))
    scores = ensure_numpy(detections.get("detection_scores", detections.get("scores")))
    classes = ensure_numpy(detections.get("detection_classes", detections.get("classes")))
    classes = classes.astype(int)

    boxes = boxes[0]; scores = scores[0]; classes = classes[0]

    items = []
    for i in range(len(scores)):
        if float(scores[i]) >= score_threshold:
            ymin, xmin, ymax, xmax = boxes[i]  # normalized
            # pixel hesaplama image'in ORİJİNAL boyutuna göre
            x1 = int(xmin * orig_w)
            y1 = int(ymin * orig_h)
            x2 = int(xmax * orig_w)
            y2 = int(ymax * orig_h)
            label = COCO_LABELS.get(int(classes[i]), "Bilinmeyen")
            items.append({
                "label": label,
                "score": float(scores[i]),
                "box": [float(ymin), float(xmin), float(ymax), float(xmax)],
                "box_pixels": [x1, y1, x2, y2]
            })
    return items

# Model yükleme
@app.on_event("startup")
def load_model():
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
    except Exception:
        pass
    logger.info("Model yükleniyor... (bu işlem biraz sürebilir)")
    try:
        model = hub.load(MODEL_HANDLE)
        app.state.model = model
        logger.info("Model yüklendi.")
    except Exception:
        logger.exception("Model yüklenirken hata oluştu.")
        app.state.model = None

# Basit API-key bağımlılığı (opsiyonel)
def require_api_key(x_api_key: Optional[str] = Header(None)):
    if API_KEY:
        if x_api_key != API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")
    # eğer API_KEY ayarlı değilse bu kontrol pasif kalır (development)

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": app.state.model is not None}

@app.post("/detect", response_model=DetectionResponse, dependencies=[Depends(require_api_key)])
async def detect_objects(request: ImageUploadRequest, threshold: Optional[float] = Query(DEFAULT_SCORE_THRESHOLD, ge=0.0, le=1.0)):
    model = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")

    try:
        image_pil = base64_to_pil(request.image)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        async with inference_semaphore:
            loop = asyncio.get_event_loop()
            items = await loop.run_in_executor(executor, run_inference_sync, model, image_pil, float(threshold))
    except Exception:
        logger.exception("Inference error")
        raise HTTPException(status_code=500, detail="Inference failed")

    detections = [DetectionItem(**i) for i in items]
    return DetectionResponse(detections=detections)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
