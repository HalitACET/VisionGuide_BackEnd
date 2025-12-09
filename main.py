"""
VisionGuide API - FastAPI tabanlı nesne tespiti servisi
EfficientDet D0 modeli kullanarak görüntülerde nesne tespiti yapar.
"""

import base64
import io
import logging
from typing import List, Dict, Any
import numpy as np
from PIL import Image
try:
    import pytesseract
except ImportError:
    pytesseract = None
try:
    import tensorflow as tf
    import tensorflow_hub as hub
except ImportError:
    print("TensorFlow not found. Running in Mock Mode.")
    tf = None
    hub = None
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from labels import get_label_name
from yolo_engine import yolo_engine
from gemini_engine import GeminiEngine

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI uygulaması
app = FastAPI(
    title="VisionGuide API",
    description="EfficientDet D0 modeli ile nesne tespiti API'si",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production'da spesifik domain'ler belirtin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global değişkenler
detector = None
gemini_engine = None
detection_threshold = 0.65  # 0.65'e yükseltildi - daha güvenilir ve doğru tespitler için

# Pydantic modelleri
class DetectionRequest(BaseModel):
    image: str  # Base64 encoded image

class Detection(BaseModel):
    label: str
    score: float
    box: List[float]  # [x1, y1, x2, y2] normalized coordinates

class DetectionResponse(BaseModel):
    detections: List[Detection]
    total_detections: int

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

def load_model():
    """EfficientDet D0 modelini yükler"""
    global detector
    if hub is None:
        logger.warning("TensorFlow Hub not available. Skipping model load.")
        return False
        
    try:
        logger.info("EfficientDet D0 modeli yükleniyor...")
        model_url = "https://tfhub.dev/tensorflow/efficientdet/d0/1"
        detector = hub.load(model_url)
        logger.info("Model başarıyla yüklendi!")
        return True
    except Exception as e:
        logger.error(f"Model yükleme hatası: {e}")
        return False

def preprocess_image(image_data: bytes) -> np.ndarray:
    """Görüntüyü model için hazırlar"""
    try:
        # PIL Image'a çevir
        image = Image.open(io.BytesIO(image_data))
        
        # RGB formatına çevir
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # NumPy array'e çevir
        image_array = np.array(image)
        
        if tf is None:
             return image_array # Return numpy array if TF is missing

        # TensorFlow tensor'a çevir
        image_tensor = tf.convert_to_tensor(image_array, dtype=tf.uint8)
        
        # Batch dimension ekle (model 4 boyutlu tensor bekliyor: [batch, height, width, channels])
        image_tensor = tf.expand_dims(image_tensor, 0)
        
        return image_tensor
    except Exception as e:
        logger.error(f"Görüntü işleme hatası: {e}")
        raise HTTPException(status_code=400, detail="Geçersiz görüntü formatı")

def postprocess_detections(detections: Dict[str, Any]) -> List[Detection]:
    """Model çıktısını API formatına çevirir"""
    try:
        detection_results = []
        
        # Model çıktısından detection'ları al
        # TensorFlow Hub model çıktısı farklı format olabilir
        if isinstance(detections, dict):
            boxes = detections.get('detection_boxes', [])
            scores = detections.get('detection_scores', [])
            classes = detections.get('detection_classes', [])
        else:
            # Eğer detections bir tensor ise
            boxes = detections[0] if len(detections) > 0 else []
            scores = detections[1] if len(detections) > 1 else []
            classes = detections[2] if len(detections) > 2 else []
        
        # Tensor'ları numpy array'e çevir
        if hasattr(boxes, 'numpy'):
            boxes = boxes.numpy()
        if hasattr(scores, 'numpy'):
            scores = scores.numpy()
        if hasattr(classes, 'numpy'):
            classes = classes.numpy()
        
        # Batch dimension'ı kaldır (eğer varsa)
        if isinstance(boxes, np.ndarray) and len(boxes.shape) > 1:
            if boxes.shape[0] == 1:
                boxes = boxes[0]  # (1, N, 4) -> (N, 4)
        if isinstance(scores, np.ndarray) and len(scores.shape) > 1:
            if scores.shape[0] == 1:
                scores = scores[0]  # (1, N) -> (N,)
        if isinstance(classes, np.ndarray) and len(classes.shape) > 1:
            if classes.shape[0] == 1:
                classes = classes[0]  # (1, N) -> (N,)
        
        # Eğer boş ise, boş liste döndür
        if isinstance(scores, np.ndarray):
            if scores.size == 0:
                return detection_results
            num_detections = scores.shape[0]
        else:
            if len(scores) == 0:
                return detection_results
            num_detections = len(scores)
        
        # Detection'ları işle
        for i in range(num_detections):
            score = float(scores[i]) if isinstance(scores, np.ndarray) else float(scores[i])
            
            if score >= detection_threshold:
                class_id = int(classes[i]) if isinstance(classes, np.ndarray) else int(classes[i])
                label = get_label_name(class_id)
                
                # Box formatını kontrol et ve dönüştür
                # EfficientDet genellikle [y1, x1, y2, x2] formatında döner
                # Biz [x1, y1, x2, y2] formatını bekliyoruz
                if isinstance(boxes, np.ndarray):
                    box = boxes[i]
                    if hasattr(box, 'tolist'):
                        box_list = box.tolist()
                    else:
                        box_list = list(box)
                else:
                    box_list = list(boxes[i])
                
                # Box formatını [x1, y1, x2, y2] formatına çevir
                if len(box_list) >= 4:
                    # Eğer [y1, x1, y2, x2] formatındaysa
                    y1, x1, y2, x2 = box_list[:4]
                    box_list = [x1, y1, x2, y2]
                
                detection = Detection(
                    label=label,
                    score=score,
                    box=box_list
                )
                detection_results.append(detection)
        
        return detection_results
    except Exception as e:
        logger.error(f"Detection işleme hatası: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # Hata durumunda boş liste döndür
        return []


class ImageUploadRequest(BaseModel):
    image: str

class SegmentRequest(BaseModel):
    image: str
    prompt: str

class OcrResponse(BaseModel):
    text: str

class AnalyzeRequest(BaseModel):
    image: str
    feature: str # "currency" or "color"

class AnalyzeResponse(BaseModel):
    result: str

class Post(BaseModel):
    id: int
    title: str
    content: str
    author: str

class CreatePostRequest(BaseModel):
    title: str
    content: str
    author: str = "Anonymous"

class AskGeminiRequest(BaseModel):
    image: str
    prompt: str

class AskGeminiResponse(BaseModel):
    answer: str

import sqlite3

# Database setup
def init_db():
    conn = sqlite3.connect('visionguide.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS posts
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, content TEXT, author TEXT)''')
    
    # Add initial data if empty
    c.execute('SELECT count(*) FROM posts')
    if c.fetchone()[0] == 0:
        c.execute("INSERT INTO posts (title, content, author) VALUES (?, ?, ?)",
                  ("Merhaba!", "VisionGuide topluluğuna hoş geldiniz.", "Admin"))
        c.execute("INSERT INTO posts (title, content, author) VALUES (?, ?, ?)",
                  ("Renkler", "Kırmızı ve yeşili ayırt etmekte zorlanıyorum, ipucu var mı?", "User123"))
        conn.commit()
    conn.close()

@app.get("/posts", response_model=List[Post])
async def get_posts():
    conn = sqlite3.connect('visionguide.db')
    c = conn.cursor()
    c.execute("SELECT * FROM posts")
    rows = c.fetchall()
    conn.close()
    
    posts = []
    for row in rows:
        posts.append(Post(id=row[0], title=row[1], content=row[2], author=row[3]))
    return posts

@app.post("/posts", response_model=Post)
async def create_post(request: CreatePostRequest):
    conn = sqlite3.connect('visionguide.db')
    c = conn.cursor()
    c.execute("INSERT INTO posts (title, content, author) VALUES (?, ?, ?)",
              (request.title, request.content, request.author))
    new_id = c.lastrowid
    conn.commit()
    conn.close()
    
    return Post(
        id=new_id,
        title=request.title,
        content=request.content,
        author=request.author
    )

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_image(request: AnalyzeRequest):
    """Analyze endpoint for Currency and Color using Gemini"""
    logger.info(f"Analyze request received for feature: {request.feature}")
    
    try:
        # Currency Recognition with Gemini
        if request.feature == "currency":
            if gemini_engine and gemini_engine.model:
                image_bytes = base64.b64decode(request.image)
                prompt = (
                    "Bu resimdeki paraları tanımla. Kağıt ve madeni paraları ayırt et. "
                    "Her bir paranın değerini listele ve en sonda 'Toplam Tutar: X TL' şeklinde toplamı yaz. "
                    "Sadece paralarla ilgili bilgi ver."
                )
                result = gemini_engine.generate_content(image_bytes, prompt)
            else:
                logger.warning("Gemini engine not available, falling back to mock.")
                result = "50 Türk Lirası (Mock - API Anahtarı Ayarlanmadı)"
            
            return AnalyzeResponse(result=result)

        elif request.feature == "color":
             # Color recognition could also be moved to Gemini easily, but keeping mock/simple for now as requested only currency
            return AnalyzeResponse(result="Kırmızı")
        else:
            return AnalyzeResponse(result="Bilinmeyen özellik")

    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/ocr", response_model=OcrResponse)
async def read_text(request: ImageUploadRequest):
    """Gerçek OCR endpoint'i.
    
    Gemini API kullanarak görseldeki metni okur.
    """
    logger.info("OCR request received")

    try:
        # Base64'ü çöz
        image_bytes = base64.b64decode(request.image)
        
        # Gemini ile metin okuma
        if gemini_engine and gemini_engine.model:
            prompt = "Görüntüdeki tüm metni olduğu gibi, satır satır oku. Yorum yapma, sadece metni ver."
            text = gemini_engine.generate_content(image_bytes, prompt)
        else:
            # Fallback (Eğer Gemini yoksa)
            logger.warning("Gemini engine not available for OCR, falling back to mock.")
            text = "Gemini API anahtarı bulunamadı. Lütfen ayarlayın."

        return OcrResponse(text=text)
    except Exception as e:
        logger.error(f"OCR processing error: {e}")
        raise HTTPException(status_code=500, detail="OCR işlemi sırasında hata oluştu")

@app.post("/ask_gemini", response_model=AskGeminiResponse)
async def ask_gemini(request: AskGeminiRequest):
    """Gemini API kullanarak görüntü hakkında soru sorar."""
    try:
        image_bytes = base64.b64decode(request.image)
        answer = gemini_engine.generate_content(image_bytes, request.prompt)
        return AskGeminiResponse(answer=answer)
    except Exception as e:
        logger.error(f"Gemini request error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """Uygulama başlatıldığında modelleri yükle"""
    logger.info("VisionGuide API başlatılıyor...")
    
    # Initialize Database
    init_db()
    
    # Load EfficientDet (opsiyonel)
    eff_loaded = load_model()
    if not eff_loaded:
        logger.error("EfficientDet model yüklenemedi!")

    # Load YOLOv8 (nesne tespiti için)
    yolo_loaded = yolo_engine.load_model()
    if not yolo_loaded:
        logger.error("YOLOv8 model yüklenemedi. 'pip install ultralytics' gerekli olabilir.")

    # Initialize Gemini Engine
    global gemini_engine
    gemini_engine = GeminiEngine()

    # SAM 3 is mocked
    logger.info("SAM 3 Mock Mode aktif.")

@app.post("/segment")
async def segment_objects(request: SegmentRequest):
    """SAM 3 ile metin tabanlı segmentasyon yapar (Mock Mode)"""
    try:
        # Mock response for testing visualization
        # Returns a box polygon: [[100, 100], [400, 100], [400, 400], [100, 400]]
        logger.info(f"Mocking segmentation for prompt: {request.prompt}")
        
        mock_results = [
            {
                "label": request.prompt,
                "score": 0.99,
                "mask": [[100, 100], [400, 100], [400, 400], [100, 400]]
            }
        ]
        
        return {"results": mock_results}
    except Exception as e:
        logger.error(f"Segmentation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect", response_model=DetectionResponse)
async def detect_objects(request: DetectionRequest):
    """YOLOv8 ile nesne tespiti yapar."""
    try:
        image_bytes = base64.b64decode(request.image)
    except Exception:
        raise HTTPException(status_code=400, detail="Geçersiz base64 görüntü")

    detections_raw = yolo_engine.predict(image_bytes)
    detections = [
        Detection(label=d["label"], score=float(d["score"]), box=d["bbox"])
        for d in detections_raw
    ]
    return DetectionResponse(detections=detections, total_detections=len(detections))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
