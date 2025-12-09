import logging
from typing import List, Dict, Any, Union

import io

from PIL import Image
import numpy as np
import torch

try:
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel
except ImportError:
    YOLO = None
    DetectionModel = None

logger = logging.getLogger(__name__)


# Önemli COCO sınıfları: sadece bu etiketler API cevabına dahil edilecek.
IMPORTANT_CLASSES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
}


class YoloEngine:
    def __init__(self, model_path: str = "yolov8m.pt"):
        """YOLOv8 tabanlı nesne tespiti motoru.

        model_path: Ultralytics YOLOv8 model dosyası veya hazır model ismi.
        """
        self.model_path = model_path
        self.model = None

    def load_model(self) -> bool:
        """Modeli yükler. YOLO paketi yoksa veya hata olursa False döner."""
        if YOLO is None:
            logger.error("Ultralytics YOLO paketi bulunamadı. 'pip install ultralytics' ile kurun.")
            return False
        try:
            if self.model is None:
                logger.info(f"YOLOv8 modeli yükleniyor: {self.model_path}")
                # PyTorch 2.6 ile gelen weights_only=True varsayılanı nedeniyle,
                # Ultralytics DetectionModel sınıfını güvenli allowlist'e ekliyoruz.
                try:
                    if DetectionModel is not None and hasattr(torch, "serialization") and hasattr(
                        torch.serialization, "add_safe_globals"
                    ):
                        torch.serialization.add_safe_globals([DetectionModel])
                except Exception as e:
                    logger.warning(f"YOLOv8 safe_globals kaydı sırasında hata oluştu: {e}")

                self.model = YOLO(self.model_path)
                logger.info("YOLOv8 modeli başarıyla yüklendi.")
            return True
        except Exception as e:
            logger.error(f"YOLOv8 model yükleme hatası: {e}")
            return False

    def _ensure_model(self) -> bool:
        if self.model is not None:
            return True
        return self.load_model()

    def predict(self, image: Union[bytes, Image.Image, np.ndarray]) -> List[Dict[str, Any]]:
        """Görüntüde nesne tespiti yapar ve sade bir dict listesi döner.

        Dönüş formatı:
        [
          {"label": str, "score": float, "bbox": [x1, y1, x2, y2]},
          ...
        ]
        Bbox koordinatları piksel cinsindedir.
        """
        if not self._ensure_model():
            logger.error("YOLO modeli yüklenemedi, boş sonuç döndürülüyor.")
            return []

        try:
            if isinstance(image, bytes):
                img = Image.open(io.BytesIO(image)).convert("RGB")
            elif isinstance(image, Image.Image):
                img = image.convert("RGB")
            else:
                # numpy array
                img = image

            results = self.model(img)[0]
            detections: List[Dict[str, Any]] = []

            for box in results.boxes:
                cls_id = int(box.cls[0])
                label = results.names.get(cls_id, str(cls_id))
                score = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(
                    {
                        "label": label,
                        "score": score,
                        "bbox": [x1, y1, x2, y2],
                    }
                )

            # Sadece önemli sınıfları API cevabına dahil et
            filtered = [d for d in detections if d["label"] in IMPORTANT_CLASSES]
            return filtered
        except Exception as e:
            logger.error(f"YOLOv8 prediction error: {e}")
            return []


# Global engine örneği
yolo_engine = YoloEngine()
