# Dosya: analyze_webcam.py (Hızlandırılmış CPU Sürümü)

import cv2  # OpenCV: Kamera erişimi, görüntü işleme ve ekrana çizim yapmak için
import tensorflow as tf  # TensorFlow: Yapay zeka modelini çalıştırmak için
import tensorflow_hub as hub  # TensorFlow Hub: Hazır modelleri kolayca indirmek için
import numpy as np  # NumPy: Görüntüleri modelin anlayacağı format (dizi) haline getirmek için
import threading  # Kamerayı ve modeli ayırmak için Threading kütüphanesi
import time  # Yapay zeka thread'ini yavaşlatmak için

# --- 1. Model ve Etiket Yükleme ---

print("Yapay zeka modeli TensorFlow Hub'dan yükleniyor... (Bu işlem biraz sürebilir)")
# Tam (ağır) CPU modelini kullanıyoruz. Hızı koddaki optimizasyonla sağlayacağız.
model_handle = "https://tfhub.dev/tensorflow/efficientdet/d0/1"
model = hub.load(model_handle)
print("Model başarıyla yüklendi.")

# COCO Veri Seti Etiketleri (TÜRKÇELEŞTİRİLMİŞ)
COCO_LABELS = {
    1: 'kisi', 2: 'bisiklet', 3: 'araba', 4: 'motosiklet', 5: 'ucak', 6: 'otobus',
    7: 'tren', 8: 'kamyon', 9: 'tekne', 10: 'trafik lambasi', 11: 'yangin muslugu',
    13: 'dur isareti', 14: 'parkmetre', 15: 'bank', 16: 'kus', 17: 'kedi',
    18: 'kopek', 19: 'at', 20: 'koyun', 21: 'inek', 22: 'fil', 23: 'ayi',
    24: 'zebra', 25: 'zurafa', 27: 'sirt cantasi', 28: 'semsiye', 31: 'el cantasi',
    32: 'kravat', 33: 'bavul', 34: 'frizbi', 35: 'kayak', 36: 'snowboard',
    37: 'spor topu', 38: 'ucurtma', 39: 'beyzbol sopasi', 40: 'beyzbol eldiveni',
    41: 'kaykay', 42: 'sorf tahtasi', 43: 'tenis raketi', 44: 'sise',
    46: 'sarap kadehi', 47: 'bardak', 48: 'catal', 49: 'bicak', 50: 'kasik',
    51: 'kase', 52: 'muz', 53: 'elma', 54: 'sandvic', 55: 'portakal',
    56: 'brokoli', 57: 'havuc', 58: 'sosisli', 59: 'pizza', 60: 'corek',
    61: 'kek', 62: 'sandalye', 63: 'kanepe', 64: 'saksi bitkisi', 65: 'yatak',
    67: 'yemek masasi', 70: 'tuvalet', 72: 'televizyon', 73: 'dizustu bilgisayar', 74: 'fare',
    75: 'uzaktan kumanda', 76: 'klavye', 77: 'cep telefonu', 78: 'mikrodalga',
    79: 'firin', 80: 'tost makinesi', 81: 'lavabo', 82: 'buzdolabi', 84: 'kitap',
    85: 'saat', 86: 'vazo', 87: 'makas', 88: 'oyuncak ayi', 89: 'sac kurutma makinesi',
    90: 'dis fircasi'
}

# --- 2. Hız Optimizasyonu Ayarları ---
# Modeller genellikle daha küçük karelerde daha hızlı çalışır.
# Kameradan 640x480 çözünürlük isteyeceğiz.
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
# EfficientDet-d0 modeli 512x512 girdi bekler.
MODEL_INPUT_SIZE = 512

# --- 3. Global Değişkenler (Thread'ler arası iletişim için) ---
latest_frame = None
latest_results = None
running = True
lock = threading.Lock()


# --- 4. İş Parçacığı (Thread) Fonksiyonları ---

def camera_thread_function():
    """
    SADECE kameradan görüntü okur ve 'latest_frame' değişkenini günceller.
    """
    global latest_frame, running

    vid = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

    # === HIZLANDIRMA ÇÖZÜMÜ 1 ===
    # Kameradan istediğimiz çözünürlüğü talep ediyoruz.
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not vid.isOpened():
        print("HATA: Kamera açılamadı.")
        running = False
        return

    print("\nKamera açıldı. Çıkmak için 'q' tuşuna basın.")

    while running:
        ret, frame = vid.read()
        if not ret:
            print("HATA: Görüntü alınamadı.")
            running = False
            break

        with lock:
            latest_frame = frame.copy()

    vid.release()
    print("Kamera thread'i durdu.")


def inference_thread_function():
    """
    SADECE yapay zeka modelini çalıştırır.
    'latest_frame'den son kareyi alır, *yeniden boyutlandırır*, modeli çalıştırır ve 'latest_results'ı günceller.
    """
    global latest_frame, latest_results, running

    while running:
        frame_to_process = None
        with lock:
            if latest_frame is not None:
                frame_to_process = latest_frame.copy()

        if frame_to_process is None:
            time.sleep(0.05)  # Kamera henüz başlamadıysa bekle
            continue

        # 1. GÖRÜNTÜYÜ HAZIRLA
        # OpenCV (BGR) formatından -> RGB formatına dönüştür
        frame_rgb = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2RGB)

        # === HIZLANDIRMA ÇÖZÜMÜ 2 ===
        # Görüntüyü modelin beklediği 512x512 boyutuna getiriyoruz.
        # Bu, büyük (HD) bir görüntüyü işlemekten ÇOK daha hızlıdır.
        frame_resized = cv2.resize(frame_rgb, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))

        # Görüntüyü TensorFlow'un anladığı tensör formatına dönüştür
        image_tensor = tf.convert_to_tensor(frame_resized, dtype=tf.uint8)
        image_tensor = tf.expand_dims(image_tensor, 0)  # (1, 512, 512, 3)

        # 2. MODELİ ÇALIŞTIR (Artık çok daha küçük veri işliyor)
        detections = model(image_tensor)

        # 3. SONUÇLARI İŞLE
        boxes = detections['detection_boxes'][0].numpy()
        scores = detections['detection_scores'][0].numpy()
        class_ids = detections['detection_classes'][0].numpy().astype(int)

        results_list = []
        score_threshold = 0.5
        for i in range(len(scores)):
            if scores[i] > score_threshold:
                label = COCO_LABELS.get(class_ids[i], 'Bilinmeyen')
                results_list.append((boxes[i], label, scores[i]))

        with lock:
            latest_results = results_list

    print("Yapay zeka thread'i durdu.")


# --- 5. Yardımcı Çizim Fonksiyonu ---

def draw_bounding_box(frame, box, label, score):
    """
    Görüntü (frame) üzerine, modelin verdiği koordinatlara (box) bir kutu
    ve etiketi (label) çizen yardımcı fonksiyon.
    (Bu kodda değişiklik yok, çünkü kutu koordinatları (box) 0-1 arasındadır,
    bu da yeniden boyutlandırmadan etkilenmez.)
    """
    # Görüntünün o anki (640x480) yüksekliğini ve genişliğini al
    height, width, _ = frame.shape
    ymin, xmin, ymax, xmax = box
    (left, top) = (int(xmin * width), int(ymin * height))
    (right, bottom) = (int(xmax * width), int(ymax * height))
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    label_text = f"{label}: {int(score * 100)}%"
    label_size, base_line = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    top = max(top, label_size[1])
    cv2.rectangle(frame, (left, top - label_size[1]), (left + label_size[0], top + base_line), (255, 255, 255),
                  cv2.FILLED)
    cv2.putText(frame, label_text, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


# --- 6. Ana Program (Thread'leri Başlatma ve Görüntüleme) ---

def run_realtime_detection():
    """
    Kamera ve Yapay Zeka thread'lerini başlatır.
    Ana döngüde SADECE görüntü çizer, böylece donma olmaz.
    """
    global running, latest_frame, latest_results

    # Kamera ve Yapay Zeka thread'lerini oluştur ve başlat
    cam_thread = threading.Thread(target=camera_thread_function)
    inf_thread = threading.Thread(target=inference_thread_function)

    cam_thread.start()
    inf_thread.start()

    while True:
        display_frame = None
        current_results = None

        with lock:
            if latest_frame is not None:
                display_frame = latest_frame.copy()
                current_results = latest_results  # O anki sonuçları kopyala

        if display_frame is None:
            time.sleep(0.01)  # Kamera henüz başlamadıysa bekle
            continue

        # Sonuçlar varsa, kilit DIŞINDA ekrana çiz
        if current_results:
            for box, label, score in current_results:
                draw_bounding_box(display_frame, box, label, score)

        # Son kareyi ekranda göster (Bu artık çok hızlı olacak)
        cv2.imshow('VisionGuide Prototipi (Çıkış için Q)', display_frame)

        # 'q' tuşuna basılırsa döngüyü kır
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False  # Thread'lere durma sinyali gönder
            break

    # Thread'lerin güvenle kapanmasını bekle
    cam_thread.join()
    inf_thread.join()

    cv2.destroyAllWindows()
    print("Program sonlandırıldı.")


# --- 7. Programı Başlat ---
if __name__ == "__main__":
    run_realtime_detection()