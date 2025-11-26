import os
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import requests
from ultralytics import YOLO

# === AYARLAR ===

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Eğitimden çıkan weight dosyan:
WEIGHTS_PATH = r"C:\Users\Hatice\Desktop\YAZİLİMMÜH\Yazilim5.sinif\YOLOv8-Fire-and-Smoke-Detection-main\runs\fire_smoke_yolov8s_cmp\weights\best.pt"

# Alarm karelerini kaydedeceğimiz klasör
ALARMS_DIR = os.path.join(ROOT_DIR, "alarms")

# Backend API url (bir sonraki adımda yazacağımız Flask app için)
BACKEND_URL = "http://127.0.0.1:5000/api/alarms"

# YOLO tahmin ayarları
CONF_THRESHOLD = 0.4  # minimum güven
IOU_THRESHOLD = 0.45

# Yanlış alarm filtresi:
WINDOW_SIZE = 30          # son 30 frame (~1 saniye gibi)
MIN_ALARM_FRAMES = 10     # bu 30 frame'in en az 10'unda yangın/duman görülsün
MIN_AREA_RATIO = 0.02     # bbox alanı frame alanının en az %2'si olsun

# Webcam veya video dosyası
# 0 -> varsayılan kamera; istersen dosya yolu ver: r"C:\video\yangin.mp4"
VIDEO_SOURCE = 0


def ensure_dirs():
    if not os.path.exists(ALARMS_DIR):
        os.makedirs(ALARMS_DIR, exist_ok=True)


def send_alarm_to_backend(label: str, confidence: float, area_ratio: float, image_path: str):
    """
    Alarm bilgisini backend'e gönderir (Flask API).
    Backend çalışmıyorsa sadece terminale uyarı basar.
    """
    payload = {
        "timestamp": datetime.utcnow().isoformat(),
        "source": str(VIDEO_SOURCE),
        "label": label,
        "confidence": float(confidence),
        "area_ratio": float(area_ratio),
        "image_path": image_path,
    }

    try:
        resp = requests.post(BACKEND_URL, json=payload, timeout=2)
        if resp.status_code != 200:
            print(f"[BACKEND] Alarm gönderilemedi, status={resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"[BACKEND] Bağlantı hatası: {e}")


def main():
    ensure_dirs()

    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"best.pt bulunamadı: {WEIGHTS_PATH}")

    print("Model yükleniyor:", WEIGHTS_PATH)
    model = YOLO(WEIGHTS_PATH)

    # sınıf isimleri (0: fire, 1: smoke, vs.)
    class_names = model.names
    print("Sınıflar:", class_names)

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Kamera/video açılamadı: {VIDEO_SOURCE}")

    # Son WINDOW_SIZE frame için "yangın/duman var mı?" bilgisi
    recent_fire_smoke = deque(maxlen=WINDOW_SIZE)

    alarm_active = False  # şu anda alarm açık mı? (sürekli tetiklemeyelim)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame okunamadı, video sonu veya kamera hatası.")
            break

        h, w = frame.shape[:2]
        frame_area = float(h * w)

        # YOLO tahmini (frame doğrudan verilebilir)
        results = model.predict(
            source=frame,
            imgsz=640,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False
        )

        frame_has_fire_smoke = False
        max_conf = 0.0
        max_label = None
        max_area_ratio = 0.0

        # Sonuçları çiz
        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = class_names.get(cls_id, str(cls_id))
                conf = float(box.conf[0])

                # bbox koordinatları
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Alan oranı
                box_area = float((x2 - x1) * (y2 - y1))
                area_ratio = box_area / frame_area

                # Fire/smoke varsayımı: tüm sınıflar yangınla ilgili
                # (Eğer "normal" sınıf varsa, onu hariç tutabilirsin)
                if area_ratio >= MIN_AREA_RATIO:
                    frame_has_fire_smoke = True

                    if conf > max_conf:
                        max_conf = conf
                        max_label = label
                        max_area_ratio = area_ratio

                # Görüntüye kutu çiz
                color = (0, 0, 255) if label.lower() in ["fire", "yangin"] else (0, 255, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"{label} {conf:.2f}",
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        # Bu frame'in durumunu kuyruğa ekle
        recent_fire_smoke.append(1 if frame_has_fire_smoke else 0)

        # Son WINDOW_SIZE frame içinde kaç tanesinde yangın/duman var?
        fire_count = sum(recent_fire_smoke)

        # Alarm koşulu:
        if fire_count >= MIN_ALARM_FRAMES and not alarm_active and max_label is not None:
            alarm_active = True

            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            image_filename = f"alarm_{timestamp}.jpg"
            image_path = os.path.join(ALARMS_DIR, image_filename)

            # O anki frame'i kaydet
            cv2.imwrite(image_path, frame)
            print(f"[ALARM] {timestamp} - {max_label} conf={max_conf:.2f} area={max_area_ratio:.3f}")
            print(f"[ALARM] Frame kaydedildi: {image_path}")

            # Backend'e gönder
            send_alarm_to_backend(
                label=max_label,
                confidence=max_conf,
                area_ratio=max_area_ratio,
                image_path=image_filename,  # sadece dosya adını yolladık
            )

        # Eğer son frame'lerde fire_count çok düştüyse alarmı kapat
        if fire_count < MIN_ALARM_FRAMES // 2:
            alarm_active = False

        # Ekrana bilgileri yaz
        status_text = f"FireFrames: {fire_count}/{len(recent_fire_smoke)}"
        cv2.putText(
            frame,
            status_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        if alarm_active:
            cv2.putText(
                frame,
                "ALARM!",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )

        cv2.imshow("Fire & Smoke Detection - Realtime", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):  # ESC veya q
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
