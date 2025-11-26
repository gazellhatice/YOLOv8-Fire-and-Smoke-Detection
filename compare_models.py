import os
import time
import json
import random
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from ultralytics import YOLO

try:
    import torch
except ImportError:
    torch = None

# ==========================
# 0) GENEL AYARLAR
# ==========================

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_YAML = os.path.join(ROOT_DIR, "datasets", "fire-8", "data.yaml")

# Test görüntüleri (FPS ölçümü için bu dizini kullanacağız)
TEST_IMAGES_DIR = os.path.join(ROOT_DIR, "datasets", "fire-8", "test", "images")

# Karşılaştırılacak YOLOv8 modelleri
# Eğer yolov8m.pt kullanmak istersen alt satırı da açarsın
MODELS = [
    {
        "name": "yolov8n",
        "weights": os.path.join(ROOT_DIR, "yolov8n.pt"),
        "run_name": "fire_smoke_yolov8n_cmp",
        "epochs": 50,
    },
    {
        "name": "yolov8s",
        "weights": os.path.join(ROOT_DIR, "yolov8s.pt"),
        "run_name": "fire_smoke_yolov8s_cmp",
        "epochs": 50,
    },
    # {
    #     "name": "yolov8m",
    #     "weights": os.path.join(ROOT_DIR, "yolov8m.pt"),
    #     "run_name": "fire_smoke_yolov8m_cmp",
    #     "epochs": 50,
    # },
]

BASE_CONFIG = {
    "imgsz": 640,
    "batch": 16,
    "project": os.path.join(ROOT_DIR, "runs"),
    "patience": 15,
    "workers": 2,
    "use_test_split": True,   # data.yaml içinde test yoksa False yap
    "seed": 42,
}


def set_seed(seed: int = 42):
    """Sonuçlar daha tutarlı olsun diye seed sabitleme."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ==========================
# 1) EĞİTİM + VAL/Test
# ==========================

def train_and_eval(model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Bir model için:
      - Eğitim
      - Test/Val split üzerinde metrik hesaplama
      - Inference FPS ölçümü
    ve hepsinin özetini dict olarak döndürür.
    """

    set_seed(BASE_CONFIG["seed"])

    weights_path = model_cfg["weights"]
    model_name = model_cfg["name"]
    run_name = model_cfg["run_name"]
    epochs = model_cfg["epochs"]

    print("\n" + "=" * 70)
    print(f"MODEL: {model_name}")
    print("=" * 70)
    print("Weights:", weights_path)

    # Eğer localde yoksa, YOLO otomatik indirir
    model = YOLO(weights_path)

    # ---- Eğitim ----
    print(f"\n[{model_name}] Eğitim başlıyor...")
    results = model.train(
        data=DATA_YAML,
        imgsz=BASE_CONFIG["imgsz"],
        epochs=epochs,
        batch=BASE_CONFIG["batch"],
        name=run_name,
        project=BASE_CONFIG["project"],
        patience=BASE_CONFIG["patience"],
        workers=BASE_CONFIG["workers"],
    )

    # run klasörünü bul
    run_dir = getattr(results, "save_dir", None)
    if run_dir is None:
        try:
            run_dir = results[-1].save_dir
        except Exception:
            run_dir = os.path.join(
                BASE_CONFIG["project"], "train", run_name
            )

    print(f"[{model_name}] Eğitim tamamlandı. Run klasörü: {run_dir}")

    # ---- Değerlendirme (test veya val) ----
    split = "test" if BASE_CONFIG["use_test_split"] else "val"
    print(f"\n[{model_name}] {split.upper()} split üzerinde değerlendirme...")
    metrics = model.val(
        data=DATA_YAML,
        split=split,
        imgsz=BASE_CONFIG["imgsz"],
    )

    summary: Dict[str, Any] = {
        "model_name": model_name,
        "run_name": run_name,
        "epochs": epochs,
        "imgsz": BASE_CONFIG["imgsz"],
        "split": split,
        "map50_95": float(metrics.box.map),
        "map50": float(metrics.box.map50),
        "map75": float(metrics.box.map75),
        "precision_mean": float(metrics.box.mp),
        "recall_mean": float(metrics.box.mr),
        "run_dir": str(run_dir),
    }

    print("==== GENEL METRİKLER ====")
    print(f"[{model_name}] mAP50-95 : {summary['map50_95']:.4f}")
    print(f"[{model_name}] mAP50    : {summary['map50']:.4f}")
    print(f"[{model_name}] mAP75    : {summary['map75']:.4f}")
    print(f"[{model_name}] Precision: {summary['precision_mean']:.4f}")
    print(f"[{model_name}] Recall   : {summary['recall_mean']:.4f}")
    print("=========================\n")

    # Sınıf bazlı mAP
    try:
        class_names = metrics.names  # {0: 'fire', 1: 'smoke', ...}
        class_maps = metrics.box.maps  # liste

        per_class = {}
        print(f"[{model_name}] SINIF BAZLI mAP50-95:")
        for cls_id, cls_name in class_names.items():
            if cls_id < len(class_maps):
                cls_map = float(class_maps[cls_id])
                per_class[cls_name] = cls_map
                print(f"  {cls_name:10s}: {cls_map:.4f}")
        summary["per_class_map50_95"] = per_class
    except Exception as e:
        print("Sınıf bazlı mAP okunurken hata:", e)

    # ---- Inference hızı (FPS) ölçümü ----
    fps = measure_inference_fps(model, model_name)
    summary["fps"] = fps

    print(f"\n[{model_name}] Inference FPS (test görüntüleri üzerinde): {fps:.2f} FPS")

    return summary


# ==========================
# 2) INFERENCE SPEED ÖLÇÜMÜ
# ==========================

def measure_inference_fps(model: YOLO, model_name: str) -> float:
    """
    Test images klasöründeki tüm görüntüler üzerinde,
    model.predict süresini ölçerek FPS hesaplar.
    """
    if not os.path.isdir(TEST_IMAGES_DIR):
        print(f"[{model_name}] TEST_IMAGES_DIR bulunamadı: {TEST_IMAGES_DIR}")
        return 0.0

    image_paths: List[str] = []
    for fname in os.listdir(TEST_IMAGES_DIR):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            image_paths.append(os.path.join(TEST_IMAGES_DIR, fname))

    if not image_paths:
        print(f"[{model_name}] Test görüntüsü bulunamadı.")
        return 0.0

    # Çok fazla görüntü varsa, test için bir kısmını kullanabiliriz (ör: ilk 50)
    max_images = 50
    if len(image_paths) > max_images:
        image_paths = image_paths[:max_images]

    print(f"[{model_name}] FPS ölçümü için {len(image_paths)} görüntü kullanılacak.")

    # Warm-up (ilk çağrıda oluşabilecek overhead'i azaltmak için)
    _ = model.predict(
        source=image_paths[:5],
        imgsz=BASE_CONFIG["imgsz"],
        device="cpu",     # hepsini CPU'da kıyaslamak daha adil
        verbose=False,
    )

    start = time.perf_counter()
    _ = model.predict(
        source=image_paths,
        imgsz=BASE_CONFIG["imgsz"],
        device="cpu",
        verbose=False,
    )
    end = time.perf_counter()

    total_time = end - start
    total_images = len(image_paths)
    fps = total_images / total_time if total_time > 0 else 0.0

    return fps


# ==========================
# 3) MAIN
# ==========================

def main():
    if not os.path.exists(DATA_YAML):
        raise FileNotFoundError(f"data.yaml bulunamadı: {DATA_YAML}")

    all_results: List[Dict[str, Any]] = []

    for cfg in MODELS:
        result = train_and_eval(cfg)
        all_results.append(result)

    # Sonuçları hem JSON hem CSV olarak kaydedelim
    out_dir = os.path.join(ROOT_DIR, "runs")
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, "model_comparison.json")
    csv_path = os.path.join(out_dir, "model_comparison.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    # CSV için bazı alanları düzleştirelim
    flat_rows = []
    for r in all_results:
        row = {
            "model_name": r["model_name"],
            "epochs": r["epochs"],
            "imgsz": r["imgsz"],
            "split": r["split"],
            "map50_95": r["map50_95"],
            "map50": r["map50"],
            "map75": r["map75"],
            "precision_mean": r["precision_mean"],
            "recall_mean": r["recall_mean"],
            "fps": r.get("fps", 0.0),
        }
        flat_rows.append(row)

    df = pd.DataFrame(flat_rows)
    df.to_csv(csv_path, index=False)

    print("\n=== MODEL KARŞILAŞTIRMA ÖZETİ ===")
    print(df)
    print(f"\nJSON: {json_path}")
    print(f"CSV : {csv_path}")


if __name__ == "__main__":
    main()
