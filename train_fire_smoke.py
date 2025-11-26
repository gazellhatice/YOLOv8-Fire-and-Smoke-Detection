import os
import json
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

try:
    import torch
except ImportError:
    torch = None

# ==========================
# 1) GENEL AYARLAR & SEED
# ==========================

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_YAML = os.path.join(ROOT_DIR, "datasets", "fire-8", "data.yaml")

MODEL_PATH = os.path.join(ROOT_DIR, "yolov8n.pt")
# İstersen daha büyük modelle denemek için:
# MODEL_PATH = os.path.join(ROOT_DIR, "yolov8s.pt")

CONFIG = {
    "imgsz": 640,
    "epochs": 50,
    "batch": 16,
    "name": "fire_smoke_yolov8n",
    "project": os.path.join(ROOT_DIR, "runs"),
    "patience": 20,
    "workers": 2,
    "use_test_split": True,  # data.yaml içinde test yoksa bunu False yap ve split='val' kullan
    "seed": 42,
}


def set_seed(seed: int = 42):
    """Sonuçlar her çalıştırmada daha tutarlı olsun diye seed sabitleme."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ==========================
# 2) EĞİTİM
# ==========================

def train_model():
    print("ROOT_DIR:", ROOT_DIR)
    print("DATA_YAML:", DATA_YAML)
    print("MODEL_PATH:", MODEL_PATH)

    if not os.path.exists(DATA_YAML):
        raise FileNotFoundError(f"data.yaml bulunamadı: {DATA_YAML}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model dosyası bulunamadı: {MODEL_PATH}")

    set_seed(CONFIG["seed"])

    model = YOLO(MODEL_PATH)

    print("\n=== EĞİTİM BAŞLIYOR ===")
    results = model.train(
        data=DATA_YAML,
        imgsz=CONFIG["imgsz"],
        epochs=CONFIG["epochs"],
        batch=CONFIG["batch"],
        name=CONFIG["name"],
        project=CONFIG["project"],
        patience=CONFIG["patience"],
        workers=CONFIG["workers"],
    )

    # YOLOv8 genelde run klasörünü results.save_dir içinde tutuyor
    run_dir = getattr(results, "save_dir", None)
    if run_dir is None:
        try:
            run_dir = results[-1].save_dir
        except Exception:
            run_dir = os.path.join(
                CONFIG["project"], "train", CONFIG["name"]
            )

    print(f"Eğitim tamamlandı. Run klasörü: {run_dir}")
    return model, run_dir


# ==========================
# 3) DEĞERLENDİRME (TEST/VAL)
# ==========================

def eval_on_split(model):
    """
    Test (veya val) split üzerinde metrikleri hesapla.
    """
    split = "test" if CONFIG["use_test_split"] else "val"

    print(f"\n=== {split.upper()} SPLIT ÜZERİNDE DEĞERLENDİRME ===")
    metrics = model.val(
        data=DATA_YAML,
        split=split,
        imgsz=CONFIG["imgsz"],
    )

    # Genel metrikler
    summary = {
        "split": split,
        "map50_95": float(metrics.box.map),     # mAP@0.5:0.95
        "map50": float(metrics.box.map50),      # mAP@0.5
        "map75": float(metrics.box.map75),      # mAP@0.75
        "precision_mean": float(metrics.box.mp),
        "recall_mean": float(metrics.box.mr),
    }

    print("==== GENEL METRİKLER ====")
    print(f"mAP50-95 : {summary['map50_95']:.4f}")
    print(f"mAP50    : {summary['map50']:.4f}")
    print(f"mAP75    : {summary['map75']:.4f}")
    print(f"Precision: {summary['precision_mean']:.4f}")
    print(f"Recall   : {summary['recall_mean']:.4f}")
    print("=========================\n")

    # Sınıf bazlı mAP
    try:
        class_names = metrics.names  # {0: 'fire', 1: 'smoke', ...}
        class_maps = metrics.box.maps  # liste, indeksler sınıf id ile aynı

        per_class = {}
        print("==== SINIF BAZLI mAP50-95 ====")
        for cls_id, cls_name in class_names.items():
            if cls_id < len(class_maps):
                cls_map = float(class_maps[cls_id])
                per_class[cls_name] = cls_map
                print(f"{cls_name:10s}: {cls_map:.4f}")
        print("==============================\n")

        summary["per_class_map50_95"] = per_class
    except Exception as e:
        print("Sınıf bazlı mAP okunurken hata:", e)

    return summary


# ==========================
# 4) EĞRİLERİ ÇİZME
# ==========================

def plot_training_curves(run_dir: str):
    """
    YOLO'nun kaydettiği results.csv'den kendi grafiklerini çiz.
    """
    csv_path = os.path.join(run_dir, "results.csv")
    if not os.path.exists(csv_path):
        print("results.csv bulunamadı, grafik çizilemiyor:", csv_path)
        return

    df = pd.read_csv(csv_path)

    if "epoch" not in df.columns:
        print("results.csv içinde 'epoch' kolonu yok, grafik çizilemiyor.")
        return

    epochs = df["epoch"]

    print("results.csv kolonları:", list(df.columns))

    # === 1) Loss grafikleri ===
    plt.figure()
    if "train/box_loss" in df.columns:
        plt.plot(epochs, df["train/box_loss"], label="train box loss")
    if "train/cls_loss" in df.columns:
        plt.plot(epochs, df["train/cls_loss"], label="train cls loss")
    if "train/dfl_loss" in df.columns:
        plt.plot(epochs, df["train/dfl_loss"], label="train dfl loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Eğitim Kayıp (Loss) Eğrileri")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    loss_png = os.path.join(run_dir, "custom_loss_curves.png")
    plt.savefig(loss_png)
    plt.close()
    print("Loss grafiği kaydedildi:", loss_png)

    # === 2) mAP, precision, recall grafikleri ===
    plt.figure()
    if "metrics/precision(B)" in df.columns:
        plt.plot(epochs, df["metrics/precision(B)"], label="Precision")
    if "metrics/recall(B)" in df.columns:
        plt.plot(epochs, df["metrics/recall(B)"], label="Recall")
    if "metrics/mAP50(B)" in df.columns:
        plt.plot(epochs, df["metrics/mAP50(B)"], label="mAP@0.5")
    if "metrics/mAP50-95(B)" in df.columns:
        plt.plot(epochs, df["metrics/mAP50-95(B)"], label="mAP@0.5:0.95")

    plt.xlabel("Epoch")
    plt.ylabel("Değer")
    plt.title("Precision / Recall / mAP Eğrileri")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    metrics_png = os.path.join(run_dir, "custom_metrics_curves.png")
    plt.savefig(metrics_png)
    plt.close()
    print("Metrik grafikleri kaydedildi:", metrics_png)


# ==========================
# 5) ÖZETİ JSON'A KAYDETME
# ==========================

def save_summary(run_dir: str, summary: dict):
    """
    Eğitimin ve test/val sonuçlarının özetini JSON olarak kaydeder.
    Rapor veya sunum için çok işe yarar.
    """
    summary_path = os.path.join(run_dir, "summary.json")

    # Config'i de ekleyelim:
    summary_to_save = {
        "config": CONFIG,
        "results": summary,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_to_save, f, indent=4, ensure_ascii=False)

    print("Özet JSON kaydedildi:", summary_path)


# ==========================
# 6) MAIN
# ==========================

def main():
    model, run_dir = train_model()
    summary = eval_on_split(model)
    plot_training_curves(run_dir)
    save_summary(run_dir, summary)


if __name__ == "__main__":
    main()
