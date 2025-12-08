import os
import json
import random
import numpy as np

try:
    import torch
except ImportError:
    torch = None


def set_seed(seed: int = 42):
    """Tüm modeller için ortak seed sabitleme."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_summary(run_dir: str, config: dict, summary: dict):
    """
    Eğitimin ve test/val sonuçlarının özetini JSON olarak kaydeder.
    Tüm modeller (YOLO, MobileNet, YOLOv5...) aynı formatı kullanacak.
    """
    summary_path = os.path.join(run_dir, "summary.json")

    summary_to_save = {
        "config": config,
        "results": summary,
    }

    os.makedirs(run_dir, exist_ok=True)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_to_save, f, indent=4, ensure_ascii=False)

    print("[OK] Özet JSON kaydedildi:", summary_path)
