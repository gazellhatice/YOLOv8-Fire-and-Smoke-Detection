import os
import shutil

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Eski YOLO dataset'i
YOLO_ROOT = os.path.join(ROOT_DIR, "datasets", "fire-8")

# Yeni classifier dataset'i
CLS_ROOT = os.path.join(ROOT_DIR, "datasets", "fire_classifier")

# Hangi splitleri kullanacağız (train -> train, valid -> val)
SPLIT_MAP = {
    "train": "train",
    "valid": "val",
    # istersen "test": "val" diyip test'ten de ekleyebilirsin
}

# Hedef sınıf isimleri
CLASSES = ["fire", "smoke", "none"]


def ensure_dirs():
    for split in ["train", "val"]:
        for cls in CLASSES:
            d = os.path.join(CLS_ROOT, split, cls)
            os.makedirs(d, exist_ok=True)


def decide_class(label_path: str) -> str:
    """
    YOLO label dosyasına göre resmi hangi sınıfa koyacağımızı seçer.
    - hiç label yoksa -> "none"
    - sadece class 0 varsa -> "fire"
    - sadece class 1 varsa -> "smoke"
    - hem 0 hem 1 varsa -> öncelik fire (istersen "fire_smoke" diye ayrı da yapabiliriz)
    """
    if not os.path.exists(label_path):
        return "none"

    with open(label_path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    if not lines:
        return "none"

    class_ids = set()
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        try:
            cid = int(parts[0])
            class_ids.add(cid)
        except ValueError:
            continue

    if not class_ids:
        return "none"

    has_fire = 0 in class_ids
    has_smoke = 1 in class_ids

    if has_fire and not has_smoke:
        return "fire"
    elif has_smoke and not has_fire:
        return "smoke"
    elif has_fire and has_smoke:
        # tek etiket zorundayız, kritik olduğu için fire'a yazalım
        return "fire"
    else:
        return "none"


def build():
    ensure_dirs()

    for yolo_split, cls_split in SPLIT_MAP.items():
        img_dir = os.path.join(YOLO_ROOT, yolo_split, "images")
        lbl_dir = os.path.join(YOLO_ROOT, yolo_split, "labels")

        if not os.path.isdir(img_dir):
            print(f"[WARN] {img_dir} yok, atlanıyor.")
            continue

        images = [
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        print(f"[{yolo_split}] {len(images)} görüntü bulundu.")

        for img_name in images:
            stem, ext = os.path.splitext(img_name)
            img_path = os.path.join(img_dir, img_name)
            lbl_path = os.path.join(lbl_dir, stem + ".txt")

            cls_name = decide_class(lbl_path)

            out_path = os.path.join(CLS_ROOT, cls_split, cls_name, img_name)
            shutil.copy2(img_path, out_path)

    print("Bitti. Yeni klasör yapısı:", CLS_ROOT)


if __name__ == "__main__":
    build()
