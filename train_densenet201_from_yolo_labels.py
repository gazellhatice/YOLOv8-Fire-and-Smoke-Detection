import os
from typing import Dict, Any, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image

from train_utils import set_seed, save_summary  # Senin utils

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# YOLO dataset kökü
YOLO_ROOT = os.path.join(ROOT_DIR, "datasets", "fire-8")

CONFIG: Dict[str, Any] = {
    "model_name": "densenet201_from_yolo_labels",
    "epochs": 25,               # istersen 30-40 yapabilirsin
    "batch_size": 32,
    "lr": 1e-4,                 # büyük model için düşük lr
    "img_size": 224,
    "num_classes": 3,           # fire / smoke / none
    "seed": 42,
    "run_dir": os.path.join(ROOT_DIR, "runs", "cls_densenet201_yolo"),
    "classes": ["fire", "smoke", "none"],
}

CLASS_TO_IDX = {c: i for i, c in enumerate(CONFIG["classes"])}


# ----------------------------------------------------
# 1) YOLO label'ına göre sınıf seçen fonksiyon
# ----------------------------------------------------
def decide_class_from_label(label_path: str) -> str:
    """
    YOLO label dosyasına göre resmi hangi sınıfa koyacağımızı seçer.
    - hiç label yoksa -> 'none'
    - sadece class 0 varsa -> 'fire'
    - sadece class 1 varsa -> 'smoke'
    - hem 0 hem 1 varsa -> 'fire' (öncelik fire)
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
        return "fire"
    else:
        return "none"


# ----------------------------------------------------
# 2) Custom Dataset
# ----------------------------------------------------
class YoloFireSmokeDataset(Dataset):
    def __init__(self, img_dir: str, lbl_dir: str, transform=None):
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.transform = transform

        self.image_paths: List[str] = [
            os.path.join(self.img_dir, f)
            for f in os.listdir(self.img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        self.image_paths.sort()

        print(f"[INFO] {self.img_dir} içinde {len(self.image_paths)} görüntü bulundu.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        stem, _ = os.path.splitext(os.path.basename(img_path))
        label_path = os.path.join(self.lbl_dir, stem + ".txt")

        cls_name = decide_class_from_label(label_path)
        cls_idx = CLASS_TO_IDX[cls_name]

        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, cls_idx


# ----------------------------------------------------
# 3) Dataloaders (augmentation ile)
# ----------------------------------------------------
def get_dataloaders():
    img_size = CONFIG["img_size"]

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_img_dir = os.path.join(YOLO_ROOT, "train", "images")
    train_lbl_dir = os.path.join(YOLO_ROOT, "train", "labels")

    val_img_dir = os.path.join(YOLO_ROOT, "valid", "images")
    val_lbl_dir = os.path.join(YOLO_ROOT, "valid", "labels")

    train_ds = YoloFireSmokeDataset(train_img_dir, train_lbl_dir, transform=train_transform)
    val_ds   = YoloFireSmokeDataset(val_img_dir,   val_lbl_dir,   transform=val_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=2,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=2,
    )

    return train_loader, val_loader


# ----------------------------------------------------
# 4) Model (DenseNet201)
# ----------------------------------------------------
def build_model():
    num_classes = CONFIG["num_classes"]

    # Torchvision versiyonuna göre farklı olabilir, güvenli yazalım:
    try:
        weights = models.DenseNet201_Weights.IMAGENET1K_V1
        model = models.densenet201(weights=weights)
    except AttributeError:
        # Eski torchvision ise:
        model = models.densenet201(pretrained=True)

    # DenseNet201'de classifier, tek Linear katman
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)

    return model


# ----------------------------------------------------
# 5) Train & Eval
# ----------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def eval_model(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# ----------------------------------------------------
# 6) MAIN
# ----------------------------------------------------
def main():
    set_seed(CONFIG["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader, val_loader = get_dataloaders()
    model = build_model().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])

    best_val_acc = 0.0

    for epoch in range(1, CONFIG["epochs"] + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_model(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch}/{CONFIG['epochs']} "
            f"- train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f} "
            f"- val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(CONFIG["run_dir"], exist_ok=True)
            best_path = os.path.join(CONFIG["run_dir"], "best_densenet201.pth")
            torch.save(model.state_dict(), best_path)
            print("[OK] En iyi DenseNet201 modeli güncellendi:", best_path)

    summary = {
        "split": "val",
        "accuracy": float(best_val_acc),
        "map50_95": None,
        "map50": None,
        "map75": None,
        "precision_mean": None,
        "recall_mean": None,
    }

    save_summary(CONFIG["run_dir"], CONFIG, summary)


if __name__ == "__main__":
    main()
