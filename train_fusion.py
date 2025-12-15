import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from fusion_model import YoloVitFusionNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(ROOT_DIR, "datasets", "fire_classifier")

# 1) Transformlar
img_size = 224
train_tf = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def get_loaders(data_root, batch_size=32):
    train_dir = os.path.join(data_root, "train")
    val_dir   = os.path.join(data_root, "val")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Train klasörü bulunamadı: {train_dir}")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"Val klasörü bulunamadı: {val_dir}")

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds   = datasets.ImageFolder(val_dir,   transform=val_tf)

    print("Train örnek sayısı:", len(train_ds))
    print("Valid   örnek sayısı:", len(val_ds))
    print("Sınıf isimleri:", train_ds.classes)

    # Windows'ta sorun olmasın diye num_workers=0 yapıyoruz
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader


def create_model():
    model = YoloVitFusionNet(
        yolo_weights_path="yolov8s.pt",   # YOLO ağırlığının path'ini burada güncelle
        vit_name="vit_base_patch16_224",
        num_classes=3,
        freeze_backbones=True
    ).to(device)
    return model


def train_one_epoch(model, train_loader, optimizer, criterion, epoch):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

    print(f"Epoch {epoch:03d} | Train Loss: {total_loss/total:.4f} | "
          f"Acc: {correct/total:.4f}")


@torch.no_grad()
def eval_one_epoch(model, val_loader, criterion, epoch):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)

        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

    print(f"Epoch {epoch:03d} | Valid   Loss: {total_loss/total:.4f} | "
          f"Acc: {correct/total:.4f}")


if __name__ == "__main__":
    batch_size = 16  # GPU durumuna göre 8/16/32 yapabilirsin

    train_loader, val_loader = get_loaders(data_root, batch_size=batch_size)

    model = create_model()

    # --- ÖNEMLİ EK: Classifier'ı inşa etmek için bir kez forward yap ---
    model.eval()
    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224).to(device)
        _ = model(dummy)
    # --------------------------------------------------------------

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4
    )

    num_epochs = 15
    for epoch in range(1, num_epochs + 1):
        train_one_epoch(model, train_loader, optimizer, criterion, epoch)
        eval_one_epoch(model, val_loader, criterion, epoch)

    torch.save(model.state_dict(), "yolo_vit_fusion.pth")
    print("Model kaydedildi: yolo_vit_fusion.pth")

