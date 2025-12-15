import os
import torch
import torch.nn as nn
from ultralytics import YOLO
import timm

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# Buraya istersen kendi eğittiğin YOLOv8s ağırlığının yolunu yaz:
YOLO_WEIGHTS = os.path.join(ROOT_DIR, "yolov8s.pt")
# örnek: YOLO_WEIGHTS = os.path.join(ROOT_DIR, "runs", "fire_smoke_yolov8s_cmp", "weights", "best.pt")


class YoloVitFusionNet(nn.Module):
    def __init__(
            self,
            yolo_weights_path=YOLO_WEIGHTS,
            vit_name="vit_base_patch16_224",
            num_classes=3,
            freeze_backbones=True
    ):
        super().__init__()

        # 1) YOLOv8s DETECTION modelini yükle
        yolo = YOLO(yolo_weights_path)
        self.yolo_model = yolo.model  # Ultralytics DetectionModel

        # ---- YOLO BACKBONE SEÇİMİ ----
        # NOT: [:10] dilimi senin YOLO versiyonuna göre küçük değişebilir.
        # Gerekirse önce ayrı bir scriptte print(self.yolo_model.model) deyip bakabilirsin.
        backbone_layers = list(self.yolo_model.model)[:10]
        self.yolo_backbone = nn.Sequential(*backbone_layers)

        # Backbone çıkış kanal sayısını ilk forward sırasında belirleyeceğiz
        self.yolo_out_dim = None
        self.yolo_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 2) ViT-Base modeli (timm)
        self.vit = timm.create_model(vit_name, pretrained=True)
        vit_out_dim = self.vit.num_features
        # Klasik classification head'i kaldır, sadece embedding alalım
        if hasattr(self.vit, "head"):
            self.vit.head = nn.Identity()
        elif hasattr(self.vit, "heads"):
            self.vit.heads = nn.Identity()

        # 3) Fusion classifier head
        self.classifier = None
        self.num_classes = num_classes

        # Backbone’ları dondurmak istersek:
        if freeze_backbones:
            for p in self.yolo_backbone.parameters():
                p.requires_grad = False
            for p in self.vit.parameters():
                p.requires_grad = False

    def _build_classifier_if_needed(self, yolo_feat):
        if self.yolo_out_dim is None:
            self.yolo_out_dim = yolo_feat.shape[1]
            fusion_dim = self.yolo_out_dim + self.vit.num_features
            self.classifier = nn.Sequential(
                nn.Linear(fusion_dim, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(512, self.num_classes)
            )

    def forward(self, x):
        """
        x: (B, 3, 224, 224) normalize edilmiş görüntü
        """
        # YOLO backbone → feature map
        yolo_feat_map = self.yolo_backbone(x)   # (B, C, H, W)
        yolo_pooled = self.yolo_pool(yolo_feat_map).flatten(1)  # (B, C)

        # ViT → embedding
        vit_feat = self.vit(x)                  # (B, D)

        # Classifier yoksa (ilk forward) oluştur
        self._build_classifier_if_needed(yolo_pooled)

        fused = torch.cat([yolo_pooled, vit_feat], dim=1)  # (B, C+D)
        logits = self.classifier(fused)
        return logits


# Test: direkt bu dosyayı çalıştırınca shape görelim
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YoloVitFusionNet(num_classes=3).to(device)
    x = torch.randn(2, 3, 224, 224).to(device)
    with torch.no_grad():
        out = model(x)
    print("Çıkış şekli:", out.shape)  # torch.Size([2, 3]) bekleniyor
