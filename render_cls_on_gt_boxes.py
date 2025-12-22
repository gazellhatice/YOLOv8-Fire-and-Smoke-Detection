import os, json, glob
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T

try:
    import timm
except ImportError:
    timm = None

# ----------------------------
# Helpers
# ----------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def load_summary(summary_path: Path) -> dict:
    if summary_path.exists():
        try:
            return json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def guess_timm_name_from_folder(folder_name: str) -> str | None:
    # folder examples: cls_densenet201_yolo, cls_vit_base_yolo, cls_swin_t_yolo ...
    name = folder_name.lower()
    if "mobilenet_v2" in name or "mobilenetv2" in name:
        return "mobilenetv2_100"
    if "efficientnet_b0" in name:
        return "efficientnet_b0"
    if "resnet50" in name:
        return "resnet50"
    if "densenet201" in name:
        return "densenet201"
    if "swin_t" in name or "swin-t" in name:
        return "swin_tiny_patch4_window7_224"
    if "vit_base" in name or "vit-b" in name:
        return "vit_base_patch16_224"
    return None

def strip_prefix(state_dict, prefixes=("model.", "module.")):
    out = {}
    for k, v in state_dict.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
        out[nk] = v
    return out

def load_model_from_run(run_dir: Path, device: str):
    """
    Expects:
      run_dir/
        best_*.pth
        summary.json
    """
    summary = load_summary(run_dir / "summary.json")

    # classes: try to read from summary, otherwise default order:
    class_names = summary.get("class_names") or summary.get("classes") or ["Fire", "Smoke", "None"]
    num_classes = int(summary.get("num_classes") or len(class_names) or 3)

    # img_size / mean / std:
    img_size = int(summary.get("img_size") or summary.get("image_size") or 224)
    mean = tuple(summary.get("mean") or IMAGENET_MEAN)
    std  = tuple(summary.get("std")  or IMAGENET_STD)

    # timm model name:
    timm_name = summary.get("timm_name") or summary.get("model_name") or guess_timm_name_from_folder(run_dir.name)
    if timm is None:
        raise RuntimeError("timm yüklü değil. `pip install timm`")

    if timm_name is None:
        raise RuntimeError(f"Model tipi anlaşılamadı: {run_dir.name}. summary.json içine model adı eklenmeli.")

    model = timm.create_model(timm_name, pretrained=False, num_classes=num_classes)

    # find weights
    pths = sorted(run_dir.glob("*.pth"))
    if not pths:
        raise FileNotFoundError(f"{run_dir} içinde .pth bulunamadı")
    weight_path = pths[0]

    ckpt = torch.load(weight_path, map_location="cpu")
    # handle common checkpoint formats
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        sd = ckpt["model"]
    elif isinstance(ckpt, dict):
        sd = ckpt
    else:
        raise RuntimeError("Checkpoint formatı beklenmeyen türde.")

    sd = strip_prefix(sd)
    model.load_state_dict(sd, strict=False)

    model.to(device).eval()

    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    return model, transform, class_names, timm_name, weight_path.name

def yolo_to_xyxy(line: str, w: int, h: int):
    # class xc yc bw bh
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    cls = int(float(parts[0]))
    xc, yc, bw, bh = map(float, parts[1:5])
    x1 = (xc - bw/2) * w
    y1 = (yc - bh/2) * h
    x2 = (xc + bw/2) * w
    y2 = (yc + bh/2) * h
    # clamp
    x1 = max(0, min(w-1, x1))
    y1 = max(0, min(h-1, y1))
    x2 = max(0, min(w-1, x2))
    y2 = max(0, min(h-1, y2))
    return cls, int(x1), int(y1), int(x2), int(y2)

def safe_font(size=16):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except:
        return ImageFont.load_default()

@torch.no_grad()
def predict_crop(model, transform, crop: Image.Image, device: str):
    x = transform(crop).unsqueeze(0).to(device)
    logits = model(x)
    probs = F.softmax(logits, dim=1).squeeze(0)
    conf, pred = torch.max(probs, dim=0)
    return int(pred.item()), float(conf.item())

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=str, required=True, help="cls_* klasörlerinin olduğu root (ör: runs)")
    ap.add_argument("--images", type=str, required=True, help="paper_vis/images klasörü")
    ap.add_argument("--labels", type=str, required=True, help="paper_vis/labels klasörü (YOLO txt)")
    ap.add_argument("--out", type=str, default="paper_vis/out_cls", help="çıktı klasörü")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--pad", type=float, default=0.05, help="kutuya ekstra padding oranı")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    img_dir = Path(args.images)
    lab_dir = Path(args.labels)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # choose cls model folders
    model_dirs = sorted([p for p in runs_root.iterdir() if p.is_dir() and p.name.lower().startswith("cls_")])
    if not model_dirs:
        raise RuntimeError(f"{runs_root} içinde cls_* klasörü bulunamadı.")

    images = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg")))
    if not images:
        raise RuntimeError(f"{img_dir} içinde görsel bulunamadı.")

    font = safe_font(16)

    for mdir in model_dirs:
        model, transform, class_names, arch, wname = load_model_from_run(mdir, args.device)

        model_out = out_root / mdir.name
        model_out.mkdir(parents=True, exist_ok=True)

        for img_path in images:
            stem = img_path.stem
            label_path = lab_dir / f"{stem}.txt"
            if not label_path.exists():
                continue

            img = Image.open(img_path).convert("RGB")
            W, H = img.size
            draw = ImageDraw.Draw(img)

            lines = [ln.strip() for ln in label_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            for ln in lines:
                parsed = yolo_to_xyxy(ln, W, H)
                if parsed is None:
                    continue
                gt_cls, x1, y1, x2, y2 = parsed

                # padding
                bw = x2 - x1
                bh = y2 - y1
                px = int(bw * args.pad)
                py = int(bh * args.pad)
                cx1 = max(0, x1 - px); cy1 = max(0, y1 - py)
                cx2 = min(W-1, x2 + px); cy2 = min(H-1, y2 + py)

                crop = img.crop((cx1, cy1, cx2, cy2))
                pred_cls, conf = predict_crop(model, transform, crop, args.device)

                gt_name = class_names[gt_cls] if gt_cls < len(class_names) else str(gt_cls)
                pr_name = class_names[pred_cls] if pred_cls < len(class_names) else str(pred_cls)

                # draw box + text
                draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
                text = f"GT:{gt_name}  Pred:{pr_name} ({conf:.2f})"
                tw, th = draw.textbbox((0,0), text, font=font)[2:]
                # background for readability
                draw.rectangle([x1, max(0, y1 - th - 4), x1 + tw + 6, y1], fill=(0,0,0))
                draw.text((x1 + 3, max(0, y1 - th - 2)), text, fill=(255,255,255), font=font)

            # add model label top-left
            header = f"{mdir.name} | {arch} | {wname}"
            draw.rectangle([0, 0, W, 26], fill=(0,0,0))
            draw.text((6, 4), header, fill=(255,255,255), font=font)

            save_path = model_out / f"{stem}.jpg"
            img.save(save_path, quality=95)

        print(f"[OK] {mdir.name} bitti -> {model_out}")

if __name__ == "__main__":
    main()
