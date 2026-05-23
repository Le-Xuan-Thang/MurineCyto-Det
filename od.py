"""Object detection training for murine BALF cytology cells.

Supported models
----------------
- ``faster_rcnn`` — Faster R-CNN (ResNet-50 FPN backbone, torchvision)

YOLO
----
YOLO training was done on Google Colab with the ``ultralytics`` library.
See ``notebooks/notebooks/YOLOv8.ipynb`` for the full training pipeline.

Annotation format
-----------------
Each image in ``data/images/`` has a corresponding COCO-format JSON file in
``data/annotations/`` with the same base name.  ``bbox`` fields use the COCO
convention: [x_min, y_min, width, height].

Usage
-----
    uv run python od.py --epochs 2 --batch-size 1 --model faster_rcnn

On a 4 GB GPU such as GTX 1650:
    uv run python od.py --epochs 2 --batch-size 1

"""

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import v2 as T
from tqdm import tqdm

from utils.data import make_splits

# ---------------------------------------------------------------------------
# Class definitions
# ---------------------------------------------------------------------------

CELL_CLASSES = [
    "__background__",        # index 0 reserved for background
    "Macrophage/Monocyte",   # 1
    "Neutrophil",            # 2
    "Eosinophil",            # 3
    "Lymphocyte",            # 4
    "Unknown cell/Debris",   # 5
]

# Map annotation category names to our class indices (case-insensitive prefix match)
_NAME_TO_IDX = {name.lower(): i for i, name in enumerate(CELL_CLASSES)}


def _category_name_to_idx(name: str) -> int:
    """Return class index for an annotation category name."""
    lower = name.lower()
    for key, idx in _NAME_TO_IDX.items():
        if lower.startswith(key[:6]):  # robust to minor label variations
            return idx
    return 0  # background fallback


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

IMAGE_SIZE = 512  # resize images to this square size; fits GTX 1650 (4 GB) at batch=1


class MurineCellDetectionDataset(Dataset):
    """Per-image COCO-format annotation dataset for object detection.

    Each image in ``images_dir`` has a matching JSON file in
    ``annotations_dir`` with the same stem.  Images are resized to
    ``IMAGE_SIZE × IMAGE_SIZE`` and bounding boxes are scaled accordingly.
    """

    def __init__(self, data_dir: str | Path, image_names: list[str], transforms=None):
        self.data_dir = Path(data_dir)
        self.image_names = image_names          # list of bare filenames, e.g. "P01_10_10.png"
        self.transforms = transforms
        self.images_dir = self.data_dir / "images"
        self.annotations_dir = self.data_dir / "annotations"

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int):
        import cv2
        import numpy as np

        img_name = self.image_names[idx]
        stem = Path(img_name).stem

        # -- Load image -------------------------------------------------------
        img_path = self.images_dir / img_name
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        orig_h, orig_w = img.shape[:2]
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        scale_x = IMAGE_SIZE / orig_w
        scale_y = IMAGE_SIZE / orig_h

        # Convert to float32 tensor [C, H, W] in [0, 1]
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # -- Load annotation --------------------------------------------------
        ann_path = self.annotations_dir / f"{stem}.json"
        boxes, labels = [], []
        if ann_path.exists():
            with open(ann_path) as f:
                ann = json.load(f)

            # Build category_id → class index map from the file's own category list
            cat_map = {}
            for cat in ann.get("categories", []):
                cat_map[cat["id"]] = _category_name_to_idx(cat["name"])

            for obj in ann.get("annotations", []):
                x, y, w, h = obj["bbox"]
                if w <= 0 or h <= 0:
                    continue
                # Scale COCO [x, y, w, h] to resized image, then convert to [x1, y1, x2, y2]
                x1 = x * scale_x
                y1 = y * scale_y
                x2 = (x + w) * scale_x
                y2 = (y + h) * scale_y
                boxes.append([x1, y1, x2, y2])
                labels.append(cat_map.get(obj["category_id"], 0))

        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros(0, dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor(idx),
        }

        if self.transforms:
            img_tensor, target = self.transforms(img_tensor, target)

        return img_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


def get_train_transforms():
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
    ])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_faster_rcnn(num_classes: int, pretrained: bool = False) -> torch.nn.Module:
    """Return a Faster R-CNN model with a fresh box predictor head."""
    weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device, desc="train"):
    model.train()
    total_loss = 0.0
    n_batches = 0
    progress = tqdm(loader, desc=desc, leave=False)
    for images, targets in progress:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(v for v in loss_dict.values())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        progress.set_postfix(loss=total_loss / n_batches)

    return total_loss / max(n_batches, 1)


@torch.inference_mode()
def evaluate(model, loader, device, desc="eval"):
    """Return mean-AP metrics using torchmetrics."""
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox")
    progress = tqdm(loader, desc=desc, leave=False)
    for images, targets in progress:
        images = [img.to(device) for img in images]
        preds = model(images)

        # torchmetrics expects CPU tensors
        preds_cpu = [{k: v.cpu() for k, v in p.items()} for p in preds]
        targets_cpu = [{k: v.cpu() for k, v in t.items()} for t in targets]
        metric.update(preds_cpu, targets_cpu)

    return metric.compute()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train MurineCyto object detection models.")
    parser.add_argument("--epochs", type=int, default=int(os.getenv("EPOCHS", "10")))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("BATCH_SIZE", "4")))
    parser.add_argument("--num-workers", type=int, default=int(os.getenv("NUM_WORKERS", "0")))
    parser.add_argument(
        "--model",
        choices=["faster_rcnn"],
        default="faster_rcnn",
        help="Detection model to train.",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=False,
        help="Use ImageNet-pretrained backbone weights.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import gc
    from datetime import datetime

    args = parse_args()
    run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")

    # Directories
    root_dir = Path(os.getcwd())
    data_dir = root_dir / "data"
    model_dir = root_dir / "models"
    logs_dir = root_dir / "logs"
    model_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)

    # PyTorch info
    print(f"PyTorch: {torch.__version__}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if torch.cuda.is_available():
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    # Splits (reuse image filenames from segmentation splits for consistency)
    splits = make_splits(data_dir)
    print(f"train: {len(splits['x_train'])}  val: {len(splits['x_val'])}  test: {len(splits['x_test'])}")

    num_classes = len(CELL_CLASSES)  # including background
    print(f"num_classes: {num_classes}")

    # Datasets & loaders
    train_ds = MurineCellDetectionDataset(data_dir, splits["x_train"])
    val_ds   = MurineCellDetectionDataset(data_dir, splits["x_val"])
    test_ds  = MurineCellDetectionDataset(data_dir, splits["x_test"])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_fn)

    # Model
    model = build_faster_rcnn(num_classes=num_classes, pretrained=args.pretrained).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # CSV logging
    run_log_dir = logs_dir / f"{args.model}_torch" / run_name
    run_log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_log_dir / "metrics.csv"
    fieldnames = ["epoch", "train_loss", "val_map", "val_map_50", "val_map_75"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    best_val_map = 0.0

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nTraining {args.model} for {args.epochs} epoch(s) | batch={args.batch_size} | run={run_name}\n")

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            desc=f"[{epoch+1}/{args.epochs}] train",
        )
        scheduler.step()

        val_metrics = evaluate(model, val_loader, device, desc=f"[{epoch+1}/{args.epochs}] val")
        val_map    = val_metrics["map"].item()
        val_map_50 = val_metrics["map_50"].item()
        val_map_75 = val_metrics["map_75"].item()

        print(
            f"Epoch {epoch+1}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"val_mAP={val_map:.4f}  "
            f"val_mAP@50={val_map_50:.4f}"
        )

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_map": val_map,
                "val_map_50": val_map_50,
                "val_map_75": val_map_75,
            })

        if val_map > best_val_map:
            best_val_map = val_map
            best_path = model_dir / f"{args.model}_{run_name}_best.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model": args.model,
                    "num_classes": num_classes,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_map": best_val_map,
                },
                best_path,
            )
            print(f"  → saved best checkpoint: {best_path.name}")

    # Final evaluation on test set
    print("\nEvaluating on test set …")
    test_metrics = evaluate(model, test_loader, device, desc="test")
    print(
        f"test_mAP={test_metrics['map'].item():.4f}  "
        f"test_mAP@50={test_metrics['map_50'].item():.4f}"
    )

    final_path = model_dir / f"{args.model}_{run_name}_final.pth"
    torch.save(
        {
            "epoch": args.epochs - 1,
            "model": args.model,
            "num_classes": num_classes,
            "model_state_dict": model.state_dict(),
            "test_metrics": {
                k: (v.tolist() if torch.is_tensor(v) else v)
                for k, v in test_metrics.items()
            },
        },
        final_path,
    )
    print(f"Saved final checkpoint: {final_path.name}")


if __name__ == "__main__":
    main()
