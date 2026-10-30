"""Detection / instance-segmentation benchmark for MurineCyto-Det.

Models
  fasterrcnn : torchvision Faster R-CNN ResNet50-FPN (COCO init), native 1024x1024 input,
               small anchors (16-256 px), AdamW + warmup + cosine, best val AP checkpoint
  maskrcnn   : same recipe with a mask head, trained on the v2 instance masks
  yolov8     : Ultralytics YOLOv8s, 1024 px input

Every model writes COCO-format predictions for val and test (score threshold 0.001,
max 300 per image) so that scripts/evaluate_det.py scores all models identically.

Usage: python scripts/train_det.py --model fasterrcnn --scheme mouse_holdout --seed 0 --gpu 1
"""

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision import tv_tensors
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import v2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from murinecyto_det.paths import data_root, run_root, annotation_file, require_semantic_data

NUM_CLASSES = 5


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, ann_file, train, with_masks):
        self.coco = COCO(str(ann_file))
        self.ids = sorted(self.coco.imgs)
        self.img_dir = Path(data_dir) / "images"
        self.with_masks = with_masks
        aug = [v2.RandomHorizontalFlip(), v2.RandomVerticalFlip(),
               v2.RandomApply([v2.RandomRotation(degrees=(90, 90))], p=0.5),
               v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03)] if train else []
        self.tf = v2.Compose([v2.ToImage(), *aug, v2.ToDtype(torch.float32, scale=True),
                              v2.SanitizeBoundingBoxes()])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img_id = self.ids[i]
        info = self.coco.imgs[img_id]
        img = Image.open(self.img_dir / info["file_name"]).convert("RGB")
        anns = self.coco.imgToAnns[img_id]
        boxes = [[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in anns]
        target = {"boxes": tv_tensors.BoundingBoxes(torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                                                    format="XYXY", canvas_size=(info["height"], info["width"])),
                  "labels": torch.tensor([a["category_id"] for a in anns], dtype=torch.int64)}
        if self.with_masks:
            m = np.stack([mask_utils.decode(a["segmentation"]) for a in anns]) if anns else np.zeros((0, info["height"], info["width"]), np.uint8)
            target["masks"] = tv_tensors.Mask(torch.from_numpy(m))
        img, target = self.tf(img, target)
        target["image_id"] = img_id
        return img, target


def collate(batch):
    return tuple(zip(*batch))


def build_tv_model(kind):
    anchors = AnchorGenerator(sizes=((16,), (32,), (64,), (128,), (256,)), aspect_ratios=((0.5, 1.0, 2.0),) * 5)
    common = dict(min_size=1024, max_size=1024, rpn_anchor_generator=anchors, box_detections_per_img=300,
                  box_score_thresh=0.001, rpn_pre_nms_top_n_train=3000, rpn_post_nms_top_n_train=2000,
                  rpn_pre_nms_top_n_test=2000, rpn_post_nms_top_n_test=1500, box_batch_size_per_image=512)
    if kind == "fasterrcnn":
        m = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="COCO_V1", **common)
    else:
        m = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="COCO_V1", **common)
        m.roi_heads.mask_predictor = MaskRCNNPredictor(256, 256, NUM_CLASSES + 1)
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, NUM_CLASSES + 1)
    return m


@torch.inference_mode()
def tv_predict(model, loader, device, with_masks):
    model.eval()
    res = []
    for imgs, tgts in loader:
        outs = model([im.to(device) for im in imgs])
        for o, t in zip(outs, tgts):
            boxes = o["boxes"].cpu().numpy()
            for k in range(len(boxes)):
                x0, y0, x1, y1 = boxes[k].tolist()
                r = {"image_id": int(t["image_id"]), "category_id": int(o["labels"][k]),
                     "bbox": [x0, y0, x1 - x0, y1 - y0], "score": float(o["scores"][k])}
                if with_masks:
                    m = (o["masks"][k, 0] > 0.5).cpu().numpy().astype(np.uint8)
                    rle = mask_utils.encode(np.asfortranarray(m))
                    rle["counts"] = rle["counts"].decode("ascii")
                    r["segmentation"] = rle
                res.append(r)
    return res


def coco_ap(ann_file, results, iou_type="bbox"):
    gt = COCO(str(ann_file))
    if not results:
        return 0.0
    dt = gt.loadRes(results)
    e = COCOeval(gt, dt, iou_type)
    e.params.maxDets = [1, 100, 300]
    e.evaluate()
    e.accumulate()
    e.summarize()
    # Preserve the completed experiment: checkpoint selection used AP50:95@100.
    pr = e.eval["precision"][:, :, :, 0, e.params.maxDets.index(100)]
    return float(pr[pr > -1].mean())


def train_tv(args, out, device):
    with_masks = args.model == "maskrcnn"
    ann = lambda p: annotation_file(args.data_dir, args.scheme, p, args.annotation_set)  # noqa: E731
    g = torch.Generator().manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(CocoDataset(args.data_dir, ann("train"), True, with_masks),
                                               batch_size=args.batch_size, shuffle=True, num_workers=8,
                                               collate_fn=collate, generator=g, drop_last=True)
    eval_loader = lambda p: torch.utils.data.DataLoader(CocoDataset(args.data_dir, ann(p), False, with_masks),  # noqa: E731
                                                        batch_size=2, num_workers=4, collate_fn=collate)
    model = build_tv_model(args.model).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    total = args.epochs * len(train_loader)
    warm = min(500, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + np.cos(np.pi * (s - warm) / max(total - warm, 1))))
    best, best_epoch, log = -1.0, -1, []
    iou_type = "segm" if with_masks else "bbox"
    for epoch in range(args.epochs):
        model.train()
        tot = 0.0
        for imgs, tgts in train_loader:
            imgs = [im.to(device) for im in imgs]
            tgts = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()} for t in tgts]
            loss = sum(model(imgs, tgts).values())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
            sched.step()
            tot += loss.item()
        row = {"epoch": epoch, "train_loss": tot / len(train_loader)}
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            ap = coco_ap(ann("val"), tv_predict(model, eval_loader("val"), device, with_masks), iou_type)
            row["val_ap"] = ap
            if ap > best:
                best, best_epoch = ap, epoch
                torch.save(model.state_dict(), out / "best.pth")
        log.append(row)
        print(f"{out.name} {row}", flush=True)
    model.load_state_dict(torch.load(out / "best.pth", map_location=device))
    for p in ("val", "test"):
        (out / f"{p}_pred.json").write_text(json.dumps(tv_predict(model, eval_loader(p), device, with_masks)))
    return {"best_epoch": best_epoch, "best_val_ap": best, "log": log,
            "optimizer": "AdamW wd 1e-4", "scheduler": f"linear warmup {warm} it + cosine",
            "input": "1024x1024 (no resize)", "anchors": "16,32,64,128,256 x (0.5,1,2)",
            "init": "COCO-pretrained (torchvision COCO_V1)",
            "augmentation": "h/v flip, 90-degree rotation, colour jitter (b/c/s 0.2, hue 0.03)"}


def train_yolo(args, out, device_str):
    from ultralytics import YOLO
    import ultralytics
    root = out / "yolo_data"
    if root.exists():
        shutil.rmtree(root)
    for p in ("train", "val", "test"):
        coco = COCO(str(annotation_file(args.data_dir, args.scheme, p, args.annotation_set)))
        (root / "images" / p).mkdir(parents=True)
        (root / "labels" / p).mkdir(parents=True)
        for img_id, info in coco.imgs.items():
            src = (Path(args.data_dir) / "images" / info["file_name"]).resolve()
            (root / "images" / p / info["file_name"]).symlink_to(src)
            lines = []
            for a in coco.imgToAnns[img_id]:
                x, y, w, h = a["bbox"]
                lines.append(f"{a['category_id'] - 1} {(x + w / 2) / 1024:.6f} {(y + h / 2) / 1024:.6f} {w / 1024:.6f} {h / 1024:.6f}")
            (root / "labels" / p / info["file_name"].replace(".png", ".txt")).write_text("\n".join(lines) + "\n")
    names = json.load(open(annotation_file(args.data_dir, args.scheme, "train", args.annotation_set)))["categories"]
    yaml = root / "data.yaml"
    yaml.write_text(f"path: {root.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
                    + "".join(f"  {c['id'] - 1}: {c['name']}\n" for c in names))
    model = YOLO(args.yolo_weights)
    model.train(data=str(yaml), epochs=args.epochs, imgsz=1024, batch=args.batch_size, device=device_str,
                seed=args.seed, deterministic=True, optimizer="AdamW", lr0=args.lr, patience=args.patience,
                project=str(out.resolve()), name="ultra", exist_ok=True, workers=8, plots=True, verbose=False)
    best = YOLO(str(out / "ultra" / "weights" / "best.pt"))
    for p in ("val", "test"):
        coco = COCO(str(annotation_file(args.data_dir, args.scheme, p, args.annotation_set)))
        res = []
        for img_id, info in coco.imgs.items():
            r = best.predict(str(Path(args.data_dir) / "images" / info["file_name"]), imgsz=1024, conf=0.001,
                             iou=0.7, max_det=300, device=device_str, verbose=False)[0]
            for b, c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                res.append({"image_id": img_id, "category_id": int(c) + 1,
                            "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])], "score": float(s)})
        (out / f"{p}_pred.json").write_text(json.dumps(res))
    return {"ultralytics": ultralytics.__version__, "weights": args.yolo_weights, "imgsz": 1024,
            "optimizer": "AdamW", "nms_iou": 0.7, "augmentation": "Ultralytics defaults (mosaic, HSV, flip, scale, translate)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["fasterrcnn", "maskrcnn", "yolov8"], required=True)
    ap.add_argument("--scheme", default="mouse_holdout")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--yolo-weights", default="yolov8s.pt")
    ap.add_argument("--data-dir", default="data_v2")
    ap.add_argument("--annotation-set", choices=["benchmark", "original"], default="benchmark")
    ap.add_argument("--out-dir", default="runs_revision/det")
    args = ap.parse_args()
    if args.epochs is None:
        args.epochs = 150 if args.model == "yolov8" else 60
    if args.lr is None:
        args.lr = 1e-3 if args.model == "yolov8" else 1e-4

    if args.annotation_set == "original" and args.out_dir == "runs_revision/det":
        args.out_dir = "runs_revision/det_original_masks"
    args.data_dir = str(data_root(args.data_dir))
    args.out_dir = str(run_root(args.out_dir))
    require_semantic_data(args.data_dir)
    seed_all(args.seed)
    out = Path(args.out_dir) / f"{args.model}_{args.scheme}_s{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if args.model == "yolov8":
        extra = train_yolo(args, out, args.gpu)
    else:
        extra = train_tv(args, out, torch.device(f"cuda:{args.gpu}"))
    cfg = vars(args) | extra | {"train_minutes": (time.time() - t0) / 60, "torch": torch.__version__,
                                "torchvision": torchvision.__version__}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    print("done", out)


if __name__ == "__main__":
    main()
