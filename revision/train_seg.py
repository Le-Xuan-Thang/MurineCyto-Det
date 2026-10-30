"""Semantic-segmentation benchmark for the CSBJ revision (U-Net / SegFormer).

Trains on random 256x256 crops, selects the checkpoint with the best validation
foreground mIoU, and saves full-resolution (1024x1024) test predictions as uint8 label
maps for revision/evaluate_seg.py.

Usage: python revision/train_seg.py --arch Unet --encoder resnet34 --scheme mouse_holdout --seed 0 --gpu 0
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataloader.dataloaders import Dataset, get_training_augmentation, get_validation_augmentation  # noqa: E402
from revision.data_paths import data_root, run_root, require_semantic_data

NUM_CLASSES = 6  # background + 5 cell classes


class SegModel(torch.nn.Module):
    def __init__(self, arch, encoder):
        super().__init__()
        self.model = smp.create_model(arch, encoder_name=encoder, encoder_weights="imagenet",
                                      in_channels=3, classes=NUM_CLASSES)
        p = smp.encoders.get_preprocessing_params(encoder)
        self.register_buffer("mean", torch.tensor(p["mean"]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(p["std"]).view(1, 3, 1, 1))

    def forward(self, x):
        return self.model((x.float() / 255.0 - self.mean) / self.std)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_split(data_dir, scheme, part):
    return (Path(data_dir) / "splits" / f"{scheme}_{part}.txt").read_text().split()


def fg_miou(conf):
    inter = np.diag(conf)
    union = conf.sum(0) + conf.sum(1) - inter
    iou = inter / np.maximum(union, 1)
    return iou[1:].mean()


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    preds, gts = [], []
    for img, mask in loader:
        logits = model(img.to(device))
        preds.append(logits.argmax(1).cpu().numpy().astype(np.uint8))
        gts.append(mask.numpy().astype(np.uint8))
    return np.concatenate(preds), np.concatenate(gts)


def confusion(pred, gt):
    return np.bincount(gt.ravel().astype(np.int64) * NUM_CLASSES + pred.ravel(),
                       minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="Unet")
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--scheme", default="mouse_holdout")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--data-dir", default="data_v2")
    ap.add_argument("--out-dir", default="runs_revision/seg")
    args = ap.parse_args()

    args.data_dir = str(data_root(args.data_dir))
    args.out_dir = str(run_root(args.out_dir))
    require_semantic_data(args.data_dir)
    seed_all(args.seed)
    device = torch.device(f"cuda:{args.gpu}")
    name = f"{args.arch}-{args.encoder}_{args.scheme}_s{args.seed}"
    out = Path(args.out_dir) / name
    out.mkdir(parents=True, exist_ok=True)

    split = {p: read_split(args.data_dir, args.scheme, p) for p in ("train", "val", "test")}
    mk = lambda files, aug: Dataset(args.data_dir, files, files, augmentation=aug)  # noqa: E731
    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(mk(split["train"], get_training_augmentation()), batch_size=args.batch_size,
                              shuffle=True, num_workers=8, generator=g, drop_last=True,
                              worker_init_fn=lambda w: np.random.seed(args.seed * 100 + w))
    val_loader = DataLoader(mk(split["val"], get_validation_augmentation()), batch_size=4, num_workers=4)
    test_loader = DataLoader(mk(split["test"], get_validation_augmentation()), batch_size=4, num_workers=4)

    model = SegModel(args.arch, args.encoder).to(device)
    loss_fn = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps = args.epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-6)

    best, best_epoch, log = -1.0, -1, []
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        tot = 0.0
        for img, mask in train_loader:
            img, mask = img.to(device), mask.to(device).long()
            loss = loss_fn(model(img), mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        vp, vg = predict(model, val_loader, device)
        miou = fg_miou(confusion(vp, vg))
        log.append({"epoch": epoch, "train_loss": tot / len(train_loader), "val_fg_miou": float(miou)})
        print(f"{name} ep{epoch} loss={tot / len(train_loader):.4f} val_fg_mIoU={miou:.4f}", flush=True)
        if miou > best:
            best, best_epoch = miou, epoch
            torch.save(model.state_dict(), out / "best.pth")
        elif epoch - best_epoch >= args.patience:
            print(f"early stop at epoch {epoch}")
            break

    model.load_state_dict(torch.load(out / "best.pth", map_location=device))
    tp, tg = predict(model, test_loader, device)
    np.savez_compressed(out / "test_pred.npz", pred=tp, files=np.array(split["test"]))
    cfg = vars(args) | {"best_epoch": best_epoch, "best_val_fg_miou": float(best),
                        "train_minutes": (time.time() - t0) / 60, "torch": torch.__version__,
                        "smp": smp.__version__, "loss": "Dice (multiclass, incl. background)",
                        "optimizer": "Adam", "scheduler": "cosine per-iteration to 1e-6",
                        "train_input": "random 256x256 crops", "test_input": "full 1024x1024 tile"}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    (out / "log.json").write_text(json.dumps(log))
    print(f"done {name}: best epoch {best_epoch}, val fg-mIoU {best:.4f}")


if __name__ == "__main__":
    main()
