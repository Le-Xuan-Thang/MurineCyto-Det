"""Evaluate semantic-segmentation runs (Reviewer 1 #1-#2, Reviewer 3 #4-#5, Reviewer 2 b/c).

For every run in runs_revision/seg/*/test_pred.npz:
  * pixel confusion matrix with background row/column, absolute counts;
  * PA, per-class IoU and Dice (background included and reported separately),
    foreground mIoU, foreground macro-Dice, Generalized Dice (Sudre et al. 2017 weights);
  * Boundary F1 per class (tolerance 2 px = 0.46 um);
  * object-level confusion: for every ground-truth instance, the majority predicted label
    over its pixels (background counts as "missed");
  * 95% bootstrap CI over test tiles (1000 resamples) for per-class IoU / Dice.
Writes runs_revision/results/seg_*.csv and seg_summary.json.
"""

import json
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torchmetrics.segmentation import GeneralizedDiceScore
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from revision.data_paths import data_root, run_root, annotation_file, require_semantic_data

CLASSES = ["Background", "Macrophage/Monocyte", "Neutrophil", "Eosinophil", "Lymphocyte", "Unknown cell/Debris"]
K = len(CLASSES)
TOL = 2
N_BOOT = 1000


def cm(gt, pred):
    return np.bincount(gt.ravel().astype(np.int64) * K + pred.ravel(), minlength=K * K).reshape(K, K)


def scores(c):
    tp = np.diag(c).astype(float)
    gt, pr = c.sum(1), c.sum(0)
    iou = tp / np.maximum(gt + pr - tp, 1)
    dice = 2 * tp / np.maximum(gt + pr, 1)
    w = 1 / np.maximum(gt.astype(float), 1) ** 2
    gds = 2 * (w * tp).sum() / (w * (gt + pr)).sum()
    return iou, dice, tp.sum() / c.sum(), gds


def boundary(m):
    return cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0


def boundary_counts(gt, pred):
    """Per class: (#pred boundary px within TOL of gt boundary, #pred bd px, #gt bd px within TOL of pred, #gt bd px)."""
    out = np.zeros((K, 4))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * TOL + 1, 2 * TOL + 1))
    for k in range(1, K):
        bg, bp = boundary(gt == k), boundary(pred == k)
        if not bg.any() and not bp.any():
            continue
        dg = cv2.dilate(bg.astype(np.uint8), ker) > 0
        dp = cv2.dilate(bp.astype(np.uint8), ker) > 0
        out[k] = [(bp & dg).sum(), bp.sum(), (bg & dp).sum(), bg.sum()]
    return out


def bf1(b):
    p = b[:, 0] / np.maximum(b[:, 1], 1)
    r = b[:, 2] / np.maximum(b[:, 3], 1)
    return 2 * p * r / np.maximum(p + r, 1e-12)


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", default="runs_revision")
    ap.add_argument("--data-dir", default="data_v2")
    ap.add_argument("--annotation-set", choices=["benchmark", "original"], default="benchmark")
    ap.add_argument("--scheme", choices=["mouse_holdout", "random_tile", "all"], default="mouse_holdout")
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args()
    root = run_root(args.runs_dir)
    res_dir = run_root(args.results_dir) if args.results_dir else root / ("results" if args.annotation_set == "benchmark" else "results_original_masks")
    res_dir.mkdir(parents=True, exist_ok=True)
    data = data_root(args.data_dir)
    require_semantic_data(data)
    coco_cache = {}
    rows, cm_rows, obj_rows, summary = [], [], [], {}
    for run in sorted((root / "seg").glob("*/test_pred.npz")):
        name = run.parent.name
        arch, scheme, seed = re.match(r"(.+)_(mouse_holdout|random_tile)_(s\d+)$", name).groups()
        if args.scheme != "all" and scheme != args.scheme:
            continue
        z = np.load(run)
        preds, files = z["pred"], list(z["files"])
        if scheme not in coco_cache:
            coco_cache[scheme] = COCO(str(annotation_file(data, scheme, "test", args.annotation_set)))
        coco = coco_cache[scheme]
        fid = {im["file_name"]: i for i, im in coco.imgs.items()}
        per_img_cm, per_img_bd = [], []
        obj = np.zeros((K, K), dtype=np.int64)
        for f, p in zip(files, preds):
            g = cv2.imread(str(data / "masks_grayscale" / f), cv2.IMREAD_GRAYSCALE)
            per_img_cm.append(cm(g, p))
            per_img_bd.append(boundary_counts(g, p))
            for a in coco.imgToAnns[fid[f]]:
                m = mask_utils.decode(a["segmentation"]).astype(bool)
                if m.any():
                    obj[a["category_id"], np.bincount(p[m], minlength=K).argmax()] += 1
        per_img_cm, per_img_bd = np.stack(per_img_cm), np.stack(per_img_bd)
        c = per_img_cm.sum(0)
        iou, dice, pa, _ = scores(c)
        gts = np.stack([cv2.imread(str(data / "masks_grayscale" / f), cv2.IMREAD_GRAYSCALE) for f in files])
        gds_metric = GeneralizedDiceScore(num_classes=K, include_background=True, input_format="index")
        gds = float(gds_metric(torch.from_numpy(preds).long(), torch.from_numpy(gts).long()))
        bf = bf1(per_img_bd.sum(0))
        rng = np.random.default_rng(0)
        boot = []
        for _ in range(N_BOOT):
            idx = rng.integers(0, len(files), len(files))
            bi, bd, _, _ = scores(per_img_cm[idx].sum(0))
            boot.append(np.concatenate([bi, bd]))
        lo, hi = np.percentile(np.array(boot), [2.5, 97.5], axis=0)
        n_inst = obj.sum(1)
        for k, cls in enumerate(CLASSES):
            rows.append({"run": name, "model": arch, "scheme": scheme, "seed": seed, "class": cls,
                         "IoU": iou[k], "IoU_lo": lo[k], "IoU_hi": hi[k],
                         "Dice": dice[k], "Dice_lo": lo[K + k], "Dice_hi": hi[K + k],
                         "BF1": bf[k] if k else np.nan, "gt_pixels": int(c[k].sum()),
                         "gt_instances": int(n_inst[k]) if k else np.nan,
                         "object_recall": obj[k, k] / max(n_inst[k], 1) if k else np.nan})
        summary[name] = {"model": arch, "scheme": scheme, "seed": seed, "PA": pa, "mIoU_all": iou.mean(),
                         "mIoU_fg": iou[1:].mean(), "mDice_fg": dice[1:].mean(), "GDS": gds,
                         "mBF1_fg": float(np.nanmean(bf[1:]))}
        for i in range(K):
            cm_rows.append({"run": name, "gt": CLASSES[i], **{CLASSES[j]: int(c[i, j]) for j in range(K)}})
            obj_rows.append({"run": name, "gt": CLASSES[i], **{CLASSES[j]: int(obj[i, j]) for j in range(K)}})
        print(name, {k: round(v, 4) for k, v in summary[name].items() if isinstance(v, float)})
    if not rows:
        raise FileNotFoundError(f"No matching segmentation test predictions under {root / 'seg'}")
    pd.DataFrame(rows).to_csv(res_dir / "seg_per_class.csv", index=False)
    pd.DataFrame(cm_rows).to_csv(res_dir / "seg_pixel_confusion.csv", index=False)
    pd.DataFrame(obj_rows).to_csv(res_dir / "seg_object_confusion.csv", index=False)
    s = pd.DataFrame(summary).T
    s.to_csv(res_dir / "seg_summary.csv")
    num = ["PA", "mIoU_all", "mIoU_fg", "mDice_fg", "GDS", "mBF1_fg"]
    s[num] = s[num].astype(float)
    agg = s.groupby(["model", "scheme"])[num].agg(["mean", "std"])
    agg.to_csv(res_dir / "seg_summary_agg.csv")
    print(agg.round(3).to_string())
    pc = pd.DataFrame(rows)
    pagg = pc.groupby(["model", "scheme", "class"])[["IoU", "Dice", "BF1", "object_recall"]].agg(["mean", "std"])
    pagg.to_csv(res_dir / "seg_per_class_agg.csv")
    print(pagg.round(3).to_string())


if __name__ == "__main__":
    main()
