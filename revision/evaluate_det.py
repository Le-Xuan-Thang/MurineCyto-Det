"""Evaluate detection / instance-segmentation runs with one protocol (Reviewer 2 #4-#5, Reviewer 3 #7, #12).

For every runs_revision/det/*/test_pred.json:
  * COCO AP, AP50, AP75, AP_small/medium (pycocotools, maxDets 300) for boxes, and for
    masks when the run has them (Mask R-CNN);
  * per-class AP50 and AP50:95, with 95% bootstrap CI for AP50 over test tiles;
  * precision / recall / F1 per class at IoU 0.5 and a confidence threshold chosen on
    the validation set (threshold maximising macro-F1), plus their macro averages;
  * object-level confusion matrix at that threshold (GT class x predicted class, with
    "missed" column and "false positive" row), absolute counts;
  * per-class precision-recall curves (IoU 0.5).
Writes runs_revision/results/det_*.csv.
"""

import contextlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from revision.data_paths import data_root, run_root, annotation_file

CLASSES = ["Macrophage/Monocyte", "Neutrophil", "Eosinophil", "Lymphocyte", "Unknown cell/Debris"]
N_BOOT = 1000


def iou_matrix(a, b):
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    ax1, ay1, bx1, by1 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax1[:, None], bx1) - np.maximum(a[:, None, 0], b[:, 0]), 0, None)
    ih = np.clip(np.minimum(ay1[:, None], by1) - np.maximum(a[:, None, 1], b[:, 1]), 0, None)
    inter = iw * ih
    return inter / (a[:, 2:].prod(1)[:, None] + b[:, 2:].prod(1) - inter + 1e-9)


def coco_eval(gt, dets, iou_type):
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(dets)
        e = COCOeval(gt, dt, iou_type)
        e.params.maxDets = [1, 100, 300]
        e.evaluate()
        e.accumulate()
        e.summarize()
    prec = e.eval["precision"]  # [T, R, K, A, M]
    per_cls = {}
    for k, cid in enumerate(e.params.catIds):
        p = prec[:, :, k, 0, e.params.maxDets.index(100)]
        per_cls[cid] = {"AP": float(np.mean(p[p > -1])) if (p > -1).any() else np.nan,
                        "AP50": float(np.mean(prec[0, :, k, 0, -1][prec[0, :, k, 0, -1] > -1])) if (prec[0, :, k, 0, -1] > -1).any() else np.nan,
                        "pr50": prec[0, :, k, 0, -1]}
    s = e.stats
    ap100 = prec[:, :, :, 0, e.params.maxDets.index(100)]
    ap100 = float(ap100[ap100 > -1].mean())
    return {"AP": ap100, "AP50": s[1], "AP75": s[2], "AP_small": s[3], "AP_medium": s[4]}, per_cls


def per_image_matches(gt, dets, iou_thr=0.5):
    """Greedy class-wise matching per image -> {img: {cid: (scores, tp_flags, n_gt)}}."""
    by_img = {}
    for d in dets:
        by_img.setdefault(d["image_id"], []).append(d)
    out = {}
    for img_id in gt.imgs:
        anns = gt.imgToAnns[img_id]
        out[img_id] = {}
        for cid in range(1, 6):
            g = [a["bbox"] for a in anns if a["category_id"] == cid]
            d = sorted([x for x in by_img.get(img_id, []) if x["category_id"] == cid], key=lambda x: -x["score"])
            sc = np.array([x["score"] for x in d])
            tp = np.zeros(len(d), bool)
            if g and d:
                ious = iou_matrix([x["bbox"] for x in d], g)
                used = np.zeros(len(g), bool)
                for i in range(len(d)):
                    cand = np.where(~used & (ious[i] >= iou_thr))[0]
                    if len(cand):
                        j = cand[np.argmax(ious[i, cand])]
                        used[j] = tp[i] = True
            out[img_id][cid] = (sc, tp, len(g))
    return out


def ap101(sc, tp, n_gt):
    if n_gt == 0:
        return np.nan
    if len(sc) == 0:
        return 0.0
    o = np.argsort(-sc, kind="mergesort")
    tp = tp[o]
    ctp, cfp = np.cumsum(tp), np.cumsum(~tp)
    rec, prec = ctp / n_gt, ctp / np.maximum(ctp + cfp, 1)
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    rs = np.linspace(0, 1, 101)
    idx = np.searchsorted(rec, rs, side="left")
    return float(np.mean([prec[i] if i < len(prec) else 0.0 for i in idx]))


def prf_at(matches, thr):
    rows = {}
    for cid in range(1, 6):
        tp = fp = ngt = 0
        for m in matches.values():
            sc, t, n = m[cid]
            keep = sc >= thr
            tp += int(t[keep].sum())
            fp += int((~t[keep]).sum())
            ngt += n
        p = tp / max(tp + fp, 1)
        r = tp / max(ngt, 1)
        rows[cid] = {"P": p, "R": r, "F1": 2 * p * r / max(p + r, 1e-12), "TP": tp, "FP": fp, "FN": ngt - tp, "n_gt": ngt}
    return rows


def object_confusion(gt, dets, thr, iou_thr=0.5):
    """Class-agnostic matching at IoU>=0.5 -> confusion GT class x predicted class."""
    cm = np.zeros((6, 6), dtype=np.int64)  # index 0 = missed (col) / false positive (row)
    by_img = {}
    for d in dets:
        if d["score"] >= thr:
            by_img.setdefault(d["image_id"], []).append(d)
    for img_id in gt.imgs:
        anns = gt.imgToAnns[img_id]
        d = sorted(by_img.get(img_id, []), key=lambda x: -x["score"])
        used = np.zeros(len(anns), bool)
        if anns and d:
            ious = iou_matrix([x["bbox"] for x in d], [a["bbox"] for a in anns])
        for i, x in enumerate(d):
            j = -1
            if anns:
                cand = np.where(~used & (ious[i] >= iou_thr))[0]
                if len(cand):
                    j = cand[np.argmax(ious[i, cand])]
            if j >= 0:
                used[j] = True
                cm[anns[j]["category_id"], x["category_id"]] += 1
            else:
                cm[0, x["category_id"]] += 1
        for j, a in enumerate(anns):
            if not used[j]:
                cm[a["category_id"], 0] += 1
    return cm


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
    res = run_root(args.results_dir) if args.results_dir else root / ("results" if args.annotation_set == "benchmark" else "results_original_masks")
    res.mkdir(parents=True, exist_ok=True)
    data = data_root(args.data_dir)
    summ, pcls, cms, prs = [], [], [], []
    for run in sorted((root / "det").glob("*/test_pred.json")):
        name = run.parent.name
        model, seed = name.split("_")[0], name.split("_")[-1]
        scheme = "_".join(name.split("_")[1:-1])
        if args.scheme != "all" and scheme != args.scheme:
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            gt_val = COCO(str(annotation_file(data, scheme, "val", args.annotation_set)))
            gt_test = COCO(str(annotation_file(data, scheme, "test", args.annotation_set)))
        dt_test = json.loads(run.read_text())
        dt_val = json.loads((run.parent / "val_pred.json").read_text())

        # confidence threshold chosen on validation (max macro-F1)
        mv = per_image_matches(gt_val, dt_val)
        grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
        f1s = [np.mean([v["F1"] for v in prf_at(mv, t).values()]) for t in grid]
        thr = float(grid[int(np.argmax(f1s))])

        mt = per_image_matches(gt_test, dt_test)
        prf = prf_at(mt, thr)
        rng = np.random.default_rng(0)
        imgs = list(mt)
        boot = np.zeros((N_BOOT, 5))
        for b in range(N_BOOT):
            sel = [imgs[i] for i in rng.integers(0, len(imgs), len(imgs))]
            for k, cid in enumerate(range(1, 6)):
                sc = np.concatenate([mt[i][cid][0] for i in sel])
                tp = np.concatenate([mt[i][cid][1] for i in sel])
                boot[b, k] = ap101(sc, tp, sum(mt[i][cid][2] for i in sel))
        lo, hi = np.nanpercentile(boot, 2.5, axis=0), np.nanpercentile(boot, 97.5, axis=0)

        for iou_type in (["bbox", "segm"] if "segmentation" in (dt_test[0] if dt_test else {}) else ["bbox"]):
            overall, per = coco_eval(gt_test, dt_test, iou_type)
            row = {"run": name, "model": model, "scheme": scheme, "seed": seed, "iou_type": iou_type, **overall}
            if iou_type == "bbox":
                row |= {"conf_thr": thr, "macro_P": np.mean([v["P"] for v in prf.values()]),
                        "macro_R": np.mean([v["R"] for v in prf.values()]),
                        "macro_F1": np.mean([v["F1"] for v in prf.values()])}
            summ.append(row)
            for k, cid in enumerate(range(1, 6)):
                r = {"run": name, "model": model, "scheme": scheme, "seed": seed, "iou_type": iou_type,
                     "class": CLASSES[cid - 1], "AP50": per[cid]["AP50"], "AP": per[cid]["AP"]}
                if iou_type == "bbox":
                    r |= {"AP50_lo": lo[k], "AP50_hi": hi[k], **prf[cid]}
                    prs.extend({"run": name, "class": CLASSES[cid - 1], "recall": rr, "precision": pp}
                               for rr, pp in zip(np.linspace(0, 1, 101), per[cid]["pr50"]))
                pcls.append(r)
        cm = object_confusion(gt_test, dt_test, thr)
        labels = ["None (missed / FP)"] + CLASSES
        for i in range(6):
            cms.append({"run": name, "gt": labels[i], **{labels[j]: int(cm[i, j]) for j in range(6)}})
        print(name, {k: round(v, 3) for k, v in summ[-1].items() if isinstance(v, float)})

    if not summ:
        raise FileNotFoundError(f"No matching detection test predictions under {root / 'det'}")
    s = pd.DataFrame(summ)
    s.to_csv(res / "det_summary.csv", index=False)
    pd.DataFrame(pcls).to_csv(res / "det_per_class.csv", index=False)
    pd.DataFrame(cms).to_csv(res / "det_object_confusion.csv", index=False)
    pd.DataFrame(prs).to_csv(res / "det_pr_curves.csv", index=False)
    num = [c for c in ["AP", "AP50", "AP75", "AP_small", "AP_medium", "macro_P", "macro_R", "macro_F1"] if c in s]
    agg = s.groupby(["model", "scheme", "iou_type"])[num].agg(["mean", "std"])
    agg.to_csv(res / "det_summary_agg.csv")
    print(agg.round(3).to_string())
    pc = pd.DataFrame(pcls)
    pagg = pc.groupby(["model", "scheme", "iou_type", "class"])[["AP50", "AP", "P", "R", "F1"]].agg(["mean", "std"])
    pagg.to_csv(res / "det_per_class_agg.csv")
    print(pagg.round(3).to_string())


if __name__ == "__main__":
    main()
