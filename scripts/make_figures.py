"""Figures and LaTeX tables for the revised manuscript (reads runs_revision/results/*.csv).

Outputs to runs_revision/figures/ and runs_revision/tables/.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
R = REPO / "runs_revision/results"
FIG = REPO / "runs_revision/figures"
TAB = REPO / "runs_revision/tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

CELL = ["Macrophage/Monocyte", "Neutrophil", "Eosinophil", "Lymphocyte", "Unknown cell/Debris"]
SHORT = {"Background": "BG", "Macrophage/Monocyte": "Mac/Mono", "Neutrophil": "Neut", "Eosinophil": "Eos",
         "Lymphocyte": "Lymph", "Unknown cell/Debris": "Other/Debris", "None (missed / FP)": "None"}
SEG_MODELS = {"Unet-resnet34": "U-Net", "Segformer-mit_b2": "SegFormer"}
DET_MODELS = {"yolov8": "YOLOv8s", "fasterrcnn": "Faster R-CNN", "maskrcnn": "Mask R-CNN"}
COLORS = {"U-Net": "#2a78d6", "SegFormer": "#eb6834", "YOLOv8s": "#1baf7a", "Faster R-CNN": "#4a3aa7",
          "Mask R-CNN": "#e34948"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dddcd6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
SCHEME = "mouse_holdout"


def fmt(m, s, d=3):
    return f"{m:.{d}f}" if np.isnan(s) else f"{m:.{d}f} $\\pm$ {s:.{d}f}"


# ------------------------------------------------------------------ confusion matrices
def plot_cm(ax, mat, rows, cols, title, row_norm=True):
    mat = np.asarray(mat, float)
    norm = mat / np.maximum(mat.sum(1, keepdims=True), 1) if row_norm else mat / mat.max()
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            cnt = f"{v / 1e6:.1f}M" if v >= 1e6 else (f"{v / 1e3:.0f}k" if v >= 1e4 else f"{int(v)}")
            txt = f"{norm[i, j]:.2f}\n({cnt})"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6,
                    color="white" if norm[i, j] > 0.6 else INK)
    ax.set_xticks(range(len(cols)), [SHORT[c] for c in cols], rotation=45, ha="right")
    ax.set_yticks(range(len(rows)), [SHORT[r] for r in rows])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title(title, fontsize=8, color=INK)
    ax.spines[:].set_visible(False)


def seg_confusion_figures():
    for kind, fname, unit in [("pixel", "seg_pixel_confusion.csv", "pixels"),
                              ("object", "seg_object_confusion.csv", "instances")]:
        df = pd.read_csv(R / fname)
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
        for ax, (key, label) in zip(axes, SEG_MODELS.items()):
            sub = df[df.run == f"{key}_{SCHEME}_s0"].set_index("gt")
            labels = list(sub.index)
            mat = sub[labels].values
            if kind == "object":
                mat, rows = mat[1:], labels[1:]
            else:
                rows = labels
            plot_cm(ax, mat, rows, labels, f"{label} ({unit})")
        fig.tight_layout()
        fig.savefig(FIG / f"seg-confusion-{kind}.pdf")
        plt.close(fig)


def det_confusion_figure():
    df = pd.read_csv(R / "det_object_confusion.csv")
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
    for ax, (key, label) in zip(axes, DET_MODELS.items()):
        sub = df[df.run == f"{key}_{SCHEME}_s0"].set_index("gt")
        labels = list(sub.index)
        mat = sub[labels].values
        plot_cm(ax, mat, labels, labels, label)
    fig.tight_layout()
    fig.savefig(FIG / "det-confusion-object.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ summary bar chart (R2 minor 6)
def summary_bars():
    seg = pd.read_csv(R / "seg_per_class.csv")
    det = pd.read_csv(R / "det_per_class.csv")
    seg = seg[(seg.scheme == SCHEME) & seg["class"].isin(CELL)]
    det = det[(det.scheme == SCHEME) & (det.iou_type == "bbox")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    x = np.arange(len(CELL))
    for ax, data, models, metric, title in [
        (axes[0], seg, SEG_MODELS, "Dice", "Segmentation: per-class Dice"),
        (axes[1], det, DET_MODELS, "AP50", "Detection: per-class AP$_{50}$ (box)")]:
        w = 0.8 / len(models)
        for k, (key, label) in enumerate(models.items()):
            g = data[data.model == key].groupby("class")[metric]
            m, s = g.mean().reindex(CELL).values, g.std().reindex(CELL).values
            ax.bar(x + (k - (len(models) - 1) / 2) * w, m, w * 0.9, yerr=s, color=COLORS[label], label=label,
                   error_kw={"elinewidth": 0.8, "capsize": 1.5, "ecolor": INK2})
        ax.set_xticks(x, [SHORT[c] for c in CELL], rotation=30, ha="right")
        ax.set_ylim(0, 1)
        ax.yaxis.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.set_title(title, fontsize=8, color=INK)
        ax.legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "summary-per-class.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ PR curves
def pr_curves():
    pr = pd.read_csv(R / "det_pr_curves.csv")
    fig, axes = plt.subplots(1, 5, figsize=(11, 2.4), sharey=True)
    for ax, cls in zip(axes, CELL):
        for key, label in DET_MODELS.items():
            sub = pr[(pr.run == f"{key}_{SCHEME}_s0") & (pr["class"] == cls)]
            p = sub.precision.where(sub.precision >= 0)
            ax.plot(sub.recall, p, lw=1.5, color=COLORS[label], label=label)
        ax.set_title(SHORT[cls], fontsize=8)
        ax.set_xlabel("Recall")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(True, color=GRID, lw=0.5)
    axes[0].set_ylabel("Precision")
    axes[-1].legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "det-pr-curves.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ object sizes
def object_sizes():
    from pycocotools.coco import COCO
    coco = COCO(str(REPO / "data_v2/annotations/benchmark/all.json"))
    rows = [{"class": coco.cats[a["category_id"]]["name"], "side_um": np.sqrt(a["bbox"][2] * a["bbox"][3]) * 0.23,
             "diam_um": 2 * np.sqrt(a["area"] / np.pi) * 0.23} for a in coco.anns.values()]
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    data = [df[df["class"] == c].diam_um.values for c in CELL]
    vp = ax.violinplot(data, showmedians=True, widths=0.8)
    for b in vp["bodies"]:
        b.set_facecolor("#2a78d6")
        b.set_alpha(0.35)
    for k in ("cmedians", "cmins", "cmaxes", "cbars"):
        vp[k].set_color(INK2)
        vp[k].set_linewidth(0.8)
    ax.set_xticks(range(1, 6), [SHORT[c] for c in CELL], rotation=30, ha="right")
    ax.set_ylabel("Equivalent diameter (µm)")
    ax.set_ylim(0, 40)
    ax.yaxis.grid(True, color=GRID, lw=0.5)
    fig.tight_layout()
    fig.savefig(FIG / "object-sizes.pdf")
    plt.close(fig)
    summ = df.groupby("class").agg(n=("diam_um", "size"), diam_med=("diam_um", "median"),
                                   diam_q1=("diam_um", lambda s: s.quantile(.25)),
                                   diam_q3=("diam_um", lambda s: s.quantile(.75)),
                                   box_med=("side_um", "median")).reindex(CELL)
    summ["box_px"] = summ.box_med / 0.23
    summ.to_csv(TAB / "object_sizes.csv")
    return summ


# ------------------------------------------------------------------ LaTeX tables
def seg_tables():
    s = pd.read_csv(R / "seg_summary.csv", index_col=0)
    pc = pd.read_csv(R / "seg_per_class.csv")
    lines = []
    for scheme in ("mouse_holdout", "random_tile"):
        for key, label in SEG_MODELS.items():
            sub = s[(s.model == key) & (s.scheme == scheme)]
            if sub.empty:
                continue
            cells = [fmt(sub[c].mean(), sub[c].std() if len(sub) > 1 else np.nan)
                     for c in ["PA", "mIoU_fg", "mDice_fg", "GDS", "mBF1_fg"]]
            lines.append(f"{label} & {scheme.replace('_', '-')} & " + " & ".join(cells) + r" \\")
    (TAB / "seg_overall.tex").write_text("\n".join(lines))
    lines = []
    for cls in ["Background"] + CELL:
        row = [SHORT[cls]]
        for key in SEG_MODELS:
            sub = pc[(pc.model == key) & (pc.scheme == SCHEME) & (pc["class"] == cls)]
            s0 = sub[sub.seed == "s0"].iloc[0]
            row.append(fmt(sub.IoU.mean(), sub.IoU.std()))
            row.append(fmt(sub.Dice.mean(), sub.Dice.std()) + f" [{s0.Dice_lo:.2f}, {s0.Dice_hi:.2f}]")
            row.append("--" if cls == "Background" else f"{sub.BF1.mean():.3f}")
        lines.append(" & ".join(row) + r" \\")
    (TAB / "seg_per_class.tex").write_text("\n".join(lines))


def det_tables():
    s = pd.read_csv(R / "det_summary.csv")
    pc = pd.read_csv(R / "det_per_class.csv")
    lines = []
    for scheme in ("mouse_holdout", "random_tile"):
        for key, label in DET_MODELS.items():
            for it in ("bbox", "segm"):
                sub = s[(s.model == key) & (s.scheme == scheme) & (s.iou_type == it)]
                if sub.empty:
                    continue
                sd = lambda c: sub[c].std() if len(sub) > 1 else np.nan  # noqa: E731
                cols = ["AP50", "AP75", "AP", "macro_P", "macro_R", "macro_F1"]
                cells = [fmt(sub[c].mean(), sd(c)) if c in sub and sub[c].notna().all() else "--" for c in cols]
                thr = f"{sub.conf_thr.mean():.2f}" if it == "bbox" else "--"
                lines.append(f"{label} ({'mask' if it == 'segm' else 'box'}) & {scheme.replace('_', '-')} & "
                             + " & ".join(cells) + f" & {thr}" + r" \\")
    (TAB / "det_overall.tex").write_text("\n".join(lines))
    lines = []
    for cls in CELL:
        row = [SHORT[cls]]
        for key in DET_MODELS:
            sub = pc[(pc.model == key) & (pc.scheme == SCHEME) & (pc.iou_type == "bbox") & (pc["class"] == cls)]
            if sub.empty:
                row += ["--", "--"]
                continue
            s0 = sub[sub.seed == "s0"].iloc[0]
            row.append(fmt(sub.AP50.mean(), sub.AP50.std(), 2) + f" [{s0.AP50_lo:.2f}, {s0.AP50_hi:.2f}]")
            row.append(fmt(sub.F1.mean(), sub.F1.std(), 2))
        n = int(sub.n_gt.iloc[0]) if not sub.empty else 0
        lines.append(" & ".join(row) + f" & {n}" + r" \\")
    (TAB / "det_per_class.tex").write_text("\n".join(lines))


if __name__ == "__main__":
    import sys
    todo = sys.argv[1:] or ["seg", "det", "sizes"]
    if "sizes" in todo:
        print(object_sizes().round(1))
    if "seg" in todo:
        seg_confusion_figures()
        seg_tables()
    if "det" in todo:
        det_confusion_figure()
        summary_bars()
        pr_curves()
        det_tables()
    print("written", sorted(p.name for p in FIG.iterdir()), sorted(p.name for p in TAB.iterdir()))
