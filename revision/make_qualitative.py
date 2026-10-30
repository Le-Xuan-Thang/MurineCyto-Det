"""Qualitative figures with scale bars (Reviewer 2 minor 4, Reviewer 3 #10).

  data-examples.pdf      Fig. 2: image | semantic map | instance boxes for three tiles
  qual-predictions.pdf   Supp. Fig. S4: test tiles of mouse 2, GT vs U-Net, SegFormer, Mask R-CNN
  class-gallery.pdf      Supp. Fig. S5: high-magnification crops of each class
"""

import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from pycocotools import mask as mask_utils  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_seg import SegModel  # noqa: E402
from train_det import build_tv_model  # noqa: E402

UM_PER_PX = 0.23
NAMES = ["Background", "Macrophage/Monocyte", "Neutrophil", "Eosinophil", "Lymphocyte", "Unknown cell/Debris"]
# categorical slots: blue, orange, aqua, magenta; debris (auxiliary) in neutral grey
COL = np.array([[0, 0, 0], [42, 120, 214], [235, 104, 52], [27, 175, 122], [232, 123, 164], [150, 150, 150]], np.uint8)
REPO = Path(__file__).resolve().parents[1]
FIG = REPO / "runs_revision/figures"
FIG.mkdir(parents=True, exist_ok=True)
DATA = REPO / "data_v2"
coco = COCO(str(DATA / "annotations/coco/all.json"))
by_name = {im["file_name"]: i for i, im in coco.imgs.items()}


def img(f):
    return cv2.cvtColor(cv2.imread(str(DATA / "images" / f)), cv2.COLOR_BGR2RGB)


def overlay(im, lab, alpha=0.55):
    out = im.copy()
    m = lab > 0
    out[m] = (alpha * COL[lab[m]] + (1 - alpha) * im[m]).astype(np.uint8)
    return out


def scalebar(ax, w_px, um=50, size=1024):
    L = um / UM_PER_PX
    x0, y0 = size * 0.95 - L, size * 0.94
    ax.add_patch(mpatches.Rectangle((x0, y0), L, size * 0.012, color="white", ec="black", lw=0.4))
    ax.text(x0 + L / 2, y0 - size * 0.015, f"{um} µm", color="white", ha="center", va="bottom", fontsize=6,
            fontweight="bold")


def legend(fig, y=0.0):
    h = [mpatches.Patch(color=COL[k] / 255, label=NAMES[k]) for k in range(1, 6)]
    fig.legend(handles=h, loc="lower center", ncol=5, frameon=False, fontsize=7, bbox_to_anchor=(0.5, y))


def gt_semantic(f):
    return cv2.imread(str(DATA / "masks_grayscale" / f), cv2.IMREAD_GRAYSCALE)


def data_examples(files):
    fig, axes = plt.subplots(len(files), 3, figsize=(7.2, 2.4 * len(files) + 0.3))
    for r, f in enumerate(files):
        im = img(f)
        anns = coco.imgToAnns[by_name[f]]
        axes[r, 0].imshow(im)
        scalebar(axes[r, 0], 1024)
        axes[r, 1].imshow(overlay(im, gt_semantic(f)))
        axes[r, 2].imshow(im)
        for a in anns:
            x, y, w, h = a["bbox"]
            axes[r, 2].add_patch(mpatches.Rectangle((x, y), w, h, fill=False, lw=0.7, ec=COL[a["category_id"]] / 255))
        for c, t in enumerate(["Image", "Semantic label map", f"Instance boxes (n = {len(anns)})"]):
            axes[r, c].set_title(t if r == 0 else (t if c == 2 else ""), fontsize=7)
            axes[r, c].axis("off")
        axes[r, 0].text(-40, 512, f.replace(".png", ""), rotation=90, va="center", ha="right", fontsize=6)
    legend(fig)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / "data-examples.pdf", dpi=200)
    plt.close(fig)


@torch.no_grad()
def predictions(files, device="cuda:0"):
    seg = {}
    for key, arch, enc in [("U-Net", "Unet", "resnet34"), ("SegFormer", "Segformer", "mit_b2")]:
        m = SegModel(arch, enc)
        m.load_state_dict(torch.load(REPO / f"runs_revision/seg/{arch}-{enc}_mouse_holdout_s0/best.pth", map_location="cpu"))
        seg[key] = m.to(device).eval()
    mr = build_tv_model("maskrcnn")
    mr.load_state_dict(torch.load(REPO / "runs_revision/det/maskrcnn_mouse_holdout_s0/best.pth", map_location="cpu"))
    mr = mr.to(device).eval()
    cols = ["Image", "Ground truth", "U-Net", "SegFormer", "Mask R-CNN (instances)"]
    fig, axes = plt.subplots(len(files), 5, figsize=(10, 2.1 * len(files) + 0.4))
    for r, f in enumerate(files):
        im = img(f)
        x = torch.from_numpy(im.transpose(2, 0, 1)).unsqueeze(0).to(device)
        panels = [im, overlay(im, gt_semantic(f))]
        for key in ("U-Net", "SegFormer"):
            panels.append(overlay(im, seg[key](x).argmax(1)[0].cpu().numpy().astype(np.uint8)))
        out = mr([x[0].float() / 255])[0]
        keep = out["scores"] > 0.35
        inst = im.copy()
        for msk, lab in zip(out["masks"][keep], out["labels"][keep]):
            mm = (msk[0] > 0.5).cpu().numpy()
            inst[mm] = (0.55 * COL[int(lab)] + 0.45 * inst[mm]).astype(np.uint8)
            cnts, _ = cv2.findContours(mm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(inst, cnts, -1, (255, 255, 255), 2)
        panels.append(inst)
        for c, p in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(p)
            ax.axis("off")
            if r == 0:
                ax.set_title(cols[c], fontsize=7)
        scalebar(axes[r, 0], 1024)
    legend(fig)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIG / "qual-predictions.pdf", dpi=200)
    plt.close(fig)


def gallery(n=6, half=64, seed=0):
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(5, n, figsize=(1.25 * n, 6.8))
    for r, cid in enumerate(range(1, 6)):
        anns = [a for a in coco.anns.values() if a["category_id"] == cid and a["area"] > 300]
        pick = rng.choice(len(anns), size=n, replace=False)
        for c, k in enumerate(pick):
            a = anns[k]
            f = coco.imgs[a["image_id"]]["file_name"]
            im = img(f)
            x, y, w, h = a["bbox"]
            cx, cy = int(x + w / 2), int(y + h / 2)
            hh = max(half, int(max(w, h) / 2) + 8)
            pad = np.pad(im, ((hh, hh), (hh, hh), (0, 0)), mode="constant", constant_values=255)
            crop = pad[cy:cy + 2 * hh, cx:cx + 2 * hh]
            m = np.pad(mask_utils.decode(a["segmentation"]), hh)[cy:cy + 2 * hh, cx:cx + 2 * hh]
            ax = axes[r, c]
            ax.imshow(crop)
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for cn in cnts:
                cn = cn[:, 0]
                ax.plot(np.r_[cn[:, 0], cn[0, 0]], np.r_[cn[:, 1], cn[0, 1]], lw=0.6, color=COL[cid] / 255)
            L = 10 / UM_PER_PX
            ax.add_patch(mpatches.Rectangle((2 * hh * 0.93 - L, 2 * hh * 0.9), L, 2 * hh * 0.025, color="black"))
            ax.axis("off")
        axes[r, 0].set_title(NAMES[cid], fontsize=8, loc="left")
    fig.text(0.99, 0.005, "scale bar: 10 µm", ha="right", fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG / "class-gallery.pdf", dpi=250)
    plt.close(fig)


if __name__ == "__main__":
    data_examples(["P01_7_7_144.png", "P01_15_7_201.png", "P39_10_5_105.png"])
    gallery()
    predictions(["P39_10_5_105.png", "P39_5_13_133.png"])
    print("done")
