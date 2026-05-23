# MurineCyto Detection and Segmentation

Code used for murine BALF cytology segmentation and object-detection experiments
published in **Paper-2025-MurineCyto-Det**.

## Project Layout

```
.
├── dataloader/
│   └── dataloaders.py            # Dataset class & augmentation pipelines
│
├── utils/
│   ├── data.py                   # HuggingFace download, extraction, train/val/test split
│   ├── det_helpers.py            # OD visualisation & annotation consistency checks
│   ├── helpers.py                # Re-exports from postprocessing helpers
│   ├── plot_segmentation.py      # Segmentation-specific plot utilities
│   ├── preprocessing/
│   │   ├── export_labelbox.py    # Export annotations from Labelbox
│   │   ├── export_labelbox2grayscale.py  # Convert colour masks → grayscale
│   │   └── masks2bboxes.py       # Derive bounding boxes from segmentation masks
│   └── postprocessing/
│       ├── helpers.py            # predict_and_plot, confusion matrix, metric bars
│       └── tools.py              # Dataloader validation utilities
│
├── train_seg_torch.py            # Segmentation trainer — plain PyTorch (UNet + SegFormer)
├── train_seg_lightning.py        # Segmentation trainer — PyTorch Lightning (UNet + SegFormer)
├── train_od.py                   # Object-detection trainer — Faster R-CNN (torchvision)
│
├── download_data.py              # CLI dataset downloader (HuggingFace Hub)
├── main.py                       # Project health-check
├── reports.py                    # Manuscript figure & metric generation
│
└── notebooks/
    ├── yolo.ipynb                # YOLO v8 training & evaluation (Google Colab)
    ├── faster_rcnn.ipynb         # Faster R-CNN exploration
    ├── segmentation_multiclass.ipynb  # Segmentation model analysis
    ├── report.ipynb              # Full report notebook
    ├── paper_benchmark_seg.ipynb # Paper benchmark: segmentation models
    ├── unet_exploration.ipynb    # UNet early exploration
    ├── pspnet_exploration.ipynb  # PSPNet exploration
    ├── seg_models_exploration.ipynb   # General seg-models exploration
    ├── data_export_labelbox.ipynb     # Export annotations from Labelbox
    ├── data_masks_to_bboxes.ipynb     # Mask → bounding-box conversion
    ├── data_extract_ndpi_to_png.ipynb # NDPI slide → PNG tiles
    └── data_openslide_demo.ipynb      # OpenSlide Python demo
```

Generated or large local artifacts live in `data/`, `logs/`, `models/`,
`figures/`, and `paper/` — all git-ignored.

## Cell Classes

| Index | Class |
|-------|-------|
| 0 | Background |
| 1 | Macrophage/Monocyte |
| 2 | Neutrophil |
| 3 | Eosinophil |
| 4 | Lymphocyte |
| 5 | Unknown cell/Debris |

## Environment

Use `uv` for Python dependency management:

```bash
uv sync
```

If the default uv cache is not writable (sandboxed environment):

```bash
uv --cache-dir /tmp/uv-cache sync
```

## Common Commands

### Health check

```bash
uv run murine-check
```

### Download dataset

```bash
uv run murine-download
```

Set `HF_TOKEN` in your environment or `.env` for private dataset access.
Do **not** hard-code tokens in source files.

---

### Segmentation — UNet & SegFormer

Train with plain PyTorch *(recommended)*:

```bash
uv run murine-train-seg
```

Train with PyTorch Lightning:

```bash
uv run murine-train-seg-lightning
```

On a **4 GB GPU** (e.g. GTX 1650):

```bash
uv run murine-train-seg --epochs 2 --batch-size 1
```

**CLI options for `murine-train-seg`:**

| Option | Default | Description |
|--------|---------|-------------|
| `--epochs N` | 10 | Training epochs |
| `--batch-size N` | 8 | Batch size |
| `--num-workers N` | 0 | DataLoader workers |
| `--models unet segformer` | both | Which models to train |
| `--run-name NAME` | timestamp | Label for logs & checkpoints |

Use `ENCODER_WEIGHTS=imagenet uv run murine-train-seg` to load pre-trained
backbone weights (requires internet access on first run).

---

### Object Detection — Faster R-CNN

```bash
uv run murine-train-od
```

On a **4 GB GPU**:

```bash
uv run murine-train-od --epochs 2 --batch-size 1
```

**CLI options for `murine-train-od`:**

| Option | Default | Description |
|--------|---------|-------------|
| `--epochs N` | 10 | Training epochs |
| `--batch-size N` | 4 | Batch size |
| `--num-workers N` | 0 | DataLoader workers |
| `--lr FLOAT` | 1e-3 | Learning rate |
| `--pretrained` | off | Use ImageNet backbone weights |
| `--run-name NAME` | timestamp | Label for logs & checkpoints |

Metrics (mAP, mAP@50, mAP@75) are logged to
`logs/faster_rcnn_torch/<run-name>/metrics.csv`.

**YOLO:** Training was performed on Google Colab with the `ultralytics` library.
See `notebooks/yolo.ipynb` for the full pipeline.
