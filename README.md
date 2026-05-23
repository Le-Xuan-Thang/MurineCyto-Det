# MurineCyto Detection and Segmentation

This project contains the code used for murine BALF cytology segmentation and
object-detection experiments published in **Paper-2025-MurineCyto-Det**.

## Layout

```
.
├── dataloader/               # PyTorch dataset & dataloader builders (segmentation)
├── utils/
│   ├── data.py               # HuggingFace download, extraction, train/val/test split
│   ├── det_helpers.py        # Object-detection visualisation & annotation checks
│   ├── preprocessing/        # Labelbox export → grayscale mask converters
│   └── postprocessing/       # Plotting, prediction, confusion-matrix helpers
├── download_data.py          # CLI dataset downloader
├── seg_v1_torch.py           # Plain-PyTorch segmentation trainer (UNet / SegFormer)
├── seg_v1.py                 # PyTorch-Lightning segmentation trainer
├── seg.py                    # Compatibility wrapper → seg_v1_torch
├── od.py                     # Faster R-CNN object-detection trainer
├── main.py                   # Health-check entry point
├── reports.py                # Manuscript figure & metric generation
└── notebooks/                # Exploratory / analysis notebooks
    └── notebooks/
        ├── YOLOv8.ipynb      # YOLO training & evaluation (Colab)
        └── FasterRCNN.ipynb  # Faster R-CNN exploration notebook
```

Generated or large local artifacts live in `data/`, `logs/`, `models/`,
`figures/`, and `paper/` — these are git-ignored.

## Environment

Use `uv` for Python dependency management:

```bash
uv sync
```

If the default uv cache is not writable in a sandboxed environment:

```bash
uv --cache-dir /tmp/uv-cache sync
```

## Common Commands

### Health check

Verify local data is present and can be split:

```bash
uv run murine-check
```

### Download dataset

```bash
uv run murine-download
```

Set `HF_TOKEN` in your environment or `.env` file for private dataset access.

### Segmentation (UNet & SegFormer)

Train both models with plain PyTorch:

```bash
uv run murine-train-torch
```

Train with PyTorch Lightning:

```bash
uv run murine-train-lightning
```

On a **4 GB GPU** (e.g. GTX 1650), use a smaller batch size:

```bash
uv run murine-train-torch --epochs 2 --batch-size 1
```

Available CLI options for `murine-train-torch`:

| Option | Default | Description |
|--------|---------|-------------|
| `--epochs N` | 10 | Training epochs |
| `--batch-size N` | 8 | Batch size |
| `--num-workers N` | 0 | DataLoader workers |
| `--models unet segformer` | both | Models to train |
| `--run-name NAME` | timestamp | Label for logs & checkpoints |

Use `ENCODER_WEIGHTS=imagenet uv run murine-train-torch` to load pre-trained
backbone weights (requires internet access).

### Object Detection (Faster R-CNN)

```bash
uv run murine-train-od
```

On a **4 GB GPU**:

```bash
uv run murine-train-od --epochs 2 --batch-size 1
```

Available CLI options for `murine-train-od`:

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
See `notebooks/notebooks/YOLOv8.ipynb` for the full pipeline.

## Cell Classes

| Index | Class |
|-------|-------|
| 0 | Background |
| 1 | Macrophage/Monocyte |
| 2 | Neutrophil |
| 3 | Eosinophil |
| 4 | Lymphocyte |
| 5 | Unknown cell/Debris |

## Dataset access

The dataset is hosted on Hugging Face (`thanglexuan/murincells`).  Set the
`HF_TOKEN` environment variable (or add it to `.env`) if the dataset is
private.  Do **not** hard-code tokens in source files.
