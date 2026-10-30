# MurineCyto-Det — revised benchmark

This directory contains the pipeline for the completed revised experiments. The primary benchmark is the fixed **mouse-level 242/47/44 split**. Historical scripts in the upstream repository are not the entry points for these results. The old `train_od.py` path now delegates to the revised detector trainer, and `utils/preprocessing/masks2bboxes.py` exports per-image files from v2 full-instance COCO annotations rather than fragment contours.

## Environment

Python 3.12, Linux and an NVIDIA GPU are needed for training. Install [uv](https://docs.astral.sh/uv/) and run from the repository root:

```sh
uv sync --frozen
```

The supplied `uv.lock` is the revision lockfile (PyTorch 2.11.0 / torchvision 0.26.0, CUDA 12.6 wheels; segmentation-models-pytorch 0.5.0, timm 1.0.26, Ultralytics 8.4.42, pycocotools 2.0.11). CUDA driver compatibility and GPU memory remain machine-dependent. Model initialization downloads ImageNet/COCO pretrained weights on first use.

## Data

Download **data_v2.zip** from the v2 Zenodo version linked by the manuscript, and extract it into the repository root. The archive contains one top-level `data_v2/` directory. No Hugging Face token or Labelbox account is required to use the public release.

```sh
uv run python revision/validate_release.py --data-dir data_v2 --check-hashes --check-masks
```

See [the data documentation](docs/DATA_README.md) for folder names, category IDs and the two instance-mask sets. All scripts resolve relative data/run paths from the repository root, so they also work when called by an absolute script path from another directory.

## Reproduce the primary training runs

Run the following once for each seed **0, 1, 2**. These are full training commands, not smoke tests. `--gpu 0` is the CUDA device index; adjust it for your workstation.

```sh
uv run python revision/train_seg.py --arch Unet --encoder resnet34 --scheme mouse_holdout --seed 0 --gpu 0
uv run python revision/train_seg.py --arch Segformer --encoder mit_b2 --scheme mouse_holdout --seed 0 --gpu 0
uv run python revision/train_det.py --model fasterrcnn --scheme mouse_holdout --seed 0 --gpu 0 --batch-size 4
uv run python revision/train_det.py --model maskrcnn --scheme mouse_holdout --seed 0 --gpu 0 --batch-size 4
uv run python revision/train_det.py --model yolov8 --scheme mouse_holdout --seed 0 --gpu 0 --batch-size 8
```

Changing batch size or other hyperparameters creates a different training configuration. The scripts use the supplied split lists; they never randomly repartition the primary benchmark. Best validation checkpoints are reloaded before inference. The default `--annotation-set benchmark` reproduces the reconstructed instance masks used in the paper. An optional `--annotation-set original` is a separate experiment, and must use a separate output directory.

## Evaluate saved predictions without retraining

Extract the optional **benchmark_artifacts.zip** into this root to obtain `runs_revision/`, including configuration files, best checkpoints and saved predictions. Then run:

```sh
uv run python revision/evaluate_seg.py --data-dir data_v2 --runs-dir runs_revision --scheme mouse_holdout
uv run python revision/evaluate_det.py --data-dir data_v2 --runs-dir runs_revision --scheme mouse_holdout
```

Outputs are written to `runs_revision/results/`. The `reference_results/` CSVs preserve the results before repackaging. The detection evaluator retains the implemented convention: AP50:95 at maxDets=100; AP50/AP75 at maxDets=300; prediction cap 300/image. All models use the same convention. It explicitly extracts these limits instead of depending silently on the COCO summary default. Mask metrics use the reconstructed benchmark masks by default. Main-table point estimates average three seeds, while the reported bootstrap intervals are for seed 0.

For a separately labeled sensitivity analysis using original Labelbox masks:

```sh
uv run python revision/evaluate_det.py --data-dir data_v2 --runs-dir runs_revision --scheme mouse_holdout --annotation-set original --results-dir runs_revision/results_original_masks
```

This does not modify the saved predictions or the reported reference CSVs. The historical `random_tile` comparison remains available with explicit `--scheme random_tile`; it is not the primary benchmark. Figure-generation commands are documented in [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Rebuild the data release (maintainers only)

```sh
uv run python revision/prepare_release.py --source-code-dir /path/to/local/revision/code --output /path/to/new/data_v2 --zip /path/to/data_v2.zip
```

The source must contain the existing `data/` and `data_v2/annotations/instances_labelbox.json`. Output must be a new directory. The script copies existing scientific data without changing masks, split membership or class labels, and strips private Labelbox identifiers/URLs.

## Distribution

Dataset: CC BY 4.0. This README does not grant a new software license for pre-existing code; retain the repository's existing licensing policy. Do not commit datasets, model weights, private exports or credentials to Git. Publish large reproducibility artifacts with Zenodo/Hugging Face and cite the final version DOI/release tag in the manuscript when ready.
