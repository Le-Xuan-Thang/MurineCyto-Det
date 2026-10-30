# Reproducibility notes

The completed benchmark used three seeds (0, 1, 2) for each of five models on the primary mouse-level 242/47/44 split. Configurations in `configs/` are copies of completed-run records; they are not regenerated training logs. Their paths are descriptive of the run directory layout. Checkpoints and predictions are distributed separately.

## Protocol preserved during repackaging

- Semantic training: 256×256 random crops; full-resolution test inference; ImageNet initialization; Adam; best validation foreground mIoU.
- Detection: native 1024×1024; COCO initialization; AdamW; best validation AP50:95 checkpoint at maxDets=100. Mask R-CNN uses mask validation AP.
- AP50:95 at maxDets=100, AP50/AP75 at maxDets=300, score floor 0.001, 300 predictions/image. Per-class AP50:95 now explicitly uses the same limit as overall AP50:95. This only resolves an unused per-class-summary inconsistency; reported overall values are preserved.
- Precision/recall/F1: validation-selected confidence threshold; class-wise score-ordered greedy matches at IoU 0.5; macro average over five classes.
- Original Labelbox masks and reconstructed benchmark masks have identical instance counts and boxes but different boundaries. Default runs use reconstructed masks to reproduce the paper.
- Bootstrap: 1,000 resamples of test tiles. Segmentation uses zero where the implemented denominator floor applies; detection drops resamples without ground-truth instances of a class from that class's AP interval. This is within-animal uncertainty.

## Figures

The original figure builders consume both primary and secondary CSVs. To reproduce all figures, evaluate with `--scheme all` after extracting artifacts (including the historical single-seed secondary runs), then run:

```sh
uv run python scripts/evaluate_seg.py --scheme all
uv run python scripts/evaluate_det.py --scheme all
uv run python scripts/make_figures.py
uv run python scripts/make_qualitative.py
```

Their relative paths are resolved from the repository root. These commands write derived files under `runs_revision/`; they do not modify images, annotations or checkpoints. See each script's `--help` where supported. The benchmark artifacts archive includes only required model/prediction files; raw private data and training caches are excluded.
