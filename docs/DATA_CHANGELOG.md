# Changelog

## v2.0

- Reconciled counts: 14,551 annotation objects − 7 empty masks = 14,544 valid instances.
- Replaced contour-level boxes with one full-mask box per instance and unique annotation IDs.
- Corrected category spelling to Macrophage/Monocyte.
- Added original Labelbox instance masks in COCO RLE and retained reconstructed benchmark masks separately for exact reproducibility.
- Added metadata, the primary mouse-level 242/47/44 split and secondary historical tile-level lists.
- Packaged `images/`, `masks/`, `masks_grayscale/`, `annotations/` without machine-specific symlinks.
- Included checksums and an annotation audit. This packaging step does not alter images, semantic maps, trained weights or reported benchmark values.
