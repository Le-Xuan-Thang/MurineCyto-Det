# MurineCyto-Det v2.0

333 RGB BALF cytology tiles, 1024 × 1024 pixels, 0.23 µm/pixel; 14,544 valid annotated instances. The primary benchmark is the fixed mouse-level split: 242 training, 47 validation and 44 test tiles. Mouse identifiers follow the study metadata: M1/P01 supplies training and validation; M2/P39 supplies test. Confirmed exposure groups, stain and detailed wet-lab methods should be taken from the final manuscript, not inferred from filenames.

## Layout

```text
data_v2/
  images/                    333 original RGB tiles
  masks/                     333 existing colour semantic masks (not instance IDs)
  masks_grayscale/           333 semantic label maps: 0 background, 1–5 classes
  annotations/
    <image-stem>.json        one COCO file per image, original Labelbox instances
    coco/                   original non-empty Labelbox instance masks and boxes
      all.json
      mouse_holdout_{train,val,test}.json
      random_tile_{train,val,test}.json
    benchmark/              exact reconstructed masks used for the reported runs
      all.json
      mouse_holdout_{train,val,test}.json
      random_tile_{train,val,test}.json
  splits/                   exact image lists
  metadata.csv              per-tile IDs, grid coordinates, crop pass and split
  class_counts_*.csv
  annotation_audit.json
  SHA256SUMS.txt
```

The four main folders match the original `data/` layout. COCO instance masks are compressed RLE inside JSON, not the colour PNGs in `masks/`. Bounding boxes use `[x, y, width, height]` in native pixels. Annotation IDs are globally unique within each aggregate COCO file. Do not concatenate all per-image and aggregate JSONs: they intentionally describe the same objects.

## Classes

| ID | Class | Instances |
|---|---|---:|
| 1 | Macrophage/Monocyte | 5,724 |
| 2 | Neutrophil | 4,154 |
| 3 | Eosinophil | 388 |
| 4 | Lymphocyte | 51 |
| 5 | Unknown cell/Debris (auxiliary) | 4,227 |

14,551 Labelbox annotation objects minus 7 author-confirmed empty masks = 14,544 valid instances. Earlier fragmented exports contained 16,228 boxes, including 1,684 extra boxes associated with existing instances. v2 has exactly one box per instance.

## Benchmark versus original instance masks

Use `annotations/benchmark/` to reproduce the reported Mask R-CNN and object-confusion results. These masks were reconstructed from semantic label maps and per-cell boxes; overlapping same-class assignments use normalized distance to the box centre. Use `annotations/coco/` for original Labelbox masks. Both have the same per-class counts and box geometry. They are different mask ground truths and must be identified when reporting results.

The primary benchmark uses only `mouse_holdout`. `random_tile` is retained solely to reproduce the historical secondary comparison, not as an alternative primary benchmark. Spatial block grouping of validation does not guarantee a gap between train and validation blocks. There is one held-out mouse; tile-bootstrap intervals do not estimate between-animal uncertainty.

## Evaluation convention used by the reported results

COCO-compatible evaluation with prediction cap 300/image, AP50:95 summarized at maxDets=100 and AP50/AP75 at maxDets=300. This convention is identical across models. It is retained to reproduce the completed experiments; do not relabel AP50:95 as AP@300.

## Reuse and citation

License: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/ . Cite MurineCyto-Det and the dataset DOI attached to this Zenodo version. Code: https://github.com/Le-Xuan-Thang/MurineCyto-Det . The manuscript is under review; publication details are not invented here.

Verify the archive after extraction with `sha256sum -c SHA256SUMS.txt` from this directory, or use the Python validator supplied with the code release. No private Labelbox exports, feature identifiers, signed URLs or credentials are included.
