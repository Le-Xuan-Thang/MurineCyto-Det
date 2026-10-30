"""Validate the public data_v2 package, split membership and optional checksums/RLE."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from revision.data_paths import data_root

COUNTS = {1: 5724, 2: 4154, 3: 388, 4: 51, 5: 4227}


def validate(root, hashes=False, masks=False):
    for sub in ("images", "masks", "masks_grayscale", "annotations", "splits"):
        assert (root / sub).is_dir(), f"Missing directory: {root / sub}"
    full = json.loads((root / "annotations/coco/all.json").read_text())
    images = {im["id"]: im for im in full["images"]}
    anns = full["annotations"]
    assert len(images) == 333 and len(anns) == 14544
    assert len({a["id"] for a in anns}) == 14544
    assert Counter(a["category_id"] for a in anns) == COUNTS
    for im in images.values():
        for sub in ("images", "masks", "masks_grayscale"):
            assert (root / sub / im["file_name"]).is_file(), (sub, im["file_name"])
        per = json.loads((root / "annotations" / (Path(im["file_name"]).stem + ".json")).read_text())
        assert len(per["images"]) == 1
        assert {a["id"] for a in per["annotations"]} == {a["id"] for a in anns if a["image_id"] == im["id"]}
    for scheme, sizes in [("mouse_holdout", [242, 47, 44]), ("random_tile", [233, 50, 50])]:
        splits = []
        for part, expected in zip(("train", "val", "test"), sizes):
            names = (root / "splits" / f"{scheme}_{part}.txt").read_text().split()
            assert len(names) == len(set(names)) == expected
            splits.append(set(names))
            for aset in ("coco", "benchmark"):
                c = json.loads((root / "annotations" / aset / f"{scheme}_{part}.json").read_text())
                assert {im["file_name"] for im in c["images"]} == set(names)
                assert len({a["id"] for a in c["annotations"]}) == len(c["annotations"])
        assert not (splits[0] & splits[1] or splits[0] & splits[2] or splits[1] & splits[2])
        assert set.union(*splits) == {im["file_name"] for im in images.values()}
        if scheme == "mouse_holdout":
            assert all(n.startswith("P01_") for n in splits[0] | splits[1])
            assert all(n.startswith("P39_") for n in splits[2])
    original_boxes = Counter((a["image_id"], a["category_id"], tuple(a["bbox"])) for a in anns)
    benchmark = json.loads((root / "annotations/benchmark/all.json").read_text())
    assert len(benchmark["annotations"]) == 14544
    assert original_boxes == Counter((a["image_id"], a["category_id"], tuple(a["bbox"])) for a in benchmark["annotations"])
    if masks:
        import numpy as np
        import cv2
        from pycocotools import mask as mu
        for d in (full, benchmark):
            for a in d["annotations"]:
                assert a["image_id"] in images and a["category_id"] in COUNTS
                assert a["segmentation"]["size"] == [1024, 1024]
                area = int(mu.area(a["segmentation"]))
                assert area > 0 and area == int(a["area"]), a["id"]
                if d is full:
                    assert np.allclose(mu.toBbox(a["segmentation"]), a["bbox"])
        for im in images.values():
            m = cv2.imread(str(root / "masks_grayscale" / im["file_name"]), 0)
            assert m is not None and m.shape == (1024, 1024)
            assert set(np.unique(m)) <= set(range(6))
    if hashes:
        for line in (root / "SHA256SUMS.txt").read_text().splitlines():
            expected, name = line.split("  ", 1)
            p = root / name
            with p.open("rb") as f:
                assert hashlib.file_digest(f, "sha256").hexdigest() == expected, name
    print("PASS: 333 images; 14,544 instances; primary split 242/47/44; unique IDs; matching boxes.")
    if masks:
        print("PASS: non-empty RLE masks, areas, original tight boxes and semantic labels.")
    if hashes:
        print("PASS: all SHA-256 checksums.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data_v2")
    ap.add_argument("--check-hashes", action="store_true")
    ap.add_argument("--check-masks", action="store_true")
    args = ap.parse_args()
    validate(data_root(args.data_dir), args.check_hashes, args.check_masks)


if __name__ == "__main__":
    main()
