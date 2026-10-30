"""Prepare a portable, credential-free data_v2 directory from the local revision.

This packages existing annotations; it does not regenerate masks or change splits.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

PRIVATE_KEYS = {"mask_url", "image_url", "labelbox_feature_id"}


def public_coco(d, note):
    d = dict(d)
    d["info"] = {"description": "MurineCyto-Det", "version": "2.0", "notes": note}
    d["licenses"] = [{"id": 1, "name": "Creative Commons Attribution 4.0 International",
                       "url": "https://creativecommons.org/licenses/by/4.0/"}]
    d["images"] = [{k: v for k, v in im.items() if k not in PRIVATE_KEYS} for im in d["images"]]
    for im in d["images"]:
        im["license"] = 1
    d["annotations"] = [{k: v for k, v in a.items() if k not in PRIVATE_KEYS} for a in d["annotations"]]
    return d


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n")


def prepare(source, out):
    source, out = source.resolve(), out.resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite an existing release: {out}")
    for rel in ["data/images", "data/masks", "data/masks_grayscale", "data_v2/splits"]:
        if not (source / rel).is_dir():
            raise FileNotFoundError(source / rel)
    original = public_coco(json.loads((source / "data_v2/annotations/instances_labelbox.json").read_text()),
                           "Original non-empty per-instance Labelbox masks (COCO RLE); one box per instance.")
    benchmark = public_coco(json.loads((source / "data_v2/annotations/all.json").read_text()),
                            "Reconstructed instance masks used for the reported benchmark; original masks are in annotations/coco.")
    assert len(original["annotations"]) == len(benchmark["annotations"]) == 14544
    assert len(original["images"]) == 333
    out.mkdir(parents=True)
    for sub in ("images", "masks", "masks_grayscale"):
        target = out / sub
        target.mkdir()
        for im in original["images"]:
            shutil.copy2(source / "data" / sub / im["file_name"], target / im["file_name"])
    shutil.copytree(source / "data_v2/splits", out / "splits")
    shutil.copy2(source / "data_v2/metadata.csv", out / "metadata.csv")
    for p in (source / "data_v2").glob("class_counts_*.csv"):
        shutil.copy2(p, out / p.name)
    for folder, full in [("coco", original), ("benchmark", benchmark)]:
        write_json(out / "annotations" / folder / "all.json", full)
        for scheme in ("mouse_holdout", "random_tile"):
            for part in ("train", "val", "test"):
                names = set((out / "splits" / f"{scheme}_{part}.txt").read_text().split())
                images = [im for im in full["images"] if im["file_name"] in names]
                ids = {im["id"] for im in images}
                subset = dict(full, images=images, annotations=[a for a in full["annotations"] if a["image_id"] in ids])
                write_json(out / "annotations" / folder / f"{scheme}_{part}.json", subset)
    # Compatibility with the original data/images + masks + masks_grayscale + annotations layout.
    # A per-image file contains ORIGINAL masks with globally unique annotation IDs.
    by_image = defaultdict(list)
    for a in original["annotations"]:
        by_image[a["image_id"]].append(a)
    for im in original["images"]:
        per_image = dict(original, images=[im], annotations=by_image[im["id"]])
        write_json(out / "annotations" / (Path(im["file_name"]).stem + ".json"), per_image)
    audit = {"annotation_objects_in_final_labelbox_export": 14551,
             "empty_masks_excluded_author_confirmed": 7, "valid_instances": 14544,
             "v1_fragment_boxes": 16228, "additional_fragment_boxes": 1684,
             "classes": dict(sorted(Counter(a["category_id"] for a in original["annotations"]).items())),
             "primary_split": {"train": 242, "val": 47, "test": 44},
             "empty_mask_evidence": "Author-confirmed; the seven excluded mask images are not included in the public release."}
    write_json(out / "annotation_audit.json", audit)
    docs = Path(__file__).resolve().parents[1] / "docs"
    for name in ("DATA_README.md", "DATA_CHANGELOG.md"):
        shutil.copy2(docs / name, out / ("README.md" if name == "DATA_README.md" else "CHANGELOG.md"))
    (out / "LICENSE.txt").write_text("Dataset license: Creative Commons Attribution 4.0 International (CC BY 4.0).\nhttps://creativecommons.org/licenses/by/4.0/legalcode\nAttribute MurineCyto-Det and its authors when reusing the data.\n")
    lines = []
    for p in sorted(out.rglob("*")):
        if p.is_file():
            h = hashlib.file_digest(p.open("rb"), "sha256").hexdigest()
            lines.append(f"{h}  {p.relative_to(out).as_posix()}")
    (out / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-code-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--zip", type=Path, help="Optional ZIP containing the top-level data_v2 directory")
    args = ap.parse_args()
    out = prepare(args.source_code_dir, args.output)
    if args.zip:
        args.zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for p in sorted(out.rglob("*")):
                if p.is_file():
                    z.write(p, Path("data_v2") / p.relative_to(out))
    print(f"Prepared {out}")


if __name__ == "__main__":
    main()
