"""Export one public COCO JSON per image from v2 original instance annotations."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from revision.data_paths import data_root


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data_v2")
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()
    root = data_root(args.data_dir)
    d = json.loads((root / "annotations/coco/all.json").read_text())
    assert len({a["id"] for a in d["annotations"]}) == len(d["annotations"])
    target = args.output_dir.resolve() if args.output_dir else root / "annotations"
    target.mkdir(parents=True, exist_ok=True)
    by_image = defaultdict(list)
    for a in d["annotations"]:
        by_image[a["image_id"]].append(a)
    for im in d["images"]:
        out = dict(d, images=[im], annotations=by_image[im["id"]])
        (target / (Path(im["file_name"]).stem + ".json")).write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"Exported {len(d['images'])} files; one annotation/box per valid instance.")


if __name__ == "__main__":
    main()
