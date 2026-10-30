"""Portable paths for the v2 release and the annotations used by the paper."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def data_root(value="data_v2"):
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (REPO_ROOT / p).resolve()


def run_root(value="runs_revision"):
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (REPO_ROOT / p).resolve()


def annotation_file(data_dir, scheme, part, annotation_set="benchmark"):
    base = Path(data_dir) / "annotations"
    folder = "benchmark" if annotation_set == "benchmark" else "coco"
    path = base / folder / f"{scheme}_{part}.json"
    # Read older local v2 layouts without writing or rebuilding their annotations.
    if annotation_set == "benchmark" and not path.exists():
        old = base / f"{scheme}_{part}.json"
        if old.exists():
            return old
    if not path.exists():
        raise FileNotFoundError(f"Missing {annotation_set} annotations: {path}. Extract data_v2.zip first.")
    return path


def require_semantic_data(data_dir):
    for sub in ("images", "masks_grayscale", "splits"):
        if not (Path(data_dir) / sub).is_dir():
            raise FileNotFoundError(f"Missing {Path(data_dir) / sub}; extract the v2 release first.")
