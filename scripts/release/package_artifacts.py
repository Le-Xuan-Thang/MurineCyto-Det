"""Package a whitelist of completed-run artifacts for Zenodo, never for Git history."""
import argparse
import json
from pathlib import Path
import zipfile


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-runs", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    files = []
    for category in ("seg", "det"):
        for run in sorted((args.source_runs / category).iterdir()):
            if not run.is_dir():
                continue
            patterns = ["config.json", "log.json", "best.pth", "test_pred.npz", "test_pred.json", "val_pred.json",
                        "ultra/weights/best.pt", "ultra/results.csv"]
            selected = [run / rel for rel in patterns if (run / rel).is_file()]
            assert (run / "config.json").is_file(), run
            files.extend(selected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
        for p in files:
            z.write(p, Path("runs_revision") / p.relative_to(args.source_runs))
        z.writestr("runs_revision/ARTIFACTS_README.txt", "Completed-run artifacts for MurineCyto-Det v2.\nExtract next to scripts/ and data_v2/.\nPrimary benchmark: mouse_holdout (seeds 0,1,2).\nrandom_tile runs are a secondary single-seed comparison only.\nDefault mask ground truth: data_v2/annotations/benchmark/.\nCheckpoint selection and reported AP50:95 use maxDets=100; AP50/AP75 use 300.\nNo credentials, raw Labelbox exports, caches or training images are included.\n")
    print(json.dumps({"archive": str(args.output), "files": len(files), "bytes": args.output.stat().st_size}))


if __name__ == "__main__":
    main()
