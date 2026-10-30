"""Replacement for the historical contour-level exporter; consumes public v2 COCO."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from revision.export_per_image import main

if __name__ == "__main__":
    main()
