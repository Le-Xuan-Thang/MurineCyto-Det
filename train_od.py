"""Compatibility entry point; revised detection training lives in revision/train_det.py."""
import sys
from revision.train_det import main

if __name__ == "__main__":
    print("Using the revised mouse-level benchmark. See README.md for the v2 CLI.", file=sys.stderr)
    sys.argv = ["fasterrcnn" if s == "faster_rcnn" else s for s in sys.argv]
    if "--model" not in sys.argv and "--help" not in sys.argv and "-h" not in sys.argv:
        sys.argv.extend(["--model", "fasterrcnn"])
    main()
