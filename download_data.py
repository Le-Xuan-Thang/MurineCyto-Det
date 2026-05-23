import argparse
import os

from dotenv import load_dotenv

from utils.data import DEFAULT_FILENAME, DEFAULT_REPO_ID, make_splits, prepare_data


def parse_args():
    parser = argparse.ArgumentParser(description="Download and extract MurineCyto data.")
    parser.add_argument("--data-dir", default="data", help="Output data directory.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face dataset ID.")
    parser.add_argument("--filename", default=DEFAULT_FILENAME, help="Dataset zip filename.")
    parser.add_argument("--force", action="store_true", help="Download and extract again.")
    parser.add_argument(
        "--skip-split",
        action="store_true",
        help="Only download/extract data, without printing split sizes.",
    )
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()
    token = os.getenv("HF_TOKEN")

    data_dir = prepare_data(
        data_dir=args.data_dir,
        repo_id=args.repo_id,
        filename=args.filename,
        token=token,
        force=args.force,
    )

    if not args.skip_split:
        splits = make_splits(data_dir)
        print(f"train size:{len(splits['x_train'])}")
        print(f"val size:{len(splits['x_val'])}")
        print(f"test size:{len(splits['x_test'])}")


if __name__ == "__main__":
    main()
