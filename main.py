from pathlib import Path

from utils.data import REQUIRED_DIRS, has_extracted_data, make_splits


def main():
    data_dir = Path("data")
    print("MurineCyto project health check")
    print(f"data_dir: {data_dir.resolve()}")

    if not has_extracted_data(data_dir):
        missing = [name for name in REQUIRED_DIRS if not (data_dir / name).is_dir()]
        print(f"missing data directories: {', '.join(missing)}")
        print("run: uv run murine-download")
        return

    splits = make_splits(data_dir)
    print(f"train size: {len(splits['x_train'])}")
    print(f"val size: {len(splits['x_val'])}")
    print(f"test size: {len(splits['x_test'])}")
    print("status: ok")


if __name__ == "__main__":
    main()
