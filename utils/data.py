import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split


DEFAULT_REPO_ID = "thanglexuan/murincells"
DEFAULT_FILENAME = "data.zip"
REQUIRED_DIRS = ("images", "masks", "masks_grayscale", "annotations")


def has_extracted_data(data_dir: str | Path) -> bool:
    data_dir = Path(data_dir)
    return all((data_dir / name).is_dir() for name in REQUIRED_DIRS)


def safe_extract(zip_path: str | Path, output_dir: str | Path) -> None:
    output_dir = Path(output_dir).resolve()
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for member in zip_ref.infolist():
            target_path = (output_dir / member.filename).resolve()
            try:
                target_path.relative_to(output_dir)
            except ValueError:
                raise RuntimeError(f"Unsafe path in zip file: {member.filename}")
        zip_ref.extractall(output_dir)


def prepare_data(
    data_dir: str | Path = "data",
    repo_id: str = DEFAULT_REPO_ID,
    filename: str = DEFAULT_FILENAME,
    token: str | None = None,
    force: bool = False,
) -> Path:
    data_dir = Path(data_dir)

    if has_extracted_data(data_dir) and not force:
        print(f"Data already exists: {data_dir.resolve()}")
        return data_dir

    print(f"Downloading {filename} from Hugging Face dataset {repo_id}")
    zip_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            token=token,
        )
    )

    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting to {data_dir.resolve()}")
    safe_extract(zip_path, data_dir)

    missing_dirs = [name for name in REQUIRED_DIRS if not (data_dir / name).is_dir()]
    if missing_dirs:
        raise FileNotFoundError(
            f"Data extracted, but missing expected directories: {missing_dirs}"
        )

    return data_dir


def make_splits(data_dir: str | Path, test_size: float = 0.3, random_state: int = 42):
    data_dir = Path(data_dir)
    images_dir = data_dir / "images"
    masks_dir = data_dir / "masks"

    image_ls = sorted(path.name for path in images_dir.iterdir() if path.is_file())
    mask_ls = sorted(path.name for path in masks_dir.iterdir() if path.is_file())

    if image_ls != mask_ls:
        missing_masks = sorted(set(image_ls) - set(mask_ls))
        missing_images = sorted(set(mask_ls) - set(image_ls))
        raise ValueError(
            "Image and mask filenames do not match. "
            f"missing_masks={missing_masks[:10]}, missing_images={missing_images[:10]}"
        )

    x_train, x_temp, y_train, y_temp = train_test_split(
        image_ls,
        mask_ls,
        test_size=test_size,
        random_state=random_state,
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.5,
        random_state=random_state,
    )

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
    }


download_data = prepare_data
