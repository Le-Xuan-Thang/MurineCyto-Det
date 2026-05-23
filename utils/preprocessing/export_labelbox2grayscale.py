# Export labeled data từ Labelbox thành ảnh + mask grayscale (background=0)
import os, json, io, concurrent.futures as cf
import labelbox as lb
from PIL import Image, ImageFile
import numpy as np
import requests
from requests.adapters import HTTPAdapter, Retry
from tqdm.auto import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


def make_http_session():
    """HTTP session with retry"""
    sess = requests.Session()
    retries = Retry(
        total=5,
        connect=3,
        read=3,
        backoff_factor=0.3,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))
    sess.mount("https://", HTTPAdapter(max_retries=retries))
    return sess


def download_image(url, headers=None, timeout=10, session=None):
    """Download image from url"""
    s = session or make_http_session()
    try:
        r = s.get(url, headers=headers or {}, timeout=timeout)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content))
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None


def _mask_to_bool(mask_img, target_size):
    """Convert mask to boolean array"""
    if mask_img.mode not in ("1", "L"):
        if "A" in mask_img.getbands():
            mask_img = mask_img.getchannel("A")
        else:
            mask_img = mask_img.convert("L")
    if mask_img.size != target_size:
        mask_img = mask_img.resize(target_size, Image.NEAREST)
    return np.asarray(mask_img, dtype=np.uint8) > 0


def export_labelbox_data(
    api_key, project_id, output_dir, export_params, filters, label2id, max_workers=6
):
    """Export Labelbox masks -> grayscale segmentation"""
    client = lb.Client(api_key=api_key)
    project = client.get_project(project_id)

    # Export task
    export_task = project.export_v2(params=export_params, filters=filters)
    export_task.wait_till_done()
    if export_task.errors:
        print(export_task.errors)

    data = export_task.result

    # I/O dirs
    IMAGE_DIR = os.path.join(output_dir, "images")
    MASK_DIR = os.path.join(output_dir, "masks_grayscale")
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(MASK_DIR, exist_ok=True)

    session = make_http_session()

    for idx, item in enumerate(
        tqdm(data, total=len(data), desc="Images", unit="img"), start=1
    ):
        # ảnh gốc
        image_url = item["data_row"]["row_data"]
        base_image = download_image(image_url, session=session)
        if base_image is None:
            continue
        if base_image.mode != "RGB":
            base_image = base_image.convert("RGB")

        W, H = base_image.size
        mask_combined = np.zeros((H, W), dtype=np.uint8)  # background=0

        # danh sách masks
        objects = item["projects"][project_id]["labels"][0]["annotations"]["objects"]

        def fetch_one(obj):
            url = obj["mask"]["url"]
            img = download_image(url, headers=client.headers, session=session)
            return obj, img

        with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(fetch_one, m) for m in objects]
            for fut in tqdm(
                cf.as_completed(futures),
                total=len(futures),
                leave=False,
                desc="  Masks",
                unit="mask",
            ):
                try:
                    obj, mask_img = fut.result()
                except Exception as e:
                    print(f"Mask fetch error: {e}")
                    continue
                if mask_img is None:
                    continue

                label = obj.get("name")
                class_id = label2id.get(
                    label, 0
                )  # nếu label không có trong dict -> background

                mask_bool = _mask_to_bool(mask_img, (W, H))
                mask_combined[mask_bool] = class_id  # gán class_id vào vùng mask

        # save
        ext_id = item["data_row"]["external_id"]
        fname = ext_id.split(".")[0] + f"_{idx}." + ext_id.split(".")[-1]
        base_path = os.path.join(IMAGE_DIR, fname)
        mask_path = os.path.join(MASK_DIR, fname)

        base_image.save(base_path)
        Image.fromarray(mask_combined, mode="L").save(mask_path)


if __name__ == "__main__":
    root_dir = os.getcwd()
    data_dir = os.path.join(root_dir, "data")
    labelbox_dir = os.path.join(root_dir, "labelbox")
    token_path = os.path.join(labelbox_dir, "token.json")

    with open(token_path, "r") as f:
        token_data = json.load(f)
        API_KEY = token_data["api_key"]
        PROJECT_ID = token_data["project_id"]

    export_params = {
        "attachments": True,
        "metadata_fields": True,
        "data_row_details": True,
        "project_details": False,
        "label_details": True,
        "performance_details": False,
        "interpolated_frames": True,
        "embeddings": False,
    }

    filters = {}

    label2id = {
        "Background": 0,
        "Marcophage/Monocyte": 1,
        "Neutrophil": 2,
        "Eosinophil": 3,
        "Lymphocyte": 4,
        "Unknown cell/Debris": 5,
    }

    export_labelbox_data(
        API_KEY,
        PROJECT_ID,
        output_dir=data_dir,
        export_params=export_params,
        filters=filters,
        label2id=label2id,
        max_workers=4,
    )
