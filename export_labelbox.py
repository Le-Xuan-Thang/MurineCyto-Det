import os, json, io, concurrent.futures as cf
import labelbox as lb
from PIL import Image
import numpy as np
import requests
from requests.adapters import HTTPAdapter, Retry
from tqdm.auto import tqdm
from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True

def make_http_session():
    sess = requests.Session()
    retries = Retry(
        total=5, connect=3, read=3, backoff_factor=0.3,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD")
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))
    sess.mount("https://", HTTPAdapter(max_retries=retries))
    return sess

def download_image(url, headers=None, timeout=10, session=None):
    """Tải ảnh bằng requests, trả về PIL.Image hoặc None."""
    s = session or make_http_session()
    try:
        r = s.get(url, headers=headers or {}, timeout=timeout)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content))
    except requests.HTTPError as e:
        print(f"HTTP Error: {e.response.status_code} for URL: {url}")
    except requests.RequestException as e:
        print(f"Request Error: {e} for URL: {url}")
    except Exception as e:
        print(f"Unknown Error: {e} for URL: {url}")
    return None
def _mask_to_bool(mask_img, target_size):
    """Đưa mask về nhị phân bool với kích thước ảnh gốc."""
    if mask_img.mode not in ("1", "L"):
        # nếu có alpha channel, ưu tiên alpha làm mask
        if "A" in mask_img.getbands():
            mask_img = mask_img.getchannel("A")
        else:
            mask_img = mask_img.convert("L")
    if mask_img.size != target_size:
        mask_img = mask_img.resize(target_size, Image.NEAREST)
    # coi mọi giá trị >0 là True
    return np.asarray(mask_img, dtype=np.uint8) > 0

def export_labelbox_data(
    api_key,
    project_id,
    output_dir,
    export_params,
    filters,
    label_colors,
    max_workers=6
):
    """Export + combine colored masks nhanh & ổn định hơn."""
    # Labelbox client
    client = lb.Client(api_key=api_key)

    # Project
    project = client.get_project(project_id)

    # Export
    export_task = project.export_v2(params=export_params, filters=filters)
    export_task.wait_till_done()
    if export_task.errors:
        print(export_task.errors)

    data = export_task.result

    # I/O dirs
    IMAGE_DIR = os.path.join(output_dir, "images")
    MASK_DIR  = os.path.join(output_dir, "masks")
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(MASK_DIR,  exist_ok=True)

    # HTTP session (reuse + retry)
    session = make_http_session()

    for item in tqdm(data, total=len(data), desc="Images", unit="img"):
        # ảnh gốc
        image_url = item["data_row"]["row_data"]
        base_image = download_image(image_url, session=session)
        if base_image is None:
            continue
        if base_image.mode != "RGB":
            base_image = base_image.convert("RGB")

        W, H = base_image.size
        mask_combined = np.zeros((H, W, 3), dtype=np.uint8)

        # danh sách masks
        objects = item["projects"][project_id]["labels"][0]["annotations"]["objects"]

        # tải masks song song (I/O bound)
        def fetch_one(obj):
            url = obj["mask"]["url"]
            img = download_image(url, headers=client.headers, session=session)
            return obj, img

        with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(fetch_one, m) for m in objects]
            for fut in tqdm(cf.as_completed(futures), total=len(futures), leave=False, desc="  Masks", unit="mask"):
                try:
                    mask, mask_img = fut.result()
                except Exception as e:
                    print(f"Mask fetch error: {e}")
                    continue
                if mask_img is None:
                    continue

                label = mask.get("name")
                color = label_colors.get(label, (0, 0, 0))

                mask_bool = _mask_to_bool(mask_img, (W, H))

                # gán màu bằng broadcast một lần
                # nơi mask_bool=True, ghi đè màu
                mask_combined = np.where(mask_bool[..., None], np.array(color, dtype=np.uint8), mask_combined)

        # lưu file
        ext_id = item["data_row"]["external_id"]
        base_path = os.path.join(IMAGE_DIR, f"{ext_id}")
        mask_path = os.path.join(MASK_DIR,  f"masks_{ext_id}")

        base_image.save(base_path)
        Image.fromarray(mask_combined, mode="RGB").save(mask_path)

if __name__ == "__main__":
    with open('token.json','r') as f:
        token_data = json.load(f)
        API_KEY = token_data['api_key']
        PROJECT_ID = token_data['project_id']

    data_dir = "data"

    # defind export parameters of labelbox
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

    filters = {
    }

    label_colors = {
        "Marcophage/Monocyte": (28, 230, 255),  # #1CE6FF
        "Neutrophil": (255, 52, 255),           # #FF34FF
        "Eosinophil": (255, 74, 70),            # #FF4A46
        "Lymphocyte": (0, 137, 65),             # #008941
        "Unknown cell/Debris": (0, 111, 166),   # #006FA6
        "Basophil": (163, 0, 89)                # #A30059
    }

    export_labelbox_data(
        API_KEY,PROJECT_ID,
        output_dir=data_dir, 
        export_params=export_params, 
        filters=filters,
        label_colors=label_colors,
        max_workers=4)
