import os, json, concurrent.futures as cf
import labelbox as lb
import numpy as np
from PIL import Image
import cv2
from datetime import datetime
from tqdm.auto import tqdm
from utils.preprocessing.export_labelbox import make_http_session, download_image

# Define coco format
coco_data = {}
coco_data = {
    "info": {
        "description": "Mourin Cell Detection Dataset",
        "url": "hhttps://www.kaggle.com/datasets/thanglexuan/cell-detection/data",
        "version": "3.0",
        "year": 2025,
        "contributor": "Research Team - Luan Vu, Lan Anh, Thang Le",
        "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    },
    "licenses": [
        {
            "id": 1,
            "name": "Research Use Only",
            "url": "following the license terms of the dataset",
        }
    ],
    # CATEGORIES
    "categories": [
        {"id": 1, "name": "Marcophage/Monocyte", "supercategory": "mourin_cells"},
        {"id": 2, "name": "Neutrophil", "supercategory": "mourin_cells"},
        {"id": 3, "name": "Eosinophil", "supercategory": "mourin_cells"},
        {"id": 4, "name": "Lymphocyte", "supercategory": "mourin_cells"},
        {"id": 5, "name": "Unknown cell/Debris", "supercategory": "mourin_cells"},
    ],
    # Empty lists for images and annotations - will be populated dynamically
    "images": [],
    "annotations": [],
}

# Step 2: data from Labelbox
# get all file names in data/images directory
root_dir = os.getcwd()
data_dir = os.path.join(root_dir, "data")
labelbox_dir = os.path.join(root_dir, "labelbox")
token_path = os.path.join(labelbox_dir, "token.json")
labelbox_path = os.path.join(labelbox_dir, "labelbox.json")

with open(token_path, "r") as f:
    token_data = json.load(f)
    API_KEY = token_data["api_key"]
    PROJECT_ID = token_data["project_id"]

# Labelbox client
client = lb.Client(api_key=API_KEY)

# Project
project = client.get_project(PROJECT_ID)

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

filters = {}

# Export
export_task = project.export_v2(params=export_params, filters=filters)
export_task.wait_till_done()
if export_task.errors:
    print(export_task.errors)


data = export_task.result

IMAGE_DIR = os.path.join(data_dir, "images")
MASK_DIR = os.path.join(data_dir, "masks")
ANNOTATION_DIR = os.path.join(data_dir, "annotations")

# Create annotations directory if it doesn't exist
os.makedirs(ANNOTATION_DIR, exist_ok=True)


def check_catagories(mask_name, coco_data):
    """
    Check if categories are defined in the coco data
    If not, define categories with id 1 and name "cell"
    """
    # take categories from coco_data
    categories = coco_data.get("categories", [])

    # Check if the mask_name exists in categories
    for category in categories:
        if category["name"] == mask_name:
            return category["id"]


def category_id_to_name(category_id, coco_data):
    """
    Convert category ID to category name
    """
    categories = coco_data.get("categories", [])
    for category in categories:
        if category["id"] == category_id:
            return category["name"]
    return None


## MAIN PART: export masks to bboxes
def process_single_mask(mask_data, mask_idx, client, coco_data, image_idx):
    """Xử lý một mask đơn lẻ và trả về annotation"""
    mask_url = mask_data["mask"]["url"]
    mask_name = mask_data["name"]

    # PHƯƠNG ÁN 1: Session riêng cho mỗi thread (KHUYẾN NGHỊ)
    session_local = make_http_session()

    # PHƯƠNG ÁN 2: Sử dụng session chung (CẨN THẬN với thread safety)
    # session_local = session  # Truyền từ bên ngoài

    category_id = check_catagories(mask_name, coco_data)
    assert category_id is not None, f"Mask ID is None for {mask_name}"
    assert mask_url is not None, f"Mask URL is None for {mask_name}"

    # Download mask
    mask_image = download_image(mask_url, headers=client.headers, session=session_local)
    assert isinstance(
        mask_image, Image.Image
    ), f"Downloaded mask is not a PIL Image: {type(mask_image)}"

    # Convert to numpy array
    mask_array = np.array(mask_image)
    assert mask_array.ndim == 2, f"Mask array should be 2D, got {mask_array.ndim}D"

    # Find contours and bounding boxes
    contours, *_ = cv2.findContours(
        mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    annotations = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bbox = [x, y, w, h]
        area = w * h
        annotation = {
            "id": mask_idx + 1,  # Unique ID for each annotation
            "image_id": image_idx,
            "mask_url": mask_url,  # URL of the mask image
            "category_id": category_id,
            "bbox": bbox,
            "area": area,
            "iscrowd": 0,
        }
        annotations.append(annotation)
    return annotations


# ======= PHƯƠNG ÁN 1: Session riêng cho mỗi thread (HIỆN TẠI) =======

# Main processing loop
session = make_http_session()

MAX_WORKERS = 4
for idx, item in enumerate(
    tqdm(data, total=len(data), desc="Processing ...", unit="img"), start=1
):
    # Get image data
    image_name = item["data_row"]["external_id"]
    image_url = item["data_row"]["row_data"]
    base_image = download_image(image_url, session=session)
    assert base_image is not None, f"Failed to download image from {image_url}"
    assert isinstance(
        base_image, Image.Image
    ), f"Downloaded image is not a PIL Image: {type(base_image)}"

    # Get image size from media attributes
    W = item["media_attributes"]["width"]
    H = item["media_attributes"]["height"]
    assert W > 0 and H > 0, f"Invalid image dimensions: {W}x{H}"

    # Add size of image to coco data
    coco_data["images"].append(
        {
            "id": idx,
            "width": W,
            "height": H,
            "image_name": image_name,
            "image_url": image_url,
        }
    )

    masks_data = item["projects"][PROJECT_ID]["labels"][0]["annotations"]["objects"]

    # Xử lý masks song song với ThreadPoolExecutor
    all_annotations = []
    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit tất cả mask tasks
        future_to_mask = {}
        for mask_idx, mask in enumerate(masks_data, start=1):
            future = executor.submit(
                process_single_mask, mask, mask_idx, client, coco_data, idx
            )
            future_to_mask[future] = mask_idx

        # Collect results từ tất cả futures với progress bar
        for future in tqdm(
            cf.as_completed(future_to_mask),
            total=len(future_to_mask),
            desc=f"Processing masks in {image_name}",
            unit="mask",
        ):
            try:
                annotations = future.result()
                all_annotations.extend(annotations)
            except Exception as e:
                mask_idx = future_to_mask[future]
                print(f"Error processing mask {mask_idx} in {image_name}: {e}")

    # Thêm tất cả annotations vào coco_data
    coco_data["annotations"].extend(all_annotations)

    # Save the coco_data to a JSON file
    file_name = image_name.split(".")[0] + f"_{idx}"
    annotation_file = os.path.join(ANNOTATION_DIR, f"{file_name}.json")
    with open(annotation_file, "w") as f:
        json.dump(coco_data, f, indent=4)
print("Complete processing data into coco format")
