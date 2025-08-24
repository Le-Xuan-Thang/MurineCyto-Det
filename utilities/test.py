import os, json, io, concurrent.futures as cf
import labelbox as lb
from PIL import Image, ImageFile
import numpy as np
import requests
from requests.adapters import HTTPAdapter, Retry
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

ImageFile.LOAD_TRUNCATED_IMAGES = True

from export_labelbox import make_http_session, download_image, _mask_to_bool


root_dir = "/Users/lexuanthang/OneDrive/WORKING/Projects/CellDetection/Code/Murincells/"
data_dir = os.path.join(root_dir, "data")
labelbox_dir = os.path.join(root_dir, "labelbox")
token_path = os.path.join(labelbox_dir, "token.json")

with open(token_path, "r") as f:
    token_data = json.load(f)
    API_KEY = token_data["api_key"]
    PROJECT_ID = token_data["project_id"]
"""Export Labelbox masks -> grayscale segmentation"""
client = lb.Client(api_key=API_KEY)
project = client.get_project(PROJECT_ID)

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
    "Basophil": 6,
}

# Export task
export_task = project.export_v2(params=export_params, filters=filters)
export_task.wait_till_done()
if export_task.errors:
    print(export_task.errors)

data = export_task.result

# I/O dirs
IMAGE_DIR = os.path.join(data_dir, "images")
MASK_DIR = os.path.join(data_dir, "masks_test")
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)

session = make_http_session()

for item in tqdm(data[0:1], desc="Images: ", unit="img"):
    image_url = item["data_row"]["row_data"]
    base_image = download_image(image_url, session=session)
    W, H = base_image.size

    mask_combined = np.zeros((H, W), dtype=np.int8)

    # danh sách masks
    objects = item["projects"][PROJECT_ID]["labels"][0]["annotations"]["objects"]

    for mask in tqdm(objects[0:1], desc="Masks", unit="msk"):
        mask_url = mask["mask"]["url"]
        mask_image = download_image(mask_url, headers=client.headers, session=session)
        label = mask.get("name")
        class_id = label2id.get(label, 0)

    ext_id = item["data_row"]["external_id"]
    print(ext_id)

plt.subplots(1, 2, figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(base_image)
plt.subplot(1, 2, 2)
plt.imshow(mask_image)
plt.show()
