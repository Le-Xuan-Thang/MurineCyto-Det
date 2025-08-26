import os
import json
from huggingface_hub import hf_hub_download
import zipfile

# Get dirs
print("be carefull when you use different system. remember to change root_dir")
root_dir = os.getcwd()
data_dir = os.path.join(root_dir, "data")
test_dir = os.path.join(root_dir, "data_test")
with open("labelbox/huggingface.json", "r") as f:
    token_data = json.load(f)

# Expected subfolders
required_folders = ["images", "masks", "masks_grayscale", "annotations"]

all_exist = all(os.path.exists(os.path.join(data_dir, f)) for f in required_folders)
if not all_exist:
    print("Data doesn't exists data folder")

# Download data from HF hub
    zip_path = hf_hub_download(
        repo_id="thanglexuan/murincells",
        filename="data.zip",
        repo_type="dataset",
        token=token_data["token"],
        local_dir=test_dir,
    )

    print("Download complete!!!")
    print("Extracting ....")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(test_dir)

    print("All files were extracted")
else:
    print("Dataset already exists locally")

# 
