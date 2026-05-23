# %%
import os
import glob
import pandas as pd
import matplotlib
matplotlib.use("Agg")    
import matplotlib.pyplot as plt
import seaborn as sns
import segmentation_models_pytorch as smp
import torch
import random

# %% Common setup of plt
plt.rcParams.update(
    {
        "font.family": "Courier New",
        "font.size": 22,
        "axes.titlesize": 22,
        "axes.labelsize": 22,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "legend.fontsize": 22,
        "figure.titlesize": 22,
    }
)
plt.rcParams["lines.markeredgewidth"] = 0.5
plt.rcParams["lines.markeredgecolor"] = "black"

# %% all directories
root_dir = os.getcwd()
logs_dir = os.path.join(root_dir, "logs")
unet_model_dir = os.path.join(logs_dir, "unet")
data_dir = os.path.join(root_dir, "data")


# %% get example images
images_dir = os.path.join(data_dir, "images")
masks_dir = os.path.join(data_dir, "masks")
annotations_dir = os.path.join(data_dir, "annotations")

image_ls = os.listdir(images_dir)
idx = random.randint(0,len(image_ls))

image_path = os.path.join(images_dir, image_ls[idx])
mask_path = os.path.join(masks_dir, image_ls[idx])
ann_path = os.path.join(annotations_dir, image_ls[idx]).replace("png","json")

# Read images
image = plt.imread(image_path)
mask = plt.imread(mask_path)
matplotlib.use("QtAgg")

# Plot images side by side
fig, ax = plt.subplots(1, 2, figsize=(8, 6))
ax[0].imshow(image)
ax[0].set_title("Image")
ax[0].axis("off")

ax[1].imshow(mask)
ax[1].set_title("Mask")
ax[1].axis("off")

plt.tight_layout()
# plt.show()

# %% plot annotation
import json
from utils.det_helpers import visualize
with open(ann_path) as f:
    ann = json.load(f)

categories_name = [cell['name'] for cell in ann['categories']]
categories_id = [cell['id'] for cell in ann['categories']]
category_id_to_name = dict(zip(categories_id, categories_name))

bboxes = [cell['bbox'] for cell in ann['annotations']]
category_ids = [cell['category_id'] for cell in ann['annotations']]

print(f"Visualizing image: {image_ls[idx]}")

# --- Visualization ---
fig, ax = plt.subplots(1, 3, figsize=(9, 6))

ax[0].imshow(image)
ax[0].axis("off")
ax[0].set_title("Image")

ax[1].imshow(mask)
ax[1].axis("off")
ax[1].set_title("Mask")

visualize(image, bboxes, category_ids, category_id_to_name, ax=ax[2])
ax[2].axis("off")
ax[2].set_title("Annotated")

# tighten spacing between axes
fig.subplots_adjust(wspace=0.05, hspace=10)

plt.savefig("data_example.pdf", dpi=300, bbox_inches="tight")
# plt.show()


# %% get metrics result and model
def get_csv_file_path(path=str):
    if path.endswith(".csv") and os.path.isfile(path):
        csv_file_path = path
    else:
        # tìm version mới nhất trong folder
        version_dirs = glob.glob(os.path.join(path, "version_*"))
        if not version_dirs:
            raise FileNotFoundError(f"No version_x folders found in {path}")
        latest_version = max(version_dirs, key=os.path.getmtime)
        csv_file_path = os.path.join(latest_version, "metrics.csv")
    return csv_file_path

latest_version_unet = "version_12"
metrics_csv_unet = os.path.join(unet_model_dir, latest_version_unet, "metrics.csv")
print(metrics_csv_unet)
unet_csv_file_path = get_csv_file_path(metrics_csv_unet)
seg_model_dir = os.path.join(logs_dir, "segformer") 
segformer_csv_file_path = get_csv_file_path(seg_model_dir)

# Load data of unet model
df_unet = pd.read_csv(unet_csv_file_path)
df_unet_train = df_unet.dropna(subset=["train_loss"])
df_unet_val = df_unet.dropna(subset=["valid_loss"])
df_unet_train = df_unet_train.dropna(axis=1, how='all')
df_unet_val = df_unet_val.dropna(axis=1, how='all')
print("Metrics dataset of UNet model:")
print(df_unet_train.head(5))

# Load data of segformer model
df_segformer = pd.read_csv(segformer_csv_file_path) 
df_segformer_train = df_segformer.dropna(subset=["train_loss"])
df_segformer_val = df_segformer.dropna(subset=["valid_loss"])
df_segformer_train = df_segformer_train.dropna(axis=1, how='all')
df_segformer_val = df_segformer_val.dropna(axis=1, how='all')
print("Metrics dataset of SegFormer model:")
print(df_segformer_train.head(5))

# %% Plot metrics curve

markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'H', '8']
colors = plt.colormaps['tab10'].colors


def plot_each_metric(dfs: dict, x_col="epoch", save_dir=None, prefix=str):
    """
    Plot each metric and compare across multiple DataFrames.

    Args:
        dfs (dict): Dictionary where keys are names and values are pandas DataFrames.
        x_col (str): Column name for the X-axis (default: 'epoch').
        save_dir (str): Directory to save figures (optional).
        prefix (str): add prefix to file name
    """
    if not dfs:
        print("No dataframes provided.")
        return

    # Get list of metric columns from the first DataFrame
    first_df = next(iter(dfs.values()))
    metric_cols = [c for c in first_df.columns if c not in ["epoch", "step"]]

    for col in metric_cols:
        fig, ax = plt.subplots(figsize=(8, 6))

        for idx, (name, df) in enumerate(dfs.items()):
            if col not in df.columns:
                print(f"⚠️ Column '{col}' not found in dataframe '{name}', skipping.")
                continue

            ax.plot(
                df[x_col], df[col],
                label=name,
                marker=markers[idx % len(markers)],
                color=colors[idx % len(colors)],
                markersize=6,
                linewidth=2,
            )

        ax.set_xlabel(x_col.capitalize())
        ax.set_ylabel(col)
        # ax.set_title(f"{col}", fontsize=12)
        ax.grid(True, which="major", linestyle="-", linewidth=0.25, alpha=0.35)
        ax.minorticks_on()
        ax.grid(True, which="minor", linestyle="-", linewidth=0.15, alpha=0.25)

        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.15),
            frameon=False,
            ncol=2,
        )

        fig.tight_layout()

        # Save or show
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            file_name = f"{prefix}-{col}.pdf"
            file_path = os.path.join(save_dir, file_name)
            plt.savefig(file_path, bbox_inches="tight", dpi=300)
            print(f"✅ Saved image: {file_path}")
            plt.close(fig)  # Close after saving to avoid memory leak
        else:
            plt.show()

dfs_train = {
  "unet": df_unet_train,
  "segformer": df_segformer_train
}
dfs_val = {
  "UNet": df_unet_val,
  "Segformer": df_segformer_val
}

save_dir = os.path.join(root_dir, "figures")
plot_each_metric(dfs_train,save_dir=save_dir, prefix="seg")
plot_each_metric(dfs_val, save_dir=save_dir, prefix="seg")

# %% plot metrics bar
from utils.helpers import plot_final_metrics_bar
metrics = ["acc", "dice", "miou", "gds", "loss"]
train_results = plot_final_metrics_bar(dfs_train, metrics, save_dir=save_dir, prefix="seg_train", prefix_col="train_")
results = plot_final_metrics_bar(dfs_val, metrics, save_dir=save_dir, prefix="seg_valid", prefix_col="valid_")
print(results)

# %% Load dataset
import zipfile
from sklearn.model_selection import train_test_split
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader
from seg_v1 import MurineCellModel
from seg_v1 import Dataset
from seg_v1 import get_validation_augmentation
import os
from dotenv import load_dotenv

load_dotenv()
token = os.getenv('HF_TOKEN')
root_dir = os.getcwd()
model_dir = os.path.join(root_dir, "models")
os.makedirs(model_dir, exist_ok=True)
data_zip_path = hf_hub_download(
    repo_id = "thanglexuan/murincells",
    filename = "data.zip",
    repo_type = "dataset",
    token = token
)
data_dir = os.path.join(root_dir, "data")
os.makedirs(data_dir, exist_ok=True)
with zipfile.ZipFile(data_zip_path, "r") as zip_ref:
    zip_ref.extractall(data_dir)
    
images_dir = os.path.join(data_dir, "images")
masks_dir = os.path.join(data_dir, "masks")
image_ls = sorted(os.listdir(images_dir))
mask_ls = sorted(os.listdir(masks_dir))

x_train_dir, x_temp_dir, y_train_dir, y_temp_dir = train_test_split(
    image_ls, mask_ls, test_size=0.3, random_state=42
)
x_val_dir, x_test_dir, y_val_dir, y_test_dir = train_test_split(
    x_temp_dir, y_temp_dir, test_size=0.5, random_state=42
)
print("train size:{}".format(len(x_train_dir)))
print("val size:{}".format(len(x_val_dir)))
print("test size:{}".format(len(x_test_dir)))

# %% Split dataset
test_dataset = Dataset(
    data_dir=data_dir,
    images_dir=x_test_dir,
    masks_dir=y_test_dir,
    augmentation=get_validation_augmentation(),
)

# Change to > 0 if not on Windows machine
BATCH_SIZE = 8
NUM_WORKERS = 0
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
# %%
model_dir = os.path.join(root_dir, "models")
unet_ckpt_path = os.path.join(model_dir, "unet.ckpt")
segformer_ckpt_path = os.path.join(model_dir, "segformer.ckpt")
NUM_CLASSES = len(test_dataset.CLASSES)

unet_loaded_model = MurineCellModel.load_from_checkpoint(
    unet_ckpt_path,
    arch='Unet',
    encoder_name="resnet34",
    in_channels=3,
    num_classes=NUM_CLASSES
)
segformer_loaded_model = MurineCellModel.load_from_checkpoint(
    segformer_ckpt_path, 
    arch='Segformer',
    encoder_name="resnet34",
    in_channels=3,
    num_classes=NUM_CLASSES
)
# %%
from utils.helpers import predict_and_plot_multi

models = {
    "U-Net": unet_loaded_model,
    "SegFormer": segformer_loaded_model,
}

predict_and_plot_multi(
    models=models,
    dataloader=test_loader,
    n_samples=3,
    save_dir=save_dir,
    prefix="seg-"
)

# %%
from utils.helpers import visualize_batch
# # lấy 1 batch từ test_loader
CLASSES = [
    "Background",
    "Macrophage/Monocyte",
    "Neutrophil",
    "Eosinophil",
    "Lymphocyte",
    "Unknown cell/Debris"
]
for i, (images, masks) in enumerate(test_loader):
    # check if this batch contains any pixels == 4 (Lymphocyte class)
    if (masks == 4).any():
        print(f"Found Lymphocyte in batch {i}, shape: {masks.shape}")
        idx = (masks == 4).nonzero(as_tuple=True)[0][0].item()
        print(f"First sample index in batch with Lymphocyte: {idx}")
        # take one sample to visualize or test
        image = images[idx]
        mask  = masks[idx]
        break
# images, masks = next(iter(test_loader))
# print(idx)
# visualize_batch(images, masks, CLASSES, n=idx)
# %%  Confusion matrix
from utils.helpers import plot_confusion_matrix
# predict
with torch.no_grad():
    logits = unet_loaded_model(images.to("cuda"))
    preds = torch.argmax(logits, dim=1).cpu().numpy()

# lấy ảnh đầu tiên trong batch
y_true = masks[idx].numpy()
y_pred = preds[idx]


SHORT_CLASSES = [
    "BG",      # Background
    "Macro",   # Macrophage/Monocyte
    "Neut",    # Neutrophil
    "Eos",     # Eosinophil
    "Lymph",   # Lymphocyte
    "Debris"   # Unknown cell/Debris
]
plot_confusion_matrix(y_true, y_pred, SHORT_CLASSES, ignore_index=0, normalize=True, save_dir=save_dir, prefix = "seg-")


# predict
with torch.no_grad():
    logits = segformer_loaded_model(images.to("cuda"))
    preds = torch.argmax(logits, dim=1).cpu().numpy()

# lấy ảnh đầu tiên trong batch
y_true_2 = masks[idx].numpy()
y_pred_2 = preds[idx]

plot_confusion_matrix(y_true_2, y_pred_2, SHORT_CLASSES, ignore_index=0, normalize=True, save_dir=save_dir, prefix = "seg1-")


