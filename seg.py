from json import encoder
import os
import random
import numpy as np

import matplotlib.pyplot as plt
import cv2

from sklearn.model_selection import train_test_split

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
from torch.utils.data import Dataset, DataLoader

import torchmetrics
from torchmetrics import Dice, JaccardIndex

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.metrics import iou_score, accuracy, f1_score, recall
from tqdm.auto import tqdm

# helper function

from utilities.plot_segmentation import plot_segmentation


# Get dirs
root_dir = "/Users/lexuanthang/OneDrive/WORKING/Projects/CellDetection/Code/Murincells"
data_dir = os.path.join(root_dir, "data")
image_dir = os.path.join(data_dir, "images")
mask_dir = os.path.join(data_dir, "masks_grayscale")

id2color = {
    0: (0, 0, 0),  # background
    1: (28, 230, 255),  # Marcophage/Monocyte
    2: (255, 52, 255),  # Neutrophil
    3: (255, 74, 70),  # Eosinophil
    4: (0, 137, 65),  # Lymphocyte
    5: (0, 111, 166),  # Unknown cell/Debris
    6: (163, 0, 89),  # Basophil
}

# Transforms
trainsize = 256
train_transform = A.Compose(
    [
        A.Resize(width=trainsize, height=trainsize),
        A.HorizontalFlip(),
        # A.RandomBrightnessContrast(),
        A.Blur(),
        A.Sharpen(),
        # A.RGBShift(),
        A.CoarseDropout(),
        A.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0
        ),
        ToTensorV2(),
    ]
)

# # Define color to class index mapping
# label_colors = {
#     (0, 0, 0): 0,  # Background
#     (28, 230, 255): 1,  # Macrophage/Monocyte
#     (255, 52, 255): 2,  # Neutrophil
#     (255, 74, 70): 3,  # Eosinophil
#     (0, 137, 65): 4,  # Lymphocyte
#     (0, 111, 166): 5,  # Unknown cell/Debris
#     # (163, 0, 89): 6   # Basophil
# }


# 2. Dataset
class CellDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None) -> None:
        """
        Args:
            image_dir: string: path
            mask_dir: string: path
            transform: callable, optional
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.files = sorted(os.listdir(self.image_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        image_path = os.path.join(self.image_dir, self.files[idx])
        mask_path = os.path.join(self.mask_dir, self.files[idx])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        print(f"unique: {np.unique(mask)}")
        print("image shape: {}".format(image.shape))
        print("mask shape: {}".format(mask.shape))

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        return image, mask


X_train, X_temp, y_train, y_temp = train_test_split(
    image_dir, mask_dir, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_train, X_temp, test_size=0.5, random_state=42
)
print("train size:{}".format(X_train.shape))
print("val size:{}".format(X_val.shape))
print("test size:{}".format(X_test.shape))

train_dataset = CellDataset(
    image_dir=X_train, mask_dir=y_train, transform=train_transform
)
val_dataset = CellDataset(image_dir=X_val, mask_dir=y_val, transform=train_transform)
test_dataset = CellDataset(image_dir=X_test, mask_dir=y_test, transform=train_transform)

# rand_idx = random.randint(0,len(train_dataset))
# image, mask = train_dataset.__getitem__(rand_idx)
# image = image.permute(1, 2, 0).cpu()
# print(image.shape, mask.shape)
#
# print(type(image))
# print(type(mask))
# plot_segmentation(image, mask)

# DataLoader
NUM_CLASSES = len(id2color)
NUM_WORKERS = 4
BATCH_SIZE = 16
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

# Models
unet_model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=NUM_CLASSES,
)

# Loss and Optimizer
loss_fn = torch.nn.CrossEntropyLoss()

# Optional: IoU / Dice metrics from smp
# Train function

# Save models

# Get reports
