import os
import random
import numpy as np
import gc

import matplotlib.pyplot as plt
import cv2

from sklearn.model_selection import train_test_split

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
from torch.utils.data import Dataset as BaseDataset, DataLoader
from torch.optim import lr_scheduler

import pytorch_lightning as pl

import segmentation_models_pytorch as smp
from tqdm.auto import tqdm

# helper function
from utilities.helpers import plot
from utilities.plot_segmentation import plot_segmentation


# Get dirs
root_dir = "/Users/lexuanthang/OneDrive/WORKING/Projects/CellDetection/Code/Murincells"
data_dir = os.path.join(root_dir, "data")
image_dir = sorted(os.listdir(os.path.join(data_dir, "images")))
mask_dir = sorted(os.listdir(os.path.join(data_dir, "masks_grayscale")))

id2color = {
    0: (0, 0, 0),  # background
    1: (28, 230, 255),  # Marcophage/Monocyte
    2: (255, 52, 255),  # Neutrophil
    3: (255, 74, 70),  # Eosinophil
    4: (0, 137, 65),  # Lymphocyte
    5: (0, 111, 166),  # Unknown cell/Debris
}

x_train_dir, x_temp_dir, y_train_dir, y_temp_dir = train_test_split(
    image_dir, mask_dir, test_size=0.3, random_state=42
)
x_val_dir, x_test_dir, y_val_dir, y_test_dir = train_test_split(
    x_temp_dir, y_temp_dir, test_size=0.5, random_state=42
)
print("train size:{}".format(len(x_train_dir)))
print("val size:{}".format(len(x_val_dir)))
print("test size:{}".format(len(x_test_dir)))

# 2. Dataset


class Dataset(BaseDataset):
    CLASSES = [
        "Background",
        "Macrophage/Monocyte",
        "Neutrophil",
        "Eosinophil",
        "Lymphocyte",
        "Unknown cell/Debris",
    ]

    def __init__(
        self, data_dir, images_dir, masks_dir, classes=None, augmentation=None
    ):
        self.data_dir = data_dir
        self.images_fps = images_dir
        self.masks_fps = masks_dir

        # Always map background ('Background') to 0
        self.background_class = self.CLASSES.index("Background")

        # If specific classes are provided, map them dynamically
        if classes:
            self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        else:
            self.class_values = list(range(len(self.CLASSES)))  # Default to all classes

        # Create a remapping dictionary: class value in dataset -> new index (0, 1, 2, ...)
        # Background will always be 0, other classes will be remapped starting from 1.
        self.class_map = {self.background_class: 0}
        self.class_map.update(
            {
                v: i
                for i, v in enumerate(self.class_values)
                if v != self.background_class
            }
        )
        self.augmentation = augmentation

    def __getitem__(self, i):
        # Read the image
        image_path = os.path.join(self.data_dir, "images", self.images_fps[i])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        # Read the mask in grayscale mode
        mask_path = os.path.join(self.data_dir, "masks_grayscale", self.masks_fps[i])
        mask = cv2.imread(mask_path, 0)

        # Create a blank mask to remap the class values
        mask_remap = np.zeros_like(mask)

        # Remap the mask according to the dynamically created class map
        for class_value, new_value in self.class_map.items():
            mask_remap[mask == class_value] = new_value

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask_remap)
            image, mask_remap = sample["image"], sample["mask"]

        image = image.transpose(2, 0, 1)
        return image, mask_remap

    def __len__(self):
        return len(self.images_fps)


def get_training_augmentation():
    train_transform = [
        # Áp dụng cho cả image + mask
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0
        ),
        A.PadIfNeeded(min_height=256, min_width=256, p=1),
        A.RandomCrop(height=256, width=256, p=1),
        # Chỉ áp dụng cho ảnh
        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),
        A.OneOf(
            [
                A.Sharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),
        A.OneOf(
            [
                A.RandomBrightnessContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return A.Compose(train_transform)


def get_validation_augmentation():
    valid_transform = [A.PadIfNeeded(min_height=256, min_width=256)]
    return A.Compose(valid_transform)


# DataLoader
train_dataset = Dataset(
    data_dir=data_dir,
    images_dir=x_train_dir,
    masks_dir=y_train_dir,
    augmentation=get_training_augmentation(),
)

valid_dataset = Dataset(
    data_dir=data_dir,
    images_dir=x_val_dir,
    masks_dir=y_val_dir,
    augmentation=get_validation_augmentation(),
)

test_dataset = Dataset(
    data_dir=data_dir,
    images_dir=x_test_dir,
    masks_dir=y_test_dir,
    augmentation=get_validation_augmentation(),
)

# Change to > 0 if not on Windows machine
NUM_WORKERS = 0
train_loader = DataLoader(
    train_dataset, batch_size=8, shuffle=True, num_workers=NUM_WORKERS
)
valid_loader = DataLoader(
    valid_dataset, batch_size=8, shuffle=False, num_workers=NUM_WORKERS
)
test_loader = DataLoader(
    test_dataset, batch_size=8, shuffle=False, num_workers=NUM_WORKERS
)

# Some training hyperparameters
EPOCHS = 50
T_MAX = EPOCHS * len(train_loader)
# Always include the background as a class
OUT_CLASSES = len(train_dataset.CLASSES)


class Murincell_model(pl.LightningModule):
    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes,
            **kwargs,
        )

        # Preprocessing parameters for image normalization
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.number_of_classes = out_classes
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # Loss function for multi-class segmentation
        self.loss_fn = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)

        # Step metrics tracking
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, image):
        # Normalize image
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image, mask = batch

        # Ensure that image dimensions are correct
        assert image.ndim == 4  # [batch_size, channels, H, W]

        # Ensure the mask is a long (index) tensor
        mask = mask.long()

        # Mask shape
        assert mask.ndim == 3  # [batch_size, H, W]

        # Predict mask logits
        logits_mask = self.forward(image)

        assert (
            logits_mask.shape[1] == self.number_of_classes
        )  # [batch_size, number_of_classes, H, W]

        # Ensure the logits mask is contiguous
        logits_mask = logits_mask.contiguous()

        # Compute loss using multi-class Dice loss (pass original mask, not one-hot encoded)
        loss = self.loss_fn(logits_mask, mask)

        # Apply softmax to get probabilities for multi-class segmentation
        prob_mask = logits_mask.softmax(dim=1)

        # Convert probabilities to predicted class labels
        pred_mask = prob_mask.argmax(dim=1)

        # Compute true positives, false positives, false negatives, and true negatives
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask, mask, mode="multiclass", num_classes=self.number_of_classes
        )

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # Aggregate step metrics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # Per-image IoU and dataset IoU calculations
        per_image_iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction="micro-imagewise"
        )
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train")
        self.training_step_outputs.append(train_loss_info)
        return train_loss_info

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        valid_loss_info = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(valid_loss_info)
        return valid_loss_info

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        test_loss_info = self.shared_step(batch, "test")
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


# Loss and Optimizer
loss_fn = torch.nn.CrossEntropyLoss()

model = Murincell_model(
    "FPN", "resnext50_32x4d", in_channels=3, out_classes=OUT_CLASSES
)


trainer = pl.Trainer(
    max_epochs=EPOCHS,
    log_every_n_steps=1,
    accelerator="mps",  # or "cpu" / "mps" (for Apple Silicon)
    devices=1,  # number of GPUs (or "auto")
)

trainer.fit(
    model,
    train_dataloaders=train_loader,
    val_dataloaders=valid_loader,
)

# run validation dataset
valid_metrics = trainer.validate(model, dataloaders=valid_loader, verbose=False)
print(valid_metrics)
# run test dataset
test_metrics = trainer.test(model, dataloaders=test_loader, verbose=False)
print(test_metrics)


def predict_and_plot(model, dataloader, device=None, n_samples=5):
    """
    Run inference on a batch from dataloader and visualize predictions.

    Args:
        model: trained PyTorch/Lightning model
        dataloader: DataLoader (val/test)
        device: torch.device (cpu/cuda)
        n_samples: number of samples to plot
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    # Get one batch
    images, masks = next(iter(dataloader))
    images, masks = images.to(device), masks.to(device)

    with torch.inference_mode():
        logits = model(images)
        pr_masks = logits.softmax(dim=1).argmax(dim=1)

    # Plot
    for idx, (image, gt_mask, pr_mask) in enumerate(zip(images, masks, pr_masks)):
        if idx >= n_samples:
            break

        plt.figure(figsize=(12, 6))

        # Original Image
        plt.subplot(1, 3, 1)
        plt.imshow(image.cpu().numpy().transpose(1, 2, 0))
        plt.title("Image")
        plt.axis("off")

        # Ground Truth Mask
        plt.subplot(1, 3, 2)
        plt.imshow(gt_mask.cpu().numpy(), cmap="tab20")
        plt.title("Ground truth")
        plt.axis("off")

        # Predicted Mask
        plt.subplot(1, 3, 3)
        plt.imshow(pr_mask.cpu().numpy(), cmap="tab20")
        plt.title("Prediction")
        plt.axis("off")

        plt.show()


# Get reports

predict_and_plot(model, valid_loader, n_samples=5)
