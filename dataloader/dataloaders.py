import os

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
import cv2
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset


class MurineCellSegmentationDataset(BaseDataset):
    CLASSES = [
        "Background",
        "Macrophage/Monocyte",
        "Neutrophil",
        "Eosinophil",
        "Lymphocyte",
        "Unknown cell/Debris",
    ]

    def __init__(self, data_dir, images_dir, masks_dir, classes=None, augmentation=None):
        self.data_dir = data_dir
        self.images_fps = images_dir
        self.masks_fps = masks_dir
        self.background_class = self.CLASSES.index("Background")

        if classes:
            class_to_idx = {name.lower(): idx for idx, name in enumerate(self.CLASSES)}
            self.class_values = [class_to_idx[cls.lower()] for cls in classes]
        else:
            self.class_values = list(range(len(self.CLASSES)))

        self.class_map = {self.background_class: 0}
        self.class_map.update(
            {
                value: idx
                for idx, value in enumerate(self.class_values)
                if value != self.background_class
            }
        )
        self.augmentation = augmentation

    def __getitem__(self, i):
        image_path = os.path.join(self.data_dir, "images", self.images_fps[i])
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_path = os.path.join(self.data_dir, "masks_grayscale", self.masks_fps[i])
        mask = cv2.imread(mask_path, 0)
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask: {mask_path}")

        mask_remap = np.zeros_like(mask)
        for class_value, new_value in self.class_map.items():
            mask_remap[mask == class_value] = new_value

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask_remap)
            image, mask_remap = sample["image"], sample["mask"]

        image = image.transpose(2, 0, 1)
        return image, mask_remap

    def __len__(self):
        return len(self.images_fps)


Dataset = MurineCellSegmentationDataset


def get_training_augmentation():
    train_transform = [
        A.HorizontalFlip(p=0.5),
        A.Affine(
            scale=[0.5, 1],
            translate_percent=[-0.05, 0.05],
            rotate=[-45, 45],
            shear=[-15, 15],
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            fit_output=False,
            keep_ratio=False,
            rotate_method="ellipse",
            balanced_scale=True,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=0,
        ),
        A.PadIfNeeded(min_height=256, min_width=256, p=1),
        A.RandomCrop(height=256, width=256, p=1),
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
    return A.Compose([A.PadIfNeeded(min_height=256, min_width=256)])


def build_segmentation_datasets(data_dir, splits):
    train_dataset = Dataset(
        data_dir=data_dir,
        images_dir=splits["x_train"],
        masks_dir=splits["y_train"],
        augmentation=get_training_augmentation(),
    )
    valid_dataset = Dataset(
        data_dir=data_dir,
        images_dir=splits["x_val"],
        masks_dir=splits["y_val"],
        augmentation=get_validation_augmentation(),
    )
    test_dataset = Dataset(
        data_dir=data_dir,
        images_dir=splits["x_test"],
        masks_dir=splits["y_test"],
        augmentation=get_validation_augmentation(),
    )
    return train_dataset, valid_dataset, test_dataset


def build_segmentation_dataloaders(
    data_dir,
    splits,
    batch_size=8,
    num_workers=0,
):
    train_dataset, valid_dataset, test_dataset = build_segmentation_datasets(
        data_dir=data_dir,
        splits=splits,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, valid_loader, test_loader
