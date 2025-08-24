import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch


def plot_segmentation(*args, id2color=None):
    """
    Plot segmentation results.

    Args:
        *args:
            (image, gt_mask)
            (image, gt_mask, pred_mask)
        id2color: dict (optional), map class_id -> RGB color tuple
                  used to colorize grayscale masks
    """
    num_args = len(args)
    if num_args not in [2, 3]:
        raise ValueError("Function accepts either 2 or 3 arguments.")

    image = args[0]
    gt_mask = args[1]
    pred_mask = args[2] if num_args == 3 else None

    # --- Convert torch -> numpy if needed ---
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return x

    image = to_numpy(image)
    gt_mask = to_numpy(gt_mask)
    if pred_mask is not None:
        pred_mask = to_numpy(pred_mask)

    # --- Assertions ---
    assert image.ndim == 3 and image.shape[2] in [
        1,
        3,
    ], f"Expected image shape (H,W,3) or (H,W,1), got {image.shape}"
    if image.shape[2] == 1:  # squeeze grayscale image
        image = np.repeat(image, 3, axis=2)

    assert gt_mask.ndim == 2, f"Ground truth mask must be 2D, got {gt_mask.shape}"
    assert (
        gt_mask.shape[:2] == image.shape[:2]
    ), "Image and ground truth mask must have the same spatial size"
    if pred_mask is not None:
        assert pred_mask.ndim == 2, f"Predicted mask must be 2D, got {pred_mask.shape}"
        assert (
            pred_mask.shape == gt_mask.shape
        ), "GT mask and Pred mask must have the same shape"

    # --- Handle BGR images (cv2) ---
    if image.shape[2] == 3 and image.dtype == np.uint8:
        # Heuristic: assume it's BGR if from cv2
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def colorize(mask):
        """Convert grayscale mask to RGB using id2color map"""
        if id2color is None:
            return plt.cm.tab20(mask.astype(np.int32))
        h, w = mask.shape
        mask_color = np.zeros((h, w, 3), dtype=np.uint8)
        for cls_id, color in id2color.items():
            mask_color[mask == cls_id] = color
        return mask_color

    # --- Plot ---
    if num_args == 2:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(image)
        axes[0].set_title("Image")
        axes[1].imshow(colorize(gt_mask))
        axes[1].set_title("Ground Truth Mask")
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(image)
        axes[0].set_title("Image")
        axes[1].imshow(colorize(gt_mask))
        axes[1].set_title("Ground Truth Mask")
        axes[2].imshow(colorize(pred_mask))
        axes[2].set_title("Predicted Mask")

    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()
