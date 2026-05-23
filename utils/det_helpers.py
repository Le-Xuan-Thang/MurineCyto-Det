# This helpers function was clone from https://github.com/pytorch/vision/tree/main/gallery/
import os
import matplotlib.pyplot as plt
import torch
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F
import cv2


# ============================== Visualization ===================================
def plot(imgs, row_title=None, bbox_width=3, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            boxes = None
            masks = None
            if isinstance(img, tuple):
                img, target = img
                if isinstance(target, dict):
                    boxes = target.get("boxes")
                    masks = target.get("masks")
                elif isinstance(target, tv_tensors.BoundingBoxes):
                    boxes = target

                    # Conversion necessary because draw_bounding_boxes() only
                    # work with this specific format.
                    if tv_tensors.is_rotated_bounding_format(boxes.format):
                        boxes = v2.ConvertBoundingBoxFormat("xyxyxyxy")(boxes)
                else:
                    raise ValueError(f"Unexpected target type: {type(target)}")
            img = F.to_image(img)
            if img.dtype.is_floating_point and img.min() < 0:
                # Poor man's re-normalization for the colors to be OK-ish. This
                # is useful for images coming out of Normalize()
                img -= img.min()
                img /= img.max()

            img = F.to_dtype(img, torch.uint8, scale=True)
            if boxes is not None:
                img = draw_bounding_boxes(img, boxes, colors="yellow", width=bbox_width)
            if masks is not None:
                img = draw_segmentation_masks(
                    img,
                    masks.to(torch.bool),
                    colors=["green"] * masks.shape[0],
                    alpha=0.65,
                )

            ax = axs[row_idx, col_idx]
            ax.imshow(img.permute(1, 2, 0).numpy(), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()


# This function was copied in: https://albumentations.ai/docs/examples/example-bboxes2/
# The visualization function is based on https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/vis.py
def visualize_bbox(img, bbox, class_name, color=None, thickness=2):
    """Visualizes a single bounding box on the image"""
    if color == None:
        BOX_COLOR = (255, 0, 0)  # Red

    TEXT_COLOR = (255, 255, 255)  # White
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(
        class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1
    )
    cv2.rectangle(
        img,
        (x_min, y_min - int(1.3 * text_height)),
        (x_min + text_width, y_min),
        BOX_COLOR,
        -1,
    )
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )
    return img


# This function was modified base on the code in: https://albumentations.ai/docs/examples/example-bboxes2/
# The visualization function is based on https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/vis.py
def visualize(image, bboxes, category_ids, category_id_to_name, ax=None):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    
    if ax is None:
        # Nếu không truyền ax thì tự tạo figure mới
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.axis("off")
        ax.imshow(img)
        plt.show()
    else:
        # Nếu đã có ax thì vẽ vào đó
        ax.axis("off")
        ax.imshow(img)



# ============================== Checking ===================================


## Check files name
def check_files_names(image_dir, annotation_dir):
    """
    Compare the file names (without extensions) in two folders.

    Args:
        image_dir (str): Path to the folder containing images.
        annotation_dir (str): Path to the folder containing annotations.

    Returns:
        dict: A dictionary with the following keys:
            - "same" (bool): True if the two folders contain the same base names.
            - "missing_in_annotations" (set): File base names present in images but missing in annotations.
            - "missing_in_images" (set): File base names present in annotations but missing in images.

    Example:
        >>> result = check_files_names("images/", "annotations/")
        >>> result["same"]
        True
    """
    image_ls = sorted(os.listdir(image_dir))
    annotation_ls = sorted(os.listdir(annotation_dir))

    files_1 = {os.path.splitext(item)[0] for item in image_ls}
    files_2 = {os.path.splitext(item)[0] for item in annotation_ls}

    diff1 = files_1 - files_2
    diff2 = files_2 - files_1

    return {
        "same": len(diff1) == 0 and len(diff2) == 0,
        "missing_in_annotations": diff1,
        "missing_in_images": diff2,
    }


def report_files_names(image_dir, annotation_dir):
    """
    Wrapper around `check_files_names` that prints a human-readable report.
    """
    result = check_files_names(image_dir, annotation_dir)

    if result["same"]:
        print("✅ Files names in both folders are [THE SAME]")
    else:
        print("❌ Files names in both folders are [NOT THE SAME]")
        if result["missing_in_annotations"]:
            print("  Missing in annotations:", result["missing_in_annotations"])
        if result["missing_in_images"]:
            print("  Missing in images:", result["missing_in_images"])

    return result
