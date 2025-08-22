import os
import matplotlib.pyplot as plt

def validate_image_mask_pairs(images_dir, masks_dir, image_prefix="P01_", mask_prefix="masks_P01_"):
    """
    Check if all images in the images directory have corresponding mask files in the masks directory.

    Args:
        images_dir (str): Path to the directory containing image files.
        masks_dir (str): Path to the directory containing mask files.
        image_prefix (str, optional): Prefix of the image files. Defaults to "P01_".
        mask_prefix (str, optional): Prefix of the mask files. Defaults to "masks_P01_".

    Raises:
        FileNotFoundError: If the images directory or masks directory does not exist.

    Returns:
        list: A list of missing mask file names.
    """
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory '{images_dir}' does not exist.")
    if not os.path.exists(masks_dir):
        raise FileNotFoundError(f"Masks directory '{masks_dir}' does not exist.")
    
    images = sorted(os.listdir(images_dir))
    masks = sorted(os.listdir(masks_dir))

    missing_masks = []

    for img_name in images:
        # Assuming the mask file has the same name as the image file but with a different prefix
        mask_name = img_name.replace(image_prefix, mask_prefix)
        if mask_name not in masks:
            missing_masks.append(mask_name)
            print(f"Missing mask for image: {img_name}")

    if not missing_masks:
        print("All image files have corresponding mask files.")
    else:
        print(f"Total missing masks: {len(missing_masks)}")
    
    return missing_masks


def check_dataloader(dataloader, num_samples_to_display=4):
    # Get a single batch from the dataloader
    sample_batch = next(iter(dataloader))

    # Extract the image and mask
    images, masks = sample_batch

    # Print the shapes of the images and masks in the batch
    print(f"Image batch shape: {images.shape}")
    print(f"Mask batch shape: {masks.shape}")

    # Plot the first few images and masks
    plt.figure(figsize=(12, 6))
    for i in range(num_samples_to_display):
        image = images[i].permute(1, 2, 0).numpy()  # Convert from (C, H, W) to (H, W, C)
        mask = masks[i].numpy()

        plt.subplot(2, num_samples_to_display, i + 1)
        plt.imshow(image)
        plt.title("Image")
        plt.axis('off')

        plt.subplot(2, num_samples_to_display, i + 1 + num_samples_to_display)
        plt.imshow(mask, cmap='gray')
        plt.title("Mask")
        plt.axis('off')

    plt.show()
    return images, masks