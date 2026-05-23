import torch
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix

def visualize(image=None, mask=None):
    fig, ax = plt.subplots(1, 2, figsize=(10,6))

    # tensor -> numpy
    if hasattr(image, "cpu"):
        image = image.cpu().numpy()

    # handle shape
    if image.ndim == 3:
        # nếu là (C,H,W) → convert
        if image.shape[0] in [1,3]:
            image = image.transpose(1,2,0)

    ax[0].imshow(image)
    ax[0].set_title("Image")

    ax[1].imshow(mask, cmap="tab20")
    ax[1].set_title("Mask")

    for a in ax:
        a.axis("off")

    plt.show()

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


def predict_and_plot_multi(models: dict, dataloader, device=None, n_samples=5, save_dir=None, prefix=""):
    """
    Run inference with multiple models on a batch and visualize predictions for easier comparison.

    Args:
        models (dict): {model_name: trained model}
        dataloader: DataLoader (val/test)
        device: torch.device (cpu/cuda)
        n_samples (int): number of samples to plot
        save_dir (str, optional): directory to save figures. If None, figures are shown instead.
        prefix (str, optional): prefix for saved filenames (e.g., 'val_', 'test_')
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move models to device & set to eval mode
    for m in models.values():
        m.to(device)
        m.eval()

    # Get one batch
    images, masks = next(iter(dataloader))
    images, masks = images.to(device), masks.to(device)

    # Predict with each model
    preds = {}
    with torch.inference_mode():
        for name, model in models.items():
            logits = model(images)
            preds[name] = logits.softmax(dim=1).argmax(dim=1)

    # Create save directory if needed
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Plot results
    for idx in range(min(n_samples, len(images))):
        n_cols = 2 + len(models)  # Image + GT + predictions
        plt.figure(figsize=(4 * n_cols, 4))

        # Original image
        plt.subplot(1, n_cols, 1)
        plt.imshow(images[idx].cpu().numpy().transpose(1, 2, 0))
        plt.title("Image")
        plt.axis("off")

        # Ground truth
        plt.subplot(1, n_cols, 2)
        plt.imshow(masks[idx].cpu().numpy(), cmap="tab20")
        plt.title("Ground Truth")
        plt.axis("off")

        # Predictions from each model
        for col, (name, pr_masks) in enumerate(preds.items(), start=3):
            plt.subplot(1, n_cols, col)
            plt.imshow(pr_masks[idx].cpu().numpy(), cmap="tab20")
            plt.title(f"{name} Prediction")
            plt.axis("off")

        plt.tight_layout()

        if save_dir:
            file_name = f"{prefix}sample_{idx+1}.pdf"
            file_path = os.path.join(save_dir, file_name)
            plt.savefig(file_path, bbox_inches="tight", dpi=300)
            plt.close()
            print(f"✅ Saved: {file_path}")
        else:
            plt.show()

def visualize_batch(images, masks, class_names, n=4):
    """
    Show n samples from a batch with their image and mask.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    images = images.cpu().numpy()
    masks = masks.cpu().numpy()
    
    batch_size = images.shape[0]
    n = min(n, batch_size)

    fig, axes = plt.subplots(n, 2, figsize=(6, 3*n))
    if n == 1:
        axes = [axes]

    for i in range(n):
        img = np.transpose(images[i], (1,2,0))

        # --- Fix: scale if float data is not in [0,1] ---
        if img.dtype in (np.float32, np.float64):
            img = img / 255.0 if img.max() > 1.0 else img

        mask = masks[i]

        axes[i][0].imshow(img)
        axes[i][0].set_title("Image")
        axes[i][0].axis("off")

        im = axes[i][1].imshow(mask, cmap="tab20")
        axes[i][1].set_title("Mask")
        axes[i][1].axis("off")

    plt.tight_layout()
    plt.show()

def plot_final_metrics_bar(dfs: dict, metrics: list, prefix=None, prefix_col="valid_", save_dir=None):
    """
    Plot grouped bar chart comparing final values of metrics across models.

    Args:
        dfs (dict): {model_name: df_val}
        metrics (list): list of short metric names (e.g. ["acc", "dice", "miou", "gds", "loss"])
        prefix (str): add prefix to file name
        prefix_col (str): either "train_" or "valid_", used to pick the right columns
        save_dir (str, optional): directory to save the figure. If None, show interactively.
    """
    results = {}

    for name, df in dfs.items():
        last_row = df.dropna().iloc[-1]
        values = []
        for m in metrics:
            col_candidates = [c for c in df.columns if c.startswith(prefix_col) and m in c]
            if not col_candidates:
                raise KeyError(f"Column with prefix '{prefix_col}' and metric '{m}' not found in DataFrame {name}")
            col = col_candidates[0]
            values.append(last_row[col])
        results[name] = values

    # Convert to dataframe for seaborn
    results_df = pd.DataFrame(results, index=metrics)
    results_df_long_form = results_df.reset_index().melt(
        id_vars='index', var_name='Model', value_name='Score'
    )

    # Plot grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    ax = sns.barplot(
        data=results_df_long_form,
        x='index',
        y='Score',
        hue='Model',
        errorbar=None,
        palette="viridis",
        edgecolor="darkgray",
        legend=True,
    )

    plt.xticks(rotation=30)
    plt.xlabel("")
    ax.tick_params(axis="x", which="both", length=10, bottom=False, top=False)
    plt.grid(True, which="major", axis="y", linestyle="-", linewidth=0.25, alpha=0.35)
    plt.minorticks_on()
    plt.grid(True, which="minor", axis="y", linestyle="-", linewidth=0.15, alpha=0.25)
    ax.set_axisbelow(True)

    for container in ax.containers:
        ax.bar_label(container=container, fmt="%.3f", label_type="edge", fontsize=16)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
        fancybox=False,
        shadow=False,
        frameon=False,
        ncol=2
    )

    plt.tight_layout()

    # --- Save or show ---
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        # remove trailing underscore in prefix like "valid_"
        file_name = f"{prefix}_final_metrics_bar.pdf"
        file_path = os.path.join(save_dir, file_name)
        plt.savefig(file_path, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"✅ Saved grouped bar chart to: {file_path}")
    else:
        plt.show()

    return results_df

    
def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names,
    ignore_index=0,
    normalize=True,
    cmap="Blues",
    save_dir=None,
    prefix=""
):
    """
    Plot confusion matrix for segmentation masks (per-pixel) and optionally save to file.

    Args:
        y_true: (H, W) ground truth mask (integers: class indices)
        y_pred: (H, W) predicted mask (integers: class indices)
        class_names (list): list of class names
        ignore_index (int): index of background class to ignore
        normalize (bool): normalize rows to percentages
        cmap (str): color map for heatmap
        save_dir (str, optional): directory to save figure. If None, figure is shown instead.
        prefix (str, optional): prefix for saved filename, e.g. 'val_', 'test_'
    """

    # Flatten into 1D pixel vectors
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # Mask out background pixels
    mask = y_true != ignore_index
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    # Select class labels (excluding ignored index)
    labels = [i for i in range(len(class_names)) if i != ignore_index]
    display_names = [class_names[i] for i in labels]

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if normalize:
        with np.errstate(all='ignore'):  # ignore divide by zero warnings
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)  # replace NaN (rows with 0 samples) with 0

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap=cmap,
        xticklabels=display_names,
        yticklabels=display_names,
        annot_kws={"size": 14},
        cbar=False,
    )

    plt.xlabel("Predicted", fontsize=14)
    plt.ylabel("Ground Truth", fontsize=14)
    # plt.title("Confusion Matrix (per-pixel, no Background)", fontsize=14)
    plt.xticks(fontsize=12, rotation=30)
    plt.yticks(fontsize=12, rotation=0)

    plt.tight_layout()

    # Save or show
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        file_name = f"{prefix}confusion_matrix.pdf"
        file_path = os.path.join(save_dir, file_name)
        plt.savefig(file_path, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"✅ Saved confusion matrix to: {file_path}")
    else:
        plt.show()