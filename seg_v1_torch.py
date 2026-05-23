import argparse
import csv
import gc
import os
import tempfile
from datetime import datetime

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib.pyplot as plt
import cv2
import segmentation_models_pytorch as smp
import torch
from torch.optim import lr_scheduler
from torchmetrics.classification import Accuracy, F1Score, Precision, Recall
from torchmetrics.segmentation import DiceScore, GeneralizedDiceScore, MeanIoU
from tqdm import tqdm

from dataloader.dataloaders import build_segmentation_dataloaders
from utils.data import make_splits, prepare_data
from utils.helpers import visualize


MODEL_ARCHES = {
    "unet": "Unet",
    "segformer": "Segformer",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train MurineCyto segmentation models.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=int(os.getenv("EPOCHS", "10")),
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("BATCH_SIZE", "8")),
        help="Training batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=int(os.getenv("NUM_WORKERS", "0")),
        help="PyTorch dataloader worker count.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_ARCHES),
        default=sorted(MODEL_ARCHES),
        help="Model names to train.",
    )
    parser.add_argument(
        "--run-name",
        default=os.getenv("RUN_NAME"),
        help="Run label for logs and checkpoints. Defaults to a timestamp.",
    )
    return parser.parse_args()


def print_runtime_setup():
    print(f"PyTorch version: {torch.__version__}")
    print(f"Segmentation Models PyTorch version: {smp.__version__}")
    if torch.cuda.is_available():
        print("CUDA is available.")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("CUDA is not available.")
        print("Use device: CPU")
    print(f"openCV version: {cv2.__version__}")


def get_encoder_preprocessing_params(encoder_name, encoder_weights):
    if encoder_weights is None:
        return {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        }
    return smp.encoders.get_preprocessing_params(
        encoder_name,
        pretrained=encoder_weights,
    )


class MurineCellModel(torch.nn.Module):
    def __init__(self, arch, encoder_name, in_channels, num_classes, **kwargs):
        super().__init__()
        encoder_weights = kwargs.get("encoder_weights")
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=num_classes,
            **kwargs,
        )
        self.num_classes = num_classes

        params = get_encoder_preprocessing_params(encoder_name, encoder_weights)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

    def forward(self, image):
        image = image.float()
        image = (image - self.mean) / self.std
        return self.model(image)


def init_metrics(num_classes, device):
    metrics = torch.nn.ModuleDict(
        {
            "gds": GeneralizedDiceScore(
                num_classes=num_classes,
                include_background=True,
                per_class=False,
                input_format="index",
            ),
            "dice": DiceScore(
                num_classes=num_classes,
                include_background=True,
                average="micro",
                input_format="index",
            ),
            "miou": MeanIoU(
                num_classes=num_classes,
                include_background=True,
                per_class=False,
                input_format="index",
            ),
            "precision": Precision(
                num_classes=num_classes,
                average="micro",
                task="multiclass",
            ),
            "recall": Recall(
                num_classes=num_classes,
                average="micro",
                task="multiclass",
            ),
            "f1": F1Score(
                num_classes=num_classes,
                average="micro",
                task="multiclass",
            ),
            "acc": Accuracy(
                num_classes=num_classes,
                average="micro",
                task="multiclass",
            ),
        }
    )
    return metrics.to(device)


def run_epoch(model, loader, loss_fn, metrics, device, optimizer=None, scheduler=None, desc=""):
    is_train = optimizer is not None
    model.train(is_train)
    for metric in metrics.values():
        metric.reset()

    total_loss = 0.0
    total_items = 0
    context = torch.enable_grad() if is_train else torch.inference_mode()

    with context:
        progress = tqdm(loader, desc=desc, leave=False)
        for image, mask in progress:
            image = image.to(device)
            mask = mask.to(device).long()

            logits = model(image).contiguous()
            loss = loss_fn(logits, mask)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            pred_mask = logits.softmax(dim=1).argmax(dim=1)
            for metric in metrics.values():
                metric.update(pred_mask, mask)

            batch_size = image.size(0)
            total_loss += loss.detach().item() * batch_size
            total_items += batch_size
            progress.set_postfix(loss=total_loss / max(total_items, 1))

    results = {"loss": total_loss / max(total_items, 1)}
    for name, metric in metrics.items():
        value = metric.compute()
        results[name] = value.item() if torch.is_tensor(value) else value
    for metric in metrics.values():
        metric.reset()
    return results


def prefixed_metrics(prefix, metrics):
    return {f"{prefix}_{name}": value for name, value in metrics.items()}


def append_metrics_row(csv_path, row, fieldnames):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    print_runtime_setup()

    token = os.getenv("HF_TOKEN")
    encoder_weights = os.getenv("ENCODER_WEIGHTS") or None
    run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    root_dir = os.getcwd()
    model_dir = os.path.join(root_dir, "models")
    logs_dir = os.path.join(root_dir, "logs")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    data_dir = prepare_data(data_dir=os.path.join(root_dir, "data"), token=token)
    splits = make_splits(data_dir)
    x_train_dir = splits["x_train"]
    y_train_dir = splits["y_train"]
    x_val_dir = splits["x_val"]
    y_val_dir = splits["y_val"]
    x_test_dir = splits["x_test"]
    y_test_dir = splits["y_test"]
    print(f"train size:{len(x_train_dir)}")
    print(f"val size:{len(x_val_dir)}")
    print(f"test size:{len(x_test_dir)}")
    
    # # plot an image and mask pair
    # if x_train_dir:
    #     img_path = os.path.join(data_dir, "images", x_train_dir[0])
    #     mask_path = os.path.join(data_dir, "masks", y_train_dir[0])
    #     img = cv2.imread(img_path)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     mask = cv2.imread(mask_path)
    #     mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    #     print(f"Image shape: {img.shape}, dtype: {img.dtype}")
    #     print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")
    #     visualize(img, mask)


    train_loader, valid_loader, test_loader = build_segmentation_dataloaders(
        data_dir=data_dir,
        splits=splits,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Clear GPU cache before training
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    num_classes = len(train_loader.dataset.CLASSES)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"num classes:{num_classes}")
    print(f"device:{device}")
    print(f"epochs:{args.epochs}")
    print(f"models:{', '.join(args.models)}")
    print(f"run name:{run_name}")

    metric_names = ["loss", "gds", "dice", "miou", "precision", "recall", "f1", "acc"]
    fieldnames = (
        ["epoch", "step"]
        + [f"train_{name}" for name in metric_names]
        + [f"valid_{name}" for name in metric_names]
        + [f"test_{name}" for name in metric_names]
    )

    for name in args.models:
        arch = MODEL_ARCHES[name]
        model = MurineCellModel(
            arch=arch,
            encoder_name="resnet34",
            encoder_weights=encoder_weights,
            in_channels=3,
            num_classes=num_classes,
        ).to(device)
        loss_fn = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
        train_metrics = init_metrics(num_classes, device)
        valid_metrics = init_metrics(num_classes, device)
        test_metrics = init_metrics(num_classes, device)

        model_log_dir = os.path.join(logs_dir, f"{name}_torch", run_name)
        os.makedirs(model_log_dir, exist_ok=True)
        csv_path = os.path.join(model_log_dir, "metrics.csv")

        best_valid_loss = float("inf")
        global_step = 0

        for epoch in range(args.epochs):
            train_results = run_epoch(
                model,
                train_loader,
                loss_fn,
                train_metrics,
                device,
                optimizer=optimizer,
                scheduler=scheduler,
                desc=f"{name} train {epoch + 1}/{args.epochs}",
            )
            global_step += len(train_loader)
            valid_results = run_epoch(
                model,
                valid_loader,
                loss_fn,
                valid_metrics,
                device,
                desc=f"{name} valid {epoch + 1}/{args.epochs}",
            )

            row = {
                "epoch": epoch,
                "step": global_step,
                **prefixed_metrics("train", train_results),
                **prefixed_metrics("valid", valid_results),
            }
            append_metrics_row(csv_path, row, fieldnames)

            print(
                f"{name} epoch {epoch + 1}/{args.epochs} "
                f"train_loss={train_results['loss']:.4f} "
                f"valid_loss={valid_results['loss']:.4f} "
                f"valid_miou={valid_results['miou']:.4f}"
            )

            if valid_results["loss"] < best_valid_loss:
                best_valid_loss = valid_results["loss"]
                best_path = os.path.join(model_dir, f"{name}_torch_{run_name}_best.pth")
                torch.save(
                    {
                        "epoch": epoch,
                        "arch": arch,
                        "encoder_name": "resnet34",
                        "num_classes": num_classes,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "valid_loss": best_valid_loss,
                    },
                    best_path,
                )

        test_results = run_epoch(
            model,
            test_loader,
            loss_fn,
            test_metrics,
            device,
            desc=f"{name} test",
        )
        append_metrics_row(
            csv_path,
            {
            "epoch": args.epochs,
                "step": global_step,
                **prefixed_metrics("test", test_results),
            },
            fieldnames,
        )

        final_path = os.path.join(model_dir, f"{name}_torch_{run_name}_final.pth")
        torch.save(
            {
                "epoch": args.epochs - 1,
                "arch": arch,
                "encoder_name": "resnet34",
                "num_classes": num_classes,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "test_metrics": test_results,
            },
            final_path,
        )
        print(f"saved final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
