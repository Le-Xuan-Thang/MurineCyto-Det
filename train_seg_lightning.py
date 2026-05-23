import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from torch.optim import lr_scheduler
# Torchmetrics
from torchmetrics.classification import Precision, Recall, F1Score, Accuracy
from torchmetrics.segmentation import DiceScore, MeanIoU, GeneralizedDiceScore
import gc
from pytorch_lightning.loggers import CSVLogger
from utils.helpers import visualize, predict_and_plot
from utils.data import make_splits, prepare_data
from dataloader.dataloaders import build_segmentation_dataloaders


# # Uncomment to check data
# # Visualize resulted augmented images and masks
# augmented_dataset = Dataset(data_dir=data_dir,
#     images_dir=x_train_dir,
#     masks_dir=y_train_dir,
#     augmentation=get_training_augmentation(),
# )

# # Visualizing the same image with different random transforms
# num_image = 3
# for i in range(num_image):
#     image, mask = augmented_dataset[3]
#     print(f"Mask shape: {mask.shape}")
#     print(np.unique(mask))
#     visualize(image=image, mask=mask)




# %% Define model

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


class MurineCellModel(pl.LightningModule):
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

        # Preprocessing parameters for image normalization
        params = get_encoder_preprocessing_params(encoder_name, encoder_weights)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # Loss function for multi-class segmentation
        self.loss_fn = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)

        self.train_metrics = self._init_metrics()
        self.valid_metrics = self._init_metrics()
        self.test_metrics = self._init_metrics()

        # Step metrics tracking
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
    def _init_metrics(self):
        metrics = torch.nn.ModuleDict({
            "gds": GeneralizedDiceScore(num_classes=self.num_classes, include_background=True, per_class=False, input_format="index"),
            "dice": DiceScore(num_classes=self.num_classes, include_background=True, average="micro", input_format="index"),
            "miou": MeanIoU(num_classes=self.num_classes, include_background=True, per_class=False, input_format="index"),
            "precision": Precision(num_classes=self.num_classes, average="micro", task="multiclass"),
            "recall": Recall(num_classes=self.num_classes, average="micro", task="multiclass"),
            "f1": F1Score(num_classes=self.num_classes, average="micro", task="multiclass"),
            "acc": Accuracy(num_classes=self.num_classes, average="micro", task="multiclass"),
        })
        return metrics
        
    def forward(self, image):
        # Normalize image (cast to float first so uint8 inputs work correctly)
        image = image.float()
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
            logits_mask.shape[1] == self.num_classes
        )  # [batch_size, num_classes, H, W]

        # Ensure the logits mask is contiguous
        logits_mask = logits_mask.contiguous()

        # Compute loss using multi-class Dice loss (pass original mask, not one-hot encoded)
        loss = self.loss_fn(logits_mask, mask)

        # Apply softmax to get probabilities for multi-class segmentation
        prob_mask = logits_mask.softmax(dim=1)

        # Convert probabilities to predicted class labels
        pred_mask = prob_mask.argmax(dim=1)
        
        metrics = getattr(self, f"{stage}_metrics")
        for metric in metrics.values():
            metric.update(pred_mask, mask)

        return {"loss": loss}

    def shared_epoch_end(self, outputs, stage):
        metrics = {f"{stage}_loss": torch.stack([x["loss"] for x in outputs]).mean()}
        tm_metrics = getattr(self, f"{stage}_metrics")
        for name, metric in tm_metrics.items():
            metrics[f"{stage}_{name}"] = metric.compute()
            metric.reset()
        self.log_dict(metrics, prog_bar=True, sync_dist=True)
        

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


def main():
    token = os.getenv("HF_TOKEN")
    encoder_weights = os.getenv("ENCODER_WEIGHTS") or None

    # %% Dirs
    root_dir = os.getcwd()
    model_dir = os.path.join(root_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    data_dir = prepare_data(data_dir=os.path.join(root_dir, "data"), token=token)
    splits = make_splits(data_dir)
    x_train_dir = splits["x_train"]
    y_train_dir = splits["y_train"]
    x_val_dir = splits["x_val"]
    y_val_dir = splits["y_val"]
    x_test_dir = splits["x_test"]
    y_test_dir = splits["y_test"]
    print("train size:{}".format(len(x_train_dir)))
    print("val size:{}".format(len(x_val_dir)))
    print("test size:{}".format(len(x_test_dir)))

    # Change to > 0 if not on Windows machine
    BATCH_SIZE = 8
    NUM_WORKERS = 0
    train_loader, valid_loader, test_loader = build_segmentation_dataloaders(
        data_dir=data_dir,
        splits=splits,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    gc.collect()
    torch.cuda.empty_cache()

    # Some training hyperparameters
    EPOCHS = 100
    logs_dir = os.path.join(root_dir, "logs")
    # Always include the background as a class
    NUM_CLASSES = len(train_loader.dataset.CLASSES)
    print(NUM_CLASSES)

    models = {
        "unet": "Unet",
        "segformer": "Segformer"
    }

    for name, arch in models.items():

        model = MurineCellModel(
            arch=arch,
            encoder_name="resnet34",
            encoder_weights=encoder_weights,
            in_channels=3,
            num_classes=NUM_CLASSES
        )

        csv_logger = CSVLogger(logs_dir, name=name)

        trainer = pl.Trainer(
            default_root_dir=logs_dir,
            max_epochs=EPOCHS,
            accelerator="auto",
            devices="auto",
            logger=csv_logger
        )

        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=valid_loader,
        )
        
        model_path = os.path.join(model_dir, f"{name}.ckpt")
        trainer.save_checkpoint(model_path)


if __name__ == "__main__":
    main()
