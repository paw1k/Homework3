import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

from homework import load_model
from homework.datasets.road_dataset import load_data  # Ensure these exist
from homework.models import save_model
from homework.metrics import DetectionMetric  # Import your detection metrics

def train_detection(
        exp_dir: str = "logs",
        model_name: str = "detection_model",  # Specify default model name for detection
        num_epoch: int =20,
        lr: float = 1e-2,  # A typical learning rate for detection tasks
        batch_size: int = 64,  # Smaller batch size for detection
        seed: int = 2024,
        **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create a directory for saving logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    # Load training and validation data
    train_data = load_data("road_data/train", shuffle=True, batch_size=batch_size, num_workers=2)
    val_data = load_data("road_data/val", shuffle=False)

    # Define the class weights for segmentation loss
    class_weights = torch.tensor([1.0, 3.0, 3.0], device=device)  # Adjust weights as needed
    segmentation_loss = torch.nn.CrossEntropyLoss(weight=class_weights)  # For segmentation task
    regression_loss = torch.nn.L1Loss()  # Absolute error loss for depth prediction
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Initialize the learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # Initialize the metric object for tracking detection performance
    metrics = DetectionMetric(num_classes=3)  # Specify the number of classes as needed

    # Training loop
    for epoch in range(num_epoch):
        metrics.reset()  # Reset metrics for the new epoch

        model.train()  # Set the model to training mode
        start_time = time.time()  # Start time for the epoch

        for batch in train_data:
            img, label, depth_label = batch['image'], batch['track'], batch['depth']
            img, label, depth_label = img.to(device), label.to(device), depth_label.to(device)

            # Forward pass
            preds, depth_preds = model(img)  # Assuming the model returns both preds and depth preds

            # Calculate the loss
            seg_loss = segmentation_loss(preds, label)  # Cross-entropy loss for segmentation
            reg_loss = regression_loss(depth_preds, depth_label)  # Regression loss for depth
            loss_val = seg_loss + reg_loss  # Combine both losses

            optimizer.zero_grad()  # Zero gradients
            loss_val.backward()  # Backward pass
            optimizer.step()  # Update weights

            # Update metrics
            metrics.add(preds.argmax(dim=1), label, depth_preds, depth_label)  # Assuming preds are logits

        # Disable gradient computation for validation
        with torch.inference_mode():
            model.eval()  # Set model to evaluation mode

            for batch in val_data:
                img, label, depth_label = batch['image'], batch['track'], batch['depth']
                img, label, depth_label = img.to(device), label.to(device), depth_label.to(device)

                preds, depth_preds = model(img)
                metrics.add(preds.argmax(dim=1), label, depth_preds, depth_label)

        # Compute metrics
        metrics_dict = metrics.compute()

        # Log metrics
        logger.add_scalar("train/loss", loss_val.item(), epoch)
        logger.add_scalar("train/iou", metrics_dict["iou"], epoch)
        logger.add_scalar("train/accuracy", metrics_dict["accuracy"], epoch)

        # Calculate the time taken for the epoch
        elapsed_time = time.time() - start_time  # Get the elapsed time
        print(
            f"Epoch {epoch + 1:2d}/{num_epoch:2d}: "
            f"loss={loss_val:.4f}, "
            f"iou={metrics_dict['iou']:.4f}, "
            f"accuracy={metrics_dict['accuracy']:.4f}, "
            f"time={elapsed_time:.2f}s"  # Display the elapsed time

        )

        # Update learning rate based on validation loss
        val_loss = metrics_dict.get("val_loss", loss_val.item())  # Use current loss if val_loss is not available
        scheduler.step(val_loss)

    save_model(model)
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=2024)

    # Pass all arguments to train
    train_detection(**vars(parser.parse_args()))
