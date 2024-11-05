from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class Classifier(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
    ):
        """
        A convolutional network for image classification.

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        # Define the convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),  # Conv1
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Conv2
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Conv3
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # Conv4
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        # Global Average Pooling layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Reduces to (B, 256, 1, 1)

        # Fully connected layer to map features to num_classes
        self.fc_layers = nn.Sequential(
            nn.Flatten(),  # Flatten the (B, 256, 1, 1) -> (B, 256)
            nn.Linear(256, num_classes)  # Fully connected layer (256 -> num_classes)
        )



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, h, w) image

        Returns:
            tensor (b, num_classes) logits
        """
        # optional: normalizes the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # Pass through convolutional layers
        z = self.conv_layers(z)

        # Global average pooling (reduce spatial dimensions)
        z = self.global_avg_pool(z)

        # Pass through fully connected layers
        logits = self.fc_layers(z)

        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Used for inference, returns class labels
        This is what the AccuracyMetric uses as input (this is what the grader will use!).
        You should not have to modify this function.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            pred (torch.LongTensor): class labels {0, 1, ..., 5} with shape (b, h, w)
        """
        return self(x).argmax(dim=1)


class Detector(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
    ):
        """
        A single model that performs segmentation and depth regression

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        # Down-sampling layers
        self.down1 = self.conv_block(in_channels, 16)  # (B, 3) -> (B, 16, 48, 64)
        self.down2 = self.conv_block(16, 32)  # (B, 16, 48, 64) -> (B, 32, 24, 32)
        self.down3 = self.conv_block(32, 64)  # (B, 32, 24, 32) -> (B, 64, 12, 16)
        self.down4 = self.conv_block(64, 128)  # (B, 64, 12, 16) -> (B, 128, 6, 8)
        self.down5 = self.conv_block(128, 256)  # New Layer: (B, 128, 6, 8) -> (B, 256, 3, 4)

        # Up-sampling layers
        self.up1 = self.upconv_block(256, 128)  # (B, 256, 3, 4) -> (B, 128, 6, 8)
        self.up2 = self.upconv_block(128 + 128, 64)  # (B, 128 + 128, 6, 8) -> (B, 64, 12, 16)
        self.up3 = self.upconv_block(64 + 64, 32)  # (B, 64 + 64, 12, 16) -> (B, 32, 24, 32)
        self.up4 = self.upconv_block(32 + 32, 16)  # (B, 32 + 32, 24, 32) -> (B, 16, 48, 64)
        self.up5 = self.upconv_block(16 + 16, 16)  # New Layer: (B, 16 + 16, 48, 64) -> (B, 16, 96, 128)

        # Output layers
        self.logits = nn.Conv2d(16, num_classes, kernel_size=1)  # Output: (B, num_classes, 96, 128)
        self.depth = nn.Conv2d(16, 1, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        """Define a convolutional block with BatchNorm and ReLU."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)  # Example dropout layer
        )

    def upconv_block(self, in_channels, out_channels):
        """Define an up-convolutional block with BatchNorm and ReLU."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )



    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used in training, takes an image and returns raw logits and raw depth.
        This is what the loss functions use as input.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.FloatTensor, torch.FloatTensor):
                - logits (b, num_classes, h, w)
                - depth (b, h, w)
        """
        # optional: normalizes the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # Down-sampling path
        down1_out = self.down1(z)  # (B, 16, 48, 64)
        down2_out = self.down2(down1_out)  # (B, 32, 24, 32)
        down3_out = self.down3(down2_out)  # (B, 64, 12, 16)
        down4_out = self.down4(down3_out)  # (B, 128, 6, 8)
        down5_out = self.down5(down4_out)  # (B, 256, 3, 4)

        # Up-sampling path with skip connections
        up1_out = self.up1(down5_out)  # (B, 128, 6, 8)
        up1_out = torch.cat([up1_out, down4_out], dim=1)  # Concatenate with skip connection
        up2_out = self.up2(up1_out)  # (B, 64, 12, 16)
        up2_out = torch.cat([up2_out, down3_out], dim=1)  # Concatenate with skip connection
        up3_out = self.up3(up2_out)  # (B, 32, 24, 32)
        up3_out = torch.cat([up3_out, down2_out], dim=1)  # Concatenate with skip connection
        up4_out = self.up4(up3_out)  # (B, 16, 48, 64)
        up4_out = torch.cat([up4_out, down1_out], dim=1)  # Concatenate with skip connection
        up5_out = self.up5(up4_out)  # (B, 16, 96, 128)

        # Final output layers
        logits = self.logits(up5_out)  # (B, num_classes, 96, 128)
        raw_depth = self.depth(up5_out)  # (B, 1, 96, 128)

        depth = raw_depth.squeeze(1)  # (B, 96, 128)

        return logits, depth

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used for inference, takes an image and returns class labels and normalized depth.
        This is what the metrics use as input (this is what the grader will use!).

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.LongTensor, torch.FloatTensor):
                - pred: class labels {0, 1, 2} with shape (b, h, w)
                - depth: normalized depth [0, 1] with shape (b, h, w)
        """
        logits, raw_depth = self(x)
        pred = logits.argmax(dim=1)

        # Optional additional post-processing for depth only if needed
        depth = raw_depth

        return pred, depth


MODEL_FACTORY = {
    "classifier": Classifier,
    "detector": Detector,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def debug_model(batch_size: int = 1):
    """
    Test your model implementation

    Feel free to add additional checks to this function -
    this function is NOT used for grading
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_batch = torch.rand(batch_size, 3, 64, 64).to(device)

    print(f"Input shape: {sample_batch.shape}")

    model = load_model("classifier", in_channels=3, num_classes=6).to(device)
    output = model(sample_batch)

    # should output logits (b, num_classes)
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    debug_model()
