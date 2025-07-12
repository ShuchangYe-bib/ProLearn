# === Third-party Libraries ===
import torch
import torch.nn as nn
from monai.losses import DiceCELoss
from torchmetrics import Accuracy, Dice, JaccardIndex

# === Project Modules ===
from .mmunet import MMUNet
from .unets import SwinUNet, TransUNet, UNet, AttnUNet, LViT, UNetPP


class ProLearn(nn.Module):
    """
    ProLearn: Wrapper model that integrates MMUNet (or alternatives) with loss functions and evaluation metrics.
    Supports optional prototype-enhanced architectures for language-guided segmentation.
    """

    def __init__(self, args, prototype=None):
        """
        Initialize the ProLearn model.

        Args:
            args (Namespace): Configuration arguments.
            prototype (Optional[nn.Module]): Optional prototype encoder module for semantic alignment.
        """
        super(ProLearn, self).__init__()

        # === Model Backbone ===
        self.model = MMUNet(args, prototype)
        # Alternative backbones (uncomment to experiment):
        # self.model = LViT()
        # self.model = SwinUNet()

        self.lr = args.lr
        self.history = {}  # For tracking losses/metrics during training

        # === Segmentation Loss Function ===
        self.losses = {
            "loss_seg": DiceCELoss()
        }

        # === Evaluation Metrics ===
        self.train_metrics = {
            "acc": Accuracy(task='binary').to("cuda"),
            "dice": Dice().to("cuda"),
            "miou": JaccardIndex(num_classes=2, task='binary').to("cuda")
        }
        self.val_metrics = self.train_metrics
        self.test_metrics = self.train_metrics

    def forward_feature(self, x):
        """
        Forward pass to extract model outputs/features.

        Args:
            x (Tensor or Tuple): Input tensor or multimodal tuple.

        Returns:
            Tensor: Segmentation logits.
        """
        return self.model(x)

    def forward(self, x, benchmark=""):
        """
        Forward pass with loss computation.

        Args:
            x (Dict): Dictionary with keys 'image', 'text', 'label', etc.
            benchmark (str): Name of model type for modality-specific behavior.

        Returns:
            Dict: Input dictionary with added 'logits' and 'loss' fields.
        """
        if benchmark == "LViT":
            x_input = (x["image"], x["text"])
        elif "UNet" in benchmark:
            x_input = x["image"]
        else:
            x_input = x  # MMUNet-style dict-based input

        x["logits"] = self.forward_feature(x_input)
        x["loss"] = self.losses["loss_seg"](x["logits"], x["label"])
        return x
