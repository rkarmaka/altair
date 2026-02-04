"""
Loss functions for segmentation tasks.

Provides common loss functions and the ability to combine multiple losses
with configurable weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from altair.core.registry import LOSSES


@LOSSES.register("bce")
class BCELoss(nn.Module):
    """Binary cross-entropy loss for binary segmentation."""

    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute BCE loss."""
        return self.loss(pred, target.float())


@LOSSES.register("ce")
class CrossEntropyLoss(nn.Module):
    """Cross-entropy loss for multi-class segmentation."""

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss."""
        return self.loss(pred, target.long())


@LOSSES.register("dice")
class DiceLoss(nn.Module):
    """
    Dice loss for segmentation.

    Works for both binary and multi-class segmentation.

    Args:
        smooth: Smoothing factor to avoid division by zero.
        multiclass: Whether to use multi-class Dice.
    """

    def __init__(self, smooth: float = 1.0, multiclass: bool = True):
        super().__init__()
        self.smooth = smooth
        self.multiclass = multiclass

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss."""
        if self.multiclass:
            return self._multiclass_dice(pred, target)
        else:
            return self._binary_dice(pred, target)

    def _binary_dice(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Binary Dice loss."""
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1).float()

        intersection = (pred * target).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred.sum() + target.sum() + self.smooth
        )
        return 1 - dice

    def _multiclass_dice(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Multi-class Dice loss."""
        num_classes = pred.shape[1]
        pred = F.softmax(pred, dim=1)

        # One-hot encode target
        target_onehot = F.one_hot(target.long(), num_classes)
        target_onehot = target_onehot.permute(0, 3, 1, 2).float()

        # Compute Dice for each class
        dice_sum = 0.0
        for c in range(num_classes):
            pred_c = pred[:, c]
            target_c = target_onehot[:, c]

            intersection = (pred_c * target_c).sum()
            dice_c = (2.0 * intersection + self.smooth) / (
                pred_c.sum() + target_c.sum() + self.smooth
            )
            dice_sum += dice_c

        return 1 - dice_sum / num_classes


@LOSSES.register("focal")
class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance.

    Args:
        alpha: Weighting factor for positive class.
        gamma: Focusing parameter.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Focal loss."""
        ce_loss = F.cross_entropy(pred, target.long(), reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


@LOSSES.register("mse")
class MSELoss(nn.Module):
    """Mean squared error loss for regression tasks."""

    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute MSE loss."""
        return self.loss(pred, target.float())


@LOSSES.register("l1")
class L1Loss(nn.Module):
    """L1 (MAE) loss for regression tasks."""

    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute L1 loss."""
        return self.loss(pred, target.float())


class CombinedLoss(nn.Module):
    """
    Combine multiple loss functions with weights.

    Args:
        losses: Dictionary mapping loss names to loss modules.
        weights: Dictionary mapping loss names to weights.
    """

    def __init__(
        self,
        losses: dict[str, nn.Module],
        weights: dict[str, float],
    ):
        super().__init__()
        self.losses = nn.ModuleDict(losses)
        self.weights = weights

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute weighted sum of losses."""
        total_loss = 0.0
        for name, loss_fn in self.losses.items():
            weight = self.weights.get(name, 1.0)
            total_loss += weight * loss_fn(pred, target)
        return total_loss


def build_loss(
    loss_name: str | list[str],
    loss_weights: dict[str, float],
    task: str,
    num_classes: int,
) -> nn.Module:
    """
    Build a loss function from configuration.

    Args:
        loss_name: Name of loss function (e.g., 'ce+dice', 'focal'), list of loss names,
            or legacy underscore format (e.g., 'ce_dice'). Use '+' separator for
            combining losses to avoid ambiguity with loss names containing underscores.
        loss_weights: Weights for combined losses.
        task: Task type ('binary', 'multiclass', 'regression').
        num_classes: Number of output classes.

    Returns:
        Loss module.

    Raises:
        KeyError: If a loss name is not found in the registry.
    """
    # Parse loss names from various formats
    if isinstance(loss_name, list):
        # Already a list of loss names
        loss_names = loss_name
    elif "+" in loss_name:
        # New format: "ce+dice"
        loss_names = [name.strip() for name in loss_name.split("+")]
    elif loss_name in LOSSES:
        # Single loss that exists in registry (handles names like "focal_loss")
        loss_names = [loss_name]
    elif "_" in loss_name:
        # Legacy format: "ce_dice" - split on underscore
        # But first check if the full name exists in registry
        loss_names = loss_name.split("_")
    else:
        # Single loss
        loss_names = [loss_name]

    if len(loss_names) == 1:
        # Single loss - try to build from registry
        name = loss_names[0]
        try:
            return LOSSES.build(name)
        except KeyError:
            available = LOSSES.registered_names
            raise KeyError(
                f"Loss '{name}' not found. Available losses: {available}"
            ) from None

    # Combined loss
    losses = {}
    for name in loss_names:
        try:
            if name == "dice":
                losses[name] = DiceLoss(multiclass=(task == "multiclass"))
            elif name == "bce" and task == "binary":
                losses[name] = BCELoss()
            elif name == "ce" and task in ("binary", "multiclass"):
                losses[name] = CrossEntropyLoss()
            elif name == "focal":
                losses[name] = FocalLoss()
            elif name == "mse" and task == "regression":
                losses[name] = MSELoss()
            elif name == "l1" and task == "regression":
                losses[name] = L1Loss()
            else:
                # Try to build from registry
                losses[name] = LOSSES.build(name)
        except KeyError:
            available = LOSSES.registered_names
            raise KeyError(
                f"Loss '{name}' not found. Available losses: {available}"
            ) from None

    return CombinedLoss(losses, loss_weights)
