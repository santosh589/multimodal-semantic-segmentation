"""Loss-function factories used by the segmentation training pipeline."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


DEFAULT_IGNORE_INDEX = 255


def create_segmentation_loss(
    ignore_index: int = DEFAULT_IGNORE_INDEX,
    class_weights: Sequence[float] | None = None,
    device: torch.device | None = None,
) -> nn.Module:
    """
    Create the semantic-segmentation loss.

    The assignment uses CrossEntropyLoss because:
    - the model produces six logits per pixel;
    - the target contains integer class IDs from 0 to 5.

    Parameters
    ----------
    ignore_index:
        Label value ignored by the loss.

    class_weights:
        Optional class weights. Leave as None for the
        standard unweighted CrossEntropyLoss.

    device:
        Device on which class weights should be stored.
    """
    weight_tensor = None

    if class_weights is not None:
        weight_tensor = torch.tensor(
            class_weights,
            dtype=torch.float32,
            device=device,
        )

        if weight_tensor.ndim != 1:
            raise ValueError("Class weights must be a one-dimensional sequence.")

    criterion = nn.CrossEntropyLoss(
        weight=weight_tensor,
        ignore_index=ignore_index,
    )

    return criterion


def validate_loss_inputs(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    """
    Validate model-output and target shapes before loss computation.

    Expected
    --------
    logits:
        [batch, classes, height, width]

    targets:
        [batch, height, width]
    """
    if logits.ndim != 4:
        raise ValueError(
            "Logits must have shape "
            "[batch, classes, height, width]. "
            f"Received: {tuple(logits.shape)}"
        )

    if targets.ndim != 3:
        raise ValueError(
            "Targets must have shape "
            "[batch, height, width]. "
            f"Received: {tuple(targets.shape)}"
        )

    if logits.shape[0] != targets.shape[0]:
        raise ValueError("Logits and targets have different batch sizes.")

    if logits.shape[2:] != targets.shape[1:]:
        raise ValueError(
            "Logits and targets have different spatial dimensions.\n"
            f"Logits: {tuple(logits.shape)}\n"
            f"Targets: {tuple(targets.shape)}"
        )

    if not logits.is_floating_point():
        raise TypeError(f"Logits must be floating point. Received: {logits.dtype}")

    if targets.dtype != torch.int64:
        raise TypeError(f"Targets must be torch.int64. Received: {targets.dtype}")


def calculate_segmentation_loss(
    criterion: nn.Module,
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Validate inputs and calculate the segmentation loss.
    """
    validate_loss_inputs(
        logits=logits,
        targets=targets,
    )

    return criterion(
        logits,
        targets,
    )
