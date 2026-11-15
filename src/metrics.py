"""Metric and confusion-matrix utilities for pixel-wise segmentation."""

from __future__ import annotations

from typing import Any

import torch


def create_confusion_matrix(
    number_of_classes: int,
) -> torch.Tensor:
    """
    Create an empty confusion matrix on CPU.

    Rows:
        Ground-truth classes.

    Columns:
        Predicted classes.
    """
    if number_of_classes <= 1:
        raise ValueError("number_of_classes must be greater than one.")

    return torch.zeros(
        (
            number_of_classes,
            number_of_classes,
        ),
        dtype=torch.int64,
    )


@torch.no_grad()
def update_confusion_matrix(
    confusion_matrix: torch.Tensor,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    number_of_classes: int,
    ignore_index: int = 255,
) -> None:
    """
    Update a confusion matrix using one prediction batch.

    Parameters
    ----------
    predictions:
        Integer predictions with shape [B,H,W].

    targets:
        Integer ground truth with shape [B,H,W].
    """
    if predictions.shape != targets.shape:
        raise ValueError(
            "Prediction and target shapes do not match.\n"
            f"Predictions: {tuple(predictions.shape)}\n"
            f"Targets: {tuple(targets.shape)}"
        )

    predictions = (
        predictions.detach()
        .to(
            device="cpu",
            dtype=torch.int64,
        )
        .reshape(-1)
    )

    targets = (
        targets.detach()
        .to(
            device="cpu",
            dtype=torch.int64,
        )
        .reshape(-1)
    )

    valid_mask = (
        (targets != ignore_index)
        & (targets >= 0)
        & (targets < number_of_classes)
        & (predictions >= 0)
        & (predictions < number_of_classes)
    )

    if not torch.any(valid_mask):
        return

    valid_targets = targets[valid_mask]

    valid_predictions = predictions[valid_mask]

    encoded_pairs = valid_targets * number_of_classes + valid_predictions

    counts = torch.bincount(
        encoded_pairs,
        minlength=(number_of_classes * number_of_classes),
    )

    confusion_matrix += counts.reshape(
        number_of_classes,
        number_of_classes,
    )


def calculate_metrics(
    confusion_matrix: torch.Tensor,
) -> dict[str, Any]:
    """
    Calculate multiclass segmentation metrics.

    Returns
    -------
    overall_accuracy:
        Correct pixels divided by all evaluated pixels.

    macro_precision:
        Average precision over classes present in ground truth.

    macro_recall:
        Average recall over classes present in ground truth.

    macro_f1:
        Average F1 over classes present in ground truth.

    mean_iou:
        Average IoU over classes present in ground truth.
    """
    if confusion_matrix.ndim != 2:
        raise ValueError("Confusion matrix must be two-dimensional.")

    if confusion_matrix.shape[0] != confusion_matrix.shape[1]:
        raise ValueError("Confusion matrix must be square.")

    matrix = confusion_matrix.to(dtype=torch.float64)

    true_positive = torch.diag(matrix)

    ground_truth_count = matrix.sum(dim=1)

    predicted_count = matrix.sum(dim=0)

    total_pixels = matrix.sum()

    precision = true_positive / predicted_count.clamp_min(1.0)

    recall = true_positive / ground_truth_count.clamp_min(1.0)

    f1_denominator = precision + recall

    f1 = torch.where(
        f1_denominator > 0,
        (2.0 * precision * recall / f1_denominator),
        torch.zeros_like(f1_denominator),
    )

    union = ground_truth_count + predicted_count - true_positive

    iou = true_positive / union.clamp_min(1.0)

    classes_present = ground_truth_count > 0

    if torch.any(classes_present):
        macro_precision = precision[classes_present].mean().item()

        macro_recall = recall[classes_present].mean().item()

        macro_f1 = f1[classes_present].mean().item()

        mean_iou = iou[classes_present].mean().item()

    else:
        macro_precision = 0.0
        macro_recall = 0.0
        macro_f1 = 0.0
        mean_iou = 0.0

    if total_pixels.item() > 0:
        overall_accuracy = (true_positive.sum() / total_pixels).item()

    else:
        overall_accuracy = 0.0

    return {
        "overall_accuracy": overall_accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "mean_iou": mean_iou,
        "class_precision": [float(value) for value in precision.tolist()],
        "class_recall": [float(value) for value in recall.tolist()],
        "class_f1": [float(value) for value in f1.tolist()],
        "class_iou": [float(value) for value in iou.tolist()],
        "class_support": [int(value) for value in ground_truth_count.tolist()],
    }
