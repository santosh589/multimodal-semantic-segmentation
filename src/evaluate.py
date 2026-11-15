"""Reusable metric calculation, report writing, and visualization helpers.

Imported by training and test-evaluation scripts.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from inference import logits_to_predictions
from losses import calculate_segmentation_loss
from metrics import (
    calculate_metrics,
    create_confusion_matrix,
    update_confusion_matrix,
)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    number_of_classes: int = 6,
    ignore_index: int = 255,
) -> dict[str, Any]:
    """
    Evaluate a model on validation or test patches.
    """
    model.eval()

    confusion_matrix = create_confusion_matrix(number_of_classes)

    total_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        images = batch["image"].to(
            device,
            non_blocking=False,
        )

        targets = batch["label"].to(
            device,
            non_blocking=False,
        )

        logits = model(images)

        loss = calculate_segmentation_loss(
            criterion=criterion,
            logits=logits,
            targets=targets,
        )

        predictions = logits_to_predictions(logits)

        batch_size = int(images.shape[0])

        total_loss += float(loss.item()) * batch_size

        total_samples += batch_size

        update_confusion_matrix(
            confusion_matrix=(confusion_matrix),
            predictions=predictions,
            targets=targets,
            number_of_classes=(number_of_classes),
            ignore_index=ignore_index,
        )

    metrics = calculate_metrics(confusion_matrix)

    metrics["loss"] = total_loss / max(total_samples, 1)

    metrics["confusion_matrix"] = confusion_matrix.tolist()

    metrics["number_of_samples"] = total_samples

    return metrics


def evaluate_class_masks(
    prediction: np.ndarray,
    target: np.ndarray,
    number_of_classes: int = 6,
    ignore_index: int = 255,
) -> tuple[
    dict[str, Any],
    torch.Tensor,
]:
    """
    Evaluate one complete prediction against one complete
    ground-truth class mask.
    """
    if prediction.shape != target.shape:
        raise ValueError(
            "Prediction and target dimensions differ.\n"
            f"Prediction: {prediction.shape}\n"
            f"Target: {target.shape}"
        )

    prediction_tensor = torch.from_numpy(
        prediction.astype(
            np.int64,
            copy=False,
        )
    )

    target_tensor = torch.from_numpy(
        target.astype(
            np.int64,
            copy=False,
        )
    )

    confusion_matrix = create_confusion_matrix(number_of_classes)

    update_confusion_matrix(
        confusion_matrix=confusion_matrix,
        predictions=prediction_tensor,
        targets=target_tensor,
        number_of_classes=(number_of_classes),
        ignore_index=ignore_index,
    )

    metrics = calculate_metrics(confusion_matrix)

    metrics["confusion_matrix"] = confusion_matrix.tolist()

    return (
        metrics,
        confusion_matrix,
    )


def class_mask_to_rgb(
    class_mask: np.ndarray,
    class_colors: dict[
        int,
        tuple[int, int, int],
    ],
) -> np.ndarray:
    """
    Convert a class-index mask into its original RGB colours.
    """
    rgb = np.zeros(
        (
            class_mask.shape[0],
            class_mask.shape[1],
            3,
        ),
        dtype=np.uint8,
    )

    for class_id, color in class_colors.items():
        rgb[class_mask == class_id] = color

    return rgb


def save_rgb_mask(
    class_mask: np.ndarray,
    class_colors: dict[
        int,
        tuple[int, int, int],
    ],
    output_path: Path,
) -> None:
    """
    Save a class-index mask as a colour PNG.
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rgb = class_mask_to_rgb(
        class_mask=class_mask,
        class_colors=class_colors,
    )

    Image.fromarray(
        rgb,
        mode="RGB",
    ).save(output_path)


def save_confusion_matrix_figure(
    confusion_matrix: torch.Tensor,
    class_names: dict[int, str],
    output_path: Path,
    title: str,
) -> None:
    """
    Save a confusion-matrix figure without seaborn.
    """
    matrix = confusion_matrix.detach().cpu().numpy()

    row_totals = matrix.sum(
        axis=1,
        keepdims=True,
    )

    normalized = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(
            matrix,
            dtype=np.float64,
        ),
        where=row_totals != 0,
    )

    labels = [class_names[class_id] for class_id in sorted(class_names)]

    figure, axis = plt.subplots(figsize=(10, 8))

    image = axis.imshow(
        normalized,
        vmin=0.0,
        vmax=1.0,
    )

    figure.colorbar(
        image,
        ax=axis,
        label="Row-normalized proportion",
    )

    axis.set_xticks(np.arange(len(labels)))

    axis.set_yticks(np.arange(len(labels)))

    axis.set_xticklabels(
        labels,
        rotation=45,
        ha="right",
    )

    axis.set_yticklabels(labels)

    axis.set_xlabel("Predicted class")

    axis.set_ylabel("Ground-truth class")

    axis.set_title(title)

    threshold = 0.5

    for row in range(normalized.shape[0]):
        for column in range(normalized.shape[1]):
            text = f"{normalized[row, column]:.2f}\n" f"({matrix[row, column]:,})"

            axis.text(
                column,
                row,
                text,
                ha="center",
                va="center",
                color=("white" if normalized[row, column] > threshold else "black"),
                fontsize=8,
            )

    figure.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_prediction_comparison(
    target: np.ndarray,
    prediction: np.ndarray,
    class_colors: dict[
        int,
        tuple[int, int, int],
    ],
    output_path: Path,
    title: str,
) -> None:
    """
    Save ground truth and prediction beside each other.
    """
    target_rgb = class_mask_to_rgb(
        class_mask=target,
        class_colors=class_colors,
    )

    prediction_rgb = class_mask_to_rgb(
        class_mask=prediction,
        class_colors=class_colors,
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(14, 7),
    )

    axes[0].imshow(target_rgb)

    axes[0].set_title("Ground truth")

    axes[0].axis("off")

    axes[1].imshow(prediction_rgb)

    axes[1].set_title("Prediction")

    axes[1].axis("off")

    figure.suptitle(title)

    figure.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def create_per_area_row(
    area_id: int,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """
    Create one summary CSV row for a test area.
    """
    row: dict[str, Any] = {
        "area_id": area_id,
        "overall_accuracy": (metrics["overall_accuracy"]),
        "macro_precision": (metrics["macro_precision"]),
        "macro_recall": (metrics["macro_recall"]),
        "macro_f1": (metrics["macro_f1"]),
        "mean_iou": (metrics["mean_iou"]),
    }

    return row


def create_per_class_rows(
    area_id: int | str,
    metrics: dict[str, Any],
    class_names: dict[int, str],
) -> list[dict[str, Any]]:
    """
    Create one result row per semantic class.
    """
    rows = []

    for class_id in sorted(class_names):
        rows.append(
            {
                "area_id": area_id,
                "class_id": class_id,
                "class_name": (class_names[class_id]),
                "precision": (metrics["class_precision"][class_id]),
                "recall": (metrics["class_recall"][class_id]),
                "f1": (metrics["class_f1"][class_id]),
                "iou": (metrics["class_iou"][class_id]),
                "support_pixels": (metrics["class_support"][class_id]),
            }
        )

    return rows


def save_csv_rows(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Save dictionaries to CSV.
    """
    if not rows:
        return

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()

        writer.writerows(rows)


def save_json(
    data: dict[str, Any],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=4,
        )
