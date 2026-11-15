"""Reusable single-epoch training routines used by ``train_experiment.py``."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from inference import logits_to_predictions
from losses import calculate_segmentation_loss
from metrics import (
    calculate_metrics,
    create_confusion_matrix,
    update_confusion_matrix,
)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    number_of_classes: int = 6,
    ignore_index: int = 255,
    accumulation_steps: int = 1,
    gradient_clip: float = 0.0,
) -> dict[str, Any]:
    """
    Train a semantic-segmentation model for one epoch.
    """
    if accumulation_steps <= 0:
        raise ValueError("accumulation_steps must be greater than zero.")

    model.train()

    confusion_matrix = create_confusion_matrix(number_of_classes)

    total_loss = 0.0
    total_samples = 0

    number_of_batches = len(dataloader)

    optimizer.zero_grad(set_to_none=True)

    for batch_index, batch in enumerate(dataloader):
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

        group_start = batch_index // accumulation_steps * accumulation_steps

        current_group_size = min(
            accumulation_steps,
            number_of_batches - group_start,
        )

        scaled_loss = loss / current_group_size

        scaled_loss.backward()

        group_finished = (batch_index + 1) % accumulation_steps == 0

        final_batch = batch_index + 1 == number_of_batches

        if group_finished or final_batch:
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=gradient_clip,
                )

            optimizer.step()

            optimizer.zero_grad(set_to_none=True)

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
