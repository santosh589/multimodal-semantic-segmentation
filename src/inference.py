"""Inference helpers for patch prediction and reconstruction of full images.

Imported by ``evaluate_test_set.py``.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def logits_to_predictions(
    logits: torch.Tensor,
) -> torch.Tensor:
    """
    Convert logits into integer class predictions.

    Input:
        [batch, classes, height, width]

    Output:
        [batch, height, width]
    """
    if logits.ndim != 4:
        raise ValueError(
            "Expected logits with shape "
            "[batch, classes, height, width]. "
            f"Received: {tuple(logits.shape)}"
        )

    return torch.argmax(
        logits,
        dim=1,
    )


@torch.no_grad()
def predict_batch(
    model: nn.Module,
    images: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    """
    Run inference on one batch.
    """
    model.eval()

    logits = model(images)

    predictions = logits_to_predictions(logits)

    return logits, predictions


def normalize_area_id(
    value: Any,
) -> int:
    """
    Convert values such as:

        23
        "23"
        "area23"
        "Area 23"

    into integer area ID 23.
    """
    if isinstance(
        value,
        torch.Tensor,
    ):
        value = value.item()

    if isinstance(
        value,
        (int, np.integer),
    ):
        return int(value)

    match = re.search(
        r"(\d+)",
        str(value),
    )

    if match is None:
        raise ValueError(f"Could not extract area ID from: {value}")

    return int(match.group(1))


def get_batch_item(
    batch_value: Any,
    index: int,
) -> Any:
    """
    Extract one item from a collated DataLoader field.
    """
    if isinstance(
        batch_value,
        torch.Tensor,
    ):
        item = batch_value[index]

        if item.ndim == 0:
            return item.item()

        return item

    if isinstance(
        batch_value,
        np.ndarray,
    ):
        return batch_value[index]

    if isinstance(
        batch_value,
        (list, tuple),
    ):
        return batch_value[index]

    return batch_value


@torch.no_grad()
def reconstruct_test_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    area_shapes: dict[int, tuple[int, int]],
    number_of_classes: int = 6,
) -> tuple[
    dict[int, np.ndarray],
    dict[int, np.ndarray],
]:
    """
    Reconstruct complete test-area predictions from patches.

    Overlapping border patches are handled by averaging model
    logits before selecting the final class.

    Parameters
    ----------
    model:
        Trained semantic-segmentation model.

    dataloader:
        Test DataLoader.

    device:
        CPU, CUDA or MPS.

    area_shapes:
        Mapping:

            area_id -> (height, width)

    number_of_classes:
        Number of semantic classes.

    Returns
    -------
    predictions:
        Dictionary containing a complete class-index prediction
        for each test area.

    coverage:
        Dictionary containing the number of patches contributing
        to every pixel.
    """
    model.eval()

    logit_sums: dict[
        int,
        np.ndarray,
    ] = {}

    coverage_counts: dict[
        int,
        np.ndarray,
    ] = {}

    for area_id, (
        height,
        width,
    ) in area_shapes.items():
        logit_sums[area_id] = np.zeros(
            (
                number_of_classes,
                height,
                width,
            ),
            dtype=np.float32,
        )

        coverage_counts[area_id] = np.zeros(
            (
                height,
                width,
            ),
            dtype=np.float32,
        )

    total_batches = len(dataloader)

    for batch_number, batch in enumerate(
        dataloader,
        start=1,
    ):
        images = batch["image"].to(
            device,
            non_blocking=False,
        )

        logits = model(images)

        logits_numpy = (
            logits.detach()
            .to("cpu")
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )

        batch_size = int(logits_numpy.shape[0])

        for batch_index in range(batch_size):
            area_id = normalize_area_id(
                get_batch_item(
                    batch["area_id"],
                    batch_index,
                )
            )

            row = int(
                get_batch_item(
                    batch["row"],
                    batch_index,
                )
            )

            column = int(
                get_batch_item(
                    batch["column"],
                    batch_index,
                )
            )

            if area_id not in area_shapes:
                raise KeyError(f"Area {area_id} is not present " "in area_shapes.")

            full_height, full_width = area_shapes[area_id]

            patch_logits = logits_numpy[batch_index]

            patch_height = min(
                patch_logits.shape[1],
                full_height - row,
            )

            patch_width = min(
                patch_logits.shape[2],
                full_width - column,
            )

            if patch_height <= 0 or patch_width <= 0:
                raise ValueError(
                    "Patch coordinates fall outside "
                    f"area {area_id}: "
                    f"row={row}, column={column}"
                )

            logit_sums[area_id][
                :,
                row : row + patch_height,
                column : column + patch_width,
            ] += patch_logits[
                :,
                :patch_height,
                :patch_width,
            ]

            coverage_counts[area_id][
                row : row + patch_height,
                column : column + patch_width,
            ] += 1.0

        print(
            f"\rInference batch " f"{batch_number}/{total_batches}",
            end="",
            flush=True,
        )

    print()

    predictions: dict[
        int,
        np.ndarray,
    ] = {}

    for area_id in sorted(area_shapes):
        coverage = coverage_counts[area_id]

        uncovered_pixels = int(np.count_nonzero(coverage == 0))

        if uncovered_pixels > 0:
            raise RuntimeError(
                f"Area {area_id} contains " f"{uncovered_pixels} uncovered pixels."
            )

        averaged_logits = (
            logit_sums[area_id]
            / coverage[
                np.newaxis,
                ...,
            ]
        )

        predictions[area_id] = np.argmax(
            averaged_logits,
            axis=0,
        ).astype(np.uint8)

    return (
        predictions,
        coverage_counts,
    )
