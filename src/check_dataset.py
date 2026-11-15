"""Validate dataset tensors, data ranges, and image-label alignment.

Run: ``python src/check_dataset.py --batch-size 8``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import (
    PROJECT_ROOT,
    SegmentationPatchDataset,
    create_dataloaders,
)


# ============================================================
# Paths
# ============================================================

NORMALIZATION_STATS_PATH = PROJECT_ROOT / "metadata" / "normalization_stats.json"

CLASS_MAPPING_PATH = PROJECT_ROOT / "metadata" / "class_mapping.json"

TRAIN_PATCHES_PATH = PROJECT_ROOT / "metadata" / "train_patches.csv"

OUTPUT_FIGURE = PROJECT_ROOT / "outputs" / "figures" / "dataset_check.png"


# ============================================================
# Arguments
# ============================================================


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Check segmentation Dataset and DataLoader outputs.")
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help=("Batch size for the smoke test. " "Default: 8."),
    )

    return parser.parse_args()


# ============================================================
# Utilities
# ============================================================


def load_json(path: Path) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def class_mask_to_rgb(
    class_mask: np.ndarray,
    class_colors: dict[
        int,
        tuple[int, int, int],
    ],
) -> np.ndarray:
    """
    Convert class IDs back into RGB colors.
    """
    height, width = class_mask.shape

    rgb = np.zeros(
        (height, width, 3),
        dtype=np.uint8,
    )

    for class_id, color in class_colors.items():
        rgb[class_mask == class_id] = color

    return rgb


def denormalize_multispectral(
    tensor: torch.Tensor,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """
    Convert normalized [3,H,W] tensor back to
    displayable [H,W,3] data.
    """
    image = tensor.detach().cpu().numpy().astype(np.float32)

    mean = mean.reshape(
        3,
        1,
        1,
    )

    std = std.reshape(
        3,
        1,
        1,
    )

    image = image * std + mean

    image = np.clip(
        image,
        0.0,
        1.0,
    )

    return np.moveaxis(
        image,
        0,
        -1,
    )


# ============================================================
# Checks
# ============================================================


def inspect_dataset(
    input_mode: str,
) -> SegmentationPatchDataset:
    dataset = SegmentationPatchDataset(
        patch_csv=TRAIN_PATCHES_PATH,
        normalization_stats_path=(NORMALIZATION_STATS_PATH),
        class_mapping_path=(CLASS_MAPPING_PATH),
        input_mode=input_mode,
        project_root=PROJECT_ROOT,
        strict_labels=True,
    )

    sample = dataset[0]

    print("\n" + "=" * 75)
    print(f"INPUT MODE: {input_mode}")
    print("=" * 75)

    print(f"Number of training patches: " f"{len(dataset)}")

    print(f"Patch ID: {sample['patch_id']}")

    print(f"Area ID: {sample['area_id']}")

    print(f"Model input shape: " f"{tuple(sample['image'].shape)}")

    print(f"Multispectral shape: " f"{tuple(sample['multispectral'].shape)}")

    print(f"DSM shape: " f"{tuple(sample['dsm'].shape)}")

    print(f"Label shape: " f"{tuple(sample['label'].shape)}")

    print(f"Model input dtype: " f"{sample['image'].dtype}")

    print(f"Label dtype: " f"{sample['label'].dtype}")

    print(
        "Normalized multispectral range: "
        f"{sample['multispectral'].min().item():.4f} "
        f"to "
        f"{sample['multispectral'].max().item():.4f}"
    )

    print(
        "Scaled DSM range: "
        f"{sample['dsm'].min().item():.4f} "
        f"to "
        f"{sample['dsm'].max().item():.4f}"
    )

    unique_classes = torch.unique(sample["label"]).tolist()

    print(f"Classes in first patch: " f"{unique_classes}")

    expected_channels = 3 if input_mode == "multispectral" else 4

    assert sample["image"].shape == (
        expected_channels,
        256,
        256,
    )

    assert sample["multispectral"].shape == (
        3,
        256,
        256,
    )

    assert sample["dsm"].shape == (
        1,
        256,
        256,
    )

    assert sample["label"].shape == (
        256,
        256,
    )

    assert sample["image"].dtype == torch.float32

    assert sample["dsm"].dtype == torch.float32

    assert sample["label"].dtype == torch.int64

    assert sample["dsm"].min().item() >= 0.0

    assert sample["dsm"].max().item() <= 1.0

    assert all(
        class_id
        in {
            0,
            1,
            2,
            3,
            4,
            5,
        }
        for class_id in unique_classes
    )

    print("Sample checks passed.")

    return dataset


def inspect_dataloader(
    input_mode: str,
    batch_size: int,
) -> None:
    dataloaders = create_dataloaders(
        input_mode=input_mode,
        batch_size=batch_size,
        num_workers=0,
        project_root=PROJECT_ROOT,
    )

    print("\nDataLoader summary:")

    for split_name, dataloader in dataloaders.items():
        batch = next(iter(dataloader))

        print(
            f"  {split_name:10s} "
            f"images={tuple(batch['image'].shape)}, "
            f"labels={tuple(batch['label'].shape)}"
        )

        expected_channels = 3 if input_mode == "multispectral" else 4

        assert batch["image"].ndim == 4
        assert batch["image"].shape[1] == (expected_channels)

        assert batch["image"].shape[2:] == (
            256,
            256,
        )

        assert batch["label"].ndim == 3

        assert batch["label"].shape[1:] == (
            256,
            256,
        )

    print("DataLoader checks passed.")


# ============================================================
# Visualization
# ============================================================


def create_check_figure(
    dataset: SegmentationPatchDataset,
) -> None:
    statistics = load_json(NORMALIZATION_STATS_PATH)

    mapping = load_json(CLASS_MAPPING_PATH)

    mean = np.asarray(
        statistics["multispectral"]["mean"],
        dtype=np.float32,
    )

    std = np.asarray(
        statistics["multispectral"]["std"],
        dtype=np.float32,
    )

    class_colors = {
        int(class_information["id"]): tuple(class_information["color"])
        for class_information in mapping["classes"]
    }

    sample = dataset[0]

    multispectral_display = denormalize_multispectral(
        tensor=sample["multispectral"],
        mean=mean,
        std=std,
    )

    dsm_display = sample["dsm"][0].detach().cpu().numpy()

    label_display = class_mask_to_rgb(
        class_mask=(sample["label"].detach().cpu().numpy()),
        class_colors=class_colors,
    )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(16, 6),
    )

    axes[0].imshow(multispectral_display)

    axes[0].set_title("Multispectral patch\n" "Denormalized for display")

    axes[0].axis("off")

    axes[1].imshow(
        dsm_display,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[1].set_title("DSM patch\n" "Scaled only to [0,1]")

    axes[1].axis("off")

    axes[2].imshow(label_display)

    axes[2].set_title("Integer class mask\n" "Displayed with configured colors")

    axes[2].axis("off")

    figure.suptitle(
        f"{sample['patch_id']} — " f"row={sample['row']}, " f"column={sample['column']}"
    )

    figure.tight_layout()

    OUTPUT_FIGURE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        OUTPUT_FIGURE,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"\nDataset figure saved to:\n" f"{OUTPUT_FIGURE}")


# ============================================================
# Main
# ============================================================


def main() -> None:
    arguments = parse_arguments()

    dataset_a = inspect_dataset(input_mode="multispectral")

    inspect_dataloader(
        input_mode="multispectral",
        batch_size=arguments.batch_size,
    )

    inspect_dataset(input_mode="early_fusion")

    inspect_dataloader(
        input_mode="early_fusion",
        batch_size=arguments.batch_size,
    )

    create_check_figure(dataset_a)

    print("\n" + "=" * 75)
    print("ALL DATASET CHECKS PASSED")
    print("=" * 75)

    print("\nExperiment A input: " "[batch, 3, 256, 256]")

    print("Experiment B input: " "[batch, 4, 256, 256]")

    print("Labels: " "[batch, 256, 256], torch.int64")


if __name__ == "__main__":
    main()
