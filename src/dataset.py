"""Dataset and DataLoader utilities for multispectral and image-depth inputs.

Imported by training and evaluation scripts; it is not a standalone entry point.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import rasterio
import torch
from PIL import Image
from rasterio.windows import Window
from torch.utils.data import DataLoader, Dataset


# ============================================================
# Type definitions
# ============================================================

InputMode = Literal[
    "multispectral",
    "early_fusion",
    "separate",
]


# ============================================================
# Default project path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# Utility functions
# ============================================================


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def resolve_project_path(
    project_root: Path,
    path_value: str | Path,
) -> Path:
    """
    Resolve either an absolute path or a path relative
    to the project root.
    """
    path = Path(path_value)

    if path.is_absolute():
        return path

    return project_root / path


# ============================================================
# Patch-reading functions
# ============================================================


def read_raster_window(
    path: Path,
    row: int,
    column: int,
    height: int,
    width: int,
    indexes: list[int] | int | None = None,
) -> np.ndarray:
    """
    Read a raster window using Rasterio.

    Rasterio returns:
        [channels, height, width]

    For a single index, it returns:
        [height, width]
    """
    window = Window(
        col_off=column,
        row_off=row,
        width=width,
        height=height,
    )

    with rasterio.open(path) as source:
        array = source.read(
            indexes=indexes,
            window=window,
            boundless=False,
        )

    return array


def read_pillow_window(
    path: Path,
    row: int,
    column: int,
    height: int,
    width: int,
    mode: str | None = None,
) -> np.ndarray:
    """
    Read a rectangular image patch using Pillow.

    Pillow crop coordinates:
        left, upper, right, lower
    """
    crop_box = (
        column,
        row,
        column + width,
        row + height,
    )

    with Image.open(path) as image:
        if mode is not None:
            image = image.convert(mode)

        patch = image.crop(crop_box)
        array = np.asarray(patch)

    return array


# ============================================================
# RGB-to-class conversion
# ============================================================


def pack_rgb(rgb: np.ndarray) -> np.ndarray:
    """
    Convert each RGB color into a single integer.

    packed = R * 65536 + G * 256 + B
    """
    rgb = rgb.astype(
        np.int64,
        copy=False,
    )

    return rgb[..., 0] * 65_536 + rgb[..., 1] * 256 + rgb[..., 2]


def rgb_to_class_mask(
    rgb_label: np.ndarray,
    class_colors: dict[int, tuple[int, int, int]],
    ignore_index: int,
    strict: bool = True,
) -> np.ndarray:
    """
    Convert an RGB label into class IDs.

    Parameters
    ----------
    rgb_label:
        Array with shape [H,W,3].

    class_colors:
        Dictionary mapping class ID to RGB color.

    ignore_index:
        Value assigned to unknown colors.

    strict:
        Raise an error when unknown colors are encountered.

    Returns
    -------
    np.ndarray
        Integer mask with shape [H,W].
    """
    if rgb_label.ndim != 3:
        raise ValueError(
            f"Expected RGB label [H,W,3], " f"received shape {rgb_label.shape}"
        )

    if rgb_label.shape[2] < 3:
        raise ValueError(
            f"Expected at least 3 label channels, " f"received shape {rgb_label.shape}"
        )

    rgb_label = rgb_label[:, :, :3]

    packed_label = pack_rgb(rgb_label)

    class_mask = np.full(
        rgb_label.shape[:2],
        fill_value=ignore_index,
        dtype=np.int64,
    )

    for class_id, color in class_colors.items():
        red, green, blue = color

        packed_color = red * 65_536 + green * 256 + blue

        class_mask[packed_label == packed_color] = class_id

    unknown_mask = class_mask == ignore_index

    if strict and np.any(unknown_mask):
        unknown_colors = np.unique(
            rgb_label[unknown_mask],
            axis=0,
        )

        raise ValueError(
            "Unexpected RGB colors found in label patch. "
            f"Examples: {unknown_colors[:10].tolist()}"
        )

    return class_mask


# ============================================================
# Dataset
# ============================================================


class SegmentationPatchDataset(Dataset):
    """
    Semantic-segmentation Dataset using patch coordinates
    stored in a CSV file.

    Input modes
    -----------
    multispectral:
        Model input has 3 channels.

    early_fusion:
        Model input has 4 channels:
        3 multispectral + 1 DSM.

    separate:
        Multispectral and DSM tensors are returned separately,
        which can later be used for a dual-encoder model.
    """

    def __init__(
        self,
        patch_csv: str | Path,
        normalization_stats_path: str | Path,
        class_mapping_path: str | Path,
        input_mode: InputMode = "multispectral",
        project_root: str | Path = PROJECT_ROOT,
        strict_labels: bool = True,
    ) -> None:
        super().__init__()

        self.project_root = Path(project_root).resolve()

        self.patch_csv = resolve_project_path(
            self.project_root,
            patch_csv,
        )

        self.normalization_stats_path = resolve_project_path(
            self.project_root,
            normalization_stats_path,
        )

        self.class_mapping_path = resolve_project_path(
            self.project_root,
            class_mapping_path,
        )

        valid_modes = {
            "multispectral",
            "early_fusion",
            "separate",
        }

        if input_mode not in valid_modes:
            raise ValueError(
                f"Invalid input_mode: {input_mode}. "
                f"Choose from {sorted(valid_modes)}"
            )

        self.input_mode = input_mode
        self.strict_labels = strict_labels

        if not self.patch_csv.exists():
            raise FileNotFoundError(f"Patch CSV not found: {self.patch_csv}")

        self.patches = pd.read_csv(self.patch_csv)

        if self.patches.empty:
            raise RuntimeError(f"Patch CSV is empty: {self.patch_csv}")

        required_columns = {
            "patch_id",
            "split",
            "area_id",
            "row",
            "column",
            "patch_height",
            "patch_width",
            "multispectral_path",
            "dsm_path",
            "label_path",
        }

        missing_columns = required_columns - set(self.patches.columns)

        if missing_columns:
            raise KeyError(
                f"Patch CSV is missing columns: " f"{sorted(missing_columns)}"
            )

        self._load_normalization_statistics()
        self._load_class_mapping()

    def _load_normalization_statistics(
        self,
    ) -> None:
        """Load scaling and normalization parameters."""
        statistics = load_json(self.normalization_stats_path)

        multispectral_stats = statistics["multispectral"]

        dsm_stats = statistics["dsm"]

        self.multispectral_divisor = float(multispectral_stats["scaling_divisor"])

        self.dsm_divisor = float(dsm_stats["scaling_divisor"])

        self.multispectral_mean = np.asarray(
            multispectral_stats["mean"],
            dtype=np.float32,
        ).reshape(3, 1, 1)

        self.multispectral_std = np.asarray(
            multispectral_stats["std"],
            dtype=np.float32,
        ).reshape(3, 1, 1)

        if self.multispectral_mean.shape != (
            3,
            1,
            1,
        ):
            raise ValueError("Multispectral mean must contain " "exactly three values.")

        if self.multispectral_std.shape != (
            3,
            1,
            1,
        ):
            raise ValueError("Multispectral std must contain " "exactly three values.")

        if np.any(self.multispectral_std <= 0):
            raise ValueError(
                "Multispectral standard deviation " "must be greater than zero."
            )

        if self.multispectral_divisor <= 0:
            raise ValueError(
                "Multispectral scaling divisor " "must be greater than zero."
            )

        if self.dsm_divisor <= 0:
            raise ValueError("DSM scaling divisor must be " "greater than zero.")

    def _load_class_mapping(
        self,
    ) -> None:
        """Load class names and RGB colors."""
        mapping = load_json(self.class_mapping_path)

        self.number_of_classes = int(mapping["number_of_classes"])

        self.ignore_index = int(
            mapping.get(
                "ignore_index",
                255,
            )
        )

        classes = mapping["classes"]

        self.class_names = {
            int(class_information["id"]): (class_information["name"])
            for class_information in classes
        }

        self.class_colors = {
            int(class_information["id"]): tuple(
                int(value) for value in class_information["color"]
            )
            for class_information in classes
        }

        expected_ids = list(range(self.number_of_classes))

        if sorted(self.class_names) != expected_ids:
            raise ValueError("Class IDs must begin at zero " "and be consecutive.")

    def __len__(self) -> int:
        return len(self.patches)

    def _read_multispectral_patch(
        self,
        path: Path,
        row: int,
        column: int,
        height: int,
        width: int,
    ) -> np.ndarray:
        """
        Read, scale, and standardize a multispectral patch.

        Output shape:
            [3,H,W]
        """
        patch = read_raster_window(
            path=path,
            row=row,
            column=column,
            height=height,
            width=width,
            indexes=[1, 2, 3],
        )

        if patch.shape != (
            3,
            height,
            width,
        ):
            raise RuntimeError(
                f"Unexpected multispectral patch shape "
                f"from {path.name}: {patch.shape}"
            )

        patch = patch.astype(
            np.float32,
            copy=False,
        )

        # Step 1: scale multispectral values to [0,1].
        patch = patch / self.multispectral_divisor

        if float(np.nanmin(patch)) < -1e-6 or float(np.nanmax(patch)) > 1.0 + 1e-6:
            raise ValueError(
                f"Scaled multispectral values from "
                f"{path.name} are outside [0,1]. "
                f"Range: {np.nanmin(patch)} to "
                f"{np.nanmax(patch)}"
            )

        patch = np.clip(
            patch,
            0.0,
            1.0,
        )

        # Step 2: normalize multispectral channels
        # using training-only mean and standard deviation.
        patch = (patch - self.multispectral_mean) / self.multispectral_std

        return patch.astype(
            np.float32,
            copy=False,
        )

    def _read_dsm_patch(
        self,
        path: Path,
        row: int,
        column: int,
        height: int,
        width: int,
    ) -> np.ndarray:
        """
        Read and scale a DSM patch.

        Mean/std normalization is intentionally not applied.

        Output shape:
            [1,H,W]
        """
        if path.suffix.lower() in {
            ".tif",
            ".tiff",
        }:
            patch = read_raster_window(
                path=path,
                row=row,
                column=column,
                height=height,
                width=width,
                indexes=1,
            )

        else:
            patch = read_pillow_window(
                path=path,
                row=row,
                column=column,
                height=height,
                width=width,
                mode="L",
            )

        if patch.shape != (
            height,
            width,
        ):
            raise RuntimeError(
                f"Unexpected DSM patch shape " f"from {path.name}: {patch.shape}"
            )

        patch = patch.astype(
            np.float32,
            copy=False,
        )

        # Scale DSM to [0,1].
        patch = patch / self.dsm_divisor

        if float(np.nanmin(patch)) < -1e-6 or float(np.nanmax(patch)) > 1.0 + 1e-6:
            raise ValueError(
                f"Scaled DSM values from "
                f"{path.name} are outside [0,1]. "
                f"Range: {np.nanmin(patch)} to "
                f"{np.nanmax(patch)}"
            )

        patch = np.clip(
            patch,
            0.0,
            1.0,
        )

        # Add the channel dimension.
        patch = patch[None, :, :]

        return patch.astype(
            np.float32,
            copy=False,
        )

    def _read_label_patch(
        self,
        path: Path,
        row: int,
        column: int,
        height: int,
        width: int,
    ) -> np.ndarray:
        """
        Read an RGB label patch and convert it to class IDs.

        Output shape:
            [H,W]
        """
        if path.suffix.lower() in {
            ".tif",
            ".tiff",
        }:
            with rasterio.open(path) as source:
                if source.count < 3:
                    raise ValueError(
                        f"Label {path.name} contains "
                        f"{source.count} bands. "
                        "Expected at least three."
                    )

            patch = read_raster_window(
                path=path,
                row=row,
                column=column,
                height=height,
                width=width,
                indexes=[1, 2, 3],
            )

            patch = np.moveaxis(
                patch,
                0,
                -1,
            )

        else:
            patch = read_pillow_window(
                path=path,
                row=row,
                column=column,
                height=height,
                width=width,
                mode="RGB",
            )

        if patch.shape != (
            height,
            width,
            3,
        ):
            raise RuntimeError(
                f"Unexpected label patch shape " f"from {path.name}: {patch.shape}"
            )

        class_mask = rgb_to_class_mask(
            rgb_label=patch,
            class_colors=self.class_colors,
            ignore_index=self.ignore_index,
            strict=self.strict_labels,
        )

        return class_mask

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
        row_information = self.patches.iloc[index]

        row = int(row_information["row"])

        column = int(row_information["column"])

        patch_height = int(row_information["patch_height"])

        patch_width = int(row_information["patch_width"])

        multispectral_path = resolve_project_path(
            self.project_root,
            row_information["multispectral_path"],
        )

        dsm_path = resolve_project_path(
            self.project_root,
            row_information["dsm_path"],
        )

        label_path = resolve_project_path(
            self.project_root,
            row_information["label_path"],
        )

        multispectral = self._read_multispectral_patch(
            path=multispectral_path,
            row=row,
            column=column,
            height=patch_height,
            width=patch_width,
        )

        dsm = self._read_dsm_patch(
            path=dsm_path,
            row=row,
            column=column,
            height=patch_height,
            width=patch_width,
        )

        label = self._read_label_patch(
            path=label_path,
            row=row,
            column=column,
            height=patch_height,
            width=patch_width,
        )

        multispectral_tensor = torch.from_numpy(
            np.ascontiguousarray(multispectral)
        ).float()

        dsm_tensor = torch.from_numpy(np.ascontiguousarray(dsm)).float()

        label_tensor = torch.from_numpy(np.ascontiguousarray(label)).long()

        if self.input_mode == "multispectral":
            model_input = multispectral_tensor

        elif self.input_mode == "early_fusion":
            model_input = torch.cat(
                [
                    multispectral_tensor,
                    dsm_tensor,
                ],
                dim=0,
            )

        else:
            # The dual-encoder experiment will use
            # multispectral and DSM separately.
            model_input = multispectral_tensor

        sample = {
            "image": model_input,
            "multispectral": (multispectral_tensor),
            "dsm": dsm_tensor,
            "label": label_tensor,
            "patch_id": str(row_information["patch_id"]),
            "area_id": str(row_information["area_id"]),
            "split": str(row_information["split"]),
            "row": row,
            "column": column,
        }

        return sample


# ============================================================
# DataLoader factory
# ============================================================


def create_dataloaders(
    input_mode: InputMode,
    batch_size: int = 32,
    num_workers: int = 0,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, DataLoader]:
    """
    Create training, validation, and test DataLoaders.
    """
    project_root = Path(project_root).resolve()

    normalization_stats_path = project_root / "metadata" / "normalization_stats.json"

    class_mapping_path = project_root / "metadata" / "class_mapping.json"

    patch_files = {
        "train": (project_root / "metadata" / "train_patches.csv"),
        "validation": (project_root / "metadata" / "validation_patches.csv"),
        "test": (project_root / "metadata" / "test_patches.csv"),
    }

    datasets = {
        split_name: SegmentationPatchDataset(
            patch_csv=patch_path,
            normalization_stats_path=(normalization_stats_path),
            class_mapping_path=(class_mapping_path),
            input_mode=input_mode,
            project_root=project_root,
            strict_labels=True,
        )
        for split_name, patch_path in patch_files.items()
    }

    loader_arguments = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": False,
        "persistent_workers": (num_workers > 0),
    }

    dataloaders = {
        "train": DataLoader(
            datasets["train"],
            shuffle=True,
            drop_last=False,
            **loader_arguments,
        ),
        "validation": DataLoader(
            datasets["validation"],
            shuffle=False,
            drop_last=False,
            **loader_arguments,
        ),
        "test": DataLoader(
            datasets["test"],
            shuffle=False,
            drop_last=False,
            **loader_arguments,
        ),
    }

    return dataloaders
