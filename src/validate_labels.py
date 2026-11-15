"""Validate label colours and create class-distribution audit reports.

Run: ``python src/validate_labels.py``
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap
from PIL import Image


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "data" / "data-config.json"

LABEL_DIR = PROJECT_ROOT / "data" / "labels"

TRAIN_AREAS_PATH = PROJECT_ROOT / "metadata" / "train_areas.txt"

VALIDATION_AREAS_PATH = PROJECT_ROOT / "metadata" / "validation_areas.txt"

TEST_AREAS_PATH = PROJECT_ROOT / "metadata" / "test_areas.txt"

LABEL_AUDIT_OUTPUT = PROJECT_ROOT / "metadata" / "label_audit.csv"

CLASS_COUNTS_OUTPUT = PROJECT_ROOT / "metadata" / "class_pixel_counts.csv"

CLASS_MAPPING_OUTPUT = PROJECT_ROOT / "metadata" / "class_mapping.json"

FIGURE_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures" / "label_validation"


LABEL_EXTENSIONS = {
    ".tif",
    ".tiff",
    ".png",
    ".jpg",
    ".jpeg",
}


# Pixels with an unknown color will receive this value.
IGNORE_INDEX = 255


# ============================================================
# Utility functions
# ============================================================


def extract_area_id(path: Path) -> str:
    """Extract area ID such as area11 from a filename."""
    match = re.search(
        r"area[\s_-]*(\d+)",
        path.stem,
        flags=re.IGNORECASE,
    )

    if match is None:
        raise ValueError(f"Could not extract area ID from {path.name}")

    return f"area{int(match.group(1))}"


def area_sort_key(area_id: str) -> int:
    """Sort area IDs numerically."""
    match = re.search(r"\d+", area_id)

    if match is None:
        return 10**9

    return int(match.group())


def load_json(path: Path) -> dict[str, Any]:
    """Load JSON content."""
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def save_json(
    data: dict[str, Any],
    path: Path,
) -> None:
    """Save JSON content."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=4,
        )


def read_area_file(path: Path) -> list[str]:
    """Read one area ID per line."""
    if not path.exists():
        raise FileNotFoundError(f"Area-list file not found: {path}")

    area_ids = [
        line.strip().lower()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    return sorted(
        area_ids,
        key=area_sort_key,
    )


def index_label_files(
    directory: Path,
) -> dict[str, Path]:
    """Index label images by area ID."""
    if not directory.exists():
        raise FileNotFoundError(f"Label directory not found: {directory}")

    indexed_files: dict[str, Path] = {}

    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue

        if path.suffix.lower() not in LABEL_EXTENSIONS:
            continue

        try:
            area_id = extract_area_id(path)

        except ValueError as error:
            print(f"[WARNING] {error}")
            continue

        if area_id in indexed_files:
            raise RuntimeError(
                f"Duplicate label files found for {area_id}: "
                f"{indexed_files[area_id].name} and {path.name}"
            )

        indexed_files[area_id] = path

    return indexed_files


# ============================================================
# Class configuration
# ============================================================


def load_class_information(
    config_path: Path,
) -> tuple[
    dict[int, str],
    dict[int, tuple[int, int, int]],
]:
    """Read class names and RGB values from data-config.json."""
    config = load_json(config_path)

    class_names_raw = config.get("Class Names")

    class_colors_raw = config.get("Class Color Codes")

    if class_names_raw is None:
        raise KeyError("Missing 'Class Names' in data-config.json")

    if class_colors_raw is None:
        raise KeyError("Missing 'Class Color Codes' in data-config.json")

    class_names = {
        int(class_id): str(class_name)
        for class_id, class_name in class_names_raw.items()
    }

    class_colors = {
        int(class_id): tuple(int(value) for value in color)
        for class_id, color in class_colors_raw.items()
    }

    expected_ids = list(range(len(class_names)))

    if sorted(class_names) != expected_ids:
        raise ValueError("Class IDs must begin at 0 and be consecutive.")

    if sorted(class_colors) != expected_ids:
        raise ValueError("Every class must have a color code.")

    for class_id, color in class_colors.items():
        if len(color) != 3:
            raise ValueError(
                f"Class {class_id} color must contain "
                f"three RGB values. Received: {color}"
            )

        if any(value < 0 or value > 255 for value in color):
            raise ValueError(f"Invalid RGB color for class {class_id}: {color}")

    return class_names, class_colors


# ============================================================
# Label reading
# ============================================================


def read_label_image(path: Path) -> np.ndarray:
    """
    Read a complete label image.

    Returns
    -------
    np.ndarray
        RGB label with shape [height, width, 3].
    """
    if path.suffix.lower() in {
        ".tif",
        ".tiff",
    }:
        with rasterio.open(path) as source:
            label = source.read()

        label = np.moveaxis(
            label,
            0,
            -1,
        )

    else:
        with Image.open(path) as image:
            label = np.asarray(image)

    if label.ndim == 2:
        raise ValueError(
            f"{path.name} is single-band with shape "
            f"{label.shape}. Expected an RGB label."
        )

    if label.ndim != 3:
        raise ValueError(f"Unexpected label shape for {path.name}: " f"{label.shape}")

    if label.shape[2] < 3:
        raise ValueError(
            f"{path.name} contains only {label.shape[2]} "
            "channels. Expected at least three."
        )

    # Ignore alpha or other additional channels.
    label = label[:, :, :3]

    return label


# ============================================================
# RGB to class-mask conversion
# ============================================================


def rgb_to_class_mask(
    rgb_label: np.ndarray,
    class_colors: dict[
        int,
        tuple[int, int, int],
    ],
    ignore_index: int = IGNORE_INDEX,
) -> np.ndarray:
    """
    Convert an RGB semantic label into integer class IDs.

    Output shape:
        [height, width]

    Output values:
        0, 1, ..., number_of_classes - 1

    Unknown RGB pixels:
        ignore_index
    """
    if rgb_label.ndim != 3:
        raise ValueError(f"Expected RGB label, received {rgb_label.shape}")

    if rgb_label.shape[2] < 3:
        raise ValueError(
            f"Expected at least three channels, " f"received {rgb_label.shape}"
        )

    rgb_label = rgb_label[:, :, :3]

    class_mask = np.full(
        rgb_label.shape[:2],
        fill_value=ignore_index,
        dtype=np.uint8,
    )

    for class_id, color in class_colors.items():
        color_array = np.asarray(
            color,
            dtype=rgb_label.dtype,
        )

        matches = np.all(
            rgb_label == color_array,
            axis=-1,
        )

        class_mask[matches] = class_id

    return class_mask


# ============================================================
# Label-color audit
# ============================================================


def count_rgb_colors(
    rgb_label: np.ndarray,
) -> dict[tuple[int, int, int], int]:
    """Count all unique RGB colors in a label image."""
    pixels = rgb_label.reshape(
        -1,
        3,
    )

    unique_colors, counts = np.unique(
        pixels,
        axis=0,
        return_counts=True,
    )

    return {
        tuple(int(value) for value in color): int(count)
        for color, count in zip(unique_colors, counts)
    }


def audit_label(
    area_id: str,
    split_name: str,
    label_path: Path,
    class_names: dict[int, str],
    class_colors: dict[
        int,
        tuple[int, int, int],
    ],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    np.ndarray,
    np.ndarray,
]:
    """
    Inspect one complete label image.

    Returns
    -------
    audit_record:
        One area-level summary row.

    class_records:
        One row per area and class.

    rgb_label:
        Original RGB label.

    class_mask:
        Converted integer class mask.
    """
    rgb_label = read_label_image(label_path)

    color_counts = count_rgb_colors(rgb_label)

    expected_colors = set(class_colors.values())

    observed_colors = set(color_counts)

    unexpected_colors = sorted(observed_colors - expected_colors)

    missing_colors = sorted(expected_colors - observed_colors)

    class_mask = rgb_to_class_mask(
        rgb_label=rgb_label,
        class_colors=class_colors,
        ignore_index=IGNORE_INDEX,
    )

    total_pixels = int(rgb_label.shape[0] * rgb_label.shape[1])

    unknown_pixel_count = int(np.sum(class_mask == IGNORE_INDEX))

    valid_pixel_count = total_pixels - unknown_pixel_count

    audit_record = {
        "area_id": area_id,
        "split": split_name,
        "label_path": str(label_path.relative_to(PROJECT_ROOT)),
        "width": rgb_label.shape[1],
        "height": rgb_label.shape[0],
        "channels": rgb_label.shape[2],
        "dtype": str(rgb_label.dtype),
        "total_pixels": total_pixels,
        "valid_pixels": valid_pixel_count,
        "unknown_pixels": unknown_pixel_count,
        "unknown_percentage": (100.0 * unknown_pixel_count / total_pixels),
        "number_of_observed_colors": len(observed_colors),
        "number_of_expected_colors": len(expected_colors),
        "unexpected_colors": str(unexpected_colors),
        "missing_expected_colors": str(missing_colors),
        "status": ("ready" if unknown_pixel_count == 0 else "unexpected_label_colors"),
    }

    class_records: list[dict[str, Any]] = []

    for class_id in sorted(class_names):
        color = class_colors[class_id]

        pixel_count = int(
            color_counts.get(
                color,
                0,
            )
        )

        percentage = 100.0 * pixel_count / total_pixels

        class_records.append(
            {
                "area_id": area_id,
                "split": split_name,
                "class_id": class_id,
                "class_name": (class_names[class_id]),
                "red": color[0],
                "green": color[1],
                "blue": color[2],
                "pixel_count": pixel_count,
                "percentage": percentage,
                "class_present": (pixel_count > 0),
            }
        )

    return (
        audit_record,
        class_records,
        rgb_label,
        class_mask,
    )


# ============================================================
# Visualization
# ============================================================


def create_label_figure(
    area_id: str,
    split_name: str,
    rgb_label: np.ndarray,
    class_mask: np.ndarray,
    class_names: dict[int, str],
    class_colors: dict[
        int,
        tuple[int, int, int],
    ],
) -> Path:
    """Save RGB-label and class-index visualizations."""
    FIGURE_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    normalized_colors = [
        np.asarray(
            class_colors[class_id],
            dtype=np.float32,
        )
        / 255.0
        for class_id in sorted(class_names)
    ]

    class_colormap = ListedColormap(normalized_colors)

    valid_mask_for_display = class_mask.astype(np.float32)

    valid_mask_for_display[class_mask == IGNORE_INDEX] = np.nan

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(15, 7),
    )

    axes[0].imshow(rgb_label)

    axes[0].set_title(
        f"Original RGB label\n"
        f"{rgb_label.shape[1]} × "
        f"{rgb_label.shape[0]} pixels"
    )

    axes[0].axis("off")

    axes[1].imshow(
        valid_mask_for_display,
        cmap=class_colormap,
        vmin=0,
        vmax=len(class_names) - 1,
        interpolation="nearest",
    )

    axes[1].set_title("Converted integer class mask")

    axes[1].axis("off")

    legend_handles = []

    for class_id in sorted(class_names):
        color = (
            np.asarray(
                class_colors[class_id],
                dtype=np.float32,
            )
            / 255.0
        )

        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="s",
                linestyle="",
                markersize=10,
                markerfacecolor=color,
                markeredgecolor="black",
                label=(f"{class_id}: " f"{class_names[class_id]}"),
            )
        )

    figure.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.02),
    )

    figure.suptitle(
        f"{area_id} — {split_name}",
        fontsize=15,
    )

    figure.tight_layout(rect=(0, 0.08, 1, 0.95))

    output_path = FIGURE_OUTPUT_DIR / f"{area_id}_{split_name}_label_validation.png"

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


# ============================================================
# Main
# ============================================================


def main() -> None:
    class_names, class_colors = load_class_information(CONFIG_PATH)

    train_areas = read_area_file(TRAIN_AREAS_PATH)

    validation_areas = read_area_file(VALIDATION_AREAS_PATH)

    test_areas = read_area_file(TEST_AREAS_PATH)

    split_areas = {
        "train": train_areas,
        "validation": validation_areas,
        "test": test_areas,
    }

    label_files = index_label_files(LABEL_DIR)

    all_required_areas = train_areas + validation_areas + test_areas

    missing_labels = [
        area_id for area_id in all_required_areas if area_id not in label_files
    ]

    if missing_labels:
        raise FileNotFoundError(f"Missing label files for: {missing_labels}")

    audit_records: list[dict[str, Any]] = []

    class_records: list[dict[str, Any]] = []

    print("=" * 80)
    print("LABEL-COLOR VALIDATION")
    print("=" * 80)

    for split_name, area_ids in split_areas.items():
        print(f"\nProcessing {split_name} labels...")

        for area_id in area_ids:
            label_path = label_files[area_id]

            print(f"  Checking {area_id}: " f"{label_path.name}")

            (
                audit_record,
                area_class_records,
                rgb_label,
                class_mask,
            ) = audit_label(
                area_id=area_id,
                split_name=split_name,
                label_path=label_path,
                class_names=class_names,
                class_colors=class_colors,
            )

            audit_records.append(audit_record)

            class_records.extend(area_class_records)

            figure_path = create_label_figure(
                area_id=area_id,
                split_name=split_name,
                rgb_label=rgb_label,
                class_mask=class_mask,
                class_names=class_names,
                class_colors=class_colors,
            )

            print(f"    Status: " f"{audit_record['status']}")

            print(f"    Unknown pixels: " f"{audit_record['unknown_pixels']}")

            print(f"    Figure: {figure_path}")

    audit_dataframe = pd.DataFrame(audit_records)

    class_dataframe = pd.DataFrame(class_records)

    audit_dataframe.to_csv(
        LABEL_AUDIT_OUTPUT,
        index=False,
    )

    class_dataframe.to_csv(
        CLASS_COUNTS_OUTPUT,
        index=False,
    )

    class_mapping = {
        "number_of_classes": len(class_names),
        "ignore_index": IGNORE_INDEX,
        "classes": [
            {
                "id": class_id,
                "name": class_names[class_id],
                "color": list(class_colors[class_id]),
            }
            for class_id in sorted(class_names)
        ],
        "color_to_class": {
            ",".join(str(value) for value in class_colors[class_id]): class_id
            for class_id in sorted(class_names)
        },
    }

    save_json(
        class_mapping,
        CLASS_MAPPING_OUTPUT,
    )

    print("\n" + "=" * 80)
    print("LABEL VALIDATION SUMMARY")
    print("=" * 80)

    print(
        audit_dataframe[
            [
                "area_id",
                "split",
                "number_of_observed_colors",
                "unknown_pixels",
                "status",
            ]
        ].to_string(index=False)
    )

    problem_areas = audit_dataframe[audit_dataframe["status"] != "ready"]

    print(f"\nTotal labels checked: " f"{len(audit_dataframe)}")

    print(f"Ready labels: " f"{len(audit_dataframe) - len(problem_areas)}")

    print(f"Problem labels: " f"{len(problem_areas)}")

    print(f"\nLabel audit saved to:\n" f"{LABEL_AUDIT_OUTPUT}")

    print(f"\nClass counts saved to:\n" f"{CLASS_COUNTS_OUTPUT}")

    print(f"\nClass mapping saved to:\n" f"{CLASS_MAPPING_OUTPUT}")

    print(f"\nValidation figures saved to:\n" f"{FIGURE_OUTPUT_DIR}")

    if not problem_areas.empty:
        print(
            "\nWARNING: Some labels contain unexpected "
            "RGB colors. Do not create patches until they "
            "have been investigated."
        )
    else:
        print(
            "\nAll label pixels match the six configured "
            "class colors. You can proceed to patch indexing."
        )


if __name__ == "__main__":
    main()
