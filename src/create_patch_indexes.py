"""Create CSV indices for fixed-size training, validation, and test patches.

Run: ``python src/create_patch_indexes.py``
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.patches import Rectangle
from PIL import Image
from rasterio.enums import Resampling
from rasterio.errors import NotGeoreferencedWarning


warnings.filterwarnings(
    "ignore",
    category=NotGeoreferencedWarning,
)


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MS_DIR = PROJECT_ROOT / "data" / "mspectral-images"

DSM_DIR = PROJECT_ROOT / "data" / "ndsm-images"

LABEL_DIR = PROJECT_ROOT / "data" / "labels"

TRAIN_AREAS_PATH = PROJECT_ROOT / "metadata" / "train_areas.txt"

VALIDATION_AREAS_PATH = PROJECT_ROOT / "metadata" / "validation_areas.txt"

TEST_AREAS_PATH = PROJECT_ROOT / "metadata" / "test_areas.txt"

AUDIT_PATH = PROJECT_ROOT / "metadata" / "data_audit.csv"

TRAIN_PATCHES_OUTPUT = PROJECT_ROOT / "metadata" / "train_patches.csv"

VALIDATION_PATCHES_OUTPUT = PROJECT_ROOT / "metadata" / "validation_patches.csv"

TEST_PATCHES_OUTPUT = PROJECT_ROOT / "metadata" / "test_patches.csv"

ALL_PATCHES_OUTPUT = PROJECT_ROOT / "metadata" / "all_patches.csv"

SUMMARY_OUTPUT = PROJECT_ROOT / "metadata" / "patch_summary.json"

GRID_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures" / "patch_grids"


MS_EXTENSIONS = {
    ".tif",
    ".tiff",
}

DSM_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
}

LABEL_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
}


# ============================================================
# Arguments
# ============================================================


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Create train, validation, and test patch-index files.")
    )

    parser.add_argument(
        "--patch-size",
        type=int,
        default=256,
        help="Patch width and height. Default: 256.",
    )

    parser.add_argument(
        "--train-stride",
        type=int,
        default=256,
        help="Training patch stride. Default: 256.",
    )

    parser.add_argument(
        "--validation-stride",
        type=int,
        default=256,
        help="Validation patch stride. Default: 256.",
    )

    parser.add_argument(
        "--test-stride",
        type=int,
        default=256,
        help="Test patch stride. Default: 256.",
    )

    parser.add_argument(
        "--max-preview-size",
        type=int,
        default=1200,
        help="Maximum preview width or height.",
    )

    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Do not create patch-grid figures.",
    )

    return parser.parse_args()


# ============================================================
# General utilities
# ============================================================


def extract_area_id(path: Path) -> str:
    match = re.search(
        r"area[\s_-]*(\d+)",
        path.stem,
        flags=re.IGNORECASE,
    )

    if match is None:
        raise ValueError(f"Could not extract area ID from {path.name}")

    return f"area{int(match.group(1))}"


def area_sort_key(area_id: str) -> int:
    match = re.search(
        r"\d+",
        area_id,
    )

    if match is None:
        return 10**9

    return int(match.group())


def read_area_file(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Area split file not found: {path}\n" "Run src/create_splits.py first."
        )

    area_ids = [
        line.strip().lower()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if not area_ids:
        raise RuntimeError(f"No area IDs found in {path}")

    return sorted(
        area_ids,
        key=area_sort_key,
    )


def index_folder(
    directory: Path,
    extensions: set[str],
) -> dict[str, Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    files: dict[str, Path] = {}

    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue

        if path.suffix.lower() not in extensions:
            continue

        try:
            area_id = extract_area_id(path)

        except ValueError as error:
            print(f"[WARNING] {error}")
            continue

        if area_id in files:
            raise RuntimeError(
                f"Duplicate files found for {area_id} " f"inside {directory}"
            )

        files[area_id] = path

    return files


def inspect_dimensions(
    path: Path,
) -> tuple[int, int, int]:
    """
    Returns
    -------
    width, height, number_of_bands
    """
    if path.suffix.lower() in {
        ".tif",
        ".tiff",
    }:
        with rasterio.open(path) as source:
            return (
                source.width,
                source.height,
                source.count,
            )

    with Image.open(path) as image:
        array = np.asarray(image)

        if array.ndim == 2:
            bands = 1

        elif array.ndim == 3:
            bands = array.shape[2]

        else:
            raise ValueError(
                f"Unsupported image shape for " f"{path.name}: {array.shape}"
            )

        return (
            image.width,
            image.height,
            bands,
        )


def save_json(
    data: dict[str, Any],
    path: Path,
) -> None:
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


# ============================================================
# Split safety checks
# ============================================================


def verify_no_area_overlap(
    train_areas: list[str],
    validation_areas: list[str],
    test_areas: list[str],
) -> None:
    train_set = set(train_areas)
    validation_set = set(validation_areas)
    test_set = set(test_areas)

    overlaps = {
        "train_validation": sorted(train_set & validation_set),
        "train_test": sorted(train_set & test_set),
        "validation_test": sorted(validation_set & test_set),
    }

    active_overlaps = {name: areas for name, areas in overlaps.items() if areas}

    if active_overlaps:
        raise RuntimeError("Dataset leakage detected: " f"{active_overlaps}")


def validate_audit_status(
    required_areas: list[str],
) -> None:
    """
    Confirm that all required areas are marked ready.

    The direct dimension checks below are still performed.
    """
    if not AUDIT_PATH.exists():
        print(
            "[WARNING] data_audit.csv was not found. "
            "Direct checks will still be performed."
        )
        return

    audit = pd.read_csv(AUDIT_PATH)

    required_columns = {
        "area_id",
        "status",
    }

    if not required_columns.issubset(audit.columns):
        print(
            "[WARNING] data_audit.csv does not contain " "area_id and status columns."
        )
        return

    audit["area_id"] = audit["area_id"].astype(str).str.lower()

    status_lookup = dict(
        zip(
            audit["area_id"],
            audit["status"].astype(str).str.lower(),
        )
    )

    problem_areas = {
        area_id: status_lookup.get(
            area_id,
            "not_found",
        )
        for area_id in required_areas
        if status_lookup.get(area_id) != "ready"
    }

    if problem_areas:
        raise RuntimeError(
            "The following areas are not marked ready "
            f"in data_audit.csv:\n{problem_areas}"
        )


# ============================================================
# Patch coordinate generation
# ============================================================


def generate_patch_positions(
    dimension: int,
    patch_size: int,
    stride: int,
) -> list[int]:
    """
    Generate valid patch start coordinates.

    The final coordinate is added separately when the regular
    stride does not land exactly on the image boundary.

    This guarantees that:
    - every patch has the full requested size;
    - the right and bottom borders are covered;
    - no zero-padding is required.
    """
    if patch_size <= 0:
        raise ValueError("Patch size must be greater than zero.")

    if stride <= 0:
        raise ValueError("Stride must be greater than zero.")

    if dimension < patch_size:
        raise ValueError(
            f"Image dimension {dimension} is smaller " f"than patch size {patch_size}."
        )

    final_start = dimension - patch_size

    positions = list(
        range(
            0,
            final_start + 1,
            stride,
        )
    )

    if not positions:
        positions = [0]

    if positions[-1] != final_start:
        positions.append(final_start)

    return sorted(set(positions))


def build_area_patch_records(
    split_name: str,
    area_id: str,
    multispectral_path: Path,
    dsm_path: Path,
    label_path: Path,
    patch_size: int,
    stride: int,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
]:
    ms_width, ms_height, ms_bands = inspect_dimensions(multispectral_path)

    dsm_width, dsm_height, dsm_bands = inspect_dimensions(dsm_path)

    (
        label_width,
        label_height,
        label_bands,
    ) = inspect_dimensions(label_path)

    dimensions_match = (
        ms_width == dsm_width == label_width and ms_height == dsm_height == label_height
    )

    if not dimensions_match:
        raise RuntimeError(
            f"Dimension mismatch for {area_id}:\n"
            f"  Multispectral: "
            f"{ms_width} × {ms_height}\n"
            f"  DSM: "
            f"{dsm_width} × {dsm_height}\n"
            f"  Label: "
            f"{label_width} × {label_height}"
        )

    if ms_bands != 3:
        raise RuntimeError(
            f"{area_id} multispectral image has " f"{ms_bands} bands. Expected 3."
        )

    if dsm_bands != 1:
        raise RuntimeError(f"{area_id} DSM has {dsm_bands} bands. " "Expected 1.")

    if label_bands < 3:
        raise RuntimeError(
            f"{area_id} label has {label_bands} bands. " "Expected an RGB label."
        )

    row_positions = generate_patch_positions(
        dimension=ms_height,
        patch_size=patch_size,
        stride=stride,
    )

    column_positions = generate_patch_positions(
        dimension=ms_width,
        patch_size=patch_size,
        stride=stride,
    )

    patch_records: list[dict[str, Any]] = []

    for row in row_positions:
        for column in column_positions:
            patch_id = f"{split_name}_{area_id}_" f"r{row:05d}_c{column:05d}"

            patch_records.append(
                {
                    "patch_id": patch_id,
                    "split": split_name,
                    "area_id": area_id,
                    "row": row,
                    "column": column,
                    "patch_height": patch_size,
                    "patch_width": patch_size,
                    "stride": stride,
                    "source_height": ms_height,
                    "source_width": ms_width,
                    "multispectral_path": str(
                        multispectral_path.relative_to(PROJECT_ROOT)
                    ),
                    "dsm_path": str(dsm_path.relative_to(PROJECT_ROOT)),
                    "label_path": str(label_path.relative_to(PROJECT_ROOT)),
                }
            )

    area_summary = {
        "area_id": area_id,
        "split": split_name,
        "source_width": ms_width,
        "source_height": ms_height,
        "patch_size": patch_size,
        "stride": stride,
        "row_positions": len(row_positions),
        "column_positions": len(column_positions),
        "number_of_patches": len(patch_records),
        "last_row_start": (row_positions[-1]),
        "last_column_start": (column_positions[-1]),
        "bottom_border_covered": (row_positions[-1] + patch_size == ms_height),
        "right_border_covered": (column_positions[-1] + patch_size == ms_width),
    }

    return (
        patch_records,
        area_summary,
    )


# ============================================================
# Patch-grid preview
# ============================================================


def calculate_preview_size(
    width: int,
    height: int,
    maximum_size: int,
) -> tuple[int, int]:
    scale = min(
        1.0,
        maximum_size
        / max(
            width,
            height,
        ),
    )

    return (
        max(
            1,
            int(round(width * scale)),
        ),
        max(
            1,
            int(round(height * scale)),
        ),
    )


def percentile_stretch(
    image: np.ndarray,
) -> np.ndarray:
    image = image.astype(np.float32)

    image = image[:, :, :3]

    output = np.zeros_like(
        image,
        dtype=np.float32,
    )

    for channel in range(3):
        band = image[:, :, channel]

        valid_values = band[np.isfinite(band)]

        if valid_values.size == 0:
            continue

        lower = np.percentile(
            valid_values,
            2,
        )

        upper = np.percentile(
            valid_values,
            98,
        )

        if upper <= lower:
            continue

        output[:, :, channel] = np.clip(
            (band - lower) / (upper - lower),
            0.0,
            1.0,
        )

    return output


def read_multispectral_preview(
    path: Path,
    preview_width: int,
    preview_height: int,
) -> np.ndarray:
    with rasterio.open(path) as source:
        image = source.read(
            indexes=[1, 2, 3],
            out_shape=(
                3,
                preview_height,
                preview_width,
            ),
            resampling=Resampling.bilinear,
        )

    image = np.moveaxis(
        image,
        0,
        -1,
    )

    return percentile_stretch(image)


def create_patch_grid_figure(
    area_summary: dict[str, Any],
    area_records: list[dict[str, Any]],
    multispectral_path: Path,
    maximum_preview_size: int,
) -> Path:
    source_width = int(area_summary["source_width"])

    source_height = int(area_summary["source_height"])

    (
        preview_width,
        preview_height,
    ) = calculate_preview_size(
        width=source_width,
        height=source_height,
        maximum_size=maximum_preview_size,
    )

    preview = read_multispectral_preview(
        path=multispectral_path,
        preview_width=preview_width,
        preview_height=preview_height,
    )

    horizontal_scale = preview_width / source_width

    vertical_scale = preview_height / source_height

    figure, axis = plt.subplots(figsize=(12, 10))

    axis.imshow(preview)

    for record in area_records:
        rectangle = Rectangle(
            (
                record["column"] * horizontal_scale,
                record["row"] * vertical_scale,
            ),
            record["patch_width"] * horizontal_scale,
            record["patch_height"] * vertical_scale,
            fill=False,
            linewidth=0.5,
        )

        axis.add_patch(rectangle)

    axis.set_title(
        f"{area_summary['area_id']} — "
        f"{area_summary['split']} patch grid\n"
        f"Original: "
        f"{source_width} × {source_height} | "
        f"Patch: "
        f"{area_summary['patch_size']} × "
        f"{area_summary['patch_size']} | "
        f"Stride: {area_summary['stride']} | "
        f"Patches: "
        f"{area_summary['number_of_patches']}"
    )

    axis.axis("off")

    GRID_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = GRID_OUTPUT_DIR / (
        f"{area_summary['split']}_" f"{area_summary['area_id']}_" f"patch_grid.png"
    )

    figure.tight_layout()

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
    arguments = parse_arguments()

    train_areas = read_area_file(TRAIN_AREAS_PATH)

    validation_areas = read_area_file(VALIDATION_AREAS_PATH)

    test_areas = read_area_file(TEST_AREAS_PATH)

    verify_no_area_overlap(
        train_areas=train_areas,
        validation_areas=validation_areas,
        test_areas=test_areas,
    )

    required_areas = sorted(
        set(train_areas) | set(validation_areas) | set(test_areas),
        key=area_sort_key,
    )

    validate_audit_status(required_areas)

    multispectral_files = index_folder(
        MS_DIR,
        MS_EXTENSIONS,
    )

    dsm_files = index_folder(
        DSM_DIR,
        DSM_EXTENSIONS,
    )

    label_files = index_folder(
        LABEL_DIR,
        LABEL_EXTENSIONS,
    )

    missing_files: dict[
        str,
        list[str],
    ] = {}

    for area_id in required_areas:
        missing_modalities = []

        if area_id not in multispectral_files:
            missing_modalities.append("multispectral")

        if area_id not in dsm_files:
            missing_modalities.append("DSM")

        if area_id not in label_files:
            missing_modalities.append("label")

        if missing_modalities:
            missing_files[area_id] = missing_modalities

    if missing_files:
        raise FileNotFoundError(f"Missing source files:\n" f"{missing_files}")

    split_configuration = {
        "train": {
            "areas": train_areas,
            "stride": (arguments.train_stride),
            "output": (TRAIN_PATCHES_OUTPUT),
        },
        "validation": {
            "areas": validation_areas,
            "stride": (arguments.validation_stride),
            "output": (VALIDATION_PATCHES_OUTPUT),
        },
        "test": {
            "areas": test_areas,
            "stride": (arguments.test_stride),
            "output": (TEST_PATCHES_OUTPUT),
        },
    }

    all_patch_records: list[dict[str, Any]] = []

    split_summaries: dict[
        str,
        Any,
    ] = {}

    print("=" * 80)
    print("PATCH INDEX CREATION")
    print("=" * 80)

    print(f"\nPatch size: " f"{arguments.patch_size} × " f"{arguments.patch_size}")

    for (
        split_name,
        configuration,
    ) in split_configuration.items():

        split_records: list[dict[str, Any]] = []

        area_summaries: list[dict[str, Any]] = []

        print(
            f"\nProcessing {split_name} split "
            f"with stride "
            f"{configuration['stride']}..."
        )

        for area_id in configuration["areas"]:
            (
                area_records,
                area_summary,
            ) = build_area_patch_records(
                split_name=split_name,
                area_id=area_id,
                multispectral_path=(multispectral_files[area_id]),
                dsm_path=(dsm_files[area_id]),
                label_path=(label_files[area_id]),
                patch_size=(arguments.patch_size),
                stride=(configuration["stride"]),
            )

            split_records.extend(area_records)

            area_summaries.append(area_summary)

            print(
                f"  {area_id}: "
                f"{area_summary['source_width']} × "
                f"{area_summary['source_height']} "
                f"→ "
                f"{area_summary['number_of_patches']} "
                f"patches"
            )

            if not arguments.no_preview:
                figure_path = create_patch_grid_figure(
                    area_summary=area_summary,
                    area_records=area_records,
                    multispectral_path=(multispectral_files[area_id]),
                    maximum_preview_size=(arguments.max_preview_size),
                )

                print(f"    Grid: " f"{figure_path}")

        split_dataframe = pd.DataFrame(split_records)

        output_path = configuration["output"]

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        split_dataframe.to_csv(
            output_path,
            index=False,
        )

        all_patch_records.extend(split_records)

        split_summaries[split_name] = {
            "areas": configuration["areas"],
            "number_of_areas": len(configuration["areas"]),
            "stride": configuration["stride"],
            "number_of_patches": len(split_records),
            "output_csv": str(output_path.relative_to(PROJECT_ROOT)),
            "area_summaries": (area_summaries),
        }

        print(f"  Saved " f"{len(split_records)} patches to:\n" f"  {output_path}")

    all_patches_dataframe = pd.DataFrame(all_patch_records)

    all_patches_dataframe.to_csv(
        ALL_PATCHES_OUTPUT,
        index=False,
    )

    summary = {
        "patch_size": (arguments.patch_size),
        "border_strategy": (
            "The final valid row and column starts are "
            "added when the regular stride does not exactly "
            "reach the image boundary. All patches remain "
            "full-sized; no padding is used."
        ),
        "total_number_of_patches": len(all_patch_records),
        "splits": split_summaries,
    }

    save_json(
        summary,
        SUMMARY_OUTPUT,
    )

    print("\n" + "=" * 80)
    print("PATCH SUMMARY")
    print("=" * 80)

    for split_name in [
        "train",
        "validation",
        "test",
    ]:
        split_information = split_summaries[split_name]

        print(
            f"{split_name.capitalize():10s}: "
            f"{split_information['number_of_areas']} "
            f"areas, "
            f"{split_information['number_of_patches']} "
            f"patches"
        )

    print(f"\nTotal patches: " f"{len(all_patch_records)}")

    print(f"\nCombined index:\n" f"{ALL_PATCHES_OUTPUT}")

    print(f"\nSummary:\n" f"{SUMMARY_OUTPUT}")

    if not arguments.no_preview:
        print(f"\nPatch-grid figures:\n" f"{GRID_OUTPUT_DIR}")

    print(
        "\nTrain, validation, and test patches remain "
        "separated by complete source areas."
    )


if __name__ == "__main__":
    main()
