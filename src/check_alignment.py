"""Verify spatial alignment between images, depth maps, and labels.

Run: ``python src/check_alignment.py``
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.errors import NotGeoreferencedWarning


warnings.filterwarnings(
    "ignore",
    category=NotGeoreferencedWarning,
)


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

# This file is expected at:
# EX_2_CNN_with_DSM/src/check_alignment.py
#
# parents[1] therefore points to:
# EX_2_CNN_with_DSM/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MS_DIR = PROJECT_ROOT / "data" / "mspectral-images"
DSM_DIR = PROJECT_ROOT / "data" / "ndsm-images"
LABEL_DIR = PROJECT_ROOT / "data" / "labels"

AUDIT_OUTPUT = PROJECT_ROOT / "metadata" / "alignment_audit.csv"

FIGURE_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures" / "alignment_checks"


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
    ".tif",
    ".tiff",
    ".png",
    ".jpg",
    ".jpeg",
}


# ---------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit multispectral, DSM, and label alignment "
            "and create visual inspection figures."
        )
    )

    parser.add_argument(
        "--areas",
        nargs="*",
        default=None,
        help=(
            "Optional area IDs to process. Example: "
            "--areas area1 area10. "
            "By default, all discovered areas are processed."
        ),
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help=("Display every figure interactively in addition " "to saving it."),
    )

    parser.add_argument(
        "--max-display-size",
        type=int,
        default=1200,
        help=(
            "Maximum width or height used for visualization. "
            "Original files are not modified."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# File matching
# ---------------------------------------------------------------------


def extract_area_id(path: Path) -> str:
    """
    Extract an area ID from a filename.

    Examples
    --------
    top_mosaic_09cm_area10.tif
        -> area10

    dsm_09cm_matching_area10_normalized.jpg
        -> area10
    """
    match = re.search(
        r"area[\s_-]*(\d+)",
        path.stem,
        flags=re.IGNORECASE,
    )

    if match is None:
        raise ValueError(f"Could not extract area ID from: {path.name}")

    area_number = int(match.group(1))

    return f"area{area_number}"


def area_sort_key(area_id: str) -> int:
    """Sort area IDs numerically rather than alphabetically."""
    match = re.search(r"\d+", area_id)

    if match is None:
        return 10**9

    return int(match.group())


def index_folder(
    folder: Path,
    allowed_extensions: set[str],
) -> tuple[dict[str, Path], list[str]]:
    """
    Index all valid files in a directory by area ID.

    Returns
    -------
    indexed_files:
        Mapping from area ID to file path.

    problems:
        Invalid filenames or duplicate area IDs.
    """
    indexed_files: dict[str, Path] = {}
    problems: list[str] = []

    if not folder.exists():
        return (
            indexed_files,
            [f"Folder does not exist: {folder}"],
        )

    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue

        if path.suffix.lower() not in allowed_extensions:
            continue

        try:
            area_id = extract_area_id(path)

        except ValueError as error:
            problems.append(str(error))
            continue

        if area_id in indexed_files:
            problems.append(
                f"Duplicate {area_id} in {folder.name}: "
                f"{indexed_files[area_id].name} and "
                f"{path.name}"
            )
            continue

        indexed_files[area_id] = path

    return indexed_files, problems


# ---------------------------------------------------------------------
# Image sizes and metadata
# ---------------------------------------------------------------------


def compute_display_shape(
    width: int,
    height: int,
    max_display_size: int,
) -> tuple[int, int]:
    """
    Calculate a reduced display size while preserving aspect ratio.
    """
    scale = min(
        1.0,
        max_display_size / max(width, height),
    )

    output_width = max(
        1,
        int(round(width * scale)),
    )

    output_height = max(
        1,
        int(round(height * scale)),
    )

    return output_width, output_height


def has_meaningful_transform(transform: Any) -> bool:
    """
    Return False when the raster transform is only the identity matrix.

    An identity transform usually means meaningful georeferencing is absent.
    """
    identity_transform = (
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )

    try:
        values = tuple(float(value) for value in transform)

    except TypeError:
        return False

    return not np.allclose(
        values,
        identity_transform,
    )


def inspect_raster(path: Path) -> dict[str, Any]:
    """
    Inspect a TIFF or another raster readable by Rasterio.

    A reduced sample is used to estimate minimum and maximum values.
    """
    with rasterio.open(path) as source:
        sample_width, sample_height = compute_display_shape(
            width=source.width,
            height=source.height,
            max_display_size=1024,
        )

        sample = source.read(
            out_shape=(
                source.count,
                sample_height,
                sample_width,
            ),
            resampling=Resampling.nearest,
        ).astype(np.float64)

        finite_values = sample[np.isfinite(sample)]

        if finite_values.size:
            minimum = float(finite_values.min())
            maximum = float(finite_values.max())
        else:
            minimum = np.nan
            maximum = np.nan

        transform = source.transform

        transform_available = has_meaningful_transform(transform)

        return {
            "width": source.width,
            "height": source.height,
            "bands": source.count,
            "dtype": str(source.dtypes[0]),
            "minimum": minimum,
            "maximum": maximum,
            "crs": source.crs,
            "crs_text": (str(source.crs) if source.crs else ""),
            "has_crs": source.crs is not None,
            "transform": transform,
            "transform_text": (str(transform) if transform_available else ""),
            "has_transform": transform_available,
            "nodata": source.nodata,
        }


def inspect_standard_image(
    path: Path,
) -> dict[str, Any]:
    """
    Inspect JPG, JPEG, or PNG files using Pillow.
    """
    with Image.open(path) as image:
        array = np.asarray(image)

        if array.ndim == 2:
            bands = 1

        elif array.ndim == 3:
            bands = array.shape[2]

        else:
            bands = None

        finite_values = array[np.isfinite(array)]

        if finite_values.size:
            minimum = float(finite_values.min())
            maximum = float(finite_values.max())
        else:
            minimum = np.nan
            maximum = np.nan

        return {
            "width": image.width,
            "height": image.height,
            "bands": bands,
            "dtype": str(array.dtype),
            "minimum": minimum,
            "maximum": maximum,
            "crs": None,
            "crs_text": "",
            "has_crs": False,
            "transform": None,
            "transform_text": "",
            "has_transform": False,
            "nodata": None,
        }


def inspect_file(path: Path) -> dict[str, Any]:
    """Choose Rasterio or Pillow based on the file extension."""
    if path.suffix.lower() in {
        ".tif",
        ".tiff",
    }:
        return inspect_raster(path)

    return inspect_standard_image(path)


# ---------------------------------------------------------------------
# Reading images for visualization
# ---------------------------------------------------------------------


def load_display_array(
    path: Path,
    target_width: int | None,
    target_height: int | None,
    max_display_size: int,
    is_label: bool = False,
) -> np.ndarray:
    """
    Read an image at a reduced display resolution.

    Nearest-neighbour resampling is used for labels.
    Bilinear resampling is used for imagery and DSM data.
    """
    if path.suffix.lower() in {
        ".tif",
        ".tiff",
    }:
        with rasterio.open(path) as source:
            if target_width is None or target_height is None:
                target_width, target_height = compute_display_shape(
                    source.width,
                    source.height,
                    max_display_size,
                )

            resampling_method = Resampling.nearest if is_label else Resampling.bilinear

            array = source.read(
                out_shape=(
                    source.count,
                    target_height,
                    target_width,
                ),
                resampling=resampling_method,
            )

        return np.moveaxis(
            array,
            0,
            -1,
        )

    with Image.open(path) as image:
        if target_width is None or target_height is None:
            target_width, target_height = compute_display_shape(
                image.width,
                image.height,
                max_display_size,
            )

        resampling_method = (
            Image.Resampling.NEAREST if is_label else Image.Resampling.BILINEAR
        )

        resized_image = image.resize(
            (
                target_width,
                target_height,
            ),
            resample=resampling_method,
        )

        return np.asarray(resized_image)


# ---------------------------------------------------------------------
# Visualization preparation
# ---------------------------------------------------------------------


def percentile_stretch(
    image: np.ndarray,
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0,
) -> np.ndarray:
    """
    Stretch a three-channel image to a visible [0,1] range.

    This transformation is only for visualization. It does not modify
    training data.
    """
    image = image.astype(np.float32)

    if image.ndim == 2:
        image = image[..., None]

    if image.shape[2] == 1:
        image = np.repeat(
            image,
            3,
            axis=2,
        )

    elif image.shape[2] >= 3:
        image = image[:, :, :3]

    else:
        missing_channels = 3 - image.shape[2]

        image = np.pad(
            image,
            (
                (0, 0),
                (0, 0),
                (0, missing_channels),
            ),
            mode="edge",
        )

    stretched = np.zeros_like(
        image,
        dtype=np.float32,
    )

    for channel in range(3):
        band = image[:, :, channel]

        finite_values = band[np.isfinite(band)]

        if finite_values.size == 0:
            continue

        lower = np.percentile(
            finite_values,
            lower_percentile,
        )

        upper = np.percentile(
            finite_values,
            upper_percentile,
        )

        if upper <= lower:
            stretched[:, :, channel] = 0.0
            continue

        stretched[:, :, channel] = np.clip(
            (band - lower) / (upper - lower),
            0.0,
            1.0,
        )

    return stretched


def prepare_dsm_for_display(
    dsm: np.ndarray,
) -> np.ndarray:
    """
    Convert the DSM into a visible grayscale image.

    This stretching is for visualization only.
    """
    if dsm.ndim == 3:
        dsm = dsm[:, :, 0]

    dsm = dsm.astype(np.float32)

    finite_values = dsm[np.isfinite(dsm)]

    if finite_values.size == 0:
        return np.zeros_like(
            dsm,
            dtype=np.float32,
        )

    lower = np.percentile(
        finite_values,
        2,
    )

    upper = np.percentile(
        finite_values,
        98,
    )

    if upper <= lower:
        return np.zeros_like(
            dsm,
            dtype=np.float32,
        )

    return np.clip(
        (dsm - lower) / (upper - lower),
        0.0,
        1.0,
    )


def prepare_label_for_display(
    label: np.ndarray,
) -> np.ndarray:
    """Prepare a label image for Matplotlib."""
    if label.ndim == 2:
        return label

    if label.shape[2] == 1:
        return label[:, :, 0]

    return label[:, :, :3]


def compute_label_boundary(
    label: np.ndarray,
) -> np.ndarray:
    """
    Find pixels where the class or RGB label changes.

    The resulting boundary is useful for checking image-label alignment.
    """
    if label.ndim == 3 and label.shape[2] >= 3:
        comparable = label[:, :, :3]

        horizontal_difference = np.any(
            comparable[:, 1:, :] != comparable[:, :-1, :],
            axis=2,
        )

        vertical_difference = np.any(
            comparable[1:, :, :] != comparable[:-1, :, :],
            axis=2,
        )

    else:
        if label.ndim == 3:
            comparable = label[:, :, 0]
        else:
            comparable = label

        horizontal_difference = comparable[:, 1:] != comparable[:, :-1]

        vertical_difference = comparable[1:, :] != comparable[:-1, :]

    boundary = np.zeros(
        comparable.shape[:2],
        dtype=bool,
    )

    boundary[:, 1:] |= horizontal_difference

    boundary[1:, :] |= vertical_difference

    return boundary


# ---------------------------------------------------------------------
# CRS and transform comparison
# ---------------------------------------------------------------------


def crs_matches(
    first_info: dict[str, Any],
    second_info: dict[str, Any],
) -> bool | None:
    """
    Compare CRS values.

    Returns None when one or both files do not have CRS information.
    """
    if not first_info["has_crs"] or not second_info["has_crs"]:
        return None

    return first_info["crs"] == second_info["crs"]


def transform_matches(
    first_info: dict[str, Any],
    second_info: dict[str, Any],
) -> bool | None:
    """
    Compare affine transforms.

    Returns None when one or both files do not have a meaningful transform.
    """
    if not first_info["has_transform"] or not second_info["has_transform"]:
        return None

    first_values = np.asarray(
        tuple(first_info["transform"]),
        dtype=float,
    )

    second_values = np.asarray(
        tuple(second_info["transform"]),
        dtype=float,
    )

    return bool(
        np.allclose(
            first_values,
            second_values,
            rtol=1e-9,
            atol=1e-9,
        )
    )


def metadata_result_text(
    value: bool | None,
) -> str:
    """Convert metadata comparison results into readable text."""
    if value is True:
        return "match"

    if value is False:
        return "mismatch"

    return "not comparable"


# ---------------------------------------------------------------------
# Audit record
# ---------------------------------------------------------------------


def build_record(
    area_id: str,
    ms_path: Path | None,
    dsm_path: Path | None,
    label_path: Path | None,
) -> dict[str, Any]:
    """Create one audit record for one area."""
    record: dict[str, Any] = {
        "area_id": area_id,
        "multispectral_path": (
            str(ms_path.relative_to(PROJECT_ROOT)) if ms_path else ""
        ),
        "dsm_path": (str(dsm_path.relative_to(PROJECT_ROOT)) if dsm_path else ""),
        "label_path": (str(label_path.relative_to(PROJECT_ROOT)) if label_path else ""),
        "multispectral_exists": (ms_path is not None),
        "dsm_exists": (dsm_path is not None),
        "label_exists": (label_path is not None),
        "dimensions_match": False,
        "ms_label_crs_match": "",
        "ms_label_transform_match": "",
        "status": "",
        "notes": "",
    }

    missing_files = []

    if ms_path is None:
        missing_files.append("multispectral")

    if dsm_path is None:
        missing_files.append("DSM")

    if label_path is None:
        missing_files.append("label")

    if missing_files:
        record["status"] = "missing_" + "_and_".join(missing_files)

        return record

    try:
        ms_info = inspect_file(ms_path)
        dsm_info = inspect_file(dsm_path)
        label_info = inspect_file(label_path)

        info_by_type = {
            "ms": ms_info,
            "dsm": dsm_info,
            "label": label_info,
        }

        for prefix, info in info_by_type.items():
            record[f"{prefix}_width"] = info["width"]

            record[f"{prefix}_height"] = info["height"]

            record[f"{prefix}_bands"] = info["bands"]

            record[f"{prefix}_dtype"] = info["dtype"]

            record[f"{prefix}_min"] = info["minimum"]

            record[f"{prefix}_max"] = info["maximum"]

            record[f"{prefix}_has_crs"] = info["has_crs"]

            record[f"{prefix}_crs"] = info["crs_text"]

            record[f"{prefix}_has_transform"] = info["has_transform"]

            record[f"{prefix}_transform"] = info["transform_text"]

            record[f"{prefix}_nodata"] = info["nodata"]

        dimensions_match = (
            ms_info["width"] == dsm_info["width"] == label_info["width"]
            and ms_info["height"] == dsm_info["height"] == label_info["height"]
        )

        record["dimensions_match"] = dimensions_match

        ms_label_crs_match = crs_matches(
            ms_info,
            label_info,
        )

        ms_label_transform_match = transform_matches(
            ms_info,
            label_info,
        )

        record["ms_label_crs_match"] = metadata_result_text(ms_label_crs_match)

        record["ms_label_transform_match"] = metadata_result_text(
            ms_label_transform_match
        )

        notes = []

        if not ms_info["has_crs"]:
            notes.append("multispectral CRS missing")

        if not dsm_info["has_crs"]:
            notes.append("DSM CRS missing")

        if not label_info["has_crs"]:
            notes.append("label CRS missing")

        if not ms_info["has_transform"]:
            notes.append("multispectral transform missing")

        if not dsm_info["has_transform"]:
            notes.append("DSM transform missing")

        if not label_info["has_transform"]:
            notes.append("label transform missing")

        if not dimensions_match:
            record["status"] = "dimension_mismatch"

        elif ms_label_crs_match is False:
            record["status"] = "crs_mismatch"

        elif ms_label_transform_match is False:
            record["status"] = "transform_mismatch"

        elif ms_info["bands"] != 3:
            record["status"] = "unexpected_multispectral_bands_" f"{ms_info['bands']}"

        elif dsm_info["bands"] != 1:
            record["status"] = "unexpected_dsm_bands_" f"{dsm_info['bands']}"

        else:
            record["status"] = "ready"

        if dimensions_match and (
            ms_label_crs_match is None or ms_label_transform_match is None
        ):
            notes.append("pixel alignment must be " "confirmed visually")

        record["notes"] = "; ".join(notes)

    except Exception as error:
        record["status"] = "inspection_error"

        record["notes"] = str(error)

    return record


# ---------------------------------------------------------------------
# Metadata panel
# ---------------------------------------------------------------------


def add_metadata_panel(
    axis: plt.Axes,
    area_id: str,
    record: dict[str, Any],
) -> None:
    """Add an audit summary to the final subplot."""
    axis.axis("off")

    text_lines = [
        f"Area: {area_id}",
        f"Status: {record.get('status', '')}",
        "",
        (
            "Dimensions:"
            f" MS={record.get('ms_width', '?')}"
            f"×{record.get('ms_height', '?')}"
        ),
        (f" DSM={record.get('dsm_width', '?')}" f"×{record.get('dsm_height', '?')}"),
        (
            f" Label={record.get('label_width', '?')}"
            f"×{record.get('label_height', '?')}"
        ),
        "",
        (
            "Bands:"
            f" MS={record.get('ms_bands', '?')},"
            f" DSM={record.get('dsm_bands', '?')},"
            f" Label={record.get('label_bands', '?')}"
        ),
        "",
        (
            "CRS available:"
            f" MS={record.get('ms_has_crs', False)},"
            f" DSM={record.get('dsm_has_crs', False)},"
            f" Label={record.get('label_has_crs', False)}"
        ),
        ("MS–Label CRS:" f" {record.get('ms_label_crs_match', '')}"),
        ("MS–Label transform:" f" {record.get('ms_label_transform_match', '')}"),
        "",
        "Notes:",
        str(record.get("notes", "")),
    ]

    axis.text(
        0.02,
        0.98,
        "\n".join(text_lines),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        wrap=True,
    )


# ---------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------


def create_alignment_figure(
    area_id: str,
    record: dict[str, Any],
    ms_path: Path,
    dsm_path: Path,
    label_path: Path,
    max_display_size: int,
    show: bool,
) -> Path:
    """
    Create and save a six-panel alignment figure.
    """
    dimensions_match = bool(record["dimensions_match"])

    if dimensions_match:
        target_width, target_height = compute_display_shape(
            width=int(record["ms_width"]),
            height=int(record["ms_height"]),
            max_display_size=max_display_size,
        )

        ms = load_display_array(
            path=ms_path,
            target_width=target_width,
            target_height=target_height,
            max_display_size=max_display_size,
            is_label=False,
        )

        dsm = load_display_array(
            path=dsm_path,
            target_width=target_width,
            target_height=target_height,
            max_display_size=max_display_size,
            is_label=False,
        )

        label = load_display_array(
            path=label_path,
            target_width=target_width,
            target_height=target_height,
            max_display_size=max_display_size,
            is_label=True,
        )

    else:
        ms = load_display_array(
            path=ms_path,
            target_width=None,
            target_height=None,
            max_display_size=max_display_size,
            is_label=False,
        )

        dsm = load_display_array(
            path=dsm_path,
            target_width=None,
            target_height=None,
            max_display_size=max_display_size,
            is_label=False,
        )

        label = load_display_array(
            path=label_path,
            target_width=None,
            target_height=None,
            max_display_size=max_display_size,
            is_label=True,
        )

    ms_display = percentile_stretch(ms)

    dsm_display = prepare_dsm_for_display(dsm)

    label_display = prepare_label_for_display(label)

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(18, 11),
    )

    axes[0, 0].imshow(ms_display)
    axes[0, 0].set_title("Multispectral image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(
        dsm_display,
        cmap="gray",
    )
    axes[0, 1].set_title("Normalized DSM")
    axes[0, 1].axis("off")

    if label_display.ndim == 2:
        axes[0, 2].imshow(
            label_display,
            cmap="tab20",
            interpolation="nearest",
        )
    else:
        axes[0, 2].imshow(label_display)

    axes[0, 2].set_title("Label")
    axes[0, 2].axis("off")

    if dimensions_match:
        axes[1, 0].imshow(ms_display)

        if label_display.ndim == 2:
            axes[1, 0].imshow(
                label_display,
                cmap="tab20",
                alpha=0.35,
                interpolation="nearest",
            )
        else:
            axes[1, 0].imshow(
                label_display,
                alpha=0.35,
            )

        axes[1, 0].set_title("Image–label overlay")
        axes[1, 0].axis("off")

        label_boundary = compute_label_boundary(label_display)

        boundary_overlay = ms_display.copy()

        boundary_overlay[label_boundary] = np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float32,
        )

        axes[1, 1].imshow(boundary_overlay)
        axes[1, 1].set_title("Label boundaries on image")
        axes[1, 1].axis("off")

    else:
        axes[1, 0].axis("off")
        axes[1, 0].set_title("Overlay unavailable")
        axes[1, 0].text(
            0.5,
            0.5,
            "Original dimensions do not match.\n" "Do not use this area for training.",
            ha="center",
            va="center",
            fontsize=12,
        )

        axes[1, 1].axis("off")
        axes[1, 1].set_title("Boundary overlay unavailable")
        axes[1, 1].text(
            0.5,
            0.5,
            "Original dimensions do not match.\n" "Alignment cannot be verified.",
            ha="center",
            va="center",
            fontsize=12,
        )

    add_metadata_panel(
        axis=axes[1, 2],
        area_id=area_id,
        record=record,
    )

    figure.suptitle(
        f"Alignment check — {area_id}",
        fontsize=16,
    )

    figure.tight_layout(rect=(0, 0, 1, 0.96))

    FIGURE_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = FIGURE_OUTPUT_DIR / f"{area_id}_alignment_check.png"

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    if show:
        plt.show()

    plt.close(figure)

    return output_path


# ---------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------


def print_summary(
    audit_dataframe: pd.DataFrame,
) -> None:
    """Print a compact terminal summary."""
    summary_columns = [
        "area_id",
        "dimensions_match",
        "ms_label_crs_match",
        "ms_label_transform_match",
        "status",
        "notes",
    ]

    available_columns = [
        column for column in summary_columns if column in audit_dataframe.columns
    ]

    print("\n" + "=" * 100)
    print("ALIGNMENT AUDIT SUMMARY")
    print("=" * 100)

    print(audit_dataframe[available_columns].to_string(index=False))

    print("\nStatus counts:")

    print(audit_dataframe["status"].value_counts(dropna=False))

    print(
        "\nImportant: missing CRS alone is not a failure. "
        "For non-georeferenced matching images, identical "
        "dimensions and visual pixel alignment are the "
        "essential checks."
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    args = parse_arguments()

    AUDIT_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ms_files, ms_problems = index_folder(
        MS_DIR,
        MS_EXTENSIONS,
    )

    dsm_files, dsm_problems = index_folder(
        DSM_DIR,
        DSM_EXTENSIONS,
    )

    label_files, label_problems = index_folder(
        LABEL_DIR,
        LABEL_EXTENSIONS,
    )

    all_problems = ms_problems + dsm_problems + label_problems

    if all_problems:
        print("Indexing warnings:")

        for problem in all_problems:
            print(f"  - {problem}")

    all_area_ids = sorted(
        set(ms_files) | set(dsm_files) | set(label_files),
        key=area_sort_key,
    )

    if args.areas:
        requested_areas = {area_id.lower() for area_id in args.areas}

        all_area_ids = [
            area_id for area_id in all_area_ids if area_id.lower() in requested_areas
        ]

    if not all_area_ids:
        raise RuntimeError("No matching area files were found.")

    audit_records: list[dict[str, Any]] = []

    for area_id in all_area_ids:
        print(f"\nChecking {area_id}...")

        ms_path = ms_files.get(area_id)
        dsm_path = dsm_files.get(area_id)
        label_path = label_files.get(area_id)

        record = build_record(
            area_id=area_id,
            ms_path=ms_path,
            dsm_path=dsm_path,
            label_path=label_path,
        )

        audit_records.append(record)

        if ms_path is not None and dsm_path is not None and label_path is not None:
            figure_path = create_alignment_figure(
                area_id=area_id,
                record=record,
                ms_path=ms_path,
                dsm_path=dsm_path,
                label_path=label_path,
                max_display_size=(args.max_display_size),
                show=args.show,
            )

            print(f"  Figure saved: " f"{figure_path}")

        else:
            print("  Visualization skipped because " "one or more files are missing.")

        print(f"  Status: " f"{record['status']}")

    audit_dataframe = pd.DataFrame(audit_records)

    audit_dataframe = audit_dataframe.sort_values(
        by="area_id",
        key=lambda column: column.map(area_sort_key),
    )

    audit_dataframe.to_csv(
        AUDIT_OUTPUT,
        index=False,
    )

    print_summary(audit_dataframe)

    print(f"\nAudit CSV saved to:\n" f"{AUDIT_OUTPUT}")

    print(f"\nAlignment figures saved to:\n" f"{FIGURE_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
