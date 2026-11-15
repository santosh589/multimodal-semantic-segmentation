"""Calculate training-set normalization statistics for image channels.

Run: ``python src/calculate_statistics.py``
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from PIL import Image


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_AREAS_PATH = PROJECT_ROOT / "metadata" / "train_areas.txt"

MS_DIR = PROJECT_ROOT / "data" / "mspectral-images"

DSM_DIR = PROJECT_ROOT / "data" / "ndsm-images"

OUTPUT_PATH = PROJECT_ROOT / "metadata" / "normalization_stats.json"

FIGURE_OUTPUT_PATH = (
    PROJECT_ROOT / "outputs" / "figures" / "training_normalization_statistics.png"
)


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


# ============================================================
# Command-line arguments
# ============================================================


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate training-only multispectral mean/std " "and inspect DSM scaling."
        )
    )

    parser.add_argument(
        "--ms-divisor",
        default="auto",
        help=(
            "Multispectral scaling divisor. Use 'auto' or "
            "provide a number such as 255 or 10000."
        ),
    )

    parser.add_argument(
        "--dsm-divisor",
        default="auto",
        help=("DSM scaling divisor. Use 'auto' or provide " "a number such as 255."),
    )

    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not save the statistics summary figure.",
    )

    return parser.parse_args()


# ============================================================
# General utilities
# ============================================================


def extract_area_id(path: Path) -> str:
    """
    Extract area ID from a filename.

    Examples
    --------
    top_mosaic_09cm_area13.tif -> area13
    dsm_09cm_matching_area13_normalized.jpg -> area13
    """
    match = re.search(
        r"area[\s_-]*(\d+)",
        path.stem,
        flags=re.IGNORECASE,
    )

    if match is None:
        raise ValueError(f"Could not extract area ID from: {path.name}")

    return f"area{int(match.group(1))}"


def area_sort_key(area_id: str) -> int:
    """Return numeric component for correct area sorting."""
    match = re.search(r"\d+", area_id)

    if match is None:
        return 10**9

    return int(match.group())


def read_area_ids(path: Path) -> list[str]:
    """Read one area ID per line."""
    if not path.exists():
        raise FileNotFoundError(
            f"Training-area file not found: {path}\n" "Run src/create_splits.py first."
        )

    area_ids = [
        line.strip().lower()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if not area_ids:
        raise RuntimeError(f"No training areas were found in {path}")

    return sorted(
        area_ids,
        key=area_sort_key,
    )


def index_folder(
    folder: Path,
    extensions: set[str],
) -> dict[str, Path]:
    """Index image files by area ID."""
    if not folder.exists():
        raise FileNotFoundError(f"Input directory not found: {folder}")

    indexed_files: dict[str, Path] = {}

    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue

        if path.suffix.lower() not in extensions:
            continue

        try:
            area_id = extract_area_id(path)

        except ValueError as error:
            print(f"[WARNING] {error}")
            continue

        if area_id in indexed_files:
            raise RuntimeError(
                f"Duplicate files found for {area_id} " f"inside {folder}"
            )

        indexed_files[area_id] = path

    return indexed_files


def parse_divisor_argument(
    value: str,
) -> float | None:
    """
    Convert divisor argument.

    Returns None when automatic detection is requested.
    """
    if str(value).strip().lower() == "auto":
        return None

    divisor = float(value)

    if divisor <= 0:
        raise ValueError("Scaling divisor must be greater than zero.")

    return divisor


def determine_divisor(
    observed_maximum: float,
) -> float:
    """
    Select a conventional scaling divisor from the observed maximum.

    Rules
    -----
    max <= 1      -> already scaled
    max <= 255    -> 8-bit range
    max <= 10000  -> common reflectance range
    max <= 65535  -> full uint16 range
    """
    if not np.isfinite(observed_maximum):
        raise ValueError("Cannot determine divisor from a non-finite maximum.")

    if observed_maximum <= 1.0:
        return 1.0

    if observed_maximum <= 255.0:
        return 255.0

    if observed_maximum <= 10_000.0:
        return 10_000.0

    if observed_maximum <= 65_535.0:
        return 65_535.0

    raise ValueError(
        "Observed values exceed the supported automatic ranges. "
        "Provide the divisor manually using --ms-divisor or "
        "--dsm-divisor."
    )


def save_json(
    data: dict[str, Any],
    path: Path,
) -> None:
    """Write a formatted JSON file."""
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
# Multispectral scanning
# ============================================================


def inspect_multispectral_raw_ranges(
    paths: list[Path],
) -> dict[str, Any]:
    """
    Calculate exact raw minimum and maximum values per channel.

    Raster windows are used to avoid loading complete large rasters.
    """
    raw_minimum = np.full(
        3,
        np.inf,
        dtype=np.float64,
    )

    raw_maximum = np.full(
        3,
        -np.inf,
        dtype=np.float64,
    )

    dtypes: set[str] = set()
    total_pixels = np.zeros(
        3,
        dtype=np.int64,
    )

    for path in paths:
        print(f"Inspecting multispectral range: {path.name}")

        with rasterio.open(path) as source:
            if source.count != 3:
                raise ValueError(
                    f"{path.name} has {source.count} bands. "
                    "Expected exactly 3 multispectral bands."
                )

            dtypes.update(source.dtypes)

            for _, window in source.block_windows(1):
                data = source.read(
                    window=window,
                    masked=True,
                )

                for channel in range(3):
                    values = data[channel].compressed()

                    if values.size == 0:
                        continue

                    values = values.astype(
                        np.float64,
                        copy=False,
                    )

                    finite_values = values[np.isfinite(values)]

                    if finite_values.size == 0:
                        continue

                    raw_minimum[channel] = min(
                        raw_minimum[channel],
                        float(finite_values.min()),
                    )

                    raw_maximum[channel] = max(
                        raw_maximum[channel],
                        float(finite_values.max()),
                    )

                    total_pixels[channel] += finite_values.size

    if np.any(~np.isfinite(raw_minimum)):
        raise RuntimeError("Some multispectral channels contain no valid pixels.")

    return {
        "raw_minimum": raw_minimum,
        "raw_maximum": raw_maximum,
        "dtypes": sorted(dtypes),
        "pixel_counts": total_pixels,
    }


def calculate_multispectral_statistics(
    paths: list[Path],
    divisor: float,
) -> dict[str, Any]:
    """
    Calculate mean and standard deviation after scaling to [0,1].

    Only training-area pixels are included.
    """
    channel_sum = np.zeros(
        3,
        dtype=np.float64,
    )

    channel_squared_sum = np.zeros(
        3,
        dtype=np.float64,
    )

    channel_count = np.zeros(
        3,
        dtype=np.int64,
    )

    scaled_minimum = np.full(
        3,
        np.inf,
        dtype=np.float64,
    )

    scaled_maximum = np.full(
        3,
        -np.inf,
        dtype=np.float64,
    )

    for path in paths:
        print(f"Calculating statistics: {path.name}")

        with rasterio.open(path) as source:
            if source.count != 3:
                raise ValueError(
                    f"{path.name} has {source.count} bands. "
                    "Expected exactly 3 bands."
                )

            for _, window in source.block_windows(1):
                data = source.read(
                    window=window,
                    masked=True,
                )

                for channel in range(3):
                    values = data[channel].compressed()

                    if values.size == 0:
                        continue

                    values = (
                        values.astype(
                            np.float64,
                            copy=False,
                        )
                        / divisor
                    )

                    finite_values = values[np.isfinite(values)]

                    if finite_values.size == 0:
                        continue

                    channel_sum[channel] += finite_values.sum(dtype=np.float64)

                    channel_squared_sum[channel] += np.square(
                        finite_values,
                        dtype=np.float64,
                    ).sum(dtype=np.float64)

                    channel_count[channel] += finite_values.size

                    scaled_minimum[channel] = min(
                        scaled_minimum[channel],
                        float(finite_values.min()),
                    )

                    scaled_maximum[channel] = max(
                        scaled_maximum[channel],
                        float(finite_values.max()),
                    )

    if np.any(channel_count == 0):
        raise RuntimeError(
            "One or more multispectral channels " "have zero valid pixels."
        )

    mean = channel_sum / channel_count

    variance = channel_squared_sum / channel_count - np.square(mean)

    variance = np.maximum(
        variance,
        0.0,
    )

    standard_deviation = np.sqrt(variance)

    if np.any(standard_deviation <= 0):
        raise RuntimeError("One or more channels have zero standard deviation.")

    return {
        "mean": mean,
        "std": standard_deviation,
        "scaled_minimum": scaled_minimum,
        "scaled_maximum": scaled_maximum,
        "pixel_counts": channel_count,
    }


# ============================================================
# DSM scanning
# ============================================================


def read_dsm_array(
    path: Path,
) -> np.ndarray:
    """Read a one-channel DSM using Rasterio or Pillow."""
    if path.suffix.lower() in {
        ".tif",
        ".tiff",
    }:
        with rasterio.open(path) as source:
            if source.count != 1:
                raise ValueError(
                    f"{path.name} has {source.count} bands. "
                    "Expected a one-channel DSM."
                )

            array = source.read(
                1,
                masked=True,
            )

            if np.ma.isMaskedArray(array):
                values = array.compressed()
            else:
                values = array.reshape(-1)

            return values.astype(
                np.float64,
                copy=False,
            )

    with Image.open(path) as image:
        array = np.asarray(image)

    if array.ndim == 3:
        if array.shape[2] == 1:
            array = array[:, :, 0]
        else:
            raise ValueError(
                f"{path.name} has shape {array.shape}. " "Expected a one-channel DSM."
            )

    return array.astype(
        np.float64,
        copy=False,
    ).reshape(-1)


def inspect_dsm_ranges(
    paths: list[Path],
    divisor: float | None,
) -> dict[str, Any]:
    """Calculate raw and scaled DSM ranges."""
    raw_minimum = np.inf
    raw_maximum = -np.inf

    dtypes: set[str] = set()
    total_pixels = 0

    for path in paths:
        print(f"Inspecting DSM range: {path.name}")

        if path.suffix.lower() in {
            ".tif",
            ".tiff",
        }:
            with rasterio.open(path) as source:
                dtypes.update(source.dtypes)
        else:
            with Image.open(path) as image:
                dtypes.add(str(np.asarray(image).dtype))

        values = read_dsm_array(path)

        finite_values = values[np.isfinite(values)]

        if finite_values.size == 0:
            continue

        raw_minimum = min(
            raw_minimum,
            float(finite_values.min()),
        )

        raw_maximum = max(
            raw_maximum,
            float(finite_values.max()),
        )

        total_pixels += finite_values.size

    if not np.isfinite(raw_minimum):
        raise RuntimeError("No valid DSM pixels were found.")

    selected_divisor = determine_divisor(raw_maximum) if divisor is None else divisor

    scaled_minimum = raw_minimum / selected_divisor

    scaled_maximum = raw_maximum / selected_divisor

    return {
        "raw_minimum": raw_minimum,
        "raw_maximum": raw_maximum,
        "scaled_minimum": scaled_minimum,
        "scaled_maximum": scaled_maximum,
        "scaling_divisor": selected_divisor,
        "dtypes": sorted(dtypes),
        "pixel_count": total_pixels,
    }


# ============================================================
# Validation
# ============================================================


def validate_scaled_ranges(
    multispectral_statistics: dict[str, Any],
    dsm_statistics: dict[str, Any],
) -> list[str]:
    """Check whether scaled values fall inside [0,1]."""
    warnings_found: list[str] = []

    ms_minimum = np.asarray(multispectral_statistics["scaled_minimum"])

    ms_maximum = np.asarray(multispectral_statistics["scaled_maximum"])

    if np.any(ms_minimum < 0):
        warnings_found.append("Multispectral scaled minimum is below 0.")

    if np.any(ms_maximum > 1):
        warnings_found.append("Multispectral scaled maximum is above 1.")

    if dsm_statistics["scaled_minimum"] < 0:
        warnings_found.append("DSM scaled minimum is below 0.")

    if dsm_statistics["scaled_maximum"] > 1:
        warnings_found.append("DSM scaled maximum is above 1.")

    return warnings_found


# ============================================================
# Visualization
# ============================================================


def create_statistics_figure(
    mean: np.ndarray,
    std: np.ndarray,
    scaled_minimum: np.ndarray,
    scaled_maximum: np.ndarray,
    dsm_statistics: dict[str, Any],
) -> None:
    """Save a compact statistics summary figure."""
    FIGURE_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    channel_labels = [
        "Channel 1",
        "Channel 2",
        "Channel 3",
    ]

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
    )

    x_positions = np.arange(3)

    axes[0].bar(
        x_positions,
        mean,
        yerr=std,
        capsize=6,
    )

    axes[0].set_xticks(
        x_positions,
        channel_labels,
    )

    axes[0].set_title("Training multispectral mean ± std")

    axes[0].set_ylabel("Scaled value")

    axes[0].set_ylim(
        bottom=0,
    )

    axes[1].bar(
        x_positions,
        scaled_maximum - scaled_minimum,
        bottom=scaled_minimum,
    )

    axes[1].set_xticks(
        x_positions,
        channel_labels,
    )

    axes[1].set_title("Training multispectral scaled ranges")

    axes[1].set_ylabel("Scaled value")

    axes[1].set_ylim(
        0,
        max(
            1.05,
            float(np.max(scaled_maximum) * 1.05),
        ),
    )

    axes[2].bar(
        ["DSM"],
        [dsm_statistics["scaled_maximum"] - dsm_statistics["scaled_minimum"]],
        bottom=[dsm_statistics["scaled_minimum"]],
    )

    axes[2].set_title("Training DSM scaled range")

    axes[2].set_ylabel("Scaled value")

    axes[2].set_ylim(
        0,
        max(
            1.05,
            dsm_statistics["scaled_maximum"] * 1.05,
        ),
    )

    figure.tight_layout()

    figure.savefig(
        FIGURE_OUTPUT_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


# ============================================================
# Main
# ============================================================


def main() -> None:
    arguments = parse_arguments()

    training_areas = read_area_ids(TRAIN_AREAS_PATH)

    multispectral_files = index_folder(
        MS_DIR,
        MS_EXTENSIONS,
    )

    dsm_files = index_folder(
        DSM_DIR,
        DSM_EXTENSIONS,
    )

    missing_multispectral = [
        area_id for area_id in training_areas if area_id not in multispectral_files
    ]

    missing_dsm = [area_id for area_id in training_areas if area_id not in dsm_files]

    if missing_multispectral:
        raise FileNotFoundError(
            "Missing multispectral files for: " f"{missing_multispectral}"
        )

    if missing_dsm:
        raise FileNotFoundError("Missing DSM files for: " f"{missing_dsm}")

    multispectral_paths = [multispectral_files[area_id] for area_id in training_areas]

    dsm_paths = [dsm_files[area_id] for area_id in training_areas]

    print("=" * 75)
    print("TRAINING-ONLY NORMALIZATION STATISTICS")
    print("=" * 75)

    print(f"\nTraining areas ({len(training_areas)}):")
    print(training_areas)

    print("\nStep 1: inspecting raw multispectral ranges")

    multispectral_raw = inspect_multispectral_raw_ranges(multispectral_paths)

    requested_ms_divisor = parse_divisor_argument(arguments.ms_divisor)

    if requested_ms_divisor is None:
        ms_divisor = determine_divisor(float(np.max(multispectral_raw["raw_maximum"])))
    else:
        ms_divisor = requested_ms_divisor

    print(f"\nSelected multispectral divisor: " f"{ms_divisor}")

    print("\nStep 2: calculating multispectral mean/std")

    multispectral_statistics = calculate_multispectral_statistics(
        paths=multispectral_paths,
        divisor=ms_divisor,
    )

    requested_dsm_divisor = parse_divisor_argument(arguments.dsm_divisor)

    print("\nStep 3: inspecting DSM ranges")

    dsm_statistics = inspect_dsm_ranges(
        paths=dsm_paths,
        divisor=requested_dsm_divisor,
    )

    range_warnings = validate_scaled_ranges(
        multispectral_statistics,
        dsm_statistics,
    )

    output_data = {
        "training_areas": training_areas,
        "multispectral": {
            "number_of_channels": 3,
            "dtypes": (multispectral_raw["dtypes"]),
            "raw_min": (multispectral_raw["raw_minimum"].tolist()),
            "raw_max": (multispectral_raw["raw_maximum"].tolist()),
            "scaling_divisor": float(ms_divisor),
            "scaled_min": (multispectral_statistics["scaled_minimum"].tolist()),
            "scaled_max": (multispectral_statistics["scaled_maximum"].tolist()),
            "mean": (multispectral_statistics["mean"].tolist()),
            "std": (multispectral_statistics["std"].tolist()),
            "valid_pixel_count_per_channel": (
                multispectral_statistics["pixel_counts"].tolist()
            ),
            "normalization_formula": ("(scaled_value - mean) / std"),
        },
        "dsm": {
            "dtypes": dsm_statistics["dtypes"],
            "raw_min": float(dsm_statistics["raw_minimum"]),
            "raw_max": float(dsm_statistics["raw_maximum"]),
            "scaling_divisor": float(dsm_statistics["scaling_divisor"]),
            "scaled_min": float(dsm_statistics["scaled_minimum"]),
            "scaled_max": float(dsm_statistics["scaled_maximum"]),
            "valid_pixel_count": int(dsm_statistics["pixel_count"]),
            "apply_mean_std_normalization": False,
        },
        "range_warnings": range_warnings,
    }

    save_json(
        output_data,
        OUTPUT_PATH,
    )

    if not arguments.no_plot:
        create_statistics_figure(
            mean=multispectral_statistics["mean"],
            std=multispectral_statistics["std"],
            scaled_minimum=(multispectral_statistics["scaled_minimum"]),
            scaled_maximum=(multispectral_statistics["scaled_maximum"]),
            dsm_statistics=dsm_statistics,
        )

    print("\n" + "=" * 75)
    print("RESULTS")
    print("=" * 75)

    print("\nMultispectral raw minimum:")
    print(multispectral_raw["raw_minimum"].tolist())

    print("\nMultispectral raw maximum:")
    print(multispectral_raw["raw_maximum"].tolist())

    print(f"\nMultispectral scaling divisor: " f"{ms_divisor}")

    print("\nMultispectral scaled minimum:")
    print(multispectral_statistics["scaled_minimum"].tolist())

    print("\nMultispectral scaled maximum:")
    print(multispectral_statistics["scaled_maximum"].tolist())

    print("\nMultispectral mean:")
    print(multispectral_statistics["mean"].tolist())

    print("\nMultispectral standard deviation:")
    print(multispectral_statistics["std"].tolist())

    print("\nDSM raw range:")
    print(
        dsm_statistics["raw_minimum"],
        "to",
        dsm_statistics["raw_maximum"],
    )

    print(f"\nDSM scaling divisor: " f"{dsm_statistics['scaling_divisor']}")

    print("\nDSM scaled range:")
    print(
        dsm_statistics["scaled_minimum"],
        "to",
        dsm_statistics["scaled_maximum"],
    )

    if range_warnings:
        print("\nWarnings:")

        for warning in range_warnings:
            print(f"  - {warning}")

    else:
        print("\nAll scaled values are inside [0,1].")

    print(f"\nStatistics saved to:\n{OUTPUT_PATH}")

    if not arguments.no_plot:
        print(f"\nFigure saved to:\n" f"{FIGURE_OUTPUT_PATH}")

    print(
        "\nImportant: mean and standard deviation were "
        "calculated using training areas only."
    )


if __name__ == "__main__":
    main()
