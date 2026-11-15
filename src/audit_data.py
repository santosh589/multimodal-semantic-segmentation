"""Audit image, depth, and label files before preparing the dataset.

Run: ``python src/audit_data.py``
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from PIL import Image


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

# audit_data.py is inside:
# EX_2_CNN_with_DSM/src/audit_data.py
#
# Therefore, parents[1] gives:
# EX_2_CNN_with_DSM/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MS_DIR = DATA_DIR / "mspectral-images"
DSM_DIR = DATA_DIR / "ndsm-images"
LABEL_DIR = DATA_DIR / "labels"

OUTPUT_PATH = PROJECT_ROOT / "metadata" / "data_audit.csv"


# ---------------------------------------------------------------------
# Supported file extensions
# ---------------------------------------------------------------------

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
# Filename matching
# ---------------------------------------------------------------------


def extract_area_id(path: Path) -> str:
    """
    Extract an area ID such as 'area10' from a filename.

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
        raise ValueError(f"Could not extract area ID from filename: {path.name}")

    area_number = int(match.group(1))

    return f"area{area_number}"


def area_sort_key(area_id: str) -> int:
    """Return the numeric component of an area ID."""
    match = re.search(r"\d+", area_id)

    if match is None:
        return 10**9

    return int(match.group())


def index_folder(
    folder: Path,
    allowed_extensions: set[str],
) -> tuple[dict[str, Path], list[str]]:
    """
    Index files in a folder by area ID.

    Returns
    -------
    indexed_files
        Dictionary mapping area IDs to file paths.

    problems
        List of duplicate or invalid filename problems.
    """
    indexed_files: dict[str, Path] = {}
    problems: list[str] = []

    if not folder.exists():
        problems.append(f"Folder does not exist: {folder}")
        return indexed_files, problems

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
                f"Duplicate files for {area_id} in {folder.name}: "
                f"{indexed_files[area_id].name} and {path.name}"
            )
            continue

        indexed_files[area_id] = path

    return indexed_files, problems


# ---------------------------------------------------------------------
# Raster inspection
# ---------------------------------------------------------------------


def inspect_rasterio_file(path: Path) -> dict:
    """
    Inspect a TIFF or another raster readable by Rasterio.

    Only a small sample is read for minimum and maximum values.
    This avoids loading a very large raster entirely into memory.
    """
    with rasterio.open(path) as source:
        sample_height = min(source.height, 1024)
        sample_width = min(source.width, 1024)

        sampled_array = source.read(
            out_shape=(
                source.count,
                sample_height,
                sample_width,
            )
        )

        sampled_array = sampled_array.astype(np.float64)

        finite_values = sampled_array[np.isfinite(sampled_array)]

        if finite_values.size > 0:
            minimum = float(finite_values.min())
            maximum = float(finite_values.max())
        else:
            minimum = np.nan
            maximum = np.nan

        return {
            "width": source.width,
            "height": source.height,
            "bands": source.count,
            "dtype": str(source.dtypes[0]),
            "minimum": minimum,
            "maximum": maximum,
            "crs": str(source.crs) if source.crs else "",
            "nodata": source.nodata,
        }


def inspect_pillow_file(path: Path) -> dict:
    """
    Inspect JPG, PNG, or other standard images using Pillow.
    """
    with Image.open(path) as image:
        image_array = np.asarray(image)

        if image_array.ndim == 2:
            bands = 1
        elif image_array.ndim == 3:
            bands = image_array.shape[2]
        else:
            bands = None

        finite_values = image_array[np.isfinite(image_array)]

        if finite_values.size > 0:
            minimum = float(finite_values.min())
            maximum = float(finite_values.max())
        else:
            minimum = np.nan
            maximum = np.nan

        return {
            "width": image.width,
            "height": image.height,
            "bands": bands,
            "dtype": str(image_array.dtype),
            "minimum": minimum,
            "maximum": maximum,
            "crs": "",
            "nodata": None,
        }


def inspect_file(path: Path) -> dict:
    """
    Inspect an image based on its extension.

    TIFF files are read with Rasterio.
    JPG, JPEG, and PNG files are read with Pillow.
    """
    suffix = path.suffix.lower()

    if suffix in {".tif", ".tiff"}:
        return inspect_rasterio_file(path)

    return inspect_pillow_file(path)


# ---------------------------------------------------------------------
# Label inspection
# ---------------------------------------------------------------------


def count_sample_label_colors(
    label_path: Path,
    max_colors_to_report: int = 20,
) -> tuple[int | None, str]:
    """
    Count unique label values or RGB colors from a downsampled label.

    This is only an audit check. Full label conversion will be implemented
    later in dataset.py.
    """
    suffix = label_path.suffix.lower()

    if suffix in {".tif", ".tiff"}:
        with rasterio.open(label_path) as source:
            sample_height = min(source.height, 1024)
            sample_width = min(source.width, 1024)

            label_array = source.read(
                out_shape=(
                    source.count,
                    sample_height,
                    sample_width,
                )
            )

        label_array = np.moveaxis(label_array, 0, -1)

    else:
        with Image.open(label_path) as image:
            image.thumbnail((1024, 1024))
            label_array = np.asarray(image)

    if label_array.ndim == 2:
        unique_values = np.unique(label_array)

        preview = unique_values[:max_colors_to_report].tolist()

        return (
            int(len(unique_values)),
            str(preview),
        )

    if label_array.ndim == 3:
        if label_array.shape[2] >= 3:
            rgb_array = label_array[:, :, :3]

            pixels = rgb_array.reshape(-1, 3)
            unique_colors = np.unique(pixels, axis=0)

            preview = unique_colors[:max_colors_to_report].tolist()

            return (
                int(len(unique_colors)),
                str(preview),
            )

        unique_values = np.unique(label_array)

        preview = unique_values[:max_colors_to_report].tolist()

        return (
            int(len(unique_values)),
            str(preview),
        )

    return None, ""


# ---------------------------------------------------------------------
# Area audit
# ---------------------------------------------------------------------


def build_audit_record(
    area_id: str,
    ms_path: Path | None,
    dsm_path: Path | None,
    label_path: Path | None,
) -> dict:
    """Create one audit-table row for one area."""
    record = {
        "area_id": area_id,
        "multispectral_exists": ms_path is not None,
        "dsm_exists": dsm_path is not None,
        "label_exists": label_path is not None,
        "multispectral_path": (
            str(ms_path.relative_to(PROJECT_ROOT)) if ms_path is not None else ""
        ),
        "dsm_path": (
            str(dsm_path.relative_to(PROJECT_ROOT)) if dsm_path is not None else ""
        ),
        "label_path": (
            str(label_path.relative_to(PROJECT_ROOT)) if label_path is not None else ""
        ),
        "ms_width": None,
        "ms_height": None,
        "ms_bands": None,
        "ms_dtype": "",
        "ms_min": None,
        "ms_max": None,
        "ms_crs": "",
        "ms_nodata": None,
        "dsm_width": None,
        "dsm_height": None,
        "dsm_bands": None,
        "dsm_dtype": "",
        "dsm_min": None,
        "dsm_max": None,
        "label_width": None,
        "label_height": None,
        "label_bands": None,
        "label_dtype": "",
        "label_min": None,
        "label_max": None,
        "sample_label_unique_count": None,
        "sample_label_values_or_colors": "",
        "dimensions_match": False,
        "is_complete_triplet": False,
        "status": "",
        "error_message": "",
    }

    missing_items = []

    if ms_path is None:
        missing_items.append("multispectral")

    if dsm_path is None:
        missing_items.append("DSM")

    if label_path is None:
        missing_items.append("label")

    if missing_items:
        record["status"] = "missing_" + "_and_".join(missing_items)

        return record

    record["is_complete_triplet"] = True

    try:
        ms_info = inspect_file(ms_path)
        dsm_info = inspect_file(dsm_path)
        label_info = inspect_file(label_path)

        record.update(
            {
                "ms_width": ms_info["width"],
                "ms_height": ms_info["height"],
                "ms_bands": ms_info["bands"],
                "ms_dtype": ms_info["dtype"],
                "ms_min": ms_info["minimum"],
                "ms_max": ms_info["maximum"],
                "ms_crs": ms_info["crs"],
                "ms_nodata": ms_info["nodata"],
                "dsm_width": dsm_info["width"],
                "dsm_height": dsm_info["height"],
                "dsm_bands": dsm_info["bands"],
                "dsm_dtype": dsm_info["dtype"],
                "dsm_min": dsm_info["minimum"],
                "dsm_max": dsm_info["maximum"],
                "label_width": label_info["width"],
                "label_height": label_info["height"],
                "label_bands": label_info["bands"],
                "label_dtype": label_info["dtype"],
                "label_min": label_info["minimum"],
                "label_max": label_info["maximum"],
            }
        )

        dimensions_match = (
            ms_info["width"] == dsm_info["width"] == label_info["width"]
            and ms_info["height"] == dsm_info["height"] == label_info["height"]
        )

        record["dimensions_match"] = dimensions_match

        unique_count, unique_preview = count_sample_label_colors(label_path)

        record["sample_label_unique_count"] = unique_count
        record["sample_label_values_or_colors"] = unique_preview

        if not dimensions_match:
            record["status"] = "dimension_mismatch"

        elif ms_info["bands"] != 3:
            record["status"] = f"unexpected_multispectral_bands_" f"{ms_info['bands']}"

        elif dsm_info["bands"] != 1:
            record["status"] = f"unexpected_dsm_bands_" f"{dsm_info['bands']}"

        else:
            record["status"] = "ready"

    except Exception as error:
        record["status"] = "inspection_error"
        record["error_message"] = str(error)

    return record


# ---------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------


def print_folder_summary(
    name: str,
    indexed_files: dict[str, Path],
) -> None:
    """Print indexed files from one input folder."""
    print(f"\n{name}: {len(indexed_files)} files")

    for area_id in sorted(
        indexed_files,
        key=area_sort_key,
    ):
        print(f"  {area_id:10s} -> " f"{indexed_files[area_id].name}")


def print_audit_summary(audit_df: pd.DataFrame) -> None:
    """Print the final audit summary."""
    print("\n" + "=" * 80)
    print("DATASET AUDIT SUMMARY")
    print("=" * 80)

    display_columns = [
        "area_id",
        "multispectral_exists",
        "dsm_exists",
        "label_exists",
        "dimensions_match",
        "status",
    ]

    print(audit_df[display_columns].to_string(index=False))

    ready_areas = audit_df.loc[
        audit_df["status"] == "ready",
        "area_id",
    ].tolist()

    incomplete_areas = audit_df.loc[
        audit_df["status"] != "ready",
        "area_id",
    ].tolist()

    print("\nReady areas:")
    print(ready_areas)

    print("\nProblematic or incomplete areas:")
    print(incomplete_areas)

    print("\nCounts:")
    print(f"  Total areas: {len(audit_df)}")
    print(f"  Ready areas: {len(ready_areas)}")
    print(f"  Problematic areas: " f"{len(incomplete_areas)}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    """Run the complete dataset audit."""
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Multispectral directory: {MS_DIR}")
    print(f"DSM directory: {DSM_DIR}")
    print(f"Label directory: {LABEL_DIR}")

    OUTPUT_PATH.parent.mkdir(
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

    print_folder_summary(
        "Multispectral images",
        ms_files,
    )

    print_folder_summary(
        "DSM images",
        dsm_files,
    )

    print_folder_summary(
        "Label images",
        label_files,
    )

    indexing_problems = ms_problems + dsm_problems + label_problems

    if indexing_problems:
        print("\nIndexing problems:")

        for problem in indexing_problems:
            print(f"  - {problem}")

    all_area_ids = sorted(
        set(ms_files) | set(dsm_files) | set(label_files),
        key=area_sort_key,
    )

    if not all_area_ids:
        raise RuntimeError("No files with valid area IDs were found.")

    audit_records = []

    for area_id in all_area_ids:
        print(f"\nAuditing {area_id}...")

        record = build_audit_record(
            area_id=area_id,
            ms_path=ms_files.get(area_id),
            dsm_path=dsm_files.get(area_id),
            label_path=label_files.get(area_id),
        )

        audit_records.append(record)

    audit_df = pd.DataFrame(audit_records)

    audit_df = audit_df.sort_values(
        by="area_id",
        key=lambda column: column.map(area_sort_key),
    )

    audit_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print_audit_summary(audit_df)

    print(f"\nAudit file saved to:\n" f"{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
