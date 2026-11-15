"""Visualize matched raw images, depth maps, and segmentation labels.

Run: ``python src/raw_visualize.py``
"""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from PIL import Image


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MS_FOLDER = PROJECT_ROOT / "data" / "mspectral-images"
DSM_FOLDER = PROJECT_ROOT / "data" / "ndsm-images"
LABEL_FOLDER = PROJECT_ROOT / "data" / "labels"

# Number of matched samples to visualize
NUMBER_OF_SAMPLES = 10


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================


def extract_area_number(file_path: Path) -> int | None:
    """
    Extract the area number from a filename.

    Examples
    --------
    top_mosaic_09cm_area1.tif
        -> 1

    dsm_09cm_matching_area10_normalized.jpg
        -> 10
    """
    match = re.search(
        r"area[_-]?(\d+)(?!\d)",
        file_path.stem,
        flags=re.IGNORECASE,
    )

    if match is None:
        return None

    return int(match.group(1))


def create_file_mapping(
    folder: Path,
    allowed_extensions: tuple[str, ...],
) -> dict[int, Path]:
    """
    Create a dictionary where:

        key   = area number
        value = file path
    """
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found:\n{folder}")

    if not folder.is_dir():
        raise NotADirectoryError(f"Path is not a folder:\n{folder}")

    file_mapping = {}

    for file_path in sorted(folder.iterdir()):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in allowed_extensions:
            continue

        area_number = extract_area_number(file_path)

        if area_number is None:
            print(f"Warning: Could not identify area number from " f"{file_path.name}")
            continue

        if area_number in file_mapping:
            print(f"Warning: Duplicate area {area_number} found in " f"{folder.name}")
            print(f"Keeping: {file_mapping[area_number].name}")
            print(f"Ignoring: {file_path.name}")
            continue

        file_mapping[area_number] = file_path

    return file_mapping


def read_multispectral(file_path: Path) -> np.ndarray:
    """
    Read a multispectral raster and convert it from:

        bands, height, width

    to:

        height, width, bands
    """
    with rasterio.open(file_path) as src:
        image = src.read()

    image = np.moveaxis(image, 0, -1)

    if image.shape[-1] < 3:
        raise ValueError(f"Multispectral image has fewer than 3 bands:\n{file_path}")

    # Use first three bands for display
    rgb_image = image[:, :, :3]

    return normalize_rgb(rgb_image)


def normalize_rgb(image: np.ndarray) -> np.ndarray:
    """
    Normalize an RGB image using percentile stretching.

    This generally gives better visualization than using
    the absolute minimum and maximum values.
    """
    image = image.astype(np.float32)

    lower_percentile = np.percentile(
        image,
        2,
        axis=(0, 1),
        keepdims=True,
    )

    upper_percentile = np.percentile(
        image,
        98,
        axis=(0, 1),
        keepdims=True,
    )

    normalized = (image - lower_percentile) / (
        upper_percentile - lower_percentile + 1e-8
    )

    return np.clip(normalized, 0, 1)


def read_single_band(file_path: Path) -> np.ndarray:
    """
    Read one-band DSM or label files.

    Supports:
    - TIF
    - TIFF
    - JPG
    - JPEG
    - PNG
    """
    image_extensions = {".jpg", ".jpeg", ".png"}

    if file_path.suffix.lower() in image_extensions:
        with Image.open(file_path) as image:
            return np.asarray(image.convert("L"))

    with rasterio.open(file_path) as src:
        return src.read(1)


def print_unmatched_files(
    ms_files: dict[int, Path],
    dsm_files: dict[int, Path],
    label_files: dict[int, Path],
) -> None:
    """
    Print area numbers that are missing from one or more folders.
    """
    all_area_numbers = set(ms_files) | set(dsm_files) | set(label_files)

    print("\nFile matching report")
    print("=" * 60)

    for area_number in sorted(all_area_numbers):
        missing = []

        if area_number not in ms_files:
            missing.append("multispectral")

        if area_number not in dsm_files:
            missing.append("nDSM")

        if area_number not in label_files:
            missing.append("label")

        if missing:
            print(f"Area {area_number}: missing " f"{', '.join(missing)}")


def validate_image_shapes(
    image: np.ndarray,
    dsm: np.ndarray,
    label: np.ndarray,
    area_number: int,
) -> None:
    """
    Check whether multispectral, DSM, and label images
    have matching height and width.
    """
    image_shape = image.shape[:2]
    dsm_shape = dsm.shape[:2]
    label_shape = label.shape[:2]

    if image_shape != dsm_shape or image_shape != label_shape:
        print(f"\nWarning for area {area_number}:")
        print(f"Multispectral shape: {image_shape}")
        print(f"nDSM shape:          {dsm_shape}")
        print(f"Label shape:         {label_shape}")


# ============================================================
# 3. MAIN FUNCTION
# ============================================================


def main() -> None:
    # --------------------------------------------------------
    # Create mappings: area number -> file path
    # --------------------------------------------------------

    ms_files = create_file_mapping(
        folder=MS_FOLDER,
        allowed_extensions=(".tif", ".tiff"),
    )

    dsm_files = create_file_mapping(
        folder=DSM_FOLDER,
        allowed_extensions=(
            ".tif",
            ".tiff",
            ".jpg",
            ".jpeg",
            ".png",
        ),
    )

    label_files = create_file_mapping(
        folder=LABEL_FOLDER,
        allowed_extensions=(
            ".tif",
            ".tiff",
            ".jpg",
            ".jpeg",
            ".png",
        ),
    )

    # --------------------------------------------------------
    # Find common area numbers
    # --------------------------------------------------------

    common_area_numbers = sorted(set(ms_files) & set(dsm_files) & set(label_files))

    print("=" * 60)
    print("Dataset summary")
    print("=" * 60)
    print(f"Multispectral images : {len(ms_files)}")
    print(f"nDSM images          : {len(dsm_files)}")
    print(f"Label images         : {len(label_files)}")
    print(f"Matched samples      : {len(common_area_numbers)}")

    print_unmatched_files(
        ms_files=ms_files,
        dsm_files=dsm_files,
        label_files=label_files,
    )

    if not common_area_numbers:
        raise RuntimeError(
            "No matching multispectral, nDSM, and label files were found."
        )

    # --------------------------------------------------------
    # Select matched samples
    # --------------------------------------------------------

    number_to_display = min(
        NUMBER_OF_SAMPLES,
        len(common_area_numbers),
    )

    selected_areas = common_area_numbers[:number_to_display]

    print("\nSelected matched samples")
    print("=" * 60)

    for area_number in selected_areas:
        print(f"\nArea {area_number}")
        print(f"Multispectral: {ms_files[area_number].name}")
        print(f"nDSM:          {dsm_files[area_number].name}")
        print(f"Label:         {label_files[area_number].name}")

    # --------------------------------------------------------
    # Create visualization
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        nrows=number_to_display,
        ncols=3,
        figsize=(15, 5 * number_to_display),
    )

    # Keep axes two-dimensional when only one row exists
    axes = np.atleast_2d(axes)

    for row_index, area_number in enumerate(selected_areas):
        ms_path = ms_files[area_number]
        dsm_path = dsm_files[area_number]
        label_path = label_files[area_number]

        image = read_multispectral(ms_path)
        dsm = read_single_band(dsm_path)
        label = read_single_band(label_path)

        validate_image_shapes(
            image=image,
            dsm=dsm,
            label=label,
            area_number=area_number,
        )

        # Multispectral image
        axes[row_index, 0].imshow(image)
        axes[row_index, 0].set_title(
            f"Area {area_number} — Multispectral\n" f"{ms_path.name}"
        )
        axes[row_index, 0].axis("off")

        # nDSM image
        axes[row_index, 1].imshow(
            dsm,
            cmap="gray",
        )
        axes[row_index, 1].set_title(f"Area {area_number} — nDSM\n" f"{dsm_path.name}")
        axes[row_index, 1].axis("off")

        # Label image
        axes[row_index, 2].imshow(
            label,
            cmap="gray",
        )
        axes[row_index, 2].set_title(
            f"Area {area_number} — Label\n" f"{label_path.name}"
        )
        axes[row_index, 2].axis("off")

    plt.tight_layout()
    plt.show()


# ============================================================
# 4. RUN SCRIPT
# ============================================================

if __name__ == "__main__":
    main()
