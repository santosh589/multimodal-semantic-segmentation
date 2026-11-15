"""Interactively inspect image, depth, and segmentation-label alignment.

Run: ``python src/interactive_alignment_viewer.py``
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.widgets import Button, CheckButtons, Slider
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

# This script is expected at:
# EX_2_CNN_with_DSM/src/interactive_alignment_viewer.py
#
# Therefore, parents[1] is the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MS_DIR = PROJECT_ROOT / "data" / "mspectral-images"
DSM_DIR = PROJECT_ROOT / "data" / "ndsm-images"
LABEL_DIR = PROJECT_ROOT / "data" / "labels"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures" / "interactive_alignment"


MS_EXTENSIONS = {".tif", ".tiff"}

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
# Command-line arguments
# ============================================================


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactively compare multispectral images, "
            "DSMs, and labels using an overlay wipe slicer."
        )
    )

    parser.add_argument(
        "--areas",
        nargs="*",
        default=None,
        help=("Optional area IDs to include. " "Example: --areas area1 area10 area11"),
    )

    parser.add_argument(
        "--max-display-size",
        type=int,
        default=1400,
        help=(
            "Maximum width or height used for display. "
            "The source files are not changed."
        ),
    )

    return parser.parse_args()


# ============================================================
# Filename matching
# ============================================================


def extract_area_id(path: Path) -> str:
    """
    Extract area ID from a filename.

    Examples
    --------
    top_mosaic_09cm_area10.tif -> area10

    dsm_09cm_matching_area10_normalized.jpg -> area10
    """
    match = re.search(
        r"area[\s_-]*(\d+)",
        path.stem,
        flags=re.IGNORECASE,
    )

    if match is None:
        raise ValueError(f"Could not identify area ID from {path.name}")

    return f"area{int(match.group(1))}"


def area_sort_key(area_id: str) -> int:
    """Sort area1, area2, area10 numerically."""
    match = re.search(r"\d+", area_id)

    if match is None:
        return 10**9

    return int(match.group())


def index_folder(
    folder: Path,
    extensions: set[str],
) -> dict[str, Path]:
    """Index files by area ID."""
    indexed_files: dict[str, Path] = {}

    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

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
            raise ValueError(f"Duplicate files found for {area_id} " f"inside {folder}")

        indexed_files[area_id] = path

    return indexed_files


# ============================================================
# Metadata inspection
# ============================================================


def inspect_file(path: Path) -> dict[str, Any]:
    """Read image dimensions, bands, datatype, CRS and transform."""
    if path.suffix.lower() in {".tif", ".tiff"}:
        with rasterio.open(path) as source:
            return {
                "width": source.width,
                "height": source.height,
                "bands": source.count,
                "dtype": str(source.dtypes[0]),
                "crs": (str(source.crs) if source.crs is not None else "Missing"),
                "transform": str(source.transform),
                "nodata": source.nodata,
            }

    with Image.open(path) as image:
        array = np.asarray(image)

        if array.ndim == 2:
            bands = 1
        elif array.ndim == 3:
            bands = array.shape[2]
        else:
            bands = None

        return {
            "width": image.width,
            "height": image.height,
            "bands": bands,
            "dtype": str(array.dtype),
            "crs": "Missing",
            "transform": "Missing",
            "nodata": None,
        }


def dimensions_match(
    ms_info: dict[str, Any],
    dsm_info: dict[str, Any],
    label_info: dict[str, Any],
) -> bool:
    """Check whether all three images have identical dimensions."""
    return (
        ms_info["width"] == dsm_info["width"] == label_info["width"]
        and ms_info["height"] == dsm_info["height"] == label_info["height"]
    )


# ============================================================
# Display-size handling
# ============================================================


def calculate_display_size(
    width: int,
    height: int,
    maximum_size: int,
) -> tuple[int, int]:
    """Reduce large images while preserving aspect ratio."""
    scale = min(
        1.0,
        maximum_size / max(width, height),
    )

    display_width = max(
        1,
        int(round(width * scale)),
    )

    display_height = max(
        1,
        int(round(height * scale)),
    )

    return display_width, display_height


# ============================================================
# Image reading
# ============================================================


def read_for_display(
    path: Path,
    display_width: int,
    display_height: int,
    nearest_neighbour: bool = False,
) -> np.ndarray:
    """
    Read an image at a reduced display resolution.

    This changes only the displayed copy, not the original data.
    """
    if path.suffix.lower() in {".tif", ".tiff"}:
        with rasterio.open(path) as source:
            resampling = (
                Resampling.nearest if nearest_neighbour else Resampling.bilinear
            )

            array = source.read(
                out_shape=(
                    source.count,
                    display_height,
                    display_width,
                ),
                resampling=resampling,
            )

        return np.moveaxis(array, 0, -1)

    with Image.open(path) as image:
        resampling = (
            Image.Resampling.NEAREST if nearest_neighbour else Image.Resampling.BILINEAR
        )

        image = image.resize(
            (display_width, display_height),
            resample=resampling,
        )

        return np.asarray(image)


# ============================================================
# Display preparation
# ============================================================


def stretch_multispectral(
    image: np.ndarray,
) -> np.ndarray:
    """
    Apply 2–98 percentile stretching for visualization.

    This is not the training normalization.
    """
    image = image.astype(np.float32)

    if image.ndim == 2:
        image = image[:, :, None]

    if image.shape[2] == 1:
        image = np.repeat(
            image,
            3,
            axis=2,
        )

    elif image.shape[2] >= 3:
        image = image[:, :, :3]

    else:
        image = np.pad(
            image,
            (
                (0, 0),
                (0, 0),
                (0, 3 - image.shape[2]),
            ),
            mode="edge",
        )

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


def normalize_dsm(
    dsm: np.ndarray,
) -> np.ndarray:
    """Prepare DSM for grayscale and colour display."""
    if dsm.ndim == 3:
        dsm = dsm[:, :, 0]

    dsm = dsm.astype(np.float32)

    valid_values = dsm[np.isfinite(dsm)]

    if valid_values.size == 0:
        return np.zeros_like(
            dsm,
            dtype=np.float32,
        )

    lower = np.percentile(
        valid_values,
        2,
    )

    upper = np.percentile(
        valid_values,
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


def dsm_to_rgb(
    dsm: np.ndarray,
) -> np.ndarray:
    """Convert normalized DSM into a coloured display."""
    colormap = plt.get_cmap("viridis")

    return colormap(dsm)[:, :, :3]


def label_to_rgb(
    label: np.ndarray,
) -> np.ndarray:
    """
    Convert the label image into RGB.

    RGB labels preserve their original colours.
    Single-band labels use a discrete tab20 colour map.
    """
    if label.ndim == 3 and label.shape[2] >= 3:
        rgb = label[:, :, :3].astype(np.float32)

        maximum = float(np.nanmax(rgb))

        if maximum > 1.0:
            if maximum <= 255:
                rgb = rgb / 255.0
            else:
                rgb = rgb / maximum

        return np.clip(
            rgb,
            0.0,
            1.0,
        )

    if label.ndim == 3:
        label = label[:, :, 0]

    unique_values = np.unique(label)

    class_indices = np.searchsorted(
        unique_values,
        label,
    )

    number_of_classes = max(
        len(unique_values),
        1,
    )

    colormap = plt.get_cmap(
        "tab20",
        number_of_classes,
    )

    denominator = max(
        number_of_classes - 1,
        1,
    )

    return colormap(class_indices / denominator)[:, :, :3]


def calculate_label_boundaries(
    label: np.ndarray,
) -> np.ndarray:
    """Calculate boundaries between neighbouring label classes."""
    if label.ndim == 3 and label.shape[2] >= 3:
        comparable = label[:, :, :3]

        horizontal_change = np.any(
            comparable[:, 1:, :] != comparable[:, :-1, :],
            axis=2,
        )

        vertical_change = np.any(
            comparable[1:, :, :] != comparable[:-1, :, :],
            axis=2,
        )

    else:
        if label.ndim == 3:
            comparable = label[:, :, 0]
        else:
            comparable = label

        horizontal_change = comparable[:, 1:] != comparable[:, :-1]

        vertical_change = comparable[1:, :] != comparable[:-1, :]

    boundaries = np.zeros(
        comparable.shape[:2],
        dtype=bool,
    )

    boundaries[:, 1:] |= horizontal_change
    boundaries[1:, :] |= vertical_change

    return boundaries


# ============================================================
# Interactive viewer
# ============================================================


class AlignmentViewer:
    """Interactive multispectral–DSM–label alignment viewer."""

    def __init__(
        self,
        area_ids: list[str],
        ms_files: dict[str, Path],
        dsm_files: dict[str, Path],
        label_files: dict[str, Path],
        max_display_size: int,
    ) -> None:
        self.area_ids = area_ids
        self.ms_files = ms_files
        self.dsm_files = dsm_files
        self.label_files = label_files
        self.max_display_size = max_display_size

        self.current_index = 0

        self.ms_display: np.ndarray | None = None
        self.dsm_display: np.ndarray | None = None
        self.dsm_rgb: np.ndarray | None = None
        self.label_raw: np.ndarray | None = None
        self.label_rgb: np.ndarray | None = None
        self.label_boundaries: np.ndarray | None = None

        self.dimension_match = False
        self.overlay_artist = None
        self.slice_line = None

        self.figure, self.axes = plt.subplots(
            2,
            2,
            figsize=(16, 11),
        )

        plt.subplots_adjust(
            left=0.04,
            right=0.97,
            top=0.91,
            bottom=0.25,
            hspace=0.15,
            wspace=0.08,
        )

        self.create_widgets()
        self.connect_events()

        self.load_area(0)

    def create_widgets(self) -> None:
        """Create sliders, layer controls, and navigation buttons."""
        wipe_axis = self.figure.add_axes([0.20, 0.175, 0.56, 0.025])

        dsm_alpha_axis = self.figure.add_axes([0.20, 0.125, 0.56, 0.025])

        label_alpha_axis = self.figure.add_axes([0.20, 0.075, 0.56, 0.025])

        check_axis = self.figure.add_axes([0.79, 0.065, 0.18, 0.135])

        previous_axis = self.figure.add_axes([0.03, 0.11, 0.07, 0.05])

        next_axis = self.figure.add_axes([0.11, 0.11, 0.07, 0.05])

        save_axis = self.figure.add_axes([0.03, 0.045, 0.15, 0.05])

        self.wipe_slider = Slider(
            ax=wipe_axis,
            label="Wipe position",
            valmin=0.0,
            valmax=1.0,
            valinit=0.5,
        )

        self.dsm_alpha_slider = Slider(
            ax=dsm_alpha_axis,
            label="DSM opacity",
            valmin=0.0,
            valmax=1.0,
            valinit=0.35,
        )

        self.label_alpha_slider = Slider(
            ax=label_alpha_axis,
            label="Label opacity",
            valmin=0.0,
            valmax=1.0,
            valinit=0.30,
        )

        self.layer_checks = CheckButtons(
            ax=check_axis,
            labels=[
                "Show DSM",
                "Show label",
                "Label edges only",
            ],
            actives=[
                True,
                True,
                False,
            ],
        )

        self.previous_button = Button(
            previous_axis,
            "Previous",
        )

        self.next_button = Button(
            next_axis,
            "Next",
        )

        self.save_button = Button(
            save_axis,
            "Save current view",
        )

    def connect_events(self) -> None:
        """Connect widgets and keyboard controls."""
        self.wipe_slider.on_changed(self.update_overlay)

        self.dsm_alpha_slider.on_changed(self.update_overlay)

        self.label_alpha_slider.on_changed(self.update_overlay)

        self.layer_checks.on_clicked(self.update_overlay)

        self.previous_button.on_clicked(self.previous_area)

        self.next_button.on_clicked(self.next_area)

        self.save_button.on_clicked(self.save_current_view)

        self.figure.canvas.mpl_connect(
            "key_press_event",
            self.on_key_press,
        )

    def previous_area(self, _event=None) -> None:
        """Move to the previous area."""
        new_index = (self.current_index - 1) % len(self.area_ids)

        self.load_area(new_index)

    def next_area(self, _event=None) -> None:
        """Move to the next area."""
        new_index = (self.current_index + 1) % len(self.area_ids)

        self.load_area(new_index)

    def on_key_press(self, event) -> None:
        """Use keyboard arrows to navigate areas."""
        if event.key in {"right", "down"}:
            self.next_area()

        elif event.key in {"left", "up"}:
            self.previous_area()

    def load_area(self, index: int) -> None:
        """Load one complete multispectral–DSM–label triplet."""
        self.current_index = index
        area_id = self.area_ids[index]

        ms_path = self.ms_files[area_id]
        dsm_path = self.dsm_files[area_id]
        label_path = self.label_files[area_id]

        ms_info = inspect_file(ms_path)
        dsm_info = inspect_file(dsm_path)
        label_info = inspect_file(label_path)

        self.dimension_match = dimensions_match(
            ms_info,
            dsm_info,
            label_info,
        )

        for axis in self.axes.flat:
            axis.clear()
            axis.axis("off")

        title = (
            f"{area_id}  |  "
            f"MS: {ms_info['width']}×{ms_info['height']}×"
            f"{ms_info['bands']}  |  "
            f"DSM: {dsm_info['width']}×{dsm_info['height']}×"
            f"{dsm_info['bands']}  |  "
            f"Label: {label_info['width']}×{label_info['height']}×"
            f"{label_info['bands']}  |  "
            f"Dimensions match: {self.dimension_match}"
        )

        self.figure.suptitle(
            title,
            fontsize=13,
        )

        if not self.dimension_match:
            self.display_dimension_mismatch(
                area_id=area_id,
                ms_path=ms_path,
                dsm_path=dsm_path,
                label_path=label_path,
                ms_info=ms_info,
                dsm_info=dsm_info,
                label_info=label_info,
            )

            self.figure.canvas.draw_idle()
            return

        display_width, display_height = calculate_display_size(
            width=ms_info["width"],
            height=ms_info["height"],
            maximum_size=self.max_display_size,
        )

        ms = read_for_display(
            path=ms_path,
            display_width=display_width,
            display_height=display_height,
            nearest_neighbour=False,
        )

        dsm = read_for_display(
            path=dsm_path,
            display_width=display_width,
            display_height=display_height,
            nearest_neighbour=False,
        )

        label = read_for_display(
            path=label_path,
            display_width=display_width,
            display_height=display_height,
            nearest_neighbour=True,
        )

        self.ms_display = stretch_multispectral(ms)

        self.dsm_display = normalize_dsm(dsm)

        self.dsm_rgb = dsm_to_rgb(self.dsm_display)

        self.label_raw = label

        self.label_rgb = label_to_rgb(label)

        self.label_boundaries = calculate_label_boundaries(label)

        self.axes[0, 0].imshow(self.ms_display)

        self.axes[0, 0].set_title(
            "Multispectral image\n" f"{ms_info['width']} × {ms_info['height']} pixels"
        )

        self.axes[0, 1].imshow(
            self.dsm_display,
            cmap="gray",
        )

        self.axes[0, 1].set_title(
            "DSM\n" f"{dsm_info['width']} × {dsm_info['height']} pixels"
        )

        self.axes[1, 0].imshow(
            self.label_rgb,
            interpolation="nearest",
        )

        self.axes[1, 0].set_title(
            "Label\n" f"{label_info['width']} × {label_info['height']} pixels"
        )

        self.overlay_artist = self.axes[1, 1].imshow(self.ms_display)

        self.slice_line = self.axes[1, 1].axvline(
            x=self.ms_display.shape[1] * 0.5,
            linewidth=2,
        )

        self.axes[1, 1].set_title(
            "Interactive wipe comparison\n" "Left: image | Right: selected overlays"
        )

        self.update_overlay()

    def display_dimension_mismatch(
        self,
        area_id: str,
        ms_path: Path,
        dsm_path: Path,
        label_path: Path,
        ms_info: dict[str, Any],
        dsm_info: dict[str, Any],
        label_info: dict[str, Any],
    ) -> None:
        """Display each layer separately when dimensions differ."""
        ms_width, ms_height = calculate_display_size(
            ms_info["width"],
            ms_info["height"],
            self.max_display_size,
        )

        dsm_width, dsm_height = calculate_display_size(
            dsm_info["width"],
            dsm_info["height"],
            self.max_display_size,
        )

        label_width, label_height = calculate_display_size(
            label_info["width"],
            label_info["height"],
            self.max_display_size,
        )

        ms = read_for_display(
            ms_path,
            ms_width,
            ms_height,
            nearest_neighbour=False,
        )

        dsm = read_for_display(
            dsm_path,
            dsm_width,
            dsm_height,
            nearest_neighbour=False,
        )

        label = read_for_display(
            label_path,
            label_width,
            label_height,
            nearest_neighbour=True,
        )

        self.axes[0, 0].imshow(stretch_multispectral(ms))

        self.axes[0, 0].set_title(
            "Multispectral\n" f"{ms_info['width']} × {ms_info['height']}"
        )

        self.axes[0, 1].imshow(
            normalize_dsm(dsm),
            cmap="gray",
        )

        self.axes[0, 1].set_title("DSM\n" f"{dsm_info['width']} × {dsm_info['height']}")

        self.axes[1, 0].imshow(
            label_to_rgb(label),
            interpolation="nearest",
        )

        self.axes[1, 0].set_title(
            "Label\n" f"{label_info['width']} × {label_info['height']}"
        )

        self.axes[1, 1].text(
            0.5,
            0.5,
            (
                f"{area_id}\n\n"
                "Dimensions do not match.\n"
                "Overlay is disabled.\n\n"
                f"MS: {ms_info['width']} × {ms_info['height']}\n"
                f"DSM: {dsm_info['width']} × {dsm_info['height']}\n"
                f"Label: {label_info['width']} × "
                f"{label_info['height']}"
            ),
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=13,
            transform=self.axes[1, 1].transAxes,
        )

        self.axes[1, 1].set_title("Dimension mismatch")

        self.overlay_artist = None
        self.slice_line = None

    def create_composite(self) -> np.ndarray:
        """Combine multispectral, DSM, and label displays."""
        if self.ms_display is None or self.dsm_rgb is None or self.label_rgb is None:
            raise RuntimeError("No area has been loaded.")

        composite = self.ms_display.copy()

        check_status = self.layer_checks.get_status()

        show_dsm = check_status[0]
        show_label = check_status[1]
        edges_only = check_status[2]

        dsm_alpha = float(self.dsm_alpha_slider.val)

        label_alpha = float(self.label_alpha_slider.val)

        if show_dsm:
            composite = composite * (1.0 - dsm_alpha) + self.dsm_rgb * dsm_alpha

        if show_label:
            if edges_only and self.label_boundaries is not None:
                boundary_mask = self.label_boundaries

                composite[boundary_mask] = (
                    composite[boundary_mask] * (1.0 - label_alpha)
                    + self.label_rgb[boundary_mask] * label_alpha
                )

            else:
                composite = (
                    composite * (1.0 - label_alpha) + self.label_rgb * label_alpha
                )

        return np.clip(
            composite,
            0.0,
            1.0,
        )

    def update_overlay(self, _value=None) -> None:
        """Update the wipe slicer and layer opacities."""
        if (
            not self.dimension_match
            or self.overlay_artist is None
            or self.ms_display is None
        ):
            return

        composite = self.create_composite()

        wipe_fraction = float(self.wipe_slider.val)

        width = self.ms_display.shape[1]

        slice_column = int(round(wipe_fraction * width))

        slice_column = int(
            np.clip(
                slice_column,
                0,
                width,
            )
        )

        wipe_image = self.ms_display.copy()

        wipe_image[
            :,
            slice_column:,
            :,
        ] = composite[
            :,
            slice_column:,
            :,
        ]

        self.overlay_artist.set_data(wipe_image)

        if self.slice_line is not None:
            self.slice_line.set_xdata([slice_column, slice_column])

        self.figure.canvas.draw_idle()

    def save_current_view(self, _event=None) -> None:
        """Save the current interactive state as a PNG."""
        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        area_id = self.area_ids[self.current_index]

        output_path = OUTPUT_DIR / f"{area_id}_interactive_view.png"

        self.figure.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight",
        )

        print(f"Saved current view:\n{output_path}")

    def show(self) -> None:
        """Open the interactive viewer."""
        plt.show()


# ============================================================
# Main
# ============================================================


def main() -> None:
    args = parse_arguments()

    print(f"Project root: {PROJECT_ROOT}")

    ms_files = index_folder(
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

    all_area_ids = sorted(
        set(ms_files) | set(dsm_files) | set(label_files),
        key=area_sort_key,
    )

    complete_area_ids = sorted(
        set(ms_files) & set(dsm_files) & set(label_files),
        key=area_sort_key,
    )

    incomplete_area_ids = [
        area_id for area_id in all_area_ids if area_id not in complete_area_ids
    ]

    if incomplete_area_ids:
        print("\nIncomplete areas:")

        for area_id in incomplete_area_ids:
            print(
                f"  {area_id}: "
                f"MS={area_id in ms_files}, "
                f"DSM={area_id in dsm_files}, "
                f"Label={area_id in label_files}"
            )

    if args.areas:
        requested_areas = {area_id.lower() for area_id in args.areas}

        complete_area_ids = [
            area_id
            for area_id in complete_area_ids
            if area_id.lower() in requested_areas
        ]

    if not complete_area_ids:
        raise RuntimeError(
            "No complete multispectral–DSM–label " "triplets were found."
        )

    print(f"\nComplete areas available: " f"{len(complete_area_ids)}")

    print(complete_area_ids)

    viewer = AlignmentViewer(
        area_ids=complete_area_ids,
        ms_files=ms_files,
        dsm_files=dsm_files,
        label_files=label_files,
        max_display_size=args.max_display_size,
    )

    viewer.show()


if __name__ == "__main__":
    main()
