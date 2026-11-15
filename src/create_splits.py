"""Create non-overlapping area-level train, validation, and test splits.

Run: ``python src/create_splits.py``
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "data" / "data-config.json"

AUDIT_PATH = PROJECT_ROOT / "metadata" / "data_audit.csv"

METADATA_DIR = PROJECT_ROOT / "metadata"

TRAIN_OUTPUT = METADATA_DIR / "train_areas.txt"

VALIDATION_OUTPUT = METADATA_DIR / "validation_areas.txt"

TEST_OUTPUT = METADATA_DIR / "test_areas.txt"

SPLIT_SUMMARY_OUTPUT = METADATA_DIR / "split_summary.json"

CLASS_MAPPING_OUTPUT = METADATA_DIR / "class_mapping.json"


# ============================================================
# Command-line arguments
# ============================================================


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Create area-level training, validation, " "and test splits.")
    )

    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.20,
        help=(
            "Fraction of provided training areas reserved "
            "for validation. Default: 0.20."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed used for reproducible validation " "selection. Default: 42."
        ),
    )

    parser.add_argument(
        "--validation-ids",
        nargs="*",
        type=int,
        default=None,
        help=(
            "Optional validation IDs selected manually. "
            "Example: --validation-ids 13 23"
        ),
    )

    return parser.parse_args()


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


def save_json(
    data: dict[str, Any],
    path: Path,
) -> None:
    """Save a dictionary as formatted JSON."""
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


def normalize_area_id(
    image_id: int | str,
) -> str:
    """
    Convert an image ID into the standard area format.

    Examples
    --------
    1 -> area1
    "11" -> area11
    "area28" -> area28
    """
    value = str(image_id).strip().lower()

    if value.startswith("area"):
        number = value.replace(
            "area",
            "",
            1,
        )
    else:
        number = value

    if not number.isdigit():
        raise ValueError(f"Invalid image ID: {image_id}")

    return f"area{int(number)}"


def area_sort_key(area_id: str) -> int:
    """Return numeric area number for sorting."""
    return int(
        area_id.lower().replace(
            "area",
            "",
        )
    )


def write_area_file(
    area_ids: list[str],
    output_path: Path,
) -> None:
    """Write one area ID per line."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    text = "\n".join(area_ids) + "\n"

    output_path.write_text(
        text,
        encoding="utf-8",
    )


# ============================================================
# Config parsing
# ============================================================


def parse_config(
    config: dict[str, Any],
) -> tuple[
    list[str],
    list[str],
    dict[int, str],
    dict[int, list[int]],
]:
    """
    Extract image splits and class definitions from data-config.json.
    """
    image_ids = config.get(
        "Image IDs",
    )

    if image_ids is None:
        raise KeyError("Missing 'Image IDs' section in data-config.json")

    train_ids_raw = image_ids.get(
        "Train Image IDs",
    )

    test_ids_raw = image_ids.get(
        "Test Image IDs",
    )

    if train_ids_raw is None:
        raise KeyError("Missing 'Train Image IDs' in data-config.json")

    if test_ids_raw is None:
        raise KeyError("Missing 'Test Image IDs' in data-config.json")

    train_areas = [normalize_area_id(image_id) for image_id in train_ids_raw]

    test_areas = [normalize_area_id(image_id) for image_id in test_ids_raw]

    class_names_raw = config.get(
        "Class Names",
        {},
    )

    color_codes_raw = config.get(
        "Class Color Codes",
        {},
    )

    class_names = {
        int(class_id): class_name for class_id, class_name in class_names_raw.items()
    }

    class_colors = {
        int(class_id): [int(value) for value in color]
        for class_id, color in color_codes_raw.items()
    }

    return (
        train_areas,
        test_areas,
        class_names,
        class_colors,
    )


# ============================================================
# Audit validation
# ============================================================


def load_ready_areas(
    audit_path: Path,
) -> set[str]:
    """
    Return all areas marked ready in data_audit.csv.
    """
    if not audit_path.exists():
        raise FileNotFoundError(
            f"Audit file not found: {audit_path}\n" "Run src/audit_data.py first."
        )

    audit_dataframe = pd.read_csv(audit_path)

    if "area_id" not in audit_dataframe.columns:
        raise KeyError("data_audit.csv does not contain 'area_id'.")

    if "status" not in audit_dataframe.columns:
        raise KeyError("data_audit.csv does not contain 'status'.")

    ready_rows = audit_dataframe[
        audit_dataframe["status"].astype(str).str.lower().eq("ready")
    ]

    return set(ready_rows["area_id"].astype(str).str.lower().tolist())


def validate_configured_areas(
    configured_areas: list[str],
    ready_areas: set[str],
    split_name: str,
) -> None:
    """
    Ensure every configured area is marked ready.
    """
    unavailable_areas = [
        area_id for area_id in configured_areas if area_id.lower() not in ready_areas
    ]

    if unavailable_areas:
        raise RuntimeError(
            f"The following {split_name} areas are not marked "
            f"'ready' in data_audit.csv:\n"
            f"{unavailable_areas}"
        )


# ============================================================
# Validation selection
# ============================================================


def create_validation_split(
    provided_train_areas: list[str],
    validation_fraction: float,
    seed: int,
    manual_validation_ids: list[int] | None,
) -> tuple[list[str], list[str]]:
    """
    Split provided training areas into training and validation sets.
    """
    if manual_validation_ids:
        validation_areas = [
            normalize_area_id(image_id) for image_id in manual_validation_ids
        ]

        invalid_validation_areas = [
            area_id
            for area_id in validation_areas
            if area_id not in provided_train_areas
        ]

        if invalid_validation_areas:
            raise ValueError(
                "Manual validation areas must come from the "
                "provided training IDs.\n"
                f"Invalid IDs: {invalid_validation_areas}"
            )

    else:
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1.")

        number_of_validation_areas = max(
            1,
            round(len(provided_train_areas) * validation_fraction),
        )

        if number_of_validation_areas >= len(provided_train_areas):
            raise ValueError("Validation split would leave no training areas.")

        random_generator = random.Random(seed)

        validation_areas = random_generator.sample(
            provided_train_areas,
            number_of_validation_areas,
        )

    validation_areas = sorted(
        set(validation_areas),
        key=area_sort_key,
    )

    training_areas = [
        area_id for area_id in provided_train_areas if area_id not in validation_areas
    ]

    training_areas = sorted(
        training_areas,
        key=area_sort_key,
    )

    return (
        training_areas,
        validation_areas,
    )


# ============================================================
# Class mapping
# ============================================================


def create_class_mapping(
    class_names: dict[int, str],
    class_colors: dict[int, list[int]],
) -> dict[str, Any]:
    """
    Create class information for later Dataset and visualization code.
    """
    class_ids = sorted(class_names.keys())

    if class_ids != list(range(len(class_ids))):
        raise ValueError("Class IDs must be consecutive and begin at 0.")

    missing_colors = [
        class_id for class_id in class_ids if class_id not in class_colors
    ]

    if missing_colors:
        raise ValueError(f"Missing colours for classes: {missing_colors}")

    classes = []

    for class_id in class_ids:
        classes.append(
            {
                "id": class_id,
                "name": class_names[class_id],
                "color": class_colors[class_id],
            }
        )

    color_to_class = {
        ",".join(str(value) for value in class_colors[class_id]): class_id
        for class_id in class_ids
    }

    return {
        "number_of_classes": len(class_ids),
        "ignore_index": 255,
        "classes": classes,
        "color_to_class": color_to_class,
    }


# ============================================================
# Split checks
# ============================================================


def verify_no_overlap(
    train_areas: list[str],
    validation_areas: list[str],
    test_areas: list[str],
) -> None:
    """Ensure all three splits are mutually exclusive."""
    train_set = set(train_areas)
    validation_set = set(validation_areas)
    test_set = set(test_areas)

    train_validation_overlap = train_set & validation_set

    train_test_overlap = train_set & test_set

    validation_test_overlap = validation_set & test_set

    if train_validation_overlap:
        raise RuntimeError(
            "Training-validation overlap detected: "
            f"{sorted(train_validation_overlap)}"
        )

    if train_test_overlap:
        raise RuntimeError(
            "Training-test overlap detected: " f"{sorted(train_test_overlap)}"
        )

    if validation_test_overlap:
        raise RuntimeError(
            "Validation-test overlap detected: " f"{sorted(validation_test_overlap)}"
        )


# ============================================================
# Main
# ============================================================


def main() -> None:
    arguments = parse_arguments()

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Config path: {CONFIG_PATH}")
    print(f"Audit path: {AUDIT_PATH}")

    config = load_json(CONFIG_PATH)

    (
        provided_train_areas,
        test_areas,
        class_names,
        class_colors,
    ) = parse_config(config)

    ready_areas = load_ready_areas(AUDIT_PATH)

    validate_configured_areas(
        configured_areas=provided_train_areas,
        ready_areas=ready_areas,
        split_name="provided training",
    )

    validate_configured_areas(
        configured_areas=test_areas,
        ready_areas=ready_areas,
        split_name="test",
    )

    train_areas, validation_areas = create_validation_split(
        provided_train_areas=provided_train_areas,
        validation_fraction=(arguments.validation_fraction),
        seed=arguments.seed,
        manual_validation_ids=(arguments.validation_ids),
    )

    test_areas = sorted(
        test_areas,
        key=area_sort_key,
    )

    verify_no_overlap(
        train_areas=train_areas,
        validation_areas=validation_areas,
        test_areas=test_areas,
    )

    class_mapping = create_class_mapping(
        class_names=class_names,
        class_colors=class_colors,
    )

    write_area_file(
        area_ids=train_areas,
        output_path=TRAIN_OUTPUT,
    )

    write_area_file(
        area_ids=validation_areas,
        output_path=VALIDATION_OUTPUT,
    )

    write_area_file(
        area_ids=test_areas,
        output_path=TEST_OUTPUT,
    )

    split_summary = {
        "source_config": str(CONFIG_PATH.relative_to(PROJECT_ROOT)),
        "random_seed": arguments.seed,
        "validation_fraction": (arguments.validation_fraction),
        "provided_training_areas": (provided_train_areas),
        "training_areas": train_areas,
        "validation_areas": (validation_areas),
        "test_areas": test_areas,
        "counts": {
            "provided_training": len(provided_train_areas),
            "training": len(train_areas),
            "validation": len(validation_areas),
            "test": len(test_areas),
        },
    }

    save_json(
        data=split_summary,
        path=SPLIT_SUMMARY_OUTPUT,
    )

    save_json(
        data=class_mapping,
        path=CLASS_MAPPING_OUTPUT,
    )

    print("\n" + "=" * 70)
    print("DATASET SPLITS CREATED")
    print("=" * 70)

    print(f"\nProvided training areas " f"({len(provided_train_areas)}):")
    print(provided_train_areas)

    print(f"\nFinal training areas " f"({len(train_areas)}):")
    print(train_areas)

    print(f"\nValidation areas " f"({len(validation_areas)}):")
    print(validation_areas)

    print(f"\nTest areas " f"({len(test_areas)}):")
    print(test_areas)

    print(f"\nNumber of classes: " f"{class_mapping['number_of_classes']}")

    print("\nClasses:")

    for class_information in class_mapping["classes"]:
        print(
            f"  {class_information['id']}: "
            f"{class_information['name']} "
            f"{class_information['color']}"
        )

    print("\nFiles created:")

    for output_path in [
        TRAIN_OUTPUT,
        VALIDATION_OUTPUT,
        TEST_OUTPUT,
        SPLIT_SUMMARY_OUTPUT,
        CLASS_MAPPING_OUTPUT,
    ]:
        print(f"  {output_path}")

    print(
        "\nTest areas remain completely separate " "from model training and validation."
    )


if __name__ == "__main__":
    main()
