"""Evaluate a trained model on held-out areas and save metrics and figures.

Run: ``python src/evaluate_test_set.py --help``
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch

from dataset import (
    PROJECT_ROOT,
    create_dataloaders,
)

from evaluate import (
    create_per_area_row,
    create_per_class_rows,
    evaluate_class_masks,
    save_confusion_matrix_figure,
    save_csv_rows,
    save_json,
    save_prediction_comparison,
    save_rgb_mask,
)

from inference import (
    reconstruct_test_predictions,
)

from metrics import (
    calculate_metrics,
    create_confusion_matrix,
)

from model_utils import (
    create_unet_resnet18,
    get_device,
)


NUMBER_OF_CLASSES = 6
IGNORE_INDEX = 255

INPUT_CHANNELS = {
    "multispectral": 3,
    "early_fusion": 4,
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained segmentation model on " "complete test images."
        )
    )

    parser.add_argument(
        "--experiment",
        choices=[
            "multispectral",
            "early_fusion",
        ],
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help=("Path to best_model_state_dict.pt or a " "training checkpoint."),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
    )

    return parser.parse_args()


def resolve_project_path(
    value: str,
) -> Path:
    path = Path(value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def load_dataset_configuration() -> tuple[
    dict[int, str],
    dict[int, tuple[int, int, int]],
]:
    config_path = PROJECT_ROOT / "data" / "data-config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration not found:\n" f"{config_path}")

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        configuration = json.load(file)

    class_names = {
        int(class_id): class_name
        for class_id, class_name in configuration["Class Names"].items()
    }

    class_colors = {
        int(class_id): tuple(int(value) for value in color)
        for class_id, color in configuration["Class Color Codes"].items()
    }

    return (
        class_names,
        class_colors,
    )


def extract_area_number(
    path: Path,
) -> int:
    match = re.search(
        r"area(\d+)",
        path.stem,
        flags=re.IGNORECASE,
    )

    if match is None:
        raise ValueError(f"Could not identify area number from: " f"{path.name}")

    return int(match.group(1))


def build_label_file_index() -> dict[
    int,
    Path,
]:
    label_directory = PROJECT_ROOT / "data" / "labels"

    label_files = sorted(
        list(label_directory.glob("*.tif")) + list(label_directory.glob("*.tiff"))
    )

    if not label_files:
        raise FileNotFoundError(f"No label TIFF files found in:\n" f"{label_directory}")

    index: dict[int, Path] = {}

    for path in label_files:
        area_id = extract_area_number(path)

        index[area_id] = path

    return index


def read_rgb_label(
    path: Path,
) -> np.ndarray:
    with rasterio.open(path) as dataset:
        if dataset.count < 3:
            raise ValueError(f"Expected a three-band RGB label: " f"{path}")

        rgb = dataset.read([1, 2, 3])

    return np.transpose(
        rgb,
        (
            1,
            2,
            0,
        ),
    ).astype(
        np.uint8,
        copy=False,
    )


def rgb_to_class_mask(
    rgb: np.ndarray,
    class_colors: dict[
        int,
        tuple[int, int, int],
    ],
) -> np.ndarray:
    class_mask = np.full(
        rgb.shape[:2],
        fill_value=IGNORE_INDEX,
        dtype=np.uint8,
    )

    for class_id, color in class_colors.items():
        color_array = np.asarray(
            color,
            dtype=np.uint8,
        )

        matching_pixels = np.all(
            rgb == color_array,
            axis=-1,
        )

        class_mask[matching_pixels] = class_id

    unknown_pixels = int(np.count_nonzero(class_mask == IGNORE_INDEX))

    if unknown_pixels > 0:
        raise ValueError(
            f"Label contains {unknown_pixels} " "pixels with unknown RGB colours."
        )

    return class_mask


def extract_state_dictionary(
    checkpoint: Any,
) -> dict[str, torch.Tensor]:
    if not isinstance(
        checkpoint,
        dict,
    ):
        raise TypeError("The loaded checkpoint is not a dictionary.")

    if checkpoint and all(
        isinstance(value, torch.Tensor) for value in checkpoint.values()
    ):
        return checkpoint

    for key in [
        "model_state_dict",
        "state_dict",
    ]:
        value = checkpoint.get(key)

        if (
            isinstance(value, dict)
            and value
            and all(
                isinstance(
                    tensor,
                    torch.Tensor,
                )
                for tensor in value.values()
            )
        ):
            return value

    raise RuntimeError(
        "Could not identify a model state dictionary " "inside the checkpoint."
    )


def load_model(
    experiment: str,
    checkpoint_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found:\n" f"{checkpoint_path}")

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

    except Exception:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

    state_dictionary = extract_state_dictionary(checkpoint)

    model = create_unet_resnet18(
        input_channels=(INPUT_CHANNELS[experiment]),
        number_of_classes=(NUMBER_OF_CLASSES),
    )

    model.load_state_dict(
        state_dictionary,
        strict=True,
    )

    model = model.to(device)

    model.eval()

    return model


def main() -> None:
    arguments = parse_arguments()

    checkpoint_path = resolve_project_path(arguments.checkpoint)

    if arguments.output_dir is None:
        output_directory = (
            PROJECT_ROOT / "outputs" / "test_evaluation" / arguments.experiment
        )

    else:
        output_directory = resolve_project_path(arguments.output_dir)

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_directory = output_directory / "predictions"

    ground_truth_directory = output_directory / "ground_truth"

    comparison_directory = output_directory / "comparisons"

    confusion_directory = output_directory / "confusion_matrices"

    class_names, class_colors = load_dataset_configuration()

    label_index = build_label_file_index()

    dataloaders = create_dataloaders(
        input_mode=(arguments.experiment),
        batch_size=(arguments.batch_size),
        num_workers=(arguments.num_workers),
        project_root=PROJECT_ROOT,
    )

    config_path = PROJECT_ROOT / "data" / "data-config.json"

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        dataset_config = json.load(file)

    test_area_ids = sorted(
        int(area_id) for area_id in dataset_config["Image IDs"]["Test Image IDs"]
    )

    area_shapes: dict[
        int,
        tuple[int, int],
    ] = {}

    ground_truth_masks: dict[
        int,
        np.ndarray,
    ] = {}

    for area_id in test_area_ids:
        if area_id not in label_index:
            raise FileNotFoundError(f"No label file found for " f"test area {area_id}.")

        rgb_label = read_rgb_label(label_index[area_id])

        class_mask = rgb_to_class_mask(
            rgb=rgb_label,
            class_colors=class_colors,
        )

        ground_truth_masks[area_id] = class_mask

        area_shapes[area_id] = class_mask.shape

    device = get_device()

    print("=" * 90)
    print("COMPLETE TEST-SET EVALUATION")
    print("=" * 90)

    print(f"\nExperiment: " f"{arguments.experiment}")

    print(f"Checkpoint:\n" f"{checkpoint_path}")

    print(f"Device: {device}")

    print(f"Test areas: {test_area_ids}")

    model = load_model(
        experiment=(arguments.experiment),
        checkpoint_path=(checkpoint_path),
        device=device,
    )

    predictions, coverage = reconstruct_test_predictions(
        model=model,
        dataloader=(dataloaders["test"]),
        device=device,
        area_shapes=area_shapes,
        number_of_classes=(NUMBER_OF_CLASSES),
    )

    per_area_rows = []
    per_class_rows = []

    global_confusion_matrix = create_confusion_matrix(NUMBER_OF_CLASSES)

    area_results: dict[
        str,
        Any,
    ] = {}

    for area_id in test_area_ids:
        prediction = predictions[area_id]

        target = ground_truth_masks[area_id]

        (
            area_metrics,
            area_confusion_matrix,
        ) = evaluate_class_masks(
            prediction=prediction,
            target=target,
            number_of_classes=(NUMBER_OF_CLASSES),
            ignore_index=IGNORE_INDEX,
        )

        global_confusion_matrix += area_confusion_matrix

        per_area_rows.append(
            create_per_area_row(
                area_id=area_id,
                metrics=area_metrics,
            )
        )

        per_class_rows.extend(
            create_per_class_rows(
                area_id=area_id,
                metrics=area_metrics,
                class_names=class_names,
            )
        )

        area_results[str(area_id)] = area_metrics

        save_rgb_mask(
            class_mask=prediction,
            class_colors=class_colors,
            output_path=(predictions_directory / (f"area{area_id}_" f"prediction.png")),
        )

        save_rgb_mask(
            class_mask=target,
            class_colors=class_colors,
            output_path=(
                ground_truth_directory / (f"area{area_id}_" f"ground_truth.png")
            ),
        )

        save_prediction_comparison(
            target=target,
            prediction=prediction,
            class_colors=class_colors,
            output_path=(comparison_directory / (f"area{area_id}_" f"comparison.png")),
            title=(f"Area {area_id} — " f"{arguments.experiment}"),
        )

        save_confusion_matrix_figure(
            confusion_matrix=(area_confusion_matrix),
            class_names=class_names,
            output_path=(
                confusion_directory / (f"area{area_id}_" f"confusion_matrix.png")
            ),
            title=(f"Area {area_id} confusion matrix"),
        )

        minimum_coverage = float(coverage[area_id].min())

        maximum_coverage = float(coverage[area_id].max())

        print(f"\nArea {area_id}")

        print(f"  OA: " f"{area_metrics['overall_accuracy']:.4f}")

        print(f"  Macro precision: " f"{area_metrics['macro_precision']:.4f}")

        print(f"  Macro recall: " f"{area_metrics['macro_recall']:.4f}")

        print(f"  Macro F1: " f"{area_metrics['macro_f1']:.4f}")

        print(f"  mIoU: " f"{area_metrics['mean_iou']:.4f}")

        print(
            f"  Patch coverage: " f"{minimum_coverage:.0f}–" f"{maximum_coverage:.0f}"
        )

    global_metrics = calculate_metrics(global_confusion_matrix)

    global_metrics["confusion_matrix"] = global_confusion_matrix.tolist()

    per_class_rows.extend(
        create_per_class_rows(
            area_id="overall",
            metrics=global_metrics,
            class_names=class_names,
        )
    )

    save_csv_rows(
        rows=per_area_rows,
        output_path=(output_directory / "per_area_metrics.csv"),
    )

    save_csv_rows(
        rows=per_class_rows,
        output_path=(output_directory / "per_class_metrics.csv"),
    )

    save_confusion_matrix_figure(
        confusion_matrix=(global_confusion_matrix),
        class_names=class_names,
        output_path=(output_directory / "overall_confusion_matrix.png"),
        title=(f"Overall test confusion matrix — " f"{arguments.experiment}"),
    )

    results = {
        "experiment": (arguments.experiment),
        "checkpoint": str(checkpoint_path),
        "test_areas": (test_area_ids),
        "class_names": (class_names),
        "per_area": (area_results),
        "overall": (global_metrics),
    }

    save_json(
        data=results,
        output_path=(output_directory / "test_results.json"),
    )

    print("\n" + "=" * 90)
    print("OVERALL TEST RESULTS")
    print("=" * 90)

    print(f"\nOverall accuracy: " f"{global_metrics['overall_accuracy']:.4f}")

    print(f"Macro precision: " f"{global_metrics['macro_precision']:.4f}")

    print(f"Macro recall: " f"{global_metrics['macro_recall']:.4f}")

    print(f"Macro F1: " f"{global_metrics['macro_f1']:.4f}")

    print(f"Mean IoU: " f"{global_metrics['mean_iou']:.4f}")

    print(f"\nResults saved to:\n" f"{output_directory}")


if __name__ == "__main__":
    main()
