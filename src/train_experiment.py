"""Train or fine-tune a U-Net--ResNet18 semantic-segmentation model.

Run: ``python src/train_experiment.py --help``
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from dataset import (
    PROJECT_ROOT,
    create_dataloaders,
)

from evaluate import (
    evaluate_model,
)

from losses import (
    DEFAULT_IGNORE_INDEX,
    create_segmentation_loss,
)

from model_utils import (
    count_parameters,
    create_experiment_model,
    create_unet_resnet18,
    get_device,
)

from training import (
    train_one_epoch,
)


NUMBER_OF_CLASSES = 6

EXPERIMENT_CHANNELS = {
    "multispectral": 3,
    "early_fusion": 4,
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and validate a U-Net ResNet18 " "semantic-segmentation model."
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
        "--initialization",
        choices=[
            "scratch",
            "checkpoint",
        ],
        default="scratch",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--accumulation-steps",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
    )

    return parser.parse_args()


def set_random_seed(
    seed: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def save_history_csv(
    history: list[dict[str, Any]],
    path: Path,
) -> None:
    if not history:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(history[0].keys()),
        )

        writer.writeheader()
        writer.writerows(history)


def save_loss_curve(
    history: list[dict[str, Any]],
    output_path: Path,
) -> None:
    epochs = [row["epoch"] for row in history]

    training_loss = [row["train_loss"] for row in history]

    validation_loss = [row["validation_loss"] for row in history]

    figure = plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        training_loss,
        marker="o",
        label="Training loss",
    )

    plt.plot(
        epochs,
        validation_loss,
        marker="o",
        label="Validation loss",
    )

    plt.xlabel("Epoch")

    plt.ylabel("Cross-entropy loss")

    plt.title("Training and validation loss")

    plt.grid(alpha=0.3)

    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_f1_curve(
    history: list[dict[str, Any]],
    output_path: Path,
) -> None:
    epochs = [row["epoch"] for row in history]

    training_f1 = [row["train_macro_f1"] for row in history]

    validation_f1 = [row["validation_macro_f1"] for row in history]

    figure = plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        training_f1,
        marker="o",
        label="Training macro F1",
    )

    plt.plot(
        epochs,
        validation_f1,
        marker="o",
        label="Validation macro F1",
    )

    plt.xlabel("Epoch")

    plt.ylabel("Macro F1")

    plt.ylim(
        0.0,
        1.0,
    )

    plt.title("Training and validation macro F1")

    plt.grid(alpha=0.3)

    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_accuracy_curve(
    history: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Save training and validation overall-accuracy curves.
    """
    if not history:
        return

    epochs = [row["epoch"] for row in history]

    training_accuracy = [row["train_overall_accuracy"] for row in history]

    validation_accuracy = [row["validation_overall_accuracy"] for row in history]

    figure = plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        training_accuracy,
        marker="o",
        label="Training OA",
    )

    plt.plot(
        epochs,
        validation_accuracy,
        marker="o",
        label="Validation OA",
    )

    plt.xlabel("Epoch")

    plt.ylabel("Overall accuracy")

    plt.ylim(
        0.0,
        1.0,
    )

    plt.title("Training and validation overall accuracy")

    plt.grid(alpha=0.3)

    plt.legend()

    plt.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def initialize_model(
    experiment: str,
    initialization: str,
    device: torch.device,
) -> nn.Module:
    input_channels = EXPERIMENT_CHANNELS[experiment]

    if initialization == "scratch":
        model = create_unet_resnet18(
            input_channels=input_channels,
            number_of_classes=(NUMBER_OF_CLASSES),
        )

        return model.to(device)

    return create_experiment_model(
        experiment=experiment,
        device=device,
    )


def move_optimizer_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    """
    Move optimizer-state tensors to the model device
    after resuming training.
    """
    for optimizer_state in optimizer.state.values():
        for key, value in optimizer_state.items():
            if isinstance(
                value,
                torch.Tensor,
            ):
                optimizer_state[key] = value.to(device)


def load_resume_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[
    int,
    float,
    list[dict[str, Any]],
]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found:\n" f"{checkpoint_path}")

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    move_optimizer_to_device(
        optimizer=optimizer,
        device=device,
    )

    start_epoch = int(checkpoint["epoch"]) + 1

    best_validation_loss = float(checkpoint["best_validation_loss"])

    history = checkpoint.get(
        "history",
        [],
    )

    return (
        start_epoch,
        best_validation_loss,
        history,
    )


def main() -> None:
    arguments = parse_arguments()

    if arguments.epochs <= 0:
        raise ValueError("epochs must be greater than zero.")

    if arguments.batch_size <= 0:
        raise ValueError("batch-size must be greater than zero.")

    set_random_seed(arguments.seed)

    device = get_device()

    if arguments.output_dir is None:
        output_directory = PROJECT_ROOT / "outputs" / "training" / arguments.experiment

    else:
        output_directory = Path(arguments.output_dir)

        if not output_directory.is_absolute():
            output_directory = PROJECT_ROOT / output_directory

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_path = output_directory / "history.csv"

    loss_curve_path = output_directory / "loss_curve.png"

    f1_curve_path = output_directory / "f1_curve.png"

    accuracy_curve_path = output_directory / "overall_accuracy_curve.png"

    best_model_path = output_directory / "best_model_state_dict.pt"

    best_checkpoint_path = output_directory / "best_training_checkpoint.pt"

    latest_checkpoint_path = output_directory / "latest_training_checkpoint.pt"

    summary_path = output_directory / "training_summary.json"

    print("=" * 85)
    print("SEMANTIC SEGMENTATION TRAINING")
    print("=" * 85)

    print(f"\nExperiment: " f"{arguments.experiment}")

    print(f"Initialization: " f"{arguments.initialization}")

    print(f"Device: {device}")

    print(f"Epochs: " f"{arguments.epochs}")

    print(f"Batch size: " f"{arguments.batch_size}")

    print(f"Learning rate: " f"{arguments.learning_rate}")

    dataloaders = create_dataloaders(
        input_mode=(arguments.experiment),
        batch_size=(arguments.batch_size),
        num_workers=(arguments.num_workers),
        project_root=PROJECT_ROOT,
    )

    model = initialize_model(
        experiment=(arguments.experiment),
        initialization=(arguments.initialization),
        device=device,
    )

    (
        total_parameters,
        trainable_parameters,
    ) = count_parameters(model)

    print(f"\nTraining patches: " f"{len(dataloaders['train'].dataset)}")

    print(f"Validation patches: " f"{len(dataloaders['validation'].dataset)}")

    print(f"Total parameters: " f"{total_parameters:,}")

    criterion = create_segmentation_loss(
        ignore_index=(DEFAULT_IGNORE_INDEX),
        class_weights=None,
        device=device,
    )

    optimizer = Adam(
        model.parameters(),
        lr=arguments.learning_rate,
    )

    start_epoch = 1
    best_validation_loss = float("inf")

    history: list[dict[str, Any]] = []

    if arguments.resume is not None:
        resume_path = Path(arguments.resume)

        if not resume_path.is_absolute():
            resume_path = PROJECT_ROOT / resume_path

        (
            start_epoch,
            best_validation_loss,
            history,
        ) = load_resume_checkpoint(
            checkpoint_path=(resume_path),
            model=model,
            optimizer=optimizer,
            device=device,
        )

    training_start = time.time()

    for epoch in range(
        start_epoch,
        arguments.epochs + 1,
    ):
        epoch_start = time.time()

        training_metrics = train_one_epoch(
            model=model,
            dataloader=(dataloaders["train"]),
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            number_of_classes=(NUMBER_OF_CLASSES),
            ignore_index=(DEFAULT_IGNORE_INDEX),
            accumulation_steps=(arguments.accumulation_steps),
            gradient_clip=(arguments.gradient_clip),
        )

        validation_metrics = evaluate_model(
            model=model,
            dataloader=(dataloaders["validation"]),
            criterion=criterion,
            device=device,
            number_of_classes=(NUMBER_OF_CLASSES),
            ignore_index=(DEFAULT_IGNORE_INDEX),
        )

        epoch_seconds = time.time() - epoch_start

        history_row = {
            "epoch": epoch,
            "train_loss": (training_metrics["loss"]),
            "validation_loss": (validation_metrics["loss"]),
            "train_overall_accuracy": (training_metrics["overall_accuracy"]),
            "validation_overall_accuracy": (validation_metrics["overall_accuracy"]),
            "train_macro_precision": (training_metrics["macro_precision"]),
            "validation_macro_precision": (validation_metrics["macro_precision"]),
            "train_macro_recall": (training_metrics["macro_recall"]),
            "validation_macro_recall": (validation_metrics["macro_recall"]),
            "train_macro_f1": (training_metrics["macro_f1"]),
            "validation_macro_f1": (validation_metrics["macro_f1"]),
            "train_mean_iou": (training_metrics["mean_iou"]),
            "validation_mean_iou": (validation_metrics["mean_iou"]),
            "epoch_seconds": (epoch_seconds),
        }

        history.append(history_row)

        improved = validation_metrics["loss"] < best_validation_loss

        if improved:
            best_validation_loss = validation_metrics["loss"]

        checkpoint_payload = {
            "epoch": epoch,
            "experiment": (arguments.experiment),
            "initialization": (arguments.initialization),
            "input_channels": (EXPERIMENT_CHANNELS[arguments.experiment]),
            "number_of_classes": (NUMBER_OF_CLASSES),
            "model_state_dict": (model.state_dict()),
            "optimizer_state_dict": (optimizer.state_dict()),
            "best_validation_loss": (best_validation_loss),
            "training_metrics": (training_metrics),
            "validation_metrics": (validation_metrics),
            "history": history,
            "configuration": vars(arguments),
        }

        torch.save(
            checkpoint_payload,
            latest_checkpoint_path,
        )

        if improved:
            torch.save(
                model.state_dict(),
                best_model_path,
            )

            torch.save(
                checkpoint_payload,
                best_checkpoint_path,
            )

        save_history_csv(
            history=history,
            path=history_path,
        )

        save_loss_curve(history=history, output_path=(loss_curve_path))

        save_f1_curve(history=history, output_path=(f1_curve_path))
        save_accuracy_curve(
            history=history,
            output_path=accuracy_curve_path,
        )

        print(f"\nEpoch {epoch:02d}/" f"{arguments.epochs:02d}")

        print(f"  Train loss: " f"{training_metrics['loss']:.6f}")

        print(f"  Validation loss: " f"{validation_metrics['loss']:.6f}")

        print(f"  Validation OA: " f"{validation_metrics['overall_accuracy']:.4f}")

        print(f"  Validation macro F1: " f"{validation_metrics['macro_f1']:.4f}")

        print(f"  Validation mIoU: " f"{validation_metrics['mean_iou']:.4f}")

        print(f"  Time: " f"{epoch_seconds:.1f} seconds")

        if improved:
            print("  Saved new best model.")

        if device.type == "mps":
            torch.mps.empty_cache()

        elif device.type == "cuda":
            torch.cuda.empty_cache()

    total_training_seconds = time.time() - training_start

    best_history_row = min(
        history,
        key=lambda row: (row["validation_loss"]),
    )

    summary = {
        "experiment": (arguments.experiment),
        "initialization": (arguments.initialization),
        "input_channels": (EXPERIMENT_CHANNELS[arguments.experiment]),
        "number_of_classes": (NUMBER_OF_CLASSES),
        "epochs": arguments.epochs,
        "batch_size": (arguments.batch_size),
        "learning_rate": (arguments.learning_rate),
        "optimizer": "Adam",
        "loss": ("CrossEntropyLoss"),
        "device": str(device),
        "total_parameters": (total_parameters),
        "trainable_parameters": (trainable_parameters),
        "best_epoch": (best_history_row["epoch"]),
        "best_validation_loss": (best_history_row["validation_loss"]),
        "best_validation_accuracy": (best_history_row["validation_overall_accuracy"]),
        "best_validation_macro_f1": (best_history_row["validation_macro_f1"]),
        "best_validation_mean_iou": (best_history_row["validation_mean_iou"]),
        "total_training_seconds": (total_training_seconds),
        "best_model_path": str(best_model_path),
    }

    save_json(
        data=summary,
        path=summary_path,
    )

    print("\n" + "=" * 85)
    print("TRAINING COMPLETE")
    print("=" * 85)

    print(f"\nBest epoch: " f"{best_history_row['epoch']}")

    print(f"Best validation loss: " f"{best_history_row['validation_loss']:.6f}")

    print(
        f"Best validation macro F1: " f"{best_history_row['validation_macro_f1']:.4f}"
    )

    print(f"Best validation mIoU: " f"{best_history_row['validation_mean_iou']:.4f}")

    print(f"\nBest model:\n" f"{best_model_path}")


if __name__ == "__main__":
    main()
