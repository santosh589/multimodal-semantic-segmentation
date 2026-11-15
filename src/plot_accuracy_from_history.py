"""Plot loss, accuracy, and F1 curves from a saved training-history CSV.

Run: ``python src/plot_accuracy_from_history.py --help``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate training and validation overall-accuracy "
            "curves from an existing history.csv file."
        )
    )

    parser.add_argument(
        "--history",
        type=str,
        required=True,
        help="Path to the training history.csv file.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Optional output path. By default, the figure is "
            "saved beside history.csv."
        ),
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    history_path = Path(arguments.history)

    if not history_path.is_absolute():
        history_path = PROJECT_ROOT / history_path

    history_path = history_path.resolve()

    if not history_path.exists():
        raise FileNotFoundError(f"History file not found:\n{history_path}")

    history = pd.read_csv(history_path)

    required_columns = {
        "epoch",
        "train_overall_accuracy",
        "validation_overall_accuracy",
    }

    missing_columns = required_columns - set(history.columns)

    if missing_columns:
        raise KeyError(
            "The history file is missing columns: " f"{sorted(missing_columns)}"
        )

    if arguments.output is None:
        output_path = history_path.parent / "overall_accuracy_curve.png"

    else:
        output_path = Path(arguments.output)

        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path

        output_path = output_path.resolve()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure = plt.figure(figsize=(8, 5))

    plt.plot(
        history["epoch"],
        history["train_overall_accuracy"],
        marker="o",
        label="Training OA",
    )

    plt.plot(
        history["epoch"],
        history["validation_overall_accuracy"],
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

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Accuracy graph saved to:\n{output_path}")


if __name__ == "__main__":
    main()
