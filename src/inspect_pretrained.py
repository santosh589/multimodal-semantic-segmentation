"""Inspect checkpoint metadata and verify model compatibility.

Run: ``python src/inspect_pretrained.py``
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PRETRAINED_DIR = PROJECT_ROOT / "models" / "pretrained"

MODEL_SOURCE_DIR = PROJECT_ROOT / "models"

CHECKPOINT_EXTENSIONS = {
    ".pt",
    ".pth",
    ".ckpt",
    ".bin",
}


# ============================================================
# Command-line arguments
# ============================================================


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect PyTorch segmentation checkpoints "
            "and report their architecture information."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Optional path to one checkpoint. "
            "When omitted, all checkpoints inside "
            "models/pretrained are inspected."
        ),
    )

    parser.add_argument(
        "--show-all-keys",
        action="store_true",
        help=(
            "Display every state-dictionary key. "
            "By default, only the first and last keys "
            "are displayed."
        ),
    )

    return parser.parse_args()


# ============================================================
# File discovery
# ============================================================


def discover_checkpoints(
    checkpoint_argument: str | None,
) -> list[Path]:
    """
    Find checkpoint files.

    When --checkpoint is supplied, only that file is returned.
    Otherwise, all supported checkpoint files under the
    pretrained directory are returned.
    """
    if checkpoint_argument is not None:
        checkpoint_path = Path(checkpoint_argument)

        if not checkpoint_path.is_absolute():
            checkpoint_path = PROJECT_ROOT / checkpoint_path

        checkpoint_path = checkpoint_path.resolve()

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found:\n" f"{checkpoint_path}")

        if not checkpoint_path.is_file():
            raise ValueError(
                f"The checkpoint path is not a file:\n" f"{checkpoint_path}"
            )

        return [checkpoint_path]

    if not PRETRAINED_DIR.exists():
        raise FileNotFoundError(
            f"Pretrained checkpoint directory " f"not found:\n{PRETRAINED_DIR}"
        )

    checkpoints = sorted(
        path
        for path in PRETRAINED_DIR.rglob("*")
        if (path.is_file() and path.suffix.lower() in CHECKPOINT_EXTENSIONS)
    )

    if not checkpoints:
        raise FileNotFoundError(
            "No checkpoint files were found under:\n" f"{PRETRAINED_DIR}"
        )

    return checkpoints


def discover_python_model_files() -> list[Path]:
    """
    Find Python model-definition files under models/.
    """
    if not MODEL_SOURCE_DIR.exists():
        return []

    return sorted(
        path
        for path in MODEL_SOURCE_DIR.rglob("*.py")
        if "__pycache__" not in path.parts
    )


# ============================================================
# Checkpoint loading
# ============================================================


def load_checkpoint_safely(
    checkpoint_path: Path,
) -> Any:
    """
    Load a trusted course checkpoint on CPU.

    weights_only=False is needed because these checkpoint files
    contain complete PyTorch model objects rather than only
    state dictionaries.
    """
    try:
        return torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

    except TypeError:
        # Compatibility with older PyTorch releases.
        return torch.load(
            checkpoint_path,
            map_location="cpu",
        )


# ============================================================
# State-dictionary detection
# ============================================================


def is_tensor_dictionary(
    value: Any,
) -> bool:
    """
    Return True when value is a non-empty dictionary in which
    every value is a PyTorch tensor.
    """
    if not isinstance(
        value,
        (dict, OrderedDict),
    ):
        return False

    if not value:
        return False

    return all(isinstance(item, torch.Tensor) for item in value.values())


def extract_state_dictionary(
    checkpoint: Any,
) -> tuple[
    dict[str, torch.Tensor] | None,
    str,
]:
    """
    Extract a model state dictionary from common checkpoint forms.

    Supported formats
    -----------------
    1. torch.save(model, path)
    2. torch.save(model.state_dict(), path)
    3. {"state_dict": ...}
    4. {"model_state_dict": ...}
    5. {"model": complete_model}
    6. {"network": complete_model}
    """

    # Case 1:
    # The checkpoint itself is a complete model.
    if isinstance(
        checkpoint,
        torch.nn.Module,
    ):
        return (
            dict(checkpoint.state_dict()),
            "complete_model.state_dict()",
        )

    # Case 2:
    # The checkpoint itself is a plain state dictionary.
    if is_tensor_dictionary(checkpoint):
        return (
            dict(checkpoint),
            "checkpoint_root_state_dict",
        )

    # Remaining supported formats must be dictionaries.
    if not isinstance(
        checkpoint,
        (dict, OrderedDict),
    ):
        return (
            None,
            "unsupported_checkpoint_type",
        )

    candidate_keys = [
        "state_dict",
        "model_state_dict",
        "weights",
        "model",
        "network",
        "net",
    ]

    for key in candidate_keys:
        candidate = checkpoint.get(key)

        # Dictionary containing weights.
        if is_tensor_dictionary(candidate):
            return (
                dict(candidate),
                key,
            )

        # Dictionary containing a complete model object.
        if isinstance(
            candidate,
            torch.nn.Module,
        ):
            return (
                dict(candidate.state_dict()),
                f"{key}.state_dict()",
            )

    return (
        None,
        "state_dictionary_not_found",
    )


def remove_common_prefix(
    state_dictionary: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    Remove common prefixes introduced by DataParallel or
    some training frameworks.
    """
    if not state_dictionary:
        return state_dictionary

    keys = list(state_dictionary.keys())

    prefixes = [
        "module.",
        "model.",
    ]

    for prefix in prefixes:
        if all(key.startswith(prefix) for key in keys):
            return {
                key.removeprefix(prefix): value
                for key, value in state_dictionary.items()
            }

    return state_dictionary


# ============================================================
# Architecture analysis
# ============================================================


def find_convolution_weights(
    state_dictionary: dict[str, torch.Tensor],
) -> list[tuple[str, tuple[int, ...]]]:
    """
    Find four-dimensional convolution weight tensors.

    Typical Conv2d shape:
        [output_channels, input_channels, kernel_h, kernel_w]
    """
    convolution_weights: list[tuple[str, tuple[int, ...]]] = []

    for key, tensor in state_dictionary.items():
        if (
            isinstance(tensor, torch.Tensor)
            and tensor.ndim == 4
            and key.endswith("weight")
        ):
            convolution_weights.append(
                (
                    key,
                    tuple(tensor.shape),
                )
            )

    return convolution_weights


def find_likely_first_convolution(
    convolution_weights: list[tuple[str, tuple[int, ...]]],
) -> (
    tuple[
        str,
        tuple[int, ...],
    ]
    | None
):
    """
    Identify the most likely input convolution.
    """
    if not convolution_weights:
        return None

    preferred_exact_endings = [
        "encoder.conv1.weight",
        "backbone.conv1.weight",
        "resnet.conv1.weight",
        "conv1.weight",
    ]

    for preferred_ending in preferred_exact_endings:
        for key, shape in convolution_weights:
            if key.endswith(preferred_ending):
                return key, shape

    # Fallback: the first Conv2d-like tensor in state-dict order.
    return convolution_weights[0]


def find_likely_output_layers(
    state_dictionary: dict[str, torch.Tensor],
) -> list[tuple[str, tuple[int, ...]]]:
    """
    Find likely segmentation classifier layers.
    """
    output_patterns = (
        "segmentation_head",
        "classifier",
        "final",
        "outconv",
        "out_conv",
        "conv_last",
        "lastconv",
        "head",
    )

    candidates: list[tuple[str, tuple[int, ...]]] = []

    for key, tensor in state_dictionary.items():
        lower_key = key.lower()

        if not (
            isinstance(tensor, torch.Tensor)
            and tensor.ndim >= 1
            and key.endswith("weight")
        ):
            continue

        if any(pattern in lower_key for pattern in output_patterns):
            candidates.append(
                (
                    key,
                    tuple(tensor.shape),
                )
            )

    return candidates


def infer_number_of_classes(
    output_candidates: list[tuple[str, tuple[int, ...]]],
) -> int | None:
    """
    Infer the number of output classes from the final candidate.

    For a segmentation output convolution, the first tensor
    dimension is the number of output channels/classes.
    """
    if not output_candidates:
        return None

    _, shape = output_candidates[-1]

    if not shape:
        return None

    return int(shape[0])


def count_state_dictionary_values(
    state_dictionary: dict[str, torch.Tensor],
) -> int:
    """
    Count all tensor values in a state dictionary.

    This includes parameters and registered buffers.
    """
    return sum(
        tensor.numel()
        for tensor in state_dictionary.values()
        if isinstance(tensor, torch.Tensor)
    )


# ============================================================
# Reporting functions
# ============================================================


def print_checkpoint_metadata(
    checkpoint: Any,
) -> None:
    """
    Print information about the loaded checkpoint object.
    """
    if isinstance(
        checkpoint,
        torch.nn.Module,
    ):
        print("\nCheckpoint contains a complete " "PyTorch model.")

        print(
            f"Model class: "
            f"{checkpoint.__class__.__module__}."
            f"{checkpoint.__class__.__name__}"
        )

        print(f"Training mode: " f"{checkpoint.training}")

        total_parameters = sum(
            parameter.numel() for parameter in checkpoint.parameters()
        )

        trainable_parameters = sum(
            parameter.numel()
            for parameter in checkpoint.parameters()
            if parameter.requires_grad
        )

        print(f"Total model parameters: " f"{total_parameters:,}")

        print(f"Trainable model parameters: " f"{trainable_parameters:,}")

        return

    if not isinstance(
        checkpoint,
        (dict, OrderedDict),
    ):
        print(f"\nCheckpoint object type: " f"{type(checkpoint).__name__}")

        return

    if is_tensor_dictionary(checkpoint):
        print("\nCheckpoint contains a state " "dictionary directly.")

        print(f"Number of tensors: " f"{len(checkpoint)}")

        return

    print("\nTop-level checkpoint keys:")

    for key, value in checkpoint.items():
        if isinstance(
            value,
            torch.nn.Module,
        ):
            description = (
                f"complete model: "
                f"{value.__class__.__module__}."
                f"{value.__class__.__name__}"
            )

        elif isinstance(
            value,
            torch.Tensor,
        ):
            description = f"Tensor{tuple(value.shape)}"

        elif isinstance(
            value,
            (dict, OrderedDict),
        ):
            description = f"{type(value).__name__} " f"with {len(value)} entries"

        else:
            description = f"{type(value).__name__}: " f"{str(value)[:100]}"

        print(f"  {key}: {description}")


def print_state_dictionary_entries(
    state_dictionary: dict[str, torch.Tensor],
    show_all_keys: bool,
) -> None:
    """
    Print state-dictionary keys and tensor shapes.
    """
    keys = list(state_dictionary.keys())

    if show_all_keys:
        print("\nAll state-dictionary entries:")

        for key in keys:
            tensor = state_dictionary[key]

            print(f"  {key:75s} " f"{tuple(tensor.shape)}")

        return

    print("\nFirst 20 state-dictionary entries:")

    for key in keys[:20]:
        tensor = state_dictionary[key]

        print(f"  {key:75s} " f"{tuple(tensor.shape)}")

    if len(keys) > 30:
        print("\n  ...")

    print("\nLast 10 state-dictionary entries:")

    for key in keys[-10:]:
        tensor = state_dictionary[key]

        print(f"  {key:75s} " f"{tuple(tensor.shape)}")


def print_state_dictionary_summary(
    state_dictionary: dict[str, torch.Tensor],
    state_dictionary_location: str,
    show_all_keys: bool,
) -> None:
    """
    Print architecture information inferred from model weights.
    """
    state_dictionary = remove_common_prefix(state_dictionary)

    print(f"\nState dictionary location: " f"{state_dictionary_location}")

    print(f"Number of state-dictionary tensors: " f"{len(state_dictionary)}")

    print(
        f"Total state-dictionary values: "
        f"{count_state_dictionary_values(state_dictionary):,}"
    )

    print_state_dictionary_entries(
        state_dictionary=state_dictionary,
        show_all_keys=show_all_keys,
    )

    convolution_weights = find_convolution_weights(state_dictionary)

    print(f"\nNumber of Conv2d-like tensors: " f"{len(convolution_weights)}")

    first_convolution = find_likely_first_convolution(convolution_weights)

    if first_convolution is None:
        print("\nThe first input convolution could " "not be identified.")

    else:
        first_key, first_shape = first_convolution

        print("\nLikely first input convolution:")

        print(f"  Key: {first_key}")

        print(f"  Weight shape: {first_shape}")

        if len(first_shape) == 4:
            output_channels = first_shape[0]

            input_channels = first_shape[1]

            kernel_height = first_shape[2]

            kernel_width = first_shape[3]

            print(f"  Output feature channels: " f"{output_channels}")

            print(f"  Expected input channels: " f"{input_channels}")

            print(f"  Kernel size: " f"{kernel_height} × " f"{kernel_width}")

            print("\nInput compatibility:")

            print(f"  Experiment A requires 3 channels: " f"{input_channels == 3}")

            print(f"  Experiment B requires 4 channels: " f"{input_channels == 4}")

    output_candidates = find_likely_output_layers(state_dictionary)

    print("\nLikely output/classifier layers:")

    if not output_candidates:
        print("  No output layer was identified " "automatically.")

    else:
        for key, shape in output_candidates:
            print(f"  {key:75s} {shape}")

        inferred_classes = infer_number_of_classes(output_candidates)

        print(f"\nInferred number of output classes: " f"{inferred_classes}")

        print(f"Expected number of classes is 6: " f"{inferred_classes == 6}")


# ============================================================
# Checkpoint inspection
# ============================================================


def inspect_checkpoint(
    checkpoint_path: Path,
    show_all_keys: bool,
) -> None:
    """
    Load and inspect one checkpoint.
    """
    try:
        relative_path = checkpoint_path.relative_to(PROJECT_ROOT)

    except ValueError:
        relative_path = checkpoint_path

    print("\n" + "=" * 90)

    print(f"CHECKPOINT: {relative_path}")

    print("=" * 90)

    print(f"File size: " f"{checkpoint_path.stat().st_size / 1_048_576:.2f} MB")

    try:
        checkpoint = load_checkpoint_safely(checkpoint_path)

    except Exception as error:
        print("\nCould not load checkpoint:")

        print(f"{type(error).__name__}: " f"{error}")

        return

    print_checkpoint_metadata(checkpoint)

    (
        state_dictionary,
        state_dictionary_location,
    ) = extract_state_dictionary(checkpoint)

    if state_dictionary is None:
        print("\nA tensor state dictionary could " "not be identified automatically.")

        print(f"Detection result: " f"{state_dictionary_location}")

        return

    print_state_dictionary_summary(
        state_dictionary=state_dictionary,
        state_dictionary_location=(state_dictionary_location),
        show_all_keys=show_all_keys,
    )


# ============================================================
# Model source inspection
# ============================================================


def inspect_model_source_files() -> None:
    """
    Report any Python architecture files found under models/.
    """
    python_files = discover_python_model_files()

    print("\n" + "=" * 90)
    print("MODEL SOURCE FILES")
    print("=" * 90)

    if not python_files:
        print("No Python model files found under:\n" f"{MODEL_SOURCE_DIR}")

        print(
            "\nThis is not an error. The supplied "
            "checkpoint files contain complete model objects."
        )

        return

    for path in python_files:
        relative_path = path.relative_to(PROJECT_ROOT)

        text = path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        lower_text = text.lower()

        indicators: list[str] = []

        if "resnet18" in lower_text:
            indicators.append("contains ResNet18")

        if "unet" in lower_text:
            indicators.append("contains U-Net")

        if "conv2d" in lower_text:
            indicators.append("contains Conv2d")

        if "forward(" in lower_text:
            indicators.append("contains forward()")

        if "num_classes" in lower_text:
            indicators.append("contains num_classes")

        if "in_channels" in lower_text:
            indicators.append("contains in_channels")

        indicator_text = (
            ", ".join(indicators) if indicators else "no obvious architecture keywords"
        )

        print(f"  {relative_path}: " f"{indicator_text}")


# ============================================================
# Main
# ============================================================


def main() -> None:
    arguments = parse_arguments()

    print(f"Project root:\n" f"{PROJECT_ROOT}")

    inspect_model_source_files()

    checkpoints = discover_checkpoints(arguments.checkpoint)

    print(f"\nCheckpoints found: " f"{len(checkpoints)}")

    for checkpoint_path in checkpoints:
        inspect_checkpoint(
            checkpoint_path=checkpoint_path,
            show_all_keys=(arguments.show_all_keys),
        )

    print("\n" + "=" * 90)
    print("INSPECTION COMPLETE")
    print("=" * 90)

    print(
        "\nFor the 3-channel model, the first "
        "convolution should have shape similar to:"
    )

    print("  [64, 3, 7, 7]")

    print(
        "\nFor the 4-channel model, the first "
        "convolution should have shape similar to:"
    )

    print("  [64, 4, 7, 7]")

    print("\nThe final segmentation layer should " "have 6 output channels.")


if __name__ == "__main__":
    main()
