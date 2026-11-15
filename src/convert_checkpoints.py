"""Convert supplied segmentation checkpoints into portable state dictionaries.

Run: ``python src/convert_checkpoints.py``
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PRETRAINED_DIR = PROJECT_ROOT / "models" / "pretrained"

CHECKPOINT_NAMES = [
    ("unet_resnet18_3channels_" "25epochs_patchsize256.pt"),
    ("unet_resnet18_4channels_" "25epochs_patchsize256.pt"),
]


# ============================================================
# Utility functions
# ============================================================


def is_tensor_dictionary(
    value: Any,
) -> bool:
    """
    Check whether value is a non-empty dictionary containing
    only PyTorch tensors.
    """
    if not isinstance(
        value,
        (dict, OrderedDict),
    ):
        return False

    if not value:
        return False

    return all(isinstance(item, torch.Tensor) for item in value.values())


def load_complete_checkpoint(
    checkpoint_path: Path,
) -> Any:
    """
    Load a trusted checkpoint containing a complete model.
    """
    try:
        return torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

    except TypeError:
        return torch.load(
            checkpoint_path,
            map_location="cpu",
        )


def extract_state_dictionary(
    checkpoint: Any,
) -> tuple[
    dict[str, torch.Tensor],
    str,
]:
    """
    Extract the state dictionary from common checkpoint formats.
    """

    # Complete model saved directly.
    if isinstance(
        checkpoint,
        torch.nn.Module,
    ):
        return (
            dict(checkpoint.state_dict()),
            "complete_model.state_dict()",
        )

    # State dictionary saved directly.
    if is_tensor_dictionary(checkpoint):
        return (
            dict(checkpoint),
            "checkpoint_root_state_dict",
        )

    if not isinstance(
        checkpoint,
        (dict, OrderedDict),
    ):
        raise TypeError(
            "Unsupported checkpoint object type: " f"{type(checkpoint).__name__}"
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

        if is_tensor_dictionary(candidate):
            return (
                dict(candidate),
                key,
            )

        if isinstance(
            candidate,
            torch.nn.Module,
        ):
            return (
                dict(candidate.state_dict()),
                f"{key}.state_dict()",
            )

    raise RuntimeError(
        "No model or state dictionary could be " "identified inside the checkpoint."
    )


def remove_module_prefix(
    state_dictionary: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    Remove a DataParallel 'module.' prefix when present.
    """
    if not state_dictionary:
        return state_dictionary

    keys = list(state_dictionary.keys())

    if all(key.startswith("module.") for key in keys):
        return {
            key.removeprefix("module."): value
            for key, value in state_dictionary.items()
        }

    return state_dictionary


def identify_first_convolution(
    state_dictionary: dict[str, torch.Tensor],
) -> (
    tuple[
        str,
        tuple[int, ...],
    ]
    | None
):
    """
    Identify the likely first encoder convolution.
    """
    candidates = []

    for key, tensor in state_dictionary.items():
        if (
            isinstance(tensor, torch.Tensor)
            and tensor.ndim == 4
            and key.endswith("weight")
        ):
            candidates.append(
                (
                    key,
                    tuple(tensor.shape),
                )
            )

    preferred_endings = [
        "encoder.conv1.weight",
        "backbone.conv1.weight",
        "resnet.conv1.weight",
        "conv1.weight",
    ]

    for preferred_ending in preferred_endings:
        for key, shape in candidates:
            if key.endswith(preferred_ending):
                return key, shape

    if candidates:
        return candidates[0]

    return None


def identify_output_layer(
    state_dictionary: dict[str, torch.Tensor],
) -> (
    tuple[
        str,
        tuple[int, ...],
    ]
    | None
):
    """
    Identify the likely final segmentation layer.
    """
    patterns = (
        "segmentation_head",
        "classifier",
        "final",
        "outconv",
        "out_conv",
    )

    candidates = []

    for key, tensor in state_dictionary.items():
        if not (isinstance(tensor, torch.Tensor) and key.endswith("weight")):
            continue

        if any(pattern in key.lower() for pattern in patterns):
            candidates.append(
                (
                    key,
                    tuple(tensor.shape),
                )
            )

    if candidates:
        return candidates[-1]

    return None


# ============================================================
# Conversion
# ============================================================


def convert_checkpoint(
    checkpoint_path: Path,
) -> Path:
    """
    Convert one complete model checkpoint to a state dictionary.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found:\n" f"{checkpoint_path}")

    print("\n" + "-" * 80)

    print(f"Loading:\n" f"{checkpoint_path}")

    checkpoint = load_complete_checkpoint(checkpoint_path)

    print(f"Loaded checkpoint type: " f"{type(checkpoint).__name__}")

    (
        state_dictionary,
        source_location,
    ) = extract_state_dictionary(checkpoint)

    state_dictionary = remove_module_prefix(state_dictionary)

    print(f"State dictionary source: " f"{source_location}")

    print(f"Number of tensors: " f"{len(state_dictionary)}")

    total_values = sum(tensor.numel() for tensor in state_dictionary.values())

    print(f"Total tensor values: " f"{total_values:,}")

    first_convolution = identify_first_convolution(state_dictionary)

    if first_convolution is not None:
        first_key, first_shape = first_convolution

        print("\nLikely first convolution:")

        print(f"  Key: {first_key}")

        print(f"  Shape: {first_shape}")

        if len(first_shape) == 4:
            print(f"  Input channels: " f"{first_shape[1]}")

    output_layer = identify_output_layer(state_dictionary)

    if output_layer is not None:
        output_key, output_shape = output_layer

        print("\nLikely segmentation output layer:")

        print(f"  Key: {output_key}")

        print(f"  Shape: {output_shape}")

        if output_shape:
            print(f"  Output classes: " f"{output_shape[0]}")

    output_path = checkpoint_path.with_name(checkpoint_path.stem + "_state_dict.pt")

    torch.save(
        state_dictionary,
        output_path,
    )

    print(f"\nSaved state dictionary:\n" f"{output_path}")

    # Verify that the newly written file is readable
    # using weights_only=True.
    try:
        loaded_state_dictionary = torch.load(
            output_path,
            map_location="cpu",
            weights_only=True,
        )

    except TypeError:
        loaded_state_dictionary = torch.load(
            output_path,
            map_location="cpu",
        )

    if not is_tensor_dictionary(loaded_state_dictionary):
        raise RuntimeError(
            "The converted file did not load as " "a valid state dictionary."
        )

    if set(loaded_state_dictionary.keys()) != set(state_dictionary.keys()):
        raise RuntimeError(
            "Converted state-dictionary keys do not "
            "match the original state dictionary."
        )

    print("Verification passed.")

    return output_path


# ============================================================
# Main
# ============================================================


def main() -> None:
    print("=" * 80)
    print("CHECKPOINT CONVERSION")
    print("=" * 80)

    output_paths: list[Path] = []

    for checkpoint_name in CHECKPOINT_NAMES:
        checkpoint_path = PRETRAINED_DIR / checkpoint_name

        output_path = convert_checkpoint(checkpoint_path)

        output_paths.append(output_path)

    print("\n" + "=" * 80)
    print("CONVERSION COMPLETE")
    print("=" * 80)

    print("\nConverted state dictionaries:")

    for output_path in output_paths:
        print(f"  {output_path}")

    print("\nThese converted files can be loaded using:")

    print(
        """
state_dict = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True,
)

model.load_state_dict(state_dict)
"""
    )


if __name__ == "__main__":
    main()
