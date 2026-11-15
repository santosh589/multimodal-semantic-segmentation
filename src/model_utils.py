"""U-Net--ResNet18 construction, checkpoint loading, and device utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import segmentation_models_pytorch as smp
import torch
from torch import nn


# ============================================================
# Type definitions
# ============================================================

ExperimentName = Literal[
    "multispectral",
    "early_fusion",
]


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PRETRAINED_DIR = PROJECT_ROOT / "models" / "pretrained"

THREE_CHANNEL_CHECKPOINT = PRETRAINED_DIR / (
    "unet_resnet18_3channels_" "25epochs_patchsize256_state_dict.pt"
)

FOUR_CHANNEL_CHECKPOINT = PRETRAINED_DIR / (
    "unet_resnet18_4channels_" "25epochs_patchsize256_state_dict.pt"
)


# ============================================================
# Device selection
# ============================================================


def get_device() -> torch.device:
    """
    Select CUDA, Apple MPS, or CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# ============================================================
# Model creation
# ============================================================


def create_unet_resnet18(
    input_channels: int,
    number_of_classes: int = 6,
) -> nn.Module:
    """
    Create the same U-Net ResNet18 architecture used by
    the supplied checkpoints.

    Parameters
    ----------
    input_channels:
        3 for multispectral only.
        4 for multispectral + DSM early fusion.

    number_of_classes:
        Number of segmentation classes.
    """
    if input_channels not in {
        3,
        4,
    }:
        raise ValueError(
            "This assignment expects either 3 or 4 "
            f"input channels. Received: {input_channels}"
        )

    if number_of_classes != 6:
        raise ValueError(
            "This dataset contains exactly six classes. "
            f"Received: {number_of_classes}"
        )

    model = smp.Unet(
        encoder_name="resnet18",
        # Do not download ImageNet weights here.
        # The supplied checkpoint will provide all weights.
        encoder_weights=None,
        in_channels=input_channels,
        classes=number_of_classes,
        # CrossEntropyLoss requires raw logits.
        activation=None,
    )

    return model


# ============================================================
# State-dictionary loading
# ============================================================


def load_state_dictionary(
    checkpoint_path: str | Path,
) -> dict[str, torch.Tensor]:
    """
    Load a converted state-dictionary checkpoint.
    """
    checkpoint_path = Path(checkpoint_path).resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"State-dictionary checkpoint not found:\n" f"{checkpoint_path}"
        )

    try:
        state_dictionary = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

    except TypeError:
        state_dictionary = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

    if not isinstance(
        state_dictionary,
        dict,
    ):
        raise TypeError(
            "Expected a state dictionary, but loaded "
            f"{type(state_dictionary).__name__}"
        )

    if not state_dictionary:
        raise RuntimeError(f"State dictionary is empty:\n{checkpoint_path}")

    if not all(isinstance(value, torch.Tensor) for value in state_dictionary.values()):
        raise TypeError(
            "The checkpoint dictionary contains values " "that are not PyTorch tensors."
        )

    # Remove DataParallel prefix if present.
    if all(key.startswith("module.") for key in state_dictionary):
        state_dictionary = {
            key.removeprefix("module."): value
            for key, value in state_dictionary.items()
        }

    return state_dictionary


def validate_checkpoint_architecture(
    state_dictionary: dict[str, torch.Tensor],
    expected_input_channels: int,
    expected_classes: int = 6,
) -> None:
    """
    Validate the first and final convolution shapes before
    loading the checkpoint into the model.
    """
    first_convolution_key = "encoder.conv1.weight"

    output_weight_key = "segmentation_head.0.weight"

    output_bias_key = "segmentation_head.0.bias"

    required_keys = {
        first_convolution_key,
        output_weight_key,
        output_bias_key,
    }

    missing_keys = required_keys - set(state_dictionary)

    if missing_keys:
        raise KeyError(
            "Checkpoint is missing required layers: " f"{sorted(missing_keys)}"
        )

    first_shape = tuple(state_dictionary[first_convolution_key].shape)

    output_weight_shape = tuple(state_dictionary[output_weight_key].shape)

    output_bias_shape = tuple(state_dictionary[output_bias_key].shape)

    if len(first_shape) != 4:
        raise ValueError(f"Unexpected first convolution shape: " f"{first_shape}")

    actual_input_channels = first_shape[1]

    if actual_input_channels != expected_input_channels:
        raise ValueError(
            "Checkpoint input-channel mismatch.\n"
            f"Expected: {expected_input_channels}\n"
            f"Checkpoint: {actual_input_channels}"
        )

    if not output_weight_shape:
        raise ValueError("Output weight tensor has no dimensions.")

    actual_output_classes = output_weight_shape[0]

    if actual_output_classes != expected_classes:
        raise ValueError(
            "Checkpoint class-count mismatch.\n"
            f"Expected: {expected_classes}\n"
            f"Checkpoint: {actual_output_classes}"
        )

    if output_bias_shape != (expected_classes,):
        raise ValueError(
            "Unexpected segmentation-head bias shape.\n"
            f"Expected: {(expected_classes,)}\n"
            f"Received: {output_bias_shape}"
        )


def load_model_from_checkpoint(
    input_channels: int,
    checkpoint_path: str | Path,
    number_of_classes: int = 6,
    device: torch.device | None = None,
) -> nn.Module:
    """
    Construct a U-Net ResNet18 model and load the supplied
    state dictionary with strict architecture checking.
    """
    if device is None:
        device = get_device()

    state_dictionary = load_state_dictionary(checkpoint_path)

    validate_checkpoint_architecture(
        state_dictionary=state_dictionary,
        expected_input_channels=input_channels,
        expected_classes=number_of_classes,
    )

    model = create_unet_resnet18(
        input_channels=input_channels,
        number_of_classes=number_of_classes,
    )

    load_result = model.load_state_dict(
        state_dictionary,
        strict=True,
    )

    if load_result.missing_keys:
        raise RuntimeError(f"Missing model keys: " f"{load_result.missing_keys}")

    if load_result.unexpected_keys:
        raise RuntimeError(
            f"Unexpected checkpoint keys: " f"{load_result.unexpected_keys}"
        )

    model = model.to(device)

    return model


# ============================================================
# Experiment-specific factory
# ============================================================


def create_experiment_model(
    experiment: ExperimentName,
    device: torch.device | None = None,
) -> nn.Module:
    """
    Create and load the correct model for an experiment.

    multispectral:
        3-channel input.

    early_fusion:
        4-channel input.
    """
    if device is None:
        device = get_device()

    if experiment == "multispectral":
        return load_model_from_checkpoint(
            input_channels=3,
            checkpoint_path=(THREE_CHANNEL_CHECKPOINT),
            number_of_classes=6,
            device=device,
        )

    if experiment == "early_fusion":
        return load_model_from_checkpoint(
            input_channels=4,
            checkpoint_path=(FOUR_CHANNEL_CHECKPOINT),
            number_of_classes=6,
            device=device,
        )

    raise ValueError(f"Unknown experiment: {experiment}")


# ============================================================
# Model information
# ============================================================


def count_parameters(
    model: nn.Module,
) -> tuple[int, int]:
    """
    Return total and trainable parameter counts.
    """
    total_parameters = sum(parameter.numel() for parameter in model.parameters())

    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    return (
        total_parameters,
        trainable_parameters,
    )
