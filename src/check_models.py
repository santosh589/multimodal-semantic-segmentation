"""Check model architecture, input shapes, and gradient flow.

Run: ``python src/check_models.py``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from dataset import (
    PROJECT_ROOT,
    create_dataloaders,
)

from model_utils import (
    FOUR_CHANNEL_CHECKPOINT,
    THREE_CHANNEL_CHECKPOINT,
    count_parameters,
    create_experiment_model,
    get_device,
)


# ============================================================
# Arguments
# ============================================================


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load and verify the supplied 3-channel and " "4-channel U-Net checkpoints."
        )
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help=("Batch size used for model smoke testing. " "Default: 2."),
    )

    parser.add_argument(
        "--run-backward",
        action="store_true",
        help=(
            "Also run one backward pass to verify that "
            "the model supports further training."
        ),
    )

    return parser.parse_args()


# ============================================================
# General validation
# ============================================================


def verify_checkpoint_exists(
    checkpoint_path: Path,
) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Converted checkpoint not found:\n"
            f"{checkpoint_path}\n\n"
            "Run:\n"
            "python src/convert_checkpoints.py"
        )


def print_tensor_information(
    name: str,
    tensor: torch.Tensor,
) -> None:
    print(f"{name}:")

    print(f"  Shape: {tuple(tensor.shape)}")

    print(f"  Dtype: {tensor.dtype}")

    print(f"  Device: {tensor.device}")

    if tensor.is_floating_point():
        print(f"  Minimum: " f"{tensor.min().item():.6f}")

        print(f"  Maximum: " f"{tensor.max().item():.6f}")


# ============================================================
# Experiment smoke test
# ============================================================


def run_experiment_check(
    experiment_name: str,
    expected_input_channels: int,
    batch_size: int,
    device: torch.device,
    run_backward: bool,
) -> None:
    print("\n" + "=" * 85)

    print(f"EXPERIMENT: {experiment_name}")

    print("=" * 85)

    dataloaders = create_dataloaders(
        input_mode=experiment_name,
        batch_size=batch_size,
        num_workers=0,
        project_root=PROJECT_ROOT,
    )

    validation_loader = dataloaders["validation"]

    batch = next(iter(validation_loader))

    images = batch["image"].to(device)

    labels = batch["label"].to(device)

    print_tensor_information(
        "Input batch",
        images,
    )

    print_tensor_information(
        "Label batch",
        labels,
    )

    expected_input_shape = (
        images.shape[0],
        expected_input_channels,
        256,
        256,
    )

    if tuple(images.shape) != (expected_input_shape):
        raise RuntimeError(
            "Unexpected input shape.\n"
            f"Expected: {expected_input_shape}\n"
            f"Received: {tuple(images.shape)}"
        )

    expected_label_shape = (
        images.shape[0],
        256,
        256,
    )

    if tuple(labels.shape) != (expected_label_shape):
        raise RuntimeError(
            "Unexpected label shape.\n"
            f"Expected: {expected_label_shape}\n"
            f"Received: {tuple(labels.shape)}"
        )

    if images.dtype != torch.float32:
        raise TypeError(f"Expected float32 input, " f"received {images.dtype}")

    if labels.dtype != torch.int64:
        raise TypeError(f"Expected int64 labels, " f"received {labels.dtype}")

    print("\nLoading model...")

    model = create_experiment_model(
        experiment=experiment_name,
        device=device,
    )

    total_parameters, trainable_parameters = count_parameters(model)

    print(
        f"Model class: " f"{model.__class__.__module__}." f"{model.__class__.__name__}"
    )

    print(f"Total parameters: " f"{total_parameters:,}")

    print(f"Trainable parameters: " f"{trainable_parameters:,}")

    first_convolution = model.encoder.conv1

    print("\nFirst convolution:")

    print(f"  Input channels: " f"{first_convolution.in_channels}")

    print(f"  Output channels: " f"{first_convolution.out_channels}")

    print(f"  Kernel size: " f"{first_convolution.kernel_size}")

    if first_convolution.in_channels != expected_input_channels:
        raise RuntimeError("Model input-channel mismatch.")

    criterion = nn.CrossEntropyLoss()

    if run_backward:
        print("\nRunning forward and backward pass...")

        model.train()

        model.zero_grad(set_to_none=True)

        logits = model(images)

        loss = criterion(
            logits,
            labels,
        )

        loss.backward()

        gradients_found = any(
            parameter.grad is not None
            for parameter in model.parameters()
            if parameter.requires_grad
        )

        if not gradients_found:
            raise RuntimeError("No model gradients were produced.")

        print("Backward pass completed successfully.")

    else:
        print("\nRunning forward pass...")

        model.eval()

        with torch.no_grad():
            logits = model(images)

            loss = criterion(
                logits,
                labels,
            )

    print_tensor_information(
        "Model logits",
        logits,
    )

    expected_output_shape = (
        images.shape[0],
        6,
        256,
        256,
    )

    if tuple(logits.shape) != (expected_output_shape):
        raise RuntimeError(
            "Unexpected model output shape.\n"
            f"Expected: {expected_output_shape}\n"
            f"Received: {tuple(logits.shape)}"
        )

    if not torch.isfinite(logits).all():
        raise RuntimeError("The model output contains NaN or " "infinite values.")

    if not torch.isfinite(loss):
        raise RuntimeError("The loss is NaN or infinite.")

    predictions = torch.argmax(
        logits,
        dim=1,
    )

    print(f"\nCrossEntropyLoss: " f"{loss.item():.6f}")

    print(f"Prediction shape: " f"{tuple(predictions.shape)}")

    print(f"Predicted classes in batch: " f"{torch.unique(predictions).tolist()}")

    print(f"Ground-truth classes in batch: " f"{torch.unique(labels).tolist()}")

    if predictions.shape != labels.shape:
        raise RuntimeError("Prediction and label shapes do not match.")

    print(f"\n{experiment_name} model check passed.")

    # Release memory before loading the next model.
    del model
    del images
    del labels
    del logits
    del predictions
    del loss

    if device.type == "cuda":
        torch.cuda.empty_cache()

    elif device.type == "mps":
        torch.mps.empty_cache()


# ============================================================
# Main
# ============================================================


def main() -> None:
    arguments = parse_arguments()

    verify_checkpoint_exists(THREE_CHANNEL_CHECKPOINT)

    verify_checkpoint_exists(FOUR_CHANNEL_CHECKPOINT)

    device = get_device()

    print("=" * 85)
    print("MODEL AND CHECKPOINT SMOKE TEST")
    print("=" * 85)

    print(f"\nProject root:\n" f"{PROJECT_ROOT}")

    print(f"\nDevice: {device}")

    print(f"Batch size: " f"{arguments.batch_size}")

    print(f"Backward test: " f"{arguments.run_backward}")

    print("\nThree-channel checkpoint:\n" f"{THREE_CHANNEL_CHECKPOINT}")

    print("\nFour-channel checkpoint:\n" f"{FOUR_CHANNEL_CHECKPOINT}")

    run_experiment_check(
        experiment_name="multispectral",
        expected_input_channels=3,
        batch_size=arguments.batch_size,
        device=device,
        run_backward=arguments.run_backward,
    )

    run_experiment_check(
        experiment_name="early_fusion",
        expected_input_channels=4,
        batch_size=arguments.batch_size,
        device=device,
        run_backward=arguments.run_backward,
    )

    print("\n" + "=" * 85)
    print("ALL MODEL CHECKS PASSED")
    print("=" * 85)

    print("\nExperiment A:")

    print("  Input:  [batch, 3, 256, 256]")

    print("  Output: [batch, 6, 256, 256]")

    print("\nExperiment B:")

    print("  Input:  [batch, 4, 256, 256]")

    print("  Output: [batch, 6, 256, 256]")

    print("\nCrossEntropyLoss works for both models.")


if __name__ == "__main__":
    main()
