# Multimodal Multi-Class Semantic Segmentation

An end-to-end PyTorch project for six-class semantic segmentation with a U-Net model and a ResNet18 encoder. The project compares two input settings:

- **Multispectral:** three image channels.
- **Early fusion:** three image channels plus one depth (nDSM) channel.

The training pipeline supports training from scratch and full-model fine-tuning from compatible U-Net--ResNet18 checkpoints. Evaluation reports overall accuracy, macro precision/recall/F1, mean IoU, per-class metrics, and confusion matrices.

## Results

The fine-tuned early-fusion model achieved **82.7% overall accuracy** and **0.565 mean IoU** on the held-out test set.

## Repository layout

```text
.
├── data/                 # Private dataset plus a small public example triplet
│   └── examples/         # One matched image, depth map, and segmentation label
│   ├── labels/           # RGB segmentation labels
│   ├── mspectral-images/ # Multispectral source images
│   └── ndsm-images/      # Normalized depth maps
├── metadata/             # Generated split, normalization, and patch-index files
├── models/               # Private pretrained or fine-tuned checkpoints
├── outputs/              # Generated curves, predictions, and evaluation artifacts
├── notebooks/            # Results-analysis notebook
├── src/                  # Data preparation, training, inference, and evaluation code
├── data-config.json      # Class mapping and train/test area configuration
└── requirements.txt      # Python dependencies
```

The full dataset and model weights are intentionally excluded from version control. A single matched example triplet, generated metadata, the results notebook, and output artifacts are included to make the project understandable and reproducible.

## Included example data

`data/examples/` contains one matched area-17 triplet:

- a multispectral image;
- a normalized depth (nDSM) image; and
- its RGB semantic-segmentation label.

These files are provided for inspection and visualization only; they are not sufficient to train or evaluate the model.

## Output artifacts

The repository includes generated visual and numerical results in `outputs/`:

| Directory | Contents |
| --- | --- |
| `outputs/figures/` | Data-alignment checks, label-validation views, patch grids, and normalization plots. |
| `outputs/training/` | Training histories and loss, accuracy, and F1 curves for models trained from scratch. |
| `outputs/fine_tuning/` | Fine-tuning histories, summaries, and learning curves for checkpoint-initialized models. |
| `outputs/test_evaluation/` | Test predictions, ground-truth and prediction comparisons, per-area confusion matrices, overall confusion matrices, and CSV/JSON evaluation metrics. |

The published outputs contain figures and evaluation results only; raw input data and model-weight files are not included.

## Source modules

| Module | Purpose |
| --- | --- |
| `audit_data.py`, `check_alignment.py`, `validate_labels.py` | Audit input files, spatial alignment, and label colours. |
| `create_splits.py`, `create_patch_indexes.py`, `calculate_statistics.py` | Create area-level splits, patch indices, and normalization statistics. |
| `dataset.py`, `augmentations.py` | Load and preprocess image, depth, and label patches. |
| `model_utils.py`, `losses.py`, `metrics.py` | Build models and define losses and evaluation metrics. |
| `train_experiment.py`, `training.py` | Train from scratch or fine-tune a checkpoint. |
| `evaluate_test_set.py`, `inference.py`, `visualization.py` | Run inference, evaluate held-out data, and create visual outputs. |

## Setup

```bash
git clone <repository-url>
cd multimodal-semantic-segmentation-unet-resnet18
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the private data in the paths shown in the repository layout. The source code derives its project root automatically, so no personal absolute paths need to be configured.

## Typical workflow

```bash
# Validate data and prepare metadata
python src/audit_data.py
python src/check_alignment.py
python src/create_splits.py
python src/calculate_statistics.py
python src/validate_labels.py
python src/create_patch_indexes.py

# Fine-tune the 4-channel early-fusion model
python src/train_experiment.py \
  --experiment early_fusion \
  --initialization checkpoint \
  --epochs 20 \
  --batch-size 8 \
  --accumulation-steps 4 \
  --learning-rate 0.00001 \
  --output-dir outputs/fine_tuning/early_fusion/epochs_20

# Evaluate the best saved model
python src/evaluate_test_set.py \
  --experiment early_fusion \
  --checkpoint outputs/fine_tuning/early_fusion/epochs_20/best_model_state_dict.pt \
  --batch-size 4 \
  --output-dir outputs/test_evaluation/early_fusion/epochs_20
```

## Data privacy

This repository contains code and configuration only. Dataset files, checkpoints, generated outputs, local environments, and absolute file paths are excluded through `.gitignore`.
