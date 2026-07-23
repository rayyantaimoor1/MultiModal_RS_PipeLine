"""
train_eurosat_cnn.py

Real end-to-end training run: loads the actual downloaded EuroSAT (all
13 Sentinel-2 bands) dataset from disk, splits it into train/val sets,
trains CustomCNN on it, prints per-epoch metrics, and saves a checkpoint
PLUS training history and validation predictions for the Streamlit
dashboard (dashboard.py).

Run from the project root (multimodal_rs_pipeline) with the rs_pipeline
conda environment active:

    python train_eurosat_cnn.py

Expects the data folder layout:
    data/
        AnnualCrop/*.tif
        Forest/*.tif
        HerbaceousVegetation/*.tif
        Highway/*.tif
        Industrial/*.tif
        Pasture/*.tif
        PermanentCrop/*.tif
        Residential/*.tif
        River/*.tif
        SeaLake/*.tif
"""

import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.datasets.eurosat_dataset import EuroSATDataset, SpatialAugment
from src.models.custom_cnn import CustomCNN
from src.training.trainer import ModelTrainer


def main() -> None:
    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    DATA_DIR = Path("data")
    NUM_CHANNELS = 13
    IMAGE_SIZE = (64, 64)
    BATCH_SIZE = 32
    NUM_EPOCHS = 4
    LEARNING_RATE = 1e-3
    VAL_FRACTION = 0.15
    RANDOM_SEED = 42
    CHECKPOINT_DIR = Path("checkpoints")
    RESULTS_DIR = Path("results")

    # Speed knob: caps how many images per class are used at all, so the
    # whole run fits a fixed time budget on modest CPU-only hardware.
    # Set to None to use every image in the dataset (full run, much slower).
    MAX_SAMPLES_PER_CLASS = 150

    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Dataset: build once without augmentation to get a stable class
    # mapping and file list, then split indices into train/val.
    # ------------------------------------------------------------------
    print(f"Loading EuroSAT dataset from: {DATA_DIR.resolve()}")

    base_dataset = EuroSATDataset(
        root_dir=DATA_DIR,
        num_channels=NUM_CHANNELS,
        image_size=IMAGE_SIZE,
    )
    print(f"Found {len(base_dataset)} total samples across {base_dataset.num_classes} classes.")
    print(f"Class mapping: {base_dataset.class_to_idx}")

    # Subsample: keep at most MAX_SAMPLES_PER_CLASS indices per class label,
    # chosen randomly, so training time stays bounded regardless of how
    # many thousands of images exist per class on disk.
    if MAX_SAMPLES_PER_CLASS is not None:
        indices_by_class: dict[int, list[int]] = defaultdict(list)
        for idx, (_path, label) in enumerate(base_dataset.samples):
            indices_by_class[label].append(idx)

        selected_indices: list[int] = []
        for label, idxs in indices_by_class.items():
            random.shuffle(idxs)
            selected_indices.extend(idxs[:MAX_SAMPLES_PER_CLASS])

        print(
            f"Subsampling to at most {MAX_SAMPLES_PER_CLASS} per class "
            f"-> {len(selected_indices)} samples total (for a bounded-time run)."
        )
        all_indices = selected_indices
    else:
        all_indices = list(range(len(base_dataset)))

    random.shuffle(all_indices)
    val_size = int(len(all_indices) * VAL_FRACTION)
    val_indices = all_indices[:val_size]
    train_indices = all_indices[val_size:]
    print(f"Train samples: {len(train_indices)} | Val samples: {len(val_indices)}")

    # Also record which original file paths ended up in the validation
    # split, and the class distribution, so the dashboard can show real
    # sample images and a class-balance chart without re-deriving them.
    idx_to_class = {v: k for k, v in base_dataset.class_to_idx.items()}
    class_counts = {
        idx_to_class[label]: len(idxs) for label, idxs in indices_by_class.items()
    } if MAX_SAMPLES_PER_CLASS is not None else None

    val_file_records = [
        {"filepath": str(base_dataset.samples[i][0]), "label_idx": base_dataset.samples[i][1]}
        for i in val_indices
    ]

    # Build a second dataset instance WITH augmentation for training,
    # reusing the identical class_to_idx mapping so labels line up.
    train_dataset_full = EuroSATDataset(
        root_dir=DATA_DIR,
        num_channels=NUM_CHANNELS,
        image_size=IMAGE_SIZE,
        transform=SpatialAugment(),
        class_to_idx=base_dataset.class_to_idx,
    )

    train_subset = Subset(train_dataset_full, train_indices)
    val_subset = Subset(base_dataset, val_indices)  # no augmentation for validation

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ------------------------------------------------------------------
    # Model, optimizer, trainer
    # ------------------------------------------------------------------
    model = CustomCNN(
        in_channels=NUM_CHANNELS,
        num_classes=base_dataset.num_classes,
        conv_channels=[32, 64, 128, 256],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    trainer = ModelTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=nn.CrossEntropyLoss(),
        task_type="classification",
        grad_clip_norm=1.0,
        checkpoint_dir=CHECKPOINT_DIR,
    )
    print(f"Training on device: {trainer.device}")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    history = trainer.fit(train_loader, val_loader, num_epochs=NUM_EPOCHS, verbose=True)

    trainer.restore_best_model()
    checkpoint_path = trainer.save_checkpoint(
        "eurosat_customcnn.pt",
        extra_metadata={"class_to_idx": base_dataset.class_to_idx, "num_epochs": NUM_EPOCHS},
    )
    print(f"\nSaved best model checkpoint to: {checkpoint_path.resolve()}")

    # ------------------------------------------------------------------
    # Save training history as JSON for the dashboard.
    # ------------------------------------------------------------------
    history_data = {
        "class_to_idx": base_dataset.class_to_idx,
        "class_counts": class_counts,
        "train_loss": [m.loss for m in history.train_metrics],
        "train_acc": [m.accuracy for m in history.train_metrics],
        "val_loss": [m.loss for m in history.val_metrics],
        "val_acc": [m.accuracy for m in history.val_metrics],
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "train_samples": len(train_indices),
        "val_samples": len(val_indices),
    }
    history_path = RESULTS_DIR / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history_data, f, indent=2)
    print(f"Saved training history to: {history_path.resolve()}")

    # ------------------------------------------------------------------
    # Run inference on the full validation set (best model) and save
    # predictions + true labels + file paths for the dashboard's
    # confusion matrix and sample-image viewer.
    # ------------------------------------------------------------------
    print("Running inference on validation set for dashboard predictions...")
    all_true_labels = []
    all_pred_labels = []
    all_confidences = []

    model.eval()
    with torch.no_grad():
        for batch_images, batch_labels in val_loader:
            batch_images = batch_images.to(trainer.device)
            logits = model(batch_images)
            probs = torch.softmax(logits, dim=1)
            confidences, predictions = torch.max(probs, dim=1)

            all_true_labels.extend(batch_labels.tolist())
            all_pred_labels.extend(predictions.cpu().tolist())
            all_confidences.extend(confidences.cpu().tolist())

    predictions_data = {
        "class_to_idx": base_dataset.class_to_idx,
        "true_labels": all_true_labels,
        "pred_labels": all_pred_labels,
        "confidences": all_confidences,
        "filepaths": [rec["filepath"] for rec in val_file_records],
    }
    predictions_path = RESULTS_DIR / "val_predictions.json"
    with open(predictions_path, "w") as f:
        json.dump(predictions_data, f, indent=2)
    print(f"Saved validation predictions to: {predictions_path.resolve()}")

    print("\nDone. Launch the dashboard with: streamlit run dashboard.py")


if __name__ == "__main__":
    main()