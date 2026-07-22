"""
trainer.py

A reusable, device-agnostic training and evaluation engine for the
Multi-Modal Remote Sensing & Satellite Analytics Pipeline. A single
`ModelTrainer` class drives the full training loop for every model in the
curriculum (MLP, CNN, ResNet transfer-learning, LSTM/GRU forecaster, ViT),
for both classification tasks (land-cover mapping) and regression tasks
(soil-moisture / NDVI forecasting), by parameterizing the task type.

Responsibilities:
    - Forward pass, loss computation, backpropagation, optimizer step.
    - Epoch-level training and validation loops.
    - Metric computation: Accuracy (classification) or RMSE/MAE (regression).
    - Gradient clipping.
    - Model checkpointing (best-model tracking + manual save/load hooks).

Author: Principal ML Engineering Team
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import torch
from torch import nn
from torch.utils.data import DataLoader


TaskType = Literal["classification", "regression"]


@dataclass
class EpochMetrics:
    """Container for a single epoch's aggregated metrics.

    Attributes:
        loss: Mean loss over the epoch.
        accuracy: Classification accuracy in [0, 1]. `None` for regression tasks.
        rmse: Root-mean-squared-error. `None` for classification tasks.
        mae: Mean-absolute-error. `None` for classification tasks.
        num_samples: Total number of samples seen during the epoch.
    """

    loss: float
    accuracy: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    num_samples: int = 0

    def as_dict(self) -> Dict[str, float]:
        """Returns a flat dict of the populated (non-None) metric fields."""
        result: Dict[str, float] = {"loss": self.loss, "num_samples": float(self.num_samples)}
        if self.accuracy is not None:
            result["accuracy"] = self.accuracy
        if self.rmse is not None:
            result["rmse"] = self.rmse
        if self.mae is not None:
            result["mae"] = self.mae
        return result

    def __str__(self) -> str:
        parts = [f"loss={self.loss:.4f}"]
        if self.accuracy is not None:
            parts.append(f"acc={self.accuracy:.4f}")
        if self.rmse is not None:
            parts.append(f"rmse={self.rmse:.4f}")
        if self.mae is not None:
            parts.append(f"mae={self.mae:.4f}")
        return ", ".join(parts)


@dataclass
class TrainingHistory:
    """Accumulates per-epoch train/validation metrics across a training run."""

    train_metrics: List[EpochMetrics] = field(default_factory=list)
    val_metrics: List[EpochMetrics] = field(default_factory=list)

    def record(self, train: EpochMetrics, val: Optional[EpochMetrics]) -> None:
        self.train_metrics.append(train)
        if val is not None:
            self.val_metrics.append(val)


class ModelTrainer:
    """Device-agnostic training and evaluation engine.

    Wraps a model, optimizer, and loss function and provides epoch-level
    training/validation loops with automatic metric computation
    (accuracy for classification, RMSE/MAE for regression), gradient
    clipping, and checkpointing.

    Args:
        model: Any `nn.Module` whose forward pass maps
            `[batch_size, ...input_dims]` -> `[batch_size, ...output_dims]`
            (e.g. `[B, num_classes]` logits for classification, or
            `[B, 1]` / `[B, horizon]` for regression).
        optimizer: A configured `torch.optim.Optimizer` bound to
            `model.parameters()`.
        loss_fn: A callable `(predictions, targets) -> scalar loss tensor`.
            Typically `nn.CrossEntropyLoss()` for classification or
            `nn.MSELoss()` / `nn.L1Loss()` for regression.
        task_type: Either "classification" or "regression". Controls which
            metrics are computed each epoch.
        device: "cuda" or "cpu". If `None`, auto-detects CUDA availability.
        grad_clip_norm: If set, gradients are clipped to this max L2 norm
            via `torch.nn.utils.clip_grad_norm_` before each optimizer step.
            `None` disables gradient clipping.
        checkpoint_dir: Directory where `save_checkpoint` writes model
            state. Created on first use if it does not already exist.

    Example:
        >>> model = SoilMoistureMLP(input_dim=7, hidden_dims=[64, 32], output_dim=1)
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        >>> trainer = ModelTrainer(
        ...     model=model, optimizer=optimizer, loss_fn=nn.MSELoss(),
        ...     task_type="regression", grad_clip_norm=1.0,
        ... )
        >>> history = trainer.fit(train_loader, val_loader, num_epochs=10)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        task_type: TaskType = "classification",
        device: Optional[str] = None,
        grad_clip_norm: Optional[float] = 1.0,
        checkpoint_dir: Union[str, Path] = "checkpoints",
    ) -> None:
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.task_type: TaskType = task_type
        self.grad_clip_norm = grad_clip_norm

        self.checkpoint_dir = Path(checkpoint_dir)

        self.history = TrainingHistory()
        self.best_val_loss: float = float("inf")
        self.best_model_state: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Core step-level logic
    # ------------------------------------------------------------------

    def _move_batch_to_device(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Moves a single (inputs, targets) batch to `self.device`."""
        return inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)

    def train_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        """Performs one forward + backward + optimizer step on a single batch.

        Args:
            inputs: Model input batch, shape `[batch_size, ...]`.
            targets: Ground-truth batch. Shape depends on `task_type`:
                classification -> `[batch_size]` (LongTensor class indices);
                regression -> `[batch_size, output_dim]` (FloatTensor).

        Returns:
            The scalar loss value for this batch (Python float).
        """
        self.model.train()
        inputs, targets = self._move_batch_to_device(inputs, targets)

        self.optimizer.zero_grad()

        predictions = self.model(inputs)  # [B, ...output_dims]
        loss = self.loss_fn(predictions, targets)  # scalar

        loss.backward()

        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)

        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def eval_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> tuple[float, torch.Tensor, torch.Tensor]:
        """Performs one forward pass (no gradient) on a single batch for evaluation.

        Args:
            inputs: Model input batch, shape `[batch_size, ...]`.
            targets: Ground-truth batch (see `train_step` for shape conventions).

        Returns:
            Tuple[float, torch.Tensor, torch.Tensor]:
                loss: scalar batch loss (Python float).
                predictions: raw model output, shape `[batch_size, ...output_dims]`.
                targets: the (device-moved) targets, for metric aggregation.
        """
        self.model.eval()
        inputs, targets = self._move_batch_to_device(inputs, targets)

        predictions = self.model(inputs)  # [B, ...output_dims]
        loss = self.loss_fn(predictions, targets)  # scalar

        return loss.item(), predictions, targets

    # ------------------------------------------------------------------
    # Epoch-level loops
    # ------------------------------------------------------------------

    def run_epoch(self, data_loader: DataLoader, training: bool) -> EpochMetrics:
        """Runs one full pass over `data_loader`, either training or evaluating.

        Args:
            data_loader: Yields `(inputs, targets)` batches.
            training: If True, performs gradient updates via `train_step`.
                If False, runs `eval_step` under `torch.no_grad()`.

        Returns:
            EpochMetrics with aggregated loss and task-appropriate metrics.
        """
        total_loss = 0.0
        total_samples = 0

        # Accumulators for metric computation across the whole epoch.
        correct_predictions = 0
        squared_error_sum = 0.0
        absolute_error_sum = 0.0
        regression_value_count = 0

        for inputs, targets in data_loader:
            batch_size = inputs.shape[0]

            if training:
                batch_loss = self.train_step(inputs, targets)
                # Re-run a no-grad forward for metric bookkeeping to avoid
                # holding the autograd graph; cheap relative to the train step.
                with torch.no_grad():
                    self.model.eval()
                    inputs_dev, targets_dev = self._move_batch_to_device(inputs, targets)
                    predictions = self.model(inputs_dev)
                    self.model.train()
            else:
                batch_loss, predictions, targets_dev = self.eval_step(inputs, targets)

            total_loss += batch_loss * batch_size
            total_samples += batch_size

            if self.task_type == "classification":
                # predictions: [B, num_classes] logits; targets: [B] class indices
                predicted_labels = torch.argmax(predictions, dim=1)  # [B]
                correct_predictions += (predicted_labels == targets_dev).sum().item()
            else:
                # predictions / targets: [B, output_dim] (or broadcastable)
                diff = predictions - targets_dev
                squared_error_sum += (diff ** 2).sum().item()
                absolute_error_sum += diff.abs().sum().item()
                regression_value_count += diff.numel()

        mean_loss = total_loss / max(total_samples, 1)

        if self.task_type == "classification":
            accuracy = correct_predictions / max(total_samples, 1)
            return EpochMetrics(loss=mean_loss, accuracy=accuracy, num_samples=total_samples)

        rmse = (squared_error_sum / max(regression_value_count, 1)) ** 0.5
        mae = absolute_error_sum / max(regression_value_count, 1)
        return EpochMetrics(loss=mean_loss, rmse=rmse, mae=mae, num_samples=total_samples)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 10,
        verbose: bool = True,
        checkpoint_on_best: bool = True,
    ) -> TrainingHistory:
        """Runs the full multi-epoch training loop.

        Args:
            train_loader: Training set DataLoader.
            val_loader: Optional validation set DataLoader. If provided,
                validation metrics are computed each epoch and used for
                best-checkpoint tracking.
            num_epochs: Number of epochs to train for.
            verbose: If True, prints per-epoch metrics.
            checkpoint_on_best: If True and `val_loader` is provided, the
                best-performing model state (lowest validation loss) is
                kept in memory (`self.best_model_state`) after each epoch.

        Returns:
            TrainingHistory containing per-epoch train/val EpochMetrics.
        """
        for epoch in range(1, num_epochs + 1):
            train_metrics = self.run_epoch(train_loader, training=True)

            val_metrics: Optional[EpochMetrics] = None
            if val_loader is not None:
                val_metrics = self.run_epoch(val_loader, training=False)

                if checkpoint_on_best and val_metrics.loss < self.best_val_loss:
                    self.best_val_loss = val_metrics.loss
                    self.best_model_state = copy.deepcopy(self.model.state_dict())

            self.history.record(train_metrics, val_metrics)

            if verbose:
                message = f"Epoch {epoch:03d}/{num_epochs} | train: {train_metrics}"
                if val_metrics is not None:
                    message += f" | val: {val_metrics}"
                print(message)

        return self.history

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        filename: str = "checkpoint.pt",
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Saves model + optimizer state (and training history) to disk.

        Args:
            filename: Name of the checkpoint file, written under
                `self.checkpoint_dir`.
            extra_metadata: Optional additional key/value pairs to store
                alongside the checkpoint (e.g. epoch number, config dict).

        Returns:
            The full path the checkpoint was written to.
        """
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "task_type": self.task_type,
        }
        if extra_metadata:
            checkpoint["extra_metadata"] = extra_metadata

        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, filepath: Union[str, Path], load_optimizer: bool = True) -> Dict[str, Any]:
        """Loads model (and optionally optimizer) state from a checkpoint file.

        Args:
            filepath: Path to a checkpoint written by `save_checkpoint`.
            load_optimizer: If True, also restores optimizer state (e.g. to
                resume training). Set False when only doing inference or
                fine-tuning with a fresh optimizer.

        Returns:
            The raw checkpoint dict (useful for reading `extra_metadata`).
        """
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        return checkpoint

    def restore_best_model(self) -> None:
        """Restores the model's in-memory weights to the best checkpoint seen during `fit`."""
        if self.best_model_state is None:
            raise RuntimeError(
                "No best model state has been recorded yet. Call fit() with a "
                "val_loader and checkpoint_on_best=True first."
            )
        self.model.load_state_dict(self.best_model_state)

    # ------------------------------------------------------------------
    # Inference utility
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Runs inference on a batch of inputs (no gradients, eval mode).

        Args:
            inputs: FloatTensor, shape `[batch_size, ...input_dims]`.

        Returns:
            Raw model output, shape `[batch_size, ...output_dims]`, moved
            back to CPU.
        """
        self.model.eval()
        inputs = inputs.to(self.device)
        predictions = self.model(inputs)
        return predictions.cpu()


if __name__ == "__main__":
    import shutil
    import tempfile

    from torch.utils.data import TensorDataset

    print("=== ModelTrainer smoke test ===")

    torch.manual_seed(0)

    # ------------------------------------------------------------------
    # 1) Regression smoke test (mirrors SoilMoistureMLP usage)
    # ------------------------------------------------------------------
    print("\n--- Regression task (tabular MLP) ---")

    class TinyRegressionMLP(nn.Module):
        def __init__(self, input_dim: int = 7, output_dim: int = 1) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, output_dim)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    num_samples = 200
    input_dim = 7
    x_reg = torch.randn(num_samples, input_dim)
    true_weights = torch.randn(input_dim, 1)
    y_reg = x_reg @ true_weights + 0.1 * torch.randn(num_samples, 1)

    train_ds_reg = TensorDataset(x_reg[:160], y_reg[:160])
    val_ds_reg = TensorDataset(x_reg[160:], y_reg[160:])
    train_loader_reg = DataLoader(train_ds_reg, batch_size=16, shuffle=True)
    val_loader_reg = DataLoader(val_ds_reg, batch_size=16, shuffle=False)

    reg_model = TinyRegressionMLP(input_dim=input_dim, output_dim=1)
    reg_optimizer = torch.optim.Adam(reg_model.parameters(), lr=1e-2)

    tmp_ckpt_dir = Path(tempfile.mkdtemp(prefix="trainer_ckpt_"))
    try:
        reg_trainer = ModelTrainer(
            model=reg_model,
            optimizer=reg_optimizer,
            loss_fn=nn.MSELoss(),
            task_type="regression",
            device="cpu",
            grad_clip_norm=1.0,
            checkpoint_dir=tmp_ckpt_dir,
        )

        reg_history = reg_trainer.fit(train_loader_reg, val_loader_reg, num_epochs=5, verbose=True)

        assert len(reg_history.train_metrics) == 5
        assert len(reg_history.val_metrics) == 5
        assert reg_history.train_metrics[-1].rmse is not None
        assert reg_history.train_metrics[-1].mae is not None
        assert reg_history.train_metrics[-1].accuracy is None

        # Loss should have generally decreased over training on this simple linear task.
        assert reg_history.train_metrics[-1].loss < reg_history.train_metrics[0].loss

        ckpt_path = reg_trainer.save_checkpoint("regression_model.pt", extra_metadata={"epochs": 5})
        print(f"Saved regression checkpoint to: {ckpt_path}")
        assert ckpt_path.exists()

        reg_trainer.restore_best_model()

        loaded_checkpoint = reg_trainer.load_checkpoint(ckpt_path, load_optimizer=True)
        assert "extra_metadata" in loaded_checkpoint

        sample_preds = reg_trainer.predict(x_reg[:4])
        print(f"Sample regression predictions shape: {tuple(sample_preds.shape)}")
        assert sample_preds.shape == (4, 1)

        print("Regression ModelTrainer assertions passed.")

        # --------------------------------------------------------------
        # 2) Classification smoke test (mirrors CustomCNN/ViT usage)
        # --------------------------------------------------------------
        print("\n--- Classification task (tabular stand-in for CNN/ViT) ---")

        class TinyClassificationMLP(nn.Module):
            def __init__(self, input_dim: int = 10, num_classes: int = 4) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, num_classes)
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.net(x)

        num_classes = 4
        input_dim_cls = 10
        x_cls = torch.randn(num_samples, input_dim_cls)
        # Construct separable-ish classes via a fixed linear projection + argmax.
        class_projection = torch.randn(input_dim_cls, num_classes)
        y_cls = torch.argmax(x_cls @ class_projection, dim=1)  # [num_samples]

        train_ds_cls = TensorDataset(x_cls[:160], y_cls[:160])
        val_ds_cls = TensorDataset(x_cls[160:], y_cls[160:])
        train_loader_cls = DataLoader(train_ds_cls, batch_size=16, shuffle=True)
        val_loader_cls = DataLoader(val_ds_cls, batch_size=16, shuffle=False)

        cls_model = TinyClassificationMLP(input_dim=input_dim_cls, num_classes=num_classes)
        cls_optimizer = torch.optim.Adam(cls_model.parameters(), lr=1e-2)

        cls_trainer = ModelTrainer(
            model=cls_model,
            optimizer=cls_optimizer,
            loss_fn=nn.CrossEntropyLoss(),
            task_type="classification",
            device="cpu",
            grad_clip_norm=1.0,
            checkpoint_dir=tmp_ckpt_dir,
        )

        cls_history = cls_trainer.fit(train_loader_cls, val_loader_cls, num_epochs=5, verbose=True)

        assert len(cls_history.train_metrics) == 5
        assert cls_history.train_metrics[-1].accuracy is not None
        assert 0.0 <= cls_history.train_metrics[-1].accuracy <= 1.0
        assert cls_history.train_metrics[-1].rmse is None

        # Accuracy should improve meaningfully above random chance (1/num_classes) by the last epoch.
        assert cls_history.train_metrics[-1].accuracy > (1.0 / num_classes)

        print("Classification ModelTrainer assertions passed.")

    finally:
        shutil.rmtree(tmp_ckpt_dir, ignore_errors=True)

    print("\nAll ModelTrainer assertions passed.")
