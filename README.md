# Multi-Modal Remote Sensing & Satellite Analytics Pipeline

An end-to-end PyTorch curriculum spanning tabular MLPs, CNNs, transfer
learning, LSTMs/GRUs, and a from-scratch Vision Transformer, applied to
satellite and environmental data: soil-moisture regression, multi-spectral
land-cover classification, and NDVI/EVI time-series forecasting.

## Project structure

```
src/
  datasets/
    climate_dataset.py    ClimateDataset      - tabular ERA5-Land/Open-Meteo -> soil moisture regression
    eurosat_dataset.py     EuroSATDataset      - N-channel Sentinel-2 raster patches -> land-cover classification
    sequence_dataset.py    NDVISequenceDataset - sliding-window NDVI/EVI series -> multi-step forecasting
  models/
    tabular_mlp.py         SoilMoistureMLP       - configurable MLP (Linear -> BatchNorm -> ReLU -> Dropout)
    custom_cnn.py           CustomCNN             - from-scratch 2D CNN classifier
    transfer_resnet.py      MultiSpectralResNet   - ImageNet ResNet-50 adapted to N spectral bands
    sequence_lstm.py         NDVILSTMForecaster    - stacked LSTM/GRU, direct multi-horizon forecasting
    spatial_vit.py           MinimalGeospatialViT  - ViT from scratch (PatchEmbedding, [CLS], TransformerEncoder)
  training/
    trainer.py              ModelTrainer          - shared device-agnostic train/eval engine for every model above
requirements.txt
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`rasterio` is only needed if you point `EuroSATDataset` at real `.tif`
Sentinel-2 chips. Without it, the dataset falls back to `.npy` patch files
(useful for synthetic/testing data, as shown in its own smoke test).

## Running each module standalone

Every file has a runnable `if __name__ == "__main__":` smoke test that
builds dummy tensors/dataframes and asserts expected output shapes — no
real data or GPU required:

```bash
python3 src/datasets/climate_dataset.py
python3 src/datasets/eurosat_dataset.py
python3 src/datasets/sequence_dataset.py
python3 src/models/tabular_mlp.py
python3 src/models/custom_cnn.py
python3 src/models/transfer_resnet.py     # downloads ImageNet weights on first run
python3 src/models/sequence_lstm.py
python3 src/models/spatial_vit.py
python3 src/training/trainer.py
```

## Wiring a full training run

Each dataset pairs with a specific model/task combination:

| Dataset | Model | Task type | Loss |
|---|---|---|---|
| `ClimateDataset` | `SoilMoistureMLP` | `regression` | `nn.MSELoss()` |
| `EuroSATDataset` | `CustomCNN` / `MultiSpectralResNet` / `MinimalGeospatialViT` | `classification` | `nn.CrossEntropyLoss()` |
| `NDVISequenceDataset` | `NDVILSTMForecaster` | `regression` | `nn.MSELoss()` |

Example — soil moisture regression end-to-end:

```python
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.datasets.climate_dataset import ClimateDataset
from src.models.tabular_mlp import SoilMoistureMLP
from src.training.trainer import ModelTrainer

train_ds = ClimateDataset("era5_land_train.csv", feature_columns=[...], target_column="swvl1")
val_ds = ClimateDataset(
    "era5_land_val.csv", feature_columns=train_ds.feature_columns, target_column="swvl1",
    feature_mean=train_ds.feature_mean, feature_std=train_ds.feature_std,
)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

model = SoilMoistureMLP(input_dim=train_ds.num_features, hidden_dims=[128, 64, 32])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

trainer = ModelTrainer(model=model, optimizer=optimizer, loss_fn=nn.MSELoss(), task_type="regression")
history = trainer.fit(train_loader, val_loader, num_epochs=30)
trainer.restore_best_model()
trainer.save_checkpoint("soil_moisture_mlp.pt")
```

Swap in `EuroSATDataset` + `CustomCNN`/`MultiSpectralResNet`/`MinimalGeospatialViT`
with `task_type="classification"` and `nn.CrossEntropyLoss()`, or
`NDVISequenceDataset` + `NDVILSTMForecaster` with `task_type="regression"`,
using the same `ModelTrainer` — that consistency is the point of the shared
engine.

## Design notes

- **`MultiSpectralResNet`**: the pretrained `conv1` (3-channel) is not
  discarded — its RGB filters are averaged and tiled across all N target
  channels, giving fine-tuning a much better starting point than random
  init on limited labeled satellite data. `freeze_backbone()`,
  `unfreeze_last_n_blocks(n)`, and `unfreeze_all()` support a staged
  fine-tuning schedule.
- **`EuroSATDataset`** augmentations are deliberately geometry-only
  (flips/90° rotations) — no color jitter, since brightness/contrast
  transforms assume 3-channel visible-light images and would corrupt
  reflectance values in non-RGB bands (SWIR, NIR, etc.).
- **`NDVILSTMForecaster`** uses direct multi-horizon output (one forward
  pass emits all `horizon` future steps) rather than autoregressive
  rollout, avoiding compounding one-step error.
- **`ModelTrainer`** is task-parameterized rather than model-specific: the
  same class drives every architecture in this repo by switching
  `task_type` between `"classification"` and `"regression"`, which
  determines whether accuracy or RMSE/MAE is computed each epoch.

## Verification

All datasets and models were smoke-tested individually via their own
`__main__` blocks, and additionally verified together end-to-end (every
dataset feeding its corresponding model through `ModelTrainer`, running
real forward/backward/optimizer steps) before delivery. `MultiSpectralResNet`
was verified with `pretrained=False` in the sandboxed build environment,
since it lacked outbound access to download ImageNet weights — the
`pretrained=True` path downloads them via `torchvision.models.resnet50`
exactly as it does in any normal environment with internet access.
