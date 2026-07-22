"""
climate_dataset.py

PyTorch Dataset for tabular climate/weather vectors (e.g. ERA5-Land reanalysis
or Open-Meteo API pulls) used to regress a continuous soil-moisture target.

Typical feature columns:
    ['t2m', 'd2m', 'tp', 'ssrd', 'sp', 'u10', 'v10', 'swvl1_lag1', ...]
Typical target column:
    'swvl1'  (volumetric soil water layer 1, m3/m3)

Author: Principal ML Engineering Team
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ClimateDataset(Dataset):
    """Tabular weather-vector dataset for soil-moisture (or similar) regression.

    Wraps a pandas DataFrame (or a CSV/Parquet path) containing numeric
    climate covariates and a single continuous regression target. Handles
    train/inference-time standardization (z-score) using statistics computed
    on the *training* split only, to avoid data leakage.

    Args:
        data: Either a path to a `.csv` / `.parquet` file, or an in-memory
            `pandas.DataFrame` already containing the feature and target
            columns.
        feature_columns: Ordered list of column names to use as model input
            features. Order is preserved in the output tensor.
        target_column: Name of the continuous regression target column
            (e.g. soil moisture volumetric water content).
        feature_mean: Optional precomputed per-feature mean, shape
            `[num_features]`. If `None`, computed from `data` itself
            (only appropriate for the training split).
        feature_std: Optional precomputed per-feature std, shape
            `[num_features]`. If `None`, computed from `data` itself.
        target_mean: Optional precomputed target mean for standardizing the
            regression label. If `None`, target is left in raw units.
        target_std: Optional precomputed target std. If `None`, target is
            left in raw units.
        dropna: If True, rows with NaNs in feature/target columns are
            dropped at load time.

    Returns (via __getitem__):
        Tuple[torch.Tensor, torch.Tensor]:
            features: FloatTensor of shape [num_features]
            target:   FloatTensor scalar tensor of shape [1]

    Example:
        >>> train_ds = ClimateDataset(
        ...     data="era5_land_train.csv",
        ...     feature_columns=["t2m", "d2m", "tp", "ssrd", "sp", "u10", "v10"],
        ...     target_column="swvl1",
        ... )
        >>> val_ds = ClimateDataset(
        ...     data="era5_land_val.csv",
        ...     feature_columns=train_ds.feature_columns,
        ...     target_column="swvl1",
        ...     feature_mean=train_ds.feature_mean,
        ...     feature_std=train_ds.feature_std,
        ...     target_mean=train_ds.target_mean,
        ...     target_std=train_ds.target_std,
        ... )
    """

    def __init__(
        self,
        data: Union[str, Path, pd.DataFrame],
        feature_columns: Sequence[str],
        target_column: str,
        feature_mean: Optional[np.ndarray] = None,
        feature_std: Optional[np.ndarray] = None,
        target_mean: Optional[float] = None,
        target_std: Optional[float] = None,
        dropna: bool = True,
    ) -> None:
        super().__init__()

        self.feature_columns: List[str] = list(feature_columns)
        self.target_column: str = target_column

        self.df: pd.DataFrame = self._load_dataframe(data)

        required_cols = self.feature_columns + [self.target_column]
        missing = set(required_cols) - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns in data source: {missing}")

        if dropna:
            self.df = self.df.dropna(subset=required_cols).reset_index(drop=True)

        # Raw numpy views: [num_samples, num_features] and [num_samples]
        raw_features = self.df[self.feature_columns].to_numpy(dtype=np.float32)
        raw_target = self.df[self.target_column].to_numpy(dtype=np.float32)

        # --- Feature standardization (z-score) ---
        if feature_mean is None or feature_std is None:
            self.feature_mean = raw_features.mean(axis=0)
            self.feature_std = raw_features.std(axis=0)
        else:
            self.feature_mean = np.asarray(feature_mean, dtype=np.float32)
            self.feature_std = np.asarray(feature_std, dtype=np.float32)
        # Guard against divide-by-zero for constant columns.
        safe_std = np.where(self.feature_std < 1e-8, 1.0, self.feature_std)
        self.features = (raw_features - self.feature_mean) / safe_std  # [N, F]

        # --- Target standardization (optional) ---
        self.target_mean = target_mean
        self.target_std = target_std
        if self.target_mean is not None and self.target_std is not None:
            safe_t_std = self.target_std if abs(self.target_std) > 1e-8 else 1.0
            self.targets = (raw_target - self.target_mean) / safe_t_std  # [N]
        else:
            self.targets = raw_target  # [N], raw units

        self.num_features: int = len(self.feature_columns)

    @staticmethod
    def _load_dataframe(data: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
        """Loads a DataFrame from disk (CSV/Parquet) or passes one through."""
        if isinstance(data, pd.DataFrame):
            return data.copy()
        path = Path(data)
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        if path.suffix.lower() in {".csv", ".txt"}:
            return pd.read_csv(path)
        raise ValueError(f"Unsupported data source type: {path.suffix}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns a single (features, target) pair.

        Returns:
            features: torch.FloatTensor, shape [num_features]
            target:   torch.FloatTensor, shape [1]
        """
        feature_vec = self.features[index]  # [F]
        target_val = self.targets[index]  # scalar

        features_tensor = torch.from_numpy(feature_vec).float()  # [F]
        target_tensor = torch.tensor([target_val], dtype=torch.float32)  # [1]
        return features_tensor, target_tensor

    def get_normalization_stats(self) -> Dict[str, Optional[np.ndarray]]:
        """Returns fitted normalization statistics to reuse on other splits."""
        return {
            "feature_mean": self.feature_mean,
            "feature_std": self.feature_std,
            "target_mean": self.target_mean,
            "target_std": self.target_std,
        }


def _build_dummy_dataframe(num_rows: int = 256) -> pd.DataFrame:
    """Creates a synthetic ERA5-Land-like DataFrame for smoke testing."""
    rng = np.random.default_rng(seed=42)
    df = pd.DataFrame(
        {
            "t2m": rng.normal(290, 8, num_rows),  # 2m temperature (K)
            "d2m": rng.normal(280, 6, num_rows),  # 2m dewpoint (K)
            "tp": rng.exponential(0.002, num_rows),  # total precipitation (m)
            "ssrd": rng.normal(1.5e7, 3e6, num_rows),  # solar radiation
            "sp": rng.normal(101000, 500, num_rows),  # surface pressure (Pa)
            "u10": rng.normal(0, 3, num_rows),  # 10m wind u
            "v10": rng.normal(0, 3, num_rows),  # 10m wind v
        }
    )
    # Synthetic soil moisture target loosely correlated with precip/temp.
    df["swvl1"] = (
        0.25
        + 5.0 * df["tp"]
        - 0.0005 * (df["t2m"] - 290)
        + rng.normal(0, 0.02, num_rows)
    ).clip(0.02, 0.55)
    return df


if __name__ == "__main__":
    print("=== ClimateDataset smoke test ===")

    dummy_df = _build_dummy_dataframe(num_rows=256)
    feature_cols = ["t2m", "d2m", "tp", "ssrd", "sp", "u10", "v10"]

    train_df = dummy_df.iloc[:200].reset_index(drop=True)
    val_df = dummy_df.iloc[200:].reset_index(drop=True)

    train_dataset = ClimateDataset(
        data=train_df,
        feature_columns=feature_cols,
        target_column="swvl1",
    )
    stats = train_dataset.get_normalization_stats()

    val_dataset = ClimateDataset(
        data=val_df,
        feature_columns=feature_cols,
        target_column="swvl1",
        feature_mean=stats["feature_mean"],
        feature_std=stats["feature_std"],
    )

    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    features, target = train_dataset[0]
    print(f"Single sample -> features: {tuple(features.shape)}, target: {tuple(target.shape)}")
    assert features.shape == (len(feature_cols),)
    assert target.shape == (1,)

    from torch.utils.data import DataLoader

    loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    batch_features, batch_targets = next(iter(loader))
    print(f"Batch -> features: {tuple(batch_features.shape)}, targets: {tuple(batch_targets.shape)}")
    assert batch_features.shape == (32, len(feature_cols))
    assert batch_targets.shape == (32, 1)

    print("All ClimateDataset assertions passed.")
