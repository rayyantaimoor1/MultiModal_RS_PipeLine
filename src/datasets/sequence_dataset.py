"""
sequence_dataset.py

PyTorch Dataset that turns a multi-year satellite vegetation-index time
series (e.g. NDVI or EVI composites derived from Sentinel-2 / MODIS) into
sliding-window (input_sequence, forecast_target) pairs suitable for
sequence models such as LSTMs, GRUs, or temporal Transformers.

Supports:
    - Univariate series (NDVI only) or multivariate series (NDVI + EVI +
      auxiliary climate covariates per timestep).
    - Multi-step forecasting horizons (predict the next `horizon` steps,
      not just the next single step).
    - Multiple independent series (e.g. one series per field/pixel/region),
      windows are never generated across a series boundary.

Author: Principal ML Engineering Team
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


class NDVISequenceDataset(Dataset):
    """Sliding-window dataset for multi-step vegetation-index forecasting.

    Given one or more time series of shape [num_timesteps, num_features]
    (num_features >= 1, e.g. NDVI alone, or [NDVI, EVI, LST, precip, ...]),
    generates fixed-length input windows and their corresponding future
    targets via a sliding window with configurable stride.

    Args:
        series: A single array/tensor of shape [T, F], or a list of such
            arrays for multiple independent series (e.g. different field
            parcels). F is the number of per-timestep features; the first
            feature column is treated as the primary forecast target
            (e.g. NDVI) unless `target_index` is overridden.
        lookback: Number of past timesteps used as model input (window size).
        horizon: Number of future timesteps to predict.
        stride: Step size between consecutive windows. `stride=1` yields the
            maximum number of overlapping windows.
        target_index: Index into the feature dimension F that is the
            forecast target. Defaults to 0 (assumes NDVI is the first
            column). All F features are still returned as model input.
        feature_mean: Optional per-feature mean, shape [F], for
            standardization. If None, computed from `series`.
        feature_std: Optional per-feature std, shape [F].

    Returns (via __getitem__):
        Tuple[torch.Tensor, torch.Tensor]:
            input_seq: FloatTensor, shape [lookback, num_features]
            target_seq: FloatTensor, shape [horizon] (target feature only)

    Example:
        >>> ndvi_series = np.sin(np.linspace(0, 20, 240)) * 0.3 + 0.5  # [240]
        >>> ds = NDVISequenceDataset(series=ndvi_series[:, None], lookback=12, horizon=3)
        >>> x, y = ds[0]
        >>> x.shape, y.shape
        (torch.Size([12, 1]), torch.Size([3]))
    """

    def __init__(
        self,
        series: Union[np.ndarray, torch.Tensor, Sequence[np.ndarray]],
        lookback: int = 12,
        horizon: int = 3,
        stride: int = 1,
        target_index: int = 0,
        feature_mean: Optional[np.ndarray] = None,
        feature_std: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()

        if lookback <= 0 or horizon <= 0:
            raise ValueError("lookback and horizon must both be positive integers.")

        self.lookback = lookback
        self.horizon = horizon
        self.stride = stride
        self.target_index = target_index

        self.series_list: List[np.ndarray] = self._standardize_series_input(series)

        # Fit or reuse normalization stats across all provided series.
        if feature_mean is None or feature_std is None:
            concatenated = np.concatenate(self.series_list, axis=0)  # [sum_T, F]
            self.feature_mean = concatenated.mean(axis=0)
            self.feature_std = concatenated.std(axis=0)
        else:
            self.feature_mean = np.asarray(feature_mean, dtype=np.float32)
            self.feature_std = np.asarray(feature_std, dtype=np.float32)

        safe_std = np.where(self.feature_std < 1e-8, 1.0, self.feature_std)
        self.normalized_series: List[np.ndarray] = [
            (s - self.feature_mean) / safe_std for s in self.series_list
        ]

        self.num_features: int = self.series_list[0].shape[1]

        # Precompute (series_idx, window_start) index for every valid window.
        self.index_map: List[Tuple[int, int]] = self._build_index_map()

    @staticmethod
    def _standardize_series_input(
        series: Union[np.ndarray, torch.Tensor, Sequence[np.ndarray]]
    ) -> List[np.ndarray]:
        """Normalizes input into a list of [T, F] float32 numpy arrays."""
        if isinstance(series, torch.Tensor):
            series = series.numpy()

        if isinstance(series, np.ndarray):
            arr = series.astype(np.float32)
            if arr.ndim == 1:
                arr = arr[:, None]  # [T] -> [T, 1]
            return [arr]

        # Assume a sequence of per-series arrays.
        result: List[np.ndarray] = []
        for s in series:
            s_arr = np.asarray(s, dtype=np.float32)
            if s_arr.ndim == 1:
                s_arr = s_arr[:, None]
            result.append(s_arr)
        return result

    def _build_index_map(self) -> List[Tuple[int, int]]:
        """Builds a flat list of (series_index, window_start) for all valid windows."""
        index_map: List[Tuple[int, int]] = []
        window_length = self.lookback + self.horizon
        for series_idx, series_arr in enumerate(self.normalized_series):
            total_timesteps = series_arr.shape[0]
            last_valid_start = total_timesteps - window_length
            for start in range(0, max(last_valid_start, -1) + 1, self.stride):
                index_map.append((series_idx, start))
        return index_map

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns one (input_window, target_window) sliding-window pair.

        Returns:
            input_seq:  FloatTensor, shape [lookback, num_features]
            target_seq: FloatTensor, shape [horizon]
        """
        series_idx, start = self.index_map[index]
        series_arr = self.normalized_series[series_idx]  # [T, F]

        input_end = start + self.lookback
        target_end = input_end + self.horizon

        input_window = series_arr[start:input_end, :]  # [lookback, F]
        target_window = series_arr[input_end:target_end, self.target_index]  # [horizon]

        input_tensor = torch.from_numpy(input_window).float()  # [lookback, F]
        target_tensor = torch.from_numpy(target_window).float()  # [horizon]
        return input_tensor, target_tensor

    def get_normalization_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (feature_mean, feature_std) fitted on this dataset's series."""
        return self.feature_mean, self.feature_std

    def inverse_transform_target(self, normalized_values: torch.Tensor) -> torch.Tensor:
        """Converts standardized target predictions back to raw NDVI/EVI units.

        Args:
            normalized_values: Tensor of any shape containing z-scored target
                values (as produced by the model / this dataset).

        Returns:
            Tensor of the same shape in raw (denormalized) units.
        """
        mean = float(self.feature_mean[self.target_index])
        std = float(self.feature_std[self.target_index])
        return normalized_values * std + mean


def _build_dummy_ndvi_series(num_years: int = 4, points_per_year: int = 46) -> np.ndarray:
    """Simulates a seasonal NDVI curve (MODIS-like 8-day composites) with noise.

    Returns:
        np.ndarray, shape [num_years * points_per_year, 2] -> [NDVI, EVI]
    """
    rng = np.random.default_rng(seed=123)
    t = np.linspace(0, num_years * 2 * np.pi, num_years * points_per_year)
    seasonal_ndvi = 0.5 + 0.3 * np.sin(t - np.pi / 2)  # peaks mid-season
    trend = np.linspace(0, 0.03, len(t))  # slight greening trend over years
    noise = rng.normal(0, 0.02, len(t))
    ndvi = np.clip(seasonal_ndvi + trend + noise, -1.0, 1.0)
    evi = np.clip(0.8 * ndvi + rng.normal(0, 0.01, len(t)), -1.0, 1.0)
    return np.stack([ndvi, evi], axis=1).astype(np.float32)  # [T, 2]


if __name__ == "__main__":
    print("=== NDVISequenceDataset smoke test ===")

    LOOKBACK = 12
    HORIZON = 4

    # --- Single series case ---
    single_series = _build_dummy_ndvi_series(num_years=4)  # [T, 2] (NDVI, EVI)
    single_ds = NDVISequenceDataset(
        series=single_series, lookback=LOOKBACK, horizon=HORIZON, stride=1, target_index=0
    )
    print(f"Single-series windows: {len(single_ds)} | num_features: {single_ds.num_features}")

    x, y = single_ds[0]
    print(f"Single sample -> input: {tuple(x.shape)}, target: {tuple(y.shape)}")
    assert x.shape == (LOOKBACK, 2)
    assert y.shape == (HORIZON,)

    # --- Multi-series case (e.g. multiple field parcels) ---
    series_list = [_build_dummy_ndvi_series(num_years=3) for _ in range(3)]
    multi_ds = NDVISequenceDataset(
        series=series_list, lookback=LOOKBACK, horizon=HORIZON, stride=2, target_index=0
    )
    print(f"Multi-series windows: {len(multi_ds)}")

    from torch.utils.data import DataLoader

    loader = DataLoader(multi_ds, batch_size=8, shuffle=True)
    batch_x, batch_y = next(iter(loader))
    print(f"Batch -> input: {tuple(batch_x.shape)}, target: {tuple(batch_y.shape)}")
    assert batch_x.shape == (8, LOOKBACK, 2)
    assert batch_y.shape == (8, HORIZON)

    # --- Inverse transform sanity check ---
    denorm = multi_ds.inverse_transform_target(batch_y)
    print(f"Denormalized target sample range: [{denorm.min().item():.3f}, {denorm.max().item():.3f}]")

    print("All NDVISequenceDataset assertions passed.")
