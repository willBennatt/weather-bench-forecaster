"""Loading WeatherBench 2 ERA5 data into (time, channel, lat, lon) arrays."""
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

ERA5_64X32 = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr"
CLIMATOLOGY_64X32 = "gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_64x32_equiangular_conservative.zarr"

# A channel is "variable" for surface fields or "variable@level" for pressure-level fields.
DEFAULT_CHANNELS = ["2m_temperature", "geopotential@500"]


def open_era5(path=ERA5_64X32):
    """Open ERA5 lazily: the public WB2 bucket (no credentials needed) or a local cache from download_era5.py."""
    if path.startswith("gs://"):
        return xr.open_zarr(path, storage_options={"token": "anon"})
    local = Path(path)
    if (local / "static.zarr").exists():
        # A download_era5.py cache: one store per year plus the static fields.
        years = sorted(p for p in local.glob("*.zarr") if p.stem.isdigit())
        ds = xr.open_mfdataset(years, engine="zarr", combine="nested", concat_dim="time",
                               data_vars="minimal", coords="minimal", compat="override")
        return xr.merge([ds, xr.open_zarr(local / "static.zarr")])
    return xr.open_zarr(path)


def get_device():
    """CUDA on Colab/Kaggle, Apple-silicon GPU (MPS) on a Mac, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_channel(channel):
    """'geopotential@500' -> ('geopotential', 500); '2m_temperature' -> ('2m_temperature', None)."""
    if "@" in channel:
        var, level = channel.split("@")
        return var, int(level)
    return channel, None


def extract_channels(ds, channels=DEFAULT_CHANNELS):
    """Stack the requested channels into a float32 array of shape (T, C, Lat, Lon)."""
    arrays = []
    for channel in channels:
        var, level = parse_channel(channel)
        da = ds[var]
        if level is not None:
            da = da.sel(level=level)
        # WB2 stores (time, longitude, latitude); models expect (time, latitude, longitude).
        arrays.append(da.transpose("time", "latitude", "longitude").values)
    return np.stack(arrays, axis=1).astype(np.float32)


class Normalizer:
    """Per-channel z-score normalization using statistics from the training set only."""

    def __init__(self, train_raw):
        self.mean = train_raw.mean(axis=(0, 2, 3), keepdims=True)
        self.std = train_raw.std(axis=(0, 2, 3), keepdims=True) + 1e-6

    def normalize(self, x):
        return (x - self.mean) / self.std

    def denormalize(self, x):
        return x * self.std + self.mean


class OneStepDataset(Dataset):
    """Consecutive (x_t, x_{t+1}) pairs for training a 6-hour step model."""

    def __init__(self, data_tensor):
        self.data = data_tensor  # (T, C, Lat, Lon)

    def __len__(self):
        return len(self.data) - 1

    def __getitem__(self, idx):
        return self.data[idx], self.data[idx + 1]


def to_tensor(array):
    return torch.tensor(array, dtype=torch.float32)
