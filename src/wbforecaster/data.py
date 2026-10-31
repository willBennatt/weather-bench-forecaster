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


# Floor on the tendency scale (std of 6-hour changes / std of the field). The loss weights each
# channel by 1 / scale**2, so a channel that barely changes would otherwise dominate the loss by
# fitting its noise. 0.01 caps that weight at 10,000; the Phase 3 channels range from 0.035
# (geopotential@50) to 0.76 (specific_humidity@50), so it only affects near-constant channels.
TENDENCY_SCALE_MIN = 0.01


class Normalizer:
    """Per-channel z-score normalization using statistics from the training set only.

    Also keeps the mean and std of each channel's 6-hour changes (tendencies), for tendency
    normalization; see `tendency_scale`.
    """

    def __init__(self, train_raw):
        self.mean = train_raw.mean(axis=(0, 2, 3), keepdims=True)
        self.std = train_raw.std(axis=(0, 2, 3), keepdims=True) + 1e-6
        # One channel at a time, so the differences never need a second copy of the whole array.
        diffs = [np.diff(train_raw[:, c], axis=0) for c in range(train_raw.shape[1])]
        self.tendency_mean = np.array([d.mean(dtype=np.float64) for d in diffs], dtype=np.float32).reshape(self.mean.shape)
        self.tendency_std = np.array([d.std(dtype=np.float64) for d in diffs], dtype=np.float32).reshape(self.std.shape) + 1e-6

    @classmethod
    def from_stats(cls, mean, std, tendency_mean=None, tendency_std=None):
        """A normalizer with statistics computed elsewhere, e.g. a year at a time by build_memmap.py."""
        normalizer = cls.__new__(cls)
        normalizer.mean, normalizer.std = mean, std
        normalizer.tendency_mean, normalizer.tendency_std = tendency_mean, tendency_std
        return normalizer

    def tendency_scale(self, min_scale=TENDENCY_SCALE_MIN):
        """(scale, shift) per channel: a typical 6-hour change, and the mean change, in normalized units.

        scale = std of the changes / std of the field, at least `min_scale`; shift = mean change / std of the field.
        """
        if self.tendency_std is None:
            raise ValueError("no tendency statistics: rebuild the cache with build_memmap.py")
        scale = np.maximum(self.tendency_std / self.std, min_scale)
        return scale.ravel(), (self.tendency_mean / self.std).ravel()

    def normalize(self, x):
        return (x - self.mean) / self.std

    def denormalize(self, x):
        return x * self.std + self.mean


def open_memmap_splits(directory):
    """Open a build_memmap.py cache: its channel names, a Normalizer, and {split: (data, times)}.

    Each `data` is a normalized float16 array of shape (T, C, Lat, Lon), memory-mapped: rows are read
    from disk when indexed, so the full array never has to fit in RAM.
    """
    directory = Path(directory)
    stats = np.load(directory / "stats.npz")
    splits = {}
    # The times file is written last, so a split interrupted mid-write isn't listed.
    for times_path in sorted(directory.glob("*_times.npy")):
        name = times_path.name.removesuffix("_times.npy")
        splits[name] = (np.load(directory / f"{name}.npy", mmap_mode="r"), np.load(times_path))
    normalizer = Normalizer.from_stats(stats["mean"], stats["std"], stats.get("tendency_mean"), stats.get("tendency_std"))
    return [str(c) for c in stats["channels"]], normalizer, splits


def _as_tensor(x):
    """Rows of a memmap come back as float16 numpy arrays; tensors pass through unchanged."""
    return torch.from_numpy(np.asarray(x, dtype=np.float32)) if isinstance(x, np.ndarray) else x


class OneStepDataset(Dataset):
    """Consecutive (x_t, x_{t+1}) pairs for training a 6-hour step model.

    If `forcing_tensor` (T, F, Lat, Lon) is given, the input is the state with the forcings for
    that step appended as extra channels: (cat(x_t, f_t), x_{t+1}). The forcings are never targets.

    `data_tensor` can also be a numpy array or memmap (e.g. from `open_memmap_splits`): each sample
    is then read and converted to float32 only when it's requested.
    """

    def __init__(self, data_tensor, forcing_tensor=None):
        self.data = data_tensor  # (T, C, Lat, Lon)
        self.forcing = forcing_tensor
        if forcing_tensor is not None:
            assert len(forcing_tensor) == len(data_tensor), "forcings must cover the same times as the data"

    def __len__(self):
        return len(self.data) - 1

    def __getitem__(self, idx):
        x, y = _as_tensor(self.data[idx]), _as_tensor(self.data[idx + 1])
        if self.forcing is not None:
            return torch.cat([x, self.forcing[idx]]), y
        return x, y


def to_tensor(array):
    return torch.tensor(array, dtype=torch.float32)
