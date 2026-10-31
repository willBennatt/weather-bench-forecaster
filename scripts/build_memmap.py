"""Write normalized float16 training and validation data to disk, so training reads batches from disk.

The Phase 3 channel set (39 channels) for 1979-2015 is ~17 GB as float32: more RAM than a laptop or a
free Colab session has, and the in-memory pipeline holds about three copies while normalizing. This
script works a year at a time, so it never holds more than one year (~0.5 GB) in memory:

1. Computes each channel's mean and std over the training period, and the mean and std of its
   6-hour changes (for tendency normalization: scaling the model's output by a typical change).
2. Writes each split, normalized with the training statistics, as a float16 .npy file (~8.6 GB for
   1979-2015). float16 keeps ~3 significant digits, which is plenty for z-scores, but not for raw
   values like geopotential (~55,000), so the data is normalized before it's stored.

Training opens the files as memory maps: each batch reads only its own samples from disk, and the
operating system caches what fits in RAM.

Example (from a download_era5.py cache with the Phase 3 channels):
    python scripts/build_memmap.py --data data/era5_64x32_phase3 --output data/memmap/phase3 \\
        --channels 2m_temperature geopotential@500 ... \\
        --train-start 1979-01-01 --train-end 2015-12-31 --val-start 2016-01-01 --val-end 2017-12-31

Then train with:
    python scripts/train_and_forecast.py --data data/era5_64x32_phase3 --memmap data/memmap/phase3 ...

Output: stats.npz (channels, mean, std, tendency_mean, tendency_std, in physical units), and
train.npy + train_times.npy (likewise val). Re-running skips splits that are already complete, so an
interrupted build can be resumed, and adds the tendency statistics to a cache built without them.
"""
import argparse
from pathlib import Path

import numpy as np

from wbforecaster.data import DEFAULT_CHANNELS, Normalizer, extract_channels, open_era5

CHUNK = 1460  # time steps loaded at once: one year of 6-hourly data


def chunks(period):
    for i in range(0, len(period.time), CHUNK):
        yield i, period.isel(time=slice(i, i + CHUNK))


def channel_stats(period, channels):
    """Per-channel mean and std of the fields and of their 6-hour changes (tendencies) over `period`.

    Accumulated a chunk at a time in float64. The tendencies are x[t+1] - x[t] for consecutive time
    steps: within each chunk, plus the step from the last time of one chunk to the first of the next.
    Returns (mean, std, tendency_mean, tendency_std), each of shape (1, C, 1, 1), in physical units.
    """
    total = total_sq = diff_total = diff_total_sq = 0.0
    count = diff_count = 0
    last = None  # the previous chunk's last time step
    for i, chunk in chunks(period):
        x = extract_channels(chunk, channels)
        total = total + x.sum(axis=(0, 2, 3), dtype=np.float64)
        total_sq = total_sq + np.square(x, dtype=np.float64).sum(axis=(0, 2, 3))
        count += x.shape[0] * x.shape[2] * x.shape[3]

        diffs = np.diff(x if last is None else np.concatenate([last, x]), axis=0)
        diff_total = diff_total + diffs.sum(axis=(0, 2, 3), dtype=np.float64)
        diff_total_sq = diff_total_sq + np.square(diffs, dtype=np.float64).sum(axis=(0, 2, 3))
        diff_count += diffs.shape[0] * diffs.shape[2] * diffs.shape[3]
        last = x[-1:]
        print(f"  stats: {i + len(x)}/{len(period.time)} time steps", flush=True)

    def mean_std(total, total_sq, count):
        mean = total / count
        std = np.sqrt(np.maximum(total_sq / count - mean**2, 0))
        # Same shape and epsilon as Normalizer, so the two are interchangeable.
        shape = (1, len(channels), 1, 1)
        return mean.reshape(shape).astype(np.float32), (std.reshape(shape) + 1e-6).astype(np.float32)

    return (*mean_std(total, total_sq, count), *mean_std(diff_total, diff_total_sq, diff_count))


def write_split(period, channels, normalizer, out, name):
    """Write `period` normalized as out/{name}.npy (float16, (T, C, Lat, Lon)), then out/{name}_times.npy."""
    data_path, times_path = out / f"{name}.npy", out / f"{name}_times.npy"
    if times_path.exists():
        print(f"{name}: already built")
        return
    tmp = out / f"{name}.tmp.npy"
    shape = (len(period.time), len(channels), len(period.latitude), len(period.longitude))
    data = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=shape)
    for i, chunk in chunks(period):
        data[i : i + len(chunk.time)] = normalizer.normalize(extract_channels(chunk, channels))
        print(f"  {name}: {i + len(chunk.time)}/{shape[0]} time steps", flush=True)
    data.flush()
    del data
    tmp.rename(data_path)
    np.save(times_path, period.time.values)  # last, marking the split complete
    print(f"{name}: wrote {data_path} ({data_path.stat().st_size / 1e6:,.0f} MB)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="a download_era5.py cache (or the WB2 bucket URL)")
    p.add_argument("--channels", nargs="+", default=DEFAULT_CHANNELS, help="e.g. 2m_temperature geopotential@500")
    p.add_argument("--train-start", default="1979-01-01")
    p.add_argument("--train-end", default="2015-12-31")
    p.add_argument("--val-start", default="2016-01-01")
    p.add_argument("--val-end", default="2017-12-31")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    ds = open_era5(args.data)
    train = ds.sel(time=slice(args.train_start, args.train_end))
    val = ds.sel(time=slice(args.val_start, args.val_end))

    stats_path = out / "stats.npz"
    stats = dict(np.load(stats_path)) if stats_path.exists() else None
    if stats is not None and list(stats["channels"]) != args.channels:
        p.error(f"{out} was built with different channels; use a new --output")
    if stats is None or "tendency_std" not in stats:
        # Caches built before the tendency statistics existed get them added here; their stored mean
        # and std are kept, since train.npy and val.npy were normalized with them.
        print("Computing normalization statistics on the training period...")
        mean, std, tendency_mean, tendency_std = channel_stats(train, args.channels)
        if stats is not None:
            mean, std = stats["mean"], stats["std"]
        np.savez(stats_path, channels=np.array(args.channels), mean=mean, std=std,
                 tendency_mean=tendency_mean, tendency_std=tendency_std)
    stats = np.load(stats_path)
    mean, std = stats["mean"], stats["std"]
    normalizer = Normalizer.from_stats(mean, std)

    write_split(train, args.channels, normalizer, out, "train")
    write_split(val, args.channels, normalizer, out, "val")


if __name__ == "__main__":
    main()
