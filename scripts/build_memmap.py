"""Write normalized float16 training and validation data to disk, so training reads batches from disk.

The Phase 3 channel set (39 channels) for 1979-2015 is ~17 GB as float32: more RAM than a laptop or a
free Colab session has, and the in-memory pipeline holds about three copies while normalizing. This
script works a year at a time, so it never holds more than one year (~0.5 GB) in memory:

1. Computes each channel's mean and std over the training period.
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

Output: stats.npz (channels, mean, std), and train.npy + train_times.npy (likewise val). Re-running
skips splits that are already complete, so an interrupted build can be resumed.
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
    """Per-channel mean and std over `period`, accumulated a chunk at a time in float64."""
    total = total_sq = 0.0
    count = 0
    for i, chunk in chunks(period):
        x = extract_channels(chunk, channels)
        total = total + x.sum(axis=(0, 2, 3), dtype=np.float64)
        total_sq = total_sq + np.square(x, dtype=np.float64).sum(axis=(0, 2, 3))
        count += x.shape[0] * x.shape[2] * x.shape[3]
        print(f"  stats: {i + len(x)}/{len(period.time)} time steps", flush=True)
    mean = total / count
    std = np.sqrt(total_sq / count - mean**2)
    # Same shape and epsilon as Normalizer, so the two are interchangeable.
    shape = (1, len(channels), 1, 1)
    return mean.reshape(shape).astype(np.float32), (std.reshape(shape) + 1e-6).astype(np.float32)


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
    if stats_path.exists():
        stats = np.load(stats_path)
        if list(stats["channels"]) != args.channels:
            p.error(f"{out} was built with different channels; use a new --output")
        mean, std = stats["mean"], stats["std"]
    else:
        print("Computing normalization statistics on the training period...")
        mean, std = channel_stats(train, args.channels)
        np.savez(stats_path, channels=np.array(args.channels), mean=mean, std=std)
    normalizer = Normalizer.from_stats(mean, std)

    write_split(train, args.channels, normalizer, out, "train")
    write_split(val, args.channels, normalizer, out, "val")


if __name__ == "__main__":
    main()
