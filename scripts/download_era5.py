"""Download a subset of WB2 ERA5 64x32 to local disk, one Zarr store per year.

Streaming from GCS on every run is slow, and pressure-level variables are stored with all
13 levels in each chunk, so picking one level still downloads all of them. This script
downloads once and keeps only the channels and levels you ask for.

Example (the data Phase 1 needs, ~4 GB to download, a few hundred MB on disk):
    python scripts/download_era5.py --channels 2m_temperature geopotential@500 \
        --start-year 1979 --end-year 2020 --output data/era5_64x32

Then train from the cache:
    python scripts/train_and_forecast.py --data data/era5_64x32 ...

Re-running skips years already downloaded, so an interrupted download can be resumed.
Pressure-level variables share one `level` coordinate: every requested level is kept for
every pressure-level variable (e.g. asking for z@500 and t@850 stores both at 500 and 850).
"""
import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from wbforecaster.data import DEFAULT_CHANNELS, ERA5_64X32, open_era5, parse_channel

STATIC_VARS = ["land_sea_mask", "geopotential_at_surface"]


def select(ds, channels):
    """Subset the dataset to the requested variables, with the union of requested levels."""
    variables, levels = [], set()
    for channel in channels:
        var, level = parse_channel(channel)
        if var not in variables:
            variables.append(var)
        if level is not None:
            levels.add(level)
    subset = ds[variables]
    if levels:
        subset = subset.sel(level=sorted(levels))
    return subset


def write_store(subset, path):
    """Load into memory and write atomically (via a temp dir) so a crash never leaves a half-written year."""
    tmp = path.with_name(path.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    subset = subset.load()
    for var in subset.variables.values():
        var.encoding = {}  # drop the source chunking; use one chunk per year
    subset.to_zarr(tmp, mode="w")
    tmp.rename(path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--channels", nargs="+", default=DEFAULT_CHANNELS)
    p.add_argument("--start-year", type=int, default=1979)
    p.add_argument("--end-year", type=int, default=2020)
    p.add_argument("--output", default="data/era5_64x32")
    p.add_argument("--source", default=ERA5_64X32)
    p.add_argument("--workers", type=int, default=4, help="years downloaded in parallel")
    args = p.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    ds = open_era5(args.source)

    static_path = out / "static.zarr"
    if not static_path.exists():
        write_store(ds[STATIC_VARS], static_path)

    def download(year):
        path = out / f"{year}.zarr"
        if path.exists():
            return f"{year}: already downloaded"
        write_store(select(ds.sel(time=str(year)), args.channels), path)
        return f"{year}: done"

    years = range(args.start_year, args.end_year + 1)
    with ThreadPoolExecutor(args.workers) as pool:
        for message in pool.map(download, years):
            print(message, flush=True)


if __name__ == "__main__":
    main()
