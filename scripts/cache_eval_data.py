"""Download the ERA5 truth and climatology that scripts/evaluate.sh scores against, for a local run.

The WB2 evaluation reads one forecast start time at a time, and every start time needs ERA5 at all
of its 41 valid times. Streaming from GCS, each read downloads whole 100-timestep chunks with all 13
pressure levels, over and over. This script keeps only the period, variables and levels you evaluate,
so the evaluation reads a few hundred MB from local disk instead.

Example (the test period and variables of Phases 1-2):
    python scripts/cache_eval_data.py --start 2020-01-01 --end 2020-12-31 \
        --variables 2m_temperature geopotential --levels 500

evaluate.sh then uses data/eval/obs.zarr and data/eval/climatology.zarr automatically. The cache only
covers what you asked for: evaluating other variables, levels or dates needs a new cache (or
OBS_PATH/CLIMATOLOGY_PATH pointing back at GCS).
"""
import argparse
import shutil
from pathlib import Path

import xarray as xr

OBS = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr"
CLIMATOLOGY = "gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_64x32_equiangular_conservative.zarr"


def subset(ds, variables, levels):
    ds = ds[variables]
    return ds.sel(level=levels) if "level" in ds.dims else ds


def write(ds, path):
    """Load into memory and write atomically (via a temp dir), dropping the source chunking."""
    tmp = path.with_name(path.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    ds = ds.load()
    for var in ds.variables.values():
        var.encoding = {}
    ds.to_zarr(tmp, mode="w")
    shutil.rmtree(path, ignore_errors=True)
    tmp.rename(path)
    print(f"Wrote {path} ({ds.nbytes / 1e6:.0f} MB)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default="2020-01-01", help="first valid time to keep")
    p.add_argument("--end", default="2020-12-31", help="last valid time to keep (inclusive)")
    p.add_argument("--variables", nargs="+", default=["2m_temperature", "geopotential"])
    p.add_argument("--levels", nargs="+", type=int, default=[500], help="pressure levels for level variables")
    p.add_argument("--output", default="data/eval")
    args = p.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    storage = {"token": "anon"}
    obs = xr.open_zarr(OBS, storage_options=storage).sel(time=slice(args.start, args.end))
    write(subset(obs, args.variables, args.levels), out / "obs.zarr")
    # The climatology has no time axis (it's indexed by day of year and hour), so it's kept whole.
    write(subset(xr.open_zarr(CLIMATOLOGY, storage_options=storage), args.variables, args.levels),
          out / "climatology.zarr")


if __name__ == "__main__":
    main()
