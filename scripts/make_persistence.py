"""Build a persistence forecast ("the weather stays as it is now") matching another forecast's shape.

Lead time 0 of our forecasts is the ERA5 initial state, so persistence is that state repeated at
every lead. (WB2's own --evaluate_persistence only works for by-valid forecasts, not our format.)

Usage: python scripts/make_persistence.py FORECAST.zarr PERSISTENCE.zarr
"""
import sys

import xarray as xr


def main():
    src, dst = sys.argv[1:3]
    forecast = xr.open_zarr(src)
    initial = forecast.isel(prediction_timedelta=0, drop=True)
    persistence = initial.expand_dims(prediction_timedelta=forecast.prediction_timedelta).transpose(*forecast.dims)
    for var in persistence.variables.values():
        var.encoding = {}
    # One forecast per chunk, like our forecasts: with 39 channels a 100-forecast chunk is ~250 MB per variable.
    persistence.chunk({"time": 1}).to_zarr(dst, mode="w")
    print(f"Wrote {dst}")


if __name__ == "__main__":
    main()
