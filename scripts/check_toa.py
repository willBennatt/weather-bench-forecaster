"""Check the analytic TOA solar radiation (wbforecaster.forcings.toa) against ERA5, and plot it.

ERA5's `toa_incident_solar_radiation` is the incoming solar radiation at the top of the atmosphere,
accumulated over the hour *before* each timestamp (J/m², so divide by 3600 for the mean W/m²) and
averaged over each 5.625° grid cell (the 64x32 data is conservatively regridded from 0.25°).
The store the rest of the project uses doesn't have it (its radiation variables are all NaN), so
this script reads it from the older WB2 64x32 store, which is on the same grid. `toa()` is instantaneous at a point, so this
script compares ERA5 against `toa()` both ways: at the cell center and time, and averaged over the
same hour and cell.

Example:
    python scripts/check_toa.py --output docs/images/phase2_toa.png
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np

from wbforecaster.data import open_era5
from wbforecaster.forcings import toa

ERA5_WITH_TOA = "gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr"
# Equinoxes and solstices in 2020, plus dates near the start and end of the training period to catch
# day-of-year drift over the years (e.g. from leap days).
DATES = ["1979-01-01", "2015-12-31", "2020-03-20", "2020-06-21", "2020-09-22", "2020-12-21"]
HOURS = [0, 6, 12, 18]


def averaged_toa(lat, lon, time, sub_points=5, sub_minutes=12):
    """toa() averaged over each grid cell and over the hour before `time`, like ERA5's mean flux."""
    dlat, dlon = lat[1] - lat[0], lon[1] - lon[0]
    offsets = (np.arange(sub_points) + 0.5) / sub_points - 0.5  # evenly spaced within a cell
    fine_lat = (lat[:, None] + offsets * dlat).ravel()
    fine_lon = (lon[:, None] + offsets * dlon).ravel()
    minutes = np.arange(sub_minutes) * 60 / sub_minutes + 60 / sub_minutes / 2 - 60
    times = np.datetime64(time, "s") + (minutes * 60).astype("timedelta64[s]")
    fine = toa(fine_lat, fine_lon, times).mean(axis=0)  # time average, (Lat * sub, Lon * sub)
    # Cell average, weighting each sub-point by its area, cos(latitude).
    weights = np.cos(np.radians(fine_lat)).reshape(len(lat), sub_points, 1, 1)
    fine = fine.reshape(len(lat), sub_points, len(lon), sub_points)
    return (fine * weights).sum(axis=(1, 3)) / (weights.sum(axis=1)[:, 0] * sub_points)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", default="docs/images/phase2_toa.png")
    args = p.parse_args()

    ds = open_era5(ERA5_WITH_TOA)
    lat, lon = ds.latitude.values, ds.longitude.values
    times = np.array([np.datetime64(f"{d}T{h:02d}") for d in DATES for h in HOURS])
    era5 = ds["toa_incident_solar_radiation"].sel(time=times).transpose("time", "latitude", "longitude").values / 3600

    print(f"{'time':<20} {'ERA5 mean':>9} {'RMSE point':>10} {'RMSE avg':>8} {'max |err| avg':>13}  (W/m²)")
    rows = []
    for t, truth in zip(times, era5):
        point = toa(lat, lon, t)[0]
        avg = averaged_toa(lat, lon, t)
        rows.append((point - truth, avg - truth))
        print(f"{str(t):<20} {truth.mean():9.1f} {np.sqrt(((point - truth) ** 2).mean()):10.1f} "
              f"{np.sqrt(((avg - truth) ** 2).mean()):8.1f} {np.abs(avg - truth).max():13.1f}")
    point_err, avg_err = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
    print(f"Overall RMSE: {np.sqrt((point_err ** 2).mean()):.1f} at a point, "
          f"{np.sqrt((avg_err ** 2).mean()):.1f} averaged (ERA5 peaks at ~{era5.max():.0f})")

    # Plot one day: toa() at the cell centers, ERA5, and their difference, at 00z and 12z.
    day = "2020-06-21"
    fig, axes = plt.subplots(2, 3, figsize=(15, 6), constrained_layout=True)
    extent = [lon[0], lon[-1], lat[0], lat[-1]]
    rows = [list(times).index(np.datetime64(f"{day}T{hour:02d}")) for hour in [0, 12]]
    # The errors are a few W/m², so scale the difference panels to the largest one shown (shared by both
    # rows so they can be compared), centered on 0 so white means no error.
    err_max = np.abs(avg_err[rows]).max()
    for row, (hour, i) in enumerate(zip([0, 12], rows)):
        panels = [(toa(lat, lon, times[i])[0], "toa() at cell centers"), (era5[i], "ERA5 (hour and cell mean)"),
                  (avg_err[i], "toa() averaged − ERA5")]
        for ax, (field, title) in zip(axes[row], panels):
            diff = title.endswith("ERA5")
            kwargs = ({"cmap": "RdBu_r", "vmin": -err_max, "vmax": err_max} if diff
                      else {"cmap": "magma", "vmin": 0, "vmax": 1400})
            im = ax.imshow(field, origin="lower", extent=extent, aspect="auto", **kwargs)
            ax.set_title(f"{title}, {day} {hour:02d}z")
            ax.set_xlabel("longitude (°E)")
            ax.set_ylabel("latitude (°N)")
            fig.colorbar(im, ax=ax, label="W/m²")
    fig.savefig(args.output, dpi=120)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
