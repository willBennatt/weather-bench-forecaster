"""Make the figures in docs/DATA.md from the local ERA5 cache (plus one small read from the WB2 bucket).

Usage: python scripts/make_data_figures.py --data data/era5_64x32 --out docs/images/data
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from wbforecaster.data import ERA5_64X32, open_era5  # noqa: E402

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#d9d8d4"
BLUE, ORANGE = "#2a78d6", "#eb6834"
SEQUENTIAL = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_red", ["#0d366b", "#256abf", "#86b6ef", "#f0efec", "#ee8f8a", "#e34948", "#7a1d1f"])
LAND = LinearSegmentedColormap.from_list("land", ["#cde2fb", "#d8cfb8"])

# Cell edges: centers are 5.625° apart; longitude centers start at 0°E, latitude at -87.1875°.
EXTENT = [-2.8125, 357.1875, -90, 90]
SNAPSHOT = "2020-01-01T00"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.edgecolor": GRID, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
})


def grid(da):
    """xarray (longitude, latitude) -> numpy (lat, lon), south at row 0."""
    return da.transpose("latitude", "longitude").values


def map_axes(ax, lsm, title):
    ax.contour(np.arange(64) * 5.625, lsm.latitude, grid(lsm), levels=[0.5], colors=INK, linewidths=0.6)
    ax.set_title(title, loc="left")
    ax.set_xticks([0, 90, 180, 270, 357.1875], ["0°E", "90°E", "180°", "270°E", ""])
    ax.set_yticks([-90, -45, 0, 45, 90], ["90°S", "45°S", "0°", "45°N", "90°N"])
    ax.set_xlim(EXTENT[:2])
    ax.set_ylim(EXTENT[2:])


def save(fig, out, name):
    path = out / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def fig_grid(static, out):
    lsm = static.land_sea_mask
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [2.6, 1]})
    ax.imshow(grid(lsm), origin="lower", extent=EXTENT, cmap=LAND, vmin=0, vmax=1, aspect="auto")
    for x in np.arange(65) * 5.625 - 2.8125:
        ax.axvline(x, color="white", lw=0.4)
    for y in np.arange(33) * 5.625 - 90:
        ax.axhline(y, color="white", lw=0.4)
    map_axes(ax, lsm, "The 64 × 32 grid: one value per 5.625° cell (land–sea mask shown)")
    ax.add_patch(Rectangle((-2.8125, 45), 5.625, 5.625, fill=False, ec=ORANGE, lw=2))
    ax.annotate("one cell ≈ 625 km × 420 km at 48°N", (3, 50), (25, 30), color=INK,
                bbox={"fc": SURFACE, "ec": "none", "alpha": 0.9}, arrowprops={"arrowstyle": "-", "color": INK})

    lat = np.linspace(-87.1875, 87.1875, 32)
    ax2.plot(lat, np.cos(np.deg2rad(lat)), color=BLUE, lw=2, marker="o", ms=4)
    ax2.set_title("Relative cell area", loc="left")
    ax2.set_xlabel("latitude of cell center")
    ax2.set_ylim(0, 1.05)
    ax2.set_xticks([-90, -45, 0, 45, 90])
    ax2.grid(color=GRID, lw=0.5)
    ax2.annotate("polar cells are ~20×\nsmaller than equatorial", (87.2, 0.05), (0, 0.2), color=INK, ha="center",
                 arrowprops={"arrowstyle": "-", "color": INK_2})
    save(fig, out, "grid")


def fig_snapshot(ds, static, out):
    snap = ds.sel(time=SNAPSHOT)
    t2m = grid(snap["2m_temperature"]) - 273.15
    z500 = grid(snap["geopotential"].sel(level=500)) / 9.80665  # geopotential -> height in metres
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
    im = axes[0].imshow(t2m, origin="lower", extent=EXTENT, cmap=DIVERGING, aspect="auto",
                        norm=TwoSlopeNorm(vcenter=0, vmin=-45, vmax=40))
    map_axes(axes[0], static.land_sea_mask, "2m_temperature (°C), 1 Jan 2020 00 UTC")
    fig.colorbar(im, ax=axes[0], shrink=0.85, label="°C (stored in kelvin)")
    im = axes[1].imshow(z500, origin="lower", extent=EXTENT, cmap=SEQUENTIAL, aspect="auto")
    axes[1].contour(np.arange(64) * 5.625, snap.latitude, z500, levels=np.arange(4800, 6000, 100),
                    colors="white", linewidths=0.5)
    map_axes(axes[1], static.land_sea_mask, "geopotential @ 500 hPa, as height (m)")
    fig.colorbar(im, ax=axes[1], shrink=0.85, label="m (stored as m²/s²; ÷ 9.81)")
    save(fig, out, "snapshot")


def fig_raw_numbers(ds, static, out):
    t2m = grid(ds["2m_temperature"].sel(time=SNAPSHOT))
    rows, cols = slice(22, 28), slice(0, 6)  # 36°N–64°N, 0°E–28°E: western/central Europe
    patch = t2m[rows, cols]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
    ax.imshow(t2m - 273.15, origin="lower", extent=EXTENT, cmap=DIVERGING, aspect="auto",
              norm=TwoSlopeNorm(vcenter=0, vmin=-45, vmax=40))
    map_axes(ax, static.land_sea_mask, "Where the 6 × 6 patch comes from")
    ax.add_patch(Rectangle((-2.8125, -90 + 22 * 5.625), 6 * 5.625, 6 * 5.625, fill=False, ec=INK, lw=2))

    ax2.imshow(patch - 273.15, origin="lower", cmap=DIVERGING, norm=TwoSlopeNorm(vcenter=0, vmin=-45, vmax=40))
    for (r, c), value in np.ndenumerate(patch):
        ax2.text(c, r, f"{value:.1f}", ha="center", va="center", fontsize=8, color=INK)
    ax2.set_title("The actual numbers (kelvin)", loc="left")
    lats = ds.latitude.values[rows]
    lons = ds.longitude.values[cols]
    ax2.set_yticks(range(6), [f"row {r}\n{lat:.1f}°" for r, lat in zip(range(22, 28), lats)], fontsize=8)
    ax2.set_xticks(range(6), [f"col {c}\n{lon:.1f}°" for c, lon in zip(range(6), lons)], fontsize=8)
    for spine in ax2.spines.values():
        spine.set_visible(False)
    save(fig, out, "raw_numbers")


def fig_tensor(ds, out):
    """Draw a few (lat, lon) frames per channel stacked along time, like the (T, C, Lat, Lon) array."""
    frames = ds.sel(time=slice("2020-01-01T00", "2020-01-01T18"))
    fig = plt.figure(figsize=(12, 4.6))
    channels = [("2m_temperature", None, DIVERGING, "channel 0: 2m_temperature"),
                ("geopotential", 500, SEQUENTIAL, "channel 1: geopotential@500")]
    for c, (var, level, cmap, label) in enumerate(channels):
        da = frames[var] if level is None else frames[var].sel(level=level)
        values = da.transpose("time", "latitude", "longitude").values
        for t in reversed(range(4)):
            left, bottom = 0.04 + c * 0.5 + t * 0.045, 0.16 + t * 0.09
            ax = fig.add_axes([left, bottom, 0.3, 0.42])
            ax.imshow(values[t], origin="lower", cmap=cmap, aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(INK)
            if t == 0:
                ax.set_xlabel("longitude → (64 columns)", fontsize=9)
                ax.set_ylabel("latitude ↑ (32 rows)", fontsize=9)
            if t == 3:
                ax.set_title(f"{label}\n(time steps 00, 06, 12, 18 UTC stacked behind)", fontsize=10, loc="left")
    fig.text(0.04, 0.95, "One training array: shape (time, channel, latitude, longitude) = (T, 2, 32, 64)",
             fontsize=12, weight="bold")
    save(fig, out, "tensor")


def fig_diurnal(ds, static, out):
    day = ds["2m_temperature"].sel(time=slice("2020-06-21T00", "2020-06-21T18"))
    values = day.transpose("time", "latitude", "longitude").values
    anomaly = values - values.mean(axis=0)
    fig, axes = plt.subplots(1, 4, figsize=(15, 2.8), sharey=True)
    for ax, frame, hour in zip(axes, anomaly, ["00", "06", "12", "18"]):
        ax.imshow(frame, origin="lower", extent=EXTENT, cmap=DIVERGING, aspect="auto",
                  norm=TwoSlopeNorm(vcenter=0, vmin=-8, vmax=8))
        map_axes(ax, static.land_sea_mask, f"{hour} UTC")
        ax.set_xticks([0, 180], ["0°E", "180°"])
    fig.suptitle("21 June 2020: 2m_temperature minus that day's mean (K). The warm patch follows the sun westward "
                 "over land.", x=0.07, y=1.06, ha="left", fontsize=11, weight="bold")
    save(fig, out, "diurnal")


def fig_timeseries(ds, out):
    point = ds.sel(time=slice("2020-01-01", "2020-12-31")).sel(latitude=50.625, longitude=8.4375, method="nearest")
    time = point.time.values
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    ax1.plot(time, point["2m_temperature"].values - 273.15, color=BLUE, lw=0.8)
    ax1.set_title("2m_temperature (°C) at one grid cell (50.6°N, 8.4°E) through 2020, every 6 hours", loc="left")
    ax2.plot(time, point["geopotential"].sel(level=500).values / 9.80665, color=BLUE, lw=0.8)
    ax2.set_title("geopotential @ 500 hPa as height (m), same cell", loc="left")
    for ax in (ax1, ax2):
        ax.grid(color=GRID, lw=0.5)
    ax1.text(0.01, 0.92, "seasons (slow) + weather systems (days) + day/night (the fast zig-zag)",
             transform=ax1.transAxes, color=INK_2)
    ax2.text(0.01, 0.92, "no day/night cycle up here; the dips are troughs passing overhead",
             transform=ax2.transAxes, color=INK_2)
    fig.tight_layout()
    save(fig, out, "timeseries")


def fig_training_pair(ds, static, out):
    z = ds["geopotential"].sel(level=500, time=slice("2020-01-01T00", "2020-01-01T06"))
    x, y = (grid(z.isel(time=i)) / 9.80665 for i in range(2))
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.2))
    for ax, field, title in [(axes[0], x, "input x_t (00 UTC)"), (axes[1], y, "target x_t+1 (06 UTC)")]:
        ax.imshow(field, origin="lower", extent=EXTENT, cmap=SEQUENTIAL, aspect="auto", vmin=4900, vmax=5950)
        map_axes(ax, static.land_sea_mask, f"Z500 height (m): {title}")
    im = axes[2].imshow(y - x, origin="lower", extent=EXTENT, cmap=DIVERGING, aspect="auto",
                        norm=TwoSlopeNorm(vcenter=0, vmin=-80, vmax=80))
    map_axes(axes[2], static.land_sea_mask, "x_t+1 − x_t: what a residual model predicts")
    fig.colorbar(im, ax=axes[2], shrink=0.85, label="m in 6 h")
    save(fig, out, "training_pair")


def fig_forecast_layout(out):
    fig, ax = plt.subplots(figsize=(11, 4.2))
    inits, leads = 6, 9
    for i in range(inits):
        for k in range(leads):
            color = BLUE if k == 0 else "#cde2fb"
            ax.add_patch(Rectangle((k, inits - 1 - i), 0.92, 0.85, color=color))
            hours = i * 12 + k * 6  # valid time, in hours after Jan 1 00z
            ax.text(k + 0.46, inits - 1 - i + 0.42, f"Jan {1 + hours // 24} {hours % 24:02d}z",
                    ha="center", va="center", fontsize=7, color="white" if k == 0 else INK)
    # Cells with the same valid time lie on a diagonal.
    ax.plot([2.46, 0.46], [inits - 1 + 0.42, inits - 2 + 0.42], color=ORANGE, lw=2)
    ax.plot([4.46, 2.46, 0.46], [inits - 1 + 0.42, inits - 2 + 0.42, inits - 3 + 0.42], color=ORANGE, lw=2)
    ax.set_xlim(0, leads)
    ax.set_ylim(0, inits)
    ax.set_xticks(np.arange(leads) + 0.46, [f"{k * 6}h" for k in range(leads)])
    ax.set_yticks(np.arange(inits) + 0.42, [f"Jan {1 + i // 2} {i % 2 * 12:02d}z" for i in reversed(range(inits))])
    ax.set_xlabel("prediction_timedelta (lead time) →")
    ax.set_ylabel("time (initialization)")
    ax.set_title("Forecast array (time, prediction_timedelta, lat, lon). Each cell is a full 32 × 64 map; "
                 "its label is the valid time (init + lead).", loc="left")
    for spine in ax.spines.values():
        spine.set_visible(False)
    save(fig, out, "forecast_layout")


def fig_levels(out):
    """Zonal-mean temperature on all 13 pressure levels (streams one time step from the WB2 bucket)."""
    t = open_era5(ERA5_64X32)["temperature"].sel(time=SNAPSHOT).mean("longitude").load() - 273.15
    fig, ax = plt.subplots(figsize=(9, 4.5))
    im = ax.pcolormesh(t.latitude, t.level, t.transpose("level", "latitude"), cmap=DIVERGING, shading="nearest",
                       norm=TwoSlopeNorm(vcenter=0, vmin=-80, vmax=30))
    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.set_yticks(t.level.values, [str(v) for v in t.level.values], fontsize=8)
    ax.minorticks_off()
    ax.set_ylabel("pressure level (hPa); ~sea level at bottom, ~20 km at top")
    ax.set_xlabel("latitude")
    ax.set_title("temperature averaged around each latitude circle (°C), 1 Jan 2020: the 13 levels", loc="left")
    ax.annotate("500 hPa ≈ 5.5 km,\nwhere Z500 lives", (-60, 500), (-85, 330), color=INK,
                arrowprops={"arrowstyle": "-", "color": INK_2})
    fig.colorbar(im, ax=ax, label="°C")
    save(fig, out, "levels")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/era5_64x32")
    p.add_argument("--out", default="docs/images/data")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ds = open_era5(args.data)
    static = open_era5(args.data + "/static.zarr")
    fig_grid(static, out)
    fig_snapshot(ds, static, out)
    fig_raw_numbers(ds, static, out)
    fig_tensor(ds, out)
    fig_diurnal(ds, static, out)
    fig_timeseries(ds, out)
    fig_training_pair(ds, static, out)
    fig_forecast_layout(out)
    fig_levels(out)


if __name__ == "__main__":
    main()
