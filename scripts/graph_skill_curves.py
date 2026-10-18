"""Plot skill curves (RMSE and ACC vs lead time) from WeatherBench 2 evaluation results.

Each argument is an evaluation results directory written by scripts/evaluate.sh, or a
deterministic.nc file. Name a curve with NAME=PATH; otherwise the directory name is used.
Baselines in a results directory's persistence/ and climatology/ subfolders are added
automatically (once, even if several model directories have them).

Examples:
    python scripts/graph_skill_curves.py evals/results_simple_2mtemp_z500
    python scripts/graph_skill_curves.py simple=evals/results_simple complex=evals/results_complex \\
        --reference hres keisler --output evals/skill_curves.png
"""
import argparse
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# Model curves take these hues in order; baselines are dashed/dotted gray; WB2 references are thin.
MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BASELINE_STYLE = {
    "persistence": {"color": "#52514e", "linestyle": "--"},
    "climatology": {"color": "#8f8e88", "linestyle": ":"},
}
# Published models are context, not the focus: thin and light, told apart by dash pattern.
REFERENCE_STYLES = [{"color": "#9e9d97", "linestyle": "-"}, {"color": "#9e9d97", "linestyle": "-."},
                    {"color": "#c4c3bd", "linestyle": "-"}, {"color": "#c4c3bd", "linestyle": "-."}]
REFERENCE_URL = "https://storage.googleapis.com/weatherbench2/benchmark_results/{name}_vs_era5_64x32_2020.nc"

UNITS = {"2m_temperature": "K", "geopotential": "m²/s²", "temperature": "K", "mean_sea_level_pressure": "Pa",
         "specific_humidity": "kg/kg", "10m_wind_speed": "m/s"}


def parse_inputs(paths):
    """Return {name: Path to deterministic.nc} for models, then baselines found alongside them."""
    models, baselines = {}, {}
    for arg in paths:
        name, _, path = arg.rpartition("=")
        path = Path(path)
        directory = path if path.is_dir() else path.parent
        models[name or directory.name] = directory / "deterministic.nc" if path.is_dir() else path
        for baseline in BASELINE_STYLE:
            candidate = directory / baseline / "deterministic.nc"
            if candidate.exists():
                baselines.setdefault(baseline, candidate)
    return models, baselines


def load_ours(path):
    """WB2 evaluate.py output: dims (metric, lead_time, [level]), lead_time as timedelta."""
    ds = xr.open_dataset(path)
    if "region" in ds.dims:
        ds = ds.sel(region="global")
    return ds.assign_coords(lead_time=ds.lead_time.values / np.timedelta64(1, "D"))


def load_reference(name, cache_dir):
    """Published WB2 scores for 2020 at 64x32: variables like 'mse.geopotential', lead_time in hours."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{name}_vs_era5_64x32_2020.nc"
    if not path.exists():
        urllib.request.urlretrieve(REFERENCE_URL.format(name=name), path)
    ds = xr.open_dataset(path, engine="scipy").sel(region="global")
    return ds.assign_coords(lead_time=ds.lead_time.values / 24)


def curve(ds, var, level, metric, reference=False):
    """One metric for one variable, or None if the dataset doesn't have it."""
    key = f"{'mse' if metric == 'rmse' else metric}.{var}" if reference else var
    if key not in ds:
        return None
    da = ds[key] if reference else ds[key].sel(metric="mse" if metric == "rmse" else metric)
    if level is not None:
        if "level" not in da.dims or level not in da.level:
            return None
        da = da.sel(level=level)
    # WB2's RMSE is the square root of the time-averaged MSE, so this matches published numbers.
    return np.sqrt(da) if metric == "rmse" else da


def variables(ds):
    """(var, level) pairs evaluated in a results file, e.g. [('2m_temperature', None), ('geopotential', 500)]."""
    out = []
    for var in ds.data_vars:
        if "level" in ds[var].dims:
            out += [(var, int(level)) for level in ds.level.values]
        else:
            out.append((var, None))
    return out


def plot_panel(ax, series, var, level, metric):
    """Draw one panel; clip the y-axis to the baselines so unstable models don't flatten everything."""
    top = None
    if metric == "rmse":
        baseline_max = [float(np.nanmax(da)) for kind, _, da in series if kind == "baseline" and da is not None]
        top = 1.6 * max(baseline_max) if baseline_max else None

    references = 0
    for kind, name, da in series:
        if da is None:
            continue
        x, y = da.lead_time.values, da.values
        if kind == "baseline":
            style = {**BASELINE_STYLE[name], "lw": 1.5}
        elif kind == "reference":
            style = {**REFERENCE_STYLES[references % len(REFERENCE_STYLES)], "lw": 1.2}
            references += 1
        else:
            style = {"color": name[1], "lw": 2}
            name = name[0]
        ax.plot(x, y, label=name, **style)

        off = np.flatnonzero(y > top) if top else []
        if len(off):
            day = x[off[0]]
            ax.plot(day, top, marker="^", ms=8, color=style["color"], clip_on=False)
            ax.text(day + 0.15, top * 0.97, f"{name} off scale after day {day:g}", fontsize=8, color="#52514e",
                    va="top")

    label = var if level is None else f"{var} @ {level} hPa"
    if metric == "rmse":
        ax.set_ylim(0, top)
        ax.set_ylabel(f"RMSE ({UNITS.get(var, 'units')})")
        ax.set_title(f"{label}: RMSE (lower is better)", loc="left", fontsize=10, fontweight="bold")
    else:
        ax.set_ylim(-0.1, 1.02)
        ax.axhline(0.6, color="#52514e", lw=0.8, linestyle=(0, (1, 3)))
        ax.set_ylabel("ACC (dotted line: 0.6 = useful skill)")
        ax.set_title(f"{label}: anomaly correlation (higher is better)", loc="left", fontsize=10, fontweight="bold")
    ax.set_xlabel("lead time (days)")
    ax.set_xticks(range(0, 11))
    ax.set_xlim(0, 10)
    ax.grid(color="#e4e3df", lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("evals", nargs="+", help="results directories or deterministic.nc files, optionally NAME=PATH")
    p.add_argument("--reference", nargs="*", default=[],
                   help="published WB2 models to overlay, e.g. hres keisler graphcast pangu")
    p.add_argument("--metrics", nargs="+", default=["rmse", "acc"], choices=["rmse", "acc", "bias", "mae"])
    p.add_argument("--output", default="evals/skill_curves.png", help="where to save the figure")
    p.add_argument("--show", action="store_true", help="also open an interactive window")
    args = p.parse_args()

    models, baselines = parse_inputs(args.evals)
    model_ds = {name: load_ours(path) for name, path in models.items()}
    baseline_ds = {name: load_ours(path) for name, path in baselines.items()}
    cache = Path(args.output).parent / "wb2_reference"
    reference_ds = {name: load_reference(name, cache) for name in args.reference}

    rows = variables(next(iter(model_ds.values())))
    fig, axes = plt.subplots(len(rows), len(args.metrics), figsize=(6.5 * len(args.metrics), 3.8 * len(rows)),
                             squeeze=False)
    for r, (var, level) in enumerate(rows):
        for c, metric in enumerate(args.metrics):
            series = [("model", (name, MODEL_COLORS[i % len(MODEL_COLORS)]), curve(ds, var, level, metric))
                      for i, (name, ds) in enumerate(model_ds.items())]
            series += [("baseline", name, curve(ds, var, level, metric)) for name, ds in baseline_ds.items()]
            series += [("reference", name, curve(ds, var, level, metric, reference=True))
                       for name, ds in reference_ds.items()]
            plot_panel(axes[r, c], series, var, level, metric)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Wrote {args.output}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
