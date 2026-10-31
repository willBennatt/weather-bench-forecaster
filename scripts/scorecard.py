"""Plot a scorecard: each variable's RMSE relative to a baseline, by lead time, as a heatmap.

Cells show how much lower (blue) or higher (red) the model's RMSE is than the baseline's at that
lead time: -40% means 40% less error than persistence. Colors saturate at half and double the
baseline's error, so a model that blows up shows as solid red rather than washing out the rest.

The argument is a results directory written by scripts/evaluate.sh with BASELINES=1 (the baseline
is read from its persistence/ or climatology/ subfolder), or use --baseline-dir for a baseline
evaluated elsewhere.

Examples:
    python scripts/scorecard.py evals/phase3/simple_all_channels --output docs/images/phase3_scorecard.png
    python scripts/scorecard.py evals/phase3/simple_all_channels --baseline climatology \\
        --variables geopotential@500 temperature@850 specific_humidity@700 2m_temperature
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgb

from wbforecaster.data import parse_channel

LEADS = [0.25, 0.5, 1, 2, 3, 5, 7, 10]  # days
# Diverging blue (better) <-> red (worse) with a neutral gray midpoint (no difference).
CMAP = LinearSegmentedColormap.from_list("better_worse", ["#2a78d6", "#f0efec", "#e34948"])
INK, INK_ON_DARK = "#1f1f1e", "#ffffff"


def load_rmse(path):
    """RMSE from a WB2 evaluate.py results file: the square root of the time-averaged MSE, as WB2 reports it."""
    ds = xr.open_dataset(path)
    if "region" in ds.dims:
        ds = ds.sel(region="global")
    return np.sqrt(ds.sel(metric="mse"))


def rows_in(ds):
    """(var, level) pairs in a results file, levels ascending (top of the atmosphere first)."""
    rows = []
    for var in ds.data_vars:
        if "level" in ds[var].dims:
            rows += [(var, int(level)) for level in sorted(ds.level.values)]
        else:
            rows.append((var, None))
    return rows


def label(var, level):
    return var if level is None else f"{var} @ {level} hPa"


def cell_text(ratio):
    """'-40%' for modest differences; '×3e5' once a model has blown up."""
    if not np.isfinite(ratio):
        return "n/a"
    if ratio >= 10:
        return f"×{ratio:.0e}".replace("e+0", "e").replace("e+", "e") if ratio >= 1000 else f"×{ratio:.0f}"
    return f"{(ratio - 1) * 100:+.0f}%".replace("-", "−")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", help="results directory from evaluate.sh (with BASELINES=1)")
    p.add_argument("--baseline", choices=["persistence", "climatology"], default="persistence")
    p.add_argument("--baseline-dir", help="baseline results directory, if not RESULTS/<baseline>/")
    p.add_argument("--variables", nargs="+", help="e.g. geopotential@500 2m_temperature (default: all evaluated)")
    p.add_argument("--leads", nargs="+", type=float, default=LEADS, help="lead times in days")
    p.add_argument("--title", help="figure title (default: the results directory name)")
    p.add_argument("--output", default="evals/scorecard.png")
    args = p.parse_args()

    results = Path(args.results)
    baseline_dir = Path(args.baseline_dir) if args.baseline_dir else results / args.baseline
    model, baseline = load_rmse(results / "deterministic.nc"), load_rmse(baseline_dir / "deterministic.nc")

    rows = rows_in(model)
    if args.variables:
        wanted = [parse_channel(v) for v in args.variables]
        missing = [v for v, pair in zip(args.variables, wanted) if pair not in rows]
        if missing:
            p.error(f"not in the results: {', '.join(missing)}")
        rows = wanted

    leads = [np.timedelta64(int(round(d * 24)), "h") for d in args.leads]
    ratio = np.full((len(rows), len(leads)), np.nan)
    for r, (var, level) in enumerate(rows):
        m, b = model[var], baseline[var]
        if level is not None:
            m, b = m.sel(level=level), b.sel(level=level)
        for c, lead in enumerate(leads):
            if lead in m.lead_time.values and lead in b.lead_time.values:
                ratio[r, c] = float(m.sel(lead_time=lead) / b.sel(lead_time=lead))

    # Color on log2(ratio): equal steps for "half the error" and "double the error".
    log_ratio = np.clip(np.log2(ratio), -1, 1)
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    fig, ax = plt.subplots(figsize=(1.0 * len(leads) + 3.5, 0.34 * len(rows) + 1.6))
    im = ax.imshow(log_ratio, cmap=CMAP, norm=norm, aspect="auto")
    for r in range(len(rows)):
        for c in range(len(leads)):
            rgb = to_rgb(CMAP(norm(log_ratio[r, c]))) if np.isfinite(log_ratio[r, c]) else (1, 1, 1)
            luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
            ax.text(c, r, cell_text(ratio[r, c]), ha="center", va="center", fontsize=8,
                    color=INK if luminance > 0.5 else INK_ON_DARK)

    # A thin surface-colored grid between cells, so neighbouring cells read as separate.
    ax.set_xticks(np.arange(len(leads) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(rows) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="#fcfcfb", linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.set_xticks(range(len(leads)), [f"{d * 24:g} h" if d < 1 else f"{d:g} d" for d in args.leads])
    ax.set_yticks(range(len(rows)), [label(*row) for row in rows], fontsize=9)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"{args.title or results.name}: RMSE relative to {args.baseline} (blue = better)",
                 loc="left", fontsize=10, fontweight="bold", pad=24)

    cbar = fig.colorbar(im, ax=ax, ticks=[-1, -0.5, 0, 0.5, 1], fraction=0.04, pad=0.02)
    cbar.ax.set_yticklabels(["≤ −50%", "−29%", "same", "+41%", "≥ +100%"], fontsize=8)
    cbar.outline.set_visible(False)

    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
