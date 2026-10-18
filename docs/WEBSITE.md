# Putting forecasts on your website

**Short answer: yes, for $0.** At 64×32 a model is a few MB, and a 10-day rollout takes seconds on a CPU. The hard part isn't compute. It's getting a *current* initial condition, because ERA5 (the training data) is published days after the fact. The solution is to start the model from ECMWF's free real-time analysis.

There are two complementary options. Build B first: it's simpler and needs no live data.

---

## Option A: live forecasts twice a day (GitHub Actions + static files)

```
 ECMWF open data (00z/12z, step 0, 0.25°)            GitHub Actions cron (free CPU)
 ─────────────────────────────────────────┐    ┌─────────────────────────────────────────┐
  z, t, u, v, q on 13 levels;              │──▶ │ 1. download only the needed fields      │
  2t, 10u, 10v, msl                        │    │ 2. regrid 1440×721 → 64×32 (conservative)│
                                           │    │ 3. add forcings (solar, time, statics)   │
                                           │    │ 4. normalize → model rollout (40 steps)  │
                                           │    │ 5. write small JSON/PNG files            │
                                           │    └──────────────────┬──────────────────────┘
                                           │                       ▼
                                           │      gh-pages branch / Cloudflare R2 (free)
                                           │                       ▼
                                           └────────────▶  your website fetches + renders
```

### Initial conditions: ECMWF open data

- `https://data.ecmwf.int/forecasts/YYYYMMDD/{00,12}z/ifs/0p25/oper/…-0h-oper-fc.grib2` (CC BY 4.0, free, no account needed).
- Step 0 contains everything the Phase 3 channel set needs. As of Oct 2026:
  - `z` (geopotential, already in m²/s², the same as ERA5), `t`, `u`, `v`, `q` on 1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50 (and 10) hPa. These are exactly the WB2 pressure levels.
  - Surface fields: `2t`, `10u`, `10v`, `msl`, `sp`, `tcwv`, `lsm`, `z` (orography).
- The full step-0 file is ~134 MB. Each `.grib2` has a JSON `.index` file with byte offsets, so you can download only your ~40 fields. The [`ecmwf-opendata`](https://github.com/ecmwf/ecmwf-opendata) Python client does this for you:

  ```python
  from ecmwf.opendata import Client
  Client(source="ecmwf").retrieve(type="fc", step=0, levtype="pl",
      param=["z", "t", "u", "v", "q"], levelist=[50, 250, 500, 700, 850, 925, 1000], target="pl.grib2")
  ```
- Read the GRIB files with `xarray` + `cfgrib`.
- **Flip latitude.** GRIB runs north→south, WB2 runs south→north.
- **Regrid conservatively** with WB2's own `weatherbench2.regridding.ConservativeRegridder`. That's how the 64×32 training data was made, so the inputs match what the model saw in training.
- Bonus: the same server publishes **AIFS**, ECMWF's own ML model. You can show yours side by side with it.

### Distribution shift (an analysis worth doing anyway)

The model trains on ERA5 but runs live from IFS analyses. They're close, and at 5.6° the difference is small. GraphCast's operational version fine-tunes on IFS HRES for exactly this reason. You can measure the shift offline: WB2 hosts IFS analyses at your resolution (`gs://weatherbench2/datasets/hres_t0/2016-2022-6h-64x32_equiangular_conservative.zarr`). Run your 2020 test forecasts from both ERA5 and HRES initial conditions and compare the scores. That's a good extra section for the write-up.

### Scheduling and hosting

- **GitHub Actions** `on: schedule` cron, e.g. `30 8,20 * * *` UTC. Open data appears a few hours after each run; the `ecmwf-opendata` client's `latest()` finds the newest available run.
- Each run takes ~5 minutes, so twice a day uses ~300 of the 2,000 free private-repo minutes a month. Public repos are unlimited.
- Keep the trained checkpoint (a few MB) and the normalization stats in the repo or a release asset.
- Store the output as a few small files on a `gh-pages` branch or a Cloudflare R2 bucket (free tier, no egress fees). Two ways to do it:
  - Quantized grids: `uint8`/`float16` for 64×32 × 41 lead times × ~6 display variables comes to roughly 1 MB of JSON/binary per run.
  - Pre-rendered PNG maps (matplotlib + cartopy).

### Front end

- **Map view:** draw the 64×32 grid to a `<canvas>` with a color scale and coastlines (`world-atlas` TopoJSON + `d3-geo`). Add a lead-time slider (0–240 h) and a variable picker: T2M, MSLP, Z500, T850, 10 m wind.
- **Point forecast / meteogram:** the user clicks a location, and you bilinearly interpolate T2M, MSLP and wind over the 10 days.
- **Verification panel:** this is what makes it science and not a widget. Each day, score the forecasts from 1–10 days ago against the newest analysis. Show a running RMSE chart: your model vs. persistence (and vs. AIFS, if you add it).
- **Label it clearly** as an educational model on a ~600 km grid, not for decisions. Attribute ECMWF (CC BY 4.0) and Copernicus ERA5.

---

## Option B: in-browser "hindcast explorer" (no server, no live data)

- Export the model to ONNX (`torch.onnx.export`) and run it in the browser with [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/). A 64×32 CNN/U-Net runs a 40-step rollout in about a second.
- Ship a small archive of historical initial conditions, e.g. 00z daily for 2020–2022 (~1,100 states × 39 channels × 2048 cells as float16 ≈ 175 MB). That's too much to load at once, so split it into one file per day and fetch on demand. You also need the matching ERA5 "truth" for scoring.
- The user picks a date, say "2021-06-24, Pacific NW heat dome", and watches the model forecast next to what actually happened, with live RMSE.
- This works on any static host (GitHub Pages, Netlify, your existing site). It also shows off ML deployment skills.

## Recommended order

1. **Option B with pre-rendered PNGs/GIFs for 3–4 case studies.** That's a weekend of work after Phase 3.
2. **Option A once you have a 39-channel model with forcings (end of Phase 3 or 5).** A 2-variable model can't run live: it needs exactly the state variables ECMWF provides.
3. Add the verification panel after Option A has run for ~2 weeks.

Build the live pipeline as `scripts/live_forecast.py` + `.github/workflows/forecast.yml` in this repo, and have the website read the outputs. Keep the website repo itself separate.
