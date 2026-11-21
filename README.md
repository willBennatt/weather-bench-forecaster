# WeatherBench Forecaster

A learning project in **machine-learning weather prediction**. It trains small neural networks to forecast the global atmosphere from [ERA5](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5) reanalysis data and scores them with the [WeatherBench 2](https://sites.research.google/gr/weatherbench/) benchmark ([code](https://github.com/google-research/weatherbench2)). That's the same protocol used to evaluate GraphCast, Pangu-Weather and ECMWF's physics-based IFS.

Everything runs on free GPUs (Google Colab T4 or Kaggle). The goal is to understand the techniques, not to beat state-of-the-art models.

## Learning Strategy

The goal of this project is to refresh my machine-learning skills while learning about weather data and analysis. The goal is not to become an expert in weather forecasting, but rather to quickly gain an overall understanding of the field and types of work being done. I am trying out an AI-as-teacher strategy to see whether I can learn faster with guidance. For more information, see [Learning Strategy](docs/LEARNING.md).

## How it works

New to the data? [`docs/DATA.md`](docs/DATA.md) shows the grid, variables, Zarr layout and the actual arrays with pictures.

1. **Data.** WeatherBench 2 hosts ERA5 regridded to a coarse **64 × 32 global grid (5.625°, ~600 km cells) every 6 hours**, from 1959 to 2023, in a public Zarr store on Google Cloud Storage. We stream a subset of variables and stack them into a `(time, channel, lat, lon)` array, e.g. 2 m temperature and 500 hPa geopotential.
2. **Normalization.** Each channel is z-scored with means and standard deviations computed on the training years only.
3. **Model.** A convolutional network maps the state at time _t_ to the state at _t_ + 6 h. The convolutions use **periodic padding in longitude**, because the globe wraps around at 0°/360°. The CNNs are _residual_: they predict the change, not the full field.
4. **Training.** Minimize the MSE between the predicted and actual next state over consecutive 6-hour pairs.
5. **Autoregressive rollout.** To forecast 48 h ahead, apply the model 8 times, feeding each prediction back in as the next input. This is how GraphCast and friends work too.
6. **Evaluation.** Forecasts are exported in WB2's `(time, prediction_timedelta, [level], latitude, longitude)` Zarr format and scored against ERA5 with the official `weatherbench2/scripts/evaluate.py`. It reports area-weighted RMSE, bias and ACC, plus persistence and climatology baselines.

## Results so far

For current progress, see [docs/phases/](docs/phases/).

## Roadmap

[`docs/ROADMAP.md`](docs/ROADMAP.md) is a phased plan from this toy to a ~40-variable global model with ensembles, all on free compute:

1. Honest evaluation: baselines, standard splits, 10-day rollouts
2. Forcings (solar radiation, time, orography) and a latitude-weighted loss
3. A full atmospheric state: z, t, u, v, q on 7 levels plus surface fields
4. Architectures: U-Net, Vision Transformer, Spherical Fourier Neural Operator
5. Multi-step rollout training, plus diagnosing forecast blurring
6. Probabilistic ensembles: CRPS and the spread/skill ratio
7. Case studies (2021 Pacific NW heat dome) and a write-up
8. Live forecasts on a website ([`docs/WEBSITE.md`](docs/WEBSITE.md))

## Repository layout

```
├── notebooks/
│   └── 01_t2m_z500_model_comparison.ipynb   # v0: three models, WB2 evaluation, skill curves (Colab)
├── src/wbforecaster/                        # reusable code, importable in Colab or locally
│   ├── data.py       # open WB2 ERA5, extract channels ("geopotential@500"), normalizer, dataset
│   ├── models.py     # PeriodicConv2d, Linear / Simple CNN / Complex CNN
│   └── forecast.py   # training loop, autoregressive rollout, export to WB2 Zarr format
├── scripts/
│   ├── download_era5.py        # cache a subset of ERA5 locally (one Zarr store per year)
│   ├── train_and_forecast.py   # CLI: train → rollout → write forecast .zarr
│   ├── evaluate.sh             # run the official WB2 evaluation (optionally with baselines)
│   ├── make_persistence.py     # build a persistence baseline forecast (used by evaluate.sh)
│   └── make_data_figures.py    # regenerate the figures in docs/DATA.md
└── docs/
    ├── DATA.md       # the WB2 data and what the arrays look like (with figures)
    ├── ROADMAP.md    # learning plan
    ├── WEBSITE.md    # how to serve forecasts on a website
    └── images/
```

## Running it

### In Google Colab (recommended, free T4 GPU)

Open the notebook straight from GitHub: **File → Open notebook → GitHub**, tick _Include private repos_, and choose this repo. Set **Runtime → Change runtime type → T4 GPU**. Use **File → Save a copy in GitHub** to commit changes back.

To use the `wbforecaster` package from Colab, store a read-only GitHub token as a Colab Secret named `GH_TOKEN`, then run:

```python
from google.colab import userdata
token = userdata.get("GH_TOKEN")
!git clone -q https://{token}@github.com/willBennatt/weather-bench-forecaster.git
!pip install -q -e weather-bench-forecaster
!git clone -q https://github.com/google-research/weatherbench2.git && pip install -q -e weatherbench2
```

### On a laptop (fine for 2-variable models; Apple-silicon GPU is used automatically)

Use Python 3.11 or 3.12 (`apache_beam`, needed by the WB2 evaluation, lags behind new Python versions):

```bash
python3.12 -m venv .venv && source .venv/bin/activate
```

```bash
pip install -e .
```

```bash
git clone https://github.com/google-research/weatherbench2.git && pip install -e weatherbench2 absl-py
```

Download the data once instead of streaming it on every run. 1979–2020 for two channels takes ~15 minutes and ~0.8 GB of disk; re-running resumes:

```bash
python scripts/download_era5.py --channels 2m_temperature geopotential@500 --start-year 1979 --end-year 2020 --output data/era5_64x32
```

### Training and forecasting (CLI)

Defaults stream from the WB2 bucket; add `--data data/era5_64x32` to use the local cache:

```bash
python scripts/train_and_forecast.py --model simple --output forecast_simple.zarr
```

Pick variables with `--channels` (pressure levels use `var@level`):

```bash
python scripts/train_and_forecast.py --model complex --channels 2m_temperature geopotential@500 temperature@850 --epochs 10 --output forecast_complex.zarr --checkpoint complex.pt
```

For the standard WB2 setup, add a validation period and 10-day forecasts from 00z/12z only. The learning rate decays over training, the checkpoint keeps the epoch with the best validation loss, and `--patience` stops early:

```bash
python scripts/train_and_forecast.py --data data/era5_64x32 --model simple --train-start 1979-01-01 --train-end 2015-12-31 --val-start 2016-01-01 --val-end 2017-12-31 --test-start 2020-01-01 --test-end 2020-12-31 --lead-steps 40 --init-hours 0 12 --epochs 5 --patience 2 --output output/forecast_simple.zarr --checkpoint checkpoints/simple.pt
```

Evaluate with the official WeatherBench 2 script. It needs a checkout of `weatherbench2` installed with `pip install -e`, plus `absl-py`. `BASELINES=1` also scores persistence and climatology for the same start times:

```bash
BASELINES=1 scripts/evaluate.sh forecast_simple.zarr eval_results_simple/ 2m_temperature,geopotential 500
```

Results are written to `eval_results_simple/deterministic.nc`, with the baselines in `eval_results_simple/persistence/` and `eval_results_simple/climatology/`. The persistence forecast is also saved next to your forecast (`forecast_simple_persistence.zarr`, same size as the forecast).

## Data and acknowledgements

- ERA5 reanalysis: Hersbach et al. 2020, Copernicus Climate Change Service.
- WeatherBench 2: Rasp et al. 2024, _JAMES_. Data at `gs://weatherbench2` (public, no credentials needed).
