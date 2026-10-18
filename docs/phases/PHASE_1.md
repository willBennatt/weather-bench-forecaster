# Phase 1

## Goal

My first [notebook](../../notebooks/01_t2m_z500_model_comparison.ipynb) trained three models on one year (2018) of T2M + Z500 and evaluated them on 48 h predictions over 1–15 Jan 2020.

The goal of Phase 1 is to extend these initial models to the full training set of WeatherBench2 and compare them against real baseline models instead of just themselves.

Steps:

1. Train 1979–2015, validate 2016–2017, test 2020 (with 00z and 12z initializations, so 732 forecasts). Keep it to only the `2m_temperature` and `geopotential@500` for now for quick local training.
2. Extend prediction rollouts to 10 days instead of only 2 days.
3. Look at ACC (anomaly correlation coefficient) for Z500 in addition to RMSE.
4. Plot the best model vs persistence vs climatology over a full year.
5. Explain why climatology's RMSE is flat.

## Step 0: Setup local env

Only using 2m temp and z500 means the data is small enough to train locally on my laptop, which is simpler than running on Google Colab.

```sh
# Create the virtual env
uv venv --python 3.12 .venv
source .venv/bin/activate
# Install dependencies with uv
uv pip install -e . -e ../weatherbench2 absl-py
```

## Step 1: Train Full History

Download the data and cache it in [data/era5_64x32](../../data/era5_64x32/) by running:

```sh
python scripts/download_era5.py --channels 2m_temperature geopotential@500 --start-year 1979 --end-year 2020 --output data/era5_64x32
```

Train the simple model on the full data with 48 hour windows:

```sh
python scripts/train_and_forecast.py --model simple --channels 2m_temperature geopotential@500 --epochs 3 --train-start 1979-01-01 --train-end 2015-12-31 --test-start 2020-01-01 --test-end 2020-12-31 --lead-steps 40 --output output/forecast_simple_2mtemp_z500.zarr --checkpoint checkpoints/simple_2mtemp_z500.pt --data data/era5_64x32
```

Evaluate:

```sh
WB2_DIR='../weatherbench2' BASELINES=1 scripts/evaluate.sh output/forecast_simple_2mtemp_z500.zarr evals/results_simple_2mtemp_z500/ 2m_temperature,geopotential 500
```
