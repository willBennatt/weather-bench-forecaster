# Phase 3

## Goal

Phase 2 add forcing parameters to improve how the model predicted temperature and geopotential by giving it knowledge of solar radiation and time of day. However, the model was still only predicting T2M and Z500.

The goal of Phase 3 is to expand the predicted channels to all 39 available on WeatherBench. The key challenges will be dealing with the extra memory required and the normalization and loss calculation for each channel.

The goal is accomplished once I have trained, evaluated, and generated plots for a 39-channel model.

## Step 1: Data on Disk

The Phase 3 channel set (35 pressure-level + 4 surface channels) for 1979–2015 is ~17 GB as float32. That's more than this laptop's 8 GB of RAM, and more than a free Colab session's ~12 GB. The Phase 1–2 pipeline also holds about three copies while it normalizes (raw, normalized, tensor). Compute isn't the problem: at 64x32, only the model's first layer grows with more input channels.

So training now reads from disk instead:

- [build_memmap.py](../../scripts/build_memmap.py) works one year at a time. It computes each channel's mean and std over the training period, then writes the normalized train and val splits as float16 `.npy` files (~8.6 GB for 1979–2015). float16 is fine for z-scores (~3 significant digits) but not for raw values like geopotential (~55,000), so the data is normalized before it's stored.
- `--memmap` in [train_and_forecast.py](../../scripts/train_and_forecast.py) opens those files as memory maps. Each batch reads only its own samples from the SSD, and the operating system caches whatever fits in RAM. The cache sets the channels, the normalization and the train/val periods.
- The test year (2020, ~0.5 GB) and the forcings still come from `--data`, as before.

Checked on the Phase 2 channels (train 2013–2015): the stored data matches the in-memory pipeline to 0.0002 RMSE (in std units), and two epochs gave the same validation loss (0.00206 vs 0.00208) at the same speed, with lower peak memory.

Download the channels (1979–2017 for train/val, plus 2020 for test). Pressure-level chunks hold all 13 levels, so this transfers more than it stores:

```sh
python scripts/download_era5.py --output data/era5_64x32_all_channels --start-year 1979 --end-year 2020 --channels geopotential@50 geopotential@250 geopotential@500 geopotential@700 geopotential@850 geopotential@925 geopotential@1000 temperature@50 temperature@250 temperature@500 temperature@700 temperature@850 temperature@925 temperature@1000 u_component_of_wind@50 u_component_of_wind@250 u_component_of_wind@500 u_component_of_wind@700 u_component_of_wind@850 u_component_of_wind@925 u_component_of_wind@1000 v_component_of_wind@50 v_component_of_wind@250 v_component_of_wind@500 v_component_of_wind@700 v_component_of_wind@850 v_component_of_wind@925 v_component_of_wind@1000 specific_humidity@50 specific_humidity@250 specific_humidity@500 specific_humidity@700 specific_humidity@850 specific_humidity@925 specific_humidity@1000 2m_temperature 10m_u_component_of_wind 10m_v_component_of_wind mean_sea_level_pressure
```

Then build the memmap cache with the same `--channels` list:

```sh
python scripts/build_memmap.py --data data/era5_64x32_all_channels --output data/memmap/era5_64x32_all_channels --train-start 1979-01-01 --train-end 2015-12-31 --val-start 2016-01-01 --val-end 2017-12-31 --channels <same list>
```

Both scripts skip work that's already done, so they can be resumed after an interruption. To change the channels or the training period, build into a new `--output`, since the normalization statistics depend on both.

Train with `--data data/era5_64x32_all_channels --memmap data/memmap/era5_64x32_all_channels` plus the usual test, rollout and forcing flags.

## Step 2: Train and Evaluate

Once the memmaps are created, train the model:

```sh
python scripts/train_and_forecast.py --data data/era5_64x32_all_channels --memmap data/memmap/era5_64x32_all_channels --model simple --train-start 1979-01-01 --train-end 2015-12-31 --val-start 2016-01-01 --val-end 2017-12-31 --test-start 2020-01-01 --test-end 2020-12-31 --lead-steps 40 --init-hours 0 12 --epochs 5 --patience 2 --lat-weighted-loss --output output/phase3/forecast_simple_all_channels.zarr --checkpoint checkpoints/phase3/simple_all_channels.pt
```

Then cache the new channels for evaluation:

```sh
python scripts/cache_eval_data.py --start 2020-01-01 --end 2020-12-31 --variables geopotential temperature u_component_of_wind v_component_of_wind specific_humidity 2m_temperature 10m_u_component_of_wind 10m_v_component_of_wind mean_sea_level_pressure --levels 50,250,500,700,850,925,1000
```

Then evaluate. Include baseline to get persistence and climatology models for the new channels:

```sh
WB2_DIR='../weatherbench2' BASELINES=1 scripts/evaluate.sh output/phase3/forecast_simple_all_channels.zarr evals/phase3/simple_all_channels/ geopotential,temperature,u_component_of_wind,v_component_of_wind,specific_humidity,2m_temperature,10m_u_component_of_wind,10m_v_component_of_wind,mean_sea_level_pressure 50,250,500,700,850,925,1000
```

Finally, plot the new model versus persistence and climatology baselines for all channels:

```sh
python scripts/graph_skill_curves.py simple=evals/phase3/simple_all_channels --output docs/images/phase3_skill_curves.png
```
