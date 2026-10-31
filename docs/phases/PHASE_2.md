# Phase 2

## Goal

Phase 1 trained the model on the full weather bench dataset from 1979 - 2015 and compared it to persistence and climatology models as benchmarks. It also extended the forecast window from 2 days to 10 days on the `2m_temperature` and `geopotential@500` outputs.

However, it was unable to beat persistence at 24 hours because it did not recognize how time of day plays a role in temperature. The goal of phase 2 is to add forcing inputs like incoming solar radiation so the model can learn to change its outputs with the time of day to achieve more accurate forecasts and beat persistence at 24 hours.

## Step 1: Separate Inputs From Outputs

1. Separated `in_channels` from `out_channels` in [models.py](../../src/wbforecaster/models.py) so they can be different.
2. Kept the residual connection on the state channels only: `x[:, :out_channels] + net(x)`. The model adds its predicted change to the weather it forecasts, not to the forcings, which it never predicts.
3. Modified the dataset to take a forcing tensor and append the forcings for each step to the state as extra input channels, so each sample is still an `(input, target)` pair and the training loop doesn't change.
4. The training script now builds models with `in_channels` and `out_channels` and saves both in the checkpoint. Until the forcings exist, both equal the number of state channels.

## Step 2: Forcing Functions

[forcings.py](../../src/wbforecaster/forcings.py) computes 9 forcing channels for the 6-hour step starting at each time:

| Channel                | What it tells the model                                                                                                        |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `toa_t`, `toa_t+6h`    | incoming solar radiation at the top of the atmosphere at the start and end of the step, divided by the solar constant (0–1.03) |
| `sin_hour`, `cos_hour` | UTC time of day                                                                                                                |
| `sin_year`, `cos_year` | time of year                                                                                                                   |
| `land_sea_mask`        | land (1) or sea (0)                                                                                                            |
| `orography`            | surface height (`geopotential_at_surface`), z-scored                                                                           |
| `sin_lat`              | latitude                                                                                                                       |

TOA is computed from solar geometry rather than read from ERA5, so it's known at any future time during a rollout. Both the start and end of the step are included so the model can tell whether the sun is rising or setting.

`Forcings` computes the channels on demand (~0.1 ms per sample) instead of storing them: precomputed for 1979–2015 they would take ~4 GB. Indexing it returns a `(9, 32, 64)` tensor, so it plugs into `OneStepDataset` as the forcing tensor, and `Forcings.at(times)` computes them for any times, which the rollout will need.

### Checking it against ERA5

```sh
python scripts/check_toa.py --output docs/images/phase2_toa.png
```

![TOA vs ERA5](../images/phase2_toa.png)

The difference panels are scaled to the largest error shown (about 1.2 W/m²). `toa()` comes out slightly too high on the morning side of the daylit area and slightly too low on the afternoon side, which is the pattern a small timing offset produces. The sun's position changes TOA by up to ~6 W/m² per minute, so 1.2 W/m² is roughly a 10–15 second difference from ERA5's solar calculation.

## Step 3: Latitude-Weighted Loss

The 64x32 grid has 64 cells in every latitude row, but a row near the pole covers far less of the globe than a row at the equator. Plain MSE counts every cell equally, so it spends as much effort on the tiny polar cells as on the huge tropical ones. WB2 weights each row by its area when scoring, so the model was being trained on a different objective from the one it's graded on.

`LatWeightedMSE` in [forecast.py](../../src/wbforecaster/forecast.py) multiplies each squared error by its row's area weight before averaging. The weights come from `lat_weights`, which uses WB2's own formula, the area of the row's cells (sin(upper edge) − sin(lower edge)), normalized to mean 1. On this grid that's practically identical to cos(latitude). An error in the polar row now costs ~5% of the same error at the equator. Because the weights average to 1, the loss stays on the same scale as plain MSE. Turn it on with `--lat-weighted-loss`. Validation uses the same loss, so the best epoch is picked by the weighted score.
