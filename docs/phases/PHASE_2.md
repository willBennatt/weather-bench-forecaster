# Phase 2

## Goal

Phase 1 trained the model on the full weather bench dataset from 1979 - 2015 and compared it to persistence and climatology models as benchmarks. It also extended the forecast window from 2 days to 10 days on the `2m_temperature` and `geopotential@500` outputs.

However, it was unable to beat persistence at 24 hours because it did not recognize how time of day plays a role in temperature. The goal of phase 2 is to add forcing inputs like incoming solar radiation so the model can learn to change its outputs with the time of day to achieve more accurate forecasts and beat persistence at 24 hours.

## Step 1: Separate Inputs From Outputs

1. Separated `in_channels` from `out_channels` in [models.py](../../src/wbforecaster/models.py) so they can be different.
2. Kept the residual connection on the state channels only: `x[:, :out_channels] + net(x)`. The model adds its predicted change to the weather it forecasts, not to the forcings, which it never predicts.
3. Modified the dataset to take a forcing tensor and append the forcings for each step to the state as extra input channels, so each sample is still an `(input, target)` pair and the training loop doesn't change.
4. The training script now builds models with `in_channels` and `out_channels` and saves both in the checkpoint. Until the forcings exist, both equal the number of state channels.
