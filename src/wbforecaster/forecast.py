"""Training, autoregressive rollout, and export to the WeatherBench 2 forecast format."""
import copy
import time

import numpy as np
import torch
import torch.nn as nn
import xarray as xr
from torch.utils.data import DataLoader

from .data import OneStepDataset, parse_channel

UNITS = {
    "2m_temperature": ("K", "2 metre temperature"),
    "temperature": ("K", "Temperature"),
    "geopotential": ("m**2 s**-2", "Geopotential"),
    "mean_sea_level_pressure": ("Pa", "Mean sea level pressure"),
    "10m_u_component_of_wind": ("m s**-1", "10 metre U wind component"),
    "10m_v_component_of_wind": ("m s**-1", "10 metre V wind component"),
    "u_component_of_wind": ("m s**-1", "U component of wind"),
    "v_component_of_wind": ("m s**-1", "V component of wind"),
    "specific_humidity": ("kg kg**-1", "Specific humidity"),
}


def lat_weights(latitudes):
    """Area weights for each latitude row, normalized to mean 1, exactly as WB2 computes them.

    Each weight is the area of the row's grid cells: sin(upper edge) - sin(lower edge), the integral
    of cos(latitude) across the cell. On the 64x32 grid this is practically identical to cos(latitude).
    """
    lat = np.radians(np.asarray(latitudes, dtype=np.float64))
    edges = np.concatenate([[-np.pi / 2], (lat[:-1] + lat[1:]) / 2, [np.pi / 2]])
    weights = np.sin(edges[1:]) - np.sin(edges[:-1])
    return weights / weights.mean()


class LatWeightedMSE(nn.Module):
    """Mean squared error with each latitude row weighted by its area (see `lat_weights`).

    The grid has as many cells near the poles as at the equator, but polar cells cover far less of the
    globe. Plain MSE counts every cell equally, so it overweights the poles; WB2 weights by area when
    scoring, and this loss trains on the same thing it's scored on. The weights have mean 1, so the
    loss is on the same scale as plain MSE.
    """

    def __init__(self, latitudes):
        super().__init__()
        weights = torch.tensor(lat_weights(latitudes), dtype=torch.float32)
        # Shape (1, 1, Lat, 1) to broadcast over (batch, channel, lat, lon). A buffer moves with .to(device).
        self.register_buffer("weights", weights.view(1, 1, -1, 1))

    def forward(self, prediction, target):
        return ((prediction - target) ** 2 * self.weights).mean()

# Surface variables' weights relative to a pressure-level variable, following GraphCast: 2 m temperature
# counts fully, the other surface fields at 0.1. Any surface variable not listed gets SURFACE_DEFAULT.
SURFACE_WEIGHTS = {"2m_temperature": 1.0}
SURFACE_DEFAULT = 0.1


def channel_weights(channels):
    """Per-channel loss weights, normalized to mean 1, following GraphCast.

    A pressure-level channel is weighted by its pressure relative to the mean of the levels in use,
    so 1000 hPa (near the surface) counts ~20x as much as 50 hPa (the stratosphere): the lower
    atmosphere has more of the air's mass and most of the weather. Surface channels get
    SURFACE_WEIGHTS. Without this, every channel counts equally, so 50 hPa matters as much as 850.

    With the Phase 3 channels the 4 surface channels end up with ~4% of the loss (vs ~10% unweighted),
    as in GraphCast: the near-surface levels (925, 1000 hPa) carry most of the same signal at high weight.
    """
    parsed = [parse_channel(channel) for channel in channels]
    levels = [level for _, level in parsed if level is not None]
    mean_level = np.mean(levels) if levels else 1.0
    weights = np.array([level / mean_level if level is not None else SURFACE_WEIGHTS.get(var, SURFACE_DEFAULT)
                        for var, level in parsed])
    return weights / weights.mean()


class LatAndPressureWeightedMSE(LatWeightedMSE):
    """`LatWeightedMSE` with each channel also weighted by `channel_weights` (pressure level, surface variable).

    Both sets of weights have mean 1, so the loss stays on the same scale as plain MSE.
    """

    def __init__(self, latitudes, channels):
        super().__init__(latitudes)
        weights = torch.tensor(channel_weights(channels), dtype=torch.float32)
        # Shape (1, C, 1, 1) to broadcast over (batch, channel, lat, lon).
        self.register_buffer("channel_weights", weights.view(1, -1, 1, 1))

    def forward(self, prediction, target):
        return ((prediction - target) ** 2 * self.weights * self.channel_weights).mean()


def train_one_step(model, train_tensor, forcing_tensor, device, epochs=5, batch_size=16, lr=1e-3,
                   val_tensor=None, val_forcing=None, patience=None, on_best=None, latitudes=None, channels=None):
    """Fit the model to predict the next 6-hour state with MSE loss on normalized data.

    If `latitudes` is given, the loss weights each latitude row by its area (`LatWeightedMSE`), as WB2
    does when scoring; otherwise every grid cell counts equally. If `channels` (names like
    "geopotential@500") is given too, each channel is also weighted by pressure level and surface
    variable (`LatAndPressureWeightedMSE`).

    `forcing_tensor` (and `val_forcing`) hold the forcings for the step starting at each time in
    `train_tensor` (and `val_tensor`), e.g. a `Forcings` over the same times, or None for no forcings.

    The learning rate follows a cosine decay from `lr` to `lr / 100` over all training steps:
    large steps early to make fast progress, small steps late to settle into a minimum.

    If `val_tensor` is given, the validation loss is computed after every epoch and the weights
    with the lowest validation loss are restored at the end. `on_best(model, epoch, val_loss)`
    is called each time the validation loss improves (e.g. to save a checkpoint), and training
    stops early after `patience` epochs without improvement.
    """
    loader = DataLoader(OneStepDataset(train_tensor, forcing_tensor), batch_size=batch_size, shuffle=True)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(loader), eta_min=lr / 100)
    if latitudes is None:
        criterion = nn.MSELoss()
    elif channels is None:
        criterion = LatWeightedMSE(latitudes)
    else:
        criterion = LatAndPressureWeightedMSE(latitudes, channels)
    criterion = criterion.to(device)

    best_loss, best_state, epochs_since_best = float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        start = time.time()
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        message = f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss / len(loader):.5f}"

        if val_tensor is not None:
            val_loss = evaluate_loss(model, val_tensor, device, criterion, val_forcing)
            message += f" - Val loss: {val_loss:.5f}"
            if val_loss < best_loss:
                best_loss, epochs_since_best = val_loss, 0
                best_state = copy.deepcopy(model.state_dict())
                message += " (best)"
                if on_best:
                    on_best(model, epoch + 1, val_loss)
            else:
                epochs_since_best += 1
        print(f"{message} - LR: {scheduler.get_last_lr()[0]:.1e} - {time.time() - start:.0f}s", flush=True)

        if patience is not None and epochs_since_best >= patience:
            print(f"Stopping early: no validation improvement for {patience} epochs")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best weights (val loss {best_loss:.5f})")
    return model


@torch.no_grad()
def evaluate_loss(model, data_tensor, device, criterion, forcing_tensor=None, batch_size=256):
    """Average one-step loss over a dataset, without updating the model."""
    model.eval()
    loader = DataLoader(OneStepDataset(data_tensor, forcing_tensor), batch_size=batch_size)
    total = 0.0
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        total += criterion(model(x_batch), y_batch).item() * len(x_batch)
    return total / len(loader.dataset)


@torch.no_grad()
def rollout_batches(model, test_raw, normalizer, lead_steps, device, init_indices=None, batch_size=64,
                    forcings=None):
    """Run the model autoregressively from each initial time in `init_indices` (default: all that fit).

    `forcings` is a `Forcings` over the same times as `test_raw`, or None for a model without forcings.
    Only the initial state comes from the data: at every later step the input is the model's own
    previous prediction plus the forcings computed for that step's time.

    Forecasts from different start times are independent, so they run `batch_size` at a time. Yields
    `(start, forecast)` for each batch, where `forecast` covers `init_indices[start : start + len(forecast)]`
    and has shape (batch, lead_steps + 1, C, Lat, Lon) in physical units; lead index 0 is the initial
    condition itself. Only one batch is in memory at a time, so a full set of forecasts never has to fit.
    """
    if init_indices is None:
        init_indices = np.arange(len(test_raw) - lead_steps)
    model.eval()
    for b in range(0, len(init_indices), batch_size):
        idx = init_indices[b : b + batch_size]
        forecast = np.zeros((len(idx), lead_steps + 1, *test_raw.shape[1:]), dtype=np.float32)
        forecast[:, 0] = test_raw[idx]
        state = torch.tensor(normalizer.normalize(test_raw[idx]), dtype=torch.float32, device=device)
        for step in range(1, lead_steps + 1):
            x = state
            if forcings is not None:
                # This step starts (step - 1) * 6 h after each initial time.
                f = forcings.at(forcings.times[idx + step - 1])
                x = torch.cat([state, torch.from_numpy(f).to(device)], dim=1)  # state first, then forcings
            state = model(x)
            forecast[:, step] = normalizer.denormalize(state.cpu().numpy())
        yield b, forecast


def rollout(model, test_raw, normalizer, lead_steps, device, init_indices=None, batch_size=64, forcings=None):
    """All of `rollout_batches` in one array of shape (num_inits, lead_steps + 1, C, Lat, Lon).

    Fine for a few channels; for many channels and a year of forecasts use `write_forecast_zarr`.
    """
    batches = rollout_batches(model, test_raw, normalizer, lead_steps, device, init_indices, batch_size, forcings)
    return np.concatenate([forecast for _, forecast in batches])


def write_forecast_zarr(batches, path, channels, init_times, latitudes, longitudes):
    """Write `rollout_batches` output to a WB2-format Zarr store one batch at a time.

    Each batch is appended along `time`. The store is chunked one forecast (init time) per chunk,
    which is how the WB2 evaluation reads it.
    """
    for start, forecast in batches:
        ds = to_wb2_dataset(forecast, channels, init_times[start : start + len(forecast)], latitudes, longitudes)
        if start == 0:
            encoding = {var: {"chunks": (1, *ds[var].shape[1:])} for var in ds.data_vars}
            ds.to_zarr(path, mode="w", encoding=encoding)
        else:
            ds.to_zarr(path, append_dim="time")
        print(f"  wrote forecasts {start + len(forecast)}/{len(init_times)}", flush=True)


def to_wb2_dataset(forecast, channels, init_times, latitudes, longitudes, step_hours=6):
    """Package a rollout as an xarray Dataset with WB2 dims (time, prediction_timedelta, [level], lat, lon)."""
    lead_times = [np.timedelta64(step * step_hours, "h") for step in range(forecast.shape[1])]
    coords = {
        "time": init_times,
        "prediction_timedelta": lead_times,
        "latitude": latitudes,
        "longitude": longitudes,
    }
    dims = ["time", "prediction_timedelta", "latitude", "longitude"]

    # Group pressure-level channels of the same variable into one array with a `level` dim.
    by_var = {}
    for c, channel in enumerate(channels):
        var, level = parse_channel(channel)
        by_var.setdefault(var, []).append((level, c))

    data_vars = {}
    for var, entries in by_var.items():
        units, long_name = UNITS.get(var, ("", var))
        attrs = {"units": units, "long_name": long_name}
        if entries[0][0] is None:
            data_vars[var] = xr.DataArray(forecast[:, :, entries[0][1]], dims=dims, coords=coords, attrs=attrs)
        else:
            levels = [level for level, _ in entries]
            stacked = np.stack([forecast[:, :, c] for _, c in entries], axis=2)
            data_vars[var] = xr.DataArray(
                stacked,
                dims=dims[:2] + ["level"] + dims[2:],
                coords={**coords, "level": levels},
                attrs=attrs,
            )
    return xr.Dataset(data_vars)
