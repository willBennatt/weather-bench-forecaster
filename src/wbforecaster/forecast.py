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


def train_one_step(model, train_tensor, device, epochs=5, batch_size=16, lr=1e-3,
                   val_tensor=None, patience=None, on_best=None):
    """Fit the model to predict the next 6-hour state with MSE loss on normalized data.

    The learning rate follows a cosine decay from `lr` to `lr / 100` over all training steps:
    large steps early to make fast progress, small steps late to settle into a minimum.

    If `val_tensor` is given, the validation loss is computed after every epoch and the weights
    with the lowest validation loss are restored at the end. `on_best(model, epoch, val_loss)`
    is called each time the validation loss improves (e.g. to save a checkpoint), and training
    stops early after `patience` epochs without improvement.
    """
    loader = DataLoader(OneStepDataset(train_tensor), batch_size=batch_size, shuffle=True)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(loader), eta_min=lr / 100)
    criterion = nn.MSELoss()

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
            val_loss = evaluate_loss(model, val_tensor, device, criterion)
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
def evaluate_loss(model, data_tensor, device, criterion, batch_size=256):
    """Average one-step loss over a dataset, without updating the model."""
    model.eval()
    loader = DataLoader(OneStepDataset(data_tensor), batch_size=batch_size)
    total = 0.0
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        total += criterion(model(x_batch), y_batch).item() * len(x_batch)
    return total / len(loader.dataset)


@torch.no_grad()
def rollout(model, test_raw, normalizer, lead_steps, device, init_indices=None, batch_size=64):
    """Run the model autoregressively from each initial time in `init_indices` (default: all that fit).

    Returns an array of shape (num_inits, lead_steps + 1, C, Lat, Lon) in physical units,
    where lead index 0 is the initial condition itself.
    """
    if init_indices is None:
        init_indices = np.arange(len(test_raw) - lead_steps)
    forecast = np.zeros((len(init_indices), lead_steps + 1, *test_raw.shape[1:]), dtype=np.float32)
    model.eval()
    # Forecasts from different start times are independent, so run them in batches.
    for b in range(0, len(init_indices), batch_size):
        idx = init_indices[b : b + batch_size]
        forecast[b : b + len(idx), 0] = test_raw[idx]
        state = torch.tensor(normalizer.normalize(test_raw[idx]), dtype=torch.float32, device=device)
        for step in range(1, lead_steps + 1):
            state = model(state)
            forecast[b : b + len(idx), step] = normalizer.denormalize(state.cpu().numpy())
    return forecast


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
