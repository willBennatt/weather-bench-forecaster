"""Training, autoregressive rollout, and export to the WeatherBench 2 forecast format."""
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


def train_one_step(model, train_tensor, device, epochs=5, batch_size=16, lr=1e-3):
    """Fit the model to predict the next 6-hour state with MSE loss on normalized data."""
    loader = DataLoader(OneStepDataset(train_tensor), batch_size=batch_size, shuffle=True)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss / len(loader):.5f}")
    return model


@torch.no_grad()
def rollout(model, test_raw, normalizer, lead_steps, device):
    """Run the model autoregressively from every test time.

    Returns an array of shape (num_inits, lead_steps + 1, C, Lat, Lon) in physical units,
    where lead index 0 is the initial condition itself.
    """
    num_inits = len(test_raw) - lead_steps
    forecast = np.zeros((num_inits, lead_steps + 1, *test_raw.shape[1:]), dtype=np.float32)
    model.eval()
    for i in range(num_inits):
        state = torch.tensor(normalizer.normalize(test_raw[i : i + 1]), dtype=torch.float32, device=device)
        forecast[i, 0] = test_raw[i]
        for step in range(1, lead_steps + 1):
            state = model(state)
            forecast[i, step] = normalizer.denormalize(state.cpu().numpy())[0]
    return forecast


def to_wb2_dataset(forecast, channels, init_times, latitudes, longitudes, step_hours=6):
    """Package a rollout as an xarray Dataset with WB2 dims (time, prediction_timedelta, [level], lat, lon)."""
    lead_times = [np.timedelta64(step * step_hours, "h") for step in range(forecast.shape[1])]
    coords = {
        "time": init_times[: forecast.shape[0]],
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
