"""Train a model on ERA5 64x32, roll it out over a test period, and write a WB2-format Zarr forecast.

Example (defaults match the v0 notebook setup, plus learning-rate decay):
    python scripts/train_and_forecast.py --model simple --output forecast_simple.zarr
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from wbforecaster.data import (
    DEFAULT_CHANNELS, ERA5_64X32, Normalizer, extract_channels, get_device, open_era5, to_tensor,
)
from wbforecaster.forecast import rollout, to_wb2_dataset, train_one_step
from wbforecaster.models import MODELS
from wbforecaster.forcings import Forcings


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=MODELS, default="simple")
    p.add_argument("--data", default=ERA5_64X32, help="WB2 bucket URL or a local cache from download_era5.py")
    p.add_argument("--channels", nargs="+", default=DEFAULT_CHANNELS, help="e.g. 2m_temperature geopotential@500")
    p.add_argument("--train-start", default="2018-01-01")
    p.add_argument("--train-end", default="2018-12-31")
    p.add_argument("--test-start", default="2020-01-01")
    p.add_argument("--test-end", default="2020-01-15")
    p.add_argument("--val-start", help="validation period start, e.g. 2016-01-01 (WB2 standard: 2016-2017)")
    p.add_argument("--val-end", help="validation period end, e.g. 2017-12-31")
    p.add_argument("--lead-steps", type=int, default=8, help="number of 6h steps (8 = 48h, 40 = 10 days)")
    p.add_argument("--init-hours", type=int, nargs="+", help="only start forecasts at these UTC hours (WB2 standard: 0 12)")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--patience", type=int, help="stop after this many epochs without validation improvement")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3, help="initial learning rate (decays to lr/100)")
    p.add_argument("--output", default="my_forecast_64x32.zarr")
    p.add_argument("--checkpoint", help="path to save model weights + normalization stats (best validation epoch)")
    p.add_argument("--no-forcings", action="store_true",
                   help="train on the state alone, without solar/time/static forcing inputs (the Phase 1 setup)")
    p.add_argument("--lat-weighted-loss", action="store_true",
                   help="weight the loss by grid-cell area (cos latitude), matching how WB2 scores forecasts")
    args = p.parse_args()

    device = get_device()
    ds = open_era5(args.data)

    def forcings_for(period):
        """Forcings aligned with `period`'s times (forcing i is for the step starting at time i), or None."""
        return None if args.no_forcings else Forcings(period.time.values, ds)

    print("Loading training data...")
    train_slice = ds.sel(time=slice(args.train_start, args.train_end))
    train_raw = extract_channels(train_slice, args.channels)
    normalizer = Normalizer(train_raw)
    val_tensor = val_forcing = None
    if args.val_start and args.val_end:
        print("Loading validation data...")
        val_slice = ds.sel(time=slice(args.val_start, args.val_end))
        val_raw = extract_channels(val_slice, args.channels)
        val_tensor = to_tensor(normalizer.normalize(val_raw))  # train statistics, never validation's own
        val_forcing = forcings_for(val_slice)

    def save_checkpoint(model, epoch=None, val_loss=None):
        Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model": args.model, "channels": args.channels, "in_channels": model_kwargs["in_channels"],
             "out_channels": model_kwargs["out_channels"], "forcings": forcing_names,
             "lat_weighted_loss": args.lat_weighted_loss, "state_dict": model.state_dict(),
             "mean": normalizer.mean, "std": normalizer.std, "epoch": epoch, "val_loss": val_loss},
            args.checkpoint,
        )

    forcing_names = [] if args.no_forcings else Forcings.names
    # The model sees the state followed by the forcing channels, and predicts only the state.
    model_kwargs = {"in_channels": len(args.channels) + len(forcing_names), "out_channels": len(args.channels)}
    print(f"Training {args.model} on {device} with forcings: {forcing_names or 'none'}...")
    model = MODELS[args.model](**model_kwargs)
    model = train_one_step(
        model, to_tensor(normalizer.normalize(train_raw)), forcings_for(train_slice), device, args.epochs,
        args.batch_size, args.lr, val_tensor=val_tensor, val_forcing=val_forcing, patience=args.patience,
        latitudes=ds.latitude.values if args.lat_weighted_loss else None,
        on_best=save_checkpoint if args.checkpoint else None,
    )
    if args.checkpoint and val_tensor is None:
        save_checkpoint(model, args.epochs)

    print("Rolling out forecasts on the test period...")
    test_slice = ds.sel(time=slice(args.test_start, args.test_end))
    init_times = test_slice.time.values[: len(test_slice.time) - args.lead_steps]
    init_indices = np.arange(len(init_times))
    if args.init_hours:
        init_indices = init_indices[np.isin(test_slice.time.dt.hour.values[init_indices], args.init_hours)]
    forecast = rollout(model, extract_channels(test_slice, args.channels), normalizer, args.lead_steps, device,
                       init_indices, forcings=forcings_for(test_slice))

    out = to_wb2_dataset(forecast, args.channels, init_times[init_indices], ds.latitude.values, ds.longitude.values)
    out.to_zarr(args.output, mode="w")
    print(f"Wrote {args.output}")
    print(out)


if __name__ == "__main__":
    main()
