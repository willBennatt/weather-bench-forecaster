"""Train a model on ERA5 64x32, roll it out over a test period, and write a WB2-format Zarr forecast.

Example (defaults reproduce notebooks/01_t2m_z500_model_comparison.ipynb):
    python scripts/train_and_forecast.py --model simple --output forecast_simple.zarr
"""
import argparse

import torch

from wbforecaster.data import DEFAULT_CHANNELS, Normalizer, extract_channels, open_era5, to_tensor
from wbforecaster.forecast import rollout, to_wb2_dataset, train_one_step
from wbforecaster.models import MODELS


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=MODELS, default="simple")
    p.add_argument("--channels", nargs="+", default=DEFAULT_CHANNELS, help="e.g. 2m_temperature geopotential@500")
    p.add_argument("--train-start", default="2018-01-01")
    p.add_argument("--train-end", default="2018-12-31")
    p.add_argument("--test-start", default="2020-01-01")
    p.add_argument("--test-end", default="2020-01-15")
    p.add_argument("--lead-steps", type=int, default=8, help="number of 6h steps (8 = 48h)")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--output", default="my_forecast_64x32.zarr")
    p.add_argument("--checkpoint", help="optional path to save model weights + normalization stats")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = open_era5()

    print("Loading training data...")
    train_raw = extract_channels(ds.sel(time=slice(args.train_start, args.train_end)), args.channels)
    normalizer = Normalizer(train_raw)

    print(f"Training {args.model} on {device}...")
    model = MODELS[args.model](channels=len(args.channels))
    model = train_one_step(model, to_tensor(normalizer.normalize(train_raw)), device, args.epochs, args.batch_size)

    if args.checkpoint:
        torch.save(
            {"model": args.model, "channels": args.channels, "state_dict": model.state_dict(),
             "mean": normalizer.mean, "std": normalizer.std},
            args.checkpoint,
        )

    print("Rolling out forecasts on the test period...")
    test_slice = ds.sel(time=slice(args.test_start, args.test_end))
    forecast = rollout(model, extract_channels(test_slice, args.channels), normalizer, args.lead_steps, device)

    out = to_wb2_dataset(forecast, args.channels, test_slice.time.values, ds.latitude.values, ds.longitude.values)
    out.to_zarr(args.output, mode="w")
    print(f"Wrote {args.output}")
    print(out)


if __name__ == "__main__":
    main()
