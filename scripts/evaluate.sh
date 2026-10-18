#!/usr/bin/env bash
# Score a WB2-format forecast against ERA5 with the official WeatherBench 2 evaluation script.
#
# Usage: scripts/evaluate.sh FORECAST.zarr OUTPUT_DIR [VARIABLES] [LEVELS]
#   VARIABLES defaults to "2m_temperature,geopotential", LEVELS to "500".
#
# Requires a checkout of https://github.com/google-research/weatherbench2 at $WB2_DIR
# (default ./weatherbench2) installed with `pip install -e`.
#
# Set BASELINES=1 to also score persistence and climatology over the same period —
# always compare against these before claiming a model has skill.
set -euo pipefail

FORECAST=${1:?forecast zarr path required}
OUTPUT_DIR=${2:?output dir required}
VARIABLES=${3:-2m_temperature,geopotential}
LEVELS=${4:-500}
WB2_DIR=${WB2_DIR:-weatherbench2}

EXTRA=()
if [[ "${BASELINES:-0}" == "1" ]]; then
  EXTRA+=(--evaluate_persistence --evaluate_climatology)
fi

python "$WB2_DIR/scripts/evaluate.py" \
  --forecast_path="$FORECAST" \
  --obs_path="gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr" \
  --climatology_path="gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_64x32_equiangular_conservative.zarr" \
  --output_dir="$OUTPUT_DIR" \
  --variables="$VARIABLES" \
  --levels="$LEVELS" \
  --runner=DirectRunner \
  "${EXTRA[@]}"
