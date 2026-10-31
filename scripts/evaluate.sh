#!/usr/bin/env bash
# Score a WB2-format forecast against ERA5 with the official WeatherBench 2 evaluation script.
#
# Usage: scripts/evaluate.sh FORECAST.zarr OUTPUT_DIR [VARIABLES] [LEVELS]
#   VARIABLES defaults to "2m_temperature,geopotential", LEVELS to "500".
#
# Requires a checkout of https://github.com/google-research/weatherbench2 at $WB2_DIR
# (default ./weatherbench2) installed with `pip install -e`, plus `pip install absl-py`.
#
# Set BASELINES=1 to also score persistence and climatology over the same initialization times,
# written to OUTPUT_DIR/persistence/ and OUTPUT_DIR/climatology/. Always compare against these
# before claiming a model has skill.
#
# The ERA5 truth and climatology come from data/eval/ when scripts/cache_eval_data.py has created it
# (much faster), otherwise they're streamed from the WB2 bucket. Set OBS_PATH / CLIMATOLOGY_PATH to
# override either one.
set -euo pipefail

FORECAST=${1:?forecast zarr path required}
OUTPUT_DIR=${2:?output dir required}
VARIABLES=${3:-2m_temperature,geopotential}
LEVELS=${4:-500}
WB2_DIR=${WB2_DIR:-weatherbench2}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
GCS=gs://weatherbench2/datasets
if [[ -d data/eval/obs.zarr && -d data/eval/climatology.zarr ]]; then
  OBS_PATH=${OBS_PATH:-data/eval/obs.zarr}
  CLIMATOLOGY_PATH=${CLIMATOLOGY_PATH:-data/eval/climatology.zarr}
else
  echo "No local cache in data/eval/ (see scripts/cache_eval_data.py): streaming ERA5 from GCS, which is slow."
  OBS_PATH=${OBS_PATH:-$GCS/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr}
  CLIMATOLOGY_PATH=${CLIMATOLOGY_PATH:-$GCS/era5-hourly-climatology/1990-2019_6h_64x32_equiangular_conservative.zarr}
fi
echo "Truth: $OBS_PATH"
echo "Climatology: $CLIMATOLOGY_PATH"

evaluate() {  # evaluate FORECAST OUTPUT_DIR [extra flags...]
  python "$WB2_DIR/scripts/evaluate.py" \
    --forecast_path="$1" \
    --obs_path="$OBS_PATH" \
    --climatology_path="$CLIMATOLOGY_PATH" \
    --output_dir="$2" \
    --variables="$VARIABLES" \
    --levels="$LEVELS" \
    --runner=DirectRunner \
    "${@:3}"
}

echo "Scoring $FORECAST..."
evaluate "$FORECAST" "$OUTPUT_DIR"

if [[ "${BASELINES:-0}" == "1" ]]; then
  echo "Scoring persistence..."
  PERSISTENCE="${FORECAST%/}"
  PERSISTENCE="${PERSISTENCE%.zarr}_persistence.zarr"
  python "$SCRIPT_DIR/make_persistence.py" "$FORECAST" "$PERSISTENCE"
  evaluate "$PERSISTENCE" "$OUTPUT_DIR/persistence"

  # --evaluate_climatology swaps the forecast for the climatology at each valid time;
  # the forecast is only used for its times and variables.
  echo "Scoring climatology..."
  evaluate "$FORECAST" "$OUTPUT_DIR/climatology" --evaluate_climatology
fi
