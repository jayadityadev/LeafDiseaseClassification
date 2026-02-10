#!/usr/bin/env bash
set -euo pipefail

# Fetch prebuilt model artifact for deployment.
# Usage:
#   MODEL_URL="https://.../leaf_densenet121_final.keras" bash scripts/fetch_model.sh

DEST_DIR="models/leaf/current/densenet121"
mkdir -p "$DEST_DIR"

if [ -z "${MODEL_URL:-}" ]; then
  echo "ERROR: MODEL_URL environment variable not set."
  echo "Set MODEL_URL to a direct downloadable URL for the .keras model file."
  exit 1
fi

OUT_PATH="$DEST_DIR/$(basename "$MODEL_URL")"
echo "Downloading model from: $MODEL_URL"
curl -fL "$MODEL_URL" -o "$OUT_PATH"
echo "Model saved to: $OUT_PATH"

# Normalize file name expected by app
if [[ "$OUT_PATH" != *leaf_densenet121_final.keras ]]; then
  mv "$OUT_PATH" "$DEST_DIR/leaf_densenet121_final.keras"
  echo "Renamed model to: $DEST_DIR/leaf_densenet121_final.keras"
fi

echo "Done."
