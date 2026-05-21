#!/usr/bin/env bash
# Download frozen DP Push-T *image* baseline (train_0 latest.ckpt) — do not commit.
# Usage (repo root, Git Bash / WSL):
#   bash scripts/download_dp_pusht_checkpoint.sh

set -euo pipefail
cd "$(dirname "$0")/.."

OUT_DIR="experiments/pusht/dp_baseline"
mkdir -p "$OUT_DIR"

BASE="https://diffusion-policy.cs.columbia.edu/data/experiments/image/pusht/diffusion_policy_cnn/train_0/checkpoints"
URL="${BASE}/latest.ckpt"
OUT="${OUT_DIR}/latest.ckpt"

echo "Downloading $URL"
echo "  -> $OUT"
wget -q --show-progress -O "$OUT" "$URL"
echo "Done. Use this path for local debugging or copy to Modal Volume under /experiments/dp_baselines/"
