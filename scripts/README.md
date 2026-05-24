# Scripts

| Script | Purpose |
|--------|---------|
| `setup_submodules.sh` | Init `third_party/diffusion_policy` and `third_party/love` |
| `download_dp_pusht_checkpoint.sh` | Optional local copy of frozen DP image baseline (~3 GB; gitignored) |
| `download_data.sh` | Download Square data (Push-T zarr in `data/raw/pusht/`; regen labels via `soda/option_discovery/`) |
| `train_soda.sh` | Train SODA: `task=pusht\|square` `discovery=supervised\|unsupervised` |
| `eval_soda.sh` | Evaluation driver |
| `sweep_train_low.py` | A4 π_low LR sweep (3 trials × short epochs; ranks `val_loss_diffusion`) |
| `sweep_train_high.py` | B4 π_high sweep (default LR; `--full-grid` for 16-trial factorial) |
| `rollout_segment.py` | A3b expert-anchored π_low rollout from zarr segment start (side-by-side MP4) |
| `modal/modal_rollout_low_policy.py` | Same as above on Modal GPU + Volume (no local checkpoint download) |

## Submodule setup (runbook §7 row **8**)

From repo root (Git Bash or WSL on Windows):

```bash
bash scripts/setup_submodules.sh
```

Or manually — remove placeholder `.gitkeep` dirs first, then:

```bash
git submodule add https://github.com/real-stanford/diffusion_policy.git third_party/diffusion_policy
git submodule add https://github.com/yidingjiang/love.git third_party/love
git submodule update --init --recursive
```

Pin a commit SHA inside each submodule when ready, then `git add .gitmodules third_party/`.

## Train

```bash
# π_low first, then π_high (set low_checkpoint on second run)
python soda/training/train_low.py  --config-path configs/pusht --config-name soda_supervised
python soda/training/train_high.py --config-path configs/pusht --config-name soda_supervised \
  train_high.low_checkpoint=experiments/train_low/soda_supervised/best.ckpt

# Or wrapper (if maintained):
./scripts/train_soda.sh task=pusht discovery=supervised
```

Maps to `configs/pusht/soda_supervised.yaml`. See [`configs/README.md`](../configs/README.md) and [`training_plan.md`](../training_plan.md).

## Sweeps (training_plan A4 / B4)

```bash
python scripts/sweep_train_low.py --num-epochs 14 --dry-run --run-readme "A4 pi_low LR sweep"
python scripts/sweep_train_low.py --num-epochs 14 --run-readme "A4 pi_low LR sweep"

python scripts/sweep_train_high.py \
  --low-checkpoint experiments/train_low/soda_supervised/best.ckpt \
  --num-epochs 14 --run-readme "B4 pi_high LR sweep"
```

Add `--modal` to launch trials on Modal. Use `--modal --no-detach` to wait for jobs,
sync `metrics.json` from the Volume, and print a local ranking table. Detached Modal
runs persist under `/experiments/sweep_low/` or `/experiments/sweep_high/` on the Volume.
