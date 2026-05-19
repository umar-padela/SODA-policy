# Scripts

| Script | Purpose |
|--------|---------|
| `setup_submodules.sh` | Init `third_party/diffusion_policy` and `third_party/love` |
| `download_data.sh` | Download Square data (Push-T zarr in `data/raw/pusht/`; regen labels via `soda/option_discovery/`) |
| `train_soda.sh` | Train SODA: `task=pusht\|square` `discovery=supervised\|unsupervised` |
| `eval_soda.sh` | Evaluation driver |

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

## Train (once implemented)

```bash
./scripts/train_soda.sh task=pusht discovery=supervised
```

Maps to `configs/pusht/soda_supervised.yaml`. See [`configs/README.md`](../configs/README.md).
