# Experiments (outputs)

Checkpoints, logs, eval runs, and qualitative videos go here. **Contents are gitignored** except this README.

## Layout (task-first)

All paths are scoped by **task** (`pusht`, `square`, …) so environments stay isolated on disk and on the Modal Volume.

```text
experiments/
  pusht/
    train_low/
      soda_supervised/          # default when train_low.output_dir is null
        best.ckpt
        latest.ckpt
        metrics.json
        epoch_*.ckpt
        runs/{YYYYMMDD_HHMMSS}/ # timestamped archive per train run
    train_high/
      soda_supervised/
    train_dp/
      dp/
    sweep_low/
      lr1.00e-04/               # A4 LR sweep trials
    sweep_high/
      lr5.00e-05/
    segment_rollout/            # expert | policy segment MP4 + JSON
    eval/
      dp_frozen/
        {YYYYMMDD}/{HHMMSS}/    # config.yaml, eval_log.json, media/*.mp4
      soda_supervised/
    dp_baselines/
      pusht_image_cnn_train0/   # frozen Columbia DP (one-time download)
        latest.ckpt
    _hydra/
      train_low/soda_supervised/  # Hydra scratch logs on Modal (not checkpoints)
  square/
    ...                         # same subdirs when Square experiments start
```

**Modal Volume `soda-experiments`:** same tree under mount `/experiments`. For `modal volume ls/get`, omit the `/experiments` prefix — e.g. `pusht/train_low/soda_supervised/best.ckpt`.

Path helpers live in `soda/experiments/paths.py` (used by training, sweeps, eval, and Modal).

## Frozen DP Push-T baseline (§7 row 8)

Official **image** policy (`diffusion_policy_cnn`, seed `train_0`):

- [checkpoints](https://diffusion-policy.cs.columbia.edu/data/experiments/image/pusht/diffusion_policy_cnn/train_0/checkpoints/)
- Eval uses `latest.ckpt` (~3 GB) only

**Do not commit** `.ckpt` files.

Download locally (optional):

```bash
bash scripts/download_dp_pusht_checkpoint.sh
```

**On Modal:** download once, then eval with yaml path:

```bash
modal run modal/modal_download_dp.py
modal run modal/modal_eval.py --config configs/pusht/dp_frozen.yaml
```

Checkpoint on Volume: `/experiments/pusht/dp_baselines/pusht_image_cnn_train0/latest.ckpt`

## Training defaults

| Script | Default output (local) | Modal override |
|--------|------------------------|----------------|
| `train_low.py` | `experiments/{task}/train_low/{name}/` | `/experiments/{task}/train_low/{name}/` |
| `train_high.py` | `experiments/{task}/train_high/{name}/` | same pattern |
| `train_dp.py` | `experiments/{task}/train_dp/{name}/` | same pattern |

`{task}` = `pusht` or `square` (from `configs/{task}/`). `{name}` = yaml `name:` (e.g. `soda_supervised`).

Override with `train_low.output_dir` / `train_high.output_dir` / `train_high.low_checkpoint` in Hydra.

See [`modal/README.md`](../modal/README.md), [`configs/pusht/dp_frozen.yaml`](../configs/pusht/dp_frozen.yaml), and [`project_plan.md`](../project_plan.md).

## Migrating from the old flat layout

If your Volume still has top-level `train_low/`, `eval/`, `segment_rollout/`, etc., move them under `pusht/` (or re-download / re-train). Old paths are no longer written by default.
