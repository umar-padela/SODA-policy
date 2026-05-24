# Experiments (outputs)

Checkpoints, logs, and wandb runs go here. **Contents are gitignored** except this README.

## Suggested layout

```text
experiments/
  train_low/
    soda_supervised/          # default when train_low.output_dir is null
      best.ckpt
      metrics.json
  train_high/
    soda_supervised/
  pusht/
    dp_frozen/              # optional local copy of frozen DP .ckpt (gitignored)
  eval/
    pusht/                    # sim eval outputs (also on Modal Volume)
```

## Frozen DP Push-T baseline (§7 row 8)

Official **image** policy (`diffusion_policy_cnn`, seed `train_0`), not `low_dim`:

- [checkpoints/](https://diffusion-policy.cs.columbia.edu/data/experiments/image/pusht/diffusion_policy_cnn/train_0/checkpoints/)
- Eval uses `latest.ckpt` (~3 GB) only

**Do not commit** `.ckpt` files.

Download locally (optional):

```bash
bash scripts/download_dp_pusht_checkpoint.sh
```

**On Modal:** download once, then pass the path to every eval:

```bash
modal run modal/modal_download_dp.py
modal run modal/modal_eval.py --checkpoint /experiments/dp_baselines/pusht_image_cnn_train0/latest.ckpt
```

Eval outputs live under `/experiments/eval/pusht/<descriptive_run_name>/` on the container mount (see `soda/eval/run_naming.py`). From your laptop, use volume path `eval/pusht/<run_dir_name>/` (no `/experiments` prefix) — see [`modal/README.md`](../modal/README.md).

See [`modal/README.md`](../modal/README.md) and [`configs/pusht/dp_frozen.yaml`](../configs/pusht/dp_frozen.yaml).

Training writes to `experiments/train_low/{name}/` and `experiments/train_high/{name}/` by default (`name` from yaml, e.g. `soda_supervised`). Override with `train_low.output_dir` / `train_high.output_dir`.

See [`project_plan.md`](../project_plan.md) for the experiment registry (E1–E4, P0).
