# Experiments (outputs)

Checkpoints, logs, and wandb runs go here. **Contents are gitignored** except this README.

## Suggested layout

```text
experiments/
  pusht/
    dp_baseline/              # optional local copy of frozen DP .ckpt (gitignored)
      latest.ckpt
    e1_supervised_low/
    e1_supervised_high/
  square/
    ...
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

Eval outputs live under `/experiments/eval/pusht/<descriptive_run_name>/` (see `soda/eval/run_naming.py`).

See [`modal/README.md`](../modal/README.md) and [`configs/pusht/baseline_vanilla.yaml`](../configs/pusht/baseline_vanilla.yaml).

Set `checkpoint.save_dir` in SODA training configs to a path under this folder when training is implemented.

See [`project_plan.md`](../project_plan.md) for the experiment registry (E1–E4, P0).
