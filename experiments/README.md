# Experiments (outputs)

Checkpoints, logs, and wandb runs go here. **Contents are gitignored** except this README.

## Suggested layout

```text
experiments/
  pusht/
    e1_supervised_low/
    e1_supervised_high/
  square/
    ...
```

Set `checkpoint.save_dir` in your config YAML to a path under this folder.

See [`project_plan.md`](../project_plan.md) for the experiment registry (E1–E4, P0).
