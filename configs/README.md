# Configs

Hydra-style YAML recipes per **task** and **supervision type**. See [`project_plan.md`](../project_plan.md) §4.3.

## Naming

Same two filenames under each task folder; the folder picks dataset path and label source:

| Config | `pusht/` | `square/` |
|--------|----------|-----------|
| `soda_supervised.yaml` | E1 — heuristic `option_id` | E2 — VLM `option_id` |
| `soda_unsupervised.yaml` | E3 — LOVE `option_id` | E4 — LOVE `option_id` |
| `baseline_vanilla.yaml` | Frozen Diffusion Policy baseline | Frozen DP baseline |

## Example

```bash
python soda/training/train_low.py --config-path configs/pusht --config-name soda_supervised
```

Or via [`scripts/train_soda.sh`](../scripts/train_soda.sh):

```bash
./scripts/train_soda.sh task=pusht discovery=supervised
```
