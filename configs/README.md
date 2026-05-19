# Configs

Hydra-style YAML recipes per **task** and **supervision type**. See [`project_plan.md`](../project_plan.md) §2 (option discovery) and §4.3 (config files).

## Naming

Same two filenames under each task folder; the folder picks dataset path and label source:

| Config | `pusht/` | `square/` |
|--------|----------|-----------|
| `soda_supervised.yaml` | E1 — `option_id_supervised` (heuristic) | E2 — `option_id_supervised` (VLM) |
| `soda_unsupervised.yaml` | E3 — `option_id_unsupervised` (LOVE) | E4 — `option_id_unsupervised` (LOVE) |
| `baseline_vanilla.yaml` | Frozen Diffusion Policy baseline | Frozen DP baseline |

## Example

```bash
python soda/training/train_low.py --config-path configs/pusht --config-name soda_supervised
```

Or via [`scripts/train_soda.sh`](../scripts/train_soda.sh):

```bash
./scripts/train_soda.sh task=pusht discovery=supervised
```
