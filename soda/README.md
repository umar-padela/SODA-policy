# `soda` package

Python implementation of **SODA** (Supervised Option Discovery for Dynamic Action chunking): hierarchical IL with option discovery and variable-horizon action chunks.

Full file tree and design: [`project_plan.md`](../project_plan.md) §3–§4.

Option labeling (offline): [`option_discovery/README.md`](option_discovery/README.md).

## Modules

| Folder | Role |
|--------|------|
| `dataset/` | PyTorch `Dataset` loaders (read zarr under repo `data/raw/`) |
| `models/` | `LowPolicy`, `HighPolicy`, `TerminationHead` |
| `training/` | `train_low.py`, `train_high.py`, `losses.py` |
| `inference/` | `HierarchicalController`, control regimes (open / closed / receding) |
| `eval/` | Metrics and eval driver |
| `option_discovery/supervised/pusht/` | Push-T heuristic → `option_id_supervised` |
| `option_discovery/supervised/square/` | Square VLM placeholder (E2) |
| `option_discovery/unsupervised/love_adapter/` | LOVE → `option_id_unsupervised` (E3/E4) |

## Training (not implemented yet)

```bash
# Example (once implemented)
python soda/training/train_low.py --config-path configs/pusht --config-name soda_supervised
python soda/training/train_high.py --config-path configs/pusht --config-name soda_supervised
```

Configs: [`configs/README.md`](../configs/README.md).
