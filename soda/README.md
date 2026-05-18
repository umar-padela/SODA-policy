# `soda` package

Python implementation of **SODA** (Supervised Option Discovery for Dynamic Action chunking): hierarchical IL with option discovery and variable-horizon action chunks.

Full file tree and design: [`project_plan.md`](../project_plan.md) §3–§4.

## Modules

| Folder | Role |
|--------|------|
| `dataset/` | PyTorch `Dataset` loaders (read zarr under repo `data/raw/`) |
| `models/` | `LowPolicy`, `HighPolicy`, `TerminationHead` |
| `training/` | `train_low.py`, `train_high.py`, `losses.py` |
| `inference/` | `HierarchicalController`, control regimes (open / closed / receding) |
| `eval/` | Metrics and eval driver |
| `option_discovery/love_adapter/` | LOVE → zarr `option_id` (E3/E4) |

## Training (not implemented yet)

```bash
# Example (once implemented)
python soda/training/train_low.py --config-path configs/pusht --config-name soda_supervised
python soda/training/train_high.py --config-path configs/pusht --config-name soda_supervised
```

Configs: [`configs/README.md`](../configs/README.md).
