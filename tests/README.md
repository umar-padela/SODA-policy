# Tests

From repo root (with `conda activate soda`):

```bash
pip install pytest   # once, if not already installed
python -m pytest tests/ -v
```

Only the Push-T zarr integration tests need `data/raw/pusht/pusht.zarr/`; other tests use synthetic data.

Useful groups:

```bash
# Labeling / balancing helpers
python -m pytest tests/test_pusht_heuristics.py tests/test_option_balance.py tests/test_losses_low.py -q

# Datasets + training scaffolds
python -m pytest tests/test_option_aware_dataset.py tests/test_option_start_dataset.py \
  tests/test_train_low_scaffold.py tests/test_high_policy.py -q
```
