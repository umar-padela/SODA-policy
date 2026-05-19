# Tests

From repo root (with `conda activate soda`):

```bash
pip install pytest   # once, if not already installed
python -m pytest tests/ -v
```

Only the Push-T zarr integration test needs `data/raw/pusht/pusht.zarr/`; other tests use synthetic data.
