# `soda/option_discovery/supervised`

Offline **supervised** option labeling: heuristics or VLM → `option_id_supervised` in task zarr.

| Task | Folder | Status |
|------|--------|--------|
| Push-T (E1) | [`pusht/`](pusht/README.md) | Heuristic pipeline + zarr build |
| Square (E2) | [`square/`](square/README.md) | Placeholder (VLM labels TBD) |

Push-T is the reference implementation. Training configs point at `option_id_supervised` via `task.dataset.option_id_key`. E1 labels are **3 contiguous skills** (0–2); see [`pusht/README.md`](pusht/README.md).

See parent [`../README.md`](../README.md) for commands and zarr layout.
