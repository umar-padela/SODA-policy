# `soda/option_discovery/unsupervised`

Offline **unsupervised** option discovery via LOVE → `option_id_unsupervised` in task zarr (E3 Push-T, E4 Square).

| Component | Path |
|-----------|------|
| LOVE adapter (WIP) | [`love_adapter/`](love_adapter/) |

After LOVE runs, relabel or rebuild zarr so `data/.../option_id_unsupervised` aligns with demo frames. Use `option_id_unsupervised` in `configs/pusht/soda_unsupervised.yaml`.

Upstream submodule: `third_party/love` (see [`project_plan.md`](../../../project_plan.md) §7 rows 13–14, 33).

See parent [`../README.md`](../README.md).
