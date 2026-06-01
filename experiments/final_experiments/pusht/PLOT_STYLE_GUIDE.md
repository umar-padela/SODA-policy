# Plot Style Guide — SODA Final Experiments (Push-T)

## General Rules

- **No error shaded bands** on score-vs-epoch curves (single training run; band would misrepresent training variance). Error bars belong only in bar/comparison charts.
- **y-axis top** is always **100** across all plots.
- **y-axis bottom** is study-specific — do not force to 0 (use `ylim` param).
- **"ep" → "epch"** in all epoch labels (e.g. `epch450`, not `ep450`).
- **"Chunk Length" → "Chunk Duration"** in all baseline labels.
- Unicode arrows (`→`) crash on Windows cp1252 — use `->` in all `print()` calls.

## y-axis Ranges by Study

| Study | ylim |
|-------|------|
| Termination study (all stages) | `(65, 100)` — data min ≈ 66.5% |
| Kernel size study (k5/k7/k9 curves + grid) | `(88, 100)` — data min ≈ 89.3% |
| Receding horizon study | `(0, 100)` (default) |
| Comparison study | `(0, 100)` (default) |

## Termination Study Visual Encoding

Defined in `shared/experiment_utils.py` via `BOTTLENECK_COLOR`, `OBS_COLOR`, `BOTH_COLOR`, `term_run_style()`.

| Dimension | Encoding |
|-----------|----------|
| **Color** | Blue (`#1f77b4`) = bottleneck features; Orange (`#ff7f0e`) = obs features; Green (`#2ca02c`) = both combined |
| **Marker shape** | Circle `o` = bottleneck; Square `s` = obs; Diamond `D` = both |
| **Line style** | Solid = completion signal only; Dashed = completion + escape signal |
| **Marker fill** | Filled = expert action source; Hollow (outline only) = DDIM-5 generated actions |

This encoding is **consistent across all stages** so plots can be placed side by side.

### Run → Style Mapping

| Run key | Color | Linestyle | Marker | Filled |
|---------|-------|-----------|--------|--------|
| `bottleneck_expert[_joint]` | blue | solid | `o` | yes |
| `bottleneck_ddim_positive[_joint]` | blue | solid | `o` | no |
| `bottleneck_ddim_positive_negative[_joint]` | blue | dashed | `o` | no |
| `obs_positive[_joint]` | orange | solid | `s` | yes |
| `obs_positive_negative[_joint]` | orange | dashed | `s` | yes |
| `both[_joint]` | green | solid | `D` | yes |

## Score-vs-Epoch Curve: Fine-tuning Origin Point

All termination study runs fine-tune from the `Chunk Duration (k=9,epch450)` checkpoint (score = **97.97%**). Score-vs-epoch plots:
- **Curves start at epoch 50** (their first real data point) — do NOT extend back to epoch 0, as that would falsely imply the termination methods match chunk-duration performance at initialization
- **Gray dashed hline** at 97.97% labeled `Chunk Duration (k=9,epch450)` is kept for reference
- **x-axis starts at 0** (`xlim_left=0`) so the baseline hline has visual context, even though curves start at epoch 50
- Legend: **outside bottom-left**, title **"Termination Method"**, `ncol=2` for plots with 5+ runs

## Baseline Label

```
Chunk Duration (k=9,epch450)   # score = 97.97%
```

## Download + Plot All Studies

```bash
python experiments/final_experiments/pusht/download_and_plot_all.py
```

This script runs `--download-only` on each modal eval script (which also re-runs the local plot script) and handles the comparison study manually via `modal volume get`.
