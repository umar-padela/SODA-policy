"""
Confusion matrix + per-class accuracy for a trained π_high checkpoint.

Runs on Modal (checkpoint and DP checkpoint both live on the Volume).

Usage:
    modal run scripts/analyze_high_policy.py \\
        --checkpoint /experiments/pusht/train_high_labelsmoothing_dropout_imagenet/best.ckpt

    # or latest:
    modal run scripts/analyze_high_policy.py \\
        --checkpoint /experiments/pusht/train_high_labelsmoothing_dropout_imagenet/latest.ckpt

    # also analyse train split to see if overfitting pattern differs:
    modal run scripts/analyze_high_policy.py \\
        --checkpoint /experiments/pusht/train_high_labelsmoothing_dropout_imagenet/best.ckpt \\
        --split train
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modal"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "diffusion_policy"))

from modal_config import EXPERIMENTS_MOUNT, app, image, volume


@app.function(
    image=image,
    gpu=None,
    timeout=600,
    volumes={EXPERIMENTS_MOUNT: volume},
)
def run_confusion_matrix(checkpoint: str, split: str = "val") -> dict:
    import dill
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader

    from soda.eval.policy_loaders import load_high_policy_from_checkpoint
    from soda.training.train_high import build_datasets

    print(f"Loading π_high from {checkpoint} ...")
    policy = load_high_policy_from_checkpoint(checkpoint, device="cpu")
    policy.eval()

    # Rebuild val/train dataset using the config saved in the checkpoint.
    payload = torch.load(checkpoint, pickle_module=dill, map_location="cpu")
    saved_cfg = payload.get("cfg")
    if saved_cfg is None:
        raise ValueError("Checkpoint has no saved cfg — cannot rebuild dataset.")
    cfg = OmegaConf.create(saved_cfg) if isinstance(saved_cfg, dict) else saved_cfg

    train_ds, val_ds = build_datasets(cfg)
    ds = val_ds if split == "val" else train_ds
    print(f"Dataset split={split}: {len(ds)} samples")

    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    num_options = policy.cfg.num_options
    confusion = np.zeros((num_options, num_options), dtype=int)

    with torch.no_grad():
        for batch in loader:
            labels = batch["option_id"].reshape(-1).long()
            feat = policy.encode_obs(batch["obs"])
            preds = policy.sample_option(feat)
            for true, pred in zip(labels.numpy(), preds.cpu().numpy()):
                confusion[true, pred] += 1

    # ── Print confusion matrix ───────────────────────────────────────────────
    print(f"\nConfusion matrix  ({split})  — rows=true label, cols=predicted")
    print(f"{'':>10}", end="")
    for j in range(num_options):
        print(f"  pred_{j:>2}", end="")
    print("   | recall  (n)")

    for i in range(num_options):
        print(f"  true_{i:>3} ", end="")
        for j in range(num_options):
            marker = "*" if i == j else " "
            print(f" {confusion[i, j]:>5}{marker}", end="")
        n = int(confusion[i].sum())
        recall = confusion[i, i] / n if n > 0 else 0.0
        print(f"   | {recall:.1%}  ({n})")

    # ── Per-class precision ──────────────────────────────────────────────────
    print("\nPer-class precision (of all predicted as class X, how many were correct):")
    for j in range(num_options):
        n_pred = int(confusion[:, j].sum())
        prec = confusion[j, j] / n_pred if n_pred > 0 else 0.0
        print(f"  option_{j}: {prec:.1%}  ({confusion[j,j]}/{n_pred} predicted as {j})")

    # ── Overall ──────────────────────────────────────────────────────────────
    total_correct = int(np.diag(confusion).sum())
    total = int(confusion.sum())
    overall = total_correct / total if total > 0 else 0.0
    print(f"\nOverall {split} accuracy: {overall:.1%}  ({total_correct}/{total})")

    # ── Most common errors ───────────────────────────────────────────────────
    print("\nTop misclassifications (true → predicted, count):")
    errors = []
    for i in range(num_options):
        for j in range(num_options):
            if i != j and confusion[i, j] > 0:
                errors.append((confusion[i, j], i, j))
    for count, true, pred in sorted(errors, reverse=True):
        print(f"  option_{true} → option_{pred}: {count}")

    result = {
        "confusion": confusion.tolist(),
        "overall_acc": overall,
        "split": split,
    }
    return result


@app.local_entrypoint()
def main(
    checkpoint: str,
    split: str = "val",
) -> None:
    run_confusion_matrix.remote(checkpoint, split)
