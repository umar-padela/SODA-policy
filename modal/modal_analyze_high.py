"""
Confusion matrix + per-class accuracy for a trained π_high checkpoint.

Usage:
    modal run modal/modal_analyze_high.py \\
        --checkpoint /experiments/pusht/train_high_labelsmoothing_dropout_imagenet/best.ckpt

    # train split (compare overfitting pattern):
    modal run modal/modal_analyze_high.py \\
        --checkpoint /experiments/pusht/train_high_labelsmoothing_dropout_imagenet/best.ckpt \\
        --split train
"""

from modal_config import analyze_high, app


@app.local_entrypoint()
def main(checkpoint: str, split: str = "val") -> None:
    analyze_high.remote(checkpoint, split)
