"""
One-time download of BID's weak Push-T checkpoint to the Modal Volume.

Only the weak policy (epoch_0050, score=0.250) is needed as a reference for
forward contrast in BID. The strong policy is our existing Columbia DP best.ckpt.

Usage:
  modal run modal/modal_download_bid.py
"""

from modal_config import app, download_bid_checkpoints, EXPERIMENTS_MOUNT


@app.local_entrypoint()
def main() -> None:
    path = download_bid_checkpoints.remote()
    print(f"BID checkpoints ready at: {path}")
    print(
        f"Weak policy: {EXPERIMENTS_MOUNT}/bid_checkpoints/pusht/epoch_0050.ckpt\n"
        "Pass this as --bid-weak-checkpoint in modal_eval_bid_comparison.py"
    )
