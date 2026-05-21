"""
Download Columbia frozen DP Push-T checkpoint to the Modal Volume (one-time).

Does not run eval. After this, pass the printed path to ``modal_eval.py --checkpoint``.

  modal run modal/modal_download_dp.py
"""

from modal_config import FROZEN_DP_PUSHT_CHECKPOINT, app, download_frozen_dp


@app.local_entrypoint()
def main(dest: str = FROZEN_DP_PUSHT_CHECKPOINT):
    path = download_frozen_dp.remote(dest_path=dest)
    print(f"\nUse for smoke eval:\n")
    print(f"  modal run modal/modal_eval.py --checkpoint {path}")
