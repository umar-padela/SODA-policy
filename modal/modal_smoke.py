"""
Local entrypoint for Modal infrastructure smoke test (§7 row 5).

  modal run modal/modal_smoke.py

Verifies GPU, torch, zarr, pusht.zarr, W&B (soda-debug), and Volume write.
Does not run training — use modal_train_low.py for that.
"""

from modal_config import app, smoke, spawn_modal_function


@app.local_entrypoint()
def main():
    result = spawn_modal_function(smoke, label="smoke")
    print("Smoke finished — see Modal logs and wandb project soda-debug.")
    print(result)

