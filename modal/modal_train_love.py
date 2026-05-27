"""
Local entrypoint — train LOVE (E3 unsupervised option discovery) on Modal.

  modal run modal/modal_train_love.py

Outputs land under /experiments/love_pusht/ on the soda-experiments volume
(best.ckpt + action_centroids.npy + config.json). Pull them locally with
`modal volume get soda-experiments love_pusht/` then run `label.py`.

Infrastructure check: modal run modal/modal_smoke.py
"""

from modal_config import app, train_love


@app.local_entrypoint()
def main():
    train_love.remote()
