"""
Local entrypoint — run LOVE labeling on Modal (E3 unsupervised, Push-T).

Reads experiments/love_pusht/best.ckpt from the soda-experiments volume,
runs the trained LOVE model over data/raw/pusht/pusht.zarr (bundled in the
image), writes data/option_id_unsupervised as a .npy back to the volume.

  modal run modal/modal_label_love.py

After it finishes:
  modal volume get soda-experiments love_pusht/option_id_unsupervised.npy ./experiments/love_pusht/
  python scripts/apply_love_labels.py
"""

from modal_config import app, label_love


@app.local_entrypoint()
def main():
    label_love.remote()
