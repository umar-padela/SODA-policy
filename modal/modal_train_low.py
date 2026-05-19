"""
Local entrypoint — train pi_low on Modal.

  modal run modal/modal_train_low.py
  modal run modal/modal_train_low.py --config-name soda_supervised --task pusht

Infrastructure check: modal run modal/modal_smoke.py
"""

from modal_config import app, train_low


@app.local_entrypoint()
def main(config_name: str = "soda_supervised", task: str = "pusht"):
    train_low.remote(config_name=config_name, task=task)
