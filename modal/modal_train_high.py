"""
Local entrypoint — train pi_high on Modal.

  modal run modal/modal_train_high.py
  modal run modal/modal_train_high.py --config-name soda_supervised --task pusht
"""

from modal_config import app, train_high


@app.local_entrypoint()
def main(config_name: str = "soda_supervised", task: str = "pusht"):
    train_high.remote(config_name=config_name, task=task)
