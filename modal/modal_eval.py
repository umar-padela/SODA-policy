"""Local entrypoint for eval on Modal (stub — wire to soda/eval/run_eval.py later)."""

from modal_config import app

# TODO: eval_run.remote(...) when run_eval.py exists


@app.local_entrypoint()
def main():
    raise NotImplementedError("Use modal_smoke.py for infra check; eval not implemented yet.")
