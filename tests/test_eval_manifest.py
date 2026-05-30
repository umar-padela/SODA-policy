"""Tests for eval run manifest and output directory layout."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from soda.eval.eval_manifest import (
    build_eval_output_dir,
    cli_overrides_dict,
    config_stem_from_path,
    rename_eval_videos,
    run_dir_name_from_path,
    write_eval_run_manifest,
)
from soda.eval.eval_yaml import EvalCliOverrides, EvalConfig


def test_config_stem_from_path():
    assert config_stem_from_path(Path("configs/pusht/dp_frozen.yaml")) == "dp_frozen"
    assert config_stem_from_path(Path("configs/pusht/soda_supervised.yaml")) == "soda_supervised"


def test_build_eval_output_dir_layout():
    ts = datetime(2026, 5, 25, 1, 48, 51, tzinfo=timezone.utc)
    out, returned_ts = build_eval_output_dir(
        Path("/experiments"),
        config_path=Path("configs/pusht/dp_frozen.yaml"),
        timestamp=ts,
    )
    assert returned_ts == ts
    assert out == Path("/experiments/pusht/eval/dp_frozen/20260525/014851")


def test_run_dir_name_from_path():
    root = Path("/experiments")
    out = root / "pusht" / "eval" / "dp_frozen" / "20260525" / "014851"
    assert run_dir_name_from_path(out, root) == "pusht/dp_frozen/20260525/014851"


def test_cli_overrides_dict_skips_defaults():
    cli = EvalCliOverrides(n_test=1, full=False, record_video=True)
    assert cli_overrides_dict(cli) == {"n_test": 1}


def test_write_eval_run_manifest(tmp_path: Path):
    config_src = tmp_path / "dp_frozen.yaml"
    config_src.write_text("name: dp_frozen\n", encoding="utf-8")
    cfg = EvalConfig(config_path=config_src, config_name="dp_frozen", n_test=1)
    ts = datetime(2026, 5, 25, 1, 48, 51, tzinfo=timezone.utc)

    write_eval_run_manifest(
        tmp_path / "run",
        config=cfg,
        invoke_command="modal run modal/modal_eval.py --config dp_frozen.yaml --n-test 1",
        cli=EvalCliOverrides(n_test=1),
        timestamp=ts,
        run_readme="Frozen DP manifest smoke test",
    )

    run_dir = tmp_path / "run"
    assert (run_dir / "README.md").read_text(encoding="utf-8").find("Frozen DP manifest") >= 0
    assert (run_dir / "config.yaml").read_text(encoding="utf-8") == "name: dp_frozen\n"
    assert (run_dir / "command.txt").read_text(encoding="utf-8").startswith("modal run")
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["config_stem"] == "dp_frozen"
    assert manifest["cli_overrides"] == {"n_test": 1}
    assert manifest["resolved_eval"]["n_test"] == 1


def test_rename_eval_videos(tmp_path: Path):
    media = tmp_path / "media"
    media.mkdir()
    old = media / "ja93itxr.mp4"
    old.write_bytes(b"fake")

    runner_log = {
        "test/sim_max_reward_10000": 0.55,
        "test/sim_video_10000": str(old),
        "train/sim_video_0": str(old),
    }
    renamed = rename_eval_videos(
        tmp_path,
        config_stem="dp_frozen",
        runner_log=runner_log,
        prefix_filter="test/",
    )
    assert len(renamed) == 1
    expected = media / "dp_frozen_ep00_seed10000_score055.mp4"
    assert expected.is_file()
    assert renamed[0] == str(expected)
    assert runner_log["test/sim_video_10000"] == str(expected)
