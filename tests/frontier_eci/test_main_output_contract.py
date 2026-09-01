import fnmatch
import json
import os
from pathlib import Path
from runpy import run_path
import shutil
import subprocess

import pandas as pd
import pytest


ROOT = Path(__file__).parents[2]
RESULT_ROOT = ROOT / "judge_results/frontier/eci-semiannual-canonical-anchors"
MANIFEST = RESULT_ROOT / "MAIN_OUTPUT.json"
LAUNCHER = ROOT / "scripts/frontier_eci/runpod/run_canonical_analysis.sh"
MODULE = run_path(ROOT / "scripts/frontier_eci/rank_regression.py")


def test_official_launcher_executes_main_output_script_and_forwards_arguments(tmp_path):
    capture = tmp_path / "arguments.txt"
    fake_uv = tmp_path / "uv"
    fake_uv.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$CAPTURE\"\n")
    fake_uv.chmod(0o755)
    environment = os.environ | {"PATH": f"{tmp_path}:{os.environ['PATH']}",
                                "CAPTURE": str(capture)}
    subprocess.run(["bash", str(LAUNCHER), "--no-figures"], cwd=ROOT,
                   env=environment, check=True)
    arguments = capture.read_text().splitlines()
    main = json.loads(MANIFEST.read_text())["main_analysis"]
    assert arguments == ["run", "python", f"scripts/frontier_eci/{Path(main).name}.py",
                         "--no-figures"]


def test_manifest_classifies_every_legacy_analysis_artifact_as_discarded():
    manifest = json.loads(MANIFEST.read_text())
    rules = manifest["discarded_provenance"]

    def is_discarded(path):
        for rule in rules:
            base = (RESULT_ROOT / rule["root"]).resolve()
            try:
                relative = path.resolve().relative_to(base).as_posix()
            except ValueError:
                continue
            if fnmatch.fnmatch(relative, rule["glob"]):
                if not any(fnmatch.fnmatch(relative, exception)
                           for exception in rule.get("except", [])):
                    return True
        return False

    main = (RESULT_ROOT / manifest["main_analysis"]).resolve()
    canonical_legacy = [path for root in (RESULT_ROOT / "analysis",
                                          RESULT_ROOT / "capability_baseline")
                        for path in root.rglob("*") if path.is_file()
                        and not path.resolve().is_relative_to(main)]
    old_root = ROOT / "judge_results/frontier/eci-semiannual"
    older_legacy = [path for root in old_root.glob("analysis*") if root.is_dir()
                    for path in root.rglob("*") if path.is_file()]
    assert canonical_legacy and older_legacy
    assert all(is_discarded(path) for path in canonical_legacy + older_legacy)
    assert all(not is_discarded(path) for path in main.rglob("*") if path.is_file())


def test_incomplete_staging_run_preserves_current_main_output(tmp_path):
    current = tmp_path / "rank_regression"
    current.mkdir()
    (current / "sentinel.txt").write_text("valid current output")
    staging = tmp_path / ".rank_regression.staging"
    staging.mkdir()
    (staging / "abilities.csv").write_text("incomplete")
    with pytest.raises(ValueError, match="incomplete staged rank-regression output"):
        MODULE["publish_staged_output"](staging, current, require_figures=False)
    assert (current / "sentinel.txt").read_text() == "valid current output"
    assert current.is_dir()


def test_invalid_complete_staging_run_preserves_current_main_output(tmp_path):
    current = tmp_path / "rank_regression"
    current.mkdir()
    (current / "sentinel.txt").write_text("valid current output")
    staging = tmp_path / ".rank_regression.staging"
    staging.mkdir()
    for name in MODULE["REQUIRED_OUTPUTS"]:
        shutil.copy2(RESULT_ROOT / "analysis/rank_regression" / name, staging / name)
    frontier = pd.read_csv(staging / "frontier_fitted.csv")
    frontier.loc[0, "actual"] = None
    frontier.to_csv(staging / "frontier_fitted.csv", index=False)
    with pytest.raises(ValueError, match="invalid main-analysis invariants"):
        MODULE["publish_staged_output"](staging, current, require_figures=False)
    assert (current / "sentinel.txt").read_text() == "valid current output"
