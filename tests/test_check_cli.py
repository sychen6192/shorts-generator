"""Checker CLI contract tests: exit codes, verdict vocabulary, fail-closed paths."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from conftest import write_thresholds

from shortsloop.verdict import validate


def run_check(*args: str) -> tuple[int, dict, str]:
    """Run the checker as a real subprocess; parse the trailing VERDICT line."""
    res = subprocess.run(
        [sys.executable, "-m", "shortsloop.check", *args],
        capture_output=True, text=True, timeout=300,
    )
    verdict_lines = [ln for ln in res.stdout.splitlines() if ln.startswith("VERDICT ")]
    summary = json.loads(verdict_lines[-1][len("VERDICT "):]) if verdict_lines else {}
    return res.returncode, summary, res.stdout + res.stderr


def load_verdict(path: Path) -> dict:
    v = json.loads(path.read_text(encoding="utf-8"))
    assert validate(v) == [], f"verdict violates invariants: {validate(v)}"
    return v


def test_l1_only_proceed(clips, thresholds_permissive, prompt_file, tmp_path):
    out = tmp_path / "v.json"
    code, summary, _ = run_check(
        str(clips["moving"]), "--prompt-file", str(prompt_file),
        "--l1-only", "--thresholds", str(thresholds_permissive), "--json", str(out))
    assert code == 0
    assert summary["verdict"] == "PROCEED"          # never PASS from --l1-only
    v = load_verdict(out)
    assert v["verdict"] == "PROCEED"
    assert v["layers_run"] == ["l1"]
    assert v["l2"] is None
    assert v["l1"]["pass"] is True
    assert v["clip"]["sha256"] and v["prompt"]["sha256"]
    assert v["thresholds"]["calibrated"] is False


def test_l1_only_fails_static_with_class(clips, prompt_file, tmp_path):
    thr = write_thresholds(tmp_path / "t.yaml", motion=0.0015)
    out = tmp_path / "v.json"
    code, summary, _ = run_check(
        str(clips["static"]), "--prompt-file", str(prompt_file),
        "--l1-only", "--thresholds", str(thr), "--json", str(out))
    assert code == 1
    assert summary["verdict"] == "FAIL"
    v = load_verdict(out)
    assert v["failure_classes"] == ["static"]
    failed = [c["name"] for c in v["l1"]["checks"] if not c["pass"]]
    assert "motion" in failed


def test_corrupt_clip_is_clip_scope_error(clips, thresholds_permissive, prompt_file, tmp_path):
    out = tmp_path / "v.json"
    code, summary, _ = run_check(
        str(clips["corrupt"]), "--prompt-file", str(prompt_file),
        "--l1-only", "--thresholds", str(thresholds_permissive), "--json", str(out))
    assert code == 2
    assert summary["verdict"] == "ERROR"
    v = load_verdict(out)
    assert v["error"]["scope"] == "clip"
    assert v["error"]["stage"] == "probe"


def test_missing_thresholds_is_infra_error(clips, prompt_file, tmp_path):
    code, summary, _ = run_check(
        str(clips["moving"]), "--prompt-file", str(prompt_file),
        "--l1-only", "--thresholds", str(tmp_path / "nope.yaml"))
    assert code == 2
    assert summary["error"]["scope"] == "infra"


def test_full_mode_without_judge_is_infra_error(clips, thresholds_permissive, prompt_file, tmp_path):
    """M1: full mode must fail closed (never PASS) while no judge is configured."""
    out = tmp_path / "v.json"
    code, summary, _ = run_check(
        str(clips["moving"]), "--prompt-file", str(prompt_file),
        "--thresholds", str(thresholds_permissive), "--json", str(out),
        "--config", str(tmp_path / "no-config.yaml"))
    assert code == 2
    v = load_verdict(out)
    assert v["verdict"] == "ERROR"
    assert v["error"]["scope"] == "infra"
    assert "judge" in v["error"]["message"].lower()


def test_full_mode_l1_fail_short_circuits_judge(clips, prompt_file, tmp_path):
    """L1 FAIL in full mode: verdict FAIL, l2 null — no judge call attempted."""
    thr = write_thresholds(tmp_path / "t.yaml", motion=0.0015)
    out = tmp_path / "v.json"
    code, summary, _ = run_check(
        str(clips["static"]), "--prompt-file", str(prompt_file),
        "--thresholds", str(thr), "--json", str(out))
    assert code == 1
    v = load_verdict(out)
    assert v["verdict"] == "FAIL"
    assert v["l2"] is None
    assert v["layers_run"] == ["l1"]


def test_expect_mismatch_fails_spec(clips, thresholds_permissive, prompt_file, tmp_path):
    out = tmp_path / "v.json"
    code, _, _ = run_check(
        str(clips["moving"]), "--prompt-file", str(prompt_file),
        "--l1-only", "--thresholds", str(thresholds_permissive), "--json", str(out),
        "--expect", '{"width": 720, "height": 1280}')
    assert code == 1
    v = load_verdict(out)
    assert v["failure_classes"] == ["broken"]
    spec = [c for c in v["l1"]["checks"] if c["name"] == "spec"][0]
    assert not spec["pass"]
