"""Checker CLI contract tests: exit codes, verdict vocabulary, fail-closed paths."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import write_thresholds

from shortsloop.schema import validate_verdict_file


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
    """Every verdict any test touches must satisfy the frozen JSON Schema AND the
    cross-field invariants (plan §5 row 17)."""
    v = json.loads(path.read_text(encoding="utf-8"))
    problems = validate_verdict_file(v)
    assert problems == [], f"verdict violates schema/invariants: {problems}"
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


def test_stale_verdict_file_is_never_reused(clips, tmp_path):
    """Runner-side: a verdict file left from an earlier invocation must not be read
    as the verdict of a checker that crashed before writing (hard rule 2)."""
    import sys as _sys
    from shortsloop.runner import Runner, RunHalted
    out = tmp_path / "v.json"
    out.write_text(json.dumps({"verdict": "PROCEED", "layers_run": ["l1"]}))
    r = Runner(dispatch_path="x", config_path="x", pipeline_path="x",
               thresholds_path=tmp_path / "t.yaml",
               checker_argv=[_sys.executable, "-c", "import sys; sys.exit(1)"])
    r.expect = {}
    r.config_path = tmp_path / "c.yaml"
    with pytest.raises(RunHalted):
        r._invoke_checker(clips["moving"], tmp_path / "p.txt", out, l1_only=True)


# ------------------------------------------------------------------ emit hardening

def test_checker_crash_outside_evaluation_is_error_exit_2(clips, thresholds_permissive,
                                                          prompt_file, tmp_path):
    """Exit 1 means FAIL; a checker that could not even write its verdict must say
    ERROR (exit 2), never FAIL or PASS."""
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x")
    code, summary, _ = run_check(
        str(clips["moving"]), "--prompt-file", str(prompt_file), "--l1-only",
        "--thresholds", str(thresholds_permissive),
        "--json", str(blocker / "v.json"))
    assert code == 2
    assert summary.get("verdict") == "ERROR"


def _inproc(monkeypatch, tmp_path, clip, thr, prompt, *extra):
    from shortsloop import check
    out = tmp_path / "v.json"
    code = check.main([str(clip), "--prompt-file", str(prompt), "--json", str(out),
                       "--thresholds", str(thr), *extra])
    return code, out


def test_off_schema_verdict_is_downgraded_to_infra_error(clips, thresholds_permissive,
                                                         prompt_file, tmp_path,
                                                         monkeypatch):
    from shortsloop import l1 as l1mod
    real = l1mod.compute_metrics

    def leaky(*a, **kw):
        m = real(*a, **kw)
        m["metrics"]["undeclared_metric"] = 1.0          # off the frozen schema
        return m
    monkeypatch.setattr(l1mod, "compute_metrics", leaky)
    code, out = _inproc(monkeypatch, tmp_path, clips["moving"], thresholds_permissive,
                        prompt_file, "--l1-only")
    assert code == 2
    v = load_verdict(out)                    # the downgraded verdict itself validates
    assert v["verdict"] == "ERROR" and v["error"]["scope"] == "infra"


def test_non_finite_metric_never_leaks_into_json(clips, thresholds_permissive,
                                                 prompt_file, tmp_path, monkeypatch):
    from shortsloop import l1 as l1mod
    real = l1mod.compute_metrics

    def nan(*a, **kw):
        m = real(*a, **kw)
        m["metrics"]["ssim_min"] = float("nan")
        return m
    monkeypatch.setattr(l1mod, "compute_metrics", nan)
    code, out = _inproc(monkeypatch, tmp_path, clips["moving"], thresholds_permissive,
                        prompt_file, "--l1-only")
    assert code == 2
    text = out.read_text()
    assert "NaN" not in text
    json.loads(text, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    assert load_verdict(out)["verdict"] == "ERROR"


def test_unexpected_l2_crash_reports_stage_l2(clips, thresholds_permissive, prompt_file,
                                             tmp_path, monkeypatch):
    from shortsloop import l2 as l2mod

    def boom(*a, **kw):
        raise KeyError("floors")
    monkeypatch.setattr(l2mod, "run_l2", boom)
    cfg = tmp_path / "config.yaml"
    cfg.write_text("judge: {adapter: ollama, base_url: 'http://127.0.0.1:9', model: m}\n")
    code, out = _inproc(monkeypatch, tmp_path, clips["moving"], thresholds_permissive,
                        prompt_file, "--config", str(cfg))
    assert code == 2
    v = load_verdict(out)
    assert v["error"]["stage"] == "l2" and v["error"]["scope"] == "infra"
