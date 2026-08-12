"""Full-mode checker CLI: PASS end-to-end against a fake judge; fail-closed rows 1-2."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from fake_judge import GOOD_SCORES, serve
from test_check_cli import load_verdict, run_check

from shortsloop.verdict import sha256_text


def write_config(tmp_path: Path, base_url: str) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump({"judge": {
        "adapter": "ollama", "base_url": base_url,
        "model": "qwen3-vl:8b-instruct", "timeout_s": 30,
    }}), encoding="utf-8")
    return p


def test_full_mode_pass(clips, thresholds_permissive, prompt_file, tmp_path):
    out = tmp_path / "v.json"
    with serve() as (_, url):
        code, summary, _ = run_check(
            str(clips["moving"]), "--prompt-file", str(prompt_file),
            "--thresholds", str(thresholds_permissive),
            "--config", str(write_config(tmp_path, url)), "--json", str(out))
    assert code == 0
    assert summary["verdict"] == "PASS"
    v = load_verdict(out)
    assert v["layers_run"] == ["l1", "l2"]
    assert v["l1"]["pass"] and v["l2"]["pass"]
    assert v["timing"]["l2_s"] is not None
    # raw judge output persisted next to the verdict, sha matches
    raw_path = out.with_suffix(".l2_raw.json")
    raw = raw_path.read_text(encoding="utf-8")
    assert json.loads(raw) == GOOD_SCORES
    assert v["l2"]["raw_response_sha256"] == sha256_text(raw)


def test_full_mode_fail_off_prompt(clips, thresholds_permissive, prompt_file, tmp_path):
    scores = {k: dict(v) for k, v in GOOD_SCORES.items()}
    scores["prompt_adherence"] = {"score": 2, "na": False,
                                  "reason": "no hummingbird visible in any frame"}
    out = tmp_path / "v.json"
    with serve(scores=scores) as (_, url):
        code, summary, _ = run_check(
            str(clips["moving"]), "--prompt-file", str(prompt_file),
            "--thresholds", str(thresholds_permissive),
            "--config", str(write_config(tmp_path, url)), "--json", str(out))
    assert code == 1
    v = load_verdict(out)
    assert v["verdict"] == "FAIL"
    assert v["failure_classes"] == ["off_prompt"]
    assert v["l2"]["dimensions"]["prompt_adherence"]["pass"] is False


def test_full_mode_judge_unreachable_is_infra(clips, thresholds_permissive, prompt_file, tmp_path):
    out = tmp_path / "v.json"
    code, summary, _ = run_check(
        str(clips["moving"]), "--prompt-file", str(prompt_file),
        "--thresholds", str(thresholds_permissive),
        "--config", str(write_config(tmp_path, "http://127.0.0.1:9")),
        "--json", str(out))
    assert code == 2
    v = load_verdict(out)
    assert v["verdict"] == "ERROR"
    assert v["error"]["scope"] == "infra"
    assert v["l2"] is None                      # no partial l2 ever


def test_full_mode_garbage_judge_is_clip_error(clips, thresholds_permissive, prompt_file, tmp_path):
    out = tmp_path / "v.json"
    with serve(scenario="garbage") as (srv, url):
        code, summary, _ = run_check(
            str(clips["moving"]), "--prompt-file", str(prompt_file),
            "--thresholds", str(thresholds_permissive),
            "--config", str(write_config(tmp_path, url)), "--json", str(out))
        assert srv.chat_calls == 2              # retried exactly once
    assert code == 2
    v = load_verdict(out)
    assert v["error"]["scope"] == "clip"
    assert v["error"]["stage"] == "l2"
