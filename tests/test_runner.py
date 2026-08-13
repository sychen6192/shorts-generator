"""Wave-runner integration tests against fake ComfyUI + fake judge (plan §5).

These run the REAL vendored client, REAL checker subprocesses, and REAL ffmpeg
encodes — only the GPU services are faked. Slow-ish by design: this is the
end-to-end dry run of the nightly loop.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from conftest import write_thresholds
from fake_comfy import serve_comfy
from fake_judge import GOOD_SCORES, serve as serve_judge

from shortsloop.policy import MOTION_PHRASES
from shortsloop.runner import Runner, ship_gate

DATA = Path(__file__).parent / "data"


def small_sheet(tmp_path: Path, n_clips: int = 1) -> Path:
    """Single-video sheet with n T2V clips (V1C1..V1Cn)."""
    rows = f"| V1C1–V1C{n_clips} | T2V | — | `v1/c.mp4` |" if n_clips > 1 else \
           "| V1C1 | T2V | — | `v1/c1.mp4` |"
    sections = []
    for k in range(1, n_clips + 1):
        sections.append(
            f"### V1C{k} — beat {k}(T2V)\n```\n"
            f"A red cube number {k} sliding across a dark slate table, slow dolly-in, "
            f"cinematic lighting, vertical 9:16 composition.\n```\n")
    md = f"""# 測試派工單

## 0. 共用參數

| 項目 | 值 |
|---|---|
| 解析度 | 480x832(draft) |
| length / fps | 49 frames(3.0 s)/ 16 fps |

## 生產清單(manifest)

| ID | 模式 | image 依賴 | 輸出 |
|---|---|---|---|
{rows}

## Video #1 — 測試

{''.join(sections)}
"""
    p = tmp_path / "dispatch.md"
    p.write_text(md, encoding="utf-8")
    return p


def make_env(tmp_path, comfy, judge_url, *, thresholds_over=None, policies_over=None,
             calibrated=True):
    """Write config/pipeline/thresholds pointing at the fakes; return paths dict."""
    thr = write_thresholds(tmp_path / "thresholds.yaml", calibrated=calibrated,
                           **(thresholds_over or {}))
    pol = {"policies": {
        "max_attempts_per_clip": 3,
        "wall_clock_budget_h": 6,
        "disk_min_free_gb": 1,
        "waves_max": 3,
        "comfy": {"timeout_s": 60, "poll_s": 1},
        "judge": {"timeout_s": 30, "retries": 1, "infra_escalation_after": 2},
        "vram_handoff": {"free_min_gb": 20, "wait_timeout_s": 5},
    }}
    for key, val in (policies_over or {}).items():
        if isinstance(val, dict):
            pol["policies"].setdefault(key, {}).update(val)
        else:
            pol["policies"][key] = val
    pipeline = tmp_path / "pipeline.yaml"
    pipeline.write_text(yaml.safe_dump(pol), encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({
        "comfy": {"host": comfy.host, "workflow_t2v": str(DATA / "test_workflow.json")},
        "judge": {"adapter": "ollama", "base_url": judge_url,
                  "model": "qwen3-vl:8b-instruct", "timeout_s": 30},
        "rewrite": {"enabled": True, "model": "fake-text-model"},
        "paths": {"runs_dir": str(tmp_path / "runs")},
    }), encoding="utf-8")
    # a passing doctor snapshot next to the config (the runner's doctor-gate)
    (tmp_path / "doctor.json").write_text(json.dumps({"ok": True, "checks": []}),
                                          encoding="utf-8")
    return {"config": config, "pipeline": pipeline, "thresholds": thr,
            "runs": tmp_path / "runs"}


def make_runner(dispatch, env, **kw):
    return Runner(dispatch_path=dispatch, config_path=env["config"],
                  pipeline_path=env["pipeline"], thresholds_path=env["thresholds"],
                  runs_dir=env["runs"], **kw)


def run_report(runner) -> dict:
    return json.loads((runner.run_dir / "report.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- happy path

def test_happy_path_end_to_end(clips, tmp_path):
    """3 accepted T2V clips + 1 loud I2V skip; all pass; encodes verified;
    one job at a time; /free precedes all judging (rules 3+4)."""
    with serve_comfy(fixture_paths=clips) as comfy, serve_judge() as (judge, jurl):
        env = make_env(tmp_path, comfy, jurl)
        runner = make_runner(DATA / "dispatch_small.md", env)
        code = runner.run()

        assert code == 0
        rep = run_report(runner)
        assert rep["run"]["status"] == "COMPLETED"
        assert {c["clip_id"]: c["status"] for c in rep["clips"]} == {
            "V1C1": "passed", "V1C2": "passed", "V2C1": "passed"}
        assert rep["skipped_at_intake"][0]["clip_id"] == "V2C2"

        # silent QC encodes exist for every PASS and nothing else
        encoded = sorted(p.name for p in (runner.run_dir / "encoded").glob("*.mp4"))
        assert encoded == ["V1C1.mp4", "V1C2.mp4", "V2C1.mp4"]

        # hard rule 4: never a second submission while one job unfinished
        assert comfy.violations == 0
        assert len(comfy.submissions) == 3
        # hard rule 3: VRAM freed (and verified) before ANY judge call
        assert comfy.free_calls >= 1
        assert judge.chat_calls == 3
        assert max(comfy.free_times) < min(judge.chat_times)
        # judge unloaded after the wave
        assert judge.generate_calls >= 1

        # evidence on disk: attempts log + verdicts + report.md
        attempts = [json.loads(l) for l in
                    (runner.run_dir / "attempts.jsonl").read_text().splitlines()]
        assert len(attempts) == 3 and all(a["verdict"] == "PASS" for a in attempts)
        assert all(a["seed"] is not None and a["output_sha256"] for a in attempts)
        report_md = (runner.run_dir / "report.md").read_text(encoding="utf-8")
        assert "V2C2" in report_md and "COMPLETED" in report_md


# ---------------------------------------------------------------- re-roll paths

def test_static_rerolls_motion_phrase_then_steps8_then_final_fail(clips, tmp_path):
    """L1-inline static path: 3 attempts in ONE wave, phrase on a2, steps 8 on a3,
    distinct seeds, judge never consulted, retry cap respected (rows 10, D8)."""
    with serve_comfy(fixture_paths=clips,
                     scenarios=[{"fixture": "static"}] * 3) as comfy, \
         serve_judge() as (judge, jurl):
        env = make_env(tmp_path, comfy, jurl,
                       thresholds_over={"motion": 0.0015})
        runner = make_runner(small_sheet(tmp_path), env)
        code = runner.run()

        assert code == 0                       # bounded, explained failure = success
        rep = run_report(runner)
        clip = rep["clips"][0]
        assert clip["status"] == "failed_final"
        assert clip["failure_classes"] == ["static"]
        assert len(clip["attempts"]) == 3
        assert rep["budget"]["waves_run"] == 1          # all inline, Wan stayed loaded
        assert rep["budget"]["cap_respected"] is True
        assert judge.chat_calls == 0                    # static never reaches the VLM

        j1, j2, j3 = (comfy.job_info(i) for i in range(3))
        assert MOTION_PHRASES[0].strip("; ") in j2["prompt"]
        assert MOTION_PHRASES[1].strip("; ") in j3["prompt"]
        assert j1["steps"] == 4 and j2["steps"] == 4 and j3["steps"] == 8
        assert len({j1["seed"], j2["seed"], j3["seed"]}) == 3
        assert (runner.run_dir / "encoded").exists()
        assert list((runner.run_dir / "encoded").glob("*.mp4")) == []


def test_flicker_reroll_bumps_steps_only(clips, tmp_path):
    with serve_comfy(fixture_paths=clips,
                     scenarios=[{"fixture": "flicker"}, {"fixture": "moving"}]) as comfy, \
         serve_judge() as (judge, jurl):
        env = make_env(tmp_path, comfy, jurl,
                       thresholds_over={"motion": 0.0015, "flicker": 2})
        runner = make_runner(small_sheet(tmp_path), env)
        assert runner.run() == 0
        rep = run_report(runner)
        assert rep["clips"][0]["status"] == "passed"
        j1, j2 = comfy.job_info(0), comfy.job_info(1)
        assert j2["steps"] == 8
        assert j2["prompt"] == j1["prompt"]            # no phrase for flicker
        a = rep["clips"][0]["attempts"]
        assert a[0]["failure_classes"] == ["flicker"]
        assert a[1]["status"] == "passed"


def test_off_prompt_rewrite_once_then_plain_reroll(clips, tmp_path):
    bad = {k: dict(v) for k, v in GOOD_SCORES.items()}
    bad["prompt_adherence"] = {"score": 2, "na": False, "reason": "no cube visible"}
    with serve_comfy(fixture_paths=clips) as comfy, \
         serve_judge(scores_queue=[bad, bad, GOOD_SCORES]) as (judge, jurl):
        env = make_env(tmp_path, comfy, jurl)
        runner = make_runner(small_sheet(tmp_path), env)
        assert runner.run() == 0
        rep = run_report(runner)
        clip = rep["clips"][0]
        assert clip["status"] == "passed"
        assert len(clip["attempts"]) == 3
        assert judge.rewrite_calls == 1                 # exactly ONE rewrite ever
        rewritten = judge.rewrite_response["rewritten_prompt"]
        assert comfy.job_info(1)["prompt"] == rewritten
        assert comfy.job_info(2)["prompt"] == rewritten  # a3 keeps it, new seed only
        assert comfy.job_info(1)["seed"] != comfy.job_info(2)["seed"]
        attempts = [json.loads(l) for l in
                    (runner.run_dir / "attempts.jsonl").read_text().splitlines()]
        assert attempts[1]["prompt_rewritten"] is True
        assert attempts[1]["prompt_diff"]               # logged old→new diff
        assert rep["budget"]["waves_run"] == 3          # each L2 fail rolls a wave


# ---------------------------------------------------------------- fail-closed

def test_doctor_gate_refuses_without_passing_snapshot(clips, tmp_path):
    with serve_comfy(fixture_paths=clips) as comfy, serve_judge() as (_, jurl):
        env = make_env(tmp_path, comfy, jurl)
        doctor_json = tmp_path / "doctor.json"

        doctor_json.unlink()                                # no snapshot at all
        assert make_runner(small_sheet(tmp_path), env).run() == 2
        assert comfy.submissions == []

        doctor_json.write_text(json.dumps({                 # failed snapshot
            "ok": False, "checks": [{"name": "judge.vision", "ok": False,
                                     "level": "FAIL", "detail": "blind"}]}))
        assert make_runner(small_sheet(tmp_path), env).run() == 2
        assert comfy.submissions == []

        # supervised override works; then a passing snapshot works
        assert make_runner(small_sheet(tmp_path), env, skip_doctor=True).run() == 0
        doctor_json.write_text(json.dumps({"ok": True, "checks": []}))
        assert make_runner(small_sheet(tmp_path), env).run() == 0


def test_uncalibrated_thresholds_refuse_unattended(clips, tmp_path):
    with serve_comfy(fixture_paths=clips) as comfy, serve_judge() as (_, jurl):
        env = make_env(tmp_path, comfy, jurl, calibrated=False)
        runner = make_runner(small_sheet(tmp_path), env)
        assert runner.run() == 2                        # refusal, before any GPU work
        assert comfy.submissions == []

        supervised = make_runner(small_sheet(tmp_path), env, allow_uncalibrated=True)
        assert supervised.run() == 0                    # explicit supervised override


def test_judge_unreachable_halts_run_ships_nothing(clips, tmp_path):
    with serve_comfy(fixture_paths=clips) as comfy:
        with serve_judge() as (_, jurl):
            env = make_env(tmp_path, comfy, jurl)
        # judge server now DOWN (context exited); config still points at its port
        runner = make_runner(small_sheet(tmp_path), env)
        code = runner.run()
        assert code == 3
        rep = run_report(runner)
        assert rep["run"]["status"].startswith("HALTED")
        assert "judge" in (rep["run"]["halt_reason"] or "").lower() \
            or "unreachable" in (rep["run"]["halt_reason"] or "").lower()
        assert list((runner.run_dir / "encoded").glob("*")) == []


def test_vram_never_frees_halts_before_judging(clips, tmp_path):
    with serve_comfy(fixture_paths=clips, never_frees=True) as comfy, \
         serve_judge() as (judge, jurl):
        env = make_env(tmp_path, comfy, jurl,
                       policies_over={"vram_handoff": {"free_min_gb": 20,
                                                       "wait_timeout_s": 1}})
        runner = make_runner(small_sheet(tmp_path), env)
        code = runner.run()
        assert code == 3
        assert judge.chat_calls == 0                    # never judged on a hot GPU
        rep = run_report(runner)
        assert "VRAM" in rep["run"]["halt_reason"]
        assert list((runner.run_dir / "encoded").glob("*")) == []


def test_checker_killed_halts_and_ships_nothing(clips, tmp_path):
    with serve_comfy(fixture_paths=clips) as comfy, serve_judge() as (_, jurl):
        env = make_env(tmp_path, comfy, jurl)
        import sys
        runner = make_runner(small_sheet(tmp_path), env,
                             checker_argv=[sys.executable, "-c",
                                           "import sys; sys.exit(137)"])
        code = runner.run()
        assert code == 3
        rep = run_report(runner)
        assert "checker contract violated" in rep["run"]["halt_reason"]
        assert list((runner.run_dir / "encoded").glob("*")) == []


def test_ship_gate_rejects_everything_but_full_pass():
    ok = {"verdict": "PASS", "layers_run": ["l1", "l2"], "error": None,
          "thresholds": {"calibrated": True}}
    assert ship_gate(ok) == (True, "ok")
    assert not ship_gate({**ok, "verdict": "PROCEED"})[0]          # row 6
    assert not ship_gate({**ok, "layers_run": ["l1"]})[0]
    assert not ship_gate({**ok, "error": {"scope": "clip"}})[0]
    assert not ship_gate({**ok, "thresholds": {"calibrated": False}})[0]
    assert ship_gate({**ok, "thresholds": {"calibrated": False}},
                     allow_uncalibrated=True)[0]
    assert not ship_gate(None)[0]


def test_oom_recovery_frees_and_rerolls(clips, tmp_path):
    with serve_comfy(fixture_paths=clips,
                     scenarios=[{"error": "oom"}, {"fixture": "moving"}]) as comfy, \
         serve_judge() as (_, jurl):
        env = make_env(tmp_path, comfy, jurl)
        runner = make_runner(small_sheet(tmp_path), env)
        assert runner.run() == 0
        rep = run_report(runner)
        clip = rep["clips"][0]
        assert clip["status"] == "passed"
        assert clip["attempts"][0]["status"] == "gen_failed"
        assert comfy.free_calls >= 2                    # OOM recovery + handoff
        assert len(comfy.submissions) == 2


def test_budget_trip_skips_generation_but_judges_existing(clips, tmp_path):
    with serve_comfy(fixture_paths=clips) as comfy, serve_judge() as (judge, jurl):
        env = make_env(tmp_path, comfy, jurl)
        runner = make_runner(small_sheet(tmp_path, n_clips=2), env)
        calls = {"n": 0}
        real_block = runner._budget_block

        def fake_block():
            calls["n"] += 1
            if calls["n"] <= 1:
                return real_block()
            runner.wall_tripped = True
            return "budget"

        runner._budget_block = fake_block
        code = runner.run()
        assert code == 0
        rep = run_report(runner)
        by_id = {c["clip_id"]: c for c in rep["clips"]}
        assert by_id["V1C1"]["status"] == "passed"      # generated before the trip
        assert by_id["V1C1"]["encoded"]                 # ...and still encoded
        assert by_id["V1C2"]["status"] == "skipped"     # never generated after it
        assert by_id["V1C2"]["skip_reason"] == "budget"
        assert rep["run"]["status"] == "COMPLETED(budget-stopped)"
        assert rep["budget"]["wall_tripped"] is True
        assert len(comfy.submissions) == 1


def test_resume_reuses_verdicts_no_regeneration(clips, tmp_path):
    with serve_comfy(fixture_paths=clips) as comfy, serve_judge() as (judge, jurl):
        env = make_env(tmp_path, comfy, jurl)
        first = make_runner(small_sheet(tmp_path), env)
        assert first.run() == 0
        subs_before, chats_before = len(comfy.submissions), judge.chat_calls

        resumed = make_runner(small_sheet(tmp_path), env, resume_dir=first.run_dir)
        assert resumed.run() == 0
        assert len(comfy.submissions) == subs_before    # nothing regenerated
        assert judge.chat_calls == chats_before         # nothing re-judged
        rep = run_report(resumed)
        assert rep["clips"][0]["status"] == "passed"


def test_resume_reattaches_in_flight_job(clips, tmp_path):
    """Crash after submit, before download: resume must wait/download, not resubmit."""
    pre_pid = "preloaded-job-1"
    preloaded = {pre_pid: {"scenario": {"fixture": "moving"}, "polls": 5,
                           "done": False, "info": {}, "order": 0}}
    with serve_comfy(fixture_paths=clips, preloaded_jobs=preloaded) as comfy, \
         serve_judge() as (judge, jurl):
        env = make_env(tmp_path, comfy, jurl)
        dispatch = small_sheet(tmp_path)

        # forge the pre-crash run dir: submitted event, no attempt line
        run_dir = env["runs"] / "crashed-run"
        from shortsloop.state import RunLog
        log = RunLog(run_dir, "crashed-run")
        log.event("schedule", "ok")
        log.event("claim", "ok", "V1C1", 1)
        log.event("generate", "enter", "V1C1", 1, seed=42)
        log.event("generate", "submitted", "V1C1", 1, prompt_id=pre_pid, seed=42)
        (run_dir / "prompts").mkdir(parents=True, exist_ok=True)
        (run_dir / "prompts" / "V1C1_a1.txt").write_text(
            "A red cube number 1 sliding across a dark slate table, slow dolly-in, "
            "cinematic lighting, vertical 9:16 composition.\n", encoding="utf-8")
        import shutil as _sh
        _sh.copyfile(dispatch, run_dir / "dispatch.md")

        runner = make_runner(dispatch, env, resume_dir=run_dir)
        assert runner.run() == 0
        assert comfy.submissions == []                  # re-attached, NOT resubmitted
        rep = run_report(runner)
        assert rep["clips"][0]["status"] == "passed"
        assert judge.chat_calls == 1
