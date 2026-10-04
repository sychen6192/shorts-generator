"""Runner hardening: hard rules 3-6 and fail-closed matrix rows (plan §5) proven at
runner level against the fakes — every test here failed before its fix."""

from __future__ import annotations

import fcntl
import json
import shutil
import time
from pathlib import Path

import yaml

from fake_comfy import serve_comfy
from fake_judge import GOOD_SCORES, serve as serve_judge
from test_runner import DATA, make_env, make_runner, run_report, small_sheet

from shortsloop.runner import Runner, ship_gate
from shortsloop.state import RunLog

PROMPT = ("A red cube number {k} sliding across a dark slate table, slow dolly-in, "
          "cinematic lighting, vertical 9:16 composition.")
BAD_ADHERENCE = {**{k: dict(v) for k, v in GOOD_SCORES.items()},
                 "prompt_adherence": {"score": 2, "na": False, "reason": "no cube"}}


def attempts(run_dir: Path) -> list[dict]:
    p = run_dir / "attempts.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def encoded(run_dir: Path, sub: str = "encoded") -> list[str]:
    return sorted(p.name for p in (run_dir / sub).glob("*.mp4"))


# ------------------------------------------------------------ hard rule 4

def test_client_crash_mid_wait_never_leaves_a_second_job_beside_the_first(clips, tmp_path):
    """The vendored client dies while the job keeps running on the server: the
    runner must stop THAT job and confirm it is gone before re-rolling."""
    with serve_comfy(fixture_paths=clips,
                     scenarios=[{"history_500": True}, {"fixture": "moving"}]) as comfy, \
         serve_judge() as (_, jurl):
        runner = make_runner(small_sheet(tmp_path), make_env(tmp_path, comfy, jurl))
        assert runner.run() == 0
        assert comfy.violations == 0
        assert comfy.interrupt_calls >= 1
        att = attempts(runner.run_dir)
        assert att[0]["status"] == "gen_failed" and "client_error" in att[0]["notes"]
        assert att[1]["status"] == "passed"


def test_timeout_interrupts_our_job_and_verifies_before_resubmitting(clips, tmp_path):
    with serve_comfy(fixture_paths=clips,
                     scenarios=[{"hang": True}, {"fixture": "moving"}]) as comfy, \
         serve_judge() as (_, jurl):
        env = make_env(tmp_path, comfy, jurl, policies_over={"comfy": {"timeout_s": 2}})
        runner = make_runner(small_sheet(tmp_path), env)
        assert runner.run() == 0
        assert comfy.violations == 0
        assert attempts(runner.run_dir)[0]["notes"].startswith("timeout")


def test_resume_reattaches_in_flight_jobs_before_any_new_submission(clips, tmp_path):
    """V1C2's job is still running on the server; V1C1 was never submitted. A
    resumed run must finish V1C2 first — submitting V1C1 now would put two jobs
    on the server (hard rule 4)."""
    pre = {"pre-v1c2": {"scenario": {"fixture": "moving"}, "polls": 0, "done": False,
                        "info": {}, "order": 0}}
    with serve_comfy(fixture_paths=clips, preloaded_jobs=pre) as comfy, \
         serve_judge() as (_, jurl):
        env = make_env(tmp_path, comfy, jurl)
        dispatch = small_sheet(tmp_path, n_clips=2)
        run_dir = env["runs"] / "crashed"
        log = RunLog(run_dir, "crashed")
        (run_dir / "prompts").mkdir(parents=True)
        shutil.copyfile(dispatch, run_dir / "dispatch.md")
        (run_dir / "prompts" / "V1C2_a1.txt").write_text(PROMPT.format(k=2) + "\n")
        log.event("schedule", "ok")
        log.event("generate", "submitted", "V1C2", 1, prompt_id="pre-v1c2", seed=5)
        runner = make_runner(dispatch, env, resume_dir=run_dir)
        assert runner.run() == 0
        assert comfy.violations == 0
        assert len(comfy.submissions) == 1                     # only V1C1 is new
        assert [c["status"] for c in run_report(runner)["clips"]] == ["passed", "passed"]


def test_resume_does_not_wait_out_a_job_a_restarted_server_forgot(clips, tmp_path):
    with serve_comfy(fixture_paths=clips) as comfy, serve_judge() as (_, jurl):
        env = make_env(tmp_path, comfy, jurl, policies_over={"comfy": {"timeout_s": 60}})
        dispatch = small_sheet(tmp_path)
        run_dir = env["runs"] / "crashed"
        log = RunLog(run_dir, "crashed")
        (run_dir / "prompts").mkdir(parents=True)
        shutil.copyfile(dispatch, run_dir / "dispatch.md")
        (run_dir / "prompts" / "V1C1_a1.txt").write_text(PROMPT.format(k=1) + "\n")
        log.event("schedule", "ok")
        log.event("generate", "submitted", "V1C1", 1, prompt_id="forgotten", seed=5)
        t0 = time.monotonic()
        runner = make_runner(dispatch, env, resume_dir=run_dir)
        assert runner.run() == 0
        assert time.monotonic() - t0 < 40                      # not the 60 s timeout
        assert comfy.interrupt_calls == 0                      # no blind interrupt
        att = attempts(run_dir)
        assert att[0]["status"] == "gen_failed" and att[0]["notes"].startswith("lost")
        assert att[1]["status"] == "passed"


# ------------------------------------------------------------ hard rule 3

def test_generation_waits_for_verified_free_vram_after_the_judge(clips, tmp_path):
    """Judge -> generation is a verified handoff too: if the VLM/LLM is still
    resident when wave 2 would start, halt instead of loading Wan beside it."""
    with serve_comfy(fixture_paths=clips, free_results=[28.0, 28.0, 4.0]) as comfy, \
         serve_judge(scores_queue=[BAD_ADHERENCE, GOOD_SCORES]) as (judge, jurl):
        runner = make_runner(small_sheet(tmp_path), make_env(tmp_path, comfy, jurl))
        assert runner.run() == 3
        rep = run_report(runner)
        assert "resident" in rep["run"]["halt_reason"]
        assert len(comfy.submissions) == 1                     # wave 2 never started
        assert judge.generate_calls >= 1                       # unload was requested
        assert judge.last_rewrite_payload["keep_alive"] == 0   # rewrite LLM not kept
        assert encoded(runner.run_dir) == []


def test_first_wave_starts_from_a_verified_clean_gpu(clips, tmp_path):
    with serve_comfy(fixture_paths=clips, never_frees=True, vram_free_gb=4.0) as comfy, \
         serve_judge() as (judge, jurl):
        runner = make_runner(small_sheet(tmp_path), make_env(tmp_path, comfy, jurl))
        assert runner.run() == 3
        assert comfy.submissions == [] and judge.chat_calls == 0


# ------------------------------------------------------------ infra vs clip

def test_every_job_failing_is_an_infra_halt_not_a_quiet_completed(clips, tmp_path):
    with serve_comfy(fixture_paths=clips, scenarios=[{"error": "oom"}] * 9) as comfy, \
         serve_judge() as (_, jurl):
        runner = make_runner(small_sheet(tmp_path, n_clips=2),
                             make_env(tmp_path, comfy, jurl))
        assert runner.run() == 3
        assert "consecutive" in run_report(runner)["run"]["halt_reason"]
        assert len(comfy.submissions) == 3
        assert len(attempts(runner.run_dir)) == 3              # every attempt logged


def test_unusable_workflow_refuses_before_any_gpu_work(clips, tmp_path):
    ui = tmp_path / "ui_export.json"
    ui.write_text(json.dumps({"nodes": [], "links": []}))
    with serve_comfy(fixture_paths=clips) as comfy, serve_judge() as (_, jurl):
        env = make_env(tmp_path, comfy, jurl)
        cfg = yaml.safe_load(env["config"].read_text())
        cfg["comfy"]["workflow_t2v"] = str(ui)
        env["config"].write_text(yaml.safe_dump(cfg))
        _refresh_doctor(env)
        assert make_runner(small_sheet(tmp_path), env).run() == 2
        assert comfy.submissions == []


# ------------------------------------------------------------ fail-closed rows 2/3/4/10/13

def test_row4_corrupt_output_consumes_an_attempt_and_rerolls(clips, tmp_path):
    with serve_comfy(fixture_paths=clips,
                     scenarios=[{"fixture": "corrupt"}, {"fixture": "moving"}]) as comfy, \
         serve_judge() as (_, jurl):
        runner = make_runner(small_sheet(tmp_path), make_env(tmp_path, comfy, jurl))
        assert runner.run() == 0
        a1, a2 = attempts(runner.run_dir)
        assert a1["verdict"] == "ERROR" and a1["failure_classes"] == ["broken"]
        assert a2["status"] == "passed"
        assert comfy.violations == 0


def test_row2_garbage_judge_consumes_attempt_then_rerolls_never_ships_it(clips, tmp_path):
    with serve_comfy(fixture_paths=clips) as comfy, \
         serve_judge(scores_queue=["garbage", "garbage", GOOD_SCORES]) as (judge, jurl):
        runner = make_runner(small_sheet(tmp_path), make_env(tmp_path, comfy, jurl))
        assert runner.run() == 0
        a1, a2 = attempts(runner.run_dir)
        assert a1["verdict"] == "ERROR" and a2["status"] == "passed"
        assert encoded(runner.run_dir) == ["V1C1.mp4"]


def test_row3_consecutive_judge_errors_escalate_across_waves(clips, tmp_path):
    """One clip per wave: the counter must survive the wave boundary, otherwise a
    judge that fails every call burns all re-rolls and the run ends 'COMPLETED'."""
    with serve_comfy(fixture_paths=clips) as comfy, \
         serve_judge(scenario="garbage") as (_, jurl):
        runner = make_runner(small_sheet(tmp_path), make_env(tmp_path, comfy, jurl))
        assert runner.run() == 3
        assert "consecutive" in run_report(runner)["run"]["halt_reason"]
        assert len(comfy.submissions) == 2
        assert encoded(runner.run_dir) == []


def test_row10_l2_path_stops_at_exactly_three_attempts(clips, tmp_path):
    with serve_comfy(fixture_paths=clips) as comfy, \
         serve_judge(scores=BAD_ADHERENCE) as (_, jurl):
        runner = make_runner(small_sheet(tmp_path), make_env(tmp_path, comfy, jurl))
        assert runner.run() == 0
        rep = run_report(runner)
        assert rep["clips"][0]["status"] == "failed_final"
        assert len(comfy.submissions) == 3 and rep["budget"]["waves_run"] == 3
        assert [a["attempt"] for a in attempts(runner.run_dir)] == [1, 2, 3]
        assert comfy.violations == 0


def test_row13_intake_refusal_lists_every_row_and_touches_no_gpu(clips, tmp_path, capsys):
    sheet = (DATA / "dispatch_small.md").read_text(encoding="utf-8")
    bad = tmp_path / "bad.md"
    bad.write_text(sheet.replace("### V1C2", "### V9C9").replace("### V2C1", "### V9C8"),
                   encoding="utf-8")
    with serve_comfy(fixture_paths=clips) as comfy, serve_judge() as (_, jurl):
        assert make_runner(bad, make_env(tmp_path, comfy, jurl)).run() == 2
        assert comfy.submissions == [] and comfy.free_calls == 0
    err = capsys.readouterr().err
    assert "V1C2" in err and "V2C1" in err


def test_row13_skipped_rows_are_loud_in_the_report(clips, tmp_path):
    with serve_comfy(fixture_paths=clips) as comfy, serve_judge() as (_, jurl):
        runner = make_runner(DATA / "dispatch_small.md", make_env(tmp_path, comfy, jurl))
        assert runner.run() == 0
        md = (runner.run_dir / "report.md").read_text(encoding="utf-8")
        assert "## Skipped at intake" in md and "`V2C2` (I2V" in md
        assert "**V2**: 1/2 clips passed" in md and "V2C2 skipped" in md


# ------------------------------------------------------------ hard rule 5 (budgets)

def test_row9_real_clock_budget_trip(clips, tmp_path):
    now = {"t": 0.0}
    with serve_comfy(fixture_paths=clips) as comfy, serve_judge() as (_, jurl):
        runner = make_runner(small_sheet(tmp_path, n_clips=2),
                             make_env(tmp_path, comfy, jurl), clock=lambda: now["t"])
        real = runner._finish_generation

        def generation_takes_seven_hours(*a, **kw):
            now["t"] += 7 * 3600
            return real(*a, **kw)
        runner._finish_generation = generation_takes_seven_hours
        assert runner.run() == 0
        rep = run_report(runner)
        assert rep["run"]["status"] == "COMPLETED(budget-stopped)"
        assert [c["status"] for c in rep["clips"]] == ["passed", "skipped"]
        assert encoded(runner.run_dir) == ["V1C1.mp4"]


def test_resume_charges_the_time_already_spent_against_the_budget(clips, tmp_path):
    with serve_comfy(fixture_paths=clips) as comfy, serve_judge() as (_, jurl):
        env = make_env(tmp_path, comfy, jurl, policies_over={"wall_clock_budget_h": 1})
        dispatch = small_sheet(tmp_path)
        run_dir = env["runs"] / "crashed"
        log = RunLog(run_dir, "crashed")
        shutil.copyfile(dispatch, run_dir / "dispatch.md")
        events = run_dir / "events.jsonl"
        base = time.time() - 4 * 3600
        events.write_text("".join(json.dumps(
            {"ts": base + dt, "run_id": "crashed", "clip_id": None, "attempt": None,
             "stage": "schedule", "event": ev, "data": {}}) + "\n"
            for dt, ev in ((0, "enter"), (2 * 3600, "ok"))))   # 2 h of prior activity
        assert log.read_events()
        runner = make_runner(dispatch, env, resume_dir=run_dir)
        # 1 h budget, 2 h already spent: nothing new may be generated
        assert runner.run() == 0
        assert comfy.submissions == []
        assert run_report(runner)["run"]["status"] == "COMPLETED(budget-stopped)"


def test_disk_below_floor_at_start_refuses(clips, tmp_path):
    with serve_comfy(fixture_paths=clips) as comfy, serve_judge() as (_, jurl):
        runner = make_runner(small_sheet(tmp_path), make_env(tmp_path, comfy, jurl),
                             disk_free_gb=lambda: 0.5)
        assert runner.run() == 2
        assert comfy.submissions == []


def test_disk_trip_mid_run_is_a_reported_stop(clips, tmp_path):
    calls = {"n": 0}

    def disk():
        calls["n"] += 1
        return 100.0 if calls["n"] <= 2 else 0.1
    with serve_comfy(fixture_paths=clips) as comfy, serve_judge() as (_, jurl):
        runner = make_runner(small_sheet(tmp_path, n_clips=2),
                             make_env(tmp_path, comfy, jurl), disk_free_gb=disk)
        assert runner.run() == 0
        rep = run_report(runner)
        assert rep["run"]["status"] == "COMPLETED(disk-stopped)"
        assert rep["clips"][1]["status"] == "skipped"
        assert "disk" in (runner.run_dir / "report.md").read_text().lower()


# ------------------------------------------------------------ halts, report honesty

def test_halt_still_encodes_clips_that_passed_before_it(clips, tmp_path):
    with serve_comfy(fixture_paths=clips, free_results=[28.0, 28.0, 4.0]) as comfy, \
         serve_judge(scores_queue=[GOOD_SCORES, BAD_ADHERENCE]) as (_, jurl):
        runner = make_runner(small_sheet(tmp_path, n_clips=2),
                             make_env(tmp_path, comfy, jurl))
        assert runner.run() == 3
        assert encoded(runner.run_dir) == ["V1C1.mp4"]
        md = (runner.run_dir / "report.md").read_text(encoding="utf-8")
        assert "1 unfinished" in md
        rep = run_report(runner)
        assert rep["clips"][0]["encoded"] and rep["clips"][1]["status"] == "pending"


def test_persist_refuses_a_clip_whose_bytes_changed_after_its_verdict(clips, tmp_path):
    class Tamper(Runner):
        def _persist_phase(self):
            for item in self.items:
                if item.status == "passed":
                    Path(item.passed_info["clip_path"]).write_bytes(
                        Path(clips["static"]).read_bytes())
            return super()._persist_phase()
    with serve_comfy(fixture_paths=clips) as comfy, serve_judge() as (_, jurl):
        env = make_env(tmp_path, comfy, jurl)
        runner = Tamper(dispatch_path=small_sheet(tmp_path), config_path=env["config"],
                        pipeline_path=env["pipeline"], thresholds_path=env["thresholds"],
                        runs_dir=env["runs"])
        runner.run()
        assert encoded(runner.run_dir) == []
        assert "sha256" in (run_report(runner)["clips"][0]["encode_error"] or "")


def test_uncalibrated_supervised_passes_never_land_in_the_shipping_folder(clips, tmp_path):
    assert ship_gate({"verdict": "PASS", "layers_run": ["l1", "l2"], "error": None,
                      "thresholds": {"calibrated": False}})[0] is False
    with serve_comfy(fixture_paths=clips) as comfy, serve_judge() as (_, jurl):
        runner = make_runner(small_sheet(tmp_path),
                             make_env(tmp_path, comfy, jurl, calibrated=False),
                             allow_uncalibrated=True)
        assert runner.run() == 0
        assert encoded(runner.run_dir) == []
        assert encoded(runner.run_dir, "encoded_uncalibrated") == ["V1C1.mp4"]
        assert "UNCALIBRATED" in (runner.run_dir / "report.md").read_text()


# ------------------------------------------------------------ run identity

def test_second_runner_on_the_same_runs_dir_refuses(clips, tmp_path):
    with serve_comfy(fixture_paths=clips) as comfy, serve_judge() as (_, jurl):
        env = make_env(tmp_path, comfy, jurl)
        env["runs"].mkdir(parents=True, exist_ok=True)
        with open(env["runs"] / ".shortsloop.lock", "a+") as held:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            assert make_runner(small_sheet(tmp_path), env).run() == 2
        assert comfy.submissions == []


def test_run_ids_never_collide(clips, tmp_path):
    with serve_comfy(fixture_paths=clips) as comfy, serve_judge() as (_, jurl):
        env = make_env(tmp_path, comfy, jurl)
        a, b = (make_runner(small_sheet(tmp_path), env) for _ in range(2))
        for r in (a, b):
            assert r._load() is None
            r._init_run_dir()
        assert a.run_dir != b.run_dir


def test_resume_refuses_a_different_dispatch_sheet(clips, tmp_path):
    with serve_comfy(fixture_paths=clips) as comfy, serve_judge() as (_, jurl):
        env = make_env(tmp_path, comfy, jurl)
        first = make_runner(small_sheet(tmp_path), env)
        assert first.run() == 0
        started = json.loads((first.run_dir / "run.json").read_text())["started_at"]
        other = tmp_path / "other.md"
        other.write_text(small_sheet(tmp_path, n_clips=2).read_text())
        assert make_runner(other, env, resume_dir=first.run_dir).run() == 2
        assert make_runner(small_sheet(tmp_path), env, resume_dir=first.run_dir).run() == 0
        meta = json.loads((first.run_dir / "run.json").read_text())
        assert meta["started_at"] == started and len(meta["resumes"]) == 1


def _refresh_doctor(env):
    from shortsloop.doctor import fingerprint
    (env["config"].parent / "doctor.json").write_text(json.dumps(
        {"ok": True, "checks": [],
         "fingerprint": fingerprint(env["config"], env["pipeline"])}))


def test_doctor_snapshot_for_another_setup_does_not_open_the_gate(clips, tmp_path):
    with serve_comfy(fixture_paths=clips) as comfy, serve_judge() as (_, jurl):
        env = make_env(tmp_path, comfy, jurl)
        env["config"].write_text(env["config"].read_text() + "# edited after doctor\n")
        assert make_runner(small_sheet(tmp_path), env).run() == 2
        _refresh_doctor(env)
        assert make_runner(small_sheet(tmp_path), env).run() == 0


def test_missing_ffprobe_on_path_refuses(clips, tmp_path, monkeypatch):
    import shortsloop.runner as runner_mod
    real_which = shutil.which
    monkeypatch.setattr(runner_mod.shutil, "which",
                        lambda b: None if b == "ffprobe" else real_which(b))
    with serve_comfy(fixture_paths=clips) as comfy, serve_judge() as (_, jurl):
        assert make_runner(small_sheet(tmp_path), make_env(tmp_path, comfy, jurl)).run() == 2


# ------------------------------------------------------------ hard rule 6 / plan §2.7

def test_attempt_lines_carry_the_frozen_fields(clips, tmp_path):
    with serve_comfy(fixture_paths=clips) as comfy, \
         serve_judge(scores_queue=[BAD_ADHERENCE, GOOD_SCORES]) as (_, jurl):
        runner = make_runner(small_sheet(tmp_path), make_env(tmp_path, comfy, jurl))
        assert runner.run() == 0
        a1, a2 = attempts(runner.run_dir)
        frozen = {"ts", "clip_id", "attempt", "seed", "prompt_text", "prompt_sha256",
                  "prompt_rewritten", "prompt_diff", "workflow_path", "workflow_sha256",
                  "patch_args", "comfy_prompt_id", "output_path", "output_sha256",
                  "gen_elapsed_s", "verdict_path", "verdict", "failure_classes",
                  "l1_pass", "l2_pass", "vram_free_before_gb", "notes"}
        assert frozen <= set(a1) and frozen <= set(a2)
        assert (a1["l1_pass"], a1["l2_pass"]) == (True, False)
        assert (a2["l1_pass"], a2["l2_pass"]) == (True, True)


def test_pipeline_judge_policy_reaches_the_checker(clips, tmp_path):
    with serve_comfy(fixture_paths=clips) as comfy, serve_judge() as (_, jurl):
        env = make_env(tmp_path, comfy, jurl,
                       policies_over={"judge": {"timeout_s": 45, "retries": 0}})
        runner = make_runner(small_sheet(tmp_path), env)
        assert runner.run() == 0
        snap = yaml.safe_load((runner.run_dir / "config.snapshot.yaml").read_text())
        assert snap["judge"]["retries"] == 0 and snap["judge"]["timeout_s"] == 45


def test_stricter_generation_threshold_is_honoured(clips, tmp_path):
    """22 GB free would satisfy the judge's 20 GB floor, but the generation side is
    configured stricter (VLM still resident) -> halt before any submission."""
    with serve_comfy(fixture_paths=clips, free_results=[22.0]) as comfy, \
         serve_judge() as (_, jurl):
        env = make_env(tmp_path, comfy, jurl, policies_over={
            "vram_handoff": {"free_min_gb": 20, "gen_free_min_gb": 26,
                             "wait_timeout_s": 2}})
        assert make_runner(small_sheet(tmp_path), env).run() == 3
        assert comfy.submissions == []
