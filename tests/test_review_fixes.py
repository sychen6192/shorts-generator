"""Regression tests for the post-fix adversarial review (2026-10-04): each failed
before its fix. Several replace hard-rule tests that the review showed were
vacuous against the old fakes."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from fake_comfy import serve_comfy
from fake_judge import GOOD_SCORES, serve as serve_judge
from test_runner import make_env, make_runner, run_report, small_sheet

import shortsloop.comfy as comfy_mod
from shortsloop.errors import InfraError
from shortsloop.runner import Runner
from shortsloop.state import RunLog

PROMPT1 = ("A red cube number 1 sliding across a dark slate table, slow dolly-in, "
           "cinematic lighting, vertical 9:16 composition.")
DEFORMED = {**{k: dict(v) for k, v in GOOD_SCORES.items()},
            "anatomy_artifacts": {"score": 1, "na": False, "reason": "fused parts"}}
OFF_PROMPT = {**{k: dict(v) for k, v in GOOD_SCORES.items()},
              "prompt_adherence": {"score": 2, "na": False, "reason": "no cube"}}


def attempts(run_dir: Path) -> list[dict]:
    p = run_dir / "attempts.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def last_lines(run_dir: Path) -> dict:
    return {(a["clip_id"], a["attempt"]): a for a in attempts(run_dir)}


def encoded(run_dir: Path) -> list[str]:
    return sorted(p.name for p in (run_dir / "encoded").glob("*.mp4"))


# ------------------------------------------------------------ resume / halt

def test_halt_between_claim_and_submit_does_not_cost_a_wave(clips, tmp_path, monkeypatch):
    real_submit = comfy_mod.ComfyClient.submit
    calls = {"n": 0}

    def flaky_submit(self, **kw):
        calls["n"] += 1
        if calls["n"] == 2:
            raise InfraError("l1", "ComfyUI briefly unreachable")
        return real_submit(self, **kw)
    monkeypatch.setattr(comfy_mod.ComfyClient, "submit", flaky_submit)
    with serve_comfy(fixture_paths=clips) as comfy, \
         serve_judge(scores=DEFORMED) as (_, jurl):
        env = make_env(tmp_path, comfy, jurl)
        first = make_runner(small_sheet(tmp_path), env)
        assert first.run() == 3
        resumed = make_runner(small_sheet(tmp_path), env, resume_dir=first.run_dir)
        assert resumed.run() == 0
        clip = run_report(resumed)["clips"][0]
        assert clip["status"] == "failed_final", clip
        assert [a["attempt"] for a in clip["attempts"]] == [1, 2, 3]


def test_escalation_halt_keeps_the_final_error_line(clips, tmp_path):
    with serve_comfy(fixture_paths=clips) as comfy, \
         serve_judge(scenario="garbage") as (_, jurl):
        runner = make_runner(small_sheet(tmp_path), make_env(tmp_path, comfy, jurl))
        assert runner.run() == 3
        final = last_lines(runner.run_dir)
        assert final[("V1C1", 2)]["status"] == "error"
        assert all(a["status"] != "unjudged" for a in attempts(runner.run_dir))


def test_failed_generation_is_logged_even_when_its_cleanup_halts(clips, tmp_path,
                                                                 monkeypatch):
    def broken_cleanup(self, *a, **k):
        raise InfraError("l1", "ComfyUI vanished during cleanup")
    monkeypatch.setattr(comfy_mod.ComfyClient, "ensure_gone", broken_cleanup)
    with serve_comfy(fixture_paths=clips, scenarios=[{"error": "oom"}]) as comfy, \
         serve_judge() as (_, jurl):
        runner = make_runner(small_sheet(tmp_path), make_env(tmp_path, comfy, jurl))
        assert runner.run() == 3
        a1 = last_lines(runner.run_dir)[("V1C1", 1)]
        assert a1["status"] == "gen_failed" and a1["seed"] is not None


def _forge(env, dispatch):
    run_dir = env["runs"] / "crashed"
    log = RunLog(run_dir, "crashed")
    for sub in ("prompts", "clips", "verdicts"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(dispatch, run_dir / "dispatch.md")
    log.event("schedule", "enter")
    return run_dir, log


def test_resume_keeps_an_in_flight_rewrite(clips, tmp_path):
    """a1 failed off_prompt, the rewrite happened, a2 (rewritten) was in flight at
    the crash. Resume must log a2 as rewritten (with its diff) and keep the
    rewritten prompt for a3 (plan §2.3: 'rewritten prompt kept')."""
    rewritten = "REWRITTEN: a red cube sliding fast; slow dolly-in, vertical 9:16 composition"
    pre = {"pre-a2": {"scenario": {"fixture": "moving"}, "polls": 0, "done": False,
                      "info": {}, "order": 0}}
    with serve_comfy(fixture_paths=clips, preloaded_jobs=pre) as comfy, \
         serve_judge(scores=OFF_PROMPT) as (judge, jurl):
        env = make_env(tmp_path, comfy, jurl)
        dispatch = small_sheet(tmp_path)
        run_dir, log = _forge(env, dispatch)
        log.event("claim", "ok", "V1C1", 1, action="initial", prompt_base=PROMPT1,
                  rewritten=False, rewrite_diff=None)
        log.event("generate", "submitted", "V1C1", 1, prompt_id="old-a1", seed=1)
        log.event("generate", "ok", "V1C1", 1, file="gone.mp4")
        log.attempt({"clip_id": "V1C1", "attempt": 1, "seed": 1, "prompt_text": PROMPT1,
                     "prompt_base": PROMPT1, "prompt_rewritten": False,
                     "status": "l2_failed", "verdict": "FAIL",
                     "failure_classes": ["off_prompt"], "steps": None})
        (run_dir / "prompts" / "V1C1_a2.txt").write_text(rewritten + "\n")
        log.event("claim", "ok", "V1C1", 2, action="rewrite", prompt_base=rewritten,
                  rewritten=True, rewrite_diff="-old\n+new")
        log.event("generate", "submitted", "V1C1", 2, prompt_id="pre-a2", seed=2)
        runner = make_runner(dispatch, env, resume_dir=run_dir)
        assert runner.run() == 0
        final = last_lines(run_dir)
        assert final[("V1C1", 2)]["prompt_rewritten"] is True
        assert final[("V1C1", 2)]["prompt_diff"] == "-old\n+new"
        assert comfy.job_info(0)["prompt"] == rewritten            # a3 keeps it
        assert judge.rewrite_calls == 0                              # one rewrite ever


def test_resume_does_not_load_the_rewrite_llm_beside_a_running_wan_job(clips, tmp_path):
    pre = {"pre-v1c2": {"scenario": {"fixture": "moving"}, "polls": 0, "done": False,
                        "info": {}, "order": 0}}
    with serve_comfy(fixture_paths=clips, preloaded_jobs=pre) as comfy, \
         serve_judge(scores_queue=[GOOD_SCORES, GOOD_SCORES]) as (judge, jurl):
        env = make_env(tmp_path, comfy, jurl)
        dispatch = small_sheet(tmp_path, n_clips=2)
        run_dir, log = _forge(env, dispatch)
        log.attempt({"clip_id": "V1C1", "attempt": 1, "seed": 1, "prompt_text": PROMPT1,
                     "prompt_base": PROMPT1, "status": "l2_failed", "verdict": "FAIL",
                     "failure_classes": ["off_prompt"], "steps": None,
                     "verdict_path": None})
        (run_dir / "prompts" / "V1C2_a1.txt").write_text(
            PROMPT1.replace("number 1", "number 2") + "\n")
        log.event("generate", "submitted", "V1C2", 1, prompt_id="pre-v1c2", seed=5)
        runner = make_runner(dispatch, env, resume_dir=run_dir)
        assert runner.run() == 0
        assert judge.rewrite_calls == 1
        assert judge.rewrite_times[0] > comfy.jobs["pre-v1c2"]["done_at"]


def test_one_gate_refusal_at_persist_does_not_block_the_other_encodes(clips, tmp_path):
    class Mismatch(Runner):
        def _persist_phase(self):
            for item in self.items:
                if item.spec.clip_id == "V1C1" and item.status == "passed":
                    p = Path(item.passed_info["verdict_path"])
                    v = json.loads(p.read_text())
                    v["layers_run"] = ["l1"]                  # checker/runner mismatch
                    p.write_text(json.dumps(v))
            return super()._persist_phase()
    with serve_comfy(fixture_paths=clips) as comfy, serve_judge() as (_, jurl):
        env = make_env(tmp_path, comfy, jurl)
        runner = Mismatch(dispatch_path=small_sheet(tmp_path, n_clips=2),
                          config_path=env["config"], pipeline_path=env["pipeline"],
                          thresholds_path=env["thresholds"], runs_dir=env["runs"])
        assert runner.run() == 3                              # still a loud halt...
        assert encoded(runner.run_dir) == ["V1C2.mp4"]        # ...after the others
        assert "ship-gate" in run_report(runner)["run"]["halt_reason"]


def _with_provenance_judge(env, model, digest):
    thr = yaml.safe_load(env["thresholds"].read_text())
    thr["provenance"] = {"judge": {"model": model, "model_digest": digest}}
    env["thresholds"].write_text(yaml.safe_dump(thr))


def test_thresholds_calibrated_for_another_judge_refuse(clips, tmp_path):
    with serve_comfy(fixture_paths=clips) as comfy, serve_judge() as (_, jurl):
        env = make_env(tmp_path, comfy, jurl)
        _with_provenance_judge(env, "ollama/some-32b-vl", "sha256:fakedigest")
        assert make_runner(small_sheet(tmp_path), env).run() == 2
        assert comfy.submissions == []


def test_judge_build_differs_from_calibration_halts_before_judging(clips, tmp_path):
    with serve_comfy(fixture_paths=clips) as comfy, serve_judge() as (judge, jurl):
        env = make_env(tmp_path, comfy, jurl)
        _with_provenance_judge(env, "ollama/qwen3-vl:8b-instruct", "sha256:otherbuild")
        runner = make_runner(small_sheet(tmp_path), env)
        assert runner.run() == 3
        assert judge.chat_calls == 0
        assert "calibrat" in run_report(runner)["run"]["halt_reason"]


def test_openai_compat_judge_without_unload_hook_refuses(clips, tmp_path):
    with serve_comfy(fixture_paths=clips) as comfy, serve_judge() as (_, jurl):
        env = make_env(tmp_path, comfy, jurl)
        cfg = yaml.safe_load(env["config"].read_text())
        cfg["judge"]["adapter"] = "openai_compat"
        env["config"].write_text(yaml.safe_dump(cfg))
        from test_runner_hardening import _refresh_doctor
        _refresh_doctor(env)
        assert make_runner(small_sheet(tmp_path), env).run() == 2


# ------------------------------------------------------------ truthful hard-rule tests

def test_judge_and_rewrite_unloaded_after_the_wave_before_the_next_generation(clips,
                                                                              tmp_path):
    with serve_comfy(fixture_paths=clips) as comfy, \
         serve_judge(scores_queue=[OFF_PROMPT, GOOD_SCORES]) as (judge, jurl):
        runner = make_runner(small_sheet(tmp_path), make_env(tmp_path, comfy, jurl))
        assert runner.run() == 0
        wave1_end = judge.rewrite_times[0]          # rewrite runs at the end of wave 1
        wave2_start = comfy.submit_times[1]
        between = [t for t in judge.generate_times if wave1_end < t < wave2_start]
        assert between, "no unload between the judge wave and the next Wan job"
        assert {"qwen3-vl:8b-instruct", "fake-text-model"} <= set(judge.unloaded_models)


def test_busy_queue_is_never_joined(clips, tmp_path):
    """A foreign job sits in ComfyUI's queue: the runner must not submit beside it."""
    pre = {"foreign": {"scenario": {"hang": True}, "polls": 0, "done": False,
                       "info": {}, "order": 0}}
    with serve_comfy(fixture_paths=clips, preloaded_jobs=pre) as comfy, \
         serve_judge() as (_, jurl):
        env = make_env(tmp_path, comfy, jurl,
                       policies_over={"vram_handoff": {"wait_timeout_s": 2}})
        runner = make_runner(small_sheet(tmp_path), env)
        assert runner.run() == 3
        assert comfy.submissions == [] and comfy.violations == 0
        assert "queue busy" in run_report(runner)["run"]["halt_reason"]


def test_oom_recovery_frees_between_the_failed_job_and_the_resubmit(clips, tmp_path):
    with serve_comfy(fixture_paths=clips,
                     scenarios=[{"error": "oom"}, {"fixture": "moving"}]) as comfy, \
         serve_judge() as (_, jurl):
        runner = make_runner(small_sheet(tmp_path), make_env(tmp_path, comfy, jurl))
        assert runner.run() == 0
        t1, t2 = comfy.submit_times[:2]
        assert any(t1 < t < t2 for t in comfy.free_times)


def test_consecutive_generation_failures_are_counted_across_clips(clips, tmp_path):
    """Cap 2/clip, escalation 3: only per-RUN counting can halt here."""
    with serve_comfy(fixture_paths=clips, scenarios=[{"error": "oom"}] * 9) as comfy, \
         serve_judge() as (_, jurl):
        env = make_env(tmp_path, comfy, jurl,
                       policies_over={"max_attempts_per_clip": 2, "waves_max": 2})
        runner = make_runner(small_sheet(tmp_path, n_clips=2), env)
        assert runner.run() == 3
        assert len(comfy.submissions) == 3


def test_verdict_cache_is_used_on_resume_and_keyed_by_judge_digest(clips, tmp_path):
    with serve_comfy(fixture_paths=clips) as comfy, serve_judge() as (judge, jurl):
        env = make_env(tmp_path, comfy, jurl)
        dispatch = small_sheet(tmp_path)
        done = make_runner(dispatch, env)
        assert done.run() == 0
        verdict = done.run_dir / "verdicts" / "V1C1_a1.json"

        def forged(name, digest=None):
            run_dir, log = _forge(env, dispatch)
            run_dir = run_dir.rename(env["runs"] / name)
            log = RunLog(run_dir, name)
            for sub in ("clips", "prompts", "verdicts"):
                for f in (done.run_dir / sub).glob("V1C1_a1*"):
                    shutil.copy(f, run_dir / sub / f.name)
            v = json.loads(verdict.read_text())
            if digest:
                v["l2"]["model_digest"] = digest
            (run_dir / "verdicts" / "V1C1_a1.json").write_text(json.dumps(v))
            log.event("generate", "ok", "V1C1", 1,
                      file=str(run_dir / "clips" / "V1C1_a1.mp4"))
            log.event("verify", "ok", "V1C1", 1, layer="l1", verdict="PROCEED")
            return run_dir

        before = judge.chat_calls
        hit = make_runner(dispatch, env, resume_dir=forged("hit"))
        assert hit.run() == 0
        assert judge.chat_calls == before                      # cache honoured
        miss = make_runner(dispatch, env, resume_dir=forged("miss", "sha256:old"))
        assert miss.run() == 0
        assert judge.chat_calls == before + 1                  # other judge: re-judged


def test_stale_verdict_consistent_with_exit_code_is_still_never_reused(clips, tmp_path):
    import sys as _sys
    from shortsloop.runner import RunHalted
    out = tmp_path / "v.json"
    out.write_text(json.dumps({"verdict": "FAIL", "failure_classes": ["static"]}))
    r = Runner(dispatch_path="x", config_path="x", pipeline_path="x",
               thresholds_path=tmp_path / "t.yaml",
               checker_argv=[_sys.executable, "-c", "import sys; sys.exit(1)"])
    r.expect, r.checker_config = {}, tmp_path / "c.yaml"
    r.pol = {"judge": {"retries": 1, "timeout_s": 5}}
    with pytest.raises(RunHalted):
        r._invoke_checker(clips["moving"], tmp_path / "p.txt", out, l1_only=True)


def test_resume_does_not_re_encode_persisted_clips(clips, tmp_path, monkeypatch):
    with serve_comfy(fixture_paths=clips) as comfy, serve_judge() as (_, jurl):
        env = make_env(tmp_path, comfy, jurl)
        first = make_runner(small_sheet(tmp_path), env)
        assert first.run() == 0
        import shortsloop.runner as runner_mod
        calls = []
        monkeypatch.setattr(runner_mod, "encode_silent",
                            lambda *a, **k: calls.append(a))
        again = make_runner(small_sheet(tmp_path), env, resume_dir=first.run_dir)
        assert again.run() == 0
        assert calls == []
        assert run_report(again)["clips"][0]["encoded"]


def test_submit_client_timeout_is_infra(monkeypatch):
    from shortsloop.comfy import ComfyClient, GenerationFailed

    def slow(self, *a, **k):
        raise GenerationFailed("timeout", "client process exceeded 300s")
    monkeypatch.setattr(ComfyClient, "_run_client", slow)
    with pytest.raises(InfraError):
        ComfyClient(host="h", workflow="w").submit(prompt="p", seed=1, width=1,
                                                   height=1, length=1)


def test_halted_report_gives_a_resume_command_the_cli_accepts(clips, tmp_path):
    with serve_comfy(fixture_paths=clips, never_frees=True) as comfy, \
         serve_judge() as (_, jurl):
        runner = make_runner(small_sheet(tmp_path), make_env(tmp_path, comfy, jurl))
        assert runner.run() == 3
        md = (runner.run_dir / "report.md").read_text()
        assert f"--dispatch {runner.run_dir / 'dispatch.md'}" in md
        assert f"--resume {runner.run_dir}" in md
