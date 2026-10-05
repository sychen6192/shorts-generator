"""Resume + halt bookkeeping (plan §5 row 14, hard rule 6).

A generated attempt must never be lost: if the run halts or crashes between
generation and the judge, the attempt is in attempts.jsonl, and --resume judges the
existing clip instead of burning GPU on a regeneration.
"""

from __future__ import annotations

import json
import shutil

from fake_comfy import serve_comfy
from fake_judge import serve as serve_judge
from test_runner import make_env, make_runner, run_report, small_sheet

from shortsloop.policy import MOTION_PHRASES
from shortsloop.state import RunLog

PROMPT1 = ("A red cube number 1 sliding across a dark slate table, slow dolly-in, "
           "cinematic lighting, vertical 9:16 composition.")


def _attempts(run_dir):
    return [json.loads(l) for l in
            (run_dir / "attempts.jsonl").read_text(encoding="utf-8").splitlines()]


def test_halt_before_judge_logs_attempts_and_resume_does_not_regenerate(clips, tmp_path):
    # the pre-generation handoff frees fine; the judge handoff finds Wan stuck
    with serve_comfy(fixture_paths=clips, free_results=[28.0, 4.0]) as comfy, \
         serve_judge() as (judge, jurl):
        env = make_env(tmp_path, comfy, jurl,
                       policies_over={"vram_handoff": {"free_min_gb": 20,
                                                       "wait_timeout_s": 1}})
        first = make_runner(small_sheet(tmp_path, n_clips=2), env)
        assert first.run() == 3
        assert judge.chat_calls == 0

        # hard rule 6: both generated attempts are logged although never judged
        att = _attempts(first.run_dir)
        assert [(a["clip_id"], a["attempt"]) for a in att] == [("V1C1", 1), ("V1C2", 1)]
        for a in att:
            assert a["status"] == "unjudged"
            assert a["seed"] is not None and a["output_sha256"]
            assert a["prompt_text"] and a["workflow_sha256"]
        subs = len(comfy.submissions)
        assert subs == 2

        resumed = make_runner(small_sheet(tmp_path, n_clips=2), env,
                              resume_dir=first.run_dir)
        assert resumed.run() == 0
        assert len(comfy.submissions) == subs            # nothing regenerated
        assert judge.chat_calls == 2                      # judged exactly once each
        rep = run_report(resumed)
        assert [c["status"] for c in rep["clips"]] == ["passed", "passed"]
        assert sorted(p.name for p in (resumed.run_dir / "encoded").glob("*.mp4")) == \
            ["V1C1.mp4", "V1C2.mp4"]
        # report history has ONE row per attempt (final outcome supersedes unjudged)
        assert [a["attempt"] for a in rep["clips"][0]["attempts"]] == [1]
        assert rep["clips"][0]["attempts"][0]["status"] == "passed"
        # same seed carried through (reproducibility)
        final = [a for a in _attempts(resumed.run_dir) if a["status"] == "passed"]
        assert {a["seed"] for a in final} == {a["seed"] for a in att}


def _forge_run(env, dispatch, run_id="crashed-run"):
    run_dir = env["runs"] / run_id
    log = RunLog(run_dir, run_id)
    for sub in ("clips", "prompts", "verdicts"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(dispatch, run_dir / "dispatch.md")
    log.event("schedule", "ok")
    return run_dir, log


def test_hard_crash_after_l1_proceed_resumes_into_judge(clips, tmp_path):
    """SIGKILL between the L1 gate and the judge wave: no attempt line, no halt
    handler — events alone must carry the attempt into the judge on resume."""
    with serve_comfy(fixture_paths=clips) as comfy, serve_judge() as (judge, jurl):
        env = make_env(tmp_path, comfy, jurl)
        dispatch = small_sheet(tmp_path)
        run_dir, log = _forge_run(env, dispatch)
        clip = run_dir / "clips" / "V1C1_a1.mp4"
        shutil.copyfile(clips["moving"], clip)
        (run_dir / "prompts" / "V1C1_a1.txt").write_text(PROMPT1 + "\n", encoding="utf-8")
        log.event("claim", "ok", "V1C1", 1)
        log.event("generate", "enter", "V1C1", 1, seed=4242, steps=None)
        log.event("generate", "submitted", "V1C1", 1, prompt_id="gone-from-history",
                  seed=4242, steps=None)
        log.event("generate", "ok", "V1C1", 1, file=str(clip))
        log.event("verify", "ok", "V1C1", 1, layer="l1", verdict="PROCEED")

        runner = make_runner(dispatch, env, resume_dir=run_dir)
        assert runner.run() == 0
        assert comfy.submissions == []
        assert judge.chat_calls == 1
        rep = run_report(runner)
        assert rep["clips"][0]["status"] == "passed"
        att = _attempts(run_dir)
        assert len(att) == 1 and att[0]["seed"] == 4242 and att[0]["attempt"] == 1


def test_hard_crash_after_download_resumes_into_l1(clips, tmp_path):
    """Crash after the clip was downloaded but before its L1 verdict: resume runs the
    L1 gate on the file on disk — the job is NOT re-awaited or regenerated."""
    with serve_comfy(fixture_paths=clips) as comfy, serve_judge() as (judge, jurl):
        env = make_env(tmp_path, comfy, jurl)
        dispatch = small_sheet(tmp_path)
        run_dir, log = _forge_run(env, dispatch)
        clip = run_dir / "clips" / "V1C1_a1.mp4"
        shutil.copyfile(clips["moving"], clip)
        (run_dir / "prompts" / "V1C1_a1.txt").write_text(PROMPT1 + "\n", encoding="utf-8")
        log.event("claim", "ok", "V1C1", 1)
        log.event("generate", "enter", "V1C1", 1, seed=77, steps=None)
        log.event("generate", "submitted", "V1C1", 1, prompt_id="gone", seed=77,
                  steps=None)
        log.event("generate", "ok", "V1C1", 1, file=str(clip))

        runner = make_runner(dispatch, env, resume_dir=run_dir)
        assert runner.run() == 0
        assert comfy.submissions == []
        rep = run_report(runner)
        assert rep["clips"][0]["status"] == "passed"
        assert _attempts(run_dir)[0]["seed"] == 77


def test_resume_reapplies_reroll_policy_on_base_prompt(clips, tmp_path):
    """Crash after a2 failed static (a2 carried motion phrase #0). Resume must plan
    a3 per the frozen table: phrase #1 on the BASE prompt (not stacked on phrase #0)
    and --steps 8."""
    from shortsloop.policy import append_motion_phrase
    with serve_comfy(fixture_paths=clips) as comfy, serve_judge() as (judge, jurl):
        env = make_env(tmp_path, comfy, jurl)
        dispatch = small_sheet(tmp_path)
        run_dir, log = _forge_run(env, dispatch)
        p2 = append_motion_phrase(PROMPT1, 0)
        for n, prompt in ((1, PROMPT1), (2, p2)):
            log.event("claim", "ok", "V1C1", n)
            log.event("generate", "submitted", "V1C1", n, prompt_id=f"old{n}", seed=n)
            log.event("generate", "ok", "V1C1", n, file="x")
            log.attempt({"clip_id": "V1C1", "attempt": n, "seed": n,
                         "prompt_text": prompt, "prompt_base": PROMPT1,
                         "prompt_rewritten": False, "status": "l1_failed",
                         "verdict": "FAIL", "failure_classes": ["static"],
                         "steps": None})

        runner = make_runner(dispatch, env, resume_dir=run_dir)
        assert runner.run() == 0
        assert len(comfy.submissions) == 1
        j3 = comfy.job_info(0)
        assert MOTION_PHRASES[1].strip("; ") in j3["prompt"]
        assert MOTION_PHRASES[0].strip("; ") not in j3["prompt"]
        assert j3["steps"] == 8
        att = _attempts(run_dir)
        assert att[-1]["attempt"] == 3 and att[-1]["status"] == "passed"
