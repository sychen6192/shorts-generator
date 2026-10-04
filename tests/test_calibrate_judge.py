"""calibrate-tune --with-l2: verified VRAM handoff before judging + unload after
(hard rule 3), judge identity keyed scores (8B -> 32B escalation), and one bad
clip never aborting the batch. Fake ComfyUI + fake Ollama, real fixture clips."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

from fake_comfy import serve_comfy
from fake_judge import GOOD_SCORES, serve as serve_judge

from shortsloop import lock
from shortsloop.calibrate.tune import main_tune, score_with_judge
from shortsloop.state import _read_jsonl

DATA = Path(__file__).parent / "data"


def _cal(tmp_path, clips, kinds=("moving", "static", "flicker")):
    cal = tmp_path / "cal"
    (cal / "clips").mkdir(parents=True)
    with open(cal / "batch_manifest.jsonl", "w") as f:
        for i, kind in enumerate(kinds):
            p = cal / "clips" / f"cal{i:03d}.mp4"
            shutil.copyfile(clips[kind], p)
            f.write(json.dumps({"clip_id": f"cal{i:03d}", "kind": "good",
                                "prompt": f"a {kind} test clip, vertical 9:16 "
                                          f"composition", "clip_path": str(p)}) + "\n")
    return cal


def _env(tmp_path, comfy_host, judge_url, model="qwen3-vl:8b-instruct",
         wait_timeout_s=2, judge=None, judge_policy=None):
    cfg = {"judge": judge or {"adapter": "ollama", "base_url": judge_url,
                              "model": model, "timeout_s": 30},
           "paths": {"runs_dir": str(tmp_path / "runs")}}   # the run lock lives here
    if comfy_host:
        cfg["comfy"] = {"host": comfy_host,
                        "workflow_t2v": str(DATA / "test_workflow.json")}
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    pipeline = tmp_path / "pipeline.yaml"
    policies = {"vram_handoff": {"free_min_gb": 20, "wait_timeout_s": wait_timeout_s}}
    if judge_policy:
        policies["judge"] = judge_policy
    pipeline.write_text(yaml.safe_dump({"policies": policies}), encoding="utf-8")
    return config, pipeline


# ------------------------------------------------ [80]/[102] hard rule 3

def test_with_l2_frees_and_verifies_vram_before_judging_and_unloads_after(clips,
                                                                          tmp_path):
    cal = _cal(tmp_path, clips)
    with serve_comfy(fixture_paths=clips) as comfy, serve_judge() as (judge, jurl):
        config, pipeline = _env(tmp_path, comfy.host, jurl)
        assert score_with_judge(str(cal), str(config),
                                pipeline_path=str(pipeline)) == 0
        assert judge.chat_calls == 3
        assert comfy.free_calls == 1 and comfy.free_times[0] < judge.chat_times[0]
        assert judge.generate_calls == 1              # VLM unloaded after the wave
    rows = _read_jsonl(cal / "l2_scores.jsonl")
    assert len(rows) == 3 and all("dimensions" in r for r in rows)


def test_with_l2_halts_before_any_judge_call_when_vram_stays_occupied(clips, tmp_path):
    cal = _cal(tmp_path, clips)
    with serve_comfy(fixture_paths=clips, never_frees=True) as comfy, \
            serve_judge() as (judge, jurl):
        config, pipeline = _env(tmp_path, comfy.host, jurl, wait_timeout_s=1)
        assert score_with_judge(str(cal), str(config),
                                pipeline_path=str(pipeline)) == 3
        assert judge.chat_calls == 0
    assert not (cal / "l2_scores.jsonl").exists()


def test_with_l2_refuses_without_a_comfy_host(clips, tmp_path, capsys):
    cal = _cal(tmp_path, clips)
    with serve_judge() as (judge, jurl):
        config, pipeline = _env(tmp_path, None, jurl)
        assert score_with_judge(str(cal), str(config),
                                pipeline_path=str(pipeline)) == 2
        assert judge.chat_calls == 0
    assert "comfy.host" in capsys.readouterr().out


# ------------------------------------------------ [84] judge escalation

def test_switching_judge_rescores_every_clip(clips, tmp_path):
    cal = _cal(tmp_path, clips)
    with serve_comfy(fixture_paths=clips) as comfy, serve_judge() as (judge, jurl):
        config, pipeline = _env(tmp_path, comfy.host, jurl)
        assert score_with_judge(str(cal), str(config),
                                pipeline_path=str(pipeline)) == 0
        assert judge.chat_calls == 3
        # same judge, same digest: nothing to redo
        assert score_with_judge(str(cal), str(config),
                                pipeline_path=str(pipeline)) == 0
        assert judge.chat_calls == 3

    with serve_comfy(fixture_paths=clips) as comfy, \
            serve_judge(model="qwen3-vl:32b") as (judge, jurl):
        config, pipeline = _env(tmp_path, comfy.host, jurl, model="qwen3-vl:32b")
        assert score_with_judge(str(cal), str(config),
                                pipeline_path=str(pipeline)) == 0
        assert judge.chat_calls == 3                  # all re-judged by the 32B
    models = [r["model"] for r in _read_jsonl(cal / "l2_scores.jsonl")]
    assert models.count("ollama/qwen3-vl:8b-instruct") == 3
    assert models.count("ollama/qwen3-vl:32b") == 3


# ------------------------------------------- [94]/[112] one bad clip, no abort

def test_unjudgeable_clip_is_recorded_and_skipped_on_rerun(clips, tmp_path):
    cal = _cal(tmp_path, clips)
    broken = {"prompt_adherence": {"score": 9, "na": False, "reason": "x"}}
    with serve_comfy(fixture_paths=clips) as comfy, \
            serve_judge(scores_queue=[broken, broken]) as (judge, jurl):
        config, pipeline = _env(tmp_path, comfy.host, jurl)
        assert score_with_judge(str(cal), str(config),
                                pipeline_path=str(pipeline)) == 0
        assert judge.chat_calls == 2 + 2                 # try+retry, then 2 clean
        rows = _read_jsonl(cal / "l2_scores.jsonl")
        errors = [r for r in rows if r.get("error")]
        assert [r["clip_id"] for r in errors] == ["cal000"]
        assert sum(1 for r in rows if "dimensions" in r) == 2
        assert score_with_judge(str(cal), str(config),
                                pipeline_path=str(pipeline)) == 0
        assert judge.chat_calls == 4                     # error row honored
        assert judge.generate_calls == 1                 # second run had no wave


def test_main_tune_with_l2_records_the_judge_in_provenance(clips, tmp_path):
    """End to end through the CLI entry: score, tune, provenance names the judge."""
    from test_calibrate_tune import BASE_M
    from shortsloop.calibrate import tune as tune_mod
    from shortsloop.verdict import sha256_file

    kinds = ["moving"] * 6 + ["static"] * 6
    cal = _cal(tmp_path, clips, kinds=kinds)
    manifest = _read_jsonl(cal / "batch_manifest.jsonl")
    with open(cal / "metrics.jsonl", "w") as f, open(cal / "labels.jsonl", "w") as lab:
        for row, kind in zip(manifest, kinds):
            sha = sha256_file(row["clip_path"])
            m = dict(BASE_M, flow_mag_median=0.004 if kind == "moving" else 1e-5)
            f.write(json.dumps({"clip_id": row["clip_id"], "clip_sha256": sha,
                                "l1_impl": tune_mod.L1_IMPL, "metrics": m}) + "\n")
            verdict, classes = ("pass", []) if kind == "moving" else ("fail", ["static"])
            lab.write(json.dumps({"clip_id": row["clip_id"], "verdict": verdict,
                                  "classes": classes, "ts": 1,
                                  "clip_sha256": sha}) + "\n")
    with serve_comfy(fixture_paths=clips) as comfy, \
            serve_judge(model="qwen3-vl:32b", scores=GOOD_SCORES) as (judge, jurl):
        config, pipeline = _env(tmp_path, comfy.host, jurl, model="qwen3-vl:32b")
        assert main_tune(["--calibration", str(cal), "--with-l2",
                          "--config", str(config), "--pipeline", str(pipeline)]) == 0
    prov = yaml.safe_load((cal / "thresholds.proposed.yaml").read_text())["provenance"]
    assert prov["judge"] == {"model": "ollama/qwen3-vl:32b",
                             "model_digest": "sha256:fakedigest"}
    assert "L1+L2" in prov["test_scope"]


# ------------------------------- [review fix-calib-2] run lock, judge unload, policies

def test_with_l2_refuses_while_another_process_holds_the_run_lock(clips, tmp_path,
                                                                  capsys):
    """[integration-seams#7] --with-l2 loads the VLM: it takes the nightly runner's
    runs_dir/.shortsloop.lock and refuses (exit 2) with zero GPU contact — no /free,
    no judge call — while another process holds it."""
    cal = _cal(tmp_path, clips)
    with serve_comfy(fixture_paths=clips) as comfy, serve_judge() as (judge, jurl):
        config, pipeline = _env(tmp_path, comfy.host, jurl)
        held = lock.acquire(tmp_path / "runs")          # e.g. the nightly runner
        try:
            capsys.readouterr()
            assert score_with_judge(str(cal), str(config),
                                    pipeline_path=str(pipeline)) == 2
            assert "lock" in capsys.readouterr().out
            assert main_tune(["--calibration", str(cal), "--with-l2",
                              "--config", str(config),
                              "--pipeline", str(pipeline)]) == 2
            assert judge.chat_calls == 0 and comfy.free_calls == 0
            assert judge.generate_calls == 0
            assert not (cal / "thresholds.proposed.yaml").exists()
        finally:
            lock.release(held)
        assert score_with_judge(str(cal), str(config),
                                pipeline_path=str(pipeline)) == 0
        assert judge.chat_calls == 3
    lock.release(lock.acquire(tmp_path / "runs"))       # released after the wave


def test_with_l2_refuses_a_judge_it_could_not_unload(clips, tmp_path, capsys):
    """[finding 7] openai_compat without judge.unload_url: the VLM could not be
    evicted after the wave (hard rule 3) — refuse before /free or any judge call."""
    from fake_openai import serve_openai
    cal = _cal(tmp_path, clips)
    with serve_comfy(fixture_paths=clips) as comfy, serve_openai() as (srv, ourl):
        judge = {"adapter": "openai_compat", "base_url": ourl, "model": "qwen3-vl-8b"}
        config, pipeline = _env(tmp_path, comfy.host, None, judge=judge)
        capsys.readouterr()
        assert score_with_judge(str(cal), str(config),
                                pipeline_path=str(pipeline)) == 2
        assert "unload_url" in capsys.readouterr().out
        assert main_tune(["--calibration", str(cal), "--with-l2",
                          "--config", str(config), "--pipeline", str(pipeline)]) == 2
        assert srv.chat_calls == 0 and comfy.free_calls == 0
    assert not (cal / "l2_scores.jsonl").exists()


def test_with_l2_judges_with_the_nightly_judge_policy(clips, tmp_path):
    """Calibration must score clips exactly the way the nightly checker will:
    pipeline.yaml's judge retries/timeout_s are authoritative over config.yaml's
    (settings.effective_judge_cfg — what the runner hands the checker)."""
    cal = _cal(tmp_path, clips)
    broken = {"prompt_adherence": {"score": 9, "na": False, "reason": "x"}}
    with serve_comfy(fixture_paths=clips) as comfy, \
            serve_judge(scores_queue=[broken]) as (judge, jurl):
        cfg_judge = {"adapter": "ollama", "base_url": jurl,
                     "model": "qwen3-vl:8b-instruct", "timeout_s": 30, "retries": 3}
        config, pipeline = _env(tmp_path, comfy.host, jurl, judge=cfg_judge,
                                judge_policy={"retries": 0})
        assert score_with_judge(str(cal), str(config),
                                pipeline_path=str(pipeline)) == 0
        assert judge.chat_calls == 1 + 2     # nightly policy: no retry on cal000
    rows = _read_jsonl(cal / "l2_scores.jsonl")
    assert [r["clip_id"] for r in rows if r.get("error")] == ["cal000"]


def test_with_l2_refuses_malformed_pipeline_policies(clips, tmp_path, capsys):
    cal = _cal(tmp_path, clips)
    with serve_comfy(fixture_paths=clips) as comfy, serve_judge() as (judge, jurl):
        config, pipeline = _env(tmp_path, comfy.host, jurl)
        pipeline.write_text("policies:\n  judge: 5\n", encoding="utf-8")
        capsys.readouterr()
        assert score_with_judge(str(cal), str(config),
                                pipeline_path=str(pipeline)) == 2
        assert "policies" in capsys.readouterr().out
        assert judge.chat_calls == 0 and comfy.free_calls == 0
