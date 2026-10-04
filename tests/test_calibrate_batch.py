"""calibrate-batch: off-prompt ground truth, VRAM handoff before Wan (hard rule 3),
queue hygiene and interrupt safety (hard rule 4). Fake ComfyUI + fake Ollama."""

from __future__ import annotations

import random
from pathlib import Path

import pytest
import yaml

from fake_comfy import serve_comfy
from fake_judge import serve as serve_judge

from shortsloop.calibrate.batchgen import N_SWAPS, plan_swaps, run_batch
from shortsloop.calibrate.prompts import GOOD, build_slate
from shortsloop.comfy import ComfyClient
from shortsloop.state import _read_jsonl

DATA = Path(__file__).parent / "data"


def _env(tmp_path, comfy_host, judge_url=None, *, timeout_s=60, free_min_gb=20,
         wait_timeout_s=2):
    cfg = {"comfy": {"host": comfy_host,
                     "workflow_t2v": str(DATA / "test_workflow.json")}}
    if judge_url:
        cfg["judge"] = {"adapter": "ollama", "base_url": judge_url,
                        "model": "qwen3-vl:8b-instruct"}
        cfg["rewrite"] = {"enabled": True, "model": "fake-text-model",
                          "base_url": judge_url}
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    pipeline = tmp_path / "pipeline.yaml"
    pipeline.write_text(yaml.safe_dump({"policies": {
        "comfy": {"timeout_s": timeout_s, "poll_s": 1},
        "vram_handoff": {"free_min_gb": free_min_gb,
                         "wait_timeout_s": wait_timeout_s}}}), encoding="utf-8")
    return config, pipeline


# ------------------------------------------------- [81] off-prompt ground truth

@pytest.mark.parametrize("count", list(range(24, 61)))
def test_swaps_always_pair_two_different_good_prompts(count):
    rows = [{**r, "clip_path": f"/clips/{r['clip_id']}.mp4", "seed": 1}
            for r in build_slate(count)]
    swaps = plan_swaps(rows)
    assert len(swaps) == N_SWAPS
    for s in swaps:
        assert s["prompt"] != s["gen_prompt"], s["clip_id"]   # wrong prompt on purpose
        assert s["prompt"] in GOOD and s["gen_prompt"] in GOOD
        src = next(r for r in rows if r["clip_id"] == s["source_clip_id"])
        assert s["clip_path"] == src["clip_path"] and s["gen_prompt"] == src["prompt"]
    # four swaps exercise four different source prompts
    assert len({s["gen_prompt"] for s in swaps}) == N_SWAPS


# ------------------------------------------ [89] VRAM handoff before first Wan job

def test_batch_unloads_judge_and_verifies_vram_before_first_generation(clips, tmp_path):
    with serve_comfy(fixture_paths=clips) as comfy, serve_judge() as (judge, jurl):
        config, pipeline = _env(tmp_path, comfy.host, jurl)
        out = tmp_path / "cal"
        assert run_batch(str(config), str(out), count=2, rng=random.Random(1),
                         pipeline_path=str(pipeline)) == 0
        assert judge.generate_calls == 2          # judge VLM + rewrite LLM unloaded
        assert comfy.free_calls >= 1              # ComfyUI caches freed + verified
        assert len(comfy.submissions) == 2 and comfy.violations == 0


def test_batch_refuses_to_load_wan_while_vram_is_occupied(clips, tmp_path):
    """A VLM still resident (VRAM never comes free) -> infra stop, zero Wan jobs."""
    with serve_comfy(fixture_paths=clips, never_frees=True) as comfy, \
            serve_judge() as (judge, jurl):
        config, pipeline = _env(tmp_path, comfy.host, jurl, wait_timeout_s=1)
        out = tmp_path / "cal"
        assert run_batch(str(config), str(out), count=2, rng=random.Random(1),
                         pipeline_path=str(pipeline)) == 3
        assert comfy.submissions == []
        assert _read_jsonl(out / "batch_manifest.jsonl") == []


# --------------------------------------------------- [88] queue hygiene (rule 4)

def test_batch_never_submits_beside_a_job_already_in_the_queue(clips, tmp_path):
    orphan = {"orphan01": {"scenario": {"hang": True}, "polls": 0, "done": False,
                           "info": {}, "order": -1}}
    with serve_comfy(fixture_paths=clips, preloaded_jobs=orphan) as comfy:
        config, pipeline = _env(tmp_path, comfy.host, timeout_s=2)
        assert run_batch(str(config), str(tmp_path / "cal"), count=1,
                         rng=random.Random(1), pipeline_path=str(pipeline)) == 3
        assert comfy.submissions == [] and comfy.violations == 0


def test_timed_out_job_is_removed_before_the_next_submit(clips, tmp_path):
    with serve_comfy(fixture_paths=clips,
                     scenarios=[{"hang": True}, {"fixture": "moving"}]) as comfy:
        config, pipeline = _env(tmp_path, comfy.host, timeout_s=2)
        out = tmp_path / "cal"
        assert run_batch(str(config), str(out), count=2, rng=random.Random(1),
                         pipeline_path=str(pipeline)) == 0
        assert comfy.interrupt_calls == 1          # the hung job was stopped …
        assert comfy.violations == 0               # … before the next submission
        gen = [r for r in _read_jsonl(out / "batch_manifest.jsonl")
               if r["kind"] != "swap"]
        assert [r["clip_id"] for r in gen] == ["cal002"]   # cal001 retried next run


class _CtrlCDuringWait(ComfyClient):
    def wait(self, prompt_id, out_dir):
        raise KeyboardInterrupt


def test_ctrl_c_stops_the_in_flight_job_and_rerun_is_clean(clips, tmp_path):
    with serve_comfy(fixture_paths=clips, scenarios=[{"hang": True}]) as comfy:
        config, pipeline = _env(tmp_path, comfy.host, timeout_s=10)
        out = tmp_path / "cal"
        client = _CtrlCDuringWait(host=comfy.host,
                                  workflow=str(DATA / "test_workflow.json"),
                                  timeout_s=10, poll_s=1)
        assert run_batch(str(config), str(out), count=1, rng=random.Random(1),
                         comfy=client, pipeline_path=str(pipeline)) == 130
        assert comfy.interrupt_calls == 1 and comfy.unfinished() == []

        # documented "re-run to resume": no second job beside an orphan
        assert run_batch(str(config), str(out), count=1, rng=random.Random(2),
                         pipeline_path=str(pipeline)) == 0
        assert comfy.violations == 0 and len(comfy.submissions) == 2
        assert [r["clip_id"] for r in _read_jsonl(out / "batch_manifest.jsonl")] \
            == ["cal001"]
