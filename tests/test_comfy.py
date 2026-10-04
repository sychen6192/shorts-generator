"""ComfyUI client primitives behind hard rules 3+4, against the fake server."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fake_comfy import serve_comfy

from shortsloop.comfy import ComfyClient, GenerationFailed
from shortsloop.errors import InfraError

DATA = Path(__file__).parent / "data"
WF = DATA / "test_workflow.json"
GEN = dict(width=480, height=832, length=49, fps=16.0)


def client(comfy, **kw):
    return ComfyClient(host=comfy.host, workflow=WF, timeout_s=kw.pop("timeout_s", 30),
                       poll_s=1, **kw)


def test_hung_job_is_interrupted_and_verified_gone(clips, tmp_path):
    with serve_comfy(fixture_paths=clips, scenarios=[{"hang": True}]) as comfy:
        c = client(comfy, timeout_s=2)
        pid = c.submit(prompt="p", seed=1, **GEN)["prompt_id"]
        with pytest.raises(GenerationFailed) as ei:
            c.wait(pid, tmp_path)
        assert ei.value.kind == "timeout"
        assert c.job_state(pid) == "running"
        c.ensure_gone(pid, timeout_s=5, sleep=lambda s: None)
        assert c.job_state(pid) in ("finished", "unknown")
        assert comfy.interrupt_calls == 1
        c.wait_queue_empty(timeout_s=1, sleep=lambda s: None)


def test_pending_job_is_deleted_not_interrupted(clips):
    with serve_comfy(fixture_paths=clips,
                     scenarios=[{"hang": True}, {"fixture": "moving"}]) as comfy:
        c = client(comfy)
        first = c.submit(prompt="p", seed=1, **GEN)["prompt_id"]
        second = c.submit(prompt="p", seed=2, **GEN)["prompt_id"]
        assert c.job_state(second) == "pending"
        c.ensure_gone(second, timeout_s=5, sleep=lambda s: None)
        assert comfy.deleted == [second] and comfy.interrupt_calls == 0
        assert c.job_state(first) == "running"          # someone else's job untouched
        with pytest.raises(InfraError):
            c.wait_queue_empty(timeout_s=0, sleep=lambda s: None)


def test_unknown_job_after_restart_is_reported_unknown(clips):
    with serve_comfy(fixture_paths=clips) as comfy:
        assert client(comfy).job_state("never-seen") == "unknown"


def test_vram_free_excludes_torch_cache(clips):
    """Memory torch has cached is not free for another process (the judge)."""
    with serve_comfy(fixture_paths=clips, vram_free_gb=28.0, torch_cache_gb=6.0) as comfy:
        assert client(comfy).vram_free_gb() == pytest.approx(22.0)


def test_wait_selects_the_video_and_drops_other_outputs(clips, tmp_path):
    with serve_comfy(fixture_paths=clips,
                     scenarios=[{"fixture": "moving", "extra_png": True}]) as comfy:
        c = client(comfy)
        pid = c.submit(prompt="p", seed=1, **GEN)["prompt_id"]
        res = c.wait(pid, tmp_path)
        assert [Path(f).suffix for f in res["files"]] == [".mp4"]
        assert sorted(p.suffix for p in tmp_path.iterdir()) == [".mp4"]


def test_submit_problems_are_infra_not_clip_failures(clips, tmp_path):
    ui_format = tmp_path / "ui.json"
    ui_format.write_text(json.dumps({"nodes": [], "links": []}))
    with serve_comfy(fixture_paths=clips) as comfy:
        with pytest.raises(InfraError):
            ComfyClient(host=comfy.host, workflow=ui_format).submit(
                prompt="p", seed=1, **GEN)
        with pytest.raises(InfraError):
            ComfyClient(host=comfy.host, workflow=tmp_path / "missing.json").submit(
                prompt="p", seed=1, **GEN)
    assert comfy.submissions == []


def test_validate_workflow_requires_every_reroll_knob(tmp_path):
    """A seed wired from a primitive node is silently NOT patched by the vendored
    client — re-rolls would repeat the same seed. Refuse at load time."""
    wf = json.loads(WF.read_text())
    assert ComfyClient(host="127.0.0.1:9", workflow=WF).validate_workflow(**GEN) == []
    for node in wf.values():
        if "noise_seed" in node.get("inputs", {}):
            node["inputs"]["noise_seed"] = ["99", 0]               # linked input
    wf["99"] = {"class_type": "PrimitiveInt", "inputs": {"value": 5}}
    linked = tmp_path / "linked.json"
    linked.write_text(json.dumps(wf))
    problems = ComfyClient(host="127.0.0.1:9", workflow=linked).validate_workflow(**GEN)
    assert any("seed" in p for p in problems)
