"""Doctor tests against fakes: reachability, model presence, VRAM handoff, the
two-call vision probe that catches a blind judge, the real 8-frame L2 dry run,
and fail-closed snapshots (a doctor that could not verify something, or crashed,
never leaves ok:true on disk)."""

from __future__ import annotations

import base64
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from conftest import write_thresholds
from fake_comfy import serve_comfy
from fake_judge import serve as serve_judge

from shortsloop import doctor, gpu, l2
from shortsloop.comfy import ComfyClient
from shortsloop.errors import InfraError
from shortsloop.doctor import run_doctor
from shortsloop.l2 import JUDGE_RESPONSE_SCHEMA

DATA = Path(__file__).parent / "data"
DEAD = "127.0.0.1:9"                                   # nothing listens on :9

WORKFLOW_MODELS = {
    "diffusion_models": ["wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
                         "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"],
    "text_encoders": ["umt5_xxl_fp8_e4m3fn_scaled.safetensors"],
    "vae": ["wan_2.1_vae.safetensors"],
}


def _cfg(tmp_path, comfy_host, judge_url, *, workflow=None, judge=None,
         rewrite=None, runs_dir=None, handoff=None) -> Path:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({
        "comfy": {"host": comfy_host,
                  "workflow_t2v": str(workflow or DATA / "test_workflow.json")},
        "judge": {"adapter": "ollama", "base_url": judge_url,
                  "model": "qwen3-vl:8b-instruct", "timeout_s": 30, **(judge or {})},
        "rewrite": rewrite if rewrite is not None else {"model": "qwen3:8b"},
        "paths": {"runs_dir": str(runs_dir or tmp_path)},
    }), encoding="utf-8")
    (tmp_path / "pipeline.yaml").write_text(yaml.safe_dump({
        "policies": {"disk_min_free_gb": 1,
                     "vram_handoff": handoff or {"free_min_gb": 20,
                                                 "wait_timeout_s": 5}}}),
        encoding="utf-8")
    write_thresholds(tmp_path / "thresholds.yaml", calibrated=False)
    return cfg


def _doctor(tmp_path, cfg, do_free=True):
    code = run_doctor(str(cfg), str(tmp_path / "pipeline.yaml"),
                      str(tmp_path / "thresholds.yaml"), do_free=do_free,
                      out_path=None)
    snap = json.loads((tmp_path / "doctor.json").read_text(encoding="utf-8"))
    return code, snap


def _run(tmp_path, comfy, judge_url, do_free=True, **cfg_over):
    return _doctor(tmp_path, _cfg(tmp_path, comfy.host, judge_url, **cfg_over),
                   do_free=do_free)


def _check(snap, name):
    return next(c for c in snap["checks"] if c["name"] == name)


def _green_snapshot(tmp_path) -> Path:
    p = tmp_path / "doctor.json"
    p.write_text(json.dumps({"ok": True, "checks": []}), encoding="utf-8")
    return p


def _spy_unload(monkeypatch, judge) -> list[int]:
    """Records how many judge chat calls had happened when the doctor unloaded."""
    seen: list[int] = []
    real = gpu.unload_llms

    def spy(*a, **k):
        seen.append(judge.chat_calls)
        return real(*a, **k)
    monkeypatch.setattr(gpu, "unload_llms", spy)
    return seen


def test_doctor_all_green(clips, tmp_path, monkeypatch):
    with serve_comfy(fixture_paths=clips, models=WORKFLOW_MODELS) as comfy, \
         serve_judge() as (judge, jurl):
        judge.probe_answers = ["red", "blue"]         # judge really sees the images
        unloads = _spy_unload(monkeypatch, judge)
        code, snap = _run(tmp_path, comfy, jurl)
    assert code == 0 and snap["ok"] is True and snap["status"] == "complete"
    for name in ("comfy.gpu", "workflow", "workflow.knobs", "models.diffusion_models",
                 "vram.handoff", "judge.vision", "judge.l2_dryrun", "judge.unload"):
        assert _check(snap, name)["ok"] is True, name
    assert _check(snap, "thresholds")["level"] == "WARN"   # uncalibrated = warn
    # hard rule 3: VLM loads only after /free was verified, and is unloaded after
    assert comfy.free_times[0] < judge.chat_times[0]
    assert judge.chat_calls == 3                       # 2 color probes + 1 L2 call
    assert unloads == [3] and judge.generate_calls >= 1


def test_l2_dryrun_is_the_real_8_frame_nightly_call(clips, tmp_path):
    """Plan §9: doctor validates the 8-image rubric call end to end (context fit)."""
    with serve_comfy(fixture_paths=clips, models=WORKFLOW_MODELS) as comfy, \
         serve_judge() as (judge, jurl):
        judge.probe_answers = ["red", "blue"]
        code, snap = _run(tmp_path, comfy, jurl)
        payload = judge.last_chat_payload               # the L2 dry-run call
    assert code == 0
    images = payload["messages"][1]["images"]
    assert len(images) == 8
    frame = cv2.imdecode(np.frombuffer(base64.b64decode(images[0]), np.uint8), 1)
    assert frame.shape[:2] == (1280, 720)              # nightly 720x1280, native res
    assert payload["format"] == JUDGE_RESPONSE_SCHEMA
    assert payload["messages"][0]["content"] == l2.SYSTEM_PROMPT
    assert "8 frames" in _check(snap, "judge.l2_dryrun")["detail"]


def test_l2_dryrun_fails_on_schema_invalid_reply(clips, tmp_path):
    with serve_comfy(fixture_paths=clips, models=WORKFLOW_MODELS) as comfy, \
         serve_judge(scenario="garbage") as (judge, jurl):
        judge.probe_answers = ["red", "blue"]         # color probe fine, rubric not
        code, snap = _run(tmp_path, comfy, jurl)
    assert code == 2 and snap["ok"] is False
    assert _check(snap, "judge.vision")["ok"] is True
    dry = _check(snap, "judge.l2_dryrun")
    assert dry["level"] == "FAIL" and "unusable" in dry["detail"]


def test_l2_dryrun_fails_on_timeout(clips, tmp_path, monkeypatch):
    with serve_comfy(fixture_paths=clips, models=WORKFLOW_MODELS) as comfy, \
         serve_judge() as (judge, jurl):
        judge.probe_answers = ["red", "blue"]
        real = l2.run_l2

        def slow_rubric(*a, **k):                    # only the 8-frame call is slow
            judge.sleep_s = 1.5
            return real(*a, **k)
        monkeypatch.setattr(l2, "run_l2", slow_rubric)
        code, snap = _run(tmp_path, comfy, jurl, judge={"timeout_s": 1})
    assert code == 2
    assert _check(snap, "judge.vision")["ok"] is True
    assert _check(snap, "judge.l2_dryrun")["level"] == "FAIL"


def test_doctor_catches_blind_judge(clips, tmp_path):
    with serve_comfy(fixture_paths=clips, models=WORKFLOW_MODELS) as comfy, \
         serve_judge() as (judge, jurl):
        judge.probe_answers = ["red", "red"]          # text-only guessing pattern
        code, snap = _run(tmp_path, comfy, jurl)
    assert code == 2 and snap["ok"] is False
    vision = _check(snap, "judge.vision")
    assert vision["ok"] is False
    assert "NOT see" in vision["detail"]


def test_doctor_catches_missing_models(clips, tmp_path):
    models = {**WORKFLOW_MODELS,
              "diffusion_models": ["some_other_model.safetensors"]}
    with serve_comfy(fixture_paths=clips, models=models) as comfy, \
         serve_judge() as (judge, jurl):
        judge.probe_answers = ["red", "blue"]
        code, snap = _run(tmp_path, comfy, jurl)
    assert code == 2
    missing = _check(snap, "models.diffusion_models")
    assert missing["ok"] is False and "MISSING" in missing["detail"]


def test_doctor_catches_vram_not_freeing(clips, tmp_path):
    with serve_comfy(fixture_paths=clips, models=WORKFLOW_MODELS,
                     never_frees=True) as comfy, \
         serve_judge() as (judge, jurl):
        judge.probe_answers = ["red", "blue"]
        code, snap = _run(tmp_path, comfy, jurl)
    assert code == 2
    assert _check(snap, "vram.handoff")["ok"] is False
    # hard rule 3: the VLM is never loaded beside a Wan that did not leave
    assert judge.chat_calls == 0
    assert "not run" in _check(snap, "judge.vision")["detail"]
    assert _check(snap, "judge.l2_dryrun")["level"] == "FAIL"


def test_doctor_comfy_down(clips, tmp_path):
    with serve_judge() as (judge, jurl):
        judge.probe_answers = ["red", "blue"]
        code, snap = _doctor(tmp_path, _cfg(tmp_path, DEAD, jurl))
    assert code == 2
    assert _check(snap, "comfy.server")["ok"] is False
    # free VRAM cannot be verified without ComfyUI -> the VLM is not loaded
    assert _check(snap, "vram.handoff")["level"] == "FAIL"
    assert judge.chat_calls == 0


def test_doctor_fails_on_off_shape_thresholds(clips, tmp_path):
    with serve_comfy(fixture_paths=clips, models=WORKFLOW_MODELS) as comfy, \
         serve_judge() as (judge, jurl):
        cfg = _cfg(tmp_path, comfy.host, jurl)
        (tmp_path / "thresholds.yaml").write_text(
            "version: 't'\ncalibrated: 'false'\nl1: {}\nl2: {floors: {}}\n")
        code, snap = _doctor(tmp_path, cfg)
    assert code != 0 and snap["ok"] is False
    thr = _check(snap, "thresholds")
    assert thr["ok"] is False and thr["level"] == "FAIL"


# ---- crash safety: a doctor run never leaves a stale green snapshot ----------------

def test_unparseable_config_overwrites_green_snapshot(tmp_path):
    _green_snapshot(tmp_path)
    cfg = tmp_path / "config.yaml"
    cfg.write_text("comfy: {host: [unclosed\n", encoding="utf-8")
    write_thresholds(tmp_path / "thresholds.yaml")
    code, snap = _doctor(tmp_path, cfg)
    assert code == 2 and snap["ok"] is False
    assert _check(snap, "config")["level"] == "FAIL"


def test_null_policies_with_dead_servers_fails_cleanly_and_runner_refuses(tmp_path):
    from shortsloop.runner import Runner
    _green_snapshot(tmp_path)
    cfg = _cfg(tmp_path, DEAD, f"http://{DEAD}")
    (tmp_path / "pipeline.yaml").write_text("policies:\n", encoding="utf-8")
    code, snap = _doctor(tmp_path, cfg)
    assert code == 2 and snap["ok"] is False and snap["status"] == "complete"
    runner = Runner(dispatch_path=tmp_path / "unused.md", config_path=cfg,
                    pipeline_path=tmp_path / "pipeline.yaml",
                    thresholds_path=tmp_path / "thresholds.yaml")
    assert runner._load() == 2


def test_exception_inside_doctor_ends_with_ok_false(tmp_path, monkeypatch):
    _green_snapshot(tmp_path)

    def boom(*a, **k):
        raise RuntimeError("synthetic crash")
    monkeypatch.setattr(doctor, "_check_binaries", boom)
    code, snap = _doctor(tmp_path, _cfg(tmp_path, DEAD, f"http://{DEAD}"))
    assert code == 2 and snap["ok"] is False
    assert any("synthetic crash" in c["detail"] and c["level"] == "FAIL"
               for c in snap["checks"])


def test_snapshot_is_invalidated_before_any_check_runs(tmp_path, monkeypatch):
    snap_path = _green_snapshot(tmp_path)
    seen = {}

    def peek(doc, *a, **k):
        seen.update(json.loads(snap_path.read_text(encoding="utf-8")))
        raise KeyboardInterrupt                      # operator hits Ctrl-C mid-run
    monkeypatch.setattr(doctor, "_check_binaries", peek)
    with pytest.raises(KeyboardInterrupt):
        _doctor(tmp_path, _cfg(tmp_path, DEAD, f"http://{DEAD}"))
    assert seen["ok"] is False and seen["status"] == "in progress"
    after = json.loads(snap_path.read_text(encoding="utf-8"))
    assert after["ok"] is False


# ---- hard rule 3 around the doctor's own judge calls -------------------------------

def test_no_free_is_fail_closed_and_never_loads_vlm_beside_wan(clips, tmp_path):
    # fake starts with Wan resident (4 GB free); --no-free must not load the VLM
    with serve_comfy(fixture_paths=clips, models=WORKFLOW_MODELS) as comfy, \
         serve_judge() as (judge, jurl):
        judge.probe_answers = ["red", "blue"]
        code, snap = _run(tmp_path, comfy, jurl, do_free=False)
    assert code == 2 and snap["ok"] is False
    handoff = _check(snap, "vram.handoff")
    assert handoff["level"] == "FAIL" and "--no-free" in handoff["detail"]
    assert comfy.free_calls == 0 and judge.chat_calls == 0


def test_no_free_on_an_empty_gpu_still_probes_but_snapshot_not_ok(clips, tmp_path):
    with serve_comfy(fixture_paths=clips, models=WORKFLOW_MODELS,
                     vram_free_gb=28.0) as comfy, \
         serve_judge() as (judge, jurl):
        judge.probe_answers = ["red", "blue"]
        code, snap = _run(tmp_path, comfy, jurl, do_free=False)
    assert comfy.free_calls == 0 and judge.chat_calls == 3
    assert _check(snap, "judge.vision")["ok"] is True
    assert code == 2 and snap["ok"] is False          # handoff itself unverified


def test_vlm_that_stays_resident_after_unload_fails(clips, tmp_path, monkeypatch):
    with serve_comfy(fixture_paths=clips, models=WORKFLOW_MODELS) as comfy, \
         serve_judge() as (judge, jurl):
        judge.probe_answers = ["red", "blue"]
        real = l2.run_l2

        def vlm_occupies_card(*a, **k):
            comfy.vram_free_gb = 4.0                  # VLM resident; unload no-op
            return real(*a, **k)
        monkeypatch.setattr(l2, "run_l2", vlm_occupies_card)
        code, snap = _run(tmp_path, comfy, jurl,
                          handoff={"free_min_gb": 20, "wait_timeout_s": 1})
    assert code == 2
    assert _check(snap, "judge.unload")["level"] == "FAIL"
    assert judge.generate_calls >= 1                  # the unload was requested


# ---- "could not verify" is a FAIL where the runner depends on it -------------------

def test_unlistable_model_folder_is_fail_not_warn(clips, tmp_path, monkeypatch):
    real = ComfyClient._http

    def no_models_api(self, method, path, *a, **k):
        if path.startswith("/models/"):
            raise InfraError("l1", f"ComfyUI HTTP 404 on {method} {path}")
        return real(self, method, path, *a, **k)
    monkeypatch.setattr(ComfyClient, "_http", no_models_api)
    with serve_comfy(fixture_paths=clips, models=WORKFLOW_MODELS) as comfy, \
         serve_judge() as (judge, jurl):
        judge.probe_answers = ["red", "blue"]
        code, snap = _run(tmp_path, comfy, jurl)
    assert code == 2 and snap["ok"] is False
    assert _check(snap, "models.diffusion_models")["level"] == "FAIL"


def test_unrecognized_loader_classes_fail(clips, tmp_path):
    wf = json.loads((DATA / "test_workflow.json").read_text(encoding="utf-8"))
    for node in wf.values():
        if node["class_type"] == "UNETLoader":
            node["class_type"] = "UnetLoaderGGUF"     # model presence unverifiable
    gguf = tmp_path / "gguf_workflow.json"
    gguf.write_text(json.dumps(wf), encoding="utf-8")
    with serve_comfy(fixture_paths=clips, models=WORKFLOW_MODELS) as comfy, \
         serve_judge() as (judge, jurl):
        judge.probe_answers = ["red", "blue"]
        code, snap = _run(tmp_path, comfy, jurl, workflow=gguf)
    assert code == 2
    loaders = _check(snap, "models.loaders")
    assert loaders["level"] == "FAIL" and "UnetLoaderGGUF" in loaders["detail"]


def test_no_gpu_device_fails(clips, tmp_path, monkeypatch):
    real = ComfyClient._http

    def cpu_only(self, method, path, *a, **k):
        if path == "/system_stats":
            return {"system": {"comfyui_version": "fake"}, "devices": []}
        return real(self, method, path, *a, **k)
    monkeypatch.setattr(ComfyClient, "_http", cpu_only)
    with serve_comfy(fixture_paths=clips, models=WORKFLOW_MODELS) as comfy, \
         serve_judge() as (judge, jurl):
        judge.probe_answers = ["red", "blue"]
        code, snap = _run(tmp_path, comfy, jurl, do_free=True)
    assert code == 2
    assert _check(snap, "comfy.gpu")["level"] == "FAIL"
    assert judge.chat_calls == 0                      # no verified VRAM, no VLM


def test_workflow_with_unpatchable_knobs_fails(clips, tmp_path):
    wf = json.loads((DATA / "test_workflow.json").read_text(encoding="utf-8"))
    for node in wf.values():
        if "noise_seed" in node.get("inputs", {}):
            node["inputs"]["noise_seed"] = ["99", 0]  # linked: re-rolls repeat a seed
    wf["99"] = {"class_type": "PrimitiveInt", "inputs": {"value": 5}}
    linked = tmp_path / "linked.json"
    linked.write_text(json.dumps(wf), encoding="utf-8")
    with serve_comfy(fixture_paths=clips, models=WORKFLOW_MODELS) as comfy, \
         serve_judge() as (judge, jurl):
        judge.probe_answers = ["red", "blue"]
        code, snap = _run(tmp_path, comfy, jurl, workflow=linked)
    assert code == 2
    knobs = _check(snap, "workflow.knobs")
    assert knobs["level"] == "FAIL" and "seed" in knobs["detail"]


def test_disk_measured_on_runs_dir_filesystem_even_if_not_created(clips, tmp_path):
    with serve_comfy(fixture_paths=clips, models=WORKFLOW_MODELS) as comfy, \
         serve_judge() as (judge, jurl):
        judge.probe_answers = ["red", "blue"]
        code, snap = _run(tmp_path, comfy, jurl, runs_dir=tmp_path / "big" / "runs")
    assert f"at {tmp_path.resolve()} (" in _check(snap, "disk")["detail"]


# ---- rewrite model, config skeleton, cron PATH -------------------------------------

def test_rewrite_model_checked_on_the_rewrite_server(clips, tmp_path):
    """The judge runs on llama-server (openai_compat); the rewrite model lives on a
    separate Ollama. Doctor must ask THAT server, and never stay silent."""
    with serve_judge(model="qwen3:8b") as (_rw, rurl):
        code, snap = _doctor(tmp_path, _cfg(
            tmp_path, DEAD, f"http://{DEAD}",
            judge={"adapter": "openai_compat"},
            rewrite={"model": "qwen3:8b", "base_url": rurl}))
    rw = _check(snap, "rewrite.model")
    assert rw["ok"] is True and rurl in rw["detail"]

    code, snap = _doctor(tmp_path, _cfg(
        tmp_path, DEAD, f"http://{DEAD}", judge={"adapter": "openai_compat"},
        rewrite={"model": "qwen3:8b", "base_url": f"http://{DEAD}"}))
    rw = _check(snap, "rewrite.model")
    assert rw["level"] == "WARN" and DEAD in rw["detail"]


def test_rewrite_model_bare_name_matches_latest_tag(tmp_path):
    """Ollama stores `ollama pull qwen3` as qwen3:latest — a bare config name is
    installed, not a false 'NOT installed' warning."""
    with serve_judge(model="qwen3:latest") as (_rw, rurl):
        code, snap = _doctor(tmp_path, _cfg(
            tmp_path, DEAD, f"http://{DEAD}",
            rewrite={"model": "qwen3", "base_url": rurl}))
    assert _check(snap, "rewrite.model")["ok"] is True


def test_missing_config_writes_skeleton_and_fails(tmp_path):
    write_thresholds(tmp_path / "thresholds.yaml")
    cfg = tmp_path / "config.yaml"
    code, snap = _doctor(tmp_path, cfg)
    assert code == 2 and snap["ok"] is False
    assert cfg.is_file()
    skeleton = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    example = yaml.safe_load((Path(__file__).parent.parent / "config.example.yaml")
                             .read_text(encoding="utf-8"))
    assert skeleton == example
    assert "skeleton" in _check(snap, "config")["detail"]
    # a second run never overwrites the (now user-owned) file
    cfg.write_text("comfy: {host: mine}\n", encoding="utf-8")
    _doctor(tmp_path, cfg)
    assert cfg.read_text(encoding="utf-8") == "comfy: {host: mine}\n"


def test_binaries_record_absolute_paths_and_cron_hint(tmp_path, capsys):
    write_thresholds(tmp_path / "thresholds.yaml")
    code, snap = _doctor(tmp_path, _cfg(tmp_path, DEAD, f"http://{DEAD}"))
    for binary in ("ffmpeg", "ffprobe"):
        where = shutil.which(binary)
        assert where in _check(snap, binary)["detail"]
        assert snap["tools"][binary] == where
    out = capsys.readouterr().out
    assert "cron" in out and "PATH=" in out
