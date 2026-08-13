"""Doctor tests against fakes: reachability, model presence, VRAM handoff,
and the two-call vision probe that catches a blind judge."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from fake_comfy import serve_comfy
from fake_judge import serve as serve_judge

from shortsloop.doctor import run_doctor

DATA = Path(__file__).parent / "data"

WORKFLOW_MODELS = {
    "diffusion_models": ["wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
                         "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"],
    "text_encoders": ["umt5_xxl_fp8_e4m3fn_scaled.safetensors"],
    "vae": ["wan_2.1_vae.safetensors"],
}


def _cfg(tmp_path, comfy_host, judge_url) -> Path:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({
        "comfy": {"host": comfy_host,
                  "workflow_t2v": str(DATA / "test_workflow.json")},
        "judge": {"adapter": "ollama", "base_url": judge_url,
                  "model": "qwen3-vl:8b-instruct", "timeout_s": 30},
        "rewrite": {"model": "qwen3:8b"},
        "paths": {"runs_dir": str(tmp_path)},
    }), encoding="utf-8")
    (tmp_path / "pipeline.yaml").write_text(yaml.safe_dump({
        "policies": {"disk_min_free_gb": 1,
                     "vram_handoff": {"free_min_gb": 20, "wait_timeout_s": 5}}}),
        encoding="utf-8")
    (tmp_path / "thresholds.yaml").write_text(
        "version: 't'\ncalibrated: false\nl1: {}\nl2: {floors: {}}\n", encoding="utf-8")
    return cfg


def _run(tmp_path, comfy, judge_url):
    cfg = _cfg(tmp_path, comfy.host, judge_url)
    code = run_doctor(str(cfg), str(tmp_path / "pipeline.yaml"),
                      str(tmp_path / "thresholds.yaml"), do_free=True,
                      out_path=None)
    snap = json.loads((tmp_path / "doctor.json").read_text(encoding="utf-8"))
    return code, snap


def _check(snap, name):
    return next(c for c in snap["checks"] if c["name"] == name)


def test_doctor_all_green(clips, tmp_path):
    with serve_comfy(fixture_paths=clips, models=WORKFLOW_MODELS) as comfy, \
         serve_judge() as (judge, jurl):
        judge.probe_answers = ["red", "blue"]         # judge really sees the images
        code, snap = _run(tmp_path, comfy, jurl)
    assert code == 0 and snap["ok"] is True
    assert _check(snap, "judge.vision")["ok"] is True
    assert _check(snap, "vram.handoff")["ok"] is True
    assert _check(snap, "models.diffusion_models")["ok"] is True
    assert _check(snap, "thresholds")["level"] == "WARN"   # uncalibrated = warn


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


def test_doctor_comfy_down(clips, tmp_path):
    with serve_judge() as (judge, jurl):
        judge.probe_answers = ["red", "blue"]
        cfg = _cfg(tmp_path, "127.0.0.1:9", jurl)     # nothing listens on :9
        code = run_doctor(str(cfg), str(tmp_path / "pipeline.yaml"),
                          str(tmp_path / "thresholds.yaml"), do_free=True,
                          out_path=None)
        snap = json.loads((tmp_path / "doctor.json").read_text(encoding="utf-8"))
    assert code == 2
    assert _check(snap, "comfy.server")["ok"] is False
