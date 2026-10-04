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
from fake_openai import serve_openai

from shortsloop import doctor, gpu, l2
from shortsloop.comfy import ComfyClient
from shortsloop.errors import InfraError
from shortsloop.doctor import run_doctor
from shortsloop.judge.base import RetryableJudgeError
from shortsloop.l2 import JUDGE_RESPONSE_SCHEMA

DATA = Path(__file__).parent / "data"
DEAD = "127.0.0.1:9"                                   # nothing listens on :9

WORKFLOW_MODELS = {
    "diffusion_models": ["wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
                         "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"],
    "text_encoders": ["umt5_xxl_fp8_e4m3fn_scaled.safetensors"],
    "vae": ["wan_2.1_vae.safetensors"],
}

# The one 32 GB card both fakes share: idle after ComfyUI's /free = 28 GB (the
# FakeComfy default); an 8B VLM loaded for the probes leaves 22 GB — MORE than the
# judge-direction free_min_gb (20), so only the stricter judge->generation
# threshold gen_free_min_gb (26) can tell a resident VLM from an idle card.
VLM_LOADED_FREE_GB = 22.0
HANDOFF = {"free_min_gb": 20, "gen_free_min_gb": 26, "wait_timeout_s": 5}


def _cfg(tmp_path, comfy_host, judge_url, *, workflow=None, judge=None,
         rewrite=None, runs_dir=None, handoff=None, judge_policy=None) -> Path:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({
        "comfy": {"host": comfy_host,
                  "workflow_t2v": str(workflow or DATA / "test_workflow.json")},
        "judge": {"adapter": "ollama", "base_url": judge_url,
                  "model": "qwen3-vl:8b-instruct", "timeout_s": 30, **(judge or {})},
        "rewrite": rewrite if rewrite is not None else {"model": "qwen3:8b"},
        "paths": {"runs_dir": str(runs_dir or tmp_path)},
    }), encoding="utf-8")
    policies = {"disk_min_free_gb": 1, "vram_handoff": handoff or dict(HANDOFF)}
    if judge_policy is not None:                 # what the NIGHTLY checker uses
        policies["judge"] = judge_policy
    (tmp_path / "pipeline.yaml").write_text(yaml.safe_dump({"policies": policies}),
                                            encoding="utf-8")
    write_thresholds(tmp_path / "thresholds.yaml", calibrated=False)
    return cfg


def _vlm_on_card(comfy, judge, free_gb: float = VLM_LOADED_FREE_GB) -> None:
    """Both fakes share one card. A vision call loads the VLM (a separate process):
    free VRAM drops to `free_gb` and stays there until ComfyUI's next /free, whose
    outcome `free_results` scripts — the default (28 GB, idle) models a VLM that
    really left after its unload; a scripted low value models one that did not
    (ComfyUI's /free cannot evict another process's model)."""
    real = judge.content

    def content():
        comfy.vram_free_gb = min(comfy.vram_free_gb, free_gb)
        return real()
    judge.content = content


def _wrap_replies(judge, fmt: str) -> None:
    """Every judge reply becomes fmt with BODY replaced by the real reply."""
    real = judge.content
    judge.content = lambda: fmt.replace("BODY", real())


def _slow_rubric(monkeypatch, judge, sleep_s: float) -> None:
    """Only the 8-frame rubric call is slow (the color probes stay fast)."""
    real = l2.run_l2

    def slow(*a, **k):
        judge.sleep_s = sleep_s
        return real(*a, **k)
    monkeypatch.setattr(l2, "run_l2", slow)


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
        _vlm_on_card(comfy, judge)
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
    # the judge->generation handoff is the runner's: unload, /free, verify against
    # gpu.gen_free_min_gb — with BOTH readings on record (loaded vs. unloaded)
    assert comfy.free_calls == 2 and comfy.free_times[1] > judge.chat_times[-1]
    unload = _check(snap, "judge.unload")
    assert unload["vram"] == {"loaded_free_gb": 22.0, "unloaded_free_gb": 28.0,
                              "gen_free_min_gb": 26.0}
    assert "22.0" in unload["detail"] and "28.0" in unload["detail"]


def test_l2_dryrun_is_the_real_8_frame_nightly_call(clips, tmp_path):
    """Plan §9: doctor validates the 8-image rubric call end to end (context fit)."""
    with serve_comfy(fixture_paths=clips, models=WORKFLOW_MODELS) as comfy, \
         serve_judge() as (judge, jurl):
        judge.probe_answers = ["red", "blue"]
        _vlm_on_card(comfy, judge)
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


@pytest.mark.parametrize("config_timeout, pipeline_timeout, ok", [
    (30, 1, False),    # config.yaml generous, the nightly's pipeline.yaml is not
    (1, 30, True),     # config.yaml tight, but the nightly checker never uses it
])
def test_l2_dryrun_is_timed_against_pipeline_yaml_judge_timeout(
        clips, tmp_path, monkeypatch, config_timeout, pipeline_timeout, ok):
    """The nightly checker's judge timeout/retries come from pipeline.yaml (plan §2.8
    amended; settings.effective_judge_cfg). Doctor must certify THOSE values and
    send the operator to that file — config.yaml's judge.timeout_s only applies to
    a standalone `shortsloop-check`."""
    with serve_comfy(fixture_paths=clips, models=WORKFLOW_MODELS) as comfy, \
         serve_judge() as (judge, jurl):
        judge.probe_answers = ["red", "blue"]
        _slow_rubric(monkeypatch, judge, 1.5)
        code, snap = _run(tmp_path, comfy, jurl, judge={"timeout_s": config_timeout},
                          judge_policy={"timeout_s": pipeline_timeout, "retries": 0})
    assert _check(snap, "judge.vision")["ok"] is True
    dry = _check(snap, "judge.l2_dryrun")
    assert dry["ok"] is ok, dry["detail"]
    assert "pipeline.yaml policies.judge.timeout_s" in dry["detail"]
    if not ok:
        assert code == 2


def test_l2_dryrun_uses_pipeline_yaml_retries(clips, tmp_path):
    with serve_comfy(fixture_paths=clips, models=WORKFLOW_MODELS) as comfy, \
         serve_judge(scenario="garbage") as (judge, jurl):
        judge.probe_answers = ["red", "blue"]         # color probe fine, rubric not
        code, snap = _run(tmp_path, comfy, jurl, judge={"retries": 3},
                          judge_policy={"retries": 0})
    assert _check(snap, "judge.l2_dryrun")["level"] == "FAIL"
    assert judge.chat_calls == 2 + 1     # 2 color probes + ONE rubric try (retries 0)


def test_latency_warning_points_at_pipeline_yaml(clips, tmp_path, monkeypatch):
    with serve_comfy(fixture_paths=clips, models=WORKFLOW_MODELS) as comfy, \
         serve_judge() as (judge, jurl):
        judge.probe_answers = ["red", "blue"]
        _slow_rubric(monkeypatch, judge, 1.6)
        code, snap = _run(tmp_path, comfy, jurl, judge={"timeout_s": 600},
                          judge_policy={"timeout_s": 3})
    assert _check(snap, "judge.l2_dryrun")["ok"] is True
    warn = _check(snap, "judge.latency")
    assert warn["level"] == "WARN"
    assert "pipeline.yaml policies.judge.timeout_s" in warn["detail"]


@pytest.mark.parametrize("policies", [{"judge": 5}, {"vram_handoff": [20]}])
def test_malformed_pipeline_yaml_is_a_fail_not_a_crash(clips, tmp_path, policies):
    """The runner refuses this pipeline.yaml (settings.load_policies); doctor must
    say so as a FAIL — not report it loaded, not crash a later check group."""
    with serve_comfy(fixture_paths=clips, models=WORKFLOW_MODELS) as comfy, \
         serve_judge() as (judge, jurl):
        judge.probe_answers = ["red", "blue"]
        cfg = _cfg(tmp_path, comfy.host, jurl)
        (tmp_path / "pipeline.yaml").write_text(
            yaml.safe_dump({"policies": policies}), encoding="utf-8")
        code, snap = _doctor(tmp_path, cfg)
    assert code == 2 and snap["ok"] is False and snap["status"] == "complete"
    pipe = _check(snap, "pipeline")
    assert pipe["level"] == "FAIL" and "must be a mapping" in pipe["detail"]
    assert not [c["name"] for c in snap["checks"] if c["name"].endswith(".crash")]


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
    # unload is a no-op and ComfyUI's /free cannot evict another process's VLM
    with serve_comfy(fixture_paths=clips, models=WORKFLOW_MODELS,
                     free_results=[28.0, 4.0]) as comfy, \
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


def test_vlm_resident_after_unload_fails_against_gen_free_min_gb(clips, tmp_path):
    """The runner's judge->generation check uses gpu.gen_free_min_gb (26 here), not
    free_min_gb (20). A VLM still leaving 22 GB after its unload clears 20 but would
    halt the night at 26 — doctor must verify against the runner's threshold."""
    with serve_comfy(fixture_paths=clips, models=WORKFLOW_MODELS,
                     free_results=[28.0, VLM_LOADED_FREE_GB]) as comfy, \
         serve_judge() as (judge, jurl):
        judge.probe_answers = ["red", "blue"]
        _vlm_on_card(comfy, judge)
        code, snap = _run(tmp_path, comfy, jurl,
                          handoff={**HANDOFF, "wait_timeout_s": 1})
    assert code == 2 and snap["ok"] is False
    unload = _check(snap, "judge.unload")
    assert unload["level"] == "FAIL" and "26" in unload["detail"]
    assert unload["vram"]["unloaded_free_gb"] == 22.0
    assert judge.generate_calls >= 1                  # the unload was requested


def test_resident_vlm_that_clears_the_generation_threshold_fails(clips, tmp_path):
    """Shipped-default shape (no gen_free_min_gb): an 8B VLM still loaded leaves
    22 GB free, more than free_min_gb 20, so the runner's judge->generation check
    could not tell a resident judge from an idle card. Doctor reads free VRAM while
    the VLM is loaded and FAILs, printing both readings and a value between them."""
    with serve_comfy(fixture_paths=clips, models=WORKFLOW_MODELS) as comfy, \
         serve_judge() as (judge, jurl):
        judge.probe_answers = ["red", "blue"]
        _vlm_on_card(comfy, judge)
        code, snap = _run(tmp_path, comfy, jurl,
                          handoff={"free_min_gb": 20, "wait_timeout_s": 1})
    assert code == 2 and snap["ok"] is False
    unload = _check(snap, "judge.unload")
    assert unload["level"] == "FAIL"
    assert unload["vram"] == {"loaded_free_gb": 22.0, "unloaded_free_gb": 28.0,
                              "gen_free_min_gb": 20.0}
    assert "22.0" in unload["detail"] and "28.0" in unload["detail"]
    assert "vram_handoff.gen_free_min_gb: 25" in unload["detail"]


def test_unload_check_needs_a_reading_with_the_vlm_loaded(clips, tmp_path):
    """No judge reply = no proof the VLM was in VRAM when doctor read it: the
    loaded-vs-idle comparison is unverified, which is a FAIL — never an OK."""
    with serve_comfy(fixture_paths=clips, models=WORKFLOW_MODELS) as comfy, \
         serve_judge(sleep_s=1.5) as (judge, jurl):  # every judge call times out
        code, snap = _run(tmp_path, comfy, jurl,
                          judge_policy={"timeout_s": 1, "retries": 0})
    assert code == 2
    unload = _check(snap, "judge.unload")
    assert unload["level"] == "FAIL" and "not verified" in unload["detail"]


def test_openai_compat_without_unload_url_fails_before_loading_the_vlm(clips,
                                                                      tmp_path):
    """settings.judge_unload_problem: such a judge cannot be evicted before a Wan
    wave (the runner refuses it). Doctor FAILs it at once and never loads it."""
    with serve_comfy(fixture_paths=clips, models=WORKFLOW_MODELS) as comfy, \
         serve_openai() as (srv, ourl):
        code, snap = _doctor(tmp_path, _cfg(tmp_path, comfy.host, ourl,
                                            judge={"adapter": "openai_compat"},
                                            rewrite={"enabled": False}))
    assert code == 2 and snap["ok"] is False
    unload = _check(snap, "judge.unload")
    assert unload["level"] == "FAIL" and "unload_url" in unload["detail"]
    assert srv.chat_calls == 0                         # never load what can't leave
    assert "not run" in _check(snap, "judge.vision")["detail"]
    assert _check(snap, "judge.l2_dryrun")["level"] == "FAIL"


def test_card_smaller_than_the_generation_threshold_fails(clips, tmp_path):
    with serve_comfy(fixture_paths=clips, models=WORKFLOW_MODELS) as comfy, \
         serve_judge() as (judge, jurl):
        judge.probe_answers = ["red", "blue"]
        code, snap = _run(tmp_path, comfy, jurl,
                          handoff={"free_min_gb": 20, "gen_free_min_gb": 40,
                                   "wait_timeout_s": 1})
    assert code == 2
    card = _check(snap, "comfy.gpu")
    assert card["level"] == "FAIL" and "gen_free_min_gb" in card["detail"]


# ---- the vision probe parses replies exactly like the nightly L2 parser ------------

class _ReplyAdapter:
    """Answers the two color probes correctly, wrapped in `fmt` (BODY = the JSON)."""
    name, model = "stub", "stub-vl"

    def __init__(self, fmt: str):
        self.fmt, self.calls = fmt, 0

    def judge(self, frames, system, user, schema, timeout_s):
        color = ("red", "blue")[self.calls]
        self.calls += 1
        return self.fmt.replace("BODY", json.dumps({"dominant_color": color}))


@pytest.mark.parametrize("fmt", [
    "BODY",
    "```json\nBODY\n```",
    "<think>the square is red</think>\nBODY",
    "<think>checking</think>\n```json\nBODY\n```",
    "the image is mostly red</think>BODY",       # opening tag lived in the template
    "Sure! Here is the answer: BODY",            # prose around the JSON
    "BODY\nActually, I am not sure.",            # JSON, then a correction
    "<think>still reasoning... BODY",            # unterminated reasoning
])
def test_vision_probe_parses_exactly_like_the_nightly_l2_parser(fmt):
    try:
        l2._load_json(fmt.replace("BODY", '{"dominant_color": "red"}'))
        nightly_accepts = True
    except RetryableJudgeError:
        nightly_accepts = False
    doc = doctor.Doctor()
    doctor._probe_vision(doc, _ReplyAdapter(fmt), {"timeout_s": 30})
    vision = doc.checks[-1]
    assert vision["name"] == "judge.vision"
    assert vision["ok"] is nightly_accepts, vision["detail"]
    if not nightly_accepts:            # a parse failure is not a broken sidecar
        assert "switch judge.adapter" not in vision["detail"]


def test_doctor_green_for_a_judge_whose_wrappers_the_nightly_tolerates(clips,
                                                                      tmp_path):
    with serve_comfy(fixture_paths=clips, models=WORKFLOW_MODELS) as comfy, \
         serve_judge() as (judge, jurl):
        judge.probe_answers = ["red", "blue"]
        _wrap_replies(judge, "<think>checking the frames</think>\n```json\nBODY\n```")
        _vlm_on_card(comfy, judge)
        code, snap = _run(tmp_path, comfy, jurl)
    assert _check(snap, "judge.vision")["ok"] is True
    assert _check(snap, "judge.l2_dryrun")["ok"] is True
    assert code == 0 and snap["ok"] is True


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
