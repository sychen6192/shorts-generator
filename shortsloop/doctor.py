"""`shortsloop doctor` — executable environment discovery for the workstation
(docs/plan.md D1). The kickoff's "discover, do not assume" happens HERE, on the
machine with the GPU: every value the pipeline depends on is verified and the
result written to doctor.json next to the config. The nightly runner refuses to
start without a passing snapshot (fail closed at 22:00, not 3 a.m.).

Checks: ffmpeg/ffprobe · ComfyUI reachable + VRAM + queue · workflow is
API-format and its model files exist on the server · VRAM handoff actually frees
(/free + /system_stats, optional) · judge reachable + model installed + a real
two-image VISION smoke test with strict-JSON output (a text-only model that
cannot see the images has a 1-in-9 chance of passing) · rewrite model (warn) ·
disk headroom · thresholds/pipeline files.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml

from .comfy import ComfyClient
from .errors import CheckError, InfraError
from .judge import make_adapter
from .judge.base import RetryableJudgeError

PROBE_SCHEMA = {"type": "object",
                "properties": {"dominant_color": {"type": "string",
                                                  "enum": ["red", "green", "blue"]}},
                "required": ["dominant_color"]}
PROBE_SYSTEM = ("You are an image inspector. Answer with JSON only, exactly "
                "matching the requested schema.")
PROBE_USER = ("What is the dominant color of the attached image? "
              'Respond as {"dominant_color": "red|green|blue"}.')


def _solid_jpeg(bgr: tuple[int, int, int]) -> bytes:
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    img[:] = bgr
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


class Doctor:
    def __init__(self):
        self.checks: list[dict] = []
        self.ok = True

    def add(self, name: str, ok: bool, detail: str, warn_only: bool = False):
        level = "OK" if ok else ("WARN" if warn_only else "FAIL")
        if not ok and not warn_only:
            self.ok = False
        self.checks.append({"name": name, "ok": bool(ok), "level": level,
                            "detail": detail})
        print(f"[doctor] {level:4s} {name}: {detail}")


def _check_binaries(doc: Doctor) -> None:
    for binary in ("ffmpeg", "ffprobe"):
        path = shutil.which(binary)
        if not path:
            doc.add(binary, False, "not found on PATH — install ffmpeg")
            continue
        out = subprocess.run([binary, "-version"], capture_output=True,
                             text=True, timeout=15).stdout.splitlines()
        doc.add(binary, True, out[0].split(" version ")[-1].split()[0] if out else "?")


def _check_comfy(doc: Doctor, cfg: dict, pol: dict, do_free: bool) -> None:
    comfy_cfg = cfg.get("comfy") or {}
    host, wf_path = comfy_cfg.get("host"), comfy_cfg.get("workflow_t2v")
    if not host or not wf_path:
        doc.add("comfy.config", False, "config needs comfy.host and comfy.workflow_t2v")
        return
    client = ComfyClient(host=host, workflow=wf_path or "")
    try:
        stats = client._http("GET", "/system_stats")
    except InfraError as e:
        doc.add("comfy.server", False, str(e))
        return
    dev = (stats.get("devices") or [{}])[0]
    free_gb = dev.get("vram_free", 0) / 2 ** 30
    total_gb = dev.get("vram_total", 0) / 2 ** 30
    doc.add("comfy.server", True,
            f"{host} · comfyui {stats.get('system', {}).get('comfyui_version', '?')} "
            f"· {dev.get('name', 'GPU?')} · VRAM {free_gb:.1f}/{total_gb:.1f} GB free")
    try:
        q = client._http("GET", "/queue")
        depth = len(q.get("queue_running", [])) + len(q.get("queue_pending", []))
        doc.add("comfy.queue", depth == 0, f"{depth} job(s) queued/running",
                warn_only=True)
    except InfraError:
        pass

    # workflow file: API format + referenced models installed on the server
    wf_file = Path(wf_path)
    if not wf_file.is_file():
        doc.add("workflow", False, f"not found: {wf_file}")
        return
    try:
        wf = json.loads(wf_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        doc.add("workflow", False, f"unparseable JSON: {e}")
        return
    if not isinstance(wf, dict) or "nodes" in wf or not all(
            isinstance(v, dict) and "class_type" in v for v in wf.values()):
        doc.add("workflow", False,
                "not API format — re-export with ComfyUI → Workflow → Export (API)")
        return
    doc.add("workflow", True, f"{wf_file.name}: API format, {len(wf)} nodes")

    loaders = {"UNETLoader": ("unet_name", "diffusion_models"),
               "CLIPLoader": ("clip_name", "text_encoders"),
               "VAELoader": ("vae_name", "vae"),
               "LoraLoaderModelOnly": ("lora_name", "loras"),
               "LoraLoader": ("lora_name", "loras")}
    wanted: dict[str, set] = {}
    for node in wf.values():
        pair = loaders.get(node.get("class_type"))
        if pair:
            key, folder = pair
            name = node.get("inputs", {}).get(key)
            if isinstance(name, str):
                wanted.setdefault(folder, set()).add(name)
    for folder, names in sorted(wanted.items()):
        try:
            installed = set(client._http("GET", f"/models/{folder}"))
        except (InfraError, TypeError):
            doc.add(f"models.{folder}", False,
                    f"could not list /models/{folder} (old ComfyUI?)", warn_only=True)
            continue
        missing = sorted(names - installed)
        doc.add(f"models.{folder}", not missing,
                "all present" if not missing else
                f"MISSING on server: {missing} — fix workflow or install")

    if do_free:
        handoff = pol.get("vram_handoff", {})
        need = float(handoff.get("free_min_gb", 24))
        try:
            got = client.vram_handoff(need, float(handoff.get("wait_timeout_s", 60)))
            doc.add("vram.handoff", True,
                    f"/free verified: {got:.1f} GB free (need {need:.0f})")
        except InfraError as e:
            doc.add("vram.handoff", False, str(e))


def _check_judge(doc: Doctor, cfg: dict) -> None:
    judge_cfg = cfg.get("judge") or {}
    if not judge_cfg.get("model"):
        doc.add("judge.config", False, "config needs judge.model")
        return
    try:
        adapter = make_adapter(judge_cfg)
        digest = adapter.model_digest()
        doc.add("judge.model", True,
                f"{adapter.name}/{adapter.model} · {str(digest)[:24]}")
    except (CheckError, RetryableJudgeError) as e:
        doc.add("judge.model", False, str(e))
        return

    # Two-call vision smoke test: answers must track the actual image colors.
    answers = []
    try:
        t0 = time.monotonic()
        for color_name, bgr in (("red", (0, 0, 255)), ("blue", (255, 0, 0))):
            raw = adapter.judge([_solid_jpeg(bgr)], PROBE_SYSTEM, PROBE_USER,
                                PROBE_SCHEMA, timeout_s=float(
                                    judge_cfg.get("timeout_s", 300)))
            answers.append((color_name, json.loads(raw).get("dominant_color")))
        elapsed = time.monotonic() - t0
    except (CheckError, RetryableJudgeError, json.JSONDecodeError) as e:
        doc.add("judge.vision", False,
                f"vision/strict-JSON probe failed: {e} — if Ollama's vision sidecar "
                f"is broken for this model family, switch judge.adapter to "
                f"openai_compat (llama.cpp llama-server with GGUF+mmproj)")
        return
    correct = all(want == got for want, got in answers)
    doc.add("judge.vision", correct,
            f"probe answers {answers} in {elapsed:.1f}s"
            + ("" if correct else " — model did NOT see the images (text-only "
               "fallback?). Do not trust this judge; switch adapters."))

    rewrite_model = (cfg.get("rewrite") or {}).get("model")
    if rewrite_model:
        try:
            tags = adapter._get_json("/api/tags", timeout_s=20)
            names = {m.get("name") for m in tags.get("models", [])}
            doc.add("rewrite.model", rewrite_model in names,
                    f"{rewrite_model} {'installed' if rewrite_model in names else 'NOT installed — off_prompt rewrites will fall back to plain re-rolls'}",
                    warn_only=True)
        except (CheckError, RetryableJudgeError):
            pass


def _check_disk_and_files(doc: Doctor, cfg: dict, pol: dict,
                          thresholds_path: Path) -> None:
    runs_dir = Path((cfg.get("paths") or {}).get("runs_dir", "runs"))
    probe_dir = runs_dir if runs_dir.exists() else Path(".")
    free_gb = shutil.disk_usage(probe_dir).free / 2 ** 30
    floor = float(pol.get("disk_min_free_gb", 20))
    doc.add("disk", free_gb >= floor,
            f"{free_gb:.0f} GB free at {probe_dir.resolve()} (floor {floor:.0f} GB)")
    if thresholds_path.is_file():
        thr = yaml.safe_load(thresholds_path.read_text(encoding="utf-8")) or {}
        calibrated = bool(thr.get("calibrated"))
        doc.add("thresholds", calibrated,
                f"v{thr.get('version')} calibrated={calibrated}"
                + ("" if calibrated else " — Phase 0 pending; unattended runs "
                   "will refuse until sign-off"), warn_only=True)
    else:
        doc.add("thresholds", False, f"missing: {thresholds_path}")


def run_doctor(config_path: str, pipeline_path: str, thresholds_path: str,
               do_free: bool, out_path: str | None) -> int:
    doc = Doctor()
    cfg_file = Path(config_path)
    if not cfg_file.is_file():
        doc.add("config", False,
                f"{cfg_file} not found — copy config.example.yaml and fill it in")
        cfg = {}
    else:
        cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) or {}
        doc.add("config", True, str(cfg_file))
    pol = {}
    pipe_file = Path(pipeline_path)
    if pipe_file.is_file():
        pol = (yaml.safe_load(pipe_file.read_text(encoding="utf-8")) or {}
               ).get("policies", {})
        doc.add("pipeline", True, f"{pipe_file} loaded")
    else:
        doc.add("pipeline", False, f"missing: {pipe_file}", warn_only=True)

    _check_binaries(doc)
    if cfg:
        _check_comfy(doc, cfg, pol, do_free)
        _check_judge(doc, cfg)
    _check_disk_and_files(doc, cfg, pol, Path(thresholds_path))

    snapshot = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ok": doc.ok,
        "config": str(cfg_file),
        "checks": doc.checks,
    }
    out = Path(out_path) if out_path else cfg_file.parent / "doctor.json"
    out.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    verdict = "ALL CHECKS PASSED" if doc.ok else "FAILURES ABOVE — fix before running"
    print(f"[doctor] {verdict} · snapshot: {out}")
    return 0 if doc.ok else 2


def main_doctor(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="shortsloop doctor")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--pipeline", default="pipeline.yaml")
    ap.add_argument("--thresholds", default="thresholds.yaml")
    ap.add_argument("--no-free", action="store_true",
                    help="skip the /free VRAM handoff verification")
    ap.add_argument("--out", default=None,
                    help="snapshot path (default: doctor.json next to config)")
    args = ap.parse_args(argv)
    return run_doctor(args.config, args.pipeline, args.thresholds,
                      do_free=not args.no_free, out_path=args.out)
