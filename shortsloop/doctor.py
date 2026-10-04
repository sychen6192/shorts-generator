"""`shortsloop doctor` — executable environment discovery for the workstation
(docs/plan.md D1). The kickoff's "discover, do not assume" happens HERE, on the
machine with the GPU: every value the pipeline depends on is verified and the
result written to doctor.json next to the config. The nightly runner refuses to
start without a passing snapshot (fail closed at 22:00, not 3 a.m.).

Checks: ffmpeg/ffprobe (absolute paths recorded — cron has its own PATH) ·
ComfyUI reachable, a real GPU, queue · workflow is API-format, every re-roll knob
patchable at the nightly 720x1280/81f/16fps/steps-8 spec, and its model files exist
on the server · VRAM handoff (/free, then free VRAM verified via /system_stats) ·
judge reachable + model installed + a two-call color VISION probe (a text-only
model that cannot see the images has a 1-in-9 chance of passing) + an L2 DRY RUN:
the real 8-frame 720x1280 rubric call, strict-JSON parsed, inside judge.timeout_s
(plan §9: context fit) · judge unload verified (VRAM comes back) · rewrite model on
ITS server (warn) · disk headroom on the runs_dir filesystem · thresholds/pipeline.

Fail closed, both ways:
- "could not verify" is a FAIL wherever the runner depends on the thing (no GPU,
  unlistable model folders, unrecognized loaders, handoff skipped with --no-free).
- doctor.json is overwritten with ok:false ("in progress") BEFORE any check runs,
  and any exception still ends with ok:false on disk — a crashed or interrupted
  doctor never leaves an older green snapshot behind.

Hard rule 3 applies to the doctor's own judge calls: the VLM is loaded only after
free VRAM was verified, and is unloaded (and VRAM re-verified) right after.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml

from . import gpu, l2
from .comfy import ComfyClient
from .errors import CheckError, InfraError
from .judge import make_adapter
from .judge.base import RetryableJudgeError
from .rewrite import effective_rewrite_cfg
from .thresholds import validate_thresholds

PROBE_SCHEMA = {"type": "object",
                "properties": {"dominant_color": {"type": "string",
                                                  "enum": ["red", "green", "blue"]}},
                "required": ["dominant_color"]}
PROBE_SYSTEM = ("You are an image inspector. Answer with JSON only, exactly "
                "matching the requested schema.")
PROBE_USER = ("What is the dominant color of the attached image? "
              'Respond as {"dominant_color": "red|green|blue"}.')

# Nightly generation spec (plan §0.4) and the re-roll table's `--steps 4→8` knob.
NIGHTLY = {"width": 720, "height": 1280, "length": 81, "fps": 16.0, "steps": 8}
L2_PROBE_FRAMES = 17          # ≥ l2.N_FRAMES so the judge gets the full 8-image call
L2_PROBE_FLOOR = 3
L2_PROBE_PROMPT = (
    "A bright red square slides steadily from the left edge of the frame to the "
    "right edge across a smooth color-gradient background; the square keeps exactly "
    "the same size, color and sharp edges in every frame and moves at a constant "
    "speed without rotating; static locked-off camera with no zoom, pan or tilt; "
    "flat, even studio lighting with no shadows, glare or reflections; minimalist "
    "motion-graphics style, clean vector look, high contrast, crisp detail, smooth "
    "continuous motion from the first frame to the last, vertical 9:16 composition")

CRON_DEFAULT_PATH = ("/usr/bin", "/bin")

# Core ComfyUI loader classes -> (filename input, /models/<folder>).
LOADERS = {"UNETLoader": ("unet_name", "diffusion_models"),
           "CLIPLoader": ("clip_name", "text_encoders"),
           "VAELoader": ("vae_name", "vae"),
           "LoraLoaderModelOnly": ("lora_name", "loras"),
           "LoraLoader": ("lora_name", "loras"),
           "CheckpointLoaderSimple": ("ckpt_name", "checkpoints"),
           "CLIPVisionLoader": ("clip_name", "clip_vision")}
# Older ComfyUI builds only know the legacy folder keys.
LEGACY_FOLDERS = {"diffusion_models": "unet", "text_encoders": "clip"}


def _solid_jpeg(bgr: tuple[int, int, int]) -> bytes:
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    img[:] = bgr
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def _synthetic_clip(path: Path, width: int, height: int, fps: float,
                    n: int = L2_PROBE_FRAMES) -> Path:
    """A short clip at the nightly resolution for the L2 dry run: a red square
    moving over a gradient. Its content is irrelevant — the CALL is under test."""
    yy, xx = np.mgrid[0:height, 0:width]
    base = np.dstack([xx * 255 // max(1, width - 1), yy * 255 // max(1, height - 1),
                      np.full_like(xx, 96)]).astype(np.uint8)
    side = max(8, width // 5)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (width, height))
    if not writer.isOpened():
        raise RuntimeError("OpenCV cannot write an mp4v clip on this machine")
    try:
        for i in range(n):
            frame = base.copy()
            x = (width - side) * i // max(1, n - 1)
            y = (height - side) // 2
            frame[y:y + side, x:x + side] = (40, 40, 220)
            writer.write(frame)
    finally:
        writer.release()
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"OpenCV wrote no data to {path}")
    return path


def fingerprint(config_path: str | Path) -> dict:
    """What a doctor snapshot vouches for. The runner refuses a snapshot whose
    fingerprint differs from the current setup (a passing doctor for last week's
    workflow or judge must not open tonight's gate)."""
    from .verdict import sha256_file
    p = Path(config_path)
    try:
        cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        cfg = {}
    if not isinstance(cfg, dict):
        cfg = {}
    comfy, judge = cfg.get("comfy") or {}, cfg.get("judge") or {}
    comfy = comfy if isinstance(comfy, dict) else {}
    judge = judge if isinstance(judge, dict) else {}
    wf = comfy.get("workflow_t2v")
    return {"config_sha256": sha256_file(p),
            "workflow_sha256": sha256_file(wf) if wf else None,
            "comfy_host": comfy.get("host"),
            "judge": f"{judge.get('adapter', 'ollama')}/{judge.get('model')}"}


class Doctor:
    def __init__(self):
        self.checks: list[dict] = []
        self.ok = True
        self.tools: dict[str, str] = {}

    def add(self, name: str, ok: bool, detail: str, warn_only: bool = False):
        level = "OK" if ok else ("WARN" if warn_only else "FAIL")
        if not ok and not warn_only:
            self.ok = False
        self.checks.append({"name": name, "ok": bool(ok), "level": level,
                            "detail": detail})
        print(f"[doctor] {level:4s} {name}: {detail}")


def _guard(doc: Doctor, name: str, fn, *args, default=None):
    """Run one check group; an unexpected exception is a FAIL entry (and the
    remaining groups still run), never a silently skipped check."""
    try:
        return fn(*args)
    except Exception as e:  # noqa: BLE001 — fail closed: a crash is a FAIL
        doc.add(f"{name}.crash", False, f"check crashed: {type(e).__name__}: {e}")
        return default


def _handoff_policy(pol: dict) -> tuple[float, float]:
    h = pol.get("vram_handoff") or {}
    return float(h.get("free_min_gb", 24)), float(h.get("wait_timeout_s", 180))


# ---------------------------------------------------------------- config / policies

def _example_config() -> Path | None:
    for cand in (Path(__file__).resolve().parent.parent / "config.example.yaml",
                 Path("config.example.yaml")):
        if cand.is_file():
            return cand
    return None


def _load_config(doc: Doctor, cfg_file: Path) -> dict | None:
    """The parsed config, or None when it is missing/unusable (FAIL recorded). A
    missing config gets a skeleton (copied from config.example.yaml) at the
    requested path — never overwriting anything — and the run still FAILs."""
    if not cfg_file.is_file():
        example = _example_config()
        if example is None or cfg_file.exists():
            doc.add("config", False, f"{cfg_file} not found — copy "
                                     f"config.example.yaml there and fill it in")
            return None
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        try:
            with open(cfg_file, "x", encoding="utf-8") as f:   # never overwrite
                f.write(f"# SKELETON written by `shortsloop doctor` on {stamp} from "
                        f"{example.name}.\n# Fill in every value for THIS machine, "
                        f"then re-run `shortsloop doctor`.\n")
                f.write(example.read_text(encoding="utf-8"))
        except OSError as e:
            doc.add("config", False, f"{cfg_file} not found, and writing a skeleton "
                                     f"there failed ({e}) — copy config.example.yaml")
            return None
        doc.add("config", False,
                f"{cfg_file} not found — wrote a skeleton there from {example.name}; "
                f"fill in comfy.host, comfy.workflow_t2v, judge.*, paths.runs_dir "
                f"and re-run doctor")
        return None
    try:
        cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as e:
        first = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
        doc.add("config", False, f"{cfg_file} unreadable/unparseable: {first}")
        return None
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        doc.add("config", False, f"{cfg_file}: top level must be a mapping")
        return None
    doc.add("config", True, str(cfg_file))
    return cfg


def _load_policies(doc: Doctor, pipe_file: Path) -> dict:
    """pipeline.yaml policies merged onto the runner's defaults exactly as the
    runner merges them (so doctor verifies the floors the runner will enforce)."""
    from .runner import DEFAULT_POLICIES     # lazy: the runner may import doctor
    pol = json.loads(json.dumps(DEFAULT_POLICIES))
    if not pipe_file.is_file():
        doc.add("pipeline", False, f"missing: {pipe_file} — runner defaults apply",
                warn_only=True)
        return pol
    try:
        loaded = yaml.safe_load(pipe_file.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as e:
        doc.add("pipeline", False, f"{pipe_file} unparseable: "
                                   f"{str(e).strip().splitlines()[0]}")
        return pol
    policies = loaded.get("policies") if isinstance(loaded, dict) else None
    if not isinstance(loaded, dict) or not isinstance(policies or {}, dict):
        doc.add("pipeline", False, f"{pipe_file}: top level and `policies` must be "
                                   f"mappings")
        return pol
    for key, val in (policies or {}).items():
        if isinstance(val, dict) and isinstance(pol.get(key), dict):
            pol[key].update(val)
        else:
            pol[key] = val
    doc.add("pipeline", True, f"{pipe_file} loaded"
            + ("" if policies else " (no policies — runner defaults apply)"))
    return pol


# ---------------------------------------------------------------- binaries

def _check_binaries(doc: Doctor) -> None:
    for binary in ("ffmpeg", "ffprobe"):
        found = shutil.which(binary)
        if not found:
            doc.add(binary, False, "not found on PATH — install ffmpeg")
            continue
        path = os.path.abspath(found)
        try:
            res = subprocess.run([path, "-version"], capture_output=True, text=True,
                                 timeout=15)
        except (subprocess.TimeoutExpired, OSError) as e:
            doc.add(binary, False, f"{path} -version failed: {e}")
            continue
        if res.returncode != 0:
            doc.add(binary, False, f"{path} -version exited {res.returncode}")
            continue
        out = res.stdout.splitlines()
        version = out[0].split(" version ")[-1].split()[0] if out else "?"
        doc.tools[binary] = path
        doc.add(binary, True, f"{path} · {version}")
    if not doc.tools:
        return
    dirs = list(dict.fromkeys(os.path.dirname(p) for p in doc.tools.values()))
    cron_path = ":".join(dirs + [d for d in CRON_DEFAULT_PATH if d not in dirs])
    print(f"[doctor] hint: cron does not inherit this shell's PATH (its default is "
          f"{':'.join(CRON_DEFAULT_PATH)}); the nightly job must find these same "
          f"binaries — put `PATH={cron_path}` at the top of the crontab")
    off = [d for d in dirs if d not in CRON_DEFAULT_PATH]
    if off:
        doc.add("cron.path", False,
                f"{', '.join(off)} is not on cron's default PATH — without "
                f"PATH={cron_path} in the crontab the nightly checker cannot find "
                f"ffprobe", warn_only=True)


# ---------------------------------------------------------------- ComfyUI

def _check_gpu(doc: Doctor, devices, need_gb: float) -> None:
    if not isinstance(devices, list) or not devices or not isinstance(devices[0], dict):
        doc.add("comfy.gpu", False, "/system_stats reports no GPU devices — ComfyUI "
                                    "without CUDA cannot generate, and free VRAM "
                                    "cannot be verified (hard rule 3)")
        return
    dev = devices[0]
    if str(dev.get("type", "")).lower() == "cpu":
        doc.add("comfy.gpu", False, f"ComfyUI runs on the CPU ({dev.get('name')}) — "
                                    f"start it with CUDA")
        return
    total = (dev.get("vram_total") or 0) / 2 ** 30
    free = max(0, (dev.get("vram_free") or 0) - (dev.get("torch_vram_free") or 0)) \
        / 2 ** 30
    if total < need_gb:
        doc.add("comfy.gpu", False, f"{dev.get('name', 'GPU?')}: {total:.1f} GB total "
                                    f"< vram_handoff.free_min_gb {need_gb:g} — the "
                                    f"handoff can never pass")
        return
    doc.add("comfy.gpu", True, f"{dev.get('name', 'GPU?')} · VRAM {free:.1f}/"
                               f"{total:.1f} GB free")


def _list_models(client: ComfyClient, folder: str) -> tuple[set | None, str]:
    errors = []
    for key in (folder, LEGACY_FOLDERS.get(folder)):
        if not key:
            continue
        try:
            listing = client._http("GET", f"/models/{key}")
        except InfraError as e:
            errors.append(e.message)
            continue
        if isinstance(listing, list) and all(isinstance(x, str) for x in listing):
            return set(listing), key
        errors.append(f"/models/{key} returned {type(listing).__name__}, not a list")
    return None, "; ".join(errors)


def _check_models(doc: Doctor, client: ComfyClient, wf: dict) -> None:
    wanted: dict[str, set] = {}
    unverified: list[str] = []
    for nid, node in wf.items():
        cls = node.get("class_type")
        pair = LOADERS.get(cls)
        if pair:
            key, folder = pair
            name = (node.get("inputs") or {}).get(key)
            if isinstance(name, str):
                wanted.setdefault(folder, set()).add(name)
            else:
                unverified.append(f"{cls} (node {nid}): {key} is not a literal")
        elif isinstance(cls, str) and "loader" in cls.lower():
            unverified.append(f"{cls} (node {nid})")
    if unverified or not wanted:
        doc.add("models.loaders", False,
                (f"unrecognized loader node(s): {unverified}" if unverified else
                 "no recognized model loader node in the workflow")
                + " — doctor cannot verify those model files exist on the server, so "
                  "a missing file would only surface as a rejected submit at night. "
                  f"Use core loaders ({', '.join(sorted(LOADERS))}) or extend "
                  "doctor.LOADERS")
    else:
        doc.add("models.loaders", True,
                f"{sum(len(v) for v in wanted.values())} model file(s) referenced")
    for folder, names in sorted(wanted.items()):
        installed, where = _list_models(client, folder)
        if installed is None:
            doc.add(f"models.{folder}", False,
                    f"could not list /models/{folder} ({where}) — model presence "
                    f"UNVERIFIED and the nightly depends on it; update ComfyUI")
            continue
        missing = sorted(names - installed)
        doc.add(f"models.{folder}", not missing,
                "all present" if not missing else
                f"MISSING on server: {missing} — fix workflow or install")


def _check_comfy(doc: Doctor, cfg: dict, pol: dict) -> ComfyClient | None:
    """Returns a client when ComfyUI answered (the handoff needs it)."""
    comfy_cfg = cfg.get("comfy") if isinstance(cfg.get("comfy"), dict) else {}
    host, wf_path = comfy_cfg.get("host"), comfy_cfg.get("workflow_t2v")
    if not host or not wf_path:
        doc.add("comfy.config", False, "config needs comfy.host and comfy.workflow_t2v")
        return None
    client = ComfyClient(host=str(host), workflow=str(wf_path))
    try:
        stats = client._http("GET", "/system_stats")
    except InfraError as e:
        doc.add("comfy.server", False, e.message)
        return None
    stats = stats if isinstance(stats, dict) else {}
    doc.add("comfy.server", True,
            f"{host} · comfyui {(stats.get('system') or {}).get('comfyui_version', '?')}")
    _check_gpu(doc, stats.get("devices"), _handoff_policy(pol)[0])
    try:
        q = client.queue_state()
        depth = len(q["running"]) + len(q["pending"])
        doc.add("comfy.queue", depth == 0, f"{depth} job(s) queued/running",
                warn_only=True)
    except InfraError as e:
        doc.add("comfy.queue", False, f"could not read /queue: {e.message}",
                warn_only=True)

    # workflow file: API format + every knob patchable + models installed on server
    wf_file = Path(wf_path)
    if not wf_file.is_file():
        doc.add("workflow", False, f"not found: {wf_file}")
        return client
    try:
        wf = json.loads(wf_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        doc.add("workflow", False, f"unreadable/unparseable JSON: {e}")
        return client
    if not isinstance(wf, dict) or "nodes" in wf or not all(
            isinstance(v, dict) and "class_type" in v for v in wf.values()):
        doc.add("workflow", False,
                "not API format — re-export with ComfyUI → Workflow → Export (API)")
        return client
    doc.add("workflow", True, f"{wf_file.name}: API format, {len(wf)} nodes")

    problems = client.validate_workflow(**NIGHTLY)
    spec = (f"{NIGHTLY['width']}x{NIGHTLY['height']}, {NIGHTLY['length']} frames, "
            f"{NIGHTLY['fps']:g} fps, steps {NIGHTLY['steps']}")
    doc.add("workflow.knobs", not problems,
            f"dry run ({spec}): prompt/seed/size/length/steps all patchable"
            if not problems else "; ".join(problems))
    _check_models(doc, client, wf)
    return client


def _handoff_to_judge(doc: Doctor, client: ComfyClient | None, pol: dict,
                      do_free: bool) -> bool:
    """Records vram.handoff. Returns True only when free VRAM for the judge has
    been VERIFIED — the precondition for loading the VLM at all (hard rule 3)."""
    need, wait = _handoff_policy(pol)
    if client is None:
        doc.add("vram.handoff", False, "not verified: ComfyUI not reachable/configured "
                                       "— the judge probes will not load the VLM "
                                       "without verified free VRAM (hard rule 3)")
        return False
    if do_free:
        try:
            got = gpu.to_judge(client, need, wait)
        except InfraError as e:
            doc.add("vram.handoff", False, e.message)
            return False
        doc.add("vram.handoff", True, f"/free verified: {got:.1f} GB free "
                                      f"(need {need:g})")
        return True
    # --no-free (debugging): the handoff the nightly depends on stays UNVERIFIED,
    # so the snapshot must not pass. Still never load the VLM beside a resident Wan.
    now, why = None, ""
    try:
        now = client.vram_free_gb()
    except InfraError as e:
        why = e.message
    ready = now is not None and now >= need
    state = (f"{now:.1f} GB free without /free, so the judge probes still run"
             if ready else
             f"{why if now is None else f'only {now:.1f} GB free'} without /free "
             f"(need {need:g}) — judge probes skipped so the VLM never loads beside "
             f"Wan")
    doc.add("vram.handoff", False,
            f"NOT VERIFIED (--no-free): the nightly generation→judge handoff depends "
            f"on /free actually freeing VRAM, so this snapshot is marked FAILED and "
            f"the runner refuses it — re-run doctor without --no-free. {state}")
    return ready


# ---------------------------------------------------------------- judge

def _probe_vision(doc: Doctor, adapter, judge_cfg: dict) -> None:
    """Two calls, one solid image each: answers must track the actual colors."""
    answers = []
    try:
        t0 = time.monotonic()
        for color_name, bgr in (("red", (0, 0, 255)), ("blue", (255, 0, 0))):
            raw = adapter.judge([_solid_jpeg(bgr)], PROBE_SYSTEM, PROBE_USER,
                                PROBE_SCHEMA, timeout_s=float(
                                    judge_cfg.get("timeout_s", 300)))
            data = json.loads(raw)
            answers.append((color_name, data.get("dominant_color")
                            if isinstance(data, dict) else None))
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


def _probe_l2(doc: Doctor, judge_cfg: dict) -> None:
    """The REAL nightly judge call (l2.run_l2: 8 native-resolution frames + the
    rubric + the strict response schema), on a synthetic 720x1280 clip. Proves the
    payload fits num_ctx, the model build handles 8 images, and it answers in time.
    Scores are irrelevant (hard rule 1: doctor judges the instrument, not a clip)."""
    timeout_s = float(judge_cfg.get("timeout_s", 300))
    w, h, fps = NIGHTLY["width"], NIGHTLY["height"], NIGHTLY["fps"]
    floors = {d: L2_PROBE_FLOOR for d in l2.DIMENSIONS}
    with tempfile.TemporaryDirectory(prefix="shortsloop-doctor-") as td:
        try:
            clip = _synthetic_clip(Path(td) / "l2_probe.mp4", w, h, fps)
        except (RuntimeError, OSError, cv2.error) as e:
            doc.add("judge.l2_dryrun", False, f"could not synthesize the probe clip: {e}")
            return
        t0 = time.monotonic()
        try:
            block, _raw = l2.run_l2(clip, {"fps": fps}, L2_PROBE_PROMPT, judge_cfg,
                                    floors)
        except (CheckError, RetryableJudgeError) as e:
            doc.add("judge.l2_dryrun", False,
                    f"real {l2.N_FRAMES}-frame {w}x{h} rubric call failed after "
                    f"{time.monotonic() - t0:.1f}s: {e} — check judge.num_ctx (8 "
                    f"images + rubric must fit), judge.timeout_s, and multi-image "
                    f"support of this model build")
            return
        elapsed = time.monotonic() - t0
    n = len(block["frames"])
    in_time = elapsed <= timeout_s
    doc.add("judge.l2_dryrun", n == l2.N_FRAMES and in_time,
            f"{n} frames {w}x{h} + rubric -> strict-JSON scores parsed in "
            f"{elapsed:.1f}s (judge.timeout_s {timeout_s:g})"
            + ("" if in_time else " — slower than timeout_s (a call timed out and "
               "was retried): the nightly judge wave would ERROR")
            + ("" if n == l2.N_FRAMES else f" — expected {l2.N_FRAMES} frames"))
    if in_time and elapsed > timeout_s / 2:
        doc.add("judge.latency", False,
                f"one rubric call took {elapsed:.0f}s, over half of judge.timeout_s "
                f"({timeout_s:g}s) — expect clip-scope ERRORs under load; raise "
                f"judge.timeout_s", warn_only=True)


def _release_judge(doc: Doctor, cfg: dict, judge_cfg: dict, pol: dict,
                   client: ComfyClient) -> None:
    """After the doctor's judge wave: unload the VLM (and the rewrite LLM) and
    VERIFY the VRAM comes back — the nightly judge->generation handoff depends on
    it (hard rule 3), and an unload that does nothing (e.g. openai_compat without
    judge.unload_url) must fail here at 22:00, not halt the night."""
    notes = gpu.unload_llms(judge_cfg, effective_rewrite_cfg(cfg))
    need, wait = _handoff_policy(pol)
    try:
        free = client.vram_wait(need, wait, "generation")
    except InfraError as e:
        doc.add("judge.unload", False,
                f"{e.message} — the judge did not leave VRAM after unload "
                f"({'; '.join(notes) or 'no unload notes'}); the nightly "
                f"judge->generation handoff would halt. openai_compat: set "
                f"judge.unload_url")
        return
    doc.add("judge.unload", True, f"{'; '.join(notes) or 'unload requested'} · "
                                  f"{free:.1f} GB free again (need {need:g})")


def _check_judge(doc: Doctor, cfg: dict, pol: dict, client: ComfyClient | None,
                 vram_ready: bool) -> None:
    judge_cfg = cfg.get("judge")
    if not isinstance(judge_cfg, dict) or not judge_cfg.get("model"):
        doc.add("judge.config", False, "config needs judge.model")
        return
    judge_cfg = dict(judge_cfg)      # verbatim: what shortsloop-check will use
    try:
        adapter = make_adapter(judge_cfg)
        digest = adapter.model_digest()
        doc.add("judge.model", True,
                f"{adapter.name}/{adapter.model} · {str(digest)[:24]}")
    except (CheckError, RetryableJudgeError) as e:
        doc.add("judge.model", False, str(e))
        return
    if not vram_ready or client is None:
        why = ("not run: free VRAM for the judge was not verified (see vram.handoff) "
               "— hard rule 3 forbids loading the VLM beside Wan")
        doc.add("judge.vision", False, why)
        doc.add("judge.l2_dryrun", False, why)
        return
    try:
        _probe_vision(doc, adapter, judge_cfg)
        _probe_l2(doc, judge_cfg)
    finally:
        _release_judge(doc, cfg, judge_cfg, pol, client)


def _ollama_name(name: str) -> str:
    """Ollama's own resolution: an untagged name means `<name>:latest` (a colon in a
    registry host:port prefix is not a tag). Local copy of the rule another branch
    adds as shortsloop.judge.ollama.normalize_model_name — use that once merged."""
    return name if ":" in name.rsplit("/", 1)[-1] else f"{name}:latest"


def _check_rewrite(doc: Doctor, cfg: dict) -> None:
    """The rewrite model on the server rewrite.py actually calls (rewrite.base_url,
    else the judge's Ollama, else local Ollama). Warn-level: a missing rewrite only
    downgrades off_prompt attempt 2 to a plain re-roll — but never silently."""
    rw = effective_rewrite_cfg(cfg)
    if not rw.get("enabled", True):
        doc.add("rewrite.model", True, "rewrite disabled — off_prompt attempt 2 is a "
                                       "plain re-roll")
        return
    model = rw.get("model")
    if not model:
        doc.add("rewrite.model", False, "rewrite enabled but no rewrite.model — "
                                        "off_prompt rewrites fall back to plain "
                                        "re-rolls", warn_only=True)
        return
    base = str(rw["base_url"]).rstrip("/")
    fallback = "off_prompt rewrites will fall back to plain re-rolls"
    try:
        with urllib.request.urlopen(base + "/api/tags", timeout=20) as r:
            tags = json.loads(r.read())
        names = {_ollama_name(n) for m in tags.get("models", [])
                 for n in (m.get("name"), m.get("model")) if isinstance(n, str)}
    except Exception as e:  # noqa: BLE001 — any failure is reported, never skipped
        doc.add("rewrite.model", False,
                f"{model} @ {base}: could not query /api/tags ({e}) — {fallback} "
                f"(rewrite needs an Ollama server: set rewrite.base_url)",
                warn_only=True)
        return
    installed = _ollama_name(str(model)) in names
    doc.add("rewrite.model", installed,
            f"{model} @ {base}: "
            + ("installed" if installed else f"NOT installed — {fallback}"),
            warn_only=True)


# ---------------------------------------------------------------- disk / thresholds

def _check_disk(doc: Doctor, cfg: dict, pol: dict) -> None:
    paths = cfg.get("paths") if isinstance(cfg.get("paths"), dict) else {}
    runs_dir = Path(paths.get("runs_dir") or "runs")
    probe = runs_dir                 # nearest existing ancestor = the target disk
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    if not probe.exists():
        probe = Path(".")
    free_gb = shutil.disk_usage(probe).free / 2 ** 30
    floor = float(pol.get("disk_min_free_gb", 20))
    doc.add("disk", free_gb >= floor,
            f"{free_gb:.0f} GB free at {probe.resolve()} (floor {floor:.0f} GB)"
            + ("" if probe == runs_dir else f"; runs_dir {runs_dir} not created yet"))


def _check_thresholds(doc: Doctor, thresholds_path: Path) -> None:
    if not thresholds_path.is_file():
        doc.add("thresholds", False, f"missing: {thresholds_path}")
        return
    try:
        thr = yaml.safe_load(thresholds_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as e:
        doc.add("thresholds", False, f"{thresholds_path} unparseable: "
                                     f"{str(e).strip().splitlines()[0]}")
        return
    problems = validate_thresholds(thr)
    if problems:
        doc.add("thresholds", False, f"{thresholds_path} off the frozen shape: "
                                     + "; ".join(problems))
        return
    calibrated = thr["calibrated"] is True
    doc.add("thresholds", calibrated,
            f"v{thr.get('version')} calibrated={calibrated}"
            + ("" if calibrated else " — Phase 0 pending; unattended runs "
               "will refuse until sign-off"), warn_only=True)


# ---------------------------------------------------------------- entry points

def _run_checks(doc: Doctor, cfg_file: Path, pipe_file: Path, thr_file: Path,
                do_free: bool) -> None:
    cfg = _load_config(doc, cfg_file)
    pol = _load_policies(doc, pipe_file)
    _guard(doc, "binaries", _check_binaries, doc)
    if cfg is not None:
        client = _guard(doc, "comfy", _check_comfy, doc, cfg, pol)
        ready = _guard(doc, "vram.handoff", _handoff_to_judge, doc, client, pol,
                       do_free, default=False)
        _guard(doc, "judge", _check_judge, doc, cfg, pol, client, ready)
        _guard(doc, "rewrite", _check_rewrite, doc, cfg)
    _guard(doc, "disk", _check_disk, doc, cfg or {}, pol)
    _guard(doc, "thresholds", _check_thresholds, doc, thr_file)


def _write_json_atomic(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_doctor(config_path: str, pipeline_path: str, thresholds_path: str,
               do_free: bool, out_path: str | None) -> int:
    cfg_file = Path(config_path)
    out = Path(out_path) if out_path else cfg_file.parent / "doctor.json"
    base = {"config": str(cfg_file), "do_free": bool(do_free)}
    # Invalidate any previous snapshot FIRST: from here on, only a run that
    # completes with every check green can put ok:true back on disk.
    try:
        _write_json_atomic(out, {"ts": _now(), "ok": False, "status": "in progress",
                                 **base, "checks": []})
    except OSError as e:
        try:
            out.unlink(missing_ok=True)
        except OSError:
            pass
        print(f"[doctor] FAIL cannot write snapshot {out}: {e}", file=sys.stderr)
        return 2

    doc = Doctor()
    status = "interrupted"
    try:
        _run_checks(doc, cfg_file, Path(pipeline_path), Path(thresholds_path), do_free)
        status = "complete"
    except Exception as e:  # noqa: BLE001 — fail closed: a crash is a FAIL
        status = "crashed"
        doc.add("doctor.crash", False, f"{type(e).__name__}: {e} — doctor crashed; "
                                       f"snapshot marked FAILED")
    finally:
        ok = doc.ok and status == "complete"
        try:
            fp = fingerprint(cfg_file)
        except Exception as e:  # noqa: BLE001
            fp, ok = {"error": str(e)}, False
        try:
            _write_json_atomic(out, {"ts": _now(), "ok": ok, "status": status, **base,
                                     "fingerprint": fp, "tools": doc.tools,
                                     "checks": doc.checks})
        except OSError as e:
            ok = False      # the "in progress" ok:false snapshot stays on disk
            print(f"[doctor] FAIL cannot write snapshot {out}: {e}", file=sys.stderr)
    verdict = "ALL CHECKS PASSED" if ok else "FAILURES ABOVE — fix before running"
    print(f"[doctor] {verdict} · snapshot: {out}")
    return 0 if ok else 2


def main_doctor(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="shortsloop doctor")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--pipeline", default="pipeline.yaml")
    ap.add_argument("--thresholds", default="thresholds.yaml")
    ap.add_argument("--no-free", action="store_true",
                    help="debugging only: do not POST /free. The VRAM handoff then "
                         "stays unverified, so the snapshot is written ok:false and "
                         "the runner refuses it; judge probes run only if the GPU "
                         "is already free")
    ap.add_argument("--out", default=None,
                    help="snapshot path (default: doctor.json next to config)")
    args = ap.parse_args(argv)
    return run_doctor(args.config, args.pipeline, args.thresholds,
                      do_free=not args.no_free, out_path=args.out)
