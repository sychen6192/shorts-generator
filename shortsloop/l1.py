"""L1 — deterministic, CPU-only quality metrics (docs/plan.md §2.2, D3/D4).

Design invariants:
- All pixel metrics are computed at the CANONICAL ANALYSIS RESOLUTION (480x832,
  flow at a further 2x downscale) regardless of input size, so thresholds calibrated
  on 480x832 draft clips transfer to 720x1280 production clips.
- One sequential decode pass computes everything; ffprobe (the only external binary)
  provides container truth.
- Constants below define METRIC SEMANTICS and are not calibratable; the calibratable
  decision values live in thresholds.yaml and are applied in run_checks().
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .errors import ClipError, InfraError

ANALYSIS_W, ANALYSIS_H = 480, 832
FLOW_DOWNSCALE = 2               # flow computed at 240x416
FLICKER_WIN = 7                  # rolling-median window (pairs) for dip detection
FLICKER_DELTA = 0.03             # SSIM drop below rolling median that counts as a dip
FREEZE_SSIM = 0.995              # pair is "frozen" if SSIM >= this ...
FREEZE_FLOW = 3e-4               # ... and normalized flow <= this
BLACK_LUMA = 16.0 / 255.0        # frame is "black" if mean luma below this
CLIP_LEVEL = 250.0 / 255.0       # pixel counts as clipped at/above this

# Spec-check tolerances against dispatch expectations (docs/plan.md §2.2).
SPEC_DURATION_TOL = 0.15         # ±15%
SPEC_FPS_TOL = 1.0               # ±1 fps
SPEC_FRAMES_TOL = 2              # ±2 frames (encoder off-by-one headroom)


def probe(path: str | Path) -> dict:
    """Container truth via ffprobe. ClipError if the file is unreadable/not video;
    InfraError if ffprobe itself is missing (broken instrument)."""
    p = Path(path)
    if not p.is_file():
        raise ClipError("probe", f"clip not found: {p}")
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(p),
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except FileNotFoundError:
        raise InfraError("probe", "ffprobe not found on PATH — install ffmpeg")
    except subprocess.TimeoutExpired:
        raise ClipError("probe", f"ffprobe timed out on {p}")
    if res.returncode != 0:
        raise ClipError("probe", f"ffprobe failed: {res.stderr.strip()[:500]}")
    try:
        info = json.loads(res.stdout)
    except json.JSONDecodeError:
        raise ClipError("probe", "ffprobe produced unparseable output")

    vstreams = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
    if not vstreams:
        raise ClipError("probe", "no video stream in container")
    v = vstreams[0]

    def _rate(expr: str | None) -> float | None:
        if not expr or "/" not in str(expr):
            return None
        num, den = str(expr).split("/", 1)
        try:
            num_f, den_f = float(num), float(den)
        except ValueError:
            return None
        return num_f / den_f if den_f else None

    duration = info.get("format", {}).get("duration")
    nb_frames = v.get("nb_frames")
    return {
        "duration_s": float(duration) if duration is not None else None,
        "width": int(v["width"]),
        "height": int(v["height"]),
        "fps": _rate(v.get("avg_frame_rate")) or _rate(v.get("r_frame_rate")),
        "nb_frames": int(nb_frames) if nb_frames not in (None, "N/A") else None,
        "vcodec": v.get("codec_name"),
    }


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean SSIM of two float32 grayscale images in [0,1] (11x11 gaussian, s=1.5)."""
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    blur = lambda img: cv2.GaussianBlur(img, (11, 11), 1.5)  # noqa: E731
    mu_a, mu_b = blur(a), blur(b)
    var_a = blur(a * a) - mu_a * mu_a
    var_b = blur(b * b) - mu_b * mu_b
    cov = blur(a * b) - mu_a * mu_b
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2)
    return float((num / den).mean())


def compute_metrics(path: str | Path, fps_hint: float | None = None) -> dict:
    """Decode once, compute all L1 metrics at the canonical analysis resolution.

    Returns {"analysis": {...}, "metrics": {...}}. ClipError if fewer than 2 frames
    decode.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ClipError("l1", f"OpenCV could not open {path}")

    flow_w, flow_h = ANALYSIS_W // FLOW_DOWNSCALE, ANALYSIS_H // FLOW_DOWNSCALE
    flow_diag = math.hypot(flow_w, flow_h)

    lumas: list[float] = []
    laplacians: list[float] = []
    clipped_fracs: list[float] = []
    ssims: list[float] = []
    flows: list[float] = []

    prev_gray = None
    prev_small = None
    n_frames = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            n_frames += 1
            frame = cv2.resize(frame, (ANALYSIS_W, ANALYSIS_H), interpolation=cv2.INTER_AREA)
            gray_u8 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = gray_u8.astype(np.float32) / 255.0

            lumas.append(float(gray.mean()))
            clipped_fracs.append(float((gray >= CLIP_LEVEL).mean()))
            laplacians.append(float(cv2.Laplacian(gray_u8.astype(np.float32), cv2.CV_32F).var()))

            small = cv2.resize(gray_u8, (flow_w, flow_h), interpolation=cv2.INTER_AREA)
            if prev_gray is not None:
                ssims.append(_ssim(prev_gray, gray))
                flow = cv2.calcOpticalFlowFarneback(
                    prev_small, small, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                flows.append(float(mag.mean()) / flow_diag)
            prev_gray, prev_small = gray, small
    finally:
        cap.release()

    if n_frames < 2:
        raise ClipError("l1", f"only {n_frames} frame(s) decodable — corrupt or not a video")

    ssims_a = np.array(ssims)
    flows_a = np.array(flows)
    lap_a = np.array(laplacians)

    # Flicker: pairs whose SSIM sits FLICKER_DELTA below the rolling median.
    dips = 0
    half = FLICKER_WIN // 2
    for i in range(len(ssims_a)):
        window = ssims_a[max(0, i - half): i + half + 1]
        if float(np.median(window)) - ssims_a[i] >= FLICKER_DELTA:
            dips += 1

    # Freeze: longest run of consecutive frozen pairs, in seconds.
    fps = fps_hint if fps_hint and fps_hint > 0 else 16.0
    longest = run = 0
    for s, f in zip(ssims_a, flows_a):
        if s >= FREEZE_SSIM and f <= FREEZE_FLOW:
            run += 1
            longest = max(longest, run)
        else:
            run = 0

    return {
        "analysis": {
            "width": ANALYSIS_W,
            "height": ANALYSIS_H,
            "flow_downscale": FLOW_DOWNSCALE,
            "frames_analyzed": n_frames,
        },
        "metrics": {
            "flow_mag_median": float(np.median(flows_a)),
            "flow_mag_p90": float(np.percentile(flows_a, 90)),
            "ssim_min": float(ssims_a.min()),
            "ssim_p05": float(np.percentile(ssims_a, 5)),
            "ssim_mean": float(ssims_a.mean()),
            "flicker_dips": int(dips),
            "freeze_longest_run_s": float(longest / fps),
            "laplacian_p10": float(np.percentile(lap_a, 10)),
            "laplacian_median": float(np.median(lap_a)),
            "luma_mean": float(np.mean(lumas)),
            "black_frame_frac": float(np.mean([l < BLACK_LUMA for l in lumas])),
            "clipped_frac": float(np.mean(clipped_fracs)),
        },
    }


def spec_check(container: dict, frames_analyzed: int, expect: dict | None) -> dict:
    """Container sanity check; `expect` (optional) comes from the dispatch sheet."""
    problems: list[str] = []
    if frames_analyzed < 2:
        problems.append(f"only {frames_analyzed} decodable frames")
    if expect:
        for key in ("width", "height"):
            want = expect.get(key)
            if want is not None and container.get(key) != int(want):
                problems.append(f"{key} {container.get(key)} != expected {want}")
        want_fps = expect.get("fps")
        if want_fps is not None and container.get("fps") is not None:
            if abs(container["fps"] - float(want_fps)) > SPEC_FPS_TOL:
                problems.append(f"fps {container['fps']:.2f} outside {want_fps}±{SPEC_FPS_TOL}")
        want_frames = expect.get("frames")
        if want_frames is not None:
            if abs(frames_analyzed - int(want_frames)) > SPEC_FRAMES_TOL:
                problems.append(f"frames {frames_analyzed} outside {want_frames}±{SPEC_FRAMES_TOL}")
        want_dur = expect.get("duration_s")
        if want_dur is not None and container.get("duration_s") is not None:
            lo = float(want_dur) * (1 - SPEC_DURATION_TOL)
            hi = float(want_dur) * (1 + SPEC_DURATION_TOL)
            if not (lo <= container["duration_s"] <= hi):
                problems.append(
                    f"duration {container['duration_s']:.2f}s outside "
                    f"[{lo:.2f}, {hi:.2f}]s")
    ok = not problems
    return {
        "name": "spec",
        "metric": "container",
        "op": "match",
        "threshold": None,
        "value": None,
        "pass": ok,
        "reason": "container matches expectations" if ok else "; ".join(problems),
    }


_OPS = {
    ">=": lambda v, t: v >= t,
    "<=": lambda v, t: v <= t,
}

_CHECK_REASONS = {
    "motion": "median optical flow below floor — near-static clip",
    "freeze": "long frozen segment detected",
    "flicker": "frame-to-frame SSIM dips — flicker/popping",
    "ssim_floor": "hard SSIM discontinuity — cut/blow-up between frames",
    "sharpness": "low Laplacian variance — soft/blurry frames",
    "black": "black frames present",
    "exposure": "large clipped-highlight fraction — blown exposure",
}


def run_checks(metrics: dict, thresholds_l1: dict) -> list[dict]:
    """Evaluate thresholds.yaml `l1:` entries against computed metrics."""
    checks = []
    for name, spec in thresholds_l1.items():
        try:
            metric, op, value = spec["metric"], spec["op"], spec["value"]
        except (TypeError, KeyError):
            raise InfraError("l1", f"thresholds.yaml l1.{name} is malformed: {spec!r}")
        if op not in _OPS:
            raise InfraError("l1", f"thresholds.yaml l1.{name}: unsupported op {op!r}")
        if metric not in metrics:
            raise InfraError("l1", f"thresholds.yaml l1.{name}: unknown metric {metric!r}")
        measured = metrics[metric]
        ok = _OPS[op](measured, value)
        checks.append({
            "name": name,
            "metric": metric,
            "op": op,
            "threshold": value,
            "value": measured,
            "pass": bool(ok),
            "reason": "ok" if ok else _CHECK_REASONS.get(name, f"{metric} {op} {value} violated"),
        })
    return checks
