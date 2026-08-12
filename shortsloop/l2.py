"""L2 — VLM judge layer (docs/plan.md §2.2, D5/D10).

Responsibilities: sample 8 frames (uniform, incl. first + last), send ONE judging
call with the generation prompt and an anchored 1–5 rubric, validate the strict-JSON
response, apply floors. The judge model reports scores + reasons only; PASS/FAIL is
decided here (hard rule 1). N/A is honored only where the rubric allows it
(subject_consistency). Retry once on bad content, then clip-scope ERROR; an
unreachable judge is infra-scope immediately (fail closed either way).
"""

from __future__ import annotations

import json
import math

import cv2

from .errors import ClipError, InfraError
from .judge import make_adapter
from .judge.base import RetryableJudgeError
from .verdict import sha256_text

N_FRAMES = 8
JPEG_QUALITY = 90

DIMENSIONS = [
    "prompt_adherence",
    "subject_consistency",
    "anatomy_artifacts",
    "temporal_coherence",
    "imaging_quality",
]
NA_ALLOWED = {"subject_consistency"}

# Structured-output schema sent to the judge server (Ollama `format:` /
# OpenAI-compat `response_format.json_schema.schema`).
_DIM_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "minimum": 1, "maximum": 5},
        "na": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["score", "na", "reason"],
}
JUDGE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {d: _DIM_SCHEMA for d in DIMENSIONS},
    "required": list(DIMENSIONS),
}

SYSTEM_PROMPT = """\
You are a strict quality judge for AI-generated short video clips. You are shown
frames sampled in time order from ONE clip, plus the text prompt the clip was
generated from. Score each dimension 1-5 against the anchors given. Judge only what
is visible in the frames; never give credit for what the prompt intended but the
frames do not show. Be harsh: a 4 means you would publish it; a 5 means flawless.
Respond with JSON only, exactly matching the requested schema — no prose outside it."""

USER_TEMPLATE = """\
GENERATION PROMPT:
{prompt}

You are given {n} frames sampled uniformly across the clip in time order
(frame 1 = clip start, frame {n} = clip end; timestamps in seconds: {timestamps}).

Score these five dimensions, each 1-5:

1. prompt_adherence — Are the prompt's subject, action, camera movement and setting
   actually visible? 5=everything present; 3=subject right but a named action/camera
   move/setting element missing; 1=wrong subject or scene. Name any missing element
   in the reason.
2. subject_consistency — Does the main subject stay the same entity across frames
   (color, shape, count, outfit)? 5=identical; 3=noticeable drift; 1=changes identity.
   If the clip has NO persistent main subject (pure landscape/abstract), set
   "na": true and explain; do not invent a score basis.
3. anatomy_artifacts — Hands, limbs, faces, object geometry: 5=clean; 3=one clearly
   wrong part (extra finger, fused limb, melting edge); 1=grossly deformed or
   duplicated parts. Cite the frame number(s).
4. temporal_coherence — SEMANTIC continuity across the sampled sequence: subjects
   teleporting, objects appearing/vanishing, scene morphing into a different scene,
   physically impossible motion between frames. 5=coherent; 1=incoherent. Do NOT
   judge frame-to-frame flicker — a separate deterministic layer measures that.
5. imaging_quality — Blur, banding, blown highlights, heavy compression artifacts:
   5=clean and sharp; 3=visibly soft or banded; 1=badly degraded.

For every dimension: "na" is false unless explicitly allowed above; "reason" is one
short sentence citing frame numbers where relevant. JSON only."""


def sample_frames(clip_path, fps: float | None) -> tuple[list[bytes], list[dict]]:
    """Decode and pick N_FRAMES uniformly (always incl. first and last), JPEG-encode
    at native resolution."""
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise ClipError("l2", f"OpenCV could not reopen {clip_path} for frame sampling")
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    if len(frames) < 2:
        raise ClipError("l2", f"only {len(frames)} frame(s) decodable for sampling")

    n = min(N_FRAMES, len(frames))
    idxs = sorted({round(i * (len(frames) - 1) / (n - 1)) for i in range(n)})
    fps_val = fps if fps and fps > 0 else 16.0
    jpegs, meta = [], []
    for idx in idxs:
        ok, buf = cv2.imencode(".jpg", frames[idx],
                               [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            raise ClipError("l2", f"JPEG encode failed for frame {idx}")
        jpegs.append(buf.tobytes())
        meta.append({"index": int(idx), "t": round(idx / fps_val, 3)})
    return jpegs, meta


def _parse_response(raw: str) -> dict:
    """Validate the judge's content against the response contract. Raises
    RetryableJudgeError on any violation (retried once by run_l2)."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RetryableJudgeError(f"judge content is not valid JSON: {e}")
    if not isinstance(data, dict):
        raise RetryableJudgeError("judge content is not a JSON object")
    parsed = {}
    for dim in DIMENSIONS:
        entry = data.get(dim)
        if not isinstance(entry, dict):
            raise RetryableJudgeError(f"missing/invalid dimension {dim!r}")
        score = entry.get("score")
        if isinstance(score, float) and score.is_integer():
            score = int(score)
        if not isinstance(score, int) or not (1 <= score <= 5):
            raise RetryableJudgeError(f"{dim}.score must be an integer 1-5, got {score!r}")
        na = bool(entry.get("na", False)) and dim in NA_ALLOWED
        reason = entry.get("reason")
        if not isinstance(reason, str):
            raise RetryableJudgeError(f"{dim}.reason must be a string")
        parsed[dim] = {"score": score, "na": na, "reason": reason.strip()}
    return parsed


def run_l2(clip_path, container, prompt_text, judge_cfg, floors) -> tuple[dict, str]:
    """Returns (l2_block, raw_response_text). ClipError / InfraError per D7."""
    for dim in DIMENSIONS:
        if dim not in floors:
            raise InfraError("l2", f"thresholds l2.floors missing {dim!r}")

    adapter = make_adapter(judge_cfg)
    digest = adapter.model_digest()  # reachability + exact model build

    jpegs, frame_meta = sample_frames(clip_path, (container or {}).get("fps"))
    user = USER_TEMPLATE.format(
        prompt=prompt_text,
        n=len(jpegs),
        timestamps=", ".join(str(m["t"]) for m in frame_meta),
    )
    timeout_s = float(judge_cfg.get("timeout_s", 300))

    last_err = None
    raw = None
    parsed = None
    for _ in range(2):  # 1 try + 1 retry on bad content (plan §5 rows 2-3)
        try:
            raw = adapter.judge(jpegs, SYSTEM_PROMPT, user,
                                JUDGE_RESPONSE_SCHEMA, timeout_s)
            parsed = _parse_response(raw)
            break
        except RetryableJudgeError as e:
            last_err = e
    if parsed is None:
        raise ClipError("l2", f"judge response unusable after retry: {last_err}")

    dimensions = {}
    for dim in DIMENSIONS:
        entry = parsed[dim]
        floor = int(floors[dim])
        ok = entry["na"] or entry["score"] >= floor
        dimensions[dim] = {
            "score": entry["score"],
            "floor": floor,
            "na": entry["na"],
            "pass": bool(ok),
            "reason": entry["reason"],
        }
    non_na = [d for d in dimensions.values() if not d["na"]]
    l2_block = {
        "model": f"{adapter.name}/{adapter.model}",
        "model_digest": digest,
        "temperature": float(judge_cfg.get("temperature", 0)),
        "seed": int(judge_cfg.get("seed", 7)),
        "frames": frame_meta,
        "dimensions": dimensions,
        "pass": all(d["pass"] for d in non_na) and len(non_na) > 0,
        "raw_response_sha256": sha256_text(raw),
    }
    return l2_block, raw
