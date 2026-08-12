"""Synthetic fixture clips — deterministic, CPU-only, encoded with the same
container family as production (H.264 yuv420p MP4 via ffmpeg rawvideo pipe).

Each kind isolates one failure signal so L1 tests prove signal DIRECTION with wide
margins; exact decision values are Phase 0's job, not the tests'.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

FPS = 16
W, H = 480, 832
N_FRAMES = 48

KINDS = ("static", "moving", "flicker", "black", "blurry", "freeze_tail")


def _encode(path: Path, frames: list[np.ndarray], fps: int = FPS) -> Path:
    h, w = frames[0].shape[:2]
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "12",
        "-pix_fmt", "yuv420p", str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg encode failed for {path}")
    return path


def _texture(rng: np.random.Generator, h: int = H, w: int = W) -> np.ndarray:
    """Smooth random field in [0.15, 0.85] — textured enough for sharpness/flow."""
    field = rng.random((h // 8 + 2, w // 8 + 2)).astype(np.float32)
    img = cv2.resize(field, (w, h), interpolation=cv2.INTER_CUBIC)
    img = (img - img.min()) / max(1e-6, img.max() - img.min())
    return 0.15 + 0.70 * img


def _to_bgr(gray01: np.ndarray) -> np.ndarray:
    g = np.clip(gray01 * 255.0, 0, 255).astype(np.uint8)
    return np.dstack([g, g, g])


def _moving_frames(rng: np.random.Generator, n: int = N_FRAMES) -> list[np.ndarray]:
    base = _texture(rng)
    frames = []
    for i in range(n):
        f = np.roll(base, shift=3 * i, axis=1).copy()
        cv2.circle(f, (int((40 + 9 * i) % W), H // 3), 40, 1.0, -1)
        frames.append(f)
    return frames


def make_clip(kind: str, path: Path, n: int = N_FRAMES) -> Path:
    rng = np.random.default_rng(abs(hash(kind)) % (2 ** 32))
    if kind == "static":
        base = _texture(rng)
        frames01 = [np.clip(base + rng.normal(0, 0.004, base.shape).astype(np.float32), 0, 1)
                    for _ in range(n)]
    elif kind == "moving":
        frames01 = _moving_frames(rng, n)
    elif kind == "flicker":
        frames01 = _moving_frames(rng, n)
        for i in range(0, n, 8):
            frames01[i] = np.clip(frames01[i] + 0.35, 0, 1)
    elif kind == "black":
        frames01 = [np.clip(np.full((H, W), 0.02, np.float32)
                            + rng.normal(0, 0.005, (H, W)).astype(np.float32), 0, 1)
                    for _ in range(n)]
    elif kind == "blurry":
        frames01 = [cv2.GaussianBlur(f, (31, 31), 8) for f in _moving_frames(rng, n)]
    elif kind == "freeze_tail":
        moving = _moving_frames(rng, n)
        cut = int(n * 0.6)
        frames01 = moving[:cut] + [moving[cut - 1]] * (n - cut)
    else:
        raise ValueError(f"unknown fixture kind {kind!r}")
    return _encode(path, [_to_bgr(f) for f in frames01])


def make_corrupt(path: Path) -> Path:
    rng = np.random.default_rng(1234)
    path.write_bytes(rng.bytes(100_000))
    return path
