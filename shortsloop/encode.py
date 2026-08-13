"""Persist stage: silent QC encode of PASSING clips via vendored ig_encode.sh,
then ffprobe re-verification (the encode script's own check plus ours — trust
commands, not eyes)."""

from __future__ import annotations

import subprocess
from pathlib import Path

from .errors import InfraError
from .l1 import probe

VENDOR_ENCODE = Path(__file__).parent / "vendor" / "ig_encode.sh"


class EncodeFailed(Exception):
    pass


def encode_silent(clip_path: str | Path, out_path: str | Path,
                  timeout_s: int = 600) -> dict:
    """Encode one clip to the 1080x1920/30fps silent QC MP4. Returns the verified
    container dict. EncodeFailed on any spec mismatch (a bad encode must not be
    reported as a shipped clip)."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["bash", str(VENDOR_ENCODE), "-o", str(out), str(clip_path)]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except FileNotFoundError:
        raise InfraError("l1", "bash/ffmpeg missing — cannot run ig_encode.sh")
    except subprocess.TimeoutExpired:
        raise EncodeFailed(f"ig_encode.sh exceeded {timeout_s}s")
    if res.returncode != 0 or not out.exists():
        raise EncodeFailed(f"ig_encode.sh failed: {(res.stderr or res.stdout)[-500:]}")

    container = probe(out)
    problems = []
    if (container["width"], container["height"]) != (1080, 1920):
        problems.append(f"size {container['width']}x{container['height']} != 1080x1920")
    if container["vcodec"] != "h264":
        problems.append(f"vcodec {container['vcodec']} != h264")
    if container["fps"] is None or abs(container["fps"] - 30.0) > 0.5:
        problems.append(f"fps {container['fps']} != 30")
    if not _has_audio_stream(out):
        problems.append("no audio stream (IG expects one, silent track included)")
    if problems:
        raise EncodeFailed("encoded file fails spec: " + "; ".join(problems))
    return container


def _has_audio_stream(path: Path) -> bool:
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
         "stream=codec_type", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, timeout=60)
    return "audio" in (res.stdout or "")
