"""Persist stage: silent QC encode of PASSING clips via vendored ig_encode.sh,
then ffprobe re-verification (the encode script's own check plus ours — trust
commands, not eyes)."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from .errors import CheckError, InfraError
from .l1 import probe

VENDOR_ENCODE = Path(__file__).parent / "vendor" / "ig_encode.sh"


class EncodeFailed(Exception):
    pass


SILENCE_MAX_DB = -80.0     # anullsrc measures -inf / ~-91 dB


def encode_silent(clip_path: str | Path, out_path: str | Path,
                  timeout_s: int = 600) -> dict:
    """Encode one clip to the 1080x1920/30fps silent QC MP4. Returns the verified
    container dict. EncodeFailed on any spec mismatch — and then NOTHING is left at
    out_path (encoded/ holds verified PASS encodes only, plan §2.7): the encode
    goes to a temp file that is renamed into place only after verification."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.stem}.tmp{out.suffix}")
    tmp.unlink(missing_ok=True)
    try:
        container = _encode_and_verify(clip_path, tmp, timeout_s)
        os.replace(tmp, out)
        return container
    except subprocess.TimeoutExpired as e:      # a verification step hung
        raise EncodeFailed(f"encode verification timed out: {e.cmd[0]} >{e.timeout}s")
    finally:
        tmp.unlink(missing_ok=True)


def _encode_and_verify(clip_path, tmp: Path, timeout_s: int) -> dict:
    cmd = ["bash", str(VENDOR_ENCODE), "-o", str(tmp), str(clip_path)]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except FileNotFoundError:
        raise InfraError("l1", "bash/ffmpeg missing — cannot run ig_encode.sh")
    except subprocess.TimeoutExpired:
        raise EncodeFailed(f"ig_encode.sh exceeded {timeout_s}s")
    if res.returncode != 0 or not tmp.exists():
        raise EncodeFailed(f"ig_encode.sh failed: {(res.stderr or res.stdout)[-500:]}")

    try:
        container = probe(tmp)
    except CheckError as e:
        raise EncodeFailed(f"encoded file unreadable: {e.message}")
    problems = []
    if (container["width"], container["height"]) != (1080, 1920):
        problems.append(f"size {container['width']}x{container['height']} != 1080x1920")
    if container["vcodec"] != "h264":
        problems.append(f"vcodec {container['vcodec']} != h264")
    if container["fps"] is None or abs(container["fps"] - 30.0) > 0.5:
        problems.append(f"fps {container['fps']} != 30")
    acodec = _audio_codec(tmp)
    if acodec is None:
        problems.append("no audio stream (IG expects one, silent track included)")
    else:
        if acodec != "aac":
            problems.append(f"audio codec {acodec} != aac")
        peak = _max_volume_db(tmp)
        if peak is None or peak > SILENCE_MAX_DB:
            problems.append(f"audio is not silent (max_volume {peak} dB > "
                            f"{SILENCE_MAX_DB} dB) — QC encodes must be silent")
    if problems:
        raise EncodeFailed("encoded file fails spec: " + "; ".join(problems))
    return container


def _audio_codec(path: Path) -> str | None:
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=codec_name", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, timeout=60)
    return (res.stdout or "").strip() or None


def _max_volume_db(path: Path) -> float | None:
    """Peak level of the first audio stream via ffmpeg volumedetect (-inf for
    digital silence). None if it cannot be measured (treated as not silent)."""
    res = subprocess.run(
        ["ffmpeg", "-nostats", "-hide_banner", "-i", str(path), "-map", "0:a:0",
         "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True, text=True, timeout=120)
    m = re.search(r"max_volume:\s*(-?inf|-?\d+(?:\.\d+)?)\s*dB", res.stderr or "")
    if not m:
        return None
    return float("-inf") if "inf" in m.group(1) else float(m.group(1))
