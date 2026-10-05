"""One GPU user per runs directory: the nightly runner, calibrate-batch and
calibrate-tune --with-l2 all take runs_dir/.shortsloop.lock (hard rule 4 starts
before ComfyUI: two processes would each believe they own the queue and VRAM)."""

from __future__ import annotations

import fcntl
from pathlib import Path

LOCK_NAME = ".shortsloop.lock"


class LockHeld(Exception):
    pass


def runs_base(cfg: dict, override: str | Path | None = None) -> Path:
    return Path(override) if override else Path((cfg.get("paths") or {})
                                                 .get("runs_dir", "runs"))


def acquire(base: str | Path):
    """Exclusive, non-blocking. Returns the open handle (keep it alive); raises
    LockHeld if another process holds it."""
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    fh = open(base / LOCK_NAME, "a+")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        raise LockHeld(str(base / LOCK_NAME))
    return fh


def release(fh) -> None:
    if fh is not None:
        fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()
