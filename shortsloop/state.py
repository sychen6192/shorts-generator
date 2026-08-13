"""Run state on disk: append-only JSONL, event-sourced, `less`-able (hard rule 6).

- events.jsonl   — every stage transition: {ts, run_id, clip_id, attempt, stage,
                   event, data}; stage names are exactly the pipeline manifest
                   vocabulary (schedule claim generate verify persist complete).
- attempts.jsonl — one line per generation attempt with everything needed to
                   reproduce the clip through vendor/comfy_client.py.
- resume         — fold events + attempts to rebuild in-flight state.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

STAGES = ("schedule", "claim", "generate", "verify", "persist", "complete")
EVENTS = ("enter", "ok", "fail", "error", "skip", "submitted", "halt")


class RunLog:
    def __init__(self, run_dir: str | Path, run_id: str):
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._events_path = self.run_dir / "events.jsonl"
        self._attempts_path = self.run_dir / "attempts.jsonl"

    def _append(self, path: Path, obj: dict) -> None:
        line = json.dumps(obj, ensure_ascii=False)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())

    def event(self, stage: str, event: str, clip_id: str | None = None,
              attempt: int | None = None, **data) -> None:
        assert stage in STAGES, stage
        assert event in EVENTS, event
        self._append(self._events_path, {
            "ts": round(time.time(), 3), "run_id": self.run_id,
            "clip_id": clip_id, "attempt": attempt,
            "stage": stage, "event": event, "data": data or {},
        })

    def attempt(self, record: dict) -> None:
        record = {"ts": round(time.time(), 3), **record}
        self._append(self._attempts_path, record)

    # ---- folding (resume/report) -------------------------------------------
    def read_events(self) -> list[dict]:
        return _read_jsonl(self._events_path)

    def read_attempts(self) -> list[dict]:
        return _read_jsonl(self._attempts_path)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            # A torn final line from a crash is expected; anything else is not,
            # but resume must still work — the torn line is simply not state.
            continue
    return out


@dataclass
class ClipState:
    """Folded view of one clip, rebuilt from the logs on resume."""
    clip_id: str
    attempts_used: int = 0
    status: str = "pending"          # pending | awaiting_l2 | passed | failed_final
                                     # | skipped | error
    last_classes: list[str] = field(default_factory=list)
    prompt_current: str | None = None
    rewritten: bool = False
    passed_attempt: int | None = None
    verdict_path: str | None = None
    clip_path: str | None = None
    in_flight: dict | None = None    # {"attempt": n, "prompt_id": ..., "seed": ...}


def fold(log: RunLog) -> dict[str, ClipState]:
    """Rebuild per-clip state from events + attempts. Attempts are authoritative
    for consumed attempts/verdicts; events supply in-flight generation info."""
    clips: dict[str, ClipState] = {}

    def st(cid: str) -> ClipState:
        return clips.setdefault(cid, ClipState(clip_id=cid))

    submitted: dict[tuple[str, int], dict] = {}
    for ev in log.read_events():
        cid, att = ev.get("clip_id"), ev.get("attempt")
        if not cid:
            continue
        key = (cid, att or 0)
        if ev["stage"] == "generate" and ev["event"] == "submitted":
            submitted[key] = {"attempt": att, **ev.get("data", {})}
        if ev["stage"] == "generate" and ev["event"] in ("ok", "fail", "error"):
            submitted.pop(key, None)
        if ev["stage"] == "claim" and ev["event"] == "skip":
            s = st(cid)
            if s.status == "pending":
                s.status = "skipped"

    for rec in log.read_attempts():
        s = st(rec["clip_id"])
        s.attempts_used = max(s.attempts_used, rec.get("attempt", 0))
        s.prompt_current = rec.get("prompt_text", s.prompt_current)
        s.rewritten = bool(rec.get("prompt_rewritten", s.rewritten))
        verdict = rec.get("verdict")
        s.last_classes = rec.get("failure_classes") or []
        if verdict == "PASS":
            s.status = "passed"
            s.passed_attempt = rec.get("attempt")
            s.verdict_path = rec.get("verdict_path")
            s.clip_path = rec.get("output_path")
        elif verdict == "PROCEED":
            s.status = "awaiting_l2"
            s.verdict_path = rec.get("verdict_path")
            s.clip_path = rec.get("output_path")
        elif s.status not in ("passed",):
            s.status = "pending"

    for (cid, _att), info in submitted.items():
        s = st(cid)
        if s.status in ("pending", "awaiting_l2"):
            s.in_flight = info
    return clips
