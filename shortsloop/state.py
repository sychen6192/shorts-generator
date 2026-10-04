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
    status: str = "pending"          # pending | awaiting_l2 | passed
    last_classes: list[str] = field(default_factory=list)
    prompt_current: str | None = None  # BASE prompt for the next attempt (rewritten
                                       # or None = the sheet's); never carries a
                                       # per-attempt motion phrase
    rewritten: bool = False
    rewrite_diff: str | None = None
    passed_attempt: int | None = None
    verdict_path: str | None = None
    clip_path: str | None = None
    last_verdict_path: str | None = None   # verdict of the last FINAL attempt
    # The latest attempt if it has no final attempts.jsonl line — a crash/halt
    # interrupted it. {"attempt", "stage": in_flight|generated|awaiting, "seed",
    # "steps", "prompt_id", "file", "record"} — the runner resumes it at that
    # stage instead of paying for a regeneration (plan §5 row 14).
    recover: dict | None = None
    in_flight: dict | None = None    # == recover when stage == "in_flight"


# attempts.jsonl `status` values that are NOT a final outcome for that attempt.
NON_FINAL_STATUSES = ("unjudged",)


def fold(log: RunLog) -> dict[str, ClipState]:
    """Rebuild per-clip state from events + attempts.

    attempts.jsonl is authoritative for outcomes; when an attempt has several lines
    (an `unjudged` line written at a halt, then the final line after --resume), the
    LAST line wins. events.jsonl fills in attempts whose line was never written (hard
    crash): submitted -> re-attach, downloaded -> run the L1 gate on the file,
    L1 PROCEED -> straight into the judge wave."""
    clips: dict[str, ClipState] = {}

    def st(cid: str) -> ClipState:
        return clips.setdefault(cid, ClipState(clip_id=cid))

    gen: dict[tuple[str, int], dict] = {}
    for ev in log.read_events():
        cid, att = ev.get("clip_id"), ev.get("attempt")
        if not cid or not att:
            continue
        data = ev.get("data") or {}
        stage, event = ev.get("stage"), ev.get("event")
        g = gen.setdefault((cid, att), {})
        if stage == "generate" and event == "enter":
            g.setdefault("seed", data.get("seed"))
            g.setdefault("steps", data.get("steps"))
        elif stage == "generate" and event == "submitted":
            g["submitted"] = True
            g["prompt_id"] = data.get("prompt_id")
            if data.get("seed") is not None:
                g["seed"] = data["seed"]
            if "steps" in data:
                g["steps"] = data["steps"]
        elif stage == "generate" and event == "ok":
            g["generated"] = True
            g["file"] = data.get("file")
        elif stage == "generate" and event in ("fail", "error"):
            g["gen_failed"] = True
        elif (stage == "verify" and event == "ok" and data.get("layer") == "l1"
              and data.get("verdict") == "PROCEED"):
            g["l1_proceed"] = True

    records: dict[str, list[dict]] = {}
    for rec in log.read_attempts():
        if rec.get("clip_id") and rec.get("attempt"):
            records.setdefault(rec["clip_id"], []).append(rec)

    for cid in sorted({c for c, _ in gen} | set(records)):
        recs = records.get(cid, [])
        last_by_attempt = {rec["attempt"]: rec for rec in recs}   # last line wins
        consumed = set(last_by_attempt) | {
            n for (c, n), g in gen.items()
            if c == cid and (g.get("submitted") or g.get("generated"))}
        if not consumed:
            continue
        s = st(cid)
        s.attempts_used = max(consumed)
        for rec in recs:
            if rec.get("prompt_base"):
                s.prompt_current = rec["prompt_base"]
            elif rec.get("prompt_rewritten"):
                s.prompt_current = rec.get("prompt_text")  # pre-prompt_base logs
            if rec.get("prompt_rewritten"):
                s.rewritten = True
            if rec.get("prompt_diff"):
                s.rewrite_diff = rec["prompt_diff"]
        for n in sorted(last_by_attempt):
            rec = last_by_attempt[n]
            if rec.get("status") not in NON_FINAL_STATUSES:
                s.last_classes = rec.get("failure_classes") or []
                s.last_verdict_path = rec.get("verdict_path")

        latest = s.attempts_used
        rec = last_by_attempt.get(latest)
        g = gen.get((cid, latest), {})
        if rec is not None and rec.get("status") not in NON_FINAL_STATUSES:
            if rec.get("verdict") == "PASS":
                s.status = "passed"
                s.passed_attempt = latest
                s.verdict_path = rec.get("verdict_path")
                s.clip_path = rec.get("output_path")
            continue
        info = {"attempt": latest, "seed": g.get("seed"), "steps": g.get("steps"),
                "prompt_id": g.get("prompt_id"), "file": g.get("file"),
                "record": rec}
        if rec is not None:                      # `unjudged` line from a halt
            info.update(stage="awaiting", seed=rec.get("seed"), steps=rec.get("steps"),
                        prompt_id=rec.get("comfy_prompt_id"),
                        file=rec.get("output_path"))
        elif g.get("l1_proceed") and g.get("file"):
            info["stage"] = "awaiting"
        elif g.get("generated") and g.get("file"):
            info["stage"] = "generated"
        elif g.get("submitted") and not g.get("gen_failed"):
            info["stage"] = "in_flight"
        else:
            # generation failed but the crash beat the attempt line: the attempt
            # was consumed (GPU time spent) and counts as `broken`.
            s.last_classes = ["broken"]
            continue
        s.recover = info
        if info["stage"] == "awaiting":
            s.status = "awaiting_l2"
        if info["stage"] == "in_flight":
            s.in_flight = info
    return clips


def active_seconds(events: list[dict]) -> float:
    """Wall-clock the runner was actually working, summed over sessions (a new
    session starts at each `schedule enter` — the original run and every
    --resume). Hard rule 5's budget is whole-run: a resume is charged for it,
    but not for the hours the machine sat crashed in between."""
    total, first, last = 0.0, None, None
    for ev in events:
        ts = ev.get("ts")
        if not isinstance(ts, (int, float)):
            continue
        new_session = ev.get("stage") == "schedule" and (
            ev.get("event") == "enter" or (ev.get("data") or {}).get("data_resumed"))
        if new_session and first is not None:
            total += last - first
            first = None
        if first is None:
            first = ts
        last = ts
    if first is not None:
        total += last - first
    return total
