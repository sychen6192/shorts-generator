"""Event log + fold tests (resume foundation)."""

from __future__ import annotations

import json

from shortsloop.state import RunLog, fold


def test_events_and_attempts_are_appended_jsonl(tmp_path):
    log = RunLog(tmp_path / "run", "test-run")
    log.event("schedule", "ok")
    log.event("generate", "submitted", "V1C1", 1, prompt_id="abc", seed=42)
    log.attempt({"clip_id": "V1C1", "attempt": 1, "verdict": "FAIL",
                 "failure_classes": ["static"], "status": "l1_failed"})
    events = log.read_events()
    assert [e["event"] for e in events] == ["ok", "submitted"]
    assert events[1]["data"]["prompt_id"] == "abc"
    attempts = log.read_attempts()
    assert attempts[0]["failure_classes"] == ["static"]
    # append-only plain JSONL: one JSON object per line, readable with less
    raw = (tmp_path / "run" / "events.jsonl").read_text().splitlines()
    assert all(json.loads(line) for line in raw)


def test_fold_passed_and_in_flight(tmp_path):
    log = RunLog(tmp_path / "run", "r")
    # clip A passed on attempt 2
    log.attempt({"clip_id": "A", "attempt": 1, "verdict": "FAIL",
                 "failure_classes": ["static"], "status": "l1_failed",
                 "prompt_text": "p1"})
    log.attempt({"clip_id": "A", "attempt": 2, "verdict": "PASS",
                 "failure_classes": [], "status": "passed",
                 "prompt_text": "p1x", "verdict_path": "v/A_a2.json",
                 "output_path": "c/A_a2.mp4"})
    # clip B crashed mid-generation: submitted, never resolved
    log.event("generate", "submitted", "B", 1, prompt_id="pid-b", seed=7)
    folded = fold(log)
    assert folded["A"].status == "passed"
    assert folded["A"].passed_attempt == 2
    assert folded["A"].attempts_used == 2
    assert folded["A"].clip_path == "c/A_a2.mp4"
    assert folded["B"].in_flight["prompt_id"] == "pid-b"
    assert folded["B"].in_flight["attempt"] == 1


def test_fold_survives_torn_last_line(tmp_path):
    log = RunLog(tmp_path / "run", "r")
    log.attempt({"clip_id": "A", "attempt": 1, "verdict": "PASS",
                 "failure_classes": [], "status": "passed"})
    with open(tmp_path / "run" / "attempts.jsonl", "a") as f:
        f.write('{"clip_id": "B", "attempt": 1, "verd')   # torn write mid-crash
    folded = fold(log)
    assert folded["A"].status == "passed"
    assert "B" not in folded
