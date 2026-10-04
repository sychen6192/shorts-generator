"""Frozen verdict schema v1 (plan §2.2, §5 row 17): golden verdicts of every kind
validate; off-contract verdicts are rejected by schema + invariants."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from shortsloop.schema import validate_verdict_file

GOLDEN = Path(__file__).parent / "data" / "golden"


def golden(name: str) -> dict:
    return json.loads((GOLDEN / f"{name}.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("name, verdict", [
    ("pass", "PASS"), ("fail_l1", "FAIL"), ("fail_l2", "FAIL"),
    ("proceed", "PROCEED"), ("error_clip", "ERROR"),
])
def test_golden_verdicts_validate(name, verdict):
    v = golden(name)
    assert v["verdict"] == verdict
    assert validate_verdict_file(v) == []


def _set(path: str, value):
    def mutate(v):
        *head, last = path.split(".")
        node = v
        for key in head:
            node = node[int(key)] if isinstance(node, list) else node[key]
        if isinstance(node, list):
            node[int(last)] = value
        else:
            node[last] = value
    return mutate


def _drop(path: str):
    def mutate(v):
        *head, last = path.split(".")
        node = v
        for key in head:
            node = node[key]
        del node[last]
    return mutate


MUTATIONS = [
    ("pass", _set("surprise", 1), "unexpected top-level key"),
    ("pass", _set("l2", None), "PASS without l2"),
    ("pass", _set("layers_run", ["l1"]), "PASS with layers_run l1"),
    ("pass", _set("failure_classes", ["blurry"]), "PASS carrying classes"),
    ("pass", _set("l2.dimensions.prompt_adherence.score", 2), "dim pass inconsistent"),
    ("pass", _set("l2.dimensions.prompt_adherence.floor", 7), "floor out of range"),
    ("pass", _set("l2.dimensions.prompt_adherence.score", True), "bool score"),
    ("pass", _set("l2.dimensions.imaging_quality.extra", 1), "extra dimension key"),
    ("pass", _set("l2.pass", False), "l2.pass inconsistent with dimensions"),
    ("pass", _set("l1.pass", False), "l1.pass inconsistent with checks"),
    ("pass", _set("l1.analysis.width", 720), "non-canonical analysis width"),
    ("pass", _drop("l1.metrics.ssim_min"), "missing metric"),
    ("pass", _set("l1.metrics.ssim_min", "0.9"), "string metric"),
    ("pass", _set("l1.metrics.bogus", 1.0), "unknown metric"),
    ("pass", _set("l1.checks.0.name", "vibes"), "unknown check name"),
    ("pass", _drop("versions.ffprobe"), "versions missing ffprobe"),
    ("pass", _set("thresholds.calibrated", "true"), "string calibrated"),
    ("pass", _set("l2.frames", [{"index": i, "t": i / 16} for i in range(9)]),
     "more than 8 judge frames"),
    ("fail_l1", _set("failure_classes", ["blurry", "static"]), "classes out of priority"),
    ("fail_l1", _set("failure_classes", ["static", "static"]), "duplicate classes"),
    ("fail_l1", _set("failure_classes", ["ugly"]), "unknown class"),
    ("fail_l1", _set("l2", golden("pass")["l2"]), "l2 block without l2 in layers_run"),
    ("error_clip", _set("failure_classes", ["broken"]), "ERROR carrying classes"),
    ("error_clip", _set("error.stage", "l3"), "bad error stage"),
    ("error_clip", _set("error.detail", "x"), "extra error key"),
    ("proceed", _set("verdict", "PASS"), "l1-only result relabelled PASS"),
]


@pytest.mark.parametrize("name, mutate, why", MUTATIONS, ids=[m[2] for m in MUTATIONS])
def test_off_contract_verdicts_rejected(name, mutate, why):
    v = copy.deepcopy(golden(name))
    mutate(v)
    assert validate_verdict_file(v), f"accepted a verdict with {why}"
