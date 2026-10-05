"""thresholds.yaml is an instrument setting: its frozen shape (plan §2.4) is
validated everywhere it is read, and anything off-shape fails closed."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from conftest import PERMISSIVE_L1

from shortsloop.check import load_thresholds
from shortsloop.errors import InfraError
from shortsloop.thresholds import validate_thresholds

REPO = Path(__file__).resolve().parents[1]


def _data(**over):
    data = {"version": "t.1", "calibrated": True, "provenance": {},
            "l1": {k: dict(v) for k, v in PERMISSIVE_L1.items()},
            "l2": {"floors": {d: 3 for d in ("prompt_adherence", "subject_consistency",
                                            "anatomy_artifacts", "temporal_coherence",
                                            "imaging_quality")}}}
    data.update(over)
    return data


def test_repo_placeholder_and_test_fixture_validate(tmp_path):
    assert validate_thresholds(yaml.safe_load((REPO / "thresholds.yaml").read_text())) == []
    assert validate_thresholds(_data()) == []


@pytest.mark.parametrize("mutate, needle", [
    (lambda d: d["l1"].pop("motion"), "motion"),
    (lambda d: d["l1"].update(extra={"metric": "luma_mean", "op": ">=", "value": 0}),
     "extra"),
    (lambda d: d["l1"]["flicker"].update(value=None), "flicker"),
    (lambda d: d["l1"]["flicker"].update(value=float("inf")), "flicker"),
    (lambda d: d["l1"]["sharpness"].update(value=float("nan")), "sharpness"),
    (lambda d: d["l1"]["black"].update(value=True), "black"),
    (lambda d: d["l1"]["black"].update(value="0.1"), "black"),
    (lambda d: d["l1"]["motion"].update(op="<="), "motion"),
    (lambda d: d["l1"]["motion"].update(metric="luma_mean"), "motion"),
    (lambda d: d["l2"]["floors"].pop("imaging_quality"), "imaging_quality"),
    (lambda d: d["l2"]["floors"].update(prompt_adherence=6), "prompt_adherence"),
    (lambda d: d["l2"]["floors"].update(prompt_adherence=True), "prompt_adherence"),
    (lambda d: d["l2"]["floors"].update(prompt_adherence=2.5), "prompt_adherence"),
    (lambda d: d.update(calibrated="false"), "calibrated"),
    (lambda d: d.update(calibrated=None), "calibrated"),
    (lambda d: d.pop("l2"), "l2"),
])
def test_off_shape_thresholds_rejected(mutate, needle):
    d = _data()
    mutate(d)
    problems = validate_thresholds(d)
    assert problems and any(needle in p for p in problems), problems


def test_loader_fails_closed_on_off_shape_file(tmp_path):
    d = _data(calibrated="false")            # a human reads this as "not calibrated"
    p = tmp_path / "thr.yaml"
    p.write_text(yaml.safe_dump(d))
    with pytest.raises(InfraError):
        load_thresholds(p)


def test_checker_with_missing_check_is_infra_error_not_proceed(clips, prompt_file,
                                                               tmp_path):
    """A thresholds file missing motion+freeze must not let a static clip PROCEED."""
    d = _data()
    d["l1"].pop("motion")
    d["l1"].pop("freeze")
    thr = tmp_path / "thr.yaml"
    thr.write_text(yaml.safe_dump(d))
    out = tmp_path / "v.json"
    res = subprocess.run([sys.executable, "-m", "shortsloop.check", str(clips["static"]),
                          "--prompt-file", str(prompt_file), "--json", str(out),
                          "--thresholds", str(thr), "--l1-only"],
                         capture_output=True, text=True)
    assert res.returncode == 2
    v = json.loads(out.read_text())
    assert v["verdict"] == "ERROR" and v["error"]["scope"] == "infra"
