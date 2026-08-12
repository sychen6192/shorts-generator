"""L1 metric direction tests (plan §5 row 15): each metric must separate its fixture
pair with a wide margin. Exact decision values are calibrated in Phase 0, not here."""

from __future__ import annotations

import pytest

from shortsloop import l1
from shortsloop.errors import ClipError, InfraError


def test_motion_separates_static_from_moving(all_metrics):
    static = all_metrics["static"]["metrics"]["flow_mag_median"]
    moving = all_metrics["moving"]["metrics"]["flow_mag_median"]
    assert static < 0.0008, f"static clip should have near-zero flow, got {static}"
    assert moving > 0.002, f"moving clip should have clear flow, got {moving}"
    assert moving > 3 * static


def test_flicker_dips_fire_on_flicker_only(all_metrics):
    assert all_metrics["flicker"]["metrics"]["flicker_dips"] >= 6
    assert all_metrics["moving"]["metrics"]["flicker_dips"] <= 1
    assert (all_metrics["flicker"]["metrics"]["ssim_min"]
            < all_metrics["moving"]["metrics"]["ssim_min"])


def test_black_frame_fraction(all_metrics):
    assert all_metrics["black"]["metrics"]["black_frame_frac"] > 0.9
    assert all_metrics["moving"]["metrics"]["black_frame_frac"] < 0.05


def test_sharpness_separates_blurry(all_metrics):
    sharp = all_metrics["moving"]["metrics"]["laplacian_p10"]
    blurry = all_metrics["blurry"]["metrics"]["laplacian_p10"]
    assert blurry < 0.4 * sharp, f"blurry {blurry} vs sharp {sharp}"


def test_freeze_tail_detected(all_metrics):
    assert all_metrics["freeze_tail"]["metrics"]["freeze_longest_run_s"] >= 0.8
    assert all_metrics["moving"]["metrics"]["freeze_longest_run_s"] < 0.3
    # freeze also registers as low median flow only when the frozen span dominates;
    # the dedicated freeze metric is what must catch a moving-then-frozen clip.


def test_metrics_are_deterministic(clips):
    a = l1.compute_metrics(clips["moving"])
    b = l1.compute_metrics(clips["moving"])
    for key, val in a["metrics"].items():
        assert val == pytest.approx(b["metrics"][key], abs=1e-9), key


def test_probe_reports_container(clips):
    c = l1.probe(clips["moving"])
    assert c["width"] == 480 and c["height"] == 832
    assert c["fps"] == pytest.approx(16.0, abs=0.01)
    assert c["vcodec"] == "h264"


def test_probe_rejects_corrupt(clips):
    with pytest.raises(ClipError):
        l1.probe(clips["corrupt"])


def test_probe_rejects_missing():
    with pytest.raises(ClipError):
        l1.probe("/nonexistent/nope.mp4")


def test_spec_check_against_expectations(clips, all_metrics):
    container = l1.probe(clips["moving"])
    frames = all_metrics["moving"]["analysis"]["frames_analyzed"]
    good = l1.spec_check(container, frames, {"width": 480, "height": 832, "fps": 16,
                                             "frames": 48, "duration_s": 3.0})
    assert good["pass"], good["reason"]
    bad = l1.spec_check(container, frames, {"width": 720, "height": 1280})
    assert not bad["pass"]
    assert "width" in bad["reason"]


def test_run_checks_flags_static(all_metrics):
    thresholds = {"motion": {"metric": "flow_mag_median", "op": ">=", "value": 0.0015}}
    checks = l1.run_checks(all_metrics["static"]["metrics"], thresholds)
    assert len(checks) == 1 and not checks[0]["pass"]
    checks_ok = l1.run_checks(all_metrics["moving"]["metrics"], thresholds)
    assert checks_ok[0]["pass"]


def test_run_checks_rejects_malformed_thresholds(all_metrics):
    with pytest.raises(InfraError):
        l1.run_checks(all_metrics["moving"]["metrics"],
                      {"motion": {"metric": "no_such_metric", "op": ">=", "value": 0}})
    with pytest.raises(InfraError):
        l1.run_checks(all_metrics["moving"]["metrics"],
                      {"motion": {"metric": "flow_mag_median", "op": "!!", "value": 0}})
