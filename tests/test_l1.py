"""L1 metric direction tests (plan §5 row 15): each metric must separate its fixture
pair with a wide margin. Exact decision values are calibrated in Phase 0, not here."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

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


def test_spec_check_fails_closed_on_unverifiable_container():
    """An expectation the container cannot confirm is a spec failure, not a skip."""
    exp = {"width": 480, "height": 832, "fps": 16.0, "frames": 48, "duration_s": 3.0}
    base = {"width": 480, "height": 832, "fps": 16.0, "nb_frames": 48,
            "duration_s": 3.0, "vcodec": "h264"}
    assert l1.spec_check(base, 48, exp)["pass"] is True
    assert l1.spec_check({**base, "fps": None}, 48, exp)["pass"] is False
    assert l1.spec_check({**base, "duration_s": None}, 48, exp)["pass"] is False


def test_spec_check_catches_truncation_against_container_header():
    """moov says 48 frames but only 20 decode (truncated faststart MP4): spec
    failure even with no dispatch expectation (plan §5 row 4)."""
    c = {"width": 480, "height": 832, "fps": 16.0, "nb_frames": 48,
         "duration_s": 3.0, "vcodec": "h264"}
    res = l1.spec_check(c, 20, None)
    assert res["pass"] is False and "decoded" in res["reason"]
    assert l1.spec_check(c, 47, None)["pass"] is True        # encoder off-by-one ok


# --- dense flicker (audit: alternate-frame strobing / bursts scored 0 dips) ----------

def test_dense_flicker_is_counted(all_metrics):
    """Alternate-frame strobing (every pair low) and a ~1 s strobe burst must register
    as flicker — a rolling median that sinks with the dips used to hide them."""
    moving = all_metrics["moving"]["metrics"]["flicker_dips"]
    for kind in ("strobe", "strobe_burst"):
        dips = all_metrics[kind]["metrics"]["flicker_dips"]
        assert dips >= 6, f"{kind}: dense flicker invisible to L1 (flicker_dips={dips})"
        assert dips >= moving + 5, f"{kind}={dips} vs moving={moving}"


def test_flicker_dips_near_zero_on_flicker_free_kinds(all_metrics):
    for kind in ("static", "moving", "blurry", "black", "freeze_tail"):
        dips = all_metrics[kind]["metrics"]["flicker_dips"]
        assert dips == 0, f"{kind} should be flicker-free, got flicker_dips={dips}"


# The flicker rule as shipped in the placeholder thresholds.yaml. Pinned here rather
# than read from the repo file: Phase 0 sign-off (`calibrate-tune --approve`)
# rewrites thresholds.yaml on the workstation, and the suite must stay green there.
PLACEHOLDER_FLICKER = {"metric": "flicker_dips", "op": "<=", "value": 2}


def test_placeholder_flicker_pin_matches_shipped_file():
    """While the repo thresholds.yaml is still the uncalibrated placeholder, the pin
    above must equal its flicker rule. A signed-off file belongs to the operator and
    is not compared (its shape is checked in test_thresholds.py)."""
    shipped = yaml.safe_load(
        (Path(__file__).parents[1] / "thresholds.yaml").read_text(encoding="utf-8"))
    if shipped["calibrated"] is not True:
        assert shipped["l1"]["flicker"] == PLACEHOLDER_FLICKER


def test_placeholder_flicker_rule_fails_strobes(all_metrics):
    """With the shipped placeholder flicker rule (flicker_dips <= 2) both strobe
    kinds FAIL the flicker check, while the clean moving clip passes it."""
    flicker_only = {"flicker": PLACEHOLDER_FLICKER}
    for kind in ("strobe", "strobe_burst", "flicker"):
        (chk,) = l1.run_checks(all_metrics[kind]["metrics"], flicker_only)
        assert chk["pass"] is False, f"{kind} passed the flicker check: {chk}"
    (ok,) = l1.run_checks(all_metrics["moving"]["metrics"], flicker_only)
    assert ok["pass"] is True


def _flat_ssims(n_frames, value=0.70):
    return np.full(n_frames - 1, value)


@pytest.mark.parametrize("period", [2, 3])
def test_periodic_luma_flicker_counted(period):
    """Period-2/3 brightness flicker with a flat SSIM trace (the hard case for a
    rolling-median dip detector) is counted via luma reversals."""
    n = 48
    lumas = [0.5 + (0.25 if i % period == period - 1 else 0.0) for i in range(n)]
    assert l1.flicker_dips(_flat_ssims(n), lumas) >= 10


@pytest.mark.parametrize("lumas", [
    [0.05 + 0.025 * i for i in range(36)],              # fade: big but monotonic
    [0.3] * 24 + [0.7] * 24,                             # one hard cut: no reversal
    [0.5 + (0.01 if i % 2 else 0.0) for i in range(48)],  # sub-threshold shimmer
])
def test_luma_term_ignores_fades_cuts_and_shimmer(lumas):
    assert l1.flicker_dips(_flat_ssims(len(lumas)), lumas) == 0


def test_isolated_ssim_dip_still_counted():
    ssims = np.full(47, 0.70)
    ssims[20] = 0.60
    assert l1.flicker_dips(ssims, [0.5] * 48) == 1
