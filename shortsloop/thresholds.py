"""thresholds.yaml — FROZEN shape (docs/plan.md §2.4) and its validator.

Thresholds are instrument settings: a file that is off-shape is a broken
instrument, never a quietly weaker one. Every reader (checker, runner, doctor,
tuner --approve) goes through validate_thresholds(); any problem is fail-closed.
"""

from __future__ import annotations

import math

from .policy import L2_DIM_CLASS

# check name -> (metric, op). Values are calibrated; names/metrics/ops are not.
FROZEN_L1 = {
    "motion":     ("flow_mag_median", ">="),
    "freeze":     ("freeze_longest_run_s", "<="),
    "flicker":    ("flicker_dips", "<="),
    "ssim_floor": ("ssim_min", ">="),
    "sharpness":  ("laplacian_p10", ">="),
    "black":      ("black_frame_frac", "<="),
    "exposure":   ("clipped_frac", "<="),
}
L2_DIMS = tuple(L2_DIM_CLASS)


def _real(v) -> bool:
    return type(v) in (int, float) and math.isfinite(v)


def validate_thresholds(data) -> list[str]:
    """Problems with a parsed thresholds.yaml (empty list = valid)."""
    if not isinstance(data, dict):
        return ["thresholds file is not a mapping"]
    errs: list[str] = []
    if not isinstance(data.get("version"), str) or not data["version"].strip():
        errs.append("version must be a non-empty string")
    if type(data.get("calibrated")) is not bool:
        errs.append(f"calibrated must be true or false (YAML boolean), "
                    f"got {data.get('calibrated')!r}")
    if "provenance" in data and not isinstance(data["provenance"], dict):
        errs.append("provenance must be a mapping")

    l1 = data.get("l1")
    if not isinstance(l1, dict):
        errs.append("l1 section missing or not a mapping")
    else:
        for name in sorted(set(FROZEN_L1) - set(l1)):
            errs.append(f"l1.{name} missing (frozen check set: {sorted(FROZEN_L1)})")
        for name in sorted(set(l1) - set(FROZEN_L1)):
            errs.append(f"l1.{name} is not a frozen check")
        for name, (metric, op) in FROZEN_L1.items():
            spec = l1.get(name)
            if name not in l1:
                continue
            if not isinstance(spec, dict):
                errs.append(f"l1.{name} must be a mapping {{metric, op, value}}")
                continue
            if spec.get("metric") != metric or spec.get("op") != op:
                errs.append(f"l1.{name} must be {{metric: {metric}, op: '{op}'}}, got "
                            f"{{metric: {spec.get('metric')}, op: {spec.get('op')!r}}}")
            if not _real(spec.get("value")):
                errs.append(f"l1.{name}.value must be a finite number, "
                            f"got {spec.get('value')!r}")

    floors = (data.get("l2") or {}).get("floors") if isinstance(data.get("l2"), dict) \
        else None
    if not isinstance(floors, dict):
        errs.append("l2.floors missing or not a mapping")
    else:
        for dim in L2_DIMS:
            f = floors.get(dim)
            if type(f) is not int or not (1 <= f <= 5):
                errs.append(f"l2.floors.{dim} must be an integer 1-5, got {f!r}")
        for dim in sorted(set(floors) - set(L2_DIMS)):
            errs.append(f"l2.floors.{dim} is not a rubric dimension")
    return errs
