from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent))  # make `synth` importable

from synth import KINDS, make_clip, make_corrupt  # noqa: E402

from shortsloop import l1  # noqa: E402


@pytest.fixture(scope="session")
def clips(tmp_path_factory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("clips")
    out = {kind: make_clip(kind, d / f"{kind}.mp4") for kind in KINDS}
    out["corrupt"] = make_corrupt(d / "corrupt.mp4")
    return out


@pytest.fixture(scope="session")
def all_metrics(clips) -> dict[str, dict]:
    """L1 metrics computed once per fixture kind (decode is the slow part)."""
    return {kind: l1.compute_metrics(clips[kind]) for kind in KINDS}


PERMISSIVE_L1 = {
    "motion":     {"metric": "flow_mag_median",      "op": ">=", "value": 0.0},
    "freeze":     {"metric": "freeze_longest_run_s", "op": "<=", "value": 1e9},
    "flicker":    {"metric": "flicker_dips",         "op": "<=", "value": 1e9},
    "ssim_floor": {"metric": "ssim_min",             "op": ">=", "value": 0.0},
    "sharpness":  {"metric": "laplacian_p10",        "op": ">=", "value": 0.0},
    "black":      {"metric": "black_frame_frac",     "op": "<=", "value": 1.0},
    "exposure":   {"metric": "clipped_frac",         "op": "<=", "value": 1.0},
}


def write_thresholds(path: Path, calibrated: bool = False, **value_overrides) -> Path:
    """Thresholds file with permissive defaults; override single values by check name."""
    l1_section = {}
    for name, spec in PERMISSIVE_L1.items():
        spec = dict(spec)
        if name in value_overrides:
            spec["value"] = value_overrides[name]
        l1_section[name] = spec
    data = {
        "version": "test.1",
        "calibrated": calibrated,
        "provenance": {},
        "l1": l1_section,
        "l2": {"floors": {
            "prompt_adherence": 3, "subject_consistency": 3, "anatomy_artifacts": 3,
            "temporal_coherence": 3, "imaging_quality": 3,
        }},
    }
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


@pytest.fixture()
def thresholds_permissive(tmp_path) -> Path:
    return write_thresholds(tmp_path / "thresholds.yaml")


@pytest.fixture()
def prompt_file(tmp_path) -> Path:
    p = tmp_path / "prompt.txt"
    p.write_text(
        "A silver mechanical hummingbird hovers over a neon-lit market stall, wings "
        "beating rapidly; slow dolly-in, cinematic lighting, vertical 9:16 composition.",
        encoding="utf-8",
    )
    return p
