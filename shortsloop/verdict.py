"""Verdict assembly + invariant validation (schema v1, docs/plan.md §2.2 — FROZEN).

Vocabulary: PASS | FAIL | ERROR in full mode; PROCEED | FAIL | ERROR in --l1-only
mode. PROCEED exists so an L1-only result can never be mistaken for a shippable PASS
(hard rule 1); the runner's ship-gate additionally requires layers_run == ["l1","l2"].
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
from pathlib import Path

from . import SCHEMA_VERSION, __version__
from .policy import failure_classes

EXIT_BY_VERDICT = {"PASS": 0, "PROCEED": 0, "FAIL": 1, "ERROR": 2}


def sha256_file(path: str | Path) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return None


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _tool_versions() -> dict:
    import cv2
    import numpy
    try:
        out = subprocess.run(["ffprobe", "-version"], capture_output=True,
                             text=True, timeout=10).stdout.splitlines()
        ffprobe = out[0].split()[2] if out else None
    except Exception:
        ffprobe = None
    return {
        "checker": __version__,
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "numpy": numpy.__version__,
        "ffprobe": ffprobe,
    }


def assemble(
    *,
    clip_path: str,
    clip_sha256: str | None,
    container: dict | None,
    prompt_text: str | None,
    l1_block: dict | None,
    l2_block: dict | None,
    thresholds_info: dict,
    layers_run: list[str],
    l1_only: bool,
    error: dict | None,
    timing: dict,
) -> dict:
    """Build a schema-v1 verdict dict. Decision logic lives HERE, nowhere else."""
    if error is not None:
        verdict = "ERROR"
        classes: list[str] = []
    else:
        failed_l1 = [c["name"] for c in (l1_block or {}).get("checks", []) if not c["pass"]]
        failed_l2 = []
        if l2_block is not None:
            failed_l2 = [
                name for name, dim in l2_block["dimensions"].items()
                if not dim["na"] and not dim["pass"]
            ]
        if (l1_block or {}).get("pass") and (l2_block or {}).get("pass"):
            verdict = "PASS"
        elif l1_only and (l1_block or {}).get("pass"):
            verdict = "PROCEED"
        else:
            verdict = "FAIL"
        classes = failure_classes(failed_l1, failed_l2)

    return {
        "schema_version": SCHEMA_VERSION,
        "verdict": verdict,
        "failure_classes": classes,
        "error": error,
        "clip": {
            "path": clip_path,
            "sha256": clip_sha256,
            "container": container,
        },
        "prompt": {
            "text": prompt_text,
            "sha256": sha256_text(prompt_text) if prompt_text is not None else None,
        },
        "l1": l1_block,
        "l2": l2_block,
        "thresholds": thresholds_info,
        "layers_run": layers_run,
        "timing": timing,
        "versions": _tool_versions(),
    }


def validate(v: dict) -> list[str]:
    """Structural invariants of schema v1. Returns a list of violations (empty = ok)."""
    errs: list[str] = []
    verdict = v.get("verdict")
    if v.get("schema_version") != SCHEMA_VERSION:
        errs.append(f"schema_version != {SCHEMA_VERSION}")
    if verdict not in ("PASS", "FAIL", "ERROR", "PROCEED"):
        errs.append(f"bad verdict {verdict!r}")
    if verdict == "ERROR":
        err = v.get("error")
        if not isinstance(err, dict) or err.get("scope") not in ("clip", "infra"):
            errs.append("ERROR verdict requires error{scope in clip|infra}")
    else:
        if v.get("error") is not None:
            errs.append("non-ERROR verdict must have error == null")
    if verdict == "PASS":
        if v.get("layers_run") != ["l1", "l2"]:
            errs.append("PASS requires layers_run == ['l1','l2']")
        if not (v.get("l1") or {}).get("pass") or not (v.get("l2") or {}).get("pass"):
            errs.append("PASS requires l1.pass and l2.pass")
    if verdict == "PROCEED":
        if v.get("layers_run") != ["l1"]:
            errs.append("PROCEED is exclusive to --l1-only (layers_run == ['l1'])")
        if v.get("l2") is not None:
            errs.append("PROCEED must not carry an l2 block")
    if verdict == "FAIL" and not v.get("failure_classes"):
        errs.append("FAIL requires at least one failure class")
    if verdict in ("PASS", "PROCEED") and v.get("failure_classes"):
        errs.append(f"{verdict} must have empty failure_classes")
    if v.get("l2") is not None:
        dims = v["l2"].get("dimensions", {})
        expected = {"prompt_adherence", "subject_consistency", "anatomy_artifacts",
                    "temporal_coherence", "imaging_quality"}
        if set(dims) != expected:
            errs.append(f"l2.dimensions keys {sorted(dims)} != expected {sorted(expected)}")
    return errs
