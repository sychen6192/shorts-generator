"""L2 — VLM judge layer. M1 stub: full implementation lands in M2 (docs/plan.md).

The full checker calls run_l2() after L1 passes. Until M2 wires the judge adapters,
this raises InfraError so full mode fails closed (never PASS without a judge).
"""

from __future__ import annotations

from .errors import InfraError


def run_l2(clip_path, container, prompt_text, judge_cfg, floors):
    raise InfraError(
        "l2",
        "judge not configured — L2 lands in M2; run with --l1-only for the L1 gate",
    )
