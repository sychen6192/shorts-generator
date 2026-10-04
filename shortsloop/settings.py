"""Run settings shared by EVERY GPU path (nightly runner, calibrate-batch,
calibrate-tune --with-l2, doctor), so each one measures and judges exactly the
way the nightly does:

- load_policies(pipeline.yaml)  — policies merged over DEFAULT_POLICIES; a
  malformed file is an InfraError (refusal), never a traceback.
- effective_judge_cfg(cfg, pol) — config.yaml's judge section with pipeline.yaml's
  judge timeout_s / retries applied (pipeline.yaml is authoritative, plan §2.8).
- judge_unload_problem(judge)   — hard rule 3 needs a way to evict the VLM: an
  openai_compat judge without judge.unload_url cannot be unloaded, and a VRAM
  check alone cannot always tell (a resident 8B VLM may leave >24 GB free).
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from .errors import InfraError

DEFAULT_POLICIES = {
    "max_attempts_per_clip": 3,
    "wall_clock_budget_h": 6,
    "disk_min_free_gb": 20,
    "waves_max": 3,
    "comfy": {"timeout_s": 1800, "poll_s": 5, "one_job_at_a_time": True},
    "judge": {"timeout_s": 300, "retries": 1, "infra_escalation_after": 2},
    "vram_handoff": {"free_min_gb": 24, "wait_timeout_s": 180},
}


def load_policies(pipeline_path: str | Path | None) -> dict:
    pol = json.loads(json.dumps(DEFAULT_POLICIES))  # deep copy
    p = Path(pipeline_path) if pipeline_path else None
    if p is None or not p.is_file():
        return pol
    try:
        loaded = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as e:
        raise InfraError("l1", f"pipeline file {p} unreadable: {e}")
    if not isinstance(loaded, dict):
        raise InfraError("l1", f"pipeline file {p} is not a mapping")
    policies = loaded.get("policies")
    if policies is None:
        return pol
    if not isinstance(policies, dict):
        raise InfraError("l1", f"pipeline file {p}: `policies:` must be a mapping")
    for key, val in policies.items():
        if isinstance(pol.get(key), dict):
            if not isinstance(val, dict):
                raise InfraError("l1", f"pipeline file {p}: policies.{key} must be a "
                                       f"mapping, got {val!r}")
            pol[key].update(val)
        else:
            pol[key] = val
    problems = _policy_problems(pol)
    if problems:
        raise InfraError("l1", f"pipeline file {p}: " + "; ".join(problems))
    return pol


def _num(v) -> bool:
    return type(v) in (int, float)


def _policy_problems(pol: dict) -> list[str]:
    """Types and ranges the runner relies on — a mistyped value must refuse at
    22:00, not crash the night with a TypeError and no report."""
    errs = []

    def need(path, ok, what):
        cur = pol
        for k in path.split("."):
            cur = cur.get(k) if isinstance(cur, dict) else None
        if not ok(cur):
            errs.append(f"policies.{path} must be {what}, got {cur!r}")
    pos_int = (lambda v: type(v) is int and v >= 1, "an integer >= 1")
    pos_num = (lambda v: _num(v) and v > 0, "a number > 0")
    nonneg = (lambda v: _num(v) and v >= 0, "a number >= 0")
    for path, (ok, what) in {
        "max_attempts_per_clip": pos_int, "waves_max": pos_int,
        "wall_clock_budget_h": pos_num, "disk_min_free_gb": nonneg,
        "comfy.timeout_s": pos_num, "comfy.poll_s": pos_num,
        "judge.timeout_s": pos_num, "judge.infra_escalation_after": pos_int,
        "judge.retries": (lambda v: type(v) is int and v >= 0, "an integer >= 0"),
        "vram_handoff.free_min_gb": pos_num, "vram_handoff.wait_timeout_s": nonneg,
    }.items():
        need(path, ok, what)
    if "gen_free_min_gb" in (pol.get("vram_handoff") or {}):
        need("vram_handoff.gen_free_min_gb", *pos_num)
    return errs


def effective_judge_cfg(cfg: dict, pol: dict) -> dict:
    judge = dict(cfg.get("judge") or {})
    judge["timeout_s"] = pol["judge"]["timeout_s"]
    judge["retries"] = pol["judge"]["retries"]
    return judge


def judge_unload_problem(judge: dict) -> str | None:
    if judge.get("adapter", "ollama") != "ollama" and not judge.get("unload_url"):
        return (f"judge.adapter {judge.get('adapter')!r} has no judge.unload_url — the "
                f"VLM could not be evicted before a Wan wave (hard rule 3); point "
                f"unload_url at the server's unload endpoint (e.g. llama-swap /unload)")
    return None
