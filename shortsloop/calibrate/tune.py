"""`shortsloop calibrate-tune` — tune thresholds against hand labels and report
agreement honestly (docs/plan.md §6 steps 3-4).

Discipline:
- Only USABLE labels count: the label's clip_sha256 must match the clip's current
  bytes, and L1 must be able to measure the clip. Stale labels and undecodable
  clips are excluded and listed in the report — never silently used.
- Stratified tune/test split (fixed seed); ALL selection happens on the tune set;
  the test set is touched once, for the report. Refused (no proposal written)
  when fewer than 10 usable clips exist or the test set lacks a pass or a fail.
- L1 detector classes (`static`: motion/freeze, `flicker`: flicker/ssim_floor) are
  swept JOINTLY under the runtime OR rule: the better-separating detector is the
  primary; the secondary is tuned only on class members the primary misses (a
  guard rail when there are none). Zero false-pass on tune is the target, but no
  threshold may false-fail more than FF_CAP_FRAC of labeled-pass tune clips, and
  no value outside the metric's feasible range is ever proposed. Candidates are
  rounded BEFORE evaluation: the reported separation is the written value's.
- Guard checks (sharpness/black/exposure) have no labeled class in the Phase 0
  vocabulary: guard rails beyond the worst labeled-PASS clip.
- L2 floors are tuned under the runtime rule (l2.run_l2): only on tune clips that
  pass the proposed L1 (L2 never sees the others), N/A dimensions excluded, a clip
  fails if ANY non-N/A dimension is below its floor; floors searched jointly with
  the same false-fail cap, ties broken toward the default 3.
- Judge scores are keyed by (clip sha, prompt sha, judge model, model digest):
  only the current judge's scores are used, and provenance records the judge.
- Held-out report (plan §6): agreement, false-pass = labeled-fail clips that
  would ship (Wilson CI, denominator explicit), false-fail, per-class confusion,
  per-layer catch attribution — for the combined L1+L2 decision when judge
  scores exist. Provenance carries those combined numbers.
- Writes thresholds.proposed.yaml (calibrated: false) LAST, after everything
  succeeded. Flipping calibrated: true happens ONLY via --approve after the human
  sign-off gate; --approve refuses stale inputs, edited values, a changed L1
  implementation (l1_impl), a judge other than the tuned one, and a proposal with
  NO judge scores behind it (untuned L2 floors = an uncalibrated instrument)
  unless the supervised --accept-untested-l2 exception is passed, which is
  recorded as provenance.l2_untested_accepted. It bumps the version above the
  target's.
- --with-l2 judges exactly as the nightly checker does (settings.
  effective_judge_cfg), refuses a judge it could not unload, and holds the
  runner's run lock while the VLM is on the GPU.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .. import l1
from ..errors import ClipError, InfraError
from ..label import resolve_clip_path
from ..policy import failure_classes
from ..state import _read_jsonl
from ..thresholds import FROZEN_L1, L2_DIMS, validate_thresholds
from ..verdict import sha256_file, sha256_text

CHECK_DEFS = FROZEN_L1   # name -> (metric, op) — §2.4 frozen shape
# Used only when a check has neither labeled examples nor labeled-pass clips.
FALLBACKS = {"motion": 0.0015, "freeze": 1.5, "flicker": 2, "ssim_floor": 0.35,
             "sharpness": 5.0, "black": 0.5, "exposure": 0.5}
FRACTION_METRICS = {"black_frame_frac", "clipped_frac"}
# What each metric can physically be — a threshold outside this range passes or
# fails EVERY clip (e.g. ssim_min >= 1.29).
FEASIBLE = {"flow_mag_median": (0.0, math.inf), "freeze_longest_run_s": (0.0, math.inf),
            "flicker_dips": (0.0, math.inf), "ssim_min": (-1.0, 1.0),
            "laplacian_p10": (0.0, math.inf), "black_frame_frac": (0.0, 1.0),
            "clipped_frac": (0.0, 1.0)}
DETECTORS = {"static": ["motion", "freeze"], "flicker": ["flicker", "ssim_floor"]}
GUARDS = ("sharpness", "black", "exposure")
DEFAULT_FLOOR = 3
FLOOR_CHOICES = (2, 3, 4, 5)
FF_CAP_FRAC = 0.2        # max share of labeled-pass tune clips one threshold may fail
MIN_USABLE = 10
SPLIT_SEED = 13
# Fingerprint of the metric definitions: cached metrics from other l1 code are stale.
L1_IMPL = (sha256_file(l1.__file__) or "unknown")[:16]
# Attribution buckets for clips the judge could NOT score: an ERROR never ships at
# runtime, but it is not the judge catching a defect (escalation decision, plan §6).
ATTR_L2_ERROR_L1_CATCHES = "L2 ERROR (L1 catches)"
ATTR_L2_ERROR_L1_MISSES = "L2 ERROR (L1 misses)"


class Refused(Exception):
    """Inputs cannot support an honest proposal — nothing is written."""


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _sig(x: float) -> float:
    """The value as WRITTEN (6 significant digits, never -0.0)."""
    return float(f"{x:.6g}") + 0.0


def _clamp(metric: str, t: float) -> float:
    lo, hi = FEASIBLE[metric]
    return min(max(t, lo), hi)


def _passes(value: float, op: str, threshold: float) -> bool:
    return value >= threshold if op == ">=" else value <= threshold


def _real(v) -> bool:
    return type(v) in (int, float) and math.isfinite(v)


# ------------------------------------------------------------------ inputs

def last_labels(cal_dir: Path) -> dict[str, dict]:
    """Latest label per clip wins (relabeling appends)."""
    out: dict[str, dict] = {}
    for rec in _read_jsonl(Path(cal_dir) / "labels.jsonl"):
        out[rec["clip_id"]] = rec
    return out


def _sha(path: str | None, cache: dict) -> str | None:
    if not path:
        return None
    if path not in cache:
        cache[path] = sha256_file(path)
    return cache[path]


def verify_labels(cal: Path, labels: dict[str, dict], manifest: list[dict],
                  shas: dict | None = None) -> tuple[dict[str, str], dict[str, str]]:
    """({clip_id: current clip sha} for labels proven to describe the clip's
    current bytes, {clip_id: why not} for the rest)."""
    shas = {} if shas is None else shas
    by_id = {r["clip_id"]: r for r in manifest}
    ok: dict[str, str] = {}
    stale: dict[str, str] = {}
    for cid, rec in labels.items():
        row = by_id.get(cid)
        if row is None:
            stale[cid] = "clip not in batch_manifest.jsonl"
            continue
        path = resolve_clip_path(cal, row.get("clip_path"))
        cur = _sha(path, shas)
        if cur is None:
            stale[cid] = f"clip file missing ({path})"
        elif not rec.get("clip_sha256"):
            stale[cid] = "label has no clip_sha256 (cannot prove which pixels it judged)"
        elif rec["clip_sha256"] != cur:
            stale[cid] = "clip bytes changed since it was labeled (sha mismatch)"
        else:
            ok[cid] = cur
    return ok, stale


def load_metrics(cal_dir, manifest: list[dict], errors: dict | None = None,
                 only: set | None = None, shas: dict | None = None) -> dict[str, dict]:
    """L1 metrics per clip_id, cached in metrics.jsonl keyed by (clip sha256, L1
    implementation fingerprint) — regenerated pixels or changed metric code are
    recomputed, never reused. Swap rows share their source's bytes, so a file is
    decoded once. A clip L1 cannot decode gets an error row (reruns skip it) and
    lands in `errors` instead of aborting calibration."""
    cal = Path(cal_dir)
    shas = {} if shas is None else shas
    cache_path = cal / "metrics.jsonl"
    cache = {r["clip_sha256"]: r for r in _read_jsonl(cache_path)
             if r.get("clip_sha256") and r.get("l1_impl") == L1_IMPL}
    out: dict[str, dict] = {}
    new_lines = []
    for row in manifest:
        cid = row["clip_id"]
        if only is not None and cid not in only:
            continue
        path = resolve_clip_path(cal, row.get("clip_path"))
        sha = _sha(path, shas)
        if sha is None:
            continue
        hit = cache.get(sha)
        if hit is None:
            hit = {"clip_id": cid, "clip_sha256": sha, "l1_impl": L1_IMPL}
            try:
                hit["metrics"] = l1.compute_metrics(path)["metrics"]
            except ClipError as e:
                hit["error"] = f"{e.stage}: {e.message}"
                print(f"[calibrate-tune] {cid}: L1 cannot measure this clip "
                      f"({hit['error']}) — excluded, recorded in metrics.jsonl")
            cache[sha] = hit
            new_lines.append(hit)
        if hit.get("error"):
            if errors is not None:
                errors[cid] = hit["error"]
        else:
            out[cid] = hit["metrics"]
    if new_lines:
        with open(cache_path, "a", encoding="utf-8") as f:
            for line in new_lines:
                f.write(json.dumps(line) + "\n")
    return out


def stratified_split(labels: dict[str, dict], ratio: float = 0.7,
                     seed: int = SPLIT_SEED) -> tuple[list[str], list[str]]:
    import random
    groups: dict[tuple, list[str]] = {}
    for cid, rec in sorted(labels.items()):
        key = (rec["verdict"], (rec.get("classes") or [""])[0] if rec.get("classes") else "")
        groups.setdefault(key, []).append(cid)
    tune, test = [], []
    rng = random.Random(seed)
    for key in sorted(groups):
        ids = groups[key]
        rng.shuffle(ids)
        cut = max(1, round(len(ids) * ratio)) if len(ids) > 1 else 1
        tune.extend(ids[:cut])
        test.extend(ids[cut:])
    return sorted(tune), sorted(test)


def _l2_key(r: dict):
    keys = ("clip_id", "clip_sha256", "prompt_sha256", "model", "model_digest")
    return tuple(r.get(k) for k in keys) if all(r.get(k) for k in keys) else None


def select_judge_rows(rows: list[dict], judge: dict | None,
                      expected: dict[str, tuple[str, str]]):
    """The current judge's scores for the expected (clip sha, prompt sha) pairs.
    judge = {"model", "model_digest"|None} or None (= the most recently written
    judge). Returns (identity | None, {clip_id: row} | None, other judges)."""
    valid = [r for r in rows if _l2_key(r) and r["clip_id"] in expected
             and (r["clip_sha256"], r["prompt_sha256"]) == expected[r["clip_id"]]]
    want_model = (judge or {}).get("model")
    want_digest = (judge or {}).get("model_digest")
    cands = [r for r in valid
             if (not want_model or r["model"] == want_model)
             and (not want_digest or r["model_digest"] == want_digest)]
    judges = sorted({(r["model"], r["model_digest"]) for r in valid})
    if not cands:
        return None, None, judges
    ident = (cands[-1]["model"], cands[-1]["model_digest"])
    chosen = {r["clip_id"]: r for r in cands if (r["model"], r["model_digest"]) == ident}
    return ({"model": ident[0], "model_digest": ident[1]}, chosen,
            [j for j in judges if j != ident])


# ---------------------------------------------------------------- L1 sweeps

def _ff_cap(n_good: int) -> int:
    return int(FF_CAP_FRAC * n_good)


def _beyond(metric: str, op: str, values: list[float]) -> float:
    """A feasible threshold every value in `values` passes (with headroom)."""
    if op == ">=":
        lo = min(values)
        t = lo * 0.5 if lo > 0 else lo - 0.5 * abs(lo)
    elif metric in FRACTION_METRICS:
        t = max(values) * 1.5 + 0.02
    else:   # counts / seconds
        t = max(values) * 1.5 + (1 if metric == "flicker_dips" else 0.5)
    return _sig(_clamp(metric, t))


def guard_value(check: str, good: list[float], fallback: float) -> float:
    """Guard rail beyond the worst labeled-PASS clip (never false-fails tune)."""
    metric, op = CHECK_DEFS[check]
    return _beyond(metric, op, good) if good else fallback


def sweep_detector(check: str, bad: list[float], good: list[float],
                   ff_cap: int | None = None) -> dict:
    """1-D sweep: fewest missed `bad` values, then fewest false-fails, among
    rounded, feasible candidates failing at most `ff_cap` `good` values; ties go
    to the more permissive threshold. Never a fail-everything candidate."""
    metric, op = CHECK_DEFS[check]
    if not bad:
        return {"value": None, "note": "untuned — no labeled examples"}
    cap = _ff_cap(len(good)) if ff_cap is None else ff_cap
    values = sorted(set(bad) | set(good))
    cands = {_beyond(metric, op, values)}
    cands |= {_sig(_clamp(metric, (a + b) / 2)) for a, b in zip(values, values[1:])}
    best = None
    capped = None      # fewest false-fails a zero-miss candidate would have cost
    for t in sorted(cands):
        missed = sum(1 for v in bad if _passes(v, op, t))
        ff = sum(1 for v in good if not _passes(v, op, t))
        if ff > cap:
            if missed == 0 and (capped is None or ff < capped):
                capped = ff
            continue
        key = (missed, ff, t if op == ">=" else -t)
        if best is None or key < best[0]:
            best = (key, t)
    (missed, ff, _), t = best
    if (missed, ff) == (0, 0):
        note = "clean separation"
    elif missed and capped is not None:
        note = (f"OVERLAP — catching all {len(bad)} would false-fail {capped}/"
                f"{len(good)} labeled-pass clips (cap {cap})")
    else:
        note = "OVERLAP — could not fully separate on tune set"
    return {"value": t, "missed_on_tune": missed, "false_fail_on_tune": ff,
            "n_bad": len(bad), "n_good": len(good), "ff_cap": cap, "note": note}


def tune_l1(tune_ids: list[str], labels: dict, metrics: dict) -> tuple[dict, dict]:
    good_ids = [c for c in tune_ids if labels[c]["verdict"] == "pass"]
    total_cap = _ff_cap(len(good_ids))
    section: dict[str, dict] = {}
    notes: dict[str, dict] = {}

    def vals(ids, check):
        metric = CHECK_DEFS[check][0]
        return [metrics[c][metric] for c in ids]

    def put(check, value, note):
        metric, op = CHECK_DEFS[check]
        section[check] = {"metric": metric, "op": op, "value": value}
        notes[check] = {**note, "value": value}

    for cls, checks in DETECTORS.items():
        members = [c for c in tune_ids if labels[c]["verdict"] == "fail"
                   and cls in (labels[c].get("classes") or [])]
        if not members:
            # A null value would crash every checker run after sign-off: guard
            # rail that never fails a labeled-pass clip — and say so loudly.
            for check in checks:
                put(check, guard_value(check, vals(good_ids, check), FALLBACKS[check]),
                    {"class": cls, "role": "untuned",
                     "note": f"UNTUNED — no labeled {cls} examples; guard rail from "
                             f"labeled-pass margins only (label some {cls} clips and "
                             f"re-tune to calibrate this check)"})
            continue
        solo = {ch: sweep_detector(ch, vals(members, ch), vals(good_ids, ch), total_cap)
                for ch in checks}
        primary = min(checks, key=lambda ch: (solo[ch]["missed_on_tune"],
                                              solo[ch]["false_fail_on_tune"],
                                              checks.index(ch)))
        res = solo[primary]
        put(primary, res["value"], {**res, "class": cls, "role": "primary"})
        metric, op = CHECK_DEFS[primary]
        residual = [c for c in members if _passes(metrics[c][metric], op, res["value"])]
        rest_good = [c for c in good_ids if _passes(metrics[c][metric], op, res["value"])]
        for sec in checks:
            if sec == primary:
                continue
            if not residual:
                put(sec, guard_value(sec, vals(good_ids, sec), FALLBACKS[sec]),
                    {"class": cls, "role": "secondary",
                     "note": f"guard rail — {primary} already catches every tune "
                             f"{cls} clip; set beyond labeled-pass margins"})
                continue
            r2 = sweep_detector(sec, vals(residual, sec), vals(rest_good, sec),
                                max(0, total_cap - res["false_fail_on_tune"]))
            put(sec, r2["value"],
                {**r2, "class": cls, "role": "secondary",
                 "note": f"{r2['note']} — tuned only on the {len(residual)} tune "
                         f"{cls} clip(s) {primary} lets through"})
    for check in GUARDS:
        put(check, guard_value(check, vals(good_ids, check), FALLBACKS[check]),
            {"class": "(guard)", "role": "guard",
             "note": "guard rail from labeled-pass margins"})
    order = list(CHECK_DEFS)
    return ({k: section[k] for k in order}, {k: notes[k] for k in order})


def l1_failed(m: dict, l1_section: dict) -> list[str]:
    return [name for name, spec in l1_section.items()
            if not _passes(m[spec["metric"]], spec["op"], spec["value"])]


# ---------------------------------------------------------------- L2 floors

def l2_verdict(row: dict, floors: dict) -> tuple[bool, list[str]]:
    """The runtime L2 rule (l2.run_l2): N/A dimensions are excluded; the clip
    passes iff every non-N/A dimension scores >= its floor (and at least one
    dimension is not N/A). Returns (passed, failed dimensions)."""
    na = row.get("na") or {}
    non_na = [d for d in L2_DIMS if not na.get(d)]
    failed = [d for d in non_na if row["dimensions"][d] < floors[d]]
    return (not failed and bool(non_na)), failed


def tune_l2(pop_ids: list[str], labels: dict, l2_rows: dict) -> tuple[dict, dict]:
    """Joint floor search over the clips L2 actually sees (tune clips passing the
    proposed L1). Judge-error rows are runtime ERRORs whatever the floors are —
    they do not inform floors (reported in the evaluation instead).

    A labeled-pass clip the judge fails under EVERY combination (a non-N/A score
    below the loosest floor, or every dimension N/A) is a false-fail no floor can
    avoid. When there are more of those than the cap allows, nothing fits the cap:
    the search then falls back EXPLICITLY to the least-bad floors — no false-fail
    beyond the forced ones, fewest misses among those — and says so in
    notes["fallback"] (-> report, stdout, provenance.l2_floors_fallback)."""
    scored = [c for c in pop_ids if not l2_rows[c].get("error")]
    bad = [c for c in scored if labels[c]["verdict"] == "fail"]
    good = [c for c in scored if labels[c]["verdict"] == "pass"]
    cap = _ff_cap(len(good))
    lowest = min(FLOOR_CHOICES)
    loosest = {d: lowest for d in L2_DIMS}
    forced = [c for c in good if not l2_verdict(l2_rows[c], loosest)[0]]
    limit = max(cap, len(forced))       # the loosest combination always fits this
    fallback = None
    if len(forced) > cap:
        def why(row: dict) -> str:
            na = row.get("na") or {}
            low = [f"{d}={row['dimensions'][d]}" for d in L2_DIMS
                   if not na.get(d) and row["dimensions"][d] < lowest]
            return ", ".join(low) or "every dimension N/A"
        fallback = (f"The judge fails {len(forced)}/{len(good)} labeled-pass tune clips "
                    f"under EVERY floor combination ("
                    + "; ".join(f"{c}: {why(l2_rows[c])}" for c in forced)
                    + f") — more than the false-fail cap of {cap}, so no combination "
                    f"fits it. Floors are the least-bad combination: no false-fail "
                    f"beyond those {len(forced)}, fewest misses among them. The judge "
                    f"disagrees with your labels — re-check those clips, or escalate to "
                    f"a ~32B judge (runbook §2c)")
    best = None
    capped = None
    for combo in itertools.product(FLOOR_CHOICES, repeat=len(L2_DIMS)):
        floors = dict(zip(L2_DIMS, combo))
        missed = sum(1 for c in bad if l2_verdict(l2_rows[c], floors)[0])
        ff = sum(1 for c in good if not l2_verdict(l2_rows[c], floors)[0])
        if ff > limit:
            if capped is None or (missed, ff) < capped:
                capped = (missed, ff)
            continue
        key = (missed, ff, sum(abs(f - DEFAULT_FLOOR) for f in combo), combo)
        if best is None or key < best[0]:
            best = (key, floors)
    if best is None:     # unreachable (the loosest combination fits `limit`)
        raise Refused("internal: no L2 floor combination could be evaluated")
    (missed, ff, _, _), floors = best
    per_dim = {}
    for d in L2_DIMS:
        per_dim[d] = {
            "floor": floors[d],
            "fails_bad": sum(1 for c in bad if d in l2_verdict(l2_rows[c], floors)[1]),
            "fails_good": sum(1 for c in good if d in l2_verdict(l2_rows[c], floors)[1]),
            "na": sum(1 for c in scored if (l2_rows[c].get("na") or {}).get(d)),
        }
    note = None
    if capped is not None and capped[0] < missed:
        note = (f"catching {missed - capped[0]} more would false-fail {capped[1]}/"
                f"{len(good)} labeled-pass clips (cap {cap}"
                + (f"; fallback limit {limit}" if fallback else "") + ")")
    return floors, {"population": len(pop_ids), "n_bad": len(bad), "n_good": len(good),
                    "judge_errors": len(pop_ids) - len(scored), "missed_on_tune": missed,
                    "false_fail_on_tune": ff, "ff_cap": cap, "per_dim": per_dim,
                    "forced_false_fail_ids": forced, "fallback": fallback,
                    "note": note}


# --------------------------------------------------------------- evaluation

def evaluate(ids: list[str], labels: dict, metrics: dict, l1_section: dict,
             l2_rows: dict | None = None, floors: dict | None = None) -> dict:
    """The runtime decision on labeled clips: L1, then (when judge scores exist)
    L2 on L1-passing clips; a judge ERROR never ships. Positive = FAIL.
    false-pass = labeled-fail clips that would ship, over labeled-fail clips."""
    tp = fp = tn = fn = 0
    fn_ids, fp_ids = [], []
    confusion: dict[str, dict[str, int]] = {}
    stopped = {"L1": 0, "L2": 0, "L2 error": 0, "missed": 0}
    ff_by = {"L1": 0, "L2": 0, "L2 error": 0}
    by_class: dict[str, dict[str, int]] = {}
    clips = {}
    for cid in ids:
        if cid not in metrics:
            continue
        l1_fail = l1_failed(metrics[cid], l1_section)
        l2_fail, l2_err, l2_catch = None, None, False
        if l2_rows is not None:
            row = l2_rows[cid]
            if row.get("error"):
                # a runtime ERROR (never ships) — but NOT the judge catching it
                l2_err, l2_fail = row["error"], []
            else:
                ok, l2_fail = l2_verdict(row, floors)
                l2_catch = not ok
        if l1_fail:
            layer = "L1"
            pred = (failure_classes(l1_fail, []) or ["fail"])[0]
        elif l2_err:
            layer, pred = "L2 error", "ERROR"
        elif l2_catch:
            layer = "L2"
            pred = (failure_classes([], l2_fail) or ["fail (all N/A)"])[0]
        else:
            layer, pred = None, "PASS"
        rec = labels[cid]
        actual_fail = rec["verdict"] == "fail"
        truth = (rec.get("classes") or ["fail"])[0] if actual_fail else "pass"
        row_c = confusion.setdefault(truth, {})
        row_c[pred] = row_c.get(pred, 0) + 1
        if actual_fail:
            if layer:
                tp += 1
                stopped[layer] += 1
            else:
                fn += 1
                fn_ids.append(cid)
                stopped["missed"] += 1
            if l2_rows is not None:
                if l2_err:
                    k = ATTR_L2_ERROR_L1_CATCHES if l1_fail else ATTR_L2_ERROR_L1_MISSES
                else:
                    k = ("both" if l1_fail and l2_catch else "L1 only" if l1_fail
                         else "L2 only" if l2_catch else "neither")
                cell = by_class.setdefault(truth, {})
                cell[k] = cell.get(k, 0) + 1
        elif layer:
            fp += 1
            fp_ids.append(cid)
            ff_by[layer] += 1
        else:
            tn += 1
        clips[cid] = {"label": truth, "predicted": pred, "stopped_by": layer,
                      "l1_failed": l1_fail, "l2_failed": l2_fail, "l2_error": l2_err}
    n, n_bad, n_good = tp + fp + tn + fn, tp + fn, fp + tn
    return {"scope": "L1+L2" if l2_rows is not None else "L1 only",
            "n": n, "agreement": (tp + tn) / n if n else None,
            "n_bad": n_bad, "false_pass": fn,
            "false_pass_rate": fn / n_bad if n_bad else None,
            "false_pass_ci": wilson(fn, n_bad),
            "n_good": n_good, "false_fail": fp,
            "false_fail_rate": fp / n_good if n_good else None,
            "false_fail_ci": wilson(fp, n_good),
            "false_pass_ids": fn_ids, "false_fail_ids": fp_ids,
            "confusion": confusion,
            "attribution": {"stopped_by": stopped, "false_fail_by": ff_by,
                            "by_class": by_class},
            "clips": clips}


def evaluate_l1(ids: list[str], labels: dict, metrics: dict, l1_section: dict) -> dict:
    """L1-only decision (what the report compares the pipeline against)."""
    return evaluate(ids, labels, metrics, l1_section)


def _rate_text(k: int, n: int, what: str) -> str:
    lo, hi = wilson(k, n)
    rate = f"{k / n:.0%}" if n else "n/a"
    return f"{k}/{n} {what} ({rate}, 95% CI {lo:.2f}–{hi:.2f})"


def _values_sha(values: dict) -> str:
    return sha256_text(json.dumps(values, sort_keys=True))


def next_version(target: str | Path, now: datetime | None = None) -> str:
    """YYYY-MM-DD.N, strictly above the target file's current version."""
    today = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%d")
    cur = None
    try:
        cur = (yaml.safe_load(Path(target).read_text(encoding="utf-8")) or {}).get("version")
    except (OSError, yaml.YAMLError, AttributeError):
        pass
    m = re.fullmatch(r"(\d{4}-\d{2}-\d{2})\.(\d+)", str(cur or "").strip())
    if m and m.group(1) >= today:
        return f"{m.group(1)}.{int(m.group(2)) + 1}"
    return f"{today}.1"


def _configured_judge(config_path: str | None) -> str | None:
    """`adapter/model` of config.yaml's judge (the l2 block's `model` format)."""
    if not config_path or not Path(config_path).is_file():
        return None
    try:
        cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return None
    judge = cfg.get("judge") if isinstance(cfg, dict) else None
    if isinstance(judge, dict) and judge.get("model"):
        return f"{judge.get('adapter', 'ollama')}/{judge['model']}"
    return None


# ------------------------------------------------------------------- tuning

def _tune(cal: Path, target: str, judge: dict | None) -> dict:
    manifest = _read_jsonl(cal / "batch_manifest.jsonl")
    if not manifest:
        raise Refused(f"no manifest in {cal} — run calibrate-batch first")
    labels_all = last_labels(cal)
    shas: dict = {}
    verified, stale = verify_labels(cal, labels_all, manifest, shas)
    metric_errors: dict[str, str] = {}
    metrics = load_metrics(cal, manifest, errors=metric_errors, only=set(verified),
                           shas=shas)
    usable = sorted(c for c in verified if c in metrics)
    labels = {c: labels_all[c] for c in usable}
    verdicts = {rec["verdict"] for rec in labels.values()}
    if len(usable) < MIN_USABLE:
        raise Refused(
            f"only {len(usable)} usable labeled clips (label matches the clip's "
            f"current bytes and L1 can measure it) — need >= {MIN_USABLE}. "
            f"{len(labels_all)} labels: {len(stale)} stale/unverifiable, "
            f"{len(metric_errors)} on undecodable clips. Label the batch first "
            f"(shortsloop label --calibration {cal}).")
    if verdicts != {"pass", "fail"}:
        raise Refused(f"every usable label is {'/'.join(sorted(verdicts))} — tuning "
                      f"needs labeled-pass AND labeled-fail clips")
    tune_ids, test_ids = stratified_split(labels)
    missing = {"pass", "fail"} - {labels[c]["verdict"] for c in test_ids}
    if missing:
        raise Refused(f"the held-out test split ({len(tune_ids)}/{len(test_ids)}) has "
                      f"no labeled-{'/'.join(sorted(missing))} clip — a stratum "
                      f"needs >= 2 clips before one can be held out; label more")

    by_id = {r["clip_id"]: r for r in manifest}
    expected = {c: (verified[c], sha256_text(by_id[c]["prompt"])) for c in usable}
    l2_file = _read_jsonl(cal / "l2_scores.jsonl")
    identity, l2_rows, other_judges = select_judge_rows(l2_file, judge, expected)
    if l2_rows is not None:
        uncovered = [c for c in usable if c not in l2_rows]
        if uncovered:
            raise Refused(
                f"judge scores from {identity['model']} ({identity['model_digest']}) "
                f"cover {len(usable) - len(uncovered)}/{len(usable)} usable clips — "
                f"finish `calibrate-tune --with-l2` first (missing: "
                f"{', '.join(uncovered[:8])}{' …' if len(uncovered) > 8 else ''})")

    l1_section, sweep_notes = tune_l1(tune_ids, labels, metrics)
    floors = {d: DEFAULT_FLOOR for d in L2_DIMS}
    l2_notes = None
    if l2_rows is not None:
        population = [c for c in tune_ids if not l1_failed(metrics[c], l1_section)]
        floors, l2_notes = tune_l2(population, labels, l2_rows)
    for name, spec in l1_section.items():
        lo, hi = FEASIBLE[spec["metric"]]
        if not (_real(spec["value"]) and lo <= spec["value"] <= hi):
            raise Refused(f"internal: l1.{name} = {spec['value']!r} is outside "
                          f"{spec['metric']}'s feasible range [{lo}, {hi}]")

    ev = {s: evaluate(ids, labels, metrics, l1_section, l2_rows, floors)
          for s, ids in (("tune", tune_ids), ("test", test_ids))}
    ev_l1 = {s: evaluate_l1(ids, labels, metrics, l1_section)
             for s, ids in (("tune", tune_ids), ("test", test_ids))}
    test = ev["test"]
    values = {"l1": l1_section, "l2": {"floors": floors}}
    scope = (f"L1+L2 pipeline (judge {identity['model']})" if identity else
             "L1 only — no judge scores; L2 floors are untuned defaults")
    proposed = {
        "version": next_version(target),
        "calibrated": False,   # flips ONLY via --approve (human sign-off gate)
        "provenance": {
            "labels_sha256": sha256_file(cal / "labels.jsonl"),
            "tuned_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tune_test_split": f"{len(tune_ids)}/{len(test_ids)} stratified "
                               f"seed={SPLIT_SEED}",
            "test_agreement": round(test["agreement"], 4),
            "test_false_pass": _rate_text(test["false_pass"], test["n_bad"],
                                          "labeled-fail test clips would ship"),
            "test_false_fail": _rate_text(test["false_fail"], test["n_good"],
                                          "labeled-pass test clips failed"),
            "test_scope": scope,
            "judge": identity,
            # null unless no L2 floor combination fit the false-fail cap (tune_l2)
            "l2_floors_fallback": (l2_notes or {}).get("fallback"),
            "l1_impl": L1_IMPL,
            "inputs_sha256": {
                "batch_manifest": sha256_file(cal / "batch_manifest.jsonl"),
                "l2_scores": sha256_file(cal / "l2_scores.jsonl"),
            },
            "values_sha256": _values_sha(values),
        },
        **values,
    }
    problems = validate_thresholds({**proposed, "calibrated": True})
    if problems:
        raise Refused("internal: proposal is off the frozen thresholds shape: "
                      + "; ".join(problems))
    return {"proposed": proposed, "labels_all": labels_all, "labels": labels,
            "usable": usable, "stale": stale, "metric_errors": metric_errors,
            "tune_ids": tune_ids, "test_ids": test_ids, "sweep_notes": sweep_notes,
            "l2_notes": l2_notes, "identity": identity, "other_judges": other_judges,
            "l2_rows": l2_rows, "ev": ev, "ev_l1": ev_l1}


def run_tune(cal_dir: str, approve: bool = False, target: str = "thresholds.yaml",
             base_thresholds: str | None = None, judge: dict | None = None,
             config_path: str | None = None, accept_untested_l2: bool = False) -> int:
    cal = Path(cal_dir)
    if approve:
        return _approve(cal, target, config_path, accept_untested_l2)
    proposed_path = cal / "thresholds.proposed.yaml"
    try:
        res = _tune(cal, target, judge)
    except Refused as e:
        print(f"[calibrate-tune] REFUSING to propose thresholds: {e}")
        if proposed_path.exists():
            proposed_path.unlink()
            print(f"[calibrate-tune] removed the older {proposed_path.name} — it does "
                  f"not describe these inputs")
        return 2
    _write_report(cal, res, proposed_path)
    # LAST, atomically: an approvable proposal exists only if everything succeeded
    tmp = proposed_path.with_name(proposed_path.name + ".tmp")
    tmp.write_text(yaml.safe_dump(res["proposed"], sort_keys=False, allow_unicode=True),
                   encoding="utf-8")
    os.replace(tmp, proposed_path)
    test = res["ev"]["test"]
    print(f"[calibrate-tune] proposed thresholds: {proposed_path}")
    print(f"[calibrate-tune] report: {cal / 'tuning_report.md'}")
    fallback = (res["l2_notes"] or {}).get("fallback")
    if fallback:
        print(f"[calibrate-tune] WARNING: L2 floors are a FALLBACK, not a fit — "
              f"{fallback}")
    print(f"[calibrate-tune] TEST ({test['scope']}): agreement {test['agreement']:.0%}, "
          f"false-pass {test['false_pass']}/{test['n_bad']} labeled-fail clips, "
          f"false-fail {test['false_fail']}/{test['n_good']} labeled-pass clips — "
          f"review, then sign off with: shortsloop calibrate-tune --calibration "
          f"{cal} --approve")
    return 0


def _approve(cal: Path, target: str, config_path: str | None = None,
             accept_untested_l2: bool = False) -> int:
    def refuse(msg: str) -> int:
        print(f"[calibrate-tune] REFUSING approve: {msg}")
        return 2

    proposed_path = cal / "thresholds.proposed.yaml"
    if not proposed_path.is_file():
        print(f"[calibrate-tune] nothing to approve — {proposed_path} missing")
        return 2
    try:
        data = yaml.safe_load(proposed_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        return refuse(f"{proposed_path} unparseable: {e}")
    if not isinstance(data, dict) or not isinstance(data.get("provenance"), dict):
        return refuse(f"{proposed_path} is not a proposal")
    prov = data["provenance"]
    current_sha = sha256_file(cal / "labels.jsonl")
    if current_sha is None:
        return refuse(f"{cal / 'labels.jsonl'} is missing — nothing proves what these "
                      f"thresholds were tuned on")
    if prov.get("labels_sha256") != current_sha:
        return refuse("labels.jsonl changed since tuning — re-run calibrate-tune first")
    inputs = prov.get("inputs_sha256")
    if not isinstance(inputs, dict):
        return refuse("proposal predates input tracking — re-run calibrate-tune")
    for key, fname in (("batch_manifest", "batch_manifest.jsonl"),
                       ("l2_scores", "l2_scores.jsonl")):
        if inputs.get(key) != sha256_file(cal / fname):
            return refuse(f"{fname} changed since tuning — re-run calibrate-tune first")
    if prov.get("values_sha256") != _values_sha({"l1": data.get("l1"),
                                                 "l2": data.get("l2")}):
        return refuse("threshold values were edited after tuning — the test numbers "
                      "in provenance describe other values; re-run calibrate-tune")
    if not _real(prov.get("test_agreement")):
        return refuse("no held-out test evaluation in provenance")
    if prov.get("l1_impl") != L1_IMPL:
        return refuse(f"L1 metric code (shortsloop/l1.py) changed since tuning "
                      f"(proposal l1_impl {prov.get('l1_impl')!r}, current {L1_IMPL!r}) — "
                      f"the thresholds and test numbers describe other metric "
                      f"definitions; re-run calibrate-tune")
    tuned_judge = (prov.get("judge") or {}).get("model") \
        if isinstance(prov.get("judge"), dict) else None
    cfg_judge = _configured_judge(config_path)
    if tuned_judge and cfg_judge and tuned_judge != cfg_judge:
        return refuse(f"L2 floors were tuned on judge {tuned_judge}, but {config_path} "
                      f"configures {cfg_judge} — re-run calibrate-tune --with-l2 "
                      f"with the judge you will run nightly")
    if tuned_judge and not cfg_judge:
        print(f"[calibrate-tune] note: could not read the judge from {config_path}; "
              f"these floors are calibrated for {tuned_judge} only")
    if not tuned_judge:
        if not accept_untested_l2:
            return refuse(
                "no judge scores behind this proposal (provenance.judge is null"
                + (f"; {config_path} configures {cfg_judge}" if cfg_judge else "")
                + ") — its L2 floors are untuned defaults and its test numbers are "
                "L1-only. An uncalibrated judge is an uncalibrated instrument: run "
                "`calibrate-tune --with-l2` with the judge you will run nightly. "
                "Supervised exception only: --approve --accept-untested-l2 (recorded "
                "in provenance as l2_untested_accepted: true)")
        print("[calibrate-tune] WARNING: --accept-untested-l2 — signing off with NO "
              "judge scores behind the L2 floors (untuned defaults; test numbers are "
              "L1-only). Recorded as provenance.l2_untested_accepted: true; "
              "test_scope stays L1-only.")
        prov["l2_untested_accepted"] = True
    if prov.get("l2_floors_fallback"):
        print(f"[calibrate-tune] WARNING: signing off L2 floors that are a FALLBACK — "
              f"{prov['l2_floors_fallback']} (kept in provenance.l2_floors_fallback)")
    data["version"] = next_version(target)
    data["calibrated"] = True
    prov["approved_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    problems = validate_thresholds(data)
    if problems:
        return refuse("proposal is off the frozen thresholds shape (plan §2.4): "
                      + "; ".join(problems))
    out = Path(target)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
                   encoding="utf-8")
    os.replace(tmp, out)
    print(f"[calibrate-tune] APPROVED → {target} (version {data['version']}, "
          f"calibrated: true). The unattended gate is now open.")
    return 0


# ------------------------------------------------------------------- report

def _pct(x) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def _perf_lines(name: str, ev: dict, labels: dict) -> list[str]:
    if not ev["n"]:
        return [f"- **{name}**: no clips"]
    lo, hi = ev["false_pass_ci"]
    flo, fhi = ev["false_fail_ci"]
    out = [f"- **{name}** ({ev['scope']}, n={ev['n']}): agreement "
           f"{_pct(ev['agreement'])}",
           f"  - **false-pass {ev['false_pass']}/{ev['n_bad']}** labeled-fail clips "
           f"would ship (rate {_pct(ev['false_pass_rate'])}, 95% CI "
           f"{lo:.0%}–{hi:.0%})",
           f"  - false-fail {ev['false_fail']}/{ev['n_good']} labeled-pass clips "
           f"failed (rate {_pct(ev['false_fail_rate'])}, 95% CI {flo:.0%}–{fhi:.0%})"]
    if ev["false_pass_ids"]:
        left = [c for c in ev["false_pass_ids"]
                if set(labels[c].get("classes") or []) & {"deformed", "off_prompt", "other"}]
        hard = [c for c in ev["false_pass_ids"] if c not in left]
        if left:
            out.append(f"  - passed {'L1' if ev['scope'] == 'L1 only' else 'L1+L2'} "
                       f"but labeled fail"
                       + (" (L2's job — deformed/off_prompt/other)"
                          if ev["scope"] == "L1 only" else "")
                       + f": {', '.join(left)}")
        if hard:
            out.append(f"  - ⚠️ L1-class clips that would ship: {', '.join(hard)}")
    if ev["false_fail_ids"]:
        out.append(f"  - labeled pass but failed: {', '.join(ev['false_fail_ids'])}")
    return out


def _confusion_md(ev: dict) -> list[str]:
    cols = sorted({p for row in ev["confusion"].values() for p in row},
                  key=lambda p: (p != "PASS", p))
    md = ["| labeled ↓ / predicted → | " + " | ".join(cols) + " |",
          "|---|" + "---|" * len(cols)]
    for truth in sorted(ev["confusion"], key=lambda t: (t != "pass", t)):
        row = ev["confusion"][truth]
        md.append(f"| {truth} | " + " | ".join(str(row.get(c, 0)) for c in cols) + " |")
    return md


def _attribution_md(ev: dict) -> list[str]:
    att = ev["attribution"]
    s = att["stopped_by"]
    md = [f"- labeled-fail clips stopped by L1: {s['L1']}, by L2: {s['L2']}, by a "
          f"judge ERROR: {s['L2 error']}, **shipped (missed): {s['missed']}**",
          f"- labeled-pass clips failed by L1: {att['false_fail_by']['L1']}, by L2: "
          f"{att['false_fail_by']['L2']}, by a judge ERROR: "
          f"{att['false_fail_by']['L2 error']}"]
    if att["by_class"]:
        kinds = ("L1 only", "L2 only", "both", "neither",
                 ATTR_L2_ERROR_L1_CATCHES, ATTR_L2_ERROR_L1_MISSES)
        md += ["", "Would each layer catch it on its own (L2 scored on every clip; a "
                   "judge ERROR is counted apart — the judge did not catch that clip, "
                   "it could not evaluate it):", "",
               "| labeled class | " + " | ".join(kinds) + " |",
               "|---|" + "---|" * len(kinds)]
        for cls in sorted(att["by_class"]):
            row = att["by_class"][cls]
            md.append(f"| {cls} | " + " | ".join(str(row.get(k, 0)) for k in kinds) + " |")
    return md


def _write_report(cal: Path, res: dict, proposed_path: Path) -> None:
    labels_all, labels = res["labels_all"], res["labels"]
    ev, ev_l1 = res["ev"], res["ev_l1"]
    by_class: dict[str, int] = {}
    for rec in labels.values():
        if rec["verdict"] == "fail":
            for c in rec.get("classes") or []:
                by_class[c] = by_class.get(c, 0) + 1
    ident = res["identity"]
    md = ["# Phase 0 tuning report", ""]
    md += [f"- labels: {len(labels_all)} clips labeled, **{len(labels)} usable** "
           f"({sum(1 for r in labels.values() if r['verdict'] == 'pass')} pass / "
           f"{sum(1 for r in labels.values() if r['verdict'] == 'fail')} fail; "
           f"fail classes: {by_class})",
           f"- excluded: {len(res['stale'])} stale/unverifiable label(s), "
           f"{len(res['metric_errors'])} undecodable clip(s) — listed below",
           f"- split: {len(res['tune_ids'])} tune / {len(res['test_ids'])} test "
           f"(stratified, seed {SPLIT_SEED})",
           (f"- judge: `{ident['model']}` @ `{ident['model_digest']}` "
            f"({len(res['l2_rows'])} clips scored)" if ident else
            "- judge: **none** — no judge scores for the configured judge; "
            "everything below is L1-only"),
           f"- proposed: `{proposed_path}` — **calibrated stays false until you "
           f"run --approve**",
           f"- objective: zero false-pass on tune, but no threshold may fail more "
           f"than {FF_CAP_FRAC:.0%} of labeled-pass tune clips", ""]
    l2_fallback = (res["l2_notes"] or {}).get("fallback")
    if l2_fallback:
        md += [f"- ⚠️ **L2 FLOORS ARE A FALLBACK — the objective above could not be "
               f"met.** {l2_fallback}. Recorded as provenance.l2_floors_fallback.", ""]
    if res["other_judges"]:
        md += ["- scores from other judges ignored: "
               + ", ".join(f"`{m}` @ `{d}`" for m, d in res["other_judges"]), ""]
    untested = sorted({(labels[c].get("classes") or ["pass"])[0] for c in res["tune_ids"]}
                      - {(labels[c].get("classes") or ["pass"])[0]
                         for c in res["test_ids"]})
    if untested:
        md += [f"- ⚠️ no held-out examples of: {', '.join(untested)} — their test "
               f"performance is unmeasured (label more to measure it)", ""]

    md += ["## L1 thresholds (selected on TUNE only)", "",
           "| check | class | role | value | tune separation |", "|---|---|---|---|---|"]
    for check, note in res["sweep_notes"].items():
        md.append(f"| {check} | {note.get('class')} | {note.get('role')} | "
                  f"{note.get('value')} | {note.get('note')}"
                  + (f" (missed {note['missed_on_tune']}, false-fail "
                     f"{note['false_fail_on_tune']}; n={note['n_bad']}+{note['n_good']})"
                     if "missed_on_tune" in note else "") + " |")
    md += ["", "## L2 floors (selected on TUNE only)", ""]
    l2n = res["l2_notes"]
    if l2n:
        md.append(f"- tuned on the {l2n['population']} tune clips that pass the proposed "
                  f"L1 ({l2n['n_bad']} labeled fail, {l2n['n_good']} labeled pass, "
                  f"{l2n['judge_errors']} judge errors) under the runtime rule: N/A "
                  f"excluded, any non-N/A dimension below its floor fails the clip")
        if l2n.get("fallback"):
            md.append(f"- ⚠️ **FALLBACK — {l2n['fallback']}**")
        md.append(f"- missed {l2n['missed_on_tune']}, false-fail "
                  f"{l2n['false_fail_on_tune']} on tune"
                  + (f" — {l2n['note']}" if l2n["note"] else ""))
        for dim, d in l2n["per_dim"].items():
            md.append(f"- {dim}: floor {d['floor']} (fails {d['fails_bad']} labeled-fail "
                      f"/ {d['fails_good']} labeled-pass tune clips; N/A on {d['na']})")
    else:
        md.append("- no judge scores for this judge — run `calibrate-tune --with-l2` "
                  "on the workstation to score the batch, or floors stay at the "
                  "default 3. The numbers below are L1-ONLY: deformed/off_prompt "
                  "fails listed as misses are exactly what L2 must catch — judge them "
                  "before trusting the pipeline.")

    md += ["", f"## Performance — {ev['test']['scope']} decision", ""]
    md += _perf_lines("TEST (held out)", ev["test"], labels)
    md += _perf_lines("tune", ev["tune"], labels)
    if ev["test"]["scope"] != "L1 only":
        md += ["", "L1 alone, for comparison:", ""]
        md += _perf_lines("TEST, L1 only", ev_l1["test"], labels)
    md += ["", "## Per-class confusion (TEST)", ""] + _confusion_md(ev["test"])
    md += ["", "## Per-class confusion (tune)", ""] + _confusion_md(ev["tune"])
    md += ["", "## Per-layer catch attribution (TEST)", ""] + _attribution_md(ev["test"])
    md += ["", "## Per-layer catch attribution (tune)", ""] + _attribution_md(ev["tune"])

    md += ["", "## Excluded / unjudgeable clips", ""]
    for cid, why in sorted(res["stale"].items()):
        md.append(f"- {cid}: label excluded — {why}")
    for cid, err in sorted(res["metric_errors"].items()):
        md.append(f"- {cid}: undecodable, excluded (at runtime: ERROR, never shipped) "
                  f"— {err}")
    for cid, row in sorted((res["l2_rows"] or {}).items()):
        if row.get("error"):
            md.append(f"- {cid}: judge could not evaluate — counted as a runtime "
                      f"ERROR (never ships) — {row['error']}")
    if md[-1] == "":
        md.append("- none")
    md += ["", "---",
           "_Test set touched once for this report. If you iterate on the judge "
           "prompt or thresholds after reading test numbers, that is a NEW "
           "calibration round — regenerate and relabel provenance accordingly._"]
    (cal / "tuning_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (cal / "tuning_report.json").write_text(json.dumps({
        "split": {"tune": res["tune_ids"], "test": res["test_ids"]},
        "excluded": {"stale_labels": res["stale"], "undecodable": res["metric_errors"]},
        "judge": ident, "other_judges": res["other_judges"],
        "l1": res["sweep_notes"], "l2": l2n,
        "tune": ev["tune"], "test": ev["test"],
        "l1_only": ev_l1}, indent=2, default=list) + "\n", encoding="utf-8")


# --------------------------------------------------------------- --with-l2

def score_with_judge(cal_dir: str, config_path: str,
                     pipeline_path: str = "pipeline.yaml",
                     out: dict | None = None) -> int:
    """--with-l2: score every manifest row with the configured VLM judge.

    Scored exactly the way the nightly checker will: config.yaml's judge with
    pipeline.yaml's judge timeout_s/retries (settings.effective_judge_cfg — what
    the runner hands the checker). A judge that could not be unloaded after the
    wave (openai_compat without judge.unload_url) is refused up front.
    Hard rule 3: ComfyUI's models are freed and free VRAM VERIFIED before the
    first judge call (gpu.to_judge, fail closed); the judge is unloaded in a
    finally. The runner's run lock is held while the VLM is on the GPU (refused,
    exit 2, while a nightly run holds it). Scores are keyed by (clip sha, prompt
    sha, judge model, digest): switching judge (8B -> 32B) re-scores everything;
    a clip the judge cannot evaluate gets an error row (reruns skip it) instead of
    aborting the batch. GPU/workstation path — UNVERIFIED-ON-GPU in the cloud
    session."""
    from .. import lock
    from ..check import load_judge_config
    from ..comfy import ComfyClient
    from ..judge.base import make_adapter
    from ..settings import effective_judge_cfg, judge_unload_problem, load_policies

    cal = Path(cal_dir)
    manifest = _read_jsonl(cal / "batch_manifest.jsonl")
    if not manifest:
        print(f"[calibrate-tune] no manifest in {cal}")
        return 2
    try:
        load_judge_config(config_path)           # a judge: mapping is required
        cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    except (InfraError, OSError, yaml.YAMLError) as e:
        print(f"[calibrate-tune] --with-l2 needs a usable judge config: "
              f"{getattr(e, 'message', e)}")
        return 2
    try:
        pol = load_policies(pipeline_path)
    except InfraError as e:
        print(f"[calibrate-tune] REFUSING --with-l2: {e.message}")
        return 2
    judge_cfg = effective_judge_cfg(cfg, pol)    # exactly the nightly checker's judge
    problem = judge_unload_problem(judge_cfg)
    if problem:
        print(f"[calibrate-tune] REFUSING --with-l2: {problem}")
        return 2
    comfy_cfg = cfg.get("comfy") or {}
    if not comfy_cfg.get("host"):
        print("[calibrate-tune] REFUSING --with-l2: config.yaml needs comfy.host — the "
              "judge may only load after ComfyUI's models are freed (/free) and free "
              "VRAM is verified (hard rule 3)")
        return 2
    comfy = ComfyClient(host=str(comfy_cfg["host"]),
                        workflow=str(comfy_cfg.get("workflow_t2v") or ""),
                        timeout_s=pol["comfy"]["timeout_s"],
                        poll_s=pol["comfy"]["poll_s"])
    try:
        adapter = make_adapter(judge_cfg)
        digest = adapter.model_digest()          # identity only (/api/tags), no VRAM
    except InfraError as e:
        print(f"[calibrate-tune] INFRA: {e.message}")
        return 3
    model = f"{adapter.name}/{adapter.model}"
    if out is not None:
        out["judge"] = {"model": model, "model_digest": digest}

    scores_path = cal / "l2_scores.jsonl"
    done = {_l2_key(r) for r in _read_jsonl(scores_path)}
    shas: dict = {}
    todo = []
    for row in manifest:
        path = resolve_clip_path(cal, row.get("clip_path"))
        csha = _sha(path, shas)
        if csha is None:
            print(f"[calibrate-tune] {row['clip_id']}: clip file missing ({path}) — "
                  f"not judged")
            continue
        psha = sha256_text(row["prompt"])
        if (row["clip_id"], csha, psha, model, digest) not in done:
            todo.append((row, path, csha, psha))
    if not todo:
        print(f"[calibrate-tune] every clip already scored by {model} ({digest})")
        return 0
    try:
        held = lock.acquire(lock.runs_base(cfg))
    except lock.LockHeld as e:
        print(f"[calibrate-tune] REFUSING --with-l2: another shortsloop process holds "
              f"{e} — calibration and the nightly runner share one GPU (hard rules "
              f"3/4); re-run when it has finished. No clip judged")
        return 2
    try:
        return _judge_wave(cal, todo, comfy, judge_cfg, pol, model, digest)
    finally:
        lock.release(held)


def _judge_wave(cal: Path, todo: list, comfy, judge_cfg: dict, pol: dict, model: str,
                digest: str) -> int:
    """The GPU part of --with-l2 — runs only while the run lock is held."""
    from .. import gpu
    from ..l2 import run_l2

    scores_path = cal / "l2_scores.jsonl"
    floors = {d: DEFAULT_FLOOR for d in L2_DIMS}   # raw scores are what we keep
    print(f"[calibrate-tune] judging {len(todo)} clip(s) with {model} ({digest}) — "
          f"freeing ComfyUI and verifying VRAM first (hard rule 3)", flush=True)
    handoff = pol["vram_handoff"]
    try:
        free_gb = gpu.to_judge(comfy, handoff["free_min_gb"], handoff["wait_timeout_s"])
    except InfraError as e:
        print(f"[calibrate-tune] INFRA: {e.message} — no clip judged")
        return 3
    print(f"[calibrate-tune] VRAM verified: {free_gb:.1f} GB free for the judge")
    errors = 0
    try:
        for row, path, csha, psha in todo:
            rec = {"clip_id": row["clip_id"], "clip_sha256": csha, "prompt_sha256": psha,
                   "model": model, "model_digest": digest,
                   "ts": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            try:
                block, _raw = run_l2(path, None, row["prompt"], judge_cfg, floors)
            except ClipError as e:
                errors += 1
                rec["error"] = f"{e.stage}: {e.message}"
                print(f"[calibrate-tune] {row['clip_id']}: judge could not evaluate "
                      f"({rec['error']}) — recorded; reruns skip it")
            else:
                rec["model_digest"] = block["model_digest"]
                rec["dimensions"] = {d: v["score"] for d, v in block["dimensions"].items()}
                rec["na"] = {d: v["na"] for d, v in block["dimensions"].items()}
                print(f"[calibrate-tune] judged {row['clip_id']}")
            with open(scores_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except InfraError as e:
        print(f"[calibrate-tune] INFRA: {e.message} — stopping; scored clips are kept, "
              f"re-run to continue")
        return 3
    finally:
        notes = gpu.unload_llms(judge_cfg, None)       # the judge unloads after its wave
        print(f"[calibrate-tune] {'; '.join(notes) or 'no judge unload hook'}")
    if errors:
        print(f"[calibrate-tune] {errors} clip(s) unjudgeable — listed in the report "
              f"as runtime ERRORs")
    return 0


def main_tune(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="shortsloop calibrate-tune")
    ap.add_argument("--calibration", default="calibration")
    ap.add_argument("--approve", action="store_true",
                    help="sign-off gate: copy proposed thresholds to --target with "
                         "calibrated: true")
    ap.add_argument("--accept-untested-l2", action="store_true",
                    help="with --approve only — SUPERVISED EXCEPTION: sign off a "
                         "proposal with no judge scores behind its L2 floors "
                         "(recorded as provenance.l2_untested_accepted: true)")
    ap.add_argument("--target", default="thresholds.yaml")
    ap.add_argument("--with-l2", action="store_true",
                    help="score the batch with the VLM judge first (workstation)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--pipeline", default="pipeline.yaml")
    args = ap.parse_args(argv)
    if args.accept_untested_l2 and not args.approve:
        print("[calibrate-tune] --accept-untested-l2 only modifies --approve (the "
              "sign-off gate); nothing done")
        return 2
    if args.approve:
        if args.with_l2:
            print("[calibrate-tune] --approve is the separate sign-off step: run "
                  "--with-l2 (and read the report) first")
            return 2
        return run_tune(args.calibration, approve=True, target=args.target,
                        config_path=args.config,
                        accept_untested_l2=args.accept_untested_l2)
    cfg_judge = _configured_judge(args.config)
    judge = {"model": cfg_judge, "model_digest": None} if cfg_judge else None
    if args.with_l2:
        got: dict = {}
        code = score_with_judge(args.calibration, args.config,
                                pipeline_path=args.pipeline, out=got)
        if code != 0:
            return code
        judge = got.get("judge") or judge
    return run_tune(args.calibration, target=args.target, judge=judge)
