"""`shortsloop calibrate-tune` — tune thresholds against hand labels and report
agreement honestly (docs/plan.md §6 steps 4-5).

Discipline:
- Stratified tune/test split (fixed seed); ALL selection happens on the tune set;
  the test set is touched once, for the report.
- Detector checks (motion/freeze for `static`, flicker/ssim_floor for `flicker`)
  are swept to zero false-pass on tune with minimal false-fail.
- Guard checks (sharpness/black/exposure) have no labeled class in the Phase 0
  vocabulary: they are set as guard rails beyond the worst labeled-PASS clip —
  they exist to catch catastrophes, never to fail a good clip.
- deformed / off_prompt are L2's job; the report says what L1 leaves for L2.
- Writes thresholds.proposed.yaml (calibrated: false). Flipping calibrated: true
  happens ONLY via --approve after the human sign-off gate.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .. import l1
from ..state import _read_jsonl
from ..verdict import sha256_file

CHECK_DEFS = {   # name -> (metric, op) — §2.4 frozen shape
    "motion":     ("flow_mag_median", ">="),
    "freeze":     ("freeze_longest_run_s", "<="),
    "flicker":    ("flicker_dips", "<="),
    "ssim_floor": ("ssim_min", ">="),
    "sharpness":  ("laplacian_p10", ">="),
    "black":      ("black_frame_frac", "<="),
    "exposure":   ("clipped_frac", "<="),
}
DETECTORS = {"static": ["motion", "freeze"], "flicker": ["flicker", "ssim_floor"]}
GUARDS = ("sharpness", "black", "exposure")
L2_DIM_FOR_CLASS = {"off_prompt": ["prompt_adherence"],
                    "deformed": ["subject_consistency", "anatomy_artifacts"],
                    "flicker": ["temporal_coherence"],
                    "other": ["imaging_quality"]}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def last_labels(cal_dir: Path) -> dict[str, dict]:
    """Latest label per clip wins (relabeling appends)."""
    out: dict[str, dict] = {}
    for rec in _read_jsonl(cal_dir / "labels.jsonl"):
        out[rec["clip_id"]] = rec
    return out


def load_metrics(cal_dir: Path, manifest: list[dict]) -> dict[str, dict]:
    """L1 metrics per clip_id, cached in metrics.jsonl; swap rows share their
    source clip's pixels so metrics are computed once per distinct file."""
    cache_path = cal_dir / "metrics.jsonl"
    cached = {r["clip_id"]: r["metrics"] for r in _read_jsonl(cache_path)}
    by_path: dict[str, dict] = {}
    for row in manifest:
        cid = row["clip_id"]
        if cid in cached:
            by_path.setdefault(str(row.get("clip_path")), cached[cid])
    new_lines = []
    for row in manifest:
        cid, path = row["clip_id"], str(row.get("clip_path"))
        if cid in cached:
            continue
        if path in by_path:
            cached[cid] = by_path[path]
        else:
            if not path or not Path(path).is_file():
                continue
            cached[cid] = l1.compute_metrics(path)["metrics"]
            by_path[path] = cached[cid]
        new_lines.append({"clip_id": cid, "metrics": cached[cid]})
    if new_lines:
        with open(cache_path, "a", encoding="utf-8") as f:
            for line in new_lines:
                f.write(json.dumps(line) + "\n")
    return cached


def stratified_split(labels: dict[str, dict], ratio: float = 0.7,
                     seed: int = 13) -> tuple[list[str], list[str]]:
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


def _passes(value: float, op: str, threshold: float) -> bool:
    return value >= threshold if op == ">=" else value <= threshold


def sweep_detector(check: str, target_class: str, ids: list[str],
                   labels: dict, metrics: dict) -> dict:
    """Pick the threshold minimizing (missed target-class fails, false-fails on
    labeled-pass clips) over the tune set."""
    metric, op = CHECK_DEFS[check]
    bad = [metrics[c][metric] for c in ids
           if labels[c]["verdict"] == "fail" and target_class in labels[c]["classes"]
           and c in metrics]
    good = [metrics[c][metric] for c in ids
            if labels[c]["verdict"] == "pass" and c in metrics]
    if not bad:
        return {"value": None, "note": f"untuned — no labeled {target_class} examples"}
    values = sorted(set(bad + good))
    candidates = [values[0] - abs(values[0]) * 0.5 - 1e-9, values[-1] + abs(values[-1]) * 0.5 + 1e-9]
    candidates += [(a + b) / 2 for a, b in zip(values, values[1:])]
    best = None
    for t in candidates:
        missed = sum(1 for v in bad if _passes(v, op, t))       # bad clip not caught
        falsefail = sum(1 for v in good if not _passes(v, op, t))
        score = (missed, falsefail)
        if best is None or score < best[0]:
            best = (score, t)
    (missed, falsefail), t = best
    return {"value": round(float(t), 6), "missed_on_tune": missed,
            "false_fail_on_tune": falsefail,
            "n_bad": len(bad), "n_good": len(good),
            "note": "clean separation" if (missed, falsefail) == (0, 0)
                    else "OVERLAP — could not fully separate on tune set"}


def guard_value(check: str, ids: list[str], labels: dict, metrics: dict,
                fallback: float) -> float:
    """Guard rail beyond the worst labeled-PASS clip (never false-fails tune)."""
    metric, op = CHECK_DEFS[check]
    good = [metrics[c][metric] for c in ids
            if labels[c]["verdict"] == "pass" and c in metrics]
    if not good:
        return fallback
    if op == ">=":
        return round(min(good) * 0.5, 6)
    return round(min(1.0, max(good) * 1.5 + 0.02), 6)


def evaluate_l1(ids: list[str], labels: dict, metrics: dict,
                l1_section: dict) -> dict:
    tp = fp = tn = fn = 0                       # positive = FAIL (defect present)
    false_pass_ids, false_fail_ids = [], []
    for cid in ids:
        if cid not in metrics:
            continue
        pred_fail = any(
            spec["value"] is not None and
            not _passes(metrics[cid][spec["metric"]], spec["op"], spec["value"])
            for spec in l1_section.values())
        actual_fail = labels[cid]["verdict"] == "fail"
        if actual_fail and pred_fail:
            tp += 1
        elif actual_fail and not pred_fail:
            fn += 1
            false_pass_ids.append(cid)
        elif not actual_fail and pred_fail:
            fp += 1
            false_fail_ids.append(cid)
        else:
            tn += 1
    n = tp + fp + tn + fn
    return {"n": n, "agreement": (tp + tn) / n if n else None,
            "false_pass": fn, "false_pass_rate": fn / n if n else None,
            "false_pass_ci": wilson(fn, n),
            "false_fail": fp, "false_pass_ids": false_pass_ids,
            "false_fail_ids": false_fail_ids}


def run_tune(cal_dir: str, approve: bool = False, target: str = "thresholds.yaml",
             base_thresholds: str | None = None) -> int:
    cal = Path(cal_dir)
    manifest = _read_jsonl(cal / "batch_manifest.jsonl")
    labels = last_labels(cal)
    if approve:
        return _approve(cal, target)
    if not manifest:
        print(f"[calibrate-tune] no manifest in {cal}")
        return 2
    if len(labels) < 10:
        print(f"[calibrate-tune] only {len(labels)} labels — need the labeled batch "
              f"(shortsloop label) before tuning")
        return 2

    metrics = load_metrics(cal, manifest)
    labeled_ids = [c for c in labels if c in metrics]
    tune_ids, test_ids = stratified_split({c: labels[c] for c in labeled_ids})

    l1_section: dict[str, dict] = {}
    sweep_notes: dict[str, dict] = {}
    for cls, checks in DETECTORS.items():
        for check in checks:
            res = sweep_detector(check, cls, tune_ids, labels, metrics)
            sweep_notes[check] = {**res, "class": cls}
            metric, op = CHECK_DEFS[check]
            l1_section[check] = {"metric": metric, "op": op, "value": res["value"]}
    fallbacks = {"sharpness": 5.0, "black": 0.5, "exposure": 0.5}
    for check in GUARDS:
        metric, op = CHECK_DEFS[check]
        val = guard_value(check, tune_ids, labels, metrics, fallbacks[check])
        l1_section[check] = {"metric": metric, "op": op, "value": val}
        sweep_notes[check] = {"value": val, "class": "(guard)",
                              "note": "guard rail from labeled-pass margins"}

    ev_tune = evaluate_l1(tune_ids, labels, metrics, l1_section)
    ev_test = evaluate_l1(test_ids, labels, metrics, l1_section)

    l2_scores = {r["clip_id"]: r for r in _read_jsonl(cal / "l2_scores.jsonl")}
    floors = {d: 3 for d in ("prompt_adherence", "subject_consistency",
                             "anatomy_artifacts", "temporal_coherence",
                             "imaging_quality")}
    l2_notes = {}
    if l2_scores:
        for cls, dims in L2_DIM_FOR_CLASS.items():
            for dim in dims:
                bad = [l2_scores[c]["dimensions"][dim] for c in tune_ids
                       if c in l2_scores and labels[c]["verdict"] == "fail"
                       and cls in labels[c]["classes"]]
                good = [l2_scores[c]["dimensions"][dim] for c in tune_ids
                        if c in l2_scores and labels[c]["verdict"] == "pass"]
                if not bad:
                    l2_notes[dim] = "untuned — no labeled examples; floor stays 3"
                    continue
                best = None
                for floor in (2, 3, 4, 5):
                    missed = sum(1 for s in bad if s >= floor)
                    falsefail = sum(1 for s in good if s < floor)
                    if best is None or (missed, falsefail) < best[0]:
                        best = ((missed, falsefail), floor)
                floors[dim] = best[1]
                l2_notes[dim] = (f"floor {best[1]} (missed {best[0][0]}, "
                                 f"false-fail {best[0][1]} on tune)")
    labels_sha = sha256_file(cal / "labels.jsonl")
    proposed = {
        "version": datetime.now(timezone.utc).strftime("%Y-%m-%d") + ".1",
        "calibrated": False,   # flips ONLY via --approve (human sign-off gate)
        "provenance": {
            "labels_sha256": labels_sha,
            "tuned_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tune_test_split": f"{len(tune_ids)}/{len(test_ids)} stratified seed=13",
            "test_agreement": ev_test["agreement"],
            "test_false_pass": f"{ev_test['false_pass']}/{ev_test['n']} "
                               f"(95% CI {ev_test['false_pass_ci'][0]:.2f}"
                               f"–{ev_test['false_pass_ci'][1]:.2f})",
        },
        "l1": l1_section,
        "l2": {"floors": floors},
    }
    proposed_path = cal / "thresholds.proposed.yaml"
    proposed_path.write_text(yaml.safe_dump(proposed, sort_keys=False,
                                            allow_unicode=True), encoding="utf-8")
    _write_report(cal, labels, tune_ids, test_ids, sweep_notes, ev_tune, ev_test,
                  l2_scores, l2_notes, proposed_path)
    print(f"[calibrate-tune] proposed thresholds: {proposed_path}")
    print(f"[calibrate-tune] report: {cal / 'tuning_report.md'}")
    print(f"[calibrate-tune] test agreement {ev_test['agreement']:.0%}, "
          f"false-pass {ev_test['false_pass']}/{ev_test['n']} — review, then "
          f"sign off with: shortsloop calibrate-tune --calibration {cal} --approve")
    return 0


def _approve(cal: Path, target: str) -> int:
    proposed_path = cal / "thresholds.proposed.yaml"
    if not proposed_path.is_file():
        print(f"[calibrate-tune] nothing to approve — {proposed_path} missing")
        return 2
    data = yaml.safe_load(proposed_path.read_text(encoding="utf-8"))
    current_sha = sha256_file(cal / "labels.jsonl")
    if data.get("provenance", {}).get("labels_sha256") != current_sha:
        print("[calibrate-tune] REFUSING approve: labels.jsonl changed since tuning "
              "— re-run calibrate-tune first")
        return 2
    data["calibrated"] = True
    Path(target).write_text(yaml.safe_dump(data, sort_keys=False,
                                           allow_unicode=True), encoding="utf-8")
    print(f"[calibrate-tune] APPROVED → {target} (calibrated: true). "
          f"The unattended gate is now open.")
    return 0


def _write_report(cal, labels, tune_ids, test_ids, sweep_notes, ev_tune, ev_test,
                  l2_scores, l2_notes, proposed_path) -> None:
    by_class: dict[str, int] = {}
    for rec in labels.values():
        if rec["verdict"] == "fail":
            for c in rec["classes"]:
                by_class[c] = by_class.get(c, 0) + 1
    md = ["# Phase 0 tuning report", ""]
    md += [f"- labels: {len(labels)} clips "
           f"({sum(1 for r in labels.values() if r['verdict'] == 'pass')} pass / "
           f"{sum(1 for r in labels.values() if r['verdict'] == 'fail')} fail; "
           f"fail classes: {by_class})",
           f"- split: {len(tune_ids)} tune / {len(test_ids)} test (stratified, seed 13)",
           f"- proposed: `{proposed_path}` — **calibrated stays false until you "
           f"run --approve**", ""]
    md += ["## L1 thresholds (selected on TUNE only)", "",
           "| check | class | value | tune separation |", "|---|---|---|---|"]
    for check, note in sweep_notes.items():
        md.append(f"| {check} | {note.get('class')} | {note.get('value')} | "
                  f"{note.get('note')}"
                  + (f" (missed {note['missed_on_tune']}, false-fail "
                     f"{note['false_fail_on_tune']}; n={note['n_bad']}+{note['n_good']})"
                     if "missed_on_tune" in note else "") + " |")
    md += ["", "## L1-only performance", ""]
    for name, ev in (("tune", ev_tune), ("TEST (held out)", ev_test)):
        if ev["n"]:
            lo, hi = ev["false_pass_ci"]
            md.append(f"- **{name}** (n={ev['n']}): agreement {ev['agreement']:.0%}, "
                      f"**false-pass {ev['false_pass']}** "
                      f"(rate {ev['false_pass_rate']:.0%}, 95% CI {lo:.0%}–{hi:.0%}), "
                      f"false-fail {ev['false_fail']}")
            if ev["false_pass_ids"]:
                left = [c for c in ev["false_pass_ids"]
                        if set(labels[c]["classes"]) & {"deformed", "off_prompt", "other"}]
                hard = [c for c in ev["false_pass_ids"] if c not in left]
                if left:
                    md.append(f"  - passed L1 but labeled fail (L2's job — "
                              f"deformed/off_prompt/other): {', '.join(left)}")
                if hard:
                    md.append(f"  - ⚠️ L1-class clips L1 missed: {', '.join(hard)}")
    md += ["", "## L2 floors", ""]
    if l2_scores:
        for dim, note in l2_notes.items():
            md.append(f"- {dim}: {note}")
        md.append(f"- (judge scores available for {len(l2_scores)} clips)")
    else:
        md.append("- no l2_scores.jsonl — run `calibrate-tune --with-l2` on the "
                  "workstation to score the batch with the judge, or floors stay "
                  "at the default 3. The false-pass numbers above are L1-ONLY: "
                  "deformed/off_prompt fails listed as L1 misses are exactly what "
                  "L2 must catch — judge them before trusting the pipeline.")
    md += ["", "---",
           "_Test set touched once for this report. If you iterate on the judge "
           "prompt or thresholds after reading test numbers, that is a NEW "
           "calibration round — regenerate and relabel provenance accordingly._"]
    (cal / "tuning_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def score_with_judge(cal_dir: str, config_path: str) -> int:
    """--with-l2: run the VLM judge over every manifest row, cache dim scores.
    GPU/workstation path — UNVERIFIED-ON-GPU in the cloud session."""
    from ..check import load_judge_config
    from ..l2 import run_l2

    cal = Path(cal_dir)
    manifest = _read_jsonl(cal / "batch_manifest.jsonl")
    judge_cfg = load_judge_config(config_path)
    done = {r["clip_id"] for r in _read_jsonl(cal / "l2_scores.jsonl")}
    floors = {d: 3 for d in ("prompt_adherence", "subject_consistency",
                             "anatomy_artifacts", "temporal_coherence",
                             "imaging_quality")}
    for row in manifest:
        if row["clip_id"] in done or not row.get("clip_path"):
            continue
        block, _raw = run_l2(row["clip_path"], None, row["prompt"], judge_cfg, floors)
        rec = {"clip_id": row["clip_id"],
               "dimensions": {d: v["score"] for d, v in block["dimensions"].items()},
               "na": {d: v["na"] for d, v in block["dimensions"].items()},
               "model": block["model"], "model_digest": block["model_digest"]}
        with open(cal / "l2_scores.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[calibrate-tune] judged {row['clip_id']}")
    return 0


def main_tune(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="shortsloop calibrate-tune")
    ap.add_argument("--calibration", default="calibration")
    ap.add_argument("--approve", action="store_true",
                    help="sign-off gate: copy proposed thresholds to --target with "
                         "calibrated: true")
    ap.add_argument("--target", default="thresholds.yaml")
    ap.add_argument("--with-l2", action="store_true",
                    help="score the batch with the VLM judge first (workstation)")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args(argv)
    if args.with_l2:
        code = score_with_judge(args.calibration, args.config)
        if code != 0:
            return code
    return run_tune(args.calibration, approve=args.approve, target=args.target)
