"""calibrate-tune: joint detector sweeps, L2 floors under the runtime rule, honest
held-out evaluation (plan §6), degenerate-input guards, label/metric staleness,
and the --approve sign-off gate."""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml

from shortsloop.calibrate import tune
from shortsloop.calibrate.tune import evaluate_l1, run_tune, stratified_split, wilson
from shortsloop.check import load_thresholds
from shortsloop.state import _read_jsonl
from shortsloop.thresholds import validate_thresholds
from shortsloop.verdict import sha256_file, sha256_text

DIMS = ("prompt_adherence", "subject_consistency", "anatomy_artifacts",
        "temporal_coherence", "imaging_quality")
JUDGE = "ollama/qwen3-vl:8b-instruct"
DIGEST = "sha256:fakedigest"

BASE_M = {"flow_mag_median": 0.003, "flow_mag_p90": 0.004, "ssim_min": 0.7,
          "ssim_p05": 0.8, "ssim_mean": 0.95, "flicker_dips": 0,
          "freeze_longest_run_s": 0.0, "laplacian_p10": 60.0,
          "laplacian_median": 90.0, "luma_mean": 0.4, "black_frame_frac": 0.0,
          "clipped_frac": 0.01}


# ------------------------------------------------------------------ fixtures

def mk_cal(tmp_path, rows, labels, *, l2=None, name="cal"):
    """Calibration dir: one small distinct file per clip, metrics pre-cached under
    the clip's sha (no decode), labels carrying the labeled clip's sha.
    rows: [{clip_id, prompt?, _m}]; labels: {cid: (verdict, classes)};
    l2: {cid: {"dims": {dim: score}, "na": {dim: bool}}} or {cid: {"error": msg}}."""
    cal = tmp_path / name
    (cal / "clips").mkdir(parents=True)
    l1_impl = getattr(tune, "L1_IMPL", None)
    shas, prompts = {}, {}
    with open(cal / "batch_manifest.jsonl", "w") as man, \
            open(cal / "metrics.jsonl", "w") as met:
        for r in rows:
            p = cal / "clips" / f"{r['clip_id']}.mp4"
            p.write_bytes(f"fake clip bytes {r['clip_id']}".encode())
            shas[r["clip_id"]] = sha256_file(p)
            prompts[r["clip_id"]] = r.get("prompt", f"prompt for {r['clip_id']}")
            man.write(json.dumps({"clip_id": r["clip_id"], "kind": "good",
                                  "prompt": prompts[r["clip_id"]],
                                  "clip_path": str(p)}) + "\n")
            met.write(json.dumps({"clip_id": r["clip_id"],
                                  "clip_sha256": shas[r["clip_id"]],
                                  "l1_impl": l1_impl, "metrics": r["_m"]}) + "\n")
    with open(cal / "labels.jsonl", "w") as f:
        for cid, (verdict, classes) in labels.items():
            f.write(json.dumps({"clip_id": cid, "verdict": verdict,
                                "classes": classes, "ts": 1,
                                "clip_sha256": shas[cid]}) + "\n")
    if l2:
        write_l2(cal, l2, shas, prompts)
    return cal


def write_l2(cal, l2, shas=None, prompts=None, model=JUDGE, digest=DIGEST):
    manifest = {r["clip_id"]: r for r in _read_jsonl(cal / "batch_manifest.jsonl")}
    with open(cal / "l2_scores.jsonl", "a") as f:
        for cid, spec in l2.items():
            row = manifest[cid]
            rec = {"clip_id": cid,
                   "clip_sha256": sha256_file(row["clip_path"]),
                   "prompt_sha256": sha256_text(row["prompt"]),
                   "model": model, "model_digest": digest}
            if "error" in spec:
                rec["error"] = spec["error"]
            else:
                rec["dimensions"] = {d: spec["dims"].get(d, 4) for d in DIMS}
                rec["na"] = {d: (spec.get("na") or {}).get(d, False) for d in DIMS}
            f.write(json.dumps(rec) + "\n")


def scores(**over):
    return {"dims": {d: over.get(d, 4) for d in DIMS}}


def base_rows_and_labels(n_pass=8):
    rows, labels = [], {}
    for i in range(n_pass):
        rows.append({"clip_id": f"p{i}",
                     "_m": dict(BASE_M, flow_mag_median=0.003 + i * 0.0004)})
        labels[f"p{i}"] = ("pass", [])
    for i in range(6):
        rows.append({"clip_id": f"s{i}",
                     "_m": dict(BASE_M, flow_mag_median=1e-5 * (i + 1),
                                freeze_longest_run_s=2.5)})
        labels[f"s{i}"] = ("fail", ["static"])
    for i in range(4):
        rows.append({"clip_id": f"f{i}",
                     "_m": dict(BASE_M, flicker_dips=8 + i, ssim_min=0.35)})
        labels[f"f{i}"] = ("fail", ["flicker"])
    for i in range(3):
        rows.append({"clip_id": f"d{i}", "_m": dict(BASE_M)})
        labels[f"d{i}"] = ("fail", ["deformed"])
    return rows, labels


def proposal(cal) -> dict:
    return yaml.safe_load((cal / "thresholds.proposed.yaml").read_text())


def report_json(cal) -> dict:
    return json.loads((cal / "tuning_report.json").read_text())


FEASIBLE = {"flow_mag_median": (0, float("inf")), "freeze_longest_run_s": (0, float("inf")),
            "flicker_dips": (0, float("inf")), "ssim_min": (-1, 1),
            "laplacian_p10": (0, float("inf")), "black_frame_frac": (0, 1),
            "clipped_frac": (0, 1)}


def assert_feasible(prop):
    for name, spec in prop["l1"].items():
        lo, hi = FEASIBLE[spec["metric"]]
        assert lo <= spec["value"] <= hi, (name, spec)


def _passes(v, op, t):
    return v >= t if op == ">=" else v <= t


def l1_false_fails(prop, rows, labels, ids):
    m = {r["clip_id"]: r["_m"] for r in rows}
    return [c for c in ids if labels[c][0] == "pass" and any(
        not _passes(m[c][s["metric"]], s["op"], s["value"]) for s in prop["l1"].values())]


def tune_test_ids(labels):
    return stratified_split({c: {"verdict": v, "classes": cl}
                             for c, (v, cl) in labels.items()})


# ------------------------------------------- [82] joint sweep vs the OR rule

def _mild_flicker_fixture():
    """14 good clips with fast motion (ssim_min 0.40-0.86); 5 mildly flickering
    clips: flicker_dips separates them perfectly, their ssim_min is HIGH."""
    rows, labels = [], {}
    for i in range(14):
        rows.append({"clip_id": f"g{i:02d}",
                     "_m": dict(BASE_M, ssim_min=round(0.40 + i * 0.035, 3),
                                flicker_dips=i % 2)})
        labels[f"g{i:02d}"] = ("pass", [])
    for i in range(5):
        rows.append({"clip_id": f"k{i}",
                     "_m": dict(BASE_M, ssim_min=0.86 + i * 0.01, flicker_dips=6 + i)})
        labels[f"k{i}"] = ("fail", ["flicker"])
    return rows, labels


def test_secondary_detector_does_not_fail_every_clip(tmp_path):
    rows, labels = _mild_flicker_fixture()
    cal = mk_cal(tmp_path, rows, labels)
    assert run_tune(str(cal)) == 0
    prop = proposal(cal)
    assert prop["l1"]["ssim_floor"]["value"] <= 1.0           # was 1.29 (> max SSIM)
    assert_feasible(prop)
    tune_ids, test_ids = tune_test_ids(labels)
    assert l1_false_fails(prop, rows, labels, tune_ids) == []
    assert l1_false_fails(prop, rows, labels, test_ids) == []
    # flicker_dips still catches every flicker clip
    assert all(rows[i]["_m"]["flicker_dips"] > prop["l1"]["flicker"]["value"]
               for i in range(14, 19))


def test_secondary_is_tuned_on_what_the_primary_misses(tmp_path):
    """One flicker clip has no dips but a catastrophic SSIM drop: ssim_floor is
    tuned to catch exactly that residual, without failing labeled-pass clips."""
    rows, labels = _mild_flicker_fixture()
    for i in range(3):
        rows.append({"clip_id": f"z{i}", "_m": dict(BASE_M, ssim_min=0.05 + 0.02 * i,
                                                    flicker_dips=0)})
        labels[f"z{i}"] = ("fail", ["flicker"])
    cal = mk_cal(tmp_path, rows, labels)
    assert run_tune(str(cal)) == 0
    prop = proposal(cal)
    floor = prop["l1"]["ssim_floor"]["value"]
    assert 0.09 < floor < 0.40
    tune_ids, _ = tune_test_ids(labels)
    assert l1_false_fails(prop, rows, labels, tune_ids) == []


def test_zero_miss_is_not_bought_with_mass_false_fails(tmp_path):
    """A residual clip only catchable by failing every good clip stays a reported
    miss: the threshold stays inside the good cluster's permissive side."""
    rows, labels = _mild_flicker_fixture()
    rows.append({"clip_id": "zz", "_m": dict(BASE_M, ssim_min=0.95, flicker_dips=0)})
    labels["zz"] = ("fail", ["flicker"])
    cal = mk_cal(tmp_path, rows, labels)
    assert run_tune(str(cal)) == 0
    prop = proposal(cal)
    assert_feasible(prop)
    tune_ids, _ = tune_test_ids(labels)
    assert len(l1_false_fails(prop, rows, labels, tune_ids)) <= 2


# ------------------------------------------- [92] evaluated value == written value

def _row_counts(report_md: str, check: str):
    m = re.search(rf"^\| {check} \|.*missed (\d+), false-fail (\d+)", report_md, re.M)
    assert m, f"no sweep row for {check}"
    return int(m.group(1)), int(m.group(2))


def test_reported_tune_separation_matches_the_written_threshold(tmp_path):
    """Bad and good motion values 2e-10 apart: the separating midpoint does not
    survive rounding. The report must describe the value actually written."""
    rows, labels = [], {}
    for i in range(8):
        rows.append({"clip_id": f"p{i}", "_m": dict(BASE_M, flow_mag_median=0.0010000003
                                                    + i * 1e-3)})
        labels[f"p{i}"] = ("pass", [])
    for i in range(6):
        rows.append({"clip_id": f"s{i}", "_m": dict(BASE_M, flow_mag_median=0.0010000001
                                                    - i * 1e-12)})
        labels[f"s{i}"] = ("fail", ["static"])
    cal = mk_cal(tmp_path, rows, labels)
    assert run_tune(str(cal)) == 0
    prop = proposal(cal)
    t = prop["l1"]["motion"]["value"]
    tune_ids, _ = tune_test_ids(labels)
    m = {r["clip_id"]: r["_m"] for r in rows}
    missed = sum(1 for c in tune_ids if labels[c][0] == "fail"
                 and m[c]["flow_mag_median"] >= t)
    ff = sum(1 for c in tune_ids if labels[c][0] == "pass"
             and m[c]["flow_mag_median"] < t)
    assert _row_counts((cal / "tuning_report.md").read_text(), "motion") == (missed, ff)


def test_zero_count_metric_never_gets_a_negative_threshold(tmp_path):
    rows, labels = base_rows_and_labels()
    for i in range(3):                               # flicker clips with zero dips
        rows.append({"clip_id": f"f9{i}", "_m": dict(BASE_M, flicker_dips=0)})
        labels[f"f9{i}"] = ("fail", ["flicker"])
    for r in rows:                                   # ssim does not separate
        if r["clip_id"].startswith("f"):
            r["_m"]["ssim_min"] = 0.7
    cal = mk_cal(tmp_path, rows, labels)
    assert run_tune(str(cal)) == 0
    prop = proposal(cal)
    assert_feasible(prop)
    v = prop["l1"]["flicker"]["value"]
    assert v >= 0 and str(v) != "-0.0"
    tune_ids, _ = tune_test_ids(labels)
    m = {r["clip_id"]: r["_m"] for r in rows}
    bad = [c for c in tune_ids if "flicker" in labels[c][1]]
    good = [c for c in tune_ids if labels[c][0] == "pass"]
    missed = sum(1 for c in bad if m[c]["flicker_dips"] <= v)
    ff = sum(1 for c in good if m[c]["flicker_dips"] > v)
    assert _row_counts((cal / "tuning_report.md").read_text(), "flicker") == (missed, ff)


# ------------------------------------------ [83]/[91] L2 floors, runtime rule

def _l2_fixture(flicker_tc=4):
    """14 good clips scoring 4 everywhere; 5 flicker clips L1 rejects (their
    temporal_coherence is `flicker_tc`); 5 deformed clips L1 passes whose defect
    registers only in subject_consistency."""
    rows, labels, l2 = [], {}, {}
    for i in range(14):
        rows.append({"clip_id": f"g{i:02d}",
                     "_m": dict(BASE_M, flow_mag_median=0.003 + i * 1e-4)})
        labels[f"g{i:02d}"] = ("pass", [])
        l2[f"g{i:02d}"] = scores()
    for i in range(5):
        rows.append({"clip_id": f"k{i}", "_m": dict(BASE_M, flicker_dips=8 + i,
                                                    ssim_min=0.3)})
        labels[f"k{i}"] = ("fail", ["flicker"])
        l2[f"k{i}"] = scores(temporal_coherence=flicker_tc)
    for i in range(5):
        rows.append({"clip_id": f"d{i}", "_m": dict(BASE_M)})
        labels[f"d{i}"] = ("fail", ["deformed"])
        l2[f"d{i}"] = scores(subject_consistency=2)
    return rows, labels, l2


def test_l2_floors_never_fail_publishable_clips(tmp_path):
    rows, labels, l2 = _l2_fixture()
    cal = mk_cal(tmp_path, rows, labels, l2=l2)
    assert run_tune(str(cal)) == 0
    floors = proposal(cal)["l2"]["floors"]
    assert max(floors.values()) <= 4, floors           # 4 = "you would publish it"
    assert floors["subject_consistency"] >= 3          # deformed clips are caught
    rep = report_json(cal)
    assert rep["tune"]["false_fail"] == 0
    assert rep["tune"]["false_pass"] == 0


def test_l1_rejected_clips_do_not_move_l2_floors(tmp_path):
    a_rows, a_labels, a_l2 = _l2_fixture(flicker_tc=4)
    b_rows, b_labels, b_l2 = _l2_fixture(flicker_tc=1)
    cal_a = mk_cal(tmp_path, a_rows, a_labels, l2=a_l2, name="a")
    cal_b = mk_cal(tmp_path, b_rows, b_labels, l2=b_l2, name="b")
    assert run_tune(str(cal_a)) == 0 and run_tune(str(cal_b)) == 0
    assert proposal(cal_a)["l2"]["floors"] == proposal(cal_b)["l2"]["floors"]


def test_na_dimension_is_excluded_like_the_runtime_does(tmp_path):
    """A deformed landscape clip whose subject_consistency is N/A (score 1) is NOT
    caught by that dimension at runtime — the tuner must not count it as caught."""
    rows, labels, l2 = _l2_fixture()
    l2["d0"] = {"dims": {d: 4 for d in DIMS} | {"subject_consistency": 1},
                "na": {"subject_consistency": True}}
    cal = mk_cal(tmp_path, rows, labels, l2=l2)
    assert run_tune(str(cal)) == 0
    rep = report_json(cal)
    split = "tune" if "d0" in rep["split"]["tune"] else "test"
    assert "d0" in rep[split]["false_pass_ids"]
    assert rep[split]["clips"]["d0"]["l2_failed"] == []


# --------------------------------------- [90] held-out pipeline evaluation (§6)

def test_test_set_evaluates_the_combined_l1_l2_decision(tmp_path):
    rows, labels, l2 = _l2_fixture()
    cal = mk_cal(tmp_path, rows, labels, l2=l2)
    assert run_tune(str(cal)) == 0
    prov = proposal(cal)["provenance"]
    assert "L1+L2" in prov["test_scope"]
    assert prov["test_false_pass"].startswith("0/")    # deformed caught by L2
    assert prov["test_agreement"] == 1.0
    rep = report_json(cal)
    test = rep["test"]
    assert test["scope"] == "L1+L2"
    assert set(test["confusion"]["deformed"]) == {"deformed"}
    assert test["attribution"]["stopped_by"]["L2"] >= 1
    assert test["attribution"]["stopped_by"]["L1"] >= 1
    assert set(test["attribution"]["by_class"]["deformed"]) >= {"L2 only"}
    md = (cal / "tuning_report.md").read_text()
    assert "Per-class confusion" in md and "Per-layer catch attribution" in md


def test_l1_only_proposal_says_so(tmp_path):
    rows, labels = base_rows_and_labels()
    cal = mk_cal(tmp_path, rows, labels)
    assert run_tune(str(cal)) == 0
    prov = proposal(cal)["provenance"]
    assert prov["test_scope"].startswith("L1 only")
    assert prov["judge"] is None


# ------------------------------------------- [93] false-pass = miss rate of bad

def test_false_pass_rate_is_misses_among_labeled_fail_clips():
    labels, metrics = {}, {}
    for i in range(3):
        labels[f"s{i}"] = {"verdict": "fail", "classes": ["static"]}
        metrics[f"s{i}"] = dict(BASE_M, flow_mag_median=1e-5)
    labels["d0"] = {"verdict": "fail", "classes": ["deformed"]}
    metrics["d0"] = dict(BASE_M)
    for i in range(8):
        labels[f"p{i}"] = {"verdict": "pass", "classes": []}
        metrics[f"p{i}"] = dict(BASE_M)
    l1_section = {"motion": {"metric": "flow_mag_median", "op": ">=", "value": 0.001}}
    ev = evaluate_l1(sorted(labels), labels, metrics, l1_section)
    assert ev["false_pass"] == 1
    assert ev["false_pass_rate"] == 0.25                 # 1 of 4 bad clips, not 1/12
    assert tuple(ev["false_pass_ci"]) == wilson(1, 4)
    assert ev["n_bad"] == 4 and ev["n_good"] == 8


# ------------------------------------------------- [85] degenerate label sets

def _assert_refused(cal):
    assert run_tune(str(cal)) == 2
    assert not (cal / "thresholds.proposed.yaml").exists()


def test_guard_counts_usable_clips_not_raw_labels(tmp_path):
    rows, labels = base_rows_and_labels()
    rows, labels = rows[:12], {r["clip_id"]: labels[r["clip_id"]] for r in rows[:12]}
    cal = mk_cal(tmp_path, rows, labels)
    for r in rows[3:]:                                # 9 clip files vanish
        (cal / "clips" / f"{r['clip_id']}.mp4").unlink()
    _assert_refused(cal)
    assert run_tune(str(cal), approve=True, target=str(tmp_path / "t.yaml")) == 2


def test_all_pass_or_all_fail_labels_are_refused(tmp_path):
    rows = [{"clip_id": f"p{i}", "_m": dict(BASE_M)} for i in range(12)]
    _assert_refused(mk_cal(tmp_path, rows, {r["clip_id"]: ("pass", []) for r in rows},
                           name="allpass"))
    _assert_refused(mk_cal(tmp_path, rows,
                           {r["clip_id"]: ("fail", ["static"]) for r in rows},
                           name="allfail"))


def test_test_split_without_a_fail_clip_is_refused(tmp_path):
    """Singleton fail strata all land in tune: nothing labeled-fail is held out."""
    rows = [{"clip_id": f"p{i}", "_m": dict(BASE_M)} for i in range(10)]
    labels = {r["clip_id"]: ("pass", []) for r in rows}
    for cls in ("static", "deformed", "flicker"):
        rows.append({"clip_id": f"x_{cls}", "_m": dict(BASE_M)})
        labels[f"x_{cls}"] = ("fail", [cls])
    _assert_refused(mk_cal(tmp_path, rows, labels))


def test_refused_tune_removes_an_older_proposal(tmp_path):
    rows, labels = base_rows_and_labels()
    cal = mk_cal(tmp_path, rows, labels)
    assert run_tune(str(cal)) == 0
    for r in rows[2:]:
        (cal / "clips" / f"{r['clip_id']}.mp4").unlink()
    _assert_refused(cal)
    assert run_tune(str(cal), approve=True, target=str(tmp_path / "t.yaml")) == 2


# -------------------------------------------- [94] undecodable clip, no abort

def test_undecodable_clip_is_recorded_and_tuning_continues(tmp_path, clips, monkeypatch):
    rows, labels = base_rows_and_labels()
    cal = mk_cal(tmp_path, rows, labels)
    bad = cal / "clips" / "broken.mp4"
    shutil.copyfile(clips["corrupt"], bad)
    with open(cal / "batch_manifest.jsonl", "a") as f:
        f.write(json.dumps({"clip_id": "broken", "kind": "good", "prompt": "x",
                            "clip_path": str(bad)}) + "\n")
    with open(cal / "labels.jsonl", "a") as f:
        f.write(json.dumps({"clip_id": "broken", "verdict": "fail",
                            "classes": ["other"], "ts": 2,
                            "clip_sha256": sha256_file(bad)}) + "\n")
    assert run_tune(str(cal)) == 0
    assert "broken" in (cal / "tuning_report.md").read_text()
    err = [r for r in _read_jsonl(cal / "metrics.jsonl") if r.get("error")]
    assert len(err) == 1 and err[0]["clip_sha256"] == sha256_file(bad)

    calls = []
    monkeypatch.setattr(tune.l1, "compute_metrics",
                        lambda *a, **k: calls.append(a) or {"metrics": BASE_M})
    assert run_tune(str(cal)) == 0
    assert calls == []                                  # error row honored on rerun


# ------------------------------------- [97] label / metric staleness by clip sha

def test_label_for_replaced_clip_bytes_is_excluded_and_reported(tmp_path):
    rows, labels = base_rows_and_labels()
    cal = mk_cal(tmp_path, rows, labels)
    p = cal / "clips" / "p0.mp4"
    p.write_bytes(b"regenerated pixels, nobody labeled these")
    with open(cal / "metrics.jsonl", "a") as f:     # metrics for the new bytes
        f.write(json.dumps({"clip_id": "p0", "clip_sha256": sha256_file(p),
                            "l1_impl": getattr(tune, "L1_IMPL", None),
                            "metrics": BASE_M}) + "\n")
    assert run_tune(str(cal)) == 0
    rep = report_json(cal)
    assert "p0" in rep["excluded"]["stale_labels"]
    assert "p0" not in rep["split"]["tune"] + rep["split"]["test"]
    assert "p0" in (cal / "tuning_report.md").read_text()


def test_metrics_cache_is_keyed_by_clip_bytes(tmp_path, clips):
    cal = tmp_path / "cal"
    cal.mkdir()
    clip = cal / "m1.mp4"
    shutil.copyfile(clips["moving"], clip)
    sentinel = dict(BASE_M, flow_mag_median=-42.0)
    with open(cal / "metrics.jsonl", "w") as f:
        f.write(json.dumps({"clip_id": "m1", "metrics": sentinel}) + "\n")
        f.write(json.dumps({"clip_id": "m1", "clip_sha256": "0" * 64,
                            "l1_impl": getattr(tune, "L1_IMPL", None),
                            "metrics": sentinel}) + "\n")
    manifest = [{"clip_id": "m1", "prompt": "x", "clip_path": str(clip)}]
    got = tune.load_metrics(cal, manifest)
    assert got["m1"]["flow_mag_median"] > 0             # recomputed, not the sentinel


# --------------------------------------------------------- [98] --approve gate

def test_approve_refuses_null_labels_sha_when_labels_file_is_gone(tmp_path, capsys):
    rows, labels = base_rows_and_labels()
    cal = mk_cal(tmp_path, rows, labels)
    assert run_tune(str(cal)) == 0
    p = cal / "thresholds.proposed.yaml"
    data = yaml.safe_load(p.read_text())
    data["provenance"]["labels_sha256"] = None
    p.write_text(yaml.safe_dump(data, sort_keys=False))
    (cal / "labels.jsonl").unlink()
    target = tmp_path / "thresholds.yaml"
    capsys.readouterr()
    assert run_tune(str(cal), approve=True, target=str(target),
                    accept_untested_l2=True) == 2
    assert "labels.jsonl is missing" in capsys.readouterr().out
    assert not target.exists()


def test_same_day_approvals_get_increasing_versions(tmp_path):
    rows, labels, l2 = _l2_fixture()
    cal = mk_cal(tmp_path, rows, labels, l2=l2)
    target = tmp_path / "thresholds.yaml"
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    versions = []
    for _ in range(2):
        assert run_tune(str(cal), target=str(target)) == 0
        assert run_tune(str(cal), approve=True, target=str(target)) == 0
        versions.append(load_thresholds(target)[1]["version"])
    assert versions == [f"{today}.1", f"{today}.2"]


def test_approve_refuses_when_judge_scores_changed_after_tuning(tmp_path):
    rows, labels, l2 = _l2_fixture()
    cal = mk_cal(tmp_path, rows, labels, l2=l2)
    assert run_tune(str(cal)) == 0
    write_l2(cal, {"g00": scores(imaging_quality=1)})
    assert run_tune(str(cal), approve=True, target=str(tmp_path / "t.yaml")) == 2


def test_approve_refuses_hand_edited_values(tmp_path, capsys):
    rows, labels = base_rows_and_labels()
    cal = mk_cal(tmp_path, rows, labels)
    assert run_tune(str(cal)) == 0
    p = cal / "thresholds.proposed.yaml"
    data = yaml.safe_load(p.read_text())
    data["l1"]["motion"]["value"] = 0.0                 # valid shape, untested value
    p.write_text(yaml.safe_dump(data, sort_keys=False))
    assert validate_thresholds({**data, "calibrated": True}) == []
    capsys.readouterr()
    assert run_tune(str(cal), approve=True, target=str(tmp_path / "t.yaml"),
                    accept_untested_l2=True) == 2
    assert "edited after tuning" in capsys.readouterr().out


# ------------------------------------------------ [84] judge identity / escalation

def test_tune_uses_only_the_configured_judges_scores(tmp_path):
    rows, labels, l2 = _l2_fixture()
    cal = mk_cal(tmp_path, rows, labels, l2=l2)
    # a later 32B pass that (say) scores every clip 5 except the deformed ones at 1
    big = "ollama/qwen3-vl:32b"
    write_l2(cal, {c: scores(subject_consistency=1) if c.startswith("d") else
                   scores(**{d: 5 for d in DIMS}) for c in labels},
             model=big, digest="sha256:bigdigest")
    assert run_tune(str(cal), judge={"model": JUDGE, "model_digest": None}) == 0
    assert proposal(cal)["provenance"]["judge"] == {"model": JUDGE,
                                                    "model_digest": DIGEST}
    assert run_tune(str(cal), judge={"model": big, "model_digest": None}) == 0
    assert proposal(cal)["provenance"]["judge"] == {"model": big,
                                                    "model_digest": "sha256:bigdigest"}


def test_approve_refuses_when_config_names_another_judge(tmp_path):
    rows, labels, l2 = _l2_fixture()
    cal = mk_cal(tmp_path, rows, labels, l2=l2)
    assert run_tune(str(cal)) == 0
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({"judge": {"adapter": "ollama",
                                             "model": "qwen3-vl:32b"}}))
    target = tmp_path / "t.yaml"
    assert run_tune(str(cal), approve=True, target=str(target),
                    config_path=str(cfg)) == 2
    cfg.write_text(yaml.safe_dump({"judge": {"adapter": "ollama",
                                             "model": "qwen3-vl:8b-instruct"}}))
    assert run_tune(str(cal), approve=True, target=str(target),
                    config_path=str(cfg)) == 0


# ------------------- [review fix-calib-2] L2 floor search when nothing fits the cap

def _tune_pass_ids(labels):
    tune_ids, _ = tune_test_ids(labels)
    return [c for c in tune_ids if labels[c][0] == "pass"]


def _assert_l2_fallback_is_loud(cal, forced, capsys):
    out = capsys.readouterr().out
    assert "WARNING" in out and "fallback" in out.lower()
    prop = proposal(cal)
    marker = prop["provenance"]["l2_floors_fallback"]
    assert isinstance(marker, str) and marker
    assert all(c in marker for c in forced)
    rep = report_json(cal)
    assert sorted(rep["l2"]["forced_false_fail_ids"]) == sorted(forced)
    # least-bad: the floors add no false-fail beyond the clips the judge forces
    assert rep["l2"]["false_fail_on_tune"] == len(forced)
    assert rep["tune"]["false_fail"] == len(forced)
    md = (cal / "tuning_report.md").read_text()
    assert "FALLBACK" in md and all(c in md for c in forced)
    assert validate_thresholds({**prop, "calibrated": True}) == []


def test_l2_floor_search_falls_back_when_the_judge_fails_publishable_clips(tmp_path,
                                                                           capsys):
    """[agent-diffs#2] An 8B judge scores imaging_quality=1 on more labeled-pass tune
    clips than the false-fail cap allows: no floor combination fits (the loosest
    floor is 2). The tuner must not crash — explicit least-bad fallback, said
    loudly in stdout, report and provenance."""
    rows, labels, l2 = _l2_fixture()
    good = _tune_pass_ids(labels)
    forced = good[:tune._ff_cap(len(good)) + 1]
    for c in forced:
        l2[c] = scores(imaging_quality=1)
    cal = mk_cal(tmp_path, rows, labels, l2=l2)
    assert run_tune(str(cal)) == 0
    _assert_l2_fallback_is_loud(cal, forced, capsys)
    # the deformed clips are still caught where possible (no avoidable miss)
    assert proposal(cal)["l2"]["floors"]["subject_consistency"] >= 3

    target = tmp_path / "thresholds.yaml"
    assert run_tune(str(cal), approve=True, target=str(target)) == 0
    out = capsys.readouterr().out
    assert "WARNING" in out and "fallback" in out.lower()
    approved = yaml.safe_load(target.read_text())
    assert approved["provenance"]["l2_floors_fallback"]       # marker survives sign-off


def test_l2_floor_search_falls_back_on_all_na_publishable_clips(tmp_path, capsys):
    """[integration-seams#3] A labeled-pass clip with every dimension N/A fails
    under every floor combination (runtime rule) — same no-crash fallback."""
    rows, labels, l2 = _l2_fixture()
    good = _tune_pass_ids(labels)
    forced = good[:tune._ff_cap(len(good)) + 1]
    for c in forced:
        l2[c] = {"dims": {d: 4 for d in DIMS}, "na": {d: True for d in DIMS}}
    cal = mk_cal(tmp_path, rows, labels, l2=l2)
    assert run_tune(str(cal)) == 0
    _assert_l2_fallback_is_loud(cal, forced, capsys)


def test_tune_l2_direct_with_a_zero_cap_never_raises():
    """4 labeled-pass clips (cap 0), one scored 1: the reviewer's direct repro."""
    labels, rows = {}, {}
    for i in range(4):
        labels[f"g{i}"] = {"verdict": "pass", "classes": []}
        rows[f"g{i}"] = {"dimensions": {d: 5 for d in DIMS}, "na": {}}
    rows["g0"]["dimensions"]["anatomy_artifacts"] = 1
    for i in range(3):
        labels[f"b{i}"] = {"verdict": "fail", "classes": ["deformed"]}
        rows[f"b{i}"] = {"dimensions": {d: 5 for d in DIMS} | {"subject_consistency": 2},
                         "na": {}}
    floors, notes = tune.tune_l2(sorted(labels), labels, rows)
    assert notes["forced_false_fail_ids"] == ["g0"]
    assert notes["fallback"] and "g0" in notes["fallback"]
    assert notes["false_fail_on_tune"] == 1 and notes["missed_on_tune"] == 0
    assert floors["subject_consistency"] >= 3


def test_l2_floor_search_without_forced_false_fails_has_no_fallback(tmp_path):
    rows, labels, l2 = _l2_fixture()
    cal = mk_cal(tmp_path, rows, labels, l2=l2)
    assert run_tune(str(cal)) == 0
    assert proposal(cal)["provenance"]["l2_floors_fallback"] is None
    assert "FALLBACK" not in (cal / "tuning_report.md").read_text()


# --------------- [review fix-calib-2] --approve: L1-only proposals fail closed

def test_approve_refuses_an_l1_only_proposal_without_the_explicit_exception(tmp_path,
                                                                            capsys):
    """[agent-diffs#1 / docs-cli-truth#4] No judge scores behind the proposal: the
    L2 floors are untuned defaults — an uncalibrated judge is an uncalibrated
    instrument. Refused unless --accept-untested-l2, which leaves a loud marker."""
    rows, labels = base_rows_and_labels()
    cal = mk_cal(tmp_path, rows, labels)
    assert run_tune(str(cal)) == 0
    capsys.readouterr()
    target = tmp_path / "thresholds.yaml"
    assert run_tune(str(cal), approve=True, target=str(target)) == 2
    assert not target.exists()
    assert "--accept-untested-l2" in capsys.readouterr().out

    assert run_tune(str(cal), approve=True, target=str(target),
                    accept_untested_l2=True) == 0
    assert "WARNING" in capsys.readouterr().out
    prov = yaml.safe_load(target.read_text())["provenance"]
    assert prov["test_scope"].startswith("L1 only")          # stays L1-only
    assert prov["l2_untested_accepted"] is True
    assert prov["judge"] is None


def test_approve_cli_needs_the_flag_for_the_32b_swap_without_rescoring(tmp_path):
    """The reviewer's scenario: 8B scores on disk, config switched to a 32B judge,
    `calibrate-tune` re-run WITHOUT --with-l2 (silently L1-only), then --approve.
    Must exit non-zero and write no thresholds.yaml."""
    rows, labels, l2 = _l2_fixture()
    cal = mk_cal(tmp_path, rows, labels, l2=l2)            # 8B scores on disk
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({"judge": {"adapter": "ollama",
                                             "model": "qwen3-vl:32b"}}))
    target = tmp_path / "thresholds.yaml"
    base = ["--calibration", str(cal), "--config", str(cfg), "--target", str(target)]
    assert tune.main_tune(base) == 0
    assert proposal(cal)["provenance"]["judge"] is None
    assert tune.main_tune(base + ["--approve"]) == 2
    assert not target.exists()
    # the flag is a sign-off modifier only
    assert tune.main_tune(base + ["--accept-untested-l2"]) == 2
    assert tune.main_tune(base + ["--approve", "--accept-untested-l2"]) == 0
    assert yaml.safe_load(target.read_text())["provenance"]["l2_untested_accepted"]


def test_judge_backed_approval_carries_no_untested_marker(tmp_path):
    rows, labels, l2 = _l2_fixture()
    cal = mk_cal(tmp_path, rows, labels, l2=l2)
    assert run_tune(str(cal)) == 0
    target = tmp_path / "thresholds.yaml"
    assert run_tune(str(cal), approve=True, target=str(target),
                    accept_untested_l2=True) == 0               # flag is moot here
    prov = yaml.safe_load(target.read_text())["provenance"]
    assert "l2_untested_accepted" not in prov and prov["judge"]["model"] == JUDGE


# ------------------ [review fix-calib-2] --approve: L1 metric code fingerprint

def test_approve_refuses_when_l1_metric_code_changed_since_tuning(tmp_path, monkeypatch,
                                                                  capsys):
    """[agent-diffs#5] Thresholds are values of the metrics l1.py computes: an l1.py
    change between tune and approve makes the proposal describe other metrics."""
    rows, labels, l2 = _l2_fixture()
    cal = mk_cal(tmp_path, rows, labels, l2=l2)
    assert run_tune(str(cal)) == 0
    target = tmp_path / "thresholds.yaml"
    monkeypatch.setattr(tune, "L1_IMPL", "0000deadbeef0000")   # a newer l1.py
    capsys.readouterr()
    assert run_tune(str(cal), approve=True, target=str(target)) == 2
    assert not target.exists()
    assert "l1.py" in capsys.readouterr().out
    monkeypatch.undo()
    assert run_tune(str(cal), approve=True, target=str(target)) == 0


def test_approve_refuses_a_proposal_without_an_l1_fingerprint(tmp_path):
    rows, labels, l2 = _l2_fixture()
    cal = mk_cal(tmp_path, rows, labels, l2=l2)
    assert run_tune(str(cal)) == 0
    p = cal / "thresholds.proposed.yaml"
    data = yaml.safe_load(p.read_text())
    del data["provenance"]["l1_impl"]
    p.write_text(yaml.safe_dump(data, sort_keys=False))
    target = tmp_path / "thresholds.yaml"
    assert run_tune(str(cal), approve=True, target=str(target)) == 2
    assert not target.exists()


# --------------- [review fix-calib-2] attribution: a judge ERROR is not a catch

def test_judge_error_is_not_counted_as_l2_catching_the_clip():
    """[agent-diffs#6] The 'would each layer catch it on its own' table drives the
    8B -> 32B escalation call: a judge that errors must not look like one that
    catches defects."""
    labels = {"d0": {"verdict": "fail", "classes": ["deformed"]},
              "d1": {"verdict": "fail", "classes": ["deformed"]},
              "s0": {"verdict": "fail", "classes": ["static"]},
              "p0": {"verdict": "pass", "classes": []}}
    metrics = {c: dict(BASE_M) for c in labels}
    metrics["s0"] = dict(BASE_M, flow_mag_median=1e-5)
    l1_section = {"motion": {"metric": "flow_mag_median", "op": ">=", "value": 0.001}}
    l2_rows = {"d0": {"error": "l2: judge context overflow"},
               "d1": {"dimensions": {d: 4 for d in DIMS} | {"subject_consistency": 2},
                      "na": {}},
               "s0": {"error": "l2: judge context overflow"},
               "p0": {"dimensions": {d: 4 for d in DIMS}, "na": {}}}
    floors = {d: 3 for d in DIMS}
    ev = tune.evaluate(sorted(labels), labels, metrics, l1_section, l2_rows, floors)
    by_class = ev["attribution"]["by_class"]
    assert by_class["deformed"].get("L2 only", 0) == 1          # d1 only, not d0
    assert by_class["deformed"].get("both", 0) == 0
    assert by_class["deformed"].get(tune.ATTR_L2_ERROR_L1_MISSES) == 1
    assert by_class["static"].get("both", 0) == 0               # s0: L1 caught it
    assert by_class["static"].get(tune.ATTR_L2_ERROR_L1_CATCHES) == 1
    # the runtime decision is unchanged: a judge ERROR never ships
    assert ev["attribution"]["stopped_by"]["L2 error"] == 1
    assert ev["clips"]["d0"]["predicted"] == "ERROR"
    md = "\n".join(tune._attribution_md(ev))
    assert tune.ATTR_L2_ERROR_L1_MISSES in md and tune.ATTR_L2_ERROR_L1_CATCHES in md
