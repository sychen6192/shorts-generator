"""M4 tests: batch generation (against fake ComfyUI), labeling server, tuner."""

from __future__ import annotations

import json
import threading
import urllib.request
from pathlib import Path

import yaml

from fake_comfy import serve_comfy

from shortsloop.calibrate.batchgen import run_batch
from shortsloop.calibrate.prompts import build_slate
from shortsloop.calibrate.tune import run_tune, stratified_split, wilson
from shortsloop.check import load_thresholds
from shortsloop.label import make_server
from shortsloop.state import _read_jsonl

DATA = Path(__file__).parent / "data"


# ------------------------------------------------------------- slate & batch

def test_slate_mix_and_determinism():
    slate = build_slate(40)
    assert len(slate) == 40
    kinds = {k: sum(1 for r in slate if r["kind"] == k)
             for k in ("good", "starved", "anatomy")}
    assert kinds["good"] >= 20 and kinds["starved"] >= 8 and kinds["anatomy"] >= 4
    assert slate == build_slate(40)                       # deterministic
    assert len({r["clip_id"] for r in slate}) == 40


def _batch_config(tmp_path, comfy_host) -> Path:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({
        "comfy": {"host": comfy_host, "workflow_t2v": str(DATA / "test_workflow.json"),
                  "timeout_s": 60},
    }), encoding="utf-8")
    return cfg


def test_batch_generates_manifest_and_swaps(clips, tmp_path):
    import random
    with serve_comfy(fixture_paths=clips) as comfy:
        cfg = _batch_config(tmp_path, comfy.host)
        out = tmp_path / "calibration"
        assert run_batch(str(cfg), str(out), count=5, rng=random.Random(7)) == 0
        rows = _read_jsonl(out / "batch_manifest.jsonl")
        gen = [r for r in rows if r["kind"] != "swap"]
        swaps = [r for r in rows if r["kind"] == "swap"]
        assert len(gen) == 5 and len(swaps) == 4
        for r in gen:
            assert Path(r["clip_path"]).is_file()
            assert r["seed"] is not None and r["width"] == 480
        for s in swaps:
            src = next(g for g in gen if g["clip_id"] == s["source_clip_id"])
            assert s["clip_path"] == src["clip_path"]      # same pixels
            assert s["prompt"] != src["prompt"]            # wrong prompt on purpose
        assert comfy.violations == 0
        subs_before = len(comfy.submissions)

        # resume: nothing regenerated, manifest unchanged
        assert run_batch(str(cfg), str(out), count=5, rng=random.Random(8)) == 0
        assert len(comfy.submissions) == subs_before
        assert len(_read_jsonl(out / "batch_manifest.jsonl")) == 9


def test_batch_dry_run_touches_nothing(tmp_path, capsys):
    assert run_batch("nonexistent-config.yaml", str(tmp_path / "cal"),
                     count=6, dry_run=True) == 0
    assert not (tmp_path / "cal").exists()
    assert "dry run" in capsys.readouterr().out


# ------------------------------------------------------------- labeling tool

def _http(url, payload=None):
    if payload is None:
        with urllib.request.urlopen(url, timeout=10) as r:
            return r.status, r.read()
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.status, r.read()


def _label_env(clips, tmp_path):
    cal = tmp_path / "cal"
    cal.mkdir()
    rows = [
        {"clip_id": "cal001", "kind": "good", "prompt": "a moving cube",
         "clip_path": str(clips["moving"])},
        {"clip_id": "cal002", "kind": "starved", "prompt": "a static vase",
         "clip_path": str(clips["static"])},
    ]
    with open(cal / "batch_manifest.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return cal


def test_label_server_serves_and_records(clips, tmp_path):
    cal = _label_env(clips, tmp_path)
    srv = make_server(cal)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        status, body = _http(base + "/")
        page = body.decode()
        assert status == 200
        assert "cal001" in page and "cal002" in page
        assert '"labeled": []' in page.replace("'", '"') or '"labeled":[]' in page
        assert "kind" not in json.loads(
            page.split("const DATA = ", 1)[1].split(";\n", 1)[0])["rows"][0]

        status, body = _http(base + "/clip/cal001")
        assert status == 200 and len(body) > 1000       # real mp4 bytes

        status, body = _http(base + "/label",
                             {"clip_id": "cal002", "verdict": "fail",
                              "classes": ["static"], "note": "", "ms": 4200})
        assert status == 200 and json.loads(body)["remaining"] == 1
        rec = _read_jsonl(cal / "labels.jsonl")[-1]
        assert rec["clip_id"] == "cal002" and rec["classes"] == ["static"]
        assert rec["clip_sha256"] and rec["ms_spent"] == 4200

        # relabel appends; page now reports it labeled
        _http(base + "/label", {"clip_id": "cal002", "verdict": "pass",
                                "classes": [], "ms": 100})
        assert len(_read_jsonl(cal / "labels.jsonl")) == 2
        _, body = _http(base + "/")
        assert "cal002" in body.decode().split("labeled", 1)[1][:200]
    finally:
        srv.shutdown()
        srv.server_close()


def test_label_server_rejects_bad_payloads(clips, tmp_path):
    cal = _label_env(clips, tmp_path)
    srv = make_server(cal)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        for bad in (
            {"clip_id": "nope", "verdict": "fail", "classes": ["static"]},
            {"clip_id": "cal001", "verdict": "fail", "classes": []},   # fail needs class
            {"clip_id": "cal001", "verdict": "pass", "classes": ["static"]},
            {"clip_id": "cal001", "verdict": "meh", "classes": []},
            {"clip_id": "cal001", "verdict": "fail", "classes": ["weird_class"]},
        ):
            try:
                status, _ = _http(base + "/label", bad)
            except urllib.error.HTTPError as e:
                status = e.code
            assert status == 400, bad
        assert not (cal / "labels.jsonl").exists()
    finally:
        srv.shutdown()
        srv.server_close()


# ------------------------------------------------------------------ tuner

def _mk_cal(tmp_path, rows, labels):
    cal = tmp_path / "cal"
    cal.mkdir()
    with open(cal / "batch_manifest.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(cal / "metrics.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps({"clip_id": r["clip_id"], "metrics": r["_m"]}) + "\n")
    with open(cal / "labels.jsonl", "w") as f:
        for cid, (verdict, classes) in labels.items():
            f.write(json.dumps({"clip_id": cid, "verdict": verdict,
                                "classes": classes, "ts": 1}) + "\n")
    return cal


BASE_M = {"flow_mag_median": 0.003, "flow_mag_p90": 0.004, "ssim_min": 0.7,
          "ssim_p05": 0.8, "ssim_mean": 0.95, "flicker_dips": 0,
          "freeze_longest_run_s": 0.0, "laplacian_p10": 60.0,
          "laplacian_median": 90.0, "luma_mean": 0.4, "black_frame_frac": 0.0,
          "clipped_frac": 0.01}


def _rows_and_labels():
    rows, labels = [], {}
    for i in range(8):                                   # separable pass clips
        m = dict(BASE_M, flow_mag_median=0.003 + i * 0.0004)
        rows.append({"clip_id": f"p{i}", "kind": "good", "prompt": "x",
                     "clip_path": f"/nope/p{i}.mp4", "_m": m})
        labels[f"p{i}"] = ("pass", [])
    for i in range(6):                                   # labeled static, low flow
        m = dict(BASE_M, flow_mag_median=1e-5 * (i + 1),
                 freeze_longest_run_s=2.5)
        rows.append({"clip_id": f"s{i}", "kind": "starved", "prompt": "x",
                     "clip_path": f"/nope/s{i}.mp4", "_m": m})
        labels[f"s{i}"] = ("fail", ["static"])
    for i in range(4):                                   # labeled flicker, many dips
        m = dict(BASE_M, flicker_dips=8 + i, ssim_min=0.35)
        rows.append({"clip_id": f"f{i}", "kind": "good", "prompt": "x",
                     "clip_path": f"/nope/f{i}.mp4", "_m": m})
        labels[f"f{i}"] = ("fail", ["flicker"])
    for i in range(3):                                   # deformed: L2's job, L1-clean
        rows.append({"clip_id": f"d{i}", "kind": "anatomy", "prompt": "x",
                     "clip_path": f"/nope/d{i}.mp4", "_m": dict(BASE_M)})
        labels[f"d{i}"] = ("fail", ["deformed"])
    return rows, labels


def test_tuner_separates_and_reports(tmp_path):
    rows, labels = _rows_and_labels()
    cal = _mk_cal(tmp_path, rows, labels)
    assert run_tune(str(cal)) == 0

    proposed = yaml.safe_load((cal / "thresholds.proposed.yaml").read_text())
    assert proposed["calibrated"] is False               # sign-off gate intact
    motion = proposed["l1"]["motion"]["value"]
    assert 6e-5 < motion < 0.003                          # between the clusters
    flick = proposed["l1"]["flicker"]["value"]
    assert 0 <= flick < 8
    # guard rails never fail a labeled-pass tune clip
    assert proposed["l1"]["sharpness"]["value"] < 60.0
    # provenance carries labels sha + test stats
    prov = proposed["provenance"]
    assert prov["labels_sha256"] and prov["tune_test_split"]

    report = (cal / "tuning_report.md").read_text()
    assert "false-pass" in report
    assert "L2's job" in report                           # deformed clips flagged
    # proposed file loads through the checker's own loader (shape-frozen)
    thr, info = load_thresholds(cal / "thresholds.proposed.yaml")
    assert info["calibrated"] is False and set(thr["l1"]) == {
        "motion", "freeze", "flicker", "ssim_floor", "sharpness", "black", "exposure"}


def test_approve_flips_calibrated_with_provenance(tmp_path):
    rows, labels = _rows_and_labels()
    cal = _mk_cal(tmp_path, rows, labels)
    assert run_tune(str(cal)) == 0
    target = tmp_path / "thresholds.yaml"
    assert run_tune(str(cal), approve=True, target=str(target)) == 0
    thr, info = load_thresholds(target)
    assert info["calibrated"] is True
    data = yaml.safe_load(target.read_text())
    assert data["provenance"]["labels_sha256"]

    # approve refuses if labels changed after tuning (stale proposal)
    with open(cal / "labels.jsonl", "a") as f:
        f.write(json.dumps({"clip_id": "p0", "verdict": "fail",
                            "classes": ["other"], "ts": 2}) + "\n")
    assert run_tune(str(cal), approve=True, target=str(target)) == 2


def test_split_is_stratified_and_deterministic():
    labels = {f"p{i}": {"verdict": "pass", "classes": []} for i in range(10)}
    labels.update({f"s{i}": {"verdict": "fail", "classes": ["static"]}
                   for i in range(6)})
    a = stratified_split(labels)
    b = stratified_split(labels)
    assert a == b
    tune, test = a
    assert set(tune) | set(test) == set(labels) and not set(tune) & set(test)
    # both strata represented in the test set
    assert any(c.startswith("s") for c in test) and any(c.startswith("p") for c in test)


def test_last_label_wins(tmp_path):
    rows, labels = _rows_and_labels()
    cal = _mk_cal(tmp_path, rows, labels)
    # relabel p0 as fail/static afterwards — tuner must see the LAST record
    with open(cal / "labels.jsonl", "a") as f:
        f.write(json.dumps({"clip_id": "p0", "verdict": "fail",
                            "classes": ["static"],
                            "ts": 2}) + "\n")
    from shortsloop.calibrate.tune import last_labels
    assert last_labels(cal)["p0"]["verdict"] == "fail"


def test_wilson_ci_sane():
    lo, hi = wilson(0, 12)
    assert lo == 0.0 and 0.2 < hi < 0.3        # n=12, zero events → CI up to ~24%
    lo2, hi2 = wilson(3, 12)
    assert lo2 > 0.05 and hi2 < 0.6
