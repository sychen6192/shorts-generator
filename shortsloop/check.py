"""shortsloop-check — the single verdict authority (docs/plan.md §2.1 — FROZEN).

Usage:
    shortsloop-check CLIP.mp4 --prompt-file PROMPT.txt [--json VERDICT.json]
                     [--l1-only] [--thresholds FILE] [--config FILE]
                     [--expect '{"width":720,"height":1280,"fps":16,"frames":81,"duration_s":5.06}']

Exit codes: 0 = PASS (full mode) or PROCEED (--l1-only) · 1 = FAIL · 2 = ERROR.
ERROR is never PASS. --l1-only can never emit PASS: its positive verdict is PROCEED,
and the runner's ship-gate requires a full-mode PASS with layers_run == ["l1","l2"].

The last stdout line is machine-readable:
    VERDICT {"verdict": ..., "failure_classes": [...], "error": ..., "json": ...}
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import yaml

from . import l1, l2
from .errors import CheckError, ClipError, InfraError
from .thresholds import validate_thresholds
from .schema import validate_verdict_file
from .verdict import EXIT_BY_VERDICT, assemble, sha256_file


def load_thresholds(path: str | Path) -> tuple[dict, dict]:
    """Returns (thresholds_dict, thresholds_info). InfraError on any problem —
    a broken/missing thresholds file is a broken instrument."""
    p = Path(path)
    if not p.is_file():
        raise InfraError("l1", f"thresholds file not found: {p}")
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise InfraError("l1", f"thresholds file unparseable: {e}")
    problems = validate_thresholds(data)
    if problems:
        raise InfraError("l1", f"thresholds file {p} is off the frozen shape "
                               f"(plan §2.4): " + "; ".join(problems))
    info = {
        "version": data["version"],
        "calibrated": data["calibrated"] is True,
        "file_sha256": sha256_file(p),
    }
    return data, info


def load_judge_config(path: str | Path | None) -> dict:
    """Judge section of config.yaml. InfraError if unusable — full mode cannot run
    without a judge (fail closed)."""
    candidates = [Path(path)] if path else [Path("config.yaml")]
    for p in candidates:
        if p.is_file():
            try:
                cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError as e:
                raise InfraError("l2", f"config file unparseable: {e}")
            judge = cfg.get("judge")
            if not isinstance(judge, dict):
                raise InfraError("l2", f"config {p} has no judge: section")
            return judge
    raise InfraError(
        "l2",
        "no judge config found (looked for config.yaml) — pass --config or use --l1-only",
    )


def _emit(v: dict, json_path: str | None) -> int:
    """Validate (frozen schema + invariants), write atomically, print the VERDICT
    line. An off-contract verdict is an instrument bug: infra ERROR (fail closed)."""
    problems = validate_verdict_file(v)
    if problems:
        stage = "l2" if "l2" in (v.get("layers_run") or []) else "l1"
        v = {**v, "verdict": "ERROR", "failure_classes": [],
             "error": {"scope": "infra", "stage": stage,
                       "message": "internal: verdict failed schema/invariants: "
                                  + "; ".join(problems)[:2000]}}
        if validate_verdict_file(v):     # the offending block itself is off-schema
            v = {**v, "l1": None, "l2": None, "layers_run": []}
    if json_path:
        out = Path(json_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".tmp")
        # allow_nan=False: the verdict file is strict JSON for every consumer
        tmp.write_text(json.dumps(v, indent=2, ensure_ascii=False, allow_nan=False)
                       + "\n", encoding="utf-8")
        os.replace(tmp, out)             # never a torn verdict file
    print("VERDICT " + json.dumps({
        "verdict": v["verdict"],
        "failure_classes": v["failure_classes"],
        "error": v["error"],
        "json": json_path,
    }, ensure_ascii=False))
    return EXIT_BY_VERDICT[v["verdict"]]


def _emit_crash(e: BaseException) -> int:
    """Last line of defense: the checker itself broke outside evaluation (e.g. the
    verdict could not be written). Exit 2 = ERROR — never 1 (FAIL) or 0."""
    print(f"shortsloop-check: internal failure: {type(e).__name__}: {e}",
          file=sys.stderr)
    print("VERDICT " + json.dumps({
        "verdict": "ERROR", "failure_classes": [],
        "error": {"scope": "infra", "stage": "l1",
                  "message": f"checker crashed: {type(e).__name__}: {e}"[:500]},
        "json": None}, ensure_ascii=False))
    return 2


def main(argv: list[str] | None = None) -> int:
    try:
        return _main(argv)
    except SystemExit:
        raise                    # argparse usage errors already exit 2
    except Exception as e:
        return _emit_crash(e)


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="shortsloop-check",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("clip")
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--json", dest="json_out", default=None)
    ap.add_argument("--l1-only", action="store_true")
    ap.add_argument("--thresholds", default="thresholds.yaml")
    ap.add_argument("--config", default=None)
    ap.add_argument("--expect", default=None,
                    help='JSON dict: width/height/fps/frames/duration_s from the dispatch sheet')
    args = ap.parse_args(argv)

    expect = None
    if args.expect:
        try:
            expect = json.loads(args.expect)
        except json.JSONDecodeError as e:
            ap.error(f"--expect is not valid JSON: {e}")

    clip_path = args.clip
    clip_sha = sha256_file(clip_path)
    layers_run: list[str] = []
    container = None
    prompt_text = None
    l1_block = None
    l2_block = None
    error_obj = None
    thresholds_info = {"version": None, "calibrated": False, "file_sha256": None}
    timing: dict = {"l1_s": None, "l2_s": None}
    stage = "l1"                 # where an unexpected failure is attributed

    try:
        thresholds, thresholds_info = load_thresholds(args.thresholds)

        try:
            prompt_text = Path(args.prompt_file).read_text(encoding="utf-8").strip()
        except OSError as e:
            raise ClipError("probe", f"cannot read prompt file: {e}")

        t0 = time.monotonic()
        stage = "probe"
        container = l1.probe(clip_path)
        stage = "l1"
        metrics = l1.compute_metrics(clip_path, fps_hint=container.get("fps"))
        bad = [k for k, x in metrics["metrics"].items()
               if not isinstance(x, (int, float)) or not math.isfinite(x)]
        if bad:
            raise ClipError("l1", f"non-finite L1 metric(s) {bad} — clip not measurable")
        checks = [l1.spec_check(container, metrics["analysis"]["frames_analyzed"], expect)]
        checks += l1.run_checks(metrics["metrics"], thresholds["l1"])
        l1_block = {
            "analysis": metrics["analysis"],
            "metrics": metrics["metrics"],
            "checks": checks,
            "pass": all(c["pass"] for c in checks),
        }
        layers_run.append("l1")
        timing["l1_s"] = round(time.monotonic() - t0, 3)

        if not args.l1_only and l1_block["pass"]:
            stage = "l2"
            judge_cfg = load_judge_config(args.config)
            floors = thresholds["l2"]["floors"]
            t1 = time.monotonic()
            l2_block, l2_raw = l2.run_l2(clip_path, container, prompt_text,
                                         judge_cfg, floors)
            layers_run.append("l2")
            timing["l2_s"] = round(time.monotonic() - t1, 3)
            if args.json_out:  # raw judge output kept for audit next to the verdict
                raw_path = Path(args.json_out).with_suffix(".l2_raw.json")
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                raw_path.write_text(l2_raw, encoding="utf-8")

    except CheckError as e:
        error_obj = e.as_error_obj()
    except Exception as e:  # unexpected bug in the checker = broken instrument
        error_obj = {"scope": "infra", "stage": stage,
                     "message": f"unexpected checker failure: {type(e).__name__}: {e}"}

    v = assemble(
        clip_path=clip_path,
        clip_sha256=clip_sha,
        container=container,
        prompt_text=prompt_text,
        l1_block=l1_block,
        l2_block=l2_block,
        thresholds_info=thresholds_info,
        layers_run=layers_run,
        l1_only=args.l1_only,
        error=error_obj,
        timing=timing,
    )
    return _emit(v, args.json_out)


if __name__ == "__main__":
    sys.exit(main())
