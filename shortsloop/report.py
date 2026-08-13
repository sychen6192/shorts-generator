"""Morning report: one markdown page + thumbnails + a machine-readable mirror.

The report is evidence, not narrative: verdicts come from verdict files, budget
numbers from the runner's counters, and every claim links to an artifact on disk.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import cv2

SHEET_FRAMES = 6
SHEET_HEIGHT = 240


def contact_sheet(clip_path: str | Path, out_jpg: str | Path) -> str | None:
    """6-frame horizontal strip for at-a-glance review. Returns path or None."""
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return None
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    if not frames:
        return None
    n = min(SHEET_FRAMES, len(frames))
    idxs = sorted({round(i * (len(frames) - 1) / max(1, n - 1)) for i in range(n)})
    tiles = []
    for idx in idxs:
        f = frames[idx]
        h, w = f.shape[:2]
        tiles.append(cv2.resize(f, (int(w * SHEET_HEIGHT / h), SHEET_HEIGHT)))
    strip = cv2.hconcat(tiles)
    out = Path(out_jpg)
    out.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", strip, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return None
    out.write_bytes(buf.tobytes())
    return str(out)


def _fmt_min(seconds: float | None) -> str:
    return "—" if seconds is None else f"{seconds / 60:.1f} min"


def render(run_dir: Path, run_meta: dict, items: list, budget: dict,
           skipped_rows: list[dict]) -> tuple[Path, Path]:
    """items: runner ClipRun objects (duck-typed: spec, status, attempts_used,
    attempt_history, encoded_path, encode_error, skip_reason)."""
    md: list[str] = []
    status = run_meta.get("status", "COMPLETED")
    md.append(f"# shortsloop nightly report — {run_meta['run_id']}")
    md.append("")
    if status.startswith("HALTED"):
        md.append(f"> ⛔ **{status}** — {run_meta.get('halt_reason', '')}")
        md.append("> No clips were encoded after the halt point. Verdicts below are "
                  "only those produced while the instrument was healthy.")
        md.append("")

    counts = {"passed": 0, "failed_final": 0, "skipped": 0, "error": 0, "pending": 0}
    for it in items:
        counts[it.status if it.status in counts else "pending"] = \
            counts.get(it.status if it.status in counts else "pending", 0) + 1
    total_attempts = sum(it.attempts_used for it in items)

    md.append("## Run summary")
    md.append("")
    md.append("| | |")
    md.append("|---|---|")
    md.append(f"| Status | **{status}** |")
    md.append(f"| Dispatch | `{run_meta.get('dispatch_path', '?')}` "
              f"(sha256 `{str(run_meta.get('dispatch_sha256'))[:12]}…`) |")
    md.append(f"| Clips accepted / skipped at intake | {len(items)} / {len(skipped_rows)} |")
    md.append(f"| Verdicts | ✅ {counts['passed']} passed · ❌ {counts['failed_final']} "
              f"failed (caught & bounded) · ⏭ {counts['skipped']} skipped · "
              f"⚠️ {counts['error']} error |")
    md.append(f"| Attempts used | {total_attempts} "
              f"(cap {budget.get('max_attempts', '?')}/clip — respected: "
              f"{'YES' if budget.get('cap_respected', True) else 'NO'}) |")
    md.append(f"| Waves | {budget.get('waves_run', '?')} / {budget.get('waves_max', '?')} |")
    md.append(f"| Wall clock | {_fmt_min(budget.get('wall_s'))} of "
              f"{_fmt_min(budget.get('wall_budget_s'))} budget — "
              f"{'TRIPPED' if budget.get('wall_tripped') else 'within budget'} |")
    md.append(f"| Generation / judge time | {_fmt_min(budget.get('gen_s'))} / "
              f"{_fmt_min(budget.get('judge_s'))} |")
    md.append(f"| Thresholds | v{run_meta.get('thresholds_version')} "
              f"(calibrated: {run_meta.get('thresholds_calibrated')}) |")
    md.append("")

    if skipped_rows:
        md.append("## Skipped at intake (not silent — v1 scope)")
        md.append("")
        for row in skipped_rows:
            md.append(f"- `{row['clip_id']}` ({row['mode']}"
                      f"{', image dep: ' + row['image_dep'] if row['image_dep'] else ''})"
                      f" — {row['reason']}")
        md.append("")

    by_video: dict[str, list] = {}
    for it in items:
        by_video.setdefault(it.spec.video_id, []).append(it)

    md.append("## Videos")
    md.append("")
    for vid in sorted(by_video):
        group = by_video[vid]
        passed = [i for i in group if i.status == "passed"]
        line = f"- **{vid}**: {len(passed)}/{len(group)} clips passed"
        misses = [i for i in group if i.status != "passed"]
        if misses:
            det = ", ".join(
                f"{i.spec.clip_id} {i.status}"
                + (f" ({'/'.join(i.last_classes)})" if i.last_classes else "")
                + (f" [{i.skip_reason}]" if i.skip_reason else "")
                for i in misses)
            line += f" — {det}"
        md.append(line)
    md.append("")

    md.append("## Clips")
    md.append("")
    for it in sorted(items, key=lambda x: x.spec.clip_id):
        icon = {"passed": "✅", "failed_final": "❌", "skipped": "⏭",
                "error": "⚠️"}.get(it.status, "⏳")
        md.append(f"### {icon} {it.spec.clip_id} — {it.status.upper()}"
                  + (f" ({'/'.join(it.last_classes)})" if it.last_classes and
                     it.status != "passed" else ""))
        md.append("")
        if it.status == "passed" and it.encoded_path:
            enc = Path(it.encoded_path)
            try:
                enc = enc.relative_to(run_dir)
            except ValueError:
                pass
            md.append(f"- encoded: `{enc}`")
        if getattr(it, "encode_error", None):
            md.append(f"- ⚠️ **encode failed after PASS**: {it.encode_error} — clip "
                      f"NOT shipped; raw file kept")
        if it.skip_reason:
            md.append(f"- skipped: {it.skip_reason}")
        if it.attempt_history:
            md.append("")
            md.append("| attempt | seed | steps | outcome | classes | notes |")
            md.append("|---|---|---|---|---|---|")
            for a in it.attempt_history:
                md.append(
                    f"| {a['attempt']} | `{a.get('seed')}` | {a.get('steps') or 4} "
                    f"| {a.get('status')} | {'/'.join(a.get('failure_classes') or []) or '—'} "
                    f"| {a.get('note', '') or a.get('reroll_action', '') or '—'} |")
            md.append("")
            for a in it.attempt_history:
                if a.get("contact_sheet"):
                    rel = Path(a["contact_sheet"]).name
                    md.append(f"![{it.spec.clip_id} a{a['attempt']}](sheets/{rel})")
            for a in it.attempt_history:
                for reason in (a.get("fail_reasons") or [])[:4]:
                    md.append(f"- a{a['attempt']}: {reason}")
        md.append("")

    md.append("---")
    md.append(f"_Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} · "
              f"state: `events.jsonl` / `attempts.jsonl` · every attempt reproducible "
              f"via `shortsloop/vendor/comfy_client.py` with the logged seed+prompt._")

    report_md = run_dir / "report.md"
    report_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    report_json = run_dir / "report.json"
    report_json.write_text(json.dumps({
        "run": run_meta,
        "budget": budget,
        "skipped_at_intake": skipped_rows,
        "clips": [{
            "clip_id": it.spec.clip_id,
            "video_id": it.spec.video_id,
            "status": it.status,
            "skip_reason": it.skip_reason,
            "failure_classes": it.last_classes,
            "attempts": it.attempt_history,
            "encoded": it.encoded_path,
            "encode_error": getattr(it, "encode_error", None),
        } for it in sorted(items, key=lambda x: x.spec.clip_id)],
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report_md, report_json


def main_report(argv: list[str] | None = None) -> int:
    """`shortsloop report RUN_DIR` — re-render report.md from report.json."""
    import argparse

    ap = argparse.ArgumentParser(prog="shortsloop report")
    ap.add_argument("run_dir")
    args = ap.parse_args(argv)
    run_dir = Path(args.run_dir)
    src = run_dir / "report.json"
    if not src.is_file():
        print(f"[shortsloop] no report.json in {run_dir}", flush=True)
        return 2
    data = json.loads(src.read_text(encoding="utf-8"))
    items = []
    for c in data.get("clips", []):
        items.append(SimpleNamespace(
            spec=SimpleNamespace(clip_id=c["clip_id"], video_id=c["video_id"]),
            status=c["status"], skip_reason=c.get("skip_reason"),
            last_classes=c.get("failure_classes") or [],
            attempts_used=len(c.get("attempts") or []),
            attempt_history=c.get("attempts") or [],
            encoded_path=c.get("encoded"), encode_error=c.get("encode_error")))
    md, _ = render(run_dir, data["run"], items, data.get("budget", {}),
                   data.get("skipped_at_intake", []))
    print(f"[shortsloop] re-rendered {md}")
    return 0
