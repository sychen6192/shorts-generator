"""`shortsloop calibrate-batch` — generate the Phase 0 calibration batch
(docs/plan.md §6 step 1). Draft resolution, sequential, deliberately UNFILTERED:
no L1 gate, no judge — bad clips are the point.

Resume-safe: rows whose clip file already exists are skipped, the manifest is
append-only JSONL. After generation, off-prompt ground truth is manufactured by
prompt-swapping good rows (swap: true rows share the clip file, zero GPU cost).

GPU-touching; UNVERIFIED-ON-GPU until run on the workstation (CLAUDE.md).
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path

import yaml

from ..comfy import ComfyClient, GenerationFailed
from ..errors import InfraError
from ..state import _read_jsonl
from .prompts import build_slate

DRAFT = {"width": 480, "height": 832, "length": 81}
N_SWAPS = 4


def plan_swaps(rows: list[dict], n_swaps: int = N_SWAPS) -> list[dict]:
    """Pair good rows with a DIFFERENT good row's prompt: the clip is fine, the
    evaluation prompt is wrong — labeled off_prompt by construction."""
    good = [r for r in rows if r["kind"] == "good"]
    swaps = []
    for i in range(min(n_swaps, max(0, len(good) - 1))):
        src = good[i]
        other = good[(i + len(good) // 2) % len(good)]
        if other["clip_id"] == src["clip_id"]:
            continue
        swaps.append({
            "clip_id": f"{src['clip_id']}swp",
            "kind": "swap",
            "prompt": other["prompt"],          # evaluation prompt (wrong on purpose)
            "gen_prompt": src["prompt"],        # what actually generated the pixels
            "source_clip_id": src["clip_id"],
            "seed": src.get("seed"),
            "clip_path": src.get("clip_path"),
        })
    return swaps


def run_batch(config_path: str, out_dir: str, count: int = 40,
              dry_run: bool = False, rng: random.Random | None = None,
              comfy: ComfyClient | None = None) -> int:
    rng = rng or random.Random()
    out = Path(out_dir)
    clips_dir = out / "clips"
    manifest_path = out / "batch_manifest.jsonl"

    slate = build_slate(count)
    if dry_run:
        for row in slate:
            print(f"{row['clip_id']}  [{row['kind']:7s}] seed_slot={row['seed_slot']}  "
                  f"{row['prompt'][:80]}…")
        print(f"[calibrate-batch] {len(slate)} rows + {N_SWAPS} prompt-swap rows "
              f"(dry run, nothing generated)")
        return 0

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    comfy_cfg = cfg.get("comfy") or {}
    if comfy is None:
        if not comfy_cfg.get("host") or not comfy_cfg.get("workflow_t2v"):
            print("[calibrate-batch] config.yaml needs comfy.host and comfy.workflow_t2v")
            return 2
        comfy = ComfyClient(host=comfy_cfg["host"], workflow=comfy_cfg["workflow_t2v"],
                            timeout_s=int(comfy_cfg.get("timeout_s", 1800)))

    clips_dir.mkdir(parents=True, exist_ok=True)
    done_ids = {r["clip_id"] for r in _read_jsonl(manifest_path)}
    generated_rows = [r for r in _read_jsonl(manifest_path) if r["kind"] != "swap"]

    def append(row: dict) -> None:
        with open(manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    failures = 0
    for row in slate:
        if row["clip_id"] in done_ids:
            continue
        clip_path = clips_dir / f"{row['clip_id']}.mp4"
        seed = rng.randint(0, 2 ** 48)
        print(f"[calibrate-batch] {row['clip_id']} [{row['kind']}] seed={seed} …",
              flush=True)
        t0 = time.monotonic()
        try:
            sub = comfy.submit(prompt=row["prompt"], seed=seed, **DRAFT)
            result = comfy.wait(sub["prompt_id"], clips_dir)
        except GenerationFailed as e:
            failures += 1
            print(f"[calibrate-batch]   FAILED ({e.kind}) — continuing; "
                  f"re-run to retry this row")
            if failures >= 5:
                print("[calibrate-batch] 5 generation failures — stopping (check server)")
                return 3
            continue
        src = Path(result["files"][0])
        if src.resolve() != clip_path.resolve():
            shutil.move(str(src), clip_path)
        full = {**row, "seed": sub.get("seed", seed), "clip_path": str(clip_path),
                "gen_prompt": row["prompt"], **DRAFT,
                "gen_elapsed_s": round(time.monotonic() - t0, 1)}
        append(full)
        generated_rows.append(full)
        done_ids.add(row["clip_id"])

    for swap in plan_swaps(generated_rows):
        if swap["clip_id"] in done_ids:
            continue
        append(swap)
        done_ids.add(swap["clip_id"])

    total = len(_read_jsonl(manifest_path))
    print(f"[calibrate-batch] manifest: {manifest_path} ({total} rows). "
          f"Next: shortsloop label --calibration {out}")
    return 0


def main_batch(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="shortsloop calibrate-batch")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="calibration")
    ap.add_argument("--count", type=int, default=40)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    try:
        return run_batch(args.config, args.out, args.count, args.dry_run)
    except InfraError as e:
        print(f"[calibrate-batch] INFRA: {e.message}")
        return 3
