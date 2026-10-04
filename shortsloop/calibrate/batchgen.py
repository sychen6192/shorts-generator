"""`shortsloop calibrate-batch` — generate the Phase 0 calibration batch
(docs/plan.md §6 step 1). Draft resolution, sequential, deliberately UNFILTERED:
no L1 gate, no judge — bad clips are the point.

Resume-safe: rows whose clip file already exists are skipped, the manifest is
append-only JSONL. After generation, off-prompt ground truth is manufactured by
prompt-swapping good rows (swap: true rows share the clip file, zero GPU cost).

Hard rules on this GPU path, same as the nightly runner:
- rule 3: before the first Wan job, the judge VLM + rewrite LLM are unloaded,
  ComfyUI is /free'd and free VRAM is VERIFIED (gpu.to_generation).
- rule 4: the ComfyUI queue must be empty before every submit; a failed or
  interrupted wait (incl. Ctrl-C) removes our job before anything else happens.

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

from .. import gpu
from ..comfy import ComfyClient, GenerationFailed
from ..errors import InfraError
from ..rewrite import effective_rewrite_cfg
from ..state import _read_jsonl
from .prompts import build_slate

DRAFT = {"width": 480, "height": 832, "length": 81}
N_SWAPS = 4


def load_policies(pipeline_path: str | Path) -> dict:
    """pipeline.yaml policies merged over the runner's defaults — exactly how the
    nightly runner reads them, so calibration uses the same VRAM/queue limits."""
    from ..runner import DEFAULT_POLICIES
    pol = json.loads(json.dumps(DEFAULT_POLICIES))  # deep copy
    p = Path(pipeline_path)
    if p.is_file():
        loaded = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        for key, val in (loaded.get("policies") or {}).items():
            if isinstance(val, dict) and isinstance(pol.get(key), dict):
                pol[key].update(val)
            else:
                pol[key] = val
    return pol


def plan_swaps(rows: list[dict], n_swaps: int = N_SWAPS) -> list[dict]:
    """Pair good rows with a DIFFERENT good PROMPT: the clip is fine, the
    evaluation prompt is wrong — labeled off_prompt by construction.

    Partners are chosen by prompt, not by clip: the slate repeats each prompt
    with a second seed, so a clip-based partner can be the same prompt."""
    good = [r for r in rows if r["kind"] == "good"]
    prompts: list[str] = []
    sources: list[dict] = []
    for r in good:                       # one source row per distinct prompt
        if r["prompt"] not in prompts:
            prompts.append(r["prompt"])
            sources.append(r)
    if len(prompts) < 2:
        return []
    swaps = []
    for k, src in enumerate(sources[:n_swaps]):
        other = prompts[(k + len(prompts) // 2) % len(prompts)]
        if other == src["prompt"]:       # unreachable for len >= 2; never emit one
            continue
        swaps.append({
            "clip_id": f"{src['clip_id']}swp",
            "kind": "swap",
            "prompt": other,                    # evaluation prompt (wrong on purpose)
            "gen_prompt": src["prompt"],        # what actually generated the pixels
            "source_clip_id": src["clip_id"],
            "seed": src.get("seed"),
            "clip_path": src.get("clip_path"),
        })
    return swaps


def run_batch(config_path: str, out_dir: str, count: int = 40,
              dry_run: bool = False, rng: random.Random | None = None,
              comfy: ComfyClient | None = None,
              pipeline_path: str = "pipeline.yaml") -> int:
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

    if not Path(config_path).is_file():
        print(f"[calibrate-batch] config not found: {config_path}")
        return 2
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    comfy_cfg = cfg.get("comfy") or {}
    pol = load_policies(pipeline_path)
    if comfy is None:
        if not comfy_cfg.get("host") or not comfy_cfg.get("workflow_t2v"):
            print("[calibrate-batch] config.yaml needs comfy.host and comfy.workflow_t2v")
            return 2
        comfy = ComfyClient(host=str(comfy_cfg["host"]),
                            workflow=comfy_cfg["workflow_t2v"],
                            timeout_s=pol["comfy"]["timeout_s"],
                            poll_s=pol["comfy"]["poll_s"])
    queue_timeout = float(pol["comfy"]["timeout_s"])
    handoff = pol["vram_handoff"]

    clips_dir.mkdir(parents=True, exist_ok=True)
    done_ids = {r["clip_id"] for r in _read_jsonl(manifest_path)}
    generated_rows = [r for r in _read_jsonl(manifest_path) if r["kind"] != "swap"]

    def append(row: dict) -> None:
        with open(manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    failures = 0
    handed_off = False
    inflight: str | None = None
    try:
        for row in slate:
            if row["clip_id"] in done_ids:
                continue
            if not handed_off:
                # hard rule 3: no language model may sit in VRAM while Wan loads
                free_gb, notes = gpu.to_generation(
                    comfy, cfg.get("judge"), effective_rewrite_cfg(cfg),
                    gpu.gen_free_min_gb(handoff), handoff["wait_timeout_s"])
                print(f"[calibrate-batch] VRAM verified free for Wan: {free_gb:.1f} GB"
                      + (f" ({'; '.join(notes)})" if notes else ""), flush=True)
                handed_off = True
            clip_path = (clips_dir / f"{row['clip_id']}.mp4").resolve()
            seed = rng.randint(0, 2 ** 48)
            q = comfy.queue_state()
            if q["running"] or q["pending"]:
                print(f"[calibrate-batch] ComfyUI queue busy ({len(q['running'])} "
                      f"running, {len(q['pending'])} pending) — waiting up to "
                      f"{queue_timeout:.0f}s before submitting (hard rule 4)", flush=True)
            comfy.wait_queue_empty(timeout_s=queue_timeout)
            print(f"[calibrate-batch] {row['clip_id']} [{row['kind']}] seed={seed} …",
                  flush=True)
            t0 = time.monotonic()
            sub = comfy.submit(prompt=row["prompt"], seed=seed, **DRAFT)
            inflight = sub["prompt_id"]
            try:
                result = comfy.wait(inflight, clips_dir)
            except GenerationFailed as e:
                comfy.ensure_gone(inflight)      # never leave it running (rule 4)
                inflight = None
                failures += 1
                print(f"[calibrate-batch]   FAILED ({e.kind}) — continuing; "
                      f"re-run to retry this row")
                if failures >= 5:
                    print("[calibrate-batch] 5 generation failures — stopping "
                          "(check server)")
                    return 3
                continue
            except InfraError:
                try:
                    comfy.ensure_gone(inflight)
                except InfraError:
                    pass
                inflight = None
                raise
            inflight = None
            src = Path(result["files"][0])
            if src.resolve() != clip_path:
                shutil.move(str(src), clip_path)
            full = {**row, "seed": sub.get("seed", seed), "clip_path": str(clip_path),
                    "gen_prompt": row["prompt"], **DRAFT,
                    "gen_elapsed_s": round(time.monotonic() - t0, 1),
                    "comfy_prompt_id": sub["prompt_id"]}
            append(full)
            generated_rows.append(full)
            done_ids.add(row["clip_id"])
    except KeyboardInterrupt:
        if inflight:
            print(f"\n[calibrate-batch] interrupted — stopping ComfyUI job {inflight} "
                  f"so a re-run never queues a second job beside it", flush=True)
            try:
                comfy.ensure_gone(inflight)
            except InfraError as e:
                print(f"[calibrate-batch] INFRA: {e.message}")
                return 3
        print("[calibrate-batch] interrupted — re-run the same command to resume")
        return 130
    except InfraError as e:
        print(f"[calibrate-batch] INFRA: {e.message}")
        return 3

    for swap in plan_swaps(generated_rows):
        if swap["clip_id"] in done_ids:
            continue
        if swap["prompt"] == swap["gen_prompt"]:     # never fake an off-prompt row
            print(f"[calibrate-batch] skipping {swap['clip_id']}: swap prompt equals "
                  f"its generation prompt")
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
    ap.add_argument("--pipeline", default="pipeline.yaml")
    ap.add_argument("--out", default="calibration")
    ap.add_argument("--count", type=int, default=40)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    try:
        return run_batch(args.config, args.out, args.count, args.dry_run,
                         pipeline_path=args.pipeline)
    except InfraError as e:
        print(f"[calibrate-batch] INFRA: {e.message}")
        return 3
