# shortsloop — rules for anyone (human or agent) working in this repo

## Hard rules — non-negotiable, from the project owner

1. **The verdict comes from the checker's exit code. Nothing else.** No stage may
   declare a clip good based on its own reading, its own summary, or the absence of an
   error. The model generates; the pipeline decides. Only a `shortsloop-check` verdict
   with `verdict == "PASS"` and `layers_run == ["l1","l2"]` can ship a clip.
2. **Fail closed.** Missing verdict, unparseable judge output, timeout, crash — all
   FAIL or ERROR, never PASS. Every such path has a test proving it cannot ship a clip.
3. **VRAM is time-separated.** Wan and the VLM never load simultaneously. ComfyUI's
   models are freed (`/free`) and the free VRAM is *verified* (`/system_stats`) before
   judging; the judge unloads after its wave. Enforced in code, not in a comment.
4. **One ComfyUI job at a time.** Never submit in parallel.
5. **Bounded work.** Per-clip retry cap (default 2 re-rolls = 3 attempts) and a
   whole-run wall-clock budget, both configurable. When either trips, stop cleanly and
   say so in the report — never quietly keep burning the night.
6. **Every attempt is logged** — clip id, seed, prompt, verdict, metrics, duration —
   as append-only JSONL. Any clip must be exactly reproducible from that log.

## Frozen contracts

`docs/plan.md` §2 freezes: verdict JSON schema v1, checker exit codes (0 PASS / 1 FAIL
/ 2 ERROR; `--l1-only` says PROCEED, never PASS), failure classes + re-roll table,
`thresholds.yaml` shape, dispatch intake contract, stage names
(`schedule claim generate verify persist complete`), disk layout. Do not drift from
them without updating the plan first.

## Generation parameters that must never be changed by re-roll logic

- `cfg` stays **1.0** whenever a lightx2v/distill LoRA is in the workflow (raising it
  burns the image — see the comfyui-ig-video skill's `wan22_notes.md`).
- `shift` is never touched.
- The workflow's built-in official negative prompt is never overridden.
- Allowed re-roll knobs: seed, positive-prompt motion append (fixed bank in
  `shortsloop/policy.py`), one logged off-prompt rewrite, `--steps 4→8`.

## Environment truth

- Development happens in a cloud session with **no GPU**: everything CPU-verifiable is
  tested here (synthetic fixtures, fake ComfyUI/judge servers). Anything GPU-touching
  is **UNVERIFIED-ON-GPU** until `shortsloop doctor` and Phase 0 run on the `llm`
  workstation. Do not claim GPU behavior is verified from this environment.
- The runner refuses unattended runs while `thresholds.yaml` has `calibrated: false`
  (an uncalibrated judge is an uncalibrated instrument).
- Vendored scripts in `shortsloop/vendor/` (`comfy_client.py`, `ig_encode.sh`) come
  from the `comfyui-ig-video` skill — do not edit them here; fix upstream instead.

## Development

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -q          # must be green before any commit
```

Test-first: a behavior exists when its test exists. Dependencies are frozen to
`numpy`, `opencv-python-headless`, `PyYAML` (+ pytest for dev) — no new runtime deps
without a plan change. Nothing is "done" from inspecting a video by eye; every claim
traces to a command.
