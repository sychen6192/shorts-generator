# shortsloop v1 — implementation plan

Status: **approved 2026-08-12 · M1–M5 implemented and CPU-verified (79 tests).**
The verdict schema, exit codes, failure classes, stage names, and file layouts in
this document are **frozen** — implementation may not drift from them without
coming back here first. Next: the workstation phase (`docs/runbook.md`) — doctor,
Phase 0 calibration (two user gates), first supervised nightly, DoD checklist.

Companion: `docs/brainstorm.md` (discovery record + argued design positions D1–D10).

## 0. Decisions locked with the user (2026-08-12)

1. **Environment**: design + code + CPU-verifiable tests in the cloud session on this
   branch; GPU verification via `shortsloop doctor` + Phase 0 on the `llm` workstation.
2. **Scheduling**: wave-based — inline L1 gate during the generation wave, batched VLM
   judging per wave, verified VRAM handoff at wave boundaries (D2).
3. **Checker amendments all adopted**: L1 owns flicker, L2 `temporal_coherence` is
   semantic continuity, `subject_consistency` has an N/A path (D5); per-dimension
   floors, no weighted composite (D6); ERROR split into clip-scope vs infra-scope (D7).
4. **Nightly resolution**: 720x1280 final, 81 frames @ 16 fps (dispatch-skill default
   for unattended runs). Calibration batch stays 480x832 draft; thresholds transfer via
   the canonical analysis resolution (D3).

## 1. System overview

```
dispatch.md ──intake──► work items ──┐
                                     ▼
              ┌────────────── wave loop (≤ max_attempts waves) ─────────────┐
              │ GENERATION WAVE (Wan loaded, one job at a time)             │
              │   per pending attempt: claim → generate → L1 gate (CPU)     │
              │   L1 FAIL/broken → classify → re-roll immediately if        │
              │   attempts remain (stays inside this wave)                  │
              │ VRAM HANDOFF: comfy /free → poll /system_stats until free   │
              │ JUDGE WAVE (VLM loaded)                                     │
              │   per L1-surviving attempt: full shortsloop-check (L1+L2)   │
              │   → PASS / FAIL(classes) / ERROR(scope)                     │
              │ VLM unload → classify fails → schedule next wave's re-rolls │
              └─────────────────────────────────────────────────────────────┘
                                     ▼
        persist: encode PASS clips (silent 1080x1920) + contact sheets
        complete: report.md / report.json; exit
```

- Budgets checked at every claim: wall-clock, per-clip attempt cap, disk floor.
  Budget trip ⇒ no new generation; everything already generated is still judged,
  encoded, and reported; skipped work is listed as `SKIPPED(budget)`.
- Every state transition appends to `events.jsonl`; every generation attempt appends
  to `attempts.jsonl`. `--resume` folds events and continues (re-attaches to an
  in-flight ComfyUI job via recorded `prompt_id`).
- The runner never interprets video content. The **only** quality authority is
  `shortsloop-check`'s exit code + verdict JSON (hard rule 1).

## 2. Frozen contracts

### 2.1 Checker CLI

```
shortsloop-check CLIP.mp4 --prompt-file PROMPT.txt --json VERDICT.json
                 [--l1-only] [--thresholds FILE] [--config FILE]
```

- Exit **0 = PASS**, **1 = FAIL**, **2 = ERROR** (could not evaluate).
- Full mode runs L1 then L2 (L2 skipped if L1 fails → verdict FAIL, `l2: null`).
- `--l1-only` verdict vocabulary is `PROCEED / FAIL / ERROR` (exit 0/1/2). It can
  **never** emit `PASS`; a PROCEED verdict is not shippable by construction.
- Ship-gate (in the runner, tested): a clip may be encoded/shipped only from a verdict
  file with `verdict == "PASS"` AND `layers_run == ["l1","l2"]` AND
  `thresholds.calibrated == true`.
- Standalone hand use works with nothing but the clip + prompt file (+ reachable judge
  for full mode).

### 2.2 Verdict JSON schema v1 (FROZEN)

Top-level fields (all always present unless noted):

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | `"1"` | bump ⇒ new schema doc |
| `verdict` | `"PASS" \| "FAIL" \| "ERROR" \| "PROCEED"` | `PROCEED` only in `--l1-only` mode |
| `failure_classes` | `string[]` | ordered by priority (2.3); `[]` unless FAIL |
| `error` | `null \| {scope, stage, message}` | `scope ∈ clip\|infra`, `stage ∈ probe\|l1\|l2`; non-null iff ERROR |
| `clip` | object | `path`, `sha256`, `container: {duration_s, width, height, fps, nb_frames, vcodec}` (from ffprobe) |
| `prompt` | object | `text`, `sha256` |
| `l1` | object | see below; `null` only if ERROR at probe stage |
| `l2` | object \| null | null when L1 failed, `--l1-only`, or ERROR before L2 |
| `thresholds` | object | `version`, `calibrated` (bool), `file_sha256` |
| `layers_run` | `string[]` | subset of `["l1","l2"]` |
| `timing` | object | `l1_s`, `l2_s` (null if not run) |
| `versions` | object | `checker`, `ffprobe`, `opencv`, `numpy`, python |

`l1` object:

- `analysis`: `{width: 480, height: 832, flow_downscale: 2, frames_analyzed}` —
  metrics are always computed at this canonical resolution regardless of input size.
- `metrics` (all numeric, all always present):
  `flow_mag_median`, `flow_mag_p90` (mean Farneback magnitude per frame-pair,
  normalized by analysis-frame diagonal; aggregated over pairs),
  `ssim_min`, `ssim_p05`, `ssim_mean`, `flicker_dips` (count of pairs whose SSIM drops
  ≥ `flicker_delta` below the rolling median of neighbors),
  `freeze_longest_run_s` (longest run of consecutive pairs with SSIM ≥ 0.995 and flow
  ≈ 0), `laplacian_p10`, `laplacian_median` (per-frame Laplacian variance),
  `luma_mean`, `black_frame_frac` (frames with mean luma < 16/255),
  `clipped_frac` (fraction of pixels ≥ 250/255 across frames).
- `checks`: array of `{name, metric, op, threshold, value, pass, reason}` — one per
  threshold in `thresholds.yaml` §2.4, plus the `spec` check (container sanity:
  decodable, ≥ 2 frames, duration within ±15% of dispatch expectation, resolution
  matches dispatch ±0, fps within ±1).
- `pass`: bool (AND of checks).

`l2` object:

- `model` (e.g. `"ollama/qwen3-vl:8b-instruct"`), `model_digest`, `temperature`,
  `seed`, `frames`: `[{index, t}]` (8 uniform incl. first + last),
  `raw_response_sha256` (raw text stored alongside verdict as
  `<verdict>.l2_raw.json`).
- `dimensions`: exactly five keys — `prompt_adherence`, `subject_consistency`,
  `anatomy_artifacts`, `temporal_coherence`, `imaging_quality` — each
  `{score: 1..5, floor, na: bool, pass, reason}`. The judge model returns only
  `score/na/reason`; floors and pass are applied by the checker (the model never
  decides pass/fail). `na` is honored only where the rubric allows it
  (`subject_consistency`); an N/A dimension is excluded from the decision rule.
- `pass`: bool — every non-N/A dimension has `score ≥ floor`.

Dimension definitions given to the judge (anchored 1–5 rubrics; 5 = flawless):

- `prompt_adherence` — are the prompt's subject, action, camera move, and setting
  visible in the frames? Missing named elements must be cited in `reason`.
- `subject_consistency` — same subject identity across frames (color, shape, count,
  wardrobe). `na: true` allowed iff no persistent subject exists (landscape/abstract).
- `anatomy_artifacts` — hands, limbs, faces (should not exist per channel rules),
  melting geometry, duplicated/fused parts, garbled text.
- `temporal_coherence` — *semantic* continuity across the sampled sequence:
  teleporting subjects, object count changes, scene morphing, impossible motion.
  Explicitly NOT frame-to-frame flicker (L1's job — the judge is told this).
- `imaging_quality` — blur, banding, blown exposure, heavy compression artifacts.

### 2.3 Failure classes and re-roll policy (FROZEN)

Class priority (first match drives the re-roll action):
`broken > black > static > deformed > flicker > off_prompt > blurry`.

| Class | Triggered by | Re-roll action (attempt 2) | Attempt 3 |
|---|---|---|---|
| `broken` | spec check fail / undecodable (clip-scope ERROR counts here for retry purposes) | new seed | new seed |
| `black` | `black_frame_frac` / `clipped_frac` checks | new seed | new seed |
| `static` | `motion` / `freeze` checks | new seed + append motion phrase (bank §2.6, deterministic pick by attempt#, logged verbatim) | + `--steps 8` (4-step → 8-step lightx2v) |
| `deformed` | `anatomy_artifacts` or `subject_consistency` floor | new seed | new seed |
| `flicker` | `flicker_dips` / `ssim_floor` checks or `temporal_coherence` floor | new seed + `--steps 8` | same |
| `off_prompt` | `prompt_adherence` floor | one prompt rewrite by local text LLM (anchors verbatim, recipe order, ≤110 words; old/new + unified diff logged in attempts.jsonl); new seed | plain re-roll, new seed, rewritten prompt kept |
| `blurry` | `sharpness` check or `imaging_quality` floor | new seed | new seed |

Never modified: `cfg` (stays 1.0 — distill LoRA), `shift`, the workflow's built-in
official negative prompt, resolution, length. Every patch arg actually sent is logged.

### 2.4 `thresholds.yaml` (FROZEN shape; values TBD in Phase 0)

```yaml
version: "YYYY-MM-DD.N"
calibrated: false            # runner refuses unattended runs while false
provenance:
  labels_sha256: null        # sha256 of labels.jsonl used for tuning
  tuned_at: null
  tune_test_split: null      # e.g. "28/12 stratified"
  test_agreement: null       # fraction
  test_false_pass: null      # fraction + CI string
l1:
  motion:     {metric: flow_mag_median,      op: ">=", value: PLACEHOLDER}
  freeze:     {metric: freeze_longest_run_s, op: "<=", value: PLACEHOLDER}
  flicker:    {metric: flicker_dips,         op: "<=", value: PLACEHOLDER}
  ssim_floor: {metric: ssim_min,             op: ">=", value: PLACEHOLDER}
  sharpness:  {metric: laplacian_p10,        op: ">=", value: PLACEHOLDER}
  black:      {metric: black_frame_frac,     op: "<=", value: PLACEHOLDER}
  exposure:   {metric: clipped_frac,         op: "<=", value: PLACEHOLDER}
l2:
  floors:
    prompt_adherence: 3
    subject_consistency: 3
    anatomy_artifacts: 3
    temporal_coherence: 3
    imaging_quality: 3
```

### 2.5 Dispatch sheet intake contract

Input: a filled `shorts-trend-dispatch` markdown sheet (the only accepted format).
Parsed elements:

1. **Manifest table** (`## 生產清單` section): columns ID / 模式 / image 依賴 / 輸出.
   Rows may use ranges (`V1C1–V1C6`) — expanded. v1 accepts rows with 模式 `T2V` and
   empty image 依賴; every other row → `SKIPPED(v2_mode)` in the report (loud, never
   silent).
2. **Per-clip prompts**: `### V{n}C{k} — …` headings; the fenced code block under the
   heading is the prompt, verbatim.
3. **共用參數 table**: `解析度`, `length / fps` parsed as expected-spec inputs for the
   L1 spec check and generation patch args; absent/unparseable values fall back to
   config defaults (and the fallback is logged).

Intake validation (before any GPU work): every accepted manifest row must have a
non-empty prompt; at least one accepted row must exist; otherwise the run refuses to
start with a per-row error listing. Fail closed at 22:00, not at 3 a.m.

### 2.6 Motion-strengthening phrase bank (static re-rolls)

Fixed, versioned list in `shortsloop/policy.py`; pick = `bank[attempt_index % len]`;
appended to the prompt end before the trailing `vertical 9:16 composition` phrase if
present, else at end. Initial bank (from wan22_notes motion guidance — over-specify
action AND camera move):

1. `; continuous visible motion, slow dolly-in, subject in constant movement`
2. `; dynamic action throughout, camera slowly orbiting the subject`
3. `; sweeping camera movement, handheld tracking shot, energetic motion`

### 2.7 State on disk

```
runs/<run_id>/                      # run_id = YYYYMMDD-HHMMSS-<slug>
  run.json                          # config snapshot, dispatch path+sha, doctor snapshot ref
  events.jsonl                      # append-only state transitions (schema below)
  attempts.jsonl                    # append-only, one line per generation attempt
  dispatch.md                       # verbatim copy of the input sheet
  clips/<clip_id>_a<n>.mp4          # raw ComfyUI outputs
  verdicts/<clip_id>_a<n>.json      # checker verdicts (+ .l2_raw.json beside)
  sheets/<clip_id>_a<n>.jpg         # 6-frame contact sheet per attempt
  encoded/<clip_id>.mp4             # silent 1080x1920 QC encodes (PASS clips only)
  report.md / report.json
```

`events.jsonl` line: `{ts, run_id, clip_id, attempt, stage, event, data}` where
`stage ∈ {schedule, claim, generate, verify, persist, complete}` (exactly the manifest
vocabulary) and `event ∈ {enter, ok, fail, error, skip}`.

`attempts.jsonl` line: `{ts, clip_id, attempt, seed, prompt_text, prompt_sha256,
prompt_rewritten: bool, prompt_diff, workflow_path, workflow_sha256, patch_args,
comfy_prompt_id, output_path, output_sha256, gen_elapsed_s, verdict_path, verdict,
failure_classes, l1_pass, l2_pass, vram_free_before_gb, notes}`.

Reproducibility check: `comfy_client.py run -w <workflow> --prompt "<prompt_text>"
--seed <seed> --width … --height … --length … [--steps …]` from one attempts.jsonl
line regenerates the clip.

### 2.8 `pipeline.yaml` (declarative manifest for future engine migration)

```yaml
pipeline: shortsloop-nightly
stages: [schedule, claim, generate, verify, persist, complete]
policies:
  max_attempts_per_clip: 3
  wall_clock_budget_h: 6
  disk_min_free_gb: 20
  waves_max: 3                      # = max_attempts
  comfy: {timeout_s: 1800, poll_s: 5, one_job_at_a_time: true}
  judge: {timeout_s: 300, retries: 1, infra_escalation_after: 2}
  vram_handoff: {free_min_gb: 24, wait_timeout_s: 180}
resources:
  comfy_host: ${COMFY_HOST}
  workflow_t2v: config              # resolved from config.yaml
  judge: config
```

v1's runner interprets this file; a workflow engine later binds the same stage names.
Machine-local values (hosts, model names, paths) live in `config.yaml` (gitignored;
skeleton written by `doctor`).

## 3. Components & repo layout

```
shorts-generator/
  CLAUDE.md                         # hard rules (verbatim from kickoff) + env notes
  pyproject.toml                    # package shortsloop; console scripts
  pipeline.yaml
  thresholds.yaml                   # calibrated:false placeholder until Phase 0
  config.example.yaml
  docs/{brainstorm.md, plan.md, runbook.md}
  shortsloop/
    __init__.py
    check.py                        # shortsloop-check CLI (L1+L2 orchestration)
    l1.py                           # ffprobe spec + one-pass OpenCV metrics
    l2.py                           # frame sampling, rubric prompt, floors
    judge/{base.py, ollama.py, openai_compat.py}
    dispatch.py                     # sheet parser + intake validation
    runner.py                       # wave loop, stages, budgets, resume
    policy.py                       # failure classes, re-roll table, phrase bank
    rewrite.py                      # off-prompt rewrite via local text LLM (logged)
    encode.py                       # ig_encode.sh wrapper + spec re-verify
    report.py                       # report.md/json + contact sheets
    label.py                        # Phase 0 labeling web tool
    calibrate/{batchgen.py, tune.py}# calibration batch script + threshold tuner
    doctor.py                       # workstation environment verification
    state.py                        # events/attempts JSONL, event-fold, verdict cache
    vendor/{comfy_client.py, ig_encode.sh, VENDOR.md}   # vendored unchanged from skill
  tests/                            # pytest; fixtures synthesized, servers faked
```

- Console scripts: `shortsloop` (subcommands `check run label doctor report
  calibrate-batch calibrate-tune`) plus alias `shortsloop-check` (kickoff contract).
- Dependencies: `numpy`, `opencv-python-headless`, `pyyaml` (+ `pytest` dev). SSIM
  implemented in-repo on grayscale (deterministic, ~30 lines). HTTP via stdlib urllib
  (matches vendored client style). No database, no framework.
- Vendored scripts are called via subprocess with `COMFY_HOST` set; their printed
  `RESULT {...}` line / exit codes are the integration contract. Tests point
  `COMFY_HOST` at a local fake ComfyUI HTTP server (stdlib) — this tests our runner
  AND the vendored client together.
- Judge adapters: Ollama native (`/api/chat`, images, `format:` JSON-schema,
  `options: {temperature: 0, seed: 7, num_ctx: 16384}`, `keep_alive` managed) and
  OpenAI-compatible (`/v1/chat/completions`, for llama.cpp `llama-server`/vLLM).
  Local JSON-schema validation happens regardless of what the server claims.
- Judge verdict cache key: `(clip_sha256, prompt_sha256, thresholds_version,
  model_digest)` — resume never re-judges an already-judged artifact.

## 4. VRAM handoff enforcement (hard rule 3, in code)

Wave boundary, generation → judge:
1. `POST /free {"unload_models": true, "free_memory": true}` to ComfyUI.
2. Poll `/system_stats` until `vram_free ≥ vram_handoff.free_min_gb` or
   `wait_timeout_s` → on timeout: **infra ERROR, halt** (models didn't unload; judging
   would OOM or swap-thrash).
3. Judge wave runs; last judge call (or explicit unload request) sets
   `keep_alive: 0` so the VLM unloads.
4. Next generation wave begins (ComfyUI reloads Wan lazily on first job).

Tests: fake ComfyUI asserts `/free` called before any judge call; fake stats report
low VRAM → runner halts with infra ERROR and no clip ships.

## 5. Test plan (fail-closed matrix is the core deliverable)

Every hard rule gets at least one test that proves the bad path cannot ship a clip:

| # | Scenario | Expected |
|---|---|---|
| 1 | Judge endpoint unreachable | check exit 2 `infra`; runner halts; report says why; 0 clips shipped |
| 2 | Judge returns schema-invalid JSON (×2, after 1 retry) | exit 2 `clip`; attempt consumed; never PASS |
| 3 | Judge timeout | retry once → exit 2 `clip`; 2 consecutive ⇒ infra halt |
| 4 | Truncated/corrupt MP4 | exit 2 `clip` at probe; attempt consumed; re-roll |
| 5 | Checker process killed / missing verdict file | runner treats as ERROR; never ships |
| 6 | L1-only PROCEED verdict presented to ship-gate | rejected (layers_run guard) |
| 7 | `thresholds.calibrated: false` | unattended `run` refuses to start |
| 8 | VRAM not freed (fake stats) | infra halt before judge wave |
| 9 | Wall-clock budget trips mid-run | no new generation; existing clips judged+reported; `SKIPPED(budget)` rows present |
| 10 | Retry cap | exactly 3 attempts then `failed_final`; wave count ≤ 3 |
| 11 | ComfyUI execution error / OOM message | attempt failed; `/free` called; re-roll within cap |
| 12 | Concurrent submit attempt | fake ComfyUI asserts ≤1 in-flight job ever |
| 13 | Dispatch: I2V rows, missing prompt, empty sheet | skipped loudly / intake refusal with per-row errors |
| 14 | Resume after crash (post-generate, pre-verdict) | re-attaches via prompt_id / re-checks; no duplicate generation; verdict cache honored |
| 15 | L1 metrics on synthetic fixtures (static / moving / flicker / black / blur / freeze-tail pairs) | each metric separates its pair with wide margin (signal direction, not exact values) |
| 16 | Encode wrapper output | ffprobe-verified 1080x1920/30fps/h264+aac, audio silent |
| 17 | Every emitted verdict | validates against the frozen JSON Schema; golden files |

Fixtures are synthesized with OpenCV/numpy (moving box, alternating luma, noise
fields) — no GPU, no network; the entire matrix runs in this cloud session.

## 6. Phase 0 calibration (workstation; two user gates)

As specified in `docs/brainstorm.md` §5, now binding:

1. `shortsloop calibrate-batch` generates ~40 clips @ 480x832 (~30–40 min GPU):
   ~24 prompts × 1–2 seeds — format-recipe positives, motion-starved prompts,
   anatomy-stress prompts, plus off-prompt ground truth manufactured by prompt-swap
   pairing (no extra GPU cost). Manifest written as a labeling sheet.
2. `shortsloop label` — local web UI, keyboard-only (`1` pass / `2` static /
   `3` deformed / `4` flicker / `5` off-prompt / `6` other+note), autoplay, resumable,
   one JSONL line per verdict. **GATE: you label (~10–15 min).**
3. `shortsloop calibrate-tune` — stratified ~28/12 tune/test split; L1 per-metric 1-D
   sweeps targeting zero false-pass on tune with minimal false-fail; L2 floors + judge
   prompt iterated on tune only. Report: test-set agreement, false-pass rate (Wilson
   CI — with n=12 it will be coarse; reported honestly), false-fail, per-class
   confusion, per-layer catch attribution (drives 8B → 32B escalation decision).
   **GATE: you sign off thresholds.** Then `thresholds.yaml` gets
   `calibrated: true` + provenance, committed.

## 7. Milestones (vertical slices, each test-first and pushed)

- **M1**: scaffold (pyproject, CLAUDE.md, pipeline.yaml, thresholds placeholder,
  config.example) + fixture synthesizer + L1 metrics + `shortsloop-check --l1-only`.
  *Verified here (CPU).*
- **M2**: judge adapters + rubric prompt + full `shortsloop-check` + frozen JSON
  Schema file + fail-closed checker tests (matrix rows 1–6, 17). *Verified here
  against fake judge servers.*
- **M3**: dispatch parser + state/events + wave runner + budgets + resume + re-roll
  policy + report generator; end-to-end dry-run against fake ComfyUI + fake judge
  (matrix rows 7–14). *Verified here.*
- **M4**: labeling tool + calibration batch generator + tuner (matrix row 15 feeds
  it). *Tool logic verified here; GPU parts marked UNVERIFIED-ON-GPU.*
- **M5**: doctor + runbook (docs/runbook.md: workstation install, doctor, Phase 0,
  nightly cron example). *Doctor's checks unit-tested against fakes.*
- **Workstation phase (you, or a local session)**: doctor → calibrate-batch → **you
  label** → tune → **you sign off** → first supervised nightly with a real ~6-clip
  dispatch sheet → DoD checklist (§8).

## 8. Definition of done (restated, binding)

- Checker CLI standalone, tests green including every fail-closed path, thresholds
  calibrated with agreement + false-pass reported on a held-out test set.
- One unattended E2E run on the workstation over a real ~6-clip dispatch sheet
  producing: per-clip verdicts + reasons in `report.md`, silent QC MP4s of passing
  clips only, complete `attempts.jsonl`, and in-report evidence that the retry cap
  and GPU budget were respected.
- Pass rate is a tuning metric, not an acceptance criterion.
- Nothing is declared done from inspection alone — every claim traces to a command
  (`pytest`, `ffprobe`, checker exit codes, report contents).

## 9. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Ollama vision broken for chosen VL model on your box (Qwen 3.6-era regression) | adapter interface; llama.cpp fallback is config, not code; `doctor` vision smoke test decides |
| n=12 test set → coarse false-pass estimate | Wilson CIs reported; more labels = new calibration round (provenance-tracked) |
| L2 `imaging_quality` calibrated on draft-res frames | flagged; watch first supervised nightly; L1 sharpness (resolution-normalized) is the primary blur instrument |
| Wan output container variance (SaveVideo vs VHS nodes) | ffprobe is the truth; spec check tolerant on codec, strict on frames/duration/resolution |
| Judge context overflow with 8 images | `num_ctx 16384` + doctor probe validates an 8-image call end-to-end |
| Disk fills mid-night | `disk_min_free_gb` guard at every claim; `doctor` reports headroom |
