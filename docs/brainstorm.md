# shortsloop — discovery + design brainstorm (pre-plan)

Status: **awaiting user feedback** — nothing here is frozen. The plan (with the frozen
verdict schema) comes after this discussion, per the working agreement.

---

## 1. Discovery record (2026-08-12, cloud session)

| Item | Result |
|---|---|
| Session environment | **Claude Code cloud container, not the `llm` workstation.** Repo `sychen6192/shorts-generator` is empty (zero commits). Branch `claude/shortsloop-nightly-pipeline-hqxyty`. |
| ComfyUI server | Unreachable from this container (expected — it lives on the workstation LAN). Client defaults to `COMFY_HOST=127.0.0.1:8188`. Real value must be discovered on the workstation → see D1 (`shortsloop doctor`). |
| `comfyui-ig-video` skill | Found at `/root/.claude/skills/synced/comfyui-ig-video/` (account-synced). Read: `SKILL.md`, `references/wan22_notes.md`, `references/comfyui_api.md`, `scripts/comfy_client.py`, `scripts/ig_encode.sh`, `workflows/README.md`. |
| `shorts-trend-dispatch` skill | Found at `/root/.claude/skills/synced/shorts-trend-dispatch/`. Read: `SKILL.md`, `assets/dispatch_template.md`, `references/format_recipes.md`. Dispatch sheet = markdown, manifest table (`ID / 模式 / image 依賴 / 輸出`) + per-clip fenced prompts. |
| Ollama | Not present in this container. Must verify on workstation: `ollama list`, vision support for the chosen VL model (user reports mid-2026 Ollama does not wire the vision sidecar for the Qwen 3.6 family — cannot verify from here; verified by `doctor`). |
| GPU / ffmpeg / disk | None of these are the workstation's values here. Workstation per kickoff: RTX 5090 32 GB VRAM, 64 GB RAM. `doctor` must report: GPU via ComfyUI `/system_stats`, `ffmpeg`/`ffprobe -version`, `df` for the clips/state volume. |
| superpowers plugin | Not available in this environment's plugin catalog. Following the same workflow manually (brainstorm → approved plan → TDD). |

**Key facts extracted from the skills (authoritative):**

- Wan 2.2 14B T2V: native **16 fps**, length **4n+1** (81 = 5.0 s), draft **480x832**,
  final **720x1280**. lightx2v 4-step: `cfg` **must stay 1.0**, shift 5.0, boundary 2;
  8-step variant = better motion/quality, still fast. Full no-LoRA ≈ 20 steps, several
  times slower. Distill LoRAs **mute motion** → prompts must over-specify action + camera
  move. Official Chinese negative prompt is baked into the bundled template.
- Reference timings (RTX 4090, 4-step, 5 s clip): ~1 min @ 480x832, ~3.5 min @ 720x1280;
  5090 faster. First job after model swap adds minutes of load time.
- `comfy_client.py`: stdlib-only; `run` = patch+submit+wait+download; machine-readable
  `RESULT {...}` last line (ok / seed / files / elapsed); exit 0 ok, 2 exec error,
  3 timeout, 4 bad workflow. `free` = POST `/free {unload_models, free_memory}` (the VRAM
  handoff primitive). `/system_stats` reports VRAM free (lets us *verify* the handoff).
  OOM appears as `execution_error` message, not HTTP error.
- Dispatch sheet contract: unattended runs go **straight to 720x1280 final** (draft pass
  is for supervised iteration only); sequential, one job at a time; seeds logged;
  I2V chain rows carry `image 依賴` = parent clip's last frame (out of scope v1);
  silent QC encodes first, audio is a morning job.
- `ig_encode.sh`: always outputs 1080x1920/30fps H.264 yuv420p + AAC (+silent stereo
  track when no `-a`), `+faststart`, prints its own spec check.

---

## 2. Adopted unchanged from the kickoff

- Two-layer checker; L1 fail ⇒ L2 never runs on that attempt.
- Checker CLI contract: `shortsloop-check <clip> --prompt-file <txt> --json <out>`;
  exit **0 PASS / 1 FAIL / 2 ERROR**; ERROR is never PASS; fail closed everywhere.
- The five L2 dimensions (with definition refinements in D5).
- Hard rules 1–6 verbatim into `CLAUDE.md` at implementation start.
- Phase 0: ~40-clip draft-res calibration batch spanning good/bad, local labeling tool,
  hard stop for hand labels, tune/test split, agreement + false-pass reporting,
  threshold sign-off gate.
- Bounded work: retry cap default 2 re-rolls (3 attempts), run wall-clock GPU budget,
  clean stop + loud report when either trips.
- Append-only JSONL state; every attempt reproducible from the log.
- v1 scope: T2V independent clips only; dispatch sheet is the only input format;
  no upload, no keyframe gate, no I2V chains.

---

## 3. Design positions (D1–D10) — argue with any of these

### D1. Environment split: build here, verify on the workstation via `shortsloop doctor`

This session cannot see the GPU. Rather than assume values, discovery becomes an
executable: `shortsloop doctor` runs **on the workstation** and verifies/reports:
ComfyUI reachable + version + VRAM (via `/system_stats`), models present for the chosen
workflow, `ollama list` + a 1-image vision smoke test + strict-JSON compliance probe,
ffmpeg/ffprobe presence, free disk on the configured clips/state volume, workflow JSON
is API-format. Doctor output fills `config.yaml`; the nightly runner refuses to start
if doctor's last snapshot is missing or stale-failed.

What this session CAN fully verify on CPU: all of L1 (with ffmpeg-synthesized fixture
clips — static / moving / flicker / black / blurry), every fail-closed path (fake judge,
fake ComfyUI), dispatch parsing, state machine, budgets, resume, report rendering.
Everything GPU-touching ships marked **UNVERIFIED-ON-GPU** until doctor + Phase 0 run
on the workstation (by you, or by a local Claude session on the same repo).

### D2. Scheduling: generation waves with inline L1, batched L2 (VRAM-swap economics)

Naive per-clip generate→judge→re-roll interleaving swaps Wan ⇄ VLM per attempt; each
swap costs minutes of model load. Instead the run proceeds in **waves**:

```
wave N: [generate attempt → L1 gate (CPU, seconds) → L1-fail? re-roll immediately
         while Wan is still loaded, within caps] for every pending clip, sequentially
        → ComfyUI /free → verify VRAM actually freed via /system_stats
        → VLM judge wave: full check (L1+L2) for every L1-surviving attempt
        → classify failures → schedule wave N+1 re-rolls → unload VLM
```

Max waves = retry cap + 1. Model swaps ≤ 2 per wave instead of 2 per attempt. Your most
common failure (near-static) is caught by L1 inline and re-rolled **without ever paying
a model swap or a VLM call**. A clip can exhaust all attempts on L1 alone — correct and
cheap. Rule 3 (VRAM time-separation) and rule 4 (one job at a time) hold by construction,
and the /free→/system_stats verification makes rule 3 *checked*, not assumed.

Cost sketch for a 6-clip sheet on the 5090: all-pass night ≈ 6×~3 min gen + 2 swaps +
6 judge calls ≈ 25–35 min. Worst case (18 attempts, 3 waves) ≈ 1.5–2 h. Comfortably
inside a night; the wall-clock budget is a backstop, not a constraint.

### D3. Canonical analysis resolution — calibration transfer trap

Phase 0 calibrates on **480x832 draft** clips; nightly production is **720x1280 final**
(dispatch skill: unattended runs skip draft). Laplacian variance and optical-flow
magnitudes are resolution-dependent — thresholds tuned on drafts would be silently wrong
at final res. Fix: the checker **normalizes every input to a fixed analysis resolution**
(decode → downscale to 480x832; flow computed at half of that) before computing any L1
metric, and flow is additionally normalized by frame diagonal. Thresholds then transfer.
The verdict JSON records the analysis resolution. L2 frames are resolution-robust
(the VLM resizes anyway); noted as a residual caveat for `imaging_quality` only.

### D4. L1 = ffprobe + one OpenCV pass, not an ffmpeg filter zoo

`blackdetect`/`freezedetect` would mean parsing ffmpeg stderr and calibrating a second
family of knobs. Instead: `ffprobe` (JSON) for container/spec sanity, then **one
sequential decode pass** computing all pixel metrics per frame/pair:

- per-pair dense optical flow (Farneback) mean magnitude, normalized by diagonal →
  aggregate median + p90 → `static` (and freeze runs: consecutive pairs with
  SSIM ≈ 1.0 *and* flow ≈ 0)
- per-pair grayscale SSIM → min / p05 / mean + count of discontinuity dips → `flicker`
  and hard-cut/blow-up detection
- per-frame luma mean + clipped-pixel fraction → `black` / blown exposure
- per-frame Laplacian variance → p10 + median → `blurry`

One code path shared by calibration and production; all raw numbers land in the verdict
JSON; reproducible; no ffmpeg-version-dependent filter parsing. Deterministic, CPU-only,
seconds per clip at analysis resolution.

### D5. L2 dimensions: keep your five, with two definition changes

**(a) Temporal coherence from 6–8 sampled stills cannot see flicker** — flicker is a
frame-to-frame phenomenon at 16 fps; the sampled frames are ~0.7 s apart. L1's SSIM
metrics see exactly that. So: **L1 owns flicker/popping**; L2 `temporal_coherence` is
redefined as *semantic* continuity across the sampled sequence — subject teleporting,
object count changes, scene morphing, impossible motion. No double-counting, and each
instrument measures what it can actually see.

**(b) `subject_consistency` gets an N/A path.** Landscape/timelapse-family clips have no
persistent subject; forcing a score makes the judge fabricate. Schema allows
`"na": true`; the decision rule skips N/A dimensions. The judge prompt states the N/A
criterion explicitly.

Also: frames go to the VLM as **individual images** (not a tiled grid — 8B-class models
lose hands/fingers in grids), uniformly sampled including first and last frame,
timestamps recorded in the verdict.

### D6. Decision rule: per-dimension floors, no weighted composite

With ~28 tuning labels, fitting weights is overfitting theater. Rule: every dimension
scored 1–5 against an anchored rubric; **FAIL if any non-N/A dimension < its floor**
(floors per-dimension, calibrated in Phase 0). L1 checks are the same shape: metric vs
threshold, any failing check ⇒ FAIL. Every floor/threshold is a single auditable number
in `thresholds.yaml`. Weighted scoring is a v2 question, only if the data demands it.

### D7. ERROR taxonomy: clip-scope vs infra-scope (both never PASS)

- **clip-scope ERROR** — this clip could not be evaluated (unreadable/corrupt file,
  0 frames, judge returned schema-invalid JSON for this clip after 1 retry). Runner
  treats it as a failed attempt: re-roll within the cap, never ship. One truncated MP4
  should not kill the night.
- **infra-scope ERROR** — the instrument is broken (judge unreachable, VLM load failure,
  VRAM not freed, ComfyUI down, thresholds file missing/uncalibrated). Runner **halts
  the run** and reports; generating more clips nobody can judge burns GPU for nothing.
  Escalation rule: ≥2 consecutive clip-scope L2 errors ⇒ reclassify as infra ⇒ halt.

Exit code stays 2 for both; the verdict JSON carries `error.scope`. Rule 2 (fail closed)
holds in both branches, with tests for each.

### D8. Re-roll policy — concrete mapping (only knobs the notes endorse)

Failure classes (priority order when several fire):
`broken > black > static > deformed > flicker > off_prompt > blurry`.

| Class | Source | Attempt 2 | Attempt 3 |
|---|---|---|---|
| broken / black | L1 (spec/decode/luma) | new seed, same everything | same |
| static | L1 flow (or L2 motion note) | new seed + **append motion/camera phrase** (fixed template bank, deterministic pick, logged verbatim) | + escalate sampler **4-step → 8-step** (notes: better motion dynamics; boundary rescales via `--steps 8`) |
| deformed | L2 anatomy or subject-consistency floor | new seed, same prompt/settings (anatomy is a seed lottery; official negative already covers 畸形/多余的手指) | same |
| flicker | L1 SSIM discontinuity (or L2 semantic-temporal) | new seed + **steps 4→8** | same |
| off_prompt | L2 adherence floor | **one prompt rewrite** by local text LLM — anchors preserved verbatim, recipe order enforced (subject→action→camera→scene→style), ≤110 words, old/new + diff logged | plain re-roll, new seed, rewritten prompt kept |
| blurry | L1 Laplacian (or L2 imaging floor) | new seed | same |

Never touched: `cfg` (must stay 1.0 with distill LoRA — raising it burns the image),
`shift` (notes give no guidance; not inventing knobs), the official negative prompt
(already contains 静态/静止 terms; appending to it is unproven). Every deviation from
the dispatch sheet's parameters is recorded in `attempts.jsonl` (exact patch args).

### D9. Checker CLI surface: only the full checker can say PASS

`shortsloop-check` runs L1+L2 and is the **sole PASS authority**. The wave scheduler's
inline gate is `shortsloop-check --l1-only`, whose verdict vocabulary is
`PROCEED / FAIL / ERROR` — exit 0 means "not rejected", and the runner's ship-gate
requires a verdict JSON with `verdict == "PASS"` **and both layers present** (enforced
in code, proven by a test that tries to ship an L1-only verdict and gets rejected).
L1 runs twice on surviving clips (gate + inside full check) — it costs seconds and buys
an airtight rule 1: one invocation, one artifact, both layers, per shipped clip.

### D10. Judge serving: Ollama + qwen3-vl 8B-class default, adapter for fallback

Default: Ollama chat API, `qwen3-vl:8b`-class instruct model, `temperature 0`, fixed
seed, **structured output** (JSON schema enforced server-side), `keep_alive` managed so
the VLM unloads at wave end (verified, not assumed). Your note says mid-2026 Ollama
doesn't wire the vision sidecar for the Qwen 3.6 family — `doctor`'s vision smoke test
is the arbiter of what actually works on your box. The judge client is a thin adapter
interface with two impls: Ollama native (`/api/chat` + `format:` schema) and
OpenAI-compatible (`/v1/chat/completions`, for llama.cpp `llama-server` with GGUF+mmproj
or vLLM) — so a broken Ollama vision path is a config change, not a rewrite. Escalation
to a ~32B model happens only if Phase 0 shows the 8B disagreeing with your labels too
often (the calibration report includes per-layer catch attribution to make that call
data-driven).

VRAM: time-separated by D2's wave structure; 8B-class Q4 ≈ 6–8 GB, 32B-class Q4 ≈
~20 GB — either fits alone on 32 GB. Judge verdicts are cached keyed by
(clip sha256, prompt sha256, thresholds version, model digest): resumes and re-runs
never re-burn GPU for an already-judged artifact.

---

## 4. Draft verdict JSON v0 (to be frozen in the plan)

```json
{
  "schema_version": "1",
  "verdict": "PASS | FAIL | ERROR",
  "failure_classes": ["static"],
  "error": null,
  "clip": {
    "path": "runs/2026-08-15/clips/V1C3_a2.mp4",
    "sha256": "…",
    "container": {"duration_s": 5.06, "width": 720, "height": 1280,
                   "fps": 16.0, "nb_frames": 81, "vcodec": "h264"}
  },
  "prompt": {"text": "…", "sha256": "…"},
  "l1": {
    "analysis": {"width": 480, "height": 832, "frames": 81, "flow_downscale": 2},
    "metrics": {
      "flow_mag_median": 0.0021, "flow_mag_p90": 0.0034,
      "ssim_min": 0.912, "ssim_p05": 0.941, "ssim_mean": 0.978, "flicker_dips": 0,
      "freeze_longest_run_s": 0.0,
      "laplacian_p10": 41.2, "laplacian_median": 88.7,
      "luma_mean": 0.41, "black_frame_frac": 0.0, "clipped_frac": 0.004
    },
    "checks": [
      {"name": "motion", "metric": "flow_mag_median", "op": ">=",
       "threshold": 0.004, "value": 0.0021, "pass": false,
       "reason": "median optical flow below floor — near-static clip"}
    ],
    "pass": false
  },
  "l2": {
    "model": "ollama/qwen3-vl:8b-instruct",
    "model_digest": "sha256:…", "temperature": 0, "seed": 7,
    "frames": [{"index": 0, "t": 0.0}, {"index": 11, "t": 0.69}],
    "dimensions": {
      "prompt_adherence":    {"score": 4, "floor": 3, "na": false, "pass": true,  "reason": "…"},
      "subject_consistency": {"score": 5, "floor": 3, "na": false, "pass": true,  "reason": "…"},
      "anatomy_artifacts":   {"score": 2, "floor": 3, "na": false, "pass": false, "reason": "left hand has six fingers in frames 3–5"},
      "temporal_coherence":  {"score": 4, "floor": 3, "na": false, "pass": true,  "reason": "…"},
      "imaging_quality":     {"score": 4, "floor": 3, "na": false, "pass": true,  "reason": "…"}
    },
    "pass": false,
    "raw_response_sha256": "…"
  },
  "thresholds": {"version": "2026-08-20.1", "calibrated": true, "file_sha256": "…"},
  "layers_run": ["l1", "l2"],
  "timing": {"l1_s": 3.1, "l2_s": 38.4},
  "versions": {"checker": "0.1.0", "ffprobe": "7.x", "opencv": "4.x", "numpy": "2.x"}
}
```

Notes: `l2` is `null` when L1 already failed (cheap layer short-circuits) or in
`--l1-only` mode (which additionally can never emit `"PASS"`); `error` object =
`{"scope": "clip|infra", "stage": "probe|l1|l2", "message": "…"}` and implies
`verdict == "ERROR"`; ship-gate requires `verdict=="PASS" && layers_run==["l1","l2"]
&& thresholds.calibrated`.

---

## 5. Phase 0 calibration protocol (sketch)

1. **Batch design (~40 clips, 480x832 draft, ~30–40 min GPU total)**: ~24 unique prompts
   × 1–2 seeds. Mix: format_recipes families (realistic positives), deliberately
   motion-starved prompts (no action verb, no camera move — static positives for the
   judge to catch), anatomy-stress prompts (hands-heavy close-ups), plus **off-prompt
   pairs manufactured by prompt-swapping** — take a good clip generated from prompt A
   and have the checker judge it against prompt B; ground-truth off-prompt failures at
   zero extra GPU cost.
2. **Labeling tool**: `shortsloop label --clips … --out labels.jsonl` — single-page
   local web UI (stdlib HTTP server), autoplay loop, prompt displayed alongside,
   keyboard-only: `1` pass, `2` static, `3` deformed, `4` flicker, `5` off-prompt,
   `6` other+note, space = replay, auto-advance, resumable, one JSONL line per verdict.
   40 clips ≈ 10–15 minutes.
3. **HARD GATE — you label.**
4. **Tuning**: stratified split ~70/30 tune/test (≈28/12). L1 thresholds: per-metric
   1-D sweeps on the tune set targeting **zero false-pass on tune** with minimal
   false-fail. L2: rubric prompt iterated on tune set only; floors likewise. Report on
   the held-out test set: overall agreement, **false-pass rate** (with honest
   small-n confidence intervals), false-fail rate, per-class confusion, and per-layer
   catch attribution (did L1 alone / L2 alone / both catch each true failure) — the
   latter drives the 8B-vs-32B escalation decision.
5. **HARD GATE — threshold sign-off.** `thresholds.yaml` then carries
   `calibrated: true` + provenance (labels file sha256, date, agreement stats). The
   nightly runner refuses to run unattended with `calibrated: false` (tested).
6. Test set stays untouched by tuning; if we iterate the judge prompt after seeing test
   results, that's a new calibration round and gets labeled as such in provenance.

## 6. State, stages, and the migration story

- Stage names are exactly your manifest vocabulary: `schedule → claim → generate →
  verify → persist → complete`, implemented as pure-ish functions with disk-visible
  inputs/outputs, orchestrated by a thin loop that reads `pipeline.yaml` (stage list +
  retry/budget policy + judge/comfy endpoints). Migration to a workflow engine =
  re-binding stage names to engine tasks; semantics are documented per stage
  (idempotency, inputs, outputs, side effects).
- Disk layout per run: `runs/<run_id>/{run.json, events.jsonl, attempts.jsonl,
  clips/, verdicts/, encoded/, report.md, report.json}` — all JSONL/JSON/MP4/MD,
  `less`-able, no database.
- `events.jsonl` is an event-sourced transition log; `--resume` folds it to rebuild
  state and continues (mid-generation crash re-attaches via recorded ComfyUI
  `prompt_id` using the client's `wait`).
- `attempts.jsonl`: clip id, attempt n, seed, full prompt text used, workflow file +
  sha256, every patch arg, ComfyUI prompt_id, output file + sha256, verdict pointer +
  summary, durations, VRAM snapshots. Reproducibility = feed those exact values back
  to `comfy_client.py`.
- Report: `report.md` leads with run totals (pass/fail/error/skipped, attempts, GPU
  minutes by stage, budget consumption, wave count), then a per-video completeness view
  (dispatch sheets group clips into videos — "V2: 5/6 passed, C4 failed twice: static"),
  then per-clip blocks with a 6-frame contact-sheet JPEG (ffmpeg tile) so the morning
  review is literally one page + thumbnails. `report.json` mirrors it for machines.

## 7. Open questions for the user

1. **Where do we build?** Recommend: design+code+CPU-tests in this cloud session
   (pushed to this branch), GPU verification via `shortsloop doctor` + Phase 0 on the
   workstation. Alternative: stop now and re-kick locally.
2. **Wave scheduling (D2)** — approve, or insist on strict per-clip interleave?
3. **Checker amendments** — D5 (temporal = semantic; L1 owns flicker; N/A path),
   D6 (per-dimension floors), D7 (clip/infra ERROR split): approve individually.
4. **Nightly resolution** — dispatch default says unattended = straight to 720x1280
   final. Confirm for shortsloop v1 (calibration transfers via D3), or do you want
   nightly drafts at 480x832 instead?
5. Anything in D8's re-roll table you'd cut or add?
