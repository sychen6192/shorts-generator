# shortsloop runbook — from empty workstation to unattended nightly

Everything below runs **on the `llm` workstation** (the machine with the RTX 5090,
ComfyUI, and Ollama). The cloud session that built this repo could not verify
GPU behavior — that is what this runbook does, step by step. Do not skip the
gates; the runner enforces them anyway.

## 0. Install

```bash
git clone <this repo> && cd shorts-generator
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -q          # must be green on the workstation too (~4 min, CPU only)
cp config.example.yaml config.yaml
$EDITOR config.yaml          # comfy.host, your VERIFIED T2V API-export workflow path,
                             # judge model, runs_dir on the big disk
```

Workflow file: export your own verified workflow via ComfyUI → Workflow →
Export (API). The `comfyui-ig-video` skill's bundled
`wan22_14b_t2v_lightx2v.json` works as a starting template, but verify the four
model filenames first (`doctor` checks them against the server).

Judge model: `ollama pull qwen3-vl:8b-instruct` (or the closest 8B-class VL
build available). If Ollama's vision sidecar is broken for your model family —
the doctor's vision probe will tell you — switch `judge.adapter:
openai_compat` and point `base_url` at a llama.cpp `llama-server` running the
GGUF + mmproj pair. No code changes, but `openai_compat` **requires**
`judge.unload_url` (e.g. llama-swap's `/unload`): without a way to evict the VLM
before a Wan wave, the runner refuses to start and doctor FAILs `judge.unload`.

Judge timeout/retries for the nightly run live in `pipeline.yaml`
(`policies.judge.timeout_s` / `retries`); `config.yaml`'s `judge.timeout_s` only
applies to a standalone `shortsloop-check`.

## 1. Doctor — executable discovery (repeat after any env change)

```bash
.venv/bin/shortsloop doctor
```

No `config.yaml` yet? Doctor writes one there as a skeleton (a copy of
`config.example.yaml`) and FAILs — fill it in and re-run.

Verifies, in this order:

- **pipeline.yaml** loaded exactly as the runner loads it; a file the runner
  would refuse (e.g. `policies.judge` not a mapping) is a FAIL. Every judge and
  VRAM check below uses these values — the nightly's, not config.yaml's.
- **ffmpeg/ffprobe** — absolute paths are recorded in `doctor.json` (`tools`).
  Cron does not inherit your shell's PATH: doctor prints the `PATH=…` line the
  crontab needs (§4) and warns when the binaries live outside cron's default
  `/usr/bin:/bin`.
- **ComfyUI** reachable, a real CUDA GPU (no devices, CPU mode, or a card smaller
  than `vram_handoff.free_min_gb` / `gen_free_min_gb` = FAIL), queue state.
- **Workflow**: API format; a dry run (`--dump`, nothing queued) proves prompt,
  seed, size, length and steps are patchable at the nightly 720x1280 · 81 frames ·
  16 fps · steps 8; every model file its loader nodes reference exists **on the
  server**. An unlistable `/models/<folder>` or an unrecognized loader class (e.g.
  GGUF loaders) is a FAIL — doctor cannot vouch for those files.
- **VRAM handoff**: `/free`, then free VRAM verified via `/system_stats`. Only then
  does the judge VLM load — hard rule 3 applies to the doctor's own probes too.
- **Judge**: configured exactly as the nightly checker gets it (config.yaml's
  `judge` section with `pipeline.yaml` `policies.judge.timeout_s` / `retries`).
  Reachable, model installed, a **two-call color vision probe** (catches a judge
  that cannot actually see), then an **L2 dry run** — the real nightly call: 8
  frames of a synthetic 720x1280 clip + the rubric + the strict response schema,
  answered within `pipeline.yaml` `policies.judge.timeout_s` (proves `num_ctx` fits
  8 images and the model build handles them; WARN if it took over half the
  timeout — raise it in `pipeline.yaml`, not `config.yaml`). Both probes parse
  replies with the nightly L2 parser: a JSON object, optionally wrapped in one
  `<think>…</think>` block and one ```` ``` ```` fence — nothing else.
- **Judge unload** — the nightly's own judge→generation handoff, proven able to
  catch a resident judge:
  - `openai_compat` without `judge.unload_url` FAILs at once; its probes are not
    run (never load a VLM that cannot be evicted).
  - Doctor reads free VRAM **while the VLM is still loaded** (right after the
    probes), then unloads the VLM + rewrite LLM, `/free`s ComfyUI and verifies
    free VRAM ≥ `vram_handoff.gen_free_min_gb` (default: `free_min_gb`) — the
    threshold the runner uses before every Wan wave.
  - FAIL if VRAM does not come back (the night would halt at its first wave), or
    if the loaded reading already clears that threshold: then the runner could not
    tell a resident judge from an idle card (a resident 8B VLM can leave more than
    24 GB free on a 32 GB card). The detail and `doctor.json` (`vram`) show both
    readings and suggest a `gen_free_min_gb` between them — set it in
    `pipeline.yaml` and re-run doctor.
  - No judge reply at all = the loaded reading proves nothing = FAIL.
- **Rewrite model** on the server the rewrite actually calls (`rewrite.base_url`,
  else the judge's Ollama, else local Ollama) — warning only.
- **Disk** headroom on the filesystem that will hold `paths.runs_dir` (even before
  it exists) · thresholds state.

Writes `doctor.json` next to `config.yaml`. The file is overwritten with
`ok: false` (`status: "in progress"`) the moment doctor starts; only a run that
completes with every check green writes `ok: true`. A crash or Ctrl-C leaves
`ok: false` — never an older green snapshot. **The nightly runner refuses to
start without a passing snapshot** (`--skip-doctor` exists for supervised
debugging only).

Anything doctor *could not verify* that the nightly depends on is a FAIL, not a
warning. `--no-free` (debugging only) skips `POST /free`, so the handoff stays
unverified and the snapshot is written `ok: false` — the runner refuses it. The
judge probes then run only if the GPU is already free (the VLM never loads beside
Wan). Run plain `shortsloop doctor` before any unattended night.

## 2. Phase 0 — calibration (one afternoon, two hard gates)

```bash
# 2a. ~40 draft-res clips, deliberately spanning good and bad (~30-40 min GPU)
.venv/bin/shortsloop calibrate-batch --count 40
#     Before the first Wan job it unloads the judge VLM + rewrite LLM, /free's
#     ComfyUI and VERIFIES free VRAM (pipeline.yaml vram_handoff) — exit 3 if a
#     model is still resident. One job at a time: it waits for an empty ComfyUI
#     queue before every submit and removes a timed-out job before moving on.
#     Resume-safe: Ctrl-C stops the in-flight ComfyUI job, then re-run the same
#     command; finished rows are skipped. 4 prompt-swap rows (a clip shown with
#     a DIFFERENT good prompt) are the off-prompt ground truth.

# 2b. label them — keyboard-only web UI, ~10-15 minutes
.venv/bin/shortsloop label
#     Binds 127.0.0.1 (no auth). Headless workstation: from your laptop
#       ssh -L 8765:127.0.0.1:8765 llm      # then open http://127.0.0.1:8765/
#     (or --host <LAN address> to bind it directly — anyone who can reach it
#     can label). Safari works (byte-range serving).
#     keys: 1 pass · 2 static · 3 deformed · 4 flicker · 5 off-prompt
#     · 6 other · space replay · p previous · n skip. One label per key press
#     (held keys ignored); "All clips labeled" only when none is left.
#     >>> GATE 1: this is YOUR judgment being encoded. <<<

# 2c. score the batch with the VLM judge, then tune
.venv/bin/shortsloop calibrate-tune --with-l2
#     Judging starts only after ComfyUI is /free'd and free VRAM is verified
#     (needs comfy.host in config.yaml); the judge is unloaded afterwards.
#     Scores are keyed by clip+prompt sha and judge model+digest; a clip the
#     judge cannot evaluate is recorded (a runtime ERROR) and skipped on re-run.
#     Tuning refuses (writes NO proposal) with < 10 usable labels, all-pass or
#     all-fail labels, a held-out split missing a pass or a fail, or partial
#     judge coverage. Labels whose clip bytes changed since labeling are
#     excluded and listed.
#     read calibration/tuning_report.md: TEST agreement, FALSE-PASS = labeled-
#     fail clips that would ship / all labeled-fail clips (+Wilson CI),
#     false-fail, per-class confusion, per-layer catch attribution (L1 vs L2,
#     and what each would catch alone) — for the combined L1+L2 decision.
#     If the 8B judge disagrees with your labels too often, pull a ~32B VL
#     model, update config.yaml, and re-run this step: every clip is re-scored
#     by the new judge and only its scores are used; provenance names it.

# 2d. sign off — THE gate that opens unattended running
.venv/bin/shortsloop calibrate-tune --approve
#     copies proposed thresholds → thresholds.yaml with calibrated: true, a
#     version above the current one (YYYY-MM-DD.N) + provenance (labels sha,
#     split, test stats, judge model+digest, input shas). Refuses if labels,
#     manifest or judge scores changed after tuning, if the values were
#     hand-edited, if labels.jsonl is gone, or if config.yaml's judge is not
#     the judge the floors were tuned on.
```

Until 2d, `shortsloop run` refuses to start without `--allow-uncalibrated`.

## 3. First supervised run

Produce a dispatch sheet with the `shorts-trend-dispatch` skill (~6 T2V clips
for the first night), then:

```bash
.venv/bin/shortsloop run --dispatch dispatch-YYYY-MM-DD.md
```

Before Phase 0 is signed off you can rehearse with `--allow-uncalibrated`: PASS
clips then land in `encoded_uncalibrated/` (never `encoded/`) and the report
carries an UNCALIBRATED banner — a rehearsal, not a night's output.

Watch the first one. Read `runs/<run_id>/report.md` against the definition of
done: per-clip verdicts + reasons, silent QC MP4s of passing clips only, full
`attempts.jsonl`, budget/cap evidence. Reproduce one clip from its
`attempts.jsonl` line to close the loop:

```bash
COMFY_HOST=<host:port> python3 shortsloop/vendor/comfy_client.py run \
  -w <workflow> --prompt "<prompt_text>" --seed <seed> \
  --width 720 --height 1280 --length 81 --fps 16 [--steps 8]
```
(every value comes from the attempt's `patch_args`.)

## 4. Unattended nightly

```bash
crontab -e
# cron runs with a minimal PATH: give it the one where `which ffmpeg ffprobe` worked
# (the runner refuses to start without both — doctor prints their absolute paths)
PATH=/usr/local/bin:/usr/bin:/bin
# 01:58 nightly; wall-clock budget + retry caps bound the night (pipeline.yaml).
# mkdir first: a redirect into a missing directory means the job never runs at all.
58 1 * * * cd /path/to/shorts-generator && mkdir -p /big/disk/runs && \
  .venv/bin/shortsloop run --dispatch /path/to/tonight.md >> /big/disk/runs/cron.log 2>&1
```

Only one runner per runs directory: a second invocation (or an overlapping
`--resume`) refuses with exit 2 while the first holds `runs/.shortsloop.lock`.

Morning workflow: open `runs/<run_id>/report.md` → review contact sheets and
reasons → layer audio per the dispatch sheet's 音檔需求 table and re-encode with
`shortsloop/vendor/ig_encode.sh -a bgm.mp3 …` → upload (AI-content label on).
A crashed or halted run resumes with `shortsloop run --dispatch
runs/<run_id>/dispatch.md --resume runs/<run_id>` (the halted report prints the exact
command): finished verdicts are reused (cache keyed by clip, prompt,
thresholds and judge digest), in-flight ComfyUI jobs are re-attached before anything
new is submitted, generated-but-unjudged clips go straight to the judge, and the
wall-clock budget keeps counting from the time already spent. A different sheet is
refused.

## 5. Exit codes & failure playbook

| Symptom | Meaning / fix |
|---|---|
| `run` exits 2 before generating | Refusal: config / doctor snapshot (missing, failed, or taken for a different config/workflow/judge — re-run doctor) / thresholds off-shape or uncalibrated / dispatch intake (per-row errors listed) / workflow knobs not patchable / ffmpeg not on PATH / disk below floor / another run holds the lock. Nothing was burned. |
| `run` exits 3, report says HALTED(infra) | Instrument broke mid-run: judge down, VRAM not freed in either direction (a VLM/LLM still resident before a Wan wave counts), ComfyUI queue busy or a stuck job that won't clear, 3 consecutive generation failures, 2 consecutive judge errors, checker contract violated. Nothing was generated or judged after the halt; clips that passed before it were still encoded. Fix, then `--resume`. |
| Report status `COMPLETED(budget-stopped)` / `COMPLETED(disk-stopped)` | A budget tripped: no new generation after it; everything already generated was judged, encoded and reported (`skipped` rows say which budget). |
| `run` exits 0 with failures in report | Working as designed: failures were caught, bounded, explained. Pass rate is a tuning metric, not an acceptance criterion. |
| Every clip ERRORs at L2 | `doctor` → vision probe + 8-frame L2 dry run. Ollama vision broken ⇒ switch to `openai_compat` + llama.cpp, and set `judge.unload_url` (e.g. llama-swap's `/unload`) — without an unload hook the next Wan wave halts on the VRAM check, by design. When the judge answered but the reply was unusable, its raw text is kept next to the ERROR verdict (`*.l2_raw.json`); timeouts and an unreachable judge leave none. |
| OOM during generation | Runner already `/free`s and re-rolls; if chronic, drop to 480x832 or 81 frames in the dispatch sheet. |
| Report shows `encode failed after PASS` | Clip passed QC but the encode failed verification (size/fps/codec/aac/silence) or the clip bytes no longer match the verdict — nothing was left in `encoded/`; the raw file is kept in `runs/<id>/clips/`. |
| `run` refuses: thresholds calibrated with another judge / halts: judge build differs | L2 floors are tuned to one judge model and build (`thresholds.yaml` provenance.judge). Re-pull that build, or re-run Phase 0 (`calibrate-tune --with-l2`, then `--approve`) for the new judge. |
| `steps8 n/a` in attempt notes | The re-roll table's `--steps 8` is defined for the 4-step lightx2v build; your workflow runs another step count, so those re-rolls reseed only. |

## 6. What is still UNVERIFIED-ON-GPU (first-night checklist)

- [ ] `doctor` fully green on the workstation
- [ ] `calibrate-batch` produces 40 playable draft clips (spot-check 2-3)
- [ ] judge wave actually fits in VRAM after `/free` (watch `nvidia-smi` once), and
      the judge->generation handoff sees the VLM gone (`generate ok layer=vram_handoff`
      events in events.jsonl show the measured free VRAM)
- [ ] handoff thresholds suit the card: `vram_handoff.free_min_gb` (24) must leave
      the judge room; set `vram_handoff.gen_free_min_gb` so an idle card clears it but
      one with the VLM still loaded (`ollama ps`) does not — otherwise the
      judge->generation check cannot tell them apart
- [ ] real Wan clip L1 metrics look sane vs fixtures (`shortsloop check --l1-only` on one)
- [ ] E2E: one real ~6-clip dispatch sheet unattended → morning report (DoD)
