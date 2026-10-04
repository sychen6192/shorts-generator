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
GGUF + mmproj pair. No code changes.

## 1. Doctor — executable discovery (repeat after any env change)

```bash
.venv/bin/shortsloop doctor
```

No `config.yaml` yet? Doctor writes one there as a skeleton (a copy of
`config.example.yaml`) and FAILs — fill it in and re-run.

Verifies, in this order:

- **ffmpeg/ffprobe** — absolute paths are recorded in `doctor.json` (`tools`).
  Cron does not inherit your shell's PATH: doctor prints the `PATH=…` line the
  crontab needs (§4) and warns when the binaries live outside cron's default
  `/usr/bin:/bin`.
- **ComfyUI** reachable, a real CUDA GPU (no devices, CPU mode, or a card smaller
  than `vram_handoff.free_min_gb` = FAIL), queue state.
- **Workflow**: API format; a dry run (`--dump`, nothing queued) proves prompt,
  seed, size, length and steps are patchable at the nightly 720x1280 · 81 frames ·
  16 fps · steps 8; every model file its loader nodes reference exists **on the
  server**. An unlistable `/models/<folder>` or an unrecognized loader class (e.g.
  GGUF loaders) is a FAIL — doctor cannot vouch for those files.
- **VRAM handoff**: `/free`, then free VRAM verified via `/system_stats`. Only then
  does the judge VLM load — hard rule 3 applies to the doctor's own probes too.
- **Judge**: reachable, model installed, a **two-call color vision probe** with
  strict JSON (catches a judge that cannot actually see), then an **L2 dry run** —
  the real nightly call: 8 frames of a synthetic 720x1280 clip + the rubric + the
  strict response schema, parsed within `judge.timeout_s` (proves `num_ctx` fits 8
  images and the model build handles them; WARN if it took over half the timeout).
- **Judge unload**: the VLM (and rewrite LLM) are unloaded right after the probes
  and VRAM is re-verified. A judge that cannot unload (`openai_compat` without
  `judge.unload_url`) FAILs here instead of halting the night at the first
  judge→generation handoff.
- **Rewrite model** on the server the rewrite actually calls (`rewrite.base_url`,
  else the judge's Ollama, else local Ollama) — warning only.
- **Disk** headroom on the filesystem that will hold `paths.runs_dir` (even before
  it exists) · pipeline policies · thresholds state.

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

Watch the first one. Read `runs/<run_id>/report.md` against the definition of
done: per-clip verdicts + reasons, silent QC MP4s of passing clips only, full
`attempts.jsonl`, budget/cap evidence. Reproduce one clip from its
`attempts.jsonl` line to close the loop:

```bash
COMFY_HOST=<host:port> python3 shortsloop/vendor/comfy_client.py run \
  -w <workflow> --prompt "<prompt_text>" --seed <seed> \
  --width 720 --height 1280 --length 81
```

## 4. Unattended nightly

```bash
crontab -e
# 02:00 nightly; wall-clock budget + retry caps bound the night (pipeline.yaml)
0 2 * * * cd /path/to/shorts-generator && .venv/bin/shortsloop run \
  --dispatch /path/to/tonight.md >> runs/cron.log 2>&1
```

Morning workflow: open `runs/<run_id>/report.md` → review contact sheets and
reasons → layer audio per the dispatch sheet's 音檔需求 table and re-encode with
`shortsloop/vendor/ig_encode.sh -a bgm.mp3 …` → upload (AI-content label on).
A crashed run resumes with `shortsloop run --dispatch … --resume runs/<run_id>`
— finished verdicts are reused, in-flight ComfyUI jobs re-attached.

## 5. Exit codes & failure playbook

| Symptom | Meaning / fix |
|---|---|
| `run` exits 2 before generating | Refusal: config/doctor/thresholds/dispatch intake — message says which. Nothing was burned. |
| `run` exits 3, report says HALTED(infra) | Instrument broke mid-run (judge down, VRAM not freed, checker contract violated). Nothing shipped after the halt point. Fix, then `--resume`. |
| `run` exits 0 with failures in report | Working as designed: failures were caught, bounded, explained. Pass rate is a tuning metric, not an acceptance criterion. |
| Every clip ERRORs at L2 | `doctor` → vision probe. Ollama vision broken ⇒ switch to `openai_compat` + llama.cpp. |
| OOM during generation | Runner already `/free`s and re-rolls; if chronic, drop to 480x832 or 81 frames in the dispatch sheet. |
| Report shows `encode failed after PASS` | Clip passed QC but ffmpeg encode broke — raw file is kept in `runs/<id>/clips/`, encode manually. |

## 6. What is still UNVERIFIED-ON-GPU (first-night checklist)

- [ ] `doctor` fully green on the workstation
- [ ] `calibrate-batch` produces 40 playable draft clips (spot-check 2-3)
- [ ] judge wave actually fits in VRAM after `/free` (watch `nvidia-smi` once)
- [ ] real Wan clip L1 metrics look sane vs fixtures (`shortsloop check --l1-only` on one)
- [ ] E2E: one real ~6-clip dispatch sheet unattended → morning report (DoD)
