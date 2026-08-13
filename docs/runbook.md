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

Verifies: ffmpeg/ffprobe · ComfyUI reachable, VRAM, queue · workflow is
API-format and its model files exist **on the server** · `/free` actually frees
VRAM (verified via `/system_stats`) · judge reachable, model installed, and a
**two-image vision probe with strict JSON** (catches a judge that cannot
actually see) · rewrite model (warning only) · disk headroom · thresholds state.

Writes `doctor.json` next to `config.yaml`. **The nightly runner refuses to
start without a passing snapshot** (`--skip-doctor` exists for supervised
debugging only).

## 2. Phase 0 — calibration (one afternoon, two hard gates)

```bash
# 2a. ~40 draft-res clips, deliberately spanning good and bad (~30-40 min GPU)
.venv/bin/shortsloop calibrate-batch --count 40
#     resume-safe: re-run after any interruption, finished rows are skipped

# 2b. label them — keyboard-only web UI, ~10-15 minutes
.venv/bin/shortsloop label
#     open the printed URL; keys: 1 pass · 2 static · 3 deformed · 4 flicker
#     · 5 off-prompt · 6 other · space replay · p previous
#     >>> GATE 1: this is YOUR judgment being encoded. <<<

# 2c. score the batch with the VLM judge, then tune
.venv/bin/shortsloop calibrate-tune --with-l2
#     read calibration/tuning_report.md: agreement, FALSE-PASS rate (+CI),
#     per-class catches, what L1 leaves for L2, per-layer attribution.
#     If the 8B judge disagrees with your labels too often, pull a ~32B VL
#     model, update config.yaml, and re-run this step — the data decides.

# 2d. sign off — THE gate that opens unattended running
.venv/bin/shortsloop calibrate-tune --approve
#     copies proposed thresholds → thresholds.yaml with calibrated: true
#     + provenance (labels sha, split, test stats). Refuses stale labels.
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
