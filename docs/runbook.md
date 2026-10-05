# shortsloop runbook — from empty workstation to unattended nightly

Everything below runs **on the `llm` workstation** (the machine with the RTX 5090,
ComfyUI, and Ollama), **from the repo root**, with `.venv/bin/shortsloop` (the venv
is never activated; `config.yaml`, `pipeline.yaml`, `thresholds.yaml` and
`calibration/` resolve against the working directory). The cloud session that built
this repo could not verify GPU behavior — that is what this runbook does, step by
step. Do not skip the gates; the runner enforces them anyway.

Step-by-step version with pass/fail criteria per step (zh-TW):
[`docs/workstation-checklist.md`](workstation-checklist.md).

Every command a tool prints for you to run next (the resume command,
calibrate-batch's `Next:`, calibrate-tune's `sign off with:`, refusals that say
`run shortsloop doctor`) starts with a bare `shortsloop`: run it from the repo root
as `.venv/bin/shortsloop …`.

Paths: `<runs_dir>` is `paths.runs_dir` from config.yaml (absolute, on the big disk,
e.g. `/big/disk/runs`); `<run_dir>` is `<runs_dir>/<run_id>`, with `run_id` = UTC
`YYYYMMDD-HHMMSS-nightly` (`-2`, `-3`, … if taken).

## 0. Install

Prerequisites: Python ≥ 3.10 with venv; `ffmpeg` + `ffprobe` (libx264 and aac
encoders) on PATH — the test fixtures and the nightly QC encode need them.

```bash
git clone <this repo> && cd shorts-generator
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -q          # must be green: last line `453 passed`, or without node
                             # `451 passed, 2 skipped` (label-UI browser tests) — both
                             # green; CPU only, ~16 min on a 4-core cloud box
cp config.example.yaml config.yaml
$EDITOR config.yaml          # comfy.host (host:port, no http://; COMFY_HOST is ignored),
                             # comfy.workflow_t2v (ABSOLUTE path to your verified T2V
                             # API export), judge, rewrite, paths.runs_dir (ABSOLUTE,
                             # on the big disk)
```

**Local changes and updates.** After Phase 0, `thresholds.yaml` (2d) and
`pipeline.yaml` (`gen_free_min_gb`, §1) stay **uncommitted local changes** — never
stash, commit, `checkout --` or `reset --hard` them — backed up outside the repo with
`calibration/` and `config.yaml` (2d). Update with `git pull --no-rebase --ff-only`
(if it refuses, stop and hand it to the owner), then pytest and doctor; a pull that
changes `shortsloop/l1.py` also needs 2c + 2d. A pull that touches `shortsloop/l2.py`
or `shortsloop/judge/` needs a re-score (recommended too after changing the
judge's temperature/seed/num_ctx or `policies.judge`), which no code enforces —
stored scores are keyed only by clip, prompt and judge model+digest, so 2c would
reuse them: run `mv calibration/l2_scores.jsonl calibration/l2_scores.old.jsonl`,
then 2c `--with-l2` and 2d. Sole exception to the no-`checkout --` rule: to revoke a
sign-off you no longer trust, `git checkout -- thresholds.yaml` (back to
`calibrated: false`; unattended runs refuse), then extend Phase 0 and re-run 2c +
2d — the signed-off copy is in its 2d backup (a new timestamped folder per backup,
so a later one never overwrites it). A re-sign-off on the same UTC day can reuse the
revoked version number (YYYY-MM-DD.1): tell them apart by `thresholds.sha256` in
run.json or by the backup folder's timestamp. `doctor.json` (§1) is not gitignored:
run `echo doctor.json >> .git/info/exclude` once, so `git status --short` shows only
` M pipeline.yaml` / ` M thresholds.yaml`.
docs/plan.md §6 says the signed-off thresholds are committed instead — that
disagreement is an owner decision.

Workflow file: export your own verified workflow via ComfyUI → Workflow →
Export (API). The `comfyui-ig-video` skill's bundled `wan22_14b_t2v_lightx2v.json`
works as a starting template, but verify its **six** model files first — two Wan 2.2
14B UNETs, a text encoder, a VAE and two lightx2v LoRAs, in four folders (file names:
checklist step 0; `doctor` checks each one on the server, `models.<folder>`).
With the lightx2v LoRAs both samplers must stay at **cfg 1.0** (higher burns the
image; doctor does not check cfg) and steps 4 (otherwise the re-roll's steps 8 is
n/a, §5). Calibrate and run with the same workflow file; save a changed one under a
new name.

Judge model: `ollama pull qwen3-vl:8b-instruct` (or the closest 8B-class VL
build available). Rewrite model (the one logged off_prompt rewrite, on by default):
`ollama pull qwen3:8b` on the server it uses — `rewrite.base_url`, else the judge's
Ollama, else local Ollama — or set `rewrite.enabled: false`. Missing, it is only
`WARN rewrite.model` and attempt 2 just reseeds. Turn off Ollama auto-updates: a new
judge digest makes `run` refuse (§5).

If Ollama's vision sidecar is broken for your model family — the doctor's vision
probe will tell you — switch `judge.adapter: openai_compat`: run llama.cpp
`llama-server` with the GGUF + mmproj pair **behind llama-swap** (a bare
`llama-server` cannot be unloaded); set `judge.base_url` to llama-swap's root (no
`/v1`), `judge.model` to the exact id it serves and `judge.unload_url` to its
`/unload` (a plain GET must evict the VLM). This adapter sends neither `num_ctx` nor
`keep_alive_wave`: set the context on the server (e.g. `-c 16384`). No code changes,
but `openai_compat` **requires** `judge.unload_url`: without a way to evict the VLM
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
  would refuse (e.g. `policies.judge` not a mapping) is a FAIL, a missing one a WARN
  (defaults apply). Every judge and VRAM check below uses these values.
- **ffmpeg/ffprobe** — absolute paths are recorded in `doctor.json` (`tools`).
  Cron does not inherit your shell's PATH: doctor prints the `PATH=…` line the
  crontab needs (§4) and warns when the binaries live outside cron's default
  `/usr/bin:/bin`.
- **GPU lock** `<runs_dir>/.shortsloop.lock`, taken before any GPU check; this
  creates `paths.runs_dir` (`mkdir -p`), so mount the big disk first. Held by a run or
  calibration = `FAIL gpu.lock`, GPU checks skipped (§4).
- **ComfyUI** reachable, a real CUDA GPU (no devices, CPU mode, or a card smaller
  than `vram_handoff.free_min_gb` / `gen_free_min_gb` = FAIL), queue state.
- **Workflow**: API format; a dry run (`--dump`, nothing queued) proves prompt,
  seed, size, length, fps (`fps`/`frame_rate` on a video node) and steps are
  literal, patchable inputs at the nightly 720x1280 · 81 frames · 16 fps · steps 8
  (a knob wired from another node FAILs `workflow.knobs`; its OK line omits fps,
  but fps is checked); every model file its loader nodes reference exists **on the
  server**. An unlistable `/models/<folder>` or an unrecognized loader class (e.g.
  GGUF loaders) is a FAIL — doctor cannot vouch for those files. Not checked: that
  every node class is installed and each job saves **exactly one** video — else the
  night halts `HALTED(infra)`; the supervised run (§3) proves both.
- **VRAM handoff**: `/free`, then free VRAM verified via `/system_stats`. Only then
  does the judge VLM load — hard rule 3 applies to the doctor's own probes too.
- **Judge**: configured exactly as the nightly checker gets it (config.yaml's
  `judge` section with the nightly's timeout/retries, §0).
  Reachable, model installed, a **two-call color vision probe** (catches a judge
  that cannot actually see), then an **L2 dry run** — the real nightly call: 8
  frames of a synthetic 720x1280 clip + the rubric + the strict response schema,
  answered within that timeout (proves the context — `judge.num_ctx` on Ollama,
  the server's own setting on `openai_compat` — fits 8 images and the model build
  handles them; `WARN judge.latency` if over half the timeout — raise it there).
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
    readings and suggest a `gen_free_min_gb` between them — set that midpoint
    (not a value near the unloaded reading) under `policies.vram_handoff` in
    `pipeline.yaml` (a local change, §0) and re-run doctor. Doctor suggests one
    only when the unload freed at least 1 GB; otherwise the detail says `the
    unload freed too little VRAM … fix the judge unload first` — no threshold can
    work: check `ollama ps` / `nvidia-smi` (judge unload, other resident models)
    and re-run doctor.
  - No judge reply at all = the loaded reading proves nothing = FAIL.
- **Rewrite model** on the server the rewrite actually calls (`rewrite.base_url`,
  else the judge's Ollama, else local Ollama) — warning only.
- **Disk** headroom at `paths.runs_dir` (floor `disk_min_free_gb`, 20) ·
  **thresholds** state (`calibrated: false` is only a WARN here; `run` refuses it
  separately).

Writes `doctor.json` next to `config.yaml` (not gitignored — exclude it once, §0).
It is overwritten with `ok: false` (`status: "in progress"`) the moment doctor
starts; only a run that completes with no FAIL writes `ok: true` (`ALL CHECKS
PASSED`, exit 0; else exit 2). A crash or Ctrl-C rewrites it with `ok: false`
(`status: crashed` / `interrupted`) and a kill leaves the `in progress` placeholder —
never an older green snapshot.
WARNs (`comfy.queue`, `cron.path`, `rewrite.model`, `judge.latency`, `thresholds`
before sign-off) do not block `ok: true` — read them anyway. A busy queue is only a
WARN itself, but if it does not drain within `vram_handoff.wait_timeout_s` (180 s)
the handoff FAILs (`vram.handoff: ComfyUI queue busy`): run doctor with an empty
queue. **The nightly runner refuses to start without a passing snapshot**
(`--skip-doctor` exists for supervised debugging only) or with a stale one: it
fingerprints the exact bytes of `config.yaml`, `pipeline.yaml` and the workflow
JSON, plus `comfy.host` and `judge.adapter`/`model` — after any change (a comment
counts) re-run doctor. `thresholds.yaml` is not fingerprinted: sign-off (2d) needs
no re-run. A daytime diagnosis that must not touch
the snapshot: `.venv/bin/shortsloop doctor --out /tmp/doctor-diag.json`.

Anything doctor *could not verify* that the nightly depends on is a FAIL, not a
warning. `--no-free` (debugging only) skips `POST /free`, so the handoff stays
unverified and the snapshot is written `ok: false` — the runner refuses it. The
judge probes then run only if the GPU is already free (the VLM never loads beside
Wan). Run plain `.venv/bin/shortsloop doctor` before any unattended night.

## 2. Phase 0 — calibration (one afternoon, two hard gates)

Calibration does not check `doctor.json`: get §1 green first. The GPU steps (2a,
and 2c's judge wave) take the nightly runner's lock (§4): while a `shortsloop run`
holds it they refuse with exit 2, and vice versa. Both steps read pipeline.yaml the
same way the runner does; a malformed `policies:` block is a refusal (exit 2). Run
2a, 2c and supervised `run`s (§3) inside tmux/screen: nothing handles SIGHUP, so an
SSH disconnect kills them like SIGTERM — no report, and 2a's in-flight ComfyUI job
is left queued (2a cannot re-attach it: stop it with `comfy_client.py interrupt`, §4,
before re-running).

```bash
# 2a. ~40 draft-res clips (480x832, 81 frames), deliberately spanning good and bad
#     (~30-40 min GPU)
.venv/bin/shortsloop calibrate-batch --count 40
#     Refuses (exit 2) up front if config.yaml's judge could not be unloaded
#     (`openai_compat` without `judge.unload_url`) or another process holds the
#     run lock. Before the first Wan job it unloads the judge VLM + rewrite LLM,
#     /free's ComfyUI and VERIFIES free VRAM (pipeline.yaml vram_handoff) — exit 3
#     if a model is still resident (also: ComfyUI unreachable, submit rejected, a
#     job saving other than one video, 5 failed generations). One job at a time:
#     it waits for an empty ComfyUI queue before every submit and removes a
#     timed-out job before moving on. Resume-safe: Ctrl-C stops the in-flight
#     ComfyUI job, then re-run the same command; finished rows are skipped (re-run
#     too after a "FAILED … continuing" row, until the manifest has 44 rows). 4
#     prompt-swap rows (a clip shown with a DIFFERENT good prompt) are the
#     off-prompt ground truth. The manifest records absolute clip paths: do not
#     move the repo or calibration/ afterwards.

# 2b. label them — keyboard-only web UI, ~10-15 minutes
.venv/bin/shortsloop label
#     Binds 127.0.0.1 (no auth). Headless workstation: from your laptop
#       ssh -L 8765:127.0.0.1:8765 llm      # then open http://127.0.0.1:8765/
#     Prefer the tunnel: --host <LAN address> exposes the unauthenticated UI to
#     anyone who can reach it. Safari works (byte-range serving).
#     keys: 1 pass · 2 static · 3 deformed · 4 flicker · 5 off-prompt
#     · 6 other · space replay · p previous · n skip. One label per key press
#     (held keys ignored); "All clips labeled" only when none is left.
#     >>> GATE 1: this is YOUR judgment being encoded. <<<

# 2c. score the batch with the VLM judge, then tune
.venv/bin/shortsloop calibrate-tune --with-l2
#     The judge scores clips exactly as the nightly checker will: config.yaml's
#     judge with pipeline.yaml's judge timeout_s/retries. Exit 2, nothing judged:
#     no manifest, no `judge:` section (or config.yaml missing/unparseable), an
#     unusable pipeline.yaml, a judge it could not unload (`judge.adapter '<x>'
#     has no judge.unload_url`: `openai_compat` without it — or a misspelled
#     adapter, anything but ollama | openai_compat: fix the spelling, do not add
#     an unload_url; doctor's `judge.model` says `unknown judge adapter`), no
#     comfy.host, or another process holds the run lock. Judging starts only
#     after ComfyUI is /free'd and free VRAM is verified; the judge is unloaded
#     afterwards. Exit 3 (INFRA): a judge section missing base_url/model
#     (`INFRA: judge config needs base_url and model` — fix config.yaml, then
#     re-run doctor) or a misspelled adapter that does have an unload_url
#     (`unknown judge adapter`), judge unreachable / model not pulled, the VRAM
#     handoff failed, or the judge died mid-wave (scored clips are kept — re-run
#     to continue). Exit 1 with `RetryableJudgeError` in the traceback: the judge
#     hung (>20 s) or answered non-JSON on its model list, before or during the
#     wave — check Ollama / llama-swap and re-run; scored clips are kept.
#     Scores are keyed by clip+prompt sha and judge model+digest; a clip the
#     judge cannot evaluate is recorded (a runtime ERROR) and skipped on re-run —
#     to re-score after fixing the judge, move calibration/l2_scores.jsonl aside.
#     Tuning refuses (exit 2, writes NO proposal) with < 10 usable labels, all-pass
#     or all-fail labels, a held-out split missing a pass or a fail, or partial
#     judge coverage. Labels whose clip bytes changed since labeling are
#     excluded and listed. To label more: `calibrate-batch --count 56` (or more —
#     the first 40 rows are identical and skipped; only cal041+ are generated),
#     `label` (press n past clips already labeled), then 2c `--with-l2` again (it
#     scores only the new clips) and 2d.
#     read calibration/tuning_report.md: TEST agreement, FALSE-PASS = labeled-
#     fail clips that would ship / all labeled-fail clips (+Wilson CI),
#     false-fail, per-class confusion, per-layer catch attribution (L1 vs L2,
#     and what each would catch alone; clips the judge could not evaluate are
#     counted in their own "L2 ERROR" columns, never as L2 catches) — for the
#     combined L1+L2 decision. stdout must say `TEST (L1+L2)`, not `L1 only`.
#     ⚠️ Fallback floors — the report says `⚠️ **L2 FLOORS ARE A FALLBACK — …`,
#     stdout `[calibrate-tune] WARNING: L2 floors are a FALLBACK, not a fit — …`,
#     and provenance.l2_floors_fallback is non-null (the key is always there,
#     `null` when fine). Check: `grep -n FALLBACK calibration/tuning_report.md`
#     prints nothing = OK. A fallback means the judge fails more labeled-pass clips than the 20% false-fail cap under EVERY
#     floor (a dimension scored 1, or all N/A) — the clips are listed. The floors are then the least-bad combination
#     (no false-fail beyond those clips), not a fit. The judge disagrees with
#     your labels: re-check those clips, or escalate as below.
#     If the 8B judge disagrees with your labels too often, pull a ~32B VL
#     model, update config.yaml, re-run doctor (config changed), and re-run this
#     step: every clip is re-scored by the new judge and only its scores are
#     used; provenance names it. Measure the new judge's resident VRAM during the
#     wave (`ollama ps` / `nvidia-smi`): `vram_handoff.free_min_gb` — the only
#     check before the judge loads — must be at least that and at most the card.
#     Make every pipeline.yaml change at once, then re-run doctor after the last
#     edit and re-check judge.unload's two readings (gen_free_min_gb).
#     Without --with-l2 nothing is re-scored: the stored scores of config.yaml's
#     judge model are reused (any build; the latest digest written wins). Only
#     with none is the proposal L1-ONLY (untuned default L2 floors — see 2d).
#     After a judge change or re-pull always use --with-l2: floors tuned on an
#     older build make `run` refuse ("judge build … differs").

# 2d. sign off — THE gate that opens unattended running
.venv/bin/shortsloop calibrate-tune --approve
#     copies proposed thresholds → thresholds.yaml with calibrated: true, a
#     version above the current one (YYYY-MM-DD.N) + provenance (labels sha,
#     split, test stats, judge model+digest, L1 implementation fingerprint,
#     input shas). Refuses (exit 2) if labels, manifest or judge scores changed
#     after tuning, if the proposal's values were hand-edited after tuning or are
#     off the frozen shape, if labels.jsonl is gone, if the L1 metric code
#     (shortsloop/l1.py) changed since tuning (provenance.l1_impl — re-run 2c), if
#     config.yaml's judge adapter/model name is not the one the floors were tuned
#     on (approve does not compare the build digest — `run` does, §5), or if NO
#     judge scores are behind the proposal (L1-only: untuned L2 floors are an
#     uncalibrated instrument). It warns (and the marker stays in provenance)
#     when the L2 floors are a fallback (2c).
#     The value check is approve-time only: `run` would not notice a later hand
#     edit of thresholds.yaml — never edit it; re-run 2c + 2d.
#     Keep thresholds.yaml as an uncommitted local change (§0) and back up what a
#     re-tune needs outside the repo, in a NEW folder each time (a date-only name
#     is overwritten by a same-day re-sign-off; restore only to the same absolute
#     path):
B=/big/disk/backup/$(date -u +%Y%m%dT%H%M%SZ); mkdir -p "$B" && cp -a calibration config.yaml thresholds.yaml pipeline.yaml "$B"/
#
#     Supervised exception, not the normal path — an ALTERNATIVE to --approve
#     (never a follow-up: a second approve bumps the version), to sign off an
#     L1-only proposal. Deliberately commented out:
# .venv/bin/shortsloop calibrate-tune --approve --accept-untested-l2
#     Writes untuned default L2 floors with L1-only test numbers and a loud
#     marker: provenance.test_scope stays "L1 only — …" and approve records
#     provenance.l2_untested_accepted: true. The runner enforces it: such
#     thresholds count as UNCALIBRATED — `run` refuses without
#     --allow-uncalibrated, and supervised PASS clips go to
#     encoded_uncalibrated/, never encoded/. Replace it with a judge-backed
#     sign-off (2c + 2d) before any unattended night.
```

## 3. First supervised run

Produce a dispatch sheet with the `shorts-trend-dispatch` skill (~6 T2V clips
for the first night; intake rules and a CPU-only pre-check: checklist step 5), then:

```bash
.venv/bin/shortsloop run --dispatch /path/to/dispatch-YYYY-MM-DD.md
```

Until 2d, `run` refuses to start without `--allow-uncalibrated`; with it you can
rehearse: PASS clips then land in `encoded_uncalibrated/` (never `encoded/`) and the
report carries an UNCALIBRATED banner — a rehearsal, not a night's output. After
sign-off the flag changes nothing (PASS clips go to `encoded/`); never put it, or
`--skip-doctor`, in the crontab.

Watch the first one. Read `<run_dir>/report.md` (the runner's last stdout line,
`[shortsloop] <status>: n/m clips passed · report: …`, prints the path) against the
definition of done: per-clip verdicts + reasons, silent QC MP4s of passing clips
only, full `attempts.jsonl`, budget/cap evidence. Reproduce one clip from its
`attempts.jsonl` line to close the loop — from the repo root, while nothing holds
the GPU (the lock is free and the queue empty — check as in §4 — and `ollama ps`
empty: the vendored client takes no lock and does no VRAM handoff):

```bash
COMFY_HOST=<run.json comfy.host> python3 shortsloop/vendor/comfy_client.py run \
  -w <workflow_path> --prompt "$(cat <run_dir>/prompts/<clip_id>_a<n>.txt)" \
  --seed <seed> --width <w> --height <h> --length <length> --fps <fps> \
  [--steps <steps>] --out /tmp/repro-<n>        # a fresh, empty directory
sha256sum <workflow_path> /tmp/repro-<n>/*.mp4  # vs workflow_sha256 / output_sha256
```
Seed, size, length, fps and `steps` (present only when a re-roll set it — pass
`--steps` only then) come from `patch_args`; the prompt file holds the attempt's
`prompt_text`; `workflow_path` must still match `workflow_sha256`; the host is not
in `attempts.jsonl` — take it from `<run_dir>/run.json` → `comfy.host`. Omitting
`--seed` means random; omitting `--out` writes to `./outputs`. Client exit codes:
0 ok · 1 server unreachable or workflow unreadable/not API format · 2 execution or
usage error · 3 timeout · 4 submit rejected. Bit-identical GPU output is
UNVERIFIED-ON-GPU: record the result.

## 4. Unattended nightly

`crontab -e`, then — the job on **one line** (cron has no `\` continuation; a
literal `%` must be written `\%`):

```
# cron runs with a minimal PATH: use the `PATH=…` line doctor printed
# (the runner refuses to start without ffmpeg and ffprobe)
PATH=/usr/local/bin:/usr/bin:/bin
# 01:58 in the system time zone (timedatectl); wall-clock budget + retry caps bound
# the night (pipeline.yaml). mkdir first: a redirect into a missing directory means
# the job never runs at all; { …; } sends a failed cd to cron.log too.
58 1 * * * mkdir -p /big/disk/runs && { cd /path/to/shorts-generator && .venv/bin/shortsloop run --dispatch /path/to/tonight.md; } >> /big/disk/runs/cron.log 2>&1
```

Replace `tonight.md` daily — the runner does not know a sheet already ran (delete it
to skip a night: exit 2). At 01:58 ComfyUI and Ollama must be up, and no resume,
calibration, doctor or supervised run may still hold the GPU.

One GPU user per `paths.runs_dir`: a second `run` (or an overlapping
`--resume`) refuses with exit 2 while the first holds `<runs_dir>/.shortsloop.lock`
(e.g. `/big/disk/runs/.shortsloop.lock`, whatever `--runs-dir` says — the lock path
always comes from config.yaml's `paths.runs_dir`, so keep that absolute: a relative
one resolved from different working directories, cron vs. your shell, gives each
process its own lock file); the same lock is taken by `calibrate-batch`,
`calibrate-tune --with-l2` and doctor's GPU checks, so they can never share the card
with a night. It is a `flock`, released when the
holder exits (even if killed); never delete the file (a second process would lock a
fresh one beside it). **Do not run doctor while a night or calibration holds the
lock**: the job is unaffected, but doctor records `FAIL gpu.lock` and rewrites
`doctor.json` with `ok: false`, so the next night (and any `--resume`) refuses until a
plain doctor passes again.

Morning workflow: open `<run_dir>/report.md` (`ls -t <runs_dir>`; run ids are UTC)
→ review contact sheets and reasons → ship only clips in `encoded/` (never
`encoded_uncalibrated/`) → layer audio per the dispatch sheet's 音檔需求 table and
re-encode → upload (AI-content label on):

```bash
bash shortsloop/vendor/ig_encode.sh -o /big/disk/final/V1.mp4 -a bgm.mp3 \
  <run_dir>/clips/V1C1_a<n>.mp4 <run_dir>/clips/V1C2_a<n>.mp4   # playback order
```
`-o` and at least one input are required (else usage, exit 1). Use each clip's PASS
attempt from `clips/` (its `output_path` in `attempts.jsonl`); `encoded/<clip_id>.mp4`
works too but adds a lossy re-encode. Source audio is dropped; the BGM is
looped/trimmed to the video with a 1.5 s fade-out.

A halted run (`HALTED(…)`, exit 3) prints its exact resume command in `report.md`
(and `report.json` → `run.resume_command`), with the run's own
`--config/--pipeline/--thresholds` (relative to the repo) and any
`--allow-uncalibrated`/`--skip-doctor` it used — keep those flags. It starts with a
bare `shortsloop`; run it from the repo root as `.venv/bin/shortsloop …`, e.g.

```bash
.venv/bin/shortsloop run --dispatch <run_dir>/dispatch.md --resume <run_dir> \
  --config config.yaml --pipeline pipeline.yaml --thresholds thresholds.yaml
```

A run killed outright (SIGTERM, `kill -9`, OOM-killer, reboot, SSH hang-up) writes no
report: resume its `<run_dir>` by hand the same way, with the original flags (a cron
night has none). To stop a run cleanly use SIGINT → `HALTED(interrupted)` with a
report; never plain `kill`. Ctrl-C, or `pgrep -af 'shortsloop run'` and then
`kill -INT <PID>` with the PID of the `…/.venv/bin/python3 .venv/bin/shortsloop run …`
line — not cron's `/bin/sh -c` wrapper: a signal sent to the wrapper never reaches
the run (the shell holds it until the run ends on its own), so the night keeps going.
The in-flight ComfyUI job keeps running and resume re-attaches it — if you will
resume, do not interrupt it. Not resuming (an abandoned rehearsal or sheet)? Stop it
from the repo root with `COMFY_HOST=<comfy.host> python3
shortsloop/vendor/comfy_client.py interrupt`, then check the queue as below. If the fix
touched config.yaml, pipeline.yaml or the workflow, re-run doctor first, and finish
before 01:58 or that night refuses on the lock. Check before that doctor:
`flock -n <runs_dir>/.shortsloop.lock true && echo lock-free` prints `lock-free`, and
`COMFY_HOST=<comfy.host> python3 shortsloop/vendor/comfy_client.py queue` prints
`queue empty` (else `FAIL gpu.lock` / `FAIL vram.handoff`). On resume, attempts
already recorded in `attempts.jsonl` keep their outcome (a PASS is encoded, a FAIL
re-rolls) even if thresholds.yaml or the judge changed since — re-signing before a
resume does not re-judge them; a verdict written just before the stop but missing
from `attempts.jsonl` is reused only if clip, prompt, thresholds sha and judge digest
all match. In-flight ComfyUI jobs are re-attached before anything new is submitted,
generated-but-unjudged clips go straight to the judge, and the wall-clock budget
keeps counting from the time already spent. A different sheet is refused.

Disk retention: nothing is pruned — `clips/`, `encoded/` and `sheets/` accumulate
until the 20 GB floor refuses a night or stops it `COMPLETED(disk-stopped)`. Delete
only runs you will not `--resume`; after upload, old runs' `clips/` and `encoded/`
may go; keep `attempts.jsonl`, `events.jsonl`, `prompts/`, `run.json` and
`report.*` (small, needed to reproduce). Clean ComfyUI's own output folder
separately: the floor only watches `paths.runs_dir`.

## 5. Exit codes & failure playbook

| Symptom | Meaning / fix |
|---|---|
| `run` exits 2 before generating (`REFUSING TO RUN: …`) | Refusal, reason after the colon: config missing or lacking `comfy.host` / `comfy.workflow_t2v` / `judge.model` / doctor snapshot missing, failed (incl. an interrupted doctor or `FAIL gpu.lock`) or stale (`changed since doctor: …` — re-run doctor) / thresholds off-shape or uncalibrated (incl. `--accept-untested-l2`) / L1 code or judge mismatch (rows below) / `judge.adapter '<x>' has no judge.unload_url` (`openai_compat` without it, or a misspelled adapter: fix the spelling) / dispatch intake (per-row errors listed; also a missing sheet) / workflow knobs not patchable / ffmpeg or ffprobe not on PATH / disk below floor / another process holds the lock / `--resume` dir missing or a different sheet. Nothing was burned. |
| `run` exits 1 with a Python traceback | Not a refusal: config.yaml has the wrong shape (top level, `comfy:` or `judge:` not a mapping) or `runs_dir` cannot be created; or, with signed-off thresholds, the judge hung (>20 s) or returned non-JSON at the start-up build check (`RetryableJudgeError` in the traceback — check Ollama / llama-swap). Run doctor — it reports these as FAIL (`judge.model` for the judge). |
| `run` exits 3, report says HALTED(infra) | Instrument broke mid-run: ComfyUI unreachable (even at start — the run dir already exists), submit rejected, queue busy or a stuck job that won't clear, a job saving other than one video; VRAM not freed in either direction (a VLM/LLM still resident before a Wan wave counts); judge down or its build changed; 3 consecutive generation failures; `policies.judge.infra_escalation_after` (default 2) consecutive clip-scope judge errors; checker hung or contract violated. Nothing was generated or judged after the halt; clips that passed before it were still encoded. Fix (judge errors: L2 ERROR row below), then resume (§4). |
| `run` exits 3, report says HALTED(interrupted) / HALTED(crash) | Ctrl-C / `kill -INT`, or a runner exception (traceback on stderr). Same clean halt and resume command — resume (§4). |
| cron.log has no final `[shortsloop] <STATUS>: n/m clips passed · report: …` line for that night (and no `REFUSING TO RUN`), and there is no report.md | Killed outright (SIGTERM/SIGKILL, OOM-killer, reboot) — resume by hand (§4) — or cron never fired (`grep CRON /var/log/syslog`, or `journalctl -u cron` where there is no syslog file: a split entry, an unescaped `%`). If cron.log ends in a Python traceback instead, see the exit-1 row: nothing was created, so there is nothing to resume — fix, run doctor, start a fresh run (or let the next night run). |
| Report status `COMPLETED(budget-stopped)` / `COMPLETED(disk-stopped)` (exit 0) | A budget tripped: no new generation after it; everything already generated was judged, encoded and reported (`skipped` rows say which budget). |
| `run` exits 0 with failures in report | Working as designed: failures were caught, bounded, explained. Pass rate is a tuning metric, not an acceptance criterion. |
| HALTED(infra): `N consecutive clip-scope judge errors`, or clips ERROR at L2 | `doctor` → vision probe + 8-frame L2 dry run; timeouts → raise `policies.judge.timeout_s` (`judge.latency`), re-run doctor. Ollama vision broken ⇒ switch adapters as in §0 (`openai_compat` behind llama-swap, with `judge.unload_url`). When the judge answered but the reply was unusable, its raw text is kept next to the ERROR verdict (`*.l2_raw.json`); timeouts and an unreachable judge leave none. |
| OOM during generation | Runner already `/free`s and re-rolls, but each OOM uses one of the clip's attempts and 3 consecutive generation failures halt the night. If chronic, lower the sheet's 共用參數: `解析度` 480x832 and/or a shorter 4n+1 length (e.g. 49 frames; 81 is already the default, and a non-4n+1 length fails intake). |
| Report shows `encode failed after PASS` | Clip passed QC but the encode failed verification (size/fps/codec/aac/silence) or the clip bytes no longer match the verdict — nothing was left in `encoded/`; the raw file is kept in `<run_dir>/clips/`. |
| `run` refuses (exit 2): thresholds calibrated with another judge, `judge build … differs`, or `cannot identify the judge build` | L2 floors are tuned to one judge model and build (`thresholds.yaml` provenance.judge), checked at start before anything is created — an `ollama pull` that changes the digest blocks the next night. Re-pull that build (or start Ollama), or re-run Phase 0 for the new judge (`calibrate-tune --with-l2`, then `--approve`). |
| `run` exits 3, HALTED(infra): the judge build changed during the run | Re-checked every judge wave; nothing judged by the other build ships. Restore the calibrated build, then resume. |
| `run` refuses (exit 2): `thresholds were tuned against different L1 metric code (l1_impl …)` | `shortsloop/l1.py` changed since tuning (any byte, e.g. via `git pull`). Re-run `calibrate-tune` (L1 metrics are recomputed, stored judge scores reused), then `--approve`. |
| `reroll_action` contains `steps8 n/a` | The re-roll table's `--steps 8` is defined for the 4-step lightx2v build; your workflow runs another step count, so steps stay unchanged: flicker re-rolls only reseed, static ones still add the motion phrase. |

## 6. What is still UNVERIFIED-ON-GPU (first-night checklist)

Complete list with pass criteria (also: both Ollama tags pulled, timing within
budget): checklist step 9.

- [ ] `doctor` fully green on the workstation (`judge.vision`, `judge.l2_dryrun`,
      `judge.unload` all OK)
- [ ] `calibrate-batch` produces 40 playable draft clips (spot-check 2-3 with
      ffprobe: 480x832, 81 frames)
- [ ] judge wave fits in VRAM after `/free` (watch `nvidia-smi` once); on a night
      with a 2nd wave, `ollama ps` is empty and `nvidia-smi` shows no judge during
      wave 2's generation, and the night did not halt (the runner logs
      `"layer": "vram_handoff"` in events.jsonl only after its check passed; a
      failed check halts HALTED(infra))
- [ ] handoff thresholds suit the card: `vram_handoff.free_min_gb` (24) must leave
      the judge room; `doctor.json` → `judge.unload` `vram` shows loaded <
      `gen_free_min_gb` <= unloaded. Only if `judge.unload` FAILed, set
      `gen_free_min_gb` to the midpoint doctor suggests (§1) and re-run doctor
- [ ] real Wan clip L1 metrics are plausible: in
      `<run_dir>/verdicts/<clip_id>_a1.l1.json` (the runner writes one per attempt)
      read `l1.checks` (each rule's `value` against the signed-off `threshold`, and
      `pass`) for one clip you would label pass and one you would label fail;
      `l1.metrics` holds the raw numbers. The test fixtures are synthetic 480x832
      48-frame patterns, not a reference for real clips.
      By hand, from the repo root: `.venv/bin/shortsloop check <run_dir>/clips/<clip_id>_a1.mp4
      --prompt-file <run_dir>/prompts/<clip_id>_a1.txt --l1-only --json /tmp/l1.json`
      (`--prompt-file` is required; stdout is only the `VERDICT` line — PROCEED,
      never PASS — so read the metrics in the JSON)
- [ ] each job saves exactly one video whose spec matches the sheet (the supervised
      run, §3)
- [ ] a flicker re-roll and a static clip's 3rd attempt really run with
      `"steps": 8` in their `patch_args` (attempts.jsonl) and generate a usable clip
      (static's 2nd attempt adds only the motion phrase)
- [ ] `encoded/*.mp4` probes as h264 1080x1920 30/1 + aac (silent)
- [ ] reproduce one clip (§3) and record whether its sha256 matches `output_sha256`
- [ ] E2E: one real ~6-clip dispatch sheet unattended from cron (minimal PATH) →
      `cron.log` line → morning report (DoD)
- [ ] the next day the judge digest is unchanged (no Ollama auto-update): the next
      night is not refused with `judge build … differs`
