# shortsloop

Unattended nightly pipeline for Wan 2.2 short-video generation: consume a
dispatch sheet → generate clips on ComfyUI → **judge every clip automatically**
(deterministic L1 metrics + local VLM rubric) → re-roll failures within strict
budgets → encode only the survivors into silent 1080x1920 QC MP4s → wake up to a
morning report instead of a review queue.

```
shortsloop-check clip.mp4 --prompt-file prompt.txt --json verdict.json
# exit 0 PASS · 1 FAIL · 2 ERROR — the ONLY authority on clip quality

shortsloop doctor            # verify the workstation (writes doctor.json)
shortsloop calibrate-batch   # Phase 0: ~40 draft clips spanning good and bad
shortsloop label             # you label them (keyboard web UI, ~15 min)
shortsloop calibrate-tune    # tune thresholds; --approve = human sign-off gate
shortsloop run --dispatch d.md   # the nightly wave loop
shortsloop report runs/<id>      # re-render a morning report
```

Design in [`docs/plan.md`](docs/plan.md) (frozen contracts: verdict schema,
exit codes, failure classes, re-roll table, stage names), decision history in
[`docs/brainstorm.md`](docs/brainstorm.md), workstation setup in
[`docs/runbook.md`](docs/runbook.md), hard rules in [`CLAUDE.md`](CLAUDE.md).

Principles: the verdict comes from the checker's exit code, nothing else; fail
closed everywhere; Wan and the VLM never share VRAM (verified, not assumed);
bounded work with a loud report when budgets trip; every attempt reproducible
from append-only JSONL.

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -q   # 80+ tests, CPU-only: synthetic fixtures + fake ComfyUI/judge
```
