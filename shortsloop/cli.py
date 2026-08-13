"""shortsloop — umbrella CLI. Subcommands land milestone by milestone (docs/plan.md §7)."""

from __future__ import annotations

import sys

USAGE = """\
shortsloop <command> [args]

commands:
  check            run the two-layer checker on one clip (alias: shortsloop-check)
  run              nightly wave runner over a dispatch sheet
  report           re-render report.md from a run directory's report.json
  label            Phase 0 labeling web UI (keyboard-only, resumable)
  calibrate-batch  generate the draft-res calibration batch (workstation)
  calibrate-tune   tune thresholds against labels; --approve = sign-off gate
  doctor           verify the workstation environment               [lands in M5]
"""

_PENDING = {
    "doctor": "M5",
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "check":
        from .check import main as check_main
        return check_main(rest)
    if cmd == "run":
        from .runner import main_run
        return main_run(rest)
    if cmd == "report":
        from .report import main_report
        return main_report(rest)
    if cmd == "label":
        from .label import main_label
        return main_label(rest)
    if cmd == "calibrate-batch":
        from .calibrate.batchgen import main_batch
        return main_batch(rest)
    if cmd == "calibrate-tune":
        from .calibrate.tune import main_tune
        return main_tune(rest)
    if cmd in _PENDING:
        print(f"shortsloop {cmd}: not implemented yet — lands in {_PENDING[cmd]} "
              f"(see docs/plan.md §7)", file=sys.stderr)
        return 2
    print(f"shortsloop: unknown command {cmd!r}\n\n{USAGE}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
