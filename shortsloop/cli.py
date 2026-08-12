"""shortsloop — umbrella CLI. Subcommands land milestone by milestone (docs/plan.md §7)."""

from __future__ import annotations

import sys

USAGE = """\
shortsloop <command> [args]

commands:
  check            run the two-layer checker on one clip (alias: shortsloop-check)
  run              nightly wave runner over a dispatch sheet        [lands in M3]
  label            Phase 0 labeling web UI                          [lands in M4]
  calibrate-batch  generate the draft-res calibration batch         [lands in M4]
  calibrate-tune   tune thresholds against labels, write report     [lands in M4]
  doctor           verify the workstation environment               [lands in M5]
  report           re-render report.md from a run directory         [lands in M3]
"""

_PENDING = {
    "run": "M3", "report": "M3",
    "label": "M4", "calibrate-batch": "M4", "calibrate-tune": "M4",
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
    if cmd in _PENDING:
        print(f"shortsloop {cmd}: not implemented yet — lands in {_PENDING[cmd]} "
              f"(see docs/plan.md §7)", file=sys.stderr)
        return 2
    print(f"shortsloop: unknown command {cmd!r}\n\n{USAGE}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
