"""Error taxonomy (docs/plan.md D7): clip-scope vs infra-scope, both never PASS.

- ClipError:  this clip could not be evaluated (corrupt file, undecodable, judge
  returned garbage for this clip after retry). The runner counts a failed attempt and
  re-rolls within the cap; the clip never ships.
- InfraError: the instrument is broken (judge unreachable, VRAM not freed, thresholds
  missing/invalid, ffprobe absent). The runner halts the run and reports.
"""


class CheckError(Exception):
    scope = "clip"

    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage  # "probe" | "l1" | "l2"
        self.message = message

    def as_error_obj(self) -> dict:
        return {"scope": self.scope, "stage": self.stage, "message": self.message}


class ClipError(CheckError):
    scope = "clip"


class InfraError(CheckError):
    scope = "infra"
