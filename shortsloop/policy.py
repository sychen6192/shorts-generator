"""Failure classes and the re-roll policy table (docs/plan.md §2.3 — FROZEN).

The checker maps failed L1 checks / L2 dimensions to failure classes; the runner maps
the highest-priority class to a re-roll action. Verdict authority stays with the
checker (hard rule 1) — this module never looks at pixels.
"""

from __future__ import annotations

# Priority order: first match drives the re-roll action.
CLASS_PRIORITY = ["broken", "black", "static", "deformed", "flicker", "off_prompt", "blurry"]

# L1 check name -> failure class.
L1_CHECK_CLASS = {
    "spec": "broken",
    "motion": "static",
    "freeze": "static",
    "flicker": "flicker",
    "ssim_floor": "flicker",
    "sharpness": "blurry",
    "black": "black",
    "exposure": "black",
}

# L2 dimension -> failure class.
L2_DIM_CLASS = {
    "prompt_adherence": "off_prompt",
    "subject_consistency": "deformed",
    "anatomy_artifacts": "deformed",
    "temporal_coherence": "flicker",
    "imaging_quality": "blurry",
}

# Motion-strengthening phrase bank for `static` re-rolls (docs/plan.md §2.6).
# Versioned: changing this list is a policy change, note it in the plan.
MOTION_PHRASES = [
    "; continuous visible motion, slow dolly-in, subject in constant movement",
    "; dynamic action throughout, camera slowly orbiting the subject",
    "; sweeping camera movement, handheld tracking shot, energetic motion",
]

# The trailing composition phrase dispatch prompts end with (motion phrases are
# inserted before it when present).
COMPOSITION_TAIL = "vertical 9:16 composition"


def failure_classes(failed_l1_checks: list[str], failed_l2_dims: list[str]) -> list[str]:
    """Ordered, deduplicated failure classes from failed check/dimension names."""
    classes = set()
    for name in failed_l1_checks:
        cls = L1_CHECK_CLASS.get(name)
        if cls:
            classes.add(cls)
    for name in failed_l2_dims:
        cls = L2_DIM_CLASS.get(name)
        if cls:
            classes.add(cls)
    return sorted(classes, key=CLASS_PRIORITY.index)


def append_motion_phrase(prompt: str, attempt_index: int) -> str:
    """Insert a motion phrase (deterministic pick) before the composition tail."""
    phrase = MOTION_PHRASES[attempt_index % len(MOTION_PHRASES)]
    stripped = prompt.rstrip()
    tail_pos = stripped.lower().rfind(COMPOSITION_TAIL)
    if tail_pos > 0:
        head = stripped[:tail_pos].rstrip().rstrip(",;")
        tail = stripped[tail_pos:]
        return f"{head}{phrase}, {tail}"
    return f"{stripped}{phrase}"
