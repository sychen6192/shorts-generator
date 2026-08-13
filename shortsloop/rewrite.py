"""off_prompt re-roll, attempt 2: ONE prompt rewrite by a local text LLM
(docs/plan.md §2.3). Narrow and fully logged: anchors preserved, recipe order
enforced, ≤110 words. A rewrite failure is NOT a verdict matter — the runner falls
back to a plain re-roll and logs that the rewrite was skipped."""

from __future__ import annotations

import difflib
import json
import urllib.error
import urllib.request

REWRITE_SCHEMA = {
    "type": "object",
    "properties": {"rewritten_prompt": {"type": "string"}},
    "required": ["rewritten_prompt"],
}

SYSTEM = """\
You rewrite text-to-video prompts that a judge flagged as not matching their output.
Rules: keep the subject and any style/scene anchor phrases word-for-word; keep the
order subject -> action/motion -> camera movement -> scene & lighting -> style; make
the named action and camera move more concrete and visually unambiguous; at most 110
words; keep the trailing 'vertical 9:16 composition' phrase if present. Return JSON
only: {"rewritten_prompt": "..."}."""


def rewrite_prompt(original: str, judge_reason: str, cfg: dict,
                   timeout_s: float = 120) -> tuple[str, str] | None:
    """Returns (rewritten_prompt, unified_diff) or None if the rewrite could not be
    obtained (caller falls back to plain re-roll)."""
    if not cfg or not cfg.get("enabled", True):
        return None
    base_url = str(cfg.get("base_url", "http://127.0.0.1:11434")).rstrip("/")
    model = cfg.get("model")
    if not model:
        return None
    payload = {
        "model": model,
        "stream": False,
        "format": REWRITE_SCHEMA,
        "options": {"temperature": 0.3, "seed": 7},
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content":
                f"Judge complaint: {judge_reason}\n\nOriginal prompt:\n{original}"},
        ],
    }
    req = urllib.request.Request(
        base_url + "/api/chat", data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            body = json.loads(r.read())
        text = json.loads(body["message"]["content"])["rewritten_prompt"].strip()
    except Exception:
        return None
    if not text or text == original.strip():
        return None
    diff = "\n".join(difflib.unified_diff(
        original.strip().splitlines(), text.splitlines(),
        fromfile="original", tofile="rewritten", lineterm=""))
    return text, diff
