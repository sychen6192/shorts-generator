"""off_prompt re-roll, attempt 2: ONE prompt rewrite by a local text LLM
(docs/plan.md §2.3). Narrow and fully logged: anchors preserved, recipe order
enforced, ≤110 words. A rewrite failure is NOT a verdict matter — the runner falls
back to a plain re-roll and logs that the rewrite was skipped."""

from __future__ import annotations

import difflib
import json
import re
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


def effective_rewrite_cfg(cfg: dict) -> dict:
    """config.yaml `rewrite:` with its server resolved: rewrite.base_url, else the
    judge's Ollama, else local Ollama (an openai_compat judge server does not
    speak Ollama's /api/chat)."""
    rw = dict(cfg.get("rewrite") or {})
    judge = cfg.get("judge") or {}
    if not rw.get("base_url"):
        rw["base_url"] = (judge.get("base_url")
                          if judge.get("adapter", "ollama") == "ollama" and
                          judge.get("base_url") else "http://127.0.0.1:11434")
    return rw


MAX_WORDS = 110
_TAIL_RE = re.compile(r"vertical\s+9:16\s+composition", re.I)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def validate_rewrite(original: str, rewritten: str, anchors: list[str] | None) -> list[str]:
    """The §2.3 contract, enforced in code: ≤110 words, the composition tail kept,
    and every sheet-anchor phrase present in the original kept verbatim."""
    problems = []
    words = len(rewritten.split())
    if words > MAX_WORDS:
        problems.append(f"{words} words > {MAX_WORDS}")
    if _TAIL_RE.search(original) and not _TAIL_RE.search(rewritten):
        problems.append("dropped the trailing 'vertical 9:16 composition'")
    orig, new = _norm(original), _norm(rewritten)
    for anchor in anchors or []:
        for phrase in re.split(r"[,;，；]", anchor):
            phrase = _norm(phrase)
            if len(phrase.split()) >= 2 and phrase in orig and phrase not in new:
                problems.append(f"dropped anchor phrase {phrase!r}")
    return problems


def rewrite_prompt(original: str, judge_reason: str, cfg: dict,
                   anchors: list[str] | None = None, timeout_s: float = 120) -> dict:
    """{"ok": True, "text", "diff"} or {"ok": False, "reason"} — a refused or
    unavailable rewrite falls back to a plain re-roll, and the reason is logged."""
    if not cfg or not cfg.get("enabled", True):
        return {"ok": False, "reason": "rewrite disabled in config"}
    base_url = str(cfg.get("base_url", "http://127.0.0.1:11434")).rstrip("/")
    model = cfg.get("model")
    if not model:
        return {"ok": False, "reason": "no rewrite.model configured"}
    payload = {
        "model": model,
        "stream": False,
        "format": REWRITE_SCHEMA,
        "options": {"temperature": 0.3, "seed": 7},
        "keep_alive": 0,          # hard rule 3: never resident into a Wan wave
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content":
                f"Judge complaint: {judge_reason}\n\n"
                + (f"Anchor phrases (keep verbatim): {' | '.join(anchors)}\n\n"
                   if anchors else "")
                + f"Original prompt:\n{original}"},
        ],
    }
    req = urllib.request.Request(
        base_url + "/api/chat", data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            body = json.loads(r.read())
        text = json.loads(body["message"]["content"])["rewritten_prompt"].strip()
    except Exception as e:  # noqa: BLE001 — any failure = plain re-roll, logged
        return {"ok": False, "reason": f"rewrite unavailable at {base_url}: "
                                       f"{type(e).__name__}: {e}"[:300]}
    if not text or text == original.strip():
        return {"ok": False, "reason": "rewrite empty or identical to the original"}
    problems = validate_rewrite(original, text, anchors)
    if problems:
        return {"ok": False, "reason": "rewrite broke the contract: " + "; ".join(problems)}
    diff = "\n".join(difflib.unified_diff(
        original.strip().splitlines(), text.splitlines(),
        fromfile="original", tofile="rewritten", lineterm=""))
    return {"ok": True, "text": text, "diff": diff}
