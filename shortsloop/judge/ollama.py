"""Ollama-native adapter: /api/chat with images + server-side JSON-schema `format`.

temperature 0 + fixed seed for near-determinism; keep_alive is the wave-level VRAM
knob (the runner passes 0 on the last call of a judge wave so the VLM unloads —
hard rule 3)."""

from __future__ import annotations

from ..errors import InfraError
from .base import JudgeAdapter


def normalize_model_name(name) -> str:
    """Ollama's canonical form: a name without a tag means `<name>:latest`. The tag is
    the part after ':' in the LAST path segment (a registry host may carry a port)."""
    name = str(name or "").strip()
    if name and ":" not in name.rsplit("/", 1)[-1]:
        name += ":latest"
    return name


class OllamaAdapter(JudgeAdapter):
    name = "ollama"

    def model_digest(self) -> str:
        tags = self._get_json("/api/tags", timeout_s=20)
        models = tags.get("models") if isinstance(tags, dict) else None
        if not isinstance(models, list):
            raise InfraError("l2", f"ollama /api/tags at {self.base_url} returned no "
                                   f"model list: {str(tags)[:200]}")
        want = normalize_model_name(self.model)
        for m in models:
            if not isinstance(m, dict):
                continue
            if want in (normalize_model_name(m.get("name")),
                        normalize_model_name(m.get("model"))):
                return m.get("digest", "unknown")
        raise InfraError(
            "l2",
            f"model {self.model!r} not installed on ollama at {self.base_url} "
            f"(ollama pull {self.model})",
        )

    def judge(self, frames_jpeg, system, user, response_schema, timeout_s):
        payload = {
            "model": self.model,
            "stream": False,
            "format": response_schema,
            "options": {
                "temperature": float(self.cfg.get("temperature", 0)),
                "seed": int(self.cfg.get("seed", 7)),
                "num_ctx": int(self.cfg.get("num_ctx", 16384)),
            },
            "keep_alive": self.cfg.get("keep_alive", self.cfg.get("keep_alive_wave", "10m")),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user,
                 "images": [self._b64(f) for f in frames_jpeg]},
            ],
        }
        res = self._post_json("/api/chat", payload, timeout_s)
        if res.get("done_reason") == "length":
            from .base import RetryableJudgeError
            raise RetryableJudgeError("ollama reply truncated at the token limit "
                                      "(done_reason=length) — not a final answer")
        content = (res.get("message") or {}).get("content")
        if not isinstance(content, str) or not content.strip():
            from .base import RetryableJudgeError
            raise RetryableJudgeError("ollama response has no message.content")
        return content

    def unload(self, timeout_s: float = 60) -> None:
        """Ask ollama to drop the model from VRAM now (keep_alive=0, empty prompt)."""
        self._post_json("/api/generate",
                        {"model": self.model, "keep_alive": 0}, timeout_s)
