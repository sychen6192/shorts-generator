"""Ollama-native adapter: /api/chat with images + server-side JSON-schema `format`.

temperature 0 + fixed seed for near-determinism; keep_alive is the wave-level VRAM
knob (the runner passes 0 on the last call of a judge wave so the VLM unloads —
hard rule 3)."""

from __future__ import annotations

from ..errors import InfraError
from .base import JudgeAdapter


class OllamaAdapter(JudgeAdapter):
    name = "ollama"

    def model_digest(self) -> str:
        tags = self._get_json("/api/tags", timeout_s=20)
        for m in tags.get("models", []):
            if m.get("name") == self.model or m.get("model") == self.model:
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
        content = (res.get("message") or {}).get("content")
        if not isinstance(content, str) or not content.strip():
            from .base import RetryableJudgeError
            raise RetryableJudgeError("ollama response has no message.content")
        return content

    def unload(self, timeout_s: float = 60) -> None:
        """Ask ollama to drop the model from VRAM now (keep_alive=0, empty prompt)."""
        self._post_json("/api/generate",
                        {"model": self.model, "keep_alive": 0}, timeout_s)
