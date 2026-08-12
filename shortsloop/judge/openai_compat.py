"""OpenAI-compatible adapter (llama.cpp llama-server with GGUF + mmproj, vLLM, ...).

Fallback path if Ollama's vision wiring is broken for the chosen model on the
workstation (docs/plan.md D10) — switching is a config change, not a rewrite."""

from __future__ import annotations

from .base import JudgeAdapter, RetryableJudgeError


class OpenAICompatAdapter(JudgeAdapter):
    name = "openai_compat"

    def model_digest(self) -> str:
        models = self._get_json("/v1/models", timeout_s=20)
        ids = [m.get("id") for m in models.get("data", [])]
        # Single-model servers (llama.cpp) often report one arbitrary id; accept it.
        if self.model in ids or len(ids) == 1:
            return f"openai-compat:{ids[0] if len(ids) == 1 else self.model}"
        from ..errors import InfraError
        raise InfraError("l2", f"model {self.model!r} not served at {self.base_url} "
                               f"(available: {ids})")

    def judge(self, frames_jpeg, system, user, response_schema, timeout_s):
        content = [{"type": "text", "text": user}]
        for f in frames_jpeg:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{self._b64(f)}"},
            })
        payload = {
            "model": self.model,
            "temperature": float(self.cfg.get("temperature", 0)),
            "seed": int(self.cfg.get("seed", 7)),
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "shortsloop_l2", "strict": True,
                                "schema": response_schema},
            },
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
        }
        res = self._post_json("/v1/chat/completions", payload, timeout_s)
        try:
            content_out = res["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise RetryableJudgeError("openai-compat response missing choices[0].message.content")
        if not isinstance(content_out, str) or not content_out.strip():
            raise RetryableJudgeError("openai-compat response has empty content")
        return content_out
