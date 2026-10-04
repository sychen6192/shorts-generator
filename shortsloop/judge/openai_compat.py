"""OpenAI-compatible adapter (llama.cpp llama-server with GGUF + mmproj, vLLM, ...).

Fallback path if Ollama's vision wiring is broken for the chosen model on the
workstation (docs/plan.md D10) — switching is a config change, not a rewrite."""

from __future__ import annotations

import json

from ..errors import InfraError
from .base import JudgeAdapter, RetryableJudgeError

# /v1/models entry fields that identify the loaded build, in digest order:
# `digest` (some servers), `root` (vLLM: weights path/repo), `owned_by` + `meta`
# (llama.cpp: GGUF n_params/size/vocab/ctx). NOT `created`: llama.cpp and vLLM stamp
# it with the request time, which would make the digest change on every call.
BUILD_FIELDS = ("digest", "root", "owned_by", "meta")


class OpenAICompatAdapter(JudgeAdapter):
    name = "openai_compat"

    def model_digest(self) -> str:
        models = self._get_json("/v1/models", timeout_s=20)
        data = models.get("data") if isinstance(models, dict) else None
        if not isinstance(data, list):
            raise InfraError("l2", f"/v1/models at {self.base_url} returned no model "
                                   f"list: {str(models)[:200]}")
        entries = [m for m in data if isinstance(m, dict)]
        ids = [m.get("id") for m in entries]
        match = [m for m in entries if m.get("id") == self.model]
        # Single-model servers (llama.cpp) often report one arbitrary id; accept it.
        entry = match[0] if match else (entries[0] if len(entries) == 1 else None)
        if entry is None:
            raise InfraError("l2", f"model {self.model!r} not served at {self.base_url} "
                                   f"(available: {ids})")
        parts = [f"openai-compat:{entry.get('id')}"]
        for key in BUILD_FIELDS:
            if entry.get(key) is not None:
                val = json.dumps(entry[key], sort_keys=True, separators=(",", ":"),
                                 ensure_ascii=False)
                parts.append(f"{key}={val}")
        return " ".join(parts)

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
            if res["choices"][0].get("finish_reason") == "length":
                raise RetryableJudgeError("openai-compat reply truncated at the token "
                                          "limit (finish_reason=length)")
            content_out = res["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, AttributeError):
            raise RetryableJudgeError("openai-compat response missing choices[0].message.content")
        if not isinstance(content_out, str) or not content_out.strip():
            raise RetryableJudgeError("openai-compat response has empty content")
        return content_out
