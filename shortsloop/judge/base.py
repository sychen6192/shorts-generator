"""Adapter interface + HTTP plumbing shared by judge backends.

Error semantics (docs/plan.md D7 / §5 rows 1–3):
- server unreachable / model missing / HTTP 5xx  → InfraError (instrument broken,
  runner halts; no retry — a dead judge does not come back mid-call)
- timeout / invalid response content             → RetryableJudgeError (l2.run_l2
  retries once, then clip-scope ERROR)
"""

from __future__ import annotations

import base64
import json
import socket
import urllib.error
import urllib.request

from ..errors import InfraError


class RetryableJudgeError(Exception):
    """Judge answered badly (timeout, non-JSON, schema mismatch) — retry once."""


class JudgeAdapter:
    name = "base"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.base_url = str(cfg.get("base_url", "")).rstrip("/")
        self.model = cfg.get("model")
        if not self.base_url or not self.model:
            raise InfraError("l2", f"judge config needs base_url and model, got {cfg!r}")

    # -- to implement -------------------------------------------------------
    def model_digest(self) -> str:
        """Identify the exact model build; also serves as the reachability probe."""
        raise NotImplementedError

    def judge(self, frames_jpeg: list[bytes], system: str, user: str,
              response_schema: dict, timeout_s: float) -> str:
        """One judging call with all frames; returns the raw text content."""
        raise NotImplementedError

    # -- shared plumbing -----------------------------------------------------
    def _post_json(self, path: str, payload: dict, timeout_s: float) -> dict:
        req = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return self._do(req, timeout_s)

    def _get_json(self, path: str, timeout_s: float) -> dict:
        req = urllib.request.Request(self.base_url + path, method="GET")
        return self._do(req, timeout_s)

    def _do(self, req: urllib.request.Request, timeout_s: float) -> dict:
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as r:
                body = r.read()
        except (socket.timeout, TimeoutError):
            raise RetryableJudgeError(f"judge call timed out after {timeout_s}s")
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            raise InfraError("l2", f"judge HTTP {e.code} at {req.full_url}: {detail}")
        except urllib.error.URLError as e:
            raise InfraError("l2", f"judge unreachable at {self.base_url} ({e.reason})")
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            raise RetryableJudgeError("judge returned non-JSON transport body")

    @staticmethod
    def _b64(data: bytes) -> str:
        return base64.b64encode(data).decode("ascii")


def make_adapter(judge_cfg: dict) -> JudgeAdapter:
    from .ollama import OllamaAdapter
    from .openai_compat import OpenAICompatAdapter

    kind = judge_cfg.get("adapter", "ollama")
    if kind == "ollama":
        return OllamaAdapter(judge_cfg)
    if kind == "openai_compat":
        return OpenAICompatAdapter(judge_cfg)
    raise InfraError("l2", f"unknown judge adapter {kind!r} (ollama | openai_compat)")
