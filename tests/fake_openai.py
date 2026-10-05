"""In-process fake OpenAI-compatible server (llama.cpp llama-server / vLLM shape) for
judge tests — GET /v1/models, POST /v1/chat/completions; records every payload."""

from __future__ import annotations

import itertools
import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from fake_judge import GOOD_SCORES

LLAMACPP_MODEL_ID = "Qwen3-VL-8B-Instruct-Q8_0.gguf"


def llamacpp_models_entry(model_id: str = LLAMACPP_MODEL_ID) -> dict:
    """One /v1/models entry as llama-server reports it: `created` is a per-request
    timestamp (not a build id); `owned_by` + `meta` describe the loaded GGUF."""
    return {
        "id": model_id,
        "object": "model",
        "owned_by": "llamacpp",
        "meta": {"vocab_type": 2, "n_vocab": 151936, "n_ctx_train": 262144,
                 "n_embd": 4096, "n_params": 8190735360, "size": 8709501952},
    }


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        srv = self.server
        if self.path == "/v1/models" and srv.scenario != "models_404":
            srv.models_calls += 1
            data = [dict(m, created=next(srv.clock)) for m in srv.models]
            self._json({"object": "list", "data": data})
        else:
            self._json({"error": {"message": "File Not Found", "code": 404}}, 404)

    def do_POST(self):
        srv = self.server
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        if self.path != "/v1/chat/completions" or srv.scenario == "chat_404":
            self._json({"error": {"message": "File Not Found", "code": 404}}, 404)
            return
        srv.chat_calls += 1
        srv.chat_payloads.append(payload)
        if srv.scenario == "empty_choices":
            self._json({"object": "chat.completion", "choices": []})
            return
        content = (srv.content_queue.pop(0) if srv.content_queue
                   else json.dumps(srv.scores or GOOD_SCORES))
        self._json({
            "object": "chat.completion",
            "model": payload.get("model"),
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}],
        })


class FakeOpenAI(ThreadingHTTPServer):
    """scenario: good | empty_choices | models_404 | chat_404."""

    def __init__(self, scenario="good", scores=None, models=None, content_queue=None):
        super().__init__(("127.0.0.1", 0), _Handler)
        self.scenario = scenario
        self.scores = scores
        self.models = list(models) if models is not None else [llamacpp_models_entry()]
        self.content_queue = list(content_queue or [])
        self.clock = itertools.count(1_760_000_000)   # `created` changes every call
        self.models_calls = 0
        self.chat_calls = 0
        self.chat_payloads: list[dict] = []

    def handle_error(self, request, client_address):
        pass


@contextmanager
def serve_openai(**kwargs):
    srv = FakeOpenAI(**kwargs)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield srv, f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()
        srv.server_close()
