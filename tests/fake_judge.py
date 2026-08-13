"""In-process fake Ollama server for judge tests — scenario-driven misbehavior."""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from shortsloop.l2 import DIMENSIONS

GOOD_SCORES = {d: {"score": 5, "na": False, "reason": "clean in all frames"}
               for d in DIMENSIONS}


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
        if self.path == "/api/tags":
            self._json({"models": [{"name": self.server.model,
                                    "digest": "sha256:fakedigest"}]})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/api/chat":
            messages = payload.get("messages") or [{}, {}]
            is_vision = any("images" in m for m in messages)
            if not is_vision:                      # text-only = rewrite request
                self.server.rewrite_calls += 1
                self.server.last_rewrite_payload = payload
                self._json({"message": {"role": "assistant",
                                        "content": json.dumps(
                                            self.server.rewrite_response)},
                            "done": True})
                return
            self.server.chat_calls += 1
            self.server.chat_times.append(time.monotonic())
            self.server.last_chat_payload = payload
            if self.server.sleep_s:
                time.sleep(self.server.sleep_s)
            self._json({"message": {"role": "assistant",
                                    "content": self.server.content()},
                        "done": True})
        elif self.path == "/api/generate":  # unload request
            self.server.generate_calls += 1
            self._json({"done": True})
        else:
            self._json({"error": "not found"}, 404)


class FakeOllama(ThreadingHTTPServer):
    def __init__(self, scenario="good", scores=None, scores_queue=None,
                 model="qwen3-vl:8b-instruct", sleep_s=0.0,
                 rewrite_response=None):
        super().__init__(("127.0.0.1", 0), _Handler)
        self.scenario = scenario
        self.scores = scores
        self.scores_queue = list(scores_queue or [])
        self.model = model
        self.sleep_s = sleep_s
        self.rewrite_response = rewrite_response or {
            "rewritten_prompt": "REWRITTEN: a red cube sliding fast across a dark "
                                "slate table; slow dolly-in, vertical 9:16 composition"}
        self.chat_calls = 0
        self.chat_times: list[float] = []
        self.generate_calls = 0
        self.rewrite_calls = 0
        self.last_chat_payload = None
        self.last_rewrite_payload = None
        self.probe_answers: list[str] = []

    def handle_error(self, request, client_address):
        pass  # client-side timeouts close sockets mid-write; keep test output clean

    def content(self) -> str:
        if self.probe_answers:
            return json.dumps({"dominant_color": self.probe_answers.pop(0)})
        if self.scenario == "garbage":
            return "the clip looks fine to me, PASS!"
        if self.scenario == "missing_dim":
            partial = {k: v for k, v in GOOD_SCORES.items() if k != "imaging_quality"}
            return json.dumps(partial)
        if self.scenario == "na_abuse":
            scores = {k: dict(v) for k, v in (self.scores or GOOD_SCORES).items()}
            scores["anatomy_artifacts"]["na"] = True  # na not allowed for this dim
            return json.dumps(scores)
        if self.scores_queue:
            return json.dumps(self.scores_queue.pop(0))
        return json.dumps(self.scores or GOOD_SCORES)


@contextmanager
def serve(**kwargs):
    srv = FakeOllama(**kwargs)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield srv, f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()
        srv.server_close()
