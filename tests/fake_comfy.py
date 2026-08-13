"""In-process fake ComfyUI server exercising the vendored client + our runner.

Scenario queue (one entry consumed per /prompt submission):
    {"fixture": "moving"}   -> job completes; /view serves that fixture's bytes
    {"error": "oom"}        -> history reports an execution_error (CUDA OOM text)
    {"hang": True}          -> history never appears (client times out)

Asserts hard rule 4 from the OUTSIDE: `violations` counts submissions that arrived
while another job was still unfinished.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.parse
import uuid
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

GB = 2 ** 30


def _extract(wf: dict) -> dict:
    """Pull prompt/seed/steps/geometry out of a patched API-format workflow."""
    info = {"prompt": None, "negative": None, "seed": None, "steps": None,
            "width": None, "height": None, "length": None}
    for node in wf.values():
        cls = node.get("class_type")
        ins = node.get("inputs", {})
        title = (node.get("_meta") or {}).get("title", "").lower()
        if cls == "CLIPTextEncode":
            if "positive" in title:
                info["prompt"] = ins.get("text")
            elif "negative" in title:
                info["negative"] = ins.get("text")
        if cls == "EmptyHunyuanLatentVideo":
            info["width"], info["height"] = ins.get("width"), ins.get("height")
            info["length"] = ins.get("length")
        if cls == "KSamplerAdvanced":
            info["seed"] = ins.get("noise_seed", info["seed"])
            info["steps"] = ins.get("steps", info["steps"])
    return info


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
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/system_stats":
            self._json({"system": {"comfyui_version": "fake"},
                        "devices": [{"name": "FakeGPU 5090",
                                     "vram_free": int(srv.vram_free_gb * GB),
                                     "vram_total": 32 * GB}]})
        elif parsed.path.startswith("/history/"):
            pid = parsed.path.rsplit("/", 1)[1]
            job = srv.jobs.get(pid)
            if job is None or job["scenario"].get("hang"):
                self._json({})
                return
            job["polls"] += 1
            if job["polls"] < 2:      # one "still running" poll before completion
                self._json({})
                return
            job["done"] = True
            if job["scenario"].get("error"):
                self._json({pid: {"status": {
                    "status_str": "error", "completed": False,
                    "messages": [["execution_error", {
                        "node_id": "4", "node_type": "KSamplerAdvanced",
                        "exception_type": "torch.OutOfMemoryError",
                        "exception_message": "CUDA out of memory (fake)"}]]}}})
                return
            fname = f"{pid}.mp4"
            self._json({pid: {
                "outputs": {"61": {"images": [
                    {"filename": fname, "subfolder": "", "type": "output"}]}},
                "status": {"status_str": "success", "completed": True,
                           "messages": []}}})
        elif parsed.path == "/view":
            q = urllib.parse.parse_qs(parsed.query)
            fname = q.get("filename", [""])[0]
            pid = fname.rsplit(".", 1)[0]
            job = srv.jobs.get(pid)
            data = b""
            if job is not None:
                fixture = job["scenario"].get("fixture", "moving")
                data = Path(srv.fixture_paths[fixture]).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif parsed.path == "/queue":
            running = [[0, pid] for pid, j in srv.jobs.items()
                       if not j["done"] and not j["scenario"].get("error")]
            self._json({"queue_running": running, "queue_pending": []})
        else:
            self._json({}, 404)

    def do_POST(self):
        srv = self.server
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/prompt":
            unfinished = [j for j in srv.jobs.values()
                          if not j["done"] and not j["scenario"].get("hang")]
            if unfinished:
                srv.violations += 1
            scenario = srv.scenarios.pop(0) if srv.scenarios else {"fixture": "moving"}
            pid = uuid.uuid4().hex
            srv.jobs[pid] = {"scenario": scenario, "polls": 0, "done": False,
                             "info": _extract(payload.get("prompt", {})),
                             "order": len(srv.submissions)}
            srv.submissions.append(pid)
            self._json({"prompt_id": pid, "number": len(srv.submissions)})
        elif self.path == "/free":
            srv.free_calls += 1
            srv.free_times.append(time.monotonic())
            if not srv.never_frees:
                srv.vram_free_gb = srv.vram_after_free_gb
            self._json({})
        elif self.path == "/interrupt":
            srv.interrupt_calls += 1
            self._json({})
        else:
            self._json({}, 404)


class FakeComfy(ThreadingHTTPServer):
    def __init__(self, fixture_paths: dict, scenarios: list[dict] | None = None,
                 vram_free_gb: float = 4.0, vram_after_free_gb: float = 28.0,
                 never_frees: bool = False, preloaded_jobs: dict | None = None):
        super().__init__(("127.0.0.1", 0), _Handler)
        self.fixture_paths = {k: str(v) for k, v in fixture_paths.items()}
        self.scenarios = list(scenarios or [])
        self.vram_free_gb = vram_free_gb
        self.vram_after_free_gb = vram_after_free_gb
        self.never_frees = never_frees
        self.jobs: dict[str, dict] = dict(preloaded_jobs or {})
        self.submissions: list[str] = []
        self.violations = 0
        self.free_calls = 0
        self.free_times: list[float] = []
        self.interrupt_calls = 0

    def handle_error(self, request, client_address):
        pass

    @property
    def host(self) -> str:
        return f"127.0.0.1:{self.server_address[1]}"

    def job_info(self, index: int) -> dict:
        """Extracted workflow info of the index-th submission."""
        return self.jobs[self.submissions[index]]["info"]


@contextmanager
def serve_comfy(**kwargs):
    srv = FakeComfy(**kwargs)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield srv
    finally:
        srv.shutdown()
        srv.server_close()
