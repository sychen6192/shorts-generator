"""ComfyUI integration: generation via the vendored client (subprocess, RESULT-line
contract), VRAM handoff via direct HTTP (docs/plan.md §4, hard rules 3+4).

The runner is strictly sequential — one generate() at a time — and the fake server
used in tests asserts that no second job is ever submitted while one runs.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from .errors import InfraError

VENDOR_CLIENT = Path(__file__).parent / "vendor" / "comfy_client.py"


class GenerationFailed(Exception):
    """One generation attempt failed (exec error / timeout / no files).
    kind ∈ exec_error | timeout | bad_workflow | no_files."""

    def __init__(self, kind: str, detail: str):
        super().__init__(f"{kind}: {detail}")
        self.kind = kind
        self.detail = detail


class ComfyClient:
    def __init__(self, host: str, workflow: str | Path, timeout_s: int = 1800,
                 poll_s: int = 5):
        self.host = host
        self.workflow = str(workflow)
        self.timeout_s = int(timeout_s)
        self.poll_s = int(poll_s)

    # ---- generation (vendored client subprocess) ---------------------------
    def _run_client(self, args: list[str], timeout_s: float) -> subprocess.CompletedProcess:
        cmd = [sys.executable, str(VENDOR_CLIENT), *args]
        env = {**os.environ, "COMFY_HOST": self.host}
        try:
            return subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=timeout_s, env=env)
        except subprocess.TimeoutExpired:
            raise GenerationFailed("timeout", f"client process exceeded {timeout_s}s")

    @staticmethod
    def _parse_result(stdout: str) -> dict | None:
        for line in reversed(stdout.splitlines()):
            if line.startswith("RESULT "):
                try:
                    return json.loads(line[len("RESULT "):])
                except json.JSONDecodeError:
                    return None
        return None

    def submit(self, *, prompt: str, seed: int, width: int, height: int,
               length: int, steps: int | None = None) -> dict:
        """Queue one job (returns fast). {"prompt_id", "seed"}. The submitted
        prompt_id is logged BEFORE waiting so a crash mid-generation can re-attach
        on resume. InfraError on unreachable server / rejected workflow (neither
        can succeed on retry — halt)."""
        args = ["submit", "-w", self.workflow, "--prompt", prompt,
                "--seed", str(seed), "--width", str(width), "--height", str(height),
                "--length", str(length)]
        if steps is not None:
            args += ["--steps", str(steps)]
        res = self._run_client(args, timeout_s=300)
        combined = (res.stdout or "") + (res.stderr or "")
        if "cannot reach http" in combined:
            raise InfraError("l1", f"ComfyUI unreachable at {self.host}")
        result = self._parse_result(res.stdout or "")
        if res.returncode == 0 and result and result.get("ok") and result.get("prompt_id"):
            return result
        if res.returncode == 4:
            raise InfraError("l1", f"workflow rejected by ComfyUI: {combined[-800:]}")
        raise GenerationFailed("exec_error", f"submit failed: {combined[-800:]}")

    def wait(self, prompt_id: str, out_dir: str | Path) -> dict:
        """Wait for a queued job and download outputs. {"files", "elapsed_sec"}.
        GenerationFailed on exec error / timeout / empty outputs."""
        res = self._run_client(["wait", prompt_id, "--out", str(out_dir),
                                "--timeout", str(self.timeout_s),
                                "--poll", str(self.poll_s)],
                               timeout_s=self.timeout_s + 120)
        combined = (res.stdout or "") + (res.stderr or "")
        if "cannot reach http" in combined:
            raise InfraError("l1", f"ComfyUI unreachable at {self.host}")
        result = self._parse_result(res.stdout or "")
        if res.returncode == 0 and result and result.get("ok") and result.get("files"):
            return result
        kind = {2: "exec_error", 3: "timeout"}.get(res.returncode, "no_files")
        raise GenerationFailed(kind, combined[-800:])

    # ---- raw HTTP (handoff / recovery) --------------------------------------
    def _http(self, method: str, path: str, payload: dict | None = None,
              timeout_s: float = 30) -> dict:
        req = urllib.request.Request(
            f"http://{self.host}{path}",
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Content-Type": "application/json"} if payload is not None else {},
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as r:
                body = r.read()
        except (urllib.error.URLError, TimeoutError) as e:
            reason = getattr(e, "reason", e)
            raise InfraError("l1", f"ComfyUI unreachable at {self.host} ({reason})")
        return json.loads(body) if body else {}

    def free(self) -> None:
        self._http("POST", "/free", {"unload_models": True, "free_memory": True})

    def interrupt(self) -> None:
        self._http("POST", "/interrupt", {})

    def vram_free_gb(self) -> float:
        stats = self._http("GET", "/system_stats")
        devices = stats.get("devices") or []
        if not devices:
            raise InfraError("l1", "/system_stats reports no GPU devices")
        return devices[0].get("vram_free", 0) / 2 ** 30

    def vram_handoff(self, free_min_gb: float, wait_timeout_s: float,
                     poll_s: float = 2.0, sleep=time.sleep) -> float:
        """Hard rule 3, enforced and VERIFIED: /free, then poll /system_stats until
        the GPU actually has room for the judge. InfraError (halt) on timeout."""
        self.free()
        deadline = time.monotonic() + wait_timeout_s
        last = self.vram_free_gb()
        while last < free_min_gb:
            if time.monotonic() >= deadline:
                raise InfraError(
                    "l2",
                    f"VRAM handoff failed: {last:.1f} GB free after {wait_timeout_s}s, "
                    f"need {free_min_gb} GB — Wan did not unload; judging would OOM")
            sleep(poll_s)
            last = self.vram_free_gb()
        return last
