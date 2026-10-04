"""ComfyUI integration: generation via the vendored client (subprocess, RESULT-line
contract), queue hygiene + VRAM handoff via direct HTTP (docs/plan.md §4, hard
rules 3+4).

The runner is strictly sequential — one job at a time — and verifies it from the
outside: a job that timed out or failed is interrupted/deleted and confirmed gone
before anything else is submitted, and the queue must be empty before a submit.
"""

from __future__ import annotations

import http.client
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from .errors import InfraError

VENDOR_CLIENT = Path(__file__).parent / "vendor" / "comfy_client.py"
VIDEO_EXTS = (".mp4", ".webm", ".mov", ".mkv")
_PATCH_LINE = re.compile(r"^\[patch\] \S+?\([^)]*\)\.(\w+) = ")
# Knobs every attempt relies on; a workflow that wires them from link inputs is
# silently NOT patched by the vendored client (re-rolls would repeat a seed).
REQUIRED_KNOBS = {"text": "prompt", "seed": "seed", "width": "width",
                  "height": "height", "length": "length"}


class GenerationFailed(Exception):
    """One generation attempt failed (exec error / timeout / client error).
    kind ∈ exec_error | timeout | client_error | lost."""

    def __init__(self, kind: str, detail: str):
        super().__init__(f"{kind}: {detail}")
        self.kind = kind
        self.detail = detail


def error_summary(stdout: str, stderr: str, limit: int = 300) -> str:
    """The lines that say WHY a client call failed (not progress chatter), as one
    table-safe line for attempts.jsonl and the report."""
    lines = [ln.strip() for ln in (stderr or "").splitlines() if ln.strip()]
    keep = [ln for ln in lines if re.search(r"error|interrupt|rejected|timeout|"
                                            r"cannot|memory|node \d+", ln, re.I)]
    if not keep:
        keep = lines[-3:] or [ln.strip() for ln in (stdout or "").splitlines()
                              if ln.strip() and not ln.startswith("RESULT ")][-2:]
    text = " / ".join(keep)
    text = re.sub(r"\s+", " ", text).replace("|", "¦")
    return text[:limit] or "(no output)"


class ComfyClient:
    def __init__(self, host: str, workflow: str | Path, timeout_s: int = 1800,
                 poll_s: int = 5):
        self.host = host
        self.workflow = str(workflow)
        self.timeout_s = int(timeout_s)
        self.poll_s = int(poll_s)

    # ---- vendored client subprocess ------------------------------------------
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

    @staticmethod
    def _patched_knobs(stdout: str) -> set[str]:
        knobs = set()
        for line in stdout.splitlines():
            m = _PATCH_LINE.match(line)
            if m:
                key = m.group(1)
                knobs.add("seed" if key in ("seed", "noise_seed") else key)
        return knobs

    def _patch_args(self, *, prompt, seed, width, height, length, fps=None,
                    steps=None) -> list[str]:
        args = ["-w", self.workflow, "--prompt", prompt, "--seed", str(seed),
                "--width", str(width), "--height", str(height),
                "--length", str(length)]
        if fps is not None:
            args += ["--fps", f"{fps:g}"]
        if steps is not None:
            args += ["--steps", str(steps)]
        return args

    def _missing_knobs(self, stdout: str, steps: int | None) -> list[str]:
        patched = self._patched_knobs(stdout)
        want = dict(REQUIRED_KNOBS)
        if steps is not None:
            want["steps"] = "steps"
        return [label for key, label in want.items() if key not in patched]

    def validate_workflow(self, *, width, height, length, fps=None,
                          steps: int | None = 8) -> list[str]:
        """Dry run (--dump, nothing submitted): the workflow loads as API format and
        every knob the runner and its re-roll table turn is actually patchable.
        Returns problems (empty = ok)."""
        try:
            res = self._run_client(
                ["run", *self._patch_args(prompt="shortsloop workflow check", seed=1,
                                          width=width, height=height, length=length,
                                          fps=fps, steps=steps), "--dump"],
                timeout_s=60)
        except GenerationFailed as e:
            return [f"workflow dry run failed: {e.detail}"]
        if res.returncode != 0:
            return [f"workflow unusable: {error_summary(res.stdout, res.stderr)}"]
        missing = self._missing_knobs(res.stdout, steps)
        return [f"workflow does not expose {', '.join(missing)} as literal inputs "
                f"the client can patch (linked from another node?) — re-rolls and "
                f"spec expectations would silently not apply"] if missing else []

    def submit(self, *, prompt: str, seed: int, width: int, height: int,
               length: int, fps: float | None = None, steps: int | None = None) -> dict:
        """Queue one job (returns fast). {"prompt_id", "seed"}. Every failure here
        is the instrument's, never the clip's: unreachable server, unreadable /
        UI-format workflow, node validation rejecting it -> InfraError (halt)."""
        res = self._run_client(
            ["submit", *self._patch_args(prompt=prompt, seed=seed, width=width,
                                         height=height, length=length, fps=fps,
                                         steps=steps)],
            timeout_s=300)
        result = self._parse_result(res.stdout or "")
        if res.returncode == 0 and result and result.get("ok") and result.get("prompt_id"):
            missing = self._missing_knobs(res.stdout or "", steps)
            if missing:
                raise InfraError("l1", f"job {result['prompt_id']} was submitted without "
                                       f"patching {', '.join(missing)} — workflow knobs "
                                       f"are linked inputs (hard rule 6: irreproducible)")
            return result
        raise InfraError("l1", f"ComfyUI submit failed (exit {res.returncode}): "
                               f"{error_summary(res.stdout, res.stderr)}")

    def wait(self, prompt_id: str, out_dir: str | Path) -> dict:
        """Wait for a queued job and download outputs. {"files": [the one video],
        "elapsed_sec"}. GenerationFailed on exec error / timeout; InfraError when
        the server is gone or the workflow's outputs are not exactly one video."""
        res = self._run_client(["wait", prompt_id, "--out", str(out_dir),
                                "--timeout", str(self.timeout_s),
                                "--poll", str(self.poll_s)],
                               timeout_s=self.timeout_s + 120)
        combined = (res.stdout or "") + (res.stderr or "")
        if "cannot reach http" in combined:
            raise InfraError("l1", f"ComfyUI unreachable at {self.host}")
        result = self._parse_result(res.stdout or "")
        if res.returncode == 0 and result and result.get("ok"):
            files = [Path(f) for f in result.get("files") or []]
            videos = [f for f in files if f.suffix.lower() in VIDEO_EXTS]
            for extra in files:
                if extra not in videos:
                    extra.unlink(missing_ok=True)   # previews / last-frame PNGs
            if len(videos) != 1:
                for v in videos:
                    v.unlink(missing_ok=True)
                raise InfraError("l1", f"workflow produced {len(videos)} video "
                                       f"outputs ({[f.name for f in files]}) — need "
                                       f"exactly one")
            return {**result, "files": [str(videos[0])]}
        kind = {2: "exec_error", 3: "timeout"}.get(res.returncode, "client_error")
        raise GenerationFailed(kind, error_summary(res.stdout, res.stderr))

    # ---- raw HTTP (queue hygiene / handoff / recovery) -------------------------
    def _http(self, method: str, path: str, payload: dict | None = None,
              timeout_s: float = 30):
        req = urllib.request.Request(
            f"http://{self.host}{path}",
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Content-Type": "application/json"} if payload is not None else {},
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as r:
                body = r.read()
            return json.loads(body) if body else {}
        except urllib.error.HTTPError as e:
            raise InfraError("l1", f"ComfyUI HTTP {e.code} on {method} {path}")
        except (urllib.error.URLError, TimeoutError, OSError,
                http.client.HTTPException) as e:
            reason = getattr(e, "reason", e)
            raise InfraError("l1", f"ComfyUI unreachable at {self.host} ({reason})")
        except ValueError as e:                     # JSONDecodeError
            raise InfraError("l1", f"ComfyUI returned non-JSON for {path}: {e}")

    def free(self) -> None:
        self._http("POST", "/free", {"unload_models": True, "free_memory": True})

    def interrupt(self) -> None:
        self._http("POST", "/interrupt", {})

    def queue_state(self) -> dict:
        q = self._http("GET", "/queue")

        def ids(items):
            return [it[1] for it in items or [] if isinstance(it, list) and len(it) > 1]
        return {"running": ids(q.get("queue_running")),
                "pending": ids(q.get("queue_pending"))}

    def job_state(self, prompt_id: str) -> str:
        """running | pending | finished (in history) | unknown (e.g. the server
        restarted and forgot it)."""
        q = self.queue_state()
        if prompt_id in q["running"]:
            return "running"
        if prompt_id in q["pending"]:
            return "pending"
        hist = self._http("GET", f"/history/{prompt_id}")
        return "finished" if prompt_id in (hist or {}) else "unknown"

    def ensure_gone(self, prompt_id: str, timeout_s: float = 60, poll_s: float = 1.0,
                    sleep=time.sleep) -> None:
        """Hard rule 4: make sure OUR job no longer occupies the server — interrupt
        it only if it is the running one, delete it if pending — and verify."""
        deadline = time.monotonic() + timeout_s
        interrupted = deleted = False
        while True:
            state = self.job_state(prompt_id)
            if state in ("finished", "unknown"):
                return
            if state == "running" and not interrupted:
                self.interrupt()
                interrupted = True
            elif state == "pending" and not deleted:
                self._http("POST", "/queue", {"delete": [prompt_id]})
                deleted = True
            if time.monotonic() >= deadline:
                raise InfraError("l1", f"ComfyUI job {prompt_id} still {state} after "
                                       f"{timeout_s:.0f}s — refusing to submit another "
                                       f"job beside it (hard rule 4)")
            sleep(poll_s)

    def wait_queue_empty(self, timeout_s: float = 60, poll_s: float = 2.0,
                         sleep=time.sleep) -> None:
        """Hard rule 4 precondition for every submit: nothing running or pending."""
        deadline = time.monotonic() + timeout_s
        while True:
            q = self.queue_state()
            if not q["running"] and not q["pending"]:
                return
            if time.monotonic() >= deadline:
                raise InfraError("l1", f"ComfyUI queue busy ({len(q['running'])} "
                                       f"running, {len(q['pending'])} pending) — "
                                       f"refusing to submit a second job (hard rule 4)")
            sleep(poll_s)

    def vram_free_gb(self) -> float:
        """Device memory free for ANY process (what the judge or Wan can actually
        use): ComfyUI's vram_free minus torch's cached-but-unallocated part."""
        stats = self._http("GET", "/system_stats")
        devices = stats.get("devices") or []
        if not devices:
            raise InfraError("l1", "/system_stats reports no GPU devices")
        dev = devices[0]
        free = dev.get("vram_free", 0) - (dev.get("torch_vram_free") or 0)
        return max(0.0, free) / 2 ** 30

    def vram_wait(self, free_min_gb: float, wait_timeout_s: float, what: str,
                  poll_s: float = 2.0, sleep=time.sleep) -> float:
        deadline = time.monotonic() + wait_timeout_s
        last = self.vram_free_gb()
        while last < free_min_gb:
            if time.monotonic() >= deadline:
                raise InfraError("l2" if what == "judge" else "l1",
                                 f"VRAM handoff to {what} failed: {last:.1f} GB free "
                                 f"after {wait_timeout_s}s, need {free_min_gb} GB "
                                 f"(hard rule 3)")
            sleep(poll_s)
            last = self.vram_free_gb()
        return last

    def vram_handoff(self, free_min_gb: float, wait_timeout_s: float,
                     poll_s: float = 2.0, sleep=time.sleep) -> float:
        """Generation -> judge, enforced and VERIFIED: /free, then poll
        /system_stats until the GPU actually has room for the judge."""
        self.free()
        try:
            return self.vram_wait(free_min_gb, wait_timeout_s, "judge", poll_s, sleep)
        except InfraError as e:
            raise InfraError("l2", f"{e.message} — Wan did not unload; judging "
                                   f"would OOM")
