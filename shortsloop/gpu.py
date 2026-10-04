"""VRAM time-separation (hard rule 3) — ONE implementation for every GPU path
(nightly runner, calibrate-batch, calibrate-tune --with-l2, doctor).

    to_judge(comfy, ...)       ComfyUI /free -> verify free VRAM   (Wan is out)
    to_generation(comfy, ...)  unload judge VLM + rewrite LLM -> /free -> verify

Unload requests are best effort; the VERIFICATION (/system_stats free VRAM, as
seen by any process) is the authority and fails closed with InfraError.
"""

from __future__ import annotations

import json
import time
import urllib.request

from .errors import InfraError


def _request(method: str, url: str, payload: dict | None, timeout_s: float) -> None:
    req = urllib.request.Request(
        url, method=method,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"} if payload is not None else {})
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        r.read()


def unload_llms(judge_cfg: dict | None, rewrite_cfg: dict | None = None,
                timeout_s: float = 60) -> list[str]:
    """Ask every LLM server we use to drop its model now. Returns notes (what was
    unloaded / what could not be); never raises — verification decides."""
    notes: list[str] = []
    judge_cfg = judge_cfg or {}
    base = str(judge_cfg.get("base_url", "")).rstrip("/")
    model = judge_cfg.get("model")
    try:
        if judge_cfg.get("unload_url"):
            _request("GET", str(judge_cfg["unload_url"]), None, timeout_s)
            notes.append(f"judge unloaded via {judge_cfg['unload_url']}")
        elif judge_cfg.get("adapter", "ollama") == "ollama" and base and model:
            _request("POST", base + "/api/generate",
                     {"model": model, "keep_alive": 0}, timeout_s)
            notes.append(f"judge {model} unloaded")
        elif model:
            notes.append(f"judge adapter {judge_cfg.get('adapter')} has no unload hook "
                         f"(set judge.unload_url) — relying on VRAM verification")
    except Exception as e:  # noqa: BLE001 — verification below is the authority
        notes.append(f"judge unload failed: {e}")
    rw = rewrite_cfg or {}
    if rw.get("enabled", True) and rw.get("model") and rw.get("base_url"):
        try:
            _request("POST", str(rw["base_url"]).rstrip("/") + "/api/generate",
                     {"model": rw["model"], "keep_alive": 0}, timeout_s)
            notes.append(f"rewrite {rw['model']} unloaded")
        except Exception as e:  # noqa: BLE001
            notes.append(f"rewrite unload failed: {e}")
    return notes


def to_judge(comfy, free_min_gb: float, wait_timeout_s: float,
             sleep=time.sleep) -> float:
    """Generation -> judge: Wan out (/free), verified free VRAM."""
    return comfy.vram_handoff(free_min_gb, wait_timeout_s, sleep=sleep)


def to_generation(comfy, judge_cfg: dict | None, rewrite_cfg: dict | None,
                  free_min_gb: float, wait_timeout_s: float,
                  sleep=time.sleep) -> tuple[float, list[str]]:
    """Judge -> generation (and before any first generation): unload the VLM and
    the rewrite LLM, /free ComfyUI's caches, then VERIFY the GPU is empty enough
    that Wan never shares VRAM with a resident language model."""
    notes = unload_llms(judge_cfg, rewrite_cfg)
    comfy.free()
    try:
        free = comfy.vram_wait(free_min_gb, wait_timeout_s, "generation", sleep=sleep)
    except InfraError as e:
        raise InfraError("l1", f"{e.message} — a judge/rewrite model is still "
                               f"resident ({'; '.join(notes) or 'no unload notes'})")
    return free, notes
