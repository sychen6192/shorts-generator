"""The nightly wave runner (docs/plan.md §1, D2) — schedule / claim / generate /
verify / persist / complete.

Wave structure (VRAM-swap economics):
  generation wave: verified judge->generation handoff (VLM/LLM unloaded, GPU
  verified empty), then sequential ComfyUI jobs; the CPU-only L1 gate runs inline
  and L1 failures re-roll immediately while Wan is still loaded (bounded by the
  cap); then ONE verified generation->judge handoff; then the VLM judges every
  surviving attempt; failures are classified and scheduled for the next wave.

The runner NEVER interprets pixels. Ship decisions come exclusively from checker
verdict files via ship_gate() (hard rule 1). Everything here fails closed: missing
verdicts, weird exit codes, un-freed VRAM, a busy ComfyUI queue, uncalibrated
thresholds all stop work rather than ship a clip or burn the night quietly.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

from . import gpu, lock
from .check import load_thresholds
from .comfy import ComfyClient, GenerationFailed
from .dispatch import ClipSpec, parse_dispatch
from .encode import EncodeFailed, encode_silent
from .l1 import impl_fingerprint
from .errors import CheckError, InfraError
from .policy import plan_reroll
from .report import contact_sheet, render
from .rewrite import effective_rewrite_cfg, rewrite_prompt
from .schema import validate_verdict_file
from .settings import (DEFAULT_POLICIES, effective_judge_cfg,  # noqa: F401
                       judge_unload_problem, load_policies)
from .state import RunLog, active_seconds, fold
from .verdict import sha256_file, sha256_text

# Nightly defaults (plan §0.4: 720x1280, 81 frames @ 16 fps — Wan 2.2 14B native).
GEN_DEFAULTS = {"width": 720, "height": 1280, "length": 81, "fps": 16.0}

# Consecutive failed generations (no successful download in between, across
# clips) that mean "the instrument is broken", not "this clip is unlucky".
GEN_INFRA_ESCALATION = 3

# The steps8 re-roll knob is defined for the 4-step lightx2v build (CLAUDE.md:
# `--steps 4→8`); on any other workflow it is not applied.
STEPS8_BASE = 4


def spec_expectations(gen: dict) -> dict:
    """What the L1 spec check verifies: exactly what generation was asked for."""
    return {"width": gen["width"], "height": gen["height"], "fps": gen["fps"],
            "frames": gen["length"],
            "duration_s": round(gen["length"] / gen["fps"], 3)}


class RunHalted(Exception):
    """Infrastructure failure: stop the run, report loudly, ship nothing further."""


def ship_gate(verdict: dict) -> tuple[bool, str]:
    """THE gate (hard rules 1+2, plan §2.1). A clip ships only through here:
    PASS, both layers, no error, calibrated thresholds. No overrides."""
    if not isinstance(verdict, dict):
        return False, "no verdict"
    if verdict.get("verdict") != "PASS":
        return False, f"verdict is {verdict.get('verdict')!r}, not PASS"
    if verdict.get("layers_run") != ["l1", "l2"]:
        return False, f"layers_run {verdict.get('layers_run')} != ['l1','l2']"
    if verdict.get("error") is not None:
        return False, "verdict carries an error object"
    if (verdict.get("thresholds") or {}).get("calibrated") is not True:
        return False, "thresholds not calibrated (Phase 0 incomplete)"
    return True, "ok"


def _workflow_base_steps(path: str | Path) -> int | None:
    try:
        wf = json.loads(Path(path).read_text(encoding="utf-8"))
        steps = [n["inputs"]["steps"] for n in wf.values()
                 if isinstance(n, dict) and isinstance((n.get("inputs") or {})
                                                       .get("steps"), int)]
        return max(steps) if steps else None
    except (OSError, ValueError, AttributeError, KeyError):
        return None


@dataclass
class ClipRun:
    spec: ClipSpec
    expect: dict
    prompt_current: str
    attempts_used: int = 0
    status: str = "pending"       # pending|awaiting_l2|passed|failed_final|skipped|error
    skip_reason: str | None = None
    last_classes: list[str] = field(default_factory=list)
    rewritten: bool = False
    rewrite_diff: str | None = None
    next_plan: dict | None = None
    awaiting: dict | None = None
    passed_info: dict | None = None
    encoded_path: str | None = None
    encode_error: str | None = None
    attempt_history: list[dict] = field(default_factory=list)
    resume: dict | None = None    # interrupted attempt to finish first (--resume)
    replan: tuple | None = None   # (classes, judge reason): re-roll to plan on resume


class Runner:
    def __init__(self, *, dispatch_path, config_path, pipeline_path, thresholds_path,
                 runs_dir=None, resume_dir=None, allow_uncalibrated=False,
                 skip_doctor=False, clock=time.monotonic, disk_free_gb=None,
                 checker_argv=None, rng=None):
        self.skip_doctor = skip_doctor
        self.dispatch_path = Path(dispatch_path)
        self.config_path = Path(config_path)
        self.pipeline_path = Path(pipeline_path)
        self.thresholds_path = Path(thresholds_path)
        self.runs_dir = Path(runs_dir) if runs_dir else None
        self.resume_dir = Path(resume_dir) if resume_dir else None
        self.allow_uncalibrated = allow_uncalibrated
        self.clock = clock
        self._disk_free_gb = disk_free_gb          # injectable for tests
        self.checker_argv = checker_argv or [sys.executable, "-m", "shortsloop.check"]
        self.rng = rng or random.Random()

        self.items: list[ClipRun] = []
        self.gen_s = 0.0
        self.judge_s = 0.0
        self.waves_run = 0
        self.wall_tripped = False
        self.disk_tripped: str | None = None
        self.halt_reason: str | None = None
        self.consecutive_gen_failures = 0
        self.consecutive_judge_errors = 0     # survives wave boundaries (row 3)
        self._gen_ready = False                # verified judge->generation handoff
        self._persist_started = False
        self._judge_digest: str | None = None
        self._lock = None

    # ---------------------------------------------------------------- setup
    def _refuse(self, msg: str) -> int:
        print(f"[shortsloop] REFUSING TO RUN: {msg}", file=sys.stderr)
        return 2

    def _load(self) -> int | None:
        """Everything that can be checked before touching the GPU (fail closed at
        22:00, not at 3 a.m.). Returns an exit code to refuse with, or None."""
        for binary in ("ffmpeg", "ffprobe"):
            if not shutil.which(binary):
                return self._refuse(f"{binary} not on PATH (cron has a minimal PATH — "
                                    f"set PATH in the crontab, see docs/runbook.md)")
        if not self.config_path.is_file():
            return self._refuse(f"config not found: {self.config_path} "
                                f"(copy config.example.yaml, run doctor)")
        try:
            self.cfg = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as e:
            return self._refuse(f"config.yaml unparseable: {e}")
        comfy_cfg = self.cfg.get("comfy") or {}
        if not comfy_cfg.get("host") or not comfy_cfg.get("workflow_t2v"):
            return self._refuse("config.yaml needs comfy.host and comfy.workflow_t2v")
        if not (self.cfg.get("judge") or {}).get("model"):
            return self._refuse("config.yaml needs judge.model")

        self.doctor_ref = None
        if not self.skip_doctor:
            refusal = self._check_doctor()
            if refusal is not None:
                return refusal

        try:
            self.pol = load_policies(self.pipeline_path)
        except InfraError as e:
            return self._refuse(str(e))

        try:
            _thr, thr_info = load_thresholds(self.thresholds_path)
        except InfraError as e:
            return self._refuse(str(e))
        self.thresholds_provenance = _thr.get("provenance") or {}
        self.thresholds_calibrated = thr_info["calibrated"]
        tuned_l1 = self.thresholds_provenance.get("l1_impl")
        if self.thresholds_calibrated and tuned_l1 and tuned_l1 != impl_fingerprint():
            return self._refuse(
                f"thresholds were tuned against different L1 metric code (l1_impl "
                f"{tuned_l1} != {impl_fingerprint()}) — the calibrated values no longer "
                f"mean what was signed off; re-run calibrate-tune and --approve")
        self.thresholds_version = thr_info["version"]
        self.thresholds_sha = thr_info["file_sha256"]
        if not self.thresholds_calibrated and not self.allow_uncalibrated:
            return self._refuse(
                "thresholds.yaml has calibrated: false — an uncalibrated judge is an "
                "uncalibrated instrument. Finish Phase 0, or pass --allow-uncalibrated "
                "for a supervised run.")

        self.sheet = parse_dispatch(self.dispatch_path)
        if self.sheet.errors:
            for e in self.sheet.errors:
                print(f"[shortsloop] dispatch error: {e}", file=sys.stderr)
            return self._refuse(f"dispatch sheet failed intake "
                                f"({len(self.sheet.errors)} error(s) above)")

        self.gen_params = {**GEN_DEFAULTS,
                           **{k: v for k, v in self.sheet.shared.items()
                              if k in GEN_DEFAULTS}}
        # plan §2.5: absent shared params fall back to defaults — and say so
        self.gen_fallback = {k: v for k, v in GEN_DEFAULTS.items()
                             if k not in self.sheet.shared}
        if self.gen_fallback:
            print(f"[shortsloop] dispatch sheet has no usable 共用參數 for "
                  f"{sorted(self.gen_fallback)} — using defaults {self.gen_fallback}",
                  file=sys.stderr)
        self.expect = spec_expectations(self.gen_params)

        self.comfy = ComfyClient(
            host=str(comfy_cfg["host"]),
            workflow=comfy_cfg["workflow_t2v"],
            timeout_s=self.pol["comfy"]["timeout_s"],
            poll_s=self.pol["comfy"]["poll_s"])
        self.base_steps = _workflow_base_steps(comfy_cfg["workflow_t2v"])
        problems = self.comfy.validate_workflow(
            **self.gen_params, steps=8 if self.base_steps == STEPS8_BASE else None)
        if problems:
            return self._refuse("; ".join(problems))

        # pipeline.yaml's judge policy is authoritative for the runner's checker
        self.judge_cfg = effective_judge_cfg(self.cfg, self.pol)
        problem = judge_unload_problem(self.judge_cfg)
        if problem:
            return self._refuse(problem)
        cal = self.thresholds_provenance.get("judge")
        self.calibrated_judge = cal if isinstance(cal, dict) else None
        here = f"{self.judge_cfg.get('adapter', 'ollama')}/{self.judge_cfg.get('model')}"
        if (self.thresholds_calibrated and self.calibrated_judge
                and self.calibrated_judge.get("model")
                and self.calibrated_judge["model"] != here):
            return self._refuse(
                f"thresholds were calibrated with judge {self.calibrated_judge['model']} "
                f"but config.yaml uses {here} — L2 floors do not transfer between "
                f"judges; re-run Phase 0 (calibrate-tune --with-l2) for this judge")
        self.rewrite_cfg = effective_rewrite_cfg(self.cfg)
        return None

    def _check_doctor(self) -> int | None:
        from .doctor import fingerprint
        doctor_path = self.config_path.parent / "doctor.json"
        if not doctor_path.is_file():
            return self._refuse(
                f"no doctor snapshot ({doctor_path}) — run `shortsloop doctor` "
                f"on this machine first (or --skip-doctor for a supervised run)")
        try:
            snap = json.loads(doctor_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return self._refuse(f"doctor snapshot unparseable: {doctor_path}")
        if not snap.get("ok"):
            failed = [c["name"] for c in snap.get("checks", [])
                      if not c.get("ok") and c.get("level") == "FAIL"]
            return self._refuse(
                f"last doctor run FAILED ({', '.join(failed) or 'see doctor.json'})"
                f" — fix and re-run `shortsloop doctor`")
        current = fingerprint(self.config_path, self.pipeline_path)
        if snap.get("fingerprint") != current:
            changed = sorted(k for k in current
                             if (snap.get("fingerprint") or {}).get(k) != current[k])
            return self._refuse(
                f"doctor snapshot does not vouch for this setup (changed since doctor: "
                f"{', '.join(changed) or 'no fingerprint'}) — re-run `shortsloop doctor`")
        self.doctor_ref = {"path": str(doctor_path), "ts": snap.get("ts"),
                           "sha256": sha256_file(doctor_path)}
        return None

    def _runs_base(self) -> Path:
        return self.runs_dir or Path((self.cfg.get("paths") or {})
                                     .get("runs_dir", "runs"))

    def _acquire_lock(self) -> int | None:
        """One GPU user per runs dir (shortsloop/lock.py) — hard rule 4 starts here."""
        base = self.resume_dir.parent if self.resume_dir else self._runs_base()
        try:
            self._lock = lock.acquire(base)
        except lock.LockHeld as e:
            return self._refuse(f"another shortsloop process holds {e}")
        return None

    def _release_lock(self) -> None:
        lock.release(self._lock)
        self._lock = None

    def _disk_free(self, where: Path) -> float:
        return (self._disk_free_gb() if callable(self._disk_free_gb)
                else shutil.disk_usage(where).free / 2 ** 30)

    def _init_run_dir(self) -> None:
        now = datetime.now(timezone.utc)
        if self.resume_dir:
            self.run_dir = self.resume_dir
            self.run_id = self.run_dir.name
            self.run_dir.mkdir(parents=True, exist_ok=True)
        else:
            base = self._runs_base()
            base.mkdir(parents=True, exist_ok=True)
            stem = now.strftime("%Y%m%d-%H%M%S") + "-nightly"
            for k in range(1, 1000):
                self.run_id = stem if k == 1 else f"{stem}-{k}"
                try:
                    (base / self.run_id).mkdir(exist_ok=False)
                    break
                except FileExistsError:
                    continue
            self.run_dir = base / self.run_id
        for sub in ("clips", "verdicts", "encoded", "sheets", "prompts"):
            (self.run_dir / sub).mkdir(parents=True, exist_ok=True)
        self.log = RunLog(self.run_dir, self.run_id)
        if not (self.run_dir / "dispatch.md").exists():
            shutil.copyfile(self.dispatch_path, self.run_dir / "dispatch.md")

        # the checker's config: config.yaml + the pipeline's judge policy (snapshot)
        self.checker_config = self.run_dir / "config.snapshot.yaml"
        self.checker_config.write_text(yaml.safe_dump(
            {**self.cfg, "judge": self.judge_cfg}, sort_keys=False, allow_unicode=True),
            encoding="utf-8")

        wf = Path(self.cfg["comfy"]["workflow_t2v"])
        session = {
            "at": now.isoformat(timespec="seconds"),
            "dispatch": {"path": str(self.dispatch_path), "sha256": self.sheet.sha256},
            "config_sha256": sha256_file(self.config_path),
            "doctor": self.doctor_ref,
            "thresholds": {"version": self.thresholds_version,
                           "calibrated": self.thresholds_calibrated,
                           "sha256": self.thresholds_sha},
            "allow_uncalibrated": self.allow_uncalibrated,
        }
        run_json = self.run_dir / "run.json"
        meta = None
        if self.resume_dir and run_json.is_file():
            try:
                meta = json.loads(run_json.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                meta = None
        if meta is None:
            meta = {
                "run_id": self.run_id,
                "started_at": session["at"],
                **{k: v for k, v in session.items() if k != "at"},
                "config_snapshot": self.checker_config.name,
                "comfy": {"host": self.comfy.host, "workflow": str(wf),
                          "workflow_sha256": sha256_file(wf),
                          "base_steps": self.base_steps},
                "judge": {"adapter": self.judge_cfg.get("adapter"),
                          "model": self.judge_cfg.get("model")},
                "policies": self.pol,
                "gen_params": self.gen_params,
                "gen_fallback": self.gen_fallback,
                "resumes": [],
            }
        else:
            meta.setdefault("resumes", []).append(session)
        run_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")

    def _init_items(self) -> None:
        self.items = [ClipRun(spec=spec, expect=dict(self.expect),
                              prompt_current=spec.prompt)
                      for spec in self.sheet.clips]
        resumed = bool(self.resume_dir)
        if resumed:
            prior = self.log.read_events()
            # hard rule 5: the budget is whole-run — charge earlier sessions
            self._t_start -= active_seconds(prior)
            # waves restart from the last COMPLETED one: a wave a halt cut short
            # (even before its first submission) is finished, not skipped
            self.waves_run = max([e["data"]["wave_done"] for e in prior
                                  if e.get("stage") == "complete"
                                  and isinstance((e.get("data") or {}).get("wave_done"),
                                                 int)] or [0])
        self.log.event("schedule", "enter", accepted=len(self.items),
                       skipped=len(self.sheet.skipped), gen_params=self.gen_params,
                       fallback=self.gen_fallback, resumed=resumed)
        if not resumed:
            for row in self.sheet.skipped:
                self.log.event("schedule", "skip", row["clip_id"], None,
                               mode=row["mode"], reason=row["reason"])
            self.log.event("schedule", "ok")
            return
        # resume: fold prior state
        folded = fold(self.log)
        last_by_attempt: dict[tuple[str, int], dict] = {}
        for rec in self.log.read_attempts():
            if rec.get("clip_id") and rec.get("attempt"):
                last_by_attempt[(rec["clip_id"], rec["attempt"])] = rec  # last wins
        for item in self.items:
            cid = item.spec.clip_id
            st = folded.get(cid)
            if not st:
                continue
            item.attempts_used = st.attempts_used
            item.prompt_current = st.prompt_current or item.prompt_current
            item.rewritten = st.rewritten
            item.rewrite_diff = st.rewrite_diff
            item.last_classes = st.last_classes
            for (c, _n), rec in sorted(last_by_attempt.items()):
                if c == cid:
                    item.attempt_history.append(self._history_entry(rec))
            if st.status == "passed":
                item.status = "passed"
                item.passed_info = {"attempt": st.passed_attempt,
                                    "clip_path": st.clip_path,
                                    "verdict_path": st.verdict_path}
            elif st.recover:
                self._restore_interrupted(item, st.recover)
            elif item.attempts_used > 0:
                # last attempt ended in a final FAIL/ERROR before the crash: re-plan
                # its re-roll exactly as the live run would have (plan §2.3) — but
                # only after interrupted jobs are finished (a rewrite loads a text
                # LLM; never beside a running Wan job — hard rule 3)
                item.replan = (st.last_classes or ["broken"],
                               self._adherence_reason(st.last_verdict_path))
        for item in self.items:            # already persisted: never re-encode
            if item.status == "passed":
                for e in prior:
                    enc = (e.get("data") or {}).get("encoded")
                    if (e.get("stage") == "persist" and e.get("event") == "ok"
                            and e.get("clip_id") == item.spec.clip_id
                            and e.get("attempt") == item.passed_info["attempt"]
                            and enc and Path(enc).is_file()):
                        item.encoded_path = enc
        self.log.event("schedule", "ok", data_resumed=True)

    def _restore_interrupted(self, item: ClipRun, rec: dict) -> None:
        """Resume an attempt a crash/halt interrupted, at the stage it reached —
        never regenerate a clip that already exists (plan §5 row 14)."""
        cid, n = item.spec.clip_id, rec["attempt"]
        prompt_path = self.run_dir / "prompts" / f"{cid}_a{n}.txt"
        if prompt_path.exists():
            prompt_used = prompt_path.read_text(encoding="utf-8").strip()
        else:
            prompt_used = ((rec.get("record") or {}).get("prompt_text")
                           or item.prompt_current)
            prompt_path.write_text(prompt_used + "\n", encoding="utf-8")
        clip_file = Path(rec["file"]) if rec.get("file") else None
        if rec["stage"] != "in_flight" and (clip_file is None or not clip_file.is_file()):
            # the clip is gone: the only honest option is to re-wait the job
            rec = {**rec, "stage": "in_flight"}
        item.next_plan = None
        if rec["stage"] == "awaiting":
            sheet = self.run_dir / "sheets" / f"{cid}_a{n}.jpg"
            record = rec.get("record") or {}
            item.status = "awaiting_l2"
            item.awaiting = {"attempt": n, "clip_path": str(clip_file),
                             "prompt_path": str(prompt_path),
                             "seed": rec.get("seed"), "steps": rec.get("steps"),
                             "prompt_used": prompt_used,
                             "prompt_id": rec.get("prompt_id"),
                             "sheet": str(sheet) if sheet.exists() else None,
                             "gen_s": record.get("gen_elapsed_s"),
                             "vram_before": record.get("vram_free_before_gb"),
                             "action": record.get("reroll_action")}
        else:
            item.status = "pending"
            item.resume = {**rec, "prompt_used": prompt_used,
                           "prompt_path": str(prompt_path)}

    def _adherence_reason(self, verdict_path: str | None) -> str:
        if not verdict_path:
            return ""
        try:
            v = json.loads(Path(verdict_path).read_text(encoding="utf-8"))
            dims = (v.get("l2") or {}).get("dimensions") or {}
            return (dims.get("prompt_adherence") or {}).get("reason", "") or ""
        except (OSError, ValueError, AttributeError):
            return ""

    # ---------------------------------------------------------------- helpers
    def _elapsed(self) -> float:
        return self.clock() - self._t_start

    def _budget_block(self) -> str | None:
        if self._elapsed() > self.pol["wall_clock_budget_h"] * 3600:
            self.wall_tripped = True
            return "budget"
        free = self._disk_free(self.run_dir)
        if free < self.pol["disk_min_free_gb"]:
            self.disk_tripped = (f"disk ({free:.1f} GB free < "
                                 f"{self.pol['disk_min_free_gb']} GB floor)")
            return self.disk_tripped
        return None

    def _checker_timeout(self, l1_only: bool) -> float:
        if l1_only:
            return 600
        j = self.pol["judge"]
        return (1 + int(j["retries"])) * float(j["timeout_s"]) + 600

    def _invoke_checker(self, clip_path, prompt_path, out_json, l1_only: bool):
        argv = [*self.checker_argv, str(clip_path),
                "--prompt-file", str(prompt_path), "--json", str(out_json),
                "--thresholds", str(self.thresholds_path)]
        if l1_only:
            argv.append("--l1-only")
        else:
            argv += ["--config", str(self.checker_config)]
        if self.expect:
            argv += ["--expect", json.dumps(self.expect)]
        out = Path(out_json)
        for stale in (out, out.with_suffix(".l2_raw.json")):
            stale.unlink(missing_ok=True)   # only THIS invocation's verdict counts
        timeout = self._checker_timeout(l1_only)
        try:
            res = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise RunHalted(f"checker subprocess hung >{timeout:.0f}s — instrument broken")
        verdict = None
        if out.exists():
            try:
                verdict = json.loads(out.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                verdict = None
        expected = {0: ("PROCEED",) if l1_only else ("PASS",), 1: ("FAIL",),
                    2: ("ERROR",)}.get(res.returncode)
        if (expected is None or verdict is None
                or verdict.get("verdict") not in expected):
            raise RunHalted(
                f"checker contract violated on {Path(str(clip_path)).name}: exit "
                f"{res.returncode}, verdict "
                f"{(verdict or {}).get('verdict') if verdict else ('unparseable' if out.exists() else 'missing')}"
                f" — a clip without a consistent verdict never ships (hard rule 2)")
        return res.returncode, verdict

    def _history_entry(self, rec: dict) -> dict:
        return {k: rec.get(k) for k in
                ("attempt", "seed", "steps", "status", "failure_classes",
                 "reroll_action", "notes", "contact_sheet", "fail_reasons",
                 "gen_elapsed_s", "verdict")}

    def _record_attempt(self, item: ClipRun, *, attempt, seed, prompt_used, steps,
                        status, classes, verdict=None, verdict_path=None,
                        output_path=None, prompt_id=None, gen_s=None, sheet=None,
                        fail_reasons=None, action=None, note=None,
                        vram_before=None, verdict_obj=None) -> None:
        wf = self.cfg["comfy"]["workflow_t2v"]
        v = verdict_obj or {}
        rec = {
            "clip_id": item.spec.clip_id,
            "attempt": attempt,
            "seed": seed,
            "prompt_text": prompt_used,
            "prompt_sha256": sha256_text(prompt_used),
            "prompt_base": item.prompt_current,   # re-roll base (resume)
            "prompt_rewritten": item.rewritten,
            "prompt_diff": item.rewrite_diff if item.rewritten else None,
            "workflow_path": str(wf),
            "workflow_sha256": sha256_file(wf),
            "patch_args": {**self.gen_params, "seed": seed,
                           **({"steps": steps} if steps else {})},
            "comfy_prompt_id": prompt_id,
            "output_path": str(output_path) if output_path else None,
            "output_sha256": sha256_file(output_path) if output_path else None,
            "gen_elapsed_s": round(gen_s, 1) if gen_s is not None else None,
            "verdict_path": str(verdict_path) if verdict_path else None,
            "verdict": verdict,
            "failure_classes": classes,
            "l1_pass": (v.get("l1") or {}).get("pass") if v.get("l1") else None,
            "l2_pass": (v.get("l2") or {}).get("pass") if v.get("l2") else None,
            "vram_free_before_gb": vram_before,
            "notes": note,
            "status": status,
            "reroll_action": action,
            "contact_sheet": sheet,
            "fail_reasons": fail_reasons or [],
            "steps": steps,
            "wave": self.waves_run,
        }
        self.log.attempt(rec)
        # one history row per attempt: a final line supersedes an `unjudged` one
        item.attempt_history = [h for h in item.attempt_history
                                if h.get("attempt") != attempt]
        item.attempt_history.append(self._history_entry(rec))

    @staticmethod
    def _num(v) -> str:
        return f"{v:.6g}" if isinstance(v, (int, float)) else repr(v)

    def _fail_reasons(self, verdict: dict) -> list[str]:
        reasons = []
        for c in (verdict.get("l1") or {}).get("checks", []):
            if not c["pass"]:
                reasons.append(f"L1 {c['name']}: {c['reason']} "
                               f"({c['metric']}={self._num(c['value'])} vs "
                               f"{c['op']} {self._num(c['threshold'])})")
        for name, dim in ((verdict.get("l2") or {}).get("dimensions") or {}).items():
            if not dim["na"] and not dim["pass"]:
                reasons.append(f"L2 {name}: score {dim['score']} < floor {dim['floor']} — "
                               f"{dim['reason']}")
        err = verdict.get("error")
        if err:
            reasons.append(f"{err['scope']} error at {err['stage']}: {err['message']}")
        return reasons

    def _schedule_reroll(self, item: ClipRun, classes: list[str],
                         judge_reason: str = "") -> None:
        item.last_classes = classes
        item.awaiting = None
        if item.attempts_used >= self.pol["max_attempts_per_clip"]:
            item.status = "failed_final"
            self.log.event("complete", "fail", item.spec.clip_id,
                           item.attempts_used, classes=classes)
            return
        plan = plan_reroll(classes, item.attempts_used + 1, item.prompt_current)
        if plan["steps"] == 8 and self.base_steps != STEPS8_BASE:
            plan["steps"] = None             # steps8 is defined for the 4-step build
            plan["action"] += f" (steps8 n/a: workflow runs {self.base_steps} steps)"
        if plan["wants_rewrite"]:
            got = rewrite_prompt(item.prompt_current, judge_reason, self.rewrite_cfg,
                                 anchors=item.spec.anchors)
            if got["ok"]:
                item.prompt_current, item.rewrite_diff = got["text"], got["diff"]
                item.rewritten = True
                plan["prompt"] = item.prompt_current
            else:
                plan["action"] = f"rewrite_skipped_reseed ({got['reason'][:160]})"
                plan["prompt"] = item.prompt_current
        item.next_plan = plan
        item.status = "pending"

    # ---------------------------------------------------------------- stages
    def _gen_handoff(self) -> None:
        """Hard rule 3, judge -> generation: no VLM/LLM resident when Wan loads.
        Verified once per wave, right before the wave's first submission."""
        if self._gen_ready:
            return
        vh = self.pol["vram_handoff"]
        free_gb, notes = gpu.to_generation(self.comfy, self.judge_cfg, self.rewrite_cfg,
                                           gpu.gen_free_min_gb(vh), vh["wait_timeout_s"])
        self.log.event("generate", "ok", None, None, layer="vram_handoff",
                       vram_free_gb=round(free_gb, 1), notes=notes)
        self._gen_ready = True

    def _generation_phase(self) -> None:
        self._gen_ready = False
        # 1) finish interrupted attempts FIRST: their jobs may still be on the
        #    server, and nothing new may be submitted beside them (hard rule 4)
        for item in self.items:
            resume, item.resume = item.resume, None
            if not resume or item.status != "pending":
                continue
            n = resume["attempt"]
            common = dict(prompt_path=Path(resume["prompt_path"]), resumed=True,
                          action=(resume.get("record") or {}).get("reroll_action"))
            if resume["stage"] == "generated":
                self.log.event("generate", "ok", item.spec.clip_id, n,
                               file=resume["file"], resumed=True)
                self._l1_gate(item, n, resume.get("seed"), resume["prompt_used"],
                              resume.get("steps"), resume.get("prompt_id"),
                              Path(resume["file"]), gen_elapsed=None,
                              prompt_path=common["prompt_path"],
                              action=common["action"])
            else:
                self._finish_generation(item, n, resume.get("seed"),
                                        resume["prompt_used"], resume.get("steps"),
                                        resume.get("prompt_id"), **common)
        # 2) re-roll plans deferred by --resume (may call the rewrite LLM)
        for item in self.items:
            replan, item.replan = item.replan, None
            if replan and item.status == "pending":
                self._schedule_reroll(item, *replan)
        # 3) new attempts, one job at a time
        for item in self.items:
            while item.status == "pending":
                cid = item.spec.clip_id
                if item.attempts_used >= self.pol["max_attempts_per_clip"]:
                    item.status = "failed_final"
                    self.log.event("complete", "fail", cid, item.attempts_used,
                                   classes=item.last_classes)
                    break
                block = self._budget_block()
                if block:
                    item.status = "skipped"
                    item.skip_reason = (block if item.attempts_used == 0 else
                                        f"{block} after {item.attempts_used} attempt(s)")
                    self.log.event("claim", "skip", cid, None, reason=block)
                    break
                self._gen_handoff()
                n = item.attempts_used + 1
                plan = item.next_plan or {"prompt": item.prompt_current, "steps": None,
                                          "action": "initial", "wants_rewrite": False,
                                          "primary_class": None}
                item.next_plan = None
                self.log.event("claim", "enter", cid, n, wave=self.waves_run)
                prompt_used, steps = plan["prompt"], plan["steps"]
                seed = self.rng.randint(0, 2 ** 48)
                prompt_path = self.run_dir / "prompts" / f"{cid}_a{n}.txt"
                prompt_path.write_text(prompt_used + "\n", encoding="utf-8")
                self.log.event("claim", "ok", cid, n, action=plan["action"],
                               prompt_base=item.prompt_current,
                               rewritten=item.rewritten,
                               rewrite_diff=item.rewrite_diff if item.rewritten else None)

                vram_before = None
                try:
                    vram_before = round(self.comfy.vram_free_gb(), 1)
                except InfraError:
                    pass  # stats endpoint hiccup is not fatal at claim time
                self.comfy.wait_queue_empty(self.pol["vram_handoff"]["wait_timeout_s"])
                self.log.event("generate", "enter", cid, n, seed=seed, steps=steps)
                t0 = self.clock()
                sub = self.comfy.submit(prompt=prompt_used, seed=seed, steps=steps,
                                        **self.gen_params)   # failures here are infra
                seed = sub.get("seed", seed)
                self.log.event("generate", "submitted", cid, n,
                               prompt_id=sub["prompt_id"], seed=seed, steps=steps,
                               prompt_sha=sha256_text(prompt_used))
                item.attempts_used = n
                self._finish_generation(item, n, seed, prompt_used, steps,
                                        sub["prompt_id"], t0=t0,
                                        prompt_path=prompt_path,
                                        vram_before=vram_before,
                                        action=plan["action"])

    def _finish_generation(self, item, n, seed, prompt_used, steps, prompt_id,
                           t0=None, prompt_path=None, resumed=False,
                           vram_before=None, action=None) -> None:
        cid = item.spec.clip_id
        item.attempts_used = max(item.attempts_used, n)
        clips_dir = self.run_dir / "clips"
        if prompt_path is None:
            prompt_path = self.run_dir / "prompts" / f"{cid}_a{n}.txt"
            if not prompt_path.exists():
                prompt_path.write_text(prompt_used + "\n", encoding="utf-8")
        t0 = t0 if t0 is not None else self.clock()
        try:
            if not prompt_id:
                raise GenerationFailed("lost", "no ComfyUI prompt_id recorded to "
                                               "re-attach to")
            if resumed and self.comfy.job_state(prompt_id) == "unknown":
                raise GenerationFailed("lost", f"ComfyUI has no record of job "
                                               f"{prompt_id} (server restarted?)")
            result = self.comfy.wait(prompt_id, clips_dir)
        except GenerationFailed as e:
            self.gen_s += self.clock() - t0
            self.log.event("generate", "fail", cid, n, kind=e.kind, detail=e.detail)
            # hard rule 6 first: the attempt is consumed and logged even if the
            # cleanup below halts the run
            self._record_attempt(item, attempt=n, seed=seed, prompt_used=prompt_used,
                                 steps=steps, status="gen_failed", classes=["broken"],
                                 prompt_id=prompt_id, note=f"{e.kind}: {e.detail}",
                                 action=action, vram_before=vram_before)
            item.status, item.last_classes = "pending", ["broken"]
            # hard rule 4: OUR job must be gone (interrupted / dequeued, verified)
            # before anything else is submitted; then OOM recovery per skill notes
            if prompt_id:
                self.comfy.ensure_gone(prompt_id,
                                       self.pol["vram_handoff"]["wait_timeout_s"])
            self.comfy.free()
            self.consecutive_gen_failures += 1
            if self.consecutive_gen_failures >= GEN_INFRA_ESCALATION:
                raise RunHalted(
                    f"{self.consecutive_gen_failures} consecutive generation failures "
                    f"across the run (last: {e.kind}: {e.detail[:160]}) — ComfyUI/"
                    f"workflow is broken, not the clips (fail closed)")
            self._schedule_reroll(item, ["broken"])
            return
        self.consecutive_gen_failures = 0
        gen_elapsed = self.clock() - t0
        self.gen_s += gen_elapsed
        src = Path(result["files"][0])
        clip_file = clips_dir / f"{cid}_a{n}{src.suffix or '.mp4'}"
        if src.resolve() != clip_file.resolve():
            shutil.move(str(src), clip_file)
        self.log.event("generate", "ok", cid, n, file=str(clip_file),
                       elapsed_s=round(gen_elapsed, 1), resumed=resumed)
        self._l1_gate(item, n, seed, prompt_used, steps, prompt_id, clip_file,
                      gen_elapsed=gen_elapsed, prompt_path=prompt_path,
                      vram_before=vram_before, action=action)

    def _l1_gate(self, item, n, seed, prompt_used, steps, prompt_id, clip_file,
                 *, gen_elapsed, prompt_path, vram_before=None, action=None) -> None:
        """Inline CPU L1 gate on a downloaded clip (Wan stays loaded)."""
        cid = item.spec.clip_id
        item.attempts_used = max(item.attempts_used, n)
        sheet = contact_sheet(clip_file, self.run_dir / "sheets" / f"{cid}_a{n}.jpg")
        self.log.event("verify", "enter", cid, n, layer="l1")
        l1_json = self.run_dir / "verdicts" / f"{cid}_a{n}.l1.json"
        code, verdict = self._invoke_checker(clip_file, prompt_path, l1_json,
                                             l1_only=True)
        common = dict(attempt=n, seed=seed, prompt_used=prompt_used, steps=steps,
                      verdict_path=l1_json, output_path=clip_file, prompt_id=prompt_id,
                      gen_s=gen_elapsed, sheet=sheet, action=action,
                      vram_before=vram_before, verdict_obj=verdict,
                      fail_reasons=self._fail_reasons(verdict))
        if code == 0:
            item.status = "awaiting_l2"
            item.awaiting = {"attempt": n, "clip_path": str(clip_file),
                             "prompt_path": str(prompt_path), "seed": seed,
                             "steps": steps, "prompt_used": prompt_used,
                             "prompt_id": prompt_id, "sheet": sheet,
                             "gen_s": gen_elapsed, "vram_before": vram_before,
                             "action": action}
            self.log.event("verify", "ok", cid, n, layer="l1", verdict="PROCEED")
        elif code == 1:
            classes = verdict.get("failure_classes") or ["broken"]
            self.log.event("verify", "fail", cid, n, layer="l1", classes=classes)
            self._record_attempt(item, status="l1_failed", classes=classes,
                                 verdict="FAIL", **common)
            self._schedule_reroll(item, classes)
        else:  # code == 2
            err = verdict.get("error") or {}
            if err.get("scope") == "infra":
                raise RunHalted(f"L1 gate infra error on {cid}: {err.get('message')}")
            self.log.event("verify", "error", cid, n, layer="l1",
                           message=err.get("message"))
            self._record_attempt(item, status="error", classes=["broken"],
                                 verdict="ERROR", **common)
            self._schedule_reroll(item, ["broken"])

    def _current_judge_digest(self) -> str:
        if self._judge_digest is None:
            from .judge import make_adapter
            self._judge_digest = make_adapter(self.judge_cfg).model_digest()
        return self._judge_digest

    def _verdict_cache_hit(self, out_json: Path, aw: dict):
        """Plan §3 cache key: (clip_sha256, prompt_sha256, thresholds, judge model
        digest) — resume never re-judges an artifact, nor reuses another judge's."""
        if not out_json.exists():
            return None
        try:
            v = json.loads(out_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        if validate_verdict_file(v) or v.get("layers_run") != ["l1", "l2"]:
            return None
        prompt_text = Path(aw["prompt_path"]).read_text(encoding="utf-8").strip()
        if not (v["clip"]["sha256"] == sha256_file(aw["clip_path"])
                and v["prompt"]["sha256"] == sha256_text(prompt_text)
                and v["thresholds"]["file_sha256"] == self.thresholds_sha):
            return None
        if v["l2"]["model_digest"] != self._current_judge_digest():
            return None
        code = {"PASS": 0, "FAIL": 1}.get(v["verdict"], 2)
        return code, v

    def _judge_phase(self) -> None:
        awaiting = [i for i in self.items if i.status == "awaiting_l2"]
        if not awaiting:
            return
        t0 = self.clock()
        self.log.event("verify", "enter", None, None, layer="l2_wave",
                       count=len(awaiting), wave=self.waves_run)
        vh = self.pol["vram_handoff"]
        self._gen_ready = False
        try:
            free_gb = gpu.to_judge(self.comfy, vh["free_min_gb"], vh["wait_timeout_s"])
            self.log.event("verify", "ok", None, None, layer="vram_handoff",
                           vram_free_gb=round(free_gb, 1))
            want = (self.calibrated_judge or {}).get("model_digest")
            if self.thresholds_calibrated and want and \
                    self._current_judge_digest() != want:
                raise RunHalted(
                    f"judge build {self._current_judge_digest()} differs from the build "
                    f"the thresholds were calibrated with ({want}) — re-pull the "
                    f"calibrated model or re-run Phase 0")
            self._judge_wave(awaiting)
        finally:
            self.judge_s += self.clock() - t0
            # hard rule 3: judge AND rewrite model out on every exit path (verified
            # by the next generation wave's handoff)
            notes = gpu.unload_llms(self.judge_cfg, self.rewrite_cfg)
            self.log.event("verify", "ok", None, None, layer="judge_unload",
                           notes=notes)

    def _judge_wave(self, awaiting: list[ClipRun]) -> None:
        for item in awaiting:
            aw = item.awaiting
            cid, n = item.spec.clip_id, aw["attempt"]
            out_json = self.run_dir / "verdicts" / f"{cid}_a{n}.json"
            cached = self._verdict_cache_hit(out_json, aw)
            if cached:
                code, verdict = cached
                self.log.event("verify", "ok", cid, n, layer="l2", cached=True)
            else:
                code, verdict = self._invoke_checker(
                    aw["clip_path"], aw["prompt_path"], out_json, l1_only=False)
            common = dict(attempt=n, seed=aw["seed"], prompt_used=aw["prompt_used"],
                          steps=aw["steps"], prompt_id=aw["prompt_id"],
                          gen_s=aw["gen_s"], sheet=aw["sheet"],
                          verdict_path=out_json, output_path=aw["clip_path"],
                          action=aw.get("action"), vram_before=aw.get("vram_before"),
                          verdict_obj=verdict)
            if code == 2:
                err = verdict.get("error") or {}
                if err.get("scope") == "infra":
                    raise RunHalted(f"judge infra error on {cid}: {err.get('message')}")
                self.consecutive_judge_errors += 1
                self.log.event("verify", "error", cid, n, layer="l2",
                               message=err.get("message"))
                self._record_attempt(item, status="error", classes=["broken"],
                                     verdict="ERROR",
                                     fail_reasons=self._fail_reasons(verdict), **common)
                self._schedule_reroll(item, ["broken"])   # the attempt is final
                limit = self.pol["judge"]["infra_escalation_after"]
                if self.consecutive_judge_errors >= limit:
                    raise RunHalted(
                        f"{self.consecutive_judge_errors} consecutive clip-scope judge "
                        f"errors — escalating to infrastructure failure (fail closed)")
                continue
            self.consecutive_judge_errors = 0
            if code == 0:
                ok, why = self._gate(verdict)
                if not ok:
                    raise RunHalted(f"internal: exit 0 verdict for {cid} rejected by "
                                    f"ship-gate ({why}) — checker/runner mismatch")
                item.status = "passed"
                item.passed_info = {"attempt": n, "clip_path": aw["clip_path"],
                                    "verdict_path": str(out_json)}
                item.awaiting = None
                item.last_classes = []
                self.log.event("verify", "ok", cid, n, layer="l2", verdict="PASS")
                self._record_attempt(item, status="passed", classes=[],
                                     verdict="PASS", **common)
            else:
                classes = verdict.get("failure_classes") or ["broken"]
                reasons = self._fail_reasons(verdict)
                self.log.event("verify", "fail", cid, n, layer="l2", classes=classes)
                self._record_attempt(item, status="l2_failed", classes=classes,
                                     verdict="FAIL", fail_reasons=reasons, **common)
                adherence_reason = ""
                dims = (verdict.get("l2") or {}).get("dimensions") or {}
                if "prompt_adherence" in dims:
                    adherence_reason = dims["prompt_adherence"].get("reason", "")
                self._schedule_reroll(item, classes, adherence_reason)

    def _gate(self, verdict: dict) -> tuple[bool, str]:
        """ship_gate, or — in a supervised --allow-uncalibrated run only — the same
        gate minus calibration (those clips go to encoded_uncalibrated/)."""
        ok, why = ship_gate(verdict)
        if not ok and self.allow_uncalibrated and \
                (verdict.get("thresholds") or {}).get("calibrated") is False:
            return ship_gate({**verdict, "thresholds": {**verdict["thresholds"],
                                                        "calibrated": True}})
        return ok, why

    def _persist_phase(self) -> None:
        self._persist_started = True
        refused: list[str] = []
        for item in self.items:
            if item.status != "passed" or item.encoded_path:
                continue
            cid, att = item.spec.clip_id, item.passed_info["attempt"]
            self.log.event("persist", "enter", cid, att)
            try:
                verdict = json.loads(Path(item.passed_info["verdict_path"])
                                     .read_text(encoding="utf-8"))
                ok, why = self._gate(verdict)
                if not ok:
                    refused.append(f"{cid}: {why}")
                    raise EncodeFailed(f"ship-gate refused at persist: {why} — "
                                       f"not shipped")
                clip = item.passed_info["clip_path"]
                if sha256_file(clip) != (verdict.get("clip") or {}).get("sha256"):
                    raise EncodeFailed(f"clip sha256 differs from the one its PASS "
                                       f"verdict judged — not shipped")
                shippable = (verdict.get("thresholds") or {}).get("calibrated") is True
                folder = "encoded" if shippable else "encoded_uncalibrated"
                out = self.run_dir / folder / f"{cid}.mp4"
                encode_silent(clip, out)
                item.encoded_path = str(out)
                self.log.event("persist", "ok", cid, att, encoded=str(out))
            except (EncodeFailed, CheckError, OSError, ValueError) as e:
                item.encode_error = str(e)
                self.log.event("persist", "fail", cid, att, error=str(e)[:300])
        if refused:
            raise RunHalted("ship-gate refused PASS verdict(s) at persist — checker/"
                            "runner mismatch: " + "; ".join(refused))

    def _halt(self, status: str, reason: str) -> str:
        """Stop cleanly: log the halt, give every generated-but-unjudged attempt
        its attempts.jsonl line (hard rule 6 — --resume judges those clips instead
        of regenerating them), and still encode clips that PASSED before the halt."""
        self.halt_reason = reason
        self.log.event("complete", "halt", None, None, reason=reason[:500])
        for item in self.items:
            aw = item.awaiting
            if item.status != "awaiting_l2" or not aw:
                continue
            try:
                self._record_attempt(
                    item, attempt=aw["attempt"], seed=aw["seed"],
                    prompt_used=aw["prompt_used"], steps=aw["steps"],
                    status="unjudged", classes=[], verdict="PROCEED",
                    verdict_path=self.run_dir / "verdicts"
                    / f"{item.spec.clip_id}_a{aw['attempt']}.l1.json",
                    output_path=aw["clip_path"], prompt_id=aw["prompt_id"],
                    gen_s=aw.get("gen_s"), sheet=aw.get("sheet"),
                    action=aw.get("action"), vram_before=aw.get("vram_before"),
                    note=f"L1 PROCEED; run halted before the judge — "
                         f"--resume judges this clip ({reason[:120]})")
            except Exception as e:  # never let bookkeeping mask the halt itself
                print(f"[shortsloop] could not log unjudged attempt for "
                      f"{item.spec.clip_id}: {e}", file=sys.stderr)
        if not self._persist_started:
            try:
                self._persist_phase()
            except Exception as e:  # noqa: BLE001 — the halt is already the story
                print(f"[shortsloop] persist after halt failed: {e}", file=sys.stderr)
        return status

    # ---------------------------------------------------------------- run
    def run(self) -> int:
        refusal = self._load()
        if refusal is not None:
            return refusal
        if self.resume_dir:
            stored = self.resume_dir / "dispatch.md"
            if not self.resume_dir.is_dir():
                return self._refuse(f"--resume: no run directory {self.resume_dir}")
            if stored.is_file() and sha256_file(stored) != self.sheet.sha256:
                return self._refuse(
                    f"--resume: {self.dispatch_path} differs from the run's own copy "
                    f"({stored}) — resume with the original sheet, or start a new run")
        refusal = self._acquire_lock()
        if refusal is not None:
            return refusal
        try:
            base = self.resume_dir or self._runs_base()
            free = self._disk_free(base)
            if free < self.pol["disk_min_free_gb"]:
                return self._refuse(f"disk: {free:.1f} GB free at {base} < "
                                    f"{self.pol['disk_min_free_gb']} GB floor")
            return self._run_locked()
        finally:
            self._release_lock()

    def _run_locked(self) -> int:
        self._init_run_dir()
        self._t_start = self.clock()

        status = "COMPLETED"
        try:
            self._init_items()
            waves_max = min(self.pol["waves_max"], self.pol["max_attempts_per_clip"])
            while self.waves_run < waves_max:
                pending = [i for i in self.items if i.status == "pending"]
                awaiting = [i for i in self.items if i.status == "awaiting_l2"]
                if not pending and not awaiting:
                    break
                self.waves_run += 1
                self._generation_phase()
                self._judge_phase()
                self.log.event("complete", "ok", None, None, wave_done=self.waves_run)
            for item in self.items:   # only reachable via misconfig; never silent
                if item.status == "pending":
                    item.status = "skipped"
                    item.skip_reason = "wave limit reached"
            self._persist_phase()
        except (RunHalted, InfraError) as e:
            status = self._halt("HALTED(infra)", str(e))
        except KeyboardInterrupt:
            status = self._halt("HALTED(interrupted)", "interrupted (SIGINT)")
        except Exception as e:  # a runner bug is a broken instrument: report, ship nothing
            import traceback
            traceback.print_exc()
            status = self._halt("HALTED(crash)",
                                f"runner crashed: {type(e).__name__}: {e}")
        if not status.startswith("HALTED"):
            if self.wall_tripped:
                status = "COMPLETED(budget-stopped)"
            elif self.disk_tripped:
                status = "COMPLETED(disk-stopped)"

        budget = {
            "wall_s": round(self._elapsed(), 1),
            "wall_budget_s": self.pol["wall_clock_budget_h"] * 3600,
            "wall_tripped": self.wall_tripped,
            "disk_tripped": self.disk_tripped,
            "disk_min_free_gb": self.pol["disk_min_free_gb"],
            "resumed": bool(self.resume_dir),
            "gen_s": round(self.gen_s, 1),
            "judge_s": round(self.judge_s, 1),
            "waves_run": self.waves_run,
            "waves_max": min(self.pol["waves_max"], self.pol["max_attempts_per_clip"]),
            "max_attempts": self.pol["max_attempts_per_clip"],
            "cap_respected": all(i.attempts_used <= self.pol["max_attempts_per_clip"]
                                 for i in self.items),
        }
        run_meta = {
            "run_id": self.run_id,
            "status": status,
            "halt_reason": self.halt_reason,
            "dispatch_path": str(self.dispatch_path),
            "dispatch_sha256": self.sheet.sha256,
            "thresholds_version": self.thresholds_version,
            "thresholds_calibrated": self.thresholds_calibrated,
            "thresholds_provenance": {k: self.thresholds_provenance.get(k) for k in
                                      ("test_scope", "l2_untested_accepted",
                                       "l2_floors_fallback", "judge")},
            "allow_uncalibrated": self.allow_uncalibrated,
            "gen_params": self.gen_params,
            "gen_fallback": self.gen_fallback,
            "base_steps": self.base_steps,
        }
        render(self.run_dir, run_meta, self.items, budget, self.sheet.skipped)
        self.log.event("complete", "ok" if status.startswith("COMPLETED") else "halt",
                       None, None, status=status)
        passed = sum(1 for i in self.items if i.status == "passed")
        print(f"[shortsloop] {status}: {passed}/{len(self.items)} clips passed · "
              f"report: {self.run_dir / 'report.md'}")
        return 0 if status.startswith("COMPLETED") else 3


def main_run(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="shortsloop run",
                                 description="Nightly wave runner over a dispatch sheet")
    ap.add_argument("--dispatch", required=True)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--pipeline", default="pipeline.yaml")
    ap.add_argument("--thresholds", default="thresholds.yaml")
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--resume", default=None, metavar="RUN_DIR")
    ap.add_argument("--allow-uncalibrated", action="store_true",
                    help="supervised runs only — PASS clips go to encoded_uncalibrated/, "
                         "never to encoded/")
    ap.add_argument("--skip-doctor", action="store_true",
                    help="skip the doctor.json snapshot gate (supervised runs)")
    args = ap.parse_args(argv)
    runner = Runner(dispatch_path=args.dispatch, config_path=args.config,
                    pipeline_path=args.pipeline, thresholds_path=args.thresholds,
                    runs_dir=args.runs_dir, resume_dir=args.resume,
                    allow_uncalibrated=args.allow_uncalibrated,
                    skip_doctor=args.skip_doctor)
    return runner.run()
