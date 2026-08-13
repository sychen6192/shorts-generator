"""`shortsloop label` — Phase 0 labeling UI (docs/plan.md §6 step 2).

Local single-page web app, keyboard-only, optimized for labeling ~40 clips in one
sitting: autoplay loop, evaluation prompt alongside (off-prompt calls need it),
one JSONL line per verdict (crash-safe, resumable — relabeling appends and the
tuner takes the LAST label per clip).

Blindness: the page never shows a clip's `kind` or swap provenance; the display
order is shuffled with a fixed seed so swap rows don't sit next to their source.

Keys:  1 pass · 2 static · 3 deformed · 4 flicker · 5 off-prompt · 6 other(+note)
       space replay · p previous · n skip
"""

from __future__ import annotations

import argparse
import json
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .state import _read_jsonl
from .verdict import sha256_file

VALID_CLASSES = ("static", "deformed", "flicker", "off_prompt", "other")

PAGE = """<!doctype html>
<meta charset="utf-8">
<title>shortsloop labeling</title>
<style>
  body { margin:0; background:#111; color:#ddd; font:14px/1.5 system-ui, sans-serif;
         display:flex; height:100vh; }
  #left { flex:0 0 46%; display:flex; align-items:center; justify-content:center;
          background:#000; }
  video { max-width:100%; max-height:100vh; }
  #right { flex:1; padding:2rem; overflow-y:auto; }
  #progress { color:#8ac; font-size:1.1rem; }
  #prompt { background:#1c1c1c; border:1px solid #333; border-radius:8px;
            padding:1rem; margin:1rem 0; white-space:pre-wrap; }
  .key { display:inline-block; background:#333; border-radius:4px; padding:0 .5em;
         margin-right:.35em; font-family:monospace; }
  #keys div { margin:.3rem 0; }
  #last { color:#7a7; min-height:1.5em; }
  #done { display:none; font-size:1.4rem; color:#8f8; }
</style>
<div id="left"><video id="v" autoplay loop muted playsinline></video></div>
<div id="right">
  <div id="progress"></div>
  <h3>Does the clip match THIS prompt?</h3>
  <div id="prompt"></div>
  <div id="keys">
    <div><span class="key">1</span> PASS — publishable</div>
    <div><span class="key">2</span> FAIL: static / no real motion</div>
    <div><span class="key">3</span> FAIL: deformed / anatomy / subject mutates</div>
    <div><span class="key">4</span> FAIL: flicker / temporal chaos</div>
    <div><span class="key">5</span> FAIL: off-prompt (wrong content for the prompt)</div>
    <div><span class="key">6</span> FAIL: other (asks for a note)</div>
    <div><span class="key">space</span> replay ·
         <span class="key">p</span> previous · <span class="key">n</span> skip</div>
  </div>
  <div id="last"></div>
  <div id="done">All clips labeled — you can close this tab.
    <br>Labels: <code id="outpath"></code></div>
</div>
<script>
const DATA = __DATA__;
let order = DATA.rows;
let idx = order.findIndex(r => !DATA.labeled.includes(r.clip_id));
if (idx < 0) idx = order.length;
let shownAt = Date.now();
const v = document.getElementById("v");
document.getElementById("outpath").textContent = DATA.labels_path;

function show() {
  const total = order.length;
  const done = order.filter(r => DATA.labeled.includes(r.clip_id)).length;
  document.getElementById("progress").textContent =
      `clip ${Math.min(idx + 1, total)} / ${total} — ${done} labeled`;
  if (idx >= total) {
    document.getElementById("done").style.display = "block";
    v.removeAttribute("src"); v.load();
    return;
  }
  const row = order[idx];
  v.src = "/clip/" + row.clip_id;
  v.play().catch(() => {});
  document.getElementById("prompt").textContent = row.prompt;
  shownAt = Date.now();
}

async function send(verdict, classes, note) {
  const row = order[idx];
  const res = await fetch("/label", {method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({clip_id: row.clip_id, verdict: verdict,
                          classes: classes, note: note || "",
                          ms: Date.now() - shownAt})});
  if (!res.ok) { alert("label write failed: " + await res.text()); return; }
  if (!DATA.labeled.includes(row.clip_id)) DATA.labeled.push(row.clip_id);
  document.getElementById("last").textContent =
      `${row.clip_id}: ${verdict}${classes.length ? " (" + classes + ")" : ""}`;
  idx += 1; show();
}

const KEYMAP = {"1": ["pass", []], "2": ["fail", ["static"]],
                "3": ["fail", ["deformed"]], "4": ["fail", ["flicker"]],
                "5": ["fail", ["off_prompt"]]};
document.addEventListener("keydown", (e) => {
  if (e.key === " ") { e.preventDefault(); v.currentTime = 0; v.play(); return; }
  if (e.key === "p") { idx = Math.max(0, idx - 1); show(); return; }
  if (e.key === "n") { idx = Math.min(order.length, idx + 1); show(); return; }
  if (idx >= order.length) return;
  if (KEYMAP[e.key]) { send(...KEYMAP[e.key]); return; }
  if (e.key === "6") {
    const note = prompt("failure note:") || "";
    send("fail", ["other"], note);
  }
});
show();
</script>
"""


class LabelServer(ThreadingHTTPServer):
    def __init__(self, addr, rows: list[dict], labels_path: Path):
        super().__init__(addr, _Handler)
        self.rows = rows
        self.labels_path = Path(labels_path)
        self.clips = {r["clip_id"]: r["clip_path"] for r in rows}
        self.lock = threading.Lock()

    def handle_error(self, request, client_address):
        pass

    def labeled_ids(self) -> list[str]:
        return sorted({r["clip_id"] for r in _read_jsonl(self.labels_path)})


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, body: bytes, ctype: str, code: int = 200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        srv: LabelServer = self.server  # type: ignore[assignment]
        if self.path == "/":
            data = {
                "rows": [{"clip_id": r["clip_id"], "prompt": r["prompt"]}
                         for r in srv.rows],
                "labeled": srv.labeled_ids(),
                "labels_path": str(srv.labels_path),
            }
            page = PAGE.replace("__DATA__", json.dumps(data, ensure_ascii=False))
            self._send(page.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path.startswith("/clip/"):
            clip_id = self.path.rsplit("/", 1)[1]
            path = srv.clips.get(clip_id)
            if not path or not Path(path).is_file():
                self._send(b"clip not found", "text/plain", 404)
                return
            self._send(Path(path).read_bytes(), "video/mp4")
        else:
            self._send(b"not found", "text/plain", 404)

    def do_POST(self):
        srv: LabelServer = self.server  # type: ignore[assignment]
        if self.path != "/label":
            self._send(b"not found", "text/plain", 404)
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length))
            clip_id = payload["clip_id"]
            verdict = payload["verdict"]
            classes = payload.get("classes", [])
            assert verdict in ("pass", "fail")
            assert clip_id in srv.clips
            assert isinstance(classes, list)
            assert all(c in VALID_CLASSES for c in classes)
            assert (verdict == "pass") == (len(classes) == 0)
        except Exception as e:
            self._send(f"bad label payload: {e}".encode(), "text/plain", 400)
            return
        record = {
            "ts": round(time.time(), 3),
            "clip_id": clip_id,
            "verdict": verdict,
            "classes": classes,
            "note": str(payload.get("note", ""))[:500],
            "ms_spent": int(payload.get("ms", 0)),
            "clip_sha256": sha256_file(srv.clips[clip_id]),
        }
        with srv.lock, open(srv.labels_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        remaining = len(srv.rows) - len(srv.labeled_ids())
        self._send(json.dumps({"ok": True, "remaining": remaining}).encode(),
                   "application/json")


def make_server(cal_dir: str | Path, port: int = 0) -> LabelServer:
    cal = Path(cal_dir)
    manifest = cal / "batch_manifest.jsonl"
    rows = _read_jsonl(manifest)
    if not rows:
        raise SystemExit(f"[label] no rows in {manifest} — run calibrate-batch first")
    random.Random(42).shuffle(rows)   # blind the labeler to batch structure
    return LabelServer(("127.0.0.1", port), rows, cal / "labels.jsonl")


def main_label(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="shortsloop label")
    ap.add_argument("--calibration", default="calibration")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args(argv)
    srv = make_server(args.calibration, args.port)
    n = len(srv.rows)
    done = len(srv.labeled_ids())
    print(f"[label] {n} clips ({done} already labeled) — open "
          f"http://127.0.0.1:{srv.server_address[1]}/  (Ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print(f"\n[label] labels: {srv.labels_path} ({len(srv.labeled_ids())}/{n})")
    return 0
