"""Labeling tool: byte-range serving (Safari), --host for headless workstations,
and the page's key handler (no double-send on auto-repeat / fast presses)."""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager

import pytest

from shortsloop.label import main_label, make_server


def _cal(clips, tmp_path):
    cal = tmp_path / "cal"
    cal.mkdir()
    with open(cal / "batch_manifest.jsonl", "w") as f:
        f.write(json.dumps({"clip_id": "cal001", "kind": "good", "prompt": "a moving cube",
                            "clip_path": str(clips["moving"])}) + "\n")
        f.write(json.dumps({"clip_id": "cal002", "kind": "starved",
                            "prompt": "a static vase",
                            "clip_path": str(clips["static"])}) + "\n")
    return cal


@contextmanager
def _serve(cal, **kw):
    srv = make_server(cal, **kw)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield srv, f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()
        srv.server_close()


def _get(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


# ------------------------------------------------------------ [100] HTTP Range

def test_clip_supports_single_byte_ranges(clips, tmp_path):
    data = clips["moving"].read_bytes()
    size = len(data)
    with _serve(_cal(clips, tmp_path)) as (_srv, base):
        status, headers, body = _get(base + "/clip/cal001")
        assert status == 200 and body == data
        assert headers.get("Accept-Ranges") == "bytes"

        status, headers, body = _get(base + "/clip/cal001", {"Range": "bytes=0-99"})
        assert status == 206 and body == data[:100]
        assert headers["Content-Range"] == f"bytes 0-99/{size}"
        assert headers["Content-Length"] == "100"
        assert headers.get("Accept-Ranges") == "bytes"

        status, headers, body = _get(base + "/clip/cal001", {"Range": "bytes=100-"})
        assert status == 206 and body == data[100:]
        assert headers["Content-Range"] == f"bytes 100-{size - 1}/{size}"

        status, headers, body = _get(base + "/clip/cal001", {"Range": "bytes=-50"})
        assert status == 206 and body == data[-50:]

        status, headers, body = _get(base + "/clip/cal001",
                                     {"Range": f"bytes=0-{size + 999}"})   # clamp end
        assert status == 206 and body == data

        status, headers, _ = _get(base + "/clip/cal001", {"Range": f"bytes={size}-"})
        assert status == 416 and headers["Content-Range"] == f"bytes */{size}"

        status, _, _ = _get(base + "/clip/nope", {"Range": "bytes=0-9"})
        assert status == 404


# ------------------------------------------------------------- [119] --host

def test_host_option_binds_the_requested_address(clips, tmp_path):
    srv = make_server(_cal(clips, tmp_path), port=0, host="0.0.0.0")
    try:
        assert srv.server_address[0] == "0.0.0.0"
    finally:
        srv.server_close()


def test_default_bind_is_localhost_with_a_tunnel_hint(clips, tmp_path, capsys,
                                                      monkeypatch):
    from shortsloop import label as label_mod
    seen = {}

    def fake_serve(self):
        seen["addr"] = self.server_address
        raise KeyboardInterrupt

    monkeypatch.setattr(label_mod.LabelServer, "serve_forever", fake_serve)
    assert main_label(["--calibration", str(_cal(clips, tmp_path)), "--port", "0"]) == 0
    out = capsys.readouterr().out
    assert seen["addr"][0] == "127.0.0.1"
    assert "ssh -L" in out and "--host" in out


# --------------------------------------------- [99] key auto-repeat / fast keys

NODE = shutil.which("node")

HARNESS = r"""
const handlers = {}, els = {};
function el(id) {
  if (!els[id]) els[id] = {id, textContent: "", style: {}, src: "", currentTime: 0,
    removeAttribute() {}, load() {}, play() { return Promise.resolve(); }};
  return els[id];
}
globalThis.document = {getElementById: el,
                       addEventListener: (t, fn) => { handlers[t] = fn; }};
const posts = [], pending = [];
globalThis.fetch = (url, opts) => {
  const b = JSON.parse(opts.body);
  posts.push(b.clip_id + ":" + b.verdict);
  return new Promise(res => pending.push(res));
};
globalThis.alert = (m) => posts.push("ALERT " + m);
globalThis.prompt = () => "note";
const key = (k, repeat = false) => handlers.keydown({key: k, repeat, preventDefault() {}});
const flush = () => new Promise(r => setTimeout(r, 0));
__SCRIPT__
(async () => {
  const out = {};
  key("1"); key("1", true); key("3");          // held key + fast "correction"
  out.burst = posts.slice();
  pending.shift()({ok: true, text: async () => ""}); await flush();
  key("2");
  out.second = posts.slice();
  pending.shift()({ok: true, text: async () => ""}); await flush();
  out.done = el("done").style.display || "";
  console.log(JSON.stringify(out));
})();
"""


def _run_page(page: str, script_extra: str = "") -> dict:
    script = page.split("<script>\n", 1)[1].split("</script>", 1)[0]
    js = HARNESS.replace("__SCRIPT__", script + script_extra)
    res = subprocess.run([NODE, "-e", js], capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(NODE is None, reason="node not installed (browser-logic test)")
def test_page_sends_one_label_per_clip_despite_repeat_and_fast_keys(clips, tmp_path):
    with _serve(_cal(clips, tmp_path)) as (_srv, base):
        _, _, body = _get(base + "/")
    page = body.decode()
    data = json.loads(page.split("const DATA = ", 1)[1].split(";\n", 1)[0])
    first, second = (r["clip_id"] for r in data["rows"])
    out = _run_page(page)
    assert out["burst"] == [f"{first}:pass"]               # one POST, not three
    assert out["second"] == [f"{first}:pass", f"{second}:fail"]   # nothing skipped
    assert out["done"] == "block"


@pytest.mark.skipif(NODE is None, reason="node not installed (browser-logic test)")
def test_page_does_not_claim_done_when_clips_were_skipped(clips, tmp_path):
    with _serve(_cal(clips, tmp_path)) as (_srv, base):
        _, _, body = _get(base + "/")
    script = body.decode().split("<script>\n", 1)[1].split("</script>", 1)[0]
    js = HARNESS.split("(async () => {", 1)[0].replace("__SCRIPT__", script) + r"""
key("n"); key("n");
console.log(JSON.stringify({done: el("done").style.display || "",
                            end: el("end").textContent}));
"""
    res = subprocess.run([NODE, "-e", js], capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, res.stderr
    out = json.loads(res.stdout.strip().splitlines()[-1])
    assert out["done"] != "block"
    assert "2 clip(s) still unlabeled" in out["end"]
