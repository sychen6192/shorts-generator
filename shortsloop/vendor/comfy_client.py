#!/usr/bin/env python3
"""comfy_client.py — stdlib-only ComfyUI API client for agent-driven video generation.

Environment:
    COMFY_HOST      ComfyUI server address, host:port (default: 127.0.0.1:8188)

Commands:
    health                          Check server; show version / VRAM / queue depth.
    queue                           Show running + pending jobs.
    models [folder]                 List model folders, or files in one (e.g. loras).
    upload FILE                     Upload an input image; prints server-side name.
    run    -w FILE [patch opts]     Patch + submit + wait + download outputs.  (main)
    submit -w FILE [patch opts]     Patch + submit only; prints prompt_id.
    wait   PROMPT_ID [--out DIR]    Attach to an existing job; download when done.
    interrupt                       Interrupt the currently running job.
    free                            Unload models and free VRAM.

Patch options (run / submit) — every applied change is printed as "[patch] ...":
    --prompt TEXT        Positive prompt (located via sampler wiring -> CLIPTextEncode).
    --negative TEXT      Negative prompt.
    --width N --height N Video latent dimensions.
    --length N           Frame count (Wan: must be 4n+1, e.g. 81).
    --seed N             -1 = random (default). Concrete seed always printed.
    --steps N            Total steps; dual-sampler start/end boundaries rescale with it.
    --cfg X              CFG on all samplers.
    --fps N              fps / frame_rate on video create/save nodes.
    --image PATH         Upload + wire into LoadImage (repeatable; node-id order).
    --set N:KEY=VAL      Raw override on node N, input KEY (repeatable; VAL json-parsed).
    --dump               Print the patched workflow JSON and exit (dry-run).

Run options:
    --out DIR            Output directory (default ./outputs)
    --timeout SEC        Max wait (default 3600)
    --poll SEC           Poll interval (default 5)

Exit codes: 0 ok · 2 execution error · 3 timeout · 4 bad workflow/args.
Last line of run/wait is machine-readable:  RESULT {"ok":..., "seed":..., "files":[...]}
"""

import argparse
import json
import mimetypes
import os
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

HOST = os.environ.get("COMFY_HOST", "127.0.0.1:8188")
BASE = f"http://{HOST}"
CLIENT_ID = f"comfy-client-{uuid.uuid4().hex[:8]}"

# Latent/video-condition nodes whose width/height/length define output size.
SIZE_CLASSES = {
    "EmptyHunyuanLatentVideo", "Wan22ImageToVideoLatent", "WanImageToVideo",
    "WanFirstLastFrameToVideo", "Wan22FunControlToVideo", "WanFunControlToVideo",
    "WanVaceToVideo", "EmptyLatentImage", "EmptyLTXVLatentVideo",
    "EmptyCosmosLatentVideo", "EmptyMochiLatentVideo",
}


class HttpError(Exception):
    def __init__(self, code, body):
        super().__init__(f"HTTP {code}")
        self.code = code
        self.body = body


def http(method, path, data=None, headers=None, timeout=60, raw=False):
    """Minimal JSON-over-HTTP helper. Raises HttpError on HTTP errors,
    exits with a clear message when the server is unreachable."""
    body = None
    hdrs = dict(headers or {})
    if isinstance(data, (bytes, bytearray)):
        body = bytes(data)
    elif data is not None:
        body = json.dumps(data).encode()
        hdrs.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(BASE + path, data=body, method=method, headers=hdrs)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = r.read()
    except urllib.error.HTTPError as e:
        raise HttpError(e.code, e.read().decode(errors="replace"))
    except urllib.error.URLError as e:
        sys.exit(f"[comfy] cannot reach {BASE} ({e.reason}). "
                 f"Is ComfyUI running and COMFY_HOST correct?")
    if raw:
        return payload
    return json.loads(payload) if payload else {}


# --------------------------------------------------------------------------- workflow

def load_workflow(path):
    try:
        wf = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as e:
        sys.exit(f"[comfy] cannot read workflow {path}: {e}")
    if isinstance(wf, dict) and isinstance(wf.get("nodes"), list):
        sys.exit("[comfy] this is a UI-format workflow. Re-export with "
                 "ComfyUI -> Workflow -> Export (API) and use that file.")
    if not isinstance(wf, dict) or not all(
            isinstance(v, dict) and "class_type" in v for v in wf.values()):
        sys.exit(f"[comfy] {path} does not look like an API-format workflow.")
    return wf


def is_link(v):
    return (isinstance(v, list) and len(v) == 2
            and isinstance(v[0], (str, int)) and isinstance(v[1], int))


def resolve_text_node(wf, link, role, depth=0):
    """Follow a sampler's positive/negative link upstream to a CLIPTextEncode,
    passing through conditioning nodes (WanImageToVideo etc.)."""
    if depth > 6 or not is_link(link):
        return None
    nid = str(link[0])
    node = wf.get(nid)
    if not node:
        return None
    if node.get("class_type") == "CLIPTextEncode":
        return nid
    ins = node.get("inputs", {})
    nxt = ins.get(role) or ins.get("conditioning")
    return resolve_text_node(wf, nxt, role, depth + 1)


def find_text_nodes(wf):
    """Return ({positive node ids}, {negative node ids})."""
    pos, neg = set(), set()
    for node in wf.values():
        if not node.get("class_type", "").startswith("KSampler"):
            continue
        ins = node.get("inputs", {})
        p = resolve_text_node(wf, ins.get("positive"), "positive")
        n = resolve_text_node(wf, ins.get("negative"), "negative")
        if p:
            pos.add(p)
        if n:
            neg.add(n)
    if not pos or not neg:  # fallback: match by node title
        for nid, node in wf.items():
            if node.get("class_type") != "CLIPTextEncode":
                continue
            title = node.get("_meta", {}).get("title", "").lower()
            if "positive" in title and not pos:
                pos.add(nid)
            if "negative" in title and not neg:
                neg.add(nid)
    return pos, neg


def patch(wf, args, changes):
    def setin(nid, key, val):
        wf[nid]["inputs"][key] = val
        changes.append(f"{nid}({wf[nid]['class_type']}).{key} = {val!r}")

    # --- prompts
    if args.prompt is not None or args.negative is not None:
        pos, neg = find_text_nodes(wf)
        if args.prompt is not None:
            if not pos:
                sys.exit("[comfy] could not locate a positive CLIPTextEncode; "
                         "use --set NODE:text=... instead.")
            for nid in sorted(pos):
                setin(nid, "text", args.prompt)
        if args.negative is not None:
            if not neg:
                sys.exit("[comfy] could not locate a negative CLIPTextEncode; "
                         "use --set NODE:text=... instead.")
            for nid in sorted(neg):
                setin(nid, "text", args.negative)

    # --- size / frame count
    for nid in sorted(wf):
        node, ins = wf[nid], wf[nid].get("inputs", {})
        sizable = node.get("class_type") in SIZE_CLASSES or "length" in ins
        if not sizable:
            continue
        if args.width is not None and "width" in ins and not is_link(ins["width"]):
            setin(nid, "width", args.width)
        if args.height is not None and "height" in ins and not is_link(ins["height"]):
            setin(nid, "height", args.height)
        if args.length is not None and "length" in ins and not is_link(ins["length"]):
            setin(nid, "length", args.length)

    # --- seed (same seed on every sampler so dual-stage Wan reproduces exactly)
    seed = None
    if args.seed is not None:
        seed = args.seed if args.seed >= 0 else random.randint(0, 2**48)
        for nid in sorted(wf):
            node, ins = wf[nid], wf[nid].get("inputs", {})
            if node.get("class_type", "").startswith(("KSampler", "SamplerCustom")) \
                    or node.get("class_type") == "RandomNoise":
                for key in ("seed", "noise_seed"):
                    if key in ins and not is_link(ins[key]):
                        setin(nid, key, seed)

    # --- steps (rescale dual-sampler boundaries proportionally)
    if args.steps is not None:
        adv = [nid for nid in wf
               if "start_at_step" in wf[nid].get("inputs", {})
               and "end_at_step" in wf[nid].get("inputs", {})]
        old_total = max((wf[n]["inputs"].get("steps", 0) for n in adv), default=0) \
            if adv else 0
        for nid in sorted(wf):
            ins = wf[nid].get("inputs", {})
            if "steps" not in ins or is_link(ins["steps"]):
                continue
            if nid in adv and old_total > 0:
                scale = args.steps / old_total
                for key in ("start_at_step", "end_at_step"):
                    v = ins[key]
                    if isinstance(v, int) and not is_link(v):
                        if v > old_total:
                            nv = v                      # sentinel like 10000
                        elif v == old_total:
                            nv = args.steps
                        else:
                            nv = max(0, round(v * scale))
                        setin(nid, key, nv)
            setin(nid, "steps", args.steps)

    # --- cfg
    if args.cfg is not None:
        for nid in sorted(wf):
            ins = wf[nid].get("inputs", {})
            if "cfg" in ins and not is_link(ins["cfg"]):
                setin(nid, "cfg", args.cfg)

    # --- fps on video create/save nodes
    if args.fps is not None:
        for nid in sorted(wf):
            ins = wf[nid].get("inputs", {})
            for key in ("fps", "frame_rate"):
                if key in ins and not is_link(ins[key]):
                    setin(nid, key, args.fps)

    # --- input images (I2V / first-last-frame)
    if args.image:
        loaders = sorted(nid for nid in wf
                         if wf[nid].get("class_type") == "LoadImage")
        if not loaders:
            sys.exit("[comfy] --image given but workflow has no LoadImage node.")
        if len(args.image) > len(loaders):
            sys.exit(f"[comfy] {len(args.image)} images given but only "
                     f"{len(loaders)} LoadImage nodes exist.")
        for img_path, nid in zip(args.image, loaders):
            info = upload_image(img_path)
            name = info["name"]
            if info.get("subfolder"):
                name = f"{info['subfolder']}/{name}"
            setin(nid, "image", name)

    # --- raw overrides
    for spec in args.set or []:
        try:
            target, val = spec.split("=", 1)
            nid, key = target.split(":", 1)
        except ValueError:
            sys.exit(f"[comfy] bad --set '{spec}', expected NODE:KEY=VALUE")
        if nid not in wf:
            sys.exit(f"[comfy] --set: node {nid} not in workflow")
        try:
            val = json.loads(val)
        except json.JSONDecodeError:
            pass  # keep as string
        setin(nid, key, val)

    return seed


# --------------------------------------------------------------------------- api ops

def upload_image(path):
    p = Path(path)
    if not p.is_file():
        sys.exit(f"[comfy] image not found: {path}")
    mt = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
    boundary = uuid.uuid4().hex
    parts = [
        (f'--{boundary}\r\nContent-Disposition: form-data; name="image"; '
         f'filename="{p.name}"\r\nContent-Type: {mt}\r\n\r\n').encode()
        + p.read_bytes() + b"\r\n",
        (f'--{boundary}\r\nContent-Disposition: form-data; name="overwrite"'
         f'\r\n\r\ntrue\r\n').encode(),
        f"--{boundary}--\r\n".encode(),
    ]
    try:
        res = http("POST", "/upload/image", data=b"".join(parts),
                   headers={"Content-Type":
                            f"multipart/form-data; boundary={boundary}"},
                   timeout=300)
    except HttpError as e:
        sys.exit(f"[comfy] upload failed: HTTP {e.code}\n{e.body[:1000]}")
    print(f"[comfy] uploaded {p.name} -> {res.get('name')} "
          f"(type={res.get('type', 'input')})")
    return res


def submit(wf):
    try:
        res = http("POST", "/prompt", data={"prompt": wf, "client_id": CLIENT_ID})
    except HttpError as e:
        print(f"[comfy] submit rejected (HTTP {e.code}):", file=sys.stderr)
        try:
            err = json.loads(e.body)
            if err.get("error"):
                print(f"  {err['error'].get('type')}: "
                      f"{err['error'].get('message')}", file=sys.stderr)
            for nid, ne in (err.get("node_errors") or {}).items():
                cls = ne.get("class_type", "?")
                for detail in ne.get("errors", []):
                    print(f"  node {nid} ({cls}): {detail.get('message')} "
                          f"{detail.get('details', '')}", file=sys.stderr)
        except (json.JSONDecodeError, AttributeError):
            print(e.body[:2000], file=sys.stderr)
        sys.exit(4)
    pid = res.get("prompt_id")
    print(f"[comfy] queued prompt_id={pid}")
    return pid


def queue_position(pid):
    q = http("GET", "/queue")
    for item in q.get("queue_running", []):
        if len(item) > 1 and item[1] == pid:
            return 0
    for i, item in enumerate(q.get("queue_pending", []), start=1):
        if len(item) > 1 and item[1] == pid:
            return i
    return None


def print_exec_errors(entry):
    for msg in entry.get("status", {}).get("messages", []):
        if not (isinstance(msg, list) and len(msg) == 2):
            continue
        kind, data = msg
        if kind == "execution_error":
            print(f"[comfy] ERROR in node {data.get('node_id')} "
                  f"({data.get('node_type')}): "
                  f"{data.get('exception_type')}: "
                  f"{data.get('exception_message')}", file=sys.stderr)
            emsg = str(data.get("exception_message", "")).lower()
            if "out of memory" in emsg or "cuda" in emsg and "memory" in emsg:
                print("[comfy] hint: run `comfy_client.py free`, then retry with "
                      "smaller --width/--height/--length.", file=sys.stderr)
        elif kind == "execution_interrupted":
            print("[comfy] job was interrupted.", file=sys.stderr)


def download_outputs(entry, out_dir, pid):
    outdir = Path(out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    saved = []
    for node_out in (entry.get("outputs") or {}).values():
        for val in node_out.values():
            if not isinstance(val, list):
                continue
            for item in val:
                if not (isinstance(item, dict) and "filename" in item):
                    continue
                if item.get("type") == "temp":     # previews
                    continue
                qs = urllib.parse.urlencode({
                    "filename": item["filename"],
                    "subfolder": item.get("subfolder", ""),
                    "type": item.get("type", "output"),
                })
                data = http("GET", f"/view?{qs}", raw=True, timeout=1800)
                dest = outdir / Path(item["filename"]).name
                if dest.exists():
                    dest = outdir / f"{pid[:8]}_{Path(item['filename']).name}"
                dest.write_bytes(data)
                print(f"[comfy] saved {dest} ({len(data)/1e6:.1f} MB)")
                saved.append(str(dest))
    return saved


def wait_and_download(pid, out_dir, timeout, poll):
    t0 = time.time()
    last_note = 0.0
    while True:
        hist = http("GET", f"/history/{pid}")
        if pid in hist:
            entry = hist[pid]
            status = entry.get("status", {})
            elapsed = round(time.time() - t0, 1)
            if status.get("status_str") == "error":
                print_exec_errors(entry)
                print("RESULT " + json.dumps(
                    {"ok": False, "prompt_id": pid, "elapsed_sec": elapsed}))
                sys.exit(2)
            files = download_outputs(entry, out_dir, pid)
            print(f"[comfy] done in {elapsed}s, {len(files)} file(s)")
            return files, elapsed
        if time.time() - t0 > timeout:
            print(f"[comfy] timeout after {timeout}s "
                  f"(job may still be running; `wait {pid}` to re-attach, "
                  f"or `interrupt`).", file=sys.stderr)
            sys.exit(3)
        if time.time() - last_note >= 15:
            pos = queue_position(pid)
            state = ("running" if pos == 0
                     else f"pending #{pos}" if pos else "not in queue (starting?)")
            print(f"[comfy] {state} … {int(time.time() - t0)}s")
            last_note = time.time()
        time.sleep(poll)


# --------------------------------------------------------------------------- commands

def cmd_health(_):
    stats = http("GET", "/system_stats")
    sysinfo = stats.get("system", {})
    print(f"[comfy] server OK at {BASE}  "
          f"(comfyui {sysinfo.get('comfyui_version', '?')}, "
          f"python {sysinfo.get('python_version', '?').split()[0]})")
    for dev in stats.get("devices", []):
        free = dev.get("vram_free", 0) / 2**30
        total = dev.get("vram_total", 0) / 2**30
        print(f"[comfy] {dev.get('name')}: VRAM {free:.1f} / {total:.1f} GB free")
    q = http("GET", "/queue")
    print(f"[comfy] queue: {len(q.get('queue_running', []))} running, "
          f"{len(q.get('queue_pending', []))} pending")


def cmd_queue(_):
    q = http("GET", "/queue")
    for label in ("queue_running", "queue_pending"):
        for item in q.get(label, []):
            pid = item[1] if len(item) > 1 else "?"
            print(f"{label.split('_')[1]}: {pid}")
    if not q.get("queue_running") and not q.get("queue_pending"):
        print("queue empty")


def cmd_models(args):
    path = f"/models/{args.folder}" if args.folder else "/models"
    try:
        for name in http("GET", path):
            print(name)
    except HttpError as e:
        sys.exit(f"[comfy] HTTP {e.code} — folder '{args.folder}' unknown? "
                 f"Try `models` with no argument to list folders.")


def cmd_upload(args):
    info = upload_image(args.file)
    print(json.dumps(info))


def cmd_run(args, wait=True):
    wf = load_workflow(args.workflow)
    changes = []
    seed = patch(wf, args, changes)
    for c in changes:
        print(f"[patch] {c}")
    if seed is not None:
        print(f"[comfy] seed = {seed}")
    if args.dump:
        print(json.dumps(wf, indent=2, ensure_ascii=False))
        return
    pid = submit(wf)
    if not wait:
        print("RESULT " + json.dumps({"ok": True, "prompt_id": pid, "seed": seed}))
        return
    files, elapsed = wait_and_download(pid, args.out, args.timeout, args.poll)
    print("RESULT " + json.dumps({"ok": True, "prompt_id": pid, "seed": seed,
                                  "elapsed_sec": elapsed, "files": files}))


def cmd_wait(args):
    files, elapsed = wait_and_download(args.prompt_id, args.out,
                                       args.timeout, args.poll)
    print("RESULT " + json.dumps({"ok": True, "prompt_id": args.prompt_id,
                                  "elapsed_sec": elapsed, "files": files}))


def cmd_interrupt(_):
    http("POST", "/interrupt", data={})
    print("[comfy] interrupt sent")


def cmd_free(_):
    http("POST", "/free", data={"unload_models": True, "free_memory": True})
    print("[comfy] models unloaded, VRAM freed")


def add_patch_opts(p):
    p.add_argument("-w", "--workflow", required=True)
    p.add_argument("--prompt")
    p.add_argument("--negative")
    p.add_argument("--width", type=int)
    p.add_argument("--height", type=int)
    p.add_argument("--length", type=int)
    p.add_argument("--seed", type=int, default=-1)
    p.add_argument("--steps", type=int)
    p.add_argument("--cfg", type=float)
    p.add_argument("--fps", type=float)
    p.add_argument("--image", action="append")
    p.add_argument("--set", action="append", metavar="NODE:KEY=VALUE")
    p.add_argument("--dump", action="store_true")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("health")
    sub.add_parser("queue")
    p = sub.add_parser("models")
    p.add_argument("folder", nargs="?")
    p = sub.add_parser("upload")
    p.add_argument("file")

    for name in ("run", "submit"):
        p = sub.add_parser(name)
        add_patch_opts(p)
        p.add_argument("--out", default="./outputs")
        p.add_argument("--timeout", type=int, default=3600)
        p.add_argument("--poll", type=int, default=5)

    p = sub.add_parser("wait")
    p.add_argument("prompt_id")
    p.add_argument("--out", default="./outputs")
    p.add_argument("--timeout", type=int, default=3600)
    p.add_argument("--poll", type=int, default=5)

    sub.add_parser("interrupt")
    sub.add_parser("free")

    args = ap.parse_args()
    {
        "health": cmd_health,
        "queue": cmd_queue,
        "models": cmd_models,
        "upload": cmd_upload,
        "run": lambda a: cmd_run(a, wait=True),
        "submit": lambda a: cmd_run(a, wait=False),
        "wait": cmd_wait,
        "interrupt": cmd_interrupt,
        "free": cmd_free,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
