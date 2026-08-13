# Vendored files — do not edit here

Copied unchanged from the `comfyui-ig-video` skill (2026-08-12) so the repo is
self-contained on the workstation. Fix bugs upstream in the skill, then re-copy.

| File | Source | sha256 |
|---|---|---|
| `comfy_client.py` | `comfyui-ig-video/scripts/comfy_client.py` | `a440dbb6b3af45f8078d59069a6bf5a56a900500c4ef0fbf4ce5100ff7b0463a` |
| `ig_encode.sh` | `comfyui-ig-video/scripts/ig_encode.sh` | `eaded93be6b659de5a2ea6e4a9971dac95dacb136a7acf8d610f0581b8245828` |

Integration contract (why subprocess, not import): the client's printed
`RESULT {...}` last line and exit codes (0 ok · 2 execution error · 3 timeout ·
4 bad workflow) are designed for automation; shortsloop's runner shells out with
`COMFY_HOST` set and parses that line. Tests point `COMFY_HOST` at a local fake
ComfyUI HTTP server, which exercises the vendored client and our runner together.
