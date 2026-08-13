"""Dispatch sheet intake (docs/plan.md §2.5 — FROZEN contract).

Input: a filled shorts-trend-dispatch markdown sheet — the ONLY accepted input
format. v1 accepts T2V manifest rows with no image dependency; every other row is
skipped LOUDLY (surfaced in the report), never silently. Intake failures refuse the
run before any GPU work: fail closed at 22:00, not at 3 a.m.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ClipSpec:
    clip_id: str                    # e.g. V1C3
    video_id: str                   # e.g. V1
    mode: str                       # T2V | I2V | ...
    image_dep: str                  # raw cell text; "" / "—" means none
    output_hint: str                # sheet's 輸出 cell, informational
    prompt: str | None = None


@dataclass
class DispatchSheet:
    path: str
    sha256: str
    clips: list[ClipSpec] = field(default_factory=list)      # accepted (T2V, no dep)
    skipped: list[dict] = field(default_factory=list)        # loud skips
    shared: dict = field(default_factory=dict)               # width/height/length/fps
    errors: list[str] = field(default_factory=list)          # intake refusal reasons


_ID_RANGE = re.compile(r"^(V\d+)C(\d+)\s*[–—-]\s*(?:V\d+)?C?(\d+)$")
_ID_SINGLE = re.compile(r"^(V\d+)C(\d+)$")
_HEADING = re.compile(r"^###\s+(V\d+C\d+)\b")
_RES = re.compile(r"(\d{3,4})\s*[x×]\s*(\d{3,4})")
_LEN_FPS = re.compile(r"(\d+)\s*frames.*?(\d+(?:\.\d+)?)\s*fps", re.S)

_NO_DEP = ("", "—", "-", "–", "none", "無", "无")


def _table_rows(lines: list[str], start: int) -> list[list[str]]:
    """Consume a markdown table starting at/after `start`; returns cell rows."""
    rows = []
    for line in lines[start:]:
        s = line.strip()
        if not s.startswith("|"):
            if rows:
                break
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if all(set(c) <= set(":- ") for c in cells):        # separator row
            continue
        rows.append(cells)
    return rows


def _expand_ids(cell: str) -> list[str] | None:
    cell = cell.replace("**", "").strip()
    m = _ID_SINGLE.match(cell)
    if m:
        return [cell]
    m = _ID_RANGE.match(cell)
    if m:
        video, lo, hi = m.group(1), int(m.group(2)), int(m.group(3))
        if hi < lo:
            return None
        return [f"{video}C{k}" for k in range(lo, hi + 1)]
    return None


def _parse_shared(text: str) -> dict:
    shared: dict = {}
    m = _RES.search(text)
    if m:
        shared["width"], shared["height"] = int(m.group(1)), int(m.group(2))
    m = _LEN_FPS.search(text)
    if m:
        shared["length"] = int(m.group(1))
        shared["fps"] = float(m.group(2))
    return shared


def _parse_prompts(lines: list[str]) -> dict[str, str]:
    """### V{n}C{k} — ... headings followed by a fenced block = the prompt."""
    prompts: dict[str, str] = {}
    i = 0
    while i < len(lines):
        m = _HEADING.match(lines[i])
        if not m:
            i += 1
            continue
        clip_id = m.group(1)
        j = i + 1
        while j < len(lines) and not lines[j].startswith("```"):
            if _HEADING.match(lines[j]) or lines[j].startswith("## "):
                break  # next section without a fence: no prompt for this heading
            j += 1
        if j < len(lines) and lines[j].startswith("```"):
            block: list[str] = []
            j += 1
            while j < len(lines) and not lines[j].startswith("```"):
                block.append(lines[j])
                j += 1
            prompts[clip_id] = "\n".join(block).strip()
        i = j + 1
    return prompts


def parse_dispatch(path: str | Path) -> DispatchSheet:
    from .verdict import sha256_file

    p = Path(path)
    sheet = DispatchSheet(path=str(p), sha256=sha256_file(p) or "")
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as e:
        sheet.errors.append(f"cannot read dispatch sheet: {e}")
        return sheet
    lines = text.splitlines()

    # shared parameters: first table after a heading containing 共用參數 (or the
    # first table of the file as fallback)
    shared_start = 0
    for i, line in enumerate(lines):
        if line.startswith("#") and "共用參數" in line:
            shared_start = i
            break
    shared_rows = _table_rows(lines, shared_start)
    sheet.shared = _parse_shared("\n".join("|".join(r) for r in shared_rows))

    # manifest: table under the heading containing 生產清單 or manifest
    manifest_start = None
    for i, line in enumerate(lines):
        if line.startswith("#") and ("生產清單" in line or "manifest" in line.lower()):
            manifest_start = i
            break
    if manifest_start is None:
        sheet.errors.append("no manifest section (生產清單/manifest heading) found")
        return sheet
    rows = _table_rows(lines, manifest_start)
    rows = [r for r in rows if r and not r[0].lower().startswith(("id", "編號"))]
    if not rows:
        sheet.errors.append("manifest table is empty")
        return sheet

    prompts = _parse_prompts(lines)

    for row in rows:
        if len(row) < 2:
            sheet.errors.append(f"manifest row malformed: {row!r}")
            continue
        id_cell, mode = row[0], row[1].upper().strip()
        image_dep = row[2].strip() if len(row) > 2 else ""
        output_hint = row[3].strip() if len(row) > 3 else ""
        ids = _expand_ids(id_cell)
        if ids is None:
            sheet.errors.append(f"manifest row has unparseable ID cell: {id_cell!r}")
            continue
        has_dep = image_dep.lower() not in _NO_DEP
        for clip_id in ids:
            video_id = clip_id.split("C")[0]
            if mode != "T2V" or has_dep:
                sheet.skipped.append({
                    "clip_id": clip_id, "mode": mode, "image_dep": image_dep,
                    "reason": "v2_mode: only independent T2V clips are in scope for v1",
                })
                continue
            spec = ClipSpec(clip_id=clip_id, video_id=video_id, mode=mode,
                            image_dep=image_dep, output_hint=output_hint,
                            prompt=prompts.get(clip_id))
            if not spec.prompt:
                sheet.errors.append(
                    f"{clip_id}: accepted T2V manifest row has no prompt block "
                    f"(### {clip_id} heading with a fenced prompt)")
            sheet.clips.append(spec)

    if not sheet.clips and not sheet.errors:
        sheet.errors.append("no accepted T2V clips in the manifest")
    return sheet


def expectations(sheet: DispatchSheet) -> dict:
    """L1 spec-check expectations from the shared-parameter table."""
    exp: dict = {}
    if "width" in sheet.shared:
        exp["width"] = sheet.shared["width"]
        exp["height"] = sheet.shared["height"]
    if "fps" in sheet.shared:
        exp["fps"] = sheet.shared["fps"]
    if "length" in sheet.shared:
        exp["frames"] = sheet.shared["length"]
        if "fps" in sheet.shared and sheet.shared["fps"]:
            exp["duration_s"] = round(sheet.shared["length"] / sheet.shared["fps"], 3)
    return exp
