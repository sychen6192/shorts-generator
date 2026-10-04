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
    anchors: list[str] = field(default_factory=list)  # video's 視覺錨/角色錨 (verbatim)


@dataclass
class DispatchSheet:
    path: str
    sha256: str
    clips: list[ClipSpec] = field(default_factory=list)      # accepted (T2V, no dep)
    skipped: list[dict] = field(default_factory=list)        # loud skips
    shared: dict = field(default_factory=dict)               # width/height/length/fps
    errors: list[str] = field(default_factory=list)          # intake refusal reasons


_ID_RANGE = re.compile(r"^(V\d+)C(\d+)\s*[–—\-~～]\s*(V\d+)?C?(\d+)$")
_ID_SINGLE = re.compile(r"^(V\d+)C(\d+)$")
_HEADING = re.compile(r"^###\s+(V\d+C\d+)\b")
_RES = re.compile(r"(\d{3,4})\s*[x×X*ｘ✕]\s*(\d{3,4})")
_FRAMES = re.compile(r"(?:length\s*[=:]\s*(\d+))|(?:(\d+)\s*(?:frames?|幀|帧|f)\b)", re.I)
_FPS = re.compile(r"(?:fps\s*[=:]\s*(\d+(?:\.\d+)?))|(?:(\d+(?:\.\d+)?)\s*fps)", re.I)

# "no image dependency" spellings seen in real sheets (after markdown/parenthetical
# stripping): dashes of every width, n/a, 無/无 variants.
_NO_DEP = {"", "-", "–", "—", "——", "－", "−", "none", "n/a", "na", "無", "无",
           "無依賴", "无依赖", "無依赖", "无依賴"}
_PAREN = re.compile(r"[（(][^）)]*[）)]")
_VIDEO = re.compile(r"^##\s+Video\s*#?\s*(\d+)", re.I)
# "視覺錨(一字不改):..." / "角色錨:..." — anchor phrases the rewrite must keep verbatim
_ANCHOR = re.compile(r"^\s*[\u4e00-\u9fff]{0,4}錨[^:：]{0,20}[:：]\s*(.+?)\s*$")


def _parse_anchors(lines: list[str]) -> dict[str, list[str]]:
    anchors: dict[str, list[str]] = {}
    video = None
    for line in lines:
        m = _VIDEO.match(line)
        if m:
            video = f"V{m.group(1)}"
            continue
        if line.startswith("## "):
            video = None
            continue
        m = _ANCHOR.match(line)
        if video and m:
            anchors.setdefault(video, []).append(m.group(1))
    return anchors


def _clean(cell: str) -> str:
    """Strip markdown emphasis/code and surrounding whitespace from a table cell."""
    return cell.replace("**", "").replace("`", "").strip()


def _norm_mode(cell: str) -> str:
    m = _PAREN.sub("", _clean(cell)).strip().upper()
    return "T2V" if m.startswith("T2V") else m


def _has_dep(cell: str) -> bool:
    d = _PAREN.sub("", _clean(cell)).strip().lower()
    return d not in _NO_DEP


def _section_tables(lines: list[str], start: int) -> list[list[str]]:
    """Every table row from `start` up to the next level-1/2 heading — a manifest
    split across tables, sub-headings or stray blank lines is still ONE manifest."""
    rows = []
    for line in lines[start + 1:]:
        if line.startswith("## ") or line.startswith("# "):
            break
        s = line.strip()
        if not s.startswith("|"):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if all(set(c) <= set(":- ") for c in cells):        # separator row
            continue
        rows.append(cells)
    return rows


def _is_header(row: list[str]) -> bool:
    return _clean(row[0]).lower() in ("id", "編號", "编号", "項目", "项目")


def _expand_ids(cell: str) -> list[str] | None:
    cell = _clean(cell)
    m = _ID_SINGLE.match(cell)
    if m:
        return [cell]
    m = _ID_RANGE.match(cell)
    if m:
        video, lo, end_video, hi = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
        if hi < lo or (end_video and end_video != video):
            return None                     # descending or cross-video: ambiguous
        return [f"{video}C{k}" for k in range(lo, hi + 1)]
    return None


def _parse_shared(rows: list[list[str]], errors: list[str]) -> dict:
    """Shared parameters by ROW LABEL (plan §2.5). A row that is present but
    unparseable/ambiguous is an intake error; an absent row falls back to defaults
    (the runner logs the fallback)."""
    shared: dict = {}
    for row in rows:
        if len(row) < 2:
            continue
        label, value = _clean(row[0]).lower(), _clean(" | ".join(row[1:]))
        if "解析度" in label or "分辨率" in label or "resolution" in label:
            found = sorted({(int(w), int(h)) for w, h in _RES.findall(value)})
            if len(found) != 1:
                errors.append(f"共用參數 解析度 {value!r}: expected exactly one WxH, "
                              f"found {len(found)} — state the one resolution to "
                              f"generate at")
                continue
            shared["width"], shared["height"] = found[0]
        elif "length" in label or "fps" in label or "長度" in label:
            frames = [next(g for g in m if g) for m in _FRAMES.findall(value)]
            fps = [next(g for g in m if g) for m in _FPS.findall(value)]
            if len(set(frames)) != 1 or len(set(fps)) != 1:
                errors.append(f"共用參數 length / fps {value!r}: expected one frame "
                              f"count and one fps (e.g. '81 frames(5.0 s)/ 16 fps')")
                continue
            length, rate = int(frames[0]), float(fps[0])
            if (length - 1) % 4 != 0:
                errors.append(f"共用參數 length {length} is not 4n+1 — Wan decodes "
                              f"((L-1)//4)*4+1 frames, so every clip would fail its "
                              f"spec check (use e.g. 81 or 121)")
                continue
            if rate <= 0:
                errors.append(f"共用參數 fps {rate} must be positive")
                continue
            shared["length"], shared["fps"] = length, rate
    return shared


def _parse_prompts(lines: list[str], errors: list[str]) -> dict[str, str]:
    """### V{n}C{k} — ... headings followed by a fenced block = the prompt
    (verbatim). A clip ID may carry exactly one prompt: a second heading for the
    same ID is an intake error, never a silent overwrite."""
    prompts: dict[str, str] = {}
    i = 0
    while i < len(lines):
        m = _HEADING.match(lines[i])
        if not m:
            i += 1
            continue
        clip_id, heading_line = m.group(1), i + 1
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
            if clip_id in prompts:
                errors.append(f"{clip_id}: duplicate prompt heading (line "
                              f"{heading_line}) — a clip has exactly one prompt; "
                              f"rename or remove the second block")
            else:
                prompts[clip_id] = "\n".join(block).strip()
            i = j + 1
        else:
            i = j          # re-examine the line that ended the search
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

    # shared parameters: the 共用參數 section's table(s), looked up by row label
    for i, line in enumerate(lines):
        if line.startswith("#") and "共用參數" in line:
            sheet.shared = _parse_shared(_section_tables(lines, i), sheet.errors)
            break

    # manifest: every table row in the section under the 生產清單/manifest heading
    manifest_start = None
    for i, line in enumerate(lines):
        if line.startswith("#") and ("生產清單" in line or "manifest" in line.lower()):
            manifest_start = i
            break
    if manifest_start is None:
        sheet.errors.append("no manifest section (生產清單/manifest heading) found")
        return sheet
    rows = [r for r in _section_tables(lines, manifest_start) if r and not _is_header(r)]
    if not rows:
        sheet.errors.append("manifest table is empty")
        return sheet

    prompts = _parse_prompts(lines, sheet.errors)
    anchors = _parse_anchors(lines)

    seen: dict[str, str] = {}
    for row in rows:
        if len(row) < 2:
            sheet.errors.append(f"manifest row malformed: {row!r}")
            continue
        id_cell, mode = row[0], _norm_mode(row[1])
        image_dep = row[2].strip() if len(row) > 2 else ""
        output_hint = row[3].strip() if len(row) > 3 else ""
        accepted_mode = mode == "T2V" and not _has_dep(image_dep)
        ids = _expand_ids(id_cell)
        if ids is None:
            if accepted_mode:
                sheet.errors.append(f"manifest row has unparseable ID cell: {id_cell!r} "
                                    f"(use V{{n}}C{{k}} or a same-video range "
                                    f"V{{n}}C{{a}}–V{{n}}C{{b}})")
            else:   # out-of-scope row anyway: skip it loudly, don't refuse the night
                sheet.skipped.append({
                    "clip_id": _clean(id_cell), "mode": mode, "image_dep": image_dep,
                    "reason": "v2_mode: only independent T2V clips are in scope for v1",
                })
            continue
        for clip_id in ids:
            if clip_id in seen:
                sheet.errors.append(
                    f"{clip_id}: duplicate manifest ID (rows {seen[clip_id]!r} and "
                    f"{id_cell!r}) — two work items would overwrite each other's files")
                continue
            seen[clip_id] = id_cell
            video_id = clip_id.split("C")[0]
            if not accepted_mode:
                sheet.skipped.append({
                    "clip_id": clip_id, "mode": mode, "image_dep": image_dep,
                    "reason": "v2_mode: only independent T2V clips are in scope for v1",
                })
                continue
            spec = ClipSpec(clip_id=clip_id, video_id=video_id, mode=mode,
                            image_dep=image_dep, output_hint=output_hint,
                            prompt=prompts.get(clip_id),
                            anchors=list(anchors.get(video_id, [])))
            if not spec.prompt:
                sheet.errors.append(
                    f"{clip_id}: accepted T2V manifest row has no prompt block "
                    f"(### {clip_id} heading with a fenced prompt)")
            sheet.clips.append(spec)

    for clip_id in prompts:
        if clip_id not in seen:
            sheet.errors.append(
                f"{clip_id}: prompt heading has no manifest row — the clip would be "
                f"silently dropped; add it to the 生產清單 table (or remove the block)")

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
