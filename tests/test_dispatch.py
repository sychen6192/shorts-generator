"""Dispatch intake contract tests (plan §5 row 13)."""

from __future__ import annotations

from pathlib import Path

from shortsloop.dispatch import expectations, parse_dispatch

DATA = Path(__file__).parent / "data"


def test_parse_small_sheet():
    sheet = parse_dispatch(DATA / "dispatch_small.md")
    assert sheet.errors == []
    ids = [c.clip_id for c in sheet.clips]
    assert ids == ["V1C1", "V1C2", "V2C1"]          # range expanded, I2V excluded
    assert all(c.mode == "T2V" for c in sheet.clips)
    assert sheet.clips[0].video_id == "V1"
    assert "kinetic sand sliding" in sheet.clips[0].prompt
    assert sheet.clips[2].prompt.endswith("vertical 9:16 composition.")
    # the I2V chain row is skipped LOUDLY, not silently
    assert len(sheet.skipped) == 1
    assert sheet.skipped[0]["clip_id"] == "V2C2"
    assert "v1" in sheet.skipped[0]["reason"]
    # shared parameters parsed
    assert sheet.shared == {"width": 480, "height": 832, "length": 49, "fps": 16.0}


def test_expectations_derived():
    sheet = parse_dispatch(DATA / "dispatch_small.md")
    exp = expectations(sheet)
    assert exp["width"] == 480 and exp["height"] == 832
    assert exp["frames"] == 49 and exp["fps"] == 16.0
    assert abs(exp["duration_s"] - 49 / 16) < 0.01


def test_missing_prompt_is_intake_error(tmp_path):
    sheet_md = (DATA / "dispatch_small.md").read_text(encoding="utf-8")
    # remove V1C2's prompt section entirely
    broken = sheet_md.replace("### V1C2", "### V9C9")
    p = tmp_path / "broken.md"
    p.write_text(broken, encoding="utf-8")
    sheet = parse_dispatch(p)
    assert any("V1C2" in e and "no prompt" in e for e in sheet.errors)


def test_empty_sheet_is_intake_error(tmp_path):
    p = tmp_path / "empty.md"
    p.write_text("# nothing here\n", encoding="utf-8")
    sheet = parse_dispatch(p)
    assert sheet.errors
    assert sheet.clips == []


def test_missing_file_is_intake_error(tmp_path):
    sheet = parse_dispatch(tmp_path / "nope.md")
    assert sheet.errors


def test_duplicate_manifest_ids_refuse_intake(tmp_path):
    """Overlapping rows would make two work items write the same clip/verdict/
    encode paths — refuse at intake, before any GPU work (plan §2.5)."""
    sheet_md = (DATA / "dispatch_small.md").read_text(encoding="utf-8")
    dup = sheet_md.replace("| V2C1 | T2V | — | `v2/c1.mp4` |",
                           "| V2C1 | T2V | — | `v2/c1.mp4` |\n| V1C2 | T2V | — | `x.mp4` |")
    p = tmp_path / "dup.md"
    p.write_text(dup, encoding="utf-8")
    sheet = parse_dispatch(p)
    assert any("V1C2" in e and "duplicate" in e for e in sheet.errors)


# ------------------------------------------------- template-faithful sheet (skill)

import re  # noqa: E402

import pytest  # noqa: E402

FULL = DATA / "dispatch_full.md"   # follows shorts-trend-dispatch's dispatch_template.md
FULL_ACCEPTED = ["V1C1", "V1C2", "V1C3", "V1C4", "V1C5", "V1C6", "V2C1", "V2C2",
                 "V2C3", "V2C6", "V3C1", "V3C5", "V3C6", "V4C1"]
FULL_SKIPPED = ["V2C4", "V2C5", "V3C2", "V3C3", "V3C4", "V4C2", "V4C3", "V4C4",
                "V4C5", "V4C6"]


def _variant(tmp_path, old, new, count=1, src=FULL):
    text = src.read_text(encoding="utf-8")
    assert text.count(old) >= 1, old
    p = tmp_path / "variant.md"
    p.write_text(text.replace(old, new, count), encoding="utf-8")
    return parse_dispatch(p)


def test_full_template_sheet_parses():
    sheet = parse_dispatch(FULL)
    assert sheet.errors == []
    assert [c.clip_id for c in sheet.clips] == FULL_ACCEPTED
    assert sorted(s["clip_id"] for s in sheet.skipped) == sorted(FULL_SKIPPED)
    assert sheet.shared == {"width": 720, "height": 1280, "length": 81, "fps": 16.0}
    assert all(c.prompt and c.prompt.rstrip(".").endswith("composition")
               for c in sheet.clips)


@pytest.mark.parametrize("old, new", [
    ("| V2C6 | T2V |", "\n| V2C6 | T2V |"),                                # blank line
    ("| V2C4 | I2V |", "\n**Phase 2 chain**\n\n| ID | 模式 | image 依賴 | 輸出 |\n"
                       "|---|---|---|---|\n| V2C4 | I2V |"),                # split table
    ("| V3C1 | T2V |", "\n### V3 manifest\n\n| ID | 模式 | image 依賴 | 輸出 |\n"
                       "|---|---|---|---|\n| V3C1 | T2V |"),                # sub-table
])
def test_manifest_rows_after_first_table_are_not_dropped(tmp_path, old, new):
    sheet = _variant(tmp_path, old, new)
    assert sheet.errors == []
    assert [c.clip_id for c in sheet.clips] == FULL_ACCEPTED
    assert sorted(s["clip_id"] for s in sheet.skipped) == sorted(FULL_SKIPPED)


def test_orphan_prompt_heading_is_intake_error(tmp_path):
    sheet = _variant(tmp_path, "| V3C1 | T2V | — | `v3/c1.mp4` |\n", "")
    assert any("V3C1" in e and "manifest" in e for e in sheet.errors)


def test_duplicate_prompt_heading_is_intake_error(tmp_path):
    sheet = _variant(tmp_path, "## Master Runbook",
                     "### V1C1 — #1 重用(排名字卡)\n```\ndrawtext=text='#1'\n```\n\n"
                     "## Master Runbook")
    assert any("V1C1" in e and "duplicate" in e for e in sheet.errors)


@pytest.mark.parametrize("cell, want", [
    ("720×1280(final)", (720, 1280)),
    ("720 X 1280", (720, 1280)),
    ("720*1280", (720, 1280)),
    ("720ｘ1280", (720, 1280)),
])
def test_resolution_formats(tmp_path, cell, want):
    sheet = _variant(tmp_path, "720x1280(無人值守直上 final,跳過 draft)", cell)
    assert sheet.errors == []
    assert (sheet.shared["width"], sheet.shared["height"]) == want


def test_ambiguous_resolution_cell_is_intake_error(tmp_path):
    sheet = _variant(tmp_path, "720x1280(無人值守直上 final,跳過 draft)",
                     "draft 480x832 → final 720x1280")
    assert any("解析度" in e for e in sheet.errors)


def test_resolution_is_read_from_its_row_only(tmp_path):
    sheet = _variant(tmp_path, "| T2V workflow | wan22_t2v_14B_api.json |",
                     "| T2V workflow | wan22_t2v_14B_480x832_api.json |")
    assert sheet.errors == []
    assert (sheet.shared["width"], sheet.shared["height"]) == (720, 1280)


@pytest.mark.parametrize("cell, length, fps", [
    ("121 frames(7.5 s)/ 16 fps", 121, 16.0),
    ("121 幀(7.5 秒)/ 16 fps", 121, 16.0),
    ("length=81 / fps=16", 81, 16.0),
    ("16 fps / 81 frames", 81, 16.0),
])
def test_length_fps_formats(tmp_path, cell, length, fps):
    sheet = _variant(tmp_path, "81 frames(5.0 s)/ 16 fps", cell)
    assert sheet.errors == []
    assert (sheet.shared["length"], sheet.shared["fps"]) == (length, fps)


@pytest.mark.parametrize("cell, needle", [
    ("120 frames(7.5 s)/ 16 fps", "4n+1"),          # Wan latent: ((L-1)//4)+1
    ("about five seconds", "length / fps"),         # row present but unparseable
])
def test_bad_length_fps_is_intake_error(tmp_path, cell, needle):
    sheet = _variant(tmp_path, "81 frames(5.0 s)/ 16 fps", cell)
    assert any(needle in e for e in sheet.errors), sheet.errors


def test_non_t2v_row_with_freeform_id_is_skipped_not_fatal(tmp_path):
    sheet = _variant(tmp_path, "| V4C2–V4C6 | 重用 |", "| V4 #5–#1 | 重用 |")
    assert sheet.errors == []
    assert any(s["clip_id"] == "V4 #5–#1" and s["mode"] == "重用" for s in sheet.skipped)


def test_cross_video_range_is_intake_error(tmp_path):
    sheet = _variant(tmp_path, "| V3C5–V3C6 | T2V |", "| V3C5–V4C6 | T2V |")
    assert any("V3C5–V4C6" in e for e in sheet.errors)


@pytest.mark.parametrize("mode, dep", [
    ("**T2V**", "—"), ("`T2V`", "—"), ("T2V(錨定)", "—"), ("t2v", "—"),
    ("T2V", "——"), ("T2V", "N/A"), ("T2V", "無依賴"), ("T2V", "(無)"),
    ("T2V", "`—`"), ("T2V", "－"), ("T2V", "—(chain parent)"),
])
def test_independent_t2v_spellings_accepted(tmp_path, mode, dep):
    sheet = _variant(tmp_path, "| V1C1–V1C6 | T2V | — |", f"| V1C1–V1C6 | {mode} | {dep} |")
    assert sheet.errors == []
    assert [c.clip_id for c in sheet.clips] == FULL_ACCEPTED


def test_heading_without_fence_blames_only_that_clip(tmp_path):
    text = FULL.read_text(encoding="utf-8")
    start = text.index("### V1C2")
    fence_open = text.index("```", start)
    fence_close = text.index("```", fence_open + 3) + 3
    p = tmp_path / "nofence.md"
    p.write_text(text[:fence_open] + "(prompt 見上方錨,早上補)" + text[fence_close:],
                 encoding="utf-8")
    sheet = parse_dispatch(p)
    blamed = sorted({m for e in sheet.errors for m in re.findall(r"V\dC\d", e)})
    assert blamed == ["V1C2"]


def test_motion_phrase_lands_before_wrapped_composition_tail():
    from shortsloop.policy import MOTION_PHRASES, append_motion_phrase
    sheet = parse_dispatch(FULL)
    v1c2 = next(c for c in sheet.clips if c.clip_id == "V1C2").prompt
    assert v1c2.endswith("vertical 9:16\ncomposition.")        # hard-wrapped tail
    out = append_motion_phrase(v1c2, 0)
    assert out.index(MOTION_PHRASES[0].strip("; ")) < out.index("vertical 9:16")
    assert out.endswith("vertical 9:16\ncomposition.")
