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
