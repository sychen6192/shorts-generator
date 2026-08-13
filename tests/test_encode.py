"""Encode stage tests (plan §5 row 16): spec verified by ffprobe, never by eye."""

from __future__ import annotations

import pytest

from shortsloop.encode import EncodeFailed, encode_silent


def test_encode_produces_ig_spec_silent_mp4(clips, tmp_path):
    out = tmp_path / "enc" / "V1C1.mp4"
    container = encode_silent(clips["moving"], out)
    assert out.exists()
    assert (container["width"], container["height"]) == (1080, 1920)
    assert container["vcodec"] == "h264"
    assert abs(container["fps"] - 30.0) <= 0.5
    # source is 3s@16fps; encoded duration must be in the same ballpark
    assert 2.0 <= container["duration_s"] <= 4.5


def test_encode_rejects_corrupt_input(clips, tmp_path):
    with pytest.raises(EncodeFailed):
        encode_silent(clips["corrupt"], tmp_path / "bad.mp4")
