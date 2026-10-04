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


def _fake_encoder(tmp_path, body: str):
    script = tmp_path / "fake_ig_encode.sh"
    script.write_text("#!/usr/bin/env bash\nset -e\nOUT=''\n"
                      "while getopts 'o:a:s:f:q:' opt; do case $opt in o) OUT=$OPTARG;; "
                      "*) ;; esac; done\nshift $((OPTIND-1))\n" + body)
    return script


def test_encoded_audio_is_aac_and_actually_silent(clips, tmp_path):
    import subprocess
    out = tmp_path / "V1C1.mp4"
    encode_silent(clips["moving"], out)
    codec = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a",
                            "-show_entries", "stream=codec_name", "-of", "csv=p=0",
                            str(out)], capture_output=True, text=True).stdout.strip()
    assert codec == "aac"
    vol = subprocess.run(["ffmpeg", "-nostats", "-i", str(out), "-map", "0:a:0",
                          "-af", "volumedetect", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    assert "max_volume: -inf dB" in vol or "max_volume: -9" in vol


def test_nonconforming_encode_leaves_nothing_in_the_shipping_folder(clips, tmp_path,
                                                                    monkeypatch):
    import shortsloop.encode as enc
    monkeypatch.setattr(enc, "VENDOR_ENCODE",
                        _fake_encoder(tmp_path, 'cp "$1" "$OUT"\n'))   # wrong spec
    out = tmp_path / "encoded" / "V1C1.mp4"
    with pytest.raises(EncodeFailed):
        encode_silent(clips["moving"], out)
    assert list((tmp_path / "encoded").iterdir()) == []


def test_audible_track_is_rejected(clips, tmp_path, monkeypatch):
    import shortsloop.encode as enc
    monkeypatch.setattr(enc, "VENDOR_ENCODE", _fake_encoder(tmp_path, (
        'ffmpeg -v error -y -f lavfi -i color=c=black:s=1080x1920:r=30:d=1 '
        '-f lavfi -i sine=frequency=440:duration=1 -c:v libx264 -pix_fmt yuv420p '
        '-c:a aac -shortest "$OUT"\n')))
    out = tmp_path / "encoded" / "V1C1.mp4"
    with pytest.raises(EncodeFailed, match="silent"):
        encode_silent(clips["moving"], out)
    assert not out.exists()
