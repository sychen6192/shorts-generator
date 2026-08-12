"""L2 judge layer tests (plan §5 rows 1-3 checker side + rubric semantics)."""

from __future__ import annotations

import json

import pytest

from fake_judge import GOOD_SCORES, serve

from shortsloop.errors import ClipError, InfraError
from shortsloop.l2 import DIMENSIONS, run_l2, sample_frames

FLOORS = {d: 3 for d in DIMENSIONS}


def cfg(base_url, **over):
    base = {"adapter": "ollama", "base_url": base_url,
            "model": "qwen3-vl:8b-instruct", "timeout_s": 30}
    base.update(over)
    return base


def test_frame_sampling_uniform_first_last(clips):
    jpegs, meta = sample_frames(clips["moving"], fps=16.0)
    assert len(jpegs) == 8 and len(meta) == 8
    assert meta[0]["index"] == 0
    assert meta[-1]["index"] == 47            # 48-frame fixture: last frame included
    assert all(jpegs[i][:2] == b"\xff\xd8" for i in range(8))  # JPEG magic
    assert meta[3]["t"] == pytest.approx(meta[3]["index"] / 16.0, abs=0.01)


def test_happy_path_pass(clips):
    with serve() as (srv, url):
        block, raw = run_l2(clips["moving"], {"fps": 16.0}, "a prompt", cfg(url), FLOORS)
    assert block["pass"] is True
    assert set(block["dimensions"]) == set(DIMENSIONS)
    assert all(d["pass"] and d["score"] == 5 for d in block["dimensions"].values())
    assert block["model_digest"] == "sha256:fakedigest"
    assert len(block["frames"]) == 8
    assert json.loads(raw) == GOOD_SCORES
    # one call, with 8 images, schema-constrained, deterministic options
    assert srv.chat_calls == 1
    payload = srv.last_chat_payload
    assert len(payload["messages"][1]["images"]) == 8
    assert payload["format"]["required"] == list(DIMENSIONS)
    assert payload["options"]["temperature"] == 0.0
    assert payload["stream"] is False


def test_floor_applied_by_checker_not_judge(clips):
    scores = {k: dict(v) for k, v in GOOD_SCORES.items()}
    scores["anatomy_artifacts"] = {"score": 2, "na": False, "reason": "six fingers, frame 3"}
    with serve(scores=scores) as (_, url):
        block, _ = run_l2(clips["moving"], {"fps": 16.0}, "p", cfg(url), FLOORS)
    assert block["pass"] is False
    dim = block["dimensions"]["anatomy_artifacts"]
    assert dim["pass"] is False and dim["floor"] == 3 and dim["score"] == 2


def test_na_honored_for_subject_consistency(clips):
    scores = {k: dict(v) for k, v in GOOD_SCORES.items()}
    scores["subject_consistency"] = {"score": 1, "na": True, "reason": "no persistent subject"}
    with serve(scores=scores) as (_, url):
        block, _ = run_l2(clips["moving"], {"fps": 16.0}, "p", cfg(url), FLOORS)
    assert block["dimensions"]["subject_consistency"]["na"] is True
    assert block["pass"] is True                # na dim excluded from the decision


def test_na_coerced_false_where_not_allowed(clips):
    with serve(scenario="na_abuse") as (_, url):
        block, _ = run_l2(clips["moving"], {"fps": 16.0}, "p", cfg(url), FLOORS)
    assert block["dimensions"]["anatomy_artifacts"]["na"] is False
    assert block["pass"] is True                # score 5 still >= floor


def test_garbage_content_retried_then_clip_error(clips):
    with serve(scenario="garbage") as (srv, url):
        with pytest.raises(ClipError) as exc:
            run_l2(clips["moving"], {"fps": 16.0}, "p", cfg(url), FLOORS)
        assert srv.chat_calls == 2              # exactly one retry
    assert exc.value.scope == "clip"


def test_missing_dimension_is_invalid(clips):
    with serve(scenario="missing_dim") as (srv, url):
        with pytest.raises(ClipError):
            run_l2(clips["moving"], {"fps": 16.0}, "p", cfg(url), FLOORS)
        assert srv.chat_calls == 2


def test_timeout_retried_then_clip_error(clips):
    with serve(sleep_s=3.0) as (srv, url):
        with pytest.raises(ClipError) as exc:
            run_l2(clips["moving"], {"fps": 16.0}, "p", cfg(url, timeout_s=1), FLOORS)
        assert srv.chat_calls == 2
    assert "timed out" in str(exc.value)


def test_unreachable_judge_is_infra_error(clips):
    with pytest.raises(InfraError):
        run_l2(clips["moving"], {"fps": 16.0}, "p",
               cfg("http://127.0.0.1:9"), FLOORS)   # port 9: nothing listens


def test_model_not_installed_is_infra_error(clips):
    with serve(model="some-other-model") as (_, url):
        with pytest.raises(InfraError) as exc:
            run_l2(clips["moving"], {"fps": 16.0}, "p", cfg(url), FLOORS)
    assert "not installed" in str(exc.value)


def test_missing_floor_is_infra_error(clips):
    floors = {d: 3 for d in DIMENSIONS if d != "imaging_quality"}
    with serve() as (_, url):
        with pytest.raises(InfraError):
            run_l2(clips["moving"], {"fps": 16.0}, "p", cfg(url), floors)
