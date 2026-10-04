"""L2 judge layer tests (plan §5 rows 1-3 checker side + rubric semantics)."""

from __future__ import annotations

import json

import pytest

import base64

import cv2

from fake_judge import GOOD_SCORES, serve
from fake_openai import LLAMACPP_MODEL_ID, llamacpp_models_entry, serve_openai

from shortsloop import l2 as l2mod
from shortsloop.errors import ClipError, InfraError
from shortsloop.judge.ollama import OllamaAdapter
from shortsloop.judge.openai_compat import OpenAICompatAdapter
from shortsloop.l2 import (DIMENSIONS, JUDGE_RESPONSE_SCHEMA, N_FRAMES, run_l2,
                           sample_frames)
from shortsloop.verdict import sha256_text

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


@pytest.mark.parametrize("dim, bad", [
    ("subject_consistency", {"score": 1, "na": "false", "reason": "identity drifts"}),
    ("subject_consistency", {"score": 1, "na": 1, "reason": "identity drifts"}),
    ("anatomy_artifacts", {"score": True, "na": False, "reason": "bool is not a score"}),
    ("imaging_quality", {"score": "4", "na": False, "reason": "string score"}),
    ("imaging_quality", {"score": 4.5, "na": False, "reason": "fractional score"}),
])
def test_type_confused_judge_output_never_passes(clips, dim, bad):
    """Hard rule 2: schema-invalid judge content is retried then clip ERROR — a
    truthy string `na` must not silently exclude a failing dimension."""
    scores = {k: dict(v) for k, v in GOOD_SCORES.items()}
    scores[dim] = bad
    with serve(scores=scores) as (srv, url):
        with pytest.raises(ClipError):
            run_l2(clips["moving"], {"fps": 16.0}, "p", cfg(url), FLOORS)
        assert srv.chat_calls == 2                      # one retry, then give up


def test_absent_or_null_na_means_scored(clips):
    scores = {k: dict(v) for k, v in GOOD_SCORES.items()}
    scores["subject_consistency"] = {"score": 1, "reason": "drifts"}        # no na
    scores["anatomy_artifacts"] = {"score": 5, "na": None, "reason": "ok"}
    with serve(scores=scores) as (_, url):
        block, _ = run_l2(clips["moving"], {"fps": 16.0}, "p", cfg(url), FLOORS)
    assert block["dimensions"]["subject_consistency"]["na"] is False
    assert block["pass"] is False



# --- Ollama model-name matching ------------------------------------------------------

def test_ollama_bare_model_name_matches_latest_tag(clips):
    """`model: qwen2.5vl` in config is the same model ollama lists as
    `qwen2.5vl:latest` — must not be reported "not installed"."""
    with serve(model="qwen2.5vl:latest") as (_, url):
        block, _ = run_l2(clips["moving"], {"fps": 16.0}, "p",
                          cfg(url, model="qwen2.5vl"), FLOORS)
    assert block["model_digest"] == "sha256:fakedigest"
    assert block["pass"] is True


def test_ollama_tag_matching_is_otherwise_exact():
    with serve(model="qwen2.5vl:latest") as (_, url):
        assert OllamaAdapter(cfg(url, model="qwen2.5vl:latest")).model_digest() \
            == "sha256:fakedigest"
    with serve(model="qwen2.5vl") as (_, url):           # bare listing, tagged config
        assert OllamaAdapter(cfg(url, model="qwen2.5vl:latest")).model_digest() \
            == "sha256:fakedigest"
    with serve(model="qwen2.5vl:7b") as (_, url):        # a different tag is not :latest
        with pytest.raises(InfraError, match="not installed"):
            OllamaAdapter(cfg(url, model="qwen2.5vl")).model_digest()


# --- tolerant parsing of judge replies (fences / <think>) -------------------------

GOOD_JSON = json.dumps(GOOD_SCORES)


@pytest.mark.parametrize("reply", [
    f"```json\n{GOOD_JSON}\n```",
    f"```\n{GOOD_JSON}\n```\n",
    f"<think>\nframes 1-8 look {{consistent}}; nothing melts\n</think>\n\n{GOOD_JSON}",
    f"<think>ok</think>\n```json\n{GOOD_JSON}\n```",
    f"Here is my evaluation:\n{GOOD_JSON}\nThanks.",
    f"frames {{1-8}} reviewed; no melting\n</think>\n\n{GOOD_JSON}",
], ids=["json-fence", "bare-fence", "think-prefix", "think-and-fence", "prose",
        "template-opened-think"])
def test_wrapped_json_reply_is_parsed_without_retry(clips, reply):
    with serve() as (srv, url):
        srv.content = lambda: reply
        block, raw = run_l2(clips["moving"], {"fps": 16.0}, "p", cfg(url), FLOORS)
        assert srv.chat_calls == 1                      # no retry burned
    assert block["pass"] is True
    assert raw == reply                                 # audit copy is verbatim
    assert block["raw_response_sha256"] == sha256_text(reply)


@pytest.mark.parametrize("reply", [
    # strict typing still applies after unwrapping
    "```json\n" + json.dumps({**GOOD_SCORES, "subject_consistency":
                              {"score": 1, "na": "false", "reason": "drifts"}}) + "\n```",
    # two objects: ambiguous, never cherry-pick one
    f"draft: {GOOD_JSON}\nfinal: {GOOD_JSON}",
    "<think>unterminated reasoning {",
    "```json\n```",
], ids=["fenced-type-confused", "two-objects", "unterminated-think", "empty-fence"])
def test_unwrapped_reply_still_validated_strictly(clips, reply):
    with serve() as (srv, url):
        srv.content = lambda: reply
        with pytest.raises(ClipError) as exc:
            run_l2(clips["moving"], {"fps": 16.0}, "p", cfg(url), FLOORS)
        assert srv.chat_calls == 2
    assert exc.value.raw == reply


def test_clip_error_keeps_last_raw_reply(clips):
    with serve(scenario="garbage") as (srv, url):
        with pytest.raises(ClipError) as exc:
            run_l2(clips["moving"], {"fps": 16.0}, "p", cfg(url), FLOORS)
    assert exc.value.raw == "the clip looks fine to me, PASS!"


def test_timeout_only_clip_error_has_no_raw(clips):
    with serve(sleep_s=3.0) as (srv, url):
        with pytest.raises(ClipError) as exc:
            run_l2(clips["moving"], {"fps": 16.0}, "p", cfg(url, timeout_s=1, retries=0),
                   FLOORS)
    assert exc.value.raw is None


# --- judge retries from config ---------------------------------------------------

@pytest.mark.parametrize("retries, calls", [(0, 1), (1, 2), (2, 3)])
def test_judge_retries_from_config(clips, retries, calls):
    with serve(scenario="garbage") as (srv, url):
        with pytest.raises(ClipError):
            run_l2(clips["moving"], {"fps": 16.0}, "p", cfg(url, retries=retries), FLOORS)
        assert srv.chat_calls == calls


def test_extra_retry_can_recover(clips):
    replies = ["nope", "still nope", GOOD_JSON]
    with serve() as (srv, url):
        srv.content = lambda: replies.pop(0)
        block, _ = run_l2(clips["moving"], {"fps": 16.0}, "p", cfg(url, retries=2), FLOORS)
        assert srv.chat_calls == 3
    assert block["pass"] is True


@pytest.mark.parametrize("bad", [-1, "1", 1.5, 1.0, True, None])
def test_invalid_retries_is_infra_error(clips, bad):
    with serve() as (srv, url):
        with pytest.raises(InfraError, match="retries"):
            run_l2(clips["moving"], {"fps": 16.0}, "p", cfg(url, retries=bad), FLOORS)
        assert srv.chat_calls == 0


# --- frame sampling: two-pass, bounded memory --------------------------------------

def _sample_frames_single_pass(clip_path, fps):
    """The pre-two-pass reference implementation (kept verbatim for equivalence)."""
    cap = cv2.VideoCapture(str(clip_path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    n = min(N_FRAMES, len(frames))
    idxs = sorted({round(i * (len(frames) - 1) / (n - 1)) for i in range(n)})
    fps_val = fps if fps and fps > 0 else 16.0
    jpegs, meta = [], []
    for idx in idxs:
        ok, buf = cv2.imencode(".jpg", frames[idx],
                               [int(cv2.IMWRITE_JPEG_QUALITY), l2mod.JPEG_QUALITY])
        jpegs.append(buf.tobytes())
        meta.append({"index": int(idx), "t": round(idx / fps_val, 3)})
    return jpegs, meta


def test_two_pass_sampling_matches_single_pass(clips):
    new_j, new_m = sample_frames(clips["moving"], fps=16.0)
    old_j, old_m = _sample_frames_single_pass(clips["moving"], 16.0)
    assert new_m == old_m
    assert new_j == old_j                       # byte-identical JPEGs


def test_sampling_materializes_only_selected_frames(clips, monkeypatch):
    real = cv2.VideoCapture
    caps = []

    class Counting:
        def __init__(self, *a):
            self._cap = real(*a)
            self.materialized = 0
            caps.append(self)

        def read(self, *a):
            self.materialized += 1
            return self._cap.read(*a)

        def retrieve(self, *a):
            self.materialized += 1
            return self._cap.retrieve(*a)

        def __getattr__(self, name):
            return getattr(self._cap, name)

    monkeypatch.setattr(l2mod.cv2, "VideoCapture", Counting)
    jpegs, _ = sample_frames(clips["moving"], fps=16.0)
    assert len(jpegs) == N_FRAMES
    assert sum(c.materialized for c in caps) == N_FRAMES   # not all 48 frames


def test_sampling_rejects_corrupt(clips):
    with pytest.raises(ClipError):
        sample_frames(clips["corrupt"], fps=16.0)


# --- openai_compat adapter (llama.cpp / vLLM) ------------------------------------

def ocfg(base_url, **over):
    return cfg(base_url, adapter="openai_compat", model="qwen3-vl-8b", **over)


def test_openai_compat_happy_path(clips):
    with serve_openai() as (srv, url):
        block, raw = run_l2(clips["moving"], {"fps": 16.0}, "a prompt", ocfg(url), FLOORS)
        digest_again = OpenAICompatAdapter(ocfg(url)).model_digest()
    assert block["pass"] is True
    assert json.loads(raw) == GOOD_SCORES
    assert block["model"] == "openai_compat/qwen3-vl-8b"
    assert srv.chat_calls == 1
    payload = srv.chat_payloads[0]
    assert payload["temperature"] == 0.0 and payload["seed"] == 7
    rf = payload["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["schema"] == JUDGE_RESPONSE_SCHEMA
    system, user = payload["messages"]
    assert system["role"] == "system" and user["role"] == "user"
    parts = user["content"]
    assert parts[0]["type"] == "text" and "a prompt" in parts[0]["text"]
    images = [p["image_url"]["url"] for p in parts if p["type"] == "image_url"]
    assert len(images) == 8
    for uri in images:
        assert uri.startswith("data:image/jpeg;base64,")
        assert base64.b64decode(uri.split(",", 1)[1])[:2] == b"\xff\xd8"
    # digest names the served build (id + llama.cpp owned_by/meta), deterministically
    # even though the server's per-request `created` timestamp changes
    d = block["model_digest"]
    assert LLAMACPP_MODEL_ID in d and "llamacpp" in d and "n_params" in d
    assert "created" not in d
    assert digest_again == d


def test_openai_compat_multi_model_server_picks_configured_id(clips):
    models = [llamacpp_models_entry("other-model"),
              {"id": "qwen3-vl-8b", "object": "model", "owned_by": "vllm",
               "root": "Qwen/Qwen3-VL-8B-Instruct"}]
    with serve_openai(models=models) as (_, url):
        d = OpenAICompatAdapter(ocfg(url)).model_digest()
    assert d.startswith("openai-compat:qwen3-vl-8b")
    assert "Qwen/Qwen3-VL-8B-Instruct" in d and "other-model" not in d


def test_openai_compat_model_not_served_is_infra(clips):
    models = [llamacpp_models_entry("a"), llamacpp_models_entry("b")]
    with serve_openai(models=models) as (srv, url):
        with pytest.raises(InfraError, match="not served"):
            run_l2(clips["moving"], {"fps": 16.0}, "p", ocfg(url), FLOORS)
        assert srv.chat_calls == 0


@pytest.mark.parametrize("scenario", ["models_404", "chat_404"])
def test_openai_compat_http_404_is_infra(clips, scenario):
    with serve_openai(scenario=scenario) as (_, url):
        with pytest.raises(InfraError) as exc:
            run_l2(clips["moving"], {"fps": 16.0}, "p", ocfg(url), FLOORS)
    assert "404" in str(exc.value)


def test_openai_compat_unreachable_is_infra(clips):
    with pytest.raises(InfraError, match="unreachable"):
        run_l2(clips["moving"], {"fps": 16.0}, "p", ocfg("http://127.0.0.1:9"), FLOORS)


def test_openai_compat_empty_choices_retried_then_clip_error(clips):
    with serve_openai(scenario="empty_choices") as (srv, url):
        with pytest.raises(ClipError) as exc:
            run_l2(clips["moving"], {"fps": 16.0}, "p", ocfg(url), FLOORS)
        assert srv.chat_calls == 2
    assert exc.value.scope == "clip"
    assert "choices" in str(exc.value)


def test_openai_compat_wrapped_reply_parsed(clips):
    reply = f"<think>looking</think>\n```json\n{GOOD_JSON}\n```"
    with serve_openai(content_queue=[reply]) as (srv, url):
        block, raw = run_l2(clips["moving"], {"fps": 16.0}, "p", ocfg(url), FLOORS)
        assert srv.chat_calls == 1
    assert block["pass"] is True and raw == reply
