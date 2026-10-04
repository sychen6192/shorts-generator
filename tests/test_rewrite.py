"""off_prompt rewrite (plan §2.3): the contract is enforced in code, not only asked
of the LLM — anchors verbatim, ≤110 words, composition tail kept."""

from __future__ import annotations

import json
from pathlib import Path

from fake_judge import GOOD_SCORES, serve as serve_judge
from fake_comfy import serve_comfy
from test_runner import make_env, make_runner, small_sheet

from shortsloop.dispatch import parse_dispatch
from shortsloop.rewrite import rewrite_prompt, validate_rewrite

DATA = Path(__file__).parent / "data"
ANCHOR = ("on a dark slate table, soft cinematic key light from the upper left, "
          "shallow depth of field, photorealistic")
ORIGINAL = ("A giant watermelon of kinetic sand resting on a dark slate table, a blade "
            "slicing down; slow dolly-in, soft cinematic key light from the upper left, "
            "shallow depth of field, photorealistic, vertical 9:16 composition.")


def test_sheet_anchors_reach_their_clips():
    sheet = parse_dispatch(DATA / "dispatch_full.md")
    by_id = {c.clip_id: c for c in sheet.clips}
    assert by_id["V1C3"].anchors == [ANCHOR]
    assert by_id["V2C1"].anchors[0].startswith("a fluffy orange tabby kitten")
    assert by_id["V3C1"].anchors == []


def test_valid_rewrite_passes():
    good = ("A giant watermelon of kinetic sand resting on a dark slate table, a steel "
            "blade slicing straight down through its center; slow dolly-in, soft "
            "cinematic key light from the upper left, shallow depth of field, "
            "photorealistic, vertical 9:16 composition.")
    assert validate_rewrite(ORIGINAL, good, [ANCHOR]) == []


def test_rewrite_violations_are_caught():
    too_long = ORIGINAL.replace("a blade slicing down", " ".join(["very"] * 120))
    assert any("110" in p for p in validate_rewrite(ORIGINAL, too_long, [ANCHOR]))
    no_tail = ORIGINAL.replace(", vertical 9:16 composition.", ".")
    assert any("9:16" in p for p in validate_rewrite(ORIGINAL, no_tail, [ANCHOR]))
    no_anchor = ORIGINAL.replace("shallow depth of field, ", "")
    assert any("shallow depth of field" in p
               for p in validate_rewrite(ORIGINAL, no_anchor, [ANCHOR]))


def test_unreachable_rewriter_reports_why():
    got = rewrite_prompt(ORIGINAL, "no blade", {"base_url": "http://127.0.0.1:9",
                                                "model": "m"}, timeout_s=2)
    assert got["ok"] is False and got["reason"]


def test_runner_rejects_a_contract_breaking_rewrite_and_reseeds(clips, tmp_path):
    bad = {**{k: dict(v) for k, v in GOOD_SCORES.items()},
           "prompt_adherence": {"score": 2, "na": False, "reason": "no cube"}}
    long_rewrite = {"rewritten_prompt": " ".join(["cube"] * 130) +
                    ", vertical 9:16 composition"}
    with serve_comfy(fixture_paths=clips) as comfy, \
         serve_judge(scores_queue=[bad, GOOD_SCORES],
                     rewrite_response=long_rewrite) as (judge, jurl):
        runner = make_runner(small_sheet(tmp_path), make_env(tmp_path, comfy, jurl))
        assert runner.run() == 0
        assert judge.rewrite_calls == 1
        assert comfy.job_info(1)["prompt"] == comfy.job_info(0)["prompt"]
        a2 = [json.loads(l) for l in
              (runner.run_dir / "attempts.jsonl").read_text().splitlines()][1]
        assert a2["reroll_action"].startswith("rewrite_skipped_reseed")
        assert "110" in a2["reroll_action"] and a2["prompt_rewritten"] is False
