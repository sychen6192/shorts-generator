from shortsloop.policy import append_motion_phrase, failure_classes, MOTION_PHRASES


def test_class_priority_ordering():
    classes = failure_classes(["sharpness", "motion", "spec"], ["prompt_adherence"])
    assert classes == ["broken", "static", "off_prompt", "blurry"]


def test_l2_dims_map_to_classes():
    assert failure_classes([], ["anatomy_artifacts"]) == ["deformed"]
    assert failure_classes([], ["temporal_coherence"]) == ["flicker"]
    assert failure_classes([], ["imaging_quality"]) == ["blurry"]


def test_dedupe():
    assert failure_classes(["motion", "freeze"], []) == ["static"]


def test_motion_phrase_inserted_before_composition_tail():
    prompt = "A cat sits on a table, cinematic lighting, vertical 9:16 composition."
    out = append_motion_phrase(prompt, 0)
    assert MOTION_PHRASES[0] in out
    assert out.rstrip(".").endswith("vertical 9:16 composition")
    assert out.index(MOTION_PHRASES[0]) < out.index("vertical 9:16")


def test_motion_phrase_appended_when_no_tail():
    prompt = "A cat sits on a table."
    out = append_motion_phrase(prompt, 4)
    assert out.startswith(prompt)
    assert MOTION_PHRASES[4 % len(MOTION_PHRASES)] in out


def test_motion_phrase_pick_is_deterministic():
    prompt = "A cat, vertical 9:16 composition"
    assert append_motion_phrase(prompt, 1) == append_motion_phrase(prompt, 1)
    assert append_motion_phrase(prompt, 0) != append_motion_phrase(prompt, 1)
