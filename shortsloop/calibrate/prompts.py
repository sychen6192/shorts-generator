"""Calibration prompt slate (docs/plan.md §6 / brainstorm §5).

A judge calibrated only on good clips is worthless, so the slate deliberately
spans the failure space:
- `good`     — format-recipe-style prompts with explicit action AND camera move.
- `starved`  — deliberately motion-starved: no action verb, no camera move
               (distill LoRAs mute motion; these should come out near-static).
- `anatomy`  — hands/limbs-heavy close-ups, the classic deformity stressors.

Off-prompt ground truth is manufactured AFTER generation by prompt-swapping
(batchgen pairs a good clip with a different prompt) — zero extra GPU cost.
"""

GOOD = [
    "A giant watermelon sculpted from fine-grain matte kinetic sand, deep green rind "
    "with dark stripes, on a dark slate table. A polished steel blade slices slowly "
    "straight down through the center, red sand interior crumbling off the cut faces "
    "in slow motion; locked-off macro shot with a slow dolly-in, dark studio backdrop, "
    "soft cinematic key light, photorealistic, vertical 9:16 composition.",

    "A translucent crystal apple glowing amber from within on a black reflective "
    "surface. A blade presses down and the apple splits cleanly, light spilling from "
    "the glowing core; slow orbit shot around the split halves, caustic refractions, "
    "dark studio backdrop, ultra-detailed, photorealistic, vertical 9:16 composition.",

    "A skyscraper-sized candle burning in the middle of a vast modern city of glass "
    "towers at dusk, wax rivers flowing slowly down its sides, flame flickering; "
    "sweeping aerial drone shot circling the flame, golden hour light, cinematic, "
    "photorealistic, vertical 9:16 composition.",

    "A silver mechanical hummingbird hovering over a neon-lit night market stall, "
    "wings beating rapidly, steam rising around it; slow dolly-in with shallow depth "
    "of field, cinematic lighting, film grain, vertical 9:16 composition.",

    "A mountain-sized ice cube melting in a desert canyon under harsh sun, meltwater "
    "cascading off its faces in sheets; slow aerial pullback revealing the scale, "
    "heat shimmer, photorealistic, vertical 9:16 composition.",

    "A stack of golden pancakes on a rustic wooden table; thick maple syrup pours "
    "from above in a glossy ribbon, spreading and dripping over the edges in slow "
    "motion; locked-off macro shot with a subtle push-in, warm morning window light, "
    "photorealistic, vertical 9:16 composition.",

    "An origami paper crane unfolding itself and refolding into a paper fox on a "
    "dark walnut desk, creases moving crisply; top-down camera slowly rotating, soft "
    "lamp light, paper texture detail, photorealistic, vertical 9:16 composition.",

    "A glassblower's molten orb of orange glass rotating on a steel rod, softly "
    "deforming and glowing, sparks drifting; handheld tracking shot slowly circling "
    "the orb, dark workshop, dramatic rim lighting, vertical 9:16 composition.",

    "Ocean waves crashing against a black basalt sea stack in a storm, spray "
    "exploding upward, gulls wheeling; slow motion telephoto shot panning with the "
    "waves, overcast dramatic light, photorealistic, vertical 9:16 composition.",

    "A tiny robot watering a sunflower on a sunlit windowsill, water arcing from a "
    "miniature can, the flower nodding; slow lateral dolly along the windowsill, warm "
    "afternoon light, shallow depth of field, photorealistic, vertical 9:16 composition.",

    "A red fox trotting through fresh snowfall between birch trees, snow kicking up "
    "from its paws, breath visible; steadicam tracking shot at fox height, soft "
    "overcast light, photorealistic, vertical 9:16 composition.",

    "A neon-blue jellyfish pulsing upward through dark water surrounded by drifting "
    "plankton sparks; slow vertical tilt following its rise, deep sea gloom, "
    "bioluminescent glow, photorealistic, vertical 9:16 composition.",

    "A ceramic teapot pouring steaming jasmine tea into a glass cup on a stone tray, "
    "steam curling in morning light; locked-off macro shot with a slow dolly-in, "
    "zen studio, soft key light, photorealistic, vertical 9:16 composition.",

    "A vinyl record spinning on a turntable, the needle riding the groove, dust "
    "motes drifting in a shaft of light; slow orbit around the tonearm, moody club "
    "lighting, shallow depth of field, photorealistic, vertical 9:16 composition.",
]

# No action verb, no camera language — these SHOULD come out near-static.
STARVED = [
    "A ceramic vase on a wooden table, studio lighting, photorealistic, vertical "
    "9:16 composition.",
    "A quiet mountain lake at dawn, mist, muted colors, photorealistic, vertical "
    "9:16 composition.",
    "A bowl of ramen on a counter, warm light, detailed, vertical 9:16 composition.",
    "An antique brass telescope by a window, dusty atmosphere, vertical 9:16 "
    "composition.",
    "A stone lighthouse on a calm evening, pastel sky, photorealistic, vertical "
    "9:16 composition.",
    "A leather armchair in a library, soft lamp light, vertical 9:16 composition.",
]

# Hands / limbs / interacting parts — deformity stressors.
ANATOMY = [
    "Close-up of two hands kneading dough on a floured wooden board, fingers "
    "pressing and folding rhythmically; locked-off macro shot with a slow push-in, "
    "warm bakery light, photorealistic, vertical 9:16 composition.",
    "A pair of hands shuffling and fanning a deck of playing cards above a green "
    "felt table, cards arcing between fingers; macro shot, slow dolly-in, dramatic "
    "spotlight, photorealistic, vertical 9:16 composition.",
    "Faceless pianist's hands playing fast arpeggios on a grand piano, fingers "
    "crossing over; side-on tracking shot gliding along the keys, concert hall "
    "light, photorealistic, vertical 9:16 composition.",
    "Two people's hands exchanging and wrapping a small gift box with ribbon, "
    "fingers tying a bow; overhead locked-off shot with a slow rotate, cozy warm "
    "light, photorealistic, vertical 9:16 composition.",
]


def build_slate(count: int, seeds_per_prompt: int = 2) -> list[dict]:
    """Deterministic slate of `count` generation rows. Mix ratio ≈ plan §6:
    good : starved : anatomy ≈ 14 : 6 : 4 per 24 prompts, cycled with distinct
    seed slots until `count` is reached."""
    ordered = ([("good", p) for p in GOOD]
               + [("starved", p) for p in STARVED]
               + [("anatomy", p) for p in ANATOMY])
    rows = []
    slot = 0
    while len(rows) < count:
        kind, prompt = ordered[slot % len(ordered)]
        rows.append({
            "clip_id": f"cal{len(rows) + 1:03d}",
            "kind": kind,
            "prompt": prompt,
            "seed_slot": slot // len(ordered),   # 0 = first seed, 1 = second …
        })
        slot += 1
    return rows
