# 2026-10-05 Shorts 批次派工單(5 支 × 30 秒)

> 單一事實來源:所有 prompt 自包含,直接餵 `--prompt`,不需組裝。
> 執行端搭配 comfyui-ig-video skill;本單只定義「生什麼、怎麼接、怎麼驗」。

## 0. 共用參數

| 項目 | 值 |
|---|---|
| T2V workflow | wan22_t2v_14B_api.json |
| I2V workflow | wan22_i2v_14B_api.json(⚠️ 需為已驗證的 API export) |
| 解析度 | 720x1280(無人值守直上 final,跳過 draft) |
| length / fps | 81 frames(5.0 s)/ 16 fps |
| sampler | lightx2v 4-step,cfg 固定 1.0,shift 5.0 |
| negative | template 內建官方 negative,不覆寫 |
| seed | 全部 `--seed -1`,RESULT 行寫入 `seeds.log` |
| 排程 | 全程 sequential,一次一個 job |

無人值守策略:draft 是給人挑 keeper 用的,夜跑沒人看,直接 final;
壞 clip 早上同 prompt 換 seed 重 roll。

## 生產清單(manifest)

| ID | 模式 | image 依賴 | 輸出 |
|---|---|---|---|
| V1C1–V1C6 | T2V | — | `v1/c1..c6.mp4` |
| V2C1–V2C3 | T2V | — | `v2/c1..c3.mp4` |
| V2C4 | I2V | V2C3 last frame | `v2/c4.mp4` |
| V2C5 | I2V | V2C4 last frame | `v2/c5.mp4` |
| V2C6 | T2V | — | `v2/c6.mp4` |
| V3C1 | T2V | — | `v3/c1.mp4` |
| V3C2–V3C4 | I2V | 前一顆 last frame(chain 3 hops) | `v3/c2..c4.mp4` |
| V3C5–V3C6 | T2V | — | `v3/c5..c6.mp4` |
| V4C1 | T2V | — | `v4/intro.mp4` |
| V4C2–V4C6 | 重用 | V1C2, V3C6, V2C6, V1C4, V1C1 | `v4/rank5..rank1.mp4` |

## Video #1 — 動力沙切切樂(純 T2V,循環)

視覺錨(一字不改):on a dark slate table, soft cinematic key light from the upper left, shallow depth of field, photorealistic

### V1C1 — 西瓜(T2V)
```
A giant photorealistic watermelon sculpted entirely from fine-grain matte kinetic
sand, deep green rind with dark stripes, resting on a dark slate table. A large
polished steel blade enters from the right and slices slowly straight down through
the center; the sand splits cleanly, revealing a red sand interior. Locked-off macro
shot with a slow subtle dolly-in. Soft cinematic key light from the upper left,
shallow depth of field, photorealistic, vertical 9:16 composition.
```

### V1C2 — 城堡(T2V)
```
A table-sized sand castle of pale lilac kinetic sand on a dark slate table, a steel
blade slicing diagonally through its tallest tower, grains cascading in slow motion;
slow orbit shot around the castle, soft cinematic key light from the upper left,
shallow depth of field, photorealistic, vertical 9:16
composition.
```

### V1C3 — 石榴(T2V)
```
A giant pomegranate of crimson kinetic sand on a dark slate table, a blade pressing
down and splitting it open to reveal ruby sand seeds; top-down tilt, soft cinematic
key light from the upper left, shallow depth of field, photorealistic, vertical 9:16 composition.
```

### V1C4 — 地球儀(T2V)
```
A globe of blue and green kinetic sand on a dark slate table, a blade slicing along
the equator, the two halves parting slowly; slow dolly-in, soft cinematic key light
from the upper left, shallow depth of field, photorealistic, vertical 9:16 composition.
```

### V1C5 — 奇異果(T2V)
```
A giant kiwi of olive-green kinetic sand on a dark slate table, a blade cutting it in
half to reveal a bright green interior with black seed specks; lateral tracking shot,
soft cinematic key light from the upper left, shallow depth of field, photorealistic,
vertical 9:16 composition.
```

### V1C6 — 循環收尾 reverse reassembly(T2V)
```
Scattered fragments of green and red kinetic sand levitate from a dark slate table and
fuse back into an intact watermelon sculpture; slow dolly-out, soft cinematic key light
from the upper left, shallow depth of field, photorealistic, vertical 9:16 composition.
```

## Video #2 — 小橘貓主廚做鬆餅(hybrid,chain 2 hops)

角色錨(一字不改):a fluffy orange tabby kitten with big round amber eyes, wearing a tiny white chef hat and a miniature beige apron, in a warm miniature kitchen with copper pots and a small sunlit window

### V2C1 — 備料(T2V)
```
A fluffy orange tabby kitten with big round amber eyes, wearing a tiny white chef hat
and a miniature beige apron, in a warm miniature kitchen with copper pots and a small
sunlit window, cracks a tiny egg into a ceramic bowl with both paws; slow dolly-in,
warm morning light, cozy miniature diorama style, vertical 9:16 composition.
```

### V2C2 — 攪拌(T2V)
```
A fluffy orange tabby kitten with big round amber eyes, wearing a tiny white chef hat
and a miniature beige apron, in a warm miniature kitchen with copper pots and a small
sunlit window, whisks batter vigorously in a ceramic bowl; slow orbit shot, warm
morning light, cozy miniature diorama style, vertical 9:16 composition.
```

### V2C3 — 下鍋(T2V)
```
A fluffy orange tabby kitten with big round amber eyes, wearing a tiny white chef hat
and a miniature beige apron, in a warm miniature kitchen with copper pots and a small
sunlit window, pours batter into a tiny copper pan where it sizzles; slow push-in,
warm morning light, cozy miniature diorama style, vertical 9:16 composition.
```

### V2C4 — 翻面(I2V,image = V2C3 last frame)
```
The kitten flips the pancake high into the air with a tiny spatula and it lands back
in the pan; camera tilts up following the pancake, vertical 9:16 composition.
```

### V2C5 — 疊盤(I2V,image = V2C4 last frame)
```
The kitten slides the golden pancake onto a growing stack on a small plate; slow
dolly-in, vertical 9:16 composition.
```

### V2C6 — hero shot(T2V)
```
A fluffy orange tabby kitten with big round amber eyes, wearing a tiny white chef hat
and a miniature beige apron, in a warm miniature kitchen with copper pots and a small
sunlit window, proudly presents a tall stack of pancakes with dripping syrup; slow
orbit, warm morning light, cozy miniature diorama style, vertical 9:16 composition.
```

## Video #3 — 雨夜小狗(敘事弧,T2V → I2V chain 3 hops → T2V)

### V3C1 — 困境(T2V)
```
A small soaked white puppy with floppy ears shivering under a bus stop bench on a cold
rainy night, blue streetlight reflecting on wet asphalt; slow dolly-in, cinematic
moody lighting, photorealistic, vertical 9:16 composition.
```

### V3C2 — 路人(I2V,image = V3C1 last frame)
```
A person in a dark coat holding a black umbrella stops beside the bench, seen only
from behind; slow push-in, vertical 9:16 composition.
```

### V3C3 — 猶豫(I2V,image = V3C2 last frame)
```
The person crouches down slowly and tilts the umbrella over the puppy; static camera
with subtle drift, vertical 9:16 composition.
```

### V3C4 — 接觸(I2V,image = V3C3 last frame)
```
A gloved hand reaches out and the puppy sniffs it, tail beginning to wag; slow
dolly-in, vertical 9:16 composition.
```

### V3C5 — 回家路上(T2V)
```
A small white puppy with floppy ears wrapped in a grey scarf carried in the arms of a
person in a dark coat, seen from behind, walking down a rainy street under a black
umbrella; tracking shot from behind, warm shop lights, photorealistic, vertical 9:16 composition.
```

### V3C6 — 溫暖結局(T2V)
```
A small white puppy with floppy ears curled up asleep on a knitted blanket beside a
fireplace, warm lamplight; slow dolly-in, cozy cinematic lighting, photorealistic,
vertical 9:16 composition.
```

## Video #4 — Top 5 倒數(重用素材)

預設排名(早上可改):#5 V1C2 城堡 → #4 V3C6 → #3 V2C6 → #2 V1C4 → #1 V1C1。
排名數字一律 drawtext 疊字,不進 prompt。

### V4C1 — lineup intro(T2V)
```
Five glowing crystal and kinetic sand sculptures — a watermelon, a castle, a
pomegranate, a globe, and a kiwi — lined up in a row on a long dark slate table, a
polished steel blade hovering above them catching the light. Slow lateral dolly
tracking along the lineup from right to left. Dark studio backdrop, dramatic rim
lighting with soft caustic reflections, shallow depth of field, photorealistic,
vertical 9:16 composition.
```

## Master Runbook

```bash
export COMFY_HOST=127.0.0.1:8188
SKILL=~/.claude/skills/comfyui-ig-video
CC="python3 $SKILL/scripts/comfy_client.py"
$CC health || exit 1
```

規則:一次一個 job;OOM 先 `$CC free` 再縮參數重試;chain parent 重 roll 則
children 連動重跑。

## 早上 QC 清單

1. 掃 `seeds.log`,確認全部 RESULT `"ok"`。
2. 重點盯:V2C5 疊盤一致性、V3C4 最後一跳。

## 音檔需求

| 影片 | 音軌 |
|---|---|
| V1 | 沙沙切割 foley,刀落點同步 |
| V2 | 輕快 BGM + 煎鍋滋滋聲 |

## 標題與上傳備忘

| 影片 | 標題 |
|---|---|
| V1 | 巨型動力沙水果切切樂 ✂️ |

全部開 AI 內容標註。
