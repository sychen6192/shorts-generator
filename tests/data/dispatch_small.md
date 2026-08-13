# 2026-08-20 Shorts 批次派工單(2 支 × 測試)

> 單一事實來源:所有 prompt 自包含,直接餵 `--prompt`,不需組裝。

## 0. 共用參數

| 項目 | 值 |
|---|---|
| T2V workflow | test_workflow.json |
| 解析度 | 480x832(測試用 draft) |
| length / fps | 49 frames(3.0 s)/ 16 fps |
| sampler | lightx2v 4-step,cfg 固定 1.0,shift 5.0 |
| negative | template 內建官方 negative,不覆寫 |
| seed | 全部 `--seed -1`,RESULT 行寫入 `seeds.log` |
| 排程 | 全程 sequential,一次一個 job |

## 生產清單(manifest)

| ID | 模式 | image 依賴 | 輸出 |
|---|---|---|---|
| V1C1–V1C2 | T2V | — | `v1/c1..c2.mp4` |
| V2C1 | T2V | — | `v2/c1.mp4` |
| V2C2 | I2V | V2C1 last frame | `v2/c2.mp4` |

## Video #1 — 測試片 A(純 T2V)

視覺錨:on a dark slate table, soft cinematic key light, photorealistic

### V1C1 — beat one(T2V)
```
A giant red cube of kinetic sand sliding slowly across a dark slate table, fine
grains crumbling off its edges; slow dolly-in, soft cinematic key light,
photorealistic, vertical 9:16 composition.
```

### V1C2 — beat two(T2V)
```
A polished steel blade slicing straight down through a blue kinetic sand sphere on
a dark slate table, sand splitting cleanly in slow motion; locked-off macro shot
with a subtle push-in, soft cinematic key light, photorealistic, vertical 9:16
composition.
```

## Video #2 — 測試片 B(hybrid,chain 進 v2)

### V2C1 — opener(T2V)
```
A green glass pyramid glowing from within on a black reflective surface, caustic
light refractions drifting; slow orbit shot, dark studio backdrop, ultra-detailed,
photorealistic, vertical 9:16 composition.
```

### V2C2 — continuation(I2V,image = V2C1 last frame)
```
The green glass pyramid slowly splits open revealing a molten gold core, light
spilling out; continued slow orbit, dark studio backdrop, photorealistic, vertical
9:16 composition.
```

## Master Runbook

```bash
# 由 shortsloop runner 接手;本區塊僅供人工備援。
```

## 早上 QC 清單

1. 掃 report.md。

## 音檔需求

| 影片 | 音軌 |
|---|---|
| V1 | 沙沙 ASMR foley |

## 標題與上傳備忘

| 影片 | 標題 |
|---|---|
| V1 | 巨型動力沙立方 ✂️ |
