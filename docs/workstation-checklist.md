# shortsloop 工作站（`llm`）操作清單：從 clone 到每晚 cron 自動跑

依據是 repo 的程式碼，加上在雲端 CPU 上實際跑過的指令。英文的詳細說明在 docs/runbook.md（已和程式碼對齊）；docs/plan.md 有一處和這份不同，列在文末。凡是碰到 GPU 的行為，在你於工作站跑過之前都算 **UNVERIFIED-ON-GPU**。

**共通規則**
- 所有指令都**在 repo 根目錄**下執行，一律用 `.venv/bin/shortsloop`。`config.yaml`、`pipeline.yaml`、`thresholds.yaml`、`calibration/` 這些預設路徑都是相對於目前所在目錄。
- 工具印出來的提示（resume 指令、`Next:`）開頭是裸的 `shortsloop`，要自己在前面補上 `.venv/bin/`。
- 範例路徑（請換成你自己的）：repo `/home/<you>/shorts-generator`、`paths.runs_dir` `/data/shortsloop/runs`、派工單 `/data/shortsloop/dispatch/`、備份 `/data/shortsloop/backup/`、workflow `/home/<you>/workflows/wan22_t2v_api.json`。
- 4a、4c、第 5 步這類長指令請在 tmux 或 screen 裡跑：SSH 斷線會直接殺掉程序，不會留下報告（見第 7 步「安全停止」）。

## 總覽

| # | 步驟 | 大約時間 | 閘門 |
|---|---|---|---|
| 0 | 事前準備（驅動、ComfyUI、模型、Ollama、ffmpeg、磁碟） | 看要下載多少模型 | |
| 1 | clone、建 venv、跑 pytest（453 個測試） | 10–20 分鐘（工作站上還沒量過） | |
| 2 | 設定 `config.yaml` 和 `pipeline.yaml` | 10 分鐘 | |
| 3 | `doctor`（可能要跑兩次） | 每次幾分鐘（還沒量過） | doctor FAIL `judge.unload` 時，由你決定 `gen_free_min_gb` |
| 3b | （選做）彩排：做第 5 步的 (a)(b)(c)，做完再回 4a | 同第 5 步 | 看著跑 |
| 4a | `calibrate-batch` 產生 40 支草稿 | GPU 約 30–40 分鐘（runbook 估計） | |
| 4b | `label` 人工標註 44 列 | 10–15 分鐘 | 🛑 **人工閘門 1** |
| 4c | `calibrate-tune --with-l2`，再讀報告 | 44 次 judge 判讀（還沒量過）＋讀報告約 15 分鐘 | 由你決定要不要換成較大的 judge |
| 4d | `calibrate-tune --approve` | 1 分鐘 | 🛑 **人工閘門 2**（簽了才能無人值守） |
| 5 | 第一次在旁邊看著跑（(d)） | 6 支全過約 25–35 分鐘，最壞約 1.5–2 小時（docs/brainstorm.md 估計） | 看著跑（程式不強制） |
| 6 | 設定 cron | 10 分鐘 | |
| 7 | 每天早上：看報告、審片、配音、上傳、放明晚的派工單 | — | 你審 |
| 8 / 9 | 出錯對照表／第一晚要親眼確認的項目 | — | |

## 什麼時候要重跑什麼

| 你改了什麼或發生了什麼 | 必須重跑 | 由誰強制 |
|---|---|---|
| `config.yaml` 的任何一個 byte（連註解也算） | 3 doctor | run 會比對 fingerprint 的 `config_sha256` → 不符就 exit 2 |
| `pipeline.yaml` 的任何一個 byte（例如加了 `gen_free_min_gb`、改了 `judge.timeout_s`） | 3 doctor（在**最後一次**修改之後） | `pipeline_sha256` |
| 重新匯出或修改 workflow JSON | 3 doctor，**再看著跑一次第 5 步**（doctor 不會真的送 job，缺 node 要到送出時才會被拒）。換了模型、步數或 sampler 建議重做 Phase 0：先 `mv calibration /data/shortsloop/backup/calibration.old-YYYYMMDD`（搬到 repo 外：repo 裡的 `calibration.old-*` 沒被 gitignore），再 4a–4d（同一個 `calibration/` 重下 4a 不會重新生成任何片） | doctor 靠 `workflow_sha256`；其餘不強制 |
| 同檔名換掉 Wan 或 LoRA 權重檔 | 3 doctor；建議重做 Phase 0（同上一列：先把 `calibration/` 搬到 repo 外，再 4a–4d） | 不強制：fingerprint 不含模型檔內容，doctor 只查檔名在不在 |
| `comfy.host`、`judge.adapter` 或 `judge.model` | 3 doctor；換了 judge 還要做 4c `--with-l2` 和 4d | run 和 `--approve` 都會比對 judge 名稱 |
| `ollama pull` 讓 judge 的 digest 變了 | 拉回原本那個 build，或重做 4c `--with-l2` 和 4d | 開跑時 digest 不符 → exit 2；夜裡中途變了 → halt |
| `git pull` 動到 `shortsloop/` 的任何檔或 `pipeline.yaml` | 1 pytest，再 3 doctor（不帶 `--out`），01:58 前必須 `ALL CHECKS PASSED`；下面三列是額外要做的 | fingerprint 整組比對：新版改了 fingerprint 的 key，舊 doctor.json 就拒跑；`pipeline.yaml` 的驗證變嚴也會拒跑；新加的 doctor 檢查只有重跑才生效 |
| `git pull` 動到 `shortsloop/l1.py` | 1 pytest、4c（`calibrate-tune`）和 4d | `l1_impl` 不符 → run 拒跑 |
| `git pull` 動到 `shortsloop/l2.py` 或 `shortsloop/judge/` | 1 pytest；`mv calibration/l2_scores.jsonl calibration/l2_scores.old.jsonl`，再 4c `--with-l2` 和 4d | 沒有程式檢查 |
| 4c 之後又改了 `labels.jsonl` | 4c 和 4d | `--approve` 會拒絕 |
| 執行 `calibrate-tune --approve` | **不需要**重跑 doctor | thresholds.yaml 不在 fingerprint 裡 |
| judge 的 `temperature`／`seed`／`num_ctx`，或 `policies.judge.timeout_s`／`retries` | 3 doctor（強制）；建議重新打分：先 `mv calibration/l2_scores.jsonl calibration/l2_scores.old.jsonl`，再 4c `--with-l2` 和 4d（不搬走的話 4c 會沿用舊分數，連 judge 錯誤列也不會重評） | 分數只用 clip、prompt、model、digest 當 key，沒有程式檢查 |
| `pip install -U`（opencv 或 numpy 換版本） | 1 pytest；要重新量 L1：先 `mv calibration/metrics.jsonl calibration/metrics.old.jsonl`，再 4c 和 4d（不強制） | L1 快取只看 clip sha 和 `l1.py`；版本只會記錄下來，不會擋 |
| 升級 ComfyUI、Ollama 或驅動，ffmpeg 搬了位置 | 3 doctor（靠自律，程式不檢查） | doctor.json 不會過期 |
| 重新 clone（`doctor.json` 不在 git 也不在備份裡） | 依第 1 步「重新 clone」還原，再 1 pytest、3 doctor | 沒有 doctor.json → run 拒跑 |
| 任何**沒有**以 `ALL CHECKS PASSED` 結束的 doctor（FAIL、Ctrl-C、crash、`--no-free`、撞鎖的 `FAIL gpu.lock`、孤兒 job 造成的 `FAIL vram.handoff`） | 修好後再跑一次 3 doctor；01:58 前最後一次不帶 `--out` 的 doctor 必須全綠 | doctor 一開始就把 doctor.json 寫成 ok:false → 下一晚拒跑 |
| 換了當天的派工單 | 不需要重跑；用第 5 步 (b) 檢查 | 派工單不在 fingerprint 裡 |

## 需要你決定的事

1. **`flicker_dips` 的定義**：影響 4b（標註鍵 `4 flicker`）、4c 調參，以及之後每晚的 L1 判定。
   - 目前的定義是兩種計數相加（`shortsloop/l1.py:106-131`，plan §2.2 在 2026-10-04 修訂）：
     - v1 的 SSIM dip：某一對畫格的 SSIM 比 7 對畫格滾動中位數低 ≥ 0.03。
     - luma 反轉：平均亮度的變化方向反轉，而且前後兩側 |Δluma| 都 ≥ 0.02。
   - 平均亮度不變的閃爍（例如紋理沸騰）L1 抓不到；L2 的 rubric 也明確要求不評逐格閃爍（`shortsloop/l2.py:84-85`，而且只取 8 格）。實際上只有早上人工審片能發現。
   - **請在 4c 之前決定。** 4a 不算 L1，所以改定義只需重做 4c 和 4d（`l1_impl` 會變）。改法是在開發環境先改 plan §2.2，再 test-first 改 `l1.py`，然後 pull 到工作站；不要在工作站上直接改。
2. **`FF_CAP_FRAC` = 20%**（`shortsloop/calibrate/tune.py:79`）：影響 4c 怎麼讀報告、4d 要不要簽。
   - 每個門檻最多只能誤殺 tune 集裡 20% 被標成 pass 的片。
   - L2 floors 如果找不到任何組合能符合這個上限，報告會出現 `L2 FLOORS ARE A FALLBACK`。
   - 要決定兩件事：20% 能不能接受？出現 FALLBACK 時，是照樣簽、重新標註，還是換大一點的 judge？改這個數值也是改程式碼（同第 1 點的流程）。
3. **`--accept-untested-l2`**：影響 4d 和 6。
   - 用這個旗標簽核的 thresholds，runner 會當成「未校準」處理：沒加 `--allow-uncalibrated` 就拒跑，PASS 的片只會進 `encoded_uncalibrated/`，**cron 不能用**。
   - 建議不要用。如果用，只限於有人在旁邊看著跑的情況。
4. **`policies.vram_handoff.gen_free_min_gb`**（選填，預設等於 `free_min_gb` 24）：影響第 3 步和之後的每一晚。
   - 32 GB 的卡上，8B judge 還常駐時空閒 VRAM 可能超過 24 GiB（取決於 judge 實際常駐多少，UNVERIFIED-ON-GPU）。超過時 doctor 會 FAIL `judge.unload` 並建議一個中間值；doctor 的 `judge.unload` 是 OK 的話就不用設。
   - 要決定是否採用 doctor 建議的值。`pipeline.yaml` 是 git 追蹤的檔案，改了要備份（見 4d），並依第 1 步的規則保留為本機改動。
   - 換成 32B judge 時要重新決定一次。

---

## 0. 事前準備

**目的：** 確認工作站已經具備 shortsloop 不會自己安裝或啟動的東西。

**指令（只檢查，不改任何東西）：**
```bash
python3 --version                                   # 需要 >= 3.10
python3 -c 'import ensurepip' && echo venv-ok       # 失敗 → Ubuntu 要裝 python3-venv（或 python3.X-venv）
for b in ffmpeg ffprobe bash; do command -v "$b" || echo "MISSING $b"; done
ffmpeg -hide_banner -encoders | grep -E ' (libx264|aac) '   # libx264 和 aac 兩行都要出現
curl -s http://127.0.0.1:8188/system_stats          # devices[0] 要是 5090，type 不能是 cpu
ollama list                                         # 要看得到 judge（和 rewrite）模型
ollama ps                                           # 夜間應該是空的
df -h /data                                         # runs_dir 所在的檔案系統
```

**通過條件：**
- [ ] Python ≥ 3.10。雲端上 3.10 和 3.11 都跑過 pytest，全部通過。
- [ ] `ffmpeg` 和 `ffprobe` 都在 PATH 上，且有 libx264 和 aac 編碼器。pytest 的 fixture 和夜間的 QC encode 都要用到。
- [ ] ComfyUI **以常駐服務的方式**在跑（cron 不會幫你啟動）。`comfy.host` 預設是 `127.0.0.1:8188`。ComfyUI 沒有驗證機制，所以只綁本機或 LAN／Tailscale。
- [ ] `/system_stats` 的 **devices[0]** 是 5090 而且是 CUDA，因為 shortsloop 只看第一張卡。多卡的話用 `CUDA_VISIBLE_DEVICES` 讓 ComfyUI 把 5090 排在第一張。
- [ ] Wan 2.2 的模型檔已經放在伺服器上。用 skill 範本 `wan22_14b_t2v_lightx2v.json` 的話共 **6 個檔，分在 4 個資料夾**：
  - `diffusion_models/`：`wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors`、`wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors`
  - `text_encoders/`：`umt5_xxl_fp8_e4m3fn_scaled.safetensors`
  - `vae/`：`wan_2.1_vae.safetensors`
  - `loras/`：`wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors`、`wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors`
- [ ] workflow 已經用 ComfyUI → Workflow → **Export (API)** 匯出成 JSON（repo 裡沒有這個檔）。條件如下：
  - 只用核心 loader：`UNETLoader`、`CLIPLoader`、`VAELoader`、`LoraLoader(ModelOnly)`、`CheckpointLoaderSimple`、`CLIPVisionLoader`。GGUF loader 會被判 FAIL。
  - prompt、seed、width、height、length、**fps**、steps 都必須是常數輸入，不能是連線。
  - 每個 job **只能輸出一支**非 temp 的影片：用 `SaveVideo`，或 `VHS_VideoCombine` 加上 `save_output=true`。
  - 有 lightx2v LoRA 時 cfg 必須是 1.0。
  - workflow 裡的每個 node class 都已裝在這台 ComfyUI 上（`VHS_VideoCombine` 是 custom node；範本的 `CreateVideo`／`SaveVideo` 需要夠新的 ComfyUI）。doctor 不檢查這點，送出被拒會變成 `HALTED(infra)`。
- [ ] Ollama 以常駐服務在跑，並且已經 `ollama pull qwen3-vl:8b-instruct`（judge）和 `ollama pull qwen3:8b`（rewrite）。不用 rewrite 的話，第 2 步設 `rewrite.enabled: false`。這兩個 tag 是否真的存在，這邊沒辦法驗證。
- [ ] 改用 openai_compat 的話：llama-server（GGUF＋mmproj）要放在 llama-swap 後面，llama-swap 的 `/unload` 要能用 **GET** 把 VLM 從 VRAM 移出去。context 大小在伺服器端設，因為 adapter 不會送 `num_ctx`。
- [ ] 關掉 Ollama 的自動更新和自動 pull。judge 的 digest 一變，run 就會拒跑。
- [ ] 大磁碟已經掛載，而且 runs_dir 至少有 20 GiB 空間（`disk_min_free_gb`）。doctor、calibrate 和 run 都會 `mkdir -p` runs_dir；沒掛載的話會直接寫到根分割區。
- [ ] ComfyUI 自己的 output 資料夾要另外清理。shortsloop 不會刪伺服器端那份副本，20 GiB 的底線也只看 runs_dir。
- [ ] 用到的埠：ComfyUI 8188、Ollama 11434、標註 UI 8765（只綁 127.0.0.1）、llama-swap（例如 8080）。
- [ ] 夜間沒有其他人用 GPU：瀏覽器裡的 ComfyUI 不能排 job，也不能有其他 Ollama 前端在載入模型。

**失敗時：**
- 缺 ffmpeg 或 venv-ok 沒印出來 → `sudo apt install ffmpeg python3-venv`。
- `type: cpu` 或 devices 是空的 → 用 CUDA 重新啟動 ComfyUI。
- 模型缺了先不用找，第 3 步 doctor 的 `models.<folder>` 會列出缺哪個檔。

## 1. 安裝與測試

**目的：** 在工作站上取得同一份程式碼，並證明 CPU 能測的部分都是綠燈。

**指令：**
```bash
git clone https://github.com/sychen6192/shorts-generator && cd shorts-generator
# 已經 clone 過的話（在 repo 根目錄）：
#   git status --short                                # Phase 0 之後預期只有 " M pipeline.yaml"、" M thresholds.yaml"
#   git fetch && git diff --stat HEAD origin/main     # 先看會動到哪些檔，對照重跑表
#   git pull --no-rebase --ff-only                    # 不管 pull.rebase 怎麼設都能保留本機改動
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -q
.venv/bin/shortsloop --help
```

**本機改動的規則（只有一條）：** `thresholds.yaml`（4d 簽核後）和 `pipeline.yaml`（`gen_free_min_gb`）是 git 追蹤的檔案，一律**保留為未 commit 的本機改動**：
- 不要 `git stash`、`git commit`、`git checkout --`、`git reset --hard` 它們。thresholds 會變回 `calibrated: false`（run 拒跑），pipeline 一變 doctor 就過期。
- 更新一律用 `git pull --no-rebase --ff-only`。單純的 `git pull` 在設了 `pull.rebase=true` 時，只要有本機改動就拒絕（exit 128，叫你 commit 或 stash）。
- upstream 也改了這兩個檔時，pull 仍會拒絕（`Your local changes to the following files would be overwritten by merge`，exit 1），本機檔案不受影響。**不管因為什麼被拒，都停下來交給 owner。**
- pull 成功後依重跑表處理（動到 `shortsloop/` → 1 pytest、3 doctor；動到 `l1.py` 再加 4c 和 4d）。

**通過條件：**
- [ ] pytest 最後一行是 `453 passed`；工作站沒裝 `node` 時是 `451 passed, 2 skipped`（跳過的是標註 UI 的瀏覽器邏輯測試），同樣算通過。4d 簽核、改了 `pipeline.yaml` 之後也必須全綠。雲端 4 核跑了約 16 分鐘，工作站上的時間還沒量過。
- [ ] `.venv/bin/shortsloop --help` 列出 7 個子指令（`check run report label calibrate-batch calibrate-tune doctor`），exit 0。

**失敗時：**
- 測試因為 ffmpeg 出錯 → 回到第 0 步裝 ffmpeg。
- pip 抓不到套件 → 檢查能不能連 PyPI。
- pytest 會在 repo 裡產生 `runs/.shortsloop.lock`。這個檔已被 gitignore，不影響任何東西。
- **重新 clone：** `config.yaml`、`calibration/`（gitignore）和 `doctor.json` 都不在 git 裡。
  - 新 clone 必須放在**同一個絕對路徑**：先把舊的搬走（例如 `mv shorts-generator shorts-generator.old`）再 clone。`batch_manifest.jsonl` 記的是 clip 的絕對路徑，路徑一變、舊 checkout 一搬走，所有標註都會變成 stale，4c 會拒絕並刪掉舊的 `thresholds.proposed.yaml`，只能從 4a 重來。
  - 從 4d 的備份把 `calibration/`、`config.yaml`、`thresholds.yaml`、`pipeline.yaml` 複製回新的 checkout，再跑 1 pytest 和 3 doctor（doctor.json 沒有備份，沒有它 run 會拒跑）。

## 2. 設定

**目的：** 寫好 `config.yaml`，並了解 `pipeline.yaml` 裡哪些值實際有作用。

**指令：**
```bash
cp config.example.yaml config.yaml
$EDITOR config.yaml
```
必改的兩個欄位是 `workflow_t2v` 和 `runs_dir`，其他欄位要逐一核對：
```yaml
comfy:
  host: "127.0.0.1:8188"                                  # 只寫 host:port，不要加 http://
  workflow_t2v: "/home/<you>/workflows/wan22_t2v_api.json" # 一定要絕對路徑（相對路徑會依 cwd 解析）
judge:
  adapter: "ollama"
  base_url: "http://127.0.0.1:11434"                      # 一定要有 http://
  model: "qwen3-vl:8b-instruct"                           # 要和 `ollama list` 一字不差
  # unload_url 不要設：ollama 用 keep_alive:0 卸載，設了反而會取代它
rewrite:
  enabled: true
  model: "qwen3:8b"                                       # 沒 pull 就改成 enabled: false
paths:
  runs_dir: "/data/shortsloop/runs"                       # 一定要絕對路徑；GPU 鎖就放在這裡
```
用 openai_compat 的話，judge 段落改成這樣：
```yaml
judge:
  adapter: "openai_compat"
  base_url: "http://127.0.0.1:8080"          # llama-swap 的根網址，結尾不要加 /v1
  model: "<GET /v1/models 回傳的 id>"
  unload_url: "http://127.0.0.1:8080/unload" # 必填；GET 時要真的把 VLM 移出 VRAM
```
openai_compat 不會用到 `num_ctx` 和 `keep_alive_wave`。rewrite 一樣需要 Ollama（預設 `http://127.0.0.1:11434`），沒有的話就設 `enabled: false`。

`pipeline.yaml` 只會讀 `policies:` 這一段，`resources:`、`stages:` 都不會被讀；`COMFY_HOST` 環境變數對 shortsloop 也沒有作用。`policies.comfy.one_job_at_a_time` 會出現在下面 `load_policies` 的輸出裡（它是預設值），但沒有任何程式用到，改了沒有效果。值得知道的參數：

| key | 預設 | 說明 |
|---|---|---|
| `max_attempts_per_clip` | 3 | 第一次生成加 2 次 re-roll |
| `waves_max` | 3 | 實際最多跑 min(`waves_max`, `max_attempts_per_clip`) 輪；只調高 `max_attempts_per_clip` 不會多出 L2 re-roll 的輪數，要一起改 |
| `wall_clock_budget_h` | 6 | 只會擋「新開始」的生成；已經在跑的生成、判讀和 encode 會做完，所以總時間可能超過 |
| `disk_min_free_gb` | 20 | 開跑時不足 → exit 2；跑到一半不足 → `COMPLETED(disk-stopped)` |
| `comfy.timeout_s` / `poll_s` | 1800 / 5 | **要寫整數**，程式會用 int() 截斷：`poll_s` 0.5 → 0 會不停輪詢；`timeout_s` 小於 1 → 0 會讓每次生成一輪詢就逾時 |
| `judge.timeout_s` / `retries` | 300 / 1 | 夜間實際用的值；config.yaml 裡那組只給單獨執行的 `shortsloop-check` 用 |
| `judge.infra_escalation_after` | 2 | 連續幾次 L2 錯誤就 halt |
| `vram_handoff.free_min_gb` | 24 | 生成 → 判讀方向，空閒 VRAM 的門檻（GiB）；要留得下 judge |
| `vram_handoff.gen_free_min_gb` | （已註解掉） | 判讀 → 生成方向；**先保持註解**，第 3 步 doctor FAIL `judge.unload` 時會告訴你該填多少 |
| `vram_handoff.wait_timeout_s` | 180 | |

- key 打錯字不會報錯，程式會默默沿用預設值（打錯的 key 會多出一個不起作用的欄位）。doctor 的 `OK   pipeline:` 那一行只顯示 judge 和 vram_handoff 的值，所以要看全部實際生效的值，請跑：
  ```bash
  .venv/bin/python -c "import json; from shortsloop.settings import load_policies; print(json.dumps(load_policies('pipeline.yaml'), indent=1))"
  ```
- `thresholds.yaml` **不要手動改**。出廠是 `calibrated: false`，只有 4d 能改它；runner 偵測不到手動修改。

**通過條件：**
- [ ] 上面的 `load_policies` 輸出和你預期的值一致，沒有多出拼錯的 key（`comfy.one_job_at_a_time` 是預設就有的，不算）。
- [ ] 第 3 步 doctor 出現 `[doctor] OK   config: config.yaml` 和 `[doctor] OK   pipeline: … gen_free_min_gb …`（`OK` 後面是三個空白，level 欄補齊到 4 個字元）。

**失敗時：**
- doctor 顯示 `FAIL config: … top level must be a mapping` → YAML 結構寫錯了。這種情況下 `run` 會直接噴 traceback 並 exit 1。
- `FAIL pipeline: …` → 依照訊息修正型別，例如 `must be an integer >= 1`。

## 3. doctor

**目的：** 讓 doctor 實際驗證整套環境，並寫出 runner 唯一會看的 `doctor.json`。

**指令：**
```bash
.venv/bin/shortsloop doctor; echo "exit=$?"
```

**通過條件：**
- [ ] 最後一行是 `[doctor] ALL CHECKS PASSED · snapshot: doctor.json`，而且 `exit=0`。產生的 `doctor.json` 在 config.yaml 旁邊，內容有 `"ok": true`。
- [ ] 可以接受的 WARN：
  - `thresholds`（`calibrated=False — Phase 0 pending`，在第 4 步完成前是正常的）
  - `comfy.queue`
  - `cron.path`（ffmpeg／ffprobe 不在 `/usr/bin:/bin`；前提是第 6 步 crontab 的 `PATH=` 用 doctor 印出的值）
  - `rewrite.model`（最好還是修好，不然 off_prompt 的改寫會退化成單純換 seed）
  - `judge.latency`（出現的話建議調高 `policies.judge.timeout_s`，然後重跑 doctor）
- [ ] 抄下 `[doctor] hint: … put PATH=… at the top of the crontab` 裡的 `PATH=…`，第 6 步要用。
- [ ] （選做）全綠之後、4a 之前，可以先做第 5 步的 (a)(b)(c) 彩排，再回到 4a。

**VRAM 兩次讀數（第一次可能 FAIL，UNVERIFIED-ON-GPU）：**
- 是否 FAIL 取決於 judge 實際常駐多少。`judge.unload` 是 OK 的話，不用設 `gen_free_min_gb`，跳過這段。
- FAIL 時會看到 `FAIL judge.unload: with the VLM still loaded X GB is already free … ; set pipeline.yaml policies.vram_handoff.gen_free_min_gb: Y (between X loaded and Z unloaded)`。
- 意思是：judge 還在 VRAM 裡時，空閒量 X 就已經超過門檻，runner 分不出「judge 沒卸掉」和「卡是空的」。
```bash
.venv/bin/python -c "import json,sys; d=json.load(open(sys.argv[1])); print('ok:', d['ok']); [print(c['name'], c.get('vram')) for c in d['checks'] if c['name']=='judge.unload']" doctor.json
$EDITOR pipeline.yaml        # 在 policies.vram_handoff 底下取消註解，改成 gen_free_min_gb: Y
.venv/bin/shortsloop doctor; echo "exit=$?"    # pipeline.yaml 改了，一定要重跑
```
- Y 必須滿足兩個條件：
  - 明顯介於 loaded 和 unloaded 兩個讀數之間（X < Y < Z）。**直接用 doctor 建議的中間值**，不要設成等於或貼近 Z：每一輪生成前都要在 `wait_timeout_s` 內重新讀到 ≥ Y，讀數比 doctor 那次略低就會 `HALTED(infra)`。
  - `max(free_min_gb, Y)` 不能超過卡的總量，否則會變成 `FAIL comfy.gpu … the handoff can never pass`。
- 單位是 ComfyUI `/system_stats` 回報的 GiB，計算方式是 vram_free − torch_vram_free。32 GB 的卡會顯示略少於 32。
- 如果看到 `the unload freed too little VRAM for any threshold to tell the two apart`，代表卸載本身失敗了，或卡上還有別的模型。先用 `ollama ps` 和 `nvidia-smi` 查清楚。

**要注意：**
- 會讓 doctor.json 失效的東西：config.yaml、pipeline.yaml、workflow JSON 的內容，以及 `comfy.host`、judge 的 adapter/model。詳見上面的重跑表。
- doctor 一開始就把 doctor.json 寫成 ok:false，只有全綠跑完才改回 true。所以任何中斷或 FAIL 的 doctor 都會讓下一晚拒跑。
- 白天只是想診斷時用 `.venv/bin/shortsloop doctor --out /tmp/doctor-diag.json`，不會動到 `doctor.json`。正式的那一次不要加 `--out`，runner 只讀 config.yaml 旁邊那份。`--no-free` 一定會得到 ok:false。
- 不要在夜間 run、校準、resume 或監督執行進行中跑 doctor。doctor 也要拿 GPU 鎖，拿不到會寫成 `FAIL gpu.lock`；ComfyUI 還有 job 在跑時，它最多等 `wait_timeout_s` 就寫 `FAIL vram.handoff`。兩種都會讓下一次 run 拒跑。
- doctor 會建立 runs_dir，所以先把大磁碟掛好。

**失敗時（最常見的幾種）：**
- `comfy.server … unreachable`：ComfyUI 沒在跑，或 host 前面多寫了 `http://`。這種情況下 workflow 和模型的檢查都會被跳過。
- `workflow: … not API format`：用 Export (API) 重新匯出。
- `workflow.knobs: … does not expose …`：有參數是連線而不是常數，例如 fps 接了 link。
- `models.loaders`：用了非核心的 loader，例如 GGUF。
- `models.<folder>`：伺服器上缺那個檔。
- `judge.model`：
  - `… not installed … (ollama pull …)` → 去 pull，或讓 `judge.model` 和 `ollama list` 一字不差。
  - `judge unreachable at … (unknown url type: …)` → `judge.base_url` 少了 `http://`。
  - `judge unreachable at … (… Connection refused)` → Ollama 沒在跑。
- `judge.vision` 或 `judge.l2_dryrun`：
  - `not run: …` → 先修好 `vram.handoff` 或 `judge.unload`。
  - `l2_dryrun` 逾時（`slower than the nightly timeout`）或呼叫失敗 → 調高 `judge.num_ctx` 或 `policies.judge.timeout_s`，再重跑 doctor。
  - `vision probe call failed` 或顏色答錯（`model did NOT see the images`）→ Ollama 的 vision 路徑壞了，改用 openai_compat（runbook §0）。
- `vram.handoff: ComfyUI queue busy (…)`：ComfyUI 上還有 job，等它跑完再跑 doctor。
- `disk`：runs_dir 空間不足 20 GiB。
- `gpu.lock`：有別的 shortsloop 程序在跑，等它結束再跑 doctor。

## 4. Phase 0 校準

calibrate-batch 和 calibrate-tune 都不會檢查 doctor.json，所以**一定要先讓第 3 步全綠**。三個指令都在 repo 根目錄執行，都用預設的 `calibration/`。細節見 runbook §2。

### 4a. calibrate-batch

**目的：** 產生 40 支刻意有好有壞的 480x832、81 格草稿片，外加 4 列換 prompt 的資料（當作 off_prompt 的正確答案）。

**指令：**
```bash
.venv/bin/shortsloop calibrate-batch --dry-run         # 只用 CPU：印出「40 rows + 4 prompt-swap rows」
.venv/bin/shortsloop calibrate-batch --count 40; echo "exit=$?"
ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=width,height,nb_read_frames -of csv=p=0 calibration/clips/cal001.mp4
```

**通過條件：**
- [ ] `exit=0`，最後一行是 `[calibrate-batch] manifest: calibration/batch_manifest.jsonl (44 rows). Next: shortsloop label --calibration calibration`。
- [ ] 一開始有印出 `[calibrate-batch] VRAM verified free for Wan: … GB`。
- [ ] `calibration/clips/` 裡有 cal001 到 cal040。抽 2–3 支跑上面的 ffprobe，應該得到 `480,832,81`。

**失敗時：**
- exit 0 但不到 `44 rows`，或輸出有 `FAILED (…) — continuing; re-run to retry this row` → 重下同一個指令，只會補失敗的列，直到 44 列再進 4b。
- exit 2（拒跑）：config 或 pipeline 有問題、GPU 鎖被佔用、缺 `comfy.host` 或 `workflow_t2v`、openai_compat 沒設 `unload_url`。
- exit 3（INFRA）：
  - ComfyUI 連不上。
  - VRAM 沒釋放（`ollama ps`）。
  - workflow 在第一次送出時被拒。這一步事先不會檢查參數能不能改，第 3 步 doctor 會。
  - workflow 一個 job 輸出不只一支影片（`INFRA: workflow produced N video outputs … need exactly one`）；doctor 不檢查這點。只留一個 `SaveVideo`（或一個 `save_output=true` 的 `VHS_VideoCombine`），重新匯出，重跑 doctor，再重下 4a。
  - 累計 5 次生成失敗。
- exit 130：你按了 Ctrl-C，在途的 job 已被移除。重下同一個指令就會接著跑，做完的列會跳過。
- 注意：
  - 草稿片的 fps 和 steps 用的是 workflow 檔裡的值，不會被改掉。**校準時用的 workflow 必須和夜間用的是同一個檔。**
  - manifest 裡已有的列永遠會被跳過。要整批重做，先把 `calibration/` 搬到 repo 外（見重跑表）。
  - manifest 記的是 clip 的絕對路徑：之後不要搬動 repo 或 `calibration/`（見第 1 步「重新 clone」）。

### 4b. label 🛑 人工閘門 1

**目的：** 把你對每支片的判斷記錄成 `calibration/labels.jsonl`，門檻就是照這個調出來的。

**指令：**
```bash
.venv/bin/shortsloop label                 # 在工作站上執行
# 筆電另開一個終端機：
ssh -L 8765:127.0.0.1:8765 llm             # 然後在瀏覽器打開 http://127.0.0.1:8765/
```
按鍵：`1` pass、`2` static、`3` deformed、`4` flicker、`5` off_prompt、`6` other（會要你填備註）、空白鍵重播、`p` 上一支、`n` 跳過。全部標完後在工作站按 Ctrl-C。

**通過條件：**
- [ ] 按 Ctrl-C 後印出 `[label] labels: calibration/labels.jsonl (44/44)`，exit 0。
- [ ] 至少要有 10 個可用的標註，而且 pass 和 fail 都要有，否則 4c 會拒絕。
- [ ] 換 prompt 的那幾列一樣要標。

**失敗時：**
- exit 1 加上 `[label] no rows in calibration/batch_manifest.jsonl — run calibrate-batch first` → 你不在 repo 根目錄，或 `--calibration` 指錯資料夾。
- 標錯了 → 按 `p` 回去重標，每支片以最後一筆為準。4c 之後才重標的話，要重做 4c 和 4d。
- 埠被佔用 → 加 `--port N`，ssh tunnel 也用同一個 N。
- 不要用 `--host <LAN IP>`，這個 UI 沒有驗證。
- 夜間 run 產生的片不能拿來標：不在 `batch_manifest.jsonl` 裡的標註，4c 會當成過期排除。

### 4c. calibrate-tune --with-l2，然後讀報告

**目的：** 讓 VLM judge 依照夜間的方式為每支片打分，再調出 L1 門檻和 L2 floors 的提案。

**指令：**
```bash
.venv/bin/shortsloop calibrate-tune --with-l2; echo "exit=$?"
less calibration/tuning_report.md
grep -n FALLBACK calibration/tuning_report.md     # 沒有輸出 = OK
```

**通過條件：**
- [ ] `exit=0`，stdout 有 `[calibrate-tune] TEST (L1+L2): agreement …, false-pass a/b labeled-fail clips, false-fail c/d labeled-pass clips — review, then sign off with: …`。範圍必須是 **L1+L2**，不能是 `L1 only`。
- [ ] 產生了這些檔：`calibration/tuning_report.md`、`tuning_report.json`、`thresholds.proposed.yaml`（`calibrated: false`）、`l2_scores.jsonl`。
- [ ] 讀 `tuning_report.md` 時逐項確認：
  - [ ] 開頭那一行 `judge:` 是你設定的 judge，例如 `` `ollama/qwen3-vl:8b-instruct` @ `sha256:…` ``。如果寫 **none**，代表沒有 L2 分數，**不要簽核**。
  - [ ] `grep -n FALLBACK` 沒有輸出。報告裡的寫法是 `- ⚠️ **L2 FLOORS ARE A FALLBACK — …`（中間有粗體符號，整句照抄去搜會找不到）；這時 stdout 在 `TEST (L1+L2)` 那行前面也會印 `[calibrate-tune] WARNING: L2 floors are a FALLBACK, not a fit — …`。有的話，被列出的片就是 judge 和你的標註衝突的地方。
  - [ ] 看有沒有 `⚠️ no held-out examples of: …`。有的話，那一類在 test 集的表現沒量到。
  - [ ] `## Performance` 底下的 **TEST (held out)**：`false-pass x/n labeled-fail clips would ship`（含 95% CI；樣本少，CI 會很寬），以及 `false-fail`。
  - [ ] 看有沒有 `⚠️ L1-class clips that would ship`。
  - [ ] `## Per-layer catch attribution (TEST)`：看 `shipped (missed)`，以及 L2 和 `judge ERROR` 各佔多少。
  - [ ] `## Excluded / unjudgeable clips`：如果很多片是 `judge could not evaluate`，代表 judge 本身有問題。修好後（例如調高 `policies.judge.timeout_s` 並重跑 doctor）要先 `mv calibration/l2_scores.jsonl calibration/l2_scores.old.jsonl` 再跑 `--with-l2`，否則錯誤列會被沿用、不會重評。

**什麼時候換大一點的 judge？** 程式和文件都沒有數字標準，由你判斷。可以參考這些訊號：
- 出現 FALLBACK 警告。
- TEST 裡被你標成 deformed 或 off_prompt 的片仍然通過了 L1+L2。
- L2 誤殺很多，或 judge ERROR 很多。

要換的話：
```bash
ollama pull <約 32B 的 VL 模型>                    # tag 由你挑，repo 沒有指定
$EDITOR config.yaml                               # 改 judge.model
.venv/bin/shortsloop doctor; echo "exit=$?"       # config 改了必須重跑；必須 ALL CHECKS PASSED，並看 judge.unload 的新讀數
.venv/bin/shortsloop calibrate-tune --with-l2     # 全部片會用新 judge 重新打分，只採用新 judge 的分數
```
- doctor 出現 `judge.l2_dryrun` 逾時或 `judge.latency` WARN 時，**先**調高 `pipeline.yaml` 的 `policies.judge.timeout_s` 並重跑 doctor，再跑 `--with-l2`。打分用的是 pipeline.yaml 的 timeout／retries，逾時的片會記成錯誤列，之後重跑不會重評。
- 判讀時用 `ollama ps`／`nvidia-smi` 量新 judge 的常駐大小。`policies.vram_handoff.free_min_gb`（24）必須 ≥ 這個大小（且不超過卡的總量），因為生成 → 判讀方向只驗這個值。
- 順序：打分之後只能再改 `free_min_gb` 和／或依 doctor 新讀數重設的 `gen_free_min_gb`（這兩個不影響分數）。量好之後一次改完 `pipeline.yaml`，**最後一次修改之後**重跑 doctor，直到 `ALL CHECKS PASSED`。改完 pipeline.yaml 卻沒重跑，run 會以 `changed since doctor: pipeline_sha256` 拒跑。

**失敗時：**
- exit 2，而且輸出是 `REFUSING to propose thresholds: …`（調參本身拒絕；舊的 `thresholds.proposed.yaml` 會被刪掉）：
  - 可用標註少於 10 個，或全部是 pass、全部是 fail。
  - held-out 分出來的 test 集缺少 pass 或缺少 fail（訊息結尾是 `label more`）。
  - judge 只評了一部分的片。
  - 標註全部 `stale/unverifiable`：clip 檔不在 manifest 記錄的絕對路徑上（repo 或 `calibration/` 被搬過，見第 1 步「重新 clone」）。
- exit 2，但是 `--with-l2` 在打分前就拒絕（**舊的提案會留著**）：沒有 manifest、config 或 pipeline 不能用、GPU 鎖被佔用、缺 `comfy.host`、openai_compat 沒設 `unload_url`。
- 要「label more」的話：`.venv/bin/shortsloop calibrate-batch --count 56`（或更多）。前 40 列和原本相同，已在 manifest 裡會被跳過，只會生成 cal041 之後的片。接著跑 `label`（從還沒標的片開始，已標過的按 `n` 跳過），再跑 4c `--with-l2`（只會評新的片），最後 4d。
- exit 3：
  - judge 連不上，或模型沒 pull。
  - 判讀前 VRAM 驗證不過。
  - judge 在中途掛掉。已經評好的分數會保留，重跑會接著評。
- 不加 `--with-l2` 的 `calibrate-tune` 會沿用 `l2_scores.jsonl` 裡同一個模型名稱最新的分數。所以只要重新 pull 過 judge，就一定要加 `--with-l2`。

### 4d. calibrate-tune --approve 🛑 人工閘門 2

**目的：** 你簽核之後，`thresholds.yaml` 會變成 `calibrated: true`，無人值守的閘門才會打開。

**指令：**
```bash
.venv/bin/shortsloop calibrate-tune --approve; echo "exit=$?"
git diff --stat thresholds.yaml
B=/data/shortsloop/backup/$(date +%F); mkdir -p "$B" && cp -a calibration config.yaml thresholds.yaml pipeline.yaml "$B"/   # 備份
echo doctor.json >> .git/info/exclude               # doctor.json 沒有被 gitignore
```

**通過條件：**
- [ ] 只有在 4c **最後一次**是 exit 0、而且印出 `TEST (L1+L2)` 那一行時才簽。
- [ ] `exit=0`，輸出 `[calibrate-tune] APPROVED → thresholds.yaml (version YYYY-MM-DD.N, calibrated: true). The unattended gate is now open.`
- [ ] **不需要**重跑 doctor，因為 thresholds 不在 fingerprint 裡。

**失敗時（exit 2）：**
- `nothing to approve`：還沒做 4c。
- `labels.jsonl changed since tuning`、batch_manifest 或 l2_scores 有變：重跑 4c。
- 數值被手動改過。
- `l1_impl` 變了：重跑 4c。
- config 裡的 judge 和調參時用的不同：改回原本的 judge，或重跑 4c。
- `no judge scores behind this proposal`：要做 4c 的 `--with-l2`。

**注意：**
- 不要重複 approve。每次 approve 的版本號都比 thresholds.yaml 目前的版本高一號，verdict 快取的 key（thresholds 檔的 sha256）也會跟著變。
- `calibration/`、`config.yaml` 是 gitignore 的，`thresholds.yaml`、`pipeline.yaml` 依第 1 步的規則不 commit，所以上面的備份是 repo 外唯一的副本；之後任何重調都需要同一份 `calibration/`，而且只能還原到**同一個絕對路徑**的 checkout（見第 1 步「重新 clone」）。
- approve 會把 thresholds.yaml 改寫成區塊格式（`flicker:` 的 `value:` 在另一行），要看某個門檻請用 `.venv/bin/python -c "import yaml; print(yaml.safe_load(open('thresholds.yaml'))['l1']['flicker'])"`，不要 grep。
- 不建議的監督用例外：`.venv/bin/shortsloop calibrate-tune --approve --accept-untested-l2`。效果見「需要你決定的事」第 3 點；這樣簽完 cron 不能用。

## 5. 第一次在旁邊看著跑

**目的：** 用一張真實的派工單走完一整晚的流程，在旁邊看著，並把其中一支片重現出來，把整個流程驗證一遍。

**指令：**
```bash
# (a) 用 shorts-trend-dispatch skill 產生派工單（第一晚大約 6 支 T2V），放到 /data/shortsloop/dispatch/
# (b) 只用 CPU 的檢查（直接呼叫內部 parser，不是 CLI 指令）：
.venv/bin/python -c "import sys; from shortsloop.dispatch import parse_dispatch as p; s=p(sys.argv[1]); print('accepted:', [c.clip_id for c in s.clips]); print('skipped:', [(r['clip_id'], r['reason']) for r in s.skipped]); print('shared:', s.shared, '| MISSING:', sorted({'width','height','length','fps'}-set(s.shared)) or 'none'); print('ERRORS:', s.errors or 'none'); sys.exit(1 if s.errors or not s.clips else 0)" /data/shortsloop/dispatch/dispatch-YYYY-MM-DD.md
# (c) 選做彩排：屬於總覽的 3b，只在 doctor 全綠之後、4d 之前（thresholds.yaml 還是 calibrated: false）做
.venv/bin/shortsloop run --dispatch /data/shortsloop/dispatch/dispatch-YYYY-MM-DD.md --allow-uncalibrated
# (d) 4d 之後的正式監督執行（不加任何旗標）
.venv/bin/shortsloop run --dispatch /data/shortsloop/dispatch/dispatch-YYYY-MM-DD.md; echo "exit=$?"
```

**派工單要符合的格式：**
- `共用參數` 表裡：
  - `解析度` 那一列**只能有一個** WxH。寫成 `480x832 draft → 720x1280 final` 會被拒。
  - `length / fps` 那一列要有一個 4n+1 的格數，加一個 fps。
  - 缺了的值會個別改用預設（720x1280／81／16）並在 stderr 警告。
- manifest 的標題要含有 `生產清單` 或 `manifest`。
- 要生成的 T2V 列，`image 依賴` 欄必須是空的或 `—`。有填東西的列，以及 I2V 列，都會被跳過並在報告中列出。
- 每個 clip 要有 `### V{n}C{k}` 標題（剛好三個 `#`），下面接一個 code fence 包住的 prompt。
- 表單裡的 `T2V workflow` 那一列會被忽略。workflow 永遠用 config.yaml 的 `comfy.workflow_t2v`。

**通過條件：**
- [ ] (b) 印出 `MISSING: none` 和 `ERRORS: none`，exit 0；`accepted` 有你要的 clip。
- [ ] (c) 報告頂端顯示 `UNCALIBRATED supervised run`，PASS 的片只會進 `encoded_uncalibrated/`。這只是彩排，產出不能出貨。4d 之後再加 `--allow-uncalibrated` 就不會有這個標示，PASS 會直接進 `encoded/`，所以 4d 之後跳過 (c)。
- [ ] (d) exit 0，最後一行是 `[shortsloop] COMPLETED: n/m clips passed · report: <runs_dir>/<UTC YYYYMMDD-HHMMSS>-nightly/report.md`。
- [ ] `report.md` 的 Run summary 要逐項看：
  - Status
  - Verdicts
  - `Attempts used … respected: YES`
  - Wall clock
  - `Generation | … from the sheet`（不能是 fallback）
  - `Thresholds | … (calibrated: True)`
  - 每支片的判定理由
- [ ] 通過的片都有 `encoded/<cid>.mp4`。用 ffprobe 檢查應該得到 `h264,1080,1920,30/1` 和 `aac,0/0`：
  `ffprobe -v error -show_entries stream=codec_name,width,height,r_frame_rate -of csv=p=0 <run_dir>/encoded/V1C1.mp4`
- [ ] 從 `attempts.jsonl` 重現一支片。必須在沒有任何 shortsloop 程序拿著 GPU 鎖、而且 `ollama ps` 是空的時候做，因為 vendored client 會繞過鎖和 VRAM 交接：
```bash
RUN=/data/shortsloop/runs/<run_id>
.venv/bin/python -c "import json,sys; [print(json.dumps({k:r.get(k) for k in ('clip_id','attempt','verdict','workflow_path','workflow_sha256','patch_args','output_sha256')})) for r in map(json.loads, open(sys.argv[1])) if r['clip_id']==sys.argv[2]]" "$RUN/attempts.jsonl" V1C1
# 取 verdict 是 PASS 的那一列；它的 attempt 就是下面的 <n>，seed、steps 也都用同一列的
sha256sum <workflow_path>                 # 必須等於 workflow_sha256
OUT=/tmp/repro-$(date +%s)                # 每次用新的空資料夾
COMFY_HOST=127.0.0.1:8188 python3 shortsloop/vendor/comfy_client.py run -w <workflow_path> --prompt "$(cat $RUN/prompts/V1C1_a<n>.txt)" --seed <patch_args.seed> --width <w> --height <h> --length <length> --fps <fps> [--steps <patch_args.steps>] --out "$OUT"
sha256sum "$OUT"/*.mp4                    # 和 output_sha256 比對（GPU 上是否逐位元相同還沒驗證，請記錄結果）
```
  `patch_args` 裡有 `steps` 才加 `--steps`。COMFY_HOST 就填 config.yaml 的 `comfy.host`。

**失敗時：**
- exit 2 或 exit 3 → 查第 8 步的對照表。
- 這個 vendored client 的 exit code：0 成功；1 連不上伺服器或 workflow 壞了；2 執行錯誤或參數用法錯誤；3 逾時；4 送出時被 `/prompt` 拒絕。
- 報告的判定和你親眼看到的不一致（門檻不可信）→ 先不要設 cron。用 `git checkout -- thresholds.yaml` 把它換回 repo 的出廠版 `calibrated: false`（第 1 步規則的唯一例外；簽核版還在 4d 的備份裡），runner 就會拒絕無人值守。然後擴充 Phase 0：`calibrate-batch --count N`、label、4c、4d。rollback 後在同一天（UTC 日期）重新簽核，版本號可能和備份裡那份相同；用 run.json 的 `thresholds.sha256` 分辨，或在備份資料夾名稱記下舊版本號。

## 6. cron 無人值守

**目的：** 讓 runner 每晚在最小的 cron 環境裡自己跑起來。

**指令：**
```bash
timedatectl | grep 'Time zone'                     # cron 用系統時區
env -i PATH=/usr/bin:/bin sh -c 'for b in ffmpeg ffprobe bash; do command -v "$b" || echo "MISSING $b"; done'   # PATH 換成 doctor 印出的那一個
crontab -e
```
crontab 內容（**每筆排程只能寫在一行**，cron 不支援行尾 `\` 接續；`{ …; }` 讓 `cd` 失敗的訊息也寫進 cron.log）：
```
PATH=/usr/bin:/bin
58 1 * * * mkdir -p /data/shortsloop/runs && { cd /home/<you>/shorts-generator && .venv/bin/shortsloop run --dispatch /data/shortsloop/dispatch/tonight.md; } >> /data/shortsloop/runs/cron.log 2>&1
```
- `PATH=` 那一行請用第 3 步 doctor 印出來的值。
- `58 1` 是系統時區的 01:58。系統時區不是 Asia/Taipei 時要換算，例如系統是 UTC 就寫 `58 17 * * *`。
- **絕對不要**在 crontab 裡放 `--skip-doctor` 或 `--allow-uncalibrated`。用了 `--skip-doctor` 的話，report.md 不會有任何警告（只有 HALTED 時，橫幅裡的 resume 指令會帶著 `--skip-doctor`；run.json 的 `doctor` 是 null）。
- 派工單路徑如果用日期組出來，裡面的 `%` 要寫成 `\%`。
- 一個 runs_dir 同時只能有一個 runner。鎖在 `<paths.runs_dir>/.shortsloop.lock`，跟 `--runs-dir` 無關。第二個程序會 exit 2。鎖是 flock：程序一結束（包括被 kill）就自動釋放，檔案留著是正常的。**不要刪這個檔**，刪了之後第二個程序會在新檔上拿到鎖，等於兩個程序同時用 GPU。

**通過條件：**
- [ ] `Time zone` 是 `Asia/Taipei`，或 crontab 的小時已換算成系統時區。只有 run 資料夾名稱是 UTC。
- [ ] `env -i` 那一行三個工具都找得到，沒有出現 `MISSING`。
- [ ] `crontab -l` 顯示上面兩行。
- [ ] 隔天早上 `cron.log` 有一行 `[shortsloop] …: n/m clips passed · report: …`。
- [ ] 每天都有人把當晚的派工單放到 `tonight.md`（見第 7 步）。runner 不會檢查「這張昨天跑過」，沒換的話就會把同一張再跑一次。
- [ ] 01:58 前 ComfyUI 和 Ollama 都已經在跑，而且白天的 resume、校準、doctor 和第 5 步的監督執行都已經結束。最簡單的做法是第 5 步跑完才裝 crontab。
- [ ] 時間預算：01:58 開跑加 6 小時 ≈ 08:00，再加上最後一批收尾的時間。

**失敗時：**
- `cron.log` 完全沒有新的一行 → 先確認 cron 有沒有觸發：`grep CRON /var/log/syslog` 或 `journalctl -u cron`，再看本機 mail。常見原因：排程寫成了兩行、`%` 沒跳脫、`mkdir -p` 沒有寫入權限。
- `cron.log` 有 `cd: can't cd to …` → crontab 裡的 repo 路徑寫錯了。
- `REFUSING TO RUN: ffmpeg not on PATH` → crontab 的 `PATH=` 不對。
- 其他狀況 → 查第 8 步的對照表。

## 7. 每天早上

**目的：** 看結果、審片、配音、上傳、放好明晚的派工單；如果夜裡中斷了，就接著跑完。

**指令：**
```bash
cd /home/<you>/shorts-generator
tail -n 3 /data/shortsloop/runs/cron.log
ls -t /data/shortsloop/runs | head -3            # 資料夾名稱一律是 UTC：台灣 01:58 = 前一天 17:58 UTC
RUN=/data/shortsloop/runs/<run_id>
less $RUN/report.md; ls $RUN/encoded/ $RUN/sheets/
# 配音：輸入用該 clip 判 PASS 的那一次 clips/<cid>_a<n>.mp4（attempts.jsonl 裡的 output_path），依派工單順序排
bash shortsloop/vendor/ig_encode.sh -o /data/shortsloop/final/V1.mp4 -a /path/to/bgm.mp3 $RUN/clips/V1C1_a1.mp4 $RUN/clips/V1C2_a2.mp4
# 明晚的派工單：用 shorts-trend-dispatch 產生，複製到工作站後
cp dispatch-YYYY-MM-DD.md /data/shortsloop/dispatch/tonight.md
# 再用第 5 步 (b) 的檢查指令檢查 /data/shortsloop/dispatch/tonight.md
```

**通過條件：**
- [ ] 讀 `report.md`：看狀態、每支片的理由，對照 `sheets/<cid>_a<n>.jpg` 的 contact sheet。
- [ ] **只有 `encoded/` 裡有檔的片才能出貨**，也就是 checker 判 PASS 而且已校準。`encoded_uncalibrated/` 裡的一律不能出貨。
- [ ] 依照派工單的 `音檔需求` 表配音。`ig_encode.sh` 回傳 exit 0；用 ffprobe 檢查輸出應該是 `h264,1080,1920,30/1` 加 aac。原始音軌一律丟掉；BGM 會循環或裁切到影片長度，結尾 1.5 秒淡出。也可以拿 `encoded/<cid>.mp4` 當輸入，但會多壓一次畫質。
- [ ] 上傳時打開 AI 內容標示。
- [ ] `tonight.md` 已換成明晚的單，5(b) 檢查 exit 0、`MISSING: none`。某晚不想跑：刪掉 `tonight.md`（當晚 exit 2，什麼都不生成），或把 cron 那行註解掉。

**夜裡中斷了要接著跑（報告會顯示 `⛔ HALTED(...)`）：**
```bash
.venv/bin/python -c "import json,sys; print(json.load(open(sys.argv[1]))['run']['resume_command'])" $RUN/report.json
# 印出來的是 `shortsloop run …`，前面補上 .venv/bin/，在 repo 根目錄執行，例如：
.venv/bin/shortsloop run --dispatch $RUN/dispatch.md --resume $RUN --config config.yaml --pipeline pipeline.yaml --thresholds thresholds.yaml
```
- 修問題時改了 config.yaml、pipeline.yaml 或 workflow JSON → resume 前先重跑 doctor（重新匯出 workflow 時存成新檔名並更新 config.yaml 的 `workflow_t2v`，見下面「磁碟保留」）。跑 doctor 之前先確認：
  - `COMFY_HOST=<comfy.host> python3 shortsloop/vendor/comfy_client.py queue` 印出 `queue empty`（中斷後 ComfyUI 上的 job 會繼續跑完）；
  - `flock -n /data/shortsloop/runs/.shortsloop.lock true && echo lock-free` 印出 `lock-free`。有 run、resume、doctor 或校準拿著鎖時它 exit 1，什麼都不印。不要用 `pgrep -af shortsloop` 判斷：開著 `less …/report.md` 也會被列出來。
  - 否則 doctor 會 `FAIL vram.handoff: ComfyUI queue busy` 或 `FAIL gpu.lock`，resume 會被拒。doctor 沒有以 `ALL CHECKS PASSED` 結束就再跑，直到全綠才 resume。
- 程序是被 kill、OOM-killer 殺掉或重開機的話，不會有 report.md，用上面同樣格式的指令手動 resume。原本加過的旗標要一起帶上；cron 跑的那一晚沒有加旗標。
- resume 要在 01:58 之前跑完，不然當晚的 run 會因為鎖被佔用而 exit 2。
- resume 時，attempt 已用完（預設 3 次）的片不會再試；還有額度的片（包括觸發 halt 的那支）會繼續生成。用完次數的片要重做，就放進新的派工單。
- resume 會把之前實際跑掉的時間算進預算（report 的 Wall clock）。已經**超過** `wall_clock_budget_h` 時，resume 只會把在途和已生成的片判讀、encode 完，不會開始新的生成。還差一點才超過時，resume 仍會開始新的生成，每支最長可跑到 `comfy.timeout_s`。要補完剩下的片：把沒做完的 clip 放進新的派工單；或暫時調高 `policies.wall_clock_budget_h` → 重跑 doctor → resume → 改回來並在 01:58 前再跑一次 doctor。
- `.venv/bin/shortsloop report $RUN` 可以用 report.json 重新產生 report.md。沒有 report.json 時會 exit 2。

**安全停止（要中途停掉 run 時）：**
```bash
pgrep -af 'shortsloop run'     # 取 /home/<you>/shorts-generator/.venv/bin/python3 .venv/bin/shortsloop run … 那一行的 PID（不是 /bin/sh -c 那行）
kill -INT <PID>                # 等同 Ctrl-C：HALTED(interrupted)，寫出 report 和 resume_command，exit 3
```
- 不要用 `kill`（SIGTERM）或 `kill -9`：程序會直接死掉，沒有 report，只能手動 resume。
- run 的 Ctrl-C／`kill -INT` 不會停掉 ComfyUI 上正在跑的 job（resume 會接回去）。不打算 resume 的話：`COMFY_HOST=<comfy.host> python3 shortsloop/vendor/comfy_client.py interrupt`。

**磁碟保留：** shortsloop 不會刪舊的 run，`clips/`、`encoded/`、`sheets/` 每晚累積，直到 20 GiB 底線觸發。每週看一次 `du -sh /data/shortsloop/runs`。
- 只刪你不會再 `--resume` 的 run。上傳後可以刪舊 run 的 `clips/` 和 `encoded/`。
- `attempts.jsonl`、`events.jsonl`、`prompts/`、`run.json`、`report.*` 要留著（很小，重現用）。
- workflow JSON 不要原地覆寫，新版存成新檔名，舊 run 的 `workflow_sha256` 才重現得出來。

## 8. 出錯對照

| 症狀／exit code | 意思 | 處理方式 |
|---|---|---|
| `run` exit 2，`[shortsloop] REFUSING TO RUN: …` | 開跑前就拒絕，沒有產生任何東西 | 看冒號後面的原因，見下面幾列 |
| ↳ `no doctor snapshot` / `last doctor run FAILED (…)` / `changed since doctor: …` | doctor 快照不存在（例如重新 clone）、失敗（包括中斷的 doctor），或已經過期 | 修好後重跑 3 doctor |
| ↳ `config not found: config.yaml` | 不在 repo 根目錄執行 | `cd` 到 repo |
| ↳ `ffmpeg not on PATH` | cron 的 PATH 不對 | 改 crontab 的 `PATH=` |
| ↳ `thresholds.yaml has calibrated: false` | Phase 0 還沒簽核（或 thresholds.yaml 被 checkout／stash 掉了） | 做完 4a–4d，或從 4d 的備份還原 |
| ↳ `approved with UNTESTED L2 floors` | 當初用了 `--accept-untested-l2` | 重做 4c `--with-l2` 和 4d |
| ↳ `tuned against different L1 metric code (l1_impl …)` | `l1.py` 改過了 | 重做 4c 和 4d |
| ↳ `calibrated with judge X but config.yaml uses Y` | 換了 judge | 改回原本的 judge，或重做 4c 和 4d |
| ↳ `judge build … differs` / `cannot identify the judge build` | judge 被重新 pull，或 Ollama 沒在跑 | 拉回原本的 build 或重做 4c 和 4d；把 Ollama 啟動 |
| ↳ `dispatch sheet failed intake` | 派工單格式錯誤，或 `tonight.md` 不存在（`cannot read dispatch sheet`） | 看前面幾行 `dispatch error:`，修好或放好派工單，用 5(b) 再檢查一次 |
| ↳ `workflow unusable` / `does not expose …` | workflow 有問題 | 重新匯出，然後跑 doctor |
| ↳ `another shortsloop process holds …/.shortsloop.lock` | 已經有別的程序在跑 | `pgrep -af '\.venv/bin/shortsloop'` 找出是誰，等它結束（`flock -n <runs_dir>/.shortsloop.lock true` exit 0 就是放開了）；**不要刪鎖檔** |
| ↳ `disk: X GB free … < 20 GB floor` | 磁碟空間不夠 | 依第 7 步「磁碟保留」清出空間 |
| ↳ `--resume: … differs from the run's own copy` | resume 時給了不同的派工單 | 用 `<run_dir>/dispatch.md` |
| ↳ `judge.adapter 'openai_compat' has no judge.unload_url` | judge 沒辦法卸載；通常會先看到 `last doctor run FAILED (judge.unload …)` | 在 config 加上 `unload_url`，重跑 doctor |
| `run` exit 1 加上 Python traceback | config.yaml 結構錯誤，或 runs_dir 無法建立 | 跑 doctor，它會報 FAIL |
| `run` exit 3，`HALTED(infra)` | 夜裡中途壞掉：ComfyUI 連不上（開跑時就連不上也算，run 資料夾會先建好）、ComfyUI queue 一直有別的 job 或卡住的 job 清不掉、workflow 被 ComfyUI 拒收、VRAM 沒釋放、judge 連不上或回 HTTP 錯誤、judge 的 build 在夜裡變了、連續 3 次生成失敗、連續 2 次 L2 錯誤、checker 卡住、workflow 輸出不是剛好一支影片 | 看 report.md 頂端的原因，修好後用第 7 步的方式 resume |
| `run` exit 3，`HALTED(interrupted)` / `HALTED(crash)` | 按了 Ctrl-C（或 `kill -INT`），或 runner 發生例外 | resume |
| exit 0，`COMPLETED(budget-stopped)` / `(disk-stopped)` | 時間預算或磁碟底線觸發，不再開始新的生成 | 報告裡被跳過的列會寫原因；有需要就調整 `pipeline.yaml`（改完重跑 doctor） |
| exit 0，但報告裡有 failed | 正常運作：壞片被抓到並且有上限 | 不需處理 |
| 有 OOM，或一直生成失敗 | 每次失敗都會用掉一次 attempt | 把派工單的 `解析度` 改成 480x832，或把長度改成較短的 4n+1（例如 49 frames） |
| 報告出現 `encode failed after PASS` | encode 驗證沒過 | 檢查 ffmpeg 有沒有 libx264 和 aac；原始片還在 `<run_dir>/clips/` |
| `reroll_action` 出現 `steps8 n/a` | workflow 不是 4 步版本，steps 8 不套用（flicker 只換 seed；static 仍會加 motion phrase 並換 seed） | 不需處理 |
| cron.log 沒有 `[shortsloop] …` 結尾行，也沒有 report.md | 程序被 kill 或機器重開（或 cron 沒觸發，見第 6 步） | 用第 7 步的手動 resume |
| 其他指令：doctor 2、calibrate-batch 2／3／130、calibrate-tune 2／3、label 1、report 2 | — | 見 3、4a、4c／4d、4b、7 的「失敗時」 |

## 9. 第一晚要親眼確認的 UNVERIFIED-ON-GPU 項目

- [ ] 在工作站上 `doctor` 全綠，其中 `judge.vision`、`judge.l2_dryrun`、`judge.unload` 都是 OK
- [ ] `doctor.json` 裡 `judge.unload` 的 `vram` 記錄了兩個讀數；有設 `gen_free_min_gb` 的話，它在兩者之間（記下第一次 doctor 是否 FAIL `judge.unload`）
- [ ] `qwen3-vl:8b-instruct` 和 `qwen3:8b` 這兩個 tag 都 pull 成功；Ollama 的 vision 功能可以用
- [ ] `calibrate-batch` 產出 40 支能播放的草稿（抽查的 ffprobe 結果是 `480,832,81`），記下實際花了多久
- [ ] 判讀階段在 `/free` 之後 VRAM 放得下（做 4c 和第一晚時，各看一次 `nvidia-smi`）
- [ ] 判讀之後的生成真的沒有 judge 常駐：有跑到第 2 輪時，`grep -nE '"layer": "(vram_handoff|judge_unload)"' <run_dir>/events.jsonl` 裡，除了最後一行 `judge_unload` 之外，每一行 `judge_unload` 後面都接著一行 `"stage": "generate"` 的 `vram_handoff`（只有下一輪真的有新生成時才會出現），而且沒有 halt；並在第 2 輪生成時親眼看 `ollama ps` 是空的、`nvidia-smi` 沒有 judge。（只有一輪的夜晚不會有這種行；這行只在檢查通過後才寫，所以它的數值本身證明不了什麼）
- [ ] 真實 Wan 片的 L1 數值合理：`<run_dir>/verdicts/<cid>_a1.l1.json` 裡的 `l1.metrics`
- [ ] 真實 workflow 每個 job 只輸出一支影片，規格符合派工單（寬高完全一致、fps 誤差 ±1、格數誤差 ±2）
- [ ] flicker 的 re-roll，以及 static 的第 3 次 attempt，`patch_args` 裡真的出現 `steps: 8`，而且能正常生成（static 的第 2 次只加 motion phrase，沒有 steps 8）
- [ ] 重現一支片：`sha256sum` 和 `output_sha256` 是否一致？把結果記下來（GPU 是否決定性未知）
- [ ] `encoded/*.mp4` 用 ffprobe 檢查是 `h264,1080,1920,30/1` 加上 `aac`（靜音）
- [ ] doctor 沒有出現 `judge.latency` WARN；整晚的生成和判讀總時間（report 的 `Generation / judge time`）在預算內，每次生成的時間看 attempts.jsonl 的 `gen_elapsed_s`
- [ ] 在 cron 的最小 PATH 下真的有觸發，並寫出 `cron.log`；一張大約 6 支的派工單無人值守跑完，早上有報告（DoD）
- [ ] 第二天 judge 的 digest 沒有變（Ollama 沒有自動更新），下一晚沒有被 `judge build … differs` 擋下

---

## 跟 docs/plan.md 不一致的地方（以這份為準）

- plan §6（`docs/plan.md:463-464`）說簽核後的 `thresholds.yaml` 要 commit → 這份刻意不 commit：工作站上的本機 commit 會和 origin/main 分岔，`git pull --no-rebase --ff-only` 就會拒絕（`fatal: Not possible to fast-forward`，exit 128）。要改成 commit 請 owner 決定。
