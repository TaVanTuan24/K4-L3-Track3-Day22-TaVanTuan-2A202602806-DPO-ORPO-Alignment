# Day 22 — DPO/ORPO Alignment: chạy toàn bộ bài bắt buộc trong **một cell** trên Google Colab

File này chứa **cell duy nhất** chạy trọn vẹn phần bắt buộc (NB0 → NB4 + `make test` + `make verify`)
trong Google Colab, kèm đường dẫn dự phòng khi mất phiên.

> ⚠️ **Trạng thái thực tế:** cell này được soạn sẵn nhưng **chưa được tôi thực thi** (xem
> `submission/REFLECTION.md` và báo cáo cuối). Không có số liệu nào trong repo là kết quả
> huấn luyện giả. Mọi số liệu phải do chính cell này sinh ra trong runtime Colab của bạn.

---

## 0. Chuẩn bị (làm 1 lần)

1. Mở <https://colab.research.google.com/> → **Runtime → Change runtime type → T4 GPU → Save**.

> ⚠️ **Đừng dán vào notebook mẫu read-only** như `colab.research.google.com/notebooks/gpu.ipynb`
> (Colab hiện băng-rôn *"Không thể lưu các thay đổi"* và cell có thể không thực thi được — đã kiểm chứng
> thực tế). Hãy **tạo notebook mới** (File → New notebook) hoặc mở notebook của repo rồi **Copy to Drive**
> — hai cách đó cho notebook writable và chạy được.
2. *(khuyến nghị)* Dán **Cell 0** (mount Google Drive) vào một cell Python rồi chạy — để Cell A tự backup, tránh mất 2 giờ công nếu Colab ngắt.
3. Tạo/dùng một cell mới, dán **Cell A** dưới đây vào và chạy (`Ctrl+Enter`).
4. Cell A tự: kiểm tra GPU → clone repo → vá `my_dpo_loss` (NB0) + in 3 cặp mẫu (NB2) →
   cài dependency → setup (jupytext + `.env`) → `make colab` → `make smoke` → NB0 → NB1 (SFT) →
   NB2 (preference data) → NB3 (DPO) → NB4 (judge + win-rate) → `make test` → `make verify` →
   backup Drive → đóng gói bằng chứng và **tải ZIP về máy**.

Thời gian dự kiến trên T4: **~90–150 phút** (SFT ~40–60 phút, DPO ~40–60 phút, NB4 judge ~20–40 phút).
Nếu gặp CUDA OOM, script **tự động chạy lại stage đó với `MAX_LEN=512`** và ghi lại vào log.

---

## Cell 0 — (tuỳ chọn, KHUYẾN NGHỊ) mount Google Drive trước

Cell A dài ~2 giờ. Nếu Colab ngắt giữa chừng, `/content` bị xoá sạch — mất cả `models/sft-merged`
(phải chạy lại SFT). Mount Drive trước để Cell A **tự backup** và bạn khôi phục được chỉ phần còn thiếu.

```python
# Cell 0 — chạy trong 1 cell PYTHON riêng (không phải %%bash), bấm Allow khi Google hỏi
from google.colab import drive
drive.mount('/content/drive')
```

Sau khi mount, Cell A tự `rsync` repo + `models/sft-merged` vào
`/content/drive/MyDrive/day22_backup/` ngay sau khi pipeline xong. Nếu bạn bỏ qua Cell 0,
Cell A vẫn chạy bình thường (chỉ in "Drive chưa mount").

---

## Cell A — chạy tất cả (dán nguyên khối, kể cả dòng `%%bash`)

```bash
%%bash
# =============================================================================
# VinUni AI20K — Track 3 — Day 22: DPO/ORPO Alignment
# Chạy toàn bộ phần BẮT BUỘC (NB0..NB4 + make test + make verify) trên Colab GPU.
# Runtime: Runtime > Change runtime type > T4 GPU.
# Mọi tính toán (pip, huấn luyện, đánh giá, pytest) chạy TRONG runtime Colab này.
# =============================================================================
set -uo pipefail
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1

REPO_URL="https://github.com/TaVanTuan24/K4-L3-Track3-Day22-TaVanTuan-2A202602806-DPO-ORPO-Alignment.git"
R=/content/day22
LOG=/content/day22_run.log
EVID=/content/day22_evidence
MAXLEN="${MAXLEN:-}"            # rỗng = theo tier (T4 768 / BIGGPU 1024); đặt MAXLEN=512 để giảm VRAM
[ -f "$LOG" ] || : > "$LOG"   # giữ log của lần chạy trước khi resume, không xoá trắng
say() { echo -e "\n\n=== $* ===" ; echo -e "=== $* ===" >> "$LOG"; }

# ---------------------------------------------------------------- [0] GPU ----
say "[0/8] Kiểm tra GPU"
nvidia-smi -L 2>&1 | tee -a "$LOG"
python - <<'PY' 2>&1 | tee -a "$LOG"
import torch
assert torch.cuda.is_available(), "CUDA khả dụng = False — hãy đặt Runtime > Change runtime type > T4 GPU"
gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
print(f"GPU : {torch.cuda.get_device_name(0)}  ({gb:.1f} GB)")
print(f"Tier: {'BIGGPU' if gb >= 22 else 'T4'}   (setup-colab.sh dùng cùng ngưỡng 22 GB)")
print(f"torch: {torch.__version__}")
PY
# Cổng chặn thật là `assert torch.cuda.is_available()` trong khối python ngay trên;
# PIPESTATUS[0] ở đây là của pipeline python đó (không phải của nvidia-smi).
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  echo "!! KHÔNG có GPU khả dụng. Runtime > Change runtime type > T4 GPU, rồi chạy lại cell này."
  exit 1
fi
# Chỉ ép MAX_LEN khi bạn tự đặt MAXLEN=... (mặc định: để tier quyết định 768 cho T4 / 1024 cho BIGGPU)
if [ -n "${MAXLEN:-}" ]; then export MAX_LEN="$MAXLEN"; fi

# -------------------------------------------------------------- [1] clone ----
say "[1/8] Clone repository"
if [ ! -d "$R/.git" ]; then
  git clone --depth 1 "$REPO_URL" "$R" 2>&1 | tee -a "$LOG"
fi
cd "$R"
git log --oneline -1 | tee -a "$LOG"
git rev-parse HEAD | tee -a "$LOG"

# Backup tuỳ chọn lên Google Drive (chỉ hoạt động nếu bạn đã chạy Cell 0 để mount Drive).
# Đây là lưới an toàn cho lần chạy ~2 giờ: nếu Colab ngắt ở NB4, bạn khôi phục
# models/sft-merged + adapters/dpo từ Drive rồi chỉ chạy lại `make eval`.
DRIVE_DIR=/content/drive/MyDrive/day22_backup
BACKUP_WEIGHTS="${BACKUP_WEIGHTS:-1}"   # =0 để KHÔNG backup models/sft-merged (~8 GB, Drive free 15 GB)
backup() {
  if [ ! -d /content/drive/MyDrive ]; then echo "[backup] Drive chưa mount — bỏ qua"; return 0; fi
  mkdir -p "$DRIVE_DIR"
  # DriveFS không hỗ trợ giữ permission/owner/group -> KHÔNG dùng `rsync -a` (nó ngầm -p -o -g).
  # Mặc định chỉ backup artifact nhỏ; weights chỉ khi BACKUP_WEIGHTS=1 (models/sft-merged ~8 GB,
  # Drive miễn phí chỉ 15 GB).
  local EX="--exclude .git --exclude gguf* --exclude __pycache__ --exclude *-checkpoints"
  if [ "$BACKUP_WEIGHTS" = "1" ]; then
    rsync -rlt --no-perms --no-owner --no-group $EX ./ "$DRIVE_DIR/repo/" \
      && echo "[backup] repo + weights -> $DRIVE_DIR/repo ($(du -sh "$DRIVE_DIR/repo" 2>/dev/null | cut -f1))" | tee -a "$LOG"
  else
    rsync -rlt --no-perms --no-owner --no-group $EX --exclude 'models' ./ "$DRIVE_DIR/repo/" \
      && echo "[backup] repo (KHÔNG gồm models/) -> $DRIVE_DIR/repo ($(du -sh "$DRIVE_DIR/repo" 2>/dev/null | cut -f1))" | tee -a "$LOG"
    echo "[backup] weights không backup (đặt BACKUP_WEIGHTS=1 ở đầu cell nếu muốn). Muốn chỉ khôi phục models/sft-merged thì chạy lại \`make sft\` sau khi restore." | tee -a "$LOG"
  fi
}

# ------------------------------------------------- [2] vá NB0 + NB2 (tối thiểu) ----
say "[2/8] Vá notebooks/00 (my_dpo_loss) và notebooks/02 (3 cặp mẫu)"
python - <<'PY' 2>&1 | tee -a "$LOG"
from pathlib import Path

# ---- NB0: cài my_dpo_loss (bản gốc còn `return None`) ----------------------
p = Path("notebooks/00_dpo_loss_from_scratch.py")
s = p.read_text(encoding="utf-8")
old = '    """pc/pr: policy log-prob chosen/rejected; rc/rr: reference. Trả về loss trung bình."""\n' \
      '    # TODO: viết bằng torch.nn.functional.logsigmoid\n' \
      '    return None'
new = ('    """pc/pr: policy log-prob chosen/rejected; rc/rr: reference. Trả về loss trung bình.\n'
       '\n'
       '    L = -mean( logsigmoid( beta * [ (pc - rc) - (pr - rr) ] ) )\n'
       '\n'
       '    beta * (pc - rc) là reward ngầm định của câu được chọn, beta * (pr - rr) là\n'
       '    của câu bị loại; loss chỉ phụ thuộc *hiệu* của hai reward đó (margin).\n'
       '    """\n'
       '    chosen_reward = beta * (torch.as_tensor(pc) - torch.as_tensor(rc))\n'
       '    rejected_reward = beta * (torch.as_tensor(pr) - torch.as_tensor(rr))\n'
       '    return -torch.nn.functional.logsigmoid(chosen_reward - rejected_reward).mean()')
if "logsigmoid(chosen_reward - rejected_reward)" in s:
    print("NB0: đã vá trước đó, bỏ qua")
elif old in s:
    p.write_text(s.replace(old, new, 1), encoding="utf-8")
    print("NB0: đã cài my_dpo_loss")
else:
    print("NB0: !! KHÔNG tìm thấy mẫu cần thay — kiểm tra tay notebooks/00_dpo_loss_from_scratch.py")

# ---- NB2: in 3 cặp mẫu thật (rubric yêu cầu) ------------------------------
p2 = Path("notebooks/02_preference_data.py")
s2 = p2.read_text(encoding="utf-8")
anchor = ('def n_tokens(text: str) -> int:\n'
          '    return len(tokenizer(text, add_special_tokens=False)["input_ids"])')
block = ('def n_tokens(text: str) -> int:\n'
         '    """Số token của một đoạn text theo tokenizer của mô hình gốc."""\n'
         '    return len(tokenizer(text, add_special_tokens=False)["input_ids"])\n'
         '\n'
         '\n'
         'for i in range(3):\n'
         '    row = train_ds[i]\n'
         '    prompt = row["prompt"][0]["content"]\n'
         '    chosen = row["chosen"][0]["content"]\n'
         '    rejected = row["rejected"][0]["content"]\n'
         '    print(f"\\n{\'=\' * 78}\\nCẶP {i + 1}\\n[prompt] {prompt[:300]}")\n'
         '    print(f"\\n[chosen]   ({n_tokens(chosen)} tok)\\n{chosen[:700]}")\n'
         '    print(f"\\n[rejected] ({n_tokens(rejected)} tok)\\n{rejected[:700]}")')
if "CẶP {i + 1}" in s2:
    print("NB2: đã vá trước đó, bỏ qua")
elif anchor in s2:
    p2.write_text(s2.replace(anchor, block, 1), encoding="utf-8")
    print("NB2: đã thêm phần in 3 cặp mẫu")
else:
    print("NB2: !! KHÔNG tìm thấy mốc n_tokens — kiểm tra tay notebooks/02_preference_data.py")

# Cổng xác nhận: nếu cả nhánh "đã vá" lẫn nhánh "khớp mẫu cũ" đều không chạy, việc vá đã
# thất bại — DỪNG ngay thay vì chạy NB0 với `return None` rồi mới chết ở assert.
assert "logsigmoid(chosen_reward - rejected_reward)" in p.read_text(encoding="utf-8"), \
    "NB0 CHƯA được vá (my_dpo_loss vẫn là return None)"
assert "CẶP {i + 1}" in p2.read_text(encoding="utf-8"), "NB2 CHƯA được vá (thiếu phần in 3 cặp mẫu)"
print("✓ xác nhận vá xong: NB0 có my_dpo_loss, NB2 in 3 cặp mẫu")
PY
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  echo "!! Vá notebook thất bại — DỪNG để không chạy NB0 với return None." | tee -a "$LOG"
  exit 1
fi

# ------------------------------------------------------------ [3] cài đặt ----
say "[3/8] Cài dependency (bỏ CHỈ llama-cpp-python: nó chỉ cần cho NB5 bonus và phải biên dịch)"
# KHÔNG bỏ lm-eval: scripts/verify.py:162 import lm_eval trong `--smoke`, thiếu nó thì
# `make smoke` trả về 1 ngay trước khi huấn luyện.
grep -vE '^llama-cpp-python' requirements.txt > /content/req_core.txt
cat /content/req_core.txt | tee -a "$LOG"
pip install -q -r /content/req_core.txt 2>&1 | tail -n 25 | tee -a "$LOG"
python -c "import unsloth, torch, transformers, trl, peft, datasets, bitsandbytes, accelerate; \
print('unsloth', unsloth.__version__); print('torch', torch.__version__); \
print('transformers', transformers.__version__); print('trl', trl.__version__); \
print('peft', peft.__version__); print('datasets', datasets.__version__)" 2>&1 | tee -a "$LOG"

# --------------------------------------------------------- [4] setup + smoke ----
# KHÔNG gọi `bash setup-colab.sh` ở đây: script đó chạy lại `pip install -q -r requirements.txt`
# ĐẦY ĐỦ, tức là cài lại llama-cpp-python (phải biên dịch, chậm và dễ lỗi) + lm-eval mà bước [3]
# đã cố ý bỏ qua (chúng chỉ cần cho NB5/NB6 bonus). Thay bằng đúng các bước CÒN LẠI của nó:
say "[4/8] Setup Colab (jupytext + .env + thư mục) + make smoke"
TIER=$(python -c "import torch;print('BIGGPU' if torch.cuda.get_device_properties(0).total_memory/1024**3>=22 else 'T4')")
jupytext --to notebook --update notebooks/*.py 2>/dev/null \
  || jupytext --to notebook notebooks/*.py 2>&1 | tail -n 3 | tee -a "$LOG"
ls -1 notebooks/*.ipynb | tee -a "$LOG"
[ -f .env ] || cp .env.example .env
sed -i "s/^COMPUTE_TIER=.*/COMPUTE_TIER=$TIER/" .env
mkdir -p data/pref data/eval adapters models gguf submission/screenshots
echo "[4/8] COMPUTE_TIER=$TIER  MAX_LEN=${MAX_LEN:-<mặc định theo tier>}  (bước pip của setup-colab.sh đã chạy ở [3])" | tee -a "$LOG"
make colab 2>&1 | tail -n 5 | tee -a "$LOG"      # đồng bộ colab/*.ipynb với notebooks đã vá (test_smoke kiểm tra)
make smoke 2>&1 | tee -a "$LOG"
echo "make smoke exit=${PIPESTATUS[0]}" | tee -a "$LOG"

# ------------------------------------------------------------- [5] pipeline ----
# LƯU Ý về marker: repo KHÔNG commit notebooks/*.ipynb (chỉ có .py). Ở [4] jupytext đã
# tạo sẵn các .ipynb nhưng CHƯA execute, và `make nb0` dựa vào việc .ipynb đã tồn tại
# (jupytext --update rồi nbconvert --execute --inplace). Vì vậy KHÔNG dùng
# notebooks/00_*.ipynb làm marker "đã xong" — nó sẽ khiến nb0 bị bỏ qua oan.
# nb0 dùng sentinel riêng (bên dưới); các stage sau dùng artifact thật do GPU sinh ra.

say "[5/8] make pipeline (NB0 -> NB1 -> NB2 -> NB3 -> NB4)"
stage() {   # $1 = make target, $2 = artifact đánh dấu đã xong
  if [ -e "$2" ]; then echo "[skip] make $1 — đã có $2"; return 0; fi
  echo -e "\n--- make $1 ---" | tee -a "$LOG"
  make "$1" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ] && [ "${MAX_LEN:-}" != "512" ]; then
    echo "!! make $1 lỗi (rc=$rc) — thử lại với MAX_LEN=512 (giảm VRAM)" | tee -a "$LOG"
    MAX_LEN=512 make "$1" 2>&1 | tee -a "$LOG"
    rc=${PIPESTATUS[0]}
    [ "$rc" -eq 0 ] && echo "NOTE: $1 thành công khi MAX_LEN=512 (đã ghi vào REFLECTION)" | tee -a "$LOG"
  fi
  return "$rc"
}
# nb0 không dùng .ipynb làm marker (xem ghi chú ở [5]) — dùng sentinel có thật sau khi chạy
if [ -f /content/.nb0_done ]; then
  echo "[skip] make nb0 (sentinel /content/.nb0_done đã có)"
else
  echo -e "\n--- make nb0 ---" | tee -a "$LOG"
  make nb0 2>&1 | tee -a "$LOG"
  if [ "${PIPESTATUS[0]}" -eq 0 ]; then
    touch /content/.nb0_done; echo "nb0: XONG" | tee -a "$LOG"
  else
    echo "!! make nb0 lỗi — xem log; NB0 là CPU-only nên lỗi ở đây thường do thiếu package" | tee -a "$LOG"
  fi
fi
stage sft  models/sft-merged/config.json
# data/pref/*.parquet được .gitignore cố tình cho phép commit (verify.py:200-201 cần chúng),
# nên `[ -e train.parquet ]` có thể đúng ngay từ đầu và NB2 bị BỎ QUA oan — khi đó cả hai
# parquet, stats.json và 02b-pref-length.png trong ZIP là của người khác, không phải của
# lần chạy này. Vì vậy dùng sentinel riêng cho stage data.
if [ -f /content/.data_done ]; then
  echo "[skip] make data (sentinel /content/.data_done đã có)"
else
  rm -f data/pref/train.parquet data/pref/eval.parquet data/pref/stats.json
  echo -e "\n--- make data ---" | tee -a "$LOG"
  make data 2>&1 | tee -a "$LOG"
  if [ "${PIPESTATUS[0]}" -eq 0 ]; then
    touch /content/.data_done; echo "data: XONG" | tee -a "$LOG"
  else
    echo "!! make data lỗi — khôi phục bản cũ (nếu có): git checkout -- data/pref" | tee -a "$LOG"
  fi
fi
stage dpo  adapters/dpo/dpo_metrics.json
stage eval data/eval/judge_summary.json

# ------------------------------------------ backup ngay sau pipeline (nếu có Drive) ----
say "[5b] Backup repo + models/sft-merged lên Drive (bỏ qua nếu chưa mount)"
backup

# ------------------------------------------------------- [6] test + verify ----
say "[6/8] make test"
make test 2>&1 | tee -a "$LOG"; echo "make test exit=${PIPESTATUS[0]}" | tee -a "$LOG"
say "[7/8] make verify (kỳ vọng còn UNEDITED REFLECTION -> chưa PASS ở bước này)"
make verify 2>&1 | tee -a "$LOG"; echo "make verify exit=${PIPESTATUS[0]}" | tee -a "$LOG"

# ------------------------------------------------------ [8] gom bằng chứng ----
say "[8/8] Đóng gói bằng chứng"

# verify.py:61 so `Path(base).resolve() != (REPO/'models/sft-merged').resolve()`.
# NB3 lưu đường dẫn TUYỆT ĐỐI của Colab (/content/day22/...), nên nếu copy adapter về
# checkout khác thì verify báo WRONG REF. Đổi sang đường dẫn TƯƠNG ĐỐI để artifact
# portable (make verify chạy ở repo root nên vẫn khớp) — làm SAU make eval vì NB4
# còn dùng adapter này để sinh câu trả lời.
python - <<'PY' 2>&1 | tee -a "$LOG"
import json, pathlib
p = pathlib.Path("adapters/dpo/adapter_config.json")
if p.exists():
    cfg = json.loads(p.read_text(encoding="utf-8"))
    old = cfg.get("base_model_name_or_path")
    cfg["base_model_name_or_path"] = "models/sft-merged"
    p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"adapter_config.json: base_model_name_or_path {old!r} -> 'models/sft-merged' (portable)")
else:
    print("adapter_config.json: THIẾU (NB3 chưa xong)")
PY

# Bằng chứng được xếp ĐÚNG cây thư mục của repo dưới $EVID/repo/ — khôi phục chỉ cần 1 lệnh:
#     cp -r day22_evidence/repo/.  <repo-của-bạn>/
rm -rf "$EVID"
mkdir -p "$EVID/repo/models/sft-merged" "$EVID/repo/adapters/dpo" "$EVID/repo/adapters/sft-mini" \
         "$EVID/repo/data/pref" "$EVID/repo/data/eval" "$EVID/repo/submission/screenshots" "$EVID/repo/notebooks"
# Môi trường runtime — dùng cho mục A của báo cáo cuối
{ echo "--- nvidia-smi ---"; nvidia-smi
  echo "--- versions ---"
  python -c "import sys,torch,transformers,trl,peft,datasets,unsloth;print('python',sys.version.split()[0]);print('torch',torch.__version__);print('transformers',transformers.__version__);print('trl',trl.__version__);print('peft',peft.__version__);print('datasets',datasets.__version__);print('unsloth',unsloth.__version__)"
  echo "--- repo ---"; pwd; git rev-parse HEAD; echo "MAX_LEN=${MAX_LEN:-<mặc định theo tier>}"; echo "COMPUTE_TIER=${COMPUTE_TIER:-<từ .env>}"
} > "$EVID/env.txt" 2>&1
R2="$EVID/repo"
cp -v models/sft-merged/config.json                                                        "$R2/models/sft-merged/" 2>/dev/null | tee -a "$LOG"
cp -v adapters/dpo/adapter_config.json adapters/dpo/dpo_metrics.json adapters/dpo/split.json "$R2/adapters/dpo/" 2>/dev/null | tee -a "$LOG"
cp -v adapters/sft-mini/adapter_config.json                                                "$R2/adapters/sft-mini/" 2>/dev/null | tee -a "$LOG"
cp -v data/pref/stats.json data/pref/train.parquet data/pref/eval.parquet                  "$R2/data/pref/" 2>/dev/null | tee -a "$LOG"
cp -v data/eval/side_by_side.jsonl data/eval/judge_summary.json                            "$R2/data/eval/" 2>/dev/null | tee -a "$LOG"
cp -v data/eval/judge_results_rm.json data/eval/judge_results_api.json                     "$R2/data/eval/" 2>/dev/null | tee -a "$LOG"
cp -v submission/screenshots/*.png                                                         "$R2/submission/screenshots/" 2>/dev/null | tee -a "$LOG"
cp -v notebooks/*.ipynb                                                                    "$R2/notebooks/" 2>/dev/null | tee -a "$LOG"
cp -v "$LOG" "$EVID/run.log" 2>/dev/null
{ echo; echo "=== bằng chứng đã gom (phải thấy models/sft-merged/config.json) ==="; find "$R2" -type f | sort; } | tee -a "$LOG"

python - <<'PY' 2>&1 | tee -a "$EVID/SUMMARY.txt"
import json, pathlib
def j(p):
    p = pathlib.Path(p)
    return json.loads(p.read_text()) if p.exists() else None
dm, js, st = j("adapters/dpo/dpo_metrics.json"), j("data/eval/judge_summary.json"), j("data/pref/stats.json")
print("================ DAY 22 — SỐ LIỆU THẬT TỪ RUNTIME COLAB ================")
if dm:
    for k in ("compute_tier","base_model","beta","lr","final_train_loss","first_logged_loss",
              "end_reward_gap","eval_reward_gap","eval_reward_accuracy","diagnosis"):
        print(f"dpo.{k:22s} = {dm.get(k)}")
else:
    print("dpo_metrics.json: THIẾU")
if js:
    print(f"judge                = {js.get('judge')}")
    print(f"heldout.n            = {(js.get('heldout') or {}).get('n')}")
    print(f"heldout.win_rate     = {(js.get('heldout') or {}).get('win_rate')}")
    print(f"heldout.ci95         = {(js.get('heldout') or {}).get('win_rate_ci95')}")
    print(f"sanity_accuracy      = {js.get('sanity_accuracy')}")
    print(f"judge_agreement      = {js.get('judge_agreement')}")
else:
    print("judge_summary.json: THIẾU")
print(f"pref.stats           = {st}")
print("=======================================================================")
PY

say "[8b] Backup lần cuối lên Drive (đã gồm adapter_config.json dạng tương đối)"
backup

cd /content
zip -qr /content/day22_evidence.zip day22_evidence
ls -l /content/day22_evidence.zip | tee -a "$LOG"
echo -e "\n\n>>> Đang tải day22_evidence.zip về máy (thư mục Downloads)..."
python -c "from google.colab import files; files.download('/content/day22_evidence.zip')"
echo ">>> XONG. File báo cáo đầy đủ: /content/day22_run.log"
```

---

## Xử lý sự cố thường gặp

| Triệu chứng | Cách xử lý |
|---|---|
| `import unsloth` lỗi sau khi pip install | **Runtime → Restart session**, rồi chạy lại nguyên Cell A. `stage()` bỏ qua mọi stage đã có artifact, chỉ chạy lại từ chỗ hỏng |
| `CUDA out of memory` | Cell A **tự** chạy lại stage đó với `MAX_LEN=512`. Muốn ép từ đầu: đổi dòng `MAXLEN="${MAXLEN:-}"` ở đầu cell thành `MAXLEN=512` |
| `make sft` lỗi ở `load_dataset` | Dataset bị gate/đổi. Ghi traceback vào REFLECTION §8; thử `datasets` khác bằng `SFT_DATASET` nếu config hỗ trợ |
| `make verify` báo `TOO FEW` (< 50 held-out) | NB2 lọc theo `MAX_LEN` nên có thể ra < 50 prompt. **Đừng** giảm `MAX_LEN`. Tăng số cặp eval rồi chỉ chạy lại 3 stage phụ thuộc: `PREF_EVAL=200 make data` → `PREF_EVAL=200 make dpo` → `PREF_EVAL=200 make eval` (KHÔNG chạy lại SFT). `PREF_EVAL` được `lab22/config.py:129` đọc từ env |
| Colab disconnect | Xem mục dưới; chạy lại Cell A, các stage đã có artifact sẽ được bỏ qua |
| `make eval` OOM ở lúc **nạp reward model** (`judge.py` nạp RM >5 GB bằng fp16) | Đây là điểm OOM dễ xảy ra nhất trên T4 15 GB, và retry `MAX_LEN=512` **không** giúp vì không liên quan độ dài chuỗi. Chạy lại **chỉ** NB4 sau khi giải phóng VRAM: `Runtime → Restart session`, khôi phục backup Drive (`cp -r /content/drive/MyDrive/day22_backup/repo/. /content/day22/`), cài lại deps, rồi `cd /content/day22 && make eval`. SFT/DPO đã xong nên không tốn lại 2 giờ |
| Mất `/content` giữa chừng nhưng đã mount Drive | Khôi phục: `cp -r /content/drive/MyDrive/day22_backup/repo/. /content/day22/` rồi chạy lại Cell A — `stage()` bỏ qua mọi stage đã có artifact |

---

## Cell B — sau khi viết xong `submission/REFLECTION.md`

`make verify` chỉ PASS khi `REFLECTION.md` không còn placeholder. Số liệu để điền nằm ở
`day22_evidence/SUMMARY.txt` (bản tóm tắt) và `day22_evidence/repo/data/eval/judge_summary.json`
(bản đầy đủ, có `per_judge` và các nhóm helpfulness/safety).

```bash
%%bash
cd /content/day22
# dán nội dung REFLECTION.md đã viết vào đây (hoặc upload file rồi copy vào đúng chỗ)
# ví dụ: từ file local đã upload lên /content/REFLECTION.md
[ -f /content/REFLECTION.md ] && cp /content/REFLECTION.md submission/REFLECTION.md
make verify 2>&1 | tee /content/day22_verify.log
echo "make verify exit=${PIPESTATUS[0]}"
make test 2>&1 | tee -a /content/day22_verify.log
echo "make test exit=${PIPESTATUS[0]}"
cd /content && zip -qr /content/day22_verify_bundle.zip day22 && ls -l /content/day22_verify_bundle.zip
python -c "from google.colab import files; files.download('/content/day22_verify_bundle.zip')"
```

---

## Nếu phiên Colab bị ngắt (disconnect)

`/content` bị xoá sạch, nhưng **không phải chạy lại từ đầu**: clone lại repo và chạy lại Cell A.
`stage()` tự bỏ qua mọi stage đã có artifact — nhưng vì `/content` mất, cần khôi phục artifact
từ `day22_evidence.zip` đã tải về:

```bash
%%bash
cd /content
# Upload lại day22_evidence.zip vào /content/ trước (khung Tệp bên trái -> biểu tượng upload)
unzip -o /content/day22_evidence.zip
# PHẢI clone trước rồi mới phủ artifact: evidence/repo/ KHÔNG có .git, nên nếu phủ trước thì
# `git clone` vào thư mục không rỗng sẽ fatal và mất Makefile/lab22/scripts.
git clone --depth 1 "$REPO_URL" /content/day22
cp -r /content/day22_evidence/repo/. /content/day22/
```
Nếu ZIP được lưu trên Google Drive: `from google.colab import drive; drive.mount('/content/drive')`
rồi `unzip -o /content/drive/MyDrive/day22_evidence.zip -d /content/`.

Muốn backup lên Drive ngay trong Cell A, thêm vào đầu cell:
`from google.colab import drive; drive.mount('/content/drive')` rồi
`cp /content/day22_evidence.zip /content/drive/MyDrive/`.

---

## Việc phải làm sau khi có ZIP (không thể tự động hoá)

1. **`submission/REFLECTION.md`**: điền đủ §1–§8 bằng số liệu thật trong ZIP.
   `make verify` bắt buộc §1, §2, §3, §4, §6 không được còn `_Trả lời ở đây._` / `_<...>_`;
   §3 ≥ 100 từ, §6 ≥ 150 từ.
2. Khôi phục bằng chứng vào repo local **bằng một lệnh** (ZIP đã xếp đúng cây thư mục repo):
   ```bash
   # Windows PowerShell: Expand-Archive day22_evidence.zip -DestinationPath .
   unzip -o ~/Downloads/day22_evidence.zip -d /tmp/d22
   cp -r /tmp/d22/day22_evidence/repo/.  <đường-dẫn-repo-của-bạn>/
   ```
   Lệnh này đặt đúng: `models/sft-merged/config.json`, `adapters/dpo/{adapter_config.json,dpo_metrics.json,split.json}`,
   `adapters/sft-mini/adapter_config.json`, `data/pref/{train,eval}.parquet` + `stats.json`,
   `data/eval/{side_by_side.jsonl,judge_summary.json}`, 4 ảnh PNG, và `notebooks/*.ipynb` đã execute.
   **Không** copy `.safetensors` (ZIP đã loại chúng).
3. Commit + push:
   ```bash
   git add -A && git status
   git commit -m "feat(day22): complete Colab DPO training and evaluation"
   git push origin main
   ```
