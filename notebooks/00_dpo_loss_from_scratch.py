# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # NB0 — DPO loss tự cài từ đầu (CPU, ~10 phút)
#
# **Không cần GPU.** Trước khi gọi `DPOTrainer`, bạn tự viết loss và kiểm tra nó
# trên số liệu đồ chơi. Phần này lấy từ lab K3 (tự cài DPO) và là nền để đọc
# đường cong reward ở NB3.
#
# Bạn sẽ thấy:
# 1. Tại bước 0 (mô hình đang học (policy) = reference) loss luôn bằng `log 2 ≈ 0.693`.
# 2. Gradient của DPO bị nhân với `sigmoid(-margin)`: cặp đã phân biệt tốt gần như không còn được học.
# 3. **Likelihood displacement**: loss vẫn giảm khi log-prob của *chosen* giảm, miễn rejected giảm nhanh hơn.
# 4. IPO, RPO, SimPO, ORPO khác DPO ở đâu, trên cùng một bộ số.

# %%
import sys
from pathlib import Path

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "lab22" / "config.py").exists())
sys.path.insert(0, str(ROOT))

import math

import torch

from lab22 import dpo_math as M

torch.manual_seed(0)

# %% [markdown]
# ## 1. Log-prob của một câu trả lời
#
# `log π(y|x) = Σ_t log π(y_t | x, y_<t)`, chỉ cộng trên token của câu trả lời
# (mask = 1), không cộng trên câu hỏi.

# %%
vocab, length = 8, 5
logits = torch.randn(1, length, vocab)
labels = torch.randint(0, vocab, (1, length))
mask = torch.tensor([[0, 0, 1, 1, 1]])  # 2 token prompt, 3 token trả lời
total, mean = M.sequence_logps(logits, labels, mask)
print(f"sum log p = {total.item():.3f}   mean log p = {mean.item():.3f}")

# %% [markdown]
# ## 2. Bài tập: tự viết DPO loss
#
# Công thức (Rafailov et al. 2023):
#
# $$\mathcal{L} = -\log\sigma\Big(\beta\big[(\log\pi_\theta(y_w) - \log\pi_{ref}(y_w)) - (\log\pi_\theta(y_l) - \log\pi_{ref}(y_l))\big]\Big)$$
#
# Điền hàm dưới đây. Ô kiểm tra sẽ so với bản tham chiếu trong `lab22/dpo_math.py`.


# %%
def my_dpo_loss(pc, pr, rc, rr, beta=0.1):
    """pc/pr: policy log-prob chosen/rejected; rc/rr: reference. Trả về loss trung bình.

    L = -mean( logsigmoid( β · [ (pc − rc) − (pr − rr) ] ) )

    `beta * (pc - rc)` là reward ngầm định của câu được chọn, `beta * (pr - rr)` là
    của câu bị loại; loss chỉ phụ thuộc *hiệu* của hai reward đó (margin).
    """
    chosen_reward = beta * (torch.as_tensor(pc) - torch.as_tensor(rc))
    rejected_reward = beta * (torch.as_tensor(pr) - torch.as_tensor(rr))
    return -torch.nn.functional.logsigmoid(chosen_reward - rejected_reward).mean()


# %%
pc, pr = torch.tensor([-12.0, -30.0]), torch.tensor([-15.0, -28.0])
rc, rr = torch.tensor([-13.0, -29.0]), torch.tensor([-14.0, -29.0])
ref_loss, _, _ = M.dpo_loss(pc, pr, rc, rr, beta=0.1)
mine = my_dpo_loss(pc, pr, rc, rr, beta=0.1)
if mine is None:
    print(f"Chưa cài my_dpo_loss. Đáp số tham chiếu: {ref_loss.item():.4f}")
else:
    assert torch.allclose(torch.as_tensor(mine), ref_loss, atol=1e-6), (mine, ref_loss)
    print(f"✓ Khớp tham chiếu: {ref_loss.item():.4f}")

# %% [markdown]
# ### 2b. Kiểm tra đầy đủ `my_dpo_loss`
#
# Một hàm loss dùng được phải thoả cả bốn điều, không chỉ khớp một bộ số:
# (a) khớp công thức đóng trên nhiều bộ ngẫu nhiên và không sinh NaN/Inf;
# (b) bằng `log 2` khi policy trùng reference; (c) gradient hữu hạn và **truyền được**
# ngược về log-prob của policy (không bị `detach()` làm đứt);
# (d) dấu gradient đúng: đẩy `chosen` lên thì loss giảm, đẩy `rejected` lên thì loss tăng.

# %%
# (a) Khớp công thức đóng trên 200 bộ số ngẫu nhiên, không NaN/Inf.
torch.manual_seed(0)
for _ in range(200):
    a, b = torch.randn(4) * 10 - 20, torch.randn(4) * 10 - 20
    c, d = torch.randn(4) * 10 - 20, torch.randn(4) * 10 - 20
    ref_i, _, _ = M.dpo_loss(a, b, c, d, beta=0.3)
    got_i = my_dpo_loss(a, b, c, d, beta=0.3)
    assert torch.isfinite(got_i) and not torch.isnan(got_i), f"NaN/Inf: {got_i}"
    assert torch.allclose(got_i, ref_i, atol=1e-5), (got_i, ref_i)
print("✓ (a) khớp công thức đóng trên 200 bộ ngẫu nhiên, không NaN/Inf")

# (b) policy = reference ⇒ margin = 0 ⇒ loss = log 2.
same_c = torch.randn(6) * 8 - 25
same_r = same_c - 2
loss_same = my_dpo_loss(same_c, same_r, same_c, same_r)
assert torch.allclose(loss_same, torch.tensor(math.log(2)), atol=1e-6), loss_same.item()
print(f"✓ (b) policy = reference ⇒ loss {loss_same.item():.6f} = log 2")

# (c) + (d) gradient hữu hạn, chảy về policy, và có dấu đúng.
pc_g = torch.tensor([-12.0], requires_grad=True)
pr_g = torch.tensor([-15.0], requires_grad=True)
loss_g = my_dpo_loss(pc_g, pr_g, torch.tensor([-13.0]), torch.tensor([-14.0]), beta=0.1)
loss_g.backward()
assert pc_g.grad is not None and pr_g.grad is not None, "gradient không truyền về policy log-prob"
assert torch.isfinite(pc_g.grad).all() and torch.isfinite(pr_g.grad).all(), "gradient không hữu hạn"
assert pc_g.grad.item() < 0 < pr_g.grad.item(), (pc_g.grad.item(), pr_g.grad.item())
print(f"✓ (c) gradient về policy hữu hạn: dL/dpc = {pc_g.grad.item():+.5f} < 0 < dL/dpr = {pr_g.grad.item():+.5f}")

# %% [markdown]
# ## 3. Bước 0: mô hình đang học (policy) = reference ⇒ loss = log 2
#
# NB3 khởi tạo mô hình đang học (policy) bằng chính mô hình SFT (LoRA mới có trọng số B = 0), nên
# reward ngầm định ban đầu bằng 0 và loss bắt đầu ở 0.693. Nếu log của bạn
# không bắt đầu gần 0.693, reference đang không phải mô hình SFT.

# %%
same = torch.tensor([-20.0, -35.0])
loss0, cr0, rr0 = M.dpo_loss(same, same - 3, same, same - 3)
print(f"loss at init = {loss0.item():.4f}   log 2 = {math.log(2):.4f}   rewards = {cr0.tolist()}, {rr0.tolist()}")

# %% [markdown]
# ## 4. Trọng số gradient = sigmoid(−margin)

# %%
for margin in (-2.0, 0.0, 2.0, 5.0):
    m = torch.tensor(margin, requires_grad=True)
    loss = -torch.nn.functional.logsigmoid(m)
    loss.backward()
    print(f"margin {margin:+.1f}: loss {loss.item():.3f}   |dL/dmargin| {abs(m.grad.item()):.3f}")

# %% [markdown]
# ## 5. Likelihood displacement bằng số
#
# Hai kịch bản đều làm margin tăng 2 nat. Loss giống hệt nhau, nhưng ở kịch
# bản B log-prob của câu *được chọn* lại giảm. DPO không phân biệt được hai
# trường hợp này; chỉ đường cong `rewards/chosen` ở NB3 cho bạn biết.

# %%
ref_c, ref_r = torch.tensor([-20.0]), torch.tensor([-22.0])
scenarios = {
    "A: chosen ↑, rejected ↓": (ref_c + 1, ref_r - 1),
    "B: chosen ↓, rejected ↓↓": (ref_c - 3, ref_r - 5),
}
for name, (pc_, pr_) in scenarios.items():
    loss, cr, rj = M.dpo_loss(pc_, pr_, ref_c, ref_r, beta=1.0)
    print(f"{name:28s} loss {loss.item():.3f}  reward chosen {cr.item():+.1f}  rejected {rj.item():+.1f}")

# %% [markdown]
# **RPO** thêm NLL của câu chosen vào loss: kịch bản B bị phạt vì chosen bị đẩy xuống.

# %%
for name, (pc_, pr_) in scenarios.items():
    nll = -pc_ / 10  # NLL trung bình trên 10 token
    print(f"{name:28s} RPO loss {M.rpo_loss(pc_, pr_, ref_c, ref_r, nll, beta=1.0).item():.3f}")

# %% [markdown]
# ## 6. Bốn biến thể trên cùng một cặp
#
# | Loss | Cần mô hình tham chiếu (reference)? | Chuẩn hoá độ dài? | Ghi chú |
# |---|---|---|---|
# | DPO (sigmoid) | có | không | mức cơ sở (baseline) |
# | IPO | có | có (TRL chia theo số token) | hồi quy margin về 1/(2β), chống quá khớp khi dữ liệu gần như tất định |
# | RPO | có | không | DPO + NLL(chosen), giảm likelihood displacement |
# | SimPO | không | có | log-prob trung bình + margin γ |
# | ORPO | không | có | NLL(chosen) + λ·log-odds-ratio, gộp SFT và sở thích vào một bước |
#
# NB3b huấn luyện thật các biến thể này (TRL `loss_type` và `trl.experimental.orpo`).

# %%
n_tokens_c, n_tokens_r = 40, 120  # chosen ngắn, rejected dài
pc_, pr_ = torch.tensor([-48.0]), torch.tensor([-130.0])
rc_, rr_ = torch.tensor([-50.0]), torch.tensor([-128.0])
avg_c, avg_r = pc_ / n_tokens_c, pr_ / n_tokens_r
print(f"DPO   {M.dpo_loss(pc_, pr_, rc_, rr_)[0].item():.4f}")
print(f"IPO   {M.ipo_loss(pc_, pr_, rc_, rr_, n_tokens_c, n_tokens_r).item():.4f}")
print(f"SimPO {M.simpo_loss(avg_c, avg_r).item():.4f}")
print(f"ORPO  {M.orpo_loss(avg_c, avg_r, -avg_c).item():.4f}")

# %% [markdown]
# **Câu hỏi cho REFLECTION §3:** tổng log-prob của câu dài luôn âm hơn câu ngắn.
# Vì sao điều đó khiến DPO gốc dễ thiên vị độ dài, và SimPO/ORPO xử lý bằng cách nào?
# Gợi ý: NB2 in ra tỉ lệ cặp có chosen dài hơn rejected trong dữ liệu tiếng Việt.

# %% [markdown]
# ## 7. Trả lời hai câu hỏi của NB0
#
# ### 7.1 Vì sao margin tăng được trong khi log-xác suất của câu `chosen` giảm?
#
# Loss chỉ phụ thuộc **hiệu** hai reward ngầm định:
# `L = −logsigmoid(β·[(pc − rc) − (pr − rr)])`. Vì vậy đạo hàm theo margin là
# `−β·sigmoid(−β·margin) (margin chưa nhân β)`: mục tiêu **không** có thành phần nào thưởng cho việc nâng
# `log π(chosen)` lên một cách tuyệt đối. Nó chỉ cần margin dương, và margin có thể
# tăng bằng hai con đường hoàn toàn khác nhau. Kết quả thật ở §5:
#
# | Kịch bản | `chosen` | `rejected` | loss |
# |---|---:|---:|---:|
# | A: chosen ↑, rejected ↓ | **+1.0** | −1.0 | 0.127 |
# | B: chosen ↓, rejected ↓↓ | **−3.0** | −5.0 | 0.127 |
#
# Hai kịch bản cho **cùng một loss 0.127**, dù ở B xác suất của câu được chọn *giảm*.
# DPO hoàn toàn không phân biệt được A với B. Trong thực tế đường `rejected` thường
# rơi nhanh hơn vì gradient âm tác động lên mọi token của câu bị loại, nên câu `chosen`
# có thể bị kéo xuống theo — hiện tượng **dịch chuyển xác suất** (likelihood displacement,
# Razin et al. 2024). Đó là lý do bắt buộc phải vẽ **riêng** `rewards/chosen`, chứ chỉ
# nhìn margin tăng là không đủ để kết luận mô hình tốt lên. RPO sửa bằng cách cộng thêm
# NLL của chosen: ở §5, RPO cho A = 2.027 < B = 2.427, tức nó **phạt** đúng kịch bản B.
#
# ### 7.2 Vì sao DPO gốc có thể thiên vị độ dài, và SimPO/ORPO xử lý thế nào?
#
# `log π(y|x) = Σ_t log π(y_t | x, y_<t)` là tổng theo token. Khi kéo dài
# cùng một chuỗi, tổng log-prob giảm; giữa hai chuỗi khác nhau, độ dài không tự
# quyết định thứ tự vì xác suất từng token cũng khác. DPO dùng log-ratio với
# reference nên không tất yếu thưởng cho câu dài. Tuy nhiên, nhãn chosen tương
# quan với độ dài có thể khiến mô hình học đặc điểm này thay cho chất lượng.
# Một cách giảm ảnh hưởng của tổng theo token là chuẩn hoá theo độ dài:
# IPO chia mỗi log-ratio theo số token, SimPO dùng log-prob trung bình và margin γ
# không cần reference; ORPO dùng log-odds của log-prob trung bình cộng NLL chosen.
# Ở §6, cùng cặp chosen 40 token / rejected 120 token cho DPO 0.5130,
# SimPO 1.1256 và ORPO 1.2783. Đây là các thang loss khác nhau, không phải
# bảng xếp hạng chất lượng. β vẫn có ý nghĩa trong DPO gốc, nhưng tổng và trung
# bình log-ratio có thang đo khác nhau; không dùng cùng β để khẳng định tương đương.
