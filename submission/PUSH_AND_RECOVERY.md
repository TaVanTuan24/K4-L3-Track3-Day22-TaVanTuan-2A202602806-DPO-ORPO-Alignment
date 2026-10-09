# Bàn giao Day 22 — thực hiện Git và phục hồi trong Google Colab

Đã huấn luyện và đánh giá thật NB0–NB4. Trọng số lớn nằm trong Drive recovery-models và recovery-adapters, ngoài ZIP bằng chứng.

## Push an toàn từ runtime Colab hiện tại

Mở terminal Google Colab, chuyển vào workspace /content/K4-L3-Track3-Day22-TaVanTuan-2A202602806-DPO-ORPO-Alignment. Đăng nhập GitHub bằng GitHub CLI qua trình duyệt:

    gh auth login --hostname github.com --web --git-protocol https
    gh auth setup-git
    git push origin main
    git rev-parse HEAD
    git ls-remote origin refs/heads/main

Đối chiếu SHA local với SHA main remote. Không dán PAT vào notebook/log. Không force push. Nếu remote đã có commit mới, push bình thường sẽ bị từ chối; giữ lại commit của mình và hợp nhất/rebase với cập nhật remote trước khi push.

## Nếu runtime đã reset

Mọi lệnh vẫn chạy trong Colab. Clone lại repository, giải nén ZIP vào đúng /content/K4-L3-Track3-Day22-TaVanTuan-2A202602806-DPO-ORPO-Alignment và cài môi trường theo setup-colab.sh. Khôi phục models/sft-merged từ Drive/recovery-models/sft-merged và adapters/sft-mini, adapters/dpo từ Drive/recovery-adapters. Đối chiếu SHA-256 với sft_reference_backup.json và submission/execution/checkpoint_restore.json; không tạo checkpoint thay thế.

ZIP chứa handoff/Day22-submission.bundle để giữ nguyên commit chưa push. Sau khi clone và đặt file bundle vào /content, có thể đưa commit vào nhánh phục hồi:

    git fetch /content/Day22-submission.bundle main:refs/remotes/backup/day22
    git switch -c codex/day22-recovered refs/remotes/backup/day22

Bản bundle cần commit nền công khai đã có trong repository. Nhánh phục hồi giữ nguyên kết quả; kiểm tra diff, hợp nhất với main mới nhất và xác minh trước khi push. File handoff/GIT-HANDOFF.json trong ZIP ghi SHA commit, trạng thái push và checksum bundle. Không chạy lại SFT/DPO khi checkpoint gốc còn hợp lệ.

## Bằng chứng

5 notebook thực thi, 4 PNG, preference parquet/stats, DPO config/metrics/split, JSONL generation, JSON judge và Reflection được commit. Log test có 54 passed, không skip; verify exit 0. Exit-code gốc make pipeline không có trong snapshot phục hồi; các notebook có đủ 32 ô mã thực thi không error. Không khai báo một exit-code chưa được lưu.
