# CLAUDE.md: ngữ cảnh dự án cho phiên làm việc tiếp theo

File này tóm tắt toàn bộ bối cảnh để một phiên Claude mới làm tiếp mà không cần đọc lại lịch sử
chat. Số liệu đầy đủ và lập luận chi tiết nằm ở `docs/report.md` (mục 1–15, tiếng Việt) và
`docs/theory.md`. Cập nhật lần cuối: 2026-11-18, sau wave8 (report Mục 18).

## 0. Quy ước làm việc với người dùng

- **Ngôn ngữ.** Trả lời người dùng bằng tiếng Việt. `docs/report.md` viết tiếng Việt. Code,
  comment, commit message, README và các đoạn LaTeX viết tiếng Anh.
- **Git.**
  - Repo `kuduck192/mpl`, nhánh làm việc `claude/lucid-meitner-0yaeoo`.
  - Commit và push sau mỗi bước có kết quả.
  - Không tạo PR nếu người dùng không yêu cầu.
  - Dòng attribution cuối commit lấy theo system reminder của phiên hiện tại.
- **Bảo mật.**
  - Không bao giờ commit Kaggle token, Kaggle username hay email của người dùng.
  - Username nằm trong `.kaggle_user` (gitignored), token trong `~/.kaggle/access_token`.
  - Khi in output của Kaggle CLI, che username bằng `sed "s/$(cat .kaggle_user)/<user>/g"`.
  - Kernel để private.
  - Người dùng đã dán 3 token Kaggle vào chat ở các phiên trước; nhắc họ thu hồi.
  - Container là tạm thời: phiên mới sẽ không có token, nên phải xin lại (tốt nhất qua cơ chế
    credential của môi trường, không qua chat).
- **Tiêu chuẩn người dùng mong đợi.**
  - So sánh công bằng: ghép cặp theo seed, t-test cặp + Wilcoxon + Holm, baseline được tinh chỉnh
    cùng ngân sách.
  - Báo cáo trung thực cả kết quả âm, không tune cho tới khi ra số đẹp.
  - Kiểm tra tính đúng của cài đặt (gradient check, sai phân hữu hạn) trước khi kết luận.
  - Người dùng thích hình kiểu "pseudo-label accuracy vs student accuracy" và log chi tiết.
- **Hình.**
  - Theo style của `analysis/figures.py`: `SERIES = #2a78d6, #eb6834, #1baf7a, ...`; mực
    `#0b0b0b` / `#52514e`; lưới `#e4e3df`.
  - Scatter dùng tối đa 3 màu đầu: bộ ba này đã qua validator dataviz (all-pairs). Màu aqua dưới
    3:1 contrast nên phải có nhãn trực tiếp.
  - Một trục y mỗi panel.

## 1. Mục tiêu nghiên cứu (yêu cầu gốc)

- Phát triển pseudo-labeling bilevel trust-region dạng pessimistic (PBPL) và expected (EBPL),
  chứng minh được là không suy biến (khác MPL), và vượt các baseline SSL truyền thống.
- Thực nghiệm trên FashionMNIST, chạy trên Kaggle (GPU T4×2).
- Có thể gắn với FixMatch, thiết kế một hoặc hai mạng.

## 2. Phương pháp (tóm tắt)

- **Teacher φ là mô hình triển khai.** Loss của nó là CE trên nhãn + λ_u·FixMatch + λ_m·L_meta,
  với:
  - L_meta = mean_i ⟨t_i, sg(c_i / RMS chạy)⟩;
  - t = softmax pseudo-label của teacher;
  - c_i = J_f(x_i)·v, với v = ∇L_l(θ' + ε) và θ' là student sau một bước SGD trên t.
- **Quy tắc feedback (`--feedback`).**
  - `opt` (MPL): ε = 0.
  - `pess` (PBPL): adversary được giữ qua các vòng. Mỗi vòng: một bước ascent dài α·r trên L_l,
    K bước co về fiber (giảm KL tới dự đoán của anchor, bước Rayleigh 2D/‖∇D‖²), rồi chiếu vào
    ‖ε‖ ≤ r.
  - `exp` (EBPL): hướng Gauss chuẩn hóa về bán kính r, rồi cùng các bước co về fiber.
  - `sam`: ε = r·∇L_l/‖∇L_l‖, đặt lại mỗi vòng. Đây là đối chứng "SAM-feedback".
- **Hai mạng (`--bilevel two`).** Student held-out chỉ học trên soft pseudo-label của teacher và
  không bao giờ thấy ảnh có nhãn (`--stu_excl_lab 1`). Đây là sửa lỗi quan trọng phát hiện ở
  pilot: nếu không held-out, feedback triệt tiêu và fiber bị ghim tại X_l.
- **Một mạng (`--bilevel self`).** Student là bản nhìn trước một bước của chính teacher.
- **Lý thuyết** (`docs/theory.md`, `docs/sections_trbpl.tex`).
  - Prop. A cho khai triển V_r = L_l + r‖Π∇L_l‖ và E_s = L_l + (s²/2)tr(ΠHΠ).
  - N2 (tại cặp supervised-only):
    - (a) opt suy biến;
    - (b) pess không suy biến, tín hiệu bậc r;
    - (c) dạng phạt với γ < 1/ℓ suy biến;
    - (d) exp không suy biến, bậc s².
  - N3: pess = MPL + r·∇‖Π∇L_l‖.
  - Remark held-out: nếu X_l ⊂ X thì V_r = E_s = L_l.
- **Kiểm chứng lý thuyết.**
  - Toy MLP (`checks/check_nondegeneracy.py`).
  - CNN tĩnh (`checks/degeneracy_static.py`): pess lớn hơn opt, trung vị 134 lần.
  - Huấn luyện một mạng: feedback opt thô co 12–39 lần; pess ở vòng 20k lớn gấp 12–65 lần.

## 3. Mã nguồn và cách chạy

| đường dẫn | vai trò |
|---|---|
| `ssl_fmnist.py` | toàn bộ phương pháp trong một file (gửi được lên Kaggle) |
| `sweeps/make_sweeps.py`, `sweeps/*.json` | sinh sweep theo pha |
| `kaggle_tools/kaggle_run.py` | `push` / `status` / `fetch` / `logs`; mỗi GPU chạy một job, hai kernel cùng lúc |
| `kaggle_tools/wait_kernels.sh` | chờ kernel tới trạng thái kết thúc |
| `analysis/tenseed_table.py` | bảng 10 seed; `--tex` in dòng LaTeX, `--ref_avg fixmatch,fixmatchft` lấy mốc trung bình |
| `analysis/finetune_table.py` | teacher / student EMA và sau finetune; `--tex`, `--ref2`, `--cfg` |
| `analysis/wave8_table.py`, `analysis/wave8_figure.py` | wave8: so cặp hai họ, chẩn đoán pseudo-label, quỹ đạo (`--traj`), mốc hai run (`--ref_avg`), `--metric test` |
| `analysis/fm_meta_grid.py` | lưới {chỉ nhãn, batch không nhãn qua BN, FixMatch} × {không meta, MPL, PBPL, EBPL}; `--metric test_ema` / `unl_ema` / `test` (teacher live) |
| `analysis/{main_table,ablation_table,feedback_dynamics,pool_diag,figures}.py` | các bảng và hình khác |
| `checks/` | gradient check, chẩn đoán suy biến, Figure 1, ba toy Two Moons; `toy_two_moons_tr.py` và `tr_fiber.py` là toy với V_r / E_s (commit `0baa285`, đẩy từ một phiên khác, chạy qua `slurm_tools/`) |
| `docs/report.md` | báo cáo đầy đủ |
| `docs/theory.md` | lý thuyết |
| `docs/sections_trbpl.tex` | đoạn bản thảo |
| `docs/tables_tenseed.tex` | dòng LaTeX của bảng 10 seed và bảng finetune |
| `results/fmnist-ssl-*/` | output Kaggle; mỗi run là `results/<name>.json` gồm `args`, `hist`, `final`, `ft` |
| `results/toy-two-moons/{results,unroll,anchor}.json` | kết quả toy |

- **Tên run** có dạng `<method>_<n_per_class>_s<seed>`; các script phân tích lọc theo tên này.
  Run bị cắt bởi budget guard (`stopped_early`) bị bỏ qua.
- **Cờ đã chọn** (theo validation, seed 0):

  | biến thể | cờ |
  |---|---|
  | PBPL | `--bilevel two --feedback pess --lam_m 0.1 --radius 0.1` |
  | EBPL | `--bilevel two --feedback exp --lam_m 0.3 --radius 3.0` |
  | MPL | `--bilevel two --feedback opt --lam_m 0.1` |
  | FixMatch tinh chỉnh | `--base fixmatch --tau 0.95 --lam_u 2.0` |
  | SAM trên L_l | `--lab_reg sam --radius 0.1` |
  | fiber-SAM | `--lab_reg fibersam --radius 0.1` |
  | tổ hợp có SAM | thêm `--lab_reg sam --lab_radius 0.1` |
  | finetune cuối run | `--ft_iters 1000` (lưới lr ∈ {0.001, 0.003, 0.01} × BN {frozen, train}, chọn theo val, báo cáo EMA) |

  Mỗi run chính thêm `--n_per_class 4 --seed s --iters 20000`.
- **Quy trình Kaggle.**

  ```bash
  python3 sweeps/make_sweeps.py <phase>
  ```

  ```bash
  python3 kaggle_tools/kaggle_run.py push sweeps/<sweep>.json
  ```

  ```bash
  python3 kaggle_tools/kaggle_run.py fetch fmnist-ssl-<sweep>
  ```

  Nếu kernel bị cắt giữa chừng, tạo sweep cho phần còn thiếu:

  ```bash
  python3 sweeps/make_sweeps.py remainder sweeps/<sweep>.json <new-slug> <hours> results/fmnist-ssl-<sweep>
  ```

- **Bảng chính.**

  ```bash
  python3 analysis/tenseed_table.py results/fmnist-ssl-* --tex
  ```

  ```bash
  python3 analysis/finetune_table.py results/fmnist-ssl-* --tex
  ```

- **Môi trường.**
  - CPU có torch, numpy, matplotlib; không có sklearn.
  - Kaggle CLI phải cài lại nếu container mới.
  - FashionMNIST: `--data_dir` trỏ tới thư mục idx, hoặc để torchvision tự tải.
  - Smoke test chạy trên CPU với `--device cpu --iters 30 --bl 8 --mu 2 --eval_every 15 --eval_max 400`.

## 4. Kết quả FashionMNIST (40 nhãn, seed 1–10, teacher EMA ở vòng cuối, so cặp với FixMatch)

Holm tính qua 12 phép so sánh; p ghi theo dạng t-test / Wilcoxon / Holm.

| biến thể | accuracy | − FixMatch, thắng, p |
|---|---|---|
| FixMatch | 72.75 ± 3.76 | — |
| FixMatch tinh chỉnh (λ_u = 2) | 73.71 ± 3.66 | +0.96, 6/10, 0.078 / 0.113 / 0.78 |
| MPL hai mạng | 73.28 ± 3.96 | +0.54, 5/10, 0.52 |
| PBPL hai mạng | 73.32 ± 3.10 | +0.58, 7/10, 0.57 |
| EBPL hai mạng | 72.74 ± 2.51 | −0.01, 6/10, 1.00 |
| SAM-feedback (ε đặt lại mỗi bước) | 73.86 ± 3.88 | +1.11, 6/10, 0.23 |
| MPL hai mạng + SAM trên L_l | 74.66 ± 3.20 | +1.91, 8/10, 0.111 / Holm 0.89 |
| PBPL hai mạng + SAM | 73.67 ± 3.47 | +0.92, 6/10, 0.26 |
| EBPL hai mạng + SAM | 73.26 ± 2.54 | +0.52, 6/10, 0.59 |
| MPL một mạng | 74.73 ± 2.97 | +1.98, 7/10, 0.041 / Holm 0.45 |
| PBPL một mạng | 74.25 ± 3.34 | +1.51, 8/10, 0.094 / Holm 0.85 |
| FixMatch + fiber-SAM | 74.28 ± 4.42 | +1.53, 5/10, 0.30 |
| **FixMatch + SAM trên L_l** | **75.79 ± 2.55** | **+3.05, 8/10, 0.014 / 0.020 / 0.17** |

- **Nhiễu chạy lại.** FixMatch chạy lại cùng seed lệch từ −3.5 đến +7.5 điểm, tức khoảng 2.2 điểm
  nhiễu mỗi run. Mã huấn luyện trùng từng bit trên CPU, nên nguồn nhiễu là tính không tất định của
  GPU (`cudnn.benchmark`, AMP). SAM ổn định hơn, khoảng 1.1 điểm.
- **SAM với mốc ít nhiễu hơn.** So với trung bình hai run FixMatch mỗi seed, SAM trên L_l hơn
  +2.65 với Holm p = 0.048. Đây là kết quả duy nhất qua được hiệu chỉnh. Lấy trung bình cả hai
  phía (hai run SAM so với hai run FixMatch): +2.61 ± 2.22, 8/10, p = 0.005.
- **Tổ hợp với SAM làm hại.** So với SAM đơn thuần: MPL −1.1, PBPL −2.1, EBPL −2.5 (EBPL thắng
  2/10, p = 0.011).
- **Finetune cuối run.** Teacher tăng khoảng +0.2, student held-out khoảng +0.4. Student finetune
  tốt nhất (MPL + SAM, 75.08) vẫn kém FixMatch + SAM đã finetune (75.92).
- **Pessimism không hơn optimism.** PBPL − MPL cùng seed: +0.04 (hai mạng), −0.48 (một mạng),
  +0.13 (một mạng, thang meta-gradient chính xác, seed 1–5; khoảng tin cậy 95% từ −3.9 đến +4.2).
- **100 nhãn (5 seed):** PBPL +1.66 (p = 0.41), MPL +1.97 (p = 0.35). Đợt nhân bản 100 nhãn trên
  seed 6–10 chưa bao giờ chạy: `rep-c.json` bị ghi đè ở wave3.
- **Bỏ loss FixMatch (wave7, report Mục 17; `--base fixmatch --lam_u 0` + cờ bilevel).**
  - Teacher EMA: chỉ nhãn 57.66; L_l + MPL / PBPL / EBPL 59.49 / 59.24 / 59.52; FixMatch 72.75.
  - So với FixMatch: −13.3 / −13.5 / −13.2, thua ở 10/10 seed, Holm p < 0.001.
  - So với chỉ nhãn: +1.6 đến +1.9 (n.s.); theo teacher live −2.1 đến −3.6 (n.s. sau Holm).
  - Tức Sup + meta ≈ Sup < Sup + FM ≈ Sup + FM + meta. PBPL / EBPL − MPL ≈ 0 ở cả hai hàng.
  - λ_m giữ như khi chọn có FixMatch, chưa tinh chỉnh lại.
- **Biến thể variational (wave8, report Mục 18, seed 1–5, ρ = 0.1 không tinh chỉnh lại).**
  - Bước q / Y_u của FixMatch đều n.s. so với SAM trên L_l (75.05):
    - SAM trên L_l + λ_u L_u (`fmsamfull`) 74.16, tức −0.9;
    - cộng pseudo-label tại θ + ε* (`fmsamfulladv`) 74.88;
    - SAM-L_l với pseudo-label tại θ + ε*_l (`labsamadv`) 74.36;
    - pseudo-label từ mạng EMA (`fmemapl`) 72.49, tức +1.5 so với FixMatch.
    - Pseudo-label adv / EMA trùng 97.7–99.1% với pseudo-label điểm.
  - SAM trên L_l theo prior:
    - không X_u +4.0 (live; EMA +6.1 bị phóng đại), 5/5;
    - Pseudo-Label +8.6 (5/5, Holm 0.005), đưa PL lên 70.7, ngang FixMatch;
    - FixMatch +4.1;
    - Mean Teacher −0.9; FlexMatch −0.7.
  - Quan sát post-hoc: SAM cho lợi thế sớm (1–2k vòng; sau đó labeled loss tại điểm nhiễu ≤ 0.02).
    Prior có ngưỡng cố định giữ lại lợi thế này; prior dùng cả X_u từ đầu (FlexMatch warm-up, MT)
    xóa nó. Chênh lệch +7.1, p = 0.002, nhưng nhóm được chọn sau khi thấy dữ liệu.
- **Các mốc khác.**
  - Pilot chưa held-out (seed 0): FixMatch 72.5, MPL 71.4, PBPL 68.3.
  - Pool student (Mục 11): hướng feedback có khác nhau nhưng đa dạng về hàm số nhỏ. Không nên theo
    tiếp.
  - Chi phí so với FixMatch: PBPL 3.6×, EBPL 3.5×, MPL 2.5×, SAM 1.25×.

## 5. Toy Two Moons (report Mục 13–15; chạy CPU vài phút đến khoảng 20 phút)

Thiết lập chung: nhiễu 0.1, 1000 điểm không nhãn, 1 hoặc 5 nhãn mỗi lớp, student MLP 2-64-64-2
ReLU chỉ học trên pseudo-label. Mỗi cấu hình có 40 student (8 khởi tạo × 5 lần rút nhãn).

- **Mục 13: pseudo-label tự do = argmin của signal bậc một c = J_f(X_u)∇L_l**
  (`checks/toy_two_moons_meta.py`).
  - Dạng kernel: c_i = Σ_j K(x_i, x_j)(p_j − y_j), một phiếu bầu theo tangent kernel của student.
  - PL acc 0.75 / 0.84, so với label spreading 0.93 / 0.996.
  - Student chép pseudo-label: nằm trên đường chéo, đồng ý 92–98%.
  - Phụ thuộc mạnh vào student ngẫu nhiên: tương quan 0.93.
  - Không quy tắc nào hơn student chỉ học trên điểm có nhãn.
- **Mục 14: unroll K = 1…100 và IFT** (`checks/toy_two_moons_unroll.py`).
  - IFT dùng Gauss–Newton với damping 0.01·λ_max và conjugate gradient. Cài đặt đã kiểm bằng sai
    phân hữu hạn (sai số ≤ 1.6·10⁻⁷), và IFT khớp unroll xuất phát từ θ* khi K ≈ 1/(ηρλ).
  - Ở mọi K, kể cả IFT, pseudo-label tối ưu cho meta objective thấp hơn nhãn thật ở 40/40 student.
  - PL acc khi tối ưu từ t đều (1 nhãn/lớp): 0.835 (K = 1), 0.658 (K = 100), 0.582 (IFT).
  - K hữu hạn: thiên lệch tầm ngắn, ưu tiên cách gán nhãn dễ học (biên gần thẳng).
  - IFT: mục tiêu gần như phẳng. Xuất phát từ nhãn thật giữ được 0.998, từ t đều chỉ 0.582, cả hai
    ở F ≈ 10⁻⁶.
  - Damping càng nhỏ thì argmin càng tệ: 0.82 → 0.64 → 0.59.
- **Mục 15: neo t* = argmin ⟨t, ĝ_K⟩ + λ·KL(t ‖ q), q = teacher FixMatch**
  (`checks/toy_two_moons_anchor.py`).
  - Nghiệm có dạng t ∝ q·exp(−ĝ/λ), giải bằng lặp điểm bất động có damping.
  - λ ≥ 1 giữ PL acc ở mức teacher (±0.007), với số lỗi được sửa ≈ số nhãn bị làm hỏng.
  - λ ≤ 0.1 làm hỏng nhiều hơn sửa. Ví dụ 5 nhãn/lớp, K = 50: hỏng 13.6–14.9%, sửa 0.4–0.6%.
  - Ô dương duy nhất: IFT, λ = 1, 5 nhãn/lớp, +0.2 điểm.
  - Teacher kém (0.628) không được sửa, vì cách gán nhãn sai của nó vẫn đủ để student khớp các
    điểm có nhãn.

## 6. Kết luận khoa học hiện tại

- **Định lý không suy biến đúng ở nơi nó áp dụng, nhưng không chuyển thành accuracy.**
  - Hai mạng không chạm tới điều kiện của định lý.
  - Ở một mạng, chuẩn hóa RMS xóa phần độ lớn, và hướng của MPL vẫn đủ tốt.
  - Ở thang meta-gradient chính xác, bước meta pess lớn gấp 2–6 lần nhưng accuracy vẫn như nhau.
- **Meta objective (CE trên vài điểm có nhãn) là một giả định yếu.** Nó chỉ là điều kiện cần
  ("nhãn đúng phải cho phép student khớp tập có nhãn"), có N_l·C ràng buộc, và lan nhãn theo kernel
  của mạng. Giả định cụm của FixMatch có N_u ràng buộc và lan nhãn dọc đa tạp dữ liệu. Meta signal
  chỉ thấy những lỗi làm hại labeled loss. Pessimism đổi student dùng để chấm điểm, không thêm
  thông tin về nhãn. Wave7 xác nhận trên FashionMNIST: không có FixMatch, meta chỉ ngang học
  thuần trên nhãn và kém FixMatch 13 điểm.
- **Thứ thực sự giúp ở 40 nhãn là SAM trên labeled loss của mạng triển khai**, không cần bilevel.
  Wave8: nó là regularizer cho phần khớp nhãn (giúp cả học thuần trên nhãn), không phải "làm phẳng
  posterior" (cận trên đúng nghĩa không hơn), và lợi ích phụ thuộc prior.
- **Đánh giá đã đưa cho người dùng.** Bỏ tham vọng "PBPL vượt baseline". Giữ dự án như một kết quả
  phân tích / kết quả âm có kiểm soát chặt, cộng với phát hiện về SAM. Hướng nối tiếp tự nhiên là
  sharpness của labeled loss trong SSL; có thể xem FlatMatch (NeurIPS 2023, cross-sharpness), cần
  tự kiểm tra lại tác giả và nội dung.

## 7. Trạng thái và việc còn mở

- **Kaggle: hai tài khoản, quota tính riêng.** Tài khoản A chạy rep-a…rep-f, ablate và wave8
  (rep-m / rep-n); tài khoản B chạy rep-g / rep-h, pool-diag, rep-k / rep-l. `.kaggle_user` và
  `~/.kaggle/access_token` hiện trỏ tài khoản A (từ 2026-11-18). Không ghi username vào repo.
- **Kaggle quota.**
  - Tài khoản B, tuần từ 2026-11-14: đã dùng khoảng 23h (13.0h trước đó + wave7 khoảng 10.1h) trên trần
    khoảng 29–30h. Tài khoản A, cùng tuần: rep-c…rep-f khoảng 14h, cộng wave8 khoảng 5.8h (rep-n
    13:17–16:11, rep-m 13:45–16:16 UTC). Trường `totalTimeAllowed` =
    21600s của API không phải trần thật. Quota làm mới 2026-11-21 00:00 UTC.
  - Người dùng **chưa chọn** đợt Kaggle tiếp theo:
    - (a) `wave6a`: MPL / PBPL / EBPL hai mạng chạy lại có finetune, khoảng 9.7h, `rep-i` / `rep-j`.
      Đã soạn, chưa push.
    - (b) `wave6b`: fiber-SAM × BPL, khoảng 14.8h. Đã soạn. Ít giá trị sau kết quả Mục 12.
    - (c) Đề xuất của tôi: MPL / PBPL **một mạng** + SAM + finetune, khoảng 6.2h. **Chưa viết.**
      Đây là tổ hợp duy nhất chưa thử của hai thành phần tốt nhất.
    - (d) Đề xuất sau wave8 (**chưa viết**, khoảng 3.5h): kiểm định cơ chế "lợi thế sớm + ngưỡng khóa
      lại":
      - SAM chỉ trong 2k vòng đầu, với FixMatch và PL;
      - FixMatch τ thấp ± SAM;
      - FlexMatch bỏ warm-up ± SAM;
      - cộng `plsam` / `supsam` lên 10 seed.
  - Script phân tích đã đăng ký sẵn tên `mplft`, `pbplft`, `ebplft`, `fsmpl`, `fspbpl`, `fsebpl`,
    `fibersamft`.
- **Việc có thể làm tiếp** (chỉ làm khi người dùng yêu cầu):
  - viết technical report ngắn 4–6 trang, đóng khung là "phân tích + kết quả âm + phát hiện SAM";
  - thêm phép đo sharpness cuối run (SAM loss một bước, λ_max bằng power iteration, toàn không gian
    và trên fiber);
  - sàng lọc EBPL một mạng trên seed 0 (chưa từng chạy);
  - thử toy với biến thể pess / exp;
  - đối chứng tách tập có nhãn (một nửa cho CE, một nửa cho meta objective).

## 8. Bẫy đã gặp

- **Kaggle.**
  - API dễ bị rate limit: poll khoảng mỗi 240 giây.
  - Kernel có thể nằm QUEUED khoảng 1h40 trước khi chạy (wave5 rep-g, cả hai kernel wave7).
  - Một kernel T4×2 chạy khoảng 6h cho 25 job loại wave5.
  - Runner bỏ qua job khi ngân sách còn lại dưới 1.1 × `est_min`.
- **Holm trong `tenseed_table.py`** tính trên các phương pháp trong `METHODS` có n ≥ 5. Thêm hàng
  mới sẽ làm đổi p Holm của các hàng cũ.
- **Python < 3.12** không cho dấu backslash trong biểu thức của f-string.
- **matplotlib:** trục log hẹp sinh nhãn minor tick chồng nhau, nên tự đặt 3 tick.
- **IFT:** Hessian thật có thể có độ cong âm khi damping ≤ 10⁻³·λ_max, nên dùng Gauss–Newton.
  Student warm-start chỉ gần dừng (|∇L_u| khoảng 10⁻²).
- **Bài toán neo:** với λ = 0.1, lặp điểm bất động đôi khi dao động; luôn log phần dư.
- **Teacher EMA không ổn định khi không có loss nào trên X_u.** Lớp `EMA` lấy trung bình trọng số
  nhưng chép buffer BN của mạng live. Trong `sup` / `supbn`, test acc của EMA dao động 30–36 điểm
  sau vòng 10k (live chỉ 1.3–1.7). Với các ô này, đọc thêm `fm_meta_grid.py --metric test`.
- **Nhánh có thể có commit từ phiên khác** (ví dụ `0baa285`). Push bị từ chối thì fetch, merge
  rồi push lại, không force-push.
- **Thư mục scratchpad** của phiên cũ không còn. Mọi thứ cần giữ đều đã nằm trong repo.

## 9. Mốc commit

- `8e18e8f`: bảng chính.
- `3fdc29c` / `0e81c50`: nhân bản và FixMatch tinh chỉnh.
- `93d6d0e` / `cdff6c7`: wave4.
- `a2cd2d5` / `9947ce6`: pool student.
- `cb96425`: finetune cuối run và `--lab_radius`.
- `b88db63`: wave5 đủ 10 seed.
- `3d4dc2c`: soạn sẵn wave6a / wave6b.
- `dac5bef`: toy Mục 13.
- `c252c76`: toy Mục 14.
- `bbe7a2f`: toy Mục 15.
- `9894576`: `fm_meta_grid.py`; `21238fe` / `b5df38e`: wave7 (bỏ FixMatch); `d2a6082`: `--metric test`.
- `1a13a2e`: wave8 (biến thể variational: `--lab_reg samfull`, `--pl_src {ema, adv}`, SAM cho `sup`);
  `a1d9892`: khung ELBO + cận sharpness trong `docs/bayes_view.md`; `9742253`: `analysis/wave8_table.py`;
  `ca9deda`: kết quả rep-n; commit sau đó: rep-m, report Mục 18, `analysis/wave8_figure.py`.
