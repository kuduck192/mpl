# Tóm tắt kết quả cuối: PBPL / EBPL trên FashionMNIST

> Tổng hợp từ bản báo cáo hoàn tất (`docs/report.md` trên branch `claude/lucid-meitner-0yaeoo`,
> đã có đủ `rep-a` đến `rep-f`). Accuracy test của mô hình EMA ở vòng lặp cuối; CNN 0.157M tham
> số, 20k vòng, 40 nhãn, seed 1–10, so sánh cặp với FixMatch mặc định trên cùng seed.

## Đánh giá tổng thể

| câu hỏi | kết quả |
|---|---|
| Lý thuyết không suy biến có đúng trên mạng thật không? | **Có.** Kiểm chứng ở toy MLP, CNN tĩnh và trong lúc huấn luyện. |
| Pessimistic / expected có tốt hơn optimistic (MPL) không? | **Không**, ở mọi cấu hình: +0.04 (hai mạng), −0.48 (một mạng), +0.13 (thang chính xác). |
| PBPL / EBPL có vượt FixMatch không? | **Không**: PBPL +0.58 (p = 0.57), EBPL −0.01 (p = 1.00). |
| Ràng buộc fiber, student phụ có cần không? | **Không.** Bỏ fiber (SAM-feedback) hay bỏ student (một mạng) đều không kém hơn. |
| Cái gì thực sự giúp? | **SAM trên labeled loss** (không bilevel): +3.05, p = 0.014, chi phí 1.25× FixMatch. |

**Kết luận:** về mặt accuracy, giả thuyết chính (pessimism giúp pseudo-labeling) là **kết quả âm
tính**. Phần lý thuyết đứng vững, nhưng tính không suy biến không chuyển thành accuracy. Sau hiệu
chỉnh Holm qua 9 phép so sánh, không biến thể nào hơn FixMatch có ý nghĩa ở mức 5%.

## 1. Mọi biến thể trên 10 seed (40 nhãn)

| biến thể | cuối | − FixMatch (cặp) | p (t / Wilcoxon / Holm) | giảm cuối | chi phí |
|---|---|---|---|---|---|
| FixMatch | 72.75 | — | — | 1.84 | 1× |
| FixMatch tinh chỉnh ($\lambda_u=2$) | 73.71 | +0.96, 6/10 | 0.078 / 0.113 / 0.54 | 0.79 | 1× |
| MPL hai mạng (optimistic) | 73.28 | +0.54, 5/10 | 0.52 / 0.49 / 1.00 | 0.63 | 2.5× |
| **PBPL hai mạng (pessimistic)** | 73.32 | +0.58, 7/10 | 0.57 / 0.45 / 1.00 | 1.24 | 3.6× |
| **EBPL hai mạng (expected)** | 72.74 | −0.01, 6/10 | 1.00 / 0.77 / 1.00 | 1.40 | 3.5× |
| SAM-feedback (bỏ fiber) | 73.86 | +1.11, 6/10 | 0.23 / 0.19 / 1.00 | 1.20 | 2.6× |
| MPL một mạng (optimistic) | 74.73 | +1.98, 7/10 | 0.041 / 0.064 / 0.33 | 0.94 | 2.5× |
| PBPL một mạng (pessimistic) | 74.25 | +1.51, 8/10 | 0.094 / 0.105 / 0.57 | 0.73 | 3.5× |
| FixMatch + fiber-SAM | 74.28 | +1.53, 5/10 | 0.30 / 0.49 / 1.00 | 1.03 | 2.1× |
| **FixMatch + SAM trên labeled loss** | **75.79** | **+3.05, 8/10** | **0.014 / 0.020 / 0.13** | 0.59 | 1.25× |

![tenseed](figures/tenseed_paired.png)

- Ưu thế ban đầu của PBPL ở seed 1–5 (+2.02, 5/5, p = 0.019) và của EBPL ở seed 1–3 (+2.36) **không
  được nhân bản**: chúng trùng với nhóm seed mà FixMatch suy giảm mạnh cuối huấn luyện (2.96 điểm,
  so với 0.72 ở seed 6–10).
- Trên các seed **chưa xem lúc quyết định chạy tiếp**, chỉ SAM trên labeled loss giữ được ưu thế
  (seed 4–10: +2.59, 6/7, p = 0.034). MPL một mạng còn +1.08 (p = 0.16), PBPL một mạng +0.99
  (p = 0.39), SAM-feedback +0.12, fiber-SAM +0.28.
- Ở 100 nhãn (5 seed): PBPL +1.66 so với FixMatch (p = 0.41), không hơn MPL (−0.31).

## 2. Các so sánh chỉ khác đúng một thành phần

| so sánh (cùng seed) | câu hỏi | n | vòng cuối |
|---|---|---|---|
| PBPL − MPL | pessimistic vs optimistic, hai mạng | 10 | +0.04, p = 0.97 |
| PBPL − MPL, một mạng | pessimistic vs optimistic, một mạng | 10 | −0.48, p = 0.64 |
| PBPL − MPL, một mạng, thang chính xác | như trên, không chuẩn hóa feedback | 5 | +0.13, p = 0.93 |
| EBPL − MPL | expected vs optimistic | 10 | −0.54, p = 0.60 |
| PBPL − SAM-feedback | có fiber vs không fiber | 10 | −0.53, p = 0.45 |
| MPL một mạng − MPL | một mạng vs hai mạng | 10 | +1.45, p = 0.24 |
| SAM labeled − MPL một mạng | không bilevel vs bilevel tốt nhất | 10 | +1.06, p = 0.17 |

Mọi thành phần đặc trưng của phương pháp đề xuất (pessimism, expected, fiber, student phụ) đều
**không có đóng góp đo được**.

## 3. Phần lý thuyết: đúng, nhưng không thành accuracy

- **CNN thật, chẩn đoán tĩnh** tại cặp supervised-only: feedback pessimistic lớn hơn optimistic
  **trung vị 134 lần** và không giảm khi mạng khớp nhãn.
- **Trong huấn luyện một mạng** (10 seed): feedback optimistic thô co 12–39 lần; pessimistic không
  co, ở vòng 20k lớn gấp 12–65 lần.
- **Thang meta-gradient chính xác** (5 seed): bước meta pessimistic gộp cả quá trình lớn gấp 2–6
  lần, nhưng accuracy như nhau (+0.13, khoảng tin cậy 95% −3.9 đến +4.2).

![degeneracy](figures/degeneracy_static.png)

**Vì sao không thành accuracy:**

1. **Hai mạng không chạm điều kiện của định lý**: student held-out không bao giờ khớp hết nhãn,
   nên feedback MPL không triệt tiêu, không có gì để pessimism sửa.
2. **Một mạng: suy biến có xảy ra nhưng không gây hại.** Chuẩn hóa RMS chạy đưa bước meta về cùng
   cỡ, chỉ còn hướng; hướng optimistic vẫn mang thông tin. Bỏ chuẩn hóa thì có một tín hiệu yếu
   đúng hướng lý thuyết (+1.90, 4/5, p = 0.18), nhưng không đủ.
3. **Lợi ích ở seed 1–5 thực chất là chống suy giảm cuối huấn luyện của FixMatch**, và tinh chỉnh
   $\lambda_u$ hoặc một bước SAM cũng làm được, rẻ hơn nhiều.

## 4. Kết quả phụ

- **FixMatch tinh chỉnh** ($\lambda_u=2$) vượt cấu hình PBPL đã chọn trên seed 0 và ngang PBPL trên
  10 seed (PBPL −0.38), với chi phí bằng 1/3.6.
- **Figure 1 của bản thảo không tái lập được** theo giao thức của review (khoảng pessimistic –
  optimistic chỉ 2–7·10⁻³ nats).
- Chọn checkpoint theo validation: mọi khác biệt của PBPL / EBPL với FixMatch đều về ~0.

## 5. Có thật là "không khả quan"?

**Đúng, với câu hỏi ban đầu.** PBPL / EBPL không vượt baseline mạnh, các thành phần đặc trưng
không có đóng góp, và chi phí gấp 3.5×. Đây không phải vấn đề thiếu seed: 10 seed đủ loại trừ một
hiệu ứng ≳ 2–3 điểm, và PBPL − MPL gần như bằng 0 ở cả ba cấu hình.

**Những gì vẫn có giá trị:**

- Kết quả lý thuyết (N2) được kiểm chứng số chặt chẽ, kể cả trên CNN thật.
- Một kết quả âm tính sạch, có kiểm soát: nhân bản, Holm, seed xác nhận, đối chứng chỉ khác một
  thành phần. Nó chỉ ra **vì sao** không suy biến không thành accuracy (ba cơ chế ở Mục 3).
- Một phát hiện thực nghiệm: ở 40 nhãn, cái giúp là **điều chỉnh cách mạng khớp dữ liệu có nhãn**
  (SAM trên labeled loss, feedback một mạng), không phải pessimism.

**Giới hạn:** một tập dữ liệu, CNN nhỏ, 20k vòng; phương sai giữa seed 3–4 điểm, nên hiệu ứng cỡ
0.5 điểm không bị loại trừ; phép thử thang chính xác chỉ 5 seed; các đối chứng được chọn chạy tiếp
sau khi xem 2–3 seed đầu.

**Nếu muốn đi tiếp, theo thứ tự ưu tiên:**

1. Đổi trọng tâm câu chuyện: kết quả lý thuyết + phân tích vì sao nó không thành accuracy, thay vì
   tuyên bố phương pháp tốt hơn.
2. Thử trên bài toán FixMatch có confirmation bias mạnh và ổn định (CIFAR-10, 40 nhãn, huấn luyện
   dài) — nơi việc giữ tín hiệu từ dữ liệu có nhãn có nhiều chỗ để tạo khác biệt.
3. Gắn feedback bilevel một mạng lên base đã có SAM trên labeled loss (hai thành phần tốt nhất chưa
   từng chạy chung).
4. Phép thử thang chính xác với nhiều seed hơn và huấn luyện dài hơn. Thí nghiệm student pool
   (branch `claude/lucid-meitner-0yaeoo`) đã có code nhưng chưa có kết quả.
