# Góc nhìn Bayes: các phương pháp SSL dùng những term nào

Ký hiệu $X = (X_l, X_u)$. Giả định: cho trước $h$, nhãn ở các điểm độc lập,
$P(Y_l, Y_u \mid h, X) = P(Y_l \mid h, X_l)\,P(Y_u \mid h, X_u)$.

$$
P(Y_u \mid X, Y_l, \mathcal H)
= \frac{1}{P(Y_l \mid X)} \int_{\mathcal H}
\underbrace{P(Y_u \mid h, X_u)}_{\text{(a) dự đoán trên } X_u}\;
\underbrace{P(Y_l \mid h, X_l)}_{\text{(b) likelihood trên nhãn}}\;
\underbrace{P(h \mid X_l, X_u)}_{\text{(c) prior trên giả thuyết}}\; dh .
$$

$X_u$ chỉ đi vào qua (c). Nếu (c) không phụ thuộc $X_u$ thì biểu thức trên đúng bằng dự đoán
Bayes của học supervised (Seeger, 2000).

Mọi phương pháp dưới đây thay tích phân bằng một điểm (MAP):
$\hat h = \arg\min_h N_l L_l(h) - \log P(h \mid X_l, X_u)$.

| | (b) likelihood trên $Y_l$ | (c) prior trên $h$ | (c) có dùng $X_u$? | (a) dùng để | Theo $h$ | Theo $Y_u$ |
|---|---|---|---|---|---|---|
| Supervised | CE trên nhãn | kiến trúc, weight decay, SGD | không | chỉ dự đoán | 1 điểm | argmax sau khi học |
| Self-training | CE trên nhãn | tự tin trên $X_u$ (entropy thấp) | **có** | mục tiêu tự sinh | 1 điểm | điểm bất động |
| FixMatch | CE trên nhãn | tự tin + bất biến augmentation | **có** | mục tiêu tự sinh | 1 điểm | điểm bất động |
| Mean Teacher | CE trên nhãn | bất biến augmentation | **có** | so với EMA | 1 điểm (EMA là trung bình thô) | không gán cứng |
| MPL thuần | CE trên nhãn, qua $\theta_K(Y_u)$ | $\theta_0$ + $K$ bước SGD | không | dữ liệu cho student | 1 điểm $\theta_K$ | **argmax theo** $P(Y_l \mid X, Y_u)$ |
| MPL + teacher FixMatch (report §15) | như trên | MPL + KL tới $q_{\text{FixMatch}}$ | có, qua teacher | dữ liệu cho student | 1 điểm | argmax |

MPL thuần nằm cùng phía với supervised ở cột "(c) có dùng $X_u$?", nên bị chặn trên bởi
supervised (khớp toy Two Moons §13–§15 và wave7 trên FashionMNIST). Lợi ích của MPL trong bài
gốc và ở §15 đến từ prior của teacher (UDA / FixMatch).

## Từ MAP đến variational: ELBO trên giả thuyết và cận sharpness

Đặt $f(h) := P(Y_u \mid h, X_u)\,P(Y_l \mid h, X_l)$. Chọn pseudo-label là tìm cách gán nhãn có
posterior lớn nhất:
$$
\hat Y_u^\star = \arg\max_{Y_u} P(Y_u \mid X, Y_l) = \arg\max_{Y_u} \int_{\mathcal H} f(h)\,P(h \mid X)\,dh .
$$
Với mọi phân phối $q$ trên $\mathcal H$ có giá chứa giá của $f(h)P(h \mid X)$, bất đẳng thức Jensen cho
$$
\log \int f(h)\,P(h \mid X)\,dh \;\ge\; \mathcal F(q, Y_u) :=
\mathbb E_{h\sim q}\big[\log P(Y_u \mid h, X_u) + \log P(Y_l \mid h, X_l)\big] - \mathrm{KL}\big(q \,\|\, P(h \mid X)\big),
$$
với dấu bằng khi và chỉ khi $q(h) = P(h \mid X, Y_l, Y_u)$, tức posterior trên giả thuyết khi coi
$Y_u$ như nhãn thật. Viết theo loss, $-\log P(Y_l \mid h, X_l) = N_l L_l(h)$ và
$-\log P(\hat Y_u \mid h, X_u) = N_u L_u(h; \hat Y_u)$, thêm hệ số tempering $\lambda_u$ cho likelihood
pseudo-label ($\lambda_u = 1$ là đúng Bayes):
$$
\min_{\hat Y_u,\; q}\; \mathbb E_{h\sim q}\big[N_l L_l(h) + \lambda_u N_u L_u(h; \hat Y_u)\big]
+ \mathrm{KL}\big(q \,\|\, P(h \mid X)\big). \qquad \text{(V)}
$$
KL chỉ xuất hiện một lần trong (V).

### Tối ưu luân phiên

- **Bước $Y_u$** (cố định $q$): (V) tách theo từng điểm, nên
  $\hat y_i = \arg\max_y \mathbb E_{h\sim q}[\log p_h(y \mid x_i)]$. Pseudo-label là argmax của trung bình
  log-xác suất dưới $q$, tức trung bình hình học của các dự đoán.
- **Bước $q$** (cố định $\hat Y_u$):
  $q^\star(h) \propto P(Y_l \mid h, X_l)\,P(\hat Y_u \mid h, X_u)^{\lambda_u}\,P(h \mid X)$, posterior Gibbs
  trên nhãn và pseudo-label.

Self-training là tối ưu luân phiên (V). FixMatch là trường hợp $q = \delta_\theta$:
- bước $Y_u$ là argmax $p_\theta$ trên weak view, cộng ngưỡng $\tau$ (chỉ giữ điểm mà bước $Y_u$ chắc chắn);
- bước $q$ là một bước SGD trên $L_l + \lambda_u L_u$, với $L_u$ tính trên strong view;
- với $q = \delta_\theta$, KL suy biến thành $-\log P(\theta \mid X)$ cộng một hằng số vô hạn, nên (V)
  thành bài toán MAP. Prior $P(h \mid X)$ của FixMatch nằm ở bất biến augmentation (weak sang strong)
  và ở ngưỡng tự tin.

### Cận sharpness

Lấy $q$ đều trên quả cầu $B(\theta, \rho)$. Khi đó $\mathbb E_q[\cdot] \le \max_{\|\varepsilon\| \le \rho}[\cdot](\theta + \varepsilon)$,
và $\mathrm{KL}(q \,\|\, P(h \mid X)) = \mathbb E_q[-\log P(h \mid X)] - H(q)$ với $H(q)$ không phụ thuộc $\theta$. Do đó
$$
-\mathcal F \;\le\; \max_{\|\varepsilon\| \le \rho} \big[N_l L_l + \lambda_u N_u L_u(\hat Y_u)\big](\theta + \varepsilon)
+ \mathbb E_{\varepsilon \sim q}\big[-\log P(\theta + \varepsilon \mid X)\big] + \text{const}.
$$
Với prior Gauss (weight decay), số hạng thứ hai là $\|\theta\|^2 / (2\sigma_p^2)$ cộng hằng số. Đây là SAM
trên toàn bộ likelihood, cùng dạng với cận PAC-Bayes mà Foret et al. (2021) dùng để dẫn ra SAM. Hai hệ
quả:
- Vì $\max(a + b) \le \max a + \max b$, SAM tách riêng từng term là một cận lỏng hơn của cùng $\mathcal F$.
  SAM chỉ trên $L_l$ (dòng tốt nhất hiện tại, 75.79) đặt max lên likelihood nhãn và đánh giá $L_u$ tại
  tâm. Nó tương đương dùng hai $q$ khác nhau cho hai term (quả cầu cho $L_l$, một điểm cho $L_u$), nên
  không còn là cận trên của $-\mathcal F$.
- Trong $\min_{\hat Y_u} \max_\varepsilon$, nghiệm luân phiên là: tính $\varepsilon^\star$ với pseudo-label điểm,
  rồi đặt $\hat Y_u = \arg\max p_{\theta + \varepsilon^\star}$, tức pseudo-label của giả thuyết xấu nhất trong
  quả cầu.

### Trừu tượng hóa prior

Vì $P(h \mid X, Y_l) \propto P(Y_l \mid h, X_l)\,P(h \mid X)$, đặt $-\log P(h \mid X) := R(h; X)$ là phần không
nhãn của một loss SSL bất kỳ: consistency trên điểm tự tin (FixMatch, FlexMatch), consistency với mạng
EMA (Mean Teacher), tự tin trên dự đoán của chính mình (Pseudo-Label). Khi đó
$-\log P(h \mid X, Y_l) = N_l L_l(h) + R(h; X) + \text{const}$ chính là loss SSL đó, và bước $Y_u$, bước $q$,
cận sharpness ở trên áp dụng cho mọi $R$.

### Biến thể đưa vào benchmark FashionMNIST (wave8)

Mỗi biến thể nới một xấp xỉ của FixMatch trong (V). 40 nhãn, seed 1–5, bán kính $\rho = 0.1$ (đã chọn
cho SAM trên $L_l$), không tinh chỉnh lại; so cặp theo seed. Cột cuối ghi dự đoán trước khi chạy.

| tên | nới xấp xỉ nào | cờ | dự đoán |
|---|---|---|---|
| `fmsamfull` | bước $q$: quả cầu cho cả $L_l + \lambda_u L_u$ (đúng cận trên) | `--lab_reg samfull --radius 0.1` | ≥ `abl_labsam` nếu lợi ích của SAM là làm phẳng posterior; kém hơn nếu lợi ích chỉ là chống học thuộc 40 nhãn, vì gradient của $L_u$ (448 ảnh) lấn hướng ascent |
| `fmsamfulladv` | thêm bước $Y_u$ tại giả thuyết xấu nhất | `--lab_reg samfull --radius 0.1 --pl_src adv` | mask thấp hơn, pseudo-label giữ lại đúng hơn; accuracy chưa rõ |
| `labsamadv` | SAM trên $L_l$, bước $Y_u$ tại giả thuyết xấu nhất cho nhãn | `--lab_reg sam --radius 0.1 --pl_src adv` | gần `abl_labsam`, vì nhiễu dọc $\nabla L_l$ ít đổi dự đoán trên $X_u$ |
| `fmemapl` | bước $Y_u$ dưới posterior quỹ đạo (EMA) | `--pl_src ema` | trong khoảng ±1 điểm quanh FixMatch, vì cuối run EMA ≈ mạng live |
| `supsam` | SAM trên $L_l$, không có $X_u$ | `--base sup --lab_reg sam --radius 0.1` | nếu tăng khoảng 3 điểm thì lợi ích của SAM không đặc thù cho SSL |
| `plsam`, `mtsam`, `flexsam` | SAM trên $L_l$ với prior khác: Pseudo-Label, Mean Teacher, FlexMatch | `--base {pseudolabel, meanteacher --lam_u 50, flexmatch} --lab_reg sam --radius 0.1` | nếu lợi ích chung cho mọi prior thì tăng 2–4 điểm ở cả ba |

- Mốc so sánh đã có đủ 10 seed: `fixmatch`, `abl_labsam`, `sup`. Baseline `pseudolabel`,
  `meanteacher`, `flexmatch` có seed 1–3; wave8 bổ sung seed 4–5.
- Log thêm: với `--pl_src` khác `point`, các key `mask`, `pl_acc_masked`, `pl_acc_all` là của
  pseudo-label thực sự dùng, còn các key đuôi `_point` là của pseudo-label FixMatch;
  `pl_agree` là tỉ lệ hai loại trùng nhau.
- Chưa đưa vào wave này, vì cần chọn thêm siêu tham số:
  - bước $q$ dạng kỳ vọng (nhiễu trọng số ngẫu nhiên thay cho max, tức tối ưu đúng $\mathcal F$ thay vì
    cận trên của nó) và bước $Y_u$ dưới $q$ Gauss (trung bình log-xác suất của vài mạng nhiễu): cần chọn
    bán kính cho hướng ngẫu nhiên;
  - bước $Y_u$ mềm, khi lấy tổng theo $Y_u$ thay vì argmax:
    $\hat y_i \propto \exp(\mathbb E_q \log p_h(\cdot \mid x_i))$.
