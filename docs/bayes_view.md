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
