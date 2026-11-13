# Trust-region bilevel pseudo-labeling không suy biến (TR-BPL)

Tài liệu này mô tả formulation, các mệnh đề lý thuyết và thuật toán được cài đặt trong
`ssl_fmnist.py`. Ký hiệu theo bản viết lại trong review (Sec. 2 của `pbpl_review.pdf`):
$p_w(x)\in\operatorname{relint}\Delta$, $L_l(\theta)$, $L_u(\hat Y,\theta)$ dùng KL,
teacher $\phi$, student $\theta$, $J_u(\theta)$ là Jacobian của $\theta\mapsto p_\theta(X)$
trên tập input $X$ mà student được huấn luyện, $\Pi_\theta$ là phép chiếu trực giao lên
$\ker J_u(\theta)$, $H=\nabla^2 L_l$.
Các mệnh đề có số liệu kiểm chứng nằm trong `checks/` (log đi kèm).

---

## 1. Formulation

### 1.1 Tập student chấp nhận được, neo tại chính student

**Definition 1 (self-anchored admissible set).** For an anchor student $\theta$, the
inputs $X$ the student is trained on, and a radius $r>0$,
$$
\mathcal F_X(\theta) := \{\theta' : p_{\theta'}(X) = p_\theta(X)\},\qquad
S_r(\theta) := \mathcal F_X(\theta)\cap \bar B(\theta,r).
$$

**Lemma 1′ (exact value function at every anchor).** Let
$D_X(\theta,\theta') := \frac1{|X|}\sum_{x\in X}\mathrm{KL}(p_\theta(x)\,\|\,p_{\theta'}(x))$.
Then $D_X\ge 0$, $D_X(\theta,\theta')=0$ iff $\theta'\in\mathcal F_X(\theta)$, and
$\min_{\theta'}D_X(\theta,\theta')=0$ is attained at $\theta'=\theta$ — for every
$\theta$, whatever the pseudo-labels (soft or hard), the augmentation, or the training
procedure that produced $\theta$. If $\theta$ fits $\hat Y$ exactly on $X$, then
$\mathcal F_X(\theta)=S(\hat Y)$.

*Proof.* KL is nonnegative and vanishes iff its arguments coincide. $\square$

Vì sao cần: Lemma 1 của bản nháp ($f^\star=0$, $\theta=\phi$ khả thi) gãy khi student nhìn
input đã augment hoặc pseudo-label là nhãn cứng (review, Remark 1, mục 8, A-T4). Neo
fiber tại *dự đoán của chính student* thay vì tại pseudo-label giữ $f^\star\equiv 0$ trong
mọi setting thực tế (FixMatch, hard label, minibatch) mà không cần biến phụ ước lượng
$f^\star(\phi)$. Cái giá: admissible set là "các student không phân biệt được với anchor
trên $X$", tức cùng mức khớp pseudo-label với anchor, thay vì "các student khớp hoàn hảo
pseudo-label". Hai khái niệm trùng nhau khi anchor khớp hoàn hảo.

### 1.2 Hai giá trị trust-region

**Definition 2.**
$$
V_r(\theta) := \max_{\theta'\in S_r(\theta)} L_l(\theta')\quad\text{(pessimistic)},\qquad
E_s(\theta) := \mathbb E_{\xi\sim\mathcal N(0,s^2 I)}\, L_l\big(R_\theta(\theta+\xi)\big)\quad\text{(expected)},
$$
với $R_\theta$ là phép co (nearest-point retraction) về $\mathcal F_X(\theta)$, xác định gần
$\theta$ khi $J_u$ có hạng hằng gần $\theta$ (fiber là đa tạp $C^2$ với không gian tiếp xúc
$\ker J_u(\theta)$).

Ý nghĩa vận hành của $E_s$: lấy student, nhiễu tham số đẳng hướng, rồi *huấn luyện lại*
cho khớp dự đoán cũ trên $X$; $E_s$ là labeled loss trung bình của các student "huấn luyện
lại" đó. Trong mô hình đặc trưng tuyến tính $R_\theta(\theta+\xi)=\theta+\Pi\xi$ đúng
bằng luật hội tụ của gradient flow trên $L_u$ từ khởi tạo Gaussian (review, mục 2.5), nên
đây là định nghĩa chặt cho measure của EBO trong bản nháp.

### 1.3 Bài toán

Teacher $\phi$ sinh pseudo-label mềm $p_\phi(X)$; anchor $a(\phi)$ là student học từ
chúng (lý tưởng: huấn luyện từ khởi tạo không phụ thuộc nhãn; thực tế: một bước SGD từ
student hiện tại, như MPL):
$$
\min_\phi\ \Phi(\phi),\qquad \Phi(\phi)=V_r(a(\phi))\ \text{(PBPL)}\quad\text{hoặc}\quad E_s(a(\phi))\ \text{(EBPL)}.
$$
Gộp với một regularizer SSL trên mạng được triển khai (ở đây là chính teacher):
$$
\min_\phi\ L_l(\phi) + \lambda_u L_{\text{FixMatch}}(\phi) + \lambda_m\,\Phi(\phi).
$$
Hai thiết kế: **2 mạng** (student phụ, không thấy nhãn, cùng kiến trúc, học từ pseudo-label
mềm của teacher; teacher là mô hình triển khai) — đây là phương pháp chính; **1 mạng**
(student là look-ahead một bước của chính mạng trên pseudo-label của nó) — ablation, vì
khi không có augmentation look-ahead trùng với teacher và ta rơi vào trường hợp "tâm tại
teacher" mù pseudo-label (review, Prop. 6).

---

## 2. Lý thuyết

### 2.1 Các formulation cũ suy biến ở đâu (tóm tắt từ review)

| feedback cho pseudo-label | hyper-objective | điểm bất động một bước |
|---|---|---|
| optimistic (MPL, BOPL) | suy biến: mọi cặp supervised-only tối ưu (Thm 1), hằng dưới interpolation (Thm 2) | mọi cặp supervised-only là điểm bất động (Prop 4) |
| pessimistic toàn cục | vô nghĩa: $+\infty$ dưới interpolation (Thm 2) | — |
| pessimistic phạt $\gamma<1/\ell_l$ (inner strongly concave) | cùng tập nghiệm với supervised-only khi tâm tại teacher (Prop 12) | **suy biến** (Prop N2(c) dưới đây) |
| **trust-region cứng, tâm tại student (PBPL)** | phân biệt các nghiệm supervised-only (bậc hai) | **không suy biến**, tín hiệu bậc $r$ (Prop N2(b)) |
| **expected trên fiber, tâm tại student (EBPL)** | phân biệt (bậc hai), hàm trơn | **không suy biến**, tín hiệu bậc $s^2$ (Prop N2(d)) |

Định nghĩa dùng xuyên suốt: một quy tắc feedback là **suy biến** nếu mọi cặp
supervised-only $(w,w)$ với $\nabla L_l(w)=0$ (ghép với chính dự đoán của nó làm
pseudo-label) là nghiệm tối ưu, hoặc là điểm bất động của động học tương ứng.

### 2.2 Khai triển

**Proposition A.** Let $J_u$ have constant rank near $\theta$.
1. $V_r(\theta)=L_l(\theta)+r\|\Pi_\theta\nabla L_l(\theta)\|+O(r^2)$; if
   $\nabla L_l(\theta)=0$ then $V_r(\theta)=L_l(\theta)+\frac{r^2}{2}\max\{0,\lambda_{\max}(\Pi H\Pi)\}+o(r^2)$
   (review, Prop. 5).
2. $E_s(\theta)=L_l(\theta)+s^2\langle\nabla L_l(\theta),m_\theta\rangle+\frac{s^2}{2}\operatorname{tr}(\Pi_\theta H_\theta\Pi_\theta)+O(s^3)$,
   where $m_\theta=\frac12\operatorname{tr}D^2R_\theta(\theta)$ is the curvature drift of the
   retraction ($m_\theta=0$ when the fiber is affine, e.g. linear features). If
   $\nabla L_l(\theta)=0$: $E_s(\theta)=L_l(\theta)+\frac{s^2}{2}\operatorname{tr}(\Pi H\Pi)+O(s^3)$.

*Proof of 2.* $R_\theta$ is $C^2$ near $\theta$ with $R_\theta(\theta)=\theta$,
$DR_\theta(\theta)=\Pi_\theta$, so $R_\theta(\theta+\xi)=\theta+\Pi\xi+\frac12 D^2R[\xi,\xi]+O(\|\xi\|^3)$.
Taylor-expand $L_l$ and take expectations: $\mathbb E\Pi\xi=0$,
$\mathbb E\,\frac12D^2R[\xi,\xi]=s^2m_\theta$, $\mathbb E\,(\Pi\xi)^\top H\Pi\xi=s^2\operatorname{tr}(\Pi H\Pi)$. $\square$

Cả hai giá trị đều phụ thuộc $X_u$ qua $\Pi_\theta$ và phân biệt các nghiệm supervised-only
ở bậc hai: $V_r$ qua độ cong lớn nhất, $E_s$ qua vết độ cong của $L_l$ dọc các hướng mà dữ
liệu không nhãn không ghim.

### 2.3 Không suy biến của động học một bước (kết quả chính)

Setting lý tưởng của lý thuyết: pseudo-label mềm, full batch, teacher và student nhìn cùng
input $X$, không augmentation. Động học một bước (MPL dạng mềm), với vector feedback $v$:
$$
\theta^+=\theta-\eta\nabla_\theta L_u(p_\phi(X),\theta),\qquad
\phi^+=\phi-\eta_\phi\,A(\phi,\theta)\,v(\theta^+),\qquad
A(\phi,\theta):=\tfrac{\eta}{N_u}J_u(\phi)^\top\operatorname{diag}(p_\theta(X))^{-1}J_u(\theta),
$$
$A(\phi,\theta)v$ chính là $\nabla_\phi L_l(\theta^+(\phi)+\varepsilon)$ với $\varepsilon$
giữ cố định (Danskin). Ba quy tắc:
$v_{\rm opt}(\theta)=\nabla L_l(\theta)$;
$v_{\rm pess}(\theta)=\nabla L_l(\theta+\varepsilon^\star(\theta))$ với
$\theta+\varepsilon^\star(\theta)\in\arg\max_{S_r(\theta)}L_l$ (một lựa chọn đo được bất kỳ);
$v_{\rm exp}(\theta)=\mathbb E\,\nabla L_l(R_\theta(\theta+\xi))$.

**Proposition N2 (fixed points).** Let $w$ satisfy $\nabla L_l(w)=0$, let $J_u$ have
constant rank near $w$, and consider the supervised-only pair $(\phi,\theta)=(w,w)$; the
student does not move ($\theta^+=w$) and $A:=A(w,w)$ is symmetric positive semidefinite
with $Av=0\iff J_u(w)v=0$.

- (a) *Optimistic.* $Av_{\rm opt}(w)=0$: $(w,w)$ is a fixed point (review, Prop. 4).
- (b) *Hard trust region.* Suppose $\lambda:=\lambda_{\max}(\Pi H\Pi)>0$ is simple with unit
  eigenvector $u\in\ker J_u(w)$. Then every maximizer satisfies
  $\varepsilon^\star(w)=\pm r u+o(r)$ and
  $$A\,v_{\rm pess}(w)=\pm r\,A(I-\Pi)Hu+o(r).$$
  Hence $(w,w)$ is **not** a fixed point for all small $r>0$ iff $(I-\Pi)Hu\ne0$, i.e.
  unless the steepest-curvature direction of $L_l$ along the fiber is an eigenvector of $H$
  (no coupling between what the unlabeled data pins and what it does not).
- (c) *Penalty with $\gamma<1/\ell_l$ (strongly concave inner problem).* For
  $F(\varepsilon)=L_l(w+\varepsilon)-\|\varepsilon\|^2/(2\gamma)-\beta D_X(w,w+\varepsilon)$ the
  unique maximizer is $\varepsilon^\star=0$ (strictly concave, zero gradient at 0), so
  $v=\nabla L_l(w)=0$ and $(w,w)$ **is** a fixed point: the GDA-friendly regime is
  degenerate, as the strong-concavity regime of the review (Prop. 12) predicts.
- (d) *Expected.* $A\,v_{\rm exp}(w)=s^2A\big(Hm_w+\tfrac12\nabla^3L_l(w)[\Pi]\big)+O(s^3)$,
  where $\nabla^3L_l(w)[\Pi]$ is the gradient of $\theta\mapsto\operatorname{tr}(\Pi H_\theta)$
  with $\Pi$ frozen. Not a fixed point for small $s>0$ whenever
  $J_u(w)\big(Hm_w+\tfrac12\nabla^3L_l(w)[\Pi]\big)\ne0$.

*Proof.* $A=\frac{\eta}{N_u}J^\top\operatorname{diag}(p)^{-1}J\succeq0$, so $Av=0$ iff
$v^\top Av=0$ iff $Jv=0$. (a) $v_{\rm opt}=\nabla L_l(w)=0$. (b) On the $C^2$ fiber,
$\varepsilon=\Pi\varepsilon+O(\|\varepsilon\|^2)$ and
$L_l(w+\varepsilon)=L_l(w)+\frac12\varepsilon^\top H\varepsilon+O(r^3)$; the maximum over
$S_r(w)$ is $\frac{r^2}2\lambda+O(r^3)$ and, since $\lambda$ is simple and positive, any
maximizer has direction within $O(r^{1/2})$ of $\pm u$ and norm $r(1-o(1))$. Then
$v_{\rm pess}=H\varepsilon^\star+O(r^2)=\pm rHu+o(r)$ and $JHu=J(I-\Pi)Hu$ because $J\Pi=0$;
$J$ is injective on $\operatorname{range}(I-\Pi)=\operatorname{range}J^\top$. (c)
$\nabla F(0)=\nabla L_l(w)-0-\beta\nabla_{\theta'}D_X(w,w)=0$ and $F$ strictly concave. (d)
$\nabla L_l(R(w+\xi))=H(R(w+\xi)-w)+\frac12\nabla^3L_l(w)[\Pi\xi,\Pi\xi]+O(\|\xi\|^3)$; take
expectations as in Prop. A. $\square$

**Kiểm chứng số** (`checks/check_nondegeneracy.py`, log trong `checks/check_nondegeneracy.log`):
MLP tanh một lớp ẩn, $d=131$, $C=3$, $N_l=6$ (nhãn mềm để $\inf L_l$ đạt được), $N_u=25$,
$\dim\ker J_u=81$. Tại $w$ với $\|\nabla L_l(w)\|=6\cdot10^{-9}$:
$\lambda_{\max}(\Pi H\Pi)=0.077$, $\|(I-\Pi)Hu\|=0.32$, độ dốc dự đoán $\|AHu\|=0.124$.

| $r$ | $\|A v_{\rm opt}\|$ | $\|A v_{\rm pess}\|$ | $\|A v_{\rm pess}\|/r$ | $V_r-L_l(w)$ |
|---|---|---|---|---|
| 0.02 | 2.9e-9 | 2.49e-3 | 0.125 | 1.6e-5 |
| 0.05 | 2.9e-9 | 6.29e-3 | 0.126 | 9.7e-5 |
| 0.10 | 2.9e-9 | 1.28e-2 | 0.128 | 3.9e-4 |
| 0.20 | 2.9e-9 | 2.66e-2 | 0.133 | 1.6e-3 |

EBPL (Monte Carlo 400 cặp antithetic): $\|A v_{\rm exp}\|/s^2 = 0.36,\ 0.43,\ 0.59$ tại
$s=0.025, 0.05, 0.1$ — bậc $s^2$ như (d), hằng số chưa hội tụ hẳn do số hạng bậc cao.

**Kiểm chứng trên CNN thật** (`checks/degeneracy_static.py`, `docs/report.md` Mục 5): CNN
0.157M tham số chỉ huấn luyện trên 100 ảnh có nhãn, cặp supervised-only tại các checkpoint,
448 ảnh không nhãn không giao tập có nhãn, BN theo batch như lúc huấn luyện, 3 seed. Khi
$L_l(w_k)$ giảm từ $2\cdot10^{-2}$ xuống $10^{-4}$, $\|g_T\|$ optimistic giảm từ 0.16–0.71
xuống 0.003–0.03 (N2(a)). $\|g_T\|$ pessimistic ($r=0.1$, chỉ lấy iterate có KL tới anchor
$\le2\cdot10^{-3}$) lớn hơn 24–679 lần (trung vị 134) ở 14/15 cặp (seed, checkpoint) sau vòng
250 (N2(b)). Ngoại lệ là một checkpoint mà chỉ 7% iterate của bộ giải ở lại trên fiber.
Expected ($s=3$) chỉ lớn hơn trung vị 2.4 lần (bậc $s^2$, N2(d)). Ở $r=0.3$, bộ giải
ascent-rồi-co với $K=5$ không ở lại được trên fiber (≤ 8% iterate hợp lệ), nên bán kính đó
không kiểm định được N2(b).

### 2.4 Dạng bậc một: feedback = MPL + gradient alignment

**Proposition N3.** If $\Pi_\theta\nabla L_l(\theta)\neq0$ and $J_u$ has constant rank near
$\theta$, then as $r\to0$
$$
v_{\rm pess}(\theta)=\nabla L_l(\theta)+r\,H_\theta\,\bar u_\theta+o(r),\qquad
\bar u_\theta=\frac{\Pi_\theta\nabla L_l(\theta)}{\|\Pi_\theta\nabla L_l(\theta)\|},
$$
and $H_\theta\bar u_\theta=\nabla_\theta\|\Pi\nabla L_l(\theta)\|$ with $\Pi$ frozen. So the
teacher step $Av_{\rm pess}$ is the MPL step plus $r$ times the gradient of the
**invisible labeled gradient norm** $\|\Pi\nabla L_l\|$, restricted to the directions in
which the teacher can move the student. In tangent-kernel form
$\|\Pi\nabla L_l\|^2=q^\top(K_{ll}-K_{lu}K_{uu}^{\dagger}K_{lu}^\top)q$ with $q$ the labeled
residual: the residual measured in the kernel *conditioned on the unlabeled inputs*.

Đọc: MPL chỉ thấy phần gradient có nhãn mà dữ liệu không nhãn "giải thích" được
($J_u\nabla L_l$); PBPL thêm phần thưởng cho pseudo-label khiến phần còn lại (phần dữ liệu
không nhãn không ghim) nhỏ đi. Đây là cơ chế duy nhất mà một bước pessimistic thêm vào; ở
điểm dừng của $L_l$ nó trở thành số hạng độ cong của N2(b).

### 2.5 Giới hạn trong lazy regime (nói thẳng)

Với đặc trưng tuyến tính và anchor là student nghiệm chuẩn nhỏ nhất (review, Prop. 7):
$V_r(a(z))=\log(1+e^{r\omega-m(z)})$ và
$E_s(a(z))=\mathbb E_{g\sim\mathcal N(0,1)}\log(1+e^{-m(z)+s\omega g})\approx\log(1+e^{s^2\omega^2/2-m(z)})$
khi margin lớn. Cả hai là logistic loss của *cùng* margin $m(z)$, dịch đi $r\omega$ hoặc
$s^2\omega^2/2$: chúng không mang thông tin mới so với feedback optimistic, chỉ không bão
hòa sớm. Với nhiều điểm có nhãn $j$, $m_j(z)=y_j z^\top k_j$ với
$k_j=(\Psi^\top\Psi)^{-1}\Psi^\top\psi(x_j)$ là trọng số hồi quy kernel của $x_j$ lên các điểm
không nhãn, nên cả MPL lẫn PBPL đều đẩy pseudo-label về nghiệm kiểu *label propagation*
($z$ thẳng hàng với $\sum_j w_j y_jk_j$); khác biệt duy nhất là trọng số
$w_j=\sigma(r\omega_j-m_j)$ thay cho $\sigma(-m_j)$, tức PBPL **nâng trọng số theo hàm mũ
$\approx e^{r\omega_j}$ cho các điểm có nhãn nằm xa span của dữ liệu không nhãn** — những
điểm mà dữ liệu không nhãn "chưa phủ". Mọi lợi ích vượt quá điều đó phải đến từ việc $\ker J_u(\theta)$ phụ thuộc
$\theta$ (feature learning) — đây là điều thực nghiệm phải trả lời, không phải điều lý
thuyết đã chứng minh.

### 2.6 Feedback phải held-out: $X_l\cap X=\emptyset$ là một phần của formulation

**Remark (labeled inputs inside the student's inputs).** If $X_l\subset X$, every
$\theta'\in\mathcal F_X(\theta)$ has $p_{\theta'}(X_l)=p_\theta(X_l)$, so $L_l$ is constant on
$S_r(\theta)$ and $V_r=E_s=L_l$: pessimism and expectation both collapse to the optimistic
value. Moreover the anchor can fit $Y_l$ through the teacher's pseudo-labels on $X_l$, so
$\nabla L_l(\text{anchor})\to0$ and the feedback vanishes for a reason unrelated to the
quality of the pseudo-labels on $X\setminus X_l$. Giả thiết $X_l\cap X_u=\emptyset$ của
Assumption 1 (review) vì vậy không phải chi tiết kỹ thuật.

Đây đúng là điều đã xảy ra trong pilot đầu tiên: theo quy ước FixMatch, tập không nhãn chứa
cả ảnh có nhãn; student học chúng qua pseudo-label (đúng) của teacher, labeled loss của
student xuống $0.015$, feedback MPL triệt tiêu, và việc chuẩn hóa feedback theo batch
khuếch đại phần nhiễu còn lại (40 nhãn, seed 0, 20k vòng: FixMatch 72.5%, MPL 71.4%,
PBPL 68.3%). Cài đặt hiện tại loại ảnh có nhãn khỏi *nhánh student* (loss của student,
fiber, JVP); teacher vẫn dùng toàn bộ dữ liệu cho FixMatch như chuẩn.

### 2.7 Hội tụ — cái gì được bảo đảm, cái gì không

- **EBPL**: $E_s$ trơn; với retraction chính xác và anchor một bước, bước teacher là
  gradient ngẫu nhiên không chệch của hàm hợp trơn, nên các kết quả SGD không lồi chuẩn
  cho $O(\delta^{-4})$ tới điểm $\delta$-dừng. Cài đặt dùng $K$ bước co gần đúng nên chỉ
  thỏa xấp xỉ.
- **PBPL bán kính cứng**: bài toán trong là cực đại một hàm (cục bộ) lồi trên quả cầu —
  không lõm, nên không có bảo đảm GDA. Vùng có bảo đảm (phạt $\gamma<1/\ell_l$, review
  Prop. 13) chính là vùng suy biến (N2(c)). Đây là đánh đổi thật giữa hai biến thể: PBPL
  có tín hiệu bậc $r$ nhưng không có bảo đảm hội tụ; EBPL có bảo đảm nhưng tín hiệu chỉ
  bậc $s^2$.

### 2.8 Phạm vi thực nghiệm: lý thuyết giải thích được gì, không giải thích được gì

Tổng hợp từ `docs/report.md`, FashionMNIST, CNN 0.157M tham số, 40 và 100 nhãn:

1. **Không suy biến tại điểm bất động: được xác nhận.** Tại các cặp supervised-only trên CNN
   thật, feedback pessimistic trên fiber lớn hơn optimistic trung vị 134 lần và không giảm khi
   mạng khớp nhãn (report Mục 5). Đây là nội dung của N2(a)–(b).
2. **Huấn luyện hai mạng không đi tới vùng đó.** Student held-out không bao giờ khớp hết nhãn
   (labeled loss ≥ 0.05 ở vòng 20k), nên feedback MPL không triệt tiêu (report Mục 4.2). Đúng
   như vậy, trên 10 seed ở 40 nhãn PBPL ≈ MPL: chênh +0.04 ± 3.47 điểm (report Mục 9). Định lý
   đúng nhưng điều kiện của nó (anchor khớp nhãn) không xảy ra trong cấu hình hai mạng.
3. **Thiết kế A4 không cần cho accuracy.** Biến thể "một bước ascent từ 0" (ε reset mỗi bước,
   tức SAM, mù với $X_u$) là thứ mà Mục 3 loại bỏ về mặt lý thuyết. Trên thực nghiệm nó ít nhất
   bằng PBPL (3 seed). SAM trên labeled loss của chính teacher (không bilevel) cũng vậy (report
   Mục 6). Lợi ích đo được ở 40 nhãn so với FixMatch mặc định không quy được cho cấu trúc
   pessimistic bilevel. Trên 10 seed, chính lợi ích đó cũng biến mất (PBPL − FixMatch = +0.58,
   p = 0.57; report Mục 9).
4. **Ở cấu hình một mạng, nơi định lý áp dụng, độ lớn đúng như dự đoán nhưng không quyết định
   accuracy.** Anchor là teacher đã thấy nhãn (labeled loss 3–7·10⁻⁴). Feedback optimistic thô
   co lại 20–30 lần trong lúc huấn luyện, đúng như N2(a); feedback pessimistic thì không (gấp
   50–65 lần ở vòng 20k). Nhưng bản optimistic một mạng *tốt hơn* bản pessimistic (+3.47
   điểm, 2/2 seed; report Mục 6.1). Lý do khớp với log: feedback được chuẩn hóa bằng RMS chạy
   của chính nó trước khi vào $L_{\rm meta}$, nên độ lớn tiến về 0 của N2(a) bị chuẩn hóa
   xóa đi và bước meta của hai biến thể có cùng cỡ. N2 nói về *độ lớn* của feedback; với
   chuẩn hóa này chỉ *hướng* còn ảnh hưởng tới huấn luyện. Về hướng, N3 chỉ nói feedback
   pessimistic bằng hướng MPL cộng gradient của chuẩn labeled gradient "vô hình" (ở đây thành
   phần này chiếm ưu thế: cos 0.33 với optimistic). Nó không nói thành phần đó tốt hơn cho
   accuracy, và trên 2 seed thì hướng optimistic tốt hơn.
   Phép thử ở thang meta-gradient chính xác (`--meta_norm 0`), nơi feedback optimistic thực sự
   tắt dần, nằm trong đợt `rep-d`.

Tóm lại: lý thuyết đảm bảo feedback pessimistic *khác 0 và khác MPL* tại các điểm mà MPL suy
biến, và điều đó được xác nhận cả ở chẩn đoán tĩnh lẫn trong lúc huấn luyện một mạng. Nó không
đảm bảo, và thực nghiệm không cho thấy, rằng sự khác biệt đó cải thiện accuracy: ở cấu hình
hai mạng MPL không suy biến; ở cấu hình một mạng sự suy biến về độ lớn bị chuẩn hóa feedback
trung hòa. Để định lý có hệ quả thực hành, cần một cài đặt mà độ lớn feedback được giữ nguyên
(thang chính xác hoặc chuẩn hóa dùng chung cho mọi biến thể, ví dụ chia cho RMS của feedback
optimistic). Đây là hướng còn để ngỏ.

---

## 3. Thuật toán (cài đặt trong `ssl_fmnist.py`)

Mỗi vòng lặp, với batch có nhãn $(x_l,y_l)$, batch không nhãn $x_u$, augment yếu $\alpha$
và mạnh $\mathcal A$:

1. **Teacher (mô hình triển khai)**: CE trên $\alpha(x_l)$ + FixMatch (ngưỡng 0.95,
   $\alpha(x_u)\to\mathcal A(x_u)$); pseudo-label mềm $t=\operatorname{softmax}(f_\phi(\alpha x_u)/T_s)$.
2. **Student** (2 mạng, không thấy nhãn): $\theta'=\theta-\eta\nabla_\theta\,\mathrm{CE}(t,f_\theta(\mathcal A x_u))$.
3. **Student chấp nhận được** quanh anchor $\theta'$, fiber neo tại dự đoán của $\theta'$
   trên đúng batch $\mathcal A x_u$ (Lemma 1′):
   - PBPL: $\varepsilon$ là **biến trạng thái** (không reset — xử lý A4: một bước ascent từ
     0 bỏ qua $X_u$ và chính là SAM), bước ascent chuẩn hóa
     $\varepsilon\leftarrow\operatorname{proj}_{B(0,r)}\big(\varepsilon+\alpha r\,g/\|g\|\big)$,
     $g=\nabla L_l(\theta'+\varepsilon)-\beta\nabla D(\theta',\theta'+\varepsilon)$.
   - EBPL: $\xi$ ngẫu nhiên trên mặt cầu bán kính $r$, co về fiber bằng $K$ bước giảm
     $D$ (bước Rayleigh), chuẩn hóa lại về bán kính $r$.
4. **Feedback**: $v=\nabla L_l(\theta'+\varepsilon)$; $c_i=J_{f_\theta}(\mathcal A x_i)\,v$
   (một JVP); $L_{\rm meta}=\frac1B\sum_i\langle t_i,\operatorname{sg}(c_i)\rangle$, có
   gradient theo $\phi$ đúng bằng $\frac{B}{\eta}\nabla_\phi L_l(\theta'(\phi)+\varepsilon)$
   (kiểm chứng `checks/check_metagrad.py`: sai số tương đối $3\cdot10^{-16}$). Mặc định
   chuẩn hóa $c$ theo RMS để $\lambda_m$ có thang cố định.
5. Teacher: SGD trên $L_l+\lambda_uL_{\rm FM}+\lambda_mL_{\rm meta}$; student: SGD với
   gradient ở bước 2; EMA cho đánh giá.

Các điểm thiết kế đáp lại review: chỉ có đường gradient qua student (không có đường qua
target trong số hạng phạt — tránh lỗ hổng A2, teacher không được lợi khi làm pseudo-label
kém tự tin); $\varepsilon$ giữ trạng thái và batch không nhãn lớn (A4); fiber ở mức xác
suất (A5); bán kính cứng thay vì phạt $\gamma<1/\ell_l$ (vấn đề 2, N2(c)).
