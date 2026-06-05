# Can Gradient Clipping Mitigate Label Noise?

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

이 논문은 **그래디언트 클리핑(gradient clipping)을 로버스트니스(robustness) 관점**에서 새롭게 분석합니다. 직관적으로는 클리핑이 노이즈에 강건해야 할 것 같지만, 논문의 핵심 주장은 다음과 같습니다:

> **표준 그래디언트 클리핑은 레이블 노이즈에 대한 로버스트니스를 제공하지 않는다.** 그러나 이의 간단한 변형인 **복합 손실 기반 그래디언트 클리핑(CL-gradient clipping)**은 로버스트하다.

### 주요 기여 (3가지)

| 기여 | 내용 |
|---|---|
| (a) 부정적 결과 | 표준 그래디언트 클리핑 → Huberised 손실과 동치 → 레이블 노이즈에 **비로버스트** |
| (b) 긍정적 결과 | CL-gradient clipping → Partially Huberised 손실과 동치 → 레이블 노이즈에 **로버스트** |
| (c) 실증 검증 | MNIST, CIFAR-10, CIFAR-100에서 Partially Huberised 손실의 우수성 확인 |

---

## 2. 상세 설명

### 2-1. 해결하고자 하는 문제

딥러닝 학습에서 **레이블 노이즈(label noise)**는 심각한 성능 저하를 유발합니다. 실제 데이터셋에서는 레이블 오류가 흔하며, 표준 손실 함수(예: Cross-Entropy)는 이에 취약합니다.

기존에는 그래디언트 클리핑이 단일 샘플의 과도한 영향을 막으므로 노이즈에 강건할 것이라는 직관이 있었습니다. 논문은 이 직관이 **틀렸음을 이론적으로 증명**하고, 올바른 방법을 제시합니다.

---

### 2-2. 배경: 그래디언트 클리핑의 수식

미니배치 $\{(x_n, y_n)\}_{n=1}^N$에 대한 그래디언트:

$$g(\theta) \doteq \frac{1}{N} \sum_{n=1}^{N} \nabla \ell_\theta(x_n, y_n) \tag{1}$$

클리핑된 그래디언트 (임계값 $\tau > 0$):

$$\bar{g}_\tau(\theta) \doteq \text{clip}_\tau(g(\theta)), \quad \text{clip}_\tau(w) \doteq \begin{cases} \tau \cdot \frac{w}{\|w\|_2} & \text{if } \|w\|_2 \geq \tau \\ w & \text{else} \end{cases} \tag{2}$$

---

### 2-3. 표준 그래디언트 클리핑 → Huberised 손실 (Lemma 1, 2)

이진 분류에서 마진 손실 $\ell_\theta(x, y) = \phi(m_\theta(x, y))$, $m_\theta(x, y) = y \cdot s_\theta(x)$이면:

$$\nabla_\theta \ell_\theta(x, y) = \nabla_\theta m_\theta(x, y) \cdot \phi'(m_\theta(x, y)) \tag{3}$$

**Lemma 1**: 선형 스코어러와 SGD (N=1) 하에서, 클리핑된 그래디언트는 다음 수정 손실의 그래디언트와 동치:

$$\bar{\ell}_\theta(x, y) \doteq \begin{cases} -\frac{\tau}{\|\nabla m_\theta(x,y)\|_2} \cdot m_\theta(x, y) & \text{if } |\phi'(m_\theta(x,y))| \geq \frac{\tau}{\|\nabla m_\theta(x,y)\|_2} \\ \phi(m_\theta(x, y)) & \text{else} \end{cases} \tag{4}$$

**L-gradient clipping** (손실 기여분만 클리핑):

$$\text{l-clip}_\tau(\nabla \ell_\theta(x,y)) \doteq \nabla m_\theta(x,y) \cdot \text{clip}_\tau(\phi'(m_\theta(x,y))) \tag{7}$$

**Lemma 2**: Fenchel 켤레 $\phi^*$를 이용하면, L-gradient clipping은 다음 **Huberised 손실**과 동치:

$$\bar{\phi}_\tau(z) \doteq \begin{cases} -\tau \cdot z - \phi^*(-\tau) & \text{if } \phi'(z) \leq -\tau \\ \phi(z) & \text{else} \end{cases} \tag{8}$$

**로지스틱 손실의 Huberised 버전** ( $\tau \in (0,1)$ ):

$$\bar{\phi}_\tau(z) = \begin{cases} -\tau \cdot z - \log(1-\tau) - \tau \cdot \sigma^{-1}(\tau) & \text{if } z \leq -\sigma^{-1}(\tau) \\ \log(1 + e^{-z}) & \text{else} \end{cases}$$

---

### 2-4. Huberised 손실의 레이블 노이즈 비로버스트성 (Proposition 4)

**Lemma 3**: Huberised 손실 $\bar{\phi}_\tau$는 **분류 보정(classification calibrated)**이다. (즉, 노이즈 없는 환경에서는 사용 가능)

그러나:

> **Proposition 4**: 어떤 admissible 마진 손실 $\phi$와 $\tau > 0$에 대해서도, **대칭 레이블 노이즈** 하에서 $\bar{\phi}_\tau$ 하의 최적 선형 분류기가 랜덤 추측과 동치인 분리 가능 분포가 존재한다.

이는 Huberised 손실이 여전히 볼록(convex)이기 때문에 이상치 관측치에 의해 지배될 수 있음을 의미합니다.

---

### 2-5. 제안 방법: CL-gradient clipping과 Partially Huberised 손실

복합 마진 손실 $\phi = \varphi \circ F$ (예: 로지스틱 손실에서 $\varphi(u) = -\log u$, $F(z) = \sigma(z)$ )에서 확률 추정치를 $p_\theta(x, y) = F(m_\theta(x, y))$로 정의합니다.

**CL-gradient clipping**: 기저 손실 $\varphi$의 미분에만 클리핑 적용:

$$\text{cl-clip}_\tau(\nabla \ell_\theta(x,y)) \doteq \nabla p_\theta(x,y) \cdot \text{clip}_\tau(\varphi'(p_\theta(x,y))) \tag{9}$$

**Lemma 5**: CL-gradient clipping은 다음 **Partially Huberised 손실** $\tilde{\phi}\_\tau = \tilde{\varphi}_\tau \circ F$와 동치:

$$\tilde{\varphi}_\tau(u) \doteq \begin{cases} -\tau \cdot u - \varphi^*(-\tau) & \text{if } \varphi'(u) \leq -\tau \\ \varphi(u) & \text{else} \end{cases} \tag{10}$$

**다중 클래스 Partially Huberised Softmax Cross-Entropy** ($\tau > 1$):

$$\tilde{\ell}_\theta(x,y) = \begin{cases} -\tau \cdot p_\theta(x,y) + \log\tau + 1 & \text{if } p_\theta(x,y) \leq \frac{1}{\tau} \\ -\log p_\theta(x,y) & \text{else} \end{cases} \tag{12}$$

**로지스틱 손실의 Partially Huberised 버전**:

$$\tilde{\phi}_\tau(z) = \begin{cases} -\tau \cdot \sigma(z) + \log\tau + 1 & \text{if } z \leq \sigma^{-1}\!\left(\frac{1}{\tau}\right) \\ \log(1 + e^{-z}) & \text{else} \end{cases} \tag{11}$$

---

### 2-6. Partially Huberised 손실의 로버스트성 (Proposition 7)

**Lemma 6**: $\tilde{\phi}_\tau$는 classification calibrated이며, $\tau \geq -\varphi'(1/2)$이면 proper composite이기도 하다.

**Proposition 7**: 어떤 proper 손실 $\varphi$와 $\tau > 0$에 대해, $f^*$를 깨끗한 분포에서 $\tilde{\varphi}_\tau$의 위험 최소화기라 하자. 대칭 레이블 노이즈의 어떤 비자명 수준에서도 다음이 성립한다:

$$\exists C > 0 \text{ s.t. } \overline{\text{reg}}_\tau(f^*) \leq C$$

**증명 아이디어**: Ghosh et al. (2015) 및 van Rooyen et al. (2015)의 결과를 활용하면:

$$\bar{R}_\tau(f) = (1 - 2\rho) \cdot R_\tau(f) + \alpha \cdot \mathbb{E}_x[\tilde{\varphi}_\tau(f(x)) + \tilde{\varphi}_\tau(1 - f(x))]$$

Partially Huberised 손실의 포화(saturation) 특성에 의해 우변의 두 번째 항이 유한한 상수 $C$로 상한이 존재합니다:

$$\bar{R}_\tau(f^*) - \bar{R}_\tau(\bar{f}^*) \leq (C_1 - C_2)$$

---

### 2-7. 세 가지 클리핑 방식 비교 (Table 1)

| 클리핑 유형 | 클리핑된 그래디언트 형태 | 동치 손실 | 레이블 노이즈 로버스트? |
|---|---|---|---|
| Gradient | $\text{clip}_\tau(\nabla m(\theta) \cdot \phi'(m(\theta)))$ | Eq. 4 | ✗ (Proposition 4) |
| L-Gradient | $\nabla m(\theta) \cdot \text{clip}_\tau(\phi'(m(\theta)))$ | Huberised | ✗ (Proposition 4) |
| **CL-Gradient** | $\nabla p(\theta) \cdot \text{clip}_\tau(\varphi'(p(\theta)))$ | **Partially Huberised** | **✓ (Proposition 7)** |

---

### 2-8. 실험 결과

| 데이터셋 | 손실 | $\rho=0.0$ | $\rho=0.2$ | $\rho=0.4$ | $\rho=0.6$ |
|---|---|---|---|---|---|
| MNIST | CE | 98.7 | 97.7 | 96.4 | 91.1 |
| MNIST | CE + clipping | 98.6 | 97.5 | 96.8 | 88.9 |
| MNIST | **PHuber-CE** $\tau=10$ | **98.5** | **98.7** | **98.5** | **97.6** |
| CIFAR-10 | CE | 92.4 | 72.4 | 61.4 | 40.6 |
| CIFAR-10 | **PHuber-CE** $\tau=2$ | **91.6** | **88.6** | **83.6** | **72.2** |
| CIFAR-100 | CE | 66.6 | 49.7 | 29.9 | 11.4 |
| CIFAR-100 | **PHuber-GCE** $\tau=10$ | **69.8** | **64.4** | **52.4** | **31.5** |

---

## 3. 일반화 성능 향상 가능성

### 3-1. 왜 Partially Huberised 손실이 일반화에 유리한가?

**(a) 포화(Saturation) 효과를 통한 이상치 저항성**

$p_\theta(x,y) \to 0$ (모델이 해당 샘플을 매우 틀리게 예측)일 때:

- 표준 CE: 그래디언트 크기 $\propto \frac{1}{p_\theta} \to \infty$ (무한히 증가)
- Partially Huberised: 그래디언트 크기 $\leq \tau^{-1}$ (**유한한 상한**)

이를 통해 노이즈 레이블을 가진 샘플이 파라미터 업데이트를 지배하는 것을 방지합니다.

**(b) 특징 공간에서의 이상치 로버스트성 (Proposition 8)**

$\|x\|_2 \to \infty$인 이상치에 대해:

$$\lim_{\|x\|_2 \to +\infty} \|\nabla \tilde{\ell}_\tau(x, y; \theta)\|_2 = 0$$

**증명**: $\|\nabla \tilde{\ell}\_\tau(x, y; \theta)\|\_2 = \|x\|\_2 \cdot |\phi'(y \cdot \theta^T x)|$이고, $\tilde{\phi}\_\tau$의 포화 특성에 의해 $\lim_{z \to -\infty} z \cdot F'(z) = 0$이 만족되므로 성립합니다.

**(c) 분류 보정성(Classification Calibration) 유지**

Lemma 6에 의해 $\tilde{\phi}_\tau$는 항상 classification calibrated입니다. 즉:
- 손실의 최적화가 분류 정확도의 최적화와 일관성을 유지
- 노이즈 없는 환경에서도 성능 저하 없음 (Table 2의 $\rho=0.0$ 결과 확인)

**(d) 정보성 있는 그래디언트 유지**

Partially Huberised 손실의 그래디언트:

$$\nabla \ell_\theta(x,y) = -(p_\theta(x,y) \vee \tau)^{-1} \cdot \nabla p_\theta(x,y)$$

이는 잘 예측된 샘플($p_\theta \approx 1$)에서는 작은 그래디언트를, 잘못 예측된 깨끗한 샘플에서는 정보성 있는 그래디언트를 제공합니다. Linear 손실과 달리 **샘플 중요도를 구별**할 수 있어 CIFAR-100 같은 복잡한 태스크에서도 효과적입니다.

**(e) 하이퍼파라미터 $\tau$의 역할**

$$\tau \to 1^+: \text{기저 손실에 근접 (정보성 ↑, 노이즈 로버스트성 ↓)}$$
$$\tau \to +\infty: \text{선형 손실에 근접 (노이즈 로버스트성 ↑, 정보성 ↓)}$$

중간 $\tau$ 값이 최적의 트레이드오프를 달성합니다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4-1. 앞으로의 연구에 미치는 영향

**(a) 손실 함수 설계 패러다임의 전환**

이 논문은 "**어떻게 클리핑하느냐**"가 "**어떤 손실을 쓰느냐**"와 동치임을 보여줍니다. 이는 향후 새로운 클리핑 전략이 새로운 손실 함수 설계로 이어질 수 있음을 시사합니다.

**(b) 기존 노이즈 강건 방법론과의 결합**

논문 자체에서 제안하듯이, Partially Huberised 손실은 다음과 결합될 수 있습니다:
- **Sample selection**: Co-teaching (Han et al., 2018)
- **Abstention**: Thulasidasan et al. (2019)
- **Loss correction**: Patrini et al. (2017)

**(c) 분산 강건 학습(Distributionally Robust Learning)으로의 확장**

논문은 미래 연구 방향으로 Shapieezadeh-Abadeh et al. (2015), Namkoong & Duchi (2016, 2017) 등의 분산 강건 학습 프레임워크와의 연결을 제안합니다.

**(d) 차분 프라이버시(Differential Privacy)와의 시너지**

기존 클리핑의 프라이버시 보장(Abadi et al., 2016)에 노이즈 로버스트성을 추가하는 방향으로 **프라이버시 보존 + 노이즈 강건 학습**의 통합 연구가 가능합니다.

---

### 4-2. 향후 연구 시 고려할 점

**(a) 이론적 한계**

- 현재 이론은 주로 **대칭 레이블 노이즈(symmetric label noise)**에 집중됩니다.
- **비대칭(asymmetric) 또는 인스턴스 의존적(instance-dependent) 노이즈**에 대한 보장은 더 강한 분포적 가정이 필요합니다 (Ghosh et al., 2015 참조).
- Proposition 4는 선형 모델에서의 최악의 경우를 보여주며, **실제 비선형 신경망**에서는 더 완화된 결과가 예상됩니다.

**(b) 미니배치 효과**

미니배치 크기 $N > 1$에서 클리핑 효과는 단순한 손실 수정으로 모방할 수 없습니다:

$$\bar{g}_\tau(\theta) = \text{clip}_\tau\!\left(\frac{1}{N}\sum_{n=1}^N \nabla \ell_\theta(x_n, y_n)\right)$$

각 샘플의 그래디언트가 전체 미니배치 그래디언트 노름 $\|g(\theta)\|_2$에 의해 수정되므로, 미니배치 클리핑과 Partially Huberised 손실의 병행 사용이 권장됩니다.

**(c) 하이퍼파라미터 $\tau$ 선택**

논문에서는 $\tau$를 교차 검증으로 선택하지만, 실제 노이즈 수준을 모르는 경우가 많습니다. **노이즈 수준을 자동으로 추정**하거나 $\tau$를 적응적으로 학습하는 방법이 필요합니다.

**(d) 대규모 언어 모델(LLM) 및 Transformer로의 확장**

RNN에서 시작된 클리핑의 활용이 Transformer 기반 모델에서도 중요해짐에 따라, **어텐션 메커니즘과 상호작용하는 Partially Huberised 손실**의 효과 분석이 필요합니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

아래 연구들은 이 논문과 직접적으로 관련된 2020년 이후 연구들로, 논문 PDF에 근거하거나 해당 논문이 직접 언급한 연구들의 후속 흐름에 해당합니다. **단, 이 논문이 2020년 ICLR에 발표된 이후의 구체적인 후속 연구들은 제가 직접 확인한 PDF에 포함되어 있지 않으므로**, 아래에서는 이 논문이 다루는 핵심 주제들(레이블 노이즈, 손실 함수 설계, 그래디언트 클리핑)과 관련하여 **논문 내에서 직접 비교 또는 언급된 관련 연구**들과의 비교를 중심으로 기술합니다.

### 논문 내에서 비교된 주요 관련 연구들

| 연구 | 접근법 | 이 논문과의 차이 |
|---|---|---|
| **Zhang & Sabuncu (2018)**, GCE | $\varphi_\alpha(u) = (1-u^\alpha)/\alpha$, 로그 손실과 선형 손실 사이를 보간 | GCE는 $p_\theta \to 0$에서 그래디언트 무한대 가능 (Lipschitz 미보장). Partially Huberised는 $\tau^{-1}$의 **경질 상한(hard cap)** 보장 |
| **van Rooyen et al. (2015)**, Unhinged/Linear Loss | 대칭 노이즈에 대해 **완전 로버스트** | 최적화 어려움 (샘플 중요도 미구별). CIFAR-100에서 노이즈 없는 경우에도 성능 저하 |
| **Han et al. (2018)**, Co-teaching | 두 네트워크의 불일치를 이용한 샘플 선택 | 아키텍처 두 배 필요. 이 논문의 손실은 Co-teaching과 **결합 가능** |
| **Amid et al. (2019)**, Bi-tempered Logistic | Tsallis/Bregman 발산 기반 두 온도 매개변수 | 더 복잡한 매개변수화. Partially Huberised는 단일 $\tau$로 제어 가능 |
| **Thulasidasan et al. (2019)**, Abstention | 불확실한 샘플 기권 | 기권 레이블 추가 필요. 이 논문과 결합 가능 |

> **⚠️ 주의**: 2020년 이후 발표된 구체적인 후속 논문들(예: DivideMix, CORES², SOP 등)에 대한 정량적 비교는 이 PDF에 포함되어 있지 않으므로, 해당 내용을 단정적으로 기술하는 것은 생략합니다.

---

## 참고자료

- **주 논문**: Menon, A.K., Rawat, A.S., Reddi, S.J., Kumar, S. "Can Gradient Clipping Mitigate Label Noise?" ICLR 2020.
- Zhang, Z. & Sabuncu, M. "Generalized cross entropy loss for training deep neural networks with noisy labels." NeurIPS 2018.
- van Rooyen, B., Menon, A.K., Williamson, R.C. "Learning with symmetric label noise: the importance of being unhinged." NeurIPS 2015.
- Long, P.M. & Servedio, R.A. "Random classification noise defeats all convex potential boosters." Machine Learning, 2010.
- Ghosh, A., Manwani, N., Sastry, P.S. "Making risk minimization tolerant to label noise." Neurocomputing, 2015.
- Han, B. et al. "Co-teaching: Robust training of deep neural networks with extremely noisy labels." NeurIPS 2018.
- Abadi, M. et al. "Deep learning with differential privacy." CCS 2016.
- Reid, M.D. & Williamson, R.C. "Composite binary losses." JMLR, 2010.
- Amid, E., Warmuth, M.K., Srinivasan, S. "Two-temperature logistic regression based on the Tsallis divergence." AISTATS 2019.
- Patrini, G. et al. "Making deep neural networks robust to label noise: a loss correction approach." CVPR 2017.
