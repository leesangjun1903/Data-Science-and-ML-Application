# Temporal Calibrated Regularization (TCR) for Robust Noisy Label Learning

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

DNNs는 노이즈 레이블 데이터에서 훈련 시 **두 단계(early pattern learning → late label memorization)**를 거치는데, 기존 방법들은 이 분기점을 수동으로 설정하거나 복잡한 스케줄러가 필요하다는 문제가 있다. TCR은 **이전 에폭의 예측값과 원본 레이블을 결합한 pseudo-label**을 활용하여 추가적인 단계 구분 없이 노이즈에 강인한 학습을 가능하게 한다.

### 주요 기여

| 기여 항목 | 설명 |
|---|---|
| **Reflection Loss** | 원본 레이블 + 이전 에폭 예측값의 볼록 결합(convex combination)으로 손실 정의 |
| **Squeeze Technique** | 학습률 감소 시 저항력 저하(Resistance Degradation) 보완을 위한 신뢰도 증폭 함수 |
| **낮은 오버헤드** | 메모리 $O(c \cdot n)$, 시간 오버헤드 거의 없음 |
| **범용성** | 다양한 아키텍처(ResNet, WRN 등) 및 노이즈 유형에서 일관된 성능 향상 |

---

## 2. 상세 설명

### 2-1. 해결하고자 하는 문제

#### 문제 정의
- DNN은 대규모 데이터 학습 시 **노이즈 레이블에 과적합(overfitting)**되는 경향이 있음
- 기존의 **두 단계 분리 방법**은 아키텍처마다 과적합 시작 시점이 다르기 때문에 일반화 어려움

$$
L(\theta) = -\frac{1}{n}\sum_{i=1}^{n} \tilde{y}_i^{\top} \log f(x_i; \theta) \quad \cdots (1)
$$

- 위 표준 Cross Entropy Loss로 학습 시, 노이즈 레이블에 의한 잘못된 gradient가 파라미터 업데이트를 오염시킴:

$$
\frac{\partial L}{\partial h} = -\sum_{i=1}^{n} \left( y_i - f(x_i; \theta) \right) \quad \cdots (2)
$$

$$
\theta_{t+1} = \theta_t - \epsilon \left(\frac{\partial L}{\partial h}\right)^{\top} \frac{\partial h}{\partial \theta_t} \quad \cdots (3)
$$

#### 노이즈 전이 확률 모델
클래스 의존적 노이즈는 전이 행렬 $T$로 표현:

$$
p(\tilde{y} = e^k | y = e^j, x) = T_{jk} \quad \cdots (5)
$$

균일 노이즈(Uniform Noise)의 경우:

$$
T_{jk} = \begin{cases} 1 - \eta, & \text{if } j = k \\ \dfrac{\eta}{c-1}, & \text{if } j \neq k \end{cases} \quad \cdots (6)
$$

---

### 2-2. 제안 방법: Temporal Calibrated Regularization (TCR)

#### (A) Reflection Loss

이전 에폭 $t-1$의 예측값 $f(x_i; \theta_{t-1})$과 원본 노이즈 레이블 $\tilde{y}_i$를 볼록 결합하여 pseudo-label 생성:

$$
\boxed{
L(\theta_t; \theta_{t-1}) = -\frac{1}{n}\sum_{i=1}^{n} \left(\beta\tilde{y}_i + (1-\beta)f(x_i; \theta_{t-1})\right)^{\top} \log f(x_i; \theta_t)
}
= \beta L_o + (1-\beta)L_r \quad \cdots (7)
$$

- $\beta$: 레이블 품질에 대한 신뢰도 (클수록 원본 레이블 의존)
- $L_o$: 표준 CE Loss (clean setting)
- $L_r$: 이전 예측 기반 저항 손실

#### Gradient 분석

TCR의 전체 gradient:

$$
\frac{\partial L}{\partial h} = -(1-\beta)\sum_i \left(f(x_i; \theta_{t-1}) - f(x_i; \theta_t)\right) - \beta\sum_i \left(\tilde{y}_i - f(x_i; \theta_t)\right) \quad \cdots (8)
$$

Reflection Loss 항의 class $l$에 대한 gradient:

$$
\frac{\partial L}{\partial h_l} = f_l^t - f_l^{t-1} \quad \cdots (12)
$$

> **해석:** 현재-이전 에폭 간 예측 차이에 비례하는 gradient로, 급격한 예측 변화를 억제 → 노이즈 레이블 과적합 방지

Bootstrap-soft와의 비교:

$$
\frac{\partial L}{\partial h_l} = f_l\left(\sum_j f_j \log f_j - \log f_l\right) \quad \cdots (9)
$$

Bootstrap-hard와의 비교:

$$
\frac{\partial L}{\partial h_l} = f_l^t - \mathbb{I}\left(l = \arg\max_j f_j^t\right) \quad \cdots (10)
$$

- Bootstrap 방법들은 예측을 더 confident하게 push → Open-set noise에 취약
- TCR의 Eq.(12)는 이런 push 없이도 우수한 성능 달성 → **Open-set noise에 적합**

---

#### (B) Squeeze Technique

학습률 감소 시 저항력 저하 문제(**Resistance Degradation Problem**) 해결:

- 저항력: $f(x_i; \theta_{t-1}) - f(x_i; \theta_t)$
- 학습률이 작아질수록 이 차이가 감소 → 저항력 부족

**Squeeze 함수**:

$$
\boxed{
\text{Squeeze}(\mathbf{p}) = \frac{\mathbf{p}^{\gamma}}{\mathbf{1}^{\top}\mathbf{p}^{\gamma}}
} \quad \cdots (13)
$$

- $(\cdot)^{\gamma}$: 원소별 거듭제곱, $\gamma \geq 1$
- $\gamma \to \infty$이면 one-hot 함수에 수렴
- 가장 큰 예측값을 증폭 → 모델 신뢰도 향상, 노이즈 샘플의 gradient 억제
- 논문에서 $\gamma = 1.1$ 사용

---

### 2-3. 전체 알고리즘 (Algorithm 1 요약)

$$
y_i^* = \begin{cases} \beta\tilde{y}_i + (1-\beta)z_i & \text{if } t \geq 1 \\ \tilde{y}_i & \text{if } t = 0 \end{cases}
$$

$$
L = \sum_{i \in B} {y_i^*}^{\top} \log f(x_i, \theta)
$$

- $t \geq T_s$이면: $z_i \leftarrow \text{Squeeze}(z_i)$
- $z_i$: 이전 에폭에서 저장된 예측값

---

### 2-4. 모델 구조

```
Input x_i ──────────────────► DNN ──► f(x_i; θ_t)
         │                              │
         │                    Delay     │
         │              f(x_i; θ_{t-1}) │
         │                   ▼          │
         │              Squeeze(·)       │
         │                   │          │
         └── y_i (label) ─► [β·ỹ + (1-β)·z_i] ─► Loss
```

- **Temporal Delay**: 이전 에폭 예측값 저장 및 제공
- **Squeeze Function**: 학습률 감소 후 신뢰도 보완
- 기존 모델에 두 모듈만 추가하면 되는 **plug-and-play 방식**

---

### 2-5. 성능 향상

#### CIFAR-10 (PreAct-ResNet-18, 5회 평균)

| Method | Clean | Sym 0.2 | Sym 0.4 | Sym 0.6 | Asym 0.4 |
|---|---|---|---|---|---|
| CE | 94.31 | 80.13 | 60.75 | 39.70 | 76.83 |
| mixup | 94.46 | 92.17 | 88.55 | 80.14 | 86.61 |
| Joint | 91.49 | 90.34 | 89.41 | 79.68 | 90.30 |
| **Ours** | **94.24** | **92.92** | **90.50** | **80.15** | **90.53** |

#### CIFAR-100

| Method | Clean | Sym 0.2 | Sym 0.4 | Sym 0.6 | Asym 0.4 |
|---|---|---|---|---|---|
| CE | 74.30 | 59.38 | 43.71 | 24.53 | 44.42 |
| GCCE | 69.76 | 67.32 | 64.08 | 50.70 | 51.84 |
| **Ours** | **75.65** | **71.68** | **65.59** | **51.23** | **63.15** |

#### Clothing1M (실제 노이즈 데이터)

| CE | Forward | Bilevel | Joint | **Ours** |
|---|---|---|---|---|
| 68.94% | 69.84% | 69.9% | 72.16% | **72.54%** |

---

### 2-6. 한계

1. **$\beta$ 하이퍼파라미터 설정**: 레이블 품질에 대한 사전 지식이 없을 경우 최적값 선택이 어려움 (논문에서는 $\beta=0.1$ 고정)
2. **극심한 노이즈에서의 한계**: Squeeze 없이 Reflection Loss만 사용 시 60% 이상의 균일 노이즈에서는 여전히 어려움
3. **레이블 의존성**: 완전히 잘못된 레이블 비율이 매우 높을 경우 이전 예측 자체가 오염될 수 있음
4. **동적 $\beta$ 미지원**: 학습 진행에 따른 동적 신뢰도 조정 메커니즘 부재
5. **이론적 수렴 보장 부족**: 알고리즘의 수렴성에 대한 엄밀한 이론적 분석이 제한적

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 일반화 성능 향상 메커니즘

TCR이 일반화 성능을 향상시키는 핵심 원리는 **DNN이 early stage에서 학습한 단순 패턴(simple pattern)을 후반부까지 유지**시키는 것이다.

#### (1) 시간적 일관성(Temporal Consistency) 강제

$$
L_r = -\frac{1}{n}\sum_{i=1}^{n} f(x_i; \theta_{t-1})^{\top} \log f(x_i; \theta_t)
$$

이 항은 현재 에폭의 예측이 이전 에폭과 크게 달라지지 않도록 규제함으로써, **노이즈 레이블에 의한 급격한 파라미터 변화를 억제**한다.

#### (2) Pseudo-label의 점진적 정제

$$
y_i^* = \beta\tilde{y}_i + (1-\beta)f(x_i; \theta_{t-1})
$$

- 초기 에폭: $f(x_i; \theta_{t-1})$이 불안정하므로 $\beta$가 상대적으로 큰 역할
- 후기 에폭: DNN이 학습한 패턴이 pseudo-label에 반영되어 **자기 지식 증류(self-knowledge distillation)** 효과 발생

#### (3) Squeeze에 의한 결정 경계 선명화

$$
\text{Squeeze}(\mathbf{p}) = \frac{\mathbf{p}^{\gamma}}{\mathbf{1}^{\top}\mathbf{p}^{\gamma}}, \quad \gamma = 1.1
$$

학습 후반부에 적용되어 예측 분포를 더 sharp하게 만들어 **일반화 가능한 특징 표현**을 강화.

#### (4) 하이퍼파라미터 안정성

실험 결과 $\beta \in [0.05, 0.3]$ 범위에서 안정적인 성능을 보여, Bootstrap 방법들보다 훨씬 **강인한 하이퍼파라미터 감도**를 가짐.

#### (5) 아키텍처 독립성

동일한 $\beta=0.1, \gamma=1.1$ 설정으로 ResNet-44, ResNet-34, PreAct-ResNet-18, WRN 모두에서 일관된 성능 향상:

| Architecture | CE (40% Sym) | **Ours** | 향상폭 |
|---|---|---|---|
| ResNet-44 | 66.17% | **89.60%** | +23.43%p |
| ResNet-34 | 65.45% | **90.72%** | +25.27%p |
| PreAct-ResNet-18 | 60.75% | **90.50%** | +29.75%p |
| WRN | 78.28% | **91.43%** | +13.15%p |

---

## 4. 향후 연구에 미치는 영향 및 고려 사항

### 4-1. 향후 연구에 미치는 영향

#### (1) Self-distillation과의 연결
TCR은 본질적으로 **자기 지식 증류(Self-KD)**의 한 형태로 볼 수 있으며, 이후 연구들에서 teacher-student 패러다임과 결합 가능성을 제시.

#### (2) Semi-supervised Learning과의 융합
$L_r$ 항은 Temporal Ensembling(Laine & Aila, ICLR 2017)과 개념적으로 연결되어, **반지도 학습 환경에서의 일관성 정규화** 연구에 영감을 줌.

#### (3) 레이블 노이즈 학습의 단순화 방향 제시
복잡한 두 단계 분리, meta-learning, bilevel optimization 없이도 경쟁력 있는 성능 달성 → **단순하고 확장 가능한 방법론**의 효용성 입증.

#### (4) Open-set Noise 연구
Eq.(12)의 gradient 특성이 Open-set noise에 유리함을 보여, 이후 **현실적 노이즈 시나리오** 연구 방향 제시.

---

### 4-2. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ 아래는 논문 발표(2020.07) 이후의 주요 관련 연구 동향을 일반적으로 알려진 내용 기반으로 기술하며, 특정 수치는 해당 논문 원문 확인이 필요합니다.

#### 주요 후속 연구 동향

| 연구 방향 | 대표 연구 | TCR과의 관계 |
|---|---|---|
| **혼합 훈련(Mixup + Noise)** | SOP (NeurIPS 2022) | TCR의 pseudo-label 방식과 결합 가능 |
| **샘플 선택 + 반지도 학습** | DivideMix (ICLR 2020) | TCR보다 복잡하지만 성능 우수 |
| **대조 학습 기반** | DISC (CVPR 2023) | 특징 공간에서의 정제로 TCR 보완 |
| **동적 임계값 기반** | Jo-SRC (CVPR 2021) | TCR의 고정 $\beta$의 한계 해결 시도 |
| **Noise-Robust Loss** | Normalized Loss (ICML 2020) | TCR과 직교적(orthogonal) 결합 가능 |

#### DivideMix (Li et al., ICLR 2020)와의 비교

| 항목 | TCR | DivideMix |
|---|---|---|
| 방법론 | Pseudo-label + Temporal | GMM + Semi-supervised |
| 복잡도 | 낮음 | 높음 |
| CIFAR-10 40% Sym | 90.50% | ~95.01% |
| 구현 난이도 | 매우 쉬움 | 복잡함 |
| 추가 clean data 필요 | 불필요 | 불필요 |

> **참고**: DivideMix의 구체적 수치는 원문(Li et al., ICLR 2020) 확인 필요.

---

### 4-3. 향후 연구 시 고려할 점

#### (1) 동적 $\beta$ 스케줄링

```math
\beta_t = \beta_0 \cdot \phi(t, \text{noise\_ratio})
```

- 학습 단계와 노이즈 비율을 반영하는 적응형 $\beta$ 설계 필요

#### (2) 노이즈 비율 추정과의 결합
- 실제 환경에서는 노이즈 비율 $\eta$를 알 수 없으므로, **자동 노이즈 비율 추정** 모듈과의 통합 연구 필요

#### (3) 대규모 언어 모델(LLM) 환경 적용
- RLHF 등 인간 피드백 기반 학습에서의 **선호도 노이즈** 문제에 TCR 아이디어 적용 가능성

#### (4) 클래스 불균형 노이즈 환경
- 클래스별 노이즈 비율이 다를 경우 ($T_{jk}$가 class에 따라 상이) 단일 $\beta$의 한계 → **클래스별 $\beta$ 또는 인스턴스별 $\beta$** 연구 필요

#### (5) Squeeze의 이론적 분석
- $\gamma$ 값의 최적화에 대한 이론적 근거 부재 → **정보 이론적 관점**에서의 분석 필요

#### (6) 타 정규화 기법과의 결합
$$L_{\text{total}} = L_{\text{TCR}} + \lambda_1 L_{\text{mixup}} + \lambda_2 L_{\text{contrastive}}$$
- Mixup, 대조 학습, 레이블 스무딩과의 체계적 결합 실험 필요

---

## 참고 자료

1. **[주 논문]** Dongxian Wu, Yisen Wang, Zhuobin Zheng, Shu-Tao Xia, "Temporal Calibrated Regularization for Robust Noisy Label Learning," *arXiv:2007.00240v1*, 2020.

2. **[인용 참고]** Patrini et al., "Making Deep Neural Networks Robust to Label Noise: A Loss Correction Approach," *CVPR*, 2017. [Forward Loss]

3. **[인용 참고]** Zhang & Sabuncu, "Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels," *NeurIPS*, 2018. [GCCE]

4. **[인용 참고]** Tanaka et al., "Joint Optimization Framework for Learning with Noisy Labels," *CVPR*, 2018. [Joint]

5. **[인용 참고]** Laine & Aila, "Temporal Ensembling for Semi-Supervised Learning," *ICLR*, 2017. [Temporal Ensemble]

6. **[인용 참고]** Reed et al., "Training Deep Neural Networks on Noisy Labels with Bootstrapping," *ICLR*, 2015. [Bootstrap]

7. **[인용 참고]** Arpit et al., "A Closer Look at Memorization in Deep Networks," *ICML*, 2017. [Two-stage learning]

8. **[인용 참고]** Ma et al., "Normalized Loss Functions for Deep Learning with Noisy Labels," *arXiv:2006.13554*, 2020.

9. **[비교 참고]** Li et al., "DivideMix: Learning with Noisy Labels as Semi-supervised Learning," *ICLR*, 2020.
