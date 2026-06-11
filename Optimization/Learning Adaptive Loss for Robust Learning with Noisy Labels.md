# Learning Adaptive Loss for Robust Learning with Noisy Labels

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

기존의 강건 손실 함수(Robust Loss Function)들은 **하이퍼파라미터를 수동으로 설정**해야 하는 한계가 있으며, 이로 인해 비볼록(non-convex) 최적화 문제에서 **일반화 성능이 낮은 해**에 수렴할 위험이 있다. 본 논문은 **메타 학습(Meta-Learning)** 기반의 적응적 하이퍼파라미터 학습 전략(Adaptive Robust Loss, ARL)을 통해 이 문제를 해결할 수 있음을 주장한다.

### 주요 기여 (3가지)

| 기여 | 설명 |
|------|------|
| **① 최초의 적응적 강건 손실 학습** | 명시적이고 간결한 형태의 강건 손실 함수에 대해 하이퍼파라미터를 데이터로부터 자동 학습하는 최초의 연구 |
| **② 4종 SOTA 강건 손실 통합** | GCE, SL, Bi-Tempered, PolySoft를 ARL 프레임워크에 통합하고, Bi-Tempered와 PolySoft의 이론적 강건성(Loss Bounded Condition) 증명 |
| **③ 일반화 성능 향상 실증** | 신중하게 튜닝된 하이퍼파라미터보다도 더 우수한 일반화 성능을 달성함을 실험으로 입증 |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

**문제 1: 하이퍼파라미터 수동 설정의 한계**

기존 강건 손실 함수들은 잡음에 대한 강건성 정도를 조절하는 하이퍼파라미터를 포함한다. 이를 교차 검증(Cross-Validation)으로 탐색하는 것은 비효율적이며 실용성이 낮다.

**문제 2: 비볼록 최적화로 인한 일반화 성능 저하**

강건 손실 함수의 복잡한 형태와 딥 네트워크의 비볼록 구조가 결합되면, 올바른 하이퍼파라미터를 설정하더라도 **일반화 성능이 낮은 지역 최솟값(local minimum)** 에 수렴할 수 있다.

---

### 2.2 제안하는 방법 (수식 포함)

#### 기본 설정

$c$-class 분류 문제를 고려한다. 네트워크 함수 $f(\mathbf{x}; \mathbf{w}): \mathcal{X} \rightarrow \mathbb{R}^c$에 대해 경험적 위험(Empirical Risk)은 다음과 같이 정의된다:

$$\mathcal{L}(D, \mathbf{w}) = \frac{1}{N} \sum_{i=1}^{N} \mathcal{L}(f(\mathbf{x}_i; \mathbf{w}), \mathbf{y}_i)$$

Cross Entropy(CE) 손실:

$$\mathcal{L}_{CE}(D, \mathbf{w}) = -\frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{c} y_{ij} \log f_j(\mathbf{x}_i; \mathbf{w}) \tag{1}$$

---

#### 통합된 4종 강건 손실 함수

**① Generalized Cross Entropy (GCE):**

$$\mathcal{L}_{GCE}(D, \mathbf{w}; q) = \frac{1}{N} \sum_{i=1}^{N} \frac{(1 - f_{j_i}(\mathbf{x}_i)^q)}{q}, \quad q \in (0, 1] \tag{2}$$

- $q \to 0$: CE 손실로 수렴
- $q = 1$: MAE 손실과 동일

**② Symmetric Cross Entropy (SL):**

$$\mathcal{L}_{SL}(D, \mathbf{w}; \gamma_1, \gamma_2) = \gamma_1 \mathcal{L}_{CE} + \gamma_2 \mathcal{L}_{RCE} \tag{3}$$

여기서 역방향 교차 엔트로피(RCE)는 다음과 같다:

$$\mathcal{L}_{RCE} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{j \neq j_i} A f_j(\mathbf{x}_i; \mathbf{w}), \quad A < 0$$

**③ Bi-Tempered Logistic Loss:**

$$\mathcal{L}_{Bi}(D, \mathbf{w}; t_1, t_2) = -\frac{1}{N} \sum_{i=1}^{N} \left[ \log_{t_1} \hat{f}_{j_i, t_2}(\mathbf{x}_i) + \frac{1}{2-t_1}\left(1 - \sum_{j=1}^{c} \hat{f}_{j, t_2}(\mathbf{x}_i)^{2-t_1}\right) \right] \tag{4}$$

- $0 \leq t_1 < 1$, $t_2 > 1$
- $\hat{f}_{j,t} = \exp_t(z_j - \gamma_t(\mathbf{z}))$

**④ Polynomial Soft Weighting Loss (PolySoft):**

$$\mathcal{L}_{Poly}(D, \mathbf{w}; \lambda, d) = \begin{cases} \frac{(d-1)\lambda}{d} \left[1 - \left(1 - \frac{\mathcal{L}_{CE}(D,\mathbf{w})}{\lambda}\right)^{\frac{d}{d-1}}\right], & \mathcal{L}_{CE} < \lambda \\ \frac{(d-1)\lambda}{d}, & \mathcal{L}_{CE} \geq \lambda \end{cases} \tag{5}$$

---

#### 메타 학습 목적 함수 (ARL의 핵심)

하이퍼파라미터 집합 $\Lambda$에 대해 **이중 수준 최적화(Bi-level Optimization)** 를 수행한다:

**하위 수준 (Inner-level)** — 네트워크 파라미터 최적화:

$$\mathbf{w}^*(\Lambda) = \arg\min_{\mathbf{w}} \mathcal{L}_{Train}(D, \mathbf{w}; \Lambda) \tag{6}$$

**상위 수준 (Outer-level)** — 하이퍼파라미터 메타 최적화:

$$\Lambda^* = \arg\min_{\Lambda} \mathcal{L}_{Meta}(D_{meta}, \mathbf{w}^*(\Lambda)) \tag{7}$$

여기서 $D_{meta}$는 소량의 **정제된 클린 데이터(clean meta-data)** 이며 $M \ll N$이다.

---

#### 온라인 근사 알고리즘 (Algorithm 1: ARL)

계산 효율성을 위해 MAML 방식의 1-step 근사를 사용한다:

**Step 1: 가상 네트워크 파라미터 계산**

$$\tilde{\mathbf{w}}^{(t)}(\Lambda) = \mathbf{w}^{(t-1)} - \alpha \nabla_{\mathbf{w}} \mathcal{L}_{Train}(D_n, \mathbf{w}; \Lambda)\bigg|_{\mathbf{w}^{(t-1)}} \tag{9}$$

**Step 2: 하이퍼파라미터 업데이트 (메타 데이터 기반)**

$$\Lambda^{(t)} = \Lambda^{(t-1)} - \beta \nabla_{\Lambda} \mathcal{L}_{Meta}(D_m, \tilde{\mathbf{w}}^{(t)}(\Lambda))\bigg|_{\Lambda^{(t-1)}} \tag{8}$$

**Step 3: 네트워크 파라미터 업데이트**

$$\mathbf{w}^{(t)} = \mathbf{w}^{(t-1)} - \alpha \nabla_{\mathbf{w}} \mathcal{L}_{Train}(D_n, \mathbf{w}; \Lambda^{(t)})\bigg|_{\mathbf{w}^{(t-1)}} \tag{10}$$

---

### 2.3 이론적 강건성 증명

본 논문은 Bi-Tempered와 PolySoft에 대해 **Loss Bounded Condition(손실 유계 조건)** 을 새롭게 증명한다.

**Theorem 1 (PolySoft의 강건성):**

대칭 잡음 $\eta \leq 1 - \frac{1}{c}$, $\lambda \geq \log c$, $d > 1$ 조건 하에서:

```math
0 \leq R_{\mathcal{L}}^{\eta}(f^*) - R_{\mathcal{L}}^{\eta}(\hat{f}) \leq A, \quad A' \leq R_{\mathcal{L}}(f^*) - R_{\mathcal{L}}(\hat{f}) \leq 0
```

$$\text{where } A = \frac{c(d-1)\eta}{d(c-1)}(\lambda - \log c) \geq 0$$

특히, $\lambda = \log c$ 일 때 $R_{\mathcal{L}}^{\eta}(f^*) = R_{\mathcal{L}}^{\eta}(\hat{f})$로 **잡음 허용(Noise Tolerant)** 이 성립한다.

**Theorem 2 (Bi-Tempered의 강건성):**

대칭 잡음 $\eta \leq 1 - \frac{1}{c}$, $0 \leq t_1 < 1$, $t_2 > 1$ 조건 하에서 Loss Bounded Condition을 만족한다:

$$A = \frac{\eta}{1-t_1} - \frac{\eta(c - c^{t_1})}{(c-1)(1-t_1)(2-t_1)} > 0$$

---

### 2.4 모델 구조

| 구성 요소 | 세부 내용 |
|-----------|-----------|
| **분류기 네트워크** | ResNet-32 (CIFAR-10/100), PreAct ResNet-18 (TinyImageNet), ResNet-50 (Clothing1M) |
| **메타 데이터** | CIFAR: 1,000장 클린 이미지; T-ImageNet: 클래스당 10장 |
| **하이퍼파라미터 최적화** | SGD 기반, 분류기와 동일한 학습률 스케줄 사용 |
| **구현 프레임워크** | PyTorch (자동 미분을 통한 효율적 그래디언트 계산) |

---

### 2.5 성능 향상

#### CIFAR-10 (ResNet-32, Symmetric Noise)

| 방법 | $\eta=0$ | $\eta=0.2$ | $\eta=0.4$ | $\eta=0.6$ |
|------|---------|-----------|-----------|-----------|
| CE | 92.89 | 76.83 | 70.77 | 63.21 |
| PolySoft | 91.40 | 87.53 | 81.49 | 75.87 |
| **A-PolySoft** | **92.12** | **89.73** | **87.22** | **82.49** |
| GCE | 90.03 | 88.51 | 85.48 | 81.29 |
| **A-GCE** | 91.47 | 89.07 | 86.36 | 81.64 |
| Meta-Weight-Net | 92.04 | 89.19 | 86.10 | 81.31 |

#### Clothing1M (실세계 잡음 데이터)

| CE | Forward | DMI | MN-Net | PolySoft | **A-PolySoft** |
|----|---------|-----|--------|----------|----------------|
| 68.94 | 70.83 | 72.46 | 73.72 | 69.96 | **73.76** |

---

### 2.6 한계점

1. **클린 메타 데이터 의존성**: 소량이지만 정제된 클린 데이터가 반드시 필요하다. 실제 환경에서 이를 확보하기 어려운 경우가 존재한다.
2. **계산 비용 증가**: 이중 수준 최적화로 인해 일반 학습 대비 추가적인 역전파 연산이 필요하다.
3. **비대칭·계층적 잡음에서 일부 한계**: 일부 손실 함수(예: A-SL, A-GCE)는 고비율 비대칭 잡음에서 A-PolySoft 대비 낮은 성능을 보인다.
4. **하이퍼파라미터 탐색 공간 정의 필요**: ARL이 하이퍼파라미터를 자동 학습하더라도, 각 손실 함수의 적절한 초기값 설정이 수렴 속도에 영향을 미친다.
5. **대규모 데이터에서의 확장성 검증 부족**: Clothing1M(100만 장)에서 제한적으로 검증되었으나, 더 대규모 데이터셋에 대한 실험이 부재하다.

---

## 3. 일반화 성능 향상 가능성 심층 분석

### 3.1 상호 개선(Mutual Amelioration)의 핵심 역할

논문의 Ablation Study(Fig. 4)는 일반화 성능 향상의 메커니즘을 명확히 보여준다:

| 전략 | 설명 | 성능 |
|------|------|------|
| **SL** | 교차 검증으로 최적 튜닝된 고정 하이퍼파라미터 | 기준선 |
| **SL-Opt1** | A-SL이 학습한 하이퍼파라미터를 고정하여 SL 재학습 | SL보다 **하락** |
| **SL-Opt2** | 각 스텝에서 A-SL의 하이퍼파라미터로 SL 초기화 후 학습 | SL보다 **향상** |
| **A-SL** | 하이퍼파라미터와 네트워크 파라미터 동시 학습 | **최고 성능** |

이 결과는 다음을 시사한다:

> **핵심 인사이트**: 일반화 성능 향상은 단순히 "더 나은 하이퍼파라미터"를 찾는 것이 아니라, **하이퍼파라미터와 네트워크 파라미터가 상호적으로 co-evolve하는 동적 학습 과정** 자체에서 비롯된다.

수식으로 표현하면, ARL은 다음의 공동 최적화 궤적을 따른다:

```math
(\Lambda^*, \mathbf{w}^*) = \arg\min_{\Lambda, \mathbf{w}} \left[ \mathcal{L}_{Train}(D, \mathbf{w}; \Lambda) + \mathcal{L}_{Meta}(D_{meta}, \mathbf{w}(\Lambda)) \right]
```

이 궤적은 고정된 하이퍼파라미터로는 도달할 수 없는 **더 평탄한(flatter) 손실 landscape 영역**으로 수렴함으로써 일반화 성능을 향상시킨다.

### 3.2 잡음 비율 적응적 손실 형태 (Fig. 1 분석)

학습된 적응 손실은 다음과 같은 바람직한 특성을 보인다:

$$\mathcal{L}_{adaptive}(\ell) \approx \begin{cases} \mathcal{L}_{CE}(\ell) & \ell \text{이 작을 때 (클린 샘플)} \\ \text{상수} & \ell \text{이 클 때 (잡음 샘플)} \end{cases}$$

- 잡음 비율이 높을수록($\eta = 60\%$) 손실 함수가 더 일찍 평탄해진다.
- 이는 잡음 샘플에 대한 기울기를 억제하여 **그래디언트 기반 학습 시 잡음의 영향을 자동으로 감소**시킨다.

### 3.3 표현 학습 품질 (Fig. 3)

t-SNE 시각화에서 A-Bi-Tempered는 60% 잡음 조건에서도 **CE(클린 데이터)와 유사한 수준의 분리된 클러스터 구조**를 학습한다. 이는 ARL이 단순히 분류 정확도뿐 아니라 **특징 표현(feature representation)의 질** 자체를 향상시킴을 의미한다.

### 3.4 샘플 가중치 분포 (Fig. 2)

A-PolySoft는 훈련이 진행됨에 따라 클린 샘플과 잡음 샘플의 가중치를 점차 명확하게 구분한다:

- 클린 샘플: 가중치 → 1에 가까운 값
- 잡음 샘플: 가중치 → 0에 가까운 값

이 특성은 Meta-Weight-Net보다 **더 명확한 샘플 구분**을 달성하며, 이것이 일반화 성능 우위의 근본적 원인이다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려 사항

### 4.1 연구 영향

**① 메타 학습과 강건 손실의 결합 패러다임 확립**

ARL은 하이퍼파라미터 최적화, 메타 학습, 강건 손실 설계를 하나의 통합 프레임워크로 결합하는 새로운 패러다임을 제시한다. 이는 이후 연구들이 "손실 함수 설계"와 "학습 알고리즘 설계"를 분리하지 않고 **공동 최적화** 문제로 접근하는 방향을 촉진시켰다.

**② 비볼록 최적화에서 일반화 향상을 위한 새로운 시각**

Ablation Study의 결과는 단순히 좋은 하이퍼파라미터를 찾는 것보다 **동적 co-evolution이 더 좋은 일반화 해를 탐색**한다는 것을 보여준다. 이는 손실 landscape 이론(loss landscape theory) 연구에 새로운 관점을 제공한다.

**③ 이론적 강건성 분석의 확장**

Bi-Tempered와 PolySoft의 Loss Bounded Condition 증명은 새로운 강건 손실 설계 시 이론적 정당성을 확인하는 방법론적 프레임워크를 제공한다.

---

### 4.2 2020년 이후 관련 최신 연구 비교 분석

> **주의**: 아래 연구들은 본 논문 발표(2020년 2월) 이후의 연구 흐름을 나타내며, 일부는 제가 학습 데이터에 포함된 범위 내에서 언급하는 것임을 밝힙니다. 각 논문의 정확한 수치나 세부 내용은 원문 확인을 권장합니다.

#### 비교 분석표

| 연구 | 핵심 방법 | ARL과의 관계 | 주요 차별점 |
|------|-----------|-------------|------------|
| **DivideMix** (Li et al., ICLR 2020) | GMM으로 클린/잡음 샘플 분리 후 MixMatch 적용 | 보완적 | 샘플 분리 중심; 손실 형태는 고정된 CE 사용 |
| **ELR (Early-Learning Regularization)** (Liu et al., NeurIPS 2020) | 초기 학습 예측값을 정규화 항으로 활용 | 보완적 | 정규화 관점에서 접근; 하이퍼파라미터 적응 없음 |
| **CORES²** (Cheng et al., ICML 2021) | 신뢰도 기반 샘플 선택 | 보완적 | 클린 샘플 선별 후 CE 적용 |
| **Normalized Loss Functions** (Ma et al., ICML 2020) | 손실 정규화를 통한 대칭성 부여 | 직접 관련 | 하이퍼파라미터 없는 강건 손실 설계 |
| **SOP (Sample-dependent Optimal Provenance)** (Liu et al., ICML 2022) | 샘플별 최적 프로비넌스 학습 | 발전적 관계 | 샘플 수준의 더 세밀한 가중치 학습 |

**핵심 관찰**: 2020년 이후 연구들은 크게 두 방향으로 분기되었다:

1. **ARL의 방향 계승**: 메타 학습을 활용한 손실/가중치 자동 학습 (e.g., 손실 함수 자체의 구조 학습)
2. **대조적 접근**: 클린/잡음 샘플 분리 후 반지도 학습 적용 (DivideMix 계열)

---

### 4.3 향후 연구 시 고려 사항

**① 클린 메타 데이터 의존성 완화**

현재 ARL은 소량의 클린 메타 데이터를 가정한다. 향후 연구에서는:

- 자기 지도 학습(Self-supervised Learning)으로 클린 샘플을 **자동 식별**하는 방법과의 결합
- **확률적 메타 데이터 선택**: 신뢰도가 높은 샘플을 동적으로 메타 데이터로 활용

**② 인스턴스 의존 잡음(Instance-Dependent Noise) 처리**

논문은 대칭적/비대칭적 잡음을 주로 다루지만, 실제 데이터에서는 개별 샘플의 특성에 따라 잡음 분포가 다르다(Instance-Dependent Label Noise). ARL의 메타 학습 프레임워크를 **샘플별 적응적 하이퍼파라미터**로 확장하는 연구가 필요하다.

**③ 대규모 데이터셋 확장성**

이중 수준 최적화의 계산 복잡도는 대규모 데이터셋에서 병목이 될 수 있다. 효율적인 근사 방법(예: 암묵적 미분(Implicit Differentiation) 기반 이중 수준 최적화)의 적용을 고려해야 한다.

**④ 비전 외 도메인 적용**

ARL은 이미지 분류에 집중되어 있으나, **자연어처리(NLP), 의료 데이터, 그래프 데이터** 등 잡음 레이블 문제가 빈번한 도메인으로의 확장 연구가 요구된다.

**⑤ 손실 경관(Loss Landscape) 이론과의 연결**

왜 동적 co-evolution이 더 나은 일반화 해를 찾는지에 대한 이론적 분석이 부족하다. Sharp/Flat minima 이론, PAC-Bayes bound 등을 활용한 **수학적 설명**이 향후 연구의 중요한 방향이다.

**⑥ 기초 모델(Foundation Model)과의 통합**

Vision-Language Model(VLM), Large Language Model(LLM) 파인튜닝 시 잡음 레이블 문제가 증가하고 있으며, ARL의 적응적 손실 학습 아이디어를 이러한 대규모 모델의 파인튜닝 안정성 향상에 적용하는 연구가 유망하다.

---

## 참고 자료

**주 논문:**
- Jun Shu, Qian Zhao, Keyu Chen, Zongben Xu, Deyu Meng. "Learning Adaptive Loss for Robust Learning with Noisy Labels." arXiv:2002.06482v1, 2020.

**논문 내 주요 참고문헌:**
- [1] Amid et al. "Robust Bi-Tempered Logistic Loss Based on Bregman Divergences." NeurIPS, 2019.
- [7] Finn et al. "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks (MAML)." ICML, 2017.
- [10] Ghosh et al. "Robust Loss Functions under Label Noise for Deep Neural Networks." AAAI, 2017.
- [13] Gong et al. "Decomposition-based Evolutionary Multiobjective Optimization to Self-Paced Learning." IEEE TEC, 2018.
- [42] Shu et al. "Meta-Weight-Net: Learning an Explicit Mapping for Sample Weighting." NeurIPS, 2019.
- [53] Wang et al. "Symmetric Cross Entropy for Robust Learning with Noisy Labels." ICCV, 2019.
- [57] Xu et al. "L_DMI: An Information-Theoretic Noise-Robust Loss Function." NeurIPS, 2019.
- [59] Zhang & Sabuncu. "Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels." NeurIPS, 2018.

**2020년 이후 관련 연구 (참고):**
- Li et al. "DivideMix: Learning with Noisy Labels as Semi-supervised Learning." ICLR, 2020.
- Liu et al. "Early-Learning Regularization Prevents Memorization of Noisy Labels." NeurIPS, 2020.
- Ma et al. "Normalized Loss Functions for Deep Learning with Noisy Labels." ICML, 2020.
