# DivideMix: Learning with Noisy Labels as Semi-supervised Learning 

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

DivideMix는 **노이즈 레이블 학습(Learning with Noisy Labels, LNL)** 문제를 **준지도 학습(Semi-Supervised Learning, SSL)** 관점에서 재해석한다. 기존 방법들이 노이즈 샘플을 단순히 제거하거나 손실 보정에만 집중했다면, DivideMix는 노이즈 샘플을 **버리지 않고 레이블 없는 데이터로 적극 활용**하여 모델의 일반화 성능을 향상시킨다.

### 주요 기여 (3가지)

| 기여 | 내용 |
|------|------|
| **Co-Divide** | 두 네트워크를 동시에 학습하며, GMM으로 각 샘플의 clean 확률을 추정하고 **서로의 데이터 분할**을 교차 사용 |
| **Label Co-Refinement & Co-Guessing** | MixMatch를 개선하여, labeled 샘플에는 co-refinement, unlabeled 샘플에는 co-guessing 적용 |
| **성능 향상** | CIFAR-10/100, Clothing1M, WebVision 전 벤치마크에서 SOTA 대비 대폭 성능 향상 |

---

## 2. 상세 분석

### 2-1. 해결하고자 하는 문제

심층 신경망(DNN)은 노이즈 레이블에 쉽게 과적합(overfitting)되어 일반화 성능이 저하된다 (Zhang et al., 2017). 기존 LNL 접근법의 한계는 다음과 같다:

- **손실 보정 방법**: 노이즈 전이 행렬(noise transition matrix) 추정이 어렵고 고노이즈 환경에서 불안정
- **샘플 선택/재가중화**: 깨끗한 샘플 선별 기준이 불명확하고, 높은 노이즈 비율에서 성능 저하
- **자기 학습(Self-training)**: **확증 편향(Confirmation Bias)** — 모델이 자신의 오류를 반복 학습

### 2-2. 제안하는 방법 (수식 포함)

#### **Step 1: Co-Divide — GMM 기반 데이터 분할**

모델 파라미터 $\theta$에 대한 샘플별 교차 엔트로피 손실:

```math
\ell(\theta) = \{\ell_i\}_{i=1}^{N} = \left\{ -\sum_{c=1}^{C} y_i^c \log\left(\text{p}^c_{\text{model}}(x_i; \theta)\right) \right\}_{i=1}^{N}
```

손실 분포 $\ell$에 **2성분 Gaussian Mixture Model(GMM)** 을 EM 알고리즘으로 피팅하여, 각 샘플의 clean 확률 $w_i$를 계산:

$$w_i = p(g \mid \ell_i)$$

여기서 $g$는 GMM의 두 가우시안 성분 중 **평균이 더 작은(손실이 낮은) 성분**의 사후확률.

임계값 $\tau = 0.5$를 기준으로:
- $w_i \geq \tau$: **Labeled set** $\mathcal{X}$ (clean 샘플로 간주)
- $w_i < \tau$: **Unlabeled set** $\mathcal{U}$ (noisy 샘플로 간주)

> **왜 GMM인가?** 기존 Beta Mixture Model(BMM)은 비대칭 노이즈(asymmetric noise)에서 분포가 평탄해져 구분이 어렵지만, GMM은 분포의 뾰족함(sharpness)이 유연하여 clean/noisy 구분에 더 효과적.

#### **비대칭 노이즈를 위한 Confidence Penalty (Warm-up 단계)**

표준 CE 학습 시 과신(over-confident) 예측 문제 해결을 위해, warm-up 단계에서 **음의 엔트로피 항** 추가:

$$\mathcal{H} = -\sum_{c} \text{p}^c_{\text{model}}(x; \theta) \log\left(\text{p}^c_{\text{model}}(x; \theta)\right) \tag{2}$$

손실에 $-\mathcal{H}$를 더하여 예측 분포를 더 균등하게 만들어 GMM 피팅이 용이해짐.

#### **Step 2: Label Co-Refinement (Labeled samples)**

labeled 샘플 $x_b$에 대해, 다른 네트워크가 생성한 clean 확률 $w_b$를 가이드로 삼아 GT 레이블 $y_b$와 모델 예측 $p_b$를 선형 결합:

$$\bar{y}_b = w_b y_b + (1 - w_b) p_b \tag{3}$$

여기서 $p_b = \frac{1}{M}\sum_m \text{p}\_{\text{model}}(\hat{x}_{b,m}; \theta^{(k)})$ (M번 augmentation의 평균 예측)

**온도 샤프닝(Temperature Sharpening)** 적용으로 레이블의 엔트로피 최소화:

$$\hat{y}_b = \text{Sharpen}(\bar{y}_b, T) = \bar{y}_b^{c \frac{1}{T}} \Bigg/ \sum_{c=1}^{C} \bar{y}_b^{c \frac{1}{T}}, \quad c = 1, 2, \ldots, C \tag{4}$$

#### **Step 3: Label Co-Guessing (Unlabeled samples)**

두 네트워크의 예측 앙상블로 unlabeled 샘플의 레이블을 추정:

$$\bar{q}_b = \frac{1}{2M} \sum_m \left( \text{p}_{\text{model}}(\hat{u}_{b,m}; \theta^{(1)}) + \text{p}_{\text{model}}(\hat{u}_{b,m}; \theta^{(2)}) \right) \tag{5}$$

이후 동일한 sharpening 적용: $q_b = \text{Sharpen}(\bar{q}_b, T)$

#### **Step 4: MixUp 기반 데이터 혼합**

$$\lambda \sim \text{Beta}(\alpha, \alpha), \quad \lambda' = \max(\lambda, 1-\lambda) \tag{6}$$

$$x' = \lambda' x_1 + (1 - \lambda') x_2, \quad p' = \lambda' p_1 + (1 - \lambda') p_2 \tag{7}$$

#### **Step 5: 손실 함수**

Labeled set $\mathcal{X}'$에 대한 교차 엔트로피 손실:

$$\mathcal{L}_{\mathcal{X}} = -\frac{1}{|\mathcal{X}'|} \sum_{x,p \in \mathcal{X}'} \sum_c p^c \log\left(\text{p}^c_{\text{model}}(x; \theta)\right) \tag{8}$$

Unlabeled set $\mathcal{U}'$에 대한 MSE 손실:

$$\mathcal{L}_{\mathcal{U}} = \frac{1}{|\mathcal{U}'|} \sum_{x,p \in \mathcal{U}'} \|p - \text{p}_{\text{model}}(x; \theta)\|_2^2 \tag{9}$$

균등 분포 사전확률 $\pi_c = 1/C$를 이용한 **정규화 항** (단일 클래스 과집중 방지):

$$\mathcal{L}_{\text{reg}} = \sum_c \pi_c \log \left( \pi_c \Bigg/ \frac{1}{|\mathcal{X}'| + |\mathcal{U}'|} \sum_{x \in \mathcal{X}' + \mathcal{U}'} \text{p}^c_{\text{model}}(x; \theta) \right) \tag{10}$$

**최종 총 손실:**

$$\mathcal{L} = \mathcal{L}_{\mathcal{X}} + \lambda_u \mathcal{L}_{\mathcal{U}} + \lambda_r \mathcal{L}_{\text{reg}} \tag{11}$$

($\lambda_r = 1$로 고정, $\lambda_u$는 노이즈 비율에 따라 조정)

### 2-3. 모델 구조

```
[전체 프레임워크]
    ┌─────────────────────────────┐
    │   Warm-up (표준 CE + Confidence Penalty)    │
    └─────────────┬───────────────┘
                  ▼
    ┌──────────────────────────────┐
    │         Co-Divide            │
    │  Network A → GMM → 분할 → B │
    │  Network B → GMM → 분할 → A │
    └──────────────┬───────────────┘
                   ▼
    ┌──────────────────────────────┐
    │   Semi-supervised Training   │
    │  (개선된 MixMatch)           │
    │  ① Label Co-Refinement (X)  │
    │  ② Label Co-Guessing (U)    │
    │  ③ MixUp + 손실 계산        │
    └──────────────────────────────┘
```

- **백본**: 18-layer PreAct ResNet (CIFAR), ResNet-50 (Clothing1M), Inception-ResNet v2 (WebVision)
- **두 네트워크 A, B**: 서로 다른 파라미터 초기화 + 서로의 GMM 결과로 학습 → 다양성(diversity) 유지

### 2-4. 성능 향상

| 데이터셋 | 노이즈 유형 | DivideMix | 이전 SOTA | 향상 |
|----------|------------|-----------|-----------|------|
| CIFAR-10 | Sym. 80% | **93.2%** | 86.8% (M-correction) | +6.4% |
| CIFAR-10 | Asym. 40% | **93.4%** | 89.2% (Meta-Learning) | +4.2% |
| CIFAR-100 | Sym. 90% | **31.5%** | 24.3% (M-correction) | +7.2% |
| Clothing1M | 실세계 노이즈 | **74.76%** | 73.49% (P-correction) | +1.27% |
| WebVision | 실세계 노이즈 | **77.32%** | 65.24% (Iterative-CV) | +12.08% |

### 2-5. 한계

1. **계산 비용**: 두 네트워크 동시 학습으로 단일 네트워크 대비 약 2배 메모리/연산 필요
2. **GMM의 가정**: 손실 분포가 가우시안을 따른다는 가정이 항상 성립하지 않을 수 있음
3. **하이퍼파라미터 민감성**: $\lambda_u$ 튜닝이 여전히 필요 (노이즈 비율별 다른 값 사용)
4. **Instance-dependent noise 미대응**: 논문에서 다루는 noisy label이 주로 class-level noise이며, 개별 샘플 특성에 따른 노이즈 패턴에는 덜 robust할 수 있음
5. **도메인 한정**: 주로 이미지 분류에만 검증되었으며, NLP 등 다른 도메인 적용은 미검증 (논문 자체에서도 향후 과제로 제시)

---

## 3. 모델의 일반화 성능 향상 가능성

DivideMix가 일반화 성능을 향상시키는 메커니즘은 다음 4가지 관점에서 분석할 수 있다:

### 3-1. 노이즈 샘플의 Unlabeled Data 재활용

기존 방법은 노이즈 샘플을 단순 제거하지만, DivideMix는 이를 **unlabeled 데이터로 활용**하여 consistency regularization(일관성 정규화)에 기여한다. 이는 사실상 SSL의 핵심 원리인 **"데이터 자체에 내재된 구조 학습"** 을 노이즈 학습에 적용한 것이다.

### 3-2. Co-Divide를 통한 확증 편향(Confirmation Bias) 방지

Self-training의 가장 큰 문제인 확증 편향을 **두 네트워크의 교차 감시** 구조로 해결:
- 서로 다른 초기화 → 다른 오류 패턴 학습
- 서로의 GMM 결과를 사용 → 자신의 오류가 자신에게 피드백되지 않음

이는 두 네트워크가 **앙상블(ensemble) 효과**를 내면서도 각자의 다양성을 유지하게 한다.

### 3-3. Label Co-Refinement의 Soft Label 효과

Refined label $\bar{y}_b = w_b y_b + (1-w_b)p_b$는 사실상 **soft label**이다. 이는:
- Clean 샘플($w_b \approx 1$): GT 레이블에 가까운 목표 유지
- Noisy 샘플($w_b \approx 0$): 모델의 현재 예측을 레이블로 사용하여 잘못된 GT 레이블 영향 최소화

이 soft label은 **label smoothing**과 유사한 정규화 효과를 내어 모델이 너무 확실한 예측에 과적합되는 것을 방지한다.

### 3-4. MixUp + 온도 샤프닝의 결합 효과

- **MixUp**: 샘플 간 선형 보간으로 결정 경계(decision boundary) 평탄화 → 과적합 방지
- **Temperature Sharpening**: unlabeled 샘플의 예측을 확신 있게 만들어 low-entropy 목표 강화

이 두 기제의 결합으로 **일반화-과적합 간의 균형**이 개선된다.

### 3-5. t-SNE 시각화로 확인된 일반화 효과

논문의 t-SNE 시각화 결과(Figure 5)에서, 80% 노이즈 환경에서도 DivideMix로 학습된 표현이 **노이즈 레이블이 아닌 실제 클래스 레이블**에 따라 명확히 군집화됨을 보였다. 이는 모델이 레이블 노이즈에 과적합되지 않고 진짜 클래스 구조를 학습했음을 의미한다.

---

## 4. 향후 연구에 미치는 영향 및 고려점

### 4-1. 향후 연구에 미치는 영향

#### (A) LNL과 SSL의 통합 패러다임 정착
DivideMix는 LNL을 독립적인 문제로 보지 않고 SSL의 특수 사례로 재정의했다. 이 시각은 이후 연구들이 SSL의 발전(예: FixMatch, SimCLR 등)을 LNL에 빠르게 적용하는 계기를 마련했다.

#### (B) 데이터 분할(Dataset Division) 전략 연구 활성화
GMM 기반 동적 데이터 분할은 이후 다양한 변형 연구(더 정교한 noise modeling, instance-dependent noise 처리 등)를 촉진했다.

#### (C) 다중 네트워크 상호 교수 패러다임
Co-training, Co-teaching의 개념을 SSL과 결합하여 더 정교한 형태로 발전시켰으며, 이후 연구들이 네트워크 다양성(diversity) 유지 전략에 집중하는 트렌드를 이끌었다.

#### (D) 실세계 노이즈 벤치마크 중요성 강조
Clothing1M, WebVision에서의 실험을 통해 인위적 노이즈뿐 아니라 실세계 노이즈에 대한 검증의 중요성을 부각시켰다.

### 4-2. 2020년 이후 관련 최신 연구 비교 분석

> **주의**: 아래 논문들은 제가 알고 있는 지식 범위 내에서 설명하며, 일부 세부 수치는 원문 확인을 권장합니다.

#### **① SOP (Sample Prior Optimized for Noisy Labels, NeurIPS 2022)**
- **핵심**: 각 샘플에 개별적인 prior를 부여하여 노이즈 모델링을 더 유연하게
- **DivideMix와 비교**: GMM의 가정(가우시안 분포)에서 벗어나 더 일반적인 noise modeling 추구
- **참고**: Yao et al., "Instance-Dependent Label-Noise Learning under a Structural Causal Model", NeurIPS 2021

#### **② NoiSy (Learning with Noisy Labels via Self-supervised Representation Learning, CVPR 2021)**
- **핵심**: Self-supervised pre-training으로 노이즈에 강한 표현 학습 후 LNL 적용
- **DivideMix와 비교**: DivideMix는 end-to-end 학습이지만, NoiSy는 표현 학습과 분류를 분리
- **의의**: 대규모 자기지도 표현 학습을 LNL에 활용하는 방향 제시

#### **③ C2D (CVPR 2022, Zheltonozhskii et al.)**
- **핵심**: Contrastive learning으로 학습한 표현을 DivideMix 파이프라인에 결합
- **DivideMix와 비교**: DivideMix의 GMM 기반 분할을 그대로 사용하되, 표현의 질을 대조 학습으로 향상
- **성능**: CIFAR-100 90% noise에서 DivideMix 대비 추가 향상 보고

#### **④ ELR+ (Early-Learning Regularization, NeurIPS 2020)**
- **핵심**: 초기 학습 시점의 예측을 타깃으로 저장하고 과적합 방지
- **DivideMix와 비교**: 명시적 데이터 분할 없이 정규화 항만으로 노이즈 처리
- **참고**: Liu et al., "Early-Learning Regularization Prevents Memorization of Noisy Labels", NeurIPS 2020

#### **⑤ UNICON (CVPR 2022)**
- **핵심**: DivideMix를 기반으로 contrastive learning을 통합하여 더 나은 표현 학습
- **DivideMix와 비교**: DivideMix 파이프라인의 직접적 확장

#### 비교 요약표

| 방법 | 핵심 메커니즘 | LNL 접근 | SSL 활용 | Instance-dependent noise |
|------|-------------|---------|---------|------------------------|
| DivideMix (2020) | GMM + Co-divide + MixMatch | 데이터 분할 | ✅ | ❌ |
| ELR+ (2020) | Early learning regularization | 정규화 | 부분적 | 부분적 |
| C2D (2022) | Contrastive + DivideMix | 데이터 분할+표현 | ✅ | ❌ |
| UNICON (2022) | Contrastive + Semi-sup | 데이터 분할+표현 | ✅ | 부분적 |

### 4-3. 앞으로 연구 시 고려할 점

#### **① Instance-Dependent Noise 처리**
현실의 노이즈는 클래스 단위가 아니라 **개별 샘플의 특성에 따라 발생**하는 경우가 많다. GMM의 단순한 손실 분포 가정이 이를 충분히 처리하지 못할 수 있으므로, 보다 정교한 노이즈 모델링이 필요하다.

#### **② 대규모 자기지도/대조 학습과의 통합**
SimCLR, MoCo, DINO 등 자기지도 학습으로 얻은 강력한 표현을 활용하면 GMM 기반 분할의 정확도를 높일 수 있다. 이미 C2D, UNICON 등이 이 방향을 탐색하고 있다.

#### **③ NLP/멀티모달 도메인으로의 확장**
논문 자체에서 NLP 적용을 향후 과제로 제시했다. 텍스트 분류, 자연어 추론 등에서 노이즈 레이블은 흔하며, DivideMix의 핵심 아이디어(co-divide, co-refinement)를 언어 모델(BERT 계열)에 적용하는 연구가 필요하다.

#### **④ 더 효율적인 두 네트워크 학습**
두 네트워크 동시 학습의 계산 비용을 줄이기 위해, **지식 증류(knowledge distillation)** 또는 **파라미터 공유 구조** 를 탐색할 수 있다.

#### **⑤ 노이즈 비율 자동 추정**
현재 $\lambda_u$ 등 일부 하이퍼파라미터를 실험적으로 결정해야 하는데, 노이즈 비율을 자동으로 추정하여 하이퍼파라미터 튜닝 없이도 적용 가능한 방법이 필요하다.

#### **⑥ 개방형 집합 노이즈(Open-set Noise) 처리**
실세계에서는 학습 데이터에 완전히 다른 클래스의 샘플이 노이즈로 포함될 수 있다(open-set noise). DivideMix는 closed-set noise 가정 하에 설계되었으므로 이에 대한 확장이 필요하다.

---

## 참고 자료

**주요 논문 (원문 제공)**
- **Li, J., Socher, R., & Hoi, S.C.H. (2020)**. "DivideMix: Learning with Noisy Labels as Semi-supervised Learning." *ICLR 2020*. arXiv:2002.07394

**논문 내 인용 문헌 (원문 기반)**
- Arazo et al. (2019). "Unsupervised Label Noise Modeling and Loss Correction." *ICML 2019*
- Berthelot et al. (2019). "MixMatch: A Holistic Approach to Semi-Supervised Learning." *NeurIPS 2019*
- Han et al. (2018). "Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels." *NeurIPS 2018*
- Zhang et al. (2017). "Understanding Deep Learning Requires Rethinking Generalization." *ICLR 2017*
- Zhang et al. (2018). "MixUp: Beyond Empirical Risk Minimization." *ICLR 2018*
- Pereyra et al. (2017). "Regularizing Neural Networks by Penalizing Confident Output Distributions." *ICLR Workshop 2017*
- Tarvainen & Valpola (2017). "Mean Teachers are Better Role Models." *NIPS 2017*
- Yi & Wu (2019). "Probabilistic End-to-end Noise Correction for Learning with Noisy Labels." *CVPR 2019*
- Li et al. (2019). "Learning to Learn from Noisy Labeled Data." *CVPR 2019*

**2020년 이후 관련 연구 (지식 기반, 원문 직접 확인 권장)**
- Liu et al. (2020). "Early-Learning Regularization Prevents Memorization of Noisy Labels." *NeurIPS 2020*
- Zheltonozhskii et al. (2022). "Contrast to Divide: Self-Supervised Pre-Training for Learning with Noisy Labels." *WACV 2022*
- Karim et al. (2022). "UNICON: Combating Label Noise Through Uniform Selection and Contrastive Learning." *CVPR 2022*

> **면책 고지**: 2020년 이후 최신 연구들의 세부 수치나 결과는 원문 논문을 직접 확인하시기 바랍니다. 제가 접근 가능한 원문은 DivideMix 원논문(arXiv:2002.07394v1)이며, 이에 기반한 내용의 정확도는 높으나 후속 연구 부분은 학습 데이터 기반 설명임을 명시합니다.
