# Transporting Causal Mechanisms for Unsupervised Domain Adaptation

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

본 논문(Yue et al., ICCV 2021)은 기존 UDA(Unsupervised Domain Adaptation) 방법론이 **Covariate Shift**와 **Conditional Shift** 가정에 기반하여 도메인 불변 특징(domain-invariant features)을 학습하는 과정에서 필연적으로 발생하는 **의미론적 손실(Semantic Loss)** 문제를 인과론적 관점에서 해결하고자 합니다.

핵심 통찰:
> 도메인 간 특징 손실은 **혼동 효과(confounding effect)** 이며, 이는 통계적 학습만으로는 제거 불가능하고 **인과적 개입(causal intervention)** 을 통해서만 근본적으로 해결 가능하다.

### 주요 기여

| 기여 항목 | 내용 |
|---|---|
| **이론적 기반** | 수송 가능성 이론(Transportability Theory)을 UDA에 적용 |
| **DCM 발견** | 비지도 방식의 분리된 인과 메커니즘(Disentangled Causal Mechanisms) 학습 |
| **Proxy Variable 활용** | 관찰 불가능한 혼동변수 $U$를 대리 변수(proxy variable)로 표현 |
| **실용적 구현** | 이론적 해법인 Eq.(1)을 실제 신경망으로 구현한 TCM 프레임워크 제안 |
| **SOTA 달성** | Office-Home(70.7%), ImageCLEF-DA(90.5%), VisDA-2017(75.8%) |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

#### 기존 방법의 한계: Semantic Loss

기존 UDA는 두 가지 가정에 의존합니다:

$$\text{Covariate Shift: } P(X|S=s) \neq P(X|S=t)$$

$$\text{Conditional Shift: } P(Y|X, S=s) \neq P(Y|X, S=t)$$

이를 해결하기 위해 대부분의 방법은 **도메인 불변 특징 학습**에 집중합니다. 그러나 이 과정에서:

- **Source domain**에서 비판별적(non-discriminative)으로 보이는 특징(예: 물체의 "형태(Shape)")이 무시됨
- 해당 특징이 **Target domain**에서는 핵심 판별 요소일 수 있음

예: Office-Home에서 실사(Real World) → 클립아트(Clipart) 적응 시, 배경(Background) 특징은 Source에서는 판별력이 있으나 Target에서는 무의미해지고, 오히려 형태(Shape) 특징이 중요해집니다.

#### 인과 그래프 모델링

논문은 DA 문제를 다음의 인과 그래프로 모델링합니다:

$$S \rightarrow U \rightarrow X \rightarrow Y, \quad U \rightarrow Y$$

- $S$: 도메인 선택 변수 (source/target)
- $U$: 관찰 불가능한 의미론적 속성(semantic attribute, confounder)
- $X$: 이미지 샘플
- $Y$: 레이블

도메인 갭의 근본 원인: $P(U|S=s) \neq P(U|S=t)$

### 2.2 제안 방법 (수식 포함)

#### 핵심 수식: 수송 가능성 이론의 인과 개입

$$P(Y|do(X), S) = \sum_{u} P(Y|X, U=u)P(U=u|S) \tag{1}$$

- $P(Y|X, U)$: 도메인 불가지론적(domain-agnostic), $S$가 $U=u$ 조건 하에 $X, Y$로부터 분리됨
- $P(U|S)$: 도메인별 사전 분포로 공정한 층화 조정(stratification)

이 식의 의미: 특정 $U$ 값에서의 조건부 예측 $P(Y|X, U)$는 도메인에 무관하게 일반화되며, $P(U|S)$로 가중치를 부여하여 도메인별로 공정한 예측을 수행합니다.

#### Stage 1: 분리된 인과 메커니즘 발견 (DCM Discovery)

$U$를 분리된 인과 요인들 $(U_1, \ldots, U_k)$로 분해하면:

$$P(Y|X) \propto \sum_{i=1}^{k} P(Y|X, U_i) \tag{분리 가정}$$

$k$쌍의 매핑 함수 $\{(M_i, M_i^{-1})\}_{i=1}^{k}$를 비지도 방식으로 학습합니다:

- $M_i: X_s \rightarrow X_t$ (Source → Target 변환, $U_i$ 개입)
- $M_i^{-1}: X_t \rightarrow X_s$ (Target → Source 변환)

**Counterfactual Faithfulness 정리**에 의해, $M_i$가 분리된 개입이 되려면:

$$M_i(x_s) \sim P(X_t)$$

이를 위해 **CycleGAN 손실**을 각 쌍에 적용합니다:

$$\min_{(M_i, M_i^{-1})} \mathcal{L}^{i}_{CycleGAN}, \quad \text{where } i = \underset{j \in \{1,\ldots,k\}}{\arg\min}\ \mathcal{L}^{j}_{CycleGAN} \tag{2}$$

각 학습 단계에서 **최소 손실을 가진 DCM 쌍만 업데이트**하여 각 쌍이 서로 다른 인과 요인에 특화되도록 합니다.

#### Stage 2: Proxy Variables를 통한 $U$ 표현

관찰 가능한 대리 변수(Proxy Variable) 도입:
- $\hat{X}$: DCM 출력 ($U \rightarrow \hat{X} \rightarrow Y$)
- $Z$: VAE로 인코딩된 잠재 변수 ($U \rightarrow Z \rightarrow X$)

**Proxy Function 정리**:

$$P(Y|Z, X, S) = \sum_{\hat{x}} h_y(X, \hat{x}) P(\hat{X}=\hat{x}|Z, X, S) \tag{3}$$

$$P(Y|X, U) = \sum_{\hat{x}} h_y(X, \hat{x}) P(\hat{X}=\hat{x}|U) \tag{4}$$

Eq.(4)에서 $U|S=t$에 대해 기대값을 취하면 실용적 추론 공식 도출:

$$P(Y|do(X), S=t) = \sum_{\hat{x} \in \hat{\mathcal{X}}_t} h_y(X, \hat{x}) P(\hat{X}=\hat{x}|S=t) \tag{5}$$

#### 선형 함수 형식

$$f_y(Z, X) = W_1 Z + W_2 X + b_1 \tag{6a}$$

$$f_{\hat{x}}(Z, X) = W_3 Z + W_4 X + b_2 \tag{6b}$$

Eq.(3)을 풀면 $h_y$의 닫힌 형식(closed-form) 해가 도출됩니다:

$$h_y(X, \hat{X}) = b_1 - W_1 W_3^+ b_2 + W_1 W_3^+ \hat{X} + (W_2 - W_1 W_3^+ W_4) X \tag{7}$$

여기서 $({\cdot})^+$는 의사역행렬(pseudo-inverse)입니다.

#### 전체 목적함수

$$\min_{\omega, \beta, \theta} (\mathcal{L}_c + \mathcal{L}_v) + \min_{\beta} \max_{\gamma} \alpha \mathcal{L}_p \tag{8}$$

- $\mathcal{L}_c$: 분류 손실(Cross-Entropy) + MSE 손실
- $\mathcal{L}_v$: VAE 손실 (ELBO)
- $\mathcal{L}_p$: 프록시 손실 (도메인 적대적 정규화)

**VAE 손실:**

$$\mathcal{L}_v = -\mathbb{E}_{Q_\theta(Z|X=x_s)}[P_\theta(X=x_s|Z)] + D_{KL}(Q_\theta(Z|X=x_s) \| P(Z)) \tag{9}$$

**Proxy 손실 (특징 수준 정렬):**

$$\mathcal{L}_p = \log D_s(x_s) + \frac{1}{k}\sum_{\hat{x}_s \in \hat{\mathcal{X}}_s} \log(1-D_t(\hat{x}_s)) + \log D_t(x_t) + \frac{1}{k}\sum_{\hat{x}_t \in \hat{\mathcal{X}}_t} \log(1-D_s(\hat{x}_t)) \tag{10}$$

### 2.3 모델 구조

```
[Stage 1: DCM Learning]
X_s, X_t → k개의 CycleGAN 쌍 {(M_i, M_i^{-1})}
           → 최소 손실 쌍만 업데이트 (winner-take-all)
           → 각 M_i는 disentangled 속성 U_i에 특화

[Stage 2: Proxy-based Causal Inference]
X → ResNet backbone → feature X
X → VAE encoder → Z (latent)
X → DCMs → X̂ (proxy variables)

Linear functions: f_y(Z,X), f_x̂(Z,X)
→ Proxy Function h_y 도출 (닫힌 형식)
→ P(Y|do(X), S=t) 계산 (Eq.5)

Loss: L_c + L_v + α·L_p (adversarial)
```

### 2.4 성능 향상

| 데이터셋 | TCM | 이전 SOTA | 향상 |
|---|---|---|---|
| Office-Home | **70.7%** | GVB-GD: 70.4% | +0.3% |
| ImageCLEF-DA | **90.5%** | ETD: 89.7% | +0.8% |
| VisDA-2017 | **75.8%** | DMRL: 75.5% | +0.3% |

특히 도메인 갭이 큰 태스크(A→C: +1.6%, P→C: +1.1%)에서 두드러진 향상을 보입니다.

### 2.5 한계점

1. **DCM 학습의 이론적 불완전성**: Eq.(2)의 winner-take-all 전략은 필요조건(necessary condition)만 사용하며, 충분조건이 아님
2. **GAN 기반 품질 의존성**: DCM이 CycleGAN 기반이므로 생성 이미지 품질에 성능이 좌우됨
3. **복잡한 속성(형태, 시점 변화 등)의 메커니즘 발견 어려움**: 현재는 밝기, 색온도 등 비교적 단순한 스타일 속성에 특화됨
4. **$k$ 수 선택**: 적절한 DCM 수 $k$를 사전에 결정해야 하며, 과도하게 크면 일부 DCM이 학습되지 않음
5. **Source domain 레이블 의존**: VAE와 $h_y$는 Source에서만 학습되므로 레이블이 전혀 없는 시나리오에는 적용 불가

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 인과 개입을 통한 일반화

TCM의 일반화 성능 향상은 다음 메커니즘에 기반합니다:

**$P(Y|X, U)$의 도메인 불가지론적 특성**:

$$P(Y|do(X), S) = \sum_u P(Y|X, U=u) \cdot P(U=u|S)$$

$P(Y|X,U)$는 $S$가 $U$에 의해 차단(blocked)되므로 도메인에 독립적입니다. 즉, Source에서 학습한 $P(Y|X,U)$가 Target에서도 그대로 적용됩니다.

**Proxy Function $h_y$의 도메인 불변성**:

논문의 Appendix에서 증명된 바에 따르면, $h_y(X, \hat{X})$는 Source에서 학습되더라도 도메인 불변(domain-agnostic)입니다. 이는 $\hat{X}$와 $Z$가 $U$의 관찰 가능한 대리 변수로서 도메인 갭을 가교(bridge)하기 때문입니다.

### 3.2 Semantic Loss 완화와 일반화

기존 도메인 불변 특징 학습의 문제점:

$$\text{Generalization Error} \geq \frac{1}{2}(d_{\mathcal{H}\Delta\mathcal{H}}(D_s, D_t) - \lambda)$$

(Ben-David et al., 2010의 이론적 하한)

이 이론은 도메인을 최대한 유사하게 만들 것을 권장하지만, 이는 의미론적으로 중요한 특징을 소실시킵니다. TCM은 도메인 정렬을 강제하지 않고도 인과적 특징을 보존함으로써, 한계 이론(Ben-David et al.)의 가정을 우회하여 일반화 성능을 높입니다.

**실험적 증거**: t-SNE 시각화에서 GVB-GD보다 도메인 정렬은 낮지만 분류 성능은 더 높음 → 도메인 정렬 = 일반화라는 통념에 의문을 제기

### 3.3 CAM 분석으로 확인된 일반화

Class Activation Map(CAM) 분석에서:
- GVB-GD/Baseline: 맥락적 특징(contextual features, e.g., 음식 ↔ 포크)에 의존 → Target에서 실패
- TCM: 객체 형태(shape) 특징 보존 → Target에서도 올바른 예측 유지

### 3.4 향후 일반화 성능 향상 가능성

1. **더 강력한 생성 모델 사용**: Diffusion 모델 등을 DCM으로 사용하면 형태, 시점 등 복잡한 속성 변환 가능
2. **멀티 도메인 확장**: $k$개의 DCM을 통해 여러 중간 도메인(intermediate domains)을 동시에 다룰 수 있음
3. **대형 사전학습 모델과 결합**: CLIP, ViT 등과 결합 시 더욱 풍부한 의미론적 특징 공간에서 인과 메커니즘 탐색 가능

---

## 4. 향후 연구에 미치는 영향 및 고려점

### 4.1 연구에 미치는 영향

#### (1) 인과추론과 도메인 적응의 결합 패러다임 확립

TCM은 UDA를 "도메인 정렬 문제"가 아닌 "인과 구조 추정 문제"로 재정의하여, 인과추론 기반 DA의 새로운 방향을 제시합니다. 이는 Pearl의 do-calculus를 컴퓨터 비전에 실용적으로 적용한 선구적 사례입니다.

#### (2) Semantic Loss에 대한 체계적 분석 제공

"도메인 불변 특징 학습이 의미론적 손실을 초래한다"는 문제를 이론적으로 정형화하여, 후속 연구들이 이 문제를 명시적으로 다루는 계기를 마련했습니다.

#### (3) 분리된 인과 메커니즘의 UDA 활용

독립적 인과 메커니즘(Independent Causal Mechanisms, ICM) 원리를 domain shift 설명에 적용한 방식은, 인과적 표현 학습(causal representation learning)과 도메인 일반화(Domain Generalization) 연구에 큰 영향을 미칩니다.

### 4.2 향후 연구 시 고려할 점

#### (1) 더 강력한 Disentanglement 방법론 필요

현재 CycleGAN 기반 DCM은 스타일 수준의 속성(밝기, 색온도)에만 효과적입니다. 향후에는:
- **Diffusion 기반 DCM**: 더 세밀한 의미론적 속성(시점, 형태) 변환
- **Causal VAE/Flow 기반 접근**: 잠재 공간에서 직접 인과 요인 분리

#### (2) 최적 DCM 수 $k$의 자동 결정

현재는 수동으로 $k$를 설정해야 합니다. 베이지안 비모수(Bayesian non-parametric) 방법이나 자동 모델 선택(automatic model selection) 방법의 통합이 필요합니다.

#### (3) 더 복잡한 인과 그래프로 확장

현재의 단순화된 인과 그래프($S \rightarrow U \rightarrow X \rightarrow Y$)를 넘어:
- 레이블 시프트(Label Shift) $P(Y|S)$ 고려
- 잠재 변수 간의 상호작용(interaction) 모델링
- 다중 소스 도메인(Multi-source DA)으로의 확장

#### (4) 대규모 사전학습 모델과의 통합

ViT, CLIP, DINO 등의 대규모 사전학습 모델은 풍부한 의미론적 표현을 제공합니다. TCM의 Proxy Variable 접근법과 이러한 모델을 결합하면 더욱 강력한 일반화 성능을 기대할 수 있습니다.

#### (5) Domain Generalization으로의 확장

TCM은 Source-Target 쌍이 고정된 UDA를 다루지만, 인과 메커니즘 자체는 Target domain 데이터 없이도 적용 가능한 **Domain Generalization**으로 확장될 수 있습니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

아래 비교는 제공된 논문 내의 참고문헌과 해당 분야의 일반적 지식을 기반으로 합니다. **논문에 명시된 비교 대상만 정확하게 기술**하며, 논문 이후 발표된 연구에 대해서는 일반적 트렌드 수준에서 서술합니다.

### 5.1 논문 내에서 비교된 2020년 이후 방법들

| 방법 | 발표 | 접근법 | Office-Home Avg | 주요 특징 |
|---|---|---|---|---|
| **GVB-GD** (Cui et al., CVPR 2020) | 2020 | 적대적 도메인 정렬 | 70.4% | 점진적으로 소멸하는 브릿지(bridge) 사용 |
| **ETD** (Li et al., CVPR 2020) | 2020 | 최적 수송 기반 | 67.3% | Enhanced Transport Distance |
| **A²LP** (Zhang et al., ECCV 2020) | 2020 | 레이블 전파 | 89.4% (ImageCLEF) | 앵커 기반 반지도 학습 |
| **DMRL** (Wu et al., ECCV 2020) | 2020 | Dual Mixup 정규화 | 75.5% (VisDA) | Mixup 기반 적대적 정규화 |
| **Heuristic DA** (Cui et al., NeurIPS 2020) | 2020 | 도메인 특화 표현 | - | S → U 가정 기반 |
| **TCM (Ours)** | ICCV 2021 | 인과 개입 | **70.7%** | Transportability + DCM + Proxy |

### 5.2 TCM 이후 관련 연구 트렌드 (일반적 지식 기반)

**주의**: 아래 내용은 논문 제출 이후 발표된 연구들에 대한 일반적 트렌드 서술입니다. 특정 논문의 정확한 수치를 확신하기 어려워 수치 비교는 생략합니다.

#### 인과추론 기반 DA 후속 연구 방향

1. **Causal Domain Generalization**: TCM의 인과 구조 분석을 Domain Generalization으로 확장하는 연구들이 등장했습니다 (예: Causal Semantic Generative Model 계열).

2. **대규모 사전학습 모델 + 인과적 접근**: ViT, CLIP 기반의 UDA 방법들이 등장하며, TCM의 인과적 관점을 Vision-Language 모델에 통합하는 시도가 이루어지고 있습니다.

3. **Diffusion 모델 기반 DCM**: TCM에서 사용한 CycleGAN 기반 DCM을 Diffusion 모델로 대체하면 더 복잡한 의미론적 변환이 가능합니다.

#### TCM의 차별점 (2020년 이후 방법 대비)

| 구분 | 기존 도메인 정렬 방법들 | TCM |
|---|---|---|
| **이론적 근거** | 통계적 거리 최소화(MMD, Wasserstein 등) | 인과 개입(do-calculus) |
| **Semantic Loss** | 암묵적 트레이드오프 | 명시적 제거 |
| **도메인 정렬 강제 여부** | 강제 정렬 | 정렬 불필요 |
| **해석 가능성** | 낮음 | 인과 그래프로 설명 가능 |
| **$U$ 처리** | 무시 또는 암묵적 처리 | 명시적 층화 및 대리 표현 |

---

## 참고 자료

**주 논문:**
- Yue, Z., Sun, Q., Hua, X.-S., & Zhang, H. (2021). **Transporting Causal Mechanisms for Unsupervised Domain Adaptation**. *ICCV 2021*, pp. 8599–8608.

**논문 내 핵심 참고문헌:**
- Pearl, J., & Bareinboim, E. (2014). **External validity: From do-calculus to transportability across populations**. *Statistical Science*.
- Pearl, J. (2009). **Causality: Models, Reasoning and Inference** (2nd ed.). Cambridge University Press.
- Miao, W., Geng, Z., & Tchetgen, E. J. T. (2018). **Identifying causal effects with proxy variables of an unmeasured confounder**. *Biometrika*.
- Parascandolo, G., et al. (2018). **Learning independent causal mechanisms**. *ICML*.
- Zhu, J.-Y., et al. (2017). **Unpaired image-to-image translation using cycle-consistent adversarial networks**. *ICCV*.
- Ben-David, S., et al. (2010). **A theory of learning from different domains**. *Machine Learning*.
- Zhao, H., et al. (2019). **On learning invariant representations for domain adaptation**. *ICML*.

**공개 코드:**
- GitHub: https://github.com/yue-zhongqi/tcm
