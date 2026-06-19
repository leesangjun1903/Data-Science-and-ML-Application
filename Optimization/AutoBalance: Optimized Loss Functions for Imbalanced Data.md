# AutoBalance: Optimized Loss Functions for Imbalanced Data 

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

AutoBalance는 **불균형 데이터(imbalanced data)** 환경에서, 대용량 딥러닝 모델이 학습 데이터에 과적합(overfit)하여 훈련 손실이 테스트 성능을 반영하지 못하는 문제를 해결하기 위해, **이중 최적화(bilevel optimization)** 프레임워크를 통해 손실 함수를 자동으로 설계하는 방법론을 제안한다.

### 주요 기여

| 기여 항목 | 설명 |
|-----------|------|
| **AutoBalance 프레임워크** | 모델 가중치(lower-level)와 손실 함수 하이퍼파라미터(upper-level)를 동시에 최적화 |
| **파라메트릭 손실 함수 설계** | $w_k$, $l_k$, $\Delta_k$ 세 가지 파라미터로 클래스별 맞춤 처리 |
| **개인화 데이터 증강 (PDA)** | 클래스/그룹별 개별 증강 정책 학습 |
| **탐색 공간 축소** | 빈도 기반 클래스 클러스터링으로 과적합 방지 |
| **그룹 공정성 적용** | 클래스 불균형을 넘어 그룹 민감 분류에도 적용 가능 |
| **이론적 뒷받침** | 검증 분할의 필요성과 손실 함수 설계에 대한 이론적 근거 제시 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

#### 문제 1: 불균형 데이터에서의 일반화 실패

현대 딥러닝 모델은 **과파라미터화(overparameterized)** 되어 있어, 훈련 데이터에 완벽히 적합(interpolating regime)하면서 훈련 손실이 0에 수렴한다. 이 경우:

- 가중 교차 엔트로피(weighted cross-entropy)와 같은 고전적인 방법이 **균형 정확도(balanced accuracy)** 향상에 효과가 미미함
- 훈련 손실이 테스트 오류를 신뢰할 수 있게 나타내지 못함

$$\mathcal{E}_{bal}(f) := \frac{1}{K}\sum_{k=1}^{K}\mathcal{E}_k(f) \quad \text{(최소화 목표)}$$

#### 문제 2: 손실 함수 하이퍼파라미터 튜닝의 어려움

기존의 파라메트릭 손실 함수 (LDAM, LA, VS-loss 등)는 이론적 직관에 의존한 고정된 하이퍼파라미터를 사용하며, **다양한 공정성 목표에 맞게 체계적으로 조정하는 방법이 없었음**.

---

### 2.2 제안하는 방법 (수식 포함)

#### 핵심 파라메트릭 손실 함수

논문의 핵심 손실 함수는 세 벡터 $\mathbf{w}, \mathbf{l}, \boldsymbol{\Delta} \in \mathbb{R}^K$로 제어된다:

$$\ell(y, f(\mathbf{x})) = w_y \log\left(1 + \sum_{k \neq y} e^{l_k - l_y} \cdot e^{\Delta_k f_k(\mathbf{x}) - \Delta_y f_y(\mathbf{x})}\right) \tag{2.1}$$

- $w_y$: 클래스별 가중치 (classical weighted CE)
- $l_y$: 가산적 로짓 조정 (additive logit adjustment)
- $\Delta_y$: 곱셈적 로짓 조정 (multiplicative logit adjustment)

#### 데이터 증강이 포함된 훈련 손실 함수

$$\ell_{train}(y, \mathbf{x}, f; \boldsymbol{\alpha}) = -\mathbb{E}_{\mathcal{A}}\left[w_y \log\left(\frac{e^{\sigma(\Delta_y)f_y(\mathcal{A}_y(\mathbf{x})) + l_y}}{\sum_{i \in [K]} e^{\sigma(\Delta_i)f_i(\mathcal{A}_y(\mathbf{x})) + l_i}}\right)\right] \tag{2.2}$$

여기서:
- $\mathcal{A} = (\mathcal{A}\_y)_{y=1}^K$: 클래스별 개인화 증강 정책
- $\sigma(\cdot)$: 시그모이드 함수 (음수 방지, $\Delta_i \in (0,1)$ 보장)
- $\boldsymbol{\alpha} = [\mathbf{w}, \mathbf{l}, \boldsymbol{\Delta}, \text{param}(\mathcal{A})]$: 최적화할 하이퍼파라미터 집합

#### 이중 최적화 (Bilevel Optimization)

$$\min_{\boldsymbol{\alpha}} \mathcal{L}_{fair}^{S_V}(f_{\boldsymbol{\alpha}}) \quad \text{where} \quad f_{\boldsymbol{\alpha}} = \arg\min_{f \in \mathcal{F}} \mathcal{L}_{train}^{S_T}(f; \boldsymbol{\alpha}) := \frac{1}{n_T}\sum_{i=1}^{n_T} \ell_{train}(y_i, \mathbf{x}_i, f; \boldsymbol{\alpha}) \tag{2.3}$$

- **Lower-level (내부 최적화)**: 훈련 데이터 $S_T$에서 모델 파라미터 $\theta$ 최적화
- **Upper-level (외부 최적화)**: 검증 데이터 $S_V$에서 공정성 손실을 기반으로 $\boldsymbol{\alpha}$ 최적화

#### 하이퍼 그래디언트 계산 (Implicit Function Theorem, IFT)

```math
\frac{\partial \mathcal{L}_{fair}(\theta^*)}{\partial \boldsymbol{\alpha}} = \frac{\partial \mathcal{L}_{fair}}{\partial \theta^*} \cdot \frac{\partial \theta^*}{\partial \boldsymbol{\alpha}}, \quad \frac{\partial \theta}{\partial \boldsymbol{\alpha}} = -\left(\frac{\partial^2 \mathcal{L}_{train}}{\partial \theta^2}\right)^{-1} \frac{\partial^2 \mathcal{L}_{train}}{\partial \theta \partial \boldsymbol{\alpha}}
```

역 헤시안(inverse Hessian) 계산의 복잡성을 피하기 위해 **Neumann 급수 근사** 활용.

#### 그룹 민감 분류를 위한 손실 함수 (4절)

$$\ell_{train}(y, g, f(\mathbf{x}); \boldsymbol{\alpha}) = -w_{yg}\log\left(\frac{e^{\sigma(\Delta_{yg})f_y(\mathbf{x}) + l_{yg}}}{\sum_{k \in [K]} e^{\sigma(\Delta_{kg})f_k(\mathbf{x}) + l_{kg}}}\right) \tag{4.2}$$

- $\mathbf{w}, \mathbf{l}, \boldsymbol{\Delta} \in \mathbb{R}^{[K] \times [G]}$: (클래스, 그룹) 쌍별 파라미터

#### DEO (Difference of Equal Opportunity)

$$\mathcal{L}_{deo}(f) = |\mathcal{L}_{+,1}(f) - \mathcal{L}_{+,2}(f)| + |\mathcal{L}_{-,1}(f) - \mathcal{L}_{-,2}(f)| \tag{4.1}$$

---

### 2.3 모델 구조 및 알고리즘

```
Algorithm 1: AutoBalance via Bilevel Optimization

[Search Phase]
1. α를 공정성 손실과 일치하도록 초기화 (Consistent initialization)
2. t1 에포크 동안 워밍업 (α 고정, θ만 학습)
3. t1 ~ t2 에포크:
   - 훈련 배치 BT 샘플링 → 클래스별 증강 적용
   - Lower: θ ← θ - η_θ ∇_θ L_train^{BT}(f_θ; α)
   - 검증 배치 BV 샘플링
   - Upper: α ← α - η_α ∇_α L_fair^{BV}(f_θ) (via IFT)

[Retraining Phase]
4. 전체 데이터 S = ST ∪ SV와 최적 α*로 θ 재훈련

[Evaluation Phase]
5. 최종 모델 θ*로 테스트
```

#### 탐색 공간 축소 (Subspace-based search)

클래스를 빈도 기반으로 클러스터링하여 차원 축소:

$$\boldsymbol{\Delta} = D_\pi \boldsymbol{\Delta}' \quad \text{and} \quad \mathbf{l} = D_\pi \mathbf{l}' \quad \text{where} \quad \mathbf{l}', \boldsymbol{\Delta}' \in \mathbb{R}^{K'}$$

$D_\pi \in \mathbb{R}^{K \times K'}$: 빈도 인식 사전 행렬, $K' \ll K$

| 데이터셋 | 클러스터 크기 C | 원래 클래스 수 K |
|---------|---------------|----------------|
| CIFAR10-LT | 1 | 10 |
| CIFAR100-LT | 10 | 100 |
| ImageNet-LT | 20 | 1,000 |
| iNaturalist | 40 | 8,142 |

---

### 2.4 성능 향상

#### 클래스 불균형 (Table 1): 균형 오류율 (낮을수록 좋음)

| 방법 | CIFAR10-LT | CIFAR100-LT | ImageNet-LT | iNaturalist |
|------|-----------|-------------|-------------|-------------|
| Cross-Entropy | 30.45 | 62.69 | 55.47 | 39.72 |
| LDAM [8] | 26.37 | 59.47 | 54.21 | 35.63 |
| LA loss [53] | 23.13 | 58.96 | 52.46 | 34.06 |
| CDT loss [67] | **20.73** | 57.26 | 53.47 | 34.46 |
| **AutoBalance: Δ&l** | 21.39 | 56.84 | 51.74 | 33.41 |
| **AutoBalance: Δ&l, LA init** | 21.15 | **56.70** | **50.91** | **33.25** |

#### 그룹 불균형 (Table 3, Waterbirds):

| 방법 | Balanced Error | Worst Error | DEO |
|------|---------------|-------------|-----|
| DRO [58] | 16.47 | 32.67 | 6.91 |
| **AutoBalance** | **15.13** | **30.33** | **4.25** |

---

### 2.5 한계점

1. **계산 비용**: 표준 훈련 대비 약 4~5배의 연산 시간 소요 (탐색 + 재훈련 단계)
2. **극단적 불균형에서의 취약성**: 클래스당 샘플이 매우 적을 경우(예: CIFAR100-LT의 최소 클래스는 검증 샘플 1개), 하이퍼 그래디언트 추정이 불안정할 수 있음
3. **하이퍼 그래디언트 수렴 불확실성**: Algo. 1이 항상 최적 설계로 수렴하지 않을 수 있음 (예: CIFAR10-LT에서 $\tau$만 튜닝한 LA 손실이 $\Delta$, $l$ 개별 튜닝보다 우수)
4. **탄소 발자국**: AutoML 계열로서 자원 소모가 크며, 환경 영향 고려 필요

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 검증 분할(Train-Validation Split)의 핵심 역할

논문에서 가장 중요한 일반화 관련 통찰은 **검증 데이터가 과파라미터화 환경에서 테스트 성능의 신뢰할 수 있는 대리자(proxy)가 된다**는 것이다.

$$\text{훈련 오류} \to 0 \quad (\text{완전 과적합}), \quad \text{검증 오류} \approx \text{테스트 오류}$$

이에 대한 이론적 근거:
- 하이퍼파라미터 $\boldsymbol{\alpha}$의 차원($K'$)이 검증 데이터 크기 $n_V$보다 작을 때, 검증 손실은 과적합되지 않음
- 논문 부록에서, 두 목표 $(\mathcal{L}_1, \mathcal{L}_2)$에 대해 균일하게(uniformly over $\lambda$):

$$\boldsymbol{\alpha}^* = \arg\min_{\boldsymbol{\alpha}} \mathcal{L}_{fair}^{S_V}(f) = (1-\lambda)\mathcal{L}_1^{S_V} + \lambda\mathcal{L}_2^{S_V}$$

가 테스트 리스크 $\mathcal{L}_{fair}(f)$를 근사적으로 최소화함을 증명

### 3.2 개인화 데이터 증강(PDA)의 일반화 기여

**Lemma 2**: 이진 분류에서 파라메트릭 손실 (2.1)의 임의의 $(l_i, \Delta_i, w_i)$ 선택에 대해, 개인화된 구면 증강(spherical augmentation) 강도가 존재하여, 정규화 없이 로지스틱 손실을 최적화하면 (2.1)을 최적화한 것과 동일한 분류기를 반환한다.

이는 **소수 클래스에 더 강한 증강을 적용**하면 결정 경계가 소수 클래스를 보호하는 방향으로 이동함을 의미한다:

$$\varepsilon_{minority} > \varepsilon_{majority} \Rightarrow \text{결정 경계가 소수 클래스에 유리하게 이동}$$

### 3.3 일관 초기화(Consistent Initialization)의 역할

알고리즘이 Fisher 일관(Bayes-consistent) 손실로 초기화된다:

$$\ell_{train}(\cdot) = \ell_{fair}(\cdot) \quad \text{초기 상태}$$

이는 초기 훈련 단계(훈련 리스크가 테스트 리스크와 상관관계가 있을 때)에서 일반화를 보장하며, 이후 과파라미터화 체제로 진입하면서 $\mathbf{l}$, $\boldsymbol{\Delta}$를 자동 조정한다.

### 3.4 검증 크기에 따른 일반화 실험 결과 (Figure 4)

CIFAR10-LT에서 다양한 검증 크기 실험:

| 검증/훈련 비율 | 테스트 균형 오류 |
|-------------|---------------|
| 1/40 (매우 작음) | 상대적으로 높음 |
| 1/8 | 중간 |
| 1/4 (권장) | **가장 낮음** |

**핵심 관찰**: 훈련 오류는 항상 0으로 수렴(과적합)하지만, 검증 오류는 완만하게 과적합하며 테스트 오류를 근사함.

---

## 4. 향후 연구에 미치는 영향과 고려할 점

### 4.1 향후 연구에 미치는 영향

#### (1) 손실 함수 자동 설계 패러다임 확립
AutoBalance는 **"손실 함수를 사전 지식 없이 데이터에서 자동으로 학습"**하는 방향을 제시하여, AutoML 연구에서 손실 함수 설계를 최적화 대상으로 포함하는 흐름을 강화한다.

#### (2) 이중 최적화의 공정성 분야 적용 확대
상위 레벨에서 공정성 지표(DEO, 균형 정확도 등)를 목표로 삼는 방식은 **알고리즘적 공정성(algorithmic fairness)** 연구에 새로운 최적화 관점을 제공한다.

#### (3) 과파라미터화 환경에서의 이론적 기반
훈련 손실이 0이 되는 보간 체제(interpolating regime)에서는 훈련 기반 최적화가 무의미하며, **검증 기반 최적화가 필수**라는 이론적 통찰은 이후 연구들에 영향을 줄 것이다.

#### (4) 연합 학습(Federated Learning) 적용 가능성
논문에서 언급된 것처럼, 클라이언트별 데이터 불균형 문제를 해결하는 데 AutoBalance의 프레임워크가 적용될 수 있다.

---

### 4.2 2020년 이후 관련 최신 연구 비교 분석

아래는 논문에서 직접 인용된 또는 동 시기 발표된 관련 연구들과의 비교이다. (논문 원문 기반으로 서술하며, 논문 발표 이후 독립적으로 발표된 연구는 본 논문 PDF에 포함되지 않으므로 정확성 제한을 명시함.)

#### 논문이 직접 비교한 2020년 이후 주요 방법론:

| 연구 | 방법 | AutoBalance 대비 한계 |
|------|------|----------------------|
| **Menon et al. (2020)** - "Long-tail learning via logit adjustment" (LA loss) | $l_i = \tau \log(\pi_i)$ 고정값 사용 | 하이퍼파라미터를 수동 설정, 다양한 공정성 목표에 최적화 불가 |
| **Kini et al. (2021)** - VS-loss (NeurIPS 2021) | $w, l, \Delta$ 세 파라미터 동시 사용 제안 | 이론적 처방전은 있으나, 자동 최적화 메커니즘 부재 |
| **Hataya et al. (2020)** - MADAO | 전체 데이터셋에 단일 증강 정책 학습 | 클래스별 개인화 증강 없음 |
| **Sagawa et al. (2019)** - DRO | 최악의 그룹 손실 최소화 | 고정 정규화 기반, DEO에서 AutoBalance보다 성능 낮음 |
| **Ye et al. (2020)** - CDT loss | 클래스 의존적 온도 조정 | 단일 파라미터 유형, 자동 조정 없음 |

#### 일반화 관련 이론 연구 (논문 인용):

- **Oymak et al. (2021)** - "Generalization guarantees for neural architecture search with train-validation split" [55]: AutoBalance의 검증 분할 이론적 기반을 제공하며, 하이퍼파라미터 차원이 검증 크기보다 작을 때 일반화 보장이 성립함을 공식화.

- **Lorraine et al. (2020)** - "Optimizing millions of hyperparameters by implicit differentiation" [46]: IFT 기반 하이퍼 그래디언트 계산 방법론 제공.

---

### 4.3 앞으로 연구 시 고려할 점

#### (1) 계산 효율성 개선
- 현재 4~5배의 추가 계산 비용이 실용적 적용의 장벽이 됨
- **One-step 근사**, **메타러닝 기반 초기화**, **경량화된 하이퍼 그래디언트 추정** 등으로 효율을 높이는 연구 필요

#### (2) 극단적 불균형 시나리오 대응
- 검증 집합에 샘플이 극히 적은 클래스(예: 1개)의 경우, 하이퍼 그래디언트 추정이 신뢰할 수 없음
- **베이지안 최적화**, **메타 학습**, **합성 데이터 생성**과의 결합을 고려해야 함

#### (3) 동적 불균형 데이터 (Non-stationary imbalance)
- 현재 프레임워크는 훈련 중 클래스 분포가 고정됨을 가정
- 온라인 학습이나 분포 변화(distribution shift) 환경에서의 적용 연구 필요

#### (4) 다목적 공정성 기준 동시 최적화
- 현재 상위 레벨에서 단일 공정성 지표를 최적화하나, **DEO + 균형 정확도 + 표준 정확도**를 동시에 최적화하는 다목적 최적화(Pareto multi-objective optimization) 연구 확장 가능

#### (5) 사전 학습 모델(Foundation Models)과의 통합
- CLIP, GPT, Vision Transformers 등 대규모 사전 학습 모델의 파인튜닝 시, 불균형 클래스 문제가 더욱 심화됨
- AutoBalance의 이중 최적화를 파인튜닝 파이프라인에 통합하는 연구 필요

#### (6) 이론적 수렴 보장 강화
- 현재 IFT 기반 접근은 고정점(fixed point) 가정이 필요하며, 이것이 실제로 만족되는지 이론적으로 불명확함
- 더 강한 수렴 보장을 위한 수학적 분석이 요구됨

#### (7) 연합 학습에서의 적용
- 클라이언트별 데이터 분포 이질성이 심한 연합 학습 환경에서, 각 클라이언트가 로컬 검증 데이터로 손실 함수를 적응적으로 조정하는 방향 탐구 가능

---

## 참고자료 (출처)

본 답변은 다음 자료를 기반으로 작성되었습니다:

1. **주요 논문**: Mingchen Li, Xuechen Zhang, Christos Thrampoulidis, Jiasi Chen, Samet Oymak. *"AutoBalance: Optimized Loss Functions for Imbalanced Data."* NeurIPS 2021. (제공된 PDF 원문)

2. **인용 논문 (원문 내 참고문헌)**:
   - [8] Cao et al. (2019). *"Learning imbalanced datasets with label-distribution-aware margin loss."* arXiv:1906.07413
   - [38] Kini, Paraskevas, Oymak, Thrampoulidis (2021). *"Label-imbalanced and group-sensitive classification under overparameterization."* NeurIPS 2021
   - [53] Menon et al. (2020). *"Long-tail learning via logit adjustment."* arXiv:2007.07314
   - [46] Lorraine, Vicol, Duvenaud (2020). *"Optimizing millions of hyperparameters by implicit differentiation."* AISTATS 2020
   - [55] Oymak, Li, Soltanolkotabi (2021). *"Generalization guarantees for neural architecture search with train-validation split."* ICML 2021
   - [58] Sagawa et al. (2019). *"Distributionally robust neural networks for group shifts."* arXiv:1911.08731
   - [67] Ye et al. (2020). *"Identifying and compensating for feature deviation in imbalanced deep learning."* arXiv:2001.01385
   - [24] Hataya et al. (2020). *"Meta approach to data augmentation optimization."* arXiv:2006.07965
   - [12] Cubuk et al. (2018). *"AutoAugment: Learning augmentation policies from data."* arXiv:1805.09501

3. **소스 코드**: https://github.com/ucr-optml/AutoBalance

> **정확도 관련 고지**: 2020년 이후 AutoBalance 이후에 독립적으로 발표된 후속 연구(예: 2022~2024년 관련 논문)에 대한 비교는 본 PDF에 포함되지 않은 정보이므로, 제공된 논문 원문 내 언급된 내용을 중심으로만 서술하였습니다.
