# Correlated Input-Dependent Label Noise in Large-Scale Image Classification

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

이 논문의 핵심 주장은 **대규모 이미지 분류 데이터셋에서 발생하는 레이블 노이즈는 클래스 간 상관관계(inter-class correlation)를 갖는 입력 의존적(input-dependent) 특성**을 지닌다는 것입니다. 기존의 표준 신경망 학습은 노이즈를 i.i.d.(독립 동일 분포) Gumbel 분포로 암묵적으로 가정하는데, 이는 실제 레이블 노이즈의 구조를 포착하지 못한다고 주장합니다. 저자들은 **다변량 정규분포(multivariate Normal)** 잠재변수를 신경망 분류기의 최종 은닉층에 배치하여 이를 해결합니다.

### 주요 기여

| 기여 항목 | 설명 |
|-----------|------|
| 새로운 확률적 노이즈 모델 | 클래스 간 상관 레이블 노이즈를 모델링하는 방법 제안 |
| 대규모 확장성 | Imagenet-21k(21K 클래스), JFT(300M 이미지)까지 확장 |
| 정량적 성능 향상 | ILSVRC12 +2.6%, WebVision SOTA, JFT +1.6% |
| 공분산 구조의 해석 가능성 | 의미적으로 유사하거나 공존하는 클래스 쌍의 노이즈 상관관계 학습 |
| 전이 학습 개선 | VTAB 19개 벤치마크에서 더 일반화된 표현 학습 |
| 드롭인 구현 제공 | TensorFlow Keras 레이어로 간단한 코드 교체 가능 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

대규모 이미지 분류 데이터셋(ImageNet, WebVision, JFT 등)에는 다음과 같은 구조적 레이블 노이즈가 존재합니다:

1. **시각적 유사성(Visual Similarity)**: Appenzeller vs. EntleBucher처럼 인간 주석자도 구별하기 어려운 클래스
2. **공존 관계(Co-occurrence)**: 웹 텍스트 기반 자동 레이블링에서 함께 등장하는 객체들의 혼동
3. **비균일 노이즈(Non-uniform Noise)**: 이미지마다 노이즈 수준이 다름 (heteroscedastic)

기존 표준 학습은 노이즈를 다음과 같이 암묵적으로 가정합니다:

$$\epsilon_j \sim \text{i.i.d.} \; G(0, 1) \; \forall j$$

이 가정은 두 가지 문제를 야기합니다:
- **Identical 가정 위반**: 클래스마다 노이즈 수준이 다름
- **Independence 가정 위반**: 유사 클래스들의 노이즈는 상관관계를 가짐

---

### 2.2 제안하는 방법 (수식 포함)

#### 생성 과정 (Generative Process)

입력 $\mathbf{x}$에 대해 유틸리티 벡터 $\mathbf{u}(\mathbf{x}) \in \mathbb{R}^K$를 정의합니다:

$$\mathbf{u}(\mathbf{x}) = \boldsymbol{\mu}(\mathbf{x}) + \boldsymbol{\epsilon}$$

레이블 생성 확률:

```math
p_c = P(y = c \mid \mathbf{x}) = \int \mathbf{1}\left\{\arg\max_{j \in [K]} u_j(\mathbf{x}) = c\right\} p(\boldsymbol{\epsilon}) \, d\boldsymbol{\epsilon}
```

#### 표준 Softmax와의 연결

노이즈가 i.i.d. Gumbel일 때 닫힌 형태 해(closed-form solution)가 존재합니다:

$$p_c = \frac{\exp(\mu_c)}{\sum_{j=1}^{K} \exp(\mu_j)} \iff \epsilon_j \sim \text{i.i.d.} \; G(0, 1) \; \forall j $$

즉, **표준 softmax cross-entropy 학습은 이미 i.i.d. Gumbel 노이즈를 암묵적으로 가정**하고 있습니다.

#### 제안 방법: 다변량 정규 노이즈

저자들은 노이즈를 다변량 정규분포로 가정합니다:

$$\boldsymbol{\epsilon}(\mathbf{x}) \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma}(\mathbf{x}))$$

이때 닫힌 형태 해가 존재하지 않으므로, **Monte Carlo 추정 + 온도 매개변수화 softmax**로 근사합니다:

$$p_c \approx \frac{1}{S} \sum_{i=1}^{S} \left(\text{softmax}_\tau \, \mathbf{u}^{(i)}(\mathbf{x})\right)_c, \quad \mathbf{u}^{(i)}(\mathbf{x}) \sim \mathcal{N}(\boldsymbol{\mu}(\mathbf{x}), \boldsymbol{\Sigma}(\mathbf{x})) $$

여기서 $\tau > 0$는 편향-분산 트레이드오프를 제어하는 온도 하이퍼파라미터입니다.

#### 공분산 행렬의 효율적 파라미터화

$K \times K$ 공분산 행렬을 직접 계산하는 것은 메모리/계산 비용이 너무 큽니다. **저랭크 근사(Low-rank Approximation)**를 사용합니다:

$$\boldsymbol{\Sigma}(\mathbf{x}) = V(\mathbf{x})V(\mathbf{x})^\top + \text{diag}(\mathbf{d}(\mathbf{x})^2)$$

여기서 $V(\mathbf{x})$는 $K \times R$ 행렬, $R \ll K$입니다.

샘플링은 다음과 같이 수행됩니다:

$$\boldsymbol{\epsilon} = \mathbf{d}(\mathbf{x}) \odot \boldsymbol{\epsilon}_K + V(\mathbf{x})\boldsymbol{\epsilon}_R, \quad \boldsymbol{\epsilon}_K \sim \mathcal{N}(\mathbf{0}_K, I_{K \times K}), \; \boldsymbol{\epsilon}_R \sim \mathcal{N}(\mathbf{0}_R, I_{R \times R})$$

#### 파라미터 효율적 버전 (대규모 클래스용)

Imagenet-21k(21K 클래스), JFT(17K 클래스)에서는 추가로 다음을 사용합니다:

$$V(\mathbf{x}) = \mathbf{v}(\mathbf{x})\mathbf{1}_R^\top \odot V$$

이때 파라미터 수는 $\mathcal{O}(DKR)$에서 $\mathcal{O}(DK + KR)$로 축소됩니다 (Imagenet-21k에서 약 50배 감소).

#### 헤테로스케다스틱 회귀와의 연관성 (참고)

Bishop & Quazaz의 헤테로스케다스틱 회귀 NLL:

$$\frac{1}{N}\sum_{i=1}^{N} \frac{1}{2\sigma(\mathbf{x}_i)^2}(y_i - \mu(\mathbf{x}_i))^2 + \frac{1}{2}\log\sigma(\mathbf{x}_i)^2 $$

이 논문의 접근법은 위를 분류 문제로 확장한 것입니다.

#### 다중 레이블 분류로의 확장

$$p_c \approx \frac{1}{S}\sum_{i=1}^{S} \text{sigmoid}_\tau\, u^{(i)}(\mathbf{x})_c, \quad \mathbf{u}^{(i)}(\mathbf{x}) \sim \mathcal{N}(\boldsymbol{\mu}(\mathbf{x}), \boldsymbol{\Sigma}(\mathbf{x})) $$

---

### 2.3 모델 구조

```
입력 이미지 x
      │
      ▼
[공유 백본 네트워크 f_θ(x)]  (ResNet-50/152, InceptionResNet-v2 등)
      │
      ├──────────────────────────────┐
      ▼                              ▼
[평균 파라미터]               [공분산 파라미터]
μ(x) = W_μ r(x) + b_μ        d(x) = W_d r(x) + b_d
                               v(x) = W_v r(x) + b_v (PE 버전)
      │                              │
      └──────────────────────────────┘
                     │
                     ▼
         [Monte Carlo 샘플링]
    u^(i)(x) ~ N(μ(x), Σ(x)), i=1,...,S
                     │
                     ▼
         [Temperature Softmax_τ]
                     │
                     ▼
         p_c = mean(softmax_τ(U(x)), axis=1)[c]
```

**알고리즘 1 요약 (Algorithm 1)**:
1. 공유 표현 $\mathbf{r}(\mathbf{x}) = f^\theta(\mathbf{x})$ 계산
2. 평균 $\boldsymbol{\mu}(\mathbf{x})$, 대각 보정 $\mathbf{d}(\mathbf{x})$ 계산
3. 저랭크 성분 $V(\mathbf{x})$ 계산 (또는 파라미터 효율적 버전)
4. $S$개의 표준정규 샘플 생성 후 $U(\mathbf{x})$ 구성
5. $p_c = \text{mean}(\text{softmax}_\tau(U(\mathbf{x})), \text{axis}=1)[c]$

---

### 2.4 성능 향상

#### ILSVRC12 (ResNet-152, 270 epochs)

| 방법 | Top-1 Acc | Top-5 Acc | NLL |
|------|-----------|-----------|-----|
| Homoscedastic | 76.7% | 92.9% | 1.08 |
| Het. Diag [9] | 78.7% | 94.0% | 0.95 |
| **Het. Full (ours)** | **79.3%** | **94.5%** | **0.92** |

#### WebVision 1.0

| 방법 | WebVision Top-1 | WebVision Top-5 |
|------|----------------|----------------|
| Cao et al. [6] (이전 SOTA) | 75.0% | 90.6% |
| **Het. Full (ours)** | **76.6%** | **92.1%** |

#### Imagenet-21k / JFT (gAP)

| 방법 | 21k gAP | JFT gAP |
|------|---------|---------|
| Homoscedastic | 45.9 | 63.1 |
| Het. Diag $\tau^*$ | 46.8 | 64.1 |
| **Het. PE (ours)** | **47.0** | **64.7** |

#### Deep Ensemble (ResNet-50, ILSVRC12)

| 방법 | Top-1 Acc | NLL | ECE |
|------|-----------|-----|-----|
| Hom. Single | 76.1% | 0.943 | 0.0392 |
| Hom. Ensemble 4× | 77.5% | 0.877 | 0.0305 |
| Het. Single | 77.5% | 0.898 | 0.033 |
| **Het. Ensemble 4×** | **79.5%** | **0.79** | **0.015** |

---

### 2.5 한계점

1. **계산 비용**: Monte Carlo 샘플링(기본 10,000회)으로 인한 추가 연산 부담
2. **하이퍼파라미터 의존성**: 소프트맥스 온도 $\tau$와 저랭크 차원 $R$을 검증 세트로 튜닝해야 함
3. **근사 오차**: argmax를 softmax로 대체하는 과정에서 $\tau > 0$일 때 편향 발생
4. **파라미터 증가**: Het. Full 방법은 표준 모델 대비 파라미터 수 증가 (단, 부록 E에서 파라미터 수를 동일하게 해도 +0.4%에 그침을 확인)
5. **아키텍처 제약**: 현재 실험은 ResNet 계열에 집중되어 있음
6. **저랭크 근사의 한계**: 완전한 공분산 구조를 표현하지 못할 수 있음
7. **도메인 특이성**: 의료 영상이나 위성 이미지 등 다른 도메인에서의 검증이 부족

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 노이즈 레이블로부터의 과적합 방지

이 논문의 핵심 일반화 메커니즘은 **노이즈 레이블에 과적합하지 않는 것**입니다. 논문에서 신경망이 먼저 깨끗한 레이블을 학습하고 나서 노이즈 레이블을 학습한다는 선행 연구[27]를 인용하며, 헤테로스케다스틱 모델이 노이즈 레이블을 "설명(explain away)"하여 더 오랜 학습 스케줄의 혜택을 받을 수 있음을 보입니다:

- Homoscedastic: 270 epochs vs 90 epochs Top-1 Acc 증가 **+0.2%**
- Het. Diag: **+0.4%**
- Het. Full: **+0.8%**

### 3.2 Taylor Series 분석을 통한 일반화 메커니즘 이해

2차 Taylor Series 근사를 통한 로그 우도 분해 (대각 공분산 경우):

$$\log \mathbb{E}_{\boldsymbol{\epsilon}}\left[s_k(W^\top\mathbf{x} + V\boldsymbol{\epsilon})\right] \approx \log(s_k) - \frac{1}{2}\sum_{j \neq k}^{K} s_j(1-2s_j)\sigma_j^2 + \frac{1}{2}(1-s_k)(1-2s_k)\sigma_k^2 $$

여기서:
- $\log(s_k)$: 표준 호모스케다스틱 로그 우도
- 나머지 항: 공분산 행렬의 근사 효과

**핵심 해석**: 잘못 분류된 예시($s_j > 0.5$)에서 $\sigma_j^2$을 증가시켜 노이즈 레이블을 설명하고, 올바르게 분류된 예시에서는 $\sigma_k^2$을 감소시켜 확신도를 높입니다.

전체 공분산 경우:

$$\log \mathbb{E}\left[s_k(W^\top\mathbf{x} + V\boldsymbol{\epsilon})\right] \approx \log(s_k) + \frac{1}{2}\sum_{i \neq j, i,j \neq k}^{K} 2s_is_j\Sigma_{ij} + \frac{1}{2}\sum_{j \neq k}^{K} -s_j(1-2s_k)\Sigma_{jk} + \frac{1}{2}\sum_{j \neq k}^{K} -s_j(1-2s_j)\Sigma_{jj} + \frac{1}{2}(1-s_k)(1-2s_k)\Sigma_{kk} $$

- 오프 대각 항 $\Sigma_{ij}$는 두 클래스가 동시에 높은 확률을 가질 때 **공존(co-occurrence)** 패턴을 학습
- $\Sigma_{jk}$ 항은 **대체 클래스(substitute)** 패턴을 학습

### 3.3 VTAB 전이 학습 실험

가장 중요한 일반화 증거는 **Visual Task Adaptation Benchmark (VTAB)** 실험입니다:

| 방법 | VTAB1K Score |
|------|-------------|
| Homoscedastic | 70.46 ± 0.5 |
| Het. Diag | 71.12 ± 0.19 |
| **Het. PE (ours)** | **71.34 ± 0.23** |

**중요 포인트**: 다운스트림 파인튜닝 시 헤테로스케다스틱 출력 레이어가 **제거**되고 표준 출력 레이어로 교체됩니다. 즉, 성능 향상은 순전히 **상류(upstream) 학습에서 더 나은 특징 표현을 학습**했기 때문입니다.

이는 노이즈 레이블 모델링이 단순히 분류 정확도를 높이는 것을 넘어 **더 의미론적으로 풍부하고 전이 가능한 표현**을 학습하게 함을 시사합니다.

### 3.4 공분산 구조의 질적 분석

학습된 평균 공분산(ILSVRC12 검증 세트 50,000장 평균):

| Class A | Class B | Avg. Cov. |
|---------|---------|-----------|
| partridge | ruffed grouse | -0.46 |
| projectile, missile | missile | -0.44 |
| Welsh springer spaniel | Blenheim spaniel | +0.28 |
| French bulldog | Boston terrier | +0.27 |

약 100만 개의 클래스 쌍 중 최상위 절대 공분산을 갖는 쌍들이 모두 의미론적으로 관련된 쌍임은 모델이 구조화된 노이즈를 학습했음을 보여줍니다.

---

## 4. 미래 연구에 미치는 영향 및 고려사항

### 4.1 미래 연구에 미치는 영향

#### 4.1.1 불확실성 추정 연구

이 논문은 **알레아토릭(aleatoric) 불확실성**과 **에피스테믹(epistemic) 불확실성**을 결합하는 방향을 제시합니다. Deep Ensemble과의 결합 실험(ECE 0.015)은 두 불확실성을 동시에 모델링하는 것이 효과적임을 보여, 불확실성 추정 연구에 새로운 방향을 제시합니다.

#### 4.1.2 대규모 언어/시각-언어 모델로의 확장 가능성

이 방법론은 CLIP, ALIGN 등의 대규모 시각-언어 모델의 학습 데이터에도 적용 가능합니다. 인터넷에서 자동 수집된 이미지-텍스트 쌍의 노이즈를 모델링하는 데 직접적으로 응용할 수 있습니다.

#### 4.1.3 레이블 노이즈 연구 패러다임 변화

기존 레이블 노이즈 연구는 주로 균일 노이즈(symmetric noise)나 비대칭 노이즈(asymmetric noise)를 가정했습니다. 이 논문은 **입력 의존적 공분산 구조**를 통해 실제 노이즈의 복잡한 구조를 모델링하는 새로운 패러다임을 제시합니다.

#### 4.1.4 전이 학습 연구

JFT → VTAB 실험 결과는 **더 나은 상류 학습이 다운스트림 전이를 개선**한다는 것을 보여주며, 사전 학습 방법론 연구에 중요한 시사점을 제공합니다.

### 4.2 미래 연구 시 고려사항

#### 4.2.1 트랜스포머 아키텍처와의 통합

현재 실험은 ResNet 기반 아키텍처에 집중되어 있습니다. Vision Transformer(ViT)나 Swin Transformer 등 최신 아키텍처와의 통합 및 성능 검증이 필요합니다.

#### 4.2.2 Monte Carlo 샘플링의 효율화

현재 방법은 10,000개의 MC 샘플을 사용합니다. 표 9에서 100개 이상부터 수확 체감이 발생함을 보였지만, 더 효율적인 추정 방법(예: 중요도 샘플링, 준몬테카를로 방법)이 필요합니다.

#### 4.2.3 온도 하이퍼파라미터 자동 학습

현재 $\tau$는 검증 세트에서 수동으로 튜닝됩니다. 이를 학습 중에 자동으로 최적화하는 방법이 고려되어야 합니다.

#### 4.2.4 다른 모달리티로의 확장

음성 인식, 자연어 처리, 의료 데이터 등 레이블 노이즈가 구조화된 상관관계를 가지는 다른 도메인에서의 적용 연구가 필요합니다.

#### 4.2.5 공분산 행렬의 동적 랭크 조정

현재는 랭크 $R$이 고정되어 있습니다. 데이터나 클래스 구조에 따라 동적으로 랭크를 조정하는 방법이 성능을 더 향상시킬 수 있습니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 핵심 방법 | 노이즈 모델 | 확장성 | 비고 |
|------|-----------|-----------|--------|------|
| **본 논문 (Collier et al., 2021)** | 다변량 정규 잠재변수 + 저랭크 공분산 | 입력 의존적, 클래스 간 상관 | 300M 이미지, 21K 클래스 | VTAB 전이 개선 |
| **Cao et al. (2020)** [6] | 헤테로스케다스틱 적응 정규화 | 대각, 입력 의존적 | 중간 규모 | WebVision 75.0% |
| **MentorMix (Jiang et al., 2020)** [24] | MentorNet + Mixup | 스칼라 가중치 | 중간 규모 | WebVision 74.3% |
| **Stochastic Segmentation Networks (Monteiro et al., 2020)** [33] | 픽셀 간 공분산 모델링 | 공간적 상관, 전체 공분산 | 의료 분할에 집중 | 저랭크 근사 사용, 온도 미적용 |
| **Noisy Labels with Partial Labels (2021~)** | 부분 레이블 학습 | 균일/비대칭 노이즈 | - | 레이블 집합 수준 노이즈 모델링 |

### 주요 차별점

1. **공분산 구조의 완전성**: 기존 연구(Kendall & Gal, Collier et al. 2020)는 대각 공분산을 사용했으나, 본 논문은 오프 대각 항을 통한 클래스 간 상관을 명시적으로 모델링합니다.

2. **확장성의 획기적 개선**: Stochastic Segmentation Networks[33]는 의료 영상 분할에만 적용되었으나, 본 논문은 파라미터 효율적 버전을 통해 21K 클래스, 300M 이미지까지 확장합니다.

3. **이론적 분석**: Taylor Series 분석(Appendix C)을 통해 공분산 행렬이 로그 우도에 미치는 영향을 수식으로 명확히 설명합니다.

4. **전이 학습 관점**: 단순 분류 성능 향상을 넘어 더 나은 표현 학습이 전이 학습을 개선함을 VTAB 벤치마크로 검증합니다.

---

## 참고 자료

**주 논문**:
- Collier, M., Mustafa, B., Kokiopoulou, E., Jenatton, R., & Berent, J. (2021). *Correlated Input-Dependent Label Noise in Large-Scale Image Classification*. arXiv:2105.10305v1.

**논문 내 주요 인용 문헌**:
- [9] Collier et al. (2020). *A simple probabilistic method for deep classification under input-dependent label noise*. arXiv:2003.06778.
- [25] Kendall, A. & Gal, Y. (2017). *What uncertainties do we need in Bayesian deep learning for computer vision?* NeurIPS.
- [28] Lakshminarayanan et al. (2017). *Simple and scalable predictive uncertainty estimation using deep ensembles*. NeurIPS.
- [33] Monteiro et al. (2020). *Stochastic Segmentation Networks*. arXiv:2006.06015.
- [47] Zhai et al. (2020). *A large-scale study of representation learning with the Visual Task Adaptation Benchmark*. ICLR.
- [42] Train, K.E. (2009). *Discrete Choice Methods with Simulation*. Cambridge University Press.
- [6] Cao et al. (2020). *Heteroskedastic and imbalanced deep learning with adaptive regularization*. arXiv:2006.15766.
- [24] Jiang et al. (2020). *Beyond synthetic noise: Deep learning on controlled noisy labels*. ICML.
- [2] Beyer et al. (2020). *Are we done with ImageNet?* arXiv:2006.07159.
- [19] He et al. (2016). *Deep residual learning for image recognition*. CVPR.
