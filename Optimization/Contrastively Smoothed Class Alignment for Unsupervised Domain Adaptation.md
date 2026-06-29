# Contrastively Smoothed Class Alignment for Unsupervised Domain Adaptation

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

기존 비지도 도메인 적응(UDA) 방법들은 **글로벌 분포 정렬**에 집중하거나, MCD처럼 **분류기 불일치(classifier discrepancy)**를 활용해 클래스 경계를 고려하더라도, 결정 경계 근방의 **모호한(ambiguous) 타겟 샘플**이 저차원 매니폴드 상에서 비부드럽게(non-smoothly) 분포하는 문제를 해결하지 못한다. CoSCA는 이 문제를 **대조적 손실(contrastive loss)**과 **MMD 기반 전역 정렬**을 결합하여 해결한다.

### 주요 기여

1. **CoSCA 모델 제안**: 클래스 내(intra-class) 및 클래스 간(inter-class) 도메인 불일치를 명시적으로 통합하는 대조적 손실을 통해 모호한 타겟 샘플을 더 잘 정렬
2. **전역 정렬 강화**: MMD 손실을 추가하여 소스-타겟 도메인 간 고차 모멘트 매칭을 통한 글로벌 정렬 개선
3. **다양한 벤치마크 검증**: 시각적(digit, CIFAR/STL, VisDA) 및 비시각적(Amazon Reviews) 도메인 적응 태스크에서 당시 SOTA 달성

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

#### 기존 방법의 한계

| 방법 | 문제점 |
|------|--------|
| DANN 등 적대적 방법 | 전역 주변 분포만 정렬, 클래스 조건부 경계 무시 |
| MCD | 클래스 경계 고려하지만, 타겟 샘플이 매니폴드 상에서 비부드럽게 분포 → 오분류 발생 |
| 모든 기존 방법 | 클래스 내 불일치만 고려, 클래스 간 불일치 무시 |

**핵심 문제**: 결정 경계 근방의 모호한 타겟 샘플들이 낮은 차원의 데이터 매니폴드 상에서 부드럽게 분포하지 않아, 이웃 샘플과 다른 클래스에 속할 수 있음.

---

### 2.2 제안 방법 (수식 포함)

#### Step 1: 소스 도메인 분류 손실

$$\mathcal{L}(\mathbf{X}^s, \mathbf{Y}^s) = -\mathbb{E}_{(\mathbf{x}^s, y^s) \sim (\mathbf{X}^s, \mathbf{Y}^s)} \left[ \sum_{k=1}^{K} \mathbb{1}_{[k=y^s]} \log p_1(\mathbf{y}|\mathbf{x}^s) + \sum_{k=1}^{K} \mathbb{1}_{[k=y^s]} \log p_2(\mathbf{y}|\mathbf{x}^s) \right] $$

#### Step 2: MMD 기반 전역 정렬

$$\mathcal{L}_{\text{MMD}}(\mathbf{X}^s, \mathbf{X}^t) = \sum_{i=1}^{n_s} \sum_{j=1}^{n_t} k\!\left(\phi\!\left(\frac{\mathbf{g}_s}{\|\mathbf{g}_s\|}\right),\ \phi\!\left(\frac{\mathbf{g}_t}{\|\mathbf{g}_t\|}\right)\right) $$

$$\mathbf{g}_s = \frac{1}{n_s} \sum_{i=1}^{n_s} G(\mathbf{x}_i^s), \quad \mathbf{g}_t = \frac{1}{n_t} \sum_{i=1}^{n_t} G(\mathbf{x}_i^t)$$

- $\phi(\cdot)$: 커널 매핑, $\|\cdot\|$: $\ell_2$-노름
- **역할**: 소스와 타겟 특징의 정규화된 평균을 힐베르트 공간에서 매칭 → 고차 모멘트 정렬

#### Step 3: 분류기 불일치 손실 (Adversarial Loss)

$$d(p_1(\mathbf{y}|\mathbf{x}), p_2(\mathbf{y}|\mathbf{x})) = \frac{1}{K} \sum_{k=1}^{K} \left| p_{1k}(\mathbf{y}|\mathbf{x}) - p_{2k}(\mathbf{y}|\mathbf{x}) \right| $$

$$\mathcal{L}_{\text{adv}}(\mathbf{X}^t) = \mathbb{E}_{\mathbf{x}^t \sim \mathbf{X}^t} \left[ d(p_1(\mathbf{y}|\mathbf{x}^t), p_2(\mathbf{y}|\mathbf{x}^t)) \right] $$

MCD 프레임워크의 적대적 학습:

$$\min_{F_1, F_2} \mathcal{L}(\mathbf{X}^s, \mathbf{Y}^s) - \lambda \mathcal{L}_{\text{adv}}(\mathbf{X}^t) $$

$$\min_{G} \mathcal{L}_{\text{adv}}(\mathbf{X}^t) $$

#### Step 4: 대조적 손실 (핵심 기여)

**타겟 샘플의 의사 레이블(pseudo label) 예측:**

```math
\tilde{y}_j^t = \arg\max_{k \in \{1,2,\ldots,K\}} \left\{ p(F_1(G(\mathbf{x}_j^t)) = k|\mathbf{x}) + p(F_2(G(\mathbf{x}_j^t)) = k|\mathbf{x}) \right\}
```

**지시 함수:**

$$c(y, y') = \begin{cases} 1, & y = y' \\ 0, & y \neq y' \end{cases}$$

**거리 측도 (Siamese 네트워크 기반):**

$$L_{\text{dis}} = \begin{cases} \|G(\mathbf{x}_i) - G(\mathbf{x}_j)\|^2 & c_{ij} = 1 \\ \max(0,\ m - \|G(\mathbf{x}_i) - G(\mathbf{x}_j)\|)^2 & c_{ij} = 0 \end{cases} $$

- $m$: 사전 정의된 마진, $c_{ij} = c(y_i, y_j)$

**소스-타겟 대조 손실:**

$$\mathcal{L}_{\text{contras}}^{\mathcal{S} \leftrightarrow \mathcal{T}} = \sum_{\mathbf{x}_i^s \in \mathcal{S},\ \mathbf{x}_j^t \in \mathcal{T}} L_{\text{dis}}(G(\mathbf{x}_i^s),\ G(\mathbf{x}_j^t),\ c(y_i^s, \tilde{y}_j^t)) $$

**타겟-타겟 대조 손실:**

$$\mathcal{L}_{\text{contras}}^{\mathcal{T} \leftrightarrow \mathcal{T}} = \sum_{\mathbf{x}_i^t, \mathbf{x}_j^t \in \mathcal{T}} L_{\text{dis}}(G(\mathbf{x}_i^t),\ G(\mathbf{x}_j^t),\ c(\tilde{y}_i^t, \tilde{y}_j^t)) $$

**전체 대조 손실:**

$$\mathcal{L}_{\text{contras}}(\mathbf{X}^s, \mathbf{Y}^s, \mathbf{X}^t) = \mathcal{L}_{\text{contras}}^{\mathcal{S} \leftrightarrow \mathcal{T}} + \mathcal{L}_{\text{contras}}^{\mathcal{T} \leftrightarrow \mathcal{T}} $$

---

### 2.3 모델 구조

```
[학습 단계]
                    ┌──────────────────────────────────┐
X^s (레이블 있음) → │                                  │→ L(X^s, Y^s)
                    │  G (특징 생성기)                  │→ L_MMD (글로벌 정렬)
X^t (레이블 없음) → │  9개 Conv층 + Dropout + MaxPool   │
                    │  + Gaussian Noise + GlobalPool    │
                    └──────────────────────────────────┘
                              ↓ 특징 g_s, g_t
               ┌─────────────────────────┐
               │  F1 (MLP 분류기 1)       │→ p1(y|x)  ─┐
               │  F2 (MLP 분류기 2)       │→ p2(y|x)  ─┤→ L_adv, L_contras
               └─────────────────────────┘             │
                                          의사 레이블 ←─┘
```

#### 학습 절차 (Algorithm 1)

| 단계 | 업데이트 대상 | 목적함수 |
|------|-------------|---------|
| 1 | $G, F_1, F_2$ | $\min_{F_1,F_2,G}\ \mathcal{L}(\mathbf{X}^s, \mathbf{Y}^s) + \lambda_1 \mathcal{L}_{\text{MMD}}(\mathbf{X}^s, \mathbf{X}^t)$ |
| 2 | $F_1, F_2$ (G 고정) | $\min_{F_1,F_2}\ \mathcal{L}(\mathbf{X}^s, \mathbf{Y}^s) - \lambda_2 \mathcal{L}_{\text{adv}}(\mathbf{X}^t)$ |
| 3 | $G$ ($F_1, F_2$ 고정) | $\min_{G}\ \lambda_2 \mathcal{L}\_{\text{adv}}(\mathbf{X}^t) + \lambda_3 \mathcal{L}_{\text{contras}}(\mathbf{X}^s, \mathbf{Y}^s, \mathbf{X}^t)$ |

**동적 파라미터화:**

$$\omega(t) = \exp\!\left[-\theta\!\left(1 - \frac{t}{\text{max-epochs}}\right)\right]\!\lambda_3$$

초기 학습에서 의사 레이블의 불신뢰성을 보상하기 위해 $\lambda_3$를 점진적으로 증가시킴.

---

### 2.4 성능 향상

#### 시각적 도메인 적응 결과 (Table 1)

| 소스→타겟 | MCD | CoSCA | 향상 |
|---------|-----|-------|------|
| MNIST→SVHN | 68.7% | **80.7%** | +12.0% |
| SVHN→MNIST | 96.2% | **98.7%** | +2.5% |
| MNIST→MNISTM | 96.7% | **98.9%** | +2.2% |
| MNIST→USPS | 94.2% | **99.3%** | +5.1% |
| CIFAR→STL | 78.1% | **81.7%** | +3.6% |
| STL→CIFAR | 69.2% | **75.2%** | +6.0% |

#### VisDA 대규모 데이터셋 (Table 2, ResNet101)

| 모델 | 평균 정확도 |
|------|-----------|
| MCD | 77.1% |
| CAN | 77.9% |
| SEDA | 82.2% |
| **CoSCA** | **82.9%** |

#### 비시각적 태스크 - Amazon Reviews (Table 3)

$$\text{CoSCA: } 83.17\% \quad \text{vs} \quad \text{DAS (이전 SOTA): } 81.96\%$$

#### Ablation Study (Table 4, MNIST→SVHN)

| 모델 | 정확도 |
|------|-------|
| MCD | 68.7% |
| MCD+MMD | 72.1% (+3.4%) |
| MCD+Contras | 75.9% (+7.2%) |
| **CoSCA (전체)** | **80.7%** (+12.0%) |

---

### 2.5 한계점

1. **의사 레이블 의존성**: 타겟 도메인 레이블 추정이 부정확할 경우, 대조 손실의 학습 방향이 왜곡될 수 있음
2. **하이퍼파라미터 민감성**: $\lambda_1, \lambda_2, \lambda_3$, 마진 $m$, 내부 루프 반복 수 $\tau, \delta$ 등 다수의 하이퍼파라미터 튜닝 필요
3. **계산 비용**: 클래스 인식 샘플링(class-aware sampling) 및 이중 분류기 구조로 인한 추가 연산
4. **이론적 해석 부재**: 대조 학습이 도메인 적응에 미치는 효과에 대한 이론적 분석이 부족함 (저자도 인정)
5. **대규모 클래스 수 확장성**: 클래스 수 $K$가 매우 클 때 클래스 인식 샘플링 전략의 효율성 저하 가능

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화를 위한 핵심 메커니즘

#### (a) 저밀도 영역 가정(Low-Density Assumption) 활용

대조 손실은 결정 경계가 **저밀도 영역**에 위치하도록 유도함으로써 타겟 도메인에서의 일반화를 강화한다. 수식 (9)의 마진 손실:

$$L_{\text{dis}} = \max(0,\ m - \|G(\mathbf{x}_i) - G(\mathbf{x}_j)\|)^2 \quad (c_{ij} = 0)$$

서로 다른 클래스의 샘플 간 거리를 마진 $m$ 이상으로 유지하여, 특징 공간에서 클래스 간 분리를 극대화한다.

#### (b) 매니폴드 부드러움(Manifold Smoothness) 강화

$\mathcal{L}_{\text{contras}}^{\mathcal{T} \leftrightarrow \mathcal{T}}$는 타겟 도메인 내 동일 클래스 샘플들을 응집시켜, **저차원 매니폴드 상에서의 부드러운 분포**를 달성한다. 이는 새로운 타겟 샘플에 대한 일반화 능력을 향상시킨다.

#### (c) 전역 + 지역 정렬의 이중 구조

$$\underbrace{\mathcal{L}_{\text{MMD}}}_{\text{전역 분포 정렬}} + \underbrace{\mathcal{L}_{\text{contras}}}_{\text{클래스 조건부 지역 정렬}}$$

전역 정렬이 없을 경우 대도메인 갭(MNIST→SVHN 등)에서 실패하고, 지역 정렬이 없을 경우 클래스 경계 근방의 모호한 샘플이 오분류된다. 두 손실의 결합이 일반화의 핵심이다.

#### (d) 동적 파라미터화를 통한 점진적 학습

$$\omega(t) = \exp\!\left[-\theta\!\left(1 - \frac{t}{\text{max-epochs}}\right)\right]\!\lambda_3$$

초기 학습 시 불신뢰한 의사 레이블의 영향을 최소화하고, 학습이 진행될수록 대조 손실의 비중을 증가시켜 **안정적인 수렴**과 일반화를 동시에 달성한다.

#### (e) 도메인 불가지론적 설계

- 비시각적 태스크(Amazon Reviews)에도 동일한 프레임워크 적용 성공
- VisDA(합성→실제, 152K 샘플) 대규모 데이터셋에서도 효과적
- 이는 특정 아키텍처나 도메인에 종속되지 않는 **범용적 일반화** 가능성을 시사

---

## 4. 앞으로의 연구에 미치는 영향 및 고려 사항

### 4.1 연구에 미치는 영향

#### (a) 대조 학습과 도메인 적응의 결합 방향 제시

CoSCA는 당시 컴퓨터 비전에서 부상하던 **자기지도 대조 학습(Self-supervised Contrastive Learning)**의 아이디어를 도메인 적응에 선제적으로 결합함으로써, 이후 연구의 방향을 제시했다. 특히 SimCLR, MoCo 등의 대조 학습 프레임워크와 UDA의 결합을 자극하는 계기가 되었다.

#### (b) 클래스 조건부 정렬의 중요성 재확인

단순한 분포 매칭을 넘어 **클래스 레벨의 정렬**이 필수적임을 실험적으로 입증하였으며, 이후 많은 연구들이 클래스 조건부 정렬을 기본 구성 요소로 채택하게 되었다.

#### (c) 의사 레이블 활용 패러다임

타겟 도메인의 의사 레이블을 활용한 반복적 자기 학습 방식은 이후 **자기 훈련(Self-Training) 기반 UDA** 연구에 영향을 미쳤다.

---

### 4.2 2020년 이후 관련 최신 연구 비교 분석

> **주의**: 아래 연구들은 CoSCA 이후의 흐름을 설명하기 위한 것으로, 논문 원문에 언급되지 않은 내용은 일반적 지식에 기반하며, 해당 논문들을 직접 참조하지 않은 부분에서는 구체적 수치를 제시하지 않겠습니다.

#### (a) SHOT (ICML 2020) - "Do We Really Need to Access the Source Data?"

- **핵심**: 소스 데이터 없이 타겟 도메인만으로 적응
- **CoSCA와의 차이**: CoSCA는 소스 데이터를 활용한 지도 신호를 사용하지만, SHOT은 소스 모델을 고정하고 타겟 도메인에서만 미세 조정
- **시사점**: 데이터 프라이버시 관점에서 소스 없는 적응이 중요한 연구 방향으로 부상

#### (b) NRC (NeurIPS 2021) - "Exploiting the Intrinsic Neighborhood Structure"

- **핵심**: 타겟 도메인 내의 이웃 관계를 활용한 구조적 정렬
- **CoSCA와의 관계**: CoSCA의 타겟-타겟 대조 손실($\mathcal{L}_{\text{contras}}^{\mathcal{T} \leftrightarrow \mathcal{T}}$)과 개념적으로 유사하나, 더 정교한 그래프 기반 이웃 구조를 활용

#### (c) CDTrans (ICLR 2022) - "Cross-Domain Transformer"

- **핵심**: Transformer 기반 교차 도메인 주의 메커니즘
- **CoSCA와의 차이**: CoSCA가 CNN 기반인 반면, CDTrans는 Vision Transformer를 활용하여 더 풍부한 전역 문맥 정보를 활용

| 비교 축 | CoSCA (2020) | 최신 연구 동향 (2021~) |
|--------|-------------|----------------------|
| 정렬 수준 | 글로벌 + 클래스 조건부 | 클래스, 인스턴스, 패치 수준으로 세분화 |
| 대조 학습 | Siamese 기반 | MoCo, SimCLR 스타일로 발전 |
| 아키텍처 | CNN | Transformer(ViT) 기반으로 이동 |
| 소스 데이터 | 필요 | 소스 없는 적응(SFDA)으로 확장 |
| 의사 레이블 | 단순 최대 확률 | 신뢰도 가중, 클래스 밸런스 고려 |

---

### 4.3 앞으로 연구 시 고려할 점

#### (a) 의사 레이블 품질 향상

현재 CoSCA는 두 분류기의 최대 사후 확률로 의사 레이블을 결정하는데, 이는 클래스 불균형이나 학습 초기의 불안정성에 취약하다.

$$\tilde{y}_j^t = \arg\max_{k} \{p(F_1 = k|\mathbf{x}) + p(F_2 = k|\mathbf{x})\}$$

→ **신뢰도 임계값(confidence threshold)**, **클래스 밸런스 보정**, **앙상블 기반 의사 레이블** 등의 개선이 필요하다.

#### (b) Transformer 백본과의 통합

ViT(Vision Transformer) 기반 백본을 활용할 경우, 멀티헤드 어텐션을 통해 더 풍부한 클래스 조건부 특징을 추출할 수 있으며, CoSCA의 대조 손실과 시너지 효과가 기대된다.

#### (c) 소스 없는 도메인 적응(Source-Free DA)으로의 확장

데이터 프라이버시 규정(GDPR 등)으로 인해 소스 데이터 접근이 불가능한 경우가 늘어나고 있다. CoSCA의 프레임워크를 소스 모델만 활용하는 방식으로 재설계할 필요가 있다.

#### (d) 이론적 일반화 경계 분석

저자들도 언급했듯이, 대조 학습이 도메인 적응의 일반화 오차 경계에 미치는 영향에 대한 이론적 분석이 부족하다. Ben-David 등의 $\mathcal{H}\Delta\mathcal{H}$-divergence 프레임워크를 활용하여 다음과 같은 분석이 필요하다:

$$\epsilon_T(h) \leq \epsilon_S(h) + d_{\mathcal{H}\Delta\mathcal{H}}(\mathcal{D}_S, \mathcal{D}_T) + \lambda$$

여기서 대조 손실이 $d_{\mathcal{H}\Delta\mathcal{H}}$ 항을 어떻게 줄이는지 이론적으로 규명해야 한다.

#### (e) 열린 집합(Open-Set) 및 부분 집합(Partial) 도메인 적응

CoSCA는 소스와 타겟의 클래스 집합이 동일하다고 가정하는데, 실제 환경에서는 타겟 도메인에 소스 도메인에 없는 클래스가 존재할 수 있다. 이러한 **개방형 도메인 적응**으로의 확장이 필요하다.

#### (f) 멀티소스 도메인 적응으로의 확장

현재는 단일 소스→단일 타겟 구조인데, 여러 소스 도메인을 동시에 활용하는 경우 클래스 조건부 대조 손실을 어떻게 확장할지 연구가 필요하다.

---

## 참고 자료

**주 참고 자료:**
- **Dai, S., Cheng, Y., Zhang, Y., Gan, Z., Liu, J., & Carin, L. (2020). "Contrastively Smoothed Class Alignment for Unsupervised Domain Adaptation." arXiv:1909.05288v4 [cs.LG]** (제공된 PDF 원문)

**논문 내 인용된 관련 연구 (연구 맥락 이해에 활용):**
- Saito, K., et al. (2018). "Maximum Classifier Discrepancy for Unsupervised Domain Adaptation." CVPR. [논문 내 참고문헌 15]
- Kang, G., et al. (2019). "Contrastive Adaptation Network for Unsupervised Domain Adaptation." CVPR. [논문 내 참고문헌 31]
- Kumar, A., et al. (2018). "Co-regularized Alignment for Unsupervised Domain Adaptation." NeurIPS. [논문 내 참고문헌 17]
- Shu, R., et al. (2018). "A DIRT-T Approach to Unsupervised Domain Adaptation." ICLR. [논문 내 참고문헌 14]
- Ganin, Y., & Lempitsky, V. (2015). "Unsupervised Domain Adaptation by Backpropagation." ICML. [논문 내 참고문헌 13]
- Long, M., et al. (2015). "Learning Transferable Features with Deep Adaptation Networks." ICML. [논문 내 참고문헌 21]
- Hadsell, R., Chopra, S., & LeCun, Y. (2006). "Dimensionality Reduction by Learning an Invariant Mapping." CVPR. [논문 내 참고문헌 27]
- French, G., et al. (2018). "Self-Ensembling for Domain Adaptation." ICLR. [논문 내 참고문헌 38]
