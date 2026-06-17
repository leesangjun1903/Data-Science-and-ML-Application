# UNICON: Combating Label Noise Through Uniform Selection and Contrastive Learning 

---

## 1. 핵심 주장 및 주요 기여 요약

### 1.1 핵심 주장

UNICON의 핵심 주장은 다음 두 가지입니다:

1. **기존 샘플 선택 방법의 근본적 문제**: 기존의 선택 기반 방법들(DivideMix 등)은 **쉬운 클래스(easy classes)의 샘플을 불균형적으로 더 많이 선택**하고 어려운 클래스(hard classes)의 샘플을 배제한다. 이로 인해 clean set에서 클래스 불균형이 발생하고, 고노이즈 환경에서 성능이 급격히 저하된다.

2. **해결책**: Jensen-Shannon Divergence(JSD) 기반의 **균등 선택(Uniform Selection)** 과 **대조 학습(Contrastive Learning)** 을 결합하면, 고노이즈 환경에서도 강인한 성능을 달성할 수 있다.

### 1.2 주요 기여 (3가지)

| 기여 | 내용 |
|------|------|
| **균등 선택 메커니즘** | 각 클래스에서 동일한 비율 $R$만큼 clean 샘플을 선택하는 class-balanced 방법 제안 |
| **대조 학습 통합** | 레이블에 의존하지 않는 비지도 feature learning으로 노이즈 암기(memorization) 위험 최소화 |
| **성능 향상** | CIFAR100 90% 노이즈에서 기존 SOTA 대비 **11.4% 향상** |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

#### 문제 1: 클래스 불균형 샘플 선택

기존 방법(예: DivideMix)은 손실 기반 기준(small-loss criterion)을 전체 데이터셋에 전역적으로 적용합니다. 이는 다음과 같은 문제를 야기합니다:

- **쉬운 클래스**: 낮은 손실값 → 과도하게 선택됨
- **어려운 클래스**: 높은 손실값 → 배제됨 (실제로는 clean 레이블임에도)

논문의 실험에서, DivideMix는 CIFAR10 90% 노이즈 환경에서 class-1에서 1228개 샘플을 선택한 반면 class-2에서는 단 10개만 선택했습니다.

#### 문제 2: 노이즈 레이블 암기(Memorization)

Deep Neural Networks는 충분한 훈련 시간이 주어지면 임의의 레이블도 암기할 수 있습니다 [Arpit et al., 2017]. SSL 기반 훈련만으로는 이 암기 위험을 완전히 제거하지 못합니다.

---

### 2.2 제안하는 방법 (수식 포함)

#### Step 1: 초기 손실 함수

전체 훈련 데이터 $\mathbb{D} = \{(\mathbf{x}\_i, \mathbf{y}\_i)\}_{i=1}^{N}$에 대한 cross-entropy 손실:

$$\mathcal{L}_{CE} = -\frac{1}{N}\sum_{i=1}^{N}\mathbf{y}_i^T \log \hat{\mathbf{y}}_i \tag{1}$$

여기서 $\hat{\mathbf{y}}_i = \text{softmax}(\mathbf{h}(\mathbf{f}(\mathbf{x}_i; \theta); \phi))$

---

#### Step 2: Jensen-Shannon Divergence 기반 분리 (핵심)

두 네트워크 $(\theta^{(1)}, \phi^{(1)}, \psi^{(1)})$, $(\theta^{(2)}, \phi^{(2)}, \psi^{(2)})$의 예측 평균:

$$\mathbf{p}_i = \frac{\hat{\mathbf{y}}_i^{(1)} + \hat{\mathbf{y}}_i^{(2)}}{2}$$

각 샘플 $x_i$에 대한 ground-truth 레이블 $\mathbf{y}_i$와 예측 확률 $\mathbf{p}_i$ 사이의 불일치를 JSD로 측정:

$$d_i = \text{JSD}(\mathbf{y}_i, \mathbf{p}_i) = \frac{1}{2}\text{KLD}\left(\mathbf{y}_i \Big\| \frac{\mathbf{y}_i + \mathbf{p}_i}{2}\right) + \frac{1}{2}\text{KLD}\left(\mathbf{p}_i \Big\| \frac{\mathbf{y}_i + \mathbf{p}_i}{2}\right) \tag{2}$$

> **JSD의 장점**: 값의 범위가 $[0, 1]$로 고정, 정규화 불필요, 대칭성(symmetric), GMM 같은 확률 모델링 불필요

---

#### Step 3: 자동 필터율 결정

컷오프 발산값 $d_{cutoff}$를 다음과 같이 자동 계산:

$$d_{cutoff} = \begin{cases} d_{avg} - \dfrac{d_{avg} - d_{min}}{\tau}, & \text{if } d_{avg} \geq d_{\mu} \\ d_{avg}, & \text{otherwise} \end{cases} \tag{3}$$

- $d_{avg}$: 모든 샘플의 JSD 평균
- $d_{min}$: 최솟값
- $\tau$: 필터 계수(filter coefficient)
- $d_{\mu}$: 조정 임계값

필터율 $R$은 $d_i < d_{cutoff}$인 샘플의 비율:

$$R = \frac{|\{d_i < d_{cutoff}\}|}{N}$$

---

#### Step 4: 균등 선택 (핵심 혁신)

각 클래스 $j$에 대해 JSD 값 기준으로 하위 $R$ 비율의 샘플만 선택:

```math
\mathbb{D}_{clean}^{(j)} = \left\{(\mathbf{x}_t^{(j)}, \mathbf{y}_t^{(j)}) : \forall\, d_t^{(j)} \in \mathbf{d}_{filtered}^{(j)}\right\}
```

$$\mathbb{D}_{clean} = \bigcup_{j=1}^{C} \mathbb{D}_{clean}^{(j)}, \quad \mathbb{D}_{noisy} = \mathbb{D} \setminus \mathbb{D}_{clean}$$

이를 통해 각 클래스에서 $NR/C$개의 샘플이 균등하게 선택됩니다.

---

#### Step 5: 대조 학습 손실 (SSL 보완)

$\mathbb{D}\_{noisy}$ 샘플의 두 augmented view $(\mathbf{x}\_{i,1}, \mathbf{x}_{i,2})$에서 projection head $\mathbf{g}(;\psi)$를 통해 feature 추출:

$$\mathbf{z}_i = \mathbf{g}(\mathbf{f}(\mathbf{x}_{i,1}; \theta); \psi), \quad \mathbf{z}_j = \mathbf{g}(\mathbf{f}(\mathbf{x}_{i,2}; \theta); \psi)$$

NT-Xent (Normalized Temperature-scaled Cross Entropy) 손실:

$$\ell_{i,j} = -\log\frac{\exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_j)/\kappa)}{\sum_{b=1}^{2B}\mathbb{1}_{b \neq i}\exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_b)/\kappa)} \tag{4}$$

$$\mathcal{L}_\mathcal{C} = \frac{1}{2B}\sum_{b=1}^{2B}[\ell_{2b-1, 2b} + \ell_{2b, 2b-1}] \tag{5}$$

여기서 $\text{sim}(\mathbf{z}_i, \mathbf{z}_j)$는 코사인 유사도, $\kappa$는 온도 상수, $B$는 미니배치 크기

---

#### Step 6: 최종 손실 함수

$$\mathcal{L}_{tot} = \mathcal{L}_{semi} + \lambda_{\mathcal{C}} \mathcal{L}_{\mathcal{C}} \tag{6}$$

- $\mathcal{L}_{semi}$: FixMatch 기반 semi-supervised 손실 (MixUp 적용)
- $\lambda_{\mathcal{C}}$: contrastive 손실 가중치

---

### 2.3 모델 구조

```
훈련 세트 D
    ↓
[Warmup: CE Loss 기반 사전 훈련 (10~30 epochs)]
    ↓
┌─────────────────────────────────────┐
│         UNICON 반복 루프            │
│                                     │
│  1. 두 네트워크 앙상블 예측         │
│     p_i = (ŷ^(1) + ŷ^(2)) / 2     │
│                                     │
│  2. JSD 계산 → d_cutoff 자동 결정  │
│                                     │
│  3. 클래스별 균등 선택              │
│     → D_clean (NR 샘플)            │
│     → D_noisy (N(1-R) 샘플)        │
│                                     │
│  4. SSL 훈련                        │
│     - D_clean: 지도 학습            │
│     - D_noisy: pseudo-label 생성   │
│     - MixUp augmentation           │
│     - Contrastive Learning         │
│       (projection head 사용)        │
└─────────────────────────────────────┘
    ↓
최종 분류기
```

**아키텍처 구성 요소:**
- Feature Extractor: PreAct ResNet18 (CIFAR, TinyImageNet), ResNet50 (Clothing1M, WebVision)
- Classification Layer: $\mathbf{h}(;\phi)$
- Projection Head: $\mathbf{g}(;\psi)$ — 128차원 임베딩 벡터 출력

---

### 2.4 성능 향상

#### 대칭 노이즈(Symmetric Noise) 성능:

| 방법 | CIFAR10 (90%) | CIFAR100 (90%) |
|------|:---:|:---:|
| CE | 42.7 | 10.1 |
| DivideMix | 76.0 | 31.5 |
| ELR | 78.7 | 33.4 |
| **UNICON** | **90.8** | **44.8** |

#### 극단적 노이즈 환경 (CIFAR10):

| 노이즈율 | DivideMix | UNICON |
|---------|:---:|:---:|
| 90% | 76.08 | **90.81** |
| 92% | 57.62 | **87.61** |
| 95% | 51.28 | **80.82** |
| 98% | 17.18 | **50.63** |

#### 실세계 데이터셋:
- **Clothing1M**: 74.98% (ELR 74.81% 대비 +0.17%)
- **WebVision Top-5**: 93.44% (SOTA)
- **Tiny-ImageNet**: 모든 노이즈율에서 약 +1% 향상

---

### 2.5 한계점

논문에서 명시적으로 인정하는 한계:

1. **극단적 클래스 불균형 데이터셋**: Class-balance prior는 데이터셋 자체가 심각하게 불균형할 경우 오히려 제약이 될 수 있습니다. (단, 클래스 분포를 안다면 prior를 조정 가능하나, 이를 사전에 아는 것 자체가 제약)

2. **낮은 노이즈율에서의 한계**: 노이즈율이 낮을 경우(예: 20%), DivideMix가 UNICON보다 CIFAR10에서 약간 우수한 성능을 보입니다. 이는 $|\mathbb{D}\_{noisy}| < |\mathbb{D}_{clean}|$ 상황에서 대조 학습의 효과가 감소하기 때문으로 해석됩니다.

3. **Computational Overhead**: 두 네트워크를 동시에 훈련하고 대조 학습까지 수행하므로, 단일 네트워크 대비 계산 비용이 높습니다.

4. **하이퍼파라미터 완전 제거 불가**: $\tau$, $d_\mu$, $\lambda_\mathcal{C}$, $\kappa$ 등의 하이퍼파라미터가 여전히 존재합니다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 균등 선택이 일반화에 미치는 영향

**클래스 균형 → 가짜 레이블(pseudo-label) 품질 향상 → 일반화 성능 향상** 의 인과관계가 핵심입니다.

논문의 Figure 2(c)에서 확인된 pseudo-label recall 비교:
- DivideMix: 일부 클래스에서 매우 낮은 recall
- UNICON: 모든 클래스에서 균등하고 높은 recall

이는 다음 메커니즘으로 일반화에 기여합니다:

$$\text{균등 선택} \Rightarrow \text{True Positive 균형} \Rightarrow \text{고품질 pseudo-label} \Rightarrow \text{SSL 성능 향상} \Rightarrow \text{일반화}$$

### 3.2 대조 학습이 일반화에 미치는 영향

**레이블-불가지론적(label-agnostic) feature learning**이 핵심입니다:

- 대조 학습은 레이블 없이 feature representation을 학습하므로 노이즈 레이블의 영향을 직접 받지 않음
- Figure 4에서 CL 적용 시 ROC-AUC가 지속적으로 향상됨을 확인
- Ablation study (Table 7)에서 CL 제거 시 CIFAR10/100 90% 노이즈에서 각각 3.53%, 2.99% 성능 하락

$$\text{CL 기반 feature} \Rightarrow \text{분류 경계 명확화} \Rightarrow \text{분포 외(OOD) 데이터에 강인} \Rightarrow \text{일반화}$$

### 3.3 다양한 데이터셋에서의 일반화 증거

| 환경 | 성능 |
|------|------|
| 합성 노이즈 (CIFAR) | 고노이즈에서 SOTA 대비 대폭 향상 |
| 대규모 실세계 노이즈 (Clothing1M) | 안정적 성능 유지 |
| 웹 크롤링 노이즈 (WebVision) | Top-5 SOTA 달성 |
| 다양한 해상도 (TinyImageNet) | 일관된 성능 향상 |

이는 UNICON의 접근법이 특정 데이터셋에 과적합되지 않음을 시사합니다.

### 3.4 암기 방지 메커니즘 (일반화의 핵심)

Figure 6에서 훈련 정확도 분석:
- 표준 CE 훈련: 훈련 정확도가 ~100%에 도달 (완전 암기)
- UNICON: 훈련 정확도가 낮게 유지됨 (암기 억제)

낮은 훈련 정확도 유지는 모델이 데이터의 **실질적인 패턴**을 학습함을 의미하며, 이는 직접적으로 테스트 데이터에 대한 일반화 성능 향상으로 이어집니다.

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

### 4.1 주요 관련 연구 비교표

| 논문 | 연도 | 핵심 방법 | 고노이즈 강인성 | 하이퍼파라미터 의존성 |
|------|------|-----------|:---:|:---:|
| **DivideMix** [Li et al.] | 2020 | GMM + SSL (MixMatch) | 중간 | 높음 |
| **ELR** [Liu et al.] | 2020 | Early Learning Regularization | 중간 | 중간 |
| **MOIT** [Ortego et al.] | 2021 | Multi-objective Interpolation + CL | 낮음 | 높음 |
| **Jo-SRC** [Yao et al.] | 2021 | JSD + ID/OOD 분리 | 낮음 | 높음 (manual threshold) |
| **AugDesc** [Nishi et al.] | 2021 | 증강 전략 강화 (DivideMix 기반) | 중간 | 높음 |
| **NCT** [Sarfraz et al.] | 2020 | Collaborative Learning | 낮음 | 중간 |
| **UNICON** [Karim et al.] | 2022 | JSD 균등 선택 + CL | **높음** | **낮음** |

### 4.2 UNICON vs DivideMix (가장 중요한 비교)

DivideMix는 UNICON의 직접적 기준선(baseline)입니다:

**DivideMix의 선택 방법:**
- 전체 손실값에 GMM을 fitting하여 클린/노이즈 분리
- $p(\text{clean}|x_i) = \frac{\pi_1 \mathcal{N}(l_i|\mu_1, \sigma_1^2)}{\pi_1 \mathcal{N}(l_i|\mu_1, \sigma_1^2) + \pi_2 \mathcal{N}(l_i|\mu_2, \sigma_2^2)}$
- **문제**: GMM은 정규화된 손실값에 의존, 클래스 간 불균형 야기

**UNICON의 개선:**
- JSD 기반 자동 임계값 결정 → 정규화 불필요
- 클래스별 균등 선택 → 불균형 해소

### 4.3 UNICON vs Jo-SRC (JSD 사용 공통점)

Jo-SRC도 JSD를 사용하지만 핵심적 차이가 있습니다:
- Jo-SRC: 에폭마다 수동으로 임계값 조정 필요
- UNICON: $d_{cutoff}$를 네트워크 예측 점수에서 자동 계산 → **하이퍼파라미터 독립적**

### 4.4 UNICON vs MOIT

MOIT도 대조 학습을 활용하지만:
- MOIT: 고노이즈(90%) 환경에서 성능이 급격히 하락 (CIFAR100 90%: 24.5%)
- UNICON: 동일 환경에서 44.8% 달성 → **20.3%p 차이**

MOIT의 문제는 supervised contrastive learning을 사용하여 노이즈 레이블에 영향을 받는 반면, UNICON은 **unsupervised** contrastive learning으로 레이블 의존성을 완전히 제거합니다.

---

## 5. 앞으로의 연구에 미치는 영향과 고려할 점

### 5.1 연구에 미치는 영향

#### (1) 방법론적 영향
- **클래스 균형 선택의 중요성 부각**: 이전 연구들이 간과했던 선택 편향(selection bias) 문제를 명시적으로 제기하고 실증적으로 증명함으로써, 향후 샘플 선택 방법 설계 시 클래스 균형이 필수적으로 고려되어야 함을 시사
- **JSD의 활용 확장**: GMM 등 복잡한 확률 모델 없이도 JSD만으로 충분히 효과적인 분리가 가능함을 보여줌
- **Unsupervised CL의 전략적 활용**: 레이블 노이즈 문제에서 비지도 대조 학습이 노이즈 암기 억제에 효과적임을 실증

#### (2) 실용적 영향
- 다양한 노이즈 유형/비율에 걸쳐 **단일 하이퍼파라미터 세트**로 적용 가능한 방향성 제시
- 웹 스크래핑 데이터, 크라우드소싱 레이블 등 실세계 노이즈 레이블 학습에 직접 적용 가능한 실용적 프레임워크

#### (3) 학문적 영향
- **노이즈 레이블 학습 + 대조 학습** 의 결합 연구 방향을 강화
- 반지도 학습(SSL)과 노이즈 레이블 학습의 통합 프레임워크 설계에 기여

---

### 5.2 향후 연구 시 고려할 점

#### (1) 자연적 클래스 불균형 데이터 처리
UNICON의 class-balance prior는 인위적으로 균등 분포를 가정합니다. 실세계 데이터는 **long-tailed distribution**을 따르는 경우가 많으므로:
- 데이터셋의 실제 클래스 분포를 반영한 **적응적 prior(adaptive prior)** 설계 필요
- Long-tail 노이즈 레이블 학습과의 결합 연구 필요

$$R_j = f(\text{class distribution}_j) \quad \text{(클래스별 적응적 필터율)}$$

#### (2) 저노이즈 환경 최적화
UNICON은 낮은 노이즈율(20%)에서 DivideMix보다 성능이 약간 낮습니다. $|\mathbb{D}\_{noisy}| < |\mathbb{D}_{clean}|$ 상황에서의 CL 효과 저하를 극복하기 위해:
- 노이즈율 추정 기반 적응적 CL 가중치 $\lambda_\mathcal{C}$ 조정
- Clean set에도 CL을 적용하는 방향 탐색

#### (3) Transformer 아키텍처와의 통합
UNICON은 ResNet 기반으로 실험되었으나, Vision Transformer(ViT), Swin Transformer 등 최신 아키텍처와의 결합:
- Attention 메커니즘이 노이즈 레이블에 어떻게 반응하는지 분석 필요
- Pre-trained Transformer + UNICON의 fine-tuning 전략 탐색

#### (4) Open-set 노이즈 환경 확장
현재 UNICON은 **closed-set 노이즈**(훈련 클래스 내의 레이블 오류)를 가정합니다. 실세계에서는:
- **Open-set 노이즈**: 훈련 클래스에 속하지 않는 OOD 샘플이 포함
- Jo-SRC의 ID/OOD 분리 개념과 UNICON의 균등 선택을 결합한 연구 필요

#### (5) 계산 효율성 개선
두 네트워크 앙상블 + CL의 계산 비용:
- Knowledge Distillation을 이용한 단일 네트워크로의 압축
- Mean Teacher 프레임워크와의 통합으로 효율성 향상 가능

#### (6) 다른 도메인으로의 적용 가능성
- **의료 영상**: 노이즈 레이블이 빈번한 의료 데이터 (피부 병변 분류 등)
- **자연어 처리**: 텍스트 분류의 노이즈 레이블 처리
- **멀티모달**: 이미지-텍스트 쌍의 노이즈 레이블 학습

#### (7) 이론적 보장 마련
현재 UNICON은 주로 경험적(empirical) 결과에 의존합니다:
- 균등 선택이 일반화 오차를 줄인다는 **이론적 보장(generalization bound)** 도출 필요
- PAC-learning 프레임워크에서의 분석

---

## 참고자료

**주 논문:**
- Karim, N., Rizve, M. N., Rahnavard, N., Mian, A., & Shah, M. (2022). **UNICON: Combating Label Noise Through Uniform Selection and Contrastive Learning**. arXiv:2203.14542v1.

**논문 내 인용 주요 참고자료:**
- Li, J., Socher, R., & Hoi, S. C. H. (2020). **DivideMix: Learning with Noisy Labels as Semi-Supervised Learning**. ICLR 2020. [ref 25]
- Han, B., et al. (2018). **Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels**. NeurIPS 2018. [ref 12]
- Yao, Y., et al. (2021). **Jo-SRC: A Contrastive Approach for Combating Noisy Labels**. CVPR 2021. [ref 63]
- Ortego, D., et al. (2021). **Multi-Objective Interpolation Training for Robustness to Label Noise (MOIT)**. CVPR 2021. [ref 39]
- Liu, S., et al. (2020). **Early-Learning Regularization Prevents Memorization of Noisy Labels (ELR)**. NeurIPS 2020. [ref 30]
- Chen, T., et al. (2020). **A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)**. ICML 2020. [ref 6]
- Sohn, K., et al. (2020). **FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence**. NeurIPS 2020. [ref 47]
- Zhang, H., et al. (2018). **MixUp: Beyond Empirical Risk Minimization**. ICLR 2018. [ref 67/68]
- Arpit, D., et al. (2017). **A Closer Look at Memorization in Deep Networks**. ICML 2017. [ref 2]
- Nishi, K., et al. (2021). **Augmentation Strategies for Learning with Noisy Labels**. CVPR 2021. [ref 35]
- Sarfraz, F., et al. (2020). **Noisy Concurrent Training for Efficient Learning Under Label Noise**. [ref 44]

**코드 공개:**
- GitHub: https://github.com/nazmul-karim170/UNICON-Noisy-Label
