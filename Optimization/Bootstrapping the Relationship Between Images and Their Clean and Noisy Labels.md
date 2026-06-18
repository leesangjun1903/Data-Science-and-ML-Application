# Bootstrapping the Relationship Between Images and Their Clean and Noisy Labels

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

이 논문의 핵심 주장은 다음과 같습니다:

> **"클린 레이블 서브셋 없이도, 이미지-노이즈 레이블-클린 레이블 간의 관계를 부트스트래핑 방식으로 학습할 수 있으며, 이를 통해 인스턴스 의존적 레이블 노이즈(IDN)를 효과적으로 처리할 수 있다."**

기존 SOTA 방법들은 노이즈 레이블을 단순히 버리고 클린 레이블만을 추정하는 방식을 사용했지만, 이 논문은 **노이즈 레이블 자체도 유용한 신호**임을 주장합니다. 특히 IDN 환경에서는 이미지 특징과 노이즈 레이블의 관계가 재레이블링(relabeling)의 정확도를 높이는 데 중요합니다.

### 주요 기여

| 기여 항목 | 설명 |
|----------|------|
| **3단계 훈련 알고리즘** | 클린 서브셋 없이 이미지-노이즈-클린 레이블 관계 학습 |
| **노이즈 전이 샘플 밸런싱** | 클래스 기반이 아닌 노이즈 전이 기반 샘플 균형 전략 |
| **단일 모델 아키텍처** | DivideMix의 2-모델 구조 대비 단순한 단일 모델로 SOTA 달성 |
| **Label Dropping 전략** | 모델 증류(distillation) 없이 노이즈 레이블 유무 모두에서 예측 가능 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

딥러닝 모델은 노이즈 레이블 데이터로 학습 시 과적합 문제가 발생합니다. 기존 방법들의 한계는 크게 세 가지입니다:

1. **IIN(인스턴스 독립 노이즈) 위주**: 대칭/비대칭 노이즈에만 집중, 실제 데이터의 IDN에 취약
2. **클린 서브셋 필요**: 일부 방법들은 클린 레이블과 노이즈 레이블이 모두 존재하는 서브셋을 요구 → 수집 비용 高
3. **노이즈 레이블 무시**: 재레이블링 후 노이즈 레이블을 버려 IDN 모델링에 유용한 정보 손실

**IDN(Instance-Dependent Noise)**의 수학적 특성:

$$P(\tilde{y} \mid y, \mathbf{x}) \neq P(\tilde{y} \mid y)$$

즉, 레이블 오류 확률이 이미지 $\mathbf{x}$에도 의존하며, 이를 무시하면 정확한 재레이블링이 어렵습니다.

---

### 2.2 제안하는 방법 (수식 포함)

알고리즘은 3단계로 구성됩니다.

#### **Stage 1: Bootstrapping (부트스트래핑)**

훈련 셋: $\mathcal{D} = \{(\mathbf{x}\_i, \tilde{\mathbf{y}}\_i)\}_{i=1}^{|\mathcal{D}|}$

여기서 $\mathbf{x} \in \mathcal{X} \subset \mathbb{R}^{H \times W \times R}$, $\tilde{\mathbf{y}} \in \mathcal{Y} \subset \{0,1\}^{|\mathcal{Y}|}$ (원-핫 노이즈 레이블)

**수식 (1): Early-stopping 분류기 훈련**

$$\theta^* = \arg\min_{\theta} \frac{1}{|\mathcal{D}|} \sum_{(\mathbf{x}_i, \tilde{\mathbf{y}}_i) \in \mathcal{D}} \mathbb{E}_{a(\cdot) \sim \mathcal{A}_S} \left[ \ell_{CE}\left(\tilde{\mathbf{y}}_i, f_{\theta}(a(\mathbf{x}_i), \mathbf{0}_{|\mathcal{Y}|})\right) \right]$$

- $a(\cdot)$: 강한 데이터 증강 함수 (from $\mathcal{A}_S$)
- $\ell_{CE}$: 크로스 엔트로피 손실
- $\mathbf{0}_{|\mathcal{Y}|}$: 'null' 레이블 벡터 (노이즈 레이블 없음을 나타냄)

**수식 (2): 테스트 타임 약한 증강을 통한 예측 분포 생성**

$$\hat{\mathbf{y}}_i = \mathbb{E}_{a(\cdot) \sim \mathcal{A}_W} \left[ f_{\theta^*}(a(\mathbf{x}_i), \mathbf{0}_{|\mathcal{Y}|}) \right]$$

- 드롭아웃 활성화 상태에서 다수의 약한 증강 결과를 평균
- 신뢰도: $\max_{c \in \mathcal{Y}} \hat{y}_i(c)$

**노이즈 전이 샘플 밸런싱:**

노이즈 전이 행렬 $\mathbf{T}$를 추정합니다:

$$T_{ij} = \hat{P}(\tilde{y} = i, y = j)$$

각 노이즈 전이에서 $K \times |\mathcal{Y}| \times T_{ij}$개의 가장 신뢰도 높은 샘플을 선택하거나, $\max_{c \in \mathcal{Y}} \hat{y}_i(c) > \tau$인 샘플을 선택합니다.

- **클린 셋**: $\mathcal{C} = \{(\mathbf{x}_i, \tilde{\mathbf{y}}_i, \hat{\mathbf{y}}_i) \mid (\mathbf{x}_i, \tilde{\mathbf{y}}_i) \in \mathcal{D}\}$
- **노이즈 셋**: $\mathcal{U} = \{(\mathbf{x}_i, \tilde{\mathbf{y}}_i) \mid (\mathbf{x}_i, \tilde{\mathbf{y}}_i, \hat{\mathbf{y}}_i) \notin \mathcal{C},\ (\mathbf{x}_i, \tilde{\mathbf{y}}_i) \in \mathcal{D}\}$

---

#### **Stage 2: Semi-Supervised Learning (반지도 학습, SSL)**

FixMatch 기반 SSL을 적용합니다.

**수식 (3): SSL 최적화 목표**

```math
\theta^* = \arg\min_{\theta \in \Theta} \underbrace{\frac{1}{|\mathcal{C}|} \sum_{(\mathbf{x}_i, \tilde{\mathbf{x}}_i, \hat{\mathbf{y}}_i) \in \mathcal{C}} \mathbb{E}_{a(\cdot) \in \mathcal{A}_W}\left[\ell_{CE}\left(\hat{\mathbf{y}}_i, f_{\theta}(a(\mathbf{x}_i), \iota_{50\%}(\tilde{\mathbf{y}}_i))\right)\right]}_{\text{Supervised Loss (Clean Set)}} + \underbrace{\frac{1}{|\mathcal{U}|} \sum_{(\mathbf{x}_i, \tilde{\mathbf{x}}_i) \in \mathcal{U}} \mathbb{1}(\max \bar{\mathbf{y}}_i > \kappa) \mathbb{E}_{a(\cdot) \in \mathcal{A}_S}\left[\ell_{CE}\left(\lceil \bar{\mathbf{y}}_i \rceil, f_{\theta}(a(\mathbf{x}_i), \iota_{50\%}(\tilde{\mathbf{y}}_i))\right)\right]}_{\text{Unsupervised Loss (Noisy Set)}}
```

각 기호의 의미:
- $\bar{\mathbf{y}}\_i = \mathbb{E}_{a(\cdot) \sim \mathcal{A}_W}[f(a(\mathbf{x}_i), \tilde{\mathbf{y}}_i)]$: 약한 증강 앙상블로 생성된 소프트 레이블
- $\iota_{50}(\tilde{\mathbf{y}}\_i)$: 50% 확률로 $\tilde{\mathbf{y}}\_i$ 또는 $\mathbf{0}_{|\mathcal{Y}|}$ 반환 (Label Dropping)
- $\lceil \bar{\mathbf{y}}_i \rceil$: 최대 확률 클래스에 1, 나머지 0인 이진 벡터 (pseudo-label)
- $\kappa$: pseudo-label 신뢰도 임계값 (= 0.95)
- $\mathbb{1}(\cdot)$: 지시 함수

SSL 이후 전체 훈련 셋 재레이블링:

**수식 (4):**

$$\bar{\mathcal{D}} = \{(\mathbf{x}_i, \tilde{\mathbf{y}}_i, \bar{\mathbf{y}}_i) \mid (\mathbf{x}_i, \tilde{\mathbf{y}}_i) \in \mathcal{D}\}$$

---

#### **Stage 3: Final Training (최종 훈련)**

$\bar{\mathcal{D}}$를 이용하여 MixUp + 강한 증강으로 최종 분류기를 훈련합니다:

$$\text{MixUp:} \quad \tilde{\mathbf{x}} = \lambda \mathbf{x}_i + (1-\lambda)\mathbf{x}_j, \quad \tilde{\mathbf{y}} = \lambda \bar{\mathbf{y}}_i + (1-\lambda)\bar{\mathbf{y}}_j, \quad \lambda \sim \text{Beta}(\alpha, \alpha)$$

MixUp 후 50% 확률로 노이즈 레이블을 null 벡터로 대체합니다.

---

### 2.3 모델 구조

```
[Normal Model]                    [Modified Model]
  x → Feature Extractor             x → Feature Extractor → Linear (→ 128-dim)
      → Linear                      ỹ              → Linear (→ 128-dim)
      → ŷ (softmax)                 Concatenate → Linear → ReLU → Linear
                                    → ŷ (softmax)
```

- **'Normal' 모델**: $f_\theta: \mathcal{X} \to \Delta^{|\mathcal{Y}|-1}$ (이미지 → 클린 레이블 예측)
- **'Modified' 모델**: $f_\theta: \mathcal{X} \times \mathcal{Y} \to \Delta^{|\mathcal{Y}|-1}$ (이미지 + 노이즈 레이블 → 클린 레이블 예측)
  - 이미지 특징과 노이즈 레이블을 각각 128차원으로 투영 후 연결(concatenation)
  - 드롭아웃($p=0.2$) + 배치 정규화 적용
- **백본**: CIFAR용 PreAct-ResNet-18, Animal10N용 VGG19, WebVision용 Inception-ResNetV2
- **자기지도 사전학습**: CIFAR/Animal10N → SimCLR, WebVision → MoCo-v2

---

### 2.4 성능 향상

#### Animal10N 결과 (표 1)

| 방법 | Top-1 정확도 |
|------|------------|
| Cross Entropy | 79.4% |
| SELFIE | 81.8% |
| PLC | 83.4% |
| NCT | 84.1% |
| **Ours (Normal Model)** | **85.84%** |
| **Ours (Modified Model)** | **88.48%** |
| **+ Test-Time Aug.** | **89.38%** |

Modified Model이 Normal Model 대비 **+2.64%** 향상 → IDN 환경에서 노이즈 레이블-클린 레이블 관계 학습의 유효성 입증

#### WebVision 결과 (표 2)

| 방법 | WebVision Top-1 | ILSVRC2012 Top-1 |
|------|----------------|-----------------|
| ELR+ | 77.78% | 70.29% |
| NGC | 79.16% | 74.44% |
| **Ours (Modified + TTA)** | **83.16%** | **79.64%** |

#### CIFAR-10/100 PMD Noise 35% (표 3)

| 방법 | CIFAR-10 Type-I | CIFAR-100 Type-I |
|------|----------------|-----------------|
| PLC | 82.80% | 60.01% |
| **Ours (Modified + TTA)** | **94.39%** | **70.13%** |

#### 각 훈련 단계별 정확도 향상 (CIFAR-10 Asym. 40%)

| 단계 | 정확도 |
|------|------|
| After Bootstrapping | 91.41% |
| After SSL | 94.98% |
| After Final Training | **95.85%** |

---

### 2.5 한계점

1. **높은 노이즈율에서의 클래스 플리핑 문제**: 70% PMD 노이즈 시 CIFAR-10에서 일부 클래스 쌍이 완전히 역전되어 성능 급락 (21.02% → PLC의 42.74% 미만)

2. **합성 IDN에서 Modified Model의 불안정성**: RoG 노이즈 벤치마크에서 Modified Model이 Normal Model보다 성능이 낮은 경우 존재 (학습된 노이즈 전이가 일반화되지 않는 경우 발생)

3. **CIFAR-100 대칭 노이즈에서의 약세**: 대칭 노이즈 가정에 최적화된 방법(AugDesc, ContrastToDivide)보다 성능 열위

4. **높은 계산 비용**: 총 24.3시간 (RTX 2080 기준) → DivideMix(5h), PropMix(10h) 대비 훨씬 높음

5. **하이퍼파라미터 민감성**: $K$, $\tau$, $\kappa$ 등 다수의 하이퍼파라미터 조율 필요

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화 성능 향상에 기여하는 핵심 메커니즘

#### (1) Self-Supervised Pre-training의 역할

SimCLR 사전학습은 노이즈 레이블에 과적합되지 않는 강건한 초기 특징 표현을 제공합니다. 표 10의 ablation 결과:

$$\text{에러 수: 자기지도 없음 } 369 \gg \text{자기지도 있음 } 21$$

이는 사전학습이 초기 클린 셋 선택의 정확도를 **94.3% 감소**시켜 이후 단계의 일반화에 결정적으로 기여함을 보여줍니다.

#### (2) Label Dropping을 통한 도메인 일반화

```math
\iota_{50\%}(\tilde{\mathbf{y}}_i) = \begin{cases} \tilde{\mathbf{y}}_i & \text{with probability 0.5} \\ \mathbf{0}_{|\mathcal{Y}|} & \text{with probability 0.5} \end{cases}
```

이 전략으로 모델은 두 가지 조건 모두에서 예측 가능:
- **노이즈 레이블 있음**: $f_\theta(\mathbf{x}, \tilde{\mathbf{y}}) \to$ 노이즈 정보를 이용한 고신뢰도 예측
- **노이즈 레이블 없음**: $f_\theta(\mathbf{x}, \mathbf{0}) \to$ 이미지만으로 예측 (테스트 시 일반화)

표 6에서 CIFAR-10 Asym. 40%:

$$\text{노이즈 레이블 없음: } 95.85\% \quad \text{vs} \quad \text{노이즈 레이블 있음: } 97.59\%$$

#### (3) 노이즈 전이 샘플 밸런싱에 의한 분포 커버리지

단순 클래스 기반 밸런싱 대비, 노이즈 전이 기반 밸런싱은:
- 다양한 노이즈 전이 경로($\tilde{y} = i \to y = j$)를 클린 셋에 포함
- SSL 단계에서 학습되는 조건부 분포 $P(y \mid \mathbf{x}, \tilde{y})$의 커버리지 향상
- 특히 IDN에서 인스턴스별 노이즈 패턴을 더 잘 반영

#### (4) MixUp과 강한 증강의 정규화 효과

표 9 (훈련/평가 증강 조합별 에러 수):

| 훈련 \ 평가 | 없음 | 약한 증강 | 강한 증강 |
|-----------|-----|---------|---------|
| 강한 증강 | 28 | **21** | 31 |

강한 증강 훈련 + 약한 증강 평가 조합이 최적 → Nishi et al. [35]의 결과와 일치

MixUp은 결정 경계를 부드럽게 만들고 남은 레이블 노이즈에 대한 내성을 제공:

$$\ell(\lambda \bar{\mathbf{y}}_i + (1-\lambda)\bar{\mathbf{y}}_j, f_\theta(\lambda \mathbf{x}_i + (1-\lambda)\mathbf{x}_j)) \leq \lambda \ell(\bar{\mathbf{y}}_i, f_\theta(\mathbf{x}_i)) + (1-\lambda)\ell(\bar{\mathbf{y}}_j, f_\theta(\mathbf{x}_j))$$

#### (5) 드롭아웃 기반 불확실성 추정

예측 신뢰도 평가 시 드롭아웃을 활성화하여:

$$\hat{\mathbf{y}}_i = \mathbb{E}_{a(\cdot) \sim \mathcal{A}_W, \text{dropout}} [f_{\theta^*}(a(\mathbf{x}_i), \mathbf{0}_{|\mathcal{Y}|})]$$

높은 신뢰도지만 불일치하는 예측을 가진 샘플을 페널티 부여 → 클린 셋의 순도 향상

---

## 4. 앞으로의 연구에 미치는 영향과 고려할 점

### 4.1 앞으로의 연구에 미치는 영향

#### (A) 패러다임 전환: 노이즈 레이블의 재해석

기존 연구는 노이즈 레이블을 **제거해야 할 오염원**으로 취급했지만, 이 논문은 노이즈 레이블을 **인스턴스 의존적 노이즈를 모델링하는 유용한 신호**로 재정의합니다. 이는 향후 연구에서 노이즈 레이블의 정보 이론적 가치를 분석하는 방향을 열어줍니다.

#### (B) 클린 서브셋 없는 관계 학습의 가능성 증명

[17, 21, 52]와 같은 방법들이 클린+노이즈 레이블 쌍의 서브셋을 요구했던 것과 달리, 부트스트래핑으로 이를 대체할 수 있음을 입증했습니다. 이는 **의료 영상, 법률 문서, 전문 도메인** 등 클린 레이블 수집 비용이 높은 분야에서 특히 중요한 기여입니다.

#### (C) 단일 모델로 SSL과 노이즈 레이블 학습의 통합

DivideMix 계열의 2-모델 공동 학습 패러다임에 대한 대안을 제시하여, 더 단순하고 효율적인 아키텍처 연구를 촉진합니다.

#### (D) 테스트 타임 노이즈 레이블 활용

태그된 이미지 분류, 웹 크롤링 데이터, 소셜 미디어 등 **테스트 시에도 노이즈 레이블이 존재하는 실제 시나리오**에서의 활용 방향을 제시합니다.

---

### 4.2 앞으로 연구 시 고려할 점

#### (A) 극단적 노이즈율에서의 강건성

70% PMD 노이즈 시 CIFAR-10에서 클래스 플리핑 문제가 발생합니다. 향후 연구는:
- 노이즈율 추정의 불확실성을 명시적으로 모델링
- 베이지안 접근법으로 클래스 플리핑 감지 메커니즘 도입
- 적응적 임계값 $\tau$를 노이즈율에 따라 동적으로 조정

#### (B) 계산 효율성 개선

24.3시간(RTX 2080)의 훈련 시간은 실용적 적용에 장벽이 됩니다:
- Knowledge Distillation을 통한 SimCLR 사전학습 비용 절감
- 단계별 훈련 대신 통합 end-to-end 학습 프레임워크 개발
- ViT(Vision Transformer) 기반 사전학습 모델(CLIP, DINO) 활용으로 사전학습 비용 절감

#### (C) 더 정교한 노이즈 모델링

현재 방법은 전역 노이즈 전이 행렬 $\mathbf{T}$를 추정하지만, 진정한 IDN은 샘플별로 상이합니다:

$$T_{ij}^{(k)} = P(\tilde{y}=i \mid y=j, \mathbf{x}_k) \neq T_{ij}$$

향후 연구에서는 클러스터별, 지역별 노이즈 전이 행렬 추정 또는 **신경망으로 인스턴스별 전이 확률을 직접 매개변수화**하는 방법을 고려할 수 있습니다.

#### (D) 대규모 데이터셋 및 파운데이션 모델과의 통합

- **CLIP, ALIGN** 등 대규모 멀티모달 사전학습 모델을 초기 특징 추출기로 활용
- **LLM을 이용한 레이블 품질 평가**: 텍스트 설명과 이미지의 의미론적 일치도로 노이즈 감지
- WebVision 이상 규모(1M+ 샘플)에서의 확장성 검증 필요

#### (E) 도메인 특화 적용

의료 영상, 위성 이미지, 음성 데이터 등 전문 도메인에서:
- 전문가 간 레이블 불일치(inter-annotator disagreement)를 IDN의 특수한 형태로 모델링
- 레이블 불확실성의 인식론적(epistemic) vs. 우연적(aleatoric) 분리

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 방법 | 발표 | 노이즈 유형 강점 | 클린 서브셋 필요 | 모델 구조 | 핵심 아이디어 |
|------|------|--------------|--------------|---------|------------|
| **DivideMix** [Li et al., 2020] | ICLR 2020 | IIN (sym/asym) | ✗ | 2-모델 | GMM 손실 분리 + MixMatch |
| **ELR/ELR+** [Liu et al., 2020] | NeurIPS 2020 | IIN | ✗ | 1-모델 | Early-learning 정규화 |
| **AugDesc** [Nishi et al., 2021] | CVPR 2021 | IIN | ✗ | 2-모델 | 증강 전략 최적화 |
| **ContrastToDivide** [Zheltonozhskii et al., 2022] | WACV 2022 | IIN | ✗ | 2-모델 | 자기지도 대조 학습 |
| **FINE** [Kim et al., 2021] | NeurIPS 2021 | IIN/IDN | ✗ | 1-모델 | 고유분해 기반 특징 분리 |
| **PropMix** [Cordeiro et al., 2021] | BMVC 2021 | IIN/IDN | ✗ | 2-모델 | 하드 샘플 필터링 + 비례 MixUp |
| **PLC** [Zhang et al., 2021] | ICLR 2021 | IDN | ✗ | 1-모델 | 점진적 레이블 수정 |
| **NGC** [Wu et al., 2021] | ICCV 2021 | Open-world | ✗ | 1-모델 | 그래프 기반 클러스터링 |
| **ScanMix** [Sachdeva et al., 2021] | arXiv 2021 | IDN | ✗ | 2-모델 | SCAN + DivideMix |
| **Ours (Smart & Carneiro, 2022)** | arXiv 2022 | **IDN (강점)** | **✗** | **1-모델** | **부트스트래핑 + 노이즈 전이 밸런싱** |

### 차별성 분석

```
노이즈 유형 처리 폭:
PLC     ████████░░  (IDN 특화)
DivideMix ██████░░░░  (IIN 특화)
Ours    ████████░░  (IDN 강점, IIN도 경쟁력)

클린 서브셋 독립성:
Ours = DivideMix = ELR = PropMix (모두 불필요)
[17,21,52] (클린 서브셋 필요) ← 이 논문이 대체

단일 모델 단순성:
DivideMix/AugDesc: 2-모델 ← 복잡
Ours/ELR/PLC: 1-모델 ← 단순
```

**핵심 차별점**: 이 논문은 IDN에서 **노이즈 레이블을 모델 입력으로 활용**하는 유일한 클린 서브셋 불필요 방법입니다. 기존 방법들은 모두 노이즈 레이블을 입력에서 제외하거나, 활용 시 클린 서브셋을 요구했습니다.

---

## 참고 자료

**주요 참고 논문 (논문 내 인용 기준):**

- **[본 논문]** Smart, B., & Carneiro, G. (2022). *Bootstrapping the Relationship Between Images and Their Clean and Noisy Labels*. arXiv:2210.08826.
- **[DivideMix]** Li, J., Socher, R., & Hoi, S. C. H. (2020). *DivideMix: Learning with Noisy Labels as Semi-supervised Learning*. ICLR 2020.
- **[ELR]** Liu, S., et al. (2020). *Early-Learning Regularization Prevents Memorization of Noisy Labels*. NeurIPS 2020.
- **[PLC]** Zhang, Y., et al. (2021). *Learning with Feature-Dependent Label Noise: A Progressive Approach*. ICLR 2021.
- **[FixMatch]** Sohn, K., et al. (2020). *FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence*. NeurIPS 2020.
- **[SimCLR]** Chen, T., et al. (2020). *A Simple Framework for Contrastive Learning of Visual Representations*. ICML 2020.
- **[PropMix]** Cordeiro, F. R., et al. (2021). *PropMix: Hard Sample Filtering and Proportional MixUp for Learning with Noisy Labels*. BMVC 2021.
- **[AugDesc]** Nishi, K., et al. (2021). *Augmentation Strategies for Learning with Noisy Labels*. CVPR 2021.
- **[ContrastToDivide]** Zheltonozhskii, E., et al. (2022). *Contrast to Divide: Self-Supervised Pre-Training for Learning with Noisy Labels*. WACV 2022.
- **[NGC]** Wu, Z. F., et al. (2021). *NGC: A Unified Framework for Learning with Open-World Noisy Data*. ICCV 2021.
- **[FINE]** Kim, T., et al. (2021). *Fine Samples for Learning with Noisy Labels*. NeurIPS 2021.
- **[MixUp]** Zhang, H., et al. (2018). *MixUp: Beyond Empirical Risk Minimization*. ICLR 2018.
- **[PMD Noise]** Zhang, L., et al. (2020). *Learning to Segment When Experts Disagree*. MICCAI 2020.
