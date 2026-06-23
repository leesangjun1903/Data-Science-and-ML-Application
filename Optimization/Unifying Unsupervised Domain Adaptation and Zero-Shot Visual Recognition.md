# Unifying Unsupervised Domain Adaptation and Zero-Shot Visual Recognition

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

이 논문은 **비지도 도메인 적응(Unsupervised Domain Adaptation, UDA)**과 **제로샷 학습(Zero-Shot Learning, ZSL)**을 **하나의 통합 프레임워크**로 다룰 수 있다고 주장합니다.

기존 UDA의 핵심 한계는 학습 시 타겟 도메인의 (레이블 없는) 테스트 데이터가 반드시 필요하다는 점입니다. 이는 실제 배포 환경에서 **미관측(out-of-sample) 데이터에 직접 적용하기 어렵다**는 문제를 야기합니다. 저자들은 이를 해결하기 위해:

> "소스 도메인의 레이블된 샘플이 ZSL의 side information 역할을 한다"

는 새로운 관점을 제시하고, **Supervised Locality Preserving Projection (SLPP)** 기반의 공동 부분공간(joint subspace) 학습과 **Confidence-Aware Pseudo Label Selection (CAPLS)** 전략을 통합한 프레임워크를 제안합니다.

### 주요 기여

| 기여 | 내용 |
|------|------|
| **통합 프레임워크 제안** | UDA와 ZSL을 단일 프레임워크로 통합 |
| **새로운 문제 정의** | 타겟 도메인 일부 클래스(known classes)만 레이블 존재 → Generalized ZSL로 정식화 |
| **CAPLS 전략** | 신뢰도 기반 의사레이블(pseudo label) 선택으로 오류 전파 억제 |
| **SLPP 활용** | 데이터 구조 보존 + 도메인 불변(domain-invariant) 부분공간 학습 |
| **SOTA 달성** | Office-Caltech, Office31, Office-Home 3개 벤치마크에서 최고 수준 성능 |

---

## 2. 상세 분석

### 2-1. 해결하고자 하는 문제

**문제 1: 기존 UDA의 한계**

기존 UDA는 학습 시 타겟 도메인 데이터(레이블 없음)에 접근 가능하다고 가정합니다. 그러나 이는 현실에서:

- 테스트 데이터를 사전에 수집해야 하므로 **범용성이 떨어짐**
- 미관측 샘플 분류 불가 → **out-of-sample 분류 문제**

**문제 2: 새로운 크로스 도메인 분류 문제 정의**

$$\mathcal{D}^s = \{(x_i^s, y_i^s)\}, \quad i=1,2,...,n^s, \quad x_i^s \in \mathbb{R}^{d^s}, \quad y_i^s \in \mathcal{Y}^s$$

$$\mathcal{D}^{tl} = \{(x_i^{tl}, y_i^{tl})\}, \quad i=1,2,...,n^{tl}, \quad y_i^{tl} \in \mathcal{Y}^{tl} \subset \mathcal{Y}^t = \mathcal{Y}^s$$

- **ZSL 조건**: $\mathcal{Y}^{tl} \subset \mathcal{Y}^t$ (타겟 도메인 중 일부 클래스만 레이블 존재)
- **목표**: 학습 후 $\mathcal{Y}^t$ 전체(known + unseen 클래스)에 속하는 새 샘플 분류
- **핵심 어려움**: 모델이 known class에 편향되어 unseen class 샘플을 잘못 분류하는 문제

---

### 2-2. 제안 방법 (수식 포함)

#### Step 1: 데이터 전처리 (L2 정규화)

$$\hat{x} = x / \|x\|_2 \tag{1}$$

L2 정규화는 데이터를 초구면(hyper-sphere) 위에 분포시켜 도메인 간 정렬을 용이하게 합니다.

#### Step 2: Joint Subspace Learning (SLPP)

투영 행렬 $P \in \mathbb{R}^{d^s \times d}$ 학습을 위한 비용 함수:

$$L(P; W, X^l) = \sum_{i,j} \|P^T x_i - P^T x_j\|_2^2 W_{ij} \tag{2}$$

유사도 행렬 $W$의 정의:

$$W_{ij} = \begin{cases} 1, & y_i = y_j \\ 0, & \text{otherwise} \end{cases} \tag{3}$$

- 동일 클래스 샘플(소스/타겟 구분 없이)을 부분공간에서 가깝게 만듦
- 클래스 내 거리 대신 **레이블 일치 여부**만 사용 → 도메인 불변성 확보

Eq.(2)의 최적화 문제를 다음과 같이 재공식화:

$$\max_P \frac{Tr(P^T X^l D X^{lT} P)}{Tr(P^T (X^l L X^{lT} + I) P)} \tag{4}$$

- $L = D - W$: 라플라시안 행렬
- $D_{ii} = \sum_j W_{ij}$: 대각 행렬
- 정규화 항 $Tr(P^T P)$: 투영 행렬의 극단값 방지

이는 다음 일반화 고유값 문제(generalized eigenvalue problem)와 동치:

$$X^l D X^{lT} p = \lambda (X^l L X^{lT} + I) p \tag{5}$$

**최적 해**: $P = [p_1, ..., p_d]$ (가장 큰 $d$개의 고유값에 대응하는 고유벡터)

#### Step 3: 부분공간에서의 투영 및 분류

투영:
$$z_i = P^T x_i \tag{6}$$

중앙화 및 L2 정규화:
$$z \leftarrow z - \bar{z} \tag{7}$$
$$z \leftarrow z / \|z\|_2 \tag{8}$$

분류 (최근접 클래스 평균):
$$\hat{y}^t = \arg\min_y \|z^t - \bar{z}_y\|_2, \quad y \in \mathcal{Y}^t \tag{9}$$

클래스 평균 벡터:
$$\bar{z}_y = \frac{\sum_i z_i^s \delta(y, y_i^s) + \sum_j z_j^{lt} \delta(y, y_j^{lt})}{\sum_i \delta(y, y_i^s) + \sum_j \delta(y, y_j^{lt})} \tag{10}$$

#### Step 4: CAPLS (Confidence-Aware Pseudo Label Selection)

UDA의 경우 타겟 레이블이 없으므로, 소프트맥스 기반 신뢰도:

$$q_i = \frac{e^{-d_i}}{\sum_{i=1}^{|\mathcal{Y}^s|} e^{-d_i}} \tag{11}$$

- $d_i = \|z - \bar{z}_i\|_2$: 테스트 샘플과 $i$번째 클래스 대표 벡터 간 거리
- $t$번째 이터레이션에서 각 클래스별 **상위 $t/T$ 퍼센트** 의사레이블 샘플만 선택
- 시간 복잡도: $\mathcal{O}(T(d^3 + dn^2))$

---

### 2-3. 모델 구조

```
[소스 도메인 레이블 데이터 D^s] ──┐
                                  ├──► [L2 정규화] ──► [SLPP 기반 Joint Subspace 학습]
[타겟 도메인 데이터]             ──┘         │                        │
  (UDA: 레이블 없음)                         │                   투영 행렬 P
  (ZSL: 일부 클래스 레이블 있음)             │                        │
                                             └────────► [부분공간 투영 z=P^T x]
                                                                 │
                                              [중앙화 + L2 정규화] │
                                                                 │
                                                    [최근접 클래스 평균 분류기]
                                                                 │
                                                           [예측 레이블 ŷ]

※ UDA 전용: CAPLS 이터레이티브 학습
   t=0 → 소스만으로 P 초기화 → 의사레이블 생성 → 신뢰도 상위 선택 → P 재학습 → 반복
```

---

### 2-4. 성능 향상

#### Office+Caltech (UDA, Decaf6 features)

| 방법 | 평균 정확도 |
|------|------------|
| MEDA [7] | 92.8% |
| **CAPLS (Ours)** | **91.8%** |
| DAN [9] | 90.1% |
| JGSA [4] | 90.0% |

#### Office31 (UDA, ResNet50 features)

| 방법 | 평균 정확도 |
|------|------------|
| **CAPLS (Ours)** | **88.2%** |
| CDAN-M [16] | 87.5% |
| iCAN [15] | 87.2% |
| MEDA [7] | 85.7% |

> Office31에서 당시 SOTA 딥러닝 방법들을 모두 능가

#### Office-Home (UDA, ResNet50 features)

| 방법 | 평균 정확도 |
|------|------------|
| **CAPLS (Ours)** | **70.6%** |
| MEDA [7] | 67.0% |
| CDAN-M [16] | 62.8% |

#### Office-Home (ZSL, Harmonic Mean H)

| 방법 | R→A (H) | R→C (H) | R→P (H) |
|------|---------|---------|---------|
| **Ours** | **68.2** | **57.0** | **83.6** |
| LDA | 64.5 | 53.3 | 83.1 |
| BiDiLEL | 15.9 | 14.9 | 24.2 |
| MR | 4.0 | 4.5 | 7.0 |

> ZSL 조건에서 기존 ZSL 방법들(BiDiLEL, MR)이 unseen 클래스 분류에서 완전히 실패한 반면, 제안 방법은 known/unseen 균형 달성

---

### 2-5. 한계

1. **핸드크래프트 특징(hand-crafted features)에서 성능 저하**: 딥 특징 대비 도메인 이동이 크면 초기 의사레이블의 정확도가 낮아 CAPLS의 이터레이티브 학습이 부정적 효과를 가져올 수 있음

2. **선형 투영의 한계**: SLPP는 선형 변환이므로, 복잡한 비선형 도메인 이동을 완전히 포착하기 어려움

3. **ZSL에서의 class imbalance**: Known/unseen 클래스 분할 시 per-class accuracy로 평가하지만, 실제 데이터 불균형 문제에 대한 심층 분석 부재

4. **하이퍼파라미터 의존성**: 부분공간 차원 $d$와 이터레이션 수 $T$를 수동 설정해야 함 (실험적으로 $d=128$, $T=20$ 고정)

5. **확장성 문제**: 시간 복잡도 $\mathcal{O}(T(d^3 + dn^2))$로 대규모 데이터셋 적용 시 계산 비용이 증가

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. SLPP의 구조 보존 특성

논문은 SLPP가 **데이터의 지역 구조(local structure)를 보존**하면서 투영을 수행하므로, 훈련 데이터에 과적합(overfitting)되지 않는다고 주장합니다. 이는 특히:

$$\text{훈련 분포} \neq \text{테스트 분포} \quad \Rightarrow \quad \text{SLPP가 유리}$$

ZSL 조건처럼 테스트 시 unseen class의 샘플이 등장해도, 부분공간에서의 **클래스 평균 기반 분류**는 새로운 클래스로의 일반화를 자연스럽게 지원합니다.

### 3-2. Generalized ZSL로의 확장

기존 ZSL은 테스트 샘플이 오직 unseen 클래스에만 속한다고 가정합니다. 이 논문은 **Generalized ZSL(GZSL)** 설정을 채택하여:

$$\mathcal{Y}^{test} = \mathcal{Y}^{known} \cup \mathcal{Y}^{unseen}$$

이를 통해 실제 배포 환경에 더 가까운 평가를 수행하며, Harmonic Mean $H$를 지표로 사용:

$$H = \frac{2 \times Acc^{known} \times Acc^{unseen}}{Acc^{known} + Acc^{unseen}}$$

이는 known/unseen 클래스 간 **균형 잡힌 일반화**를 장려합니다.

### 3-3. 도메인 불변 부분공간에서의 일반화

$W_{ij}$를 도메인에 무관하게 **레이블 일치 여부**로만 정의함으로써:

- 소스 도메인 샘플과 타겟 도메인 샘플이 같은 클래스이면 부분공간에서 가깝게 배치
- 이 특성은 **새로운 타겟 도메인 샘플**에도 자동으로 적용 → out-of-sample 일반화

### 3-4. CAPLS의 점진적 학습을 통한 일반화

$$\text{iteration } t: \text{상위 } \frac{t}{T} \times 100\% \text{ 신뢰도 샘플 선택}$$

- 초기에는 보수적으로 선택 → 오류 전파 최소화
- 점진적으로 선택 범위 확대 → **점진적 도메인 정렬**
- 이는 curriculum learning과 유사한 방식으로 일반화를 향상

---

## 4. 앞으로의 연구에 미치는 영향 및 고려사항

### 4-1. 연구에 미치는 영향

**① UDA-ZSL 통합 연구 방향 제시**

이 논문은 UDA와 ZSL을 별개로 다루던 기존 관행에서 벗어나, 두 문제를 **단일 프레임워크로 통합**할 수 있음을 보여주었습니다. 이는 후속 연구들이 도메인 적응과 학습 가능성(learnability)을 함께 고려하도록 촉진합니다.

**② 현실적인 문제 설정의 중요성**

테스트 데이터 없이 학습하는 ZSL 조건은 실제 산업 적용에서 매우 중요합니다. 이 프레임워크는:
- 의료 영상 분석 (새로운 질병 유형)
- 자율주행 (미관측 도로 상황)
- 산업 결함 검출 (새로운 결함 유형)

등에 영감을 제공합니다.

**③ 부분공간 학습 + 의사레이블의 결합**

CAPLS의 신뢰도 기반 선택 전략은 이후 **자기지도 학습(self-supervised learning)**과 **준지도 학습(semi-supervised learning)** 연구에서 의사레이블 품질 관리 방법으로 널리 영향을 미쳤습니다.

---

### 4-2. 2020년 이후 관련 최신 연구 비교 분석

#### 비교 1: 딥러닝 기반 도메인 적응 발전

| 논문 | 핵심 방법 | 이 논문과의 차이 |
|------|-----------|----------------|
| **SHOT** (Liang et al., ICML 2020) | 소스 가설 전이 (source hypothesis transfer), 정보 최대화 | 소스 데이터 없이 타겟만으로 적응; 이 논문은 소스 필요 |
| **TransDA** (Xu et al., NeurIPS 2021) | Transformer 기반 도메인 적응 | 자기 주의(self-attention)로 비선형 관계 포착; 이 논문은 선형 SLPP |
| **CLIP-based DA** (various, 2022~) | 대규모 사전학습 모델(CLIP) 활용 | 파운데이션 모델의 강력한 표현력 활용; 이 논문은 태스크 특화 학습 |

#### 비교 2: 제로샷 학습 발전

| 논문 | 핵심 방법 | 이 논문과의 차이 |
|------|-----------|----------------|
| **f-CLSWGAN** (Xian et al., CVPR 2018, 이후 확장) | 특징 생성(feature generation) via GAN | 합성 unseen 특징 생성; 이 논문은 부분공간 정렬 |
| **FREE** (Chen et al., NeurIPS 2021) | Feature refinement for ZSL | 세밀한 특징 정제; 이 논문은 전역 부분공간 학습 |
| **TransZero** (Chen et al., AAAI 2022) | Transformer + attribute-guided ZSL | 의미적 속성 활용; 이 논문은 소스 샘플을 side information으로 활용 |

#### 비교 3: 소스 없는 도메인 적응 (Source-Free DA)

이 논문 이후 중요한 연구 흐름으로, 소스 데이터 자체를 보호해야 하는 **프라이버시 보존** 관점에서:

- **SFDA** (Kim et al., ECCV 2020): 소스 없이 타겟만으로 적응
- 이 논문의 ZSL 설정은 소스 데이터는 있지만 타겟 테스트 데이터가 없다는 점에서 **상호 보완적** 연구 방향

#### 비교 4: 오픈셋 도메인 적응 (Open-Set DA)

이 논문의 ZSL 설정(unseen 클래스 존재)은 오픈셋 DA와 밀접:

- **DAOD** (various): 타겟에 소스에 없는 클래스가 존재
- 이 논문은 unseen 클래스를 소스 클래스 공간 내에서 처리하지만, 완전한 오픈월드 시나리오는 미지원

---

### 4-3. 앞으로 연구 시 고려할 점

**① 딥러닝과의 통합**

현재 SLPP는 선형 투영이므로, **비선형 심층 부분공간 학습**으로 확장이 필요합니다. 예를 들어:
- Transformer 인코더를 SLPP의 딥 버전으로 대체
- $W_{ij}$ 행렬을 학습 가능한 어텐션 메커니즘으로 대체

**② 대규모 데이터셋 확장성**

시간 복잡도 $\mathcal{O}(T(d^3 + dn^2))$ 문제를 해결하기 위해:
- 근사 고유값 분해 (Randomized SVD)
- 미니배치 기반 온라인 부분공간 학습

**③ 더 현실적인 ZSL 설정**

- 클래스 수 불균형 처리
- 도메인 이동의 강도가 다를 때의 적응
- **증분 학습(continual learning)**: 새로운 unseen 클래스가 순차적으로 등장

**④ 파운데이션 모델(Foundation Model) 시대에서의 재정립**

CLIP, DALL-E 등 대규모 사전학습 모델이 등장한 이후, 이 논문의 프레임워크를 **파운데이션 모델의 파인튜닝** 맥락에서 재해석할 필요:
- 소스 도메인 → 사전학습 데이터 분포
- SLPP → 프롬프트 기반 도메인 정렬

**⑤ 이론적 일반화 경계 분석**

$\mathcal{H}$-divergence나 도메인 이동 거리 관점에서 제안 방법의 **일반화 경계(generalization bound)**를 이론적으로 분석하는 연구가 필요합니다.

---

## 참고 자료

### 논문 원문 (제공된 PDF)
- **Wang, Q., Bu, P., & Breckon, T. P. (2019). "Unifying Unsupervised Domain Adaptation and Zero-Shot Visual Recognition." arXiv:1903.10601v2.**

### 논문 내 주요 인용 문헌
- Long et al., "Transfer Feature Learning with Joint Distribution Adaptation," ICCV 2013
- Wang, J. et al., "Visual Domain Adaptation with Manifold Embedded Distribution Alignment (MEDA)," ACM MM 2018
- Long et al., "Conditional Adversarial Domain Adaptation (CDAN)," NeurIPS 2018
- Wang, Q. & Chen, K., "Zero-Shot Visual Recognition via Bidirectional Latent Embedding (BiDiLEL)," IJCV 2017
- He, X. & Niyogi, P., "Locality Preserving Projections," NIPS 2004
- Xian et al., "Zero-Shot Learning - A Comprehensive Evaluation," IEEE TPAMI 2018

### 비교 분석에 활용한 2020년 이후 연구 (일반적으로 알려진 주요 연구)
- Liang et al., "Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation (SHOT)," ICML 2020
- Chen et al., "TransZero: Attribute-guided Transformer for Zero-Shot Learning," AAAI 2022

> ⚠️ **정확도 주의**: 2020년 이후 최신 연구 비교 분석 부분은 제공된 PDF에 포함되지 않은 내용으로, 일반적으로 알려진 연구 동향을 기반으로 서술하였습니다. 세부 수치 비교는 해당 논문을 직접 확인하시기를 권장합니다.
