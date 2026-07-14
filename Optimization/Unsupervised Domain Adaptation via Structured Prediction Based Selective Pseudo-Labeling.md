# Unsupervised Domain Adaptation via Structured Prediction Based Selective Pseudo-Labeling

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

본 논문은 **비지도 도메인 적응(UDA)** 문제에서 기존 의사 레이블링(pseudo-labeling)의 부정확성으로 인한 오류 누적 문제를 해결하기 위해, **구조적 예측(Structured Prediction) 기반의 선택적 의사 레이블링(Selective Pseudo-Labeling, SPL)** 전략을 제안한다.

타겟 도메인의 샘플들이 딥 피처 공간에서 잘 군집화(cluster)된다는 사실에 착안하여, K-means 클러스터링을 통해 타겟 도메인의 구조적 정보를 탐색하고, 이를 활용해 의사 레이블의 정확도를 높인다.

### 주요 기여

| 기여 항목 | 설명 |
|-----------|------|
| **반복적 학습 알고리즘** | SLPP 기반 부분공간 학습과 선택적 의사 레이블링을 결합한 반복 학습 프레임워크 |
| **구조적 예측 기반 의사 레이블링** | K-means 클러스터링 + 선형 프로그래밍으로 타겟 클러스터와 소스 클래스를 1:1 매칭 |
| **클래스별 선택적 샘플 선택** | 특정 클래스 편향을 방지하는 클래스별 샘플 선택 전략 |
| **성능 검증** | 4개 벤치마크 데이터셋에서 당시 SOTA 달성 |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

**도메인 시프트(Domain Shift)** 문제:
- 소스 도메인의 레이블 데이터로 학습된 분류기는 타겟 도메인에 직접 적용 시 성능이 크게 저하됨
- 기존 의사 레이블링 방법은 타겟 샘플의 **구조적 정보(structural information)를 무시**하여 부정확한 레이블링 발생
- 부정확한 의사 레이블은 반복 학습 과정에서 **오류 누적(catastrophic error accumulation)** 야기

**기존 방법의 한계:**
- **선택 없는 의사 레이블링**: 초기 약한 분류기로 모든 샘플에 레이블 부여 → 오류 전파
- **Easy-to-hard 선택 전략**: 특정 "쉬운(easy)" 클래스에 치우친 샘플 선택 → 모델 편향
- **최근접 클래스 프로토타입(NCP)**: 소스 분포에 가까운 샘플만 선택 → 도메인 정렬 불완전

---

### 2.2 제안하는 방법 (수식 포함)

#### Step 1: 차원 축소 (PCA)

소스/타겟 데이터를 합쳐 행렬 구성:

$$\mathbf{X} = [\mathbf{x}^s_1, ..., \mathbf{x}^s_{n_s}, \mathbf{x}^t_1, ..., \mathbf{x}^t_{n_t}] \in \mathbb{R}^{d \times n}$$

PCA 목적함수:

$$\max_{\mathbf{V}^T \mathbf{V} = \mathbf{I}} \text{tr}(\mathbf{V}^T \mathbf{X} \mathbf{H} \mathbf{X}^T \mathbf{V}) \tag{1}$$

여기서 $\mathbf{H} = \mathbf{I} - \frac{1}{n}\mathbf{1}$은 centering matrix. 이는 다음 고유값 문제와 동치:

$$\mathbf{X}\mathbf{H}\mathbf{X}^T \mathbf{v} = \phi \mathbf{v} \tag{2}$$

저차원 피처:

$$\tilde{\mathbf{X}} = \mathbf{V}^T \mathbf{X} \tag{3}$$

이후 L2 정규화 적용: $\tilde{x} \leftarrow \tilde{x}/||\tilde{x}||_2$

---

#### Step 2: 도메인 정렬 - SLPP (Supervised Locality Preserving Projection)

SLPP 목적함수 (같은 클래스 샘플을 부분공간에서 가깝게):

$$\min_{\mathbf{P}} \sum_{i,j} ||\mathbf{P}^T \tilde{\mathbf{x}}_i - \mathbf{P}^T \tilde{\mathbf{x}}_j||^2_2 M_{ij} \tag{4}$$

유사도 행렬 $\mathbf{M}$:

$$M_{ij} = \begin{cases} 1, & y_i = y_j \\ 0, & \text{otherwise} \end{cases} \tag{5}$$

등가 최대화 형태 (라플라시안 정규화 포함):

$$\max_{\mathbf{P}} \frac{\text{tr}(\mathbf{P}^T \tilde{\mathbf{X}}^l \mathbf{D} \tilde{\mathbf{X}}^{lT} \mathbf{P})}{\text{tr}(\mathbf{P}^T (\tilde{\mathbf{X}}^l \mathbf{L} \tilde{\mathbf{X}}^{lT} + \mathbf{I}) \mathbf{P})} \tag{6}$$

여기서 $\mathbf{L} = \mathbf{D} - \mathbf{M}$ (라플라시안 행렬), $D_{ii} = \sum_j M_{ij}$

일반화 고유값 문제:

$$\tilde{\mathbf{X}}^l \mathbf{D} \tilde{\mathbf{X}}^{lT} \mathbf{p} = \lambda (\tilde{\mathbf{X}}^l \mathbf{L} \tilde{\mathbf{X}}^{lT} + \mathbf{I}) \mathbf{p} \tag{7}$$

---

#### Step 3a: 최근접 클래스 프로토타입(NCP) 기반 의사 레이블링

투영:

$$\mathbf{z}^s = \mathbf{P}^T \tilde{\mathbf{x}}^s, \quad \mathbf{z}^t = \mathbf{P}^T \tilde{\mathbf{x}}^t \tag{8}$$

소스 클래스 프로토타입:

$$\bar{\mathbf{z}}^s_y = \frac{\sum_{i=1}^{n_s} \mathbf{z}^s_i \delta(y, y^s_i)}{\sum_{i=1}^{n_s} \delta(y, y^s_i)} \tag{9}$$

타겟 샘플의 클래스 $y$ 조건부 확률:

$$p_1(y|\mathbf{x}^t) = \frac{\exp(-||\mathbf{z}^t - \bar{\mathbf{z}}^s_y||)}{\sum_{y=1}^{|\mathcal{Y}|} \exp(-||\mathbf{z}^t - \bar{\mathbf{z}}^s_y||)} \tag{10}$$

---

#### Step 3b: 구조적 예측(SP) 기반 의사 레이블링

K-means로 $|\mathcal{Y}|$개 클러스터 생성 후, **1:1 매칭 최적화**:

$$\min_{\mathbf{A}} \sum_{i=1}^{|\mathcal{Y}|} \sum_{j=1}^{|\mathcal{Y}|} A_{ij} d(\bar{\mathbf{z}}^t_i, \bar{\mathbf{z}}^s_j)$$

$$\text{s.t.} \quad \forall i, \sum_j A_{ij} = 1; \quad \forall j, \sum_i A_{ij} = 1 \tag{11}$$

여기서 $\mathbf{A} \in \{0,1\}^{|\mathcal{Y}| \times |\mathcal{Y}|}$는 매칭 행렬. **선형 프로그래밍**으로 효율적 해결.

타겟 클러스터 중심 기반 확률:

$$p_2(y|\mathbf{x}^t) = \frac{\exp(-||\mathbf{z}^t - \bar{\mathbf{z}}^t_y||)}{\sum_{y=1}^{|\mathcal{Y}|} \exp(-||\mathbf{z}^t - \bar{\mathbf{z}}^t_y||)} \tag{12}$$

---

#### Step 4: 선택적 의사 레이블링 - 두 방법의 결합

두 확률의 최대값을 취해 상호 보완:

$$p(y|\mathbf{x}^t) = \max\{p_1(y|\mathbf{x}^t), p_2(y|\mathbf{x}^t)\} \tag{13}$$

최종 의사 레이블 예측:

$$\hat{y}^t = \arg\max_{y \in \mathcal{Y}} p(y|\mathbf{x}^t) \tag{14}$$

$k$번째 반복에서 클래스별로 상위 $kn^c_t/T$개 샘플 선택하여 $\mathcal{S}_k$ 구성.

---

### 2.3 모델 구조

```
입력: 소스 레이블 데이터 D^s, 타겟 비레이블 데이터 D^t
  │
  ▼
[1단계] PCA 차원 축소 (d → d1) + L2 정규화
  │
  ▼
[2단계] 소스 데이터만으로 초기 SLPP 투영 P0 학습
  │
  ▼
[반복 학습 (k = 1, ..., T)]
  │
  ├─► NCP 기반 확률 p1(y|x^t) 계산
  ├─► K-means 클러스터링 → SP 기반 확률 p2(y|x^t) 계산
  ├─► p = max{p1, p2} 결합 → 의사 레이블 부여
  ├─► 클래스별 선택적 샘플 선택 (S_k)
  └─► D^s + S_k 로 SLPP 재학습 (P_k 갱신)
  │
  ▼
출력: 최종 투영 행렬 P, 타겟 샘플 예측 레이블 {ŷ^t}
```

**하이퍼파라미터:**
- $d_1$: PCA 공간 차원 (128~1024, 클래스 수에 비례 설정)
- $d_2 = 128$: SLPP 공간 차원 (전 데이터셋 동일)
- $T = 10$: 반복 횟수

**계산 복잡도:**
- PCA: $\mathcal{O}(dn^2 + d^3)$
- SLPP (T회 반복): $\mathcal{O}(T(2d_1 n^2 + d_1^3))$

---

### 2.4 성능

| 데이터셋 | SPL (제안) | 2위 방법 | 향상폭 |
|----------|-----------|---------|--------|
| Office-Caltech | **93.0%** | MEDA (92.8%) | +0.2% |
| Office31 | **89.6%** | SymNets/TADA (88.4%) | +1.2% |
| ImageCLEF-DA | **90.3%** | SymNets (89.9%) | +0.4% |
| Office-Home | **71.0%** | CAPLS (70.6%) | +0.4% |

**Ablation Study 주요 결과:**

| PL | S | NCP | SP | Office-Caltech | Office-Home |
|----|---|-----|----|----------------|-------------|
| ✗ | ✗ | ✓ | ✗ | 81.8 | 63.9 |
| ✗ | ✗ | ✗ | ✓ | 90.3 | 68.0 |
| ✓ | ✓ | ✗ | ✓ | 93.0 | 71.0 |
| ✓ | ✓ | ✓ | ✓ | **93.0** | **71.0** |

---

### 2.5 한계

1. **메모리 사용량**: 샘플 수가 많은 데이터셋(MNIST-SVHN 등 digit 데이터셋)에 적용 불가
2. **비딥러닝 기반**: Matlab 구현의 피처 변환 방식으로, 완전한 end-to-end 딥러닝 모델 대비 유연성 부족
3. **K-means 의존성**: 클래스 수 $|\mathcal{Y}|$가 사전 알려져야 하며, 클러스터 품질이 성능에 영향
4. **계산 비용**: Office-Home의 경우 약 2070초 소요 (CPU 기준)
5. **동적 도메인**: 복수 타겟 도메인이나 지속적으로 변화하는 도메인에 대한 고려 없음

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화를 높이는 설계 요소

**① 구조적 예측을 통한 전역적 타겟 정보 활용**

NCP가 타겟 샘플 각각을 독립적으로 레이블링하는 반면, SP는 타겟 도메인 전체의 클러스터 구조를 고려하여 **집단적(collective) 레이블링**을 수행한다. 이는 클러스터 내 샘플들이 공유하는 구조적 정보를 의사 레이블 부여에 활용함으로써 개별 샘플의 이상값(outlier) 영향을 줄인다.

**② 클래스별 선택적 샘플 선택**

단순 확률 상위 $k$개 선택이 아닌 **클래스별 균등 선택**:

$$\mathcal{S}_k: \text{각 클래스 } c \in \mathcal{Y} \text{에서 상위 } \frac{k n^c_t}{T} \text{개 선택}$$

이를 통해 클래스 불균형으로 인한 모델 편향을 방지하고, 조건부 분포 $P(Y|X)$ 정렬을 클래스별로 균형 있게 달성한다.

**③ 두 의사 레이블링 방법의 상호보완적 결합**

$$p(y|\mathbf{x}^t) = \max\{p_1(y|\mathbf{x}^t), p_2(y|\mathbf{x}^t)\}$$

- $p_1$ (NCP): 소스에 가까운 타겟 샘플에 높은 확신
- $p_2$ (SP): 타겟 클러스터 중심에 가까운 샘플에 높은 확신
- 결합: 소스-타겟 간 **겹치는 영역부터 외곽 영역까지** 순차적으로 포괄

이는 단일 방법보다 다양한 위치의 타겟 샘플을 정확하게 레이블링하여 일반화를 향상시킨다.

**④ L2 정규화를 통한 초구(hypersphere) 정렬**

$$\tilde{x} \leftarrow \tilde{x}/||\tilde{x}||_2$$

소스/타겟 샘플을 동일한 초구 표면에 분포시켜 도메인 간 특징 분포를 암묵적으로 정렬하고, 피처 스케일 차이로 인한 일반화 저하를 방지한다.

**⑤ 하이퍼파라미터 강건성**

실험에서 $T \geq 2$이면 수렴, $d_2 > |\mathcal{Y}|$이면 안정적 성능을 보임 → 다양한 설정에서 안정적 일반화 가능성 확인.

### 3.2 일반화 한계와 개선 가능성

| 현재 한계 | 잠재적 개선 방향 |
|-----------|-----------------|
| 고정된 K-means 클러스터 수 ( $= \mid \mathcal{Y} \mid$ ) | 계층적 클러스터링 또는 동적 클러스터 수 결정 |
| 선형 투영(SLPP) 기반 | 비선형 커널 SLPP 또는 딥러닝과 결합 |
| 타겟 도메인 단일 가정 | 멀티 소스/멀티 타겟 확장 |
| 클로즈드-셋 가정 ($\mathcal{Y}^t = \mathcal{Y}^s$) | 오픈셋 또는 부분 도메인 적응으로 확장 |

---

## 4. 앞으로의 연구에 미치는 영향 및 고려사항

### 4.1 연구에 미치는 영향

**① 구조적 정보 활용의 중요성 부각**

타겟 도메인의 비지도 군집 구조를 적극적으로 활용하는 패러다임을 제시. 이후 연구들이 self-supervised learning, contrastive learning과 결합하여 타겟 도메인의 구조적 표현 학습을 심화시키는 방향으로 발전.

**② 선택적 의사 레이블링의 체계화**

단순한 신뢰도 기반 선택을 넘어, **클래스별 균형**, **구조적 정보**, **방법의 상호보완성**을 체계적으로 설계하는 프레임워크 제공.

**③ 딥러닝과의 통합 가능성 제시**

논문의 결론에서 "selective pseudo-labeling and structured prediction can also be employed to train the deep learning models for UDA"를 명시 → 이후 딥 UDA 연구의 의사 레이블링 전략 설계에 영향.

---

### 4.2 2020년 이후 최신 연구 비교 분석

> ⚠️ **주의**: 아래 비교는 논문 제출 시점(2019년) 이후 발표된 연구들과의 개념적 비교이며, 직접 수치 비교는 실험 설정(백본, 프로토콜)이 다를 수 있어 주의가 필요합니다.

#### 주요 후속 연구 동향

**① Contrastive Adaptation Network 계열 (2020~)**

- **CDAC (Li et al., CVPR 2021)**: 대조 학습(contrastive learning)을 UDA에 적용, 클래스 경계를 명확히 하는 방향. SPL의 클러스터 구조 활용과 유사한 동기이나, 딥러닝 end-to-end로 구현.
- **CDTrans (Xu et al., ICLR 2022)**: Transformer 기반으로 소스-타겟 교차 주의(cross-attention)를 활용. 구조적 정보 활용의 관점에서 SPL의 후속 발전.

**② 자기지도 학습 결합 (Self-Supervised + DA)**

- **MDD (Zhang et al., ICML 2019) → SHOT (Liang et al., ICML 2020)**: 소스 데이터 없이 타겟 도메인의 정보 최대화(information maximization)로 적응. SPL과 달리 소스 데이터 접근 불필요 → **소스-프리(source-free) UDA** 패러다임 등장.

**③ Transformer 기반 UDA (2021~)**

- **TVT (Yang et al., 2021)**: Vision Transformer(ViT)를 UDA 백본으로 사용. SPL은 ResNet50 피처 기반으로 ViT 시대의 성능과 직접 비교 어려움.

#### 개념적 비교표

| 항목 | SPL (본 논문, 2019) | SHOT (ICML 2020) | CDTrans (ICLR 2022) |
|------|---------------------|------------------|----------------------|
| **방법론** | SLPP + 구조적 예측 | 정보 최대화 | Cross-attention Transformer |
| **딥러닝 여부** | 피처 변환 (비딥) | End-to-end | End-to-end |
| **소스 데이터** | 필요 | 불필요 (source-free) | 필요 |
| **타겟 구조 활용** | K-means + 선형 프로그래밍 | 정보 엔트로피 최소화 | Attention 기반 |
| **Office31 평균** | 89.6% | ~94% | ~97% |
| **해석 가능성** | 높음 (명시적 클러스터링) | 중간 | 낮음 |

---

### 4.3 앞으로의 연구 시 고려할 점

**① End-to-End 딥러닝과의 통합**
- 현재 피처 변환 방식을 딥러닝과 결합하여 SLPP와 유사한 목적함수를 신경망의 손실 함수로 구현 가능
- 예: SLPP 목적함수를 정규화 항으로 포함하는 학습 프레임워크

**② 소스-프리(Source-Free) UDA로의 확장**
- 프라이버시/보안 이슈로 소스 데이터 접근 불가한 경우가 증가
- SP 기반의 타겟 도메인 내 구조 탐색은 소스 데이터 없이도 활용 가능한 방향성 제공

**③ 오픈셋 및 부분 도메인 적응**
- $\mathcal{Y}^t = \mathcal{Y}^s$ 가정 완화: 타겟에만 존재하거나 소스에만 존재하는 클래스 처리 필요
- K-means 클러스터 수를 적응적으로 결정하는 메커니즘 연구 필요

**④ 대규모 데이터셋 확장성**
- 현재 메모리 한계(MNIST-SVHN 등 부적합) 극복 필요
- 미니배치 기반 SLPP 또는 온라인 클러스터링 알고리즘 도입 고려

**⑤ 노이즈에 강건한 의사 레이블링**
- 초기 클러스터링 품질에 대한 민감도 분석 및 노이즈 레이블에 강건한 손실 함수(예: symmetric cross-entropy) 결합

**⑥ Transformer 백본과의 호환**
- ResNet50 피처 대신 ViT/DINO 등 최신 피처를 사용했을 때의 성능 변화 및 재설계 필요성 탐구

---

## 참고자료

**주 논문:**
- Wang, Q., & Breckon, T. P. (2019). *Unsupervised Domain Adaptation via Structured Prediction Based Selective Pseudo-Labeling*. arXiv:1911.07982v1. AAAI 2020.

**논문 내 인용 참고문헌 (주요):**
- He, X., & Niyogi, P. (2004). Locality Preserving Projections. *NIPS*.
- Zhang, Z., & Saligrama, V. (2016). Zero-Shot Recognition via Structured Prediction. *ECCV*.
- Wang, Q., Bu, P., & Breckon, T. P. (2019). Unifying Unsupervised Domain Adaptation and Zero-Shot Visual Recognition. *IJCNN*.
- Long, M., et al. (2018). Conditional Adversarial Domain Adaptation. *NeurIPS*.
- Zhang, Y., et al. (2019). Domain-Symmetric Networks for Adversarial Domain Adaptation. *CVPR*.
- Pei, Z., et al. (2018). Multi-Adversarial Domain Adaptation. *AAAI*.

**비교 분석 관련 후속 연구 (2020년 이후):**
- Liang, J., et al. (2020). Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation. *ICML 2020*. (SHOT)
- Li, R., et al. (2021). Contrastive Adaptation Network for Single- and Multi-Source Domain Adaptation. *CVPR 2021*.
- Xu, T., et al. (2022). CDTrans: Cross-domain Transformer for Unsupervised Domain Adaptation. *ICLR 2022*.
