# Learning with Twin Noisy Labels for Visible-Infrared Person Re-Identification 

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

본 논문은 Visible-Infrared Person Re-Identification(VI-ReID) 분야에서 기존 연구들이 간과해온 **Twin Noisy Labels(TNL)** 문제를 최초로 정식으로 정의하고, 이를 해결하기 위한 **DuAlly Robust Training(DART)** 방법론을 제안한다.

TNL은 두 가지 노이즈가 동시에 존재하는 상황을 의미한다:
- **Noisy Annotation(NA)**: 적외선 모달리티의 낮은 인식 가능성으로 인한 잘못된 신원 레이블
- **Noisy Correspondence(NC)**: NA로 인해 교차 모달 쌍(cross-modal pairs)의 대응 관계가 잘못 구성되는 문제

### 주요 기여 (3가지)

| 기여 | 내용 |
|------|------|
| **문제 정의** | VI-ReID에서 TNL(NA + NC의 동시 발생) 문제를 최초 정형화 |
| **방법론 제안** | TNL에 강건한 DART 프레임워크 제안 (co-modeling + pair division + dually robust loss) |
| **실험 검증** | SYSU-MM01, RegDB 두 데이터셋에서 5개의 SOTA 대비 우수한 성능 입증 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**Twin Noisy Labels(TNL)**의 구체적 발생 메커니즘:

$$\text{NA: } y^t_i \neq \hat{y}^t_i \Rightarrow \text{NC: } y^p_{ij} \text{가 TP/TN/FP/FN 중 하나로 잘못 분류}$$

기존 방법들의 한계:
- 단일 모달리티 ReID의 NA 처리 방법(PurifyNet[30], [31])은 VI-ReID의 다중 모달 상황에 적용 불가
- 기존 noisy label 방법들은 카테고리 수가 적은 분류 문제 중심 (ReID는 수백~수천 개의 identity 존재)
- FP 또는 FN 중 하나만 처리하는 기존 방법([9],[24],[25])과 달리 TNL은 TP/TN/FP/FN 모두를 고려해야 함

---

### 2.2 제안 방법 (수식 포함)

DART는 세 가지 핵심 모듈로 구성된다.

#### **Module 1: Co-modeling (공동 모델링)**

두 개의 독립 네트워크 $A = \{F^v_A, C^v_A, F^r_A, C^r_A\}$와 $B = \{F^v_B, C^v_B, F^r_B, C^r_B\}$를 학습하여, 딥러닝의 **Memorization Effect**(깨끗한 샘플을 먼저 학습하는 특성)를 활용한다.

**Step 1.** 샘플별 cross-entropy 손실 계산:

$$\ell^{id}(\theta_t) = \{\ell^{id}_i\}^{N_t}_{i=1} = \{\mathcal{L}^{id}(x^t_i, y^t_i)\}^{N_t}_{i=1} \tag{2}$$

$$\mathcal{L}^{id}(x^t_i, y^t_i) = -\log P\left(y^t_i \mid C^t\left(F^t\left(x^t_i\right)\right)\right) \tag{3}$$

**Step 2.** 2-component Gaussian Mixture Model(GMM) 피팅:

$$p(\ell^{id} \mid \theta_t) = \sum^{K}_{k=1} \gamma_k \phi(\ell^{id} \mid k) \tag{4}$$

여기서 $\gamma_k$는 혼합 계수, $\phi(\ell^{id} \mid k)$는 $k$번째 구성요소의 확률 밀도.

**Step 3.** 작은 평균값을 가진 컴포넌트 $\kappa$(clean 샘플에 해당)에 대한 사후 확률로 **정제 신뢰도(clean confidence)** 계산:

$$w_i = p(\kappa \mid \ell^{id}_i) \tag{5}$$

> **핵심**: 네트워크 A가 추정한 신뢰도 $w$는 네트워크 B에 전달되고, 반대도 마찬가지 → 편향(bias) 누적 방지

---

#### **Module 2: Pair Division (쌍 분할)**

모달 내/간 쌍 $(x^{t_1}_i, x^{t_2}_j)$에 대해 임계값 $\eta = 0.5$를 기준으로:

$$\mathcal{S}^c = \{(x^{t_1}_i, x^{t_2}_j), y^p_{ij} \mid w_i > \eta, w_j > \eta\} \quad \text{(clean portion)}$$

$$\mathcal{S}^n = \{(x^{t_1}_i, x^{t_2}_j), y^p_{ij} \mid w_i > \eta, w_j \leq \eta\} \quad \text{(noisy portion)}$$

**대응 관계 정제 (Correspondence Rectification)**:

$$\hat{y}^p_{ij} = \mathbb{I}(y^p_{ij} \in \mathcal{S}^c) \odot y^p_{ij} \tag{6}$$

여기서 $\odot$는 XNOR 연산. FN 쌍의 추가 정제:

$$\hat{y}^p_{ij} = \mathbb{I}(C^t(F^t(x^{t_1}_i)) = C^t(F^t(x^{t_2}_j))), \quad \forall(x^{t_1}_i, x^{t_2}_j) \in \mathcal{S}^n \tag{7}$$

**분류 결과 요약:**

| 집합 | 원래 대응 | 신뢰도 조건 | 결과 |
|------|----------|------------|------|
| $\mathcal{S}^c$, Positive | $y^p_{ij}=1$ | $w_i>\eta, w_j>\eta$ | **TP** |
| $\mathcal{S}^n$, Positive | $y^p_{ij}=1$ | $w_j \leq \eta$ | **FP** |
| $\mathcal{S}^c$, Negative | $y^p_{ij}=0$ | $w_i>\eta, w_j>\eta$ | **TN** |
| $\mathcal{S}^n$, Negative | $y^p_{ij}=0$ | $w_j \leq \eta$ | **FN** |

---

#### **Module 3: Dually Robust Objective Function (이중 강건 목적 함수)**

전체 손실:

$$\mathcal{L} = \mathcal{L}^{sid} + \mathcal{L}^{qdr} \tag{8}$$

**① Soft Identification Loss (NA에 강건):**

$$\mathcal{L}^{sid} = -w_i \log P\left(y^t_i \mid C^t\left(F^t\left(x^t_i\right)\right)\right) \tag{9}$$

신뢰도 $w_i$로 손실을 가중하여 노이즈가 많은 샘플의 영향 감소.

**② Adaptive Quadruplet Loss (NC에 강건):**

$$\mathcal{L}^{qdr} = \mathcal{L}^{tri} + \mathcal{L}^{qdt} \tag{10}$$

$\mathcal{L}^{tri}$는 네 가지 쌍 조합에 적응적으로 작동:

$$\mathcal{L}^{tri} = m + \frac{(-1)^{(\hat{y}^p_{ij} \otimes \hat{y}^p_{ik})(1-\hat{y}^p_{ij})}d_{ij} + (-1)^{(\hat{y}^p_{ij} \otimes \hat{y}^p_{ik})(1-\hat{y}^p_{ik})}d_{ik}}{(-1)^{(1-\hat{y}^p_{ij})(1-\hat{y}^p_{ik})} 2^{\hat{y}^p_{ij} \odot \hat{y}^p_{ik}}} \tag{11}$$

여기서 $\otimes$는 XOR, $\odot$는 XNOR 연산. 추가 quadruplet term:

$$\mathcal{L}^{qdt} = (-1)^{\hat{y}^p_{ij}\hat{y}^p_{ik}}(\hat{y}^p_{ij} \odot \hat{y}^p_{ij})d_{is} \tag{12}$$

**네 가지 케이스별 손실 함수 적응 변환:**

| 케이스 | 조건 | 목표 | 손실 형태 |
|--------|------|------|-----------|
| **TP-TN** | $\hat{y}^p_{ij}=1, \hat{y}^p_{ik}=0$ | TP 당기기, TN 밀기 | $\mathcal{L}^{qdr} = [d_{ij} - d_{ik} + m]_+$ (식 13) |
| **FP-FN** | $\hat{y}^p_{ij}=0, \hat{y}^p_{ik}=1$ | FP 밀기, FN 당기기 | $\mathcal{L}^{qdr} = [-d_{ij} + d_{ik} + m]_+$ (식 14) |
| **TP-FN** | $\hat{y}^p_{ij}=1, \hat{y}^p_{ik}=1$ | 재샘플링된 기준쌍 활용 | $\mathcal{L}^{qdr} = [-d_{is} + \frac{d_{ij}+d_{ik}}{2} + m]_+$ (식 15) |
| **FP-TN** | $\hat{y}^p_{ij}=0, \hat{y}^p_{ik}=0$ | 재샘플링된 기준쌍 활용 | $\mathcal{L}^{qdr} = [d_{is} - \frac{d_{ij}+d_{ik}}{2} + m]_+$ (식 16) |

---

### 2.3 모델 구조

```
입력: Visible 이미지 {x^v_i} + Infrared 이미지 {x^r_i}
        ↓
[Warm-up 단계]
 두 네트워크 A, B를 vanilla cross-entropy로 초기화
        ↓
[매 에폭 반복]
  ┌─────────────────────────────────────┐
  │  Network A                          │
  │  F^v_A, F^r_A → 특징 추출          │
  │  C^v_A, C^r_A → 분류 예측          │
  │  GMM 피팅 → 신뢰도 w_A 추정        │
  │  w_A → Network B로 전달            │
  └─────────────────────────────────────┘
        ↕ (상호 교환)
  ┌─────────────────────────────────────┐
  │  Network B                          │
  │  동일 구조, 다른 초기화             │
  │  GMM 피팅 → 신뢰도 w_B 추정        │
  │  w_B → Network A로 전달            │
  └─────────────────────────────────────┘
        ↓
  [Pair Division]
  신뢰도 기반 → {TP, TN, FP, FN} 분류
        ↓
  [Dually Robust Loss]
  L^sid + L^qdr (= L^tri + L^qdt)
        ↓
출력: 추론 시 A, B의 평균 특징 벡터 사용
```

---

### 2.4 성능 향상

#### SYSU-MM01 데이터셋

| 노이즈 비율 | 방법 | Rank-1 (All) | mAP (All) |
|------------|------|-------------|-----------|
| 0% | ADP (기준) | 69.88% | 66.89% |
| 0% | **DART** | 68.72% | 66.29% |
| 20% | ADP | 25.44% | 23.71% |
| 20% | ADP-C (클린 데이터) | 63.67% | 61.57% |
| 20% | **DART** | **66.31%** | **64.13%** |
| 50% | ADP | 8.00% | 10.83% |
| 50% | ADP-C (클린 데이터) | 59.17% | 56.49% |
| 50% | **DART** | **60.27%** | **58.69%** |

> **핵심 발견**: DART는 노이즈가 20% 포함된 데이터로 학습했음에도 클린 데이터로 학습한 ADP-C보다 mAP 기준 **+4.16% 향상** (20% 노이즈 기준).

#### RegDB 데이터셋 (Visible→Thermal, 20% 노이즈)

| 방법 | Rank-1 | mAP |
|------|--------|-----|
| ADP | 50.71% | 35.92% |
| ADP-C | 78.39% | 70.02% |
| **DART** | **82.04%** | **74.18%** |

---

### 2.5 한계

논문에서 명시적으로 언급되거나 분석을 통해 도출되는 한계:

1. **임계값 $\eta$ 민감성**: $\eta = 0.5$ 고정값 사용. 다양한 노이즈 분포에 대한 적응적 임계값 전략 부재.
2. **계산 비용**: 두 개의 네트워크 쌍을 동시에 학습하는 co-modeling 구조로 인한 메모리/시간 비용 약 2배 증가.
3. **초기 warm-up 의존성**: GMM 피팅의 정확도가 warm-up 품질에 의존. 노이즈 비율이 매우 높은 경우 초기 모델 품질 저하 우려.
4. **데이터셋 제한**: SYSU-MM01, RegDB 두 데이터셋에만 평가. 더 다양한 실세계 조건 검증 필요.
5. **노이즈 유형 가정**: 무작위 노이즈(random noise) 가정. 실제 데이터에서 발생하는 구조적/체계적 노이즈(systematic noise)에 대한 강건성 미검증.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 프레임워크 독립성 (Plugin 방식)

논문 Section 4.1에 명시되어 있듯, DART는 **범용 프레임워크**로 설계되었다:

> *"DART is a general framework which could endow almost all existing VI-ReID methods with robustness against twin noisy labels."*

이를 검증하기 위해 ADP와 AGW 두 가지 서로 다른 백본에 DART를 적용했으며, 두 경우 모두 노이즈 비율 0~50% 전 범위에서 일관된 성능 향상을 보였다 (Figure 6).

### 3.2 일반화 검증 실험 (Section 4.5)

DART + AGW vs. AGW 비교에서 노이즈 비율 0%, 10%, 20%, 30%, 40%, 50% 전 구간에서 DART가 우세하며, 특히 고노이즈 환경에서 성능 격차가 더 커지는 경향을 보인다. 이는 DART의 강건성이 특정 아키텍처에 종속되지 않음을 시사한다.

### 3.3 일반화 메커니즘 분석

**① Soft Identification Loss를 통한 일반화**

$$\mathcal{L}^{sid} = -w_i \log P(y^t_i \mid C^t(F^t(x^t_i)))$$

$w_i$를 통해 학습 과정에서 노이즈 샘플의 기여도를 동적으로 조절함으로써, 모델이 깨끗한 데이터에서 얻은 패턴에 집중할 수 있어 **과적합 방지** 효과를 갖는다.

**② Adaptive Quadruplet Loss를 통한 일반화**

TP-TN, FP-FN, TP-FN, FP-TN의 네 가지 케이스를 통합적으로 처리함으로써, 훈련 데이터의 노이즈 패턴에 무관하게 올바른 임베딩 공간을 구성한다.

**③ Co-modeling을 통한 일반화**

자기 자신의 예측이 아닌 다른 네트워크의 신뢰도를 사용하는 co-teaching 전략은 **확증 편향(confirmation bias)**을 방지하여 더 나은 일반화를 달성한다.

### 3.4 다른 도메인으로의 확장 가능성

논문 Conclusion에서 저자들이 직접 언급:

> *"We plan to explore other scenarios of the twin noisy labels, such as category-level cross-modal retrieve, face recognization, and so on."*

TNL의 구조(NA → NC의 전파)는 다음 영역에서도 동일하게 발생:
- RGB-Depth 기반 ReID
- 텍스트-이미지 크로스모달 검색
- 의료 영상 다중 모달 분석

---

## 4. 향후 연구에 미치는 영향 및 고려 사항

### 4.1 연구에 미치는 영향

**① 새로운 문제 패러다임 제시**

TNL은 단순한 NA 처리를 넘어, 멀티모달 학습에서 **노이즈가 계층적으로 전파**된다는 새로운 인식을 제공한다. 이는 이후 멀티모달 학습 연구에서 데이터 품질 분석의 표준 프레임워크로 활용될 수 있다.

**② Robust VI-ReID 연구의 촉발**

DART 이전에는 VI-ReID에서의 노이즈 연구가 전무하였다. 이 논문은 다음과 같은 후속 연구 방향을 열었다:
- 더 정교한 신뢰도 추정 방법 (GMM을 넘어선 베이지안 추론, 앙상블 기반 등)
- 적응형 임계값 전략
- 준지도 학습(semi-supervised learning) 관점에서의 접근

**③ 크로스모달 매칭에서의 노이즈 연구 확대**

[9](NeurIPS 2021)에서 제시된 NC 개념을 VI-ReID에 결합하여 확장함으로써, 크로스모달 검색 전반에서 노이즈 대응 연구의 방향성을 제시한다.

---

### 4.2 향후 연구 시 고려할 점

#### 기술적 측면

| 고려 사항 | 설명 |
|-----------|------|
| **적응형 임계값** | 고정된 $\eta=0.5$ 대신 데이터셋 특성이나 학습 단계에 따른 동적 임계값 설계 |
| **노이즈 유형 다양화** | 무작위 노이즈 외에 클래스 의존적 노이즈(class-dependent noise), 인스턴스 의존적 노이즈(instance-dependent noise) 처리 |
| **단일 모델 구조** | Co-modeling의 2배 계산 비용을 줄이기 위한 Knowledge Distillation 또는 Self-Distillation 기반 단일 모델 접근 |
| **GMM 한계 극복** | 이분 GMM은 복잡한 실제 노이즈 분포를 과단순화할 수 있음. Flow-based 모델이나 VAE 기반 밀도 추정 활용 가능 |
| **대규모 데이터셋 검증** | LaST, LLCM 등 더 규모 있고 다양한 조건의 VI-ReID 데이터셋에서 검증 필요 |

#### 연구 방법론적 측면

| 고려 사항 | 설명 |
|-----------|------|
| **실제 노이즈 vs. 시뮬레이션 노이즈** | 논문은 무작위로 노이즈를 주입하는 시뮬레이션 방식 사용. 실제 수집 데이터의 노이즈 패턴과의 차이 분석 필요 |
| **노이즈 비율 추정 문제** | 실제 상황에서 노이즈 비율이 사전에 알려지지 않음. 자동 노이즈 비율 추정 전략 필요 |
| **멀티모달 확장** | VI-ReID에서 RGB-D, RGB-T-IR 3모달 이상으로 확장 시 NC의 기하급수적 증가 대응 방안 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

아래는 논문에서 직접 인용된 2020년 이후 관련 연구들을 중심으로 비교 분석한다. (논문에 포함되지 않은 외부 연구는 정확도를 위해 포함하지 않음)

### 5.1 VI-ReID 관련 연구 비교

| 논문 | 발표 | 핵심 방법 | 노이즈 처리 | SYSU-MM01 Rank-1 (0% 노이즈) |
|------|------|-----------|------------|------------------------------|
| AGW [28] | TPAMI 2021 | 범용 ReID 서베이 + 기준 모델 | ✗ | 47.50% |
| DDAG [27] | ECCV 2020 | 동적 이중 어텐션 집계 | ✗ | 54.75% |
| LbA [14] | ICCV 2021 | 크로스모달 대응 정렬 | ✗ | 55.41% |
| MPANet [23] | CVPR 2021 | 크로스모달 뉘앙스 발견 | ✗ | 70.58% |
| ADP [26] | ICCV 2021 | 채널 증강 공동 학습 | ✗ | 69.88% |
| **DART (본 논문)** | **CVPR 2022** | **Co-modeling + Pair Division + Dually Robust Loss** | **✓ (TNL 전체)** | **68.72% (노이즈 무관 경쟁력)** |

### 5.2 Noisy Label 관련 연구 비교

| 논문 | 방법 | 적용 대상 | TNL 처리 |
|------|------|-----------|----------|
| DivideMix [10] (ICLR 2020) | GMM + Semi-supervised | 이미지 분류 | NA만 처리, NC 불가 |
| Co-teaching [6] (NeurIPS 2018) | 두 네트워크 상호 교육 | 이미지 분류 | NA만 처리 |
| PurifyNet [30] (TIFS 2020) | Instance reweighting | 단일모달 ReID | NA만 처리, 멀티모달 불가 |
| [31] Yu et al. (ICCV 2019) | Feature uncertainty | 단일모달 ReID | NA만 처리 |
| NCR [9] (NeurIPS 2021) | Noisy correspondence | 크로스모달 매칭 | NC만 처리(FP만), NA 미처리 |
| [25] Yang et al. (CVPR 2021) | Noise-robust contrastive | 멀티뷰 클러스터링 | FN만 처리 |
| **DART (본 논문)** | Co-modeling + Adaptive Quadruplet | VI-ReID | **NA + NC(TP/TN/FP/FN) 모두 처리** |

### 5.3 핵심 차별점 요약

```
기존 Noisy Label 연구:
  NA only: DivideMix, Co-teaching, PurifyNet, Yu et al.
  NC only (FP only): NCR [9]
  NC only (FN only): [24], [25]

DART의 위치:
  NA + NC(TP+TN+FP+FN) 동시 처리
  → VI-ReID 특화 최초 TNL 해결 방법
```

---

## 참고 자료

**주요 참고 논문 (본 PDF에서 직접 인용된 문헌):**

1. **Yang et al., "Learning with Twin Noisy Labels for Visible-Infrared Person Re-Identification"**, CVPR 2022 — *본 분석의 주 대상 논문*
2. Han et al., "Co-teaching: Robust training of deep neural networks with extremely noisy labels", NeurIPS 2018
3. Li et al., "DivideMix: Learning with noisy labels as semi-supervised learning", arXiv:2002.07394, 2020
4. Ye et al., "Deep learning for person re-identification: A survey and outlook", IEEE TPAMI, 2021
5. Ye et al., "Channel augmented joint learning for visible-infrared recognition", ICCV 2021
6. Wu et al., "RGB-infrared cross-modality person re-identification", ICCV 2017
7. Ye and Yuen, "PurifyNet: A robust person re-identification model with noisy labels", IEEE TIFS, 2020
8. Yu et al., "Robust person re-identification by modelling feature uncertainty", ICCV 2019
9. Huang et al., "Learning with noisy correspondence for cross-modal matching", NeurIPS 2021
10. Yang et al., "Partially view-aligned representation learning with noise-robust contrastive loss", CVPR 2021
11. Ye et al., "Dynamic dual-attentive aggregation learning for visible-infrared person re-identification", ECCV 2020
12. Nguyen et al., "Person recognition system based on a combination of body images from visible light and thermal cameras", Sensors, 2017

> **주의**: 2020년 이후 DART를 인용한 후속 연구(예: LLCM 데이터셋 기반 연구, 더 고도화된 TNL 방법 등)는 본 PDF에 포함되어 있지 않아 해당 내용은 제외하였습니다. 정확도를 위해 제공된 논문 내용에만 기반하여 분석하였습니다.
