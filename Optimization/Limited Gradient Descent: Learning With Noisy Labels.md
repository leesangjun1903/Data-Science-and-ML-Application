# Limited Gradient Descent: Learning With Noisy Labels

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

본 논문의 핵심 주장은 다음과 같습니다:

> **"클린(clean) 검증 세트 없이도, 소수의 역방향(reverse) 샘플을 활용하여 노이즈 레이블 학습의 최적 조기 종료 시점을 추정할 수 있다."**

이는 DNN이 단순/규칙적 패턴(main pattern)을 먼저 학습하고, 이후 노이즈 패턴을 암기하는 특성에 기반합니다.

### 주요 기여

| 기여 항목 | 설명 |
|-----------|------|
| **LGD 알고리즘 제안** | 클린 검증 세트 없이 최적 학습 시점 추정 |
| **역방향 샘플 생성** | 레이블 시프팅으로 역패턴 생성 (기존 연구에서 시도되지 않은 방법) |
| **이론적 필요 조건 증명** | 대칭/비대칭 노이즈에 대한 학습 가능 조건 수학적 증명 |
| **모델 독립성** | SGD 기반 대부분의 DNN 및 손실 함수에 적용 가능 |

---

## 2. 상세 분석

### 2-1. 해결하고자 하는 문제

**문제 정의:**
- DNN은 강력한 fitting 능력으로 노이즈까지 암기(memorization)하여 일반화 성능이 저하됨
- 기존 조기 종료(early stopping) 방법들은 **클린 검증 세트에 의존**하며, 검증 세트의 품질에 성능이 민감하게 반응
- 실제 환경에서 고품질 클린 검증 세트를 얻기 어려움

**주요 동기:** Arpit et al. (2017) [4]의 관찰 — DNN은 노이즈 패턴 암기 전에 단순 패턴을 먼저 학습함

---

### 2-2. 문제 공식화

**오염된 데이터셋 설정:**

$k$-클래스 분류 문제에서:
- 특징 공간: $X \subset \mathbb{R}^d$
- 레이블 공간: $Y = \{0, 1, \cdots, k-1\}$
- 학습 데이터: $\{(x_i, y_i^*)\}_{i=1}^{n}$

노이즈 레이블 $y_i$는 오염 비율 $\eta \in (0, 1)$로 정의:

```math
y_i = \begin{cases} y_{i\neg}^* & \text{if } U(0,1) \in (0, \eta] \\ y_i^* & \text{if } U(0,1) \in (\eta, 1) \end{cases}
```

여기서 $y_{i\neg}^\*$는 $y_i^*$를 제외한 임의의 레이블.

**두 가지 노이즈 종류:**

- **대칭 노이즈(Symmetric):** $P(y_i = y_{i\neg}^* | y_i^*) = \frac{1}{k-1}$ (균등 분포)
- **비대칭 노이즈(Asymmetric):** $y_i = f(y_i^*)$, 고정된 규칙에 의한 레이블 플리핑

---

### 2-3. 제안 방법: Limited Gradient Descent (LGD)

#### 핵심 아이디어: 규칙성(Regularity)과 규모(Scale)

- **규칙성(Regularity):** 샘플들이 특정 규칙을 따르는 패턴 (예: label shifting으로 생성된 역패턴은 주요 패턴과 상호 배타적)
- **규모(Scale):** 패턴의 샘플 수. 경사하강법에서 대규모 규칙 패턴(LSRS)이 소규모(SSRS)보다 먼저 학습됨

#### 레이블 시프팅 연산

역방향 레이블 생성:

$$\hat{y} = \text{MOD}(y + 1, k)$$

#### LGD의 핵심 지표: LoR (Leftover-over-Reverse)

$$\text{LoR} = \frac{Acc_l}{Acc_r}$$

- $Acc_l$: 나머지(leftover) 샘플들의 훈련 정확도 (주요 패턴 근사)
- $Acc_r$: 역방향(reverse) 샘플들의 훈련 정확도 (역패턴 근사)

**LoR이 최대일 때 주요 패턴의 일반화 성능이 최적**으로 추정됨.

#### Algorithm 1: Limited Gradient Descent

```
β·n 개의 샘플을 훈련 세트 S에서 무작위 선택하여 레이블 시프팅 → Sr 생성
나머지 샘플 → Sl, 새 훈련 세트 S' = Sr ∪ Sl

입력: Net, 손실 함수, 훈련 세트 S', LoR ← 0, 반복 횟수 N

for each i ∈ [1, N]:
    SGD로 Net을 S'로 1 step 훈련
    Sl, Sr에 대한 훈련 정확도 Accl, Accr 계산
    if Accl/Accr > LoR:
        LoR ← Accl/Accr
        net_rec ← Net

테스트 세트에 net_rec 적용 → 최종 정확도 계산
```

---

### 2-4. 이론적 필요 조건

#### 대칭 노이즈에 대한 조건 (Theorem 1)

**Lemma 1:** $r$개의 대칭 노이즈 레이블에 레이블 시프팅을 적용하면, $\frac{r}{k-1}$개의 샘플이 참 레이블을 획득:

$$\sum_{j=0}^{k-1} \frac{r_j}{k-1} = \frac{1}{k-1}\sum_{j=0}^{k-1} r_j = \frac{r}{k-1} \tag{1}$$

**Theorem 1:** 역방향 패턴이 주요 패턴과 상호 배타적이 되려면:

$$\eta < \frac{k-1}{k} \tag{2}$$

주요 패턴의 규모가 역방향 패턴보다 크려면:

$$(1-\eta)(1-\beta)n + \eta\beta\frac{1}{k-1}n > (1-\eta)\beta n$$

$$\Rightarrow \beta < \frac{1-\eta}{2-2\eta-\frac{\eta}{k-1}} \tag{3}$$

**Proposition 1:** 주요 패턴의 규모가 역방향 패턴의 $\delta$배 이상이 되려면:

$$(1-\eta)(1-\beta)n + \eta\beta\frac{1}{k-1}n \geq \delta(1-\eta)\beta n \tag{4}$$

$$\Rightarrow \beta \leq \frac{1-\eta}{(1+\delta)(1-\eta)-\frac{\eta}{k-1}}$$

실용적으로 $\delta \geq 9$ 설정 → $\beta \leq \frac{1}{10}$

#### 비대칭 노이즈에 대한 조건 (Theorem 2)

$$\begin{cases} (1-\eta)(1-\beta)n > \eta(1-\beta)n & (5a)\\ (1-\eta)(1-\beta)n > (1-\eta)\beta n & (5b)\\ (1-\eta)(1-\beta)n > \eta\beta n & (5c) \end{cases}$$

$(5a) \Rightarrow \eta < \frac{1}{2}$

$(5b) \Rightarrow \beta < \frac{1}{2}$

**Proposition 2:** 주요 패턴이 역방향 패턴의 $\delta$배 이상이 되려면:

$$\beta \leq \frac{1}{1+\delta}$$

실용적으로 $\delta \geq 9$ → $\beta \leq \frac{1}{10}$

---

### 2-5. 모델 구조

| 구성 요소 | 세부 내용 |
|-----------|-----------|
| **백본 네트워크** | PreAct ResNet-18 (CIFAR-10/100), ResNet-50 pretrained on ImageNet (Clothing-1M) |
| **정규화** | Dropout (rate=0.3) |
| **손실 함수** | CCE, $\mathcal{L}_q$ (q=0.7), Mixup ($\alpha=8$) |
| **최적화** | SGD (lr=0.1, mini-batch=128) |
| **하이퍼파라미터** | $\beta = 0.05$ (유일한 하이퍼파라미터) |

---

### 2-6. 성능 향상

#### CIFAR-10 결과

- LGD는 클린 검증 세트 5,000개를 사용하는 기존 방법과 **동등한 정확도**
- **분산(variance)이 현저히 감소** → 더 높은 강건성(robustness)

#### CIFAR-100 결과 (일부 발췌)

| 방법 | Sym η=0.8 (검증) | Sym η=0.8 (LGD) | Asym η=0.4 (검증) | Asym η=0.4 (LGD) |
|------|-----------------|-----------------|-----------------|-----------------|
| CCE | 17.73% | **20.26%** | 50.45% | **53.11%** |
| $\mathcal{L}_q$ | 25.87% | **28.10%** | 57.22% | **60.43%** |
| Mixup | 27.90% | **29.35%** | 63.96% | **65.57%** |

높은 오염 비율에서 LGD가 기존 방법을 **상회**하는 결과.

#### Clothing-1M 결과

| 방법 | 검증 세트 사용 | 정확도 |
|------|--------------|--------|
| Forward [21] | ✓ | 69.84% |
| Tanaka [6] | ✓ | 72.23% |
| CCE | ✓ | 68.87% |
| LGD+CCE | ✗ | 69.65% |
| LGD+Mixup | ✗ | 73.67% |
| **LGD+ $\mathcal{L}_q$ ** | ✗ | **74.36%** |

검증 세트 없이 **최고 성능(74.36%)** 달성. $\mathcal{L}_q$ 대비 +2.45%p 향상.

---

### 2-7. 한계점

1. **노이즈 유형 및 오염 비율 사전 지식 필요:** Theorem 1, 2를 적용하기 위해 노이즈 유형을 추정해야 함
2. **$\beta$ 하이퍼파라미터 설정:** 실험적으로 $\beta = 0.05$가 최적이나, 데이터셋에 따라 달라질 수 있음
3. **훈련 데이터 감소:** $\beta$-비율의 샘플을 역패턴으로 전용하므로 주요 패턴 학습에 사용 가능한 데이터 감소
4. **LoR 피크의 근사적 추정:** LoR 피크가 실제 최적 시점보다 약간 이른 경향 (Figure 5 참조)
5. **복잡한 실제 노이즈에 대한 검증 부족:** 대칭/비대칭 노이즈 외의 복잡한 현실 노이즈에 대한 추가 분석 필요

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 일반화를 위한 최적 조기 종료

DNN의 학습 동역학(dynamics):

$$\text{훈련 초기: 주요 패턴(LSRS) 학습} \rightarrow \text{일반화 성능 증가}$$
$$\text{훈련 후기: 노이즈 패턴 암기} \rightarrow \text{일반화 성능 감소}$$

LGD는 LoR 지표를 통해 이 **최적 시점을 클린 검증 세트 없이 추정**함으로써:

- 노이즈 암기 전에 학습을 종료 → 과적합(overfitting) 방지
- 결과적으로 테스트 세트에서의 일반화 성능을 최대화

### 3-2. 강건성(Robustness) 향상 메커니즘

검증 세트 크기(100~5,000)에 따른 기존 방법의 정확도 분산과 비교 시:

> "LGD의 분산은 검증 샘플 5,000개를 사용하는 기존 방법보다도 **현저히 작다**." — 논문 Section 6.3

검증 세트의 품질에 독립적이므로, **재현 가능하고 안정적인 일반화 성능**을 기대할 수 있음.

### 3-3. 높은 오염 비율에서의 우수성

$\eta = 0.8$ (대칭), $\eta = 0.4$ (비대칭) 같은 높은 오염 환경에서 LGD가 기존 방법을 상회:
- 오염 비율이 높을수록 검증 세트의 품질이 더 중요해지는데, LGD는 이에 무관하게 동작
- 이는 **높은 오염 환경에서의 실용적 일반화 성능 우위**를 의미

### 3-4. 역방향 패턴의 이론적 역할

역방향 패턴은 주요 패턴과 **상호 배타적(mutually exclusive)**:

$$\hat{y} = \text{MOD}(y + 1, k)$$

이를 통해 모델이 주요 패턴을 학습할 때와 다른 패턴을 학습하기 시작할 때의 신호를 분리하여, **일반화 한계점(generalization boundary)을 탐지**할 수 있음.

---

## 4. 향후 연구에 미치는 영향 및 고려할 점

### 4-1. 향후 연구에 미치는 영향

#### (A) 검증 세트 불필요 패러다임의 확산

LGD는 레이블 노이즈 학습에서 **클린 검증 세트 의존성을 탈피**하는 새로운 방향을 제시. 이는 반지도 학습(semi-supervised learning), 자기 지도 학습(self-supervised learning)과 결합 가능성을 열어줌.

#### (B) 메타러닝 및 조기 종료 연구 자극

LoR과 같은 내부 신호(intrinsic signal)를 활용한 조기 종료 메커니즘은 메타러닝, 신경망 아키텍처 탐색(NAS) 등에서의 적용 가능성 시사.

#### (C) 이론적 기반 강화

Theorem 1, 2는 노이즈 레이블 학습의 **학습 가능 조건(learnability condition)**을 이론적으로 규명. 이는 PAC 학습 이론과의 연결 및 더 엄밀한 generalization bound 연구를 자극할 수 있음.

#### (D) 실제 산업 응용

Clothing-1M 같은 대규모 실세계 데이터셋에서의 성능 검증은 **웹 크롤링 데이터, 의료 이미징, 자율주행 데이터** 등 레이블 노이즈가 불가피한 도메인에서의 활용 가능성을 보여줌.

---

### 4-2. 향후 연구 시 고려할 점

#### (A) 노이즈 유형 자동 추정과의 통합

현재 LGD는 노이즈 유형(대칭/비대칭)을 사전에 알아야 한다는 가정. 노이즈 유형 자동 추정(예: GMM 기반 방법)과 LGD를 결합하여 완전 자동화된 파이프라인 구축 필요.

#### (B) $\beta$ 자동 최적화

실험적으로 $\beta = 0.05$가 최적이나, 데이터셋 특성에 따라 적응적으로 $\beta$를 조정하는 방법 연구 필요:

$$\beta^* = \arg\max_{\beta} \text{GeneralizationPerformance}(\beta)$$

#### (C) 준지도/자기지도 학습과의 결합

LGD의 LoR 지표를 **대조 학습(contrastive learning)**이나 **자기 지도 학습(SSL)**의 학습 진행도 모니터링에 활용하는 연구 가능.

#### (D) 동적 레이블 수정과의 결합

LGD의 최적 종료 시점 탐지 능력과 **동적 레이블 수정(pseudo-labeling)** 방법을 결합하면 반복적 학습 개선 가능.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의:** 아래 비교는 제가 학습한 지식에 기반한 것으로, 2020년 이후 연구에 대한 일부 내용은 제 지식 한계(2024년 초)로 인해 부정확할 수 있습니다. 확실하게 알고 있는 대표 연구들만 제시합니다.

### 5-1. 주요 비교 연구

#### DivideMix (Li et al., 2020) - ICLR 2020

- **방법:** GMM으로 클린/노이즈 샘플 분리 후 반지도 학습 적용
- **LGD와 비교:**
  - DivideMix: 클린/노이즈 분리를 위해 GMM 추정 필요, 더 복잡한 파이프라인
  - LGD: 단순한 레이블 시프팅 기반, 모델 독립적
  - 성능: DivideMix가 CIFAR 벤치마크에서 높은 수치 달성하나, 클린 데이터 일부 활용 가능한 설정

#### CORES² (Cheng et al., 2021)

- **방법:** 샘플 손실의 분포를 모델링하여 신뢰 가능한 샘플 선택
- **LGD와 비교:**
  - 공통점: 클린 검증 세트 불필요
  - 차이점: LGD는 역방향 샘플 생성이라는 독창적 아이디어 사용

#### ELR+ (Liu et al., 2020) - NeurIPS 2020

- **방법:** Early-Learning Regularization으로 초기 학습 단계의 예측을 정규화 항으로 활용
- **LGD와 비교:**
  - ELR+는 정규화 관점, LGD는 조기 종료 관점에서 동일한 문제(DNN의 초기 클린 패턴 학습 특성)를 활용
  - 두 접근법은 상호 보완적으로 결합 가능

### 5-2. LGD의 위치와 의의

```
노이즈 레이블 학습 방법 분류
├── 노이즈 전이 행렬 추정 계열: Forward [2017], PENCIL [2019]
├── 샘플 선택 계열: Co-teaching [2018], DivideMix [2020]
├── 손실 함수 설계 계열: GCE [2018], ELR [2020]
└── 조기 종료/훈련 제어 계열: ← LGD (2019) 위치
    └── 클린 검증 세트 불필요 + 이론적 보장
```

LGD는 **클린 검증 세트 없이 이론적 근거를 갖춘 최적 종료 시점 탐지**라는 측면에서 독창적 위치를 차지하며, 이후 연구들이 클린 데이터 의존성을 줄이는 방향으로 발전하는 데 기여했습니다.

---

## 참고 자료

1. **Yi Sun, Yan Tian, Yiping Xu, Jianxiang Li.** "Limited Gradient Descent: Learning With Noisy Labels." arXiv:1811.08117v4, 2019. (본 논문)

2. **Arpit, D. et al.** "A closer look at memorization in deep networks." ICML 2017. (논문 내 참조 [4])

3. **Zhang, C. et al.** "Understanding deep learning requires rethinking generalization." arXiv:1611.03530, 2016. (논문 내 참조 [3])

4. **Han, B. et al.** "Co-teaching: Robust training of deep neural networks with extremely noisy labels." NeurIPS 2018. (논문 내 참조 [24])

5. **Zhang, Z. and Sabuncu, M.** "Generalized cross entropy loss for training deep neural networks with noisy labels." NeurIPS 2018. (논문 내 참조 [29])

6. **Tanaka, D. et al.** "Joint optimization framework for learning with noisy labels." CVPR 2018. (논문 내 참조 [6])

7. **Li, J. et al.** "DivideMix: Learning with Noisy Labels as Semi-supervised Learning." ICLR 2020.

8. **Liu, S. et al.** "Early-Learning Regularization Prevents Memorization of Noisy Labels." NeurIPS 2020.

9. **Frénay, B. and Verleysen, M.** "Classification in the presence of label noise: a survey." IEEE TNNLS, 2014. (논문 내 참조 [8])

> **정확도 고지:** 2020년 이후 최신 연구 비교 부분(Section 5)은 제 학습 데이터에 기반하며, 일부 세부 수치나 비교는 실제 논문과 다를 수 있습니다. 정확한 비교를 위해서는 각 논문을 직접 확인하시기 바랍니다.
