# Class-imbalanced Domain Adaptation: An Empirical Odyssey

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

기존 비지도 도메인 적응(UDA) 방법들은 **레이블 분포가 도메인 간에 동일하다( $p(y) = q(y)$ )** 는 비현실적 가정 하에 설계되어, 실제 환경에서 빈번히 발생하는 **특징 이동(Feature Shift)**과 **레이블 이동(Label Shift)**이 동시에 존재하는 상황에서 심각한 성능 저하(negative transfer)를 초래한다.

### 주요 기여

| 기여 항목 | 내용 |
|---|---|
| **새로운 문제 설정 (CDA)** | Feature Shift + Label Shift를 동시에 다루는 Class-imbalanced Domain Adaptation 정의 |
| **최초 벤치마크 구축** | 6개 실제 이미지 데이터셋, 22개 교차 도메인 태스크 |
| **포괄적 실증 분석** | 10개 최신 UDA 방법 평가 → 8/10이 negative transfer 유발 확인 |
| **COAL 모델 제안** | 조건부 특징 분포 + 레이블 분포를 동시에 정렬하는 프레임워크 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**기존 UDA의 한계:**

기존 방법들은 다음 가정을 따름:
$$p(y|x) = q(y|x), \quad p(x) \neq q(x)$$

즉, 조건부 레이블 분포는 불변이고 특징 분포만 다르다고 가정. 그러나 현실에서는:
$$p(x|y) \neq q(x|y), \quad p(x) \neq q(x), \quad p(y) \neq q(y)$$

**CDA의 세 가지 주요 도전:**
1. 레이블 이동이 주변 특징 분포 정렬(marginal feature alignment)의 효과를 저해
2. 레이블 이동 존재 시 조건부 특징 분포 $p(x|y)$, $q(x|y)$ 정렬이 어려움
3. 불균형 클래스 분포로 인해 편향된 분류기 학습

---

### 2.2 제안 방법 (COAL: CO-ALignment of Feature and Label Distribution)

#### 2.2.1 이론적 동기

**정리 1 (조건부 특징 정렬 동기):**

Zhao et al. (2019)의 이론에 따라, 타겟 오류는 다음과 같이 바운드됨:

$$\epsilon_T(h) \leq \epsilon_S(h) + d_{\hat{\mathcal{H}}}(\mathcal{D}_S, \mathcal{D}_T) + \min\{\epsilon_S(f_T), \epsilon_T(f_S)\} \tag{1}$$

여기서:
- $\epsilon_S(h)$: 소스 도메인 오류
- $d_{\hat{\mathcal{H}}}(\mathcal{D}_S, \mathcal{D}_T)$: 주변 분포 간 불일치
- $\min\{\epsilon_S(f_T), \epsilon_T(f_S)\}$: 소스와 타겟 최적 레이블 함수 간 거리

주변 분포만 정렬하면 세 번째 항 $\min\{\epsilon_S(f_T), \epsilon_T(f_S)\}$을 최소화할 수 없으므로, **조건부 특징 분포 정렬**이 필수.

**정리 2 (클래스 균형 자기 학습 동기):**

Jensen-Shannon(JS) 거리를 $d_{JS}$로 표기할 때:

$$\epsilon_S(h) + \epsilon_T(h) \geq \frac{1}{2}\left(d_{JS}(p(y), q(y)) - d_{JS}(p(x), q(x))\right)^2 \tag{2}$$

레이블 분포 발산 $d_{JS}(p(y), q(y))$이 클 때, 주변 특징 분포만 정렬하면 타겟 오류 $\epsilon_T(h)$가 증가할 수 있음 → **레이블 분포 정렬 필요성** 입증.

---

#### 2.2.2 프로토타입 기반 조건부 정렬 (Feature Shift 처리)

**유사도 기반 분류기:**

특징 추출기 $F$와 분류기 $C$로 구성. 분류기 $C$는 가중치 행렬 $\mathbf{W} = [\mathbf{w}_1, \mathbf{w}_2, ..., \mathbf{w}_c] \in \mathbb{R}^{d \times c}$와 온도 파라미터 $T$로 구성.

입력 특징 $F(x)$와 $i$번째 프로토타입 $\mathbf{w}_i$ 간의 유사도:

$$s_i = \frac{F(x)\mathbf{w}_i}{T\|F(x)\|}$$

클래스 $i$에 속할 확률:

$$h_i(x) = \sigma\left(\frac{F(x)\mathbf{w}_i}{T\|F(x)\|}\right)$$

소스 도메인에 대한 프로토타입 기반 분류 손실:

$$\mathcal{L}_{SC} = \mathbb{E}_{(x,y) \in \mathcal{D}_S} \mathcal{L}_{ce}(h(x), y) \tag{3}$$

**미니맥스 엔트로피 기반 조건부 정렬:**

타겟 도메인 샘플 $x^t \in \mathcal{D}_T$에 대한 평균 엔트로피:

$$\mathcal{L}_H = \mathbb{E}_{x \in \mathcal{D}_T} H(x) = -\mathbb{E}_{x \in \mathcal{D}_T} \sum_{i=1}^{c} h_i(x) \log h_i(x) \tag{4}$$

- **분류기 $C$**: $\mathcal{L}_H$를 **최대화** → 소스 프로토타입을 타겟 샘플 방향으로 이동
- **특징 추출기 $F$**: $\mathcal{L}_H$를 **최소화** → 타겟 샘플을 가까운 프로토타입 주변에 클러스터링

이를 위해 $C$와 $F$ 사이에 **Gradient Reverse Layer**를 삽입.

---

#### 2.2.3 클래스 균형 자기 학습 (Label Shift 처리)

**자기 학습(Self-training):**

각 카테고리별로 상위 $k\%$ 고신뢰도 타겟 샘플을 선택하여 의사 레이블(pseudo-label) $\hat{y}$ 부여.

의사 레이블 타겟 셋: $\hat{\mathcal{D}}\_T = \{(x_i^t, \hat{y}\_i^t, m_i)_{i=1}^{N_t}\}$

선택 마스크:

```math
m = \begin{cases} 1 & \text{if } h(x) \text{ is among top-}k\% \text{ within pseudo-class} \\ 0 & \text{otherwise} \end{cases}
```

전체 분류 손실:

$$\mathcal{L}_{ST} = \mathcal{L}_{SC} + \mathbb{E}_{(x, \hat{y}, m) \in \hat{\mathcal{D}}_T} \mathcal{L}_{ce}(h(x), \hat{y}) \cdot m \tag{5}$$

$k$는 에포크마다 $k_{step}$씩 증가: $k_0 = 5$, $k_{step} = 5$, $k_{max} = 30$.

**균형 소스 샘플러(Balanced Source Sampler):**

각 미니배치에서 클래스별 동일한 수의 소스 샘플을 추출하여 편향 방지.

---

#### 2.2.4 통합 학습 목표

$$\hat{C} = \arg\min_C \mathcal{L}_{ST} - \alpha \mathcal{L}_H, \quad \hat{F} = \arg\min_F \mathcal{L}_{ST} + \alpha \mathcal{L}_H \tag{6}$$

여기서 $\alpha$는 분류 학습과 특징 분포 정렬 간 균형 조절 파라미터.

---

### 2.3 모델 구조

```
[Step A: 의사 레이블 생성]
x_T → F → C → Softmax → argmax/top-k → ŷ_T, mask m

[Step B: 적응적 학습]
x_S, x_T → F → C → Softmax → P
                              ↓
              L_CE(P, y_S, ŷ_T, m) [자기 학습]
              -H(P) → Gradient Reverse → [미니맥스 엔트로피 정렬]
```

**두 단계 반복 학습:**
1. **Step A**: 타겟 샘플 전달 → 의사 레이블 및 마스크 생성
2. **Step B**: 소스 레이블 + 의사 레이블로 분류기 학습 + 미니맥스 엔트로피로 조건부 특징 정렬

**백본 네트워크:**
- Digits: MCD 제안 아키텍처
- Office-Home, DomainNet: ResNet-50 (마지막 FC 레이어를 N-way 분류기로 교체)

---

### 2.4 성능 향상

**평가 지표:** 클래스별 평균 정확도(per-class mean accuracy)
$$S = \frac{1}{c} \sum_{i=1}^{c} S_i, \quad S_i = \frac{n(i,i)}{n_i}$$

**Digits 데이터셋 결과:**

| 방법 | USPS→MNIST | MNIST→USPS | SVHN→MNIST | SYN→MNIST | AVG |
|---|---|---|---|---|---|
| Source Only | 75.31 | 87.92 | 50.25 | 85.74 | 74.81 |
| Best Baseline (DANN) | 77.28 | 91.88 | 57.16 | 77.60 | 75.98 |
| **COAL (Ours)** | **88.12** | **93.04** | **65.67** | **90.60** | **84.33** |

→ 최고 베이스라인 대비 **+8.35%** 향상

**Office-Home 결과:** COAL 58.87% (최고 베이스라인 DANN 56.85% 대비 +2.02%)

**DomainNet 결과:** COAL 75.89% (최고 베이스라인 DANN 74.46% 대비 +1.43%)

**레이블 이동 강도별 안정성:**
- MCD: 91.45%(0%) → 77.18%(100%) (급격한 성능 저하)
- COAL: 93.42%(0%) → 88.12%(100%) (**안정적**)

---

### 2.5 한계

1. **의사 레이블 품질 의존성**: 초기 모델 품질이 낮으면 의사 레이블 오류가 누적되어 학습이 불안정해질 수 있음
2. **레이블 이동 추정의 불확실성**: $q(\hat{y})$로 $q(y)$를 근사하는 방식이 특징 정렬이 불완전할 때 부정확할 수 있음
3. **하이퍼파라미터 민감성**: $k_0$, $k_{step}$, $k_{max}$, $\alpha$ 등의 튜닝 필요
4. **확장성**: 클래스 수가 매우 많거나 타겟 도메인 샘플이 극히 적을 때의 동작 미검증
5. **RSUT 프로토콜의 제약**: 소스와 타겟의 레이블 분포가 역전되는 특정 구조만 다룸 → 더 다양한 레이블 이동 패턴 검증 필요

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화에 기여하는 핵심 메커니즘

**① 조건부 특징 정렬의 일반화 효과**

주변 분포 정렬은 레이블 이동 시 서로 다른 클래스의 샘플을 잘못 정렬시킬 수 있음(negative transfer). 반면 조건부 분포 $p(x|y)$와 $q(x|y)$를 정렬하면:
- 클래스별로 독립적인 정렬이 이루어져 **클래스 경계가 명확히 유지**
- 이론적 바운드 식 (1)의 세 번째 항 $\min\{\epsilon_S(f_T), \epsilon_T(f_S)\}$ 최소화에 직접 기여
- t-SNE 시각화에서 COAL이 클래스별로 소스-타겟 특징을 잘 정렬함을 확인

**② 자기 학습의 적응적 레이블 분포 추정**

에포크가 진행될수록 더 정확한 의사 레이블 → 더 정확한 타겟 레이블 분포 추정 → 더 나은 결정 경계의 **양성 피드백 루프**:

$$q(\hat{y}) \approx q(y) \text{ (수렴 시)}$$

이 반복적 정제(iterative refinement)는 타겟 도메인에 대한 일반화 성능을 점진적으로 향상시킴.

**③ 균형 샘플링의 일반화 기여**

Table 4의 실험 결과: 25개 태스크 중 20개에서 균형 샘플러 적용 시 성능 향상. 이는 소스 편향이 제거되어 마이너리티 클래스에 대한 일반화가 개선됨을 의미.

**④ 클래스별 신뢰도 기반 선택**

전체 상위 $k\%$가 아닌 **클래스별** 상위 $k\%$를 선택함으로써 쉬운 클래스가 의사 레이블 집합을 지배하는 현상 방지 → 모든 클래스에 걸친 균등한 일반화 가능성 확보.

### 3.2 일반화 성능의 이론적 보장

식 (2)에 따르면:
$$\epsilon_S(h) + \epsilon_T(h) \geq \frac{1}{2}\left(d_{JS}(p(y), q(y)) - d_{JS}(p(x), q(x))\right)^2$$

COAL은 $d_{JS}(p(y), q(y))$를 자기 학습으로 줄이고, $d_{JS}(p(x), q(x))$는 조건부 정렬로 관리함으로써, 이론적 하한을 줄여 타겟 오류를 감소시킴.

### 3.3 일반화의 한계와 불확실성

- 의사 레이블의 초기 품질에 일반화가 민감하게 의존
- 레이블 이동의 정도가 극단적일 경우 ($d_{JS}(p(y), q(y))$가 매우 클 경우) 자기 학습 초기 단계에서 오류 누적 위험

---

## 4. 연구에 미치는 영향과 앞으로의 고려 사항

### 4.1 향후 연구에 미치는 영향

**① 새로운 문제 패러다임 확립**

CDA는 "특징 이동만 고려"하는 기존 UDA의 비현실성을 정면으로 지적하며, **레이블 이동과 특징 이동의 동시 처리**라는 새로운 연구 방향을 제시. 이후 연구들이 CDA를 기본 가정으로 삼는 계기가 됨.

**② 벤치마크 기여**

6개 데이터셋, 22개 태스크로 구성된 CDA 벤치마크는 후속 연구의 표준 평가 환경으로 활용될 수 있는 토대를 마련.

**③ 방법론적 기여**

- 프로토타입 기반 조건부 정렬 + 자기 학습의 결합이 CDA의 실용적 해법임을 보임
- 균형 샘플러가 단순하지만 강력한 개선 도구임을 실증

**④ 이론적 기여**

식 (1), (2)를 통해 레이블 이동이 존재할 때 주변 분포 정렬이 오히려 해로울 수 있음을 이론적으로 근거를 가지고 주장 → 향후 이론 연구의 방향을 제시.

---

### 4.2 2020년 이후 관련 최신 연구 비교 분석

> **주의**: 아래는 제가 학습한 데이터 내에서 확인 가능한 관련 연구들이며, 논문 PDF에 직접 인용된 연구가 아닌 후속 연구들입니다. 개별 논문의 정확한 수치는 원문 확인을 권장합니다.

#### CDA/Label Shift 관련 후속 연구 동향

**① 레이블 이동 추정의 정교화 방향**

- **RLLS (Azizzadenesheli et al., 2019)**: 레이블 이동 하에서 정규화 학습 → COAL 이전 방식으로, 특징 이동은 다루지 않음
- **Lipton et al. BBSE (2018)**: 블랙박스 예측기로 레이블 이동 감지 및 보정 → 특징 이동 부재 가정으로 CDA에 불충분

**② 조건부 도메인 적응 강화**

COAL 이후 조건부 정렬을 중심으로 한 연구들이 활발해짐:

- **JUMBOT (Fatras et al., 2021)**: Unbalanced Optimal Transport를 활용해 샘플 단위의 조건부 정렬을 보다 정밀하게 수행. COAL의 프로토타입 기반 정렬보다 더 정밀한 mini-batch 수준의 최적 수송 정렬.

- **SDAT (Rangwani et al., 2022)**: 스무딩된 도메인 적대적 훈련으로 조건부 정렬의 안정성 향상. 클래스 불균형에 대한 강건성 개선.

**③ 자기 학습의 발전**

- **NoisyStudent / UDA (Xie et al., 2020)**: 자기 학습과 데이터 증강을 결합하여 의사 레이블의 품질 향상 → COAL의 자기 학습 구성요소를 개선할 수 있는 방향 제시

- **SHOT (Liang et al., 2021, ICML)**: 소스 모델을 고정하고 타겟에서만 정보 최대화 기반 자기 학습 수행. 레이블 이동에 대한 명시적 처리 없이도 어느 정도 강건성을 보임.

**④ Long-tail + Domain Adaptation 교차 연구**

- **GVRT (Min et al., 2022)**: Long-tail 분포에서의 도메인 적응을 다룸. COAL의 RSUT 프로토콜과 유사한 방향의 연구.

**COAL과 비교 요약:**

| 특성 | COAL (2020) | 후속 연구들 |
|---|---|---|
| **레이블 이동 처리** | 자기 학습 기반 추정 | 더 정밀한 추정 방법 (OT, 통계적 추정) |
| **조건부 정렬** | 프로토타입 + 미니맥스 엔트로피 | 더 세밀한 샘플별 정렬 |
| **이론적 보장** | Ben-David 바운드 기반 | 더 tight한 바운드 제시 |
| **실용성** | 단순하고 효과적인 파이프라인 | 더 복잡하지만 더 높은 성능 |

---

### 4.3 앞으로 연구 시 고려할 점

**① 레이블 이동 감지 및 정량화**

레이블 이동의 존재 여부와 크기를 적응적으로 감지하는 메커니즘이 필요. 이를 통해 정렬 강도를 동적으로 조절 가능:
$$\alpha \propto d_{JS}(\hat{p}(y), \hat{q}(y))$$

**② 의사 레이블의 불확실성 정량화**

베이지안 딥러닝 또는 앙상블 기반으로 의사 레이블의 신뢰도를 더 정밀하게 추정하여 오류 누적 방지.

**③ 더 다양한 레이블 이동 패턴**

RSUT 외에도 부분적 레이블 이동(일부 클래스만 비율 차이), 점진적 이동 등 다양한 현실 시나리오에 대한 검증 필요.

**④ 대규모 모델과의 통합**

Vision Transformer(ViT), CLIP 등 대규모 사전 훈련 모델에서 COAL 아이디어(조건부 정렬 + 레이블 균형)를 어떻게 효율적으로 적용할지 연구 필요.

**⑤ 멀티소스 CDA 확장**

단일 소스-타겟 설정을 넘어, 각기 다른 레이블 분포를 가진 여러 소스 도메인에서의 CDA 처리.

**⑥ 공정성(Fairness)과의 연관성**

클래스 불균형 도메인 적응은 AI 공정성 문제와 밀접하게 연관됨. 마이너리티 클래스에 대한 공정한 성능 보장을 위한 연구 방향.

**⑦ 자동 하이퍼파라미터 최적화**

$k_0$, $k_{step}$, $k_{max}$, $\alpha$ 등의 하이퍼파라미터를 메타 학습(meta-learning) 또는 AutoML로 자동화하여 실용성 향상.

---

## 참고 자료

**주 논문:**
- Tan, S., Peng, X., & Saenko, K. (2020). *Class-imbalanced Domain Adaptation: An Empirical Odyssey*. arXiv:1910.10320v2.

**논문 내 인용된 주요 참고 문헌:**
- Zhao, H., et al. (2019). *On Learning Invariant Representations for Domain Adaptation*. ICML 2019.
- Saito, K., et al. (2019). *Semi-supervised Domain Adaptation via Minimax Entropy*. ICCV 2019.
- Zou, Y., et al. (2018). *Unsupervised Domain Adaptation for Semantic Segmentation via Class-balanced Self-training*. ECCV 2018.
- Long, M., et al. (2015). *Learning Transferable Features with Deep Adaptation Networks (DAN)*. ICML 2015.
- Ganin, Y., & Lempitsky, V. (2015). *Unsupervised Domain Adaptation by Backpropagation (DANN)*. ICML 2015.
- Saito, K., et al. (2018). *Maximum Classifier Discrepancy for Unsupervised Domain Adaptation (MCD)*. CVPR 2018.
- Lipton, Z.C., et al. (2018). *Detecting and Correcting for Label Shift with Black Box Predictors (BBSE)*. ICML 2018.
- He, K., et al. (2016). *Deep Residual Learning for Image Recognition (ResNet)*. CVPR 2016.
- Venkateswara, H., et al. (2017). *Deep Hashing Network for Unsupervised Domain Adaptation (Office-Home)*. CVPR 2017.
- Peng, X., et al. (2019). *Moment Matching for Multi-Source Domain Adaptation (DomainNet)*. ICCV 2019.

**후속 연구 (논문 외 참조):**
- Liang, J., et al. (2021). *Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation (SHOT)*. ICML 2021.
- Rangwani, H., et al. (2022). *A Closer Look at Smoothness in Domain Adversarial Training (SDAT)*. ICML 2022.

> **정확도 고지**: 후속 연구 비교 분석 부분(4.2절)의 일부 연구들은 제 학습 데이터 기반이며, 특정 논문들의 정확한 수치나 방법론 세부 사항은 원문을 통해 재확인하시기를 권장합니다.
