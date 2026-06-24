# Re-weighted Adversarial Adaptation Network (RAAN)

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

RAAN(Re-weighted Adversarial Adaptation Network)은 **도메인 간 분포 차이가 클 때**(disparate domain discrepancy) 기존 방법들이 실패하는 문제를 해결하기 위해 제안된 비지도 도메인 적응(UDA) 방법입니다.

기존 MMD 및 f-divergence(JS divergence 포함) 기반 방법들은 **공통 지지(common support)** 가정을 필요로 하지만, 현실의 도메인 차이는 이 가정을 충족하기 어렵습니다. RAAN은 이를 극복하기 위해 **최적 운송(Optimal Transport) 기반 Earth-Mover(EM) 거리**를 adversarial 학습에 통합합니다.

### 주요 기여 2가지

1. **OT 기반 EM 거리를 end-to-end adversarial 학습에 통합**: 공통 지지 가정 없이 큰 도메인 차이에서도 특징 분포를 효과적으로 매칭
2. **레이블 분포 재가중(Re-weighting) 기법**: 소스 도메인의 레이블 분포를 목표 도메인 레이블 분포에 근사시키는 밀도비 벡터 $\boldsymbol{\beta}$를 추정하여 분류기 적응을 동시에 수행

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

UDA에서 두 가지 핵심 문제가 존재합니다:

| 문제 | 기존 방법의 한계 | RAAN의 접근 |
|------|----------------|-------------|
| 특징 분포 매칭 | MMD/JS divergence: common support 필요 | OT 기반 EM distance 사용 |
| 분류기 적응 | 타겟 레이블 없이 분류기 이전 어려움 | 레이블 분포 재가중으로 해결 |

특히 **도메인 차이가 매우 클 때** (SVHN→MNIST, RGB→HHA 등) 기존 방법들은 성능이 크게 저하됩니다.

---

### 2.2 제안하는 방법 (수식 포함)

#### (A) 최적 운송 기반 EM 거리

소스/타겟 도메인의 경험적 분포를 다음과 같이 정의합니다:

$$\boldsymbol{\mu}^s = \sum_{i}^{n_s} p_i^s \delta_{\mathcal{T}_s(\boldsymbol{x}_i^s)}, \quad \boldsymbol{\mu}^t = \sum_{j}^{n_t} p_j^t \delta_{\mathcal{T}_t(\boldsymbol{x}_j^t)} $$

수송 계획(transportation plan) $\gamma$의 집합 $\boldsymbol{B}$:

```math
\boldsymbol{B} = \left\{ \gamma \in (\mathbb{R}^+)^{n_s \times n_t} \mid \gamma \mathbf{1}_{n_t} = \boldsymbol{\mu}^s, \gamma^T \mathbf{1}_{n_s} = \boldsymbol{\mu}^t \right\}
```

총 수송 비용:

$$J(\boldsymbol{\mu}^s, \boldsymbol{\mu}^t) = \langle \gamma, \boldsymbol{C} \rangle_F, \quad \gamma \in \boldsymbol{B} $$

Earth-Mover 거리 (Wasserstein distance):

$$W(\boldsymbol{\mu}^s, \boldsymbol{\mu}^t) = \min_{\gamma \in \boldsymbol{B}} J(\boldsymbol{\mu}^s, \boldsymbol{\mu}^t) $$

이를 **쌍대 형식(dual formulation)**으로 adversarial 학습에 통합:

$$W(\boldsymbol{\mu}^s, \boldsymbol{\mu}^t) = \max_{\mathcal{D}, \hat{\mathcal{D}}} \mathcal{L}_{adv}$$

$$\mathcal{L}_{adv} = \mathbb{E}_{\boldsymbol{x}^s \sim P^s(\boldsymbol{X}^s)} \mathcal{D}(\mathcal{T}_s(\boldsymbol{x}^s)) + \mathbb{E}_{\boldsymbol{x}^t \sim P^t(\boldsymbol{X}^t)} \hat{\mathcal{D}}(\mathcal{T}_t(\boldsymbol{x}^t)) $$

$$\text{s.t.} \quad \mathcal{D}(\mathcal{T}_s(\boldsymbol{x}^s)) + \hat{\mathcal{D}}(\mathcal{T}_t(\boldsymbol{x}^t)) \leq c(\mathcal{T}_s(\boldsymbol{x}^s), \mathcal{T}_t(\boldsymbol{x}^t)) $$

거리 함수로 $c(\mathcal{T}_s(\boldsymbol{x}^s), \mathcal{T}_t(\boldsymbol{x}^t)) = \|\mathcal{T}_s(\boldsymbol{x}^s) - \mathcal{T}_t(\boldsymbol{x}^t)\|$를 사용하면, 제약 조건 (10)은 $\mathcal{D}$가 **1-Lipschitz 함수**임을 요구하는 것과 동치가 됩니다.

이를 minimax 형태로 재구성:

$$\min_{\mathcal{T}_t} W(\boldsymbol{\mu}^s, \boldsymbol{\mu}^t) = \min_{\mathcal{T}_t} \max_{\mathcal{D}} \mathcal{L}_{adv}$$

$$\mathcal{L}_{adv} = \sum_{(\boldsymbol{x}^s, y^s) \sim P^s(\boldsymbol{X}, Y^s)} \mathcal{D}(\mathcal{T}_s(\boldsymbol{x}^s)) P^s(\mathcal{T}_s(\boldsymbol{x}^s)|y^s) P^s(y^s) - \mathbb{E}_{\boldsymbol{x}^t \sim P^t(\boldsymbol{X}^t)} \mathcal{D}(\mathcal{T}_t(\boldsymbol{x}^t))$$

$$\text{s.t.} \quad \|\nabla_{\mathcal{T}_t(\boldsymbol{x}^t)} \mathcal{D}(\mathcal{T}_t(\boldsymbol{x}^t))\|_2 \leq 1, \quad \|\nabla_{\mathcal{T}_s(\boldsymbol{x}^s)} \mathcal{D}(\mathcal{T}_s(\boldsymbol{x}^s))\|_2 \leq 1 $$

---

#### (B) 레이블 분포 재가중 (핵심 기여)

베이즈 규칙에 의해:

$$P(\mathcal{T}(\boldsymbol{X})|Y) P(Y) \propto P(Y|\mathcal{T}(\boldsymbol{X})) $$

타겟 레이블 정보 없이 직접 $P^t(Y^t)$를 알 수 없으므로, **소스 도메인 레이블 분포를 재가중**하여 $P^{Re}(Y^s) \approx P^t(Y^t)$를 추정합니다.

밀도비 벡터:

$$\beta(y^s) = \frac{P^{Re}(Y^s = y^s)}{P^s(Y^s = y^s)} $$

$P^{Re}(Y^s)$는 $\boldsymbol{\alpha} \in \mathbb{R}^{n_{cls}}$를 softmax로 변환하여 얻으며:

$$\sum_{i=1}^{n_{cls}} P^{Re}(Y^s = y_i) = 1 $$

재가중된 adversarial 손실 $\mathcal{L}^{Re}_{adv}$:

$$\min_{\mathcal{T}_t} \max_{\mathcal{D}, \boldsymbol{\beta}} \mathcal{L}^{Re}_{adv}$$

$$\mathcal{L}^{Re}_{adv} = \mathbb{E}_{(\boldsymbol{x}^s, y^s) \sim P^s(\boldsymbol{X}^s, Y^s)} \beta(y^s) \mathcal{D}(\mathcal{T}_s(\boldsymbol{x}^s)) - \mathbb{E}_{\boldsymbol{x}^t \sim P^t(\boldsymbol{X}^t)} \mathcal{D}(\mathcal{T}_t(\boldsymbol{x}^t)) $$

---

#### (C) 경험적 손실 및 정규화

경험적 손실:

$$\mathcal{L}^{Re}_{adv} = \frac{1}{n_s} \sum_{i=1}^{n_s} \mathcal{D}(\beta(y_i^s) \mathcal{T}_s(\boldsymbol{x}_i^s)) - \frac{1}{n_t} \sum_{j=1}^{n_t} \mathcal{D}(\mathcal{T}_t(\boldsymbol{x}_j^t)) $$

Gradient Penalty (1-Lipschitz 강제):

$$\mathcal{L}_{gp} = \|\nabla_{\hat{\mathcal{T}}(\hat{\boldsymbol{x}})} \mathcal{L}^{Re}_{adv} - 1\|_2 $$

최종 목적 함수:

$$\min_{\mathcal{T}_t, \mathcal{D}, \boldsymbol{\beta}} -\mathcal{L}^{Re}_{adv} + \lambda_{gp} \mathcal{L}_{gp} + \lambda_{reg} \|\boldsymbol{\beta}\|_2 $$

$$\min_{\mathcal{T}_t} -\frac{1}{n_t} \sum_{j=1}^{n_t} \mathcal{D}(\mathcal{T}_t(\boldsymbol{x}_j^t)) $$

$$\min_{\mathcal{T}_s, CLS} \frac{1}{n_s} \sum_{i=1}^{n_s} \mathcal{L}_{CE}(CLS(\mathcal{T}_s(\boldsymbol{x}_i^s)), y_i^s) $$

---

### 2.3 모델 구조

```
[Source Domain]                    [Target Domain]
  x_s, y_s                              x_t
     │                                   │
  T_s (DCNN)                         T_t (DCNN)
     │                                   │
   CLS ──→ L_CE                     ┌────┴────┐
                                    │         │
              ←── β (ratio vector) ──┤    D    │──→ L^Re_adv
                   (softmax(α))     │         │
                                    └─────────┘
```

**구성 요소:**
- $\mathcal{T}_s$: 소스 도메인 DCNN (특징 추출기)
- $\mathcal{T}_t$: 타겟 도메인 DCNN (특징 추출기)
- $CLS$: 분류기 (소스 도메인에서 훈련)
- $\mathcal{D}$: 도메인 판별기 (3개 FC 레이어, ReLU 활성화)
- $\boldsymbol{\beta}$: 레이블 분포 밀도비 벡터

---

### 2.4 성능 향상

#### 손글씨 숫자 데이터셋 결과 (Table 1)

| 방법 | MNIST→USPS | USPS→MNIST | SVHN→MNIST |
|------|-----------|-----------|-----------|
| Source Only | 0.725 | 0.612 | 0.593 |
| DANN | 0.771 | 0.730 | 0.739 |
| ADDA | 0.894 | 0.901 | 0.760 |
| Co-GAN | 0.912 | 0.891 | No Converge |
| **RAAN(-)** | 0.883 | 0.915 | 0.807 |
| **RAAN(+)** | **0.890** | **0.921** | **0.892** |

SVHN→MNIST에서 ADDA 대비 **+13.2%** 향상

#### 교차 모달리티 데이터셋 (RGB→HHA, Table 4)

| 방법 | 전체 정확도 |
|------|-----------|
| ADDA | 0.276 |
| RAAN(-) | 0.308 |
| **RAAN(+)** | **0.343** |

#### A-Distance 비교 (Table 3, 낮을수록 좋음)

| 방법 | A-Distance |
|------|-----------|
| Source Only | 1.673 |
| ADDA | 1.548 |
| RAAN(-) | 1.526 |
| **RAAN(+)** | **1.506** |

---

### 2.5 한계점

1. **계산 복잡도**: 두 개의 DCNN ($\mathcal{T}_s$, $\mathcal{T}_t$)를 별도로 학습하고 gradient penalty 계산이 필요하여 메모리 및 연산 비용 증가

2. **레이블 분포 추정의 부정확성**: Figure 3에서 확인되듯, 학습된 $\boldsymbol{\beta}$가 일부 클래스에서 실제 비율과 다름 (상대적 추세는 유사하나 정확한 값은 다를 수 있음)

3. **클래스 불균형 처리의 한계**: Table 4에서 'bathtub'(19개), 'toilet'(17개) 등 극소수 클래스에서 0% 인식률 기록

4. **하이퍼파라미터 민감성**: $\lambda_{gp}$, $\lambda_{reg}$ 등 다수의 하이퍼파라미터 조정 필요

5. **단일 소스-단일 타겟 가정**: 다중 소스/타겟 도메인으로의 확장 미고려

6. **조건부 분포 매칭 한계**: 수식 (12)의 근사는 $P^s(\mathcal{T}_s(\boldsymbol{X}^s)|Y^s) = P^t(\mathcal{T}_t(\boldsymbol{X}^t)|Y^t)$ 가정에 의존

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 OT 기반 EM 거리의 일반화 효과

EM 거리는 분포 간 **공통 지지 없이도** 두 분포 간 거리를 측정할 수 있어, 분포가 크게 달라도 안정적인 그래디언트를 제공합니다. 이는 도메인 차이가 큰 실세계 시나리오에서의 일반화를 직접적으로 지원합니다.

$$W(\mu^s, \mu^t) \text{는 } \mu^s \text{와 } \mu^t \text{의 support가 겹치지 않아도 유한한 값을 가짐}$$

### 3.2 레이블 분포 매칭을 통한 일반화

논문에서 다음 관계를 이론적으로 제시합니다:

만약 $P^s(\mathcal{T}_s(\boldsymbol{X}^s)|Y^s) = P^t(\mathcal{T}_t(\boldsymbol{X}^t)|Y^t)$이면:

$$P^{Re}(Y^s) = P^t(Y^t) \Rightarrow P^s(\mathcal{T}_s(\boldsymbol{X}^s)) = P^t(\mathcal{T}_t(\boldsymbol{X}^t))$$

이는 **레이블 분포 매칭이 주변 특징 분포 매칭을 자동으로 지원**함을 의미합니다. 분류기 적응 측면에서, 베이즈 규칙 (12)에 의해 $P(Y|\mathcal{T}(\boldsymbol{X}))$를 직접 매칭하는 대신 $P(\mathcal{T}(\boldsymbol{X})|Y)P(Y)$를 매칭함으로써 더 용이하게 classifier를 적응시킬 수 있습니다.

### 3.3 재가중 기법의 불균형 데이터 처리 효과

$\boldsymbol{\beta}$ 벡터는 소수 클래스 샘플에 더 큰 가중치를 부여하여:
- 클래스 불균형 문제를 완화
- 타겟 도메인의 미지 클래스 분포에 대한 적응력 향상
- Table 4에서 소수 클래스('counter', 'door' 등)에서 RAAN(+)이 ADDA보다 월등히 우수한 성능

### 3.4 A-Distance 감소와 일반화 관계

$$d_{\mathcal{A}} = 2(1 - 2\theta)$$

A-Distance가 낮을수록 도메인 판별기가 두 도메인을 구분하기 어렵다는 것을 의미하며, 이는 **더 도메인 불변적(domain-invariant)인 특징**이 학습되었음을 뜻합니다. RAAN(+)의 A-Distance(1.506)는 ADDA(1.548) 대비 유의미하게 낮아 더 강한 일반화 능력을 보입니다.

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 연구에 미치는 영향

**① OT의 UDA 적용 확산 촉진**
RAAN은 OT 기반 EM 거리를 end-to-end DNN 학습에 통합한 선구적 연구 중 하나로, 이후 많은 연구에서 Wasserstein distance를 UDA에 적용하는 방향을 촉진했습니다.

**② 레이블 분포 편향 문제 인식 제고**
소스-타겟 도메인 간 레이블 분포 불일치가 UDA 성능에 미치는 영향을 명시적으로 모델링한 점은, 이후 클래스 수준의 정렬(class-level alignment) 연구에 중요한 동기를 제공했습니다.

**③ Wasserstein GAN과 DA의 연결**
Wasserstein GAN의 아이디어를 DA에 접목함으로써, 생성 모델과 도메인 적응 연구의 교차점을 넓혔습니다.

**④ 불균형 데이터셋에서의 DA 연구 방향 제시**
실세계 데이터의 클래스 불균형을 명시적으로 다루는 DA 방법론의 필요성을 실증했습니다.

---

### 4.2 2020년 이후 관련 최신 연구 비교 분석

아래는 RAAN의 문제의식을 계승·발전시킨 주요 연구들입니다. 단, 논문 원문에서 직접 인용된 연구가 아니므로, 해당 연구들의 세부 수치는 일반적으로 알려진 내용을 기반으로 서술하며 정확도에 일부 불확실성이 있음을 밝힙니다.

#### (A) Minimum Class Confusion (MCC, 2020)
- **Jin et al., "Minimum Class Confusion for Versatile Domain Adaptation," ECCV 2020**
- RAAN과 유사하게 클래스 수준의 정렬을 추구하지만, 타겟 도메인의 예측 출력에서 클래스 간 혼동을 최소화하는 방식으로 분류기를 적응시킴
- RAAN의 레이블 분포 매칭 아이디어를 클래스 예측 레벨에서 더 정교하게 구현

#### (B) Transferable Curriculum Learning (TCL, 2021)
- **Shu et al., "Transferable Curriculum for Weakly-Supervised Domain Adaptation," AAAI 2019 / 후속 연구들**
- 재가중 기법을 커리큘럼 학습과 결합하여 신뢰도 높은 샘플부터 적응

#### (C) Conditional Adversarial Domain Adaptation (CDAN, 2018-2020 후속)
- **Long et al., "Conditional Adversarial Domain Adaptation," NeurIPS 2018**
- 특징과 분류기 예측을 결합한 조건부 적대적 학습으로 RAAN의 한계인 조건부 분포 매칭 문제를 개선

#### (D) OT 기반 후속 연구들
- **Fatras et al., "Minibatch Optimal Transport Distances; Analysis and Applications," 2021**
- Mini-batch OT의 통계적 특성을 분석하여 RAAN에서 사용된 배치 단위 EM 거리 추정의 이론적 기반 강화

| 항목 | RAAN (2018) | CDAN (2018) | MCC (2020) |
|------|-------------|-------------|------------|
| 분포 매칭 방식 | OT/EM distance | 조건부 adversarial | 클래스 혼동 최소화 |
| 레이블 활용 | 소스만, 분포 재가중 | 소스 + 의사 레이블 | 소스 + 예측 출력 |
| Common support 가정 | 불필요 | 필요 | 불필요 |
| 불균형 처리 | 명시적 (β 벡터) | 암묵적 | 부분적 |

---

### 4.3 향후 연구 시 고려해야 할 점

**① 더 정확한 타겟 레이블 분포 추정**
현재 RAAN의 $\boldsymbol{\beta}$ 추정은 상대적 추세만 포착하며 정확도에 한계가 있습니다. Optimal Transport 이론이나 클러스터링 기반의 더 정교한 추정 방법 (예: EM 알고리즘과의 결합) 연구가 필요합니다.

**② 조건부 분포 매칭으로의 확장**
RAAN은 주변 분포(marginal distribution) 매칭에 집중하나, $P(Y|\mathcal{T}(\boldsymbol{X}))$의 명시적 매칭을 위한 조건부 OT 기반 접근법 연구가 필요합니다.

**③ 다중 소스/타겟 도메인으로의 확장**
실세계에서는 단일 소스-타겟 쌍이 아닌 다양한 도메인이 혼재합니다. Multi-source 및 multi-target DA 시나리오에서의 OT 기반 방법 확장이 요구됩니다.

**④ Semi-supervised 및 Few-shot 설정으로의 적용**
타겟 도메인에 소량의 레이블이 있는 경우, 레이블 분포 추정을 더 정확히 할 수 있어 RAAN의 성능이 더욱 향상될 수 있습니다.

**⑤ Transformer 기반 백본과의 결합**
Vision Transformer(ViT) 등 최신 백본 아키텍처와 OT 기반 DA의 결합 시 attention 메커니즘이 도메인 불변 특징 추출에 미치는 영향 연구가 필요합니다.

**⑥ 이론적 일반화 경계 도출**
RAAN의 경험적 성능 우수성에도 불구하고, OT 기반 DA의 일반화 오차 경계(generalization bound) 도출이 부족합니다. Ben-David 등의 이론적 프레임워크를 OT 설정으로 확장하는 연구가 필요합니다.

**⑦ 계산 효율성 개선**
Mini-batch OT의 편향 및 분산 문제를 해결하는 효율적인 알고리즘 (Sinkhorn divergence 등) 적용을 고려해야 합니다.

---

## 참고 자료

**주요 참고 논문 (논문 원문 내 인용):**
- Chen, Q., Liu, Y., Wang, Z., Wassell, I., & Chetty, K. (2018). **Re-weighted Adversarial Adaptation Network for Unsupervised Domain Adaptation.** *CVPR 2018*, pp. 7976–7985. *(본 분석의 주 논문)*
- Arjovsky, M., Chintala, S., & Bottou, L. (2017). **Wasserstein Generative Adversarial Networks.** *ICML 2017*, pp. 214–223.
- Courty, N., Flamary, R., Tuia, D., & Rakotomamonjy, A. (2017). **Optimal Transport for Domain Adaptation.** *IEEE TPAMI*, 39(9):1853–1865.
- Courty, N., Flamary, R., Habrard, A., & Rakotomamonjy, A. (2017). **Joint Distribution Optimal Transportation for Domain Adaptation.** *arXiv:1705.08848*.
- Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., & Courville, A. (2017). **Improved Training of Wasserstein GANs.** *arXiv:1704.00028*.
- Ganin, Y., et al. (2016). **Domain-Adversarial Training of Neural Networks.** *JMLR*, 17(59):1–35.
- Tzeng, E., Hoffman, J., Saenko, K., & Darrell, T. (2017). **Adversarial Discriminative Domain Adaptation.** *arXiv:1702.05464*.
- Villani, C. (2008). **Optimal Transport: Old and New.** Springer.
- Ben-David, S., et al. (2010). **A Theory of Learning from Different Domains.** *Machine Learning*, 79(1):151–175.

**2020년 이후 비교 연구 (일반적으로 알려진 연구, 세부 수치 불확실성 있음):**
- Jin, Y., et al. (2020). **Minimum Class Confusion for Versatile Domain Adaptation.** *ECCV 2020*.
- Long, M., et al. (2018). **Conditional Adversarial Domain Adaptation.** *NeurIPS 2018*.
