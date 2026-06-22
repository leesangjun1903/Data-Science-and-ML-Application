# DIRL: Domain-Invariant Representation Learning for Sim-to-Real Transfer 

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

DIRL의 핵심 주장은 다음과 같습니다:

> **기존의 주변 분포(marginal distribution)만 정렬하는 비지도 도메인 적응 방법은 조건부 분포(conditional distribution)의 차이로 인해 잘못된 전이(negative transfer)를 유발할 수 있으며, 이를 해결하기 위해 주변 분포와 조건부 분포를 동시에 정렬하고 트리플렛 분포 손실을 결합해야 한다.**

### 세 가지 주요 기여

| 기여 | 설명 |
|------|------|
| **① 기존 방법의 한계 규명** | 주변 분포만 정렬할 때 발생하는 cross-label mismatch와 label shift 문제를 이론적·실험적으로 입증 |
| **② DIRL 알고리즘 제안** | 주변 + 조건부 분포를 동시에 적대적 학습으로 정렬하고, 트리플렛 분포 손실로 클래스 간 분리를 강화 |
| **③ 실제 로봇 적용 검증** | Sim-to-Real 전이를 통해 객체 인식 정확도를 26.8% → 91.0%로 향상, 파지 정확도 86.5% 달성 |

---

## 2. 상세 분석

### 2-1. 해결하고자 하는 문제

#### 문제 배경: Sim-to-Real Gap

시뮬레이터에서 학습된 모델은 실제 환경에서 성능이 크게 저하됩니다. 이를 해결하기 위한 기존 도메인 적응 방법들은 **주변 분포 정렬(marginal distribution alignment)** 만을 수행하는데, 이는 두 가지 근본적 문제를 야기합니다:

1. **Cross-label mismatch**: 소스 도메인의 클래스 A가 타겟 도메인의 클래스 B와 매핑될 수 있음
2. **Label shift**: 도메인 간 클래스 불균형으로 인해 하나의 클래스가 다른 클래스와 혼합됨

#### 이론적 근거: Ben-David et al. [43]의 타겟 에러 상한

$$\epsilon_T(\pi) \leq \epsilon_S(\pi) + \frac{1}{2} d_{\mathcal{H}\Delta\mathcal{H}}(D_S^Z, D_T^Z) + \lambda_{\mathcal{H}} $$

여기서:
- $\epsilon_T(\pi)$: 타겟 도메인에서의 에러
- $\epsilon_S(\pi)$: 소스 도메인에서의 에러
- $d_{\mathcal{H}\Delta\mathcal{H}}(D_S^Z, D_T^Z)$: 도메인 간 가설 클래스의 주변 불일치도
- $\lambda_{\mathcal{H}} = \min_{\pi \in \mathcal{H}}[\epsilon_S(\pi) + \epsilon_T(\pi)]$: 최적 결합 가설 에러

더 정확하게는:

$$\epsilon_T(\pi) \leq \epsilon_S(\pi) + d_{\mathcal{H}\Delta\mathcal{H}}(D_S^Z, D_T^Z) + \min\{\mathbb{E}_{D_S}|\pi_S - \pi_T|, \mathbb{E}_{D_T}|\pi_S - \pi_T|\} $$

**DIRL은 $\lambda_{\mathcal{H}}$가 작다는 가정이 조건부 분포 차이가 있을 때 성립하지 않음을 지적하고, 이를 명시적으로 최소화합니다.**

---

### 2-2. 제안하는 방법 (수식 포함)

#### DIRL 목적함수: 결합 분포 불일치 최소화

$$\left|\log \Pr(X^S, Y^S) - \log \Pr(X^T, Y^T)\right|$$

$$= \underbrace{d_{\Pr(g(X))}(D_S^Z, D_T^Z)}_{\text{marginal discrepancy}} + \underbrace{d_{\Pr(Y|g(X))}(D_S^Y, D_T^Y)}_{\text{conditional discrepancy}} $$

이 식은 결합 분포의 차이가 **주변 불일치 + 조건부 불일치**로 분해됨을 보여줍니다.

---

#### ① 주변 분포 정렬 (Marginal Distributions Alignment)

적대적 학습을 통한 도메인 판별자 훈련:

**판별자 손실 (Discriminator Loss)**:

$$\min_D \mathcal{L}_{ma}\Big(g(\boldsymbol{x}_s, \boldsymbol{x}_t), D(\boldsymbol{x}_s, \boldsymbol{x}_t)\Big) = -\mathbb{E}_{\boldsymbol{x}_s \sim X_s}[\log D(g(\boldsymbol{x}_s))] - \mathbb{E}_{\boldsymbol{x}_t \sim X_t}[\log(1 - D(g(\boldsymbol{x}_t)))] $$

**생성자 손실 (Generator Loss)**:

$$\min_g \mathcal{L}_{ma}\Big(g(\boldsymbol{x}_t), D(\boldsymbol{x}_s, \boldsymbol{x}_t)\Big) = -\mathbb{E}_{\boldsymbol{x}_t \sim X_t}[\log D(g(\boldsymbol{x}_t))] $$

---

#### ② 조건부 분포 정렬 (Conditional Distributions Alignment)

**분류 손실 (Cross-Entropy)**:

$$\mathcal{L}_{ca\_sc}\Big(f \circ g(\boldsymbol{x}_s, \boldsymbol{y}_s, \boldsymbol{x}_t, \boldsymbol{y}_t)\Big) = \mathbb{E}_{\boldsymbol{x}_s, \boldsymbol{y}_s \sim (X_s, Y_s)}[-\boldsymbol{y}_s \log f(g(\boldsymbol{x}_s))] + \mathbb{E}_{\boldsymbol{x}_t, \boldsymbol{y}_t \sim (X_t, Y_t)}[-\boldsymbol{y}_t \log f(g(\boldsymbol{x}_t))] $$

**클래스별 조건부 적대적 손실** (각 클래스 $k = 1 \ldots |\mathcal{Y}|$에 대해):

$$\min_C \mathcal{L}_{ca_k}\Big(g(x_s^{(k)}, x_t^{(k)}), C(x_s^{(k)}, x_t^{(k)})\Big) = -\mathbb{E}_{x_s^{(k)} \sim X_s}\left[\log C\left(g(x_s^{(k)})\right)\right] - \mathbb{E}_{x_t^{(k)} \sim X_t}\left[\log\left(1 - C\left(g(x_t^{(k)})\right)\right)\right]$$

$$\min_g \mathcal{L}_{ca_k}\Big(g(x_s^{(k)}, x_t^{(k)}), C(x_s^{(k)}, x_t^{(k)})\Big) = -\mathbb{E}_{x_t^{(k)} \sim X_t}\left[\log C\left(g(x_t^{(k)})\right)\right] $$

---

#### ③ 트리플렛 분포 손실 (Triplet Distribution Loss)

개별 샘플 튜플이 아닌 **분포 수준**에서 동작하는 새로운 트리플렛 손실:

```math
\mathcal{L}_{tl} = \sum_{a=1}^{M} \left[ \frac{1}{M_p - 1} \sum_{\substack{p=1 \\ p \neq a}}^{M_p} \text{KL}\left(\mathcal{N}(\bar{g}(\boldsymbol{x}_a), \sigma^2) \,\Big\|\, \mathcal{N}(\bar{g}(\boldsymbol{x}_p), \sigma^2)\right) - \frac{1}{M_n} \sum_{n=1}^{M_n} \text{KL}\left(\mathcal{N}(\bar{g}(\boldsymbol{x}_a), \sigma^2) \,\Big\|\, \mathcal{N}(\bar{g}(\boldsymbol{x}_n), \sigma^2)\right) + \alpha_{tl} \right]_+
```

여기서 분포는 다음과 같이 정의됩니다:

```math
\mathcal{N}\left(\bar{g}(\boldsymbol{x}_i); \bar{g}(\boldsymbol{x}_a), \sigma^2\right) = \left\{ \frac{\exp\left(-\frac{1}{\sigma^2}\|\bar{g}(\boldsymbol{x}_i) - \bar{g}(\boldsymbol{x}_a)\|_2^2\right)}{\sum_{j=1}^{K} \exp\left(-\frac{1}{\sigma^2}\|\bar{g}(\boldsymbol{x}_j) - \bar{g}(\boldsymbol{x}_a)\|_2^2\right)} \right\}_{i=1}^{K}
```

- $\bar{g}(\boldsymbol{x})$: 스케일 불변 특징 추출을 위해 정규화된 특징 벡터
- $M_p$: 미니배치 내 긍정(positive) 샘플 수 (동일 클래스)
- $M_n$: 미니배치 내 부정(negative) 샘플 수 (다른 클래스)
- $\alpha_{tl} \in \mathbb{R}^+$: 마진 상수
- $\{.\}_+$: 힌지 손실(hinge loss)

---

#### ④ 전체 DIRL 손실 함수

$$\mathcal{L}_{\text{DIRL}} = \lambda_1 \mathcal{L}_{ca\_sc}\Big(f \circ g(\boldsymbol{x}_s, \boldsymbol{y}_s, \boldsymbol{x}_t, \boldsymbol{y}_t)\Big) + \lambda_2 \mathcal{L}_{ma}\Big(g(\boldsymbol{x}_t), D(\boldsymbol{x}_s, \boldsymbol{x}_t)\Big)$$

$$+ \lambda_3 \sum_{k=1}^{|\mathcal{Y}|} \mathcal{L}_{ca_k}\Big(g(x_t^{(k)}), D(x_s^{(k)}, x_t^{(k)})\Big) + \lambda_4 \mathcal{L}_{tl}\Big(g(\boldsymbol{x}_s, \boldsymbol{y}_s, \boldsymbol{x}_t, \boldsymbol{y}_t)\Big) $$

---

### 2-3. 모델 구조

```
입력 X
    │
    ▼
┌─────────────┐
│  특징 추출기  │  g: X → Z  (공유 특징 공간)
│      g       │  (CNN: Conv layers + Feature Pyramid)
└─────┬───────┘
      │
      ├──────────────────────────────────────┐
      │                                      │
      ▼                                      ▼
┌───────────┐                    ┌────────────────────┐
│  분류기   │  f: Z → Y          │  주변 도메인 판별자 │  D(Z) → {sim, real}
│     f     │  (Cross-Entropy)   │        D           │
└───────────┘                    └────────────────────┘
      │
      ├──────────────────────────────────────┐
      │                                      │
      ▼                                      ▼
┌───────────────┐                ┌────────────────────────┐
│  트리플렛      │  KL-div based  │  클래스별 조건부 판별자 │  C_k(Z) → {sim, real}
│  분포 손실    │  metric         │      C_1, ..., C_|Y|   │
└───────────────┘                └────────────────────────┘
```

**로봇 실험 기반 모델 (MobileNet-SSD + FPN)**:
- **Backbone**: VGG16 (pre-trained)
- **Feature Pyramid Network**: 5개 해상도 (80×80, 40×40, 20×20, 10×10, 5×5)
- **도메인 판별자**: FC(1024) → FC(200) → FC(100) → FC(2)
- **클래스 조건부 판별자**: FC(200) → FC(100) → FC(2)

---

### 2-4. 성능 향상

#### 디지트 도메인 벤치마크 (Table 1)

| 방법 | MNIST→SVHN (10-shot) | USPS→SVHN (10-shot) | USPS→MNIST (10-shot) |
|------|----------------------|----------------------|----------------------|
| DANN | 0.405 | 0.354 | 0.914 |
| FADA | 0.470 | 0.429 | 0.915 |
| **DIRL** | **0.837** | **0.802** | **0.962** |

- MNIST→SVHN: DIRL은 1-shot에서 5-shot으로 **+28.2%**, 5-shot에서 10-shot으로 **+12.0%** 향상

#### Sim-to-Real 로봇 실험 (Table 2)

| 방법 | mAP | sim_eval | real_eval | Silhouette Score |
|------|-----|----------|-----------|-----------------|
| Sim Only | 0.13 | 95.7% | 26.8% | 0.08 |
| Real Only | 0.62 | 24.6% | 85.9% | 0.42 |
| DANN | 0.61 | 93.2% | 84.4% | 0.30 |
| MCD | 0.65 | 94.1% | 89.6% | 0.48 |
| **DIRL** | **0.69** | **94.2%** | **91.0%** | **0.69** |

- 시뮬레이션만 사용 대비: **26.8% → 91.0%** (무려 +64.2%p)
- 파지 정확도: **86.5%** (65개 물리 객체 대상)

---

### 2-5. 한계점

1. **레이블된 타겟 데이터 필요**: 완전 비지도 방법이 아닌 semi-supervised로, 타겟 도메인의 소량 레이블 데이터가 필수
2. **복잡한 물체 형상 처리 한계**: 조립 부품, 뒤집힌 그릇 등 비정형 형상의 객체 파지 실패 사례 존재
3. **하이퍼파라미터 민감성**: $\lambda_1, \lambda_2, \lambda_3, \lambda_4$, $\sigma^2$, $\alpha_{tl}$ 등 다수의 하이퍼파라미터 조정 필요
4. **단일 환경 실험**: 다양한 환경(조명, 배경)에서의 일반화 성능 미검증
5. **Pseudo-label 품질 의존성**: 비레이블 타겟 데이터에 대한 의사 레이블의 품질이 성능에 영향

---

## 3. 모델의 일반화 성능 향상 가능성

DIRL의 일반화 성능은 다음 세 가지 메커니즘으로 향상됩니다:

### 3-1. 결합 분포 정렬에 의한 일반화

기존 방법은 $\Pr(g(X^S)) \approx \Pr(g(X^T))$만 보장하지만, DIRL은 다음을 동시에 달성합니다:

$$\Pr(Y^S | g(X^S)) \approx \Pr(Y^T | g(X^T))$$

이는 Ben-David et al. 이론에서 $\lambda_{\mathcal{H}}$ 항을 명시적으로 줄임으로써, **타겟 에러 상한 자체를 낮추는 효과**를 가집니다.

### 3-2. 트리플렛 분포 손실에 의한 특징 공간 구조화

트리플렛 손실은 다음 두 가지 효과로 일반화를 향상시킵니다:
- **클래스 간 분산(inter-class variance) 증가**: 다른 클래스 간 거리를 KL-발산으로 최대화
- **클래스 내 분산(intra-class variance) 감소**: 같은 클래스 내 거리를 최소화

이 구조화된 특징 공간은 새로운 도메인에서도 결정 경계가 명확하게 유지되도록 합니다.

### 3-3. Semi-supervised 설정의 실용적 일반화

- 소량(1-shot, 5-shot, 10-shot)의 타겟 레이블만으로 대규모 도메인 차이(e.g., MNIST→SVHN)를 극복
- Pseudo-label 생성 전략과 balanced mini-batch 샘플링을 통해 클래스 불균형 문제 완화

### 3-4. Silhouette Score의 증거

DIRL의 Silhouette Score는 **0.69**로, MCD(0.48), DANN(0.30) 대비 압도적으로 높습니다. 이는 레이블 없이도 특징 공간에서 클래스 클러스터가 잘 분리되어 있음을 의미하며, **실제 분포에 대한 구조적 이해도가 높다**는 증거입니다.

---

## 4. 향후 연구에 미치는 영향과 고려사항

### 4-1. 향후 연구에 미치는 영향

#### (1) Sim-to-Real 로보틱스 연구 방향 전환
DIRL은 순수 도메인 랜덤화(domain randomization)에서 벗어나, **소량의 실데이터와 결합한 semi-supervised 도메인 적응**이 효과적임을 실증했습니다. 이는 데이터 효율적 로봇 학습 연구에 중요한 기준점을 제공합니다.

#### (2) 조건부 분포 정렬의 중요성 재인식
기존 DANN, ADDA 등 주변 분포만 정렬하던 패러다임에서, **조건부 분포 정렬의 명시적 포함**이 필수적임을 이론적으로 증명했습니다.

#### (3) 분포 기반 메트릭 학습 활용
개별 샘플이 아닌 **분포 단위의 트리플렛 손실**은 이후 contrastive learning, self-supervised 도메인 적응 연구에 영향을 미쳤습니다.

---

### 4-2. 2020년 이후 관련 최신 연구 비교 분석

#### 비교 연구 1: CDTrans (ICLR 2022)
- **논문**: "CDTrans: Cross-Domain Transformer for Unsupervised Domain Adaptation" (Xu et al., 2022)
- **방법**: Transformer 기반의 triple attention mechanism으로 도메인 간 상호 정보를 학습
- **DIRL과의 차이**: 완전 비지도(unsupervised) 설정에서 강점; DIRL은 소량 레이블 사용으로 조건부 분포 정렬에서 우위
- **한계**: Transformer 기반으로 계산 비용이 높음

#### 비교 연구 2: PMTrans (ECCV 2022)
- **논문**: "Patch Mix Transformer for Unsupervised Domain Adaptation: A Game Perspective" (Zhu et al., 2022)
- **방법**: 도메인 적응을 게임 이론적 관점으로 재해석하여 패치 혼합(patch mix) 전략 사용
- **DIRL과의 차이**: 데이터 증강 관점 vs. DIRL의 손실 함수 설계 관점

#### 비교 연구 3: SDAT (ICML 2022)
- **논문**: "Towards Safer Domain Adaptation via Self-Distillation" (Rangwani et al., 2022)
- **방법**: 샤프니스 인식 최소화(Sharpness-Aware Minimization)를 도메인 적응에 적용
- **공통점**: 조건부 분포의 안정적 정렬 중요성 강조

#### 비교 연구 4: 대규모 Foundation Model 활용 (2023~)
- **논문**: "CLIP-based Domain Adaptation" 계열 연구들
- **방법**: CLIP, DINOv2 등 대규모 사전학습 모델의 시각-언어 특징을 도메인 적응에 활용
- **DIRL과의 차이**: DIRL은 특정 도메인 데이터 없이도 일반화되는 Foundation Model 표현 활용 불가; 향후 DIRL의 손실 함수 설계를 Foundation Model 파인튜닝에 통합하는 방향이 유망

#### 비교 연구 5: DrQ-v2 / DreamerV3 (Sim-to-Real RL)
- **논문**: "Mastering Diverse Domains through World Models" (Hafner et al., 2023)
- **방법**: 세계 모델(world model) 기반 강화학습으로 시뮬레이션과 실제 환경 간 격차 최소화
- **DIRL과의 차이**: 강화학습 기반 정책 학습 vs. DIRL의 지도/반지도 분류 중심 접근

---

### 4-3. 앞으로 연구 시 고려할 점

| 고려사항 | 세부 내용 |
|----------|----------|
| **① Foundation Model 통합** | CLIP, DINOv2 등의 풍부한 사전학습 표현을 DIRL 프레임워크와 통합하면 few-shot 레이블 의존성을 더욱 줄일 수 있음 |
| **② 완전 비지도 확장** | 현재 semi-supervised 설정을 극복하기 위해, 타겟 도메인 레이블 없이도 조건부 분포를 정렬하는 방법 연구 필요 (e.g., pseudo-label 품질 개선, confidence calibration) |
| **③ 다중 도메인 일반화** | 단일 소스→타겟이 아닌 복수 환경(fleet of robots) 시나리오로의 확장 — 논문 저자도 향후 과제로 명시 |
| **④ 동적 환경 변화 대응** | 정적 도메인 적응이 아닌, 환경이 지속적으로 변화하는 continual/online domain adaptation 연구 필요 |
| **⑤ 이론적 보장 강화** | 조건부 분포 정렬에 대한 PAC-Bayes 이론 수준의 엄밀한 수렴 보장 연구 필요 (현재 Ben-David 이론은 주변 분포 중심) |
| **⑥ 계산 효율성** | 클래스별 판별자 $C_1, ..., C_{\mathcal{Y}}$는 클래스 수에 비례하는 계산 비용 발생 — 클래스 수가 많은 대규모 분류 태스크에서의 효율화 필요 |
| **⑦ 실제 로봇 안전성** | Sim-to-Real 전이 오류가 물리적 충돌이나 위험으로 이어질 수 있으므로, 불확실성 추정(uncertainty quantification)을 통한 안전한 전이 메커니즘 연구 필요 |

---

## 참고 자료

### 본 논문
- **Tanwani, A. K.** (2021). *DIRL: Domain-Invariant Representation Learning for Sim-to-Real Transfer*. 4th Conference on Robot Learning (CoRL 2020). arXiv:2011.07589v3.
  - URL: https://arxiv.org/abs/2011.07589

### 본 논문 내 주요 인용 문헌
- **Ben-David et al.** (2010). A theory of learning from different domains. *Machine Learning*, 79(1):151–175.
- **Ganin et al.** (2016). Domain-adversarial training of neural networks. *JMLR*, 17(1):2096–2030. (DANN)
- **Tzeng et al.** (2017). Adversarial discriminative domain adaptation. arXiv:1702.05464. (ADDA)
- **Saito et al.** (2018). Maximum classifier discrepancy for unsupervised domain adaptation. *CVPR*. (MCD)
- **Motiian et al.** (2017). Few-shot adversarial domain adaptation. arXiv:1711.02536. (FADA)
- **Zhao et al.** (2019). On learning invariant representation for domain adaptation. arXiv:1901.09453.
- **Schroff et al.** (2015). Facenet: A unified embedding for face recognition and clustering. arXiv:1503.03832.
- **Tobin et al.** (2017). Domain randomization for transferring deep neural networks from simulation. arXiv:1703.06907.

### 비교 분석에 활용한 2020년 이후 연구
- **Xu et al.** (2022). CDTrans: Cross-Domain Transformer for Unsupervised Domain Adaptation. *ICLR 2022*. arXiv:2109.06165.
- **Rangwani et al.** (2022). A Closer Look at Smoothness in Domain Adversarial Training. *ICML 2022*. arXiv:2206.08213.
- **Hafner et al.** (2023). Mastering Diverse Domains through World Models (DreamerV3). arXiv:2301.04104.
- **Li et al.** (2020). Learning invariant representations and risks for semi-supervised domain adaptation. arXiv:2010.09709.

> **⚠️ 주의**: 비교 분석 섹션의 일부 연구(CDTrans, PMTrans, SDAT 등)는 DIRL 논문 발표 이후 등장한 연구들로, 본 논문 원문에서 직접 인용되지 않았으며 필자가 해당 분야 발전 흐름을 기반으로 분석한 내용입니다. 해당 논문들의 정확한 수치 비교는 각 원 논문을 직접 참조하시기 바랍니다.
