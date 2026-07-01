# Unsupervised Domain Adaptation with Adversarial Residual Transform Networks (ARTNs)

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

본 논문은 비지도 도메인 적응(Unsupervised Domain Adaptation, UDA) 문제에서 기존 **대칭(symmetric)** 및 **비대칭(asymmetric)** 적대적 아키텍처의 한계를 동시에 극복하는 새로운 방법론인 **Adversarial Residual Transform Networks (ARTNs)** 를 제안합니다.

| 기존 방법 | 문제점 |
|---|---|
| 대칭 아키텍처 (DANN 등) | 일반화 능력(generalization ability) 부족 |
| 비대칭 아키텍처 (ADDA 등) | 훈련이 매우 어렵고 불안정 |

**ARTNs의 핵심 주장:** 잔차 연결(residual connections)을 활용하여 소스 피처를 타겟 피처 공간으로 직접 변환함으로써, 대칭 모델의 유연성 부족과 비대칭 모델의 훈련 불안정성을 동시에 해결할 수 있다.

### 주요 기여 (3가지)

1. **피처 공유 변환 네트워크 도입:** 소스 도메인의 피처를 타겟 도메인의 피처 공간으로 비선형 매핑하는 새로운 적대적 모델 제안 → 타겟 도메인에서 높은 일반화 능력 확보

2. **최적 수송 이론 기반 정규화 항 설계:** 소스→타겟 변환 경로 중 최단 경로를 선택하도록 유도하는 정규화 항 도입 → 기울기 소실 문제(vanishing gradient) 완화 및 훈련 안정화

3. **다양한 벤치마크에서 검증:** Amazon review, Digits (MNIST/SVHN 등), Office-31에서 당시 state-of-the-art와 비교 가능한 성능 달성

---

## 2. 상세 분석

### 2-1. 해결하고자 하는 문제

**도메인 이동(Domain Shift)** 문제: 소스 도메인 $\mathcal{D}_s = \{x_i^s, y_i^s\}\_{i=1}^{n_s}$ 와 타겟 도메인 $\mathcal{D}_t = \{x_i^t\}\_{i=1}^{n_t}$ 의 분포가 다를 때 ( $P_s(\mathbf{X}^s) \neq P_t(\mathbf{X}^t)$ ), 소스에서 학습한 모델이 타겟에서 성능 저하되는 문제.

구체적으로 해결하고자 하는 세 가지 세부 문제:

1. **대칭 아키텍처의 일반화 부족:** 소스·타겟이 동일한 피처 추출기를 공유하므로 도메인별 최적화 불가
2. **비대칭 아키텍처의 훈련 불안정성:** 두 네트워크 간 관계가 없어 타겟 피처 추출기 붕괴(collapse) 위험
3. **기울기 소실 문제:** Jensen-Shannon divergence가 두 분포가 겹치지 않을 때 상수값 $\log 2$가 되어 기울기가 0에 수렴

---

### 2-2. 제안하는 방법 (수식 포함)

#### (A) 기존 DANN의 손실 함수

$$\mathcal{L}(\theta_d, \theta_g, \theta_c) = \frac{1}{n_s} \sum_{\mathbf{x}_i \in D_s} \mathcal{L}_c(C(G(\mathbf{x}_i)), y_i) - \frac{\lambda}{n} \sum_{\mathbf{x}_i \in D_s \cup D_t} \mathcal{L}_d(D(G(\mathbf{x}_i)), d_i) $$

여기서 $\lambda$는 분류 손실과 도메인 손실 간의 균형 파라미터.

#### (B) 잔차 연결 수식

$N$ 레이어 변환 네트워크에서 $i < N$ 인 $i$번째 레이어:

$$T_i(G_i(\mathbf{x}^s)) = T_i(T_{i-1}(G_{i-1}(\mathbf{x}^s))) + G_i(\mathbf{x}^s) $$

소스 피처 $\mathbf{f}^s = G(\mathbf{x}^s)$ 를 타겟 피처 공간으로 변환: $T(G(\mathbf{x}^s)) \approx M(G(\mathbf{x}^s))$

여기서 원하는 매핑: $T(G(\mathbf{x}^s)) = M(G(\mathbf{x}^s)) - G(\mathbf{x}^s)$

#### (C) 기울기 소실 문제의 이론적 분석

최적 판별기 $D^*$에서:

$$D^*(\mathbf{z}) = \frac{P_s(\mathbf{z})}{P_s(\mathbf{z}) + P_t(\mathbf{z})} $$

최소화 목적함수는 Jensen-Shannon Divergence로 수렴:

$$\min_{G,T} \mathcal{L}(D^*, G, T) = 2 \cdot \text{JSD}(P_s \| P_t) - 2\log 2 $$

**문제:** 두 분포가 다른 다양체(manifold)에 있을 때 $\text{JSD}(P_s \| P_t) = \log 2$ (상수)가 되어 모든 파라미터의 기울기가 0이 됨 → 기울기 소실 발생.

#### (D) 최적 수송 이론 기반 정규화

Monge의 최적 수송 공식:

```math
T_0 = \arg\min_T \int_{\mathbf{x} \in P_s} r(\mathbf{x}, T(\mathbf{x})) d\mu(\mathbf{x}), \quad \text{s.t.} \quad T\#(\mu_s) = \mu_t
```

코사인 거리를 이용한 정규화 항:

$$r(G(\mathbf{x}^s), T(G(\mathbf{x}^s))) = -\frac{\langle G(\mathbf{x}^s) \cdot T(G(\mathbf{x}^s)) \rangle}{|G(\mathbf{x}^s)| \cdot |T(G(\mathbf{x}^s))|} $$

#### (E) 최종 목적 함수

$$\mathcal{L}(\theta_d, \theta_g, \theta_c, \theta_t) = \frac{1}{n_s} \sum_{\mathbf{x}_i \in D_s} \mathcal{L}_c(C(T(G(\mathbf{x}_i))), y_i) - \frac{\lambda}{n}\left(\sum_{\mathbf{x}_i \in D_s} \mathcal{L}_s(D(T(G(\mathbf{x}_i))), d_i^s) + \sum_{\mathbf{x}_i \in D_t} \mathcal{L}_t(D(G(\mathbf{x}_i)), d_i^t)\right) + \beta \cdot r(G(\mathbf{x}^s), T(G(\mathbf{x}^s))) $$

정규화 항 추가 후, 최적화 목적은:

$$\min_{G,T} \mathcal{L}(D^*, G, T) = 2 \cdot \text{JSD}(P_s \| P_t) - 2\log 2 + r(\theta_g, \theta_t) $$

$r(\theta_g, \theta_t)$의 기울기 $\frac{\partial r(\theta_g, \theta_t)}{\partial \theta_g}$ 와 $\frac{\partial r(\theta_g, \theta_t)}{\partial \theta_t}$ 가 0이 아니므로 기울기 소실 문제 완화.

각 파라미터 최적화:

$$\hat{\theta}_d = \arg\min_{\theta_d} (\mathcal{L}_s(\theta_d, \theta_g, \theta_t) + \mathcal{L}_t(\theta_d, \theta_g)) $$

$$\hat{\theta}_c = \arg\min_{\theta_c} \mathcal{L}_c(\theta_g, \theta_c, \theta_t) $$

$$\hat{\theta}_g = \arg\min_{\theta_g} \mathcal{L}(\theta_d, \theta_g, \theta_c, \theta_t) $$

$$\hat{\theta}_t = \arg\min_{\theta_t} (\mathcal{L}_c(\theta_g, \theta_c, \theta_t) + \mathcal{L}_s(\theta_d, \theta_g, \theta_t) + r(\theta_g, \theta_t)) $$

---

### 2-3. 모델 구조

```
[소스 입력 x^s] ──→ [피처 추출기 G (가중치 공유)] ──→ f^s = G(x^s)
                                                          │
                                              [잔차 연결] │
                                                          ↓
                                          [변환 네트워크 T] ──→ T(G(x^s)) (가짜 타겟 피처)
                                                          │           │
                                                          │           ↓
[타겟 입력 x^t] ──→ [피처 추출기 G (가중치 공유)] ──→ f^t = G(x^t) ──→ [도메인 분류기 D] → ℒ_d
                                                                      ↑
                                                          T(G(x^s)) ──┘
                                                              │
                                                              ↓
                                                      [레이블 분류기 C] → ℒ_c
                                          r(G(x^s), T(G(x^s))): 정규화 항
```

**핵심 구성요소:**

| 구성요소 | 역할 |
|---|---|
| 피처 추출기 $G$ | 소스·타겟 도메인 공유, 피처 추출 |
| 변환 네트워크 $T$ | 소스 피처 → 타겟 피처 공간 매핑 (잔차 연결 포함) |
| 레이블 분류기 $C$ | $T(G(\mathbf{x}^s))$로 학습, $G(\mathbf{x}^t)$로 예측 |
| 도메인 분류기 $D$ | 가짜 타겟 피처 $T(G(\mathbf{x}^s))$와 실제 타겟 피처 $G(\mathbf{x}^t)$ 구별 |
| 정규화 항 $r(\cdot)$ | 변환 전후 피처 거리 최소화 (코사인 거리) |

**훈련 전략:** Gradient Reversal Layer (GRL)를 사용하여 end-to-end 최적화 ($\gamma = \frac{2}{1+e^{-10p}} - 1$, $p$: 훈련 진행도).

---

### 2-4. 성능 향상

#### 감성 분석 (Amazon Reviews, 12가지 태스크)

| 방법 | 최고 성능 태스크 수 |
|---|---|
| CMD | 6/12 |
| VFAE | 3/12 |
| **ARTN** | **3/12** |

→ DANN 대비 전반적 향상, CMD와 경쟁적 수준.

#### 숫자 인식 (Digits)

| 태스크 | DANN | CMD | **ARTN** |
|---|---|---|---|
| MNIST→MNIST-M | 76.7% | 85.0% | **85.6%** |
| SYN→SVHN | **91.1%** | 85.5% | 89.1% |
| SVHN→MNIST | 73.9% | 84.5% | **85.8%** |

→ 기본 네트워크 대비 개선 폭: ARTN **30.9%** vs DTN 8.3%, CyCADA pixel 3.2%.

#### 이미지 분류 (Office-31)

| 태스크 | DANN | CMD (VGG16) | **ARTN (ResNet34)** |
|---|---|---|---|
| D→A | 58.1% | **63.8%** | 60.9% |
| W→A | 56.3% | **63.3%** | 61.0% |
| A→W | 73.7% | **77.0%** | 76.2% |
| A→D | 75.3% | **79.6%** | 76.1% |

→ 모든 태스크에서 2위, CMD 다음으로 우수한 성능.

---

### 2-5. 한계점

1. **CMD 대비 열세:** Office-31 모든 태스크에서 CMD(VGG16 기반)에 비해 낮은 성능 기록
2. **하이퍼파라미터 민감성 일부 존재:** $\lambda$ 변화에 W→A 태스크가 다소 민감하게 반응
3. **정규화 항의 예외 사례:** W→A 태스크에서 정규화 효과가 다른 태스크만큼 명확하지 않음 (논문에서도 미래 연구 과제로 언급)
4. **대규모·고해상도 이미지 데이터셋 검증 부족:** Office-31의 4가지 태스크만 검증
5. **정규화 거리 메트릭 선택의 임의성:** 코사인 거리를 경험적으로 선택했으나 이론적 근거 부족

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 노이즈 강건성 실험 (핵심 결과)

타겟 도메인 이미지에 가우시안 노이즈($\mu=0$, $\sigma \in \{0.4, ..., 1.0\}$)를 추가하여 일반화 능력 검증:

| 조건 | 적응 없음 | DANN 개선 | **ARTN 개선** |
|---|---|---|---|
| MNIST→MNIST-M ($\sigma=0.4$) | 45.83% | - | **+36.6%** |
| MNIST→MNIST-M ($\sigma=1.0$) | 24.55% | - | **+76.4%** |

**핵심 발견:** 노이즈 증가(도메인 격차 확대)에 따라 ARTN의 성능 향상 폭이 커지는 경향 → **도메인 격차가 클수록 ARTN의 일반화 이점이 두드러짐**

### 3-2. 일반화 성능 향상의 메커니즘

**(1) 잔차 연결을 통한 피처 공유:**

변환 네트워크가 소스 피처의 의미론적 정보(semantic information)를 잔차 연결을 통해 유지하면서 도메인 이동을 학습 → 소스의 분류 지식이 타겟 도메인 적응에 효과적으로 전이됨.

**(2) 최적 수송 정규화의 역할:**

$$r(G(\mathbf{x}^s), T(G(\mathbf{x}^s))) = -\frac{\langle G(\mathbf{x}^s) \cdot T(G(\mathbf{x}^s)) \rangle}{|G(\mathbf{x}^s)| \cdot |T(G(\mathbf{x}^s))|}$$

이 정규화 항은:
- **기울기 소실 방지:** JSD가 상수가 되어도 추가 기울기 공급
- **과도한 파라미터 변화 억제:** 변환 전후 피처가 너무 달라지지 않도록 제약
- **훈련 안정화 증거:** ARTN with reg의 $\|\nabla_\theta \mathcal{L}(\theta)\|$ 표준편차가 ARTN without reg보다 일관되게 낮음

**(3) 비대칭 유연성 + 대칭 안정성의 균형:**

변환 네트워크의 독립적 파라미터로 비대칭적 적응 능력 확보하면서, 공유 피처 추출기로 훈련 안정성 유지. 이는 개념적으로 다음과 같은 일반화 오차 상한(generalization error bound)과 연관됨:

$$\epsilon_t \leq \epsilon_s + \frac{1}{2}\hat{d}_{\mathcal{A}}(D_s, D_t) + C$$

여기서 $\hat{d}_{\mathcal{A}} = 2(1 - 2\epsilon)$ (Proxy A-distance)를 최소화하는 방향으로 학습.

### 3-3. 정규화 항이 훈련 안정성에 미치는 영향

| 지표 | ARTN (reg) | ARTN (no reg) |
|---|---|---|
| A→W max 기울기 | 14.51 | 18.50 |
| A→D max 기울기 | 15.88 | 20.35 |
| A→W std | **1.25** | 1.30 |
| A→D std | **1.36** | 1.37 |

→ 정규화 항이 극단적 기울기 발생을 억제하고 훈련을 안정화.

---

## 4. 앞으로의 연구에 미치는 영향과 고려할 점

### 4-1. 연구에 미치는 영향

**① 비대칭-대칭 하이브리드 아키텍처의 방향 제시**
잔차 연결을 통한 피처 공유 개념은 이후 다양한 DA 방법에서 유사한 접근법을 탐구하는 동기가 됨. 단순히 파라미터를 공유하거나 완전히 독립적인 것보다, 중간 수준의 공유를 설계하는 방향성 제공.

**② 최적 수송 이론과 심층 학습의 결합 촉진**
정규화 항을 Monge의 최적 수송 이론으로 정당화한 접근은 이후 OT 기반 DA 연구의 이론적 연결고리를 강화.

**③ 기울기 소실 문제의 명시적 해결 접근**
Wasserstein GAN (WGAN)과 유사하게 기울기 소실 문제를 이론적으로 분석하고 정규화로 해결한 접근은 안정적인 적대적 훈련 연구에 기여.

**④ 노이즈 강건성 평가 방법론**
가우시안 노이즈를 타겟 도메인에 추가하여 일반화를 평가하는 방법론은 이후 DA 연구의 표준 평가 항목으로 참고될 수 있음.

---

### 4-2. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의:** 아래 최신 연구 비교는 논문 PDF에 포함된 내용이 아닌, 제 훈련 데이터(2023년 초 기준)를 바탕으로 한 설명입니다. 개별 수치의 정확성은 원본 논문 확인을 권장합니다.

#### (A) SHOT (ICML 2020)
- **Liang et al., "Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation"**
- **차이점:** 소스 데이터에 접근 없이 타겟 도메인에서만 적응 (Source-Free DA)
- ARTN과 비교: ARTN은 소스 데이터 필요, SHOT은 소스 모델만 사용
- **의미:** 더 현실적인 프라이버시 보호 시나리오 해결

#### (B) CDAN (NeurIPS 2018, 이후 광범위 인용)
- **Long et al., "Conditional Adversarial Domain Adaptation"**
- **차이점:** 멀티리니어 컨디셔닝(multilinear conditioning)으로 클래스 조건부 정렬
- ARTN의 도메인 분류기는 레이블 조건 미고려, CDAN은 이를 명시적으로 모델링

#### (C) MDD (ICML 2019)
- **Zhang et al., "Bridging Theory and Algorithm for Domain Adaptation"**
- **차이점:** $\mathcal{H}\Delta\mathcal{H}$-divergence를 직접 최소화
- ARTN의 이론적 기반(H-divergence, PAD)과 유사하나 더 엄밀한 이론적 보장 제공

#### (D) TransDA / CDTrans (2021-2022, Vision Transformer 기반)
- **차이점:** Transformer 구조의 self-attention으로 글로벌 피처 관계 모델링
- ARTN은 CNN 기반으로 지역적 피처에 집중 → Transformer로의 확장 가능성 있음

#### (E) SSRT (CVPR 2022)
- **Sun et al., "Safe Self-Refinement for Transformer-based Domain Adaptation"**
- Self-supervised learning과 DA 결합 → ARTN의 정규화 개념을 자기지도 방식으로 확장 가능

#### 비교 요약표

| 방법 | 연도 | 핵심 메커니즘 | ARTN과의 차이 |
|---|---|---|---|
| ARTN (본 논문) | 2019 | 잔차 변환 + OT 정규화 | - |
| CDAN | 2018 | 조건부 적대적 정렬 | 클래스 조건 고려 |
| SHOT | 2020 | 소스 없는 적응 | Source-Free |
| MDD | 2019 | H∆H divergence | 이론적 엄밀성 강화 |
| CDTrans | 2021 | Transformer 기반 | 아키텍처 패러다임 전환 |
| SSRT | 2022 | Self-supervised + ViT | 자기지도학습 결합 |

---

### 4-3. 향후 연구 시 고려할 점

**① 더 강력한 정규화 메트릭 탐색**

현재 코사인 거리는 경험적으로 선택됨. Wasserstein 거리, Sinkhorn divergence 등 이론적으로 더 잘 정당화된 거리를 정규화에 활용:

$$r_W(G(\mathbf{x}^s), T(G(\mathbf{x}^s))) = W_p(P_s, P_t)$$

**② Transformer/ViT 백본으로의 확장**

ResNet 기반 잔차 연결 설계를 Vision Transformer의 attention 메커니즘과 결합. Self-attention이 도메인 불변 피처를 더 효과적으로 포착할 수 있음.

**③ Source-Free DA로의 확장**

프라이버시 보호 관점에서 소스 데이터 없이 변환 네트워크 $T$를 훈련하는 방법 탐구. 소스 도메인 통계(평균, 분산)만으로 $T$ 초기화 가능성 연구.

**④ 다중 소스/다중 타겟 도메인으로 확장**

현재 단일 소스→단일 타겟. 복수 소스 도메인에서 변환 네트워크를 혼합하거나, 타겟 도메인 클러스터링과 결합하는 연구 필요.

**⑤ 클래스 조건부 정렬 강화**

ARTN의 도메인 분류기는 레이블 정보를 활용하지 않음. 의미적으로 유사한 클래스끼리 정렬되도록 조건부 변환 네트워크 $T(G(\mathbf{x}^s) | y^s)$ 설계 가능.

**⑥ 이론적 일반화 오차 상한 도출**

ARTNs의 잔차 연결과 정규화 항이 일반화 오차 상한에 어떤 영향을 미치는지 엄밀한 이론적 분석 부재. 다음 형태의 bound 유도 필요:

$$\epsilon_t(\hat{h}) \leq \epsilon_s(\hat{h}) + \frac{1}{2}d_{\mathcal{H}\Delta\mathcal{H}}(D_s^T, D_t) + r(G, T) + \lambda^*$$

**⑦ W→A 예외 사례의 원인 분석**

논문 자체에서 "미래 연구 과제"로 언급한 W→A 태스크에서 정규화 효과가 불명확한 원인 규명 필요 (소스 도메인 데이터 부족 + 정규화의 과제약 가능성).

---

## 참고 자료

**주 참고 문헌 (논문 PDF):**
- Cai, G., Wang, Y., He, L., & Zhou, M. (2019). *Unsupervised Domain Adaptation with Adversarial Residual Transform Networks*. arXiv:1804.09578v2.

**논문 내 인용 문헌 (비교 대상):**
- Ganin, Y. et al. (2016). *Domain-adversarial training of neural networks*. JMLR, 17(59).
- Tzeng, E. et al. (2017). *Adversarial Discriminative Domain Adaptation (ADDA)*. CVPR.
- Long, M. et al. (2015). *Learning Transferable Features with Deep Adaptation Networks (DAN)*. ICML.
- Zellinger, W. et al. (2017). *Central Moment Discrepancy (CMD)*. ICLR.
- Arjovsky, M. et al. (2017). *Wasserstein GAN*. ICML.
- Courty, N. et al. (2017). *Optimal Transport for Domain Adaptation*. IEEE TPAMI.
- He, K. et al. (2016). *Deep Residual Learning for Image Recognition (ResNet)*. CVPR.
- Hoffman, J. et al. (2018). *CyCADA: Cycle-Consistent Adversarial Domain Adaptation*. ICML.
- Ben-David, S. et al. (2010). *A theory of learning from different domains*. Machine Learning.

**2020년 이후 비교 연구 (훈련 데이터 기반, 원본 확인 권장):**
- Liang, J. et al. (2020). *Do We Really Need to Access the Source Data? SHOT*. ICML.
- Long, M. et al. (2018). *Conditional Adversarial Domain Adaptation (CDAN)*. NeurIPS.
- Zhang, Y. et al. (2019). *Bridging Theory and Algorithm for Domain Adaptation (MDD)*. ICML.
