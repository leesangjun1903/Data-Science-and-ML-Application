# Unsupervised Domain Adaptation via Calibrating Uncertainties

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장
소스 도메인에서 학습된 모델은 타겟 도메인의 미지 데이터에 대해 **더 높은 불확실성(uncertainty)**을 예측하는 경향이 있다. 따라서 **두 도메인 간 예측 불확실성을 조정(calibrate)** 함으로써 효과적인 비지도 도메인 적응(UDA)이 가능하다.

### 주요 기여 (3가지)

| 기여 | 내용 |
|------|------|
| **RER Framework** | Rényi 엔트로피 기반 일반화된 불확실성 정규화 프레임워크 제안 |
| **BNN 도입** | 변분 베이즈 신경망으로 신뢰할 수 있는 불확실성 추정 |
| **GVR 제안** | 그래디언트 분산 정규화(Gradient Variance Regularization)를 모델/목적함수 무관 플러그인 정규화기로 제안 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**비지도 도메인 적응(UDA)** 상황에서:
- 레이블이 있는 소스 도메인 $\mathcal{D}\_S = \{x^{(s)}, y^{(s)}\}_{s \in S}$
- 레이블이 없는 타겟 도메인 $\mathcal{D}\_T = \{x^{(t)}\}_{t \in T}$

**핵심 문제점:**
1. 전통적인 DNN은 잘못된 예측에도 **과도한 확신(overconfidence)**을 부여함
2. 불확실성을 신뢰성 있게 정량화하지 못해 **pseudo-label 노이즈** 발생
3. Shannon 엔트로피 직접 최적화 시 **locally-Lipschitz 조건** 충족 필요

---

### 2.2 제안하는 방법 (수식 포함)

#### (A) Rényi 엔트로피 정의

이산 확률분포 $P = (P_1, \ldots, P_K)$에 대한 Rényi 엔트로피 (order $\alpha > 0$):

$$H_\alpha(P) = \frac{1}{1-\alpha} \log\left(\sum_k P_k^\alpha\right) \tag{1}$$

- $\alpha \to 1$: Shannon 엔트로피로 수렴
- $\alpha \to \infty$: min-엔트로피 $H_\infty(P) = -\log \max_k P_k$

#### (B) Bayesian Neural Network (BNN)

로짓을 가우시안으로 가정하고 재매개변수화 트릭(reparameterization trick) 적용:

$$\hat{f}(x) = \mu_\theta(x) + \sigma_\theta(x)\epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

Monte Carlo 샘플링 ($M$개)으로 예측 확률 근사:

$$P(y|x; \theta) = \frac{1}{M} \sum_{m=1}^{M} \text{softmax}(\hat{f}^{(m)}(x)) \tag{2}$$

#### (C) 변분 추론 (ELBO)

사후 분포 $P(w|\mathcal{D})$를 $Q_\theta(w)$로 근사:

$$ELBO = \underbrace{\mathbb{E}_{Q_\theta(w)} \log P(y|x, w)}_{\text{(I) Cross-Entropy}} - \underbrace{D_{KL}(Q_\theta \| P(w))}_{\text{(II) KL 정규화}} \tag{3}$$

#### (D) Rényi 엔트로피 정규화 (RER) - 핵심 목적함수

소스에서는 CE 손실 최소화, 타겟에서는 엔트로피 제약:

$$\min_\theta \mathcal{L}_{CE} = \frac{1}{|S|} \sum_{s \in S} H_{CE}(y^{(s)}, P(y|x^{(s)}; \theta))$$
$$\text{s.t.} \quad \frac{1}{|T|} \sum_{t \in T} H_\alpha(P(y|x^{(t)}; \theta)) \leq C \tag{5}$$

라그랑지안으로 변환 후 상한:

$$\mathcal{L}_\alpha = \frac{1}{|S|} \sum_{s \in S} H_{CE}(y^{(s)}, P(y|x^{(s)}; \theta)) + \frac{\beta}{|T|} \sum_{t \in T} H_\alpha(P(y|x^{(t)}; \theta)) \tag{7}$$

#### (E) Self-Training과의 통합 ($\alpha \to \infty$)

$$\mathcal{L}_\infty = \frac{1}{|S|} \sum_{s \in S} H_{CE}(y^{(s)}, P(y|x^{(s)}; \theta)) + \frac{\beta}{|T|} \sum_{t \in T} H_{CE}(\hat{y}^{(t)}, P(y|x^{(t)}; \theta)) \tag{8}$$

여기서 $\hat{y}^{(t)} = \text{onehot}(\arg\max_{k} P(y_k|x^{(t)}; \theta))$는 pseudo-label.

Shannon 엔트로피와 min-엔트로피의 관계:

$$H_1(P) = -\sum_k P_k \log(P_k) \geq -\log(\max_k P_k) = H_\infty(P) = H_{CE}(\hat{y}, P) \tag{9}$$

#### (F) Gradient Variance Regularization (GVR)

미니배치 $\mathcal{B}_i$에서 한 스텝 적응된 파라미터:

$$\theta'_i = \theta - \eta \nabla_\theta \mathcal{L}_{\mathcal{B}_i}(f_\theta)$$

파라미터 분산 (그래디언트 분산):

$$\text{Var}(\{\theta'_i\}) = \text{trace}(\text{Cov}(\{\text{vec}(\theta'_i)\})) \tag{4}$$

$$\text{where} \quad \text{Cov}(\{\text{vec}(\theta'_i)\}) = \eta^2 \text{Cov}(\{\text{vec}(\nabla_\theta \mathcal{L}_{\mathcal{B}_i}(f_\theta))\})$$

최종 업데이트 (GVR 포함):

$$\theta \leftarrow \theta - \eta' \nabla_\theta \left(\sum_i \mathcal{L}_{\mathcal{B}_i}(f_\theta) - \lambda \text{Var}(\Theta')\right)$$

---

### 2.3 모델 구조

```
[소스 도메인 (레이블 있음)]           [타겟 도메인 (레이블 없음)]
        ↓                                      ↓
  ELBO 사전학습                        Rényi 엔트로피 제약
  (BNN: μ_θ, σ_θ 출력)               (불확실성 보정)
        ↓                                      ↓
  Cross-Entropy Loss                   H_α 최소화 / pseudo-label
        ↓                                      ↓
        └──────── 통합 목적함수 L_α ───────────┘
                          ↓
              GVR (그래디언트 분산 최대화)
                    [정규화 항]
```

- **Base model**: DTN (digits), ResNet-101 (VisDA17)
- **Bayesian 변형**: BDTN = DTN + log-variance 예측 분류기 추가
- **CBST (Class-Balanced Self-Training)**을 백본으로 활용

---

### 2.4 성능 향상

#### MNIST → USPS

| 모델 | 타겟 정확도 (%) | 향상 (%) |
|------|---------------|---------|
| Source-DTN (baseline) | 83.94 | - |
| RER-∞ | 93.57 ± 0.30 | +9.63 |
| RER-∞-GVR | 93.88 ± 0.14 | +9.94 |
| BRER-∞ | 94.42 ± 0.12 | +9.53 |
| **BRER-∞-GVR** | **94.53 ± 0.23** | **+9.64** |

#### SVHN → MNIST

| 모델 | 타겟 정확도 (%) | 향상 (%) |
|------|---------------|---------|
| Source-DTN (baseline) | 64.48 | - |
| RER-∞-GVR | 90.31 ± 2.31 | +25.83 |
| BRER-∞ | 96.06 ± 0.68 | +25.08 |
| **BRER-∞-GVR** | **96.38 ± 0.05** | **+25.40** |

#### VisDA17 (Synthetic → Real, 12 classes)

| 모델 | 평균 정확도 (%) | 향상 (%) |
|------|--------------|---------|
| Source-Res101 | 48.02 | - |
| MMD | 61.1 | - |
| GTA-Res152 | 77.1 | - |
| RER-∞ | 76.81 ± 2.73 | +28.79 |
| **BRER-∞** | **80.59 ± 1.39** | **+34.56** |

---

### 2.5 한계점

1. **계산 비용**: BNN의 Monte Carlo 샘플링($M$회)으로 인한 추론 시간 증가
2. **GVR 메모리 문제**: VisDA17에서 GPU 메모리 초과로 GVR 미적용
3. **하이퍼파라미터 민감성**: $\beta$, $\alpha$, $\lambda$ 등 다수의 하이퍼파라미터 수동 설정 필요
4. **GVR 불안정성**: RER-1($\alpha=1$) 설정에서 GVR이 오히려 성능 저하 유발
5. **실험 범위 제한**: 디지털 데이터셋과 VisDA17에만 검증, 자연어 처리 등 다른 모달리티 미검증

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 불확실성 보정의 일반화 기여

**Rényi 엔트로피 정규화**는 타겟 도메인에 대한 예측 분포를 **저엔트로피(고확신)** 상태로 유도함으로써:

$$\frac{1}{|T|} \sum_{t \in T} H_\alpha(P(y|x^{(t)}; \theta)) \leq C$$

이는 도메인 불변 특징(domain-invariant features)을 암묵적으로 학습하게 만들어 **일반화 성능 향상**에 기여한다.

### 3.2 BNN의 일반화 기여

BNN은 베이지안 앙상블(Bayesian ensembling)을 본질적으로 수행:

$$P(y|x) = \mathbb{E}_{P(w|\mathcal{D})} P(y|x, w)$$

- **과적합 방지**: KL 정규화 항 $D_{KL}(Q_\theta \| P(w))$이 weight의 사전분포로부터 과도한 이탈 방지
- **신뢰 가능한 pseudo-label**: 불확실성이 높은 샘플을 걸러내어 self-training의 노이즈 감소

### 3.3 GVR의 일반화 기여

MAML과의 비교에서:

$$\mathcal{L}_{\mathcal{B}_i}(f_{\theta'_i}) \approx \mathcal{L}_{\mathcal{B}_i}(f_\theta) - \eta \|\nabla_\theta \mathcal{L}_{\mathcal{B}_i}(f_\theta)\|_2^2 \tag{10}$$

- **MAML**: 그래디언트 L2 노름 **최대화** → 빠른 적응 (task sensitivity↑)
- **GVR**: 그래디언트 **분산 최대화** → **나쁜 local minima 탈출** → 평탄한 loss landscape 탐색

평탄한 손실 곡면(flat minima)은 일반화 성능과 직결되며, GVR은 이를 간접적으로 달성한다.

### 3.4 프레임워크의 범용성

- $\alpha$ 값 조정으로 Shannon 엔트로피 정규화 → self-training까지 **하나의 통합 프레임워크**로 포괄
- GVR은 **모델 무관(model-agnostic)**, **목적함수 무관(objective-agnostic)** → 다양한 UDA 방법에 플러그인 적용 가능
- 지도학습 정규화에도 응용 가능성 있음

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 연구에 미치는 영향

#### (1) 불확실성 기반 UDA의 새 패러다임 제시
전통적 adversarial alignment 대신 **불확실성 보정**이라는 새로운 관점을 제시함으로써, 적대적 학습의 불안정성을 회피하는 대안적 방향을 개척했다.

#### (2) Rényi 엔트로피의 통합 프레임워크
Shannon 엔트로피 정규화, self-training을 **단일 프레임워크** 하에 통합하여, 기존 방법들이 RER의 특수 케이스임을 이론적으로 규명했다.

#### (3) BNN × UDA 연구 촉진
Bayesian 방법을 UDA에 체계적으로 도입한 초기 연구로, 이후 **신뢰 가능한 도메인 적응(trustworthy DA)** 연구의 기반을 마련했다.

#### (4) GVR의 광범위한 적용 가능성
메타러닝(MAML)과의 연결고리를 통해, 도메인 적응 맥락에서 **최적화 동역학(optimization dynamics) 정규화**라는 새로운 연구 방향을 시사했다.

---

### 4.2 2020년 이후 관련 최신 연구 비교 분석

본 논문과 직접 연결되는 후속 연구들을 비교한다. (**주의**: 아래 논문들은 제가 사전학습 데이터를 기반으로 언급하는 것으로, 제공된 PDF에는 포함되지 않은 외부 문헌입니다.)

#### 비교 분석표

| 논문 | 핵심 방법 | RER과의 관계 | 한계 극복 여부 |
|------|----------|------------|-------------|
| **SHOT** (Liang et al., ICML 2020) | 정보 극대화 + pseudo-labeling (소스 모델 동결) | 엔트로피 최소화를 RER-1로 해석 가능 | 소스 데이터 불필요 → 프라이버시 문제 해결 |
| **NRC** (Yang et al., NeurIPS 2021) | 이웃 관계 일관성 기반 pseudo-label 생성 | RER-∞의 noisy pseudo-label 문제 개선 | GVR 없이도 안정적 self-training |
| **SDAT** (Rangwani et al., ICML 2022) | 평탄한 손실 곡면(SAM) + 도메인 적응 | GVR과 유사한 flat minima 추구 | 이론적 일반화 경계 증명 |
| **UniDA** (You et al., CVPR 2021) | 통합 도메인 적응 (open-set 포함) | RER 프레임워크 확장 가능성 | 클래스 불일치 문제 다룸 |
| **T3A** (Iwasawa & Matsuo, NeurIPS 2021) | 테스트 시간 적응 + 프로토타입 | 추론 시 불확실성 활용 | 온라인 적응으로 확장 |

#### 핵심 비교: SHOT vs. RER

| 항목 | RER (본 논문) | SHOT (2020) |
|------|-------------|-------------|
| 불확실성 정량화 | Rényi 엔트로피 | Shannon 엔트로피 + 상호정보 |
| 소스 데이터 필요 | 필요 | 불필요 (소스-free) |
| BNN 사용 | 명시적 BNN | 없음 |
| 실용성 | GPU 메모리 제약 | 경량화 |

#### 핵심 비교: SDAT vs. GVR

| 항목 | GVR (본 논문) | SDAT (2022) |
|------|-------------|-------------|
| 평탄한 minima 추구 방식 | 그래디언트 분산 최대화 | SAM(Sharpness-Aware Minimization) |
| 이론적 보장 | MAML 연결로 직관적 설명 | PAC-Bayes 일반화 경계 증명 |
| 계산 복잡도 | $O(B^2)$ | $O(2 \times \text{gradient steps})$ |

---

### 4.3 향후 연구 시 고려할 점

#### (1) 소스-프리(Source-Free) UDA로의 확장
현재 RER은 소스 도메인 데이터를 학습 시 계속 사용한다. **데이터 프라이버시** 관점에서 소스 데이터 없이 사전학습 모델만으로 타겟 적응하는 방향으로 확장이 필요하다.

#### (2) $\alpha$의 자동 조정 메커니즘
현재 $\alpha$는 수동 설정 하이퍼파라미터다. 학습 과정에서 도메인 갭과 현재 모델 상태에 따라 **$\alpha$를 동적으로 조정**하는 adaptive 메커니즘 연구가 필요하다:

$$\alpha_t = f(\text{domain gap}_t, \text{model confidence}_t)$$

#### (3) GVR의 이론적 정당화 강화
GVR과 일반화 성능 간의 **형식적 이론(PAC-Bayes, Rademacher complexity 등)** 연결이 부족하다. SDAT처럼 수학적 일반화 경계를 도출하는 연구가 필요하다.

#### (4) 다중 소스/타겟 도메인으로 확장
단일 소스→타겟 설정을 넘어 **다중 소스 도메인** 또는 **지속적 도메인 적응(Continual DA)** 시나리오에서의 불확실성 보정 방법론 연구가 필요하다.

#### (5) 대형 언어/비전 모델(LLM/VLM)과의 결합
CLIP, DINOv2 등 대형 사전학습 모델의 등장으로, RER 프레임워크를 **프롬프트 튜닝** 또는 **adapter 기반** UDA에 통합하는 연구가 유망하다.

#### (6) 불확실성 분해의 정교화
Aleatoric과 Epistemic 불확실성을 분리 추정하되, 두 유형에 따라 **차별적 보정 전략**을 적용하는 연구가 필요하다:

$$H_\alpha(P) = \underbrace{H_{\alpha}^{\text{aleatoric}}}_{\text{데이터 노이즈}} + \underbrace{H_{\alpha}^{\text{epistemic}}}_{\text{모델 불확실성}}$$

---

## 참고 자료

### 본 논문
- **Han, L., Zou, Y., Gao, R., Wang, L., & Metaxas, D.** (2019). *Unsupervised Domain Adaptation via Calibrating Uncertainties*. arXiv:1907.11202v1. (제공된 PDF)

### 본 논문 인용 문헌 (PDF 내 References)
- Zou, Y., Yu, Z., Kumar, B. V., & Wang, J. (2018). *Unsupervised Domain Adaptation for Semantic Segmentation via Class-Balanced Self-Training*. ECCV 2018.
- Gal, Y., & Ghahramani, Z. (2016). *Dropout as a Bayesian Approximation*. ICML 2016.
- Kendall, A., & Gal, Y. (2017). *What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?* NeurIPS 2017.
- Grandvalet, Y., & Bengio, Y. (2004). *Semi-supervised Learning by Entropy Minimization*. NeurIPS 2005.
- Finn, C., Abbeel, P., & Levine, S. (2017). *Model-Agnostic Meta-Learning*. ICML 2017.
- Vu, T.-H., et al. (2018). *ADVENT: Adversarial Entropy Minimization for Domain Adaptation*. arXiv:1811.12833.

### 비교 분석 참고 문헌 (사전학습 지식 기반, PDF 외부)
- Liang, J., et al. (2020). *Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation*. ICML 2020. (SHOT)
- Rangwani, H., et al. (2022). *A Closer Look at Smoothness in Domain Adversarial Training*. ICML 2022. (SDAT)
- Yang, S., et al. (2021). *Exploiting the Intrinsic Neighborhood Structure for Source-free Domain Adaptation*. NeurIPS 2021. (NRC)
- Iwasawa, Y., & Matsuo, Y. (2021). *Test-Time Classifier Adjustment Module for Model-Agnostic Domain Generalization*. NeurIPS 2021. (T3A)

> ⚠️ **정확도 관련 고지**: 비교 분석 섹션의 2020년 이후 논문들은 제공된 PDF에 포함되지 않은 외부 문헌으로, 저의 사전학습 지식을 기반으로 기술하였습니다. 세부 수치나 방법론적 세부사항은 원문 확인을 권장드립니다.
