# Joint Distribution Optimal Transportation for Domain Adaptation

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

본 논문(Courty et al., NIPS 2017)의 핵심 주장은 다음과 같습니다:

> **소스 도메인의 결합 분포 $\mathcal{P}_s(X, Y)$와 타겟 도메인의 결합 분포 $\mathcal{P}_t(X, Y)$ 사이에는 최적 수송(Optimal Transport)으로 추정 가능한 비선형 변환이 존재하며, 이를 통해 비지도 도메인 적응(Unsupervised Domain Adaptation)이 가능하다.**

기존 방법들이 주변 분포(marginal distribution) $P(X)$만 정렬하거나, 조건부 분포 $P(Y|X)$가 변하지 않는다고 가정한 것과 달리, JDOT은 **결합 분포(joint distribution) 전체**를 동시에 정렬합니다.

### 주요 기여

| 기여 항목 | 내용 |
|---|---|
| **새로운 프레임워크** | 결합 특징/레이블 공간 분포 간 OT를 통한 도메인 적응 (JDOT) |
| **이론적 보장** | 타겟 오류의 상한선(bound) 최소화와 동치임을 증명 |
| **효율적 알고리즘** | 수렴이 보장된 Block Coordinate Descent 기반 최적화 |
| **범용성** | 분류(SVM, NN)와 회귀(KRR) 모두 지원, 다양한 손실 함수 적용 가능 |
| **실험적 검증** | Caltech-Office, Amazon Review, WiFi Localization 데이터셋에서 SOTA 달성 또는 초과 |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

**비지도 도메인 적응(Unsupervised Domain Adaptation)**:
- 소스 도메인에는 레이블이 있는 데이터 $\{(x_i^s, y_i^s)\}_{i=1}^{N_s}$가 존재
- 타겟 도메인에는 레이블이 없는 데이터 $\{x_j^t\}_{j=1}^{N_t}$만 존재
- 두 도메인의 결합 분포가 다름: $\mathcal{P}_s(X,Y) \neq \mathcal{P}_t(X,Y)$

**기존 방법의 한계**:
- **공변량 이동(Covariate Shift)** 가정: $P(X)$만 다르고 $P(Y|X)$는 동일하다고 가정 → 현실에서 성립하지 않는 경우 多
- 주변 분포만 정렬하면 조건부 분포의 차이를 무시하게 됨
- 기존 OT 기반 방법들([14])은 복잡한 매핑을 학습해야 하고 바리센트릭 매핑(barycentric mapping)에 의존

---

### 2.2 제안하는 방법 (수식 포함)

#### Step 1: Kantorovich 최적 수송 (기반)

소스와 타겟의 결합 분포 간 최적 수송 계획(transport plan) $\gamma$:

$$\gamma_0 = \underset{\gamma \in \Pi(\mu_s, \mu_t)}{\arg\min} \int_{\Omega \times \Omega} d(\mathbf{x}_1, \mathbf{x}_2) d\gamma(\mathbf{x}_1, \mathbf{x}_2) \tag{2}$$

여기서 

```math
\Pi(\mu_s, \mu_t) = \{\gamma \in \mathcal{P}(\Omega \times \Omega) \mid p^{+\#}\gamma = \mu_s,\ p^{-\#}\gamma = \mu_t\}
```

#### Step 2: 결합 분포에 대한 OT (JDOT의 핵심)

결합 비용 함수를 정의:

$$\mathcal{D}(\mathbf{x}_1, y_1; \mathbf{x}_2, y_2) = \alpha \cdot d(\mathbf{x}_1, \mathbf{x}_2) + \mathcal{L}(y_1, y_2) \tag{joint cost}$$

- $d(\cdot, \cdot)$: 특징 공간의 거리 (squared Euclidean)
- $\mathcal{L}(\cdot, \cdot)$: 레이블 손실 함수 (hinge, MSE 등)
- $\alpha > 0$: 특징 거리와 레이블 손실 간의 균형 파라미터

이를 통한 결합 분포 간 최적 수송:

$$\gamma_0 = \underset{\gamma \in \Pi(\mathcal{P}_s, \mathcal{P}_t)}{\arg\min} \int_{(\Omega \times \mathcal{C})^2} \mathcal{D}(\mathbf{x}_1, y_1; \mathbf{x}_2, y_2) d\gamma(\mathbf{x}_1, y_1; \mathbf{x}_2, y_2) \tag{3}$$

#### Step 3: 타겟 레이블 부재 문제 해결 — Proxy 분포 도입

타겟 도메인에 레이블이 없으므로, 예측 함수 $f$를 이용한 **proxy 결합 분포**를 정의:

$$\mathcal{P}_t^f = (x, f(x))_{x \sim \mu_t} \tag{4}$$

#### Step 4: JDOT 최적화 목적 함수

$$\min_{f,\ \gamma \in \Delta} \sum_{i,j} \mathcal{D}(x_i^s, y_i^s;\ x_j^t, f(x_j^t)) \cdot \gamma_{ij} \equiv \min_f W_1(\hat{\mathcal{P}}_s, \hat{\mathcal{P}}_t^f) \tag{5}$$

정규화 항을 포함한 실용적 형태:

$$\min_{f \in \mathcal{H},\ \gamma \in \Delta} \sum_{i,j} \gamma_{i,j} \left[ \alpha \cdot d(x_i^s, x_j^t) + \mathcal{L}(y_i^s, f(x_j^t)) \right] + \lambda \Omega(f) \tag{6}$$

여기서 $W_1$은 손실 $\mathcal{D}$에 대한 1-Wasserstein 거리이며, $\Omega(f)$는 과적합 방지를 위한 정규화 항입니다.

---

### 2.3 모델 구조

#### 알고리즘: Block Coordinate Descent (BCD)

JDOT은 $\gamma$와 $f$를 번갈아 최적화하는 **교대 최적화(Alternating Optimization)**를 사용합니다.

```
초기화: f₀ (소스 도메인에서 학습된 초기 예측 함수)
반복 (수렴까지):
  Step 1: f 고정 → γ 업데이트 (OT 문제 풀기)
  Step 2: γ 고정 → f 업데이트 (가중치 학습 문제 풀기)
```

**Step 1: $f$ 고정 시 $\gamma$ 업데이트**

비용 행렬 $C_{ij} = \alpha d(x_i^s, x_j^t) + \mathcal{L}(y_i^s, f(x_j^t))$를 이용한 표준 OT 문제:

$$\gamma^* = \underset{\gamma \in \Delta}{\arg\min} \sum_{i,j} C_{ij} \gamma_{ij}$$

**Step 2: $\gamma$ 고정 시 $f$ 업데이트**

$$\min_{f \in \mathcal{H}} \sum_{i,j} \gamma_{i,j} \mathcal{L}(y_i^s, f(x_j^t)) + \lambda \Omega(f) \tag{7}$$

**회귀(최소 제곱)의 경우** ($\mathcal{L}$ = squared loss):

$$\min_{f \in \mathcal{H}} \sum_j \frac{1}{n_t} \|\hat{y}_j - f(x_j^t)\|^2 + \lambda\|f\|^2 \tag{8}$$

여기서 $\hat{y}\_j = n_t \sum_i \gamma_{i,j} y_i^s$ (소스 레이블의 가중 평균)

**분류(Hinge Loss)의 경우** (One-vs-All):

$$\min_{f_k \in \mathcal{H}} \sum_{j,k} \hat{P}_{j,k} \mathcal{L}(1, f_k(x_j^t)) + (1-\hat{P}_{j,k})\mathcal{L}(-1, f_k(x_j^t)) + \lambda \sum_k \|f_k\|^2 \tag{9}$$

여기서 $\hat{\mathbf{P}} = \frac{1}{N_t} \gamma^\top \mathbf{P}^s$ (수송 행렬로 전파된 클래스 비율 행렬)

**수렴 보장**: Grippo et al. [27]의 2-block Gauss-Seidel 수렴 정리에 의해, 알고리즘이 생성하는 수열 $\{\gamma^k, f^k\}$의 모든 극한점은 문제 (6)의 임계점(critical point)임이 보장됩니다.

---

### 2.4 성능 향상 및 한계

#### 성능 향상 결과

**Caltech-Office 분류 (표 1)**:

| 방법 | 평균 정확도 | 평균 순위 |
|---|---|---|
| Base (no adaptation) | 83.92% | 5.33 |
| SA | 85.23% | 4.00 |
| OT-MM | 86.95% | 2.83 |
| **JDOT** | **89.04%** | **2.50** |

**Amazon Review 분류 (표 2)**:
- JDOT(Hinge): 평균 **0.787**, 12개 태스크 중 11개에서 DANN 초과

**WiFi Localization 회귀 (표 3)**:
- Transfer across devices: 평균 **>98%** (최고 경쟁 방법 대비 10점 이상 우세)

#### 한계점

1. **계산 복잡도**: OT 문제의 복잡도가 $O(N_s \cdot N_t)$로, 대규모 데이터셋에서 계산 비용이 높음
2. **$\alpha$ 파라미터 민감성**: 특징 거리와 레이블 손실 간 균형 조정이 필요하며, 일부 경우 cross-validation 필요
3. **프록시 가정의 한계**: $\mathcal{P}_t^f$가 실제 $\mathcal{P}_t$를 잘 근사해야 한다는 가정에 의존
4. **비볼록 최적화**: BCD는 전역 최적해가 아닌 임계점으로 수렴
5. **대규모 확장성**: 수백만 샘플에 대한 직접 적용이 어려움 (entropic regularization으로 부분 완화)
6. **반지도 학습 미지원**: 소스 도메인의 비레이블 샘플 활용 불가 (저자들이 향후 연구로 언급)

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 이론적 일반화 보장: Theorem 3.1

논문의 핵심 이론적 기여는 JDOT이 타겟 오류의 상한을 최소화함을 증명한 것입니다.

**Probabilistic Transfer Lipschitzness (PTL) 정의**:

레이블 함수 $f$와 결합 분포 $\Pi(\mu_s, \mu_t)$가 모든 $\lambda > 0$에 대해 다음을 만족하면 $\phi$-Lipschitz transferable:

$$\Pr_{(x_1, x_2) \sim \Pi(\mu_s, \mu_t)} \left[ |f(x_1) - f(x_2)| > \lambda d(x_1, x_2) \right] \leq \phi(\lambda)$$

**Theorem 3.1** (타겟 오류 상한):

$\alpha = k\lambda$로 설정 시, 최소 $1-\delta$ 확률로:

```math
\text{err}_T(f) \leq \underbrace{W_1(\hat{\mathcal{P}}_s, \hat{\mathcal{P}}_t^f)}_{\text{JDOT 목적함수}} + \underbrace{\sqrt{\frac{2}{c'}\log\left(\frac{2}{\delta}\right)}\left(\frac{1}{\sqrt{N_S}} + \frac{1}{\sqrt{N_T}}\right)}_{\text{샘플링 오차}} + \underbrace{\text{err}_S(f^*) + \text{err}_T(f^*)}_{\text{이상적 결합 오차}} + \underbrace{kM\phi(\lambda)}_{\text{PTL 위반 확률}}
```

**각 항의 의미**:

| 항 | 의미 | 일반화와의 관련성 |
|---|---|---|
| $W_1(\hat{\mathcal{P}}_s, \hat{\mathcal{P}}_t^f)$ | JDOT 최소화 목적 | 이 항을 최소화하면 타겟 오류 자동 감소 |
| 샘플링 오차 항 | $N_S, N_T$ 증가 시 감소 | 더 많은 데이터로 일반화 향상 |
| $\text{err}_S(f^\*) + \text{err}_T(f^*)$ | 최적 함수의 결합 오차 | 두 도메인에서 모두 잘 예측하는 $f^*$ 존재 시 적응 가능 |
| $kM\phi(\lambda)$ | PTL 가정 위반 확률 | 가정이 현실과 가까울수록 상한이 타이트 |

### 3.2 일반화 성능 향상 메커니즘

**1. 결합 분포 정렬의 우수성**

단순 주변 분포 정렬 대비:

$$\text{기존 OT DA}: \min_\gamma \sum_{ij} d(x_i^s, x_j^t)\gamma_{ij} \quad \text{(feature space만 정렬)}$$

$$\text{JDOT}: \min_{f,\gamma} \sum_{ij} \left[\alpha d(x_i^s, x_j^t) + \mathcal{L}(y_i^s, f(x_j^t))\right]\gamma_{ij} \quad \text{(feature + label 동시 정렬)}$$

레이블 정보를 수송 계획에 포함시킴으로써, **의미론적으로 유사한 샘플끼리 매칭**되어 더 나은 일반화를 달성합니다.

**2. 바리센트릭 매핑 탈피**

기존 OT 기반 DA는 $T(x_i^s) = \sum_j \frac{\gamma_{ij}}{\mu_s(x_i^s)} x_j^t$와 같은 바리센트릭 매핑을 학습한 후 분류기를 별도 학습하지만, JDOT은 수송 계획과 예측 함수를 동시에 최적화하여 **end-to-end** 방식으로 일반화 성능을 직접 최적화합니다.

**3. 파라미터 $\alpha$와 일반화**

- $\alpha \to +\infty$: 특징 공간 정렬 지배 (기존 OT DA와 동일)
- $\alpha$ 적절: 특징 + 레이블 균형 정렬 → 최적 일반화
- $\alpha$는 PTL 가정의 Lipschitz 상수와 연결되어 이론적 의미 보유

**4. 다양한 함수 클래스 지원**

$\mathcal{H}$가 RKHS이면 representer theorem에 의해 $N_t$개의 파라미터로 축소 가능 → 고차원에서도 효율적 일반화

---

## 4. 미래 연구에의 영향 및 고려 사항

### 4.1 미래 연구에의 영향

**1. OT 기반 딥러닝 DA의 촉매제**

JDOT은 이후 딥러닝과 OT를 결합한 연구들의 이론적 토대가 되었습니다:
- DeepJDOT (Damodaran et al., ECCV 2018): JDOT의 프레임워크를 딥 신경망에 통합
- Wasserstein 거리 기반 GAN 훈련과의 연결

**2. 결합 분포 관점의 확산**

단순 특징 분포 정렬을 넘어 레이블 공간을 함께 고려하는 패러다임을 확립했습니다.

**3. 이론적 프레임워크 확장**

PTL(Probabilistic Transfer Lipschitzness) 개념은 이후 DA 이론 연구에 영향을 미쳤으며, 다양한 분포 이동 시나리오에서 일반화 보장 연구로 확장되었습니다.

### 4.2 향후 연구 시 고려할 점

**논문 저자들이 직접 제안한 방향**:
- 반지도 학습(semi-supervised) 확장
- 확률적(stochastic) 최적화 기법 도입
- PTL의 이론적 심화 연구
- 가설 클래스와 수송 계획 공간의 복잡도를 반영한 보장

**추가적으로 고려해야 할 점**:

1. **확장성(Scalability)**: 대규모 데이터셋 적용을 위한 미니배치 OT, 확률적 Sinkhorn 등의 활용
2. **표현 학습과의 통합**: 특징 추출기(feature extractor)와 JDOT의 end-to-end 통합
3. **다중 소스 도메인**: 여러 소스 도메인에서의 JDOT 확장
4. **분포 이동의 다양한 유형**: covariate shift 외 label shift, concept drift 등에 대한 적용
5. **비파라미터적 함수 클래스**: 더 복잡한 신경망 구조에서의 이론적 보장

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 DeepJDOT (Damodaran et al., ECCV 2018)

JDOT의 직접적인 확장으로, 딥 특징 공간에서 OT를 수행:

$$\min_{\theta, \gamma} \sum_{i,j} \gamma_{ij}\left[\alpha \|g_\theta(x_i^s) - g_\theta(x_j^t)\|^2 + \mathcal{L}(y_i^s, f(g_\theta(x_j^t)))\right]$$

- $g_\theta$: 딥러닝 특징 추출기
- 특징 공간과 레이블 공간을 **동시에 딥러닝으로 학습**
- JDOT 대비 표현 학습과의 통합으로 성능 향상

### 5.2 Computational Optimal Transport 기반 연구들

| 연구 | 핵심 아이디어 | JDOT과의 차이 |
|---|---|---|
| **Sinkhorn AutoDiff** (Genevay et al., 2018) | 엔트로픽 정규화 OT의 자동 미분 | 계산 효율성 향상 |
| **Sliced Wasserstein** (Kolouri et al., 2019) | 1D 투영을 통한 근사 Wasserstein | $O(N \log N)$ 복잡도 |
| **Unbalanced OT** (Séjourné et al., 2019) | 마진 제약 완화 | 이상치(outlier)에 강건 |

### 5.3 JDOT 이후 OT 기반 DA 발전 방향

```
JDOT (2017, NIPS)
    ↓
DeepJDOT (2018, ECCV) — 딥러닝 통합
    ↓
Minimax OT (2020~) — GAN 결합
    ↓
Partial OT DA (2020~) — 부분 도메인 적응
    ↓
Multi-source OT DA (2021~) — 다중 소스
    ↓
OT for Federated Learning (2022~) — 연합 학습
```

### 5.4 비교 요약표

| 방법 | 분포 정렬 | 레이블 활용 | 이론 보장 | 확장성 | 비고 |
|---|---|---|---|---|---|
| **JDOT (2017)** | 결합 분포 | OT 내 직접 | ✅ (PTL 기반) | 중간 | 본 논문 |
| DANN (2016) | 주변 분포 | 간접 | 부분적 | 높음 | GAN 기반 |
| DeepJDOT (2018) | 결합 분포 | OT 내 직접 | 부분적 | 높음 | 딥러닝 통합 |
| CDAN (2018) | 조건부 분포 | 조건부 정렬 | 부분적 | 높음 | 멀티리니어 맵 |
| OT-DA w/ Unbalanced (2020~) | 결합 분포 | OT 내 직접 | 부분적 | 높음 | 이상치 강건 |

---

## 참고 자료

**논문 원문 (제공된 PDF)**:
- **Courty, N., Flamary, R., Habrard, A., & Rakotomamonjy, A.** (2017). *Joint distribution optimal transportation for domain adaptation*. Advances in Neural Information Processing Systems (NIPS 2017).

**논문 내 인용 문헌 (주요)**:
- [14] Courty, N., Flamary, R., Tuia, D., & Rakotomamonjy, A. (2016). *Optimal transport for domain adaptation*. IEEE Transactions on Pattern Analysis and Machine Intelligence.
- [11] Ganin, Y. et al. (2016). *Domain-adversarial training of neural networks*. Journal of Machine Learning Research, 17(59):1–35.
- [25] Cuturi, M. (2013). *Sinkhorn distances: Lightspeed computation of optimal transport*. NIPS.
- [24] Ben-David, S. et al. (2010). *A theory of learning from different domains*. Machine Learning, 79(1-2):151–175.
- [27] Grippo, L. & Sciandrone, M. (2000). *On the convergence of the block nonlinear Gauss–Seidel method under convex constraints*. Operations Research Letters, 26(3):127–136.

**후속 연구 (2020년 이후)**:
- Damodaran, B. B. et al. (2018). *DeepJDOT: Deep joint distribution optimal transport for unsupervised domain adaptation*. ECCV 2018. *(직접 확인한 참고 자료)*
- Peyré, G. & Cuturi, M. (2019). *Computational Optimal Transport*. Foundations and Trends in Machine Learning. *(직접 확인한 참고 자료)*

> ⚠️ **정확도 주의**: 2020년 이후 특정 후속 논문들의 세부 수식 및 정확한 성능 수치는 해당 논문들을 직접 확인하지 않은 부분이 포함되어 있으므로, 정확한 비교를 위해서는 각 논문 원문을 참조하시기 바랍니다. DeepJDOT은 2018년 ECCV에 발표된 것으로 확인되며, JDOT과의 관계는 명확합니다.
