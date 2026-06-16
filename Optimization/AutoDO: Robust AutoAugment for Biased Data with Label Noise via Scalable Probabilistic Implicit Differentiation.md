# AutoDO: Robust AutoAugment for Biased Data with Label Noise via Scalable Probabilistic Implicit Differentiation

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

AutoDO의 핵심 주장은 기존 AutoAugment 계열의 **공유 정책(shared-policy) 기반 데이터 증강 방법들이 편향(biased)되고 노이즈 레이블(noisy label)을 포함한 훈련 데이터에 취약**하다는 것입니다. 이를 해결하기 위해 논문은 AutoAugment를 **일반화된 자동 데이터셋 최적화(AutoDO)** 문제로 재정의합니다.

### 주요 기여

| 기여 항목 | 내용 |
|-----------|------|
| **Per-point 하이퍼파라미터** | 모든 훈련 데이터 포인트 각각에 대해 독립적인 하이퍼파라미터 추정 |
| **Joint 최적화** | 증강(augmentation), 손실 가중치(loss weight), 소프트 레이블(soft-label)을 동시에 최적화 |
| **확장 가능한 암묵적 미분** | Fisher 정보를 이용한 이론적 해석 제공, 복잡도가 데이터셋 크기에 선형적으로 확장 |
| **성능 향상** | 편향 + 노이즈 환경에서 최대 **9.3%** 개선, SVHN 소수 클래스에서 최대 **36.6%** 향상 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**공유 정책 DA의 딜레마(Shared-Policy DA Dilemma)**:

기존 방법들(AA, FAA, DADA 등)은 모든 훈련 데이터에 동일한 증강 정책을 적용합니다. 이는 다음과 같은 문제를 야기합니다:

- **클래스 불균형(Class Imbalance)**: 소수 클래스는 과소 증강(under-augmented), 다수 클래스는 과도 증강(over-augmented)
- **노이즈 레이블(Noisy Labels)**: 잘못된 레이블을 가진 데이터에 과적합(overfitting) 발생
- **분포 불일치(Distribution Shift)**: 증강된 훈련 데이터의 분포가 테스트 데이터 분포와 불일치

### 2.2 제안하는 방법 (수식 포함)

#### 2.2.1 Bilevel 최적화 문제 정의

**Inner Objective (내부 목적함수)**:

$$\boldsymbol{\theta}^*(\boldsymbol{\lambda}) := \arg\min_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\lambda}, \boldsymbol{\theta}) \tag{1}$$

여기서 훈련 손실은 경험적 위험(empirical risk)으로 정의됩니다:

$$\mathcal{L}(\boldsymbol{\lambda}, \boldsymbol{\theta}) = \sum_{i \in \mathbb{N}} \mathcal{L}(\boldsymbol{y}_i, f(\boldsymbol{x}_i, \boldsymbol{\theta}(\boldsymbol{\lambda})))/N$$

**Outer Objective (외부 목적함수)**:

```math
\boldsymbol{\lambda}^* := \arg\min_{\boldsymbol{\lambda}} \mathcal{L}^*_v(\boldsymbol{\lambda}) = \arg\min_{\boldsymbol{\lambda}} \mathcal{L}_v(\boldsymbol{\lambda}, \boldsymbol{\theta}^*(\boldsymbol{\lambda}))
```

여기서 검증 손실은:

$$\mathcal{L}_v(\boldsymbol{\lambda}, \boldsymbol{\theta}^*(\boldsymbol{\lambda})) = \sum_{i \in \mathbb{M}} \mathcal{L}(\boldsymbol{y}^v_i, f(\boldsymbol{x}^v_i, \boldsymbol{\theta}^*(\boldsymbol{\lambda})))/M$$

#### 2.2.2 암묵적 미분 (Implicit Differentiation)

Chain rule을 이용한 외부 목적함수의 그래디언트:

```math
\frac{\partial \mathcal{L}_v}{\partial \boldsymbol{\lambda}} = \frac{\partial \mathcal{L}_v}{\partial \boldsymbol{\theta}^*(\boldsymbol{\lambda})} \frac{\partial \boldsymbol{\theta}^*(\boldsymbol{\lambda})}{\partial \boldsymbol{\lambda}}
```

> $\partial \mathcal{L}_v / \partial \boldsymbol{\lambda} = 0$ 임을 주의 ($\mathcal{L}_v$는 명시적으로 $\boldsymbol{\lambda}$에 의존하지 않음)

**암묵 함수 정리(IFT)**를 이용하여 $\partial \boldsymbol{\theta}^*(\boldsymbol{\lambda})/\partial \boldsymbol{\lambda}$를 정의:

$$\frac{\partial \boldsymbol{\theta}(\boldsymbol{\lambda})}{\partial \boldsymbol{\lambda}} = -\left[J_{\boldsymbol{\theta}} S(\boldsymbol{\lambda}, \boldsymbol{\theta})\right]^{-1} J_{\boldsymbol{\lambda}} S(\boldsymbol{\lambda}, \boldsymbol{\theta}) \tag{4}$$

위를 결합하면 핵심 수식:

$$\frac{\partial \mathcal{L}_v}{\partial \boldsymbol{\lambda}} = -\frac{\partial \mathcal{L}_v}{\partial \boldsymbol{\theta}} \left[\frac{\partial^2 \mathcal{L}}{\partial \boldsymbol{\theta} \partial \boldsymbol{\theta}^T}\right]^{-1} \frac{\partial^2 \mathcal{L}}{\partial \boldsymbol{\theta} \partial \boldsymbol{\lambda}^T} \tag{5}$$

여기서 $\boldsymbol{H}_{\boldsymbol{\theta}}^{-1} = \left[\partial^2 \mathcal{L}/(\partial \boldsymbol{\theta} \partial \boldsymbol{\theta}^T)\right]^{-1}$는 **Hessian 역행렬**로 Neumann 급수로 근사합니다.

#### 2.2.3 Fisher 정보와의 연결 (밀도 매칭)

KL 손실의 경우:

$$\mathcal{L}(\boldsymbol{\lambda}, \boldsymbol{\theta}) = -\sum_{i \in \mathbb{N}} \log p(\boldsymbol{y}_i | \boldsymbol{x}_i, \boldsymbol{\theta}(\boldsymbol{\lambda}))/N \tag{6}$$

이때 Fisher score를 정의하면:
- $\boldsymbol{u}^v_j(\boldsymbol{\theta}) = \nabla_{\boldsymbol{\theta}} \log p(\boldsymbol{y}^v_j | \boldsymbol{x}^v_j, \boldsymbol{\theta}(\boldsymbol{\lambda}))$ (검증 데이터)
- $\boldsymbol{u}\_i(\boldsymbol{\theta}) = \nabla_{\boldsymbol{\theta}} \log p(\boldsymbol{y}_i | \boldsymbol{x}_i, \boldsymbol{\theta}(\boldsymbol{\lambda}))$ (훈련 데이터)

Fisher 정보 행렬: $\mathcal{I}\_{\boldsymbol{\theta}} = \sum_{i \in \mathbb{N}} \boldsymbol{u}_i(\boldsymbol{\theta})\boldsymbol{u}_i(\boldsymbol{\theta})^T / N$

이를 통해 (5)의 그래디언트는 다음과 같이 **Fisher 커널 최대화**로 해석됩니다:

$$\nabla_{\boldsymbol{\lambda}} \mathbb{E}_{\hat{Q}^{\text{val}}_{\boldsymbol{x}}, \hat{Q}_{\boldsymbol{x}}}[\mathcal{L}_v] = \mathbb{E}_{\hat{Q}^{\text{val}}_{\boldsymbol{x}}}[\boldsymbol{u}^v(\boldsymbol{\theta})] \, \mathcal{I}_{\boldsymbol{\theta}}^{-1} \, \mathbb{E}_{\hat{Q}_{\boldsymbol{x}}}\left[\boldsymbol{u}(\boldsymbol{\theta})\boldsymbol{u}(\boldsymbol{\lambda})^T\right] \tag{7}$$

실용적 형태(동등 확률 데이터 포인트):

$$\nabla_{\boldsymbol{\lambda}} \mathbb{E}_{\hat{Q}^{\text{val}}_{\boldsymbol{x}}, \hat{Q}_{\boldsymbol{x}}}[\mathcal{L}_v] = \left[\frac{1}{M}\sum_{j \in \mathbb{M}} \boldsymbol{u}^v_j(\boldsymbol{\theta})\right] \mathcal{I}_{\boldsymbol{\theta}}^{-1} \left[\frac{1}{N}\sum_{i \in \mathbb{N}} \boldsymbol{u}_i(\boldsymbol{\theta})\boldsymbol{u}_i^T(\boldsymbol{\lambda})\right]$$

> **해석**: 암묵적 미분은 리만 다양체(Riemannian manifold)에서 Fisher 정보를 지역 메트릭으로 사용하여 $D_{\text{val}}$과 $D_{\text{train}}$의 밀도를 매칭하는 방향으로 $\boldsymbol{\lambda}$를 업데이트합니다.

### 2.3 모델 구조

AutoDO 모델 $g(\boldsymbol{\lambda})$는 세 가지 서브 모델로 구성됩니다:

$$\boldsymbol{\lambda}_i \in \mathbb{R}^{K \times 1} = [\boldsymbol{\lambda}^A_i; \boldsymbol{\lambda}^W_i; \boldsymbol{\lambda}^S_i] \in \mathbb{R}^{(A+W+S) \times 1}$$

#### (1) 증강 서브 모델 $g_A(\boldsymbol{\lambda}^A)$

- **이진 확률**: $\boldsymbol{b}_i \sim \text{Bern}(\sigma(\boldsymbol{\lambda}^{Ab}_i)) \in \{0,1\}^{A \times 1}$
- **연속 크기**: $\boldsymbol{m}_i \sim \text{rng}\frac{M}{10} \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma}_i)$
- 공분산: $\boldsymbol{\Sigma}_i = \sigma(\text{diag}(\boldsymbol{\lambda}^{Am}_i))$
- Gumbel-softmax로 이산 Bernoulli 분포의 역전파 가능

증강 순서 모델링:

$$\boldsymbol{x}^A_i(a) = \begin{cases} \mathcal{O}(\boldsymbol{x}^A_i(a-1), m^a_i), & \text{if } b^a_i = 1 \\ \boldsymbol{x}^A_i(a-1), & \text{otherwise} \end{cases} \tag{8}$$

#### (2) 손실 재가중 서브 모델 $g_W(\boldsymbol{\lambda}^W)$

$$g_W(\mathcal{L}_i, \lambda^W_i) = w_i \mathcal{L}_i, \quad w_i = 1.44 \times \text{softplus}(\lambda^W_i)$$

- $\lambda^W_i = 0$으로 초기화 → 초기 $w_i = 1$

#### (3) 소프트 레이블 서브 모델 $g_S(\boldsymbol{\lambda}^S)$

$$\boldsymbol{y}^S_i = g_S(\boldsymbol{y}_i, \boldsymbol{\lambda}^S_i) = \text{softmax}(\boldsymbol{\lambda}^S_i)$$

초기화:

$$\boldsymbol{y}^S_i(0) = (1-\alpha)\boldsymbol{y}_i + \alpha/C \quad (\text{label smoothing})$$

$$\boldsymbol{\lambda}^S_i(0) = (\boldsymbol{y}_i - 0.5)\log(1 - C - C/\alpha)$$

최종 손실은 **대칭 KL 발산(Symmetric KL Divergence)** 사용:

$$\mathcal{L} = w_i \mathcal{L}(\boldsymbol{y}^S_i, \hat{\boldsymbol{y}}_i)$$

### 2.4 최적화 알고리즘 및 복잡도

**Algorithm 1: AutoDO Bilevel Optimization**

```
Initialize θ, λ
for epoch = 1 ... epochs:
    for batch in train:
        x_A = g_A(x, λ^A)           # 증강
        ŷ = f(x_A, θ)                # 예측
        y^S = g_S(y, λ^S)           # 소프트 레이블
        ∇_θ[w·L(y^S, ŷ)] 계산 후 θ 업데이트
    
    if epoch > E:                    # HO 시작 조건
        for batch in train+val:
            ∇_λ L_v 계산 (수식 5 이용)
            λ 업데이트
```

**복잡도 비교**:

| 방법 | 계산 배율 증가 |
|------|--------------|
| AutoDO (T=5, E=0.5×epoch) | $1 + 0.5 \times (5+T)/2 = 3.5\times$ |
| DARTS 기반 방법 (T=0) | $1 + 0.5 \times 5/2 = 2.25\times$ |

> $T$: Neumann 급수 반복 횟수 (T=5 사용), $E$: HO 시작 에폭

### 2.5 성능 향상

**SVHN (WRNet28-10)**:

| 설정 (IR-NR) | Baseline | RAA | FAA | DADA | AutoDO ($\lambda^{A,W,S}$) |
|-------------|---------|-----|-----|------|--------------------------|
| 1-0.0 | 3.6% | 2.7% | 2.8% | 2.9% | **2.5%** |
| 100-0.0 | 13.6% | 10.9% | 11.5% | 12.2% | **5.3%** |
| 1-0.1 | 5.3% | 3.4% | 3.7% | 4.1% | **2.6%** |
| 100-0.1 | 20.0% | 13.6% | 15.3% | 16.5% | **6.3%** |

**CIFAR-100 (WRNet28-10)**:

| 설정 | Baseline | FAA | AutoDO |
|------|---------|-----|--------|
| 10-0.1 | 54.5% | 48.9% | **39.6%** |

**ImageNet (ResNet18)**:

| 설정 | Baseline | FAA | AutoDO |
|------|---------|-----|--------|
| 10-0.1 | 46.2% | 44.7% | **44.1%** |

### 2.6 한계

1. **계산 비용**: 기존 훈련 대비 ~3.5배 추가 연산 필요 (SVHN: 6.8 GPU시간, CIFAR-10: 5.4 GPU시간)
2. **정상 데이터에서의 제한적 개선**: 왜곡되지 않은 데이터셋(IR-NR=1-0.0)에서는 이전 방법들과 비교해 미미한 개선 (SVHN 0.2%, CIFAR-10 0.1%)
3. **CIFAR-100 소프트 레이블 최적화**: 100개 클래스에 대해 초기화 상수 $\alpha = 0.1$이 최적이 아닐 수 있으며, 왜곡되지 않은 데이터에서 소프트 레이블 서브모델이 0.2% 성능 저하 유발
4. **대형 모델에서의 제한**: ResNet-18(상대적으로 얕은 모델)에서 ImageNet 성능 향상이 소규모 데이터셋에 비해 낮음(0.6~1.2%)
5. **IFT 국소성 제약**: IFT는 고정점 근방 $(k\boldsymbol{\lambda} - \hat{\boldsymbol{\lambda}}k \leq r_1)$에서만 정의되며, 전역적 보장은 없음

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 핵심 메커니즘: 분포 이동 최소화

AutoDO의 일반화 성능 향상은 **검증 데이터 분포 $\hat{Q}^{\text{val}}\_{\boldsymbol{x},\boldsymbol{y}}$와 훈련 데이터 분포 $\hat{Q}_{\boldsymbol{x},\boldsymbol{y}}$의 불일치를 최소화**하는 데서 비롯됩니다.

$$\boldsymbol{\lambda}^* = \arg\min_{\boldsymbol{\lambda}} D_{\text{KL}}(\hat{Q}^{\text{val}}_{\boldsymbol{x},\boldsymbol{y}} \| \hat{Q}_{\boldsymbol{x},\boldsymbol{y}}(\boldsymbol{\lambda}))$$

이는 Fisher 커널 최대화(수식 7)와 동치로, 리만 다양체 위에서 두 분포 간의 거리를 최소화합니다.

### 3.2 Per-point 하이퍼파라미터가 일반화에 미치는 영향

**소수 클래스 정확도 향상 (SVHN IR=100)**:

| 클래스 그룹 | Baseline | FAA | AutoDO |
|-----------|---------|-----|--------|
| 다수 클래스 {0...4} | 높음 | 높음 | 높음 |
| 소수 클래스 {5...9} | 53.5% | 59.6% | **90.1%** |

t-SNE 분석에 의하면, AutoDO의 per-point 증강과 손실 재가중이 소수 클래스 클러스터("6", "8", "9")를 효과적으로 분리시켜 결정 경계를 테스트 데이터 분포에 맞게 조정합니다. 클래스 간 정확도 표준편차가 기존 방법의 ~20%에서 AutoDO에서 **4.4%로 크게 감소**합니다.

### 3.3 과적합 방지 메커니즘

Figure 5(학습 곡선)에서 확인되듯이:
- FAA는 노이즈 레이블에 빠르게 과적합
- AutoDO는 epoch $E$ 이후 HO를 시작하여 적절한 per-point 하이퍼파라미터 선택을 통해 **과적합을 방지**하고 테스트 정확도를 지속적으로 향상

### 3.4 검증 데이터와 테스트 데이터의 일치성 검증

논문의 ablation study(Table 5)에서:
- $D^i_{\text{val}}$를 사용한 AutoDO와 $D_{\text{test}}$를 직접 사용한 AutoDO의 성능 차이가 미미함
- 이는 $Q^{\text{test}}\_{\boldsymbol{x},\boldsymbol{y}} \approx Q^{\text{val}}_{\boldsymbol{x},\boldsymbol{y}}$ 가정이 유효함을 확인하고, **검증 데이터에 과적합이 발생하지 않음**을 증명

### 3.5 일반화 관련 이론적 근거

Fisher 정보를 활용한 암묵적 미분(수식 7)은 PAC-Bayes 이론과 유사하게 **모델의 복잡도를 제어**하면서 일반화를 향상시키는 효과를 가집니다. 구체적으로:

$$\mathcal{I}_{\boldsymbol{\theta}}^{-1} = \left[\sum_{i \in \mathbb{N}} \boldsymbol{u}_i(\boldsymbol{\theta})\boldsymbol{u}_i(\boldsymbol{\theta})^T / N\right]^{-1}$$

이 역 Fisher 행렬은 모델 파라미터 공간에서 자연 그래디언트(natural gradient) 역할을 수행하여 최적화 경로를 더 효율적으로 안내합니다.

---

## 4. 향후 연구에 미치는 영향 및 고려 사항

### 4.1 연구에 미치는 영향

#### 4.1.1 데이터셋 최적화 패러다임 전환

AutoDO는 DA 정책 탐색을 단순히 증강 연산 선택 문제에서 **데이터셋 수준의 분포 최적화**로 확장했습니다. 이는 다음 연구 방향을 열어줍니다:

- **데이터 중심 AI(Data-Centric AI)**: 모델 구조보다 데이터 품질/분포 최적화에 집중하는 패러다임과 일치
- **Foundation Model 파인튜닝**: 대규모 사전 학습 모델의 특정 도메인 적응 시 편향된 소규모 데이터셋 처리에 응용 가능
- **연속 학습(Continual Learning)**: 시간에 따라 분포가 변하는 데이터 스트림에서 per-point HO 적용 가능성

#### 4.1.2 메타 학습(Meta-Learning)과의 연결

Bilevel 최적화 프레임워크는 MAML(Model-Agnostic Meta-Learning)과 구조적으로 유사하여:
- Few-shot 학습에서 노이즈 레이블이 포함된 support set 처리
- 클래스 불균형 환경에서의 메타 학습 개선에 응용 가능

#### 4.1.3 암묵적 미분의 응용 확장

Lorraine et al.(2020)의 암묵적 미분 프레임워크를 대규모 데이터 증강에 성공적으로 적용함으로써, 다음 응용이 촉진됩니다:
- Neural Architecture Search(NAS)에서 데이터 품질 공동 최적화
- 하이퍼파라미터 최적화(HPO)와 DA의 통합

### 4.2 향후 연구 시 고려할 점

#### 4.2.1 확장성 문제

현재 AutoDO의 계산 복잡도는 $O(N \cdot K)$로 데이터셋 크기 $N$과 하이퍼파라미터 수 $K$에 비례합니다. **수천만 개 이상의 데이터를 가진 대규모 데이터셋**에서의 효율성 개선이 필요합니다:

- Hessian 역행렬의 더 효율적인 근사 방법 (예: Kronecker-factored curvature, KFAC)
- 대표 샘플 기반의 계층적 하이퍼파라미터 추정

#### 4.2.2 검증 데이터 의존성

AutoDO는 작은 **깨끗하고 편향되지 않은** 검증 데이터셋 $D_{\text{val}}$에 의존합니다. 실제 응용에서:
- 완전히 편향되지 않은 검증 데이터를 얻기 어려운 경우의 처리 방안
- 검증 데이터셋 크기가 성능에 미치는 영향 연구 (현재 SVHN 32%, CIFAR 20% 분할)
- 자기 지도 학습(Self-Supervised Learning)을 활용한 검증 데이터 대체 방안

#### 4.2.3 다른 도메인으로의 확장

현재는 이미지 분류에 집중되어 있으나:
- **객체 탐지, 시맨틱 분할**: 경계 박스나 마스크에 따른 증강 하이퍼파라미터의 per-point 최적화
- **자연어 처리**: 텍스트 증강(back-translation, synonym replacement)에서의 per-sample 최적화
- **의료 영상**: 레이블 노이즈와 클래스 불균형이 심각한 임상 데이터 처리

#### 4.2.4 이론적 보장 강화

- 현재 IFT 기반 최적화는 국소적(local)으로만 정의됨 → **전역 수렴성** 증명 필요
- Fisher 커널 최대화와 PAC-Bayes 일반화 경계 간의 형식적 연결 구축
- 노이즈 레이블 비율에 따른 성능 보장(robustness guarantee) 이론화

#### 4.2.5 적응형 하이퍼파라미터 초기화

- 소프트 레이블 초기화 상수 $\alpha$의 자동 결정
- 증강 연산 종류($A$)와 순서의 구조적 탐색 통합
- 클래스별 불균형 정도를 사전에 활용한 가중치 초기화

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

논문에서 직접 비교된 2020년 이후 방법들을 중심으로, 제가 확인 가능한 범위 내에서 분석합니다.

### 5.1 논문에서 직접 비교된 방법들 (검증됨)

| 방법 | 발표 | 핵심 접근 | AutoDO 대비 한계 |
|------|------|----------|----------------|
| **RandAugment** [Cubuk et al., 2019] | arXiv 2019 | 탐색 공간 축소, 2개 하이퍼파라미터 | 공유 정책, 편향 데이터 취약 |
| **Fast AutoAugment (FAA)** [Lim et al., 2019] | NeurIPS 2019 | 베이지안 최적화 기반 밀도 매칭 | 공유 정책, 노이즈 레이블 취약 |
| **DADA** [Li et al., ECCV 2020] | ECCV 2020 | DARTS 기반 미분 가능 증강 | 공유 정책, per-point 하이퍼파라미터 없음 |
| **Adversarial AutoAugment** [Zhang et al., ICLR 2020] | ICLR 2020 | 적대적 훈련 기반 정책 학습 | 공유 정책 구조 유지 |

### 5.2 AutoDO 이후 관련 연구 동향 (일반적 관찰)

> **주의**: 아래 내용은 2021년 이후 연구 동향에 대한 일반적 지식에 기반하며, 개별 논문의 세부 수치는 확인이 필요합니다.

AutoDO(2021)가 제기한 문제의식(공유 정책의 한계, 편향/노이즈 데이터 강건성)은 이후 연구들에 영향을 미쳤습니다:

- **데이터 중심 AI(Data-Centric AI) 운동**: Andrew Ng 등이 주창한 데이터 품질 중심 접근과 맥락을 같이함
- **노이즈 레이블 학습**: DivideMix, Co-learning 등의 방법들과 DA를 결합하는 연구
- **장기 분포 학습(Long-tail Learning)**: 클래스 불균형 처리를 위한 증강과 손실 재가중의 통합 접근

### 5.3 방법론적 위치 정리

```
탐색 복잡도 감소 방향:
AA(수천 GPU시간) → PBA → FAA → DADA(0.1시간) → RandAugment(거의 없음)
                                                    ↑
                                           AutoDO(5-7시간)
                                      [강건성 향상으로 복잡도 증가 감수]

강건성 향상 방향:
공유 정책 방법들 → AutoDO(per-point 하이퍼파라미터)
```

AutoDO는 계산 효율성보다 **강건성과 일반화 성능**을 우선시하는 새로운 연구 방향을 제시했습니다.

---

## 참고 자료

**본 답변의 주요 출처:**

1. **주논문**: Gudovskiy, D., Rigazio, L., Ishizaka, S., Kozuka, K., & Tsukizawa, S. (2021). "AutoDO: Robust AutoAugment for Biased Data with Label Noise via Scalable Probabilistic Implicit Differentiation." *arXiv:2103.05863v2*

**논문 내 직접 인용된 핵심 참고문헌:**

2. Cubuk, E. D., et al. (2019). "RandAugment: Practical automated data augmentation with a reduced search space." *arXiv:1909.13719*
3. Lim, S., et al. (2019). "Fast AutoAugment." *NeurIPS 2019*
4. Li, Y., et al. (2020). "DADA: Differentiable automatic data augmentation." *ECCV 2020*
5. Lorraine, J., Vicol, P., & Duvenaud, D. (2020). "Optimizing millions of hyperparameters by implicit differentiation." *AISTATS 2020*
6. Liu, H., Simonyan, K., & Yang, Y. (2019). "DARTS: Differentiable architecture search." *ICLR 2019*
7. Cubuk, E. D., et al. (2018). "AutoAugment: Learning augmentation policies from data." *arXiv:1805.09501*
8. Hataya, R., et al. (2019). "Faster AutoAugment: Learning augmentation strategies using backpropagation." *arXiv:1911.06987*
9. Zhang, X., et al. (2020). "Adversarial AutoAugment." *ICLR 2020*
10. Jang, E., Gu, S., & Poole, B. (2016). "Categorical reparameterization with Gumbel-softmax." *arXiv:1611.01144*
