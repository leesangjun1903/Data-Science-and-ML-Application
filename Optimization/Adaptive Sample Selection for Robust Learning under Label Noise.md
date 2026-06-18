# Adaptive Sample Selection for Robust Learning under Label Noise

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

본 논문(Deep Patel & P S Sastry, IISc Bangalore, arXiv:2106.15292v3, 2022)은 **BARE(BAtch REweighting)**라는 적응형 샘플 선택 알고리즘을 제안합니다. 핵심 주장은 다음과 같습니다:

> **"레이블 노이즈가 존재하는 환경에서, 노이즈율 정보나 클린 검증 데이터, 보조 네트워크 없이도 미니배치 통계만으로 적응적 샘플 선택이 가능하다."**

### 주요 기여 (Contributions)

| 기여 항목 | 설명 |
|-----------|------|
| **하이퍼파라미터 불필요** | 샘플 선택에 추가 하이퍼파라미터 없음 |
| **노이즈율 정보 불필요** | $\eta_{kk'}$ 값을 사전에 알 필요 없음 |
| **클린 데이터 불필요** | 별도의 클린 검증 셋 불필요 |
| **보조 네트워크 불필요** | 단일 네트워크로 동작 |
| **계산 효율성** | CCE 표준 훈련 대비 유사한 수준의 연산량 |
| **이론적 정당화** | Self-Paced Learning 프레임워크에서 동적 임계값의 이론적 근거 제시 |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

#### 문제 배경: 레이블 노이즈 하의 DNN 학습

딥러닝 모델은 대규모 학습 데이터를 필요로 하며, 이는 크라우드소싱이나 자동화된 레이블링으로 수집됩니다. 이 과정에서 **레이블 노이즈**가 필연적으로 발생합니다. DNN은 이런 노이즈가 있는 데이터에 대해 **기억화(memorization)** 또는 **과적합(overfitting)**에 취약합니다.

#### 기존 방법의 한계

기존 샘플 선택 알고리즘들의 문제점:

- **Co-Teaching [Han et al., 2018]**: 노이즈율 $\eta$ 사전 지식 필요, 두 네트워크 교차 훈련 → 높은 계산 비용
- **MentorNet [Jiang et al., 2018]**: 보조 신경망 필요, 클린 데이터 필요
- **Meta-Ren [Ren et al., 2018]**: 클린 검증 데이터 필요, 메타러닝으로 높은 연산 비용
- **Curriculum Loss [Lyu & Tsang, 2020]**: 노이즈율 추정 필요
- **고정 임계값 문제**: 학습 과정에서 손실값이 동적으로 변화하는데, 고정 임계값은 이를 반영하지 못함

### 2.2 제안 방법: BARE 알고리즘

#### 문제 공식화

$K$-클래스 분류 문제에서, 레이블 노이즈는 **클래스 조건부 노이즈(Class Conditional Noise)**로 모델링됩니다:

$$P[y_i = e_{k'} \mid y_i^c = e_k] = \eta_{kk'} \tag{1}$$

여기서 $\eta_{kk'}$는 노이즈율 행렬이며, $e_k$는 클래스 $k$에 해당하는 one-hot 벡터입니다. **대각 우세(diagonally dominant)** 조건 $\eta_{kk} > \eta_{kk'}, \forall k' \neq k$를 가정합니다.

**대칭 노이즈(Symmetric Noise)** 특수 케이스:
$$\eta_{kk} = 1 - \eta, \quad \eta_{kk'} = \frac{\eta}{K-1}, \quad \forall k' \neq k$$

#### 이론적 동기: Curriculum Learning 프레임워크

일반적인 커리큘럼 학습은 가중 손실 최소화로 표현됩니다:

$$\min_{\theta, \mathbf{w} \in [0,1]^m} \mathcal{L}_{\text{wtd}}(\theta, \mathbf{w}) = \sum_{i=1}^m w_i \mathcal{L}(f(x_i;\theta), y_i) + G(\mathbf{w}) + \beta||\theta||^2 \tag{2}$$

여기서 $G(\mathbf{w})$는 커리큘럼을 나타냅니다. Self-Paced Learning [Kumar et al., 2010]의 $G(\mathbf{w}) = -\lambda||\mathbf{w}||_1$를 적용하면:

$$\min_{\theta, \mathbf{w} \in [0,1]^m} \mathcal{L}_{\text{wtd}}(\theta, \mathbf{w}) = \sum_{i=1}^m (w_i l_i - \lambda w_i) = \sum_{i=1}^m (w_i l_i + (1-w_i)\lambda) - m\lambda \tag{4}$$

최적해는 $w_i = 1$ if $l_i < \lambda$, $w_i = 0$ otherwise.

#### BARE의 핵심 확장: 클래스별 동적 임계값

$\lambda$를 클래스 레이블 $y_i$에 의존하도록 확장하면:

$$\min_{\theta, \mathbf{w} \in [0,1]^m} \mathcal{L}_{\text{wtd}}(\theta, \mathbf{w}) = \sum_{j=1}^K \sum_{\substack{i=1 \\ i: y_i = e_j}}^{m} \left(w_i l_i + (1-w_i)\lambda_j\right) - \sum_{j=1}^K \sum_{\substack{i=1 \\ i: y_i = e_j}}^{m} \lambda_j \tag{6}$$

여기서 $\lambda_j = \lambda(e_j)$는 클래스 $j$에 대한 임계값입니다. 최적 $w_i$는 동일하게: $y_i = e_j$인 샘플 $i$에 대해 $w_i = 1$ if $l_i < \lambda_j$.

**핵심 인사이트**: $\lambda_j$를 해당 클래스의 미니배치 통계에 의존하게 만들어도 최적해의 구조가 동일하게 유지됩니다.

#### 샘플 선택 기준 (핵심 수식)

CCE 손실 $l_i = -\ln(f_{y_i}(x_i;\theta))$를 사용하므로, 손실이 작다 ↔ 사후 확률이 크다는 관계를 이용합니다:

$$w_i = \begin{cases} 1 & \text{if } f_{y_i}(\mathbf{x}_i;\theta) \geq \lambda_{y_i} = \mu_{y_i} + \kappa \cdot \sigma_{y_i} \\ 0 & \text{else} \end{cases} \tag{7}$$

여기서:

$$\mu_{y_i} = \frac{1}{|S_{y_i}|} \sum_{s \in S_{y_i}} f_{y_i}(\mathbf{x}_s;\theta), \quad \sigma^2_{y_i} = \frac{1}{|S_{y_i}|} \sum_{s \in S_{y_i}} \left(f_{y_i}(\mathbf{x}_s;\theta) - \mu_{y_i}\right)^2$$

- $S_{y_i} = \{k \in [m] \mid y_k = y_i\}$: 미니배치에서 클래스 $y_i$를 가진 샘플 인덱스 집합
- $m$: 미니배치 크기
- $\kappa = 1$ (논문 기본값, $\kappa > 0$이면 유사한 성능)

#### 파라미터 업데이트

$$\theta_{t+1} = \theta_t - \alpha \nabla \left(\frac{1}{|R|} \sum_{(\mathbf{x}, y_\mathbf{x}) \in R} \mathcal{L}(\mathbf{x}, y_\mathbf{x}; \theta_t)\right)$$

여기서 $R$은 선택된 샘플 집합입니다.

### 2.3 모델 구조

#### 네트워크 아키텍처

| 데이터셋 | 네트워크 | 최적화기 |
|---------|---------|---------|
| **MNIST** | 1-hidden layer MLP (28×28 → 256 → 10) | Adam (lr = $2 \times 10^{-4}$), ReduceLROnPlateau |
| **CIFAR-10** | 4-layer CNN (Conv 64→128→196→16 → Dense 256→10) | Adam (lr = $2 \times 10^{-3}$), ReduceLROnPlateau |
| **Clothing-1M** | Pre-trained ResNet-50 | SGD (lr = $10^{-3}$, 에폭 6, 11에서 절반), weight decay = $10^{-3}$, momentum 0.9 |

#### BARE 알고리즘 (Algorithm 1 요약)

```
Input: 노이즈 데이터셋 D_η, K (클래스 수), T_max, α, |M|
for each epoch:
  for each mini-batch M:
    1. 각 클래스 p에 대해:
       - S_p = {k ∈ [m] | y_k = e_p} 수집
       - μ_p, σ_p² 계산 (사후확률의 평균, 분산)
       - λ_p = μ_p + σ_p (임계값 설정)
    2. 각 샘플 x에 대해:
       - f_{y_x}(x;θ) ≥ λ_{y_x}이면 R에 추가
    3. θ를 R로 업데이트
Output: 최종 파라미터 θ
```

### 2.4 성능 향상

#### MNIST 결과

| 알고리즘 | $\eta=0.5$ (SYM) | $\eta=0.7$ (SYM) | $\eta=0.45$ (CC) |
|---------|-----------------|-----------------|-----------------|
| Co-Teaching | 90.80±0.18 | 87.17±0.45 | **95.20±0.22** |
| Co-Teaching+ | **93.17±0.3** | 87.26±0.67 | 91.10±1.51 |
| Meta-Ren | 90.39±0.07 | 85.10±0.28 | **95.40±0.31** |
| Meta-Net | 74.94±9.56 | 65.52±21.35 | 75.03±0.59 |
| Curriculum Loss | 92.00±0.26 | **88.28±0.45** | 81.52±3.27 |
| CCE (standard) | 74.30±0.55 | 61.19±1.29 | 74.96±0.21 |
| **BARE (제안)** | **94.38±0.13** | **91.61±0.60** | 94.11±0.77 |

#### CIFAR-10 결과

| 알고리즘 | $\eta=0.3$ (SYM) | $\eta=0.7$ (SYM) | $\eta=0.4$ (CC) |
|---------|-----------------|-----------------|----------------|
| Co-Teaching | **71.72±0.30** | **58.95±1.31** | 65.26±0.78 |
| Co-Teaching+ | 60.14±0.35 | 37.69±0.70 | 63.05±0.39 |
| Meta-Ren | 62.96±0.70 | 45.14±1.04 | **70.27±0.77** |
| CCE | 54.83±0.28 | 23.46±0.37 | 64.06±0.32 |
| **BARE (제안)** | **75.85±0.41** | **59.53±1.12** | **70.63±0.46** |

#### Clothing-1M 결과

| 알고리즘 | 정확도 |
|---------|-------|
| CCE | 68.94% |
| Joint Opt. | 72.23% |
| **BARE (제안)** | **72.28%** |
| C2D | 74.58% |
| DivideMix | 74.76% |

#### 계산 효율성 (200 에폭 기준)

| 알고리즘 | MNIST (초) | CIFAR-10 (초) |
|---------|-----------|--------------|
| **BARE** | **310.64** | **930.78** |
| Co-Teaching | 504.5 | 1687.9 |
| Co-Teaching+ | 537.7 | 1790.57 |
| Meta-Ren | 807.4 | 8130.87 |
| Meta-Net | 1138.4 | 8891.6 |
| CCE | 229.27 | 825.68 |

> BARE는 표준 CCE 훈련과 거의 동일한 수준의 계산 비용으로 운용됩니다.

### 2.5 한계

논문에서 명시적으로 인정하는 한계:

1. **다수 클래스 문제**: 클래스 수가 미니배치 크기와 비슷해지면 클래스별 통계의 신뢰성 저하 가능. 논문은 Food-101N(101클래스, batch=128)에서 84.12% 달성으로 부분적 대응을 보이나, 완전한 해결책은 제시하지 않음.
2. **클래스 조건부 노이즈 일부 케이스**: MNIST 클래스 조건부 노이즈에서 CoT, MR보다 소폭 낮은 성능.
3. **고수준 노이즈 환경의 한계**: Clothing-1M에서 DivideMix, C2D 대비 약 2-2.5% 낮은 성능.
4. **인스턴스 의존 노이즈**: 특징 의존적 노이즈(feature-dependent noise)에 대한 이론적 보장이 제한적.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화 성능을 높이는 핵심 메커니즘

#### (1) 동적 적응형 임계값에 의한 과적합 방지

BARE의 임계값 $\lambda_{y_i} = \mu_{y_i} + \kappa \cdot \sigma_{y_i}$는 매 미니배치마다 자동으로 갱신됩니다. 이는 학습 초반에는 클린 샘플과 노이즈 샘플의 사후확률 분포가 유사하다가, 학습이 진행될수록 클린 샘플의 사후확률이 높아지는 DNN의 학습 동역학과 자연스럽게 연동됩니다.

결과적으로 **동적 임계값 $\lambda$**가 자동으로 "클린 샘플 필터"로 기능하며, 네트워크가 노이즈에 과적합되는 것을 억제합니다.

이를 수식으로 표현하면: 어떤 클래스 $j$에 대해, 에폭 $t$에서의 임계값은:

$$\lambda_j^{(t, b)} = \mu_j^{(t,b)} + \sigma_j^{(t,b)}$$

여기서 상첨자 $(t, b)$는 에폭 $t$, 배치 $b$를 나타냅니다. 이 값은 에폭과 배치에 따라 달라지므로, **자동으로 진화하는 커리큘럼**을 형성합니다.

#### (2) 클래스별 분리 임계값 (Class-specific Threshold)

기존 방법들은 전체 데이터셋에 단일 임계값을 적용합니다. BARE는 **클래스별 임계값**을 사용하므로, 클래스마다 다른 난이도와 노이즈 수준에 적응할 수 있습니다. 특히 클래스 불균형이나 클래스별 노이즈율이 다른 현실적 시나리오에 강건합니다.

#### (3) 레이블 재현율(Recall) 향상을 통한 일반화

논문의 Figure 5에서 BARE는 다른 알고리즘들에 비해 **일관되게 높은 레이블 재현율**을 보입니다. 재현율이 높다는 것은:

$$\text{Recall} = \frac{\text{선택된 클린 샘플 수}}{\text{전체 클린 샘플 수}}$$

분모(전체 클린 샘플)를 더 많이 활용한다는 의미입니다. 이는 모델이 클린 샘플로부터 더 많은 정보를 학습하게 하여 일반화 성능을 높입니다.

#### (4) 과적합 안정성: 장시간 훈련 실험 (Table 17)

500 에폭 훈련 결과, BARE는 정확도가 유지되거나 소폭만 변화하는 반면, 다른 알고리즘은 성능이 저하됩니다. 이는 BARE의 **암묵적 정규화 효과**를 보여줍니다. 임계값 기반 필터링이 에폭이 늘어나도 노이즈 샘플로의 과적합을 지속적으로 억제하기 때문입니다.

#### (5) 배치 크기에 대한 비민감성 (Table 2, 16)

배치 크기 $\{64, 128, 256\}$에 걸쳐 일관된 성능이 관찰됩니다:

| 데이터셋 | 노이즈 | Batch 64 | Batch 128 | Batch 256 |
|---------|--------|---------|---------|---------|
| CIFAR-10 | 40% CC | 71.87±0.28 | 70.63±0.46 | 69.03±0.35 |

이는 BARE가 배치 크기에 관계없이 안정적인 일반화 성능을 보임을 나타냅니다.

#### (6) 노이즈율 미스스펙 대응

Figure 6에서 보듯이, CoT, CoT+, CL은 노이즈율을 잘못 추정하면 성능이 크게 저하되지만, BARE는 노이즈율 정보 자체를 사용하지 않으므로 이런 취약점이 없습니다. 실제 환경에서 노이즈율을 정확히 알기 어렵다는 점을 고려하면, **BARE의 일반화 성능은 실제 배포 환경에서 더욱 두드러집니다.**

---

## 4. 앞으로의 연구에 미치는 영향 및 고려사항

### 4.1 연구에 미치는 영향

#### (1) 패러다임 전환: 통계 기반 동적 임계값

기존 연구들이 노이즈율이나 클린 데이터라는 "외부 지식"에 의존했다면, BARE는 **배치 내부 통계만으로 자율적으로 동작**하는 패러다임을 제시합니다. 이는 후속 연구들에게 "추가 정보 없이도 robust learning이 가능하다"는 방향성을 제공합니다.

#### (2) Self-Paced Learning과 동적 커리큘럼의 연결

BARE는 SPL 프레임워크에서 $\lambda$를 클래스별, 배치별로 다르게 설정해도 최적해 구조가 동일하다는 **새로운 이론적 통찰**을 제공합니다. 이는 커리큘럼 학습 연구에서 탐구되지 않았던 방향을 개척합니다.

#### (3) 실용적 영향

노이즈율 정보 없이도 경쟁력 있는 성능을 달성하는 BARE는 **의료 영상, 위성 영상, 소셜 미디어 데이터** 등 레이블 노이즈율을 알기 어려운 실제 응용에서 즉시 적용 가능한 기준선(baseline)으로 기능할 수 있습니다.

### 4.2 2020년 이후 관련 최신 연구 비교 분석

아래는 논문에 인용된 2020년 이후 주요 연구들과의 비교 분석입니다:

#### DivideMix [Li, Socher, Hoi, ICLR 2020]

- **방법**: 반지도학습 방식으로 GMM(Gaussian Mixture Model)을 사용해 클린/노이즈 샘플을 분리하고, MixMatch로 학습
- **장점**: Clothing-1M에서 74.76% 달성 (BARE 72.28% 대비 우수)
- **단점**: 5개 하이퍼파라미터 튜닝 필요, BARE 대비 약 2.4배 계산 비용
- **BARE 대비**: BARE는 하이퍼파라미터 없이 더 간단하지만 최고 성능에는 미치지 못함

#### C2D (Contrast to Divide) [Zheltonozhskii et al., WACV 2022]

- **방법**: 자기지도학습(SimCLR) 사전학습으로 더 나은 초기화 제공, ELR+와 결합
- **장점**: Clothing-1M에서 74.58%
- **단점**: 자기지도학습 사전학습 단계로 인해 더 높은 계산 비용
- **BARE 대비**: BARE는 사전학습 없이도 경쟁력 있는 성능

#### Early-Learning Regularization (ELR) [Liu et al., NeurIPS 2020]

- **방법**: 초기 학습 단계(클린 샘플을 주로 학습하는 시기)에 정규화를 적용해 노이즈 레이블 기억화 방지
- **BARE와 유사점**: 둘 다 DNN이 클린 샘플을 먼저 학습한다는 현상을 활용
- **차이점**: ELR은 손실 함수 수정 방식, BARE는 샘플 선택 방식

#### PLC (Progressive Label Correction) [Zhang et al., ICLR 2020]

- **방법**: 특징 의존적 노이즈에 대응하는 점진적 레이블 교정
- **Food-101N 결과**: 85.28% (BARE 84.12% 대비 소폭 우세)
- **BARE 대비**: BARE는 레이블 교정 없이도 유사한 성능

#### 비교 요약표

| 방법 | 노이즈율 필요 | 클린 데이터 필요 | 보조 네트워크 | 하이퍼파라미터 | Clothing-1M |
|------|-------------|----------------|------------|-------------|------------|
| DivideMix | ✗ | ✗ | ✗ | ✅ (5개) | 74.76% |
| C2D | ✗ | ✗ | ✅ (SSL) | ✅ | 74.58% |
| Co-Teaching | ✅ | ✗ | ✅ (2개) | ✅ | 70.15% |
| **BARE** | **✗** | **✗** | **✗** | **✗** | **72.28%** |

### 4.3 앞으로 연구 시 고려할 점

#### (1) 대규모 클래스 수에서의 안정성

클래스 수 $K$가 크면 미니배치 내 각 클래스의 샘플 수가 매우 적어져 통계 추정이 불안정해집니다. 해결 방향:
- 클래스 인식 미니배치 구성(class-aware batch sampling)
- 지수 이동 평균(EMA)을 활용한 점진적 통계 업데이트: $\tilde{\mu}\_{y_i}^{(t)} = \alpha \mu_{y_i}^{(t)} + (1-\alpha)\tilde{\mu}_{y_i}^{(t-1)}$

#### (2) 인스턴스 의존 노이즈 대응

BARE는 클래스 조건부 노이즈를 주로 다루지만, 현실적 노이즈는 **인스턴스 의존적**입니다 (Chen et al., 2020; Xia et al., 2020). BARE를 인스턴스 수준의 표현 학습(contrastive learning 등)과 결합하는 연구가 필요합니다.

#### (3) 이론적 수렴 보증

BARE의 경험적 성능은 우수하지만, 동적 임계값 하에서의 **수렴 이론**이 부재합니다. 향후 연구에서는 특정 노이즈 모델과 네트워크 복잡도 하에서 BARE의 수렴 속도와 일반화 오차 상한을 이론적으로 분석할 필요가 있습니다.

#### (4) 준지도학습과의 결합

BARE가 "노이즈 샘플"로 표시한 데이터는 버려집니다. 이를 **비지도/준지도 학습**으로 활용하면 추가적인 성능 향상이 가능합니다. DivideMix처럼 MixMatch 류의 기법과 결합하는 방향이 유망합니다.

#### (5) 다른 모달리티로의 확장

BARE는 이미지 분류에서만 검증되었습니다. 텍스트, 오디오, 시계열 등 다른 모달리티에서 클래스별 사후확률 통계가 동일하게 유효한지 검증이 필요합니다.

#### (6) 임계값 통계의 다양화

현재 $\mu + \kappa \sigma$ 형태의 임계값 외에, **분위수(quantile)** 기반이나 **적응형 커널 밀도 추정** 기반의 임계값을 탐색하면 더욱 강건한 샘플 선택이 가능할 수 있습니다.

---

## 참고 자료

**주요 논문 (본문에서 직접 인용)**

1. **Deep Patel, P S Sastry** - "Adaptive Sample Selection for Robust Learning under Label Noise", arXiv:2106.15292v3, 2022 (분석 대상 논문)
2. **Han et al.** - "Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels", NeurIPS 2018
3. **Yu et al.** - "How does Disagreement Help Generalization against Label Corruption?", ICML 2019
4. **Ren et al.** - "Learning to Reweight Examples for Robust Deep Learning", ICML 2018
5. **Shu et al.** - "Meta-Weight-Net: Learning an Explicit Mapping for Sample Weighting", NeurIPS 2019
6. **Lyu & Tsang** - "Curriculum Loss: Robust Learning and Generalization against Label Corruption", ICLR 2020
7. **Jiang et al.** - "MentorNet: Learning Data-Driven Curriculum for Very Deep Neural Networks on Corrupted Labels", ICML 2018
8. **Li, Socher, Hoi** - "DivideMix: Learning with Noisy Labels as Semi-Supervised Learning", ICLR 2020
9. **Zheltonozhskii et al.** - "Contrast to Divide: Self-Supervised Pre-Training for Learning with Noisy Labels", WACV 2022
10. **Liu et al.** - "Early-Learning Regularization Prevents Memorization of Noisy Labels", NeurIPS 2020
11. **Kumar et al.** - "Self-Paced Learning for Latent Variable Models", NeurIPS 2010
12. **Bengio et al.** - "Curriculum Learning", ICML 2009
13. **Zhang et al.** - "Understanding Deep Learning Requires Rethinking Generalization", arXiv 2016
14. **Arpit et al.** - "A Closer Look at Memorization in Deep Networks", ICML 2017
15. **Xiao et al.** - "Learning from Massive Noisy Labeled Data for Image Classification (Clothing-1M)", CVPR 2015
16. **Zhang et al.** - "Learning with Feature-Dependent Label Noise: A Progressive Approach", ICLR 2020
17. **Xia et al.** - "Part-Dependent Label Noise: Towards Instance-Dependent Label Noise", NeurIPS 2020
