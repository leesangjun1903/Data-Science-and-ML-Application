# Causal Generative Domain Adaptation Networks (CG-DAN)

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

이 논문(Gong et al., 2018, arXiv:1804.04333)은 도메인 적응(Domain Adaptation, DA)에서 **도메인 간 분포 변화를 명시적으로 모델링하고 활용**해야 한다는 점을 핵심 주장으로 삼습니다. 특히 기존 연구들이 주로 가정했던 **covariate shift** ($P_X$ 변화, $P_{Y|X}$ 불변)보다 더 어려운 **conditional shift** ($P_{X|Y}$ 변화) 상황을 다룹니다.

### 주요 기여 (Two-fold)

| 기여 | 내용 |
|------|------|
| **G-DAN** | 잠재 변수 $\boldsymbol{\theta}$를 이용해 도메인 간 불변성과 변화를 명시적으로 포착하는 생성적 도메인 적응 네트워크 제안 |
| **CG-DAN** | 인과 그래프 구조를 활용하여 결합 분포를 모듈로 분해함으로써 통계적·계산적 효율성을 향상시킨 확장 모델 제안 |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

**비지도 도메인 적응(Unsupervised DA)**에서 타겟 도메인에 레이블이 없는 상황 하에, $P_{X|Y}$가 도메인마다 달라지는 **conditional shift** 문제를 해결하고자 합니다.

$$P_{XY}^{source} \neq P_{XY}^{target}$$

기존 방법들은 주로 다음 두 가정 중 하나에 의존했습니다:
- **Covariate shift**: $P_X$ 변화, $P_{Y|X}$ 불변 → 밀도비 재가중치 또는 불변 표현 학습
- **Location-scale transform**: $P_{X|Y}$ 변화를 위치-척도 변환으로 제한 → 너무 강한 가정

본 논문은 이보다 **더 일반적이고 비모수적인** 방식으로 $P_{X|Y}$의 변화를 모델링합니다.

---

### 2.2 제안하는 방법 및 수식

#### (A) G-DAN: Generative Domain Adaptation Network

$Y$가 $X$의 원인이라는 인과 가정 하에, 다음과 같은 함수형 모델(functional model)을 제안합니다:

$$X = g(Y, E; \boldsymbol{\theta}) \tag{1}$$

- $g$: 신경망(NN)으로 표현되는 함수, 모든 도메인에서 공유 (불변 부분)
- $E \sim Q_E$: 도메인 간 고정된 분포를 갖는 독립 노이즈
- $\boldsymbol{\theta} \in \mathbb{R}^d$: 도메인 변화를 포착하는 잠재 변수 (도메인 $s$에서 $\boldsymbol{\theta}^{(s)}$ 취함)

도메인 인덱스 $s$의 one-hot 표현 $\mathbf{1}_s$를 이용한 재매개변수화:

$$\boldsymbol{\theta} = \Theta \mathbf{1}_s$$

**소스 도메인에서의 손실 함수 (MMD 기반)**:

$$J^s_{kl} = \left\| \mathbb{E}_{(X,Y) \sim P^s_{XY}} [\phi(X) \otimes \psi(Y)] - \mathbb{E}_{(X,Y) \sim Q^s_{XY}} [\phi(X) \otimes \psi(Y)] \right\|^2_{\mathcal{H}_x \otimes \mathcal{H}_y} \tag{2}$$

**타겟 도메인에서의 손실 함수 (레이블 없이 주변 분포만 매칭)**:

$$M_k = \left\| \mathbb{E}_{X \sim P^t_X} [\phi(X)] - \mathbb{E}_{X \sim Q^t_X} [\phi(X)] \right\|^2_{\mathcal{H}_x}$$

**최종 학습 목표**:

$$\min_{g, \Theta} \; J^s_{kl} + \alpha M_k$$

여기서 $\alpha$는 소스/타겟 손실의 균형을 맞추는 하이퍼파라미터 (실험에서 $\alpha=1$).

경험적 MMD 추정치:

$$\hat{J}^s_{kl} = \frac{1}{n^2}\sum_i\sum_j k(x^s_i, x^s_j)l(y^s_i, y^s_j) - \frac{2}{n^2}\sum_i\sum_j k(x^s_i, \hat{x}^s_j)l(y_i, \hat{y}^s_j) + \frac{1}{n^2}\sum_i\sum_j k(\hat{x}^s_i, \hat{x}^s_j)l(\hat{y}^s_i, \hat{y}^s_j)$$

---

#### (B) CG-DAN: Causal G-DAN

고차원 특징 $X$에 대해 $P_{X|Y}$를 통째로 모델링하는 것은 비효율적입니다. 이를 해결하기 위해 **인과 그래프(DAG)에 따른 마르코프 인수분해**를 활용합니다.

**함수형 인과 모델(Functional Causal Model, FCM)**:

$$F_i: X_i = g_i(\mathbf{PA}_i, E_i, \boldsymbol{\theta}_i), \quad i = 1, \ldots, D \tag{3}$$

- $\mathbf{PA}_i$: $X_i$의 직접 원인(direct causes)
- $\boldsymbol{\theta}_i$: 해당 인과 모듈의 변화를 포착하는 잠재 변수 (서로 독립)
- $g_i$: 도메인 간 공유되는 불변 메커니즘

**KL 발산의 인과 분해**:

$$\text{KL}(P^s_{XY} \| Q^s_{XY|\boldsymbol{\theta}}) = \sum_{i=1}^D \text{KL}(P^s_{X_i|\mathbf{PA}_i} \| Q^s_{X_i|\mathbf{PA}_i, \boldsymbol{\theta}_i} P^s_Y)$$

이를 통해 각 모듈을 **독립적으로 학습(divide-and-conquer)**할 수 있습니다.

---

### 2.3 모델 구조

#### G-DAN 구조
```
Y ──┐
    ├──► g (Neural Network) ──► X
E ──┤
    │
θ ──┘
```

#### CG-DAN 구조 (모듈: Y→X₁, (Y,X₁)→X₂)
```
Y ──┐                    X₁ ──┐
    ├──► g₁ (NN) ──► X₁      ├──► g₂ (NN) ──► X₂
E₁ ─┤               E₂ ───────┘
    │               θ₂
θ₁ ─┘
```

#### 인과 구조 학습
- **단일 소스 도메인**: PC 알고리즘 사용 → Markov equivalence class까지 인과 그래프 학습
- **복수 소스 도메인**: **CD-NOD** 방법 사용 → 도메인 인덱스 $S$를 추가 변수로 포함하여 변화하는 인과 모듈 감지

---

### 2.4 식별 가능성(Identifiability) 이론

**Proposition 1** (기본 식별성):
> $P_{X|Y;\boldsymbol{\eta}}$에서 $\boldsymbol{\eta}$가 식별 가능하고, $n_s \to \infty$일 때 $P^s_{X|Y} = Q^s_{X|Y}$이면:
$$\hat{\boldsymbol{\theta}}_1 = \hat{\boldsymbol{\theta}}_2 \Rightarrow \eta_1 = \eta_2$$

**Proposition 2** (선형 기댓값 가정 하 완전 식별):
> $E_{Q_{X|Y;\boldsymbol{\theta},g}}[X] = A\boldsymbol{\theta} + h(Y)$이고 $\text{rank}(\Theta^\*) = d+1$ (즉 $m \geq d+1$ ), $\text{rank}(A^\*_{aug}) = d+1$이면, 추정된 $\hat{\boldsymbol{\theta}}$는 $\boldsymbol{\theta}^*$의 일대일 매핑:

```math
\boldsymbol{\theta}^* = A^{*\dagger}_{aug}\hat{A}_{aug}\hat{\boldsymbol{\theta}}
```

**Proposition 3** (단일 소스 도메인에서의 식별성):
> 모델 조건부 분포들이 선형 독립이고, $Q_{X|\boldsymbol{\theta}} = Q_{X|\boldsymbol{\theta}'}$이면:
$$Q_{X|Y;\boldsymbol{\theta}} = Q_{X|Y;\boldsymbol{\theta}'} \quad \text{and} \quad Q_Y = Q'_Y$$

이는 **단일 소스 + 레이블 없는 타겟** 상황에서도 잠재 변수와 결합 분포를 복원할 수 있음을 보장합니다.

---

### 2.5 성능 향상

#### MNIST-USPS (분류 정확도, %)

| CORAL | DAN | DANN | DSN | CoGAN | **G-DAN** |
|-------|-----|------|-----|-------|-----------|
| 81.7 | 81.1 | 91.3 | 91.2 | 95.7 | **95.9** |

G-DAN은 CoGAN 대비 소폭 향상, DANN/DSN 대비 유의미한 향상.

#### WiFi 실내 측위 (시간 전이, 정확도 %)

| 태스크 | KRR | TCA | SuK | DIP | CTC | G-DAN | **CG-DAN** |
|--------|-----|-----|-----|-----|-----|-------|------------|
| t1→t2 | 80.84 | 86.85 | 90.36 | 87.98 | 89.36 | 86.33 | **91.66** |
| t1→t3 | 76.44 | 80.48 | **94.97** | 84.20 | 94.80 | 83.91 | 93.17 |
| t2→t3 | 67.12 | 72.02 | 85.83 | 80.58 | 87.92 | 82.65 | **89.01** |

CG-DAN은 인과 구조 활용으로 G-DAN을 능가하며, 대부분의 태스크에서 최고 수준의 성능을 보임.

---

### 2.6 한계

1. **인과 구조 가정의 의존성**: 인과 그래프가 올바르게 학습되지 않으면 CG-DAN 성능이 저하됨
2. **이미지 데이터 미적용**: 이미지는 픽셀 간 인과 관계 모델링이 비현실적이므로 CG-DAN 적용 불가 → G-DAN만 사용
3. **소스 도메인 수 요구**: Proposition 2에 따르면 $m \geq d+1$개의 소스 도메인이 필요 (잠재 변수 차원 $d$에 비례)
4. **클래스 사전 분포 변화 미고려**: 논문 자체도 미래 연구로 남겨둔 한계
5. **비선형 식별성의 어려움**: $g$가 $\boldsymbol{\theta}$에 대해 비선형이면 두 도메인만으로는 완전한 식별이 불가능 (실험에서 D0°-D90° 케이스 실패)
6. **장치 전이 태스크 미적용**: WiFi 장치 전이 태스크에서 CG-DAN은 보고되지 않음 (인과 구조 불안정)

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 인과적 분해를 통한 일반화

CG-DAN의 핵심 일반화 메커니즘은 **인과 시스템의 모듈성(modularity)**에 기반합니다:

$$P_{XY} = P_Y \cdot \prod_{i} P_{X_i | \mathbf{PA}_i}$$

각 인과 모듈 $P_{X_i|\mathbf{PA}_i}$는 서로 독립적으로 변화하므로, **새로운 도메인은 기존 모듈들의 조합**으로 표현 가능합니다. 예를 들어 두 도메인에서 학습된 $(\boldsymbol{\theta}_1, \boldsymbol{\theta}_2)$와 $(\boldsymbol{\theta}'_1, \boldsymbol{\theta}'_2)$가 있다면, $(\boldsymbol{\theta}_1, \boldsymbol{\theta}'_2)$ 또한 유효한 인과 과정에 해당합니다. 이는 **지수적인 수의 새로운 도메인**을 생성할 수 있음을 의미합니다.

### 3.2 새로운 도메인 생성을 통한 데이터 증강

$$\boldsymbol{\theta}_{new} = \lambda \hat{\boldsymbol{\theta}}_s + (1-\lambda) \hat{\boldsymbol{\theta}}_t, \quad \lambda \in [0,1]$$

$\boldsymbol{\theta}$ 공간에서의 보간(interpolation)과 노이즈 $E$의 실현(realization)을 통해 **새로운 가상 도메인 데이터를 생성**할 수 있습니다. 이는 데이터 증강(data augmentation)의 역할을 하여 타겟 도메인의 일반화 성능을 향상시킵니다.

### 3.3 통계적 효율성 향상

CG-DAN에서 각 모듈의 잠재 변수 $\boldsymbol{\theta}_i$는 저차원이므로, **고차원 공간에서의 분포 추정 문제**를 회피합니다. 이는 **차원의 저주(curse of dimensionality)**를 완화하여 적은 데이터로도 더 나은 일반화가 가능합니다.

### 3.4 마르코프 담요(Markov Blanket) 활용

예측 목적의 DA에서는 $Y$의 Markov Blanket $MB(Y)$에 속하는 변수들만 고려하면 충분합니다:

$$Y \perp \{X \setminus MB(Y)\} \mid MB(Y)$$

이를 통해 불필요한 변수를 제거하고, 모델의 복잡도를 낮추어 **과적합(overfitting) 위험을 줄이고** 일반화 성능을 향상시킵니다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4.1 연구에 미치는 영향

**① 인과 추론과 도메인 적응의 통합**
이 논문은 인과 표현(causal representation)이 도메인 적응의 일반화 성능을 향상시킬 수 있음을 실증적으로 보였습니다. 이후 연구들은 인과적 불변성을 활용한 도메인 일반화(Domain Generalization) 방향으로 발전하고 있습니다.

**② 잠재 변수 기반 분포 변화 모델링**
$\boldsymbol{\theta}$로 도메인 변화를 명시적으로 파악하는 아이디어는 이후 **메타 학습(Meta-Learning)** 및 **연속적 도메인 적응(Continual DA)** 연구에 영향을 미쳤습니다.

**③ 새로운 도메인 생성 가능성**
$\boldsymbol{\theta}$ 공간 보간을 통한 새로운 도메인 생성은 이후 **데이터 증강 기반 도메인 일반화** 연구의 선구적 아이디어가 되었습니다.

---

### 4.2 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의**: 아래 비교는 제가 학습한 데이터를 바탕으로 하며, 논문 원문에 직접 인용된 내용이 아닙니다. 일부 세부 수치는 부정확할 수 있으므로 원문 확인을 권장합니다.

#### (A) 인과적 도메인 일반화 방향

**IRM (Invariant Risk Minimization, Arjovsky et al., 2019)**
- 모든 도메인에서 최적인 분류기를 유도하는 **불변 특징 표현** 학습
- CG-DAN과 달리 생성 모델이 아닌 판별 모델 기반
- 수식: $\min_\Phi \sum_e \mathcal{R}^e(\Phi) \text{ s.t. } w^* \in \arg\min_w \mathcal{R}^e(w \circ \Phi), \forall e$

| 항목 | CG-DAN | IRM |
|------|--------|-----|
| 접근 방식 | 생성 모델 + 인과 분해 | 판별 모델 + 불변 표현 |
| 분포 변화 처리 | 명시적 잠재 변수 | 암묵적 불변성 |
| 새 도메인 생성 | ✅ 가능 | ❌ 불가 |
| 이론적 보장 | 식별 가능성 명제 | Risk bound |

**DIRL (Domain-Invariant Representation Learning, Mahajan et al., 2021)**
- 인과 불변성을 활용하되, 더 강력한 표현 학습 아키텍처 사용

#### (B) 인과 표현 학습 방향

**CausalVAE (Yang et al., 2021)**
- VAE 구조에 인과 그래프를 통합하여 **해석 가능한 잠재 공간** 구성
- CG-DAN의 FCM 아이디어를 생성 모델(VAE)과 결합

**iVAE (Identifiable VAE, Khemakhem et al., 2020)**
- 보조 변수(auxiliary variable, 도메인 인덱스와 유사)를 활용한 비선형 ICA 식별 가능성 이론 정립
- CG-DAN의 Proposition들과 유사한 식별 가능성 조건을 더 엄밀하게 수학적으로 증명

$$p_{\theta}(\mathbf{x}) = \int p_f(\mathbf{x}|\mathbf{z}) p_{T,\lambda}(\mathbf{z}|\mathbf{u}) d\mathbf{z}$$

**비교**: iVAE는 CG-DAN보다 더 일반적인 식별 가능성 이론을 제공하지만, 도메인 적응의 실용적 측면은 덜 다룸.

#### (C) 대규모 사전학습 모델 활용 방향 (2022~)

**CLIP 기반 도메인 적응 (Gao et al., 2022 등)**
- 대규모 멀티모달 사전학습 모델(CLIP)을 활용한 제로샷(zero-shot) 도메인 적응
- CG-DAN의 명시적 분포 변화 모델링과는 대조적으로 **암묵적 일반화** 활용
- 한계: 인과 구조 무시, 해석 가능성 부족

| 항목 | CG-DAN | CLIP 기반 방법 |
|------|--------|--------------|
| 인과 구조 활용 | ✅ 명시적 | ❌ 없음 |
| 새 도메인 외삽 | ✅ 가능 | 제한적 |
| 계산 비용 | 낮음 | 매우 높음 |
| 레이블 요구 | 소스 도메인만 | 거의 불필요 |

---

### 4.3 앞으로 연구 시 고려할 점

**① 클래스 사전 분포 변화 통합**
논문 자체가 미래 연구로 남긴 $P_Y$ 변화를 $P_{X|Y}$ 변화와 동시에 처리하는 통합 프레임워크 개발이 필요합니다.

**② 인과 구조의 자동 학습**
현재 PC 알고리즘으로 인과 그래프를 학습하지만, 이미지 등 고차원 데이터에는 적용하기 어렵습니다. **신경망 기반 인과 발견(Causal Discovery)**과의 통합이 필요합니다.

$$\min_{A \in \text{DAG}} \mathcal{L}(A) + \lambda \|A\|_1 \quad \text{s.t.} \quad h(A) = 0$$
(NOTEARS 스타일의 연속 최적화)

**③ 다중 타겟 도메인으로의 확장**
현재는 단일 타겟 도메인을 가정하지만, **연속적·점진적 도메인 적응** 시나리오로의 확장이 필요합니다.

**④ 대규모 사전학습 모델과의 결합**
Foundation Model(GPT, CLIP 등)의 표현력을 $g$ 함수에 활용하면서도 인과 구조를 유지하는 하이브리드 접근이 유망합니다.

**⑤ 전이 가능성(Transferability) 정량화**
논문이 미래 연구로 제시한 "전이 가능성 수준 정량화"는 실용적으로 중요합니다. 이는 **어떤 도메인으로의 전이가 신뢰할 수 있는지**를 사전에 판단하는 데 필수적입니다.

**⑥ 비선형 식별 가능성 이론 강화**
현재 Proposition 2는 선형 기댓값 가정에 한정됩니다. 비선형 경우의 식별 가능성 조건을 확립하면 실용 범위가 크게 넓어집니다 (iVAE, Khemakhem et al., 2020 방향 참고).

---

## 참고 자료

**주요 참고 문헌 (논문 원문 인용 기준)**:
- **Gong et al. (2018)**: "Causal Generative Domain Adaptation Networks", arXiv:1804.04333v3
- **Pearl, J. (2000)**: *Causality: Models, Reasoning, and Inference*, Cambridge University Press
- **Spirtes, Glymour, Scheines (2001)**: *Causation, Prediction, and Search*, MIT Press
- **Zhang et al. (2013)**: "Domain adaptation under target and conditional shift", ICML
- **Gong et al. (2016)**: "Domain adaptation with conditional transferable components", ICML
- **Zhang et al. (2017)**: "Causal discovery from nonstationary/heterogeneous data" (CD-NOD), IJCAI
- **Ganin et al. (2016)**: "Domain-adversarial training of neural networks", JMLR
- **Goodfellow et al. (2014)**: "Generative adversarial nets", NIPS

**2020년 이후 비교 연구 (학습 데이터 기반, 원문 직접 확인 권장)**:
- **Arjovsky et al. (2019)**: "Invariant Risk Minimization", arXiv:1907.02893
- **Khemakhem et al. (2020)**: "Variational Autoencoders and Nonlinear ICA", AISTATS
- **Yang et al. (2021)**: "CausalVAE: Disentangled Representation Learning via Neural Structural Causal Models", CVPR
