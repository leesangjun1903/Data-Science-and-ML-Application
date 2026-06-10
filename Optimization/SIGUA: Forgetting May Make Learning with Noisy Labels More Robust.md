# SIGUA: Forgetting May Make Learning with Noisy Labels More Robust 

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

SIGUA의 핵심 주장은 **"잊는 것(forgetting)이 노이즈 레이블 학습을 더 강건하게 만들 수 있다"** 는 것입니다.

과파라미터화된 딥 네트워크는 훈련 데이터를 점진적으로 암기(memorize)하며, 결국 노이즈 레이블까지 모두 피팅합니다. 기존의 노이즈 레이블 보정 방법들도 **원치 않는 암기(undesired memorization)** 로 인한 과적합(overfitting)을 완전히 해결하지 못합니다. SIGUA는 이를 해결하기 위해:

- **좋은 데이터(good data)** → 일반적인 경사 하강(gradient descent)
- **나쁜 데이터(bad data)** → 학습률이 축소된 경사 상승(learning-rate-reduced gradient ascent)

을 미니배치 내에서 동시에 수행합니다.

### 주요 기여

| 기여 항목 | 내용 |
|-----------|------|
| **새로운 최적화 관점** | 최적화와 일반화의 목표가 충돌할 때 최적화를 "후퇴"시켜 일반화를 보호 |
| **철학적 통찰** | 원치 않는 암기를 잊는 것이 원하는 암기를 강화할 수 있음 |
| **다목적 프레임워크** | 기존 노이즈 레이블 학습 방법(self-teaching, backward correction 등)에 플러그인 가능 |
| **$\text{SIGUA}_\text{SL}$** | 샘플 선택(sample selection) 기반 방법 강화 |
| **$\text{SIGUA}_\text{BC}$** | 손실 보정(loss correction) 기반 방법 강화 |
| **nnBC 제안** | Non-negative backward correction을 추가로 제안 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

딥 네트워크는 **모든 것을 암기할 수 있는 능력(Zhang et al., 2017)** 을 가지고 있습니다. 노이즈 레이블 학습에서 이는 두 가지 종류의 암기로 나뉩니다:

$$\text{암기} = \underbrace{\text{desired memorization}}_{\text{일반화에 도움}} + \underbrace{\text{undesired memorization}}_{\text{일반화에 해로움}}$$

기존 방법들의 한계:
- **샘플 선택(Sample selection)**: 충분한 에포크 이후에야 손실이 신뢰 가능해지나, 그 시점에는 이미 많은 노이즈 데이터가 암기됨
- **역방향 보정(Backward correction)**: 보정된 손실 $\ell^\leftarrow$이 음수가 될 수 있으며, 이는 과도한 암기를 의미함

### 2.2 제안하는 방법 (수식 포함)

#### 기본 설정

훈련 데이터는 오염된 결합 밀도에서 샘플링됩니다:

$$\mathcal{S} = \{(x_i, \tilde{y}_i)\}_{i=1}^{n} \overset{\text{i.i.d.}}{\sim} p(x, \tilde{y}) = p(\tilde{y} \mid x) p(x)$$

분류 위험(classification risk):

$$R(f) = \mathbb{E}_{p(x,y)}[\ell(f(x), y)] \tag{1}$$

경험적 위험(empirical risk):

$$\widehat{R}(f) = \frac{1}{n} \sum_{i=1}^{n} \ell(f(x_i), y_i) \tag{2}$$

#### SIGUA 핵심 수식

미니배치 $\mathcal{S}_b$에 대해 손실 누적기를 다음과 같이 계산합니다:

$$\ell_b \leftarrow \left(\boldsymbol{\ell}_b^\top (\mathcal{C}_\text{good}(\mathcal{S}_b) - \gamma \mathcal{C}_\text{bad}(\mathcal{S}_b))\right) / n_b \tag{3}$$

여기서:
- $\mathcal{C}\_\text{good}(\mathcal{S}\_b) - \gamma \mathcal{C}_\text{bad}(\mathcal{S}_b)$의 각 원소는:
  - 좋은 데이터: $+1$
  - 불확실한 데이터: $0$
  - 나쁜 데이터: $-\gamma$
- $\gamma \in [0, 1]$: 경사 상승의 강도를 제어하는 하이퍼파라미터

#### Algorithm 1 (SIGUA Prototype)

```
Require: 기저 알고리즘 B, 옵티마이저 O, 미니배치 Sb, 
         모델 fθ, 조건 Cgood/Cbad, 언더웨이트 파라미터 γ

1: {ℓi} ← B.forward(fθ, Sb)       # 순전파
2: ℓb ← 0
3: for i = 1, ..., nb do
4:   if Cgood(xi, ỹi): ℓb ← ℓb + ℓi      # 경사 하강
6:   elif Cbad(xi, ỹi): ℓb ← ℓb - γℓi   # 경사 상승
8: end for
10: ℓb ← ℓb / nb
11: ∇θ ← B.backward(fθ, ℓb)
12: O.step(∇θ)
```

#### $\text{SIGUA}_\text{SL}$: 자기 교수(Self-Teaching) 강화

선택률(selection rate):

$$\rho(t) = 1 - \epsilon \cdot \min(t/T_k, 1) \tag{4}$$

좋은 데이터 조건:

$$\mathcal{C}_\text{good}(x_i, \tilde{y}_i) = \mathbb{I}\!\left(\sum_{j=1}^{n_b} \mathbb{I}(\ell_i > \ell_j) \leq n_b \rho(t)\right) \tag{5}$$

나쁜 데이터 조건 (중간 손실 데이터 선택):

$$\mathcal{C}_\text{bad}(x_i, \tilde{y}_i) = \neg \mathcal{C}_\text{good}(x_i, \tilde{y}_i) \wedge \mathbb{I}\!\left(\sum_{j=1}^{n_b} \mathbb{I}(\ell_i > \ell_j) \leq n_b \rho(t) + n_b \delta(t)\right) \tag{6}$$

> **중요한 직관**: 대손실 데이터는 아직 잘 암기되지 않아 $\mathcal{C}_\text{good}$을 혼동시킬 가능성이 낮습니다. 반면, 중간 손실 데이터는 상대적으로 잘 암기되어 있으며 노이즈 레이블일 가능성이 높습니다.

#### $\text{SIGUA}_\text{BC}$: 역방향 보정(Backward Correction) 강화

전이 행렬(transition matrix) $T \in \mathbb{R}^{k \times k}_+$를 사용한 보정 손실:

$$\ell^\leftarrow(f(x), \tilde{y}) = [T^{-1} \boldsymbol{\ell}_{y|f(x)}]_{\tilde{y}} \tag{8}$$

이는 위험 일관성(risk-consistent)을 보장합니다:

$$\mathbb{E}_\mathcal{S}[\widehat{R}^\leftarrow(f)] = \mathbb{E}_{p(x,\tilde{y})}[\ell^\leftarrow(f(x), \tilde{y})] = R(f) \tag{9}$$

그러나 $T^{-1} \in \mathbb{R}^{k \times k}$ (부호 불확정)이므로 $\ell^\leftarrow$이 음수가 될 수 있습니다. 다음 관계를 활용합니다:

$$\boldsymbol{p}_{\tilde{y}|x}^\top \boldsymbol{\ell}^\leftarrow_{\tilde{y}|f(x)} = \boldsymbol{p}_{y|x}^\top \boldsymbol{\ell}_{y|f(x)} \geq 0 \tag{10}$$

균등 분포로 $\boldsymbol{p}_{\tilde{y}|x}$를 대체하면:

$$\mathcal{C}_\text{good}(x_i, \tilde{y}_i) = \mathbb{I}\!\left(\mathbf{1}^\top \boldsymbol{\ell}^\leftarrow_{\tilde{y}|f(x)} \geq 0\right) \tag{11}$$

$$\mathcal{C}_\text{bad}(x_i, \tilde{y}_i) = \neg \mathcal{C}_\text{good}(x_i, \tilde{y}_i) \tag{12}$$

### 2.3 모델 구조

SIGUA는 특정 아키텍처를 요구하지 않는 **최적화 수준의 프레임워크**입니다.

실험에서 사용된 주요 아키텍처:

| 데이터셋 | 아키텍처 |
|----------|----------|
| MNIST/CIFAR-10/100 | 9-layer CNN (3 blocks, Conv-BN-LReLU) |
| CIFAR-10 (오픈셋) | 6-layer CNN |
| NEWS (SET1) | 1D CNN + GloVe 임베딩 |
| NEWS (SET2) | MLP + GloVe 임베딩 |

또한 ResNet-18로 교체 시 성능이 더욱 향상됨을 확인:

| 방법 | Symmetry-20% | Symmetry-50% | Pair-45% |
|------|-------------|-------------|---------|
| 9-layer CNN | 84.05% | 77.12% | 81.82% |
| ResNet-18 | **89.41%** | **81.96%** | **89.56%** |

### 2.4 성능 향상

#### MNIST (80% 대칭 노이즈) 실험 결과

| 방법 | 뒤집힌 데이터 망각률 | 테스트 정확도 |
|------|-------------|-------------|
| StopGrad (Wide net) | 5% | 59% |
| StopGrad (Deep net) | 0% | 41% |
| SIGUA (Wide/Deep net) | **99%** | **95%** |

#### CIFAR-10 오픈셋 노이즈 (40%) 결과

| Standard | Self-Teach | $\text{SIGUA}_\text{SL}$ | BC | nnBC | $\text{SIGUA}_\text{BC}$ |
|----------|-----------|----------------------|-----|------|----------------------|
| 56.44% | 79.72% | **81.31%** | 52.03% | 73.39% | 74.33% |

#### MNIST에서 지도 학습과의 비교

| 방법 | Symmetry-20% | Symmetry-50% | Pair-45% |
|------|-------------|-------------|---------|
| 지도 학습(클린) | 99.61% | 99.61% | 99.61% |
| $\text{SIGUA}_\text{SL}$ | 98.91% | 98.10% | 89.37% |
| $\text{SIGUA}_\text{BC}$ | **99.42%** | 97.73% | **99.47%** |

### 2.5 한계점

1. **하이퍼파라미터 $\gamma$ 민감성**: $\gamma$ 값이 부적절할 경우 Adam 같은 적응형 최적화기에서 훈련이 불안정해질 수 있음
2. **$\epsilon$ 또는 $T$ 사전 지식 필요**: 실험에서 진짜 값을 사용했으므로, 실제 환경에서의 추정 오류가 성능에 영향을 미칠 수 있음
3. **인스턴스 의존 노이즈(Instance-dependent noise)에 대한 한계**: CCN(Class-Conditional Noise) 가정 하에서만 이론적 보장이 있으며, 인스턴스 의존 노이즈는 다른 접근이 필요
4. **$\delta(t)$ 설계의 임시방편성**: 잊어버릴 데이터 비율 $\delta(t)$를 데이터셋별로 수동 조정해야 함
5. **이진 분류에서의 특수성**: 이진 분류에서는 SIGUA가 레이블 보정과 동일해지나, 다중 분류($k \geq 3$)에서는 다름

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 원치 않는 암기 억제를 통한 일반화

딥 네트워크의 학습 과정은 다음 단계로 진행됩니다 (Arpit et al., 2017):

$$\underbrace{\text{패턴 학습}}_{\text{desired memorization}} \rightarrow \underbrace{\text{노이즈 암기}}_{\text{undesired memorization}}$$

SIGUA는 두 번째 단계에서 경사 상승을 통해 **원치 않는 암기를 능동적으로 지움**으로써 이 과정을 방해합니다.

### 3.2 모델 용량 관점의 일반화 향상

딥 러닝 이론(Suzuki, 2019)에 따르면:
- 딥 네트워크는 타깃 함수의 **공간적 비균질성(spatial inhomogeneity)** 에 높은 적응성을 가집니다
- 잘못 레이블된 데이터는 **점프 불연속성(jump discontinuity)** 을 요구하므로, 올바른 데이터보다 훨씬 많은 모델 용량을 소비합니다

$$\text{잘못 레이블 데이터 암기} \Rightarrow \text{과도한 모델 용량 소비} \Rightarrow \text{일반화 저하}$$

SIGUA의 망각은 이 소비된 용량을 회수하여 원하는 암기에 재투자합니다:

$$\text{망각(forgetting)} \Rightarrow \text{용량 회수} \Rightarrow \text{desired memorization 강화} \Rightarrow \text{일반화 향상}$$

### 3.3 최적화-일반화 트레이드오프

기존에 알려진 학습 단계별 특성:

| 단계 | 최적화 목표 | 일반화 목표 | 상태 |
|------|------------|------------|------|
| 초기 | 일치 | 일치 | 정상 |
| 중기 | 수렴 중 | 일치 | 최적 |
| 후기 | 계속 하강 | 더 이상 일치 안 함 | **과적합** |

SIGUA는 이 후기 단계에서 **나쁜 데이터에 경사 상승을 적용**함으로써 최적화를 의도적으로 제약하여 일반화를 보호합니다. 이는 early stopping보다 우월한데, epoch-wise double descent(Nakkiran et al., 2020) 현상 때문에 early stopping이 항상 최적 시점을 찾기 어렵기 때문입니다.

### 3.4 일반화 성능 향상의 실증적 증거

CIFAR-100 실험에서 특히 주목할만한 현상이 관찰됩니다:
- 전반부에 정확도가 크게 하락한 후
- $\text{SIGUA}_\text{SL}$ 적용 시 후반부에 정확도가 **early stopping으로 얻을 수 있는 최고 정확도를 초과**
- 이는 test error 기준으로 **epoch-wise double descent**(Nakkiran et al., 2020)에 해당

$$\text{SIGUA} \overset{\text{empirically}}{\approx} \text{지도 학습(클린 레이블)}$$

MNIST에서 $\text{SIGUA}_\text{BC}$는 Pair-45% 노이즈 하에서도 99.47%를 달성하며 지도 학습(99.61%)에 근접합니다.

### 3.5 보완 레이블 학습과의 연결

이진 분류에서, 대칭 손실 $\ell$이 $\ell(t, +1) + \ell(t, -1) = \text{Const.}$를 만족하면:

$$-\nabla_\theta \ell(f_\theta(x_i), \tilde{y}_i) = \nabla_\theta \ell(f_\theta(x_i), -\tilde{y}_i)$$

즉, SIGUA의 경사 상승은 레이블을 뒤집은 것에 대한 경사 하강과 동일합니다. 다중 분류에서는 어느 클래스가 정확한지 알 수 없으므로, **$x_i$가 클래스 $\tilde{y}_i$에 속한다는 잘못된 정보를 모델이 잊도록** 유도합니다. 이는 보완 레이블 학습(Ishida et al., 2017; 2019)과 암묵적으로 연결되며, 일반화 성능에 직접 기여합니다.

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

아래 분석은 SIGUA 논문(2020)의 내용을 기반으로 하되, 2020년 이후 노이즈 레이블 학습 분야의 일반적인 연구 흐름을 함께 기술합니다. **단, 이후 논문들의 구체적 수치 및 세부 내용은 해당 논문을 직접 확인하시기 바랍니다.**

### 4.1 SIGUA와 동시대 및 이후 연구 비교

| 연구 방향 | 대표 연구 | SIGUA와의 관계 |
|----------|----------|--------------|
| **혼합 훈련(Mixup 기반)** | DivideMix (Li et al., 2020, ICLR) | 두 모델로 데이터 분리 후 반지도 학습; SIGUA와 직교적 접근 |
| **인스턴스 의존 노이즈** | Berthon et al. (2020); Cheng et al. (2020, ICML) | SIGUA의 CCN 가정을 완화하는 방향 |
| **전이 행렬 추정** | Dual-T (Yao et al., 2020); Xia et al. (2019, NeurIPS) | SIGUA의 $T$ 추정을 개선하면 $\text{SIGUA}_\text{BC}$ 성능 향상 가능 |
| **샘플 선택 개선** | Co-learning, SELF 등 | SIGUA는 이들을 기저 알고리즘으로 사용 가능 |
| **정규화 관점** | Ishida et al. (2020, ICML) - "Do we need zero training loss?" | SIGUA의 최적화-일반화 트레이드오프 이론과 직접 연결 |

### 4.2 SIGUA의 차별성

```
기존 방법들:
  - 샘플 선택: 나쁜 데이터를 무시 (Stop Gradient)
  - 손실 보정: 손실 함수를 수정
  - 레이블 보정: 레이블을 추정·수정

SIGUA:
  - 나쁜 데이터에 능동적으로 경사 상승 적용
  - 기존 방법들과 직교적(orthogonal)이며 플러그인 가능
  - "잊는 것"을 일반화의 도구로 활용
```

### 4.3 한계 측면에서의 최신 연구 대응

| SIGUA의 한계 | 이후 연구 방향 |
|------------|-------------|
| CCN 가정 | 인스턴스 의존 노이즈 연구(Xia et al., 2020; Berthon et al., 2020) |
| $T$ 사전 지식 필요 | Dual-T(Yao et al., 2020), CORES 등 추정 개선 |
| $\gamma$ 튜닝 | 자동 하이퍼파라미터 탐색 연구와 결합 가능 |
| 단순 아키텍처 | ResNet-18 적용 시 추가 성능 향상 확인 (논문 내) |

---

## 5. 앞으로의 연구에 미치는 영향과 고려할 점

### 5.1 연구에 미치는 영향

#### 철학적·방법론적 영향

1. **"잊는 것의 가치" 재조명**: 딥러닝에서 망각(forgetting)이 단순한 부작용이 아니라 **의도적으로 활용 가능한 도구**임을 보임

2. **플러그인 프레임워크의 가능성**: SIGUA는 다양한 기저 학습 방법에 결합 가능한 범용 프레임워크로, 향후 연구가 더 복잡한 기저 방법(DivideMix, Co-teaching 등)과 결합할 수 있는 가능성을 열어줌

3. **최적화-일반화 트레이드오프의 명시적 제어**: 기존 early stopping이나 weight decay 대신, **데이터 수준에서 최적화를 선택적으로 제어**하는 새로운 패러다임 제시

4. **보완 레이블 학습과의 연결**: 경사 상승이 보완 레이블 학습과 동치임을 보여, 두 분야의 융합 연구 가능성 제시

#### 실용적 영향

- PyTorch/TensorFlow의 동적 계산 그래프를 활용하면 구현이 매우 단순하므로, 실제 응용에서 쉽게 채택 가능
- 의료 영상, 크라우드소싱 레이블, 웹 크롤링 데이터 등 노이즈가 많은 실제 데이터셋에 직접 적용 가능

### 5.2 앞으로 연구 시 고려할 점

#### 이론적 고려사항

1. **인스턴스 의존 노이즈로의 확장**

$$p(\tilde{y} \mid x, y) \neq p(\tilde{y} \mid y)$$

인스턴스 의존 노이즈에서 $\mathcal{C}\_\text{good}$과 $\mathcal{C}_\text{bad}$를 어떻게 정의할지에 대한 이론적 분석이 필요합니다.

2. **수렴 보장(Convergence Guarantee)**

경사 상승과 하강이 혼합된 최적화의 수렴 조건에 대한 이론적 분석이 부족합니다. 특히 Adam 같은 적응형 옵티마이저와의 상호작용에 대한 엄밀한 이론이 필요합니다.

3. **$\gamma$의 이론적 최적값 도출**

$$\gamma^* = \arg\min_{\gamma \in [0,1]} \mathbb{E}[\text{generalization error}]$$

현재는 검증 데이터로 수동 튜닝하는데, 이를 이론적으로 도출하거나 자동화하는 연구가 필요합니다.

4. **Epoch-wise double descent와의 관계**

SIGUA가 double descent 현상을 어떻게 변형시키는지에 대한 이론적 이해가 필요합니다.

#### 실험적 고려사항

5. **더 강력한 기저 방법과의 결합**

DivideMix(Li et al., 2020), ELR+(Liu et al., 2020) 등 최신 방법들을 기저 알고리즘으로 사용했을 때의 성능 향상 검증이 필요합니다.

6. **대규모 실세계 데이터셋 검증**

WebVision, Clothing1M 같은 실세계 노이즈 데이터셋에서의 성능 검증이 부족합니다.

7. **$\delta(t)$의 자동 결정**

현재 $\delta(t)$는 데이터셋별로 수동 설정됩니다. 온라인 또는 적응적으로 $\delta(t)$를 결정하는 메커니즘 연구가 필요합니다.

8. **반지도 학습(Semi-supervised Learning)과의 통합**

SIGUA의 불확실 데이터(uncertain data, $\mathcal{C}\_\text{good} = \mathcal{C}_\text{bad} = \text{false}$)를 레이블 없는 데이터로 활용하는 반지도 학습 접근과의 통합을 고려할 수 있습니다.

9. **대조 학습(Contrastive Learning)과의 결합**

최근 자기지도 학습(self-supervised learning)의 발전과 결합하여, 표현 학습 단계에서 SIGUA를 적용하는 연구가 흥미로울 수 있습니다.

10. **LLM 파인튜닝 맥락에서의 적용**

대형 언어 모델(LLM)의 파인튜닝 시 노이즈 레이블 문제는 더욱 심각해질 수 있으며, SIGUA의 아이디어를 이 맥락에 적용하는 연구가 유망합니다. 특히 RLHF(Reinforcement Learning from Human Feedback)에서 인간 피드백의 노이즈를 처리하는 데 적용 가능성이 있습니다.

---

## 참고자료

**주요 참고 논문 (논문 내 인용)**

1. **Han, B., Niu, G., Yu, X., et al. (2020)**. "SIGUA: Forgetting May Make Learning with Noisy Labels More Robust." *ICML 2020, PMLR 119*. arXiv:1809.11008v3

2. **Arpit, D., et al. (2017)**. "A closer look at memorization in deep networks." *ICML 2017*

3. **Zhang, C., et al. (2017)**. "Understanding deep learning requires rethinking generalization." *ICLR 2017*

4. **Patrini, G., et al. (2017)**. "Making deep neural networks robust to label noise: a loss correction approach." *CVPR 2017*

5. **Han, B., et al. (2018b)**. "Co-teaching: Robust training of deep neural networks with extremely noisy labels." *NeurIPS 2018*

6. **Jiang, L., et al. (2018)**. "MentorNet: Learning data-driven curriculum for very deep neural networks on corrupted labels." *ICML 2018*

7. **Nakkiran, P., et al. (2020)**. "Deep double descent: Where bigger models and more data hurt." *ICLR 2020*

8. **Ishida, T., et al. (2020)**. "Do we need zero training loss after achieving zero training error?" *ICML 2020*

9. **Suzuki, T. (2019)**. "Adaptivity of deep ReLU network for learning in Besov and mixed smooth Besov spaces." *ICLR 2019*

10. **Xia, X., et al. (2019)**. "Are anchor points really indispensable in label-noise learning?" *NeurIPS 2019*

11. **Kiryo, R., et al. (2017)**. "Positive-unlabeled learning with non-negative risk estimator." *NeurIPS 2017*

12. **Natarajan, N., et al. (2013)**. "Learning with noisy labels." *NeurIPS 2013*

13. **Kingma, D. and Ba, J. (2015)**. "Adam: A method for stochastic optimization." *ICLR 2015*

14. **Vapnik, V. N. (1998)**. *Statistical Learning Theory*. John Wiley & Sons

15. **Ishida, T., et al. (2017)**. "Learning from complementary labels." *NeurIPS 2017*
