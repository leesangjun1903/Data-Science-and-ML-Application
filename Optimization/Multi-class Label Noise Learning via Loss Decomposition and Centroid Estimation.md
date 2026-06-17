# Multi-class Label Noise Learning via Loss Decomposition and Centroid Estimation (MC-LDCE)

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

본 논문의 핵심 주장은 다음과 같습니다:

> **기존의 LDCE(Loss Decomposition and Centroid Estimation) 기반 LNL(Label Noise Learning) 방법들은 이진 분류(binary classification)에만 적용 가능하며, 다중 클래스(multi-class) 분류로 직접 확장할 수 없다. 이를 해결하기 위해 다중 클래스 설정에서 동작하는 새로운 MC-LDCE 프레임워크를 제안한다.**

### 주요 기여

| 기여 항목 | 내용 |
|-----------|------|
| **최초의 다중 클래스 LDCE 확장** | 기존 이진 분류 전용 LDCE를 다중 클래스로 일반화 |
| **다중 클래스 손실 분해(Multi-class Loss Decomposition)** | 평균 제곱 손실(MSE)을 레이블 독립 부분과 레이블 의존 부분으로 분해 |
| **새로운 데이터 센트로이드 정의** | 다중 클래스를 위한 행렬 형태의 centroid $\hat{\mu}(S) = \frac{1}{n}\sum_{i=1}^n \mathbf{x}_i \mathbf{y}_i^\top$ 도입 |
| **전이 행렬 기반 센트로이드 추정** | Imputation Matrix와 VolMinNet을 활용한 노이즈 전이 행렬 기반 센트로이드 복원 |
| **모델 독립적 프레임워크** | 선형 모델 및 딥러닝(CNN, MLP) 모두에 적용 가능 |
| **보조 정제 데이터 불필요** | 추가적인 clean 데이터 없이 학습 가능 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**문제 배경:**

실세계에서 대규모 데이터셋을 수집할 때, 웹 크롤링, 크라우드소싱 등의 방법으로 레이블을 획득하면 **노이즈 레이블(Noisy Labels)** 이 필연적으로 발생합니다. 이는 모델의 성능 저하를 유발합니다.

기존 LDCE 기반 방법들(LICS, µSGD, CEGE)은 다음 두 가지 이유로 다중 클래스 분류에 적용 불가합니다:

1. **힌지 손실(hinge loss), 퍼셉트론 손실(perceptron loss)** 등 이진 손실만 분해 가능
2. 이진 레이블의 특성인 $y^2 = 1$ (where $y \in \{+1, -1\}$)과 부호 반전 관계가 다중 클래스에서 성립하지 않음

---

### 2.2 제안 방법 (수식 포함)

#### Step 1: 문제 정의

- 입력 공간: $\mathcal{X} \in \mathbb{R}^d$
- 출력 레이블 공간: $\mathcal{Y} = \{0,1\}^c$ (one-hot 벡터, $c$는 클래스 수)
- 노이즈 전이 행렬: $\mathbf{T} \in [0,1]^{c \times c}$

$$T_{ij} = p(\tilde{Y} = \mathbf{e}_j \mid Y = \mathbf{e}_i)$$

즉, $T_{ij}$는 클래스 $i$의 정답 레이블이 클래스 $j$로 잘못 레이블링될 확률입니다.

- Clean set: $S = \{(\mathbf{x}\_i, \mathbf{y}\_i)\}_{i=1}^n$
- Noisy set: $\tilde{S} = \{(\mathbf{x}\_i, \tilde{\mathbf{y}}\_i)\}_{i=1}^n$

**목표:** 노이즈 집합 $\tilde{S}$를 이용하여 clean set $S$에 대한 경험적 위험(empirical risk)의 불편 추정량(unbiased estimator)을 구성한다.

---

#### Step 2: 다중 클래스 손실 분해 (Multi-class Loss Decomposition)

결정 함수를 $h(\mathbf{x}; \mathbf{W}) = \langle \mathbf{W}, \mathbf{x} \rangle = \mathbf{W}^\top \mathbf{x}$ 로 정의하고, 평균 제곱 손실을 사용하면 clean set $S$에서의 경험적 위험은:

$$\hat{\mathcal{R}}(h, S) = \frac{1}{n} \sum_{i=1}^n \|\mathbf{y}_i - \mathbf{W}^\top \mathbf{x}_i\|_2^2 \tag{4.3}$$

이를 전개하면:

$$= \frac{1}{n} \sum_{i=1}^n \left( \mathbf{y}_i^\top \mathbf{y}_i - 2\mathbf{y}_i^\top \mathbf{W}^\top \mathbf{x}_i + \mathbf{x}_i^\top \mathbf{W}\mathbf{W}^\top \mathbf{x}_i \right)$$

One-hot 벡터에서 $\mathbf{y}_i^\top \mathbf{y}_i = 1$ 이므로, 선형대수의 trace 성질:

$$\mathbf{y}_i^\top \mathbf{W}^\top \mathbf{x}_i = \text{trace}(\mathbf{W}^\top \mathbf{x}_i \mathbf{y}_i^\top) \tag{4.4}$$

을 적용하면:

$$\hat{\mathcal{R}}(h, S) = \underbrace{\left(1 + \frac{1}{n}\sum_{i=1}^n \mathbf{x}_i^\top \mathbf{W}\mathbf{W}^\top \mathbf{x}_i\right)}_{\text{레이블 독립 부분}} - \underbrace{2\,\text{trace}\left(\mathbf{W}^\top \hat{\mu}(S)\right)}_{\text{레이블 의존 부분}} \tag{4.5}$$

**핵심:** 레이블 값에 영향받는 부분은 두 번째 항뿐이며, 이를 **데이터 센트로이드(Data Centroid)** 로 표현합니다:

$$\hat{\mu}(S) = \frac{1}{n} \sum_{i=1}^n \mathbf{x}_i \mathbf{y}_i^\top \in \mathbb{R}^{d \times c} \tag{4.6}$$

$$\mu(\mathcal{D}) = \mathbb{E}_{(X,Y)\sim\mathcal{D}}[XY^\top] \tag{4.7}$$

> ⚠️ 이 센트로이드 정의는 기존 이진 분류의 스칼라 형태와 달리 **행렬 형태** 로 정의되어 다중 클래스를 처리합니다.

---

#### Step 3: 센트로이드 추정 (Centroid Estimation)

노이즈 집합 $\tilde{S}$의 센트로이드 $\hat{\mu}(\tilde{S})$를 이용하여 clean 센트로이드 $\hat{\mu}(S)$를 추정합니다.

**Imputation Matrix 정의:**

두 one-hot 벡터 간의 변환 행렬 $\mathbf{K}_{i \to j}$ (단위 행렬의 $i$번째와 $j$번째 행을 교환):

$$\mathbf{y}_j = \mathbf{K}_{i \to j} \mathbf{y}_i \tag{4.9}$$

이를 이용하면, 클래스 $i$에 속한 샘플의 노이즈 레이블에 대한 기댓값:

$$\mathbb{E}_{\tilde{Y}}[X\tilde{Y}^\top \mid (X, Y = \mathbf{e}_i)] = \sum_{j=1}^c T_{ij} XY^\top \mathbf{K}_{i \to j}^\top \tag{4.10}$$

전체 분포에 대해 클래스 사전 확률 $\pi_i = P(Y = \mathbf{e}_i)$ 로 가중합:

$$\mathbb{E}_{\tilde{Y}}[X\tilde{Y}^\top \mid (X, Y)] = \sum_{i=1}^c \pi_i \sum_{j=1}^c T_{ij} XY^\top \mathbf{K}_{i\to j}^\top = XY^\top \underbrace{\left[\sum_{i=1}^c \pi_i \sum_{j=1}^c T_{ij} \mathbf{K}_{i\to j}^\top\right]}_{\mathbf{M}} \tag{4.11}$$

따라서 clean 센트로이드의 불편 추정량:

$$\tilde{\mu}(S) = \hat{\mu}(\tilde{S}) \mathbf{M}^\dagger \tag{4.12}$$

여기서 $\mathbf{M}^\dagger$는 $\mathbf{M}$의 유사 역행렬(pseudo-inverse)입니다.

---

#### Step 4: 불편 위험 추정량 (Unbiased Risk Estimator)

식 (4.12)를 식 (4.5)에 대입하면 최종 **MC-LDCE의 불편 위험 추정량**:

$$\tilde{\hat{\mathcal{R}}}(h, \tilde{S}) = 1 + \frac{1}{n}\sum_{i=1}^n \mathbf{x}_i^\top \mathbf{W}\mathbf{W}^\top \mathbf{x}_i - 2\,\text{trace}\left(\mathbf{W}\hat{\mu}(\tilde{S})\mathbf{M}^\dagger\right) \tag{4.13}$$

---

#### Step 5: 클래스 사전 확률 추정 (Class Prior Estimation)

$\pi_1, \ldots, \pi_c$는 다음 선형 연립방정식을 풀어 추정합니다:

$$\begin{cases} P(\tilde{Y} = \mathbf{e}_1) = T_{11}\pi_1 + T_{21}\pi_2 + \cdots + T_{c1}\pi_c \\ \quad\quad\quad\quad\quad\quad\quad\quad \vdots \\ P(\tilde{Y} = \mathbf{e}_i) = T_{1i}\pi_1 + T_{2i}\pi_2 + \cdots + T_{ci}\pi_c \\ \quad\quad\quad\quad\quad\quad\quad\quad \vdots \\ P(\tilde{Y} = \mathbf{e}_c) = T_{1c}\pi_1 + T_{2c}\pi_2 + \cdots + T_{cc}\pi_c \end{cases} \tag{4.14}$$

좌변은 노이즈 집합 $\tilde{S}$에서 직접 추정 가능하며, 전이 행렬 $\mathbf{T}$는 **VolMinNet** (Li et al., 2021)을 통해 추정합니다.

---

### 2.3 전체 알고리즘 (Algorithm 1)

```
입력: 노이즈 학습 데이터 S̃ = {(xᵢ, ỹᵢ)}ⁿᵢ₌₁
1. VolMinNet으로 전이 행렬 T 추정
2. 식 (4.14)로 클래스 사전 확률 π₁,...,πc 계산
3. 식 (4.11)로 M 계산
4. 식 (4.12)로 추정 센트로이드 μ̃(S) 계산
5. 식 (4.13)으로 불편 위험 추정량 R̃̂(h, S̃) 계산
6. R̃̂(h, S̃)를 손실 함수로 사용하여 모델 최적화
출력: 최적 파라미터 W
```

---

### 2.4 모델 구조

| 구분 | 아키텍처 | 적용 데이터셋 |
|------|----------|---------------|
| **6층 CNN** | 3×3 conv (128→128→128) + MaxPool + Dropout + 3×3 conv (512→256→128) + AvgPool + FC | CIFAR-10, SVHN, Animal-10N |
| **2층 MLP** | FC (784→256, LReLU) + FC (256→#classes) | MNIST, FASHION-MNIST |
| **선형 모델** | 단일 선형 분류기 | CIFAR-10 (선형 실험) |

**최적화:** Adam (lr=0.001, momentum=0.9), 200 epochs, batch size 128, 80 epoch 이후 선형 학습률 감소

---

### 2.5 성능 향상 및 한계

#### 성능 향상

**딥 모델 실험 결과 (Table 3, Table 4):**

| 데이터셋 | 노이즈 유형 | 2위 방법 | MC-LDCE | 향상 |
|---------|------------|---------|---------|------|
| CIFAR-10 | Pairflip-20% | JoCoR: 83.63% | **최고** | +1.83% |
| CIFAR-10 | Pairflip-40% | JoCoR: 62.95% | **최고** | +6.85% |
| Animal-10N | 실제 노이즈 | JoCoR: 75.7% | **76.6%** | +0.9% |

**선형 모델 실험 결과 (Table 5, CIFAR-10):**

| 노이즈율 | LNSI | SCD | **MC-LDCE** |
|---------|------|-----|------------|
| 20% | 84.7% | 86.5% | **87.1%** |
| 40% | 83.8% | 84.5% | **85.1%** |
| 60% | 77.4% | 77.6% | **79.7%** |

#### 한계점

1. **전이 행렬 T의 정확한 추정 의존성:** $\mathbf{M}$ 계산이 $\mathbf{T}$ 추정 품질에 크게 의존하며, 전이 행렬 추정 오류가 전파될 수 있음
2. **인스턴스 의존 노이즈(Instance-dependent Noise) 미처리:** 본 논문은 클래스 조건부 노이즈(class-conditional noise)만 가정하므로, 샘플별로 노이즈율이 다른 경우 적용 한계 존재
3. **평균 제곱 손실(MSE) 중심:** 크로스 엔트로피 등 다른 손실 함수로의 일반화가 명시적으로 다루어지지 않음
4. **높은 클래스 수에서의 계산 비용:** $\mathbf{M} \in \mathbb{R}^{c \times c}$ 행렬의 유사 역행렬 계산이 클래스 수가 매우 많을 때 비효율적일 수 있음
5. **보조 정보(Auxiliary Information) 불활용:** 일부 최신 방법들이 활용하는 메타 학습, 반지도 학습 정보를 사용하지 않음

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 불편 위험 추정량이 일반화에 기여하는 메커니즘

MC-LDCE의 핵심적인 일반화 기여는 **수학적으로 보장된 불편 추정량** 구성에 있습니다.

$$\mathbb{E}_{\tilde{S}}[\tilde{\hat{\mathcal{R}}}(h, \tilde{S})] = \hat{\mathcal{R}}(h, S)$$

즉, 노이즈 데이터에서 학습하더라도, 기대값 측면에서 clean 데이터에서의 위험과 동일한 추정량을 최소화하게 됩니다. 이는 다음과 같은 일반화 이점을 제공합니다:

**① 과적합 방지:**
- 기존 노이즈 레이블 학습 시 발생하는 **Memorization Effect** (신경망이 초기에 clean 샘플을 학습하다가 점차 노이즈 샘플에 과적합되는 현상)를 원천 차단
- 실험에서 Co-teaching+, GCE가 60% 대칭 노이즈에서 정확도가 먼저 오르다가 하락하는 반면, MC-LDCE는 단조 증가하는 것으로 확인됨

**② 레이블 의존 부분의 정확한 복원:**

$\hat{\mu}(S)$의 추정 오차 $\epsilon = \|\hat{\mu}(S) - \tilde{\mu}(S)\|$가 작을수록, 불편 위험 추정량 (4.13)의 분산이 감소하고 일반화 오차가 줄어듭니다:

$$\tilde{\hat{\mathcal{R}}}(h, \tilde{S}) - \hat{\mathcal{R}}(h, S) = -2\,\text{trace}\left(\mathbf{W}^\top \epsilon\right)$$

**③ 정규화와의 시너지:**

식 (4.13)에 $\ell_2$ 정규화 또는 Dropout을 결합하여 과적합을 추가로 억제합니다:

- **선형 모델:** $\ell_2$ 정규화 적용
- **딥 모델:** Dropout(p=0.25) 적용

이는 불편 추정량 기반 손실 최소화와 정규화의 결합으로, **편향-분산 트레이드오프**를 효과적으로 조절합니다.

---

### 3.2 모델 독립성(Model-agnostic)이 일반화에 미치는 영향

MC-LDCE는 분류 모델 $h(\mathbf{x})$의 종류에 무관하게 적용 가능합니다:

$$\tilde{\hat{\mathcal{R}}}(h, \tilde{S}) = 1 + \frac{1}{n}\sum_{i=1}^n \mathbf{x}_i^\top \mathbf{W}\mathbf{W}^\top \mathbf{x}_i - 2\,\text{trace}\left(\mathbf{W}\hat{\mu}(\tilde{S})\mathbf{M}^\dagger\right)$$

이 프레임워크에서 $h$가 CNN이든 MLP든 선형 모델이든, **동일한 불편 추정량 구조를 유지**하므로, 더 강력한 backbone과 결합할수록 일반화 성능이 향상될 잠재력이 있습니다.

---

### 3.3 고노이즈율 환경에서의 일반화 강건성

실험 결과, 60% 대칭 노이즈와 40% 페어플립 노이즈 환경에서도 MC-LDCE가 안정적 성능을 보입니다. 이는 센트로이드 추정 기반 접근이 **노이즈율이 높을수록 상대적 이점이 커지는** 특성을 가짐을 보여주며, 실세계 극단적 노이즈 환경에서의 일반화 가능성을 시사합니다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4.1 앞으로의 연구에 미치는 영향

**① 다중 클래스 LNL 연구의 새로운 기준선 제시**

MC-LDCE는 LDCE 패러다임을 다중 클래스로 확장한 **최초의 방법**으로서, 이후 연구들이 비교해야 할 중요한 기준선(baseline)이 됩니다. 특히 불편 위험 추정량의 이론적 보장은 향후 이론 중심 LNL 연구의 토대가 됩니다.

**② 손실 분해 패러다임의 확장 가능성**

MSE 손실을 분해하는 방식을 **크로스 엔트로피 손실, focal loss** 등에 적용하는 연구로 이어질 수 있습니다. 손실 분해의 가능 조건(label-independent/dependent 분리 가능성)에 대한 이론적 탐구가 촉진될 것입니다.

**③ 전이 행렬 추정과 LNL의 통합 연구**

MC-LDCE는 전이 행렬 추정에 VolMinNet (Li et al., 2021)을 외부 모듈로 활용하는데, 이를 **엔드-투-엔드(end-to-end)** 로 통합하는 방향의 연구를 자극합니다.

**④ 인스턴스 의존 노이즈로의 확장 필요성 인식**

클래스 조건부 노이즈 가정의 한계를 명확히 드러냄으로써, **인스턴스 의존 노이즈(instance-dependent noise)** 환경에서 LDCE를 적용하는 연구의 필요성을 제기합니다.

---

### 4.2 앞으로 연구 시 고려할 점

| 고려 사항 | 상세 내용 |
|-----------|-----------|
| **인스턴스 의존 노이즈 처리** | 현재 클래스 조건부 가정 → 샘플별 노이즈 전이를 모델링하는 방향으로 확장 필요 |
| **전이 행렬 추정 불확실성 전파** | $\hat{\mathbf{T}}$ 추정 오차가 $\mathbf{M}$과 최종 성능에 미치는 영향 분석 및 강건한 추정 방법 연구 |
| **크로스 엔트로피 손실로의 일반화** | 실용적으로 더 많이 쓰이는 cross-entropy에 LDCE 적용 가능성 탐구 |
| **대규모 클래스 수 확장성** | $\mathbf{M} \in \mathbb{R}^{c \times c}$ 유사 역행렬 계산의 클래스 수 확장성(ImageNet 1000 클래스 등) |
| **반지도 학습과의 결합** | 노이즈 레이블 + 미레이블 데이터를 함께 활용하는 방향 |
| **연속 레이블(Soft Label) 처리** | One-hot 가정을 벗어나 확률적 레이블 환경으로 확장 |
| **이론적 수렴 보장** | 유한 샘플 하에서의 일반화 오차 상한(generalization error bound) 도출 필요 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 비교 대상 방법 분류

```
LNL 방법 분류
├── 샘플 선택 (Sample Selection)
│   ├── Co-teaching+ (Yu et al., 2019)
│   ├── JoCoR (Wei et al., 2020)
│   └── SIGUA (Han et al., 2020)
├── 손실 보정 (Loss Correction)
│   ├── GCE (Zhang & Sabuncu, 2018)
│   ├── MC-LDCE (본 논문, 2022) ← 제안
│   └── VolMinNet (Li et al., 2021)
└── 레이블 수정 (Label Correction)
    └── SELFIE (Song et al., 2019)
```

### 5.2 최신 연구와의 상세 비교

| 방법 | 연도 | 접근법 | 다중 클래스 | 이론 보장 | 전이 행렬 필요 | 보조 clean 데이터 |
|------|------|--------|------------|----------|--------------|----------------|
| **Co-teaching+** (Yu et al.) | 2019 | 샘플 선택 | ✅ | ❌ | ❌ | ❌ |
| **JoCoR** (Wei et al.) | 2020 | 샘플 선택 + 정규화 | ✅ | ❌ | ❌ | ❌ |
| **SIGUA** (Han et al.) | 2020 | 그래디언트 제어 | ✅ | 부분 | ❌ | ❌ |
| **GCE** (Zhang & Sabuncu) | 2018 | 강건 손실 | ✅ | 부분 | ❌ | ❌ |
| **VolMinNet** (Li et al.) | 2021 | 전이 행렬 추정 | ✅ | ✅ | ✅(자체) | ❌ |
| **DivideMix** (Li et al.) | 2020 | 반지도 + GMM | ✅ | ❌ | ❌ | ❌ |
| **ELR** (Liu et al.) | 2020 | 정규화 기반 | ✅ | 부분 | ❌ | ❌ |
| **MC-LDCE** (본 논문) | 2022 | LDCE 손실 분해 | ✅ **(최초 LDCE 다중 클래스)** | ✅ | ✅(외부) | ❌ |

### 5.3 MC-LDCE의 차별성

**① DivideMix (Li et al., 2020, NeurIPS)와 비교:**

DivideMix는 GMM으로 clean/noisy 샘플을 구분하고 MixUp 기반 반지도 학습을 결합합니다. 매우 높은 성능을 보이나 학습 파이프라인이 복잡하고 이론적 보장이 약합니다. 반면 MC-LDCE는 단순하고 이론적으로 보장된 불편 추정량을 사용합니다.

**② ELR (Liu et al., 2020, NeurIPS)과 비교:**

Early Learning Regularization(ELR)은 모델 예측을 지수 이동 평균으로 정규화하여 노이즈 내성을 높입니다. 이는 경험적 설계로 이론적 근거가 MC-LDCE보다 약합니다.

**③ VolMinNet (Li et al., 2021, ICML)과의 관계:**

MC-LDCE는 전이 행렬 $\mathbf{T}$ 추정에 VolMinNet을 활용합니다. 즉, 두 방법은 **상보적(complementary)** 관계로, VolMinNet이 발전할수록 MC-LDCE의 성능도 향상됩니다.

### 5.4 MC-LDCE가 비교적 불리한 최신 방법

논문에서 직접 비교하지 않은 방법들 중 일부는 MC-LDCE보다 높은 성능을 보일 수 있습니다:

- **DivideMix** (Li et al., 2020): CIFAR-10 고노이즈에서 매우 높은 정확도
- **Sieve** (Ibrahim et al., 2023): 더 정교한 노이즈 모델링

이는 MC-LDCE의 비교 베이스라인이 다소 보수적으로 선택되었음을 의미하며, 향후 더 강력한 최신 방법들과의 비교가 필요합니다.

---

## 참고 자료

**본 논문 (주요 출처):**
- Ding, Y., Zhou, T., Zhang, C., Luo, Y., Tang, J., & Gong, C. (2022). *Multi-class Label Noise Learning via Loss Decomposition and Centroid Estimation*. arXiv:2203.10858v1

**논문 내 인용 문헌:**
- Li, X., Liu, T., Han, B., Niu, G., & Sugiyama, M. (2021). *Provably end-to-end label-noise learning without anchor points*. ICML 2021
- Wei, H., Feng, L., Chen, X., & An, B. (2020). *Combating noisy labels by agreement: A joint training method with co-regularization*. CVPR 2020
- Han, B., et al. (2020). *SIGUA: Forgetting may make learning with noisy labels more robust*. ICML 2020
- Zhang, Z., & Sabuncu, M. (2018). *Generalized cross entropy loss for training deep neural networks with noisy labels*. NeurIPS 2018
- Gong, C., Yang, J., You, J.J., & Sugiyama, M. (2020). *Centroid estimation with guaranteed efficiency: A general framework for weakly supervised learning*. IEEE TPAMI
- Patrini, G., et al. (2016). *Loss factorization, weakly supervised learning and label noise robustness*. ICML 2016
- Natarajan, N., et al. (2013). *Learning with noisy labels*. NeurIPS 2013
- Yu, X., et al. (2019). *How does disagreement help generalization against label corruption?* ICML 2019
- Han, B., et al. (2018). *Co-teaching: Robust training of deep neural networks with extremely noisy labels*. NeurIPS 2018

**2020년 이후 비교 분석 참고:**
- Li, J., Socher, R., & Hoi, S.C.H. (2020). *DivideMix: Learning with Noisy Labels as Semi-supervised Learning*. ICLR 2020
- Liu, S., Niles-Weed, J., Razavian, N., & Fernandez-Granda, C. (2020). *Early-learning regularization prevents memorization of noisy labels*. NeurIPS 2020

> **⚠️ 주의:** 2020년 이후 최신 연구(DivideMix, ELR 등)와의 수치적 직접 비교는 본 논문에 포함되지 않으므로, 해당 수치 비교는 각 논문의 개별 보고 수치에 기반한 정성적 비교임을 명시합니다. 논문에서 명시적으로 보고하지 않은 수치는 임의로 기재하지 않았습니다.
