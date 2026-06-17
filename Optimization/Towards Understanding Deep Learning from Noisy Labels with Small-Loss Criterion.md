# Towards Understanding Deep Learning from Noisy Labels with Small-Loss Criterion

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

이 논문의 핵심 주장은 **small-loss criterion(소손실 기준)이 왜 작동하는지를 이론적으로 설명**하고, 이를 기반으로 개선된 방법론(RSL)을 제안하는 것입니다.

구체적으로:

1. **대각 우세 조건(Diagonally-Dominant Condition)** 이 Noisy Label 학습의 성공을 위한 필요충분조건임을 이론적으로 증명
2. 이 조건 하에서 clean 샘플이 noisy 샘플보다 항상 작은 손실을 가짐을 증명
3. Vanilla small-loss criterion의 한계를 지적하고 개선된 **RSL(Reformalization of Small-Loss criterion)** 제안

### 주요 기여

| 기여 항목 | 내용 |
|-----------|------|
| 이론적 기반 제공 | Small-loss criterion의 작동 원리를 처음으로 엄밀하게 증명 |
| Lemma 1 & 2 | 대각 우세 조건과 최적 분류기 학습의 동치 관계 규명 |
| Theorem 1 & 2 | Clean 샘플이 동일 noisy label 내에서 더 작은 손실을 가짐을 증명 |
| RSL 알고리즘 | 클래스별 mean loss 기반 샘플 선택 전략 제안 |
| RSL_WM | Semi-supervised learning(MixMatch)과의 결합으로 성능 향상 |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

**문제 1: 이론적 공백**
- Small-loss criterion 기반 방법들(Co-teaching, JoCoR 등)이 실용적으로 효과적임에도 불구하고, 왜 작동하는지에 대한 이론적 설명이 전무

**문제 2: Vanilla Small-loss Criterion의 한계**
- 서로 다른 클래스 간 손실 값이 비교 불가능(not comparable)
- 단일 epoch의 손실 값은 SGD로 인해 큰 변동(fluctuation)이 존재
- 클래스별 노이즈율 차이를 고려하지 않아 class distribution shift 발생 가능

### 2.2 이론적 배경 및 핵심 수식

**설정:** $c$-클래스 분류 문제에서 노이즈 전이 행렬(noise transition matrix) $T \in \mathbb{R}^{c \times c}$, 여기서

$$T_{ij} = p(\tilde{y} = j \mid y = i)$$

는 $i$번째 클래스 샘플이 $j$번째 클래스로 flip될 확률입니다.

**Softmax 출력:**

$$\hat{p}_i(\boldsymbol{x}) = \frac{\exp\left(\boldsymbol{w}_i^\top \phi(\boldsymbol{x};\boldsymbol{\theta})\right)}{\sum_{j=1}^{c} \exp\left(\boldsymbol{w}_j^\top \phi(\boldsymbol{x};\boldsymbol{\theta})\right)}$$

**학습 목표 (Cross-Entropy 기대 손실 최소화):**

$$\Theta^* = \arg\min_{\Theta} \mathbb{E}_{(\boldsymbol{x}, \tilde{y})}\left[\ell_{CE}(g(\boldsymbol{x};\Theta), \tilde{y})\right] $$

---

**Lemma 1 (0-1 손실 최소성):**

> $T$가 행-대각 우세 조건 $T_{ii} > \max_{j \neq i} T_{ij}, \forall i$를 만족하면,
> 목표 개념 $f^\*$는 노이즈 데이터에서 최소 기대 0-1 손실을 가짐:
> $$\mathbb{E}\_{(\boldsymbol{x},\tilde{y})}\left[\ell_{01}(f^\*(\boldsymbol{x}), \tilde{y})\right] \leq \mathbb{E}\_{(\boldsymbol{x},\tilde{y})}\left[\ell_{01}(f(\boldsymbol{x}), \tilde{y})\right], \quad \forall f \neq f^*$$

**증명의 핵심:**

$$P(f(\boldsymbol{x}) \neq \tilde{y}) = \int_{\boldsymbol{x} \in \mathcal{X}} \left[1 - T_{f^*(\boldsymbol{x})f(\boldsymbol{x})}\right] p(\boldsymbol{x}) \, d\boldsymbol{x}$$

대각 우세 조건에 의해 $T\_{f^\*(\boldsymbol{x})f^*(\boldsymbol{x})} > T\_{f^\*(\boldsymbol{x})j}$ 이므로 $f^\*$가 최소 손실을 가집니다.

---

**Lemma 2 (Cross-Entropy 최적 분류기):**

> $g^\*$를 식 (1)을 최소화하는 DNN이라 할 때, 유도 분류기 $f\_{g^\*}$가 $f_{g^*}(\boldsymbol{x}) = y, \forall \boldsymbol{x} \in \mathcal{X}$를 만족하는 것은 $T$가 행-대각 우세 조건 $T_{ii} > \max_{j \neq i} T_{ij}, \forall i$를 만족하는 것과 **동치**.

**증명의 핵심:** $g^*$의 softmax 출력이 다음을 만족함:

$$\hat{p}_k(\boldsymbol{x}) = p(\tilde{y} = k \mid \boldsymbol{x}) = T_{f^*(\boldsymbol{x})k}$$

따라서 $f\_{g^\*}(\boldsymbol{x}) = \arg\max_k \hat{p}\_k(\boldsymbol{x}) = f^\*(\boldsymbol{x})$이려면 $T_{ii} > T_{ij}, \forall j \neq i$가 필요충분조건.

---

**Theorem 1 (Small-Loss의 이론적 근거):**

> $g^\*$가 식 (1)을 최소화하는 DNN이고, $(x_1, \tilde{y})$와 $(x_2, \tilde{y})$가 같은 observed label $\tilde{y}$를 갖는 두 샘플로서 $f^\*(x_1) = \tilde{y}$ (clean)이고 $f^\*(x_2) \neq \tilde{y}$ (noisy)일 때, $T$가 대각 우세 조건 $T_{ii} > \max\{\max_{j \neq i} T_{ij}, \max_{j \neq i} T_{ji}\}$를 만족하면:
> $$\ell\_{CE}(g^\*(\boldsymbol{x}\_1), \tilde{y}) < \ell\_{CE}(g^\*(\boldsymbol{x}\_2), \tilde{y})$$

**증명:**

$$\ell_{CE}(g^*(\boldsymbol{x}_1), \tilde{y}) = -\log(T_{\tilde{y}\tilde{y}})$$

```math
\ell_{CE}(g^*(\boldsymbol{x}_2), \tilde{y}) = -\log(T_{f^*(\boldsymbol{x}_2)\tilde{y}})
```

대각 우세 조건 $T_{\tilde{y}\tilde{y}} > \max_{j \neq \tilde{y}} T_{j\tilde{y}}$에 의해 $-\log(T_{\tilde{y}\tilde{y}}) < -\log(T_{f^*(\boldsymbol{x}_2)\tilde{y}})$.

---

**Theorem 2 (실용적 모델에서의 Small-Loss 성립 조건):**

> $g$가 $g^\*$에 $\epsilon$ -근접, 즉 $\|g - g^*\|_\infty = \epsilon$이면, $T$가 대각 우세 조건을 만족하고

$$\epsilon < \frac{1}{2} \cdot (T_{\tilde{y}\tilde{y}} - T_{f^*(\boldsymbol{x}_2)\tilde{y}})$$

> 이면 $\ell_{CE}(g(\boldsymbol{x}\_1), \tilde{y}) < \ell_{CE}(g(\boldsymbol{x}\_2), \tilde{y})$.

**손실 격차의 하한:**

$$\ell_{CE}(g(\boldsymbol{x}_2), \tilde{y}) - \ell_{CE}(g(\boldsymbol{x}_1), \tilde{y}) \geq \log\!\left(\frac{T_{\tilde{y}\tilde{y}} - \epsilon}{T_{f^*(\boldsymbol{x}_2)\tilde{y}} + \epsilon}\right)$$

### 2.3 제안 방법: RSL (Reformalization of Small-Loss Criterion)

**세 가지 핵심 개선:**

1. **Mean Loss 사용** (단일 epoch 손실 대신)

$$\bar{\ell}(\boldsymbol{x}, \tilde{y}) = \frac{1}{E} \sum_{t=1}^{E} \ell_t(\boldsymbol{x}, \tilde{y})$$

2. **클래스별 개별 선택** (전체 손실 비교 대신)

각 클래스 $i$에 대해 $\tilde{D}_i = \{(\boldsymbol{x}, \tilde{y}) \in \tilde{D} \mid \tilde{y} = i\}$에서 별도로 소손실 샘플 선택.

3. **선택 비율 조정**

노이즈율 $\eta_i$를 고려한 선택 비율:

$$\text{prop}(i) = \max\{1 - (1+\beta)\eta_i,\ (1-\beta)(1-\eta_i)\}$$

클래스 분포 보정을 위한 선택 수:

$$\text{num}(i) = \min\{\gamma \cdot p_i \times m,\ \text{prop}(i) \times n_i\} $$

여기서 

```math
m = \min_{1 \leq i \leq c}\left\{\frac{\text{prop}(i) \cdot n_i}{p_i}\right\}
```

**RSL_WM: Semi-supervised 확장**

선택된 샘플에 대한 가중치 부여:

```math
w(\boldsymbol{x}, \tilde{y}) = \exp\!\left(-\kappa \cdot \frac{\bar{\ell}(\boldsymbol{x}, \tilde{y}) - \ell_*(i)}{\ell^*(i) - \ell_*(i)}\right)
```

비선택 샘플을 unlabeled 데이터로 활용한 Weighted MixMatch:

$$D_{\text{sel WM}} = \text{Weighted MixMatch}(D_{\text{sel}}, D_u)$$

### 2.4 모델 구조

| 구성 요소 | 상세 내용 |
|-----------|----------|
| Backbone (CIFAR) | Wide ResNet-28 |
| Backbone (CIFAR 검증용) | PreAct ResNet-32 |
| Backbone (WebVision) | Inception-ResNet-v2 |
| Optimizer | SGD (momentum=0.9) |
| Batch size | 128 |
| Semi-supervised | Weighted MixMatch |
| 기본 하이퍼파라미터 | $\beta=0.2$, $\gamma=(\gamma_0+\gamma_1)/2$, $\kappa=-\log(0.7)$ |

**2단계 학습 파이프라인:**

```
Stage 1: 전체 noisy 데이터로 warm-up 학습
         → 각 epoch의 손실 기록 → mean loss 계산
         → 클래스별 소손실 샘플 선택 (Algorithm 1)

Stage 2: RSL: 선택된 clean 데이터만으로 재학습
         RSL_WM: Weighted MixMatch로 unlabeled 데이터까지 활용
```

### 2.5 성능 향상

**CIFAR-10 (대표 결과):**

| Method | Uniform r=0.5 | Pairwise r=0.4 | Structured r=0.4 |
|--------|--------------|----------------|-----------------|
| Co-teaching | 85.14 | 78.18 | 80.59 |
| JoCoR | 87.27 | 83.48 | 83.59 |
| RSL | 88.21 | 86.73 | 87.91 |
| **RSL_WM** | **93.38** | **89.27** | **91.17** |

**WebVision (top-1):**

| Method | WebVision Val. |
|--------|---------------|
| Co-teaching | 63.58 |
| JoCoR | 65.28 |
| **RSL_WM** | **66.56** |

### 2.6 한계점

1. **노이즈율 사전 지식 요구**: $\beta$, $\gamma$ 설정에 노이즈율 추정값 필요 (Co-teaching, JoCoR 동일 조건이나 현실적 제약)
2. **Class-conditional noise 가정**: 인스턴스 의존적(instance-dependent) 노이즈로의 확장 미완성 (논문 자체가 향후 과제로 언급)
3. **대각 우세 조건 의존성**: 조건 불만족 시($r \geq 0.5$ pairwise 등) small-loss criterion 자체가 실패
4. **Pairwise noise 고노이즈율 약점**: CIFAR-100 pairwise $r=0.4$에서 JoCoR 대비 경쟁력 저하 (선택 데이터 수가 적기 때문)
5. **계산 비용**: 2단계 학습 + MixMatch 결합으로 계산량 증가

---

## 3. 모델 일반화 성능 향상 가능성

### 3.1 이론적 일반화 보장

**Theorem 2의 일반화 함의:**

```math
\epsilon < \frac{1}{2} \cdot \min_{1 \leq i \leq c}\left\{T_{ii} - \max_{j \neq i} T_{ji}\right\}
```

이 조건은 모델 $g$가 이상적 모델 $g^*$에 충분히 가까울 때 **모든 클래스에 대해 clean 샘플이 더 작은 손실**을 가짐을 보장합니다. 이는 **더 큰 학습 데이터**로 훈련할수록 $\epsilon$이 작아지고, 일반화 성능이 향상됨을 의미합니다.

실험적으로도 이를 검증: 학습 데이터 크기를 $\frac{1}{4}, \frac{1}{2}, \frac{3}{4}, 1$로 변화시킬 때 선택된 데이터의 precision이 각각 $0.908, 0.937, 0.949, 0.955$로 단조 증가.

### 3.2 Mean Loss를 통한 안정적 선택

단일 epoch 손실은 SGD 기반 최적화로 인해 큰 변동이 있어 신뢰할 수 없습니다. **Mean loss**를 사용함으로써:

$$\bar{\ell}(\boldsymbol{x}, \tilde{y}) = \frac{1}{E} \sum_{t=1}^{E} \ell_t(\boldsymbol{x}, \tilde{y})$$

- 노이즈에 의한 손실 변동 평균화
- Clean 샘플의 일관되게 낮은 손실 패턴 포착
- 결과적으로 더 높은 precision의 clean 데이터 선택 → 일반화 성능 향상

### 3.3 Semi-supervised Learning으로 일반화 개선

비선택 데이터를 단순히 버리지 않고 **unlabeled 데이터**로 활용:

$$\mathcal{L} = \mathcal{L}_\mathcal{L} + \lambda_U \mathcal{L}_\mathcal{U}$$

여기서:
- $\mathcal{L}\_\mathcal{L} = \frac{1}{|\mathcal{L}'|} \sum_{(\boldsymbol{x},p) \in \mathcal{L}'} H(p, p_{\text{model}}(\hat{y}|\boldsymbol{x}))$ (labeled 손실)
- $\mathcal{L}\_\mathcal{U} = \frac{1}{c|\mathcal{U}'|} \sum_{(\boldsymbol{u},q) \in \mathcal{U}'} \|q - p_{\text{model}}(\hat{y}|\boldsymbol{u})\|_2^2$ (unlabeled 손실)

이 프레임워크는 **유효 학습 데이터 증가** + **반지도 학습의 정규화 효과**를 동시에 제공하여 일반화 성능을 크게 향상시킵니다 (RSL 대비 RSL_WM의 지속적 성능 우위가 이를 입증).

### 3.4 Class Distribution 보정을 통한 일반화

클래스 불균형은 모델의 일반화를 해치는 주요 요인입니다. $\gamma$ 파라미터를 통해 선택 데이터의 클래스 분포를 원본 분포 $[p_1, \ldots, p_c]$에 맞게 조정함으로써, **편향되지 않은 경계면 학습**이 가능해집니다. 특히 structured noise 환경에서 이 효과가 두드러집니다.

### 3.5 가중치 부여를 통한 Robust 일반화

선택된 샘플 중에도 남아있는 noisy 샘플의 영향을 완화:

```math
w(\boldsymbol{x}, \tilde{y}) = \exp\!\left(-\kappa \cdot \frac{\bar{\ell}(\boldsymbol{x}, \tilde{y}) - \ell_*(i)}{\ell^*(i) - \ell_*(i)}\right)
```

높은 손실 = 높은 noisy 가능성에 낮은 가중치를 부여하여, **gradient 업데이트의 품질**을 향상시키고 일반화 성능에 기여합니다.

---

## 4. 향후 연구에 미치는 영향 및 고려할 점

### 4.1 향후 연구에 미치는 영향

**① 이론적 분석 프레임워크 확립**

이 논문은 noisy label 학습에서 최초로 체계적인 이론적 분석 틀을 제공했습니다. 이후 연구들이 자신들의 방법론을 정당화할 때 이 프레임워크를 참조하거나 확장하는 방향으로 발전할 것입니다.

**② Instance-Dependent Noise 연구로의 확장**

논문 자체가 class-conditional noise의 한계를 인정하며 instance-dependent noise(Xia et al., 2020)로의 확장을 향후 과제로 명시했습니다. 이는 현실적 노이즈 모델링을 위한 중요한 연구 방향입니다.

**③ Semi-supervised 학습과의 결합 패러다임**

비선택 데이터를 unlabeled로 활용하는 RSL_WM의 접근은 **noisy label learning + semi-supervised learning** 융합 연구의 방향을 제시했습니다.

**④ 손실 기반 샘플 선택의 정교화**

Mean loss, 클래스별 선택, 가중치 부여 등의 아이디어는 이후 더 정교한 샘플 선택 전략 연구(curriculum learning, self-paced learning 등)에 영향을 줍니다.

### 4.2 향후 연구 시 고려할 점

**① Instance-Dependent Noise로의 이론 확장**

현재 이론은 $p(\tilde{y}|y, \boldsymbol{x}) = p(\tilde{y}|y)$의 class-conditional 가정에 의존합니다. 실제 데이터에서는 인스턴스에 따라 노이즈 패턴이 달라질 수 있으므로, 더 현실적인 노이즈 모델에 대한 이론 확장이 필요합니다.

**② 노이즈율 추정의 자동화**

현재 방법은 노이즈율 $\eta_i$를 알고 있다고 가정합니다. 실제 환경에서는 이를 알 수 없으므로, **자동 노이즈율 추정**과의 결합이 중요한 연구 과제입니다.

**③ 대형 언어 모델(LLM) 및 Foundation Model 환경 적용**

현재 AI 연구의 주류인 대규모 사전 학습 모델 환경에서 noisy label이 미치는 영향과 small-loss criterion의 유효성 검증이 필요합니다. 모델 규모가 커질수록 memorization 특성이 달라질 수 있습니다.

**④ Open-Set Noise 처리**

본 논문은 closed-set noise(클래스 내 오분류)를 다루지만, 실제 웹 크롤링 데이터에는 학습 클래스에 속하지 않는 out-of-distribution 샘플(open-set noise)도 존재합니다. 이에 대한 이론적 확장이 필요합니다.

**⑤ 클래스 불균형과 노이즈의 상호작용**

$\gamma$ 파라미터로 어느 정도 처리하지만, 극단적 클래스 불균형 + 고노이즈율 조합에 대한 더 정교한 이론 및 방법론 개발이 필요합니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

> **⚠️ 주의:** 아래 비교 분석은 제가 학습한 데이터(2021년 6월 이전까지의 논문) 및 본 논문에서 직접 인용된 연구들을 중심으로 서술합니다. 2020년 이후의 일부 논문에 대해서는 정보가 불완전할 수 있어, 확인된 사실 위주로만 기술합니다.

### 5.1 본 논문에서 직접 비교한 2020년 연구

| 논문 | 방법 | 핵심 아이디어 | 본 논문 대비 |
|------|------|--------------|-------------|
| JoCoR (Wei et al., CVPR 2020) | Co-regularization + Co-teaching | 두 네트워크 예측의 agreement 활용 | RSL_WM이 대부분 설정에서 우위 |
| Early-Learning Regularization (Liu et al., NeurIPS 2020) | 조기 학습 단계 활용 | memorization 현상 이론적 분석 | 이론적 접근 방향 유사, 적용 방법 상이 |
| Hu et al. (ICLR 2020) | 정규화 방법 | gradient descent + 정규화의 일반화 보장 | 이론 보장 측면에서 상보적 |

### 5.2 2020년 이후 주요 연구 방향 (학습 데이터 기반, 부분적 확인)

**① DivideMix (Li et al., ICLR 2020)**
- GMM(Gaussian Mixture Model)으로 clean/noisy 분리 + MixMatch
- RSL_WM과 유사한 semi-supervised 결합 접근이나 GMM 기반 확률적 분리 활용
- 본 논문의 이론적 기반(대각 우세 조건)은 DivideMix에 적용 가능한 설명 틀 제공

**② CORES (Cheng et al., ICML 2021)**
- 확률적 전이를 이용한 인스턴스별 신뢰도 추정

**③ Instance-Dependent Noise 연구 (Xia et al., NeurIPS 2020)**
- 본 논문이 향후 과제로 지목한 방향의 연구
- Part-dependent label noise 모델링

### 5.3 방법론적 위치 비교

```
이론적 깊이: 본 논문 >> DivideMix ≈ Co-teaching
실용적 성능: RSL_WM ≈ DivideMix > Co-teaching > JoCoR
Semi-supervised 활용: RSL_WM, DivideMix > 나머지
노이즈율 의존성: 본 논문 = Co-teaching (필요) vs. DivideMix (추정)
```

---

## 참고 자료

**주 논문:**
- Gui, X.-J., Wang, W., & Tian, Z.-H. (2021). *Towards Understanding Deep Learning from Noisy Labels with Small-Loss Criterion*. arXiv:2106.09291v1. [본 논문 PDF 직접 참조]

**본 논문에서 인용된 주요 참고문헌:**
- Han et al. (2018). *Co-teaching: Robust training of deep neural networks with extremely noisy labels*. NeurIPS 2018.
- Yu et al. (2019). *How does disagreement help generalization against label corruption?* ICML 2019.
- Wei et al. (2020). *Combating noisy labels by agreement: A joint training method with co-regularization*. CVPR 2020.
- Patrini et al. (2017). *Making deep neural networks robust to label noise: A loss correction approach*. CVPR 2017.
- Berthelot et al. (2019). *MixMatch: A holistic approach to semi-supervised learning*. NeurIPS 2019.
- Arpit et al. (2017). *A closer look at memorization in deep networks*. ICML 2017.
- Hu et al. (2020). *Simple and effective regularization methods for training on noisily labeled data with generalization guarantee*. ICLR 2020.
- Liu et al. (2020). *Early-learning regularization prevents memorization of noisy labels*. NeurIPS 2020.
- Xia et al. (2020). *Part-dependent label noise: Towards instance-dependent label noise*. NeurIPS 2020.
- Chen et al. (2019). *Understanding and utilizing deep neural networks trained with noisy labels*. ICML 2019.
