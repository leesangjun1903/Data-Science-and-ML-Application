# Can Cross Entropy Loss Be Robust to Label Noise?

> **참고 자료:**
> - Lei Feng, Senlin Shu, Zhuoyi Lin, Fengmao Lv, Li Li, Bo An. "Can Cross Entropy Loss Be Robust to Label Noise?" *IJCAI-20*, pp. 2206–2212. (제공된 PDF 원문)
> - Zhang & Sabuncu, "Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels," *NeurIPS 2018*
> - Wang et al., "Symmetric Cross Entropy for Robust Learning with Noisy Labels," *ICCV 2019*
> - Menon et al., "Can Gradient Clipping Mitigate Label Noise?" *ICLR 2020*
> - Ghosh et al., "Robust Loss Functions under Label Noise for Deep Neural Networks," *AAAI 2017*

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장
Cross Entropy(CE) 손실 함수는 원래 형태로는 레이블 노이즈에 취약하지만, **Taylor Series 전개를 통해 CE를 재해석하면 노이즈에 강건한 일반화 프레임워크**를 만들 수 있다.

### 주요 기여 (3가지)

| 기여 | 내용 |
|------|------|
| **프레임워크 제안** | Taylor Cross Entropy (TCE, $\mathcal{L}_{t\text{-CE}}$) 손실 함수 제안 |
| **이론적 분석** | Uniform/Class-conditional 노이즈 하에서의 강건성 증명 |
| **관계 규명** | CCE, MAE, MSE 간의 내재적 관계를 Taylor Series로 통합 설명 |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2-1. 해결하고자 하는 문제

표준 CCE(Categorical Cross Entropy)로 학습된 DNN은 **레이블 노이즈에 쉽게 과적합**된다.

$$\mathcal{L}_{\text{CCE}}(f(\boldsymbol{x}), y) = -\log f_y(\boldsymbol{x})$$

CCE의 기울기는 다음과 같다:

$$\frac{\partial \mathcal{L}_{\text{CCE}}(f(\boldsymbol{x}), y)}{\partial \boldsymbol{\theta}} = -\frac{1}{f_y(\boldsymbol{x})} \nabla_{\boldsymbol{\theta}} f_y(\boldsymbol{x})$$

$f_y(\boldsymbol{x}) \to 0$ 일 때 가중치가 무한대로 발산하여 **잘못된 레이블(Hard Example)에 지나치게 집중**하는 문제가 발생한다.

반면, MAE는 모든 샘플을 동일하게 취급하여 노이즈에 강건하나, **복잡한 데이터셋에서 최적화 문제**로 성능이 낮다:

$$\mathcal{L}_{\text{MAE}}(f(\boldsymbol{x}), y) = \|\boldsymbol{e}_y - f(\boldsymbol{x})\|_1 = 2 - 2f_y(\boldsymbol{x})$$

$$\frac{\partial \mathcal{L}_{\text{MAE}}(f(\boldsymbol{x}), y)}{\partial \boldsymbol{\theta}} = -2\nabla_{\boldsymbol{\theta}} f_y(\boldsymbol{x})$$

**목표:** CCE와 MAE의 장점을 모두 취하면서 레이블 노이즈에 강건한 **통합 프레임워크** 설계

---

### 2-2. 제안하는 방법 (수식 포함)

#### Step 1: CCE의 Taylor Series 전개

$g(f_y(\boldsymbol{x})) = -\log f_y(\boldsymbol{x})$ 로 정의하고, $f_y(\boldsymbol{x}_0) = 1$ 에서 전개하면:

$$g^{(i)}(f_y(\boldsymbol{x}_0) = 1) = (-1)^i (i-1)! \quad \forall i \geq 1$$

따라서 CCE는 다음과 같이 표현된다:

$$\mathcal{L}_{\text{CCE}}(f(\boldsymbol{x}), y) = \sum_{i=1}^{\infty} \frac{(1 - f_y(\boldsymbol{x}))^i}{i}$$

#### Step 2: Taylor Cross Entropy (TCE) 정의

무한급수를 $t$차항까지 **유한 근사**하여 TCE를 정의한다:

$$\boxed{\mathcal{L}_{t\text{-CE}}(f(\boldsymbol{x}), y) = \sum_{i=1}^{t} \frac{(1 - f_y(\boldsymbol{x}))^i}{i}}$$

여기서 $t \in \mathbb{N}_+$는 Taylor Series의 차수(하이퍼파라미터)이며, CCE에 대한 근사 정도를 조절한다.

#### Step 3: Theorem 1 - TCE의 주요 성질

**Property 1)** $t = 1$ 일 때:

$$\mathcal{L}_{t\text{-CE}}(f(\boldsymbol{x}), y) = 1 - f_y(\boldsymbol{x}) = \frac{1}{2}\mathcal{L}_{\text{MAE}}$$

즉, $t=1$은 MAE와 동치이며 가장 강건한 형태이다.

**Property 2)** $t = 2$ 일 때:

$$\mathcal{L}_{t\text{-CE}}(f(\boldsymbol{x}), y) = (1 - f_y(\boldsymbol{x})) + \frac{(1 - f_y(\boldsymbol{x}))^2}{2}$$

두 번째 항에 대해:

$$\frac{(1 - f_y(\boldsymbol{x}))^2}{2} \leq \frac{1}{2}\|f(\boldsymbol{x}) - \boldsymbol{e}_y\|_2^2 = \frac{1}{2}\mathcal{L}_{\text{MSE}}$$

따라서 $t=2$는 **MAE와 MSE 하한의 평균 조합**이다.

**Property 3)** $t \to \infty$ 일 때:

$$\mathcal{L}_{t\text{-CE}} \to \mathcal{L}_{\text{CCE}}$$

**Property 4)** $\forall t \in \mathbb{N}_+$:

$$\mathcal{L}_{\text{MAE}} \leq 2\mathcal{L}_{t\text{-CE}}(f(\boldsymbol{x}), y)$$

TCE를 최소화하면 자연스럽게 MAE도 어느 정도 최소화된다.

---

### 2-3. 모델 구조

논문은 **새로운 신경망 구조를 제안하지 않고**, 기존 구조에 TCE 손실 함수를 적용한다.

| 데이터셋 | 모델 | 최적화기 | Epochs | Batch Size |
|----------|------|----------|--------|------------|
| MNIST / Fashion / Kuzushiji | LeNet-5 | Adam | 200 | 256 |
| CIFAR-10 / CIFAR-100 | ResNet-34 | Adam | 200 | 256 |

- 하이퍼파라미터 $t$는 $\{2, \ldots, 6\}$에서 선택
- 학습률은 $\{10^{-2}, 10^{-3}, 10^{-4}, 10^{-5}\}$에서 선택

---

### 2-4. 성능 향상

240개의 실험 케이스(6개 비교 방법 × 5개 데이터셋 × 8개 노이즈 설정) 기준:

- **83.33%**의 케이스에서 TCE가 비교 방법보다 통계적으로 우수 (paired t-test, p=0.05)
- **8.33%**의 케이스에서만 비교 방법보다 열등

주요 실험 결과 (CIFAR-100, Symmetric Noise 기준):

| 방법 | $\eta=0.2$ | $\eta=0.4$ | $\eta=0.6$ | $\eta=0.8$ |
|------|-----------|-----------|-----------|-----------|
| CCE | 47.00 | 34.34 | 19.37 | 7.34 |
| MAE | 33.33 | 26.56 | 12.26 | 2.01 |
| GCE | 58.99 | 50.37 | 39.41 | 15.26 |
| **TCE ($t=6$)** | **59.11** | **50.99** | **38.31** | **15.96** |

---

### 2-5. 한계

1. **하이퍼파라미터 $t$ 선택 문제**: 데이터셋과 노이즈 유형에 따라 최적 $t$가 다르며, 자동 탐색 방법이 없다.
2. **노이즈율이 매우 높을 때의 한계**: $\eta = 0.8$ 대칭 노이즈에서 일부 방법보다 성능이 제한적이다.
3. **Instance-dependent noise 미고려**: 이론 분석이 instance-independent noise 가정에 한정된다.
4. **하이퍼파라미터 없는 강건 손실 함수**: 저자들도 미래 과제로 인정한 한계이다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 이론적 강건성 보장

#### Lemma 1: TCE의 유계성

모든 $\boldsymbol{x}$와 유한한 양의 정수 $t$에 대해:

$$k - 1 \leq \sum_{y=1}^{k} \mathcal{L}_{t\text{-CE}}(f(\boldsymbol{x}), y) \leq (k-1)C_t$$

여기서 $C_t = \sum_{i=1}^{t} \frac{1}{i}$ (조화급수의 부분합). 이 유계성이 일반화 성능 향상의 핵심 근거이다.

#### Theorem 2: Uniform Noise 하에서의 일반화 성능

$\eta \leq 1 - \frac{1}{k}$ 조건에서, 노이즈 레이블로 학습된 모델 $\tilde{f}$와 클린 데이터로 학습된 최적 모델 $f^*$ 간의 위험 차이가 다음으로 상한된다:

$$0 \leq \mathcal{R}_{\mathcal{L}_{t\text{-CE}}}(\tilde{f}) - \mathcal{R}_{\mathcal{L}_{t\text{-CE}}}(f^*) \leq \frac{\eta(k-1)(C_t - 1)}{(1-\eta)k - 1}$$

**일반화 성능에 대한 함의:**
- $t$가 작을수록 $C_t$가 작아져 상한이 더 타이트해진다 → **강건성 향상**
- $t=1$일 때 MAE와 동일한 이론적 보장을 가진다
- 노이즈율 $\eta$가 증가할수록 상한이 커지지만, TCE는 이를 $t$ 조절로 완화 가능

#### Theorem 3: Class-conditional Noise 하에서의 일반화 성능

$\eta_{ij} < 1 - \eta_i$ 조건과 $\mathcal{R}\_{\mathcal{L}_{t\text{-CE}}}(f^*) = 0$ 가정 하에:

$$0 \leq \mathcal{R}^{\eta}_{\mathcal{L}_{t\text{-CE}}}(f^*) - \mathcal{R}^{\eta}_{\mathcal{L}_{t\text{-CE}}}(\tilde{f}) \leq A$$

여기서:
$$A = (k-1)(C_t - 1)\mathbb{E}_{p(\boldsymbol{x},y)}(1 - \eta_i) > 0$$

$C_t = \sum_{n=1}^{t} \frac{1}{n}$

**주요 관찰:**
- $t$가 작을수록 $C_t - 1$이 작아져 $A$가 줄어들고 → 더 나은 일반화
- Noisy 환경에서 학습된 $\tilde{f}$가 실제 최적 $f^*$와 유사한 성능을 보장

### 3-2. 일반화 성능 향상 메커니즘

```
t 증가 → CCE에 근접 → Hard Example에 높은 가중치 → 노이즈 레이블 과적합 위험 ↑
t 감소 → MAE에 근접 → 균등한 가중치 부여 → 노이즈에 강건 ↑ (단, 최적화 어려움 ↑)
```

TCE는 $t$ 조절을 통해 **피팅 정도와 강건성 사이의 트레이드오프를 유연하게 조절**하여 일반화 성능을 향상시킨다.

- ** $\mathcal{L}\_{\text{MAE}} \leq 2\mathcal{L}_{t\text{-CE}}$ ** 관계에 의해, TCE 최소화는 자연스럽게 MAE 최소화를 유도한다
- 이는 TCE가 **암묵적인 정규화 효과**를 가짐을 의미한다

---

## 4. 앞으로의 연구에 미치는 영향과 고려할 점

### 4-1. 연구에 미치는 영향

#### (1) 손실 함수 설계의 새로운 패러다임
Taylor Series를 통한 **손실 함수의 계층적 통합**이라는 아이디어는 이후 연구에서 다양한 손실 함수를 단일 프레임워크로 이해하는 방향을 열었다.

#### (2) 이론-실험 연계 연구 강화
Theorem 2, 3의 위험 상한 분석은 이후 **PAC-Bayes**, **Rademacher Complexity** 기반 노이즈 강건성 분석 연구에 영향을 준다.

#### (3) 하이브리드 손실 함수 연구 촉진
MAE와 CCE의 관계 규명은 이후 GCE, SCE, NCE 등의 하이브리드 손실 함수 연구의 이론적 기반을 제공한다.

---

### 4-2. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의:** 아래 2020년 이후 연구들은 논문 원문에 직접 포함된 내용이 아니며, 해당 분야의 공개된 연구 흐름을 기반으로 서술합니다. 개별 논문의 세부 수치에 대해 100% 정확성을 보장하기 어려우므로, 대략적인 연구 방향과 특징 비교를 중심으로 기술합니다.

| 연구 | 핵심 아이디어 | TCE와의 비교 |
|------|-------------|-------------|
| **CORES (Cheng et al., 2021)** | 샘플별 신뢰도 기반 손실 재가중 | TCE는 샘플 레벨이 아닌 손실 함수 레벨에서 강건성 확보 |
| **NCE+MAE (Ma et al., 2020)** | Normalized CE와 MAE 결합 | TCE와 유사하게 CCE 변형을 통해 강건성 확보, 단 정규화 방식이 다름 |
| **AGCE (Zhou et al., 2021)** | Active Passive Loss 프레임워크 | TCE와 마찬가지로 단일 하이퍼파라미터로 CCE-MAE 스펙트럼 조절 |
| **SOP (Liu & Guo, 2022)** | 과적합 방지를 위한 샘플별 가중치 학습 | TCE보다 복잡한 구조이나 더 적응적 |
| **DivideMix (Li et al., 2020)** | GMM 기반 노이즈/클린 샘플 분리 후 반지도학습 | TCE는 순수 손실 함수 접근, DivideMix는 알고리즘적 접근 |

**연구 흐름 요약:**

```
TCE (2020): 손실 함수 재설계 (Taylor Series 기반)
    ↓
NCE+MAE: 정규화를 통한 CCE 변형 → 상보적 손실 조합
    ↓
AGCE/APL: Active+Passive Loss 통합 프레임워크
    ↓
SOP/DivideMix: 알고리즘 수준의 노이즈 처리 + 반지도학습
```

---

### 4-3. 앞으로 연구 시 고려할 점

#### ① 자동 하이퍼파라미터 탐색
$t$ 값을 데이터셋과 노이즈 특성에 맞게 **자동으로 조정하는 메타러닝 또는 AutoML 기법** 연구가 필요하다.

$$t^* = \arg\min_{t \in \{1, \ldots, T\}} \mathcal{R}^{\text{val}}_{\mathcal{L}_{t\text{-CE}}}(f)$$

#### ② Instance-dependent Noise 대응
현재 이론은 instance-independent noise를 가정하므로, **샘플별 노이즈 의존성**을 다루는 확장이 필요하다.

#### ③ 대규모 언어 모델(LLM)에의 적용
LLM의 파인튜닝 과정에서 발생하는 **Human Feedback 노이즈** (RLHF의 레이블 오류) 문제에 TCE 프레임워크 적용 가능성 탐색.

#### ④ 장기 훈련에서의 동적 $t$ 조절
훈련 초기에는 $t$를 작게 (강건성 우선), 후기에는 $t$를 크게 (정확도 우선) 하는 **커리큘럼 기반 동적 $t$ 스케줄링** 연구.

$$t(\text{epoch}) = \min\left(t_{\max}, \left\lceil \frac{\text{epoch}}{E} \cdot t_{\max} \right\rceil\right)$$

#### ⑤ 다중 노이즈 유형 통합
Symmetric, Asymmetric, Instance-dependent, Feature-dependent 노이즈를 **하나의 통합 프레임워크**에서 처리하는 이론적 확장.

#### ⑥ Calibration과의 연계
노이즈 강건성과 **모델 신뢰도 보정(Calibration)**의 관계를 분석하여, TCE가 confidence calibration에 미치는 영향 연구.

---

## 결론 요약

$$\underbrace{\mathcal{L}_{t\text{-CE}}}_{\text{TCE}} = \sum_{i=1}^{t} \frac{(1-f_y(\boldsymbol{x}))^i}{i} \quad \begin{cases} t=1: \text{MAE (가장 강건)} \\ t \to \infty: \text{CCE (가장 정확)} \end{cases}$$

TCE는 **단순한 수식 변환**을 통해 CCE의 노이즈 취약성을 해소하고, MAE~CCE 스펙트럼을 단일 파라미터 $t$로 제어함으로써 **이론적 강건성과 실용적 성능을 동시에** 달성한 우아한 프레임워크이다. 향후 연구에서는 자동 $t$ 탐색, LLM 적용, 동적 스케줄링 등 다양한 확장이 기대된다.
