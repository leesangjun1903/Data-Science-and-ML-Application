# Supervised Learning from Multiple Experts: Whom to trust when everyone lies a bit

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

이 논문은 **골드 스탠다드(절대적 정답)가 없는 상황에서 여러 전문가/어노테이터가 제공하는 노이즈 있는 레이블로부터 분류기를 학습하는 확률론적 방법**을 제안합니다. 단순 다수결 투표(Majority Voting)는 모든 전문가를 동등하게 취급한다는 근본적 한계를 가지며, 이를 극복하기 위해 각 전문가의 신뢰도를 자동으로 추정하는 EM 알고리즘 기반 프레임워크를 제시합니다.

### 주요 기여

| 기여 항목 | 내용 |
|---|---|
| **Two-coin 어노테이터 모델** | 각 전문가를 민감도(sensitivity)와 특이도(specificity)로 특성화 |
| **Joint Estimation** | 분류기, 전문가 정확도, 숨겨진 실제 레이블을 동시에 추정 |
| **Bayesian MAP 확장** | Beta prior를 통한 사전 지식 반영 |
| **일반화된 프레임워크** | 이진/범주형/순서형/연속형 레이블 및 누락 레이블 처리 가능 |
| **실증적 우월성 입증** | 디지털 유방촬영, Breast MRI, 텍스트 수반(Textual Entailment) 데이터셋에서 다수결 대비 성능 향상 |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

훈련 데이터 $\mathcal{D} = \{(\mathbf{x}\_i, y_i)\}_{i=1}^{N}$에서 실제 레이블 $y_i$를 취득하기 어렵거나 비용이 과다한 경우, $R$명의 전문가로부터 노이즈 있는 레이블 $y_i^1, \ldots, y_i^R$을 수집하게 됩니다. 이때 다음 세 가지를 동시에 해결해야 합니다:

1. **분류기 학습**: 일반화 가능한 분류 함수 $f: \mathbb{R}^d \rightarrow \mathcal{Y}$ 학습
2. **전문가 신뢰도 평가**: 골드 스탠다드 없이 각 전문가의 정확도 추정
3. **숨겨진 실제 레이블 추정**: True label $y_i$의 사후 확률 추정

**기존 다수결 투표의 문제점:**

$$\hat{y}_i = \begin{cases} 1 & \text{if } \frac{1}{R}\sum_{j=1}^{R} y_i^j \geq 0.5 \\ 0 & \text{otherwise} \end{cases} \tag{1}$$

위 방법은 모든 전문가를 동등하게 취급하므로, 소수의 진짜 전문가와 다수의 초보자가 있을 때 초보자 의견에 편향될 수 있습니다.

---

### 2.2 제안하는 방법

#### (A) Two-coin 어노테이터 모델

$j$번째 전문가의 성능을 두 가지 파라미터로 모델링합니다:

$$\alpha^j := \Pr[y^j = 1 \mid y = 1] \quad \text{(민감도, sensitivity)} \tag{3}$$

$$\beta^j := \Pr[y^j = 0 \mid y = 0] \quad \text{(특이도, specificity)} \tag{4}$$

단, $\alpha^j$와 $\beta^j$는 인스턴스 $\mathbf{x}$에 독립적이라고 가정합니다.

#### (B) 분류 모델 (로지스틱 회귀)

양성 클래스의 사후 확률:

$$\Pr[y = 1 \mid \mathbf{x}, \mathbf{w}] = \sigma(\mathbf{w}^\top \mathbf{x}) \tag{5}$$

여기서 로지스틱 시그모이드 함수 $\sigma(z) = \frac{1}{1 + e^{-z}}$

#### (C) 최대우도추정 (MLE)

파라미터 $\theta = \{\mathbf{w}, \boldsymbol{\alpha}, \boldsymbol{\beta}\}$에 대한 우도:

$$\Pr[\mathcal{D} \mid \theta] = \prod_{i=1}^{N} \Pr[y_i^1, \ldots, y_i^R \mid \mathbf{x}_i, \theta] \tag{6}$$

실제 레이블 $y_i$에 대해 조건부로 분리하면:

$$\Pr[\mathcal{D} \mid \theta] = \prod_{i=1}^{N} \left[ a_i p_i + b_i (1 - p_i) \right] \tag{8}$$

여기서:

$$p_i = \sigma(\mathbf{w}^\top \mathbf{x}_i), \quad a_i = \prod_{j=1}^{R} [\alpha^j]^{y_i^j} [1 - \alpha^j]^{1 - y_i^j}, \quad b_i = \prod_{j=1}^{R} [\beta^j]^{1 - y_i^j} [1 - \beta^j]^{y_i^j}$$

$$\hat{\theta}_{\text{ML}} = \{\hat{\boldsymbol{\alpha}}, \hat{\boldsymbol{\beta}}, \hat{\mathbf{w}}\} = \arg\max_{\theta} \{\ln \Pr[\mathcal{D} \mid \theta]\} \tag{9}$$

---

#### (D) EM 알고리즘

숨겨진 실제 레이블 $y_i$를 결측 데이터로 취급하여 EM 알고리즘을 적용합니다.

**완전 데이터 로그 우도:**

$$\ln \Pr[\mathcal{D}, \mathbf{y} \mid \theta] = \sum_{i=1}^{N} y_i \ln p_i a_i + (1 - y_i) \ln(1 - p_i) b_i \tag{10}$$

**E-step**: 현재 파라미터 추정치를 이용해 숨겨진 레이블의 사후 확률 계산:

$$\mu_i = \Pr[y_i = 1 \mid y_i^1, \ldots, y_i^R, \mathbf{x}_i, \theta] = \frac{a_i p_i}{a_i p_i + b_i (1 - p_i)} \tag{12}$$

**M-step**: $\mu_i$를 이용한 파라미터 업데이트:

$$\alpha^j = \frac{\sum_{i=1}^{N} \mu_i y_i^j}{\sum_{i=1}^{N} \mu_i}, \qquad \beta^j = \frac{\sum_{i=1}^{N} (1 - \mu_i)(1 - y_i^j)}{\sum_{i=1}^{N} (1 - \mu_i)}$$

**분류기 파라미터 $\mathbf{w}$ 업데이트** (Newton-Raphson):

$$\mathbf{w}^{t+1} = \mathbf{w}^t - \eta \mathbf{H}^{-1} \mathbf{g}$$

$$\mathbf{g}(\mathbf{w}) = \sum_{i=1}^{N} \left[\mu_i - \sigma(\mathbf{w}^\top \mathbf{x}_i)\right] \mathbf{x}_i$$

$$\mathbf{H}(\mathbf{w}) = -\sum_{i=1}^{N} \sigma(\mathbf{w}^\top \mathbf{x}_i)\left[1 - \sigma(\mathbf{w}^\top \mathbf{x}_i)\right] \mathbf{x}_i \mathbf{x}_i^\top$$

초기값: $\mu_i = \frac{1}{R}\sum_{j=1}^{R} y_i^j$ (다수결)

---

#### (E) Bayesian MAP 접근법

전문가의 민감도/특이도에 Beta prior 부여:

$$\text{Beta}(\delta \mid a, b) = \frac{\delta^{a-1}(1-\delta)^{b-1}}{B(a, b)} \tag{13}$$

MAP 추정에서의 파라미터 업데이트:

$$\alpha^j = \frac{a_1^j - 1 + \sum_{i=1}^{N} \mu_i y_i^j}{a_1^j + a_2^j - 2 + \sum_{i=1}^{N} \mu_i}$$

$$\beta^j = \frac{b_1^j - 1 + \sum_{i=1}^{N} (1-\mu_i)(1-y_i^j)}{b_1^j + b_2^j - 2 + \sum_{i=1}^{N} (1-\mu_i)} \tag{14}$$

가중치 $\mathbf{w}$에는 Gaussian prior $\Pr[\mathbf{w}] = \mathcal{N}(\mathbf{w} \mid \mathbf{0}, \Gamma^{-1})$를 적용하며, gradient와 Hessian에 정규화 항이 추가됩니다:

$$\mathbf{g}(\mathbf{w}) = \sum_{i=1}^{N} \left[\mu_i - \sigma(\mathbf{w}^\top \mathbf{x}_i)\right] \mathbf{x}_i - \Gamma\mathbf{w}$$

$$\mathbf{H}(\mathbf{w}) = -\sum_{i=1}^{N} \sigma(\mathbf{w}^\top \mathbf{x}_i)\left[1-\sigma(\mathbf{w}^\top \mathbf{x}_i)\right] \mathbf{x}_i \mathbf{x}_i^\top - \Gamma$$

---

#### (F) 프레임워크의 핵심 통찰 (Logit 해석)

$\mu_i$의 로짓(logit) 형태로 표현하면:

$$\text{logit}(\mu_i) = \mathbf{w}^\top \mathbf{x}_i + b + \sum_{j=1}^{R} y_i^j \left[\text{logit}(\alpha^j) + \text{logit}(\beta^j)\right]$$

이는 추정된 실제 레이블이 **모든 전문가 레이블의 가중 선형 결합**임을 보여주며, 각 전문가의 가중치는 $\text{logit}(\alpha^j) + \text{logit}(\beta^j)$로 결정됩니다.

---

### 2.3 모델 구조

```
입력: {x_i, y_i^1, ..., y_i^R}_{i=1}^{N}
        ↓
초기화: μ_i = 다수결 투표
        ↓
┌─────────────────────────────────┐
│  E-step: μ_i 업데이트           │
│  (숨겨진 레이블의 사후 확률)     │
│         ↓                       │
│  M-step: α^j, β^j, w 업데이트  │
│  (전문가 성능 + 분류기 파라미터) │
└─────────────────────────────────┘
        ↓ (수렴 시까지 반복)
출력:
 - 분류기 w (일반화 가능한 분류 함수)
 - 각 전문가의 α^j, β^j (신뢰도)
 - μ_i (실제 레이블의 사후 확률)
```

---

### 2.4 성능 향상

| 데이터셋 | 제안 방법 AUC | 다수결 AUC | 향상 |
|---|---|---|---|
| 디지털 유방촬영 (분류기) | 0.913 | 0.882 | **+3.5%** |
| 디지털 유방촬영 (추정 GT) | 0.991 | 0.962 | **+3%** |
| Breast MRI (LOO CV) | 0.879 | 0.828 | **+6%** |
| Breast MRI (추정 GT) | 0.944 | 0.937 | **+0.7%** |
| Joint vs Decoupled (분류기) | 0.905 | 0.884(Decoupled) | **+2.4%** |

---

### 2.5 한계

1. **인스턴스 독립 가정**: $\alpha^j$와 $\beta^j$가 특징 벡터 $\mathbf{x}$에 무관하다고 가정하지만, 실제로 전문가는 특정 유형의 데이터에 강점이 있을 수 있음.
2. **전문가 독립 가정**: 전문가들이 독립적으로 레이블링한다고 가정하지만, 실제로는 협의나 정보 공유가 일어날 수 있음.
3. **이진 분류 중심 설계**: 비록 확장이 제시되지만 주요 실험은 이진 분류에 집중됨.
4. **EM의 지역 최적해 문제**: EM 알고리즘 특성상 초기값에 따라 다른 해에 수렴할 가능성.
5. **전문가 수 확장성**: 전문가 수가 매우 많을 경우(예: Mechanical Turk) 계산 복잡도 증가.
6. **레이블 비율 불균형**: 클래스 불균형 문제에 대한 명시적 처리 없음.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화 성능을 높이는 핵심 메커니즘

#### (A) 더 정확한 훈련 레이블 확보

다수결 투표가 노이즈 초보자에 편향되는 반면, 제안 방법은 전문가별 가중치를 부여하여 더 신뢰할 수 있는 훈련 신호를 생성합니다. 이는 **훈련 데이터의 레이블 품질 향상** → **일반화 성능 향상**으로 이어집니다.

구체적으로 $\mu_i$의 logit 표현에서:

$$\text{logit}(\mu_i) = \mathbf{w}^\top \mathbf{x}_i + b + \sum_{j=1}^{R} y_i^j \left[\text{logit}(\alpha^j) + \text{logit}(\beta^j)\right]$$

- 높은 민감도·특이도 전문가에게 더 큰 가중치 $(\text{logit}(\alpha^j) + \text{logit}(\beta^j))$ 부여
- 낮은 신뢰도 전문가의 영향 자동 감소
- 결과적으로 더 **정제된 소프트 레이블** $\mu_i$를 이용한 분류기 학습

#### (B) Bayesian 정규화 효과

MAP 추정에서 Gaussian prior $\Pr[\mathbf{w}] = \mathcal{N}(\mathbf{w} \mid \mathbf{0}, \Gamma^{-1})$는 L2 정규화와 동일한 효과를 제공하여 과적합을 방지합니다. 이는 특히 **소규모 데이터셋에서 중요**하며, 일반화 성능에 직접 기여합니다.

Beta prior의 경우:
- 특정 전문가에 대한 사전 지식 반영 가능
- 전문가 파라미터 추정의 분산 감소
- 신뢰 구간 추정을 통한 불확실성 정량화

#### (C) Joint Estimation의 일반화 이점

분류기와 실제 레이블을 **동시에** 추정하는 것이 순차 추정(Decoupled)보다 우월합니다:

- Decoupled 방법: 레이블 추정 오차가 분류기 학습에 그대로 전파
- Joint 방법: 분류기의 특징 정보( $p_i = \sigma(\mathbf{w}^\top \mathbf{x}_i)$ )가 레이블 추정에도 활용됨

이를 $\mu_i$ 업데이트 수식에서 확인:

$$\mu_i = \frac{a_i p_i}{a_i p_i + b_i(1-p_i)} \tag{15}$$

여기서 $p_i$는 분류기의 사전 확률로, 인스턴스의 특징 정보를 레이블 정제에 활용합니다. 이것이 일반화 능력이 더 높은 분류기 학습을 가능하게 합니다.

#### (D) 누락 레이블 처리

실제 환경에서 모든 전문가가 모든 인스턴스에 레이블을 제공하지 않는 경우가 많습니다. 이를 자연스럽게 처리함으로써:

- 더 많은 데이터 활용 가능
- 편향 없는 전문가 성능 추정
- 결과적으로 더 강건한 분류기 학습

#### (E) 소프트 레이블의 정규화 효과

$\mu_i$는 하드 레이블이 아닌 소프트 확률값으로, 이를 이용한 학습은 암묵적으로 **레이블 스무딩(label smoothing)**과 유사한 효과를 제공하여 과적합을 방지하고 일반화를 향상시킵니다.

### 3.2 일반화 성능의 한계와 개선 방향

| 한계 | 잠재적 개선 방향 |
|---|---|
| 인스턴스 독립적 전문가 가정 | 인스턴스별 적응형 전문가 모델 |
| 소규모 실험 | 대규모 벤치마크 검증 필요 |
| 단순 로지스틱 회귀 | 심층 신경망과의 결합 |

---

## 4. 향후 연구에 미치는 영향과 고려할 점

### 4.1 연구에 미치는 영향

#### (A) Crowdsourcing 학습의 기반 확립

Amazon Mechanical Turk 등 크라우드소싱 플랫폼에서 수집된 레이블의 품질 문제를 체계적으로 다루는 프레임워크를 제시하여, **인간-AI 협력 학습** 연구의 기초를 마련했습니다.

#### (B) 의료 AI에 대한 파급효과

골드 스탠다드 취득이 어렵고 비싼 의료 영상 분야에서, 여러 방사선과 의사의 레이블을 효과적으로 활용하는 방법론을 제시함으로써 실용적 임상 AI 개발의 방향을 제시했습니다.

#### (C) 레이블 노이즈 학습(Learning with Noisy Labels) 분야 발전

이 논문은 후속 연구들이 다음 방향으로 발전하는 데 기여했습니다:
- 딥러닝 기반 어노테이터 모델
- 인스턴스 의존적 노이즈 모델
- 능동 학습(Active Learning)과의 결합

### 4.2 향후 연구 시 고려할 점

#### (A) 인스턴스 의존적 전문가 모델

논문이 명시적으로 언급한 한계로, 전문가의 성능이 인스턴스에 따라 달라질 수 있습니다. 예를 들어, 특정 방사선과 의사는 특정 종류의 병변에 더 뛰어날 수 있습니다. 이를 모델링하는 방향이 필요합니다.

#### (B) 전문가 독립 가정 완화

전문가들 사이의 **상관관계**를 모델링하는 것이 현실적으로 중요합니다. 예를 들어, 같은 병원 출신 의사들은 유사한 편향을 가질 수 있습니다.

#### (C) 딥러닝과의 통합

기본 분류기로 로지스틱 회귀를 사용했지만, 현대 딥러닝 모델(CNN, Transformer 등)과의 통합이 중요합니다. 미분 가능한 EM 또는 변분 추론(Variational Inference) 방법이 필요합니다.

#### (D) 능동 학습과의 결합

어떤 인스턴스에 어떤 전문가에게 레이블링을 요청할지 결정하는 **능동적 쿼리 전략**이 비용 효율성에 중요합니다.

#### (E) 공정성(Fairness) 고려

전문가의 편향이 특정 인구 집단에 불균형적으로 영향을 미칠 수 있음을 고려해야 합니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 주요 후속 연구

#### (A) CrowdLayer (Rodrigues & Pereira, 2018 → 2020년대 활발히 인용)

- **방법**: 딥러닝과 어노테이터 모델을 end-to-end로 결합
- **핵심**: 어노테이터별 가중치 행렬을 신경망 레이어로 구현
- **Raykar 대비 개선**: 비선형 분류기 직접 통합, GPU 가속 가능

#### (B) Learning from Crowds with Deep Neural Networks (Chu et al., 2021)

- **방법**: Raykar의 Two-coin 모델을 딥러닝에 확장
- **핵심**: 인스턴스 의존적 어노테이터 가중치 학습
- **Raykar 대비 개선**: 인스턴스 독립 가정 완화

#### (C) MBEM: Minimax Entropy Learning (Shu et al., 2019 → 이후 연구에 영향)

- **방법**: 어노테이터 신뢰도와 분류기를 반복적으로 최적화
- **핵심**: 엔트로피 기반 레이블 정제

#### (D) Union.ai / Label Studio 등 실용적 구현체

- Raykar의 이론을 실제 레이블링 플랫폼에 구현
- 대규모 크라우드소싱 품질 관리에 적용

#### (E) Instance-dependent Label Noise (2020년 이후 활발)

Cheng et al. (2022), "Learning with Instance-Dependent Label Noise" 등의 연구가 Raykar의 인스턴스 독립 가정을 극복하는 방향으로 발전했습니다.

### 5.2 비교 분석 표

| 항목 | Raykar et al. (2009) | CrowdLayer (2018+) | Instance-Dependent (2020+) |
|---|---|---|---|
| **분류기** | 로지스틱 회귀 | 딥러닝 (CNN 등) | 딥러닝 |
| **어노테이터 모델** | 인스턴스 독립 Two-coin | 인스턴스 독립 행렬 | 인스턴스 의존적 |
| **추정 방법** | EM (MLE/MAP) | End-to-end 역전파 | 변분 추론/EM |
| **확장성** | 중간 | 높음 | 높음 |
| **해석 가능성** | 높음 (α, β 직관적) | 중간 | 낮음 |
| **사전 지식 반영** | Beta prior | 제한적 | 다양한 정규화 |
| **누락 레이블** | 지원 | 부분 지원 | 모델별 상이 |

### 5.3 Raykar 논문의 지속적 의의

2020년 이후에도 이 논문은 다음 이유로 중요한 기준점이 됩니다:
1. **이론적 엄밀성**: 확률론적 모델링의 명확한 수식 유도
2. **해석 가능성**: 전문가 성능의 직관적 파라미터화
3. **출발점**: 대부분의 다중 어노테이터 학습 논문이 이 연구를 기반으로 확장
4. **Benchmark baseline**: 새로운 방법의 성능 비교 기준으로 활용

---

## 참고 자료

**주 논문:**
- Raykar, V. C., Yu, S., Zhao, L. H., Jerebko, A., Florin, C., Valadez, G. H., Bogoni, L., & Moy, L. (2009). *Supervised Learning from Multiple Experts: Whom to trust when everyone lies a bit*. Proceedings of the 26th International Conference on Machine Learning (ICML 2009), Montreal, Canada.

**논문 내 인용 참고문헌:**
- Dawid, A. P., & Skeene, A. M. (1979). Maximum likelihood estimation of observed error-rates using the EM algorithm. *Applied Statistics, 28*, 20–28.
- Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. *Journal of the Royal Statistical Society: Series B, 39*, 1–38.
- Smyth, P., Fayyad, U., Burl, M., Perona, P., & Baldi, P. (1995). Inferring ground truth from subjective labelling of venus images. *Advances in Neural Information Processing Systems 7*, 1085–1092.
- Snow, R., O'Connor, B., Jurafsky, D., & Ng, A. (2008). Cheap and Fast - But is it Good? Evaluating Non-Expert Annotations for Natural Language Tasks. *EMNLP 2008*, 254–263.
- Sheng, V. S., Provost, F., & Ipeirotis, P. G. (2008). Get another label? *KDD 2008*, 614–622.

**비교 분석을 위한 참고 (논문에서 직접 확인하지 못한 부분은 일반적 학술 지식 기반):**
- Rodrigues, F., & Pereira, F. C. (2018). Deep Learning from Crowds. *AAAI 2018*.
- 2020년 이후 Instance-Dependent Noise 연구 동향 (일반적 학술 지식)

> **⚠️ 주의**: 2020년 이후 최신 연구의 구체적 수치와 세부 내용은 해당 논문들을 직접 확인하지 못하였으므로, 일반적인 연구 방향성을 기술하였습니다. 정확한 비교를 위해서는 각 논문을 직접 참조하시기 바랍니다.
