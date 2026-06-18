# A Weighted-Likelihood Framework for Class Imbalance in Bayesian Prediction Models

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

이 논문(Lazic, 2025; arXiv:2504.17013v2)은 **베이지안 예측 모델에서 클래스 불균형 문제를 가중 우도(weighted likelihood, 또는 power likelihood)를 통해 직접 해결**할 수 있다고 주장합니다. 각 관측값의 우도를 해당 클래스의 비율에 반비례하는 지수로 거듭제곱함으로써, 소수 클래스(minority class)의 기여도를 베이지안 업데이팅 과정에 내재화합니다.

### 주요 기여

| 기여 영역 | 내용 |
|-----------|------|
| **방법론적 기여** | 베이지안 업데이팅에 비용 민감 학습(cost-sensitive learning)을 직접 통합 |
| **일반성** | 이진 분류, 순서형 로지스틱 회귀 등 다양한 모델에 적용 가능 |
| **구현 용이성** | Stan, PyMC, Turing.jl에서 1~2줄 코드 수정으로 구현 가능 |
| **임계값 안정성** | 가중 분석에서는 $p = 0.5$가 자연스러운 결정 임계값으로 사용 가능 |
| **응용 실증** | 시뮬레이션 이진 데이터 및 DILI(약물 유발 간 손상) 실제 데이터로 검증 |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

**클래스 불균형(Class Imbalance)** 은 예측 독성학에서 만연한 문제입니다:

- 비독성 화합물 수 >> 독성 화합물 수
- 표준 모델은 전체 정확도는 높으나 소수 클래스(독성) 탐지에 실패
- 기존 대응 방법의 한계:
  - **샘플링 기반**: SMOTE, ADASYN 등은 유효 표본 크기를 변경하여 베이지안 사후 추정의 정밀도를 인위적으로 높임
  - **사후 확률 조정** (Nassiri et al., 2024): 파라미터 추정값, 결정 경계, 분류 지표는 변경하지 않음

논문은 **우도 함수 자체를 수정**함으로써 클래스 불균형을 모델 학습 단계에서 근본적으로 해결하고자 합니다.

---

### 2.2 제안하는 방법 (수식 포함)

#### 기본 베이지안 업데이팅

$$p(\theta | x_{1:N}) \propto p(\theta) \prod_{i=1}^{N} p(x_i | \theta)$$

#### 가중 우도 베이지안 업데이팅

$$p(\theta | x_{1:N}) \propto p(\theta) \prod_{i=1}^{N} p(x_i | \theta)^{w_i} \tag{1}$$

여기서:
- $w_i > 0$: 각 관측값에 부여되는 가중치
- $\sum_{i} w_i = N$: 전체 정보량 보존 조건

#### 가중치 계산 방법

클래스 $k$의 비율이 $\pi_k$일 때, 비정규화 가중치는:

$$\tilde{w}_i = \frac{1}{\pi_{k(i)}}$$

정규화 과정 (전체 정보량이 $N$이 되도록):

$$w_i = \frac{\tilde{w}_i}{\sum_{j=1}^{N} \tilde{w}_j} \times N$$

**예시**: 클래스 비율이 75%, 25%인 경우

$$\tilde{w}_{\text{Class 0}} = \frac{1}{0.75} = 1.33, \quad \tilde{w}_{\text{Class 1}} = \frac{1}{0.25} = 4.00$$

8개 샘플(6개: Class 0, 2개: Class 1)의 경우:
- 비정규화 합: $6 \times 1.33 + 2 \times 4 = 16$
- 정규화 가중치: $w_{\text{Class 0}} = \frac{1.33}{16} \times 8 = 0.67$, $w_{\text{Class 1}} = \frac{4}{16} \times 8 = 2.00$

---

### 2.3 모델 구조

#### 이진 분류 모델 (Bayesian Binary Classification)

$$y_i \sim \text{Bernoulli}(\eta_i)^{w_i}, \quad i = 1, \ldots, N$$

$$\text{logit}(\eta_i) = X_{ij} \beta_j$$

$$\beta_j \sim \mathcal{N}(0, \sigma)$$

로그 우도 가중치 적용 시 (Stan 코드):

```stan
target += bernoulli_lpmf(y | eta) * w;
```

Turing.jl (Julia):

```julia
@addlogprob! sum(loglikelihood.(Bernoulli.(eta), y) .* w)
```

PyMC (Python):

```python
logprob = pm.Bernoulli.logp(y, p=eta)
pm.Potential("weighted_LL", (w * logprob).sum())
```

#### DILI 순서형 로지스틱 모델

- Williams et al. (2020)의 순서형 로지스틱 회귀 모델에 가중 로그 우도를 적용
- 3단계 순서형 결과: Class 1(안전) → Class 2(경도 간독성) → Class 3(고독성)
- 클래스 비율: 0.34, 0.42, 0.24

---

### 2.4 성능 향상 결과

#### 시뮬레이션 이진 데이터 (클래스 비율: 0.87 vs 0.13)

| 지표 | 비가중 분석 | 가중 분석 | 개선 여부 |
|------|-----------|---------|----------|
| AUC | 0.83 | 0.83 | 동일 |
| 정확도(Accuracy) | **0.88** | 0.68 | 비가중 ↑ |
| 균형 정확도(Balanced Accuracy) | 0.57 | **0.69** | 가중 ↑ |
| Brier Score | **0.09** | 0.18 | 비가중 ↑ |
| 균형 Brier Score | 0.28 | **0.17** | 가중 ↑ |
| 민감도(Sensitivity) | 0.15 | **0.69** | 가중 ↑ |
| 특이도(Specificity) | **0.99** | 0.68 | 비가중 ↑ |
| F1 Score | 0.25 | **0.36** | 가중 ↑ |
| P4 Metric | 0.39 | **0.49** | 가중 ↑ |

#### DILI 순서형 데이터 (1 vs {2,3} / {1,2} vs 3)

- 전체 정확도: 동일 (0.70)
- 균형 정확도: 가중 모델 소폭 우세 (0.66 vs 0.65)
- $\{1,2\}$ vs $3$ 구분 시 민감도: **0.52(가중) vs 0.22(비가중)** → 가중 모델 현저히 우수

---

### 2.5 한계점

1. **캘리브레이션 저하**: 가중 분석은 클래스 불균형 보정 과정에서 예측 확률의 캘리브레이션을 저하시킴
   - 평균 캘리브레이션 MSE: 0.02(비가중) vs 0.11(가중)
   - Intercept: 0.07(비가중) vs -2.00(가중) → 비가중 모델이 완벽 캘리브레이션(0)에 더 근접

2. **메트릭 트레이드오프**: 민감도 향상 시 특이도 감소, 정확도 감소는 불가피

3. **소규모 표본에서의 재캘리브레이션 어려움**: 재캘리브레이션 방법은 더 큰 표본 크기가 필요

4. **광범위한 시뮬레이션 비교 미실시**: 가중/비가중 모델의 체계적 비교 연구 미진행

5. **모델 선택의 경험적 의존성**: 어떤 방법이 우수한지는 적용마다 경험적으로 결정해야 함

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화 성능 향상을 지지하는 근거

#### (1) 클래스 불균형 편향 제거

비가중 모델에서 소수 클래스에 대한 예측 확률이 낮게 편향되는 문제를 해결합니다. 시뮬레이션 데이터에서 비가중 모델의 Class 1 예측 확률은 대부분 낮은 범위에 집중된 반면, 가중 모델은 전체 확률 범위를 사용하여 더 균형 잡힌 예측 분포를 생성합니다.

#### (2) 결정 경계의 이동

가중 분석은 절편($\beta_0$) 추정값을 변화시켜 **결정 경계(decision boundary)를 이동**시킵니다:

$$\text{logit}(p) = \beta_0 + \beta_1 X_1 + \beta_2 X_2$$

가중 모델은 소수 클래스 방향으로 결정 경계를 이동시켜, $p = 0.5$를 자연스러운 임계값으로 사용할 수 있게 합니다. 이는 새로운 데이터에 적용 시 임계값 재설정 필요성을 없애므로 **일반화 안정성이 향상**됩니다.

#### (3) LOO(Leave-One-Out) 검증 적용

DILI 데이터 분석에서 **LOO 교차 검증**을 통해 예측 성능을 평가함으로써, 단순 학습 데이터 과적합이 아닌 일반화 성능을 평가하였습니다.

#### (4) 임계값 안정성

가중 분석의 $p = 0.5$ 임계값 사용 가능성은 중요한 일반화 이점입니다:

> "Thresholds defined after a model has been built are often unstable, even when the sample size is reasonably large [Wynants et al., 2019]"

- 비가중 분석: 클래스 불균형 정도에 따라 임계값이 다르며, 사전 지정이 어려움
- 가중 분석: $p = 0.5$가 적절한 임계값으로 기능하여 **새로운 데이터셋에 대한 이전 가능성 향상**

#### (5) 전체 사후 예측 분포 활용

베이지안 프레임워크의 특성상 단순 점 추정이 아닌 **전체 사후 예측 분포(posterior predictive distribution)**를 제공합니다. 이는 불확실성 정량화를 통해 실제 환경에서의 의사결정 지원 능력을 향상시킵니다.

### 3.2 일반화 성능 향상의 제약

#### 캘리브레이션과의 트레이드오프

$$\text{Calibration MSE} = \frac{1}{n} \sum_{i=1}^{n}(\hat{p}_i - y_i)^2$$

가중 모델의 캘리브레이션 저하는 실제 환경에서 훈련 데이터의 클래스 불균형이 실제 모집단을 반영하는 경우 **과도한 소수 클래스 탐지로 인한 오경보(false alarm) 증가** 문제를 초래할 수 있습니다.

#### 훈련-실제 불균형 차이 문제

훈련 데이터의 불균형이 실제 배포 환경을 반영한다면, 가중 분석은 과도한 보정으로 실제 환경에서 성능이 저하될 수 있습니다. 반대로, 훈련 데이터의 불균형이 인위적인 것이라면 가중 분석이 실제 일반화 성능을 향상시킵니다.

---

## 4. 앞으로의 연구에 미치는 영향과 고려할 점

### 4.1 향후 연구에 미치는 영향

#### (1) 베이지안 독성 예측 연구의 방향성 제시
가중 우도 프레임워크는 Tox21, eTOX와 같은 대규모 불균형 독성 데이터셋에 적용 가능한 일반적 기반을 제공합니다. 논문 자체에서도 "multi-endpoint toxicity prediction" 및 "large-scale imbalanced datasets"로의 확장 가능성을 제시하고 있습니다.

#### (2) 비용 민감 베이지안 학습의 체계화
기존에는 주로 빈도주의 머신러닝에서 활용되던 비용 민감 학습이 베이지안 프레임워크에 통합될 수 있는 원리적 토대를 제공합니다. 이는 규제 과학(regulatory science) 분야에서 비대칭적 오류 비용을 갖는 의사결정 모델 개발에 중요한 영향을 미칩니다.

#### (3) 확률적 프로그래밍 언어 생태계와의 통합
Stan, PyMC, Turing.jl 등 주요 확률적 프로그래밍 언어에서 구현 예시를 제공함으로써, 재현 가능한 연구와 실용적 적용을 촉진합니다.

#### (4) 평가 지표 다양화 논의 촉진
P4-metric(Sitarz, 2023), 균형 Brier Score, 보정 계층(calibration hierarchy) 등 다양한 평가 지표의 중요성을 강조함으로써, 단순 정확도 중심의 평가 관행에서 벗어나는 연구 흐름을 강화합니다.

### 4.2 향후 연구 시 고려할 점

#### (1) 최적 가중치 결정 방법론 연구
현재 논문은 가중치를 클래스 비율의 역수로 객관적으로 설정하지만, **최적 가중치를 데이터로부터 학습**하는 방법론이 필요합니다. 예를 들어:

$$w^* = \arg\min_{w} \mathcal{L}(\text{balanced accuracy}, w)$$

교차 검증 등을 통해 최적 가중치를 탐색하는 연구가 필요합니다.

#### (2) 캘리브레이션 복원 방법과의 결합
가중 우도 적용 후 발생하는 캘리브레이션 저하를 복원하기 위한 방법(예: Platt scaling, Beta calibration, Temperature scaling)과의 결합 연구가 필요합니다.

#### (3) 대규모 데이터셋에서의 체계적 시뮬레이션
논문은 소규모 데이터셋(100개, 96개)에서만 검증되었습니다. 다양한 불균형 비율, 표본 크기, 특징 공간 차원에서의 **체계적 시뮬레이션 연구**가 필요합니다.

#### (4) 배포 환경의 클래스 비율과 훈련 비율 불일치 문제
실제 배포 환경에서 클래스 비율이 훈련 데이터와 다를 경우의 성능 영향을 평가하는 연구가 필요합니다. 특히 **covariate shift** 및 **label shift** 상황에서의 가중 우도 접근법의 강건성 평가가 중요합니다.

#### (5) 다중 엔드포인트 독성 예측으로의 확장
단일 독성 엔드포인트가 아닌 다중 독성 엔드포인트(간독성, 심장독성, 신장독성 등)를 동시에 예측하는 **멀티태스크 베이지안 모델**에서의 가중 우도 적용 연구가 필요합니다.

#### (6) 사전 분포와의 상호작용
가중 우도가 사전 분포의 영향력과 어떻게 상호작용하는지에 대한 이론적 분석이 부족합니다. 특히 정보적 사전 분포(informative prior) 사용 시 가중치와의 균형 문제를 연구해야 합니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 방법 | 베이지안 여부 | 파라미터 변경 | 캘리브레이션 고려 | 결정 경계 변경 |
|------|------|------------|------------|----------------|------------|
| **Lazic (2025)** | 가중 우도(Power Likelihood) | ✅ | ✅ | ⚠️ 저하 가능 | ✅ |
| **Nassiri et al. (2024)** | 사후 확률 조정 | ✅ | ❌ | ✅ 유지 | ❌ |
| **Piccininni et al. (2024)** | 재샘플링 방법 비교 분석 | ❌ | - | ⚠️ 저하 확인 | - |
| **van den Goorbergh et al. (2022)** | 클래스 불균형 보정의 해악 연구 | ❌ | - | ⚠️ 저하 확인 | - |

### 주요 비교 분석

#### Nassiri et al. (2024) vs Lazic (2025)

- **Nassiri et al. (2024)**: 베이지안 분류에서 사후 클래스 확률을 학습 데이터의 클래스 비율을 반영하도록 조정. 파라미터 추정값, 결정 경계, 표준 분류 지표는 변경하지 않음
- **Lazic (2025)**: 우도 함수 자체를 수정하여 파라미터 추정값과 결정 경계 모두 변경. 소수 클래스 탐지 능력 향상

두 방법은 상호 보완적이며, 목적에 따라 선택하거나 조합할 수 있습니다.

#### Piccininni et al. (2024) 및 van den Goorbergh et al. (2022)

- 무작위 재샘플링 기법이 캘리브레이션을 저하시킨다는 것을 실증적으로 확인
- Lazic (2025)의 가중 우도 방법도 캘리브레이션 저하 문제에서 자유롭지 않음을 시사
- 두 연구 모두 **클래스 불균형 보정 방법 적용 후 캘리브레이션 복원 절차의 필요성**을 강조

---

## 참고 자료

- **주 논문**: Lazic, S. E. (2025). *A weighted-likelihood framework for class imbalance in Bayesian prediction models*. arXiv:2504.17013v2 [stat.AP].
- Nassiri, V., Tekle, F., Tatikola, K., & Geys, H. (2024). Addressing class imbalance in Bayesian classification through posterior probability adjustment. *Biometrical Journal*, 66(8). doi:10.1002/bimj.70004
- Piccininni, M. et al. (2024). Understanding random resampling techniques for class imbalance correction and their consequences on calibration and discrimination of clinical risk prediction models. *Journal of Biomedical Informatics*, 155:104666.
- van den Goorbergh, R. et al. (2022). The harm of class imbalance corrections for risk prediction models. *Journal of the American Medical Informatics Association*, 29(9):1525–1534.
- Williams, D. P. et al. (2020). Predicting drug-induced liver injury with Bayesian machine learning. *Chem. Res. Toxicol.*, 33(1):239–248.
- Van Calster, B. et al. (2019). Calibration: the Achilles heel of predictive analytics. *BMC Medicine*, 17(1).
- Holmes, C. C., & Walker, S. G. (2017). Assigning a value to a power likelihood in a general Bayesian model. *Biometrika*, 104(2):497–503.
- Krawczyk, B. (2016). Learning from imbalanced data: open challenges and future directions. *Progress in Artificial Intelligence*, 5(4):221–232.
- Wynants, L. et al. (2019). Three myths about risk thresholds for prediction models. *BMC Medicine*, 17(1).
- Sitarz, M. (2023). Extending F1 metric, probabilistic approach. *Advances in Artificial Intelligence and Machine Learning*, 03(02):1025–1038.
