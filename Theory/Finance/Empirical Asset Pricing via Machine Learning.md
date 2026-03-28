# Empirical Asset Pricing via Machine Learning

**저자:** Shihao Gu (University of Chicago), Bryan Kelly (Yale University / AQR Capital Management / NBER), Dacheng Xiu (University of Chicago)

**학술지:** *The Review of Financial Studies*, 33(5), 2223–2273, 2020

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장
이 논문은 실증 자산 가격 결정(empirical asset pricing)의 핵심 문제인 **주식 위험 프리미엄(risk premium) 측정**에 머신러닝 기법들을 체계적으로 비교·적용하여, 전통적인 회귀분석 기반 방법론 대비 **대폭적인 예측 성능 향상**이 가능함을 입증한다. 특히 **신경망(neural networks)**과 **회귀 트리(regression trees)**가 가장 우수한 성능을 보이며, 이는 다른 방법들이 포착하지 못하는 **비선형 예측 변수 간 상호작용(nonlinear predictor interactions)**을 허용하기 때문이다.

### 주요 기여
1. **예측 정확도의 새로운 벤치마크 수립:** 다양한 ML 방법론의 out-of-sample $R^2$를 체계적으로 비교하여 개별 주식 및 포트폴리오 수준에서 예측 정확도의 새로운 기준을 설정
2. **경제적 가치 입증:** ML 예측 기반 포트폴리오 전략이 기존 회귀 기반 전략 대비 Sharpe ratio를 최대 2배 이상 개선 (신경망 기반 long-short 전략의 연간 Sharpe ratio 1.35 달성)
3. **방법론 간 체계적 비교 분석:** OLS, Elastic Net, PCR, PLS, 일반화 선형모형, 랜덤포레스트, Gradient Boosted Trees, 신경망(1~5 hidden layers)까지 13개 모형을 동일 조건 하에서 비교
4. **핵심 예측 변수 식별:** 모든 방법론이 공통적으로 **가격 추세(모멘텀, 단기 반전)**, **유동성**, **변동성** 관련 변수를 가장 중요한 예측 인자로 선택

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 향상 및 한계

### 2.1 해결하고자 하는 문제

자산의 위험 프리미엄은 미래 실현 초과수익률의 조건부 기댓값, 즉 $E(r_{i,t+1}|\mathcal{F}_t)$로 정의되며, 이를 정확하게 측정하는 것이 실증 자산 가격 결정의 근본적 과제이다. 그러나 세 가지 핵심 도전이 존재한다:

1. **고차원 예측 변수 문제:** 문헌에서 보고된 주식 수준 예측 특성이 수백 개에 달하며, 거시경제 변수까지 포함하면 900개 이상의 후보 신호가 존재
2. **함수 형태의 불확실성:** 예측 변수가 선형으로 진입하는지, 비선형 변환이 필요한지, 변수 간 상호작용을 고려해야 하는지에 대한 이론적 지침 부족
3. **낮은 신호 대 잡음 비율:** 주식 수익률의 변동이 예측 불가능한 뉴스에 의해 지배되어 위험 프리미엄 신호가 매우 미약

### 2.2 총괄 모형 프레임워크 (수식 포함)

논문의 가장 일반적인 모형은 가산적 예측 오차 모형(additive prediction error model)으로 정의된다:

$$
r_{i,t+1} = E_t(r_{i,t+1}) + \epsilon_{i,t+1} 
$$

여기서 조건부 기대수익률은 예측 변수의 유연한 함수로 표현된다:

$$
E_t(r_{i,t+1}) = g^{\star}(z_{i,t})
$$

- $r_{i,t+1}$: 주식 $i$의 $t+1$ 시점 무위험 이자율 대비 초과수익률
- $z_{i,t}$: $P$차원 예측 변수 벡터
- $g^{\star}(\cdot)$: 미지의 비선형 함수 (주식 $i$나 시점 $t$에 의존하지 않음)

**예측 변수 구성:** 주식 수준 특성 $c_{i,t}$ (94개)와 거시경제 변수 $x_t$ (8개 + 상수)의 상호작용으로 구성:

$$
z_{i,t} = x_t \otimes c_{i,t}
$$

이는 조건부 베타 가격 결정 모형 $E_t(r_{i,t+1}) = \beta'\_{i,t}\lambda_t$를 내포한다. 구체적으로 $\beta_{i,t} = \theta_1 c_{i,t}$, $\lambda_t = \theta_2 x_t$이면:

$$
g^{\star}(z_{i,t}) = c'_{i,t}\theta'_1\theta_2 x_t = (x_t \otimes c_{i,t})'\text{vec}(\theta'_1\theta_2) =: z'_{i,t}\theta 
$$

산업 더미 74개를 포함하여 총 공변량 수는 $94 \times (8+1) + 74 = 920$개이다.

### 2.3 제안하는 방법론 (모델 구조)

#### (1) 단순 선형 모형 (OLS)

$$
g(z_{i,t};\theta) = z'_{i,t}\theta 
$$

목적 함수:

$$
\mathcal{L}(\theta) = \frac{1}{NT}\sum_{i=1}^{N}\sum_{t=1}^{T}\left(r_{i,t+1} - g(z_{i,t};\theta)\right)^2 
$$

가중 최소제곱(WLS) 변형:

$$
\mathcal{L}_W(\theta) = \frac{1}{NT}\sum_{i=1}^{N}\sum_{t=1}^{T}w_{i,t}\left(r_{i,t+1} - g(z_{i,t};\theta)\right)^2 
$$

**Huber 강건 목적 함수:**

$$
\mathcal{L}_H(\theta) = \frac{1}{NT}\sum_{i=1}^{N}\sum_{t=1}^{T}H\left(r_{i,t+1} - g(z_{i,t};\theta), \xi\right) 
$$

여기서

$$
H(x;\xi) = \begin{cases} x^2, & \text{if } |x| \leq \xi \\ 2\xi|x| - \xi^2, & \text{if } |x| > \xi \end{cases}
$$

#### (2) 벌점 선형 모형 (Elastic Net)

정규화된 목적 함수:

$$
\mathcal{L}(\theta;\cdot) = \mathcal{L}(\theta) + \phi(\theta;\cdot) 
$$

Elastic Net 벌점 함수:

$$
\phi(\theta;\lambda,\rho) = \lambda(1-\rho)\sum_{j=1}^{P}|\theta_j| + \frac{1}{2}\lambda\rho\sum_{j=1}^{P}\theta_j^2 
$$

- $\rho = 0$: **Lasso** ($l_1$ 벌점, 변수 선택)
- $\rho = 1$: **Ridge** ($l_2$ 벌점, 축소)
- 중간값: 축소와 선택의 결합

#### (3) 차원 축소: PCR과 PLS

벡터화된 선형 모형 $R = Z\theta + E$에서 $P$차원을 $K$개 선형 결합으로 축소:

$$
R = (Z\Omega_K)\theta_K + \tilde{E}
$$

**PCR** — $Z$의 공분산 구조를 가장 잘 보존하는 성분 선택:

$$
w_j = \arg\max_w \text{Var}(Zw), \quad \text{s.t.} \quad w'w = 1, \quad \text{Cov}(Zw, Zw_l) = 0, \quad l = 1,\ldots,j-1 
$$

**PLS** — 예측 대상과의 공분산을 극대화하는 성분 선택:

$$
w_j = \arg\max_w \text{Cov}^2(R, Zw), \quad \text{s.t.} \quad w'w = 1, \quad \text{Cov}(Zw, Zw_l) = 0, \quad l = 1,\ldots,j-1 
$$

#### (4) 일반화 선형 모형 (GLM + Group Lasso)

스플라인 기저 함수 확장:

$$
g(z;\theta, p(\cdot)) = \sum_{j=1}^{P}p(z_j)'\theta_j 
$$

Group Lasso 벌점:

$$
\phi(\theta;\lambda,K) = \lambda\sum_{j=1}^{P}\left(\sum_{k=1}^{K}\theta_{j,k}^2\right)^{1/2}
$$

#### (5) 회귀 트리 (Boosted Trees, Random Forests)

$K$개의 말단 노드와 깊이 $L$의 트리 예측:

$$
g(z_{i,t};\theta,K,L) = \sum_{k=1}^{K}\theta_k \mathbf{1}_{\{z_{i,t} \in C_k(L)\}} 
$$

각 분기에서의 불순도(impurity):

$$
H(\theta, C) = \frac{1}{|C|}\sum_{z_{i,t} \in C}(r_{i,t+1} - \theta)^2 
$$

- **Gradient Boosted Regression Trees (GBRT):** 얕은 트리를 반복적으로 잔차에 적합시켜 앙상블 구성. 학습률 $\nu \in (0,1)$로 축소. 튜닝 파라미터 $(L, \nu, B)$
- **Random Forest:** 부트스트랩 샘플링 + "dropout" 방식으로 트리 간 상관을 줄여 분산 감소

#### (6) 신경망 (Neural Networks)

각 은닉층 $l > 0$의 뉴런 $k$ 출력:

$$
x_k^{(l)} = \text{ReLU}\left(x^{(l-1)'}\theta_k^{(l-1)}\right) 
$$

최종 출력:

$$
g(z;\theta) = x^{(L-1)'}\theta^{(L-1)} 
$$

여기서 $\text{ReLU}(x) = \max(0, x)$

- **NN1~NN5:** 1~5개 은닉층, 각각 32, 16, 8, 4, 2 뉴런 (기하 피라미드 규칙)
- 정규화: $l_1$ 벌점, 학습률 축소(Adam), 조기 종료, 배치 정규화, 앙상블

### 2.4 성능 평가

Out-of-sample $R^2$:

$$
R^2_{\text{oos}} = 1 - \frac{\sum_{(i,t) \in \mathcal{T}_3}(r_{i,t+1} - \hat{r}_{i,t+1})^2}{\sum_{(i,t) \in \mathcal{T}_3}r_{i,t+1}^2} 
$$

분모에서 평균을 차감하지 않고 **제로 예측 대비** 벤치마킹한다 (역사적 평균 수익률이 극히 noisy하기 때문).

**Diebold-Mariano 검정 통계량:**

$$
d_{12,t+1} = \frac{1}{n_{3,t+1}}\sum_{i=1}^{n_{3,t+1}}\left(\left(\hat{e}_{i,t+1}^{(1)}\right)^2 - \left(\hat{e}_{i,t+1}^{(2)}\right)^2\right) 
$$

### 2.5 주요 실증 결과 (성능 향상)

| 모형 | 월간 $R^2_{\text{oos}}$ (%) |
|---|---|
| OLS (전체 변수) | −3.46 |
| OLS-3 (size, B/M, momentum) | 0.16 |
| Elastic Net + Huber | 0.11 |
| PLS | 0.27 |
| PCR | 0.26 |
| GLM + Huber | 0.19 |
| Random Forest | 0.33 |
| GBRT + Huber | 0.34 |
| NN1 | 0.33 |
| NN2 | 0.39 |
| **NN3** | **0.40** |
| NN4 | 0.39 |
| NN5 | 0.36 |

**포트폴리오 수준 성과:**
- S&P 500 bottom-up 예측: NN3 $R^2_{\text{oos}} = 1.80\%$/월
- 신경망 기반 long-short decile spread: Sharpe ratio **1.35** (가치 가중), **2.45** (동일 가중)
- S&P 500 타이밍 전략: Sharpe ratio **0.77** vs. buy-and-hold **0.51**

**Sharpe ratio 향상 공식 (Campbell and Thompson 2008):**

$$
SR^* = \sqrt{\frac{SR^2 + R^2}{1 - R^2}}
$$

### 2.6 한계

1. **"얕은" 학습이 "깊은" 학습보다 우수:** 3개 은닉층 이후 성능 하락. 이는 금융 데이터의 작은 표본 크기와 낮은 신호 대 잡음 비율에 기인
2. **경제적 메커니즘 식별 불가:** ML 방법은 조건부 기댓값의 근사 측정치만 제공하며, 균형(equilibrium) 관계나 인과적 메커니즘을 밝히지 못함
3. **높은 포트폴리오 회전율:** ML 포트폴리오의 월간 회전율이 110~130%로 거래비용에 민감
4. **데이터 의존성:** 미국 주식시장 1957~2016 기간에 한정되어 다른 시장이나 자산군으로의 일반화 검증 부족
5. **비선형 모형의 해석 가능성 제한:** 신경망, 트리 모형의 "블랙박스" 특성으로 경제적 직관 도출에 어려움

---

## 3. 모델의 일반화 성능 향상 가능성

이 논문에서 일반화 성능 향상과 관련된 핵심 내용을 다음과 같이 정리할 수 있다:

### 3.1 과적합 방지를 위한 정규화(Regularization) 전략

논문은 ML 방법의 **핵심 방어 수단**으로 정규화를 강조한다. 예측 오차를 분해하면:

$$
r_{i,t+1} - \hat{r}_{i,t+1} = \underbrace{g^{\star}(z_{i,t}) - g(z_{i,t};\theta)}_{\text{근사 오차(approximation error)}} + \underbrace{g(z_{i,t};\theta) - g(z_{i,t};\hat{\theta})}_{\text{추정 오차(estimation error)}} + \underbrace{\epsilon_{i,t+1}}_{\text{내재 오차(intrinsic error)}}
$$

- **근사 오차:** 유연한 함수 형태(트리, 신경망)를 사용하여 감소 가능
- **추정 오차:** 정규화(벌점, 차원축소, 조기종료 등)를 통해 통제
- 유연성과 정규화 사이의 **편향-분산 트레이드오프**가 일반화의 핵심

### 3.2 샘플 분할 및 검증(Validation) 전략

논문은 시간 순서를 유지하는 **3-way 분할**을 사용한다:
- **훈련 표본(1957~1974):** 모형 파라미터 추정
- **검증 표본(1975~1986):** 하이퍼파라미터 튜닝
- **테스트 표본(1987~2016):** 순수 out-of-sample 평가

교차검증(cross-validation)을 사용하지 않는 이유는 **시계열 데이터의 시간적 순서를 유지**하기 위함이다. 매년 훈련 표본을 1년씩 확장하고 검증 표본을 forward roll하여 **준실시간 환경**을 시뮬레이션한다.

### 3.3 일반화 성능에 기여하는 구체적 기법들

| 기법 | 적용 모형 | 일반화 성능 기여 메커니즘 |
|---|---|---|
| $l_1 / l_2$ 벌점 | Elastic Net, 신경망 | 파라미터 희소성/축소 → 과적합 방지 |
| Group Lasso | GLM | 특성 단위 변수 선택 |
| 차원 축소 (PCR/PLS) | 선형 모형 | 다중공선성 해소, 잡음 평균화 |
| 조기 종료(Early Stopping) | 신경망 | 훈련 과정에서 검증 오차 증가 시 학습 중단 |
| 배치 정규화(Batch Normalization) | 신경망 | 내부 공변량 이동 문제 완화 |
| 앙상블 | 신경망, 트리 | 다수의 약한 학습기/초기값 조합으로 예측 분산 감소 |
| Dropout (predictor subset) | Random Forest | 트리 간 상관 감소 → 분산 감소 |
| 학습률 축소 (Adam) | 신경망 | 그래디언트 잡음 지배 방지 |
| Huber 손실 함수 | 다수 모형 | 두꺼운 꼬리 분포에 대한 강건성 |

### 3.4 "얕은 학습"이 우수한 이유와 일반화 시사점

신경망 성능이 3개 은닉층에서 정점을 찍고 하락하며, 트리 모형도 평균 6개 미만의 말단 노드를 선택한다. 이는 금융 데이터의 **제한된 표본 크기**와 **극히 낮은 신호 대 잡음 비율** 때문이다. 컴퓨터 비전 등에서 깊은 신경망이 성공하는 것은 천문학적 데이터와 강한 신호 덕분이며, 자산 가격 문제에서는 **과도한 모형 복잡도가 오히려 일반화를 해침**을 시사한다.

### 3.5 일반화 성능 향상을 위한 추가적 가능성

논문에서 직·간접적으로 제시하는 일반화 성능 향상 방향:

1. **포트폴리오 수준 집계(aggregation):** 개별 주식의 예측 불가능한 잡음이 평균화되어 포트폴리오 수준에서 $R^2$가 대폭 상승 (개별 0.40% → S&P 500 1.80%)
2. **Placebo 변수 테스트:** 5개의 가짜 특성을 추가해도 핵심 예측 변수의 중요도 순위가 변하지 않음 → 모형의 강건성 확인
3. **연간 수익률 예측:** 월간 대비 $R^2$가 거의 한 자릿수(order of magnitude) 높아져 ML 모형이 비즈니스 사이클 빈도의 지속적 위험 프리미엄을 포착
4. **메타전략:** 여러 ML 모형의 예측을 결합하면 단일 모형보다 높은 $R^2$ 달성 가능 (패널 $R^2$ 0.43~0.45%)

---

## 4. 향후 연구에 미치는 영향과 고려할 점

### 4.1 향후 연구에 미치는 영향

1. **ML 기반 자산 가격 결정의 표준 벤치마크 수립:** 이 논문은 후속 연구가 ML 방법의 예측 성능을 비교할 때 참조하는 사실상의 표준(de facto standard)이 되었음
2. **경제적 구조와 ML의 결합:** 저자들은 ML이 단순히 예측 도구이며 경제적 균형 메커니즘을 밝히지 못한다고 명시적으로 언급. 이에 따라 **구조적 모형에 ML을 결합하는 연구**(예: Kelly, Pruitt, and Su 2019; Gu, Kelly, and Xiu 2019; Feng, Giglio, and Xiu forthcoming)가 후속으로 활발히 진행
3. **핀테크 산업의 학술적 정당화:** 포트폴리오 선택, 시장 타이밍, 위험 관리에서 ML의 역할을 실증적으로 뒷받침
4. **비선형 상호작용의 중요성 부각:** 단순 비선형 변환(스플라인)보다 변수 간 상호작용이 핵심이라는 발견은 이후 연구에서 모형 설계의 방향을 제시

### 4.2 향후 연구 시 고려할 점

1. **거래비용과 시장 마찰:** 높은 회전율(110~130%/월)은 실제 투자에서 성과를 크게 잠식할 수 있으며, 거래비용을 목적 함수에 직접 반영하는 연구가 필요
2. **해석 가능성(Interpretability):** 비선형 모형의 경제적 해석을 위한 Shapley value, attention mechanism 등의 도구 개발
3. **다른 시장/자산군으로의 확장:** 미국 주식시장 외 국제 시장, 채권, 파생상품 등으로의 일반화 검증
4. **대안적 데이터 활용:** 텍스트, 위성 이미지, 소셜 미디어 등의 대안 데이터를 예측 변수에 추가
5. **시간 변동 모형 구조:** $g(\cdot)$ 함수가 시간이나 개별 주식에 따라 변하는 것을 허용하는 모형 탐색
6. **인과 추론과의 결합:** ML 예측과 인과적 식별(causal identification)을 결합하는 방법론 개발

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 주요 후속 연구

| 연구 | 핵심 기여 | Gu et al. (2020)과의 관계 |
|---|---|---|
| **Gu, Kelly, and Xiu (2021)** "Autoencoder Asset Pricing Models," *Journal of Econometrics* | 오토인코더를 활용하여 조건부 베타와 팩터를 동시에 추정. 비선형 팩터 구조 학습 | Gu et al. (2020)의 예측 프레임워크에 **균형 자산 가격 구조(no-arbitrage 조건)**를 결합하여 경제적 해석 가능성 향상 |
| **Kelly, Pruitt, and Su (2019, JFE 게재 2020)** "Characteristics are Covariances" | 조건부 IPCA 모형으로 특성-기반 팩터 추정 | Gu et al. (2020)이 순수 예측에 초점이라면, IPCA는 자산 가격 결정의 구조적 모형을 제공 |
| **Feng, Giglio, and Xiu (2020)** "Taming the Factor Zoo," *Journal of Finance* | 수백 개의 팩터 후보를 체계적으로 검증하는 프레임워크 | 다중비교 문제를 엄밀히 처리하여 Gu et al.이 식별한 예측 변수의 통계적 유효성을 보완 |
| **Chen, Pelger, and Zhu (2024)** "Deep Learning in Asset Pricing," *Management Science* | GAN(생성적 적대 신경망) 기반 조건부 자산 가격 모형. SDF를 직접 추정 | Gu et al. (2020)의 예측 접근 대신 **no-arbitrage 조건을 신경망 학습 목적에 직접 통합** |
| **Bianchi, Büchner, and Tamoni (2021)** "Bond Risk Premiums with Machine Learning," *RFS* | ML을 채권시장에 적용하여 기간 프리미엄 측정 | Gu et al.의 프레임워크를 **채권시장으로 확장** |
| **Avramov, Cheng, and Metzker (2023)** "Machine Learning vs. Economic Restrictions," *Management Science* | ML 예측에 경제적 제약(no-short-sale, 거래비용 등)을 부과 | Gu et al.의 한계인 **거래비용 미반영 문제**를 직접 해결 시도 |
| **Leippold, Wang, and Zhou (2022)** "Machine Learning in the Chinese Stock Market," *Journal of Financial Economics* | 중국 주식시장에서 ML 자산 가격 분석 | Gu et al.의 방법론을 **미국 외 시장에 적용**하여 일반화 가능성 검증 |
| **Kaniel, Lin, Pelger, and Van Nieuwerburgh (2023)** "Machine-Learning the Skill of Mutual Fund Managers" | ML로 펀드 매니저의 스킬을 측정 | Gu et al.의 예측 도구를 **펀드 성과 평가**에 적용 |
| **Bryzgalova, Pelger, and Zhu (2020)** "Forest Through the Trees," *Journal of Finance* | 관리 포트폴리오 선택에 의사결정 트리 활용 | Gu et al.의 트리 기반 모형을 **포트폴리오 구성에 직접 응용** |
| **Freyberger, Neuhierl, and Weber (2020)** "Dissecting Characteristics Nonparametrically," *RFS* | 비모수적 방법으로 특성-수익률 관계 분석, adaptive group lasso 사용 | Gu et al.의 GLM과 유사한 접근이나 **더 세밀한 비모수적 함수 형태** 허용 |

### 5.2 최신 연구에서 나타나는 주요 트렌드

1. **경제적 구조 통합:** 순수 예측 모형에서 no-arbitrage 조건, SDF 추정, 팩터 구조 등을 ML에 내장하는 방향으로 진화
2. **대안 데이터 활용:** 텍스트, 뉴스 감성, 옵션 내재 정보, ESG 데이터 등의 통합
3. **시간 변동 허용:** 정적 함수 $g(\cdot)$ 대신 시간에 따라 변하는 구조 학습 (예: recurrent neural networks, transformer)
4. **거래비용 인식 모형:** 거래비용을 목적 함수에 직접 반영하여 실현 가능한 전략 수익률 평가
5. **국제적 확장:** 미국 이외 시장에서의 ML 자산 가격 유효성 검증
6. **해석 가능성 향상:** SHAP, LIME, attention 기반 해석 도구를 금융 ML에 적용

---

## 참고자료

1. **Gu, S., Kelly, B., & Xiu, D. (2020).** "Empirical Asset Pricing via Machine Learning." *The Review of Financial Studies*, 33(5), 2223–2273. (본 논문)
2. **Gu, S., Kelly, B., & Xiu, D. (2021).** "Autoencoder Asset Pricing Models." *Journal of Econometrics*, 222(1), 429–450.
3. **Kelly, B., Pruitt, S., & Su, Y. (2019).** "Characteristics are Covariances: A Unified Model of Risk and Return." *Journal of Financial Economics*, 134(3), 501–524.
4. **Feng, G., Giglio, S., & Xiu, D. (2020).** "Taming the Factor Zoo: A Test of New Factors." *Journal of Finance*, 75(3), 1327–1370.
5. **Chen, L., Pelger, M., & Zhu, J. (2024).** "Deep Learning in Asset Pricing." *Management Science*, 70(2), 714–750.
6. **Freyberger, J., Neuhierl, A., & Weber, M. (2020).** "Dissecting Characteristics Nonparametrically." *Review of Financial Studies*, 33(5), 2326–2377.
7. **Campbell, J. Y., & Thompson, S. B. (2008).** "Predicting Excess Stock Returns Out of Sample: Can Anything Beat the Historical Average?" *Review of Financial Studies*, 21(4), 1509–1531.
8. **Welch, I., & Goyal, A. (2008).** "A Comprehensive Look at the Empirical Performance of Equity Premium Prediction." *Review of Financial Studies*, 21(4), 1455–1508.
9. **Lewellen, J. (2015).** "The Cross-section of Expected Stock Returns." *Critical Finance Review*, 4, 1–44.
10. **Hastie, T., Tibshirani, R., & Friedman, J. (2009).** *The Elements of Statistical Learning*. Springer.
11. **Bianchi, D., Büchner, M., & Tamoni, A. (2021).** "Bond Risk Premiums with Machine Learning." *Review of Financial Studies*, 34(2), 1046–1089.
12. **Avramov, D., Cheng, S., & Metzker, L. (2023).** "Machine Learning vs. Economic Restrictions: Evidence from Stock Return Predictability." *Management Science*, 69(5), 2547–2576.
13. **Leippold, M., Wang, Q., & Zhou, W. (2022).** "Machine Learning in the Chinese Stock Market." *Journal of Financial Economics*, 145(2), 64–82.
14. **Bryzgalova, S., Pelger, M., & Zhu, J. (2020).** "Forest Through the Trees: Building Cross-Sections of Stock Returns." Working Paper, Stanford University.
15. **Diebold, F. X., & Mariano, R. S. (1995).** "Comparing Predictive Accuracy." *Journal of Business & Economic Statistics*, 13, 134–144.

---

> **참고:** 위 분석은 제공된 원문을 기반으로 작성되었으며, 2020년 이후 후속 연구에 대한 내용은 해당 논문들의 공개된 버전 및 학술지 게재 정보를 참고하였습니다. 일부 최신 연구의 세부 결과는 원문 접근 가능 여부에 따라 제한적으로 기술하였습니다.
