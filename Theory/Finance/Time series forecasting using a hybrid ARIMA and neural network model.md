# Time Series Forecasting Using a Hybrid ARIMA and Neural Network Model
**G. Peter Zhang, Neurocomputing 50 (2003) 159–175**

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

본 논문의 핵심 주장은 다음과 같다:

> **"실세계 시계열 데이터는 선형(linear)과 비선형(nonlinear) 패턴을 동시에 포함하며, ARIMA(선형)와 ANN(비선형) 모델을 결합한 하이브리드 접근법이 각 단독 모델보다 우수한 예측 성능을 달성할 수 있다."**

즉, 어떤 단일 모델도 모든 상황에서 최선이 될 수 없으므로, 서로 이질적인 두 모델(ARIMA + ANN)의 강점을 순차적으로 활용하는 것이 효과적이라는 주장이다.

### 주요 기여 (Contributions)

| 기여 항목 | 내용 |
|---|---|
| **하이브리드 프레임워크 제안** | ARIMA로 선형 구조를 먼저 포착하고, 그 잔차(residuals)를 ANN으로 모델링하는 2단계 순차적 결합 구조 최초 제안 |
| **이론적 근거 제시** | 시계열 = 선형 성분 + 비선형 성분으로 분해 가능하다는 이론적 가정 제시 |
| **실증 검증** | 3개의 실제 데이터셋(Sunspot, Lynx, Exchange Rate)에서 하이브리드 모델이 단독 모델 대비 우수함을 실험적으로 입증 |
| **과적합 완화 메커니즘** | ARIMA를 선행 적용함으로써 ANN의 과적합(overfitting) 문제를 간접적으로 완화하는 효과 제시 |

---

## 2. 상세 설명

### 2-1. 해결하고자 하는 문제

논문이 해결하고자 한 핵심 문제는 다음 세 가지다:

**① 모델 선택의 불확실성 (Model Selection Uncertainty)**
- 실제 데이터가 선형인지 비선형인지 사전에 알기 어려움
- 단일 모델 선택은 표본 변동(sampling variation), 모델 불확실성(model uncertainty), 구조 변화(structure change)에 취약

**② 선형/비선형 패턴의 혼재**
- ARIMA: 선형 상관관계만 포착 가능, 비선형 패턴 포착 불가
- ANN: 비선형 패턴에 강하나, 선형 문제에서 일관된 성능 보장 어려움

**③ 단일 모델의 한계**
- 예측 문헌에서 널리 인정되듯, 모든 상황에서 최선인 단일 모델은 없음 (Chatfield, 1988; Makridakis et al., 1982)

---

### 2-2. 제안하는 방법 (수식 포함)

#### ARIMA 모델

ARIMA 모델은 시계열의 미래값을 과거 관측값과 오차항의 선형 함수로 표현한다:

$$y_t = \theta_0 + \phi_1 y_{t-1} + \phi_2 y_{t-2} + \cdots + \phi_p y_{t-p} + \varepsilon_t - \theta_1 \varepsilon_{t-1} - \theta_2 \varepsilon_{t-2} - \cdots - \theta_q \varepsilon_{t-q} \tag{1}$$

- $y_t$: 시점 $t$에서의 실제값
- $\varepsilon_t$: 평균 0, 분산 $\sigma^2$의 백색잡음(white noise)
- $\phi_i$ ($i=1,\ldots,p$): 자기회귀 계수
- $\theta_j$ ($j=0,1,\ldots,q$): 이동평균 계수
- $p, q$: 모델 차수(order)

#### ANN 모델

단층 은닉층(single hidden layer) 피드포워드 신경망:

$$y_t = \alpha_0 + \sum_{j=1}^{q} \alpha_j \, g\!\left(\beta_{0j} + \sum_{i=1}^{p} \beta_{ij} y_{t-i}\right) + \varepsilon_t \tag{2}$$

활성화 함수 (로지스틱 함수):

$$g(x) = \frac{1}{1 + \exp(-x)} \tag{3}$$

이를 비선형 자기회귀 모델로 표현하면:

$$y_t = f(y_{t-1}, y_{t-2}, \ldots, y_{t-p};\, \mathbf{w}) + \varepsilon_t \tag{4}$$

- $\alpha_j$: 은닉층→출력층 연결 가중치
- $\beta_{ij}$: 입력층→은닉층 연결 가중치
- $p$: 입력 노드 수 (지연 관측값 수)
- $q$: 은닉 노드 수
- $\mathbf{w}$: 전체 파라미터 벡터

#### 하이브리드 모델 (핵심)

시계열을 선형 성분과 비선형 성분의 합으로 분해한다:

$$y_t = L_t + N_t \tag{5}$$

**Step 1**: ARIMA로 선형 성분 $L_t$ 모델링 → 잔차 계산

$$e_t = y_t - \hat{L}_t \tag{6}$$

**Step 2**: ANN으로 잔차(비선형 패턴) 모델링

$$e_t = f(e_{t-1}, e_{t-2}, \ldots, e_{t-n}) + \varepsilon_t \tag{7}$$

**Step 3**: 최종 예측값 결합

$$\hat{y}_t = \hat{L}_t + \hat{N}_t \tag{8}$$

이 구조는 ARIMA 잔차에 여전히 비선형 패턴이 존재한다는 전제에 기반하며, ANN이 이를 포착하도록 설계되었다.

---

### 2-3. 모델 구조

```
[원 시계열 데이터 y_t]
        │
        ▼
 ┌─────────────┐
 │  ARIMA 모델  │  ← 선형 패턴 포착 (Box-Jenkins 방법론)
 └─────────────┘
        │
        ▼
   [잔차 e_t = y_t - L̂_t]
        │
        ▼
 ┌─────────────────────────────┐
 │  ANN 모델 (단층 은닉층)      │  ← 비선형 패턴 포착
 │  구조: p×q×1               │
 └─────────────────────────────┘
        │
        ▼
   [ANN 예측 N̂_t]
        │
        ▼
 [최종 하이브리드 예측: ŷ_t = L̂_t + N̂_t]
```

**실험에 사용된 ANN 구조:**

| 데이터셋 | ANN 구조 | ARIMA 모델 |
|---|---|---|
| Sunspot | $4 \times 4 \times 1$ | AR(9) subset |
| Lynx | $7 \times 5 \times 1$ | AR(12) subset |
| Exchange Rate | $7 \times 6 \times 1$ | Random Walk |

---

### 2-4. 성능 향상 결과

#### Sunspot 데이터

| 모델 | MSE (35-step) | MAD (35-step) | MSE (67-step) | MAD (67-step) |
|---|---|---|---|---|
| ARIMA | 216.965 | 11.319 | 306.082 | 13.034 |
| ANN | 205.302 | 10.243 | 351.194 | 13.544 |
| **Hybrid** | **186.827** | **10.831** | **280.160** | **12.780** |

- 35-step MSE 기준: ARIMA 대비 **16.13%** 개선, ANN 대비 **9.89%** 개선

#### Lynx 데이터

| 모델 | MSE | MAD |
|---|---|---|
| ARIMA | 0.020486 | 0.112255 |
| ANN | 0.020466 | 0.112109 |
| **Hybrid** | **0.017233** | **0.103972** |

- MSE 기준: ARIMA 대비 **18.87%**, ANN 대비 **18.76%** 개선

#### Exchange Rate 데이터

| 모델 | MSE (1M) | MAD (1M) | MSE (6M) | MSE (12M) |
|---|---|---|---|---|
| ARIMA | $3.685 \times 10^{-5}$ | 0.005016 | $5.657 \times 10^{-5}$ | $4.530 \times 10^{-5}$ |
| ANN | $2.764 \times 10^{-5}$ | 0.004218 | $5.711 \times 10^{-5}$ | $4.527 \times 10^{-5}$ |
| **Hybrid** | $\mathbf{2.673 \times 10^{-5}}$ | **0.004146** | $\mathbf{5.655 \times 10^{-5}}$ | $\mathbf{4.359 \times 10^{-5}}$ |

- 하이브리드 모델은 3가지 예측 기간 모두에서 일관되게 두 기준 모델을 상회

---

### 2-5. 한계점

본 논문이 명시적·암묵적으로 인정하는 한계는 다음과 같다:

| 한계 항목 | 내용 |
|---|---|
| **핵심 가정의 취약성** | 시계열이 반드시 선형+비선형으로 분해 가능하다는 가정이 항상 성립하지 않을 수 있음 |
| **하이퍼파라미터 선택의 주관성** | ANN의 입력 노드 수 $p$, 은닉 노드 수 $q$ 선택에 체계적 이론 없음; 실험적 탐색에 의존 |
| **ANN 구성요소의 차선 최적성** | ARIMA 잔차가 ANN의 충분한 훈련 데이터를 제공하지 못할 수 있음 |
| **단방향 순차 구조의 제한** | ARIMA → ANN의 단방향 구조이므로, ARIMA 모델 오명세(misspecification)가 ANN 성능에 연쇄적으로 영향 |
| **장기 예측 한계** | Exchange rate 6개월 이상 예측에서 개선 폭이 미미함 |
| **one-step-ahead에 집중** | 실험이 주로 1-step 예측에 한정되어 다중 스텝 예측의 일반화 불분명 |
| **데이터셋 수 제한** | 3개의 데이터셋만으로 일반화 결론 도출의 한계 |

---

## 3. 모델의 일반화 성능 향상 가능성 (심층 분석)

본 논문에서 일반화(generalization) 성능 향상과 관련된 내용은 논문 전반에 걸쳐 다음과 같은 맥락으로 등장한다.

### 3-1. 하이브리드 구조 자체의 일반화 이점

논문은 Granger (1989), Krogh & Vedelsby (1995), Perrone & Cooper (1993)을 인용하여 다음을 주장한다:

> **"이질적(dissimilar) 모델을 결합할수록, 또는 서로 강하게 불일치하는 모델을 결합할수록 하이브리드 모델의 일반화 분산(generalization variance)이 낮아진다."**

이는 앙상블 이론의 편향-분산 트레이드오프(bias-variance tradeoff)와 직결된다. 수식으로 표현하면:

$$\text{MSE}_{\text{ensemble}} = \bar{\varepsilon}^2 + \overline{\text{Var}} - \overline{\text{Cov}}$$

여기서 $\overline{\text{Cov}}$는 개별 모델 예측 간의 공분산이다. **ARIMA와 ANN은 구조적으로 이질적이므로 공분산이 낮아**, 앙상블 효과가 극대화된다.

### 3-2. ARIMA 선행 적용의 과적합 완화 효과

논문 결론부에서 다음을 명시한다:

> "Furthermore, by fitting the ARIMA model first to the data, the **overfitting problem** that is more strongly related to neural network models **can be eased**."

ARIMA가 선형 패턴을 먼저 제거하면, ANN이 학습해야 할 잔차 시계열의 복잡도가 감소한다. 이는 ANN 입력 공간의 차원 축소 효과를 간접적으로 유발하여, ANN의 과적합 경향을 억제하고 테스트 데이터에 대한 일반화 성능을 향상시킨다.

### 3-3. 모델 불확실성 감소

논문은 Chatfield (1996)을 인용하여:

> "using the hybrid method can **reduce the model uncertainty** which typically occurred in statistical inference and time series forecasting"

단일 모델에 지나치게 의존할 때 발생하는 모델 불확실성(model uncertainty)이, 두 이질적 모델의 결합을 통해 분산되어 전체 예측의 불확실성이 감소한다.

### 3-4. 구조 변화(Structure Change)에 대한 강건성

> "the combined model is **more robust** with regard to the possible structure change in the data"

실세계 시계열은 시간에 따라 데이터 생성 메커니즘이 변할 수 있다(non-stationarity, regime change). 하이브리드 모델은 선형 및 비선형 메커니즘을 모두 내포하므로, 구조 변화 시에도 최소한 한 구성요소가 적절히 반응할 가능성이 높다.

### 3-5. 일반화 성능의 조건과 한계

그러나 일반화 성능 향상은 **무조건적이지 않다**. 논문은 암묵적으로 다음 조건을 전제한다:

1. **잔차에 실제로 비선형 패턴이 존재할 것** → 잔차가 순수한 백색잡음이라면 ANN의 추가 기여가 없거나 오히려 과적합 유발
2. **ANN 훈련 데이터로 잔차가 충분할 것** → 소규모 데이터셋에서는 ANN이 잔차를 과적합할 위험 존재
3. **ARIMA 모델이 적절히 명세될 것** → ARIMA 오명세 시 잔차에 선형 패턴이 잔류하여 ANN이 혼란

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4-1. 향후 연구에 미치는 영향

본 논문은 출판 이후 **시계열 예측 하이브리드 모델링의 기반 문헌**으로 광범위하게 인용되었으며, 다음과 같은 연구 흐름을 촉발하였다.

#### ① 다양한 구성요소 모델로의 확장

Zhang (2003)의 ARIMA+ANN 프레임워크는 이후 다음과 같이 확장되었다:
- **ARIMA + LSTM**: LSTM의 장기 의존성 포착 능력을 활용
- **ARIMA + SVR (Support Vector Regression)**: 커널 기반 비선형 모델과의 결합
- **SARIMA + Deep Learning**: 계절성 시계열에 대한 확장
- **EMD (Empirical Mode Decomposition) + 딥러닝**: 신호 분해 후 개별 예측 결합

#### ② 결합 방법론의 정교화

단순한 덧셈 결합($\hat{y}_t = \hat{L}_t + \hat{N}_t$)에서 다음으로 발전:
- **가중 결합**: 각 모델의 예측 신뢰도에 따른 동적 가중치 부여
- **적층 일반화(Stacked Generalization)**: 메타학습기(meta-learner)로 결합 방식 학습
- **어텐션 기반 결합**: 시점별로 선형/비선형 기여도를 동적으로 조정

#### ③ 딥러닝 시대의 하이브리드 패러다임

Zhang (2003)의 정신은 현대의 N-BEATS, Temporal Fusion Transformer(TFT), PatchTST 등 딥러닝 기반 예측 모델에도 계승되었다. 이들 모델은 선형 및 비선형 성분을 **내부적으로 분리**하여 처리하는 구조를 채택하고 있다.

---

### 4-2. 향후 연구 시 고려할 점

#### ① 기본 가정의 검증 필요성

$$y_t = L_t + N_t$$

이 분해 가정이 실제 데이터에서 성립하는지 사전 검증이 필요하다. 비선형성 검정 (예: BDS 검정, Teräsvirta's 비선형성 검정)을 통해 잔차에 비선형 구조가 실제로 존재하는지 확인해야 한다.

#### ② 하이퍼파라미터 최적화 자동화

ANN의 $p$ (입력 노드 수)와 $q$ (은닉 노드 수) 선택이 실험적 탐색에 의존하는 문제를 해결하기 위해:
- **AutoML / NAS (Neural Architecture Search)** 적용
- **베이지안 최적화(Bayesian Optimization)** 활용
- **교차검증(Cross-Validation)** 기반 자동 파라미터 선택

#### ③ 다중 스텝 예측(Multi-step Forecasting)으로의 확장

본 논문은 주로 one-step-ahead 예측에 집중하였으나, 실용적 응용에서는 다중 스텝 예측이 필수적이다. 다중 스텝 예측 시 오차 누적 문제(error accumulation)가 발생하므로, 이를 완화하는 Direct Multi-output 전략이나 Seq2Seq 구조와의 결합을 고려해야 한다.

#### ④ 현대적 비선형 구성요소로의 교체

단순 MLP(ANN) 대신 다음 모델과의 결합을 고려:
- **LSTM / GRU**: 순환 신경망으로 장기 의존성 포착
- **Transformer / Attention**: 전역 패턴 포착
- **TCN (Temporal Convolutional Network)**: 병렬 처리 효율성

#### ⑤ 해석 가능성(Explainability) 확보

ARIMA+ANN 하이브리드 모델에서 ANN 부분은 여전히 블랙박스로 남는다. 금융, 의료, 에너지 분야에서는 예측 근거의 설명이 필수적이므로:
- **SHAP (SHapley Additive exPlanations)** 기반 피처 중요도 분석
- **LIME (Local Interpretable Model-Agnostic Explanations)** 적용
- 선형 성분($\hat{L}_t$)과 비선형 성분($\hat{N}_t$)의 기여도 시각화

#### ⑥ 비정상성(Non-stationarity) 및 구조 변화 처리

실세계 데이터는 체계적 구조 변화(regime shift)를 겪는다. 향후 연구에서:
- **Online Learning / Incremental Learning**으로 모델 동적 업데이트
- **변화점 탐지(Changepoint Detection)**와 결합하여 구조 변화 시 모델 재보정
- **롤링 윈도우(Rolling Window)** 기반 동적 하이브리드 적용

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

Zhang (2003)의 하이브리드 프레임워크 이후 2020년대에 등장한 주요 관련 연구들과의 비교는 다음과 같다.

> **⚠️ 주의**: 아래 최신 연구 비교는 제 학습 데이터(2024년 초)까지의 지식에 기반하며, 구체적 수치는 원 논문을 직접 확인하시기 바랍니다.

| 구분 | Zhang (2003) | N-BEATS (Oreshkin et al., 2020) | Temporal Fusion Transformer (Lim et al., 2021) | PatchTST (Nie et al., 2023) |
|---|---|---|---|---|
| **접근법** | ARIMA + ANN 순차 결합 | 순수 딥러닝 (잔차 분해) | 어텐션 기반 멀티모달 | Transformer + 패치 분할 |
| **선형/비선형 처리** | 명시적 분리 (ARIMA→ANN) | 내재적 분리 (Trend/Seasonality 블록) | 암묵적 (어텐션 메커니즘) | 암묵적 (패치 임베딩) |
| **해석 가능성** | 중간 (선형 부분만 해석 가능) | 중간 | 높음 (어텐션 가중치) | 낮음 |
| **데이터 요구량** | 소규모 가능 | 대규모 필요 | 대규모 필요 | 대규모 필요 |
| **하이퍼파라미터** | 적음 (ANN 구조만) | 많음 | 매우 많음 | 많음 |
| **장기 예측** | 제한적 | 강함 | 강함 | 강함 |
| **핵심 유산** | 하이브리드 아이디어의 기원 | 잔차 기반 분해 구조 계승 | 선형+비선형 통합 발전 | 시계열 자기지도학습 |

### Zhang (2003) vs. 현대 딥러닝 하이브리드 연구의 핵심 차이

**1. ARIMA + LSTM 하이브리드 (2020년대 주요 흐름)**

Zhang 방법의 ANN 부분을 LSTM으로 교체:

$$\hat{y}_t = \hat{L}_t^{\text{ARIMA}} + \hat{N}_t^{\text{LSTM}}$$

LSTM은 MLP 대비 시간적 의존성을 더 효과적으로 포착하며, 특히 긴 잔차 시퀀스에서 유리하다.

**2. N-BEATS (Oreshkin et al., NeurIPS 2020)**

Zhang (2003)의 "선형+비선형 분리" 아이디어를 딥러닝 내부로 내재화:

$$\hat{y} = \sum_{k} \hat{y}_k^{\text{block}}, \quad \hat{y}_k = \text{Trend block} + \text{Seasonality block}$$

순수 딥러닝이지만, 각 블록이 해석 가능한 성분(추세, 계절성)을 명시적으로 분리하는 구조는 Zhang의 철학적 후계자로 볼 수 있다.

**3. Temporal Fusion Transformer (Lim et al., IJF 2021)**

어텐션 메커니즘을 통해 시점별로 선형/비선형 특성의 기여도를 동적으로 가중:

$$\hat{y}_t = \text{GRN}\!\left(\text{Static context} + \text{Temporal self-attention}\right)$$

Zhang (2003) 대비 다변량(multivariate) 시계열 처리, 공변량(covariate) 통합, 불확실성 정량화가 가능하다는 점에서 실용성이 크게 향상되었다.

---

## 참고문헌 (References)

**주 논문 (본 분석의 직접 출처):**
- Zhang, G.P. (2003). **Time series forecasting using a hybrid ARIMA and neural network model**. *Neurocomputing*, 50, 159–175.

**논문 내 인용 문헌 (주요):**
- Box, G.E.P., & Jenkins, G. (1970). *Time Series Analysis, Forecasting and Control*. Holden-Day.
- Granger, C.W.J. (1989). Combining forecasts—Twenty years later. *Journal of Forecasting*, 8, 167–173.
- Krogh, A., & Vedelsby, J. (1995). Neural network ensembles, cross validation, and active learning. *Advances in Neural Information Processing*, 7, 231–238.
- Hornik, K., Stinchcombe, M., & White, H. (1990). Using multi-layer feedforward networks for universal approximation. *Neural Networks*, 3, 551–560.
- Makridakis, S. et al. (1982). The accuracy of extrapolation (time series) methods: results of a forecasting competition. *Journal of Forecasting*, 1, 111–153.
- Chatfield, C. (1996). Model uncertainty and forecast accuracy. *Journal of Forecasting*, 15, 495–508.
- Perrone, M.P., & Cooper, L. (1993). When networks disagree: ensemble method for hybrid neural networks. In *Neural Networks for Speech and Image Processing*. Chapman & Hall.

**2020년 이후 비교 연구 (학습 데이터 기반, 원문 확인 권장):**
- Oreshkin, B.N., et al. (2020). **N-BEATS: Neural basis expansion analysis for interpretable time series forecasting**. *ICLR 2020*.
- Lim, B., et al. (2021). **Temporal Fusion Transformers for interpretable multi-horizon time series forecasting**. *International Journal of Forecasting*, 37(4), 1748–1764.
- Nie, Y., et al. (2023). **A Time Series is Worth 64 Words: Long-term Forecasting with Transformers**. *ICLR 2023* (PatchTST).
