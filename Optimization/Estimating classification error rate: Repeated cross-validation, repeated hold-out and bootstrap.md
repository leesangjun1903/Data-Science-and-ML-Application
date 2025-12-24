# Estimating classification error rate: Repeated cross-validation, repeated hold-out and bootstrap

### 1. 핵심 주장 및 주요 기여 요약

**핵심 주장**:

이 논문은 분류 모델의 참 오류율을 추정할 때 공정한 계산량 기준 하에서 반복 교차 검증(repeated cross-validation)과 .632+ 부트스트랩 추정자를 비교합니다. 논문의 핵심 발견은 **분류 알고리즘의 적응성(adaptivity)에 따라 최적 추정 방법이 달라진다**는 것입니다.[1]

**주요 기여**:

1. **반복 교차 검증의 효과성 입증**: 5회 반복 10-fold CV가 비반복 CV에 비해 표준편차를 1.08배(의사결정 나무)와 1.21배(부스팅)로 감소시킴을 시뮬레이션으로 증명

2. **.632+ 부트스트랩의 한계 규명**: 적응적 분류기(boosting)에서 .632+ 부트스트랩이 소표본뿐만 아니라 **대규모 표본(n=1000)에서도 심각한 음의 편향을 나타냄**을 처음 체계적으로 보고

3. **계산량 공정성 보장**: 모든 추정자의 모델 적합 횟수를 약 50회로 통제하여 진정한 비교 가능성 확보

***

### 2. 상세한 기술 설명: 문제, 방법, 모델, 한계

#### 2.1 해결하고자 하는 문제

**문제의 정의**: 훈련 표본 $D_n = \{(x_1, y_1), ..., (x_n, y_n)\}$으로 구성한 분류기 $r_n(x) = r(D_n, x)$의 **참 조건부 오류율**을 독립적인 테스트 표본이 없는 상황에서 추정하기

$$\epsilon_n = E[I(Y \neq r_n(X))|D_n]$$

여기서 기댓값은 미래의 관측치 $(X, Y)$의 분포를 따릅니다.[1]

**기존 방법의 한계**:
- **재대입(Resubstitution)**: 편향적으로 과소추정 (훈련 데이터로 테스트)
- **교차검증(CV)**: 거의 불편이지만 높은 분산
- **부트스트랩**: 소표본에서는 낮은 분산이지만, 높은 계산 비용

#### 2.2 제안하는 방법 및 수식

**4가지 주요 추정자**:

**(1) 재대입 추정자** (편향 기준선):

$$\hat{\epsilon}_{resub} = \frac{1}{n}\sum_{i=1}^{n} I(y_i \neq r_n(x_i))$$

**(2) k-fold 교차검증 추정자**:

$$\hat{\epsilon}_{cv10} = \frac{1}{n}\sum_{k=1}^{K}\sum_{i \in I_k} I(y_i \neq r_{(-k)}(x_i))$$

여기서 $r_{(-k)}$는 k번째 폴드를 제외한 데이터로 구성한 분류기입니다.[1]

**(3) 반복 교차검증 추정자**:

$$\hat{\epsilon}_{rcv} = \frac{1}{R}\sum_{r=1}^{R} \hat{\epsilon}_{cv10}^{(r)}$$

논문에서는 R=5 반복으로 설정하여 모델 적합 횟수를 50회로 통제합니다.[1]

**(4) .632 부트스트랩 추정자** (Efron, 1983):

$$\hat{\epsilon}_{b632} = 0.368\hat{\epsilon}_{resub} + 0.632\hat{\epsilon}_{b0}$$

여기서 $\hat{\epsilon}_{b0}$는 부트스트랩 표본에서 훈련되지 않은 관측치(out-of-bag)에 대한 오류율입니다.[1]

**(5) .632+ 부트스트랩 추정자** (Efron & Tibshirani, 1997) - **편향 보정 버전**:

$$\hat{\epsilon}_{b632+} = (1-\hat{w})\hat{\epsilon}_{resub} + \hat{w}\hat{\epsilon}_{b0}$$

여기서 적응적 가중치는:
$$\hat{w} = \frac{0.632}{1-0.368\hat{R}}$$

그리고:

$$\hat{R} = \frac{\hat{\epsilon}_{b0} - \hat{\epsilon}_{resub}}{\hat{\gamma} - \hat{\epsilon}_{resub}}$$

$\hat{\gamma}$는 정보가 없을 때의 오류율 (이진 분류에서):

$$\hat{\gamma} = \hat{p}_1(1-\hat{q}_1) + (1-\hat{p}_1)\hat{q}_1$$

여기서 $\hat{p}_1 = \text{observed } P(y=1)$, $\hat{q}_1 = \text{observed } P(\hat{y}=1)$.[1]

**성능 지표 (균형잡힌 분석)**:

각 추정자의 성능을 정량화하기 위해 기댓값 평방 오차를 분산과 편향으로 분해:

$$E[(\hat{\epsilon} - \epsilon_n)^2] = \text{Var}(\hat{\epsilon} - \epsilon_n) + [E(\hat{\epsilon} - \epsilon_n)]^2$$

$$E[(\hat{\epsilon} - \epsilon_n)^2] = \text{RMS}^2 \text{ (Root Mean Squared Error)}$$

여기서:
- 분산 성분: $\text{Var}(\hat{\epsilon} - \epsilon_n)$ (추정자의 변동성)
- 편향 성분: $[E(\hat{\epsilon} - \epsilon_n)]^2$ (체계적 오류)[1]

#### 2.3 모델 구조 및 시뮬레이션 설계

**두 가지 분류 알고리즘의 대비**:

| 특성 | 의사결정 나무 (Pruned Tree) | 부스팅 (Boosting) |
|------|------------------------|-----------------|
| **적응성** | 중간 (relatively smooth) | 높음 (highly adaptive) |
| **재대입 오류율** | 적당함 | 거의 0에 가까움 |
| **부트스트랩의 문제** | 약함 | 심각함 |
| **최적 추정자** | 소표본: .632+ | 모든 표본: 반복 CV |

**Simulation 1의 데이터 생성**:

$$P(Y=1|X) = \frac{1}{1+\exp(-F(x))}$$

$$F(x) = -10 + 10\sin(\pi x_1 x_2) + 5(x_3-1/2)^2 + 5x_4 + 2x_5$$

- 처음 4개: $U(0,1)$, 5번째: 이산 균등 분포 {1,2,3}
- 5개 추가 노이즈 변수로 분류 난이도 증가
- 표본 크기: n = 40 ~ 1000[1]

**Simulation 2의 데이터 생성**:

$$Y = \begin{cases} \text{Bernoulli}(p) & \text{if } X_1 X_2 > 0 \\ \text{Bernoulli}(1-p) & \text{if } X_1 X_2 < 0 \end{cases}$$

- 신호 강도 변화: p = 0.5 (무관) ~ 1.0 (완전 결정적)
- 학습 곡선 기울기에 따른 성능 변화 분석[1]

#### 2.4 성능 향상 방법 및 결과

**실험 결과 - Simulation 1**:

**의사결정 나무 (소표본, n=40-140)**:
- .632+ 부트스트랩: RMS 측도에서 최고 성능
- 이유: 낮은 표준편차가 약간의 음의 편향을 보상
- 시사점: 비적응적 알고리즘에서는 부트스트랩이 경쟁력 있음[1]

**부스팅 (중표본, n=40-300)**:
- 반복 CV ($\hat{\epsilon}_{rcv}$): 최고 성능
- .632+ 부트스트랩의 심각한 음의 편향 발현
- 이유: 재대입 오류율이 ~0이므로 $\hat{\epsilon}\_{b0} - \hat{\epsilon}_{resub}$이 커지고, 부트스트랩이 의존하는 재대입 추정자의 극단값이 문제[1]

**부스팅 (대표본, n=300-1000)**:
- 반복 CV가 지속적으로 우수 (RMS 기준)
- .632+ 부트스트랩의 편향: 소표본뿐만 아니라 대표본에서도 심각
- 표준편차 감소: 표본 크기 증가에도 편향이 상쇄되지 않음[1]

**시뮬레이션 1 결과 시각화** (Figures 3-6):

$$\sqrt{\frac{\text{Var}(\hat{\epsilon}_{cv10} - \epsilon_n)}{\text{Var}(\hat{\epsilon}_{rcv} - \epsilon_n)}} = 1.08 \text{ (나무)}, 1.21 \text{ (부스팅)}$$

반복이 분산을 효과적으로 감소시킴을 입증.[1]

#### 2.5 한계 및 이론적 근거 부족

**이론적 한계**:

1. **분류 문제의 이론 미흡**: 회귀 문제에 대한 반복 CV의 대표본 성질만 알려져 있음 (Burman, 1989). 분류 문제에 대한 정확한 이론은 부재[1]

2. **부트스트랩 이론의 약점**: Efron (1983)도 지적하듯이 부트스트랩의 이론적 근거가 약함. 특히 극단적 적응성 조건에서의 성능 분석 부족[1]

3. **조건부 vs 무조건부 오류율**: 논문은 조건부 오류율 $\epsilon_n = E[I(Y \neq r_n(X))|D_n]$에 초점. 모델 선택을 위한 무조건부 오류율 $E(\epsilon_n)$과의 차이 미논의[1]

**실험 설계 한계**:

1. **데이터 구조의 제한성**: 두 가지 인공 데이터만 사용. 다양한 실제 데이터셋 부족

2. **표본 크기 범위**: 40~1000 범위로 초대규모 데이터 미검토

3. **알고리즘 종류**: 부스팅과 의사결정 나무만 비교. 신경망, SVM 등 다른 적응적 알고리즘 미포함[1]

***

### 3. 모델의 일반화 성능 향상 가능성 (중점)

#### 3.1 재대입 편향의 메커니즘과 해결책

**문제**:

$$\text{Bias}(\hat{\epsilon}_{resub}) = E[\hat{\epsilon}_{resub}] - \epsilon_n < 0$$

재대입 추정자는 훈련 데이터로 테스트하므로 **항상 과도하게 낙관적**입니다. 이 편향은 적응적 알고리즘에서 극심함.[1]

**반복 교차검증의 해결책**:

$$E[\hat{\epsilon}_{rcv}] \approx \epsilon_n \quad \text{(거의 불편)}$$

각 폴드에서 훈련 데이터와 테스트 데이터가 분리되므로 독립성 보장. 반복을 통해 폴드 분할의 무작위성으로 인한 분산 감소.[1]

$$\text{SE}(\hat{\epsilon}_{rcv}) = \sqrt{\frac{\text{Var}(\hat{\epsilon}_{rcv} - \epsilon_n)}{1}} < \text{SE}(\hat{\epsilon}_{cv10})$$

#### 3.2 부트스트랩의 구조적 문제

**.632 부트스트랩의 기초**:

각 부트스트랩 표본은 크기 n이지만 약 **63.2% 만큼의 고유 관측치**만 포함 (반복 표본화). 이로 인해 out-of-bag 오류율이 과대평가되는 경향.[1]

$$P(\text{observation in bootstrap sample}) = 1 - (1-1/n)^n \approx 0.632$$

**.632 부트스트랩의 수정 메커니즘**:

$$\hat{\epsilon}_{b632} = 0.368 \cdot \hat{\epsilon}_{resub} + 0.632 \cdot \hat{\epsilon}_{b0}$$

가중치 0.632는 재대입 편향과 부트스트랩 과대추정 사이의 절충.[1]

**.632+ 부트스트랩의 적응적 가중치**:

$$\hat{w} = \frac{0.632}{1-0.368\hat{R}}, \quad \hat{R} = \frac{\hat{\epsilon}_{b0} - \hat{\epsilon}_{resub}}{\hat{\gamma} - \hat{\epsilon}_{resub}}$$

**문제점**:
- $\hat{\epsilon}_{resub} \approx 0$ (부스팅)일 때: $\hat{R}$ 분모가 매우 커지거나 작아짐
- 극단적 과적합 상황에서 가중치 조정이 충분하지 못함
- **논문의 발견**: .632+ 조정도 적응적 알고리즘에서 음의 편향 제거 실패[1]

#### 3.3 일반화 성능 향상의 실제 지침

**시뮬레이션 기반 권장사항**:

| 상황 | 추천 방법 | 이유 |
|------|---------|------|
| **비적응적 분류기 + 소표본** | .632+ 부트스트랩 | 낮은 분산이 중요; 편향 무시할 수 있음 |
| **적응적 분류기 (부스팅, 1-NN) + 모든 표본** | 반복 교차검증 | 편향 문제가 더 심각; 안정적 성능 |
| **일반적 권장** | 반복 10-fold CV | 거의 모든 경우에서 안정적 성능 제공[1] |

**반복 횟수의 최적화**:

계산 비용 제약 하에서:
$$\text{Model Fits} = K \times R = 10 \times 5 = 50$$

이는 부트스트랩과 비교 가능하면서도 더 안정적. 구체적 권장값:[1]
- K=10 (전통적 선택)
- R=5 (분산 감소의 실질적 효과)

***

### 4. 논문의 영향 및 앞으로의 연구 방향

#### 4.1 분류 모델 평가 분야에 미친 영향

**직접적 기여**:

1. **부트스트랩 방법의 재평가**: 2000년대 초반까지 부트스트랩이 교차검증보다 우수하다는 일반적 통념 도전[1]

2. **계산량 공정성 강조**: 이후 연구들이 모든 비교에서 동일 계산 비용을 기준으로 설정하는 관례 정착

3. **적응적 알고리즘의 특수성 인식**: 부스팅, SVM 등 현대적 분류기에 특화된 평가 방법의 필요성 제기[1]

#### 4.2 2020년 이후 관련 최신 연구 비교 분석

##### (1) Bates, Hastie, Tibshirani (2021) - 교차검증이 실제로 무엇을 추정하는가?

**핵심 발견**:

$$E[\hat{\epsilon}^{CV}] \approx Err_n = E[Err(D_n)]$$

교차검증은 **특정 훈련 데이터에 대한 오류**가 아니라 **동일 크기의 다양한 훈련 데이터에서의 평균 오류**를 추정.[2]

**공식화**:

- $Err_{XY}$: 당신의 특정 데이터 $(X,Y)$로 훈련한 모델의 오류 (원래 목표)
- $Err_X = E[Err_{XY}|X]$: 반응 Y에 대한 평균 (중간값)
- $Err_n = E[Err_{XY}]$: 모든 가능한 훈련 데이터에 대한 평균 (CV가 추정)

$$\text{MSE}(\hat{\epsilon}^{CV}, Err_{XY}) > \text{MSE}(\hat{\epsilon}^{CV}, Err_n)$$

**선형 모델에서 증명** (조건부 독립성):

$$\hat{\epsilon}^{CV} \perp\!\!\perp Err_{XY} \mid X$$

이는 교차검증이 실제 관심 대상(특정 모델의 오류)과 약하게 연관됨을 의미.[2]

**시사점 (Kim 2009과의 연결)**:

- Kim 논문의 "참 오류율" $\epsilon_n$이 실제로는 특정 데이터에 대한 오류가 아니라 평균 오류임을 이론적으로 밝힘
- 반복 CV의 우수성은 더 "정확한" 평균 오류 추정의 결과[2]

##### (2) Cai et al. (2025) - 교차검증 추정의 불확실성 정량화

**문제 인식**:

교차검증은 점 추정자일 뿐, 신뢰 구간이 필요하지만 표준 방법이 부정확함:

$$\text{Var}(\bar{e}) = \frac{a_1}{n} + \frac{n/K-1}{n}a_2 + \frac{n-n/K}{n}a_3 > \frac{a_1}{n}$$

폴드 간 오류 상관으로 인해 표준 분산 추정치가 과소평가.[2]

**제안 방법 (Bootstrap Cross-Validation)**:

효율적 부트스트랩-교차검증 결합:

**Algorithm 2**: 외부 부트스트랩 루프 ($B_{BOOT} \approx 400$) + 내부 CV 루프 ($B_{CV} \approx 10-20$)

1. 부트스트랩 표본 생성 (가중치 $W_i$ 적용)
2. 조정된 훈련 크기 $m_{adj}$ 선택: 63% 고유 관측치 문제 보정

$$m_{adj} = \arg\min_{m'} \left[ \left(\frac{m'}{m/0.632}-1\right)^2 + 0.368\left(\frac{n-m'}{n-m}-1\right)^2 \right]$$

3. 각 부트스트랩에 대해 여러 번 CV 실행
4. **무작위 효과 모델**: 부트스트랩 간 분산($\hat{\sigma}_{BT}^2$)과 내부 몬테카를로 분산($\hat{\tau}_0^2$) 분리

$$\theta_{bk} = \theta_0 + \epsilon_b^* + \epsilon_{bk}$$

여기서 $\epsilon_b^* \sim N(0, \sigma_{BT}^2)$, $\epsilon_{bk}$ ~ $N(0, \tau_0^2)$[2]

5. 최종 신뢰 구간:

$$\hat{\epsilon}^{CV} \pm 1.96 \cdot \frac{\hat{\sigma}_{BT}}{\sqrt{n}}$$

**계산 효율성**:
- 순진한 부트스트랩: 80,000+ 모델 적합 필요
- Algorithm 2: ~8,000 적합만 필요 (90% 감소)[2]

**적용 사례**:
- 정밀의학 (개별화 치료 효과): PEACE 임상시험
- 이진 예측 (c-index/AUC): 다중 관계자 데이터

**성능**:

| 설정 | 명목 커버리지 | 달성된 커버리지 |
|------|-----------|-----------|
| 저차원 로지스틱 (p=20) | 90% | **92-97%** |
| 고차원 희소 (p=1000) | 90% | **87-94%** |
| OLS (소표본) | 90% | **90-105%** |

#### 4.3 앞으로의 연구 시 고려할 점

##### (1) 이론과 실무의 격차

**남은 과제**:

- 부트스트랩과 교차검증의 대역폭 차이를 좁힐 방법 개발
- 극도로 적응적인 알고리즘(신경망, 깊은 학습)에 특화된 추정 방법
- 고차원 데이터(p >> n) 환경에서의 성능 분석[2][1]

##### (2) 모델 선택과 오류 추정의 분리

**중요한 통찰**:

최근 연구(Lei 2019, Wager 2020)는 모델 간 비교가 절대적 오류 추정보다 통계적으로 더 쉬움을 보임.

**권장사항**:
- 하이퍼파라미터 선택: 표준 CV로도 충분
- 최종 성능 보고: 중첩 교차검증 (Nested CV) 사용[2]

##### (3) 적응적 가중치의 개선

Kim 논문 이후 .632+ 부트스트랩의 가중치 개선 시도들:

**미해결 문제**:
- 적응적 알고리즘 특성을 직접 반영한 가중치 함수 설계
- $\hat{R}$이 극단값을 가질 때의 안정화 방법

**제안** (향후 연구):

$$\hat{w}_{robust} = \frac{0.632}{1-0.368 \cdot \text{clip}(\hat{R}, \epsilon, 1-\epsilon)}$$

여기서 clipping은 극단값 방지.[1]

##### (4) 계산 효율과 통계 정확성의 균형

**Cai et al. (2025)의 기여로도** 여전한 과제:

- $B_{BOOT}$과 $B_{CV}$의 최적 배분 이론
- 작은 데이터셋에서 몬테카를로 오차의 영향
- 병렬화를 넘어선 기하급수적 가속 방법[2]

##### (5) 비교차선적 손실 함수

**Kim 논문의 제한**:

이진 분류 오류율만 고려. 최근 요구사항:

- 확률적 예측: cross-entropy 손실
- 불균형 데이터: AUC-ROC, precision-recall
- 비용-민감 분류: 가중 오류율
- 다중 클래스: 일반화된 성능 지표[2]

각 손실 함수에 대한 최적 추정 방법은 상이할 가능성.

***

### 5. 2020년 이후 관련 연구 비교 분석 종합표

| 연구 | 출판년 | 초점 | 주요 발견 | Kim(2009)과의 관계 |
|------|-------|------|---------|------------------|
| **Kim** | 2009 | CV vs 부트스트랩 (실증) | 반복 CV > .632+ (적응) | 기준 연구 |
| **Bates et al.** | 2021 | CV의 추정 대상 규명 (이론) | CV는 평균 오류 추정, 신뢰구간 문제 | Kim의 $\epsilon_n$이 평균 오류임을 증명 |
| **Cai et al.** | 2025 | CV 신뢰 구간 (방법론) | 효율적 부트스트랩-CV로 정확한 CI | 부트스트랩-CV 결합으로 Kim의 한계 극복 |

***

## 최종 결론 및 실무 권장사항

### 핵심 메시지

1. **Kim (2009)**: 반복 교차검증은 적응적 분류 알고리즘에서 부트스트랩보다 우수한 추정자

2. **Bates et al. (2021)**: 교차검증이 추정하는 것은 특정 데이터의 오류가 아니라 **평균 오류**이며, 신뢰 구간에 특별한 주의 필요

3. **Cai et al. (2025)**: 효율적인 부트스트랩-교차검증 결합으로 신뢰 구간을 정확하게 구성 가능

### 실무 적용

**분류 모델 평가 체크리스트**:

- ✅ 기본: 10-fold 교차검증 (계산 가능하면 반복 5회)
- ✅ 신뢰 구간 필요: 중첩 교차검증 (Bates 방법) 또는 부트스트랩-CV (Cai 방법)
- ✅ 극도로 제한된 데이터: .632+ 부트스트랩도 고려
- ✅ 부스팅, SVM 등 적응적 알고리즘: **반복 CV 필수**

***

## 인용 목록

 Kim, J.-H. (2009). Estimating classification error rate: Repeated cross-validation, repeated hold-out and bootstrap. *Computational Statistics and Data Analysis*, 53(10), 3735–3745. https://doi.org/10.1016/j.csda.2009.04.009[1]

 Bates, S., Hastie, T., & Tibshirani, R. (2021). Cross-validation: What does it estimate and how well does it do it? *arXiv preprint arXiv:2104.00673*. [Apr 2021, published Apr 2021][2]

 Cai, B., Luo, Y., Guo, X., Pellegrini, F., Pang, M., de Moor, C., Shen, C., Charu, V., & Tian, L. (2025). Bootstrapping the cross-validation estimate. *arXiv preprint arXiv:2307.00260v2*. [Published 2025][3]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/663b9037-353f-42e4-afb3-df8d4475f169/kim2009.pdf)
[2](https://lex-localis.org/index.php/LexLocalis/article/view/801047)
[3](https://www.mdpi.com/2079-9292/14/15/2953)
[4](https://arxiv.org/abs/2509.19027)
[5](https://onlinelibrary.wiley.com/doi/10.1155/int/9953223)
[6](https://www.mdpi.com/2072-4292/12/19/3148)
[7](https://www.semanticscholar.org/paper/f7fc98ae45b90043ec55fca305031c9299663ed7)
[8](http://www.tandfonline.com/doi/abs/10.1080/01621459.1983.10477973)
[9](https://journals.sagepub.com/doi/10.1177/0962280216656246)
[10](https://www.semanticscholar.org/paper/394a175c1c2e35e70fc7e20ff8bb062d802818ea)
[11](http://ieeexplore.ieee.org/document/75516/)
[12](https://arxiv.org/pdf/2307.00260.pdf)
[13](https://gjeta.com/sites/default/files/GJETA-2021-0068.pdf)
[14](http://arxiv.org/pdf/1708.07180.pdf)
[15](https://figshare.com/articles/journal_contribution/Cross-validation_what_does_it_estimate_and_how_well_does_it_do_it_/22512229/1/files/39974221.pdf)
[16](https://arxiv.org/pdf/2310.10740.pdf)
[17](https://academic.oup.com/aje/article-pdf/180/3/318/17342247/kwu140.pdf)
[18](https://arxiv.org/pdf/2104.00673v2.pdf)
[19](https://arxiv.org/pdf/1801.02817.pdf)
[20](https://hastie.su.domains/MOOC-Slides/cv_boot.pdf)
[21](https://arxiv.org/html/2407.02754v1)
[22](https://pmc.ncbi.nlm.nih.gov/articles/PMC11610218/)
[23](https://arxiv.org/html/2307.00260v2)
[24](https://www.sciencepublishinggroup.com/article/10.11648/j.ajtas.20241305.13)
[25](https://pmc.ncbi.nlm.nih.gov/articles/PMC12582444/)
[26](https://www.sciencedirect.com/science/article/pii/S2215016124004643)
[27](https://pmc.ncbi.nlm.nih.gov/articles/PMC9964614/)
[28](https://www.ijcai.org/Proceedings/95-2/Papers/016.pdf)
[29](https://arxiv.org/html/2510.05942v1)
[30](https://arxiv.org/pdf/2510.14669.pdf)
[31](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0232076)
[32](https://arxiv.org/html/2509.26542v1)
[33](https://arxiv.org/html/2508.12213v1)
[34](https://arxiv.org/html/2412.02108v1)
[35](https://arxiv.org/pdf/2506.22946.pdf)
[36](https://arxiv.org/html/2511.01409v2)
[37](https://arxiv.org/html/2511.01468v1)
[38](https://arxiv.org/pdf/2510.05942.pdf)
