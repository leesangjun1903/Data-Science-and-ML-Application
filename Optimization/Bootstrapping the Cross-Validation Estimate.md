# Bootstrapping the Cross-Validation Estimate

### 1. 핵심 주장 및 주요 기여

**"Bootstrapping the Cross-Validation Estimate"**는 예측 모델의 성능 평가에서 가장 널리 사용되는 교차 검증(Cross-Validation, CV)의 불확실성 정량화 문제를 해결하는 논문입니다.[1]

**논문의 핵심 주장**:
- CV 추정치는 데이터에 의존하는 확률 변수이므로 신뢰도를 정량화해야 함[1]
- 기존 부트스트랩 방법은 계산 비용이 과도함 (모델 훈련 80,000회 이상 필요)[1]
- Random effects model을 통한 분산 성분 추정으로 계산 부담을 10배 이상 감소시킬 수 있음[1]

**주요 기여**:
1. **일반적 프레임워크**: 정밀 의학, C-지수 등 다양한 성능 지표에 적용 가능[1]
2. **계산 효율화**: Algorithm 2를 통해 B_BOOT × B_CV = 8,000 수준으로 감소[1]
3. **이중 보정**: Algorithm 3을 통해 유한 표본에서도 95% 신뢰도 달성[1]

***

### 2. 해결 문제와 제안 방법

#### 2.1 문제의 본질

Cross-validation은 다음의 일반적 형태를 가집니다:

$$\widehat{Err}_{CV_m} = \frac{1}{B_{CV}} \sum_{b=1}^{B_{CV}} L(D_b^{test}, \hat{\psi}(D_b^{train}))$$

여기서:
- $L(\cdot)$: 성능 지표 함수 (예: 평균 절대 오차, C-지수)
- $D_b^{test}$, $D_b^{train}$: $b$번째 분할의 검증, 훈련 집합
- $\hat{\psi}$: 훈련된 모델 파라미터

**핵심 문제**:
1. **Estimand 모호성**: CV는 정확히 어떤 모집단 모수를 추정하는가?[1]
2. **분산 과소추정**: CV 분할 간 상관성으로 인해 표준 오차 추정이 편향됨[1]
3. **계산 부담**: 정확한 분산 추정을 위해 대량의 부트스트랩 반복 필요[1]

#### 2.2 Estimand의 명확화

논문은 두 가지 주요 estimand를 정의합니다:[1]

**1) Err(D_n)**: 특정 데이터셋에서 훈련된 모델의 성능
$$Err(D_n) = \lim_{N \to \infty} L(\tilde{D}_N, \hat{\psi}(D_n))$$

**2) Err_m**: 크기 m의 훈련 집합에 대한 평균 성능 (논문의 선택)
$$Err_m = E\{Err(D_n)\}$$

**핵심 관찰**: CV 추정치 $\widehat{Err}_{CV_m}$은 $Err_m$을 추정하며, $Err(D_n)$과는 근사적으로 독립적입니다. 즉:[1]

$$\widehat{Err}_{CV_m} - Err(D_n) = (Err_m - Err_n) + (\epsilon - \zeta)$$

여기서 $\epsilon$, $\zeta$는 독립적인 평균 0 잡음입니다.[1]

#### 2.3 점근 정규성과 신뢰 구간

충분한 정규성 조건 하에서:[1]

$$\sqrt{n}(\widehat{Err}_{CV_m} - Err_m) \xrightarrow{d} N(0, \sigma_0^2)$$

표준 신뢰 구간:

$$\left[\widehat{Err}_{CV_m} - 1.96\frac{\hat{\sigma}}{\sqrt{n}}, \widehat{Err}_{CV_m} + 1.96\frac{\hat{\sigma}}{\sqrt{n}}\right]$$

#### 2.4 Algorithm 2: 혁신적 계산 방법

**기존 부트스트랩의 문제**:[1]
- 중복 관찰: 부트스트랩 표본에서 동일 관찰이 훈련과 검증에 모두 포함 → 낙관적 편향
- 효과적 표본 크기 감소: 부트스트랩된 훈련 집합에서 서로 다른 관찰은 평균 0.632m개
- 계산 비용: B_BOOT × B_CV ≥ 80,000 모델 훈련

**핵심 해법**: 가중 데이터와 random effects model[1]

**Step 1) Sample Size 조정**:
$$\min_{m_{adj}} \left(\frac{m_{adj}}{m/0.632} - 1\right)^2 + \lambda_0\left(\frac{n-m}{n-m_{adj}} - 1\right)^2$$

여기서 $\lambda_0 = 0.368 = 1 - 0.632$로 설정. 이는 다음을 균형잡습니다:[1]
- 부트스트랩된 훈련 집합의 효과적 크기를 원래의 m에 가깝게 유지
- 검증 집합 크기의 상대적 변화 최소화

**Step 2) Weighted Cross-Validation**:

부트스트랩된 데이터를 가중 표본으로 처리:

```math
\theta_{bk}^* = L(D_{b,test}^*, \hat{\psi}(D_{b,train}^*))
```

여기서 $D_{b,test}^*$는 원래의 $D_{test}$에 부트스트랩 가중치 $W_i$를 적용한 것.[1]

**Step 3) Random Effects Model**:

$$\theta_{bk}^* = \theta_0 + \epsilon_b^* + \epsilon_{bk}$$

$$\epsilon_b^* \sim N(0, \sigma_{BT}^2), \quad \epsilon_{bk} \sim \text{i.i.d. mean-zero noise}$$

- Between-bootstrap 분산: $\sigma_{BT}^2$ (우리가 추정할 목표)
- Within-bootstrap 분산: $\tau_0^2$ (random divisions으로 인한 변동성)

**분산 성분 추정**:[1]

$$\hat{\sigma}_{BT}^2 = \frac{1}{B_{BOOT}-1}\sum_{b=1}^{B_{BOOT}}(\bar{\theta}_b^* - \bar{\theta}^*)^2 - \frac{\hat{\tau}_0^2}{B_{CV}}$$

여기서:

```math
\bar{\theta}_b^* = \frac{1}{B_{CV}}\sum_{k=1}^{B_{CV}}\theta_{bk}^*, \quad \hat{\tau}_0^2 = \frac{1}{(B_{CV}-1)B_{BOOT}}\sum_{b,k}(\theta_{bk}^* - \bar{\theta}_b^*)^2
```

**계산 효율성**: Moderate $B_{CV}$ (10-20)과 $B_{BOOT}$ (400)으로도 충분[1]
$$\text{총 훈련 횟수} = 8,000 \text{ (vs. } 80,000\text{)}$$

#### 2.5 실제 신뢰 구간 구성

**Algorithm 2 기반 신뢰 구간**:[1]

$$\left[\widehat{Err}_{CV_m} - 1.96 \times \hat{\sigma}_{CV_m}, \widehat{Err}_{CV_m} + 1.96 \times \hat{\sigma}_{CV_m}\right]$$

**Algorithm 3: 소수 반복에 대한 보정**[1]

B_BOOT와 B_CV가 매우 작을 때, Monte Carlo 오차로 인한 분포 왜곡 보정:

$$Z_l^* = Z_l \cdot \frac{\hat{\sigma}_{BT}}{\hat{\sigma}_{l,BT}^*}, \quad l=1,\ldots,L$$

여기서 $Z_l \sim N(0,1)$이고 $\hat{\sigma}_{l,BT}^*$는 재부트스트랩된 분산 추정.[1]

최종 신뢰 구간:

$$\left[\widehat{Err}_{CV_m} - c_{1-\alpha/2} \times \hat{\sigma}_{CV_m}, \widehat{Err}_{CV_m} + c_{1-\alpha/2} \times \hat{\sigma}_{CV_m}\right]$$

***

### 3. 모델 구조와 응용 사례

#### 3.1 Application 1: Precision Medicine Strategy Evaluation[1]

**배경 (PEACE Trial)**:
8,290명의 관상동맥 질환 환자를 ACE 억제제 또는 위약에 무작위 배정. 전체 모집단에서는 유의하지 않지만(HR=0.92, p=0.30), 특정 부분군에서는 치료 효과가 있을 수 있음.[1]

**Individualized Treatment Response (ITR) 점수 추정**:[1]

$$\min_{\gamma, \beta} \frac{1}{m}\sum_{X_i \in D_{train}} \left[Y_i - \gamma' \tilde{Z}_i - (G_i-\pi)\beta'\tilde{Z}_i\right]^2$$

- $Y_i$: 결과 변수 (제한된 평균 생존 시간)
- $\tilde{Z}_i = (1, Z_i')$: 절편 포함 기저 공변량
- $G_i \in \{0,1\}$: 치료 배정 지시자 ($G_i \perp Z_i$, 무작위화)
- $\pi = \Pr(G_i=1)$

**ITR 점수**:[1]
$$\hat{\Delta}(z|D_{train}) = \hat{\beta}(D_{train})'\tilde{z}$$

이는 조건부 평균 치료 효과(CATE)를 근사합니다.

**성능 지표 (추천 부분군에서의 ATE)**:[1]

$$\widehat{\Delta}_1(D_{train}, D_{test}) = \frac{\sum_{X_i \in \tilde{D}^{(1)}_{test}} Y_i G_i}{\sum_{X_i \in \tilde{D}^{(1)}_{test}} G_i} - \frac{\sum_{X_i \in \tilde{D}^{(1)}_{test}} Y_i(1-G_i)}{\sum_{X_i \in \tilde{D}^{(1)}_{test}} (1-G_i)}$$

여기서 $\tilde{D}^{(1)}\_{test} = \{X \in D_{test} : \hat{\Delta}(Z|D_{train}) > 0\}$[1].

**PEACE 시험 결과**:[1]
- 고가치 부분군 RMST 차이: 21.1일 (95% CI: [-1.3, 45.5], p=0.064)
- 저가치 부분군 RMST 차이: -13.2일 (95% CI: [-31.5, 5.2], p=0.161)
- **상호작용 효과**: 34.3일 (95% CI: [4.3, 64.3], p=0.025) ← **통계적으로 유의**

이는 기존의 점 추정치 차이(34.3일)가 통계적으로 유의함을 최초로 입증.[1]

#### 3.2 Application 2: Binary Outcome C-index[1]

로지스틱 회귀로 훈련된 모델의 ROC 곡선 아래 면적(AUC/C-index) 평가.[1]

**C-index (ROC-AUC)**:[1]

$$\hat{\theta}(D_{train}, D_{test}) = \frac{1}{\tilde{n}_{test,0}\tilde{n}_{test,1}}\sum_{X_i \in D_{test}^{(0)}}\sum_{X_j \in D_{test}^{(1)}} I(\hat{\beta}'_{\text{train}}\tilde{Z}_i < \hat{\beta}'_{\text{train}}\tilde{Z}_j)$$

- $\tilde{n}_{test,g}$: 검증 집합에서 $Y_i=g$인 표본 수
- $I(\cdot)$: 지시 함수 (음성 점수 < 양성 점수면 1)

**실제 사례 (MI 데이터)**:[1]
- 652명의 환자, 100개의 예측 변수
- 4가지 모델 비교 (Day 0 vs Day 3, 이진화 여부)
- 모델 간 AUC 차이: -0.004 ~ 0.014 (모두 95% CI에서 0 포함)
- 결론: 통계적으로 유의한 성능 차이 없음

***

### 4. 성능 향상 및 일반화 능력

#### 4.1 시뮬레이션 결과

**Precision Medicine (p=10, n=180)**:[1]

| m | Err_m | E(Err_CV_m) | SD | Coverage (σ_CV,m) | Coverage (σ_CV,m,adj) |
|---|-------|-------------|----|--------------------|------------------------|
| 80 | 0.369 | 0.377 | 0.196 | 95.1% | 92.7% |
| 100 | 0.398 | 0.409 | 0.198 | 95.1% | 92.3% |
| 120 | 0.421 | 0.433 | 0.199 | 95.1% | 91.4% |
| 140 | 0.439 | 0.449 | 0.202 | 94.9% | 91.4% |

- **무편향성**: 경험적 편향은 표준편차 대비 무시할 수준
- **적절한 신뢰도**: σ_CV_m 사용 시 95% CI coverage 달성[1]
- **조정 필요성**: σ_CV,m,adj 적용 시 약간의 저보험(under-coverage)[1]

**고차원 케이스 (p=1000, Lasso)**:[1]
- σ_CV_m: 95.0% ~ 95.9% coverage
- σ_CV,m,adj: 89.4% ~ 91.0% coverage
- **해석**: 비정규성(lasso, random forest)에서도 우수한 성능[1]

**소수 부트스트랩 반복 (B_BOOT=20, B_CV=25)**:[1]

| p | Algorithm 2 Coverage | Algorithm 3 Coverage |
|---|---------------------|---------------------|
| 10 | 90.8% | 94.8% |
| 1000 | 89.9% | 94.2% |

- **Algorithm 2만으로**: ~90% coverage (목표 95% 미달)
- **Algorithm 3 적용 후**: ~95% coverage 달성[1]
- **대가**: 신뢰 구간 폭 14-28% 증가[1]

#### 4.2 일반화 능력에 대한 함의

**1) 더 정확한 모델 비교**:[1]

- PEACE 시험에서 ITR 상호작용의 유의성을 첫 번째로 실증적으로 확인
- MI 데이터에서 모델 성능 차이의 통계적 비유의성 명확히 입증

**2) 신뢰도 있는 모델 선택**:[1]

$$\text{CI}_g - \text{CI}_h = \text{신뢰성 있는 성능 차이 추정}$$

CV 추정치의 신뢰도 구간이 좁으면 모델 선택이 더 신뢰할 수 있음.[1]

**3) Overfitting 탐지**:[1]

큰 표준 오차는 다음을 시사합니다:
- 훈련 절차의 불안정성
- 데이터 크기 또는 모델 복잡도 부조화
- 재교육 또는 정규화 필요성

**4) 샘플 크기 계획**:[1]

신뢰 구간 폭을 목표 수준으로 조정하기 위한 필요 표본 크기 계산 가능.

***

### 5. 논문의 한계 및 개선 방향

#### 5.1 명시된 한계

**1) 이론과 실제의 갭**:[1]

논문은 정규성 조건(Section 1, Supplementary Material)을 가정하지만:
- Lasso 정규화, Random Forest는 이 조건을 엄격히 만족하지 않음
- 그럼에도 경험적으로 우수한 성능 달성

**2) 비i.i.d. 데이터**:[1]

현재 방법은 i.i.d. 가정 기반. 확장 가능성:
- 종단 데이터: 같은 피험자의 관찰을 cross-validation/bootstrap 단위로 유지
- 시계열: 블록 bootstrap 또는 시계열 CV 적용

**3) 신뢰도-계산 트레이드오프**:[1]

| 방법 | 신뢰도 | 계산 비용 | 실무성 |
|------|-------|---------|-------|
| Algorithm 2 (B=400, 20) | 95% | 8,000 train | 우수 |
| Algorithm 3 (B=20, 50) | 95% | 1,000 train | 매우 우수 |
| Naive Bootstrap | 95% | 400,000 train | 불가능 |

**4) λ₀ = 0.368의 최적성**:[1]

Sample size 조정에서 사용되지만, 이론적 정당성 부족. 광범위한 실험에서 robust함을 확인.[1]

#### 5.2 개선 방향

**1) 이론 강화** (제1, 2 저자들의 향후 과제):[1]
- 비정규 모델(lasso, random forest)에 대한 중심 극한 정리 확장
- Donsker 조건 완화

**2) 고급 응용**:
- 신경망, 앙상블 모델에 대한 특화된 이론
- 개별화된 치료 추천 시스템의 불확실성 정량화

**3) 소프트웨어 구현**:[1]
- 공개 R/Python 패키지 제공 (Code and Data 섹션)
- 의료 연구자들의 접근성 향상

***

### 6. 2020년 이후 관련 연구 비교 분석

#### 6.1 주요 관련 논문들

**1. Bates, Hastie, Tibshirani (2021) - "Cross-validation: what does it estimate and how well does it do it?"**[2]

| 측면 | 본 논문 (Cai et al.) | BHT (2021) |
|------|------------------|-----------|
| **Estimand 정의** | $Err_m$ 명시 | 두 estimand 제시 |
| **계산 방식** | Algorithm 2: Random effects | Nested CV: 추가 루프 |
| **성능 지표** | 임의의 함수 $L(\cdot)$ | 특정 형태만 |
| **정밀 의학** | ✓✓ (전용 알고리즘) | ✗ |
| **계산량** | 8,000 | 80,000+ |
| **신뢰도** | 95% (B=400) | 95% (B=증가필요) |

**핵심 차이**:[2][1]
- BHT의 nested CV: $outer \times inner$ 루프로 계산 폭증
- Cai의 Algorithm 2: Random effects 분해로 within-part variance 효율적 추정

**2. Lei (2020) - "Cross-validation with confidence"**[3]

- CV 기반 변수 선택의 신뢰성 분석
- Set-valued selection (여러 모델 포함)
- 본 논문과 상호보완적

**3. 현대적 CV 이론 (2025)**[4][5]

최근 2025년 발표된 "A Modern Theory of Cross-Validation Through the Lens of Stability":
- Stability 개념으로 CV 통합 이론화
- Post-hoc randomization (permutation, jackknife, bootstrap, conformal)
- Black-box inference 관점[4]

**4. Conformal Prediction Methods (2023-2025)**[6][7]

| 측면 | Bootstrap (본 논문) | Conformal Prediction |
|------|-------------------|---------------------|
| **이론적 요구** | 점근 정규성 | 유한 표본 보장 |
| **모델 가정** | 일부 정규성 | 모델 비의존적 |
| **계산 비용** | 중간-높음 | 낮음 |
| **적용성** | 특정 지표 | 일반적 예측 |
| **해석** | 신뢰 구간 | 예측 집합 |

**Conformal의 강점**:[6]
- 유한 표본에서 (1-α) 적응 보장
- 모델 구조 무관
- 고차원 설정에서도 유효

**5. Nested CV 발전 (2023-2025)**[8][9]

**NACHOS (2025)**: Nested & Automated Cross-validation using Supercomputing[9]
- HPC 활용한 자동화
- 의료 영상 딥러닝 벤치마킹
- 테스트 성능 분산 정량화

#### 6.2 학문적 위치 파악

```
논문 타임라인 (Model Performance Uncertainty Quantification)

2005: Dudoit, van der Laan - CV asymptotic theory
2015: LeDell et al. - Efficient ROC AUC variance
2017: Lei - CV with confidence
2020: Bayle et al. - CV confidence intervals
2020: Lei - CVC method formalized
2021: Bates, Hastie, Tibshirani - What does CV estimate?
2023: Cai et al. (본 논문) ← 효율적 bootstrap 제안
2023-: Conformal prediction rapid development
2024-: Block CV, correlated data extensions
2025: Unified stability theory perspective
2025: NACHOS integration
```

**본 논문의 기여도 평가**:[2][1]
- **순서상**: BHT(2021)의 문제 지적 이후 해결책 제시
- **효율성**: 부트스트랩의 계산 병목을 10배 해결
- **응용성**: Precision medicine 등 실무 문제 해결
- **완성도**: 유한 표본 보정(Algorithm 3)까지 제공

***

### 7. 미래 연구 및 영향

#### 7.1 즉각적 영향

**1. 의료 연구 커뮤니티**:[1]
- PEACE 사례: Precision medicine strategy 평가에 첫 적용
- 임상시험에서 부분군 분석의 신뢰도 향상

**2. 머신러닝 모델 평가**:
- 모델 비교의 통계적 유의성 검증 가능
- 하이퍼파라미터 선택의 정당성 강화

**3. 소프트웨어 에코시스템**:
- Code and Data 공개로 재현성 보장
- R/Python 패키지화로 접근성 향상

#### 7.2 장기 연구 방향

**1. 이론 확장**:[1]
- 비정규 모델 (deep learning, ensemble)에 대한 엄밀한 이론
- 고차 항 분석을 통한 유한 표본 정확도 향상

**2. 실용화**:
- 시계열, 클러스터 데이터로의 확대
- Online/streaming 설정에서의 CV

**3. 다른 평가 방법과의 통합**:
- Conformal prediction과의 하이브리드
- Bayesian model averaging과의 조합

**4. 새로운 응용**:
- Causal inference (IPW, doubly robust estimators)
- Fairness-aware model selection
- Reinforcement learning 성능 평가

#### 7.3 연구 시 고려할 점

**1. 모델 선택**:
- 문제 특성에 맞는 성능 지표 선택 (CATE, AUC 등)
- Estimand의 명확한 정의 (Err_m vs Err(D_n))

**2. 계산 전략**:
- 모델 훈련 비용이 낮으면: B_BOOT=400, B_CV=20 추천[1]
- 훈련 비용이 높으면: Algorithm 3 적용, B_BOOT=20, B_CV=50[1]

**3. 조건 검증**:
- 정규성 조건 만족 여부 경험적 확인
- Q-Q plot, Kolmogorov-Smirnov test로 정규성 검증

**4. 보고**:
- Point estimate: $\widehat{Err}_{CV_m}$
- Standard error: $\hat{\sigma}\_{CV_m}$ (또는 $\hat{\sigma}_{CV_m,adj}$)
- 95% CI: $[\cdot]$ with coverage rate 명시

**5. 표본 크기 계획**:
- 원하는 CI 폭에 필요한 n 계산
- Cross-validation 반복 수 결정 (B_CV, B_BOOT)

***

### 결론

**"Bootstrapping the Cross-Validation Estimate"**는 머신러닝 모델 평가의 오래된 문제인 **CV 추정치의 불확실성 정량화**를 해결하는 획기적인 논문입니다.[1]

**핵심 기여**:
1. **개념적**: Estimand를 명확히 정의(Err_m)하고 CV의 성격을 재정의
2. **방법론적**: Random effects model을 통해 계산 효율성 10배 향상
3. **응용적**: Precision medicine, 의료 연구의 실제 문제 해결

**일반화 능력**:
- 다양한 복잡한 성능 지표에 적용 가능
- 95% 신뢰도 달성 (B_BOOT=400) 또는 97%+ (Algorithm 3)
- 고차원 데이터, 비정규 모델에서도 경험적 우수성

**앞으로의 과제**:
1. 이론과 실제 갭 축소 (비정규 모델)
2. 비i.i.d. 데이터로의 확장
3. Conformal prediction 등 대안 방법과의 통합

이 논문은 통계학과 머신러닝의 교점에서 **실무적으로 즉시 사용 가능한 도구**를 제공함으로써, 모델 평가의 과학화에 중요한 기여를 하고 있습니다.[1]

***

### 참고문헌

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b0d980d0-22b7-45f6-ba79-c2f264ff83ec/2307.00260v2.pdf)
[2](https://www.tandfonline.com/doi/full/10.1080/01621459.2023.2197686)
[3](https://arxiv.org/pdf/1703.07904.pdf)
[4](http://www.nowpublishers.com/article/Details/STA-005)
[5](https://arxiv.org/html/2505.23592v1)
[6](https://pmc.ncbi.nlm.nih.gov/articles/PMC12091895/)
[7](https://pmc.ncbi.nlm.nih.gov/articles/PMC11238240/)
[8](https://academic.oup.com/bioinformaticsadvances/advance-article-pdf/doi/10.1093/bioadv/vbad048/49880117/vbad048.pdf)
[9](https://arxiv.org/abs/2503.08589)
[10](https://essd.copernicus.org/articles/17/5571/2025/)
[11](https://arxiv.org/abs/2509.16926)
[12](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025EF006704)
[13](https://editoncpublishing.org/ecpj/index.php/ECJECS/article/view/627)
[14](https://www.mdpi.com/2073-4441/17/10/1445)
[15](https://www.cinc.org/archives/2025/pdf/CinC2025-368.pdf)
[16](https://link.springer.com/10.1007/s44288-025-00307-2)
[17](https://www.tandfonline.com/doi/full/10.1080/01621459.2020.1796676)
[18](https://link.springer.com/10.1007/s00477-025-03041-w)
[19](http://arxiv.org/pdf/1708.07180.pdf)
[20](https://arxiv.org/pdf/2307.00260.pdf)
[21](https://arxiv.org/pdf/2110.08720.pdf)
[22](http://arxiv.org/pdf/2404.19145.pdf)
[23](https://arxiv.org/pdf/2403.20182.pdf)
[24](https://authors.library.caltech.edu/records/1b23n-q5002/files/2103.09982.pdf?download=1)
[25](http://arxiv.org/pdf/2408.16763.pdf)
[26](https://arxiv.org/pdf/2201.11676.pdf)
[27](https://arxiv.org/html/2307.00260v2)
[28](https://pmc.ncbi.nlm.nih.gov/articles/PMC11412612/)
[29](https://proceedings.neurips.cc/paper_files/paper/2022/file/949b3011c50300a2b4e60377466f52a8-Paper-Conference.pdf)
[30](https://pmc.ncbi.nlm.nih.gov/articles/PMC10734684/)
[31](http://proceedings.mlr.press/v139/akbari21a/akbari21a.pdf)
[32](https://sundong.kim/courses/mldl24f/notes/mldl24f-ch5-Resampling-Methods.pdf)
[33](https://www.reddit.com/r/statistics/comments/1iyqosf/question_calculating_confidence_intervals_from/)
[34](https://www.emergentmind.com/topics/generalization-gap)
[35](https://pmc.ncbi.nlm.nih.gov/articles/PMC7304018/)
[36](https://arxiv.org/pdf/2507.06266.pdf)
[37](https://arxiv.org/pdf/2404.09247.pdf)
[38](https://arxiv.org/pdf/2511.03684.pdf)
[39](https://arxiv.org/pdf/2102.02016.pdf)
[40](https://arxiv.org/html/2512.01123v1)
[41](https://arxiv.org/pdf/2503.07325.pdf)
[42](https://arxiv.org/html/2510.08359)
[43](https://hastie.su.domains/MOOC-Slides/cv_boot.pdf)
[44](https://ai.stanford.edu/~ronnyk/accEst-talk.pdf)
[45](https://www.jmlr.org/papers/volume22/20-1164/20-1164.pdf)
[46](https://www.semanticscholar.org/paper/e39367f8b5d5e9488c5e22bda970350e6965e914)
[47](https://www.semanticscholar.org/paper/fec7bba921df39c0ce038ba4e362482caa335512)
[48](http://rusraptors.ru/index.php/RC/article/view/420)
[49](https://arxiv.org/pdf/2104.00673v2.pdf)
[50](https://arxiv.org/pdf/1809.09446.pdf)
[51](http://arxiv.org/pdf/2408.03138.pdf)
[52](https://arxiv.org/pdf/2102.06814.pdf)
[53](http://arxiv.org/pdf/2502.14808.pdf)
[54](https://www.sciencedirect.com/science/article/abs/pii/S0957417421006540)
[55](https://koekvall.github.io/files/univ_vc.pdf)
[56](https://mucollective.northwestern.edu/files/2024-conformal.pdf)
[57](https://www.reddit.com/r/statistics/comments/mi6jaj/r_crossvalidation_what_does_it_estimate_and_how/)
[58](https://pmc.ncbi.nlm.nih.gov/articles/PMC2762235/)
[59](https://transferlab.ai/blog/cross-validation-what-does-it-estimate/cross-validation.pdf)
[60](https://arxiv.org/html/2509.00255v1)
[61](https://arxiv.org/html/2410.06494v2)
[62](http://arxiv.org/abs/2104.00673v3)
[63](https://arxiv.org/pdf/2503.17395.pdf)
[64](https://arxiv.org/pdf/2511.00727.pdf)
[65](https://www.arxiv.org/pdf/2512.05611.pdf)
[66](https://arxiv.org/pdf/2408.03138.pdf)
[67](https://arxiv.org/html/2510.07649v1)
[68](https://arxiv.org/pdf/2507.23113.pdf)
[69](https://arxiv.org/abs/2104.00673)
[70](https://www.crosstab.io/articles/bates-cross-validation/)
[71](https://pmc.ncbi.nlm.nih.gov/articles/PMC3271712/)
[72](https://arxiv.org/html/2503.23561v1)
