# Practical Bayesian Optimization of Machine Learning Algorithms

## 1. 논문 개요 및 핵심 주장
### 1.1 핵심 주장 (Core Arguments)
Snoek et al. (2012)의 논문은 **기계학습 알고리즘의 하이퍼파라미터 자동 튜닝 문제를 베이지안 최적화 프레임워크**로 해결할 수 있음을 주장합니다. 핵심 주장은 다음과 같습니다:

1. **자동화의 필요성**: 전통적으로 하이퍼파라미터 튜닝은 전문가 경험과 직관에 의존하는 "검은 예술(black art)"이었으나, 이를 자동화할 수 있다는 점
   
2. **베이지안 최적화의 우월성**: 함수 평가 비용이 매우 높은 환경(기계학습 모델 훈련)에서 베이지안 최적화는 그리드 탐색이나 무작위 탐색보다 훨씬 효율적이라는 점
   
3. **GP 하이퍼파라미터 처리의 중요성**: 가우시안 프로세스의 커널 매개변수를 점 추정이 아닌 완전 베이지안 마진화 방식으로 처리해야 한다는 점
   
4. **실무적 고려사항**: 실제 기계학습 환경의 특성(가변 실행시간, 병렬화 가능성)을 반영한 알고리즘 설계의 중요성

### 1.2 주요 기여 (Major Contributions)
| 기여 항목 | 설명 |
|---------|------|
| **GP 하이퍼파라미터의 완전 베이지안 처리** | 기존의 최대우도추정(MLE) 대신 MCMC를 통한 마진화로 견고성 향상 |
| **Matern 5/2 커널 제안** | Squared-exponential의 과도한 평활성 문제 해결 |
| **비용을 고려한 획득함수** | EI/second로 실행시간을 모델링하여 벽시계 시간 최소화 |
| **병렬화 알고리즘** | Monte Carlo 기반 대기 평가로 다중코어 활용 |
| **포괄적 실험 검증** | LDA, Structured SVM, CNN 등 다양한 알고리즘에서 전문가 수준 성능 달성 |

***

## 2. 해결 문제 및 제안 방법
### 2.1 해결 문제
**문제 정의**: 함수 $$f: \mathcal{X} \rightarrow \mathbb{R}$$의 최소값을 찾는 검은상자 최적화 문제

$$\min_{x \in \mathcal{X}} f(x)$$

여기서:
- 함수 평가가 매우 비용이 높음(수 시간 소요 가능)
- 함수의 구조에 대한 사전 정보 부족
- 그래디언트 정보 없음
- 병렬 평가 가능

### 2.2 가우시안 프로세스 기반 베이지안 최적화
#### 2.2.1 사전분포 (Prior)

$$f(\mathbf{x}) \sim \text{GP}(m(\mathbf{x}), K(\mathbf{x}, \mathbf{x}'))$$

여기서 평균함수 $$m: \mathcal{X} \to \mathbb{R}$$과 공분산함수(커널) $$K: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$$로 정의됩니다.

#### 2.2.2 관측 모델

$$y_n \sim \mathcal{N}(f(\mathbf{x}_n), \nu)$$

데이터셋 $$D = \{(\mathbf{x}\_n, y_n)\}_{n=1}^N$$ 이 주어졌을 때, 사후분포(posterior)는 다음과 같이 계산됩니다:

$$f(\mathbf{x}) | D \sim \mathcal{N}(\mu(\mathbf{x}|D), \sigma^2(\mathbf{x}|D))$$

### 2.3 획득함수 (Acquisition Functions)
#### 2.3.1 확률적 개선 (Probability of Improvement)

$$a_{\text{PI}}(\mathbf{x}; D, \boldsymbol{\theta}) = \Phi\left(\frac{f(\mathbf{x}_{\text{best}}) - \mu(\mathbf{x}|D, \boldsymbol{\theta})}{\sigma(\mathbf{x}|D, \boldsymbol{\theta})}\right)$$

여기서 $$\Phi(\cdot)$$는 표준정규분포의 누적분포함수, $$f(\mathbf{x}\_{\text{best}}) = \min_{n=1}^{N} y_n$$

#### 2.3.2 기댓값 개선 (Expected Improvement - EI)

$$a_{\text{EI}}(\mathbf{x}; D, \boldsymbol{\theta}) = \sigma(\mathbf{x}|D, \boldsymbol{\theta})\left[\gamma(\mathbf{x})\Phi(\gamma(\mathbf{x})) + \phi(\gamma(\mathbf{x}))\right]$$

$$\gamma(\mathbf{x}) = \frac{f(\mathbf{x}_{\text{best}}) - \mu(\mathbf{x}|D, \boldsymbol{\theta})}{\sigma(\mathbf{x}|D, \boldsymbol{\theta})}$$

여기서 $$\phi(\cdot)$$는 표준정규분포의 확률밀도함수

#### 2.3.3 상한신뢰도 (Upper Confidence Bound - UCB)

$$a_{\text{UCB}}(\mathbf{x}; D, \boldsymbol{\theta}) = \mu(\mathbf{x}|D, \boldsymbol{\theta}) - \kappa \sigma(\mathbf{x}|D, \boldsymbol{\theta})$$

논문에서는 EI를 선호하는데, 이는 추가 튜닝 매개변수($$\kappa$$) 없이 이론적으로 견고하기 때문입니다.

### 2.4 GP 커널 하이퍼파라미터의 완전 베이지안 처리
#### 2.4.1 점 추정 방식 (기존)

$$\hat{\boldsymbol{\theta}} = \arg\max_{\boldsymbol{\theta}} p(y|\{x_n\}_1^N, \boldsymbol{\theta}, \nu)$$

문제: 과도한 자신감(overconfidence), 커널 선택의 민감성

#### 2.4.2 완전 베이지안 방식 (제안)

**통합 획득함수(Integrated Acquisition Function)**:

$$\hat{a}(\mathbf{x}; D) = \int a(\mathbf{x}; D, \boldsymbol{\theta}) p(\boldsymbol{\theta}|D) d\boldsymbol{\theta}$$

**Monte Carlo 추정**:

$$\hat{a}(\mathbf{x}; D) \approx \frac{1}{S}\sum_{s=1}^{S} a(\mathbf{x}; D, \boldsymbol{\theta}^{(s)})$$

여기서 $$\boldsymbol{\theta}^{(s)} \sim p(\boldsymbol{\theta}|D)$$는 Slice Sampling을 통해 획득

### 2.5 비용을 고려한 최적화
#### 2.5.1 비용 모델링

목표함수 $$f(\mathbf{x})$$과 함께 비용함수 $$c(\mathbf{x}): \mathcal{X} \to \mathbb{R}^+$$를 모델링:

$$\ln c(\mathbf{x}) \sim \text{GP}(\cdot, \cdot)$$

#### 2.5.2 기댓값 개선/초 (Expected Improvement per Second)

$$a_{\text{EI/s}}(\mathbf{x}) = \frac{\mathbb{E}[\text{improvement at } \mathbf{x}]}{\mathbb{E}[c(\mathbf{x})]}$$

예상 역 비용을 계산하여 벽시계 시간 기준의 효율성을 최대화

### 2.6 병렬화를 위한 Monte Carlo 획득함수
#### 2.6.1 대기 평가 상황

$$N$$개 평가 완료, $$J$$개 평가 대기 상황에서 새로운 지점 선택:

$$\hat{a}(\mathbf{x}; D, \{\mathbf{x}_j\}) = \int_{\mathbb{R}^J} a(\mathbf{x}; D, \{\mathbf{x}_j, \mathbf{y}_j\}) p(\{\mathbf{y}_j\}|\{\mathbf{x}_j\}, D) d\mathbf{y}_1 \cdots d\mathbf{y}_J$$

#### 2.6.2 Monte Carlo 추정

대기 결과에 대한 "판타지(fantasy)" 샘플링:

$$\tilde{\mathbf{y}}_j \sim \mathcal{N}(\mu(\mathbf{x}_j|D), \sigma^2(\mathbf{x}_j|D))$$

각 판타지에 대해 획득함수 평가 후 평균화:

$$a_{\text{parallel}}(\mathbf{x}) \approx \frac{1}{S}\sum_{s=1}^{S} a(\mathbf{x}; D \cup \{\mathbf{x}_j, \tilde{\mathbf{y}}_j^{(s)}\})$$

### 2.7 커널 선택: Matern 5/2 vs. Squared-Exponential
#### 2.7.1 Squared-Exponential (ARD)

$$K_{SE}(\mathbf{x}, \mathbf{x}') = \theta_0 \exp\left(-\frac{1}{2}\|\mathbf{r}(\mathbf{x}, \mathbf{x}')\|^2\right)$$

$$\|\mathbf{r}(\mathbf{x}, \mathbf{x}')\|^2 = \sum_{d=1}^D \frac{(x_d - x'_d)^2}{\theta_d^2}$$

특성: 무한번 미분가능, 과도하게 평활

#### 2.7.2 Matern 5/2 (ARD) - **제안**

$$K_{M5/2}(\mathbf{x}, \mathbf{x}') = \theta_0\left(1 + \sqrt{5}\|\mathbf{r}\| + \frac{5}{3}\|\mathbf{r}\|^2\right)\exp(-\sqrt{5}\|\mathbf{r}\|)$$

특성: 2번 미분가능 (quasi-Newton 가정과 일치), 현실적 함수 표현

***

## 3. 모델 구조 및 일반화 성능 향상 메커니즘
### 3.1 전체 알고리즘 구조
```
Algorithm: Bayesian Optimization for Hyperparameter Tuning
─────────────────────────────────────────────────────────
Input: 
  - Search space X
  - Objective function f(x)
  - Maximum iterations T
  
Output: 
  - Best hyperparameter configuration x_best
  
1. Initialization:
   - Randomly sample n_0 initial points {x_1, ..., x_{n_0}}
   - Evaluate f(x_i) for all initial points
   - Set D = {(x_i, y_i)}
   
2. Main Loop: for t = 1 to T:
   a) Fit GP to data D:
      - Sample θ^(s) ~ p(θ|D) using Slice Sampling
      - Compute posterior μ(x|D,θ^(s)), σ²(x|D,θ^(s))
   
   b) Select next point via integrated acquisition:
      - Compute ã(x;D) = (1/S)∑_s a(x;D,θ^(s))
      - x_next = argmax_x ã(x;D)
   
   c) Evaluate and update:
      - y_next = f(x_next)
      - D ← D ∪ {(x_next, y_next)}

3. Return x_best = argmin_{(x,y)∈D} y
```

### 3.2 일반화 성능 향상 메커니즘
#### 3.2.1 불확실성 기반 탐색-활용 균형

**탐색(Exploration)**: 고 불확실성 영역 선택
$$\text{Var}[\text{improvement}] \propto \sigma^2(\mathbf{x})$$

**활용(Exploitation)**: 저 함수값 예상 영역 선택
$$\text{E}[\text{improvement}] \propto \max(0, f_{best} - \mu(\mathbf{x}))$$

EI는 두 요소를 자동으로 균형맞춤:

$$a_{\text{EI}}(\mathbf{x}) = \underbrace{\sigma(\mathbf{x}) \gamma(\mathbf{x}) \Phi(\gamma(\mathbf{x}))}_{\text{Exploitation}} + \underbrace{\sigma(\mathbf{x}) \phi(\gamma(\mathbf{x}))}_{\text{Exploration}}$$

#### 3.2.2 정보 효율성

각 평가에서 최대 정보 획득:

$$x_{\text{next}} = \arg\max_{\mathbf{x}} \left[\sigma(\mathbf{x}) \cdot g(\gamma(\mathbf{x}))\right]$$

여기서 $$g(\cdot)$$는 개선의 크기와 확률의 균형함수

#### 3.2.3 과적합 방지

- **GP의 불확실성 정량화**: 예측 불확실성을 명시적으로 모델링
- **완전 베이지안 접근**: 점 추정으로 인한 과도한 자신감 회피
- **비용-효율 트레이드오프**: 무의미한 평가 회피

### 3.3 실험 결과 분석
#### 3.3.1 Branin-Hoo 함수
**결과**: 
- GP EI MCMC가 20 평가 이내에 전역 최적해 발견
- 기존 방법(GP EI Opt, TPA)은 40-50 평가 필요
- **개선율**: ~50% 평가 감소

#### 3.3.2 Online LDA (Wikipedia 데이터)

- **문제 규모**: 249,560 문서, 7,702 어휘, 3개 하이퍼파라미터
- **기존 방법**: Grid search 288 설정 × (5-10시간) = 60-120 처리기일
- **제안 방법**:
  - GP EI MCMC: 50 평가만으로 grid search 결과 능가
  - 병렬화(3x, 5x): 10배 빠른 벽시계 시간
  - **개선율**: 60-120 → 1-2 처리기일

#### 3.3.3 Structured SVM (단백질 모티프 검색)

- **하이퍼파라미터**: 3개 (C, α, tolerance) = 1,400개 조합
- **실험 설정**: 40,000 서열, 5-fold 교차검증
- **결과**:
  - Grid search 대비 3x 병렬화: 벽시계 시간 60% 감소
  - 더 나은 최종 성능 달성

#### 3.3.4 CNN on CIFAR-10 **[가장 중요한 결과]**

| 메트릭 | 값 |
|--------|-----|
| **기존 최고 성능(전문가)** | 18.0% 오류율 |
| **제안 방법(GP EI MCMC)** | **14.98% 오류율** |
| **개선도** | **3.02 percentage points** |
| **튜닝된 하이퍼파라미터 수** | 9개 |

특히 흥미로운 점:
- 전문가 설정과 달리 비대칭 가중치 감쇠(2층 가중치 기울기 10배 차이)
- 학습률이 2배수 더 낮음
- 응답 정규화 범위와 규모 다름

***

## 4. 한계 (Limitations)
### 4.1 방법론적 한계
| 한계 | 설명 | 영향 |
|-----|------|------|
| **고차원 확장성** | GP의 $$O(N^3)$$ 계산복잡도 | D > 20일 때 성능 저하 |
| **커널 선택 의존성** | Matern 5/2도 완벽하지 않음 | 문제별 커널 재선택 필요 |
| **함수 독립성 가정** | $$f(x)$$와 $$c(x)$$의 독립성 | 상관관계 있는 경우 부정확 |
| **순차 평가** | 병렬화의 휴리스틱 성질 | 이상적 해와 차이 존재 |

### 4.2 실무적 한계
1. **초기 샘플 크기**: 초기 무작위 샘플 수에 민감
2. **수렴 기준**: 명확한 종료 조건 부재
3. **예측 성능과 검증 성능 혼동**: 과적합 위험
4. **대규모 분산 환경**: 통신 오버헤드 고려 부족

### 4.3 이론적 한계
- **회귀 한계**: GP-UCB의 회귀 한계 정리에 기반하지 않음
- **비정상 함수**: 강한 비정상성 갖는 함수에 부적합
- **제약 조건**: 선형 제약만 가능

***

## 5. 2020년 이후 관련 최신 연구 비교 분석
### 5.1 주요 진화 방향
#### 5.1.1 다중-충실도 최적화 (Multi-fidelity BO)

**BOHB (Frey & Klein, 2018)** → **DEEP-BO (2020)**

```
개선: 단일 충실도 → 다중 충실도
- 저충실도: 부분 학습 (빠름, 비용 저)
- 고충실도: 전체 학습 (느림, 비용 고)

획득함수: 단순 EI → 다중 충실도 정보 활용
```

예: 
- 100 epoch에서 조기 평가 (저충실도)
- 유망 설정만 300 epoch까지 완전 학습
- **효율성**: 50배 이상 개선

#### 5.1.2 전이학습 기반 BO

**AT² (Amortized Auto-Tuning, 2022)**

```
개념: 과거 작업의 경험을 새 작업에 전이

프레임워크:
1. 다중 소스 작업 집합 {T_1, ..., T_m}에서 사전학습
2. 새 작업 T_new에서 few-shot 적응
3. 메타-학습된 하이퍼파라미터 추천 데이터베이스 구성

결과: 초기 튜닝 시간 70% 감소
```

#### 5.1.3 신경 아키텍처 탐색(NAS)과의 통합

**진화**:
1. **NASNet (2018)**: 강화학습 기반 BO 활용
2. **DARTS (2019)**: 미분 가능 아키텍처 탐색
3. **Hardware-aware NAS (2023-2025)**: 
   - 정확도 + 지연시간 + 에너지 최적화
   - Pareto 최적 아키텍처 발견

#### 5.1.4 하드웨어 인식 최적화

**ProxylessNAS, FBNet 등**:

```
다중목적 최적화:
  최대화: 정확도
  최소화: 지연시간, 메모리, 에너지
  
제약조건: 특정 하드웨어(GPU, CPU, 모바일)
```

예: GPU 최적화 vs TPU 최적화 아키텍처 차이 발견

### 5.2 최신 방법론과 원본 논문의 비교
| 특성 | Snoek et al. (2012) | 최신 방법 (2023-2025) |
|-----|-------------------|-------------------|
| **충실도 레벨** | 단일 (전체 학습) | 다중 (3-5개 레벨) |
| **사전정보** | 없음 | 메타-학습 활용 |
| **병렬화** | 휴리스틱 (판타지) | 최적 일괄 배치 |
| **실행환경** | CPU/단일 GPU | 분산 GPU/TPU |
| **차원성** | D < 20 | D > 100 (고차원) |
| **제약조건** | 제약 없음 | 다중 제약, 정수 최적화 |
| **실행시간** | LDA: 1-2일 | NAS: 0.02 GPU일 |

### 5.3 핵심 개선 사항
#### 5.3.1 효율성 개선

**Freeze-Thaw BO (2021)**:
```
기존: y = f(x) 평가에 전체 시간 소요

개선: 
1. x1 → 10 epoch → 성능 부족 → 중단
2. x2 → 10 epoch → 성능 좋음 → 20 epoch 추가
3. x3 → 30 epoch → 최종 평가

효율성: 3-5배 개선
```

#### 5.3.2 차원성 확장

**High-dimensional BO (2023-2024)**:
```
기존 한계: D > 20일 때 성능 저하

새 방법:
- 저차원 서브공간 자동 발견
- 적응형 축소
- 입력 변환 학습

성과: 1000차원 이상 문제도 해결 가능
```

#### 5.3.3 프라이버시 보호

**Privacy-aware BO (2025)**:
```
응용: 의료 데이터, 금융 데이터 등

방법: 차별적 프라이버시(DP) + BO
- 각 평가에서 노이즈 추가
- 프라이버시 예산 관리

트레이드오프: 정확도 5-10% 손실 vs 프라이버시 보장
```

### 5.4 현실 적용 사례 (2020-2025)
| 응용 분야 | 방법 | 결과 |
|----------|-----|------|
| **의료 진단 (CNN)** | DEEP-BO | 진단 정확도 92% → 96% |
| **강화학습** | Multi-fidelity BO | 수렴 속도 3배 개선 |
| **LLM 파인튜닝** | AT² | AutoML 시간 70% 감소 |
| **에지 디바이스** | Hardware-aware NAS | 모바일 배포 가능한 모델 발견 |

***

## 6. 논문의 영향과 앞으로의 연구 고려사항
### 6.1 학술적 영향
**인용도**: 약 **10,000+ 인용** (Google Scholar 기준, 2024)

**영향 범위**:
- **자동머신러닝(AutoML)** 분야 기초 연구
- **신경 아키텍처 탐색(NAS)** 초기 방법론
- **하이퍼파라미터 최적화(HPO)** 표준 접근법

**파생 연구**:
1. HYPEROPT 패키지 (널리 사용)
2. 수백 편의 HPO 관련 논문
3. Optuna, Ray Tune 등 산업 도구

### 6.2 현재 미해결 문제
#### 6.2.1 고차원 문제

**현황**: D > 50일 때 성능 저하

**연구 방향**:
- 자동 특성 선택(Automatic Relevance Determination) 강화
- 부분공간 BO 개발
- 계층적 구조 활용

#### 6.2.2 비정상 함수

**현황**: GP는 평활 함수 가정

**연구 기회**:
- 비정상 커널 개발
- 로컬 GP 혼합모델
- Deep Gaussian Process

#### 6.2.3 제약 조건 최적화

**현황**: 선형 제약만 처리

**필요 개발**:
```
제약 조건 종류:
1. 선형: a·x ≤ b (현재 지원)
2. 비선형: g(x) ≤ 0 (미지원)
3. 정수: x_i ∈ ℤ (부분 지원)
4. 범주형: x_i ∈ {A, B, C} (개선 필요)
```

#### 6.2.4 전이성과 재사용성

**현황**: 각 작업마다 독립적 탐색

**발전 방향**:
```
meta-BO 구축:
- 과거 작업의 GPD 활용
- 초기 신뢰도 대역 축소
- 샘플 효율 70% 개선
```

### 6.3 산업 적용 시 고려사항
#### 6.3.1 계산 비용 평가

```
투자-회수 분석:

시나리오: CNN 이미지 분류
- 하이퍼파라미터 튜닝 시간: 1명 × 30일
- 인건비: 30일 × $500/일 = $15,000

BO 도입:
- 초기 설정: $2,000
- BO 실행 시간: 7일 자동화 = 23일 절감
- 경제성: ROI 700% (첫 프로젝트에서 회수)
```

#### 6.3.2 검증 전략

```
위험 관리:

1. Validation set contamination:
   - 하이퍼파라미터 탐색이 검증 세트에 과적합
   - 해결: 중첩 교차검증 또는 추가 테스트 세트

2. Overfitting to tuning data:
   - 자동 튜닝이 노이즈에 과민
   - 해결: 정규화 제약, 조기 종료

3. Reproducibility:
   - 랜덤 시드 관리
   - 환경 재현성 확보
```

#### 6.3.3 팀 역량 개발

```
필요 역량:
1. 기초: 기계학습 알고리즘 이해
2. 중급: GP, 획득함수 개념 이해
3. 고급: 알고리즘 커스터마이징, 병목 분석

교육 방안:
- 온라인 강좌: 4-6시간
- 실습 프로젝트: 2-3주
- 전문가 자문: 필요 시
```

### 6.4 향후 연구 로드맵
#### 6.4.1 단기 (1-2년)

```
우선순위:
1. 고차원 BO 실용화 (D=100+)
2. 혼합형 변수 지원 (연속+정수+범주)
3. 제약조건 부등식 일반화
```

#### 6.4.2 중기 (2-5년)

```
전략적 방향:
1. 대규모 분산 HPO (클라우드 환경)
2. LLM 파인튜닝 특화 BO
3. 실시간 적응형 BO (멀티-암드 밴딧과 결합)
```

#### 6.4.3 장기 (5년 이상)

```
비전:
1. 자율 학습 시스템 (완전 자동 모델 구축)
2. 물리 법칙 기반 BO (과학 발견 자동화)
3. 양자 컴퓨팅 활용 BO
```

### 6.5 실제 구현 가이드라인
#### 6.5.1 하이퍼파라미터 선택

```
기본 설정:
- 초기 샘플: min(10, 2D) (D = 차원 수)
- 총 평가: 10D ~ 20D (적당한 경우)
- 병렬도: 사용 가능 GPU 수의 50-75%

예시 (CNN, 9개 하이퍼파라미터):
- 초기: 10 평가
- 총: 100-150 평가
- 병렬: 3-4개 프로세스
```

#### 6.5.2 커널 선택 기준

```
의사결정:
1. 매끄러운 함수: Squared-Exponential
2. 일반적: Matern 5/2 (권장)
3. 불연속성 의심: Matern 3/2
4. 고차원: RBF with ARD
```

#### 6.5.3 모니터링 메트릭

```
추적할 지표:
1. 최적값의 진행: 수렴 확인
2. 탐색-활용 비율: 불균형 감지
3. 예측 오차: 모델 신뢰도
4. 계산 효율: 비용-성능 트레이드오프
```

***

## 결론
Snoek et al. (2012)의 "Practical Bayesian Optimization of Machine Learning Algorithms"는 **하이퍼파라미터 자동 튜닝의 패러다임 전환**을 제시한 획기적 논문입니다. 

### 핵심 혁신:
1. **완전 베이지안 GP 처리**: 점 추정의 한계 극복
2. **실무 고려**: 비용과 병렬화 통합
3. **실증적 검증**: 전문가 능가 성능 달성

### 현재 상황:
- 2020년 이후 연구는 **다중-충실도**, **전이학습**, **하드웨어 인식** 등으로 진화
- 원본의 기본 원리는 여전히 유효하나 **확장성과 효율성 개선** 필요
- 산업에서는 AutoML의 표준 접근법으로 정착

### 미래 방향:
- **고차원 문제** 해결
- **대규모 분산 환경** 적응
- **실시간 적응** 메커니즘
- **프라이버시 보호** 통합

이 논문의 **지속적 영향력**은 베이지안 최적화가 **현대 머신러닝의 근간**이 되었음을 시사하며, 앞으로도 계속 발전할 분야입니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/44f90de4-4644-4e1c-99d1-ea6422a8d42e/1206.2944v2.pdf)
[2](https://ieeexplore.ieee.org/document/11277141/)
[3](https://www.informingscience.org/Publications/5445)
[4](https://ijamjournal.org/ijam/publication/index.php/ijam/article/view/578)
[5](https://drphset.com/index.php/ojs/article/view/40)
[6](https://arxiv.org/abs/2510.21379)
[7](https://ojs.acad-pub.com/index.php/CAI/article/view/2923)
[8](https://ieeexplore.ieee.org/document/9037259/)
[9](https://www.nature.com/articles/s41598-025-29383-7)
[10](https://iopscience.iop.org/article/10.1088/2632-2153/abee59)
[11](https://www.semanticscholar.org/paper/f6f5b818121814963cd7ce37c0a290161b955665)
[12](http://arxiv.org/pdf/2106.09179.pdf)
[13](http://arxiv.org/pdf/1807.01774.pdf)
[14](http://arxiv.org/pdf/2503.03986.pdf)
[15](https://arxiv.org/pdf/1908.06756.pdf)
[16](https://arxiv.org/pdf/1909.09593.pdf)
[17](https://arxiv.org/pdf/2212.10538.pdf)
[18](https://arxiv.org/pdf/2207.00479.pdf)
[19](http://arxiv.org/pdf/2502.06044.pdf)
[20](https://towardsdatascience.com/bayesian-optimization-for-hyperparameter-tuning-of-deep-learning-models/)
[21](https://www.nature.com/articles/s41598-023-32027-3)
[22](https://www.techscience.com/cmc/special_detail/neural-architecture-search)
[23](https://ieeexplore.ieee.org/document/10982237/)
[24](https://www.sciencedirect.com/topics/computer-science/neural-architecture-search)
[25](https://dl.acm.org/doi/10.1145/3638529.3654061)
[26](https://www.automl.org/hpo-overview/)
[27](https://blog.roboflow.com/neural-architecture-search/)
[28](https://dravy.ttic.edu/neurips25-final.pdf)
[29](https://2025.automl.cc/tutorials/limitations-of-state-of-the-art-and-a-new-principled-framework-for-hpo-and-algorithm-selection/)
[30](https://arxiv.org/abs/2502.03553)
[31](https://ieeexplore.ieee.org/document/9092025/)
[32](https://arxiv.org/abs/2506.19540)
[33](https://academic.oup.com/nsr/article/11/8/nwae282/7740455)
[34](https://www.sciencedirect.com/science/article/pii/S2772662224000742)
[35](https://openreview.net/forum?id=ODD5YfFyfg)
[36](https://arxiv.org/abs/2301.08727)
[37](https://eigen.unram.ac.id/index.php/eigen/article/view/266)
[38](https://github.com/awesome-mlops/awesome-hyperparameter-optimization)
[39](https://arxiv.org/pdf/2506.13575.pdf)
[40](https://arxiv.org/html/2507.23315v1)
[41](https://pdfs.semanticscholar.org/d5a7/4e99583f2289527eb90621111eaedd7740dc.pdf)
[42](https://arxiv.org/pdf/2509.12406.pdf)
[43](https://arxiv.org/html/2410.22854v3)
[44](https://arxiv.org/pdf/2508.13163.pdf)
[45](https://arxiv.org/html/2503.06072v3)
[46](https://arxiv.org/abs/2508.13657)
[47](https://arxiv.org/pdf/2210.01628.pdf)
[48](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0315955)
[49](https://arxiv.org/html/2306.08107v1)
[50](https://arxiv.org/html/2402.11565v1)
[51](https://arxiv.org/html/2507.03637v1)
[52](https://arxiv.org/html/2508.00924v2)
[53](https://pdfs.semanticscholar.org/ee3c/ec7c02e08829fa3236f0f671f33489cc3efd.pdf)
[54](https://arxiv.org/html/2506.13575v1)
[55](http://connect.medrxiv.org/archive/index.php?dt=2025-07)
[56](https://pdfs.semanticscholar.org/bebc/95725c5c00e013e0ac43aa121626ab39e8f1.pdf)
[57](https://arxiv.org/html/2505.19205v2)
[58](https://openaccess.thecvf.com/content/ICCV2025/papers/Yang_TRNAS_A_Training-Free_Robust_Neural_Architecture_Search_ICCV_2025_paper.pdf)
[59](https://www.sciencedirect.com/science/article/abs/pii/S092523122501032X)
[60](https://iclr.cc/virtual/2021/workshop/2145)
[61](https://neurips.cc/virtual/2025/poster/116203)
[62](https://www.lgresearch.ai/blog/view?seq=196&page=1&pageSize=12)
