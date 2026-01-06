# Bayesian Convolutional Neural Networks with Bernoulli Approximate Variational Inference

### 1. 논문의 핵심 주장 및 주요 기여

**"Bayesian Convolutional Neural Networks with Bernoulli Approximate Variational Inference"**는 Yarin Gal과 Zoubin Ghahramani가 2016년에 발표한 논문으로, 합성곱 신경망(CNN)이 소규모 데이터셋에서 과적합(overfitting)되는 문제를 베이지안 확률론적 접근으로 해결하는 방법을 제시합니다.

**핵심 주장**:
- CNN은 대규모 데이터에서는 우수하지만 레이블이 적은 소규모 데이터에서 빠르게 과적합된다
- Dropout을 베이지안 신경망의 근사적 변분 추론으로 해석하면 수학적으로 정당화된 정규화 기법을 얻을 수 있다
- Bernoulli 변분 분포를 사용하면 추가 모델 매개변수 없이 효율적인 베이지안 CNN을 구현할 수 있다

**주요 기여**:
1. Dropout이 특정 신경망 구조(특히 합성곱 계층)에서 실패하는 이유를 수학적으로 입증
2. Dropout 훈련을 베이지안 신경망의 변분 추론으로 재해석
3. 테스트 시간에 여러 확률적 순전파를 평균화하는 MC Dropout으로 실패 문제 해결
4. 소규모 데이터에서 상당한 성능 개선을 실험적으로 입증

***

### 2. 해결하고자 하는 문제, 제안 방법, 모델 구조

#### 2.1 해결하고자 하는 문제

**근본적 문제**:
- CNN은 매개변수가 많아 소규모 데이터에서 과적합되기 쉬움
- 기존 베이지안 신경망(BNN)의 Gaussian 변분 분포는 매개변수를 배로 증가시켜 CNN에 비실용적
- Dropout을 합성곱 계층에 적용하면 테스트 오차가 증가하는 역설적 현상 발생

**예시**: LeNet을 CIFAR-10에서 테스트할 때, 모든 계층에 Dropout을 적용한 모델(lenet-all)에 표준 방식을 사용하면 성능이 급격히 저하됨.

#### 2.2 제안하는 방법 (수식 포함)

**변분 추론 프레임워크**:

주어진 입력 $\{x_1, \ldots, x_N\}$과 출력 $\{y_1, \ldots, y_N\}$에 대해, 사후 분포(posterior)는 일반적으로 계산 불가능하므로 변분 분포 $q(\omega)$로 근사합니다:

$$\text{KL}(q(\omega) \| p(\omega|X, Y))$$

이를 최소화하는 것은 증거 하한(Evidence Lower Bound, ELBO)을 최대화하는 것과 동치입니다:

$$L_{VI} := \int q(\omega)p(F|X, \omega) \log p(Y|F) dF d\omega - \text{KL}(q(\omega)\|p(\omega))$$

**Bernoulli 변분 분포**:

각 계층 $i$에 대해 Bernoulli 분포를 사용합니다:

$$W_i = M_i \cdot \text{diag}([z_{i,j}]_{j=1}^{K_i})$$

여기서:
- $M_i$: 학습 가능한 변분 매개변수 (평균)
- $z_{i,j} \sim \text{Bernoulli}(p_i)$: 각 계층별 고정 확률 $p_i$를 가진 베르누이 변수
- $\text{diag}(\cdot)$: 대각 행렬 변환

**핵심 통찰**: $z_{i,j}$의 샘플링은 정확히 Dropout과 동일합니다. 값 1일 때 뉴런 활성화, 0일 때 드롭됩니다.

**Monte Carlo Dropout (MC Dropout) 예측**:

테스트 시간에 다중 확률적 순전파를 통해 사후 예측 분포를 근사합니다:

```math
p(y^*|x^*, X, Y) \approx \frac{1}{T} \sum_{t=1}^{T} p(y^*|x^*, \hat{\omega}_t), \quad \hat{\omega}_t \sim q(\omega)
```

여기서 $T$는 MC 샘플 수(일반적으로 20-100)입니다.

#### 2.3 모델 구조

**Bayesian CNN 아키텍처**:

기본 CNN에 다음을 추가합니다:

1. **훈련 단계**:
   - 모든 합성곱 계층(Conv) 뒤에 Dropout 적용 (확률 $p=0.5$)
   - 완전 연결(FC) 계층 뒤에도 Dropout 적용
   - 풀링 전에 Dropout 배치
   - 표준 SGD 최적화 (계산 비용 동일)

2. **테스트 단계**:
   - Dropout을 비활성화하지 않고 유지
   - T번의 순전파 수행 (각각 서로 다른 Dropout 마스크)
   - 출력 평균화: $\hat{y} = \frac{1}{T} \sum_{t=1}^{T} \text{forward}(x, \text{mask}_t)$

**합성곱 연산 재구성**:

합성곱을 선형 연산으로 변환하여 베이지안 가중치 분포 적용:

$$\text{Conv}(\mathbf{x}, \mathbf{w}) = \text{Patches}(\mathbf{x}) \times W$$

여기서 $\text{Patches}(\mathbf{x})$는 $n \times (h \cdot w \cdot K_{i-1})$ 행렬, $W$는 $(h \cdot w \cdot K_{i-1}) \times K_i$ 가중치 행렬입니다. 각 패치에 독립적인 Bernoulli 변수를 적용하면, 풀링 전 각 위치에서 Dropout 효과를 얻습니다.

***

### 3. 성능 향상 및 한계

#### 3.1 성능 향상

**MNIST 실험 (Figure 1)**:

| 설정 | MNIST 오류 (%) | CIFAR-10 오류 (%) |
|------|---|---|
| No Dropout (lenet-none) | ~0.8 | ~35-40 |
| Standard Dropout (lenet-ip) | ~0.68 | ~25-30 |
| Standard Dropout (lenet-all) | ~1.5 | ~50+ (실패) |
| **MC Dropout (lenet-all)** | **~0.5** | **~18-20** |

표준 방식으로는 모든 계층에 Dropout을 적용한 모델이 실패하지만, MC Dropout을 사용하면 모든 모델을 능가합니다.

**CIFAR-10 기존 모델 개선**:

| 모델 | Standard Dropout | MC Dropout (T=100) | 개선 |
|------|---|---|---|
| NIN | 10.43% | **10.27 ± 0.05%** | 0.16% |
| DSN | 9.37% | **9.32 ± 0.02%** | 0.05% |
| Augmented-DSN | 7.95% | **7.71 ± 0.09%** | 0.24% |
| 최소값 | - | **7.51%** (state-of-the-art) | - |

**소규모 데이터셋 성능 (Figure 2)**:

- **전체 MNIST (60,000)**: MC Dropout이 약간 우수
- **1/4 MNIST (15,000)**: Standard Dropout 과적합 시작, MC Dropout 견고함
- **1/32 MNIST (1,875)**: 둘 다 과적합 (추가 정규화 필요)

**MC 샘플 수 영향 (Figure 3)**:

- T=1: 기본 성능
- T=20: 성능 개선 > 1표준편차 (통계적 유의)
- T=100: 수렴 (추가 개선 없음)

따라서 실무에서는 T=20-50이 효율적입니다.

#### 3.2 한계

**1. 이론적 한계**:
- **약한 근사**: Bernoulli 변분 분포는 진정한 사후 분포에 대한 상당히 약한 근사입니다. 예를 들어, 복잡한 다중 봉우리(multimodal) 사후분포를 단순 Bernoulli로 포착할 수 없습니다.
- **데이터 크기 의존성**: 충분히 작은 데이터셋(예: MNIST 1/32)에서는 여전히 과적합 발생. 이는 Bernoulli 근사의 약함을 반영합니다.

**2. 구조적 한계**:
- **ImageNet 실패**: 대규모 데이터(1.2M 이미지)에서는 개선 없음. 저자는 이를 충분한 데이터가 이미 정규화를 제공하기 때문으로 추측
- **풀링 연산**: 풀링의 비선형성이 Dropout 근사를 교란할 가능성

**3. 계산 비용**:
- **훈련**: 추가 비용 없음 (기존과 동일)
- **테스트**: T배 증가 (T=50일 때 50배 느림)
- **메모리**: 추가 메모리 점증적 증가 불필요 (같은 모델, 다른 마스크)

**4. 실무적 고려**:
- **하이퍼파라미터 민감도**: Dropout 확률 $p$는 실험적으로 결정 필요
- **정규화 과다**: 충분히 큰 데이터셋에서는 정규화 과다(underfitting) 가능성

***

### 4. 모델의 일반화 성능 향상 가능성

#### 4.1 일반화 메커니즘

**1. 확률적 적분을 통한 정규화**:

베이지안 관점에서, MC Dropout은 모든 가능한 가중치에 대한 적분(평균화)을 근사합니다:

$$\mathbb{E}_{p(\omega|D)}[f(x, \omega)]$$

이는 단일 가중치 세트에 의존하는 것을 피하고, 가능한 여러 함수를 앙상블하는 효과를 만듭니다. 이것이 "Bayesian model averaging"입니다.

**2. Ensemble 효과**:

MC Dropout의 T개 샘플은 각각 다른 Dropout 마스크(서로 다른 가중치 부분집합)에 해당합니다. 이는 암묵적 앙상블과 유사하여 일반화를 향상시킵니다.

**3. 엣지 근처 불확실성**:

실험 결과, 클래스 경계(decision boundary)에서 불확실성이 높습니다. 이는:
- 신뢰할 수 없는 예측 식별 가능
- 능동 학습(Active Learning)에 활용 가능
- 의사결정에 신뢰도 정보 제공

#### 4.2 작은 데이터셋에서의 향상

**MNIST 1/4 실험의 분석**:

- **Standard Dropout**: 훈련 손실은 감소하지만 검증 손실 증가 (과적합 신호)
- **MC Dropout**: 훈련과 검증 손실이 균형있게 감소, 더 높은 검증 정확도

이는 MC Dropout의 정규화 강도가 더 강함을 시사합니다.

#### 4.3 향상의 한계

**1. 데이터 크기 임계값**:

Figure 2 실험에서 MNIST 1/32(1,875개)일 때 두 방법 모두 과적합됩니다. 이는:
- 베이지안 방법도 극도로 부족한 데이터에는 한계
- 추가 정규화 기법(데이터 증강, 사전학습) 필요

**2. 모델 용량 의존성**:

실험에서 사용된 모델(LeNet)은 상대적으로 작음. 더 큰 모델에서:
- 추가 개선 가능성 있음 (더 많은 매개변수 = 더 강한 정규화 필요)
- 또는 계산 비용이 과도해질 수 있음

***

### 5. 논문의 앞으로의 연구에 미치는 영향

#### 5.1 직접적 영향

**1. Dropout 해석의 패러다임 전환**:
- 이전: Dropout은 경험적 정규화 기법
- 이후: Dropout은 베이지안 변분 추론의 구체화

이 재해석으로 Dropout을 사용하는 모든 신경망이 암묵적 베이지안 모델로 해석되며, 불확실성 정량화가 "자유로이" 가능해집니다.

**2. 불확실성 정량화의 실용화**:
- MC Dropout을 통해 기존 모델에 쉽게 불확실성 추정 추가 가능
- 추가 모델 매개변수나 훈련 변경 불필요
- 의료, 자율주행, 금융 등 고위험 응용에 적용 가능

**3. 소규모 데이터 문제의 해결책**:
- 합성곱 계층의 Dropout 문제를 이론적으로 정당화
- MC Dropout으로 소규모 데이터셋에서도 CNN의 효과적 활용 가능

#### 5.2 파생 연구 분야

**1. 깊은 신경망의 불확실성**:
- Gal & Ghahramani (2015, 2016)의 일반화
- 더 깊은 CNN, RNN, Transformer에 적용

**2. 불확실성 유형 분류**:
- **Aleatoric (데이터) 불확실성**: 데이터 자체의 노이즈
- **Epistemic (모델) 불확실성**: 모델이 데이터로부터 배워야 할 것의 부족

**3. 변분 분포 개선**:
- Gaussian 분포로의 복귀 (Blundell et al. 2015 비판)
- Normalizing flows를 통한 더 표현력 있는 분포
- Concrete Dropout (자동 Dropout 확률 학습)

***

### 6. 2020년 이후 관련 최신 연구 비교 분석

#### 6.1 불확실성 정량화의 진화

**초기 발전 (2020-2022)**:
| 연도 | 주요 기여 | 한계 |
|------|---------|------|
| 2020-2021 | MC Dropout 의료 이미징 적용 | 불확실성 정확도 문제 |
| 2022 | Bayesian Neural Networks for Uncertainty in Materials Science (Olivier et al.) | 계산 복잡도 높음 |
| 2022 | MC Dropout 반복성 개선 연구 (Lemay et al.) | T≥20 필요 |

**고도화 단계 (2023-2024)**:

1. **SOL MC Dropout (Stable Output Layer, 2025)**:
   - 문제: 기본 MC Dropout은 출력 계층 정규화 부족
   - 해결: 마지막 계층의 배치 정규화/드롭아웃 제거
   - 개선: Bootstrap 수준의 불확실성 품질, 계산 시간 동일

2. **Residual Bayesian Attention Networks (2025)**:
   - 혁신: 깊은 네트워크에서 불확실성의 계층적 전파
   - 방법: Gaussian Process 커널 개념을 Attention에 통합
   - 성과: 엔지니어링 최적화(R²=0.972), 시계열(정확도=0.920)

3. **Credal Bayesian Deep Learning (2023-2024)**:
   - 문제: Epistemic/aleatoric 불확실성 혼재
   - 해결: Credal sets을 통한 분리
   - 의의: 분포 이동(distribution shift)에 더 강건

#### 6.2 응용 분야의 확대

**의료 이미징 (2023-2025)**:
- **ComBiNet (2021)**: Compact Bayesian CNN, 파라미터 효율성 개선
- **Cardiac Amyloidosis (2023)**: 데이터 부족 환경에서 신뢰도 향상
- **Brain Tumor Segmentation (2024-2025)**: MC Dropout 불확실성의 한계 지적

**원격 센싱 (2024)**:
- **Bayes R-CNN**: 객체 탐지에서 각 객체의 불확실성 정량화
- **BayesNet (2024)**: UAV 원격 센싱에서 Aleatoric/Epistemic 불확실성 분리

**시계열 예측 (2023-2025)**:
- **CB-LSTM (2023)**: 합성곱+LSTM 조합의 베이지안 해석
- **전력 가격 예측 (2025)**: MC Dropout 기반 확률 예측
- **RUL 예측 (2024)**: 장비 수명 예측의 신뢰 구간

#### 6.3 이론적 깊이의 심화

**변분 추론 재검토 (2024)**:

Variational Bayesian Neural Networks via Singular Learning Theory (Wei et al., 2024):
- **발견**: Variational Free Energy (VFE) 최소화 ≠ 좋은 일반화
- **원인**: "Variational Approximation Gap"의 존재
- **해결**: Singular Learning Theory 기반 개선된 변분족 설계
- **의의**: 변분 분포 선택의 이론적 기초 제공

**Dropout의 재해석 (2024-2025)**:
- **Graph Convolution Networks (ICLR 2025)**: GCN에서 Dropout의 역할 재규명
  - 표준 NN: 뉴런 간 공동적응 방지
  - GCN: 과평활(oversmoothing) 완화가 주 기능
  - 의미: 아키텍처에 따라 Dropout의 메커니즘이 다름

#### 6.4 MC Dropout의 한계 적시

**중요한 비판 (2025)**:

"Unreliable Monte Carlo Dropout Uncertainty Estimation" (arXiv:2512.14851):
- **문제점 발견**:
  1. 단순 회귀에서 MC Dropout이 불확실성 포착 실패
  2. 외삽 영역에서 과신뢰(overconfidence) 문제
  3. Gaussian Process, BNN과 다른 동작

- **원인**: MC Dropout의 Bernoulli 근사가 너무 약함
- **결론**: MC Dropout 불확실성 해석 시 실증적 검증 필수

- **시사점**: Gal & Ghahramani (2016)의 약한 근사라는 한계가 2025년에도 유효

#### 6.5 방법론 비교

| 방법 | 장점 | 단점 | 2024+ 상태 |
|------|------|------|---------|
| **MC Dropout** | 구현 용이, 기존 모델 적용 | 약한 근사 | 비판적 재검토 중 |
| **Deep Ensemble** | 안정적, 다양성 | 계산 비용 (T배) | 여전히 유효하나 비판 제기 |
| **MCMC (HMC)** | 이론적 견고성 | 확장성 나쁨 | 작은 모델에만 실용 |
| **Normalizing Flows** | 표현력 풍부 | 복잡, 계산 비용 높음 | 최신 최고 방법 (2024) |
| **Bayesian Optimization** | 초매개변수 탐색 최적화 | 간접적 불확실성 | 응용 확대 중 |

***

### 7. 앞으로 연구 시 고려할 점

#### 7.1 이론적 고려사항

**1. 근사의 품질 향상**:
- Bernoulli 분포를 넘어선 더 표현력 있는 변분족 탐색
- 각 계층별/뉴런별 맞춤형 변분 분포
- Normalizing flows를 통한 비파라메트릭 근사

**2. 불확실성 유형 구분**:
- MC Dropout 자체로는 Aleatoric 불확실성 포착 불가
- 출력 분포 모델링 필요 (예: Gaussian 혼합 출력)
- 여러 불확실성의 명시적 분리

**3. 깊은 네트워크에서의 불확실성 전파**:
- 초기 계층의 불확실성이 후기 계층에 미치는 영향
- 층별 누적 효과 분석
- 정보병목(Information Bottleneck) 관점 통합

#### 7.2 실무적 고려사항

**1. 적응형 MC 샘플 수 결정**:
- 현재: 수동으로 T 선택 (보통 20-100)
- 개선: 수렴 진단을 통한 자동 최적화
- 지표: Predictive variance stabilization 기준

**2. 데이터셋 크기별 전략**:
- 극소(n<1000): 사전학습 + MC Dropout
- 소규모(1000-10000): 원본 제안 방법 유효
- 중규모(10000-100000): MC Dropout 개선 필요할 수 있음
- 대규모(>100000): MC Dropout 개선 효과 미미 가능성

**3. 모델 복잡도와의 균형**:
- 더 깊은/넓은 모델일수록 더 강한 정규화 필요
- 계산 비용(T배)와 성능 향상의 trade-off
- 하드웨어 가속(GPU 배치 처리) 활용

#### 7.3 평가 및 검증

**1. 불확실성의 정량적 평가**:
- **Calibration**: Expected Calibration Error (ECE)
- **Sharpness**: Prediction Interval Width
- **Coverage**: Prediction Interval Coverage Probability (PICP)
- **Reliability**: 불확실성과 실제 오차의 상관성

**2. Out-of-Distribution (OOD) 감지**:
- MC Dropout이 OOD 샘플 식별 능력 검증
- 도메인 외 샘플에서 높은 불확실성 확인
- 기준선(Random Baseline, Single Model)과 비교

**3. 도메인 적응 성능**:
- 다른 데이터분포에서의 일반화
- 소수 레이블 데이터 추가 시 성능 변화

#### 7.4 새로운 방향

**1. 구조화된 예측**:
- 이미지 분할(Segmentation): 픽셀별 불확실성
- 객체 탐지(Detection): 박스 좌표의 불확실성
- 시계열: 각 타임스텝의 신뢰도

**2. 멀티모달 학습**:
- Vision Transformer + Bayesian
- 언어-비전 모델에서의 불확실성
- 크로스모달 불확실성 전파

**3. 계속 학습(Continual Learning)**:
- 새로운 작업 추가 시 기존 지식 유지
- Bayesian 해석으로 "Catastrophic Forgetting" 완화
- Sequential Variational Inference 활용

**4. 물리 정보 신경망(Physics-Informed NNs)**:
- 과학 시뮬레이션에서의 불확실성 정량화
- 물리 제약 하의 베이지안 추론
- 관측과 모델 불확실성의 결합

***

### 8. 결론

Gal & Ghahramani (2016)의 **"Bayesian Convolutional Neural Networks with Bernoulli Approximate Variational Inference"** 논문은:

**1. 이론적 기여**:
- Dropout을 베이지안 변분 추론으로 정당화하여 수십 년간의 경험적 기법에 수학적 기초 제공
- CNN에 베이지안 정규화를 이론적으로 부여

**2. 실무적 혁신**:
- 기존 도구(Dropout)로 추가 매개변수 없이 불확실성 정량화 실현
- 의료, 자율주행, 원격 센싱 등에 즉시 적용 가능

**3. 미래 방향 제시**:
- 2020-2025년 불확실성 정량화 연구의 기초 설정
- 100+ 후속 논문의 출발점

**4. 앞으로의 과제**:
- Bernoulli 근사의 약함 극복
- 더 깊은 네트워크에서 불확실성 안정성 개선
- 다양한 아키텍처(Transformer, GNN 등)로의 확장
- 실무 응용에서의 신뢰도 검증 강화

**결국, MC Dropout은 완벽한 베이지안 방법은 아니지만, 계산 효율성과 구현 용이성의 관점에서 "충분히 좋은(Good Enough)" 실용적 해결책입니다. 2025년 현재도 여전히 널리 사용되는 이유입니다.**

***

#### 주요 참고 문헌 (2020년 이후 최신 연구)

[1](https://www.nature.com/articles/s41598-025-24093-6)
[2](https://arxiv.org/html/2512.14851v1)
[3](https://www.tandfonline.com/doi/full/10.1080/29979676.2025.2497555)
[4](https://journals.sagepub.com/doi/10.1177/16878132241239802)
[5](https://www.mdpi.com/2072-4292/16/13/2405)
[6](https://www.mdpi.com/2076-3417/13/7/4547)
[7](https://www.sciencedirect.com/science/article/abs/pii/S0045782521004102)
[8](https://www.tandfonline.com/doi/full/10.1080/10618600.2024.2325455)
[9](https://arxiv.org/abs/2302.13425)
[10](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c44de9b3-bcf9-49ff-9272-b4fe767219a7/1506.02158v6.pdf)
[11](https://journalwjarr.com/node/2140)
[12](https://www.mdpi.com/2076-3417/15/11/6173)
[13](https://www.mdpi.com/2079-6374/15/7/402)
[14](https://link.springer.com/10.1007/s00044-025-03407-3)
[15](https://link.springer.com/10.1007/s44442-025-00011-3)
[16](https://link.springer.com/10.1007/s00170-025-16898-6)
[17](https://www.tandfonline.com/doi/full/10.1080/10408347.2025.2527741)
[18](https://www.tandfonline.com/doi/full/10.1080/17568919.2025.2571029)
[19](https://xlink.rsc.org/?DOI=D5RA05002B)
[20](https://www.mdpi.com/1422-0067/26/14/6672)
[21](https://arxiv.org/pdf/2104.06957.pdf)
[22](https://www.mdpi.com/2072-4292/16/5/925/pdf?version=1709718717)
[23](http://arxiv.org/pdf/2210.09560.pdf)
[24](https://pmc.ncbi.nlm.nih.gov/articles/PMC11390735/)
[25](https://arxiv.org/pdf/2403.07657.pdf)
[26](http://arxiv.org/pdf/2304.01762.pdf)
[27](https://arxiv.org/pdf/1506.02158.pdf)
[28](https://pmc.ncbi.nlm.nih.gov/articles/PMC10584795/)
[29](https://pmc.ncbi.nlm.nih.gov/articles/PMC12110443/)
[30](https://www.emergentmind.com/topics/monte-carlo-dropout)
[31](https://www.cs.ox.ac.uk/teaching/courses/2024-2025/UDL/)
[32](https://www.nature.com/articles/s41746-022-00709-3)
[33](https://www.sciencedirect.com/science/article/pii/S0950705125014777)
[34](https://www.sciencedirect.com/science/article/abs/pii/S0045782524007400)
[35](https://www.sciencedirect.com/science/article/abs/pii/S0925231225025998)
[36](https://arxiv.org/pdf/2510.05338.pdf)
[37](https://arxiv.org/html/2509.19180v1)
[38](https://pmc.ncbi.nlm.nih.gov/articles/PMC8588128/)
[39](https://journals.sagepub.com/doi/abs/10.1177/16878132241239802)
[40](https://openreview.net/forum?id=xJXq6FkqEw)
[41](https://www.tandfonline.com/doi/full/10.1080/00295450.2025.2518613)
[42](https://www.sciencedirect.com/science/article/abs/pii/S0022169421012944)
[43](https://papers.phmsociety.org/index.php/phmconf/article/view/4344)
[44](https://arxiv.org/html/2506.14831v2)
[45](https://www.arxiv.org/pdf/2511.11701.pdf)
[46](https://arxiv.org/pdf/2510.09586.pdf)
[47](https://arxiv.org/html/2510.15541v1)
[48](https://pdfs.semanticscholar.org/b740/2acc8b8ccbd2d46784b1f90b94fcd8d85ade.pdf)
[49](https://arxiv.org/html/2511.23440v1)
[50](https://arxiv.org/html/2511.11701v1)
[51](https://arxiv.org/pdf/2408.17059.pdf)
[52](https://arxiv.org/html/2411.16370v4)
[53](https://arxiv.org/html/2508.16891v1)
[54](https://arxiv.org/pdf/2509.04153.pdf)
[55](https://arxiv.org/pdf/2403.10671.pdf)
[56](https://arxiv.org/html/2503.09224v1)
[57](https://www.biorxiv.org/lookup/external-ref?access_num=10.1093%2Fbib%2Fbbaf136&link_type=DOI)
[58](https://arxiv.org/html/2508.07458v1)
[59](https://www.arxiv.org/abs/2510.15541)
[60](https://peerj.com/articles/cs-3193.pdf)
[61](https://arxiv.org/html/2512.10602v1)
[62](https://ieeexplore.ieee.org/document/10159307/)
[63](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12463/2654318/Patient-specific-uncertainty-and-bias-quantification-of-non-transparent-convolutional/10.1117/12.2654318.full)
[64](https://arxiv.org/abs/2403.12729)
[65](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024SW003909)
[66](https://scholar.kyobobook.co.kr/article/detail/4010068672903)
[67](https://aapm.onlinelibrary.wiley.com/doi/10.1002/mp.17189)
[68](https://pubs.geoscienceworld.org/geophysics/article/89/1/WA53/632247/Combining-unsupervised-deep-learning-and-Monte)
[69](https://ieeexplore.ieee.org/document/10263586/)
[70](https://arxiv.org/html/2504.07696v1)
[71](https://arxiv.org/pdf/2302.09656v2.pdf)
[72](https://arxiv.org/pdf/2312.15297.pdf)
[73](https://arxiv.org/pdf/2402.17915.pdf)
[74](https://arxiv.org/abs/2210.11737)
[75](http://arxiv.org/pdf/2302.10975.pdf)
[76](https://openreview.net/pdf?id=PwxYoMvmvy)
[77](https://www.pymc.io/projects/examples/en/latest/variational_inference/bayesian_neural_network_advi.html)
[78](https://arxiv.org/pdf/1904.03392.pdf)
[79](https://pubs.acs.org/doi/abs/10.1021/acs.jctc.2c01267)
[80](https://arxiv.org/html/2510.10268v2)
[81](https://ieeexplore.ieee.org/iel8/6287639/10820123/11186816.pdf)
[82](https://www.sciencedirect.com/science/article/pii/S2352484725003579)
[83](https://www.sciencedirect.com/science/article/abs/pii/S0893608018301096)
[84](https://arxiv.org/abs/2302.10975)
[85](https://arxiv.org/html/2510.23684v1)
[86](https://dl.acm.org/doi/full/10.1145/3510413)
[87](https://repository.uwl.ac.uk/id/eprint/12845/1/ochella-et-al-2024-bayesian-neural-networks-for-uncertainty-quantification-in-remaining-useful-life-prediction-of.pdf)
[88](https://openreview.net/forum?id=JRBctqPV8U)
[89](https://arxiv.org/html/2506.12738v1)
[90](https://epubs.siam.org/doi/10.1137/21M1439456)
[91](https://proceedings.neurips.cc/paper_files/paper/2024/file/750a56383caf20b92fe070732f969300-Paper-Conference.pdf)
[92](https://arxiv.org/pdf/2410.14390.pdf)
[93](https://arxiv.org/html/2502.01342v2)
[94](https://arxiv.org/html/2411.16370v1)
[95](https://arxiv.org/pdf/2503.07114.pdf)
[96](https://arxiv.org/html/2512.22192v1)
[97](https://arxiv.org/html/2411.16370v3)
[98](https://openaccess.thecvf.com/content/CVPR2025/papers/Xu_Adaptive_Dropout_Unleashing_Dropout_across_Layers_for_Generalizable_Image_Super-Resolution_CVPR_2025_paper.pdf)
[99](https://arxiv.org/html/2406.14838v1)
[100](https://arxiv.org/html/2406.04317v3)
[101](https://arxiv.org/html/2503.21419v1)
[102](https://arxiv.org/html/2510.06025v1)
[103](https://arxiv.org/html/2511.10282v1)
[104](https://arxiv.org/pdf/2502.21143.pdf)
[105](https://arxiv.org/html/2505.22342v3)
[106](https://arxiv.org/pdf/2412.08776.pdf)
[107](https://arxiv.org/html/2402.00809v4)
[108](https://www.geeksforgeeks.org/deep-learning/dropout-regularization-in-deep-learning/)
