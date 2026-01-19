
# InfluenceNet: AI Models for Banzhaf and Shapley Value Prediction

## 개요

이 연구는 협력 게임 이론의 핵심 개념인 전력지수(power indices)를 신경망을 통해 효율적으로 추정하는 방법론을 제시하며, 지수 시간 복잡도 계산의 병목을 극복하고자 한다.

***

## 1. 핵심 주장 및 주요 기여

### 문제 정의

전통적으로 Banzhaf와 Shapley-Shubik 전력지수는 다중 에이전트 시스템에서 의사결정 과정의 권력 분배를 정량화하는 필수 도구이다. 정치 연합, 기업 이사회, 분산 네트워크 등 광범위한 응용이 존재하나, **계산 복잡도가 치명적이다**. Banzhaf 지수의 정확한 계산은 O(2^m), Shapley-Shubik은 O(m!)의 지수 시간을 요구하므로 m ≥ 10인 대규모 연립에서는 실무적으로 불가능하다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/628656f9-a7b3-4d07-a057-663dc1a34791/2503.08381v1.pdf)

기존 근사 방법(Monte-Carlo, Maximum Sample Reuse)도 속도와 정확도 간의 본질적 트레이드오프를 해결하지 못했다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/628656f9-a7b3-4d07-a057-663dc1a34791/2503.08381v1.pdf)

### 제안 해결책

InfluenceNet은 **신경망 기반 접근법**으로 이 계산 병목을 극복한다. 저자들은 다음을 입증한다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/628656f9-a7b3-4d07-a057-663dc1a34791/2503.08381v1.pdf)

- 신경망이 대규모 연립(n ≥ 10)의 Banzhaf와 Shapley-Shubik 지수를 효과적으로 근사 가능
- 기존 Monte-Carlo 방법 대비 현저히 빠른 계산(15-50분 → 8분)
- 수용 가능한 정확도 수준 유지

### 주요 기여

저자들은 세 가지 차원의 기여를 명시한다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/628656f9-a7b3-4d07-a057-663dc1a34791/2503.08381v1.pdf)

1. **신경망 아키텍처 설계**: 간단한 피드포워드 구조가 power index 근사에 효과적임을 입증하며, 동시에 신경망의 tabular 데이터 처리 약점을 극복하는 설계 원리 제시
   
2. **포괄적 경험적 증거**: 다양한 coalition 구성(sparse-dense, 균등-비균등 가중치), 에이전트 수(10-50), 데이터 생성 방법(Uniform, Coin-flip, MoG)에 따른 성능 특성화 및 편향 경향 명시적 분석

3. **실무 프레임워크**: 이전에 계산 불가능했던 대규모 연립 분석을 가능케 하는 accessible하고 scalable한 도구 제공으로, 다중 에이전트 시스템 연구의 새로운 지평 개방

***

## 2. 해결하고자 하는 문제와 제안 방법

### 2.1 문제 도메인: Marginal Contribution Networks (MCN)

InfluenceNet이 대상으로 하는 문제는 **Marginal Contribution Networks** 클래스의 연립 게임이다. MCN에서 coalition의 가치는 구성원의 한계 기여도(marginal contribution)에 기반한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/628656f9-a7b3-4d07-a057-663dc1a34791/2503.08381v1.pdf)

게임을 형식적으로 정의하면:

$$G = (L, R)$$

여기서:
- $L = \{1, \ldots, m\}$: 에이전트 집합
- $R = \{r_1, \ldots, r_n\}$: Rule 집합

각 rule은 다음 형태:
$$r_i: \text{Pattern}_i \rightarrow v_i$$

Rule의 패턴은 두 부분으로 구성:
- **긍정 요구(req)**: 반드시 포함되어야 할 에이전트
- **부정 요구(ban)**: 제외되어야 할 에이전트

Coalition C의 총 가치:
$$v(C) = \sum_{r_i \in R: \text{Pattern}_i \text{ matches } C} v_i$$

### 2.2 전력지수의 수학적 정의

#### Banzhaf Power Index

Banzhaf 지수는 coalition에서 에이전트가 critical인 경우의 비율을 측정한다. 에이전트 j가 critical이란 coalition C에서 j를 제거하면 가치가 감소함을 의미한다.

$$\beta_j(v) = \frac{1}{2^{m-1}} \sum_{C \ni j} [v(C) - v(C \setminus \{j\})]$$

여기서:
- C ∋ j: 에이전트 j를 포함하는 모든 coalition
- v(C) - v(C \ {j}) > 0이면 j는 critical
- 정규화: 모든 coalition의 동일 확률 가정

#### Shapley-Shubik Power Index

Shapley-Shubik 지수는 에이전트의 모든 가능한 순열(permutation)에서의 평균 한계 기여도를 계산한다.

$$Sh_i(v) = \frac{1}{n!} \sum_{\sigma \in \Pi} [v(C_i^\sigma) - v(C_i^\sigma \setminus \{i\})]$$

여기서:
- $\sigma$: 에이전트들의 순열
- $C_i^\sigma$: permutation σ에서 i보다 먼저 오는 에이전트들 + i
- $v(C_i^\sigma) - v(C_i^\sigma \setminus \{i\})$: 순열 σ에서 i의 한계 기여도

**핵심 차이**: Banzhaf는 모든 coalition을 동등하게 취급하고, Shapley-Shubik은 모든 순열을 동등하게 취급한다. 이는 상이한 권력 평가를 낳으며 상호 보완적 통찰을 제공한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/628656f9-a7b3-4d07-a057-663dc1a34791/2503.08381v1.pdf)

### 2.3 신경망 학습을 위한 데이터 생성

모델을 학습하기 위해 저자들은 **현실적 다양성을 포착하는 3가지 Coalition Rule 생성 전략**을 개발했다.

#### 전략 1: Uniform Random Sampling

$$\text{req}_{jk} = \begin{cases} 1 & \text{if } x_{jk} < p \\ 0 & \text{otherwise} \end{cases}, \quad x_{jk} \sim U(0,1)$$

균등 분포에서 threshold p를 기준으로 요구 여부를 결정. 가장 기본적인 baseline.

#### 전략 2: Coin Flip Assignment

각 coalition rule n에 대해:
$$X_l \sim \text{Uniform}(1, \ldots, m), \quad Y_l \sim \text{Uniform}(0, 1)$$

동전 던지기 방식으로 c번 반복하며 요구/금지 그룹 동시 구성:

$$\text{req}_{X_l} = \begin{cases} 1 & \text{if } Y_l > 0.5 \\ 0 & \text{otherwise} \end{cases}, \quad \text{ban}_{X_l} = 1 - \text{req}_{X_l}$$

더 구조화되면서 비참여 에이전트 감소.

#### 전략 3: Probabilistic Mixture of Gaussian (MoG)

$$\mu_i, \sigma_i^2 \sim \text{Gamma}(\alpha, \beta)$$
$$X_i \sim \mathcal{N}(\mu_i, \sigma_i^2), \quad Y_i \sim \mathcal{N}(\mu_i, \sigma_i^2)$$

$$\text{req}_i = \begin{cases} 1 & \text{if } x_{ab} > p \\ 0 & \text{otherwise} \end{cases}$$

Gaussian 혼합 분포로 더 복잡하고 현실적인 coalition 구조 모사.

#### Rule Value 가중치 할당

추가적으로 rule 중요도에 3가지 분포 적용:

| 방식 | 설명 | 실제 의미 |
|------|------|----------|
| **Uniform** | 모든 rule 동일 가중치 | Baseline scenario |
| **Low-variance Gaussian (σ²=5)** | 미묘한 rule 중요도 차이 | 비교적 균형 잡힌 가중치 |
| **High-variance Gaussian (σ²=15)** | 큰 rule 중요도 편차 | 일부 rule의 지배적 영향 |

Dataset 최종 구성:

$$\text{Dataset shape: } (k, n, 2m+1)$$

- k = 200,000: 데이터포인트 수
- n: coalition(rule) 개수
- 2m: m개 요구 + m개 금지 인덱스
- +1: rule value/score

### 2.4 Label 생성: Monte-Carlo 근사

정확한 power index 계산이 불가능하므로, 저자들은 **Monte-Carlo 근사**로 label을 생성했다.

#### Banzhaf Monte-Carlo 근사 알고리즘

```
알고리즘: Monte-Carlo Banzhaf Index 근사

입력: 에이전트 집합 A, Rule 집합 R, Rule 가중치 W, 시뮬레이션 수 N
출력: 각 에이전트의 근사 가중 Banzhaf 지수

1. 각 에이전트 a에 대해 P_a ← 0

2. i = 1부터 N까지:
   a. C를 2^A에서 균등 임의 샘플
   b. 각 에이전트 a에 대해:
      - C' ← C \ {a}
      - C에서 제거했을 때 rule 상태가 변하면:
        * 상태 변한 rule들의 가중치 합 계산
        * P_a에 누적

3. 각 에이전트 a의 최종 지수: β_a = P_a / N
```

**수식 표현**:

$$\hat{\beta}_a = \frac{1}{N \sum_{r \in R} W_r} \sum_{i=1}^{N} \sum_{r \in R} W_r \cdot \mathbb{1}[\text{status}(r, C_i) \neq \text{status}(r, C_i \setminus \{a\})]$$

N = 10,000 샘플 사용으로 충분한 통계적 신뢰성 확보. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/628656f9-a7b3-4d07-a057-663dc1a34791/2503.08381v1.pdf)

#### Shapley-Shubik Monte-Carlo 근사 알고리즘

```
알고리즘: Monte-Carlo Shapley-Shubik Index 근사

입력: 에이전트 집합 A, Rule 집합 R, Rule 가중치 W, 시뮬레이션 수 N
출력: 각 에이전트의 근사 가중 Shapley-Shubik 지수

1. 각 에이전트 a에 대해 P_a ← 0

2. i = 1부터 N까지:
   a. A의 임의 순열 π 샘플링
   b. C ← ∅ (빈 coalition)
   c. coalition_won ← false
   
   d. π에 따라 각 에이전트 a를 순차 추가:
      - C ← C ∪ {a}
      - C가 winning이고 C\{a}가 losing이면:
        * 상태 변한 rule들의 가중치 합 P_a에 누적
        * coalition_won ← true
      - C에 다음 에이전트 추가

3. 각 에이전트의 최종 지수: Sh_a = P_a / N
```

**수식 표현**:

$$\hat{Sh}_a = \frac{1}{N \sum_{r \in R} W_r} \sum_{i=1}^{N} \sum_{r \in R} W_r \cdot [v(C_a^{\sigma_i}) - v(C_a^{\sigma_i} \setminus \{a\})]$$

이 근사는 정확한 계산 불가능 시에도 합리적 label을 제공한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/628656f9-a7b3-4d07-a057-663dc1a34791/2503.08381v1.pdf)

### 2.5 신경망 모델 구조

저자들은 **간결하면서도 효과적한 feedforward 신경망 아키텍처**를 설계했다.

#### 모델 아키텍처

| 계층 | 유닛 수 | 활성화 함수 | Dropout |
|------|--------|----------|---------|
| Input | Variable (변수) | - | - |
| Hidden 1 | 512 | ReLU | 20% |
| Hidden 2 | 256 | ReLU | 20% |
| Hidden 3 | 128 | ReLU | 20% |
| Output | m (에이전트 수) | Linear | - |

#### 핵심 설계 선택

1. **입력 전처리**: 다양한 크기의 coalition matrix를 처리하기 위해 **오른쪽 zero-padding** 적용
   - 최대 크기에 맞춰 padding하여 배치 학습 가능

2. **활성화 함수**: ReLU 선택으로 비선형성 확보하면서 계산 효율성 유지

3. **정규화**: 20% Dropout으로 과적합 방지

4. **손실 함수**: Mean Squared Error (MSE) 회귀
   $$L = \frac{1}{N} \sum_{i=1}^{N} \|\hat{y}_i - y_i\|^2$$

#### 학습 설정

- **Dataset split**: 80% 학습, 20% 테스트
- **Dataset 크기**: k = 200,000 datapoint (결과적으로 최적)
- **Optimizer**: 표준 backpropagation (구체 명시 없음, 아마 Adam 또는 SGD)
- **학습 전략**: 각 configuration마다 별도 모델 학습
  - 다양한 (n, m, 생성 방법, rule value, p-threshold) 조합에 대해

***

## 3. 성능 향상 특성 및 일반화 성능

### 3.1 실험 설정 및 결과

#### 테스트 설계

저자들은 **교차 검증(cross-validation) 방식**으로 일반화 성능을 평가했다:

1. 모델을 특정 configuration (예: n=20, m=10, uniform rules)에서 학습
2. **다른 모든 configuration에서 테스트**하여 외삽 능력 평가

이를 통해 각 파라미터의 영향을 체계적으로 분석.

#### Uniform Random Dataset 결과

| 에이전트 수 (m) | Rule 가중치 | 성능 (MAE) | 특성 |
|---|---|---|---|
| 10 | 균등 | 0.008-0.012 | 상대적으로 높은 오류 |
| 20 | 균등 | 0.001-0.003 | 우수 성능 |
| 50 | 균등 | 0.0005-0.001 | **최우수 성능** |
| 50 | 고분산 | 0.002-0.005 | 분산 증가하면 악화 |

### 3.2 모델의 일반화 성능 향상 가능성 - 심층 분석

#### 발견 1: 역설적 규모 효과 (Paradoxical Scale Effect) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/628656f9-a7b3-4d07-a057-663dc1a34791/2503.08381v1.pdf)

**예상**: 더 많은 에이전트 → 더 복잡한 공간 → 더 어려운 예측
**실제**: m=50일 때가 m=10보다 **더 낮은 MAE**

$$\text{성능: } \text{MAE}(m=50) \ll \text{MAE}(m=10)$$

**원인 분석**:

더 큰 시스템은 더 **규칙적이고 예측 가능한 패턴**을 포함한다. 이는 다음 수학적 직관으로 설명 가능:

- m이 작을 때: 각 coalition의 구조가 고도로 특이(idiosyncratic), 상대적으로 소수의 데이터
- m이 클 때: coalition 공간이 더 균질화되고, 신경망이 학습할 수 있는 통계적 규칙성 증가

저자들의 해석: "Larger systems may contain more regular patterns that neural networks can leverage for prediction." [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/628656f9-a7b3-4d07-a057-663dc1a34791/2503.08381v1.pdf)

#### 발견 2: Sparse vs. Dense Coalition Rules의 이중성

**Sparse rules (p-threshold 낮음, 예: p=0.1)**:
- 더 많은 자유도(degree of freedom)
- 강한 variance → 더 학습 가능한 신호
- **더 낮은 오류**

**Dense rules (p-threshold 높음, 예: p=0.7)**:
- 많은 제약 → 낮은 권력 지수
- 적은 분산 → 약한 신호
- **더 높은 오류**

그러나 이 패턴은 **비대칭**: 
- Dense에서 학습한 모델이 Sparse 데이터에서 **심각하게 실패**
- Sparse에서 학습한 모델이 Dense 데이터에서는 **비교적 견고**

**근거**: Dense 데이터는 정보가 적어서, 모델이 정보 부족으로 robust feature를 학습하지 못함.

#### 발견 3: Rule Value Distribution의 약한 영향 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/628656f9-a7b3-4d07-a057-663dc1a34791/2503.08381v1.pdf)

최대 MAE 차이: 0.004

$$|\text{MAE}_{\text{high-variance}} - \text{MAE}_{\text{low-variance}}| \leq 0.004$$

**해석**: Rule 가중치의 분산은 model performance에 미미한 영향
- 신경망이 가중치 분포 자체보다는 **rule의 구조적 패턴**에 더 의존
- 실무 의미: 복잡한 가중치 체계도 단순 균등 가중치만큼 잘 처리

#### 발견 4: Padding의 양면성 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/628656f9-a7b3-4d07-a057-663dc1a34791/2503.08381v1.pdf)

Zero-padding의 효과:

**장점**:
- n=10에서 일관성 크게 향상
- 서로 다른 크기 모델 간 비교 가능

**한계**:
- Padding 없는 결과에서도 기본 패턴은 동일
  - Sparse > Dense (여전히)
  - 큰 n > 작은 n (여전히)
- 즉, 결론은 padding 여부와 무관하게 robust

### 3.3 한계 및 성능 장벽

#### 한계 1: 분포 변화에 대한 극심한 취약성 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/628656f9-a7b3-4d07-a057-663dc1a34791/2503.08381v1.pdf)

**Mixture of Gaussian (MoG) 방법의 특별한 실패**:

- p ∈ [0.2, 0.3] 구간에서 **10배 이상의 오류 증가**
- 훈련 분포와 테스트 분포 변화에 불안정
- 저자: "Neural networks cannot reliably learn multiple changing distributions" [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/628656f9-a7b3-4d07-a057-663dc1a34791/2503.08381v1.pdf)

**한계**: 현재 feedforward 아키텍처는 **정적 데이터 분포**만 가정
- Attention 메커니즘 또는 Conditional Networks로 개선 가능
- 하지만 본 논문 범위 초과

#### 한계 2: 에이전트 수 변화에 대한 민감성

**가장 심각한 일반화 한계**:

- m=50에서 학습 → m=10에서 테스트: 오류 최악 10배
- m=10에서 학습 → m=50에서 테스트: 오류 약 0.05배 증가

$$\text{error}_{\text{cross-size}} / \text{error}_{\text{in-domain}} = O(10)$$

**근거**: 신경망이 특정 입력 dimensionality에 과적합
- 입력 크기 변화 = 본질적으로 다른 문제 공간
- Zero-padding은 증상 치료일 뿐, 근본 해결 아님

#### 한계 3: 초기 근사의 병목 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/628656f9-a7b3-4d07-a057-663dc1a34791/2503.08381v1.pdf)

"NN rely on existing previous labels. For such large rule set (m ≥ 20) and number of agents (n ≥ 50), the initial approximation needed in order to generate the labels is a **severe bottleneck**." [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/628656f9-a7b3-4d07-a057-663dc1a34791/2503.08381v1.pdf)

**구체적 문제**:
- 생성할 coalition 개수: 2^m
- m=20: 1,048,576 가능 coalition
- Monte-Carlo 10,000 샘플도 매우 제한적

**해결 방향**: 더 정교한 근사 알고리즘 필요
- 현재: 균일 Monte-Carlo
- 개선 안: 중요도 샘플링(importance sampling), 또는 hierarchical approximation

#### 한계 4: 정적 시스템만 다룸 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/628656f9-a7b3-4d07-a057-663dc1a34791/2503.08381v1.pdf)

"Our current approach focuses on **static voting systems**; extending the methodology to handle dynamic coalitions where relationships between agents evolve over time represents an important direction for future work." [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/628656f9-a7b3-4d07-a057-663dc1a34791/2503.08381v1.pdf)

**현실의 격차**:
- 현실의 voting systems는 동적 (예: 주식 소유권 변화, 정치 연합 변화)
- 본 모델은 한 시점의 스냅샷만 분석 가능

### 3.4 계산 효율성 비교

실제 계산 시간 비교 (저자 데이터):

| 방법 | n=10 agents | n=50 agents | 특징 |
|-----|------------|-----------|------|
| Monte-Carlo | ~15분 | ~50분 | 지수 증가 |
| **NN 학습** | ~8분 | ~8분 | **일정** |
| NN 추론 | <1초 | <1초 | 극도로 빠름 |

**의미**:
- NN 학습은 coalition 크기와 거의 무관하게 linear한 시간 소비
- 훈련 후 추론은 실시간 적용 가능 (1-2초 이내)
- 대규모 시스템 분석에서 NN의 이점 극대화

***

## 4. 2020년 이후 관련 최신 연구와의 비교 분석

### 4.1 연구 생태계 개요

협력 게임 이론과 신경망의 결합은 2020년 이후 급격히 성장했다. 세 가지 주요 방향:

1. **Shapley Value 계산 최적화** (2019-2025)
2. **Banzhaf Index의 재발견과 적용** (2021-2025)
3. **대안적 할당(allocation) 방법** (2024-2025)

### 4.2 Shapley Value 신경망 기반 추정 (2020-2025)

#### A. G-DeepSHAP (2022) - Chen et al., Nature Communications [nature](https://www.nature.com/articles/s41467-022-31384-3)

**핵심**: 모델 시리즈를 통한 Shapley value 전파

$$\text{ψ}_i = \text{Attribution}_i \text{ where } \sum_{i} \text{ψ}_i = \Delta \text{Outcome}$$

**적용**: 
- Stacked generalization (앙상블)
- 신경망 feature extraction
- Mixed model types (tree + neural + linear)

**vs InfluenceNet**:
- 장점: 다층 모델 파이프라인 지원
- 한계: 정적 Shapley value만 처리, power index 미포함

#### B. BONES 벤치마크 (2024) [arxiv](http://arxiv.org/pdf/2407.16482.pdf)

**목표**: Neural Shapley Value 추정의 표준화

- 여러 신경망 추정자 비교
- 표준 벤치마크 데이터셋
- 평가 메트릭 통일

**vs InfluenceNet**:
- InfluenceNet: 특정 문제(MCN power index)에 특화
- BONES: 일반적 Shapley value 추정 프레임워크
- 상호 보완적 (InfluenceNet이 BONES의 special case 가능)

#### C. SHAPNN (2023) [arxiv](http://arxiv.org/pdf/2309.08799.pdf)

**목표**: Tabular 데이터에 Shapley value 정규화 적용

$$L_{\text{total}} = L_{\text{supervised}} + \lambda L_{\text{SHAP regularization}}$$

**vs InfluenceNet**:
- 유사성: 둘 다 tabular/structured data 처리
- 차이: SHAPNN은 모델 설명(interpretability), InfluenceNet은 power index 추정

### 4.3 Banzhaf Index의 부흥 (2021-2025)

#### A. Shapley vs. Banzhaf 비교 (2021) - Karczmarz et al. [arxiv](https://arxiv.org/pdf/2108.04126.pdf)

**결론**:

| 측면 | Shapley | Banzhaf |
|------|---------|---------|
| 직관성 | 복잡한 수열 기반 | 직관적: 투표 추가의 영향 |
| 계산 복잡도 | O(m!) | O(2^m) |
| 해석 | 기여도의 평균 | 기여도의 기대값 |
| ML 적용 | 해석성(SHAP) | 특성 중요도 |

$$\text{Banzhaf}_j = \mathbb{E}_C[\Delta_j | C]$$
$$\text{Shapley}_j = \frac{1}{m!} \sum_{\sigma} [v(C_j^\sigma) - v(C_j^\sigma \setminus \{j\})]$$

**InfluenceNet의 입장**: 둘 다 동등하게 처리하되, **계산 복잡도는 무관화**

#### B. Data Banzhaf (2023) - Wang & Jia [proceedings.mlr](https://proceedings.mlr.press/v206/wang23e/wang23e.pdf)

**혁신**: Maximum Sample Reuse (MSR) 원칙으로 data valuation에 Banzhaf 적용

$$\hat{\beta}_j^{\text{MSR}} = \frac{1}{|S|_j} \sum_{S \ni j} [v(S) - v(S \setminus \{j\})]$$

여기서 $|S|_j$는 j를 포함하는 subset의 수.

**실제 성능** (저자):
- Shapley value (SHAP) 대비 **더 robust** to noisy labels
- 계산 효율: **10배 이상 빠름**
- Feature ranking 정확도: 우수

**InfluenceNet과의 관계**:
- Data Banzhaf: MSR를 통한 **효율적 근사**
- InfluenceNet: **신경망을 통한 직접 예측**
- 보완 관계: MSR로 label 생성 후 NN으로 일반화 가능

#### C. Kernel Banzhaf (2024) - Liu et al. [arxiv](https://arxiv.org/pdf/2410.08336.pdf)

**혁신**: 회귀 기반 Banzhaf value 추정 (처음)

새로운 회귀 공식:

$$\vec{w} = \arg\min_{\vec{w}} \left\| K\vec{w} - \vec{v} \right\|^2$$

여기서 K는 Banzhaf 행렬.

**성과**:
- Monte Carlo 대비 **정확도 10배 향상**
- 샘플 효율: **9.9배 가속화**
- **이론적 보장** 제공

$$\text{error} = O(\sqrt{\frac{p}{n}} + \frac{\|\vec{\beta}\|_{\text{true}}}{n})$$

**vs InfluenceNet**:

| 측면 | InfluenceNet | Kernel Banzhaf |
|------|-------------|----------------|
| 방법 | 신경망 (비선형) | 커널 회귀 (선형) |
| 일반화 | empirical 평가 | 이론적 보장 |
| 구현 복잡도 | 중간 | 낮음 |
| 확장성 | m≤50 | m≤100+ 이론적 |
| 강점 | 대규모 시스템, 비선형 | 이론적 엄밀성 |
| 약점 | 이론적 보장 부재 | 작은 규모에만 최적 |

**통합 가능성**: 높음
- Kernel Banzhaf로 label 생성 → InfluenceNet으로 일반화
- 또는 InfluenceNet 예측값을 Kernel Banzhaf 입력으로 사용 가능

### 4.4 Graph Neural Networks & Power Indices (2024-2025)

#### Approximating Banzhaf Values via GNNs (2024) [arxiv](https://arxiv.org/html/2510.13391v1)

**핵심**: Network flow games에서 GNN을 통한 Banzhaf 근사

$$\vec{\beta} = \text{GNN}(\text{graph adjacency}, \text{edge weights})$$

**구조 활용**: 그래프의 위상(topology) 직접 활용

**vs InfluenceNet**:
- 구조화된 데이터(그래프): GNN 우수
- 비구조화 coalition rules: InfluenceNet 우수
- **직교 방향**: 서로 다른 문제 클래스

### 4.5 협력 게임 이론의 광범위 재평가 (2024-2025)

#### A. Beyond Shapley Values (2025) - Idrissi et al. [arxiv](https://arxiv.org/html/2506.13900v1)

**혁신적 메시지**: Shapley value에 과도하게 집중하지 말 것

**제시된 대안 할당(allocation) 클래스**:

1. **Weber Set**: Satisfying all lower/upper bounds
   $$\text{Weber}(v) = \{\vec{x} : x_i \geq v(\{i\}), \sum_i x_i = v(N) \text{ for some orderings}\}$$

2. **Harsanyi Set**: Coalition structure based
   $$\text{Harsanyi}(v) = \{\vec{x} : \text{consistent with some hierarchy}\}$$

**함의**:
- Shapley는 유일한 해가 아님
- 상황에 따라 다른 allocation이 더 적절할 수 있음
- InfluenceNet도 Shapley 외 다른 power index (Banzhaf, 새로운 지수) 확장 가능

#### B. DeepNeurogame (2024) - Bouchaffra et al. [arxiv](https://arxiv.org/pdf/2410.12264.pdf)

**아이디어**: 신경망 정규화에 게임 이론 적용

$$L = L_{\text{training}} + \lambda \sum_k \text{(neurons in winning coalitions)}_k$$

**메커니즘**:
- 각 layer를 cooperative game으로 해석
- 뉴런들의 coalition 형성
- Shapley value로 contribution 계산 → 강한 coalition 선택

**vs InfluenceNet**:
- InfluenceNet: power index **예측 작업** (output)
- DeepNeurogame: power index **활용한 정규화** (process)
- 상호 보완: DeepNeurogame이 InfluenceNet 예측값 이용 가능

### 4.6 계산 최적화의 진전

#### A. Accelerated Shapley Value (2023) [arxiv](https://arxiv.org/pdf/2311.05346.pdf)

**δ-Shapley 전략**: 작은 subset만 사용

$$\hat{Sh}_i^{\delta} = \frac{1}{N_\delta} \sum_{S: |S| \leq \delta} [v(S \cup \{i\}) - v(S)]$$

**결과**:
- 계산량: **9.9배 가속화**
- 값 정확도: 유지됨
- Pre-trained networks에서 추가 효율성

**vs InfluenceNet**:
- δ-Shapley: 특정 subset에만 관심 (구조적 제약)
- InfluenceNet: 모든 coalition 패턴 학습 (전체적 근사)

#### B. 학습 기반 성능 예측 (2022) [iclr](https://iclr.cc/virtual/2022/8403)

**아이디어**: 신경망으로 미지의 subset에 대한 모델 성능 예측

$$\hat{v}(S) = \text{NN}(S) \approx v_{\text{true}}(S)$$

그 후 이를 이용해 Shapley value 계산 가속화.

**이론적 기여**:
$$\text{ApproxError}(Sh) \leq f(\text{ApproxError}(v))$$

명시적 경계 제시.

**vs InfluenceNet**:
- 이 연구: **성능 함수** 근사 → power index 계산 가속
- InfluenceNet: **직접** power index 예측
- 철학적 차이: 간접 vs 직접

### 4.7 비교 분석 종합 표

| 연구 | 연도 | 방법 | 대상 | 강점 | 약점 | 계산 복잡도 |
|------|------|------|------|------|------|-----------|
| **InfluenceNet** | 2025 | NN (Feedforward) | MCN Power Index | 간결, 확장성, 속도 | 이론 부재, 분포 취약 | O(8분) |
| G-DeepSHAP | 2022 | NN (Deep propagation) | 모델 시리즈 Shapley | 계층 지원, 융통성 | 정적만 | O(fast) |
| BONES | 2024 | 벤치마크 | Shapley 추정 비교 | 표준화, 공개 | 구현 복잡 | Varies |
| SHAPNN | 2023 | NN + SHAP 정규화 | Tabular interpretability | 설명성 강화 | Shapley만 | O(training) |
| Data Banzhaf | 2023 | MSR 원칙 | Data valuation | Robust, 효율적 | 일반화 부정확 | O(sampling) |
| Kernel Banzhaf | 2024 | 커널 회귀 | Banzhaf 추정 | 이론 보장, 정확도 | 작은 규모만 | O(polynomial) |
| GNN 기반 | 2024 | Graph NN | Network flow games | 구조 활용 | 그래프만 | O(message passing) |
| Beyond Shapley | 2025 | 협력 게임 재평가 | 일반 allocation | 이론적 풍요성 | 실무 불명확 | Varies |
| DeepNeurogame | 2024 | Game-theoretic regularization | NN 정규화 | 신경망 개선 | 학습 오버헤드 | O(training) |

***

## 5. 논문이 향후 연구에 미치는 영향과 고려사항

### 5.1 이론적 영향

#### 1. 신경망의 Tabular 데이터 처리 재평가 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/628656f9-a7b3-4d07-a057-663dc1a34791/2503.08381v1.pdf)

기존 통념: "Neural networks are not suitable for tabular data" (Shwartz-Ziv & Armon, 2021) [apem-journal](http://apem-journal.org/Archives/2020/Abstract-APEM15-2_164-178.html)

**InfluenceNet의 기여**:
- MCN이라는 **구조화된 tabular 표현** 사용으로 신경망 효과성 입증
- 단순 tabular features가 아닌, **게임 이론적 구조를 인코딩**하면 신경망 유효

**함의**: 도메인 지식 통합이 신경망의 tabular 성능 결정 요인
$$\text{NN success} = f(\text{input structure}, \text{domain knowledge}, \text{architecture})$$

#### 2. Power Index 계산의 복잡도 하한 조정 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/628656f9-a7b3-4d07-a057-663dc1a34791/2503.08381v1.pdf)

전통: Power index 계산은 **intrinsically exponential**

**새로운 이해**: 
- 정확 계산: 여전히 exponential
- 근사 계산: **polynomial로 가능** (신경망 사용 시)

대가: 근사 오차 존재, 새로운 데이터 분포에 약함

#### 3. 다중 에이전트 시스템 이해의 확장 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/628656f9-a7b3-4d07-a057-663dc1a34791/2503.08381v1.pdf)

**전 (Before InfluenceNet)**:
- Power index 분석은 소규모 시스템(n ≤ 10)으로 제한
- 대규모 현실 시스템(회사 이사회, 국제 조직) 분석 불가능

**후 (After InfluenceNet)**:
- n ≥ 50 시스템 가능성 열림
- 새로운 응용 시나리오 탐색 기회

### 5.2 실무적 응용 영역

#### 1. 기업 지배구조 분석 (Corporate Governance)

복잡한 주식 소유권 구조:
- 보유 지분(shareholder)들의 실제 투표력 계산
- 모의 투표(dummy voting) 탐지
- 소수주주 권리 보호

$$\text{Actual Power}_i = f(\text{share ownership}, \text{governance rules})$$

**InfluenceNet 적용**:
- 시뮬레이션 시간: 50분 → 8분 (6배 가속)
- 대규모 포트폴리오 회사 네트워크 분석 가능

#### 2. 국제 정치 및 투표 제도 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/628656f9-a7b3-4d07-a057-663dc1a34791/2503.08381v1.pdf)

유럽 의회, UN, IMF 투표:
- 의석 수 ≠ 실제 권력 (강대국 연합의 영향력)
- 투표 규칙 개혁의 영향 평가

**예**: 
$$\beta_{\text{USA}} = \text{critical in how many coalitions?}$$

#### 3. 분산 시스템 및 블록체인 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/628656f9-a7b3-4d07-a057-663dc1a34791/2503.08381v1.pdf)

다중 서명(multisig) 지갑, DAO 거버넌스:
- 실제 거버넌스 권력 분포 평가
- 51% 공격 가능성, 카르텔 형성 위험 분석

#### 4. 데이터 가치 평가 (Data Valuation)

Data Banzhaf와의 결합:
- 머신러닝 파이프라인의 각 데이터 포인트 가치 평가
- 데이터 시장에서 공정한 가격 결정

### 5.3 향후 연구 시 고려사항

#### 단기 과제 (1-2년)

**1. 아키텍처 개선**

현재의 간단한 feedforward 한계를 극복:

```
제안 1: Attention Mechanism 추가
- MoG 분포 변화에 대한 적응력 강화
- Query: rule pattern
- Key/Value: coalition responses
- Output: adaptive Banzhaf/Shapley prediction

제안 2: Conditional Neural Network
- p-threshold나 분포를 explicit condition으로
- f(coalition_matrix | distribution_params)

제안 3: Ensemble Approach
- 여러 NN을 조합 (uncertainty estimate)
- Bayesian NN으로 confidence interval 제공
```

**2. 이론적 보장 추가**

Kernel Banzhaf처럼 approximation error bound 도출:

$$|\hat{Sh}_i^{\text{NN}} - Sh_i| \leq \epsilon(\text{architecture}, \text{dataset size}, m, n)$$

**3. 동적 시스템 확장**

시간에 따른 coalition 변화 모델링:

$$\text{Power}_i(t) = f(\text{Graph}(t), \text{Rules}(t))$$

예: LSTM/Temporal CNN 활용

#### 중기 과제 (2-3년)

**1. 하이브리드 방법론**

여러 접근법의 장점 통합:

```
Pipeline:
Data → Kernel Banzhaf (label) → NN (일반화) → Ensemble
```

- Kernel Banzhaf: 이론적 보장
- NN: 확장성
- Ensemble: 신뢰성

**2. 산업 파일럿 프로젝트**

실제 대규모 시스템에서 검증:
- 실제 corporate board의 power index 분석
- 예측과 실제 투표 행동 비교
- 모델 정확도 실증

**3. 인터프레터빌리티(Interpretability) 강화**

왜 특정 에이전트가 높은 power index를 가지는가?

```
SHAP를 InfluenceNet에 역적용:
각 rule의 contribution to power index 분석
```

#### 장기 과제 (3년+)

**1. 통합 프레임워크**

협력 게임 이론의 다양한 도구를 단일 플랫폼:
- Power indices (Banzhaf, Shapley, 새로운 지수)
- Stability concepts (Core, Nucleolus)
- Coalition structure learning

**2. 실시간 의사결정 지원**

동적 coalition formation:
- "만약 이 에이전트가 들어오면 power 어떻게 변함?"
- 온라인 강화학습과 결합

**3. 다중 학습 영역 통합**

Game Theory + Deep Learning의 더 깊은 융합:
- 신경망 자체를 game으로 해석 (DeepNeurogame 연장)
- Coalition learning in multi-agent RL

### 5.4 기술적 고려사항

| 측면 | 현재 상태 | 개선 필요 |
|------|---------|----------|
| **에이전트 수 일반화** | m=10-50 | m>100 지원 |
| **분포 강건성** | Uniform/Coin-flip 우수 | MoG 악화 (10배 오류) |
| **이론적 분석** | Empirical only | Error bound 필요 |
| **동적 성능** | Static only | Temporal extension |
| **계산 병목** | Label 생성 (초기) | 더 효율적 근사 필요 |
| **배포 용이성** | 높음 (표준 NN) | 낮음 (domain knowledge 필요) |

### 5.5 오픈 문제

1. **근본적 질문**: 신경망이 power index를 '이해'하는가? 아니면 단순히 패턴 매칭?
   - 수학적으로 입증 필요

2. **일반화 한계**: 왜 에이전트 수 변화에 극도로 민감한가?
   - Input dimensionality의 근본적 문제인지, 아키텍처 탓인지 미해결

3. **MoG의 실패**: 분포 변화 학습 불가능성의 이론적 근거는?
   - 신경망의 한계인가, 문제 설정의 한계인가?

4. **최적 모델 크기**: 512-256-128 아키텍처가 왜 최적인가?
   - Systematic hyperparameter search 부재

***

## 결론

InfluenceNet은 **협력 게임 이론의 계산 병목을 신경망으로 극복한 실질적 혁신**이다. Banzhaf와 Shapley-Shubik 전력지수 계산의 복잡도를 O(2^m, m!)에서 실무적으로 다루기 쉬운 수준으로 낮추었으며, 특히 대규모 다중 에이전트 시스템(n ≥ 10) 분석을 가능하게 했다.

주요 강점은 **단순성, 속도, 확장성**이다. 단순한 피드포워드 신경망으로도 우수한 성능을 달성했으며, 계산 시간을 8분으로 단축했고, n=50까지의 시스템에서 일반화 가능성을 보였다.

그러나 **이론적 보장 부재, 분포 변화에 대한 취약성, 에이전트 수 변화에 대한 극심한 민감성**이라는 명확한 한계도 있다. 특히 MoG 분포에서 10배 오류 증가는 실무 적용 시 심각한 제약이다.

2020년 이후의 관련 연구들(Data Banzhaf, Kernel Banzhaf, SHAPNN, G-DeepSHAP 등)과 비교하면, InfluenceNet은 **직접적 신경망 예측**이라는 독특한 접근을 취한다. Kernel Banzhaf는 이론적으로 더 견고하지만 확장성에서 뒤지고, SHAP 기반 방법들은 interpretability에 강하지만 power index 도메인이 아니다.

향후 연구는 다음을 우선순위로 해야 한다:
1. **Attention/Conditional 아키텍처**: 분포 강건성 개선
2. **이론적 경계 도출**: Kernel Banzhaf 수준의 근거 추가
3. **하이브리드 접근**: 신경망 + 커널 방법 결합
4. **동적 확장**: 시간 변화하는 coalition 대응

InfluenceNet은 **계산적으로는 탁월하지만 이론적으로는 불완전한, 실무 지향적인 contribution**이다. 향후 3-5년 동안 더 정교한 아키텍처와 이론적 분석으로 보완된다면, 대규모 거버넌스 시스템 분석의 표준 도구로 자리잡을 수 있을 것으로 예상된다.

***

## 참고 논문 인덱스
<span style="display:none">[^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48]</span>

<div align="center">⁂</div>

[^1_1]: 2503.08381v1.pdf

[^1_2]: https://www.nature.com/articles/s41467-022-31384-3

[^1_3]: http://arxiv.org/pdf/2407.16482.pdf

[^1_4]: http://arxiv.org/pdf/2309.08799.pdf

[^1_5]: https://arxiv.org/pdf/2108.04126.pdf

[^1_6]: https://proceedings.mlr.press/v206/wang23e/wang23e.pdf

[^1_7]: https://arxiv.org/pdf/2410.08336.pdf

[^1_8]: https://arxiv.org/html/2510.13391v1

[^1_9]: https://arxiv.org/html/2506.13900v1

[^1_10]: https://arxiv.org/pdf/2410.12264.pdf

[^1_11]: https://arxiv.org/pdf/2311.05346.pdf

[^1_12]: https://iclr.cc/virtual/2022/8403

[^1_13]: http://apem-journal.org/Archives/2020/Abstract-APEM15-2_164-178.html

[^1_14]: https://ieeexplore.ieee.org/document/11346475/

[^1_15]: https://www.semanticscholar.org/paper/c7efb861976b8c3a13c1ecf0a1e3cfd13a1c8184

[^1_16]: https://soil.copernicus.org/articles/6/389/2020/soil-6-389-2020-discussion.html

[^1_17]: https://www.semanticscholar.org/paper/0932abfd0fb90e8a28f7bd195633c9891bfd7ecb

[^1_18]: https://link.springer.com/10.1007/s10822-020-00314-0

[^1_19]: https://link.springer.com/10.1007/s00330-020-07312-8

[^1_20]: http://pubs.rsna.org/doi/10.1148/radiol.2020191160

[^1_21]: https://revistas.ucc.edu.co/index.php/in/article/view/3796

[^1_22]: https://link.springer.com/10.1007/s00521-020-05444-y

[^1_23]: https://arxiv.org/pdf/2202.05594.pdf

[^1_24]: https://arxiv.org/pdf/1904.02868.pdf

[^1_25]: https://arxiv.org/html/2311.01010v2

[^1_26]: https://arxiv.org/pdf/1903.10992.pdf

[^1_27]: https://arxiv.org/html/2207.07038v5

[^1_28]: https://www.cs.unc.edu/~livingst/Banzhaf/

[^1_29]: https://proceedings.mlr.press/v235/wang24an.html

[^1_30]: https://www.sciencedirect.com/science/article/abs/pii/S0957417425038473

[^1_31]: https://arxiv.org/html/2503.08381v1

[^1_32]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11631413/

[^1_33]: https://arxiv.org/pdf/2506.05281.pdf

[^1_34]: https://deepmind.google/research/publications/approximating-the-core-of-cooperative-games/

[^1_35]: https://openreview.net/forum?id=eBVCZj3RZN

[^1_36]: https://www.cs.huji.ac.il/~jeff/papers/aamas07bachrach.pdf

[^1_37]: https://dl.acm.org/doi/10.1145/3589334.3645599

[^1_38]: https://arxiv.org/abs/1711.04992

[^1_39]: https://purl.stanford.edu/xq291xs8637

[^1_40]: https://pdfs.semanticscholar.org/ae0a/6a2b344e2e11b0d3ea50e05c51a755d4036e.pdf

[^1_41]: https://arxiv.org/abs/2506.13900

[^1_42]: https://arxiv.org/abs/2311.10468

[^1_43]: https://arxiv.org/html/2410.08336v1

[^1_44]: https://pubmed.ncbi.nlm.nih.gov/41543342/

[^1_45]: https://arxiv.org/html/2502.09053v2

[^1_46]: https://arxiv.org/abs/1507.06105

[^1_47]: https://arxiv.org/abs/2509.02391

[^1_48]: https://www.sciencedirect.com/science/article/abs/pii/S1570870525002173
