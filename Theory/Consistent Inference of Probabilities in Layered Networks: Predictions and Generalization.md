# Consistent Inference of Probabilities in Layered Networks: Predictions and Generalization

***

## 1. 핵심 주장 및 주요 기여

이 논문은 계층화된 신경망의 학습을 통계적 역학(statistical mechanics) 관점에서 재해석함으로써 신경망의 일반화 능력을 이해하기 위한 혁신적인 프레임워크를 제시한다.[1]

**핵심 주장**:
- 오류 최소화(error minimization)와 우도 최대화(likelihood maximization)를 동치로 만드는 **일관성 조건(consistency condition)**을 도입[1]
- 이 일관성 조건으로부터 유도되는 **깁스 분포(Gibbs distribution)**가 학습된 신경망 앙상블의 정규 정준 분포(canonical ensemble)를 나타냄[1]
- 훈련 집합에서 계산 가능한 **예측 오류(prediction error)**가 실제 일반화 능력과 높은 상관관계를 보임[1]

**주요 기여**:
1. 신경망 학습에 통계적 역학 이론을 적용한 최초의 형식적 프레임워크[1]
2. 훈련 오류와 일반화 능력 사이의 불일치 문제에 대한 해결책 제시[1]
3. 최적 네트워크 아키텍처 선택을 위한 실용적 기준 제공[1]

***

## 2. 해결하고자 하는 문제, 제안 방법 및 모델 구조

### 2.1 핵심 문제

신경망 학습의 기본적 딜레마:[1]
- 훈련 오류 최소화와 일반화 성능 간의 괴리
- 훈련 집합에서만 계산 가능하면서 일반화 능력을 반영하는 지표 부재
- 신경망의 고도로 비선형인 구조로 인한 직접 분석 불가능

### 2.2 제안 방법: 일관성 조건

논문은 다음 함수 방정식을 만족하는 확률 분포를 찾는다:[1]

$$\prod_{i=1}^{m} p(x_i | \omega) = \Phi \left(\prod_{i=1}^{m} e(x_i | \omega)\right)$$

이 조건은 강한 제약을 부과하며, 유일한 해는 다음과 같다:[1]

$$p(x|\omega) = \frac{1}{z(\beta)} \exp[-\beta e(x|\omega)]$$

여기서:
- $$\beta$$: 오류에 대한 확률의 민감도를 결정하는 양의 상수[1]
- $$z(\beta) = \int_{X \times Y} \exp[-\beta e(x|\omega)]dx$$: 정규화 상수[1]
- 평균 오류 $$\bar{e} = \int e(x|\omega)p(x)dx$$는 "허용 가능한 오류 수준"을 나타냄[1]

### 2.3 깁스 분포와 모델 구조

훈련 후 신경망 가중치의 확률 분포:[1]

$$p_m(\omega | x^{(m)}) = \frac{1}{\mathcal{Z}_m} p_0(\omega) \exp[-\beta E_m(\omega)]$$

여기서:
- $$E_m(\omega) = \sum_{i=1}^{m} e(x_i|\omega)$$: 훈련 집합에 대한 총 오류[1]
- $$\mathcal{Z}_m = \int_W p_0(\omega) \exp[-\beta E_m(\omega)] d\omega$$: 분할 함수[1]
- $$p_0(\omega)$$: 사전 분포[1]

### 2.4 신경망 아키텍처

피드포워드 계층 신경망 구조:[1]

$$u_i^{(l+1)} = \sum_{j=1}^{N_l} w_{ij}^{(l+1)} v_j^{(l)} + w_i^{(l+1)}$$

$$v_i^{(l+1)} = g(u_i^{(l+1)}) \quad (1 \leq i \leq N_{l+1})$$

여기서:
- $$g(x) = \frac{1}{1+\exp(-x)}$$: 시그모이드 활성화 함수[1]
- $$w_{ij}^{(l)}$$: 계층 l의 연결 가중치[1]
- $$L$$: 처리 계층 수[1]

***

## 3. 모델의 일반화 성능 향상 메커니즘

### 3.1 예측 확률과 일반화

훈련 후 새로운 입력-출력 쌍 x에 대한 예측 확률:[1]

$$p_m(x | x^{(m)}) = \int_W p_m(\omega | x^{(m)}) p(x|\omega) d\omega$$

예측 오류 (음의 로그 예측 확률):[1]

$$g_m(x) = -\log p_m(x | x^{(m)})$$

**핵심 발견**: 훈련 집합 내에서 계산된 예측 오류는 훈련 집합 외부의 실제 일반화 능력과 매우 높은 상관관계를 보임[1]

### 3.2 일반화 성능에 대한 한계

예측 오류는 사전 및 사후 훈련 오류로 한계지어짐:[1]

$$G_m \approx \log \frac{\mathcal{Z}_m}{\mathcal{Z}_{m-1}} - \log z$$

이는 다음과 같이 분해된다:[1]

- 상한: 사전 훈련 오류들의 합
- 하한: 평균 훈련 오류와 관련된 항

### 3.3 학습 곡선 이론

일반화 히스토그램 h_m(ρ)을 통한 학습 곡선 분석:[1]

$$h_m(\rho) = \frac{1}{\mathcal{Z}_{m-1}} \frac{\mathcal{Z}_m}{\mathcal{Z}_{m+1}} h_{m-1}(\rho)$$

단순 예제 (균등 히스토그램 + α 분수의 완벽한 네트워크)에서:[1]

$$m_c \approx \log\alpha / \log\beta_0$$

에서 급격한 전이가 발생하며, 임계 훈련 크기는 일반화의 갑작스러운 개선과 일치함[1]

### 3.4 정보론적 해석

엔트로피 변화를 통한 정보 획득:[1]

$$S_m = D_{\text{KL}}(p_m || p_0) = -\log \mathcal{Z}_m - \bar{E}_m \frac{d\log\mathcal{Z}_m}{d\beta}$$

- 접근 가능한 구성 공간의 부피 감소
- 시스템 자유 에너지의 단조 증가
- 훈련 오류 분산: $$\sigma_E^2 = \frac{d\bar{E}_m}{d\beta} = -\frac{d^2\log\mathcal{Z}_m}{d\beta^2}$$[1]

***

## 4. 성능 향상 및 한계

### 4.1 실험 결과: 연접성 문제 (Contiguity Problem)

**문제 설정**: 10비트 이진 패턴에서 연속된 1의 개수(2개 vs 3개)를 구별[1]

**주요 발견**:

| 지표 | 훈련 오류 | 예측 오류 | 실제 일반화 |
|------|---------|---------|----------|
| 상관관계 | 약함 | 강함 | - |
| 조기 종료 시점 | 신뢰성 낮음 | 높음 | ✓ |

- 예측 오류는 훈련 집합 내에서만 계산되었으나 실제 일반화 능력과 높은 상관성[1]
- 수용 영역 너비(receptive width) ρ=1, 5는 예측 오류가 높게 유지
- ρ=2, 3은 약 50% 훈련 크기 이후 급격한 예측 오류 감소[1]

### 4.2 최적 아키텍처 선택

Figure 4 결과:[1]
- 예측 오류를 통해 다양한 아키텍처의 일반화 능력을 구별 가능
- 훈련 오류만으로는 불가능한 차별화 달성
- 임계 훈련 크기가 명확하게 식별됨

### 4.3 한계 및 제약

**이론적 한계**:
1. 단순한 예제(균등 히스토그램)에 대해서만 해석적 학습 곡선 도출[1]
2. 일반화 히스토그램의 초기 분포에 대한 가정 필요[1]
3. 실제 신경망 훈련에서 깁스 분포가 정확히 구현되지 않을 수 있음[1]

**실용적 한계**:
1. 계산 복잡도: 전체 네트워크 앙상블에 대한 정규화 상수 계산 어려움
2. 확장성: 대규모 네트워크에 대한 직접 적용 불가능
3. 정규화 매개변수 β 설정의 불명확성[1]

**방법론적 한계**:
1. 오류 함수의 형태에 제약 (매끄럽고 양수)
2. 사전 분포 p₀의 선택이 결과에 미치는 영향 미충분 분석[1]
3. 종료 조건(stopping criterion)의 명확하지 않은 정의

***

## 5. 모델의 일반화 성능 향상 가능성 (중점 분석)

### 5.1 예측 확률의 이론적 토대

일반화 능력을 측정하는 방식의 혁신:[1]

$$\text{일반화 점수} \propto \sum_{t=0}^{T-1} \log p_t(x^{(t)})$$

where p_t는 처음 t개 훈련 샘플 후의 분포. 이 점수는:
- 훈련 집합만으로 계산 가능
- 실제 일반화 오류의 하한 제공
- 베이즈 하한보다 정보 이론적으로 타당[1]

### 5.2 깁스 앙상블의 이점

**다양한 관점에서의 일반화**:
1. **확률론적 관점**: 다양한 가중치 구성의 가중치 평균 → 과적합 감소
2. **정보론적 관점**: 엔트로피 최대화를 통한 최적 압축
3. **통계 역학적 관점**: 열역학적 평형 상태의 안정성[1]

### 5.3 아키텍처 선택을 통한 일반화 향상

**원칙**:
$$\text{최적 아키텍처} = \arg\min_{\text{architecture}} E[g_m(x)]$$

where E는 훈련 집합 외부의 데이터에 대한 기댓값[1]

**실무적 적용**:
- 연접성 문제에서 ρ=2, 3이 ρ=1, 5보다 우월
- 과도한 네트워크 용량(ρ=1)은 학습 곡선 악화
- 부족한 용량(ρ=5)도 동일 효과[1]

### 5.4 데이터 크기와 일반화

학습 곡선 분석으로부터:[1]

$$m_c \propto \log(1/\alpha)$$

여기서 α는 올바른 함수를 구현할 수 있는 네트워크의 비율. 따라서:
- 필요 훈련 샘플 수는 로그적으로 증가
- 아키텍처와 문제 특성에 따라 크게 변함
- 임계점 이후 급격한 성능 개선[1]

***

## 6. 2020년 이후 최신 연구와의 비교 분석

### 6.1 깁스 분포의 진화

**1989년 Tishby 원본**:[1]
$$p_m(\omega | x^{(m)}) = \frac{1}{\mathcal{Z}_m} p_0(\omega) \exp[-\beta E_m(\omega)]$$

**2020년 Physical Review X - "Statistical Mechanics of Deep Linear Neural Networks"**[2]
- Tishby의 깁스 분포 유지하면서 역전파 커널 정규화(BPKR) 도입
- 계층별 가중치 적분을 통한 일반화 오류의 정확한 계산
- 깊이, 너비, 정규화의 영향을 명시적으로 분석

| 측면 | 1989 | 2020 BPKR |
|-----|------|-----------|
| 모델 | 비선형 | 선형 (단순화) |
| 깁스 분포 | ✓ | ✓ |
| 계층별 분석 | 암묵적 | 명시적 |
| 일반화 경계 | 상한/하한 | 정확한 계산 |

### 6.2 정보 이론적 경계의 진화

**최신 연구: "Information-Theoretic Generalization Bounds for DNNs" (2024)**[3]

**계층적 KL 발산 경계** (Theorem 1):

$$\text{Gen}\_L \leq \frac{1}{m}\left(\sum_{l=1}^{L} \text{KL}(p_{l,\text{test}} || p_{l,\text{train}})\right)$$

이는 Tishby의 단순 경계를 다음과 같이 확장:
- 각 계층의 표현 분포를 명시적으로 포함
- 깊이의 이점을 직접 보여줌 (더 깊은 계층의 경계가 더 타이트)
- 강한 데이터 처리 부등식(SDPI)을 통한 정량화

**바서슈타인 거리 기반 경계** (Theorem 2):

$$\text{Gen} \leq \min_l W_1(P_l^{\text{train}}, P_l^{\text{test}})$$

새로운 개념: **일반화 깔때기 계층(generalization funnel layer)**
- 실제 일반화 성능을 결정하는 특정 계층 존재
- 네트워크마다, 훈련 방법마다 다름[3]

### 6.3 정보 병목 이론과의 연결

**원본 Tishby의 예측 확률**:
- 비정보론적 접근
- 경험적 상관성 제시

**2023-2025 정보 병목 확장**:

**Kawaguchi et al. (2023)** - "How Does Information Bottleneck Help Deep Learning?"[4]
- IB 원칙과 일반화 오류 경계의 수학적 연결 증명
- 정보 병목은 일반화 제어의 충분조건이지만 필요조건은 아님

**Generalized Information Bottleneck (2025)**[5]
$$\min_{p(z|x)} \left[I(X;Z) - \beta I(Y;Z) + \gamma \text{Synergy}(X \to Z \gets Y)\right]$$

이전의 무한 복잡성 문제 해결:
- Synergy 항 추가로 공동 특성 처리 정량화
- 압축 단계 더 명확하게 식별[5]

### 6.4 통계 역학적 관점의 재해석

**Balsubramani (2024) - "Entropy, Concentration, and Learning"**[6]

Tishby의 통계 역학 프레임워크를 제1원리부터 재구축:

| 개념 | Tishby 1989 | Balsubramani 2024 |
|------|------------|------------------|
| 기초 | 깁스 분포 | Boltzmann의 확률 계산 |
| 엔트로피 | 자유 에너지 변화 | 거시/미시 상태의 축중성 |
| 일반화 | 예측 확률 | 지수 족의 정보 사영 |
| 공식화 | 오류 함수 → 확률 | 손실 최소화 → 최대 엔트로피 |

**핵심 등식** (Sanov 정리의 확장):

$$\frac{1}{m}\log\Pr(\hat{P}_m \in A) = -D_{\text{KL}}(P^*_A || P)$$

여기서 $$P^*\_A = \arg\min_{Q \in A} D_{\text{KL}}(Q||P)$$는 **정보 사영**(information projection)

**Tishby와의 관계**:
- Tishby의 p_m(ω)는 가중치 공간의 정보 사영
- Balsubramani의 P^*_A는 표현/특성 공간의 정보 사영
- 동일한 수학적 원리의 다양한 적용[6]

### 6.5 신경 붕괴와 동역학

**최신 발견 (2025)**: "Explaining Grokking and Information Bottleneck through Neural Collapse"[7]

Tishby의 정적 일반화 분석 → 동적 학습 과정 분석:

**신경 붕괴(Neural Collapse)의 역할**:
$$\text{Grokking} = \text{클래스 내 분산 축소} = \text{IB 압축 단계}$$

증명된 연결 (Theorem 3.4):

$$I(Z;X)\_{\text{excess}} \leq \text{Var}_{\text{within-class}}$$

**함의**:
- Tishby의 예측 확률이 암묵적으로 신경 붕괴와 연결
- 분류기가 동일 클래스 샘플을 가까이 당길수록 일반화 향상
- 장-시간 척도에서 나타나는 grokking 현상도 설명[7]

### 6.6 비교표: 연구 발전

| 차원 | 1989 | 2020 | 2024-2025 |
|-----|------|------|----------|
| **이론적 토대** | Gibbs 앙상블 | Gibbs + 커널 이론 | 통계역학 1원리 |
| **일반화 경계** | 경험적 상관 | 정확 계산 | 계층별 계약 정량화 |
| **정보론** | 암묵적 | IB 원칙 | 일반화 IB |
| **아키텍처** | 단순 2계층 | 선형 심층망 | 비선형 임의 깊이 |
| **동역학** | 학습 곡선 | 가중치 업데이트 | 신경 붕괴, Grokking |
| **적용** | 아키텍처 선택 | 모델 검사 | 신경망 검증, 안전성 |

***

## 7. 한계점 (Limitations)

### 7.1 이론적 한계

1. **강한 가정들**:
   - 오류 함수의 매끈함과 양성 조건
   - 사전 분포의 일반적 형태 미지정
   - 일관성 조건이 모든 실제 훈련 알고리즘에 적용되는지 불명확[1]

2. **학습 곡선의 제한된 적용**:
   - 단순한 히스토그램 가정 (균등 + α 분율 완벽 네트워크)
   - 일반적인 문제 클래스로의 확장 불명확[1]

3. **정규화 상수 계산**:
   - 실제 신경망에서 분할 함수 $$\mathcal{Z}_m$$ 계산 불가능[1]

### 7.2 실무적 한계

1. **계산 불가능성**:
   - 대규모 신경망의 경우 전체 네트워크 앙상블 다루기 불가능
   - 예측 확률의 실제 계산 방법 미제시[1]

2. **확장성 문제**:
   - 연접성 문제는 10비트 이진 분류 (매우 단순)
   - 현대 딥러닝의 고차원 데이터에 적용 불명확[1]

3. **매개변수 설정**:
   - β(민감도) 값 결정 방법 제시 안 함
   - 실제 훈련에서의 β와의 관계 불명확

***

## 8. 향후 연구에 미치는 영향 및 고려사항

### 8.1 근본적 이론적 기여

**신경망 학습의 통계 역학화**:
- Tishby (1989)는 신경망을 처음으로 물리 시스템처럼 분석
- 2020-2025 연구들이 이를 정교화
- 현재 깊은 학습의 수학적 기초로 작용[8][2][6]

**정보 이론과의 통합**:
- 정보 병목 이론으로 발전 (Tishby et al., 2000)
- 2023-2025에 정보론적 경계와 엄밀하게 연결
- 복합 시스템의 이해에 새로운 관점 제공[4][5]

### 8.2 기술적 고려사항

#### 8.2.1 현대 신경망에의 적용

**원칙적 접근**:
- 깊은 신경망에 대해 깁스 분포는 여전히 유효한 앙상블 모델
- 하지만 실제 확률 계산 위해 근사 필요

**근사 전략**:
1. **변분 근사**: 깁스 분포의 평균장(mean-field) 근사
2. **샘플링 기반**: 마르코프 연쇄 몬테카를로 또는 확산 모델
3. **점근 근사**: 큰 데이터 극한에서의 경계

#### 8.2.2 아키텍처 설계

**Tishby의 제안 (1989) → 현대적 발전**:
- 수용 영역 너비 대신 → 신경망 폭/깊이 비율
- 예측 오류 기준 → Wasserstein 거리의 "일반화 깔때기 계층" 식별[3]

**실무 권장사항**:
```
for each candidate_architecture:
    compute generalization_funnel_layer
    estimate_gen_error = compute_wasserstein_at_funnel_layer
select architecture with minimum generalization error
```

#### 8.2.3 학습 동역학

**Tishby의 정적 분석 → 동적 분석으로의 진화**:

신경 붕괴 관점에서:[7]

$$\frac{d}{dt}\text{Var}\_{\text{within-class}} \approx -\alpha(t) \cdot \nabla_{\theta} L(\theta)$$

이는:
- Grokking 시간 척도 설명
- 압축 단계 정량화 가능
- 조기 종료 최적 시점 예측 가능[7]

### 8.3 미해결 문제 및 향후 방향

#### 8.3.1 우선순위 높은 문제

1. **경계의 타이트함**:
   - 정보론적 경계가 종종 실제 오류보다 3자리 큼[9]
   - 더 타이트한 경계 유도 필요

2. **상호 정보 추정**:
   - 실제 I(Z;X), I(Z;Y) 계산의 신뢰성 문제
   - 고차원 설정에서 편향된 추정[5]

3. **비독립동일분포(Non-i.i.d.) 설정**:
   - 대부분의 이론이 i.i.d. 가정
   - 시계열, 그래프 데이터, 분포 이동에 대한 이론 부족[10]

#### 8.3.2 새로운 응용 분야

1. **강건성 보장**:
   - IB 복잡도 항 ↔ 적대적 취약성[5]
   - 실용적인 검증 프로토콜 개발

2. **모델 검사(Model Inspection)**:
   - 신경 붕괴 지표를 통한 감춘 취약성 탐지[11]
   - 신뢰할 수 있는 AI의 기초

3. **전이 학습 이론**:
   - 정보 사영의 관점에서 도메인 적응 분석
   - 사전 학습의 이론적 정당화

### 8.4 방법론적 권장사항

#### 전통적 신경망 훈련 시:
1. 훈련 오류로만 모니터링 ✗ → 예측 확률/IB 복잡도 함께 모니터링 ✓
2. 조기 종료 기준 → 일반화 깔때기 계층의 신호 활용
3. 아키텍처 선택 → 검증 집합 대신 정보론적 기준 고려

#### 심층망 훈련 시:
1. **신경 붕괴 모니터링**:
   ```python
   within_class_var = compute_within_class_variance(representations)
   # 감소하는 패턴 = 일반화 개선의 신호
   ```

2. **계층별 정보 흐름**:
   - 각 계층에서 I(Z;Y)는 유지, I(Z;X)는 감소하는지 확인
   - 불균형하면 훈련 동역학 조정[7]

3. **Grokking 예측**:
   - 신경 붕괴 시간 척도 추정으로 grokking 시점 예측
   - 장시간 훈련의 필요성을 미리 판단

***

## 결론

Tishby, Levin, Solla (1989)의 "Consistent Inference of Probabilities in Layered Networks"는 신경망 학습을 통계 역학의 관점에서 재해석한 획기적 논문이다. 깁스 분포를 통한 네트워크 앙상블의 도입과 예측 확률을 일반화 지표로 제시한 아이디어는 35년 후 2020-2025년의 최신 연구에서도 중추적 역할을 하고 있다.[8][2][6][4][3][5][1]

특히:
- **2020 깁스 분포의 정교화**: 깃허브 BPKR을 통한 정확한 일반화 오류 계산[2]
- **2024 정보론적 경계**: 계층별 KL 발산 및 Wasserstein 거리를 통한 명시적 깊이 이점 증명[3]
- **2025 신경 붕괴**: Tishby의 정적 분석이 동적 학습 과정과 어떻게 연결되는지 규명[7]

그러나 남은 과제들—실제 계산 불가능성, 경계의 느슨함, 비i.i.d. 설정에서의 이론 부족—은 향후 연구에서 우선적으로 해결해야 할 문제들이다. 특히 강건성, 모델 검사, 전이 학습 등의 실무 응용 분야에서 이론을 더욱 정교화할 필요가 있다.

***

### 참고문헌

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4498c144-0f17-4bba-8dff-b01351cf3dff/TishbyLevinSolla89.pdf)
[2](https://link.aps.org/doi/10.1103/PhysRevX.11.031059)
[3](https://arxiv.org/html/2404.03176v1)
[4](https://proceedings.mlr.press/v202/kawaguchi23a/kawaguchi23a.pdf)
[5](https://arxiv.org/html/2509.26327v1)
[6](http://arxiv.org/pdf/2409.18630.pdf)
[7](https://arxiv.org/html/2509.20829v1)
[8](https://arxiv.org/abs/2501.19281)
[9](https://arxiv.org/html/2410.22371v1)
[10](https://www.nature.com/articles/s42005-024-01837-w)
[11](https://arxiv.org/abs/2508.07397)
[12](https://www.semanticscholar.org/paper/27f95de30bccbce00d07db556893f046ac7f4005)
[13](https://iopscience.iop.org/article/10.1088/1742-5468/adf822)
[14](https://www.semanticscholar.org/paper/0f0533c18d4e49eab159de5090fd84c219efceb5)
[15](https://www.nature.com/articles/s41467-021-23103-1)
[16](https://www.semanticscholar.org/paper/5844cfc72c1b75c544ff487a3737de6305df8935)
[17](https://www.semanticscholar.org/paper/8f499e296ccf520b8e1b9d91478dffe24b610abb)
[18](https://www.semanticscholar.org/paper/2b8415acafd88368b3e54d650e54b660112be1e6)
[19](http://arxiv.org/pdf/2402.05916.pdf)
[20](http://arxiv.org/pdf/2501.19281.pdf)
[21](http://arxiv.org/pdf/2209.01610.pdf)
[22](https://arxiv.org/pdf/1811.12889.pdf)
[23](http://arxiv.org/pdf/2408.00082.pdf)
[24](https://www.pnas.org/doi/pdf/10.1073/pnas.2311805121)
[25](http://arxiv.org/pdf/2211.01258.pdf)
[26](https://www.nature.com/articles/s41467-022-34305-6)
[27](https://en.wikipedia.org/wiki/Information_bottleneck_method)
[28](https://ganguli-gang.stanford.edu/pdf/20.StatMechDeep.pdf)
[29](https://www.joseluismontielolea.com/Best__worst_case__Linear_Predictors_2024.pdf)
[30](https://pmc.ncbi.nlm.nih.gov/articles/PMC7764901/)
[31](https://www.sciencedirect.com/science/article/abs/pii/S0925231224014723)
[32](https://openreview.net/pdf?id=EoebmBe9fG)
[33](https://www.tencentcloud.com/techpedia/100169)
[34](https://arxiv.org/pdf/2409.13904.pdf)
[35](https://www.sciencedirect.com/science/article/abs/pii/S0022509624003740)
[36](https://www.sciencedirect.com/science/article/abs/pii/S0893608023002009)
[37](https://openaccess.thecvf.com/content_ICCVW_2019/papers/SDL-CV/Elad_Direct_Validation_of_the_Information_Bottleneck_Principle_for_Deep_Nets_ICCVW_2019_paper.pdf)
[38](https://www.annualreviews.org/content/journals/10.1146/annurev-statistics-040522-013920)
[39](http://www.kdiss.org/journal/view.html?doi=10.7465%2Fjkdi.2023.34.6.893)
[40](https://www.reddit.com/r/MachineLearning/comments/be8qie/discussion_what_is_the_status_of_the_information/)
[41](https://arxiv.org/pdf/2507.14336.pdf)
[42](https://www.biorxiv.org/content/10.1101/2024.09.16.613342v3.full-text)
[43](https://arxiv.org/pdf/2508.11004.pdf)
[44](https://arxiv.org/html/2506.14831v2)
[45](https://arxiv.org/html/2404.03176v2)
[46](https://pubmed.ncbi.nlm.nih.gov/40422407/)
[47](https://arxiv.org/pdf/2504.20571.pdf)
[48](https://arxiv.org/html/2504.06328v1)
[49](https://arxiv.org/html/2510.23935v1)
[50](https://arxiv.org/abs/2509.26327)
[51](https://arxiv.org/html/2510.06684v1)
[52](https://arxiv.org/pdf/2504.08489.pdf)
[53](https://arxiv.org/abs/2507.19832)
[54](https://arxiv.org/pdf/2502.01146.pdf)
[55](https://arxiv.org/html/2409.01498v1)
[56](https://arxiv.org/html/2412.08222v1)
[57](https://arxiv.org/html/2508.11004)
[58](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0296904)
