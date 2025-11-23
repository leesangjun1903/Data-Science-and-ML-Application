# Symbolic Discovery of Optimization Algorithms

### 1. 핵심 주장 및 주요 기여 요약

**Symbolic Discovery of Optimization Algorithms**는 프로그램 탐색(program search)을 통해 신경망 학습용 최적화 알고리즘을 자동으로 발견하는 새로운 방법론을 제시합니다. 구글과 UCLA의 연구팀이 발표한 이 논문의 핵심 기여는 다음과 같습니다:

**주요 기여:**

1. **프로그램 검색 방법론**: 무한하고 희소한 프로그램 공간에서 고품질의 최적화 알고리즘을 효율적으로 탐색하기 위한 기법 개발
2. **일반화 전략**: 프록시 태스크(proxy tasks)에서 목표 태스크(target tasks)로의 큰 일반화 간격을 해소하기 위한 펀넬 선택(funnel selection)과 프로그램 단순화 전략 도입
3. **Lion 알고리즘 발견**: 간단하면서도 효과적인 새로운 최적화 알고리즘 **Lion(EvoLved Sign Momentum)** 발견
4. **광범위한 실증적 검증**: 이미지 분류, 비전-언어 대조학습(vision-language contrastive learning), 확산 모델(diffusion models), 언어 모델링 등 다양한 작업에서 우수한 성능 입증

***

### 2. 해결하는 문제, 제안 방법, 모델 구조 및 성능

#### 2.1 해결하는 문제

기존의 최적화 알고리즘(Adam, AdamW, Adafactor 등)은 수년간 수동으로 설계되어 왔으나, 더 효율적이고 범용적인 알고리즘 발견이 어려운 상황입니다. 이전의 자동 최적화 알고리즘 발견 방법들(L2O, AutoML-Zero 등)은 제한된 검색 공간이나 작은 태스크에서만 작동하여 실제 대규모 신경망 학습에 적용할 수 없었습니다.

**주요 문제점:**
- 무한하고 희소한 프로그램 공간에서의 효율적 탐색
- 프록시 태스크와 실제 목표 태스크 간의 거대한 일반화 간격(최대 10,000배 규모 차이)
- 발견된 알고리즘의 다양한 아키텍처, 데이터셋, 도메인으로의 전이 가능성

#### 2.2 제안 방법

**2.2.1 프로그램 검색 공간(Program Search Space)**

논문에서 정의한 프로그램은 n차원 배열을 다루는 NumPy/JAX 스타일의 명령형 언어입니다. 프로그램의 구조:

$$\text{Program}:\ \text{train}(w, g, v_1, v_2, lr) \rightarrow \text{update}$$

여기서:
- $$w$$: 모델 가중치
- $$g$$: 기울기
- $$v_1, v_2$$: 추가적 상태 변수(예: 모멘텀 추적용)
- $$lr$$: 학습률 스케줄 값

프로그램은 45개의 기본 수학 함수(절댓값, 삼각함수, 지수/로그 함수, 부호 연산 등)로 구성되며, 선형보간 함수 $$\text{interp}(x, y, a) = (1-a)x + ay$$를 포함합니다.

**2.2.2 진화적 탐색(Evolutionary Search)**

정규화된 진화(regularized evolution)를 따뜻한 시작(warm-start)과 재시작(restart)으로 강화:

1. **따뜻한 시작**: 초기 모집단을 AdamW로 초기화하여 탐색 가속화
2. **토너먼트 선택**: 모집단에서 무작위로 선택된 $$T$$ 개 알고리즘 중 최적 성능을 가진 것을 부모로 선택
3. **돌연변이**: 무작위 위치에 명령문 삽입, 명령문 삭제, 함수/인자 수정
4. **재시작 전략**:
   - 초기 프로그램에서 재시작: 탐색 공간 다양화
   - 최적 프로그램에서 재시작(300K 진행 후): 착취(exploitation) 강화

**2.2.3 추상 실행(Abstract Execution)**

프로그램 공간 가지치기(pruning)를 위한 3단계 필터링:

1. **타입/형태 추론**: 프로그램 오류 검출
   - 변수의 타입과 형태 자동 추론
   - 함수 시그니처와 인자 유형 검증

2. **함수 해싱**: 의미론적으로 동일한 프로그램 제거
   - 각 명령문의 해시값 계산: $$H = \text{hash}(\text{function}, \text{args})$$
   - 해시 테이블로 캐싱하여 중복 계산 제거 (약 10배 비용 감소)

3. **중복 명령문 식별**: 최종 출력에 영향을 주지 않는 명령문 제거
   - 의존성 추적으로 프로그램 길이 평균 3배 단축

**성능**: 캐시 히트율 89.1%, 중복 명령문 69.8% (최종 프로그램에서)

**2.2.4 프록시 태스크 및 탐색 비용**

- 프록시 태스크: 모델 크기, 데이터 샘플, 학습 스텝을 축소(예: ViT 3-레이어, 96 은닉 유닛, ImageNet의 10%)
- 평가 시간: 1개 TPU V2에서 20분
- 탐색 비용: 100개 TPU V2 × 72시간 = 3,000 TPU V2 일
- 생성 프로그램 수: 200-300K 개 중 실제 평가: 20-30K 개

#### 2.3 일반화 전략: 메타-오버피팅과 펀넬 선택

**메타-오버피팅 현상**:

프록시 태스크에서의 탐색 적합도는 계속 증가하지만 메타-검증 성능이 감소하는 현상 관찰.

$$\text{Meta-overfitting occurs when}: \text{search fitness} \uparrow \text{ while } \text{meta-validation metric} \downarrow$$

**관찰**: 메타-오버피팅이 늦게 발생하는 탐색 실험에서 발견된 알고리즘이 더 잘 일반화됨을 발견.

**펀넀 선택 전략**:

$$\text{Task A (proxy)} \rightarrow \text{Task B (10×larger)} \rightarrow \text{Task C (100×larger)} \rightarrow \text{Task D (target)}$$

각 단계에서 기준선을 초과하는 알고리즘만 다음 단계로 평가하여 계산 효율성 극대화.

#### 2.4 프로그램 단순화

세 단계 단순화 절차:

1. **중복 명령문 제거**: 추상 실행으로 식별된 비영향 명령문 제거
2. **미미한 명령문 제거**: 제거 시 성능 저하가 무시할 수 있는 명령문 제거
3. **수동 정리**: 명령문 재배열, 변수 이름 단순화, 수학적 동등 형태로 변환

**예시**: Program 8 (원본) → Program 4 (중복 제거) → Program 1 (Lion 최종형)

#### 2.5 Lion 알고리즘 구조

**Algorithm 2: Lion 의사코드**

$$
\begin{align}
c_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \quad \text{(현재 기울기와 모멘텀 보간)} \\
\theta_t &= \theta_{t-1} - \alpha \cdot \text{sign}(c_t) \quad \text{(부호 기반 가중치 업데이트)} \\
m_t &= \beta_2 m_{t-1} + (1-\beta_2) g_t \quad \text{(모멘텀 EMA 업데이트)} \\
\theta_t &= \theta_t - \lambda \cdot \theta_{t-1} \quad \text{(가중치 감쇠)}
\end{align}
$$

여기서:
- $$\beta_1 = 0.9, \beta_2 = 0.99$$ (기본값)

$$\text{sign}(x) = \begin{cases} +1 & \text{ if } x > 0 \\ -1 & \text{if } x < 0 \\ 0 & \text{if } x = 0 \end{cases}$$

- $$\lambda$$: 가중치 감쇠 강도

**핵심 차이점**:

| 측면 | Lion | Adam | AdamW |
|------|------|------|-------|
| 추적 상태 | 모멘텀만 ($$m_t$$) | 1차, 2차 모멘트 ($$m_t, v_t$$) | 1차, 2차 모멘트 |
| 업데이트 규칙 | $$\text{sign}(c_t)$$ | $$\frac{m_t}{\sqrt{v_t}+\epsilon}$$ | $$\frac{m_t}{\sqrt{v_t}+\epsilon}$$ |
| 메모리 효율성 | **2배** | 기준 | 기준 |
| 업데이트 크기 | 모든 차원에서 균일 ($$\pm 1$$) | 차원별 가변 | 차원별 가변 |

#### 2.6 성능 향상 결과

**2.6.1 이미지 분류**

*ImageNet에서 스크래치부터 학습:*

| 모델 | 매개변수 | AdamW | Lion | 향상도 |
|------|---------|-------|------|--------|
| ResNet-50 | 25.56M | 76.34% | 76.45% | +0.11% |
| ViT-B16 | 86.57M | 80.12% | 80.77% | +**0.65%** |
| ViT-L16 | 304.72M | 85.07% | 85.59% | +**0.52%** |
| CoAtNet-3 | 166.97M | 84.45% | 84.87% | +**0.42%** |

*JFT-300M 사전학습 후 ImageNet 미세조정:*

- ViT-L16: 이전 ViT-H14 (AdamW)와 동등하나 **3배 사전학습 비용 절감**
- ImageNet ReaL: **5배 계산 절감**
- ViT-G14: 1.8배 더 적은 매개변수로 이전 SOTA 초과

**2.6.2 비전-언어 대조학습 (BASIC-L)**

$$\text{Zero-shot ImageNet accuracy} = 88.3\% \text{ (Adafactor: } 85.7\%)$$
$$\text{Fine-tuning accuracy} = 91.1\% \text{ (이전 SOTA: } 91.0\%)$$

다양한 데이터셋에서 일관된 개선:
- ImageNet V2: +0.6%
- ImageNet-A: +6.08%
- ImageNet-R: +5.54%

**2.6.3 확산 모델**

256×256 이미지 생성:
- FID 점수: Lion 4.1 vs AdamW 4.7 (참고: ADM 10.94)
- 수렴 속도: **2.3배 반복 횟수 감소** (AdamW 440K 스텝과 동등)

**2.6.4 언어 모델링**

*작은 규모(Wiki-40B, PG-19):*
- Lion: AdamW보다 일관되게 낮은 검증 난해도(perplexity)
- 중간 크기 모델: **1.6-2배 훈련 속도 향상**

*대규모(1.1B-7.5B 매개변수):*
- 난해도 측면: 동등 성능 (거대 고품질 데이터셋)
- 인-컨텍스트 학습: 약간 우수

*T5 미세조정 (GLUE):*
- T5-Base: 10/12 태스크 승리
- T5-Large: 12/12 태스크 승리
- T5-11B: 10/12 태스크 승리

***

### 3. 일반화 성능 향상 분석

#### 3.1 손실 경관(Loss Landscape) 분석

**핵심 발견**: Lion은 더 평평한 손실 영역으로 수렴합니다.

측정 지표 - 경관 평탄성 $$L_N^{\text{train}} = E_N[\text{Loss}_{\text{train}}(\theta + w)]$$:

| 모델 | 훈련 오류 | 경관 평탄성 |
|------|---------|-----------|
| AdamW | 0.61 | 3.74 |
| Lion | 0.75 | **1.37** |

**해석**: Lion은 약간 높은 훈련 오류(0.14 차이)를 허용하지만 **2.7배 더 평평한** 손실 경관에 도달하여 더 나은 일반화를 달성합니다.

#### 3.2 부호 연산의 정규화 효과

**부호 연산의 규칙화 메커니즘**:

부호 연산 $$\text{sign}(c_t)$$은 연속적 업데이트를 이산적 ±1 업데이트로 변환하면서:

1. **노이즈 추가**: 양자화로 인한 노이즈는 암묵적 정규화로 작용
2. **모멘텀 평탄화**: 기울기 방향 정보만 유지하고 크기 정보는 제거
3. **배치 크기 효과**: 큰 배치에서 기울기 추정의 분산이 감소하여 부호 연산의 신뢰도 향상

**실증 증거**: ImageNet ViT-B16 학습:
- 훈련 오류: Lion **더 높음** (0.14 차이)
- 검증 정확도: Lion **더 높음** (+2%)
- 일반화 간격: 훨씬 **더 작음**

#### 3.3 배치 크기 의존성

**배치 크기 실험** (ViT-B16, ImageNet):

| 배치 크기 | AdamW | Lion | 우월성 |
|----------|-------|------|--------|
| 64 | 78.5% | 77.9% | AdamW 우위 (-0.6%) |
| 256 | 79.2% | 79.1% | 거의 동등 |
| 4,096 | 80.12% | 80.77% | **Lion 우위** (+0.65%) |
| 32,768 | 75.4% | **77.9%** | **+2.5%** |

**핵심 통찰**:
$$\text{Lion 성능 향상} \propto \text{배치 크기}$$

가설: 큰 배치에서는 기울기 추정이 더 안정적이므로 부호 연산이 모멘텀 방향을 더 정확하게 보존합니다.

#### 3.4 모멘텀 추적 구조

**두 개의 보간 팩터의 필요성**:

| 요소 | 목적 | 값 | 해석 |
|------|------|-----|------|
| $$\beta_1 = 0.9$$ | 현재 기울기에 높은 가중치 | $$(1-\beta_1) = 0.1$$ | 즉각적 반응 |
| $$\beta_2 = 0.99$$ | 모멘텀에 높은 가중치 | $$(1-\beta_2) = 0.01$$ | 10배 장기 역사 유지 |

$$c_t = 0.9 \cdot m_{t-1} + 0.1 \cdot g_t \quad \text{(업데이트용)}$$
$$m_t = 0.99 \cdot m_{t-1} + 0.01 \cdot g_t \quad \text{(다음 반복용)}$$

**절제 실험** (ViT-B16, ImageNet):

| 설정 | ImageNet | ReaL | V2 |
|------|----------|------|-----|
| Ablation ($$\beta=0.9$$) | 79.54% | 85.10% | 68.07% |
| Ablation ($$\beta=0.99$$) | 79.90% | 85.36% | 68.20% |
| **Lion ($$\beta_1=0.9, \beta_2=0.99$$)** | **80.77%** | **86.15%** | **69.19%** |

두 팩터 모두 필요하며, 이들의 조합이 균형 잡힌 성능을 제공합니다.

#### 3.5 하이퍼파라미터 견고성

**학습률($$\alpha$$)과 가중치 감쇠($$\lambda$$) 민감도**:

| 측면 | AdamW | Lion |
|------|-------|------|
| 최적 배치 크기 | 256 | 4,096 |
| 학습률 범위 | 넓음 | 좁음 (3-10배 더 작은 값) |
| 가중치 감쇠 | $$\lambda_{\text{eff}} = \alpha \cdot \lambda$$ | 3-10배 더 큰 값 필요 |
| 초하이퍼파라미터 |$$\alpha, \lambda, \epsilon$$ | $$\alpha, \lambda, \beta_1, \beta_2$$ |

**견고성 결과** (Figure 8 히트맵):
- Lion: 하이퍼파라미터 변화에 더 둔감 (선택 폭 넓음)
- AdamW: 더 민감한 응답성

***

### 4. 한계 및 제약사항

#### 4.1 탐색 방법의 한계

**1. 검색 공간 편향:**
- 1차 최적화 알고리즘에 의한 암묵적 편향 존재
- 2차 알고리즘(K-FAC, Shampoo 등) 구성에 필요한 함수 부재

**2. 탐색 비용:**
- 3,000 TPU V2일의 상당한 계산 비용
- 알고리즘 단순화에 수동 개입 필요

**3. 프로그램 구조의 단순성:**
- 조건문, 루프, 함수 정의 등 고급 구조의 이점 미발견
- 미래 연구 방향으로 제시

#### 4.2 Lion 알고리즘의 한계

**1. 작은 배치 크기:**
$$\text{배치 크기} < 64 \Rightarrow \text{Lion 성능 이점 미미 또는 역전}$$

**2. 강한 정규화 하에서 성능 감소:**
- 강력한 데이터 증강(RandAug, Mixup) 사용 시 이점 감소
- 예: CoAtNet-3 (강한 정규화) 시 Lion 향상도 +0.42% (vs. ViT +0.65%)

**3. 대규모 고품질 데이터셋에서 동등 성능:**
- Imagen 기본 모델 (64×64): 명백한 개선 없음
- 대규모 내부 언어 모델 데이터셋의 난해도: 동등 수준
- 원인: 마진 있는 충분한 학습 신호

**4. 메모리 오버헤드:**
- bfloat16 모멘텀 추적은 여전히 메모리 비용 존재
- 극대규모 모델: 모멘텀 인수분해 필요 (향후 연구)

**5. 통계적 유의성 미달:**
- 일부 대규모 언어/이미지-텍스트 데이터셋에서 개선이 통계적으로 유의하지 않음

***

### 5. 최신 연구 동향 및 향후 고려사항 (2024-2025)

#### 5.1 Lion의 이론적 분석 진전

**2025년 최신 연구 (Jiang & Zhang, 2025)**

중앙집중식 설정에서의 수렴률 분석:

$$\text{표준 Lion: } \mathcal{O}(d^{1/2}T^{-1/4})$$
$$\text{분산 감소 Lion: } \mathcal{O}(d^{1/2}T^{-1/3})$$

여기서 $$d$$는 문제 차원, $$T$$는 반복 횟수.

**분산 설정에서의 수렴률:**
$$\text{표준 분산 Lion: } \mathcal{O}(d^{1/2}(nT)^{-1/4})$$
$$\text{분산 감소 분산 Lion: } \mathcal{O}(d^{1/2}(nT)^{-1/3})$$

여기서 $$n$$은 노드 수.

**통신 효율적 변형:**

```math
\mathcal{O}\left(\max\left\{\frac{d^{1/4}}{T^{1/4}}, \frac{d^{1/10}}{n^{1/5}T^{1/5}}\right\}\right)
```

#### 5.2 분산 학습 최적화

**2024년 주요 기여:**

1. **Distributed Lion (2024, OpenReview)**
   - 부호 연산 활용으로 이진/저정밀 벡터 통신만 필요
   - 대역폭: 기존 Adam 대비 **크게 감소**
   - 성능: 동등 수준 유지

2. **Lion Cub (2024.11)**
   - 분산 설정에서 통신 오버헤드 최적화
   - 모멘텀 동기화 최소화로 **5배 엔드-투-엔드 속도 향상**

3. **FedLion (2024.02)**
   - 연합학습(federated learning) 프레임워크에 Lion 적응
   - 느린 수렴 문제 해결

4. **Dion (2025.04)**
   - 통신 효율적 옵티마이저
   - 직교정규화된 업데이트와 장치-로컬 모멘텀 버퍼 활용

#### 5.3 Lion의 일반화 구조화

**Lion- $$\kappa$$ 패밀리 (2025.06)**

컨벡스 분석과 Lyapunov 안정성 원리로 일반화:

$$\theta_t = \theta_{t-1} - \alpha \cdot (\kappa \circ \text{sign})(c_t) - \lambda \theta_{t-1}$$

여기서 $$\kappa$$는 컨벡스 함수 또는 운동 맵.

**의미:**
- $$\kappa(x) = x$$ (선형): 부호 기반 Lion
- $$\kappa(x) = \text{프로젝션}(x, \mathcal{C})$$ (제약): 기하학적 제약 최적화
- 다양한 구조에 적응 가능한 통일 프레임워크

#### 5.4 자동 알고리즘 발견의 발전

**2025년 LLM 기반 알고리즘 발견**

AlphaResearch (2025.09): 자율 연구 에이전트를 통한 알고리즘 발견

$$\text{제안} \rightarrow \text{검증 (실행 기반 + 피어 리뷰 환경)} \rightarrow \text{최적화}$$

- O4-mini LLM 에이전트 활용
- 개방형 알고리즘 문제 해결
- 기존 방법(OpenEvolve, ShinkaEvolve) 초과 성능

**FunBO (2024.07)**: 베이즈 최적화 획득 함수 자동 발견
- LLM 기반 설계로 문제별 최적 획득 함수 발견

#### 5.5 향후 연구 방향 및 고려사항

**1. 검색 공간 확대:**
- 2차 알고리즘 구성 가능한 함수 추가
- 조건문, 루프, 중첩 함수 등 고급 프로그램 구조 포함
- 검색 편향 감소를 위한 더 일반적 프로그램 표현

**2. 효율성 개선:**
- 탐색 비용 감축 (현재 3,000 TPU V2일)
- 프로그램 단순화의 자동화 (현재 수동 개입)
- 메타 학습을 통한 프록시-목표 간극 축소

**3. 적응적 최적화:**
- 작은 배치 크기 성능 향상
- 강정규화 환경에 특화된 알고리즘 발견
- 도메인/작업별 맞춤형 옵티마이저

**4. 분산 확장성:**
- 극대규모 모델(수 조 매개변수)에 대한 최적화
- 통신 효율성 극대화 (Lion Cub, Dion 방향 계속)
- 이기종 환경(엣지-클라우드)에서의 최적화

**5. 이론-실무 간극 해소:**
- 수렴 이론과 실제 성능 간 불일치 분석
- 일반화 경계(generalization bounds) 개선
- 손실 경관 기하학과의 연결

**6. 크로스 도메인 평가:**
- 강화학습, 메타학습, 그래프 신경망 등 새로운 도메인
- 매우 큰 배치 설정(>32K)에서의 성능
- 비컨벡스 최적화 이론 틀에서의 엄밀한 분석

***

### 결론

**Symbolic Discovery of Optimization Algorithms**는 프로그램 탐색을 통한 자동 최적화 알고리즘 발견의 새로운 패러다임을 제시합니다. Lion 알고리즘은 간단성, 메모리 효율성, 광범위한 작업 범위에서의 우수한 성능으로 인해 실무에 배포되고 있으며(Google 검색 광고 CTR 모델), 후속 연구에서 분산 학습, 연합학습, 통신 효율성 등 다양한 확장을 통해 그 영향력이 확대되고 있습니다.

그러나 작은 배치 크기, 강한 정규화, 거대 데이터셋에서의 제한, 그리고 검색 공간 편향 등의 한계가 존재하며, 향후 연구는 이러한 제약을 극복하고 더욱 일반화된 알고리즘 발견 프레임워크를 구축하는 데 집중할 필요가 있습니다. 특히 2024-2025년의 이론적 분석, 분산 최적화 확장, 그리고 LLM 기반 자동 발견 방법의 성장은 AI 시스템의 훈련 효율성 개선을 위한 중요한 기반을 마련하고 있습니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8b8fb4f1-848c-4e49-819a-82d8715e37a9/2302.06675v4.pdf)
[2](https://arxiv.org/pdf/2302.06675.pdf)
[3](https://arxiv.org/html/2406.04824)
[4](http://arxiv.org/pdf/2410.12942.pdf)
[5](http://arxiv.org/pdf/2405.10976.pdf)
[6](http://arxiv.org/pdf/2405.18884.pdf)
[7](http://arxiv.org/pdf/2308.12644.pdf)
[8](https://arxiv.org/pdf/1912.09237.pdf)
[9](https://arxiv.org/pdf/2201.01441.pdf)
[10](https://syncedreview.com/2023/02/21/google-ucla-formulate-algorithm-discovery-as-program-search-yielding-lion-for-sota-dnn-optimization/)
[11](https://kaanberke.hashnode.dev/this-new-optimizer-called-lion-could-replace-adam-as-the-go-to-for-training-neural-nets)
[12](https://pmc.ncbi.nlm.nih.gov/articles/PMC12004515/)
[13](https://openreview.net/forum?id=ne6zeqLFCZ&noteId=TpyCxUvwr7)
[14](https://arxiv.org/pdf/2508.12327.pdf)
[15](https://www.snowflake.com/en/fundamentals/automl/)
[16](https://arxiv.org/html/2511.08522v1)
[17](https://www.emergentmind.com/topics/lion-mathcal-k-family-of-optimizers)
[18](https://mobidev.biz/blog/future-machine-learning-trends-impact-business)
[19](https://www.directagents.com/seo/the-search-revolution-how-ai-optimization-is-redefining-digital-discovery-in-2025-and-beyond/)
[20](https://arxiv.org/pdf/2411.16462.pdf)
[21](https://arxiv.org/pdf/2404.00438.pdf)
[22](https://arxiv.org/pdf/2411.07724.pdf)
[23](http://arxiv.org/pdf/2402.09941.pdf)
[24](https://arxiv.org/html/2504.05295v1)
[25](https://arxiv.org/pdf/2403.02589.pdf)
[26](https://arxiv.org/pdf/2303.00039.pdf)
[27](https://arxiv.org/html/2409.12392v1)
[28](https://arxiv.org/abs/2508.12327)
[29](https://pmc.ncbi.nlm.nih.gov/articles/PMC10386236/)
[30](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf)
[31](https://openreview.net/forum?id=wDirCeTIoz)
[32](https://proceedings.neurips.cc/paper/2020/file/a3a3e8b30dd6eadfc78c77bb2b8e6b60-Paper.pdf)
[33](https://openaccess.thecvf.com/content/ICCV2021/papers/Yuan_DecentLaM_Decentralized_Momentum_SGD_for_Large-Batch_Deep_Training_ICCV_2021_paper.pdf)
[34](https://chatpaper.com/paper/181593)
[35](https://theaisummer.com/regularization/)
[36](https://www.reddit.com/r/MachineLearning/comments/18uh79r/d_momentum_and_batch_size/)
