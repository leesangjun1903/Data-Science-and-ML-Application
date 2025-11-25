
# Matrix Capsules with EM Routing

## 1. 핵심 주장 및 주요 기여 (간결 요약)

"Matrix Capsules with EM Routing"은 Geoffrey Hinton, Sara Sabour, Nicholas Frosst가 ICLR 2018에서 발표한 획기적인 논문으로, 기존 CNN의 한계를 극복하기 위한 새로운 신경망 구조를 제시합니다.[1]

**핵심 주장:**
- **캡슐(Capsule) 개념**: 뉴런의 집합이 특정 엔티티의 여러 특성(위치, 방향, 크기 등)을 벡터나 행렬로 인코딩해야 함
- **4×4 포즈 행렬**: 각 캡슐이 객체와 뷰어 간의 기하학적 관계를 나타내는 행렬 구조
- **EM 라우팅(Expectation-Maximization Routing)**: 계층 간 정보 전달을 확률적으로 최적화하는 알고리즘
- **부분-전체 관계 학습**: 변환 행렬을 통해 하위 캡슐의 자세를 상위 캡슐로 변환하는 파트-홀(part-whole) 관계 명시적 학습

**주요 기여:**
- smallNORB 벤치마크에서 **45% 오류 감소** (최고 기술 대비)
- **백색 상자 적대적 공격(white-box adversarial attack)에 대한 강건성 향상**
- **새로운 뷰포인트로의 일반화 능력 30% 향상** (매칭된 정확도 기준)
- 기존 CNN 대비 **파라미터 수 90-95% 감소**

***

## 2. 문제 정의, 해결 방법 및 모델 구조

### 2.1 해결하고자 하는 문제

**CNN의 근본적 한계:**

1. **풀링 레이어로 인한 정보 손실**: CNN은 평행이동 불변성을 위해 풀링을 사용하지만, 이는 객체의 정확한 위치, 방향, 크기 정보를 파괴
2. **스칼라 활성값의 한계**: 표준 신경망은 단일 활성값으로 피처를 표현하여 객체의 다양한 속성을 동시에 인코딩할 수 없음
3. **뷰포인트 변화에 취약성**: 이미지의 픽셀 강도는 뷰포인트 변화에 비선형적으로 반응하지만, 이를 체계적으로 다루지 않음

**핵심 직관**: 뷰포인트 변화는 포즈 행렬(pose matrix)에는 선형적 효과를 미친다는 기하학적 사실

### 2.2 제안 방법: EM 라우팅 알고리즘

#### 기본 구조

각 캡슐 i는 다음을 포함합니다:
- **포즈 행렬**: $$M_i \in \mathbb{R}^{4 \times 4}$$
- **활성도**: $$a_i \in $$ (시그모이드 함수로 출력되는 엔티티 존재 확률)[1]

인접한 캡슐 층(L과 L+1) 간의 연결:
- **변환 행렬**: $$W_{ij} \in \mathbb{R}^{4 \times 4}$$ (학습되는 가중치)
- **투표**: $$V_{ij} = M_i W_{ij}$$ (캡슐 i에서 캡슐 j로의 자세 예측)

#### EM 라우팅 절차

**Step 1: 확률 밀도 계산**

$h$번째 차원에서 캡슐 j의 가우시안 모델 하의 투표 확률:

$$P^h_{i|j} = \frac{1}{\sqrt{2\pi(\sigma^h_j)^2}} \exp\left(-\frac{(V^h_{ij} - \mu^h_j)^2}{2(\sigma^h_j)^2}\right)$$

이로부터 로그 확률:

$$\ln(P^h_{i|j}) = -\frac{(V^h_{ij} - \mu^h_j)^2}{2(\sigma^h_j)^2} - \ln(\sigma^h_j) - \frac{\ln(2\pi)}{2}$$

**Step 2: 코스트 함수**

차원 h에서 캡슐 j의 총 코스트:

$$cost^h_j = \sum_i \left[\ln(\sigma^h_j) + \frac{1}{2}\right] \left(\sum_i r_{ij}\right)$$

여기서 $r_{ij}$는 캡슐 i에서 j로의 할당 확률(assignment probability)

**Step 3: 활성도 함수**

$$a_j = \text{logistic}\left(\lambda\left[\beta_a - \beta_u \sum_i r_{ij} - \sum_h cost^h_j\right]\right)$$

- $\beta_a$: 캡슐 활성화 코스트 (학습됨)
- $\beta_u$: 균일 선행(uniform prior) 코스트 (학습됨)  
- $\lambda$: 역온도 파라미터 (고정 스케줄)

**Procedure 1: EM 라우팅 알고리즘**

```
Initialization: ∀i ∈ ΩL, j ∈ ΩL+1: Rij ← 1/|ΩL+1|

for t iterations do
  M-STEP(a, R, V, j) for ∀j ∈ ΩL+1:
    ∀i ∈ ΩL: Rij ← Rij * ai
    ∀h: μⁿj ← Σᵢ Rij * Vʰij / Σᵢ Rij
    ∀h: (σⁿj)² ← Σᵢ Rij(Vʰij - μⁿj)² / Σᵢ Rij
    costʰ ← [βu + log(σⁿj)](Σᵢ Rij)
    aj ← logistic(λ(βa - Σₕ costʰ))
    
  E-STEP(μ, σ, a, V, i) for ∀i ∈ ΩL:
    ∀j ∈ ΩL+1: pj ← 1 / √(∏ₕ 2π(σⁿj)²) 
                        * exp(-Σₕ (Vʰij - μⁿj)² / 2(σⁿj)²)
    ∀j ∈ ΩL+1: Rij ← (aj * pj) / Σₖ (aₖ * pₖ)
```

이 절차는 **3번 반복** (보통)하며, 각 이미지에 대해 매번 재계산됩니다.

### 2.3 모델 구조

#### 네트워크 아키텍처

**Fig. 1 구조:**

1. **입력**: 48×48 정규화 이미지
2. **Convolutional Layer**: 5×5, 32 채널, stride 2, ReLU
3. **Primary Capsule Layer**: 32개 캡슐 타입 (B=32)
   - 각 캡슐: 4×4 포즈 행렬
   - 하위 ReLU의 선형 변환으로 학습
4. **Convolutional Capsule Layer 1**: 3×3, 32 캡슐 타입 (C=32), stride 2
5. **Convolutional Capsule Layer 2**: 3×3, 32 캡슐 타입 (D=32), stride 1
6. **Final Capsule Layer**: 클래스별 캡슐 (5개 클래스)

#### 좌표 추가 (Coordinate Addition) 기법

마지막 컨볼루션 캡슐 층에서 최종 층으로의 연결 시:

$$V_{ij} = V_{ij} + \text{scale} \cdot \begin{pmatrix} row \\ col \\ 0 \\ 0 \end{pmatrix}$$

- 같은 캡슐 타입의 모든 위치에서 **공유되는 변환 행렬**
- 수용 필드 중심의 상대 위치 정보 추가
- 세밀한 위치 정보 보존

#### 손실 함수: Spread Loss

$$L_i = \max(0, m - (a_t - a_i))^2 \quad (i \neq t)$$

$$L = \sum_{i \neq t} L_i$$

- $a_t$: 정답 클래스의 활성도
- $m$: 마진 (0.2에서 0.9로 선형 증가)
- 클래스 활성도 간의 갭을 직접 최대화

***

## 3. 성능 향상 및 실험 결과

### 3.1 smallNORB 벤치마크 결과

| 모델 | 테스트 오류율 | 파라미터 수 |
|------|---------|----------|
| **우리 모델 (Matrix Capsules)** | **1.8%** | 310K |
| 소형 캡슐 모델 | 2.2% | 68K |
| Cireşan et al. (2011)* | 2.56% | 2.7M |
| 기준 CNN | 5.2% | 4.2M |
| 우리 모델 (멀티 크롭 테스트) | **1.4%** | - |

*추가 전처리 (필터링, 아핀 변환) 사용

**45% 오류 감소**: 2.56% → 1.4%[1]

### 3.2 라우팅 반복 횟수의 효과

| 반복 횟수 | 포즈 구조 | 손실 함수 | 좌표 추가 | 테스트 오류율 |
|----------|---------|---------|---------|----------|
| 1 | Matrix | Spread | Yes | 9.7% |
| **2** | **Matrix** | **Spread** | **Yes** | **2.2%** |
| **3** | **Matrix** | **Spread** | **Yes** | **1.8%** |
| 5 | Matrix | Spread | Yes | 3.9% |
| 3 | Vector | Spread | Yes | 2.9% |
| 3 | Matrix | Margin* | Yes | 3.2% |
| 3 | Matrix | CrossEnt | Yes | 5.8% |

**인사이트**:
- 3번의 반복이 최적 (2번도 합리적)
- **행렬 기반 포즈가 벡터보다 우수**
- Spread Loss가 다른 손실 함수보다 효과적

### 3.3 뷰포인트 일반화 성능

**설정**: 제한된 뷰포인트로 학습, 새로운 뷰포인트로 테스트

학습된 모델을 같은 정확도로 매칭한 후 비교:

| 시나리오 | CNN | Capsules | 개선율 |
|--------|-----|---------|-------|
| **신규 방위각** (친숙한 방위: 3.7%) | 13.5% | **6.2%*** | 30% ↓ |
| **신규 방위각** (친숙한 방위: 4.3%) | 17.8% | **12.3%** | 30% ↓ |
| **신규 고도** (친숙한 고도: 4.3%) | 12.5% | **8.8%** | 30% ↓ |

*친숙한 뷰포인트에서 CNN 성능 매칭

**결론**: Capsules은 **체계적인 기하학적 변환에 대해 훨씬 우수한 일반화**

### 3.4 적대적 견고성

#### FGSM (Fast Gradient Sign Method) 공격[1]

$$x^{adv} = x + \epsilon \cdot \text{sign}(\nabla_x L(f(x), y))$$

**결과 (ε = 0.2일 때)**:
- **CNN**: 정확도 급락 → 기회율(20%) 이하로 감소
- **Capsules**: 정확도 유지 → 항상 기회율 이상

**초점**: EM Routing의 구조적 특성으로 인한 고유한 견고성

#### 기본 반복 방법 (Basic Iterative Method) 공격

더 정교한 멀티-스텝 공격에서도 **Capsules이 CNN보다 훨씬 견고**

#### 평가: 기울기 포화 불검증

우려사항: 숫자 불안정성(numerical instability)으로 인한 거짓 견고성?

**검증 결과**:
- Capsules 그래디언트에서 0의 비율 < CNN의 비율
- 그래디언트 크기 차이: 단 2차수 (CNN-like 모델의 16차수 대비)
- **결론: 진정한 견고성 증명**

***

## 4. 모델의 일반화 성능 향상 가능성

### 4.1 일반화 능력의 이론적 근거

#### 기하학적 등변성(Equivariance)

포즈 행렬의 핵심 특성:

$$M^{new}_{i,viewer} = \text{Transformation} \times M^{old}_{i,viewer}$$

**효과**:
- 뷰포인트 변화가 **픽셀 강도에는 비선형적** → 포즈에는 **선형적**
- 시각 시스템이 이 선형성을 활용 → 기본적으로 **데이터 효율성 증대**

#### 부분-전체 관계 인코딩

Capsules는 다음을 명시적으로 학습:
- 하위 캡슐의 자세가 상위 캡슐의 자세로 **어떻게 변환되는가**
- 이는 기하학적 구성 원칙 학습 → **적은 샘플로도 강한 일반화**

### 4.2 실험 증거

#### 신규 뷰포인트 일반화[1]

```
학습 데이터: 아지무스 (300, 320, 340, 0, 20, 40)만 포함
테스트 데이터: 아지무스 (60-280) 포함
```

**결과**:
- CNN (매칭된 정확도): 친숙한 뷰에서 3.7%, 신규 뷰에서 13.5%
- **Capsules (매칭된 정확도)**: 친숙한 뷰에서 3.7%, 신규 뷰에서 **6.2%**
- **상대적 오류 감소: 30%**

**해석**: Capsules의 기하학적 자세 인코딩이 보이지 않은 뷰포인트에 **더 잘 전이**

#### MNIST 및 CIFAR-10 결과[1]

| 데이터셋 | 테스트 오류율 |
|---------|----------|
| MNIST | 0.44% |
| CIFAR-10 | 11.9% |

표준 설정에서도 경쟁력 있는 성능

### 4.3 파라미터 효율성과 일반화 연결성

**핵심 관찰**:

$$\text{Parameter reduction} = 90-95\% \quad \Rightarrow \quad \text{Generalization gain}$$

**이유**:
- **구조화된 제약**: 변환 행렬의 구조 (4×4) 자체가 기하학적 구조를 강제
- **공유 변환**: 위치별로 변환 행렬 공유 → 부분-전체 관계의 일관성 강제
- **오버피팅 감소**: 파라미터 수 감소 자체가 정규화 역할

***

## 5. 논문의 한계

### 5.1 구조적 한계

#### 1. 표현성 제약 (Expressivity Limitation)[2]

**이론적 발견**:

EM-라우팅과 라우팅-바이-어그리먼트(routing-by-agreement)는 **캡슐 네트워크의 표현성을 감소**시킵니다.

**증명**:
- 입력 $x$와 그 **음수 $-x$를 구분할 수 없음**
- **대칭 함수만 표현 가능** → 모든 함수를 근사할 수 없음 (Universal approximator 아님)

**영향**: 특정 비대칭적 객체 인식 작업에서 한계 가능성

#### 2. 스케일 제약

**smallNORB만 높은 성능**:
- MNIST: 0.44% (기준: ~0.3%)
- CIFAR-10: 11.9% (기준: ~5-6%)
- ImageNet: 구현 없음 (계산 복잡도)

**원인**: 행렬 연산의 계산 복잡도

#### 3. 라우팅 반복의 필요성

- **3번 반복 필수** → 정방향-역방향 패스 각 3회
- CNN의 단순 max-pooling에 비해 **계산량 증가**
- 1회 반복만으로는 성능 저하 (9.7% 오류율)

### 5.2 실제 구현의 어려움[3]

**재현성 문제**:
- 논문에 **공개 소스 코드 부재**
- 구현 세부사항이 상세하지 않음
- 연구자들의 재현 시도 중 많은 어려움 보고

### 5.3 계산 복잡도

#### 시간 복잡도 분석

각 라우팅 반복에서:
$$O(n_{L} \times n_{L+1} \times d \times iter)$$

- $n_L$: 하위 캡슐 수
- $n_{L+1}$: 상위 캡슐 수  
- $d$: 포즈 차원 (16)
- $iter$: 라우팅 반복 수 (3)

**결과**: CNN의 convolution에 비해 **10-100배 느림**

### 5.4 하이퍼파라미터 민감도

**학습 가능한 파라미터**:
- $\beta_a$: 캡슐 활성화 비용
- $\beta_u$: 균일 선행 비용

**고정 파라미터**:
- $\lambda$ (역온도): 고정 스케줄 필수 → 수동 조정 필요

**초기화 민감도**: 좋은 수렴을 위해 신중한 초기화 필요

***

## 6. 최신 연구 기반 영향 및 향후 고려사항

### 6.1 논문의 학술적 영향 (2018-2025)

#### 연구 방향 1: 효율성 개선[4][5][6]

**Efficient-CapsNet (2021)**[4]
- **Self-Attention 기반 라우팅** 도입
- 비반복적(non-iterative), 고도로 병렬화 가능한 라우팅
- 계산 복잡도 감소 + 성능 유지

**Towards Efficient CapsNet (2022)**[5]
- 희소성(sparsity) 활용으로 계산 효율성 증대
- 캡슐 수 감소 (90% 이상) + 높은 일반화 성능 유지
- 메모리 요구량, 훈련/추론 시간 대폭 감소

**Quick-CapsNet (2025)**[7]
- Primary Capsule(PC) 생성 방법 혁신
- Fully-Connected Layer로 특징 통합
- 기존 대비 **훈련 속도 10배 이상 개선**
- 파라미터 수 대폭 감소 (1152개 PC → 4-8개)

#### 연구 방향 2: 해석 가능성[8][9][10]

**Hierarchical Object-Centric Learning (2024)**[8]
- 라우팅 알고리즘 효과성 재검토
- 낮은 엔트로피 캡슐로 **더 명확한 파트-홀 관계** 추출
- 의료 영상 분야에 실제 적용 시연

**Capsule Networks Do Not Need to Model Everything (2025)**[9]
- **REM (Routing Entropy Minimization)** 제안
- 캡슐 네트워크가 관심 객체에만 초점 → parse tree 감소
- 불필요한 복잡도 제거로 효율성 + 해석성 동시 향상

**Interpretable Capsule Networks via Self-Attention Routing (2025)**[10]
- **SISA-CapsNet**: 공간 불변 자기 주의 라우팅
- 해석 가능한 공간 특징 인코딩
- 투명성과 성능의 좋은 균형

#### 연구 방향 3: 의료 및 실제 응용[11][12][8]

**의료 영상 응용**[11]
- Capsule-ConvKAN: Capsule + Kolmogorov-Arnold Network 하이브리드
- 조직병리 이미지 분류에서 CNN, CapsNet, ConvKAN 모두 능가
- 해석 가능성 + 정확도 동시 향상

**고장 진단 (Fault Diagnosis)**[12]
- **DPMI-CapsNet**: 다중 스케일 상호 정보 손실 통합
- 복합 고장 감지: 98.9% 정확도
- Pooling 제약 극복 + 로버스트한 표현 학습

#### 연구 방향 4: 기하학적 이해[13]

**Building Deep, Equivariant Capsule Networks (2019)**[13]
- 캡슐 네트워크의 등변성(equivariance) 증명
- **Space-of-Variation (SOV)** 개념: 각 캡슐 타입의 포즈 변화 다양체
- 깊은 캡슐 네트워크의 이론적 기초 제공

### 6.2 여전히 열려있는 과제

#### 1. 만능성(Universality) 문제

**현황**:
- Capsule networks는 **보편 근사기(universal approximator) 아님**[2]
- 대칭 함수만 표현 가능 → 비대칭적 객체에 한계

**필요한 연구**:
- 비대칭 함수 표현을 위한 라우팅 메커니즘 개선
- 엔트로피 기반 라우팅 등 새로운 접근

#### 2. 대규모 데이터셋 확장

**현황**:
- smallNORB: 1.8% 오류율 ⭐
- MNIST: 0.44% (경쟁력)
- CIFAR-10: 11.9% (약함)
- ImageNet: 미구현

**필요한 연구**:
- 계산 복잡도 획기적 감소
- GPU/TPU 최적화 알고리즘
- 프루닝(pruning) 기법 통합

#### 3. 라우팅 메커니즘 개선

**현황**:
- 3회 반복이 최적이지만 **정지 기준 명확하지 않음**
- 수렴 보장 미흡

**선도 연구**:
- 비반복적 라우팅 (Efficient-CapsNet)
- Self-Attention 기반 라우팅
- 엔트로피 최소화 (REM)

#### 4. 포즈 표현의 다양성

**현황**:
- 4×4 행렬이 표준 (논문 설정)
- 다른 포즈 표현의 효과 미검증

**가능한 연구**:
- 8×8, 16×16 등 큰 행렬 실험
- 벡터 대비 행렬의 이론적 우월성 증명
- 도메인별 최적 포즈 차원 연구

#### 5. 멀티모달 및 시계열 데이터 확장

**현황**:
- 정적 이미지 중심 연구
- 비디오, 3D, 음성 등 미흡

**유망한 방향**:
- 시공간 캡슐 네트워크
- 멀티모달 라우팅 메커니즘

### 6.3 향후 연구 시 고려사항

#### 1. 계산 효율성-성능 트레이드오프

**권고사항**:
```
프로젝트 성격별 선택
├─ 소규모 정확도 중시 (smallNORB)
│  └─ 표준 EM Routing (3회 반복)
├─ 실시간 응용 (AR/자율주행)
│  └─ Efficient-CapsNet (Self-Attention)
├─ 의료/해석성 중시
│  └─ Hierarchical + Entropy Minimization
└─ 대규모 데이터
   └─ 프루닝 + GPU 최적화 병행
```

#### 2. 하이퍼파라미터 최적화

**중요 파라미터**:
| 파라미터 | 기본값 | 조정 범위 | 민감도 |
|---------|------|--------|-------|
| $\lambda$ (역온도) | 고정 스케줄 | 0.1-1.0/iter | 높음 |
| 라우팅 반복 | 3 | 1-5 | 중간 |
| $\beta_a$, $\beta_u$ | 학습됨 | - | 낮음* |

*학습 가능하므로 초기값에 덜 민감

#### 3. 데이터셋 선택

**적합한 도메인**:
- ✅ **3D 객체 인식** (회전 불변성 필요)
- ✅ **의료 영상** (구조 중요)
- ✅ **Small-sample 학습** (일반화 우수)
- ❌ **텍스쳐 기반 분류** (포즈 정보 불필요)
- ❌ **초대규모 데이터** (계산 비용)

#### 4. 모델 검증 및 해석

**권고 평가 지표**:
1. 표준 정확도 (벤치마크 비교용)
2. **신규 뷰포인트 일반화** (Capsule 고유의 장점)
3. **적대적 견고성** (화이트박스 공격)
4. **파라미터 효율성** (계산 복잡도 대비)
5. **포즈 표현 해석성** (캡슐 활성도 시각화)

#### 5. 구현 참고사항

**주의점**:
- 원본 논문 **공개 코드 부재** → 신중한 재현 필요
- Tensorflow 구현 권장 (저자 사용)
- Batch normalization 대신 explicit covariance 학습 필수
- 초기화: Xavier/He initialization보다 **small random values** 권장

***

## 결론

**Matrix Capsules with EM Routing**은 신경망의 기본 패러다임에 도전하는 중요한 연구로, **부분-전체 기하학적 관계를 명시적으로 모델링하는 새로운 방식**을 제시했습니다.[1]

**주요 성과**:
- smallNORB에서 45% 오류 감소로 우월한 성능 증명
- 신규 뷰포인트 일반화에서 30% 개선으로 **기하학적 이해의 우수성** 입증
- 적대적 공격 견고성으로 구조적 우월성 증명

**현실적 제약**:
- **계산 복잡도 여전히 높음** → 최신 효율화 기법 적용 필수
- **표현성 제약** (비대칭 함수 불가) → 기초 이론 보강 필요
- **대규모 데이터 확장 난제** → 프루닝 + 아키텍처 혁신 병행

**향후 전망**:
2023-2025의 후속 연구들은 **효율성, 해석성, 실제 응용**에 집중하고 있으며, 특히 의료 영상과 소규모 데이터셋 환경에서 **CNN을 능가하는 결과**를 보이고 있습니다. 캡슐 네트워크는 더 이상 "미래 기술"이 아닌 **실용적 도구**로 진화 중이므로, 올바른 도메인과 최적화 기법 선택 시 **강력한 성능을 기대**할 수 있습니다.[12][11][8]

***

### 참고 문헌

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6d6a387a-3db5-4f8c-9b92-4c02ccfbe994/789_matrix_capsules_with_em_routin.pdf)
[2](https://arxiv.org/pdf/1905.08744.pdf)
[3](https://arxiv.org/pdf/1907.00652.pdf)
[4](https://arxiv.org/pdf/2101.12491.pdf)
[5](http://arxiv.org/pdf/2208.09203.pdf)
[6](https://arxiv.org/pdf/1907.06062.pdf)
[7](https://www.themoonlight.io/ko/review/quick-capsnet-qcn-a-fast-alternative-to-capsule-networks)
[8](https://arxiv.org/abs/2405.19861)
[9](https://www.sciencedirect.com/science/article/pii/S0031320325007794)
[10](https://www.nature.com/articles/s41598-025-96903-w)
[11](https://arxiv.org/html/2507.06417v1)
[12](https://www.sciencedirect.com/science/article/pii/S0957417425024327)
[13](https://arxiv.org/pdf/1908.01300.pdf)
[14](https://pmc.ncbi.nlm.nih.gov/articles/PMC5123693/)
[15](https://pmc.ncbi.nlm.nih.gov/articles/PMC9464680/)
[16](https://pmc.ncbi.nlm.nih.gov/articles/PMC5937845/)
[17](https://pmc.ncbi.nlm.nih.gov/articles/PMC10393802/)
[18](https://pmc.ncbi.nlm.nih.gov/articles/PMC6528812/)
[19](https://pubs.acs.org/doi/pdf/10.1021/acs.nanolett.7b04982)
[20](https://hyper.ai/kr/papers/matrix-capsules-with-em-routing)
[21](https://www.themoonlight.io/ko/review/mamba-capsule-routing-towards-part-whole-relational-camouflaged-object-detection)
[22](https://www.themoonlight.io/ko/review/fastcaps-a-design-methodology-for-accelerating-capsule-network-on-field-programmable-gate-arrays)
[23](https://blog.naver.com/sogangori/221129974140)
[24](https://itpe.jackerlab.com/entry/Capsule-Network-%EC%BA%A1%EC%8A%90-%EB%84%A4%ED%8A%B8%EC%9B%8C%ED%81%AC)
[25](http://t1.daumcdn.net/brunch/service/user/1oU7/file/oOell6Lmldt3UQlD6SRIdYSujSs.pdf?download)
[26](https://scienceon.kisti.re.kr/srch/selectPORSrchArticle.do?cn=NART126934598)
[27](https://jayhey.github.io/deep%20learning/2017/11/29/CapsNet_3/)
[28](https://brunch.co.kr/@kakao-it/158)
[29](https://arxiv.org/abs/1804.10172)
[30](https://arxiv.org/pdf/1805.04001.pdf)
[31](http://arxiv.org/pdf/1911.03451.pdf)
[32](https://arxiv.org/abs/2206.02664)
[33](https://www.themoonlight.io/ko/review/a-fast-3-approximation-for-the-capacitated-tree-cover-problem-with-edge-loads)
[34](https://ksas.or.kr/proceedings/2024c/data/%EB%B3%84%EC%B2%A85.%202024%EB%85%84%EB%8F%84%EC%9A%B0%EC%A3%BC%ED%95%99%EC%88%A0%EB%8C%80%ED%9A%8C_%EB%85%BC%EB%AC%B8%EC%A7%91_All.pdf)
[35](https://jayhey.github.io/deep%20learning/2017/11/28/CapsNet_2/)
[36](https://koreascience.kr/article/JAKO201723839836649.pdf)
