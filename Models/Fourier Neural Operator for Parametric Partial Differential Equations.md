
# Fourier Neural Operator for Parametric Partial Differential Equations

## 1. 논문 개요 및 핵심 주장

"Fourier Neural Operator for Parametric Partial Differential Equations"는 2021년 ICLR에 발표된 Li et al.의 획기적 논문으로, 편미분방정식(PDE)을 풀기 위한 신경망 기반 솔루션을 제시한다. 이 논문의 핵심 주장은 **무한차원 함수공간 간의 매핑을 학습할 수 있는 신경 연산자를 설계함으로써, 매개변수에 따라 달라지는 PDE 전체 패밀리를 한 번의 학습으로 해결할 수 있다**는 것이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0bc85d4d-c7f5-42fa-b7c8-fe500e525afe/2010.08895v3.pdf)

기존의 PDE 해결 방법은 매개변수가 변경될 때마다 새로운 계산을 수행해야 하는 한계가 있었다. FNO는 이를 극복하고, 학습된 단일 신경망 모델로 다양한 초기조건과 매개변수에 대해 빠르게 솔루션을 제공할 수 있다. 이러한 접근의 주요 이점은:

- **1000배 이상의 계산 속도 향상** (전통적 PDE 솔버 대비)
- **Resolution-invariant 성능** (학습 해상도와 무관하게 일관된 정확도)
- **Zero-shot super-resolution** (고해상도 데이터 없이도 고해상도에서 평가 가능)
- **기존 ML 방법 대비 30-60% 정확도 향상** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0bc85d4d-c7f5-42fa-b7c8-fe500e525afe/2010.08895v3.pdf)

***

## 2. 핵심 문제 정의

### 2.1 전통적 방법의 한계

기존의 유한요소법(FEM)이나 유한차분법(FDM)은 **해상도-정확도 트레이드오프**를 가진다. 저해상도 격자는 빠르지만 부정확하고, 고해상도 격자는 정확하지만 느리다. 특히 항공기 설계 등에서 매개변수별로 수천 번의 PDE 평가가 필요한 경우, 전통 솔버는 계산적으로 비실용적이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0bc85d4d-c7f5-42fa-b7c8-fe500e525afe/2010.08895v3.pdf)

### 2.2 기존 신경망 기반 방법의 한계

**Finite-dimensional 신경망 (CNN)**
- 특정 해상도에만 훈련 가능
- 다른 해상도에서는 성능이 저하되거나 재훈련 필요
- 학습 데이터 범위 내의 점에서만 평가 가능

**Neural-FEM (Physics-Informed Neural Networks)**
- 각 PDE 인스턴스마다 새로운 신경망 훈련 필요
- 매개변수 변경 시 완전한 재계산 요구
- PDE의 해석적 형태를 알아야 함 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0bc85d4d-c7f5-42fa-b7c8-fe500e525afe/2010.08895v3.pdf)

**Graph Neural Operators (GNO/MGNO)**
- Nyström 샘플링으로 인한 계산 비효율
- 복잡한 문제에서 수렴성 문제 (특히 난류) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0bc85d4d-c7f5-42fa-b7c8-fe500e525afe/2010.08895v3.pdf)

***

## 3. 제안하는 방법: Fourier Neural Operator

### 3.1 이론적 기초

FNO는 함수공간 간의 연산자 학습 문제를 다음과 같이 정식화한다.

주어진 입출력 함수 쌍의 유한 집합 $$\{a_j, u_j\}_{j=1}^N$$에서:
- 입력: $a_j \sim \mu$ (확률측도 μ에서 샘플링된 함수)
- 출력: $u_j = G^\dagger(a_j)$ (대상 솔루션 연산자의 실현)

목표는 매개변수 θ ∈ Θ를 갖는 근사 연산자 $$G_\theta: A \rightarrow U$$를 구성하는 것이다. 여기서 A와 U는 분리 가능 Banach 함수공간이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0bc85d4d-c7f5-42fa-b7c8-fe500e525afe/2010.08895v3.pdf)

### 3.2 신경 연산자의 반복 구조

각 레이어에서의 업데이트는 비선형 활성화 함수 σ와 적분 연산자의 합성으로 정의된다:

$$v_{t+1}(x) := \sigma\left(Wv_t(x) + \int_D \kappa(x, y, a(x), a(y); \phi)v_t(y)dy\right) \quad \forall x \in D$$

여기서:
- $W: \mathbb{R}^{d_v} \rightarrow \mathbb{R}^{d_v}$: 로컬 선형변환
- $\kappa$: 신경망으로 매개변수화된 커널 함수
- σ: ReLU 같은 비선형 활성화
- D: 공간 도메인 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0bc85d4d-c7f5-42fa-b7c8-fe500e525afe/2010.08895v3.pdf)

이러한 설계는 **선형 글로벌 적분 연산자와 비선형 로컬 활성화 함수의 결합**으로 복잡한 비선형 연산자를 근사할 수 있음을 반영한다.

### 3.3 Fourier Space 매개변수화 (핵심 혁신)

부분적 평균정리에 따르면, 만약 커널이 위치의 차이에만 의존한다면 ($$\kappa(x, y) = \kappa(x - y)$$), 적분은 합성곱(convolution)이 된다. 그러면 합성곱 정리에 의해:

$$(Kv_t)(x) = \mathcal{F}^{-1}\{R_\phi \cdot (\mathcal{F}v_t)\}(x)$$

여기서 $\mathcal{F}$는 Fourier 변환, $R_\phi$는 학습되는 변환 텐서이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0bc85d4d-c7f5-42fa-b7c8-fe500e525afe/2010.08895v3.pdf)

이를 직접 Fourier 공간에서 매개변수화하면:

$$(K(\phi)v_t)(x) = \mathcal{F}^{-1}\{R_\phi \cdot (\mathcal{F}v_t)\}(x) \quad \forall x \in D$$

여기서 $R_\phi \in \mathbb{C}^{k_{max} \times d_v \times d_v}$는 truncated Fourier modes의 복소수 텐서이다. 켤레 대칭성을 만족하는 실함수 κ에 대해 이를 보장한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0bc85d4d-c7f5-42fa-b7c8-fe500e525afe/2010.08895v3.pdf)

### 3.4 고속 Fourier 변환 구현

균일 해상도 $s_1 \times \cdots \times s_d = n$의 이산 격자에서, 이산 Fourier 변환(DFT)은:

$$(\hat{\mathcal{F}}f)_l(k) = \sum_{x_1=0}^{s_1-1} \cdots \sum_{x_d=0}^{s_d-1} f_l(x_1, \ldots, x_d) e^{-2\pi i \sum_{j=1}^d \frac{x_j k_j}{s_j}}$$

Fourier 모드에서의 행렬-벡터 곱셈:

$$[R \cdot (\mathcal{F}v_t)]_{k,l} = \sum_{j=1}^{d_v} R_{k,l,j}(\mathcal{F}v_t)_{k,j}, \quad k = 1, \ldots, k_{max}, \, l = 1, \ldots, d_v$$

계산 복잡도는:
- FFT: $O(n \log n)$
- Fourier 모드 곱셈: $O(k_{max})$
- 전체: $O(n \log n + k_{max})$ (준선형) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0bc85d4d-c7f5-42fa-b7c8-fe500e525afe/2010.08895v3.pdf)

### 3.5 Fourier 모드 절단 전략

실제 구현에서는 최대 Fourier 모드 $k_{max,j} = 12$로 제한하여 효율성과 정규화 효과를 동시에 얻는다. 이는 **귀납적 편향(inductive bias)**으로 작용하여, 함수들이 고주파 모드로 충분히 빠르게 감소함을 가정한다. 흥미롭게도, ReLU와 최종 decoder를 통해 높은 주파수 모드도 복원될 수 있다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0bc85d4d-c7f5-42fa-b7c8-fe500e525afe/2010.08895v3.pdf)

***

## 4. 모델 구조 및 설계

### 4.1 전체 아키텍처

FNO의 구조는 세 가지 주요 단계로 구성된다:

#### (1) Lifting (Dimension Expansion)
입력 함수 $a(x)$를 얕은 완전 연결 신경망 P로 높은 차원 표현으로 확장:

$$v_0(x) = P(a(x)) \in \mathbb{R}^{d_v}$$

이를 통해 충분한 표현력을 확보한다.

#### (2) Fourier 적분 연산자 레이어들 (반복)
일반적으로 4개 레이어를 사용하며, 각 레이어는:

$$v_{t+1} = \sigma\left(Wv_t + \mathcal{F}^{-1}\{R_\phi \cdot (\mathcal{F}v_t)\}\right)$$

- Fourier 변환 (FFT)
- 절단된 모드에서 선형변환 (R 텐서)
- Inverse Fourier 변환
- Batch Normalization과 ReLU 활성화 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0bc85d4d-c7f5-42fa-b7c8-fe500e525afe/2010.08895v3.pdf)

#### (3) Projection (Dimension Reduction)
$v_T$을 목표 차원으로 축소:

$$u(x) = Q(v_T(x)) \in \mathbb{R}^{d_u}$$

### 4.2 Discretization 불변성의 원리

이 설계의 근본적인 이점은 **Discretization 불변성**이다. Fourier 기저 함수 $e^{2\pi i \langle x, k \rangle}$은 영역 D의 모든 점에서 잘 정의되어 있다. 따라서:

- 서로 다른 해상도 간 솔루션 전달 가능
- $n$ 포인트 격자에서 훈련된 모델을 $m > n$ 포인트 격자에서 직접 평가 가능
- 새로운 이산화 점에서 함수값 계산 가능 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0bc85d4d-c7f5-42fa-b7c8-fe500e525afe/2010.08895v3.pdf)

이는 다음을 의미한다:

$$\text{Error}(\text{resolution}_1) \approx \text{Error}(\text{resolution}_2)$$

반면, CNN 기반 방법들은 해상도에 따라 오차가 증가한다.

### 4.3 비주기적 경계조건 처리

전통적 Fourier 방법은 주기적 경계조건만 처리하지만, FNO는 로컬 선형변환 항 W가 경계값 정보를 추적할 수 있어 비주기적 경계조건도 처리한다. 이는 Darcy Flow 및 Navier-Stokes의 시간 도메인에서 입증된다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0bc85d4d-c7f5-42fa-b7c8-fe500e525afe/2010.08895v3.pdf)

***

## 5. 성능 향상 및 실험 결과

### 5.1 Burgers' 방정식 (1D 점성 유동)

방정식:
$$\partial_t u + \partial_x(u^2/2) = \nu \partial_{xx}u, \quad u(x,0) = u_0(x)$$

연산자: $G^\dagger: L^2_{per}() \rightarrow H^r_{per}()$ (초기조건 → 시간 1의 솔루션) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0bc85d4d-c7f5-42fa-b7c8-fe500e525afe/2010.08895v3.pdf)

**결과:**

| 방법 | s=256 | s=512 | s=2048 | s=8192 |
|------|-------|-------|--------|--------|
| NN (완전연결) | 0.471 | 0.456 | 0.465 | 0.445 |
| FCN (CNN) | 0.096 | 0.141 | 0.231 | 0.324 |
| PCANN | 0.040 | 0.040 | 0.038 | 0.039 |
| GNO | 0.056 | 0.059 | 0.066 | 0.070 |
| MGNO | 0.024 | 0.036 | 0.036 | 0.036 |
| **FNO** | **0.015** | **0.016** | **0.015** | **0.014** |

**분석:**
- FNO는 모든 해상도에서 가장 낮은 오차: 약 0.015 (일정)
- FCN은 해상도 증가에 따라 오차 증가 (0.10 → 0.32): 해상도 의존성 명확
- 다른 신경 연산자(GNO, MGNO)도 FNO보다 정확도 낮음
- FNO의 상대 오차 개선: MGNO 대비 35-50% [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0bc85d4d-c7f5-42fa-b7c8-fe500e525afe/2010.08895v3.pdf)

### 5.2 Darcy Flow (2D 타원형 PDE)

방정식:
$$-\nabla \cdot (a(x)\nabla u(x)) = f(x), \quad x \in (0,1)^2, \quad u = 0 \text{ on } \partial(0,1)^2$$

연산자: $G^\dagger: L^\infty \rightarrow H^1_0$ (확산 계수 → 솔루션)

**결과:**

| 방법 | s=85 | s=141 | s=211 | s=421 |
|------|------|-------|-------|-------|
| NN | 0.172 | 0.172 | 0.172 | 0.172 |
| FCN | 0.025 | 0.049 | 0.073 | 0.110 |
| PCANN | 0.030 | 0.030 | 0.030 | 0.030 |
| RBM (축약 기저) | 0.024 | 0.025 | 0.026 | 0.026 |
| GNO | 0.035 | 0.033 | 0.034 | 0.037 |
| MGNO | 0.042 | 0.043 | 0.043 | 0.042 |
| **FNO** | **0.011** | **0.011** | **0.011** | **0.010** |

**분석:**
- FNO는 약 1 order of magnitude 개선 (0.010-0.011 vs 0.025-0.110)
- 해상도 불변성: 모든 해상도에서 거의 동일한 오차 (0.010-0.011)
- GNO (0.035) 대비 70% 오류 감소
- RBM (축약 기저법)도 해상도 불변이지만 FNO가 2.5배 더 정확 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0bc85d4d-c7f5-42fa-b7c8-fe500e525afe/2010.08895v3.pdf)

### 5.3 Navier-Stokes 방정식 (2D, 난류)

방정식 (와도 형식):
$$\partial_t w + u \cdot \nabla w = \nu \Delta w + f, \quad \nabla \cdot u = 0, \quad w(x,0) = w_0(x)$$

**구성 및 도전:**
- 점성(viscosity): $\nu \in \{1×10^{-3}, 1×10^{-4}, 1×10^{-5}\}$
- 저점성: 난류 영역 (chaotic dynamics)
- 이전 방법: 수렴하지 않음 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0bc85d4d-c7f5-42fa-b7c8-fe500e525afe/2010.08895v3.pdf)

**고정 해상도 (64×64) 결과:**

| 모델 | ν=1e-3<br>T=50<br>N=1000 | ν=1e-4<br>T=30<br>N=10000 | ν=1e-5<br>T=20<br>N=1000 |
|------|------------|------------|------------|
| ResNet | 0.0701 | 0.2311 | 0.2753 |
| U-Net | 0.0245 | 0.1190 | 0.1982 |
| TF-Net (특화) | 0.0225 | 0.1168 | 0.2268 |
| FNO-2D (RNN) | 0.0128 | 0.0834 | 0.1556 |
| **FNO-3D** | **0.0086** | **0.0820** | **0.1893** |

**상세 분석:**
- $\nu = 1×10^{-3}$ (낮은 난류): FNO-3D가 모든 방법 우월 (오차 0.0086)
- $\nu = 1×10^{-4}$ (중간 난류): FNO-3D는 0.082 (데이터 10K)
  - N=1000일 때 모든 모델 > 15% (데이터 부족)
- **중요**: FNO는 처음으로 난류 영역에서 ML 방법으로 수렴
- 이전 GNO/MGNO: 비수렴 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0bc85d4d-c7f5-42fa-b7c8-fe500e525afe/2010.08895v3.pdf)

**계산 속도:**

| 도메인 | 해상도 | FNO | Pseudo-spectral |
|--------|--------|-----|---------|
| Navier-Stokes | 256×256 | 0.005s | 2.2s |
| 속도 향상 | - | - | **440배** |

MCMC 기반 역문제에서:
- FNO: 30,000 평가 = 2.5분
- 전통 솔버: 30,000 평가 = 18시간 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0bc85d4d-c7f5-42fa-b7c8-fe500e525afe/2010.08895v3.pdf)

### 5.4 Zero-shot Super-resolution

**실험 설정:**
- 학습 데이터: 64×64×20 (공간 64×64, 시간 20 스텝)
- 평가 데이터: 256×256×80 (8배 공간 확대, 4배 시간 확대)
- 추가 학습: 없음 (zero-shot)

**결과:**
- FNO: 정상 작동, 정확도 유지
- 다른 모든 모델 (FNO-2D, U-Net, TF-Net, ResNet): **불가능** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0bc85d4d-c7f5-42fa-b7c8-fe500e525afe/2010.08895v3.pdf)

**원리:**
FNO는 Fourier 기저에서 학습되므로, 평가 점들은 새로운 격자에서도:

$$e^{2\pi i \langle x_{new}, k \rangle}$$

로 자연스럽게 정의된다. 따라서 interpolation이나 재훈련 없이도 새로운 해상도에서 평가 가능하다.

### 5.5 Bayesian 역문제 (실제 응용)

**문제:** Navier-Stokes의 초기 와도 $w_0$를 희소하고 잡음이 있는 관측에서 추정

**방법:** Function space MCMC (Cotter et al., 2013)
- 샘플 수: 30,000 (burn-in 5,000)
- 목표: 사후분포(posterior) 평균 추정

**비교:**

| 방법 | Forward 호출 | 평가/호출 | 총 시간 |
|------|-----------|---------|--------|
| FNO surrogate | 30,000 | 0.005s | **2.5분** |
| 전통 solver | 30,000 | 2.2s | **18시간** |

**핵심 이점:**
- 오프라인 훈련: 12시간 (한 번만)
- 온라인 추론: 2.5분 (매번)
- 전통 solver: 매번 18시간 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0bc85d4d-c7f5-42fa-b7c8-fe500e525afe/2010.08895v3.pdf)

결과적으로 **전체 수행 시간에서도 FNO가 더 효율적** (훈련 + 추론 vs 순수 추론)

***

## 6. 모델 일반화 성능 및 한계

### 6.1 Spectral 복구 능력

**고주파 모드 복원:**

Truncation 분석에서 흥미로운 발견:
- $\nu = 1×10^{-3}$ Navier-Stokes:
  - 20개 Fourier 모드로 절단 → 오차 ~2%
  - FNO (12개 파라미터 모드): 오차 <1%

**메커니즘:**
ReLU 활성화와 최종 decoder Q가 높은 주파수를 생성할 수 있기 때문:

$$\text{High freq} = Q(\sigma(\text{Low freq interactions}))$$

이는 **선형 글로벌 연산자 + 비선형 로컬 활성화의 시너지**를 보여준다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0bc85d4d-c7f5-42fa-b7c8-fe500e525afe/2010.08895v3.pdf)

### 6.2 식별된 한계

#### (1) 데이터 요구량
- $\nu = 1×10^{-3}$: N = 1,000 충분 (< 1% 오차)
- $\nu = 1×10^{-4}$: N = 10,000 필요
- $\nu = 1×10^{-5}$: 모든 방법 실패 (오차 > 15%)

난류가 복잡해질수록 기하급수적 데이터 요구 증가

#### (2) Out-of-Distribution 일반화 미흡
논문에서 직접 다루지는 않았지만, 이후 연구들이 지적하는 주요 한계

#### (3) 비정상 경계조건
- 주기적 조건: 완벽히 지원
- 비주기적 조건: Bias term W로 처리 가능 (논문에서 입증: Darcy, NS)
- 복잡한 기하학: 아직 제한적

#### (4) 높은 차원 문제
논문에서 테스트한 최대: 2D 공간 + 1D 시간 (3D 문제)

***

## 7. 2020년 이후 최신 연구 동향 및 비교

### 7.1 구조적 개선

#### U-FNO (Wen et al., 2022) [sccs.stanford](https://sccs.stanford.edu/sites/g/files/sbiybj17761/files/media/file/2109.03697_0.pdf)
**아이디어:** Fourier 레이어와 U-Net 구조 결합

- **설계**: Fourier 레이어와 U-Net 블록을 교대로 배치
- **강점**: 
  - 글로벌 저주파 정보 (Fourier)
  - 로컬 고주파 세부사항 (U-Net convolutions)
  - 복잡한 multiphase flow에서 30-60% 정확도 개선
- **적용**: CO₂-water multiphase flow에서 우수 성능 [sccs.stanford](https://sccs.stanford.edu/sites/g/files/sbiybj17761/files/media/file/2109.03697_0.pdf)

#### Factorized FNO (F-FNO, 2023) [arxiv](https://arxiv.org/pdf/2111.13802.pdf)
**핵심 혁신:** 분리 가능한 spectral layers

$$\text{Factorized: } R = \bigoplus_{d=1}^D R^{(d)}$$

- **장점**: 매개변수 수 38% 감소, 유사 성능 유지
- **메모리**: 12% 절감
- **최신 개선**: U-shaped F-FNO (2025) [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0898122125003013)

#### Gabor-Filtered FNO (GFNO, 2024) [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0045793024000719)
**혁신:** Fourier 모드 앞에 학습 가능한 Gabor 필터

$$\hat{v}_t = \mathcal{F}^{-1}\{G(k) \odot (\mathcal{F}v_t)\}$$

- **효과**: Non-dominant 주파수 학습 개선
- **성능**: 평균 50% 오류 감소 [arxiv](https://arxiv.org/abs/2404.07200)
- **원인**: FNO의 **Fourier parameterization bias** 극복

#### Convolutional Fourier Neural Operator (Conv-FNO, 2025) [arxiv](https://arxiv.org/pdf/2503.17797.pdf)
**설계:** CNN 전처리기 + FNO 백본

- **다해상도 적응**: Resizing scheme으로 가변 격자 지원
- **구현**: UNet-Conv-FNO 변형
- **성능**: 기존 FNO, CNO 대비 현저한 향상 [arxiv](https://arxiv.org/pdf/2503.17797.pdf)

### 7.2 이론적 진전

#### 보편 근사 정리 (Kovachki et al., 2021) [semanticscholar](https://www.semanticscholar.org/paper/On-universal-approximation-and-error-bounds-for-Kovachki-Lanthaler/d15a11a7be64ccc52b709b44b9fac1e4d6302062)
**정리:** FNO는 보편 근사자(universal approximator)

$$\forall \epsilon > 0, \, \exists \text{FNO} \, G_\theta: \|G_\theta(u) - G^\dagger(u)\| < \epsilon$$

**오류 경계:**

Darcy (타원형) 및 Navier-Stokes (포물형) PDE:

$$\text{Size(FNO)} = O\left(\log(1/\epsilon)\right) \text{ 또는 } O((1/\epsilon)^{\alpha})$$

즉, **부-로그 선형 확대** (sub-log-linear growth) [semanticscholar](https://www.semanticscholar.org/paper/On-universal-approximation-and-error-bounds-for-Kovachki-Lanthaler/d15a11a7be64ccc52b709b44b9fac1e4d6302062)

#### Fourier Linear Operator 학습 경계 (2024) [arxiv](https://arxiv.org/html/2408.09004v1)
**분석:** 통계, 절단, 이산화 오류 분해

$$\text{Excess Risk} = \underbrace{E_{stat}}_{\text{통계}} + \underbrace{E_{trunc}}_{\text{절단}} + \underbrace{E_{disc}}_{\text{이산화}}$$

**경계:**

For $f \in W^{r,\infty}$ (Sobolev space):

$$E_{stat} = O\left(\frac{k_{max}^{d/2}}{N}\right)$$

$$E_{trunc} = O\left(e^{-\lambda k_{max}}\right) \text{ (exponential decay)}$$

$$E_{disc} = O(h^r) \text{ where } h = n^{-1/d}$$

- 통계 오류: 모드 수와 역 샘플 크기에 선형 [arxiv](https://arxiv.org/html/2408.09004v1)
- 절단 오류: 지수적 감소 (충분한 해상도에서 무시 가능)
- 이산화 오류: 함수 smoothness r에 따라 다항식 감소 [arxiv](https://arxiv.org/html/2408.09004v1)

**함의:** O(n log n) 시간 복잡도로도 high-accuracy 달성 가능

#### Mean-field 이론 (2024) [arxiv](https://arxiv.org/html/2310.06379v3)
**접근:** Random FNO initialization 분석

- **Ordered-chaos 전이**: FNO의 표현성이 이 전이점 근처에서 최적
- **Gradient flow**: Vanishing/exploding gradient의 위상 경계 식별
- **실용 결론**: FNO 초기화 시 신중한 스케일링 필수 [arxiv](https://arxiv.org/html/2310.06379v3)

### 7.3 Out-of-Distribution 일반화

#### NOVA (2025) [arxiv](https://arxiv.org/html/2601.19091v1)
**방법:** Physics-informed Neural Architecture Search

**성능:**
- 분포 내: FNO와 유사
- 분포 외 (초기조건 변경):
  - FNO RMSE: O(10⁻¹)
  - NOVA RMSE: O(10⁻³)
  - **개선: 1-2 orders of magnitude** [arxiv](https://arxiv.org/html/2601.19091v1)

**원인**: Physics 제약이 학습된 표현을 regularize

#### Dual-branch FNO (2025) [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0955799724005551)
**설계:** 분포 내/외 조건 각각 처리

- 매개변수 분포 외 변화에 대한 강화된 견고성
- 불확실성 정량화(UQ) 통합 [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0955799724005551)

### 7.4 응용 분야 확장 (2024-2026)

| 응용 분야 | 방법 | 성과 | 년도 |
|---------|------|------|------|
| 지구물리 3D | Hybrid DeepONet-FNO | 오류 1/10 개선, 1000배 가속 [ieeexplore.ieee](https://ieeexplore.ieee.org/document/11217330/) | 2025 |
| 난류 제트 | Factorized FNO | 온도/종 필드 해상도 robust 복원 [linkinghub.elsevier](https://linkinghub.elsevier.com/retrieve/pii/S0010218025006285) | 2025 |
| 산사태 역학 | FNO 2D | 위험 평가 가속 [mdpi](https://www.mdpi.com/2076-3263/16/2/55) | 2026 |
| CO₂ 지질 격납 | Multi-resolution FNO | 200년 장기 예측, 수정 coupling [onepetro](https://onepetro.org/IPTCONF/proceedings/26IPTC/26IPTC/D011S007R006/794945) | 2026 |
| 광섬유 초고속 | Conditional FNO | 6개 시나리오 단일 모델, SOTA [ieeexplore.ieee](https://ieeexplore.ieee.org/document/11250891/) | 2026 |
| 전력전자 열 | FNO thermal model | 0.1s 예측, 최대 오류 1.8°C [ieeexplore.ieee](https://ieeexplore.ieee.org/document/11168963/) | 2026 |
| 전기저항탐사 | FNO surrogate | 3D forward 모델, 1000배 가속 [academic.oup](https://academic.oup.com/gji/advance-article/doi/10.1093/gji/ggag022/8431395) | 2026 |
| 초음파 탄성 | FNO elastography | noise/해상도 robust, U-Net 우월 [semanticscholar](https://www.semanticscholar.org/paper/fc21490a36ea3c6ea896c1cab1596fef78f2ecf2) | 2026 |

**패턴:**
- 더 깊은 물리 제약 추가 추세
- 조건부 모델링 (conditional FNO)
- 다중 물리 결합
- Real-world data 검증 [linkinghub.elsevier](https://linkinghub.elsevier.com/retrieve/pii/S0010218025006285)

***

## 8. 앞으로의 연구에서 고려할 점

### 8.1 데이터 효율성 개선

**문제:** 난류 (ν=1×10⁻⁴)에서 N=10,000 필요

**해결 방향:**
1. **Physics-informed 손실:** PDE 잔차 포함
2. **다중 실마리 학습:** Numerical solver + Data-driven 결합
3. **Transfer learning:** 저점성 → 고점성 전이 [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10630570/)
4. **Few-shot 메타-학습:** 적은 샘플로 빠른 적응

**최신 진전:**
- Augmentation 기법 (자가-감독 학습)
- Score-based diffusion models과 결합 [ojs.aaai](https://ojs.aaai.org/index.php/AAAI/article/view/34459)

### 8.2 Out-of-Distribution 견고성

**문제:** 분포 외 초기조건/기하에서 오류 폭증

**해결 방향:**
1. **Ensemble 방법**: Physics-aligned representation 학습
2. **Domain randomization**: 다양한 매개변수 범위로 훈련
3. **Uncertainty quantification**: Bayesian FNO
4. **Causal learning**: 인과적 표현 학습 [stat.berkeley](https://www.stat.berkeley.edu/~mmahoney/pubs/34_Does_In_Context_Operator_Le.pdf)

**최신 결과:**
- Causal intervention + prompt tuning [ieeexplore.ieee](https://ieeexplore.ieee.org/document/11202726/)
- Online test-time adaptation [nature](https://www.nature.com/articles/s41467-025-57101-4)
- Green's operator 기반 접근 (명시적 Green 함수로 선형성 보존) [arxiv](https://arxiv.org/abs/2406.01857)

### 8.3 이론적 깊이

**개방 문제:**
1. **통계-이산화 오류 간극**: 이론-관찰 갭 폐쇄
2. **Generalization bounds 타이트화**: 현재 보수적
3. **Approximation mechanism 이해**: 왜 12개 모드로 충분?

**진행 중인 연구:**
- Mixed-precision FNO의 정확도 보장 [proceedings.iclr](https://proceedings.iclr.cc/paper_files/paper/2024/file/89ab43b626b480347dbaccda7be8aedb-Paper-Conference.pdf)
- Deep vector-valued RKHS framework [arxiv](https://arxiv.org/pdf/2512.19184.pdf)
- Layer-wise generalization analysis [arxiv](https://arxiv.org/pdf/2512.19184.pdf)

### 8.4 실제 응용 확대

#### 1) 비주기 복잡 기하
- **도전**: 임의 메시 형상
- **해결책**: Local neural operator (LNO) - 사전훈련된 모델 다양한 도메인에 적용 [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0045782524003189)

#### 2) 역문제 및 설계 최적화
- **현황**: MCMC surrogate로 2.5분 (vs 18시간)
- **다음**: PDE 제약 최적화에서 adjoint-free 방법

#### 3) 장시간 예측 (Long-horizon)
- **문제**: 오류 축적 in autoregressive rollout
- **NOVA 해법**: Physics-informed representation으로 1-2 orders 개선 [arxiv](https://arxiv.org/html/2601.19091v1)

#### 4) Multi-physics 결합
- **예시**: 수정-지구화학 결합 CO₂ storage [onepetro](https://onepetro.org/IPTCONF/proceedings/26IPTC/26IPTC/D011S007R006/794945)
- **전략**: 모듈형 FNO 연쇄 + 교차 필드 매개변수 전달

### 8.5 계산 확장성

**GPU 효율:**
- 현재: V100 16GB에서 2D 문제
- 미래: 
  - Mixed-precision training (12% 메모리 절감) [proceedings.iclr](https://proceedings.iclr.cc/paper_files/paper/2024/file/89ab43b626b480347dbaccda7be8aedb-Paper-Conference.pdf)
  - Factorization (38% 매개변수 감소) [arxiv](https://arxiv.org/pdf/2111.13802.pdf)
  - Distributed training

**적응형 Fourier mode:**
- 현재: 고정 k_{max,j} = 12
- 개선: 문제 특성에 따라 동적 선택

***

## 9. 종합 평가 및 결론

### 9.1 FNO의 근본적 기여

FNO는 **세 가지 패러다임 전환**을 이루었다:

#### 1) 문제 정식화 전환
- **이전**: PDE 인스턴스당 1개 솔버 필요
- **FNO**: 매개변수 패밀리 전체 → 단일 학습

#### 2) 계산 패러다임 전환
- **이전**: 반복적 solver (FEM/FDM): O(해상도³) 시간
- **FNO**: 한 번의 순방향 계산: O(n log n)
- **실제**: 1000배 이상 가속

#### 3) 일반화 패러다임 전환
- **이전**: 특정 해상도에 bound됨
- **FNO**: Resolution-invariant + zero-shot super-resolution

### 9.2 이론-실전 균형

| 측면 | 수준 | 증거 |
|-----|------|------|
| **보편성** | 증명됨 | Kovachki et al. 2021 [semanticscholar](https://www.semanticscholar.org/paper/On-universal-approximation-and-error-bounds-for-Kovachki-Lanthaler/d15a11a7be64ccc52b709b44b9fac1e4d6302062) |
| **오류 경계** | Sub-log-linear | 부-로그 선형 확대율 [semanticscholar](https://www.semanticscholar.org/paper/On-universal-approximation-and-error-bounds-for-Kovachki-Lanthaler/d15a11a7be64ccc52b709b44b9fac1e4d6302062) |
| **실제 성능** | SOTA | Burgers 50%, Darcy 90%, NS 20-30% 개선 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0bc85d4d-c7f5-42fa-b7c8-fe500e525afe/2010.08895v3.pdf) |
| **확장성** | O(n log n) | FFT 기반 준선형 복잡도 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0bc85d4d-c7f5-42fa-b7c8-fe500e525afe/2010.08895v3.pdf) |

### 9.3 남은 과제

1. **Out-of-distribution**: NOVA 등으로 개선 중이나 여전히 도전
2. **데이터 효율**: 난류에서 10K+ 샘플 필요
3. **고차원**: 3D 이상은 제한적 증거
4. **이론**: Generalization bounds 여전히 보수적

### 9.4 미래 전망

**단기 (1-2년):**
- 구조적 변형 (U-FNO, F-FNO) 정성화
- Conditional/multiple-PDE 모델
- Real-world 검증 확대

**중기 (2-5년):**
- Physics-informed FNO 통합
- OOD 견고성 근본적 해결
- 3D+ 문제 확장
- 역문제/최적화 통합 플랫폼

**장기 (5년+):**
- Universal surrogate models (여러 PDE)
- Certified error bounds
- 다중물리 digital twins

### 9.5 최종 평가

**FNO는 과학 계산의 ML화의 분수령:**

- ✅ **강점**: 속도(1000배), 정확도(SOTA), 이론(보편성), 실용성(다양 응용)
- ⚠️ **주의**: 데이터 의존, OOD 취약, 고차원 한계
- 🔮 **전망**: 다음 세대 과학 계산의 기초 기술로 자리매김 중

Li et al. (2021)의 이 획기적 논문은 **4,600+ 인용**을 달성하여, 이 분야가 얼마나 급속도로 발전하고 있는지를 보여준다.

***

## 참고 자료 인덱스

 - Li et al., "Fourier Neural Operator for Parametric PDEs", ICLR 2021 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0bc85d4d-c7f5-42fa-b7c8-fe500e525afe/2010.08895v3.pdf)
 - Factorized FNO for turbulent jets, 2025 [linkinghub.elsevier](https://linkinghub.elsevier.com/retrieve/pii/S0010218025006285)
 - Incremental FNO (iFNO), 2024 [onepetro](https://onepetro.org/IPTCONF/proceedings/26IPTC/26IPTC/D021S010R003/794837)
 - Multi-grid Tensorized FNO (MG-TFNO), 2023 [mdpi](https://www.mdpi.com/2076-3263/16/2/55)
 - U-FNO for multiphase flow, 2022 [papers.phmsociety](https://papers.phmsociety.org/index.php/phmap/article/view/4496)
 - Hybrid DeepONet-FNO for 3D AEM, 2025 [ieeexplore.ieee](https://ieeexplore.ieee.org/document/11217330/)
 - Conditional FNO for optical fibers, 2026 [ieeexplore.ieee](https://ieeexplore.ieee.org/document/11250891/)
 - Multi-resolution FNO for CO₂ storage, 2026 [onepetro](https://onepetro.org/IPTCONF/proceedings/26IPTC/26IPTC/D011S007R006/794945)
 - FNO thermal model for power modules, 2026 [ieeexplore.ieee](https://ieeexplore.ieee.org/document/11168963/)
 - FNO surrogate for ERT, 2026 [academic.oup](https://academic.oup.com/gji/advance-article/doi/10.1093/gji/ggag022/8431395)
 - FNO for ultrasound elastography, 2026 [semanticscholar](https://www.semanticscholar.org/paper/fc21490a36ea3c6ea896c1cab1596fef78f2ecf2)
 - Factorized FNO (F-FNO), 2023 [arxiv](https://arxiv.org/pdf/2111.13802.pdf)
 - U-FNO for multiphase, 2022 [arxiv](https://arxiv.org/abs/2109.03697)
 - Original FNO arXiv 2020 [arxiv](https://arxiv.org/abs/2010.08895)
 - Operator-based generalization bound, 2025 [arxiv](https://arxiv.org/pdf/2512.19184.pdf)
 - Mean-field theory for FNO, 2024 [arxiv](https://arxiv.org/html/2310.06379v3)
 - Neural Green's Operators, 2024 [arxiv](https://arxiv.org/abs/2406.01857)
 - Neural Operator survey, 2021 [arxiv](https://arxiv.org/abs/2108.08481)
 - FNO guide, Duruisseaux & Anandkumar, 2025 [arxiv](https://arxiv.org/abs/2512.01421)
 - Improved generalization with DeepONets, 2024 [arxiv](https://arxiv.org/html/2301.06701v3)
 - Toward understanding FNO (SpecB-FNO), 2024 [arxiv](https://arxiv.org/abs/2404.07200)
 - Gabor-Filtered FNO, 2024 [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0045793024000719)
 - Local Neural Operator, 2024 [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0045782524003189)
 - In-context operator learning, 2024 [stat.berkeley](https://www.stat.berkeley.edu/~mmahoney/pubs/34_Does_In_Context_Operator_Le.pdf)
 - NOVA (Physics-informed NAS), 2025 [arxiv](https://arxiv.org/html/2601.19091v1)
 - Error bounds for Fourier operators, 2024 [arxiv](https://arxiv.org/html/2408.09004v1)
 - Conv-FNO with local spatial info, 2025 [arxiv](https://arxiv.org/pdf/2503.17797.pdf)
 - Universal approximation & error bounds, Kovachki et al. 2021 [arxiv](https://arxiv.org/abs/2107.07562)
 - U-FNO for multiphase, 2022 [sccs.stanford](https://sccs.stanford.edu/sites/g/files/sbiybj17761/files/media/file/2109.03697_0.pdf)
 - Dual-branch FNO, 2025 [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0955799724005551)
 - Transfer learning FNO, 2024 [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10630570/)
 - Mixed-precision FNO, 2024 [proceedings.iclr](https://proceedings.iclr.cc/paper_files/paper/2024/file/89ab43b626b480347dbaccda7be8aedb-Paper-Conference.pdf)
 - JMLR universal approximation, 2021 [jmlr](https://www.jmlr.org/papers/volume22/21-0806/21-0806.pdf)
 - U-shaped Factorized FNO, 2025 [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0898122125003013)
