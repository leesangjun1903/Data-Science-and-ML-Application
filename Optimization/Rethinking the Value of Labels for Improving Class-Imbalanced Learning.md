# Rethinking the Value of Labels for Improving Class-Imbalanced Learning

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

이 논문(Yang & Xu, NeurIPS 2020)의 핵심은 **불균형 레이블(imbalanced label)이 "양날의 검"** 이라는 것입니다. 레이블은 두 가지 상반된 역할을 합니다:

| 관점 | 내용 |
|------|------|
| **긍정적 관점** | 레이블은 가치 있음 → 추가 비레이블 데이터와 결합 시 준지도학습(SSL)으로 성능 향상 가능 |
| **부정적 관점** | 레이블은 항상 유용하지 않음 → 불균형 레이블이 결정 경계를 편향시킴(label bias) → 자기지도 사전학습(SSP)으로 이를 보완 가능 |

### 주요 기여

1. **최초의 체계적 분석**: 불균형 레이블의 이중적 역할을 이론적·실험적으로 동시에 분석
2. **반지도학습(Semi-supervised) 전략**: 추가 비레이블 데이터를 활용한 label bias 완화 방법론 제시
3. **자기지도 사전학습(Self-supervised Pre-training, SSP) 전략**: 추가 데이터 없이도 label bias를 극복하는 새로운 프레임워크 제시
4. **이론적 보장**: 두 전략 모두에 대한 Gaussian 모델 기반 수학적 증명 제공

---

## 2. 해결 문제, 제안 방법, 모델 구조, 성능, 한계

### 2.1 해결하고자 하는 문제

현실 세계의 데이터는 **롱테일 분포(long-tailed distribution)**를 따르는 경우가 많습니다. 이로 인해 발생하는 핵심 문제는 **레이블 편향(label bias)**입니다:

$$\rho = \frac{\max_i\{n_i\}}{\min_i\{n_i\}}$$

여기서 $\rho$는 **불균형 비율(imbalance ratio)**로, 헤드 클래스(다수 클래스) 샘플 수를 테일 클래스(소수 클래스) 샘플 수로 나눈 값입니다. $\rho$가 클수록 불균형이 심각합니다.

기존 방법(데이터 재샘플링, 클래스 균형 손실함수 등)만으로는 극단적 불균형 상황에서 여전히 성능 저하가 발생합니다.

---

### 2.2 제안하는 방법 (수식 포함)

#### ▶ 방법 1: 반지도학습 기반 불균형 학습 (Semi-Supervised Imbalanced Learning)

**이론적 동기 (Theorem 1)**

이진 분류 문제에서 데이터 생성 분포 $P_{XY}$가 두 Gaussian의 혼합이라고 가정합니다:

$$X|Y=+1 \sim \mathcal{N}(\mu_1, \sigma^2), \quad X|Y=-1 \sim \mathcal{N}(\mu_2, \sigma^2)$$

최적 분류기는 $f(x) = \text{sign}\left(x - \frac{\mu_1 + \mu_2}{2}\right)$이며, 목표는 $\frac{\mu_1 + \mu_2}{2}$를 정확히 추정하는 것입니다.

비레이블 데이터 $\{\tilde{X}\_i\}_{i=1}^{\tilde{n}}$에 대해 기반 분류기 $f_B$로 생성한 의사 레이블(pseudo-label)을 사용하여 추정치를 구성합니다:

$$\hat{\theta} = \frac{1}{2}\left(\frac{\sum_{i=1}^{\tilde{n}_+}\tilde{X}_i^+}{\tilde{n}_+} + \frac{\sum_{i=1}^{\tilde{n}_-}\tilde{X}_i^-}{\tilde{n}_-}\right)$$

**Theorem 1**: $\Delta \triangleq p - q$ (양성/음성 클래스 정확도 차이, 불균형 정도를 반영)라 할 때, 임의의 $\delta > 0$에 대해 다음 확률로:

$$P \geq 1 - 2e^{-\frac{2\delta^2}{9\sigma^2} \cdot \frac{\tilde{n}_+\tilde{n}_-}{\tilde{n}_-+\tilde{n}_+}} - 2e^{-\frac{8\tilde{n}_+\delta^2}{9(\mu_1-\mu_2)^2}} - 2e^{-\frac{8\tilde{n}_-\delta^2}{9(\mu_1-\mu_2)^2}}$$

추정치 $\hat{\theta}$는 다음을 만족합니다:

$$\left|\hat{\theta} - (\mu_1+\mu_2)/2 - \Delta(\mu_1-\mu_2)/2\right| \leq \delta$$

**해석**:
- $\Delta$가 클수록(불균형이 심할수록) 추정 정확도 저하
- 비레이블 데이터가 많을수록( $\tilde{n}\_+$, $\tilde{n}_-$ 증가) 성공 확률이 **지수적으로** 증가
- 비레이블 데이터가 불균형이더라도 레이블 데이터보다 많으면 여전히 도움이 됨

**실제 프레임워크**

자기 학습(self-training)을 이용한 반지도학습으로, 손실 함수는 다음과 같습니다:

$$\mathcal{L}(\mathcal{D}_L, \theta) + \omega \mathcal{L}(\mathcal{D}_U, \theta)$$

여기서:
- $\mathcal{D}_L$: 원본 불균형 레이블 데이터셋
- $\mathcal{D}_U$: 의사 레이블이 부여된 비레이블 데이터셋
- $\omega$: 비레이블 데이터 가중치

**단계**:
1. 원본 불균형 데이터 $\mathcal{D}\_L$로 중간 분류기 $f_{\hat{\theta}}$ 훈련
2. $f_{\hat{\theta}}$로 비레이블 데이터 $\mathcal{D}_U$에 의사 레이블 $\hat{y}$ 생성
3. $\mathcal{D}_L + \mathcal{D}_U$를 합쳐 최종 모델 훈련

---

#### ▶ 방법 2: 자기지도 사전학습 기반 불균형 학습 (Self-Supervised Pre-training, SSP)

**이론적 동기 (Theorem 2 & 3)**

$d$차원 이진 분류 문제를 고려합니다:

$$X|Y=+1 \sim \mathcal{N}(0, \sigma_1^2\mathbf{I}_d), \quad X|Y=-1 \sim \mathcal{N}(0, \beta\sigma_1^2\mathbf{I}_d), \quad \beta > 3$$

**Theorem 2** (표준 학습의 한계): $b > 0$인 선형 분류기 $f(X) = \text{sign}(\langle\theta, X\rangle + b)$에 대해:

$$\text{err}_f = p_+\Phi\left(-\frac{b}{\|\theta\|_2\sigma_1}\right) + p_-\Phi\left(\frac{b}{\|\theta\|_2\sqrt{\beta}\sigma_1}\right) \geq \frac{1}{4}$$

즉, 표준 훈련으로는 **정확도가 3/4을 넘을 수 없습니다**.

**자기지도 표현 학습**: 자기지도 태스크를 통해 $Z = \psi(X) = k_1\|X\|_2^2 + k_2$ ($k_1, k_2 > 0$) 형태의 표현을 학습합니다.

자기지도 분류기는 다음과 같이 정의됩니다:

$$f_{ss}(X) = \text{sign}(-Z + b), \quad b = \frac{1}{2}\left(\frac{\sum_{i=1}^N \mathbf{1}_{\{Y_i=+1\}}Z_i}{N_+} + \frac{\sum_{i=1}^N \mathbf{1}_{\{Y_i=-1\}}Z_i}{N_-}\right)$$

**Theorem 3**: 임의의 $\delta \in \left(0, \frac{\beta-1}{\beta+1}\right)$에 대해, 확률 $1 - 2e^{-N_-d\delta^2/8} - 2e^{-N_+d\delta^2/8}$ 이상으로 자기지도 분류기의 오류 확률은:

$$\text{err}_{f_{ss}} \leq \begin{cases} p_+e^{-d \cdot \frac{(\beta-1-(1+\beta)\delta)^2}{32}} + p_-e^{-d \cdot \frac{(\beta-1-(1+\beta)\delta)^2}{32\beta^2}}, & \text{if } \delta \in \left[\frac{\beta-3}{\beta+1}, \frac{\beta-1}{\beta+1}\right); \\ p_+e^{-d \cdot \frac{(\beta-1-(1+\beta)\delta)}{16}} + p_-e^{-d \cdot \frac{(\beta-1-(1+\beta)\delta)^2}{32\beta^2}}, & \text{if } \delta \in \left(0, \frac{\beta-3}{\beta+1}\right). \end{cases}$$

**핵심 해석**: 자기지도 사전학습을 통해 얻은 표현을 사용하면, 오류 확률이 **차원 $d$에 대해 지수적으로 감소**합니다. 이는 고차원 현대 데이터에서 특히 유리합니다.

**실제 프레임워크 (2단계)**:
1. **1단계 (SSP)**: 레이블 정보를 버리고 자기지도 학습으로 사전훈련 → label-agnostic 초기화 획득
   - CIFAR-LT: Rotation 예측 사용
   - ImageNet-LT, iNaturalist: MoCo 사용
2. **2단계 (표준 훈련)**: SSP로 초기화된 네트워크로 임의의 불균형 학습 기법 적용

---

### 2.3 모델 구조

```
[SSP 프레임워크]
입력 데이터 (불균형)
    ↓
1단계: 자기지도 사전학습 (Rotation/MoCo 등)
    → Label-agnostic 표현 학습
    ↓
2단계: 표준 불균형 학습 (CE/LDAM-DRW/cRT 등)
    → SSP로 초기화된 네트워크 파인튜닝
    ↓
최종 분류기

[반지도학습 프레임워크]
레이블 데이터 (DL, 불균형)
    ↓
중간 분류기 학습
    ↓
비레이블 데이터 (DU)에 의사 레이블 생성
    ↓
DL + DU 결합하여 최종 모델 학습
    ↓
최종 분류기
```

- **백본**: ResNet-32 (CIFAR-LT), ResNet-50/ResNet-10 (ImageNet-LT, iNaturalist)
- **자기지도 방법**: Rotation, Jigsaw, Selfie, MoCo
- **반지도학습 방법**: Pseudo-label, VAT, Mean Teacher

---

### 2.4 성능 향상

**반지도학습 결과 (CIFAR-10-LT, ResNet-32, Top-1 Error %)**

| 방법 | $\rho=100$ | $\rho=50$ | $\rho=10$ |
|------|-----------|-----------|-----------|
| CE | 29.64 | 25.19 | 13.61 |
| CE + $\mathcal{D}_U$@5x ($\rho_U=1$) | **17.48** | **16.79** | **10.22** |
| LDAM-DRW | 22.97 | 19.06 | 11.84 |
| LDAM-DRW + $\mathcal{D}_U$@5x ($\rho_U=1$) | **14.96** | **14.33** | **8.72** |

**자기지도 사전학습 결과 (CIFAR-10-LT, ResNet-32, Top-1 Error %)**

| 방법 | $\rho=100$ | $\rho=50$ | $\rho=10$ |
|------|-----------|-----------|-----------|
| LDAM-DRW | 22.97 | 19.06 | 11.84 |
| LDAM-DRW + SSP | **22.17** | **17.87** | **11.47** |
| CB-CE | 27.63 | 21.95 | 13.23 |
| CB-CE + SSP | **23.47** | **19.60** | **11.57** |

**대규모 데이터셋 결과 (iNaturalist 2018, ResNet-50, Top-1 Error %)**

| 방법 | Error |
|------|-------|
| cRT | 34.8 |
| cRT + SSP | **31.9** |
| LDAM-DRW | 35.4 |
| LDAM-DRW + SSP | **33.7** |

---

### 2.5 한계

논문이 직접 언급하거나 실험 결과에서 확인되는 한계점들입니다:

1. **비레이블 데이터 의존성**: 반지도학습은 비레이블 데이터의 **관련성(relevance)**에 민감합니다. 관련성이 60% 미만이면 오히려 성능이 저하될 수 있습니다.
2. **비레이블 데이터 불균형**: $\rho_U$가 $\rho$보다 클 경우 성능 향상이 제한됩니다.
3. **SSP 방법 선택의 데이터 규모 의존성**: 소규모 데이터셋은 Rotation이, 대규모 데이터셋은 MoCo가 더 효과적 → **범용 방법이 없음**
4. **학술 데이터셋 한정 검증**: 자율주행, 의료 진단 등 실제 응용에서의 검증 부족
5. **공정성(fairness) 문제 미고려**: 소수 클래스의 윤리적 측면(fairness, privacy)을 다루지 않음
6. **계산 비용**: SSP 2단계 훈련 과정은 표준 방법보다 훈련 시간이 늘어남
7. **이론적 모델의 단순성**: Gaussian 모델 기반 이론이 실제 복잡한 데이터 분포를 완전히 반영하지 못할 수 있음

---

## 3. 모델의 일반화 성능 향상과 관련된 분석

### 3.1 소수 클래스(Minority Class) 일반화

논문의 핵심 기여 중 하나는 소수 클래스에 대한 일반화 성능 향상입니다.

**반지도학습에서의 소수 클래스 일반화**:
- 헤드 클래스(다수 클래스)는 성능 유지 또는 소폭 향상
- **테일 클래스(소수 클래스)는 훨씬 큰 폭으로 성능 향상**
- CIFAR-10-LT에서 테일 클래스의 예측이 헤드 클래스로 혼동(confusion)되는 비율이 크게 감소

혼동 행렬(Confusion Matrix) 분석:
- 표준 CE: 테일 클래스(C8, C9)의 헤드 클래스로의 오분류율이 매우 높음 (예: C8→C0: 42%, C9→C1: 32%)
- CE + $\mathcal{D}_U$@5x: 동일 테일 클래스의 오분류율 크게 감소 (C8→C0: 18%, C9→C1: 16%)

**SSP에서의 소수 클래스 일반화**:
- ImageNet-LT에서 Few-shot 클래스(20개 미만 샘플)에서 일관된 성능 향상 확인
- SSP는 테일 클래스의 결정 경계 누수(leakage)를 크게 감소시킴

### 3.2 일반화 성능 향상의 메커니즘

**t-SNE 시각화를 통한 분석**:

```
[반지도학습 효과]
비레이블 데이터 없이: 테일 클래스의 표현이 헤드 클래스와 혼합
비레이블 데이터 사용: 테일 클래스의 결정 경계가 명확히 형성

[SSP 효과]
표준 CE: 헤드 클래스가 결정 경계를 지배 → 테일 클래스 누수
SSP 적용: 레이블에 독립적인 표현 → 더 균형 잡힌 결정 경계
```

### 3.3 이론적 일반화 보장

Theorem 3의 핵심 결과는 자기지도 분류기의 오류가 **차원 $d$에 대해 지수적으로 감소**한다는 것입니다:

$$\text{err}_{f_{ss}} \leq p_+e^{-d \cdot C_1} + p_-e^{-d \cdot C_2}$$

여기서 $C_1, C_2$는 데이터 특성에 따른 상수입니다. 현대 고차원 데이터에서 이는 매우 강력한 일반화 보장입니다.

반면, 표준 선형 분류기는 불균형과 무관하게 오류가 최소 $\frac{1}{4}$로 고정됩니다(Theorem 2).

### 3.4 일반화 성능의 다양한 설정에서의 강건성

| 설정 | 반지도학습 효과 | SSP 효과 |
|------|----------------|----------|
| 다양한 $\rho$ 값 | ✓ 일관된 향상 | ✓ 일관된 향상 |
| 다양한 기반 학습 방법 | ✓ CE, LDAM-DRW 모두 향상 | ✓ CE, Focal, LDAM 모두 향상 |
| 다양한 데이터셋 규모 | ✓ CIFAR → ImageNet | ✓ CIFAR → iNaturalist |
| 다양한 불균형 유형 | - | ✓ Long-tail, Step imbalance 모두 |
| 다양한 반/자기지도 방법 | ✓ Pseudo-label, VAT, MT | ✓ Rotation, Jigsaw, MoCo |

### 3.5 라벨 없이 레이블 절반만 사용할 때의 일반화

흥미로운 발견: **레이블 데이터를 50%만 사용 + 비레이블 데이터(5x)**의 조합이 **전체 레이블 데이터만 사용**하는 완전 지도학습 기준선을 능가합니다. 이는 반지도학습 프레임워크의 뛰어난 일반화 능력을 보여줍니다.

---

## 4. 향후 연구에 미치는 영향과 고려사항

### 4.1 향후 연구에 미치는 영향

**패러다임 전환 측면**:
1. **불균형 학습에서 레이블의 역할 재정의**: 기존 연구는 레이블을 최대한 활용하는 방향이었으나, 이 논문은 레이블을 버리는 것(SSP)이 더 유리할 수 있음을 보임
2. **플러그앤플레이(plug-and-play) 프레임워크 제시**: SSP가 기존 모든 불균형 학습 방법과 호환됨 → 향후 연구에서 SSP를 기본 구성 요소로 채택하는 계기 마련
3. **이론-실험의 긴밀한 연결**: 이론적 모델과 실험 결과 간의 일관성 → 이후 연구에 이론적 기반을 제공하는 방법론적 모범 사례

**직접적 파생 연구 방향**:
- 비지도/자기지도 표현 학습과 불균형 학습의 결합 심화
- 더 강력한 대조 학습(contrastive learning)과의 통합
- 반지도학습에서 비레이블 데이터의 불균형 자동 처리 방법 연구

### 4.2 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의**: 아래 연구들은 논문에서 직접 인용되지 않았으며, 제가 학습한 데이터에 기반한 내용입니다. 세부 수치는 원문을 직접 확인하시기 바랍니다.

#### 관련 연구 흐름

**① 대조 학습 + 불균형 학습 결합**

- **Supervised Contrastive Learning for Long-tailed Recognition (ICLR 2022, Cui et al.)**: 지도 대조 손실(supervised contrastive loss)을 불균형 학습에 적용. 본 논문의 SSP 아이디어를 확장하여, 사전학습 단계에서 지도 신호를 활용하는 방향으로 발전

- **Parametric Contrastive Learning (ICCV 2021, Cui et al.)**: 클래스 프로토타입을 학습하는 파라메트릭 대조 학습으로 소수 클래스 일반화 개선

**② 두 단계(decoupling) 학습 프레임워크 발전**

- **Decoupling Representation and Classifier for Long-tailed Recognition (ICLR 2020, Kang et al.)**: 본 논문이 참조한 cRT 방법의 출처. 표현 학습과 분류기를 분리하는 아이디어가 SSP의 두 단계 학습과 유사한 철학을 가짐

**③ 생성 모델 기반 접근**

- 비레이블 데이터를 실제로 수집하기 어려운 경우, 생성 모델(GAN/VAE/Diffusion)로 소수 클래스 데이터를 합성하는 연구들이 활발해짐 → 본 논문의 반지도학습 프레임워크가 실용적 한계를 보이는 시나리오에 대한 보완책

**④ 최신 자기지도 학습 방법과의 통합**

| 방법 | 본 논문 사용 | 이후 통합 가능성 |
|------|-------------|----------------|
| MoCo v1 | ✓ | MoCo v2/v3, SimCLR, BYOL, SimSiam 등으로 확장 |
| Rotation | ✓ | MAE(Masked Autoencoder) 등 생성형 자기지도 방법으로 발전 |

**⑤ 클래스 불균형 + 반지도학습 결합의 심화 연구**

- **DASO (Distribution-Aware Semantics-Oriented Pseudo-label, CVPR 2022)**: 불균형 반지도학습에서 의사 레이블의 클래스 분포를 고려하는 방법 → 본 논문의 반지도학습 프레임워크의 직접적 발전

#### 비교 분석 표

| 측면 | 본 논문 (Yang & Xu, 2020) | 이후 연구 동향 |
|------|--------------------------|---------------|
| 자기지도 방법 | Rotation, MoCo | SimCLR, BYOL, MAE 등 더 강력한 방법 |
| 반지도학습 | Pseudo-label, VAT, MT | 클래스 균형 인식 의사 레이블 |
| 이론적 분석 | Gaussian 모델 | 더 일반적인 분포에 대한 분석 |
| 적용 도메인 | 이미지 분류 | 객체 탐지, 세분화, NLP로 확장 |

---

### 4.3 앞으로 연구 시 고려할 점

**1. 비레이블 데이터의 품질 관리**

반지도학습 접근에서 비레이블 데이터의 관련성과 불균형 비율이 핵심 변수입니다. 실용적 시스템에서는 이를 자동으로 평가하고 필터링하는 메커니즘이 필요합니다:

$$\rho_U < \rho \implies \text{비레이블 데이터가 더 균형적일수록 도움이 됨}$$

**2. 자기지도 방법의 선택 기준 확립**

- 데이터 규모에 따라 최적 SSP 방법이 다름 (소규모: Rotation > MoCo, 대규모: MoCo > Rotation)
- 데이터 도메인(의료, 위성 이미지 등)별로 적합한 자기지도 태스크 설계 필요

**3. 계산 효율성**

SSP는 두 단계 훈련으로 인한 추가 계산 비용이 발생합니다. 효율적인 사전학습 방법이나 단일 단계 통합 방법 연구가 필요합니다.

**4. 공정성(Fairness)과 안전성**

논문 자체가 언급하듯, 불균형 데이터에서의 소수 클래스 처리는 단순 정확도 지표 이상의 윤리적 고려가 필요합니다:
- 의료 진단: 희귀 질환(소수 클래스) 검출 실패의 위험
- 편향된 예측의 공정성 검증 필요

**5. 개방 세계(Open World) 설정으로의 확장**

실제 환경에서는 훈련 클래스에 없는 새로운 클래스(OOD)가 등장할 수 있습니다. 불균형 학습 + 개방 세계 인식의 결합 연구가 필요합니다.

**6. 멀티모달 및 다양한 도메인으로의 확장**

현재 주로 이미지 분류에 집중되어 있으나, NLP, 의료, 시계열 데이터 등 다양한 도메인에서의 검증 및 맞춤 방법론 개발이 필요합니다.

**7. 더 엄밀한 이론적 분석**

현재의 Gaussian 모델 기반 이론은 직관적이지만 실제 복잡한 데이터 분포를 완전히 반영하지 못합니다. PAC 학습 이론, Rademacher complexity 등을 활용한 더 일반적인 이론적 보장 연구가 필요합니다.

---

## 참고 자료

**주요 논문 (PDF 직접 참조)**:
- Yang, Y., & Xu, Z. (2020). **"Rethinking the Value of Labels for Improving Class-Imbalanced Learning"**. *NeurIPS 2020*. arXiv:2006.07529v2.
- 코드: https://github.com/YyzHarry/imbalanced-semi-self

**논문 내 인용 문헌 (직접 확인)**:
- Cao, K., et al. (2019). "Learning imbalanced datasets with label-distribution-aware margin loss." *NeurIPS 2019*. [논문 내 참조 7]
- Kang, B., et al. (2019). "Decoupling representation and classifier for long-tailed recognition." arXiv:1910.09217. [논문 내 참조 25]
- He, K., et al. (2019). "Momentum contrast for unsupervised visual representation learning." arXiv:1911.05722. [논문 내 참조 19]
- Gidaris, S., et al. (2018). "Unsupervised representation learning by predicting image rotations." arXiv:1803.07728. [논문 내 참조 16]
- Cui, Y., et al. (2019). "Class-balanced loss based on effective number of samples." *CVPR 2019*. [논문 내 참조 11]
- Liu, Z., et al. (2019). "Large-scale long-tailed recognition in an open world." *CVPR 2019*. [논문 내 참조 33]
- Oliver, A., et al. (2018). "Realistic evaluation of deep semi-supervised learning algorithms." *NeurIPS 2018*. [논문 내 참조 39]
- Lee, D. (2013). "Pseudo-label: The simple and efficient semi-supervised learning method." *ICML Workshop*. [논문 내 참조 30]
