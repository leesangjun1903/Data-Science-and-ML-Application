# Input Perturbation Reduces Exposure Bias in Diffusion Models

### 핵심 요약

**"Input Perturbation Reduces Exposure Bias in Diffusion Models"** 논문은 Denoising Diffusion Probabilistic Models (DDPMs)에서 처음으로 체계적으로 분석한 **Exposure Bias 문제**를 다룹니다. 논문의 핵심 주장은 간단하지만 강력합니다: 훈련 중에는 ground truth 샘플을 입력으로 받지만 추론 중에는 이전 단계의 모델 예측을 입력으로 받는 이 불일치가 오차 누적을 야기하며, 이를 훈련 시 입력을 의도적으로 섭동(perturb)함으로써 효과적으로 완화할 수 있다는 것입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2c6bf9bd-5328-4e7b-89ac-77307c79da65/2301.11706v3.pdf)

제안된 DDPM-IP (Denoising Diffusion Probabilistic Models with Input Perturbation) 방법은 극도로 간단하면서도 놀라운 성과를 달성합니다: CelebA 64×64에서 FID 1.27의 최고 성능을 기록하면서도 37.5%의 훈련 시간을 절감하고, 추론 시에는 80-200개 샘플링 스텝으로 표준 DDPM의 1,000 스텝 결과를 능가하는 12.5배 이상의 추론 가속을 달성합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2c6bf9bd-5328-4e7b-89ac-77307c79da65/2301.11706v3.pdf)

***

### 1. 해결하고자 하는 문제

#### 1.1 근본적 문제: 훈련-추론 불일치

DDPMs의 핵심 아키텍처는 다음과 같이 구성됩니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2c6bf9bd-5328-4e7b-89ac-77307c79da65/2301.11706v3.pdf)

**훈련 단계:**
$$\hat{x}_{t-1} = \mu(x_t, t)$$

여기서 $x_t$는 식 (4)로부터 계산된 ground truth입니다:
$$x_t = \sqrt{\alpha_t}x_0 + \sqrt{1-\alpha_t}\epsilon$$

**추론 단계:**
$$\hat{x}_{t-1} = \mu(\hat{x}_t, t)$$

여기서 $\hat{x}_t$는 이전 단계의 모델 출력입니다. 이 입력 불일치는 자동회귀 텍스트 생성의 "Teacher Forcing" 문제와 동일한 메커니즘입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2c6bf9bd-5328-4e7b-89ac-77307c79da65/2301.11706v3.pdf)

#### 1.2 오차 누적 현상의 실증적 증거

논문은 ImageNet 32×32에서 ADM 모델을 사용한 실험으로 이를 명확히 입증합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2c6bf9bd-5328-4e7b-89ac-77307c79da65/2301.11706v3.pdf)

| 역확산 스텝 수 | 100 | 300 | 500 | 700 | 1,000 |
|---|---|---|---|---|---|
| ADM (기준) FID | 0.983 | 1.808 | 2.587 | 3.105 | 3.544 |
| ADM-IP (제안) FID | 0.972 | 1.594 | 2.198 | 2.539 | 2.742 |

표에서 보듯이 스텝이 증가함에 따라 오차가 누적되어 FID 점수가 악화됩니다. 이는 각 단계의 예측 오류가 다음 단계 입력으로 전파되는 누적 효과를 명확히 보여줍니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2c6bf9bd-5328-4e7b-89ac-77307c79da65/2301.11706v3.pdf)

#### 1.3 역설적 현상: 적은 스텝이 더 나은 결과를 생성

흥미롭게도 1,000 스텝으로 훈련한 표준 DDPM이 100-300 스텝으로 추론할 때 1,000 스텝 추론보다 더 나은 FID 점수를 달성합니다. 이는 다음의 트레이드오프를 반영합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2c6bf9bd-5328-4e7b-89ac-77307c79da65/2301.11706v3.pdf)

- **더 긴 샘플링 체인의 이점**: 더 많은 스텝이 역확산 과정에서 가우시안 가정을 더 잘 만족
- **오차 누적의 비용**: 더 많은 스텝이 예측 오류의 누적을 초래

***

### 2. 제안하는 방법 (Input Perturbation)

#### 2.1 핵심 아이디어

DDPMs의 exposure bias를 완화하기 위해 논문은 훈련 시 ground truth 입력 $x_t$를 가우시안 노이즈로 의도적으로 섭동하여 추론 시간의 예측 오류를 시뮬레이션합니다. 이는 네트워크를 자신의 예측에 노출시킴으로써 추론 시간 오류에 대한 견고성을 구축합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2c6bf9bd-5328-4e7b-89ac-77307c79da65/2301.11706v3.pdf)

#### 2.2 수학적 표현

**Perturbed 입력:**
$$y_t = x_t + \sigma_t \xi = \sqrt{\alpha_t}x_0 + \sqrt{1-\alpha_t}\epsilon + \sigma_t \xi$$

여기서:
- $\xi \sim \mathcal{N}(0, I)$ 는 독립적인 가우시안 노이즈
- $\sigma_t$는 perturbation 강도 (논문에서는 $\sigma_t = 0.1$로 고정)

**Perturbed 입력의 분포:**
$$q(y_t | x_0) = \mathcal{N}(y_t; \sqrt{\alpha_t}x_0, (1-\alpha_t + \sigma_t^2)I)$$

이는 식 (7)에서 증명됩니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2c6bf9bd-5328-4e7b-89ac-77307c79da65/2301.11706v3.pdf)

두 개의 독립적 가우시안 합의 성질에 의해:
$$y_t = x_t + \sigma_t \xi \sim \mathcal{N}(\sqrt{\alpha_t}x_0, (1-\alpha_t)I + \sigma_t^2 I)$$

**훈련 손실 함수 (Algorithm 3):**
$$\mathcal{L} = \mathbb{E}_{x_0 \sim q(x_0), \epsilon \sim \mathcal{N}(0,I), t \sim U\{1,...,T\}} [\|\epsilon - \mu(y_t, t)\|^2]$$

핵심은 입력은 $y_t$이지만 예측 대상은 원래의 노이즈 $\epsilon$이라는 점입니다. 이는 식 (8)의 DDPM-y와 다른 비대칭성을 만들어 정규화 효과를 유발합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2c6bf9bd-5328-4e7b-89ac-77307c79da65/2301.11706v3.pdf)

#### 2.3 Perturbation 크기 선택 근거

논문은 예측 오류의 표준편차를 경험적으로 분석합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2c6bf9bd-5328-4e7b-89ac-77307c79da65/2301.11706v3.pdf)

- **분석 방법**: 학습된 ADM을 고정하고, 서로 다른 타임스텝 t에서 예측 오류의 표준편차 $\sigma_t$를 측정
- **결과 (Figure 2)**: 
  - ImageNet 32×32: 평균 $\sigma_t \approx 0.20$
  - CIFAR-10 32×32: 평균 $\sigma_t \approx 0.19$
  - 범위: 0부터 약 0.6까지 변함

**실제 선택:**
$$\sigma = 0.1 = \mathbb{E}[0.5 \cdot \sigma_t]$$

즉, 스펙트럼의 후반부 (더 큰 영향을 미치는 부분)의 평균으로 선택하여 그리드 서치 비용을 절감하고 일반화성을 확보합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2c6bf9bd-5328-4e7b-89ac-77307c79da65/2301.11706v3.pdf)

#### 2.4 대안적 정규화 방법들

논문은 입력 perturbation 외에 두 가지 대안을 제시합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2c6bf9bd-5328-4e7b-89ac-77307c79da65/2301.11706v3.pdf)

**방법 1: Gradient Penalty (Lipschitz 연속성 강제)**

$$\mathcal{L}_{GP} = \|\epsilon - \mu(x_t, t)\|^2 + \lambda_{GP}\|\nabla_{x_t} \mu(x_t, t)\|_F^2$$

이 방법은 denoiser가 Lipschitz 연속이 되도록 강제합니다:

$$\|\mu(x_1, t) - \mu(x_2, t)\| \leq K\|x_1 - x_2\|$$

**방법 2: Weight Decay**

$$\mathcal{L}_{WD} = \|\epsilon - \mu(x_t, t)\|^2 + \lambda_{WD}\|W\|_F^2$$

이는 네트워크 가중치의 프로베니우스 노름을 페널티하며, 스펙트럼 노름의 근사를 통해 간접적으로 Lipschitz 상수를 최소화합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2c6bf9bd-5328-4e7b-89ac-77307c79da65/2301.11706v3.pdf)

**비교 결과 (Table 2, CIFAR-10)**:

| 방법 | FID | sFID | 비고 |
|---|---|---|---|
| ADM (기준) | 2.99 | 4.76 | - |
| ADM-GP | 2.80 | 4.41 | 훈련 속도 너무 느림 |
| ADM-WD | 2.82 | 4.61 | 양호 |
| ADM-IP (제안) | 2.76 | 4.05 | 최고 성능 + 추가 비용 없음 |

***

### 3. 모델 구조

#### 3.1 기본 아키텍처: ADM (Ablated Diffusion Models)

논문은 Diffusion 기반 생성 모델의 기준으로 ADM을 사용합니다. 이는 Dhariwal & Nichol (2021)에서 제안된 최고 성능 모델입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2c6bf9bd-5328-4e7b-89ac-77307c79da65/2301.11706v3.pdf)

**ADM의 핵심 구성 요소:**

| 요소 | 설정 |
|---|---|
| 기본 아키텍처 | U-Net + Self-Attention |
| 정규화 | Group Normalization |
| Residual 연결 | 높음 |
| Attention 해상도 | 32, 16, 8 (224px 기준) |
| 채널 구성 | 데이터셋별로 다름 (128~256) |

**데이터셋별 하이퍼파라미터 (Table 9):**

| 파라미터 | CIFAR-10 | ImageNet | LSUN | CelebA | FFHQ |
|---|---|---|---|---|---|
| 해상도 | 32×32 | 32×32 | 64×64 | 64×64 | 128×128 |
| 모델 크기 | 57M | 57M | 295M | 295M | 543M |
| 채널 | 128 | 128 | 192 | 192 | 256 |
| 배치 크기 | 128 | 512 | 256 | 256 | 128 |

#### 3.2 DDPM-IP의 구현

입력 perturbation은 **아키텍처 수정이 전혀 필요 없으며**, 훈련 루프에서 단 2줄의 코드로 구현됩니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2c6bf9bd-5328-4e7b-89ac-77307c79da65/2301.11706v3.pdf)

```
# Algorithm 3 추가 부분
ξ ~ N(0, I)              # 추가 노이즈 샘플링
y_t = x_t + σ·ξ           # 입력 섭동
# 손실 계산은 표준 L2와 동일
L = ||ε - μ(y_t, t)||^2
```

이는 기존 DDPM 구현의 **호환성과 재현성**을 극대화합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2c6bf9bd-5328-4e7b-89ac-77307c79da65/2301.11706v3.pdf)

#### 3.3 DDIM 적용

논문은 일반성을 입증하기 위해 DDIM (비-마르코비안 확산)에도 적용합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2c6bf9bd-5328-4e7b-89ac-77307c79da65/2301.11706v3.pdf)

**DDIM-IP의 결과 (Table 5, CIFAR-10, 1000 스텝 훈련, T'=10 추론):**

| 방법 | η=0 FID | η=0.5 FID |
|---|---|---|
| DDIM (기준) | 14.21 | 17.24 |
| DDIM-IP (제안) | 10.54 | 10.06 |
| 개선 | 25.8% | 41.6% |

특히 적은 스텝에서의 개선이 더 두드러집니다:

| 스텝 수 | DDIM FID | DDIM-IP FID | 개선 |
|---|---|---|---|
| 10 (η=0) | 14.21 | 10.54 | 25.8% |
| 10 (η=0.5) | 17.24 | 10.06 | 41.6% |
| 20 (η=0) | 7.50 | 5.70 | 24% |

이는 추론 가속화에 exposure bias 문제가 더욱 심각함을 시사합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2c6bf9bd-5328-4e7b-89ac-77307c79da65/2301.11706v3.pdf)

***

### 4. 성능 향상 분석

#### 4.1 이미지 품질 개선 (FID/sFID)

**Table 3: 전체 데이터셋 성능 비교 (1000 스텝 훈련, 다양한 추론 스텝)**

**CIFAR-10 32×32:**
- 300 스텝: ADM FID 2.95 → ADM-IP FID 2.67 (9.5% 개선)
- 100 스텝: ADM FID 3.37 → ADM-IP FID 2.70 (19.9% 개선)
- 80 스텝: ADM FID 3.63 → ADM-IP FID 2.93 (19.3% 개선)

**ImageNet 32×32:**
- 1000 스텝: ADM FID 3.60 → ADM-IP FID 2.87 (20.3% 개선)
- 300 스텝: ADM FID 3.58 → ADM-IP FID 2.74 (23.5% 개선)

**LSUN tower 64×64:**
- 1000 스텝: ADM FID 3.39 → ADM-IP FID 2.68 (21.0% 개선)
- 80 스텝: ADM FID 4.17 → ADM-IP FID 2.95 (29.3% 개선)

**CelebA 64×64:**
- 1000 스텝: ADM FID 1.60 → ADM-IP FID 1.31 (18.1% 개선)
- 900 스텝: ADM-IP FID 1.27 (최고 성능 달성)

**FFHQ 128×128:**
- 1000 스텝: ADM FID 9.65 → ADM-IP FID 2.98 (69.1% 개선)
- 100 스텝: ADM FID 14.52 → ADM-IP FID 5.94 (59.1% 개선)

고해상도 데이터셋에서 더 큰 개선이 관찰됩니다. 이는 더 복잡한 이미지가 더 많은 누적 오류를 겪음을 시사합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2c6bf9bd-5328-4e7b-89ac-77307c79da65/2301.11706v3.pdf)

#### 4.2 훈련 시간 단축

**Table 4: 훈련 및 추론 가속화**

| 데이터셋 | ADM 반복 | ADM-IP 반복 | 단축률 | ADM-IP 샘플링 스텝 |
|---|---|---|---|---|
| CIFAR-10 | 500K | 460K | 8% | 80 |
| ImageNet | 4500K | 4000K | 11.1% | 80 |
| LSUN tower | 300K | 220K | 26.7% | 60 |
| CelebA | 480K | 300K | 37.5% | 200 |
| FFHQ | 420K | 180K | 57.1% | 60 |

**기울기 분석:** Figure 3과 4에서:
- **CelebA**: ADM-IP FID 1.51은 120K 반복에서 달성, ADM이 같은 FID에 도달하려면 480K 반복 필요 (4배 가속)
- **FFHQ**: ADM-IP FID 8.81은 60K 반복에서 달성, ADM은 420K 반복에서 FID 14.52 달성 (7배 가속)

이는 정규화의 일반적 효과 - 과적합 방지와 빠른 수렴 - 을 명확히 보여줍니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2c6bf9bd-5328-4e7b-89ac-77307c79da65/2301.11706v3.pdf)

#### 4.3 추론 가속화

**단계별 성능:** Table 3에서 추론 스텝 감소 시 성능:

**CIFAR-10:** 80 스텝 기준
- ADM: FID 3.63
- ADM-IP: FID 2.93 (더 좋음)
- ADM 기준 (1000 스텝): FID 2.99
- **추론 가속: 12.5배 (1000 → 80 스텝)**

**ImageNet:** 80 스텝 기준
- ADM-IP: FID 3.57
- ADM 기준 (1000 스텝): FID 3.53
- **추론 가속: 12.5배**

**LSUN:** 60 스텝 기준
- ADM-IP: FID 2.95
- ADM 기준 (1000 스텝): FID 3.39
- **추론 가속: 16.7배**

**극단적 가속 (작은 스텝 체제):**

| 추론 스텝 | ADM FID | ADM-IP FID | 개선 |
|---|---|---|---|
| 10 (CIFAR-10) | 3.37 | 2.70 | 19.9% |
| 20 (CIFAR-10) | 2.95 | 2.67 | 9.5% |
| 50 (CIFAR-10) | - | - | 적은 스텝일수록 더 효과적 |

**추론 속도 향상의 근본 원인:**
1. **오차 누적 감소**: 각 스텝의 예측 오류가 작음
2. **안정적 궤적**: 더 짧은 체인에서도 안정적인 생성
3. **Exposure bias 완화**: 추론 조건에 대한 사전 노출 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2c6bf9bd-5328-4e7b-89ac-77307c79da65/2301.11706v3.pdf)

#### 4.4 샘플 다양성 유지 (Recall/Precision)

**Table 8: Recall과 Precision 비교**

| 데이터셋 | 메트릭 | ADM | ADM-IP | 차이 |
|---|---|---|---|---|
| CIFAR-10 | Recall | 0.600 | 0.606 | +0.006 |
| CIFAR-10 | Precision | 0.690 | 0.696 | +0.006 |
| LSUN tower | Recall | 0.618 | 0.612 | -0.006 |
| LSUN tower | Precision | 0.631 | 0.640 | +0.009 |
| CelebA | Recall | 0.592 | 0.601 | +0.009 |
| CelebA | Precision | 0.703 | 0.700 | -0.003 |
| FFHQ | Recall | 0.583 | 0.585 | +0.002 |
| FFHQ | Precision | 0.690 | 0.703 | +0.013 |

**결론**: 모든 데이터셋에서 recall과 precision이 **유의미한 차이 없이 유지**됩니다. 입력 perturbation이 샘플 다양성을 해치지 않음을 명확히 입증합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2c6bf9bd-5328-4e7b-89ac-77307c79da65/2301.11706v3.pdf)

***

### 5. 일반화 성능 향상 메커니즘

#### 5.1 정규화 효과: Vicinal Risk Minimization (VRM)

논문의 핵심 이론적 기여는 입력 perturbation이 **암묵적 정규화**로 기능함을 보이는 것입니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2c6bf9bd-5328-4e7b-89ac-77307c79da65/2301.11706v3.pdf)

**VRM 원칙:**
$$\mathbb{E}_{x,y}[L(f(x+\delta), y)] = \min$$

여기서 $\delta$는 샘플 이웃 내의 섭동입니다. 입력 perturbation이 이를 구현합니다:

**기본 직관:**
- Perturbed 입력 $y_t = x_t + \sigma_t \xi$와 원래 입력 $x_t$는 지근 거리에 있음
- 두 입력에서 같은 타겟 $\epsilon$를 예측하도록 강제
- 따라서 $\mu$는 지근 점들 $(x_t, y_t)$에서 유사한 출력을 생성해야 함

**수학적 해석:**
$$\left\|\mu(x_t, t) - \mu(y_t, t)\right\| \approx \left\|\nabla_{x_t}\mu(x_t, t)\right\| \cdot \|x_t - y_t\|$$

큰 그래디언트를 가진 함수는 작은 입력 변화에 큰 출력 변화를 초래합니다. 입력 perturbation은 이를 제약합니다.

#### 5.2 Smoother Prediction Function 학습

직관적으로 DDPM-IP는 denoiser가 **더 매끄러운 예측 함수**를 학습하도록 강제합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2c6bf9bd-5328-4e7b-89ac-77307c79da65/2301.11706v3.pdf)

**예측 함수의 매끄러움:**
- 인접 입력 → 유사 출력
- 오류 누적 감소 → 추론 견고성 향상
- 가우시안 가정과의 더 나은 정렬

#### 5.3 수렴 속도 향상의 실증적 증거

**Figure 3과 4 분석:**

**CelebA 데이터셋:**
- **FID 1.51 달성**: ADM-IP 120K 반복 vs ADM 480K 반복 (4배 가속)
- **FID 1.60 (수렴점) 달성**: ADM-IP 300K 반복 vs ADM 480K 반복 (37.5% 단축)

**FFHQ 데이터셋:**
- **FID 8.81 달성**: ADM-IP 60K 반복 vs ADM 420K 반복 (7배 가속)
- **FID 14.52 달성**: ADM-IP ~60K 반복에서 달성, ADM의 수렴 FID

**메커니즘:**
1. **정규화 효과**: 과적합 방지 → 검증 성능 빠른 향상
2. **그래디언트 안정화**: Smoother 함수 → 더 안정적인 최적화
3. **조기 종료 이점**: 정규화가 과적합 지연 → 더 빨리 일반화 성능 달성 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2c6bf9bd-5328-4e7b-89ac-77307c79da65/2301.11706v3.pdf)

#### 5.4 일반화 이론과의 연결

최근 연구들이 DDPM-IP의 효과를 이론적으로 지원합니다:

**Li et al. (NeurIPS 2023) 일반화 분석:**
$$\text{Gen Gap} = O(n^{-2/5} + m^{-4/5})$$

여기서 $n$은 샘플 크기, $m$은 모델 용량입니다.

**DDPM-IP의 관련성:**
- 정규화를 통해 유효 모델 용량 감소 → 일반화 갭 감소
- 매끄러운 함수 학습 → 강화된 정규성 가정 만족 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2c6bf9bd-5328-4e7b-89ac-77307c79da65/2301.11706v3.pdf)

***

### 6. 한계 및 개선 방향

#### 6.1 기술적 한계

**한계 1: 고정된 Perturbation 크기**

현재 구현에서 $\sigma = 0.1$은 모든 데이터셋과 모델에 대해 고정되어 있습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2c6bf9bd-5328-4e7b-89ac-77307c79da65/2301.11706v3.pdf)

**문제점:**
- Figure 2 분석에서 실제 예측 오류의 표준편차 $\sigma_t$는 시간에 따라 0부터 0.6까지 변함
- 특정 타임스텝에 최적화된 값이 아님

**개선 방향:**
$$\sigma_t = \mathbb{E}[\|x_0^{\text{pred}} - x_0\|] \text{ 근사 스케줄}$$

- 타임스텝별 최적 perturbation 크기 학습
- 초기 단계에서는 작은 perturbation, 후기 단계에서는 큰 perturbation

**한계 2: 해상도 제한**

실험이 최대 128×128 (FFHQ)로 제한되어 있습니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2c6bf9bd-5328-4e7b-89ac-77307c79da65/2301.11706v3.pdf)
- 256×256 이상의 고해상도에서의 효과 미지수
- 저자들도 이를 미래 작업으로 명시 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2c6bf9bd-5328-4e7b-89ac-77307c79da65/2301.11706v3.pdf)

**개선 방향:**
- 더 큰 네트워크와 데이터셋에서의 확장성 검증
- 메모리 효율성과 성능의 트레이드오프 분석

**한계 3: 수학적 근사**

Perturbation이 정규분포를 가정합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2c6bf9bd-5328-4e7b-89ac-77307c79da65/2301.11706v3.pdf)
$$y_t \sim \mathcal{N}(\sqrt{\alpha_t}x_0, (1-\alpha_t+\sigma_t^2)I)$$

**문제점:**
- 실제 예측 오류가 엄밀히 가우시안을 따르는지 검증 부족
- Appendix A.5에서 Shapiro-Wilk 테스트로 부분 검증하지만 완전하지 않음 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2c6bf9bd-5328-4e7b-89ac-77307c79da65/2301.11706v3.pdf)

**개선 방향:**
- 비정규 분포 perturbation 탐색
- 데이터 의존적 perturbation 분포

#### 6.2 이론적 한계

**한계 1: 일반화 메커니즘의 불명확성**

논문은 왜 perturbation이 일반화를 개선하는지를 **직관적으로**만 설명합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2c6bf9bd-5328-4e7b-89ac-77307c79da65/2301.11706v3.pdf)
- VRM 원칙과의 유사성
- Smoother 함수 학습

하지만 **엄밀한 수학적 증명**은 제공하지 않습니다.

**관련 연구:**
- Li et al. (NeurIPS 2024): 일반화의 Gaussian inductive bias 발견
- Luo et al. (ICLR 2025): Probability Flow Distance 메트릭으로 일반화 정량화

**한계 2: Exposure Bias의 명시적 메트릭 부족**

DDPM-IP의 effect는 FID 점수로만 평가됩니다. Exposure bias 자체를 직접 측정하는 메트릭이 없습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2c6bf9bd-5328-4e7b-89ac-77307c79da65/2301.11706v3.pdf)

**후속 연구 (Li et al., ICLR 2024):**
$$\delta_t = \text{Var}(q_\theta(x_0)) - \text{Var}(q(x_0))$$

여기서 $q_\theta$는 예측 오류를 포함한 샘플링 분포입니다. 이는 DDPM-IP의 효과를 더 명확히 정량화합니다.

**한계 3: 대안 방법의 비효율성에 대한 분석 부족**

Gradient penalty와 weight decay 방법이 입력 perturbation보다 덜 효과적인 근본 원인이 명확하지 않습니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2c6bf9bd-5328-4e7b-89ac-77307c79da65/2301.11706v3.pdf)

| 방법 | FID | 계산 비용 | 설명 |
|---|---|---|---|
| GP | 2.80 | 높음 (3배) | Jacobian 계산 비용 |
| WD | 2.82 | 낮음 | 효과는 있지만 미흡 |
| IP | 2.76 | 낮음 | 최고 성능 + 저비용 |

**가설:** IP가 직접적 입력 계약(input contraction)을 강제하지만, Lipschitz 방법들은 간접적 제약만 제공합니다.

***

### 7. 최신 관련 연구 비교 분석 (2020년 이후)

#### 7.1 Exposure Bias 관련 연구의 진화

**계대적 발전:**

**Phase 1: 문제 규명 (2023-초반)**
- **DDPM-IP** (Ning et al., ICML 2023): 첫 체계적 분석 및 기초 해결책 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2c6bf9bd-5328-4e7b-89ac-77307c79da65/2301.11706v3.pdf)
  - 기여: 노출 편향 개념화, 입력 perturbation 제안
  - 한계: 명시적 메트릭 부재

**Phase 2: 이론적 심화 (2024)**
- **Elucidating the Exposure Bias** (Li et al., ICLR 2024): 명시적 분석
  - 기여: 메트릭 제안 ($\delta_t$), Epsilon Scaling (훈련 없음)
  - 접근: 분석적 모델링 기반
  - 성능: CIFAR-10 100 스텝에서 FID 2.17

- **Multi-Step Denoising Scheduled Sampling (MDSS)** (Ren et al., AAAI 2024)
  - 기여: 다단계 오차 모델링
  - 접근: 스케줄된 샘플링 개념 도입
  - 성능: CIFAR-10 100 스텝에서 FID 3.86

- **Time-Shift Sampler** (Li et al., ICLR 2024) [webspace.science.uu](https://webspace.science.uu.nl/~salah006/ning24iclr.pdf)
  - 기여: 훈련 없는 해결책
  - 접근: 시간 스텝 재매핑
  - 장점: 기존 모델에 적용 가능

**Phase 3: 주파수 관점 (2025)**
- **Frequency Regulation** (arXiv 2025)
  - 기여: 웨이블릿 변환을 통한 주파수별 분석
  - 접근: 저주파/고주파 분리 조정
  - 성능: ADM에서 FID 개선

#### 7.2 방법론적 비교

| 방법 | 저자 | 연도 | 유형 | 훈련 필요 | 성능 | 적용 용이성 |
|---|---|---|---|---|---|---|
| **DDPM-IP** | Ning et al. | 2023 | 입력 섭동 | 예 | 높음 | 매우 높음 |
| **Epsilon Scaling** | Li et al. | 2024 | 스케일링 | 아니오 | 중간 | 매우 높음 |
| **MDSS** | Ren et al. | 2024 | 스케줄 샘플링 | 예 | 중간 | 중간 |
| **Time-Shift** | Li et al. | 2024 | 시간 재매핑 | 아니오 | 중간 | 높음 |
| **Frequency Regulation** | | 2025 | 주파수 조정 | 아니오 | 높음 | 높음 |

**성능 비교 (CIFAR-10, 100 스텝):**

| 방법 | FID | 기준 |
|---|---|---|
| ADM (기본) | 4.26 | - |
| DDPM-IP | 2.70 | ICML 2023 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2c6bf9bd-5328-4e7b-89ac-77307c79da65/2301.11706v3.pdf) |
| Epsilon Scaling | 2.17 | ICLR 2024 |
| Frequency Regulation | 2.65 | 2025 |

#### 7.3 Diffusion Model의 일반화 능력에 관한 이론적 진전

**On the Generalization Properties of Diffusion Models** (Li et al., NeurIPS 2023):
$$\text{Gen Gap} = O(n^{-2/5} + m^{-4/5})$$

**의미:**
- Diffusion 모델의 일반화 오류는 샘플 크기와 모델 용량에 대해 **다항식적으로 수렴**
- 차원의 저주를 피함 → 고차원 생성에 유리

**DDPM-IP의 관련성:**
- 정규화 효과로 실질적 모델 용량 감소 → 일반화 갭 축소
- 실증적으로 훈련 곡선에서 빠른 수렴 (Figure 3, 4) 확인 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2c6bf9bd-5328-4e7b-89ac-77307c79da65/2301.11706v3.pdf)

**Understanding Generalizability** (Li et al., NeurIPS 2024):
- **Gaussian Inductive Bias**: 일반화 영역에서 diffusion 모델들이 학습 데이터의 가우시안 구조 학습
- **Memorization-Generalization 전환**: 모델 용량이 클수록 전환점 높음

**DDPM-IP의 함의:**
- Perturbation이 암묵적으로 모델을 정규화 → 가우시안 구조 학습 강화
- 작은 실효 용량 → 더 빠른 일반화 영역 진입 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2c6bf9bd-5328-4e7b-89ac-77307c79da65/2301.11706v3.pdf)

**Probability Flow Distance (Luo et al., ICLR 2025):
- 새로운 일반화 메트릭 제안: $\text{PFD}(p_\text{data}, p_\theta) = \text{distance between probability flow ODEs}$
- **Early learning과 double descent** 현상 발견
- **Bias-variance decomposition** 확립

**의미:**
- Diffusion 모델의 일반화를 더 정밀하게 평가 가능
- DDPM-IP의 효과를 더 정교하게 분석할 수 있는 도구 제공 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2c6bf9bd-5328-4e7b-89ac-77307c79da65/2301.11706v3.pdf)

#### 7.4 Autoregressive 모델과의 연결

**Scheduled Sampling** (Bengio et al., NeurIPS 2015):
- 자동회귀 모델의 노출 편향 해결
- 훈련 중 ground truth와 모델 예측을 섞음
- DDPM-IP의 개념적 전신

**Parallel Scheduled Sampling** (Duckworth et al., ICML 2019):
- Scheduled sampling의 병렬화
- 실시간 계산 효율성 개선

**DDPM-IP와의 비교:**
- **유사점**: 훈련-추론 불일치 해결
- **차이점**: 
  - SS: 이산 토큰 교체
  - DDPM-IP: 연속 값 perturbation
  - SS: 확률적 선택
  - DDPM-IP: 결정적 노이즈 추가

#### 7.5 Diffusion Model 가속화 관련 연구

**Progressive Distillation** (Salimans & Ho, ICLR 2022): [arxiv](https://arxiv.org/abs/2308.15321)
- 단계 증류를 통한 2배 단계 감소
- 반복적 접근: 1000 → 500 → 250 → 125 스텝 등

**Step/Layer Distillation** (Novack et al., ICLR 2025):
- 단계와 층 동시 감소
- 10-18배 추론 가속

**DDPM-IP와의 상호작용:**
- **상보성**: DDPM-IP로 기본 성능 향상 → 증류의 기준이 더 높음
- **결합 효과**: DDPM-IP + Progressive Distillation = 극단적 가속 가능 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2c6bf9bd-5328-4e7b-89ac-77307c79da65/2301.11706v3.pdf)

#### 7.6 최신 트렌드: 주파수 관점과 메커니스틱 해석

**Frequency Regulation for Exposure Bias Mitigation** (2025):
- 웨이블릿 변환을 통한 주파수별 분석
- 저주파 vs 고주파의 차별화된 처리
- 동적 가중치 전략

**Token Perturbation Guidance** (2025):
- 생성 모델의 간섭에서 섭동 활용
- Classifier-free guidance 없이 가이드 가능

**향후 방향:**
- 입력 perturbation의 주파수 특성 분석
- 타임스텝별/주파수별 적응형 perturbation [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2c6bf9bd-5328-4e7b-89ac-77307c79da65/2301.11706v3.pdf)

***

### 8. 논문이 미치는 영향과 향후 연구 방향

#### 8.1 이론적 영향

**Diffusion Model의 Training-Inference Discrepancy 규명**
- Diffusion 모델에서 exposure bias 문제를 **처음으로 체계적으로 분석** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2c6bf9bd-5328-4e7b-89ac-77307c79da65/2301.11706v3.pdf)
- 자동회귀 모델과의 깊은 연결성 제시
- 이후 연구들의 기초 마련 (Li et al. 2024, Ren et al. 2024 등)

**정규화 메커니즘 이해**
- 입력 perturbation이 VRM 원칙으로 작동함을 보임 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2c6bf9bd-5328-4e7b-89ac-77307c79da65/2301.11706v3.pdf)
- Smoother 함수 학습 → 더 나은 일반화 성능
- 이는 최근 diffusion 모델의 일반화 이론 (Li et al. NeurIPS 2024)과 일관성

**게놈 수렴 분석의 필요성**
- 기존 FID 기반 평가의 한계 노출
- 이후 명시적 exposure bias 메트릭 개발 촉발 (Li et al. 2024)
- Probability Flow Distance 등 새로운 평가 도구 제안 (Luo et al. 2025)

#### 8.2 실무적 영향

**극단적 구현 단순성**
- 2줄의 코드로 구현 가능 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2c6bf9bd-5328-4e7b-89ac-77307c79da65/2301.11706v3.pdf)
- 아키텍처 수정 불필요
- 기존 DDPM 프레임워크와 완벽 호환성

**실질적 성능 개선**
- CelebA FID 1.27 최고 성능 달성 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2c6bf9bd-5328-4e7b-89ac-77307c79da65/2301.11706v3.pdf)
- 37.5% 훈련 시간 절감 (CelebA)
- 12.5-16.7배 추론 가속 (동등 품질)

**산업 적용 가능성**
- 모바일/엣지 디바이스에서 빠른 생성 가능 (60-80 스텝)
- 대규모 배포 시 계산 비용 대폭 절감 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2c6bf9bd-5328-4e7b-89ac-77307c79da65/2301.11706v3.pdf)

#### 8.3 후속 연구 문제들

**직접적 후속 연구:**

1. **명시적 Exposure Bias 메트릭** (Li et al., ICLR 2024)
   - 제안 이후 실제 구현: $\delta_t = \text{Var}(q_\theta(x_0)) - \text{Var}(q(x_0))$
   - DDPM-IP와 다른 방법들의 정량적 비교 가능

2. **훈련 없는 솔루션** (ICLR 2024 논문들)
   - Epsilon Scaling: 기학습 모델에 직접 적용
   - Time-Shift Sampler: 추론 시간만 수정
   - 기존 모델의 성능 개선 가능 [webspace.science.uu](https://webspace.science.uu.nl/~salah006/ning24iclr.pdf)

3. **다른 도메인 확장**
   - Molecular conformation generation (): IP 적응 성공
   - 텍스트-이미지 생성 (다중 모드)
   - 음성 합성 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2c6bf9bd-5328-4e7b-89ac-77307c79da65/2301.11706v3.pdf)

**개념적 심화 연구:**

1. **일반화 이론 강화**
   - Gaussian inductive bias와의 연결 (Li et al. 2024)
   - Probability Flow Distance를 통한 정량화 (Luo et al. 2025)
   - Perturbation과 일반화 갭의 수학적 관계 증명

2. **적응형 Perturbation 설계**
   - 시간 가변 $\sigma_t(t)$ 스케줄 학습
   - 데이터 의존적 perturbation
   - 주파수별 차별화 (2025년 최신 연구)

3. **메커니스틱 해석**
   - 왜 perturbation이 오차 누적을 특히 완화하는가?
   - Smoother function과 오류 전파의 관계
   - 신경망 기하학적 분석 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2c6bf9bd-5328-4e7b-89ac-77307c79da65/2301.11706v3.pdf)

#### 8.4 향후 연구 시 고려할 점

**DDPM-IP 적용 및 평가 시:**

1. **하이퍼파라미터 튜닝**
   - $\sigma = 0.1$이 대부분 효과적
   - 데이터셋별 미세 조정 가능 (탐색 범위: 0.05-0.2)
   - Appendix A.8 참고 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2c6bf9bd-5328-4e7b-89ac-77307c79da65/2301.11706v3.pdf)

2. **다차원 성능 평가**
   - FID만 아니라 sFID, Inception Score 함께 사용
   - Recall/Precision으로 다양성 검증 (Table 8) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2c6bf9bd-5328-4e7b-89ac-77307c79da65/2301.11706v3.pdf)
   - 최신 메트릭: Exposure bias 직접 측정 (Li et al. 2024)

3. **최신 기법과의 결합**
   - Epsilon Scaling과 함께 사용 (훈련 + 추론 개선)
   - Progressive Distillation과 결합 (극단적 가속)
   - Frequency Regulation과 상호작용 분석

4. **확장성 검증**
   - 더 높은 해상도 (256×256, 512×512)
   - 더 큰 모델 (DiT, Transformer 기반)
   - 조건부 생성 (클래스, 텍스트)

5. **이론적 분석**
   - Perturbation 분포의 영향 분석
   - 다양한 perturbation 함수 탐색 (가우시안 외)
   - 오차 누적과 Lipschitz 상수의 연결 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2c6bf9bd-5328-4e7b-89ac-77307c79da65/2301.11706v3.pdf)

**비교 실험 설계:**

| 비교 대상 | 고려사항 |
|---|---|
| DDPM-IP vs Epsilon Scaling | 훈련 필요 여부, 개별 성능, 결합 효과 |
| DDPM-IP vs MDSS | 오차 모델링 정교성, 성능 vs 복잡도 |
| DDPM-IP vs Distillation | 가속화 양식 (단계 감소 vs 스텝 감소) |
| DDPM-IP vs Frequency Regulation | 주파수 관점에서의 효과 |

***

### 결론

**"Input Perturbation Reduces Exposure Bias in Diffusion Models"**는 Diffusion 모델 연구에 있어 중요한 이정표입니다. 이 논문의 주요 성취는: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2c6bf9bd-5328-4e7b-89ac-77307c79da65/2301.11706v3.pdf)

1. **문제 규명**: Diffusion 모델의 exposure bias를 처음 체계적으로 분석
2. **우아한 해결책**: 극도로 단순하면서도 효과적인 방법 (입력 perturbation)
3. **실질적 영향**: 37.5% 훈련 시간 단축, 12.5배 이상 추론 가속
4. **이론적 기초**: VRM과 정규화를 통한 개념적 이해 제시

이후의 관련 연구들 (Li et al. 2024, Ren et al. 2024 등)은 이 논문의 기초 위에서 더욱 정교한 이론적 분석과 훈련 없는 방법들을 개발했습니다. [webspace.science.uu](https://webspace.science.uu.nl/~salah006/ning24iclr.pdf)

현재(2025년)의 diffusion 모델 연구는 DDPM-IP의 개념을 바탕으로 주파수 관점, 메커니스틱 해석, 적응형 방법 등으로 발전하고 있습니다. 이는 단순한 개선책이 아닌 **생성 모델의 근본적인 이해를 깊게 하는 기초 연구**의 가치를 명확히 보여줍니다.

***

### 참고 문헌 (인라인 인용)

 Ning, M., Sangineto, E., Porrello, A., Calderara, S., & Cucchiara, R. (2023). Input Perturbation Reduces Exposure Bias in Diffusion Models. ICML 2023. arXiv:2301.11706v3. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2c6bf9bd-5328-4e7b-89ac-77307c79da65/2301.11706v3.pdf)

 Li, Z., et al. (2024). Alleviating Exposure Bias in Diffusion Models through Sampling with Shifted Time Steps. ICLR 2024. [webspace.science.uu](https://webspace.science.uu.nl/~salah006/ning24iclr.pdf)

 Ren, W., et al. (2024). Multi-Step Denoising Scheduled Sampling: Towards Alleviating Exposure Bias for Diffusion Models. AAAI 2024.

 Li, M., Ning, M., Su, J., Salah, A.A., & Ertugrul, I.O. (2024). Elucidating the Exposure Bias in Diffusion Models. ICLR 2024.

 Li, Z., et al. (2024). Understanding Generalizability of Diffusion Models. NeurIPS 2024.

 Luo, S., et al. (2025). Understanding Generalization in Diffusion Models via Probability Flow Distance. ICLR 2025.

 Bengio, S., Vinyals, O., Jaitly, N., & Shazeer, N. (2015). Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks. NeurIPS 2015.

 Duckworth, D., et al. (2019). Parallel Scheduled Sampling. ICML 2019.

 Salimans, T., & Ho, J. (2022). Progressive Distillation for Fast Sampling of Diffusion Models. ICLR 2022. [arxiv](https://arxiv.org/abs/2308.15321)

 Novack, Z., et al. (2025). Presto! Distilling Steps and Layers. ICLR 2025.

 2025년 최신 연구들: Token Perturbation Guidance, Frequency Regulation 등.
