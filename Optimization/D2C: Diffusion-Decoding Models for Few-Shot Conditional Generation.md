# D2C: Diffusion-Decoding Models for Few-Shot Conditional Generation

### 1. 논문의 핵심 주장 및 주요 기여

D2C(Diffusion-Decoding Models with Contrastive representations)는 **제한된 수의 레이블된 데이터(100개 정도)만으로도 조건부 이미지 생성이 가능한 새로운 생성 모델**을 제안합니다. 논문의 핵심 기여는 다음과 같습니다:[1]

**핵심 기여:**

- **변분 오토인코더(VAE)의 표현 학습 능력과 확산 모델의 샘플 품질을 결합**: 대조 학습(contrastive learning)을 통해 풍부한 잠재 표현을 학습하면서, 잠재 공간 위의 확산 모델을 학습하여 높은 품질의 생성을 달성합니다.

- **선행 분포 불일치 문제 해결**: VAE의 "선행 구멍(prior hole)" 문제를 이론적으로 증명하고 해결합니다.

- **효율적인 조건부 생성**: 단 **100개의 레이블된 예시**로부터 학습하여 새로운 조건에 대한 생성이 가능합니다.

- **경쟁력 있는 성능**: StyleGAN2 대비 **100배 빠른 이미지 조작**을 제공하면서 50-60%의 인간 평가에서 선호됩니다.[1]

***

### 2. 해결하고자 하는 문제

#### 2.1 기존 방법의 한계

D2C가 직면한 핵심 문제들:[1]

| 문제 | 설명 | D2C의 해결책 |
|------|------|-----------|
| **쌍 데이터 부족** | 조건부 생성을 위해 많은 쌍 데이터 필요 | 무조건부 모델에서 학습 후 조건화 |
| **VAE의 낮은 샘플 품질** | VAE는 명시적 인코더를 가지지만 생성 품질 낮음 | 잠재 공간의 확산 모델 도입 |
| **GAN의 역함수 문제** | GAN 역변환 어려움 | 명시적 인코더 구조 유지 |
| **선행 구멍 문제** | VAE의 사후 분포와 선행 분포의 불일치 | 확산 프로세스로 구성적 해결 |

#### 2.2 이상적인 생성 모델의 조건[1]

논문은 Table 1에서 세 가지 필수 특성을 제시합니다:

1. **명시적 x→z 매핑** (자기-지도 학습 가능): VAE, 정규화 흐름 만 해당
2. **선행 구멍 없음** (생성 시 훈련과 동일한 분포 사용): 확산 모델, GAN만 해당
3. **비적대적 훈련** (안정적): VAE, 정규화 흐름, 확산 모델만 해당

D2C는 **세 가지 특성을 모두 만족하는 유일한 방법**입니다.[1]

***

### 3. 제안 방법 (수식 포함)

#### 3.1 생성 프로세스

D2C의 생성 과정은 **확산 단계**와 **디코딩 단계** 두 가지로 구성됩니다.[1]

$$z^{(0)} \sim p^{(0)}(z^{(0)}) := \mathcal{N}(0, I), \quad z^{(1)} \sim p_{\theta}^{(0,1)}(z^{(1)}|z^{(0)}) \quad \text{[확산]}, \quad x \sim p_{\theta}(x|z^{(1)}) \quad \text{[디코딩]}$$

여기서:
- $$z^{(0)}$$: 순수 가우시안 노이즈 (α=0)
- $$z^{(1)}$$: 깨끗한 잠재 표현 (α=1)
- $$p_{\theta}^{(0,1)}(z^{(1)}|z^{(0)})$$: DDIM을 이용한 확산 프로세스[1]

#### 3.2 훈련 목적 함수

D2C의 훈련은 세 가지 손실 함수의 결합입니다:[1]

$$L_{D2C}(\theta, \phi; w) := L_{D2}(\theta, \phi; w) + \lambda L_C(q_\phi)$$

여기서 D2(Diffusion-Decoding) 손실은:[1]

$$L_{D2}(\theta, \phi; w) := \mathbb{E}_{x \sim p_{data}, z^{(1)} \sim q_\phi(z^{(1)}|x)} \left[-\log p_\theta(x|z^{(1)}) + \ell_{diff}(z^{(1)}; w, \theta)\right]$$

**각 구성 요소의 의미:**

1. **재구성 손실** $$-\log p_\theta(x|z^{(1)})$$: 원본 이미지를 복원하는 능력

2. **확산 손실** $$\ell_{diff}(z^{(1)}; w, \theta)$$:[1]

$$\ell_{diff}(z; w, \theta) := \sum_{i=1}^T w(\alpha_i) \mathbb{E}_{\epsilon \sim \mathcal{N}(0,I)} \left[\|\epsilon - \epsilon_\theta(z^{(\alpha_i)}, \alpha_i)\|_2^2\right]$$

여기서 $$z^{(\alpha_i)} = \sqrt{\alpha_i}z^{(1)} + \sqrt{1-\alpha_i}\epsilon$$

3. **대조 손실** $$L_C(q_\phi)$$: MoCo-v2 기반 자기-지도 학습

$$L_{CPC}(g; q_\phi) := \mathbb{E}\left[\frac{1}{n}\sum_{i=1}^n \log \frac{m \cdot g(y_i, w_i)}{g(y_i, w_i) + \sum_{j=1}^{m-1} g(y_i, w_{i,j})}\right]$$[1]

#### 3.3 최대우도 추정과의 관계

**정리 1 (형식적)**: 적절한 가중치 $$\hat{w}$$가 존재하면, $$-L_{D2}$$는 로그-우도의 변분 하한입니다.[1]

$$-L_{D2}(\theta, \phi; \hat{w}) \leq \mathbb{E}_{p_{data}}[\log p_\theta(x)]$$

**증명 스케치**: 확산 항 $$\ell_{diff}$$는 적절한 가중치에서 KL 발산 $$D_{KL}(q_\phi(z^{(1)}|x) \| p_\theta^{(1)}(z^{(1)}))$$의 상한이 되어 VAE 목적 함수를 복원합니다.[1]

***

### 4. 모델 구조

#### 4.1 전체 아키텍처

D2C의 구조는 **세 가지 주요 컴포넌트**로 구성됩니다:[1]

```
입력 이미지
     ↓
[인코더 (q_φ)]  ← 대조 학습을 통한 특성 학습
     ↓
잠재 표현 z^(1)
     ↓
[확산 모델 (p_θ^(0,1))]  ← 잠재 공간의 사전 분포 모델링
     ↓
깨끗한 잠재 z^(1)
     ↓
[디코더 (p_θ)]  ← 이미지 재구성
     ↓
생성 이미지
```

#### 4.2 구체적 구현

**아키텍처 선택:**[1]
- **인코더/디코더**: NVAE(Nouveau VAE) 구조 기반
- **확산 모델**: U-Net 기반 (Denoising Diffusion Implicit Models 사용)
- **대조 학습**: MoCo-v2 (Momentum Contrast) 구현

**주요 하이퍼파라미터:**[1]
- 확산 단계: 100 DDIM 스텝
- 대조 손실 가중치: λ = 10⁻⁴
- 잠재 공간 정규화: 전역 평균과 표준편차로 정규화

***

### 5. 선행 구멍 문제와 해결책

#### 5.1 선행 구멍의 형식적 정의[1]

**정의 1**: 분포 $$p(z), q(z)$$에서 $$supp(q) \subseteq supp(p)$$일 때, 다음을 만족하면 q가 p에 대해 $$(ϵ, δ)$$-선행 구멍을 가집니다:

$$\exists S \in supp(p): \int_S p(z)dz \geq \delta \text{ and } \int_S q(z)dz \leq \epsilon$$

**직관**: 선행 분포의 높은 확률 영역(δ)이 사후 분포의 낮은 확률 영역(ϵ)에 존재합니다. 생성 시 이 "구멍"에서 샘플링되면 품질 저하된 이미지를 얻게 됩니다.[1]

#### 5.2 핵심 이론적 기여: 정리 2[1]

**정리 2 (형식적)**: $$p_\theta(z) = \mathcal{N}(0, I)$$일 때, **KL 발산과 Wasserstein 거리가 작아도 큰 선행 구멍이 존재할 수 있습니다.**

어떤 $$\epsilon > 0$$에 대해 다음을 만족하는 분포 $$q_\phi(z)$$가 존재합니다:
- $$(ϵ, 0.49)$$-선행 구멍 존재
- $$D_{KL}(q_\phi \| p_\theta) \leq \log 2.3$$
- $$W_2(q_\phi, p_\theta) < \gamma$$ (임의의 γ > 0)

**의미**: 기존 VAE 방식의 KL 최소화나 Wasserstein 거리 최소화로는 선행 구멍을 제거할 수 없습니다.[1]

#### 5.3 D2C의 구성적 해결책[1]

확산 모델은 **구성에 의해** 선행 구멍을 제거합니다:

$$q^{(\alpha)}(z^{(\alpha)}) = \mathbb{E}_{z^{(1)} \sim q^{(1)}(z^{(1)})} [\mathcal{N}(\sqrt{\alpha}z^{(1)}, (1-\alpha)I)]$$

$$\alpha \to 0$$일 때: $$D_{KL}(q^{(\alpha)}(z^{(\alpha)}) \| \mathcal{N}(0, I)) \to 0$$

따라서 훈련 시 사용되는 분포($$\alpha=0$$에서의 가우시안)와 생성 시 사용되는 분포($$\alpha=1$$에서의 잠재)가 같은 확산 경로로 연결되어 있습니다.[1]

***

### 6. 성능 향상 및 실험 결과

#### 6.1 무조건부 생성 성능

**표 2, 3의 FID 점수 비교:**[1]

| 데이터셋 | NVAE | DDIM | D2C |
|---------|------|------|-----|
| CIFAR-10 | 36.4 | 4.16 | 10.15 |
| CIFAR-100 | 42.5 | 10.16 | 14.62 |
| CelebA-64 | 13.48 | 6.53 | 5.7 |
| CelebA-HQ-256 | 40.26 | 25.6 | 18.74 |
| FFHQ-256 | 26.02 | - | 13.04 |

**특성 품질 비교 (표 2):**
- **MSE (낮을수록 좋음)**: D2C가 NVAE보다 우수한 재구성 능력
- **선형 분류 정확도 (높을수록 좋음)**: 
  - NVAE: 18.8% (CIFAR-10)
  - D2C: **76.02%** → 대조 학습의 효과[1]

#### 6.2 Few-Shot 조건부 생성

**표 5 - 레이블 기반 100개 샘플로 학습:**[1]

| 조건 | D2C | DDIM | NVAE | 순진한 방법 |
|-----|-----|------|------|----------|
| Male (42%) | 13.44 | 38.38 | 41.07 | 26.34 |
| Blond (15%) | 17.61 | 31.39 | 31.24 | 27.51 |
| Non-Blond (85%) | 8.94 | 9.67 | 16.73 | 3.77 |

**결과 해석**:
- D2C는 소수 클래스(15%)에서 기존 방법 대비 **큰 성능 향상**
- 대조 학습을 통한 표현이 few-shot 학습에 유리[1]

#### 6.3 이미지 조작 성능

**AMT 인간 평가 (그림 5):**[1]

- **Blond 속성**: D2C 51.5% vs StyleGAN2 (선호도)
- **Red Lipstick 속성**: D2C **60.8%** vs StyleGAN2 (선호도)
- **속도**: D2C 0.013초 vs StyleGAN2 8초 → **약 615배 빠름**

**성능 향상의 요인**:
1. D2C는 재구성 손실로 세밀한 특징(눈, 귀걸이, 배경) 보존
2. 최적화 불필요 (StyleGAN2는 인코딩에 시간 소요)
3. 빠른 Langevin 동역학 기반 샘플링[1]

***

### 7. 모델의 일반화 성능 향상 가능성

#### 7.1 대조 학습의 일반화 능력

D2C의 **표현 학습이 일반화 성능을 향상시키는 메커니즘:**[1]

**1. 자기-지도 학습의 전이성**
- MoCo-v2는 대규모 무조건부 데이터에서 풍부한 특성 학습
- 이 특성은 few-shot 조건부 생성에 직접 전이
- 선형 분류 정확도: NVAE 18.8% → D2C 76.02% (CIFAR-10)[1]

**2. 표현의 의미론적 구조**
- 대조 학습은 클래스 간 차별화되고 클래스 내 응집된 표현 생성
- 적은 수의 조건 샘플로도 결정 경계 학습 가능
- 결과: 100개 샘플만으로도 새로운 속성 학습[1]

**3. 확산 기반 선행의 역할**
- 학습된 표현 위의 확산 모델이 표현 공간의 분포 구조 캡처
- 부족한 조건 샘플을 보완하기 위해 훈련 분포의 통계 활용
- CRDI(Conditional Relaxing Diffusion Inversion) 같은 최신 방법도 동일 원리 활용[2]

#### 7.2 크로스 도메인 일반화

**실험적 증거:**[1]

1. **다양한 도메인에서 테스트** (표 2, 3):
   - 자연 이미지 (CIFAR-10/100)
   - 위성 이미지 (fMoW)
   - 얼굴 이미지 (CelebA, FFHQ)
   → 모든 도메인에서 우수한 성능

2. **해상도 일반화**:
   - 32×32 (CIFAR)에서 256×256 (CelebA-HQ, FFHQ)까지 확장
   - 구조 변화 최소 (하이퍼파라미터만 조정)

3. **속성 다양성**:
   - Binary 속성 (Male/Female, Blond/Non-Blond)
   - Positive-Unlabeled 학습 (레이블 부분만 사용)
   - 조작 제약 (특정 속성 + 원본 이미지 유사성)[1]

#### 7.3 이론적 일반화 보장

**일반화 성능 향상의 수학적 기반:**[1]

VAE의 증거 하한(ELBO)은:
$$\log p_\theta(x) \geq -L_{D2}(\theta, \phi; \hat{w}) + H(q_\phi(z^{(1)}|x))$$

이 하한이:
1. **대조 손실** $$L_C$$로 더욱 강화 (표현 질 향상)
2. **확산 선행** $$\ell_{diff}$$로 포괄적 (선행 구멍 제거)

결과적으로:
- 더 나은 표현 $$q_\phi(z|x)$$
- 더 나은 선행 $$p_\theta(z)$$
- **두 가지 모두 일반화 향상에 기여**[1]

#### 7.4 메타-학습 관점의 해석

D2C는 암묵적인 메타-학습이 가능합니다:[1]

1. **외부 루프 (무조건부 학습)**: 풍부한 표현과 강력한 선행 학습
2. **내부 루프 (조건부 적응)**: 적은 수의 조건 샘플로 빠른 적응

이는 FSDM(Few-Shot Diffusion Models)과 메타-학습 기반 방법의 개선으로 이어집니다.[2]

***

### 8. 한계 및 제약 사항

#### 8.1 모델 수준의 한계

**1. 표본 품질 vs 표현 품질 트레이드오프:**[1]
- D2C의 무조건부 생성 FID (CIFAR-10: 10.15)는 DDIM (4.16)보다 높음
- 대조 학습 가중치 λ를 높이면 표현은 좋아지나(78.3%) 재구성 오류 증가
- ResNet 인코더 사용 시 평균 풀링으로 정보 손실

**2. 아키텍처 최적화 미흡:**[1]
- NVAE 기본 구조 사용 (StyleGAN2나 Transformer 같은 최신 아키텍처 미적용)
- 더 나은 아키텍처면 성능 향상 가능성 높음

#### 8.2 방법론적 한계

**1. 조건 타입의 제한:**[1]
- 현재: 이미지 레이블과 조작 제약만 처리
- 미래 확장 필요: 텍스트 설명, 보상 값 등 다양한 조건화

**2. Rejection Sampling의 비효율성:**[1]
- Algorithm 1의 line 4에서 거부 샘플링 사용
- 더 정교한 방법(Langevin 동역학)이 더 효율적일 수 있음

**3. 하이퍼파라미터 의존성:**[1]
- 조작 제약 생성에서 단계 크기 η, 노이즈 수준 α 등 수동 튜닝 필요
- α ∈ [0.65, 0.9] 범위에서는 비교적 안정적이지만 최적화 필요

#### 8.3 사회적 영향 및 윤리적 고려

**1. 편향 상속:**[1]
- D2C는 훈련 데이터의 편향을 충실히 복원하는 재구성 손실 사용
- 공정성 저해 가능성

**2. 딥페이크 악용 우려:**[1]
- 100개 이미지만으로도 얼굴 조작 가능
- 저자는 잠재 공간에서 문제 잠재 변수 거부로 방어 제안

***

### 9. 최신 관련 연구 (2020년 이후)

#### 9.1 VAE의 선행 구멍 문제 관련 연구

**NCP-VAE (Noise Contrastive Prior, 2020):**[3]
- VAE 선행을 대조 추정으로 개선
- KL 발산 기반이 아닌 에너지 기반 선행 학습

**DG-VAE (Density Gap, 2022):**[4]
- 밀도 간격 정규화로 선행 구멍과 사후 붕괴 동시 해결
- D2C의 이론적 기초 선행 연구

#### 9.2 확산 모델의 잠재 공간 활용

**LDM (Latent Diffusion Models, 2022):**[5]
- 픽셀 공간이 아닌 잠재 공간에서 확산 수행
- 계산 효율성 향상으로 실용성 증대
- Stable Diffusion의 기초

**CDM (Conditional Distribution Modelling, 2024):**[2]
- Few-shot 이미지 생성을 위한 조건부 분포 모델링
- D2C와 유사하게 학습 통계 활용으로 편향 제거

#### 9.3 Few-Shot 생성 모델의 최신 발전

**CRDI (Conditional Relaxing Diffusion Inversion, 2024):**[2]
- 확산 역변환을 통한 few-shot 이미지 생성
- 훈련 불필요(training-free) 접근

**FSDM (Few-Shot Diffusion Models, 2022):**[2]
- 조건부 DDPM 기반 few-shot 생성
- 메타-학습과 확산의 결합

**DualAnoDiff (2025):**[2]
- 이상 탐지 이미지 생성을 위한 LoRA 기반 확산
- 듀얼 브랜치로 글로벌-로컬 특성 분리

#### 9.4 대조 학습과 생성 모델의 결합

**FACoG (Contrastive Learning-Based Generative Model, 2024):**[6]
- 대조 학습을 이용한 특성 증강
- 불균형 데이터 분류에서의 응용

**Multi-Modal 접근:**[7]
- 조건부 생성 모델에서 대조 학습 활용
- 의료 이미지 등 특수 도메인 확장

#### 9.5 일반화 성능 관련 최신 연구

**Domain Generalization with Latent Space (2025):**[8]
- 확산 잠재 공간의 의미론적 특성 활용
- 보지 못한 도메인으로의 일반화 개선

**Quantum Diffusion Models (2025):**[9]
- 양자 컴퓨팅 기반 few-shot 학습
- 새로운 계산 패러다임에서의 성능 향상

***

### 10. 논문의 영향과 미래 연구 방향

#### 10.1 학문적 영향

**1. 패러다임 전환:**
- VAE의 "표현 학습" 능력과 확산의 "고품질 생성" 능력 결합
- 이후 LDM, Stable Diffusion 등에 영감

**2. 이론적 기여:**
- 선행 구멍의 형식적 정의와 증명 제공
- VAE 개선을 위한 새로운 관점 제시

**3. 실용적 해결책:**
- 적은 데이터로 조건부 생성 실현
- 이미지 조작의 실시간 가능성 제시

#### 10.2 앞으로의 연구 고려사항

**1. 아키텍처 개선:**
- StyleGAN2, Transformer 같은 최신 아키텍처 통합
- 계산 효율성 향상
- 더 큰 해상도 지원

**2. 다중 모달리티 확장:**
- 텍스트-이미지 쌍 데이터 활용 (DALL-E, CLIP 방식)
- 오디오, 3D 모델 등 다양한 모달리티로 확장
- 반-지도 학습 프레임워크 구축[1]

**3. 조건화 방식 다양화:**
- 복잡한 텍스트 설명으로의 조건화
- 강화 학습 보상과의 결합
- 계층적 조건화[1]

**4. 이론적 심화:**
- 일반화 경계 증명
- 메타-학습과의 이론적 연결
- 확산-VAE 하이브리드의 최적 설정 도출

**5. 공정성과 윤심:**
- 편향 완화 기법 통합
- 프라이버시 보호 (차등 프라이버시)
- 해석 가능성 향상[1]

**6. 현실 적용:**
- 의료 이미지 합성 (데이터 부족 문제)
- 드문 질환 탐지를 위한 데이터 증강
- 개인화 이미지 생성 (추천 시스템 등)

#### 10.3 최신 연구와의 시너지

**2024-2025년 최신 연구들의 방향:**[8][9][2]
- 더 간단한 few-shot 적응 (CRDI의 훈련 불필요 접근)
- 성능 vs 효율의 균형 (QDM의 양자 접근)
- 특수 도메인 최적화 (DualAnoDiff의 이상 탐지)

이들은 D2C의 기초 위에서 구체적 문제별 맞춤형 솔루션을 제시합니다.

***

### 결론

D2C는 **제한된 쌍 데이터로도 고품질 조건부 이미지 생성**이 가능함을 보여주는 획기적인 연구입니다. 이는 VAE의 표현 학습 능력, 확산 모델의 생성 품질, 대조 학습의 일반화 능력을 정교하게 결합함으로써 달성됩니다.

특히 **선행 구멍 문제의 이론적 분석과 해결책**은 VAE 연구에 새로운 방향을 제시했고, 이후 Latent Diffusion Model 등 실용적인 진전으로 이어졌습니다. 앞으로의 연구는 이 기초 위에서 (1) 아키텍처 최적화, (2) 다중 모달리티 통합, (3) 윤리적 고려를 병행하여 더욱 강력하고 책임감 있는 생성 모델을 개발해야 할 것입니다.

***

## 참고문헌

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9749df16-cc97-4676-b805-10a3c00b2928/2106.06819v1.pdf)
[2](https://link.springer.com/10.1631/FITEE.2400395)
[3](https://arxiv.org/abs/2507.02686)
[4](https://ieeexplore.ieee.org/document/10704743/)
[5](https://arxiv.org/abs/2506.09416)
[6](https://arxiv.org/abs/2502.04475)
[7](https://ieeexplore.ieee.org/document/10700806/)
[8](https://www.nature.com/articles/s41598-025-93954-x)
[9](https://openaccess.cms-conferences.org/publications/book/978-1-964867-75-5/article/978-1-964867-75-5_244)
[10](https://www.semanticscholar.org/paper/f0a3835950ca893547d3688e50da1dc77326b22c)
[11](https://iopscience.iop.org/article/10.1149/MA2025-031244mtgabs)
[12](https://arxiv.org/abs/2106.06819)
[13](http://arxiv.org/pdf/2404.16556.pdf)
[14](https://arxiv.org/html/2310.00224)
[15](https://arxiv.org/html/2407.07249v1)
[16](https://arxiv.org/pdf/2411.06438.pdf)
[17](https://arxiv.org/html/2410.11439v1)
[18](https://arxiv.org/pdf/2210.08933.pdf)
[19](https://arxiv.org/html/2503.06674v1)
[20](https://www.merl.com/publications/docs/TR2025-025.pdf)
[21](https://www.sciencedirect.com/science/article/abs/pii/S0925231225021289)
[22](https://openaccess.thecvf.com/content/ICCV2023W/LIMIT/papers/Park_Augmenting_Features_via_Contrastive_Learning-Based_Generative_Model_for_Long-Tailed_Classification_ICCVW_2023_paper.pdf)
[23](https://www.semanticscholar.org/paper/2c525c0a0e058b0f0d0a351c1fd43fd92929433a)
[24](https://liner.com/ko/review/diffusion-models-already-have-a-semantic-latent-space)
[25](https://www.pm.mh.tum.de/miti/ausschreibungen/conditional-generative-models-for-contrastive-learning-in-medical-image-classification/)
[26](https://openaccess.thecvf.com/content/CVPR2025/papers/Jin_Dual-Interrelated_Diffusion_Model_for_Few-Shot_Anomaly_Image_Generation_CVPR_2025_paper.pdf)
[27](https://arxiv.org/abs/2503.06698)
[28](https://arxiv.org/html/2510.09129v1)
[29](https://arxiv.org/abs/2205.15463)
[30](https://www.semanticscholar.org/paper/c97dfad7a023fbec97a901dae02b73e2e8e0fff1)
[31](https://www.semanticscholar.org/paper/1697897e7b528f851590701b7922fd830e22832a)
[32](https://arxiv.org/abs/2211.00321)
[33](https://arxiv.org/pdf/2311.07693.pdf)
[34](http://arxiv.org/pdf/2407.02681.pdf)
[35](http://arxiv.org/pdf/2306.05023.pdf)
[36](https://arxiv.org/pdf/1912.10702.pdf)
[37](https://arxiv.org/pdf/1911.02469.pdf)
[38](https://www.aclweb.org/anthology/2020.coling-main.216.pdf)
[39](https://arxiv.org/pdf/2103.11349.pdf)
[40](https://arxiv.org/pdf/1804.00891.pdf)
[41](https://aclanthology.org/2024.lrec-main.1250v2.pdf)
[42](https://arxiv.org/abs/2007.03898)
[43](https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf)
[44](https://www.merl.com/publications/docs/TR2022-071.pdf)
[45](https://kozistr.tech/2020-09-07-NVAE/)
[46](https://sander.ai/2025/04/15/latents.html)
[47](https://proceedings.neurips.cc/paper_files/paper/2022/file/7b2e844c52349134268e819a9b56b9e8-Paper-Conference.pdf)
[48](https://dda-on.tistory.com/entry/NVAE-A-Deep-Hierarchical-Variational-Autoencoder-%EB%A6%AC%EB%B7%B0)
[49](https://en.wikipedia.org/wiki/Latent_diffusion_model)
[50](https://arxiv.org/html/2311.07693)
