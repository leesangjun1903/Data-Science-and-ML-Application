# One Step Diffusion via Shortcut Models

이 논문은 하나의 네트워크를 한 번의 end-to-end 학습만으로, “많은 step–적은 step–완전 1 step”까지 모든 샘플링 예산에서 높은 품질의 샘플을 내는 *shortcut model*을 제안하는 것이 핵심입니다.[^1][^2]
핵심 아이디어는 네트워크를 “현재 시간 $t$”뿐 아니라 “원하는 step 크기 $d$”에도 조건부로 만들어, 평범한 flow‑matching ODE의 *미분값* 대신 “한 번에 $d$만큼 점프하는 shortcut 벡터”를 학습하고, 이를 자기‑일관성(self‑consistency) 제약으로 훈련하는 것입니다.[^2][^1]

***

## 1. 핵심 주장과 주요 기여

- **핵심 주장**
    - 전통적인 diffusion/flow‑matching 모델은 다수의 denoising step이 필요해 느리고, 단순히 step 수를 줄이면 평균 방향으로 쏠려 한‑두 step에서는 멀티모달 분포를 재현할 수 없다는 문제를 가진다.[^1][^2]
    - *Shortcut model*은 단일 네트워크를 한 번 학습한 뒤, step 크기 $d$를 조절하는 것만으로 128 step, 4 step, 1 step 등 다양한 예산에서 안정적으로 고품질 샘플을 낼 수 있음을 보인다.[^2][^1]
- **주요 기여**[^1][^2]
    - (1) **Step‑size 조건부 shortcut 모델**: $(x_t, t, d)$를 입력받아 “ $d$만큼 앞으로 이동하는 방향” $s(x_t,t,d)$를 직접 예측하는 네트워크와, 이를 훈련시키는 self‑consistency 기반 손실을 제안.
    - (2) **단일 end‑to‑end 학습**: 기존 distillation/consistency 모델이 요구하는 *teacher 모델 + 다단계 스케줄* 없이, 한 번의 학습만으로 one‑step 및 multi‑step 생성을 모두 지원.
    - (3) **경쟁력 있는 성능**: CelebA‑HQ‑256, ImageNet‑256에서 동일한 DiT‑B 아키텍처 기준, 1‑step/4‑step에서 기존 ReFlow, consistency training 등보다 낮은 FID를 달성하면서, 128‑step에서는 flow‑matching과 비슷하거나 약간 더 좋은 FID를 보임.[^1]
    - (4) **범용성**: 이미지 생성 외에도 로봇 조작 정책(diffusion policy)을 shortcut policy로 치환해, 100 step diffusion과 유사한 성능을 1 step으로 달성함으로써 제안 방식의 일반화 가능성을 보여줌.[^1]

***

## 2. 해결하고자 하는 문제

### 2.1 Few‑step / One‑step diffusion의 근본 문제

논문은 flow‑matching 기반 diffusion 모델이 *연속시간 ODE* 수준에서는 이론적으로 잡음 분포를 데이터 분포로 정확히 보낼 수 있지만, 이를 유한 개 step으로 근사하는 순간 문제가 생긴다고 지적합니다.[^2][^1]

- 데이터 샘플 $x_1\sim D$, 잡음 $x_0\sim \mathcal N(0,I)$를 선형 보간해

$$
x_t = (1-t)x_0 + t x_1,\quad v_t = x_1 - x_0
$$

로 정의하고, 주어진 $x_t$에서 가능한 모든 $(x_0,x_1)$쌍에 대한 평균 속도 $\bar v_t = \mathbb E[v_t \mid x_t]$를 근사하는 신경망 $\bar v_\theta(x_t,t)$를 학습하는 것이 flow‑matching입니다.[^2][^1]
- 이때 학습 손실은

$$
L_F(\theta) = \mathbb E_{x_0,x_1,t}\left[\left\lVert \bar v_\theta(x_t,t) - (x_1 - x_0)\right\rVert^2\right]
$$

로 정의되며, 이 손실을 충분히 최적화하면 연속 시간에서는 올바른 ODE를 얻게 됩니다.[^1]

하지만 유한한 step 크기 $\Delta t$로 ODE를 적분하면, $\bar v_\theta$는 “다수의 모드를 평균낸 방향”을 예측하기 때문에 큰 step을 한 번에 적용하면 데이터 분포의 평균 쪽으로 급격히 끌려가며 모드 붕괴, 흐릿한 이미지가 발생합니다. 특히 $t=0$에서는 모델 입력이 pure noise이고 $(x_0,x_1)$가 완전 랜덤 페어이므로, $\bar v_\theta(x_0,0)$는 사실상 데이터셋 평균을 가리키게 되어, 1‑step 생성은 근본적으로 실패할 수밖에 없습니다.[^1]

### 2.2 기존 가속화 방법의 한계

- **Two‑stage distillation (ReFlow, Progressive distillation 등)**:
사전 학습된 diffusion/flow‑matching *teacher*를 완전 시뮬레이션하여 synthetic pair $(x_0,x_1)$나 $(x_t,x_{t+d})$를 만든 뒤, 별도의 *student* one‑step 또는 few‑step 모델을 distill 합니다.[^3][^2][^1]
    - 장점: 강력한 teacher의 분포를 그대로 물려받을 수 있음.
    - 한계: teacher 학습 + distillation 등 여러 phase가 필요하고, synthetic 데이터 생성에 대규모 계산이 들어가며, 언제 distillation을 시작·종료할지 스케줄 설계가 필요합니다.[^3][^1]
- **Consistency models** (Song et al., 2023 등):
직접 $x_t \mapsto x_0$ 또는 $x_1$를 1 step으로 예측하되, 서로 다른 시간 $t, t+d$에서의 출력이 일관되도록 하는 *consistency loss*를 이용합니다.[^4][^5][^6]
    - 장점: one‑step 생성을 설계 수준에서 지원, distillation 또는 data‑only training 모두 가능.
    - 한계: 매우 많은 부트스트랩 step과 복잡한 스케줄, 종종 LPIPS 등 지각 거리 기반 손실, teacher EMA 등 트릭이 필요하며, 훈련 불안정성이 잘 알려져 있습니다.[^7][^8][^1]

이 논문이 겨냥하는 문제는 “teacher나 복잡한 스케줄 없이, 한 번의 end‑to‑end 학습으로 one‑step과 many‑step을 동시에 만족하는 간단한 레시피”를 만드는 것입니다.[^2][^1]

***

## 3. 제안 방법: Shortcut Models (수식 포함)

### 3.1 Shortcut의 정의

기존 flow‑matching은 “순간 속도”를 학습하는 반면, shortcut model은 “step 크기 $d$만큼 앞으로 점프하는 방향”을 직접 예측합니다.[^2][^1]

- shortcut 벡터 $s(x_t,t,d)$를 다음과 같이 정의합니다.

$$
x'_{t+d} = x_t + s(x_t,t,d)\, d
$$

여기서 $x'_{t+d}$는 “정확한 denoising ODE를 $d$만큼 적분한 결과”에 해당하는 타겟 포인트입니다.[^1]
- $d\to 0$일 때는

$$
s(x_t,t,0) \approx \bar v_\theta(x_t,t)
$$

가 되어 flow‑matching의 순간 속도와 일치하도록 설계합니다.[^1]

즉, shortcut 모델은 *일반화된 flow field*를 학습하는 셈이며, 작은 $d$에서는 표준 ODE, 큰 $d$에서는 “큰 점프를 고려한 수정된 방향”을 예측합니다.[^2][^1]

### 3.2 Self‑consistency 제약

핵심 아이디어는 “두 번의 작은 shortcut을 합친 효과가 한 번의 큰 shortcut과 같아야 한다”는 자기‑일관성입니다.[^1]

- 이 논문은 다음 관계를 이용합니다.

```math
s(x_t,t,2d)
=
\frac12\,s(x_t,t,d)
+
\frac12\,s(x'_{t+d},t+d,d)
```

여기서 $x'_{t+d} = x_t + s(x_t,t,d)\,d$입니다.[^1]

직관적으로, $d$ 크기의 shortcut을 두 번 연속 적용한 경로와, $2d$ 크기의 shortcut을 한 번 적용한 경로가 같아야 한다는 제약입니다. 이 self‑consistency를 이용해 큰 step 크기에 대한 타겟을 *ODE 완전 시뮬레이션 없이* 자기‑지도 방식으로 생성할 수 있습니다.[^2][^1]

### 3.3 Shortcut 손실 함수

논문에서 제안하는 최종 손실은 “ $d=0$에서의 flow‑matching 손실”과 “ $d > 0$에서의 self‑consistency 손실”을 합친 형태입니다.[^1]

- **Flow‑matching 부분 ($d=0$)**

```math
L_{\text{flow}}(\theta)
=
\mathbb E_{x_0,x_1,t}\left[
  \left\lVert s_\theta(x_t,t,0) - (x_1 - x_0)\right\rVert^2
\right]
```

이는 shortcut 모델이 작은 step에서 기존 flow‑matching ODE를 정확히 재현하도록 ground 해 줍니다.[^1]

- **Self‑consistency 부분 ( $d>0$ )**
임의의 $d>0$에 대해, 먼저 작은 step $d$ 두 개를 이용해 타겟을 구성합니다.

$$
\begin{aligned}
s_1 &= s_\theta(x_t,t,d), \\
x'_{t+d} &= x_t + s_1\, d, \\
s_2 &= s_\theta(x'_{t+d}, t+d, d), \\
s_{\text{target}} &= \tfrac12(s_1 + s_2).
\end{aligned}
$$

그 다음, 같은 시작점 $x_t$에서 step 크기를 $2d$로 입력해

```math
L_{\text{sc}}(\theta)
=
\mathbb E_{x_t,t,d}\left[
  \left\lVert s_\theta(x_t,t,2d) - s_{\text{target}}\right\rVert^2
\right]
```

를 최소화합니다.[^1]
- **최종 손실**

```math
L_S(\theta)
=
L_{\text{flow}}(\theta) + L_{\text{sc}}(\theta)
```

이 두 항은 하나의 네트워크 $s_\theta$를 사용해, 단일 training run에서 공동으로 최적화됩니다.[^2][^1]

실제로는 배치의 일부 비율(예: 75%)은 $d=0$ flow‑matching으로, 나머지는 $d>0$ self‑consistency로 학습하여, shortcut 모델의 추가 계산 비용을 *기본 diffusion 대비 약 16% 수준*으로 제한합니다.[^1]

### 3.4 Sampling 절차

학습된 shortcut 모델을 사용한 sampling은 알고리즘 2로 정리됩니다.[^1]

1. 초기 잡음 $x\sim\mathcal N(0,I)$를 샘플.
2. 총 step 수 $M$을 정하고 (예: 128), 각 step에서

$$
x \leftarrow x + s_\theta(x,t,d)\,d,\quad t\leftarrow t+d
$$

를 반복해 최종 $x_T$를 얻습니다.[^1]

여기서 $d = 1/M$이고, $M=1$인 경우 바로 one‑step 생성이 됩니다. 같은 네트워크가 $M$을 다르게 설정하는 것만으로 128‑step, 4‑step, 1‑step 샘플링을 모두 지원한다는 점이 중요한 특징입니다.[^2][^1]

***

## 4. 모델 구조

- **아키텍처**
    - 기본적으로 *DiT‑B / DiT‑XL* 계열의 diffusion transformer를 사용하며, latent 공간은 Stable Diffusion VAE(sd‑vae‑ft‑mse)의 32×32×4 latent를 입력으로 사용합니다.[^1]
    - 클래스 조건부(ImageNet‑256)에서는 class embedding을 추가하고, timestep $t$ 뿐 아니라 step‑size $d$도 임베딩하여 transformer에 주입합니다.[^1]
- **학습 디테일**
    - Optimizer: AdamW, learning rate $10^{-4}$, weight decay 0.1, EMA decay 0.999.[^1]
    - 가장 작은 step 크기를 $1/128$로 두고, 이진 재귀 구조를 사용해 $d\in\{1/128, 1/64, \dots, 1\}$의 8개 길이에 대해 shortcut을 학습합니다.[^1]
    - 배치 내 75%는 $d=0$ flow‑matching, 25%는 self‑consistency 타겟에 사용하여, self‑consistency에서 비롯되는 잡음과 불안정을 제한합니다.[^1]

이러한 설계는 기존 diffusion과 동일한 backbone을 유지하면서, “ $t,d$” 조건부 head와 self‑consistency loss만 추가하는 상대적으로 간단한 변경으로 구현됩니다.[^2][^1]

***

## 5. 성능 향상 및 한계

### 5.1 이미지 생성 성능

논문에서는 CelebA‑HQ‑256(비조건부)와 ImageNet‑256(클래스 조건부)에서, 동일한 DiT‑B 아키텍처와 유사한 compute로 여러 방법을 비교합니다.[^1]

- **128‑step (many‑step)**
    - Flow‑matching(DiT‑B): CelebA FID 7.3, ImageNet FID 17.3.[^1]
    - Shortcut(DiT‑B): CelebA FID 6.9, ImageNet FID 15.5로, teacher 없이도 flow‑matching보다 약간 더 좋은 FID를 달성합니다.[^1]
- **4‑step / 1‑step (few/one‑step)**
    - Flow‑matching: 4‑step, 1‑step에서 FID가 크게 악화(예: ImageNet 1‑step FID 324.8 수준)되어 실용적이지 않습니다.[^1]
    - ReFlow(two‑stage distillation): 1‑step FID 23.2(CelebA), 44.8(ImageNet).[^1]
    - Consistency training(end‑to‑end): 1‑step FID 33.2(CelebA), 69.7(ImageNet).[^1]
    - Shortcut models: 1‑step FID 20.5(CelebA), 40.3(ImageNet)으로, teacher가 없는 end‑to‑end 접근 중 가장 우수하고, 일부 distillation 계열(ReFlow, consistency distillation 등)과도 비슷하거나 더 나은 성능을 보입니다.[^1]

또한 같은 noise에서 128‑step과 1‑step을 비교하면, flow‑matching은 모드 붕괴와 블러가 심하지만, shortcut은 전역적인 구조·스타일이 유사하고 고주파 디테일만 다소 열화되는 경향을 보여, “one‑step 결과를 rough draft로 쓰고, 필요하면 multi‑step으로 refinement” 하는 사용 패턴이 가능함을 시각적으로 보여줍니다.[^1]

### 5.2 스케일 업에 따른 성능과 일반화

모델 파라미터 수를 늘리면 one‑step 성능이 계속 향상되는지가, self‑bootstrapping 방식에서 특히 중요한 질문입니다.[^1]

- DiT‑XL 기반 shortcut 모델은 ImageNet‑256에서 1‑step FID 10.6, 128‑step FID 3.8을 달성해, DiT‑B 기반 결과보다 상당히 나은 성능을 보여줍니다.[^1]
- 이는 부트스트랩 방식임에도 불구하고, RL Q‑learning에서 관찰되는 “rank collapse”와 달리, 모델 크기를 키울수록 일반화 성능(여기서는 FID)이 계속 개선된다는 것을 시사합니다.[^2][^1]

또한 noise interpolation 실험에서, 두 noise $x_0^{(a)}, x_0^{(b)}$ 사이를

$$
x_0^{(n)} = n x_0^{(a)} + \sqrt{1-n^2}\,x_0^{(b)}
$$

로 보간하고 one‑step shortcut을 적용했을 때, 생성 이미지가 부드럽게 변화하며 중간 샘플도 모두 자연스러운 이미지를 형성함을 보여줍니다. 이는 shortcut 모델이 latent noise 공간에서 의미 있는, *연속적이며 일반화 가능한 표현*을 학습했음을 간접적으로 보여줍니다.[^1]

### 5.3 로봇 제어에서의 성능

Diffusion policy(100 step denoising)를 shortcut policy(1 step)로 치환해, Push‑T와 Transport 작업에서 성공률을 비교합니다.[^1]

- Push‑T: Diffusion policy(100 step) 성공률 0.95, shortcut policy(1 step) 0.87.[^1]
- Transport: Diffusion policy 1.00, shortcut policy 0.80.[^1]

반면, 동일한 환경에서 one‑step diffusion policy는 완전히 실패(성공률 0 근처)하는 것으로 보고되어, “shortcut 구조 자체가 멀티모달 행동 분포를 한 shot에 포착하는 데 효과적”임을 시사합니다.[^1]

### 5.4 한계

논문이 명시하는 주요 한계는 다음과 같습니다.[^2][^1]

- **Noise–data mapping의 비가역성**
    - shortcut 모델은 주어진 noise에서 *데이터 분포의 기대값*을 따라가는 deterministic mapping을 학습하며, GAN이나 VAE처럼 mapping 자체를 유연하게 재설계하는 구조적 여지가 적습니다.[^1]
    - 이는 학습 용이성 면에서 제약이 될 수 있고, 학습된 mapping이 특정 데이터셋 통계에 강하게 종속될 위험이 있습니다.
- **Many‑step vs one‑step 품질 격차**
    - 실험 결과, many‑step과 one‑step 사이에 여전히 눈에 띄는 품질 격차가 남아 있으며, 이 격차를 완전히 해소하지 못했습니다.[^1]
    - 특히 최고 수준의 one‑step consistency/EM‑distillation 계열 SOTA와 비교하면, ImageNet‑64/128 등에서 FID 상 열세입니다.[^8][^3]
- **CFG 및 하이퍼파라미터 제약**
    - classifier‑free guidance는 작은 step(d=0)에서만 안정적으로 작동하며, 큰 step에서는 선형 근사가 깨져 사용하지 않는 것으로 보고됩니다.[^1]
    - CFG scale을 학습 중에 고정해야 한다는 점도 실용상 유연성을 떨어뜨립니다.[^1]

***

## 6. 일반화 성능 향상 관점에서의 분석

질문에서 특히 요청하신 “모델의 일반화 성능 향상 가능성”을 shortcut 프레임워크 관점에서 정리하면 다음과 같습니다.

1. **시간 축 self‑consistency가 주는 정규화 효과**
    - shortcut 손실은 임의의 step 길이 $d$에 대해 “두 번의 작은 step”과 “한 번의 큰 step”이 일치하도록 강한 구조적 제약을 가합니다.[^1]
    - 이는 다양한 sampling budget(1, 4, 128 step 등)에 대해 *동일한 생성 분포를 유지*하도록 강제하는 것으로, 학습 데이터에 대한 단순 적합을 넘어 “시간 해상도에 대한 불변성(invariance)”이라는 추가적인 일반화 조건을 부여한다고 볼 수 있습니다.[^2][^1]
2. **단일 모델로 여러 budget을 커버하는 multi‑task 학습 효과**
    - 같은 파라미터 $\theta$가 $d\in\{0,1/128,\dots,1\}$ 전 범위에 대해 성능을 내야 하므로, 모델은 특정 step 크기에 overfit 되기보다는 다양한 step에서 안정적인 동작을 학습하게 됩니다.[^1]
    - 이는 한 종류의 task에만 특화된 one‑step 전용 모델보다 “sampling 시간 분해능 변화”라는 hidden domain shift에 더 잘 일반화할 가능성이 큽니다.
3. **스케일 업에서도 collapse 없이 성능이 증가**
    - DiT‑XL로 스케일 업했을 때도 one‑step FID가 계속 감소하고, latent interpolation이 부드럽게 유지된다는 결과는, self‑bootstrapping에도 불구하고 표현력 증가가 온전히 품질 향상으로 이어짐을 보여줍니다.[^1]
    - RL에서 보고된 것처럼 Q‑function 부트스트랩이 고차원에서 rank collapse를 일으키는 현상과 대조적이며, shortcut self‑consistency가 “안정적인 프리‑트레이닝 구조”로 작동할 수 있음을 시사합니다.[^1]
4. **도메인 일반화: 이미지에서 로봇 제어로의 전이**
    - 같은 수식 구조(ODE 기반 flow‑matching + shortcut self‑consistency)를 로봇 조작 정책에 그대로 적용했을 때도, one‑step에서 diffusion policy와 유사한 성능을 낸다는 점은, 이 프레임워크가 *이미지에 특화된 기법이 아니라 일반적인 확률 경로(shortcut) 학습 방식*임을 보여줍니다.[^1]
    - 이는 향후 시계열·음성·비디오·3D 생성 등으로 확장할 때도, 복잡한 teacher 없이 동일한 self‑consistency 구조를 사용할 수 있음을 시사합니다.

반면, 일반화 측면에서 아직 검증되지 않은 부분도 있습니다.

- 학습 데이터 분포에서 벗어난 out‑of‑distribution 조건(예: 도메인 shift, extreme guidance scale)에 대한 robustness는 별도로 평가되지 않았습니다.[^1]
- self‑bootstrapping이 초기 학습 단계에서 잘못된 shortcut을 강화할 위험이 있으므로, weight decay와 EMA, d 분포 설계가 일반화 성능에 미치는 영향을 추가 분석할 필요가 있습니다.[^1]

***

## 7. 2020년 이후 관련 최신 연구 비교

여기서는 “one‑step / few‑step diffusion”이라는 큰 맥락에서, 2020년 이후 대표적인 연구와 본 논문의 위치를 비교합니다.

### 7.1 주요 계열 요약

1. **Consistency Models (Song et al., 2023; Song \& Dhariwal, 2024 등)**[^5][^6][^4][^8]
    - 아이디어: PF‑ODE/VP‑SDE 경로를 따라 $x_t\mapsto x_0$를 직접 한 step으로 예측하면서, 서로 다른 시간 $(t,t')$에서의 예측이 “consistency”를 만족하도록 학습.
    - 장점: 이론적으로 1‑step/2‑step에서 매우 빠른 샘플링, zero‑shot 편집(inpainting 등) 지원.
    - 한계: 초기 버전은 학습 불안정·스케줄 의존성이 크고, distillation+LPIPS와 같은 복잡한 트릭이 필요.
    - 개선 연구: Improved Techniques for Training Consistency Models(2024)은 EMA teacher 제거, Pseudo‑Huber loss, lognormal noise schedule 등으로 CIFAR‑10에서 1‑step FID 2.51, ImageNet‑64에서 3.25를 달성.[^8]
2. **Distillation 기반 one‑step 모델 (ReFlow, EMD 등)**[^9][^10][^11][^3][^1]
    - ReFlow(Liu et al., 2022): teacher flow‑matching 모델을 완전히 시뮬레이션해 synthetic pair를 만들고 student를 학습.
    - EM Distillation for One‑step Diffusion Models (Xie et al., NeurIPS 2024): maximum likelihood/EM 관점에서 teacher diffusion을 one‑step generator로 distill, reparameterized sampling과 noise cancellation으로 안정을 개선, ImageNet‑64/128에서 기존 one‑step 방법보다 낮은 FID 달성.[^9][^3]
    - 장점: 강력한 teacher 품질을 유지하면서 one‑step으로 압축 가능.
    - 한계: teacher 학습과 distillation의 다단계 파이프라인, teacher에 대한 접근과 full/sparse sampling이 필요.
3. **Multistep Consistency Models (2024)**[^12]
    - consistency model과 TRACT(시간 distillation)를 결합해 “1‑step consistency–∞‑step diffusion” 사이를 연속적으로 보간하는 프레임워크 제안.[^12]
    - sampling step 수를 늘릴수록 품질을 개선하면서도, 일관된 objective 하에서 학습할 수 있도록 설계.
4. **Teacher‑free one‑step 학습 (DiffRatio, 2025 등)**[^13]
    - “Training One-Step Diffusion Models Without Teacher Supervision”는 더 이상 teacher diffusion이 필요 없는 density‑ratio 기반 one‑step 학습(DiffRatio)을 제안.[^13]
    - 경량 density‑ratio 네트워크만 추가해, 다수의 teacher‑supervised distillation 방법보다 경쟁력 있는 one‑step FID를 보고합니다.[^13]
5. **Shortcut 계열의 확장 연구 (HOMO, 설계 프레임워크 등)**[^14][^15]
    - High‑Order Matching for One-Step Shortcut Diffusion Models(HOMO, 2025): shortcut이 본질적으로 1차 근사에 의존해 고곡률 영역에서 어려움을 겪는다는 문제를 지적하고, 고차 정보까지 반영한 high‑order matching을 제안.[^14]
    - “On the Design of One-step Diffusion via Shortcutting Flow Paths”(2026): shortcut 계열 모델들을 위한 공통 설계 프레임워크를 제시하고, 이론적 정당화 및 구성 요소별 설계 공간을 체계화해 ImageNet‑256 one‑step FID 2.85까지 끌어올림.[^15]

### 7.2 Shortcut Models의 위치와 차별점

- **Teacher 필요 여부 및 학습 단계**
    - Consistency distillation, EMD, ReFlow 등은 모두 *teacher diffusion/flow*가 필요하고, teacher 학습 + distillation의 2‑단계 이상 파이프라인을 구성합니다.[^3][^1]
    - Shortcut 모델은 teacher 없이 data‑only로, flow‑matching + self‑consistency를 단일 objective로 한 번에 학습합니다.[^2][^1]
- **Loss 구조와 제약 방식**
    - Consistency models: 서로 다른 시간에서의 *empirical* 샘플 $(x_t,x_{t+d})$에 대해 output의 consistency를 강제하며, 모든 step에 대해 bootstrapping을 수행해야 합니다.[^4][^1]
    - Shortcut models: “ODE 상의 예측 경로”를 이용해 self‑consistency를 구성하므로, log $_2 T$ 단계의 부트스트랩만 필요하고, base case로는 표준 flow‑matching을 사용합니다. 이는 계산 효율과 구현 단순성 면에서 장점입니다.[^1]
- **성능–복잡도 트레이드오프**
    - 최신 consistency/EMD 계열은 ImageNet‑64/128 기준 최고의 one‑step FID를 보여주지만, teacher 훈련, 복잡한 스케줄 및 loss가 필요합니다.[^8][^3]
    - Shortcut models는 best‑in‑class FID는 아니지만, “backbone 공유 + objective만 교체” 수준의 간단한 변경으로, teacher‑free one‑step/128‑step을 동시에 지원합니다.[^1]
    - HOMO 및 Shortcut 설계 프레임워크 연구들은 이 shortcut 방향을 이론적으로 정교화하면서, 품질을 최상위 수준까지 끌어올리고 있어, shortcut 계열이 “실용성+성능”의 균형을 점점 맞춰가는 추세입니다.[^15][^14]

***

## 8. 앞으로의 연구 영향과 고려할 점

### 8.1 연구/응용 측면 영향

1. **단일 모델로 전 budget을 커버하는 표준 패러다임 가능성**
    - 지금까지 one‑step 가속화는 대부분 “사전 학습 diffusion + 별도 student” 구조였는데, shortcut은 “단일 backbone + step‑size conditioning + self‑consistency”만으로 동일 효과를 얻을 수 있음을 보여줍니다.[^2][^1]
    - 이는 실무에서 “한 모델을 학습해 놓고, 서버·모바일·로봇 등 환경 제약에 따라 step 수만 조정해 사용하는” 구조를 가능하게 해, 시스템 설계 및 배포를 크게 단순화할 수 있습니다.
2. **시간‑축 self‑distillation이라는 새로운 정규화 관점**
    - shortcut objective는 본질적으로 “시간 방향으로의 self‑distillation”이며, 이는 teacher‑student distillation과 달리 추가 네트워크 없이 동일 파라미터 내에서 일어나는 자기 지도입니다.[^1]
    - 이 아이디어는 diffusion 외에도, temporal dynamics가 존재하는 다른 모델(예: 시계열 예측, RL policy 등)에 일반화 가능한 새로운 정규화/훈련 트릭으로 확장될 여지가 큽니다.
3. **고차/복잡 데이터에 대한 설계 프레임워크의 기초**
    - 이후 연구들(HOMO, shortcut 설계 프레임워크 등)은 이 논문의 아이디어를 일반화해, 높은 곡률 영역에서의 안정성, 고차 정보 사용, 구성 요소별 설계 공간 탐색 등으로 이어지고 있습니다.[^14][^15]
    - 이는 “few/one‑step diffusion”을 하나의 체계적인 설계 문제로 다룰 수 있게 해, 향후 텍스트‑투‑이미지, 비디오, 3D 생성 등 광범위한 응용에서 shortcut 계열이 표준 옵션이 될 가능성을 열어 줍니다.[^16][^2]

### 8.2 앞으로 연구 시 고려할 점 (연구자가 봐야 할 포인트)

연구자로서 이 논문을 확장·활용할 때 특히 고려해야 할 지점은 다음과 같습니다.

1. **Self‑bootstrapping의 안정성과 오류 전파 분석**
    - self‑consistency가 strong prior를 제공하는 동시에, 초기 단계에서 잘못 학습된 shortcut이 반복적으로 강화될 위험도 존재합니다.[^1]
    - weight decay, EMA, $d$ 샘플링 분포, bootstrap 깊이(log $_2 T$ ) 등이 안정성·일반화에 미치는 영향에 대한 이론적·실험적 분석이 필요합니다.
2. **Many‑step과 One‑step 품질 격차를 줄이는 방법**
    - 현재는 many‑step에서의 teacher 수준 품질과 one‑step 품질 사이에 non‑trivial한 gap이 남아 있습니다.[^1]
    - ReFlow/EMD처럼 distribution matching, KL 기반 최대우도, high‑order matching(HOMO) 등을 shortcut objective에 결합해, one‑step 품질이 오히려 many‑step을 향상시키는 “양방향 개선” 구조를 설계할 수 있을지 연구 과제로 남아 있습니다.[^14][^3]
3. **Guidance 및 조건부 생성의 안정적 통합**
    - CFG를 작은 step(d=0)에서만 사용하는 현재 설계는, 강한 guidance가 필요한 고해상도·텍스트 조건부 생성에서 제약이 됩니다.[^1]
    - guidance vector 자체를 shortcut 구조에 통합하거나, teacher‑free guidance(예: classifier‑free 외 다른 scoring)를 shortcut과 함께 설계하는 방향이 중요합니다.
4. **도메인/분포 shift에 대한 일반화 평가**
    - 현 논문은 in‑distribution FID와 특정 로봇 작업에 집중하고 있어, 도메인 shift, 공격적 noise/guidance 조건, 장기 시계열 등에서의 일반화는 미평가 상태입니다.[^1]
    - out‑of‑distribution 설정에서 shortcut self‑consistency가 오히려 regularizer로 작용하는지, 아니면 분포 외 영역에서 “시간‑축 일관된 오차”를 만들어낼 위험이 있는지 정교한 평가가 필요합니다.
5. **이후 shortcut 관련 연구와 연계**
    - HOMO, DiffRatio, shortcut 설계 프레임워크 등 최신 연구는 이미 이 논문을 기반으로 성능 및 안정성을 더욱 개선하고 있습니다.[^15][^13][^14]
    - 향후 연구에서는 shortcut의 간결한 구현을 유지하되, 이러한 후속 방법에서 제안하는 고차 정보, density‑ratio 추정, 보다 이론적인 설계 지침 등을 적절히 융합하는 것이 핵심 과제가 될 것입니다.

***

## 참고 문헌 및 자료 (모두 오픈 액세스)

- Kevin Frans, Danijar Hafner, Sergey Levine, Pieter Abbeel, “One Step Diffusion via Shortcut Models,” *arXiv:2410.12557*.[^17][^1]
- Xingchao Liu et al., “Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow,” *arXiv:2209.03003*.[^1]
- Yang Song, Prafulla Dhariwal, Mark Chen, Ilya Sutskever, “Consistency Models,” *arXiv:2303.01469*, PMLR 2023.[^6][^5][^4]
- Yang Song, Prafulla Dhariwal, “Improved Techniques for Training Consistency Models,” ICLR 2024 oral.[^8]
- Sirui Xie et al., “EM Distillation for One-step Diffusion Models,” *arXiv:2405.16852*, NeurIPS 2024.[^10][^11][^9][^3]
- “Multistep Consistency Models,” *arXiv:2403.06807*.[^12]
- “Training One-Step Diffusion Models Without Teacher Supervision (DiffRatio),” *arXiv:2502.08005*.[^13]
- “High-Order Matching for One-Step Shortcut Diffusion Models (HOMO),” *arXiv:2502.00688*.[^14]
- “On the Design of One-step Diffusion via Shortcutting Flow Paths,” 2026, HuggingFace Papers (open access link 제공).[^15]
- Kevin Frans et al., GitHub repository “shortcut-models” (공개 코드 및 체크포인트).[^16][^2][^1]
<span style="display:none">[^18][^19][^20][^21][^22][^23][^24]</span>

<div align="center">⁂</div>

[^1]: 2410.12557v3.pdf

[^2]: https://arxiv.org/html/2410.12557v2

[^3]: https://www.arxiv.org/abs/2405.16852

[^4]: https://arxiv.org/abs/2303.01469

[^5]: https://dl.acm.org/doi/10.5555/3618408.3619743

[^6]: https://proceedings.mlr.press/v202/song23a/song23a.pdf

[^7]: https://arxiv.org/html/2410.11081v2

[^8]: https://iclr.cc/virtual/2024/oral/19754

[^9]: https://arxiv.org/abs/2405.16852

[^10]: https://arxiv.org/html/2405.16852v1

[^11]: https://ui.adsabs.harvard.edu/abs/2024arXiv240516852X/abstract

[^12]: https://arxiv.org/html/2403.06807v3

[^13]: https://arxiv.org/html/2502.08005v4

[^14]: https://arxiv.org/html/2502.00688v1

[^15]: https://huggingface.co/papers/2512.11831

[^16]: https://linnk.ai/vi/insight/machine-learning/one-step-image-generation-with-shortcut-models-achieving-high-quality-with-reduced-sampling-steps-TTHu3LIF/

[^17]: https://arxiv.org/abs/2410.12557

[^18]: https://www.semanticscholar.org/paper/EM-Distillation-for-One-step-Diffusion-Models-Xie-Xiao/00cda1c832fc7c89b2785440675b10122137869d

[^19]: https://arxiv.org/html/2410.18958v1

[^20]: https://arxiv.org/html/2505.12674v1

[^21]: https://www.semanticscholar.org/paper/One-Step-Diffusion-via-Shortcut-Models-Frans-Hafner/e086a9009a857fbe26dd83a34854a4c317f575b8

[^22]: https://www.semanticscholar.org/paper/Consistency-Models-Song-Dhariwal/ac974291d7e3a152067382675524f3e3c2ded11b

[^23]: http://arxiv.org/list/stat/2024-05?skip=1020\&show=50

[^24]: https://fugumt.com/fugumt/paper_check/2410.12557v1_enmode

