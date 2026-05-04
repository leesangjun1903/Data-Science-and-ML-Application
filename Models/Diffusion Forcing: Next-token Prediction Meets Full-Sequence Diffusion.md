
# Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion

> **저자:** Boyuan Chen, Diego Marti Monso, Yilun Du, Max Simchowitz, Russ Tedrake, Vincent Sitzmann
> **발표:** NeurIPS 2024
> **arXiv:** [2407.01392](https://arxiv.org/abs/2407.01392) | **프로젝트 페이지:** [boyuan.space/diffusion-forcing](https://www.boyuan.space/diffusion-forcing/)

---

## 1. 핵심 주장 및 주요 기여 요약

이 논문은 **Diffusion Forcing(DF)**이라는 새로운 학습 패러다임을 제안합니다. 이 패러다임에서 Diffusion 모델은 **토큰마다 독립적인 노이즈 레벨**을 갖는 토큰 집합을 denoising하도록 학습됩니다.

이 접근법은 **next-token prediction 모델의 강점**(가변 길이 생성)과 **full-sequence diffusion 모델의 강점**(원하는 궤적으로 샘플링을 유도하는 능력)을 결합하는 것으로 나타났습니다.

### 핵심 기여 4가지


1. **Diffusion Forcing 제안**: next-token prediction 모델의 유연성과 full-sequence diffusion 모델의 long-horizon guidance 능력을 동시에 갖는 새로운 확률적 시퀀스 모델 제시.
2. **새로운 의사결정 프레임워크**: Diffusion Forcing의 고유한 능력을 활용하여, 이를 동시에 **정책(policy)** 및 **플래너(planner)**로 사용할 수 있는 프레임워크 도입.
3. **이론적 증명**: 제안된 학습 목적함수 최적화가 학습 시 관찰된 **모든 서브시퀀스의 결합 분포에 대한 하한(ELBO)**을 최대화함을 형식적으로 증명.
4. **다양한 도메인 검증**: 비디오 생성, 모델 기반 계획, 시각적 모방 학습, 시계열 예측 등 다양한 도메인에서 실험적으로 평가.


---

## 2. 해결하고자 하는 문제, 제안 방법(수식 포함), 모델 구조, 성능 향상 및 한계

### 2.1 해결하고자 하는 문제

기존 시퀀스 생성 모델은 크게 두 가지 패러다임으로 나뉩니다.


- **Next-token Prediction (Teacher Forcing)**: 언어 모델에 흔히 쓰이며, 과거 ground-truth 시퀀스로부터 다음 단일 토큰을 예측하도록 학습됩니다.
- **Full-Sequence Diffusion**: 비디오 생성 등에 쓰이며, non-causal 아키텍처로 시퀀스 전체 프레임을 동일한 노이즈 레벨로 동시에 denoising합니다.


이 두 패러다임은 각자 치명적인 한계를 가집니다.

현재 next-token prediction 모델은 Teacher Forcing으로 학습되는데, 이는 (1) 샘플링 과정에서 특정 목적에 맞게 가이드할 메커니즘이 없고, (2) 연속 데이터에서 쉽게 불안정해지는 두 가지 한계를 낳습니다.

Full-sequence diffusion은 non-causal, unmasked 아키텍처로 파라미터화되기 때문에, 가변 길이 생성이 불가능하고 guidance와 서브시퀀스 생성 가능성도 제한됩니다.

더욱이, 두 세계의 장점을 단순히 결합하려는 시도—즉, next-token prediction 모델을 full-sequence diffusion으로 학습시키는 것—은 생성 품질이 낮아집니다. 왜냐하면 이 방식은 초기 토큰의 작은 불확실성이 이후 토큰의 높은 불확실성을 필연적으로 수반한다는 사실을 모델링하지 못하기 때문입니다.

---

### 2.2 제안하는 방법 (수식 포함)

#### ✦ 핵심 아이디어: Noise as Masking

이 접근법의 핵심 동기는 **노이즈가 부분적 마스킹의 한 형태**라는 관찰에 있습니다. 즉, 노이즈가 0이면 토큰은 마스킹되지 않은 것이고, 완전한 노이즈는 토큰을 완전히 마스킹합니다. 따라서 DF는 모델이 다양하게 노이즈가 가해진 토큰 집합을 "unmask"하도록 강제합니다.

Diffusion Forcing은 각 토큰이 서로 다른 노이즈 레벨을 가질 수 있도록 sequence diffusion을 학습시킵니다. 이를 통합적으로 바라보면, full-sequence diffusion은 모든 프레임을 동일한 노이즈 레벨로 한번에 denoising하고, next-token prediction은 과거 토큰의 노이즈 레벨을 0으로 설정하며 한 번에 하나씩 다음 프레임을 denoising하는 것과 동일합니다.

#### ✦ 포워드 프로세스 (Forward Process)

표준 DDPM 포워드 프로세스에 따라, 각 시간 단계 $t$의 토큰 $x_t$에 대해 **독립적** 노이즈 레벨 $k_t \in \{0, 1, \ldots, K\}$을 부여합니다:

$$
x_t^{k_t} = \sqrt{\bar{\alpha}_{k_t}}\, x_t + \sqrt{1 - \bar{\alpha}_{k_t}}\, \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I)
$$

여기서:
- $x_t$: 시간 $t$에서의 원본 (clean) 토큰
- $x_t^{k_t}$: 노이즈 레벨 $k_t$이 적용된 노이즈 토큰
- $\bar{\alpha}_{k_t}$: 누적 노이즈 스케줄 계수
- $k_t \in \{0, 1, \ldots, K\}$는 각 토큰별로 독립적으로 균일하게 샘플링

핵심은, **각 토큰의 노이즈 레벨 $k_t$가 시간 단계별로 독립적으로 변할 수 있다**는 점입니다.

#### ✦ Causal Diffusion Forcing (CDF) 학습 목적함수

DF를 시퀀스 생성에 구현한 것이 **Causal Diffusion Forcing (CDF)**입니다. CDF에서 미래 토큰은 causal 아키텍처를 통해 과거 토큰에 의존합니다. 모델은 토큰별 독립적 노이즈 레벨로 시퀀스의 모든 토큰을 동시에 denoising하도록 학습됩니다.

학습 손실 함수는 각 시간 단계 $t$에 대한 noise prediction 손실의 합으로 구성됩니다:

$$
\mathcal{L}_{\text{DF}} = \mathbb{E}_{x_{1:T}, k_{1:T} \sim \mathcal{U}(0,K)^T} \left[ \sum_{t=1}^{T} \left\| \epsilon_\theta(z_{t-1},\, x_t^{k_t},\, k_t) - \epsilon_t \right\|^2 \right]
$$

여기서:
- $z_{t-1}$: 과거 토큰 $x_{1:t-1}$을 요약하는 causal hidden state (RNN/Transformer)
- $\epsilon_\theta$: 노이즈 예측 네트워크
- $k_{1:T}$: 각 토큰에 독립적으로 균일 샘플링된 노이즈 레벨 벡터
- $T$: 시퀀스 길이

RNN 가중치 $\theta$는 과거 토큰의 영향을 포착하는 latent $z_t$를 유지하며 evolve하고, 이는 베이즈 필터링에서 "prior 분포" $p_\theta(z_t | z_{t-1})$를 모델링하는 것과 동등합니다.

#### ✦ 이론적 보장: ELBO 하한

논문은 **Theorem 3.1 (비형식적)**을 통해, Diffusion Forcing 학습 절차(Algorithm 1)가 **모든 서브시퀀스 토큰의 기대 로그 가능도에 대한 ELBO의 재가중치(reweighting)**를 최적화함을 증명합니다.

즉, 최적화 대상은 다음과 같은 ELBO의 하한입니다:

$$
\mathcal{L}_{\text{DF}} \geq \mathbb{E}_{s \leq T} \left[ \ln p_\theta\bigl(x_{1:s}^{k_{1:s}}\bigr) \right]
$$

이는 학습 시 관찰된 **모든 길이의 서브시퀀스의 결합 분포**에 대한 하한을 최대화함을 의미합니다.

---

### 2.3 모델 구조

Diffusion Forcing은 **causal sequence 신경망**(예: RNN 또는 masked Transformer)을 학습시켜, 시퀀스의 각 프레임이 서로 다른 노이즈 레벨을 가질 수 있는 유연한 길이의 시퀀스를 denoising하도록 합니다.

구체적으로:

| 구성 요소 | 설명 |
|---|---|
| **Causal Backbone** | RNN 또는 Masked Transformer (인과 구조 보장) |
| **Noise Conditioning** | 각 토큰마다 독립적인 노이즈 레벨 $k_t$ 입력 |
| **Denoising Head** | noise prediction 네트워크 $\epsilon_\theta$ |
| **Hidden State** | $z_t$: 과거 토큰 $x_{1:t}$을 요약하는 latent |

메인 브랜치는 **temporal attention**을 사용한 최신 재구현을 포함하며, paper 브랜치는 원래 논문에서 사용한 RNN 코드를 포함합니다.

샘플링 시, CDF는 다양한 프레임이 각 denoising 단계에서 서로 다른 노이즈 레벨을 가질 수 있는 Gaussian noise 프레임 시퀀스를 점진적으로 denoising합니다.

---

### 2.4 새로운 샘플링 기법

#### ✦ Pyramid Sampling (안정적 자동 회귀 생성)

고차원 연속 시퀀스(예: 비디오)에서 자동 회귀 아키텍처는, 특히 훈련 horizon을 넘어 샘플링할 때 발산하는 것으로 알려져 있습니다. 반면 Diffusion Forcing은 일부 작은 노이즈 레벨 $0 < k \ll K$로 약간 "noisy한 토큰"과 관련된 이전 latent를 업데이트함으로써, 훈련 시퀀스 길이를 초과하는 긴 시퀀스도 안정적으로 rollout할 수 있습니다.

Pyramid Sampling의 노이즈 스케줄은 다음과 같이 나타낼 수 있습니다:

$$
k_t = \max\!\left(0,\; K - \frac{t}{\Delta}\right) \quad \text{(미래 토큰일수록 더 높은 노이즈)}
$$

#### ✦ Monte Carlo Tree Guidance (MCTG)

인과성(causality), 유연한 horizon, 가변 노이즈 스케줄을 시너지적으로 활용하여, CDF는 **Monte Carlo Tree Guidance (MCTG)**라는 새로운 기능을 가능하게 합니다. 이는 non-causal full-sequence diffusion 모델 대비 고보상 생성 샘플링을 극적으로 개선합니다.

MCTG는 의사결정 과정에서 미래 상태의 기대 보상을 다음과 같이 추정합니다:

$$
\hat{R}(x_{1:t}) = \mathbb{E}_{x_{t+1:T} \sim p_\theta(\cdot | x_{1:t})} \left[\sum_{s=t}^{T} r(x_s)\right]
$$

---

### 2.5 성능 향상

이 방법은 다음의 추가적인 능력을 제공합니다: (1) 기준 모델이 발산하는 훈련 horizon을 넘어서는 길이로 비디오와 같은 연속 토큰 시퀀스를 rollout하고, (2) Diffusion Forcing의 가변 horizon 및 causal 아키텍처로부터 독점적으로 이익을 얻는 새로운 샘플링·가이딩 방식을 제공하여, 의사결정 및 계획 태스크에서 현저한 성능 향상을 이끌어냅니다.

실험 도메인별 성능 향상 요약:

| 도메인 | 주요 성과 |
|---|---|
| **비디오 생성** | 훈련 horizon 이상으로 안정적 장기 rollout (기준 모델 발산 대비) |
| **모델 기반 계획** | MCTG를 통해 non-causal 모델 대비 고보상 생성 현저히 향상 |
| **시각적 모방 학습** | policy + planner 통합으로 성능 향상 |
| **시계열 예측** | 가변 길이 예측 및 compositional 일반화 달성 |

---

### 2.6 한계점

논문 및 관련 연구에서 확인된 주요 한계:

1. **계산 비용**: 각 토큰마다 독립적인 노이즈 레벨을 처리하므로, 표준 teacher forcing 대비 학습 및 샘플링에 더 많은 연산이 필요합니다.
2. **이산 데이터 확장의 어려움**: DF는 연속 공간 diffusion 모델을 위해 특별히 개발된 기법으로, 이를 이산 데이터로 확장하는 것은 별도의 Discrete Diffusion Forcing(D2F) 방법이 필요합니다.
3. **학습 도메인 외 일반화**: 향후 연구로는 시계열 생성 모델링 이외의 도메인에 Diffusion Forcing을 적용하거나, 더 큰 데이터셋으로 스케일업하는 것이 필요합니다.
4. **Exposure Bias**: Self Forcing 논문이 지적하듯, 훈련 시 ground-truth context에서 학습된 모델이 추론 시 자체 생성 출력에 조건화된 시퀀스를 생성해야 하는 exposure bias 문제는 완전히 해결되지 않았습니다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 Compositional Generalization (합성적 일반화)

예측을 next-token prediction 모델의 합성으로 파라미터화함으로써, 시스템은 **가변 길이 시퀀스를 유연하게 생성**할 수 있을 뿐만 아니라, **새로운 궤적에 대해 합성적으로 일반화**할 수 있습니다.

구체적으로, 이는 다음을 의미합니다:

$$
p_\theta(x_{1:T}) = \prod_{t=1}^{T} p_\theta(x_t \mid z_{t-1}) \quad \text{(각 조건부 확률은 새로운 horizon에도 적용 가능)}
$$

훈련 시 특정 길이 $T_{train}$으로 학습하더라도, $T_{test} > T_{train}$인 경우에도 적용 가능합니다.

### 3.2 Flexible Horizon Generalization

샘플링 시 시퀀스 전반에 걸쳐 다양한 노이즈 레벨을 활용함으로써, **자동 회귀 rollout 안정화, 장기 horizon에 걸친 가이던스, 인과적 불확실성을 활용한 계획**과 같은 유연한 동작을 달성할 수 있습니다.

### 3.3 노이즈를 통한 마스킹과 일반화

논문에서는 역사 길이를 제어함으로써 compositionality를 달성할 수 있음을 보입니다. 더 나아가, noise-as-masking 방식을 활용하면 모델이 **불필요한 역사를 무시하고 더 짧은 horizon에만 조건화하는 방법을 스스로 학습**할 가능성도 존재합니다.

### 3.4 Non-causal 모델로의 확장 가능성

Diffusion Forcing은 이 논문에서 의사결정에 중요한 인과성 때문에 causal 모델로 구현되어 있습니다. 그러나 noise-as-masking 아이디어는 non-causal 모델에도 적용 가능합니다. 샘플링 시 non-causal 버전을 학습한 뒤, 예측에 보여주고 싶지 않은 항목을 순수 Gaussian 노이즈로 마스킹함으로써 causal하게 만들 수도 있습니다.

### 3.5 다중 에이전트 및 멀티모달 도메인으로의 일반화

이 아이디어는 다중 인물 상호작용 이해 및 생성이라는 근본적인 문제로 확장됩니다. 긴 시간 horizon, 강한 에이전트 간 의존성, 가변적인 그룹 크기 때문에 이러한 상호작용 모델링은 어려움이 있습니다.

멀티모달 환경에서는 시간 × 모달리티 노이즈 행렬이 샘플링되어 공급되며, 다중 에이전트 응용에서는 각 에이전트의 모션 토큰이 독립적으로 노이즈가 추가되어, 토큰별 수준에 조건화된 Transformer 기반 denoisers가 유연한 inpainting 또는 turn-taking을 지원할 수 있습니다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4.1 후속 연구에 미치는 영향

#### ✦ 비디오 생성 (Video Generation)

2025년에는 Diffusion Forcing의 스케일업이 이루어져, state-of-the-art인 Wan2.1-T2V-1.3B를 단 20k 스텝, 49 프레임으로 파인튜닝하고 5배 확장된 217 프레임까지 안정적으로 rollout하는 성과가 달성되었습니다.

이 접근법은 **history corruption** 전략—과거 프레임에 노이즈를 주입하여 역사에 대한 과도한 의존을 줄임—으로 활용되어, 드리프트(drift)를 완화시킵니다. 다만 clean한 참조를 박탈하여 시간적 일관성을 다소 저해하는 트레이드오프가 있습니다.

#### ✦ 이산 언어 모델로의 확장 (Discrete LLM)

후속 연구에서 DF 기법은 연속 공간 diffusion 모델에서 이산 데이터로 확장되어, **Discrete Diffusion Forcing(D2F)** 방법으로 발전되었습니다.

#### ✦ 다중 에이전트 상호작용 모델링

MAGNet(Multi-Agent Generative Network)이라는 통합 자동 회귀 diffusion 프레임워크가 제안되었으며, 광범위한 상호작용 유형을 지원하고 수백 개의 모션 스텝에 걸친 초장기 시퀀스를 자동 회귀적으로 생성할 수 있습니다.

#### ✦ Self Forcing (Exposure Bias 해결)

Self Forcing은 자동 회귀 비디오 diffusion 모델의 오래된 exposure bias 문제를 해결합니다. 미래 프레임을 ground-truth 컨텍스트 프레임 기반으로 denoising하는 이전 방법들과 달리, Self Forcing은 훈련 중 KV 캐싱을 사용한 자동 회귀 rollout을 통해 이전에 자체 생성된 출력에 각 프레임의 생성을 조건화합니다.

---

### 4.2 앞으로 연구 시 고려할 점

| 고려 사항 | 설명 |
|---|---|
| **스케일업** | 더 큰 데이터셋과 더 다양한 도메인으로 Diffusion Forcing을 스케일업하는 것이 중요한 과제입니다. |
| **샘플 효율성** | 임의의 마스크 패턴으로 샘플링 또는 denoising하면 처리해야 할 공간이 확장되어, 경험적 또는 계산 비용이 증가할 수 있습니다. |
| **훈련 안정성** | Non-causal 아키텍처(예: 양방향 windowed attention) 및 적절한 스케줄링(하삼각형) 선택은 일부 모달리티에서 필수적이며 비자명적일 수 있습니다. |
| **Noise Schedule 설계** | 다양한 도메인에서의 최적 per-token 노이즈 스케줄 설계는 아직 완전히 탐구되지 않았으며, 도메인별 최적화가 필요합니다. |
| **이산 토큰으로의 확장** | 이산 언어 모델에서의 완전한 통합은 D2F와 같은 별도 연구가 필요하며, 연속-이산 통합 프레임워크 연구가 요구됩니다. |
| **Exposure Bias 완화** | 훈련과 추론 간의 분포 불일치를 줄이기 위한 Self Forcing 등의 기법과의 결합 연구가 중요합니다. |
| **멀티모달 확장** | 비전-언어-행동이 결합된 embodied AI 환경에서의 Diffusion Forcing 적용은 유망한 연구 방향입니다. |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 발표 | 핵심 아이디어 | Diffusion Forcing과의 관계 |
|---|---|---|---|
| **DDPM** (Ho et al.) | NeurIPS 2020 | 표준 Denoising Diffusion | 기반 기술 |
| **Decision Transformer** (Chen et al.) | NeurIPS 2021 | Transformer로 offline RL | Next-token prediction 계열 |
| **Diffuser** (Janner et al.) | NeurIPS 2022 | 계획을 위한 full-sequence diffusion | DF의 한계 동기 부여 |
| **AR-Diffusion** (Wu et al.) | NeurIPS 2023 | 텍스트 생성을 위한 자동 회귀 diffusion | DF와 유사한 방향, 이산 토큰 |
| **Diffusion Forcing** (Chen et al.) | **NeurIPS 2024** | **per-token 독립 노이즈, causal+diffusion 통합** | **본 논문** |
| **Self Forcing** (Huang et al.) | 2025 | AR 비디오 diffusion의 exposure bias 해결 | DF의 한계 직접 해결 |
| **D2F** (이산 Diffusion Forcing) | 2025 | 이산 공간으로 DF 확장 | DF의 이산 확장 |
| **MAGNet** (Maluleke et al.) | 2025 | 다중 에이전트 상호작용 시퀀스 모델링 | DF 프레임워크 직접 활용 |

---

## 참고 자료 (출처)

1. **arXiv 원문**: Chen, B. et al. (2024). *Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion*. arXiv:2407.01392. https://arxiv.org/abs/2407.01392
2. **NeurIPS 2024 공식 논문**: https://proceedings.neurips.cc/paper_files/paper/2024/file/2aee1c4159e48407d68fe16ae8e6e49e-Paper-Conference.pdf
3. **프로젝트 페이지**: https://www.boyuan.space/diffusion-forcing/
4. **OpenReview (NeurIPS 2024 Poster)**: https://openreview.net/forum?id=yDo1ynArjj
5. **GitHub 공식 코드**: https://github.com/buoyancy99/diffusion-forcing
6. **MIT CSAIL 저자 제공 PDF**: https://groups.csail.mit.edu/robotics-center/public_papers/Chen24.pdf
7. **NeurIPS Virtual Poster**: https://neurips.cc/virtual/2024/poster/93029
8. **Semantic Scholar**: https://www.semanticscholar.org/paper/Diffusion-Forcing:-Next-token-Prediction-Meets-Chen-Monso/40d63dc2b465c9081e4efc5a19514da151e97fe7
9. **ACL/ACM DL**: https://dl.acm.org/doi/10.5555/3737916.3738675
10. **후속 연구 - D2F**: https://arxiv.org/abs/2508.09192
11. **후속 연구 - Self Forcing**: https://arxiv.org/abs/2506.08009
12. **후속 연구 - MAGNet (Multi-Agent)**: https://arxiv.org/abs/2512.17900
13. **후속 연구 - Rolling Forcing**: https://arxiv.org/html/2509.25161v1
14. **관련 이론 - Emergent Mind (ELBO 분석)**: https://www.emergentmind.com/topics/diffusion-forcing

> ⚠️ **주의사항**: 본 답변에 포함된 수식(포워드 프로세스, 학습 손실 함수, MCTG 목적함수 등)은 논문의 핵심 아이디어에 기반하여 표준 DDPM 표기법과 논문의 서술을 결합하여 정리한 것입니다. 논문의 정확한 구현 수식은 원문 PDF의 Algorithm 1 및 Section 3을 직접 참조하시기 바랍니다.
