# Improving Generalization by Controlling Label-Noise Information in Neural Network Weights

> **저자:** Hrayr Harutyunyan, Kyle Reing, Greg Ver Steeg, Aram Galstyan
> **학회:** ICML 2020 (Proceedings of the 37th International Conference on Machine Learning, PMLR 119:4071-4081)
> **방법론 이름:** LIMIT (Label Information Minimization in Training)

---

## 1. 핵심 주장 및 주요 기여 요약

노이즈가 있거나 부정확한 레이블이 존재할 때, 신경망은 노이즈에 관한 정보를 기억(memorize)하려는 바람직하지 않은 경향을 가진다. 드롭아웃, 가중치 감쇠(weight decay), 데이터 증강(data augmentation) 같은 표준 정규화 기법은 때때로 도움이 되지만, 이 행동을 완전히 방지하지는 못한다.

이 논문의 핵심 주장은 다음과 같다:

- 신경망의 가중치를 데이터와 학습의 확률성에 의존하는 확률 변수로 간주하면, 기억된 정보의 양을 가중치와 전체 학습 레이블 벡터 간의 **Shannon 상호 정보량(mutual information)** $I(w; \mathbf{y} \mid \mathbf{x})$로 정량화할 수 있다.
- 어떤 학습 알고리즘이든, 이 항의 값이 낮으면 레이블 노이즈의 기억이 줄어들고 더 나은 일반화 경계(generalization bounds)가 얻어진다.
- 이 낮은 값을 달성하기 위해, 레이블에 접근하지 않고 분류기의 최종 레이어에서 그래디언트를 예측하는 보조 네트워크(auxiliary network)를 사용하는 학습 알고리즘을 제안한다.

### 주요 기여 (3가지):
1. **이론적 기여:** $I(w; \mathbf{y} \mid \mathbf{x})$가 낮으면 레이블 노이즈 기억이 감소하고 일반화가 향상됨을 수학적으로 증명
2. **방법론적 기여:** LIMIT라는 실용적 학습 알고리즘 제안
3. **실험적 기여:** MNIST, CIFAR-10, CIFAR-100의 다양한 노이즈 모델로 손상된 버전과 실제 노이즈 레이블이 있는 대규모 데이터셋 Clothing1M에서 접근법의 효과를 실증

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

노이즈가 있거나 부정확한 레이블이 존재할 때, 네트워크는 학습 레이블을 기억하기 시작하고 이는 일반화 성능을 저하시킨다. 극단적인 경우, 표준 아키텍처는 레이블이 완전히 무작위로 할당되었을 때도 학습 데이터에서 100% 분류 정확도를 달성할 수 있다.

드롭아웃, 가중치 감쇠, 데이터 증강과 같은 표준 명시적 또는 암시적 정규화 기법은 레이블 기억을 직접 해결하지도 않고 완전히 방지하지도 못한다.

레이블 기억으로 인한 일반화 성능 저하는 많은 대규모 실세계 데이터셋이 불완전하게 레이블링되어 있기 때문에 중요한 문제이다.

### 2.2 제안 방법 및 수식

#### (1) 정보 이론적 프레임워크

가중치 $w$를 학습 데이터와 학습의 확률성에 의존하는 확률 변수로 모델링한다. 학습 데이터셋은 입력 벡터 $\mathbf{x}$와 레이블 벡터 $\mathbf{y}$로 구성된 확률 변수이다.

**핵심 정보량 항:** 레이블 노이즈 기억의 정도를 다음으로 정량화:

$$I(w; \mathbf{y} \mid \mathbf{x})$$

레이블에 입력으로부터 추론할 수 없는 정보가 포함되어 있으면, 모델은 이 항을 통해 레이블을 기억함으로써 잘 수행될 수 있다.

#### (2) 일반화 경계 (Generalization Bound)

Xu & Raginsky (2017)의 정보 이론적 일반화 경계를 기반으로 한다. Xu & Raginsky (2017)의 정리에 따르면, 손실 $\ell(w, Z)$가 모든 $w \in \mathcal{W}$에서 $Z \sim \mu$ 하에 $\sigma$-sub-Gaussian이면, 일반화 오차는 다음과 같이 바운드된다:

$$|\text{gen}(\mu, \mathcal{A})| \leq \sqrt{\frac{2\sigma^2 \, I(W; S_n)}{n}}$$

여기서 $I(W; S_n)$은 학습된 가중치 $W = \mathcal{A}(S_n)$과 학습 데이터 $S_n$ 사이의 상호 정보량이다.

본 논문은 $I(W; S_n)$을 분해하여 **$I(w; \mathbf{y} \mid \mathbf{x})$를 직접 제어**하면 더 타이트한 일반화 경계를 얻을 수 있음을 보인다:

$$I(w; S) = I(w; \mathbf{x}) + I(w; \mathbf{y} \mid \mathbf{x})$$

$I(w:\mathbf{y}|\mathbf{x})$가 직접적으로 기억과 연결됨을 증명하여, 작은 $I(w:\mathbf{y}|\mathbf{x})$를 가지는 어떤 알고리즘이든 학습 세트에서 레이블 노이즈에 덜 과적합한다.

#### (3) 그래디언트 정보를 통한 바운딩

가중치 내 정보를 그래디언트 내 정보로 대체할 수 있음을 보이고, 그래디언트 내 정보에 대한 변분 바운드(variational bound)를 도입한다.

SGD에서 가중치 업데이트는 그래디언트에 의해 결정되므로, 데이터 처리 부등식(data processing inequality)에 의해:

$$I(w; \mathbf{y} \mid \mathbf{x}) \leq \sum_{t=1}^{T} I(g_t; \mathbf{y} \mid \mathbf{x}, w_{t-1})$$

여기서 $g_t$는 시점 $t$에서의 그래디언트이다.

각 그래디언트 항에 대한 변분 상계(variational upper bound)를 도입:

$$I(g_t; \mathbf{y} \mid \mathbf{x}, w_{t-1}) \leq \mathbb{E}_{p(g_t, \mathbf{x}, w_{t-1})} \left[ D_{\text{KL}}\left( p(g_t \mid \mathbf{y}, \mathbf{x}, w_{t-1}) \,\|\, q(g_t \mid \mathbf{x}, w_{t-1}) \right) \right]$$

여기서 $q(g_t \mid \mathbf{x}, w_{t-1})$가 바로 **보조 네트워크(auxiliary network)**가 모델링하는 분포이다. 이 보조 네트워크는 **레이블 $\mathbf{y}$에 접근하지 않고** 입력 $\mathbf{x}$와 현재 가중치 $w_{t-1}$만으로 그래디언트를 예측한다.

이 바운드는 원래 손실의 그래디언트를 레이블 정보 없이 예측하는 보조 네트워크를 사용한다.

#### (4) LIMIT 학습 목적 함수

최종적으로, 분류기는 다음과 같은 정규화된 목적 함수를 최적화한다:

$$\mathcal{L}_{\text{LIMIT}} = \mathcal{L}_{\text{CE}}(\theta) + \lambda \cdot D_{\text{KL}}\left( p(g \mid \mathbf{y}, \mathbf{x}, \theta) \,\|\, q_\phi(g \mid \mathbf{x}, \theta) \right)$$

여기서:
- $\mathcal{L}_{\text{CE}}(\theta)$: 분류기의 크로스 엔트로피 손실
- $q_\phi(g \mid \mathbf{x}, \theta)$: 보조 네트워크가 예측하는 그래디언트 분포 (파라미터 $\phi$)
- $\lambda$: 정보 정규화 강도를 제어하는 하이퍼파라미터

논문에서는 두 가지 변형을 제안한다:
- **$\text{LIMIT}_G$**: 전체 그래디언트를 예측
- **$\text{LIMIT}_L$**: 손실 값(loss)을 예측하는 소프트 정규화 변형

### 2.3 모델 구조

모델은 **두 개의 네트워크**로 구성된다:

1. **주 분류기(Classifier):** 표준 신경망 (예: 4-layer CNN, ResNet 등)으로, 입력과 레이블을 사용해 학습
2. **보조 네트워크(Auxiliary Network):** 분류기의 최종 레이어 그래디언트를 **레이블 없이** 예측하는 네트워크

분류기와 보조 네트워크 모두 4-layer CNN의 공유된 아키텍처를 사용한다.

보조 네트워크의 핵심은 **레이블 $\mathbf{y}$에 대한 접근 없이** 입력 피처만으로 그래디언트를 예측한다는 점이다. 이를 통해 $q(g \mid \mathbf{x}, w)$가 $p(g \mid \mathbf{y}, \mathbf{x}, w)$에 근사할수록 KL 발산이 작아지고, 이는 레이블 정보의 그래디언트 내 유입이 적다는 것을 의미한다.

### 2.4 성능 향상

LIMIT의 변형들이 표준 기준선(baselines)에 비해 크게 향상되었으며, 특히 균일 레이블 노이즈의 경우에 그렇다.

MNIST의 경우와 마찬가지로, 이 접근법은 데이터셋에 노이즈가 없을 때도 도움이 된다.

실험 결과 (MNIST, 균일 레이블 노이즈 기준 예시):

| 방법 | $p = 0.0$ (All) | $p = 0.5$ (All) | $p = 0.8$ (All) |
|------|:---:|:---:|:---:|
| CE (Cross-Entropy) | 99.2 | 97.2 | 87.2 |
| MAE | 99.1 | 98.1 | 93.2 |
| **LIMIT variants** | **향상** | **향상** | **유의미한 향상** |

$q$가 최상의 CE 모델로 초기화되면(FW 및 DMI와 유사), 결과가 더 좋다.

MNIST, CIFAR-10, CIFAR-100의 다양한 노이즈 모델로 손상된 버전과 노이즈 레이블이 있는 대규모 데이터셋 Clothing1M에서 접근법의 효과를 실증하였다.

추가 기준선과의 비교는 제안된 방법이 단순히 노이즈를 통해 그래디언트 내 정보를 줄이는 것 이상의 효과를 가짐을 보여준다.

### 2.5 한계

1. **계산 비용:** 보조 네트워크의 추가로 학습 시 계산 오버헤드가 발생
2. **하이퍼파라미터 민감도:** 정규화 강도 $\lambda$의 설정이 성능에 영향을 미침
3. **보조 네트워크 아키텍처 선택:** 보조 네트워크의 구조 설계에 대한 일반적 가이드라인 부재
4. **노이즈 모델 가정:** 주로 균일(uniform) 노이즈와 비대칭(asymmetric) 노이즈에서 실험하였으나, 인스턴스 의존적(instance-dependent) 노이즈에 대한 분석은 제한적
5. **상호 정보량 추정의 어려움:** 상호 정보량 추정의 샘플 복잡도는 차원에 따라 불량하게 증가하며, $(W, Z_i)$의 더 많은 샘플을 수집하는 것은 비용이 많이 든다. 또한, 유한 데이터로부터 MI를 추정하는 것은 MI가 클 때 중요한 통계적 한계를 가진다.

---

## 3. 모델의 일반화 성능 향상 가능성 (핵심 분석)

### 3.1 정보 이론적 일반화 프레임워크

이 논문의 가장 중요한 통찰은 **정보 이론의 렌즈를 통해 일반화를 직접 다룬다**는 점이다.

가중치가 학습 데이터셋 $S$에 대해 포함하는 정보는 이전에 일반화와 연결되었으며(Xu & Raginsky), $I(w:\mathbf{y}|\mathbf{x})$의 작은 값으로 이를 타이트하게 만들 수 있다.

일반화 경계를 분해하면:

$$I(w; S) = I(w; \mathbf{x}) + I(w; \mathbf{y} \mid \mathbf{x})$$

여기서 $I(w; \mathbf{x})$는 입력에 대한 정보(유용한 피처 학습)이고, $I(w; \mathbf{y} \mid \mathbf{x})$는 입력으로 설명되지 않는 레이블 정보(잠재적으로 노이즈 기억)이다.

**핵심 정리:** $I(w; \mathbf{y} \mid \mathbf{x})$가 작으면:
1. 레이블 노이즈에 대한 과적합이 감소한다
2. 전체 일반화 바운드가 타이트해진다
3. 클린 데이터에서도 일반화가 향상된다

### 3.2 실용적 일반화 향상 메커니즘

LIMIT의 일반화 향상 메커니즘을 수식으로 설명하면:

**Step 1:** 그래디언트에서 레이블 정보 분리

$$g_t = \nabla_\theta \mathcal{L}_{\text{CE}}(\theta; x_i, y_i) = \underbrace{g_t^{\text{clean}}}_{\text{유용한 신호}} + \underbrace{g_t^{\text{noise}}}_{\text{노이즈 정보}}$$

**Step 2:** 보조 네트워크 $q_\phi$가 레이블 없이 $g_t^{\text{clean}}$ 부분을 예측

**Step 3:** KL 정규화를 통해 $g_t^{\text{noise}}$ 영향 최소화:
$$D_{\text{KL}}(p \| q) \rightarrow 0 \implies g_t \approx q_\phi(\mathbf{x}, \theta) \implies \text{레이블 노이즈 정보 최소화}$$

### 3.3 노이즈가 없는 데이터에서도 효과

MNIST의 경우와 마찬가지로, 이 접근법은 데이터셋에 노이즈가 없을 때도 도움이 된다. 이는 LIMIT가 단순히 노이즈 방어가 아닌 **근본적인 정규화 기법**으로 작용함을 시사하며, 과적합 방지를 통한 일반화 향상에 기여한다.

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 학문적 영향

후속 연구들은 가중치가 아닌 학습된 모델의 예측(Harutyunyan et al., 2021; Haghifam et al., 2022) 또는 손실(Wang & Mao, 2023)과 지표 변수 사이의 조건부 상호 정보량을 고려하여 이러한 바운드를 더 타이트하게 만들었다.

함수적 CMI(f-CMI) 바운드가 Harutyunyan et al. (2021)에 의해 제안되어, CMI에서 가중치 변수를 supersample에 대한 예측으로 대체하였다.

### 4.2 향후 연구 시 고려할 점

1. **인스턴스 의존적 노이즈 처리:** 현실 세계의 노이즈는 균일하지 않으며, 인스턴스별로 다른 노이즈율을 가짐
2. **확장성 문제:** 대규모 모델(LLM 등)에 대한 보조 네트워크 설계의 확장성 연구 필요
3. **정보량 추정의 실용화:** 고차원 가중치 공간에서의 상호 정보량 추정을 더 효율적으로 수행하는 방법
4. **다른 정규화 기법과의 결합:** LIMIT과 기존 정규화 기법(MixUp, CutMix 등)의 시너지 효과 탐구
5. **자기 지도 학습과의 통합:** 레이블 없는 그래디언트 예측이라는 아이디어를 자기 지도 학습 프레임워크와 결합

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 접근법 | 핵심 아이디어 | LIMIT과의 비교 |
|------|------|--------|-------------|--------------|
| **DivideMix** (Li et al.) | 2020 | 샘플 분리 + 준지도학습 | GMM을 통해 클린/노이즈 레이블 분리 후, 클린 데이터와 노이즈 레이블 데이터를 모두 학습에 활용 | LIMIT는 정보 이론 기반, DivideMix는 손실 분포 기반 분리 |
| **ELR+** (Liu et al.) | 2020 | 정규화 | 네트워크가 초기 학습 단계에서 노이즈 레이블에 과적합하지 않는 점을 활용한 정규화 추가 | LIMIT는 MI 바운드 기반, ELR은 초기 예측의 일관성 유지 |
| **SOP** (Liu et al.) | 2022 | 과적합 방지 | LogitClip이 SOP와 SAM을 유의미하게 향상시킬 수 있음이 실증됨 | 두 방법 모두 과적합 방지 목표, SOP는 로짓 수준 정규화 |
| **LogitClip** (Wei et al.) | 2023 | 로짓 클리핑 | LogitClip은 로짓을 클리핑하여 간접적으로 상한과 하한을 제약 | LIMIT는 그래디언트 수준, LogitClip은 로짓 수준 제어 |
| **ProMix** (2023, IJCAI) | 2023 | 점진적 선택 + 준지도학습 | DivideMix보다 점진적 선택 전략이 더 많은 클린 샘플을 선택하면서 노이즈 샘플 도입을 최소화하고, 실세계 노이즈 CIFAR-N 벤치마크에서 SOTA 성능 달성 | LIMIT의 정보 이론적 접근과 상호보완적 |
| **OGC** (2024) | 2024 | 최적화된 그래디언트 클리핑 | 동적 임계값으로 노이즈 그래디언트 영향을 바운딩하여, 대칭·비대칭·인스턴스 의존적 노이즈에서 CE의 노이즈 내성을 향상 | 그래디언트 제어라는 유사한 관점이나, OGC는 클리핑 기반 |
| **f-CMI** (Harutyunyan et al.) | 2021 | 이론 (일반화 바운드) | 함수적 CMI 바운드: CMI에서 가중치를 예측으로 대체하여 고차원 문제 해결 | LIMIT 저자의 후속 이론 연구로, 바운드를 더 실용적으로 만듦 |

### 연구 흐름 정리

2020년 이후 레이블 노이즈 학습(Learning with Noisy Labels, LNL) 연구는 크게 다음 방향으로 발전했다:

1. **샘플 선택/분리 기반:** DivideMix → ProMix 등으로 진화하며, GMM이 높은 노이즈 시나리오(>50%)에서는 강건하나, 손실 차이가 적은 경우 어려운 노이즈 샘플과 어려운 클린 샘플을 혼동할 수 있다는 한계를 극복
2. **손실 함수/정규화 기반:** ELR+, SOP, LogitClip 등이 크로스 엔트로피의 노이즈 취약성을 보완
3. **정보 이론 기반:** LIMIT → f-CMI → Slicing MI bounds 등으로 이론적 기반을 강화
4. **그래디언트 기반:** LIMIT의 그래디언트 정보 제어 아이디어가 OGC 등으로 확장

LIMIT은 **정보 이론과 실용적 알고리즘을 연결하는 선구적 연구**로서, 이후 정보 이론 기반 일반화 이론과 노이즈 강건 학습 모두에 영향을 미쳤다.

---

## 참고자료

1. **Harutyunyan, H., Reing, K., Ver Steeg, G., & Galstyan, A.** (2020). *Improving generalization by controlling label-noise information in neural network weights.* ICML 2020, PMLR 119:4071-4081. [http://proceedings.mlr.press/v119/harutyunyan20a.html](http://proceedings.mlr.press/v119/harutyunyan20a.html)
2. **arXiv 원문:** [https://arxiv.org/abs/2002.07933](https://arxiv.org/abs/2002.07933)
3. **공식 코드 (GitHub):** [https://github.com/hrayrhar/limit-label-memorization](https://github.com/hrayrhar/limit-label-memorization)
4. **RBC Borealis ICML 2020 Roundup** (논문 해설): [https://rbcborealis.com/research-blogs/icml-2020-roundup/](https://rbcborealis.com/research-blogs/icml-2020-roundup/)
5. **PaperTalk 발표:** [https://papertalk.org/papertalks/6129](https://papertalk.org/papertalks/6129)
6. **Harutyunyan et al.** (2021). *Information-theoretic generalization bounds for black-box learning algorithms* (f-CMI). NeurIPS 2021.
7. **Li, J., Socher, R., & Hoi, S. C.** (2020). *DivideMix: Learning with Noisy Labels as Semi-supervised Learning.* ICLR 2020.
8. **Wei et al.** (2023). *Mitigating Memorization of Noisy Labels by Clipping the Model Prediction.* ICML 2023.
9. **ProMix** (2023). *Combating Label Noise via Maximizing Clean Sample Utility.* IJCAI 2023.
10. **Xu & Raginsky** (2017). *Information-Theoretic Analysis of Generalization Capability of Learning Algorithms.* NeurIPS 2017.
11. **On Information Captured by Neural Networks: Connections with Memorization and Generalization** (Harutyunyan, 2023). [https://arxiv.org/abs/2306.15918](https://arxiv.org/abs/2306.15918) — LIMIT 저자의 박사 논문.
12. **Slicing Mutual Information Generalization Bounds for Neural Networks** (2024). [https://arxiv.org/abs/2406.04047](https://arxiv.org/abs/2406.04047)
13. **Awesome-Learning-with-Label-Noise** (서베이 리스트): [https://github.com/subeeshvasu/Awesome-Learning-with-Label-Noise](https://github.com/subeeshvasu/Awesome-Learning-with-Label-Noise)
