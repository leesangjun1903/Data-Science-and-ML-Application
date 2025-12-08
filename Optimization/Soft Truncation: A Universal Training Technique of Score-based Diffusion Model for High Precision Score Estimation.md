# Soft Truncation: A Universal Training Technique of Score-based Diffusion Model for High Precision Score Estimation

***

# 1. 핵심 주장 및 주요 기여 요약

이 논문은 **확산 모델(Diffusion Models)**에서 발생하는 **밀도 추정(NLL) 성능과 샘플 생성(FID) 성능 간의 트레이드오프(Trade-off)** 문제를 해결하는 것을 핵심 목표로 합니다.

*   **핵심 주장:** 기존의 확산 모델은 학습 시 고정된 '절단 시간(Truncation time, $\epsilon$)'을 사용하는데, $\epsilon$이 작으면 NLL(작은 $t$ 의존)이 좋아지지만 FID(큰 $t$ 의존)가 나빠지고, 반대의 경우도 성립합니다. 이 역상관 관계는 학습 손실(Loss)의 스케일 불균형 때문입니다.
*   **주요 기여:** 고정된 $\epsilon$ 대신 확률 변수 $\tau$를 사용하는 **Soft Truncation** 기법을 제안했습니다. 이 방법은 모델 구조 변경 없이 학습 과정만 수정하여, **NLL 손실을 유지하면서도 FID 성능을 비약적으로 향상(SOTA 달성)**시키는 범용적인 학습 기법입니다.

***

# 2. 논문 상세 분석

### 2.1. 해결하고자 하는 문제 (Problem Statement)
*   **역상관 관계(Inverse Correlation):** 확산 모델 연구에서 **밀도 추정(NLL)** 성능은 확산 시간이 매우 작은 구간(small diffusion time)의 스코어 추정에 크게 의존하는 반면, **샘플 품질(FID)**은 확산 시간이 큰 구간(large diffusion time)의 스코어 정확도에 좌우됩니다.
*   **손실 불균형(Loss Imbalance):** 확산 시간이 0에 가까워질수록 손실 함수의 분산이 폭발적으로 증가하기 때문에, 기존 연구들은 $t \in [\epsilon, T]$ 범위로 적분을 제한하는 **Hard Truncation**을 사용했습니다.
*   **딜레마:** 고정된 $\epsilon$을 사용하면, 작은 $\epsilon$은 NLL을 개선하지만 큰 $t$에서의 학습을 방해하여 FID를 망치고, 큰 $\epsilon$은 그 반대 현상을 초래합니다.

### 2.2. 제안하는 방법: Soft Truncation
저자들은 고정된 하이퍼파라미터 $\epsilon$을 확률 분포 $P(\tau)$를 따르는 확률 변수 $\tau$로 대체하는 **Soft Truncation**을 제안합니다.

*   **기본 수식:**
    매 학습 스텝마다 새로운 절단 시간 $\tau$를 샘플링하여 아래의 손실 함수를 최적화합니다.

$$ L_{ST}(\theta; g^2, P) := E_{P(\tau)} [L(\theta; g^2, \tau)] $$

$$ = \int_{0}^{T} P(\tau) \left( \frac{1}{2} \int_{\tau}^{T} g^2(t) E_{x_t} [\| s_\theta(x_t, t) - \nabla \log p_{0t}(x_t|x_0) \|^2_2] dt \right) d\tau $$

여기서 $g(t)$는 확산 계수, $s_\theta$는 스코어 네트워크입니다.

*   **MPLE (Maximum Perturbed Likelihood Estimation):**
    이 방법은 이론적으로 **MPLE**로 해석될 수 있습니다. 즉, $\tau$ 시점만큼 섭동(perturbation)된 데이터 분포의 우도(Likelihood)를 최대화하는 과정으로 볼 수 있으며, 이는 보조적인 무작위성(auxiliary randomness)을 주입하여 학습을 안정화하고 손실 분산을 제어하는 효과를 냅니다.

### 2.3. 모델 구조
*   **범용성:** Soft Truncation은 특정 모델 아키텍처에 종속되지 않는 **학습 기법(Training Technique)**입니다.
*   **적용 대상:** 논문에서는 NCSN++, DDPM++ 등 다양한 백본(Backbone)과 VP-SDE, VE-SDE 등 여러 확산 확률 미분 방정식(SDE) 설정에 이 기법을 적용하여 그 효과를 검증했습니다.

### 2.4. 성능 향상 및 한계
*   **성능 향상:**
    *   **CIFAR-10, CelebA, STL-10** 등 주요 벤치마크에서 기존 SOTA(State-of-the-Art) 모델 대비 동등하거나 더 우수한 NLL을 유지하면서도, **FID 점수를 크게 개선**했습니다.
    *   특히 CelebA 데이터셋에서 기존 최고 성능을 압도하는 결과를 보였습니다.
*   **한계:**
    *   **최적 분포 탐색:** 확률 변수 $\tau$의 분포 $P(\tau)$를 어떻게 설정하느냐에 따라 성능이 달라집니다. 논문에서는 실험적으로 $P(\tau) \propto 1/\tau^k$ 형태에서 $k \approx 1$이 좋다는 것을 발견했으나, 이론적으로 최적의 분포를 도출하는 것은 향후 과제로 남겨두었습니다.[1][2]

***

# 3. 모델의 일반화 성능 향상 가능성

이 논문에서 가장 강조하는 강점 중 하나는 **"범용성(Universality)"**과 이에 따른 일반화 성능입니다.

1.  **아키텍처 독립성:** 특정 네트워크 구조(예: U-Net, Transformer)에 의존하지 않고, 손실 함수 계산 방식만 변경하므로 모든 스코어 기반 확산 모델에 즉시 적용 가능합니다.
2.  **다양한 SDE 지원:** 선형 SDE(VP, VE)뿐만 아니라 비선형 SDE(INDM 등)에서도 성능 향상이 입증되었습니다. 이는 모델이 학습 데이터나 확산 과정의 종류에 구애받지 않고 **강건한(Robust) 일반화 성능**을 보일 수 있음을 시사합니다.
3.  **전 구간 학습 균형:** 기존 방식이 특정 시간대($\epsilon$ 근처)에 학습이 편중되는 과적합(Overfitting) 경향을 보였다면, Soft Truncation은 $P(\tau)$를 통해 전체 시간대 $[0, T]$에 걸쳐 고르게 학습 기회를 제공합니다. 이는 보지 못한 데이터나 다양한 생성 시나리오에서도 안정적인 성능을 내는 **일반화 능력의 핵심 요인**이 됩니다.

***

# 4. 향후 연구에 미치는 영향 및 고려할 점

### 4.1. 연구 영향력 (Impact)
*   **NLL-FID 이분법 타파:** 기존 연구들이 "NLL 최적화 모델"과 "FID 최적화 모델"로 양분되어 발전하던 경향을 깨고, 단일 학습 프레임워크로 두 지표를 동시에 잡을 수 있다는 가능성을 열었습니다.[2][3]
*   **랜덤화된 손실 함수 연구 촉진:** 고정된 하이퍼파라미터를 확률 변수로 "Softening"하는 접근법은 이후 확산 모델의 학습 스케줄링이나 가중치 함수 연구에 영감을 주었습니다.

### 4.2. 연구 시 고려할 점
*   **분포 $P(\tau)$의 최적화:** 후속 연구를 진행할 때, 단순한 역수 형태($1/\tau^k$) 외에 데이터셋의 특성에 맞는 적응형(Adaptive) 분포나 학습 가능한(Learnable) 분포를 탐색할 필요가 있습니다.
*   **계산 비용:** 매 스텝마다 적분 구간이 달라지므로, 이를 효율적으로 근사하거나 계산하는 알고리즘 최적화가 대규모 모델 적용 시 고려되어야 합니다.

***

# 5. 2020년 이후 관련 최신 연구 탐색

본 논문 이후(2022년~현재), NLL과 FID 트레이드오프를 다루거나 Soft Truncation을 인용/발전시킨 주요 연구들은 다음과 같습니다.

1.  **Breaking the Likelihood–Quality Trade-off in Diffusion Models (ArXiv, 2024/2025):**
    *   Soft Truncation이 트레이드오프를 완화하긴 했으나 여전히 한계가 있다고 지적하며, 이를 뛰어넘는 새로운 손실 가중치 조절 기법을 제안했습니다. 이 연구는 Soft Truncation을 직접적인 비교군(Baseline)으로 삼고 있습니다.[4][3]
2.  **Diffusion Models Without Time Truncation (OpenReview):**
    *   Soft Truncation과 같은 휴리스틱한 절단 기법이 실제 SDE와 추정 SDE 간의 불일치를 야기할 수 있음을 지적하며, 절단 없이(Truncation-free) 학습할 수 있는 이론적 토대를 마련하고자 하는 연구입니다.[5]
3.  **Consistency Trajectory Models (CTM) (Kim et al., 2023):**
    *   Soft Truncation과 유사하게 NLL과 FID를 동시에 잡으려는 시도로, 데이터 증강(Augmentation)과 다양한 손실 항을 결합하여 Soft Truncation보다 더 나은 성능을 달성했다고 보고했습니다.[4]
4.  **Global Well-posedness of Score-based Generative Models (2024):**
    *   확산 모델의 수렴성을 수학적으로 증명하는 과정에서 Soft Truncation과 같은 기법이 스코어 추정의 정밀도에 미치는 영향을 이론적으로 분석하며 인용하였습니다.[6]

이러한 연구 흐름은 **"단순한 휴리스틱 절단을 넘어, 이론적으로 정당하고 성능 균형이 잡힌 완전한 확산 학습법"**으로 나아가고 있음을 보여줍니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/300156a2-b950-4add-a528-1fb68b739f4e/2106.05527v5.pdf)
[2](https://proceedings.mlr.press/v162/kim22i/kim22i.pdf)
[3](https://arxiv.org/html/2511.19434v1)
[4](https://arxiv.org/html/2511.19434)
[5](https://openreview.net/pdf?id=1rg56KzwsS)
[6](https://www.semanticscholar.org/paper/Soft-Truncation:-A-Universal-Training-Technique-of-Kim-Shin/3f8109dfcca0bf6154ae860f5571fbcc4fc69930)
[7](https://www.semanticscholar.org/paper/3f8109dfcca0bf6154ae860f5571fbcc4fc69930)
[8](https://ieeexplore.ieee.org/document/10628863/)
[9](https://arxiv.org/abs/2307.00773)
[10](https://aapm.onlinelibrary.wiley.com/doi/10.1002/mp.17865)
[11](https://www.tarupublications.com/doi/10.47974/CJSIM-2022-0083)
[12](https://pubs.aip.org/cha/article/34/9/093132/3313779/Exponential-stability-and-fixed-time-control-of-a)
[13](http://www.emerald.com/el/article/41/1/111-136/36405)
[14](https://xlink.rsc.org/?DOI=D2SM01094A)
[15](https://ieeexplore.ieee.org/document/10915728/)
[16](https://link.springer.com/10.1007/s10334-024-01153-y)
[17](https://arxiv.org/pdf/2310.01693.pdf)
[18](https://arxiv.org/html/2404.11895v2)
[19](https://arxiv.org/html/2412.02852v1)
[20](https://arxiv.org/abs/2410.12557v1)
[21](https://arxiv.org/pdf/2202.05910.pdf)
[22](https://aclanthology.org/2023.repl4nlp-1.6.pdf)
[23](https://arxiv.org/pdf/2310.09469.pdf)
[24](https://arxiv.org/html/2410.21721)
[25](https://pure.kaist.ac.kr/en/publications/soft-truncation-a-universal-training-technique-of-score-based-dif)
[26](https://arxiv.org/abs/2106.05527)
[27](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
[28](https://github.com/Kim-Dongjun/Soft-Truncation)
[29](https://openreview.net/notes/edits/attachment?id=R9hnBH0fRI&name=pdf)
