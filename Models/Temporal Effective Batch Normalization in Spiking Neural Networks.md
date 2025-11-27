# Temporal Effective Batch Normalization in Spiking Neural Networks

### 1. 핵심 주장과 주요 기여

**TEBN(Temporal Effective Batch Normalization)**은 스파이킹 신경망(SNN)의 훈련 효율성과 성능을 향상시키기 위해 제안된 혁신적인 정규화 기법입니다. 핵심 주장은 기존 배치 정규화(Batch Normalization, BN)가 SNN의 추가적인 시간 차원을 충분히 활용하지 못한다는 점입니다. TEBN의 주요 기여는 다음과 같습니다:[1]

- **시간적 분포 정규화**: 각 시간 단계(time-step)에서 상이한 가중치로 시냅스 전 입력을 재스케일링하여 시간 분포를 더 부드럽고 균일하게 만듦
- **최적화 경관 평탄화**: 이론적 분석을 통해 TEBN이 SNN의 최적화 경관을 평탄화하고 그래디언트 노름을 안정화할 수 있음을 증명
- **하이퍼파라미터 강건성**: 시간 상수 τ 변화에 대한 안정적인 정확도 유지로 다양한 신경 매개변수에 대한 강건성 입증

***

### 2. 논문이 해결하는 문제 및 제안 방법

#### 2.1 핵심 문제: 시간적 공변량 편이(Temporal Covariate Shift, TCS)

기존 배치 정규화는 스파이킹 신경망에서 발생하는 시간적 공변량 편이를 적절히 처리하지 못합니다. SNN의 시냅스 전류가 시간에 따라 순차적으로 공급될 때, 각 시간 단계의 입력 분포가 크게 변할 수 있으며, 이는 훈련 불안정성을 초래합니다.[1]

#### 2.2 제안 방법: TEBN의 수학적 공식

**LIF(Leaky-Integrate-and-Fire) 뉴런 모델:**[1]

시냅스 전 입력과 막 전위의 관계는 다음과 같이 표현됩니다:

$$x^{l-1}[t] = W^l o^{l-1}[t]$$

$$u^l[t] = \tau u^l[t-1](1 - o^l[t-1]) + \hat{x}^{l-1}[t]$$

여기서 $u^l[t]$는 $l$번째 계층의 막 전위, $\tau$는 누수 계수, $o^l[t]$는 출력 스파이크입니다.

**TEBN의 핵심 공식:**[1]

$$\hat{x}[t] = \text{TEBN}(x[t]) = \hat{\gamma}[t] \frac{x[t] - \mu}{\sqrt{\sigma^2 + \epsilon}} + \hat{\beta}[t]$$

$$\hat{\gamma}[t] = \gamma \times p[t], \quad \hat{\beta}[t] = \beta \times p[t]$$

여기서 $p[t]$는 각 시간 단계에서 학습 가능한 가중치이며, $\mu$와 $\sigma^2$는 모든 시간 단계의 샘플로부터 계산된 전체 평균과 분산입니다.

**시간적 공변량 편이의 분석:**[1]

전체 시간 단계에 대한 총 분산은 다음과 같이 분해됩니다:

$$\sigma_{\text{total}}^2 = \frac{1}{T}\sum_{i=1}^{T} \sigma^2[i] + \frac{1}{T}(1-\frac{1}{T})\sum_{i=1}^{T} \mu^2[i] - \frac{2}{T^2}\sum_{i\neq j} \mu[i]\mu[j]$$

각 시간 단계의 평균이 근사적으로 동일하다고 가정하면:

$$\hat{\sigma}_{\text{total}}^2 \approx \frac{1}{T}\sum_{i=1}^{T} \sigma^2[i]$$

이는 전체 분포가 T개의 독립 분포 조합으로 볼 수 있음을 의미합니다.

#### 2.3 그래디언트 계산

$p[t]$에 대한 손실 함수의 그래디언트:[1]

$$\frac{\partial L}{\partial p^l[t]} = \sum_{i} \left(\frac{\partial L}{\partial u_i^l[t]}\left[\gamma_i^l \frac{x_i^{l-1}[t] - \mu_i^l}{\sqrt{(\sigma_i^l)^2 + \epsilon}} + \beta_i^l\right]\right)$$

이를 통해 각 시간 단계의 가중치가 자동으로 최적의 분포 조정을 학습합니다.

#### 2.4 모델 구조

TEBN은 기존 SNN 아키텍처에 최소한의 추가 매개변수로 통합됩니다:

- **BNTT(시간-독립 매개변수)**: 각 시간 단계마다 별도의 $\gamma[t], \beta[t], \mu[t], \sigma^2[t]$ 사용 → 매개변수 복잡도 높음
- **tdBN(공유 매개변수)**: 모든 시간 단계에 대해 공유된 $\gamma, \beta$와 전체 $\mu_{\text{total}}, \sigma^2_{\text{total}}$ 사용 → 시간 특성 무시
- **TEBN(효율적 혼합)**: 공유된 $\gamma, \beta$와 시간 학습 가중치 $p[t]$를 결합 → 매개변수 효율성과 시간 표현력 균형[1]

***

### 3. 이론적 분석: 최적화 경관 평탄화

#### 3.1 정리 1: 그래디언트 노름 제약

시간 의존적 입력 신호 $x_i[t]$에 대해, TEBN 네트워크의 그래디언트와 non-TEBN 네트워크의 관계:[1]

$$\|\nabla_{x_i[t]} L\| \leq \frac{\hat{\gamma}_i[t]}{\sigma_i} \|\nabla_{x_i[t]} \tilde{L}\|$$

여기서 $L$은 TEBN 손실, $\tilde{L}$은 non-TEBN 손실이고, L2 노름은 미니배치 차원에서 계산됩니다.

**의의**: 이 부등식은 TEBN이 전통적 BN과 동일한 역할을 수행하며 시간 의존적 $\hat{\gamma}_i[t]$의 존재로 인해 **비정적 Lipschitz 상수**를 제공합니다. 이는 최적화 경관을 더 효과적으로 평탄화합니다.

#### 3.2 정리 2: 시간 단계 간 그래디언트 안정화

인접한 시간 단계 $t-1$과 $t$ 사이의 그래디언트 관계:[1]

$$\|\nabla_{\bar{x}^{l-1}_{i}[t-1]} L\| \leq \frac{p^l[t-1]}{p^l[t]} \tau \sqrt{k} (1 + \theta h_{\max}) \|\nabla_{\bar{x}^{l-1}_{i}[t]} L\| + p^l[t-1] h_{\max} \|\nabla_{o_i^l[t-1]} L\|$$

여기서 $k$는 미니배치 크기, $h_{\max}$는 대체 그래디언트(surrogate gradient)의 최댓값입니다.

**의의**: 모든 $p[t]$를 1로 설정하면 상수 Lipschitz 계수 $\tau\sqrt{k}(1 + \theta h_{\max})$를 얻으나, 학습 가능한 $p[t]$는 이를 동적으로 조정하여 **그래디언트 폭발과 소실을 완화**합니다.

***

### 4. 성능 향상 및 실험 결과

#### 4.1 벤치마크 성능

**표 2: 정규화 방법 간 비교**[1]

| 데이터셋 | 모델 | 방법 | 시간 단계 | 정확도(%) |
|---------|------|------|---------|----------|
| CIFAR-10 | ResNet-19 | tdBN | 6 | 93.16 |
| CIFAR-10 | ResNet-19 | TEBN | **2** | **94.57** |
| CIFAR-100 | VGG-11 | BNTT | 50 | 66.60 |
| CIFAR-100 | VGG-11 | TEBN | **4** | **74.37** |
| DVS-CIFAR10 | 7-layer CNN | BNTT | 20 | 63.20 |
| DVS-CIFAR10 | 7-layer CNN | TEBN | **10** | **75.10** |

TEBN은 더 **적은 시간 단계로 더 높은 정확도**를 달성합니다.

#### 4.2 최신 방법과의 비교

**표 3: 최첨단 방법 비교**[1]

- **CIFAR-100 (ResNet-19, T=6)**: TET 대비 4.04% 향상 (74.72% → 78.76*)
- **DVS-CIFAR10 (VGGSNN, T=10)**: TET 대비 1.73% 향상 (83.17% → 84.90%)
- **ImageNet (SEW ResNet-34, T=4)**: 기존 기록 68.00%를 68.28%로 개선

#### 4.3 분포 시각화

Figure 2-3의 분포 분석: TEBN은 다른 정규화 방법 대비 시간 단계 전체에서 **더 일관되고 평탄한 정규분포**를 생성하며, 신경모픽 데이터(DVS-CIFAR10)에서 서로 다른 시간 단계의 분포를 효과적으로 통일합니다.[1]

***

### 5. 일반화 성능 향상 가능성

TEBN의 일반화 성능 개선은 여러 메커니즘을 통해 달성됩니다:

#### 5.1 하이퍼파라미터 강건성

**Figure 4 결과**: 막 시간 상수 $\tau \in [0.1, 1.0]$에서:[1]
- **Default BN**: 92.2% ~ 93.6% (변동폭: 1.4%)
- **BNTT**: 불안정한 변동
- **tdBN**: 중간 수준 변동
- **TEBN**: 92.2% ~ 92.8% (변동폭: 0.6%) - **최고의 안정성**

학습 가능한 $p[t]$가 $\tau$ 변화에 **자동으로 적응**하여 강건성을 제공합니다.

#### 5.2 학습 가능한 입력 저항으로의 해석

TEBN의 $p[t]$는 LIF 뉴런의 학습 가능한 입력 저항 $R$로 해석할 수 있습니다. 이는:[1]

- 막 시간 상수 τ와 유사하게 메모리 효과 조절
- 각 시간 단계에서 **최적의 기억/입력 비율** 자동 학습
- **시간 변화 특성**으로 인한 더 나은 표현 능력

따라서 TEBN은 이전의 고정 τ 접근 방식 대비 더 유연한 일반화를 가능하게 합니다.

#### 5.3 최적화 경관 효과

이론 정리 1-2에서 입증된 Lipschitz 상수 제어는:

- **더 안정적인 그래디언트 흐름**: 그래디언트 폭발/소실 완화
- **평탄한 손실 경관**: 보다 나은 일반화 특성
- **시간 차원 별 최적 조정**: 각 단계의 동역학을 고려한 훈련

이러한 특징들이 결합되어 정적(CIFAR) 및 신경모픽(DVS-CIFAR10) 데이터셋 모두에서 일반화 성능을 향상시킵니다.

***

### 6. 모델의 한계

#### 6.1 현재 논문의 한계

1. **고정된 임계값(threshold)**: Theorem 2에서 $\theta$가 고정 상수로 취급되며, 학습 가능한 임계값 구현 시 이론과의 괴리 발생 가능

2. **매개변수 수렴성**: $p[t]$의 초기화 및 수렴 속도에 대한 명시적 분석 부재

3. **대규모 네트워크 제한**: ImageNet 실험은 제한적이며, 매우 깊은 네트워크(50+ 계층)에서의 효과는 미검증

4. **계산 오버헤드**: 각 시간 단계마다 추가 가중치 계산으로 인한 약간의 오버헤드 (정량화 미포함)

#### 6.2 SNNs 전반의 근본적 한계

1. **훈련-추론 격차**: 훈련 시 시간 展開(time unfolding)와 실제 추론 간의 근본적 차이

2. **시간 단계 선택**: 최적의 T 값 선택에 대한 체계적 지침 부재

3. **신경생물학적 타당성**: 대체 그래디언트 사용으로 인한 생물학적 신뢰성 감소

***

### 7. 최신 연구 기반 미래 영향 및 고려 사항

#### 7.1 TEBN의 후속 영향

**시간적 정규화 연장 연구:** 최근 발표된 논문들은 TEBN의 개념을 확장하고 있습니다. **Temporal Regularization Training (TRT)**는 초반 시간 단계에 더 강한 정규화 제약을 적용하여 과적합을 완화하며, TEBN의 시간 차원 조정 개념을 정규화 강도에 통합했습니다. 이를 통해 DVS-CIFAR10에서 T=4일 때 83.20%의 정확도를 달성하였고, 이는 TEBN 기반 시간적 조정의 효과를 재증명합니다.[2]

**앙상블 학습 관점의 재해석:** 최근 연구에서는 SNN을 시간 단계별 서브네트워크의 앙상블로 재해석하면서, TEBN의 시간 단계별 가중치 조정이 각 서브네트워크의 일관성을 강화한다고 분석했습니다. 이는 TEBN의 일반화 성능 향상이 단순 정규화 효과를 넘어 **구조적 일관성 강화**에 있음을 시사합니다.[2]

#### 7.2 하이브리드 및 멀티스케일 방향

**다중 스케일 공간-시간 상호작용 학습:** 2025년 최신 연구는 TEBN의 시간적 정규화와 **멀티스케일 어텐션**을 결합하여 더 정교한 시간-공간 특징 추출을 제안했습니다. 이는 TEBN의 단순 가중치 조정에서 한 단계 진화하여, 다양한 시간 스케일에서의 정보를 동시에 처리합니다.[3]

#### 7.3 신경모픽 하드웨어 응용

**시장 전망 및 하드웨어 최적화:** SNN 뉴로모픽 칩 시장이 2025년 214만 달러에서 2031년 661만 달러로 **연 63.2% 성장**할 것으로 예측되고 있습니다. TEBN과 같은 효율적인 훈련 방법은 이러한 시장 성장의 핵심 기술로 작용하며, 특히 **엣지 AI, 자율주행, 의료 영상** 등 저지연/저전력 응용에서 필수적입니다.[4]

#### 7.4 미래 연구 시 고려 사항

**1. 생물학적 메커니즘의 적용**

논문에서 언급한 **단기 가소성(Short-Term Plasticity, STP)**을 TEBN과 통합하는 연구 필요:[1]

$$p[t] = \text{STP}(\text{input history}, \tau_{\text{STP}})$$

STP는 시냅스가 반복 자극에 반응하여 전달 효율을 시간에 따라 변화시키는 메커니즘으로, TEBN의 시간 가중치 조정과 생물학적 타당성을 동시에 강화할 수 있습니다.

**2. 학습 가능한 신경 매개변수의 통합**

막 시간 상수 τ의 학습과 TEBN $p[t]$의 통합:[5]

$$\text{loss} = L_{\text{CE}} + \lambda_1 \|\tau - \tau_0\|^2 + \lambda_2 \|p[t] - 1\|^2$$

이를 통해 TEBN 가중치와 신경 동역학의 **동시 최적화**로 더욱 강건한 훈련이 가능해집니다.

**3. 낮은 레이턴시 추론 최적화**

TEBN의 T(시간 단계) 축소 효과를 극대화하기 위해 **적응형 시간 단계 결정** 메커니즘 개발:

$$T_{\text{adaptive}} = \arg\min_T \text{accuracy}(T) \text{ s.t. } \text{latency}(T) < \text{threshold}$$

신경모픽 하드웨어 상에서 실시간 처리의 필요성이 높아짐에 따라, TEBN의 효율성 이점을 극대화하는 추론 최적화가 중요합니다.

**4. 다중 작업 및 도메인 적응**

TEBN의 하이퍼파라미터 강건성을 활용한 **전이 학습(transfer learning)** 연구:

$$p^{\text{target}}[t] = p^{\text{source}}[t] + \Delta p[t]$$

서로 다른 작업 간 $p[t]$의 전이 가능성 검증은 TEBN의 일반화 능력을 한 단계 상향할 수 있습니다.

**5. 극한 에너지 효율 조건에서의 성능**

저전력 디바이스(IoT, 웨어러블)에서의 실제 하드웨어 배포를 위해:

- **메모리 효율성**: $p[t]$ 저장 공간 최소화 (fixed-point quantization)
- **연산 병렬화**: GPU/SNN 가속기에서의 TEBN 최적화 구현
- **온라인 학습 적용**: 배포 후 실시간 $p[t]$ 미세 조정

***

### 결론

**Temporal Effective Batch Normalization**은 스파이킹 신경망의 훈련에 대한 **패러다임 전환**을 제시합니다. 시간 차원을 명시적으로 모델링하면서도 매개변수 효율성을 유지한 설계는, SNN이 ANNs와의 성능 격차를 좁히는 동시에 저전력 효율성을 활용할 수 있는 경로를 제공합니다.[1]

특히 이론적 분석을 통한 **최적화 경관 평탄화**와 **그래디언트 안정화**의 입증은 TEBN이 단순한 휴리스틱이 아닌 원리 기반의 솔루션임을 보여줍니다. 앞으로 생물학적 메커니즘의 통합, 신경모픽 하드웨어 최적화, 그리고 극한 저전력 환경에서의 응용이 진행된다면, **제3세대 신경망의 실질적 상용화**를 견인하는 핵심 기술로 자리잡을 것으로 예상됩니다.

***

### 참고 자료 인덱스

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0613c032-cd39-4ed3-b3d3-6f844b30706f/NeurIPS-2022-temporal-effective-batch-normalization-in-spiking-neural-networks-Paper-Conference.pdf)
[2](https://www.themoonlight.io/ko/review/rethinking-spiking-neural-networks-from-an-ensemble-learning-perspective)
[3](https://www.themoonlight.io/ko/review/advancing-spiking-neural-networks-towards-multiscale-spatiotemporal-interaction-learning)
[4](https://contents.premium.naver.com/qyresearch/insight/contents/251002122725545wa)
[5](https://inforience.net/2024/10/26/spiking_neural_network_story/)
[6](http://arxiv.org/pdf/2406.01072.pdf)
[7](https://pmc.ncbi.nlm.nih.gov/articles/PMC10452895/)
[8](http://arxiv.org/pdf/2406.02923.pdf)
[9](https://www.frontiersin.org/articles/10.3389/fnins.2023.1229951/pdf)
[10](http://arxiv.org/pdf/2406.12726.pdf)
[11](https://www.frontiersin.org/articles/10.3389/fnins.2023.1261543/pdf?isPublishedV2=False)
[12](http://arxiv.org/pdf/2501.14484.pdf)
[13](https://arxiv.org/html/2210.06836v4)
[14](https://rupijun.tistory.com/entry/SNN-Spiking-Neural-Networks-%EC%83%9D%EB%AC%BC%ED%95%99%EC%A0%81-%EB%87%8C%EB%A5%BC-%EB%AA%A8%EB%B0%A9%ED%95%9C-%EC%A0%9C3%EC%84%B8%EB%8C%80-%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5-%EA%B8%B0%EC%88%A0)
[15](https://wikidocs.net/300499)
[16](https://donghunkang.tistory.com/129)
[17](https://ettrends.etri.re.kr/ettrends/183/0905183007/35-3_76-84.pdf)
[18](https://www.themoonlight.io/ko/review/enhancing-generalization-of-spiking-neural-networks-through-temporal-regularization)
[19](https://liner.com/ko/review/rethinking-spiking-neural-networks-from-an-ensemble-learning-perspective)
