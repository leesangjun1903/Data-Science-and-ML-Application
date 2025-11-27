# Online Training Through Time for Spiking Neural Networks

### 1. 핵심 주장과 주요 기여

"Online Training Through Time (OTTT) for Spiking Neural Networks" 논문은 스파이킹 신경망(SNN)의 훈련에 있어 메모리 효율성, 이론적 명확성, 생물학적 타당성을 동시에 달성하는 혁신적인 방법론을 제시합니다.[1]

**주요 기여:**

논문은 네 가지 핵심 기여를 제시합니다. 첫째, OTTT는 시간을 통한 전방향 학습(forward-in-time learning)을 가능하게 하면서 훈련 메모리를 시간 단계 수에 무관한 상수 비용으로 유지합니다. 이는 기존 BPTT(Backpropagation Through Time)의 선형적 메모리 증가 문제를 근본적으로 해결합니다. 둘째, OTTT의 그래디언트와 스파이크 표현(spike representation) 기반 방법의 그래디언트 간의 이론적 연결고리를 확립하고, 피드포워드 및 순환 조건 모두에서 하강 보장(descent guarantee)을 증명합니다. 셋째, OTTT가 생물학적으로 타당한 3-요소 헤비안 학습 규칙(three-factor Hebbian learning rule) 형태이며, 이를 통해 BPTT with SG, 스파이크 표현 기반 방법, 생물학적 학습 규칙 간의 처음으로 연결을 구축합니다. 넷째, CIFAR-10, CIFAR-100, ImageNet, CIFAR10-DVS 등 대규모 정적 및 신경형태 데이터셋에서 우수한 성능을 달성합니다.[1]

### 2. 해결하는 문제와 제안 방법

#### 문제점 분석

기존 SNN 훈련 방법들은 두 가지 주류 접근법을 가지고 있습니다.[1]

첫 번째는 BPTT with SG(Surrogate Gradients)로, 매우 낮은 지연시간(4-6 시간 단계)으로 ImageNet 같은 대규모 데이터셋에서 우수한 성능을 달성합니다. 그러나 훈련 중 시간 단계 수에 비례하는 막대한 메모리 비용이 발생하며, 근사 그래디언트 사용으로 인한 이론적 명확성 부족, 그리고 생물학적 온라인 학습 규칙과의 불일치 문제가 있습니다.

두 번째는 스파이크 표현 기반 방법(예: 가중된 발화율)으로, 폐쇄형 ANN과 유사한 매핑을 구축하여 명확한 최적화 방향을 제공합니다. 하지만 높은 지연시간(수십 개의 시간 단계 필요)으로 에너지 소비가 증가하며, 온라인 학습 특성을 부족합니다.

#### OTTT의 핵심 혁신

논문은 **시간적 의존성 분리(temporal dependency decoupling)**를 핵심 아이디어로 제시합니다. BPTT의 그래디언트 계산 식을 분석하면:[1]

$$\frac{\partial L}{\partial W^l} = \sum_{t=1}^{T} \frac{\partial L}{\partial s^{l+1}[t]} \frac{\partial s^{l+1}[t]}{\partial u^{l+1}[t]} \left[ \frac{\partial u^{l+1}[t]}{\partial W^l} + \sum_{\tau < t} \left( \frac{\partial u^{l+1}[i+1]}{\partial u^{l+1}[i]} + \frac{\partial u^{l+1}[i+1]}{\partial s^{l+1}[i]} \frac{\partial s^{l+1}[i]}{\partial u^{l+1}[i]} \right) \frac{\partial u^{l+1}[\tau]}{\partial W^l} \right]$$

이 식에서 시간적 의존성은 Heaviside 함수의 미분이 거의 0이므로, $$\frac{\partial u^{l+1}[i+1]}{\partial u^{l+1}[i]} = \lambda I$$만 남게 됩니다.[1]

**사전 활성화 추적(presynaptic activity tracking)**을 도입하여:

$$\hat{a}^l[t] = \sum_{\tau \leq t} \lambda^{t-\tau} s^l[\tau]$$

이를 반복적으로 업데이트하면: $$\hat{a}^l[t+1] = \lambda \hat{a}^l[t] + s^l[t+1]$$[1]

각 시간 단계에서 독립적으로 그래디언트를 계산할 수 있게 됩니다:

$$\nabla_{W^l}L[t] = g^{u^{l+1}}[t] \hat{a}^l[t]^{\top}$$

여기서 $$g^{u^{l+1}}[t] = \left( \frac{\partial L[t]}{\partial s^{N}[t]} \prod_{i=N-1}^{l+1} \frac{\partial s^{i+1}[t]}{\partial s^{i}[t]} \frac{\partial s^{l+1}[t]}{\partial u^{l+1}[t]} \right)^{\top}$$[1]

**순간적 손실(Instantaneous Loss)** 사용:

$$L[t] = \frac{1}{T}L(s^N[t], y), \quad L := \sum_{t=1}^{T} L[t]$$[1]

이를 통해 계산이 시간을 통해 앞으로 진행되며, 상수 메모리만 필요합니다.

### 3. 이론적 분석: 스파이크 표현 기반 방법과의 연결

#### 피드포워드 네트워크

논문은 다음을 증명합니다. 가중된 발화율이 $$a[t] = \frac{\sum_{\tau=1}^{t} \lambda^{t-\tau} s[\tau]}{\sum_{\tau=1}^{t} \lambda^{t-\tau}}$$로 정의되고, 수렴한다면:[1]

$$a^{l+1}[T] \approx \sigma\left(\frac{1}{V_{th}}(W^l a^l[T] + b^{l+1})\right)$$

스파이크 표현 기반 방법의 그래디언트는:

$$(\nabla_{W^l}L_{sr})_{sr} = \sum_{t=1}^{T} \left[ \frac{1}{T} \frac{1}{\lambda^{T-t}} \frac{\partial L_{sr}}{\partial s^N[t]} \prod_{i=N-1}^{l+1} \frac{\partial a^{i+1}[T]}{\partial a^{i}[T]} \odot d^{l+1}[T] \right] \hat{a}^l[T]^{\top}$$

여기서 $$d^{l+1}[T] = \sigma'\left(\frac{1}{V_{th}}(W^l a^l[T] + b^{l+1})\right)$$이고, ⊙는 요소별 곱셈입니다.[1]

**정리 1 (피드포워드):** 가정 1이 성립하고, $$V_{th} = 1$$이며, 오차 $$\epsilon^l[t] = a^l[t] - a^l[T]$$가 다음 조건을 만족한다면:

$$\left\| \sum_{t=1}^{T} \hat{g}^{u^{l+1}}[t] \epsilon^l[t]^{\top} \right\| < \left\| \sum_{t=1}^{T} \hat{g}^{u^{l+1}}[t] a^l[T]^{\top} \right\| - \left\| \sum_{t=1}^{T} \frac{\lambda^t(1-\lambda^{T-t})}{1-\lambda^T} \hat{g}^{u^{l+1}}[t] a^l[t]^{\top} \right\|$$

$$(\nabla_{W^l}L_{sr})_{sr} \neq 0$$일 때, 다음이 성립합니다:[1]

$$\langle \nabla_{W^l}L, (\nabla_{W^l}L_{sr})_{sr} \rangle > 0$$

이는 OTTT의 그래디언트가 스파이크 표현 기반 방법과 유사한 하강 방향을 제공함을 의미합니다.

#### 순환 네트워크

순환 조건에서 가중된 발화율은 고정점 방정식을 따릅니다:

$$a^* = \sigma\left(\frac{1}{V_{th}}(W a^* + F x^* + b)\right)$$

**정리 2 (순환):** 가정 1과 다음 조건들:
- $$\| J_{f_\theta|a[T]} \| \leq \eta < \frac{\sigma_{min}^2}{\sigma_{max}^2}$$
- 오차 $$\epsilon^1[t] = a[t] - a[T]$$, $$\epsilon^0[t] = x[t] - x[T]$$가 충분히 작다

이 경우, $$(\nabla_\theta L_{sr})_{sr} \neq 0$$일 때:[1]

$$\langle \nabla_\theta L, (\nabla_\theta L_{sr})_{sr} \rangle > 0$$

여기서 $$\theta \in \{W, F, b\}$$는 네트워크 매개변수입니다.

### 4. 3-요소 헤비안 학습과 생물학적 타당성

**방정식 7**에서 보듯이, OTTT의 순간 그래디언트는:[1]

$$\nabla_{W_{i,j}}L[t] = \hat{a}_i[t] f(u_j[t]) \delta_j[t]$$

여기서:
- $$\hat{a}_i[t]$$: 사전 시냅스 활성도 (presynaptic activity)
- $$f(u_j[t])$$: 대용 미분 함수 (surrogate derivative)
- $$\delta_j[t]$$: 전역 조절 신호 (global modulator)

이는 생물학적 시스템에서 관찰되는 **3-요소 헤비안 학습(three-factor Hebbian learning)**입니다. 중요하게도, 이 구조는 오류 신호 전파의 지연을 허용하여 생물학적으로 더욱 타당합니다.[1]

### 5. 모델 구조 및 구현 세부 사항

#### 뉴런 모델

논문은 표준 Leaky Integrate-and-Fire (LIF) 모델을 사용합니다:[1]

$$u_i[t+1] = \lambda(u_i[t] - V_{th}s_i[t]) + \sum_j w_{ij}s_j[t] + b_i$$

$$s_i[t+1] = H(u_i[t+1] - V_{th})$$

여기서 $$\lambda < 1$$은 누수 항(leaky term), $$H(x)$$는 Heaviside 함수입니다.[1]

#### 구현 변형

**OTTTA (누적)**: 그래디언트를 T 시간 단계 동안 누적 후 매개변수 업데이트
**OTTTO (온라인)**: 각 시간 단계 후 즉시 매개변수 업데이트 (업데이트가 다음 계산에 미미한 영향을 미친다고 가정)[1]

#### 배치 정규화 대체

OTTT는 온라인 특성으로 인해 배치 정규화를 사용할 수 없으므로, **Scaled Weight Standardization (sWS)**를 채택합니다:[1]

$$\hat{W}_{i,j} = \gamma \cdot \frac{W_{i,j} - \mu_{W_{i,\cdot}}}{\sigma_{W_{i,\cdot}} \sqrt{N}}$$

### 6. 성능 향상 및 한계

#### 실험 결과

**표 1**에서 보듯이 OTTT는 기존 방법들을 능가합니다:[1]

| 데이터셋 | 방법 | 정확도 | 시간 단계 |
|---------|------|-------|---------|
| CIFAR-10 | ANN | 94.43% | N.A. |
| | OTTTO (ours) | 93.73% | 6 |
| | BPTT | 92.78% | 6 |
| CIFAR-100 | OTTTO (ours) | 71.11% | 6 |
| | BPTT | 69.15% | 6 |
| ImageNet | OTTTO (ours) | 64.16% | 6 |
| | BPTT (tdBN) | 63.72% | 6 |

#### 메모리 효율성

**그림 2**에서 OTTT는 시간 단계 수에 관계없이 상수 메모리를 유지하는 반면, BPTT는 선형적으로 증가합니다. 6 시간 단계에서 OTTT는 BPTT 대비 2-3배 메모리 감소를 달성합니다.[1]

#### 한계

1. **배치 정규화와의 비호환성**: sWS가 배치 정규화보다 낮은 성능을 보입니다. ANN과의 성능 격차가 CIFAR-10에서 약 0.7%, CIFAR-100에서 2.08%입니다.[1]

2. **가정의 제약성**: 정리 1, 2의 가정들은 충분히 수렴한 입력을 가정하며, 시간 변화가 큰 입력에서 검증이 필요합니다. DVS128-Gesture 결과는 일반화 가능성을 시사하지만, 이론적 분석이 명시적으로 다루지는 않습니다.[1]

3. **순환 구조의 제한**: 완전 순환 구조(예: Fashion-MNIST의 400개 순환 뉴런)에서 BPTT가 약간 나은 성능을 보입니다(, 표 6).[1]

4. **발화율 통계**: 모델이 첫 번째 층에서 높은 발화율(약 0.35), 후기 층에서 낮은 발화율(약 0.1)을 보여, 에너지 효율 최적화 여지가 있습니다.[1]

### 7. 일반화 성능 향상 메커니즘

#### 신호 전파 특성

논문에서 명시적으로 다루지 않지만, OTTT의 일반화 향상은 여러 메커니즘에서 비롯됩니다:

1. **순간적 손실 기반 정규화**: 각 시간 단계에서 독립적 손실을 계산하면 자연스러운 정규화 효과가 발생합니다.[1]

2. **낮은 메모리 비용**: 배치 크기 1로 훈련 가능성이 증명되며(표 4), 진정한 온라인 학습을 통한 향상된 일반화를 시사합니다.[1]

3. **발화율 분포의 차이**: BPTT 대비 OTTT가 후기 층에서 낮은 발화율을 유지하여, 더 희소한 표현을 학습합니다. 이는 일반화 성능 향상과 연관됩니다.[1]

#### 추론 시간 단계 효과

**그림 3**에서 6 시간 단계로 훈련된 모델이 더 많은 추론 시간 단계(10-12)에서 향상된 성능을 보여, 정보 누적의 여유 공간을 시사합니다.[1]

### 8. 최신 연구 동향 및 향후 고려사항

#### 2023-2025년 연구 진전

최근 SNN 연구 분야는 다음과 같은 방향으로 진전되고 있습니다:[2][3][4][5]

1. **메모리 효율성의 극단화**: Lei et al. (2023)은 역가능 SNN 노드를 통해 메모리 사용을 58.65배 감소시키고, 훈련 시간을 23.8% 단축하는 연구를 수행했습니다.[2]

2. **병렬 훈련 기법**: 2025년 ICML에서 발표된 Fixed-Point Parallel Training (FPT)은 시간 단계 처리를 순차적에서 병렬 방식으로 전환하여, 정확도 손실 없이 훈련 시간을 대폭 단축합니다.[3]

3. **일반화 성능 향상**: Zhang et al. (2025)의 Temporal Regularization Training (TRT)은 시간 의존적 정규화 메커니즘으로 오버피팅을 완화하고, 특히 신경형태 데이터셋에서 일반화를 개선합니다.[4]

4. **지연 학습**: Nature Communications (2025)에 발표된 연구는 EventProp를 기반으로 정확한 그래디언트 계산을 통해 시냅스 지연을 학습하는 방법을 제시하여, 네트워크 용량을 유의미하게 증가시킵니다.[5]

5. **로봇 제어 확장**: Huebotter et al.은 end-to-end 예측 제어 SNN을 통해 고차원 로봇 팔 조작 과제를 성공적으로 수행하여, SNN의 실제 응용 범위를 확대하고 있습니다.[6]

#### 향후 연구 고려사항

**이론적 측면:**
- OTTT의 가정 1 완화: 현재 surrogate 미분이 모든 시간 단계에서 일정하다고 가정하지만, 이를 완화하면 더 일반적인 이론이 가능합니다.
- 수렴 속도 분석: 정리 1, 2는 하강 방향만 보장하며, 수렴 속도에 대한 정량적 분석이 필요합니다.
- 비정상 입력에 대한 이론: 시간 변화가 큰 입력에 대한 엄밀한 이론적 분석이 필요합니다.

**실제 응용 측면:**
- 신경형태 하드웨어 배포: Loihi, SpiNNaker와 같은 실제 신경형태 칩에서의 온칩 학습 구현
- 온라인 지속 학습: OTTT의 온라인 특성을 활용한 지속 학습(continual learning) 프레임워크 개발
- 하이브리드 접근법: OTTT와 ANN-SNN 변환의 결합으로 더욱 강력한 성능 달성

**아키텍처 혁신:**
- Spiking Transformers: Vision Transformer와 같은 최신 아키텍처를 SNN으로 확장하되, OTTT의 메모리 효율성 유지
- 적응형 시간 단계: 입력 특성에 따라 동적으로 시간 단계를 조정하는 메커니즘

**정규화 기법 개선:**
- sWS 최적화: 배치 정규화 수준의 성능 달성을 위한 더 나은 정규화 방법 개발
- 시간 차원 정규화: OTTT와 호환되는 시간 차원 정규화 기법 개발

**신경형태 데이터 활용:**
- 대규모 신경형태 데이터셋 구축: 현재 제한된 신경형태 데이터셋의 확대
- 도메인 적응: 정적 이미지와 신경형태 데이터 간의 전이 학습

***

## 결론

OTTT는 **세 가지 핵심 문제를 동시에 해결**하는 획기적 방법론입니다. 첫째, 상수 메모리 비용으로 BPTT의 메모리 폭증 문제를 해결하고, 둘째, 스파이크 표현 기반 방법과의 이론적 연결을 통해 최적화의 명확성을 제공하며, 셋째, 생물학적으로 타당한 3-요소 헤비안 학습 형태로 신경형태 칩에서의 온칩 학습 경로를 제시합니다.

다만 현재 배치 정규화와의 비호환성, 이론적 가정의 제약성, 완전 순환 구조에서의 제한이 존재합니다. 최신 연구는 메모리 극단화, 병렬 훈련, 시간적 정규화, 지연 학습 등으로 OTTT의 한계를 보완하는 방향으로 진행 중입니다. 향후 신경형태 하드웨어 배포, Spiking Transformers, 온라인 지속 학습 등의 분야에서 **뇌-영감 AI의 실질적 구현**을 가능하게 할 핵심 기술이 될 것으로 기대됩니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e77bbd35-2755-4bd0-9513-1853a81464fd/2210.04195v2.pdf)
[2](https://www.osti.gov/servlets/purl/2462994/)
[3](https://jurnal.polgan.ac.id/index.php/sinkron/article/view/15341)
[4](https://www.mdpi.com/2076-3417/15/13/7573)
[5](https://arxiv.org/abs/2411.01663)
[6](https://dl.acm.org/doi/10.1145/3687273.3687295)
[7](https://ojs.bonviewpress.com/index.php/AIA/article/view/5930)
[8](https://www.elibrary.ru/item.asp?id=82778008)
[9](https://dergipark.org.tr/en/doi/10.12995/bilig.8402)
[10](https://photonics.pl/PLP/index.php/letters/article/view/17-5)
[11](https://www.ijltemas.in/submission/index.php/online/article/view/2230)
[12](https://arxiv.org/pdf/2401.10843.pdf)
[13](https://pmc.ncbi.nlm.nih.gov/articles/PMC10452895/)
[14](https://arxiv.org/pdf/2409.02111.pdf)
[15](http://arxiv.org/pdf/2410.07547.pdf)
[16](https://arxiv.org/pdf/2109.12894.pdf)
[17](https://www.frontiersin.org/articles/10.3389/fnins.2023.1229951/pdf)
[18](https://arxiv.org/pdf/2412.13610.pdf)
[19](http://arxiv.org/pdf/2408.00280.pdf)
[20](https://icml.cc/virtual/2025/poster/45776)
[21](https://arxiv.org/html/2209.01610v3)
[22](https://pubs.aip.org/aip/apm/article/12/10/109201/3317314/Roadmap-to-neuromorphic-computing-with-emerging)
[23](https://www.nature.com/articles/s41467-025-65394-8)
[24](https://arxiv.org/html/2506.19256v3)
[25](https://arxiv.org/html/2507.18139v1)
[26](https://openreview.net/forum?id=yqIJoALgdD)
[27](https://www.engineering.org.cn/sscae/EN/10.15302/J-SSCAE-2023.06.011)
[28](https://www.frontiersin.org/research-topics/35440/digital-neuromorphic-system-with-online-learning-algorithms-implementation-and-large-scale-design/magazine)
[29](https://arxiv.org/html/2509.05356v2)
