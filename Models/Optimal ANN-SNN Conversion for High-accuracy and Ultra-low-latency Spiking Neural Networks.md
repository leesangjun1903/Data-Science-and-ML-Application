# Optimal ANN-SNN Conversion for High-accuracy and Ultra-low-latency Spiking Neural Networks

## 1. 핵심 주장 및 주요 기여 요약

이 논문은 **ANN-SNN 변환에서 발생하는 변환 오류를 이론적으로 분석하고, 초저지연(ultra-low latency)에서도 고정확도를 달성할 수 있는 새로운 변환 기법**을 제시한다. 핵심적으로 다음 세 가지 기여를 제공한다:[1]

**첫째**, ANN-SNN 변환 오류를 **클리핑 오류(clipping error)**, **양자화 오류(quantization error)**, **불균등 오류(unevenness error)**의 세 가지 유형으로 세분화하여 분석했다. 특히 기존 연구에서 간과되었던 불균등 오류가 예상보다 많거나 적은 스파이크를 유발할 수 있음을 밝혔다.[1]

**둘째**, ReLU 활성화 함수를 대체하는 **Quantization Clip-Floor-Shift (QCFS) 활성화 함수**를 제안했다. 이 함수는 SNN의 실제 활성화 함수를 더 정확하게 근사하며, 기대 변환 오류가 0임을 수학적으로 증명했다.[1]

**셋째**, CIFAR-10에서 2 time-step으로 91.18%, 4 time-step으로 93.96%의 정확도를 달성하며, 기존 ANN-SNN 변환 및 직접 훈련 방법 대비 최고 성능을 기록했다. 이는 **초저지연 고성능 ANN-SNN 변환을 최초로 탐구**한 사례이다.[1]

***

## 2. 문제 정의, 제안 방법, 모델 구조 및 성능

### 2.1 해결하고자 하는 문제

ANN-SNN 변환의 핵심 딜레마는 **잔여 전위(remaining potential)**의 존재로, 소수의 time-step에서 이를 제거하기 어렵다는 점이다. 기존 방법들은 ReLU 활성화 함수와 Integrate-and-Fire (IF) 모델의 발화율(firing rate) 간의 동등성에 기반하지만, 이를 정확하게 매칭하려면 수십에서 수백 time-step이 필요하다. 그 결과 SNN의 실질적인 에너지 효율 및 저지연 장점이 상쇄된다.[1]

변환 오류는 다음 식으로 정의된다:

$$Err^l = \phi^l(T) - a^l = z^l - \frac{v^l(T) - v^l(0)}{T} - h(z^l)$$

여기서 $$\phi^l(T)$$는 SNN의 평균 시냅스 후 전위, $$a^l$$는 ANN의 출력, $$z^l = W^l \phi^{l-1}(T)$$는 가중 입력, $$h(\cdot)$$는 ReLU 함수이다.[1]

### 2.2 변환 오류의 세 가지 원인

| 오류 유형 | 설명 | 수식적 특성 |
|-----------|------|-------------|
| **클리핑 오류** | SNN 출력 범위 $$[0, \theta^l]$$이 ANN 범위 $$[0, a^l_{max}]$$보다 좁음 | $$\lambda^l$$과 $$a^l_{max}$$ 간 불일치[1] |
| **양자화 오류** | 이산적 스파이크로 인해 $$\phi^l(T)$$가 $$\theta^l/T$$ 해상도로 양자화됨 | Floor 함수에 의한 반올림 손실[1] |
| **불균등 오류** | 입력 스파이크의 도착 타이밍 변화로 예상과 다른 발화율 발생 | 시간적 비동기성으로 인한 오차[1] |

### 2.3 제안 방법: Quantization Clip-Floor-Shift (QCFS) 활성화 함수

**단계 1: Quantization Clip-Floor 함수**

기존 ReLU를 다음의 clip-floor 함수로 대체한다:

$$a^l = \bar{h}(z^l) = \lambda^l \cdot \text{clip}\left(\frac{1}{L}\left\lfloor\frac{z^l L}{\lambda^l}\right\rfloor, 0, 1\right)$$

여기서 $$L$$은 양자화 단계 수, $$\lambda^l$$은 학습 가능한 임계값이다.[1]

**Theorem 1**: $$T = L$$, $$\theta^l = \lambda^l$$, $$v^l(0) = 0$$일 때, 추정 변환 오류 $$\widehat{Err}^l = 0$$이다.[1]

**단계 2: Shift 항 추가**

$$T \neq L$$인 경우에도 오류를 최소화하기 위해 shift 항 $$\phi$$를 도입한다:

$$a^l = \tilde{h}(z^l) = \lambda^l \cdot \text{clip}\left(\frac{1}{L}\left\lfloor\frac{z^l L}{\lambda^l} + \phi\right\rfloor, 0, 1\right)$$

**Theorem 2**: $$\theta^l = \lambda^l$$, $$v^l(0) = \theta^l \phi$$일 때, shift 항 $$\phi = 0.5$$에서 임의의 $$T$$와 $$L$$에 대해 변환 오류의 기댓값이 0이다:

$$\forall T, L \quad \mathbb{E}_{z}\left[\widehat{Err}^l \middle| \phi = \frac{1}{2}\right] = 0$$

이 결과는 **단일 훈련된 ANN**으로 **다양한 time-step에서 고성능 SNN**을 얻을 수 있음을 의미한다.[1]

### 2.4 모델 구조 및 학습 알고리즘

**네트워크 수정 사항**:
- MaxPooling → AveragePooling 대체
- ReLU → QCFS 활성화 함수 대체
- SNN 변환 시 $$\theta^l = \lambda^l$$, $$v^l(0) = \theta^l/2$$ 설정[1]

**Gradient 계산**: Floor 함수의 미분을 위해 **Straight-Through Estimator (STE)**를 사용한다:

$$\frac{\partial \tilde{h}_i(z^l)}{\partial z^l_i} = \begin{cases} 1, & -\frac{\lambda^l}{2L} < z^l_i < \lambda^l - \frac{\lambda^l}{2L} \\ 0, & \text{otherwise} \end{cases}$$

### 2.5 실험 결과 및 성능 향상

| 데이터셋 | 아키텍처 | 방법 | T=2 | T=4 | T=8 | T=32 |
|----------|----------|------|-----|-----|-----|------|
| CIFAR-10 | VGG-16 | 본 논문 | **91.18%** | **93.96%** | 94.95% | **95.54%**[1] |
| CIFAR-10 | VGG-16 | SNNC-AP | - | - | - | 93.71%[1] |
| ImageNet | ResNet-34 | 본 논문 | - | - | - | **69.37%**[1] |
| ImageNet | ResNet-34 | SNNC-AP | - | - | - | 64.54%[1] |

**핵심 성과**: CIFAR-10에서 4 time-step으로 93.96% 달성은 SNNC-AP가 32 time-step으로 달성한 93.71%를 초과하며, **8배 빠른 추론**을 가능하게 한다.[1]

### 2.6 한계점

1. **불균등 오류의 완전 제거 불가**: $$L = T$$인 경우에도 ANN과 SNN 정확도 간 갭이 존재한다.[1]
2. **T=1 time-step에서 고성능 달성 어려움**: 극단적 저지연에서는 여전히 성능 저하가 발생한다.[1]
3. **양자화 단계 $$L$$의 선택**: 저지연과 최대 정확도 간 trade-off가 존재하며, $$L=4$$ 또는 $$L=8$$이 권장된다.[1]

***

## 3. 일반화 성능 향상 가능성

### 3.1 다양한 Time-step에서의 일관된 성능

QCFS 활성화 함수의 핵심 장점은 **단일 훈련 ANN이 다양한 time-step에서 고성능을 유지**한다는 것이다. Theorem 2가 보장하듯, shift 항 $$\phi = 0.5$$는 $$T$$와 $$L$$의 불일치에도 기대 오류를 0으로 만든다.[1]

실험적으로, QCFS로 훈련된 SNN은 $$T$$가 증가함에 따라 정확도가 단조 증가하며, $$T \geq 16$$에서 소스 ANN과 동등한 성능에 도달한다. 반면 shift 항이 없는 clip-floor 함수는 $$T \neq L$$일 때 급격한 성능 저하를 보인다.[1]

### 3.2 대규모 데이터셋에서의 확장성

ImageNet 규모에서도 VGG-16과 ResNet-34로 검증되었으며, 이는 제안 방법이 **복잡한 실제 데이터셋에서도 일반화됨**을 시사한다.[1]

### 3.3 에너지 효율과의 균형

뉴로모픽 프로세서에서 SNN의 연산은 **Synaptic Operations (SOPs)**로 측정되며, 본 논문의 방법은 동일 정확도 달성 시 SNNC-AP 대비 **더 낮은 에너지 소비**를 보인다. 예를 들어, CIFAR-100에서 73.55% 정확도 달성 시 본 방법은 0.056mJ, SNNC-AP는 0.660mJ를 소비한다.[1]

***

## 4. 후속 연구에 미치는 영향 및 고려사항

### 4.1 최신 연구 동향과의 연계

이 논문(ICLR 2022)은 후속 연구에 중요한 기반을 제공했다:

**Spiking Large Language Models**: 2025년에 제안된 FAS(Fast ANN-SNN conversion) 방법은 LLM을 spiking LLM으로 변환하는 2단계 전략을 도입했으며, 이는 QCFS의 양자화 개념을 대규모 언어 모델로 확장한 것이다.[2]

**Spiking Transformers**: 2025년 연구에서 ANN-to-SNN 변환을 Vision Transformer에 적용하여 4 time-step으로 88.60% 정확도를 달성했으며, 이는 원래 Transformer 전력의 35%만 소비한다.[3]

**음성 분야 확장**: 2025년 연구는 3단계 하이브리드 SNN 미세조정 기법을 Wave-U-Net과 ConvTasNet에 적용하여 음성 향상 분야로 확장했다.[4]

**새로운 뉴런 모델**: 최근 연구들은 **음수 스파이크를 지원하는 뉴런 모델**과 **시간 의존적 IF 뉴런(tdIF)**을 제안하여 변환 성능을 더욱 개선했다.[5][6]

### 4.2 향후 연구 시 고려해야 할 점

**불균등 오류의 근본적 해결**: 현재 방법은 기대값 기준으로 오류를 최소화하지만, 분산(variance)을 줄이는 추가 연구가 필요하다.[5][1]

**모델 압축과의 결합**: Kundu et al.(2021)이 제안한 것처럼, 변환 기반 방법과 가지치기(pruning) 등 모델 압축 기법을 결합하면 정확도 손실 없이 뉴런 활동과 에너지 소비를 크게 줄일 수 있다.[1]

**새로운 아키텍처 지원**: 최근 연구는 ConvNext, MLP-Mixer, ResMLP 등 ReLU 이외의 비선형성을 사용하는 DNN 아키텍처로의 변환을 가능하게 했다. 이는 QCFS의 clip-floor 패러다임을 다양한 활성화 함수로 확장하는 방향을 제시한다.[7]

**하드웨어 공동 설계**: 2025년 연구는 알고리즘-하드웨어 공동 설계 프레임워크를 통해 Ternary-8-bit 하이브리드 가중치 양자화를 제안했으며, 이는 QCFS와 같은 양자화 기반 방법이 뉴로모픽 칩 설계와 긴밀히 연계될 필요성을 보여준다.[8]

**Post-Training Quantization**: 최근 접근법은 채널별 임계값(channel-wise thresholds)이 레이어별 임계값보다 변환 오류 감소에 더 효과적임을 이론적으로 분석했으며, 이는 QCFS의 학습 가능한 $$\lambda^l$$ 파라미터를 더욱 세밀화할 수 있는 방향을 제시한다.[9]

***

## 결론

이 논문은 ANN-SNN 변환의 근본적 한계를 세 가지 오류 유형으로 분석하고, QCFS 활성화 함수를 통해 기대 변환 오류가 0인 이론적 프레임워크를 확립했다. 4 time-step이라는 초저지연에서 기존 방법들을 능가하는 성능을 달성함으로써, SNN의 뉴로모픽 하드웨어 배치와 실용적 응용 가능성을 크게 확장했다. 후속 연구들은 이 프레임워크를 대규모 언어 모델, Vision Transformer, 음성 처리 등 다양한 영역으로 확장하고 있으며, 하드웨어 공동 설계와 새로운 뉴런 모델 개발을 통해 지속적인 발전이 이루어지고 있다.[2][3][4][1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3683be01-d520-4197-a2ff-3718d2048bc5/2303.04347v1.pdf)
[2](http://arxiv.org/pdf/2502.04405.pdf)
[3](https://arxiv.org/html/2502.21193)
[4](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1567347/full)
[5](https://www.ijcai.org/proceedings/2025/0719.pdf)
[6](https://arxiv.org/html/2508.20392v1)
[7](http://arxiv.org/pdf/2407.01645v1.pdf)
[8](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1665778/full)
[9](https://www.sciencedirect.com/science/article/abs/pii/S0893608025007129)
[10](https://journals.sagepub.com/doi/10.1177/17479541251333942)
[11](https://www.mdpi.com/2413-8851/9/11/488)
[12](https://revues.cirad.fr/index.php/BFT/article/view/37727)
[13](https://www.ahajournals.org/doi/10.1161/CIR.0000000000001303)
[14](https://www.sciltp.com/journals/gefr/2025/1/477)
[15](https://iopscience.iop.org/article/10.1149/MA2024-02483439mtgabs)
[16](https://scimatic.org/show_manuscript/6647)
[17](https://eia.feaa.ugal.ro/images/eia/2025_2/Suciu_et_al.pdf)
[18](https://wellcomeopenresearch.org/articles/9-133/v1)
[19](https://www.johs.org.uk/article/doi/10.54531/MSKP8376)
[20](https://pmc.ncbi.nlm.nih.gov/articles/PMC9597447/)
[21](https://arxiv.org/html/2411.17431v1)
[22](http://arxiv.org/pdf/2311.09266.pdf)
[23](https://arxiv.org/pdf/2304.09101.pdf)
[24](https://arxiv.org/pdf/2403.18388.pdf)
[25](https://openreview.net/forum?id=KjiNHPinrS)
[26](https://openaccess.thecvf.com/content/CVPR2024/papers/Shen_Are_Conventional_SNNs_Really_Efficient_A_Perspective_from_Network_Quantization_CVPR_2024_paper.pdf)
[27](https://arxiv.org/html/2506.01968v1)
[28](https://openaccess.thecvf.com/content/CVPR2025/papers/Bu_Inference-Scale_Complexity_in_ANN-SNN_Conversion_for_High-Performance_and_Low-Power_Applications_CVPR_2025_paper.pdf)
[29](https://pubs.aip.org/aip/apr/article/12/2/021309/3344844/Recent-advances-in-fluidic-neuromorphic-computing)
[30](https://www.sciencedirect.com/science/article/abs/pii/S0925231225020235)
[31](https://www.sciencedirect.com/science/article/abs/pii/S0893608024001680)
[32](https://arxiv.org/html/2507.15958v2)
[33](https://dl.acm.org/doi/10.24963/ijcai.2025/719)
[34](https://ieeexplore.ieee.org/document/10472977/)
[35](https://www.sciencedirect.com/science/article/pii/S0925231225018934)
[36](https://dl.acm.org/doi/10.1016/j.neunet.2024.107076)
[37](https://www.nature.com/articles/s44335-025-00036-2)
[38](https://snu.elsevierpure.com/en/publications/sign-gradient-descent-based-neuronal-dynamics-ann-to-snn-conversi)
[39](https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/aisy.202300383)
