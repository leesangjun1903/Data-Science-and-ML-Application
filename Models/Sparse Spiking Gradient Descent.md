
# Sparse Spiking Gradient Descent

## 1. 핵심 주장과 주요 기여 요약

**Sparse Spiking Gradient Descent**는 Nicolas Perez-Nieves와 Dan F.M. Goodman이 Imperial College London에서 개발한 최초의 **희소(sparse) SNN 역전파 알고리즘**으로, NeurIPS 2021에서 발표되었다. 이 연구의 핵심 주장은 스파이킹 신경망(SNN)의 시공간적 희소성(spatio-temporal sparsity)을 학습 과정에서도 활용할 수 있다는 것이다.[1]

**주요 기여**는 다음과 같다:
- 기존 surrogate gradient 방법과 **동일하거나 더 나은 정확도**를 유지하면서 역전파 속도를 **최대 150배 향상**
- GPU 메모리 사용량 **최대 85% 절감**
- 기울기가 **활성 뉴런(active neurons)만을 통해 역전파**되도록 하는 새로운 surrogate gradient 정의 제안
- SNN 학습이 **3-factor Hebbian 학습 규칙**으로 자연스럽게 해석될 수 있음을 이론적으로 입증[1]

***

## 2. 해결하고자 하는 문제

### 2.1 기존 문제점

SNN은 뇌의 에너지 효율적인 스파이크 기반 계산을 모방하여 뉴로모픽 하드웨어에서 **저전력 추론**이 가능하지만, **학습 과정**은 여전히 ANN을 위해 개발된 **밀집 텐서(dense tensor) 연산**에 의존하고 있었다. 이로 인해:[1]

1. **계산 비효율성**: SNN의 희소 활성화 패턴을 학습 시 활용하지 못함
2. **메모리 병목**: 모든 시간 단계에서 기울기를 저장해야 함
3. **Forward-Backward 격차**: 순전파는 희소하지만 역전파는 밀집 연산 필요
4. **비미분성 문제**: 스파이크 함수의 도함수가 임계값에서 무한대, 나머지에서 0[1]

### 2.2 Surrogate Gradient의 한계

기존 surrogate gradient 방법은 스파이크 함수의 비미분성을 해결했지만, **모든 뉴런에 대해 기울기를 계산**해야 하므로 희소성의 이점을 활용하지 못했다.[2][1]

***

## 3. 제안하는 방법 (수식 포함)

### 3.1 Forward Model

각 층 $$l$$의 뉴런 $$j$$에 대해, 막전위(membrane potential)는 다음과 같은 Leaky Integrate-and-Fire (LIF) 모델을 따른다:[1]

$$
V^{(l+1)}_j[t+1] = \alpha(V^{(l+1)}_j[t] - V_{rest}) + V_{rest} + \sum_i S^{(l)}_i[t]W^{(l)}_{ij} - (V_{th} - V_r)S^{(l+1)}_j[t]
$$

여기서:
- $$\alpha = \exp(-\Delta t/\tau)$$: 누출 계수
- $$S^{(l)}_i[t]$$: 뉴런 $$i$$의 시간 $$t$$에서의 스파이크
- $$W^{(l)}_{ij}$$: 시냅스 가중치
- $$V_{th}$$: 발화 임계값, $$V_r$$: 리셋 전위

스파이킹 함수는 계단 함수로 정의된다:[1]

$$
S^{(l)}_i[t] = f(V^{(l)}_i[t]) = \begin{cases} 1, & V > V_{th} \\ 0, & \text{otherwise} \end{cases}
$$

### 3.2 핵심 정의: 활성 뉴런과 희소 기울기

**정의 1 (활성 뉴런)**: 역전파 임계값 $$B_{th} \in \mathbb{R}$$이 주어졌을 때, 뉴런 $$j$$가 시간 $$t$$에서 **활성(active)**이라 함은:[1]

$$
|V_j[t] - V_{th}| < B_{th}
$$

**정의 2 (스파이크 기울기)**: 스파이크 기울기는 다음과 같이 정의된다:[1]

$$
\frac{dS_j[t]}{dV_j[t]} := \begin{cases} g(V_j[t]), & \text{if } V_j[t] \text{ is active} \\ 0, & \text{otherwise} \end{cases}
$$

여기서 $$g(V) = \frac{1}{(\beta|V - V_{th}| + 1)^2}$$로 설정된다[1].

### 3.3 희소 가중치 기울기

이 정의에 따라, 가중치 기울기는 **활성 뉴런에서만 계산**된다:[1]

$$
\nabla W^{(l)}_{ij}[t] = \begin{cases} \epsilon^{(l+1)}_j[t] \frac{dS^{(l+1)}_j[t]}{dV^{(l+1)}_j[t]} \left(\sum_{k<t} \alpha^{t-k-1} S^{(l)}_i[k]\right), & V^{(l+1)}_j[t] \text{ is active} \\ 0, & \text{otherwise} \end{cases}
$$

$$
\nabla W^{(l)}_{ij} = \sum_t \nabla W^{(l)}_{ij}[t]
$$

### 3.4 희소 스파이크 기울기

스파이크 기울기 역시 활성 뉴런에서만 계산된다:[1]

$$
\nabla S^{(l)}_j[t] = \begin{cases} \sum_j W_{ij} \delta^{(l+1)}_j[t], & V^{(l)}_j[t] \text{ is active} \\ 0, & \text{otherwise} \end{cases}
$$

**Proposition 1 (재귀 관계)**: $$\delta_j[t]$$는 메모이제이션을 통해 효율적으로 계산된다:[1]

$$
\delta_j[t] = \begin{cases} \alpha^n \delta_j[t+n], & \text{if } \frac{dS_j[k]}{dV_j[k]} = 0 \text{ for } t+1 \leq k \leq t+n \\ \nabla S_j[t+1] \frac{dS_j[t+1]}{dV_j[t+1]} + \alpha^n \delta_j[t+n], & \text{otherwise} \end{cases}
$$

***

## 4. 모델 구조

논문에서 사용된 네트워크 구조는 **3층 완전 연결 SNN**으로, 입력층, 두 개의 은닉층(동일한 뉴런 수), 그리고 무한 임계값을 가진 판독층(readout layer)으로 구성된다.[1]

| 구성 요소 | 설명 |
|-----------|------|
| **뉴런 모델** | Simplified LIF (Leaky Integrate-and-Fire)[1] |
| **층 구조** | 입력 → 은닉1 → 은닉2 → 판독층[1] |
| **연결** | 완전 연결 (Fully Connected)[1] |
| **은닉층 뉴런 수** | 200~1000개 (실험에 따라 변경)[1] |
| **구현** | PyTorch CUDA Extension[1] |

추가적으로, **5개 은닉층 네트워크**(300개 뉴런)와 **합성곱 SNN**에서도 검증되었다.[1]

***

## 5. 성능 향상

### 5.1 속도 및 메모리 개선

| 데이터셋 | 역전파 속도 향상 | 메모리 절감 | 테스트 정확도 |
|----------|------------------|-------------|---------------|
| **Fashion-MNIST**[1] | ~40x | ~35% | 82.2%[1] |
| **N-MNIST**[1] | ~40x | ~35% | 92.7%[1] |
| **SHD**[1] | ~40x | ~35% | 77.5%[1] |

$$B_{th} = 0.95$$로 설정 시 **최대 150배 속도 향상**과 **85% 메모리 절감**이 달성되었다.[1]

### 5.2 활성 뉴런 비율

실험 결과, 평균 활성 뉴런 비율은 **2% 미만**으로 유지되어 이론적 에너지 절감 상한이 **98% 이상**에 달한다:[1]

| 데이터셋 | 층 1 활성도 | 층 2 활성도 | ∇W 에너지 절감 | ∇S 에너지 절감 |
|----------|-------------|-------------|----------------|----------------|
| F-MNIST[1] | 1.06% | 0.87% | 99.13% | 98.94%[1] |
| N-MNIST[1] | 1.12% | 0.77% | 99.23% | 98.88%[1] |
| SHD[1] | 1.70% | 1.09% | 98.91% | 98.30%[1] |

### 5.3 희소성-정확도 트레이드오프

$$B_{th}$$를 0.95까지 높여도 **학습이 손상되지 않고** 동일한 정확도를 유지하며, $$B_{th} \rightarrow 1$$에 가까워질수록 활성 뉴런이 급격히 감소하여 학습이 중단된다.[1]

***

## 6. 일반화 성능 향상 가능성

### 6.1 논문에서 제시된 관련 내용

본 논문에서 직접적으로 일반화 성능 향상을 주장하지는 않지만, 다음과 같은 요소들이 일반화에 긍정적으로 기여할 수 있다:

**희소 기울기의 정규화 효과**: 희소 역전파는 **적응적 드롭아웃(adaptive dropout)**이나 **적응적 희소성(adaptive sparsity)**과 유사한 효과를 제공한다. 이러한 기법들은 ANN에서 과적합 방지에 효과적임이 입증되어 있다.[3][4][1]

**3-factor Hebbian 학습 규칙 해석**: 제안된 학습 규칙은 생물학적 3-factor Hebbian 학습 규칙로 해석되며, 이는 신경과학에서 강건한 학습을 위한 원리로 알려져 있다.[5][1]

### 6.2 최신 연구에서의 일반화 성능 향상 접근법

최근 연구들은 SNN의 일반화 성능 향상을 위한 다양한 방법을 제안하고 있다:

**Temporal Regularization Training (TRT)**: 2025년 연구에서는 시간 의존적 정규화 메커니즘을 도입하여 초기 시간 단계에 더 강한 제약을 가함으로써 과적합을 효과적으로 완화했다. 이 방법은 손실 지형(loss landscape)을 평탄화하여 일반화 성능을 향상시킨다.[6]

**희소 기울기와 일반화의 트레이드오프**: 최근 연구에 따르면, SNN에서 **희소 기울기는 적대적 강건성을 향상**시키지만 **일반화 능력을 저하**시킬 수 있으며, 반대로 **밀집 기울기는 일반화를 지원**하지만 공격에 취약해진다. 이러한 트레이드오프는 Sparse Spiking Gradient Descent 적용 시 고려해야 할 중요한 요소이다.[7]

**Masked Surrogate Gradients (MSG)**: 희소 surrogate gradient를 사용하면서도 학습 효과와 기울기의 희소성 간 균형을 맞춤으로써 SNN의 일반화 능력을 향상시킬 수 있다.[8][9][10]

### 6.3 일반화 성능 향상을 위한 제언

본 논문의 방법론과 최신 연구를 종합하면, 다음과 같은 전략이 일반화 성능 향상에 기여할 수 있다:

1. **$$B_{th}$$ 값의 적절한 조절**: 너무 엄격한 희소성은 기울기 소실을 유발하고, 너무 느슨한 희소성은 효율성을 저하시키므로 작업에 맞는 최적값 탐색이 필요하다[1]
2. **정규화 기법과의 결합**: 활성도 정규화(activity regularization)와 함께 사용하여 과적합 방지[1]
3. **데이터 증강**: 제한된 뉴로모픽 데이터셋의 규모 문제 해결[6]
4. **시간적 이질성 활용**: 다양한 시간 단계에서의 출력 다양성을 증가시켜 성능 향상[11]

***

## 7. 한계점

### 7.1 논문에서 인정한 한계

1. **희소 CUDA 커널 개발 필요**: 각 층 유형마다 별도의 희소 CUDA 커널을 개발해야 하며, 이는 자동미분 플랫폼에서 지원되지 않는다[1]
2. **합성곱 층의 제한된 검증**: 합성곱 SNN에서는 희소 연산자를 구현하지 않고 기울기 클램핑만 테스트했다[1]
3. **메모리 절감의 한계**: 연산 수는 크게 줄었지만 메모리 요구량은 최대 85%까지만 감소하여 여전히 병목이 존재한다[1]
4. **동적 계산 그래프**: 역전파 그래프가 매 배치마다 변경되어 GPU보다 FPGA나 뉴로모픽 하드웨어가 더 적합할 수 있다[1]

### 7.2 추가적인 한계점

- **완전 연결 층에 집중**: 대규모 비전 작업에 필수적인 합성곱, 어텐션 메커니즘에 대한 희소 구현이 부족[1]
- **하드웨어 의존성**: 실제 에너지 효율성은 특정 GPU/하드웨어 구성에 따라 다름[1]
- **일반화 성능 검증 부족**: 과적합 방지나 일반화에 대한 직접적인 실험이 미비[1]

***

## 8. 향후 연구에 미치는 영향과 고려사항

### 8.1 연구 영향

**뉴로모픽 컴퓨팅 분야**: 이 연구는 SNN 학습의 주요 병목인 역전파 비효율성을 해결하여, **온칩 학습(on-chip training)**의 가능성을 열었다. 최근 연구들은 이를 기반으로 더욱 효율적인 학습 방법을 개발하고 있다.[12][13][14][1]

**SparseProp (2023)**: Perez-Nieves의 후속 연구인 SparseProp은 이벤트 기반 시뮬레이션과 학습을 위한 효율적 솔루션을 제공하며, **백만 개의 LIF 뉴런**을 가진 희소 SNN 시뮬레이션에서 이전 구현 대비 **4자릿수 이상의 속도 향상**을 달성했다.[14]

**TT-SNN (2024)**: Tensor Train 분해를 활용하여 **모델 크기 7.98배, FLOPs 9.25배 감소**와 함께 학습 에너지를 28.3% 절감하는 방법이 제안되었다.[15]

**Cannistraci-Hebb SNN (2025)**: 초희소(ultra-sparse) SNN을 위한 동적 희소 학습 프레임워크로, **97.75% 희소성**에서도 기준 모델 대비 0.16% 정확도 향상을 달성했다.[16][17]

### 8.2 향후 연구 시 고려할 점

**효율적인 희소 연산자 개발**: 합성곱, 어텐션 등 다양한 층 유형에 대한 희소 CUDA 커널이나 라이브러리 수준의 지원이 필요하다.[13][1]

**뉴로모픽 하드웨어와의 통합**: Intel Loihi, IBM TrueNorth 등 뉴로모픽 칩에서의 직접적인 학습 구현을 위한 알고리즘 적응이 요구된다.[18]

**일반화와 희소성의 균형**: 최근 연구에서 밝혀진 희소 기울기와 일반화 간의 트레이드오프를 고려한 최적화 전략 개발이 필요하다.[7][6]

**대규모 모델로의 확장**: Transformer 기반 Spiking Neural Network나 대규모 언어 모델로의 확장 연구가 진행 중이며, 희소 학습이 핵심 역할을 할 것으로 예상된다.[19][12]

**하이브리드 접근법**: ANN-SNN 변환과 직접 학습의 장점을 결합하거나, 자기 증류(self-distillation) 기법과의 통합을 통한 성능 향상이 유망하다.[20][12]

**하드웨어-소프트웨어 공동 설계**: 알고리즘과 하드웨어의 공동 최적화를 통해 실제 에너지 효율성을 극대화하는 연구가 중요해지고 있다.[21][13][18]

***

이 연구는 SNN 학습의 효율성을 획기적으로 개선하여 뉴로모픽 컴퓨팅의 실용화에 중요한 기여를 했으며, 후속 연구들이 이를 기반으로 더욱 발전된 희소 학습 방법들을 개발하고 있다. 특히 에너지 효율적인 AI 시스템 구현이 점점 중요해지는 현 시점에서, 이 연구의 가치는 더욱 부각되고 있다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/a62d5256-f665-4ab2-8f65-0c7cfce8b7f8/2105.08810v2.pdf)
[2](https://www.ijcai.org/proceedings/2023/0335.pdf)
[3](https://www.mdpi.com/2079-9292/13/15/2948)
[4](https://www.frontiersin.org/articles/10.3389/fnins.2025.1551656/full)
[5](https://www.nature.com/articles/s41598-025-90113-0)
[6](https://arxiv.org/abs/2506.19256)
[7](https://arxiv.org/abs/2509.23762)
[8](https://linkinghub.elsevier.com/retrieve/pii/S0893608024004234)
[9](https://www.sciencedirect.com/science/article/abs/pii/S0893608024004234)
[10](https://arxiv.org/abs/2406.19645)
[11](https://openreview.net/forum?id=l7ZmdeFyM1&noteId=p6KuLAXRCy)
[12](https://arxiv.org/abs/2510.06254)
[13](https://ieeexplore.ieee.org/document/10558351/)
[14](http://arxiv.org/pdf/2312.17216.pdf)
[15](https://arxiv.org/abs/2401.08001)
[16](https://www.semanticscholar.org/paper/c1f3cea533c1430146efa0e6b926387f3dda958a)
[17](https://arxiv.org/html/2511.05581v1)
[18](https://pubs.aip.org/aip/apm/article/12/10/109201/3317314/Roadmap-to-neuromorphic-computing-with-emerging)
[19](https://arxiv.org/pdf/2302.13939.pdf)
[20](https://arxiv.org/html/2502.21193)
[21](https://ieeexplore.ieee.org/document/10454472/)
[22](https://arxiv.org/abs/2403.03409)
[23](https://ieeexplore.ieee.org/document/10558156/)
[24](https://ieeexplore.ieee.org/document/10546346/)
[25](https://ieeexplore.ieee.org/document/10982407/)
[26](https://arxiv.org/pdf/2304.12214.pdf)
[27](https://arxiv.org/pdf/2204.05422.pdf)
[28](https://arxiv.org/pdf/2206.09449.pdf)
[29](https://arxiv.org/pdf/2401.10843.pdf)
[30](https://arxiv.org/pdf/2302.00232.pdf)
[31](https://arxiv.org/pdf/1911.11134.pdf)
[32](http://arxiv.org/pdf/2406.01072.pdf)
[33](https://openreview.net/forum?id=4ILqqOJFkS)
[34](https://arxiv.org/html/2507.16043v2)
[35](https://arxiv.org/abs/2405.15616)
[36](https://arxiv.org/html/2502.13572v1)
[37](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.865897/full)
[38](https://firstignite.com/exploring-the-latest-neuromorphic-computing-advancements-in-2024/)
[39](https://liner.com/review/sparse-spiking-neural-network-exploiting-heterogeneity-in-timescales-for-pruning)
[40](https://proceedings.neurips.cc/paper/2021/hash/c4ca4238a0b923820dcc509a6f75849b-Abstract.html)
[41](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2024.1455530/pdf)
[42](https://proceedings.mlr.press/v202/deng23d/deng23d.pdf)
[43](https://www.backend.ai/blog/2024-06-27-neuromorphic-computing-deep-learning)
[44](https://www.nature.com/articles/s41467-025-65394-8)
[45](https://www.nature.com/collections/jaidjgeceb)
[46](https://ieeexplore.ieee.org/document/10388553/)
[47](https://www.sciencedirect.com/science/article/abs/pii/S0952197625015453)
[48](https://dl.acm.org/doi/10.1016/j.neunet.2024.106499)
[49](http://biorxiv.org/lookup/doi/10.1101/2023.04.19.537473)
[50](https://www.mdpi.com/2079-9292/11/13/2097)
[51](https://ieeexplore.ieee.org/document/11134387/)
[52](https://arxiv.org/abs/2502.12172)
[53](https://onlinelibrary.wiley.com/doi/10.1002/oca.70023)
[54](https://www.worldscientific.com/doi/10.1142/S0129065725500455)
[55](https://pmc.ncbi.nlm.nih.gov/articles/PMC11330889/)
[56](https://arxiv.org/pdf/2211.08397.pdf)
[57](https://pmc.ncbi.nlm.nih.gov/articles/PMC8603828/)
[58](https://arxiv.org/pdf/2208.01204.pdf)
[59](https://pmc.ncbi.nlm.nih.gov/articles/PMC11850897/)
[60](https://arxiv.org/abs/2201.03299)
[61](https://proceedings.neurips.cc/paper_files/paper/2024/hash/3b6aaffec941f98930753fa6d6de7263-Abstract-Conference.html)
[62](https://neptune.ai/blog/fighting-overfitting-with-l1-or-l2-regularization)
[63](https://www.themoonlight.io/ko/review/enhancing-generalization-of-spiking-neural-networks-through-temporal-regularization)
[64](https://stackoverflow.com/questions/36139980/prevention-of-overfitting-in-convolutional-layers-of-a-cnn)
[65](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1383844/full)
[66](https://wikidocs.net/61374)
[67](https://pmc.ncbi.nlm.nih.gov/articles/PMC11350220/)
[68](https://pmc.ncbi.nlm.nih.gov/articles/PMC8589121/)
[69](https://dl.acm.org/doi/full/10.1145/3510413)
[70](https://www.sciencedirect.com/science/article/abs/pii/S0031320324008458)
[71](https://blog.naver.com/laonple/220527647084)
[72](https://www.sciencedirect.com/science/article/pii/S0925231225024129)
[73](https://www.nature.com/articles/s41467-024-51110-5)
[74](https://www.kaggle.com/code/pythonafroz/preventing-overfitting-a-guide-to-regularization)
