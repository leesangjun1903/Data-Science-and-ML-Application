
# Deep Residual Learning in Spiking Neural Networks

## 1. 핵심 주장과 주요 기여 요약

본 논문은 **Spike-Element-Wise (SEW) ResNet**을 제안하여 스파이킹 신경망(Spiking Neural Networks, SNNs)에서 진정한 의미의 잔여학습(residual learning)을 구현할 수 있음을 보여줍니다.[1]

기존 Spiking ResNet은 표준 ResNet의 ReLU 활성화 함수를 스파이킹 뉴런으로 단순 치환하여 **항등 사상(identity mapping) 구현 불가** 및 **소실/폭발 기울기(vanishing/exploding gradient) 문제**를 야기했습니다. 이와 달리, SEW ResNet의 핵심 기여는 다음과 같습니다:[1]

- **항등 사상의 일반적 구현**: 모든 스파이킹 뉴런 모델에 대해 항등 사상을 효과적으로 구현 가능
- **기울기 안정성 보증**: 깊은 네트워크에서도 기울기 소실/폭발 문제 극복
- **깊이에 따른 성능 향상**: 100층 이상의 깊은 직접 학습(direct training) SNN 최초 구현[1]

***

## 2. 해결하는 문제와 제안 방법

### 2.1 문제 정의

**Spiking ResNet의 한계:**

기존 Spiking ResNet의 기본 블록은 다음과 같이 공식화됩니다:

$$O^l[t] = \text{SN}(\mathcal{F}_l(S^l[t]) + S^l[t])$$

여기서 $S^l[t]$는 입력, $\mathcal{F}_l$는 학습할 잔여 매핑, $\text{SN}$은 스파이킹 뉴런입니다.[1]

**문제점 1: 항등 사상 불가능성**

$\mathcal{F}_l(S^l[t]) \equiv 0$일 때, $O^l[t] = \text{SN}(S^l[t]) \neq S^l[t]$이므로 항등 사상이 구현되지 않습니다. LIF(Leaky Integrate-and-Fire) 뉴런과 같이 시간 상수 $\tau$가 학습 중 변하는 모델에서는 특히 심각합니다:[1]

$$H[t] = V[t-1] + \frac{1}{\tau}(X[t] - (V[t-1] - V_{reset}))$$

**문제점 2: 기울기 소실/폭발**

연쇄 미분을 통해 $k$개 블록의 기울기를 계산하면:

$$\frac{\partial O^{l+k-1}_j[t]}{\partial S^l_j[t]} = \prod_{i=0}^{k-1} \Theta'(S^{l+i}_j[t] - V_{th}) \to \begin{cases} 0 & \text{if } 0 < \Theta'(S^l_j[t] - V_{th}) < 1 \\ +\infty & \text{if } \Theta'(S^l_j[t] - V_{th}) > 1 \end{cases}$$

여기서 $\Theta'(x)$는 대리 기울기(surrogate gradient)입니다.[1]

### 2.2 제안 방법: SEW ResNet

**핵심 아이디어**: 스파이크의 이진 특성을 활용하여 **요소별 함수(element-wise function)** $g$를 도입합니다:

$$O^l[t] = g(\text{SN}(\mathcal{F}_l(S^l[t])), S^l[t]) = g(A^l[t], S^l[t])$$

**지원하는 요소별 함수:**[1]

| 함수명 | 식 | 항등 사상 조건 |
|--------|-----|-------------|
| ADD | $A^l[t] + S^l[t]$ | $A^l[t] \equiv 0$ |
| AND | $A^l[t] \land S^l[t]$ | $A^l[t] \equiv 1$ |
| IAND | $(1-A^l[t]) \land S^l[t]$ | $A^l[t] \equiv 0$ |

**항등 사상 구현:**

ADD 또는 IAND를 선택할 때, 마지막 배치 정규화(BN)의 가중치를 0으로, 편향을 0으로 설정하면:

$$O^l[t] = g(A^l[t], S^l[t]) = g(0, S^l[t]) = S^l[t]$$

이는 모든 스파이킹 뉴런 모델에 적용 가능합니다.[1]

**기울기 안정성:**

항등 사상이 구현될 때:

$$\frac{\partial O^{l+k-1}_j[t]}{\partial S^l_j[t]} = \prod_{i=0}^{k-1} \frac{\partial g(A^{l+i}_j[t], S^{l+i}_j[t])}{\partial S^{l+i}_j[t]} = 1$$

따라서 기울기가 상수로 유지되어 소실/폭발 문제가 해결됩니다.[1]

***

## 3. 모델 구조

### 3.1 스파이킹 뉴러 모델

기본 동역학은 다음과 같습니다:

$$H[t] = f(V[t-1], X[t])$$
$$S[t] = \Theta(H[t] - V_{th})$$
$$V[t] = H[t](1 - S[t]) + V_{reset}S[t]$$

여기서:
- $X[t]$: 시간 스텝 $t$에서의 입력 전류
- $H[t]$, $V[t]$: 막전위 및 스파이크 후 막전위
- $\Theta(x)$: 헤비사이드 함수
- $V_{th}$, $V_{reset}$: 발화 임계값 및 리셋 전위[1]

**IF(Integrate-and-Fire) 모델:**

$$H[t] = V[t-1] + X[t]$$

**LIF(Leaky Integrate-and-Fire) 모델:**

$$H[t] = V[t-1] + \frac{1}{\tau}(X[t] - (V[t-1] - V_{reset}))$$

### 3.2 SEW 기본 블록 (ADD 함수)

```
입력 S^l[t] → Conv-BN-SN (F_l) → A^l[t]
                                    ↓
                              g 함수 (ADD) → O^l[t]
                                    ↑
                            S^l[t] (shortcut)
```

**다운샘플링 블록:**

입력과 출력 차원이 다를 때, 쇼트컷에 스파이킹 뉴런을 추가합니다. 기존 Spiking ResNet은 {Conv-BN}을 사용하지만, SEW ResNet은 {Conv-BN-SN}을 사용합니다.[1]

***

## 4. 성능 향상 및 실험 결과

### 4.1 ImageNet 분류

**degradation problem 해결:**

[1]

SEW ResNet은 깊이 증가에 따라 훈련 손실이 감소하고 정확도가 증가하는 반면, Spiking ResNet은 깊어질수록 성능이 저하됩니다.[1]

**정량적 성능 비교:**

| 네트워크 | SEW ResNet (ADD) | Spiking ResNet |
|---------|-----------------|----------------|
| ResNet-18 | 63.18% | 62.32% |
| ResNet-34 | 67.04% | 61.86% |
| ResNet-50 | 67.78% | 57.66% |
| ResNet-101 | 68.76% | 31.79% |
| ResNet-152 | 69.26% | 10.03% |

SEW ResNet-101/152는 **100층 이상의 깊은 직접 학습 SNN의 첫 사례**입니다.[1]

**기존 방법과 비교:**

| 방법 | 정확도(%) | 시뮬레이션 시간스텝 (T) |
|-----|---------|---------------------|
| SEW ResNet-34 | 67.04 | 4 |
| Spiking ResNet-34(td-BN) | 67.05 | 6 |
| 최고 성능 ANN2SNN-34 | 74.61 | 256 |

직접 학습 방식에서 SEW ResNet은 최고 성능을 달성하며, ANN2SNN 방법에 비해 **64배 적은 시간스텝**으로 경쟁력 있는 정확도를 유지합니다.[1]

### 4.2 DVS Gesture (신경형태 데이터셋)

테스트 정확도: **97.92%** (7B-Net 구조, 16 시뮬레이션 시간스텝)[1]

기존 방법들과 비교:

| 방법 | 정확도(%) | 파라미터 | T |
|-----|---------|---------|---|
| SEW ResNet (7B-Net) | 97.92 | 0.13M | 16 |
| 기존 SOTA | 97.57 | 1.70M | 20 |
| Spiking ResNet-17(td-BN) | 96.87 | 11.18M | 40 |

**더 적은 파라미터와 시간스텝으로 우수한 성능**을 달성합니다.[1]

### 4.3 CIFAR10-DVS 분류

테스트 정확도: **70.2%** (8 시뮬레이션 시간스텝 기준)[1]

- T=4: 64.8%
- T=8: 70.2%
- T=16: 74.4%

기존 Spiking ResNet-19 (67.8%, T=10)을 **더 적은 시간스텝으로 초과**합니다.[1]

***

## 5. 일반화 성능 향상 가능성

### 5.1 발화율 분석

![1]

ImageNet에서 SEW ResNet의 각 블록 발화율(firing rate)을 분석하면:[1]

- SEW ResNet-18/34/50: 평균 1회 이하의 스파이크 발생 (T=4)
- SEW ResNet-101/152: 평균 2회 이하의 스파이크 발생

**낮은 발화율은 대부분의 블록이 항등 사상에 가까워지고 있음을 의미하며, 이는 네트워크가 필요한 부분에만 계산을 집중할 수 있음을 시사합니다.**[1]

### 5.2 기울기 흐름 분석

ResNet-152에서 기울기 진폭 분석 결과:[1]

**대리 기울기 함수:** $\sigma(x) = \frac{1}{\pi}\arctan(\frac{\pi}{2}\alpha x) + \frac{1}{2}$, $\sigma'(x) = \frac{\alpha}{2(1+(\frac{\pi}{2}\alpha x)^2)}$

- **Spiking ResNet**: 기울기가 깊은 층에서 얕은 층으로 지수적으로 감소 (소실)
- **SEW ResNet**: 항등 사상 영역에서 기울기가 상수로 유지 (안정적)

### 5.3 과적합(Overfitting) 제어

DVS Gesture에서 Random Temporal Delete (RTD) 데이터 증강 기법 적용 결과:[1]

- RTD 미적용: 훈련 정확도 높음, 테스트 정확도 낮음
- RTD 적용: 훈련 정확도 감소, 테스트 정확도 향상

**일반화 성능이 향상됨을 보여줍니다.**

***

## 6. 모델의 한계

### 6.1 ANN2SNN 방법과의 격차

직접 학습된 SEW ResNet-34 (67.04%)은 최고 성능 ANN2SNN 방법 (74.61%)에 비해 **약 7.6% 낮은 정확도**를 보입니다.[1]

### 6.2 요소별 함수 선택의 어려움

- **ADD**: 깊은 층에서 출력이 최대 k+1까지 증가 가능 (무한 출력 문제 부분 해결)
- **AND**: "침묵 문제(silence problem)"로 인해 극심한 기울기 소실
- **IAND**: AND보다 안정적이나 AND 함수에 비해 낮은 성능[1]

### 6.3 시뮬레이션 시간스텝 필요

정적 이미지 분류에서도 최소 4개 시간스텝 필요하며, 이는 단일 패스 추론 대비 **4배 연산량 증가**를 의미합니다.[1]

### 6.4 대리 기울기의 민감성

대리 기울도 함수의 선택이 학습에 매우 민감합니다. 예를 들어:

- Rectangular 함수 ( $\sigma'(x) = \text{sign}(|x| < \frac{1}{2})$ ): 0/1 기울기로 gradient vanishing 이론적으로 해결하나, 실제 학습 실패
- ArcTan 함수: 상대적으로 안정적이나 여전히 기울기 소실 위험[1]

***

## 7. 향후 연구에 미치는 영향과 고려사항

### 7.1 학계의 영향

**798회 인용 (NeurIPS 2021)**으로, SEW ResNet은 직접 학습 기반 깊은 SNN 연구의 핵심 기준이 되었습니다.[2][1]

**주요 영향:**

1. **다층 SNN 아키텍처 개선**: 최근 연구에서 SEW ResNet의 개념을 확장한 **XOResNet** (OR-ADD 쇼트컷과 XOR 메타-잔여 구조)이 제안되었으며, 스파이크 중복성 및 정보 손실 문제를 추가로 해결하고 있습니다.[3]

2. **Spiking Transformer 개발**: 대규모 SNN 구축의 필요성 증대에 따라, 2024년 연구에서는 **Spiking Transformers** 아키텍처가 활발히 연구되고 있습니다.[4]

3. **신경형태 하드웨어 적용**: SEW ResNet의 안정적인 훈련 방법이 신경형태 칩(예: SpiNNaker, Loihi) 위의 효율적 구현을 가능하게 했습니다.[5][4]

### 7.2 향후 연구 방향

**단기 과제 (1-2년):**

1. **정확도 격차 해소**: ANN2SNN 방법과의 7-8% 정확도 차이를 해결하기 위해:
   - 더 정교한 배치 정규화 변형 (threshold-dependent BN 확장)
   - 적응형 시간스텝 할당 메커니즘

2. **요소별 함수 최적화**: ADD 함수의 출력 범위 제한 문제 해결:
   - 조건부 정규화 기법
   - 새로운 이진 연산자 설계

3. **시뮬레이션 효율성**: 시간스텝 감소를 통한 연산 복잡도 감소:
   - 시간 코딩(temporal coding) 개선
   - 스파이크 압축 기법

**중기 과제 (2-5년):**

1. **Spiking Vision Transformer**: 최근 2024년 연구에서 **Spiking Transformers**가 주목받고 있으며, SEW ResNet의 원리를 트랜스포머 아키텍처에 적용하는 것이 중요 과제입니다.[4]

2. **Multimodal SNN**: 신경형태 센서(DVS, Neuromorphic Audio) 데이터와 기존 RGB/Audio 데이터의 통합 학습:
   - 하이브리드 인코딩 전략
   - 교차 모달 주의 메커니즘

3. **지연 학습(Delay Learning)**: 2025년 Nature Neuroscience 논문에서 제시된 **시냅스 지연의 학습 가능성**을 SEW ResNet과 결합하여 네트워크 용량 증가.[6]

**장기 과제 (5년 이상):**

1. **대규모 SNN 학습 안정성**: 최근 2024-2025년 연구에서 **GPU 가속 SNN 훈련** 및 **유전적 알고리즘 기반 SNN 설계**가 주목받고 있습니다. SEW ResNet의 수렴성 보증을 기반으로:[7][8]
   - 1000+층 극초 깊은 SNN 구축
   - 분산 학습(distributed training) 안정성 보증

2. **신경형태 하드웨어 최적화**: SEW ResNet의 효율성을 활용한:
   - 엣지 AI 칩(Intel Loihi 2, BrainScaleS-2) 기반 실시간 추론
   - 로보틱스, 자율주행, 센서 네트워크 응용

3. **이론적 기초 강화**: 
   - 스파이킹 활성화 함수의 기울기 동역학 완전한 이해
   - 대리 기울기 함수의 최적성 증명
   - SNN의 일반화 경계(generalization bound) 이론 개발

### 7.3 구체적 고려 사항

**구현 관점:**

1. **대리 기울기 선택의 중요성**: ArcTan 함수 ($\sigma'(x) = \frac{\alpha}{2(1+(\frac{\pi}{2}\alpha x)^2)}$, $\alpha=2$)가 실험적으로 가장 안정적이며, 새로운 대리 함수 개발 시 이를 벤치마크로 사용.[1]

2. **초기화 전략**: 영초기화(zero initialization)를 사용하면 기울기 흐름이 안정화되나, 역으로 underfitting 위험 증가. 적응형 초기화 기법 필요.[1]

3. **데이터셋 특성에 맞는 시간스텝 선택**:
   - 정적 이미지: 4-6 시간스텝
   - 신경형태 센서: 16-20 시간스텝
   - 동영상: 30+ 시간스텝

**이론적 관점:**

1. **일반화 경계**: 현재 SEW ResNet은 경험적으로 우수한 일반화 성능을 보이지만, VC 차원(VC dimension) 또는 Rademacher 복잡도(complexity)에 기반한 이론적 보증 부족.

2. **과적합 제어**: Random Temporal Delete 기법이 효과적이나, 더 일반적인 정규화 이론 필요.

***

## 8. 결론

SEW ResNet은 스파이킹 신경망에서 진정한 의미의 깊은 잔여학습을 최초로 실현한 획기적 연구입니다. 이진 스파이크의 특성을 활용한 요소별 함수 설계를 통해 항등 사상 문제를 우아하게 해결하고, 기울기 안정성을 수학적으로 보증했습니다.[1]

**핵심 성과:**
- 100층 이상의 직접 학습 SNN 최초 구현
- ImageNet에서 경쟁력 있는 정확도 달성 (69.26% @ ResNet-152)
- DVS 신경형태 데이터셋에서 SOTA 성능 (97.92%)

**향후 도전:**
- ANN2SNN 방법과의 정확도 격차 축소
- 시뮬레이션 시간스텝 감소 및 실시간 추론 구현
- Spiking Transformer 등 새로운 아키텍처와의 통합
- 신경형태 하드웨어 위의 극저전력 배포

SEW ResNet의 원리는 이미 학계에서 널리 채용되었으며, 최근 XOResNet, Spiking Transformer, 지연 학습 등의 연구에서 기초를 제공하고 있습니다.[6][3]

***

## 참고 문헌

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6654d428-b87f-4997-a6ca-e4c00ae981b3/2102.04159v6.pdf)
[2](https://proceedings.neurips.cc/paper/2021/file/afe434653a898da20044041262b3ac74-Paper.pdf)
[3](https://papers.ssrn.com/sol3/Delivery.cfm/0a9c4f59-a93e-4aef-8fc8-8fb72b3cd6e6-MECA.pdf?abstractid=5311949&mirid=1)
[4](https://arxiv.org/pdf/2409.02111.pdf)
[5](https://pmc.ncbi.nlm.nih.gov/articles/PMC11557524/)
[6](https://www.nature.com/articles/s41467-025-65394-8)
[7](http://arxiv.org/pdf/2408.00280.pdf)
[8](http://arxiv.org/pdf/2411.06792.pdf)
[9](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13449/3053865/Artificial-intelligence-and-related-topics-eg-machine-learning-artificial-neural/10.1117/12.3053865.full)
[10](https://dl.acm.org/doi/10.1145/3687272.3690884)
[11](https://ieeexplore.ieee.org/document/11159460/)
[12](https://arxiv.org/abs/2503.12645)
[13](https://ieeexplore.ieee.org/document/10914740/)
[14](https://journal.unesa.ac.id/index.php/jieet/article/view/38177)
[15](https://ijimds.org/articles/a-deep-learning-based-framework-for-dynamic-ecommerce-recommendation-using-online-reviews-and-product-features-2025-09-09)
[16](https://link.springer.com/10.1007/s00247-025-06311-5)
[17](https://peninsula-press.ae/Journals/index.php/EDRAAK/article/view/172)
[18](https://dl.acm.org/doi/10.1145/3687273.3687295)
[19](https://arxiv.org/pdf/2302.13939.pdf)
[20](https://pmc.ncbi.nlm.nih.gov/articles/PMC6209684/)
[21](http://arxiv.org/pdf/2406.03287.pdf)
[22](https://arxiv.org/pdf/2109.12894.pdf)
[23](https://pmc.ncbi.nlm.nih.gov/articles/PMC9894879/)
[24](https://pmc.ncbi.nlm.nih.gov/articles/PMC12528140/)
[25](https://arxiv.org/html/2510.06721v1)
[26](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1623497/pdf)
[27](https://openaccess.thecvf.com/content/ICCV2023/papers/Su_Deep_Directly-Trained_Spiking_Neural_Networks_for_Object_Detection_ICCV_2023_paper.pdf)
[28](https://www.sciencedirect.com/science/article/abs/pii/S0167739X25002481)
[29](https://arxiv.org/html/2510.14235v1)
