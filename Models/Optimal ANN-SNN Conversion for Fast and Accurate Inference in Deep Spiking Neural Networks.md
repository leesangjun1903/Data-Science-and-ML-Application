# Optimal ANN-SNN Conversion for Fast and Accurate Inference in Deep Spiking Neural Networks

### 1. 핵심 주장과 주요 기여

본 논문은 **인공신경망(ANN)에서 스파이킹신경망(SNN)으로의 변환 과정에서 정확도 손실과 긴 추론 시간 문제를 해결**하기 위한 이론적 분석과 실용적 방법론을 제시합니다. 주요 기여는 다음과 같습니다.[1]

**핵심 기여:**
- ANN-SNN 변환에 대한 이론적 기초 제공 및 최적 변환의 충분조건 도출
- **Rate Norm Layer(RNL)** 제안으로 ReLU 활성화 함수를 대체하여 직접 변환 가능하게 함
- 최적 적합 곡선(Optimal Fit Curve)을 정의하고 **Rate Inference Loss(RIL)** 개념 도입
- 2단계 학습 전략을 통해 높은 정확도와 빠른 추론 속도를 동시에 달성
- VGG-16, PreActResNet-18 등 깊은 네트워크에서 **거의 손실 없는 변환** 달성
- **8.6배 빠른 추론 성능**을 **0.265배의 에너지 소비**로 구현[1]

***

### 2. 해결하고자 하는 문제와 제안 방법

#### 2.1 기본 문제

SNNs 훈련의 어려움:
- SNNs의 뉴런은 이산 스파이크를 생성하여 직접 역전파 불가능
- ANN-SNN 변환은 GPU 계산이 적게 필요하고 최고 성능 달성 가능한 방법
- 그러나 변환 과정에서 **정확도 손실** 및 **긴 추론 시간** 발생[1]

#### 2.2 이론적 기초

**Lemma 1: 스파이킹 뉴런의 발화율 관계식**

$$r_l = \text{clip}\left(\frac{W_{l-1}r_{l-1} + b_{l-1}}{v_{th,l}}, 0, 1\right)$$

여기서 $r_l$은 층 $l$의 발화율, $v_{th,l}$은 발화 임계값입니다.[1]

**Theorem 1: 최적 ANN-SNN 변환 조건**

L층 ANN과 SNN 간의 변환이 $t \to \infty$일 때 가능하려면 다음 조건을 만족해야 합니다:[1]

$$\frac{W^{SNN}_{l-1}}{v_{th,l}} = \frac{W^{ANN}_{l-1}}{\text{max}_{l-1}/\text{max}_l}, \quad \frac{b^{SNN}_{l-1}}{v_{th,l}} = \frac{b^{ANN}_{l-1}}{\text{max}_l}$$

이는 **가중치 정규화(Weight Normalization)가 임계값 조정(Threshold Balancing)과 동등**함을 의미합니다.[1]

#### 2.3 Rate Norm Layer(RNL) 제안

ReLU를 대체하는 새로운 활성화 함수로, 훈련 가능한 상한선으로 발화율을 직접 시뮬레이션합니다:[1]

$$\theta_l = p_l \cdot \text{max}(W_{l-1}\hat{r}_{l-1} + b_{l-1})$$

$$z_l = \text{clip}(W_{l-1}\hat{r}_{l-1} + b_{l-1}, 0, \theta_l)$$

$$\hat{r}_l = \frac{z_l}{\theta_l}$$

여기서 $p_l \in $은 훈련 가능한 스칼라 매개변수이고, $\hat{r}_l$은 시뮬레이션된 발화율입니다.[1]

**RNL의 주요 장점:**
- 기존의 수동 임계값 설정(최대값 또는 99.9% 백분위수)을 훈련 가능한 방식으로 개선
- 모든 훈련 데이터를 사용하여 일반화 성능 향상
- Batch Normalization과 유사하게 배치별 최대값을 사용하여 정보 손실 최소화[1]

#### 2.4 빠른 추론을 위한 최적화

**K 곡선(Fit Curve)의 정의:**

$$K(\hat{r}, r(t)) = \frac{\|r(t) - \hat{r}\|_2^2}{\|\hat{r}\|_2^2}$$

이는 ANN의 시뮬레이션된 발화율과 SNN의 실제 발화율 간 차이를 정량화합니다.[1]

**Theorem 2: K 곡선의 상한선**

$$K_l < \frac{2\Omega_l}{t}$$

여기서 **Rate Inference Loss(RIL)**은:

$$\Omega_l = \frac{\|\hat{r}_l\|_1}{\|\hat{r}_l\|_2^2}$$

이는 발화율 수렴 속도를 역비례적으로 제어합니다.[1]

#### 2.5 손실 함수

최종 훈련 손실은:

$$L'(f(x), y) = L(f(x), y) + \lambda\sum\frac{\Omega_l}{L}$$

여기서 첫 번째 항은 분류 손실, 두 번째 항은 빠른 추론을 위한 정규화 항입니다.[1]

***

### 3. 모델 구조와 2단계 훈련 전략

#### 3.1 전체 아키텍처

기본 구조는 RNL을 포함한 ANN에서 시작하여 두 단계 훈련을 수행합니다:[1]

**Stage 1: 정확도 훈련**
- 가중치 $W, b$ 최적화
- 손실: $\min_{W,b} L(\hat{r}^*_L, y)$
- $p_l = 1$ 고정 상태로 진행[1]

**Stage 2: 빠른 추론 훈련**
- 임계값 $\theta_l$ 최적화 ($p_l$ 학습)
- 손실: $\min_{\theta} T(\hat{r}^\*_j, \hat{r}'_j) = 1 - \cos(\hat{r}^*_L, \hat{r}'_L) + \lambda\sum\frac{\Omega_l}{L}$
- 코사인 거리를 사용하여 뉴런 정보 유지[1]

#### 3.2 변환 알고리즘

훈련된 ANN에서 SNN으로의 직접 매핑:
- 가중치: $W^{SNN}_k = W^{ANN}_k$
- 편향: $b^{SNN}_k = b^{ANN}_k$
- **임계값: $v_{th,k} = \theta_k$ (RNL에서 학습된 값)**[1]

이는 기존 방법의 사후 정규화(Post-hoc Normalization)와 달리 훈련 중에 최적 임계값을 자동으로 학습합니다.

***

### 4. 성능 향상 및 실험 결과

#### 4.1 정확도 성능

| 데이터셋 | 네트워크 | 본 논문 정확도 | 변환 손실 | 이전 SOTA |
|---------|---------|-----------|---------|----------|
| MNIST | 7-CNN | 96.51% | 0.00% | 99.44%[1] |
| CIFAR-10 | VGG-16 | 92.86% | -0.04% | 93.63%[1] |
| CIFAR-10 | PreActResNet-18 | 93.45% | -0.39% | - |
| CIFAR-100 | VGG-16 | 75.02% | +0.54% | 70.93%[1] |
| CIFAR-100 | PreActResNet-34 | 72.91% | +0.80% | - |

**특히 CIFAR-100에서 VGG-16 기준 4.09% 성능 향상** 달성[1]

#### 4.2 빠른 추론 성능

| 추론 시간 | Max Norm | Robust Norm | RMP-SNN | 본 논문 |
|----------|----------|-----------|---------|--------|
| T=32 | 10.00% | 43.03% | - | 85.40%[1] |
| T=64 | 12.17% | 81.52% | - | 91.15%[1] |
| T=256 | 81.85% | 92.75% | - | 92.95%[1] |

**90% 정확도 달성 시: 52 시간스텝 (Max Norm은 446 시간스텝, 8.6배 빠름)**[1]

#### 4.3 에너지 효율성

$$P = \frac{\text{총 스파이크 수}}{1 \times 10^{-3}} \times \alpha \text{ (Watts)}$$

- 90% 정확도 달성 시 에너지 소비: 본 논문 0.265배 vs Max Norm 기준선[1]
- 빠른 추론성이 추가 에너지 증가를 야기하지 않음을 입증

***

### 5. 일반화 성능 향상 메커니즘

#### 5.1 RNL을 통한 일반화 개선

기존 방법들의 한계:
- **Max Norm**: 최대 활성화값만 사용하여 데이터 분포의 일부만 반영
- **Robust Norm**: 99.9% 백분위수를 수동으로 설정하여 데이터셋 의존성 높음
- **오프라인 정규화**: 훈련 데이터의 부분집합만 사용하여 분포 왜곡[1]

RNL의 일반화 메커니즘:
1. **전체 훈련 데이터 활용**: Running max를 통해 모든 배치 정보 누적
2. **적응적 임계값**: $p_l$의 훈련을 통해 각 층의 특성에 맞는 임계값 자동 학습
3. **배치 정규화와의 상호작용**: BN이 정규분포를 가정하는 반면, RNL은 발화율의 제약 조건 ()을 직접 모델링[1]

#### 5.2 Rate Inference Loss를 통한 일반화

Squeeze Theorem 기반 분석:

$$\lim_{t \to \infty} K_l = 0 \text{ (보장)}$$

$\Omega_l$ 최소화는:
- 계층별로 균형잡힌 발화율 분포 유지
- 깊은 네트워크에서의 누적 오차 감소
- 뉴런 포화 현상 완화[1]

#### 5.3 실험적 증거

VGG-16 구조에서 $\Omega$ 분포:
- Max Norm: 계층 간 큰 편차 (일부 계층은 매우 높은 값)
- 본 논문: **상대적으로 낮고 균형잡힌 $\Omega$ 분포**
- 결과: 모든 계층에서 균등한 수렴 속도[1]

***

### 6. 한계 및 개선 필요 영역

#### 6.1 논문의 한계

1. **데이터셋 제한성**
   - MNIST, CIFAR-10/100 등 정적 이미지 분류만 평가
   - 동적 신경형태 데이터셋(DVS, 이벤트 카메라) 미포함[1]

2. **2단계 훈련의 복잡성**
   - 첫 단계에서 $p_l = 1$ 고정 필요
   - 하이퍼파라미터 $\lambda$ 튜닝 필요
   - Stage 2에서 모든 층이 동일 $p_l$ 공유하여 유연성 감소[1]

3. **실제 하드웨어 검증 부족**
   - 신경형태 칩(Loihi, SpiNNaker, TrueNorth) 구현 미실행
   - 에너지 모델링은 Cao et al. 추정치 기반[1]

4. **깊은 네트워크 제한**
   - 최대 34층(PreActResNet-34)까지만 평가
   - Transformer 구조 미처리

#### 6.2 이론적 제한

1. **무한 시간 가정 ($t \to \infty$)**
   - Lemma 1의 발화율 관계식은 정상 상태(Steady-state) 가정
   - 초기 계층에서의 일시적 오류 미분석[1]

2. **Constant Coding 의존성**
   - 분석이 상수 코딩에만 국한
   - Poisson 코딩의 확률적 특성 미적용

***

### 7. 최신 연구 동향 및 미래 연구 고려사항

#### 7.1 2024-2025년 최신 연구 진전

**1) 향상된 ANN-SNN 변환 방법**

2025년 ICML에서 발표된 **Efficient ANN-SNN Conversion with Error Compensation Learning**:[2]
- **Learnable Threshold Clipping Function** 제안으로 적응적 임계값 설정
- **Dual-Threshold Neurons**를 통해 정량화 오류 동적 감소
- ResNet-34에서 **64배 시간스텝 감소**로 ImageNet 74% 정확도 달성
- 본 논문(2021)과 유사한 방향이나 더 나은 성능[2]

**2) PASCAL: Precise and Efficient ANN-SNN Conversion (2025)**[2]
- **Quantization-Clip-Floor-Shift(QCFS)** 활성화 함수 도입
- **계층별 양자화 설정**으로 추론 시간 최적화
- ResNet-34에서 2시간스텝으로 74% ImageNet 정확도 달성 (혁신적 성과)

**3) 멀티스케일 주의 메커니즘 (2024)**[3]
- Spiking Multiscale Attention(SMA) 모듈 제안
- **Attention ZoneOut(AZO)** 정규화로 일반화 오류 감소
- ImageNet-1K에서 **77.1% 정확도** (104층 ResNet)[3]

#### 7.2 일반화 성능 개선 연구

**1) 시간-공간 일관성 기반 자기 증류 (2024)**[4]
- Self-Distillation Learning을 SNNs에 적용
- Temporal Self-Distillation: 긴 시간스텝을 암묵적 교사로 활용
- Spatial Self-Distillation: 중간층 출력을 최종 출력으로 가이드
- 추론 오버헤드 없이 **우수한 일반화 능력** 입증[4]

**2) 신경형태 데이터셋에서의 성능 (2024-2025)**
- DVS(Dynamic Vision Sensor) 데이터에 특화된 SNNs 개발
- CIFAR-10-DVS, DVS-Gesture 등 이벤트 기반 벤치마크에서 SOTA 성능 달성
- 본 논문(2021)은 정적 이미지만 다루었으나, 최신 연구는 **동적 데이터 처리** 중심[4]

**3) 긴 시간 의존성을 위한 구조 (2024)**[5]
- **Autaptic Synaptic Circuit(STC) 모델** 제안으로 LIF 뉴런 확장
- 생물학적 자촉시냅스(Autaptic Synapse) 개념 도입
- 시공간 예측 작업에서 적응형 모델 대비 우수 성능

#### 7.3 대규모 SNNs 및 Transformer 아키텍처 (2024)**[6]
- Spiking Transformers 출현 (2024년 동향)
- DCNNs뿐 아니라 Attention 메커니즘 SNNs 적용
- **100M+ 파라미터 규모** SNNs 개발 추진
- LLM 수준의 대규모 학습 가능성 탐색[6]

#### 7.4 신경형태 하드웨어 활용 (2024-2025)**[7][8]
- **Neuromorphic Hardware Co-design** 트렌드
- SpiNNaker, Loihi 2 등 실제 칩에서의 구현 검증 연구 활발
- Algorithm-Hardware Co-design Framework로 통합 최적화
- 실제 배포 환경에서의 **에너지 효율성 검증** 진행 중[8][7]

***

### 8. 앞으로의 연구 시 고려할 점

#### 8.1 즉시적 개선 방향

1. **동적 신경형태 데이터 처리**
   - DVS, 이벤트 카메라 같은 시간 해상도 높은 데이터에 적응
   - Constant Coding 대체 코딩 스킴 연구[3][4]

2. **더 깊은 네트워크 및 새로운 아키텍처**
   - Residual 구조, Batch Norm 깊은 상호작용 분석
   - Spiking Transformer 패러다임 통합[6]

3. **실제 하드웨어 검증**
   - Loihi 2, SpiNNaker 등에서의 실제 구현 및 에너지 측정
   - Algorithm-Hardware Co-design 통합 최적화[7]

#### 8.2 이론적 심화

1. **비정상 상태(Transient) 분석**
   - Lemma 1의 $t \to \infty$ 가정 완화
   - 초기 계층에서의 오류 누적 메커니즘 분석

2. **확률적 코딩 분석**
   - Poisson 코딩에 대한 이론 확장
   - 하이브리드 코딩 방식 탐색

#### 8.3 일반화 성능 최적화

1. **정규화 전략 고도화**
   - RNL과 다른 정규화 기법(Dropout, Layer Norm) 결합
   - 데이터셋 특성에 따른 적응적 정규화[4]

2. **손실 함수 설계 개선**
   - $\Omega_l$ 계산 고도화
   - 계층별 가중치 할당으로 유연성 증대

#### 8.4 응용 확대

1. **시계열 및 자연어 처리**
   - 시공간 작업에 SNNs 확대 적용[5]
   - Long Short-Term Memory 패턴을 SNNs에서 구현

2. **엣지 컴퓨팅 및 실시간 처리**
   - 자동주행차, 로봇 등 저전력 응용
   - 온디바이스 학습 가능성 탐색[8]

3. **Neuromorphic AI와 뇌-컴퓨터 인터페이스**
   - 신경과학 해석 가능성 강화
   - 생물학적 리얼리즘과 성능의 균형[7][6]

***

### 결론

본 논문 "Optimal ANN-SNN Conversion for Fast and Accurate Inference in Deep Spiking Neural Networks"는 **ANN-SNN 변환의 이론적 기초를 확립**하고 **Rate Norm Layer와 Rate Inference Loss** 개념을 통해 정확도와 추론 속도를 획기적으로 향상시킨 중요한 기여를 했습니다. 2024-2025년 최신 연구는 이 논문의 기초 위에서 **동적 데이터 처리, 대규모 아키텍처, 실제 하드웨어 통합, 일반화 성능 개선** 방향으로 진화하고 있습니다.

특히 **PASCAL(2025)**과 **Error Compensation Learning(2025)** 방법들이 더욱 극단적인 지연 시간 단축(1-2 시간스텝)을 달성하며, Spiking Transformers의 출현은 SNNs의 적용 범위를 혁신하고 있습니다. 앞으로의 연구자들은 신경형태 데이터 처리, 하드웨어 공동 설계, 이론적 심화를 통해 본 논문의 성과를 발전시킬 수 있을 것입니다.[5][8][2][3][6][7][4][1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6ce2f25e-fb3b-407a-b1c1-f1786d681b12/2105.11654v1.pdf)
[2](https://www.semanticscholar.org/paper/926f52d457683a39889f0ea7cda9e8f7fef4f42a)
[3](https://arxiv.org/abs/2405.13672)
[4](https://arxiv.org/abs/2406.07862)
[5](https://www.nature.com/articles/s44335-024-00002-4)
[6](https://arxiv.org/abs/2406.00405)
[7](https://arxiv.org/abs/2409.02111)
[8](https://link.springer.com/10.1007/s13534-024-00406-y)
[9](https://ieeexplore.ieee.org/document/10647018/)
[10](https://ieeexplore.ieee.org/document/10835501/)
[11](https://ieeexplore.ieee.org/document/10529973/)
[12](http://arxiv.org/pdf/2406.03287.pdf)
[13](https://arxiv.org/abs/2312.01213)
[14](https://pmc.ncbi.nlm.nih.gov/articles/PMC10847652/)
[15](https://arxiv.org/pdf/2204.07050.pdf)
[16](http://arxiv.org/pdf/2406.02923.pdf)
[17](https://arxiv.org/pdf/2409.02111.pdf)
[18](http://arxiv.org/pdf/2501.14490.pdf)
[19](https://arxiv.org/abs/2407.04525)
[20](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1665778/full)
[21](https://openreview.net/forum?id=9lw5HYPT4Y&noteId=lLzoMaf7nw)
[22](https://arxiv.org/abs/2411.11575)
[23](https://www.nature.com/articles/s41467-025-65394-8)
[24](https://arxiv.org/abs/2505.01730)
[25](https://pubs.rsc.org/en/content/articlehtml/2023/ma/d3ma00449j)
[26](https://www.ebrains.eu/news-and-events/spiking-neural-networks-reach-a-new-level)
[27](https://proceedings.mlr.press/v202/jiang23a.html)
[28](https://www.pnas.org/doi/10.1073/pnas.2109194119)
[29](https://arxiv.org/html/2510.14235v1)
