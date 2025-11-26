# Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural Networks

### 핵심 주장 및 주요 기여

"Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural Networks" 논문은 스파이킹 신경망(SNN)의 성능을 제한하는 중요한 문제를 해결합니다. 논문의 핵심 주장은 기존 SNN 학습 방법들이 **시냅스 가중치만 학습하고 뉴런의 막 시간 상수(τ, membrane time constant)를 고정된 하이퍼파라미터로 취급**한다는 점입니다.[1]

이는 생물학적 현실과 불일치합니다. 뇌의 다양한 영역에서 뉴런들은 서로 다른 막 시간 상수를 가지며, 이 차이는 작업 기억 표현과 학습 형성에 필수적입니다. 논문은 이 문제를 극복하기 위해 **모든 뉴런에 동일한 τ를 강제하는 것이 뉴런의 이질성을 제한하고 결과적으로 SNN의 표현력을 감소**시킨다고 주장합니다.[1]

주요 기여는 다음과 같습니다:[1]

1. **Parametric Leaky Integrate-and-Fire (PLIF) 뉴런 모델** 제안: 시냅스 가중치와 막 시간 상수를 모두 학습할 수 있는 백프로파게이션 기반 학습 알고리즘 개발

2. **SNN의 초기값 민감도 감소**: PLIF 뉴런은 초기 막 시간 상수 값에 덜 민감하며 더 빠른 수렴 속도를 보임

3. **Spike Max-pooling 재평가**: 최대값 풀링이 SNN에서 정보 손실을 야기하지 않으며, 오히려 계산 비용이 낮고 이진 호환성이 우수함을 입증

***

### 해결하고자 하는 문제

#### 1. 뉴런 이질성 부족(Neuron Heterogeneity)
기존 SNN 학습 방법은 모든 계층의 뉴런에 동일한 막 시간 상수를 설정합니다. 이는 뉴런의 다양성을 제한하고 SNN이 표현할 수 있는 복잡한 시간적 동역학을 감소시킵니다.[1]

#### 2. 초기값 민감도(Initialization Sensitivity)
고정된 τ 값이 최적하지 않으면 네트워크의 학습 성능이 급격히 저하됩니다. 표 4에서 보면, LIF 뉴런의 경우 τ = 16으로 설정했을 때 CIFAR-10에서의 정확도가 93.03%에서 47.50%로 급락합니다.[1]

#### 3. 시간적 정보 처리 효율성
현재 접근법은 뉴런이 입력의 시간적 변화에 적응하지 못하게 합니다. 막 시간 상수는 뉴런의 시간적 감수성을 결정하므로, 이를 학습하면 시간적 특징 추출이 개선됩니다.[1]

***

### 제안하는 방법 및 수식

#### 1. 시냅스 가중치와 막 시간 상수의 역할 분석

논문은 먼저 두 파라미터의 역할을 구분합니다:[1]

기본 LIF 뉴런 동역학:

$$\frac{dV(t)}{dt} = -\frac{V(t) - V_{rest}}{\tau} + X(t)$$

여기서:
- $$V(t)$$: 시간 t에서의 막 전위
- $$V_{rest}$$: 안정 전위
- $$\tau$$: 막 시간 상수
- $$X(t)$$: 입력

**핵심 통찰**: 시냅스 가중치 w를 증가시키면 정상 상태 전압 $$V_\infty = wI$$가 V 방향으로 증가하지만, 막 시간 상수 τ를 감소시키면 정상 상태 전압은 변하지 않고 **시간 방향으로 더 빠른 충방전을 만듭니다.** 이는 두 파라미터가 다양한 차원에서 뉴런의 반응을 조절할 수 있음을 의미합니다.[1]

#### 2. Parametric Leaky Integrate-and-Fire (PLIF) 모델

이산 시간 표현으로 변환하면:[1]

$$H_t = V_{t-1} + \frac{1 - k_a}{\tau}(X_t - V_{t-1})$$

$$S_t = \Theta(H_t - V_{th})$$

$$V_t = H_t(1 - S_t) + V_{reset}S_t$$

기존 접근법의 문제점을 해결하기 위해, 논문은 다음과 같이 재구성합니다:[1]

$$H_t = V_{t-1} + k_a(X_t - V_{t-1})$$

여기서 $$k_a$$는 클램프 함수:

$$k_a = \text{clamp}(a, 0, 1)$$

실제 구현에서는 시그모이드 활성화 함수를 사용합니다:

$$k_a = \frac{1}{1 + e^{-a}}$$

이렇게 하면:
- 수치 안정성 보장
- $$dt < \tau$$ 조건 자동 만족
- 막 시간 상수가 분모에 오지 않아 직접 최적화 가능

#### 3. 역전파 알고리즘

손실 함수는 평균 제곱 오차(MSE):[1]

$$L_{MSE} = \frac{1}{T}\sum_{t=0}^{T-1} \frac{1}{C}\sum_{i=0}^{C-1}(o_{t,i} - y_{t,i})^2$$

예측 레이블: $$l_p = \arg\max_i \frac{1}{T}\sum_{t=0}^{T-1} o_{t,i}$$

그래디언트 계산:[1]

$$\frac{\partial L}{\partial a_i} = \sum_{t=0}^{T-1} \frac{\partial L}{\partial H_t^i} \cdot \frac{\partial H_t^i}{\partial a_i}$$

체인 룰을 통한 시간적 역전파:

$$\frac{\partial L}{\partial H_t^i} = \frac{\partial L}{\partial H_{t+1}^i} \cdot \frac{\partial H_{t+1}^i}{\partial V_t^i} + \frac{\partial L}{\partial S_t^i} \cdot \frac{\partial S_t^i}{\partial H_t^i}$$

Surrogate 그래디언트 적용:

$$\frac{\partial S_t}{\partial H_t} = \sigma'(H_t)$$

여기서 $$\sigma(x) = \frac{1}{1 + (1 + \tan\pi x)^2}$$는 surrogate 함수입니다.[1]

#### 4. Max-pooling in SNNs

기존 접근법과 달리, 논문은 평균 풀링이 아닌 **최대값 풀링을 제안**합니다:[1]

평균 풀링: 모든 뉴런의 출력을 평균화
$$\text{AvgPool} = \frac{1}{n}\sum_{i=1}^{n} S_i$$

최대값 풀링: winner-take-all 메커니즘
$$\text{MaxPool} = \max(S_1, S_2, \ldots, S_n)$$

**장점**:
- 비동기 스파이킹 특성 보존
- 이진 출력 유지 (하드웨어 호환성)
- 동적 연결 제어 (시간적 위상-주파수 응답 향상)

***

### 모델 구조

네트워크는 인코더-분류기 구조입니다:[1]

```
입력 이미지 
    ↓
Spiking Encoder (특징 추출)
  - Conv2d + BN + PLIF 뉴런 반복
  - Max-pooling (stride 2)
    ↓
분류기 (Classifier)
  - Fully Connected + PLIF 뉴런
    ↓
투표층 (Voting Layer)
    ↓
출력 레이블
```

각 계층의 PLIF 뉴런들은 **공유하는 막 시간 상수**(같은 계층 내)를 가지지만, **계층 간에는 서로 다른 값**을 학습합니다.[1]

이 설계의 이유:
1. 생물학적 타당성: 인접 뉴런은 유사한 특성 보유
2. 계산 효율성: 계층당 1개의 스칼라 파라미터만 학습
3. 표현력 증대: 계층 간 다양한 시간 역학

***

### 성능 향상

#### 1. 최첨단 정확도 달성

표 2에서 보이는 성과:[1]

| 데이터셋 | 정확도 | 개선 사항 |
|---------|------|---------|
| MNIST | 99.72% | SOTA |
| Fashion-MNIST | 94.38% | SOTA |
| CIFAR-10 | 93.50% | 준SOTA (ANN2SNN: 93.63% vs 직접학습 비교) |
| N-MNIST | 99.61% | SOTA |
| CIFAR10-DVS | 74.80% | SOTA (62.5% 향상) |
| DVS128 Gesture | 97.57% | SOTA |

#### 2. 시간 스텝 감소

표 3의 추론 시간 스텝 비교:[1]

| 데이터셋 | 기존 SOTA | 제안 방법 | 개선율 |
|---------|---------|---------|------|
| CIFAR-10 | 2048 스텝 | 8 스텝 | 256배 감소 |
| N-MNIST | 64 스텝 | 10 스텝 | 6.4배 감소 |
| CIFAR10-DVS | 230-292 스텝 | 20 스텝 | 11.6-14.6배 감소 |

이는 **에너지 소비와 지연 시간을 극적으로 감소**시킵니다.[1]

#### 3. 초기값 안정성

그림 6과 표 4의 결과:[1]

PLIF 뉴런은 초기 τ₀ 값과 무관하게 훈련 중 최적값으로 수렴합니다. 반면 LIF 뉴런은:
- τ = 2에서 93.03% (CIFAR-10)
- τ = 16에서 47.50% (CIFAR-10) - 45.53% 성능 하락

**PLIF는 이러한 초기값 변화에 강건**합니다.[1]

#### 4. 수렴 속도 향상

그림 6의 훈련 곡선에서 PLIF는 LIF보다 더 빠르고 안정적인 수렴을 보입니다.[1]

***

### 한계

#### 1. 정적 이미지 데이터셋에서의 성능

CIFAR-10에서 ANN2SNN 변환 방법(93.63%)에 비해 0.13% 낮은 93.50% 달성했습니다. 이는:[1]
- 정적 이미지는 시간적 역동성을 충분히 활용하지 못함
- ANN2SNN은 rate-coding에 최적화됨

#### 2. 후층 뉴런의 비누수 통합(Non-Leaky Integrate)

그림 7에서 관찰된 현상으로, 일부 후층 뉴런들의 막 시간 상수가 무한대로 수렴합니다. 이는:[1]
- 뉴런이 Non-Leaky Integrate-and-Fire로 변환
- 이러한 뉴런의 역할과 최적화 의미에 대한 추가 연구 필요

#### 3. 제한된 이론적 분석

논문은 주로 경험적 검증에 집중하며, 다음에 대한 이론적 근거 부족:
- 왜 PLIF가 초기값 민감도를 감소시키는가?
- 막 시간 상수 학습의 최적성 보장

#### 4. 뉴로모르픽 하드웨어 구현

논문에서는 GPU 기반 시뮬레이션만 제시하며, 실제 뉴로모르픽 칩(예: Loihi, Spiking Jelly)에서의 구현 가능성을 명시하지 않습니다.[1]

***

### 일반화 성능 향상 가능성

#### 1. 신경 네트워크 표현력 강화

막 시간 상수 학습을 통해 각 계층이 **서로 다른 시간 스케일의 특징을 캡처**할 수 있습니다:[1]
- 초층: 짧은 τ (빠른 시간 변화, 엣지 검출)
- 중층: 중간 τ (중기 의존성)
- 후층: 긴 τ (장기 시간적 맥락)

이는 RNN의 LSTM 게이트 메커니즘과 유사한 역할을 수행합니다.

#### 2. 과적합 완화

Ablation study 결과에서:[1]
- PLIF는 다양한 초기 조건에서도 안정적 성능
- LIF는 최적 τ 선택이 과적합을 결정

$$P(\text{overfitting}) \propto \sigma(\tau_{\text{mismatch}})$$

#### 3. 도메인 적응성

신경모픽 데이터셋과 정적 이미지 모두에서 우수한 성능:
- 도메인별 최적의 τ 자동 학습
- Transfer learning 시 τ 재조정 가능

#### 4. 최신 연구에서의 확장

2024-2025년 최신 연구들이 이 아이디어를 확장하고 있습니다:[2][3][4]

**시간적 정규화 훈련(TRT, 2024)**: PLIF의 개념을 발전시켜 시간 의존적 정규화 메커니즘 추가

$$L_{TRT} = \frac{1}{T}\sum_{t=1}^{T} \left( L_{loss}(t) + r(t) \right)$$

여기서 $$r(t) = \sum_{i=1}^{l-1} \frac{\lambda}{1 + (|W_{i,i+1}| + \epsilon) \cdot (e^{\delta \cdot t} - 1)} \odot W_{i,i+1}^2$$

초기 시간 스텝에 더 강한 제약을 가하여 일반화 성능 향상.[3]

**적응형 그래디언트 학습(MPD-AGL, 2025)**: 막 전위 역학(Membrane Potential Dynamics)을 활용하여 surrogate 그래디언트를 동적으로 조정하는 방법 제안.[2]

**시간적 계층 구조(Temporal Hierarchy, 2024)**: 신경과학 영감의 시간적 계층 구조를 뉴런 시간 상수에 도입하여 성능 향상.[4]

***

### 앞으로의 연구 시 고려할 점

#### 1. 이론적 근거 강화

**필요 연구**:
- PLIF 최적화의 수렴성 증명
- 막 시간 상수 학습이 선형성과 표현력에 미치는 영향 분석
- Neural Tangent Kernel 이론 확장[5]

#### 2. 뉴로모르픽 하드웨어 적응

**실제 구현 과제**:
- Loihi 2, SpiNNaker 등 칩에서의 PLIF 구현
- 학습 가능한 τ의 온칩(on-chip) 저장 및 업데이트
- 저정밀도(low-precision) 양자화 영향 평가[6]

#### 3. 다중 시간 스케일 학습

**향상된 구조**:
- 계층당 여러 τ 값 학습 (τ 풀, pool)
- 뉴런별 독립 τ 학습 (계산 비용 증가)
- 계층 간 계층적 시간 구조 강제[4]

#### 4. Transfer Learning과 일반화

**미해결 문제**:
- 다른 데이터셋/도메인으로의 τ 전이
- 미지의 시간 역학을 가진 작업에서의 적응 학습
- 최근 temporal flexibility 연구[7]

#### 5. 에너지 효율성 정밀 분석

**측정 필요**:
- 실제 뉴로모르픽 하드웨어에서의 전력 소비
- τ 학습 오버헤드 vs. 시간 스텝 감소 이득 비교
- 프로토콜별(DVS vs. 정적) 에너지 프로파일 분석[8]

#### 6. 적대적 견고성

**새로운 도전**:
- PLIF의 초기값 안정성이 적대적 견고성에 미치는 영향
- 시간적 정보와 rate 정보의 균형[9]
- Surrogate 그래디언트와 막 시간 상수의 상호작용[10]

#### 7. 다양한 뉴런 모델 확장

**대안 모델**:
- Adaptive Leaky Integrate-and-Fire (ALIF)와 결합
- 적응형 임계값(Adaptive Threshold)과 함께
- CLIF(Complementary LIF)의 시간적 그래디언트 개선[11]

***

## 결론

"Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural Networks"는 SNN 연구에 중요한 기여를 했습니다. PLIF 뉴런의 도입으로 **뉴런의 이질성을 증가**시켜 표현력을 향상시켰고, **초기값 민감도를 감소**시켜 훈련을 안정화했으며, **추론 시간을 256배까지 단축**하는 놀라운 성과를 달성했습니다.[1]

특히 2024-2025년의 후속 연구들은 이 기초 위에서 시간적 정규화, 적응형 그래디언트, 시간적 계층 구조 등의 고급 기법들을 개발하고 있으며, 온칩 구현과 에너지 효율성에 대한 실무적 진전이 이루어지고 있습니다.[3][7][2][4]

앞으로의 연구는 이론적 기반 강화, 실제 뉴로모르픽 하드웨어 통합, 그리고 다양한 시간적 동역학을 가진 실세계 작업으로의 확장에 초점을 맞춰야 할 것입니다.

***

### 참고 문헌

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/1476338c-9c63-41c1-92ca-0de37d1dc159/2007.05785v5.pdf)
[2](https://www.ijcai.org/proceedings/2025/0464.pdf)
[3](https://arxiv.org/html/2506.19256v3)
[4](https://www.research-collection.ethz.ch/server/api/core/bitstreams/9cedcaf7-c3a7-4915-9bd5-6d99ad90a183/content)
[5](http://arxiv.org/pdf/2405.15539.pdf)
[6](https://arxiv.org/pdf/2304.12760.pdf)
[7](https://openreview.net/forum?id=9HsfTgflT7)
[8](https://ieeexplore.ieee.org/document/10423179/)
[9](https://openreview.net/pdf?id=xv8iGxENyI)
[10](http://arxiv.org/pdf/2503.03272.pdf)
[11](https://arxiv.org/abs/2402.04663)
[12](https://www.frontiersin.org/articles/10.3389/fnins.2024.1383844/full)
[13](https://arxiv.org/pdf/1901.09948.pdf)
[14](https://www.semanticscholar.org/paper/93a119b3af0322d729ee39da6be879e44d3ebf88)
[15](https://iopscience.iop.org/article/10.1088/1741-2552/ad731f)
[16](http://journal.frontiersin.org/Article/10.3389/fncom.2015.00145/abstract)
[17](http://journal.frontiersin.org/article/10.3389/fncom.2010.00007/abstract)
[18](https://arxiv.org/pdf/2204.07050.pdf)
[19](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1406502/pdf)
[20](http://arxiv.org/pdf/2406.03287.pdf)
[21](https://arxiv.org/pdf/2409.02111.pdf)
[22](https://arxiv.org/pdf/2403.00270.pdf)
[23](https://pmc.ncbi.nlm.nih.gov/articles/PMC10452895/)
[24](https://arxiv.org/pdf/2309.04426.pdf)
[25](https://arxiv.org/pdf/2407.05262v2.pdf)
[26](https://arxiv.org/abs/2303.13077)
[27](https://arxiv.org/abs/2007.05785)
[28](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1383844/full)
[29](https://arxiv.org/html/2509.05356v2)
[30](https://www.osti.gov/servlets/purl/1760126)
[31](https://www.sciencedirect.com/science/article/abs/pii/S0031320324008458)
[32](https://arxiv.org/abs/2406.19230)
[33](https://arxiv.org/abs/2409.07776)
[34](https://www.nature.com/articles/s41467-024-51641-x)
[35](https://advanced.onlinelibrary.wiley.com/doi/10.1002/aisy.202400075)
[36](https://ieeexplore.ieee.org/document/10672817/)
[37](https://ieeexplore.ieee.org/document/10865244/)
[38](https://ieeexplore.ieee.org/document/10650059/)
[39](https://arxiv.org/html/2404.08786v1)
[40](https://arxiv.org/pdf/2202.00282.pdf)
[41](http://arxiv.org/pdf/2404.14024.pdf)
[42](http://arxiv.org/pdf/2406.19645.pdf)
[43](http://arxiv.org/pdf/2402.04663.pdf)
[44](https://arxiv.org/html/2508.11279v1)
[45](https://www.themoonlight.io/ko/review/enhancing-generalization-of-spiking-neural-networks-through-temporal-regularization)
[46](https://arxiv.org/html/2505.11863v1)
[47](https://proceedings.mlr.press/v202/hemachandra23a/hemachandra23a.pdf)
[48](https://zenkelab.org/wp-content/uploads/2022/10/rossbroich_gygax_2022.pdf)
[49](https://cvpr.thecvf.com/virtual/2025/poster/34574)
