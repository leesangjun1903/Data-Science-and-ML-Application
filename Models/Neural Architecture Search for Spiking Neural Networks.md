# Neural Architecture Search for Spiking Neural Networks

### 1. 핵심 주장과 주요 기여 (요약)

본 논문의 핵심 주장은 **기존 SNN 연구가 ANN 아키텍처(VGG, ResNet 등)를 그대로 차용하고 있으며, 이는 SNNs의 시간적 특성을 충분히 활용하지 못해 최적화되지 않은 구조**라는 것입니다.[1]

주요 기여는 다음과 같습니다:[1]

- **SNN 특화 NAS 기법 개발**: 최초로 이미지 인식 작업을 위한 SNN 아키텍처 탐색 기법을 제시했습니다.

- **학습 없는 아키텍처 탐색**: 초기화된 네트워크의 시간적 활성화 패턴을 비교하여 최적 아키텍처를 선택하므로, SNNs의 긴 학습 시간 문제를 해결합니다.

- **Sparsity-Aware Hamming Distance (SAHD) 제안**: LIF 뉴런의 큰 스파시티 변동을 고려한 거리 측정 지표를 도입했습니다.

- **역방향 연결(Backward Connections) 탐색**: ANN과 달리 SNNs는 시간 정보를 시간 역행을 통해 전달할 수 있으며, 역방향 연결 탐색이 정확도를 최대 3% 향상시킵니다.

***

### 2. 해결 문제, 제안 방법, 모델 구조 및 성능

#### 2.1 해결하고자 하는 핵심 문제

SNNs는 이벤트 기반 계산으로 에너지 효율적이지만 다음과 같은 문제가 있습니다:[1]

1. **아키텍처 갭(Architectural Gap)**: 주로 인간이 설계한 ANN 아키텍처를 그대로 사용하여 SNNs의 특성에 맞지 않습니다.
2. **학습 비용**: SNNs의 학습은 ANNs에 비해 약 **11.43배 더 오래 걸립니다**.
3. **아키텍처 탐색의 어려움**: 기존 NAS 기법은 여러 학습 단계를 요구하여 SNNs에 적용 불가능합니다.

#### 2.2 제안하는 핵심 방법론

##### 2.2.1 LIF 뉴런 모델

논문의 기초가 되는 Leaky Integrate-and-Fire(LIF) 뉴런의 이산 시간 모델:[1]

$$u_i^t = \left(1 - \frac{1}{\tau_m}\right) u_i^{t-1} + \frac{1}{\tau_m} \sum_j w_{ij}o_j^t$$

여기서:
- $u_i^t$: 시간 $t$에서 뉴런 $i$의 막전위(membrane potential)
- $\tau_m$: 막전위 감쇠의 시간 상수
- $w_{ij}$: 뉴런 $j$에서 $i$로의 가중치
- $o_j^t$: 시간 $t$에서 뉴런 $j$의 스파이크 출력

##### 2.2.2 Sparsity-Aware Hamming Distance (SAHD)

**핵심 혁신**: 기존 Hamming Distance(HD)는 LIF 뉴런의 스파시티 변동을 고려하지 못합니다.

스파시티 $r_i^l$을 이용한 확률 모델링:[1]

$$o_i^l \sim \text{Bern}(1-r_i^l)$$

두 샘플 간 활성화 차이 확률:[1]

$$\text{Pr}(|o_i^l - o_j^l| = 1) = \text{Bern}(r_i^l(1-r_j^l) + (1-r_i^l)r_j^l)$$

HD의 기댓값:[1]

$$\mathbb{E}[d_H(c_i^l, c_j^l)] = N_A^l \{r_i^l(1-r_j^l) + (1-r_i^l)r_j^l\}$$

**제안된 SAHD**:[1]

$$d_{SAH}(c_i^l, c_j^l) = \frac{\alpha}{N_A^l\{r_i^l(1-r_j^l) + (1-r_i^l)r_j^l\}} d_H(c_i^l, c_j^l)$$

이를 통해 커널 행렬을 구성합니다:[1]

$$\mathrm{K}_H = \begin{pmatrix} N_A - d_H(c_1, c_1) & \cdots & N_A - d_H(c_1, c_N) \\ \vdots & \ddots & \vdots \\ N_A - d_H(c_N, c_1) & \cdots & N_A - d_H(c_N, c_N) \end{pmatrix}$$

최종 아키텍처 점수:[1]

$$s = \log(|\det|\sum_l \mathrm{K}_H^l|)$$

##### 2.2.3 학습 없는 아키텍처 탐색

기존 NASWOT 접근법을 SNNs에 맞게 확장했습니다. 핵심은 초기 네트워크가 시간 경과에 따라 다양한 스파이크 활성화 패턴을 표현할 수 있는 능력을 측정하는 것입니다. 이는 HD 대신 SAHD를 사용함으로써 정확도 상관계수(Kendall's τ)가 **0.519에서 0.646으로 향상**됩니다.[1]

#### 2.3 모델 구조

**SNASNet의 매크로 구조**:[1]

1. **Spike Encoding Layer**: 입력 이미지를 스파이크로 변환 (Direct Encoding)
2. **First Neuron Cell**: C 채널 입력/출력
3. **Reduction Cell**: Conv(C, 2C)-BN(2C)-LIF + AvgPool(2)
4. **Second Neuron Cell**: 2C 채널 입력/출력  
5. **Vectorize Block**: AvgPool(2) + Vectorization
6. **Classifier**: Dropout(0.5)-FC(1024)-Voting Layer

**탐색 공간**:[1]

각 셀은 V=4개의 노드를 포함하며, 작업 집합은:
$$O = \{\text{Zeroize, Skip Connection, 1×1 Conv, 3×3 Conv, 3×3 AvgPool}\}$$

**혁신적 특징 - 역방향 연결 탐색**: 기존 NAS는 순방향 연결만 탐색하지만, 본 논문은 시간을 통해 정보를 전달하는 **역방향 연결(Backward Connections)**도 탐색합니다. 이는 $l'$층(시간 $t-1$)의 특성을 $l$층(시간 $t$, $l' < l$)에 추가하는 구조입니다.[1]

#### 2.4 성능 향상

실험 결과 (5 timesteps):[1]

| 데이터셋 | 모델 | 정확도(%) | 이전 SOTA 대비 |
|---------|------|----------|--------------|
| CIFAR10 | SNASNet-Fw | 93.12 ± 0.42 | 비교 가능 |
| CIFAR10 | **SNASNet-Bw** | **93.73 ± 0.32** | **+0.23% (SOTA)** |
| CIFAR100 | SNASNet-Fw | 70.06 ± 0.45 | 비교 가능 |
| CIFAR100 | **SNASNet-Bw** | **73.04 ± 0.36** | **+2.98% vs Fw** |
| TinyImageNet | SNASNet-Fw | 52.81 ± 0.56 | 최고 성능 |
| TinyImageNet | **SNASNet-Bw** | **54.60 ± 0.48** | **+1.79% vs Fw** |

특히 복잡한 데이터셋에서 역방향 연결의 효과가 두드러집니다.

#### 2.5 주요 한계

1. **제한된 탐색 공간**: NAS-Bench-201 기반으로 설계되어 네트워크 크기가 제한적입니다.

2. **Timestep 의존성**: 최적 성능은 더 많은 timestep에서 달성되지만, 에너지 효율성과 트레이드오프가 발생합니다.

3. **일반화 격차**: 순방향 연결만으로는 여전히 복잡 데이터셋에서 성능이 제한적입니다 (CIFAR100에서 70.06%).

4. **하드웨어 제약**: 역방향 연결이 모든 신경망 하드웨어에서 효율적으로 구현되지 않을 수 있습니다.

***

### 3. 일반화 성능 향상 가능성

#### 3.1 전이 가능성(Transferability) 분석

논문의 주목할 만한 발견은 검색된 아키텍처의 높은 **전이 가능성**입니다.[1]

아키텍처를 데이터셋 A에서 탐색한 후 데이터셋 B에서 평가한 결과:

- **순방향 연결만**: ∆Acc = -0.56% ~ +0.11% (매우 작은 편차)
- **역방향 연결**: ∆Acc = -0.80% ~ +0.59% (약간 더 큼)

이는 **검색된 SNN 아키텍처가 대용량 및 복잡한 데이터셋에서 탐색 시간을 제거할 수 있음**을 시사합니다.

#### 3.2 아키텍처 속성 분석

광범위한 아키텍처 속성 분석(100개 무작위 탐색, CIFAR100)으로부터:[1]

**순방향 연결에서의 관찰**:
- 더 깊고 넓은 셀이 성능 향상 (SNNs에서 스케일링의 중요성)
- 합성곱 계층이 중요, 평균 풀링은 권장되지 않음

**역방향 연결에서의 관찰**:
- 소수의 역방향 연결이 선호됨 (1-2개)
- 2개 이상의 역방향 Skip 연결은 정확도를 크게 저하시킴
- 변환 없는 피드백(Skip 연결)보다는 합성곱/풀링 연산을 포함한 역방향 연결이 유리

#### 3.3 Timestep과의 상호작용

시간 스텝에 따른 정확도 변화:[1]

| Timestep | SNASNet-Fw | SNASNet-Bw | 차이 |
|----------|-----------|-----------|------|
| 5 | 70.06% | 73.04% | +2.98% |
| 10 | 70.08% | 73.46% | +3.38% |
| 15 | 70.56% | 73.49% | +2.93% |
| 20 | 70.52% | 74.24% | +3.72% |

**중요한 발견**: SNASNet-Bw는 더 많은 timestep에서 **더 큰 성능 향상**을 보이며, 이는 역방향 연결이 **시간 정보를 더 효과적으로 활용**함을 의미합니다.

#### 3.4 거리 지표의 영향

SAHD vs HD의 성능 비교:[1]

| 아키텍처 | HD | SAHD |
|---------|----|----|
| SNASNet-Fw | 64.16 ± 2.02% | 70.06 ± 0.45% |
| SNASNet-Bw | 66.80 ± 1.73% | 73.04 ± 0.36% |

**SAHD 사용으로 표준 편차가 약 4-5배 감소**하여, 더 안정적이고 신뢰할 수 있는 아키텍처 선택이 가능합니다.

***

### 4. 논문의 영향과 향후 연구 고려사항

#### 4.1 이 논문이 미치는 영향

**현재(2024-2025) SNN 분야의 발전:**

1. **하드웨어 인식 NAS로 확장**: 최근 연구들(SpikeNAS, NeuroNAS)은 본 논문의 학습 없는 접근법을 기반으로 하드웨어 제약(메모리, 지연, 에너지)을 고려한 효율적 NAS로 확장했습니다.[2][3][4]

2. **시공간 최적화로 진화**: Spatial-Temporal Search 논문(2024)은 본 논문의 공간적 탐색을 넘어 **시간 상수와 뉴런 동역학까지 최적화**하는 방향으로 발전시켰습니다.[5]

3. **다중 스케일 진화 탐색**: MSE-NAS(2023)는 생물학적 뇌 구조의 미시(neuron), 중시(circuit motif), 거시(global connectivity) 수준을 모두 고려한 아키텍처 탐색을 제시했습니다.[6]

4. **저용량 SNN 설계**: LitE-SNN은 공간적, 시간적 압축을 결합하여 **엣지 디바이스용 경량 SNN** 설계를 가능하게 했습니다.[7]

5. **Spiking Transformer 탐색**: AutoST(2023)는 본 논문의 학습 없는 접근법을 **Transformer 아키텍처**에 적용했습니다.[8]

#### 4.2 향후 연구 시 고려할 점

##### 4.2.1 일반화 성능 향상 관점

1. **더 큰 탐색 공간의 필요성**: 현재 NAS-Bench-201 기반 탐색 공간은 제한적입니다. 최근 연구는 더 다양한 작업 집합(예: Depthwise Separable Convolution, Group Convolution)을 포함해야 합니다.[9]

2. **신경망 초기화 이론 고도화**: 본 논문의 선형 영역 개념을 더 정교하게 발전시켜, SNNs의 비선형성과 시간적 특성을 더 잘 반영할 필요가 있습니다.

3. **다중 데이터셋 공동 최적화**: 현재 전이성은 제한적이므로, 여러 데이터셋에 동시에 최적화된 아키텍처 탐색 기법 개발이 필요합니다.

##### 4.2.2 시간 동역학 활용

1. **적응형 Time Constant 탐색**: LIF 뉴런의 시간 상수 $\tau_m$을 아키텍처 탐색의 일부로 포함시키면 성능이 향상될 것으로 예상됩니다.

2. **역방향 연결의 최적 설계**: 역방향 연결이 단순한 정보 전달을 넘어, **시간적 주의 메커니즘(Temporal Attention)**으로 발전될 수 있습니다.[10]

3. **혼합 정밀도(Mixed Precision) 최적화**: 시간 스텝에 따라 다른 정밀도를 사용하는 아키텍처 탐색이 에너지 효율성을 크게 향상시킬 수 있습니다.

##### 4.2.3 신경망 안정성 및 신뢰성

1. **학습 안정성 개선**: 최근 연구는 시간적 일관성(Temporal Consistency) 강화를 통해 SNN 학습 안정성을 크게 개선했습니다.[11]

2. **적대적 강건성**: 검색된 아키텍처가 적대적 공격에 견디는 성능에 대한 분석이 필요합니다.

3. **불확실성 정량화**: 베이지안 NAS 접근법을 SNNs에 적용하여 아키텍처 선택의 신뢰도를 정량화해야 합니다.

##### 4.2.4 하드웨어-소프트웨어 공설계

1. **신경형 하드웨어 최적화**: Loihi 칩과 같은 신경형 컴퓨팅 하드웨어의 특성을 고려한 NAS 개발이 진행 중입니다.[2]

2. **메모리-지연 트레이드오프**: 임베디드 시스템의 메모리 제약 하에서 최소 지연을 달성하는 아키텍처 탐색이 중요합니다.[3]

3. **동적 전력 소비 모델링**: 단순 FLOPs가 아닌 실제 **스파이크 율(Spike Rate)**을 고려한 에너지 소비 모델이 필요합니다.

##### 4.2.5 이론적 토대 강화

1. **표현 용량 분석**: SNNs의 선형 영역 개수와 표현 능력의 관계를 더 정밀하게 분석해야 합니다.

2. **뉴런 전이 다양성**: LIF 뉴런의 시간 변화하는 전이점으로 인한 다양성을 수학적으로 모델링할 필요가 있습니다.

3. **수렴 이론**: 검색된 아키텍처의 학습 동역학 수렴성에 대한 이론적 보장이 필요합니다.

***

### 5. 결론

본 논문은 **SNN 아키텍처 설계에 대한 혁신적 접근법**을 제시했습니다. 학습 없는 아키텍처 탐색, SAHD 지표, 그리고 역방향 연결 탐색을 통해 SNNs의 시간적 정보 활용을 혁신적으로 개선했습니다.

최근(2024-2025) 연구 동향은 본 논문의 기초 위에서:
- **하드웨어 제약 통합**
- **시간 동역학 최적화**
- **다중 스케일 설계**
- **저용량 구현**

등으로 발전하고 있습니다. 특히 역방향 연결의 효과와 SAHD의 유효성은 향후 SNN 연구의 핵심 방향을 제시하며, 신경형 컴퓨팅의 실용적 배포를 가속화하는 중요한 이정표가 되었습니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/64330e2e-a867-4a07-87d2-a935453305f5/2201.10355v3.pdf)
[2](https://ieeexplore.ieee.org/document/10528706/)
[3](https://ieeexplore.ieee.org/document/11071976/)
[4](https://www.semanticscholar.org/paper/3dea338a3671053c65997647048aa70ee8fa7a2b)
[5](https://arxiv.org/abs/2410.18580)
[6](https://arxiv.org/abs/2304.10749)
[7](https://arxiv.org/abs/2401.14652)
[8](https://ieeexplore.ieee.org/document/10445971/)
[9](https://arxiv.org/html/2510.14235v1)
[10](http://arxiv.org/pdf/2411.00902.pdf)
[11](https://www.sciencedirect.com/science/article/abs/pii/S0031320324008458)
[12](https://linkinghub.elsevier.com/retrieve/pii/S0925231224009524)
[13](https://linkinghub.elsevier.com/retrieve/pii/S0893608024000960)
[14](https://arxiv.org/abs/2402.11322)
[15](https://arxiv.org/html/2402.11322v2)
[16](https://arxiv.org/abs/2201.10355)
[17](https://arxiv.org/pdf/2409.02111.pdf)
[18](https://arxiv.org/html/2410.18580v1)
[19](https://arxiv.org/abs/2312.01213)
[20](http://arxiv.org/pdf/2406.02923.pdf)
[21](https://www.frontiersin.org/articles/10.3389/fncom.2023.1215824/pdf)
[22](https://www.sciencedirect.com/science/article/abs/pii/S0893608024000960)
[23](https://proceedings.neurips.cc/paper_files/paper/2024/file/b8bf2c0dd0b48511889b7d3b2c5fc8f5-Paper-Conference.pdf)
[24](https://openreview.net/forum?id=4jEuiMPKSF)
[25](https://www.arxiv.org/abs/2510.14235)
[26](https://arxiv.org/html/2506.19256v3)
[27](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840036.pdf)
[28](https://dl.acm.org/doi/abs/10.1007/978-3-031-20053-3_3)
