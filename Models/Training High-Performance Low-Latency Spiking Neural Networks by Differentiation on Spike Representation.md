# Training High-Performance Low-Latency Spiking Neural Networks by Differentiation on Spike Representation

### 1. 핵심 주장 및 주요 기여

**"Training High-Performance Low-Latency Spiking Neural Networks by Differentiation on Spike Representation"** 논문은 SNN 훈련의 오랜 난제인 **고성능과 저지연성의 동시 달성** 문제를 해결합니다. 논문의 핵심 주장은 다음과 같습니다:[1]

기존의 두 가지 주류 훈련 방법들의 한계를 극복합니다:[2][1]
- **ANN-to-SNN 변환**: 높은 성능 달성 가능하나 극도로 높은 지연성(수천 시간 단계) 필요
- **Surrogate Gradient (SG) 방법**: 낮은 지연성 달성 가능하나 성능이 ANN에 미치지 못함

논문은 **Differentiation on Spike Representation (DSR)** 방법을 제안하여 이 trade-off를 해결합니다. 주요 기여는:[1]

1. **Spike Representation을 통한 체계적 분석**: 가중 발화율 코딩(weighted firing rate coding)을 사용하여 SNN의 순전파를 부분미분 가능(sub-differentiable) 매핑으로 표현[1]

2. **시간 도메인 역전파 제거**: 기존 BPTT(Backpropagation Through Time) 방법과 달리 시간 단계별 그래디언트 계산을 피하여 훈련 효율성 향상[1]

3. **표현 오차 감소 기법**: Spike threshold 학습과 새로운 신경 모델 하이퍼파라미터 도입으로 저지연성 환경에서의 성능 저하 완화[1]

***

### 2. 해결하는 문제와 제안 방법

#### 2.1 근본적인 문제

SNN 훈련의 주요 어려움은 **비미분 가능성**(non-differentiability)입니다. 스파이크 생성 함수가 불연속 Heaviside 함수이므로 표준 역전파가 적용 불가능합니다. 이는 다음 두 가지 문제를 초래합니다:[1]

- **높은 훈련 비용**: BPTT를 사용할 경우 공간(layer) 및 시간(timestep) 도메인 모두에서 역전파 필요
- **성능-지연성 trade-off**: 높은 성능 또는 낮은 지연성 중 하나만 달성 가능

#### 2.2 DSR 방법의 핵심

**Spike Representation** 정의:[1]

LIF 모델의 경우:
$$a_N^i = \frac{\sum_{n=1}^{N} \lambda_N^{-n} s_n^i}{V_{th}^i \sum_{n=1}^{N} \lambda_N^{-n} \Delta t}$$

여기서 $\lambda = e^{-\Delta t/\tau}$이고 $a_N^i$는 정규화된 가중 발화율입니다.[1]

**Sub-differentiable 매핑 표현**:[1]

$$o^i = \text{clamp}\left(\frac{1}{\lambda^i} W^i o^{i-1}, 0, \frac{V_{th}^i}{\Delta t}\right), \quad i=1,2,\ldots,L$$

이 매핑을 통해 SNN은 다음과 같이 표현됩니다:[1]

$$o^{i+1}_N \approx g_{W^i}(o^i_N)$$

여기서 $g_{W^i}$는 가중치 $W^i$로 매개변수화된 부분미분 가능 함수입니다.[1]

**역전파 알고리즘**:[1]

표현을 기반으로 한 그래디언트 계산:

$$\frac{\partial \mathcal{L}}{\partial W^i} = \frac{\partial o^{i+1}}{\partial o^i} \cdot \frac{\partial o^i}{\partial W^i}$$

#### 2.3 표현 오차 분석 및 감소 기법

논문은 유한 시간 단계에서의 표현 오차를 두 가지로 분해합니다:[1]

- **양자화 오차** ($e_q$): 불완전한 발화율 정밀도로 인한 오차
- **편차 오차** ($e_d$): 시간 단계 간 입력 전류의 불일치

**Spike Threshold 훈련**:[1]

표현 오차를 감소시키기 위해 각 층의 임계값을 학습 가능한 파라미터로 설정하고 L2 정규화 추가:

$$\frac{\partial a}{\partial V_{th}} = \begin{cases} -1 & \text{if } I \geq V_{th} \\ 0 & \text{otherwise} \end{cases}$$

**새로운 하이퍼파라미터 도입**:[1]

발화 메커니즘 수정:

$$s_n = H(U_n - \alpha V_{th})$$

여기서 $0 < \alpha < 1$은 양자화 오차를 감소시키는 하이퍼파라미터입니다. IF 모델의 경우 $\alpha = 0.5$일 때 최대 절대 양자화 오차가 절반으로 감소합니다.[1]

***

### 3. 모델 구조 및 아키텍처

#### 3.1 신경 모델

논문은 두 가지 표준 SNN 신경 모델을 사용합니다:[1]

**Integrate-and-Fire (IF) 모델**:
$$U_n = f(V_{n-1}, I_n) = V_{n-1} + I_n$$

**Leaky Integrate-and-Fire (LIF) 모델**:
$$U_n = f(V_{n-1}, I_n) = e^{-\Delta t/\tau} V_{n-1} + (1-e^{-\Delta t/\tau})I_n$$

이산화된 스파이크 생성:[1]
$$s_n = H(U_n - V_{th})$$
$$V_n = U_n - V_{th}s_n$$

#### 3.2 계층 역학

L층 피드포워드 SNN의 동역학:[1]

**IF 동역학**:
$$V_n^i = V_{n-1}^i + W^i s_{n-1}^i - V_{th}^i s_n^i$$

**LIF 동역학**:
$$V_n^i = e^{-\Delta t/\tau_i} V_{n-1}^i + (1-e^{-\Delta t/\tau_i}) W^i s_{n-1}^i - V_{th}^i s_n^i$$

#### 3.3 배치 정규화 (Batch Normalization)

SNN 훈련 안정화를 위해 시간 차원 배치 정규화 적용:[1]

입력 데이터 $x \in \mathbb{R}^{B \times N}$ (배치크기 $B$, 시간 단계 $N$)에 대해:

$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

여기서 평균과 분산은 시간-배치 결합 차원에서 계산됩니다.[1]

#### 3.4 네트워크 아키텍처

실험에 사용된 아키텍처:

- **CIFAR-10/100**: Pre-activation ResNet-18
- **ImageNet**: Pre-activation ResNet-18 (하이브리드 훈련 사용)
- **DVS-CIFAR10**: VGG-11

네트워크는 최대 풀링을 평균 풀링으로 대체하고, 풀링 후 및 마지막 완전연결층 후에 스파이킹 신경원층을 추가합니다.[1]

***

### 4. 성능 향상 및 실험 결과

#### 4.1 정적 데이터셋 성능

**CIFAR-10** (20시간 단계):[1]
- IF 모델: 95.24% ± 0.17
- LIF 모델: 95.40% ± 0.15
- 비교: ANN (1시간 단계) 95.41%

**CIFAR-100** (20시간 단계):[1]
- IF 모델: 78.20% ± 0.13
- LIF 모델: 78.50% ± 0.12
- 비교: ANN (1시간 단계) 78.12%
- 기존 방법 대비: 5-10% 성능 향상

**ImageNet** (50시간 단계):[1]
- IF 모델: 67.74%
- 기존 직접 훈련 방법보다 우수

**Neuromorphic 데이터셋 (DVS-CIFAR10, 20시간 단계)**:[1]
- IF 모델: 75.03% ± 0.39
- LIF 모델: 77.27% ± 0.24

#### 4.2 저지연성 성능 (Ultra-low Latency)

표준 ResNet-18 구조로 다양한 시간 단계에서 평가:[1]

| 시간 단계 | IF 정확도 | LIF 정확도 |
|----------|---------|---------|
| 20 | 95.24% | 95.40% |
| 15 | 94.85% | 95.39% |
| 10 | 94.69% | 94.47% |
| 5 | 94.48% | 94.42% |

5시간 단계에서도 1% 이내의 정확도 손실[1]

#### 4.3 깊은 네트워크 구조 성능

Pre-activation ResNet의 깊이별 성능 (CIFAR-10, 20시간 단계):[1]

| 네트워크 깊이 | IF 정확도 | LIF 정확도 |
|-------------|---------|---------|
| 20층 | 92.67% | 92.82% |
| 32층 | 93.73% | 93.74% |
| 44층 | 93.74% | 93.99% |
| 56층 | 94.15% | 94.03% |
| 110층 | 94.61% | 94.60% |

**중요 발견**: 깊어질수록 성능이 향상되는 특이한 현상으로 대규모 네트워크 확장의 잠재력 시사[1]

#### 4.4 표현 오차 감소 기법의 효과 (Ablation Study)

Pre-activation ResNet-18, IF 모델, CIFAR-10 (20시간 단계):[1]

| 설정 | 정확도 | 표준편차 |
|------|------|--------|
| 전체 DSR (F+T) | **95.24%** | 0.17 |
| 기법 없음 (초기 $V_{th}=6$) | 90.21% | - |
| T만 제거 ($V_{th}=6$) | 90.45% | 1.84 |
| T만 제거 ($V_{th}=2$) | 90.47% | 0.12 |
| F만 제거 ($V_{th}=6$) | 92.88% | 0.25 |

**분석**: Threshold 훈련(T)과 발화 메커니즘 수정(F) 모두 필수적으로 중요[1]

#### 4.5 발화 희소성 (Firing Sparsity)

에너지 효율성 평가를 위한 발화율:[1]

- 모든 층의 발화율: 20% 이하
- 많은 층: 5% 이하의 발화율
- 전체 평균 발화율: 시간 단계 개수와 무관하게 7.5-9.5%
- **중요**: 지연성 감소 시에도 발화율 증가 없음

#### 4.6 무게 양자화 (Weight Quantization)

실제 신경형 하드웨어 배포를 위한 저비트 무게:[1]

| 무게 정밀도 | IF 정확도 | LIF 정확도 |
|-----------|---------|---------|
| 32-bit | 95.38% | 95.63% |
| 8-bit | 95.45% | 95.65% |
| 4-bit | 95.31% | 95.39% |

**결론**: 무게 양자화에 매우 강인함[1]

***

### 5. 한계 및 제약사항

논문은 자체적으로 다음과 같은 한계를 인정합니다:[1]

#### 5.1 극도 저지연성 환경에서의 성능 저하

**문제**: 2-3개 시간 단계만 있을 경우 상당한 성능 저하[1]

**원인**: DSR 방법은 정확한 spike representation에 의존하므로, 극도로 제한된 시간 단계에서는 표현이 부족함[1]

**해결 가능성**: 추가적인 정규화 기법 또는 다른 훈련 패러다임 필요

#### 5.2 표현 오차의 이론적 분석 한계

완전한 오차 바운드를 제공하지 못함. Proposition 1과 2의 가정들:[1]

- 입력 전류가 충분히 일정할 것
- 잔여 막전위 $V_N^* \in [0, V_{th}]$로 제한될 것 (극단적 경우는 제외)

이 가정들이 항상 만족되지 않는 실제 데이터에서는 오차가 증가할 수 있음

#### 5.3 신경형 데이터에 대한 제한된 실험

- DVS-CIFAR10에서만 신경형 데이터 실험[1]
- 더 큰 규모의 신경형 데이터셋(DVS-ImageNet 등)에 대한 검증 미흡

#### 5.4 계산 복잡도 분석 부족

정확한 시간 및 메모리 복잡도 분석이 제한적. BPTT 대비 개선도가 명확히 정량화되지 않음[1]

***

### 6. 일반화 성능 향상 가능성

#### 6.1 현재 논문에서의 일반화 분석

논문은 일반화 성능에 대해 직접적 분석을 제시하지 않지만, 다음과 같은 간접적 증거를 제공합니다:[1]

1. **다양한 데이터셋 성능**: 정적(CIFAR-10/100, ImageNet) 및 신경형(DVS-CIFAR10) 모두에서 일관되게 우수한 성능
2. **깊은 네트워크에서의 개선**: 깊어질수록 성능 향상은 과적합이 아닌 실제 개선을 시사
3. **고정 발화 희소성**: 지연성 감소 시에도 발화율 증가 없음은 훈련-테스트 불일치 가능성 낮음

#### 6.2 최신 SNN 일반화 연구와의 연결

최근 연구에서 SNN의 일반화 문제를 다루고 있습니다:[3][4]

**Temporal Reversal Regularization (TRR, 2024)**:[3]
- SNN의 심각한 과적합 문제 인식
- 시간적 역전 섭동을 통한 정규화로 일반화 오차 상한 개선
- 저지연성 신경형 객체 인식에서 현저한 개선

**Temporal Regularization Training (TRT, 2024)**:[5]
- SNN이 신경형 데이터셋의 제한된 규모로 인한 심각한 과적합 경험
- 시간-종속 정규화 메커니즘 도입
- Fisher 정보 분석을 통한 시간적 정보 집중도 향상

**Noise-aided SNNs (NSNNs, 2023)**:[6]
- 내부 노이즈 도입으로 일반화 능력 향상
- DVS-CIFAR 및 DVS-Gesture에서 DSNNs 대비 월등한 성능 개선
- 제한된 샘플에서 과적합 완화

#### 6.3 DSR 방법의 일반화 개선 메커니즘

논문의 설계 특징이 일반화 개선에 기여하는 메커니즘:[1]

1. **Spike Threshold 훈련**:
   - 각 층이 입력 분포에 적응적 임계값 학습
   - L2 정규화로 과도한 임계값 증가 억제
   - 더 균형 잡힌 신경원 반응 범위

2. **$\alpha$ 하이퍼파라미터**:
   - 양자화 오차 감소로 더 정확한 표현
   - 신경형과 정적 데이터 모두에서 효과

3. **시간 단계 독립적 발화율**:
   - 훈련과 테스트 간 일관된 동역학
   - 시간 단계 변화에 대한 강건성

#### 6.4 향후 일반화 성능 향상 가능성

**단기적 개선 가능성**:
1. 기존 정규화 기법(Dropout, Batch Normalization 개선) 통합
2. 데이터 증강 기법 도입
3. 신경형 데이터셋 확대 실험

**중장기적 개선 가능성**:
1. **Temporal Reversal Regularization 통합**: TRR + DSR 결합으로 시간적 불변성 강화[4][3]
2. **시간-종속 정규화**: Temporal information dynamics 고려한 TRT 유사 기법[5]
3. **아키텍처 탐색**: 신경형 데이터에 최적화된 구조 자동 탐색
4. **Dendritic 신경원**: 다중 분지 동역학으로 표현력 증가[7]

***

### 7. 앞으로의 연구 영향 및 고려사항

#### 7.1 영향 분석

**학술적 영향**:[2][1]
- SNN 훈련의 근본적 난제 해결에 기여
- Spike representation 개념의 체계적 분석으로 새로운 연구 방향 제시
- 222회 인용(2022년 발표 이후)으로 커뮤니티에 큰 영향

**실무적 영향**:
- 실시간 신경형 하드웨어 배포 가능성 증대
- 에너지 효율 요구 응용에 직접 적용 가능
- 엣지 컴퓨팅 환경에서의 AI 실현

#### 7.2 최신 연구 동향에서의 위치

**2024-2025년 SNN 연구 동향**:[8][9][10][11][12]

1. **대규모 SNN 개발**: 1000층 이상의 깊은 SNN 구현[9]
2. **신경형 데이터 중심**: 성능과 에너지 효율 모두 추구[11][8]
3. **하이브리드 접근**: SNN과 ANN 결합으로 장점 통합[8]
4. **시간적 유연성**: 다양한 시간 단계에서의 일반화[12]
5. **양자화 최적화**: 초저비트 가중치 및 막전위[13]

#### 7.3 향후 연구 시 고려할 점

**기술적 고려사항**:

1. **극도 저지연성 해결**:
   - 2-3시간 단계에서의 성능 저하 극복 필요
   - 새로운 정보 인코딩 방식 탐색
   - 하이브리드 훈련 패러다임 통합

2. **일반화 성능 개선**:
   - 최신 정규화 기법(TRR, TRT) 통합[5][3]
   - 노이즈 활용 접근법 검토[6]
   - 시간적 변동성에 대한 강건성 강화

3. **신경형 하드웨어 배포**:
   - 다양한 신경형 칩(Loihi, TrueNorth, SpiNNaker 등)에 대한 최적화
   - 무게 양자화 기법 고도화
   - 온-칩 학습 가능성 탐색

4. **확장성**:
   - 1000층 이상의 초심층 SNN 훈련[9]
   - ImageNet, COCO 등 대규모 데이터셋 확대
   - 멀티모달 데이터 처리(비전, 오디오, 센서 융합)[14]

**연구 방향**:

1. **이론적 해석**:
   - 표현 오차의 엄밀한 바운드 도출
   - 수렴성 분석 강화
   - 일반화 경계(generalization bound) 이론화

2. **새로운 아키텍처**:
   - Dendritic SNN으로 표현력 향상[7]
   - 주의 메커니즘 통합
   - 신경형 데이터 최적 구조 설계

3. **멀티태스크 학습**:
   - 단일 훈련 네트워크로 다양한 작업 처리
   - 연속 학습(continual learning) 환경 적응
   - 도메인 일반화 개선

4. **실시간 응용 개발**:
   - 자율주행, 로봇제어 등 실제 응용
   - 생의학 신호 처리(EEG, EMG)
   - 웨어러블 장치 배포

#### 7.4 협력 연구 기회

- 신경형 하드웨어 개발 팀과의 협업
- 신경과학 연구와의 생물학적 타당성 검증
- 산업 파트너와의 실제 응용 개발

***

### 결론

**Differentiation on Spike Representation** 방법은 SNN 연구에서 획기적인 기여를 제시합니다. 고성능과 저지연성의 오랫동안의 trade-off를 혁신적으로 해결함으로써, 에너지 효율적인 AI 시대의 도래를 가능하게 합니다. 

향후 연구는 극도 저지연성 환경 강화, 일반화 성능 개선, 신경형 하드웨어 최적화에 초점을 맞춰야 하며, 최신 정규화 기법 통합과 깊은 네트워크 확장을 통해 더욱 강력한 SNN 시스템 구현이 기대됩니다.[12][3][5][1]

***

**주요 참고 문헌**:

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/d763a0ed-4afa-4e51-8ca5-72b9c12b2f2a/2205.00459v2.pdf)
[2](https://openaccess.thecvf.com/content/CVPR2022/papers/Meng_Training_High-Performance_Low-Latency_Spiking_Neural_Networks_by_Differentiation_on_Spike_CVPR_2022_paper.pdf)
[3](https://www.semanticscholar.org/paper/1c99561dd2a11d9bb93aa7dc75d52274a0efe018)
[4](https://arxiv.org/abs/2408.09108)
[5](https://arxiv.org/html/2506.19256v3)
[6](https://pmc.ncbi.nlm.nih.gov/articles/PMC10591140/)
[7](https://arxiv.org/abs/2412.06355)
[8](https://ieeexplore.ieee.org/document/10322581/)
[9](https://arxiv.org/pdf/2409.02111.pdf)
[10](https://www.frontiersin.org/articles/10.3389/fnins.2025.1536771/full)
[11](https://www.themoonlight.io/ko/review/neuromorphic-sequential-arena-a-benchmark-for-neuromorphic-temporal-processing)
[12](https://arxiv.org/html/2503.17394v1)
[13](https://www.frontiersin.org/articles/10.3389/fnins.2024.1440000/full)
[14](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1623497/pdf)
[15](https://ieeexplore.ieee.org/document/10835501/)
[16](https://ieeexplore.ieee.org/document/10647018/)
[17](https://ieeexplore.ieee.org/document/10771076/)
[18](https://ieeexplore.ieee.org/document/10650857/)
[19](https://onlinelibrary.wiley.com/doi/10.1002/smsc.202400133)
[20](https://ieeexplore.ieee.org/document/11233776/)
[21](https://www.semanticscholar.org/paper/c4436ff18ea16fd8ab4c481da768252e9960c649)
[22](https://arxiv.org/pdf/2204.07050.pdf)
[23](https://arxiv.org/abs/2312.01213)
[24](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1406502/pdf)
[25](http://arxiv.org/pdf/2406.03287.pdf)
[26](https://pmc.ncbi.nlm.nih.gov/articles/PMC11294191/)
[27](https://arxiv.org/pdf/2309.04426.pdf)
[28](https://www.nature.com/articles/s41467-025-65394-8)
[29](https://docs.lib.purdue.edu/dissertations/AAI30505320/)
[30](https://pmc.ncbi.nlm.nih.gov/articles/PMC8695433/)
[31](https://arxiv.org/html/2510.14235v1)
[32](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136710709.pdf)
[33](https://www.sciencedirect.com/science/article/abs/pii/S0893608025001352)
[34](https://arxiv.org/abs/2402.12808)
[35](https://ieeexplore.ieee.org/document/10744118/)
[36](https://ieeexplore.ieee.org/document/10375520/)
[37](https://www.semanticscholar.org/paper/2254f3888cc6614c167af917bf73270cc6b8aca2)
[38](https://arxiv.org/abs/2410.05101)
[39](https://ieeexplore.ieee.org/document/10903229/)
[40](https://arxiv.org/abs/2409.14737)
[41](https://link.springer.com/10.1007/s10845-023-02318-7)
[42](https://ieeexplore.ieee.org/document/10862588/)
[43](https://arxiv.org/pdf/1611.03530.pdf)
[44](http://arxiv.org/pdf/2205.08836.pdf)
[45](http://arxiv.org/pdf/2209.09298.pdf)
[46](https://arxiv.org/pdf/2303.05506.pdf)
[47](https://arxiv.org/pdf/2502.09193.pdf)
[48](https://arxiv.org/html/2405.00699)
[49](https://arxiv.org/pdf/2412.13610.pdf)
[50](https://pmc.ncbi.nlm.nih.gov/articles/PMC7059737/)
[51](https://pmc.ncbi.nlm.nih.gov/articles/PMC9854259/)
[52](https://arxiv.org/html/2509.21345v2)
[53](https://pmc.ncbi.nlm.nih.gov/articles/PMC11880274/)
[54](http://papers.neurips.cc/paper/8995-spike-train-level-backpropagation-for-training-deep-recurrent-spiking-neural-networks.pdf)
[55](https://arxiv.org/html/2510.15542v2)
[56](https://proceedings.neurips.cc/paper_files/paper/2024/file/b8bf2c0dd0b48511889b7d3b2c5fc8f5-Paper-Conference.pdf)
