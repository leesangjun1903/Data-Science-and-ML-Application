# Event-based Video Reconstruction via Potential-assisted Spiking Neural Network

### 1. 핵심 주장과 주요 기여

이 논문은 **뉴로모르픽 비전 센서(neuromorphic vision sensor)에서 발생하는 이벤트 데이터로부터 고품질의 비디오를 재구성하기 위해 완전히 뉴로모르픽한 스파이킹 신경망(SNN) 기반의 접근법을 제시**합니다.[1]

논문의 핵심 기여는 다음과 같습니다:[1]

**첫째, 이미지 재구성 작업에 처음으로 깊은 SNN 구조를 적용**했습니다. 기존의 인공 신경망(ANN) 기반 방법들이 높은 계산 복잡도와 전력 소비를 야기하는 반면, SNN은 인공지능 하드웨어에서 더 효율적으로 동작할 수 있다는 장점을 활용합니다.

**둘째, 하이브리드 포텐셜 보조 프레임워크(PA-EVSNN)를 제안**하면서 **적응 막 포텐셜(Adaptive Membrane Potential, AMP) 뉴런을 새롭게 도입**했습니다. AMP 뉴런은 입력 스파이크에 따라 시간 상수를 동적으로 조정하여 시간적 수용장(temporal receptive field)을 향상시킵니다.

**셋째, 에너지 효율성의 획기적인 개선을 달성**했습니다. EVSNN과 PA-EVSNN은 ANN 아키텍처 대비 각각 **19.36배와 7.75배의 에너지 효율 개선**을 이루었으며, E2VID 방식 대비 **24.15배와 8.76배의 계산 효율성 향상**을 보였습니다.[1]

***

### 2. 문제 정의, 제안 방법, 모델 구조 및 성능 분석

#### 2.1 해결하고자 하는 문제

이벤트 카메라는 비동기적으로 픽셀 단위의 밝기 변화를 기록하며, 다음과 같은 장점을 제공합니다:[1]

- **높은 시간 해상도**: 마이크로초 단위의 정확한 타이밍
- **높은 다이나믹 레인지**: 140dB (표준 카메라는 60dB)
- **낮은 전력 소비**: 기존 카메라 대비 훨씬 효율적

그러나 비동기 이벤트의 희소성(sparsity)과 불규칙성으로 인해 전통적인 컴퓨터 비전 알고리즘과의 호환성이 낮아, **이벤트 데이터를 인간이 이해할 수 있는 형태의 영상으로 재구성하는 것이 핵심 과제**입니다.[1]

#### 2.2 입력 표현(Input Representation)

논문에서는 연속 복셀 그리드(continuous voxel grid)를 입력 표현으로 사용합니다:[1]

$$E(x, y, t_n) = \sum_{i} p_i \max(0, 1 - |t_n - t_i^*|)$$

여기서:
- $t_i^*$: 정규화된 이벤트 타임스탬프
- $p_i$: 이벤트의 극성(polarity, on/off)
- 각 시간 단계에서 이벤트는 시간적 정보를 보존하는 N개의 시간 빈으로 변환됩니다.

#### 2.3 스파이킹 뉴런 모델

**Leaky Integrate-and-Fire (LIF) 뉴런:**

LIF 뉴런의 정수 형태는 다음과 같이 표현됩니다:[1]

$$V_t = V_{t-1} + \frac{1}{\tau}(-(V_{t-1} - V_{rest}) + X_t)$$
$$S_t = H(V_t - V_{th})$$

여기서:
- $V_t$: 시간 t에서의 막 전위(membrane potential)
- $X_t$: 입력 신호
- $\tau$: 막 시간 상수(membrane time constant)
- $H(\cdot)$: 헤비사이드 계단 함수
- $S_t$: 스파이크 출력 (0 또는 1)

LIF 뉴런은 통합 과정에서 시간 정보를 추출하지만, 이진 스파이크로 인해 표현 능력이 제한됩니다.[1]

**막 포텐셜(Membrane Potential, MP) 뉴런:**

MP 뉴런은 스파이크를 발생시키지 않고 막 전위 자체를 출력하는 비스파이킹 뉴런으로, LSTM의 간단한 버전으로 볼 수 있습니다:[1]

$$V_t = (1 - \frac{1}{\tau})V_{t-1} + \frac{1}{\tau}X_t$$
$$O_t = V_t$$

이 구조는 더 풍부한 시간 정보를 추출할 수 있습니다.

#### 2.4 적응 막 포텐셜(AMP) 뉴런

논문의 핵심 혁신 중 하나는 **시간 상수를 입력에 따라 동적으로 조정하는 AMP 뉴런**입니다.[1]

**스파이크 발화율 계산:**
$$F = \text{AvgPool}(S_l)$$

**국소 움직임 강도 추정:**
$$I = \text{MaxPool}(\text{Conv}(S_l))$$

**적응 시간 상수 업데이트:**
$$\tau = \frac{1}{\sigma(\text{Linear}([F, I]))}$$

여기서:
- $\sigma(\cdot)$: 시그모이드 활성화 함수
- $F$: 채널별 스파이크 발화율
- $I$: 입력 스파이크의 국소 움직임 강도

이를 통해 **빠른 광 변화는 높은 $\tau$ 값으로 새로운 정보에 더 집중**하고, **느린 변화는 낮은 $\tau$ 값으로 이전 정보를 더 유지**합니다.[1]

#### 2.5 제안 모델 구조

**EVSNN (완전 스파이킹 신경망):**

EVSNN은 U-Net 기반 구조로 구성됩니다:[1]

- **입력**: 1×W×H 이벤트 복셀
- **인코더**: $N_e = 3$개의 인코더 레이어 (각 레이어마다 채널 수 2배)
- **병목**: $N_r = 1$개의 잔차 블록
- **디코더**: $N_d = 3$개의 디코더 레이어
- **모든 뉴런**: LIF 뉴런 (계산 효율성)
- **최종 레이어**: MP LIF 뉴런 (회색조 이미지 예측)

스파이크 스킵 연결(spike skip connection)로 CONCAT 연산을 사용하여 인코더와 디코더 정보를 결합합니다.

**PA-EVSNN (포텐셜 보조 EVSNN):**

PA-EVSNN은 EVSNN을 기반으로:[1]

- 각 인코더 및 디코더 레이어에 **MP 뉴런 추가**
- **AMP 뉴런 도입**으로 시간적 수용장 증대
- 약 8.4%의 ANN 부동소수점 연산 포함 (하이브리드 네트워크)

#### 2.6 손실 함수 및 학습

전체 손실 함수:[1]

$$L_{total} = \sum_{k=0}^{L} L_k^R + \lambda \sum_{k=L_0}^{L} L_k^{TC}$$

여기서:
- $L_k^R$: LPIPS 손실 (지각적 유사성)
- $L_k^{TC}$: 시간 일관성 손실
- $\lambda = 1$: 가중치
- $L = 40-60$: 훈련 시퀀스 길이
- $L_0 = 2$: 손실 계산 시작점

**역전파 및 대리 그래디언트(Surrogate Gradient):**

스파이킹 뉴런의 비미분 특성 때문에 역전파 시간 통합(BPTT)을 통해:[1]

$$\Delta w_l = \sum_n \frac{\partial L_{total}}{\partial o_t^l} \frac{\partial o_t^l}{\partial V_t^l} \frac{\partial V_t^l}{\partial w_l}$$

$\frac{\partial o_t^l}{\partial V_t^l}$는 비미분이므로 **ArcTan 대리 함수를 사용**합니다:[1]

$$H_1(x) = \frac{1}{\pi}\arctan(\pi x) + \frac{1}{2}$$

MP 뉴런의 경우 직접 미분 가능하므로 $\frac{\partial o_t^l}{\partial V_t^l} = 1$입니다.

#### 2.7 성능 비교

**정량적 성능 (표 1):**

| 데이터셋 | 방법 | MSE ↓ | SSIM ↑ | LPIPS ↓ |
|---------|------|-------|--------|---------|
| IJRR | E2VID | 0.059 | 0.643 | 0.338 |
| IJRR | PA-EVSNN | 0.046 | 0.626 | 0.367 |
| MVSEC | E2VID | 0.138 | 0.377 | 0.651 |
| MVSEC | PA-EVSNN | 0.107 | 0.403 | 0.566 |
| HQF | E2VID | 0.081 | 0.545 | 0.406 |
| HQF | PA-EVSNN | 0.061 | 0.532 | 0.416 |

PA-EVSNN은 **최첨단 ANN 기반 방법들(E2VID, FireNet)과 비슷한 성능을 달성**하면서도 **훨씬 낮은 에너지 소비**를 보입니다.[1]

#### 2.8 시간적 수용장 분석

논문의 절제 연구(ablation study)에서 **시간 입력이 없을 때(T > 50) 성능 저하를 측정**하여 각 모델의 시간적 기억 능력을 평가합니다:[1]

- **ANN w/o recurrent**: 시간 성분이 없어 변화 없음
- **ANN + LSTM (E2VID)**: 빠른 성능 향상 후 지속적 개선
- **SNN + LIF (EVSNN)**: LSTM 수준의 시간적 수용장 달성
- **SNN + LIF + AMP LIF**: 최고의 시간적 정보 활용

**스파이킹 뉴런이 시간 정보를 효과적으로 저장하고 활용할 수 있음을 입증**합니다.

***

### 3. 에너지 효율성 분석 및 한계

#### 3.1 에너지 소비 계산

45nm CMOS 기술 기준으로:[1]

- **ANN 연산**: MAC (multiply-accumulate) = 5.1배 더 비쌈
- **SNN 연산**: 이진 스파이크로 인해 덧셈만 수행 (0.9pJ vs ANN 4.6pJ)

**에너지 효율성:**

| 모델 | #OPANN | #OPSNN | 스파이크 발화율 | 에너지 (10⁻³J) | ANN 대비 효율 |
|------|--------|--------|----------------|-----------------|--------------|
| E2VID-LSTM | 20.07G | 0 | - | 92.32 | 1× |
| EVSNN | 0 | 16.12G | 26.4% | 3.83 | **24.15×** |
| PA-EVSNN | 1.49G | 16.35G | 25.1% | 10.55 | **8.76×** |

**핵심 통찰:**
- EVSNN의 **26.4% 스파이크 발화율**이 높은 에너지 효율을 실현
- PA-EVSNN은 성능 향상을 위해 약간의 ANN 연산 추가로 에너지 효율성 감소 (하이브리드)[1]

#### 3.2 한계 및 제한사항

논문은 다음과 같은 한계를 명시합니다:[1]

**배치 정규화(BN) 문제:**
- BN이 활성화 함수 역할을 하여 입력이 없어도 비영(non-zero) 값 생성
- 불필요한 스파이크 발생으로 에너지 소비 증가
- 스파이크율 감소가 **향후 연구 방향**

**성능 트레이드오프:**
- EVSNN은 완전 스파이킹으로 에너지 효율적이지만 성능 제한
- PA-EVSNN은 성능 개선으로 에너지 효율성 감소

***

### 4. 모델 일반화 성능 및 확장 가능성

#### 4.1 일반화 성능 향상 가능성

**현재 논문의 일반화 관련 결과:**

1. **데이터셋 간 성능 유지**: IJRR, MVSEC, HQF 데이터셋 간 일관된 성능 입증[1]
2. **시간적 수용장의 안정성**: 시간 입력이 없을 때도 점진적 성능 저하로 robustness 보임[1]
3. **아키텍처 설계**: 수많은 절제 연구를 통해 최적화된 모델 구조 도출[1]

**향후 일반화 개선 가능성:**[2][3][4]

최신 연구 동향에 따르면 SNN의 일반화 성능을 향상시킬 수 있는 방법들이 제시되고 있습니다:

1. **시간적 반전 정규화(Temporal Reversal Regularization, TRR)**[4]
   - SNN의 시간적 특성을 활용한 입력/특성 시간 반전 섭동
   - 원본-반전 일관된 출력으로 섭동 불변 표현 학습
   - 일반화 오류의 상한을 이론적으로 축소

2. **적응 초매개변수 최적화**[5]
   - SNN의 추가 초매개변수(시간 상수, 임계값 등) 자동 최적화
   - 애플리케이션별 맞춤형 모델 개발

3. **도메인 적응 및 전이 학습**[6]
   - 프레임 기반 카메라 데이터에서 이벤트 기반 데이터로의 지식 전이
   - 어노테이션 데이터 부족 문제 해결

#### 4.2 이벤트 기반 비전의 교차 도메인 일반화[7]

**상태 공간 모델(State Space Models, SSM) 기반 접근:**[7]
- 서로 다른 추론 빈도에서 최소 성능 저하 (3.76 mAP vs RNN/Transformer 20+ mAP)
- 33% 빠른 학습 속도 달성
- **가변 시간 해상도에 대한 자동 적응**

이러한 방법을 PA-EVSNN과 결합하면 **시간 윈도우 크기 변화에 더 강건한 모델** 구현 가능합니다.

#### 4.3 배치 정규화 최적화

최신 연구에서 제시된 개선 방향:[8]

- **스파이크 정규화(Spike Normalization)**: 활성 함수 역할 최소화
- **비스테이블 뉴런(Bistable Neurons)**: 위상 동기화 향상으로 더 안정적인 스파이크 전파
- 결과: **낮은 발화율(21%)에서도 높은 정확도(59.1% mAP) 달성**

***

### 5. 연구 영향 및 향후 고려사항

#### 5.1 학술적 영향

**기여:**

1. **첫 번째 이미지 재구성 SNN**: 대규모 회귀 작업에 SNN 적용 가능성 입증[1]
2. **AMP 뉴런의 혁신**: 적응형 시간 상수로 다양한 장면 대응
3. **에너지-성능 트레이드오프 정량화**: 정확한 에너지 계산으로 신뢰성 있는 효율성 분석

**학문적 범위 확장:**
- 과거 SNN 연구는 분류(classification), 광학 추정(optical flow) 등에 집중
- 본 논문은 **연속 값 출력이 필요한 회귀 작업**으로 적용 범위 확대[1]

#### 5.2 업계 적용 시나리오[9]

**높은 활용 잠재력:**

1. **자율주행 자동차**: 고속 객체 감지 + 저전력 소비[9]
2. **로봇 공학**: 동적 환경에서의 실시간 시각 인식[9]
3. **의료 영상**: 시각 보철 및 고속 진단[9]
4. **산업 검사**: 저조도 고속 촬영 필요 시나리오[9]

#### 5.3 향후 연구 고려사항

**기술적 도전과제:**[10][11][4][1]

1. **배치 정규화 문제 해결**
   - 스파이크 정규화 기법 개발
   - 불필요한 스파이크 발생 최소화

2. **깊은 SNN 구축**
   - 그래디언트 오류 누적 감소[12]
   - 대리 그래디언트 함수 최적화
   - 잔차 구조 및 비스테이블 뉴런 활용[10]

3. **도메인 일반화 강화**
   - 시간적 반전 정규화 적용[4]
   - 미니배치 감독 일반화 학습(mini-batch supervised generalization learning)[3]
   - 실제 이벤트 카메라 데이터와 합성 데이터 간 sim-to-real 갭 해소

4. **신경형 하드웨어 구현**
   - Loihi, TrueNorth 등 신경형 칩에 최적화된 모델 개발
   - 온칩 학습(on-chip learning) 구현

5. **대규모 SNN 개발**
   - Spiking Transformer 아키텍처 탐색[10]
   - 시각 트랜스포머 기반 SNN 설계

#### 5.4 최신 연구 동향 (2024-2025)[13][2][3][5][4][10]

**주요 개발:**

1. **멀티스케일 특성 융합 SNN**[3]
   - BCI 신경 신호 디코딩에서 크로스-데이 일반화 개선
   - 미니배치 감독 일반화 학습으로 robustness 강화

2. **이벤트 카메라 응용 확대**[14][7]
   - 단일 뷰에서의 배경 재구성 (동적 폐색 처리)
   - 저조도 고속 이미지 재구성 (SPAD 센서 융합)
   - NeRF 기반 3D 재구성

3. **효율성 개선**[15]
   - 뉴런당 0.3 스파이크 이하로 고성능 달성
   - 메모리 벽 문제 해결을 위한 비폰 노이만 아키텍처

4. **비동기 처리**[16]
   - 레이어 동기화 제거로 **50% 적은 스파이크, 2배 빠른 추론**
   - 에너지-지연 트레이드오프 최적화

#### 5.5 실무적 고려사항

**성공적인 적용을 위한 요소:**

1. **데이터 문제**: 이벤트 카메라 어노테이션 데이터 부족
   - 자기 감독 학습(self-supervised learning) 접근[17]
   - 합성 데이터와 실제 데이터 간 도메인 불일치 해결

2. **하드웨어 가용성**: 신경형 칩 제한적 사용
   - 기존 GPU/TPU 호환성 개선
   - SpikingJelly 등 소프트웨어 프레임워크 발전[1]

3. **성능-효율 트레이드오프**: 완전 스파이킹 vs 하이브리드 선택
   - 애플리케이션 요구사항에 따른 최적 모델 선택

***

### 결론

본 논문은 **이벤트 기반 비디오 재구성에 처음으로 깊은 SNN을 적용**하여 신경형 컴퓨팅의 실용성을 입증합니다. AMP 뉴런의 도입으로 **적응형 시간 처리**를 실현하고, **24배 이상의 에너지 효율성 개선**을 달성했습니다. 

다만 현재 모델의 일반화 성능은 최첨단 ANN 방법과 비교해 약간 미흡하며, 배치 정규화 문제와 도메인 적응 측면에서 개선의 여지가 있습니다. 향후 연구는 **시간적 반전 정규화**, **상태 공간 모델**, **비동기 처리** 등의 최신 기법을 통합하여 일반화 성능을 강화하고, 신경형 하드웨어의 광범위한 활용을 촉진해야 할 것입니다.

***

### 참고 자료

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/205b18fd-0e3e-43e4-910a-513a2f57b618/2201.10943v3.pdf)
[2](https://www.mdpi.com/2079-9292/13/15/2948)
[3](https://www.frontiersin.org/articles/10.3389/fnins.2025.1551656/full)
[4](https://www.semanticscholar.org/paper/1c99561dd2a11d9bb93aa7dc75d52274a0efe018)
[5](https://arxiv.org/abs/2502.12172)
[6](https://openaccess.thecvf.com/content/ICCV2023/papers/Jian_Unsupervised_Domain_Adaptation_for_Training_Event-Based_Networks_Using_Contrastive_Learning_ICCV_2023_paper.pdf)
[7](https://rpg.ifi.uzh.ch/research_dvs.html)
[8](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0327513)
[9](https://www.arquimea.com/blog/implementing-neuromorphic-vision-in-ai/)
[10](https://arxiv.org/pdf/2409.02111.pdf)
[11](http://arxiv.org/pdf/2405.04289.pdf)
[12](https://proceedings.mlr.press/v202/deng23d/deng23d.pdf)
[13](https://www.worldscientific.com/doi/10.1142/S0129065725500455)
[14](https://www.sciencedirect.com/science/article/abs/pii/S0925231225004485)
[15](https://www.nature.com/articles/s41467-024-51110-5)
[16](https://www.semanticscholar.org/paper/568db888ed30866d1514bfeacc79036bfa024f62)
[17](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06586.pdf)
[18](https://ieeexplore.ieee.org/document/11134387/)
[19](https://arxiv.org/abs/2509.23762)
[20](https://www.nature.com/articles/s41598-025-90113-0)
[21](https://onlinelibrary.wiley.com/doi/10.1002/oca.70023)
[22](http://arxiv.org/pdf/2406.03287.pdf)
[23](https://arxiv.org/pdf/2401.10843.pdf)
[24](https://arxiv.org/pdf/2303.10780.pdf)
[25](https://arxiv.org/pdf/2208.01204.pdf)
[26](https://arxiv.org/pdf/2309.04426.pdf)
[27](https://arxiv.org/pdf/2302.13939.pdf)
[28](https://www.nature.com/articles/s41528-024-00313-3)
[29](https://arxiv.org/html/2506.19256v3)
[30](https://ieeexplore.ieee.org/document/10795229/)
[31](https://www.sciencedirect.com/science/article/abs/pii/S0031320324008458)
[32](https://www.semanticscholar.org/paper/333955fee9bfea858491c2410e39124de4a1e35a)
[33](https://arxiv.org/abs/2509.02585)
[34](https://healthinnovationpress.com/index.php/hir/article/view/v1n2-004)
[35](https://arxiv.org/abs/2510.04243)
[36](https://arxiv.org/abs/2503.00025)
[37](https://arxiv.org/abs/2402.02694)
[38](https://arxiv.org/abs/2407.00291)
[39](https://ieeexplore.ieee.org/document/10635202/)
[40](https://www.semanticscholar.org/paper/6f4778dff537681994481475ea0120e91b5fe895)
[41](https://arxiv.org/abs/2509.02601)
[42](https://arxiv.org/pdf/2411.12913.pdf)
[43](https://www.mdpi.com/2313-7673/8/4/375/pdf?version=1692326752)
[44](https://arxiv.org/pdf/2303.11674.pdf)
[45](https://arxiv.org/pdf/2107.02053.pdf)
[46](http://arxiv.org/pdf/2208.05853.pdf)
[47](https://arxiv.org/html/2503.06288v1)
[48](https://arxiv.org/pdf/2407.15085.pdf)
[49](http://arxiv.org/pdf/2111.10221v3.pdf)
[50](https://www.nature.com/articles/s41598-023-49956-8)
[51](https://arxiv.org/html/2511.16979v1)
[52](https://openaccess.thecvf.com/content/WACV2024/papers/Fox_Unsupervised_Event-Based_Video_Reconstruction_WACV_2024_paper.pdf)
[53](https://proceedings.neurips.cc/paper/2021/file/c4ca4238a0b923820dcc509a6f75849b-Paper.pdf)
[54](https://openreview.net/forum?id=XvHQnCywxN)
[55](https://arxiv.org/html/2505.08438v2)
[56](https://menttor.live/library/introduction-to-surrogate-gradients-for-snn-training)
[57](https://www.sciencedirect.com/science/article/abs/pii/S0031320325008325)
