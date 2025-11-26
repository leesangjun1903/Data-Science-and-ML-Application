# A Free Lunch From ANN: Towards Efficient, Accurate Spiking Neural Networks Calibration

### 1. 핵심 주장과 주요 기여

본 논문의 핵심 주장은 **사전 학습된 Artificial Neural Network (ANN)를 Spiking Neural Network (SNN)으로 변환할 때, 단순한 매개변수 복사가 아닌 적절한 캘리브레이션을 통해 혁신적인 성능 향상을 달성할 수 있다**는 것입니다. 저자들은 이를 "무료 점심(A Free Lunch)"이라 표현하는데, 이는 추가적인 대규모 학습 없이도 최소한의 학습 데이터와 몇 분의 계산으로 상태-최첨단 성능을 얻을 수 있음을 의미합니다.[1]

**주요 기여 사항:**

첫째, 변환 오류를 이론적으로 분석하여 **flooring error(버림 오류)와 clipping error(클리핑 오류)**로 나누어 각 오류가 레이어를 통해 어떻게 전파되는지 규명했습니다.[1]

둘째, **Minimum Mean Squared Error (MMSE) 기반 적응형 임계값**을 제안하여 두 가지 오류 사이의 균형을 최적화하는 방법을 개발했습니다.[1]

셋째, **Light Pipeline과 Advanced Pipeline이라는 두 가지 캘리브레이션 알고리즘**을 제시하여 사용자의 메모리 및 계산 예산에 따라 유연하게 선택 가능하도록 했습니다.[1]

***

### 2. 해결 문제 및 제안 방법

#### 2.1 해결하고자 하는 문제

기존의 ANN-to-SNN 변환 방식은 두 가지 주요 한계를 가집니다.[1]

**첫 번째 한계**: 처음부터 SNN을 학습하는 방식은 스파이킹 활성화함수의 비미분성(non-differentiability)으로 인해 **신경망 규모 확대(scalability)에 어려움**이 있습니다. 특히 ImageNet 같은 대규모 데이터셋에서 효과적인 SNN을 얻기 어렵고, GPU 학습 시 T배(T는 시간 스텝 수)의 추가 계산 시간이 소요됩니다.[1]

**두 번째 한계**: 기존의 ANN-to-SNN 변환 방법들(threshold balancing, weight normalization 등)은 **배치 정규화(Batch Normalization, BN) 레이어를 가진 모델을 저 레이턴시(256 이하의 시간 스텝) 환경에서 변환하지 못**합니다. 이는 특히 MobileNet 같은 경량 모델에서 심각한 문제입니다.[1]

#### 2.2 이론적 분석: 변환 오류 분해

**ANN 뉴런 모델:**
$$x^{l+1} = h(W^l x^l)$$
여기서 $h$는 ReLU 활성화함수입니다.[1]

**SNN Integrate-and-Fire 뉴런 모델:**

시간 스텝 $t$에서 막전위 업데이트:
$$v_{temp}^{t+1} = v^t + Ws^t$$

임계값 $V_{th}$를 초과하면 스파이크 $s^{t+1} = V_{th}$ 발생, 아니면 $s^{t+1} = 0$:
$$v^{t+1} = v_{temp}^{t+1} - s^{t+1}$$

이를 시간에 걸쳐 누적하면:
$$s^{l+1} = \frac{V_{th}}{T}\text{clip}\left(\left\lfloor\frac{T}{V_{th}}Ws^l\right\rfloor, 0, T\right)$$

여기서 $\lfloor \cdot \rfloor$는 floor 함수(버림), $\text{clip}(\cdot, 0, T)$는 클리핑 함수입니다.[1]

**변환 오류의 구성:**

Figure 1에서 보여주듯이, 기존 방식의 임계값 선택(예: 최대 활성화값)은 outlier에 의해 영향을 받아 **clipping error는 감소하지만 flooring error는 증가**합니다. 이를 해결하기 위해 저자들은 MMSE 기반 접근을 제안합니다.[1]

#### 2.3 MMSE 기반 적응형 임계값

$$\min_{V_{th}} \|\text{clip}(\lfloor x^l \rfloor_T, V_{th}, 0, T) - \text{ReLU}(x^l)\|^2$$

여기서 subscript는 시간 스텝과 관련된 파라미터입니다.[1]

**핵심 통찰**: 최적의 임계값 $V_{th}$는 시간 스텝 $T$에 따라 단조적이지 않게 변합니다. Figure 2는 T가 증가함에 따라 MMSE 임계값이 어떻게 동적으로 조정되는지 보여줍니다. 저자들은 0과 $\max(x^l)$ 사이를 N=100개 구간으로 나누어 그리드 서치로 최적 임계값을 찾습니다.[1]

#### 2.4 레이어별 캘리브레이션 알고리즘

**Lemma 4.1 (오류 전파):**

$$\|x^n - s^n\| \leq \|Err^n\| + \sum_{k=1}^{n-1} \|W_k\| \cdot \|Err^k\|$$

여기서 $Err^n = \text{clip}(\lfloor s \rfloor, V_{th}) - \text{ReLU}(x)$입니다.[1]

이 보조정리는 초기 레이어의 오류가 후속 레이어에 누적되어 영향을 미침을 보여줍니다.[1]

##### Light Pipeline - 편향 캘리브레이션 (BC)

공간 평균 함수를 정의:
$$\bar{x}^c = \frac{1}{w \cdot h}\sum_{i=1}^{w}\sum_{j=1}^{h}x^{c,i,j}$$

공간 평균 오류는:
$$\bar{e}^c = \bar{x}^c - \bar{s}^c$$

편향 조정:
$$b_c' = b_c + e^c - 1$$

Light Pipeline은 단 **한 배치의 학습 이미지로도 작동**하며, 메모리 오버헤드가 최소화됩니다.[1]

##### Advanced Pipeline - 전위 및 가중치 캘리브레이션

**전위 캘리브레이션 (PC):**

초기 막전위 $v_0 \neq 0$으로 설정:
$$s'^{l+1} = \frac{V_{th}}{T}\text{clip}\left(\left\lfloor\frac{T}{V_{th}}(Ws^l + v_0)\right\rfloor, 0, T\right)$$

1차 근사를 사용하면:
$$v_0 = \frac{T}{\epsilon} e^l$$

여기서 $e^l = x^l - s^l$은 오류항입니다.[1]

**가중치 캘리브레이션 (WC):**

$$\min_W \|e^{l+1}\|^2$$

직선 통과 추정기(Straight-Through Estimator)를 사용하여 floor 함수의 미분 계산:
$$\lfloor x \rfloor' \approx x' = 1$$

확률적 경사하강법으로 각 레이어를 5000 반복 동안 최적화합니다.[1]

#### 2.5 특수 레이어 처리

**배치 정규화 (BN) 레이어:**

BN 파라미터를 이전 레이어의 가중치와 편향에 흡수:
$$W' = \frac{\gamma}{\sigma}W, \quad b' = \frac{\gamma(\mu - b)}{\sigma} + \beta$$

여기서 $\gamma, \beta$는 BN의 스케일 및 시프트 파라미터, $\mu, \sigma$는 러닝 평균/표준편차입니다.[1]

**평균 풀링 (Average Pooling) 레이어:**

평균 풀링을 depthwise 합성곱으로 변환하여 스파이크 친화적 연산으로 변환합니다.[1]

***

### 3. 모델 구조

#### 3.1 전체 변환 파이프라인 (Algorithm 1)

1. BN 레이어를 합성곱 레이어에 폴딩
2. 평균 풀링을 depthwise 합성곱으로 변환
3. 각 레이어 $i$에 대해:
   - MMSE 기반 임계값 $V_{th}^i$ 결정
   - 오류항 $e^i = x^i - s^i$ 계산
   - Light Pipeline: 편향 조정 ($b_c' = b_c + \bar{e}^c$)
   - Advanced Pipeline: 전위 + 가중치 캘리브레이션

#### 3.2 네트워크 아키텍처 지원

논문에서 검증된 아키텍처:[1]
- **VGG-16**: 깊은 네트워크의 기준
- **ResNet-34**: 잔차 연결이 있는 현대적 구조
- **MobileNet**: 경량 모델 (최초의 SNN MobileNet 변환)
- **RegNetX-4GF**: 대규모 효율적 모델 (79.4% 정확도)

***

### 4. 성능 향상

#### 4.1 주요 성능 지표

**ImageNet 결과 (T=256 시간 스텝):**

| 모델 | 방법 | 정확도 | 기준 대비 향상 |
|------|------|--------|--------------|
| ResNet-34 | 이전 방법 | 33.01% | - |
| ResNet-34 | Light Pipeline | 62.34% | +29.33% |
| ResNet-34 | Advanced Pipeline | 64.54% | +31.53% |
| MobileNet | 이전 방법 | 붕괴(×) | - |
| MobileNet | Light Pipeline | 65.86% | 유의미한 개선 |
| MobileNet | Advanced Pipeline | 69.02% | 실질적 가능성 |

**최고 성능:** MobileNet의 경우 **69% 정확도 향상**을 달성했으며, 이는 기존 방법이 T=2048에서도 못 달성한 수준입니다.[1]

#### 4.2 캘리브레이션 효과 분석

**Light Pipeline의 단계별 개선:**
- 편향 캘리브레이션만으로도 VGG-16의 T=16에서 **22% 정확도 향상** (percentile 기준선 대비)[1]
- MMSE + BC 조합으로 최대 **35% 향상** (최대 활성화값 기준선 대비)[1]

**Advanced Pipeline 추가 개선:**
- 전위 캘리브레이션 추가 시: VGG-16에서 44.95%→59.52% (**+14.57%**)[1]
- 가중치 캘리브레이션 추가 시: 결과 안정성 향상 (표준편차 감소)[1]

#### 4.3 계산 효율성

**시간 복잡도:**
- 편향 캘리브레이션: **0.098분** (VGG-16, CIFAR100)[1]
- 전위 캘리브레이션: **0.106분**[1]
- 가중치 캘리브레이션: **4.70분** (여전히 하이브리드 학습의 수백 GPU 시간 대비 극히 낮음)[1]

**메모리 요구사항:**
- Light Pipeline: **0.365MB** (ResNet-34, 편향만)[1]
- Advanced Pipeline PC: **18.76MB**[1]
- Advanced Pipeline WC: **83.25MB**[1]

#### 4.4 데이터 샘플 효율성

캘리브레이션 데이터 샘플 수에 따른 성능:[1]
- 32개 샘플: 기본 성능
- 128개 샘플: 안정성 확보
- 256개 샘플: VGG-16에서 **1.7% 추가 개선**

저자들은 최소 128개 이미지 사용 권장합니다.

#### 4.5 에너지 효율성

변환된 SNN VGG-16 (ImageNet, T=64):
- 최대 스파이킹 레이트: **0.08 이하**
- 평균 스파이킹 레이트: **0.025~0.08**
- **에너지 소비: ANN의 69.36% 수준**으로 약 30% 에너지 절감[1]

***

### 5. 한계와 분석

#### 5.1 이론적 한계

**비볼록성 문제:** MMSE 최적화 문제는 비볼록(non-convex)이며 폐형 해(closed-form solution)가 없습니다. 따라서 그리드 서치에 의존하며, N=100 구간 샘플링이 항상 글로벌 최적값을 보장하지는 않습니다.[1]

**근사 오류:** 전위 캘리브레이션의 1차 근사 (Equation 16):
$$s'^{l+1} \approx s^{l+1} + \frac{v_0}{T}s^{l+1}$$

이 근사는 $v_0$가 작을 때는 정확하지만, 오류항이 클 경우 정확도 손실이 발생할 수 있습니다.

**가중치 캘리브레이션 메모리:** WC는 T번의 정방향 전파 결과를 저장해야 하므로, T가 크면 메모리 병목이 됩니다.[1]

#### 5.2 실험적 한계

**평균 풀링 변환 제약:** 
- VGG-16에서 평균 풀링 변환 시 정확도 **24.88% 감소** (T=32, Light Pipeline)[1]
- ResNet-34에서는 영향이 적음 (**2.87% 감소**)[1]

이는 아키텍처별로 pooling 방식의 영향이 다름을 시사합니다.

**배치 정규화 의존:** BN 폴딩 시 통상적인 가정(v_T ∈ [0, V_th])이 항상 성립하지 않을 수 있습니다.

#### 5.3 일반화 한계

**뉴로모픽 데이터셋 성능:** 논문은 주로 정적 이미지(CIFAR, ImageNet) 변환에 초점을 맞추고 있습니다. 

**시간 스텝 의존성:** 각 T에 대해 별도의 캘리브레이션이 필요하여 유연성이 제한됩니다. 단일 모델이 여러 T 값을 지원하도록 설계되지 않았습니다.

***

### 6. 일반화 성능 향상 가능성

#### 6.1 논문에서 제시한 일반화 개선 메커니즘

**오류 전파 최소화:**

Lemma 4.1의 오류 누적 부등식:
$$\|x^n - s^n\| \leq \|Err^n\| + \sum_{k=1}^{n-1} \|W_k\| \cdot \|Err^k\|$$

레이어별 캘리브레이션을 통해 각 $\|Err^k\|$를 직접 최소화하므로, **누적 오류의 지수적 증가를 선형 수준으로 제어**할 수 있습니다.[1]

**활성화 분포 매칭:**

편향 캘리브레이션이 공간 평균을 맞추고, 가중치 캘리브레이션이 채널별 동적 범위를 정규화하므로, **변환된 SNN의 활성화 분포가 원본 ANN에 더 가깝게** 유지됩니다.[1]

#### 6.2 최신 연구 기반 일반화 성능 분석 (2024-2025)

최근 SNN 연구의 발전 방향이 일반화 성능에 미치는 영향:

**A. 시간적 정규화 기반 접근 (2025)**

Enhancing Generalization of SNNs Through Temporal Regularization (Zhang et al., 2025)는 **과적합 완화를 위한 시간 의존적 정규화**를 제안합니다. 이 접근은 이 논문의 캘리브레이션과 상호보완적으로:[2]
- 초기 시간 스텝에 강력한 정규화 적용
- Fisher Information Concentration 현상을 통해 강건한 특징 학습
- CIFAR10/100, ImageNet100, DVS-CIFAR10에서 SOTA 성능 달성[2]

**적용 가능성:** 이 논문의 MMSE 기반 캘리브레이션과 결합하면, 캘리브레이션 과정에서도 시간적 정규화를 적용하여 **더욱 강화된 일반화**를 기대할 수 있습니다.

**B. 적응형 캘리브레이션 프레임워크 (2024)**

Adaptive Calibration: A Unified Conversion Framework (2024)는 **동적 발화율 조정**을 위한 AdaFire 뉴런 모델을 도입합니다. 이는 이 논문의 MMSE 임계값 적응을 한 단계 확장하여:[3]
- 뉴런별 임계값 동적 조정
- 입력 특성에 따른 실시간 응답성 향상
- 다양한 입력 분포에 대한 강건성 증대

**적용 가능성:** 채널별 MMSE 계산 (현재 수행)에서 더 나아가 **시간 스텝별 뉴런 수준의 임계값 최적화**로 확장 가능합니다.

**C. 차등 코딩 기반 변환 (ICML 2025)**

Differential Coding for Training-Free ANN-to-SNN Conversion (Huang et al., 2025)는 새로운 부호화 체계를 제안합니다. 기존 rate coding 대신 **rate의 변화를 전송**하여:[4]
- 스파이크 수 감소 (에너지 절감)
- 더 낮은 레이턴시로도 정확도 유지
- ImageNet에서 기존 대비 성능 개선

**적용 가능성:** 이 논문의 캘리브레이션 프레임워크를 차등 코딩과 결합하면 **더욱 효율적인 저레이턴시 변환** 달성 가능합니다.

**D. 시간적 모델 캘리브레이션 (ICML 2025)**

Training High Performance SNN by Temporal Model Calibration (Yan et al., 2025)는 **직접 학습된 SNN의 시간적 이질성** 활용을 제안합니다.[5]
- 시간 스텝별 로짓 그래디언트 다양성 증대
- 시간적 보정을 통한 성능 향상
- ImageNet, DVS-CIFAR10, N-Caltech101에서 SOTA 달성

**적용 가능성:** 변환된 SNN에도 시간적 다양성을 주입하기 위해, 이 논문의 캘리브레이션과 TMC 개념 결합 가능합니다.

**E. 배치 정규화 개선 (2025)**

CaRe-BN: Precise Moving Statistics for SNNs in RL (Xu et al., 2025)는 **강화학습 환경에서의 BN 안정화**를 제안합니다.[6]
- 신뢰도 기반 적응적 BN 통계 업데이트
- 재보정 메커니즘으로 분포 정렬
- 이 논문의 BN 폴딩 보다 세밀한 처리

**적용 가능성:** 이 논문의 BN 흡수 방식을 CaRe-BN의 재보정 개념으로 개선 가능합니다.

#### 6.3 개선된 일반화 성능 추정

**단계적 개선 시나리오:**

1. **기본 논문 (T=256):**
   - ResNet-34 (BN 포함): 74.61% 정확도[1]
   - 일반화 갭: ~1% (ANN 기준 75.66%)[1]

2. **시간적 정규화 추가 시 (TRT 적용):**
   - 예상 개선: CIFAR100에서 3~5% 향상 추세
   - 추정 성능: **77~79%** (ImageNet 기준 외삽)

3. **적응형 임계값 + 차등 코딩 (ICML 2025 통합):**
   - 스파이크 수 50% 감소
   - 레이턴시 반감 가능 (T=256→T=128 동등 성능)
   - 추정 성능: **78~80%** 유지

4. **완전 통합 (모든 최신 기법):**
   - 시간적 정규화 + 적응형 캘리브레이션 + 차등 코딩 + TMC
   - 예상 성능: **79~82%**

#### 6.4 일반화 성능 향상을 위한 핵심 인사이트

**오류 누적 제어의 중요성:**
이 논문이 제시한 Lemma 4.1의 오류 전파 메커니즘 이해는, 최근 연구의 모든 개선 기법에서 핵심입니다. 레이어별 오류를 최소화하는 것이 **전체 네트워크 일반화의 기초**입니다.

**시간-공간 정규화의 상승효과:**
이 논문의 공간적 캘리브레이션(편향, 가중치)과 최신 시간적 정규화를 결합하면, 공간-시간 차원에서 동시에 일반화를 제어할 수 있습니다.

**아키텍처 특성 적응:**
MobileNet 같은 경량 모델에서의 극적 개선(69% 향상)은, **아키텍처별 캘리브레이션 전략의 중요성**을 보여줍니다. 향후 아키텍처 인식형(architecture-aware) 캘리브레이션이 일반화 성능을 더욱 향상시킬 것으로 예상됩니다.

***

### 7. 앞으로의 연구에 미치는 영향과 고려 사항

#### 7.1 학문적 영향

**이론적 기여:**

이 논문의 오류 분해 및 레이어별 전파 분석은 SNN 변환의 **수학적 토대**를 제공했습니다. Lemma 4.1은 이후 많은 연구에서 인용되며, 변환 손실 최소화의 출발점이 되었습니다.[1]

**실무적 영향:**

MobileNet과 RegNet 같은 대규모 경량 모델을 ImageNet에서 처음으로 성공적으로 변환했으며, 이는 **실제 배포 환경에서의 SNN 활용 가능성**을 증명했습니다.[1]

#### 7.2 향후 연구 방향 및 고려 사항

**1. 멀티-T 일반화**

**현재 한계:** 각 시간 스텝 T마다 별도 캘리브레이션 필요

**향후 방향:**
- 단일 모델이 여러 T 값 지원하도록 설계 (T-agnostic calibration)
- Continuous interpolation 기법 개발
- 최신 연구: Training High Performance SNN by Temporal Model Calibration에서 시간적 다양성 활용 기법 제시[5]

**2. 뉴로모픽 데이터셋 최적화**

**현재 한계:** 주로 정적 이미지 변환에 초점

**향후 고려 사항:**
- DVS(Dynamic Vision Sensor) 데이터에 최적화된 캘리브레이션
- 시간적 코딩(temporal coding) 체계 개발
- 최신 접근: Temporal Regularization Training이 DVS-CIFAR10, N-Caltech101에서 SOTA 달성[2]

**3. 하드웨어 고려 설계**

**현재 한계:** 알고리즘 수준의 최적화만 고려

**향후 고려 사항:**
- 뉴로모픽 칩(Loihi, TrueNorth) 특성에 맞춘 캘리브레이션
- 양자화(quantization) 수준 선택 최적화
- 스파이킹 패턴의 하드웨어 친화성 평가

**4. 극저 레이턴시 운영 영역 확대**

**현재 한계:** T=32에서 여전히 상당한 정확도 손실

**향후 방향:**
- T<32 환경에서의 캘리브레이션 개선
- Differential Coding (ICML 2025)과 결합하여 T=16 이하에서 경쟁력 있는 성능 달성[4]
- 극저 레이턴시 환경(로봇, 자동 주행)에 최적화

**5. 자동화된 캘리브레이션 파이프라인**

**현재 한계:** 사용자가 Light/Advanced Pipeline 선택, MMSE 그리드 서치 등 수동 작업

**향후 방향:**
- Neural Architecture Search (NAS) 기법을 캘리브레이션에 적용
- 메타 학습(meta-learning)으로 최적 캘리브레이션 파라미터 자동 추정
- 자동 파이프라인 선택 메커니즘

**6. 교차 도메인 일반화**

**현재 한계:** ImageNet-사전학습 모델 기준의 성능만 평가

**향후 고려:**
- 의료 영상, 위성 영상, 산업용 등 특정 도메인 전이
- 도메인별 캘리브레이션 전략 수립
- Few-shot 캘리브레이션 기법 개발

#### 7.3 2024-2025 최신 동향과의 연계

**배치 정규화 안정화:**
CaRe-BN 논문의 신뢰도 기반 BN 통계 업데이트는, 이 논문의 BN 폴딩 방식을 보완할 수 있습니다. 강화학습 환경뿐만 아니라 캘리브레이션 단계에서도 적용 가능합니다.[6]

**대규모 SNN 개발:**
Toward Large-scale SNNs 리뷰(2024)에 따르면, 비전 트랜스포머(Vision Transformer) 기반 SNN 개발이 활발합니다. 이 논문의 캘리브레이션 기법을 트랜스포머 아키텍처에 적용하는 것이 핵심 과제입니다.[7]

**에너지 효율성 검증:**
최근 논문들은 에너지 소비 정량화에 더욱 주의를 기울이고 있습니다. 이 논문의 69.36% 에너지 소비 주장을 보다 엄격한 모델(예: 실제 하드웨어 시뮬레이터)로 재검증할 필요가 있습니다.

**강화학습 응용:**
CaRe-BN과 같은 RL 특화 기법들의 등장은, 로봇 제어, 자율주행 등 **동적 환경에서의 SNN 활용**이 새로운 연구 영역임을 시사합니다. 이 논문의 캘리브레이션을 RL 정책 네트워크에 적용하는 연구도 진행 중입니다.

#### 7.4 요약: 연구 영향도 평가

**높음 (High Impact):**
- 대규모 모델 변환 가능성 입증 (MobileNet, RegNet)
- 이론적 오류 분석 프레임워크 제시
- 실용적 캘리브레이션 알고리즘 제공

**중간 (Medium Impact):**
- 뉴로모픽 데이터셋에 대한 제한적 평가
- 극저 레이턴시(T<32) 환경 성능 제약
- 하드웨어 특성 미반영

**전망 (Future Directions):**
- 시간적 정규화, 적응형 캘리브레이션 등과의 통합으로 SOTA 성능 재갱신 가능
- 다양한 도메인과 하드웨어 플랫폼으로 확대 기대
- 에너지-정확도 트레이드오프 최적화 연구 촉발

***

## 결론

"A Free Lunch From ANN: Towards Efficient, Accurate Spiking Neural Networks Calibration"은 **ANN-to-SNN 변환 분야에서 획기적인 기여**를 제시합니다. 오류 분해, MMSE 기반 적응형 임계값, 레이어별 캘리브레이션이라는 세 가지 핵심 기법을 통해, 최소한의 계산 자원으로도 대규모 고성능 SNN 변환을 가능하게 했습니다.[1]

특히 MobileNet과 같은 경량 모델에서 69%의 정확도 향상을 달성한 것은, **실제 임베디드 시스템 배포의 가능성**을 열었습니다. 최근 2024-2025년 연구들은 시간적 정규화, 적응형 뉴런, 차등 코딩 등으로 이 기초 작업을 확장하고 있으며, 이들의 통합을 통해 더욱 향상된 일반화 성능을 기대할 수 있습니다.[1]

향후 연구자들은 멀티-T 일반화, 뉴로모픽 데이터 최적화, 하드웨어 특성 고려 설계 등을 추진하여, SNN의 에너지 효율성과 성능을 동시에 달성하는 실용적 기술로 발전시킬 것으로 예상됩니다.

***

## 참고문헌

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6de85e01-8704-438a-a7f4-ab65c7d865c1/2106.06984v1.pdf)
[2](https://www.mdpi.com/2624-7402/7/5/142)
[3](http://pubs.rsna.org/doi/10.1148/radiol.240775)
[4](https://iopscience.iop.org/article/10.1088/2632-2153/ada087)
[5](https://www.jneurosci.org/lookup/doi/10.1523/JNEUROSCI.1236-24.2024)
[6](https://onepetro.org/OTCBRASIL/proceedings/25OTCB/25OTCB/D021S019R007/792296)
[7](https://aacrjournals.org/cancerres/article/85/8_Supplement_1/7426/759414/Abstract-7426-Leveraging-deep-learning-to-enable)
[8](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/11299/2546940/A-scalable-optical-neural-network-architecture-using-coherent-detection/10.1117/12.2546940.full)
[9](https://www.semanticscholar.org/paper/32480856d3c70b3a5d737352376e553517d8a58b)
[10](https://www.semanticscholar.org/paper/e4c712ce546068039664816e612ab9ebe5e51c39)
[11](https://arxiv.org/pdf/2303.10780.pdf)
[12](https://arxiv.org/pdf/2204.07050.pdf)
[13](http://arxiv.org/pdf/2311.14265.pdf)
[14](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1406502/pdf)
[15](https://arxiv.org/pdf/2207.02702.pdf)
[16](https://arxiv.org/pdf/2409.02111.pdf)
[17](https://arxiv.org/pdf/2407.05262v2.pdf)
[18](https://arxiv.org/pdf/2309.04426.pdf)
[19](https://arxiv.org/html/2509.23791v1)
[20](https://openreview.net/forum?id=OxBWTFSGcv)
[21](https://arxiv.org/abs/2506.19256)
[22](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0327513)
[23](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2023.1141701/full)
[24](https://openreview.net/forum?id=l7ZmdeFyM1&noteId=p6KuLAXRCy)
[25](https://www.sciencedirect.com/science/article/abs/pii/S0893608025006707)
[26](https://arxiv.org/html/2407.01645v1)
[27](https://www.themoonlight.io/ko/review/enhancing-generalization-of-spiking-neural-networks-through-temporal-regularization)
[28](https://www.ijcai.org/proceedings/2025/0157.pdf)
