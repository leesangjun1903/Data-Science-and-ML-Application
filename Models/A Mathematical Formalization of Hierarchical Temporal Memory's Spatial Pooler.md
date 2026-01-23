
# A Mathematical Formalization of Hierarchical Temporal Memory's Spatial Pooler

## 개요

Mnatzaganian, Fokoue, Kudithipudi가 2016년 발표한 이 논문은 Hierarchical Temporal Memory (HTM)의 공간 풀러(Spatial Pooler, SP) 컴포넌트에 대한 처음의 포괄적 수학적 형식화를 제시한다. 신경과학에 영감을 받은 알고리즘을 엄밀한 수학 프레임워크로 변환함으로써 머신러닝 커뮤니티에서의 신뢰성을 확보하고, 최적화 및 하드웨어 구현의 길을 열었다.

***

## 1. 핵심 주장 및 주요 기여

### 1.1 핵심 주장

본 논문의 기본 주장은 세 가지로 정리된다:

**첫째**, HTM 알고리즘은 신경과학에 기반하여 설계되었으나 엄밀한 수학적 형식화가 부재하여 알고리즘의 특성 파악, 개선 가능성 탐색, 하드웨어 최적화가 어렵다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e240668b-e68d-49ae-997e-116159b21e58/1601.06116v3.pdf)

**둘째**, SP는 HTM의 핵심 학습 성분으로, 입력 데이터를 희소 분산 표현(Sparse Distributed Representation, SDR)으로 변환하는 벡터 양자화(Vector Quantization) 방식의 비지도 학습 알고리즘이다. 이 컴포넌트의 동작 원리를 완전히 이해하려면 수학적 형식화가 필수적이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e240668b-e68d-49ae-997e-116159b21e58/1601.06116v3.pdf)

**셋째**, permanence 업데이트 양의 결정 메커니즘은 최대 우도 추정(Maximum Likelihood Estimation)으로 설명할 수 있으며, 부스팅 메커니즘은 이차적 학습 메커니즘에 불과하다. 이는 알고리즘의 핵심 학습 프로세스를 단순화할 수 있음을 시사한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e240668b-e68d-49ae-997e-116159b21e58/1601.06116v3.pdf)

### 1.2 주요 기여

논문이 제시하는 다섯 가지 주요 기여는 다음과 같다:

1. **SP의 완전한 수학적 프레임워크 구축**: 부스팅, 국소 억제를 포함한 세 단계(Overlap, Inhibition, Learning) 알고리즘의 벡터화된 수학적 표현 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e240668b-e68d-49ae-997e-116159b21e58/1601.06116v3.pdf)

2. **특성 학습(Feature Learning) 능력 입증**: SP가 임의의 입력 패턴을 효과적인 특성으로 변환하는 능력을 이론적으로 및 실증적으로 검증 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e240668b-e68d-49ae-997e-116159b21e58/1601.06116v3.pdf)

3. **비공간 데이터 전처리기로의 활용**: 공간적 특성이 없는 데이터를 공간적 표현으로 변환하여 다운스트림 알고리즘의 성능 향상 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e240668b-e68d-49ae-997e-116159b21e58/1601.06116v3.pdf)

4. **Permanence 업데이트량의 최대 우도 추정 설명**: 경험적으로만 알려진 업데이트 규칙의 이론적 기초 제시 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e240668b-e68d-49ae-997e-116159b21e58/1601.06116v3.pdf)

5. **실용적 설계 가이드라인 제공**: 파라미터 초기화, 성능 예측, 하드웨어 최적화를 위한 수학적 도구 제공 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e240668b-e68d-49ae-997e-116159b21e58/1601.06116v3.pdf)

***

## 2. 해결하고자 하는 문제 및 제안 방법

### 2.1 주요 문제점

| 문제 영역 | 구체적 내용 | 영향 |
|---------|-----------|------|
| **수학적 형식화 부재** | HTM 알고리즘이 신경생물학적 구조에만 기반, 엄밀한 수식 부재 | 성능 최적화 불가, 이론적 분석 곤란 |
| **알고리즘 이해 부족** | SP의 각 단계별 동작 원리 불명확 | 개선 방향 설정 어려움 |
| **파라미터 최적화 난제** | 10개 이상의 사용자 정의 파라미터 필요 | 응용 확대 제약 |
| **하드웨어 구현 불가능** | 수학적 형식화 부재로 최적화 불가 | 에너지 효율 개선 불가 |
| **일반화 성능 미검증** | SP의 비공간 데이터 처리 능력 미증명 | 응용 범위 제한 |

### 2.2 제안 방법: 수학적 형식화

#### 2.2.1 3단계 공간 풀링 알고리즘

**Phase 1: Overlap 계산**

$$\vec{\omega}_i = b_i \cdot (\vec{\Phi}_i \cdot \vec{x})$$

여기서:
- $\vec{\omega}_i$: 컬럼 $i$의 overlap (활성 연결 시냅스 수)
- $b_i$: 컬럼 $i$의 부스트 값 (초기값 1)
- $\vec{\Phi}_i = I[\vec{\Pi}_i \geq \sigma_s]$: 컬럼 $i$의 연결 시냅스 마스크
- $\vec{\Pi}_i$: permanence 값 벡터
- $\sigma_s$: 연결 임계값
- $\vec{x}$: 입력 패턴 (이진)

최종 overlap:

$$\omega_i = \begin{cases} \omega_i & \text{if } \omega_i \geq d \\ 0 & \text{otherwise} \end{cases}$$

**Phase 2: 억제(Inhibition)**

$$\tau_i = \text{kmax}(H_i \cdot \vec{\omega}, k)$$

여기서 $k$는 원하는 컬럼 활동성 수준이며:

$$c_i = I[\omega_i > 0 \text{ and } \omega_i \geq \max(1, \tau_i)]$$

$c = [c_1, c_2, \ldots, c_m]$은 활성 컬럼의 지표 벡터다.

**Phase 3: 학습**

Permanence 적응:
$$\vec{\Pi} = \text{clip}(\vec{\Pi} + \Delta, 0, 1)$$

여기서 permanence 업데이트는:
$$\Delta = c^T \vec{X} \cdot \sigma^+ - \overline{\vec{X}} \cdot \sigma^-$$

논문의 핵심 기여는 $\sigma^+$와 $\sigma^-$의 관계를 최대 우도 추정으로 설명한 것이다:

**MLE 기반 영구값 업데이트 설명**

입력이 Bernoulli 분포를 따른다고 가정하면:
$$\mathcal{L}(\sigma|X) = \prod_{i,k} \sigma^{X_{i,k}}(1-\sigma)^{1-X_{i,k}}$$

로그 우도:

$$\ell(\sigma|X) = \sum_{i,k}[X_{i,k}\log\sigma + (1-X_{i,k})\log(1-\sigma)]$$

미분:

$$\frac{\partial \ell}{\partial \sigma} = \sum_{i,k}\left[\frac{X_{i,k}}{\sigma} - \frac{1-X_{i,k}}{1-\sigma}\right]$$

각 $X_{i,k}$에 대해:

$$\frac{\partial \ell}{\partial \sigma}\bigg|_{X_{i,k}} = \frac{X_{i,k}}{\sigma} - \frac{1-X_{i,k}}{1-\sigma}$$

경사 상승(Gradient Ascent)에 $\sigma$ 스케일을 적용하면:
$$\Delta\sigma = \sigma^+ X_{i,k} - \sigma^- (1-X_{i,k})$$

이것이 바로 SP의 permanence 업데이트 규칙이며, $\sigma^+$와 $\sigma^-$는 스케일 파라미터로 기능한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e240668b-e68d-49ae-997e-116159b21e58/1601.06116v3.pdf)

#### 2.2.2 부스팅 메커니즘

활동성 의무율 (duty cycle):
$$\text{adc}_i = \frac{\sum_t c_i(t)}{w}, \quad \text{odc}_i = \frac{\sum_t I[\omega_i(t) \geq d]}{w}$$

여기서 $w$는 시간 윈도우 크기다.

최소 활동성 의무율:
$$\text{mdc}_i = 0.01 \times \max_j[\text{adc}_j], \quad j \in N(i)$$

부스트 함수:

$$b_i(\text{adc}_i, \text{mdc}_i) = \begin{cases} b_{max} & \text{if } \text{mdc}_i = 0 \\ 1 & \text{if } \text{adc}_i \geq \text{mdc}_i \\ \frac{\text{adc}_i}{10 \times \text{mdc}_i} \times (b_{max} - 1) & \text{otherwise} \end{cases}$$

#### 2.2.3 억제 반경 업데이트

거리 행렬:
$$D_{i,k} = d(\text{pos}(c_i, 0), \text{pos}(i_k, 1))$$

평균 수용장(Receptive Field):
$$r = \left\lfloor \max\left(1, \frac{\sum_{i,k} D_{i,k} \cdot Y_{i,k}}{\sum_{i,k} Y_{i,k}}\right) \right\rfloor$$

### 2.3 특성 학습 및 차원 감소

#### 확률적 특성 맵핑

Permanence 값을 중요도 확률로 해석:
$$\psi_r = \max_i[\Phi_{i,r} \cdot \Pi_{i,:}]$$

여기서 각 입력 $r$에 대해 모든 컬럼의 permanence 최댓값을 취한다.

#### 차원 감소

중요 특성 마스크:
$$z_r = I[\psi_r \geq \sigma_s]$$

이를 통해 원본 데이터 차원을 35-38% 감소 가능하다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e240668b-e68d-49ae-997e-116159b21e58/1601.06116v3.pdf)

#### 입력 재구성

활성 컬럼과 permanence를 사용한 입력 복원:
$$u_r = I\left[\max_{i:c_i=1}\left(\max_k[\Phi_{i,r} \cdot \Pi_{i,k}]\right) \geq \sigma_s\right]$$

***

## 3. 모델 구조 분석

### 3.1 HTM의 계층적 조직

```
┌─────────────────────────────────────┐
│         Hierarchical Levels         │
│  ┌────────────────────────────────┐ │
│  │        Region (Level L)         │ │
│  │  ┌──────┐  ┌──────┐  ┌──────┐ │ │
│  │  │Col1  │  │Col2  │  │Col3  │ │ │
│  │  │[Cell]│  │[Cell]│  │[Cell]│ │ │
│  │  │[Cell]│  │[Cell]│  │[Cell]│ │ │
│  │  └──────┘  └──────┘  └──────┘ │ │
│  └────────────────────────────────┘ │
│           ↓ (Synapses)              │
│  ┌────────────────────────────────┐ │
│  │  Proximal/Distal Dendrites     │ │
│  │  • Feedforward connections     │ │
│  │  • Lateral connections         │ │
│  └────────────────────────────────┘ │
└─────────────────────────────────────┘
```

### 3.2 SP의 내부 구조

**입력 처리 흐름**:

$$\text{Input } \vec{x} \rightarrow \text{Overlap Calculation} \rightarrow \text{Inhibition} \rightarrow \text{SDR Output } \vec{c}$$

각 컬럼의 구조:
- **Proximal Dendrite Segment**: 입력에서의 시냅스 학습
- **Competitiveness**: k-winners-take-all 억제
- **Boosting Mechanism**: 언더-활용 컬럼의 동적 강화

### 3.3 SDR (Sparse Distributed Representation)의 특성

SDR은 세 가지 핵심 파라미터로 정의된다:

- **n**: 전체 비트 수 (통상 2000 이상)
- **w**: 활성 비트 수 (약 2% 희소도)
- **θ**: 매칭 임계값

이러한 높은 차원과 낮은 희소도로 인해:

$$P(\text{false match}) = \left(1 - \theta^w\right)^n$$

$n=2048, w=40, \theta=0.4$일 때, 오매칭 확률이 10억 분의 1로 감소한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e240668b-e68d-49ae-997e-116159b21e58/1601.06116v3.pdf)

***

## 4. 성능 향상 및 실험 결과

### 4.1 공간 데이터 실험 (MNIST)

| 방법 | Global Inhibition | Local Inhibition | 차원 감소 |
|------|------------------|------------------|---------|
| **SP + Linear SVM** | 7.70% | 7.85% | - |
| **Linear SVM (베이스라인)** | 7.95% | 7.95% | - |
| **Probabilistic 특성** | 8.98% | 9.07% | 차원 38-35% 감소 |
| **Dimensionality Reduction** | 9.03% | 9.07% | - |

**해석**: 공간 데이터(이미지)에서 SP는 약간의 성능 향상을 제공하지만, 더 중요한 것은 원본 특성을 유지하면서 차원을 35% 이상 감소시킬 수 있다는 점이다. 이는 다운스트림 분류기의 계산 효율성 향상을 의미한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e240668b-e68d-49ae-997e-116159b21e58/1601.06116v3.pdf)

### 4.2 비공간 데이터 실험 (자동차 평가)

| 방법 | 에러율 |
|------|--------|
| **SP + Linear SVM** | **1.73%** |
| **Random Forest** | 8.96% |
| **Linear SVM** | 26.01% |
| **Best Known (Boosted MLP)** | 0.37% |

**해석**: 비공간 데이터에서 SP의 성능 향상이 극적이다. 원본 데이터의 비선형 관계를 공간적 표현으로 변환하면서 SVM의 선형 분류 능력을 크게 향상시킨다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e240668b-e68d-49ae-997e-116159b21e58/1601.06116v3.pdf)

### 4.3 부스팅 메커니즘 분석

연구자들이 관찰한 중요한 발견은 다음과 같다:

**Permanence 부스팅은 입력 희소도가 70-76% 범위에서만 발생한다.** 이는 알고리즘 파라미터 선택(q=40, d=15)과 밀접한 관련이 있다.

- 희소도 75% 이상: 활성 입력이 너무 적어 컬럼이 충분한 overlap 달성 불가능
- 희소도 70% 이하: 입력 커버리지 충분하여 부스팅 불필요

이 관찰은 **리소스 제한 시스템에서 부스팅을 제거할 수 있음**을 시사한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e240668b-e68d-49ae-997e-116159b21e58/1601.06116v3.pdf)

### 4.4 일반화 성능 평가

#### 노이즈 강건성
SP는 최대 40% 노이즈를 추가해도 출력에 변화가 없다. 이는 다음 이유에서 비롯된다:
1. SDR의 높은 차원 (n ≫ w)
2. Hebbian 학습의 의미론적 특성
3. Permanence 값의 연속적 업데이트 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e240668b-e68d-49ae-997e-116159b21e58/1601.06116v3.pdf)

#### 입력 통계 변화 적응
새로운 데이터셋으로 전환 직후 엔트로피가 급락하지만, 수십 회 반복 후 안정화된다. 이는 온라인 학습의 강점을 보여준다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e240668b-e68d-49ae-997e-116159b21e58/1601.06116v3.pdf)

***

## 5. 모델의 한계

### 5.1 이론적 한계

| 한계 | 설명 | 영향 |
|------|------|------|
| **MLE 설명의 가설적 특성** | Permanence 업데이트의 MLE 유도는 명시적 검증이 아닌 사후적 설명 | 이 메커니즘이 정말 MLE를 최적화하는지 증명 필요 |
| **부스팅의 이차적 역할 주장** | 특정 파라미터 조건에서만 검증됨 | 다양한 파라미터 범위에서의 일반화 불명확 |
| **초기화의 장기 영향 미분석** | 랜덤 초기화의 성능 편차 크지만 원인 분석 미흡 | 재현 가능성과 파라미터 견고성 미확보 |
| **정보 이론적 하한 부재** | SP의 특성 학습 성능의 이론적 상한 미제시 | 성능 개선의 근본적 한계 불명확 |

### 5.2 실증적 한계

1. **제한된 실험 규모**:
   - MNIST: 축소 버전 (800 학습 샘플)
   - 카테고리 데이터: 단일 데이터셋만 사용
   - 계층적 HTM 네트워크의 다단계 상호작용 미검증

2. **파라미터 최적화의 비효율성**:
   - 1,000개의 무작위 파라미터 조합 필요
   - 파라미터 공간의 명확한 가이드라인 부재
   - 자동 하이퍼파라미터 튜닝 전략 미제시

3. **기준선 부족**:
   - 다른 신경망 기반 특성 학습 방법(Autoencoders, CNNs)과 비교 없음
   - Temporal Memory(TM)와의 통합 성능 평가 미함

### 5.3 일반화 성능 관련 한계

#### 공간 데이터에서의 제한된 개선
MNIST에서 SP + SVM이 기존 SVM(7.95%)보다 겨우 3.2% 향상(7.70%)된 것은 공간 구조를 이미 포함하는 데이터에서 SP의 추가 가치가 제한적임을 시사한다.

#### 고차원 데이터 미검증
논문의 실험은 모두 저차원 또는 중간 차원 데이터(28×28 이미지, 6 속성)에 제한되어 있다. 고차원 데이터(예: 1000개 이상 특성)에서의 성능은 미확인이다.

#### 과적합 방지 메커니즘 부족
- SP는 과적합을 명시적으로 방지하는 메커니즘이 없다
- 정규화(Regularization) 전략 미제시
- 모델 선택(Model Selection) 기준 미정의

#### 전이 학습 가능성 미탐색
한 데이터셋에서 학습한 SP를 다른 데이터셋에 적용할 가능성이 검토되지 않았다.

***

## 6. 2020년 이후 최신 관련 연구

### 6.1 일반화 성능 강화 연구

#### A. 정보 이론적 분석 (Sanati et al., 2023)

**연구 내용**: Information Bottleneck (IB) 이론을 사용하여 SP의 희소화 과정을 분석하고, Modified-IB 상한을 제시 [frontiersin](https://www.frontiersin.org/articles/10.3389/fncom.2023.1140782/full)

**주요 결과**:
- 2% 희소도에서 최적의 정보 보존 및 재구성 성능
- 40% 노이즈 추가 후에도 출력 특성 유지
- MNIST, Fashion-MNIST, NYC-Taxi 데이터셋에서 검증

**일반화 성능 관점에서의 의의**:
희소도와 노이즈 수준이 재구성 성능에 미치는 정량적 영향을 정보 이론으로 측정하여, 파라미터 선택의 수학적 근거 제공 [frontiersin](https://www.frontiersin.org/articles/10.3389/fncom.2023.1140782/full)

#### B. Hardware-Accelerated HTM (Bera et al., 2025)

**혁신점**: 생물학적 척수 반사(Spinal Reflex) 메커니즘에 영감을 받은 Reflex Memory (RM) 모듈 도입 [arxiv](https://arxiv.org/abs/2504.03746)

**성능 개선**:
- AHTM (소프트웨어): 7.55배 속도 향상
- H-AHTM (하드웨어): 10.10배 속도 향상
- 일반화 성능은 유지하면서 계산 효율성 극대화

**기술적 기여**: 첫 순서 시간 관계가 대부분의 패턴에서 충분함을 보이면서, 고차 추론의 계산 오버헤드 제거 가능성 입증 [arxiv](https://arxiv.org/abs/2504.03746)

#### C. 활성화 강도 기반 새로운 HTM 알고리즘 (2022)

**개선 사항**: 기존 SP의 압축 방식을 동적 활성화 강도 기반으로 개선 [hindawi](https://www.hindawi.com/journals/cin/2022/6072316/)

**성능 향상**: 시계열 데이터에서 특성 표현의 품질 증가로 일반화 성능 개선

### 6.2 고차원 데이터 처리 확장

#### 다변량 HTM (Multivariate HTM, 2024-2025)

**문제 해결**: 기존 HTM은 일변량 시계열만 처리

**해결책**: 각 차원별 SP 인스턴스 + 신경망 결합기 [ijmems](https://ijmems.in/cms/storage/app/public/uploads/volumes/39-IJMEMS-24-0522-10-3-777-796-2025.pdf)

$$\text{Output} = \text{NeuralNet}(\text{Concat}[\text{SP}_1(x_1), \text{SP}_2(x_2), \ldots, \text{SP}_d(x_d)])$$

**성능**: 다변량 이상 탐지에서 기존 심층학습 방법과 경쟁 가능 [ijmems](https://ijmems.in/cms/storage/app/public/uploads/volumes/39-IJMEMS-24-0522-10-3-777-796-2025.pdf)

#### GridHTM for Video (2023)

**확장**: 공간 그리드 기반 구조로 고차원 비디오 데이터 처리 [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC9961912/)

**의의**: 기존 HTM의 고차원 데이터 약점 극복, 이상 탐지 능력 확대

### 6.3 이상 탐지(Anomaly Detection) 응용 혁신

#### A. 이상 행동 탐지 프레임워크 (2021)

**기술**: SDR 기반 이상 점수 계산 및 롤링 정규분포 [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC7987406/)

$$\text{Anomaly Score} = 1 - \frac{\text{Overlap}(\text{predicted}, \text{actual})}{|\text{predicted}|}$$

**강점**: 
- 연속 학습으로 "새로운 정상" 자동 적응
- 기존 통계 방법 대비 낮은 오양성
- 온라인 학습으로 재학습 불필요

**응용 분야**: 금융거래 이상, 의료 신호 모니터링, IoT 센서, 네트워크 침입 탐지 [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC7987406/)

#### B. 실시간 데이터 드리프트 탐지 (2024-2025)

**결합**: HTM + Sequential Probability Ratio Test (SPRT) [ijmems](https://ijmems.in/cms/storage/app/public/uploads/volumes/39-IJMEMS-24-0522-10-3-777-796-2025.pdf)

**특징**:
- 일변량 드리프트 탐지 (비지도)
- 다변량 이상 탐지 (지도)
- 자동 임계값 적응

**성능**: 다른 드리프트 탐지 방법보다 높은 정확도 및 적응력 [ijmems](https://ijmems.in/cms/storage/app/public/uploads/volumes/39-IJMEMS-24-0522-10-3-777-796-2025.pdf)

### 6.4 신경생물학적 이론 통합

#### A. Universal Cortical Algorithm 프레임워크 (2025)

**핵심 개념**: 뇌의 모든 피질 영역은 공통의 계산 원리를 사용 [emergentmind](https://www.emergentmind.com/topics/universal-cortical-algorithm)

**4가지 기본 요소**:
1. 희소 분산 표현 (SDR)
2. 계층적 추상화
3. 시간적 예측
4. 경쟁적 Hebbian 학습

**HTM의 역할**: CLA(Cortical Learning Algorithm)로 이 네 요소를 구현 [emergentmind](https://www.emergentmind.com/topics/universal-cortical-algorithm)

**확장**: 전두엽 기능(활성 유지, 게이팅, 조절) 모델링으로 보다 완전한 뇌 모델에 접근 [emergentmind](https://www.emergentmind.com/topics/universal-cortical-algorithm)

#### B. 동적 예측 코딩 (Dynamic Predictive Coding, 2023)

**재해석**: HTM을 계층적 시퀀스 학습의 동적 예측 코딩으로 이해 [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC10880975/)

**신경과학적 연결**:
- 하위 계층: 공간-시간 수용장 학습
- 상위 계층: 더 느린 시간 척도의 추상적 표현
- 피드백 신호: 예측 오류 신호

**성능**: 이상 탐지 및 활동 회상 능력 향상 [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC10880975/)

### 6.5 지속적 학습(Continual Learning) 패러다임

#### Sparse Distributed Memory의 지속적 학습

**기여**: SDM을 기반으로 한 신경망이 지속적 학습에서 강력한 성능 발휘 [openreview](https://openreview.net/forum?id=JknGeelZJpHP)

**의의**: HTM/SP가 카타스트로픽 포겟팅(Catastrophic Forgetting) 방지의 한 해법으로 인식 [openreview](https://openreview.net/forum?id=JknGeelZJpHP)

***

## 7. 향후 연구에 미치는 영향 및 고려사항

### 7.1 학문적 영향

#### 이론적 기초 제공
이 논문은 신경과학 영감 알고리즘을 머신러닝 커뮤니티의 엄밀한 표준으로 끌어올렸다. 향후 HTM 연구는:

1. **수학적 엄밀성**: 알고리즘의 수렴성, 최적성, 오류 경계에 대한 형식적 분석 가능 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e240668b-e68d-49ae-997e-116159b21e58/1601.06116v3.pdf)
2. **비교 분석 기반 제공**: 다른 신경망 모델(CNN, RNN 등)과의 이론적 비교 용이 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e240668b-e68d-49ae-997e-116159b21e58/1601.06116v3.pdf)
3. **하이브리드 설계**: 기존 딥러닝 아키텍처와 HTM의 결합을 수학적으로 정당화 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e240668b-e68d-49ae-997e-116159b21e58/1601.06116v3.pdf)

#### 신경생물학 연결
SP의 벡터 표현은:
- 신경생물학적 실현 가능성 검증 기초 제공
- 생물학적 제약과 계산 효율성의 트레이드오프 분석 도구 제공
- 뇌의 보편적 계산 원리 이해에 기여 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e240668b-e68d-49ae-997e-116159b21e58/1601.06116v3.pdf)

### 7.2 향후 연구 시 고려할 중요 사항

#### 7.2.1 일반화 성능 강화

**적응형 파라미터 학습**:
현재 SP의 파라미터는 사전 설정이 필수인데, 향후 연구는 데이터 분포로부터 최적 파라미터를 자동 학습하는 메타러닝 접근을 탐색해야 한다.

**제안**:

$$\theta^* = \arg\min_\theta \mathcal{L}_{\text{validation}}(\text{SP}_\theta)$$

**전이 학습**:
한 데이터셋에서 학습한 SP의 permanence를 다른 도메인에 미세조정(Fine-tuning)할 가능성 탐색.

#### 7.2.2 이론적 심화

**정보 이론적 최적성**:
현재까지의 정보 이론 분석(Modified-IB)을 넘어, SP가 정보 보존과 압축 간의 Pareto 최적 지점에 위치하는지 증명하는 연구 필요. [frontiersin](https://www.frontiersin.org/articles/10.3389/fncom.2023.1140782/full)

**오류 경계**:
다양한 입력 분포에 대한 분류 오류의 상한을 도출:
$$P(\text{error}) \leq f(n, w, \sigma_s, \text{noise}, \ldots)$$

**수렴성**: 온라인 학습 상황에서 permanence 벡터의 수렴성 증명

#### 7.2.3 고차원 데이터 처리

**차원 저주 해결**:
현재 SP는 고차원 데이터에서 희소도를 유지하기 어렵다. 해결책:
- 계층적 부분공간 학습
- 구조적 프로젝션 (manifold learning)과의 결합
- 어텐션 메커니즘을 통한 중요 특성 선택 [arxiv](https://arxiv.org/abs/1406.4729)

**고차 상호작용**: 특성 간의 비선형 상호작용을 SDR에 인코딩하는 방법 개발

#### 7.2.4 신경형(Neuromorphic) 하드웨어 설계

**메모리 효율성**: 현재 permanence 행렬은 $m \times q$ 크기로, 대규모 네트워크에서 메모리 병목 발생. 압축 기법 필요.

**전력 효율성**: sp 구현의 저전력 설계를 위해:
- 시냅스 수정 주기 최적화
- 부스팅 제거/조건부 활성화 (임계값 기반)
- 부동소수점 연산을 고정소수점으로 전환

**병렬 처리**: 멀티코어/GPU에서의 효율적 병렬화 알고리즘 개발

#### 7.2.5 실제 응용 확대

**실시간 시스템**:
- 자율주행: 센서 스트림의 이상 탐지 (0.945초 → 0.094초) [arxiv](https://arxiv.org/abs/2504.03746)
- 로봇공학: 온라인 학습으로 환경 변화 적응
- 의료 모니터링: 환자 생체신호의 이상 탐지

**엣지 컴퓨팅**: 리소스 제약 환경(IoT 기기, 임베디드 시스템)에서의 최적화

**멀티모달 데이터**: 텍스트, 영상, 음성의 통합 처리

#### 7.2.6 기술 통합

**어텐션 메커니즘과의 결합**:
현재의 고정 억제 반경 대신, 입력 특성에 따른 동적 어텐션 가중치 도입

**강화학습과의 통합**:
SP의 학습된 표현을 강화학습의 상태 표현으로 활용하는 하이브리드 접근

**메타러닝**:
적은 샘플로 신속하게 새로운 패턴에 적응하는 메타-특성 학습

### 7.3 연구 활용 체크리스트

향후 연구자들은 다음 사항을 반드시 검토해야 한다:

| 항목 | 검토 사항 | 논문의 기여도 |
|------|---------|------------|
| **수학적 형식화** | 제시된 벡터 표기법과 방정식 활용 가능성 | ⭐⭐⭐⭐⭐ |
| **파라미터 최적화** | 초기화 전략과 예측 가능 성능 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e240668b-e68d-49ae-997e-116159b21e58/1601.06116v3.pdf)과 비교 | ⭐⭐⭐ |
| **특성 학습** | Probabilistic feature mapping의 확장 가능성 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e240668b-e68d-49ae-997e-116159b21e58/1601.06116v3.pdf) | ⭐⭐⭐⭐ |
| **일반화 분석** | Modified-IB 상한 [frontiersin](https://www.frontiersin.org/articles/10.3389/fncom.2023.1140782/full)과의 비교 | ⭐⭐ (후속 연구가 더 강함) |
| **실시간 성능** | 부스팅 제거 가능성 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e240668b-e68d-49ae-997e-116159b21e58/1601.06116v3.pdf)과 H-AHTM 속도 [arxiv](https://arxiv.org/abs/2504.03746) 비교 | ⭐⭐⭐⭐ |
| **고차원 데이터** | 그리드HTM [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC9961912/) 또는 다변량 HTM [ijmems](https://ijmems.in/cms/storage/app/public/uploads/volumes/39-IJMEMS-24-0522-10-3-777-796-2025.pdf)으로 확장 | ⭐⭐⭐ (확장 필요) |

***

## 8. 결론

Mnatzaganian et al. (2016)의 "A Mathematical Formalization of Hierarchical Temporal Memory's Spatial Pooler"는 신경과학에서 영감을 받은 알고리즘을 수학적으로 정당화함으로써 머신러닝 분야에서의 신뢰성을 크게 높였다.

### 8.1 주요 성과

1. **이론적 기초**: SP의 세 단계 알고리즘을 벡터 형태로 완전히 형식화하여 최적화와 하드웨어 설계의 토대 마련 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e240668b-e68d-49ae-997e-116159b21e58/1601.06116v3.pdf)

2. **실용적 가치**: Permanence update의 MLE 설명과 부스팅의 이차적 역할 규명으로 불필요한 계산 제거 가능성 제시 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e240668b-e68d-49ae-997e-116159b21e58/1601.06116v3.pdf)

3. **확장성 입증**: 비공간 데이터에서 SP의 우수한 성능(카테고리 데이터 1.73% 에러)을 통해 응용 범위 확대 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e240668b-e68d-49ae-997e-116159b21e58/1601.06116v3.pdf)

### 8.2 남은 과제

1. **일반화 성능**: 다양한 데이터 분포, 고차원 데이터에서의 견고성 검증 필요 (후속 연구들이 진행 중) [arxiv](https://arxiv.org/abs/2504.03746)

2. **이론적 완성도**: MLE 설명의 형식적 증명과 수렴성, 오류 경계 분석 필요

3. **실제 배포**: 파라미터 자동 최적화 메커니즘과 엣지 환경 최적화

### 8.3 미래 방향

2020년 이후의 진화를 볼 때, HTM의 미래는 다음 방향으로 진행될 것으로 예상된다:

- **하이브리드 접근**: 희소 아키텍처(MoE)와 신경형 컴퓨팅의 대세 속에서 HTM의 재조명 [emergentmind](https://www.emergentmind.com/topics/universal-cortical-algorithm)
- **온라인 학습의 중요성 증대**: 지속적 학습과 일반화 성능의 균형 추구 [openreview](https://openreview.net/forum?id=JknGeelZJpHP)
- **생물학적 타당성과 계산 효율성의 양립**: 신경과학 기반 설계의 실용성 입증 [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC10880975/)

이 논문이 제시한 수학적 프레임워크는 향후 10년간의 HTM 발전의 초석이 될 것이며, 특히 실시간 이상 탐지, 엣지 컴퓨팅, 지속적 학습 등의 분야에서 그 가치가 지속적으로 증명되고 있다.

***

## 참고문헌
<span style="display:none">[^1_100][^1_13][^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58][^1_59][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_70][^1_71][^1_72][^1_73][^1_74][^1_75][^1_76][^1_77][^1_78][^1_79][^1_80][^1_81][^1_82][^1_83][^1_84][^1_85][^1_86][^1_87][^1_88][^1_89][^1_90][^1_91][^1_92][^1_93][^1_94][^1_95][^1_96][^1_97][^1_98][^1_99]</span>

<div align="center">⁂</div>

[^1_1]: 1601.06116v3.pdf

[^1_2]: https://www.frontiersin.org/articles/10.3389/fncom.2023.1140782/full

[^1_3]: https://arxiv.org/abs/2504.03746

[^1_4]: https://www.hindawi.com/journals/cin/2022/6072316/

[^1_5]: https://ijmems.in/cms/storage/app/public/uploads/volumes/39-IJMEMS-24-0522-10-3-777-796-2025.pdf

[^1_6]: https://pmc.ncbi.nlm.nih.gov/articles/PMC9961912/

[^1_7]: https://pmc.ncbi.nlm.nih.gov/articles/PMC7987406/

[^1_8]: https://www.emergentmind.com/topics/universal-cortical-algorithm

[^1_9]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10880975/

[^1_10]: https://openreview.net/forum?id=JknGeelZJpHP

[^1_11]: https://arxiv.org/abs/1406.4729

[^1_12]: https://www.numenta.com/assets/pdf/biological-and-machine-intelligence/BaMI-SDR.pdf

[^1_13]: https://ieeexplore.ieee.org/document/9150821/

[^1_14]: https://www.semanticscholar.org/paper/d77238409cfb48b8518c094bc7fcef4ab939df5b

[^1_15]: https://www.semanticscholar.org/paper/5f6f7b20b25fcc84bf9447764ecdec868d0ad7cb

[^1_16]: https://link.springer.com/10.1007/s42514-020-00048-3

[^1_17]: https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/iet-its.2020.0189

[^1_18]: https://onlinelibrary.wiley.com/doi/10.1002/cpe.5452

[^1_19]: https://www.semanticscholar.org/paper/d5c9e293cd0c25858a21c3b1f81c84ae1c2551ce

[^1_20]: https://link.springer.com/10.1023/A:1022140919877

[^1_21]: https://link.springer.com/10.1023/A:1025696116075

[^1_22]: https://www.semanticscholar.org/paper/f09f49207333101368ce1ded5888b2f583b11868

[^1_23]: http://arxiv.org/pdf/2111.03456.pdf

[^1_24]: https://arxiv.org/abs/1808.05839

[^1_25]: https://arxiv.org/html/2504.03746v1

[^1_26]: https://pmc.ncbi.nlm.nih.gov/articles/PMC8803450/

[^1_27]: https://arxiv.org/pdf/1402.2902.pdf

[^1_28]: https://arxiv.org/pdf/1611.02792.pdf

[^1_29]: https://pmc.ncbi.nlm.nih.gov/articles/PMC5712570/

[^1_30]: https://pubs.aip.org/aip/acp/article/3270/1/020058/3343775/Spatial-pyramid-pooling-SPP-NET-compared-with

[^1_31]: https://www.rctn.org/vs265/HTM_CorticalLearningAlgorithms.pdf

[^1_32]: https://www.sciencedirect.com/science/article/pii/S1877050920302465

[^1_33]: https://emerginginvestigators.org/articles/22-046/pdf

[^1_34]: https://hearingbrain.org/docs/HTM_white_paper.pdf

[^1_35]: https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2023.1140782/full

[^1_36]: https://www.mukpublications.com/resources/84. Hamid Masood Khan_pagenumber.pdf

[^1_37]: https://openreview.net/forum?id=TjCDNssXKU

[^1_38]: https://www.sciencedirect.com/science/article/pii/S2211675325000053

[^1_39]: https://www.numenta.com/assets/pdf/whitepapers/hierarchical-temporal-memory-cortical-learning-algorithm-0.2.1-kr.pdf

[^1_40]: https://pdfs.semanticscholar.org/88db/218efe7c492b08a35fe2ac4cc70192998d81.pdf

[^1_41]: https://www.biorxiv.org/content/10.1101/085035v2.full-text

[^1_42]: https://arxiv.org/pdf/1601.06116.pdf

[^1_43]: https://www.biorxiv.org/content/10.64898/2025.12.13.694121v1.full.pdf

[^1_44]: https://arxiv.org/vc/arxiv/papers/1511/1511.00083v1.pdf

[^1_45]: https://www.biorxiv.org/content/10.64898/2025.12.13.694121v1.full-text

[^1_46]: https://www.biorxiv.org/lookup/external-ref?access_num=10.3389%2FFNCOM.2017.00111\&link_type=DOI

[^1_47]: https://arxiv.org/pdf/1411.4702.pdf

[^1_48]: https://arxiv.org/html/2407.17877v1

[^1_49]: https://www.biorxiv.org/content/10.1101/085035v1.full-text

[^1_50]: https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1000532

[^1_51]: https://pdfs.semanticscholar.org/d0d4/2fa6fb4ab0650854f8f8080f7b7c8a4dd88a.pdf

[^1_52]: https://arxiv.org/pdf/1607.00791.pdf

[^1_53]: https://arxiv.org/pdf/1708.01659.pdf

[^1_54]: https://ieeexplore.ieee.org/document/10040907/

[^1_55]: https://www.mdpi.com/1424-8220/23/4/1846

[^1_56]: https://ieeexplore.ieee.org/document/10429200/

[^1_57]: https://ieeexplore.ieee.org/document/10657670/

[^1_58]: https://ieeexplore.ieee.org/document/10097504/

[^1_59]: https://link.springer.com/10.1007/s11071-025-11986-4

[^1_60]: https://besjournals.onlinelibrary.wiley.com/doi/10.1111/2041-210X.14188

[^1_61]: https://academic.oup.com/ofid/article/doi/10.1093/ofid/ofad500.1204/7446800

[^1_62]: https://onlinelibrary.wiley.com/doi/10.1111/jfr3.12936

[^1_63]: https://arxiv.org/abs/2110.09132

[^1_64]: https://arxiv.org/pdf/2205.15043.pdf

[^1_65]: http://arxiv.org/pdf/2304.00737.pdf

[^1_66]: https://arxiv.org/pdf/2102.03112.pdf

[^1_67]: http://arxiv.org/pdf/2407.06346.pdf

[^1_68]: http://arxiv.org/pdf/2405.16883.pdf

[^1_69]: https://arxiv.org/pdf/2402.02855.pdf

[^1_70]: https://arxiv.org/abs/2110.00653

[^1_71]: https://www.nature.com/articles/s41524-023-01062-z

[^1_72]: https://www.linkedin.com/pulse/unsupervised-learning-market-size-2026-region-trends-2nlbe

[^1_73]: https://pdfs.semanticscholar.org/7149/ccfc3992895a3ef894e1d50ce267f4cf398b.pdf

[^1_74]: https://neurips.cc/virtual/2025/events/Competition

[^1_75]: https://www.sciencedirect.com/science/article/pii/S0893608023005014

[^1_76]: https://www.sciencedirect.com/science/article/abs/pii/S0167739X17327292

[^1_77]: https://www.studocu.com/in/document/saranathan-college-of-engineering/electrical-engineering/aiml-5-unsupervised-learning-techniques-and-algorithms/147331037

[^1_78]: https://www.ijmems.in/cms/storage/app/public/uploads/volumes/39-IJMEMS-24-0522-10-3-777-796-2025.pdf

[^1_79]: https://www.alliedmarketresearch.com/unsupervised-learning-market-A224213

[^1_80]: https://en.wikipedia.org/wiki/Sparse_distributed_memory

[^1_81]: https://www.kcl.ac.uk/nmes/assets/informatics-pdfs-2026-27/machine-learning-deep-learning-projects-2026-27.pdf

[^1_82]: https://arxiv.org/html/2601.14053v1

[^1_83]: https://pdfs.semanticscholar.org/3ee8/1a41ad78f29e1939ef8e1892919cc122f72d.pdf

[^1_84]: https://journals.plos.org/plosone/article/file?type=printable\&id=10.1371%2Fjournal.pone.0293879

[^1_85]: https://arxiv.org/pdf/2109.14868.pdf

[^1_86]: https://pdfs.semanticscholar.org/010a/29936b274e280c2f252fb3b3de1a189c079a.pdf

[^1_87]: https://arxiv.org/html/2510.05374v1

[^1_88]: https://arxiv.org/pdf/2209.03147.pdf

[^1_89]: https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0323482

[^1_90]: https://www.biorxiv.org/lookup/external-ref?access_num=10.1093%2Fnar%2Fgkac1098\&link_type=DOI

[^1_91]: https://arxiv.org/pdf/2601.14053.pdf

[^1_92]: https://journals.plos.org/plosone/article/file?id=10.1371%2Fjournal.pone.0323482\&type=printable

[^1_93]: https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0293879

[^1_94]: https://arxiv.org/html/2410.07840v2

[^1_95]: https://arxiv.org/pdf/2508.19577.pdf

[^1_96]: https://www.biorxiv.org/content/10.1101/2023.03.10.531570v1.full.pdf

[^1_97]: https://proceedings.iclr.cc/paper_files/paper/2025/file/8514a5203b87cba5e440bd62ab18f2b4-Paper-Conference.pdf

[^1_98]: https://www.nature.com/articles/s41598-025-25621-0

[^1_99]: https://www.sciencedirect.com/science/article/pii/S2772671125002621

[^1_100]: https://ieeexplore.ieee.org/document/9574505/
