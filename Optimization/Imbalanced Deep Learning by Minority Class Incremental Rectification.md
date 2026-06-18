# Imbalanced Deep Learning by Minority Class Incremental Rectification

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

본 논문은 **심각하게 불균형한(severely imbalanced) 대규모 학습 데이터**에서 발생하는 딥러닝 모델의 편향(bias) 문제를 해결하기 위해, **배치 단위의 점진적 소수 클래스 정류(batch-wise incremental minority class rectification)** 기법을 제안한다. 기존 딥러닝 방법들이 균형 데이터 또는 완만한 불균형 데이터만을 다루었던 한계를 극복하여, **1:4,162**에 이르는 극단적 불균형 비율에서도 효과적인 학습을 가능하게 한다.

### 주요 기여 (논문 직접 명시)

| 기여 | 설명 |
|------|------|
| **(I)** 대규모 불균형 딥러닝 | 기존 소규모·단일 레이블 연구와 달리, 대규모 멀티레이블 극단 불균형 문제 해결 |
| **(II)** 새로운 학습 방법 | 배치 단위 하드 샘플 마이닝을 통한 소수 클래스 점진적 정류 |
| **(III)** Class Rectification Loss (CRL) | 미니배치 기반으로 계산 복잡도를 제어하는 정규화 손실 함수 공식화 |

---

## 2. 해결하고자 하는 문제 / 제안 방법 / 모델 구조 / 성능 향상 및 한계

### 2.1 해결하고자 하는 문제

**Cross-Entropy (CE) 손실 함수의 구조적 한계:**

CE 손실은 각 샘플과 클래스를 동등하게 취급하여, 다수 클래스(majority class)가 과도하게 학습을 지배하는 **귀납적 편향(inductive bias)**을 유발한다:

$$\mathcal{L}_{ce} = -\frac{1}{n_{bs}} \sum_{i=1}^{n_{bs}} \sum_{j=1}^{n_{attr}} \log\left(p(y_{i,j} = a_{i,j} | \boldsymbol{x}_{i,j})\right) \tag{2}$$

여기서 $n_{bs}$는 미니배치 크기, $n_{attr}$는 속성 레이블 수, $a_{i,j}$는 정답 클래스이다.

이 손실은 개별 클래스 간 **구조적(inter-class geometry) 관계**를 무시하며, 소수 클래스의 결정 경계를 학습하는 데 취약하다 (Fig. 2(b) 참조).

**구체적 문제 특성:**
- 학습 데이터 규모가 매우 크고 (대규모)
- 샘플당 멀티레이블이 존재하며
- 불균형 비율이 1:1,000 이상으로 극단적이고
- 속성별 클래스 수가 가변적 (이진~55클래스)

---

### 2.2 제안 방법 (수식 포함)

#### ① 소수 클래스 정의 (Incremental Batch-Wise Class Profiling)

미니배치 내에서 속성 $j$의 클래스 분포 $\boldsymbol{h}^j = [h_1^j, \ldots, h_{|Z_j|}^j]$를 측정하고, 소수 클래스 집합 $C^j_{\min}$을 다음과 같이 정의한다:

$$\sum_{k \in C^j_{\min}} h_k^j \leq \rho \cdot n_{bs} \tag{3}$$

$\rho = 0.5$로 설정하여, 모든 소수 클래스가 배치 내 전체 샘플의 절반 이하를 차지하도록 정의한다.

---

#### ② 하드 샘플 마이닝

**클래스 수준(Class-Level):**

$$\mathcal{P}^{cls}_{c,j} = \{\boldsymbol{x}_{i,j} | a_{i,j} = c, \text{ low } p(y_{i,j} = c | \boldsymbol{x}_{i,j})\} \tag{4}$$

$$\mathcal{N}^{cls}_{c,j} = \{\boldsymbol{x}_{i,j} | a_{i,j} \neq c, \text{ high } p(y_{i,j} = c | \boldsymbol{x}_{i,j})\} \tag{5}$$

**인스턴스 수준(Instance-Level):**

$$\mathcal{P}^{ins}_{i,c,j} = \{\boldsymbol{x}_{k,j} | a_{k,j} = c, \text{ large dist}(\boldsymbol{x}_{i,j}, \boldsymbol{x}_{k,j})\} \tag{6}$$

$$\mathcal{N}^{ins}_{i,c,j} = \{\boldsymbol{x}_{k,j} | a_{k,j} \neq c, \text{ small dist}(\boldsymbol{x}_{i,j}, \boldsymbol{x}_{k,j})\} \tag{7}$$

---

#### ③ Class Rectification Loss (CRL) - 세 가지 손실 기준

**통합 학습 목적 함수:**

$$\mathcal{L}_{bln} = \alpha \mathcal{L}_{crl} + (1-\alpha)\mathcal{L}_{ce}, \quad \alpha = \eta \, \Omega_{imb} \tag{8}$$

여기서 $\Omega_{imb}$는 훈련 데이터의 클래스 불균형 척도이고, $\eta$는 교차 검증으로 추정되는 하이퍼파라미터이다.

---

**(I) 상대적 비교 (Relative Comparison) - Triplet Ranking Loss:**

$$\mathcal{L}_{crl} = \frac{\sum_T \max\left(0, \; m_j + d(\boldsymbol{x}_{a,j}, \boldsymbol{x}_{+,j}) - d(\boldsymbol{x}_{a,j}, \boldsymbol{x}_{-,j})\right)}{|T|} \tag{9}$$

**클래스 수준 거리:**

$$d(\boldsymbol{x}_{a,j}, \boldsymbol{x}_{+,j}) = |p_{a,j} - p_{+,j}|, \quad d(\boldsymbol{x}_{a,j}, \boldsymbol{x}_{-,j}) = p_{a,j} - p_{-,j} \tag{10}$$

**인스턴스 수준 거리:**

$$d(\boldsymbol{x}_{a,j}, \boldsymbol{x}_{*,j}) = \|\boldsymbol{f}_{(a,j)} - \boldsymbol{f}_{(*,j)}\|_2 \tag{11}$$

클래스 마진은 균일 원형 투영 방식으로:

$$m_j = \frac{2\pi}{|Z_j|} \tag{12}$$

---

**(II) 절대적 비교 (Absolute Comparison) - Contrastive Loss:**

$$\mathcal{L}_{crl} = \frac{1}{2}\left(\frac{1}{|P^+|}\sum_{P^+} d(\boldsymbol{x}_{a,j}, \boldsymbol{x}_{+,j})^2 + \frac{1}{|P^-|}\sum_{P^-} \max\left(m_{ac} - d(\boldsymbol{x}_{a,j}, \boldsymbol{x}_{-,j}),\; 0\right)^2\right) \tag{13}$$

---

**(III) 분포 비교 (Distribution Comparison) - Histogram Loss:**

히스토그램 $H^+ = [h_1^+, \ldots, h_\tau^+]$, $H^- = [h_1^-, \ldots, h_\tau^-]$를 구성하고:

$$h_t^+ = \frac{1}{|P^+|} \sum_{(i,j) \in P^+} \varsigma_{i,j,t} \tag{14}$$

$$\varsigma_{i,j,t} = \begin{cases} \frac{d(\boldsymbol{x}_{a,j}, \boldsymbol{x}_{+,j}) - b_{t-1}}{\Delta}, & \text{if } d(\boldsymbol{x}_{a,j}, \boldsymbol{x}_{+,j}) \in [b_{t-1}, b_t] \\ \frac{b_{t+1} - d(\boldsymbol{x}_{a,j}, \boldsymbol{x}_{+,j})}{\Delta}, & \text{if } d(\boldsymbol{x}_{a,j}, \boldsymbol{x}_{+,j}) \in [b_t, b_{t+1}] \\ 0, & \text{otherwise} \end{cases} \tag{15}$$

두 분포의 중첩을 최소화하는 CRL:

$$\mathcal{L}_{crl} = \sum_{t=1}^{\tau}\left(h_t^+ \sum_{k=1}^{t} h_k^-\right) \tag{16}$$

---

#### ④ 성능 평가 지표 (Class-Balanced Accuracy)

클래스 균형 정확도를 위한 민감도(Sensitivity) 기반 지표:

$$S_i = \frac{n_{(i,i)}}{n_i}, \quad n_i = \sum_{j=1}^{c} n_{(i,j)}, \quad i \in \{1, 2, \ldots, c\} \tag{17}$$

$$A_{bln} = \frac{1}{c}\sum_{i=1}^{c} S_i \tag{18}$$

---

### 2.3 모델 구조

```
[Class Imbalanced Data]
        ↓
[Mini-batch Sampling]
        ↓
[CNN Backbone]  ← CelebA: DeepID2(5-layer CNN)
                   X-Domain: MTCT(NIN 기반 멀티태스크)
                   CIFAR-100: CifarNet / ResNet32 / DenseNet
        ↓
[Minority Profiling]  ← Eq.(3): 배치 내 소수 클래스 식별
        ↓
[Hard Sample Mining]  ← Class-level (Score 기반)
                         Instance-level (Feature 거리 기반)
        ↓
[Class Rectification Loss] ← Lcrl (Triplet / Contrastive / Histogram)
        ↓
[Lbln = α·Lcrl + (1-α)·Lce]  ← 불균형 적응적 가중치
        ↓
[End-to-End SGD 최적화]
```

**핵심 설계 원칙:**
- CE 손실이 **단일 클래스 독립 모델링(per-sample single-class)**을 수행하는 것을 보완하여
- CRL이 **클래스 간 구조적 관계(inter-class geometry structure)**를 명시적으로 모델링

---

### 2.4 성능 향상

| 벤치마크 | 불균형 비율 | CRL vs 최고 경쟁 모델 |
|----------|------------|----------------------|
| CelebA (얼굴 속성) | 최대 1:43 | LMLE 대비 **+3%** 평균 정확도, 학습 속도 **9.7배 빠름** |
| X-Domain (의류 속성) | 최대 1:4,162 | LMLE 대비 **+4.65%** 평균 정확도, **7.1배 빠름** |
| DeepFashion | 최대 1:733 | CE loss 단독 대비 **+2.36%** |
| CIFAR-100 (균형) | 1:1 | CifarNet +3.6%, ResNet32 +1.2%, DenseNet +0.8% |

---

### 2.5 한계

논문에서 **명시적으로 언급된 한계**와 **분석에서 드러난 한계**는 다음과 같다:

1. **하이퍼파라미터 민감도:** $\eta$ (손실 가중치), $\kappa$ (하드 샘플 수), $\rho$ (소수 클래스 기준)를 교차 검증으로 별도 설정해야 한다.

2. **클래스 수준 CRL의 우위에 대한 이론적 설명 부재:** 실험적으로 class-level이 instance-level보다 우수함을 보이지만, 이유는 "CE 손실과의 호환성"이라는 직관적 설명에 그친다.

3. **Minority/Majority 클래스 분류의 임의성:** $\rho=0.5$ 설정이 이진 분류에서 차용된 것이며, 멀티클래스에 대한 이론적 최적성이 보장되지 않는다.

4. **일반화 성능과 불균형 비율의 비선형 관계:** 논문 자체에서 " $\gamma$와 모델 성능 간 명확한 추세가 없다"고 인정한다. 일반화는 클래스 분포뿐만 아니라 개별 샘플의 정보 함량에 의존한다.

5. **탐지 불가능한 소수 클래스 처리:** 배치 내 샘플이 1개 이하인 소수 클래스는 CRL 적용에서 제외된다 (각주 2).

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화를 위한 핵심 메커니즘

#### (a) 클래스 결정 경계의 구조적 정류

CRL은 소수 클래스의 **희소하게 샘플링된 결정 경계(sparsely sampled decision boundary)**를 점진적으로 발견·확장한다. 이는 CE 손실만으로는 과도하게 다수 클래스 쪽으로 치우친 결정 경계를 교정함으로써, 모델이 **보지 못한 소수 클래스 테스트 샘플에 대한 일반화 능력**을 향상시킨다 (Fig. 2(c)).

#### (b) 균형 데이터에서의 일반화 향상

흥미롭게도 **균형 데이터(CIFAR-100)**에서도 성능이 향상된다. 논문은 두 가지 이유를 제시한다:

> "① 전역적으로 균형된 데이터라도, 미니배치 랜덤 샘플링 과정에서 배치 단위 불균형이 발생할 수 있다. CRL의 배치별 클래스 균형 전략이 이를 정규화한다. ② CRL은 클래스 수준의 구조적 분리를 최적화하여, 샘플별 단일 클래스 최적화를 수행하는 CE 손실에 보완적 이점을 제공한다."

즉, **CRL은 단순한 불균형 학습 기법을 넘어 일종의 일반적 정규화 기법(regularizer)**으로 기능한다.

#### (c) 배치 정규화와의 개념적 유사성

논문은 CRL이 **Batch Normalization [Ioffe & Szegedy, 2015]** 과 유사한 원리로 학습 확장성을 달성한다고 명시한다:

> "Due to the batch-wise design, the class balancing effect by our proposed regularisor is incorporated throughout the whole training process progressively. Conceptually, our CRL shares a similar principle to Batch Normalisation in achieving learning scalability."

이는 배치 단위 정규화가 학습 과정 전반에 걸쳐 점진적으로 작용하여 **SGD 최적화의 수렴 특성을 보존하면서 일반화를 개선**함을 의미한다.

#### (d) 세밀한 클래스 구분(Fine-Grained Discrimination)에 대한 함의

논문의 실험 결과는 중요한 관찰을 보고한다:

> "클래스 훈련 데이터 분포가 모델의 세밀한 클래스 구분 능력에 영향을 미친다. 중요하게도, 불균형 데이터 학습에 효과적으로 대처하는 모델의 능력이 세밀한 클래스 구분 학습 개선에 도움이 된다."

예컨대 "Sleeve Shape" 속성(불균형 비율 1:4,115)에서 CRL이 LMLE 대비 **+10.03%** 의 성능 향상을 달성하는데, 이는 불균형 처리와 세밀한 특징 학습이 상호 보완적임을 시사한다.

### 3.2 일반화 관련 한계 및 미해결 과제

- **비선형적 일반화 특성:** $\gamma$ (불균형 정도)와 일반화 성능 간 명확한 단조적 관계가 없어, 특정 불균형 수준에서의 성능 예측이 어렵다.
- **도메인 일반화(Domain Generalization) 미탐구:** CRL이 도메인 시프트 상황에서도 유효한지는 검증되지 않았다.
- **레이블 노이즈 취약성:** 하드 샘플 마이닝은 레이블 노이즈에 민감할 수 있다. 아웃라이어 노이즈가 결정 경계 추정을 왜곡할 가능성이 있다 ($\kappa=1$ 최강 마이닝 시 성능 저하가 이를 간접적으로 시사).

---

## 4. 미래 연구에 미치는 영향 및 고려 사항

### 4.1 미래 연구에 미치는 영향

#### (a) 손실 함수 설계 패러다임 전환

CRL은 단일 CE 손실의 한계를 지적하고, **구조적 손실(structural loss)과 분류 손실의 결합**이라는 설계 방향을 제시한다. 이는 이후 Focal Loss, Class-Balanced Loss, LDAM 등의 연구에서 더욱 정교하게 발전된다.

#### (b) 멀티레이블 불균형 학습 연구 확대

기존 연구가 이진 분류·단일 레이블에 집중했던 것과 달리, CRL은 **멀티레이블, 멀티클래스, 극단적 불균형**이라는 복합 조건을 동시에 다루어 이 연구 방향의 선구적 역할을 한다.

#### (c) 하드 마이닝과 불균형 학습의 결합

온라인 하드 예제 마이닝(OHEM)과 불균형 학습을 결합한 방향성은 이후 **MisLAS, PaCo** 등의 연구에서 계승된다.

#### (d) 확장 가능한(Scalable) 엔드-투-엔드 학습 프레임워크

전처리 없이 엔드-투-엔드로 불균형 학습이 가능한 프레임워크를 제시하여, 대규모 실세계 응용의 가능성을 열었다.

---

### 4.2 2020년 이후 관련 최신 연구 비교 분석

아래는 논문에서 제안한 CRL과 2020년 이후 주요 후속 연구들의 비교이다.

> ⚠️ **주의:** 아래 비교는 해당 논문들의 공개된 arXiv/발표 정보에 기반하며, 세부 수치는 논문별 실험 설정에 따라 다를 수 있습니다.

#### (a) **Decoupling Representation and Classifier (Kang et al., ICLR 2020)**

- **방법:** 표현 학습과 분류기 학습을 분리(decoupling)하여, 표현은 불균형 데이터로, 분류기는 균형 데이터로 미세조정
- **CRL과의 관계:** CRL이 엔드-투-엔드 방식을 고수하는 반면, 이 연구는 2단계 학습이 실제로 더 유리할 수 있음을 보임. CRL의 한계(엔드-투-엔드 고집)를 간접적으로 지적
- **참고:** Kang, B. et al., "Decoupling Representation and Classifier for Long-Tailed Recognition," ICLR 2020

#### (b) **LDAM (Label-Distribution-Aware Margin Loss, Cao et al., NeurIPS 2019)**

- **방법:** 소수 클래스에 더 큰 마진을 부여하는 클래스별 마진 손실

$$\mathcal{L}_{LDAM} = -\log \frac{e^{z_{y} - \Delta_y}}{e^{z_y - \Delta_y} + \sum_{j \neq y} e^{z_j}}, \quad \Delta_y \propto n_y^{-1/4}$$

- **CRL과의 관계:** CRL이 고정된 마진($m_j = 0.5$ 또는 $2\pi/|Z_j|$)을 사용하는 것과 달리, LDAM은 클래스 빈도에 기반한 이론적으로 유도된 가변 마진을 사용하여 이론적 근거가 더 탄탄함
- **참고:** Cao, K. et al., "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss," NeurIPS 2019

#### (c) **Class-Balanced Loss (Cui et al., CVPR 2019)**

- **방법:** "유효 샘플 수(effective number of samples)" 개념을 도입하여 클래스별 가중치 설정

$$\text{Effective number} = \frac{1 - \beta^{n_i}}{1 - \beta}$$

- **CRL과의 관계:** CRL의 $\Omega_{imb}$ 기반 가중치보다 이론적으로 정교한 클래스 가중치 계산 방법을 제안. 그러나 CRL의 구조적 손실(hard mining 기반)은 단순 재가중치보다 더 많은 정보를 활용함
- **참고:** Cui, Y. et al., "Class-Balanced Loss Based on Effective Number of Samples," CVPR 2019

#### (d) **MiSLAS (Zhang et al., CVPR 2021)**

- **방법:** 믹스업(Mixup) 기반 데이터 증강과 레이블 인식 평활화를 결합한 2단계 학습
- **CRL과의 관계:** CRL이 데이터 증강 없이 순수 손실 함수 수정으로 접근하는 반면, 이 연구는 데이터 증강과 손실 설계를 결합하여 더 강력한 성능을 달성
- **참고:** Zhang, Z. et al., "Bag of Tricks for Long-Tailed Visual Recognition with Deep Convolutional Neural Networks," AAAI 2021 / Zhong et al., "Improving Calibration for Long-Tail Recognition," CVPR 2021

#### (e) **PaCo (Cui et al., ICCV 2021)**

- **방법:** 파라메트릭 대조 학습(Parametric Contrastive Learning)으로 불균형 학습 수행. 감독 대조 손실에 학습 가능한 클래스 프로토타입을 통합
- **CRL과의 관계:** CRL이 트리플렛/대조 손실을 배치 내 하드 샘플에 적용하는 것과 달리, PaCo는 **자기지도 대조 학습 패러다임**을 불균형에 적용하여 표현력이 크게 향상됨
- **참고:** Cui, J. et al., "Parametric Contrastive Learning," ICCV 2021

#### 종합 비교표

| 방법 | 접근 방식 | 이론적 근거 | 계산 효율성 | 멀티레이블 | 극단 불균형(>1:1000) |
|------|----------|------------|------------|------------|----------------------|
| **CRL (본 논문)** | 손실 정규화 + 하드 마이닝 | 중간 | 높음 (배치별) | ✅ | ✅ |
| LDAM | 마진 기반 손실 | 높음 (이론) | 높음 | 제한적 | 부분적 |
| Class-Balanced | 재가중치 | 중간 | 매우 높음 | 부분적 | 부분적 |
| Decoupling | 2단계 학습 | 중간 | 중간 | 제한적 | ✅ |
| PaCo | 대조 학습 | 높음 | 낮음 | 제한적 | ✅ |

---

### 4.3 앞으로 연구 시 고려할 점

#### (a) **이론적 일반화 보장의 강화**
CRL의 마진 설정($m_j$)이 직관적으로 설계되었으나, PAC 학습이나 Rademacher 복잡도 관점에서의 이론적 분석이 부재하다. 향후 연구는 클래스 불균형 비율과 일반화 오류 간의 이론적 관계를 규명해야 한다.

#### (b) **자기지도/대조 학습과의 결합**
SimCLR, MoCo 등의 자기지도 표현 학습을 CRL과 결합하면, 레이블이 희소한 소수 클래스의 표현력을 추가 데이터 없이 향상시킬 수 있다.

#### (c) **레이블 노이즈 강건성 연구**
하드 샘플 마이닝은 레이블 노이즈가 있는 경우 잘못된 방향으로 학습을 유도할 수 있다. **노이즈-강건(noise-robust) 하드 마이닝 전략**의 개발이 필요하다.

#### (d) **동적 불균형 비율 대응**
훈련 과정에서 $\Omega_{imb}$는 고정된 훈련 데이터의 통계를 기반으로 계산된다. 그러나 **온라인 학습이나 연속 학습(continual learning)** 환경에서는 클래스 분포가 동적으로 변화하므로, 이에 대응하는 적응적 메커니즘이 필요하다.

#### (e) **비전-언어 모델(VLM)에의 적용**
CLIP, BLIP 등의 대형 비전-언어 모델을 불균형 다운스트림 태스크에 파인튜닝할 때, CRL과 같은 구조적 손실이 효과적일 수 있다. 프롬프트 학습(prompt learning)과 CRL의 결합 가능성이 열려 있다.

#### (f) **멀티모달 불균형 학습**
텍스트, 오디오, 이미지가 결합된 멀티모달 환경에서의 불균형은 단순히 시각적 불균형보다 복잡하다. 모달리티별 소수 클래스가 상이할 수 있으므로, CRL의 확장이 요구된다.

---

## 참고 자료

### 본 분석의 직접 참고 논문
- **Dong, Q., Gong, S., & Zhu, X. (2018).** "Imbalanced Deep Learning by Minority Class Incremental Rectification." *arXiv:1804.10851v1* [cs.CV]. (제공된 PDF)

### 비교 분석에 참고한 관련 논문
- Cao, K., Wei, C., Gaidon, A., Arechiga, N., & Ma, T. (2019). "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss." *NeurIPS 2019.*
- Cui, Y., Jia, M., Lin, T.-Y., Song, Y., & Belongie, S. (2019). "Class-Balanced Loss Based on Effective Number of Samples." *CVPR 2019.*
- Kang, B., Xie, S., Rohrbach, M., Yan, Z., Gordo, A., Feng, J., & Kalantidis, Y. (2020). "Decoupling Representation and Classifier for Long-Tailed Recognition." *ICLR 2020.*
- Cui, J., Zhong, Z., Liu, S., Yu, B., & Jia, J. (2021). "Parametric Contrastive Learning." *ICCV 2021.*
- Huang, C., Li, Y., Loy, C. C., & Tang, X. (2016). "Learning Deep Representation for Imbalanced Classification." *CVPR 2016.* (논문 내 인용 [17])
- Ioffe, S., & Szegedy, C. (2015). "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift." *arXiv 2015.* (논문 내 인용 [76])

> ⚠️ **정확도 참고:** 본 답변에서 2020년 이후 최신 연구 비교 부분의 세부 수치는 해당 논문들의 공개 정보를 기반으로 하였으나, 실험 설정 차이로 인해 직접 비교 시 유의가 필요합니다. 불확실한 구체적 수치는 의도적으로 기재하지 않았습니다.
