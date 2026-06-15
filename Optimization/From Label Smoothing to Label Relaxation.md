# From Label Smoothing to Label Relaxation

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

이 논문의 핵심 주장은 다음과 같습니다:

> **Label Smoothing은 하나의 "정밀하지만 임의적인" 대체 분포를 타겟으로 강제함으로써 바람직하지 않은 편향(bias)을 유발할 수 있으며, 이를 해결하기 위해 단일 분포 대신 후보 분포들의 집합(set)을 타겟으로 사용하는 "Label Relaxation"이 더 이론적으로 타당하다.**

표준 지도학습에서 클래스 레이블 $y_i$는 암묵적으로 퇴화 분포(degenerate distribution)로 처리됩니다:

$$p_i(y | x_i) = \begin{cases} 1 & \text{if } y = y_i \\ 0 & \text{otherwise} \end{cases}$$

Label Smoothing은 이를 아래와 같이 완화합니다:

$$p^s = (1 - \alpha) p + \alpha u$$

그러나 이 $p^s$ 역시 **특정 단일 분포**로서, 실제 조건부 확률 $p^*$에 맞지 않을 수 있어 편향을 도입합니다. 논문은 이에 대한 대안으로 **Label Relaxation**을 제안합니다.

### 주요 기여

| 기여 항목 | 설명 |
|---|---|
| **이론적 프레임워크** | 불정확 확률(imprecise probabilities) 이론을 활용한 집합-값 타겟(set-valued targets) 도입 |
| **새로운 손실 함수** | KL 발산의 일반화로서 Label Relaxation(LR) 손실 함수 및 폐쇄형 표현식 제안 |
| **편향 감소** | 임의적인 단일 분포 대신 후보 분포 집합을 통해 학습 편향 원천적 감소 |
| **보정(Calibration) 향상** | 별도 데이터를 사용하는 명시적 보정 기법(Temperature Scaling)보다 더 나은 캘리브레이션 성능 |
| **실증적 검증** | MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100에서 VGG16, ResNet56, DenseNet-BC 등 다양한 아키텍처로 검증 |

---

## 2. 해결 문제, 제안 방법(수식), 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

#### 문제 1: 퇴화 분포의 과잉 확신 유발

훈련 데이터의 클래스 레이블을 퇴화 분포(one-point distribution)로 처리하면 모델이 극단적 확률 예측(0 또는 1에 가까운)을 하도록 장려되어 오버피팅과 과잉 확신(overconfidence)이 발생합니다.

#### 문제 2: Label Smoothing의 편향 도입

Label Smoothing은 이를 완화하지만, 선택된 스무딩 분포 $p^s$가 실제 조건부 확률 $p^*$와 일치하지 않을 경우 **새로운 편향**을 도입합니다. Li et al. (2020)은 이러한 편향이 일반화 성능을 해칠 수 있음을 지적했습니다.

#### 문제 3: 보정 품질 저하

현대 딥 신경망은 보정이 매우 불량하며(Guo et al. 2017), Label Smoothing은 보정을 개선하지만 명시적 보정 기법(Temperature Scaling)보다는 여전히 열등합니다(Müller et al. 2019).

---

### 2.2 제안 방법 및 수식

#### Step 1: Label Smoothing의 손실 함수 분석

Label Smoothing의 교차 엔트로피는 다음과 같이 분해됩니다:

$$H(p^s, \hat{p}) = (1 - \alpha)H(p, \hat{p}) + \alpha H(u, \hat{p})$$

$$= (1 - \alpha)H(p, \hat{p}) + \alpha \left(D_{KL}(u \| \hat{p}) + H(u)\right)$$

퇴화 분포 $p$에서 $H(p) = 0$이므로:

$$\mathcal{L}(p^s, \hat{p}) = (1 - \alpha) D_{KL}(p \| \hat{p}) + \alpha H(\hat{p}) $$

이는 예측 분포의 엔트로피를 높이도록 패널티를 부여하는 정규화 역할을 합니다. 단, $p^s$는 여전히 **임의적인 단일 분포**입니다.

#### Step 2: 가능성 분포(Possibility Distribution)를 통한 집합 표현

**불정확 확률 이론(Walley 1991)**을 활용하여, 관측된 클래스 레이블 $y_i$에 대한 가능성 분포를 정의합니다:

$$\pi_i(y) = \begin{cases} 1 & \text{if } y = y_i \\ \alpha & \text{if } y \neq y_i \end{cases}$$

이 분포와 관련된 후보 확률 분포의 집합은:

```math
Q_i^\alpha := \left\{ p \in \mathbb{P}(\mathcal{Y}) \;\middle|\; \sum_{y \neq y_i, y \in \mathcal{Y}} p(y) \leq \alpha \right\}
```

즉, 관측 클래스 $y_i$에 최소 $(1-\alpha)$의 확률을 할당하고, 나머지 클래스들의 총 확률이 $\alpha$ 이하인 모든 분포의 집합입니다.

#### Step 3: Label Relaxation 손실 함수 정의

후보 집합 $Q$와 예측 분포 $\hat{p}$의 거리를 집합 내 **최솟값**으로 정의합니다:

$$\mathcal{L}^*(Q, \hat{p}) := \min_{p \in Q} \mathcal{L}(p, \hat{p}) $$

이는 Hüllermeier and Cheng (2015)의 **Optimistic Superset Loss** 및 Cabannes et al. (2020)의 **Infimum Loss**의 특수 사례입니다.

#### Step 4: KL 발산 기반 폐쇄형 표현식

$\mathcal{L}$을 KL 발산으로 인스턴스화할 때:

$$\mathcal{L}(p, \hat{p}) := D_{KL}(p \| \hat{p}) = \sum_{y \in \mathcal{Y}} p(y) \log \frac{p(y)}{\hat{p}(y)}$$

$Q_i^\alpha$ 형태의 집합에 대해 다음과 같은 폐쇄형 표현식이 유도됩니다:

$$\mathcal{L}^*(Q_i^\alpha, \hat{p}_i) = \begin{cases} 0 & \text{if } \hat{p}_i \in Q_i^\alpha \\ D_{KL}(p_i^r \| \hat{p}_i) & \text{otherwise} \end{cases} $$

여기서 최적 투영 분포 $p_i^r$은:

$$p_i^r(y) = \begin{cases} 1 - \alpha & \text{if } y = y_i \\ \alpha \cdot \dfrac{\hat{p}_i(y)}{\sum_{y' \neq y_i} \hat{p}_i(y')} & \text{otherwise} \end{cases} $$

**직관적 해석:**
- 예측이 이미 집합 $Q_i^\alpha$ 안에 있으면 (즉, $\hat{p}_i(y_i) \geq 1-\alpha$이면) **손실 = 0**
- 그렇지 않으면, 집합 경계에 있는 가장 가까운 분포 $p_i^r$까지의 KL 발산을 최소화

이 손실의 **볼록성(convexity)**은 부록에서 증명되어 있으며, 이는 최적화를 계산적으로 실현 가능하게 합니다.

---

### 2.3 모델 구조

논문은 특정 신경망 아키텍처를 제안하는 것이 아니라, **손실 함수 수준의 개입**을 다루므로 다양한 기존 아키텍처에 적용 가능합니다.

**실험에서 사용된 아키텍처:**

| 데이터셋 | 모델 |
|---|---|
| MNIST, Fashion-MNIST | 2-layer Dense (1024 노드/층, ReLU 활성화) |
| CIFAR-10, CIFAR-100 | VGG16, ResNet56(V2), DenseNet-BC-100-12 |

**훈련 설정:**
- 옵티마이저: SGD (Nesterov momentum = 0.9)
- 배치 크기: 64
- 학습률: VGG(0.01), Dense(0.05), ResNet/DenseNet(0.1)
- 데이터 증강: 수평 뒤집기, 너비/높이 이동

---

### 2.4 성능 향상

#### 분류 정확도

Label Relaxation(LR)은 Label Smoothing(LS)보다 약간 낮거나 동등한 분류 정확도를 보이며, 다른 기법들(CE, CP, FL)과 경쟁력을 유지합니다.

**Table 3 평균 순위 요약 (전체 랭킹):**

| 손실 함수 | 정확도 평균 순위 | ECE 평균 순위 |
|---|---|---|
| CE | 3 | 3 |
| **LS** | **2** | 3.38 |
| CP | 2.31 | 4.5 |
| FL | 4 | 2.25 |
| **LR** | 2.56 | **1.56** |

#### 보정 성능 (ECE)

- LR은 대부분의 실험에서 **가장 낮은 ECE**를 달성
- 별도 데이터를 사용하는 Temperature Scaling보다도 LR이 더 나은 ECE를 기록하는 경우 다수 존재
- ResNet56(V2) + CIFAR-100에서: LR ECE = **0.017**, TS ECE = 0.041

---

### 2.5 한계

1. **단일 파라미터 의존성**: 집합 크기를 결정하는 $\alpha$ 하이퍼파라미터의 최적화가 여전히 필요하며, 이를 위한 별도의 검증 데이터가 요구됩니다.

2. **이미지 분류 중심 검증**: 실험이 이미지 분류 태스크에 집중되어 있어, 자연어 처리, 음성 인식 등 다른 도메인으로의 일반화 가능성은 검증되지 않았습니다.

3. **가능성 분포의 단순성**: 논문에서 사용된 가능성 분포는 관측 클래스에 1, 나머지에 동일한 $\alpha$ 값을 부여하는 단순한 형태로, 클래스 간 의미적 관계를 반영하지 않습니다.

4. **ResNet56에서의 불안정성**: ResNet56(V2) + CIFAR-10에서 LR의 ECE 분산이 상대적으로 높아(0.059 ± 0.090) 불안정한 면이 있습니다.

5. **다른 손실 함수 인스턴스화 미검토**: KL 발산 외의 다른 손실 함수(Brier Score 등)와의 결합은 향후 연구 과제로 남겨졌습니다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화 향상의 이론적 메커니즘

#### 편향-분산 트레이드오프 관점

Label Smoothing이 고정된 스무딩 분포 $p^s$로 편향을 도입하는 반면, Label Relaxation은 학습기(learner)가 집합 $Q_i^\alpha$ 내에서 가장 적절한 분포 $p_i^r$을 **데이터 기반으로 선택**하게 합니다:

$$p_i^r \in \underset{p \in Q_i^\alpha}{\arg\min} \; D_{KL}(p \| \hat{p}_i)$$

이를 통해:
- **편향 감소**: 임의적인 $p^s$ 대신 데이터 기반 $p_i^r$ 사용
- **과잉 확신 방지**: $\hat{p}_i \in Q_i^\alpha$이면 손실이 0이므로, 지나치게 극단적인 예측을 강요하지 않음

#### 정규화 관점에서의 해석

LR 손실의 구조적 특성:

$$\mathcal{L}^*(Q_i^\alpha, \hat{p}_i) = \begin{cases} 0 & \text{if } \hat{p}_i(y_i) \geq 1 - \alpha \\ D_{KL}(p_i^r \| \hat{p}_i) & \text{otherwise} \end{cases}$$

- ** $\epsilon$ -비민감 손실($\epsilon$-insensitive loss)**과 유사하게, 임계값 이상의 예측에 대해 손실이 0이 됨
- 이는 SVM 회귀의 $\epsilon$-tube 개념과 유사하며, 모델이 필요 이상으로 타겟에 집착하지 않도록 함

#### 데이터 명확화(Data Disambiguation) 관점

Hüllermeier and Cheng (2015)의 **Superset Learning** 프레임워크와 연결됩니다. 집합-값 타겟을 사용함으로써 학습기는 훈련 과정에서 가장 "일관된" 정밀 분포를 찾아내며, 이는 실제 조건부 확률 $p^*$에 더 가까울 것으로 기대됩니다.

### 3.2 보정을 통한 일반화

보정(calibration)된 모델은 다음과 같은 이유로 더 나은 일반화를 보입니다:

1. **확률 추정의 신뢰성**: 예측 확률이 실제 빈도와 일치할 때, 다운스트림 의사결정 태스크에서 더 신뢰할 수 있는 결과를 제공합니다.
2. **분포 이탈(distribution shift) 강건성**: 잘 보정된 모델은 훈련 분포와 테스트 분포가 다소 다를 때도 상대적으로 안정적인 성능을 보입니다.

LR이 Temperature Scaling을 능가하는 핵심 이유는 Temperature Scaling이 **별도 검증 데이터**를 사용하여 훈련 데이터를 줄이는 반면, LR은 **전체 훈련 데이터를 활용**하면서 암묵적 보정을 달성하기 때문입니다.

### 3.3 실험적 근거

**CIFAR-100 + ResNet56(V2) 결과 (ECE 기준):**

| 방법 | 정확도 | ECE |
|---|---|---|
| CE (no calibration) | 0.737 | 0.126 |
| LS (ECE opt.) | 0.730 | 0.053 |
| CE + Temperature Scaling | 0.709 | 0.041 |
| **LR (ECE opt.)** | **0.729** | **0.017** |

LR은 정확도 손실 없이 ECE를 대폭 감소시키며, 명시적 보정 기법(TS)보다도 우월한 결과를 보입니다. 이는 LR이 **분류 정확도와 확률 보정을 동시에** 최적화할 수 있음을 시사합니다.

---

## 4. 향후 연구에 미치는 영향 및 고려 사항

### 4.1 향후 연구에 미치는 영향

#### 4.1.1 손실 함수 설계의 새로운 패러다임

이 논문은 "**단일 정밀 타겟 → 집합-값 불정확 타겟**"이라는 패러다임 전환을 제안합니다. 이는 다양한 기계학습 문제에 적용 가능한 일반적 프레임워크를 제시하며, 특히:

- **회귀 문제**: 타겟 실수값 대신 구간(interval)을 타겟으로 사용
- **구조적 예측**: 구조적 출력 공간에서의 집합-값 손실 (Cabannes et al. 2020의 Infimum Loss와 연결)
- **다중 레이블 학습**: 레이블 불확실성이 있는 환경에서의 적용

#### 4.1.2 불정확 확률 이론의 머신러닝 응용 활성화

Walley (1991)의 불정확 확률 이론을 딥러닝의 손실 함수 설계에 적용한 선구적 사례로서, 이 방향의 추가 연구를 촉진할 수 있습니다.

#### 4.1.3 보정 연구와의 교차점

명시적 보정 단계 없이도 우수한 보정 성능을 달성하는 것은 의료 AI, 자율주행 등 **신뢰성이 중요한 응용 분야**에서 특히 중요하며, 이 방향의 연구 관심을 높일 수 있습니다.

---

### 4.2 2020년 이후 관련 최신 연구 비교 분석

아래는 논문에서 직접 언급되었거나 해당 분야와 연관된 주요 연구들입니다.

#### 논문에서 직접 인용된 2020년 연구들

| 연구 | 내용 | LR과의 관계 |
|---|---|---|
| **Lukasik et al. (ICML 2020)** "Does Label Smoothing Mitigate Label Noise?" | LS가 레이블 노이즈에 효과적임을 분석 | LR이 LS의 한계를 지적하는 데 이 연구를 인용; LR은 노이즈 환경에서도 잠재적 이점 |
| **Li, Dasarathy, Berisha (AISTATS 2020)** "Regularization via Structural Label Smoothing" | 구조적 정보를 이용한 LS | LS의 편향 문제를 공통으로 인식; LR은 더 일반적인 해결책 제시 |
| **Cabannes, Rudi, Bach (ICML 2020)** "Structured Prediction with Partial Labelling through the Infimum Loss" | 부분 레이블링에서의 infimum loss | LR 손실과 직접적으로 연결; LR을 infimum loss의 특수 사례로 위치 |
| **Yun et al. (CVPR 2020)** "Regularizing Class-Wise Predictions via Self-Knowledge Distillation" | 자기 지식 증류를 통한 정규화 | 유사한 일반화 목표; LR은 외부 교사 없이 타겟 자체를 유연화 |

> **⚠️ 주의**: 2020년 이후 LR을 직접 후속 연구하거나 비교한 특정 논문들(예: 특정 제목과 저자)에 대해서는, 제가 해당 논문을 직접 확인하지 않은 상태에서 구체적 수치나 제목을 제시하는 것은 정확성을 보장할 수 없어 생략합니다. 아래는 **연구 방향**만 기술합니다.

#### LR과 관련된 연구 방향들 (일반적 동향)

1. **Soft Label / Distribution Distillation 연구**: 교사 모델의 소프트 레이블을 활용하는 지식 증류(Knowledge Distillation) 연구들은 LR의 집합-값 타겟 개념과 유사한 동기를 가집니다.

2. **Conformal Prediction과의 연결**: 집합-값 예측(set-valued prediction)은 Conformal Prediction 연구와 연결되며, 불확실성 정량화(uncertainty quantification) 분야에서 주목받고 있습니다.

3. **Partial Label Learning**: 여러 후보 레이블 중 하나가 정답인 설정에서의 학습은 LR의 집합-값 타겟 철학과 일맥상통합니다.

---

### 4.3 향후 연구 시 고려할 점

#### 4.3.1 방법론적 확장

```
1. 다양한 손실 함수 인스턴스화
   - KL 발산 외에 Brier Score, JS 발산 등과의 결합 탐구
   - 각 손실 함수의 볼록성 및 수렴성 이론적 분석

2. 가능성 분포의 정교화
   - 클래스 간 의미적 유사성을 반영한 비균일 α 설정
   - 예: 클래스 계층 구조(class hierarchy) 활용
   - 훈련 과정에서 α를 동적으로 적응시키는 방법

3. 집합 구조의 다양화
   - 현재 단순한 확률 심플렉스 부분집합 Q^α_i 외에
     더 복잡한 구조의 집합 탐구
```

#### 4.3.2 적용 도메인 확장

- **자연어 처리**: 기계 번역, 텍스트 분류에서의 LR 적용 (LS는 이미 NLP에서 효과가 검증됨)
- **의료 AI**: 레이블 불확실성이 높은 의료 영상 분류에서의 적용
- **준지도 학습(Semi-supervised Learning)**: 레이블이 없는 데이터의 의사 레이블(pseudo-label)을 집합-값으로 처리

#### 4.3.3 이론적 분석 강화

$$\text{일반화 오류 상한}: \quad \mathcal{R}[\hat{f}] \leq \hat{\mathcal{R}}_{\mathcal{L}^*}[\hat{f}] + \mathcal{O}\left(\sqrt{\frac{\text{VC}(\mathcal{F})}{n}}\right)$$

- LR 손실의 **Rademacher 복잡도** 분석
- 집합 크기 $\alpha$와 일반화 오류의 이론적 관계 규명
- 레이블 노이즈 환경에서의 LR 강건성 이론적 보증

#### 4.3.4 계산 효율성

현재 제안된 폐쇄형 표현식은 효율적이지만, 더 복잡한 집합 구조나 손실 함수를 사용할 경우 $\min_{p \in Q} \mathcal{L}(p, \hat{p})$의 계산이 어려워질 수 있습니다. 근사 알고리즘이나 효율적인 투영 방법 개발이 필요합니다.

#### 4.3.5 하이퍼파라미터 $\alpha$ 자동화

- 현재 $\alpha$는 검증 세트를 통해 수동 탐색
- **적응형 $\alpha$**: 훈련 손실이나 모델 신뢰도에 따라 동적 조정
- **클래스별 $\alpha$**: 클래스마다 다른 이완 정도 적용

---

## 참고 자료

**주요 참고 자료 (논문 원문 기준):**

1. **Lienen, J., & Hüllermeier, E. (2021).** "From Label Smoothing to Label Relaxation." *Proceedings of the Thirty-Fifth AAAI Conference on Artificial Intelligence (AAAI-21)*, pp. 8583–8591.

2. **Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016).** "Rethinking the Inception Architecture for Computer Vision." *CVPR 2016.*

3. **Müller, R., Kornblith, S., & Hinton, G. E. (2019).** "When Does Label Smoothing Help?" *NeurIPS 2019*, pp. 4696–4705.

4. **Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017).** "On Calibration of Modern Neural Networks." *ICML 2017*, pp. 1321–1330.

5. **Hüllermeier, E., & Cheng, W. (2015).** "Superset Learning Based on Generalized Loss Minimization." *ECML PKDD 2015*, pp. 260–275.

6. **Cabannes, V., Rudi, A., & Bach, F. R. (2020).** "Structured Prediction with Partial Labelling through the Infimum Loss." *ICML 2020*, pp. 1230–1239.

7. **Lukasik, M., Bhojanapalli, S., Menon, A. K., & Kumar, S. (2020).** "Does Label Smoothing Mitigate Label Noise?" *ICML 2020*, pp. 6448–6458.

8. **Li, W., Dasarathy, G., & Berisha, V. (2020).** "Regularization via Structural Label Smoothing." *AISTATS 2020*, pp. 1453–1463.

9. **Walley, P. (1991).** *Statistical Reasoning with Imprecise Probabilities.* Chapman & Hall.

10. **Lin, T., Goyal, P., Girshick, R. B., He, K., & Dollár, P. (2020).** "Focal Loss for Dense Object Detection." *IEEE TPAMI* 42(2), pp. 318–327.

11. **Pereyra, G., Tucker, G., Chorowski, J., Kaiser, L., & Hinton, G. E. (2017).** "Regularizing Neural Networks by Penalizing Confident Output Distributions." *CoRR abs/1701.06548.*
