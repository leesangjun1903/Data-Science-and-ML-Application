# Tackling Instance-Dependent Label Noise via a Universal Probabilistic Model

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

이 논문은 **인스턴스 의존 노이즈(Instance-Dependent Noise, IDN)** 문제를 해결하기 위해, 학습 데이터를 "혼동 인스턴스(confusing instance)"와 "비혼동 인스턴스(unconfusing instance)"로 구분하는 **범용 확률적 모델**을 제안합니다.

기존 연구들은:
- 임시방편적(ad-hoc) 휴리스틱에 의존하거나
- 클래스 조건부 노이즈(CCN) 등 특정 노이즈 가정에 국한되어 있어

**실제 세계에서 더 일반적인 IDN 상황에 대처하지 못한다**는 점을 지적합니다.

### 주요 기여

| 기여 항목 | 설명 |
|---|---|
| 범용 확률 모델 제안 | IDN을 혼동/비혼동 인스턴스 구분으로 명시적으로 모델링 |
| DNN 구현 | 모델을 DNN으로 실현하고 혼동 확률을 학습 가능한 파라미터로 설정 |
| 교차 최적화 알고리즘 | 진짜 레이블 추정과 파라미터 업데이트를 교대로 수행 |
| 실험적 검증 | CIFAR-10, CIFAR-100, Clothing1M에서 SOTA 달성 |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

레이블 노이즈 학습에서 기존 연구들은 크게 두 범주로 나뉩니다.

- **범주 1 (휴리스틱 기반)**: 데이터 정제, 위험 재가중, 레이블 보정 등 → 노이즈 생성 과정을 명시적으로 모델링하지 않아 편향된 결과 초래 가능
- **범주 2 (노이즈 생성 가정 기반)**:
  - **RCN (Random Classification Noise)**: 레이블이 무작위로 오염 (단순)
  - **CCN (Class-Conditional Noise)**: 특정 클래스 쌍 간 혼동 (전이 행렬 사용)
  - **IDN (Instance-Dependent Noise)**: 인스턴스 특성에 따라 오염 확률이 달라짐 → **가장 현실적이나 기존 연구 부족**

본 논문은 **IDN을 범용적이고 명시적으로 모델링**하는 것이 목표입니다.

---

### 2.2 제안하는 방법 (수식 포함)

#### 레이블 노이즈 설정

훈련 샘플 $S = (\mathbf{x}_1, \ldots, \mathbf{x}_N)$에 대해 진짜 레이블 $Y = (\mathbf{y}_1, \ldots, \mathbf{y}_N)$는 관측 불가능하고, 대신 노이즈 레이블 $\tilde{Y} = (\tilde{\mathbf{y}}_1, \ldots, \tilde{\mathbf{y}}_N)$만 관측됩니다.

각 인스턴스 $\mathbf{x}_i$에 대해 이진 변수 $s_i \in \{0, 1\}$를 도입합니다:
- $s_i = 0$: 비혼동 인스턴스 → 노이즈 레이블 = 진짜 레이블
- $s_i = 1$: 혼동 인스턴스 → 노이즈 레이블이 진짜 레이블과 독립적

**비혼동 인스턴스의 조건부 확률:**

$$P(\tilde{\mathbf{y}}_i | \mathbf{y}_i, \mathbf{x}_i, s_i = 0) = \mathbb{1}\{\mathbf{y}_i = \tilde{\mathbf{y}}_i\} \tag{1}$$

**혼동 인스턴스의 조건부 확률:**

$$P(\tilde{\mathbf{y}}_i | \mathbf{y}_i, \mathbf{x}_i, s_i = 1) = P(\tilde{\mathbf{y}}_i | \mathbf{x}_i) \tag{2}$$

**노이즈 레이블의 사후 확률 (핵심 모델):**

$$P(\tilde{\mathbf{y}}_i | \mathbf{y}_i, \mathbf{x}_i) = (1 - \eta_i)\mathbb{1}\{\mathbf{y}_i = \tilde{\mathbf{y}}_i\} + \eta_i \psi_i \tag{3}$$

여기서:
- $\eta_i = P(s_i = 1 | \mathbf{x}_i)$: **혼동 확률(confusing probability)** — 학습 가능한 파라미터
- $\psi_i = P(\tilde{\mathbf{y}}_i | \mathbf{x}_i)$: **노이즈 레이블 확률** — 노이즈 레이블로 학습한 naive classifier로 추정

#### 목적 함수

최대 우도 추정(Maximum Likelihood Estimation):

$$\ell(\Theta) = \sum_{i \in [N]} \log \hat{P}(\tilde{\mathbf{y}}_i | \mathbf{x}_i; \Theta) = \sum_{i \in [N]} \log \sum_{j \in [c]} \hat{P}(\tilde{\mathbf{y}}_i, y_i^j = 1 | \mathbf{x}_i; \Theta) \tag{4}$$

여기서 $\Theta = \{\mathbf{w}, \eta_1, \ldots, \eta_N\}$이며, 핵심 분해식은:

$$\hat{P}(\tilde{\mathbf{y}}_i, y_i^j = 1 | \mathbf{x}_i; \Theta) = \left[(1 - \eta_i)\tilde{y}_i^j + \eta_i \psi_i\right] h_\mathbf{w}^j(\mathbf{x}_i) \tag{5}$$

---

### 2.3 교차 최적화 알고리즘

숨겨진 변수(진짜 레이블) 때문에 식 (4)를 직접 최적화하기 어려우므로, Jensen 부등식을 이용한 하한을 최대화합니다:

$$\underset{\Theta}{\arg\max} \sum_{i \in [N]} \sum_{j \in [c]} q_i^j \log \hat{P}(\tilde{\mathbf{y}}_i, y_i^j = 1 | \mathbf{x}_i; \Theta) \tag{6}$$

#### Predicting Step (진짜 레이블 사후 확률 추정)

$$q_i^j = \frac{1}{K_i} \left[(1 - \eta_i)\tilde{y}_i^j + \eta_i \psi_i\right] h_\mathbf{w}^j(\mathbf{x}_i) \tag{7}$$

벡터 형태:

$$\mathbf{q}_i = \frac{1}{K_i} \mathbf{h}_\mathbf{w}(\mathbf{x}_i) \ast \left[(1 - \eta_i)\tilde{\mathbf{y}}_i + \eta_i \psi_i \mathbf{1}\right] \tag{8}$$

여기서 $K_i$는 정규화 상수 (합이 1이 되도록).

#### Updating Step - 혼동 확률 $\eta_i$ 업데이트 (Projected Gradient Ascent):

$$\eta_i \leftarrow \eta_i + \alpha_1 \frac{\left[\mathbf{1} + (\psi_i \eta_i - \eta_i - 1)\tilde{\mathbf{y}}_i\right]^\top \mathbf{q}_i}{\eta_i + \epsilon} \tag{11}$$

$$\eta_i \leftarrow \min(\max(\eta_i, 0), 1) \tag{12}$$

#### Updating Step - DNN 파라미터 $\mathbf{w}$ 업데이트:

$$\mathbf{w} \leftarrow \mathbf{w} + \alpha_2 \sum_{i \in [N]} \sum_{j \in [c]} \frac{q_i^j}{h_\mathbf{w}^j(\mathbf{x}_i)} \nabla_\mathbf{w} h_\mathbf{w}^j(\mathbf{x}_i) \tag{13}$$

---

### 2.4 모델 구조

```
[입력 인스턴스 x_i]
       ↓
[DNN Backbone (ResNet-32 / ResNet-50)]
       ↓
[Softmax 분류기 h_w(x_i)]  ←→  [혼동 확률 η_i (학습 가능 파라미터)]
       ↓                              ↓
[진짜 레이블 사후확률 q_i 추정]  ←  [노이즈 레이블 확률 ψ_i (사전 추정)]
       ↓
[교차 최적화: Predicting ↔ Updating]
```

**초기화 전략**: $\eta_i^{\text{INIT}} = 0.01$ (작은 값으로 시작하여 초기 불안정성 방지)

---

### 2.5 성능 향상 결과

#### CIFAR-100 (합성 IDN)

| 방법 | 테스트 정확도(%) |
|---|---|
| Co-teaching | 45.15 ± 0.53 |
| Forward | 44.97 ± 0.77 |
| LDMI | 45.07 ± 0.42 |
| Bootstrapping | 44.52 ± 0.35 |
| Tanaka | 46.02 ± 0.42 |
| PENCIL | 45.57 ± 0.40 |
| **Ours** | **47.51 ± 0.28** |

#### CIFAR-10 (합성 IDN + CCN)

| 방법 | IDN(%) | CCN r=0.1 | CCN r=0.3 | CCN r=0.5 |
|---|---|---|---|---|
| PENCIL | 69.51 | 93.27 | 91.29 | 76.18 |
| **Ours** | **69.82** | **93.81** | **92.79** | 78.03 |

#### Clothing1M (실제 IDN)

| 방법 | 테스트 정확도(%) | 사이드 정보 |
|---|---|---|
| PENCIL | 73.49 | - |
| **Ours** | **74.02** | - |
| **Ours (+50k finetune)** | **80.68** | +50k |

---

### 2.6 한계점

1. **$\psi_i$ 추정의 의존성**: 노이즈 레이블 확률 $\psi_i$를 사전에 naive classifier로 추정해야 하므로, 이 단계에서의 오류가 전체 성능에 영향
2. **독립성 가정의 단순화**: 혼동 인스턴스에서 노이즈 레이블이 진짜 레이블과 **완전히 독립**이라고 가정 — 실제로는 어느 정도 상관관계가 있을 수 있음
3. **이진 분류 외 확장**: 일부 기존 IDN 연구가 이진 분류에 한정된 것과 달리 다중 클래스를 지원하나, 클래스 수가 매우 많을 때 스케일링 이슈 존재
4. **하이퍼파라미터 민감성**: 혼동 확률의 학습률 $\alpha_1$에 성능이 민감 (Fig. 3 참조)
5. **계산 복잡도**: 각 인스턴스마다 $\eta_i$를 독립적으로 학습하므로, 데이터셋 크기 $N$에 비례하는 추가 파라미터 필요

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화를 가능하게 하는 핵심 메커니즘

#### (a) 노이즈 레이블 과적합 방지

식 (8)에서 $\eta_i$가 트레이드오프 역할을 합니다:

$$\mathbf{q}_i = \frac{1}{K_i} \mathbf{h}_\mathbf{w}(\mathbf{x}_i) \ast \left[(1 - \eta_i)\tilde{\mathbf{y}}_i + \eta_i \psi_i \mathbf{1}\right]$$

- $\eta_i \to 0$: 노이즈 레이블을 그대로 사용 → 과적합 위험
- $\eta_i \to 1$: 모델 예측을 주로 사용 → 초기에는 불안정하지만 수렴 후 노이즈에 강건

이는 **점진적으로 노이즈로부터 벗어나는 커리큘럼 학습** 효과를 냅니다.

#### (b) DNN의 초기 학습 특성 활용

Arpit et al. (2017)의 관찰 — DNN은 초기에는 노이즈를 학습하지 않고 패턴을 먼저 학습 — 을 활용하여:
- 초기 $\eta_i = 0.01$로 설정하여 초기에는 노이즈 레이블 신뢰
- 35 epoch부터 $\eta_i$ 업데이트 시작 → DNN이 어느 정도 수렴한 후 혼동 인스턴스 식별

#### (c) 범용성 (Universality)

제안된 모델은 **CCN의 특수 경우도 포함**합니다:
- CCN에서는 $\eta_i$와 $\psi_i$가 클래스에만 의존하는 경우로 해석 가능
- CIFAR-10 CCN 실험에서도 경쟁력 있는 성능 확인 (Table 2)

#### (d) 클린 데이터와의 통합

클린 레이블 인스턴스에 대해 $\eta_i = 0$으로 고정하면 자연스럽게 통합됩니다:
- Clothing1M에서 +50k 클린 데이터 사용 시 74.02% → 77.55% 향상

#### (e) 레이블 부드럽게 처리 (Label Smoothing 효과)

식 (8)의 $(1 - \eta_i)\tilde{\mathbf{y}}_i + \eta_i \psi_i \mathbf{1}$는 레이블 스무딩과 유사하게 작동하여 모델이 과도하게 특정 레이블에 확신을 갖지 않도록 합니다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4.1 연구에 미치는 영향

#### (a) IDN 연구의 새로운 패러다임 제시
혼동/비혼동 인스턴스 구분이라는 직관적이고 해석 가능한 프레임워크는 이후 연구들이 IDN을 다루는 데 있어 유용한 출발점이 됩니다.

#### (b) 교차 최적화 프레임워크의 확장 가능성
EM 알고리즘 계열의 교차 최적화는 다른 약지도 학습(weakly supervised learning), 준지도 학습(semi-supervised learning) 등으로 확장 가능합니다.

#### (c) 혼동 확률의 해석 가능성
$\eta_i$가 클수록 해당 인스턴스가 혼동 가능성이 높다는 의미를 가지므로, 데이터 품질 분석 및 능동 학습(active learning)에 활용 가능합니다.

### 4.2 앞으로 연구 시 고려할 점

#### (a) $\psi_i$ 추정의 개선
현재 naive classifier로 추정하는 $\psi_i$를 더 정교하게 추정하는 방법 (예: 베이지안 추정, 앙상블 기반 추정) 연구 필요

#### (b) 독립성 가정 완화
혼동 인스턴스에서 $P(\tilde{\mathbf{y}}_i | \mathbf{y}_i, \mathbf{x}_i, s_i = 1) = P(\tilde{\mathbf{y}}_i | \mathbf{x}_i)$ 가정을 완화하여 진짜 레이블과의 상관관계를 반영하는 연구

#### (c) 메모리 효율 개선
$N$개의 $\eta_i$ 파라미터를 모두 저장하는 것은 대규모 데이터셋에서 부담 → 인스턴스 임베딩으로부터 $\eta_i$를 예측하는 신경망 도입 가능

#### (d) 이론적 보장 강화
현재 경험적 성능 위주의 검증 → 일반화 오차 상한(generalization error bound) 등 이론적 분석 필요

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

아래는 본 논문과 관련된 2020년 이후 주요 연구들입니다. **단, 아래 설명은 제공된 논문의 참고문헌 및 공개된 사실에 기반하며, 논문 내에 언급된 연구들을 중심으로 기술합니다.**

| 연구 | 발표 | 핵심 방법 | IDN 대응 | 본 논문과의 비교 |
|---|---|---|---|---|
| **Xia et al. (2020)** "Parts-dependent label noise" (NeurIPS) | 2020 | 인스턴스 의존 전이 행렬을 부분-의존 행렬의 가중 합으로 표현 | ✅ | 강한 구조적 가정 필요; 본 논문은 사이드 정보 불필요 |
| **Berthon et al. (2020)** "Confidence scores make IDN learning possible" | 2020 | 정답 레이블일 확률(confidence score)이 주어진다고 가정 | ✅ | 사전 confidence score 요구; 본 논문은 $\eta_i$를 학습으로 추정 |
| **Cheng et al. (2020)** "Learning with bounded instance-and label-dependent label noise" (ICML) | 2020 | 이진 분류에서 유계 노이즈 가정 | 부분적 | 이진 분류 한정; 본 논문은 다중 클래스 지원 |
| **Han et al. (2020)** "SIGUA" (ICML) | 2020 | 나쁜 데이터를 잊어버리는 방식으로 강건성 확보 | 간접적 | 휴리스틱 기반; 본 논문은 명시적 확률 모델 |

### 본 논문의 차별점 요약

$$\text{본 논문} = \underbrace{\text{명시적 IDN 모델링}}_{\text{확률론적}} + \underbrace{\text{범용성}}_{\text{CCN도 포함}} + \underbrace{\text{사이드 정보 불필요}}_{\text{실용성}} + \underbrace{\text{DNN 기반 실현}}_{\text{확장성}}$$

---

## 참고 자료 및 출처

1. **Wang, Q., Han, B., Liu, T., Niu, G., Yang, J., & Gong, C. (2021/2022).** "Tackling Instance-Dependent Label Noise via a Universal Probabilistic Model." *AAAI 2021*. arXiv:2101.05467v3.

2. **Xia, X., Liu, T., Han, B., et al. (2020).** "Parts-dependent label noise: Towards instance-dependent label noise." *NeurIPS 2020*.

3. **Berthon, A., Han, B., Niu, G., Liu, T., & Sugiyama, M. (2020).** "Confidence scores make instance-dependent label-noise learning possible." arXiv:2001.03772.

4. **Cheng, J., Liu, T., Ramamohanarao, K., & Tao, D. (2020).** "Learning with bounded instance-and label-dependent label noise." *ICML 2020*.

5. **Patrini, G., et al. (2017).** "Making deep neural networks robust to label noise: A loss correction approach." *CVPR 2017*.

6. **Han, B., et al. (2018).** "Co-teaching: Robust training of deep neural networks with extremely noisy labels." *NeurIPS 2018*.

7. **Yi, K., & Wu, J. (2019).** "Probabilistic end-to-end noise correction for learning with noisy labels." *CVPR 2019*.

8. **Tanaka, D., et al. (2018).** "Joint optimization framework for learning with noisy labels." *CVPR 2018*.

9. **Arpit, D., et al. (2017).** "A closer look at memorization in deep networks." *ICML 2017*.

10. **Xiao, T., et al. (2015).** "Learning from massive noisy labeled data for image classification." *CVPR 2015*.

> **⚠️ 주의**: 2020년 이후의 최신 연구 비교 분석 부분 중, 논문 내에 직접 인용되지 않은 연구들(예: 2021년 이후의 후속 연구들)에 대해서는 언급하지 않았습니다. 제공된 논문(arXiv:2101.05467v3)의 내용과 해당 논문의 참고문헌 범위 내에서 분석하였습니다.
