# Posterior Re-calibration for Imbalanced Datasets

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

이 논문은 불균형 데이터셋에서 학습된 신경망의 성능 저하를 **재훈련 없이** 사후(post-training) 교정만으로 해결할 수 있다고 주장합니다. 핵심 아이디어는 최적 베이즈 분류기(Optimal Bayes Classifier) 관점에서 **사후 확률 재보정(Posterior Re-calibration)**을 수행하는 것입니다.

### 주요 기여

| 기여 | 설명 |
|---|---|
| **이론적 기반** | 최적 베이즈 분류기로부터 Prior Rebalancing의 최적성 증명 |
| **KL-발산 기반 최적화** | 단일 하이퍼파라미터 $\lambda$로 precision-recall 균형 조절 |
| **효율적 탐색 알고리즘** | $\mathcal{O}(\log N)$ 복잡도의 이진 탐색으로 $\lambda$ 최적화 |
| **통합 프레임워크** | Label Prior Shift + Non-Semantic Likelihood Shift를 동시 처리 |
| **광범위한 검증** | 6개 데이터셋, 5개 아키텍처에서 SOTA 달성 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

논문은 두 가지 분포 이동(Distribution Shift) 문제를 다룹니다.

**① Label Prior Shift (레이블 사전 분포 이동)**

$$P_s(Y) \neq P_t(Y)$$

훈련 데이터의 클래스 분포가 테스트 분포와 다른 경우입니다. 예: 자연계의 Long-tail 분포, 시맨틱 분할에서 보행자 픽셀의 희소성.

**② Non-Semantic Likelihood Shift (비의미론적 우도 이동)**

$$f_s(X|Y) \neq f_t(X|Y)$$

조명, 날씨, 센서 노이즈 등으로 인해 동일 클래스의 입력 표현이 달라지는 경우입니다.

기존 방법들(CB Loss, LDAM, DRW 등)의 한계:
- 재훈련이 필요하여 새로운 테스트 분포에 대응 불가
- Precision-Recall 간의 유연한 조절 불가
- 특정 아키텍처나 손실 함수에 종속적

---

### 2.2 제안 방법 및 수식

#### Step 1: 이론적 배경 — 최적 베이즈 분류기

베이즈 결정 규칙:

$$y^* = \arg\max_{y \in \mathbb{C}} P_t(y|x) = \arg\max_{y \in \mathbb{C}} f_t(x|y)P_t(y) $$

**[정리 1]** 소스 분포 $P_s(X,Y)$에서 학습한 베이즈 분류기 $h_s(x)$가 주어졌을 때, 타겟 분포 $P_t(X,Y)$ ( $f_t(X|Y) = f_s(X|Y)$이고 $P_t(Y) \neq P_s(Y)$ )에 대한 최적 베이즈 분류기는:

$$h_t(x) = \arg\max_{y \in \mathbb{C}} \frac{P_s(y|x)P_t(y)}{P_s(y)} $$

이며, 베이즈 리스크는 $R(h_t) = P(h_t(x) \neq y)$입니다.

**증명의 핵심** (최적성):

$$1 - R(h_t) = \int_{\mathbb{R}^D} \left( \sum_{k=1}^{K} \mathbb{I}_{\Gamma_k(h_t)}(x) \frac{P_s(k|x)P_t(k)}{P_s(k)} f_s(x) \right) dx $$

결정 규칙 $h_t(x)$에 의해 적분 내부 함수가 최대화되며, 다른 어떤 결정 규칙도 더 높은 리스크를 초래합니다.

---

#### Step 2: 실용적 근사 — 재보정 사후 확률

실제로는 $P_s(Y|X)$를 완벽히 학습할 수 없으므로, 학습된 판별적 사후 확률 $P_d(y|x)$를 사용한 **재보정 사후 확률(Rebalanced Posterior)**:

$$\tilde{h}_t(x) = \arg\max_{y \in \mathbb{C}} \frac{P_d(y|x)P_t(y)}{P_s(y)} $$

여기서 $P_r(y|x) = \frac{P_d(y|x)P_t(y)}{P_s(y)}$를 재보정 사후 확률이라 정의합니다.

그러나 이 직접 적용은 소수 클래스에 대한 과도한 false positive를 유발할 수 있어 (논문 내 [13] 인용), 추가 최적화가 필요합니다.

---

#### Step 3: KL-발산 기반 최적화 (핵심 공식)

**[가설 1]** 판별 분류기 $P_d(y|x)$와 재보정 분류기 $P_r(y|x)$ 사이의 절충점을 찾으면 더 나은 분류기를 얻을 수 있다.

$$P_f^*(y|x) = \arg\min_{P_f} \left[ (1-\lambda)\mathcal{KL}(P_f, P_d(y|x)) + \lambda \mathcal{KL}(P_f, P_r(y|x)) \right] $$

- $\lambda = 0$: 원래 판별 분류기 $P_d$ 복원
- $\lambda = 1$: 재보정 분류기 $P_r$ 복원
- $0 < \lambda < 1$: 두 분류기 사이의 보간(Interpolation)

**닫힌 형태(Closed-form) 해**:

$$\boxed{P_f^*(y|x) = \frac{1}{Z(x)} \left( P_d(y|x)^{1-\lambda} P_r(y|x)^{\lambda} \right)} $$

여기서 $Z(x)$는 정규화 인수입니다.

---

#### Step 4: 오즈 비율 분석

수식 (7)에서 오즈 비율을 유도하면:

```math
\frac{P_f^*(c_{gt}|x)}{P_f^*(c_i|x)} = \frac{P_d(c_{gt}|x)}{P_d(c_i|x)} \left(\frac{P_s(c_i)}{P_s(c_{gt})}\right)^{\lambda} \left(\frac{P_t(c_{gt})}{P_t(c_i)}\right)^{\lambda}
```

이 분석에서:
- $\lambda \leq 1$: 소수 클래스의 소스 사전 비율 증폭 효과가 **감소**(아선형, sublinear)
- $\lambda \geq 1$: 소수 클래스의 소스 사전 비율 증폭 효과가 **증가**(초선형, superlinear)

이를 통해 $\lambda$가 소수 클래스와 다수 클래스 간의 결정 경계를 조절하는 메커니즘을 이론적으로 설명합니다.

---

#### Step 5: 우도 평탄화 (Likelihood Flattening)

**[가설 2]** 소프트맥스 활성화 판별 모델에서 온도 스케일링(Temperature Scaling)은 클래스 조건부 우도 $f(x|y)$를 평탄화하는 것과 수치적으로 동등합니다.

$$\mathbf{P}(Y|x) = \text{Softmax}\left([l_1, \ldots, l_{N_c}] * \delta\right) $$

여기서 $\delta$가 작을수록 소프트맥스 출력이 평탄해지며, 이는 불확실성 증가를 의미합니다. UNO [19]의 입력 의존 온도 파라미터 $\delta(x)$를 활용하여 다중 모달 융합에 적용합니다.

---

#### Step 6: 통합 알고리즘 (UNO-IC)

```
Algorithm 1: UNO-IC
입력: 테스트 데이터 {x} ∈ D_test
출력: arg max_y P_f(Y|x)

For each modality m:
  1. Likelihood Flattening: P^m_d(Y|x) = Softmax(L^m_d(Y|x) * δ_m(x))
  2. Prior Rebalancing: P^m_r(Y|x) = P^m_d(Y|x) * P^m_r(Y) / P^m_s(Y)
  3. Calibrated Posterior: P^m_f(Y|x) = (1/Z(x))(P^m_d^(1-λ) * P^m_r^λ)

Noisy-Or Fusion: P_f(Y|x) = (1/Z(x))(1 - ∏_m(1 - P^m_f(Y|x)))
```

---

#### Step 7: $\lambda$ 탐색 알고리즘

검증 세트에서의 성능이 $\lambda$에 대해 **오목 함수(concave)**를 보이는 경험적 관찰을 기반으로 수정된 이진 탐색을 사용합니다.

- 시간 복잡도: $\mathcal{O}(\log N)$ ($N$: 탐색 범위의 $\lambda$ 수)
- 탐색 범위: $L=0.0$, $H=2.0$, $\text{prec}=0.1$

---

### 2.3 모델 구조

본 논문은 특정 신경망 구조를 새로 설계하지 않습니다. 대신 **사후 교정(Post-training Calibration)** 방법으로서 기존 모델에 독립적으로 적용됩니다.

실험에 사용된 아키텍처:

| 데이터셋 | 아키텍처 |
|---|---|
| Two Moon, Circle | 3-layer FCN |
| CIFAR-10/100 | ResNet-32 |
| iNaturalist18 | InceptionV3, ResNet-50 |
| Synthia | DeepLab |

---

### 2.4 성능 향상

**CIFAR 실험 (Top-1 Validation Error ↓)**

| 방법 | CIFAR-10 (LT-100) | CIFAR-100 (LT-100) | AVE |
|---|---|---|---|
| CE (베이스라인) | 28.48 | 62.16 | 37.70 |
| LDAM-DRW | 23.38 | 57.77 | 33.62 |
| BNN | 20.18 | 57.44 | — |
| **CE-IC (제안)** | **20.14** | 59.08 | 32.95 |
| **CE-DRW-IC (제안)** | **18.91** | **56.89** | **32.00** |

**iNaturalist2018 (Validation Error ↓)**

| 방법 | InceptionV3 | ResNet-50 |
|---|---|---|
| CE-DRW | 35.35 | 33.73 |
| BNN | — | 33.71 |
| **CE-IC (제안)** | **34.16±0.03** | **32.16±0.41** |

**Synthia 시맨틱 분할 (in-distribution)**

| 방법 | mIOU ↑ | mACC ↑ |
|---|---|---|
| CE (Unweighted) | 84.48 | 88.59 |
| CE (Median Freq.) | 76.85 | **97.89** |
| **CE-IC (제안)** | **84.71** | 94.93 |

CE-IC는 mIOU를 거의 유지하면서 mACC를 크게 향상시킵니다.

---

### 2.5 한계점

1. **검증 세트 의존성**: 최적 $\lambda$ 탐색을 위한 레이블된 검증 세트가 반드시 필요합니다.
2. **이상적 가정**: 정리 1은 $f_t(X|Y) = f_s(X|Y)$ (우도 동일)라는 이상적 조건을 가정하며, 현실에서는 이 조건이 완전히 충족되지 않습니다.
3. **단일 하이퍼파라미터**: $\lambda$가 모든 클래스에 동일하게 적용되므로, 클래스별 개별 조정이 불가능합니다.
4. **직접 적용의 한계**: 수식 (5)의 직접 적용($\lambda=1$)은 소수 클래스에 대한 과도한 false positive를 유발합니다.
5. **테스트 분포 사전 지식**: 테스트 사전 확률 $P_t(y)$를 어느 정도 알거나 가정해야 합니다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화를 가능하게 하는 핵심 메커니즘

**① 재훈련 없는 사후 교정**

$$P_f^*(y|x) = \frac{1}{Z(x)} \left( P_d(y|x)^{1-\lambda} P_r(y|x)^{\lambda} \right)$$

이 공식은 기학습된 모델의 **표현 능력(representation capacity)** 은 그대로 유지하면서, 출력 분포만을 조정합니다. 이로 인해 훈련 데이터에 대한 과적합 없이 새로운 테스트 분포에 적응할 수 있습니다.

**② 결정 경계의 체계적 이동**

오즈 비율 분석 (수식 8):

```math
\frac{P_f^*(c_{gt}|x)}{P_f^*(c_i|x)} = \frac{P_d(c_{gt}|x)}{P_d(c_i|x)} \left(\frac{P_s(c_i)}{P_s(c_{gt})}\right)^{\lambda} \left(\frac{P_t(c_{gt})}{P_t(c_i)}\right)^{\lambda}
```

$\lambda$를 통해 결정 경계가 소수 클래스로부터 체계적으로 멀어지며, 이는 소수 클래스의 **recall**을 향상시킵니다. 장난감 데이터셋 실험에서 이 경계 이동이 시각적으로 검증되었습니다.

**③ 다양한 분포 이동에 대한 강건성**

UNO-IC 통합 알고리즘은 Synthia의 7가지 미보유(out-of-distribution) 날씨 조건(안개, 비, 겨울 등)에서도 효과적임이 실험적으로 검증되었습니다:

| 방법 | AVE mIOU | AVE mACC |
|---|---|---|
| Baseline SoftAve | 78.46 | 82.03 |
| UNO | 79.12 | 82.74 |
| IC ($\lambda=0.4$) | 78.55 | 87.92 |
| **UNO-IC ($\lambda=0.4$)** | **78.55** | **90.45** |

**④ 아키텍처 독립성**

IC 방법은 확률적 분류 출력을 사용하는 어떤 모델에도 적용 가능하며, ResNet-32부터 InceptionV3, DeepLab까지 모두에서 효과적이었습니다. 이는 특정 아키텍처 가정 없이 일반화됨을 의미합니다.

**⑤ 명시적-암묵적 레이블 사전 이동 모두 처리**

논문은 두 유형의 레이블 사전 이동을 구분합니다:

- **명시적**: $P_s(Y) \neq P_t(Y)$ (분류 태스크)
- **암묵적**: 클래스 평균 메트릭(mACC, mIOU) 사용 시 발생하는 이동 (시맨틱 분할)

$$mACC(X,Y) = \frac{1}{K}\sum_{k=1}^{K}\frac{1}{N_k}\sum_{i=1}^{N}\mathbb{I}(h(x_i)=k, y_i=k) = \mathbb{E}_{Y \sim \mathcal{U}(1/K)}[acc(Y)] $$

이 통찰은 시맨틱 분할과 같은 태스크에서도 IC 방법이 효과적인 이유를 설명합니다.

### 3.2 일반화의 한계

그러나 일반화 성능에는 다음과 같은 제약이 있습니다:

1. **학습된 표현의 품질에 의존**: IC는 기학습된 모델의 $P_d(y|x)$를 조정하는 방법이므로, 기반 모델이 클래스 조건부 특징을 제대로 학습하지 못했다면 효과가 제한적입니다.

2. **테스트 분포 추정 필요**: 최적 $\lambda$는 검증 세트가 테스트 분포를 충분히 대표해야 찾을 수 있습니다. 검증-테스트 분포 불일치 시 최적이 아닐 수 있습니다.

3. **극단적 불균형에서의 한계**: 어떤 클래스가 훈련 데이터에 극히 드물게 등장하면, $P_d(y|x)$의 추정 자체가 부정확하여 재보정의 효과가 제한됩니다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려사항

### 4.1 연구에 미치는 영향

**① 사후 교정 방법론의 정착**

이 논문은 불균형 학습에서 **재훈련 없는 사후 교정**이 재훈련 기반 방법들과 경쟁하거나 능가할 수 있음을 보여주었습니다. 이는 이후 연구들이 훈련 단계와 추론 단계를 분리하여 생각하도록 하는 패러다임 전환에 기여합니다.

**② 베이지안 관점의 통합적 프레임워크 제시**

Label Prior Shift와 Likelihood Shift를 동일한 베이즈 관점에서 통합한 프레임워크는 이후 다양한 분포 이동 문제를 통합적으로 다루는 연구의 기반이 됩니다.

**③ 시맨틱 분할에서 불균형 문제의 재인식**

암묵적 레이블 사전 이동 개념(수식 12)은 시맨틱 분할에서 mIOU와 mACC 간의 trade-off를 제어할 수 있는 방법을 제시하며, 자율주행, 의료 영상 분석 등 안전 중요 응용에 큰 시사점을 줍니다.

**④ 대규모 불균형 데이터셋 연구 촉진**

iNaturalist18(8,142개 클래스, 불균형 비율 500)과 같은 극단적 불균형 설정에서의 성능 향상은 생물 다양성 모니터링, 의료 진단 등 실세계 long-tail 분포 문제 연구를 촉진합니다.

---

### 4.2 앞으로 연구 시 고려할 점

**① 클래스별 $\lambda$ 개인화**

현재 단일 $\lambda$를 모든 클래스에 적용하지만, 클래스별 불균형 정도가 다를 경우 클래스별 개별 $\lambda_k$를 학습하는 방향이 더 효과적일 수 있습니다:

$$P_f^*(y=k|x) \propto P_d(y=k|x)^{1-\lambda_k} P_r(y=k|x)^{\lambda_k}$$

**② 테스트 시간 적응(Test-Time Adaptation)과의 결합**

검증 세트 없이도 $\lambda$를 결정하기 위해, 배치 정규화 통계나 엔트로피 최소화와 같은 TTA 기법과 결합하는 연구가 필요합니다.

**③ Self-Supervised/Foundation Model과의 통합**

사전학습된 대규모 모델(예: CLIP, DINO 등)의 표현을 활용하여 IC 방법의 기반 모델 품질을 향상시키는 연구가 가능합니다.

**④ 연속 학습(Continual Learning)에의 응용**

새로운 클래스가 추가되거나 분포가 변화하는 연속 학습 설정에서, IC 방법을 통해 재훈련 없이 새로운 prior를 반영하는 방향을 탐색할 수 있습니다.

**⑤ 불확실성 추정과의 결합**

$\lambda$ 최적화 시 예측 불확실성을 함께 고려하면, 분포 이탈(OOD) 샘플에 대한 더 강건한 처리가 가능할 것입니다.

**⑥ 다중 레이블 분류로의 확장**

현재 방법은 단일 레이블 분류를 가정하므로, 다중 레이블 설정에서의 이론적 확장이 필요합니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

아래는 본 논문 이후 관련 분야의 주요 연구 흐름입니다. 단, **논문 PDF에서 직접 인용된 내용이 아닌 부분**은 제 학습 데이터에 기반한 것임을 명시합니다. 일부 세부 수치는 부정확할 수 있으므로 원문 확인을 권장합니다.

### 5.1 Logit Adjustment 계열

**Menon et al., "Long-tail learning via logit adjustment" (ICLR 2021)**

$$f_y(x) \leftarrow f_y(x) + \tau \log \pi_y$$

여기서 $\pi_y$는 클래스 $y$의 훈련 사전 확률이며, $\tau$는 조정 강도입니다. 이 방법은 IC 방법과 유사한 사후 로짓 조정을 수행하지만, 소프트맥스 이전 단계에서 직접 조정한다는 차이가 있습니다.

**비교**: IC는 소프트맥스 이후 확률 공간에서 조정하는 반면, Logit Adjustment는 로짓 공간에서 조정합니다. IC가 더 유연한 $\lambda$ 파라미터를 통해 trade-off를 제어할 수 있습니다.

### 5.2 Decoupled Training 계열

**Kang et al., "Decoupling Representation and Classifier for Long-tailed Recognition" (ICLR 2020)**

표현 학습과 분류기 학습을 분리하여, 균형 잡힌 미세 조정(cRT, $\tau$-normalization 등)이 효과적임을 보였습니다. IC는 이러한 decoupled 접근법의 교정 단계를 더 이론적으로 뒷받침합니다.

### 5.3 Class-Conditional Feature 재보정 계열

**Zhang et al., "Distribution Alignment: A Unified Framework for Long-tail Visual Recognition" (CVPR 2021)**

클래스 분포 정렬을 훈련과 추론 모두에서 고려하는 통합 프레임워크를 제안했습니다.

### 5.4 비교 요약 테이블

| 방법 | 재훈련 필요 | 아키텍처 독립 | Prior-Likelihood 동시처리 | 이론적 최적성 |
|---|---|---|---|---|
| LDAM-DRW (2019) | ✓ | ✗ | ✗ | 부분적 |
| **IC (본 논문, 2020)** | **✗** | **✓** | **✓** | **✓ (Bayes)** |
| Logit Adjustment (2021) | ✗ | ✓ | ✗ | ✓ |
| Decoupling (2020) | 부분적 | ✓ | ✗ | ✗ |
| Distribution Alignment (2021) | ✓ | ✓ | 부분적 | ✗ |

### 5.5 공통적 연구 트렌드

2020년 이후 관련 연구들이 공통적으로 보이는 경향:
1. **훈련-추론 분리**: 훈련 단계에서 표현 학습에 집중하고, 추론 단계에서 분포 교정을 수행
2. **이론적 정당화**: 단순 휴리스틱이 아닌 정보이론, 베이즈 이론에 기반한 방법 선호
3. **효율성 강조**: 대규모 데이터셋에서도 적용 가능한 낮은 계산 비용

---

## 참고 자료

**직접 참고한 원문 논문 (제공된 PDF):**
- Tian, J., Liu, Y.-C., Glaser, N., Hsu, Y.-C., & Kira, Z. (2020). **Posterior Re-calibration for Imbalanced Datasets**. *NeurIPS 2020*. arXiv:2010.11820v1

**논문 내 인용된 주요 참고문헌:**
- [1] Cui et al., "Class-Balanced Loss Based on Effective Number of Samples," CVPR 2019
- [2] Cao et al., "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss," NeurIPS 2019
- [9] Lin et al., "Focal Loss for Dense Object Detection," ICCV 2017
- [10] Zhou et al., "BBN: Bilateral-Branch Network with Cumulative Learning for Long-Tailed Visual Recognition," arXiv 2019
- [14] Platt et al., "Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods," 1999
- [15] Saerens et al., "Adjusting the Outputs of a Classifier to New A Priori Probabilities," Neural Computation 2002
- [16] Abbas, "A Kullback-Leibler View of Linear and Log-Linear Pools," Decision Analysis 2009
- [18] Guo et al., "On Calibration of Modern Neural Networks," ICML 2017
- [19] Tian et al., "UNO: Uncertainty-aware Noisy-Or Multimodal Fusion," arXiv 2019

**비교 분석에 참고한 후속 연구 (학습 데이터 기반, 원문 확인 권장):**
- Menon et al., "Long-tail learning via logit adjustment," ICLR 2021
- Kang et al., "Decoupling Representation and Classifier for Long-tailed Recognition," ICLR 2020
- Zhang et al., "Distribution Alignment: A Unified Framework for Long-tail Visual Recognition," CVPR 2021
