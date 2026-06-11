# Confidence Scores Make Instance-dependent Label-noise Learning Possible

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

이 논문의 핵심 주장은 **신뢰도 점수(Confidence Scores)를 활용하면 기존에 다루기 어려웠던 인스턴스 의존적 노이즈(Instance-Dependent Noise, IDN)를 실용적으로 처리할 수 있다**는 것입니다.

기존 연구들은 클래스 조건부 노이즈(Class-Conditional Noise, CCN)에 집중하거나, IDN을 다루더라도 이진 분류에만 적용 가능하거나 강한 가정을 필요로 했습니다. 본 논문은 **신뢰도 점수라는 추가 정보**를 활용하여 이 문제를 보다 약한 가정 하에서 해결합니다.

### 주요 기여 (Table 1 기준)

| 접근법 | 다중 클래스 | 전이율 식별 가능 | 무한 노이즈 |
|---|---|---|---|
| Du & Cai [2015] | ✗ | ✗ | ✓ |
| Menon et al. [2018] | ✗ | ✓ | ✓ |
| Bootkrajang & Chaijaruwanich [2018] | ✗ | ✗ | ✓ |
| Cheng et al. [2020b] | ✗ | ✓ | ✗ |
| **본 논문 (ILFC)** | **✓** | **✓** | **✓** |

1. **CSIDN 모델 제안**: 각 인스턴스-레이블 쌍에 신뢰도 점수를 부여하는 새로운 노이즈 모델 정의
2. **인스턴스 수준 순방향 보정(ILFC) 알고리즘**: 각 인스턴스별 전이 행렬을 추정하는 최초의 실용적 다중 클래스 IDN 알고리즘
3. **다중 클래스 + 무한 노이즈 + 식별 가능성**을 동시에 만족하는 최초의 방법
4. **Clothing1M** 등 실제 데이터셋에서의 우수한 성능 검증

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

**노이즈 레이블 학습(Noisy Label Learning)**에서 가장 현실적인 노이즈 형태는 IDN입니다. IDN에서는 레이블 오류 확률이 레이블뿐 아니라 인스턴스 $x$ 자체에도 의존합니다.

노이즈 전이 행렬은 다음과 같이 정의됩니다:

$$P(\bar{Y} = j | X = x) = \sum_{i=1}^{K} T_{i,j}(x) P(Y = i | X = x) $$

여기서 $T_{i,j}(x) = P(\bar{Y} = j | Y = i, X = x)$이며, 이는 $K^2$개의 함수를 입력 공간 $\mathcal{X}$ 위에서 추정해야 함을 의미합니다. 입력 공간이 매우 고차원(예: $d \sim 10^4 - 10^6$)일 경우 이는 비가역적(intractable)입니다.

#### 기존 방법들의 한계

- **손실 보정(Loss Correction)**: 고정된 전이 행렬을 가정하므로 인스턴스 수준 정보를 포함하지 않음
- **레이블 보정(Label Correction)**: IDN 상황에서 노이즈가 심한 영역에서 분류기 성능이 저하되어 오류 보정 가능성이 큼
- **샘플 선택(Sample Selection, 소손실 접근법)**: IDN 환경에서 **공변량 이동(Covariate Shift)** 발생

공변량 이동 문제를 수식으로 나타내면, 소손실 접근법이 선택하는 분포는:

$$P_{\text{selected}}(x) \neq P_{\text{global}}(x)$$

즉, 노이즈가 적은 영역에 집중되어 훈련 분포와 테스트 분포 간 차이가 발생합니다.

---

### 2.2 제안하는 방법: CSIDN + ILFC

#### 신뢰도 점수 정의

각 데이터 포인트 $(x, \bar{y})$에 대해 신뢰도 점수 $r_x$를 다음과 같이 정의합니다:

$$r_x = P(Y = \bar{y} | \bar{Y} = \bar{y}, X = x) $$

즉, 할당된 레이블이 실제로 올바를 확률입니다.

> **실용적 동기**: 딥러닝 모델에서 softmax 출력값이 이 조건을 근사적으로 만족합니다. cross-entropy 손실은 분류 보정(classification-calibrated)되고 proper composite이므로, softmax 출력을 신뢰도 점수로 활용할 수 있습니다.

---

#### 전이 행렬 추정

**대각 항 추정 (Diagonal Terms)**

$S_i := \{(x, \bar{y}, r_x) \in S | \bar{y} = i\}$에 속하는 샘플에 대해:

$$T_{i,i}(x) = P(\bar{Y} = i | Y = i, X = x) = r_x \cdot \beta_i(x) $$

여기서 $\beta_i(x) = \dfrac{P(\bar{Y} = i | X = x)}{P(Y = i | X = x)}$는 밀도 비율(density ratio)입니다.

$S_i$에 속하지 않는 샘플에 대해서는 경험적 평균으로 대체합니다:

$$\hat{T}_{i,i}(x) = \frac{1}{|S_i|} \sum_{(x', \bar{y}', r'_x) \in S_i} T_{i,i}(x') = \mu_i $$

**비대각 항 추정 (Non-diagonal Terms)**

핵심 가정: **할당된 레이블이 오류인 경우를 조건으로 할 때, 클래스 전이 확률은 인스턴스 $x$에 독립적**입니다.

$$T_{i,j}(x) = \alpha_{i,j}(1 - T_{i,i}(x)), \quad \forall i \neq j $$

여기서 $\alpha_{i,j} = P(\bar{Y} = j | \bar{Y} \neq i, Y = i)$는 인스턴스에 독립적인 상수입니다.

**앵커 포인트(Anchor Points)**를 이용한 $\alpha_{i,j}$ 추정:

```math
\alpha_{i,j} = \frac{\frac{1}{|S^*_i|} \sum_{(x, \bar{y}, r_x) \in S^*_i} h_{\text{noisy}}(x)_j}{1 - \frac{1}{|S^*_i|} \sum_{(x, \bar{y}, r_x) \in S^*_i} r_x \cdot h_{\text{noisy}}(x)_i}
```

여기서 $S^*_i = \{(x, \bar{y}, r_x) \in S | P(Y = i | X = x) \approx 1\}$은 클래스 $i$의 앵커 포인트 집합입니다.

---

#### 인스턴스 수준 순방향 보정 (ILFC) 알고리즘

**Algorithm 1: Instance-Level Forward Correction (ILFC)**

1. 노이즈 데이터로 naive 분류기 $h_{\text{noisy}}$ 훈련
2. 앵커 포인트로 $\alpha_{i,j}$ 계산 (수식 7)
3. $\beta_i(\cdot) = 1$로 초기화
4. **각 에포크마다**:
   - $\mu_i$ 업데이트 (수식 6)
   - 각 샘플 $(x, \bar{y}, r_x)$에 대해 $T[i,i] = r_x \beta_i(x)$ 계산
   - 비대각 항: $T[i,j] = \alpha_{i,j}(1 - T[i,i])$
   - 보정된 손실로 분류기 훈련: $l_T: (y, \hat{y}) \mapsto l(y, T\hat{y})$
   - $\beta_i(x)$ 업데이트: $\beta_i(x) = \frac{h_{\text{noisy},i}(x)}{h_i(x)}$

---

### 2.3 모델 구조

```
┌─────────────────────────────────────────┐
│           ILFC 전체 구조                 │
├─────────────────────────────────────────┤
│  입력: {(xi, ȳi, r_xi)} + Anchor Points │
│                                         │
│  ① Naive 분류기 (h_noisy)               │
│     - 노이즈 데이터로 훈련               │
│     - P(Ȳ|X=x) 추정                    │
│                                         │
│  ② 전이 행렬 T(x) 추정                  │
│     - 대각: T_ii(x) = r_x * β_i(x)     │
│     - 비대각: T_ij(x) = α_ij(1-T_ii)   │
│                                         │
│  ③ 주 분류기 (h)                        │
│     - 보정 손실 l_T로 훈련              │
│     - h(x) = P̂(Y|X=x) 출력            │
│                                         │
│  ④ 밀도 비율 β_i(x) 반복 갱신          │
│     β_i(x) = h_noisy_i(x) / h_i(x)    │
└─────────────────────────────────────────┘
```

---

### 2.4 성능 향상

**합성 데이터셋 결과 (동심원 3-클래스)**:
- 저노이즈($\rho = 0.25$): 모든 방법이 양호
- 중간 노이즈($\rho = 0.35$): Co-teaching과 ILFC가 다른 기준선 능가
- 고노이즈($\rho = 0.45, 0.50$): **기준선들은 붕괴, ILFC만 안정적 성능 유지**

**실제 데이터셋 결과**:

| 데이터셋 | 노이즈 수준 | ILFC 성능 |
|---|---|---|
| SVHN | IDN-25% | 가장 빠른 수렴 + 최고 정확도 |
| SVHN | IDN-45% | 다른 방법 대비 우수 |
| CIFAR10 | IDN-45% | 빠른 수렴 + 최고 정확도 |

**Clothing1M 결과 (Table 2)**:

| Method | Forward | MAE | LQ | Co-teaching | **ILFC** |
|---|---|---|---|---|---|
| Accuracy | 60.62 | 60.02 | 67.65 | 70.11 | **73.35** |

---

### 2.5 한계

1. **앵커 포인트 필요성**: 알고리즘이 앵커 포인트에 의존하며, 실제 환경에서 이를 식별하는 것이 항상 가능하지 않을 수 있음
2. **조건부 독립 가정**: 비대각 항이 $\alpha_{i,j}(x) = \alpha_{i,j}$라는 가정은 모든 실제 노이즈를 완전히 반영하지 못할 수 있음
3. **신뢰도 점수 품질 의존성**: 신뢰도 점수의 정확도가 성능에 영향을 미침 (민감도 분석에서 $\sigma = 0.6$까지는 강건하나, 한계 존재)
4. **계산 비용**: 두 개의 분류기(naive + main)를 동시에 유지하고 반복 업데이트가 필요
5. **이론적 보장의 제한**: 전이 행렬의 수렴성에 대한 엄밀한 이론적 분석이 충분히 제시되지 않음

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 공변량 이동 문제 해결을 통한 일반화

기존 소손실 접근법은 IDN 환경에서 **공변량 이동**을 야기합니다. 노이즈가 심한 영역의 샘플을 배제하면:

$$P_{\text{train}}(X) \neq P_{\text{test}}(X)$$

ILFC는 **전체 데이터를 사용하되 각 인스턴스별로 전이 행렬을 보정**함으로써, 훈련 분포와 테스트 분포 간의 불일치를 줄입니다. 이는 일반화 성능에 직결됩니다.

### 3.2 인스턴스별 손실 보정의 일반화 효과

보정된 손실:

$$l_T(y, \hat{y}) = l(y, T(x)\hat{y})$$

여기서 $T(x)$는 각 인스턴스에 맞춤화된 전이 행렬입니다. 이를 통해:

- **고노이즈 영역**: 더 강한 보정 적용 → 해당 영역의 학습 품질 향상
- **저노이즈 영역**: 약한 보정 → 기존 정보를 잘 보존
- 결과적으로 **전체 입력 공간에 걸쳐 균등한 학습** 가능

### 3.3 밀도 비율 추정의 역할

$\beta_i(x) = \frac{P(\bar{Y}=i|X=x)}{P(Y=i|X=x)}$는 노이즈 분포와 클린 분포 간의 비율입니다. 이를 반복적으로 추정함으로써:

$$T_{i,i}(x) = r_x \cdot \beta_i(x)$$

가 더 정확해지고, 이는 **보정된 손실의 정확성을 높여 일반화 성능 향상**으로 이어집니다.

### 3.4 결정 경계의 일관성

Figure 4에서 시각적으로 확인되듯, LQ 방법은 노이즈가 심한 영역에서 결정 경계가 왜곡되는 반면, ILFC는 클린 분포와 일관된 결정 경계를 유지합니다. 이는 **실제 데이터 분포에 대한 더 나은 근사**를 의미하며 일반화 성능에 기여합니다.

### 3.5 신뢰도 점수의 정규화 효과

신뢰도 점수는 사실상 **암묵적 정규화** 역할을 합니다:
- 낮은 신뢰도($r_x \approx 0$): 해당 샘플의 기여도 감소 → 노이즈 레이블의 영향 감소
- 높은 신뢰도($r_x \approx 1$): 해당 샘플이 훈련에 강하게 기여

이는 특히 **고노이즈 환경에서 과적합을 방지**하고 일반화를 향상시킵니다.

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 향후 연구에 미치는 영향

#### ① IDN 연구의 실용성 확대
CSIDN 프레임워크는 자동 레이블링 시스템(웹 크롤링, 의료 AI 등)에서 softmax 출력을 신뢰도 점수로 바로 활용할 수 있어, **실용적 IDN 연구의 문을 열었습니다**.

#### ② 전이 행렬의 인스턴스 의존적 추정 패러다임
기존의 고정 전이 행렬( $T$ ) 패러다임에서 **인스턴스별 전이 행렬( $T(x)$ )** 패러다임으로의 전환을 촉진합니다.

#### ③ 레이블 보정 및 샘플 선택과의 결합 가능성
저자들이 직접 언급한 대로, CSIDN 모델의 신뢰도 점수를 **레이블 보정 및 샘플 선택 방법과 결합**하는 연구로 확장 가능합니다.

#### ④ 약한 지도 학습과의 연계
Snorkel, Data Programming 등의 약한 지도 학습 패러다임과 CSIDN의 신뢰도 점수 개념을 연결하는 연구가 기대됩니다.

---

### 4.2 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의**: 아래에 언급하는 2020년 이후 논문들 중 일부는 본 논문의 참고문헌에 직접 언급된 것들입니다. 제가 직접 해당 논문 원문을 확인하지 않은 경우에는 과장된 주장을 삼가고, 확인 가능한 정보만 기술합니다.

#### Cheng et al. [2020a] - "Learning with Instance-dependent Label Noise: A Sample Sieve Approach"
- **본 논문 대비**: 이진 분류 및 유한 노이즈 가정에 한정
- ILFC는 이를 다중 클래스 + 무한 노이즈로 일반화

#### Cheng et al. [2020b] - "Learning with Bounded Instance and Label-dependent Label Noise" (ICML 2020)
- **본 논문 대비**: 이진 분류만 지원, 소규모 UCI 데이터셋 실험
- ILFC는 CIFAR10, SVHN, Clothing1M 등 대규모 데이터셋에서 검증

#### 관련 연구 방향 (논문 내 언급 기반)

| 연구 방향 | 본 논문과의 관계 |
|---|---|
| 레이블 보정 (Tanaka et al., 2018) | ILFC와 결합 가능성 제시 |
| Co-teaching (Han et al., 2018) | 실험 기준선으로 사용; ILFC가 고노이즈에서 능가 |
| Forward Correction (Patrini et al., 2017) | ILFC의 인스턴스 수준 확장 |

---

### 4.3 앞으로 연구 시 고려할 점

#### ① 앵커 포인트 없는 방법론 개발
현재 ILFC는 앵커 포인트에 의존합니다. 앵커 포인트 없이도 $\alpha_{i,j}$를 추정할 수 있는 방법 연구가 필요합니다. 예를 들어:
- 자기 지도 학습(Self-supervised Learning) 기반 앵커 포인트 자동 탐색
- EM(Expectation-Maximization) 기반 반복적 추정

#### ② 비대각 독립 가정의 완화
$$\alpha_{i,j}(x) = \alpha_{i,j}$$
이 가정을 완화하여 비대각 항도 인스턴스 의존적으로 추정하는 연구가 필요합니다.

#### ③ 대규모 데이터셋에서의 확장성
현재 Clothing1M에서는 ResNet-18을 사용했으나, 더 큰 모델(ViT, ResNet-50 등)과 대규모 데이터셋(ImageNet 등)에서의 확장성 검증이 필요합니다.

#### ④ 신뢰도 점수 품질 향상
신뢰도 점수의 품질이 성능에 중요하므로:
- **모델 캘리브레이션(Model Calibration)** 기술과의 결합
- Temperature Scaling, Platt Scaling 등 적용

#### ⑤ 이론적 수렴 분석 강화
현재 논문은 경험적 실험에 집중되어 있으며, $\hat{T}(x) \to T(x)$의 수렴 속도에 대한 이론적 분석이 부족합니다. 샘플 복잡도(Sample Complexity) 분석이 향후 연구에서 요구됩니다.

#### ⑥ LLM/Foundation Model 시대의 적용
대규모 언어 모델(LLM) 또는 Foundation Model의 자동 레이블링 시나리오에서 CSIDN 프레임워크를 적용하는 연구가 매우 중요해질 것입니다. 특히 GPT-4, Claude 등이 생성하는 레이블에 대한 신뢰도 점수 추출 및 ILFC 적용 가능성이 주목받을 것입니다.

#### ⑦ 레이블 보정 및 샘플 선택과의 통합
저자들이 직접 언급한 미래 방향으로, CSIDN의 신뢰도 정보를:
- **레이블 보정**: 신뢰도 점수로 가중화된 pseudo-label 생성
- **샘플 선택**: 신뢰도 기반 커리큘럼 학습

등과 통합하는 연구가 필요합니다.

---

## 참고 자료

1. **본 논문**: Berthon, A., Han, B., Liu, T., Niu, G., & Sugiyama, M. (2021). "Confidence Scores Make Instance-dependent Label-noise Learning Possible." *arXiv:2001.03772v2*

2. **참고문헌 (논문 내 인용)**:
   - Patrini, G. et al. (2017). "Making deep neural networks robust to label noise: A loss correction approach." *CVPR 2017*
   - Han, B. et al. (2018). "Co-teaching: Robust training of deep neural networks with extremely noisy labels." *NeurIPS 2018*
   - Cheng, J. et al. (2020b). "Learning with bounded instance and label-dependent label noise." *ICML 2020*
   - Cheng, H. et al. (2020a). "Learning with instance-dependent label noise: A sample sieve approach." *arXiv:2010.02347*
   - Zhang, Z. & Sabuncu, M. (2018). "Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels." *NeurIPS 2018*
   - Ghosh, A. et al. (2017). "Robust loss functions under label noise for deep neural networks." *AAAI 2017*
   - Tong Xiao et al. (2015). "Learning from massive noisy labeled data for image classification." *CVPR 2015* (Clothing1M 데이터셋)
   - Menon, A.K. et al. (2018). "Learning from binary labels with instance-dependent noise." *Machine Learning, 107(8-10)*
   - Liu, T. & Tao, D. (2015). "Classification with noisy labels by importance reweighting." *IEEE TPAMI*
   - Gneiting, T. & Raftery, A.E. (2007). "Strictly proper scoring rules, prediction, and estimation." *JASA*
