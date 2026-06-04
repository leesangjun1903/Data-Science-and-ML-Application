# Addressing the Curse of Imbalanced Training Sets: One-Sided Selection

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

Kubat & Matwin (1997)의 이 논문은 **클래스 불균형 문제(class imbalance problem)**를 정면으로 다룬 선구적 연구입니다. 핵심 주장은 다음과 같습니다:

> *다수 클래스(majority class)의 예제가 과도하게 많을 때, 학습기(learner)의 성능이 소수 클래스(minority class)에 대해 심각하게 저하되며, 이를 해결하기 위해 다수 클래스 예제 중 경계선(borderline)·잡음(noisy)·중복(redundant) 샘플만 선택적으로 제거하는 **단방향 선택(One-Sided Selection, OSS)** 기법이 효과적이다.*

### 주요 기여

| 기여 항목 | 설명 |
|-----------|------|
| 불균형 문제의 이론적 분석 | 1-NN, 베이즈 분류기, 결정 트리가 불균형 데이터에서 왜 실패하는지 수학적으로 설명 |
| 적절한 평가 기준 제안 | 단순 정확도 대신 $g$-mean(기하 평균) 및 ROC 커브 활용 |
| OSS 알고리즘 제안 | Tomek Links + Consistent Subset 결합으로 다수 클래스만 선택적 제거 |
| 실험적 검증 | 위성 이미지, UCI 데이터셋 등 7개 도메인에서 1-NN 및 C4.5로 검증 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

#### 문제 1: 부적절한 평가 기준

기존의 평균 **정확도(accuracy)**는 불균형 데이터에서 무의미합니다.

$$acc = \frac{a + d}{a + b + c + d}$$

여기서 혼동 행렬(confusion matrix)의 항목은 다음과 같습니다:

| | 예측: 음성 | 예측: 양성 |
|---|---|---|
| **실제: 음성** | $a$ | $b$ |
| **실제: 양성** | $c$ | $d$ |

예를 들어, 양성 예제가 0.2%인 경우 항상 음성으로 예측해도 정확도는 **99.8%**에 달하지만 실제로는 쓸모없는 분류기입니다.

#### 문제 2: 다수 클래스의 압도

- **1-NN 관점**: 양성 예제가 희소할수록 임의 예제의 최근접 이웃이 음성일 확률이 높아짐

$$\lim_{|negative| \to \infty} P(\text{nearest neighbor is negative}) \to 1$$

- **베이즈 분류기 관점**: 순수 베이즈 분류기는 $P(+)p_+(x) > P(-)p_-(x)$일 때 양성으로 분류하는데, 희소 양성 데이터에서는 $P(-) \gg P(+)$이므로 이 조건이 거의 충족되지 않음

- **결정 트리 관점**: 양성 예제가 극소수인 영역은 대부분 음성으로 레이블링되어 과적합(overfitting) 발생

---

### 2.2 제안 방법 (수식 포함)

#### 평가 기준: $g$-mean

논문은 평균 정확도 대신 양성/음성 클래스 각각의 정확도의 기하 평균을 사용합니다:

$$g = \sqrt{a^+ \cdot a^-}$$

여기서:

$$a^+ = \frac{d}{c + d} \quad \text{(양성 예제에 대한 정확도, Recall)}$$

$$a^- = \frac{a}{a + b} \quad \text{(음성 예제에 대한 정확도)}$$

$g$가 높으려면 두 클래스 모두에서 정확도가 균형 있게 높아야 합니다. 한 클래스에서만 높으면 기하 평균이 낮아집니다.

또한 정보 검색 분야에서 사용하는 지표도 소개됩니다:

$$\text{Precision} = p = \frac{d}{b + d}, \quad \text{Recall} = r = \frac{d}{c + d}$$

$$F\text{-measure} = \sqrt{p \cdot r}$$

---

#### Tomek Links 정의

두 예제 $\mathbf{x}$와 $\mathbf{y}$가 서로 다른 클래스 레이블을 가질 때, $\delta(\mathbf{x}, \mathbf{y})$를 거리라 하면, 다음 조건을 만족할 때 $(\mathbf{x}, \mathbf{y})$는 **Tomek Link**라 합니다:

$$\nexists \, \mathbf{z} \text{ such that } \delta(\mathbf{x}, \mathbf{z}) < \delta(\mathbf{x}, \mathbf{y}) \text{ or } \delta(\mathbf{y}, \mathbf{z}) < \delta(\mathbf{y}, \mathbf{x})$$

즉, 두 샘플 사이에 다른 어떤 예제도 존재하지 않는 **가장 가까운 이종(異種) 쌍**입니다. Tomek Link에 참여하는 예제들은 경계선(borderline)이거나 잡음(noisy)입니다.

---

#### OSS 알고리즘

논문에서 제시하는 알고리즘(Table 2)은 다음과 같습니다:

> **Algorithm: One-Sided Selection**
>
> 1. $S$를 원래 훈련 집합으로 설정
> 2. $C$를 $S$의 모든 양성 예제와 임의로 선택된 음성 예제 1개로 초기화
> 3. $C$의 예제들로 1-NN 규칙을 사용해 $S$를 분류; 오분류된 예제들을 $C$로 이동 (이제 $C$는 $S$와 일관성 유지하면서 더 작음)
> 4. $C$에서 Tomek Links에 참여하는 **음성 예제** 모두 제거; 모든 양성 예제는 유지. 결과 집합을 $T$라 함

$$S \xrightarrow{\text{Consistent Subset}} C \xrightarrow{\text{Tomek Link 제거}} T$$

이 알고리즘은 **단방향(one-sided)**으로만 작동합니다: 음성 예제만 제거하고, 양성 예제는 항상 보존합니다.

---

### 2.3 모델 구조

OSS는 독립적인 분류기가 아니라 **전처리(pre-processing) 기법**입니다. 이를 두 단계 구조로 정리하면:

```
원본 데이터 S (불균형)
        │
        ▼
[단계 1] Consistent Subset 생성
  - 1개의 음성 + 모든 양성 → C 초기화
  - 1-NN으로 S 재분류
  - 오분류된 샘플 → C 추가
  (중복 음성 제거)
        │
        ▼
[단계 2] Tomek Links 제거
  - 경계선·잡음 음성 예제 제거
  (불확실 영역 음성 제거)
        │
        ▼
균형화된 훈련 집합 T
        │
        ▼
기존 분류기 (1-NN, C4.5 등)
```

음성 예제의 네 가지 분류:

| 유형 | 설명 | OSS 처리 |
|------|------|----------|
| Class-label noise | 잘못된 레이블 | 제거 |
| Borderline | 경계 근처 | 제거 (Tomek) |
| Redundant | 다른 예제로 대체 가능 | 제거 (Consistent Subset) |
| Safe | 미래 분류에 유용 | 유지 |

---

### 2.4 성능 향상 및 한계

#### 성능 향상

논문의 실험 결과를 요약하면:

**Oil-slick I 도메인 (1-NN):**

| 집합 | $g$ | $a^+$ | $a^-$ |
|------|-----|--------|--------|
| $S$ (전체) | 44.3 | 20.8 | 94.4 |
| $T$ (OSS 적용) | **90.6** | **87.5** | **93.7** |

$g$가 44.3 → 90.6으로 약 **46% 향상**되었습니다.

**Oil-slick II 도메인 (C4.5):**

| 집합 | $g$ |
|------|-----|
| $S$ | 49.5 |
| $T$ | **66.0** |

약 **16% 이상** 향상되었습니다.

**벤치마크 도메인 (vehicles, g7, vw0):**
- vehicles 도메인: 1-NN과 C4.5 모두 유의미한 향상
- glass(g7) 도메인: 1-NN에서 소폭 향상, C4.5에서는 오히려 소폭 하락

#### 한계

1. **이분류(binary classification) 문제에만 실험**: 다중 클래스 문제로의 일반화는 이론적 주장에 그침
2. **도메인 의존성**: glass, vowels 도메인에서는 C4.5가 이미 불균형 값을 산출하지 않아 OSS 적용 불필요 → 오히려 성능 하락 가능
3. **적용 조건**: 논문은 $a^+$와 $a^-$ 모두를 확인하고, 한쪽이 비정상적으로 낮을 때만 OSS를 적용할 것을 권장함
4. **연속형 속성 한정**: 실험에 사용된 모든 속성이 연속형(continuous)
5. **Tomek Links의 계산 비용**: 대규모 데이터셋에서 거리 계산이 $O(n^2)$

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화 성능과 불균형 문제의 관계

불균형 데이터에서 분류기의 **일반화 성능 저하 원인**은 크게 세 가지입니다:

$$\text{일반화 오류} = \underbrace{\text{편향}^2}_{\text{다수 클래스 편향}} + \underbrace{\text{분산}}_{\text{과적합}} + \text{불가피한 오류}$$

OSS는 이 두 항을 동시에 줄이고자 합니다:

**① 편향 감소**: 경계선/잡음 음성 샘플 제거 → 분류기가 결정 경계를 더 균형 있게 학습

**② 분산 감소**: 중복 음성 샘플 제거 → 훈련 집합 크기 감소 → 과적합 완화

### 3.2 베이즈 이론적 관점에서의 일반화

순수 베이즈 분류기의 조건:

$$P(+) \cdot p_+(\mathbf{x}) > P(-) \cdot p_-(\mathbf{x})$$

불균형 데이터에서 $P(-) \gg P(+)$이므로 분류기는 음성 클래스로 편향됩니다. OSS를 통해 훈련 집합의 클래스 비율을 조정하면:

$$P'(-) < P(-), \quad P'(+) = P(+)$$

이로 인해 분류 경계가 이동하여 양성 클래스에 대한 일반화 성능이 향상됩니다.

### 3.3 VC 이론적 관점

Floyd & Warmuth (1995)의 연구를 인용하면서, Consistent Subset 크기를 줄이면 모델의 VC 차원에 대한 상한이 유지되면서도 복잡도가 줄어드는 효과가 있습니다:

$$R(\text{emp}) \leq R(\text{true}) + O\left(\sqrt{\frac{d \cdot \log(n/d)}{n}}\right)$$

여기서 $n$은 훈련 샘플 수, $d$는 VC 차원입니다. $n$을 줄이면서 $d$를 유지한다면 일반화 오류의 상한이 타이트해지지 않을 수 있으나, **잡음 샘플 제거**의 효과가 더 큰 실질적 이득을 줍니다.

### 3.4 일반화 성능 향상의 실증적 근거

논문의 실험에서 집합 $T$(OSS 적용)가 집합 $S$(전체)보다 $g$-mean이 높은 것은, 단순히 훈련 집합에서의 성능이 아닌 **교차 검증(k-fold cross-validation)**을 통한 테스트 성능이므로 일반화 성능 향상을 직접적으로 입증합니다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4.1 연구에 미치는 영향

이 논문은 이후 수많은 연구의 기폭제가 되었습니다:

**① SMOTE (Chawla et al., 2002)의 등장**
- OSS가 다수 클래스 언더샘플링에 집중한 반면, SMOTE는 소수 클래스 오버샘플링(합성 데이터 생성)을 제안
- 두 접근법을 결합하는 **SMOTE + Tomek**, **SMOTE + ENN** 등의 하이브리드 기법 탄생

**② 평가 기준의 정립**
- $g$-mean, F1-score, AUC-ROC가 불균형 학습의 표준 평가 지표로 정착하는 데 기여

**③ Undersampling 연구의 확장**
- ENN(Edited Nearest Neighbor), NearMiss, Cluster Centroids 등 다양한 언더샘플링 기법 파생

**④ 알고리즘 레벨 접근법 촉진**
- 비용 민감 학습(cost-sensitive learning), 앙상블 기법(BalancedBagging, EasyEnsemble)의 이론적 기반

### 4.2 향후 연구 시 고려할 점

#### (1) 방법론적 고려

- **하이퍼파라미터 민감성**: Consistent Subset 생성 시 초기 음성 예제 1개 선택이 랜덤이므로 결과 분산 발생 가능
- **다중 클래스 확장**: 논문은 이분류만 다루었으나, 다중 클래스 불균형 문제(One-vs-Rest, One-vs-One 전략 등)로 확장 필요
- **범주형 속성 처리**: 현재는 연속형 속성 기반 거리 계산에 의존

#### (2) 평가 기준 관련

$$\text{Matthews Correlation Coefficient (MCC)} = \frac{ad - bc}{\sqrt{(a+b)(a+c)(b+d)(c+d)}}$$

MCC와 같은 더 robust한 지표도 함께 사용 권장

#### (3) 대규모 데이터 환경

- Tomek Links의 거리 계산 복잡도가 $O(n^2)$이므로 대용량 데이터에서의 효율적 근사 알고리즘 필요
- 분산 컴퓨팅 환경(예: Spark)에서의 구현 연구 필요

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 비교 분석 표

| 논문/방법 | 연도 | 핵심 아이디어 | OSS 대비 특징 |
|-----------|------|--------------|---------------|
| **OSS** (Kubat & Matwin) | 1997 | Tomek Link + Consistent Subset | 기준점 |
| **SMOTE-variants review** (Fernández et al.) | 2018/계속 | 다양한 SMOTE 변형 종합 | 오버샘플링 중심 |
| **MixUp for Imbalance** | 2020~ | 선형 보간으로 합성 데이터 생성 | 피처 공간에서 오버샘플링 |
| **MESA (Meta-Sampler)** | 2021 | 메타러닝으로 샘플링 전략 자동 선택 | 도메인 적응 가능 |
| **BalancedMSE** | 2022 | 회귀 문제에서의 불균형 처리 | 회귀로 확장 |
| **LGBM/XGBoost의 scale_pos_weight** | 상시 | 알고리즘 내부에서 비용 조정 | 샘플링 불필요 |
| **Imbalanced Deep Learning (LDAM, CB Loss)** | 2019~2021 | 딥러닝에서 클래스 균형 손실 함수 | 딥러닝 특화 |

### 5.2 OSS와 최신 연구의 차별점

#### Deep Learning 시대의 변화

최신 딥러닝 기반 접근법들은 OSS의 데이터 수준 처리 대신 **손실 함수 수준**에서 불균형을 처리합니다:

**Class-Balanced Loss (Cui et al., 2019)**:
$$\mathcal{L}_{CB} = \frac{1-\beta}{1-\beta^{n_y}} \cdot \mathcal{L}(\hat{y}, y)$$

여기서 $n_y$는 클래스 $y$의 샘플 수, $\beta \in [0,1)$는 하이퍼파라미터입니다.

**LDAM (Label-Distribution-Aware Margin, Cao et al., 2019)**:

$$\mathcal{L}_{LDAM} = \max(0, 1 - (z_{y_i} - \Delta_{y_i}) + \max_{j \neq y_i} z_j)$$

여기서 $\Delta_j \propto n_j^{-1/4}$로 소수 클래스에 더 큰 마진을 부여합니다.

#### OSS의 현대적 재해석

**MESA (Meta-Sampler, Liu et al., 2021)**와 같은 접근법은 OSS의 아이디어를 메타러닝으로 확장하여, **어떤 샘플을 제거/유지할지를 학습**하는 방향으로 발전시켰습니다.

### 5.3 종합적 위치

```
데이터 수준          알고리즘 수준         앙상블 수준
─────────────────────────────────────────────────────
OSS (1997)          Cost-sensitive        BalancedBagging
SMOTE (2002)        Learning              EasyEnsemble
BorderlineSMOTE     LDAM (2019)           Self-paced
ADASYN              CB Loss (2019)        Ensemble
MESA (2021)    →    Meta-learning    →    Ensemble+Meta
```

OSS는 여전히 **해석 가능성(interpretability)**과 **계산 효율성** 측면에서 장점을 가지며, 딥러닝이 적합하지 않은 소규모 정형 데이터(tabular data) 도메인에서 유효합니다.

---

## 참고 자료

**논문 원문:**
- Kubat, M., & Matwin, S. (1997). **Addressing the Curse of Imbalanced Training Sets: One-Sided Selection**. *Proceedings of the 14th International Conference on Machine Learning (ICML'97)*, pp. 179–186.

**논문 내 인용 문헌 (주요):**
- Tomek, I. (1976). Two Modifications of CNN. *IEEE Transactions on Systems, Man and Communications*, SMC-6, 769–772.
- Hart, P.E. (1968). The Condensed Nearest Neighbor Rule. *IEEE Transactions on Information Theory*, IT-14, 515–516.
- Floyd, S., & Warmuth, M. (1995). Sample Compression, Learnability, and the Vapnik-Chervonenkis Dimension. *Machine Learning*, 21, 269–304.
- Fawcett, T., & Provost, F. (1996). Combining Data Mining and Machine Learning for Effective User Profile. *KDD-96 Proceedings*.
- Lewis, D., & Catlett, J. (1994). Heterogeneous Uncertainty Sampling for Supervised Learning. *ICML'94*.

**비교 분석 관련 최신 문헌:**
- Chawla, N.V., et al. (2002). **SMOTE: Synthetic Minority Over-sampling Technique**. *Journal of Artificial Intelligence Research*, 16, 321–357.
- Cui, Y., et al. (2019). **Class-Balanced Loss Based on Effective Number of Samples**. *CVPR 2019*.
- Cao, K., et al. (2019). **Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss**. *NeurIPS 2019*.
- Fernández, A., et al. (2018). **SMOTE for Learning from Imbalanced Data: Progress and Challenges**. *Journal of Artificial Intelligence Research*, 61, 863–905.
- He, H., & Garcia, E.A. (2009). **Learning from Imbalanced Data**. *IEEE Transactions on Knowledge and Data Engineering*, 21(9), 1263–1284.

> **주의**: 2020년 이후 특정 논문들(MESA 등)의 세부 수식 및 결과는 제가 직접 접근하지 못한 논문들로, 해당 내용은 공개된 정보에 기반하였으며 일부 내용은 확인이 필요할 수 있습니다. 정확한 수식과 결과는 원문을 직접 확인하시기 바랍니다.
