# Exploratory Undersampling for Class-Imbalance Learning

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

기존 언더샘플링(Undersampling)은 다수 클래스(Majority Class)의 유용한 정보를 버린다는 **핵심 결함**이 있다. 본 논문은 이를 극복하기 위해 **"버려지는 다수 클래스 샘플을 탐색적으로 활용(Exploratory Undersampling)"** 하는 두 가지 앙상블 알고리즘을 제안한다.

### 주요 기여

| 기여 항목 | 내용 |
|---|---|
| **EasyEnsemble** | 다수 클래스에서 독립적으로 여러 서브셋을 샘플링하여 앙상블 구성 (비지도 탐색) |
| **BalanceCascade** | 이전 학습기가 올바르게 분류한 다수 클래스 샘플을 제거하며 순차적으로 학습 (지도 탐색) |
| **효율성** | 언더샘플링과 동일한 훈련 시간 유지 |
| **성능 우수성** | AUC, F-measure, G-mean에서 기존 방법 대비 높은 성능 달성 |

---

## 2. 상세 분석

### 2-1. 해결하고자 하는 문제

**클래스 불균형(Class Imbalance)** 문제는 다수 클래스가 소수 클래스보다 압도적으로 많은 상황이다. 불균형 비율은 최대 $10^6$에 달할 수 있으며, 이 경우 단순 오류율을 최소화하는 학습기는 모든 샘플을 다수 클래스로 분류하는 경향이 생긴다.

예를 들어 불균형 비율이 99:1인 경우, 모든 샘플을 다수 클래스로 예측하면 오류율이 1%에 불과하지만, 소수 클래스(예: 사기 탐지, 희귀 질환)의 탐지율은 0%가 된다.

**기존 언더샘플링의 한계:**

$$\text{기존 방법: } N' \subset N, \quad |N'| = |\mathcal{P}|$$

$N$의 나머지 $(N \setminus N')$ 정보는 완전히 폐기되어 유용한 정보 손실이 발생한다.

---

### 2-2. 제안하는 방법 (수식 포함)

#### ✅ EasyEnsemble

**기본 아이디어:** 다수 클래스 $N$에서 $T$개의 서브셋 $N_1, N_2, \ldots, N_T$를 독립적으로 샘플링하고, 각각에 대해 AdaBoost 분류기 $H_i$를 학습한 후, 모든 약분류기를 결합한다.

각 서브앙상블 $H_i$의 출력:

$$H_i(x) = \text{sgn}\left(\sum_{j=1}^{s_i} \alpha_{i,j} h_{i,j}(x) - \theta_i\right)$$

최종 앙상블 출력:

$$H(x) = \text{sgn}\left(\sum_{i=1}^{T} \sum_{j=1}^{s_i} \alpha_{i,j} h_{i,j}(x) - \sum_{i=1}^{T} \theta_i\right)$$

여기서:
- $h_{i,j}$: $i$번째 서브셋에서 학습된 $j$번째 약분류기(weak classifier)
- $\alpha_{i,j}$: 해당 약분류기의 가중치
- $\theta_i$: $i$번째 앙상블의 결정 임계값
- $s_i$: $i$번째 AdaBoost 앙상블의 반복 횟수

**특징:** $|\mathcal{N}_i| = |\mathcal{P}|$ 로 설정하여 각 서브문제가 균형 잡힌 상태로 유지된다.

---

#### ✅ BalanceCascade

**기본 아이디어:** 학습된 분류기 $H_i$가 올바르게 분류한 다수 클래스 샘플을 $N$에서 제거하고, 이후 학습기는 더 어려운(정보가 풍부한) 샘플에 집중한다.

**False Positive Rate(FPR) 목표값 설정:**

$$f = \sqrt[T]{\frac{|\mathcal{P}|}{|\mathcal{N}|}}$$

이 값은 각 단계에서 $H_i$가 달성해야 할 FPR로, $T$회 반복 후 $|\mathcal{N}| \cdot f^{T-1} = |\mathcal{P}|$가 되도록 설계된다.

각 서브앙상블 출력 (동일):

$$H_i(x) = \text{sgn}\left(\sum_{j=1}^{s_i} \alpha_{i,j} h_{i,j}(x) - \theta_i\right)$$

임계값 $\theta_i$를 조정하여 $H_i$의 FPR이 $f$가 되도록 한다. 이후 $H_i$가 올바르게 분류한 다수 클래스 샘플을 $N$에서 제거한다.

최종 앙상블 출력:

$$H(x) = \text{sgn}\left(\sum_{i=1}^{T} \sum_{j=1}^{s_i} \alpha_{i,j} h_{i,j}(x) - \sum_{i=1}^{T} \theta_i\right)$$

**두 알고리즘의 핵심 차이:**

| 구분 | EasyEnsemble | BalanceCascade |
|---|---|---|
| 샘플링 방식 | 독립적 랜덤 샘플링 (비지도) | 이전 학습기 기반 제거 (지도) |
| 다수 클래스 탐색 | 중복 허용 무작위 탐색 | 잔여 어려운 샘플 집중 탐색 |
| 과적합 위험 | 낮음 | 상대적으로 높음 |
| 고불균형 적합성 | $T$ 설정이 어려움 | 효율적 (FPR 기반 $T$ 설정 가능) |

---

### 2-3. 평가 지표

$$\text{F-measure} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

$$\text{G-mean} = \sqrt{\text{Acc}_+ \times \text{Acc}_-}$$

$$\text{Acc}_+ = \frac{TP}{TP + FN}, \quad \text{Acc}_- = \frac{TN}{TN + FP}$$

AUC는 ROC 곡선 아래 면적으로, $\theta$를 $-\infty$에서 $+\infty$로 변화시키며 (FPR, TPR) 쌍을 계산하여 구한다.

---

### 2-4. 성능 향상

16개의 UCI 데이터셋을 대상으로 10-fold stratified cross validation (5회 반복)을 수행하였다. 데이터셋을 AdaBoost의 AUC 기준으로 두 그룹으로 분류하였다:

- **"Easy" tasks**: AdaBoost AUC $\geq 0.95$ (6개)
- **"Hard" tasks**: AdaBoost AUC $< 0.95$ (10개, 실제 불균형 문제)

**"Hard" tasks에서의 주요 결과:**

| 방법 | 평균 AUC | 비고 |
|---|---|---|
| AdaBoost | $\approx 0.760$ | 기준선 |
| Under | $\approx 0.769$ | 단순 언더샘플링 |
| SMOTE | $\approx 0.772$ | 합성 오버샘플링 |
| Chan | $\approx 0.781$ | 분할 앙상블 |
| **EasyEnsemble** | $\approx 0.787$ | **제안 방법** |
| **BalanceCascade** | $\approx 0.778$ | **제안 방법** |

(수치는 Table V 평균값 기준 근사치)

---

### 2-5. 한계

1. **BalanceCascade의 과적합 위험:** 순차적 제거 방식에서 초기에 제거된 샘플이 후반 단계에서 유용할 수 있음
2. **해석 불가능성(Black-Box):** 앙상블의 특성상 모델이 불투명함
3. **EasyEnsemble의 고불균형 비효율성:** 불균형 비율이 매우 클 경우, 모든 다수 클래스 정보를 커버하기 위한 $T$ 값 결정이 어려움
4. **약분류기 가중치 고정:** 현재 $\alpha_{i,j}$를 약학습기가 반환한 값 그대로 사용하며, 최적화되지 않음
5. **이진 분류 한정:** 다중 클래스 불균형 문제에는 직접 적용 불가

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 일반화 향상의 핵심 메커니즘

**EasyEnsemble의 일반화 향상 원리:**

EasyEnsemble은 **부스팅(Boosting)** 과 **배깅(Bagging)** 을 결합한 구조로 일반화를 달성한다:

$$\text{EasyEnsemble} = \underbrace{\text{Boosting}}_{\text{편향(Bias) 감소}} + \underbrace{\text{Bagging-like sampling}}_{\text{분산(Variance) 감소}}$$

- **부스팅** : 각 서브셋 내에서 AdaBoost로 편향을 감소
- **배깅 유사 효과** : 서로 다른 서브셋 $N_1, \ldots, N_T$에서 학습하여 분산을 감소
- **균형 잡힌 클래스 분포** : 각 $|N_i| = |\mathcal{P}|$ 설정으로 소수 클래스 중심 학습 유도

논문에서는 이를 MultiBoosting [35], Stochastic Gradient Boosting [19], Cocktail Ensemble [42]과 같이 서로 다른 앙상블 전략을 결합하는 방식과 유사하다고 설명한다.

**Diverse Feature Extraction 관점:**

약분류기 $h_{i,j}$를 이진값을 갖는 특징(feature)으로 볼 때:

$$\{h_{i,j} \mid i=1,\ldots,T; \; j=1,\ldots,s_i\}$$

서로 다른 $N_i$에서 추출된 특징들은 $N$의 **다양한 측면**을 반영하므로, 최종 선형 분류기의 일반화 성능이 향상된다.

### 3-2. 과적합 억제 메커니즘

**소수 클래스 과적합 방지:**
- EasyEnsemble: 각 $H_i$에 모든 소수 클래스 샘플 $\mathcal{P}$가 포함되나, 다수 클래스 다양성 덕분에 과적합이 억제됨
- 스태킹(Stacking) 대비: 논문의 Table XIII에서 스태킹이 "hard" tasks에서 일관되게 열등함을 보임 → 소수 클래스 반복 사용 시 스태킹은 과적합 위험

**Majority Class 정보 완전 활용:**
단일 언더샘플링 대비, $T$개의 서브셋이 $N$의 다양한 영역을 커버함으로써:

$$\bigcup_{i=1}^{T} N_i \approx N \quad (\text{기댓값 측면})$$

이로 인해 다수 클래스의 결정 경계 추정이 더 정확해진다.

### 3-3. 일반화 향상의 한계

- **BalanceCascade** 의 경우, "easy" tasks에서는 EasyEnsemble보다 우수하나, "hard" tasks에서 열등함. 이는 지도적 제거 방식이 특정 샘플을 과도하게 제거하여 후속 학습기의 훈련 데이터가 편향될 수 있음을 시사
- $\alpha_{i,j}$ 최적화 미적용: 현재 약학습기가 반환한 가중치를 그대로 사용하며, 이를 최적화하면 일반화 성능이 추가로 향상 가능

---

## 4. 앞으로의 연구에 미치는 영향과 고려 사항

### 4-1. 미치는 영향

1. **앙상블 기반 불균형 학습의 표준화:** EasyEnsemble은 이후 불균형 학습 연구의 강력한 기준선(baseline)으로 자리잡음

2. **정보 손실 최소화 패러다임 제시:** 언더샘플링의 정보 폐기 문제를 앙상블로 해결하는 아이디어는 이후 다양한 변형 연구로 이어짐

3. **평가 지표의 중요성 강조:** 불균형 학습에서 정확도 대신 AUC, F-measure, G-mean을 사용하는 관행을 강화함

4. **클래스 불균형 탐지 필요성 제기:** 불균형이 실제 해로운지 판단하는 사전 진단 방법론 연구 필요성 제시

### 4-2. 향후 연구 시 고려 사항

| 고려 항목 | 내용 |
|---|---|
| **다중 클래스 확장** | 이진 분류에 한정된 알고리즘을 다중 클래스로 일반화 |
| **딥러닝과의 결합** | BERT, ResNet 등 딥러닝 모델에서의 불균형 처리 방안 |
| **동적 임계값 조정** | BalanceCascade의 FPR 목표값 $f$를 데이터 적응적으로 설정 |
| **불균형 탐지 방법론** | 불균형이 해로운지 사전 판단하는 자동화 방법 개발 |
| **스트리밍 데이터** | 온라인 학습 환경에서의 동적 불균형 처리 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

> **주의:** 아래 연구들은 본 논문의 PDF 원문에 포함되지 않으며, 제가 학습한 지식 범위 내에서 제공합니다. 개별 논문의 세부 수치는 직접 확인을 권장합니다.

### 5-1. 주요 후속 연구 방향

#### ① 딥러닝 기반 불균형 학습

**"Class-Balanced Loss Based on Effective Number of Samples"** (Cui et al., CVPR 2019)는 샘플의 유효 개수(Effective Number)를 기반으로 클래스별 손실 가중치를 조정하는 방법을 제안하였다:

$$\text{Effective Number} = \frac{1 - \beta^{n_y}}{1 - \beta}$$

여기서 $\beta \in [0,1)$, $n_y$는 클래스 $y$의 샘플 수이다. 이는 EasyEnsemble의 앙상블 기반 접근과 달리 손실 함수 수준에서 불균형을 해소한다.

#### ② 자기지도학습(Self-supervised) 기반 접근

**"Rethinking the Value of Labels for Improving Class-Imbalanced Learning"** (Yang & Xu, NeurIPS 2020)은 자기지도 사전학습이 불균형 데이터에서 더 강건한 표현을 학습함을 보였다. EasyEnsemble의 샘플 수준 탐색 대비, 표현 학습 수준에서의 불균형 해소를 시도한다.

#### ③ 메타학습 기반 방법

**"Learning to Self-Train for Semi-Supervised Few-Shot Classification"** 계열 연구에서는 소수 클래스의 극단적 부족 시 메타학습(Meta-Learning)을 통한 일반화가 탐구되었다.

#### ④ GAN 기반 오버샘플링

**"CTGAN"** (Xu et al., NeurIPS 2019) 및 후속 연구들은 조건부 GAN을 활용하여 소수 클래스 합성 샘플을 생성한다. SMOTE의 단순 보간 대비 더 실제적인 샘플을 생성하나, 훈련 복잡도가 높다.

### 5-2. EasyEnsemble과의 비교

| 방법 | 핵심 전략 | 장점 | 단점 |
|---|---|---|---|
| **EasyEnsemble** (2009) | 다수 클래스 다중 서브셋 앙상블 | 빠른 학습, 강건성 | 이진 분류 한정, 딥러닝 미결합 |
| **Class-Balanced Loss** (2019) | 손실 함수 재가중치 | 딥러닝 통합 용이 | 하이퍼파라미터 민감 |
| **Self-supervised + 불균형** (2020~) | 표현 학습 수준 해소 | 레이블 효율성 | 사전학습 비용 |
| **GAN 기반 오버샘플링** (2019~) | 합성 소수 클래스 생성 | 현실적 샘플 | 학습 불안정, 비용 |
| **Logit Adjustment** (Menon et al., ICLR 2021) | 사후 예측 보정 | 이론적 보장 | 추론 시 수정 필요 |

EasyEnsemble은 기본 앙상블 학습기에 적용 가능한 **모델 비종속적(model-agnostic)** 방법론으로, 딥러닝 기반 최신 방법 대비 단순하지만 여전히 경쟁력 있는 기준선으로 활용된다.

---

## 참고 자료

**본 답변의 주요 출처:**

1. **Liu, X.-Y., Wu, J., & Zhou, Z.-H. (2009).** "Exploratory Undersampling for Class-Imbalance Learning." *IEEE Transactions on Systems, Man, and Cybernetics—Part B: Cybernetics*, Vol. 39, No. 2, pp. 539–550. DOI: 10.1109/TSMCB.2008.2007853 *(제공된 PDF 원문)*

**비교 분석에 참고한 문헌 (학습 지식 기반, 직접 검색 확인 권장):**

2. Cui, Y., Jia, M., Lin, T.-Y., Song, Y., & Belongie, S. (2019). "Class-Balanced Loss Based on Effective Number of Samples." *CVPR 2019*.

3. Yang, Y., & Xu, Z. (2020). "Rethinking the Value of Labels for Improving Class-Imbalanced Learning." *NeurIPS 2020*.

4. Menon, A. K., Jayasumana, S., Rawat, A. S., Jain, H., Veit, A., & Kumar, S. (2021). "Long-tail learning via logit adjustment." *ICLR 2021*.

5. Xu, L., Skoularidou, M., Cuesta-Infante, A., & Veeramachaneni, K. (2019). "Modeling Tabular data using Conditional GAN." *NeurIPS 2019*.

6. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). "SMOTE: Synthetic Minority Over-sampling Technique." *JAIR*, vol. 16. *(논문 내 참조)*
