# Density-based Weighting for Imbalanced Regression

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

이 논문은 **회귀(Regression) 문제에서의 불균형 데이터(Imbalanced Data)** 문제를 해결하기 위해, 목표값(target value)의 밀도(density)를 기반으로 각 샘플에 가중치를 부여하는 **비용 민감 학습(Cost-Sensitive Learning)** 접근법을 제안합니다. 분류(Classification) 문제에서는 다양한 불균형 처리 기법이 존재하지만, 연속적인 목표값을 갖는 회귀 문제에는 직접 적용이 어렵다는 문제를 지적하며, 이를 해결하는 새로운 알고리즘 수준의 방법론을 제시합니다.

### 주요 기여 (5가지)

| 기여 | 설명 |
|------|------|
| ① **DenseWeight 제안** | KDE 기반 샘플 가중치 계산 방법론 |
| ② **DenseLoss 제안** | DenseWeight를 활용한 신경망 비용 민감 학습 손실 함수 |
| ③ **합성 데이터 분석** | 다양한 분포에서 DenseLoss 효과 검증 |
| ④ **SMOGN과 비교** | 당시 최신 방법 대비 성능 우위 입증 |
| ⑤ **실제 문제 적용** | 강수량 다운스케일링(precipitation downscaling) 실증 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

많은 실세계 데이터셋에서 목표값의 분포는 불균형합니다. 예를 들어, 강수량 추정에서 극단적인 강우 사건은 드물지만 매우 중요합니다. 이 경우 일반적인 학습 방법은 **빈번한 값에 편향(biased)**되어, 희귀 값에 대한 예측 성능이 저하됩니다.

- **분류 문제**: SMOTE, ADASYN 등 다양한 해법 존재
- **회귀 문제**: SMOGN, SmoteR 등 **샘플링 기반** 방법만 존재 → 오버샘플링으로 인한 과적합, 언더샘플링으로 인한 정보 손실 등의 단점 존재
- **비용 민감 학습(Cost-Sensitive Learning)**: 분류에서는 효과적이나, 회귀에서는 거의 탐구되지 않음

---

### 2.2 제안하는 방법 (수식 포함)

#### Step 1: 커널 밀도 추정 (KDE)

훈련 데이터의 목표값 $Y = \{y_1, y_2, \ldots, y_N\}$에 대해 **Kernel Density Estimation(KDE)**으로 밀도 함수를 추정합니다:

$$p(y) = \frac{1}{Nh} \sum_{i=1}^{N} K\!\left(\frac{y - y_i}{h}\right) \tag{1}$$

- $K$: 커널 함수 (가우시안 커널 사용)
- $h$: 대역폭 (Silverman's rule로 자동 선택)

#### Step 2: 정규화된 밀도 함수

밀도 값을 $[0, 1]$ 범위로 정규화합니다:

$$p'(y) = \frac{p(y) - \min(p(Y))}{\max(p(Y)) - \min(p(Y))} \tag{2}$$

- $p'(y) = 1$: 가장 밀집된(common) 영역
- $p'(y) = 0$: 가장 희박한(rare) 영역

#### Step 3: 기본 가중치 함수

$$f'_w(\alpha, y) = 1 - \alpha p'(y) \tag{3}$$

- $\alpha = 0$: 균일 가중치 (DenseWeight 비활성화)
- $\alpha = 1$: 가장 일반적인 데이터 포인트의 가중치가 0에 수렴

#### Step 4: 클리핑 적용 (음수 및 0 가중치 방지)

$$f''_w(\alpha, y) = \max(1 - \alpha p'(y),\ \epsilon) \tag{4}$$

- $\epsilon$: 아주 작은 양수 상수 (논문에서 $10^{-6}$ 사용)

#### Step 5: 평균 정규화 (DenseWeight 최종 수식)

학습률에 영향을 주지 않도록 평균 가중치가 1이 되게 정규화:

$$f_w(\alpha, y) = \frac{f''_w(\alpha, y)}{\frac{1}{N}\sum_{i=1}^{N} f''_w(\alpha, y_i)} = \frac{\max(1 - \alpha p'(y),\ \epsilon)}{\frac{1}{N}\sum_{i=1}^{N}\max(1 - \alpha p'(y_i),\ \epsilon)} \tag{5}$$

**DenseWeight의 설계 원칙 (Properties):**

| 속성 | 내용 |
|------|------|
| **P.1** | 희귀 샘플 > 일반 샘플 가중치 |
| **P.2** | $\alpha=0$이면 균일 가중치, $\alpha$가 클수록 강조 |
| **P.3** | 가중치는 음수가 되지 않음 |
| **P.4** | 가중치는 0이 되지 않음 |
| **P.5** | 전체 가중치의 평균 = 1 |

#### Step 6: DenseLoss (최종 손실 함수)

$$L_{\text{DenseLoss}}(\alpha) = \frac{1}{N} \sum_{i=1}^{N} f_w(\alpha, y_i) \cdot M(\hat{y}_i, y_i) \tag{6}$$

- $M(\hat{y}_i, y_i)$: 선택한 메트릭 (예: MSE)
- $\hat{y}_i$: 모델 예측값
- 희귀 샘플의 손실이 더 크게 반영되어 해당 샘플에 대한 gradient가 더 커짐

---

### 2.3 모델 구조

논문에서는 두 가지 신경망 구조를 사용합니다:

#### (a) 합성 데이터 및 SMOGN 비교 실험
- **MLP (Multi-Layer Perceptron)**
  - 은닉층 3개, 각 층 뉴런 수 10개
  - 활성화 함수: ReLU
  - 출력층: 선형 활성화 (단일 뉴런)
  - 최적화: Adam ($lr = 10^{-4}$), Weight decay $= 10^{-9}$
  - 조기 종료(Early Stopping): validation loss 10 epoch 미개선 시 중단
  - 가중치 초기화: Kaiming Uniform

#### (b) 강수량 다운스케일링 (실제 문제)
- **DeepSD** (Vandal et al., 2017) 기반 CNN
  - 합성곱 층 3개: 필터 수 64, 32, 1; 커널 크기 9, 1, 5
  - 배치 크기 200
  - Adam 최적화 ($lr = 10^{-4}$ / $10^{-5}$)
  - DenseLoss를 기존 DeepSD 코드에 통합

---

### 2.4 성능 향상 및 한계

#### 성능 향상

**실험 1: 합성 데이터 (pareto, rpareto, normal, dnormal)**
- $\alpha = 1.0$ 기준, 가장 희귀한 bin(rank 1)에서 **RMSE 최대 7.02, MAE 최대 7.00 향상**
- 반면 가장 일반적인 bin(rank 5)에서 RMSE 최대 1.68 증가 (성능 트레이드오프 존재)
- 통계적 유의성(Wilcoxon signed-rank test, $p < 0.05$) 확인

**실험 2: 20개 데이터셋 SMOGN 비교**
- 가장 희귀한 bin(rank 1): DenseLoss **8개** 데이터셋 최우수 vs. SMOGN **3개**
- Bin rank 1~4 전반에서 DenseLoss가 절반 이상의 데이터셋에서 승리
- 다양한 MLP 아키텍처(2~4개 은닉층, 5~20개 뉴런)에서 일관된 결과

**실험 3: 강수량 다운스케일링 (PRISM 데이터셋)**
- **모든 bin rank에서** RMSE 최대 약 8% 개선 (희귀 샘플뿐 아니라 일반 샘플까지 개선)
- $\alpha \geq 0.8$에서 모든 bin rank에 대해 통계적으로 유의미한 개선
- 성능은 $\alpha \approx 2.0$에서 plateau에 도달

#### 한계

| 한계 | 설명 |
|------|------|
| **신경망에만 평가** | DenseWeight는 이론적으로 모든 샘플 가중치 지원 알고리즘에 적용 가능하나, 실험은 신경망으로만 검증 |
| **하이퍼파라미터 $\alpha$ 튜닝** | 최적 $\alpha$를 찾기 위해 도메인 지식 또는 추가 검증이 필요 |
| **SMOGN과 대규모 데이터 비교 불가** | SMOGN의 거리 계산 연산이 대규모 데이터셋에서 수년이 소요될 수 있어 실험 불가 |
| **KDE 품질 의존성** | 노이즈가 많은 아웃라이어 데이터에서 KDE 정확도가 저하될 수 있음 |
| **균일 데이터에 적용 불가** | $p(Y)$의 정규화 과정에서 분모가 0이 되므로 완전히 균일한 데이터에는 사용 불가 |
| **성능 트레이드오프** | 희귀 케이스 성능 향상 시, 일반 케이스 성능이 저하될 수 있음 (작은 모델에서 두드러짐) |

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 조기 종료(Early Stopping)를 통한 과적합 방지

논문은 validation loss가 10 epoch 동안 개선되지 않으면 훈련을 중단하는 **조기 종료** 기법을 사용합니다. 이는 일반화 성능을 향상시키는 핵심 메커니즘으로, 저자들은 훈련 성능과 테스트/검증 성능 간 차이가 없음을 spot check로 확인하였습니다.

### 3.2 데이터 증강 없는 알고리즘 수준 접근

SMOGN과 같은 샘플링 기반 방법은 **합성 데이터를 생성**하므로 다음과 같은 일반화 위협 요소가 있습니다:
- 오버샘플링 → 과적합(Overfitting) 위험
- 언더샘플링 → 정보 손실 → 일반화 저하

반면, **DenseLoss는 데이터셋을 직접 변경하지 않으며**, 단순히 손실 함수의 가중치만 조정하므로 이러한 위험으로부터 자유롭습니다.

### 3.3 모델 용량(Capacity)과 일반화의 관계

강수량 다운스케일링 실험에서 발견된 **흥미로운 현상**:

> *"DeepSD의 용량이 충분히 크기 때문에 희귀 및 일반 데이터 포인트 모두에 대해 좋은 함수를 동시에 학습할 수 있으며, 작은 모델은 이 능력이 부족할 수 있다."*

즉, **대용량 모델 + DenseLoss** 조합은 희귀 케이스와 일반 케이스 모두에서 일반화 성능을 향상시키는 시너지를 보입니다. 이는 DenseLoss가 단순히 희귀 케이스에 편향되는 것이 아니라, **모델이 더 나은 전반적 해(solution)로 수렴**하도록 유도할 가능성을 시사합니다.

### 3.4 $\alpha$를 통한 일반화 제어

$$f_w(\alpha=0, y) \equiv 1 \quad \text{(균일 가중치, 표준 학습)}$$

$$f_w(\alpha>0, y): \text{희귀 샘플에 더 큰 gradient} \rightarrow \text{희귀 케이스 일반화 향상}$$

$\alpha$ 값을 validation set에서 최적화함으로써, **특정 도메인에 맞는 일반화 성능**을 달성할 수 있습니다. 예를 들어, 극단적 강수량 예측에는 높은 $\alpha$, 전반적 예측에는 낮은 $\alpha$가 적합합니다.

### 3.5 여러 아키텍처에서의 일관된 일반화

실험에서 2~4개 은닉층, 5~20개 뉴런을 가진 다양한 MLP 아키텍처에서 일관된 성능 향상을 확인하였습니다. 이는 DenseLoss가 특정 아키텍처에 의존하지 않고 **범용적으로 일반화 성능을 향상**시킬 수 있음을 보여줍니다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4.1 연구에 미치는 영향

#### (a) 불균형 회귀 연구의 새로운 방향 제시
비용 민감 학습(Cost-Sensitive Learning)을 회귀 영역으로 확장하는 새로운 패러다임을 제시하였습니다. 이전에는 SMOGN 등 샘플링 기반 방법이 주류였으나, 이 논문은 **알고리즘 수준의 해결책**이 더 효과적일 수 있음을 입증하였습니다.

#### (b) KDE의 새로운 활용
KDE를 단순 밀도 추정이 아닌 **학습 과정의 가중치 생성 도구**로 활용하는 독창적인 아이디어를 제시하였습니다.

#### (c) 앙상블 접근 가능성 제시
저자들은 미래 연구 방향으로 **DenseLoss 앙상블**을 제안합니다. 서로 다른 $\alpha$로 훈련된 모델들은 각기 다른 목표값 범위에서 전문성을 갖게 되며, 메타 모델이 특정 샘플에 최적인 앙상블 멤버를 선택하도록 학습하면 전체 범위에서 최적 성능을 달성할 수 있습니다.

#### (d) 분류 기법의 회귀 전이 가능성
논문은 다음과 같은 분류 기법들의 회귀 전이를 제안합니다:
- **유효 샘플 수(Effective Number of Samples) 기반 가중치** (Cui et al., 2019)
- **샘플 난이도 기반 가중치** (Dong et al., 2017, Class Rectification Hard Mining)

### 4.2 앞으로 연구 시 고려할 점

#### (a) $\alpha$ 자동 최적화 방법 개발

현재는 수동 또는 grid search로 $\alpha$를 탐색하지만, 자동화된 최적화 기법이 필요합니다:

$$\alpha^* = \arg\min_{\alpha} \mathcal{L}_{\text{rare}}(\alpha)$$

여기서 $\mathcal{L}_{\text{rare}}$는 희귀 데이터에 특화된 평가 지표입니다. 베이지안 최적화(Bayesian Optimization) 등을 활용하면 효율적입니다.

#### (b) 다변량 목표값(Multi-output Regression)으로의 확장

현재는 단일 목표값 변수에만 적용되나, 여러 목표값을 동시에 예측하는 다변량 회귀에서의 불균형 문제는 아직 연구가 부족합니다. KDE를 다변량으로 확장하거나, 각 목표 변수별 DenseWeight를 결합하는 방법이 고려될 수 있습니다.

#### (c) KDE 대안 탐색

KDE는 간단하지만 고차원 데이터에서는 **차원의 저주(Curse of Dimensionality)** 문제가 발생할 수 있습니다. 목표값이 고차원인 경우 더 효율적인 밀도 추정 방법이 필요합니다:
- **Normalizing Flows**
- **Variational Autoencoders(VAE)** 기반 밀도 추정
- **Gaussian Mixture Models(GMM)**

#### (d) 트리 기반 모델 및 다른 알고리즘으로의 확장 검증

논문은 이론적으로 모든 샘플 가중치 지원 알고리즘에 적용 가능하다고 주장하지만, 실험적 검증은 신경망에만 이루어졌습니다. XGBoost, Random Forest, LightGBM 등 트리 기반 모델에서의 효과 검증이 필요합니다.

#### (e) 데이터 분할(Data Splitting) 전략 개선

소규모 데이터셋에서 훈련/검증/테스트 분포의 유사성을 보장하기 위한 자동화된 분할 전략이 필요합니다. 예를 들어, **Stratified Sampling for Regression**이나 분포 유사도 점수를 최대화하는 분할 최적화가 고려될 수 있습니다.

#### (f) 극단값 예측에서의 평가 지표 표준화

SERA(Squared Error-Relevance Area, Ribeiro & Moniz, 2020)와 같이 불균형 회귀에 특화된 평가 지표의 표준화가 필요합니다. 단순 RMSE/MAE는 일반적인 샘플에 편향될 수 있어, 희귀 케이스를 적절히 평가하는 메트릭의 개발 및 채택이 중요합니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의**: 아래 비교 분석은 제가 학습한 지식 범위 내의 정보를 포함하며, 2024년 이후 최신 연구에 대해서는 정보가 불완전할 수 있습니다. 정확한 최신 동향 파악을 위해서는 추가 검색이 필요합니다.

| 논문/방법 | 연도 | 핵심 아이디어 | DenseLoss와의 차이점 |
|-----------|------|--------------|---------------------|
| **SMOGN** (Branco et al.) | 2017 | SmoteR + Gaussian noise 오버샘플링 | 데이터 수준 방법, 합성 샘플 생성 |
| **SERA** (Ribeiro & Moniz) | 2020 | 불균형 회귀용 평가 메트릭 제안 | 평가 지표이며, 학습 방법이 아님 |
| **REBAGG** (Branco et al.) | 2019 | 앙상블 기반 불균형 회귀 | 앙상블 방법 vs. 단일 손실 함수 수정 |
| **Imbalanced Regression via Distribution Matching** | ~2021~2022 | 분포 매칭 기반 접근 | 방법론적 접근 방식 상이 |
| **LDS/FDS** (Yang et al., NeurIPS 2021) | 2021 | Label Distribution Smoothing, Feature Distribution Smoothing | 레이블 분포 스무딩으로 불균형 회귀 처리, DenseLoss와 유사한 방향이나 특징 공간에서도 처리 |

### LDS/FDS vs. DenseLoss 상세 비교

Yang et al. (2021, "Delving into Deep Imbalanced Regression", ICML 2021)은 DenseLoss와 유사한 문제를 다루며, 다음과 같은 차이점을 가집니다:

- **LDS (Label Distribution Smoothing)**: 레이블 밀도를 커널로 스무딩하여 유효 레이블 밀도를 추정 → DenseLoss와 개념적으로 유사
- **FDS (Feature Distribution Smoothing)**: 특징 공간에서 인접 레이블의 통계량을 전이 → DenseLoss에는 없는 특징 수준 처리
- **주요 차이**: DenseLoss는 단순 KDE + 가중치이지만, LDS/FDS는 분포 스무딩 + 특징 캘리브레이션을 추가하여 더 정교한 방법론을 제시

> ⚠️ 단, Yang et al. (2021)의 구체적 논문 내용은 제가 직접 원문을 확인한 것이 아니므로, 세부 내용의 정확성을 위해 원문 확인을 권장합니다.

---

## 참고자료

**주요 논문 (원문 첨부 기준):**
- Steininger, M., Kobs, K., Davidson, P., Krause, A., & Hotho, A. (2021). **Density-based weighting for imbalanced regression**. *Machine Learning*. https://doi.org/10.1007/s10994-021-06023-5

**논문 내 참고문헌 (원문에서 인용된 주요 자료):**
- Branco, P., Torgo, L., & Ribeiro, R. P. (2017). SMOGN: A pre-processing approach for imbalanced regression. In *LIDTA*.
- Chawla, N. V., et al. (2002). SMOTE: Synthetic minority oversampling technique. *JAIR*, 16, 321–357.
- Cui, Y., et al. (2019). Class-balanced loss based on effective number of samples. *CVPR 2018*, 9268–9277.
- He, H., et al. (2008). ADASYN: Adaptive synthetic sampling approach for imbalanced learning. *IJCNN 2008*.
- Ribeiro, R. P., & Moniz, N. (2020). Imbalanced regression and extreme value prediction. *Machine Learning*, 109(9), 1803–1835.
- Silverman, B. W. (1986). *Density estimation for statistics and data analysis*. CRC Press.
- Vandal, T., et al. (2017). DeepSD: Generating high resolution climate change projections through single image super-resolution. *KDD 2017*, 1663–1672.
- Dong, Q., Gong, S., & Zhu, X. (2017). Class rectification hard mining for imbalanced deep learning. *ICCV 2017*, 1851–1860.
- Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. *arXiv:1412.6980*.
- Prechelt, L. (1998). Early stopping—but when? In *Neural Networks: Tricks of the Trade*.

**코드 저장소:**
- https://github.com/SteiMi/density-based-weighting-for-imbalanced-regression
