# Selective Pre-processing of Imbalanced Data for Improving Classification Performance

## 논문 정보
- **저자**: Jerzy Stefanowski, Szymon Wilk
- **출판**: DaWaK 2008, LNCS 5182, pp. 283–292, Springer-Verlag Berlin Heidelberg 2008

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장
이 논문의 핵심 주장은 다음과 같습니다:

> **"소수 클래스의 민감도(Sensitivity) 향상에만 집중하는 기존 접근법은 다수 클래스의 특이도(Specificity)를 과도하게 희생시키며, 이를 해결하기 위해 지역적(local) 이웃 분석 기반의 선택적 전처리 방법이 필요하다."**

### 주요 기여

| 기여 항목 | 설명 |
|-----------|------|
| **새로운 선택적 전처리 알고리즘** | 소수 클래스의 지역적 오버샘플링 + 다수 클래스의 어려운 샘플 제거를 결합 |
| **3가지 증폭(Amplification) 기법 제안** | Weak Amplification, Weak Amplification + Relabeling, Strong Amplification |
| **민감도-특이도 균형 유지** | NCR보다 특이도를 더 잘 보존하면서 기저선(baseline) 대비 민감도 향상 |
| **데이터 분포 변화 최소화** | SMOTE 대비 적은 데이터 분포 변화로 유사한 성능 달성 |
| **인위적 샘플 비생성** | SMOTE와 달리 기존 샘플만 복제 또는 재레이블링하여 도메인 신뢰도 유지 |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

**불균형 데이터 분류 문제(Imbalanced Data Classification Problem)**

표준 학습 알고리즘은 다수 클래스(majority class)에 편향되어 소수 클래스(minority class) 샘플을 오분류하는 경향이 있습니다. 이를 측정하기 위한 주요 지표:

$$\text{Sensitivity} = \frac{\text{소수 클래스에서 올바르게 분류된 샘플 수}}{|\text{소수 클래스}|}$$

$$\text{Specificity} = \frac{\text{다수 클래스에서 올바르게 분류된 샘플 수}}{|\text{다수 클래스}|}$$

기존 방법들의 문제점:
- **NCR(Neighborhood Cleaning Rule)**: 특이도와 전체 정확도를 과도하게 희생
- **SMOTE**: 전역(global) 파라미터로 인위적 샘플 생성 → 클래스 경계 중복(overlapping) 위험, 전역 파라미터 튜닝 필요
- **일반 무작위 오버/언더샘플링**: 지역적 난이도를 반영하지 못함

### 2.2 제안 방법 (알고리즘 및 수식)

#### Phase 1: 예제 유형 분류

HVDM(Heterogeneous Value Difference Metric)을 기반으로 한 NNR(Nearest Neighbor Rule)을 사용합니다.

$$\text{예제 } x \text{의 유형} = \begin{cases} \text{safe} & \text{if } classify\_knn(x, k=3) \text{ is correct} \\ \text{noisy} & \text{otherwise} \end{cases}$$

#### Phase 2: 선택적 전처리 (3가지 기법)

**[기법 1] Weak Amplification**

노이지 소수 클래스 샘플 $x \in C$에 대해 복제본 수를 결정:

$$\text{copies}(x) = |knn(x, 3, O, \text{safe})|$$

즉, $x$의 3-최근접 이웃 중 다수 클래스($O$)에 속하는 safe 샘플의 수만큼 복제합니다.

**[기법 2] Weak Amplification + Relabeling**

1단계: Weak Amplification과 동일하게 노이지 소수 샘플 증폭

$$\text{copies}(x) = |knn(x, 3, O, \text{safe})|, \quad \forall x \in C_{\text{noisy}}$$

2단계: 노이지 소수 샘플 $x$의 3-최근접 이웃 중 노이지 다수 샘플 $y$를 재레이블링:

$$y \in knn(x, 3, O, \text{noisy}) \Rightarrow \text{label}(y): O \to C$$

**[기법 3] Strong Amplification**

Safe 소수 샘플 처리:

$$\text{copies}(x) = |knn(x, 3, O, \text{safe})|, \quad \forall x \in C_{\text{safe}}$$

Noisy 소수 샘플 처리 (확장된 이웃 활용):

$$\text{copies}(x) = \begin{cases} |knn(x, 3, O, \text{safe})| & \text{if } classify\_knn(x, 5) \text{ is correct} \\ |knn(x, 5, O, \text{safe})| & \text{otherwise} \end{cases}$$

마지막으로, 다수 클래스의 noisy 샘플을 모두 제거 (ENNR 원리 적용):

$$D = \{y \in O \mid y \text{ is noisy}\} \Rightarrow \text{remove all } y \in D$$

#### 전체 알고리즘 의사코드 요약

```
Phase 1: 모든 예제 safe/noisy 분류 (k=3 NNR)
Phase 2:
  - D ← 다수 클래스의 noisy 예제 집합
  - 선택된 기법(weak/relabel/strong)으로 소수 클래스 증폭
  - D에 속한 예제 제거
```

### 2.3 모델 구조

```
[입력: 불균형 훈련 데이터]
        ↓
[Phase 1: HVDM 기반 NNR(k=3)으로 safe/noisy 레이블링]
        ↓
[Phase 2: 선택적 전처리]
   ├─ Weak Amplification
   ├─ Weak Amplification + Relabeling  
   └─ Strong Amplification
        ↓
[전처리된 균형 데이터]
        ↓
[분류기 학습: C4.5 (결정 트리) 또는 MODLEM (규칙 기반)]
        ↓
[평가: Sensitivity, Specificity, Overall Accuracy]
```

### 2.4 성능 향상 및 한계

#### 성능 향상 결과

9개 UCI 데이터셋(acl, breast cancer, bupa, cleveland, ecoli, haberman, hepatitis, new-thyroid, pima)에서 10-fold cross-validation × 5회 반복 실험:

**민감도 비교 (MODLEM 기준, 주요 데이터셋)**

| 데이터셋 | Base | SMOTE | NCR | Weak | Relabel | Strong |
|----------|------|-------|-----|------|---------|--------|
| Bupa | 0.520 | 0.737 | **0.873** | 0.799 | 0.838 | 0.805 |
| Haberman | 0.240 | 0.301 | **0.626** | 0.404 | 0.468 | 0.483 |
| Pima | 0.485 | 0.640 | **0.793** | 0.685 | 0.738 | 0.738 |

**특이도 비교 (MODLEM 기준, 주요 데이터셋)**

| 데이터셋 | Base | SMOTE | NCR | Weak | Relabel | Strong |
|----------|------|-------|-----|------|---------|--------|
| Bupa | **0.820** | 0.568 | 0.308 | 0.453 | 0.473 | 0.459 |
| Breast can. | **0.804** | 0.657 | 0.523 | 0.710 | 0.621 | 0.606 |
| Pima | **0.856** | 0.778 | 0.658 | 0.774 | 0.720 | 0.698 |

**핵심 결과 요약:**
- NCR은 민감도는 가장 높지만 특이도가 최대 0.512 하락 (bupa)
- 제안 방법(특히 Weak Amplification)은 특이도를 SMOTE보다 잘 유지
- SMOTE는 소수 클래스 샘플을 평균 250% 증가시키는 반면, 제안 방법은 훨씬 적은 분포 변화

#### 한계점

1. **실험 범위 제한**: 9개 데이터셋에 한정, 고차원 데이터셋 미포함
2. **고정 파라미터**: $k=3$으로 고정, 최적 $k$ 탐색 미수행
3. **이진 분류에 특화**: 다중 클래스 불균형에 대한 직접 적용 어려움
4. **복제 기반 한계**: 인위적 샘플 미생성이 장점이나, 소수 클래스가 매우 극단적으로 적을 경우 효과 제한
5. **재레이블링 수용성**: 도메인에 따라 레이블 변경이 불수용될 수 있음
6. **AUC 미보고**: ROC 분석 대신 sensitivity/specificity만 보고하여 종합적 비교 어려움

---

## 3. 모델의 일반화 성능 향상 가능성

이 논문에서 일반화 성능과 관련된 핵심 요소들을 분석합니다.

### 3.1 지역적 이웃 분석을 통한 일반화 개선

제안 방법의 핵심인 **지역 이웃 기반 분류**는 다음 수식으로 표현됩니다:

$$\text{Type}(x) = \begin{cases} \text{safe} & \text{if } \sum_{i=1}^{k} \mathbb{1}[\text{label}(n_i(x)) = \text{label}(x)] > \frac{k}{2} \\ \text{noisy} & \text{otherwise} \end{cases}$$

여기서 $n_i(x)$는 $x$의 $i$번째 최근접 이웃입니다 (HVDM 거리 기준).

이 접근법이 일반화에 기여하는 메커니즘:

**1) 노이즈 제거 효과**
- 다수 클래스의 noisy 샘플 제거로 결정 경계(decision boundary) 정화
- 과적합(overfitting) 위험 감소

$$\text{정화된 훈련 데이터} = \mathcal{T} \setminus \{y \in O \mid classify\_knn(y, 3) \text{ incorrect}\}$$

**2) 어려운 영역(difficult region) 집중 처리**
- "어려운 영역"(소수 클래스가 다수 클래스에 둘러싸인 영역)의 소수 샘플만 선택적으로 증폭
- 불필요한 데이터 분포 왜곡 방지

$$\text{증폭 대상} = \{x \in C_{\text{noisy}} \mid |knn(x, 3, O, \text{safe})| > 0\}$$

**3) SMOTE 대비 클래스 경계 중복 방지**

SMOTE는 새로운 합성 샘플을 생성할 때 클래스 경계를 확인하지 않아 다음 문제가 발생할 수 있습니다:

$$x_{\text{synthetic}} = x_i + \lambda \cdot (x_j - x_i), \quad \lambda \in [0,1]$$

반면 제안 방법은 safe 다수 샘플의 수를 복제 기준으로 사용하여 안전한 영역만 강화:

$$\text{copies}(x) \propto |knn(x, 3, O, \text{safe})| \leq 3$$

이는 최대 3개의 복제본만 생성하므로 분포 왜곡이 제한적입니다.

### 3.2 분류기 독립성(Classifier-agnostic) 특성

전처리 단계가 학습 알고리즘(C4.5, MODLEM)과 분리되어 있어:
- 다양한 학습기에 적용 가능 → 범용적 일반화 가능성
- 단, 논문에서 두 가지 알고리즘에서만 검증

### 3.3 일반화 성능의 실험적 증거

Wilcoxon Signed Ranks Test ($\alpha = 0.05$) 결과:
- Weak Amplification + MODLEM: 6개 데이터셋에서 최고 또는 2번째 전체 정확도 달성
- SMOTE 대비 데이터 분포 변화가 적어 분류기의 훈련 데이터 의존도 감소

### 3.4 일반화 성능 향상의 잠재적 한계

$$\text{일반화 위험} = f(\text{데이터 분포 변화량}, \text{노이즈 수준}, \text{클래스 불균형 비율})$$

- 클래스 비율 $R_C = N_C/N$이 매우 낮은 경우(예: ecoli, $R_C = 0.10$), 복제만으로는 충분한 결정 경계 학습이 어려울 수 있음
- 고차원 공간에서 HVDM 기반 이웃 탐색의 차원의 저주(curse of dimensionality) 영향

---

## 4. 향후 연구에 미치는 영향 및 고려할 점

### 4.1 향후 연구에 미치는 영향

**1) 하이브리드 전처리 패러다임의 확립**
- 오버샘플링 + 언더샘플링의 단순 결합이 아닌, **지역적 분석 기반 선택적 처리**라는 새로운 패러다임 제시
- 이후 연구에서 ADASYN, Borderline-SMOTE 등 지역적 오버샘플링 방법의 이론적 기반 강화에 기여

**2) Safe/Noisy 예제 구분 개념의 확산**
- 데이터 복잡도(data complexity) 관점에서 샘플을 구분하는 개념이 이후 연구(예: 불균형 앙상블 학습)에서 널리 활용됨

**3) 민감도-특이도 트레이드오프 명시적 고려**
- 단순 AUC 최대화가 아닌 두 지표의 균형을 명시적으로 고려하는 평가 프레임워크 제시

**4) 도메인 수용성 고려**
- 의료, 금융 등 해석 가능성이 중요한 도메인에서 인위적 샘플을 생성하지 않는 접근의 중요성을 부각

### 4.2 향후 연구 시 고려할 점

**① 고차원 및 복잡한 데이터 대응**

$$d_{\text{HVDM}}(x, y) \text{ in high-dim} \Rightarrow \text{차원의 저주 대응 필요}$$

- 차원 축소(PCA, t-SNE) 또는 특징 선택과 결합한 전처리 연구 필요
- 희소 고차원 데이터(텍스트, 유전체)에서의 적용성 검증

**② 최적 $k$ 값 자동 선택**

현재 $k=3$으로 고정되어 있으나:
$$k^* = \arg\min_k \mathcal{L}(\text{Sensitivity}(k), \text{Specificity}(k))$$
- 교차검증 기반 또는 베이지안 최적화를 통한 $k$ 자동 선택 연구 필요

**③ 극단적 불균형(Extreme Imbalance) 대응**

$R_C \ll 0.01$인 경우(예: 사기 탐지, 희귀 질환):
- 복제 기반 증폭만으로는 부족하며 생성 모델(GAN, VAE) 기반 합성과의 결합 고려

**④ 다중 클래스 불균형으로 확장**

현재 이진 분류에 특화되어 있으며 다중 클래스 불균형 문제로의 일반화 필요:
$$\mathcal{C} = \{C_1, C_2, \ldots, C_m\} \text{ with } |C_1| \ll |C_2| \leq \ldots \leq |C_m|$$

**⑤ 딥러닝 환경과의 통합**

전처리 기반 방법의 특성상 딥러닝 파이프라인에의 통합:
- 배치 단위 온라인 전처리 전략
- 자기 지도 학습(self-supervised learning)과의 결합

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의**: 아래 연구들은 제 학습 데이터(2024년 초까지)를 기반으로 한 일반적인 지식이며, 논문 원문을 직접 확인하지 않아 일부 세부 내용에 오차가 있을 수 있습니다. 반드시 원문을 확인하시기 바랍니다.

### 5.1 주요 후속 연구 방향

**[방향 1] 생성 모델 기반 오버샘플링**

| 연구 | 핵심 접근 | Stefanowski(2008)과의 차이 |
|------|-----------|---------------------------|
| CTGAN (Xu et al., 2019) | 조건부 GAN으로 표 형식 합성 | 인위적 샘플 생성 vs. 복제 |
| SMOTE-GAN 계열 | GAN + SMOTE 결합 | 분포 학습 기반 vs. 이웃 기반 |

$$x_{\text{synthetic}} \sim G_\theta(z), \quad z \sim \mathcal{N}(0, I)$$

**[방향 2] 메타-러닝 및 소수샷 학습**

불균형 데이터를 소수샷(few-shot) 문제로 재정의하여 프로토타입 네트워크 등 활용. Stefanowski(2008)의 지역 이웃 개념과 유사하나 학습된 거리 메트릭 활용.

**[방향 3] 앙상블 + 불균형 처리 통합**

- **BalancedRandomForest, EasyEnsemble**: 각 약 분류기 훈련 시마다 재샘플링
- Stefanowski(2008)의 전처리 방법을 앙상블 내 서브샘플링과 결합하는 연구

**[방향 4] 비용 민감 학습(Cost-sensitive Learning)**

$$\mathcal{L}_{\text{cost}} = C_{FN} \cdot \text{FN} + C_{FP} \cdot \text{FP}$$

여기서 $C_{FN} \gg C_{FP}$ (소수 클래스 오분류 비용 우선)

Stefanowski(2008)의 방법이 전처리 수준에서 암묵적으로 수행하는 것을 비용 함수로 명시화.

**[방향 5] 데이터 복잡도(Data Complexity) 연구**

논문의 safe/noisy 구분 개념을 정형화:

$$\text{Typicality}(x) = \frac{|knn(x, k, \text{same class})|}{k}$$

이를 통해 샘플을 safe, borderline, rare, outlier 등으로 세분화하는 연구들이 발전.

### 5.2 종합 비교표

| 특성 | Stefanowski(2008) | SMOTE 계열 (현재) | GAN 기반 (2020+) | 앙상블 기반 (2020+) |
|------|-------------------|-------------------|------------------|---------------------|
| 인위적 샘플 생성 | ❌ (복제/재레이블링) | ✅ | ✅ | 방법에 따라 다름 |
| 지역적 분석 | ✅ | 부분적 | ❌ (전역적) | 부분적 |
| 파라미터 자동화 | 부분적 | ❌ | ✅ (학습 기반) | ✅ |
| 해석 가능성 | 높음 | 중간 | 낮음 | 중간 |
| 계산 복잡도 | $O(n^2)$ KNN | $O(n^2)$ KNN | $O(\text{GAN 훈련})$ | $O(T \cdot n)$ |
| 고차원 적용성 | 제한적 | 제한적 | 높음 | 높음 |

---

## 참고 자료

### 논문 원문
1. **Stefanowski, J., Wilk, S.** (2008). "Selective Pre-processing of Imbalanced Data for Improving Classification Performance." *DaWaK 2008, LNCS 5182*, pp. 283–292. Springer-Verlag Berlin Heidelberg. *(본 분석의 주요 원문)*

### 논문 내 참고문헌 (원문에 명시된 것)
2. Batista, G., Prati, R., Monard, M. (2004). "A study of the behavior of several methods for balancing machine learning training data." *ACM SIGKDD Explorations Newsletter* 6(1), 20–29.
3. Chawla, N. (2005). "Data mining for imbalanced datasets: An overview." *The Data Mining and Knowledge Discovery Handbook*, pp. 853–867. Springer.
4. Chawla, N., Bowyer, K., Hall, L., Kegelmeyer, W. (2002). "SMOTE: Synthetic Minority Over-sampling Technique." *Journal of Artificial Intelligence Research* 16, 341–378.
5. Japkowicz, N., Stephen, S. (2002). "The Class Imbalance Problem: A Systematic Study." *Intelligent Data Analysis* 6(5), 429–450.
6. Kubat, M., Matwin, S. (1997). "Addressing the curse of imbalanced training sets: one-side selection." *Proc. of the 14th Int. Conf. on Machine Learning*, pp. 179–186.
7. Laurikkala, J. (2001). "Improving identification of difficult small classes by balancing class distribution." *Tech. Report A-2001-2*, University of Tampere.
8. Stefanowski, J. (1998). "The rough set based rule induction technique for classification problems." *Proc. EUFIT 1998*, pp. 109–113.
9. Stefanowski, J., Wilk, S. (2006). "Rough sets for handling imbalanced data." *Fundamenta Informaticae* 72, 379–391.
10. Stefanowski, J., Wilk, S. (2007). "Improving Rule Based Classifiers Induced by MODLEM by Selective Pre-processing of Imbalanced Data." *Proc. RSKD Workshop at ECML/PKDD*, pp. 54–65.
11. Van Hulse, J., Khoshgoftaar, T., Napolitano, A. (2007). "Experimental perspectives on learning from imbalanced data." *Proc. ICML 2007*, pp. 935–942.
12. Weiss, G.M. (2004). "Mining with rarity: a unifying framework." *ACM SIGKDD Explorations Newsletter* 6(1), 7–19.
13. Wilson, D.R., Martinez, T. (2000). "Reduction techniques for instance-based learning algorithms." *Machine Learning Journal* 38, 257–286.
