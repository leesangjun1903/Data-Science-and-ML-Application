# RFE 및 RFECV: 재귀적 특성 제거 기법의 종합 분석

> **참고 사이트**: https://process-mining.tistory.com/138 (프로세스 마이닝 블로그 - RFE/RFECV 관련 포스트)

---

## 1. 핵심 주장 및 주요 기여 요약

해당 사이트는 **Scikit-learn의 RFE(Recursive Feature Elimination)와 RFECV(RFE with Cross-Validation)** 기반 특성 선택(Feature Selection) 기법을 다루고 있습니다.

핵심 주장은 다음과 같습니다:
- 고차원 데이터에서 불필요한 특성을 제거하면 **모델의 과적합(overfitting)을 방지하고 일반화 성능을 향상**시킬 수 있다.
- RFE는 데이터셋에서 가장 관련성 높은 특성의 부분집합을 선택하기 위한 특성 선택 알고리즘으로, 모든 특성에서 시작하여 원하는 수의 특성에 도달할 때까지 가장 덜 중요한 특성을 반복적으로 제거하는 재귀적 과정이다.
- 교차 검증과 결합한 RFE는 주어진 모델에 대해 가장 관련성 높은 특성을 식별하는 유용한 기법으로, 머신러닝 모델의 성능과 해석 가능성을 향상시킬 수 있다.

**주요 기여**:
1. 래퍼(Wrapper) 기반 특성 선택 방법으로서 모델 성능에 직접적으로 기반한 특성 부분집합 선택
2. RFECV를 통한 최적 특성 개수의 자동 결정
3. Python Scikit-learn을 활용한 실용적 구현 가이드 제공

---

## 2. RFE와 RFECV의 상세 분석

### 2.1 해결하고자 하는 문제

많은 특성을 갖는 것이 항상 더 나은 머신러닝 모델을 의미하지는 않으며, 사실 너무 많은 특성은 과적합, 증가된 계산 비용, 감소된 모델 해석 가능성을 초래할 수 있다. 이것이 특성 선택 기법이 필요한 이유이며, RFE는 가장 효과적인 방법 중 하나로 꼽힌다.

구체적으로 해결하고자 하는 문제:
- **차원의 저주 (Curse of Dimensionality)**: 고차원 데이터에서 노이즈 특성이 모델 성능을 저하
- **과적합 방지**: 덜 중요한 특성에서 발생하는 불필요한 노이즈를 제거하여 모델을 정규화하고 과적합을 방지하며, 입력 특성 간의 종속성과 다중공선성을 제거하여 역시 과적합을 방지한다.
- **최적 특성 수 결정의 어려움**: RFE의 또 다른 근본적 한계는 유지할 특성의 수 선택이 사용자에게 맡겨져 있다는 것이다.

### 2.2 제안하는 방법 및 수식

#### RFE (Recursive Feature Elimination)

DNA 마이크로어레이에 기록된 광범위한 유전자 발현 데이터에서 작은 유전자 부분집합을 선택하는 문제를 다루며, 재귀적 특성 제거(RFE) 기반의 SVM 방법을 활용한 새로운 유전자 선택 방법을 제안하였다. 이 방법은 Guyon et al. (2002)에 의해 최초로 제안되었습니다.

**알고리즘 단계:**

RFE는 재귀적으로 특성을 제거하고 남은 속성으로 모델을 구축하는 역방향 특성 선택 알고리즘이다. 모델의 계수 또는 특성 중요도를 사용하여 예측에 가장 적게 기여하는 특성을 식별하고 제거하며, 이 과정은 원하는 특성 수에 도달할 때까지 계속된다.

**SVM-RFE의 수식적 기반:**

SVM의 결정 함수는 다음과 같습니다:

$$f(\mathbf{x}) = \mathbf{w} \cdot \mathbf{x} + b$$

여기서 $\mathbf{w}$는 가중치 벡터이며, SVM 학습은 다음 최적화 문제를 풀어 수행합니다:

$$\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^{\ell} \xi_i$$

$$\text{s.t.} \quad y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

RFE에서의 **특성 중요도 랭킹 기준 (Ranking Criterion)**은 가중치의 제곱입니다:

$$J(i) = w_i^2$$

특성 $i$를 제거했을 때의 목적함수 변화량은:

$$\Delta J(i) = \frac{1}{2} \|\mathbf{w}\|^2 - \frac{1}{2} \|\mathbf{w}_{(-i)}\|^2 \approx \frac{1}{2} w_i^2$$

따라서 **가장 작은 $\Delta J(i)$를 가진 특성**이 매 반복에서 제거됩니다.

일반적인 모델(트리 기반 등)에서는:

```math
\text{Ranking Score}_i = |\texttt{coef\_}[i]| \quad \text{또는} \quad \texttt{feature\_importances\_}[i]
```

**RFE 알고리즘 의사코드:**

```
입력: 학습 데이터 S₀ = {(x₁,y₁), ..., (xₗ,yₗ)}, 전체 특성 집합 F₀ = {1, 2, ..., n}
     선택할 특성 수 k

반복 (t = 1, 2, ...):
    1. 모델을 Fₜ₋₁ 특성으로 학습
    2. 각 특성 i ∈ Fₜ₋₁에 대해 중요도 계산: J(i)
    3. 가장 중요도가 낮은 특성 제거:
       f* = argmin_{i ∈ Fₜ₋₁} J(i)
       Fₜ = Fₜ₋₁ \ {f*}
    4. |Fₜ| = k이면 종료

출력: 선택된 특성 집합 Fₜ, 특성 랭킹 r
```

#### RFECV (RFE with Cross-Validation)

RFECV는 교차 검증을 통해 특성을 선택하는 재귀적 특성 제거 방법이다. 선택되는 특성의 수는 서로 다른 교차 검증 분할에서 RFE 선택자를 피팅하여 자동으로 조정된다. 각 RFE 선택자의 성능은 다양한 수의 선택된 특성에 대해 scoring으로 평가되고 합산된다. 최종적으로 점수는 폴드별로 평균되고, 교차 검증 점수를 최대화하는 특성 수가 선택된다.

**RFECV의 수식적 표현:**

$k$-fold 교차 검증에서 특성 부분집합 크기 $d$에 대한 평균 교차 검증 점수:

$$\bar{S}(d) = \frac{1}{k} \sum_{j=1}^{k} S_j(d)$$

여기서 $S_j(d)$는 $j$번째 폴드에서 $d$개의 특성을 사용했을 때의 성능 점수입니다.

**최적 특성 수 $d^*$** 결정:

$$d^* = \arg\max_{d \in \{d_{\min}, d_{\min}+\text{step}, \ldots, n\}} \bar{S}(d)$$

여기서 $d_{\min}$은 `min_features_to_select` 파라미터입니다.

**점수의 표준편차:**

$$\sigma_S(d) = \sqrt{\frac{1}{k} \sum_{j=1}^{k} (S_j(d) - \bar{S}(d))^2}$$

cv_results_ 내 모든 값의 크기는 

```math
\lceil(n_{\text{features}} - \text{min\_features\_to\_select}) / \text{step}\rceil + 1
```

 과 같으며, 여기서 step은 각 반복에서 제거되는 특성 수이다.

### 2.3 모델 구조

```
┌─────────────────────────────────────────────────┐
│                   RFECV 전체 구조                  │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌───────────────────────────────────────────┐  │
│  │         k-Fold Cross-Validation           │  │
│  │  ┌─────────────────────────────────────┐  │  │
│  │  │      Fold j (j=1,...,k)             │  │  │
│  │  │                                     │  │  │
│  │  │  ┌───────────────────────────────┐  │  │  │
│  │  │  │      RFE (내부 루프)            │  │  │  │
│  │  │  │                               │  │  │  │
│  │  │  │  1. 전체 특성으로 모델 학습     │  │  │  │
│  │  │  │  2. 특성 중요도 계산           │  │  │  │
│  │  │  │  3. 최하위 특성 제거           │  │  │  │
│  │  │  │  4. 각 d에서 Score(d) 기록     │  │  │  │
│  │  │  │  5. d_min까지 반복             │  │  │  │
│  │  │  └───────────────────────────────┘  │  │  │
│  │  │                                     │  │  │
│  │  │  출력: S_j(d) for all d             │  │  │
│  │  └─────────────────────────────────────┘  │  │
│  │                                           │  │
│  │  평균 점수 계산: S̄(d) = (1/k)ΣS_j(d)     │  │
│  │  최적 d* = argmax S̄(d)                   │  │
│  └───────────────────────────────────────────┘  │
│                                                 │
│  최종: d* 특성으로 전체 데이터에서 모델 재학습     │
└─────────────────────────────────────────────────┘
```

외부 추정기가 특성에 가중치를 부여할 때, RFE의 목표는 점점 더 작은 특성 집합을 재귀적으로 고려하여 특성을 선택하는 것이다. 먼저, 초기 특성 집합에서 추정기를 학습시키고 각 특성의 중요도를 얻는다. 그런 다음, 가장 덜 중요한 특성이 현재 특성 집합에서 제거된다. 이 절차는 원하는 수의 특성에 도달할 때까지 축소된 집합에서 재귀적으로 반복된다.

### 2.4 성능 향상

실험적으로 RFE 기법으로 선택된 유전자들이 더 나은 분류 성능을 산출하며 암과 관련하여 생물학적으로 관련이 있음을 입증하였다. 기존 방법과 달리, 이 방법은 유전자 중복성을 자동으로 제거하며 더 나은, 더 컴팩트한 유전자 부분집합을 산출한다. 백혈병 환자에서 2개의 유전자만으로 leave-one-out 오류 0을 달성한 반면, 기존 방법은 64개 유전자가 필요했다. 대장암 데이터에서 4개 유전자만으로 98% 정확도를 달성한 반면, 기존 방법은 86% 정확도에 그쳤다.

DT-RFECV를 사용하여 UNSW-NB15의 42개 특성 중 최적의 15개 특성 부분집합을 선택하고 여러 ML 분류기로 평가하였다. 제안된 NIDS는 전체 특성 세트 사용 시 95.56% 대비 95.30%의 이진 분류 정확도를 보여, 특성 수를 대폭 줄이면서도 거의 동등한 성능을 유지하였다.

**RFE의 장점:**
- 모델에 구애받지 않음(model-agnostic): 특성 중요도 점수를 제공하는 모든 추정기와 함께 작동하며, 단변량 방법과 달리 특성 간 상호작용을 고려한다.

### 2.5 한계점

| 한계점 | 설명 |
|--------|------|
| **계산 비용** | 각 부분집합에 대해 모델을 재학습시키는 것은, 특히 XGBoost 같은 복잡한 모델이나 대규모 데이터셋에서 시간 집약적일 수 있다. |
| **중복 특성 미처리** | 특성 수가 많은 경우, Boruta가 일반적으로 RFE보다 훨씬 적은 반복으로 모든 불필요한 특성을 탐지하지만, RFE와 마찬가지로 Boruta도 중복 특성을 탐지하고 제거하는 데 한계가 있다. |
| **추정기 의존성** | 서로 다른 추정기가 서로 다른 특성을 선택할 수 있다. |
| **데이터 드리프트 취약성** | RFE(SHAP 사용)와 Boruta를 벤치마킹한 결과, 재귀적 특성 선택은 데이터의 노이즈가 증가함에 따라 진정한 의미 있는 예측 변수를 탐지하는 데 어려움을 보였다. |
| **높은 카디널리티 변수 편향** | 의사결정 트리의 표준 특성 중요도 방법은 고빈도 또는 높은 카디널리티 변수의 중요성을 과대평가하는 경향이 있으며, Boruta와 RFE에서 이는 잘못된 특성 선택으로 이어질 수 있다. |

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 교차 검증을 통한 일반화 보장

RFECV의 핵심 설계 철학은 **교차 검증을 통해 일반화 성능을 직접 최적화**하는 것입니다.

$$d^* = \arg\max_d \left[ \frac{1}{k} \sum_{j=1}^{k} S_j^{\text{test}}(d) \right]$$

여기서 $S_j^{\text{test}}(d)$는 $j$번째 폴드의 **테스트 세트**에서 평가된 점수입니다. 이는 학습 세트에서의 성능이 아닌, 보지 않은 데이터에서의 성능을 기준으로 특성 수를 결정하므로 일반화 성능과 직결됩니다.

RFECV는 머신러닝 알고리즘을 사용하여 가장 관련성 높은 특성을 선택하는 래퍼 특성 선택 방법이다. 견고성을 보장하기 위해 RFECV는 재귀적 특성 제거와 교차 검증을 결합하여 모델 성능을 최대화하는 최적의 특성 수를 식별한다.

### 3.2 과적합 방지 메커니즘

특성 선택은 **모델 복잡도를 감소**시키는 정규화(regularization)의 한 형태로 기능합니다:

$$\text{Bias-Variance Tradeoff}: \quad \text{MSE} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

- **특성이 과도하게 많을 때**: Variance ↑ → 과적합
- **RFE로 불필요한 특성 제거 시**: Variance ↓ → 일반화 향상
- 5개 이상의 선택된 특성에서 테스트 정확도가 감소하는데, 이는 비정보적 특성을 유지하면 과적합으로 이어지며 따라서 통계적 성능에 해롭다는 것을 보여준다.

### 3.3 특성 선택의 안정성과 일반화

다섯 개의 폴드에서 선택된 특성이 일관적이었으며, 이는 선택이 폴드 간에 안정적이라는 것을 의미하고, 해당 특성들이 가장 정보적이라는 것을 확인해 준다.

교차 검증 폴드 간 특성 선택의 안정성(stability)은 다음과 같이 측정할 수 있습니다:

$$\text{Stability}(F_1, F_2) = \frac{|F_1 \cap F_2|}{\sqrt{|F_1| \cdot |F_2|}}$$

폴드 간 높은 안정성은 모델의 일반화 성능이 높다는 간접적 지표가 됩니다.

### 3.4 일반화 향상을 위한 실무 권장사항

| 전략 | 효과 |
|------|------|
| StratifiedKFold 사용 | 클래스 불균형 시에도 안정적 평가 |
| 충분한 fold 수 (k=5~10) | 평가의 분산 감소 |
| 홀드아웃 검증 | 항상 홀드아웃 데이터에서 결과를 검증해야 한다. |
| SHAP 기반 중요도 사용 | 특성 중요도 편향 완화 |

---

## 4. 향후 연구에 미치는 영향 및 고려할 점

### 4.1 RFE/RFECV가 후속 연구에 미친 영향

Guyon, Weston, Barnhill & Vapnik (2002)이 SVM과 함께 RFE를 개발한 것은 생물정보학에서 널리 사용되게 되었으며, Guyon & Elisseeff (2003)는 특성 선택 방법의 분류 체계를 필터, 래퍼, 임베디드 범주로 형식화하였다.

RFE는 이후 **다양한 도메인**으로 확장되었습니다:
- **침입 탐지 시스템**: 머신러닝 기반 침입 탐지 시스템을 위한 이진 의사결정 트리 분류를 사용한 RFECV 접근법이 제안되었다.
- **유전체학**, **금융 데이터**, **IoT 보안** 등 고차원 데이터 분야 전반

### 4.2 앞으로의 연구 시 고려할 점

1. **계산 효율성**: RFECV의 시간 복잡도는 $O(n^2 \cdot k \cdot T_{\text{model}})$로, 대규모 데이터셋에서는 근사 기법이나 샘플링 전략이 필요
2. **SHAP 기반 중요도 통합**: 특성 중요도 계산에 SHAP을 활용하면, 높은 카디널리티 변수의 특성 선택에 대한 영향을 완화하는 데 도움이 된다.
3. **시간적 데이터 드리프트 대응**: 대부분의 데이터 기반 알고리즘과 마찬가지로 특성 선택도 시간이 지나면 부정확해질 수 있으므로, 새로운 패턴을 포착하기 위해 최신 데이터로 필터링 프로세스를 반복해야 하며, 이는 모든 래퍼 선택 방법에 대한 진정한 필요성이다.
4. **중복 특성 탐지 한계 극복**: 상호 정보량(Mutual Information) 기반 방법과의 결합 연구 필요

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 주요 최신 연구 비교표

| 방법 | 연도 | 핵심 특징 | RFE 대비 장점 | RFE 대비 단점 |
|------|------|----------|--------------|--------------|
| **RFE/RFECV** (Baseline) | 2002 (원본) | 재귀적 제거 + CV | - | 계산 비용 높음, 중복 미처리 |
| **Boruta + SHAP** | 2020+ | Shadow feature + SHAP 중요도 | RFE가 데이터 노이즈 증가 시 어려움을 보인 반면, Boruta는 데이터 드리프트가 심해져도 항상 진정한 시스템 패턴을 탐지하였다. | 계산량이 더 큼 |
| **shap-select** (2024) | 2024 | SHAP + 통계적 유의성 회귀 | HISEL, RFE, shap-selection 등 전통적 방법보다 모델 성능 면에서 우수하며, SHAP 값을 한 번 계산하고 선형/로지스틱 회귀를 통해 반복적으로 최소 유의 특성을 제거하여 모델을 다른 부분집합에서 재학습시키지 않으므로 계산적으로 효율적이다. | 비교적 새로운 방법 |
| **BorutaShapPlus** | 2024 | Boruta + SHAP 통합 | Boruta 특성 선택 기법과 SHAP 값을 활용한 특성 점수화를 결합하여, 모델의 네이티브 특성 중요도 대신 사용한다. | 하이퍼파라미터 튜닝 필요 |
| **RFECV + DT** (IDS용, 2023) | 2023 | RFECV를 침입 탐지에 적용 | 42개 특성에서 15개로 감소시키면서 95.30%의 정확도를 유지하여, 실시간 탐지에 적합한 빠른 예측과 낮은 저장 공간을 실현하였다. | 도메인 특화적 |
| **Boruta, SHAP, BorutaShap 비교** (Ejiyi et al., 2024) | 2024 | 질병 진단에서 비교 분석 | 해석 가능한 ML 모델이 질병 진단에 핵심적이며, SHAP은 당뇨병, 심혈관, 갑상선 질환 데이터셋에서 각각 80.17%, 85.13%, 90.00%, 99.55%의 평균 정확도로 우수한 성능을 보였다. | SHAP 계산 비용 |

### 5.2 정보이론적 대안

RFE, Boruta, SHAP 값이 충분하지 않은 이유를 설명하고, 정보이론적 대안을 제안하는 연구도 등장하였다.

### 5.3 발전 방향 종합

```
2002: SVM-RFE (Guyon et al.)
  ↓
2010: Boruta (Kursa & Rudnicki) - Shadow Feature 개념 도입
  ↓
2017: SHAP (Lundberg) - 게임이론 기반 설명 가능성
  ↓
2020+: SHAP + RFE/Boruta 융합
  ↓
2024: shap-select, BorutaShapPlus
  - 계산 효율 + 해석 가능성 + 성능의 균형점 추구
```

최근 연구의 핵심 트렌드는 **RFE의 재귀적 제거 프레임워크를 유지하되, 특성 중요도 평가 단계에서 SHAP 값 등 더 견고한 방법을 결합**하는 방향으로 발전하고 있습니다.

---

## 참고 문헌 및 출처

1. **해당 사이트**: https://process-mining.tistory.com/138
2. **Guyon, I., Weston, J., Barnhill, S., & Vapnik, V.** (2002). "Gene Selection for Cancer Classification using Support Vector Machines." *Machine Learning*, 46(1-3), 389–422.
3. **Scikit-learn 공식 문서**: `sklearn.feature_selection.RFE` — https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html
4. **Scikit-learn 공식 문서**: `sklearn.feature_selection.RFECV` — https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html
5. **Scikit-learn Feature Selection User Guide**: https://scikit-learn.org/stable/modules/feature_selection.html
6. **GeeksforGeeks**: "Recursive Feature Elimination with Cross-Validation in Scikit Learn" — https://www.geeksforgeeks.org/recursive-feature-elimination-with-cross-validation-in-scikit-learn/
7. **Kraev, E., Koseoglu, B., Traverso, L., Topiwalla, M.** (2024). "shap-select: Lightweight Feature Selection Using SHAP Values and Regression." — arXiv:2410.06815
8. **MDPI Journal** (2023). "Recursive Feature Elimination with Cross-Validation with Decision Tree: Feature Selection Method for ML-Based IDS." — https://www.mdpi.com/2224-2708/12/5/67
9. **Ejiyi, C. J., et al.** (2024). "Comparative performance analysis of Boruta, SHAP, and BorutaShap for disease diagnosis." *Network: Computation in Neural Systems*, 36(3), 507-544.
10. **Towards Data Science**: "Boruta and SHAP for better Feature Selection" — https://towardsdatascience.com/boruta-and-shap-for-better-feature-selection-20ea97595f4a/
11. **Towards Data Science**: "Boruta SHAP for Temporal Feature Selection" — https://towardsdatascience.com/boruta-shap-for-temporal-feature-selection-96a7840c7713/
12. **KXY Blog**: "Effective Feature Selection: Beyond SHAP, RFE and Boruta" — https://blog.kxy.ai/effective-feature-selection/
13. **Yellowbrick Documentation**: RFECV Visualizer — https://www.scikit-yb.org/en/latest/api/model_selection/rfecv.html
