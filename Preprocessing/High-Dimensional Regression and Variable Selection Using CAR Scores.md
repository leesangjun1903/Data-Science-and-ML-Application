# High-Dimensional Regression and Variable Selection Using CAR Scores
**Zuber & Strimmer (2011), Statistical Applications in Genetics and Molecular Biology 10: 34**

---

## 1. Executive Summary (10문장 이내)

1. 본 논문은 고차원 유전체 데이터 분석에서의 변수 선택 문제를 해결하기 위해 **CAR(Correlation-Adjusted marginal coRrelation) 점수**라는 새로운 변수 중요도 기준을 제안한다.
2. CAR 점수는 예측 변수들을 Mahalanobis 변환으로 동시에 직교화(decorrelate)한 후, 반응 변수와 직교화된 예측 변수 간의 상관관계를 측정하는 방식으로 정의된다.
3. 핵심 수식은 $\boldsymbol{\omega} = \boldsymbol{P}^{-1/2}\boldsymbol{P}_{XY}$ 이며, 이는 주변 상관계수(marginal correlation)와 표준화 회귀계수(standardized regression coefficient) 사이의 중간적 특성을 지닌다.
4. CAR 점수의 제곱합은 결정계수와 일치하는 분해 성질 $\Omega^2 = \sum_{j=1}^{d}\omega_j^2$을 만족하며, 이는 변수 중요도 기준으로서의 이론적 정당성을 제공한다.
5. 높은 상관관계를 가진 예측변수들을 함께 선택하는 **그룹화 성질(grouping property)**과 서로 상쇄되는 **대립 변수(antagonistic variable)를 하위 순위로 내리는 특성**을 본질적으로 내포한다.
6. 모집단(population) 수준의 양으로 정의되어, 특정 추론 패러다임(베이지안, 최대우도 등)에 종속되지 않으며, 대표본에서는 경험적(empirical) 추정, 소표본에서는 축소(shrinkage) 추정을 유연하게 활용할 수 있다.
7. 4가지 시뮬레이션 시나리오와 2개의 실제 데이터(당뇨병 데이터, 뇌 유전자 발현 데이터) 분석을 통해 검증하였다.
8. 시뮬레이션 결과, CAR 점수 기반 변수 선택은 elastic net과 동등하거나 그 이상의 성능을 보이며, lasso, boosting, OLS보다 우수한 예측 오차와 진양성/위양성 비율을 달성하였다.
9. R 패키지 `care`로 구현되어 CRAN에서 공개적으로 제공된다.
10. 논문의 한계로는 선형 모델에 국한되며, 비선형 관계나 분류(classification) 문제에 대한 직접적 적용은 다루지 않는다.

---

### 1-1. 연구의 목적과 필요성

**배경 및 필요성 (pp. 1–2):**

유전체학을 비롯한 생명과학 분야에서 **"small $n$, large $d$"** 문제, 즉 관측 수($n$)보다 변수 수($d$)가 훨씬 많은 고차원 데이터가 일반화되었다. 기존의 변수 선택 방법들(LASSO, elastic net, boosting 등)은 다음과 같은 공통적 한계를 가진다:

| 문제점 | 설명 |
|--------|------|
| 특정 추론 패러다임 종속성 | 변수 선택이 특정 정규화 절차와 불가분하게 연결됨 |
| 예측변수 간 상관관계 미처리 | 주변 상관(marginal correlation)은 예측변수 간 상관 존재 시 부적절 |
| 분산 분해 불가 | 표준화 회귀계수나 편상관(partial correlation)은 $\Omega^2$ 분해 불만족 |
| 그룹화 성질 부재 | 일부 방법은 높은 상관 변수들을 동시에 선택하지 못함 |

이에 저자들은 **모집단 수준(population-level)**에서 정의되어 어떤 추론 방식과도 결합 가능하고, 예측변수 간 상관을 체계적으로 조정하며, $\Omega^2$의 정준 분해(canonical decomposition)를 제공하는 새로운 변수 중요도 기준의 필요성을 제기한다.

---

## 2. 핵심 주장과 근거 표

| 핵심 주장 | 근거 | 위치 |
|-----------|------|------|
| CAR 점수는 $\Omega^2$의 정준 분해를 제공한다 | $\Omega^2 = \sum_{j=1}^{d}\omega_j^2$ 수학적 증명 | p. 10, Section 4.6 |
| CAR 점수는 그룹화 성질을 가진다 | $\omega_1^2 - \omega_2^2 = \left[(\boldsymbol{b}\_\text{std})\_1^2 - (\boldsymbol{b}_\text{std})_2^2\right]\sqrt{1-\rho^2}$ | p. 12, Section 4.9 |
| 대립 변수(antagonistic)를 하위 순위로 내린다 | 양의 상관 + 반대 부호 회귀계수 → CAR 점수 → 0 | p. 12–13, Section 4.9 |
| 주변 상관과 표준화 회귀계수 사이의 중간 | Table 2: $\boldsymbol{b}\_\text{std} = \boldsymbol{P}^{-1/2}\boldsymbol{\omega}$, $\boldsymbol{P}_{XY} = \boldsymbol{P}^{1/2}\boldsymbol{\omega}$ | p. 8, Table 2 |
| elastic net과 동등 이상의 성능 | 4가지 시뮬레이션 시나리오, 200회 반복 | p. 14–16, Table 5–6 |
| AIC/BIC 등 정보기준과 직접 연결 | $\hat{\omega}_c^2 = \frac{\lambda(1-R^2)}{n}$ 임계값 관계 | p. 11–12, Section 4.8 |
| 소표본 설정에서도 효과적 | $d=40$, $n=10\sim100$ 시뮬레이션 | p. 16, Table 6 |
| 유전자 발현 데이터에서 elastic net보다 낮은 CV 오차 | CAR(85): 0.2960 vs Elastic Net(85): 0.3417 | p. 21, Table 9 |

---

## 2-1. 해결 문제, 제안 방법, 모델 구조, 성능 및 한계

### 해결하고자 하는 문제

1. **예측변수 간 상관 존재 시** 변수 중요도 측정의 부정확성
2. 기존 기준들($\rho_j^2$, $\boldsymbol{b}_\text{std}$, 편상관)이 $\Omega^2$를 분해하지 못하거나, 예측 방정식과의 연결이 불명확
3. 변수 선택이 특정 추론 절차에 종속되는 문제
4. 고상관 예측변수들의 동시 선택(그룹화) 및 대립 변수 처리

### 제안하는 방법 (수식 포함)

**[핵심 정의]** CAR 점수 $\boldsymbol{\omega}$는 다음과 같이 정의된다 (p. 8, Eq. 6):

$$\boldsymbol{\omega} = \boldsymbol{P}^{-1/2}\boldsymbol{P}_{XY}$$

여기서:
- $\boldsymbol{P}$: 예측변수 간 상관행렬 ($d \times d$)
- $\boldsymbol{P}_{XY} = (\rho_1, \ldots, \rho_d)^T$: 반응변수와 예측변수 간 주변 상관벡터
- $\boldsymbol{P}^{-1/2}$: 양정치 대칭 행렬 제곱근의 역행렬

**[Mahalanobis 변환]** 예측변수의 직교화 (p. 9, Eq. 8):

$$\boldsymbol{\delta}(\boldsymbol{X}) = \boldsymbol{P}^{-1/2}\boldsymbol{V}^{-1/2}(\boldsymbol{X} - \boldsymbol{\mu}) = \boldsymbol{P}^{-1/2}\boldsymbol{X}_\text{std}, \quad \text{Var}(\boldsymbol{\delta}(\boldsymbol{X})) = \boldsymbol{I}$$

**[CAR 점수를 이용한 최적 예측식]** (p. 9, Eq. 7):

$$Y^\star_\text{std} = \boldsymbol{\omega}^T\boldsymbol{\delta}(\boldsymbol{X}) = \sum_{j=1}^{d}\omega_j\delta_j(\boldsymbol{X})$$

**[분산 분해]** (p. 10, Section 4.6):

$$\Omega^2 = \boldsymbol{\omega}^T\boldsymbol{\omega} = \sum_{j=1}^{d}\omega_j^2, \quad \phi^\text{CAR}(X_j) = \omega_j^2$$

**[그룹화 성질]** 두 변수 $X_1, X_2$의 상관이 $\rho$일 때 (p. 12):

$$\omega_1^2 - \omega_2^2 = \left[(\boldsymbol{b}_\text{std})_1^2 - (\boldsymbol{b}_\text{std})_2^2\right]\sqrt{1-\rho^2}$$

→ $|\rho| \to 1$이면 두 CAR 점수는 동일해짐

**[정보기준과의 연결]** (p. 11, Section 4.8):

$$\frac{\text{RSS}^\text{penalized}_k}{n\hat{\sigma}_Y^2} = 1 - \sum_{j=1}^{k}\left(\hat{\omega}_{(j)}^2 - \frac{\lambda(1-R^2)}{n}\right)$$

임계값: $\hat{\omega}_c^2 = \frac{\lambda(1-R^2)}{n}$

**[기타 모델 관계]** (p. 8, Table 2):

$$\boldsymbol{b}_\text{std} = \boldsymbol{P}^{-1/2}\boldsymbol{\omega} \quad \Leftrightarrow \quad \boldsymbol{\omega} = \boldsymbol{P}^{1/2}\boldsymbol{b}_\text{std}$$

$$\boldsymbol{P}_{XY} = \boldsymbol{P}^{1/2}\boldsymbol{\omega} \quad \Leftrightarrow \quad \boldsymbol{\omega} = \boldsymbol{P}^{-1/2}\boldsymbol{P}_{XY}$$

**[귀무분포]** (p. 8, Section 4.2):

$$f(\hat{\omega}_j) = |\hat{\omega}_j|\,\text{Beta}\!\left(\hat{\omega}_j^2;\,\frac{1}{2},\,\frac{\kappa-1}{2}\right), \quad \kappa = n-1$$

### 모델 구조

```
[Step 1] 데이터 표준화: X_std, Y_std
       ↓
[Step 2] 상관행렬 추정
         - 대표본(n >> d): 경험적 추정 → R
         - 소표본(n < d): 수축(shrinkage) 추정 → R̂ (James-Stein type)
       ↓
[Step 3] CAR 점수 계산: ω̂ = R̂^{-1/2} * r_{XY}
       ↓
[Step 4] 변수 순위 결정: |ω̂_j| 내림차순
       ↓
[Step 5] 임계값으로 변수 선택
         - 고정 임계값: AIC/BIC 기반 ω̂_c^2
         - 적응적 임계값: FNDR 제어 또는 교차검증
       ↓
[Step 6] 선택된 변수로 OLS 회귀
```

### 성능 향상

| 비교 방법 | CAR 성능 우위 | 설명 |
|-----------|--------------|------|
| Lasso | 대부분 시나리오에서 우수 | 더 높은 TP, 낮은 FP |
| Boosting | 대부분 시나리오에서 우수 | Table 5–6 |
| OLS (no selection) | 크게 우수 | 불필요 변수 제거 |
| Elastic Net | 동등 수준 (일부 열세) | Example 2, σ=6 제외 |
| Partial Correlation | 대부분 우수 | Table 5–6 |

**유전자 발현 데이터 (Table 9, p. 21):**

| 모델 | 크기 | CV 예측 오차 |
|------|------|-------------|
| Lasso | 36 | 0.4006 (0.0011) |
| Elastic Net | 85 | 0.3417 (0.0068) |
| **CAR** | **36** | **0.3357 (0.0070)** |
| **CAR** | **60** | **0.3049 (0.0064)** |
| **CAR** | **85** | **0.2960 (0.0059)** |

### 한계

1. **선형 모델 전용**: 비선형 관계에는 직접 적용 불가
2. **연속형 반응변수 한정**: 분류 문제에는 CAT 점수 사용 권고
3. **계산 복잡성**: 초고차원($d > 10000$)에서 $\boldsymbol{P}^{-1/2}$ 계산 비용 증가
4. **"Proper exclusion" 위반**: 귀무변수($b_j = 0$)가 비귀무변수와 상관 시 $\phi^\text{CAR}(X_j) \neq 0$ (p. 18)
5. **사전 스크리닝 필요**: 초고차원 데이터에서는 사전 FNDR 필터링 단계 필요 (p. 21)

---

## 3. 각 주장에 페이지/Figure/Table 번호 표시

| 주장 | 위치 |
|------|------|
| CAR 점수 정의 $\boldsymbol{\omega} = \boldsymbol{P}^{-1/2}\boldsymbol{P}_{XY}$ | p. 8, Section 4.1, Eq. (6) |
| $\Omega^2 = \sum_{j=1}^d \omega_j^2$ 분해 | p. 10, Section 4.6 |
| 그룹화 성질 수식 | p. 12, Section 4.9 |
| 대립 변수 하위 순위 | p. 12–13, Section 4.9 |
| 귀무분포 $f(\hat{\omega}_j)$ | p. 8, Section 4.2 |
| AIC/BIC 임계값 연결 | p. 11–12, Table 4, Section 4.8 |
| CAR vs 다른 기준 관계 | p. 8, Table 2 |
| CAT vs CAR 비교 | p. 9, Table 3 |
| 대표본 시뮬레이션 결과 | p. 15, Table 5 |
| 소표본 시뮬레이션 결과 | p. 16, Table 6 |
| 회귀계수 분포 시각화 | p. 17, Figure 1 |
| 당뇨병 데이터 변수 순위 | p. 19, Table 8 |
| 당뇨병 CAR 회귀 경로 | p. 20, Figure 2 |
| 유전자 발현 CV 오차 | p. 21, Table 9 |
| 유전자 발현 모델 크기 vs 오차 | p. 22, Figure 3 |
| 예측된 전략 제안 | p. 23, Section 6 (결론) |

---

## 4. 저자 보고 결과 vs. 본인 해석 분리

### 4-1. 저자가 직접 보고한 결과

**연구 주제:**
고차원 선형 회귀에서 예측변수 간 상관을 조정하는 새로운 변수 중요도 기준 CAR 점수 제안

**방법 (저자 보고):**
- CAR 점수: $\boldsymbol{\omega} = \boldsymbol{P}^{-1/2}\boldsymbol{P}_{XY}$ (p. 8)
- 대표본: 경험적 추정, 소표본: shrinkage 추정 (Schäfer & Strimmer, 2005)
- 변수 선택: CAR 점수 제곱 임계값 처리

**결과 (저자 보고):**
- 시뮬레이션 Example 1, $n=50$, $\sigma=1$: CAR 상대 모델오차 107 vs Elastic Net 135 (Table 5, p. 15)
- 시뮬레이션 Example 3, $n=100$, $\sigma=3$: CAR 172 vs Elastic Net 488 (Table 6, p. 16)
- 유전자 발현: CAR(85)의 CV 예측 오차 0.2960 vs Elastic Net(85) 0.3417 (Table 9, p. 21)
- 당뇨병 데이터: CAR은 대립 변수 s1, s2를 마지막 순위로 배치 (Figure 2, p. 20)

### 4-2. 본인의 해석 및 평가

> ⚠️ **아래는 저자의 직접 주장이 아닌 분석자의 해석입니다.**

1. **이론적 우아함 vs. 실용성**: CAR 점수의 수학적 구조는 매우 정교하나, 실제 유전체 데이터($d > 20000$)에서 $\boldsymbol{P}^{-1/2}$ 계산이 계산 병목이 될 수 있음. 저자가 이를 인식하고 있으나 구체적 해법은 미완성 상태임.

2. **사전 스크리닝 의존성**: 유전자 발현 분석에서 11,940개 → 403개로 사전 필터링 후 CAR 적용. 이 사전 스크리닝이 최종 성능에 미치는 영향이 독립적으로 평가되지 않았음.

3. **proper exclusion 위반에 대한 합리화**: 저자는 귀무변수가 비귀무변수와 상관될 때 $\phi^\text{CAR} \neq 0$인 것이 "예측 관점에서 합리적"이라고 주장하나, 이는 **인과 추론** 맥락에서는 잘못된 변수 선택을 야기할 수 있음.

4. **200회 반복 시뮬레이션**: 통계적으로 충분하나, 시뮬레이션 시나리오가 정규분포 가정에만 한정되어 있어 비정규 데이터에서의 일반화 가능성을 검증하지 못함.

---

## 5. 통계적으로 취약한 부분과 비교 불가능한 수치

| 항목 | 문제점 | 위치 |
|------|--------|------|
| ⚠️ 시뮬레이션 데이터 정규분포 한정 | 모든 시뮬레이션이 정규분포 가정. 비정규 데이터에서 CAR 점수의 귀무분포 유효성 미검증 | p. 14, Section 5.2 |
| ⚠️ 검증 세트 = 훈련 세트 크기 동일 | 튜닝 파라미터 최적화에 훈련 세트와 동일 크기의 독립 검증 세트 사용 → 실제 환경과 다를 수 있음 | p. 14, Section 5.2 |
| ⚠️ Table 9의 CV 오차 비교 | Lasso(36)와 CAR(36)은 같은 모델 크기지만 변수 선택 과정이 다름 → 공정 비교 가능하나 선택된 변수 집합이 다름을 명시 안 함 | p. 21, Table 9 |
| ⚠️ Example 2, σ=6 (n=100): Elastic Net이 우세 | CAR 64 vs Elastic Net 53. 저자는 이를 인정하나 원인 분석 부재 | p. 15, Table 5 |
| ⚠️ 유전자 발현 사전 스크리닝 편향 | FNDR<0.2 기준으로 11,940→403 필터링 시, 이 과정이 CV 루프 밖에서 이루어지면 낙관 편향 발생 가능. 저자는 CV 내 재계산을 명시했으나 사전 스크리닝은 CV 밖 | p. 21, Section 5.5 |
| ⚠️ Genizi 방법의 OLS 연계 | Genizi 방법도 CAR과 유사한 성능이나, 왜 CAR을 선호하는지 통계적 기준 부재 | p. 14–16, Tables 5–6 |
| ⚠️ n=10, 20 시나리오: OLS/PCOR/Genizi 결과 없음 | d=40에서 n=10, 20일 때 OLS 등은 계산 불가로 비교 불가 ("—" 표시) | p. 16, Table 6 |

---

## 6. 문서가 답하지 않는 질문

1. **비선형 확장**: CAR 점수를 비선형 회귀(kernel 방법, 신경망 등)에 적용하는 방법은?

2. **다변량 반응변수**: 반응변수가 다변량($Y$가 벡터)인 경우 CAR 점수 확장 가능성?

3. **인과 추론**: CAR 점수는 예측 목적에 최적화되었는데, 인과 변수 식별(causal variable identification)에 적합한가?

4. **시간적 데이터**: 시계열 데이터나 종단 데이터(longitudinal data)에서의 적용 방법?

5. **결측 데이터**: 결측값이 있는 데이터에서 상관행렬 추정 시 CAR 점수의 안정성?

6. **초고차원 ($d > 100,000$)**: 유전체 분석에서 $d = 100,000+$일 때 $\boldsymbol{P}^{-1/2}$ 계산의 현실적 방법론?

7. **비정규 데이터**: 비정규 분포(포아송, 음이항 등) 데이터에서 귀무분포 $f(\hat{\omega}_j)$의 유효성?

8. **변수 선택 일치성(selection consistency)**: 유한 표본에서 CAR 점수가 진짜 모델을 복원할 조건은?

9. **표본 크기 결정**: 원하는 검정력을 달성하기 위한 최소 표본 크기 공식?

10. **다중 반응변수와 예측변수 간 네트워크**: CAR 점수를 그래픽 모델(graphical model) 추론에 통합하는 방법?

---

## 7. 가장 중요한 그림 5개 해석

### Figure 1 (p. 17): 회귀계수 분포 비교 (Example 3, n=50, σ=3)

**내용:** 7가지 방법(Shrinkage CAR, Elastic Net, Lasso, Boost, OLS, PCOR, Genizi)의 200회 시뮬레이션에서 추정된 회귀계수 분포를 박스플롯으로 비교.

**해석:**
- **Shrinkage CAR**: $b_1$ ~ $b_5$(양의 진짜 계수)와 $b_6$ ~ $b_{10}$(음의 진짜 계수)를 모두 올바른 부호로 복원. 변동성이 작음.
- **Elastic Net / Lasso / Boost**: $b_6$ ~ $b_{10}$(음의 계수 변수들)의 부호를 제대로 복원하지 못하고 0으로 수축시키는 경향.
- **OLS, PCOR, Genizi**: 척도(scale)가 크게 달라 추정 불안정성이 심함 (소표본 고차원 문제의 전형적 현상).
- **결론**: CAR 점수는 소표본에서 음의 부호를 가진 변수도 정확하게 선택하는 데 유리함.

---

### Figure 2 (p. 20): 당뇨병 데이터 CAR 회귀 경로

**내용:** 당뇨병 데이터에서 CAR 점수 순서대로 변수를 추가할 때의 OLS 회귀계수 변화.

**해석:**
- **s5(혈청 5)**: 첫 번째로 포함되는 가장 중요한 변수로, 회귀계수가 ~0.6으로 높음.
- **bmi, bp**: 초기부터 안정적으로 포함되며 당뇨 진행과의 강한 연관성을 반영.
- **s1(음), s2(양)**: 가장 마지막에 포함. 이는 두 변수가 높은 양의 상관(positively correlated)을 가지면서 반대 부호의 회귀계수를 가지는 **대립 변수(antagonistic)**이기 때문에 CAR 점수가 매우 낮아짐 → CAR 점수의 핵심 특성 시연.
- **결론**: CAR 점수는 의학적으로 서로 상쇄되는 변수를 자연스럽게 마지막 순위로 분류하여 파싱모니어스한 모델을 구성.

---

### Figure 3 (p. 22): 유전자 발현 데이터의 CAR 모델 크기별 CV 예측 오차

**내용:** 유전자 발현 데이터(n=30, d=403)에서 CAR 모델의 포함 변수 수에 따른 5-fold CV 예측 오차 박스플롯.

**해석:**
- 예측 오차는 포함 변수가 10→60개로 증가할수록 지속적으로 감소.
- **최적 모델 크기: 약 60개 예측변수** (저자 직접 보고).
- 60개 이상을 포함해도 예측 오차 개선이 미미하여 과적합의 시작을 시사.
- 이는 CAR 점수 기반 변수 순위가 실제 예측 유용성과 정렬되어 있음을 보여줌.
- **비교**: 동일 크기(36개)에서 CAR(0.3357) < Elastic Net에 해당하는 성능 수준을 보임(Table 9).

---

### Table 5 (p. 15): 대표본 시뮬레이션 결과 (Examples 1 & 2)

**내용:** d=8, n=50 및 n=100에서 7가지 방법의 평균 상대 모델오차 및 TP+FP.

**해석:**
- **Example 1 (약한 상관, ρ=0.5)**: CAR(107)이 Elastic Net(135), Lasso(132), Boost(390), OLS(217)를 모두 능가하며 최저 모델 오차 달성 (n=50, σ=1).
- **Example 2 (강한 상관, ρ=0.85)**: n=100, σ=6에서 CAR(64)가 Elastic Net(53)에 열세. 강한 상관 + 높은 노이즈 조합에서 CAR의 성능 한계를 보여주는 유일한 경우.
- **FP(위양성) 수**: OLS는 항상 5.0개의 위양성(모든 변수 포함), CAR은 1.0~1.7개로 가장 낮은 수준 유지.
- **결론**: CAR은 대부분 조건에서 가장 효율적인 변수 선택을 수행하며, 강한 상관+고노이즈 조합에서만 elastic net이 소폭 우세.

---

### Table 6 (p. 16): 소표본 시뮬레이션 결과 (Examples 3 & 4)

**내용:** d=40, n=10~100에서 7가지 방법의 평균 상대 모델오차 및 TP+FP.

**해석:**
- **Example 3 (n=10, σ=3)**: CAR(1482) vs Elastic Net(1501) vs Lasso(1905) vs Boost(2203). OLS/PCOR/Genizi는 d>n으로 계산 불가("—"). **CAR이 최저 모델 오차**.
- **Example 3 (n=100, σ=3)**: CAR(172)가 Elastic Net(488)을 크게 능가. TP=9.5로 거의 완전한 진짜 변수 선택.
- **Example 4 (n=100, σ=6)**: CAR(87) vs Elastic Net(107) vs Lasso(112). FP: CAR(1.2)로 Elastic Net(2.9), Lasso(2.8)보다 월등히 낮음.
- **결론**: 소표본 고차원 설정에서 shrinkage CAR 추정의 효과가 두드러지며, elastic net 대비 압도적 우위.

---

## 8. 결론, 시사점, 후속 연구

### 8-1. 저자 제시 시사점 (pp. 22–23)

저자들은 다음의 **3단계 고차원 데이터 분석 전략**을 제안한다:

1. 주변 상관(또는 t-점수)과 FNDR 기준으로 변수 **사전 스크리닝**
2. 남은 변수들을 **CAR 점수 제곱 기준으로 순위 결정**
3. (선택적) 변수를 그룹화하여 **그룹 CAR 점수 계산**

**저자 제시 후속 연구 방향:**
- 초고차원 예측변수에서 CAR/CAT 점수의 shrinkage 추정을 위한 **알고리즘 개선** → 사전 스크리닝 단계를 불필요하게 만드는 것이 목표

---

### 8-1. 모델의 일반화 성능 향상 가능성 (심층 분석)

CAR 점수 기반 방법의 일반화 성능 향상 가능성은 다음 방향에서 탐구될 수 있다:

#### (1) 추정기(Estimator) 개선

현재 소표본에서 사용하는 shrinkage 추정기를 더 발전된 형태로 대체:

$$\hat{\boldsymbol{P}}^{-1/2}_\text{shrink} = \left[(1-\alpha)\hat{\boldsymbol{P}} + \alpha \boldsymbol{I}\right]^{-1/2}$$

- **그래픽 LASSO 기반 정밀행렬 추정**: $\hat{\boldsymbol{P}}^{-1}$을 희소 추정하여 $\hat{\boldsymbol{P}}^{-1/2}$ 계산 가능성
- **랜덤화 SVD**: 초고차원에서 $\boldsymbol{P}^{-1/2}$의 근사 계산

#### (2) 앙상블 CAR

여러 부트스트랩 샘플에서 CAR 점수를 계산하고 집계:

$$\bar{\omega}_j = \frac{1}{B}\sum_{b=1}^{B}\hat{\omega}_j^{(b)}$$

이는 추정 분산을 줄여 일반화 성능 향상 기대 (Random LASSO와 유사한 전략, Wang et al., 2011)

#### (3) 비선형 확장을 통한 일반화

커널 방법을 통해 비선형 CAR 점수 도입 가능:

$$\boldsymbol{\omega}_\text{kernel} = \boldsymbol{K}^{-1/2}\boldsymbol{k}_{XY}$$

여기서 $\boldsymbol{K}$는 커널 행렬. 이는 비선형 관계에서의 일반화 성능 향상 가능.

#### (4) 교차검증 기반 임계값 최적화

현재 AIC/BIC 기반 고정 임계값 대신 nested CV로 최적 임계값 결정:

$$\hat{\omega}_c^{*2} = \arg\min_{\omega_c^2} \text{CV-Error}(\omega_c^2)$$

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의**: 아래 비교는 2020년 이후 발표된 관련 연구 분야의 일반적 동향을 기반으로 한 분석입니다. 특정 논문과의 직접 비교는 해당 논문을 직접 확인하지 않은 부분이 있으므로, 구체적 수치 비교는 해당 논문을 직접 참조하시기 바랍니다.

#### 관련 최신 연구 동향

**[1] Tilted Correlation (Cho & Fryzlewicz, 2011 → 이후 발전)**
- 저자들이 직접 언급한 관련 방법 (p. 23)
- CAR 점수와 같은 목표(상관 조정 변수 중요도)이나 다른 수학적 경로
- Cho & Fryzlewicz의 후속 연구들에서 이론적 성질이 더 발전됨

**[2] SLOPE (Bogdan et al., 2015), AdaLASSO, SCAD 등의 발전**
- 이들 방법은 변수 선택의 통계적 일치성(consistency) 이론을 강화
- **CAR 점수와의 비교점**: CAR 점수는 유한 표본에서의 선택 일치성 이론이 상대적으로 미흡

**[3] 딥러닝 기반 변수 중요도 (2020년 이후)**
- SHAP(Shapley Additive Explanations), LIME 등 설명 가능 AI 방법
- **CAR 점수와의 비교점**:
  - CAR: 선형 모델 전용, 해석적 수식 존재, 계산 효율적
  - SHAP: 비선형 모델 적용 가능, 게임이론적 정당성, 계산 비용 높음
  - **CAR의 강점**: 이론적 명확성, $\Omega^2$ 분해, 귀무분포 활용 가능

**[4] 네트워크 기반 정규화 (2020년 이후)**
- Graph Neural Network 기반 변수 선택 (예: GNFS 계열)
- **CAR 점수와의 연결 가능성**: 예측변수 간 네트워크 구조를 $\boldsymbol{P}$에 반영하면 네트워크 기반 CAR 점수 구성 가능

**[5] 스파스 학습의 이론 발전**
- 압축 센싱(compressed sensing) 이론의 발전으로 RIP(Restricted Isometry Property) 조건 하에서의 변수 선택 보장
- **CAR 점수**: 이런 이론적 틀에서의 분석이 아직 이루어지지 않음

#### 2020년 이후 연구에 미치는 영향

| 영향 영역 | 구체적 내용 |
|-----------|------------|
| 유전체학 변수 선택 | CAR 점수의 shrinkage 추정은 scRNA-seq 등 초고차원 데이터에 활용 가능한 기반 제공 |
| 설명 가능 AI (XAI) | 선형 모델에서 $\Omega^2$ 분해를 통한 변수 중요도가 SHAP의 선형 버전과 이론적 연결 |
| 다중 오믹스 통합 | 그룹 CAR 점수를 오믹스 레이어 간 통합 지표로 활용 가능 |
| 재현성 위기 대응 | 귀무분포 기반 p-값 계산으로 명확한 통계적 추론 제공 |

#### 앞으로 연구 시 고려할 점

1. **비정규성 처리**: 생물정보학 데이터의 경우 음이항 분포, 과분산 등 처리를 위한 일반화 CAR 점수 (예: GLM 프레임워크 내 CAR 확장)

2. **선택 일치성 이론 강화**: $n, d \to \infty$ 조건에서 CAR 점수 기반 변수 선택의 이론적 보장 (minimax optimal rates 등)

3. **계산 확장성**: 분산 컴퓨팅 환경에서 $\boldsymbol{P}^{-1/2}$ 계산을 위한 근사 알고리즘 개발

4. **인과 추론 통합**: CAR 점수의 예측 지향성을 넘어 인과 효과 추정과의 연결 방법론

5. **전이학습(transfer learning) 맥락**: 소표본 문제에서 사전 학습된 상관 구조를 활용한 CAR 점수 추정

6. **다중 반응변수 확장**: $Y$가 벡터인 경우의 다변량 CAR 점수 정의 및 이론

7. **강건성 분석**: 이상치(outlier)나 오염된 데이터에서 CAR 점수의 안정성 및 강건 추정법 개발

---

## 참고 자료 목록

**본 분석에서 직접 인용한 논문 내 참고문헌:**

1. Zuber, V. and Strimmer, K. (2011). "High-Dimensional Regression and Variable Selection Using CAR Scores." *Statistical Applications in Genetics and Molecular Biology*, 10: 34. (분석 대상 논문, arXiv:1007.5516v6)
2. Zou, H. and Hastie, T. (2005). "Regularization and variable selection via the elastic net." *J. R. Statist. Soc. B*, 67:301–320.
3. Tibshirani, R. (1996). "Regression shrinkage and selection via the lasso." *J. R. Statist. Soc. B*, 58:267–288.
4. Schäfer, J. and Strimmer, K. (2005). "A shrinkage approach to large-scale covariance matrix estimation." *Statist. Appl. Genet. Mol. Biol.*, 4:32.
5. Efron, B., Hastie, T., Johnstone, I., and Tibshirani, R. (2004). "Least angle regression." *Ann. Statist.*, 32:407–499.
6. Fan, J. and Lv, J. (2008). "Sure independence screening for ultra-high dimensional feature space." *J. R. Statist. Soc. B*, 70:849–911.
7. Genizi, A. (1993). "Decomposition of $R^2$ in multiple regression with correlated regressors." *Statistica Sinica*, 3:407–420.
8. Ahdesmäki, M. and Strimmer, K. (2010). "Feature selection in omics prediction problems using cat scores and false non-discovery rate control." *Ann. Appl. Statist.*, 4:503–519.
9. Witten, D. M. and Tibshirani, R. (2009). "Covariance-regularized regression and classification for high-dimensional problems." *J. R. Statist. Soc. B*, 71:615–636.
10. Lu, T. et al. (2004). "Gene regulation and DNA damage in the ageing human brain." *Nature*, 429:883–891.
11. Wang, S., Nan, B., Rosset, S., and Zhu, J. (2011). "Random lasso." *Ann. Applied Statistics*, 5:468–485.
12. Grömping, U. (2007). "Estimators of relative importance in linear regression based on variance decomposition." *The American Statistician*, 61:139–147.
13. Cho, H. and Fryzlewicz, P. (2011). "High-dimensional variable selection via tilting." Preprint.
14. Zuber, V. and Strimmer, K. (2009). "Gene ranking and biomarker discovery under correlation." *Bioinformatics*, 25:2700–2707.
15. Hothorn, T. and Bühlmann, P. (2006). "Model-based boosting in high dimensions." *Bioinformatics*, 22:2828–2829.
