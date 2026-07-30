# Estimators of Relative Importance in Linear Regression Based on Variance Decomposition

---

## 1. Executive Summary (10문장 이내)

이 논문은 선형 회귀에서 각 회귀변수의 **상대적 중요도(Relative Importance)**를 분산 분해(Variance Decomposition)를 통해 정량화하는 방법론을 체계적으로 정리하고 비교한다. 관측 데이터 기반 연구에서 회귀변수들이 상관되어 있을 때, 표준 회귀 출력만으로는 각 변수의 기여도를 명확히 분리하기 어렵다는 문제에서 출발한다. 논문은 두 가지 핵심 추정량인 **LMG**(Lindeman, Merenda & Gold, 1980)와 **PMVD**(Proportional Marginal Variance Decomposition; Feldman, 2005)를 중심으로 이론적 성질을 분석한다. LMG는 회귀변수의 모든 순열에 걸쳐 순차적 설명분산의 평균을 취하며, Shapley Value와 수학적으로 동치임을 보인다. PMVD는 데이터 의존적 가중치를 활용한 LMG의 가중 평균 버전으로, 비례적 가치(Proportional Value)의 게임이론적 인스턴스이다. 논문은 적절한 분해(a), 비음성(b), 배제(c), 포함(d)의 네 가지 바람직성 기준을 제시하고 각 방법의 충족 여부를 평가한다. 시뮬레이션 연구를 통해 PMVD가 LMG보다 추정량의 변동성이 전반적으로 높음을 실증적으로 보인다. 인과적 해석이 목적일 경우 LMG가 선호되고, 배제 기준이 필수적인 경우 PMVD를 사용해야 한다. 이 논문은 분산된 관련 문헌들을 통합하고, 두 추정량의 통계적 특성에 대한 이해를 제고하는 데 기여한다.

---

### 1-1. 연구의 목적과 필요성

**목적:**
- 분산된 상대적 중요도 관련 문헌을 통합(Reconcile)
- LMG와 PMVD의 이론적·경험적 특성을 체계적으로 비교

**필요성:**
- 관측 데이터(심리학, 생물학, 경제학 등)에서 회귀변수 간 상관이 일반적이며, 이 경우 표준 회귀 출력($t$ 통계량, Type III SS)은 개별 기여도 분해에 적합하지 않음 (p.139)
- $R^2$ 분해에 대한 실용적 수요가 높으나, 분야별로 방법론이 파편화되어 있고 동일 방법이 여러 이름으로 재발명되는 문제 발생 (p.140)
- 기존 방법(Hoffman 1960)은 음수 기여를 허용하는 문제점이 있으며, 이를 해결할 적절한 대안 방법론의 체계적 평가가 부재

---

## 2. 핵심 주장과 근거 표

| 주장 | 근거 | 위치 |
|------|------|-------|
| LMG는 모든 회귀변수 순열에 대한 순차적 설명분산의 단순 평균으로, Shapley Value와 동치 | 수식 (7)에서 $\frac{1}{p!}\sum_{S \subseteq \{2,...,p\}} n(S)!(p-n(S)-1)! \cdot \text{svar}(\{1\} \mid S)$로 유도 | p.142, 식(7) |
| PMVD는 데이터 의존적 가중치를 사용하는 LMG의 가중 평균 | 식 (8)~(9): 가중치 $p(r) \propto L(r)$이며, 데이터에 의해 결정됨 | p.142, 식(8)(9) |
| LMG는 배제(Exclusion) 기준을 위반하나 다른 세 기준은 충족 | $\beta_1=0, \beta_2 \neq 0, \rho_{12} \neq 0$이면 식(6)의 세 번째 항이 0이 아님 | p.142, 식(6) |
| PMVD는 네 가지 바람직성 기준을 모두 충족 | $\beta_j=0$이면 0-계수 변수를 마지막에 배치하는 순열에만 양의 가중치 부여 | p.142, Feldman(2005) |
| 시뮬레이션에서 PMVD가 양의 상관 구조에서 LMG보다 변동성이 현저히 높음 | Figure 3: IQR 비교, 500회 반복 시뮬레이션 결과 | p.145, Figure 3 |
| 인과적 해석 맥락에서 LMG의 균등화 성질은 모델 불확실성의 자연스러운 반영 | Figure 1의 두 인과 모델 사례: 동일 회귀모델이 상이한 인과구조 허용 | p.141, Figure 1 |
| 두 방법은 무상관이거나 동일 $\beta \sqrt{v_j}$를 가진 equi-correlated 상황에서 일치 | 대칭성과 LMG 균등가중, PMVD의 익명성 공리(Anonymity Axiom)로 증명 | p.143 |

---

### 2-1. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

#### 해결하고자 하는 문제

회귀변수 간 상관이 존재할 때, 전체 모형 $R^2$를 각 변수의 기여분으로 분해하는 문제:

$$
\text{var}(Y) = \sum_{j=1}^{p} \beta_j^2 v_j + 2\sum_{j=1}^{p-1}\sum_{k=j+1}^{p} \beta_j \beta_k \sqrt{v_j v_k} \rho_{jk} + \sigma^2 \quad \text{(식 2, p.140)}
$$

상관이 없으면 $\beta_j^2 v_j$로 유일하게 분해되나, 상관이 있으면 혼합항(cross-term)을 어떻게 배분할지가 핵심 문제이다.

#### 제안하는 방법과 수식

**기본 표기 정의 (p.140):**

$$
\text{evar}(S) = \text{var}(Y) - \text{var}(Y | X_j, j \in S) \quad \text{(식 3)}
$$

$$
\text{svar}(M | S) = \text{evar}(M \cup S) - \text{evar}(S) \quad \text{(식 4)}
$$

**[방법 1] LMG 추정량 (p.142, 식 7):**

$$
\text{LMG}(1) = \frac{1}{p!} \sum_{r \in \text{permutation}} \text{svar}(\{1\} | S_1(r))
= \frac{1}{p!} \sum_{S \subseteq \{2,...,p\}} n(S)!(p - n(S) - 1)! \cdot \text{svar}(\{1\} | S)
$$

직관적으로는 모델 크기 $i$에 걸친 평균 (식 7*):

$$
\text{LMG}(1) = \frac{1}{p} \sum_{i=0}^{p-1} \left( \sum_{\substack{S \subseteq \{2,...,p\} \\ n(S)=i}} \text{svar}(\{1\}|S) \bigg/ \binom{p-1}{i} \right)
$$

**[방법 2] PMVD 추정량 (p.142, 식 8, 9):**

$$
\text{PMVD}(1) = \sum_{r \in \text{permutation}} p(r) \cdot \text{svar}(\{1\} | S_1(r))
$$

가중치 $p(r)$:

$$
L(r) = \prod_{i=1}^{p-1} \left[\text{svar}\left(\{r_{i+1},...,r_p\} | \{r_1,...,r_i\}\right)\right]^{-1}
= \prod_{i=1}^{p-1} \left(\text{evar}(\{1,...,p\}) - \text{evar}(\{r_1,...,r_i\})\right)^{-1}
$$

$$
p(r) = \frac{L(r)}{\sum_{r'} L(r')}
$$

**2개 회귀변수의 경우 PMVD 명시적 결과 (식 10, p.142):**

$$
\text{PMVD 배분}(X_1) = \beta_1^2 v_1 + \frac{\beta_1^2 v_1}{\beta_1^2 v_1 + \beta_2^2 v_2} \cdot 2\beta_1\beta_2\sqrt{v_1 v_2}\rho_{12}
$$

#### 모델 구조

선형 회귀 모형 (식 1, p.140):

$$
Y = \beta_0 + X_1\beta_1 + \cdots + X_p\beta_p + \varepsilon
$$

- $E(\varepsilon) = 0$, $\text{var}(\varepsilon) = \sigma^2$, 회귀변수와 무상관
- 회귀변수 $X_j$는 랜덤 변수 (random regressor model)
- 공분산 행렬은 양정치(positive definite) 가정

#### 성능 향상 및 한계

| 구분 | LMG | PMVD |
|------|-----|------|
| 계산 복잡도 | $2^{p-1}$ 부분집합 계산 | $p!$ 순열 계산 필요(더 복잡) |
| 바람직성 기준 (a)(b)(d) | 충족 | 충족 |
| 배제 기준 (c) | **위반** (상관 있을 때) | 충족 |
| 추정량 변동성 | 상대적으로 낮음 | 양의 상관에서 현저히 높음 |
| 인과적 해석 | 모델 불확실성 반영하여 적합 | 계수 크기 쏠림 위험 |
| 극단적 상관(→1)에서의 수렴 | 균등 배분으로 수렴(연속성) | 계수에 의존적 수렴(비연속) |
| 부트스트랩 CI | 다소 반보수적(anti-conservative) | 미보고 |

---

## 3. 각 주장에 페이지 또는 Figure/Table 번호 표시

| 주장 | 위치 |
|------|------|
| 상대적 중요도 문제는 관측 데이터에서 광범위하게 요구됨 | p.139, Introduction |
| 네 가지 바람직성 기준 (a)~(d) 정의 | p.141, Section 2.2 |
| LMG 추정량 공식 | p.142, 식 (7), (7*) |
| PMVD 가중치 공식 | p.142, 식 (8), (9) |
| 2변수 LMG 배분 결과 (식 6) | p.142, 식 (6) |
| 2변수 PMVD 배분 결과 (식 10) | p.142, 식 (10) |
| Table 1: PMVD 가중치 예시 | p.143, Table 1 |
| Figure 2: LMG vs PMVD 비교 시나리오 | p.144, Figure 2 |
| 시뮬레이션 설정 | p.144, Table 2 |
| Figure 3: IQR 비교(LMG vs PMVD) | p.145, Figure 3 |
| 인과적 해석에서 배제 기준의 불합리성 | p.141, Figure 1 |
| 부트스트랩 CI의 anti-conservative 문제 | p.146, Section 5 |
| 전반적 결론 및 권고 | p.145~146, Section 5 |

---

## 4. 저자 직접 보고 결과 vs. 검토자 해석 분리

### 저자가 직접 보고한 결과

**연구 주제 (저자 기술):**
> "This article serves two purposes: (i) to reunite the relative-importance related aspects of the literature from the various fields and (ii) to investigate the statistical properties of the key competitors that decompose the model $R^2$." (p.140)

**방법 (저자 제시 수식):**

LMG (식 7):
$$\text{LMG}(1) = \frac{1}{p!}\sum_{S \subseteq \{2,...,p\}} n(S)!(p-n(S)-1)!\cdot\text{svar}(\{1\}|S)$$

PMVD (식 8):
$$\text{PMVD}(1) = \sum_{r} p(r)\cdot\text{svar}(\{1\}|S_1(r))$$

**결과 (저자 직접 기술):**
- "PMVD is distinctly more variable than LMG for positive correlations" (p.145)
- "Overall, since variability differences in favor of PMVD are typically much smaller than those in favor of LMG, LMG is preferable in terms of variation." (p.145)
- "bootstrap percentile confidence intervals for LMG have shown a somewhat anti-conservative behavior, with error levels up to about twice the nominal in some situations." (p.146)
- "LMG and PMVD coincide for uncorrelated regressors" (p.143)

---

### 검토자(본 분석자)의 해석

- LMG가 Shapley Value와 동치라는 사실은 게임이론적 공정성(fairness) 공리—효율성, 대칭성, 더미공리, 가산성—를 만족함을 시사하며, 이는 LMG를 단순 휴리스틱이 아닌 공리론적으로 정당화된 방법으로 볼 수 있다는 의미이다.
- PMVD의 높은 변동성은 데이터 의존적 가중치가 소표본에서 불안정할 수 있음을 시사하며, 실용적 적용에서 PMVD의 이론적 우수성(배제 기준 충족)이 통계적 효율성으로 연결되지 않을 수 있다.
- 저자는 배제 기준의 불필요성을 인과적 관점에서만 논증하였으나, 예측 목적의 응용에서는 배제 기준 충족이 오히려 중요할 수 있어, 사용 목적에 따른 방법 선택 가이드라인이 추가로 필요하다.
- 시뮬레이션이 4개 회귀변수, 500회 반복, 두 가지 분포(정규/지수)로 제한되어, 결과의 일반화 가능성에 대한 추가 검증이 필요하다.

---

## 5. 통계적으로 취약한 부분과 비교 불가능한 수치

| 항목 | 취약성/비교 불가 이유 |
|------|----------------------|
| ⚠️ 시뮬레이션 반복 횟수 500회 | 정확한 분포 꼬리 추정에 부족할 수 있음; IQR 안정성은 보장하나 극단값 분포는 불확실 (p.143, Section 4.1) |
| ⚠️ 4개 회귀변수로 제한 | $p > 4$ 상황에서의 일반화 불확실; 고차원 문제에서 $2^{p-1}$ 계산 부담 급증 |
| ⚠️ 부트스트랩 CI 결과 ("not presented here") | 실제 수치 미제시, "error levels up to about twice the nominal"은 구체적 조건 불명확 (p.146) |
| ⚠️ 지수분포 결과 미상세 보고 | "The simulations show no surprises here"로 요약, 구체적 수치 미제시 (p.145) |
| ⚠️ $R^2 = 0.9$ 고 $R^2$ 시나리오 결과 | Figure 3은 $R^2=0.25$, $n=100$ 최악 시나리오만 도시; 다른 조건 결과 불투명 |
| ⚠️ LMG와 PMVD의 통계적 유의성 비교 부재 | 분배 차이가 실질적으로 유의한지에 대한 공식 검정 없음 |
| ⚠️ PMVD 부트스트랩 CI 분석 미수행 | LMG에 대한 부트스트랩 결과만 언급, PMVD와의 직접 비교 불가 |

---

## 6. 문서가 답하지 않는 질문

1. **$p > 4$의 고차원 상황**에서 LMG와 PMVD의 상대적 성능은 어떠한가?
2. **PMVD의 부트스트랩 신뢰구간**의 커버리지 확률은 어떠한가? LMG와 비교 시 어느 쪽이 더 신뢰할 수 있는가?
3. **비선형 회귀모델** 혹은 **일반화선형모델(GLM)**에서 LMG/PMVD 유사 방법론은 어떻게 확장될 수 있는가?
4. **결측치나 이상치**가 있을 때 두 추정량의 강건성(Robustness)은 어느 정도인가?
5. 두 방법 간 배분 차이가 **통계적으로 유의한지** 판단하는 공식적 검정 방법은 무엇인가?
6. **고다중공선성(near-multicollinearity)** 상황에서 두 추정량의 수치적 안정성은?
7. **시계열 데이터** 또는 **종속 관측치**가 있을 때 방법론을 어떻게 적용해야 하는가?
8. **고차원 $p \gg n$** 상황에서의 추정 가능성 및 대안 방법은?

---

## 7. 가장 중요한 그림 5개의 해석

### Figure 1 (p.141) — 두 가지 인과 모델

**내용:** $p=3$ 회귀변수를 포함하는 동일한 선형 회귀모형 $E(Y|X_1,X_2,X_3) = \beta_0 + X_1\beta_1 + X_2\beta_2 + X_3\beta_3$을 유도하는 두 가지 상이한 인과 구조(Model I, Model II)를 방향성 그래프로 도시

**해석:**
- **Model I**: $X_1$이 $X_2$, $X_3$에 직접 인과적 영향을 미치는 구조. 점선 화살표($X_1 \to Y$)를 제거하면 $\beta_1 = 0$이 되지만, $X_1$은 여전히 $X_2, X_3$을 통해 $Y$에 간접 영향 → 배제 기준이 불합리
- **Model II**: $X_2$가 $X_1$, $X_3$에 영향을 미치는 구조. 점선 화살표 제거 시 $\beta_1 = 0$이며 $X_1$의 직접/간접 영향 없음 → 배제 기준이 합리적
- **핵심 시사점:** 동일한 회귀모형이 다양한 인과 구조와 양립 가능하므로, 배제 기준이 항상 타당한 것은 아님. 인과적 목적의 분석에서 LMG의 균등화가 정당화됨.

---

### Figure 2 (p.144) — LMG vs PMVD 이론적 배분 비교

**내용:** 4개 회귀변수, 4가지 시나리오 (a)~(d)에서 $R^2$ 배분 비율을 상관 파라미터 $\rho$ 함수로 도시. 굵은 선=LMG, 가는 선=PMVD

**해석:**
- **(a)** $\beta=(1,1,1,1)^T$, 등상관: LMG=PMVD (완전 일치)
- **(b)** $\beta=(1,1,1,1)^T$, AR(1) 상관: 두 방법 모두 중간 변수($X_2, X_3$)가 양의 $\rho$에서 더 높은 배분 → 유사한 거동
- **(c)** $\beta=(4,1,1,0.3)^T$, 등상관: **LMG는 $|\rho|$ 증가 시 배분이 균등화**되는 반면, **PMVD는 상관에 덜 민감**하고 계수 차이를 더 유지
- **(d)** $\beta=(4,1,1,0.3)^T$, AR(1) 상관: 불균등 계수에서 두 방법 차이가 극대화됨
- **핵심 시사점:** 계수 차이가 크고 상관이 강할수록 방법 선택이 결론에 결정적 영향을 미침

---

### Figure 3 (p.145) — 시뮬레이션 IQR 비교

**내용:** $R^2=0.25$, $n=100$, 정규분포 조건에서 500회 반복의 LMG(굵은 선)와 PMVD(가는 선)의 사분위범위(IQR)를 $\rho = -0.9$ ~ $0.9$에 걸쳐 7가지 $\beta$-벡터별 도시

**해석:**
- **전반적으로** PMVD의 IQR이 양의 상관에서 LMG보다 현저히 크며, 이는 데이터 의존적 가중치의 불안정성을 반영
- **$\beta_3=(4,1,0,0)^T$, $\beta_6=(1,1,1,0)^T$**: 0-계수 변수에 대해 PMVD의 IQR이 매우 낮음 → 배제 기준 충족의 실증적 반영
- **$\beta_4=(1,1,1,1)^T$ (동일 계수)**: 두 방법의 IQR 차이가 최소화
- **검토자 해석:** 소표본($n=100$)에서 PMVD의 이론적 장점(배제 기준)이 통계적 효율성 손실로 상쇄될 수 있음을 강력히 시사

---

### Table 1 (p.143) — PMVD 가중치 예시

**내용:** 3개 회귀변수, 3가지 $\beta$-시나리오, 3가지 $\rho$ 값에서 6개 순열의 PMVD 가중치 $p(r)$ 표시

**해석:**
- $\beta=(1,1,1)^T$: 모든 $\rho$에서 가중치가 $1/6 \approx 0.167$에 근접 → 균등 가중치(LMG와 유사)
- $\beta=(5,4,3)^T$: 큰 계수 변수를 먼저 포함하는 순열(1,2,3)에 가중치 집중(0.257~0.344)
- $\beta=(4,1,0.3)^T$: 순열 (2,3,1), (3,2,1)의 가중치 ≈ 0 → 0에 가까운 계수의 변수가 마지막에 오는 순열 선호, **배제 기준 구현 메커니즘** 가시화
- **핵심 시사점:** PMVD 가중치는 계수 차이가 클수록 특정 순열에 집중되며, 이것이 배제 기준 충족과 높은 변동성 모두를 야기하는 구조적 원인

---

### Table 2 (p.144) — 시뮬레이션 설계

**내용:** 상관 구조 2종, 분포 2종, $\beta$-벡터 7종, 표본 크기 2종, $R^2$ 수준 3종의 완전 교차 시뮬레이션 설계

**해석:**
- **설계 강점:** 다양한 조건 조합($\rho$ 범위, 비정규분포 포함)으로 일반화 가능성 제고
- **설계 한계:** 4개 변수 고정, 500회 반복으로 극단적 시나리오 추정 불안정 가능
- $\beta_6=(1,1,1,0)^T$, $\beta_3=(4,1,0,0)^T$ 포함은 배제 기준 검증을 위해 의도적으로 설계
- **검토자 해석:** 지수분포 조건의 결과가 본문에 상세 보고되지 않아, 비정규성 영향의 정량적 평가가 불완전함

---

## 8. 결론: 저자 제시 시사점, 후속 연구, 추가 방향

### 저자가 제시한 시사점 (p.145~146)

1. **방법 선택 가이드라인:**
   - 배제 기준이 필수적인 경우 → PMVD 사용 (단, 변동성 증가와 구현 복잡성 감수)
   - 인과적 해석이 목적인 경우 → LMG 선호 (모델 불확실성 반영)
   - 실용적으로 두 방법을 모두 적용하여 결과 비교 권장

2. **상대적 중요도의 한계:**
   - $R^2$ 분해는 개입(Intervention) 효과를 직접 예측하지 못함
   - 개입이 회귀변수 상관 구조도 변화시킬 수 있어, 결과 해석에 주의 필요

3. **저자가 언급한 후속 연구 필요 사항:**
   - 부트스트랩 신뢰구간의 커버리지 확률에 대한 심층 연구
   - 분산 분해에 대한 완전한 이해를 위한 추가 이론적 연구
   - 상대적 중요도 추정량의 변동성 보고 표준화

---

### 8-1. 모델의 일반화 성능 향상 가능성

본 논문은 선형 회귀 모형에 특화되어 있으며, 일반화 가능성과 관련하여 다음 관점에서 분석할 수 있다:

**현재 모형의 일반화 제약:**

$$Y = \beta_0 + \sum_{j=1}^p X_j \beta_j + \varepsilon \quad \text{(선형, 가산적 구조 가정)}
$$

이 구조 가정이 충족되지 않을 경우(비선형 효과, 상호작용 등) LMG/PMVD의 분해 결과가 왜곡될 수 있다.

**일반화 방향:**

| 방향 | 내용 | 기대 효과 |
|------|------|-----------|
| GLM 확장 | 로지스틱/포아송 회귀로의 LMG 확장 | 비연속 반응변수 적용 가능 |
| 정규화 회귀 | Ridge/LASSO 계수 기반 분해 | 고차원 $p \gg n$ 문제 해결 |
| 비선형 모형 | 트리 기반 모형의 SHAP value | LMG와 유사한 게임이론적 분해 |
| 혼합효과 모형 | 고정효과/랜덤효과 분해 | 위계적 데이터 처리 |
| 시계열 | 시간 의존적 중요도 추정 | 동적 중요도 파악 |

**핵심 고려사항:** 상관된 회귀변수를 포함한 고차원 모형에서 $2^{p-1}$ 계산 복잡도는 $p > 15$ 이상에서 실용적 한계에 도달하므로, 근사 알고리즘 개발이 일반화의 전제 조건이다.

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의:** 아래 비교는 이 논문의 내용과 AI 연구자로서의 배경 지식을 기반으로 작성되었습니다. 2020년 이후 특정 논문의 세부 수치는 직접 원문 확인을 권고합니다.

#### 2020년 이후 주요 연구 흐름

| 연구 방향 | 주요 내용 | Grömping(2007)과의 관계 |
|-----------|-----------|------------------------|
| **SHAP (Lundberg & Lee, 2017; 이후 확장)** | 비선형 ML 모형에서 Shapley Value 기반 설명 (SHAP값) | LMG = 선형 모형에서의 SHAP과 동치임을 확인; SHAP은 LMG를 ML로 일반화 |
| **Kernel SHAP (Lundberg et al., 2020)** | 임의 모형에 대한 Shapley 기반 근사 | Grömping(2007)의 계산 병목을 우회하는 근사 방법 |
| **상대적 중요도의 인과적 재정의** | Pearl의 do-calculus와 상대적 중요도의 통합 시도 | Figure 1에서 제기된 인과 해석 문제를 공식화 |
| **고차원 확장** | $p \gg n$에서 LASSO 기반 분해 | 저자가 미해결로 남긴 고차원 문제 부분적 해결 |
| **Shapley 기반 FI (Feature Importance)** | 랜덤포레스트/XGBoost에서 Shapley FI | LMG의 게임이론적 기반을 ML로 확장 |

#### 본 논문이 이후 연구에 미치는 영향

1. **SHAP의 이론적 선구:** LMG = 선형 모형 Shapley Value임을 명시적으로 연결한 이 논문은, 이후 Lundberg & Lee(2017)의 SHAP 개발의 이론적 토대 중 하나가 되었다.

2. **R 패키지 `relaimpo`의 지속적 영향:** 저자의 R 패키지는 현재도 광범위하게 사용되며, 이 논문의 방법론이 실용적으로 생존하고 있음을 보여준다.

3. **배제 기준 논쟁의 확장:** SHAP에서도 유사한 논쟁이 제기되며, "correlated features" 문제는 Janzing et al.(2020) 등에서 재조명되었다.

#### 앞으로 연구 시 고려할 점

```
1. 비선형 모형으로의 확장 시 LMG의 Shapley 동치성을 
   활용한 통합 프레임워크 개발

2. 인과 추론(Causal Inference)과 상대적 중요도의 
   공식적 통합 — Figure 1의 문제 해결

3. 고차원($p > 20$) 상황에서 Monte Carlo Shapley 
   근사 방법의 통계적 특성 분석

4. 부트스트랩 CI의 anti-conservative 문제 해결을 위한 
   수정된 신뢰구간 방법론 개발

5. 시간적으로 변동하는 중요도(Time-varying Importance) 
   추정을 위한 동적 확장
```

---

## 참고자료

**주요 참고 논문 (본문 내 인용):**
- Grömping, U. (2007). "Estimators of Relative Importance in Linear Regression Based on Variance Decomposition." *The American Statistician*, 61(2), 139–147. (DOI: 10.1198/000313007X188252)
- Lindeman, R. H., Merenda, P. F., and Gold, R. Z. (1980). *Introduction to Bivariate and Multivariate Analysis*. Scott, Foresman.
- Feldman, B. (2005). "Relative Importance and Value." Unpublished manuscript.
- Budescu, D. V. (1993). "Dominance Analysis." *Psychological Bulletin*, 114, 542–551.
- Azen, R., and Budescu, D. V. (2003). "The Dominance Analysis Approach." *Psychological Methods*, 8, 129–148.
- Shapley, L. (1953). "A Value for n-Person Games."
- Chevan, A., and Sutherland, M. (1991). "Hierarchical Partitioning." *The American Statistician*, 45, 90–96.
- Lipovetsky, S., and Conklin, M. (2001). "Analysis of Regression in Game Theory Approach." *Applied Stochastic Models in Business and Industry*, 17, 319–330.
- Grömping, U. (2006). "Relative Importance for Linear Regression in R: The Package relaimpo." *Journal of Statistical Software*, 17(1).

**2020년 이후 비교 참고 (배경 지식 기반, 원문 확인 필요):**
- Lundberg, S. M., and Lee, S.-I. (2017). "A Unified Approach to Interpreting Model Predictions." *NeurIPS 2017*.
- Lundberg, S. M., et al. (2020). "From Local Explanations to Global Understanding with Explainable AI for Trees." *Nature Machine Intelligence*, 2, 56–67.
- Janzing, D., et al. (2020). "Feature relevance quantification in explainable AI." *AISTATS 2020*.
