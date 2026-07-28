# The Dominance Analysis Approach for Comparing Predictors in Multiple Regression
### *The Dominance Analysis Approach for Comparing Predictors in Multiple Regression*
**출처**: Azen, R., & Budescu, D. V. (2003). *Psychological Methods*, 8(2), 129–148. DOI: 10.1037/1082-989X.8.2.129

---

## 0. 이 분석서의 검증 범위와 한계 (먼저 읽어주세요)

정확도에 대한 요청이 있었으므로, 무엇을 검증했고 무엇을 검증하지 못했는지 먼저 밝힙니다.

| 항목 | 상태 |
|---|---|
| 본문·표·보충자료(SAS 매크로) | 업로드된 PDF 전문을 직접 확인 |
| 표 3, 7, 11, 13의 산술 (일반지배 가중치 합, 조건부 평균, 부분집합 $R^2$ 차분) | **Python으로 재계산하여 검증 완료** |
| 표 9, 12의 $\bar{D}_{ij}$ 및 SE 값 | **$P_{ij}, P_{ji}, P_{no_{ij}}$로부터 재계산하여 일치 확인** |
| 페이지 번호 | PDF 텍스트 스트림의 러닝헤드 기준 (인쇄본과 ±1 오차 가능) |
| **2020년 이후 최신 연구 (8-2)** | **이 세션에서 웹 검색이 불가하여 온라인 검증 불가.** 각 문헌에 신뢰도 등급을 표기했으며, 확신이 낮은 항목은 명시적으로 "미검증"으로 표시했습니다. 인용 전 반드시 직접 확인하시기 바랍니다. |

> **중요한 사전 정정**: 이 논문에는 **Figure가 단 하나도 없습니다.** 모든 시각 자료는 Table 1~13입니다. 따라서 요청 7번("가장 중요한 그림 5개")은 **가장 중요한 표 5개**에 대한 해석으로 대체했습니다.

> **두 번째 사전 정정**: 이 논문은 딥러닝/ML 논문이 아니라 **1970~90년대 통계 방법론 논쟁에 대한 방법론 제안 논문**입니다. 따라서 "모델 구조"는 신경망 아키텍처가 아니라 **분석 절차의 계산 구조**로, "성능 향상"은 벤치마크 정확도가 아니라 **판정 가능성·해석 가능성의 개선**으로 재해석하여 기술했습니다. 이 재해석 없이 답하면 논문에 없는 내용을 지어내게 됩니다.

---

## 1. Executive Summary (10문장)

1. 다중회귀에서 "어떤 예측변수가 더 중요한가"를 묻는 질문은 사회과학에서 보편적이지만, 표준화계수·단순상관·편상관·구조계수 등 기존 지표들은 서로 다른 순위를 산출하며 통일된 중요도 정의가 없다 (pp. 130–133).
2. 저자들은 Budescu(1993)의 **지배분석(Dominance Analysis, DA)** 을 확장하여, "예측변수 $X_i$가 $X_j$보다 중요하다"를 **$X_i$의 $R^2$ 추가기여도가 $X_j$보다 크다**는 직관적 정의로 조작화한다.
3. 핵심 아이디어는 $2^p-1$개 모든 부분집합 모형을 적합한 뒤, 두 예측변수를 **동일한 나머지 변수 맥락($2^{p-2}$개 부분집합)** 안에서 쌍별로 비교하는 것이다 (Table 3, p. 136).
4. 이 논문의 최대 기여는 원래의 엄격한 정의를 완화하여 **완전지배(complete) → 조건부지배(conditional) → 일반지배(general)** 의 3단계 위계를 도입한 것으로, 상위 단계는 하위 단계를 함의하지만 역은 성립하지 않는다 (p. 137).
5. 일반지배 측도는 예측변수 전체에 대해 합하면 정확히 전체 모형의 $R^2$가 되며(표 3: $0.292+0.076+0.095+0.121=0.584$, 재계산 검증 완료), 이는 Lindeman et al.(1980)의 LMG 및 Johnson(2000)의 relative weight와 일치한다.
6. DA는 억제변수(suppressor)를 탐지할 수 있는데, 일반 예측변수의 조건부 기여도는 모형 크기 $k$가 커질수록 단조 감소하는 반면 억제변수는 **증가**한다 (Table 6, p. 138: $X_2$가 $0.000 \to 0.014 \to 0.034$).
7. 특정 변수를 강제 포함시키는 **제약 DA(constrained DA)** 를 통해 이론적으로 의미 있는 부분집합만 비교할 수 있으며, 이때 중요도 순위가 뒤바뀔 수 있음을 보인다 (Table 7, p. 139: $X_1,X_4,X_3,X_2 \to X_3,X_2,X_4$).
8. 표본 결과의 안정성 평가를 위해 **부트스트랩**($S=1{,}000$)을 도입하고, $D_{ij}\in\{0,0.5,1\}$의 평균 $\bar{D}_{ij}$, 표준오차, 그리고 표본 결과가 재현되는 비율인 **재현성(reproducibility)** 을 보고한다 (Tables 9, 12).
9. 실증 예시로 생활만족도 자료($n=428$, self > family > finance/housing > health)와 캘리포니아대 입학 자료($n=77{,}893$, HSGPA가 SAT II를 일반지배, HSGPA·SAT II가 SAT I을 완전지배)를 제시한다 (Tables 12, 13).
10. 다만 논문은 지배 판정에 대한 **정식 유의성 검정·신뢰구간·시뮬레이션 검증을 제공하지 않으며**, 부트스트랩 재현성을 "모집단에 대한 신뢰수준"으로 해석하는 부분은 통계적으로 과잉 주장이다(§5 참조).

### 1-1. 연구의 목적과 필요성

**목적.** 상관된 예측변수들 사이의 상대적 중요도를, (a) 명확히 정의 가능하고 (b) 직관적으로 해석 가능하며 (c) 다양한 연구 질문(단독 예측력, 전체 통제 후 기여, 특정 부분집합 대비 기여)을 하나의 틀 안에서 포괄하는 방식으로 측정·비교하는 일반 절차를 제시하는 것.

**필요성 (논문이 제시하는 4가지 근거).**

1. **측도 간 순위 불일치** (p. 135, Table 2). 동일한 상관행렬에서 지표별로 순위가 갈린다:
   - $\hat\beta_i$ / semipartial $r_i^2$ / partial $r_i^2$ → $X_1, X_2, X_3, X_4$
   - $\rho_{X_iY}$ / Kruskal / Theil / 구조계수 → $X_1, X_4, X_3, X_2$
   - $\beta_i\rho_{X_iY}$ → $X_1, X_3, X_4, X_2$
   
   즉 **어떤 지표를 쓰느냐가 결론을 결정**하는데, 선택 기준이 없다.
2. **해석의 오남용** (p. 133). 회귀계수는 유의하지 않은데 이변량 상관은 유의한 상황 등이 연구자를 혼란시키며, Courville & Thompson(2001)은 실제 출판 논문에서의 오용 사례를 문서화했다.
3. **모형선택 단계와 예측변수 비교 단계의 분리 필요** (pp. 129–130). 이 논문은 모형이 이미 선택되었다고 **가정**하고 두 번째 단계만 다룬다.
4. **연구질문의 일반성** (pp. 133–134). "부모 지지가 또래 지지보다 중요한가?"라는 질문은 특정 모형 하나가 아니라 **모든 가능한 통제 맥락**에서의 우위를 묻는 것이므로, 단일 맥락 지표로는 답할 수 없다.

---

## 2. 핵심 주장과 근거 정리표 (요청 2 + 3 통합)

각 행에 페이지/표 번호를 명시했습니다.

| # | 핵심 주장 | 근거 | 위치 | 근거 유형 | 내 검증 |
|---|---|---|---|---|---|
| C1 | 기존 중요도 지표들은 동일 데이터에서 상이한 순위를 낳으므로 통일된 정의가 필요하다 | 4개 예측변수 가상 예시에서 8개 지표가 3가지 다른 순위 산출 | p. 135, **Table 2** (모집단 값), p. 143 **Table 11** (실자료) | 구성된 반례 | 순위 재계산 일치 |
| C2 | 중요도는 "동일 맥락에서의 $R^2$ 추가기여도"로 정의되어야 한다 | 정의 제시 및 예시 계산 | pp. 133–135, **Table 3** | 개념적 정의 | — |
| C3 | 완전지배는 판정 불가(indeterminate) 사례를 많이 남긴다 | 4변수 모집단 예시에서 6쌍 중 3쌍만 판정 ( $X_2$ – $X_3$ , $X_2$ – $X_4$ , $X_3$ – $X_4$ 미판정) | p. 136, **Table 3** | 구성 예시 | 표 3에서 확인 |
| C4 | 조건부·일반지배 도입으로 판정 불가가 감소한다 | 일반지배는 6쌍 전부 판정 ($X_1 > X_4 > X_3 > X_2$) | p. 137 본문 + **Table 3** 최하단 | 구성 예시 | 판정률 3/6 → 3/6 → 6/6 확인 |
| C5 | 세 수준은 위계적이다 (완전 ⇒ 조건부 ⇒ 일반, 역은 $p>3$에서 불성립) | 논리적 정의로부터 도출 | p. 137 | 연역 | 정의상 타당 |
| C6 | 일반지배 측도의 합 = 전체 모형 $R^2$ | $0.292+0.076+0.095+0.121=0.584$ | p. 136, **Table 3** | 수치 예시 | **재계산 일치** |
| C7 | 일반지배 측도 = Lindeman et al.(1980) LMG = Johnson(2000) relative weight 근사 | 서술적 동일성 주장 | p. 137 | 문헌 대조 (증명 없음) | 표 2에서 Kruskal 평균 $(0.362, 0.115, 0.132, 0.144)$ ≠ 일반지배 $(0.292, 0.076, 0.095, 0.121)$ — **§5-8 참조** |
| C8 | DA는 SSE의 단조함수인 모든 적합도 지표에서 동일한 지배 패턴을 산출 | adjusted $R^2$, AIC, $C_p$에 대한 증명 | p. 138 (증명은 Azen, 2000 학위논문에 위임) | **논문 내 증명 없음** | 동일 크기 모형 간 비교이므로 성립함을 확인 |
| C9 | DA는 억제변수를 탐지할 수 있다 | 억제변수 $X_2$의 조건부 기여가 $k$ 증가에 따라 $0.000\to0.014\to0.034$로 **증가**, 반면 $X_1$은 $0.250\to0.239\to0.234$, $X_3$은 $0.062\to0.042\to0.026$ 감소 | pp. 137–138, **Tables 5, 6** | 단일 구성 예시 | 표 6에서 확인 |
| C10 | $\rho_{X_iY}$ 기반 지표는 억제변수에 0을 부여해 오도한다 | $\rho_{X_2Y}=0$이므로 $\beta\rho$와 구조계수 모두 0 | p. 138 (익명 심사자 지적으로 명시) | 논리 | 타당 |
| C11 | 제약 DA는 이론적 제약 하의 순위를 산출하며 순위가 바뀔 수 있다 | 전체 분석 $X_1,X_4,X_3,X_2$ → $X_1$ 강제 시 $X_3,X_2,X_4$; $X_3$이 $X_4$를 완전지배 | p. 139, **Table 7** | 구성 예시 | $0.078{+}0.087{+}0.059{+}0.360=0.584$ **재계산 일치** |
| C12 | 부트스트랩으로 지배 결과의 안정성을 정량화할 수 있다 | $\bar{D}\_{ij}$, $SE$, $P_{ij}$, $P_{ji}$, $P_{no_{ij}}$, 재현성 | pp. 139–142, **Tables 9, 12** | 절차 제안 | SE가 $P$들의 결정론적 함수임을 **재계산으로 확인** |
| C13 | 지배 수준을 완화할수록 $D_{ij}=1$의 재현성은 증가, $D_{ij}=0.5$의 재현성은 감소 | $D_{34}$ 재현성: 완전 .566 → 조건부 .604 → 일반 .848 | p. 142, **Table 9** | 수치 예시 | 표에서 확인. 단 **§5-5의 순환성 문제** |
| C14 | DA는 실제 정책 결정에 쓰일 수 있다 | Geiser & Studley(2002) UC 자료 재분석: HSGPA가 SAT II를 일반지배, HSGPA·SAT II가 SAT I을 완전지배 | p. 146, **Table 13** | 2차 자료 재분석 | **전체 DA를 표 13의 $R^2$로부터 재계산: HSGPA .0945, SAT I .0540, SAT II .0745, 합 .223 — 논문과 일치** |
| C15 | DA는 인과·매개·경로모형 질문에는 부적합 | 명시적 한계 선언 | p. 146 | 저자 자인 | 타당 |

---

### 2-1. 상세 설명: 문제 / 방법(수식) / 구조 / 성능·한계

#### (A) 해결하고자 하는 문제

**형식화.** 표준화된 모집단 회귀모형 (Eq. 1, p. 130):

$$Y_j=\beta_1X_{1j}+\beta_2X_{2j}+\cdots+\beta_pX_{pj}+e_j=\sum_{i=1}^{p}\beta_iX_{ij}+e_j,\quad e_j\sim N(0,\sigma^2)$$

$Y, X_1,\dots,X_p$는 모두 평균 0·분산 1로 표준화되어 절편이 없다. 예측값은 $\hat Y_j=\sum_i \beta_i X_{ij}$, 모형 적합도는 $\rho^2_{Y\hat Y}$(표본에서는 $R^2$)이다.

**문제.** 예측변수들이 상관되어 있을 때($\rho_{X_iX_j}\neq 0$), $\rho^2_{Y\hat Y}$를 개별 변수에 **유일하게** 배분하는 방법이 존재하지 않는다. 상관이 0이면 $\sum_i \rho^2_{YX_i}=\rho^2_{Y\hat Y}$가 성립하고 문제는 사라지지만, 사회과학에서 이런 경우는 사실상 없다.

**구체적 병리 (Table 1, p. 134의 상관행렬로 예시)**

| | $Y$ | $X_1$ | $X_2$ | $X_3$ | $X_4$ |
|---|---|---|---|---|---|
| $Y$ | — | .6 | .3 | .4 | .5 |
| $X_1$ | .6 | — | **.8** | .1 | .3 |
| $X_2$ | .3 | .8 | — | .1 | .1 |
| $X_3$ | .4 | .1 | .1 | — | .5 |
| $X_4$ | .5 | .3 | .1 | .5 | — |

$\rho_{X_1X_2}=.8$의 강한 공선성 때문에 $\beta_1=0.905$, $\beta_2=-0.466$이 되어(Table 2), $\rho_{X_2Y}=+.3$인데도 계수 부호가 뒤집힌다. 그 결과 Pratt(1987)의 $\beta_i\rho_{X_iY}=-0.140$이라는 **음수 "분산 비율"** 이 나온다(논문 각주 1, p. 134에서 저자도 지적).

#### (B) 제안 방법 (수식)

**Step 1 — 추가기여도.** 부분집합 $\mathbf{X}_h \subseteq \mathbf{X}\setminus\{X_i\}$에 대해:

$$C_{X_i}(\mathbf{X}_h)=\rho^2_{Y\cdot \mathbf{X}_h\cup\{X_i\}}-\rho^2_{Y\cdot \mathbf{X}_h}=\rho^2_{Y(X_i\cdot \mathbf{X}_h)}$$

이는 $\mathbf{X}_h$를 통제한 **제곱 준편상관(squared semipartial correlation)** 과 정확히 같다 (p. 135).

*예시 (Table 3)*: $C_{X_1}(\{X_3\})=\rho^2_{YX_1X_3}-\rho^2_{YX_3}=0.477-0.160=0.317$

**Step 2 — 완전지배 (complete dominance).**

$$X_i \succ_{C} X_j \iff C_{X_i}(\mathbf{X}_h) > C_{X_j}(\mathbf{X}_h)\quad \forall\,\mathbf{X}_h\subseteq \mathbf{X}\setminus\{X_i,X_j\}$$

비교 횟수는 쌍당 $2^{p-2}$개다. 이 관계는 **이행적(transitive)** 이다: $X_i\succ X_j \wedge X_j \succ X_k \Rightarrow X_i \succ X_k$ (Budescu, 1993; p. 136 인용).

**Step 3 — 조건부지배 (conditional dominance).** 모형 크기 $k=|\mathbf{X}_h|$별 평균:

$$\bar{C}^{(k)}_{X_i}=\binom{p-1}{k}^{-1}\!\!\sum_{\substack{\mathbf{X}_h\subseteq \mathbf{X}\setminus\{X_i\}\\ |\mathbf{X}_h|=k}}\!\! C_{X_i}(\mathbf{X}_h),\qquad k=0,1,\dots,p-1$$

$$X_i \succ_{A} X_j \iff \bar{C}^{(k)}_{X_i} > \bar{C}^{(k)}_{X_j}\quad \forall\, k$$

\*예시*: $\bar{C}^{(1)}_{X_1}=\frac{0.360+0.317+0.223}{3}=0.300$ (Table 3의 " $k=1$ average" 행)

**Step 4 — 일반지배 (general dominance).** 크기별 평균을 **등가중**으로 다시 평균:

$$\bar{C}_{X_i}=\frac{1}{p}\sum_{k=0}^{p-1}\bar{C}^{(k)}_{X_i}, \qquad X_i \succ_{G} X_j \iff \bar{C}_{X_i}>\bar{C}_{X_j}$$

**가법성 정리** (p. 137에서 주장, 증명 없음):

$$\sum_{i=1}^{p}\bar{C}_{X_i}=\rho^2_{Y\hat Y}$$

*검증*: $0.292+0.076+0.095+0.121=0.584=\rho^2_{Y\hat Y}$ ✔

**등가 표현 (논문에는 없는 내 정리).** 위 식은 협조적 게임 $v(\mathbf{X}\_h)=\rho^2_{Y\cdot\mathbf{X}_h}$에 대한 **Shapley 값**과 대수적으로 동일하다:

$$\bar{C}_{X_i}=\sum_{\mathbf{X}_h\subseteq \mathbf{X}\setminus\{X_i\}}\frac{|\mathbf{X}_h|!\,(p-|\mathbf{X}_h|-1)!}{p!}\Big[\rho^2_{Y\cdot\mathbf{X}_h\cup\{X_i\}}-\rho^2_{Y\cdot\mathbf{X}_h}\Big]$$

이 등가성이 §8-2에서 논할 SHAP과의 연결고리다. **논문 자체는 Shapley를 한 번도 언급하지 않는다** — 이는 내 해석이다.

**Step 5 — 부트스트랩 추론** (pp. 139–141).

$$D_{ij}=\begin{cases}1 & X_i \text{ dominates } X_j\\ 0 & X_j \text{ dominates } X_i\\ 0.5 & \text{판정 불가}\end{cases}$$

$$\bar{D}_{ij}=\frac{1}{S}\sum_{s=1}^{S}D^{s}_{ij} \quad \text{(Eq. 2)}, \qquad SE(\bar{D}_{ij})=\sqrt{\frac{1}{S-1}\sum_{s=1}^{S}\left(D^{s}_{ij}-\bar{D}_{ij}\right)^2}\quad\text{(Eq. 3)}$$

$$\bar{D}_{ij}=P_{ij}+\tfrac{1}{2}P_{no_{ij}}, \qquad D_{ij}+D_{ji}=1$$

**참고 (Theil 측도, p. 132)**: $I(x)=-\log_2(1-x)$, $I(\rho^2_{Y\hat Y})=I(\rho^2_{YX_1})+I(\rho^2_{YX_2\cdot X_1})+\cdots$

#### (C) "모델 구조" — 계산 아키텍처

신경망 구조는 없다. 보충자료의 SAS 매크로 `%dom`은 3단계 파이프라인이다:

```
입력: Y, X1...Xp (최대 p = 10)
  │
  ├─ PART 1: 원자료 DA
  │    proc reg (selection=adjrsq, best=2^p−1) → 2^p−1개 부분집합 R²
  │    → 사전식 정렬 → IML로 추가기여도 행렬 contrib[nrow × p]
  │    → Dcsample, Dasample, Dgsample (각 p×p 지배행렬)
  │
  ├─ PART 2: 부트스트랩 루프 (기본 B = 1000)
  │    predtype='r' → 케이스(쌍) 리샘플링   [본 논문의 모든 예시]
  │    predtype='f' → 잔차 리샘플링
  │    각 반복마다 2^p−1개 회귀 재적합 → Dc/Da/Dg → 빈도행렬 누적
  │
  └─ PART 3: 확률·재현성 요약표
       Dij, Dij_mean, Dij_SE, Pij, Pji, Pijno, reprod
```

**계산 복잡도**: 부트스트랩 총 회귀 적합 횟수 $\approx B\cdot(2^p-1)$. $p=10, B=1000$이면 약 $1.02\times10^6$회. 보충자료가 명시적으로 **"최대 10개 예측변수"** 제한을 건 이유이며, 이것이 이 방법의 **가장 실질적인 확장성 병목**이다.

#### (D) "성능 향상" — 논문이 실제로 개선했다고 주장할 수 있는 것

정확도 벤치마크는 없다. 개선은 다음 4가지로 측정 가능하다:

**(D-1) 판정 가능성 (내가 계산한 수치)**

| 자료 | 쌍의 수 | 완전지배 판정 | 조건부 | 일반 |
|---|---|---|---|---|
| Table 3 (4변수 모집단) | 6 | 3 (50%) | 3 (50%) | 6 (100%) |
| Table 7 (제약, $X_1$ 고정) | 3 | 1 (33%) | 1 (33%)* | 3 (100%) |
| Table 9 ($n=100$ 표본) | 6 | 5 (83%) | 5 (83%) | 6 (100%) |
| Table 12 (생활만족도 $n=428$) | 10 | 9 (90%) | 9 (90%) | 10 (100%) |

\* 표 7에 대해 논문은 조건부 판정을 명시적으로 보고하지 않음 — 내 추론.

→ **핵심 개선: 미판정률 50% → 0%.** 다만 이것은 "더 정확해졌다"가 아니라 **"판정 기준을 약화시켜 더 많이 판정하게 만들었다"** 는 것이며, 저자들도 p. 145에서 이를 인정한다.

**(D-2) 가법성**: $\sum\bar{C}_{X_i}=R^2$ (편상관·구조계수는 불성립)

**(D-3) 억제변수 탐지**: 다른 어떤 리뷰된 지표도 못 하는 기능 (Table 6, C9–C10)

**(D-4) 안정성 정량화**: 표본 결과에 재현성 수치를 부여 (Tables 9, 12)

#### (E) 한계 (저자 자인 + 내 추가) → §5, §6에서 상세

- 저자 자인: 측정오차·불량 표본에 취약(p. 146), 인과/매개/경로모형 부적합(p. 146), 위계적 진입 순서 질문에 부적합(p. 146), 등가중 선택은 자의적(p. 137)
- 내 추가: $2^p$ 폭발, 지배 판정에 대한 검정 부재, 부트스트랩 재현성의 과잉 해석, 모형선택 불확실성 무시, in-sample $R^2$의 상향 편의 — §5 참조

---

## 4. 저자의 보고 vs. 나의 해석 (엄격 분리)

### 4-1. 연구 주제

| 저자가 직접 보고한 것 | 나의 해석 |
|---|---|
| "예측변수 비교(predictor comparison)" 단계에 초점을 두며, 모형선택 단계는 이미 완료되어 선택된 모형이 옳다고 **가정**한다 (pp. 129–130). | 이 가정이 논문 전체의 아킬레스건이다. 실제 연구에서는 stepwise 등으로 모형을 고른 뒤 DA를 돌리는데, 이때 **선택 후 추론(post-selection inference)** 문제가 발생한다. 논문은 이 문제를 인지하고 회피했을 뿐 해결하지 않았다. |
| "중요도는 예측 기여도이며, 둘 중 하나만 고를 수 있다면 어느 것을 고르겠는가의 문제"(p. 145). | 이 정의는 **예측적(predictive)** 정의이지 **설명적/인과적** 정의가 아니다. 논문은 "중요도"라는 단어의 일상적 함의(= 원인으로서의 중요성)와 조작적 정의 사이의 간극을 충분히 경고하지 않는다. 정책 결정(§Table 13의 UC 입시)에 쓰일 때 이 간극은 실질적 위험이 된다. |

### 4-2. 방법

| 저자가 직접 보고한 것 | 나의 해석 |
|---|---|
| 등가중 평균 $\bar{C}\_{X_i}=\frac1p\sum_k \bar{C}^{(k)}_{X_i}$을 쓰면 합이 $R^2$가 되는 "매력적 성질(appealing property)"이 있다 (p. 137). | 이는 **성질로부터 가중치를 역산한 사후 정당화**다. 저자도 "이론적으로 불균등 가중을 막을 것은 없다"고 인정한다. 게임이론적으로 보면 등가중은 Shapley 공리(효율성·대칭성·더미·가법성)의 유일해이므로 **훨씬 강한 정당화가 가능한데, 논문은 이 논거를 놓쳤다.** |
| SSE의 단조함수인 적합도 지표는 동일한 지배 패턴을 낳는다 (p. 138, 증명은 Azen 2000에 위임). | 내가 확인한 바로 이 주장은 성립한다. 이유: 모든 지배 비교는 $\rho^2_{Y\cdot\mathbf{X}\_h\cup\{X_i\}}$ vs $\rho^2_{Y\cdot\mathbf{X}_h\cup\{X_j\}}$로 환원되는데 두 모형은 **크기가 같으므로** adjusted $R^2$ /AIC/ $C_p$의 벌점항이 상쇄된다. 단, **논문 본문에 증명이 없어 독자가 검증할 수 없다**는 점은 결함이다. |
| $D_{ij}$의 부트스트랩 표준오차 (Eq. 3). | $D_{ij}$는 $\{0,0.5,1\}$ 3점 이산변수이므로 SE는 $P$들의 **결정론적 함수**다: $\mathrm{Var}=P_{ij}+0.25P_{no}-\bar{D}^2$. 내가 재계산한 결과 표 9·12의 모든 SE 값이 이 공식과 일치한다. 즉 **SE는 새로운 정보를 전혀 담고 있지 않으며**, $P$들을 이미 보고한 상황에서 중복 열이다. 또한 분포가 극단적으로 비정규·유계이므로 $\bar{D}\pm 2SE$ 형태의 구간 해석은 무의미하다. |

### 4-3. 결과

| 저자가 직접 보고한 것 | 나의 해석 |
|---|---|
| 표 9: $n=100$ 표본에서 $X_3$이 $X_2$와 $X_4$를 완전지배 ($D_{32}=1$, $D_{34}=1$). | **모집단(표 3)에서는 이 두 쌍 모두 완전지배가 성립하지 않는다.** 즉 이 표본은 존재하지 않는 지배관계를 두 건 만들어냈다. 재현성이 각각 .505, .566으로 낮게 나온 것이 이를 부분적으로 포착하지만, **논문은 이 불일치를 명시적으로 지적하지 않는다.** 이것이 표 9에서 얻을 수 있는 가장 중요한 교훈인데도 그렇다. |
| "1,000개 부트스트랩 중 97%에서 재현되었다면 모집단에서 $X_3$이 $X_2$를 지배한다고 97% 확신할 수 있다" (p. 140). | **통계적으로 부정확하다.** 부트스트랩 분포는 표본 통계량을 중심으로 하므로 재현성은 "표본 결과가 재표집에서 얼마나 자주 재생산되는가"이지 "모집단에서 참일 확률"이 아니다. 편의(bias)가 있으면 높은 재현성이 곧 정확성을 뜻하지 않는다. 실제로 위의 $D_{32}$ 사례가 반례에 가깝다. |
| 표 13: HSGPA가 SAT II를 일반지배하고, HSGPA·SAT II가 SAT I을 완전지배한다. | 재계산 결과 정확하다(HSGPA .0945 > SAT II .0745 > SAT I .0540). 그러나 **차이의 실질적 크기가 작다**: 전체 $R^2=0.223$ 중 HSGPA .094 vs SAT II .074, 차이 .020(전체 설명분산의 3.4%p). 또 $n=77{,}893$은 UC 여러 캠퍼스에 걸친 **군집 자료**인데 케이스 리샘플링 부트스트랩(iid 가정)은 적용되지 않았고 SE도 보고되지 않았다. 이 결과가 SAT 개편의 근거로 인용되었다는 저자의 서술(p. 146, Barnes 2002)을 고려하면 **불확실성 미보고는 심각한 누락**이다. |
| 생활만족도: self > family > finance/housing > health (Table 12). | 순위 자체는 안정적이다(self 관련 $\bar{D}$가 대부분 1.000). 그러나 이는 **횡단·자기보고·단일문항 예측변수 5개**의 결과이며, "self 영역을 개선하면 생활만족도가 오른다"는 개입 함의는 도출되지 않는다. 논문의 도입부(pp. 131–132)가 바로 그런 개입 시나리오로 동기를 부여한다는 점에서 **동기와 결론 사이에 인과적 비약**이 있다. |

---

## 5. 통계적으로 취약한 부분 및 비교 불가능한 수치

### 5-A. 통계적 취약점

| # | 취약점 | 근거 위치 | 심각도 |
|---|---|---|---|
| 5-1 | **지배 판정에 대한 유의성 검정·신뢰구간이 없다.** $\bar{C}\_{X_i}-\bar{C}_{X_j}$의 표집분포가 제시되지 않으며, 부트스트랩은 지표값이 아니라 **이진 판정 결과**에만 적용된다. 표 13의 SAT I 조건부 기여 $.001$과 $.002$ 같은 값은 표집오차 안에 완전히 묻힌다. | Tables 9, 12, 13 | **높음** |
| 5-2 | **재현성 = 신뢰수준 해석 오류.** §4-3 참조. 부트스트랩 편의 보정(bias-corrected, BCa)이 없다. | p. 140 | **높음** |
| 5-3 | **표본 DA의 지배 과잉선언.** 표 9에서 모집단에는 없는 완전지배 2건이 발생. $R^2$는 상향 편의를 갖고 $2^p-1$개 모형에 걸쳐 최대값을 반복 비교하므로 **다중비교 성격의 편의**가 누적된다. | Table 3 vs Table 9 | **높음** |
| 5-4 | **시뮬레이션 검증 부재.** 지배 판정의 1종/2종 오류율, 재현성의 캘리브레이션, 필요 표본크기에 대한 어떤 연구도 없다. 모든 증거가 소수의 **구성된 예시**다. | 논문 전체 | **높음** |
| 5-5 | **재현성 비교의 순환성.** "지배 수준을 완화하면 재현성이 올라간다"(C13)는 발견이 아니라 **정의상 필연**이다. 일반지배에서는 판정 불가가 거의 불가능하므로 $P_{no}=0$이 되고, $D=1$의 재현성이 기계적으로 상승한다. 저자도 각주 3에서 부분적으로 인정하나, 본문 p. 142에서는 여전히 " $X_3$이 $X_4$를 지배한다는 확신이 일반지배에서 높다"고 서술한다. | p. 142 + 각주 3 | 중간 |
| 5-6 | **군집·비독립 자료 미대응.** 케이스 리샘플링은 iid를 가정. UC 자료(캠퍼스 군집), Suh et al. 자료(41개국 중 미국만 추출) 모두 위배 소지. | pp. 140, 142, 146 | 중간 |
| 5-7 | **모형선택 불확실성 무시.** 선택된 모형을 "옳다"고 가정(p. 130). 실제로는 선택 단계의 불확실성이 DA 결과에 전파된다. | p. 130 | 중간 |
| 5-8 | **표 2의 Kruskal 평균과 표 3의 일반지배 값 불일치.** 논문은 일반지배 측도가 Lindeman et al.(1980)/Kruskal류 평균과 일치한다고 주장(p. 137)하나, 표 2의 "Kruskal's average" $(0.362, 0.115, 0.132, 0.144)$는 합이 $0.753 \neq 0.584$이며 표 3의 일반지배 값과 다르다. Kruskal(1987)은 **제곱 편상관**, Lindeman et al.(1980)은 **제곱 준편상관**을 평균하므로 다른 양이다 — 논문이 이 구분을 표 2 라벨에서 명확히 하지 않아 **독자가 직접 오독하기 쉽다.** (반면 표 11의 Kruskal 평균은 합 $=0.6045$로 $R^2$와 정합적으로 보여, 두 표의 라벨 의미가 서로 다를 가능성이 있다.) | Table 2 (p. 135) vs Table 3 (p. 136) vs Table 11 (p. 143) | **높음 — 논문 내 일관성 문제** |
| 5-9 | **$C_p$·AIC 동치 증명이 본문에 없음** (Azen, 2000 학위논문 위임). | p. 138 | 낮음 |
| 5-10 | **억제변수 주장의 근거 부족.** "음성·상호 억제에서도 유사한 패턴이 관찰된다"고 하나 **"not shown"** 으로 처리. 조건부 기여의 단조 감소가 일반 예측변수에서 **항상** 성립한다는 주장도 증명 없음(p. 137은 "expected to decrease"라고만 서술). | pp. 137–138 | 중간 |
| 5-11 | **표 11에 전체 $R^2$가 보고되지 않음.** Kruskal 평균 합으로 역산하면 $0.6045$이나, 논문 어디에도 명시되지 않아 독자가 가중치를 정규화할 수 없다. | Table 11 (p. 143) | 낮음 |

### 5-B. 비교 불가능한 수치 (⚠️ 표시)

| ⚠️ | 비교 불가 조합 | 이유 |
|---|---|---|
| ⚠️1 | **Table 2 / Table 11의 행(row) 간 값** | 척도가 전부 다르다. $\beta$(SD 단위), $\rho$(무차원 $[-1,1]$), $r^2$(분산 비율), Theil 평균(**비트 단위, 로그 척도**), 구조계수(비율). **열 방향 순위만 비교 가능하며, 행 간 크기 비교는 무의미하다.** 예: 표 2의 Theil $0.651$과 semipartial $r^2$ $0.246$은 같은 축 위에 있지 않다. |
| ⚠️2 | $\beta_i\rho_{X_iY}$의 음수값 ($X_2$: $-0.140$) | 합이 $R^2$가 되도록 설계됐지만 음수가 가능해 "분산 분해"로 해석 불가. 다른 비음 측도와 크기 비교 불가. |
| ⚠️3 | **Table 3(모집단) vs Table 9($n=100$ 표본)** | 하나는 모수, 하나는 추정치. 직접 대조는 "표집오차 예시"로만 유효하며 정확도 평가로 쓸 수 없다(모집단이 인위적으로 구성된 것이므로 외적 타당도도 없음). |
| ⚠️4 | **완전지배 재현성 vs 일반지배 재현성** (예: $D_{34}$의 .566 vs .848) | **서로 다른 사건의 확률**이다. 전자는 " $2^{p-2}$개 비교 전부에서 우위", 후자는 "평균 하나에서 우위". 5-5의 순환성 때문에 후자가 항상 높다. **"확신이 커졌다"로 읽으면 안 된다.** |
| ⚠️5 | **Table 12의 지배 결과 vs Table 11의 $t$값· $\beta$ ** | health: $\beta=.0050$, $t=0.13$(비유의), semipartial $r^2=.0000$ → "기여 없음". 그러나 일반지배 가중치는 $.0213$으로 0이 아니고 순위도 5위에 안착. **두 수치는 다른 질문에 답한다**(전체 통제 후 한계기여 vs 모든 맥락 평균 기여). 모순이 아니라 비교 대상이 아니다. |
| ⚠️6 | **Table 13 vs Tables 3/9/12** | $n=77{,}893$, 2차 자료(Geiser & Studley 2002 Table 2에서 전재), 반올림된 $R^2$만 사용, 부트스트랩 미실시, SE 없음. 다른 예시들과 불확실성 수준이 전혀 다르다. |
| ⚠️7 | Table 13의 일반지배 합 $0.094+0.054+0.074=0.222$ vs 전체 $R^2=0.223$ | 반올림 오차(정확 재계산 시 $0.0945+0.0540+0.0745=0.2230$). 보고 자릿수 한계 — 소수 셋째 자리 차이를 실질적으로 해석하면 안 된다. |
| ⚠️8 | Table 13의 SAT I 조건부 기여 $k=2$: $0.001$ | 사실상 0. 두 자리 반올림 수준의 값으로 지배 판정을 논하는 것은 부적절. |

---

## 6. 이 문서가 답하지 않는 질문

### 6-A. 통계적 추론
1. 일반지배 가중치 $\bar{C}_{X_i}$ 자체의 **신뢰구간**은 어떻게 구하는가?
2. $H_0:\bar{C}\_{X_i}=\bar{C}_{X_j}$에 대한 **검정 절차**는? 검정력은 표본크기·공선성 수준에 따라 어떻게 변하는가?
3. 안정적 재현성 추정에 필요한 **$S$(부트스트랩 반복 수)** 는? 논문은 1,000을 쓸 뿐 근거를 제시하지 않는다.
4. 재현성이 **얼마 이상이어야** 결과를 신뢰할 수 있는가? .505(Table 9, $D_{32}$)와 .999(Table 12, $D_{31}$)를 가르는 기준선이 없다.
5. $2^p-1$개 모형에 걸친 반복 비교의 **다중성 보정**은 필요한가?

### 6-B. 방법론적 범위
6. $p>10$일 때 어떻게 하는가? (**보충자료가 명시적으로 10개 제한**) 근사·표집·순차 알고리즘은?
7. **결측치** 처리는? (listwise deletion 시 부분집합마다 $n$이 달라지는 문제 포함)
8. **범주형 예측변수**, 더미 집합, **변수군(grouped predictors)** 은 어떻게 다루는가?
9. **상호작용항·다항항**이 있는 모형에서 위계 원칙(hierarchy)과 부분집합 열거는 어떻게 조화시키는가?
10. **로지스틱회귀·다층모형·생존분석** 등 비선형/비독립 모형으로의 확장은? (Azen & Traxel 2009, Luo & Azen 2013이 후에 다루지만 이 논문에는 없음)
11. **정규화 회귀(ridge/lasso)** 하에서 DA는 정의되는가?

### 6-C. 해석과 실무
12. 지배 순위와 **인과적 개입 효과**의 관계는? (논문은 인과 해석을 금지하지만, 표 13의 정책 활용은 사실상 인과적이다)
13. **측정신뢰도 차이**가 지배 순위를 어떻게 왜곡하는가? (Cooper & Richardson 1986을 인용만 하고 보정법 미제시)
14. 지배 순위의 **효과크기 기준**은? " $\bar{C}_1=.094$ vs $\bar{C}_2=.074$ "의 실질적 차이 크기를 어떻게 판단하는가?
15. **표본 외 예측(out-of-sample)** 에서도 지배관계가 유지되는가? — §8-1의 핵심
16. Johnson(2000) relative weight와 일반지배가 **언제 갈라지는가**? 논문은 "매우 잘 근사한다"고만 서술하고 오차 상한을 제시하지 않는다.
17. 모형선택 절차(stepwise, LASSO 등)를 거친 뒤 DA를 적용할 때의 **선택 후 추론** 문제는?

---

## 7. 가장 중요한 "표" 5개 해석

> 반복 확인: **이 논문에 Figure는 없습니다.** 아래는 핵심 Table 5개입니다.

### 7-1. Table 2 (p. 135) — 문제 제기의 결정적 증거

| 측도 | $X_1$ | $X_2$ | $X_3$ | $X_4$ | 함의 순위 |
|---|---|---|---|---|---|
| $\beta_i$ | 0.905 | **−0.466** | 0.291 | 0.130 | $X_1,X_2,X_3,X_4$ |
| Semipartial $r_i^2$ | 0.246 | 0.071 | 0.061 | 0.010 | $X_1,X_2,X_3,X_4$ |
| Partial $r_i^2$ | 0.372 | 0.146 | 0.129 | 0.025 | $X_1,X_2,X_3,X_4$ |
| $\rho_{X_iY}$ | 0.6 | 0.3 | 0.4 | 0.5 | $X_1,X_4,X_3,X_2$ |
| Kruskal 평균 | 0.362 | 0.115 | 0.132 | 0.144 | $X_1,X_4,X_3,X_2$ |
| Theil 평균 ⚠️(비트) | 0.651 | 0.177 | 0.206 | 0.233 | $X_1,X_4,X_3,X_2$ |
| $\rho_{X_i\hat{Y}}$ | 0.785 | 0.392 | 0.523 | 0.654 | $X_1,X_4,X_3,X_2$ |
| $\beta_i\rho_{X_iY}$ | 0.543 | **−0.140** | 0.117 | 0.065 | $X_1,X_3,X_4,X_2$ |

**해석.** 이 표 하나가 논문 전체의 존재 이유다. 동일한 상관행렬에서 $X_2$는 지표에 따라 **2위에서 4위까지** 오간다. $\rho_{X_1X_2}=.8$의 공선성이 $\beta_2$를 음수로 뒤집기 때문이다. 다만 이 표는 **문제의 존재를 증명할 뿐, DA가 그 해답임을 증명하지는 않는다** — DA도 결국 또 하나의 순위($X_1,X_4,X_3,X_2$)를 내놓을 뿐이고, 우연히 상관 기반 지표들과 같은 순위다. DA의 우월성 논거는 이 표가 아니라 **정의의 명확성**에 있다.

### 7-2. Table 3 (p. 136) — 방법의 심장부

| 부분집합 | $\rho^2_{Y\cdot\mathbf{X}}$ | $X_1$ | $X_2$ | $X_3$ | $X_4$ |
|---|---|---|---|---|---|
| Null ($k=0$) | 0 | .360 | .090 | .160 | .250 |
| $k=1$ 평균 | | **.300** | .074 | .095 | .152 |
| $k=2$ 평균 | | **.263** | .069 | .063 | .073 |
| $k=3$ 평균 | | **.246** | .071 | .061 | **.010** |
| **전체 평균** | .584 | **.292** | .076 | .095 | .121 |

**해석 4가지.**
1. **가법성**: 마지막 행 합 = .584 = 전체 $R^2$. 재계산 검증 완료.
2. **$X_1$의 완전지배**: 모든 행에서 $X_1$이 최대 → 3개 쌍 모두 완전지배 성립.
3. **미판정의 원인**: $X_4$는 $k=0$에서 .250(2위)이지만 $k=3$에서 .010(최하위)으로 급락한다. $\rho_{X_3X_4}=.5$ 때문에 $X_3$이 이미 모형에 있으면 $X_4$가 더할 게 없기 때문이다. **이 교차(crossing)가 바로 완전지배 실패의 기하학적 정체**이며, 조건부·일반지배를 도입한 이유다.
4. **경고**: $X_4$의 전체 평균 .121이 $X_3$의 .095보다 크지만, $k=3$에서는 .010 < .061로 완전히 역전된다. **일반지배 순위 하나만 보고하면 이 역전을 놓친다.** 저자가 "가능하면 완전지배를 먼저 보고하라"고 권고한 이유(p. 145)다.

### 7-3. Table 6 (p. 138) — 억제변수 탐지, DA만의 고유 기능

Table 5의 모집단: $\rho_{YX_1}=.50$, $\rho_{YX_2}=\mathbf{0.00}$, $\rho_{YX_3}=.25$; $\rho_{X_1X_2}=.30$

| 모형 크기 $k$ | $X_1$ | $X_2$ (고전적 억제변수) | $X_3$ |
|---|---|---|---|
| 0 | .250 | **.000** | .062 |
| 1 | .239 ↓ | **.014 ↑** | .042 ↓ |
| 2 | .234 ↓ | **.034 ↑** | .026 ↓ |
| 전체 평균 | .241 | .016 | .043 |

**해석.** 조건부 지배 프로파일의 **기울기 부호**가 진단 도구가 된다. 일반 예측변수는 다른 변수와의 상관 때문에 모형이 복잡해질수록 한계기여가 줄어들지만($X_1$: −.016, $X_3$: −.036), 억제변수는 **반대로 증가**한다($X_2$: +.034). $\rho_{YX_2}=0$이므로 $\beta\rho$와 구조계수는 정확히 0을 부여하여 "기여 없음"으로 오도하는데, DA는 $\bar{C}_{X_2}=.016>0$을 준다.

**단, 세 가지 주의**: (a) 근거가 **인위적 모집단 하나**뿐이다. (b) "음성·상호 억제에서도 유사"는 **미제시**. (c) 단조성 자체가 정리로 증명되지 않아, **역단조 = 억제**라는 진단 규칙의 오탐률을 알 수 없다.

### 7-4. Table 9 (p. 141) — 부트스트랩 추론과 그 함정

$n=100$ 표본, $S=1{,}000$ . (표 1 모집단에서 생성)

| 수준 | $i$ – $j$ | 표본 $D_{ij}$ | $\bar{D}_{ij}$ | SE | $P_{ij}$ | $P_{no}$ | 재현성 |
|---|---|---|---|---|---|---|---|
| 완전 | 1–2 | 1.0 | 1.0000 | .000 | 1.000 | .000 | 1.000 |
| 완전 | 1–4 | 1.0 | .9680 | .122 | .936 | .064 | .936 |
| 완전 | **3–2** | **1.0** | .7475 | .260 | .505 | .485 | **.505** |
| 완전 | **3–4** | **1.0** | .7800 | .254 | .566 | .428 | **.566** |
| 완전 | 2–4 | 0.5 | .5080 | .116 | .035 | .946 | .946 |
| 일반 | 2–4 | 0.0 | .2640 | .441 | .264 | .000 | .736 |

**해석 3가지.**
1. **가장 중요한 발견은 논문이 명시하지 않은 것이다.** 모집단(표 3)에서 $X_2$ – $X_3$ , $X_3$ – $X_4$는 **완전지배 미판정**인데, $n=100$ 표본은 둘 다 " $X_3$이 완전지배"라고 선언했다. 즉 **표본 DA는 없는 지배관계를 만들어낸다.** 재현성 .505/.566이 경고 신호를 주지만, 표본 결과만 보고하는 관행이라면 잘못된 결론이 그대로 출판된다.
2. **SE 열은 중복 정보다.** $\mathrm{Var}=P_{ij}+0.25P_{no}-\bar{D}^2$로 완전히 결정된다(재계산 확인: 3–2 쌍 → $\sqrt{.505+.121-.559}=.260$ ✔). 3점 이산분포이므로 정규 근사 구간은 성립하지 않는다.
3. **부호 반전 사례**: 2–4 쌍은 완전·조건부에서는 미판정($\bar{D}\approx.51$)인데 일반지배에서는 $X_4$가 $X_2$를 지배($\bar{D}_{24}=.264$)로 뒤집힌다. 표 3의 모집단 일반지배 값($X_4$ .121 > $X_2$ .076)과는 일치하지만, **"수준을 바꾸면 결론이 바뀐다"** 는 사실은 사전 등록 없이 세 수준을 모두 돌린 뒤 마음에 드는 것을 고를 위험(researcher degrees of freedom)을 낳는다.

### 7-5. Table 13 (p. 146) — 실제 정책에 쓰인 사례

Geiser & Studley(2002) UC 신입생 자료, $n=77{,}893$

| 부분집합 | $R^2$ |
|---|---|
| {} | 0 |
| {HSGPA} | .154 |
| {SAT I} | .133 |
| {SAT II} | .160 |
| {HSGPA, SAT I} | .208 |
| {HSGPA, SAT II} | .222 |
| {SAT I, SAT II} | .162 |
| 전체 | .223 |

| 조건부 $k$ | HSGPA | SAT I | SAT II |
|---|---|---|---|
| 0 | .154 | .133 | **.160** |
| 1 | **.068** | .028 | .048 |
| 2 | **.061** | .001 | .015 |
| **일반** | **.094** | .054 | .074 |

**해석 4가지.**
1. **재계산 검증**: 표 13의 $R^2$만으로 전체 DA를 복원했고 HSGPA .0945 / SAT I .0540 / SAT II .0745, 합 .2230 = 전체 $R^2$로 정확히 일치했다. **논문에서 산술적으로 가장 신뢰할 수 있는 표다.**
2. **왜 HSGPA가 SAT II를 완전지배하지 **못**하는가**: $k=0$에서 SAT II(.160)가 HSGPA(.154)보다 높기 때문이다. 단독 예측력은 SAT II가 낫지만, 다른 변수가 통제되면 HSGPA가 이긴다. **DA의 세 수준이 실제로 유용한 정보 차이를 만든 유일한 실증 사례**다.
3. **SAT I의 붕괴**: SAT I은 단독으로는 $R^2=.133$으로 나쁘지 않지만, 다른 둘이 있으면 한계기여가 .001로 사라진다. SAT I이 HSGPA·SAT II와 **중복 정보**임을 보여준다. 논문은 3문단짜리 원저자 결론을 한 문장으로 압축했다고 자평한다(p. 145).
4. **⚠️ 심각한 한계**: (a) 부트스트랩·SE **미보고** — 이 결과가 SAT 개편 논거로 인용되었다는데도 그렇다. (b) $n$이 매우 커서 통계적 유의성은 자동 확보되지만 **실질적 차이는 작다**(전체 $R^2$의 3.4%p). (c) 캠퍼스 **군집 구조** 미고려. (d) 전체 설명력이 $R^2=.223$에 불과해 — **어떤 변수가 지배하든 대학 성적 분산의 78%는 설명되지 않는다.** 지배 순위는 "남은 22% 안에서의 순위"다.

---

## 8. 결론

### 8-1. 저자가 제시한 시사점

| 시사점 | 위치 |
|---|---|
| 다른 지표들은 통계량에서 출발해 해석을 "추출"하지만, DA는 **중요도 정의에서 출발**해 측도를 도출한다 | p. 145 |
| **보고 지침**: 가능하면 완전지배를 먼저 보고하라. 판정 불가면 그 사실 자체가 유용한 정보이므로 보고하라. 그 후 선택적으로 약한 수준으로 내려가라. | p. 145 |
| 도시 기온 비유: (a) 매일 더움 = 완전, (b) 매월 평균 더움 = 조건부, (c) 연평균 더움 = 일반 | p. 145 |
| DA는 기존 접근들의 **일반화**다: null 모형 행 = 단순 상관, $k=p-1$ 행 = 준편상관/ $t$검정, 제약 DA = 특정 맥락 비교 | p. 145 |
| **명시적 금지**: 인과모형, 매개/간접효과, 위계적 진입 순서 질문에는 부적합 | p. 146 |
| DA는 불량 데이터 문제를 해결하지 못한다 | p. 146 |

### 8-2. 저자가 밝힌 후속 연구 계획

논문은 별도의 "Future Work" 절을 두지 않으며, 계획은 결론부에 흩어져 있다(p. 147):
1. SSE 기반의 **모든 적합도 지표**로의 일반화 (개념적으로 단순하다고 서술)
2. **일반선형모형(GLM)** 전반으로의 확장
3. Whittaker(1984), Chevan & Sutherland(1991)의 hierarchical partitioning과의 통합
4. 정성적 판정(누가 더 중요한가)을 넘어선 **정량적 우위 크기(magnitude of advantage)** 측정 — 저자들이 명시적으로 "이 논문은 정성적 판정에 초점"이라 밝히며 남긴 과제

### 8-3. 모델의 일반화 성능 향상 가능성 (요청 8-1 — 중점)

이 논문의 **가장 큰 미개척 영역**입니다. 논문은 "일반화" 문제를 사실상 다루지 않습니다.

**문제의 정확한 진단.**

DA의 모든 것이 **표본 내(in-sample) $R^2$** 위에 세워져 있습니다. 그런데 $R^2$는 예측변수 수에 대해 **단조 증가**하며 상향 편의를 갖습니다:

$$E[R^2] \approx \rho^2 + \frac{(1-\rho^2)\,k}{n-k-1}$$

따라서 추가기여도 $C_{X_i}(\mathbf{X}_h)$는 **참값이 0인 경우에도 기대값이 양수**입니다. $n$이 작고 $p$가 클수록 이 편의가 커지며, $2^p-1$개 모형에 걸쳐 반복 비교하면 편의가 체계적으로 누적됩니다. 표 9에서 존재하지 않는 완전지배 2건이 발생한 것이 바로 이 현상의 발현입니다.

**개선 방향 5가지 (내 제안).**

**(1) 교차검증 기반 지배분석 (CV-DA).** $R^2$를 표본 외 결정계수로 교체:

$$C^{\text{CV}}_{X_i}(\mathbf{X}_h)=R^2_{\text{CV}}(\mathbf{X}_h\cup\{X_i\})-R^2_{\text{CV}}(\mathbf{X}_h)$$

$$R^2_{\text{CV}}=1-\frac{\sum_{j=1}^{n}\left(y_j-\hat{y}^{(-f(j))}_j\right)^2}{\sum_{j=1}^{n}(y_j-\bar{y})^2}$$

여기서 $\hat{y}^{(-f(j))}\_j$는 $j$가 속한 폴드를 제외하고 적합한 모형의 예측값입니다. 이 정의에서는 $C^{\text{CV}}$가 **음수가 될 수 있으며**, 이는 결함이 아니라 "그 변수를 넣으면 예측이 나빠진다"는 실질적 정보입니다. 다만 가법성 $\sum_i \bar{C}_{X_i}=R^2$는 **깨집니다** — 저자들이 가장 아끼는 성질과 일반화 성능이 상충하는 구조적 긴장입니다.

**(2) 조정 지표 사용.** 논문 자신의 C8 정리(SSE 단조함수 불변성)를 활용하면, adjusted $R^2$나 AIC로 바꿔도 **지배 패턴은 동일**합니다. 왜냐하면 모든 비교가 동일 크기 모형 간에 이루어져 벌점항이 상쇄되기 때문입니다. 즉 **단순 벌점화로는 일반화 문제가 해결되지 않습니다.** 이 점을 명확히 인식하는 것이 중요합니다 — 진짜 해법은 (1)의 표본 외 평가뿐입니다.

**(3) $\bar{C}_{X_i}$에 대한 부트스트랩 신뢰구간.** 이진 $D_{ij}$가 아니라 **연속 가중치 자체**를 부트스트랩하여 BCa 구간을 산출:

```math
\hat{\theta}^{*}_{ij}=\bar{C}^{*}_{X_i}-\bar{C}^{*}_{X_j},\qquad \text{BCa } 95\%\text{ CI for }\theta_{ij}
```

구간이 0을 포함하면 "순위 미결정"으로 보고. 현행 재현성 지표보다 훨씬 방어 가능한 추론입니다.

**(4) .632+ 부트스트랩 / 중첩 리샘플링.** 표본 내 낙관 편의를 명시적으로 보정하는 Efron–Tibshirani .632+ 추정량을 $R^2$ 자리에 넣는 방법.

**(5) $p$ 확장을 위한 몬테카를로 순열 근사.** $2^p$ 열거 대신 $p!$개 순열 중 $M$개를 무작위 추출하여 Shapley 값을 추정하면 $p>10$도 처리 가능합니다:

$$\hat{\bar{C}}_{X_i}=\frac{1}{M}\sum_{m=1}^{M}\Big[R^2\big(\mathbf{P}^{(m)}_{<i}\cup\{X_i\}\big)-R^2\big(\mathbf{P}^{(m)}_{<i}\big)\Big]$$

$\mathbf{P}^{(m)}_{<i}$는 $m$번째 무작위 순열에서 $X_i$ 앞에 오는 변수 집합. 이것이 사실상 SHAP의 표본 추정 방식이며, 논문의 10변수 제한을 우회하는 표준 경로입니다.

**(6) 안정성 선택과의 결합.** 모형선택 불확실성(취약점 5-7)을 다루려면, 부트스트랩 재표본마다 변수 선택 + DA를 함께 수행하여 **선택-포함 지배 분포**를 얻는 방식이 필요합니다.

### 8-4. 2020년 이후 관련 최신 연구 비교 (요청 8-2)

> ⚠️ **필수 고지**: 이 세션에서는 웹 검색을 사용할 수 없어 아래 문헌을 **온라인으로 검증하지 못했습니다.** 각 항목에 제 신뢰도를 표기했으며, 논문에 인용하기 전 반드시 직접 확인하시기 바랍니다. 확신이 낮은 항목은 아예 서술을 최소화했습니다.

#### (가) 계보: DA → Shapley/SHAP

이것이 2020년 이후 가장 중요한 흐름입니다. §2-1(B)에서 보인 대로 **일반지배 가중치는 $v(\mathbf{S})=R^2_{\mathbf{S}}$ 게임의 Shapley 값과 대수적으로 동일**합니다. 이 연결은 다음 경로로 확립되었습니다:

| 시기 | 문헌 | 역할 | 신뢰도 |
|---|---|---|---|
| 2001 | Lipovetsky, S., & Conklin, M. "Analysis of regression in game theory approach," *Applied Stochastic Models in Business and Industry* | 회귀 $R^2$ 분해 = Shapley 값임을 명시 | 높음 |
| 2007 | Grömping, U. "Estimators of relative importance in linear regression based on variance decomposition," *The American Statistician*, 61(2) | LMG/PMVD/relative weight 비교, `relaimpo` R 패키지 | 높음 |
| 2017 | Lundberg, S., & Lee, S.-I. "A unified approach to interpreting model predictions," *NeurIPS* | SHAP — 동일 수학을 임의 ML 모형의 **개별 예측** 수준으로 이식 | 높음 |
| 2020 | Covert, I., Lundberg, S., & Lee, S.-I. "Understanding global feature contributions with additive importance measures" (**SAGE**), *NeurIPS 2020* | **전역(global) 중요도**를 Shapley로 정의 — 일반지배와 개념적으로 가장 가까운 현대 ML 대응물 | 중간–높음 |
| 2020 | Kumar, I. E., Venkatasubramanian, S., Scheidegger, C., & Friedler, S. "Problems with Shapley-value-based explanations as feature importance measures," *ICML 2020* | Shapley 기반 중요도의 **개념적 한계** 비판 — Azen & Budescu의 미해결 문제(§6-C 12번, 인과 해석)와 정확히 같은 지점 | 중간–높음 |
| 2021 | Covert, I., Lundberg, S., & Lee, S.-I. "Explaining by removing: A unified framework for model explanation," *JMLR* | 제거 기반 설명의 통합 프레임 — DA의 "추가기여도"가 이 프레임의 특수 사례 | 중간 |
| 2020 | Williamson, B., & Feng, J. "Efficient nonparametric statistical inference on population feature importance using Shapley values," *ICML 2020* | **Shapley 중요도에 대한 신뢰구간·검정** — Azen & Budescu의 취약점 5-1을 정면으로 해결한 현대적 답안 | 중간 |
| 2022 | Rozemberczki, B., et al. "The Shapley value in machine learning," *IJCAI 2022* (survey) | 분야 개관 | 중간 |

**비교 분석: Azen & Budescu(2003) vs 현대 Shapley 계열**

| 축 | Azen & Budescu (2003) | SHAP / SAGE 계열 (2017–) |
|---|---|---|
| 대상 모형 | 선형 다중회귀 (표준화) | 임의의 모형 (트리, 신경망 등) |
| 값 함수 | $v(\mathbf{S})=R^2_{\mathbf{S}}$ (재적합) | 재적합(SAGE) 또는 조건부/주변 기대값(SHAP) |
| 해상도 | **전역, 쌍별** | 전역(SAGE) + **개별 관측치별**(SHAP) |
| 세분화 | **3수준 위계(완전/조건부/일반)** ← **독창적 기여** | 단일 수준(Shapley 값 하나)만 존재 |
| 계산 | $2^p$ 완전 열거, $p\le 10$ | 몬테카를로/커널/TreeSHAP 근사, $p$ 수백~수천 |
| 추론 | 부트스트랩 재현성 (통계적으로 취약) | Williamson & Feng(2020) 등 정식 CI |
| 일반화 | 표본 내 $R^2$만 | SAGE는 손실 기반, 홀드아웃 적용 가능 |

**핵심 통찰**: **완전지배·조건부지배는 현대 ML 설명가능성 문헌에 대응물이 거의 없습니다.** SHAP은 항상 하나의 스칼라(=일반지배에 해당)로 요약하므로, 표 3의 $X_4$ 사례($k=0$에서 2위, $k=3$에서 4위)와 같은 **맥락 의존적 순위 역전을 구조적으로 은폐**합니다. Azen & Budescu의 위계가 오히려 현대 XAI에 역수출될 가치가 있는 아이디어입니다.

#### (나) DA 자체의 직계 확장

| 문헌 | 내용 | 신뢰도 |
|---|---|---|
| Azen, R., & Traxel, N. (2009). "Using dominance analysis to determine predictor importance in logistic regression," *Journal of Educational and Behavioral Statistics*, 34(3) | 의사 $R^2$ 기반 로지스틱 DA | 높음 |
| Luo, W., & Azen, R. (2013). "Determining predictor importance in hierarchical linear models using dominance analysis," *JEBS*, 38(1) | 다층모형 DA (취약점 5-6 대응) | 높음 |
| Budescu, D. V., & Azen, R. (2004). "Beyond global measures of relative importance: Some insights from dominance analysis," *Organizational Research Methods*, 7(3) | 후속 개념 정리 | 중간–높음 |
| Braun, M. T., Converse, P. D., & Oswald, F. L. (2019). "The accuracy of dominance analysis as a metric to assess relative importance: The joint impact of sampling error variance and measurement unreliability," *Journal of Applied Psychology* | **표집오차 + 측정 비신뢰도가 DA 정확도에 미치는 영향을 시뮬레이션**. 취약점 5-4와 §6-C 13번을 직접 다룬 연구 | 중간–높음 |
| Luchman, J. N. (2021). "Determining relative importance in Stata using dominance analysis," *The Stata Journal*, 21(2) | Stata `domin` 명령어, GLM·다범주 종속변수로 확장 | 중간 |
| Bustos Navarrete, C., & Coutinho Soares, F. — `dominanceanalysis` R 패키지 (CRAN) | lm/glm/lmer 지원, 부트스트랩 포함 | 중간 |
| `domir` R 패키지 (Luchman) — 모형 비특정(model-agnostic) DA | 임의 적합도 함수 지원 | **낮음 — 미검증** |
| Nimon, K., & Oswald, F. L. (2013). "Understanding the results of multiple linear regression: Beyond standardized regression coefficients," *ORM* | commonality/DA/relative weight 통합 해설 | 중간–높음 |
| Thomas, D. R., Zumbo, B. D., Kwan, E., & Schweitzer, L. (2014). "On Johnson's (2000) relative weights method...," *Psychometrika* | **relative weight 비판** — 논문의 C7 주장(일반지배 ≈ relative weight)에 대한 반론 | 중간 |

#### (다) 이 논문이 후속 연구에 미친 영향

1. **심리학·경영학의 표준 실무 정착.** `relaimpo`, `yhat`, `dominanceanalysis`(R), `domin`(Stata), Johnson의 relative weight(SPSS 매크로) 등 도구 생태계의 개념적 기초가 되었습니다.
2. **"상대적 중요도"의 정의 논쟁 종결에 기여.** 이후 문헌은 "어떤 지표가 옳은가"보다 "어떤 연구질문에 어떤 지표가 맞는가"로 프레임이 이동했는데, 이는 이 논문의 "level of analysis" 개념(p. 133)의 직접적 영향입니다.
3. **정책 영향.** UC의 SAT 재분석(Table 13)이 SAT 개편 논의에 인용되었다는 서술(p. 146)은 방법론 논문이 정책에 닿은 드문 사례입니다. 동시에 §7-5의 한계들 때문에 **경계 사례**로도 읽혀야 합니다.
4. **XAI와의 수렴.** 2003년의 회귀 분산분해 문제가 2020년대에 SHAP/SAGE로 재발견되었다는 사실 자체가, 이 논문이 **모형 비특정 특성 중요도 문제의 초기 정식화**였음을 보여줍니다.

#### (라) 앞으로 연구 시 고려할 점 (실무 체크리스트)

1. **세 수준을 모두 보고하되 사전에 정하라.** 완전 → 조건부 → 일반 순으로 보고하고, 미판정도 결과로 보고할 것(저자 권고, p. 145). 세 수준을 다 돌린 뒤 유리한 것만 고르는 것은 p-hacking과 동형입니다.
2. **부트스트랩 재현성을 "신뢰수준"이라 부르지 말 것.** "표본 결과의 내적 재현율"로 서술하고, 가능하면 $\bar{C}_{X_i}$ 차이에 대한 BCa 신뢰구간을 함께 제시하십시오.
3. **$n/p$ 비율을 명시하고 $R^2$ 편의를 보고하라.** $p\ge 6$이고 $n$이 수백 이하면 표 9식 과잉선언 위험이 큽니다. 교차검증 기반 재계산을 민감도 분석으로 붙이십시오.
4. **군집·다층 구조가 있으면 Luo & Azen(2013) 계열 방법 또는 군집 부트스트랩을 사용하라.**
5. **측정 신뢰도를 보고하라.** Braun et al.(2019)이 보인 대로 비신뢰도가 순위를 체계적으로 왜곡합니다. 신뢰도가 다른 변수 간 순위 비교는 그 자체로 편향됩니다.
6. **인과적 언어를 쓰지 말라.** "가장 중요한 예측변수"는 "개입해야 할 지점"이 아닙니다. 논문 스스로 매개·경로모형 부적합을 선언했습니다(p. 146).
7. **전체 $R^2$를 반드시 함께 보고하라.** 표 13처럼 $R^2=.223$인 상황에서의 1위는 "설명되지 않는 78% 밖의 1위"입니다. 표 11이 $R^2$를 누락한 것은 반면교사입니다.
8. **$p>10$이면 순열 몬테카를로 근사를 쓰고 근사오차를 보고하라.**
9. **ML 맥락이라면 SAGE/SHAP과의 등가성을 명시하라.** 같은 수학을 다른 이름으로 재발명하지 않도록, 그리고 3수준 위계라는 DA 고유의 추가 정보를 활용하도록.

---

## 9. 참고자료 목록

### 9-1. 1차 자료 (직접 확인)
- Azen, R., & Budescu, D. V. (2003). The dominance analysis approach for comparing predictors in multiple regression. *Psychological Methods*, 8(2), 129–148. **(업로드된 PDF 전문 + 보충자료 SAS 매크로 `%dom`)**

### 9-2. 논문이 인용한 문헌 중 본 분석에서 언급한 것 (해당 논문 pp. 147–148 참고문헌 목록에서 확인)
- Azen, R. (2000). *Inference for predictor comparisons: Dominance analysis and the distribution of R² differences.* Dissertation Abstracts International B, 61/10, 5616.
- Azen, R., Budescu, D. V., & Reiser, B. (2001). Criticality of predictors in multiple regression. *British Journal of Mathematical and Statistical Psychology*, 54, 201–225.
- Barnes, J. E. (2002, Nov 11). The SAT revolution. *U.S. News & World Report*, 133, 51–60.
- Bring, J. (1994). How to standardize regression coefficients. *The American Statistician*, 48, 209–213.
- Budescu, D. V. (1993). Dominance analysis: A new approach to the problem of relative importance of predictors in multiple regression. *Psychological Bulletin*, 114, 542–551.
- Chevan, A., & Sutherland, M. (1991). Hierarchical partitioning. *The American Statistician*, 45, 90–96.
- Cooper, W. H., & Richardson, A. J. (1986). Unfair comparisons. *Journal of Applied Psychology*, 71, 179–184.
- Courville, T., & Thompson, B. (2001). Use of structure coefficients in published multiple regression articles: β is not enough. *Educational and Psychological Measurement*, 61, 229–248.
- Efron, B. (1979). Bootstrap methods: Another look at the jackknife. *The Annals of Statistics*, 7, 1–26.
- Geiser, S., & Studley, R. (2002). UC and the SAT: Predictive validity and differential impact of the SAT I and SAT II at the University of California. *Educational Assessment*, 8, 1–26. **(Table 13의 원자료)**
- Green, P. E., Carroll, D., & DeSarbo, W. (1978). A new measure of predictor variable importance in multiple regression. *Journal of Marketing Research*, 15, 356–360.
- Johnson, J. W. (2000). A heuristic method for estimating the relative weight of predictor variables in multiple regression. *Multivariate Behavioral Research*, 35, 1–19.
- Kruskal, W. (1987). Relative importance by averaging over orderings. *The American Statistician*, 41, 6–10.
- Lindeman, R. H., Merenda, P. F., & Gold, R. Z. (1980). *Introduction to bivariate and multivariate analysis.* Scott Foresman.
- Pratt, J. W. (1987). Dividing the indivisible: Using simple symmetry to partition variance explained. In *Proceedings of the Second International Tampere Conference in Statistics* (pp. 245–260).
- Suh, E., Diener, E., Oishi, S., & Triandis, H. C. (1998). The shifting basis of life satisfaction judgments across cultures. *JPSP*, 74, 482–493. **(Tables 10–12의 원자료)**
- Theil, H. (1987). How many bits of information does an independent variable yield in a multiple regression? *Statistics & Probability Letters*, 6, 107–108.
- Thompson, B. (1994). The pivotal role of replication in psychological research. *Journal of Personality*, 62, 157–176. **("internal replicability" 개념 출처)**
- Tzelgov, J., & Henik, A. (1991). Suppression situations in psychological research. *Psychological Bulletin*, 109, 524–536. **(Table 5–6의 억제변수 유형론)**
- Whittaker, J. (1984). Model interpretation from the additive elements of the likelihood function. *Applied Statistics*, 33, 52–64.

### 9-3. §8-2에서 언급한 논문 외부 문헌 — ⚠️ 웹 검증 불가
** 신뢰도 표기를 참고하시고, 인용 전 직접 검증하십시오.

| 신뢰도 | 문헌 |
|---|---|
| 높음 | Lipovetsky, S., & Conklin, M. (2001). Analysis of regression in game theory approach. *Applied Stochastic Models in Business and Industry*, 17(4). |
| 높음 | Grömping, U. (2007). Estimators of relative importance in linear regression based on variance decomposition. *The American Statistician*, 61(2). |
| 높음 | Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS 2017*. |
| 높음 | Azen, R., & Traxel, N. (2009). Using dominance analysis to determine predictor importance in logistic regression. *JEBS*, 34(3). |
| 높음 | Luo, W., & Azen, R. (2013). Determining predictor importance in HLM using dominance analysis. *JEBS*, 38(1). |
| 중간–높음 | Covert, I., Lundberg, S., & Lee, S.-I. (2020). Understanding global feature contributions with additive importance measures (SAGE). *NeurIPS 2020*. |
| 중간–높음 | Kumar, I. E., et al. (2020). Problems with Shapley-value-based explanations as feature importance measures. *ICML 2020*. |
| 중간–높음 | Braun, M. T., Converse, P. D., & Oswald, F. L. (2019). The accuracy of dominance analysis as a metric to assess relative importance. *Journal of Applied Psychology*. |
| 중간–높음 | Budescu, D. V., & Azen, R. (2004). Beyond global measures of relative importance. *Organizational Research Methods*, 7(3). |
| 중간–높음 | Nimon, K., & Oswald, F. L. (2013). Understanding the results of multiple linear regression. *ORM*. |
| 중간 | Covert, I., Lundberg, S., & Lee, S.-I. (2021). Explaining by removing: A unified framework for model explanation. *JMLR*. |
| 중간 | Williamson, B., & Feng, J. (2020). Efficient nonparametric statistical inference on population feature importance using Shapley values. *ICML 2020*. |
| 중간 | Rozemberczki, B., et al. (2022). The Shapley value in machine learning. *IJCAI 2022*. |
| 중간 | Luchman, J. N. (2021). Determining relative importance in Stata using dominance analysis. *The Stata Journal*, 21(2). |
| 중간 | Bustos Navarrete, C., & Coutinho Soares, F. `dominanceanalysis` R package (CRAN). |
| 중간 | Thomas, D. R., Zumbo, B. D., Kwan, E., & Schweitzer, L. (2014). On Johnson's (2000) relative weights method. *Psychometrika*. |
| **낮음 — 미검증** | `domir` R package (Luchman). 존재는 기억하나 저자·연도·문헌 정보를 확신할 수 없어 인용 근거로 삼지 마십시오. |
