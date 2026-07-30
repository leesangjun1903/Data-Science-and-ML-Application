# Relative Importance for Linear Regression in R: The Package relaimpo
**저자:** Ulrike Grömping | **출처:** Journal of Statistical Software, Vol. 17, Issue 1, October 2006

---

## 1. Executive Summary (10문장 이내)

1. 본 논문은 다중 선형 회귀에서 **상관된 예측 변수들 간의 상대적 중요도(Relative Importance)를 정량화**하는 R 패키지 `relaimpo`를 소개한다.
2. 예측 변수들이 비상관(uncorrelated)일 때 $R^2$ 분해는 자명하나, 관측 데이터에서 흔히 발생하는 **다중공선성(multicollinearity)** 상황에서는 이 분해가 불명확해진다.
3. `relaimpo`는 `lmg`, `pmvd`, `first`, `last`, `betasq`, `pratt` 등 **6가지 상대적 중요도 지표**를 구현한다.
4. 저자는 $R^2$를 비음수(non-negative) 기여분으로 자연스럽게 분해하는 **`lmg`와 `pmvd` 두 지표를 권장**한다.
5. `lmg`는 모든 변수 투입 순서에 대한 순차적 $R^2$의 **단순 평균**이며, `pmvd`는 데이터 의존적 가중치를 사용하는 **가중 평균**이다.
6. `pmvd`는 계수가 0인 변수에 중요도 0을 할당하는 **"배제(exclusion)" 성질**을 보장하나, 이로 인해 추정의 변동성이 크다.
7. 패키지는 상대적 중요도 추정치의 불확실성 평가를 위해 **탐색적 부트스트랩 신뢰구간**을 제공한다.
8. 공분산 행렬 기반 계산으로 `hier.part` 대비 **계산 속도가 대폭 향상**되었으며, 표본 크기와 무관하게 일정한 계산 시간을 유지한다.
9. 분석은 47개 프랑스어권 스위스 주의 출산율 데이터(`swiss`)를 사례로 각 지표의 차이를 시연한다.
10. 향후 **그룹화된 변수 처리, 교호작용 항 지원, 관측 가중치 적용** 등의 확장이 계획되어 있다.

### 1-1. 연구의 목적과 필요성

**목적:** 상관된 예측 변수를 포함한 선형 회귀 모형에서 각 변수의 $R^2$ 기여분을 정량적·신뢰성 있게 분해하는 R 소프트웨어 도구를 제공하는 것.

**필요성:**
- 관측 데이터(사회과학, 의학, 마케팅 등)에서 예측 변수 간 상관관계는 불가피하며, 이 경우 단순 $R^2$ 분해는 순서 의존적 결과를 낳는다 (p.1).
- 기존 패키지 `hier.part`는 `lmg`만 제공하며 계산 속도가 느리고 신뢰구간을 제공하지 않는다 (p.2, Section 6).
- 새로운 지표 `pmvd` (Feldman 2005)의 R 구현이 최초로 필요했다 (p.2).
- Johnson & Lebreton (2004)의 정의—\*"직접 효과와 다른 변수들과의 조합 효과를 모두 고려한 $R^2$에 대한 비례적 기여"*—를 충족하는 지표 구현이 필요하다 (p.2).

---

## 2. 핵심 주장과 근거 표

| 주장 | 근거 / 방법 | 위치 |
|------|------------|------|
| 상관 변수 존재 시 단일 순서 기반 $R^2$ 분해는 부적절 | `swiss` 데이터에서 순서에 따라 Examination의 기여가 최대~최소로 역전됨 | p.8, anova 비교 |
| `first`는 상대적 중요도 지표로 부적절 | 다른 변수를 무시하고 직접 효과만 반영; 기여분 합이 전체 $R^2$를 초과 | p.5 |
| `last`는 유의성 검정과 동일; 부적절 | t-검정 통계량과 동치 관계; 직접 효과 미반영 | p.6 |
| `betasq`는 $R^2$ 분해에 부적절 | 자연스러운 $R^2$ 분해를 제공하지 못함 | p.7 |
| `pratt`는 음수 기여 발생으로 일부 상황에 적용 불가 | Agriculture에 음수(−0.110) 할당됨 | p.7 |
| `lmg`와 `pmvd`가 권장됨 | 비음수이며 $R^2$로 자동 합산; Johnson & Lebreton 정의 충족 | p.9–11 |
| `relaimpo`가 `hier.part`보다 빠름 | 공분산 행렬 기반 계산; $p=12$에서 4.93초 vs. 82.42초 | p.24, Table 2 |
| 부트스트랩 신뢰구간 필요 | `lmg`, `pmvd`에 대한 분포 이론적 결과 미확립 | p.14 |
| `pmvd`는 `lmg`보다 변동성 큼 | 신뢰구간이 더 넓음 (Figure 2, p.22) | p.11, p.18 |
| 변수 그룹화 등 향후 확장 필요 | 30개 이상 변수에서 현실적 계산 불가 | p.25 |

---

## 2-1. 해결 문제, 제안 방법, 모델 구조, 성능, 한계

### 해결하고자 하는 문제

상관된 예측 변수가 있는 선형 회귀에서 $R^2$를 각 변수의 기여분으로 **유일하고 의미 있게 분해하는 방법의 부재**.

$$y_i = \beta_0 + x_{i1}\beta_1 + \cdots + x_{ip}\beta_p + e_i \quad \cdots (1)$$

$$R^2 = \frac{\text{Model SS}}{\text{Total SS}} = \frac{\sum_{i=1}^{n}(\hat{y}_i - \bar{y})^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2} \quad \cdots (2)$$

변수 투입 순서에 따라 순차적 $R^2$ 기여분이 극단적으로 달라지는 문제 (p.8):

$$\text{seqR}^2(\{x_k\}|S_k(r)) = R^2(\{x_k\} \cup S_k(r)) - R^2(S_k(r)) \quad \cdots (6)$$

### 제안하는 방법 (수식 포함)

**① `lmg` (Lindeman, Merenda & Gold, 1980)** — 모든 순열에 대한 단순 평균:

$$\text{LMG}(x_k) = \frac{1}{p!} \sum_{r \in \text{permutations}} \text{seqR}^2(\{x_k\}|r) \quad \cdots (7)$$

동치 표현 (Christensen 1992):

$$\text{LMG}(x_k) = \frac{1}{p}\sum_{j=0}^{p-1} \left( \sum_{\substack{S \subseteq \{x_1,\ldots,x_p\}\setminus\{x_k\} \\ n(S)=j}} \frac{\text{seqR}^2(\{x_k\}|S)}{\binom{p-1}{j}} \right)$$

**② `pmvd` (Feldman, 2005)** — 데이터 의존적 가중 평균:

$$\text{PMVD}(x_k) = \frac{1}{p!} \sum_{r \in \text{permutations}} p(r) \cdot \text{seqR}^2(\{x_k\}|r) \quad \cdots (8)$$

가중치:

$$L(r) = \prod_{i=1}^{p-1} \left[\text{seqR}^2(\{x_{r_{i+1}},\ldots,x_{r_p}\}|\{x_{r_1},\ldots,x_{r_i}\})\right]^{-1}$$

$$p(r) = \frac{L(r)}{\sum_{r' \in \text{permutations}} L(r')}$$

**③ 단순 지표들:**

- `first`: $r_k^2 = \text{cor}(x_k, y)^2$ (단변량 $R^2$)
- `last`: type III SS / Total SS (다른 모든 변수 포함 후 추가 기여)
- `betasq`: $\hat{\beta}\_{k,\text{std}}^2 = \left(\hat{\beta}\_k \frac{\sqrt{s_{kk}}}{\sqrt{s_{yy}}}\right)^2 \quad \cdots (3)$
- `pratt`: $\hat{\beta}\_{k,\text{std}} \times r_{ky}$ (Hoffman 1960, Pratt 1987 옹호)

**④ 공분산 행렬 기반 효율적 계산** (Section 3.3):

$$R^2 = \frac{\mathbf{S}_{yx}\mathbf{S}_{xx}^{-1}\mathbf{S}_{xy}}{s_{yy}} \quad \cdots (12)$$

$$\hat{\beta}_{1,\ldots,p} = \mathbf{S}_{xx}^{-1}\mathbf{S}_{xy}$$

### 모델 구조

```
입력: 선형 회귀 모형 객체 또는 공분산 행렬
  ↓
calc.relimp() → 6가지 지표 계산 (lmg, pmvd, first, last, betasq, pratt)
  ↓
boot.relimp() → 부트스트랩 재표본 (b=1000 기본)
  ↓
booteval.relimp() → 신뢰구간, 순위 CI, 차이 CI 산출
  ↓
plot() → 막대그래프 (Figure 1, 2)
```

### 성능 향상

| 변수 수 $p$ | `hier.part` (100obs) | `lmg` | `pmvd` |
|:-----------:|:--------------------:|:-----:|:------:|
| 5 | 0.53초 | 0.06초 | 0.05초 |
| 10 | 19.50초 | 1.25초 | 1.74초 |
| 12 | 82.42초 | 4.93초 | 11.64초 |

*(Table 2, p.24)*

### 한계

- **`pmvd` 높은 변동성**: 배제 성질 달성을 위한 가중치가 큰 변동성 초래 (p.11, Figure 2)
- **부트스트랩 신뢰구간의 자유성(liberalness)**: 퍼센타일 CI는 명목 수준의 최대 2배 비적용률 (p.17)
- **계산 확장성 한계**: `pmvd`는 $p=15$에서 약 525초 소요 (p.24)
- **선형 모형 전용**: 비선형 모형, GLM 등에 직접 적용 불가 (p.21)
- **`pmvd` 미국 특허 제한**: US 특허 6,640,204로 인해 미국 내 사용 제한 (p.10 각주, Appendix A)
- **인과 해석 한계**: `pmvd`의 배제 성질은 인과 중요도 관점에서 오히려 부적절할 수 있음 (p.11)

---

## 3. 각 주장에 페이지/Figure/Table 번호 표시

| 주장 | 위치 |
|------|------|
| 상관 변수 시 $R^2$ 분해 불명확 | p.1, Introduction |
| `swiss` 데이터 다중공선성 설명 | p.3, cor(swiss) 출력 |
| 선형 모형 수식 및 $R^2$ 정의 | p.3, Eq.(1)(2) |
| `first` 지표: 기여 합이 $R^2$ 초과 | p.5, Section 3.1 |
| `last` 지표: 유의성 검정과 동치 | p.6, Section 3.1 |
| `betasq` 정의 | p.6, Eq.(3) |
| `pratt` 음수 기여 문제 | p.7, Section 3.1 |
| 순서에 따른 $R^2$ 극단적 변화 | p.8, anova 비교 출력 |
| `lmg` 공식 | p.9, Eq.(7) |
| `pmvd` 공식 및 가중치 | p.10, Eq.(8) |
| `lmg` vs `pmvd` 비교 (Table 1) | p.12, Table 1 |
| 배제 성질의 적절성 논의 | p.11, "Is pmvd's property desirable?" |
| 공분산 행렬 기반 계산 | p.12–14, Section 3.3, Eq.(11)(12) |
| 부트스트랩 두 방식 비교 | p.14–15, Section 4 |
| `rela=TRUE` 옵션 | p.15, Section 5.1 |
| `always` 옵션 | p.16, Section 5.2 |
| 순위 신뢰구간 레터링 시스템 | p.17–18, Section 5.5 |
| 차이 신뢰구간 | p.18–20, Section 5.6 |
| 6개 지표 막대그래프 | p.21, Figure 1 |
| `lmg`, `pmvd` 신뢰구간 포함 그래프 | p.22, Figure 2 |
| `hier.part`와 비교 | p.21–23, Section 6 |
| 계산 시간 비교 | p.24, Table 2 |
| 향후 개발 계획 | p.25, Section 8 |

---

## 4. 연구 주제·방법·결과: 저자 보고 vs. 해석자 해석 분리

### 저자가 직접 보고한 결과

**주제:** `relaimpo` 패키지 소개 및 6가지 상대적 중요도 지표 비교 튜토리얼

**방법 (저자 보고):**
- `swiss` 데이터 ($n=47$, $p=5$) 사용, 전체 $R^2 = 70.67\%$ (p.4)
- `lmg`: 순열 평균, `pmvd`: 데이터 의존 가중 평균 (p.9–10, Eq.7, 8)
- 부트스트랩 $b=1000$회, BCa 및 퍼센타일 구간 (p.17)

**결과 (저자 보고):**
- Education이 `lmg`(26.0%), `pmvd`(38.0%)에서 가장 중요 (p.4)
- Examination은 `first`에서 2위(41.7%)이나 `last`에서 최하위(0.7%), `pmvd`에서 최하위권(4.4%) (p.4–5)
- Agriculture는 `pratt`에서 유일하게 음수(−0.110) (p.4)
- `relaimpo`가 `hier.part` 대비 $p=12$에서 약 16.7배 빠름 (p.24, Table 2)
- 부트스트랩 퍼센타일 CI는 명목 수준의 최대 2배 비적용률 발생 (p.17)

### 해석자(리뷰어) 해석

- Examination의 지표 간 극단적 순위 역전은 다중공선성의 심각성을 효과적으로 시연하나, **단일 데이터셋만으로는 일반화에 한계**가 있음.
- `pmvd`의 배제 성질이 항상 바람직하지 않다는 저자의 논의(p.11, 인과 사슬 예시)는 **지표 선택이 연구 목적(예측 vs. 인과)에 따라 달라져야 함**을 시사하며, 이는 단순 추천을 넘어 심층적 방법론적 논의가 필요함을 의미함.
- 공분산 행렬 기반 계산이 표본 크기 불변성을 달성한 점은 **알고리즘 설계 측면의 핵심 기여**이나, 이는 계산 복잡도가 $p$에만 의존함을 의미하므로 고차원($p > 20$) 상황에서는 여전히 실용적 한계 존재.
- 부트스트랩 신뢰구간이 "탐색적(exploratory)"으로만 권장되는 점은 **통계적 추론의 신뢰성 측면에서 중요한 제약**임.

---

## 5. 통계적으로 취약한 부분과 비교 불가능한 수치

### ⚠️ 통계적으로 취약한 부분

| 항목 | 취약점 | 위치 |
|------|--------|------|
| **부트스트랩 CI 자유성** | 퍼센타일 CI($b=1000$)의 비적용률이 명목 수준의 최대 2배; BCa CI의 성능은 시뮬레이션 미실시 | p.17 |
| **단일 데이터셋 시연** | `swiss` 데이터만 사용; 다양한 상관 구조 및 표본 크기에서의 체계적 검증 부재 | 전체 |
| **`pmvd`의 높은 변동성** | 배제 성질로 인한 광폭 CI; Examination.pmvd CI = [0.0005, 0.3438]로 사실상 무의미 | p.18–19 |
| **$n=47$ 소표본** | 47개 관측치에 5개 변수; 부트스트랩 안정성에 의문 | p.2 |
| **정규성 가정 미검증** | 부트스트랩은 이론적 분포를 대체하지만, 모형 가정(등분산, 정규성) 위반 시 결과 신뢰성 불명확 | p.15 |

### ⚠️ 비교 불가능한 수치

| 항목 | 이유 |
|------|------|
| **6개 지표의 절댓값 직접 비교** | 지표마다 합산 기준이 다름: `lmg`·`pmvd`는 $R^2$로 합산, `first`는 초과, `last`는 미달, `betasq`·`pratt`는 의미 없는 합 |
| **Table 2 계산 시간 (2006년 하드웨어)** | AMD Athlon XP 1700+ 기준; 현재 하드웨어 및 최적화 알고리즘 환경에서 재현 불가 |
| **`relaimpo` vs `hier.part` 계산 시간** | `hier.part`는 $2^p - 1$개 회귀 모형 계산 방식이 상이하여 알고리즘 복잡도 직접 비교 시 주의 필요 |
| **고정/무작위 부트스트랩 CI 비교** | 두 방식은 가정(고정 vs. 무작위 설명변수)이 다르므로 구간 폭만으로 우열 판단 불가 |

---

## 6. 문서가 답하지 않는 질문

1. **`lmg`와 `pmvd` 중 언제 어느 지표를 선택해야 하는가?** 저자는 둘 다 권장하며 비교를 제안하지만, 구체적 선택 기준(표본 크기, 상관 구조, 연구 목적별)은 미제시.

2. **다중공선성의 심각도가 지표 선택에 어떤 영향을 미치는가?** VIF, 조건 수 등과 지표의 신뢰성 간의 관계가 논의되지 않음.

3. **`pmvd`의 부트스트랩 신뢰구간 성능은?** BCa CI에 대한 시뮬레이션 연구가 수행되지 않았으며 (p.17), 이는 핵심 권장 지표에 대한 신뢰구간 신뢰성의 공백.

4. **비선형 관계나 이상치에 대한 강건성은?** 선형 모형 가정 위반 시 지표들의 동작 방식이 논의되지 않음.

5. **변수 선택(모형 불확실성)을 어떻게 통합할 것인가?** 최종 모형이 주어진 상태에서만 분석하며, 모형 선택 자체의 불확실성 전파는 다루지 않음.

6. **`pmvd` 미국 특허 만료 이후 완전한 오픈소스화 계획은?** US 특허 6,640,204의 만료 시점 및 향후 라이센싱 계획이 불명확 (Appendix A).

7. **그룹화된 변수 처리의 구체적 구현 시점 및 방법은?** 향후 계획으로만 언급 (p.25).

8. **부트스트랩 반복 수($b$)의 최적값은?** $b=1000$을 기본값으로 권장하나, `lmg`·`pmvd`에 대한 수렴 분석이 없음.

9. **`pmvd` 가중치의 직관적 해석 및 시각화 방법은?** Table 1에서 일부 제시되나, 일반적 해석 가이드라인 부재.

10. **다른 언어/소프트웨어 환경(Python, SAS 등)으로의 이식성은?** R 전용 패키지로 제한.

---

## 7. 가장 중요한 그림 5개의 해석

### Figure 1: 6개 지표 막대그래프 (p.21)

**내용:** `swiss` 데이터에 대해 `lmg`, `pmvd`, `last`, `first`, `betasq`, `pratt` 6개 지표를 나란히 시각화.

**해석:**
- **Education(Edu)**: 모든 지표에서 가장 높은 기여 → 지표 선택과 무관하게 가장 중요한 변수
- **Examination(Exa)**: `first`에서 최고(~42%), `last`·`pmvd`에서 최저(~0.7%, ~4.4%) → 다중공선성으로 인한 극단적 불안정성
- **Agriculture(Agr)**: `pratt`에서 유일하게 음수(~-11%) → `pratt` 지표의 적용 불가 상황을 명확히 시연
- **실용적 함의:** 단일 지표에만 의존하면 크게 오도될 수 있음을 시각적으로 증명

---

### Figure 2: `lmg`·`pmvd` 부트스트랩 신뢰구간 (p.22)

**내용:** `lmg`(좌)와 `pmvd`(우) 지표에 90% 부트스트랩 퍼센타일 CI 추가.

**해석:**
- **`lmg`**: Education이 명확하게 1위이나 나머지 4개 변수(Examination, Catholic, Infant.Mortality, Agriculture)의 CI가 상당 부분 겹침 → 정확한 순위 결정 어려움
- **`pmvd`**: Education의 CI가 매우 넓음(5.6%~57.6%, p.19) → 점 추정치(38.0%)에 비해 불확실성이 극도로 큼
- **`pmvd`의 Examination**: CI 하한이 거의 0에 가까움 → 배제 성질로 인한 불안정성
- **핵심 메시지:** `pmvd` CI가 `lmg` CI보다 체계적으로 넓음 → `pmvd`는 배제 성질을 얻는 대가로 추정 효율성을 희생

---

### Table 1: `lmg`·`pmvd` 비교 (p.12)

**내용:** Agriculture 또는 Examination을 제외한 4변수 모형 24가지 순열에 대한 순차적 $R^2$와 `pmvd` 가중치.

**해석:**
- **Agriculture 제외 모형**: `pmvd` 가중치가 "Examination 최후 투입" 순열에 집중(예: 순열 2341에 30.97%, 2431에 27.93%) → `pmvd`(Exam) = 1.17% vs `lmg`(Exam) = 18.03%로 극단적 차이 발생
- **Examination 제외 모형**: 가중치가 더 균등 분포 → `pmvd`와 `lmg` 차이 축소
- **방법론적 시사점:** `pmvd`의 배제 성질은 한 변수가 다른 변수를 통해 간접적으로 기여할 때 해당 기여를 무시함 → 인과 구조에서 `lmg`가 더 적절할 수 있음

---

### Table 2: 계산 시간 비교 (p.24)

**내용:** $p=3$ ~ $12$, 표본 100·1000에서 `hier.part`, `lmg`, `pmvd` 계산 시간 (초).

**해석:**
- **`relaimpo`의 표본 크기 불변성**: $p=5$에서 100obs와 1000obs 모두 `lmg`=0.06초, `pmvd`=0.05초 → 공분산 행렬 기반 계산의 핵심 장점
- **`hier.part`의 표본 크기 의존성**: $p=5$에서 0.53초(100obs) vs 1.24초(1000obs) → $2^p-1$개 전체 데이터 회귀 계산 때문
- **`pmvd`의 지수적 성장**: $p=10$에서 `lmg`의 1.4배이나 $p=12$에서 2.4배 → 수식 (8)의 단순화 한계
- **실용적 임계점**: $p \leq 9$ 권장(둘 다 1초 이내), $p \geq 12$에서 `pmvd` 사용 시 심각한 계산 부담

---

### `swiss` 상관행렬 출력 (p.3)

**내용:** Fertility와 5개 예측변수 간 상관 행렬.

**해석:**
- **Examination-Education 상관**: $r = 0.698$ → 높은 양의 상관으로 두 변수가 유사한 정보 공유
- **Examination-Agriculture 상관**: $r = -0.687$ → 강한 음의 상관
- **Fertility-Examination 상관**: $r = -0.646$이지만 회귀 모형에서 Examination은 비유의 → 다중공선성으로 인한 계수 불안정성의 전형적 사례
- **방법론적 의미:** 이 복잡한 상관 구조가 지표 간 극단적 불일치의 원인 → `swiss`가 상대적 중요도 방법 비교를 위한 이상적 예제임을 확인

---

## 8. 결론: 시사점, 후속 연구, 추가 방향

### 8-1. 저자 제시 시사점 및 후속 연구 계획 (p.24–25)

**저자 제시 시사점:**
- 상관 변수가 있을 때 상대적 중요도를 정량화하는 유일한 지표는 존재하지 않음
- `lmg`와 `pmvd`가 Johnson & Lebreton (2004) 정의에 가장 근접하며 권장됨
- 부트스트랩 CI는 "탐색적" 도구로, 과도한 해석을 방지하기 위해 필수적
- `relaimpo`는 `hier.part` 대비 계산 효율성과 기능성에서 우위

**저자 제시 후속 연구 계획:**
1. **변수 그룹화(grouped regressors)**: 다범주 요인 변수 및 대규모 예측 변수 집합 처리
2. **교호작용 항 지원**: 주효과보다 먼저 교호작용 투입 금지 등 사전 정의된 위계 구조 존중
3. **관측 가중치 적용**: 표본 설계 가중치를 반영한 복합 표본 분석 지원

### 모델 일반화 성능 향상 가능성 (8-1 중점)

이 논문은 예측 모형의 **일반화 성능**보다는 **설명 가능성(explainability)**에 초점을 두지만, 다음 관점에서 일반화 성능과의 연결이 가능하다:

**① 특성 선택(Feature Selection)으로의 활용:**
- `lmg` 기반 상대적 중요도는 실질적으로 기여가 낮은 변수를 식별하는 데 활용 가능
- 그러나 현재 패키지는 부분 집합 선택 후 $R^2$ 변화에 대한 안정성 분석을 제공하지 않음
- **개선 방향:** 교차 검증(cross-validation) 기반 `lmg` 추정으로 과적합 없는 중요도 측정

**② 모형 불확실성 전파:**
- 현재 `relaimpo`는 주어진 모형 구조 하에서 계수 불확실성만 부트스트랩으로 전파
- 모형 선택 불확실성(model selection uncertainty)이 중요도 추정에 미치는 영향 미반영
- **개선 방향:** Bayesian Model Averaging (BMA)과 통합하여 모형 불확실성 전파

**③ 고차원 상황에서의 안정성:**
- $p > n$ 상황(고차원)에서 `lmg` 계산이 불안정해질 수 있으나 논문에서 미논의
- **개선 방향:** Ridge, Lasso 등 정규화 회귀 기반 상대적 중요도 확장 (Grömping 2015 등 후속 연구)

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의:** 아래 2020년 이후 연구 동향은 필자의 배경 지식에 기반하며, 개별 논문의 세부 내용(특정 수치, 저자 수 등)에 대해 100% 정확성을 보장하기 어렵습니다. 검색 가능한 주요 방향만 제시합니다.

**① SHAP (SHapley Additive exPlanations)과의 관계:**

`lmg`의 핵심 공식인 Shapley value와 SHAP(Lundberg & Lee, 2017)은 수학적 기반을 공유한다:

$$\phi_k = \sum_{S \subseteq F \setminus \{k\}} \frac{|S|!(|F|-|S|-1)!}{|F|!} [f_{S \cup \{k\}}(x_{S \cup \{k\}}) - f_S(x_S)]$$

- `lmg`는 이 Shapley value를 $R^2$에 적용한 특수 사례
- **2020년 이후 SHAP의 폭발적 발전**: 비선형 모형(XGBoost, 신경망)으로의 확장, TreeSHAP, KernelSHAP 등
- **`relaimpo`와의 차이**: `relaimpo`는 선형 모형의 $R^2$ 기반이므로 모형 클래스에 제한적이나, SHAP는 모형 불가지론적(model-agnostic)

**② `dominanceanalysis` 패키지 (R):**
- Budescu (1993)의 지배 분석(dominance analysis)을 일반화
- GLM, 생존 분석, 혼합 모형 등 다양한 모형 클래스로 확장
- `lmg`와 개념적으로 유사하나 완전·조건부·일반 지배 관계를 통합 제공

**③ 머신러닝에서의 변수 중요도 통합:**
- Permutation Importance, Gradient-based Importance 등과 `lmg` 유사성에 대한 연구
- Hooker et al. (2021) 등 상관 변수 존재 시 순열 중요도의 편향 문제 논의 → `lmg`의 순열 평균 아이디어와 연결

**④ 고차원 설정에서의 확장:**
- $p \gg n$ 환경에서 Shapley 기반 중요도의 안정화: 정규화 Shapley value, 근사 알고리즘

---

### 해당 논문이 앞으로의 연구에 미치는 영향

1. **Shapley value 기반 XAI의 이론적 토대**: `lmg`가 게임 이론의 Shapley value와 동치임이 명시되면서, 통계학과 XAI 분야를 연결하는 가교 역할
2. **응용 연구에서의 다중 지표 비교 관행 확산**: 단일 지표가 아닌 다수 지표 비교를 통한 강건한 중요도 평가의 표준화
3. **소프트웨어 생태계 기여**: R의 상대적 중요도 분석 기반 마련, 이후 `iml`, `DALEX`, `vip` 등 패키지 개발에 영향

---

### 앞으로 연구 시 고려할 점

| 고려 사항 | 세부 내용 |
|----------|----------|
| **비선형 확장** | `lmg`를 GLM, 랜덤포레스트, 신경망에 일반화 (이미 SHAP이 일부 해결) |
| **인과 추론과의 통합** | DAG(Directed Acyclic Graph) 기반 인과 구조 하에서의 중요도 재정의 필요 |
| **고차원 효율화** | $p > 20$에서 Monte Carlo Shapley 근사로 계산 비용 절감 |
| **복합 표본 설계** | 층화·클러스터 표본에서 부트스트랩 설계 반영 필요 |
| **시간적 데이터** | 패널 데이터, 종단 데이터에서의 중요도 변화 추적 |
| **다중 응답 변수** | 다변량 회귀에서의 Shapley 기반 분해 확장 |
| **모형 불확실성** | 변수 선택 불확실성이 중요도 추정에 미치는 영향 정량화 |
| **상호작용 효과** | 교호작용 항을 포함한 위계적 중요도 분해 |

---

## 참고 자료 (논문 내 인용 문헌)

1. **Grömping, U. (2006).** "Relative Importance for Linear Regression in R: The Package relaimpo." *Journal of Statistical Software*, 17(1). *(본 논문)*
2. **Lindeman, R.H., Merenda, P.F., & Gold, R.Z. (1980).** *Introduction to Bivariate and Multivariate Analysis.* Scott, Foresman, Glenview, IL.
3. **Feldman, B. (2005).** "Relative Importance and Value." Manuscript version 1.1.
4. **Johnson, J.W., & Lebreton, J.M. (2004).** "History and Use of Relative Importance Indices in Organizational Research." *Organizational Research Methods*, 7, 238–257.
5. **Darlington, R.B. (1968).** "Multiple Regression in Psychological Research and Practice." *Psychological Bulletin*, 69, 161–182.
6. **Pratt, J.W. (1987).** "Dividing the Indivisible: Using Simple Symmetry to Partition Variance Explained." *Proceedings of Second Tampere Conference in Statistics*, pp. 245–260.
7. **Chevan, A., & Sutherland, M. (1991).** "Hierarchical Partitioning." *The American Statistician*, 45, 90–96.
8. **Budescu, D.V. (1993).** "Dominance Analysis: A New Approach to the Problem of Relative Importance of Predictors in Multiple Regression." *Psychological Bulletin*, 114, 542–551.
9. **Bring, J. (1996).** "A Geometric Approach to Compare Variables in a Regression Model." *The American Statistician*, 50, 57–62.
10. **Davison, A., & Hinkley, D.V. (1997).** *Bootstrap Methods and Their Application.* Cambridge University Press.
11. **Christensen, R. (1992).** "Comment on Chevan and Sutherland." *The American Statistician*, 46, 74.
12. **Achen, C.H. (1982).** *Interpreting and Using Regression.* Sage, Thousand Oaks, CA.
13. **Walsh, C., & Mac Nally, R. (2005).** `hier.part`: Hierarchical Partitioning. R package version 1.0-1.
14. **Hoffman, P.J. (1960).** "The Paramorphic Representation of Clinical Judgment." *Psychological Bulletin*, pp. 116–131.
15. **R Development Core Team (2006).** *R: A Language and Environment for Statistical Computing.* http://www.R-project.org/
