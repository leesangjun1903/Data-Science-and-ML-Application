# Variable Importance Assessment in Regression: Linear Regression versus Random Forest

---

## 1. Executive Summary (10문장 이내)

Grömping(2009)은 회귀 분석에서 변수 중요도(variable importance)를 평가하는 두 가지 접근법—선형 회귀 기반의 $R^2$ 분해 방법(LMG, PMVD)과 랜덤 포레스트 기반 방법(RF-CART, RF-CI)—을 체계적으로 비교한다. 선형 회귀에서 LMG는 모든 변수 순서의 평균 $R^2$ 기여분을 사용하고, PMVD는 강한 예측변수에 편향된 데이터 의존적 가중치를 부여한다. 랜덤 포레스트에서는 OOB(Out-of-Bag) 데이터의 퍼뮤테이션 기반 MSE 감소량을 변수 중요도로 사용한다. 스위스 출산율 데이터와 체계적 시뮬레이션 연구를 통해 네 가지 방법을 비교한 결과, RF-CART(mtry=1)는 LMG와 유사한 경향을 보이고, RF-CI는 mtry가 커질수록 PMVD에 가까워지는 경향이 있다. 변수 간 상관관계는 모든 방법에서 중요도 배분에 결정적 영향을 미치며, 특히 correlated regressors 상황에서 방법 간 차이가 두드러진다. RF-CI는 mtry 튜닝 파라미터에 매우 민감한 반면, RF-CART는 상대적으로 안정적이다. 논문은 변수 중요도가 설명적(explanatory) 목적과 예측적(predictive) 목적에 따라 다르게 해석되어야 함을 강조한다. 랜덤 포레스트의 변수 중요도는 $R^2$를 자연스럽게 분해하지 않으므로, 비교를 위해 정규화가 필요하다. 이 논문은 랜덤 포레스트 변수 중요도의 블랙박스적 특성을 선형 모델과의 비교를 통해 부분적으로 해명하지만, mtry와 상관구조의 상호작용 등 일부 현상은 여전히 미해결로 남긴다.

### 1-1. 연구의 목적과 필요성

**목적:** 선형 회귀의 $R^2$ 기반 변수 중요도 방법(LMG, PMVD)과 랜덤 포레스트의 OOB-MSE 퍼뮤테이션 기반 변수 중요도(RF-CART, RF-CI)를 체계적으로 비교하여, 특히 변수 간 상관관계가 중요도 배분에 미치는 영향을 규명한다.

**필요성:**
- 변수 중요도는 응용 통계에서 반복적으로 요구되는 핵심 주제이나, 아직 합의된 표준이 없다 (p.308).
- 랜덤 포레스트의 변수 중요도는 블랙박스적 성격으로, 그 동작 메커니즘이 충분히 이해되지 않았다.
- RF-CART의 impurity 기반 중요도는 편향이 알려져 있고(Strobl et al. 2007), 퍼뮤테이션 기반 대안의 특성 비교가 필요하다.
- 상관된 변수들 간의 중요도 배분 문제는 선형 모델에서도 랜덤 포레스트에서도 만족스럽게 해결되지 않은 상태다.

---

## 2. 핵심 주장과 근거 표

| # | 핵심 주장 | 근거/방법 | 위치 |
|---|-----------|-----------|------|
| 1 | RF-CART(mtry=1)의 변수 중요도는 LMG와 유사한 '평등화(equalizing)' 경향을 보인다 | 시뮬레이션 100회 평균 비교, $\beta_1=(4,1,1,0.3)^T$ 등 7가지 계수 벡터 | p.314, Figure 4 |
| 2 | RF-CI는 mtry 증가 시 PMVD에 수렴하는 경향이 있다 | mtry=1~4 시뮬레이션 비교 | p.314, Figures 5, 6 |
| 3 | RF-CI는 mtry에 민감하고, RF-CART는 상대적으로 안정적이다 | Table 1, Figures 5, 6 | p.312, Table 1 |
| 4 | 변수 간 상관관계는 모든 방법의 중요도 배분에 결정적 영향을 미친다 | 상관계수 $\rho= -0.9$ ~ $0.9$ 시나리오 | p.313, Section 6.1 |
| 5 | RF-CART의 Gini/impurity 기반 중요도는 편향되어 있으므로 퍼뮤테이션 기반 MSE 감소를 사용해야 한다 | Strobl et al.(2007) 인용, Section 4.2.3 | p.311, Section 4.2.3 |
| 6 | 변수 중요도의 목적(설명적 vs. 예측적)에 따라 적합한 방법이 다르다 | 인과 사슬 예시 $(X_2 \to X_1 \to Y)$ vs. $(X_2 \leftarrow X_1 \to Y)$ | p.317, Section 7 |
| 7 | 랜덤 포레스트의 MSE reduction은 $R^2$를 자연 분해하지 않으므로 정규화 필요 | Section 4.2.3, Table 1 각주 | p.312, Section 4.2.3 |
| 8 | RF-CI의 Figure 6에서의 비정상적 상관 의존성은 아직 설명되지 않는다 | Section 6.3 논의 | p.316, Section 6.3 |

---

## 2-1. 해결 문제, 제안 방법, 모델 구조, 성능, 한계

### 🔴 해결하고자 하는 문제

변수 간 상관관계가 있는 회귀 분석에서, 각 변수에게 중요도를 공정하게 배분하는 문제. 특히 랜덤 포레스트의 변수 중요도 메커니즘이 선형 회귀의 방법들과 어떻게 다른지 명확히 하지 못한 상태이다.

---

### 🔵 제안하는 방법 (수식 포함)

#### (1) 선형 회귀 모델

$$Y = \beta_0 + X_1\beta_1 + \cdots + X_p\beta_p + \varepsilon $$

주변 분산 모델:

$$\text{var}(Y) = \sum_{j=1}^{p} \beta_j^2 v_j + 2\sum_{j=1}^{p-1}\sum_{k=j+1}^{p} \beta_j \beta_k \sqrt{v_j v_k} \rho_{jk} + \sigma^2 $$

- $v_j$: $X_j$의 분산, $\rho_{jk}$: 변수 간 상관계수

#### (2) LMG (Lindeman, Merenda & Gold, 1980)

모든 $p!$ 개의 변수 순서에 대해 $X_k$를 추가할 때의 $R^2$ 증가분을 평균:

$$\text{LMG}(X_k) = \frac{1}{p!} \sum_{\text{orderings}} \Delta R^2(X_k | \text{predecessors})$$

#### (3) OOB-MSE (랜덤 포레스트 평가 지표)

$$\text{OOB-MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \bar{\hat{y}}_{i,\text{OOB}})^2 $$

$$\text{OOB-}R^2 = 1 - \frac{\text{OOB-MSE}}{\text{SST}}$$

#### (4) 퍼뮤테이션 기반 MSE 감소 (변수 중요도, Eq.3)

나무 $t$에서 $X_j$ 퍼뮤테이션 후 OOB-MSE:

$$\text{OOBMSE}_t(X_j \text{ permuted}) = \frac{1}{n_{\text{OOB},t}} \sum_{i \in \text{OOB}_t} (y_i - \hat{y}_{i,t}(X_j \text{ permuted}))^2 $$

변수 $X_j$의 중요도 = $\text{OOBMSE}_t(X_j \text{ permuted}) - \text{OOBMSE}_t$의 모든 나무 평균

---

### 🟢 모델 구조

| 구성 요소 | RF-CART | RF-CI |
|-----------|---------|-------|
| 기반 나무 | CART (최대 불순도 감소 기준) | Conditional Inference Tree (조건부 검정 기반) |
| 샘플링 | 복원 추출 (크기 $n$) | 비복원 추출 (크기 $0.632n$) |
| 기본 ntree | 500 | 500 |
| 기본 mtry | $\lfloor p/3 \rfloor$ | 명확한 기본값 없음 |
| 나무 크기 | 크고 미정제 (terminal node ≥ 5) | 작음 (min split=20, min node=7) |
| 변수 선택 편향 | 있음 (impurity 기반) | 없음 (p값 기반으로 분리) |

---

### 🟡 성능 향상

- RF-CART는 RF-CI보다 큰 나무를 사용하므로, 선형 모델 근사에 더 효율적 (p.313)
- 퍼뮤테이션 기반 MSE 감소는 impurity 기반 중요도의 편향 문제를 해소
- ntree를 2000으로 증가시킴으로써 변수 중요도 추정의 안정성 확보 (Table 1)
- mtry=p를 사용하는 RF-CI가 조건부 중요도에 더 가까워질 수 있음 (p.318)

---

### 🔴 한계

- 랜덤 포레스트는 항상 계단 함수를 적합하므로 선형 모델을 정확히 근사하지 못함 (p.313)
- RF-CI의 mtry와 상관구조 간 상호작용 메커니즘이 완전히 해명되지 않음 (Figure 6, p.316)
- 변수 중요도에 대한 이론적으로 합의된 정의(참조 표준)가 존재하지 않음 (p.317)
- $n \gg p$ 상황에 집중하며, $p \gg n$ 상황에서의 완전한 분석은 미래 과제로 남김
- 랜덤 포레스트의 변수 중요도는 $R^2$ 자연 분해가 아니므로 직접 비교 시 정규화 필요

---

## 3. 각 주장에 페이지/Figure/Table 번호 표시

| 주장 | 위치 |
|------|------|
| RF-CART ≈ LMG (mtry=1) | p.314, Figure 4 |
| RF-CI → PMVD (mtry 증가 시) | p.314–315, Figures 5, 6 |
| RF-CI의 mtry 민감성 > RF-CART | p.312, Table 1; p.315, Section 6.3 |
| 상관관계의 중요도 배분 영향 | p.313, Section 6.1; p.314, Figure 4 |
| 퍼뮤테이션 MSE 감소 공식 | p.311, Eq.(3) |
| Gini/impurity 편향 문제 | p.311, Section 4.2.3 |
| 설명적 vs. 예측적 변수 중요도 구분 | p.317, Section 7, Eqs.(4),(5) |
| MSE reduction ≠ $R^2$ 자연 분해 | p.312, Section 4.2.3 |
| Figure 6 비정상 패턴 미해명 | p.316, Section 6.3 |
| OOB-MSE 공식 | p.311, Section 4.2.2 |

---

## 4. 저자 직접 보고 결과 vs. 내 해석 분리

### 📋 연구 주제
**저자 직접 보고:** "This article compares the two approaches (linear model on the one hand and two versions of random forests on the other hand)" (p.308 Abstract)

**해석:** 이 논문은 변수 중요도라는 오래된 문제를 머신러닝과 통계학의 교차점에서 재조명하며, 두 패러다임의 방법론적 유사성과 차이를 실증적으로 분석한다는 점에서 방법론 비교 연구의 성격을 띤다.

---

### 📋 방법 (수식 포함)
**저자 직접 보고:** 퍼뮤테이션 기반 MSE 감소 공식 (Eq.3, p.311):

$$\text{OOBMSE}_t(X_j \text{ permuted}) = \frac{1}{n_{\text{OOB},t}} \sum_{\substack{i=1 \\ i \in \text{OOB}_t}}^{n} (y_i - \hat{y}_{i,t}(X_j \text{ permuted}))^2$$

**해석:** 이 방법은 Shapley value의 정신과 유사하게 변수를 "제거"했을 때의 예측 성능 저하를 측정하지만, 실제 변수 제거가 아닌 퍼뮤테이션을 사용하므로 변수 간 상관관계가 있을 때 해석이 복잡해진다. 특히 저자 스스로 "이 MSE 감소는 $X_j$를 사용 가능 여부에 따른 포레스트 MSE의 감소와 동일하지 않다"고 명시한다 (p.312).

---

### 📋 결과
**저자 직접 보고 (Table 1, p.312):**

| 변수 | PMVD | LMG | RF-CART(mtry=1) | RF-CI(mtry=1) |
|------|------|-----|-----------------|---------------|
| Agriculture | 21.3% | 22.0% | 26.1% | 20.7% |
| Examination | 1.0% | 25.6% | 22.9% | 28.8% |
| Education | 56.3% | 31.5% | 28.5% | 35.9% |
| Catholic | 18.2% | 18.3% | 17.6% | 12.9% |
| Infant.Mortality | 3.3% | 2.7% | 4.9% | 1.7% |

**저자 직접 보고:** "Average variable importance from the forests with mtry=1 is found to be quite similar to LMG." (p.314)

**해석:** Examination과 Education 사이의 높은 상관관계(0.79)가 방법 간 가장 큰 차이를 만들어낸다. LMG는 두 변수에게 모두 상당한 중요도를 부여하는 반면, PMVD는 Education에 56.3%를 집중시키며 Examination을 거의 무시(1.0%)한다. RF-CART와 RF-CI는 그 사이에 위치한다. 이는 각 방법이 채택하는 조건부-주변부 균형의 차이를 반영한다.

---

## 5. 통계적으로 취약한 부분과 비교 불가능한 수치

| 항목 | 취약점/비교 불가능 이유 |
|------|------------------------|
| ⚠️ Table 1의 모든 수치 | 단일 데이터셋(n=182)에 대한 결과로, 통계적 불확실성(신뢰구간) 미제공. 저자 스스로 "normalization to sum 100% is not recommended for data analysis purposes"라고 각주에 명시 (p.312) |
| ⚠️ 시뮬레이션 100회 평균 | 표준오차 미보고. "averages over 100 simulation runs are reasonably stable"이라고만 언급 (p.313) |
| ⚠️ RF-CART vs. RF-CI 비교 | 두 방법은 샘플링 방식, 나무 크기, 분할 기준이 모두 달라 단순 비교 불가. 저자도 "이 차이는 inconsequential"이라 하나 이는 무상관 연속 변수에 한정된 주장 (p.311) |
| ⚠️ n=100 시뮬레이션 | "not shown"으로 처리되어 재현 불가능 (p.315) |
| ⚠️ LMG/PMVD의 "참값" | 저자는 이를 $n \to \infty$ 극한으로 정의하나, 유한 표본에서의 추정 오차는 불명확 |
| ⚠️ OOB- $R^2$ vs. 선형 $R^2$ 비교 | 선형 모델 $R^2=61.3\%$와 OOB- $R^2$는 계산 기반이 달라 직접 비교 불가 |
| ⚠️ mtry 최적화 주장 | "mtry=1 or mtry=2"가 OOB-MSE를 최소화한다는 주장은 특정 시뮬레이션 설정에 한정되며 일반화 근거 부족 (p.315) |

---

## 6. 문서가 답하지 않는 질문

1. **Figure 6의 비정상 패턴**: 저자 스스로 " $\beta_7$에서 RF-CI의 강한 상관 의존성이 아직 설명되지 않는다"고 인정 (p.316)

2. **나무 크기 효과 vs. 분할 기준 효과 분리**: RF-CART와 RF-CI의 차이가 나무 크기 때문인지 p값 대 최대 불순도 감소 기준 차이 때문인지 분리되지 않음 (p.316)

3. **비선형 모델에서의 랜덤 포레스트 변수 중요도**: 논문은 $n \gg p$ 선형 시나리오에 집중하며, 비선형 또는 상호작용이 있는 경우는 완전히 다루지 않음

4. **$p \gg n$ 상황**: 언급하지만 체계적 분석 없음 (p.308)

5. **변수 중요도의 인과적 해석**: 인과 사슬 (4), (5)를 데이터만으로 구분하는 방법 (p.317)

6. **회귀 공간 형태의 영향**: 보충 자료로만 언급되며 본문에서 체계적으로 다루지 않음 (p.317, Section 6.4)

7. **변수 중요도의 표준 오차/신뢰구간**: 특히 랜덤 포레스트 중요도에 대한 분포적 특성 미제공

8. **mtry 최적 선택 기준**: 예측 정확도와 변수 중요도 해석 간의 트레이드오프에 대한 명확한 가이드라인 부재

---

## 7. 가장 중요한 그림 5개 해석

### Figure 1 (p.309) — 스위스 출산율 데이터 산점도 행렬

**해석:** 5개 예측변수와 반응변수(Fertility) 간의 이변량 관계를 선형 및 LOESS 곡선으로 시각화. Examination과 Education 간의 높은 상관관계($r=0.79$)가 명확히 보이며, Agriculture와 Catholic의 비선형 관계도 확인된다. 이는 선형 모델에서 Agriculture와 Catholic에 이차항을 포함시킨 근거이며, 변수 간 상관관계가 이후 모든 비교 분석의 핵심 동인임을 시각적으로 정당화한다.

---

### Figure 2 (p.310) — 개별 나무 vs. 포레스트용 나무 비교

**해석:** 좌측은 교차검증으로 정제된 단일 CART 나무(3개 분할, 4개 말단 노드), 우측은 정제 없이 완전히 성장한 포레스트용 나무(약 50~70개 말단 노드). 이 대조는 포레스트에서 나무 하나하나의 과적합은 허용하되, 평균화(bagging)를 통해 분산을 줄이는 랜덤 포레스트의 핵심 전략을 시각화한다. 나무 크기의 차이가 RF-CART와 RF-CI 간 중요도 차이의 주요 원인 중 하나임을 이해하는 데 핵심적이다.

---

### Figure 3 (p.313) — 선형 모델 vs. RF-CART 주효과 플롯

**해석:** 상단은 95% 신뢰대를 포함한 선형 모델의 주효과, 하단은 mtry=1(검정)과 mtry=2(회색)의 RF-CART 주효과. 두 방법 모두 Education의 강한 음의 효과와 Catholic의 U자형 관계를 포착한다. RF-CART는 계단 함수로 비선형 관계를 근사하며, mtry=1과 mtry=2 간 차이가 크지 않음이 확인된다(RF-CART의 mtry 안정성). 선형 모델이 Agriculture와 Catholic에 이차항을 명시적으로 모델링한 반면, 랜덤 포레스트는 이를 자동으로 학습한다는 점에서 두 접근법의 근본적 차이가 드러난다.

---

### Figure 4 (p.314) — mtry=1에서 LMG/PMVD와 RF-CART/RF-CI 비교

**해석:** 세 가지 계수 벡터($\boldsymbol{\beta}_1, \boldsymbol{\beta}_6, \boldsymbol{\beta}_7$)에 대해 상관 파라미터 $\rho \in [-0.9, 0.9]$를 변화시키며 정규화된 평균 변수 중요도를 비교. **핵심 발견:** RF-CART($\triangle$)는 LMG(회색 선)와 매우 유사하게 움직이며, RF-CI($\times$)는 LMG와 PMVD 사이에 위치. $\rho>0$에서 LMG는 "equalizing behavior"(강한 변수에서 약한 변수로 중요도 이동)를 보이며, RF-CART도 이를 따름. 이 그림이 논문의 핵심 실증 결과를 가장 명확하게 보여준다.

---

### Figure 5 (p.315) — $\boldsymbol{\beta}_1$에서 mtry=1~4 변화에 따른 RF-CART vs. RF-CI

**해석:** mtry를 1에서 4로 증가시킬 때 RF-CART($\triangle$)는 LMG 근처에서 안정적으로 유지되는 반면, RF-CI($\times$)는 mtry 증가에 따라 PMVD(검정 선) 쪽으로 점진적으로 이동. 특히 $X_4$(계수 0.3, 가장 약한 변수)의 중요도가 mtry 증가 시 RF-CI에서 급격히 감소한다. 이는 mtry가 클수록 약한 변수가 강한 변수와 경쟁에서 이길 확률이 낮아지기 때문으로 설명된다. **RF-CART의 mtry 안정성은 개별 나무의 큰 크기에서 기인**한다는 저자의 해석을 뒷받침한다.

---

## 8. 결론: 시사점, 후속 연구 계획, 추가 방향

### 8-1. 저자가 제시한 시사점과 후속 연구

**시사점 (p.316–318):**
- RF-CART(작은 mtry)는 LMG와 유사한 균형적(equalizing) 중요도를 제공하며, 설명적 목적의 변수 선택에 적합
- RF-CI(큰 mtry)는 PMVD와 유사하게 조건부 중요도에 가까우며, 예측 목적의 변수 선택에 적합
- 변수 중요도 해석 시 **목적(설명 vs. 예측)을 명확히** 해야 한다
- Strobl et al.(2008)의 조건부 퍼뮤테이션 접근이 조건부 중요도에 근접하나, mtry < p인 한 주변적(marginal) 요소가 잔존한다

**저자 제시 후속 연구 (p.316, Section 6.4):**
1. Figure 6의 RF-CI 비정상 패턴 메커니즘 규명
2. 나무 크기 효과와 분할 기준(불순도 vs. p값) 효과 분리 연구
3. 회귀 공간의 형태(shape of regressor space)가 변수 중요도에 미치는 영향
4. mtry 선택이 변수 중요도에 미치는 영향의 체계적 이해

---

### 모델의 일반화 성능 향상 가능성

**저자 직접 언급:**
- Segal, Barbour & Grant(2004)의 제안 인용: 최소 노드 크기를 늘리면 RF-CART의 예측 정확도가 향상될 수 있음 (p.311)
- mtry=p가 예측 성능에 유리하다는 Genuer et al.(2008)의 결과 인용 (p.315)
- RF-CART가 선형 모델보다 파시모니하지 않으나, 큰 표본과 많은 나무에서 어떤 평활 기대 모델도 근사한다는 Ishwaran(2007)의 이론 결과 (p.313)

**내 해석 — 일반화 성능 관련:**
- 이 논문은 변수 중요도 해석에 초점을 두며, 일반화 성능(out-of-sample prediction)을 직접 최적화하는 연구가 아님
- 그러나 mtry 선택이 변수 중요도와 예측 성능 양쪽에 영향을 주므로, **두 목적을 동시에 최적화하는 mtry 선택 기준**이 필요하다 — 이 논문에서는 미해결
- RF-CI의 작은 나무(small tree)는 변수 중요도의 mtry 민감성을 높이는 동시에, 선형 모델 근사에서의 효율성도 낮춘다. 이는 **해석 가능성(interpretability)과 예측 성능 간의 트레이드오프**를 내포한다
- $n=100$ 소표본에서 RF-CI의 PMVD 근사가 크게 저하되는 점(p.315)은 소표본에서 랜덤 포레스트의 일반화 성능 불안정성을 시사한다

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의:** 아래는 2020년 이후 공개된 관련 연구들에 대한 분석입니다. 제가 직접 해당 논문들의 원문을 확인한 것이 아니므로, 구체적인 수치나 세부 결과보다는 연구 흐름과 방향을 중심으로 기술합니다. 확실하지 않은 세부 내용은 명시합니다.

#### 관련 연구 흐름

**① SAGE (Shapley Additive Global Importances) — Covert, Lundberg & Lee (2020)**
- *"Understanding Global Feature Importance with Information Theoretic Shapley Values"* (2020, NeurIPS)
- Grömping(2009)이 LMG가 Shapley value임을 지적한 것과 연결하여, 신경망 등 비선형 모델에서도 Shapley 기반 글로벌 변수 중요도를 계산하는 방법 제안
- **연결점:** Grömping(2009)의 LMG가 선형 모델에서 Shapley value와 동치임을 보였는데, SAGE는 이를 일반적 모델로 확장

**② SHAP (SHapley Additive exPlanations) — Lundberg & Lee (2017), 이후 tree SHAP (2020)**
- *"From local explanations to global understanding with explainable AI for trees"* (Lundberg et al., 2020, Nature Machine Intelligence)
- 트리 모델에서 Shapley 기반 로컬/글로벌 설명을 효율적으로 계산
- **Grömping(2009)과의 관계:** LMG와 유사한 Shapley 기반 접근을 트리 앙상블에 적용하되, 계산 복잡도를 다항 시간으로 감소시킴. Grömping의 논문은 LMG가 본질적으로 Shapley value임을 연결 짓는 다리 역할을 함

**③ 조건부 변수 중요도의 발전**
- Strobl et al.(2008)의 조건부 퍼뮤테이션 기반 중요도에서 출발한 후속 연구들이 2020년 이후에도 지속
- *Conditional Permutation Importance* 관련 연구들: 상관 변수 처리에서의 편향 감소 문제
- **Grömping(2009)의 기여:** 조건부 vs. 주변부 중요도의 개념적 구분을 명확히 하여, 이후 연구의 이론적 토대 제공

**④ 변수 중요도의 신뢰 구간 및 검정**
- 2020년 이후 변수 중요도의 통계적 불확실성을 정량화하는 연구 증가
- **Grömping(2009)의 한계 반영:** 논문이 신뢰구간을 제공하지 못한 점이 후속 연구의 동기가 됨

---

#### Grömping(2009)이 앞으로의 연구에 미치는 영향

| 영향 영역 | 내용 |
|-----------|------|
| **XAI(설명 가능한 AI)** | LMG = Shapley value 연결은 SHAP 등 XAI 방법의 통계적 정당화에 기여 |
| **변수 중요도 이론** | 설명적 vs. 예측적 중요도의 개념 구분은 현재 XAI 논의에서도 핵심 프레임 |
| **랜덤 포레스트 이해** | mtry의 중요도 배분 영향 규명은 AutoML에서의 하이퍼파라미터 튜닝 연구의 토대 |
| **방법 비교 프레임워크** | 선형 모델을 기준점(benchmark)으로 비선형 방법을 비교하는 방법론 정립 |

---

#### 앞으로 연구 시 고려할 점

1. **Shapley 기반 통합 프레임워크 활용:** LMG(= Shapley value)와 SHAP의 연결을 통해 선형/비선형 모델에서 일관된 변수 중요도 해석 가능성 탐색. 단, 계산 비용과 상관관계 처리 방식의 차이 주의

2. **조건부 vs. 주변부 중요도의 목적 명확화:** Grömping(2009)이 제시한 인과 사슬 프레임($X_2 \to X_1 \to Y$ vs. $X_2 \leftarrow X_1 \to Y$)은 인과 추론 기반 변수 중요도 연구와 연결될 수 있음

3. **mtry 적응형 선택:** 변수 중요도 목적에 따라 mtry를 다르게 설정하는 전략 탐색 (예: 설명 목적→작은 mtry, 예측 목적→큰 mtry)

4. **소표본($n=100$) 성능:** 랜덤 포레스트 변수 중요도의 소표본 불안정성에 대한 체계적 연구 필요. 이 논문이 지적했으나 충분히 분석하지 않은 부분

5. **다중 상관 구조의 영향:** AR(1) 구조($\rho^{|j-k|}$) 이외의 복잡한 상관 구조(클러스터형, 계층형)에서의 방법 비교

6. **비선형/상호작용 모델:** 논문은 선형 시뮬레이션에 집중하므로, 실제 비선형 모델에서 방법 간 비교는 별도 연구 필요

---

## 📚 참고 자료

**논문 원문:**
- Grömping, U. (2009). "Variable Importance Assessment in Regression: Linear Regression versus Random Forest." *The American Statistician*, 63(4), 308–319. DOI: 10.1198/tast.2009.08199

**논문 내 인용 주요 참고문헌:**
- Breiman, L. (2001). "Random Forests." *Machine Learning*, 45, 5–32.
- Lindeman, R.H., Merenda, P.F., & Gold, R.Z. (1980). *Introduction to Bivariate and Multivariate Analysis*. Scott, Foresman.
- Strobl, C., et al. (2007). "Bias in Random Forest Variable Importance Measures." *BMC Bioinformatics*, 8, 25.
- Strobl, C., et al. (2008). "Conditional Variable Importance for Random Forests." *BMC Bioinformatics*, 9, 307.
- Feldman, B. (2005). "Relative Importance and Value." Unpublished manuscript.
- Hothorn, T., Hornik, K., & Zeileis, A. (2006b). "Unbiased Recursive Partitioning." *JCGS*, 15, 651–674.
- Ishwaran, H. (2007). "Variable Importance in Binary Regression Trees and Forests." *EJS*, 1, 519–537.

**2020년 이후 관련 연구 (방향 참고용, 원문 직접 확인 권장):**
- Lundberg, S.M., et al. (2020). "From local explanations to global understanding with explainable AI for trees." *Nature Machine Intelligence*, 2, 56–67.
- Covert, I., Lundberg, S., & Lee, S.-I. (2020). "Understanding Global Feature Importance with Information Theoretic Shapley Values." *NeurIPS 2020*.
