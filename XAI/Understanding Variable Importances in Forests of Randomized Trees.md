# Understanding Variable Importances in Forests of Randomized Trees

---

## Executive Summary (10문장 이내)

본 논문은 Random Forest 등 트리 기반 앙상블에서 널리 쓰이는 Mean Decrease Impurity(MDI) 변수 중요도가 실무에서 광범위하게 사용됨에도 이론적 근거가 부족하다는 문제의식에서 출발한다(p.1). 저자들은 totally randomized trees(완전 무작위 트리)와 무한 표본·무한 앙상블 크기라는 점근 조건 하에서 MDI를 Shannon 상호정보량으로 정확히 표현하는 정리(Theorem 1)를 유도한다(p.3, Eq.3). 이를 통해 전체 정보량 $I(X_1,...,X_p;Y)$가 (i) 변수별, (ii) 상호작용 차수별, (iii) 상호작용 조합별로 3단계 분해된다는 것을 증명한다(p.3-4, Theorem 2, Eq.4). 또한 MDI 중요도가 0이 되는 것은 해당 변수가 Y와 무관(irrelevant)한 경우와 필요충분조건이며(Theorem 3), 무관한 변수를 추가/제거해도 관련 변수의 중요도가 불변함을 증명한다(Theorem 5, p.4). 반면 K>1인 Random Forest/Extra-Trees와 같은 비완전 무작위 트리에서는 masking effect로 인해 이러한 바람직한 성질이 깨진다는 것을 이론적·실험적으로 보인다(p.5-6, 7-segment 예제). 7-segment 디스플레이 숫자 인식 문제를 통해 이론값과 시뮬레이션 결과가 일치함을 검증한다(Table 2, Figure 2). 본 연구는 MDI에 대한 최초의 엄밀한 이론적 정당화를 제공하지만, 이진 분할(binary split)·연속형 변수·유한 표본 상황에 대한 확장은 향후 과제로 남겨둔다(p.8, Conclusion).

---

## 1-1. 연구의 목적과 필요성

**목적**: Random Forest 계열 알고리즘에서 파생되는 MDI(Mean Decrease Impurity) 변수 중요도를 점근적(asymptotic) 조건 하에서 수학적으로 특성화(characterize)하는 것.

**필요성**: 
- Random Forest(Breiman, 2001)와 Extra-Trees(Geurts et al., 2006)는 과학 분야 전반에서 변수 중요도 측정 도구로 널리 쓰이지만, 그 작동 원리에 대한 이론적 연구는 Ishwaran(2007)의 MDA(Mean Decrease Accuracy) 연구가 유일했다(p.1).
- Strobl et al.(2007, 2008)은 MDI가 범주 수가 많은 변수에 편향된다는 것을 실험적으로만 보였을 뿐, MDI 자체의 수학적 정체성은 규명되지 않았다(p.3).
- 저자들은 이 이론적 공백(gap)을 메우고자 함(p.1, "we aim at filling this gap").

---

## 2. 핵심 주장과 근거 표

| # | 핵심 주장 | 근거(정리/식) | 페이지 |
|---|---|---|---|
| 1 | MDI는 조건부 상호정보량의 가중합으로 정확히 표현됨 | Theorem 1, Eq.(3) | p.3 |
| 2 | 모든 변수의 MDI 합은 전체 상호정보량과 같음 | Theorem 2, Eq.(4) | p.3 |
| 3 | MDI=0 ⟺ 변수가 무관(irrelevant) | Theorem 3 | p.4 |
| 4 | 무관 변수 추가/제거는 관련 변수의 MDI에 영향 없음 | Lemma 4, Theorem 5, Eq.(5) | p.4 |
| 5 | 위 정리들은 Gini, 분산 등 다른 불순도 척도로 일반화 가능 | Appendix I 언급 | p.5 |
| 6 | 깊이 제한(pruning)된 트리는 Random Subspace 방법과 동일한 중요도 산출 | Prop.6, Prop.7 | p.5 |
| 7 | K>1(RF, Extra-Trees)에서는 masking effect로 위 성질들이 깨짐 | 예제(X1,X2), 7-segment 실험 | p.5-6, Table 2, Fig.2 |

---

## 2-1. 상세 설명

### 해결하고자 하는 문제
- MDI 중요도가 "왜", "어떤 조건에서" 타당한 변수 중요도 척도인지에 대한 수학적 정당화 부재.

### 제안하는 방법 (수식)

**MDI 정의** (Breiman 원래 정의, p.2):
$$Imp(X_m) = \frac{1}{N_T}\sum_{T}\sum_{t\in T: v(s_t)=X_m} p(t)\Delta i(s_t,t) \quad (2)$$

**핵심 정리 (Theorem 1)** — totally randomized trees, 무한 표본 조건:
$$Imp(X_m) = \sum_{k=0}^{p-1} \frac{1}{C_p^k}\frac{1}{p-k}\sum_{B\in\mathcal{P}_k(V^{-m})} I(X_m;Y|B) \quad (3)$$

여기서 $\mathcal{P}_k(V^{-m})$는 $X_m$을 제외한 나머지 변수 집합에서 크기 $k$인 부분집합들의 모임.

**총 정보량 보존 (Theorem 2)**:
$$\sum_{m=1}^{p} Imp(X_m) = I(X_1,\ldots,X_p;Y) \quad (4)$$

**Relevant/Irrelevant 정의** (Kohavi & John, 1997 기반, p.4):
- Relevant: $\exists B\subseteq V, I(X_m;Y|B)>0$
- Irrelevant: $\forall B\subseteq V, I(X_i;Y|B)=0$

**상한 (bound)**:
$$\frac{1}{C_p^k}\frac{1}{p-k}\sum_{B\in\mathcal{P}_k(V^{-m})} H(Y) = \frac{1}{p}H(Y)$$

### 모델 구조
- Totally randomized tree: 각 노드에서 아직 사용되지 않은 변수 중 하나를 균등 확률로 무작위 선택, 해당 변수의 모든 값에 대해 분기(비이진), 모든 p개 변수가 사용될 때까지 완전 성장.
- 비교 대상: K개 변수를 무작위로 뽑은 후 $\Delta i(t)$ 최대화하는 변수 선택 (K=1: totally randomized, K=p: 결정론적 단일 트리, RF/Extra-Trees는 중간 K).

### 성능 향상
- 본 논문은 "예측 성능" 논문이 아니라 "이론적 특성화" 논문이므로 정확도(accuracy) 향상을 다루지 않음. 대신 "바람직한 통계적 성질(desirable properties)"의 확보를 성과로 제시(p.4).

### 한계 (저자 명시, p.8 Conclusion)
1. 이진 분할(binary split) 미반영 — 실제 RF는 이진 분할만 사용, 변수가 한 가지(branch)에서 여러 번 등장 가능하여 카디널리티 의존성 발생 가능.
2. 연속형 변수로의 확장 미비 — 논문은 categorical variable 가정(p.3).
3. 점근적(asymptotic) 결과만 제공 — 유한 표본에서의 분포적 특성은 미해결.
4. K>1(실제 RF, Extra-Trees)에 대해서는 완전한 이론적 특성화가 아닌 정성적 논의(masking effect)에 그침(p.5-6).

---

## 3. 주장별 페이지/Figure/Table 표시

| 주장 | 위치 |
|---|---|
| MDI의 3단계 분해 | Theorem 1 (p.3), Eq.3 |
| 정보량 합 보존 | Theorem 2 (p.3), Eq.4 |
| Irrelevant ⟺ Imp=0 | Theorem 3 (p.4) |
| Irrelevant 변수 무관성 | Lemma 4, Theorem 5 (p.4), Eq.5 |
| Pruning/Random Subspace 동치성 | Prop 6,7 (p.5), Eq.6 |
| Masking effect 정성적 설명 | p.5-6, 수식 예제 (X1, X2) |
| 7-segment 실험 검증 | Table 1, 2 (p.7), Figure 1, 2 (p.7-8) |

---

## 4. 저자 보고 결과 vs. 해석 분리

### 저자가 직접 보고한 결과
- Table 2: K=1일 때 이론값(Eq.3)과 시뮬레이션(10,000 트리) 결과가 거의 일치(예: X5 이론값 0.656, K=1 시뮬 0.658).
- 모든 K에서 $\sum Imp(X_m) = 3.321 = \log_2(10) = H(Y)$ 로 일정함(Theorem 2 확인).
- K 증가 시 X2, X5의 중요도는 증가(0.581→0.799, 0.656→0.835)하고 X1, X3, X4, X6는 감소하는 경향 관찰(Table 2).
- Figure 2: K=1에서는 모든 조건부 상호정보항이 고르게 반영되나, K=7에서는 k=0에서 X2, X5만 양수이고 나머지는 0으로 마스킹됨.

### 필자(AI)의 해석
- 이 결과는 "MDI가 트리 구조의 탐욕적(greedy) 선택 방식에 의해 왜곡될 수 있다"는 것을 보여주는 사례 연구로, 일반적인 고차원 데이터에서도 유사한 마스킹이 발생할 가능성을 시사한다고 해석할 수 있음(단, 논문이 이를 일반화하여 증명하지는 않음).
- K=p(완전 결정론적 트리)의 극단적 사례이므로, 실제 RF에서 사용하는 낮은 K(예: $\sqrt{p}$)에서는 masking effect가 논문의 예시보다는 완화될 가능성이 있으나, 이는 논문에서 정량적으로 검증되지 않았음(추정).

---

## 5. 통계적으로 취약한 부분 및 비교 불가능한 수치

- **Table 2의 시뮬레이션은 단일 인공 데이터셋(7-segment, p=7, 매우 작은 차원)** 에 대한 것으로, 고차원 실제 데이터에 대한 일반화 가능성은 검증되지 않음.
- **10,000 트리 앙상블**이라는 유한 앙상블 크기에서 발생하는 오차(Table 2의 Eqn.3과 K=1 값의 미세한 차이, 예: X1 0.412 vs 0.414)는 무한 앙상블 가정과의 괴리를 나타내나, 오차의 통계적 유의성(신뢰구간, 표준오차 등)은 보고되지 않음.
- 논문 전체에 **가설 검정, p-value, 신뢰구간 등 통계적 검증 절차가 전혀 없음** — 순수 이론적 유도와 단일 예제 검증에 의존.
- Random Forests/Extra-Trees(K>1)에 대한 결과는 **정성적 논의와 하나의 counter-example**에 그치며, 다양한 K, 다양한 데이터 구조에 대한 체계적 실험은 부재.

---

## 6. 문서가 답하지 않는 질문

1. 이진 분할(binary split) 기반의 실제 CART/RF 알고리즘에서 MDI의 정확한 수식적 형태는 무엇인가?
2. 연속형 변수에 대해서도 동일한 3단계 분해가 성립하는가?
3. 유한 표본 크기에서 MDI의 분산(variance)이나 분포는 어떻게 되는가? (신뢰구간 추정 가능한가?)
4. Bootstrap(bagging)이 도입되었을 때(즉 실제 RF처럼) 정리들이 어떻게 변형되는가? (본 논문은 bagging 없이 순수 randomization만 고려)
5. K(각 노드에서 고려하는 변수 후보 수)와 masking effect 정도 사이의 정량적 관계식은?
6. 변수 간 상관관계(correlation)가 존재할 때 MDI의 편향 정도를 수식으로 표현할 수 있는가?
7. MDA(permutation importance)와 MDI 간의 이론적 관계는 무엇인가? (본 논문은 MDI만 다룸)

---

## 7. 가장 중요한 그림/표 5개 해석

### ① Table 2 (p.7)
K=1(totally randomized)일 때 이론식(Eq.3)과 시뮬레이션 결과가 사실상 일치함을 보여 Theorem 1을 실증적으로 검증. K가 증가할수록(2→7) X2, X5의 중요도가 과대평가되고 X1,X3,X4,X6은 과소평가되는 masking effect를 수치로 제시.

### ② Figure 2 (p.8)
좌측(K=1) 히트맵은 모든 상호작용 차수(k=0~6)에서 고르게 정보가 반영됨을 보여줌. 우측(K=7)에서는 k=0에서 X2, X5만 밝은 색(높은 값)을 보이고 나머지는 거의 0(masking)이며, k≥4에서는 모든 값이 0에 가까워 해당 조건부 조합이 실제로 발생하지 않았음(트리가 그 깊이에 도달하기 전에 순수해짐)을 시각적으로 증명.

### ③ Theorem 1의 수식(Eq.3) 구조 자체
그림은 아니지만 본 논문의 핵심 "결과물"로서, MDI를 상호정보량의 조합론적 가중합으로 명시적으로 표현하여 이후 모든 정리(2,3,5)의 기반이 됨.

### ④ Figure 1 / Table 1 (7-segment 디스플레이, p.7)
실험에 사용된 인공 데이터의 구조를 정의. 각 숫자(0~9)가 7개 이진 변수의 특정 조합으로 표현되는 구조로, 이론값을 직접 계산 가능하게 하는 "정답이 있는" 검증용 데이터셋 역할.

### ⑤ p.6의 counter-example 수식 (X1, X2 예제)
$$Imp(X_1)=\frac{1}{2}I(X_1;Y)+\frac{\epsilon}{2}, \quad Imp(X_2)=\frac{1}{2}I(X_2;Y)$$
vs K=2일 때
$$Imp_{K=2}(X_1)=I(X_1;Y), \quad Imp_{K=2}(X_2)=0$$
이는 masking effect를 가장 간결하게 보여주는 사례로, "X2가 실제로는 유용한 정보를 담고 있음에도 K=2에서는 완전히 무시된다"는 것을 수식적으로 증명하여 이후 실험(Table 2, Figure 2)의 이론적 근거가 됨.

---

## 8. 결론: 시사점 및 후속 연구

### 저자가 제시한 시사점 (p.8)
- Totally randomized trees의 MDI는 "무관한 변수는 정확히 0의 중요도를 가지며, 관련 변수의 중요도는 무관 변수의 존재와 무관하다"는 바람직한 성질을 만족하는 유일하게 엄밀히 증명된 척도.
- 반면 실제 RF/Extra-Trees(K>1)는 masking effect로 인해 이 성질이 보장되지 않음 — 즉, "성능(예측 정확도)"과 "해석가능성(변수 중요도의 순수성)" 사이에 트레이드오프가 존재함을 시사.

### 저자가 제시한 후속 연구 계획 (p.8)
1. 이진 분할(binary split) 기반 실제 알고리즘으로의 확장.
2. 연속형 변수에 대한 프레임워크 확장.
3. 유한 표본 조건에서 MDI의 분포(distribution) 특성화.

### 추가 후속 연구 방향 (AI 제안)
- Shapley value 기반 변수 중요도(SHAP, Lundberg & Lee 2017)와의 이론적 연결 규명.
- Bootstrap sampling(bagging)이 포함된 실제 Random Forest에 대한 정리 확장.
- Masking effect의 정량적 상한(bound)을 K와 상관계수의 함수로 유도.

---

## 8-1. 모델의 일반화 성능 향상 가능성

본 논문은 **예측 성능(generalization) 자체를 다루지 않는다** — MDI는 정확도가 아닌 "해석"을 위한 도구이며, 논문은 이 해석 도구의 신뢰성(theoretical soundness)을 다룬다. 다만 간접적 시사점은 다음과 같다:

- Theorem 5(무관 변수 추가/제거 무관성)는 **변수 선택(feature selection)** 을 통한 일반화 성능 개선의 이론적 근거를 제공한다. 만약 MDI로 무관 변수를 정확히 식별할 수 있다면(단, totally randomized tree라는 이상적 조건 하에서만 보장됨), 이를 제거하여 차원을 축소하고 과적합을 줄여 일반화 성능을 높일 수 있다.
- 그러나 실제 RF(K>1)에서는 masking effect로 인해 "중요도 0"이 "진짜 무관함"을 의미하지 않을 수 있으므로(p.6), MDI 기반 변수 선택이 실제로는 관련 변수를 잘못 제거하여 오히려 일반화 성능을 저해할 위험이 있다는 것이 논문의 암묵적 경고다.
- 논문 자체는 이 부분에 대한 실험적 검증(예측 정확도 비교)을 제공하지 않으므로, 이는 어디까지나 이론적 추론이며 실증적 근거는 부재하다.

---

## 8-2. 2020년 이후 관련 최신 연구 비교 분석

*(주의: 아래 내용은 본 논문 PDF에 포함되어 있지 않으며, 일반적으로 알려진 연구 동향에 기반한 설명입니다. 정확한 인용정보나 세부수치는 확인이 어려워 제한적으로만 기술합니다.)*

- **SHAP (Lundberg & Lee, 2017) 및 후속 연구**: 게임이론 기반 Shapley value를 트리 앙상블에 특화시킨 TreeSHAP(Lundberg et al., 2020, Nature Machine Intelligence)은 본 논문과 마찬가지로 "일관성(consistency)" 개념을 공리적으로 요구하며, 본 논문의 Theorem 3(irrelevant ⟺ importance 0)과 유사한 공리를 만족시키고자 함. 다만 TreeSHAP은 게임이론적 접근이라는 점에서 본 논문의 정보이론적(mutual information) 접근과 이론적 뿌리가 다름.
- **Debiased/Conditional variable importance 연구**: Strobl et al.(2008)의 conditional importance 개념을 발전시킨 연구들이 계속되고 있으며, 상관 변수 상황에서의 편향 문제를 다루는 것은 본 논문이 제기한 한계(이진분할, 상관관계)와 직접 연결됨.
- **MDI+ (Agarwal et al., 2023, 유사 연구 계열)**과 같이 MDI의 편향(특히 과적합으로 인한 편향)을 표본분할(sample-splitting)로 교정하려는 시도들이 이 논문이 제기한 "유한 표본에서의 특성화" 문제의식을 계승하고 있다고 볼 수 있음.

**본 논문이 후속 연구에 미친 영향**: 본 논문(Louppe et al., 2013)은 MDI에 대한 최초의 엄밀한 이론적 정당화로서, scikit-learn 등 실무 라이브러리에서 사용되는 feature_importances_의 이론적 근거로 자주 인용되며, 이후 MDI 편향 교정 연구(sample-splitting 기반 방법 등)의 출발점 역할을 함.

**향후 연구 시 고려할 점**:
1. 본 논문의 결과는 categorical, 완전 무작위 트리, 무한 표본/앙상블이라는 강한 가정 하에 성립하므로, 실제 RF 적용 시 결과를 그대로 신뢰하기보다 masking effect 가능성을 항상 염두에 두어야 함.
2. 상관관계가 높은 변수들이 존재하는 실제 데이터(유전체 데이터, 금융 데이터 등)에서는 본 논문의 이상적 조건이 크게 깨질 수 있으므로, MDA나 SHAP 등 대안적 척도와 병행 사용이 권장됨.

---

**참고자료**: 
- Louppe, G., Wehenkel, L., Sutera, A., & Geurts, P. (2013). *Understanding variable importances in forests of randomized trees*. NIPS 2013 (제공된 PDF 문서, 전체 페이지 1-9 참조).
