# Accurate Intelligible Models with Pairwise Interactions

> **참고 자료**: Lou, Y., Caruana, R., Gehrke, J., & Hooker, G. (2013). Accurate intelligible models with pairwise interactions. *Proceedings of the 19th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (KDD '13), pp. 623–631. ACM.

---

## 1. Executive Summary (10문장 이내)

1. 본 논문은 해석 가능성과 예측 정확도를 동시에 달성하는 모델 클래스 **GA²M(Generalized Additive Models plus Interactions)**을 제안한다.
2. 기존 GAM은 단변량 항만 포함하여 해석은 용이하지만, 랜덤 포레스트 등 전체 복잡도 모델 대비 정확도가 현저히 낮다는 한계가 있다.
3. GA²M은 표준 GAM에 선택된 쌍별(pairwise) 상호작용 항을 추가하여, 1차원 및 2차원 성분만으로 구성되어 시각화(히트맵)를 통한 해석이 가능하다.
4. 쌍별 상호작용의 수가 $O(n^2)$으로 매우 많기 때문에, 저자들은 모든 후보 쌍을 효율적으로 순위화하는 **FAST(Fast Interaction Detection)** 알고리즘을 개발하였다.
5. FAST는 누적 히스토그램 기반 룩업 테이블을 활용하여 쌍당 $O(b^2 + N)$ 복잡도로 상호작용 강도를 측정한다.
6. 10개의 실제 데이터셋 실험에서 GA²M은 랜덤 포레스트와 거의 동등하거나 일부 데이터셋에서 더 우수한 성능을 달성하였다.
7. FAST는 ANOVA, Grove 등 기존 방법 대비 3~4 오더 수준으로 빠르면서 유사한 정밀도를 보인다.
8. GA²M은 특성 간 상관관계로 인한 허위 상호작용(spurious pairs)을 그래디언트 부스팅 과정에서 자동으로 감쇄시키는 능력을 갖는다.
9. MSLR10k 데이터셋의 사례 연구를 통해 GA²M이 전문가가 해석 가능한 1차원 형상함수 및 쌍별 상호작용 히트맵을 제공함을 실증하였다.
10. 본 논문은 많은 현실 문제에서 GA²M이 **"해석 가능하면서 정확한"** 모델을 동시에 제공할 수 있음을 제안한다.

### 1-1. 연구의 목적과 필요성

**목적**: 고성능 블랙박스 모델(랜덤 포레스트, SVM, 신경망)과 해석 가능한 GAM 사이의 정확도 격차를 최소화하면서, 사용자가 이해할 수 있는 모델을 구축하는 것이다.

**필요성**:

| 문제 | 세부 내용 |
|------|-----------|
| 해석 가능성의 중요성 | 의료, 금융, 법률 등 고위험 도메인에서 모델 결정 근거 파악이 필수적 [논문 p.1] |
| 기존 GAM의 정확도 한계 | 단변량 항만 모델링하여 특성 간 상호작용을 무시함으로써 전체 복잡도 모델 대비 큰 성능 차이 존재 [p.1] |
| 기존 상호작용 탐지의 비효율 | Grove 방법은 정확하나 계산 비용이 매우 크며(약 1주일), ANOVA·PDF는 저밀도 영역에서 허위 탐지 문제 발생 [p.2] |
| 확장성 문제 | $n$개 특성 시 $O(n^2)$ 쌍 존재 → 대규모 데이터에 실용적인 방법 필요 [p.1~2] |

---

## 2. 핵심 주장과 근거 표

| # | 핵심 주장 | 근거 | 위치 |
|---|----------|------|------|
| 1 | GA²M은 해석 가능성을 유지하면서 GAM보다 훨씬 높은 정확도를 달성 | 2차원까지의 성분은 히트맵으로 시각화 가능; 10개 데이터셋 실험 결과 | p.1, Table 2, 3 |
| 2 | GA²M은 다수의 데이터셋에서 랜덤 포레스트와 동등하거나 더 우수한 성능 | 회귀 평균 정규화 점수 0.84 vs RF 0.83; 분류 0.81 vs RF 0.79 | Table 2, 3 |
| 3 | FAST는 정확한 상호작용 순위화를 매우 효율적으로 수행 | 합성 함수에서 상위 10쌍 정확 탐지; Grove 대비 3~4 오더 빠름 | Figure 5, p.7 |
| 4 | FAST의 최적 빈 수는 $b=8$ 근방이며 넓은 범위에서 안정적 | 평균 정밀도의 빈 수별 민감도 분석 | Figure 4 |
| 5 | 허위 쌍은 그래디언트 부스팅에 의해 자동 감쇄 | 상관 특성 실험에서 $(x_2, x_6)$ 가중치가 빠르게 0에 수렴 | Figure 7 |
| 6 | FAST의 쌍당 계산 복잡도는 $O(b^2 + N)$ | 동적 프로그래밍 기반 룩업 테이블 구성 이론 분석 | Section 4.1.4 |

### 2-1. 해결 문제, 제안 방법, 모델 구조, 성능, 한계

#### ① 해결하고자 하는 문제

- **정확도-해석가능성 트레이드오프**: 표준 GAM은 해석은 되지만 특성 간 상호작용 부재로 정확도 손실
- **상호작용 탐지의 계산 비용**: 기존 Grove는 정확하나 대규모 데이터에 적용 불가

#### ② 제안하는 방법

**GA²M 모델 (Eq. 2, p.1)**:

$$g(E[y]) = \sum_{i} f_i(x_i) + \sum_{i < j} f_{ij}(x_i, x_j)$$

**목적 함수 (Eq. 3, p.2)**:

$$\min_{F \in \mathcal{H}} E[L(y, F(\boldsymbol{x}))]$$

여기서 $\mathcal{H} = \sum_{u \in \mathcal{U}} \mathcal{H}_u$, $\mathcal{U} = \mathcal{U}^1 \cup \mathcal{U}^2$

**FAST의 RSS 계산 (Eq. 8–9, p.4)**:

$$RSS = \sum_{k=1}^{N}(y_k - T_{ij}(\boldsymbol{x}_k))^2 = \left(\sum_{k=1}^{N} y_k^2 - 2\sum_r T_{ij}.r \cdot L^t.r + \sum_r (T_{ij}.r)^2 L^w.r \right)$$

실제 구현 시 상대 순위만 필요하므로:

$$\text{Score}_{ij} = \sum_r (T_{ij}.r)^2 L^w.r - 2\sum_r T_{ij}.r \cdot L^t.r$$

**Friedman-Popescu의 기존 H-통계량 (Eq. 4, p.2)**:

$$H^2_{ij} = \frac{\sum_{k=1}^{N}[\hat{F}_{ij}(x_{ki}, x_{kj}) - \hat{F}_i(x_{ki}) - \hat{F}_j(x_{kj})]^2}{\sum_{k=1}^{N} \hat{F}^2_{ij}(x_{ki}, x_{kj})}$$

**Grove 방법의 상호작용 강도 (Eq. 6–7, p.3)**:

$$stRMSE(F(\boldsymbol{x})) = \frac{RMSE(F(\boldsymbol{x}))}{StD(F^*(\boldsymbol{x}))}$$

$$I_{ij} = stRMSE(R_{ij}(\boldsymbol{x})) - stRMSE(F(\boldsymbol{x}))$$

**상호작용 가중치**:

$$w_u = \sqrt{E[f_u^2]}$$

#### ③ 모델 구조

```
[Stage 1] 그래디언트 부스팅으로 GAM (1차원 형상함수 fi) 학습
         ↓
[FAST] 잔차 R = y - F(x)에 대해 모든 쌍 (xi, xj)의 상호작용 강도 순위화
         ↓ (상위 K쌍 선택)
[Stage 2] 선택된 쌍에 대해 잔차 위에서 2차원 형상함수 fij 학습
         ↓
[최종 모델] GA²M: fi들 + fij들의 합, 각 항을 히트맵/곡선으로 시각화
```

- 특성 이산화: 연속형 특성 → 256개 등빈도 빈
- FAST용: 8개 빈 사용
- 쌍 선택 상한: 최대 1,000쌍

#### ④ 성능 향상

| 비교 | 회귀 (평균 정규화 RMSE) | 분류 (평균 정규화 오류율) |
|------|------------------------|-------------------------|
| Linear/Logistic Regression | 1.52 / 1.79 | — |
| GAM (기준) | 1.00 | 1.00 |
| **GA²M FAST** | **0.84** | **0.81** |
| Random Forests | 0.83 | 0.79 |

*(Table 2, 3; 낮을수록 우수)*

#### ⑤ 한계

| 한계 | 설명 |
|------|------|
| 3차 이상 상호작용 불가 | 모델 구조상 1차원 + 2차원 항만 포함 |
| 수백만 특성 규모 부적합 | 저자 명시: 수천 개 특성까지 실용적 (p.1) |
| 허위 쌍 포함 가능 | 특성 상관 시 GA²M이 사후 필터링하나, 완전 제거 보장 없음 |
| 최적 K 탐색 비용 | 포함할 쌍 수 K의 최적값 탐색이 고비용 (p.6) |
| FAST의 약한 상호작용 탐지 한계 | 합성 실험에서 11번째 쌍 $(x_8, x_{10})$ 누락 (p.7) |

---

## 3. 각 주장에 페이지/Figure/Table 번호 표시

| 주장 | 근거 위치 |
|------|----------|
| GA²M 모델 정의 (Eq. 2) | p.1 |
| 목적 함수 (Eq. 3) | p.2, Section 2 |
| FAST 알고리즘 개요 | p.3, Section 4.1 |
| 룩업 테이블 구성 (Algorithm 2) | p.4 |
| RSS 계산 효율화 (Eq. 8–9) | p.4, Section 4.1.3 |
| 복잡도 $O(b^2+N)$ | p.4–5, Section 4.1.4 |
| 2단계 구성(Two-stage) | p.5, Section 4.2 |
| 실험 데이터셋 목록 | Table 1, p.5 |
| 회귀 성능 비교 | Table 2, p.6 |
| 분류 성능 비교 | Table 3, p.6 |
| FAST 빈 수 민감도 | Figure 4, p.7 |
| 방법별 정밀도/시간 비교 | Figure 5, p.7 |
| 허위/진짜 쌍 히트맵 | Figure 6, p.8 |
| 상관 특성에서의 가중치 변화 | Figure 7, p.8 |
| 실제 데이터 계산 비용 | Figure 8, p.8 |
| 케이스 스터디 (MSLR10k) | Figure 9, p.9 |

---

## 4. 저자 직접 보고 결과 vs. 해석 분리

### 4-1. 저자가 직접 보고한 결과

**연구 주제**: GAM에 쌍별 상호작용을 추가하여 해석 가능성을 유지하면서 정확도를 향상

**방법 (수식)**:

$$g(E[y]) = \sum_i f_i(x_i) + \sum_{i < j} f_{ij}(x_i, x_j)$$

**직접 보고된 수치**:
- 회귀: GA²M FAST 평균 정규화 RMSE = **0.84 ± 0.20**, RF = **0.83 ± 0.17** (Table 2)
- 분류: GA²M FAST = **0.81 ± 0.21**, RF = **0.79 ± 0.26** (Table 3)
- FAST 실행 시간: 합성 데이터(10,000샘플) 기준 **~10초**, Grove는 **~1주일** (p.8)
- 합성 함수에서 FAST 평균 정밀도: **상위 10쌍 정확 탐지**, 11번째 쌍 누락 (Figure 5a)

### 4-2. 검토자(나)의 해석

| 항목 | 해석 |
|------|------|
| GA²M vs RF 성능 동등 | 저자는 "GA²M의 bias가 variance 감소로 상쇄"라고 설명하나, 이는 가설적 해석임. 데이터별 분산이 매우 크고(RF: ±0.17 vs ±0.26 등), 통계적 유의성 검정 미실시 |
| FAST의 $b=8$ 최적 | 합성 데이터 기반 결과이며 실제 데이터에서의 최적 $b$ 검증은 불충분 |
| 허위 쌍 자동 감쇄 | Figure 7의 사례는 단 2개 상관 수준($\rho=0.5, 0.95$)만 테스트; 일반화 주장에 한계 |
| BM25가 70위 | 흥미로운 발견이나 단일 데이터셋 결과로 일반화 불가 |

---

## 5. 통계적 취약점 및 비교 불가능한 수치 ⚠️

| 문제 | 세부 내용 |
|------|-----------|
| ⚠️ 유의성 검정 없음 | Table 2, 3에서 GA²M FAST와 RF의 차이에 대한 통계적 유의성 검정(t-test, Wilcoxon 등) 미실시 |
| ⚠️ 표준편차 중복 | 다수 데이터셋에서 GA²M FAST와 RF의 오차 범위가 중복됨 (예: CalHousing: 5.00±0.91 vs 4.90±0.81) |
| ⚠️ 평균 정규화 점수의 편향 | 정규화 기준이 GAM이고 데이터셋별 가중치가 동일하게 처리되어 특정 데이터셋의 극단값에 취약 |
| ⚠️ 합성 실험의 제한성 | FAST 정밀도 평가에 사용된 합성 함수(Eq. 10)가 단일 함수 패밀리로 편향 가능 |
| ⚠️ K=1000 임의성 | 최대 1,000쌍 선택 기준이 실험적 편의에 의한 것으로, 최적성 미검증 |
| ⚠️ 비교 불가 수치 | GA²M Rand, Coef, Order는 일부 데이터셋에만 적용 → 전체 비교 불완전 (Table 2, 3의 "-" 항목) |
| ⚠️ 단일 무작위 분할 | 80/20 train/test 분할을 단일 실시했는지, 교차 검증 횟수가 불명확 |

---

## 6. 논문이 답하지 않는 질문

| # | 미답 질문 |
|---|----------|
| Q1 | 최적 쌍 수 K를 데이터 기반으로 자동 결정하는 방법은? |
| Q2 | 3차 이상의 상호작용이 중요한 문제에서 어떻게 대응할 것인가? |
| Q3 | 연속형 특성의 256빈 이산화가 특정 분포(heavy-tail 등)에서 미치는 영향은? |
| Q4 | GA²M의 과적합 방지를 위한 정규화 전략은 무엇인가? |
| Q5 | 수백만 개 이상의 특성을 가진 초고차원 문제에서의 확장성은? |
| Q6 | GA²M에서 형상함수 $f_{ij}$의 신뢰 구간 또는 불확실성 정량화가 가능한가? |
| Q7 | 범주형 특성과 수치형 특성이 혼재하는 경우의 처리 방법은? |
| Q8 | 온라인/스트리밍 데이터 환경에서의 점진적 업데이트 가능성은? |
| Q9 | 다중 출력(multi-output) 또는 다중 클래스 분류에서의 적용 방법은? |
| Q10 | 특성 스케일 또는 전처리에 대한 민감도는? |

---

## 7. 가장 중요한 그림 5개 해석

### Figure 1 (p.3) — FAST의 입력 공간 분할 개념도

**해석**: FAST의 핵심 아이디어를 도시. 두 특성 $x_i$, $x_j$의 입력 공간에 각각 하나의 컷 $c_i$, $c_j$를 배치하여 4개의 사분면을 생성하고, 각 사분면의 평균값을 예측값으로 사용하는 극도로 단순한 예측기 $T_{ij}$를 구성. 이 단순함 덕분에 $O(b^2+N)$ 복잡도가 실현된다. 직관적으로, 진정한 상호작용이 존재하면 이 단순한 모델이 잔차를 크게 줄일 수 있음을 보여줌.

---

### Figure 4 (p.7) — FAST의 빈 수 민감도

**해석**: $b = 2$부터 256까지, 샘플 수 $N = 10^2$ ~ $10^6$에 걸쳐 평균 정밀도 측정. 10개 특성(좌)에서는 $b=8$ 근방에서 최고 성능이며, $N$이 클수록 성능 향상. 100개 특성(우)에서는 4,950쌍 중 진짜 상호작용을 찾아야 하므로 $N \geq 10^6$이 필요. **핵심 시사점**: $b=8$이 bias-variance 트레이드오프의 스위트스팟이며, 저자들이 FAST에 $b=8$을 사용한 근거를 실증적으로 지지함.

---

### Figure 5 (p.7) — 방법별 정밀도/계산 시간 비교

**해석**: 좌측(a)은 합성 함수(Eq. 10, N=10,000)에서의 평균 정밀도 비교. Grove와 ANOVA가 모든 11쌍을 정확히 탐지(AP ≈ 1.0)하며 FAST는 10쌍 탐지(AP ≈ 0.95). 우측(b)은 계산 시간으로, FAST는 약 10초, ANOVA는 수백 초, Grove는 약 $10^6$초(약 11일). **핵심 시사점**: FAST는 약 5% 정밀도 손실로 3~4 오더의 속도 이득을 달성 → 실용적 최선의 선택임을 강하게 지지.

---

### Figure 7 (p.8) — 상관 특성 환경에서의 상호작용 가중치 변화

**해석**: $x_1$과 $x_6$가 상관($\rho=0.5$ 또는 $0.95$)인 상황에서 FAST가 허위 쌍 $(x_2, x_6)$를 선택하더라도, 그래디언트 부스팅 반복이 진행될수록 해당 항의 가중치 $\sqrt{E[f_{ij}^2]}$가 빠르게 감소함을 보임. 이는 GA²M이 **자체적인 사후 필터링 메커니즘**을 내장하고 있음을 의미하며, 허위 쌍 포함이 최종 모델의 해석 가능성을 심각하게 훼손하지 않음을 시사.

---

### Figure 9 (p.9) — MSLR10k 케이스 스터디

**해석**: 상위 2행은 10개 주요 1차원 형상함수($f_i$ vs $x_i$ 곡선), 하위 2행은 10개 주요 쌍별 상호작용 히트맵( $f_{ij}(x_i, x_j)$ ). 가중치(각 서브플롯 상단 수치)로 중요도 정량화. 주목할 점은 BM25(정보검색 분야 표준 피처)가 shaping 후 70위에 불과하고, IDF 등이 높은 가중치를 가짐. 히트맵들은 additive 항만으로는 표현 불가능한 복잡한 비선형 패턴을 명확히 드러냄. **실용적 해석 가능성의 구체적 증거**.

---

## 8. 결론, 시사점, 후속 연구

### 8-1. 저자 제시 시사점 및 후속 연구 계획

**저자 제시 시사점** (Section 6, p.8):
- 쌍별 상호작용 추가가 해석 가능성을 유지하면서 GAM 대비 정확도를 크게 향상
- 많은 현실 문제에서 2차 상호작용까지만으로 전체 복잡도 모델의 성능에 근접 가능
- FAST가 기존 방법 대비 수천 배 빠르면서 유사한 정밀도 제공

**저자가 명시한 후속 연구**: 논문 내에 구체적 후속 연구 계획은 명시되어 있지 **않음** (단, 고차원 문제로의 확장 필요성 암시).

### 8-1. 모델의 일반화 성능 향상 가능성

**현재 일반화 관련 내용**:

저자들은 GA²M의 성능이 랜덤 포레스트에 근접하는 이유를 다음과 같이 설명한다 (p.2):

> "The performance may be due to the difficulty of estimating intrinsically high dimensional functions from limited data, suggesting that the **bias associated with the GA²M structure is outweighed by a drop in variance**."

즉, GA²M의 구조적 편향(고차 상호작용 무시)이 차원의 저주에 따른 분산 감소 이익으로 상쇄됨.

**일반화 성능 향상 가능성**:

| 방향 | 내용 |
|------|------|
| 정규화 강화 | $f_i$ 및 $f_{ij}$에 L1/L2 정규화 추가로 과적합 억제 및 희소 모델 유도 |
| 교차 검증 기반 K 선택 | 현재 임의 설정된 K=1,000을 교차 검증으로 최적화 |
| 베이지안 앙상블 | 불확실성 정량화를 통해 저밀도 영역에서의 예측 신뢰도 향상 |
| 전이 학습 적용 | 대규모 데이터로 학습한 형상함수를 소규모 데이터에 전이 |
| 도메인 지식 통합 | 특정 쌍에 단조성(monotonicity) 제약 추가로 과적합 방지 |

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **중요 고지**: 아래는 본 논문의 내용과 AI 분야의 공개된 연구 흐름을 바탕으로 작성하였으나, 각 후속 논문의 세부 수치 및 내용에 대해서는 원문 직접 확인을 권장합니다. 제가 확실히 알지 못하는 구체적 실험 수치는 포함하지 않았습니다.

| 후속 연구 | 핵심 내용 | GA²M과의 관계 |
|----------|-----------|--------------|
| **EBM (Nori et al., 2019; Microsoft Research)** — *InterpretML* | GA²M을 기반으로 한 구현체로, 단조성 제약, 신뢰 구간, 빠른 학습 지원 | GA²M의 직접 후속/구현 |
| **NODE-GAM (Chang et al., 2022, ICLR)** | 신경망 기반 GAM으로 미분 가능한 구조, GPU 가속 학습 | GA²M의 신경망 확장 |
| **GAMI-Net (Yang et al., 2021, Pattern Recognition)** | 신경망 기반 GA²M으로 희소 상호작용 학습 | GA²M의 딥러닝 버전 |
| **TabNet (Arik & Pfister, 2021, AAAI)** | 어텐션 기반 해석 가능 표 형 데이터 학습 | 다른 접근법이나 해석 가능성 목표 공유 |
| **SHAP (Lundberg & Lee, 2017, NeurIPS)** | Shapley 값 기반 사후 설명 | GA²M의 사전 해석 vs. 사후 설명의 대비 |
| **ExNN (Yang et al., 2021)** | 계층적 상호작용 구조를 신경망으로 학습 | 3차 이상 상호작용으로 확장 |

**참고 자료**:
- Nori, H., Jenkins, S., Koch, P., & Caruana, R. (2019). InterpretML: A unified framework for machine learning interpretability. *arXiv:1909.09223*.
- Chang, C. H., Tan, S., Lengerich, B., Goldenberg, A., & Caruana, R. (2022). How interpretable and accurate are local explanations of neural networks? *ICLR 2022*.
- Yang, Z., Zhang, A., & Sudjianto, A. (2021). GAMI-Net: An explainable neural network based on generalized additive models with structured interactions. *Pattern Recognition*, 120, 108192.

**GA²M이 후속 연구에 미치는 영향**:

1. **해석 가능 ML의 기준선 정립**: GA²M은 해석 가능성 연구에서 비교 기준(baseline)으로 광범위하게 인용됨
2. **InterpretML 패키지**: Microsoft의 오픈소스 해석 가능 ML 라이브러리의 핵심 모델로 채택
3. **의료 AI 적용**: 임상 의사결정 지원 시스템에서 GAM/GA²M 계열 모델이 활발히 채택됨
4. **FAST 아이디어**: 효율적 상호작용 탐지를 위한 히스토그램 기반 접근법이 후속 연구에 영향

**앞으로 연구 시 고려할 점**:

| 고려사항 | 세부 내용 |
|---------|-----------|
| **LLM 시대의 해석 가능성** | LLM 기반 표 형 데이터 처리가 부상함에 따라 GA²M과의 성능/해석 가능성 비교 필요 |
| **불확실성 정량화** | GA²M의 예측 신뢰 구간 제공이 임상/금융 응용의 필수 요건 |
| **인과 추론과의 결합** | 상호작용 탐지를 인과 그래프 학습과 연계하는 연구 기회 |
| **연속 학습** | 데이터 드리프트 환경에서 형상함수의 점진적 업데이트 방법론 필요 |
| **공정성(Fairness)** | 보호 속성과의 상호작용 항이 모델 편향에 미치는 영향 분석 필요 |
| **고차 상호작용** | 3차원 이상 상호작용의 효율적 탐지 및 시각화 방법론 부재 |

---

**주요 참고 자료 목록**

1. Lou, Y., Caruana, R., Gehrke, J., & Hooker, G. (2013). *Accurate intelligible models with pairwise interactions*. KDD '13.
2. Lou, Y., Caruana, R., & Gehrke, J. (2012). *Intelligible models for classification and regression*. KDD '12. [논문 내 참고문헌 19]
3. Friedman, J. H. (2001). *Greedy function approximation: A gradient boosting machine*. Annals of Statistics, 29, 1189–1232. [참고문헌 10]
4. Hastie, T., & Tibshirani, R. (1990). *Generalized additive models*. Chapman & Hall/CRC. [참고문헌 13]
5. Sorokina, D., Caruana, R., Riedewald, M., & Fink, D. (2008). *Detecting statistical interactions with additive groves of trees*. ICML. [참고문헌 22]
6. Wood, S. (2006). *Generalized additive models: An introduction with R*. CRC Press. [참고문헌 25]
7. Nori, H., Jenkins, S., Koch, P., & Caruana, R. (2019). *InterpretML: A unified framework for machine learning interpretability*. arXiv:1909.09223.
8. Yang, Z., Zhang, A., & Sudjianto, A. (2021). *GAMI-Net: An explainable neural network based on generalized additive models with structured interactions*. Pattern Recognition, 120, 108192.
9. Hooker, G. (2007). *Generalized functional ANOVA diagnostics for high-dimensional functions of dependent variables*. Journal of Computational and Graphical Statistics, 16(3), 709–732. [참고문헌 15]
