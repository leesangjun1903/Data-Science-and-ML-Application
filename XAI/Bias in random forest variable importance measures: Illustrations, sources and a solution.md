# Bias in random forest variable importance measures: Illustrations, sources and a solution

Strobl 등(2007)은 랜덤포레스트(Random Forest)의 변수 중요도 측정치가 예측변수의 척도(연속형 vs. 범주형)나 범주 수가 서로 다를 때 신뢰할 수 없다는 것을 시뮬레이션과 실제 RNA 편집 데이터 분석을 통해 보였다. 편향의 원인은 (1) CART 기반 개별 트리의 분할 변수 선택 편향과 (2) 복원추출(bootstrap) 방식이 유발하는 인위적 연관성 두 가지로 규명되었다. 해결책으로 조건부 추론 트리(conditional inference tree) 기반의 `cforest` 함수를 비복원추출(subsampling without replacement)과 함께 사용할 것을 제안하였다.

## 1-1. 연구의 목적과 필요성

유전체학·생물정보학에서는 "small n large p" 문제(적은 표본, 많은 변수)가 흔하며, 랜덤포레스트는 변수 선택과 예측을 동시에 수행할 수 있어 널리 쓰인다(p.2). 그러나 SNP, 아미노산 서열, 연속형 변수(예: folding energy)처럼 척도·범주 수가 다른 변수들이 혼재된 경우, 기존 변수 중요도 측정치(Gini importance, permutation importance, selection frequency)가 실제 중요도가 아닌 변수의 "카테고리 수"에 의해 왜곡될 위험이 있다(p.1, Abstract). 이는 잘못된 유전 마커나 예측변수를 선택하게 하여 후속 생물학적 해석에 심각한 오류를 초래할 수 있으므로, 편향의 근원을 규명하고 신뢰할 수 있는 대안을 제시하는 것이 연구의 목적이다.

## 2. 핵심 주장과 근거 (표)

| 핵심 주장 | 근거 (Figure/Table/Page) |
|---|---|
| randomForest의 selection frequency는 범주 수가 많은 변수를 인위적으로 선호한다 | Fig.1 (p.6), Fig.5 (p.10) |
| Gini importance가 가장 심하게 편향된다 | Fig.2 (p.7), Fig.6 (p.11) |
| Permutation importance는 평균은 편향 없으나 분산이 범주 수에 따라 증가한다 | Fig.3, Fig.4 (p.8-9), Fig.7, Fig.8 (p.12-13) |
| cforest + bootstrap도 여전히 일부 편향 존재 | Fig.1 하단좌, Fig.5 하단좌 (p.6, p.10) |
| cforest + subsampling(무복원)만이 편향 없는 결과 산출 | Fig.1, 3, 4, 5, 7, 8 하단우 (p.6-13) |
| 편향의 원인 ①: 개별 트리(CART)의 변수선택 편향(다범주 선호) | Fig.10 (p.17), p.14-16 |
| 편향의 원인 ②: Bootstrap 복원추출이 인위적 연관성 유발 | Fig.11 (p.18), p.17-18 |
| cforest(무복원)가 분류 정확도(오분류율)에서도 randomForest보다 우수 | Table 4 (p.16), Table 5 (p.16) |
| 정보성 변수 탐지율(power case)도 cforest가 더 높음 | Table 3 (p.15) |

## 2-1. 문제, 방법(수식), 모델 구조, 성능, 한계

**해결하고자 하는 문제**: 서로 다른 척도/범주 수를 가진 예측변수들이 혼재할 때, randomForest 기반 변수 중요도 측정치의 신뢰성 결여.

**제안 방법 (수식 포함)**:

1) Permutation importance의 정의: 변수 $X_j$를 무작위로 치환(permute)한 후 예측 정확도의 감소량을 측정

$$VI(X_j) = \text{Accuracy}_{original} - \text{Accuracy}_{permuted(X_j)}$$

2) 시뮬레이션 설계 (Null case, p.4, Table 2):
$$Y \sim B(0.5)$$

3) Power case (Table 2, p.4):
$$Y|X_2=1 \sim B(0.5 - relevance), \quad Y|X_2=2 \sim B(0.5 + relevance)$$
relevance $\in \{0.05, 0.1, 0.15, 0.2\}$

4) 예측변수 분포 (Table 1, p.4):
$$X_1 \sim N(0,1), \quad X_2 \sim M(2), \quad X_3 \sim M(4), \quad X_4 \sim M(10), \quad X_5 \sim M(20)$$

5) Subsampling 비율: 원 표본크기 $n$의 0.632배 (bootstrap 시 기대되는 unique sample 비율과 동일하게 설정, p.4)

6) 정확한 식별률의 표준오차 (Table 3 각주, p.15):
$$SE = \sqrt{\frac{r(1-r)}{1000}}$$

**모델 구조**: cforest는 Gini index 대신 조건부 추론 프레임워크(conditional inference framework, Hothorn et al. 2006)를 사용하여 각 분할에서 독립성 검정의 p-value를 최소화하는 변수를 선택하며, 이 p-value 계산 시 변수의 범주 수를 자유도에 반영하여 다범주 변수에 대한 편향을 제거한다(p.15-16).

**성능 향상**: Table 3에서 relevance=0.2일 때 cforest(무복원, scaled)의 정보변수 정확 식별률은 0.994로 randomForest(복원, scaled)의 0.956보다 높음. Table 4에서 오분류율도 cforest가 최대 5%p 이상 낮음(relevance=0.2일 때 0.3491~0.3384 vs. 0.4028~0.4026).

**한계**: cforest는 조건부 추론 계산 비용으로 인해 randomForest보다 계산 시간이 훨씬 김(C-to-U 데이터 기준 cforest 4.82~8.38초 vs. randomForest 0.18~0.24초, p.19). 또한 논문은 분류(classification) 문제만 다루며 회귀(regression) 문제는 다루지 않음(p.3, Methods).

## 3. 페이지/Figure 표시
(위 표 2, 2-1에 개별 표기 완료)

## 4. 저자 보고 결과 vs. 해석 분리

**저자 직접 보고**:
- "the Gini importance is most strongly biased" (p.4, Results and discussion)
- "For all degrees of dependence between X2 and the response Y the cforest function detects the informative variable more reliably" (p.9)
- "the randomForest function can produce a higher mean misclassification rate than the cforest function" (p.11)
- Table 4, 5의 수치는 1000회/100회 시뮬레이션 평균값으로 직접 보고됨.

**본인(AI) 해석 (분리)**:
- Table 4, 5의 misclassification rate 차이(예: 0.40 vs 0.35, 약 5%p)는 통계적으로 유의미하나 실무적으로는 "moderate"하다고 저자도 인정(p.11) — 이는 극단적 시나리오에서는 차이가 더 커질 수 있다는 저자의 추측성 언급이며 실증되지 않음.
- 계산 시간 차이(cforest가 최대 46배 느림)는 대규모 유전체 데이터(수천~수만 변수)에 실제 적용 시 실무적 병목이 될 수 있다는 것은 본문에 명시적으로 논의되지 않은 추가 해석임.

## 5. 통계적으로 취약한 부분 및 비교 불가능한 수치

- **Table 4, 5의 misclassification rate**: 표준오차가 매우 작게 보고되어(예: 0.0014) 유의성이 있어 보이나, 이는 1000회 반복의 표본 평균 표준오차일 뿐 실제 임상적/생물학적 효과크기(effect size)는 아님 — 실질적 중요성 판단에 주의 필요.
- **C-to-U 데이터 결과(Table 5)와 원 논문[11] 결과 비교 불가**: "Differences to the accuracy values reported in [11] are most likely due to their use of a different validation scheme, that is not reported in detail in [11]"(p.13) — 저자 스스로 direct comparison이 불가능함을 명시.
- **Scaled permutation importance의 절대값 해석 금지**: "the scaled variable importance of the randomForest function depends on the number of trees grown"(p.5) — 즉 randomForest의 scaled importance 수치는 트리 개수에 따라 달라지므로 절대적 비교 지표로 부적절함.
- **Subsampling 비율(0.632) 선택의 임의성**: "Other fractions...are possible, for instance 0.5"(p.4) — 이 비율 선택이 결과에 미치는 민감도 분석은 수행되지 않음.

## 6. 문서가 답하지 않는 질문

- 회귀(regression) 문제에서도 동일한 편향 패턴이 나타나는가?
- 변수 개수(p)가 매우 커지는 고차원 상황(수천~수만 변수)에서 cforest의 계산 비용은 실제로 어느 정도까지 확장 가능한가?
- Subsampling 비율(0.5, 0.632 외 다른 값)에 따른 민감도는?
- 다중공선성이 있는 변수들 사이에서 cforest의 성능은 어떠한가?
- 연속형 변수 내에서도 결측치나 이상치가 있을 때 편향에 영향이 있는가?
- van der Laan(2006)이 제안한 변수 중요도의 통계적 추론 프레임워크를 랜덤포레스트에 실제로 적용한 결과는? (미래 연구로만 언급, p.19)

## 7. 가장 중요한 그림 5개 해석

1. **Figure 1 (p.6, Null case selection frequency)**: 4개 패널 비교. randomForest(top)와 cforest+bootstrap(하단좌)은 범주 수가 많은 X5(20개 범주)를 압도적으로 선호(선택빈도 약 200)하는 반면, cforest+subsampling(하단우)만 5개 변수 모두 균등한 선택빈도(~100)를 보여 편향 없음을 시각적으로 증명하는 핵심 그림.

2. **Figure 2 (p.7, Null case Gini importance)**: X5의 Gini importance가 X2 대비 약 10배 이상 높게 나타나(20~25 vs 2), Gini 기준이 가장 심각하게 왜곡됨을 보여줌. 이는 randomForest 사용자들이 흔히 사용하는 기본 중요도 지표의 근본적 결함을 지적.

3. **Figure 5 (p.10, Power case selection frequency)**: 실제 정보성 변수 X2가 있음에도, randomForest(top)는 X2가 아닌 X5(비정보 변수)를 더 자주 선택하여(약 175 vs 145) 완전히 잘못된 결론을 유도할 수 있음을 보여줌. cforest+subsampling(하단우)에서만 X2가 명확히 두드러짐(~200 vs 나머지 ~75).

4. **Figure 10 (p.17, Variable selection bias in individual trees, rpart vs ctree)**: 근본 원인을 직접 증명하는 그림. rpart(CART)는 X5를 90% 가까이 선택하는 극단적 편향을 보이나, ctree(조건부 추론)는 5개 변수 모두 약 20%로 균등 선택 — 편향의 "뿌리"가 개별 트리 알고리즘에 있음을 명확히 보임.

5. **Figure 11 (p.18, Effects induced by bootstrapping)**: Bootstrap 전(좌) χ² test p-value는 균등분포(median 0.5)이나, bootstrap 후(우)에는 범주 수가 많을수록(X4, X5) p-value가 0에 가깝게 쏠림 — 복원추출 자체가 다범주 변수에 인위적 연관성을 만들어낸다는 두 번째 편향 메커니즘을 통계적으로 입증하는 결정적 그림.

## 8. 결론 요약 및 후속 연구 방향

**저자 제시 시사점**: 척도/범주가 혼재된 데이터에서는 randomForest의 어떤 중요도 지표도 신뢰할 수 없으며, cforest(조건부추론트리)+subsampling(무복원)이 유일하게 신뢰 가능한 대안이다(p.19, Conclusion). 다만 연속형 변수만 있거나 모든 범주형 변수의 범주 수가 동일한 경우(예: 대부분의 유전자발현 데이터)에는 기존 randomForest도 문제없다고 명시(p.19).

**저자가 언급한 후속 연구**: van der Laan(2006)의 통계적 추론 프레임워크에 착안하여, 변수 중요도에 대한 가설검정/신뢰구간 등 formal statistical inference 방법을 개발하는 것이 향후 과제로 제시됨(p.19-20).

### 8-1. 모델의 일반화 성능 향상 가능성

이 논문의 제안(unbiased tree + subsampling)은 특정 변수(다범주)에 대한 과적합성 선호를 제거함으로써 모델이 실제 신호가 있는 변수에 집중하게 하여 일반화 성능(test set 성능)을 개선한다. Table 4, 5에서 cforest가 randomForest보다 test set misclassification rate가 낮다는 것이 실증적 근거이다. 이는 변수선택 편향이 곧 과적합(비정보 변수의 인위적 분할 사용)으로 이어져 일반화 성능을 저하시킨다는 것을 시사하며, 편향 제거가 해석가능성뿐 아니라 예측 성능 자체의 일반화에도 기여함을 보여준다. 다만 본 논문의 실험은 저차원(p=5~44) 시뮬레이션에 국한되어 있어, 고차원 유전체 데이터(수만 변수)에서도 동일한 일반화 이득이 유지되는지는 후속 검증이 필요하다.

### 8-2. 2020년 이후 관련 최신 연구 비교 분석 (참고: 문서 내 명시적 정보 없음, 일반적 학술 동향 기반 서술 — 정확도 확신 낮음)

이 부분은 본 PDF 문서에 포함된 내용이 아니므로, 문서 근거만으로 답변하기 어렵습니다. 2020년 이후 문헌에 대한 구체적 서지 정보(논문 제목, 저자, 발표 연도, 정확한 결론)를 문서에서 확인할 수 없어 정확한 비교 분석을 제공하면 사실이 아닌 내용을 지어낼 위험이 있습니다. 일반적으로 알려진 방향성만 조심스럽게 언급하면: Strobl et al.(2007, 2008 논문 "Conditional variable importance for random forests")의 conditional permutation importance 개념이 이후 발전되었고, scikit-learn 등 주요 ML 라이브러리에서도 permutation importance와 impurity-based importance의 편향 문제가 문서화되어 있다는 점 정도는 업계에서 널리 알려진 사실이나, 이를 2020년 이후 특정 논문과 정확히 매칭하여 인용하는 것은 본 문서만으로는 검증 불가능합니다. **정확한 최신 문헌 비교를 원하시면 별도의 문헌 검색이 필요합니다.**

**앞으로의 연구에 미치는 영향(문서 기반)**: 이 논문은 (1) 변수 중요도 사용 시 반드시 unbiased tree 알고리즘 여부를 확인해야 한다는 방법론적 경각심을 제공했고, (2) `party`/`cforest` R 패키지의 실무 채택을 촉진했으며(p.1, Abstract), (3) 이후 연구자들이 새로운 변수 중요도 지표를 제안할 때 "null case에서 균등해야 한다"는 검증 기준(본 논문의 시뮬레이션 설계, Table 1-2)을 표준적 벤치마크로 참고하게 되는 계기를 마련했다.

**향후 연구 시 고려할 점**: ① 변수 선택 알고리즘의 종류(편향 여부)를 먼저 확인할 것, ② 복원추출 여부가 결과에 미치는 영향을 반드시 점검할 것, ③ scaled importance의 절대값을 트리 개수와 무관하게 해석하지 않도록 주의할 것(p.5), ④ 무작위 permutation에 기반한 지표는 여러 random seed로 반복 검증할 것(p.14, "the analysis should be repeated...to test the stability").

---
**참고자료**: Strobl, C., Boulesteix, A.-L., Zeileis, A., & Hothorn, T. (2007). "Bias in random forest variable importance measures: Illustrations, sources and a solution." *BMC Bioinformatics*, 8:25. doi:10.1186/1471-2105-8-25.
