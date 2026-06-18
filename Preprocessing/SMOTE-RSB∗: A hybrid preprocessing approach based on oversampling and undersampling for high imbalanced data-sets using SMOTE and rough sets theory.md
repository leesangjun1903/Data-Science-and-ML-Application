# SMOTE-RSB∗: A hybrid preprocessing approach based on oversampling and undersampling for high imbalanced data-sets using SMOTE and rough sets theory

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

SMOTE-RSB∗는 **고불균형 데이터셋**(불균형 비율 $IR \geq 9$, 즉 소수 클래스가 전체의 10% 미만)을 위한 **하이브리드 전처리 방법**이다. SMOTE로 생성된 합성 샘플 중 **노이즈가 될 수 있는 경계 영역 샘플을 제거**하기 위해, 거친 집합 이론(Rough Set Theory, RST)의 하부 근사(lower approximation)를 활용한다.

### 주요 기여

| 기여 항목 | 내용 |
|---|---|
| 새로운 하이브리드 전처리 알고리즘 | SMOTE(오버샘플링) + RST 기반 편집(언더샘플링) 결합 |
| RST를 데이터 정제에 적용 | 소수 클래스의 하부 근사에 속하는 합성 샘플만 보존 |
| 실험적 검증 | UCI 저장소 44개 고불균형 데이터셋, C4.5 분류기, AUC 기준 |
| 통계적 우월성 입증 | Iman-Davenport + Holm 사후 검정으로 기존 6개 방법 대비 통계적으로 유의미한 성능 향상 |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

**클래스 불균형 문제(Class Imbalance Problem)**는 다수 클래스 인스턴스 수가 소수 클래스보다 압도적으로 많을 때 발생하며, 특히 $IR \geq 9$인 고불균형 상황에서 심각하다.

불균형 비율은 다음과 같이 정의된다:

$$IR = \frac{\text{다수 클래스 인스턴스 수}}{\text{소수 클래스 인스턴스 수}}$$

이로 인해 분류기가 소수 클래스를 노이즈로 취급하고, 전반적 정확도는 높지만 소수 클래스에 대한 분류 성능이 저하된다.

또한 **SMOTE의 과일반화(overgeneralization) 문제**가 있다: 소수 클래스 합성 샘플이 다수 클래스 영역에 침범하여 경계 영역에 노이즈를 생성할 수 있다.

---

### 2.2 제안하는 방법: SMOTE-RSB∗

#### 알고리즘 2단계 구조

**[1단계]** SMOTE를 이용한 소수 클래스 오버샘플링

SMOTE의 합성 샘플 생성 수식:

$$x_{new} = x_i + \lambda \cdot (x_{nn} - x_i), \quad \lambda \in [0, 1]$$

여기서 $x_i$는 선택된 소수 클래스 샘플, $x_{nn}$은 $k$-최근접 이웃 중 하나, $\lambda$는 [0,1] 사이의 랜덤 값이다.

---

**[2단계]** RST 기반 합성 샘플 정제(편집)

유사도 행렬(Similarity Matrix)을 구축한다:

$$\text{similarityMatrix}(i,j) = \frac{\displaystyle\sum_{k=1}^{n} w_k \cdot \delta_k(x_{ik}, x_{jk})}{M} $$

여기서:
- $n$: 특성 수
- $w_k$: 특성 $k$의 가중치
- $M$: 동치 관계에서 고려하는 특성 수
- $B$: 동치 관계에서 고려하는 특성 집합

가중치 $w_k$:

$$w_k = \begin{cases} 1 & \text{if } k \in B \\ 0 & \text{otherwise} \end{cases} $$

이산형 속성에 대한 비교 함수 $\delta_k$:

$$\delta_k(x_{ik}, x_{jk}) = \begin{cases} 1 & \text{if } x_{ik} = x_{jk} \\ 0 & \text{otherwise} \end{cases} $$

연속형 속성에 대한 비교 함수 $\delta_k$:

$$\delta_k(x_{ik}, x_{jk}) = 1 - \frac{|x_{ik} - x_{jk}|}{\max A_k - \min A_k} $$

---

**RST 하부 근사 및 경계 영역 정의:**

집합 $X \subset U$에 대한 유사도 관계 $R'$를 사용한 **하부 근사(lower approximation)**:

$$B_*(X) = \{x \in X : R'(x) \subseteq X\} $$

**상부 근사(upper approximation)**:

$$B^*(X) = \bigcup_{x \in X} R'(x) $$

**경계 영역(boundary region)**:

```math
BN_B(X) = B^*(X) - B_*(X)
```

→ $BN_B(X) = \emptyset$이면 집합 $X$는 $R'$에 대해 정확(crisp). $BN_B(X) \neq \emptyset$이면 근사적(approximate).

**핵심 아이디어:** 합성 샘플 중 소수 클래스의 **하부 근사 $B_*(X)$에 속하는 것만** 최종 훈련 세트에 포함시키고, 경계 영역에 있는 샘플은 노이즈로 간주하여 제거한다.

---

#### AUC 평가 지표

$$\text{AUC} = \frac{1 + \text{TP}_{\text{rate}} - \text{FP}_{\text{rate}}}{2} $$

여기서:
- $\text{TP}_{\text{rate}} = \frac{TP}{TP + FN}$
- $\text{FP}_{\text{rate}} = \frac{FP}{FP + TN}$

---

### 2.3 모델 구조 (알고리즘 흐름)

```
입력: 불균형 훈련 데이터셋
│
├── Step 1: SMOTE로 소수 클래스 합성 샘플 생성 (50:50 목표)
│
├── Step 2: resultSet ← 원본 인스턴스 포함
│
├── Step 3: 원본+합성 샘플 간 similarityMatrix 구축
│          (similarityValue 초기값 = 0.4)
│
├── Step 4: while (resultSet is empty) AND (similarityValue ≤ 0.9)
│           │
│           ├── 각 합성 샘플 i에 대해 다수 클래스 j와 유사도 비교
│           ├── similarityMatrix(i,j) > similarityValue이면 cont[i]++
│           ├── cont[i] == 0 → 하부 근사 B*에 속함 → resultSet에 추가
│           └── similarityValue += 0.05 (재시도)
│
└── Step 5: resultSet이 여전히 비어있으면
           → 모든 합성 샘플 포함 (SMOTE 결과 그대로 사용)

출력: 정제된 균형 훈련 데이터셋
```

---

### 2.4 성능 향상

| 평가 기준 | 결과 |
|---|---|
| 평균 AUC | S-RSB∗: **0.8402** (최고), SMOTE: 0.8142, S-ENN: 0.8247, S-TL: 0.8247 |
| Friedman 랭킹 | S-RSB∗: **2.61364** (1위), Borderline-SMOTE1: 5.36364 (최하위) |
| 1위 달성 횟수 | 44개 데이터셋 중 **15회 단독 1위, 3회 공동 1위** |
| 통계적 검정 | Iman-Davenport: p ≈ 0, Holm 사후검정: 모든 비교 대상 Reject |

---

### 2.5 한계

1. **단일 분류기 의존성**: C4.5만을 학습 알고리즘으로 사용하여 SVM, 랜덤 포레스트, 딥러닝 등 다른 분류기에서의 성능 일반화가 검증되지 않음.

2. **similarityValue 민감성**: 유사도 임계값이 결과에 크게 영향을 미치며, 초기값(0.4) 및 증가분(0.05)이 휴리스틱하게 설정됨.

3. **계산 복잡도**: 유사도 행렬 구축이 $O(n^2)$ 복잡도를 가져 대규모 데이터셋에서 비용이 클 수 있음.

4. **이진 분류 한정**: 실험이 이진 분류에만 집중되어 다중 클래스 불균형 문제에의 직접 적용 여부가 불명확.

5. **극단적 불균형 한계**: `abalone19` ($IR = 128.87$)에서 AUC 0.5244로 다른 데이터셋 대비 성능이 저조하여, 극단적 불균형에서의 한계 노출.

6. **특성 선택 미고려**: 유사도 행렬 계산에서 특성 중요도를 동등하게 취급.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화 성능 향상 메커니즘

SMOTE-RSB∗의 핵심 일반화 강점은 **"하부 근사 기반 노이즈 제거"**에 있다.

$$\text{훈련 데이터 품질} = \frac{|B_*(X_{\text{minority}})|}{|X_{\text{synthetic}}|}$$

이 비율이 높을수록 분류기가 학습하는 소수 클래스 결정 경계가 **더 명확하고 안정적**이 된다.

**일반화 성능 향상 경로:**

```
SMOTE 과일반화 문제
  ↓ (해결)
RST 하부 근사: "절대적으로 소수 클래스에 속하는" 샘플만 선택
  ↓
결정 경계의 노이즈 감소
  ↓
분류기의 소수 클래스 과적합 억제
  ↓
일반화 성능(AUC) 향상
```

### 3.2 분류기 독립성(Classifier Independence)에 의한 일반화

데이터 레벨 접근법은 **전처리 단계**이므로 분류기 선택과 무관하다:

$$\hat{y} = f_{\text{classifier}}(D_{\text{preprocessed}})$$

동일한 전처리 데이터 $D_{\text{preprocessed}}$가 다양한 분류기에 입력될 수 있어, 전처리 비용이 1회만 발생하며 여러 분류기에 재사용 가능하다.

### 3.3 Step 5 (폴백 메커니즘)의 역할

모든 합성 샘플이 다수 클래스와 유사한 극단적 상황에서도:

$$\text{if } B_*(X_{\text{synthetic}}) = \emptyset \Rightarrow \text{resultSet} \leftarrow \text{All SMOTE samples}$$

이 설계는 최악의 경우에도 기존 SMOTE 수준의 성능을 보장하여 **하한 성능을 보장**한다.

### 3.4 일반화 관련 한계 및 고려사항

- **도메인 이전(Domain Transfer)**: 44개 데이터셋 모두 UCI 저장소 기반으로, 의료 영상, 텍스트, 시계열 등 다른 도메인에서의 일반화는 미검증.
- **고차원 데이터**: 고차원에서 유사도 행렬의 거리 척도 신뢰도가 낮아지는 "차원의 저주" 문제가 발생할 수 있음.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4.1 연구에 미치는 영향

**① 데이터 정제 이론의 확장 가능성**

RST를 데이터 정제에 적용한 것은 이후 퍼지 거친 집합(Fuzzy Rough Set), 변수 정밀도 거친 집합(Variable Precision Rough Set) 등의 응용으로 이어질 수 있다.

**② 하이브리드 접근법의 방향성 제시**

"생성 + 정제"의 두 단계 파이프라인은 이후 GAN 기반 오버샘플링 + 필터링, VAE 기반 생성 + 정제 등의 연구에 영향을 미친다.

**③ 불균형 학습 벤치마크 확립**

44개 고불균형 UCI 데이터셋, 비모수 통계 검정(Friedman + Holm)의 체계적 사용이 후속 연구의 표준 실험 프레임워크로 활용된다.

---

### 4.2 2020년 이후 최신 연구와의 비교 분석

> ⚠️ **주의**: 아래 2020년 이후 연구 목록은 제가 접근 가능한 학습 데이터 기반의 일반적 지식이며, 각 논문의 세부 수치는 원문을 직접 확인하시기 바랍니다.

#### 비교 테이블

| 방법 | 연도 | 핵심 아이디어 | SMOTE-RSB∗ 대비 개선점 | 한계 |
|---|---|---|---|---|
| **SMOTE-RSB∗** | 2012 | RST 하부 근사 기반 정제 | 기준선 | C4.5 한정, 계산 비용 |
| **CTGAN** | 2019 | GAN 기반 표형 데이터 합성 | 데이터 분포를 직접 모델링 | 훈련 불안정, 소규모 데이터 취약 |
| **SMOTIFIED-GAN** | ~2021 | SMOTE + GAN 결합 | 더 현실적인 합성 샘플 | 계산 비용 높음 |
| **ProWSyn** | 2021 | 근접도 기반 가중 합성 | 클래스 분포 보존 강화 | 파라미터 민감성 |
| **AMSCO** | 2022 | 적응적 다수결 언더샘플링 + 오버샘플링 | 클러스터 구조 고려 | 클러스터 수 설정 어려움 |
| **Transformer 기반 접근** | 2023~ | 어텐션 메커니즘으로 중요 소수 샘플 식별 | 고차원 데이터 처리 우수 | 레이블 데이터 다량 필요 |

#### 주요 트렌드와 SMOTE-RSB∗의 위치

```
2012: SMOTE-RSB∗ (RST 기반 정제) ← 현재 논문
2014~2018: ADASYN, SVM-SMOTE, Cluster-SMOTE (다양한 오버샘플링 변형)
2019~2021: 딥러닝 기반 합성 (GAN, VAE, 오토인코더)
2022~현재: 불균형 + 대규모/스트리밍/연합학습 통합, LLM 기반 데이터 증강
```

SMOTE-RSB∗는 **해석 가능한 정제 메커니즘**(RST)을 통해 블랙박스 GAN 기반 방법 대비 **설명 가능성(Explainability)** 측면에서 여전히 유의미하다.

---

### 4.3 앞으로 연구 시 고려할 점

**① 다양한 분류기로 확장 검증**
- SVM, Random Forest, XGBoost, 딥러닝 모델 등에서 성능 검증 필요.

**② 적응적 유사도 임계값 설계**

현재 고정된 증가 방식:
$$\text{similarityValue}_{t+1} = \text{similarityValue}_t + 0.05$$

→ 데이터 특성에 따라 자동으로 최적 임계값을 탐색하는 베이지안 최적화 또는 메타학습 기반 접근이 필요:

$$\theta^* = \arg\max_{\theta} \mathbb{E}_{D \sim \mathcal{D}}[\text{AUC}(f(\text{SMOTE-RSB*}(D; \theta)))]$$

**③ 고차원/희소 데이터 대응**
- 텍스트, 유전체, 이미지 특성 벡터 등 고차원 데이터에서 유클리드 거리 기반 유사도의 한계 극복 필요 (코사인 유사도, 학습된 임베딩 거리 활용).

**④ 스트리밍/온라인 불균형 학습 통합**
- 실시간으로 도착하는 데이터에서 RST 기반 정제를 동적으로 적용하는 온라인 알고리즘 설계.

**⑤ 다중 클래스 불균형 직접 처리**
- 현재는 이진 분류 기반이며, OvR(One-vs-Rest) 또는 OvO(One-vs-One) 방식으로 확장 시 정제 기준의 재정의 필요.

**⑥ 공정성(Fairness)과 불균형의 교차 연구**
- 인종, 성별 등 보호 속성에서의 불균형 문제와 RST 기반 정제가 공정성 지표에 미치는 영향 분석.

**⑦ 설명 가능한 AI(XAI)와의 결합**
- RST의 상·하부 근사를 통해 "왜 이 합성 샘플이 유효한가"를 설명하는 XAI 파이프라인 구성 가능.

---

## 📚 참고 자료

1. **[주 논문]** Ramentol, E., Caballero, Y., Bello, R., & Herrera, F. (2012). *SMOTE-RSB∗: a hybrid preprocessing approach based on oversampling and undersampling for high imbalanced data-sets using SMOTE and rough sets theory*. Knowledge and Information Systems, 33(2), 245–265. DOI: 10.1007/s10115-011-0465-6

2. **[SMOTE 원논문]** Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). *SMOTE: Synthetic minority over-sampling technique*. Journal of Artificial Intelligence Research, 16, 321–357.

3. **[Borderline-SMOTE]** Han, H., Wang, W. Y., & Mao, B. H. (2005). *Borderline-SMOTE: A new over-sampling method in imbalanced data sets learning*. ICIC 2005, LNCS 3644, 878–887.

4. **[Safe-Level-SMOTE]** Bunkhumpornpat, C., Sinapiromsaran, K., & Lursinsap, C. (2009). *Safe-Level-SMOTE*. PAKDD 2009, LNCS, 475–482.

5. **[Batista et al.]** Batista, G. E. A. P. A., Prati, R. C., & Monard, M. C. (2004). *A study of the behaviour of several methods for balancing machine learning training data*. SIGKDD Explorations, 6(1), 20–29.

6. **[RST 원논문]** Pawlak, Z. (1982). *Rough sets*. International Journal of Computer and Information Sciences, 11, 145–172.

7. **[통계 검정]** Demšar, J. (2006). *Statistical comparisons of classifiers over multiple data sets*. Journal of Machine Learning Research, 7, 1–30.

8. **[AUC 평가]** Huang, J., & Ling, C. X. (2005). *Using AUC and accuracy in evaluating learning algorithms*. IEEE Transactions on Knowledge and Data Engineering, 17(3), 299–310.

9. **[KEEL 도구]** Alcalá-Fdez, J., et al. (2011). *KEEL data-mining software tool*. Journal of Multiple-Valued Logic and Soft Computing, 17(2–3), 255–287.
