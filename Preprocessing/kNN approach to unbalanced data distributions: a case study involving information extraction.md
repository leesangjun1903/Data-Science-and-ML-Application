# kNN Approach to Unbalanced Data Distributions: A Case Study involving Information Extraction

**[Jianping Zhang & Inderjeet Mani, ICML Workshop on Learning from Imbalanced Datasets II, 2003]**

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장
이 논문은 **불균형 클래스 분포(unbalanced class distribution)** 를 가진 정보 추출(Information Extraction) 문제에서 **k-최근접 이웃(kNN) 알고리즘**과 **언더샘플링(under-sampling)** 전략을 결합하는 방식을 실증적으로 분석합니다.

구체적으로, 생물의학 문헌(MEDLINE 초록)에서 **단백질 이름을 추출**하는 과제를 사례 연구로 삼아, 전체 데이터의 약 4%만이 긍정 예제(단백질 명칭)인 극단적 불균형 상황에서 kNN 방법론의 효과를 검증합니다.

### 주요 기여

| 기여 항목 | 내용 |
|-----------|------|
| **실증적 비교 분석** | 5가지 음성 예제 선택 방법(Random, NearMiss-1/2/3, Distant) 비교 |
| **kNN vs. C5.0 비교** | 불균형 데이터에서 kNN이 의사결정트리(C5.0)보다 우수함 입증 |
| **언더샘플링 민감도 분석** | 음성 예제 비율에 따른 정밀도-재현율 트레이드오프 정량화 |
| **Random 선택의 효과 입증** | 단순 무작위 선택이 정교한 NearMiss 방법과 동등하거나 우수 |
| **최초 연구** | kNN과 언더샘플링 전략을 결합한 불균형 분류 최초 연구로 명시 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**클래스 불균형 문제(Class Imbalance Problem):**
- MEDLINE 초록 데이터에서 긍정 클래스(단백질 이름) = 약 **4%**
- 나머지 약 **96%**는 부정 클래스(비단백질 토큰)
- 표준 분류 알고리즘은 전체 정확도(overall accuracy)를 최대화하도록 설계되어 있어, 소수 클래스를 완전히 무시하는 **trivial classifier** 문제 발생
- 결정트리 등은 소수 클래스 영역에서 **과적합된 소규모 규칙(small disjuncts)** 생성

**과제 정의:**
- 각 단어 위치를 `{protein-start, protein-end, protein-middle, none}` 중 하나로 분류
- 실질적으로 3개의 독립 분류 과제: `Protein tag`, `Start of protein tag`, `End of protein tag`

### 2.2 데이터셋

- **300개** MEDLINE 초록 (Protein Information Resource에서 무작위 선택)
- 약 **3,300개** 단백질 이름 어노테이션
- 학습/테스트 분할:

| | Start | Protein | End |
|--|--|--|--|
| 긍정 학습 예제 | 2,434 | 2,555 | 2,434 |
| 부정 학습 예제 | 56,255 | 58,593 | 56,255 |
| 긍정 테스트 예제 | 505 | 509 | 505 |
| 부정 테스트 예제 | 11,812 | 11,808 | 11,812 |

### 2.3 특징 벡터 구성

각 단어 위치는 다음 15개 변수로 이루어진 특징 벡터를 구성합니다:

1. **Word token** (단어 자체)
2. **Part-of-speech tag** (품사 태그, Alembic tagger)
3. **Subdictionary tag**: macromolecule terms (1047), biomedical terms (852), chemical terms (806), common English terms, non-word tokens (2212)
4. **Protein tag** (단백질 클래스)
5. **Start of protein tag**
6. **End of protein tag**
7. **Context window features** (좌우 3단어의 품사/서브사전 태그)

### 2.4 제안하는 방법 및 수식

#### (A) kNN 유사도 함수

두 단어 위치 벡터 $x$, $y$에 대해 다음 유사도 메트릭을 사용합니다:

$$\text{Similarity}(x, y) = \frac{2 \times LCS(x_0, y_0)}{\max(|x_0|, |y_0|)} + \sum_{i=1}^{14} match(x_i, y_i)$$

- $x_0$, $y_0$: Word token 특징값
- $LCS(x_0, y_0)$: $x_0$와 $y_0$의 **최장 공통 부분 문자열(Longest Common Subsequence)** 길이
- $|x_0|$, $|y_0|$: 각 단어의 길이
- $match(x_i, y_i)$: $x_i = y_i$이면 1, 아니면 0 (나머지 14개 이진 특징)
- Word token 특징은 다른 특징 대비 **2배 가중치** 부여

> **해석:** Word token은 부분 매칭(LCS 기반)으로 유사도를 계산하고, 나머지 14개 특징은 이진 매칭으로 계산하여 합산. 즉 최대 가능 유사도 = $2 + 14 = 16$.

#### (B) F-measure (평가 지표)

$$F\text{-}measure(r) = \frac{2 \times recall \times precision}{recall + precision}$$

- $recall = \frac{\text{올바르게 인식된 긍정 예제 수}}{\text{전체 긍정 예제 수}}$
- $precision = \frac{\text{올바르게 인식된 긍정 예제 수}}{\text{긍정으로 예측된 전체 예제 수}}$

#### (C) 언더샘플링 전략 (5가지 음성 예제 선택 방법)

논문은 다수 클래스(부정 예제)를 일정 비율(%）로 선택하되, 다음 5가지 방식을 비교합니다:

| 방법 | 설명 | 기댓값 |
|------|------|--------|
| **Random** | 무작위 선택 | 전체 부정 분포 유지 |
| **NearMiss-1** | 가장 가까운 긍정 예제 3개까지의 **평균 거리가 최소**인 부정 예제 선택 | 고정밀도·저재현율 예상 |
| **NearMiss-2** | **모든** 긍정 예제까지의 평균 거리가 최소인 부정 예제 선택 (가장 먼 긍정 3개 기준) | 균형 예상 |
| **NearMiss-3** | 각 긍정 예제마다 가장 가까운 부정 예제 $k$개 선택 (긍정 예제 모두 포위 보장) | 고정밀도·저재현율 |
| **Distant** | 가장 가까운 긍정 예제 3개까지의 **평균 거리가 최대**인 부정 예제 선택 | 저정밀도·고재현율 |

#### (D) k값 결정

- 1NN, 3NN, 5NN, 7NN, 9NN 비교 실험
- **5NN**이 최적 성능 → 이후 실험 모두 5NN으로 진행

### 2.5 모델 구조 (파이프라인)

```
[원시 텍스트]
      ↓
[Alembic Workbench 품사 태깅]
      ↓
[특징 벡터 생성 (15차원)]
      ↓
[언더샘플링: 부정 예제 선택 (5가지 방법 중 하나)]
      ↓
[5NN 분류기 학습 (LCS 기반 유사도)]
      ↓
[단백질 이름 경계 판별: protein-start / protein-end / protein]
      ↓
[세 분류기 결합 → 최종 단백질 이름 추출]
```

### 2.6 성능 향상

#### 주요 실험 결과 (F-measure 기준):

**Table 2 (Random 선택, 5NN vs. C5.0 비교):**

| 부정 예제 비율 | 5NN F-measure | C5.0 F-measure | 비고 |
|----------------|---------------|----------------|------|
| 5% | 0.44 | 0.27 | 5NN 우위 |
| 10% | **0.50** | 0.29 | **5NN 최고점** |
| 15% | 0.50 | 0.33 | |
| 20% | 0.49 | 0.37 | |
| 40% | 0.35 | 0.22 | |

**Table 3 (5가지 방법 비교, 10% 부정 예제):**

| 방법 | Precision | Recall | F-measure |
|------|-----------|--------|-----------|
| **Random** | 0.44 (0.030) | 0.64 (0.019) | **0.52 (0.022)** |
| NearMiss-1 | 0.28 (0.026) | 0.52 (0.009) | 0.36 (0.022) |
| **NearMiss-2** | 0.43 (0.030) | 0.64 (0.012) | **0.51 (0.022)** |
| NearMiss-3 | 0.68 (0.039) | 0.35 (0.015) | 0.46 (0.020) |
| Distant | 0.08 (0.006) | 0.97 (0.003) | 0.14 (0.010) |

**핵심 발견:**
- 최적 성능은 **10% 무작위 선택** 시 달성 (정밀도-재현율 균형)
- **5NN > C5.0**: 모든 경우에서 kNN이 의사결정트리보다 F-measure 우위
- **Random ≈ NearMiss-2 >> NearMiss-1, NearMiss-3, Distant**

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화 성능 관련 핵심 논의

#### (A) 언더샘플링 비율과 일반화

논문에서 가장 중요한 일반화 관련 발견은 **"10% 음성 예제 비율"에서 최적 F-measure 달성**이라는 사실입니다.

$$\text{최적 비율}: \underset{r}{\arg\max} \; F\text{-}measure(r) \approx 10\%$$

- 음성 예제 비율이 너무 낮으면 → 정밀도는 높지만 재현율 급감 → 과소 일반화
- 음성 예제 비율이 너무 높으면 → 재현율은 높지만 정밀도 급감 → 다수 클래스 과대 표현

이는 훈련 데이터의 클래스 분포가 **모델의 결정 경계**에 직접 영향을 미침을 의미합니다.

#### (B) C5.0 과일반화 문제와 kNN의 강건성

논문은 C5.0(의사결정트리)이 소수 클래스를 **과일반화(overgeneralize)** 하는 경향을 명시합니다:

> *"C5.0 achieved the low precisions for the small amount of negative examples. This is probably due to the overgeneralization of the positive class."*

반면 kNN은:
- 국소적(local) 결정을 내리므로 **소규모 분리 가능 영역(small disjuncts)에 강건**
- Zhang(1990)의 연구를 인용하여 kNN이 small disjuncts 처리에 더 적합함을 이론적으로 뒷받침

이는 다음 수식으로 이해할 수 있습니다. kNN의 분류 결정은:

$$\hat{y}(x) = \underset{c}{\arg\max} \sum_{x_i \in \mathcal{N}_k(x)} \mathbf{1}[y_i = c]$$

여기서 $\mathcal{N}_k(x)$는 $x$의 k개 최근접 이웃 집합. 이는 **전역 결정 경계 없이 국소 다수결**로 작동하므로 불균형 데이터에서 소수 클래스 패턴을 더 잘 보존합니다.

#### (C) NearMiss 방법들의 일반화 실패 분석

**NearMiss-1 실패 원인:**

$$\text{NearMiss-1}: \arg\min_{x^- \in \mathcal{N}^-} \frac{1}{3} \sum_{j=1}^{3} d(x^-, x^+_{\text{closest}_j})$$

- 일부 긍정 예제 주변에 부정 예제가 집중 → 비균일 커버리지 → 과적합

**NearMiss-3 특성:**

$$\text{NearMiss-3}: \forall x^+ \in \mathcal{P}, \; \text{select } k \text{ closest } x^- \text{ to } x^+$$

- 높은 정밀도(최대 80%)이지만 낮은 재현율 → 과도하게 보수적인 경계

**Distant 방법의 과일반화:**

$$\text{Distant}: \arg\max_{x^- \in \mathcal{N}^-} \frac{1}{3} \sum_{j=1}^{3} d(x^-, x^+_{\text{closest}_j})$$

- 극단적 고재현율(97%) + 극단적 저정밀도(6~13%) → 긍정 클래스 경계 극도로 확장

#### (D) 5-fold Cross Validation을 통한 일반화 신뢰도

- 5회 반복 실험 (4/5 학습, 1/5 테스트) → 평균 및 표준편차 보고
- 표준편차가 상대적으로 작음 → 결과의 **안정성 및 일반화 가능성** 시사

#### (E) 일반화 성능 향상을 위한 제언 (논문 내 언급)

논문은 미래 연구로 **오버샘플링 및 비용 민감 방법(cost-sensitive methods)** 과의 결합을 제안합니다:

> *"Study of over-sampling and cost-sensitive methods will be our future research."*

비용 민감 학습에서 misclassification cost $C$를 조정하면:

$$\text{Expected Cost} = C_{FN} \cdot P(\text{FN}) + C_{FP} \cdot P(\text{FP})$$

여기서 $C_{FN} \gg C_{FP}$로 설정 시 소수 클래스 재현율을 높일 수 있어 일반화 성능과 연결됩니다.

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 향후 연구에 미치는 영향

#### (A) 언더샘플링 전략 연구의 기초

이 논문은 NearMiss 계열 방법을 **체계적으로 비교**한 초기 연구로, 이후 다음 연구들의 기반이 됩니다:

- **SMOTE (Chawla et al., 2002)**: 오버샘플링과 언더샘플링의 혼합
- **EasyEnsemble/BalanceCascade**: 앙상블 기반 불균형 처리
- **Tomek Links, CNN Rule**: 경계 정제(borderline cleaning) 방법론

#### (B) 정보 추출에서의 불균형 문제 인식 확산

생물의학 NLP에서 불균형 데이터 처리의 중요성을 실증적으로 보여줌으로써, 이후 **NER(Named Entity Recognition), Relation Extraction** 등에서 불균형 처리가 표준적 전처리 단계로 자리잡는 데 기여합니다.

#### (C) kNN의 불균형 데이터 강건성 이론적 근거 제공

kNN이 small disjuncts에 강건하다는 실증적 증거를 제공하여, 이후 **인스턴스 기반 학습(instance-based learning)** 의 불균형 데이터 응용 연구를 촉진합니다.

### 4.2 향후 연구 시 고려할 점

#### (A) 단일 도메인 한계 극복

- 이 논문은 **생물의학 단백질 추출**이라는 단일 도메인에 국한
- 향후: 다양한 도메인(금융, 법률, 의료 기록 등)에서 불균형 비율을 체계적으로 변화시킨 실험 필요

#### (B) 오버샘플링과의 결합

논문 자체에서 제안한 방향으로, SMOTE를 예로 들면:

$$\tilde{x}^+ = x^+_i + \lambda \cdot (x^+_{nn} - x^+_i), \quad \lambda \sim U(0,1)$$

합성 소수 클래스 샘플 생성 + kNN 결합 효과 검증 필요

#### (C) 비용 민감 kNN

$$\hat{y}(x) = \underset{c}{\arg\min} \sum_{x_i \in \mathcal{N}_k(x)} C(y_i, c) \cdot \frac{1}{d(x, x_i)}$$

거리 기반 가중치와 misclassification cost를 동시에 고려하는 kNN 확장

#### (D) 고차원성 문제 (Curse of Dimensionality)

$$d(x, y) \xrightarrow{p \to \infty} \text{const}$$

특징 차원이 증가할수록 kNN 유사도 구별력 저하 → **차원 축소(PCA, t-SNE)** 또는 **특징 선택** 전처리 필요

#### (E) 현대적 평가 지표 도입

- AUC-ROC, AUC-PR(Precision-Recall Curve Under Area), MCC(Matthews Correlation Coefficient) 등 불균형 데이터에 더 적합한 지표 활용

$$MCC = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의:** 아래는 제가 알고 있는 연구 흐름 기반 분석이며, 특정 논문의 정확한 수치는 직접 확인을 권장합니다.

### 5.1 비교 개요

| 비교 항목 | Zhang & Mani (2003) | 2020년 이후 연구 동향 |
|-----------|--------------------|-----------------|
| **기본 분류기** | kNN (instance-based) | Transformer (BERT, RoBERTa 등) |
| **불균형 처리** | 언더샘플링 | 오버샘플링+앙상블+손실함수 조정 |
| **특징 추출** | 수작업 (LCS, POS 등) | End-to-end 자동 학습 |
| **평가 지표** | F-measure, Precision, Recall | F1, AUC-PR, MCC |
| **데이터 스케일** | ~70K 예제 | 수백만 단위 가능 |

### 5.2 주요 2020년 이후 연구 방향

#### (A) Focal Loss 기반 불균형 처리

Lin et al. (2017)에서 제안되었으나 2020년 이후 NLP에 광범위 적용:

$$FL(p_t) = -\alpha_t (1-p_t)^\gamma \log(p_t)$$

- $\gamma > 0$: 쉬운 예제의 가중치 감소 → 어려운 소수 클래스에 집중
- kNN의 단순 다수결 투표 대비 **학습 중 동적 가중치 조정** 가능

#### (B) SMOTE 변형 연구

- **ADASYN (He et al., 2008)** → 2020년대에도 널리 사용
- **Borderline-SMOTE**: 결정 경계 근처 소수 클래스만 오버샘플링
- **SVM-SMOTE**: SVM 결정 경계를 활용한 합성 샘플 생성

이 논문의 NearMiss 계열과 개념적으로 연결됩니다(경계 근처 예제에 주목).

#### (C) Transformer + 불균형 처리의 결합

**예:** BioNER(생물의학 NER)에서:

- **BioBERT (Lee et al., 2020)** + Class-weighted CrossEntropy
- **PubMedBERT (Gu et al., 2021)** + Label Smoothing

이러한 연구들은 Zhang & Mani(2003)의 문제 설정(MEDLINE 단백질 추출)과 동일한 도메인을 다루면서도, 언어모델의 사전학습으로 불균형 문제의 상당 부분을 완화합니다.

#### (D) 메타학습 기반 불균형 처리 (2020년 이후)

$$\theta^* = \theta - \alpha \nabla_\theta \mathcal{L}_{meta}(\theta)$$

Few-shot learning 관점에서 소수 클래스를 학습하는 방향으로 발전, 이 논문의 "소수 클래스 인식 기반 학습" 아이디어를 일반화합니다.

### 5.3 Zhang & Mani(2003)의 현재적 의의

| 측면 | 평가 |
|------|------|
| **언더샘플링 전략 체계화** | NearMiss 방법론 비교 → 현재도 표준 벤치마크 방법 |
| **Random 선택의 강건성** | 2020년대 연구에서도 Random undersampling이 경쟁력 있음 확인 |
| **kNN의 한계** | 대규모 데이터에서 계산 복잡도 $O(n \cdot d)$ → 딥러닝 대체 |
| **단백질 NER** | BioBERT 등으로 성능 대폭 향상되었으나, 불균형 문제 자체는 여전히 관련 |

---

## 참고 자료

**원본 논문:**
- Zhang, J., & Mani, I. (2003). *kNN Approach to Unbalanced Data Distributions: A Case Study involving Information Extraction*. Workshop on Learning from Imbalanced Datasets II, ICML, Washington DC.

**논문 내 인용 문헌 (분석에 활용):**
- Kubat, M., & Matwin, S. (1997). Addressing the curse of imbalanced data sets. *Proceedings of ICML*. Morgan Kaufmann.
- Japkowicz, N. (2000). Learning from imbalanced data sets. *AAAI Workshop Technical Report* WS-00-05.
- Japkowicz, N., & Stephen, S. (2002). The class imbalance problem: A systematic study. *Intelligent Data Analysis*, 6(5).
- Ling, C.X., & Li, C. (1998). Data mining for direct marketing. *Proceedings of ACM SIGKDD*.
- Holte, R.C., Acker, L., & Porter, B. (1989). Concept learning and the problem of small disjuncts. *IJCAI-89*.
- Zhang, J. (1990). A method that combines inductive learning with exemplar-based learning. *IEEE ICTAI*.
- Quinlan, J.R. (1993). *C4.5: Programs for Machine Learning*. Morgan Kaufmann.
- Weiss, G.M., & Hirsh, H. (2000). Learning to predict extremely rare events. *AAAI Workshop*.

**2020년 이후 관련 연구 (일반적 지식 기반, 직접 확인 권장):**
- Lee, J. et al. (2020). BioBERT: A pre-trained biomedical language representation model. *Bioinformatics*, 36(4).
- Gu, Y. et al. (2021). Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing. *ACM CHIL*.
- Lin, T.Y. et al. (2017). Focal Loss for Dense Object Detection. *ICCV*. (2020년대 NLP 적용 기반)
- Chawla, N.V. et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *JAIR*, 16. (2020년대 여전히 표준 방법)
