# An Instance Level Analysis of Data Complexity 

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

기존 데이터 복잡도 연구는 **데이터셋 전체** 수준의 분석에 집중하였으나, 이 논문은 **개별 인스턴스 수준**에서의 분석이 필요하다고 주장한다. 특히:

- **어떤 인스턴스가** 자주 오분류되는가?
- **왜** 그 인스턴스가 오분류되는가?
- 이 정보를 학습 과정에 **어떻게 통합**하여 성능을 향상시킬 수 있는가?

### 주요 기여

| 기여 | 설명 |
|------|------|
| **Instance Hardness (IH) 정의** | 인스턴스가 오분류될 확률의 경험적 정의 제시 |
| **Hardness Measures 제안** | 왜 어렵게 분류되는지 설명하는 8가지 측도 |
| **대규모 실증 분석** | 64개 데이터셋, 190,000개 이상 인스턴스, 9개 학습 알고리즘 |
| **클래스 겹침이 주원인** | 클래스 겹침(class overlap)이 instance hardness의 주요 원인임을 실증 |
| **학습 과정 통합** | Informative Error(IE) 및 필터링을 통한 정확도 향상 |

---

## 2. 상세 분석

### 2-1. 해결하고자 하는 문제

기존 머신러닝 평가 지표(정확도, 정밀도 등)는 **집계 정보**만 제공하며, 개별 인스턴스 수준에서 *왜* 오분류가 발생하는지 알 수 없다. 또한 기존 데이터 복잡도 연구(Ho & Basu, 2002)도 데이터셋 전체 수준에 머물렀다.

이 논문은 다음 질문에 답하고자 한다:

> "데이터셋의 특정 인스턴스가 오분류될 확률은 얼마이며, 그 원인은 무엇인가?"

---

### 2-2. 제안 방법 및 수식

#### (A) Instance Hardness의 이론적 정의

베이즈 정리를 이용한 $p(h|t)$ 분해로부터 instance hardness를 도출한다:

$$p(h|t) = \frac{p(t|h)\,p(h)}{p(t)} = \frac{\prod_{i=1}^{|t|} p(y_i|x_i, h)\, p(x_i|h)\, p(h)}{p(t)}$$

가설 $h$에 대한 인스턴스 $\langle x_i, y_i \rangle$의 **Instance Hardness**:

$$IH_h(\langle x_i, y_i \rangle) = 1 - p(y_i | x_i, h)$$

가설 공간 $\mathcal{H}$ 전체에 대한 기댓값:

$$IH(\langle x_i, y_i \rangle) = \sum_{\mathcal{H}} \bigl(1 - p(y_i|x_i, h)\bigr) p(h|t) = 1 - \sum_{\mathcal{H}} p(y_i|x_i, h)\, p(h|t) \tag{1}$$

#### (B) 실용적 근사 (Empirical IH)

대표 학습 알고리즘 집합 $\mathcal{L}$을 이용한 근사:

$$IH_{\mathcal{L}}(\langle x_i, y_i \rangle) = 1 - \frac{1}{|\mathcal{L}|} \sum_{j=1}^{|\mathcal{L}|} p\bigl(y_i \mid x_i, g_j(t, \alpha)\bigr) \tag{2}$$

여기서 $p(h|t) \approx \frac{1}{|\mathcal{L}|}$로 균등 근사한다.

#### (C) Hardness Measures 수식

**k-Disagreeing Neighbors (kDN)** — 클래스 겹침의 국소적 측정:

$$kDN(x) = \frac{\bigl|\{y : y \in kNN(x) \wedge t(y) \neq t(x)\}\bigr|}{k}$$

**Disjunct Size (DS)** — 결정 경계 복잡도:

$$DS(x) = \frac{|\,disjunct(x)\,| - 1}{\max_{y \in D}|\,disjunct(y)\,| - 1}$$

**Disjunct Class Percentage (DCP)** — 피처 부분집합의 겹침:

$$DCP(x) = \frac{|\{z : z \in disjunct(x) \wedge t(z) = t(x)\}|}{|\,disjunct(x)\,|}$$

**Class Likelihood (CL)** — 전역적 겹침 및 클래스 소속 가능성:

$$CL(x) = CL(x, t(x)) = \prod_{i}^{|x|} P(x_i \mid t(x))$$

**Class Likelihood Difference (CLD)** — 상대적 겹침:

$$CLD(x) = CL(x) - \underset{y \in Y - t(x)}{\arg\max}\, CL(x, y)$$

**Minority Value (MV)** — 클래스 불균형:

$$MV(x) = 1 - \frac{|\{z : z \in D \wedge t(z) = t(x)\}|}{\max_{y \in Y} |\{z : z \in D \wedge t(z) = y\}|}$$

**Class Balance (CB)**:

$$CB(x) = \frac{|\{z : z \in D \wedge t(z) = t(x)\}|}{|D|} - \frac{1}{|Y|}$$

#### (D) Class Overlap 및 Class Skew의 수식적 표현

이진 분류에서의 **클래스 겹침**:

$$classOverlap(\langle x_i, y_i \rangle) = p(\bar{y}_i \mid x_i, t) - p(y_i \mid x_i, t) \tag{3}$$

**클래스 불균형(skew)**:

$$classSkew(\langle x_i, y_i \rangle) = \frac{p(y_i \mid t)}{p(\bar{y}_i \mid t)} \tag{4}$$

**기대 instance hardness**는 두 요소의 함수:

$$\mathbb{E}\bigl[IH(\langle x_i, y_i \rangle)\bigr] \sim f\bigl(classOverlap(\langle x_i, y_i \rangle),\; classSkew(\langle x_i, y_i \rangle)\bigr)$$

---

### 2-3. 모델 구조

#### ESLA 선택 방법
- 20개의 일반적 학습 알고리즘을 대상으로 **Classifier Output Difference (COD)** 기반 비지도 메타러닝으로 다양성 측정
- 계층적 집적 클러스터링으로 덴드로그램 생성
- COD 임계값 0.18에서 절단 → 9개의 대표 알고리즘 집합 $\mathcal{L}$ 구성

**최종 ESLA 집합 $\mathcal{L}$:**

| 알고리즘 | 유형 |
|----------|------|
| RIDOR | 규칙 기반 |
| Naïve Bayes | 확률적 |
| Multilayer Perceptron (BP) | 신경망 |
| Random Forest | 앙상블 |
| LWL | 거리 기반 |
| 5-NN | 이웃 기반 |
| NNge | 예시 기반 |
| C4.5 | 트리 기반 |
| RIPPER | 규칙 기반 |

#### 실험 설계
- **5 × 10-fold 교차검증** (5회 반복 × 10-fold)
- 64개 데이터셋 (57개 UCI + 7개 non-UCI)
- 총 190,000개 이상 인스턴스, 28,750개 모델 생성

---

### 2-4. 성능 향상

#### Informative Error (IE)

MLP의 오류 함수를 수정하여 hard instance의 영향을 감소:

**기존 오류 함수:**

$$error(x) = \begin{cases} 1 - o_k & \text{if } t(x) = k_{class} \\ 0 - o_k & \text{otherwise} \end{cases}$$

**수정된 오류 함수 (IE):**

$$error(x) = \begin{cases} 1 - IH(x, t(x)) - o_k & \text{if } t(x) = k_{class} \\ 0 - o_k & \text{otherwise} \end{cases}$$

**결과 (52개 데이터셋, Wilcoxon 부호 순위 검정):**

| 방법 | 평균 정확도 | p-value |
|------|-------------|---------|
| 기존 MLP (Orig) | 81.05% | — |
| RENN | 82.45% | <0.001 |
| FaLKNR | 82.14% | <0.001 |
| AdaBoost | 81.37% | <0.001 |
| MultiBoost | 81.60% | <0.001 |
| $\text{IE}_{ESLA}$ | 84.03% | <0.001 |
| $\text{IE}_{MLP}$ | **84.57%** | 0.003 |

#### 인스턴스 필터링

IH 임계값 기반 필터링 (IH_0.5, IH_0.7, IH_0.9) 및 적응형 필터링(A_0.5, A_0.7, A_0.9):

| 방법 | 평균 정확도 |
|------|-------------|
| 필터링 없음 | 78.02% |
| IH_0.7 | 79.74% |
| A_0.9 (적응형) | **85.34%** |

- 9개 알고리즘 모두에서 통계적으로 유의한 향상 (대부분 p < 0.001)

#### 주요 발견
- 인스턴스의 **17.5%**가 절반 이상의 알고리즘에서 오분류됨
- **2.3%**는 모든 알고리즘에서 오분류됨
- 전체 인스턴스의 **38.3%**만이 항상 올바르게 분류됨

---

### 2-5. 한계

1. **계산 비용**: $N$개 학습 알고리즘으로 IH 계산 필요 → 대규모 데이터셋에서 비효율
2. **ESLA 집합의 진화**: 새로운 알고리즘(예: 딥러닝) 등장 시 $\mathcal{L}$ 업데이트 필요
3. **$f$의 정확한 형태 미지**: 클래스 겹침과 클래스 불균형이 IH에 미치는 함수 관계 불명확
4. **CL/CLD 중복**: CL과 CLD의 Spearman 상관계수가 0.989로 매우 높아 중복 측정
5. **이진 분류 중심**: 다중 클래스 문제에서 수식 (3), (4)는 1-vs-1 또는 1-vs-all 확장 필요
6. **실험 알고리즘 범위**: 2014년 기준 알고리즘 → 딥러닝, 그래디언트 부스팅 등 미포함

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. Instance Hardness 기반 일반화 향상 메커니즘

논문의 핵심 통찰은 **hard instance가 일반화를 방해**한다는 점이다. 특히 클래스 겹침 영역의 인스턴스는 학습 알고리즘이 과적합하거나 잘못된 결정 경계를 학습하게 만든다.

$$\text{일반화 오류} \approx \text{bias}^2 + \text{variance} + \underbrace{\text{noise}}_{\text{hard instances}}$$

### 3-2. 구체적 향상 방법

#### 방법 1: Informative Error (IE)
- Hard instance의 오류 신호를 약화시킴
- $IH(x, t(x)) \to 1$이면 target이 $1 - 1 = 0$에 가까워짐 → 해당 인스턴스 무시
- **효과**: 결정 경계가 비겹침 영역에 집중하여 일반화 향상
- **수식적 해석**: $IH=0$이면 기존 학습과 동일, $IH=1$이면 학습 신호 제거

#### 방법 2: 인스턴스 필터링
- IH > threshold인 인스턴스를 학습 전 제거
- **Adaptive Filtering**: 데이터셋/알고리즘 조합별로 최적 필터 집합 탐색 (그리디 알고리즘)

```
알고리즘 1 (Adaptive Filter):
F ← {}
currAcc ← runLA({})
while L ≠ {} do
    bestAcc ← currAcc
    for each g ∈ L do
        acc ← runLA(F + g)
        if acc > bestAcc then update bestLA
    if bestAcc > currAcc then F ← F + bestLA
    else break
```

#### 방법 3: 인스턴스 가중치
- Hard instance에 낮은 가중치 부여 → KNN, Naïve Bayes 등에 적용 가능
- 기존 부스팅(AdaBoost)이 오분류 인스턴스에 **높은** 가중치를 부여하는 것과 반대 방향

### 3-3. 일반화 향상의 조건 및 분석

Spearman 상관분석 결과:

| Hardness 측도 | IH_ind와의 상관 | IH_class와의 상관 |
|---------------|-----------------|-------------------|
| kDN | **0.830** | **0.875** |
| CL | -0.670 | -0.782 |
| CLD | -0.660 | -0.767 |
| MV | 0.522 | 0.542 |

- **kDN, CL, CLD** (클래스 겹침 측정)이 IH와 가장 강한 상관관계
- 클래스 불균형(MV, CB)은 독립적으로 오분류를 유발하지 않고, **클래스 겹침의 효과를 증폭**함

$$\text{소수 클래스 평균 IH class} = 0.41 \gg \text{다수 클래스 평균 IH class} = 0.16$$

### 3-4. 실용적 함의

1. **데이터 품질 진단**: IH가 높은 인스턴스는 레이블 오류, 노이즈, 또는 내재적 클래스 겹침의 지표
2. **조기 종료 기준**: 학습 과정에서 hard instance의 IH 분포 모니터링으로 최적 종료 시점 결정
3. **메타러닝**: 데이터셋 수준의 평균 IH가 알고리즘 선택 기준으로 활용 가능

---

## 4. 앞으로의 연구에 미치는 영향 및 고려 사항

### 4-1. 앞으로의 연구에 미치는 영향

#### (A) 데이터 중심 AI (Data-Centric AI) 패러다임과의 연계
이 논문은 알고리즘 개선이 아닌 **데이터 이해**를 통한 성능 향상이라는 관점을 선구적으로 제시하였다. 이는 Andrew Ng이 주창한 Data-Centric AI 방향성과 일치하며, 이후 데이터 품질 연구의 이론적 토대가 되었다.

#### (B) 커리큘럼 러닝 (Curriculum Learning)과의 연결
Bengio et al. (2009)의 커리큘럼 러닝이 쉬운 인스턴스에서 어려운 인스턴스로 학습 순서를 조정하는 것과 직접적으로 연결된다. IH는 인스턴스의 "난이도"를 정량화하는 객관적 기준을 제공한다.

#### (C) 불균형 학습 연구에의 기여
클래스 불균형 + 클래스 겹침이 결합될 때 hardness가 증폭된다는 발견은 SMOTE 등 오버샘플링 기법의 개선 연구에 기여하였다.

#### (D) 메타러닝 및 AutoML
데이터셋 수준의 hardness 측도가 알고리즘 선택 및 하이퍼파라미터 최적화에 사용될 수 있음을 보여주어, AutoML 연구의 기반을 제공한다.

---

### 4-2. 앞으로 연구 시 고려할 점

#### 기술적 고려사항

1. **딥러닝 시대의 IH 재정의**
   - ESLA 집합 $\mathcal{L}$에 Transformer, CNN, GNN 등 포함 필요
   - 딥러닝 모델의 경우 classifier score 추출 방법이 다름

2. **계산 효율성**
   - 대규모 데이터셋(수억 건)에서 9개 알고리즘의 교차검증은 비현실적
   - kDN처럼 계산 비용이 낮은 단일 측도로 IH 추정하는 방법 탐구 필요

3. **시계열/비정형 데이터로의 확장**
   - 현재 방법은 정형 분류 데이터에 집중
   - 텍스트, 이미지, 그래프 데이터에서의 instance hardness 정의 필요

4. **함수 $f$의 명시적 모델링**
   $$\mathbb{E}[IH] \sim f(classOverlap, classSkew)$$
   - 이 함수의 정확한 형태를 규명하는 것이 중요한 미해결 과제

5. **IH의 동적 추적**
   - 온라인 러닝 환경에서 데이터 분포 변화(concept drift)에 따른 IH 변화 추적

#### 방법론적 고려사항

6. **레이블 노이즈와 내재적 모호성 구분**
   - IH가 높은 인스턴스 중 수정 가능한 레이블 오류 vs. 내재적 클래스 겹침 구분 방법 필요

7. **적응형 필터링의 최적화**
   - 현재 그리디 탐색은 $O(N^2)$ → 더 효율적인 탐색 방법 필요

8. **IH와 공정성(Fairness)**
   - 소수 집단에 속하는 인스턴스의 IH가 높을 경우, 이를 제거하면 편향이 증가할 수 있음

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5-1. 연구 흐름 분류

#### (A) IH 프레임워크의 직접 확장

**Moreno-Torres et al. (2012, 2022 후속)**: Dataset complexity 관련 survey들이 IH 측도를 표준적 기준으로 채택하고 있으며, `pymfe` 라이브러리(PyMFE team, 2020)에 instance-level complexity 측도가 구현됨.

| 연구 방향 | 논문 예시 | Smith et al.과의 차이 |
|-----------|-----------|----------------------|
| 딥러닝 커리큘럼 러닝 | Hacohen & Weinshall, 2019; 이후 후속 연구 | Self-paced learning에서 difficulty를 loss 기반으로 측정 |
| 데이터 맵(Cartography) | Swayamdipta et al., 2020 | 훈련 동역학(confidence, variability) 기반 difficulty 정의 |
| 클래스 불균형+겹침 | Rivera et al., 2022 | IH를 SMOTE 전처리에 직접 적용 |
| 메타러닝 | Rivolli et al., 2022 | 확장된 dataset complexity 측도 |

#### (B) Dataset Cartography (Swayamdipta et al., 2020) — 가장 직접적 비교

**논문**: "Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics" (EMNLP 2020)

**핵심 방법**: 훈련 과정에서의 동적 신호(training dynamics)를 이용하여 인스턴스 난이도 분류

$$\hat{\mu}_i = \frac{1}{E} \sum_{e=1}^{E} p_{\theta^e}(y_i^* | x_i) \quad \text{(평균 신뢰도)}$$

$$\hat{\sigma}_i = \sqrt{\frac{\sum_{e=1}^{E}(p_{\theta^e}(y_i^*|x_i) - \hat{\mu}_i)^2}{E}} \quad \text{(변동성)}$$

인스턴스를 3가지로 분류:
- **Easy-to-learn**: 높은 신뢰도, 낮은 변동성
- **Hard-to-learn**: 낮은 신뢰도, 낮은 변동성
- **Ambiguous**: 중간 신뢰도, 높은 변동성

**Smith et al. vs. Swayamdipta et al. 비교:**

| 항목 | Smith et al. (2014) | Swayamdipta et al. (2020) |
|------|---------------------|--------------------------|
| 난이도 정의 | 여러 알고리즘의 오분류 확률 | 훈련 동역학 (confidence + variability) |
| 알고리즘 의존성 | 9개 ESLA 앙상블 | 단일 모델의 에폭별 변화 |
| 적용 도메인 | 정형 데이터 (UCI) | NLP (텍스트 분류) |
| 계산 비용 | 높음 (N개 알고리즘 × CV) | 중간 (단일 모델 훈련 과정) |
| 인스턴스 구분 | Easy/Hard (연속값) | Easy/Ambiguous/Hard (3분류) |
| 일반화 향상 방법 | IE, 필터링 | Ambiguous 인스턴스 우선 학습 |

**공통점**: 클래스 겹침(ambiguity) 영역 인스턴스가 학습에 핵심적 역할을 한다는 결론은 동일하나, Swayamdipta et al.은 **ambiguous 인스턴스를 제거하지 않고 활용**하는 방향을 제안한다는 점에서 차이가 있다.

#### (C) 클래스 불균형 + 겹침 통합 연구

최근 연구들은 클래스 불균형과 클래스 겹침을 분리하여 처리해야 함을 강조한다. Smith et al.의 발견(클래스 불균형 단독으로는 hardness 유발 안 함, 겹침 증폭)과 일치한다.

**PyMFE (Alcalá-Fdez et al. 관련 후속, 2020)**: Smith et al.의 instance-level 측도(kDN 등)를 포함한 확장 라이브러리 구현

#### (D) Data-Centric AI와의 연계 (2021~)

Andrew Ng의 Data-Centric AI 운동(2021~)은 Smith et al.의 철학과 직접 연결된다:
- 데이터의 **어떤 부분**이 문제인지 진단
- 알고리즘 개선보다 데이터 품질 개선 우선
- IH는 이를 위한 정량적 도구

#### (E) 대규모 언어모델(LLM)에서의 관련성

최근 **instruction tuning** 및 **RLHF** 연구에서도 유사한 문제가 등장:
- 어떤 훈련 예시가 모델에 유익한가?
- 노이즈 레이블이 있는 데이터의 처리

IH 프레임워크는 LLM의 학습 데이터 선택 및 커리큘럼 설계에 응용 가능하나, 계산 비용 문제가 주요 장벽이다.

---

## 참고 자료

**주요 논문 (제공된 PDF):**
- Smith, M. R., Martinez, T., & Giraud-Carrier, C. (2014). **An instance level analysis of data complexity**. *Machine Learning*, 95, 225–256. DOI: 10.1007/s10994-013-5422-z

**논문 내 참조 문헌:**
- Ho, T. K., & Basu, M. (2002). Complexity measures of supervised classification problems. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 24, 289–300.
- Batista, G. E. A. P. A., Prati, R. C., & Monard, M. C. (2004). A study of the behavior of several methods for balancing machine learning training data. *SIGKDD Explorations Newsletter*, 6(1), 20–29.
- Breunig, M. M., et al. (2000). LOF: identifying density-based local outliers. *SIGMOD Record*, 29(2), 93–104.
- Tomek, I. (1976). An experiment with the edited nearest-neighbor rule. *IEEE Transactions on Systems, Man and Cybernetics*, 6, 448–452.
- Brodley, C. E., & Friedl, M. A. (1999). Identifying mislabeled training data. *Journal of Artificial Intelligence Research*, 11, 131–167.
- Wolpert, D. H. (1996). The lack of a priori distinctions between learning algorithms. *Neural Computation*, 8(7), 1341–1390.

**비교 분석에 활용한 2020년 이후 연구:**
- Swayamdipta, S., et al. (2020). Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics. *EMNLP 2020*.
- Rivolli, A., et al. (2022). Meta-features for meta-learning. *Knowledge-Based Systems*, 240, 108101.
- Lorena, A. C., et al. (2019). How Complex Is Your Classification Problem? A Survey on Measuring Classification Complexity. *ACM Computing Surveys*, 52(5).

> **주의**: 2020년 이후 관련 연구 비교 분석 부분에서 일부 인용 정보는 제공된 PDF의 직접적 내용이 아닌, 해당 분야의 대표적 연구 흐름에 기반한 것임을 명시합니다. 특히 특정 논문의 수치나 세부 내용은 원문 확인을 권장합니다.
