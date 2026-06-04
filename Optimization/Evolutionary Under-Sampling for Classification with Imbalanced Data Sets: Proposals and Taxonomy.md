# Evolutionary Under-Sampling for Classification with Imbalanced Datasets: Proposals and Taxonomy

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

García & Herrera (2008)의 본 논문은 **불균형 데이터셋(imbalanced datasets)에서의 분류 문제**를 해결하기 위해 진화 알고리즘(Evolutionary Algorithms, EAs) 기반의 언더샘플링(under-sampling) 방법론인 **Evolutionary Under-Sampling (EUS)** 을 제안한다. 핵심 주장은 다음과 같다:

> *진화 알고리즘을 활용한 언더샘플링은 불균형 비율(Imbalance Ratio, IR)이 높아질수록 비진화적 언더샘플링 방법보다 우수한 성능을 보인다.*

### 주요 기여

| 기여 항목 | 설명 |
|-----------|------|
| **EUS 방법론 제안** | 8가지 EUS 모델 설계 |
| **분류 체계(Taxonomy) 제시** | 목적, 선택 방식, 평가 지표에 따른 분류 |
| **비매개변수 통계 검증** | Friedman, Iman-Davenport, Holm 절차 적용 |
| **종합 비교 실험** | UCI 28개 데이터셋 대상 기존 방법론과 비교 |
| **프로토타입 선택의 한계 규명** | 불균형 데이터에 PS 방법 부적절함을 실증 |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

**클래스 불균형 문제(Class Imbalance Problem)**는 한 클래스(다수 클래스, majority class)의 인스턴스 수가 다른 클래스(소수 클래스, minority class)보다 현저히 많을 때 발생한다.

불균형 비율(Imbalance Ratio)은 다음과 같이 정의된다:

$$IR = \frac{N^-}{N^+} \tag{1}$$

여기서 $N^-$는 다수 클래스 인스턴스 수, $N^+$는 소수 클래스 인스턴스 수이다.

- IR > 1: 불균형 데이터셋
- IR > 9: **고도 불균형(high imbalance)** 으로 정의 (소수 클래스 무시 시 오류율 ≤ 10%)

표준 분류기는 다수 클래스 편향으로 인해 소수 클래스(예: 사기 탐지, 의료 진단, 이상 탐지)를 무시하는 경향이 있다.

### 2.2 평가 지표

기존 정확도(Accuracy)는 불균형 데이터에서 의미가 없으므로 다음 두 지표를 사용한다:

**혼동 행렬 기반 지표:**

$$TPrate = \frac{TP}{TP+FN}, \quad TNrate = \frac{TN}{FP+TN}$$

**기하평균(Geometric Mean, GM):**

$$g = \sqrt{a^+ \cdot a^-} = \sqrt{TPrate \cdot TNrate} \tag{GM}$$

**AUC (Area Under the ROC Curve):**
ROC 곡선 아래 면적으로 불균형 성능을 단일 수치로 표현.

### 2.3 제안하는 방법: EUS의 기반 — 진화적 프로토타입 선택(EPS)

**이진 표현(Binary Representation):**

훈련 셋 $TR$의 $N$개 인스턴스에 대해, 염색체는 $N$개의 유전자(gene)로 구성된다:
- 유전자 값 = 1: 해당 인스턴스가 선택된 부분집합 $S$에 포함
- 유전자 값 = 0: 제외

**기존 EPS의 피트니스 함수:**

$$Fitness(S) = \alpha \cdot clas\_rat + (1-\alpha) \cdot perc\_red \tag{2}$$

여기서 퍼센트 감소율은:

$$perc\_red = 100 \cdot \frac{|TR| - |S|}{|TR|} \tag{3}$$

이 함수는 불균형 데이터의 특수성을 고려하지 않으므로 EUS는 별도의 피트니스 함수를 설계한다.

### 2.4 EUS 분류 체계(Taxonomy)

EUS는 두 가지 기준으로 분류된다:

**① 목적(Objective)에 따른 분류:**

| 분류 | 설명 |
|------|------|
| **EBUS** (Evolutionary Balancing Under-Sampling) | 클래스 균형을 최우선 목표로 설정 |
| **EUSCM** (EUS guided by Classification Measures) | 분류 성능을 주 목표, 균형은 암묵적 부목표 |

**② 선택 방식(Selection Scheme)에 따른 분류:**

| 분류 | 설명 |
|------|------|
| **GS** (Global Selection) | 소수·다수 클래스 모두 제거 가능 |
| **MS** (Majority Selection) | 다수 클래스 인스턴스만 제거, 소수 클래스는 보존 |

**③ 평가 지표:** GM 또는 AUC

→ 총 **8가지 EUS 모델** 생성 (2 × 2 × 2):

```
EUS
├── EBUS
│   ├── GS: EBUS-GS-GM, EBUS-GS-AUC
│   └── MS: EBUS-MS-GM, EBUS-MS-AUC
└── EUSCM
    ├── GS: EUSCM-GS-GM, EUSCM-GS-AUC
    └── MS: EUSCM-MS-GM, EUSCM-MS-AUC
```

### 2.5 EBUS 피트니스 함수 (수식 상세)

**EBUS-GS-GM (전역 선택, GM 평가):**

$$Fitness_{Bal}(S) = \begin{cases} g - \left|1 - \frac{n^+}{n^-}\right| \cdot P & \text{if } n^- > 0 \\ g - P & \text{if } n^- = 0 \end{cases} \tag{4}$$

여기서:
- $g$: 학습 데이터의 기하평균(GM) 정확도
- $n^+$: 선택된 소수 클래스 인스턴스 수
- $n^-$: 선택된 다수 클래스 인스턴스 수
- $P$: 페널티 계수 (실험적으로 $P = 0.2$ 권장)

**EBUS-MS-GM (다수 클래스만 선택, GM 평가):**

$$Fitness_{Bal}(S) = \begin{cases} g - \left|1 - \frac{N^+}{n^-}\right| \cdot P & \text{if } n^- > 0 \\ g - P & \text{if } n^- = 0 \end{cases} \tag{5}$$

여기서 $N^+$는 원래 훈련 데이터의 소수 클래스 전체 인스턴스 수(상수).

**EBUS-GS-AUC:**

$$Fitness_{Bal}(S) = \begin{cases} AUC - \left|1 - \frac{n^+}{n^-}\right| \cdot P & \text{if } n^- > 0 \\ AUC - P & \text{if } n^- = 0 \end{cases} \tag{6}$$

**EBUS-MS-AUC:**

$$Fitness_{Bal}(S) = \begin{cases} AUC - \left|1 - \frac{N^+}{n^-}\right| \cdot P & \text{if } n^- > 0 \\ AUC - P & \text{if } n^- = 0 \end{cases} \tag{7}$$

### 2.6 EUSCM 피트니스 함수

페널티 계수 없이 분류 지표만 사용:

**EUSCM-GS-GM, EUSCM-MS-GM:**

$$Fitness(S) = g \tag{8}$$

**EUSCM-GS-AUC, EUSCM-MS-AUC:**

$$Fitness(S) = AUC \tag{9}$$

### 2.7 진화 알고리즘: CHC 모델

EUS는 **CHC (Cross generational elitist selection, Heterogeneous recombination, Cataclysmic mutation)** 모델을 핵심 진화 알고리즘으로 사용:

- **HUX(Heuristic Uniform Crossover):** 부모 간 다른 비트의 절반을 교환
- **근친 교배 방지(Incest Prevention):** 해밍 거리 기반 교배 임계값 ($L/4$로 초기화)
- **재초기화(Reinitialization):** 수렴 시 최적 염색체 기반으로 35% 비트 무작위 변경
- **파라미터:** 집단 크기 = 50, 평가 횟수 = 10,000, $P = 0.2$, HUX 포함 확률 = 0.25

### 2.8 성능 향상

**전체 28개 데이터셋 기준 주요 결과 (Table 8):**

| 방법 | GM (test) | AUC (test) |
|------|-----------|------------|
| None (1-NN) | 0.6958 | 0.7606 |
| RUS | 0.7757 | 0.7892 |
| NCL | 0.7385 | 0.7862 |
| OSS | 0.7455 | 0.7837 |
| **EBUS-MS-GM** | **0.7971** | **0.8085** |
| SBC | 0.3382 | 0.6063 |

**고도 불균형 데이터셋 (IR > 9)에서의 비교:**

- EBUS-MS-GM은 TL, NCL, OSS보다 통계적으로 유의미하게 우수 (Holm's test, $\alpha = 0.05$)
- RUS와는 AUC 기준으로 유의미한 차이 존재 ($p = 0.0086$)

### 2.9 한계

1. **계산 비용:** 진화 알고리즘 특성상 대규모 데이터셋에서 실행 시간이 매우 증가 (염색체 크기 = 인스턴스 수)
2. **1-NN 의존성:** 피트니스 함수 평가에 1-NN만 사용 → 다른 분류기로의 일반화 미검증
3. **이진 분류 한정:** 다중 클래스 문제에 대한 직접 적용 미연구
4. **P 파라미터 민감성:** $P$ 값에 따라 결과가 불안정할 수 있음 (논문에서 $P=0.2$를 경험적으로 선택)
5. **저도 불균형(IR < 9)에서의 성능:** NCL이 더 우수한 경우 존재

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화 성능 향상 메커니즘

EUS가 일반화 성능을 향상시키는 핵심 메커니즘은 다음과 같다:

**① 다수 클래스 선택(Majority Selection, MS)의 일반화 효과:**

MS 방식은 소수 클래스 인스턴스를 전부 보존하면서 다수 클래스를 선별적으로 제거한다. 이는 결정 경계(decision boundary) 근방에 있는 정보를 보존하는 효과를 가져와 과적합을 방지한다.

$$\text{목표}: \quad n^- \approx N^+ \implies |1 - \frac{N^+}{n^-}| \to 0$$

**② 페널티 기반 균형 유지:**

$Fitness_{Bal}$의 페널티 항 $\left|1 - \frac{n^+}{n^-}\right| \cdot P$는 극단적인 클래스 편향을 방지하여, 선택된 부분집합이 특정 클래스에만 특화되지 않도록 한다.

**③ GS 모델의 잠재적 일반화 위험:**

논문 내에서도 명시적으로 언급되듯, EUSCM-GS 모델은 페널티 없이 전역 선택을 수행하므로:

> *"The disadvantage of lack of generalization capability may be pointed out in this model."* (Section 4.2)

반면, EBUS-GS 모델은 페널티 항이 있어 소수 클래스 과적합을 억제한다.

**④ 진화 탐색의 전역 최적화:**

CHC의 HUX 연산자와 재초기화는 탐색 공간의 다양성을 유지하여 지역 최적에 빠지지 않도록 한다. 이는 훈련 데이터 내 노이즈에 강인한 부분집합 선택을 가능하게 한다.

### 3.2 실험적 일반화 성능 증거

**훈련 vs. 테스트 GM 비교 (Table 6):**

| 모델 | GM (train) | GM (test) | 차이 |
|------|-----------|-----------|------|
| EBUS-MS-GM | 0.8862 | 0.7971 | 0.089 |
| EUSCM-GS-GM | 0.9068 | 0.7575 | **0.149** |
| EBUS-GS-GM | 0.8996 | 0.7927 | 0.107 |

→ **EBUS-MS-GM이 훈련-테스트 갭이 가장 작아 일반화 성능이 가장 우수**

**EPS-IGA의 과적합 사례:**

> *"EPS-IGA obtains the best result in training data, indicating us that it over-fits the selected instances to the training data."* (Section 6.1)

이는 단순 정확도 최적화 기반 PS가 불균형 데이터에서 일반화 실패를 야기함을 보여준다.

### 3.3 IR 수준에 따른 일반화 전략 권장

| IR 수준 | 권장 모델 | 이유 |
|---------|----------|------|
| IR < 9 | EUSCM-GS-AUC 또는 NCL | 균형 메커니즘 불필요, 전역 탐색이 유리 |
| IR > 9 | **EBUS-MS-GM** | 균형 페널티 + 소수 클래스 보존으로 일반화 향상 |

---

## 4. 향후 연구에 미치는 영향과 고려할 점

### 4.1 향후 연구에 미치는 영향

**① 분류 체계의 기초 확립**

본 논문이 제시한 (목적 × 선택 방식 × 평가 지표)의 3차원 분류 체계는 이후 불균형 학습 전처리 연구의 기준 틀이 되었다. 이 분류법은 이후 연구에서 새로운 방법론의 위치를 정의하는 기준으로 활용된다.

**② 전처리 독립성의 중요성 부각**

EUS가 분류기에 독립적인 전처리임을 강조함으로써, SVM, 딥러닝, 앙상블 등 다양한 분류기와 결합 가능한 전처리 방법론 연구를 촉진하였다.

**③ 비진화적 방법의 한계 명시**

특히 고도 불균형(IR > 9) 상황에서 NCL, TL 등의 전통적 방법이 성능 저하를 보임을 통계적으로 검증함으로써, 고도 불균형 도메인에 특화된 연구 방향성을 제시하였다.

**④ 통계적 비교 방법론의 표준화**

Friedman → Iman-Davenport → Holm's 절차의 계층적 통계 검증 프레임워크는 이후 머신러닝 비교 연구의 표준으로 자리잡았다.

### 4.2 향후 연구 시 고려할 점

**① 계산 확장성(Scalability) 문제**

염색체 길이가 데이터셋 크기($N$)와 동일하므로, 대용량 데이터(빅데이터)에서는 적용이 어렵다. 논문 자체에서도 이를 미래 연구 과제로 제시한다:

> *"A study on the scalability for making it feasible to apply Evolutionary Under-Sampling for very large data sets"*

→ 분산 진화 알고리즘, 계층적 샘플링(stratified sampling), 차원 축소와의 결합 연구 필요.

**② 다중 클래스 불균형으로의 확장**

본 논문은 이진 분류에 한정되어 있다. 다중 클래스 불균형 문제(multi-class imbalance)에서는 클래스 간 관계가 복잡해지므로 별도의 피트니스 함수 설계 필요.

**③ 데이터 복잡도(Data Complexity) 고려**

논문에서 제안하는 미래 연구:

> *"The analysis of Evolutionary Under-Sampling in terms of data complexity"*

클래스 오버랩(class overlap), 소수 클래스 클러스터링, 경계 인스턴스(borderline instances) 등 데이터 복잡도 측도를 피트니스 함수에 통합 필요.

**④ 다양한 분류기와의 결합 평가**

현재 1-NN만을 평가 분류기로 사용. C4.5, SVM, Random Forest, 딥러닝 등과의 결합 효과 검증이 필요하다.

**⑤ 피트니스 함수의 P 파라미터 자동화**

$P = 0.2$는 경험적으로 설정된 값이다. IR이나 데이터 특성에 따라 $P$를 적응적으로 조정하는 메타 학습(meta-learning) 기반 접근 연구 필요.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의:** 아래 내용은 2020년 이후 관련 분야의 연구 동향에 대한 일반적인 지식을 기반으로 하며, 개별 논문의 구체적 수치는 직접 해당 논문을 확인하시기 바랍니다. 부정확한 인용을 피하기 위해 확인된 방향성 중심으로 기술합니다.

### 5.1 연구 동향 비교

| 비교 항목 | García & Herrera (2008) | 2020년 이후 연구 동향 |
|-----------|------------------------|----------------------|
| **기반 알고리즘** | CHC, IGA | 딥러닝 기반 오버샘플링, GAN 기반 데이터 증강 |
| **주요 접근** | 진화적 언더샘플링 | 하이브리드 방법 (오버+언더), 앙상블+리샘플링 |
| **평가 지표** | GM, AUC | F1-score, PR-AUC, MCC 추가 |
| **데이터 규모** | 소규모 UCI 28개 | 대규모, 실시간 스트리밍 데이터 |
| **다중 클래스** | 이진 분류 한정 | 다중 클래스 불균형 적극 연구 |
| **설명 가능성** | 미고려 | XAI 관점 통합 연구 증가 |

### 5.2 주목할 만한 후속 연구 방향

**① SMOTE 계열의 발전:**
SMOTE (Chawla et al., 2002)의 한계를 극복하는 다양한 변형들이 제안되어 García & Herrera의 언더샘플링과 결합하는 하이브리드 접근이 연구되고 있다.

**② 딥러닝 기반 데이터 증강:**
GAN(Generative Adversarial Network)을 활용한 소수 클래스 합성 샘플 생성은 전통적 언더샘플링의 정보 손실 문제를 보완한다.

**③ 메타러닝 기반 자동 전처리:**
AutoML 맥락에서 데이터 특성에 따라 최적의 불균형 처리 방법을 자동 선택하는 연구 — 본 논문의 분류 체계가 탐색 공간 정의에 활용될 수 있다.

**④ 온라인/스트림 학습:**
실시간 데이터 스트림에서의 불균형 처리에서 진화 알고리즘의 점진적 학습 적용 연구.

**⑤ 불균형 + 개념 드리프트(Concept Drift):**
시간에 따라 클래스 분포가 변하는 동적 환경에서의 적응형 언더샘플링 연구.

---

## 참고 자료

- **주 논문:** García, S., & Herrera, F. "Evolutionary Under-Sampling for Classification with Imbalanced Data Sets: Proposals and Taxonomy." *Evolutionary Computation*, MIT Press (제공된 PDF 기반)
- Chawla, N.V., et al. (2002). "SMOTE: Synthetic minority over-sampling technique." *Journal of Artificial Intelligence Research*, 16:321–357.
- Batista, G.E.A.P.A., et al. (2004). "A study of the behavior of several methods for balancing machine learning training data." *SIGKDD Explorations*, 6(1):20–29.
- Demšar, J. (2006). "Statistical comparisons of classifiers over multiple data sets." *Journal of Machine Learning Research*, 7:1–30.
- Wilson, D.R., & Martinez, T.R. (2000). "Reduction techniques for instance-based learning algorithms." *Machine Learning*, 38(3):257–286.
- Cano, J.R., Herrera, F., & Lozano, M. (2003). "Using evolutionary algorithms as instance selection for data reduction in KDD." *IEEE Transactions on Evolutionary Computation*, 7(6):561–575.
- Eshelman, L.J. (1991). "The CHC adaptive search algorithm." *Foundations of Genetic Algorithms*, 265–283.
- Laurikkala, J. (2001). "Improving identification of difficult small classes by balancing class distribution." *AIME 2001*, 63–66.
