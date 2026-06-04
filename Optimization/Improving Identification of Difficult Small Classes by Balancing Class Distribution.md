
# Improving Identification of Difficult Small Classes by Balancing Class Distribution

> **저자:** Jorma Laurikkala (University of Tampere, Department of Computer and Information Sciences)
> **출처:** AIME 2001 (8th Conference on AI in Medicine in Europe), Cascais, Portugal
> **수록:** Lecture Notes in Artificial Intelligence, Vol. 2101, Springer, pp. 63–66
> **참고 링크:** [Springer Nature Link](https://link.springer.com/chapter/10.1007/3-540-48229-6_9) | [Semantic Scholar](https://www.semanticscholar.org/paper/Improving-Identification-of-Difficult-Small-Classes-Laurikkala/e51e0633e12f5c037f1e405ccc31c4c50f5ae87f) | [ResearchGate](https://www.researchgate.net/publication/225132277_Improving_Identification_of_Difficult_Small_Classes_by_Balancing_Class_Distribution) | [Tampere University Research Portal](https://researchportal.tuni.fi/en/publications/improving-identification-of-difficult-small-classes-by-balancing-)

---

## 1. 📌 핵심 주장 및 주요 기여 요약

이 논문은 **클래스 불균형(imbalanced class distribution)** 문제에서 소수 클래스(minority class)의 식별 성능을 향상시키기 위해, 데이터 축소(data reduction)를 통한 클래스 분포 균형화 방법 3가지를 연구한다.

**핵심 기여:**

논문의 핵심 기여는 새로운 방법인 **Neighborhood Cleaning Rule (NCL)**을 제안한 것으로, 이는 단순 무작위 샘플링 및 One-Sided Selection(OSS) 방법을 10개의 실제 데이터셋 실험에서 능가하였다.

모든 축소 방법이 소수 클래스 식별률을 20~30% 향상시켰지만, 방법 간 차이는 미미하였다. 그러나 3-최근접이웃(3-NN) 분류기와 C4.5 의사결정나무에서의 정확도, 진양성률(True-Positive Rate), 진음성률(True-Negative Rate)의 유의미한 차이는 NCL에 유리하게 나타났다.

결과적으로 NCL은 **어려운 소수 클래스의 모델링 향상** 및 실세계 불균형 데이터로부터 해당 클래스를 식별하는 분류기 구축에 유용한 방법임이 제안된다.

---

## 2. 🔍 상세 분석

### 2.1 해결하고자 하는 문제

불균형 데이터셋은 일부 클래스가 다른 클래스보다 인스턴스를 훨씬 더 많이 갖는 상황으로, 이는 편향된 예측과 머신러닝 모델의 낮은 성능을 유발한다.

구체적으로 이 논문은 다음 두 가지 어려움이 **동시에** 존재하는 상황을 다룬다:

1. **소수 클래스(Small Class):** 데이터 내 인스턴스 수가 절대적으로 적음
2. **어려운 클래스(Difficult Class):** 클래스 간 경계가 불분명하거나 노이즈가 많아 분류 자체가 어려움

기존 분류기(예: C4.5, kNN)는 다수 클래스에 편향(bias)되어, 소수 클래스의 재현율(recall)이 극도로 낮아지는 문제가 발생한다.

---

### 2.2 비교 대상 3가지 방법

| 방법 | 유형 | 특징 |
|------|------|------|
| **Random Undersampling (RUS)** | 비선택적 언더샘플링 | 다수 클래스에서 무작위 제거 |
| **One-Sided Selection (OSS)** | 선택적 언더샘플링 | Tomek Links + CNN 결합 |
| **Neighborhood Cleaning Rule (NCL)** | 선택적 언더샘플링 (신규 제안) | ENN 기반 2단계 정제 |

---

### 2.3 NCL 알고리즘 및 수식

NCL은 **Edited Nearest Neighbor (ENN)** 방법을 수정하여 데이터 정제(cleaning)의 역할을 강화한 방법이다.

#### 📐 NCL의 2단계 절차

NCL 알고리즘은 두 단계로 구성된다. **1단계**에서는 ENN 알고리즘을 사용하여 관심 클래스($C_l$)에 속하지 않는 샘플 중 오분류된 것을 제거한다. **2단계**에서는 $C_l$에 속하는 샘플들의 이웃을 추가적으로 정제한다.

이를 위해 $C_l$의 샘플들에 대한 $k$-최근접이웃을 탐색하며, 해당 이웃이 모두 $C_l$에 속하지 않는 클래스 레이블을 가지고, 그 샘플이 $C_l$ 내 소수 클래스의 절반보다 큰 클래스에 속할 경우 해당 샘플을 제거한다. 두 단계 모두에서 $C_l$에 속하는 샘플은 항상 유지된다.

**수식 표현:**

**1단계 — ENN 기반 제거 조건:**

$$
\text{제거 조건: } \hat{y}(\mathbf{x}_i) \neq y_i \quad \text{(3-NN 분류 결과가 실제 레이블과 다를 때)}
$$

$$
\hat{y}(\mathbf{x}_i) = \arg\max_{c} \sum_{j \in \mathcal{N}_k(\mathbf{x}_i)} \mathbf{1}[y_j = c]
$$

여기서 $\mathcal{N}_k(\mathbf{x}_i)$는 $\mathbf{x}_i$의 $k$-최근접 이웃 집합 ($k=3$).

**2단계 — 추가 정제 조건:**

$$
\text{제거 조건: } \forall j \in \mathcal{N}_k(\mathbf{x}_i),\ y_j \notin C_l \quad \text{AND} \quad |C_{y_i}| > \frac{1}{2}|C_{\min}|
$$

여기서:
- $C_l$ : 보호할 소수 클래스 집합
- $|C_{y_i}|$ : $\mathbf{x}_i$가 속한 클래스의 크기
- $|C_{\min}|$ : $C_l$ 내 가장 작은 클래스의 크기

**평가 지표:**

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

$$
\text{True Positive Rate (Recall)} = \frac{TP}{TP + FN}
$$

$$
\text{True Negative Rate (Specificity)} = \frac{TN}{TN + FP}
$$

$$
\text{Geometric Mean (G-Mean)} = \sqrt{TPR \times TNR}
$$

---

### 2.4 모델 구조

논문의 분류기(classifier) 구조는 다음과 같다:

```
[원본 불균형 데이터]
        ↓
[데이터 전처리: 언더샘플링]
  ├─ RUS (Random Undersampling)
  ├─ OSS (One-Sided Selection)
  └─ NCL (Neighborhood Cleaning Rule) ← 제안 방법
        ↓
[균형화된 훈련 데이터]
        ↓
[분류기 학습]
  ├─ 3-Nearest Neighbor (3-NN)
  └─ C4.5 Decision Tree
        ↓
[평가: Accuracy, TPR, TNR, G-Mean]
```

- **데이터셋:** 10개의 실세계 데이터셋 (의료 분야 중심)
- **분류기:** 3-NN, C4.5 (Quinlan, 1993)
- **비교 기준:** 균형화 전/후 성능, 방법 간 통계적 유의성 검정

---

### 2.5 성능 향상

NCL은 10개의 데이터셋 실험에서 단순 무작위 및 OSS 방법을 능가하였다. 모든 축소 방법이 소수 클래스 식별률을 20–30% 향상시켰으나, 방법 간 차이는 미미하였다. 그러나 축소 데이터에서 3-NN과 C4.5로 얻은 정확도, 진양성률, 진음성률의 유의미한 차이는 NCL에 유리하였다.

| 지표 | 균형화 전 | 균형화 후 (전체) | NCL 우위 |
|------|-----------|-----------------|---------|
| TPR (소수 클래스 재현율) | 낮음 | +20~30% 향상 | ✅ 유의미한 우위 |
| Accuracy | 높음 (편향) | 감소 가능 | ✅ NCL 우위 |
| TNR | 높음 (편향) | 약간 감소 | ✅ NCL 우위 |

---

### 2.6 한계점

1. **소규모 실험 규모:** 10개 데이터셋에 국한 → 일반화에 한계
2. **고정 $k=3$:** 데이터셋 특성에 따라 최적 $k$가 다를 수 있음
3. **단순 분류기만 사용:** SVM, 앙상블, 딥러닝 미검증
4. **다중 클래스 불균형 미고려:** 이진 분류 중심
5. **오버샘플링과의 미결합:** NCL 단독 사용 → 하이브리드 방법 미탐색
6. **통계적 유의성 불명확:** 일부 차이가 "미미(insignificant)"하다고 보고됨

---

## 3. 🎯 모델의 일반화 성능 향상 가능성

결과는 NCL이 어려운 소수 클래스 모델링 향상뿐 아니라, **불균형 분포를 가진 실세계 데이터**에서 이러한 클래스를 식별하는 분류기 구축에도 유용한 방법임을 제안한다.

### 일반화 성능 향상 메커니즘

NCL은 다음 두 가지 방식으로 일반화(generalization)에 기여한다:

**① 노이즈 제거에 의한 결정 경계 정제**

$$
\text{일반화 오류} \approx \text{훈련 오류} + \Omega(\mathcal{H})
$$

- ENN 기반 노이즈 제거 → 결정 경계 근처의 모호한 샘플 제거
- 분류기가 과적합하지 않도록 훈련 데이터를 "정제"

**② 클래스 분포 균형화에 의한 편향(bias) 감소**

$$
\text{Imbalance Ratio (IR)} = \frac{|C_{\text{majority}}|}{|C_{\text{minority}}|}
$$

- IR이 높을수록 다수 클래스로의 편향 증가
- NCL은 다수 클래스의 경계 근처 샘플만 선택적으로 제거 → IR 감소 + 정보 손실 최소화

**③ 실세계 적용 가능성**

NCL은 다수 클래스 중 소수 클래스 인스턴스에 가까운 인스턴스를 제거함으로써 데이터셋의 균형을 맞추고 분류 알고리즘의 성능을 향상시킨다. 이는 이웃 기반 접근 방식을 통해 이러한 인스턴스를 탐색하고 제거하는 방식으로 작동한다.

**④ 하이브리드 확장 가능성**

클래스 불균형 문제 해결을 위해 언더샘플링과 오버샘플링 절차를 함께 활용하며, SMOTE와 NCL의 결합 방식이 데이터셋 균형화에 탐구되고 있다.

---

## 4. 🔮 앞으로의 연구에 미치는 영향 및 연구 시 고려할 점

### 4.1 연구에 미치는 영향

**① NCL의 표준 도구화**

NCL은 Laurikkala가 제안한 가장 인기 있는 언더샘플링 방법 중 하나로 자리 잡았다. `imbalanced-learn`(Python) 등의 주요 머신러닝 라이브러리에 표준 기법으로 포함되어 광범위하게 사용되고 있다.

**② SMOTE+NCL 하이브리드 연구의 촉발**

불균형 데이터 분류 문제에서 소수 클래스가 심각하게 과소 표현되어 편향된 모델 성능을 초래하는 도전에 대응하기 위해, SMOTE와 NCL의 결합(SMOTE-NCL) 및 SMOTE-ENN과 같은 하이브리드 샘플링 기법의 효과성이 집중 연구되고 있다.

**③ 딥러닝 시대로의 확장**

DeepSMOTE와 같이 딥러닝과 SMOTE를 융합한 방법이 IEEE Trans. Neural Networks Learn. Syst.에 발표(2023)되는 등, NCL의 아이디어는 딥러닝 기반 불균형 학습으로 계승 확장되고 있다.

**④ 앙상블 기반 불균형 학습**

랜덤 언더샘플링을 배깅 또는 부스팅 앙상블과 결합하는 것이 효과적임이 입증되어, 전처리 단독 방법보다 앙상블 기반 알고리즘의 우수성이 강조되고 있다.

---

### 4.2 앞으로 연구 시 고려할 점

| 고려 항목 | 설명 |
|-----------|------|
| **적응적 $k$ 선택** | 데이터셋별 최적 이웃 수 $k$ 자동 결정 필요 |
| **딥러닝과의 통합** | NCL + CNN/Transformer 기반 특징 공간에서의 정제 |
| **다중 클래스 불균형** | 이진 분류 → 다중 클래스 확장 알고리즘 설계 |
| **고차원/희소 데이터** | 유클리드 거리 기반 이웃 탐색의 한계 극복 (차원의 저주) |
| **GAN 기반 오버샘플링 결합** | NCL(언더샘플링) + GAN/CTGAN(오버샘플링) 하이브리드 |
| **평가 지표 다양화** | G-Mean, F1-score, AUC-ROC, MCC 등 종합 사용 |
| **실시간/스트리밍 데이터** | 정적 데이터 가정에서 벗어난 온라인 학습 환경 적용 |

---

## 5. 📊 2020년 이후 관련 최신 연구 비교 분석

언더샘플링은 다수 클래스 샘플 수를 줄여 균형을 맞추는 방법으로, 불균형 클래스 분포로부터의 학습은 수십 년간의 연구 발전에도 불구하고 여전히 주목받는 연구 분야이다. 불균형 클래스 분포는 실제 응용에서 머신러닝과 딥러닝 모델의 실용적 유용성을 저해하며, 최근 다수의 연구가 이 분야에서 상당한 진전을 이루고 있다.

| 연구 | 방법 | 특징 | Laurikkala(2001) 대비 |
|------|------|------|-----------------------|
| **DeepSMOTE (2023)** | 딥러닝+SMOTE | 잠재 공간에서 소수 클래스 합성 | 특징 공간 활용 강화 |
| **SMOTE-NCL 하이브리드** | Over+Undersampling 결합 | SMOTE 후 NCL로 노이즈 제거 | NCL을 2단계 정제에 활용 |
| **αDBAMOTE+DenseNet (2021)** | SMOTE 변형 + CNN | 경계 근처 소수 샘플만 합성 | 딥러닝 구조 결합 |
| **SMOTE+BiLSTM (2023)** | 오버샘플링+딥러닝 | 시계열 의료 데이터 적용 | 순차 데이터 확장 |
| **k-Means SMOTE** | 클러스터링+SMOTE | 군집 기반 샘플 합성 | 공간 분포 고려 |

불균형 데이터는 다수 클래스가 소수 클래스보다 훨씬 많은 샘플을 가져 모델 학습 과정이 다수 클래스로 편향되는 상황이며, 최근 몇 년간 소수 클래스의 새 데이터를 합성 생성하거나 다수 클래스 수를 줄이는 방식의 여러 해법이 제안되었다. DNN과 CNN 기반 방법과 다양한 오버샘플링/언더샘플링의 결합 효과성이 연구되고 있다.

불균형 클래스 분포는 실제 응용에서 머신러닝뿐 아니라 딥러닝 모델의 실용적 유용성도 저해한다는 점에서, NCL과 같은 전통적 언더샘플링 기법의 원리는 현대 딥러닝 기반 파이프라인에서도 전처리 단계로 여전히 활발히 사용되고 있다.

---

## 📚 참고 자료 출처

1. **Laurikkala, J. (2001).** *Improving Identification of Difficult Small Classes by Balancing Class Distribution.* AIME 2001, Lecture Notes in Artificial Intelligence, Vol. 2101, Springer, pp. 63–66.
   - Springer Nature Link: https://link.springer.com/chapter/10.1007/3-540-48229-6_9
   - Semantic Scholar: https://www.semanticscholar.org/paper/Improving-Identification-of-Difficult-Small-Classes-Laurikkala/e51e0633e12f5c037f1e405ccc31c4c50f5ae87f
   - ResearchGate: https://www.researchgate.net/publication/225132277

2. **Laurikkala, J. (2002).** *Instance-based data reduction for improved identification of difficult small classes.* Intelligent Data Analysis, SAGE Journals. https://journals.sagepub.com/doi/abs/10.3233/IDA-2002-6402

3. **Tampere University Research Portal.** Improving Identification of Difficult Small Classes by Balancing Class Distribution. https://researchportal.tuni.fi/en/publications/improving-identification-of-difficult-small-classes-by-balancing-

4. **Activeloop Glossary.** Neighbourhood Cleaning Rule (NCL). https://www.activeloop.ai/resources/glossary/neighbourhood-cleaning-rule-ncl/

5. **Machine Learning Mastery (2021).** Undersampling Algorithms for Imbalanced Classification. https://machinelearningmastery.com/undersampling-algorithms-for-imbalanced-classification/

6. **Dablain, D., Krawczyk, B. & Chawla, N.V. (2023).** DeepSMOTE: Fusing Deep Learning and SMOTE for Imbalanced Data. *IEEE Trans. Neural Networks Learn. Syst.*, 34(9), 6390–6404.

7. **Springer AI Review (2024).** A survey on imbalanced learning: latest research, applications and future directions. https://link.springer.com/article/10.1007/s10462-024-10759-6

8. **Springer AI Review (2024).** Handling imbalanced medical datasets: review of a decade of research. https://link.springer.com/article/10.1007/s10462-024-10884-2

9. **MDPI Applied Sciences (2023).** Effective Class-Imbalance Learning Based on SMOTE and Convolutional Neural Networks. https://www.mdpi.com/2076-3417/13/6/4006

10. **RDocumentation (UBL Package).** NCLClassif: Neighborhood Cleaning Rule Algorithm. https://www.rdocumentation.org/packages/UBL/versions/0.0.6/topics/NCLClassif

> ⚠️ **정확도 주의사항:** 원문 논문(2001)은 단 4페이지 분량의 단편 논문으로, 본 분석은 공개된 초록, 인용 데이터베이스(Semantic Scholar, ResearchGate, Springer), 관련 후속 연구를 기반으로 작성되었습니다. 수식의 세부 표기는 NCL의 공식 문서 및 후속 연구를 참고하여 재구성하였으며, 원문에 직접 접근하여 검증하는 것을 권장합니다.
