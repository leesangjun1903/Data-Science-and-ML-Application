# Safe-Level-SMOTE: Safe-Level-Synthetic Minority Over-sampling TEchnique for handling the class imbalanced problem

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장
Safe-Level-SMOTE는 기존 SMOTE의 **과일반화(overgeneralization) 문제**를 해결하기 위해, 소수 클래스 합성 인스턴스를 생성할 때 **안전 수준(safe level)**이라는 가중치 개념을 도입하여, 노이즈 및 경계 영역이 아닌 **안전한 영역(safe region)에만 합성 샘플을 배치**해야 한다는 것이다.

### 주요 기여
| 기여 항목 | 내용 |
|---|---|
| Safe Level 개념 도입 | 각 소수 클래스 인스턴스의 $k$ 최근접 이웃 내 소수 클래스 수를 기반으로 안전도 수치화 |
| Safe Level Ratio 기반 배치 | 합성 인스턴스의 생성 위치를 두 인스턴스의 안전도 비율에 따라 조정 |
| 5가지 케이스 분류 | 노이즈-노이즈, 노이즈-안전, 동등, p가 더 안전, n이 더 안전 상황별 처리 |
| SMOTE·Borderline-SMOTE 대비 성능 향상 | Precision 및 F-value에서 C4.5 기반 실험 기준 우수한 성능 입증 |

---

## 2. 상세 설명

### 2-1. 해결하고자 하는 문제

**클래스 불균형 문제(Class Imbalanced Problem)**는 소수 클래스의 인스턴스 수가 다수 클래스에 비해 현저히 적을 때 발생한다. 일반적인 분류기는 다수 클래스에 편향되어 소수 클래스를 무시하는 경향이 있다.

기존 접근법의 한계:
- **Random Over-sampling**: 소수 클래스 인스턴스를 단순 복제 → 과적합(overfitting) 유발
- **Random Under-sampling**: 다수 클래스 제거 → 중요한 정보 손실
- **SMOTE**: $k$ 최근접 이웃을 잇는 선분 위에 무작위로 합성 인스턴스 생성 → **과일반화 문제** (다수 클래스 영역, 노이즈 영역에도 샘플 생성)
- **Borderline-SMOTE**: 경계선 인스턴스만 과샘플링 → 경계 판별이 이분법적이어서 $n=k$인 노이즈와 $n=k-1$인 경계 인스턴스 처리가 불합리

### 2-2. 제안하는 방법 (수식 포함)

#### 성능 평가 지표

$$\text{Accuracy} = \frac{TP + TN}{TP + FN + FP + TN} $$

$$\text{Recall} = \frac{TP}{TP + FN} $$

$$\text{Precision} = \frac{TP}{TP + FP} $$

$$\text{F-value} = \frac{(1+\beta^2) \cdot \text{Recall} \cdot \text{Precision}}{\beta^2 \cdot \text{Recall} + \text{Precision}} $$

$$\text{TP Rate} = \frac{TP}{TP + FN} $$

$$\text{FP Rate} = \frac{FP}{TN + FP} $$

#### Safe Level 및 Safe Level Ratio 정의

$$sl = \text{(k 최근접 이웃 내 소수 클래스 인스턴스 수)} $$

$$\text{sl ratio} = \frac{sl_p}{sl_n} $$

여기서 $sl_p$는 소수 클래스 인스턴스 $p$의 safe level, $sl_n$은 $p$의 최근접 이웃 $n$의 safe level이다.

#### 합성 인스턴스 생성 공식

$$s[\text{atti}] = p[\text{atti}] + \text{gap} \cdot (n[\text{atti}] - p[\text{atti}]) $$

여기서 $\text{gap}$의 범위는 아래 5가지 케이스에 따라 결정된다.

### 2-3. 모델 구조 (5가지 케이스)

| 케이스 | 조건 | gap 범위 | 의미 |
|---|---|---|---|
| 1 | $\text{sl ratio} = \infty$ AND $sl_p = 0$ | 생성 안 함 | $p$, $n$ 모두 노이즈 |
| 2 | $\text{sl ratio} = \infty$ AND $sl_p \neq 0$ | $\text{gap} = 0$ | $n$이 노이즈 → $p$ 복제 |
| 3 | $\text{sl ratio} = 1$ | $\text{gap} \in [0, 1]$ | $p$와 $n$ 동일 안전도 → 선분 전체 |
| 4 | $\text{sl ratio} > 1$ | $\text{gap} \in \left[0, \frac{1}{\text{sl ratio}}\right]$ | $p$가 더 안전 → $p$ 쪽에 생성 |
| 5 | $\text{sl ratio} < 1$ | $\text{gap} \in [1-\text{sl ratio}, 1]$ | $n$이 더 안전 → $n$ 쪽에 생성 |

**케이스 4의 수식 상세:**

$$sl_p > sl_n \Rightarrow \text{gap} \in \left[0, \frac{sl_n}{sl_p}\right] $$

**케이스 5의 수식 상세:**

$$sl_p < sl_n \Rightarrow \text{gap} \in \left[1 - \frac{sl_p}{sl_n}, 1\right] $$

### 알고리즘 흐름

```
Input:  D (원본 소수 클래스 인스턴스 집합)
Output: D' (합성 소수 클래스 인스턴스 집합)

1. D' = ∅
2. for each p in D:
   a. p의 k 최근접 이웃 계산, 무작위로 n 선택
   b. sl_p, sl_n 계산
   c. sl_ratio = sl_p / sl_n (sl_n=0이면 ∞)
   d. 5가지 케이스에 따라 gap 결정
   e. s[i] = p[i] + gap × (n[i] - p[i])
   f. D' = D' ∪ {s}
3. return D'
```

### 2-4. 실험 설정 및 성능 향상

**실험 설정:**
- 데이터셋: UCI Repository의 Satimage (9.73% 소수), Haberman (26.47% 소수)
- 분류기: C4.5 결정 트리, Naïve Bayes, SVM
- 평가: 10-fold 교차 검증, $k=5$, $\beta=1$

**성능 결과 요약:**

| 평가 지표 | 분류기 | 데이터셋 | 결과 |
|---|---|---|---|
| Precision | C4.5 | Haberman | **Safe-Level-SMOTE 최우수** |
| F-value | C4.5 | Satimage | **Safe-Level-SMOTE 최우수** |
| Recall | Naïve Bayes | Satimage | Borderline-SMOTE 1위, Safe-Level-SMOTE 2위 |
| AUC | SVM | Haberman | SMOTE와 유사, Borderline-SMOTE 저조 |

**한계점:**
1. **수치형 속성만 처리 가능**: 범주형(nominal) 속성을 가진 데이터셋에 적용 불가
2. **SVM에서 효과 미미**: 모든 과샘플링 기법이 유사한 볼록(convex) 영역을 생성하여 초평면 결과가 구별되지 않음
3. **실험 데이터셋 수 부족**: 단 2개의 UCI 데이터셋만 사용
4. **합성 인스턴스 생성량의 자동 결정 미해결**: 최적의 과샘플링 비율을 자동으로 결정하는 메커니즘 부재
5. **Safe Level 정의의 단일성**: 하나의 safe level 정의만 사용

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 일반화 성능 향상 메커니즘

Safe-Level-SMOTE는 다음 메커니즘을 통해 **일반화 성능**을 향상시킨다:

**① 안전 영역 집중 샘플링:**

$$sl_p \approx k \Rightarrow \text{안전한 영역(safe region)} \Rightarrow \text{노이즈 없는 합성 샘플}$$

$k$ 최근접 이웃 대부분이 소수 클래스인 영역에 합성 샘플을 집중 배치함으로써, 학습 결정 경계가 다수 클래스 영역을 침범하지 않는다.

**② 과일반화(Overgeneralization) 방지:**

SMOTE는 아래처럼 선분 전체에서 균일하게 샘플링한다:

$$s = p + \text{gap} \cdot (n - p), \quad \text{gap} \in [0,1] \quad \text{(SMOTE)}$$

반면 Safe-Level-SMOTE는:

$$s = p + \text{gap} \cdot (n - p), \quad \text{gap} \in \left[0, \frac{1}{\text{sl ratio}}\right] \text{ 또는 } [1-\text{sl ratio}, 1] \quad \text{(Safe-Level-SMOTE)}$$

이를 통해 다수 클래스가 밀집한 방향으로의 과도한 확장을 억제한다.

**③ 노이즈 인스턴스 제외:**

$sl_p = 0$이고 $sl_n = 0$인 경우(케이스 1), 합성 인스턴스를 생성하지 않음으로써 노이즈 주변 영역의 학습 왜곡을 방지한다.

**④ 결정 트리의 일반화:**

논문에서 SMOTE의 원래 의도인 "더 크고 덜 구체적인 결정 영역 학습"을 Safe-Level-SMOTE가 더 정교하게 달성한다. C4.5에서 Precision과 F-value 모두 향상된 것이 이를 뒷받침한다.

### 3-2. 일반화 관련 한계

- **Recall-Precision 트레이드오프**: Safe-Level-SMOTE는 안전 영역에만 샘플을 생성하므로 Recall이 Borderline-SMOTE보다 낮을 수 있다 (Naïve Bayes 실험에서 확인됨).
- **소수 클래스 밀도에 민감**: 소수 클래스가 매우 희소하면 safe level이 전반적으로 낮아져 대부분의 인스턴스가 노이즈로 판별될 수 있다.
- **$k$ 값 민감도**: $k$ 값 선택에 따라 safe level 계산 결과가 달라져 일반화 성능이 변동될 수 있다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려 사항

### 4-1. 연구에 미치는 영향

**① 안전 영역 기반 샘플링 패러다임 확립:**
Safe-Level-SMOTE는 단순한 선분 보간이 아닌, **이웃 정보를 활용한 적응형 보간**의 선구적 사례로서, 이후 ADASYN, ROSE, 클러스터 기반 SMOTE 등의 이론적 토대가 되었다.

**② 노이즈 필터링과 과샘플링의 통합 필요성 인식:**
논문이 노이즈 인스턴스를 명시적으로 처리(케이스 1, 2)함으로써, 이후 연구들이 **전처리 단계에서의 노이즈 제거**와 **과샘플링을 통합**하는 방향으로 발전하는 데 기여했다.

**③ 다양한 분류기와의 상호작용 연구 촉진:**
SVM에서 효과가 미미하다는 발견은, 과샘플링 기법이 분류기의 특성에 따라 다르게 작동함을 보여줌으로써 **분류기-샘플링 기법 간 상호작용 연구**의 필요성을 제기했다.

### 4-2. 향후 연구 시 고려할 점

| 고려 사항 | 설명 |
|---|---|
| Safe Level 정의 다양화 | 밀도 기반, 거리 가중 등 다양한 safe level 정의 탐색 |
| 범주형 속성 지원 | 수치형 외 범주형 속성을 가진 혼합 데이터 처리 방법 개발 |
| 합성량 자동 결정 | 데이터 분포에 따라 최적 과샘플링 비율을 자동 결정하는 메커니즘 |
| 딥러닝과의 결합 | GAN, VAE 기반 생성 모델과 Safe Level 개념 통합 |
| 다중 클래스 불균형 확장 | 이진 분류를 넘어 다중 클래스 불균형 처리로 확장 |
| 고차원 데이터 적용 | 텍스트, 이미지 등 고차원 데이터에서의 $k$-NN 기반 safe level 계산 효율화 |
| 클래스 오버랩 처리 | 클래스 간 경계가 복잡하게 겹치는 경우에 대한 robust한 처리 방법 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의**: 아래 연구들은 제가 학습한 데이터 기반 정보이며, 논문 원문을 직접 확인하지 않은 항목은 정확성에 제한이 있을 수 있습니다. 따라서 확실하게 알려진 연구 방향과 대표적 방법론 위주로 기술합니다.

### 5-1. 주요 후속 연구 방향

#### (1) 생성적 적대 신경망(GAN) 기반 오버샘플링

**CTGAN, TVAE** 등의 생성 모델은 Safe-Level-SMOTE가 해결하지 못한 **복잡한 비선형 분포**의 데이터를 합성할 수 있다.

$$G(z) \approx p_{data}(x_{minority}), \quad z \sim \mathcal{N}(0, I) $$

Safe-Level-SMOTE와의 차이점:
- Safe-Level-SMOTE: 두 인스턴스 간 선형 보간
- GAN 기반: 전체 분포를 학습하여 비선형 샘플 생성

#### (2) 클러스터 기반 SMOTE 계열

**K-Means SMOTE** (Douzas et al., 2018; 이후 2020년대에도 활발히 연구):

$$\text{클러스터 내 소수 클래스 비율} = \frac{|C_i^{minority}|}{|C_i|} $$

클러스터별로 과샘플링 비율을 조정하여 **밀도 불균형**을 추가로 해결한다. Safe-Level-SMOTE의 국소적(local) 안전도 개념을 클러스터 수준으로 확장한 개념으로 볼 수 있다.

#### (3) 앙상블 기반 불균형 학습

**BalancedBaggingClassifier**, **EasyEnsemble** 등은 샘플링과 앙상블을 통합한다. 이는 Safe-Level-SMOTE가 단일 분류기(C4.5, Naïve Bayes, SVM)에 의존했던 한계를 보완한다.

### 5-2. 비교 테이블

| 방법 | 합성 방식 | 노이즈 처리 | 비선형 분포 | 범주형 지원 | 고차원 적용 |
|---|---|---|---|---|---|
| SMOTE (2002) | 선형 보간 | ❌ | ❌ | ❌ | 제한적 |
| Borderline-SMOTE (2005) | 선형 보간 (경계) | 부분적 | ❌ | ❌ | 제한적 |
| **Safe-Level-SMOTE (2009)** | **가중 선형 보간** | **✅ (케이스 1,2)** | ❌ | ❌ | 제한적 |
| ADASYN (2008) | 적응형 밀도 기반 | 부분적 | ❌ | ❌ | 제한적 |
| K-Means SMOTE (2018~) | 클러스터 내 보간 | ✅ | 부분적 | ❌ | 제한적 |
| CTGAN (2019~) | GAN 생성 | ✅ | **✅** | **✅** | **✅** |

### 5-3. Safe-Level-SMOTE의 현재적 의의

Safe-Level-SMOTE는 **"어디에 샘플을 생성할 것인가"** 라는 근본적 질문을 제기했다. 2020년 이후 연구들은 이 질문에 대해:

1. **공간적 안전성** (Safe-Level-SMOTE의 계승): 밀도 추정, 클러스터링 활용
2. **분포 학습** (GAN 계열): 전체 소수 클래스 분포를 학습하여 고품질 샘플 생성
3. **메타러닝** (AutoML 결합): 데이터 특성에 따라 최적 샘플링 전략 자동 선택

의 세 방향으로 발전시키고 있다.

---

## 참고 자료

**주요 참고 자료 (논문 원문 기반):**

1. **Bunkhumpornpat, C., Sinapiromsaran, K., & Lursinsap, C. (2009).** Safe-Level-SMOTE: Safe-Level-Synthetic Minority Over-sampling TEchnique for Handling the Class Imbalanced Problem. *Advances in Knowledge Discovery and Data Mining (PAKDD 2009)*, LNAI 5476, pp. 475–482. Springer.

2. **Chawla, N., Bowyer, K., Hall, L., & Kegelmeyer, W. (2002).** SMOTE: Synthetic Minority Over-Sampling Technique. *Journal of Artificial Intelligence Research*, 16, 321–357.

3. **Han, H., Wang, W., & Mao, B. (2005).** Borderline-SMOTE: A New Over-Sampling Method in Imbalanced Data Sets Learning. *ICIC 2005*, LNCS 3644, pp. 878–887. Springer.

4. **Bradley, A. (1997).** The Use of the Area Under the ROC Curve in the Evaluation of Machine Learning Algorithms. *Pattern Recognition*, 30(6), 1145–1159.

5. **Quinlan, J. (1992).** C4.5: Programs for Machine Learning. Morgan Kaufmann.

6. **Blake, C., & Merz, C. (1998).** UCI Repository of Machine Learning Databases. http://archive.ics.uci.edu/ml/

**비교 분석 참고 (2020년 이후 연구 방향 관련):**

7. **Douzas, G., Bacao, F., & Last, F. (2018).** Improving imbalanced learning through a heuristic oversampling method based on k-means and SMOTE. *Information Sciences*, 465, 1–20. *(2020년 이후에도 활발히 인용됨)*

8. **Xu, Z., Shen, D., Nie, T., & Kou, Y. (2020).** A cluster-based oversampling algorithm combining SMOTE and k-means for imbalanced medical data. *Information Sciences*, 572, 574–589.

> **정확도 관련 고지**: 2020년 이후 최신 연구 비교 분석 부분은 제가 학습한 데이터(2023년 초까지)를 기반으로 하며, 일부 세부 수치나 논문 제목은 실제와 다를 수 있습니다. 정확한 최신 연구는 Google Scholar, IEEE Xplore, Springer Link 등에서 "Safe-Level-SMOTE", "imbalanced learning oversampling 2020" 등으로 직접 검색하시길 권장합니다.
