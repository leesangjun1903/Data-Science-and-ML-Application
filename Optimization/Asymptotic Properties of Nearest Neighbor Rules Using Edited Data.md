
# Asymptotic Properties of Nearest Neighbor Rules Using Edited Data

**저자:** Dennis L. Wilson | **게재:** IEEE Transactions on Systems, Man, and Cybernetics, Vol. SMC-2, No. 3, pp. 408–421, 1972

---

## 1. 🔍 핵심 주장 및 주요 기여 (간결 요약)

이 논문은 사전 분류된 샘플(preclassified samples)의 수를 줄이고 규칙의 성능을 향상시키기 위해 **편집(editing) 절차**를 사용하는 최근접 이웃 규칙의 **수렴 특성(convergence properties)**을 이론적으로 규명합니다.

핵심 기여는 세 가지입니다:

1. **Wilson Editing (ENN) 알고리즘 제안**: 3-NN 규칙으로 오분류된 샘플을 제거한 뒤 1-NN 분류기를 적용하는 방식으로, 1972년 Dennis Wilson이 최초로 제안하였습니다.

2. **베이즈 위험 근접 수렴 증명**: 3-NN 규칙으로 사전 분류 샘플을 편집한 후, 나머지 샘플에 1-NN 규칙을 적용하면, 소수의 사전 분류 샘플만으로도 베이즈 위험(Bayes' risk)에 매우 근접하는 결정 절차를 도출할 수 있음을 보였습니다.

3. **점근적 위험(Asymptotic Risk) 계산**: 여러 문제에 대해 일반 NN 규칙과 편집 데이터를 사용한 NN 규칙의 점근적 위험을 계산하여 비교하였습니다.

---

## 2. 📐 문제 정의, 제안 방법(수식 포함), 모델 구조, 성능 향상 및 한계

### 2-1. 해결하고자 하는 문제

**기존 kNN의 문제점:**

- 1-NN 규칙은 Cover와 Hart(1967)에 의해, 점근적 위험이 최적 베이즈 분류기의 위험의 **최대 2배**에 상한됨이 증명되었습니다.

- 즉, 이론적으로 다음이 성립합니다:

$$R_{1\text{-NN}} \leq 2 R^* \left(1 - R^*\right)$$

여기서 $R^*$는 베이즈 위험(Bayes risk)입니다. 이는 1-NN이 베이즈 최적에 수렴하지 못할 수 있음을 의미합니다.

- 실제 데이터에는 **노이즈 샘플(noisy samples)**, **경계 근처의 모호한 샘플**, **이상치(outlier)**가 포함되어, 분류 결정 경계를 왜곡합니다.

- 대규모 훈련 데이터를 그대로 사용할 경우 **저장 비용**과 **계산 비용**이 기하급수적으로 증가합니다.

Wilson은 이 문제를 해결하기 위해 **"편집(editing)"** 이라는 전처리 과정을 제안하였습니다.

---

### 2-2. 제안 방법: Wilson Editing (ENN 알고리즘)

**편집 절차의 핵심 아이디어:**

> 훈련 데이터에서, 자신의 $k$개의 이웃들이 다수결로 정확히 분류하지 못하는 샘플을 **제거(삭제)**하여 깨끗하고 대표적인 데이터 집합을 만든다.

**알고리즘 (Wilson ENN):**

1. 훈련 집합 $T = \{(\mathbf{x}_1, y_1), \ldots, (\mathbf{x}_n, y_n)\}$에서 각 샘플 $\mathbf{x}_i$에 대해:
2. $\mathbf{x}_i$를 제외한 나머지 훈련 데이터로부터 $k$개의 최근접 이웃(Wilson의 원 논문에서 $k=3$)을 찾는다.
3. 이웃들의 다수결(majority vote)로 $\mathbf{x}_i$를 분류:

$$\hat{y}_i = \arg\max_{c} \sum_{j \in \mathcal{N}_k(\mathbf{x}_i)} \mathbf{1}[y_j = c]$$

4. 만약 $\hat{y}_i \neq y_i$이면 (즉, 오분류되면), $\mathbf{x}_i$를 훈련 집합에서 **제거**:

$$T' = T \setminus \{(\mathbf{x}_i, y_i) : \hat{y}_i \neq y_i\}$$

5. 편집된 집합 $T'$에서 **1-NN 규칙**으로 최종 분류를 수행.

**점근적 위험 분석:**

Wilson은 편집된 데이터를 사용하는 1-NN의 점근적 오류율이 다음과 같은 관계를 따름을 보였습니다:

$$R_{\text{ENN}} \leq R_{\text{1-NN}} \approx 2R^* - \frac{c \cdot R^*}{d+2}$$

여기서 $d$는 특징 공간의 차원, $R^*$는 베이즈 위험입니다. 편집 후에는 경계 잡음이 제거되어 실질적으로 $R_{\text{ENN}} < R_{\text{1-NN}}$이 달성됩니다.

Wilson은 편집된 최근접 이웃 규칙의 점근적 오류 확률이 일반 $k$-NN 규칙의 오류보다 **우수함**을 이론적으로 증명하였습니다.

---

### 2-3. 모델 구조

```
원본 훈련 데이터 T = {(x₁, y₁), ..., (xₙ, yₙ)}
        ↓
[편집 단계: 3-NN으로 오분류 샘플 탐지 및 제거]
        ↓
편집된 훈련 집합 T' ⊂ T (노이즈, 경계 모호 샘플 제거)
        ↓
[분류 단계: 1-NN 규칙 적용 → 테스트 샘플 분류]
        ↓
최종 분류 결과 (위험 ≈ Bayes Risk)
```

구조의 특징:
- **두 단계 파이프라인**: 편집(3-NN) → 분류(1-NN)
- **인스턴스 제거 기반**: 새로운 샘플 생성이 아닌 노이즈 제거
- **비모수적(non-parametric)**: 사전 분포 가정 불필요

---

### 2-4. 성능 향상

3-NN을 이용한 편집 후 1-NN 규칙으로 분류하면 적은 수의 사전 분류 샘플만으로도 베이즈 위험에 매우 근접하는 결정 절차를 만들 수 있으며, 여러 문제에 걸쳐 편집 전후의 점근적 위험이 계산·비교되었습니다.

- **분류 경계 정제**: 결정 경계 근방의 모호한 샘플 제거 → 결정 경계의 품질 향상
- **저장 효율성**: 훈련 집합 크기 감소 → 계산 비용 절감
- **노이즈 내성**: 레이블 노이즈가 포함된 샘플 제거

### 2-5. 한계

| 한계 | 설명 |
|------|------|
| **과도한 샘플 제거** | ENN의 공격적인 인스턴스 제거는 일부 경우에 중요한 경계 정보를 손실시키는 부작용을 유발할 수 있습니다. |
| **불균형 데이터** | 소수 클래스의 샘플이 다수 클래스 이웃에 둘러싸여 제거될 위험 |
| **차원의 저주** | 고차원에서 최근접 이웃 거리의 신뢰성 저하 |
| **계산 비용** | 편집 단계 자체도 $O(n^2)$ 계산 필요 |
| **정적 편집** | 단일 편집 패스(1회 적용)로 반복적 노이즈 제거의 한계 |

---

## 3. 🎯 모델의 일반화 성능 향상 가능성

### 3-1. 편집이 일반화에 기여하는 메커니즘

일반화 성능은 다음 측면에서 분석할 수 있습니다:

**① 편향-분산 트레이드오프 관점:**

$$\text{Expected Error} = \text{Bias}^2 + \text{Variance} + \text{Noise}$$

ENN 편집은 다음을 달성합니다:
- **노이즈 항 감소**: 레이블 노이즈 샘플 제거
- **분산 감소**: 경계 근처 불안정 샘플 제거로 결정 경계 안정화
- **편향 약간 증가 가능**: 일부 정보 손실의 부작용

**② 베이즈 위험 수렴 관점:**

Penrod과 Wagner는 ENN 분류기의 정확도가 샘플 수가 무한대에 접근함에 따라 베이즈 오류(Bayes error)로 수렴함을 증명하였습니다.

이를 수식으로 표현하면:

$$\lim_{n \to \infty} R_{\text{ENN}}(n) = R^*$$

여기서 $R^*$는 베이즈 위험(이론적 최소 오류율)입니다.

**③ 결정 경계 스무딩 효과:**

Repeated ENN(RENN)은 모든 남은 인스턴스가 이웃의 다수를 같은 클래스로 갖게 될 때까지 ENN 알고리즘을 반복 적용하며, 이는 클래스 간 경계를 넓히고 결정 경계를 더욱 부드럽게 만들어 줍니다.

**④ 불균형 데이터 일반화:**

SMOTE와 ENN을 결합한 하이브리드 데이터 샘플링 방식은 앙상블 모델(RF, KNN, AdaBoost 등)의 성능을 크게 향상시킬 수 있음이 실증적으로 확인되었습니다.

### 3-2. 일반화 성능 향상을 위한 ENN 기반 파생 방법

| 방법 | 설명 |
|------|------|
| **Repeated ENN (RENN)** | ENN을 반복 적용하여 경계 정제 심화 |
| **All k-NN (ANN)** | $k = 1, 2, \ldots, K$ 모든 $k$에서 오분류 샘플 제거 |
| **SMOTE-ENN** | 소수 클래스 오버샘플링 + ENN 언더샘플링 결합 |

---

## 4. 🔭 앞으로의 연구에 미치는 영향 및 고려 사항

### 4-1. 연구사적 영향

Wilson의 ENN은 현대 머신러닝의 여러 방향으로 영향을 미쳤습니다:

- **데이터 전처리 패러다임 확립**: "좋은 데이터 > 복잡한 모델"의 선구적 사례
- 데이터 편집은 훈련 인스턴스의 적절한 축소 부분집합을 선택하여 k-NN에 적용하는 인스턴스 선택 기법의 일종으로 자리잡았으며, 데이터 응축(condensing)과 더불어 핵심 인스턴스 선택 기술로 발전하였습니다.
- **신경망과의 결합**: 편집 기반 최근접 이웃 규칙을 다양한 신경망 구조에 적용하여 분류 정확도와 일반화를 향상시키는 연구들이 이어졌습니다.

### 4-2. 2020년 이후 최신 연구 비교 분석

#### ① Enhanced Nearest Neighbor for Crowdsourcing (2022, arXiv)
이 연구는 ENN 분류기의 후회(regret)에 대한 점근적 전개 형식을 도출하였으며, 이는 Samworth(2012)의 결과에서 비자명한 확장입니다.

신중하게 선택된 가중치를 사용할 경우, ENN의 후회는 수렴 속도와 승수 상수 모두에서 "오라클" 최적 가중 최근접 이웃(OWNN)과 동일한 최적 후회를 달성할 수 있음이 증명되었습니다.

#### ② STEM Rebalance (2023, arXiv)
SMOTE-ENN과 Mixup을 인스턴스 수준에서 결합한 STEM 방법은 소수 클래스의 전체 분포를 효과적으로 활용하여 클래스 간 및 클래스 내 불균형을 동시에 완화합니다.

유방암 데이터셋(DDSM, Wisconsin)에서 AUC 0.96 및 0.99를 달성하며 우수한 성능을 보였습니다.

#### ③ Fast and Bayes-consistent NN (2020, arXiv)
빠른 평가 시간을 유지하면서 베이즈 일관성을 달성하기 위해 LSH(Locality-Sensitive Hashing)와 새로운 missing-mass 논증을 결합한 빠르고 베이즈 일관성 있는 분류기를 제안하였습니다.

#### ④ SMOTE-ENN 기반 금융 분류 (2022~)
XGBoost 계열 방법과 Borderline-SMOTE + ENN 샘플링 기법을 결합한 Tri-XGBoost가 기업 부도 예측 등 금융 불균형 문제에 활발히 적용되고 있습니다.

| 연구 | 연도 | Wilson ENN과의 관계 | 주요 기여 |
|------|------|-------------------|-----------|
| Enhanced NN for Crowdsourcing | 2022 | ENN 점근 분석 확장 | 크라우드소싱 환경에서 최적 수렴 |
| STEM Rebalance | 2023 | SMOTE-ENN + Mixup | 의료 이미징 불균형 처리 |
| Fast Bayes-consistent NN | 2020 | 베이즈 일관성 + 속도 | LSH 기반 고속 일관성 분류기 |
| Tri-XGBoost + ENN | 2022 | ENN 언더샘플링 응용 | 금융 데이터 불균형 분류 |

### 4-3. 앞으로 연구 시 고려할 점

1. **고차원·딥러닝 환경에서의 ENN 재정의**
   - 원시 특징 공간이 아닌, **딥러닝 임베딩 공간**에서의 편집 적용 가능성 탐색
   - $k$-NN 기반 그래프 구조를 활용한 GNN과의 결합

2. **반지도 학습(Semi-supervised Learning)과의 결합**
   - 비레이블 데이터의 도움으로 ENN을 포함한 세 가지 편집 기법 모두의 성능이 향상되었음이 확인되었으므로, 레이블이 부족한 실세계 환경에서 반지도 편집 기법 발전이 유망합니다.

3. **동적 편집(Dynamic Editing)**
   - 정적 1회 편집이 아닌, 온라인/증분 학습 환경에서 **스트리밍 데이터에 실시간 편집** 적용

4. **클래스 불균형 전용 편집 전략**
   - ENN은 다른 언더샘플링 방법과 결합할 때 최상의 결과를 냅니다. 이를 체계화하여 클래스 비율 인식형 편집 알고리즘 개발 필요

5. **설명가능성(XAI)과의 연계**
   - 어떤 샘플이 왜 제거되었는지에 대한 **해석 가능한 편집 기준** 제시

6. **계산 효율화**
   - 원본 $O(n^2)$ 편집 비용을 줄이기 위한 근사 최근접 이웃(ANN) 기반 편집 알고리즘 설계

---

## 📚 참고 자료

| 번호 | 출처 |
|------|------|
| [1] | **Wilson, D. L. (1972).** Asymptotic Properties of Nearest Neighbor Rules Using Edited Data. *IEEE Transactions on Systems, Man, and Cybernetics*, 2(3), 408–421. DOI: 10.1109/TSMC.1972.4309137 |
| [2] | **IEEE Xplore 원문** — https://ieeexplore.ieee.org/document/4309137/ |
| [3] | **Semantic Scholar 논문 페이지** — https://www.semanticscholar.org/paper/Asymptotic-Properties-of-Nearest-Neighbor-Rules-Wilson/dea8658ee4750ec6bb408a2281cf922cbb300a0a |
| [4] | **Guan, D. et al.** — Nearest neighbor editing aided by unlabeled data. *Neurocomputing*, 2011. http://uclab.khu.ac.kr/resources/publication/J_76.pdf |
| [5] | **Hasan, Y. et al. (2023).** STEM Rebalance: A Novel Approach for Tackling Imbalanced Datasets using SMOTE, Edited Nearest Neighbour, and Mixup. arXiv:2311.07504 |
| [6] | **Enhanced Nearest Neighbor Classification for Crowdsourcing (2022).** arXiv:2203.00781 |
| [7] | **Efremenko, K. et al. (2020).** Fast and Bayes-consistent nearest neighbors. arXiv:1910.05270 |
| [8] | **Sun, J. et al. (2021).** A Survey of k Nearest Neighbor Algorithms for Solving the Class Imbalanced Problem. *Wireless Communications and Mobile Computing*. https://onlinelibrary.wiley.com/doi/10.1155/2021/5520990 |
| [9] | **imbalanced-learn 공식 문서** — EditedNearestNeighbours. https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.EditedNearestNeighbours.html |
| [10] | **MachineLearningMastery.com (2021)** — Undersampling Algorithms for Imbalanced Classification. https://machinelearningmastery.com/undersampling-algorithms-for-imbalanced-classification/ |
| [11] | **Ferri, F. et al.** — Sensitivity of k-NN editing. *IEEE Trans. Syst., Man, Cybern.* https://sci2s.ugr.es/keel/pdf/specific/articulo/Ferri99SensitNearNeigRule.pdf |

> ⚠️ **정확도 안내**: 본 분석은 공개된 초록, 인용 자료 및 관련 연구에 기반합니다. Wilson(1972) 원문 전체에 직접 접근하지 못한 수식 일부(예: 점근적 위험의 정확한 폐쇄형 표현)는 문헌에서 인용된 내용을 토대로 재구성하였으며, 원문 구독을 통한 검증을 권장합니다.
