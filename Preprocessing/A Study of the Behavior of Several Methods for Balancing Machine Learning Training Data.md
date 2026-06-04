# A Study of the Behavior of Several Methods for Balancing Machine Learning Training Data

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

Batista et al. (2004)의 본 논문은 **클래스 불균형(class imbalance) 문제가 단독으로 분류 성능을 저하시키지 않는다**는 것을 실험적으로 입증합니다. 핵심 주장은 다음 세 가지입니다:

1. **클래스 불균형 자체보다 소수 클래스 샘플 부족 + 클래스 중첩(class overlapping)의 복합 작용이 성능 저하의 실질적 원인**이다.
2. **오버샘플링(over-sampling) 방법이 언더샘플링(under-sampling) 방법보다 AUC 기준으로 더 우수한 성능**을 보인다.
3. 새롭게 제안된 **SMOTE + Tomek Links** 및 **SMOTE + ENN**은 소수 클래스 샘플이 적은 데이터셋에서 특히 효과적이다.

### 주요 기여

| 기여 항목 | 내용 |
|---|---|
| 실험적 검증 | 13개 UCI 데이터셋, 10개 방법 비교 |
| 신규 제안 방법 | CNN + Tomek, SMOTE + Tomek, SMOTE + ENN |
| 평가 지표 | 정확도 대신 AUC(ROC 곡선 하 면적) 채택 |
| 모델 복잡도 분석 | 유도된 규칙 수, 규칙당 조건 수 측정 |
| 반직관적 발견 | Random over-sampling이 복잡한 방법과 경쟁적 성능 달성 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**클래스 불균형 문제(Class Imbalance Problem)**:
- 다수 클래스(negative) 샘플이 소수 클래스(positive) 샘플을 압도하는 상황
- 의료 희귀 질환 진단, 결함 모니터링 등 실세계 응용에서 빈번히 발생
- 기존 ML 알고리즘은 전체 정확도를 최대화하려 하므로 소수 클래스를 무시하는 경향

**기존 평가지표의 한계**:

$$\text{Accuracy} = \frac{TP + TN}{TP + FN + FP + TN}$$

$$\text{Error Rate} = \frac{FP + FN}{TP + FN + FP + TN}$$

이 두 지표는 클래스 불균형 상황에서 다수 클래스에 편향되어 **misleading한 결론**을 도출할 수 있습니다. 예를 들어, 다수 클래스 비율이 99%인 도메인에서 모든 샘플을 다수 클래스로 예측해도 정확도 99%를 달성합니다.

### 2.2 평가 지표: AUC와 ROC 분석

논문은 클래스 비율과 독립적인 성능 지표를 사용합니다:

$$FN_{rate} = \frac{FN}{TP + FN}$$

$$FP_{rate} = \frac{FP}{FP + TN}$$

$$TN_{rate} = \frac{TN}{FP + TN}$$

$$TP_{rate} = \frac{TP}{TP + FN}$$

**AUC(Area Under the ROC Curve)**를 주요 평가 지표로 채택. AUC는 Wilcoxon rank 검정과 동치이며, 클래스 분포 변화에 강건(robust)합니다.

### 2.3 제안 방법 및 수식

#### (1) k-NN 분류 수식 (방법들의 핵심 구성요소)

$$\mathbf{h}(E_q) = \arg\max_{c \in C} \sum_{i=1}^{k} \omega_i \delta(c, f(\hat{E}_i)), \quad \omega_i = \frac{1}{d(E_q, \hat{E}_i)^2} \tag{1}$$

여기서 $\delta(a, b) = 1$ if $a = b$, otherwise $\delta(a, b) = 0$

거리 함수로는 **HVDM(Heterogeneous Value Difference Metric)**을 사용:
- 수치형 속성: 유클리드 거리
- 범주형 속성: VDM(Value Difference Metric)

#### (2) Tomek Links 정의

두 샘플 $E_i$, $E_j$가 서로 다른 클래스에 속할 때, $(E_i, E_j)$ 쌍이 Tomek Link인 조건:

$$\nexists \, E_l : d(E_i, E_l) < d(E_i, E_j) \text{ or } d(E_j, E_l) < d(E_i, E_j)$$

Tomek Link를 형성하는 샘플은 **노이즈(noise)** 또는 **경계선 샘플(borderline)**입니다.

#### (3) 제안 방법들

**① CNN + Tomek Links** (신규 제안)
- OSS(One-Sided Selection)와 유사하나 순서가 반대
- CNN 적용 → 데이터 축소 → Tomek Links 제거 (계산 효율 향상 목적)

**② SMOTE + Tomek Links** (신규 제안)

$$\tilde{x} = x_i + \lambda \cdot (x_{nn} - x_i), \quad \lambda \in [0, 1] \tag{SMOTE 보간}$$

적용 순서:
1. SMOTE로 소수 클래스 합성 샘플 생성 → 클래스 균형 달성
2. 오버샘플된 데이터셋에 Tomek Links 적용 → **양 클래스**의 노이즈/경계 샘플 제거

**③ SMOTE + ENN** (신규 제안)
- SMOTE 적용 후 Wilson's ENN(Edited Nearest Neighbor) 규칙 적용
- ENN: 3개의 최근접 이웃 중 2개 이상이 다른 클래스로 분류하는 샘플 제거
- Tomek Links보다 더 공격적인 데이터 정제 효과

### 2.4 실험 설계

| 항목 | 내용 |
|---|---|
| 데이터셋 | 13개 UCI 데이터셋 (불균형 비율 2.5% ~ 35%) |
| 분류기 | C4.5 (pruned & unpruned) |
| 검증 방법 | 10-fold cross-validation |
| 통계 검정 | Hsu's MCB test (95% 신뢰수준) |
| 비교 방법 수 | 10개 (Random Over/Under, Tomek, CNN, OSS, NCL, SMOTE, + 3개 신규 제안) |

### 2.5 주요 실험 결과

**AUC 기준 오버샘플링 vs 언더샘플링:**

> 전반적으로 오버샘플링 방법이 언더샘플링보다 높은 AUC를 달성

이는 Drummond & Holte (2003)의 "언더샘플링이 오버샘플링을 능가한다"는 기존 주장과 **상반되는 결과**입니다.

**소수 클래스 샘플 수 < 100인 데이터셋 (Flag, Glass, Post-operative, New-thyroid, E.Coli, Haberman):**
- SMOTE + Tomek 또는 SMOTE + ENN이 6개 중 최소 5개 데이터셋에서 의미 있는 결과 달성

**모델 복잡도 (unpruned 결정 트리):**

| 방법 | 규칙 수 증가 | 조건 수 증가 |
|---|---|---|
| Random Over-sampling | 가장 적은 증가 | 중간 |
| SMOTE + ENN | 중간 | **가장 적은 증가** |
| SMOTE | 가장 큰 증가 | 큰 증가 |

---

## 3. 모델 일반화 성능 향상 가능성

### 3.1 일반화 성능 저하의 근본 원인 규명

논문은 일반화 성능 저하가 **클래스 불균형 + 클래스 중첩의 복합 효과**에서 비롯됨을 실증합니다.

$$\text{성능 저하} \propto f(\underbrace{\text{클래스 불균형}}_{\text{필요조건}} \times \underbrace{\text{클래스 중첩}}_{\text{충분조건}})$$

실험 결과(Figure 3)에서 Letter-a(불균형 비율 24.4:1)와 Nursery(38.2:1)가 거의 100% AUC를 달성한 것은, **클래스가 잘 분리되어 있으면 심각한 불균형에도 일반화 성능이 유지됨**을 보여줍니다.

### 3.2 SMOTE + 데이터 정제의 일반화 메커니즘

**SMOTE + Tomek / SMOTE + ENN이 일반화를 향상시키는 메커니즘:**

```
① SMOTE로 소수 클래스 경계 확장
      ↓
② 그러나 합성 샘플이 다수 클래스 공간을 침범할 위험
      ↓
③ Tomek/ENN으로 결정 경계 근방의 노이즈/애매한 샘플 제거
      ↓
④ 더 명확하게 정의된 클래스 클러스터 생성
      ↓
⑤ 단순하고 일반화 능력이 높은 모델 유도 가능
```

논문은 이를 다음과 같이 명시합니다:
> *"The removal of noisy examples might aid in finding better-defined class clusters, therefore, allowing the creation of simpler models with better generalization capabilities."*

### 3.3 Pruning 분석과 일반화

**Figure 5** 분석 결과, 원본 및 균형 데이터셋 모두에서 **pruning이 AUC 향상에 거의 기여하지 않음**이 확인됩니다.

이는 다음 논리로 설명됩니다:
- C4.5의 pruning은 전체 오류율 최소화 목적 → 다수 클래스에 편향
- 인위적으로 균형 잡힌 데이터에서 pruning은 **훈련/테스트 분포 불일치 가정**을 만들어냄

$$\text{Pruning 조건}: P(\text{test}) \approx P(\text{train}) \quad \text{→ 균형 데이터에서 위반됨}$$

따라서 **unpruned 트리 + 오버샘플링 조합이 일반화 성능 향상에 더 적합**합니다.

### 3.4 SMOTE + ENN의 조건 수 감소 효과

Table 9에서 SMOTE + ENN은 **10개 데이터셋에서 규칙당 조건 수 최소화**, 6개 데이터셋에서 원본 데이터 대비 조건 수를 줄이는 성과를 달성했습니다.

$$\bar{c}_{SMOTE+ENN} < \bar{c}_{Original} \quad \text{(6개 데이터셋에서)}$$

이는 오버샘플링 후 ENN 정제가 **결정 트리를 단순화하여 과적합을 억제**하는 효과를 보임을 의미합니다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려 사항

### 4.1 연구에 미치는 영향

#### ① 불균형 학습 연구의 방향 전환
- 클래스 불균형을 단순 비율 문제가 아닌 **데이터 기하학적 구조(class overlapping, subclusters) 문제**로 재정의
- 이후 연구들이 "데이터 복잡도(data complexity)"를 핵심 변수로 포함하는 계기 제공

#### ② 하이브리드 샘플링의 표준화
- SMOTE + 데이터 정제 방법의 조합이 이후 연구의 표준 베이스라인으로 자리잡음
- imbalanced-learn 등 실용 라이브러리의 기본 방법론으로 채택

#### ③ 평가 지표 패러다임 전환
- 정확도 대신 AUC, G-mean, F1-score 등 불균형에 강건한 지표 사용의 중요성 부각

#### ④ 데이터 중심 AI(Data-Centric AI)의 선구
- 알고리즘 개선보다 **데이터 품질 개선(노이즈 제거, 경계 정제)**이 성능에 미치는 영향 강조

### 4.2 앞으로 연구 시 고려할 점

#### ① 최적 클래스 비율 탐색
논문 자체에서 미해결 과제로 제시:
> *"allocating half of the training examples to the minority class does not always provide optimal results"*

향후 연구는 **데이터셋별 최적 균형 비율을 자동 탐색**하는 방법론이 필요합니다.

#### ② 단일 분류기 의존성
- C4.5만으로 실험 → SVM, Random Forest, Neural Network 등 다양한 분류기에서의 일반성 검증 필요
- 특히 앙상블 방법(Bagging, Boosting)과의 결합 효과 분석 필요

#### ③ 다중 클래스 불균형으로 확장
- 이진 분류에 국한된 실험 설계
- 다중 클래스 불균형(multi-class imbalance) 상황에서의 적용성 검증 필요

#### ④ 데이터 복잡도 측정의 정량화
- 클래스 중첩 정도를 정량적으로 측정하는 지표 개발
- 데이터 복잡도에 따른 방법 선택 가이드라인 구축

#### ⑤ 고차원 및 대규모 데이터 적용
- 13개 소규모 UCI 데이터셋에 한정
- 고차원(curse of dimensionality)에서 k-NN 기반 방법의 효율성 저하 문제 해결 필요

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 주요 최신 연구 동향

#### ① 딥러닝 기반 오버샘플링

**CTGAN / TVAE (Xu et al., 2019~2020대)**
- GAN(Generative Adversarial Network)을 활용한 합성 데이터 생성
- SMOTE의 선형 보간 한계를 극복하여 복잡한 분포 학습 가능

$$\min_G \max_D \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

**SMOTE 대비 장점**: 고차원 데이터에서 더 현실적인 합성 샘플 생성  
**SMOTE 대비 단점**: 계산 비용 증가, 학습 불안정성

#### ② 정보 이론 기반 방법

**ADASYN (He et al., 2008 → 2020년대 개선 연구들)**
- 학습이 어려운 소수 클래스 샘플에 더 많은 합성 샘플 생성
- 밀도 분포를 고려한 적응적 오버샘플링

$$\hat{r}_i = \frac{\Delta_i / K}{Z}, \quad G_i = \hat{r}_i \times G$$

여기서 $\Delta_i$는 $K$개 이웃 중 다수 클래스 샘플 수, $G$는 총 생성 샘플 수

#### ③ 앙상블 + 불균형 처리 결합

**BalancedBaggingClassifier / EasyEnsemble (2020년대 활발 연구)**
- 언더샘플링 + 부스팅/배깅 결합
- Batista et al.의 발견(언더샘플링 단독 한계)을 보완하는 방향

**RUSBoost (Seiffert et al.)**:
$$H(x) = \text{sign}\left(\sum_{t=1}^{T} \alpha_t h_t(x)\right)$$
랜덤 언더샘플링과 AdaBoost 결합

#### ④ 비용 민감 학습(Cost-Sensitive Learning)

**Focal Loss (Lin et al., 2017 → 2020년대 광범위 적용)**
$$FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

분류가 쉬운 샘플의 손실 가중치를 줄여 어려운 소수 클래스 학습에 집중

### 5.2 Batista et al. (2004) vs 최신 연구 비교표

| 비교 항목 | Batista et al. (2004) | 최신 연구 (2020년 이후) |
|---|---|---|
| **핵심 접근** | 샘플링 기반 데이터 전처리 | 샘플링 + 모델 수준 결합 |
| **소수 클래스 생성** | SMOTE 선형 보간 | GAN, VAE 기반 생성 |
| **데이터 정제** | Tomek Links, ENN | 학습 기반 노이즈 감지 |
| **분류기** | C4.5 단독 | 딥러닝, 앙상블 다양화 |
| **평가 데이터** | 13개 소규모 UCI | 대규모, 고차원, 스트림 데이터 |
| **평가 지표** | AUC 중심 | AUC, F1, G-mean, MCC 복합 |
| **불균형 비율** | 최대 ~38:1 | 100:1 이상 극단적 불균형 연구 |
| **이론적 기반** | 경험적 실험 중심 | 이론적 수렴 보장 연구 병행 |

### 5.3 지속되는 영향력

Batista et al. (2004)의 **SMOTE + 데이터 정제 패러다임**은 2020년대에도 다음 형태로 이어집니다:

- **imbalanced-learn 라이브러리**: `SMOTETomek`, `SMOTEENN` 클래스로 직접 구현 제공
- **AutoML 파이프라인**: 불균형 처리 단계에서 SMOTE 계열 방법이 기본 후보로 포함
- **의료 AI, 사기 탐지, 이상 탐지** 등 실응용 분야에서 기준 방법으로 지속 사용

---

## 참고자료

### 논문 원문
- **Batista, G. E. A. P. A., Prati, R. C., & Monard, M. C. (2004).** "A Study of the Behavior of Several Methods for Balancing Machine Learning Training Data." *SIGKDD Explorations Newsletter*, 6(1), 20-29.

### 논문 내 참조 문헌 (직접 인용)
- Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). "SMOTE: Synthetic Minority Over-sampling Technique." *JAIR 16*, 321–357.
- Tomek, I. (1976). "Two Modifications of CNN." *IEEE Transactions on Systems Man and Communications SMC-6*, 769–772.
- Wilson, D. L. (1972). "Asymptotic Properties of Nearest Neighbor Rules Using Edited Data." *IEEE Transactions on Systems, Man, and Communications 2*(3), 408–421.
- Kubat, M., & Matwin, S. (1997). "Addressing the Course of Imbalanced Training Sets: One-sided Selection." *ICML*, 179–186.
- Prati, R. C., Batista, G. E. A. P. A., & Monard, M. C. (2004). "Class Imbalances versus Class Overlapping." *MICAI*, LNAI 2972, 312–321.
- Drummond, C., & Holte, R. C. (2003). "C4.5, Class Imbalance, and Cost Sensitivity: Why Under-sampling beats Over-sampling." *Workshop on Learning from Imbalanced Data Sets II*.
- Weiss, G. M., & Provost, F. (2003). "Learning When Training Data are Costly: The Effect of Class Distribution on Tree Induction." *JAIR 19*, 315–354.
- Laurikkala, J. (2001). "Improving Identification of Difficult Small Classes by Balancing Class Distribution." Tech. Rep. A-2001-2, University of Tampere.

### 최신 관련 연구 참조
- He, H., Bai, Y., Garcia, E. A., & Li, S. (2008). "ADASYN: Adaptive Synthetic Sampling Approach for Imbalanced Learning." *IJCNN*.
- Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). "Focal Loss for Dense Object Detection." *ICCV*.
- Xu, L., Skoularidou, M., Cuesta-Infante, A., & Veeramachaneni, K. (2019). "Modeling Tabular data using Conditional GAN." *NeurIPS*.
- **imbalanced-learn 라이브러리**: https://imbalanced-learn.org/stable/ (SMOTE+Tomek, SMOTEENN 구현 참조)
- Fernández, A., García, S., Galar, M., Prati, R. C., Krawczyk, B., & Herrera, F. (2018). *Learning from Imbalanced Data Sets*. Springer. (Batista et al. 방법의 체계적 정리 포함)
