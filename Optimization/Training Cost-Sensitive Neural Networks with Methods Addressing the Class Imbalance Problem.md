# Training Cost-Sensitive Neural Networks with Methods Addressing the Class Imbalance Problem

**저자:** Zhi-Hua Zhou & Xu-Ying Liu  
**출판:** IEEE Transactions on Knowledge and Data Engineering (2006)

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장
이 논문은 클래스 불균형 문제(class imbalance problem)를 해결하기 위해 개발된 기법들(오버샘플링, 언더샘플링, Threshold-Moving, 앙상블)을 **비용 민감 신경망(cost-sensitive neural network)** 학습에 적용하여 실증적으로 분석한 연구입니다.

### 주요 기여
| 기여 항목 | 내용 |
|-----------|------|
| **실증적 비교 분석** | 21개 UCI 데이터셋 + KDD-99 실세계 데이터셋에서 체계적 실험 |
| **핵심 발견** | 이진 분류에서 유효한 방법들이 다중 분류에서는 부정적 효과 유발 |
| **방법론 권고** | Threshold-Moving과 Soft-Ensemble이 전반적으로 최선의 선택 |
| **이론적 시사점** | 클래스 불균형 학습과 비용 민감 학습이 서로 다른 특성을 가질 수 있음을 제시 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**현실적 배경:**  
- 실세계에서 오분류 비용(misclassification cost)은 클래스마다 다름  
  - 예: 의료 진단에서 환자를 건강하다고 오진하는 비용 >> 건강한 사람을 환자로 오진하는 비용
- 기존 신경망은 오류 수 최소화에 집중(cost-blind), 비용 불균등 무시
- 의사결정 트리용 비용 민감 기법(예: instance-weighting)을 신경망에 직접 적용 불가
- 클래스 불균형 해결 기법이 비용 민감 학습에도 유효한지 미검증

**연구 질문:**
> 클래스 불균형 문제에 효과적인 방법들이 비용 민감 학습에도 도움이 되는가?

---

### 2.2 제안하는 방법 및 수식

#### 공통 표기

$C$: 클래스 수, $N_i$: $i$번째 클래스의 학습 예제 수

$$Cost[i] = \sum_{c=1}^{C} Cost[i, c]$$

클래스 정렬 조건: $i < j$이면 $Cost[i] < Cost[j]$ 또는 $(Cost[i] = Cost[j]$ and $N_i \geq N_j)$

---

#### (A) Over-Sampling (오버샘플링)

비용에 비례하여 높은 비용 클래스의 샘플을 복제

재샘플링 후 $k$번째 클래스의 목표 샘플 수:

$$N_k^* = \left\lfloor \frac{Cost[k]}{Cost[\lambda]} N_\lambda \right\rfloor \tag{1}$$

기준 클래스 $\lambda$ 식별 (복제 최소화 클래스):

$$\lambda = \arg\min_j \frac{\frac{Cost[j]}{\min_c Cost[c]} N_{\arg\min_c Cost[c]}}{N_j} \tag{2}$$

- $N_k^* > N_k$이면 $(N_k^* - N_k)$개 샘플을 복원추출(random sampling with replacement)
- **SMOTE** 변형도 실험: 소수 클래스 샘플 간 선분 위의 합성 예제 생성

---

#### (B) Under-Sampling (언더샘플링)

비싼 클래스는 유지하고 저비용 클래스 샘플을 제거

기준 클래스 $\lambda$ 식별 (제거 최소화 클래스):

$$\lambda = \arg\max_j \frac{\frac{Cost[j]}{\max_c Cost[c]} N_{\arg\max_c Cost[c]}}{N_j} \tag{3}$$

- 불필요 샘플 탐지: **1-NN 규칙**으로 중복 예제(redundant examples) 제거
- 경계 샘플 탐지: **Tomek Link** 방법 활용

거리 계산:
$$\text{Dist}(\mathbf{x_1}, \mathbf{x_2}) = \sqrt{\sum_{l=1}^{j} VDM(x_{1l}, x_{2l}) + \sum_{l=j+1}^{d} |x_{1l} - x_{2l}|^2} \tag{4}$$

VDM (Value Difference Metric):
$$VDM(u, v) = \sum_{c=1}^{C} \left| \frac{N_{a,u,c}}{N_{a,u}} - \frac{N_{a,v,c}}{N_{a,v}} \right|^2 \tag{5}$$

---

#### (C) Threshold-Moving (임계값 이동)

신경망 구조 변경 없이, **테스트 단계에서** 출력값을 비용 행렬로 조정

표준 분류: $\arg\max_i O_i$ → 비용 민감 분류: $\arg\max_i O_i^*$

$$O_i^* = \eta \sum_{c=1}^{C} O_i \cdot Cost[i, c] \tag{6}$$

여기서 $\eta$는 $\sum_{i=1}^{C} O_i^* = 1$, $0 \leq O_i^* \leq 1$을 만족하는 정규화 상수

> **핵심 장점:** 훈련 데이터 및 신경망 구조를 변경하지 않고, 출력 후처리만으로 비용 민감 분류 달성

---

#### (D) Hard-Ensemble & Soft-Ensemble

세 개의 신경망(NN1: 오버샘플링, NN2: 언더샘플링, NN3: Threshold-Moving)을 훈련 후 결합

**Hard-Ensemble:** 각 분류기의 이진 투표(crisp decision) 결합

**Soft-Ensemble:** 정규화된 실수 출력값의 합산

$$V = \sum_i V_i \tag{소프트 앙상블 최종 결정}$$

최종 클래스 = $V$의 최대 성분에 해당하는 클래스 (동점 시 비용이 가장 큰 클래스 선택)

---

#### (E) 앙상블 다양성 측정

$Q_{av}$ 통계량 (Kuncheva & Whitaker, 2003):

$$Q_{av} = \frac{2}{L(L-1)} \sum_{i=1}^{L-1} \sum_{k=i+1}^{L} Q_{i,k} \tag{9}$$

$$Q_{i,k} = \frac{N^{11}N^{00} - N^{01}N^{10}}{N^{11}N^{00} + N^{01}N^{10}} \tag{10}$$

- $Q_{av}$가 작을수록 구성 학습기들 간 다양성이 높음

---

### 2.3 모델 구조

- **기본 신경망:** 역전파(Backpropagation, BP) 신경망
- **구조:** 입력층 - 은닉층(10 유닛) - 출력층
- **훈련:** 200 에포크
- **특징:** 모든 기법이 신경망 구조나 훈련 알고리즘 변경 없이 적용 가능

---

### 2.4 성능 향상 및 주요 실험 결과

#### 실험 설계
- 21개 UCI 데이터셋 + KDD-99 실세계 데이터
- 3종류 비용 행렬 (Type a, b, c)
- 10회 × 10-겹 교차 검증

#### 견고성 비교 기준

$$r_\alpha = \frac{cost_\alpha}{\max_i cost_i} \tag{7}$$

$r_\alpha$가 작을수록 성능이 좋음

#### 이진 분류(Two-Class) 결과

| 방법 | 평균 비용 비율(BP 대비) | 특이사항 |
|------|----------------------|---------|
| BP (기준) | 1.000 | 기준선 |
| Over-sampling | 0.888 | 효과적 |
| Under-sampling | 0.955 | 불균형 심할수록 부정적 효과 |
| **Threshold-Moving** | **0.736** | **가장 우수, 심각한 불균형에도 효과적** |
| Hard-ensemble | 0.745 | 효과적 |
| **Soft-ensemble** | **0.731** | **전반적으로 최고 성능** |
| SMOTE | 0.927 | 상대적으로 낮은 성능 |

> Euthyroid, Hypothyroid (심각한 불균형 데이터)에서 Threshold-Moving만 유효

#### 다중 분류(Multi-Class) 결과

| 방법 | Type(a) | Type(b) | Type(c) | 평가 |
|------|---------|---------|---------|------|
| Over-sampling | 0.914 | 1.011 | 1.172 | 종종 부정적 효과 |
| Under-sampling | 1.879 | 2.058 | 2.410 | 심각한 부정적 효과 |
| **Threshold-Moving** | **0.804** | **0.888** | **0.970** | **일관되게 효과적** |
| Hard-ensemble | 0.757 | 0.895 | 1.067 | 불균형 심하면 부정적 |
| **Soft-ensemble** | **0.733** | **0.826** | **0.978** | **최선의 선택** |
| SMOTE | 1.606 | 1.663 | 1.911 | 부정적 효과 빈번 |

#### 비용 행렬 난이도: Type(a) < Type(b) < Type(c)

---

### 2.5 한계점

1. **구성 학습기 수 제한:** 앙상블에 3개의 신경망만 사용
2. **아키텍처 미최적화:** 은닉층 유닛 수(10개), 에포크(200) 등 하이퍼파라미터 미조정
3. **비용 행렬 고정 가정:** 실제 환경에서 비용이 변할 수 있음
4. **다중 분류 문제에 대한 불완전한 해결:** 샘플링 기법의 한계가 명확히 드러남
5. **신경망 외 다른 분류기 미비교**

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 Soft-Ensemble의 다양성과 일반화

$Q_{av}$ 분석 결과, 다중 분류 문제에서 구성 학습기들 간 다양성이 이진 분류보다 높았습니다:

| 데이터셋 유형 | 평균 $Q_{av}$ |
|-------------|-------------|
| 이진 분류 | 0.829 |
| 다중 분류 Type(a) | 0.648 |
| 다중 분류 Type(b) | 0.675 |
| 다중 분류 Type(c) | 0.633 |
| KDD-99 | 0.245 |

**다양성과 일반화의 관계:**
$$\text{좋은 앙상블} \Leftrightarrow \text{높은 개별 성능} + \text{높은 다양성}$$

- 다중 분류에서 다양성이 높음에도 앙상블 성능이 항상 좋지 않은 이유: **개별 학습기(특히 언더샘플링)의 성능 저하**
- 일반화 향상의 핵심: 다양성(diversity)과 개별 정확도(accuracy)의 균형

### 3.2 Threshold-Moving의 일반화 우수성

- **훈련 데이터 미변경** → 오버피팅 위험 없음
- 신경망이 원래 데이터 분포에서 학습 → 더 나은 일반화
- 비용 민감성을 테스트 단계에서만 적용 → 유연성 높음

### 3.3 Soft-Ensemble의 일반화 메커니즘

$$\text{Soft-Ensemble}: V = V_1 + V_2 + V_3$$

각 구성 학습기의 예측 불확실성(실수값 출력)을 종합적으로 반영함으로써:
- 단일 학습기 대비 분산(variance) 감소
- 편향(bias)과 분산의 균형 개선

### 3.4 오버샘플링의 일반화 위험성

오버샘플링은 정확한 복사본(exact copies)을 생성하므로:
$$\text{오버피팅 위험} \uparrow \Leftrightarrow \text{일반화 성능} \downarrow$$

- 특히 심각한 불균형 데이터(euthyroid: 비율 293:2870)에서 일반화 실패

### 3.5 SMOTE의 합성 데이터 품질 문제

KDD-99처럼 클래스 불균형이 심한 경우:
- 소수 클래스 주변이 다수 클래스 샘플로 둘러싸여 있음
- 합성 샘플이 오해를 유발하는 결정 경계 생성
- → 일반화 성능 저하

### 3.6 일반화 성능 향상을 위한 권고사항 (논문 기반)

1. **Threshold-Moving 우선 적용**: 안전하고 효과적인 기준선
2. **Soft-Ensemble 추가 고려**: 데이터가 심각하게 불균형하지 않을 때
3. **Bootstrap 샘플링 확장**: 더 많은 구성 학습기로 앙상블 강화 가능
4. **다양성-정확도 균형 설계**: 단순 다양성 증가만으로는 불충분

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 연구에 미치는 영향

#### (1) 다중 클래스 비용 민감 학습의 중요성 제고
논문은 다중 분류에서의 비용 민감 학습이 이진 분류와 **질적으로 다른 문제**임을 실증하였습니다. 이후 연구들이 다중 클래스 환경에 특화된 알고리즘을 개발하는 계기를 마련하였습니다.

#### (2) Threshold-Moving의 재발견
Threshold-Moving이 오랫동안 과소평가되어 왔음을 지적하고, 이것이 클래스 불균형과 비용 민감 학습 모두에서 강력한 베이스라인임을 입증하였습니다.

#### (3) 클래스 불균형 학습 ≠ 비용 민감 학습
두 문제가 유사하지만 동일하지 않음을 실험적으로 보여줌으로써, 각 문제에 적합한 별도의 방법론 개발 필요성을 촉진하였습니다.

#### (4) 앙상블의 다양성 분석 프레임워크 제시
$Q_{av}$ 통계량을 활용한 다양성 분석 방법론을 비용 민감 학습에 도입하여, 앙상블 설계 시 다양성-정확도 균형의 중요성을 강조하였습니다.

---

### 4.2 2020년 이후 최신 연구 비교 분석

> **주의:** 아래 2020년 이후 연구들은 제공된 논문 PDF에 포함되어 있지 않으며, 저의 학습 데이터를 기반으로 기술합니다. 일부 세부 내용은 부정확할 수 있으므로, 반드시 해당 논문을 직접 확인하시길 권장합니다.

#### (A) 딥러닝 기반 비용 민감 학습으로의 발전

Zhou & Liu(2006)는 BP 신경망에 한정되었지만, 최신 연구들은 딥러닝 아키텍처로 확장되었습니다.

**예시 연구 방향:**
- **Focal Loss** (Lin et al., 2017): 클래스 불균형을 손실 함수 자체에 내재화
  $$FL(p_t) = -\alpha_t (1-p_t)^\gamma \log(p_t)$$
  여기서 $\gamma > 0$는 easy examples의 가중치를 낮추는 focusing parameter

- Zhou & Liu(2006)의 Threshold-Moving과의 비교: Focal Loss는 훈련 단계에서 비용을 통합하는 반면, Threshold-Moving은 테스트 단계에서 처리한다는 근본적인 차이 존재

#### (B) 클래스 불균형 다중 분류 특화 연구

Zhou & Liu(2006)가 다중 클래스 환경에서의 한계를 지적한 이후, 관련 연구들이 진행되었습니다:

- **LDAM (Label-Distribution-Aware Margin Loss)** (Cao et al., 2019): 클래스 빈도에 따른 마진 조정
$$\Delta_j = C \cdot n_j^{-1/4}$$
- **Balanced Softmax** (Ren et al., 2020): 클래스 빈도를 반영한 소프트맥스 보정
$$P(y=j|x) = \frac{n_j \exp(z_j)}{\sum_k n_k \exp(z_k)}$$

이는 Zhou & Liu(2006)의 Threshold-Moving과 개념적으로 유사하지만, 확률적 관점에서 더 엄밀하게 접근합니다.

#### (C) 샘플링 기법의 발전

SMOTE의 한계(경계 불안정, 노이즈 생성)를 보완한 변형들:
- **ADASYN** (He et al., 2008): 학습 난이도에 따른 적응적 샘플링
- **SVM-SMOTE, Borderline-SMOTE**: 경계 영역 집중 샘플링

Zhou & Liu(2006)가 SMOTE의 심각한 불균형 상황에서의 문제점을 지적하였고, 이후 이를 개선하는 방향으로 발전하였습니다.

#### (D) 비용 민감 앙상블 학습의 진화

- **AdaCost** (Fan et al., 1999) → 이후 비용 민감 부스팅 계열 발전
- **MetaCost** 기반 확장 연구들
- **GBSE** (Abe et al., 2004): 논문에서도 언급된 다중 클래스 비용 민감 학습

Zhou & Liu(2006)의 Soft-Ensemble 아이디어는 현대 앙상블 학습에서의 soft voting과 일맥상통하며, 비용 민감 맥락에서의 선구적 적용이라 할 수 있습니다.

#### (E) 딥러닝 시대의 비용 민감 학습 비교

| 기준 | Zhou & Liu (2006) | 2020년 이후 연구 |
|------|-------------------|----------------|
| 기본 모델 | BP 신경망 (1 hidden layer) | 딥 뉴럴넷, Transformer, GNN 등 |
| 비용 통합 시점 | 데이터 전처리 or 테스트 후처리 | 손실 함수 내재화 (Focal Loss 등) |
| 다중 클래스 처리 | 직접 적용 어려움 | 다양한 특화 기법 존재 |
| 대규모 데이터 | 제한적 (KDD-99 최대) | 수백만~수십억 샘플 처리 |
| 해석 가능성 | 비교적 높음 | XAI 기법과의 결합 필요 |

---

### 4.3 앞으로 연구 시 고려할 점

#### (1) 🔑 다중 클래스 비용 구조의 복잡성 처리
논문이 밝힌 바와 같이, 다중 분류에서 비용 행렬의 복잡성이 급격히 증가합니다. 향후 연구에서는:
- **비용 행렬 자동 학습** 기법 개발
- **계층적 비용 구조** 모델링 (클래스 간 관계 반영)

$$Cost[i, j] \neq Cost[j, i] \text{ (비대칭 비용 고려)}$$

#### (2) 🔑 동적/변동 비용 환경 대응
논문이 한계로 지적한 고정 비용 행렬 문제:
- **온라인 학습(online learning)** 환경에서 비용이 실시간 변화
- 강화학습(RL) 프레임워크를 통한 적응적 비용 민감 학습 탐색

#### (3) 🔑 딥러닝과의 통합
- Threshold-Moving을 딥러닝의 **소프트맥스 온도 스케일링(temperature scaling)**과 결합
- 비용 인식 손실 함수 설계 (cost-aware loss):

$$\mathcal{L}_{cost} = -\sum_{i=1}^{C} \sum_{c \neq i} Cost[i,c] \cdot y_i \cdot \log P(c|x)$$

#### (4) 🔑 다양성-정확도 최적화 메커니즘
$Q_{av}$ 분석이 시사하듯, 앙상블의 다양성만으로는 충분하지 않음:
- 목적 함수에 다양성 항 명시적 추가

$$\mathcal{L}_{ensemble} = \mathcal{L}_{cost} - \lambda \cdot \text{Diversity}(\{h_1, h_2, \ldots, h_L\})$$

#### (5) 🔑 클래스 불균형과 비용 민감 학습의 통합 이론 개발
논문은 두 문제가 동일하지 않을 수 있음을 시사했지만 이론적 근거가 불충분:
- Bayes 결정 이론(Bayesian decision theory) 관점에서의 통합 프레임워크 개발
- **최적 임계값 선택의 이론적 보장** 연구

$$\hat{y} = \arg\min_c \sum_{j} P(y=j|x) \cdot Cost[j, c]$$

#### (6) 🔑 대규모 불균형 다중 분류
- 롱테일 분포(long-tail distribution) 환경에서의 비용 민감 학습
- Self-supervised / contrastive learning과의 결합
- Few-shot 학습 환경에서의 비용 민감 학습

#### (7) 🔑 실험적 타당성 강화
- 단순 UCI 데이터셋을 넘어 실세계 비용 정보가 있는 도메인 확장 (금융, 의료, 사이버보안)
- **비용 민감 평가 지표** 표준화 필요 (현재 연구마다 평가 방법이 상이)

---

## 참고 자료

**주요 참고 논문 (제공된 PDF 내 인용문헌):**

1. Zhou, Z.-H., & Liu, X.-Y. (2006). "Training Cost-Sensitive Neural Networks with Methods Addressing the Class Imbalance Problem." *IEEE Transactions on Knowledge and Data Engineering*. (**본 논문**)

2. Chawla, N.V., Bowyer, K.W., Hall, L.O., & Kegelmeyer, W.P. (2002). "SMOTE: Synthetic Minority Over-sampling Technique." *Journal of Artificial Intelligence Research*, 16, 321–357.

3. Domingos, P. (1999). "MetaCost: A General Method for Making Classifiers Cost-Sensitive." *Proceedings of the 5th ACM SIGKDD*, 155–164.

4. Elkan, C. (2001). "The Foundations of Cost-Sensitive Learning." *Proceedings of the 17th IJCAI*, 973–978.

5. Drummond, C., & Holte, R.C. (2003). "C4.5, Class Imbalance, and Cost Sensitivity: Why Under-Sampling Beats Over-Sampling." *ICML'03 Workshop on Learning from Imbalanced Data Sets*.

6. Kuncheva, L.I., & Whitaker, C.J. (2003). "Measures of Diversity in Classifier Ensembles." *Machine Learning*, 51(2), 181–207.

7. Abe, N., Zadrozny, B., & Langford, J. (2004). "An Iterative Method for Multi-Class Cost-Sensitive Learning." *Proceedings of the 10th ACM SIGKDD*, 3–11.

8. Maloof, M.A. (2003). "Learning when Data Sets are Imbalanced and when Costs are Unequal and Unknown." *ICML'03 Workshop on Learning from Imbalanced Data Sets*.

9. Provost, F. (2000). "Machine Learning from Imbalanced Data Sets 101." *AAAI'00 Workshop on Learning from Imbalanced Data Sets*.

10. Ting, K.M. (2002). "An Instance-Weighting Method to Induce Cost-Sensitive Trees." *IEEE Transactions on Knowledge and Data Engineering*, 14(3), 659–665.

**2020년 이후 관련 연구 방향 참고 (학습 데이터 기반, 직접 검증 권장):**

11. Cao, K., Wei, C., Gaidon, A., Arechiga, N., & Ma, T. (2019). "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss." *NeurIPS 2019*.

12. Ren, J., Yu, C., Ma, X., Zhao, H., et al. (2020). "Balanced Meta-Softmax for Long-Tailed Visual Recognition." *NeurIPS 2020*.

13. Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollar, P. (2017). "Focal Loss for Dense Object Detection." *ICCV 2017*. (ArXiv: 1708.02002)
