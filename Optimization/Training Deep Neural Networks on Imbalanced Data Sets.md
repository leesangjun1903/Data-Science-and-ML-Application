# Training Deep Neural Networks on Imbalanced Data Sets

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장
딥러닝의 표준 손실 함수인 MSE(Mean Squared Error)는 **클래스 불균형 데이터에서 다수 클래스에 편향**되어 소수 클래스를 제대로 학습하지 못한다. 이를 해결하기 위해 **클래스별로 오류를 동등하게 반영하는 새로운 손실 함수**를 제안한다.

### 주요 기여 (4가지)
| 기여 | 내용 |
|------|------|
| ① 새로운 손실 함수 제안 | MFE(Mean False Error) + MSFE(Mean Squared False Error) |
| ② 이론적 분석 | MSE 대비 제안 손실 함수의 우월성 수식적으로 증명 |
| ③ 역전파 분석 | 손실 함수별 그래디언트 계산 및 전파 효과 분석 |
| ④ 실증적 검증 | 8개 실제 데이터셋(이미지 3개, 문서 5개)에서 실험 검증 |

---

## 2. 상세 설명

### 2-1. 해결하고자 하는 문제

**클래스 불균형(Class Imbalance) 문제**:
- 현실 데이터에서 다수 클래스(majority)와 소수 클래스(minority)의 샘플 수가 크게 불균형
- 표준 MSE 손실 함수는 전체 데이터셋의 오류를 평균화하기 때문에, 다수 클래스의 오류가 손실값을 지배
- 결과적으로 분류기가 다수 클래스에 편향되어 소수 클래스를 잘못 분류

**MSE의 문제점 수식적 설명**:

$$l_{\text{MSE}} = \frac{1}{M} \sum_i \sum_n \frac{1}{2}(d_n^{(i)} - y_n^{(i)})^2 \tag{3.3}$$

여기서 $M$은 전체 샘플 수, $d_n^{(i)}$는 $i$번째 샘플의 $n$번째 뉴런에 대한 실제값, $y_n^{(i)}$는 예측값.

**문제 예시 (Table I의 혼동 행렬 기준)**:

| | 예측 P | 예측 N |
|--|--|--|
| 실제 P' (90개) | 86 | 4 |
| 실제 N' (10개) | 5 | 5 |

$$l_{\text{MSE}} = \frac{4+5}{90+10} = 0.09$$

→ MSE는 전체 오류를 단순 평균하여 소수 클래스의 오류를 과소평가함.

---

### 2-2. 제안하는 방법 (수식 포함)

#### ① MFE (Mean False Error)

**False Positive Error (FPE)**: 음성 클래스에서의 평균 오류

$$FPE = \frac{1}{N} \sum_{i=1}^{N} \sum_n \frac{1}{2}(d_n^{(i)} - y_n^{(i)})^2 \tag{3.5}$$

**False Negative Error (FNE)**: 양성 클래스에서의 평균 오류

$$FNE = \frac{1}{P} \sum_{i=1}^{P} \sum_n \frac{1}{2}(d_n^{(i)} - y_n^{(i)})^2 \tag{3.6}$$

**MFE 손실 함수**:

$$l' = FPE + FNE \tag{3.7}$$

예시 계산:

$$l_{\text{MFE}} = \frac{5}{10} + \frac{4}{90} = 0.54$$

→ MSE(0.09)보다 훨씬 큰 손실값 → 소수 클래스 오류에 더 민감

---

#### ② MSFE (Mean Squared False Error)

**MFE의 한계**: $FPE + FNE$를 최소화해도 FNE(소수 클래스 오류)만 독립적으로 낮아진다는 보장이 없음.

$$l'' = FPE^2 + FNE^2 \tag{3.8}$$

이를 전개하면:

$$l'' = \frac{1}{2}\left[(FPE + FNE)^2 + (FPE - FNE)^2\right] \tag{3.9}$$

> 이 식은 $(FPE+FNE)^2$와 $(FPE-FNE)^2$를 **동시에 최소화**함으로써, 두 클래스의 오류 합을 줄이는 동시에 두 클래스 간 오류 차이도 줄임 → **균형 잡힌 분류 보장**

예시 계산:

$$l_{\text{MSFE}} = \left(\frac{5}{10}\right)^2 + \left(\frac{4}{90}\right)^2 = 0.25$$

---

#### ③ 역전파 (Back-Propagation) 분석

출력 뉴런의 활성화 함수로 로지스틱 함수 사용:

$$y_n^{(i)} = \frac{1}{1 + \exp(-o_n^{(i)})} \tag{3.4}$$

**MSE 역전파 그래디언트**:

$$\frac{\partial l(\mathbf{d}^{(i)}, \mathbf{y}^{(i)})}{\partial o_n^{(i)}} = -(d_n^{(i)} - y_n^{(i)}) y_n^{(i)}(1 - y_n^{(i)}) \tag{3.12}$$

**MFE 역전파 그래디언트** (클래스별로 다른 그래디언트 적용):

$$\frac{\partial l}{\partial o_n^{(i)}} = -\frac{1}{N}(d_n^{(i)} - y_n^{(i)}) y_n^{(i)}(1 - y_n^{(i)}), \quad (i \in \mathbf{N}) \tag{3.14}$$

$$\frac{\partial l}{\partial o_n^{(i)}} = -\frac{1}{P}(d_n^{(i)} - y_n^{(i)}) y_n^{(i)}(1 - y_n^{(i)}), \quad (i \in \mathbf{P}) \tag{3.15}$$

**MSFE 역전파 그래디언트**:

$$\frac{\partial l}{\partial o_n^{(i)}} = -\frac{2 \cdot FPE}{N}(d_n^{(i)} - y_n^{(i)}) y_n^{(i)}(1 - y_n^{(i)}), \quad (i \in \mathbf{N}) \tag{3.17}$$

$$\frac{\partial l}{\partial o_n^{(i)}} = -\frac{2 \cdot FNE}{P}(d_n^{(i)} - y_n^{(i)}) y_n^{(i)}(1 - y_n^{(i)}), \quad (i \in \mathbf{P}) \tag{3.18}$$

> MSFE는 현재 FPE와 FNE 크기에 따라 그래디언트를 **동적으로 가중**하여, 더 큰 오류를 가진 클래스에 더 강한 학습 신호를 전달함.

---

### 2-3. 모델 구조

- **기본 구조**: 다중 은닉층을 가진 **DNN (Deep Neural Network)**
- 손실 함수만 교체(MFE/MSFE), 네트워크 구조 자체는 표준 DNN과 동일
- 각 데이터셋별 최적 구조 (휴리스틱 탐색으로 결정):

| 데이터셋 | 은닉층 수 | 뉴런 수 (하단→상단) |
|---------|---------|-------------------|
| Household | 3 | 1000, 300, 100 |
| Tree 1, 2 | 3 | 1000, 100, 10 |
| Doc. 1~3 | 6 | 3000, 1000, 300, 100, 30, 10 |
| Doc. 4~5 | 6 | 3000, 1500, 800, 400, 200, 50 |

---

### 2-4. 성능 향상 및 한계

#### 성능 향상
실험은 CIFAR-100(이미지 3종)과 20 Newsgroups(문서 5종), 불균형 수준 20%/10%/5%로 구성.

**대표 결과 (Tree 2 데이터셋, Imb. level = 5%)**:

| 방법 | F-measure | AUC |
|-----|-----------|-----|
| DNN (MSE) | 0.0000 | 0.548 |
| DNN (MFE) | **0.1071** | **0.652** |
| DNN (MSFE) | **0.1481** | **0.700** |

**핵심 발견**:
- MFE/MSFE는 대부분의 실험에서 MSE보다 같거나 높은 F-measure, AUC 달성
- 불균형 정도가 심할수록(5% 수준) 성능 향상 폭이 더 커짐
- 동일 손실값 조건에서 MFE/MSFE가 MSE보다 F-measure와 AUC 모두 높음
- 학습 곡선이 MSE보다 **안정적**이어서 최적점 탐색 용이

#### 한계
1. **이진 분류에 한정**: 다중 클래스 문제는 직접 다루지 않음 (이진화로 변환 가능하다고만 언급)
2. **MFE의 불완전성**: FPE+FNE 최소화가 개별 클래스 오류를 균등하게 줄이지 못할 수 있음 → MSFE로 보완하나 완전 해결은 아님
3. **네트워크 구조 탐색 자동화 부재**: 최적 구조를 휴리스틱하게 결정 (Layer 수, 뉴런 수 수작업 탐색)
4. **실험 범위**: CNN, DBN 등 다른 딥러닝 아키텍처에 대한 검증 없음
5. **불균형 비율 한정**: 5%, 10%, 20%만 실험. 더 극단적 불균형(1% 이하)에 대한 검증 미흡
6. **비교 대상 제한**: SMOTE, 비용민감 학습 등 기존 불균형 처리 기법과의 직접 비교 없음

---

## 3. 모델의 일반화 성능 향상 가능성

본 논문에서 일반화 성능과 관련된 내용을 중점적으로 분석하면 다음과 같다.

### 3-1. 일반화 성능 향상 메커니즘

**① 클래스별 균등 학습을 통한 편향(Bias) 감소**

MSE의 핵심 문제는 불균형 데이터에서 **다수 클래스에 편향된 특징(feature)을 학습**한다는 것이다. MFE/MSFE는 각 클래스의 오류를 독립적으로 계산하고 합산하므로:

$$l' = FPE + FNE = \frac{1}{N}\sum_{\text{neg}} \text{error} + \frac{1}{P}\sum_{\text{pos}} \text{error}$$

이를 통해 소수 클래스의 개념(concept)을 충분히 학습하여 **테스트 데이터에서도 소수 클래스를 올바르게 분류**할 수 있는 일반화 능력을 향상시킨다.

**② 안정적인 학습 곡선**

논문의 Figure 1, 2에서 보여주듯, MFE/MSFE는 MSE보다 **훨씬 안정적인 손실 감소 곡선**을 보인다. 이는:
- 그래디언트 업데이트가 일관성 있게 이루어짐
- 과도한 진동 없이 최적점으로 수렴
- **과적합(overfitting)을 간접적으로 억제**하는 효과

**③ MSFE의 분산 감소 효과**

MSFE의 수식 분해:

$$l'' = FPE^2 + FNE^2 = \frac{1}{2}[(FPE+FNE)^2 + (FPE-FNE)^2]$$

$(FPE - FNE)^2$를 최소화함으로써 두 클래스의 오류를 **균형 있게** 줄인다. 이는 특정 클래스에만 과도하게 최적화되는 것을 방지하여 **더 나은 일반화**를 기대할 수 있다.

**④ 고차원 데이터에서의 표현 학습**

논문에서는 DNN의 다중 은닉층이 "강력한 일반화 및 특징 추출 능력"을 가진다고 강조한다. MFE/MSFE는 이러한 DNN의 특징 추출 능력을 **소수 클래스에도 균등하게 적용**하도록 유도한다.

### 3-2. 일반화 성능의 한계 및 고려 사항

- **데이터 의존성**: 일부 데이터셋(예: Tree 1, Imb. 20%)에서는 MSE와 동등한 성능을 보여, 모든 상황에서 일반화 향상이 보장되지 않음
- **불균형 정도에 따른 효과 차이**: 불균형이 심할수록 일반화 향상 효과가 크지만, 약한 불균형에서는 효과가 제한적
- **정규화 기법 미통합**: Dropout, L2 정규화 등 일반화 향상 기법과의 결합 효과 미검토

---

## 4. 앞으로의 연구에 미치는 영향과 고려할 점

### 4-1. 앞으로의 연구에 미치는 영향

**① 딥러닝 손실 함수 설계의 새로운 방향 제시**

본 논문은 딥러닝에서 손실 함수 자체를 수정하여 불균형 문제를 해결하는 **알고리즘 수준 접근법**의 선구적 연구다. 이후 많은 연구가 손실 함수 재설계를 통해 불균형 문제를 접근하는 데 영향을 미쳤다.

**② Focal Loss 등 후속 연구의 기반**

클래스별 오류 가중치 부여 개념은 이후 **Focal Loss** (Lin et al., 2017, RetinaNet)로 발전하였다. Focal Loss는 어려운 샘플(주로 소수 클래스)에 더 높은 가중치를 부여하는 개념으로, 본 논문과 동일한 문제의식을 공유한다.

**③ 의료, 금융 등 고위험 도메인 적용 가능성**

소수 클래스(예: 질병 환자, 사기 거래)의 올바른 분류가 중요한 도메인에서, 손실 함수 수정 접근법은 기존 샘플링 방식 없이 직접 적용 가능하여 **실용적 영향력**이 크다.

---

### 4-2. 2020년 이후 관련 최신 연구 비교 분석

#### ① Class-Balanced Loss (Cui et al., 2019/2020)

- **제안**: 각 클래스의 유효 샘플 수(effective number of samples)를 기반으로 손실 가중치를 조정

$$\text{Effective Number} = \frac{1 - \beta^n}{1 - \beta}$$

- **Wang et al. 대비**: Wang et al.은 단순히 양성/음성 클래스를 나눠 평균 오류를 계산하지만, Cui et al.은 클래스별 샘플 수의 다양성을 더 세밀하게 반영함

#### ② Focal Loss (Lin et al., 2017, ICCV) → 이후 다양한 변형 연구 지속

$$FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

- 쉽게 분류되는 샘플의 손실을 감소시키고, 어려운 샘플(소수 클래스 포함)에 집중
- **Wang et al. 대비**: 클래스 기반 분리 대신 **샘플별 난이도 기반** 동적 가중치 부여로 더 세밀한 제어 가능

#### ③ Label-Distribution-Aware Margin Loss (LDAM, Cao et al., 2019, NeurIPS)

$$\Delta_j = \frac{C}{n_j^{1/4}}$$

- 소수 클래스에 더 큰 마진을 부여하여 일반화 성능 향상
- 이론적 기반이 더 탄탄하며, Wang et al.의 직관적 접근을 이론적으로 확장

#### ④ MixUp, Manifold Mixup 기반 불균형 처리 (2020년 이후)

- 데이터 증강과 손실 함수를 결합하는 방향으로 발전
- Wang et al.의 순수 손실 함수 접근법과 상호 보완적

#### ⑤ Self-paced Ensemble (Liu et al., 2020, ICSE)

- 앙상블과 커리큘럼 학습을 결합한 불균형 처리
- 샘플링 + 모델 수준 접근으로, Wang et al.의 손실 함수 접근법과 다른 방향

#### 비교 요약표

| 연구 | 접근 방식 | Wang et al. 대비 장점 | Wang et al. 대비 단점 |
|------|----------|---------------------|---------------------|
| Wang et al. (2016) | 클래스별 손실 분리 | 단순, 구현 용이 | 다중 클래스, 극단 불균형 한계 |
| Focal Loss (2017) | 샘플 난이도 기반 가중치 | 더 세밀한 제어 | 하이퍼파라미터($\gamma$) 민감 |
| Class-Balanced Loss (2019) | 유효 샘플 수 기반 | 이론적 근거 탄탄 | 복잡한 계산 |
| LDAM (2019) | 마진 기반 | 일반화 이론 제공 | 구현 복잡도 높음 |

---

### 4-3. 앞으로 연구 시 고려할 점

**① 다중 클래스 불균형으로의 확장**

현 논문은 이진 분류에 한정됨. 실제 응용에서는 다수의 클래스가 서로 다른 비율로 불균형한 경우가 일반적이므로, MFE/MSFE의 다중 클래스 버전 개발이 필요하다.

**② 다양한 아키텍처와의 결합**

논문 결론부에서도 언급되었듯이, CNN, RNN, Transformer 등 최신 아키텍처에서의 효과 검증이 필요하다. 특히 Vision Transformer(ViT)나 BERT 계열 모델에 적용 시의 효과를 분석해야 한다.

**③ 극단적 불균형 상황 (Long-tail Distribution)**

5% 수준 이하(1:100, 1:1000 등)의 극단적 불균형에서의 성능 검토가 필요하다. Long-tail recognition은 2020년 이후 중요한 연구 주제이며, MFE/MSFE가 이런 환경에서 얼마나 효과적인지 불명확하다.

**④ 정규화 기법과의 결합**

- Dropout, Batch Normalization, Label Smoothing 등과의 결합 효과 연구
- 불균형 데이터에서 정규화가 소수 클래스 학습에 미치는 영향 분석

**⑤ 샘플링 기법과의 하이브리드 접근**

SMOTE 등 데이터 수준 기법과 MFE/MSFE를 결합한 하이브리드 방법의 효과를 체계적으로 비교해야 한다. 두 접근법이 상호 보완적일 가능성이 높다.

**⑥ 평가 지표 다양화**

F-measure와 AUC 외에도 G-mean, Balanced Accuracy, Matthews Correlation Coefficient(MCC) 등 다양한 불균형 데이터 평가 지표를 함께 활용해야 한다.

**⑦ 동적 가중치 조정 메커니즘**

학습 과정에서 FPE와 FNE의 비율이 변함에 따라 손실 가중치를 **동적으로 조정**하는 메커니즘 (Curriculum Learning, Self-paced Learning과 결합) 연구가 유망하다.

**⑧ 이론적 일반화 경계(Generalization Bound) 분석**

본 논문은 경험적 검증에 치중하고 있어, MFE/MSFE 손실 함수에 대한 PAC Learning 이론이나 VC Dimension 기반의 이론적 일반화 경계 분석이 필요하다.

---

## 참고 자료

**주 논문**:
- Wang, S., Liu, W., Wu, J., Cao, L., Meng, Q., & Kennedy, P. J. (2016). *Training deep neural networks on imbalanced data sets*. 2016 International Joint Conference on Neural Networks (IJCNN), pp. 4368–4374. IEEE.

**관련 비교 연구**:
- Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). *Focal loss for dense object detection*. ICCV 2017.
- Cui, Y., Jia, M., Lin, T. Y., Song, Y., & Belongie, S. (2019). *Class-balanced loss based on effective number of samples*. CVPR 2019.
- Cao, K., Wei, C., Gaidon, A., Arechiga, N., & Ma, T. (2019). *Learning imbalanced datasets with label-distribution-aware margin loss*. NeurIPS 2019.
- He, H., & Garcia, E. A. (2009). *Learning from imbalanced data*. IEEE Transactions on Knowledge and Data Engineering, 21(9), 1263–1284.
- Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). *SMOTE: Synthetic minority over-sampling technique*. Journal of Artificial Intelligence Research, 16, 321–357.
- Zhou, Z. H., & Liu, X. Y. (2006). *Training cost-sensitive neural networks with methods addressing the class imbalance problem*. IEEE Transactions on Knowledge and Data Engineering, 18(1), 63–77.
