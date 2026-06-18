# SVMs Modeling for Highly Imbalanced Classification

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

본 논문(Tang et al., 2009, IEEE TSMCB)은 **극심한 클래스 불균형(highly imbalanced) 데이터**에서 SVM의 분류 성능 저하 문제를 해결하기 위해, **4가지 SVM 기반 리밸런싱 전략**을 체계적으로 비교·평가한다. 특히 **GSVM-RU(Granular SVMs–Repetitive Undersampling)** 알고리즘이 효과성(effectiveness)과 효율성(efficiency) 두 측면 모두에서 가장 우수함을 주장한다.

### 주요 기여

| 기여 항목 | 내용 |
|-----------|------|
| GSVM-RU 제안 | 그래뉼러 컴퓨팅 기반 반복적 언더샘플링 + 새로운 "combine" 집계 연산 설계 |
| 체계적 비교 | G-mean, AUC-ROC, F-measure, AUC-PR 4가지 메트릭으로 25개 실험 그룹 비교 |
| 이론적 확장 | 정보손실 최소화 원리(information-loss-minimization principle) 기반 이론화 |
| 실용적 검증 | McAfee TrustedSource 네트워크 보안 시스템에 실제 적용 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**클래스 불균형 문제(Class Imbalance Problem)**:
- 소수 클래스(minority/positive class)가 다수 클래스(majority/negative class)에 비해 극히 드문 상황
- 표준 SVM은 다수 클래스 방향으로 **편향(bias)**되어 소수 클래스에 대한 **거짓 음성(False Negative)** 증가
- 전통적 해결책(오버샘플링, 언더샘플링)은 효율성 저하 문제 존재
- SVM 예측 속도가 **지지벡터(SV) 수**에 비례하여 결정되므로, 불균형 해결 시 SV 수 증가 → 예측 속도 저하

**실제 응용 배경**: 악성 IP 탐지, 희귀 질병 진단, 신용카드 사기 탐지 등

---

### 2.2 평가 메트릭

$$\text{accuracy} = \frac{TP + TN}{TP + FP + FN + TN} \tag{1}$$

$$\text{sensitivity} = \frac{TP}{TP + FN} \tag{2}$$

$$\text{specificity} = \frac{TN}{TN + FP} \tag{3}$$

$$\text{G-Mean} = \sqrt{\text{sensitivity} \times \text{specificity}} \tag{4}$$

$$\text{precision} = \frac{TP}{TP + FP} \tag{5}$$

$$\text{recall} = \frac{TP}{TP + FN} \tag{6}$$

$$F\text{-Measure} = \frac{2 \times \text{precision} \times \text{recall}}{\text{precision} + \text{recall}} \tag{7}$$

> **Note**: 불균형 데이터에서 accuracy만으로는 불충분하므로, 4가지 메트릭을 모두 사용하는 것이 이 논문의 차별점이다.

---

### 2.3 제안 방법

#### (A) GSVM-RU (핵심 알고리즘)

**그래뉼러 컴퓨팅(Granular Computing)** 원리를 SVM에 통합:
- **Divide-and-Conquer**: 큰 문제를 정보 그래뉼(granule)로 분할
- **Data Cleaning**: 불필요한 정보 제거로 노이즈 감소

**GSVM-RU 동작 원리**:

$$\text{Training Set at step } k: Tr(k) = Tr(k-1) - NLSV(k-1)$$

여기서 $NLSV(k)$는 $k$번째 SVM 모델에서 추출된 **부정 로컬 지지벡터(Negative Local Support Vectors)**이다.

**알고리즘 흐름**:
1. 모든 양성(positive) 샘플을 정보 그래뉼로 초기화
2. 현재 훈련 데이터에 SVM 학습 → $NLSV(k)$ 추출
3. $NLSV(k)$를 집계 데이터셋에 추가
4. 분류 성능 개선 시 → $NLSV(k)$를 원본 데이터에서 제거 후 반복
5. 성능 개선 없으면 중단

**두 가지 집계(Aggregation) 연산**:

| 연산 | 방식 | 가정 |
|------|------|------|
| **Discard** | 새 그래뉼만 유지, 이전 그래뉼 폐기 | "boundary push": 이전 NLSV가 제거되면 경계가 이상적 위치로 이동 |
| **Combine** | 모든 이전 그래뉼과 새 그래뉼을 누적 | "information loss": 한 번에 모든 정보 샘플 추출 불가능 |

**하이퍼파라미터**: $Gr$ (부정 그래뉼 수)

#### (B) 세 가지 비교 SVM 알고리즘

**SVM-WEIGHT (비용 민감 학습)**:
$$C_{FN} = \frac{N_n}{R_w \times N_p}$$
여기서 $N_n$: 부정 샘플 수, $N_p$: 양성 샘플 수, $R_w$: 그리드 탐색으로 최적화

**SVM-SMOTE (오버샘플링)**:
- SMOTE로 $R_o \times N_p$개의 인공 양성 샘플 생성 후 SVM 학습
- 파라미터 $R_o$는 그리드 탐색으로 결정

**SVM-RANDU (랜덤 언더샘플링)**:
- $R_u \times N_p$개의 부정 샘플을 무작위 선택
- 파라미터 $R_u$는 그리드 탐색으로 결정

---

### 2.4 시간 복잡도 분석

| 알고리즘 | 훈련 복잡도 | 비고 |
|----------|------------|------|
| 표준 SVM | $O((N_p + N_n)^3)$ | 기준 |
| SVM-RANDU | $O((N_p + N_p \cdot R_u)^3)$ | $N_p \cdot R_u \ll N_n$이므로 빠름 |
| SVM-SMOTE | $O((N_p \cdot (R_o+1) + N_n)^3)$ | 데이터 증가로 느림 |
| SVM-WEIGHT | $O((N_p + N_n)^3)$ | 수렴 지연으로 실질적으로 느림 |
| GSVM-RU | $O(Gr \cdot (N_p + N_n)^3)$ | 후반 단계는 빠름 (hard 샘플 제거됨) |

**예측 복잡도**: SVM이 $N_s$개의 SV와 $N_u$개의 미지 샘플에 대해

$$\text{Prediction Time} = O(N_s \times N_u)$$

---

### 2.5 모델 구조

```
원본 훈련 데이터 Tr(1)
        ↓
    SVM_1 학습
        ↓
   NLSV(1) 추출
        ↓
Tr(2) = Tr(1) - NLSV(1)
        ↓
    SVM_2 학습
        ↓
   NLSV(2) 추출
        ↓
      ... (반복)
        ↓
집계(Discard 또는 Combine)
        ↓
최종 SVM (분류기)
```

---

### 2.6 성능 결과

**25개 실험 그룹 평균 성능 (Table V)**:

| 메트릭 | SVM-WEIGHT | SVM-SMOTE | SVM-RANDU | **GSVM-RU** |
|--------|-----------|-----------|-----------|-------------|
| G-Mean/8 | 85.1 | 63.0 | 83.4 | **85.2** |
| AUC-ROC/7 | 91.6 | 90.1 | 90.7 | **91.4** |
| F-Measure/5 | 67.2 | 67.4 | 65.4 | **66.5** |
| AUC-PR/5 | 66.5 | 66.4 | 64.2 | **65.2** |
| **SV 수 (효율성)** | 794 | 655 | 143 | **181** |
| **안정성** | YES | NO | NO | **YES** |

**이전 최고 알고리즘 대비 (Table IV, 18개 그룹)**:

| 메트릭 | 이전 최고 | GSVM-RU |
|--------|----------|---------|
| G-Mean/8 | 79.7 | **85.2** |
| AUC-ROC/5 | 90.2 | **92.4** |
| F-Measure/5 | 59.9 | **66.5** |

---

### 2.7 한계점

1. **시간 복잡도**: 그래뉼 수 $Gr$에 비례하여 훈련 시간 증가
2. **하이퍼파라미터 민감성**: $Gr$, 집계 방식(discard/combine) 선택이 데이터와 메트릭에 의존적
3. **그리드 탐색 비용**: 모든 알고리즘이 파라미터 최적화에 그리드 탐색 사용 → 계산 비용 높음
4. **소규모 데이터셋 한계**: 매우 작은 데이터셋에서는 반복적 SV 추출의 이점이 감소
5. **멀티클래스 미적용**: 이진 분류에만 초점, 다중 클래스 불균형은 미다룸
6. **선형 분리 불가 문제**: 커널 선택에 따라 성능이 크게 달라질 수 있음

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 구조적 위험 최소화(Structural Risk Minimization)와 일반화

SVM의 기본 원리인 구조적 위험 최소화는:

$$R_{struct} \leq R_{emp} + \Phi\left(\frac{h}{n}\right)$$

여기서 $R_{emp}$는 경험적 위험(훈련 오류), $\Phi$는 VC 차원 $h$와 샘플 수 $n$에 의존하는 복잡도 페널티이다.

GSVM-RU는 이 원리에 기반하여 **일반화 성능을 향상**시키는 두 가지 메커니즘을 갖는다:

### 3.2 일반화 향상 메커니즘

#### (1) 정보 손실 최소화를 통한 일반화

```
반복적 NLSV 추출
    → 각 단계에서 서로 다른 "경계 근방" 샘플 포착
    → 단일 SVM으로 놓친 정보 보완
    → 결정 경계의 정확도 향상
    → 테스트 데이터에 대한 일반화 성능 향상
```

#### (2) 노이즈 및 중복 데이터 제거를 통한 일반화

$$\text{Generalization Error} \propto \frac{\text{Noisy/Redundant Samples}}{\text{Total Training Samples}}$$

GSVM-RU는 노이즈/중복 부정 샘플을 제거함으로써:
- **과적합(overfitting) 위험 감소**
- 다수 클래스의 불필요한 지배력 억제
- 소수 클래스 경계 근방의 구조를 더 정확히 학습

#### (3) 지식 기반 그래뉼화(Knowledge-guided Granulation)

단순 랜덤 샘플링과 달리, **사전 지식(SVM의 SV 개념)**을 활용한 데이터 정제:

$$\text{Random Undersampling: } P(\text{informative sample selected}) = \frac{k}{N_n}$$

$$\text{GSVM-RU: } P(\text{informative sample selected}) \approx 1 \text{ (반복으로 수렴)}$$

#### (4) 집계 전략 선택의 일반화 효과

- **Discard 연산**: "boundary push" 가정 → 결정 경계를 이상적 위치로 점진적 이동 → 새 데이터에 대한 편향 감소
- **Combine 연산**: "information loss" 최소화 → 더 많은 정보 샘플 보존 → 소수 클래스 구조 학습 강화

#### (5) 안정성(Stability)과 일반화

SVM-SMOTE와 SVM-RANDU는 무작위성으로 인해 실행마다 결과가 달라지는 반면, **GSVM-RU는 파라미터 고정 시 항상 동일한 결과** → 일반화 성능의 **재현 가능성(reproducibility)** 보장

---

## 4. 앞으로의 연구에 미치는 영향 및 고려사항

### 4.1 연구에 미치는 영향

#### (1) 벤치마크 기준 확립
- 4개 메트릭(G-mean, AUC-ROC, F-measure, AUC-PR)을 동시에 사용하는 **포괄적 평가 프레임워크** 제시
- 7개 불균형 데이터셋에 대한 체계적 비교 → 후속 연구의 비교 기준점

#### (2) 그래뉼러 컴퓨팅과 SVM 결합 패러다임
- 지식 기반 데이터 정제 + SVM의 강점 결합 → **하이브리드 학습 프레임워크** 연구 촉진
- 반복적 정제(iterative refinement) 패러다임이 앙상블 방법론과 연결 가능

#### (3) 효율성과 효과성의 동시 추구
- 기존 연구가 둘 중 하나에만 집중한 것과 달리, **Pareto 최적(efficiency-effectiveness tradeoff)** 관점 제시

#### (4) 실제 시스템 적용 가능성 입증
- McAfee TrustedSource 시스템 적용 → 산업 현장에서의 실용적 가치 검증

---

### 4.2 향후 연구 시 고려사항

#### (1) 딥러닝 시대에서의 확장
현대의 불균형 학습 연구에서 고려해야 할 방향:

$$\mathcal{L}_{imbalanced} = \alpha \cdot \mathcal{L}_{minority} + (1-\alpha) \cdot \mathcal{L}_{majority}$$

GSVM-RU의 반복적 정제 아이디어를 **딥러닝의 커리큘럼 학습(Curriculum Learning)**과 결합 가능

#### (2) 하이퍼파라미터 최적화
- 그리드 탐색 대신 **베이지안 최적화(Bayesian Optimization)** 또는 **AutoML** 활용
- $Gr$ (그래뉼 수) 자동 결정 메커니즘 필요

#### (3) 멀티클래스 불균형으로 확장
- 현재 이진 분류에 한정 → **계층적 멀티클래스 GSVM-RU** 설계 필요
- One-vs-Rest, One-vs-One 전략과 결합 시 성능 검증 필요

#### (4) 대규모 데이터(Big Data) 적용
- $O(Gr \cdot (N_p + N_n)^3)$ 복잡도는 대규모 데이터에 부적합
- **분산 GSVM-RU**, **커널 근사(kernel approximation)** 기법과 결합 필요
- 예: Random Fourier Features를 활용한 $O(n)$ 근사 SVM과 결합

#### (5) 클래스 불균형 비율에 따른 적응적 전략
- 불균형 비율(imbalance ratio)에 따라 최적 전략이 다를 수 있음
- **적응형 집계 전략(adaptive aggregation)** 개발 필요

#### (6) 데이터 특성 다양화
- 현재 실험은 정형 데이터(tabular data)에 한정
- 이미지, 텍스트, 시계열 등 비정형 데이터에서의 검증 필요

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의**: 이하 내용은 제공된 논문 원문에 포함되지 않은 2020년 이후 연구들에 대한 내용입니다. 해당 논문들의 전문을 직접 검토하지 않았으므로, **일반적으로 알려진 연구 동향**을 기반으로 서술합니다. 구체적 수치나 세부 내용은 원 논문 확인을 권장합니다.

### 5.1 주요 연구 흐름 비교

| 연구 방향 | Tang et al. (2009) | 2020년 이후 동향 |
|-----------|-------------------|-----------------|
| **기반 모델** | SVM (커널 기반) | 딥러닝, GNN, Transformer |
| **샘플링 전략** | 반복적 언더샘플링 (GSVM-RU) | 생성 모델(GAN, VAE) 기반 오버샘플링 |
| **비용 민감** | 클래스 가중치 조정 | Focal Loss, Class-Balanced Loss |
| **평가 메트릭** | G-mean, AUC-ROC, F-measure, AUC-PR | 위 메트릭 + MCC(Matthews Correlation Coefficient) |
| **데이터 규모** | 수백~수만 샘플 | 수백만~수억 샘플 |

### 5.2 주요 최신 연구 방향

#### (1) GAN 기반 소수 클래스 증강
- **CTGAN**, **TVAE** 등 표 형식 데이터용 생성 모델
- SMOTE의 단순 선형 보간 한계를 극복, 더 복잡한 분포 학습 가능
- Tang et al.의 오버샘플링(SVM-SMOTE) 개념의 발전적 형태

#### (2) Focal Loss (Lin et al., 2017, RetinaNet)
$$\text{FL}(p_t) = -\alpha_t (1-p_t)^\gamma \log(p_t)$$

- 어려운 샘플(hard examples)에 가중치 부여 → GSVM-RU의 NLSV 개념과 유사한 철학
- 딥러닝 맥락에서 비용 민감 학습을 자연스럽게 구현

#### (3) 앙상블 방법과 불균형 학습 결합
- **BalancedRandomForest**, **EasyEnsemble**, **RUSBoost** 등
- Tang et al.의 반복적 접근법이 앙상블 학습으로 확장된 형태

#### (4) 메타 학습(Meta-Learning) 기반 접근
- **MAML** 등 few-shot learning 기법을 소수 클래스 학습에 적용
- 극소수 샘플(1~5개)만으로 소수 클래스 학습 가능

### 5.3 Tang et al. (2009)의 현재적 의의

```
2009년 GSVM-RU              2020년 이후 발전
─────────────────────────────────────────────
반복적 SV 기반 정제    →    Hard example mining
그래뉼러 집계          →    앙상블/부스팅
비용 민감 SVM         →    Focal Loss / Class-balanced Loss
4가지 메트릭 평가      →    표준 평가 프레임워크로 정착
```

이 논문의 핵심 아이디어인 **"어려운 샘플을 반복적으로 발굴하여 정보 손실을 최소화하면서 데이터를 정제한다"**는 개념은 최신 딥러닝 기반 불균형 학습에서도 여전히 유효한 원리로 작동한다.

---

## 참고자료

**원본 논문**:
- Tang, Y., Zhang, Y.-Q., Chawla, N. V., & Krasser, S. (2009). SVMs Modeling for Highly Imbalanced Classification. *IEEE Transactions on Systems, Man, and Cybernetics—Part B: Cybernetics*, 39(1), 281–288. DOI: 10.1109/TSMCB.2008.2002909

**논문 내 인용 참고문헌 (주요)**:
- Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16, 321–357.
- Wu, G., & Chang, E. Y. (2005). KBA: Kernel boundary alignment considering imbalanced data distribution. *IEEE Transactions on Knowledge and Data Engineering*, 17(6), 786–795.
- Akbani, R., Kwek, S., & Japkowicz, N. (2004). Applying support vector machines to imbalanced data sets. *Proc. 15th ECML*, pp. 39–50.
- Vapnik, V. N. (1998). *Statistical Learning Theory*. Wiley.
- Chang, C.-C., & Lin, C.-J. (2001). LIBSVM: A Library for Support Vector Machines.
- Japkowicz, N., & Stephen, S. (2002). The class imbalance problem: A systematic study. *Intelligent Data Analysis*, 6(5), 429–449.
- Kubat, M., & Matwin, S. (1997). Addressing the curse of imbalanced training sets: One-sided selection. *Proc. 14th ICML*, pp. 179–186.
