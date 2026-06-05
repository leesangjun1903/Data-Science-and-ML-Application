# Learning with Bounded Instance- and Label-dependent Label Noise

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

이 논문은 **Bounded Instance- and Label-dependent label Noise (BILN)** 하에서 이론적 보장을 갖춘 학습 알고리즘을 최초로 제안합니다. 기존 연구들이 주로 RCN(Random Classification Noise)이나 CCN(Class-Conditional Noise)에 집중한 것과 달리, 더 현실적인 노이즈 모델인 ILN(Instance- and Label-dependent Noise)의 특수 사례를 다룹니다.

### 주요 기여

| 기여 항목 | 내용 |
|-----------|------|
| **새로운 노이즈 모델** | BILN 정의 및 공식화 |
| **Distilled Examples 개념** | Bayes 최적 분류기와 일치하는 레이블을 가진 예제 정의 |
| **이론적 보장** | 통계적 일관성(statistical consistency) 및 성능 경계(performance bound) 증명 |
| **실용적 알고리즘** | 능동 학습 + 중요도 재가중치(importance reweighting) 결합 |
| **노이즈율 불필요 옵션** | $\rho_{\pm 1\text{max}}$ 없이도 작동하는 변형 알고리즘 제안 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**ILN의 형식적 정의:**

노이즈율 $\rho_y(\mathbf{x}) = P(\tilde{Y} = -y \mid X = \mathbf{x}, Y = y)$에서, ILN은 $\rho_y(\mathbf{x})$가 인스턴스 $\mathbf{x}$와 레이블 $y$ 모두에 의존합니다.

- **RCN**: $\rho_{+1}(\mathbf{x}) = \rho_{-1}(\mathbf{x}) = \rho$ (상수)
- **CCN**: $\rho_y(\mathbf{x})$는 $\mathbf{x}$에 독립, $y$에만 의존
- **ILN**: $\rho_y(\mathbf{x})$가 $\mathbf{x}$와 $y$ 모두에 의존 → 가장 현실적

**BILN 가정 (Assumption 1):**

$$\begin{cases} 0 \leq \rho_{+1}(\mathbf{x}) \leq \rho_{+1\text{max}} < 1 \\ 0 \leq \rho_{-1}(\mathbf{x}) \leq \rho_{-1\text{max}} < 1 \\ 0 \leq \rho_{+1}(\mathbf{x}) + \rho_{-1}(\mathbf{x}) < 1 \end{cases}$$

세 번째 조건 $\rho_{+1}(\mathbf{x}) + \rho_{-1}(\mathbf{x}) < 1$은 "평균적으로 노이즈 레이블과 실제 레이블이 일치해야 함"을 의미합니다.

---

### 2.2 제안하는 방법 (수식 포함)

#### Step 1: Distilled Examples의 정의와 이론적 근거

**Definition 1 (Distilled Example):**

예제 $(\mathbf{x}, y)$는 $y = g^*_D(\mathbf{x})$일 때, 즉 레이블이 Bayes 최적 분류기의 예측과 동일할 때 *distilled example*이라 합니다.

**Lemma 1:**

$\eta(\mathbf{x}) = P_D(Y = +1 \mid X = \mathbf{x})$라 할 때, Bayes 최적 분류기는:

$$g^*_D(\mathbf{x}) = \text{sgn}\!\left(\eta(\mathbf{x}) - \frac{1}{2}\right)$$

**Theorem 1:**

$P_D(\mathbf{x})$와 $P_{D^*}(\mathbf{x})$가 동일한 support를 공유하면:

```math
g^*_{D^*} = g^*_D
```

즉, distilled examples의 분포 $D^*$에서 학습한 Bayes 최적 분류기는 clean 분포 $D$에서의 Bayes 최적 분류기와 동일합니다.

**Proposition 1 (성능 경계):**

$L$이 $[0, b]$-valued이고, $f^\*_{D^*, L} \in \mathcal{F}$이면, 확률 $1 - \delta$로:

```math
R_{D^*,L}(\hat{f}_{D^*,L}) - R_{D^*,L}(f^*_{D^*,L}) \leq 2\mathfrak{R}(L \circ \mathcal{F}) + 2b\sqrt{\frac{\log(1/\delta)}{2m}}
```

여기서 Rademacher 복잡도:

$$\mathfrak{R}(L \circ \mathcal{F}) = \mathbb{E}_{D^*, \sigma}\!\left[\sup_{f \in \mathcal{F}} \frac{2}{m}\sum_{i=1}^{m} \sigma_i L(f(\mathbf{x}_i), y_i)\right]$$

---

#### Step 2: 노이즈 예제에서 Distilled Examples 자동 수집

**Theorem 2:**

$\tilde{\eta}(\mathbf{x}) = P_{D_\rho}(\tilde{Y} = +1 \mid X = \mathbf{x})$라 할 때, $UB(\rho_{\pm 1}(\mathbf{x}))$를 $\rho_{\pm 1}(\mathbf{x})$의 상한이라 하면:

$$\tilde{\eta}(\mathbf{x}) < \frac{1 - UB(\rho_{+1}(\mathbf{x}))}{2} \Rightarrow (\mathbf{x}, Y = -1) \text{ is distilled}$$

$$\tilde{\eta}(\mathbf{x}) > \frac{1 + UB(\rho_{-1}(\mathbf{x}))}{2} \Rightarrow (\mathbf{x}, Y = +1) \text{ is distilled}$$

**Corollary 1:**

$$\tilde{\eta}(\mathbf{x}) < \frac{1 - \rho_{+1\text{max}}}{2} \Rightarrow (\mathbf{x}, Y = -1) \text{ is distilled}$$

$$\tilde{\eta}(\mathbf{x}) > \frac{1 + \rho_{-1\text{max}}}{2} \Rightarrow (\mathbf{x}, Y = +1) \text{ is distilled}$$

---

#### Step 3: 능동 학습으로 편향 해소

자동 수집된 distilled examples는 다음 영역의 예제를 포함하지 못합니다:

```math
\text{supp}(P_{D^*_{\text{auto}}}(\mathbf{x})) = \left\{\mathbf{x} \in \mathcal{X} \mid \tilde{\eta}(\mathbf{x}) \in \left[0, \frac{1 - \rho_{+1\text{max}}}{2}\right) \cup \left(\frac{1 + \rho_{-1\text{max}}}{2}, 1\right]\right\}
```

이로 인해 $\text{supp}(P_{D^*_{\text{auto}}}(\mathbf{x})) \neq \text{supp}(P_D(\mathbf{x}))$이 되어 통계적 일관성이 깨집니다.

**해결책:** $n_{\text{act}}$개의 예제를 무작위로 선택하여 전문가가 레이블링 → $\text{supp}(P_{D^*}(\mathbf{x})) = \text{supp}(P_D(\mathbf{x}))$ 보장

---

#### Step 4: 공변량 이동 보정 (Importance Reweighting)

**Assumption 2:** $P_D(\mathbf{x}, y)$와 $P_{D^*}(\mathbf{x}, y)$는 marginal 분포 $P(\mathbf{x})$에서만 차이납니다.

중요도 $\beta(\mathbf{x}) = \frac{P_D(\mathbf{x})}{P_{D^*}(\mathbf{x})}$를 이용한 위험 변환:

$$R_{D,L}(f) = \mathbb{E}_{(\mathbf{X},Y) \sim D^*}\!\left[\beta(\mathbf{X}) L(f(\mathbf{X}), Y)\right] = R_{D^*, \beta L}(f) \tag{1}$$

**경험적 추정:**

$$\hat{R}_{D^*, \beta L}(f) = \frac{1}{m}\sum_{i=1}^{m} \beta(\mathbf{x}_i^{\text{distilled}}) L(f(\mathbf{x}_i^{\text{distilled}}), y_i^{\text{distilled}})$$

**학습 목표:**

$$\hat{f}_{D^*, \beta L} = \arg\min_{f \in \mathcal{F}} \hat{R}_{D^*, \beta L}(f) \tag{2}$$

**KMM (Kernel Mean Matching)으로 $\beta$ 추정:**

```math
\min_{\beta} \frac{1}{2}\boldsymbol{\beta}^T \mathbf{K} \boldsymbol{\beta} - \boldsymbol{\kappa}^T \boldsymbol{\beta}
```

$$\text{s.t.} \quad \forall i, \beta_i \in [0, B] \quad \text{and} \quad \left|\sum_{i=1}^{m} \beta_i - m\right| \leq m\epsilon \tag{3}$$

여기서 $K_{ij} = k(\mathbf{x}\_i^{\text{distilled}}, \mathbf{x}\_j^{\text{distilled}})$, $\kappa_i = \frac{m}{n}\sum_{j=1}^n k(\mathbf{x}_i^{\text{distilled}}, \mathbf{x}_j)$

**Proposition 2:**

확률 $1 - \delta$로:

$$R_{D,L}(\hat{f}_{D^*, \beta L}) - R_{D,L}(f^*_{D,L}) \leq 2\mathfrak{R}(\beta \circ L \circ \mathcal{F}) + 2b\sqrt{\frac{\log(1/\delta)}{2m}}$$

---

#### Step 5: $\rho_{\pm 1\text{max}}$ 없이 Distilled Examples 수집

**Theorem 3:**

$$\rho_{+1}(\mathbf{x}) \leq 1 - \tilde{\eta}(\mathbf{x}), \quad \rho_{-1}(\mathbf{x}) \leq \tilde{\eta}(\mathbf{x})$$

$k$-최근접 이웃 $\mathcal{N}_k(\mathbf{x})$를 이용하여 근사적 상한 추정:

$$UB(\rho_{+1}(\mathbf{x})) \approx \frac{\sum_{\mathbf{x}_j \in \mathcal{N}_k(\mathbf{x})} \tilde{\eta}(\mathbf{x}_j)}{k}, \quad UB(\rho_{-1}(\mathbf{x})) \approx \frac{\sum_{\mathbf{x}_j \in \mathcal{N}_k(\mathbf{x})} (1 - \tilde{\eta}(\mathbf{x}_j))}{k}$$

---

### 2.3 모델 구조

```
[노이즈 데이터 입력]
        ↓
[η̃(x) 추정 - 로지스틱 회귀/DNN]
        ↓
[Corollary 1 기준으로 Distilled Examples 자동 수집]
        ↓
[능동 학습: n_act개 무작위 샘플링 후 전문가 레이블링]
        ↓
[KMM으로 중요도 β 추정]
        ↓
[중요도 재가중치 손실로 최종 분류기 학습]
```

---

### 2.4 성능 향상

**합성 데이터셋 결과 (Table 1):**

| 설정 | noisy | auto+act (ours) | Algo.1 (ours) |
|------|-------|-----------------|---------------|
| $(0.25, 0.25, 3)$ | $98.55\pm1.28$ | $99.28\pm0.70$ | $\mathbf{99.30\pm0.68}$ |
| $(0.49, 0.49, 3)$ | $88.57\pm10.74$ | $98.16\pm2.57$ | $\mathbf{98.43\pm2.29}$ |

**실제 데이터셋 (UCI Image):**

| 설정 | noisy | Algo.1 (ours) |
|------|-------|---------------|
| $(0.5, 0.5, 20)$ | $68.72\pm5.91$ | $\mathbf{75.64\pm4.69}$ |

### 2.5 한계

1. **이진 분류 중심**: 다중 클래스 확장은 보충 자료에서만 논의
2. **KMM 계산 복잡도**: $O(m^3)$ 수준의 이차 프로그래밍
3. **능동 레이블링 비용**: 전문가 레이블링이 필요하여 실용적 한계 존재
4. **$\rho_{\pm 1\text{max}}$ 지식 가정**: 기본 설정에서 노이즈 상한값을 알아야 함
5. **딥러닝과의 통합**: 로지스틱 회귀 기반으로 복잡한 모델로의 확장이 미흡
6. **Assumption 2의 강건성**: 공변량 이동 가정이 항상 성립하지 않을 수 있음

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 이론적 일반화 보장

본 논문의 일반화 성능은 **Rademacher 복잡도 기반 경계**로 보장됩니다.

**핵심 메커니즘:**

**[메커니즘 1] Distilled Examples의 분포 일치**

Theorem 1에 의해 $P_D(\mathbf{x})$와 $P_{D^*}(\mathbf{x})$가 동일한 support를 공유하면:

```math
g^*_{D^*} = g^*_D
```

이는 distilled examples로 학습해도 원래 clean 분포에서의 최적 분류기와 동일한 솔루션으로 수렴함을 의미합니다. **즉, 노이즈 하에서도 generalization gap이 발생하지 않습니다.**

**[메커니즘 2] 중요도 재가중치에 의한 공변량 이동 보정**

Proposition 2는 중요도 재가중치를 적용했을 때의 일반화 경계를 제공합니다:

$$R_{D,L}(\hat{f}_{D^*, \beta L}) - R_{D,L}(f^*_{D,L}) \leq 2\mathfrak{R}(\beta \circ L \circ \mathcal{F}) + 2b\sqrt{\frac{\log(1/\delta)}{2m}}$$

이 경계는 $m \to \infty$일 때 0으로 수렴하므로, **표본 수가 증가할수록 일반화 성능이 향상**됩니다.

**[메커니즘 3] 통계적 일관성 (Statistical Consistency)**

능동 학습으로 $\text{supp}(P\_{D^\*}(\mathbf{x})) = \text{supp}(P_D(\mathbf{x}))$를 보장하면, $L$이 분류-보정(classification-calibrated)이고 $f^\*_{D^*,L} \in \mathcal{F}$이면:

```math
\text{sgn}(\hat{f}_{D^*,L}) \xrightarrow{m \to \infty} g^*_D
```

즉, **distilled examples의 수가 증가함에 따라 학습된 분류기가 Bayes 최적 분류기에 수렴**합니다.

### 3.2 실험적 일반화 증거

- 높은 노이즈율 $(\rho_{+1\text{max}}, \rho_{-1\text{max}}) = (0.49, 0.49)$에서도 noisy 대비 약 10%p 향상
- 표준편차 감소: noisy($\pm10.74$) vs Algo.1($\pm2.29$) → **일관성 있는 일반화**
- $k$ 하이퍼파라미터에 대한 robust성: Fig. 2에서 $k \in [5, 15]$ 전 구간에서 안정적 성능

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 향후 연구에 미치는 영향

**① ILN 연구의 이론적 기반 제공**

이 논문은 ILN에 대한 최초의 이론적 보장을 제공하여, 이후 연구들이 더 복잡한 ILN 시나리오를 다룰 때 참조할 수 있는 기반을 마련했습니다.

**② Distilled Examples 패러다임의 확산**

"깨끗한 부분집합을 추출하여 학습"하는 아이디어는 이후 DivideMix (Li et al., 2020), MentorNet (Jiang et al., 2018) 등의 방향성과 연결되며, 노이즈 레이블 학습의 주요 패러다임으로 자리잡았습니다.

**③ 능동 학습과 노이즈 학습의 결합**

노이즈 학습에 능동 학습을 도입한 선구적 사례로, 이후 소수의 정확한 레이블을 활용하는 semi-supervised 노이즈 학습 연구를 자극했습니다.

**④ 보완 레이블 학습으로의 확장**

저자들이 명시한 바와 같이, 인스턴스 의존적 보완 레이블 학습(complementary label learning)으로의 확장 가능성이 있습니다.

### 4.2 앞으로 연구 시 고려할 점

**① 딥러닝과의 통합**

현재 알고리즘은 로지스틱 회귀 기반이므로, DNN과 결합할 때 다음을 고려해야 합니다:
- $\tilde{\eta}(\mathbf{x})$ 추정의 정확도가 DNN의 과적합으로 저하될 수 있음
- KMM의 계산 복잡도가 대규모 데이터에서 병목이 될 수 있음
- DNN에서의 Rademacher 복잡도 제어 방법 연구 필요

**② 노이즈율 추정의 정확도**

$\rho_{\pm 1\text{max}}$를 알지 못하는 경우 $k$-NN 기반 추정을 사용하는데, 고차원 공간에서의 $k$-NN 저하 문제(curse of dimensionality)를 해결해야 합니다.

**③ 다중 클래스 확장**

이진 분류에서 다중 클래스로 확장할 때 노이즈 전이 행렬(noise transition matrix)의 추정 문제가 복잡해집니다.

**④ 현실적인 능동 학습 예산**

실제 환경에서 $n_{\text{act}}$ 레이블링 비용을 최소화하면서도 coverage를 보장하는 전략 연구가 필요합니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 주요 후속 연구 비교

| 논문 | 발표연도 | 노이즈 모델 | 이론적 보장 | 핵심 방법 |
|------|----------|-------------|-------------|-----------|
| **본 논문 (BILN)** | ICML 2020 | ILN (bounded) | ✅ 통계적 일관성 | Distilled Examples + KMM |
| **Parts-dependent (Xia et al.)** | NeurIPS 2020 | ILN (parts-dependent) | 부분적 | 부분 의존 가정 활용 |
| **CORES (Cheng et al.)** | ICML 2021 | ILN | ✅ | Confidence regularized self-training |
| **Instance-dependent label noise (Berthon et al.)** | AISTATS 2021 | ILN | ✅ | 전이 행렬 추정 |
| **Sample Selection + Mixup** | 2021 이후 | CCN/ILN | 부분적 | Small-loss trick |

### 5.2 본 논문 대비 발전 방향

**[발전 1] DNN 기반 $\tilde{\eta}$ 추정 정교화**

2020년 이후 연구들은 DNN을 이용한 $\eta(\mathbf{x})$ 추정 정확도를 높이는 데 집중했습니다. 특히 **DivideMix (Li et al., ICLR 2020)**는 GMM을 이용해 깨끗한 예제와 노이즈 예제를 분리하여 본 논문의 distilled examples 개념과 유사한 접근을 딥러닝에 적용했습니다.

**[발전 2] 이론적 틀의 확장**

본 논문의 BILN 조건은 $\rho_{+1}(\mathbf{x}) + \rho_{-1}(\mathbf{x}) < 1$을 요구하지만, 이후 연구들은 이 조건을 완화하거나 다른 형태의 조건으로 대체하는 방향을 모색하고 있습니다.

**[발전 3] 자기 지도 학습과의 결합**

2021년 이후 **자기 지도 사전학습(self-supervised pretraining)**과 노이즈 레이블 학습을 결합하는 연구가 활발해졌으며, 이는 본 논문에서 제안한 distilled examples의 품질을 높이는 데 기여할 수 있습니다.

**[발전 4] 전이 행렬의 인스턴스 의존적 추정**

Xia et al. (2020, NeurIPS)의 "Parts-dependent label noise"는 인스턴스 의존적 노이즈 전이 행렬을 추정하여 본 논문의 접근법을 보완합니다. 그러나 두 접근법 모두 고차원 데이터에서의 추정 정확도 문제가 남아 있습니다.

---

## 참고문헌

**본 논문:**
- Cheng, J., Liu, T., Ramamohanarao, K., & Tao, D. (2020). *Learning with Bounded Instance- and Label-dependent Label Noise*. ICML 2020. arXiv:1709.03768v3

**논문 내 인용 주요 문헌:**
- Bartlett, P. L. & Mendelson, S. (2002). Rademacher and Gaussian complexities. *JMLR*
- Bartlett, P. L., Jordan, M. I., & McAuliffe, J. D. (2006). Convexity, classification, and risk bounds. *JASA*
- Huang, J., et al. (2007). Correcting sample selection bias by unlabeled data. *NeurIPS*
- Liu, T. & Tao, D. (2016). Classification with noisy labels by importance reweighting. *TPAMI*
- Menon, A. K., van Rooyen, B., & Natarajan, N. (2018). Learning from binary labels with instance-dependent noise. *Machine Learning*
- Natarajan, N., et al. (2013). Learning with noisy labels. *NeurIPS*
- Xia, X., et al. (2020). Parts-dependent label noise: Towards instance-dependent label noise. arXiv:2006.07836
- Li, J., Socher, R., & Hoi, S. C. (2020). DivideMix: Learning with noisy labels as semi-supervised learning. *ICLR*
- Zheng, S., et al. (2020). Error-bounded correction of noisy labels. *ICML*
- Massart, P. & Nédélec, E. (2006). Risk bounds for statistical learning. *The Annals of Statistics*

> **⚠️ 정확도 주의사항:** 2020년 이후 최신 연구 비교 분석 부분(CORES, Berthon et al. 등)은 제가 보유한 학습 데이터 기반으로 작성되었으며, 해당 논문들의 세부 내용에 대한 100% 정확성을 보장하기 어렵습니다. 실제 연구 활용 시에는 각 논문 원문을 직접 확인하시기 바랍니다.
