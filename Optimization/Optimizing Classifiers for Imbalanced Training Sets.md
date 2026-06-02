# Optimizing Classifiers for Imbalanced Training Sets

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

Karakoulas & Shawe-Taylor (1998, NeurIPS)의 이 논문은 **클래스 불균형(class imbalance) 문제**에서 분류 임계값(threshold)을 최적화하는 이론적 프레임워크를 제시합니다. 핵심 주장은 다음과 같습니다:

> *"fat-shattering 차원(dimension)과 대마진(large margin)의 관계를 활용하면, 불균형 데이터셋에서 비대칭 손실(unequal loss)을 고려하여 최적 임계값을 이론적으로 도출할 수 있다."*

### 주요 기여

| 기여 | 내용 |
|------|------|
| 이론적 프레임워크 | fat-shattering 차원 기반 불균형 데이터 일반화 오차 경계 도출 |
| AdaUBoost | 비대칭 손실함수를 지원하는 AdaBoost 확장 |
| ThetaBoost | 마진 조정 기능을 추가한 부스팅 알고리즘 |
| 임계값 결정 방법 2종 | 경험적 방법 (수식 1)과 이론 기반 방법 (수식 2) 제안 |
| 성능 지표 | g-mean을 불균형 데이터 평가 지표로 적극 활용 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

불균형 데이터셋에서는 두 가지 핵심 문제가 발생합니다:

1. **클래스 불균형으로 인한 편향**: 다수 클래스(majority class)에 편향된 예측
2. **비대칭 손실의 무시**: 소수 클래스(minority class)의 오분류 손실이 더 클 때, 표준 알고리즘은 이를 반영하지 못함

예를 들어, 금융 사기 탐지, 의료 진단 등에서 소수 클래스(사기, 양성)를 놓치는 비용이 훨씬 크지만, 표준 AdaBoost는 동등 손실을 가정하므로 부적절합니다.

---

### 2.2 이론적 배경: Fat-Shattering 차원

**Definition 2.1 (Fat-Shattering Dimension)**

집합 $\mathcal{F}$가 점집합 $X$를 $\gamma$-shatter한다는 것은:

$$\exists \{r_x\}_{x \in X} \subset \mathbb{R}, \; \forall b \in \{0,1\}^{|X|}, \; \exists f_b \in \mathcal{F}: \; y \cdot f_b(x) \geq \gamma$$

이 만족될 때이며, $\text{Fat}_{\mathcal{F}}(\gamma)$는 $\gamma$-shattered 가능한 최대 집합의 크기입니다.

**Theorem 2.4 (Shawe-Taylor, 1998)**

함수 클래스 $\mathcal{F}$가 상수 덧셈에 대해 닫혀 있고, fat-shattering 차원이 $\text{Fat}_{\mathcal{F}}(\gamma)$로 유계일 때, 확률 $1-\delta$ 이상으로 마진 $\eta$를 가지는 새 예제의 오분류 확률은:

$$\text{er}_P(f|\eta) \leq \frac{3}{n} \left(2d \log_2(288m) \log_2(12em) + \log_2 \frac{32m^2}{\delta} \right)$$

여기서 $d = \text{Fat}_{\mathcal{F}}(\gamma/6)$, $n$은 마진 $\eta + 2\gamma$ 이상을 가지는 훈련 예제 수입니다.

---

### 2.3 제안 방법: 비대칭 손실함수

**Definition 3.1 (비대칭 손실함수 $L^\beta$)**

$$L^\beta(x, y) = \begin{cases} \beta y + (1 - y) & \text{if } h(x) \neq y \\ 0 & \text{otherwise} \end{cases}$$

- $y = +1$ (양성 클래스) 오분류 시 손실: $\beta$
- $y = -1$ (음성 클래스) 오분류 시 손실: $1$
- $\beta > 1$이면 소수 양성 클래스에 더 높은 손실 부여

**경험적 리스크:**

$$ER(h) = \sum_{i=1}^{m} L^\beta(x_i, y_i)$$

---

### 2.4 임계값 결정: 두 가지 접근법

#### 접근법 1: 경험적 임계값 $b$ (수식 1)

최대마진 초평면 $\mathbf{w}$를 구한 후, 양성/음성 지지벡터를 이용해 임계값을 조정:

$$b = \frac{1}{1 + \beta}\left[(\mathbf{w} \cdot x^+) + \beta(\mathbf{w} \cdot x^-)\right] \tag{1}$$

- $x^+$: 결정 경계에 가장 가까운 양성 예제 (positive support vector)
- $x^-$: 결정 경계에 가장 가까운 음성 예제 (negative support vector)
- 마진 비율: 양성 측 마진 = 음성 측 마진의 $\beta$배

#### 접근법 2: 이론 기반 마진 조정 $\eta$ (수식 2)

Theorem 3.2를 기반으로 fat-shattering 차원 최소화를 통해 최적 마진 이동량 $\eta$ 도출:

$$\eta = \gamma_0 \left(\frac{\sqrt[3]{\beta} - 1}{\sqrt[3]{\beta} + 1}\right) \tag{2}$$

- $\gamma_0$: 최대마진 초평면의 마진
- $\beta > 1$일 때 $\eta > 0$: 초평면을 음성 클래스 방향으로 이동

**Theorem 3.2의 기대 손실 상한:**

$$\frac{3}{n_0}\left(2\max(\beta d^+, d^-) \log_2(288m)\log_2(12em) + \beta \log_2\frac{32m^2}{\delta}\right)$$

여기서:
- $\gamma^+ = (\gamma_0 + \eta)/2$, $\gamma^- = (\gamma_0 - \eta)/2$
- $d^+ = \text{Fat}\_{\mathcal{F}}(\gamma^+/6)$, $d^- = \text{Fat}_{\mathcal{F}}(\gamma^-/6)$

---

### 2.5 모델 구조: AdaUBoost & ThetaBoost

#### AdaUBoost 알고리즘

초기 가중치를 비대칭으로 설정:
- 양성 예제: $D_1(i) = w^+$
- 음성 예제: $D_1(i) = w^-$
- 조건: $w^+ n^+ + w^- n^- = 1$, $w^+/w^- = \beta$

가중치 업데이트 규칙:

$$D_{t+1}(i) = \frac{D_t(i) \exp(-\alpha_t \beta_i y_i h(x_i))}{Z_t}$$

여기서 $\beta_i = 1/\beta$ (양성 예제), $\beta_i = 1$ (음성 예제)

정규화 인자를 최소화하는 $\alpha_t$ 결정:

$$Z = \sum_i D(i) \exp(-\alpha \beta_i y_i h(x_i)) \tag{4}$$

$Z$를 $\alpha$에 대해 최소화하면, $Y = \exp(\alpha)$에 대한 다항식:

$$C_1 Y^{1-1/\beta} + C_2 Y^{1+1/\beta} + C_3 Y^2 + C_4 = 0 \tag{6}$$

여기서:
- $C_1 = -W_{++}/\beta$
- $C_2 = W_{-+}/\beta$
- $C_3 = W_{+-}$
- $C_4 = -W_{--}$

최종 분류기:

$$H(x) = \text{sgn}\left(\sum_{t=1}^{T} \alpha_t h_t(x)\right)$$

#### ThetaBoost 알고리즘

```
1. H(x) = AdaUBoost(X, Y, β)
2. 훈련셋에서 false positive 및 경계점 제거
3. 최소 H(x+)를 SV+로 표시; H(SV+)보다 큰 음성점 제거
4. SV+ 다음 순위의 첫 음성점을 SV-로 표시; 마진 계산
5. 마진이 δM 이상 변하는 SV- 후보 검사
6. SV+, SV-로부터 수식 (1), (2)를 사용해 θ 계산
7. 최종 출력: H(x) = sgn(Σ αt ht(x) - θ)
```

---

### 2.6 성능 평가

**평가 지표:**

$$g = \sqrt{\text{precision} \cdot \text{recall}}$$

```math
\text{precision} = \frac{\text{\# positives correct}}{\text{\# positives predicted}}, \quad \text{recall} = \frac{\text{\# positives correct}}{\text{\# true positives}}
```

**실험 결과 (SatImage 데이터셋, 양성 클래스 비율 9.73%):**

| $\beta$ | AdaUBoost g-mean | AdaBoost g-mean | C4.5 g-mean |
|---------|-----------------|-----------------|-------------|
| 1 | 0.773 | 0.773 | 0.724 |
| 2 | **0.865** | 0.773 | 0.724 |
| 4 | **0.889** | 0.773 | 0.724 |
| 8 | **0.898** | 0.773 | 0.724 |
| 16 | **0.890** | 0.773 | 0.724 |

**주요 발견:**
- $\beta$가 클수록 AdaUBoost의 g-mean이 크게 향상
- ThetaBoost의 $b$ 방식이 $\eta$ 방식보다 실제로 더 나은 성능
- 단, 실험 규모가 작아 통계적 유의성 확인 필요

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 이론적 기반

논문의 핵심 통찰은 **마진이 크면 일반화 오차 상한이 낮아진다**는 것입니다. Theorem 2.4에서:

$$\text{일반화 오차} \leq \frac{3}{n}\left(2d\log_2(288m)\log_2(12em) + \log_2\frac{32m^2}{\delta}\right)$$

이 상한은:
- $n$ (충분한 마진을 가진 훈련 예제 수)이 클수록 감소 ✓
- $d = \text{Fat}_{\mathcal{F}}(\gamma/6)$가 작을수록 감소 ✓
- $m$ (훈련 데이터 수)이 클수록 안정화 ✓

### 3.2 불균형 데이터에서의 마진 최적화

임계값 이동 $\eta$를 통해:

- **양성 클래스 마진**: $\gamma^+ = (\gamma_0 + \eta)/2$ → 증가
- **음성 클래스 마진**: $\gamma^- = (\gamma_0 - \eta)/2$ → 감소

이때 $\beta d^+ = \beta \cdot \text{Fat}_{\mathcal{F}}(\gamma^+/6)$를 최소화하는 방향으로 $\eta$를 선택하면 기대 손실 상한이 최적화됩니다. 수식 (2)는 이 최적화 조건에서 도출됩니다.

### 3.3 일반화 성능 향상의 메커니즘

```
[마진 이동 메커니즘]
표준 임계값 0
       |
음성  ←|→ 양성
  γ₀  |  γ₀

임계값 θ로 이동 (β > 1일 때 양성 방향으로)
           θ
       |   |
음성  ←|---|→ 양성
  γ⁻  |   |  γ⁺ (γ⁺ > γ⁻)
```

- 소수 클래스(양성)의 마진을 확대 → 양성 예제 오분류 가능성 감소
- fat-shattering 차원을 비대칭적으로 제어 → 이론적 일반화 보장

### 3.4 한계점

| 한계 | 상세 내용 |
|------|----------|
| **실험 규모** | 단일 데이터셋(SatImage)만 사용, 통계적 유의성 불충분 |
| **$\beta$ 설정** | 최적 $\beta$ 결정 방법 미제시, 도메인 지식 필요 |
| **이론-실험 괴리** | 수식 (1)과 (2)의 결과가 서로 다르며, 어떤 방법이 우월한지 불명확 |
| **선형 가정** | 초평면이 최대마진 초평면과 평행하다는 가정, 비선형 경우 확장 필요 |
| **계산 복잡도** | 다항식 (6)의 수치적 풀이가 필요, 고차원에서 비용 증가 |
| **이론적 경계의 느슨함** | PAC 기반 상한이 실제 성능보다 훨씬 느슨할 수 있음 |

---

## 4. 후속 연구에의 영향 및 고려사항

### 4.1 후속 연구에 미친 영향

이 논문은 불균형 학습 분야의 선구적 이론 연구로서 다음 방향에 영향을 미쳤습니다:

#### (1) 비용 민감 학습(Cost-Sensitive Learning) 확장

AdaUBoost의 비대칭 가중치 아이디어는 이후 비용 민감 앙상블 방법론의 기반이 되었습니다:
- **AdaCost** (Fan et al., 1999): 오분류 비용을 직접 가중치에 반영
- **MetaCost** (Domingos, 1999): 사후 확률에 비용을 적용하는 메타 학습

#### (2) 임계값 이동(Threshold Moving) 연구

수식 (1), (2)의 임계값 최적화 아이디어는 후속 연구에서 광범위하게 활용:
- 다양한 성능 지표(F1, AUC 등)에 대한 임계값 최적화 연구
- ROC 분석과 임계값 선택의 연결

#### (3) 불균형 학습 평가 지표 정착

g-mean을 성능 지표로 적극 활용한 것이 후속 연구에서 이 지표를 표준화하는 데 기여했습니다.

---

### 4.2 2020년 이후 최신 연구 비교 분석

#### 4.2.1 데이터 수준(Data-level) 방법론의 발전

| 논문 | 방법 | 특징 |
|------|------|------|
| SMOTE-NC (2021) | 범주형+수치형 혼합 오버샘플링 | 본 논문의 가중치 방식 대비 데이터 증강 |
| ADASYN 변형 (2020~) | 적응형 오버샘플링 | 경계 근방 샘플 집중 생성 |

#### 4.2.2 알고리즘 수준(Algorithm-level) 방법론

**Focal Loss (Lin et al., RetinaNet에서 도입, 이후 NLP/표 데이터 확장)**

$$FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

본 논문의 $L^\beta$와 유사하게 소수 클래스에 높은 가중치를 부여하나, 신뢰도 기반으로 동적 조정한다는 점이 차별화됩니다.

**Class-Balanced Loss (Cui et al., CVPR 2019 → 2020년 이후 확산)**

$$w_y = \frac{1 - \beta}{1 - \beta^{n_y}}$$

유효 샘플 수(effective number of samples) 개념을 도입하여 클래스별 가중치를 이론적으로 결정합니다. 본 논문의 $\beta$ 파라미터 설정 한계를 데이터 기반으로 해결하려는 시도입니다.

#### 4.2.3 딥러닝 기반 불균형 학습 (2020년 이후)

| 연구 | 핵심 아이디어 | 본 논문과의 관계 |
|------|-------------|----------------|
| **MiSLAS** (Zhong et al., CVPR 2021) | 마진 조정 + 레이블 스무딩 결합 | 본 논문의 마진 최적화 아이디어와 유사 |
| **LDAM Loss** (Cao et al., NeurIPS 2019) | 소수 클래스에 더 큰 마진 강제 부여 | 수식 (2)의 비대칭 마진 아이디어의 딥러닝 버전 |
| **LogitAdjustment** (Menon et al., ICLR 2021) | 사후 확률 보정을 통한 임계값 조정 | 수식 (1)의 임계값 이동과 개념적 연결 |
| **BBN** (Zhou et al., CVPR 2020) | 양방향 분기 네트워크 | 클래스별 다른 처리를 구조적으로 구현 |

**LDAM Loss 상세 비교:**

$$\mathcal{L}_{\text{LDAM}}(f(x), y) = -\log \frac{e^{z_y - \Delta_y}}{e^{z_y - \Delta_y} + \sum_{j \neq y} e^{z_j}}$$

여기서 $\Delta_y = \frac{C}{n_y^{1/4}}$ ($n_y$: 클래스 $y$의 샘플 수)

> 본 논문의 $\eta = \gamma_0\left(\frac{\sqrt[3]{\beta}-1}{\sqrt[3]{\beta}+1}\right)$와 유사하게, 클래스 불균형 정도에 따라 마진을 비대칭적으로 설정한다는 핵심 철학을 공유합니다.

#### 4.2.4 Transformer/LLM 시대의 불균형 학습 (2022~)

- **Imbalanced Fine-tuning of LLMs**: ChatGPT, LLaMA 등 대규모 언어모델의 파인튜닝 시 클래스 불균형 처리
- **Prompt-based 방법**: 인컨텍스트 러닝(ICL)에서 불균형 처리
- 본 논문의 이론적 프레임워크는 딥러닝에 직접 적용하기 어려우나, **마진 기반 사고방식**은 여전히 유효

---

### 4.3 앞으로의 연구 시 고려사항

#### (1) 이론적 측면

- **더 타이트한 일반화 경계**: PAC-Bayesian 이론, Rademacher 복잡도를 활용한 비대칭 손실에 대한 더 정밀한 경계 도출
- **비선형 분류기로의 확장**: 커널 SVM, 딥 신경망에서의 fat-shattering 차원 대응 개념 연구
- **동적 $\beta$ 결정**: 데이터 기반으로 $\beta$ 또는 손실 가중치를 자동으로 결정하는 이론 개발

#### (2) 방법론적 측면

- **다중 클래스 불균형 처리**: 이진 분류에서 다중 클래스(multi-class)로의 자연스러운 확장
- **분포 이동(Distribution Shift) 고려**: 훈련과 테스트 분포가 다를 때의 임계값 결정
- **불균형 비율에 따른 적응적 알고리즘**: 극단적 불균형(예: 1:1000 이상)에서의 안정성

#### (3) 실용적 측면

- **$\beta$ 파라미터 튜닝**: 교차 검증이나 베이지안 최적화를 통한 체계적 탐색
- **평가 지표 다양화**: g-mean 외에도 AUC-PR (Precision-Recall 곡선 아래 면적)이 불균형 데이터에서 더 정보적일 수 있음
- **실제 도메인 적용**: 의료 진단, 이상 탐지, 금융 사기 탐지에서의 $\beta$ 설정은 도메인 전문가와 협력 필요

#### (4) 현대 딥러닝과의 결합

- **마진 기반 손실(Margin-based Loss)**과 임계값 이동을 딥러닝 학습에 통합
- **자기지도 학습(Self-Supervised Learning)** 전처리 후 불균형 파인튜닝 전략
- **LLM의 Instruction Tuning**에서 불균형 태스크 처리 시 이 논문의 비대칭 손실 철학 적용 가능

---

## 참고 자료

**논문 자체 참고문헌 (논문 내 인용):**
1. Karakoulas, G. & Shawe-Taylor, J. (1998). "Optimizing Classifiers for Imbalanced Training Sets." *NeurIPS 1998* (NIPS-1998)
2. Cortes, C. & Vapnik, V. (1995). "Support-Vector Networks." *Machine Learning*, 20, 273-297
3. Freund, Y. & Schapire, R. (1996). *ICML '96*
4. Kearns, M.J. & Schapire, R.E. (1990). *FOCS'90*
5. Kubat, M., Holte, R. & Matwin, S. (1998). *Machine Learning*, 30, 195-215
6. Schapire, R., Freund, Y., Bartlett, P. & Lee, W.S. (1997). *ICML '97*
7. Schapire, R. & Singer, Y. (1998). *COLT'98*
8. Shawe-Taylor, J. (1998). *Algorithmica*, 22, 157-172
9. Shawe-Taylor, J., Bartlett, P., Williamson, R. & Anthony, M. (1998). *IEEE Trans. Inf. Theory*, 44(5), 1926-1940

**2020년 이후 비교 연구 (내 지식 기반, 직접 논문 확인 권장):**
- Cao, K., Wei, C., Gaidon, A., Aréchiga, N., & Ma, T. (2019). "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss." *NeurIPS 2019*
- Menon, A.K., Jayasumana, S., Rawlinson, D., Bhatt, H., Rawat, A., & Kumar, S. (2021). "Long-tail learning via logit adjustment." *ICLR 2021*
- Zhong, Z., Cui, J., Liu, S., & Jia, J. (2021). "Improving Calibration for Long-Tail Recognition." *CVPR 2021*
- Zhou, B., Cui, Q., Wei, X.S., & Chen, Z.M. (2020). "BBN: Bilateral-Branch Network with Cumulative Learning for Long-Tailed Visual Recognition." *CVPR 2020*
- Cui, Y., Jia, M., Lin, T.Y., Song, Y., & Belongie, S. (2019). "Class-Balanced Loss Based on Effective Number of Samples." *CVPR 2019*
- Lin, T.Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017/2020 확산). "Focal Loss for Dense Object Detection." *ICCV 2017*

> **주의**: 2020년 이후 최신 연구 부분은 제 훈련 데이터 기반 지식이며, 인용 정보의 세부 사항(페이지, 권호 등)은 직접 확인을 권장합니다.
