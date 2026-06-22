# Domain Adaptation with Conditional Distribution Matching and Generalized Label Shift

## 참고 자료

- **주 논문**: Tachet des Combes, R., Zhao, H., Wang, Y.-X., & Gordon, G. (2020). "Domain Adaptation with Conditional Distribution Matching and Generalized Label Shift." *NeurIPS 2020*. arXiv:2003.04475v3.
- **관련 참고 논문** (논문 내 인용 기준):
  - Zhao et al. (2019). "On learning invariant representations for domain adaptation." *ICML 2019*.
  - Ganin et al. (2016). "Domain-adversarial training of neural networks." *JMLR*.
  - Long et al. (2018). "Conditional adversarial domain adaptation." *NeurIPS 2018*.
  - Long et al. (2017). "Deep transfer learning with joint adaptation networks." *ICML 2017*.
  - Lipton et al. (2018). "Detecting and correcting for label shift with black box predictors." *ICML 2018*.
  - Ben-David et al. (2010). "A theory of learning from different domains." *Machine Learning*.

> **주의**: 2020년 이후 관련 최신 연구 비교 분석 항목에서, 논문 본문에 포함되지 않은 2020년 이후 문헌에 대해서는 제가 직접 확인한 범위 내에서만 기술하며, 불확실한 수치는 제시하지 않겠습니다.

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

기존 Adversarial Domain Adaptation(ADA) 알고리즘(DANN, CDAN 등)은 소스와 타깃 도메인 간 **레이블 분포가 다를 때(label distribution mismatch)** 근본적으로 성능 한계를 가진다. 이 논문은 이러한 한계를 이론적으로 규명하고, **Generalized Label Shift(GLS)** 라는 새로운 가정을 도입하여 레이블 분포 불일치에 강건한 도메인 적응 알고리즘을 제안한다.

### 주요 기여

| 기여 항목 | 내용 |
|-----------|------|
| **이론적 하한 확장** | Zhao et al. [66]의 하한을 $k$-class 분류 및 CDAN으로 확장 |
| **GLS 가정 도입** | 표현 공간에서의 레이블 시프트 일반화 |
| **오차 분해 정리** | GLS 하에서 소스-타깃 오차 간격에 대한 새로운 상한 도출 |
| **중요도 가중치 추정** | 이차계획법(QP)으로 클래스 가중치 $\mathbf{w}$ 추정 |
| **알고리즘 설계** | IWDAN, IWJAN, IWCDAN 세 가지 알고리즘 제안 |
| **실험 검증** | 4개 표준 DA 벤치마크 및 인공 태스크에서 성능 향상 확인 |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 향상 및 한계

### 2.1 해결하고자 하는 문제

**기존 ADA의 근본적 한계**: 소스와 타깃의 레이블 분포가 다를 때 불변 표현을 학습하는 것은 오히려 성능을 저하시킨다.

**정보이론적 하한 (Theorem 2.1)**:

$$
\varepsilon_S(h \circ g) + \varepsilon_T(h \circ g) \geq \frac{1}{2}\left(\sqrt{D_{\text{JS}}(\mathcal{D}_S^Y \| \mathcal{D}_T^Y)} - \sqrt{D_{\text{JS}}(\mathcal{D}_S^{\tilde{Z}} \| \mathcal{D}_T^{\tilde{Z}})}\right)^2
$$

여기서 $\tilde{Z} = Z$ (DANN) 또는 $\tilde{Z} = \hat{Y} \otimes Z$ (CDAN)이다.

이 하한은 **알고리즘 독립적**이며, 레이블 분포 차이 $D_{\text{JS}}(\mathcal{D}\_S^Y \| \mathcal{D}\_T^Y)$가 클수록 표현 분포를 완벽히 정렬하더라도 ( $D_{\text{JS}}(\mathcal{D}_S^{\tilde{Z}} \| \mathcal{D}_T^{\tilde{Z}}) = 0$ ) 결합 오차가 커진다는 것을 보인다.

**코바리에이트 시프트와 레이블 시프트의 한계**:

| 가정 | 정의 | 한계 |
|------|------|------|
| Covariate Shift | $\forall x: \mathcal{D}_S(Y\|X=x) = \mathcal{D}_T(Y\|X=x)$ | 특성 변환에 취약, 부정적 전이 발생 |
| Label Shift | $\forall y: \mathcal{D}_S(X\|Y=y) = \mathcal{D}_T(X\|Y=y)$ | 실제 응용에서 대부분 불성립 |

---

### 2.2 제안하는 방법

#### (1) Generalized Label Shift (GLS) 정의

**Definition 3.1**: 표현 $Z = g(X)$이 GLS를 만족하면:

$$
\mathcal{D}_S(Z \mid Y = y) = \mathcal{D}_T(Z \mid Y = y), \quad \forall y \in \mathcal{Y} \tag{2}
$$

GLS의 핵심 특성:
- $g$가 항등 함수이면 표준 Label Shift로 환원
- 완벽한 분류기 $h^\*$ (즉, $Y = h^*(X)$ )는 항상 GLS를 만족
- 임의의 분포 쌍 $(\mathcal{D}_S, \mathcal{D}_T)$에 대해 항상 달성 가능

#### (2) 오차 분해 정리 (Theorem 3.1)

임의의 분류기 $\hat{Y} = (h \circ g)(X)$에 대해:

$$
|\varepsilon_S(h \circ g) - \varepsilon_T(h \circ g)| \leq \|\mathcal{D}_S^Y - \mathcal{D}_T^Y\|_1 \cdot \text{BER}_{\mathcal{D}_S}(\hat{Y} \| Y) + 2(k-1)\Delta_{\text{CE}}(\hat{Y})
$$

여기서:
- $\|\mathcal{D}\_S^Y - \mathcal{D}\_T^Y\|\_1 = \sum_{i=1}^k |\mathcal{D}_S(Y=i) - \mathcal{D}_T(Y=i)|$: 레이블 분포 간 $L_1$ 거리
- $\text{BER}\_{\mathcal{D}\_S}(\hat{Y} \| Y) = \max_{j \in [k]} \mathcal{D}_S(\hat{Y} \neq Y \mid Y = j)$: 균형 오차율
- $\Delta_{\text{CE}}(\hat{Y}) = \max_{y \neq y'} |\mathcal{D}_S(\hat{Y}=y' \mid Y=y) - \mathcal{D}_T(\hat{Y}=y' \mid Y=y)|$: 조건부 오차 간격

**Theorem 3.2** (GLS 하에서의 결합 오차 상한):

$$
\varepsilon_S(\hat{Y}) + \varepsilon_T(\hat{Y}) \leq 2\,\text{BER}_{\mathcal{D}_S}(\hat{Y} \| Y)
$$

#### (3) GLS의 필요조건 (Lemma 3.1)

중요도 가중치를 다음과 같이 정의:

$$
\mathbf{w}_y := \frac{\mathcal{D}_T(Y=y)}{\mathcal{D}_S(Y=y)}, \quad \forall y \in \mathcal{Y} \tag{4}
$$

GLS가 성립하면:

$$
\mathcal{D}_T(\tilde{Z}) = \sum_{y \in \mathcal{Y}} \mathbf{w}_y \cdot \mathcal{D}_S(\tilde{Z}, Y=y) =: \mathcal{D}_S^{\mathbf{w}}(\tilde{Z})
$$

즉, 타깃 특성 분포를 소스의 **재가중 주변 분포**와 정렬해야 한다.

#### (4) 충분조건 (Theorem 3.4)

$$
\max_{y \in \mathcal{Y}} d_{\text{TV}}(\mathcal{D}_S(Z \mid Y=y),\, \mathcal{D}_T(Z \mid Y=y)) \leq \frac{\mathbf{w}_M \varepsilon_S(\hat{Y}) + \varepsilon_T(\hat{Y}) + \sqrt{8D_{\text{JS}}(\mathcal{D}_S^{\mathbf{w}}(\tilde{Z}) \| \mathcal{D}_T(\tilde{Z}))}}{\gamma}
$$

여기서 $\gamma = \min_{y} \mathcal{D}_T(Y=y)$, $\mathbf{w}_M = \max_y \mathbf{w}_y$.

#### (5) 중요도 가중치 추정 (이차계획법)

혼동 행렬 $\mathbf{C}$와 타깃 예측 분포 $\boldsymbol{\mu}$를 정의:

$$
\mathbf{C}_{y,y'} := \mathcal{D}_S(\hat{Y}=y, Y=y'), \quad \boldsymbol{\mu}_y := \mathcal{D}_T(\hat{Y}=y)
$$

**Lemma 3.2**: GLS가 성립하고 $\mathbf{C}$가 가역이면 $\mathbf{w} = \mathbf{C}^{-1}\boldsymbol{\mu}$.

수치적 안정성을 위해 이차계획법으로 추정:

$$
\min_{\mathbf{w}} \frac{1}{2}\|\hat{\boldsymbol{\mu}} - \hat{\mathbf{C}}\mathbf{w}\|_2^2, \quad \text{subject to} \quad \mathbf{w} \geq 0,\; \mathbf{w}^T \mathcal{D}_S(Y) = 1 \tag{5}
$$

시간 복잡도 $O(|\mathcal{Y}|^3)$으로 효율적으로 해결 가능.

#### (6) F-적분 확률 척도 (F-IPM) 프레임워크

$$
d_{\mathcal{F}}(\mathcal{D}, \mathcal{D}') := \sup_{f \in \mathcal{F}} |\mathbb{E}_{X \sim \mathcal{D}}[f(X)] - \mathbb{E}_{X \sim \mathcal{D}'}[f(X)]| \tag{6}
$$

$\mathcal{F}$ 선택에 따라 다양한 DA 알고리즘으로 인스턴스화 가능 (MMD, Wasserstein, GAN 등).

---

### 2.3 모델 구조

**마르코프 체인**: $X \xrightarrow{g} Z \xrightarrow{h} \hat{Y}$

세 가지 알고리즘 모두 동일한 **Algorithm 1 (Importance-Weighted Domain Adaptation)** 프레임워크를 따른다:

```
입력: 소스 레이블 데이터 (x_S, y_S), 타깃 비레이블 데이터 x_T
      네트워크: g_θ (특성 추출기), h_φ (분류기), d_ψ (판별기)

매 에폭 t마다:
  1. 배치 샘플링
  2. 재가중 DA 손실 L^w_DA 및 분류 손실 L^w_C 최적화
  3. 혼동행렬 Ĉ, 예측 분포 μ̂ 누적
  4. QP 풀어 w 업데이트: w_{t+1} = λ·QP(Ĉ, μ̂) + (1-λ)·w_t
```

#### IWDAN 손실 함수:

$$
\mathcal{L}^{\mathbf{w}}_{\text{DA}}(x_S^i, y_S^i, x_T^i;\theta,\psi) = -\frac{1}{s}\sum_{i=1}^s \mathbf{w}_{y_S^i}\log(d_\psi(g_\theta(x_S^i))) + \log(1 - d_\psi(g_\theta(x_T^i))) \tag{7}
$$

#### 재가중 분류 손실 (BER 최소화):

$$
\mathcal{L}^{\mathbf{w}}_C(x_S^i, y_S^i;\theta,\phi) = -\frac{1}{s}\sum_{i=1}^s \frac{1}{k \cdot \mathcal{D}_S(Y=y)}\log(h_\phi(g_\theta(x_S^i))_{y_S^i}) \tag{8}
$$

#### 세 알고리즘 비교:

| 알고리즘 | 기반 | 정렬 대상 | 판별기 입력 |
|---------|------|-----------|------------|
| **IWDAN** | DANN | $\mathcal{D}_S^{\mathbf{w}}(Z)$ vs $\mathcal{D}_T(Z)$ | $g_\theta(x)$ |
| **IWCDAN** | CDAN | $\mathcal{D}_S^{\mathbf{w}}(\hat{Y} \otimes Z)$ vs $\mathcal{D}_T(\hat{Y} \otimes Z)$ | $h_\phi(g_\theta(x)) \otimes g_\theta(x)$ |
| **IWJAN** | JAN | MMD 기반 재가중 | RKHS 커널 |

---

### 2.4 성능 향상

**표준 데이터셋 평균 성능 (Table 2 기준)**:

| 데이터셋 | DANN → IWDAN | CDAN → IWCDAN | JAN → IWJAN |
|---------|-------------|--------------|------------|
| Digits | 93.15 → 94.90 (+1.75%) | 95.72 → 95.90 (+0.18%) | N/A |
| VisDA | 61.88 → 63.52 (+1.64%) | 65.60 → 66.49 (+0.89%) | 56.98 → 57.56 (+0.58%) |
| Office-31 | 82.74 → 83.90 (+1.16%) | 87.23 → 87.30 (+0.07%) | 85.13 → 85.32 (+0.19%) |
| Office-Home | 59.62 → 62.27 (+2.65%) | 64.59 → 65.66 (+1.07%) | 59.59 → 59.78 (+0.19%) |

**서브샘플링(레이블 분포 불일치 강화) 시 성능 향상**:

| 데이터셋 | DANN → IWDAN | CDAN → IWCDAN |
|---------|-------------|--------------|
| sDigits | 83.24 → 92.54 (+**9.30%**) | 88.23 → 93.22 (+**4.99%**) |
| sVisDA | 52.85 → 60.18 (+**7.33%**) | 60.19 → 65.83 (+**5.64%**) |
| sOffice-31 | 76.17 → 82.60 (+**6.43%**) | 81.62 → 83.88 (+**2.26%**) |
| sOffice-Home | 51.83 → 57.61 (+**5.78%**) | 56.25 → 61.24 (+**4.99%**) |

- $D_{\text{JS}}(\mathcal{D}_S^Y \| \mathcal{D}_T^Y)$가 클수록 성능 향상이 선형적으로 증가 (Figure 1, 3)
- IWDAN은 100개 인공 태스크 전부에서 DANN을 능가
- Oracle 버전(IWDAN-O, IWCDAN-O)은 더 큰 향상을 보여 GLS 가정의 유효성 지지

---

### 2.5 한계

1. **가중치 수렴 미보장**: Lemma 3.2는 GLS가 성립할 때 $\mathbf{w} = \mathbf{C}^{-1}\boldsymbol{\mu}$임을 보이지만, 훈련 중 GLS가 항상 성립하지는 않으므로 참 가중치로의 수렴이 이론적으로 보장되지 않는다.
2. **클러스터링 가정의 강도**: Theorem 3.3의 충분조건은 특성 공간의 완벽한 클러스터 구조를 가정하며, 실제로는 이상적인 조건이다.
3. **하이퍼파라미터 민감성**: 지수이동평균 계수 $\lambda = 0.5$를 고정하여 사용했으나, 세밀한 튜닝이 필요할 수 있다.
4. **해석가능성 부재**: 딥러닝 기반이므로 수렴 보장 및 해석가능성에 한계가 있다.
5. **타깃 레이블 부재**: QP 추정이 소스 예측에 의존하므로 초기 분류기 성능이 낮으면 가중치 추정이 부정확하다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 이론적 기반

GLS 가정은 일반화 성능 향상의 강력한 이론적 근거를 제공한다.

**Theorem 3.2**에 의해, GLS가 성립하면:

$$
\varepsilon_S(\hat{Y}) + \varepsilon_T(\hat{Y}) \leq 2\,\text{BER}_{\mathcal{D}_S}(\hat{Y} \| Y)
$$

이는 **타깃 오차가 소스 도메인에서의 BER로만 결정**됨을 의미한다. 즉:
- 소스에서 균형 잡힌 분류 성능을 달성하면
- 타깃 도메인의 레이블 분포와 무관하게 우수한 타깃 성능이 보장된다

### 3.2 일반화를 향상시키는 메커니즘

#### (a) 조건부 분포 정렬

기존 방법이 주변 분포 $\mathcal{D}_S(Z)$와 $\mathcal{D}_T(Z)$를 정렬하는 반면, GLS 기반 방법은:

$$
\mathcal{D}_S(Z \mid Y=y) \approx \mathcal{D}_T(Z \mid Y=y), \quad \forall y
$$

를 목표로 한다. 이는 **클래스별 특성 분포**를 정렬하므로, 레이블 분포가 달라도 클래스 내 표현의 일관성이 유지된다.

#### (b) 균형 오차율(BER) 최소화

재가중 분류 손실(식 8)은 각 클래스에 $\frac{1}{k \cdot \mathcal{D}_S(Y=y)}$의 역가중치를 부여하여 **소수 클래스에 대한 과소학습을 방지**한다. 이는 클래스 불균형이 있는 실제 상황에서 일반화에 직접 기여한다.

#### (c) 레이블 분포 불일치에 대한 강건성

오차 분해(Theorem 3.1)에서:

$$
|\varepsilon_S - \varepsilon_T| \leq \underbrace{\|\mathcal{D}_S^Y - \mathcal{D}_T^Y\|_1}_{\text{고정 상수}} \cdot \underbrace{\text{BER}}_{\text{최소화 가능}} + \underbrace{2(k-1)\Delta_{\text{CE}}(\hat{Y})}_{\text{GLS 성립 시 0}}
$$

GLS가 성립하면 $\Delta_{\text{CE}}(\hat{Y}) = 0$이 되어:

$$
|\varepsilon_S - \varepsilon_T| \leq \|\mathcal{D}_S^Y - \mathcal{D}_T^Y\|_1 \cdot \text{BER}_{\mathcal{D}_S}(\hat{Y} \| Y)
$$

레이블 분포 차이가 상수로 고정되어 있으므로, BER만 최소화하면 타깃 오차도 제어된다.

### 3.3 일반화 성능 향상의 실증적 증거

- **Oracle 버전의 우수성**: IWDAN-O, IWCDAN-O가 일관되게 최고 성능을 달성하며, 이는 정확한 가중치 추정이 일반화에 결정적임을 시사
- **$D_{\text{JS}}$와 성능 향상의 선형 상관관계**: 레이블 분포 불일치가 클수록 GLS 기반 방법의 우위가 두드러짐
- **가중치 추정 정확도와 성능 상관**: Figure 2(우측)에서 가중치 추정 오차와 타깃 성능이 강한 음의 상관관계를 가짐

### 3.4 도메인 일반화로의 확장 가능성

논문 저자들이 명시적으로 언급한 미래 연구 방향:
- 다중 소스/타깃 도메인: 각 소스-타깃 쌍에 대해 독립적인 $\mathbf{w}$ 벡터 유지
- **도메인 일반화(Domain Generalization)**: 타깃 데이터가 전혀 없는 상황에서도 GLS 원리를 적용하는 것이 목표

---

## 4. 앞으로의 연구에 미치는 영향과 고려할 점

### 4.1 연구에 미치는 영향

#### (a) 이론적 영향

- **불변 표현 학습의 한계 명시화**: 레이블 분포 불일치가 있을 때 단순 불변 표현 학습이 실패함을 엄밀히 증명하여, 이후 연구의 이론적 기준점을 제공
- **새로운 가정 프레임워크**: GLS는 코바리에이트 시프트와 레이블 시프트 사이의 중간 가정으로, 더 현실적인 설정에서의 이론 개발을 촉진
- **IPM 기반 통합 프레임워크**: 다양한 DA 알고리즘을 단일 프레임워크로 통합하여 새로운 알고리즘 설계에 지침 제공

#### (b) 실용적 영향

- **플러그인 방식**: 기존 DA 알고리즘에 최소한의 계산 비용으로 적용 가능한 방법론 제시
- **실제 응용 관련성**: 의료 진단, 스팸 필터, 자율주행 등 클래스 불균형이 흔한 실제 시스템에 직접 적용 가능
- **공정성 측면**: 클래스 불균형으로 인한 특정 클래스에 대한 편향을 줄이는 데 기여 가능

### 4.2 2020년 이후 관련 연구 비교 분석

> **주의**: 아래 내용은 제가 알고 있는 범위 내의 연구 흐름에 대한 서술이며, 구체적 수치는 해당 논문을 직접 확인해야 합니다.

#### (a) Test-Time Adaptation (TTA) 방향

GLS 아이디어는 **Test-Time Adaptation** 연구와 자연스럽게 연결된다. 타깃 도메인에서 온라인으로 레이블 분포를 추정하고 모델을 적응시키는 방법들이 이 논문의 통찰과 유사한 방향으로 발전하였다. 예를 들어, **TENT** (Wang et al., ICLR 2021)는 타깃에서의 엔트로피 최소화를 통해 적응을 수행하지만, 레이블 분포 불일치를 명시적으로 다루지 않는다는 점에서 GLS 기반 방법이 보완적 역할을 할 수 있다.

#### (b) Source-Free Domain Adaptation

소스 데이터 없이 타깃 데이터만으로 적응하는 연구에서도 레이블 분포 추정 문제가 핵심 과제로 부상하였다. GLS의 QP 기반 가중치 추정 아이디어는 이러한 설정에서도 활용 가능성이 있다.

#### (c) Prompt/Foundation Model 기반 DA

대형 언어 모델 및 비전-언어 모델(CLIP 등)을 활용한 DA에서도 레이블 분포 불일치 문제는 여전히 중요하다. GLS의 조건부 분포 정렬 원리는 프롬프트 기반 적응 방법과 결합될 수 있다.

#### (d) 한계 대비 후속 연구 방향

| GLS의 한계 | 이후 연구 방향 |
|-----------|-------------|
| 가중치 수렴 미보장 | 더 강건한 추정 방법 (e.g., robust optimization) |
| 클러스터링 가정 강도 | 완화된 조건 하의 이론 개발 |
| 단일 소스-타깃 쌍 | 다중 도메인으로 확장 |

### 4.3 앞으로 연구 시 고려할 점

#### (a) 이론적 고려사항

1. **가중치 추정의 수렴 보장**: QP 기반 추정이 훈련 중 실제 가중치로 수렴하는 조건을 이론적으로 규명해야 한다. 현재는 점근적 일관성(asymptotic consistency)만 보장된다.

2. **GLS의 검증 가능성**: 주어진 데이터에서 GLS가 성립하는지 사전에 검증하는 방법이 필요하다. 이는 알고리즘 선택에 중요한 정보를 제공한다.

3. **클러스터링 가정 완화**: Theorem 3.3의 충분조건은 완벽한 클러스터 구조를 요구하는데, 이를 $\varepsilon$-근사 클러스터로 완화한 이론이 필요하다.

4. **유한 샘플 보장**: 현재 이론은 대부분 점근적 결과이므로, 유한 샘플에서의 통계적 보장 도출이 필요하다.

#### (b) 방법론적 고려사항

1. **동적 가중치 업데이트 전략**: $\lambda = 0.5$로 고정된 지수이동평균보다 적응적인 업데이트 전략(예: 훈련 단계에 따른 스케줄링)이 성능을 향상시킬 수 있다.

2. **소수 클래스 처리**: 가중치 $\mathbf{w}_y$가 매우 작은 클래스(소스에서 과대표집된 경우)에 대한 처리가 수치적으로 불안정할 수 있다.

3. **준지도 DA로의 확장**: 일부 타깃 레이블이 가용할 때 $\mathcal{D}_T^Y$를 직접 추정하여 더 정확한 가중치를 얻는 방법을 고려해야 한다.

4. **다중 소스 DA**: 각 소스-타깃 쌍에 대한 가중치 벡터를 공동으로 최적화하는 방법이 필요하다.

#### (c) 실용적 고려사항

1. **레이블 분포 사전 지식**: 실제 응용에서는 타깃 레이블 분포에 대한 사전 지식이 가용한 경우가 있으므로, 이를 활용하는 반지도 방식의 GLS 적용을 고려할 수 있다.

2. **연산 효율성**: 대규모 클래스 수(예: $k > 1000$)에서는 QP의 계산 비용이 무시할 수 없게 되므로, 근사 방법이 필요할 수 있다.

3. **분포 시프트의 복합성**: 실제 시나리오에서는 레이블 시프트 외에도 코바리에이트 시프트가 동시에 발생하므로, 이를 복합적으로 처리하는 방법이 필요하다.

4. **공정성과 편향 완화**: 특정 클래스에 대한 가중치 추정 오류가 해당 클래스에 대한 편향으로 이어질 수 있으므로, 공정성 제약을 QP에 명시적으로 추가하는 방향도 고려해야 한다.

---

## 핵심 수식 요약

| 수식 | 의미 |
|------|------|
| $\mathcal{D}_S(Z \mid Y=y) = \mathcal{D}_T(Z \mid Y=y)$ | GLS 정의 |
| $\mathbf{w}_y = \mathcal{D}_T(Y=y)/\mathcal{D}_S(Y=y)$ | 중요도 가중치 |
| $\min_{\mathbf{w}} \frac{1}{2}\|\hat{\boldsymbol{\mu}} - \hat{\mathbf{C}}\mathbf{w}\|_2^2$ | QP 최적화 |
| $\varepsilon_S + \varepsilon_T \leq 2\,\text{BER}_{\mathcal{D}_S}$ | GLS 하의 결합 오차 상한 |
| $\mathcal{D}_T(\tilde{Z}) = \mathcal{D}_S^{\mathbf{w}}(\tilde{Z})$ | GLS 필요조건 |
