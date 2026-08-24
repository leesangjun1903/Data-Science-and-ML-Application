# Newton–Raphson and EM Algorithms for Linear Mixed-Effects Models for Repeated-Measures Data
**Lindstrom & Bates (1988), Journal of the American Statistical Association, Vol. 83, No. 404, pp. 1014–1022**

---

## 1. Executive Summary (10문장 이내)

Lindstrom과 Bates(1988)는 반복측정 데이터(repeated-measures data)를 위한 선형 혼합효과 모델(linear mixed-effects model)에서 분산 성분(variance components)을 추정하기 위한 효율적인 Newton-Raphson(NR) 알고리즘 구현을 제안한다.  
기존 EM 알고리즘(Laird & Ware 1982)과 Jennrich & Schluchter(1986)의 NR 알고리즘의 한계를 극복하고자 네 가지 핵심 개선점을 제시한다.  
핵심 개선점은 ① $\sigma$를 계산에서 분리(profile likelihood 최적화), ②조건부 선형성을 활용한 $\boldsymbol{\beta}$의 GLS 추정값 대체, ③QR 분해 등 행렬 분해 기법 활용, ④D의 Cholesky 인수 $\mathbf{L}$을 최적화 변수로 사용하는 재모수화(reparameterization)이다.  
이를 통해 양정치(positive-definite) 공분산 행렬을 보장하고 수치적 안정성을 향상시킨다.  
ML(최대우도) 및 REML(제한 최대우도) 모두에 대한 도함수 공식을 유도하였다.  
두 개의 실제 데이터셋(말 난포 예시, 골밀도 예시)을 사용하여 NR, EM, EM+Aitken 가속 알고리즘을 비교하였다.  
결과적으로 잘 구현된 NR 알고리즘이 대부분의 상황에서 EM 알고리즘보다 우월함을 보였다.  
NR 알고리즘은 훨씬 적은 반복 횟수로 수렴하며, 수렴 진단(convergence criterion)도 명확히 제공할 수 있다.  
논문은 또한 직렬 상관 오차 구조(serially correlated errors)와 중첩 설계(nested designs) 등으로의 모델 확장 방향도 논의한다.

---

### 1-1. 연구의 목적과 필요성

**목적:**
반복측정 데이터를 위한 선형 혼합효과 모델에서 ML 및 REML 추정을 위한 **수치적으로 안정적이고 계산 효율이 높은 NR 알고리즘**을 개발하고, 기존 EM 알고리즘과 체계적으로 비교하는 것이다.

**필요성:**
- 반복측정 데이터는 임상, 종단 연구 등에서 빈번히 발생하며, 불균형 설계(unbalanced design)와 결측값(missing data)을 동시에 처리해야 한다.
- 기존 EM 알고리즘(Laird & Ware 1982)은 수렴이 매우 느리고, Jennrich & Schluchter(1986)의 NR 알고리즘은 수치 불안정성(D의 비양정치 추정 문제) 및 느린 수렴 문제를 가지고 있었다.
- 분산 성분의 추정은 양정치 제약 조건이 있어 단순한 최적화가 어렵고, 당시에는 이를 효과적으로 처리하는 방법이 부족하였다.

> 💡 **반복측정 데이터(Repeated-Measures Data)**: 동일한 개체(individual)에 대해 여러 시점에 걸쳐 반복 측정된 데이터. 예: 환자의 혈압을 매주 측정, 학생의 성적을 학기마다 측정.

> 💡 **혼합효과 모델(Mixed-Effects Model)**: 집단 전체에 적용되는 고정효과(fixed effects)와 개체별로 다르게 작용하는 랜덤효과(random effects)를 동시에 포함하는 통계 모델.

> 💡 **ML(Maximum Likelihood, 최대우도법)**: 관측된 데이터가 나올 확률(우도)을 최대화하는 파라미터를 추정하는 방법.

> 💡 **REML(Restricted Maximum Likelihood, 제한 최대우도법)**: ML이 분산 성분을 과소추정(downward bias)하는 문제를 보정하기 위해, 고정효과를 제거한 오차 대조(error contrasts)를 기반으로 분산 성분을 추정하는 방법.

---

## 2. 핵심 주장과 근거 표

| 핵심 주장 | 근거 | 위치 |
|---|---|---|
| NR 알고리즘이 EM보다 적은 반복으로 수렴 | 말 난포 예: NR 4회 vs EM 50회; 골밀도(복잡 모델): NR 6회 vs EM >200회 | Table 1 (p.1020) |
| Profile likelihood 최적화가 수렴 안정성 향상 | $\sigma$를 제거한 profile likelihood로 반복 횟수 감소 및 수렴 실패 사례 감소 | p.1015, Sec. 2 |
| Cholesky 재모수화로 양정치 보장 | $\mathbf{L}^T\mathbf{L}=\mathbf{D}$로 변환해 제약 없는 최적화 가능 | p.1015, Sec. 2 |
| QR 분해로 계산 효율 개선 | EM 반복 계산 차수: $M(q^3+q^2p)$ (기존 $O(N)$에서 감소) | p.1018–1019, Sec. 4-5 |
| EM 알고리즘은 평탄한 우도 면에서 수렴 실패 | 이차항 추가 골밀도 모델에서 200회 반복 후에도 EM 미수렴 | Table 1, p.1020 |
| NR은 헤시안(Hessian)을 통한 표준오차 추정 가능 | 수렴 후 역헤시안이 $\boldsymbol{\beta}$와 $\boldsymbol{\theta}$의 분산-공분산 행렬 추정값 제공 | p.1016, p.1021 |
| EM의 수렴 진단 기준이 부재 | EM은 우도/파라미터 변화량만 측정 가능; NR은 직교성 기준(orthogonality criterion) 사용 가능 | p.1021, Sec. 7.1 |

> 💡 **Cholesky 분해(Cholesky Decomposition)**: 양정치 대칭 행렬 $\mathbf{D}$를 $\mathbf{D} = \mathbf{L}^T\mathbf{L}$ 형태로 분해하는 것. $\mathbf{L}$은 상삼각행렬(upper triangular matrix). 이를 통해 $\mathbf{D}$의 양정치 제약을 자동으로 만족시킬 수 있다.

> 💡 **양정치(Positive-Definite)**: 행렬 $\mathbf{D}$가 양정치라는 것은 임의의 영벡터가 아닌 벡터 $\mathbf{x}$에 대해 $\mathbf{x}^T\mathbf{D}\mathbf{x} > 0$임을 의미. 공분산 행렬은 반드시 양정치(또는 반양정치)이어야 한다.

---

## 2-1. 상세 설명

### 해결하고자 하는 문제

1. **계산 비효율성**: 기존 EM 알고리즘은 수렴에 수십~수백 번의 반복이 필요하고, $n_i \times n_i$ 행렬 역산이 필요해 계산 비용이 큼.
2. **수치 불안정성**: 랜덤효과의 공분산 행렬 $\mathbf{D}$ 추정 시 비양정치(non-positive-definite) 행렬이 산출될 수 있음.
3. **수렴 진단의 어려움**: EM은 객관적 수렴 기준이 없어 과소/과다 반복의 위험이 있음.
4. **평탄한 우도 면에서의 수렴 실패**: EM은 복잡한 모델(모수가 많아 우도 면이 평탄한 경우)에서 실패하는 경우가 있음.

---

### 제안하는 방법 (수식 포함)

#### 기본 모델 (Section 1, p.1014)

개체 $i$의 관측값 벡터 $\mathbf{y}_i$에 대해:

$$\mathbf{y}_i \mid \mathbf{b}_i \sim \mathcal{N}(\mathbf{X}_i\boldsymbol{\beta} + \mathbf{Z}_i\mathbf{b}_i,\ \sigma^2\mathbf{\Lambda}_i), \quad i = 1, \ldots, M$$

$$\mathbf{b}_i \sim \mathcal{N}(\mathbf{0},\ \sigma^2\mathbf{D})$$

전체 데이터의 주변 분포(marginal distribution):

$$\mathbf{y} \sim \mathcal{N}(\mathbf{X}\boldsymbol{\beta},\ \boldsymbol{\Sigma}), \quad \boldsymbol{\Sigma} = \sigma^2(\mathbf{\Lambda} + \mathbf{Z}\mathbf{D}\mathbf{Z}^T) $$

**기호 설명:**
- $\mathbf{y}_i$: 개체 $i$의 $n_i \times 1$ 관측값 벡터
- $\mathbf{X}_i$: $n_i \times p$ 고정효과 설계 행렬(design matrix for fixed effects)
- $\boldsymbol{\beta}$: $p \times 1$ 고정효과 파라미터 벡터
- $\mathbf{Z}_i$: $n_i \times q$ 랜덤효과 설계 행렬(design matrix for random effects)
- $\mathbf{b}_i$: $q \times 1$ 랜덤효과 벡터
- $\sigma^2$: 오차 분산(error variance)
- $\mathbf{\Lambda}_i$: $n_i \times n_i$ 조건부 오차의 상관 구조 행렬 (독립 오차 가정 시 $\mathbf{I}$)
- $\mathbf{D}$: $q \times q$ 랜덤효과의 (scaled) 공분산 행렬
- $M$: 총 개체 수

> 💡 **고정효과(Fixed Effects)**: 모든 개체에 공통으로 적용되는 효과. 예: 처리 집단 간의 평균 차이.

> 💡 **랜덤효과(Random Effects)**: 개체마다 다르게 나타나는 효과로, 특정 확률분포에서 추출된 것으로 가정.

> 💡 **주변 분포(Marginal Distribution)**: 랜덤효과 $\mathbf{b}_i$를 적분(평균)하여 제거한 $\mathbf{y}_i$만의 분포.

---

#### ML 및 REML 로그우도 (Section 1, p.1014)

**ML 로그우도:**

$$l_F(\boldsymbol{\beta}, \sigma, \boldsymbol{\theta} \mid \mathbf{y}) = -\frac{1}{2}\log|\sigma^2\mathbf{V}| - \frac{1}{2}\sigma^{-2}(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^T\mathbf{V}^{-1}(\mathbf{y} - \mathbf{X}\boldsymbol{\beta}) $$

**REML 로그우도:**

$$l_R(\hat{\boldsymbol{\beta}}(\boldsymbol{\theta}), \sigma, \boldsymbol{\theta} \mid \mathbf{y}) = -\frac{1}{2}\log|\sigma^{-2}\mathbf{X}^T\mathbf{V}^{-1}\mathbf{X}| + l_F(\hat{\boldsymbol{\beta}}(\boldsymbol{\theta}), \sigma, \boldsymbol{\theta} \mid \mathbf{y}) $$

**기호 설명:**
- $\mathbf{V} = \mathbf{\Lambda} + \mathbf{Z}\mathbf{D}\mathbf{Z}^T$: (scaled) 주변 공분산 행렬
- $\boldsymbol{\theta}$: $\mathbf{D}$와 $\mathbf{\Lambda}$의 고유 원소(unique elements)로 구성된 분산 성분 벡터
- $\hat{\boldsymbol{\beta}}(\boldsymbol{\theta}) = (\mathbf{X}^T\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{X}^T\mathbf{V}^{-1}\mathbf{y}$: GLS(일반화 최소자승) 추정량

> 💡 **GLS(Generalized Least Squares, 일반화 최소자승)**: 오차의 분산이 동일하지 않거나 상관이 있을 때 적용하는 최소자승 추정법. 공분산 행렬 $\mathbf{V}$를 이용해 가중치를 부여한다.

---

#### Profile Likelihood (Section 3, p.1015-1016)

$\sigma^2$의 ML 추정량을 $\boldsymbol{\beta}$와 $\boldsymbol{\theta}$의 함수로 표현:

$$\hat{\sigma}^2_{\text{ML}}(\boldsymbol{\beta}, \boldsymbol{\theta}) = \frac{1}{N}\sum_{i=1}^{M}\mathbf{r}_i^T\mathbf{V}_i(\boldsymbol{\theta})^{-1}\mathbf{r}_i$$

$$\hat{\sigma}^2_{\text{REML}}(\boldsymbol{\beta}, \boldsymbol{\theta}) = \frac{1}{N-p}\sum_{i=1}^{M}\mathbf{r}_i^T\mathbf{V}_i(\boldsymbol{\theta})^{-1}\mathbf{r}_i$$

이를 대입한 **ML Profile log-likelihood:**

$$p_F(\boldsymbol{\beta}, \boldsymbol{\theta} \mid \mathbf{y}) = -\frac{1}{2}\sum_{i=1}^{M}\log|\mathbf{V}_i(\boldsymbol{\theta})| - \frac{N}{2}\log\left[\sum_{i=1}^{M}\mathbf{r}_i^T\mathbf{V}_i^{-1}(\boldsymbol{\theta})\mathbf{r}_i\right] $$

**기호 설명:**
- $\mathbf{r}_i = \mathbf{y}_i - \mathbf{X}_i\boldsymbol{\beta}$: 잔차 벡터(residual vector)
- $N = \sum_{i=1}^{M}n_i$: 전체 관측값 수
- $p$: 고정효과 파라미터 수

> 💡 **Profile Likelihood(프로파일 우도)**: 관심 파라미터 외의 나머지 파라미터(여기서는 $\sigma^2$)를 그것의 최대우도 추정값으로 대체하여 얻은 우도 함수. 최적화를 단순화하고 수렴 안정성을 높인다.

---

#### Cholesky 재모수화 (Section 2, p.1015)

$\mathbf{D}$를 직접 추정하는 대신 Cholesky 인수 $\mathbf{L}$ (상삼각행렬, $\mathbf{L}^T\mathbf{L} = \mathbf{D}$)의 원소들을 최적화:

$$\boldsymbol{\theta}_{\text{new}} = \text{vec}(\mathbf{L}) \quad (\mathbf{L}^T\mathbf{L} = \mathbf{D},\ \mathbf{L}\ \text{상삼각행렬})$$

이 변환에 의해 체인 룰(chain rule)을 이용한 도함수:

$$\frac{\partial g(\mathbf{D})}{\partial \text{vec}(\mathbf{L})} = \tilde{\mathbf{L}}\left(\frac{\partial g(\mathbf{D})}{\partial \text{vec}(\mathbf{D}^T)} + \frac{\partial g(\mathbf{D})}{\partial \text{vec}(\mathbf{D})}\right)$$

여기서 $\tilde{\mathbf{L}} = \text{diag}(\mathbf{L}, \mathbf{L}, \ldots, \mathbf{L})$ ($q^2 \times q^2$ 행렬)

> 💡 **재모수화(Reparameterization)**: 원래 파라미터를 새로운 파라미터로 변환하여 제약 조건을 제거하거나 최적화를 용이하게 하는 기법.

---

#### QR 분해를 이용한 계산 효율화 (Section 4, p.1017)

$[\mathbf{Z}_i, \mathbf{X}_i]$에 QR 분해 적용:

$$[\mathbf{Z}_i, \mathbf{X}_i] = \mathbf{Q}_i\mathbf{R}_i = [\mathbf{Q}_{i(1)}, \mathbf{Q}_{i(2)}, \mathbf{Q}_{i(3)}]\begin{bmatrix}\mathbf{R}_{i(11)} & \mathbf{R}_{i(12)} \\ \mathbf{0} & \mathbf{R}_{i(22)} \\ \mathbf{0} & \mathbf{0}\end{bmatrix} $$

변환 후 조건부 분포:

$$\mathbf{c}_{i(1)} \mid \mathbf{b}_i \sim \mathcal{N}(\mathbf{R}_{i(12)}\boldsymbol{\beta} + \mathbf{R}_{i(11)}\mathbf{b}_i,\ \sigma^2\mathbf{I})$$

**기호 설명:**
- $\mathbf{Q}_i$: $n_i \times n_i$ 직교 행렬(orthogonal matrix)
- $\mathbf{R}_{i(11)}$: $q \times q$ 상삼각행렬
- $\mathbf{R}_{i(22)}$: $p \times p$ 상삼각행렬
- $\mathbf{R}_{i(12)}$: $q \times p$ 행렬
- $\mathbf{c}_i = \mathbf{Q}_i^T\mathbf{y}_i$: 회전된 관측값 벡터

> 💡 **QR 분해(QR Decomposition)**: 행렬을 직교 행렬 $\mathbf{Q}$와 상삼각행렬 $\mathbf{R}$의 곱으로 분해하는 방법. 수치적 안정성이 높아 선형 방정식 풀이와 최소자승 문제에 널리 사용된다.

---

#### EM 알고리즘 업데이트 공식 (Section 5, p.1018-1019)

$$\mathbf{D}_{\text{(R)ML}}^{(\omega+1)} = \mathbf{D}^{(\omega)} + \frac{1}{M}\sum_{i=1}^{M}\left[(\sigma^{(\omega)})^{-2}\hat{\mathbf{b}}_i\hat{\mathbf{b}}_i^T - \mathbf{D}^{(\omega)}\mathbf{R}_{i(11)}^T\hat{\mathbf{P}}_{i(\text{R})ML}\mathbf{R}_{i(11)}\mathbf{D}^{(\omega)}\right] $$

**기호 설명:**
- $\hat{\mathbf{b}}_i$: 개체 $i$의 랜덤효과 사후 평균(posterior mean)
- $\hat{\mathbf{P}}\_{i\text{ML}} = \mathbf{V}\_i^{-1}$, $\hat{\mathbf{P}}_{i\text{RML}} = \mathbf{V}_i^{-1} - \mathbf{V}_i^{-1}\mathbf{X}_i(\mathbf{X}^T\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{X}_i^T\mathbf{V}_i^{-1}$: 투영 행렬(projection matrices)

> 💡 **EM 알고리즘(Expectation-Maximization Algorithm)**: 잠재 변수(latent variable, 여기서는 랜덤효과 $\mathbf{b}_i$)가 있는 모델의 최대우도 추정을 위한 반복 알고리즘. E-step에서 잠재 변수의 기댓값을 계산하고, M-step에서 파라미터를 업데이트한다.

---

### 모델 구조

```
[데이터 구조]
y_i = X_i β (고정효과) + Z_i b_i (랜덤효과) + e_i (오차)

[분포 가정]
b_i ~ N(0, σ²D)        ← 개체 간 변동성 모델링
e_i ~ N(0, σ²Λ_i)     ← 조건부 오차 (보통 Λ_i = I 가정)

[주변 분포]
y_i ~ N(X_i β, σ²(Λ_i + Z_i D Z_i^T))

[추정 파라미터]
- β: 고정효과 (GLS로 조건부 추정)
- σ²: 오차 분산 (profile likelihood로 제거)
- D(또는 L: Cholesky 인수): 랜덤효과 공분산 (NR로 최적화)
```

---

### 성능 향상 및 한계

**성능 향상:**

| 측면 | 개선 내용 |
|---|---|
| 반복 횟수 | NR: 3~6회, EM: 40~>200회 (Table 1) |
| 계산 차수 | EM/NR 모두 $M(q^3 + q^2p)$으로 감소 |
| 수치 안정성 | Cholesky 재모수화로 양정치 보장 |
| 수렴 진단 | 직교성 기준(orthogonality criterion) 적용 가능 |
| 표준오차 | 역헤시안에서 즉시 획득 가능 |

**한계:**
- NR은 수렴이 보장되지 않음 (EM은 국소 최대값으로 항상 수렴)
- 랜덤효과 수 $q$가 매우 크면 NR의 계산 차수($Mq^4$)가 커짐
- iid 조건부 오차($\mathbf{\Lambda}_i = \mathbf{I}$) 가정에 주로 집중
- 시작값(starting values)의 선택이 수렴에 영향을 미침

> 💡 **직교성 기준(Orthogonality Convergence Criterion)**: Bates & Watts(1981)가 제안한 수렴 진단 지표로, 수치적 불확실성의 크기를 통계적 변동성에 대한 비율로 표현한 무차원 양. 이 값이 0.001 미만일 때 수렴으로 판정.

---

## 3. 각 주장에 페이지/Figure/Table 번호 표시

| 주장 | 위치 |
|---|---|
| 기본 혼합효과 모델 정의 및 GLS 추정량 | p.1014, Eq. (1.1), (1.2) |
| ML/REML 로그우도 정의 | p.1014, Eq. (1.3), (1.4) |
| Profile likelihood 도입 권장 | p.1015–1016, Sec. 2 & 3 |
| $\sigma$ 제거로 반복 횟수 감소 | p.1015, Sec. 2 |
| Cholesky 재모수화 필요성 | p.1015, Sec. 2 |
| ML/REML profile log-likelihood 공식 | p.1016, Eq. (3.3), (3.4) |
| QR 분해 기반 계산 효율화 | p.1017, Sec. 4, Eq. (4.1) |
| EM 업데이트 공식 | p.1018–1019, Eq. (5.1), (5.2) |
| 알고리즘 비교 결과 (Table 1) | p.1020, Table 1 |
| 반복 경로 시각화 (Figure 1) | p.1020, Figure 1 |
| NR의 우수성 결론 | p.1020–1021, Sec. 7.1 |
| 확장 모델 논의 (직렬 상관, 중첩 설계) | p.1021–1022, Sec. 7.2 |

---

## 4. 저자가 직접 보고한 결과 vs. 해석자의 해석

### 저자가 직접 보고한 결과

**연구 주제:**
> "We develop an efficient and effective implementation of the Newton-Raphson (NR) algorithm for estimating the parameters in mixed-effects models for repeated-measures data." (Abstract, p.1014)

**방법:**
- Profile likelihood를 최적화 대상으로 사용 (Eq. 3.3, 3.4, p.1016)
- D의 Cholesky 인수 L로 재모수화하여 제약 없는 최적화 실현 (p.1015)
- QR 분해로 EM/NR 반복 계산 차수를 $M(q^3 + q^2p)$으로 감소 (p.1018)

**결과 (Table 1, p.1020):**

| 예시 | 방법 | 필요 반복 수 | 반복당 평균 시간(초) |
|---|---|---|---|
| 말 난포 [Eq. 6.1] | NR | **4** | 1.58 |
| | EM | 50 | 0.50 |
| | EM+Aitken's | 9 | 0.50 |
| 골밀도 [Eq. 6.2] | NR | **3** | 3.53 |
| | EM | 40 | 1.18 |
| | EM+Aitken's | 6 | 1.18 |
| 골밀도 [Eq. 6.3] | NR | **6** | 9.21 |
| | EM | >200 | 2.45 |
| | EM+Aitken's | >200 | 2.45 |

**저자의 결론:**
> "In most situations a well-implemented NR algorithm is preferable to the EM algorithm or EM algorithm with Aitken's acceleration." (Abstract, p.1014)

---

### 해석자(리뷰어)의 해석

- **총 계산 시간 관점에서 비교의 복잡성**: 저자들은 반복 횟수와 반복당 시간을 각각 보고하지만, **총 계산 시간(반복 횟수 × 반복당 시간)** 관점에서 보면 결과가 다르게 해석될 수 있다. 예를 들어 말 난포 예시에서 NR 총 시간은 약 $4 \times 1.58 = 6.32$초인 반면, EM+Aitken은 $9 \times 0.50 = 4.5$초로 실제로는 EM+Aitken이 더 빠를 수 있다. ⚠️ **이 점이 통계적으로 취약한 부분임 (Section 5 참조).**

- **일반화 가능성의 제한**: 두 개의 데이터셋만을 사용한 비교는 다양한 실제 상황에서의 일반화에 한계가 있다. 특히 모든 예시에서 $q \leq 3$으로 랜덤효과 차원이 낮다.

- **역사적 의의**: 이 논문은 이후 R의 `nlme` 패키지(Pinheiro & Bates 2000)의 직접적 이론적 기반이 되었으며, 현대 혼합효과 모델 소프트웨어의 표준을 확립하였다는 점에서 매우 중요한 논문으로 평가된다.

---

## 5. 통계적으로 취약한 부분 및 비교 불가능한 수치

> ⚠️ **주의가 필요한 부분**

| 문제 | 설명 |
|---|---|
| ⚠️ **소표본 비교** | 말 난포 예시: $M=11$개체, 골밀도: $M=74$개체만 사용. 알고리즘 비교를 위한 시뮬레이션 연구 없음 |
| ⚠️ **총 계산 시간 미보고** | 반복 횟수와 반복당 시간을 별도로 보고하여, 총 실행 시간 비교가 불명확함. NR이 반복 횟수는 적지만 반복당 시간이 훨씬 더 길어 총 시간 우위가 불분명한 경우 존재 (말 난포 예: NR ≈6.3초 vs EM+Aitken ≈4.5초) |
| ⚠️ **하드웨어 의존성** | Vax 11/750이라는 1980년대 구형 컴퓨터에서의 타이밍 결과로, 현재 하드웨어에서의 상대적 성능과 직접 비교 불가 |
| ⚠️ **수렴 기준의 비일치성** | EM 알고리즘의 수렴 판단 시간에는 직교성 기준 계산 시간이 미포함. 이는 EM의 총 계산 시간을 과소추정하게 함 |
| ⚠️ **단일 시작값** | 시작값(starting values)이 Laird et al.(1987)의 방법으로 고정되어 있어, 다른 시작값에서의 알고리즘 성능 비교가 없음 |
| ⚠️ **낮은 $q$ 차원에 한정** | 모든 실험에서 $q \leq 3$. 고차원 랜덤효과($q \gg 3$)에서의 NR vs EM 비교 없음 |
| ⚠️ **통계적 유의성 검정 없음** | 알고리즘 비교에 반복 실험이나 통계적 검정이 수행되지 않음 |

---

## 6. 문서가 답하지 않는 질문

| 질문 | 설명 |
|---|---|
| 🔍 고차원 랜덤효과에서의 성능은? | $q$가 크면(예: $q > 10$) NR의 $Mq^4$ 차수가 EM보다 불리해지는 정확한 임계점이 제시되지 않음 |
| 🔍 비정규 분포 데이터에 대한 적용 가능성? | 모든 분포 가정이 정규 분포에 기반. 비정규 데이터나 이진/계수 반응변수(generalized mixed models)에 대한 논의 없음 |
| 🔍 시작값이 수렴에 미치는 영향? | 다양한 시작값에서의 알고리즘 민감도 분석 없음 |
| 🔍 국소 최대값(local maximum) 문제는? | NR이 국소 최대값에 수렴하는 경우의 빈도와 이를 탐지/해결하는 방법이 논의되지 않음 |
| 🔍 직렬 상관 오차 구조의 구체적 구현? | Sec. 7.2에서 AR(1) 등을 간략히 언급하지만 구체적 공식과 구현 방법 제시 없음 |
| 🔍 결측 메커니즘(MAR/MCAR/MNAR)의 영향? | 결측값을 자동 처리한다고 언급하나, 결측 메커니즘에 따른 편향 가능성 논의 없음 |
| 🔍 베이지안 방법과의 비교? | 빈도주의(frequentist) 관점에서만 논의; 베이지안 추정(Bayesian estimation)과의 비교 없음 |
| 🔍 수렴 실패 시의 진단과 대처? | NR이 수렴하지 않을 때의 체계적인 대응 방법이 제시되지 않음 |

> 💡 **MAR/MCAR/MNAR**: 결측 메커니즘의 세 가지 유형. MCAR(완전 무작위 결측), MAR(무작위 결측, 관측된 데이터에 조건부), MNAR(비무작위 결측, 결측 자체가 결측값에 의존).

---

## 7. 가장 중요한 그림/표 5개 해석

논문에는 Figure 1과 Table 1이 주요 시각적 자료로 제시되어 있다. 나머지는 수식 중심의 서술이므로, 핵심 구조를 중심으로 해석한다.

---

### 📊 Figure 1 (p.1020): 말 난포 예시의 반복 경로 시각화

**내용:** $D_{11}$, $D_{33}$, $\sigma^2$의 3개 파라미터에 대해 NR과 EM 알고리즘의 반복 경로를 2차원 쌍별(pairwise) 플롯으로 제시.

**해석:**
- **NR (숫자로 표시)**: 반복 7부터 표시되며, 3~4번째 반복에서 최적값에 정확히 수렴. 최적값 주변을 빠르게 "괄호로 묶어가는(bracket)" 양상을 보임.
- **EM (별표 및 점선으로 표시)**: 같은 최적값을 향해 매우 느리게, 거의 직선에 가까운 경로로 접근. 50번의 반복 후에도 최적값 근방에 머물지만 정확히 수렴하지 못함.
- **핵심 시사점**: EM의 경로 방향이 반복마다 거의 변하지 않아 "가속도" 없이 기계적으로 수렴하는 반면, NR은 곡률(curvature) 정보(헤시안)를 활용하여 동적으로 방향을 조정함을 시각적으로 명확히 보여줌.

> 💡 **헤시안(Hessian Matrix)**: 다변수 함수의 2차 편미분으로 이루어진 행렬. Newton-Raphson 알고리즘에서 다음 이동 방향과 보폭을 결정하는 데 사용된다. 우도 함수의 곡률 정보를 포함한다.

---

### 📊 Table 1 (p.1020): NR과 EM 알고리즘 비교

**내용:** 3개의 모델 예시에 대해 NR, EM, EM+Aitken 가속의 필요 반복 수와 반복당 평균 계산 시간 비교.

**해석:**

| 관찰 | 세부 내용 |
|---|---|
| **NR의 우월한 수렴 속도** | 모든 예시에서 NR은 3~6회로 수렴, EM은 9~>200회 필요 |
| **복잡한 모델에서 EM 실패** | Eq.(6.3) 모델에서 EM과 EM+Aitken 모두 200회 이내 미수렴 |
| **Aitken 가속의 조건부 효과** | 단순 모델(Eq.6.1, 6.2)에서는 효과적이나 복잡 모델에서는 무력함 |
| ⚠️ **반복당 시간의 역전** | NR의 반복당 시간이 EM보다 3~4배 길어(1.58초 vs 0.50초), 총 계산 시간 관점에서는 일부 경우 EM+Aitken이 더 빠를 수 있음 |

---

### 📊 수식 구조 시각화 1: QR 분해의 핵심 (p.1017, Eq. 4.1)

**내용:** 설계 행렬 $[\mathbf{Z}_i, \mathbf{X}_i]$에 QR 분해를 적용하여 계산 복잡도를 감소시키는 구조.

**해석:**
- $n_i \times n_i$ 크기의 $\mathbf{V}_i$ 역행렬 계산을 피하고, $q \times q$ 및 $p \times p$ 상삼각 행렬로 문제를 축소.
- 원래 $O(n_i^3)$ 계산을 $O(q^3)$ 수준으로 감소. 개체당 관측값이 많아도($n_i \gg q$) 계산 부담이 증가하지 않음.
- 이는 현대 혼합효과 모델 소프트웨어의 계산 구조와 동일하며, 1988년 당시 획기적인 개선임.

---

### 📊 수식 구조 시각화 2: EM vs NR 업데이트 구조 비교 (p.1018-1019)

**EM 업데이트의 구조:**
$$\mathbf{D}^{(\omega+1)} \leftarrow \text{현재 추정값 기반 1차 정보만 사용} \quad (\text{선형 수렴})$$

**NR 업데이트의 구조:**
$$\boldsymbol{\theta}^{(\omega+1)} = \boldsymbol{\theta}^{(\omega)} - \mathbf{H}^{-1}\mathbf{g} \quad (\text{2차 정보 활용, 2차 수렴})$$

여기서 $\mathbf{H}$는 헤시안, $\mathbf{g}$는 기울기(gradient) 벡터.

**해석:** EM은 1차 수렴(선형 수렴)으로 매우 느리게 수렴하는 반면, NR은 2차 수렴(quadratic convergence)으로 최적값 근방에서 오차가 제곱으로 감소하여 훨씬 빠르게 수렴한다.

> 💡 **2차 수렴(Quadratic Convergence)**: 알고리즘이 수렴 목표에 가까워질수록 오차가 이전 오차의 제곱에 비례하여 감소하는 성질. 예: 오차 0.01 → 0.0001 → 0.00000001 순으로 급격히 감소.

---

### 📊 모델 확장 구조 (p.1021-1022, Sec. 7.2)

**내용:** 기본 모델을 직렬 상관 오차 구조 및 중첩 설계로 확장하는 방법 제시.

**해석:**
- **AR(1) 조건부 오차**: $\mathbf{\Lambda}_i$의 $(j,k)$ 원소를 $\rho^{|d(j,k)|}$로 정의하여 시간적 자기상관을 모델링. $d(j,k)$는 $j$번째와 $k$번째 관측 시점 간의 거리.
- **중첩 설계**: $\mathbf{Z}\_{i1}\mathbf{b}\_{i1} + \mathbf{Z}\_{i2}\mathbf{b}_{i2}$ 형태로 여러 수준의 랜덤효과를 추가.
- NR은 이러한 확장에 유연하게 대응 가능하지만, EM은 일부 확장에서 적용이 어려움을 저자들이 명시함.

---

## 8. 결론: 시사점, 후속 연구 계획, 추가 방향

### 8-1. 저자들이 제시한 시사점

1. **알고리즘 선택 원칙**: 랜덤효과 수 $q$가 적당한 경우 NR이 EM보다 우월하며, $q$가 매우 큰 경우에만 EM을 고려할 수 있음.
2. **재모수화의 중요성**: 공분산 행렬의 Cholesky 인수를 이용한 재모수화가 NR의 수렴 일관성에 핵심적.
3. **Profile likelihood 활용**: $\sigma$를 포함한 original likelihood 대신 profile likelihood를 최적화하면 수렴이 빠르고 안정적.
4. **확장 가능성**: NR 기반 프레임워크는 직렬 상관 오차, 중첩 설계, 시간 의존 공분산 등으로 비교적 쉽게 확장 가능.

**저자들이 제시한 후속 연구 방향:**
- 비독립 조건부 오차 구조(serially correlated errors)의 구체적 구현
- 중첩/분할 구획 설계(nested/split-plot designs)로의 확장
- $\mathbf{\Lambda}_i$가 고정/랜덤효과에 의존하는 모델로의 확장

---

### 8-1. 모델의 일반화 성능 향상 가능성

#### 현재 모델의 일반화 제약

1. **정규분포 가정**: 모든 분포 가정이 다변량 정규분포에 기반하여, 비정규 데이터(이진, 계수, 생존시간 등)에는 직접 적용 불가.
2. **선형 관계 가정**: 고정/랜덤효과와 반응변수 간의 선형 관계만 가정.
3. **독립 개체 가정**: 개체 간 독립성 가정으로, 계층적 군집 구조(clustered data) 처리에 한계.
4. **저차원 랜덤효과**: $q$가 크면 계산 복잡도가 급증.

#### 일반화 성능 향상을 위한 방향

**① 비선형 혼합효과 모델로의 확장:**

$$y_{ij} = f(\mathbf{x}_{ij}, \boldsymbol{\phi}_i) + \epsilon_{ij}, \quad \boldsymbol{\phi}_i = \mathbf{A}_i\boldsymbol{\beta} + \mathbf{B}_i\mathbf{b}_i$$

Lindstrom & Bates(1990)에서 이를 확장. QR 분해 기반 계산 구조가 비선형 모델에도 유사하게 적용됨.

**② 일반화 선형 혼합효과 모델(GLMM):**

$$g(\mu_{ij}) = \mathbf{x}_{ij}^T\boldsymbol{\beta} + \mathbf{z}_{ij}^T\mathbf{b}_i$$

여기서 $g(\cdot)$는 연결함수(link function). 이진 반응변수, 계수 데이터 등에 적용 가능. Breslow & Clayton(1993)에서 발전.

> 💡 **연결함수(Link Function)**: 반응변수의 기댓값과 선형 예측자(linear predictor)를 연결하는 단조 미분 가능 함수. 이진 자료의 경우 로짓(logit) 함수, 계수 자료의 경우 로그 함수를 사용.

**③ 베이지안 혼합효과 모델:**
$$p(\boldsymbol{\beta}, \mathbf{D}, \sigma^2 \mid \mathbf{y}) \propto p(\mathbf{y} \mid \boldsymbol{\beta}, \mathbf{D}, \sigma^2) \cdot p(\boldsymbol{\beta}) \cdot p(\mathbf{D}) \cdot p(\sigma^2)$$

MCMC(Markov Chain Monte Carlo)를 이용하면 소표본, 고차원 랜덤효과, 비공액 사전 분포 등에서 더 유연한 추론 가능.

**④ 스파스 행렬 기법(Sparse Matrix Methods) 활용:**
랜덤효과의 설계 행렬 $\mathbf{Z}$가 희소(sparse)한 경우 스파스 QR 분해를 활용하면 고차원 문제에서도 계산 효율을 유지할 수 있음. 이는 현대 `lme4` 패키지(Bates et al. 2015)에서 채택된 접근법임.

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

> **⚠️ 주의**: 아래 최신 연구 목록은 제 학습 데이터(2023년 초까지)에 기반한 것으로, 일부 세부 내용은 정확하지 않을 수 있습니다. 반드시 원문을 확인하시기 바랍니다.

#### 2020년 이후 주요 연구 흐름

| 연구 방향 | 대표 연구/방법 | Lindstrom & Bates(1988)와의 관계 |
|---|---|---|
| **고차원 랜덤효과** | 스파스 Cholesky 분해 기반 REML (`lme4` 확장) | Cholesky 재모수화 아이디어를 고차원으로 확장 |
| **베이지안 혼합효과** | Stan/BRMS를 이용한 베이지안 추정 (Bürkner 2017+) | NR 기반 추정의 베이지안 대안 |
| **GLMM의 확장** | NIMBLE, TMB(Template Model Builder)를 이용한 효율적 적분 | 비선형/비정규 반응변수로의 일반화 |
| **딥러닝 혼합효과** | Neural Mixed Effects (NME), DeepLME | 혼합효과 구조를 딥러닝과 결합 |
| **인과추론과 혼합효과** | 종단 데이터에서의 인과효과 추정 | 기본 모델 구조를 인과적 맥락으로 재해석 |
| **스케일러블 추정** | Stochastic Variational Inference (SVI) 기반 혼합효과 | 대규모 데이터에서 NR의 확장성 한계 극복 |

#### Lindstrom & Bates(1988)가 후속 연구에 미친 영향

**직접적 영향:**
1. **Pinheiro & Bates(2000)의 `nlme` R 패키지**: 이 논문의 알고리즘이 직접 구현됨. 현재까지 가장 널리 사용되는 혼합효과 모델 소프트웨어 중 하나.
2. **Bates et al.(2015)의 `lme4` R 패키지**: NR 기반 구조를 스파스 행렬 방법으로 발전시켜 고차원 데이터에서의 확장성 확보.
3. **Lindstrom & Bates(1990)의 비선형 확장**: 동일 저자들이 이 논문의 프레임워크를 비선형 혼합효과 모델로 확장.

**간접적 영향:**
- REML 추정, Cholesky 재모수화, QR 분해 기반 계산이 현대 통계 소프트웨어(SAS PROC MIXED, SPSS, Stata 등)의 표준이 됨.
- 반복측정 및 종단 데이터 분석의 교과서적 방법론 확립.

#### 앞으로 연구 시 고려할 점

1. **계산 확장성**: 현대 데이터는 개체 수($M$)와 관측값 수($N$)가 모두 크므로, 병렬 컴퓨팅(GPU/분산 처리)과 스파스 행렬 기법을 결합한 알고리즘 개발이 필요.

2. **비정규 반응변수**: 이진, 순서형, 계수, 생존시간 등 다양한 반응변수에 대한 혼합효과 모델링을 위해 GLMM과의 통합이 중요하며, Laplace 근사(Laplace Approximation)나 가우스-에르미트 구적법(Gauss-Hermite Quadrature)의 정확도와 효율성 균형이 핵심 과제.

   > 💡 **Laplace 근사**: 함수를 최빈값(mode) 근방에서 2차 테일러 전개로 근사하는 방법. GLMM의 주변 우도(marginal likelihood) 계산에 널리 사용됨.

3. **모델 선택과 정규화**: 고차원 랜덤효과에서 어떤 효과를 랜덤으로 포함할지 결정하는 문제가 중요해졌으며, LASSO 유형의 정규화를 혼합효과에 적용하는 연구(예: Schelldorfer et al. 2011)가 진행 중.

4. **결측 메커니즘의 명시적 모델링**: MNAR(비무작위 결측) 상황에서의 편향 없는 추정을 위한 선택 모델(selection model)이나 패턴 혼합 모델(pattern mixture model)과의 통합.

5. **인과추론과의 통합**: 종단 혼합효과 모델을 인과적 프레임워크(예: 시간 의존 교란 변수의 처리)와 결합하는 것이 임상 및 사회과학 연구에서 점점 중요해지고 있음.

6. **불확실성 정량화**: 베이지안 접근 또는 부트스트랩을 통한 신뢰구간 추정이 REML 기반 점추정의 한계를 보완할 수 있음.

---

## 참고자료

**원 논문:**
- Lindstrom, M. J., & Bates, D. M. (1988). Newton–Raphson and EM algorithms for linear mixed-effects models for repeated-measures data. *Journal of the American Statistical Association*, 83(404), 1014–1022. DOI: 10.1080/01621459.1988.10478693

**논문 내 인용 참고문헌:**
- Laird, N. M., & Ware, J. H. (1982). Random-effects models for longitudinal data. *Biometrics*, 38, 963–974.
- Jennrich, R. I., & Schluchter, M. D. (1986). Unbalanced repeated measures models with structural covariance matrices. *Biometrics*, 42, 805–820.
- Laird, N., Lange, N., & Stram, D. (1987). Maximum likelihood computations with repeated measures: Application of the EM algorithm. *Journal of the American Statistical Association*, 82, 97–105.
- Harville, D. A. (1974). Bayesian inference for variance components using only error contrasts. *Biometrika*, 61, 383–385.
- Bates, D. M., & Watts, D. G. (1981). A relative offset orthogonality convergence criterion for nonlinear least squares. *Technometrics*, 23, 179–183.

**관련 후속 연구:**
- Lindstrom, M. J., & Bates, D. M. (1990). Nonlinear mixed effects models for repeated measures data. *Biometrics*, 46(3), 673–687.
- Pinheiro, J. C., & Bates, D. M. (2000). *Mixed-Effects Models in S and S-PLUS*. Springer.
- Bates, D., Mächler, M., Bolker, B., & Walker, S. (2015). Fitting linear mixed-effects models using lme4. *Journal of Statistical Software*, 67(1), 1–48.
- Breslow, N. E., & Clayton, D. G. (1993). Approximate inference in generalized linear mixed models. *Journal of the American Statistical Association*, 88(421), 9–25.
- Bürkner, P. C. (2017). brms: An R package for Bayesian multilevel models using Stan. *Journal of Statistical Software*, 80(1), 1–28.
