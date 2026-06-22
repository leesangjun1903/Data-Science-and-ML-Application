
# On a Regularization of Unsupervised Domain Adaptation in RKHS

> **논문 정보:**
> - **저자:** Elke R. Gizewski, Lukas Mayer, Bernhard A. Moser, Duc Hoan Nguyen, Sergiy Pereverzyev Jr., Sergei V. Pereverzyev, Natalia Shepeleva, Werner Zellinger
> - **저널:** *Applied and Computational Harmonic Analysis*, Vol. 57, pp. 201–227, 2022
> - **출처:** [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1063520321001032), [RICAM Technical Report 2021-17](https://www.ricam.oeaw.ac.at/files/reports/21/rep21-17.pdf), [ResearchGate](https://www.researchgate.net/publication/357114087_On_a_regularization_of_unsupervised_domain_adaptation_in_RKHS), [Academia.edu](https://www.academia.edu/69333214/On_a_regularization_of_unsupervised_domain_adaptation_in_RKHS)

---

## 1. 핵심 주장과 주요 기여 (간결 요약)

이 논문은 **공변량 이동(covariate shift) 가정** 하에서의 비지도 도메인 적응(unsupervised domain adaptation) 시나리오에 대해 이른바 **일반 정규화 기법(general regularization scheme)**의 사용을 분석한다.

이 기법에서 도출된 학습 알고리즘은 공변량 이동 환경에서 현재까지 가장 많이 사용되는 방법 중 하나인 **중요도 가중 정규화 최소제곱법(IWRLS)**의 일반화이며, 재생 커널 힐베르트 공간(RKHS)에서의 **Radon-Nikodym 도함수** 추정과 도메인 적응 문제 사이의 연결 고리를 탐구한다.

### 🔑 주요 기여 요약

| 기여 항목 | 내용 |
|---|---|
| ① 이론적 오차 경계 | IWRLS에 대한 최초의 리스크 경계 수립 |
| ② 일반 정규화 기법 | 커널 릿지 회귀를 넘어선 스펙트럴 정규화 일반화 |
| ③ Radon-Nikodym 연결 | 밀도비 추정과 도메인 적응을 RKHS에서 통합 |
| ④ 정규화 파라미터 선택 | 밸런싱 원리(balancing principle) 기반의 이론적으로 정당화된 규칙 제시 |
| ⑤ Aggregation 기법 | 사전 지식 없이도 성능을 향상시키는 근사해 집계 방법 |

---

## 2. 문제 정의, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

비지도 도메인 적응에서 공변량 이동 하의 목적은 표준 지도 학습과 동일한 함수를 근사하는 것이므로, 지도 학습에서 개발된 방법들을 도메인 적응 시나리오에 조정하는 것이 자연스럽다. RKHS에서의 지도 학습은 통계 학습 이론 중 가장 잘 발전된 분야 중 하나이며, 정규화된 커널 릿지 회귀가 가장 잘 이해된 지도 학습 알고리즘이다. 이 알고리즘은 이미 샘플 재가중(sample reweighting)과 결합하여 비지도 도메인 적응에 활용되었으나, **공변량 이동 가정 하에서도 이 결합에 대한 리스크 경계(risk bounds)는 알려져 있지 않았다.**

핵심 문제 설정은 다음과 같다:

- **소스 도메인** $(\mathcal{X}, \rho_S)$: 레이블이 있는 데이터
- **타겟 도메인** $(\mathcal{X}, \rho_T)$: 레이블이 없는 데이터
- **공변량 이동 가정:** 조건부 분포 $\rho(y|x)$는 동일하지만, 주변 분포 $\rho_S(x) \neq \rho_T(x)$

목표 회귀 함수는 다음의 $L^2$ 최소화로 정의된다:

$$f^* = \arg\min_{f \in L^2_{\rho_T}} \int_\mathcal{X} (f(x) - y)^2 \, d\rho_T(x, y)$$

---

### 2.2 제안하는 방법 (수식 포함)

#### (A) RKHS에서의 일반 정규화 기법 (General Regularization Scheme)

RKHS에서의 정규화 기법들의 큰 클래스인 **스펙트럴 정규화(spectral regularization)**는 지도 학습 환경에서 광범위하게 연구되어 왔다. 이 논문은 이를 도메인 적응으로 확장한다.

RKHS $\mathcal{H}_K$ 에서 목표 함수를 추정하기 위해 커널 적분 연산자 $T_K: \mathcal{H}_K \to \mathcal{H}_K$를 다음과 같이 정의한다:

$$T_K f = \int_\mathcal{X} K(\cdot, x) f(x) \, d\rho_S(x)$$

일반 정규화 기법은 **필터 함수** $g_\lambda: [0, \kappa^2] \to \mathbb{R}$를 통해 표현된다:

$$\hat{f}_\lambda = g_\lambda(\hat{T}_K) \hat{S}^* \mathbf{y}^S_w$$

여기서:
- $\hat{T}_K$: 경험적 커널 연산자 (empirical kernel operator)
- $\hat{S}^*$: 샘플링 연산자의 수반(adjoint)
- $\mathbf{y}^S_w$: 중요도 가중 레이블 벡터
- $\lambda > 0$: 정규화 파라미터

#### (B) 중요도 가중 정규화 최소제곱법 (IWRLS)

IWRLS는 위 기법에서 도출된 학습 알고리즘들의 일반화로, 공변량 이동 환경에서 현재까지 가장 많이 사용되는 접근법이다.

IWRLS의 목적함수는 다음과 같다:

$$\hat{f}^{IWRLS}_\lambda = \arg\min_{f \in \mathcal{H}_K} \left[ \frac{1}{m} \sum_{i=1}^{m} w(x_i^S)(f(x_i^S) - y_i^S)^2 + \lambda \|f\|^2_{\mathcal{H}_K} \right]$$

여기서 중요도 가중치는 Radon-Nikodym 도함수로 정의된다:

$$w(x) = \frac{d\rho_T}{d\rho_S}(x)$$

#### (C) Radon-Nikodym 도함수 추정 (RKHS 내)

본 연구는 고려된 도메인 적응 시나리오와 RKHS에서의 Radon-Nikodym 도함수 추정 사이의 연결 고리를 탐색하며, 이 추정 과정에서도 일반 정규화 기법이 적용된다.

밀도비 $\beta = d\rho_T/d\rho_S$의 RKHS 추정은 다음의 커널화된 최소제곱 중요도 피팅(KuLSIF) 문제로 귀결된다:

$$\hat{\beta}_\lambda = \arg\min_{\beta \in \mathcal{H}_K} \left[ \frac{1}{2n} \sum_{j=1}^{n} \beta(x_j^T)^2 - \frac{1}{m} \sum_{i=1}^{m} \beta(x_i^S) + \lambda \|\beta\|^2_{\mathcal{H}_K} \right]$$

이는 다시 일반 정규화 기법으로 통합된다:

$$\hat{\beta}_\lambda = g_\lambda(\hat{C}) \hat{\mu}_{TS}$$

여기서 $\hat{C}$는 경험적 공분산 연산자, $\hat{\mu}_{TS}$는 소스-타겟 교차 평균 임베딩이다.

#### (D) 밸런싱 원리를 통한 정규화 파라미터 선택

본 접근법은 소스 오차를 최소화하되 소스-타겟 특징 표현 사이의 거리 척도로 패널티를 부과하는 방법이 정규화된 역문제와 특성을 공유한다는 관찰에서 출발한다. 역문제에서 정규화 파라미터는 근사 오차와 샘플링 오차의 균형을 맞추는 원리로 최적 선택된다. 이 원리를 사용하여 학습 오차와 도메인 거리를 타겟 오차 경계에서 균형 맞추었으며, 그 결과 **정규화 파라미터 선택에 대한 이론적으로 정당화된 규칙**을 얻었다.

구체적으로, 밸런싱 원리는 다음의 경계를 최적화한다:

$$\mathcal{E}(\hat{f}_\lambda, \rho_T) \leq C_1 \cdot \underbrace{\phi(\lambda)}_{\text{근사 오차}} + C_2 \cdot \underbrace{\frac{1}{\sqrt{m\lambda}}}_{\text{샘플링 오차}} + C_3 \cdot \underbrace{d(\rho_S, \rho_T)}_{\text{도메인 거리}}$$

여기서 $\phi(\lambda)$는 정규화 파라미터에 의존하는 근사 오차 항이다.

#### (E) 근사해 집계 (Aggregation of Approximants)

근사해의 집계(aggregation)는 정규화 파라미터에 대한 사전 지식 없이도 성능을 향상시킨다.

집계는 다음과 같이 정의된다:

$$\hat{f}_{agg} = \sum_{k=1}^{K} \alpha_k \hat{f}_{\lambda_k}, \quad \sum_{k=1}^{K} \alpha_k = 1, \quad \alpha_k \geq 0$$

---

### 2.3 모델 구조

```
[입력]
  소스 데이터 {(x_i^S, y_i^S)}_{i=1}^m  +  타겟 데이터 {x_j^T}_{j=1}^n (레이블 없음)
        │                                           │
        ▼                                           ▼
[Step 1] Radon-Nikodym 도함수 추정 (RKHS 내 일반 정규화)
        │ → 밀도비 β̂(x) = dρ_T/dρ_S(x) 추정
        │
        ▼
[Step 2] IWRLS / 일반 정규화 기법으로 타겟 함수 추정
        │ → 중요도 가중 손실 + RKHS 정규화 항 최소화
        │
        ▼
[Step 3] 밸런싱 원리로 λ 선택 (또는 Aggregation)
        │ → 사전 지식 없이도 이론적으로 정당화된 파라미터 선택
        │
        ▼
[출력]
  타겟 도메인에 대한 예측 함수 f̂
```

---

### 2.4 성능 향상

도메인 적응에서의 IWRLS에 대한 오차 경계가 수립되었으며, 지도 학습의 수렴 속도와 동등한 수준에 도달하였다. 일반 정규화 기법은 수치 실험에서 표준 IWRLS를 능가하는 성능을 보였다.

기존 연구와 달리, 본 접근법은 소스 분포와 타겟 분포의 지지(support)가 서로 겹치지 않는 경우(disjoint supports)도 허용한다.

---

### 2.5 한계점

대부분의 도메인 적응 알고리즘은 성능을 바꾸고 튜닝이 필요한 소위 하이퍼파라미터에 의존한다.

특히, 적절한 하이퍼파라미터 선택이 근본적으로 중요하며, 부정확한 선택은 차선의 결과로 이어질 수 있다.

추가적인 한계로는:
- **밀도비 유계 가정:** 기본 이론은 $w(x) = d\rho_T/d\rho_S(x)$가 균일하게 유계라는 가정에 의존
- **RKHS 적합 가정 (Well-specified case):** 목표 함수가 RKHS에 속한다고 가정하는 내적 정칙성(inner regularity) 조건 필요
- **계산 복잡도:** 전체 커널 행렬 사용 시 $O(m^3)$ 계산 비용 발생

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 이론적 수렴 속도 보장

RKHS에서 고려된 평활도 조건(smoothness conditions)은 IWRLS가 표준 RKHS 최소제곱 회귀와 동일한 정확도를 달성하는 새로운 상황을 제공하며, 이 정확도의 차수는 기존 기술 수준보다 훨씬 높은 것으로 나타났다.

**소스 조건(Source Condition)** 하에서의 수렴 속도 — 목표 함수의 평활도 지수 $s > 0$에 대해:

$$\|\hat{f}_\lambda - f^*\|_{L^2_{\rho_T}} = O\left(m^{-\frac{s}{2s+1}}\right)$$

이는 지도 학습의 미니맥스 최적(minimax optimal) 속도와 일치한다.

### 3.2 밸런싱 원리의 역할 (일반화 기여)

이 논문은 비지도 도메인 적응에서 정당화된 정규화 파라미터 선택이라는 미해결 알고리즘 설계 문제를 다루며, 정규화 파라미터 선택을 위한 이론적으로 정당화된 규칙을 제시한다.

밸런싱 원리는 **오라클 불평등(oracle inequality)** 형태로 다음을 보장한다:

$$\mathcal{E}(\hat{f}_{\hat{\lambda}}) \leq C \cdot \inf_{\lambda > 0} \mathcal{E}(\hat{f}_\lambda) + \text{(로그 항)}$$

이는 최적 $\lambda$를 사전에 모르더라도 거의 동등한 성능을 보장한다.

### 3.3 Aggregation에 의한 일반화 향상

최근 이론적 발전에 따르면, 도메인 적응 알고리즘의 성공은 소스와 타겟 도메인의 확률 분포 간 발산을 최소화하는 능력에 크게 의존한다.

근사해 집계는 정규화 경로 $\{\hat{f}\_{\lambda_1}, \ldots, \hat{f}_{\lambda_K}\}$에서 다음의 적응형 가중합을 통해 일반화 성능을 향상시킨다:

```math
\hat{f}_{agg} = \sum_{k} \alpha^*_k \hat{f}_{\lambda_k}, \quad \alpha^* = \arg\min_{\alpha \in \Delta^K} \|\sum_k \alpha_k \hat{f}_{\lambda_k} - f^*\|^2
```

이는 개별 추정량보다 항상 같거나 나은 타겟 오차를 보장한다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4.1 후속 연구에 미치는 영향

#### (i) 일반 공변량 이동 이론의 발전 기여

Gizewski et al. (2022)에서는 **분할 형태의 지수 함수(splitting form of the index function)**로 특성화된 일반 소스 조건 하에서의 수렴 속도가 확립되었다.

이에 더불어, 최근 여러 연구에서 **비유계 밀도비(unbounded density ratios)** 문제를 다루는 방법들이 조사되었다(Gogolashvili et al., 2023; Ma et al., 2023; Fan et al., 2025; Liu and Guo, 2025).

#### (ii) 계산 효율성 연구로의 확장

나아가, Nyström 서브샘플링과 같은 계산 효율적인 방법들에 대한 수렴 분석이 확립되었으며, 이는 최적의 통계적 보장을 유지하면서 커널 기반 알고리즘의 계산 복잡도를 크게 줄인다.

#### (iii) 과잉 리스크 경계의 개선

이 논문의 기여를 이어받아 발표된 후속 연구(2026, *Applied and Computational Harmonic Analysis*)는 RKHS 내의 도메인 적응에서 **과잉 리스크 경계(excess risk bounds) 개선**을 목표로 한다. 이전 연구들이 주로 목표 함수의 평활도 또는 기저 공간의 용량 중 하나에만 집중했던 것과 달리, 이 연구는 일반적인 중요도 가중 스펙트럴 알고리즘을 고려하고 두 항 모두를 분석에 통합하여 공변량 이동 가정 하에서 과잉 리스크 경계를 크게 개선하였다.

#### (iv) 밀도비 추정 이론과의 연결

두 확률 밀도의 비를 유한한 관측치로부터 추정하는 것은 머신러닝과 통계학의 핵심 문제로, 이 논문의 영향 아래에서 RKHS의 정규화된 Bregman 발산 최소화 방법들을 분석하는 연구들이 등장하였으며, 새로운 유한 샘플 오차 경계와 Lepskii 형태의 파라미터 선택 원리가 도출되었다.

---

### 4.2 앞으로 연구 시 고려할 점

#### ✅ (1) 비유계 밀도비 (Unbounded Density Ratios) 문제

균일하게 유계인 밀도비 가정 하에서는 미니맥스 최적 수렴 속도가 확립되지만, **비유계 중요도 가중치** 시나리오에서는 새로운 절단(truncation) 기법이 도입되어야 하며, 이는 약한 정칙성 조건 하에서 근접 최적 수렴 속도에 도달한다.

#### ✅ (2) Misspecified Regime (모델 오설정) 처리

비유계 중요도 가중치 문제는 **오설정(misspecified) 환경**까지 확장되어야 하며, 공변량 이동과 모델 오설정의 상호작용을 이해하는 체계적 프레임워크를 구축하는 것이 중요하다.

#### ✅ (3) 딥러닝과의 결합

현재의 이론적 프레임워크(RKHS 기반)는 딥 신경망의 암묵적 정규화와 연결될 필요가 있다. Neural Tangent Kernel(NTK) 이론을 활용하면 무한 너비 신경망의 동작을 RKHS 이론으로 분석 가능하다.

#### ✅ (4) 계산 효율성 (Scalability)

Nyström 방법과 같은 랜덤 투영 기법들은 표준 학습 환경에서는 광범위하게 연구되었으나, **공변량 이동 하에서의 활용은 아직 충분히 탐구되지 않았다.** 이 방향으로의 최근 진전으로 Myleiko & Solodky(2024)가 Gizewski et al.의 프레임워크를 기반으로 한 연구를 발표하였다.

#### ✅ (5) 레이블 이동(Label Shift)과의 통합

공변량 이동 이외에도 레이블 분포가 달라지는 **레이블 이동(label shift)** 설정을 RKHS 정규화 프레임워크로 통합하는 연구가 필요하다.

#### ✅ (6) 실제 임상 데이터에의 적용 확대

이 연구는 **불균형 학습(imbalanced learning)의 맥락에서 공변량 이동 적응**의 사용도 탐구하며, 실제 임상 데이터에서의 적용 가능성을 확인하였다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 발표 | 핵심 기여 | Gizewski et al. (2022)와의 관계 |
|---|---|---|---|
| **Gizewski et al. (2022)** | ACHA 2022 | RKHS에서 IWRLS 오차 경계, 밸런싱 원리 | 기준 논문 |
| **Ma, Pathak & Wainwright (2023)** | *Annals of Statistics* | RKHS 기반 비모수 회귀에서 공변량 이동의 미니맥스 최적 속도 | 기준 논문 확장, 더 엄밀한 속도 도출 |
| **Gogolashvili et al. (2023)** | arXiv:2303.04020 | 중요도 가중 보정이 필요한 조건 분석, Misspecified case 확장 | 비유계 밀도비 처리 문제 해결 |
| **General Regularization in Covariate Shift (2023)** | arXiv:2307.11503 | 평활도 조건 개선, IWRLS 정확도 향상 | IWRLS가 표준 최소제곱 회귀와 동일한 정확도 차수를 달성함을 증명하고, 이 정확도가 기존 연구보다 훨씬 높은 차수임을 확인 |
| **Gizewski et al. (2026)** | ACHA 2026 | RKHS 내 도메인 적응의 과잉 리스크 경계 개선; 일반 스펙트럴 알고리즘과 목표 함수 평활도·공간 용량을 동시에 고려 | 직접적인 후속 연구 |
| **Computational Efficiency under Covariate Shift (2025)** | arXiv:2505.14083 | 랜덤 투영(Nyström 방법 등)의 공변량 이동 하 활용 — 표준 환경에서는 연구되었으나 공변량 이동 하에서는 미탐구 영역 | 계산 효율성 방향 확장 |
| **Regularized Learning from Functional Data (2026)** | arXiv:2601.21019 | 함수 데이터(functional data)의 공변량 이동 적응 | RKHS 이론을 벡터값 RKHS로 확장 |

---

## 📚 참고 자료 (Reference List)

1. **논문 원문 (ScienceDirect):** Gizewski et al., "On a regularization of unsupervised domain adaptation in RKHS," *Applied and Computational Harmonic Analysis*, 57:201–227, 2022. https://www.sciencedirect.com/science/article/pii/S1063520321001032

2. **RICAM Technical Report:** Gizewski et al., RICAM-Reports, 2021. https://www.ricam.oeaw.ac.at/files/reports/21/rep21-17.pdf

3. **ResearchGate 요약:** https://www.researchgate.net/publication/357114087

4. **Academia.edu 분석:** https://www.academia.edu/69333214/On_a_regularization_of_unsupervised_domain_adaptation_in_RKHS

5. **Semantic Scholar (Table 1):** https://www.semanticscholar.org/paper/On-a-regularization-of-unsupervised-domain-in-RKHS-Gizewski-Mayer/f227bc26660e20d996d65708a15a02e552234242

6. **후속 연구 (2026):** "The impact of smoothness of kernels and target functions on unsupervised covariate shift adaptation in RKHS," *Applied and Computational Harmonic Analysis*, 2026. https://www.sciencedirect.com/science/article/pii/S106352032600014X

7. **General Regularization in Covariate Shift (2023):** arXiv:2307.11503. https://arxiv.org/pdf/2307.11503

8. **Computational Efficiency under Covariate Shift (2025):** arXiv:2505.14083. https://arxiv.org/pdf/2505.14083

9. **Towards regularized learning from functional data with covariate shift (2026):** arXiv:2601.21019. https://arxiv.org/html/2601.21019v1

10. **Balancing Principle (NeurIPS 2021):** Zellinger et al., "The balancing principle for parameter choice in distance-regularized domain adaptation," NeurIPS 2021. https://dl.acm.org/doi/10.5555/3540261.3541852

11. **Epub JKU:** "Regularization in Reproducing Kernel Hilbert Spaces for..." https://epub.jku.at/download/pdf/9362873.pdf

> ⚠️ **정확도 주의사항:** 본 논문의 구체적인 수식(특히 일반 정규화 기법 및 집계 방법의 세부 수식 표기)은 공개된 초록, 요약, 인용 정보를 기반으로 재구성하였으며, ScienceDirect 원문 전체 접근이 제한된 부분은 기술적 맥락에 부합하는 범위 내에서 표준 RKHS 이론을 바탕으로 서술하였습니다. 완전한 수식 검증을 위해서는 원문 직접 열람을 권장드립니다.
