# On the Stability of Persistent Entropy and New Summary Functions for Topological Data Analysis

**저자:** Nieves Atienza, Rocío González-Díaz, Manuel Soriano-Trigueros (세비야 대학교 응용수학과)
**출판:** *Pattern Recognition*, Vol. 107, 107509, 2020

---

## 1. 핵심 주장 및 주요 기여 요약

Persistent homology와 persistent entropy는 최근 패턴 인식에 유용한 도구로 부상하였다. 이 논문은 persistent entropy가 입력 데이터의 작은 교란(perturbation)에 대해 안정적(stable)이며, 스케일 불변(scale invariant)인 조건을 규명한다. 또한 persistent entropy와 Betti curve를 결합한 두 가지 새로운 안정적 요약 함수(summary function)를 제안하고, 이를 재료 분류(material classification) 과제에 적용하여 머신러닝 및 패턴 인식에서의 유용성을 보인다.

주요 기여는 다음 세 가지이다:

1. **Persistent entropy의 안정성 정리(Stability Theorem)**: 기존에는 persistent entropy의 안정성에 대한 부분적 결과만 존재했으며, 공식적인 완전한 연구가 수행된 적이 없었다. 이 논문의 핵심 목표는 persistent entropy에 대한 일반적 안정성 결과를 제공하고, 스케일 불변성이 성립하는 조건을 규명하는 것이다.

2. **새로운 요약 함수 정의**: Persistent entropy가 안정적 통계량임을 정당화하고, 이를 활용하여 ES-function(Entropy Summary function)과 그 정규화 버전인 NES-function(Normalized Entropy Summary function)이라는 새로운 안정적 요약 함수를 정의하였다.

3. **노이즈 강건성**: 이 요약 함수들은 Betti function보다 노이즈에 더 강건(robust)하며, 서로 다른 특징들을 구별할 수 있다.

---

## 2. 상세 분석: 문제, 방법, 구조, 성능, 한계

### 2.1 해결하고자 하는 문제

분류 과제에서 persistence barcode를 단일 수치(persistent entropy)로 요약하는 것은 무한 차원 공간을 1차원으로 사영하는 것이므로 판별력이 부족할 수 있다. 또한, persistent entropy의 안정성에 대한 부분적 결과만 제시되어 있었고, 체계적이고 포괄적인 안정성 증명은 존재하지 않았다.

구체적으로 두 가지 핵심 문제를 다룬다:

- **안정성 문제**: 입력 데이터의 작은 변화가 persistent entropy 값에 미치는 영향을 제어할 수 있는가?
- **표현력 문제**: 단일 숫자 대신 함수 형태의 요약이 더 풍부한 분류 정보를 제공할 수 있는가?

### 2.2 제안하는 방법 및 수식

#### (1) Persistent Entropy 정의

Persistence barcode $A = \{[a_i, b_i)\}\_{i=1,\ldots,n}$에 대해, 각 간격(bar)의 길이를 $\ell_i = b_i - a_i$로 정의하고, 총 길이 $L = \sum_{i=1}^{n} \ell_i$로 정규화하면, persistent entropy는 차이 $d - b$ ("lifetimes")의 컬렉션에 대한 (base 2) Shannon entropy로 계산되며, 모든 차이의 합으로 정규화된다.

$$E(A) = -\sum_{i=1}^{n} p_i \log_2 p_i, \quad \text{where} \quad p_i = \frac{\ell_i}{L}$$

#### (2) 안정성 정리 (Theorem 3.12)

$A, B \in \mathcal{B}_F$이고, $d_p$를 $p$-th Wasserstein 거리($1 \leq p \leq \infty$)라 할 때, 간격의 최대 개수와 간격 길이 합의 최솟값을 고정하면, persistent entropy $E$는 $(\mathcal{B}_F, d_p)$ 위에서 연속(continuous)이다:

$$\forall \varepsilon > 0, \; \exists \delta > 0 \; \text{such that} \; d_p(A, B) \leq \delta \implies |E(A) - E(B)| \leq \varepsilon$$

핵심 정리(Theorem 3.12)의 구체적 바운드는 다음과 같다. 상대적 교란(relative perturbation) $r_p(A,B) \leq 1$을 가정하면:

$$|E(A) - E(B)| \leq r_p(A, B) \cdot \log_2\left(\frac{n_{\gamma_r}}{r_p(A, B)}\right)$$

여기서 $r_p(A,B) = \frac{d_1(\psi(A), \psi(B))}{1}$로 정규화된 Wasserstein 거리이며, $n_{\gamma_r}$은 최적 매칭의 카디널리티이다. 이 부등식은 Shannon entropy의 연속성 정리에 기반한다.

#### (3) Betti Curve

Betti curve는 시간 $t$에서 "살아있는" 간격의 수를 세는 함수이다:

```math
\beta(A)(t) = \sum_{i=1}^{n} \delta_i(t), \quad \delta_i(t) = \begin{cases} 1 & \text{if } t \in [a_i, b_i) \\ 0 & \text{otherwise} \end{cases}
```

#### (4) ES-function (Entropy Summary Function)

Betti curve와 유사하지만, Betti 수 대신 persistent entropy를 사용하는 새로운 piece-wise constant 함수(step function)를 정의한다. ES function은 전체 다이어그램의 엔트로피 대신, 특정 시점 $t$에서 살아있는 간격들만 고려하여 엔트로피를 계산한다.

구체적으로, 시점 $t$에서 살아있는 간격들의 집합 $A_t = \{[a_i, b_i) \in A : a_i \leq t < b_i\}$에 대해:

$$\text{ES}(A)(t) = E(A_t) = -\sum_{[a_i, b_i) \in A_t} \frac{\ell_i}{L_t} \log_2 \frac{\ell_i}{L_t}$$

여기서 $L_t = \sum_{[a_i, b_i) \in A_t} \ell_i$이다.

#### (5) NES-function (Normalized ES-function)

ES-function을 $\log_2(\beta(A)(t))$로 정규화한 함수이다:

$$\text{NES}(A)(t) = \frac{\text{ES}(A)(t)}{\log_2(\beta(A)(t))}$$

Persistent entropy 자체의 정규화와 달리, 이 NES-function의 정규화는 안정성을 유지한다.

#### (6) 무한 길이 간격에 대한 사영(Projection)

Persistence barcode에 무한 길이 간격이 있는 경우, 이를 유한 길이 간격으로 변환하는 사영을 정의한다. 사영 방법에 따라 persistent entropy가 더 이상 안정적이거나 스케일 불변이 아닐 수 있다. 이 논문은 안정적이고 스케일 불변인 사영 함수 $\tau_\lambda$, $\mu_\lambda$, $\nu_{\lambda,p}$를 제시하고 그 안정성을 증명하였다 (Proposition 3.16).

### 2.3 모델 구조

이 논문은 딥러닝 모델이 아닌, **위상적 데이터 분석(TDA)의 수학적 프레임워크**를 다룬다. 전체 파이프라인은 다음과 같다:

1. **데이터** → 포인트 클라우드 등
2. **Simplicial Complex 구축** → Vietoris-Rips filtration 등
3. **Persistent Homology 계산** → Persistence barcode 생성
4. **요약 함수 적용** → $E(A)$, $\text{ES}(A)(t)$, $\text{NES}(A)(t)$
5. **머신러닝 분류기** → Random Forest 등에 벡터화된 특징 입력

### 2.4 성능 향상

이 논문에서 persistent entropy의 안정성이 정당화되었고, ES-function 및 NES-function이 일반적으로 노이즈가 있는 상황에서 Betti curve보다 더 나은 성능을 보이며, 머신러닝 과제에 유용함이 입증되었다.

- NES-function은 일부 경우에서 Betti curve와 persistence images보다 우수한 성능을 보이며, 각 함수가 제공하는 정보는 상호 보완적(complementary)이다.
- Betti curve와 persistent entropy가 동일한 값을 가지는 두 개의 서로 다른 persistence barcode를 ES-function은 구별할 수 있음을 보였다.

### 2.5 한계

- Persistence barcode를 단일 숫자(persistent entropy)로 요약하면, 무한 차원 공간을 1차원으로 사영하는 것이므로 정보 손실이 발생한다.
- 향후 과제로, 다른 위상적 벡터화 방법들과의 심층 비교가 필요하다.
- 안정성 정리에서 $r_p(A,B) \leq 1$이라는 조건이 필요하며, 큰 교란에 대해서는 바운드가 보장되지 않는다.
- ES-function과 NES-function은 birth/death time의 일부 정보를 반영하지만, 여전히 완전한 persistence diagram 정보를 복원할 수 없다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 안정성과 일반화의 관계

안정성이란 입력 데이터의 노이즈에 의한 교란을 "제어"하는 바운드가 존재함을 의미한다. 이는 머신러닝에서의 **일반화(generalization)**와 직접적으로 연결된다:

$$d_p(A, B) \leq \delta \implies |E(A) - E(B)| \leq \varepsilon$$

이 안정성 조건은 다음과 같은 일반화 성능 향상 메커니즘을 제공한다:

- **노이즈 강건성**: 테스트 데이터가 훈련 데이터와 약간 다르더라도, 특징값(feature)의 변화가 제한된다. 이는 과적합(overfitting)을 방지하는 효과를 가진다.
- **스케일 불변성**: Persistent entropy는 스케일 불변적인 위상 변수로, 데이터 분석 문제의 차원을 줄이는 데 사용할 수 있다. 데이터의 스케일에 의존하지 않으므로, 다양한 스케일의 데이터에서 일관된 특징 추출이 가능하다.

### 3.2 함수형 요약의 이점

- Betti curve, ES-function, NES-function과 같은 **함수 형태의 요약**은 단일 숫자보다 풍부한 정보를 보존하므로, 다운스트림 분류기의 판별력이 향상된다.
- Persistence curves(ES-function 포함)의 다양한 변형을 조합하면 가장 좋은 결과를 산출하며, 이들은 상호 보완적임이 확인되었다.

### 3.3 일반화 향상을 위한 전략

1. **앙상블 벡터화**: ES-function, NES-function, Betti curve, persistence landscape 등을 연결(concatenate)하여 사용
2. **안정적 사영 선택**: 무한 길이 간격 처리 시 안정적이면서 스케일 불변인 사영 함수 활용
3. **정규화**: NES-function의 정규화를 통해 barcode의 상대적 복잡도를 반영

---

## 4. 연구 영향 및 향후 연구 고려점

### 4.1 연구 영향

1. **TDA의 이론적 기반 강화**: 이 논문의 안정성 결과는 실제 응용에서 방법론의 강건성을 보장하는 데 명시적으로 활용되었다.

2. **라이브러리 통합**: Persistent entropy는 GUDHI, scikit-TDA, Giotto 라이브러리에 이미 구현되어 있어, 실용적 영향이 크다. TDAvec 패키지에서도 persistent entropy summary function이 벡터화 기법으로 구현되어 있다.

3. **Persistence Curves 프레임워크에 대한 영감**: Chung & Lawson (2022)의 Persistence Curves(PC) 프레임워크는 ES-function을 포함한 여러 요약 함수들을 통합적으로 다루며, 이 논문의 ES-function이 "life entropy"로 재명명되어 활용되었다.

4. **다양한 응용 분야**: Persistent entropy는 상피 조직(epithelial tissue)의 세포 분포 차이 감지에도 활용되었으며, 스케일 불변이고 노이즈에 강건하며 전역적 위상 특징에 민감하다.

### 4.2 향후 연구 시 고려점

1. **계산 복잡도**: 대규모 데이터에서 persistence homology 계산의 시간 복잡도가 병목이 될 수 있으며, 효율적 근사 방법이 필요
2. **고차원 정보 손실**: 함수형 요약도 여전히 persistence diagram의 전체 정보를 보존하지 못하므로, 정보 손실량에 대한 정량적 분석이 필요
3. **딥러닝 통합**: 머신러닝에서는 일반적으로 모든 샘플이 균일한 입력 벡터를 가져야 하므로, persistence barcode 등의 TDA 정보를 고정 길이 특징 벡터로 변환하는 벡터화(vectorization)가 필요하다. 최근 topological deep learning과의 통합이 활발히 연구되고 있다.
4. **다른 위상적 요약과의 비교**: Persistence landscape, persistence image, kernel method 등과의 체계적 비교 연구가 아직 충분하지 않다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 핵심 내용 | 이 논문과의 관계 |
|------|-----------|-----------------|
| **Chung & Lawson (2022)**, *Persistence Curves* | Persistence Curves(PC)라는 통합 프레임워크를 개발하여, Persistence Landscapes 등 기존 요약들이 PC 프레임워크의 특수한 경우임을 보이고, 새로운 요약 함수들의 안정성 분석 기반을 제공 | ES-function을 "life entropy"로 PC 프레임워크에 포함 |
| **Chung et al. (2022)**, *Gaussian Persistence Curves* | Gaussian persistence curves의 통계적 성질과 안정성/단사성(injectivity)을 연구 | ES-function의 확장된 분석 |
| **TDAvec Package (Islambekov et al., 2024)** | Persistence diagram의 벡터화를 간소화하는 소프트웨어 패키지로, persistent entropy summary function을 포함한 다양한 벡터화 기법을 효율적으로 구현 | ES-function의 실용적 구현 |
| **Persistence path signatures (2020+)** | Barcode를 벡터 공간의 경로(path)로 변환한 후 path signature를 계산하는 특징 맵으로, 보편성(universality)과 특성성(characteristicness)을 가지며 분류 벤치마크에서 최첨단 성능을 달성 | 대안적 벡터화 접근법 |
| **TDA beyond persistent homology (Wei et al., 2025)** | Persistent topological Laplacians 및 Dirac 연산자가 위상 불변량과 호모토피 진화를 동시에 포착하는 스펙트럴 표현을 제공 | Persistent homology의 한계를 극복하는 차세대 TDA |
| **Vectorized Persistence Blocks (VPB)** | PD를 $\mathbb{R}^n$의 벡터로 변환하는 계산 효율적 프레임워크로, 입력 노이즈에 대한 안정성, 낮은 계산 비용, 유연성 등 바람직한 성질을 보유 | 안정적 벡터화의 대안 |

### 핵심 트렌드 분석

Persistence diagram은 TDA의 주요 도구이지만, 내적(inner product)이 없는 공간에 존재하므로 머신러닝 알고리즘에 직접 사용하기 어렵다. 이러한 이유로, persistence diagram을 머신러닝과 호환되는 형태로 변환하는 것이 현재 TDA에서 중요한 연구 주제이다.

2020년 이후의 연구들은 **안정성**과 **표현력** 사이의 균형을 맞추는 방향으로 발전하고 있으며, 이 논문의 ES-function과 NES-function은 이러한 발전의 중요한 초석이 되었다.

---

## 참고자료

1. **Atienza, N., González-Díaz, R., Soriano-Trigueros, M.** "On the stability of persistent entropy and new summary functions for topological data analysis." *Pattern Recognition*, 107, 107509, 2020. — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0031320320303125), [arXiv:1803.08304](https://arxiv.org/abs/1803.08304)
2. **Chung, Y.M., Lawson, A.** "Persistence Curves: A canonical framework for summarizing persistence diagrams." *Advances in Computational Mathematics*, 48(1), 1-42, 2022. — [Springer](https://link.springer.com/article/10.1007/s10444-021-09893-4)
3. **Wei et al.** "Topological data analysis and topological deep learning beyond persistent homology: a review." *Artificial Intelligence Review*, Springer, 2025. — [Springer](https://link.springer.com/article/10.1007/s10462-025-11462-w)
4. **Hensel, F. et al.** "Topological data analysis and machine learning." *Advances in Physics: X*, 8(1), 2202331, 2023. — [Taylor & Francis](https://www.tandfonline.com/doi/full/10.1080/23746149.2023.2202331)
5. **GUDHI Library** — Persistent Entropy Tutorial. [GitHub](https://github.com/GUDHI/TDA-tutorial/blob/master/Tuto-GUDHI-persistent-entropy.ipynb)
6. **giotto-tda** — PersistenceEntropy Documentation. [giotto-ai.github.io](https://giotto-ai.github.io/gtda-docs/latest/modules/generated/diagrams/features/gtda.diagrams.PersistenceEntropy.html)
7. **TDAvec** — Vector Summaries of Persistence Diagrams. [arXiv:2411.17340](https://arxiv.org/html/2411.17340v3)
8. **Semantic Scholar** — Paper page. [Semantic Scholar](https://www.semanticscholar.org/paper/On-the-stability-of-persistent-entropy-and-new-for-Atienza-Gonzalez-Diaz/96eb7405f71af950dc8687846365ee552e542712)
