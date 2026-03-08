# Introduction to Topological Data Analysis

---

## 1. 핵심 주장과 주요 기여 (요약)

이 문서는 저차원(low dimensions)에서의 위상적 데이터 분석(Topological Data Analysis, TDA)의 핵심 개념과 알고리즘을 소개합니다. 먼저 입력 데이터 표현을 형식화하고, 임계점(critical points), 지속적 호몰로지(Persistent Homology), Reeb 그래프, Morse-Smale 복합체(Morse-Smale Complexes) 등 TDA의 핵심 개념을 제시하며, 마지막으로 최신 알고리즘에 대한 간략한 리뷰를 제공합니다.

독자의 편의를 위해 가장 중요한 정의와 성질은 상자(box)로 강조되어 있으며, 추가 학습을 위해 Edelsbrunner와 Harer의 Computational Topology 입문서를 참조하도록 권장합니다.

**핵심 기여:**
- 위상수학의 추상적 개념을 **데이터 과학자와 컴퓨터 과학자가 접근 가능한 체계적 교육 자료**로 정리
- Simplicial Complex, Persistent Homology, Reeb Graph, Morse-Smale Complex를 **통합적 파이프라인**으로 연결
- 저차원 PL(Piecewise-Linear) 스칼라 필드에서의 위상적 분석을 위한 **실용적 알고리즘** 소개

---

## 2. 해결 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

데이터의 양, 다양성, 차원이 폭발적으로 증가함에 따라, 데이터의 기저 구조(underlying structure)를 식별, 추출, 활용하는 것이 데이터 분석과 통계적 학습에서 근본적으로 중요한 문제가 되었습니다.

TDA는 고차원, 불완전, 노이즈가 있는 데이터에서 정보를 추출하는 문제를 해결하기 위해, 특정 메트릭 선택에 민감하지 않으면서도 차원 축소와 노이즈에 대한 강건성을 제공하는 범용적 프레임워크를 제안합니다.

### 2.2 제안 방법 (핵심 수식 포함)

#### (A) 단체 복합체 (Simplicial Complex)와 데이터 표현

Tierny는 위상 공간 $X$, 그 경계 $\partial X$, 다양체 $M$, $d$-단체 $\sigma$, 단체 복합체 $K$, 삼각분할 $\mathcal{T}$, Betti 수 $\beta_i$, 오일러 특성 $\chi$, PL 스칼라 필드 $f: \mathcal{T} \to \mathbb{R}$ 등의 표기법을 체계적으로 정의합니다.

**오일러 특성(Euler Characteristic)**은 교대 합으로 정의됩니다:

$$\chi = \sum_{k=0}^{d} (-1)^k \beta_k$$

여기서 $\beta_k$는 $k$차 Betti 수로, $\beta_0$은 연결 성분 수, $\beta_1$은 1차원 루프(구멍) 수, $\beta_2$는 2차원 공동(cavity) 수를 나타냅니다.

호몰로지(Homology)는 대수적 위상수학의 고전적 개념으로, 위상적 특징의 개념을 대수적으로 형식화하고 다루기 위한 강력한 도구입니다. 각 차원 $k$에 대해 $k$차원 "구멍"은 벡터 공간 $H_k$로 표현되며, 그 차원은 직관적으로 독립적인 특징의 수입니다. 예를 들어, 0차 호몰로지 군 $H_0$은 연결 성분을, 1차 호몰로지 군 $H_1$은 1차원 루프를, 2차 호몰로지 군 $H_2$는 2차원 공동을 나타냅니다.

#### (B) 지속적 호몰로지 (Persistent Homology)

이것이 TDA의 **핵심 도구**입니다.

Persistent Homology는 다양한 공간 해상도(spatial resolution)에서 위상적 특징을 계산하는 방법으로, 넓은 범위의 공간 스케일에서 감지되는 더 지속적인(persistent) 특징은 노이즈나 샘플링의 인공물(artifacts)이 아닌 기저 공간의 진정한 특징을 나타낼 가능성이 높다고 간주합니다.

**Filtration** 과정: 스칼라 함수 $f: \mathcal{T} \to \mathbb{R}$가 주어지면, 부분 수준 집합(sublevel set)의 중첩된 시퀀스를 구성합니다:

$$\emptyset = \mathcal{T}_{-\infty} \subseteq \mathcal{T}_{a_1} \subseteq \mathcal{T}_{a_2} \subseteq \cdots \subseteq \mathcal{T}_{a_n} = \mathcal{T}$$

여기서 $\mathcal{T}_a = f^{-1}((-\infty, a])$ 입니다.

각 위상적 특징 $\gamma$는 **탄생(birth)** 시점 $b(\gamma)$와 **사멸(death)** 시점 $d(\gamma)$를 가지며, 이를 **지속 다이어그램(Persistence Diagram)** 상의 점 $(b(\gamma), d(\gamma))$로 표현합니다.

직관적으로, 바코드에서 간격이 길수록(또는 다이어그램에서 대각선으로부터 멀수록), 해당 호몰로지적 특징이 더 지속적(persistent)이며 따라서 filtration 전체에서 더 관련성이 높다고 볼 수 있습니다.

**지속성(Persistence)**:

$$\text{pers}(\gamma) = d(\gamma) - b(\gamma)$$

#### (C) 안정성 정리 (Stability Theorem)

TDA의 가장 핵심적인 이론적 기여는 **안정성 정리**입니다:

연속 tame 함수 $f, g: X \to \mathbb{R}$에 대해, 임의의 $p \geq 0$에서 $p$차원 지속 다이어그램 사이의 bottleneck 거리는 함수 간 거리를 초과하지 않습니다:

$$d_B(\text{Dgm}_p(f), \text{Dgm}_p(g)) \leq \|f - g\|_{\infty}$$

Wasserstein 거리를 사용한 안정성도 성립합니다: $W_\infty(\text{Dgm}(f), \text{Dgm}(g)) \leq \|f - g\|_\infty$ (단, $f$와 $g$가 모두 tame — 유한한 임계값과 유한 차수 호몰로지 군을 가짐).

이 안정성 정리는 **입력 데이터의 작은 섭동이 위상적 요약의 작은 변화만을 초래**함을 보장합니다.

#### (D) Reeb 그래프와 Morse-Smale 복합체

Tierny는 임계점, 지속적 호몰로지 외에도 **Reeb 그래프**와 **Morse-Smale 복합체**를 TDA의 핵심 개념으로 제시합니다.

Reeb 그래프는 스칼라 함수 $f$에 대해 등위면(level set)의 연결 성분을 추적하는 1차원 골격 구조입니다:

$$\mathcal{R}_f = \mathcal{T} / \sim_f$$

여기서 $x \sim_f y \iff f(x) = f(y)$이고 $x, y$가 $f^{-1}(f(x))$의 같은 연결 성분에 속하는 관계입니다.

### 2.3 모델 구조: TDA 파이프라인

TDA는 21세기 초반 응용 (대수적) 위상수학과 계산 기하학의 다양한 연구에서 등장한 분야입니다. 추출된 위상적/기하학적 정보의 가시화와 해석을 넘어, 입력 데이터의 섭동이나 노이즈 존재에 대한 안정성(stability)을 보여주는 것이 핵심 과제이며, 이를 위해 추론된 특징의 통계적 행태를 이해하는 것도 중요합니다.

TDA 파이프라인은 다음과 같은 단계로 구성됩니다:

1. **데이터 → 단체 복합체 구성** (Vietoris-Rips, Čech 등)
2. **Filtration 구성** (부분수준 집합)
3. **지속적 호몰로지 계산** → 지속 다이어그램/바코드 출력
4. **위상적 특징 → 벡터화/특징 추출 → 머신러닝/딥러닝 연계**

### 2.4 한계

최근까지 TDA의 이론적 측면은 대부분 결정론적(deterministic) 접근에 의존해왔습니다. 이러한 결정론적 접근은 데이터의 확률적 특성과 추론된 위상적 양의 본질적인 변동성을 고려하지 않습니다. 결과적으로, 대부분의 관련 방법은 탐색적(exploratory) 수준에 머물며, 정보와 "위상적 노이즈(topological noise)" 사이를 효율적으로 구별하지 못합니다.

Persistent Homology는 높은 수준의 추상화, 비위상적 변화에 대한 비민감성(insensitivity to non-topological changes), 그리고 포인트 클라우드 데이터에 대한 제한 등 많은 한계를 가지고 있습니다.

---

## 3. 모델의 일반화 성능 향상 가능성

이 부분은 Tierny의 교육 자료 자체보다는 TDA 프레임워크가 **딥러닝에 적용될 때의 일반화 성능 향상**과 관련된 중요한 연구 방향입니다.

### 3.1 안정성에 기반한 일반화 보장

Persistent Homology는 정확한 의미에서 안정적(stable)이며, 이는 노이즈에 대한 강건성을 제공합니다. Bottleneck 거리는 지속 다이어그램의 공간 위의 자연스러운 메트릭입니다.

이 안정성은 **학습된 표현(representation)의 일반화**에 직접적 함의를 가집니다. 학습 데이터와 테스트 데이터 사이의 위상적 구조가 일관되면, 모델의 예측도 안정적일 수 있습니다.

### 3.2 위상적 정규화 (Topological Regularization)

정규화는 과적합을 방지하고 딥러닝 모델의 일반화 능력을 향상시키는 데 핵심적인 역할을 합니다. 향후 연구에서는 TDA 기반 정규화 기법을 딥러닝 프레임워크에 통합하는 방향을 탐구할 것이며, 의미 있는 위상적 특징을 포착하도록 위상적 페널티나 제약조건을 도입하여 모델의 일반화와 강건성을 향상시킬 수 있습니다.

Chen et al. (2019)은 Persistent Homology에 기반한 정규화기(regularizer)가 더 나은 일반화와 더 높은 예측 성능으로 이어진다는 것을 보여주었습니다.

위상적 정규화의 수식 형태:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda \cdot \mathcal{L}_{\text{topo}}$$

여기서 $\mathcal{L}_{\text{topo}}$는 지속 다이어그램으로부터 추출된 위상적 특징에 기반한 페널티 항입니다.

### 3.3 표현 학습에서의 위상적 분리

과매개변수화(over-parameterized)된 딥러닝 네트워크에서 소규모 샘플로의 일반화는 어렵습니다. Persistent Homology를 활용하여 특징 추출기가 유도하는 push-forward 확률 측도의 "분리(separation)" 개념을 정의하고, 이 성질을 강제하면 더 나은 일반화로 이어진다는 것을 이론적으로 증명했습니다. 위상적 정보를 추출하기 위한 새로운 가중 함수와, 위상 인식 방식(topology-aware manner)으로 표현 학습을 유도하는 3개 항목을 포함한 새로운 정규화기를 제안했습니다.

### 3.4 노이즈 강건성과 분포 이동 하에서의 일반화

Deep TDA는 복잡한 데이터 구조와 노이즈 데이터셋을 처리하는 데 우수한 성능을 보이며, 이러한 조건에서 정확도를 유지하는 강건성을 보여줍니다. 또한 Deep TDA 모델은 더 해석 가능한(interpretable) 것으로 빈번히 발견되며, 이는 모델 결정을 이해하는 것이 중요한 핵심 응용에서 큰 장점입니다.

지속적 호몰로지가 표현 학습에서 메트릭 안정성을 향상시키는 영역, 특히 중미(heavy-tailed) 노이즈 하에서, 새로운 안정성 결과를 수립하여 해당 방법이 달성하는 강건성 이득에 대한 엄밀한 기반을 제공합니다.

---

## 4. 연구 영향 및 향후 고려사항

### 4.1 앞으로의 연구에 미치는 영향

Tierny의 입문서를 포함한 TDA 프레임워크는 다음과 같은 분야에 광범위한 영향을 미쳤습니다:

TDA는 복잡한 분자 데이터에서 강건하고 다중스케일이며 해석 가능한 특징을 추출하기 위한 강력한 프레임워크로 부상했습니다. Persistent Homology, Persistent Laplacian, 위상적 머신러닝 등의 혁신을 조명하며, 생체분자 안정성, 단백질-리간드 상호작용, 약물 발견, 재료 과학, 바이러스 진화 등에 걸친 TDA의 변혁적 영향을 탐구합니다.

Topological Deep Learning (TDL)은 TDA와 딥러닝 기법의 원리를 결합하는 신흥 분야입니다. TDA는 데이터의 형태(shape)에 대한 통찰을 제공하며, 변형과 노이즈에 대한 강건성을 보이면서 다차원 데이터의 전역적 기술을 얻습니다. 이러한 속성은 딥러닝 파이프라인에서 바람직하지만, 전통적으로 비-TDA 전략을 사용하여 얻어져 왔습니다.

### 4.2 향후 연구 시 고려할 점

1. **계산 복잡도**: Persistent Homology 계산의 기본 알고리즘은 단체(simplex)의 수에 대해 최악의 경우 3차(cubical) 복잡도로 수행됩니다. 가장 빠른 알려진 알고리즘은 행렬 곱셈 시간(matrix multiplication time)으로 실행됩니다.

2. **통계적 접근 필요성**: TDA에 대한 통계적 접근의 주요 목표는 (1) TDA 방법의 일관성 증명 및 수렴 속도 연구, (2) 위상적 특징에 대한 신뢰 영역 제공, (3) 관측 데이터의 함수로서 관련 스케일 선택, (4) 아웃라이어 처리 및 강건한 TDA 방법 제공 등입니다.

3. **Persistent Homology의 한계 극복**: Persistent Homology의 높은 추상화, 비위상적 변화에 대한 비민감성, 포인트 클라우드 데이터에 대한 제한 등의 한계를 극복하기 위해, Persistent Topological Laplacian과 Dirac 연산자 등이 위상적 불변량과 호모토피적 진화를 모두 포착하는 스펙트럼 표현을 제공합니다.

4. **딥러닝 통합의 어려움**: 바코드와 지속 다이어그램 같은 TDA 구성물을 현재의 딥러닝 알고리즘과 결합하는 것의 어려움이 주요 장벽입니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구/년도 | 핵심 기여 | Tierny 대비 발전 |
|-----------|----------|-----------------|
| **Chazal & Michel (2021)** | TDA의 통계적 기초 포괄적 입문 | 통계적 추론과 실용적 GUDHI 라이브러리 활용 추가 |
| **Persistent Laplacian (Wang et al., 2020–2024)** | Persistent Combinatorial Laplacian 도입 | Betti 수 넘어 기하학적 형태 변화까지 포착 |
| **Topological Regularization (Chen et al., 2019; 후속 2023)** | PH 기반 분류기 정규화기 | 일반화 성능 향상을 이론적·실험적으로 입증 |
| **TDL Survey (Zia et al., 2024)** | 위상적 딥러닝 종합 서베이 | TDA-DL 통합의 체계적 분류 |
| **Normalized Bottleneck Distance (2024)** | 스케일 불변 지속 다이어그램 거리 | 차원 축소 시 호몰로지 보존 평가 개선 |
| **TopoX Suite (Hajij et al., 2024)** | TDL 소프트웨어 생태계 | 실용적 구현 도구 표준화 |

### 주요 최신 발전:

Persistent Topological Laplacian이 다양한 데이터 구조를 위해 개발되어 머신러닝 모델에 통합될 수 있는 표현을 제공합니다. 이 스펙트럼적 접근에서, 조화 스펙트럼(harmonic spectra)은 위상적 불변량의 변화를 포착하여 Persistent Homology가 포착하는 위상 정보를 복원할 수 있고, 비조화 스펙트럼(non-harmonic spectra)은 filtration 과정에서의 기하학적 형태 변화를 인코딩합니다.

Topological Deep Learning은 이진 또는 고차 관계적 데이터로부터의 학습을 위한 프레임워크를 제공하며, 기저 데이터 공간의 위상에 따라 신경망 아키텍처 선택을 결정하고, 다양체에 내재한 규칙성을 포착하며, 위상적 등변성(topological equivariance)을 포착합니다.

최근 연구에서는 가중치나 활성화의 Persistent Homology와 일반화 능력 사이의 연결 고리를 강조하며, 상관관계 기반 지속 다이어그램을 활용한 정규화 항 공식화가 신경망 학습 중 전체 정확도를 향상시키고 전통적 정규화 접근을 능가하는 결과를 보여주고 있습니다.

---

## 참고자료 출처

1. **Julien Tierny**, *Introduction to Topological Data Analysis*, Sorbonne Universités, 2017. ([PDF](https://www-apr.lip6.fr/~tierny/stuff/teaching/tierny_topologicalDataAnalysis.pdf))
2. **HAL Archive**: [cel-01581941](https://inria.hal.science/cel-01581941v1)
3. **Chazal & Michel**, *An Introduction to Topological Data Analysis: Fundamental and Practical Aspects for Data Scientists*, Frontiers in AI, 2021. ([Link](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2021.667963/full))
4. **Wikipedia**, *Persistent Homology*. ([Link](https://en.wikipedia.org/wiki/Persistent_homology))
5. **Cohen-Steiner, Edelsbrunner, Harer**, *Stability of Persistence Diagrams*, 2005. ([PDF](http://www.math.uchicago.edu/~shmuel/AAT-readings/Data%20Analysis%20/Edelsbrunner,%20Harer,%20Stability.pdf))
6. **Zia et al.**, *Topological deep learning: a review of an emerging paradigm*, Artificial Intelligence Review, 2024. ([Link](https://link.springer.com/article/10.1007/s10462-024-10710-9))
7. **Wang et al.**, *TDA and TDL beyond persistent homology: a review*, Artificial Intelligence Review, 2025. ([Link](https://link.springer.com/article/10.1007/s10462-025-11462-w))
8. **Topological Regularization for Representation Learning via Persistent Homology**, Mathematics, 2023. ([ResearchGate](https://www.researchgate.net/publication/368583212))
9. **Hajij et al.**, *Position: Topological Deep Learning is the New Frontier for Relational Learning*, ICML, 2024. ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11973457/))
10. **Morozov**, *Homological Illusions of Persistence and Stability*, PhD Thesis. ([PDF](https://www.mrzv.org/publications/thesis/phd/))
11. **TDA 2024 Workshop** — Persistent Homology와 일반화 관련 발표. ([Schedule](https://sites.google.com/view/tda2024/schedule))
12. **TDA in Molecular Sciences Review**, J. Chem. Inf. Model., 2025. ([Link](https://pubs.acs.org/doi/10.1021/acs.jcim.5c02266))
13. **Wikipedia**, *Topological Data Analysis*. ([Link](https://en.wikipedia.org/wiki/Topological_data_analysis))
14. **ETH Zürich**, *Distances and Stability*, Chapter 6 lecture notes. ([PDF](https://ti.inf.ethz.ch/ew/courses/TDA24/Chapter6.pdf))
15. **Kerber & Wang**, *Computing the Bottleneck Distance between Persistent Homology Transforms*, arXiv:2512.00821, 2025.

---

> **참고 사항**: Tierny의 자료는 학술 논문(peer-reviewed paper)이 아닌 **박사과정 수준의 교육 자료(lecture notes)**이므로, 새로운 모델이나 실험 결과를 제시하기보다는 기존 TDA 이론과 알고리즘의 **체계적 정리와 교육적 전달**에 초점을 둡니다. 위에서 논의한 일반화 성능 향상, 딥러닝 통합 등의 발전은 이 교육 자료 위에 구축된 **후속 연구**들의 기여입니다.
