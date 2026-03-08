# Persistent Homology — a Survey

---

## 1. 핵심 주장과 주요 기여 요약

"Persistent homology is an algebraic tool for measuring topological features of shapes and functions. It casts the multi-scale organization we frequently observe in nature into a mathematical formalism."

이 논문의 핵심 기여는 다음과 같습니다:

- Persistent homology의 짧은 역사와 기본 개념을 체계적으로 정리하고, 알고리즘에 초점을 맞추며 생체분자, 생물학적 네트워크, 데이터 분석, 기하학적 모델링 등 다양한 응용과의 연결을 소개합니다.

- Persistent homology는 대수적 위상수학에 기반한 방법으로, 여러 해상도(resolution)에 걸쳐 지속되는(persist) 점(point)들의 공간에 대한 위상적 특징(topological features)을 추정합니다.

---

## 2. 상세 분석: 문제, 방법, 구조, 성능, 한계

### 2.1 해결하고자 하는 문제

Persistent homology가 다루는 핵심 문제는 **데이터의 다중 스케일(multi-scale) 위상적 구조를 정량적으로 측정하고 분류하는 것**입니다.

전통적 호몰로지(homology)는 고정된 단일 스케일에서의 위상적 특징만 포착하므로, 노이즈가 포함된 현실 데이터에서 어떤 특징이 "진짜"이고 어떤 것이 노이즈인지를 구별하기 어렵습니다. Persistent homology는 대수적 위상수학에 기반하여, 여러 해상도에 걸쳐 지속되는 점들의 공간에 대한 위상적 특성이나 불변 특징(invariant features)을 결정하는 데 사용됩니다.

### 2.2 제안하는 방법 (수식 포함)

#### (a) Simplicial Complex와 Filtration

데이터 포인트 집합 $X \subset \mathbb{R}^d$가 주어졌을 때, **Vietoris–Rips complex** 또는 **Čech complex**를 통해 위상적 구조를 구성합니다. 파라미터 $\epsilon > 0$에 대해:

$$\text{VR}_\epsilon(X) = \{ \sigma \subseteq X \mid \text{diam}(\sigma) \leq \epsilon \}$$

**Filtration**은 스케일 파라미터 $\epsilon$의 증가에 따른 중첩된 복체(complex)의 열입니다:

$$\emptyset = K_0 \subseteq K_1 \subseteq K_2 \subseteq \cdots \subseteq K_n = K$$

PH는 포인트 클라우드를 중첩된 복체(nested complexes)의 필터링된 시퀀스로 표현하고, 이를 바코드 등의 새로운 표현으로 변환한 뒤, 지속적인 위상 특징에 기반하여 통계적/정성적으로 해석합니다.

#### (b) Persistent Homology Group

Filtration 내에서 두 인덱스 $i \leq j$에 대해, 포함 사상(inclusion map) $\iota^{i,j}: K_i \hookrightarrow K_j$가 유도하는 호몰로지 사상이 정의됩니다:

$$f^{i,j}_p: H_p(K_i) \rightarrow H_p(K_j)$$

여기서 $H_p(\cdot)$는 $p$차원 호몰로지 군(homology group)입니다. **$p$-th persistent homology group**은:

$$H^{i,j}_p = \text{im}(f^{i,j}_p)$$

로 정의되며, 이는 $K_i$에서 태어나서(born) $K_j$까지 살아남은(survive) $p$차원 위상 특징을 포착합니다.

#### (c) Persistence Diagram과 Barcode

Persistence diagram은 Edelsbrunner, Letscher, Zomorodian에 의해 도입되었으며, 함수의 하위 레벨 집합(sub-level sets)의 호몰로지 차이를 인코딩하는 확장 평면의 점 집합입니다. 각 점은 하나의 특징에 대응합니다.

각 위상 특징이 태어난 시점(birth) $b_i$와 사라진 시점(death) $d_i$를 쌍으로 기록합니다:

$$\text{Dgm}(f) = \{(b_i, d_i) \mid b_i < d_i\} \subset \mathbb{R}^2$$

**Persistence(지속성)**은 각 특징의 수명으로 정의됩니다:

$$\text{pers}(b_i, d_i) = d_i - b_i$$

짧은 바(bar)는 데이터의 노이즈나 아티팩트에 해당하며, 긴 바는 관련성 있는(relevant) 위상적 특징에 대응합니다.

#### (d) 안정성 정리 (Stability Theorem)

Edelsbrunner과 Harer가 공동 저자로 참여한 핵심 결과 중 하나인 **안정성 정리**는:

$$d_B(\text{Dgm}(f), \text{Dgm}(g)) \leq \|f - g\|_\infty$$

여기서 $d_B$는 **Bottleneck distance**입니다.

Persistence diagram은 확장 평면의 다중집합(multiset)이며, 함수에 대한 약한(mild) 가정 하에서 persistence diagram은 안정적(stable)입니다: 함수의 작은 변화는 다이어그램에서도 작은 변화만을 야기합니다.

#### (e) 알고리즘 복잡도

Filtration에 포함된 simplex의 수가 $m$일 때, 표준 Persistent Homology 알고리즘의 시간 복잡도는:

$$O(m^3)$$

이며, 이후 다양한 최적화를 통해 실질적으로 거의 선형에 가까운 성능을 달성합니다.

### 2.3 성능 향상

이 안정성 결과는 메트릭 공간에서 집합의 호몰로지를 추정하고 기하학적 형태를 비교 및 분류하는 데 적용됩니다.

- **노이즈 강건성**: Persistence가 짧은 특징을 제거함으로써 자연스럽게 노이즈 필터링 역할
- **다중 스케일 분석**: 단일 스케일이 아닌 모든 스케일에서의 위상적 특징을 동시에 파악
- **정량화**: 위상적 특징에 수치적 "수명"을 부여하여 정량적 비교 가능

### 2.4 한계

Persistent homology에는 주목할 만한 제약이 있습니다. 주로 위상 불변량(topological invariants)만을 포착하여, 필트레이션 동안 위상 변화가 없는 경우의 기하학적 형상 진화(geometric shape evolution)를 간과하는 경향이 있습니다. 또한 데이터의 지역적(localized) 위상 정보를 다루기 어렵고, 미분 다양체 위의 데이터나 3차원 공간에 매립된 1차원 곡선(예: 매듭, 링크)을 직접 분석할 수 없습니다.

추가적인 한계:
- **계산 비용**: $O(m^3)$ 복잡도로 인해 대규모 데이터에 대한 적용이 어려움
- **다변량 필터링의 어려움(Multidimensional Persistence)**: 단일 파라미터 필트레이션을 넘어선 다변량 필트레이션에서는 완전한 이산 불변량(complete discrete invariant)이 존재하지 않음
- 위상은 본질적으로 데이터를 단순화하며, 이는 특정 정보의 비가역적 손실(irreversible loss of certain information)을 의미합니다.

---

## 3. 모델의 일반화 성능 향상 가능성

Persistent homology가 모델의 일반화(generalization)에 기여할 수 있는 메커니즘은 여러 측면에서 확인됩니다.

### 3.1 위상적 특징의 노이즈 강건성

안정성 정리 $d_B(\text{Dgm}(f), \text{Dgm}(g)) \leq \|f - g\|_\infty$는 입력 데이터의 작은 섭동(perturbation)에 대해 위상적 특징이 안정적으로 유지됨을 보장합니다. 이는 학습 모델이 데이터의 근본적인 구조적 특징에 집중할 수 있게 하여, 과적합(overfitting)을 방지하는 데 기여합니다.

### 3.2 일반화 갭 예측

신경망의 일반화 갭(generalization gap)을 위상 데이터 분석 방법으로 연구한 결과, 학습 후 뉴런 활성화 상관관계로부터 구성된 가중 그래프의 호몰로지 persistence diagram을 계산하여 네트워크의 일반화 능력과 연결되는 패턴을 포착할 수 있으며, persistence diagram으로부터 도출된 수치적 요약의 조합이 테스트 셋 없이도 일반화 갭을 정확하게 예측하고 부분적으로 설명할 수 있음이 CIFAR10과 SVHN 인식 과제에서 검증되었습니다.

### 3.3 위상 정규화(Topological Regularization)

기하학적·위상적 속성을 올바르게 학습하는 것은 일반화 능력과 생성 품질에 핵심적이며, persistent homology 기반의 위상 메트릭은 GAN 등의 효과적인 평가 지표를 제공합니다.

위상적 손실 함수(topological loss function)를 통한 정규화:

$$\mathcal{L}_{\text{topo}} = \sum_{(b_i, d_i) \in \text{Dgm}} w(b_i, d_i) \cdot \phi(b_i, d_i)$$

여기서 $\phi$는 원하지 않는 위상적 특징에 페널티를 부여하는 함수입니다. 이러한 정규화는 모델의 결정 경계(decision boundary)가 데이터의 본질적 위상 구조를 반영하도록 유도합니다.

### 3.4 반지도학습에서의 일반화 향상

Persistence Homology Distillation(PsHD)은 반지도 지속 학습에서 노이즈에 둔감한 본질적 구조 정보를 보존하며, 이론적·실험적으로 샘플 표현 및 쌍별 유사성 증류 방법에 비해 우수한 안정성을 보입니다. 세 가지 널리 사용되는 데이터셋에서 PsHD가 최첨단 방법 대비 평균 3.9% 향상되었으며, 60% 메모리 버퍼 감소 시에도 1.5% 향상을 달성합니다.

---

## 4. 향후 연구에 미치는 영향 및 고려할 점

### 4.1 학문적 영향

Edelsbrunner & Harer (2008)의 서베이는 TDA(Topological Data Analysis) 분야의 **기초 참고문헌(foundational reference)**으로 자리매김하였습니다.

- 대수적 위상수학에 깊이 뿌리를 둔 persistent homology는 데이터 단순화와 본질적 구조 특성화 사이의 섬세한 균형을 제공하며, 다양한 분야에 성공적으로 적용되어 왔습니다.
- 이 서베이에서 신경망 분석에 가장 널리 사용되는 TDA 도구는 persistent homology로, 이는 정렬된 단체 복합체(simplicial complex) 군을 따라 호몰로지 군의 진화를 연구합니다.

### 4.2 앞으로 연구 시 고려할 점

1. **계산 확장성(Scalability)**: 대규모 포인트 클라우드의 persistent homology 계산은 금지적으로 비용이 높으며, 근사 알고리즘, 분산 계산, 서브샘플링 전략 등이 필요합니다.

2. **기하 정보 보완**: Persistent homology가 필트레이션 동안의 호모토피 기하학적 변화를 포착하지 못하는 한계를 극복하기 위해, 최근 persistent combinatorial Laplacian이 도입되었습니다. 이는 persistent homology보다 더 깊은 데이터 분석을 가능하게 하며, 조화 스펙트럼(harmonic spectra, 즉 영 고유값)은 persistent homology의 위상적 출력을 완전히 복구하고, 비조화 스펙트럼(non-harmonic spectra, 즉 비영 고유값)은 추가적인 기하학적·조합적 정보를 포착합니다.

3. **딥러닝과의 통합**: Topological deep learning(TDL)은 TDA의 원리와 딥러닝 기법을 결합하는 새로운 영역으로, TDA는 데이터 형태에 대한 통찰을 제공하며 변형과 노이즈에 대한 강건성을 보입니다. 이러한 속성은 딥러닝 파이프라인에서 바람직하지만, 일반적으로 비TDA 전략을 통해 얻어집니다.

4. **미분가능성(Differentiability)**: 기계학습에서 PH의 적용은 PH의 미분가능성(differentiability) 속성에 대한 이론적 연구에 의해 가능해졌으며, 이는 다변수(multiparameter) 설정으로도 확장되고 있습니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 핵심 기여 | Edelsbrunner & Harer (2008)과의 관계 |
|------|------|-----------|--------------------------------------|
| **Persistent Combinatorial Laplacians** (Wang et al.) | 2020 | 위상+기하 정보를 동시 포착하는 스펙트럴 접근 | PH의 기하 정보 손실 한계 극복 |
| **PH-based Machine Learning Survey** (Pun et al., *AI Review*) | 2022 | PH 기반 ML의 체계적 리뷰, 벡터화 방법론 비교 | PH의 ML 적용 방법론 체계화 |
| **RePHINE** (Immonen et al., *NeurIPS 2023*) | 2023 | 꼭짓점/간선 레벨 PH를 결합한 그래프 학습 | PH의 표현력 한계를 PH 자체로 극복 |
| **Persistent Homology for High-dim Data** (*NeurIPS 2024*) | 2024 | kNN 그래프 기반 스펙트럼 거리로 고차원 PH 개선 | 차원의 저주 문제 해결 시도 |
| **PsHD** (Fan et al., *NeurIPS 2024*) | 2024 | PH 증류를 통한 반지도 지속학습의 일반화 향상 | PH의 안정성을 지속학습에 활용 |
| **TopInG** (*ICML 2025*) | 2025 | PH를 활용한 GNN의 해석 가능한 그래프 학습 | PH의 해석가능성 활용 |
| **TDA & TDL Beyond PH** (Wei et al., *AI Review*) | 2025 | Persistent Laplacian, Dirac 연산자 등 PH를 넘어선 종합 리뷰 | PH의 한계를 넘어선 차세대 도구 제시 |

### 주요 최신 동향과 비교 분석

**(1) 기하 정보 보완 — Persistent Laplacians**

기하학적 형상 진화를 포착하기 위한 노력으로, 포인트 클라우드 데이터와 그래프에 대한 persistent combinatorial Laplacian이 Wang et al.(2020)에 의해 도입되었고, 최근 몇 년간 광범위하게 연구되었습니다.

Edelsbrunner & Harer가 정리한 기본 PH 프레임워크에서의 핵심 한계인 "위상 변화가 없는 구간에서의 기하학적 정보 손실"을 직접적으로 해결합니다.

**(2) 딥러닝 통합 — Topological Deep Learning**

Topological deep learning(TDL) 또는 topological machine learning과 결합된 persistent homology는 과학, 공학, 의학, 산업 분야에서 광범위한 응용에 놀라운 성공을 거두었습니다.

특히 persistent homology 기반 TDA 접근법이 딥러닝과 통합되어, 전통적 질서 변수(order parameters)보다 로컬 및 글로벌 구조적 특징을 동시에 포착하여 막 조직을 강건하게 예측하는 데 성공했습니다.

**(3) 그래프 신경망 향상**

RePHINE은 그래프 위의 위상적 특징을 학습하기 위해 제안되었으며, 꼭짓점 레벨과 간선 레벨 PH를 효율적으로 결합하여 두 접근법 모두보다 증명 가능하게 더 강력한 체계를 달성합니다.

**(4) 고차원 데이터 문제 — 차원의 저주 극복**

고차원 데이터에 대한 persistent homology의 한계를 극복하기 위해 kNN 그래프 위의 spectral distance를 활용하는 방법이 NeurIPS 2024에서 제안되었습니다.

**(5) 실용적 응용 — 바이러스 진화 예측**

TDL이 다른 경쟁 방법들에 비해 일관된 우위를 보인 가장 설득력 있는 사례로는 D3R Grand Challenges에서의 승리, SARS-CoV-2 진화 메커니즘의 발견, 그리고 SARS-CoV-2 변이 BA.2 및 BA.4/BA.5를 약 2개월 앞서 성공적으로 예측한 것이 있습니다.

---

## 참고문헌 및 출처

1. **Edelsbrunner, H. & Harer, J.** (2008). "Persistent homology — a survey." *Contemporary Mathematics*, 453, pp. 257–282. American Mathematical Society. [DOI:10.1090/conm/453](https://doi.org/10.1090/conm/453)

2. **Cohen-Steiner, D., Edelsbrunner, H. & Harer, J.** (2007). "Stability of persistence diagrams." *Discrete & Computational Geometry*, 37(1), 103–120.

3. **Edelsbrunner, H., Letscher, D. & Zomorodian, A.** (2002). "Topological persistence and simplification." *Discrete & Computational Geometry*, 28(4), 511–533.

4. **Pun, C.S., Lee, S.X. & Xia, K.** (2022). "Persistent-homology-based machine learning: a survey and a comparative study." *Artificial Intelligence Review*, Springer.

5. **Wei, G.-W. et al.** (2025). "Topological data analysis and topological deep learning beyond persistent homology — a review." *Artificial Intelligence Review*, Springer. [링크](https://link.springer.com/article/10.1007/s10462-025-11462-w)

6. **Immonen, J., Souza, A. & Garg, V.** (2024). "Going Beyond Persistent Homology Using Persistent Homology." *NeurIPS 2023 Proceedings*.

7. **Fan, Y. et al.** (2024). "Persistence Homology Distillation for Semi-supervised Continual Learning." *NeurIPS 2024*. [GitHub](https://github.com/fanyan0411/PsHD)

8. **NeurIPS 2024.** "Persistent Homology for High-dimensional Data Based on Spectral Methods." [PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/4a32a646254d2e37fc74a38d65796552-Paper-Conference.pdf)

9. **Topological deep learning: a review of an emerging paradigm.** (2024). *Artificial Intelligence Review*, Springer. [링크](https://link.springer.com/article/10.1007/s10462-024-10710-9)

10. **Neurocomputing** (2024). "Predicting the generalization gap in neural networks using topological data analysis." [DOI](https://dl.acm.org/doi/10.1016/j.neucom.2024.127787)

11. **ICML 2025.** "TopInG: Topologically Interpretable Graph Learning via Persistent Rationale Filtration."

12. **Edelsbrunner, H. & Harer, J.** (2010). *Computational Topology: An Introduction.* American Mathematical Society.

13. **TDA for Neural Network Analysis: A Comprehensive Survey.** arXiv:2312.05840. [링크](https://arxiv.org/html/2312.05840v2)

---

> **참고**: 본 분석은 Edelsbrunner & Harer (2008)의 공개된 초록, 관련 인용 문헌, 그리고 2020년 이후 발표된 후속 연구들에 기반하여 작성되었습니다. 본 논문은 전통적인 ML "모델"을 제안하는 논문이 아니라, 수학적 이론과 알고리즘을 정리한 **서베이 논문**이므로, "모델 구조" 및 "벤치마크 성능"의 논의는 PH 프레임워크 자체의 수학적 구조와 그 응용 잠재력에 초점을 맞추었습니다.
