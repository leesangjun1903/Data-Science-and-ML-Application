# Extracting Insights from the Shape of Complex Data Using Topology

> **논문 정보**: Lum, P.Y., Singh, G., Lehman, A., Ishkanov, T., Vejdemo-Johansson, M., Alagappan, M., Carlsson, J., & Carlsson, G. (2013). *Scientific Reports*, **3**, 1236. DOI: 10.1038/srep01236

---

## 1. 핵심 주장 및 주요 기여 (요약)

이 논문은 위상적(topological) 방법론을 고차원 복잡 데이터 집합에 적용하여 데이터의 형상(shape), 즉 패턴을 추출하고 통찰을 얻는 방법을 제시한다. 주성분 분석(PCA)과 클러스터 분석 등 기존 표준 방법론의 장점을 결합하여, 복잡한 데이터 집합의 기하학적 표현(geometric representation)을 제공한다. 이 하이브리드 방법을 통해, 기존 방법론이 발견하지 못하는 데이터 하위 그룹(subgroup)을 종종 찾아낸다.

핵심 알고리즘은 Stanford의 Gunnar Carlsson과 Gurjeet Singh가 개발한 **Mapper** 알고리즘이다. 이 핵심 알고리즘 "Mapper"는 Stanford 계산 토폴로지 그룹에서 개발되어 Ayasdi라는 기업의 제품으로 상용화되었다.

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

대규모 복잡 데이터 집합을 유용한 지식으로 변환하기 위한 방법론이 필요하며, 현재 사용되는 많은 방법론은 연구자가 생성한 가설을 검증(또는 반증)하는 메커니즘으로 작동하기 때문에, 가능한 가설의 수가 매우 많은 복잡한 데이터셋에서는 유용한 가설을 생성하는 작업이 매우 어렵다. 이 논문은 사전에 질의나 가설을 수립하지 않고도 데이터를 탐색할 수 있는 방법을 논의한다.

대부분의 빅데이터 마이닝 접근법이 쌍별 관계(pairwise relationships)를 기본 구성 요소로 집중하는 반면, 이 논문은 데이터의 "형상(shape)"을 이해하는 것의 중요성을 보여준다. 데이터의 형상을 이해하는 것은 우리를 위상수학(topology)이라는 수학 분야로 이끈다.

### 2.2 제안하는 방법: Mapper 알고리즘

위상수학(topology)은 형상(shape)을 연구하는 수학 분야로, 18세기 Euler의 연구에서 기원하며, 최근 15년간 다양한 응용 문제에 위상적 방법을 적용하려는 노력이 있었다. 이를 위상적 데이터 분석(Topological Data Analysis, TDA)이라 부르며, 그 근본적 아이디어는 위상적 방법이 데이터 내 패턴 또는 형상 인식에 대한 기하학적 접근 역할을 한다는 것이다.

#### Mapper 알고리즘의 수학적 구조

Mapper 알고리즘은 Reeb 그래프의 이산적 근사(discrete approximation)로 이해할 수 있다. 주요 단계는 다음과 같다:

**Step 1: 필터 함수(Filter function) 적용**

데이터 집합 $X$가 주어졌을 때, 연속 함수(필터) $f: X \to \mathbb{R}^d$ 를 적용한다. 일반적으로 $d=1$ 또는 $d=2$를 사용한다. 필터 함수의 선택에는 다음이 포함된다:

- 밀도 추정기 (Kernel Density Estimator)
- 이심률 (Eccentricity): $\text{ecc}(x) = \max_{y \in X} d(x, y)$ 또는 $\text{ecc}\_p(x) = \left(\sum_{y \in X} d(x, y)^p\right)^{1/p}$
- 특이값 분해(SVD) 기반 필터

**Step 2: 열린 덮개(Open cover) 구성**

필터 함수의 값역(range) $f(X) \subseteq \mathbb{R}$을 겹치는 구간(overlapping intervals) $\{U_i\}_{i \in I}$으로 덮는다. 여기서 두 가지 파라미터가 중요하다:

- **해상도(Resolution)** $r$: 구간의 수
- **이득(Gain)** $g$: 겹침(overlap)의 정도 (퍼센트)

$$f(X) \subseteq \bigcup_{i \in I} U_i, \quad U_i \cap U_{i+1} \neq \emptyset$$

**Step 3: 클러스터링(Clustering)**

각 역상(preimage) $f^{-1}(U_i) \subseteq X$에 대해 클러스터링 알고리즘을 적용하여 노드(node)를 생성한다:

$$f^{-1}(U_i) = C_{i,1} \sqcup C_{i,2} \sqcup \cdots \sqcup C_{i,k_i}$$

**Step 4: 신경 복합체(Nerve complex) 구성**

서로 다른 클러스터 $C_{i,j}$와 $C_{i',j'}$가 공통 데이터 포인트를 공유하면 에지(edge)로 연결하여 단순 복합체(simplicial complex)를 구성한다:

$$\text{edge}(C_{i,j}, C_{i',j'}) \iff C_{i,j} \cap C_{i',j'} \neq \emptyset$$

이것은 Nerve Lemma에 기반한다:

> **신경 정리 (Nerve Theorem)**: 위상 공간 $X$의 열린 덮개 $\mathcal{U} = \{U_\alpha\}_{\alpha \in A}$에서 모든 유한 교집합이 수축 가능(contractible)하면, 신경(Nerve) $N(\mathcal{U})$는 $X$와 동형(homotopy equivalent)이다.

이러한 네트워크에서 나타나는 전형적인 형상은 "루프(loops)"(연속적 원형 세그먼트)와 "플레어(flares)"(긴 선형 세그먼트)이다.

### 2.3 응용 사례 및 성능 향상

유방암 유전자 발현 데이터, 미국 하원 투표 데이터, NBA 선수 성능 데이터의 세 가지 매우 다른 종류의 데이터에 이 방법을 적용하여, 기존 방법보다 더 정교한 데이터 계층화(stratification)를 발견하였다.

#### (a) 유방암 데이터

첫 번째 응용은 유방암 환자 하위 집단의 식별이다. 위상적 맵이 표준 클러스터링 방법보다 환자를 더 세밀하게 계층화할 수 있음을 보여주었고, 표적 치료에 중요할 수 있는 흥미로운 환자 하위 그룹을 식별하였다.

#### (b) 정치 투표 데이터

네트워크는 하원 의원의 투표 행동으로 구성되며, "찬성"은 1, "기권"은 0, "반대"는 -1로 코딩된다. 각 노드는 의원 집합을 포함하고, 2010년의 높은 분열(fragmentation)을 감지할 수 있었다.

#### (c) NBA 데이터

선수 성능 데이터에 Mapper를 적용하여 기존 포지션 분류보다 더 세밀한 플레이 스타일 계층 구조를 발견하였다.

### 2.4 Persistent Homology의 수학적 기초

Mapper와 보완적으로 사용되는 Persistent Homology의 핵심 개념:

**Vietoris-Rips 복합체**: 매개변수 $\epsilon > 0$에 대해

$$\text{VR}(X, \epsilon) = \{\sigma \subseteq X : d(x_i, x_j) \leq \epsilon, \; \forall x_i, x_j \in \sigma\}$$

**Persistence Diagram**: 필트레이션(filtration) $\epsilon_1 \leq \epsilon_2 \leq \cdots$에 걸쳐 각 위상적 특징(연결 성분, 루프, 공동 등)의 탄생-소멸 쌍 $(b_i, d_i)$을 기록하며, 수명(persistence)이 긴 특징이 데이터의 진정한 구조를 반영한다:

$$\text{pers}(i) = d_i - b_i$$

다양한 근접 임계값에 걸쳐 구성된 단순 복합체의 족(family), 즉 필트레이션(filtration)을 통해, TDA는 persistence diagram에서 위상적 특징의 탄생과 소멸을 기록하며, 이는 데이터의 형상을 나타내는 구간의 다중집합(multiset of intervals)이다.

**안정성 정리 (Stability Theorem)**:

$$d_B\big(\text{Dgm}(f), \text{Dgm}(g)\big) \leq \|f - g\|_\infty$$

여기서 $d_B$는 bottleneck distance이며, 이 정리는 입력 데이터의 작은 섭동이 persistence diagram의 작은 변화만을 유발함을 보장한다.

### 2.5 한계

1. **파라미터 민감성**: 해상도(Resolution), 이득(Gain), 필터 함수, 거리 메트릭, 클러스터링 알고리즘의 선택에 따라 결과가 달라질 수 있다.
2. **계산 복잡도**: 고차원 대규모 데이터에서 persistent homology 계산은 $O(n^{k+1})$ 수준의 단순 복합체를 생성할 수 있다.
3. persistent homology 또는 TDA는 본질적으로 데이터를 단순화한다. 따라서 단순한 데이터에는 적합하지 않을 수 있으며, TDA 단순화가 필수 정보의 손실로 이어질 수 있다.
4. **해석의 주관성**: 네트워크 시각화의 해석이 연구자에 따라 달라질 수 있다.
5. **이론적 보장의 부족**: 초기 Mapper 논문은 통계적 수렴성이나 일반화 경계에 대한 형식적 증명이 제한적이었다.

---

## 3. 모델의 일반화 성능 향상 가능성

TDA의 핵심 강점 중 하나는 **좌표 자유(coordinate-free)**, **연속 변형에 대한 불변성(invariance under continuous deformation)**, **압축된 표현** 능력이다. 위상 공간과 그 속성의 중요한 측면은 좌표 자유이고, 연속 변형에 불변이며, 압축된 형태로 표현될 수 있다는 것이다.

### 3.1 위상적 특징의 노이즈 강건성

TDA는 데이터 형상에 대한 통찰을 제공하며, 이러한 방법으로 얻은 요약은 다차원 데이터의 원칙적이고 전역적인 기술(description)이면서 변형과 노이즈에 대한 강건한(robust) 안정적 속성을 보인다.

이는 앞서 언급한 **안정성 정리**에 의해 수학적으로 보장된다:

$$d_W^p\big(\text{Dgm}(f), \text{Dgm}(g)\big) \leq C \cdot \|f - g\|_\infty$$

### 3.2 TDA 기반 정규화(Regularization)를 통한 일반화

정규화는 과적합(overfitting)을 방지하고 딥러닝 모델의 일반화 능력을 개선하는 데 중요한 역할을 하며, 향후 연구에서는 TDA 기반 정규화 기법이 딥러닝 프레임워크에 어떻게 통합될 수 있는지 탐구할 것이다.

Chen et al. (2019)은 persistent homology에 기반한 정규화기(regularizer)가 더 나은 일반화(generalization)와 더 높은 예측 성능을 이끌어냄을 보여주었다.

**위상적 정규화의 수학적 형식화** (Hofer et al., 2019; Chen et al., 2019):

분류기의 결정 경계(decision boundary)의 위상적 복잡도를 줄이기 위해 다음과 같은 정규화 항을 손실 함수에 추가할 수 있다:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda \cdot \mathcal{R}_{\text{topo}}$$

여기서:

$$\mathcal{R}_{\text{topo}} = \sum_{(b_i, d_i) \in \text{Dgm}} w(b_i, d_i) \cdot (d_i - b_i)^p$$

$w(b_i, d_i)$는 가중 함수이며, 불필요한 위상적 특징(noise에 의한 짧은 수명의 특징)을 제거하도록 유도한다.

소규모 표본 환경에서 과잉 매개변수화된 심층 신경망의 일반화는 어려우며, 더 나은 표현이 일반적으로 일반화에 유리하다. Hofer et al.은 TDA의 관점에서 심층 신경망의 내부 표현을 제어하는 새로운 방법을 제시하였으며, 특징 추출기에 의해 유도된 push-forward 확률 측도를 연구하고, persistent homology 관점에서 "분리(separation)"의 개념을 처음으로 공식화하였다. 이 속성을 강제하는 것이 더 나은 일반화로 이어짐을 이론적으로 증명하였으며, 위상 정보를 추출하는 새로운 가중 함수와 위상 인식 방식으로 표현 학습을 안내하는 세 가지 항목을 포함하는 새로운 정규화기를 도입하였다.

### 3.3 TDA의 다규모(Multiscale) 특성에 의한 일반화

TDA는 복잡한 데이터셋을 그들의 본질적(위상적) 특징으로 압축하여, 널리 사용되는 계산 비용이 높은 범용 인공 신경망과 비교하여 더 단순한 머신러닝 모델의 학습을 가능하게 한다.

이는 곧 **낮은 모델 복잡도 → 더 나은 일반화**라는 VC 이론적 관점에서의 일반화 향상과 연결된다.

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 연구에 미치는 영향

이 논문의 혁신은 시간, 이질적 데이터 집합, 또는 스케일 변화에 걸친 다수의 네트워크 간 대응관계(correspondence)가 매우 중요하며, 새로운 통찰을 이끌어낼 수 있음을 보여주었다는 것이다.

이 논문 이후 TDA는 다음 분야로 확장되었다:

1. **의료 영상**: TDA는 수학적 위상수학에 기반하여, persistent homology와 위상적 특징을 통해 복잡하고 고차원적인 방사선학 데이터에 새로운 통찰을 제공하며, 종양 특성화, 심혈관 영상, COVID-19 탐지 등 다양한 의료 영상 분야에서 전통적 방법 대비 15–20%의 성능 향상을 보여주었다.

2. **분자 과학**: TDA는 복잡한 분자 데이터에서 강건하고, 다규모적이며, 해석 가능한 특징을 추출하는 강력한 프레임워크로 부상하였다.

3. **NLP**: 연구자들은 Mapper를 사용하여 BERT 트랜스포머 모델의 은닉 표현에서 다의어(polysemous words)를 시각화하는 등 감정 및 의미 분석과 구조 시각화를 결합하는 연구를 수행하였다.

4. **GNN 해석 가능성**: TopInG는 persistent homology를 활용하여 영속적 이유 하위 그래프를 식별하는 새로운 위상적 프레임워크이며, 이유 필트레이션 학습 접근법을 통해 이유 하위 그래프의 자기 회귀적 생성 과정을 모델링한다.

### 4.2 앞으로 연구 시 고려할 점

1. **계산 확장성 (Scalability)**:
   대규모 포인트 클라우드에 대한 persistent homology 계산은 비용이 매우 높으며, 최악의 경우 시간 복잡도가 $O(n^{k+1})$ 수준의 단순 복합체를 생성한다.

2. **Persistent Homology의 한계 극복**:
   TDA는 응용 수학 및 데이터 과학에서 급속히 진화하는 분야로, 그 주요 도구는 대수적 위상수학에 뿌리를 둔 persistent homology이다. persistent homology는 다양한 응용 분야에서 엄청난 성공을 거두었지만, 높은 수준의 추상화, 비위상적 변화에 대한 무감각, 포인트 클라우드 데이터에 대한 제한 등 많은 한계를 가지고 있다. 이를 넘어 persistent topological Laplacians와 Dirac 연산자가 위상적 불변량과 호모토피 진화를 모두 포착하는 스펙트럼 표현을 제공한다.

3. **고차원 데이터 구조의 활용**:
   고차원 데이터가 부족하며, TDL의 발전을 위한 주요 이정표 중 하나는 공개된 고차원 데이터셋의 풍부한 생성이다.

4. **TDA-정규화의 심화**:
   정규화는 과적합 방지와 일반화 향상에 핵심적 역할을 하며, 향후 연구에서 TDA 기반 정규화 기법을 딥러닝 프레임워크에 통합하는 방향이 탐구될 것이다. 이는 모델이 의미 있는 위상적 특징을 포착하도록 유도하는 위상적 페널티 또는 제약조건을 포함할 수 있다.

5. **수학적 접근성**: TDA를 NLP 작업에 적용하는 주요 과제 중 하나는 수학적 기초와 관련된 가파른 학습 곡선이며, 이러한 고급 개념을 개발하고 이해하는 이론가들이 항상 계산 과학자와 협력하여 이를 실행 가능한 코드로 변환하는 것은 아니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 방향 | 대표 연구 | 핵심 기여 | Lum et al. (2013)과의 비교 |
|---|---|---|---|
| **Persistent Topological Laplacians** | Wang et al. (2020)에 의해 포인트 클라우드 데이터와 그래프에 대한 persistent combinatorial Laplacians이 도입되었다. | 위상적 불변량 + 기하학적 형상 변화를 동시에 포착하는 스펙트럼 표현 제공 | Mapper/PH가 놓치는 비위상적 형상 정보까지 포착 |
| **Topological Deep Learning (TDL)** | 초기에는 TDL이 persistent homology로 생성된 특징을 DNN 입력 파이프라인에 통합하는 것을 지칭했으나, 이제는 딥러닝에서 위상적 개념을 활용하는 아이디어와 방법의 집합체를 지칭한다. | 단순 복합체, 셀 복합체, 하이퍼그래프 등에서의 메시지 패싱 | 원래 Mapper는 탐색적 도구였으나, TDL은 학습 가능한 end-to-end 파이프라인으로 발전 |
| **위상적 정규화기** | Chen et al. (2019); Hofer et al. (2023)은 TDA 관점에서 심층 신경망의 내부 표현을 제어하며, persistent homology 관점에서 "분리" 개념을 공식화하고, 이를 강제하면 일반화가 향상됨을 증명하였다. | 위상적 손실항으로 결정 경계의 복잡도 제어 | Lum et al.의 탐색적 분석을 넘어 학습 목적 함수에 직접 통합 |
| **확장 가능한 위상적 정규화** | Gómez & Mémoli (2025)의 PPM-Reg는 계산 비용이 높은 Wasserstein 메트릭의 대안으로, 더 깊은 네트워크 학습을 위한 안정적 그래디언트를 생성하며 대규모 ML 작업에 위상적 특징을 안정적으로 통합할 수 있게 한다. | GAN 학습에 위상적 매칭 적용 | 계산 확장성 문제 직접 해결 |
| **의료 영상 TDA** | 의료 영상에서 TDA는 종양 특성화, 심혈관 영상, COVID-19 탐지에 적용되어 전통적 방법 대비 15–20% 성능 향상을 달성하며, TDA와 AI의 시너지가 진단 정확도 향상을 위한 유망한 기회를 제시한다. | 임상 적용 가능성 직접 검증 | Lum et al.의 유방암 분석을 훨씬 넘어선 범위 |
| **NLP에서의 TDA** | 딥페이크 텍스트 탐지(Uchendu et al., 2024)와 LLM 환각 탐지(Bazarova et al., 2025) 등 분류 작업에 TDA가 적용되고 있다. | 언어 모델의 구조적 이해에 위상적 도구 활용 | 완전히 새로운 응용 도메인 |
| **뇌 기능 연결성** | persistent homology를 사용하여 건강한 노화와 자폐 스펙트럼 장애에서의 안정 상태 기능적 연결성 변화를 연구하며, 전역, 중규모, 국소 세 가지 스케일에서 기능적 연결성 변화를 분석하고 node persistence라는 새로운 PH 기반 측도를 도입하였다. | 다규모 뇌 네트워크 분석 | Mapper의 네트워크 분석을 뇌과학으로 확장 |

---

## 참고문헌 및 출처

1. Lum, P.Y. et al. (2013). "Extracting insights from the shape of complex data using topology." *Scientific Reports*, 3, 1236. — [Nature](https://www.nature.com/articles/srep01236)
2. Carlsson, G. (2009). "Topology and data." *Bull. Amer. Math. Soc.*, 46, 255–308.
3. Zia, A. et al. (2024). "Topological deep learning: a review of an emerging paradigm." *Artificial Intelligence Review*. — [Springer](https://link.springer.com/article/10.1007/s10462-024-10710-9)
4. Papamarkou, T. et al. (2024). "Position: Topological Deep Learning is the New Frontier for Relational Learning." *ICML 2024*. — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11973457/)
5. Wei, G.-W. et al. (2025). "Topological data analysis and topological deep learning beyond persistent homology: a review." *Artificial Intelligence Review*. — [Springer](https://link.springer.com/article/10.1007/s10462-025-11462-w)
6. Hofer et al. (2023). "Topological Regularization for Representation Learning via Persistent Homology." — [ResearchGate](https://www.researchgate.net/publication/368583212)
7. TDA and ML review (2023). "Topological data analysis and machine learning." *Advances in Physics: X*. — [Taylor & Francis](https://www.tandfonline.com/doi/full/10.1080/23746149.2023.2202331)
8. Comprehensive Mapper Review (2025). "A Comprehensive Review of the Mapper Algorithm." — [arXiv:2504.09042](https://arxiv.org/pdf/2504.09042)
9. TDA in Radiology (2025). "Unraveling the Invisible: TDA as the New Frontier in Radiology." — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11768448/)
10. TDA in Molecular Sciences (2025). "A Review of TDA and TDL in Molecular Sciences." — [ACS JCIM](https://pubs.acs.org/doi/10.1021/acs.jcim.5c02266)
11. TDA in NLP (2024). "Unveiling Topological Structures from Language." — [arXiv:2411.10298](https://arxiv.org/html/2411.10298v3)
12. Towards Scalable Topological Regularizers (2025). — [arXiv:2501.14641](https://arxiv.org/html/2501.14641)
13. Chen, C. et al. (2019). "A topological regularizer for classifiers via persistent homology." *AISTATS*. PMLR, pp 2573–2582.
14. TopInG (2025). "Topologically Interpretable Graph Learning via Persistent Rationale Filtration." *ICML 2025*. — [ICML](https://icml.cc/virtual/2025/poster/43748)
