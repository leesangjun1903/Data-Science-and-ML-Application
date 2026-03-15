# giotto-tda: A Topological Data Analysis Toolkit for Machine Learning and Data Exploration

**논문 정보**: Tauzin, G., Lupo, U., Tunstall, L., Burella Pérez, J., Caorsi, M., Medina-Mardones, A. M., Dassatti, A., & Hess, K. (2021). *Journal of Machine Learning Research*, 22(39), 1–6.

---

## 1. 핵심 주장과 주요 기여 요약

giotto-tda는 고성능 위상적 데이터 분석(TDA)을 머신러닝과 통합하는 Python 라이브러리로, scikit-learn 호환 API와 최첨단 C++ 구현을 기반으로 한다. 다양한 데이터 유형을 처리할 수 있는 광범위한 전처리 기법과 데이터 탐색 및 해석 가능성에 초점을 맞춘 직관적인 플로팅 API를 제공한다.

**핵심 기여**:
- 시계열, 이미지, 그래프 및 그 고차원 유사체인 단체 복합체(simplicial complexes)에 TDA를 적용할 수 있어, 가장 포괄적인 Python 위상 머신러닝 및 데이터 탐색 라이브러리이다.
- L2F SA, EPFL의 위상수학 및 신경과학 연구소, HEIG-VD의 재구성 가능 및 임베디드 디지털 시스템 연구소(REDS)의 공동 협력 결과물이다.

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

위상적 데이터 분석(TDA)은 데이터의 근본적인 "형태(shape)"를 파악하는 강력한 도구이나, 기존에는 다음과 같은 문제가 존재했다:

1. **TDA와 ML 파이프라인 간의 단절**: 기존 TDA 라이브러리(GUDHI, Scikit-TDA, Dionysus 등)는 scikit-learn과의 완전한 통합이 부족
2. **다양한 데이터 유형에 대한 일관된 인터페이스 부재**: 포인트 클라우드, 시계열, 이미지, 그래프 등에 대한 통합적 TDA 적용 방법이 부재
3. **하이퍼파라미터 튜닝의 어려움**: 위상적 특징은 persistence diagram에서 곡선, 이미지, 커널로 표현되며, 각 방법은 문제에 맞게 튜닝해야 하는 하이퍼파라미터 집합을 포함한다.

### 2.2 제안하는 방법 및 수식

#### (A) Persistent Homology (지속적 호몰로지)

지속적 호몰로지(Persistent Homology)는 TDA의 주요 도구로, persistence diagram이라 불리는 형태로 다중 스케일 관계 정보를 추출 및 요약하며, 계층적 클러스터링과 유사하지만 고차 연결성도 고려한다.

**Vietoris-Rips Complex** 구성: 포인트 클라우드 $X = \{x_1, x_2, \ldots, x_n\}$에 대해 스케일 파라미터 $\epsilon$에서의 Vietoris-Rips 복합체는 다음과 같이 정의된다:

$$VR_\epsilon(X) = \{\sigma \subseteq X \mid d(x_i, x_j) \leq \epsilon, \quad \forall x_i, x_j \in \sigma\}$$

**Filtration (여과)**: 스케일 파라미터 $\epsilon$를 점진적으로 증가시키면서 복합체의 연쇄를 구성한다:

$$\emptyset = K_0 \subseteq K_1 \subseteq K_2 \subseteq \cdots \subseteq K_m$$

각 단계에서 나타나는(birth) 호몰로지 특징 $b_i$와 사라지는(death) 시점 $d_i$를 기록하여 **Persistence Diagram**을 생성한다:

$$\text{Dgm} = \{(b_i, d_i) \mid b_i < d_i, \quad i = 1, 2, \ldots, k\}$$

각 특징의 **persistence**(지속성)는 다음으로 정의된다:

$$\text{pers}_i = d_i - b_i$$

#### (B) Persistence Diagram의 벡터화 표현

giotto-tda는 persistence diagram을 ML 모델에 입력 가능한 형태로 변환하는 다양한 방법을 제공한다:

**Persistence Entropy**:

$$E = -\sum_{i=1}^{k} p_i \log(p_i), \quad \text{where} \quad p_i = \frac{\text{pers}_i}{\sum_{j=1}^{k} \text{pers}_j}$$

**Persistence Landscape**: persistence diagram의 $k$차 landscape 함수:

$$\lambda_k(t) = k\text{-th largest value of } \{\Lambda_{(b_i, d_i)}(t)\}_{i=1}^{n}$$

여기서:

$$\Lambda_{(b,d)}(t) = \max\left(0, \min\left(t - b, d - t\right)\right)$$

**Persistence Image**: persistence diagram을 이미지로 변환:

$$\rho(x,y) = \sum_{i=1}^{k} w(b_i, \text{pers}_i) \cdot \phi_{(b_i, \text{pers}_i)}(x,y)$$

여기서 $\phi$는 Gaussian 커널이고 $w$는 가중 함수이다.

**Heat Kernel (Persistence Scale Space Kernel)**:

$$k_\sigma(\text{Dgm}_1, \text{Dgm}_2) = \frac{1}{8\pi\sigma} \sum_{p \in \text{Dgm}_1}\sum_{q \in \text{Dgm}_2} \left[ e^{-\frac{\|p-q\|^2}{8\sigma}} - e^{-\frac{\|p-\bar{q}\|^2}{8\sigma}}\right]$$

여기서 $\bar{q} = (q_y, q_x)$는 대각선에 대한 반사이다.

#### (C) 시계열 처리를 위한 Takens 임베딩

TDA 기법을 시계열에 적용하려면, 입력 데이터를 먼저 고차원 공간에 임베딩해야 한다.

Takens의 지연 임베딩 정리에 기반하여, 시계열 $\{x_t\}$를 다음과 같이 변환한다:

$$\mathbf{v}_t = (x_t, x_{t+\tau}, x_{t+2\tau}, \ldots, x_{t+(d-1)\tau}) \in \mathbb{R}^d$$

여기서 $d$는 임베딩 차원, $\tau$는 시간 지연이다.

#### (D) Mapper 알고리즘

Mapper는 필터 함수의 적용과 부분 클러스터링을 결합하여 고차원 데이터의 단순하고 위상적으로 의미 있는 비가중 그래프 설명을 생성하는 표현 기법이다.

### 2.3 모델 구조 (라이브러리 아키텍처)

giotto-tda의 파이프라인은 3단계로 구성된다:

giotto-tda는 scikit-learn 호환 컴포넌트를 제공하여 사용자가 (a) 다양한 입력 데이터 유형을 persistent homology 계산에 적합한 형태로 변환, (b) 대규모 알고리즘 선택에 따라 persistence diagram 계산, (c) persistence diagram에서 풍부한 특징 집합을 추출할 수 있게 한다.

```
Raw Data → [전처리/임베딩] → [Persistent Homology 계산] → [특징 추출/벡터화] → [ML 모델]
```

그 결과, 입력 원시 데이터 컬렉션의 각 샘플에서 신중하게 설계된 위상적 특징을 생성하기 위한 엔드-투-엔드 Pipeline 객체를 구축하는 프레임워크이다.

**주요 모듈**:

| 모듈 | 기능 |
|------|------|
| `gtda.time_series` | Takens 임베딩, 슬라이딩 윈도우 |
| `gtda.homology` | VietorisRipsPersistence, CubicalPersistence, FlagserPersistence |
| `gtda.diagrams` | PersistenceEntropy, PersistenceLandscape, PersistenceImage, 커널 |
| `gtda.mapper` | Mapper 알고리즘 시각화 |
| `gtda.graphs` | 그래프 전처리 |

### 2.4 성능 향상

Vietoris-Rips 바코드 계산에서, giotto-tda는 edge collapse 알고리즘을 ripser와 결합하여 기존 최첨단 런타임을 개선한다.

giotto-tda는 대규모 알고리즘 선택을 제공하고, scikit-learn API와 긴밀하게 통합하여 하이퍼파라미터 검색, 교차 검증 및 특징 선택을 통해 관련된 많은 하이퍼파라미터의 간단한 데이터 기반 튜닝을 가능하게 한다.

방향성 persistent homology의 존재는 많은 실세계 상호작용의 비대칭적 특성을 강조하는 관점이며, giotto-tda는 다양한 입력 데이터 유형에 이를 활용하기 위한 전처리 변환기를 제공한다.

### 2.5 한계

1. **계산 복잡도**: Vietoris-Rips 복합체의 계산은 $O(2^n)$의 최악 복잡도를 가지며, 대규모 데이터셋에서는 여전히 병목
2. **하이퍼파라미터 의존성**: 각 방법은 문제에 맞게 튜닝해야 하는 하이퍼파라미터 집합을 포함한다.
3. **이론적 보장 부재**: 위상적 특징이 항상 ML 성능 향상을 보장하지는 않음
4. persistent homology는 고수준 추상화, 비위상적 변화에 대한 불감성, 포인트 클라우드 데이터에 대한 제한 등 여러 한계를 가진다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 위상적 특징의 일반화 이점

위상적 특징은 데이터의 **구조적·형태학적 불변량**을 캡처하므로, 노이즈나 작은 변형에 대해 안정적(stable)이다. 이는 persistence diagram의 안정성 정리에 의해 보장된다:

$$d_B(\text{Dgm}(f), \text{Dgm}(g)) \leq \|f - g\|_\infty$$

여기서 $d_B$는 bottleneck 거리이며, 이 부등식은 입력의 작은 변동이 위상적 특징의 작은 변동만을 야기함을 보장한다.

### 3.2 TDA를 통한 일반화 갭 추정

Birdal et al. (2021, NeurIPS)은 TDA의 관점에서 이 문제를 고찰하며, 학습 이론과 TDA 사이의 새로운 연결을 만들어, 일반화 오류가 'persistent homology dimension' (PHD)이라는 개념으로 동등하게 바운딩될 수 있음을 보였다.

PHD는 다음과 같이 정의된다:

```math
\text{PHD} = \inf\left\{\alpha \geq 0 : \sum_{i} \text{pers}_i^\alpha < \infty\right\}
```

최근 수립된 이론적 결과와 TDA 도구를 활용하여 현대적 심층 신경망 규모에서 PHD를 추정하는 효율적 알고리즘을 개발했으며, 실험 결과 제안된 접근법이 다양한 설정에서 일반화 오류를 예측하는 데 효과적임을 보였다.

### 3.3 위상적 특징 엔지니어링에 의한 일반화 향상

Persistent Homology Feature Neural Networks (PHFNNs)는 사용된 특징 수에 관계없이 일반 FNN보다 일반화 성능에서 일관되게 우수했으며, 이는 persistent homology와 persistent landscape를 통한 특징 엔지니어링 기법이 시계열 예측에서 인공 신경망 모델의 예측 능력을 향상시키는 데 효과적임을 뒷받침한다.

### 3.4 giotto-tda의 일반화 성능 향상 메커니즘

giotto-tda가 일반화에 기여하는 핵심 경로:

1. **교차 검증과의 통합**: scikit-learn의 `GridSearchCV`, `RandomizedSearchCV`와 완전 호환되어 위상적 특징의 하이퍼파라미터까지 체계적으로 최적화
2. **다중 스케일 정보 캡처**: Filtration을 통해 단일 스케일이 아닌 전체 스케일에 걸친 구조 정보를 제공
3. **보완적 특징**: 기하학적 특징(좌표, 거리 등)이 놓치는 위상적 불변량을 추가하여 특징 공간 확장

위상적 요약은 기하학적 특징을 보완하여 객체 인식이나 유사성 비교와 같은 다운스트림 태스크에서 일반화와 견고성을 향상시킨다.

### 3.5 Topological Regularization

Chen et al. (2019)이 제안한 위상적 정규화 방법은 다음 형태의 손실함수를 사용한다:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda \cdot \mathcal{L}_{\text{topo}}$$

여기서:

$$\mathcal{L}\_{\text{topo}} = \sum_{(b_i, d_i) \in \text{Dgm}} (d_i - b_i)^p$$

이 정규화 항은 결정 경계의 위상적 복잡성을 제어하여 과적합을 방지한다.

---

## 4. 향후 연구 영향 및 고려사항

### 4.1 향후 연구에 미치는 영향

1. **TDA의 민주화**: giotto-tda는 위상수학 비전문가도 TDA를 쉽게 활용할 수 있게 하여, TDA 기반 연구의 저변을 확대
2. **재현 가능한 연구**: scikit-learn Pipeline 구조로 실험의 재현성 및 비교 가능성 향상
3. TDL(Topological Deep Learning)은 TDA와 딥러닝 기법을 결합하는 신흥 분야로, TDA는 데이터 형태에 대한 통찰을 제공하고 변형과 노이즈에 대한 견고성을 보이며, 이러한 속성은 딥러닝 파이프라인에서 바람직하다.

### 4.2 향후 연구 시 고려사항

1. **확장성(Scalability)**: 대규모 데이터셋에서의 persistent homology 계산 효율화 (approximate methods, sub-sampling 전략 등)
2. **End-to-End 미분 가능 TDA**: 현재 giotto-tda의 위상적 특징 추출은 미분 불가능하여 딥러닝 역전파에 직접 통합이 어려움
3. 일반화 오류 예측을 위해 TDL 모델을 계산 그래프에 학습시키거나, 신경망의 확률적 최적화 과정에서 추적되는 가중치 궤적의 위상적 속성을 제어하거나, 데이터의 내부 표현에 대한 바람직한 위상적 속성을 적절히 강제하는 것이 딥러닝의 일반화 행동을 이해하고 설명 가능한 AI에 한 걸음 더 다가가는 열쇠가 될 수 있다.
4. **다중 파라미터 persistent homology**: 단일 필트레이션을 넘어 다중 파라미터 설정으로의 확장
5. persistent homology의 고수준 추상화, 비위상적 변화에 대한 불감성 등의 한계를 극복하기 위한 persistent topological Laplacian, Dirac 연산자 등 새로운 방법론 탐구가 필요하다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 핵심 내용 | giotto-tda와의 관계 |
|------|------|-----------|-------------------|
| **Birdal et al.** "Intrinsic Dimension, Persistent Homology and Generalization in Neural Networks" (NeurIPS 2021) | 2021 | PHD를 통한 일반화 오류 바운딩 | giotto-tda의 PH 계산 활용 가능 |
| **de Surrel et al.** "RipsNet" (PMLR 2022) | 2022 | 신경망으로 PH를 빠르게 근사 | giotto-tda의 계산 병목 해결 시도 |
| **Zia et al.** "Topological Deep Learning: A Review" (AI Review 2024) | 2024 | TDA + 딥러닝의 포괄적 서베이 | giotto-tda를 주요 도구로 인용 |
| **Pérez et al.** "Predicting Generalization Gap using PH" (2022) | 2022 | PH 기반 일반화 갭 예측 | giotto-tda 및 giotto-ph를 직접 사용 |
| **Hajij et al.** "TopoX / TopoBenchmarkX" (2023-2024) | 2023+ | 고차 도메인에서의 TDL 벤치마크 | giotto-tda를 보완하는 새로운 생태계 |

### 최신 연구 동향 상세 분석

**(1) Persistent Topological Laplacians (2020~)**

스펙트럼 접근에서, persistent combinatorial Laplacian의 조화 스펙트라는 위상적 불변량의 변화를 포착하여 persistent homology가 캡처하는 위상 정보를 복원할 수 있으며, 비조화 스펙트라는 필트레이션 과정에서의 기하학적 형태 변화를 인코딩한다.

이는 giotto-tda가 다루는 persistent homology의 한계(비위상적 변화에 대한 불감성)를 극복하는 방향이다.

**(2) Topological Deep Learning (TDL)**

초기 TDL은 persistent homology가 생성한 특징을 DNN 입력 파이프라인에 통합하는 것을 의미했으나, 현재는 딥러닝에서 위상적 개념을 사용하는 아이디어와 방법의 집합 전체를 지칭한다.

**(3) TDA in NLP (2022~)**

Gardinazzi et al. (2024)은 LLM의 중복 레이어를 프루닝하기 위한 새로운 메트릭인 persistence similarity를 제안했으며, Balderas et al. (2025)은 중복 레이어 프루닝을 통해 BERT를 압축하는 Persistent BERT Compression and Explainability (PBCE)를 제안했다.

**(4) TDA in Molecular Science (2023~2025)**

TDA는 복잡한 분자 데이터에서 견고하고, 다중 스케일이며, 해석 가능한 특징을 추출하는 강력한 프레임워크로 부상했으며, persistent homology, persistent Laplacians, topological machine learning 등의 혁신을 포함하여 생체분자 안정성, 단백질-리간드 상호작용, 약물 발견, 재료 과학 등 다양한 도메인에서 변혁적 영향을 미치고 있다.

---

## 참고자료 및 출처

1. **Tauzin, G. et al.** (2021). "giotto-tda: A Topological Data Analysis Toolkit for Machine Learning and Data Exploration." *JMLR*, 22(39), 1–6. — [arxiv.org/abs/2004.02551](https://arxiv.org/abs/2004.02551)
2. **GitHub 저장소**: [github.com/giotto-ai/giotto-tda](https://github.com/giotto-ai/giotto-tda)
3. **JMLR 공식 페이지**: [jmlr.org/papers/v22/20-325.html](https://www.jmlr.org/papers/v22/20-325.html)
4. **Birdal, T. et al.** (2021). "Intrinsic Dimension, Persistent Homology and Generalization in Neural Networks." *NeurIPS 2021*. — [arxiv.org/abs/2111.13171](https://arxiv.org/abs/2111.13171)
5. **Zia, A. et al.** (2024). "Topological deep learning: a review of an emerging paradigm." *Artificial Intelligence Review*. — [Springer](https://link.springer.com/article/10.1007/s10462-024-10710-9)
6. **Wei, G.-W. et al.** (2025). "Topological data analysis and topological deep learning beyond persistent homology: a review." *Artificial Intelligence Review*. — [Springer](https://link.springer.com/article/10.1007/s10462-025-11462-w)
7. **Leykam, D. & Angelakis, D. G.** (2023). "Topological data analysis and machine learning." *Advances in Physics: X*, 8(1). — [arxiv.org/abs/2206.15075](https://arxiv.org/abs/2206.15075)
8. **Pérez, J.B. et al.** (2022). "Predicting the generalization gap in neural networks using topological data analysis." — [arxiv.org/pdf/2203.12330](https://arxiv.org/pdf/2203.12330)
9. **Chen, C. et al.** (2019). "A topological regularizer for classifiers via persistent homology." *AISTATS*.
10. **Pun, C.S. et al.** (2022). "Persistent-homology-based machine learning: a survey and a comparative study." *Artificial Intelligence Review*. — [Springer](https://link.springer.com/article/10.1007/s10462-022-10146-z)
11. **ACS Review** (2025). "A Review of Topological Data Analysis and Topological Deep Learning in Molecular Sciences." *J. Chem. Inf. Model.* — [pubs.acs.org](https://pubs.acs.org/doi/10.1021/acs.jcim.5c02266)
12. **Hajij, M. et al.** (2024). "Position: Topological Deep Learning is the New Frontier for Relational Learning." *ICML 2024*. — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11973457/)

> **참고**: 본 분석에서 제시된 수식 중 논문 본문에 명시적으로 포함되지 않은 것(예: Persistence Entropy, Landscape, Image의 일반 수학적 정의)은 TDA 분야의 표준 정의에 기반한 것이며, giotto-tda가 구현하고 있는 알고리즘의 수학적 기초를 설명하기 위해 포함하였습니다.
