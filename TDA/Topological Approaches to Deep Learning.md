

# Topological Approaches to Deep Learning

**저자:** Gunnar Carlsson (Stanford, 수학과), Rickard Brüel Gabrielsson (Stanford, 컴퓨터과학과)
**출처:** arXiv:1811.01122v1 [cs.LG], 2018년 11월 2일

---

## 1. 핵심 주장 및 주요 기여 요약

본 논문은 **위상적 데이터 분석(Topological Data Analysis, TDA)**을 딥러닝의 내부 상태 분석 및 신경망 구조 설계에 적용하는 두 가지 핵심 방향을 제시한다.

**핵심 주장:**
1. **CNN의 합성곱 층에서 학습된 가중치 벡터(weight vectors)는 자연 이미지 패치의 위상적 구조(원, Klein bottle 등)를 재현한다** — 이는 포유류 시각 경로(primary visual cortex)의 특성과 일관된다.
2. **피처 공간(feature space)의 기하학적 구조(geometry)를 활용하여 신경망 아키텍처를 체계적으로 구성하면, 학습 속도 향상과 일반화(generalization) 성능 개선이 가능하다.**

**주요 기여:**
- Persistent homology와 Mapper를 이용한 CNN 내부 상태의 위상적 분석
- 피처 공간의 메트릭 기반으로 신경망 연결 구조를 제한하는 수학적 형식주의(formalism) 제안
- 데이터 분석 기반, 사전 기하학(a priori geometry), 순수 데이터 주도(data-driven) 기하학의 세 가지 시나리오 제시
- MNIST→SVHN 간 교차 데이터셋 일반화 실험을 통한 초기 검증

---

## 2. 상세 분석: 문제, 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

1. **딥 뉴럴 네트워크의 해석 불가능성(interpretability):** 내부 상태가 어떻게 작동하는지 이해되지 않으며, 이는 적대적 공격(adversarial attack)에 대한 취약성과 직결된다.
2. **과적합(overfitting) 및 일반화 실패:** 하나의 데이터셋(예: MNIST)에서 학습한 모델이 유사한 다른 데이터셋(예: SVHN)에서는 사실상 랜덤 수준의 성능을 보인다.
3. **CNN 구조의 일반화 부재:** CNN의 합성곱 구조가 이미지의 그리드 기하학에 특화되어 있으며, 다른 유형의 데이터셋으로 일반화하는 체계적 방법론이 부족하다.

### 2.2 제안하는 방법

#### 2.2.1 피드포워드 시스템의 수학적 형식화

**정의 2.1 — 피드포워드 시스템:** 깊이 $r$의 방향성 비순환 그래프(DAG) $\Gamma$로, 꼭짓점 집합 $V(\Gamma)$가 레이어로 분해된다:

$$V(\Gamma) = V_0(\Gamma) \sqcup V_1(\Gamma) \sqcup \cdots \sqcup V_r(\Gamma)$$

여기서 $v \in V_i(\Gamma)$이면 모든 간선 $(v, w)$에 대해 $w \in V_{i+1}(\Gamma)$이다.

#### 2.2.2 활성화 및 계수 시스템을 통한 계산

각 비초기(non-initial) 꼭짓점 $v$에 대해 **활성자(activator)** $(\mu_v, S_v, f_v)$가 할당되고, 각 간선 $(u,v)$에 대해 계수 $\lambda_{(u,v)} \in S_v$가 할당된다. 레이어 간 함수는 다음과 같이 정의된다:

$$\varphi_i(g)(v) = f_v\left(\sum_{(u,v) \in \Gamma} \lambda_{(u,v)} \, g(u)\right)$$

전체 네트워크 함수는 다음의 합성으로 구성된다:

$$\Phi = \Phi(-; \mathcal{A}, \Lambda) = \varphi_r \circ \varphi_{r-1} \circ \cdots \circ \varphi_1$$

여기서 $\mathcal{A} = (\mu_v, S_v, f_v)$는 활성화 시스템, $\Lambda = \{\lambda_{(u,v)}\}$는 계수 시스템이다.

#### 2.2.3 손실 함수

분류 문제에서 정규화 함수 $\sigma_n$은 다음과 같이 정의된다:

$$\sigma_n(x_1, \ldots, x_n) = \frac{1}{x_1 + \cdots + x_n}(x_1, \ldots, x_n)$$

Softmax 함수는 $\sigma_n \circ \exp$로 정의되며, 여기서 $\exp(x_1, \ldots, x_n) = (e^{x_1}, \ldots, e^{x_n})$이다.

#### 2.2.4 메트릭 대응(Metric Correspondence)을 통한 구조 제한

핵심 아이디어는 피처 공간 $X$에 거리 함수 $d$가 주어졌을 때, **메트릭 대응**을 정의하는 것이다:

$$\mathcal{C}_d(r)(x) = \{x' \mid d(x, x') \leq r\}$$

이를 통해 신경망의 연결 구조를 피처 공간의 기하학에 따라 제한한다. 이미지의 경우 $L^\infty$ 거리가 사용되어 $3 \times 3$ 패치 내의 연결만 허용된다.

#### 2.2.5 각도 피처(Angular Features)를 통한 확장

자연 이미지 패치의 위상 분석 결과, 빈도 높은 패치들이 원(circle) 구조를 형성함을 발견하고, 이를 기반으로 선형 강도 함수를 정의한다:

$$f_\theta(x, y) = x\cos(\theta) + y\sin(\theta)$$

각 픽셀 $(m,n)$과 각도 $\theta$에 대해 새로운 피처를 구성한다:

$$q_{m,n,\theta}(p) = \sum_{(i,j) \in \mathcal{L}} p(m+i, n+j) \cdot f_\theta(i,j)$$

여기서 $\mathcal{L} = \{-1, 0, 1\} \times \{-1, 0, 1\}$은 $3 \times 3$ 격자이다.

이렇게 구성된 피처 공간의 연속 기하학은 $\mathbb{R}^2 \times S^1$이며, 이산화된 형태는 $\mathbb{Z}^2 \times \mu_n$ ($\mu_n$: $n$차 단위근)이다.

#### 2.2.6 Mapper 기반 순수 데이터 주도 기하학

피처 공간에 대한 사전 지식이 없는 경우, Mapper 구성[12]을 사용하여:
1. 피처 공간 $X$에 필터 함수 $f: X \to \mathbb{R}$을 적용
2. 오픈 커버링 $\mathcal{U}(l, s)$을 구성 — 길이 $l$, 보폭 $s$의 구간 $(ks - \frac{l}{2}, ks + \frac{l}{2})$, $k \in \mathbb{Z}$
3. 반복적 "더블링(doubling)" $\Gamma(0), \Gamma(1), \ldots, \Gamma(r)$을 통해 다중 해상도 Mapper 모델을 생성
4. 대응 $\mathcal{C}(\Gamma(i), \Gamma(i+1))$을 풀링의 일반화로 사용

### 2.3 모델 구조

#### MNIST 구조

깊이 6, 생성자 $F = F^c \times F^s$:
- 완전 인자: $F^c$의 유형 $[1, 64, 64, 32, 32, 64, 1]$
- 구조 인자:

$$G_{28} \xrightarrow{\mathcal{C}_d(1)} G_{28} \xrightarrow{\pi^2(0,1,2)} G_{14} \xrightarrow{\mathcal{C}_d(1)} G_{14} \xrightarrow{\pi^2(0,1,2)} G_7 \xrightarrow{\mathcal{C}^c} X(1) \xrightarrow{\mathcal{C}^c} X(10)$$

활성화 시스템:
- 합성곱 층 $F(1), F(3)$: $(+, \mathbb{R}, \text{ReLU})$, 여기서 $\text{ReLU}(x) = \max(0, x)$
- 풀링 층 $F(2), F(4)$: $(\max, \{1\}, \text{id})$
- 완전 연결 층 $F(5)$: $(+, \mathbb{R}, \text{ReLU})$
- 출력 층 $F(6)$: $(+, \mathbb{R}, \exp)$

#### 각도 피처 확장 구조

기존 구조에 다음의 각도 인자를 곱한다:

$$(\mu_{16})_+ \xrightarrow{\mathcal{C}^c} X(1) \xrightarrow{\mathcal{C}^c} X(1) \xrightarrow{\mathcal{C}^c} X(1) \xrightarrow{\mathcal{C}^c} X(1) \xrightarrow{\mathcal{C}^c} X(1) \xrightarrow{\mathcal{C}^c} X(1)$$

여기서 $(\mu_{16})\_+$는 $\mu_{16}$에 분리된 기저점(base point)을 추가한 것으로, 원래의 "원시(raw)" 픽셀 피처를 포함한다.

### 2.4 성능 향상

#### TDA 분석 결과
- **MNIST 제1 합성곱 층:** Mapper 분석과 persistent homology 바코드가 모두 **1차 베티 수 $\beta_1 = 1$의 원(circle) 구조**를 확인 — 자연 이미지 패치 분석[2]의 primary circle과 일치
- **CIFAR-10 컬러 채널 분리 분석:** 제1 합성곱 층에서 **$\beta_1 = 5$의 3-원(three circle) 모델**을 복원 — [2]의 결과와 일치
- **VGG16 분석:** 13개 합성곱 층에 대해 하위 층은 에지/선 감지(primary circle), 상위 층은 더 복잡한 패턴(bullseye, 교차선 등)을 캡처함을 확인

#### 학습 속도 향상
| 데이터셋 | 기존 CNN | 각도 피처 추가 CNN | 속도 향상 |
|---------|---------|------------------|---------|
| MNIST | 기준 | 각도 인자 포함 | **2배** |
| SVHN | 기준 | 각도 인자 포함 | **3.5배** |

#### 일반화 성능
| 학습 데이터 | 평가 데이터 | 기존 CNN 정확도 | 각도 피처 추가 CNN 정확도 |
|-----------|-----------|---------------|---------------------|
| MNIST | SVHN | ~10% (랜덤 수준) | **~22%** |

### 2.5 한계

1. **일반화 성능이 아직 불충분:** MNIST→SVHN 교차 평가에서 22%는 여전히 실용적 수준에 미달한다.
2. **가장 단순한 버전만 실험:** 논문에서 제안한 다양한 각도 인자(메트릭 대응 포함, 풀링 포함 등)의 고급 변형은 아직 실험되지 않았다.
3. **순수 데이터 주도 기하학(Mapper 기반) 구성은 미구현:** "개발 중(in development)"으로 명시되어 있다.
4. **제한적 실험 범위:** 이미지 데이터(MNIST, CIFAR-10, SVHN)에만 적용되었으며, 텍스트나 시계열 등 다른 데이터 유형에 대한 검증이 없다.
5. **정량적 위상 분석의 한계:** 밀도 기반 임계값(thresholding) 설정에 민감하며, 이를 위한 체계적 방법론이 부재하다.

---

## 3. 모델의 일반화 성능 향상 가능성 (중점 분석)

### 3.1 일반화 문제의 본질

논문은 일반화를 두 가지 수준으로 구분한다:

1. **표준 일반화(Standard Generalization):** 동일 데이터셋 내에서 학습 세트→테스트 세트
2. **교차 데이터셋 일반화(Cross-Dataset Generalization):** 한 데이터셋에서 학습한 모델을 완전히 다른 데이터셋에 적용

논문의 핵심 발견은 **교차 데이터셋 일반화가 기존 CNN에서 치명적으로 실패한다**는 것이다 — MNIST에서 학습한 모델을 SVHN에 적용하면 약 10% 정확도로, 10개 클래스에 대한 랜덤 선택과 동등하다.

### 3.2 위상적 접근이 일반화에 기여하는 메커니즘

#### (a) 피처 공간의 기하학적 구조가 데이터셋 불변 속성을 인코딩

자연 이미지 패치의 위상 분석[2]에서 발견된 원(circle) 및 Klein bottle 구조는 **특정 데이터셋에 종속되지 않는 보편적 속성**이다. 이 구조를 활용한 각도 피처 $f_\theta(x,y) = x\cos(\theta) + y\sin(\theta)$는 에지 방향이라는 데이터셋 불변 속성을 직접 인코딩한다.

#### (b) 연결 구조의 제한이 과적합을 감소

메트릭 대응 $\mathcal{C}_d(r)$을 통한 연결 제한은:
- 학습해야 할 파라미터 수를 줄여 과적합 위험을 감소시킨다.
- 피처 공간의 국소적(local) 관계만 캡처하도록 강제하여, 데이터셋 특화된 전역적(global) 상관을 학습하지 못하게 한다.

#### (c) 사전 구성된 피처가 "학습의 짐"을 경감

각도 피처를 입력에 포함함으로써 네트워크가 에지 감지를 "학습"할 필요 없이 **직접 사용**할 수 있다. 논문에서 명시적으로:

> "It permits the algorithm to use them directly rather than having to 'learn' them."

이는 학습 속도 향상(MNIST 2배, SVHN 3.5배)과 일반화 향상(10%→22%)에 모두 기여한다.

### 3.3 추가적 일반화 향상 가능성

논문이 제시하지만 **실험하지 않은** 고급 구성들:

1. **메트릭 대응을 포함한 각도 인자:**

$$\mu_n \xrightarrow{\mathcal{C}_d(\xi)} \mu_n \xrightarrow{\mathcal{C}^c} X(1) \xrightarrow{\mathcal{C}^c} X(1)$$

여기서 $\xi$는 단위원에서 인접한 단위근 사이의 거리 — 각도 방향에서도 국소성을 부과

2. **각도 방향 풀링 포함:**

$$(\mu_{4n})_+ \xrightarrow{\mathcal{C}_d(\xi_{4n})_+} (\mu_{4n})_+ \xrightarrow{(\pi_{4n,2n})_+} (\mu_{2n})_+ \xrightarrow{\mathcal{C}_d(\xi_{2n})_+} (\mu_{2n})_+ \xrightarrow{(\pi_{2n,n})_+} (\mu_n)_+ \xrightarrow{\mathcal{C}^c} X(1) \xrightarrow{\mathcal{C}^c} X(1)$$

3. **Klein bottle 기하학 기반 피처 구성** — 에지 방향뿐 아니라 에지의 "극성(polarity)"까지 인코딩

### 3.4 일반화에 대한 근본적 통찰

논문은 CNN의 성공이 두 가지 속성 — **국소성(Locality)**과 **동질성(Homogeneity)** — 에 기반한다고 분석한다:

- **국소성:** 피처 공간의 메트릭에 의한 연결 제한
- **동질성:** 합성곱 구조(weight sharing)에 의한 공간 불변성

이 두 속성을 이미지 그리드 $\mathbb{Z}^2$ 너머의 **일반적인 메트릭 공간**으로 확장하는 것이 본 논문의 핵심 프로그램이며, 이는 그래프, 시계열, 일반 행렬 데이터 등에 대해서도 CNN과 유사한 성능을 달성할 수 있는 **체계적 경로**를 제시한다.

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 연구적 영향

1. **TDA와 딥러닝의 융합 분야 개척:** 이 논문은 위상수학적 도구(persistent homology, Mapper)를 딥러닝의 해석 및 설계에 동시 적용하는 선구적 연구로, 이후 "Topological Deep Learning" 분야의 기초를 마련했다.

2. **피처 기하학 기반 아키텍처 설계 패러다임:** 데이터의 피처 공간에 내재된 기하학적/위상적 구조를 신경망 설계에 반영하는 아이디어는 Graph Neural Networks(GNN), Equivariant Neural Networks 등의 발전과 맥을 같이 한다.

3. **신경망 해석 가능성(Interpretability)에 대한 새로운 관점:** 가중치 벡터의 위상적 분석은 feature visualization, network dissection 등 기존 방법론과 상보적인 해석 도구를 제공한다.

4. **교차 데이터셋 일반화에 대한 문제 제기:** Domain adaptation/transfer learning 분야에서 피처 기하학의 역할에 대한 새로운 연구 방향을 제시한다.

### 4.2 향후 연구 시 고려할 점

1. **확장성(Scalability):** Persistent homology 계산의 시간 복잡도는 $O(n^3)$이며, 대규모 네트워크에 적용하기 위해서는 효율적인 근사 알고리즘이 필요하다.

2. **자동화된 기하학 선택:** 피처 공간의 메트릭 및 위상적 구조를 자동으로 발견하고 선택하는 방법론의 개발이 필요하다.

3. **이론적 보장:** 피처 기하학 기반 연결 제한이 일반화에 미치는 영향에 대한 **이론적 분석**(PAC-Bayes, Rademacher complexity 등)이 부재하며, 이를 개발해야 한다.

4. **다양한 데이터 유형에 대한 검증:** 이미지 외에 그래프, 분자 구조, 시계열, 텍스트 등에 대한 체계적 실험이 필요하다.

5. **end-to-end 학습과의 통합:** 사전 구성된 피처(각도 피처 등)와 학습된 피처를 동시에 최적화하는 프레임워크의 개발이 필요하다.

6. **적대적 강건성(Adversarial Robustness)과의 관계:** 논문이 초반에 언급한 adversarial behavior 문제에 대해 위상적 접근이 구체적으로 어떤 방어를 제공하는지에 대한 후속 연구가 필요하다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 Topological Deep Learning 분야의 발전

| 연구 | 핵심 내용 | 본 논문과의 관계 |
|-----|---------|-------------|
| **Hensel et al. (2021)** "A Survey of Topological Machine Learning Methods" [Front. Artif. Intell.] | TDA와 ML의 융합에 대한 포괄적 서베이 | Carlsson의 접근을 포함한 분야 전반을 정리 |
| **Papillon et al. (2023)** "Architectures of Topological Deep Learning: A Survey on Topological Neural Networks" [arXiv:2304.10031] | Cell complex, simplicial complex, hypergraph 위의 메시지 패싱 신경망 분류 체계 | Carlsson의 피처 기하학 아이디어를 고차 위상 구조로 대폭 확장 |
| **Bodnar et al. (2021)** "Weisfeiler and Leman Go Topological: Message Passing Simplicial Networks" [ICML 2021] | Simplicial complex 위의 메시지 패싱 → GNN의 표현력 한계 극복 | 본 논문의 "피처 공간의 기하학적 구조 활용" 아이디어를 simplicial 구조로 구현 |
| **Hajij et al. (2023)** "Topological Deep Learning: Going Beyond Graph Data" [arXiv:2206.00606] | Cell complex neural networks, simplicial attention networks 등 통합 프레임워크 | Carlsson의 "correspondences 기반 형식주의"를 범주론적으로 확장 |

### 5.2 Persistent Homology와 딥러닝의 결합

| 연구 | 핵심 내용 | 본 논문과의 관계 |
|-----|---------|-------------|
| **Brüel-Gabrielsson et al. (2020)** "A Topology Layer for Machine Learning" [AISTATS 2020] | Persistent homology를 미분 가능한 레이어로 구현 | 본 논문 공저자의 후속 연구 — TDA를 end-to-end 학습에 통합 |
| **Hofer et al. (2020)** "Topologically Densified Distributions" [ICML 2020] | Persistence diagram 기반 생성 모델 | Carlsson의 밀도 기반 위상 분석을 생성 모델에 적용 |
| **Hensel et al. (2021)** 의 서베이 내 PLay (Persistent Landscape) 기반 분류 연구들 | Persistent landscape를 피처로 사용한 분류/회귀 | 본 논문의 "가중치 벡터의 위상 분석"과 상보적 |

### 5.3 피처 기하학 및 등변 신경망(Equivariant Networks)

| 연구 | 핵심 내용 | 본 논문과의 관계 |
|-----|---------|-------------|
| **Cohen et al. (2019)** "Gauge Equivariant Convolutional Networks and the Icosahedral CNN" [ICML 2019] | 다양체(manifold) 위의 게이지 등변 CNN | Carlsson의 "국소성+동질성" 원리를 리만 기하학으로 엄밀하게 구현 |
| **Bronstein et al. (2021)** "Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges" [arXiv:2104.13478] | 기하학적 딥러닝의 통합 프레임워크 (5G) | Carlsson의 "피처 기하학 기반 아키텍처 설계"를 대칭 그룹, 게이지 이론의 관점에서 체계화 |
| **Weiler & Cesa (2019)** "General E(2)-Equivariant Steerable CNNs" [NeurIPS 2019] | 2D 유클리드 군 $E(2)$에 대한 등변 CNN | 본 논문의 $\mathbb{Z}^2$ 병진 대칭 활용을 전체 $E(2)$ 대칭(회전, 반사 포함)으로 확장 |

### 5.4 비교 분석 요약

**본 논문(Carlsson & Brüel-Gabrielsson, 2018)의 위치:**
- **선구적이지만 예비적(preliminary):** 수학적 형식주의는 풍부하나 실험은 제한적
- **이후 연구에 의해 구체화:** 피처 기하학 → Geometric Deep Learning[Bronstein 2021], Mapper 기반 구성 → Topological Neural Networks[Papillon 2023], 미분 가능한 TDA → Topology Layer[Brüel-Gabrielsson 2020]

**2020년 이후 연구와의 핵심 차별점:**
1. 본 논문은 **분석(analysis) 도구로서의 TDA**와 **설계(design) 원리로서의 피처 기하학**을 동시에 다룬 반면, 후속 연구들은 대부분 둘 중 하나에 집중한다.
2. 본 논문의 Mapper 기반 "순수 데이터 주도 기하학" 아이디어는 아직 완전히 실현되지 않았으며, 향후 연구의 잠재적 방향으로 남아 있다.
3. 교차 데이터셋 일반화에 대한 위상적 접근은 domain adaptation 분야에서 아직 충분히 탐구되지 않은 영역이다.

---

## 참고자료

1. **Carlsson, G. & Brüel-Gabrielsson, R.** "Topological Approaches to Deep Learning," arXiv:1811.01122v1, 2018. (본 분석 대상 논문)
2. **Carlsson, G., Ishkhanov, T., de Silva, V., & Zomorodian, A.** "On the local behavior of spaces of natural images," *Intl. Jour. Computer Vision*, 76, 2008, 1-12. (논문 참고문헌 [2])
3. **Gabrielsson, R.B. & Carlsson, G.** "A look at the topology of convolutional neural networks," arXiv:1810.03234v1, 2018. (논문 참고문헌 [4])
4. **Singh, G., Mémoli, F., & Carlsson, G.** "Topological methods for the analysis of high dimensional data sets and 3D object recognition," SPBG, 2007, 91-100. (논문 참고문헌 [12] — Mapper 원논문)
5. **Papillon, M. et al.** "Architectures of Topological Deep Learning: A Survey on Topological Neural Networks," arXiv:2304.10031, 2023.
6. **Bodnar, C. et al.** "Weisfeiler and Leman Go Topological: Message Passing Simplicial Networks," *ICML 2021*.
7. **Hajij, M. et al.** "Topological Deep Learning: Going Beyond Graph Data," arXiv:2206.00606, 2023.
8. **Brüel-Gabrielsson, R. et al.** "A Topology Layer for Machine Learning," *AISTATS 2020*.
9. **Bronstein, M. et al.** "Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges," arXiv:2104.13478, 2021.
10. **Cohen, T.S. et al.** "Gauge Equivariant Convolutional Networks and the Icosahedral CNN," *ICML 2019*.
11. **Weiler, M. & Cesa, G.** "General E(2)-Equivariant Steerable CNNs," *NeurIPS 2019*.
12. **Hensel, F. et al.** "A Survey of Topological Machine Learning Methods," *Frontiers in Artificial Intelligence*, 4, 2021.
13. **Hofer, C. et al.** "Topologically Densified Distributions," *ICML 2020*.

> **정확도 관련 참고사항:** 본 분석은 제공된 원 논문의 내용에 충실하게 작성되었습니다. 2020년 이후 비교 분석 부분의 일부 연구들에 대해서는 일반적으로 알려진 정보에 기반하였으며, 개별 논문의 세부 수치나 결과에 대해 100% 검증이 어려운 부분이 있을 수 있습니다.
