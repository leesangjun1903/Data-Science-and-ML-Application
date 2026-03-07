# A review of topological data analysis and topological deep learning in molecular sciences

**저자:** JunJie Wee, Jian Jiang (Michigan State University / Wuhan Textile University)
**출처:** arXiv:2509.16877v1 [q-bio.BM], 2025년 9월 23일

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장
본 논문은 **위상적 데이터 분석(Topological Data Analysis, TDA)**이 복잡한 분자 데이터로부터 **강건하고(robust), 다중 스케일(multiscale)이며, 해석 가능한(interpretable) 특징**을 추출하는 강력한 프레임워크임을 주장한다. 특히 TDA와 딥러닝을 결합한 **위상적 딥러닝(Topological Deep Learning, TDL)**이 분자과학 전 영역에서 기존 방법론을 능가하는 예측 성능과 해석력을 제공한다는 점을 강조한다.

### 주요 기여
1. **TDA의 역사적 발전 추적**: 초기 정성적(qualitative) 도구에서 고급 정량적·예측적 모델로의 진화를 체계적으로 정리
2. **방법론적 혁신 종합 정리**: 지속적 호몰로지(persistent homology), 지속적 라플라시안(persistent Laplacians), 원소 특이적 지속적 호몰로지(ESPH), 전기정적 지속성(electrostatic persistence) 등
3. **다양한 응용 영역 포괄**: 생체분자 안정성, 단백질-리간드 상호작용, 약물 발견, 재료과학, 바이러스 진화
4. **TDL의 실질적 성과 입증**: D3R Grand Challenge 우승, SARS-CoV-2 변이 예측(BA.2, BA.5를 약 2개월 앞서 예측)
5. **미래 연구 방향 제시**: LLM, 기반 모델(foundation models), AGI 등 차세대 AI와 TDA 통합 전망

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 향상 및 한계

### 2.1 해결하고자 하는 문제

분자과학 데이터는 다음과 같은 본질적 난제를 가진다:

- **고차원성(High dimensionality):** sc-RNA seq 데이터는 수만 차원에 달함
- **다중 스케일 상호작용(Multiscale interactions):** 전자, 원자, 잔기, 도메인, 분자 간 스케일
- **고차 상호작용(High-order interactions):** 다체(many-body) 효과
- **비선형 관계(Nonlinear relations):** 공유결합, 수소결합, van der Waals, $\pi$ - $\pi$ 스태킹, 정전기, 소수성 상호작용 등

기존의 기하학적·통계적 방법으로는 이러한 복잡한 위상적 불변량(topological invariants)과 패턴을 포착하기 어려우며, 이것이 TDA가 해결하고자 하는 핵심 문제이다.

### 2.2 제안하는 방법 (수식 포함)

#### (A) 지속적 호몰로지 (Persistent Homology)

지속적 호몰로지는 대수적 위상수학과 다중 스케일 분석을 결합한다. 점 구름(point cloud) 데이터에 대해 필트레이션 매개변수 $\epsilon$를 변화시키며 단체 복합체(simplicial complex) $K(\epsilon)$를 구성한다:

$$K(\epsilon_1) \subseteq K(\epsilon_2) \subseteq \cdots \subseteq K(\epsilon_n)$$

각 스케일에서 $k$차 Betti 수 $\beta_k$가 계산된다:
- $\beta_0$: 연결 성분(connected components)의 수
- $\beta_1$: 루프(loops/tunnels)의 수
- $\beta_2$: 공동(cavities/voids)의 수

위상적 특징의 탄생(birth, $b$)과 소멸(death, $d$)을 기록하여 **지속 다이어그램(persistence diagram)** $\{(b_i, d_i)\}$을 생성하고, 이를 벡터화하여 머신러닝 입력으로 사용한다. 주요 벡터화 방법으로는:
- **지속 바코드(persistence barcodes)** [64]
- **지속 이미지(persistence images)** [2]
- **지속 랜드스케이프(persistence landscapes)** [13]

#### (B) 지속적 라플라시안 (Persistent Laplacians)

Wang et al. [165]이 도입한 지속적 스펙트럴 이론은 지속적 호몰로지를 확장한다. $k$차 조합적 라플라시안은 다음과 같이 정의된다:

$$\Delta_k = \partial_{k+1} \partial_{k+1}^T + \partial_k^T \partial_k$$

여기서 $\partial_k$는 $k$차 경계 연산자(boundary operator)이다. 지속적 라플라시안 $\Delta_k^{[p,q]}$는 포함 사상(inclusion) $K_p \hookrightarrow K_q$에 대해 정의되며:

- **조화 스펙트라(harmonic spectra):** 영 고유값의 다중도 = $\beta_k$ (위상적 정보 복원)
- **비조화 스펙트라(non-harmonic spectra):** 비위상적 형상 진화 정보 제공

이 방법은 30개 이상의 데이터셋 테스트에서 지속적 호몰로지를 능가하는 성능을 보였다 [128].

#### (C) 원소 특이적 지속적 호몰로지 (Element-Specific Persistent Homology, ESPH)

전통적 지속적 호몰로지가 모든 원자를 무차별적으로 처리하는 한계를 극복하기 위해, 점 구름을 원소별 부분집합으로 분할한다 [17, 20]:

$$\text{Point Cloud} = \bigcup_{\alpha \in \mathcal{E}} S_\alpha$$

여기서 $\mathcal{E}$는 원소 유형(C, N, O, S 등)의 집합이다. 각 원소 쌍 $(S_\alpha, S_\beta)$에 대해 독립적으로 지속적 호몰로지를 계산하여:
- 탄소 원자 → **소수성 상호작용** 포착
- 질소-산소 원자 → **친수성 상호작용 및/또는 수소결합** 포착

이를 통해 결합 부위로부터 약 40Å까지 확장되는 소수성 상호작용을 밝혀냈다 [20].

#### (D) 전기정적 지속성 (Electrostatic Persistence)

분자 정전기를 위상적 분석에 통합하기 위해, 원자 부분 전하(atomic partial charges)를 전하 매립 체계(charge embedding scheme)를 통해 단체 복합체에 삽입한다 [14]. 이를 통해 **물리 정보 기반 신경망(Physics-Informed Neural Networks, PINNs)**의 개발이 가능해진다.

#### (E) 지속적 경로 위상 (Persistent Path Topology, PPT)

원소 유형을 매립(embedding)으로 통합하여 분자를 특성화하며, 거리 기반 필트레이션과 각도 기반 필트레이션을 모두 제공한다 [30]. 이를 기반으로 **위상적 섭동 분석(Topological Perturbation Analysis, TPA)**이 복잡 네트워크의 핵심 노드 식별에 활용된다.

#### (F) 지속적 쉬프 라플라시안 (Persistent Sheaf Laplacian, PSL)

단백질 유연성(B-factor) 분석을 위해 도입되었으며, 국소적 위상 및 기하 정보를 다중 스케일 조화·비조화 스펙트라를 통해 표현한다 [68, 177].

### 2.3 모델 구조

본 리뷰에서 다루는 주요 AI 모델 아키텍처는 다음과 같다:

| 모델명 | 구조 | 적용 분야 |
|--------|------|---------|
| **TopologyNet** [19] | ESPH + CNN/MTNN | 단백질-리간드 결합 친화도, 돌연변이 안정성 |
| **TopNetTree** [161] | PH 특징 + CNN + Gradient Boosting Trees | PPI 결합 친화도 변화 예측 |
| **TopNetmAb** [35, 36] | TDA + AI 앙상블 | SARS-CoV-2 RBD-ACE2/항체 결합 에너지 |
| **TopLapNetGBT** [37] | Persistent Laplacian + Deep Learning + GBT | 돌연변이 유도 PPI 결합 에너지 변화 |
| **TopoFormer** [27, 28] | 3D 위상 서열 + Transformer/LLM | 단백질-리간드 상호작용 |
| **TopoDockQ** [50] | TDL + DockQ 예측 | 펩타이드-단백질 복합체 선별 |
| **TIDAL** [194] | 다중 스케일 위상 라플라시안 + Bidirectional Transformer + Ensemble NN | 약물 중독 가상 선별 |
| **PLD-Tree** [188] | Persistent Laplacian + Decision Tree | 단백질-단백질 결합 친화도 |
| **PerSpect-EL** [173] | Persistent Spectral + Ensemble Learning | PPI 결합 친화도 |

### 2.4 성능 향상

**주요 성능 입증 사례:**

1. **D3R Grand Challenges 우승** [115, 117]: 컴퓨터 지원 약물 설계의 글로벌 연례 경쟁에서 TDL 모델이 최고 성적 달성
2. **SARS-CoV-2 변이 예측**: BA.4와 BA.5가 주도 변이가 될 것을 WHO 공식 발표(2022년 6월) 약 2개월 전에 정확히 예측 [37]
3. **단백질 분류**: MTF-SVM이 M2 채널 약물 결합 구분에서 96% 정확도, 단백질 도메인 식별에서 85% 성공률 [15]
4. **단백질-리간드 결합 친화도**: PDBbind 벤치마크에서 기존 SOTA 모델 대비 우수한 성능 [20, 108, 196]
5. **독성 예측**: ClinTox 데이터셋에서 SOTA 대비 2.4% 향상 [131]
6. **TopoDockQ**: AlphaFold2의 내장 신뢰도 점수 대비 false positive 42% 이상 감소, precision 6.7% 향상 [50]
7. **재료 과학**: 결함 민감 특성 예측 오류 55% 감소 [159]
8. **30개 이상 데이터셋 테스트**: Persistent Laplacian이 persistent homology를 체계적으로 능가 [128]

### 2.5 한계

1. **지역화(Localization) 부재**: 지속적 호몰로지의 위상적 불변량은 전체 데이터셋에 대한 전역적(global) 특성 → B-factor 등 국소적 예측에 부적합
2. **점 구름 데이터에 국한**: 시퀀스, 매니폴드 등 다양한 데이터 유형에 대한 직접 적용 한계
3. **비위상적 정보 표현 불가**: 기하학적 세부정보의 과도한 단순화
4. **계산 복잡도**: 대규모 분자 데이터(예: 다수의 교차점을 갖는 Khovanov 호몰로지)에 대한 계산적 도전
5. **단순 데이터에 부적합**: TDA의 내재적 단순화가 단순한 데이터에서는 필수 정보 손실 초래 가능
6. **급변하는 AI 기술 대비 위치 설정**: AlphaFold, ChatGPT 등 급속 진화하는 AI와의 통합이 아직 열린 문제

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화 성능을 높이는 TDA의 본질적 특성

TDA 기반 특징이 일반화 성능을 향상시키는 근본적 이유는 **위상적 불변량의 안정성(stability)**에 있다:

$$d_B(\text{Dgm}(f), \text{Dgm}(g)) \leq \|f - g\|_\infty$$

여기서 $d_B$는 bottleneck distance, $\text{Dgm}(\cdot)$는 persistence diagram이다. 이 안정성 정리는 입력 데이터의 작은 섭동이 위상적 특징에 제한된 변화만 일으킴을 보장하여, **노이즈에 강건한 특징 추출**을 가능하게 한다.

### 3.2 다양한 화학적 스캐폴드에 걸친 일반화

논문에서 언급된 일반화 향상 메커니즘:

1. **원소 특이적 분할(ESPH)**: 원자 유형별 분할을 통해 물리·화학적으로 의미 있는 상호작용을 보존하므로, 새로운 분자 스캐폴드에 대한 전이 학습(transfer learning) 효과 극대화 [17, 20]

2. **다중 스케일 위상 라플라시안**: 여러 스케일에서 위상적·비위상적 정보를 동시 추출하여, 단일 스케일 의존성 탈피
$$\text{Features} = \{\lambda_i^{(k)}(\epsilon)\}_{i,k,\epsilon}$$
여기서 $\lambda_i^{(k)}(\epsilon)$는 스케일 $\epsilon$에서 $k$차 라플라시안의 $i$번째 고유값

3. **위상적 퍼뮤테이션 불변성**: 분자 내 원자 순서에 무관한 특징 생성 → 다양한 분자 표현에 대한 일관된 예측

4. **가상 선별에서의 일반화**: "TDA를 통합함으로써, 가상 선별은 분자 공간의 숨겨진 기하학적·위상적 특징을 포착하여... 다양한 화학적 스캐폴드에 걸쳐 일반화를 향상시킨다" (Section 4.2)

### 3.3 크로스 도메인 일반화 사례

| 일반화 측면 | 구체적 사례 |
|-----------|---------|
| **단백질 → 바이러스** | TopNetTree가 8,338개 PPI 데이터로 훈련 후 SARS-CoV-2 데이터에 전이 적용 [40] |
| **구조 → 서열** | 구조 기반 ESPH 특징 + 서열 기반 ESM 임베딩의 결합으로 상보적 정보 활용 [128] |
| **약물 발견 → 약물 재창출** | TDA 기반 위상 서명의 교차 종 유사성으로 광범위 항균 활성 암시 [154] |
| **결정 → 비정질** | 지속적 호몰로지가 BCC/FCC 결정 구조부터 비정질 재료까지 분류 가능 [143] |

### 3.4 일반화 성능 향상을 위한 구체적 기술적 발전

1. **Persistent Sheaf Laplacian (PSL)**: 국소적 위상·기하 정보를 동시에 캡처하여 단백질 B-factor 예측의 일반화 향상 [68]

2. **Persistent Path Topology (PPT)**: 원소별 분할이나 persistent cohomology 없이도 분자 구조를 다룰 수 있어, 다양한 분자 유형에 대한 범용성 확보 [30]

3. **다중 모달리티 통합**: 위상적 특징 + Transformer 기반 사전 훈련 임베딩의 결합
   - TopLapNet: Persistent Laplacian + Deep Learning [37]
   - TopoFormer: 3D 구조 → 위상 서열 → LLM [27, 28]

4. **다중 작업 학습(Multi-task Learning)**: 독성, 용해도, 분배 계수 등 관련 작업의 동시 학습으로 공유 표현의 일반화 능력 향상 [179, 180]

5. **AlphaFold 통합**: AlphaFold3 예측 구조를 TDA 파이프라인에 통합하여 실험적 구조 부재 시에도 일반화 가능 [66, 172]

---

## 4. 앞으로의 연구에 미치는 영향과 연구 시 고려할 점

### 4.1 연구 영향

#### (A) 새로운 연구 패러다임 확립
- **위상적 딥러닝(TDL)**이 그래프 표현 학습(graph representation learning)과 기하학적 딥러닝(geometric deep learning)을 보완하는 **제3의 관계적 학습 프론티어**로 자리매김 [122]
- 해석 가능한 신경망(Interpretable Neural Networks)의 발전을 촉진

#### (B) 분야별 파급효과
- **약물 발견**: TDA 기반 가상 선별과 결합 친화도 예측의 산업적 채택 가속화
- **전염병 대응**: 바이러스 변이 조기 예측 능력으로 공중보건 의사결정 지원
- **단백질 공학**: 방향성 진화(directed evolution)의 돌연변이 공간 ($20^N$) 탐색 효율화
- **재료 과학**: 고처리량 스크리닝에서 TDA 기반 디스크립터의 역할 확대

#### (C) 학제간 융합 촉진
수학, 생물학, 화학, 재료과학, 물리학, 컴퓨터과학을 통합하는 학제간 연구 환경 조성의 필요성 강조

### 4.2 향후 연구 시 고려할 점

#### (1) 새로운 위상적 불변량 개발
- **상호작용 호모토피 및 상호작용 호몰로지** [87]: 단백질-리간드, 단백질-단백질, 약물-타겟, 항체-항원 상호작용에 적용
- **Persistent sheaf 분석** [177]과 **가중 지속적 호몰로지**: 분자 작용기(벤젠, 에스테르, 싸이올, 아민 등)에 가중치 부여 가능한 하이퍼그래프 임베딩 설계
- **미분 위상(Differential Topology)** [149]과 **기하 위상(Geometric Topology)** [136]의 역할 확대

#### (2) 국소화된 위상 분석
- 위상적 섭동 분석(TPA)의 **국소화 버전** 개발 → 복잡한 생물학적 네트워크 내 기능적 모듈의 정밀 검출
- Persistent cohomology [21], persistent sheaf Laplacians [177], persistent interaction topology [87, 88]이 국소적 위상 분석을 가능하게 하나, 추가 발전 필요

#### (3) 최신 AI 기술과의 통합
- **대규모 언어 모델(LLMs)**: 위상 서열(topological sequences)을 LLM에 입력하여 분자 상호작용 분석
- **기반 모델(Foundation Models)**: 대규모 비표지 데이터에 대한 사전 훈련 후 TDA 기반 모델의 성능 부스팅
- **인공 일반 지능(AGI)** 및 **모델 컨텍스트 프로토콜(MCP)**: 차세대 AI 플랫폼과의 통합
- AlphaFold, ChatGPT 등과의 시너지 극대화 [128, 164]

#### (4) 계산 효율성
- 대규모 분자 데이터에 대한 효율적 TDA 계산 도구 개발 (예: PETLS 소프트웨어 [79])
- Khovanov 호몰로지 등 복잡한 위상적 계산의 확장성 확보 [136, 138, 139]

#### (5) 데이터 관련 고려사항
- **소규모 데이터 문제**: 분자과학의 소규모 독성/용해도 데이터셋에 대한 다중 작업 학습 전략 [52, 179]
- **대규모 딥 돌연변이 스캐닝 데이터베이스**: 단백질 공학 모델 성능의 지속적 향상 기반 [128]

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연도 | 연구/방법 | 핵심 기여 | TDA와의 관계 |
|------|---------|---------|------------|
| 2020 | **Persistent spectral graph** (Wang et al. [165]) | Persistent Laplacian 도입, 비위상적 형상 정보 추출 | TDA의 스펙트럴 확장 |
| 2020 | **TopNetTree** (Wang et al. [161]) | PH + CNN + GBT로 PPI 돌연변이 결합 친화도 예측 | *Nature Machine Intelligence* 게재 |
| 2020 | **Weighted PH for RNA** (Pun et al. [126]) | 가중 지속적 호몰로지로 RNA 유연성 분석 | ESPH 아이디어 확장 |
| 2021 | **PerSpect ML** (Meng & Xia [108]) | Persistent spectral 기반 ML로 단백질-리간드 결합 친화도 SOTA | *Science Advances* 게재 |
| 2021 | **AlphaFold2** (DeepMind [1, 174]) | 단백질 구조 예측 혁명 | TDA와 통합 연구 진행 중 [66, 172] |
| 2022 | **SARS-CoV-2 BA.4/BA.5 예측** (Chen et al. [37]) | TopLapNetGBT로 주도 변이 2개월 전 예측 | Persistent Laplacian + DL |
| 2022 | **Topological perovskite design** (Anand et al. [5]) | PH + persistent Ricci curvature로 태양전지 재료 설계 | TDA의 재료과학 확장 |
| 2023 | **Persistent spectral protein engineering** (Qiu & Wei [128]) | Persistent Laplacian 기반 단백질 공학 | *Nature Computational Science* 게재 |
| 2023 | **Path topology** (Chen et al. [30]) | 분자·재료과학에 경로 위상 도입, TPA 제안 | *J. Phys. Chem. Lett.* 게재 |
| 2023 | **Persistent Dirac** (Wee et al. [169]) | 분자 표현을 위한 persistent Dirac 연산자 | 양자 지속성(quantum persistence) 일반화 |
| 2024 | **TopoFormer** (Chen et al. [27]) | 3D 위상 → 서열 변환 + Transformer | *Nature Machine Intelligence* 게재 |
| 2024 | **TDL position paper** (Papamarkou et al. [122]) | TDL을 "관계적 학습의 새 프론티어"로 공식 선언 | ICML 2024 게재 |
| 2024 | **AlphaFold3** (Abramson et al. [1]) | 생체분자 상호작용 구조 예측 | TDA 통합 파이프라인 개발 [66, 171, 172] |
| 2025 | **Persistent Sheaf Laplacian** (Hayes et al. [68]) | PSL로 단백질 유연성(B-factor) 분석 | 국소적 TDA의 새 방향 |
| 2025 | **PDFL** (Zia et al. [196]) | Persistent directed flag Laplacian으로 단백질-리간드 결합 예측 | 방향성 복합체의 TDA 활용 |
| 2025 | **LSIC discovery** (Chen et al. [31]) | TDA 기반 다중스케일 학습으로 14개 새 리튬 초이온 전도체 발견 | *JACS* 게재 |
| 2025 | **Knot data analysis** (Shen et al. [134]) | 다중스케일 Gauss 연결 적분으로 매듭 데이터 분석 | *PNAS* 게재 |
| 2025 | **Category/Delta complex TSA** (Liu et al. [90, 92]) | 범주론/Δ-복합체 기반 게놈 위상 서열 분석 | 서열 데이터의 TDA 일반화 |

### 비교 분석의 핵심 트렌드

1. **스펙트럴 확장의 우위**: 2020년 이후 persistent Laplacian 계열(PSL, PDFL, path Laplacian 등)이 기존 persistent homology를 체계적으로 능가하는 경향 → 비위상적 형상 정보의 중요성 확인

2. **AI 모델 진화와의 동반 성장**: Transformer(2024 TopoFormer), LLM, AlphaFold 등 최신 AI 아키텍처와의 통합이 가속화

3. **응용 범위의 급속 확장**: 2020년 이전 생체분자 중심 → 2020년 이후 재료과학, 게놈학, 단일세포 분석, 약물 재창출 등으로 확장

4. **이론적 심화**: Persistent Mayer homology [137], persistent Khovanov homology [91], interaction homology [87] 등 수학적 기초의 지속적 발전

---

## 참고자료

1. **주 논문**: Wee, J. & Jiang, J. "A review of topological data analysis and topological deep learning in molecular sciences." arXiv:2509.16877v1, 2025.
2. Cang, Z. & Wei, G.-W. "TopologyNet: Topology based deep convolutional and multi-task neural networks for biomolecular property predictions." *PLoS Computational Biology*, 13(7):e1005690, 2017. [19]
3. Wang, R., Nguyen, D.D. & Wei, G.-W. "Persistent spectral graph." *Int. J. Numer. Methods Biomed. Eng.*, 36(9):e3376, 2020. [165]
4. Qiu, Y. & Wei, G.-W. "Persistent spectral theory-guided protein engineering." *Nature Computational Science*, 3(2):149–163, 2023. [128]
5. Chen, J. et al. "Persistent Laplacian projected Omicron BA.4 and BA.5 to become new dominating variants." *Computers in Biology and Medicine*, 151:106262, 2022. [37]
6. Chen, D., Liu, J. & Wei, G.-W. "Multiscale topology-enabled structure-to-sequence transformer for protein–ligand interaction predictions." *Nature Machine Intelligence*, 6(7):799–810, 2024. [27]
7. Papamarkou, T. et al. "Position: Topological deep learning is the new frontier for relational learning." *Proc. Machine Learning Research*, 235:39529, 2024. [122]
8. Xia, K. & Wei, G.-W. "Persistent homology analysis of protein structure, flexibility, and folding." *Int. J. Numer. Methods Biomed. Eng.*, 30(8):814–844, 2014. [183]
9. Cang, Z. & Wei, G.-W. "Integration of element specific persistent homology and machine learning for protein-ligand binding affinity prediction." *Int. J. Numer. Methods Biomed. Eng.*, 34(2):e2914, 2018. [20]
10. Hayes, N. et al. "Persistent sheaf Laplacian analysis of protein flexibility." *J. Phys. Chem. B*, 129(17):4169–4178, 2025. [68]
11. Zia, M. et al. "Persistent directed flag Laplacian (PDFL)-based machine learning for protein–ligand binding affinity prediction." *J. Chem. Theory Comput.*, 21(8):4276–4285, 2025. [196]
12. Wang, M., Cang, Z. & Wei, G.-W. "A topology-based network tree for the prediction of protein–protein binding affinity changes following mutation." *Nature Machine Intelligence*, 2(2):116–123, 2020. [161]
