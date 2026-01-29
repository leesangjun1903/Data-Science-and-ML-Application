
# Sheaf Neural Networks

***

## 1. 핵심 주장 및 주요 기여 (Core Claims and Key Contributions)

**Sheaf Neural Networks** (Hansen & Gebhart, 2020)는 그래프 신경망의 근본적인 제약을 해결하는 획기적인 일반화를 제시합니다. 표준 GCN이 모든 엣지에 대해 동일하고 대칭적인 관계(trivial sheaf)만 처리할 수 있다는 한계를 극복하기 위해, 대수 위상수학의 **cellular sheaf 구조**를 도입했습니다.[1]

### 주요 기여도
1. **Graph Laplacian의 일반화**: 표준 그래프 라플라시안을 sheaf Laplacian으로 확장하여 비균등, 비대칭, 가변 차원의 엣지 관계를 표현[1]
2. **확산 연산자의 재정의**: 신호 처리 이론을 기반으로 한 새로운 diffusion operator 제안[1]
3. **부호 있는 그래프 성능**: 양수/음수 엣지를 가진 그래프 분류에서 GCN을 유의미하게 능가[1]
4. **대수-위상 기초**: 그래프 신경망 연구에 범주론적 수학 도구 도입[2]

***

## 2. 해결하고자 하는 문제 (Problem Formulation)

### 표준 GCN의 한계

표준 GCN은 다음 방정식으로 정의됩니다:

$$\text{GCN}(X,A) = σ(D^{-1/2}AD^{-1/2}XW)$$

여기서 암묵적으로 모든 노드 간 관계가 **trivial sheaf**를 따릅니다:
- 모든 엣지가 동일한 구조(constant scalar weights)
- 대칭적 관계(Fv⊴e = Fu⊴e)
- 1차원 벡터 공간만 처리

### 실제 문제 사례

**문제 1: 비대칭 관계 처리 불능**
- 추천 시스템의 방향성 선호도
- 소셜 네트워크의 신뢰/불신 관계
- 지식 그래프의 다양한 엣지 타입

**문제 2: 부호 있는 그래프 성능 저하**
다수의 실제 네트워크에서 "약한 호모필리(low homophily)" 발생:
- 노드 $v$와 $u$가 연결되어 있어도 다른 라벨을 가짐
- GCN의 neighborhood averaging이 역효과

**문제 3: 과도한 평활화(Oversmoothing)**
깊은 네트워크에서 모든 노드의 표현이 수렴: $X^{(L)} → c·\mathbf{1}$

***

## 3. 제안하는 방법 (Methodology)

### 3.1 Cellular Sheaf 구조

그래프 $G = (V, E)$ 위의 cellular sheaf $F$는 다음으로 구성:[1]

$$\text{Cellular Sheaf } (G, F):$$
$$\bullet \text{ Vector space } F(v) \text{ for each vertex } v \in V$$
$$\bullet \text{ Vector space } F(e) \text{ for each edge } e \in E$$  
$$\bullet \text{ Linear map } F_{v⊴e}: F(v) → F(e) \text{ for each incident pair}$$

**직관적 의미**: 노드 $v$의 "private opinion" $x_v ∈ F(v)$가 엣지 $e$의 "discourse space" $F(e)$에서 어떻게 표현되는지를 정의합니다.

### 3.2 Coboundary 연산자 (Coboundary Operator)

노드 신호를 엣지 신호로 매핑:[1]

$$\boxed{(δx)_e = F_{v⊴e}x_v − F_{u⊴e}x_u}$$

여기서 $e = (u,v)$이고, 지향성(orientation)을 임의로 선택할 수 있습니다.

**의미**: 엣지의 양 끝에서 노드의 "의견"이 얼마나 "불일치"하는지를 측정합니다.

### 3.3 Sheaf Laplacian

Coboundary 연산자로부터:[1]

$$\boxed{L_F = δ^T δ}$$

명시적으로 노드 $v$에 대해:

$$\boxed{(L_F x)_v = \sum_{v⊴e} F_{v⊴e}^T(F_{v⊴e}x_v − F_{u⊴e}x_u)}$$

**성질**:
- 양의 준정부호(positive semidefinite)
- 방향성 선택과 무관(orientation-independent)
- Kernel: global sections $H^0(G;F) = \{x : F_{v⊴e}x_v = F_{u⊴e}x_u, \forall e\}$

### 3.4 정규화된 Sheaf Laplacian

Block-diagonal 행렬 $D$에 대해:[1]

$$\boxed{\tilde{L}_F = D^{-1/2}L_F D^{-1/2}}$$

여기서 각 블록 $D_{vv} = d_v I_k$ (degree 기반, 스톡 차원 $k$)

**장점**: 고유값이 $[0,2]$ 범위 → Diffusion에 안정적[3]

### 3.5 Sheaf Diffusion 연산자

**기본형**:

$$\boxed{H_F^α = I − αL_F}$$

(적절한 $α$ 선택으로 2-norm = 1 보장)

**정규화형**:

$$\boxed{\tilde{H}_F = I − \tilde{L}_F}$$

**r-step diffusion**:

$$(L_F)^r \text{ 및 } (H_F^α)^r$$
로 다양한 이웃 범위 처리 가능

### 3.6 Sheaf Convolution (다항식 매개변수화)

$D_F = SΛS^{-1}$ 고유분해를 이용:[1]

$$\text{Convolution: } x * y = S(S^{-1}x ⊙ S^{-1}y)$$

(고유 기저에서 element-wise 곱)

이를 다항식으로 근사:

$$\boxed{P(D_F) = a_0 I + a_1 D_F + \cdots + a_N D_F^N}$$

**이점**: 고유값 분해 계산 불필요 → 계산 효율적

### 3.7 SheafConv 신경망 층 (SheafConv Layer)

Hyperparameters: $N_{in}^{\text{feat}}$, $N_{out}^{\text{feat}}$, 스톡 차원 $k$, diffusion operator $D_F$, nonlinearity $ρ: \mathbb{R}^k → \mathbb{R}^k$

Learnable parameters: $A ∈ \mathbb{R}^{N_{in}^{\text{feat}} × N_{out}^{\text{feat}}}$, $B ∈ \mathbb{R}^{k×k}$

**Forward pass**:[1]

$$\boxed{\text{SheafConv}(A, B)(X) = ρ(D_F(I_N ⊗ B)XA)}$$

where:
- $X ∈ \mathbb{R}^{Nv·k × N_{in}^{\text{feat}}}$ (모든 노드의 $k$-차원 feature 스택)
- $I_N ⊗ B$ = Kronecker 곱 (각 노드에 $B$ 적용)
- $D_F$ = Diffusion 연산자

### 3.8 다중 Diffusion 연산자 결합

성능 향상을 위해 다수의 diffusion 연산자를 병렬 사용:

$$\text{Output} = \text{concat}(\text{SheafConv}_1(X), \text{SheafConv}_2(X), \ldots)$$

또는 학습 가능한 선형 결합:

$$\text{Output} = \sum_{i=1}^{K} α_i · \text{SheafConv}_i(X)$$

***

## 4. 모델 구조 (Model Architecture)

### 4.1 계층 구조

$$\begin{array}{|c|c|}
\hline
\text{Input Layer} & X_0 ∈ \mathbb{R}^{N_v·k × f_0} \\
\hline
\text{SheafConv Layer 1} & X_1 = \text{SheafConv}_1(A_1, B_1)(X_0) \\
& \text{Output: } X_1 ∈ \mathbb{R}^{N_v·k × f_1} \\
\hline
\text{SheafConv Layer 2} & X_2 = \text{SheafConv}_2(A_2, B_2)(X_1) \\
& \text{Output: } X_2 ∈ \mathbb{R}^{N_v·k × f_2} \\
\hline
\vdots & \vdots \\
\hline
\text{Output Layer} & \text{Final classification/regression} \\
\hline
\end{array}$$

### 4.2 원본 논문의 실험 설정

**아키텍처 A**: 32 hidden dimension × 3 layers (SheafNN-32)  
**아키텍처 B**: 16 hidden dimension × 4 layers (SheafNN-16)

비교 대상: GCN-32 및 GCN-16 (동일한 구조, trivial sheaf)

***

## 5. 성능 향상 분석 (Performance Enhancement Analysis)

### 5.1 원본 논문 결과 (합성 부호 있는 그래프)

**선형 특성(Linear Feature) 구성**:
- Input feature: $x_{in,v} = Px_v + \epsilon_v$ (선형 변환)
- Class label: $C_v = \text{sign}(⟨c, x_v⟩)$ (이진 분류)
- 부호 있는 엣지: 내부 유사도 기반, 가중치 $w_{uv} = ⟨x_u, x_v⟩ + ε_{uv}$

**결과**: SheafNN-32 및 SheafNN-16이 대부분 GCN 변형을 능가[1]

### 5.2 Neural Sheaf Diffusion 실제 성능 (2022)

9개 데이터셋에서 9가지 방법론 비교:[2]

| 데이터셋 | 호모필리 | Diag-NSD | O(d)-NSD | 최고 기준선 | 우위 |
|---------|--------|----------|----------|-----------|------|
| Texas   | 0.11   | 85.67%   | **85.95%** | GGCN 84.86% | **+1.09%** |
| Wisconsin| 0.21  | 88.63%   | **89.41%** | GGCN 86.86% | **+2.55%** |
| Film    | 0.22   | 37.79%   | 37.81%   | GCNII 37.44%| −0.35% |
| Squirrel| 0.22   | 54.78%   | **56.34%** | GGCN 55.17% | **+1.17%** |
| Chameleon| 0.23  | **68.68%** | 68.04%   | GGCN 71.14% | −2.46% |
| Cornell | 0.30   | **86.49%** | 84.86%   | GGCN 85.68% | **+0.81%** |
| Citeseer| 0.74   | 77.14%   | 76.70%   | 77.XX%    | ~0% |
| Pubmed  | 0.80   | 89.42%   | **89.49%** | GCNII 90.15%| −0.66% |
| Cora    | 0.81   | 87.14%   | 86.90%   | GCNII 88.37%| −1.23% |

**핵심 발견**:
- **이질성 그래프(h < 0.3)**: 6개 중 5개에서 1위 달성
- **동질성 그래프(h > 0.7)**: 기준선 대비 ~1% 범위
- **전체**: 9개 중 8개 데이터셋에서 상위 3위[2]

### 5.3 모델 변형별 성능

**Diag-NSD (대각 제약)**
- 계산 효율: O(mdc) (d=스톡 차원)
- 표현력: 각 차원이 독립적
- 안정성: 우수

**O(d)-NSD (직교 번들)**
- 전체 최고 성능
- 과적합 방지 (직교 제약)
- 이론적 해석: parallel transport

**Gen-NSD (일반 행렬)**
- 최대 유연성
- 과적합 위험
- 수치 안정성 문제 가능

***

## 6. 일반화 성능 향상 메커니즘 (Generalization Mechanism)

### 6.1 이질성(Heterophily) 처리

**이질성 정의**: 노드 호모필리 비율 

```math
h = \frac{\# \text{edges between different classes}}{|\text{E}|}
```

**GCN의 실패 이유**: Heat diffusion 가정
- $\dot{X}(t) = −\tilde{L}_0 X(t)$ 
- 이웃 노드 특성을 평균화 → 이질 그래프에서는 신호 손실

**SheafNN의 해결책**: 비대칭 제약 맵

완전 이분 그래프($A ∪ B$, 모든 엣지가 클래스 간)에 대해:[1]

$$F_{v⊴e} = \begin{cases} α & v ∈ A \\ −α & v ∈ B \end{cases}$$

그러면 극한에서($t → ∞$):
$$\hat{x}_v^{(∞)} ∈ \text{span}\{\text{polarized features}\}$$

즉, 클래스를 선형 분리 가능하게 만듭니다.

### 6.2 고차원 스톡(Higher-dimensional Stalks)

**핵심 정리**: 
- 1-차원 스톡(GCN): $\dim(\ker(L_F)) ≤ 1$ → 최대 이진 분류
- d-차원 스톡: $\dim(\ker(L_F)) ≤ d$ → d-클래스 분류 가능[2]

**다중 클래스 분류 예시**:

3-클래스 문제에 필요한 최소 스톡 차원:
- 대각 제약: $d ≥ 3$
- 직교 맵: $d ≥ 2$ (효율적 공간 활용)

### 6.3 과도한 평활화(Oversmoothing) 저항

**표준 GCN의 에너지 감소**:

$$E_0(X^{(t+1)}) ≤ λ^* ||W||_2^2 E_0(X^{(t)})$$

where $λ^* < 1$ → 지수 수렴 → 표현 붕괴

**Sheaf의 우월성**:

비대칭 제약 맵의 경우:

$$E_F(X^{(t+1)}) > E_F(X^{(t)})$$

아무리 작은 $W$에 대해서도 가능[2]
→ 에너지 증가 가능 → 깊은 네트워크 수용

### 6.4 조화 공간(Harmonic Space) 분석

**경로 독립성(Path Independence)**:

노드 $v$에서 $u$로의 transport:
$$P_γ: F(v) → F(u) = \prod_{e ∈ γ} F_{v⊴e}^T F_{u⊴e}$$

GCN(trivial sheaf): 항상 경로 독립 → $\dim(H^0) = 1$

Sheaf diffusion(비자명): 경로 의존 가능 → $\dim(H^0)$ 증가 가능

***

## 7. 한계 (Limitations)

### 원본 논문(2020)

1. **제한된 벤치마크**: 합성 부호 있는 그래프만 테스트
   - 실제 sheaf 구조가 명확한 그래프 데이터셋 부족
   
2. **Sheaf 학습 미제공**: 구조를 수동 설계해야 함
   
3. **계산 복잡도 분석 부재**

### 최근 연구(2022+)

1. **과적합 위험**: 제약 없는 sheaf 학습 시 문제
   - 해결책: 직교 제약, Bayesian 방식[4]
   
2. **하이퍼파라미터 민감성**:
   - 스톡 차원 $d$ 선택 중요
   - 제약 맵 유형(대각/직교/일반) 영향 큼
   
3. **확장성**: 일반 행렬은 SVD 필요 → 대규모 그래프 문제

***

## 8. 2020년 이후 관련 최신 연구 비교 분석

### 8.1 Neural Sheaf Diffusion (Bodnar et al., 2022) - 276인용

**핵심 혁신**:
- Sheaf를 데이터로부터 학습
- 이질성과 과평활화의 통일적 설명
- Cheeger 유형 스펙트럼 부등식[2]

**학습 메커니즘**:

$$F_{v⊴e:(v,u)} = Φ(x_v, x_u)$$

여기서 $Φ$는 MLP: $[x_v || x_u] → \mathbb{R}^{d×d}$ (또는 제약 버전)

**성능**: 이질성 데이터셋에서 SOTA

### 8.2 Attention-based SheafNN (2022)

**결합**:
$$α_{v,u} = \text{softmax}(a^T[x_v || x_u])$$
$$F_{v⊴e} = α_{v,u} × W$$

**결과**: 주의 메커니즘 + 위상 유연성

### 8.3 Joint Diffusion Processes as Inductive Bias (2024)

**의견 동역학 영감**:
- Sheaf를 합의 과정으로 모델링
- 이질성과 평활화 해결의 유도 편향 제공

### 8.4 Bayesian Sheaf Neural Networks (2024)

**변분 학습**:
$$p(F|X) ∝ p(X|F)p(F)$$

**SO(n) 상의 확률 분포** (Cayley 변환 사용):

$$R(v) = (I − \frac{v^∧}{2})(I + \frac{v^∧}{2})^{-1}$$

(여기서 $v^∧$는 skew-symmetric)

**이점**: 불확실성 정량화 → 과적합 저감

### 8.5 Cooperative SheafNN (2025)

**기여**:
- Long-range 상호작용 처리 (oversquashing 없이)
- 13+ 벤치마크에서 기존 SNN 능가

### 8.6 Directional Sheaf Hypergraph Networks (2025)

**확장**: 유향 하이퍼그래프로 일반화

$$\text{Directed Restriction Map: } F_{e ⊴ σ}: F(e) → F(σ)$$
(부분 σ 방향성)

**성능**: 7개 데이터셋에서 2-20% 상대 정확도 향상[5]

### 8.7 비교 요약표

| 방법 | 기여도 | 이질성 | 평활화 | 확장성 | 논문연도 |
|-----|------|-------|------|-------|--------|
| GCN | 기준 | 약함 | 문제 | 우수 | 2016 |
| SheafNN | 구조 일반화 | 우수 | 이론 | 중간 | 2020 |
| NSD | 학습 sheaf | 우수 | 우수 | 중간 | 2022 |
| Att-SheafNN | 주의 결합 | 우수 | 우수 | 중간 | 2022 |
| Bayesian SNN | 불확실성 | 우수 | 우수 | 중간 | 2024 |
| CSNN | Long-range | 우수 | 우수 | 중간 | 2025 |
| DSHN | 유향 하이퍼 | 우수 | 우수 | 중간 | 2025 |

***

## 9. 향후 연구에 미치는 영향 및 고려사항

### 9.1 이론적 영향

**1. 위상 기하학의 도입**
- 범주론, 호모로지 대수 도구의 GNN 적용
- 새로운 표현력 분석 프레임워크[2]

**2. 스펙트럼 이론 확장**
- 블록 대각 행렬의 고유값 분석
- Laplacian 스펙트럼 갭 특성화[2]

**3. 기하학적 해석**
- Parallel transport (미분 기하학)
- Vector bundle 개념의 이산화

### 9.2 실무 응용 확대

**추천 시스템**: Sheaf4Rec (2024)
- 비대칭 선호도 관계 캡처
- 각 노드를 단일 벡터 대신 벡터 공간으로 표현

**분자 모델링**: 화학 결합의 방향성 처리

**지식 그래프**: 다양한 관계 유형의 자연스러운 표현

### 9.3 향후 연구 시 고려사항

#### **1. Sheaf 학습의 안정성**
- 무제약 학습 → 과적합 위험
- **권장사항**: 
  - 직교 제약 기본 사용
  - 정규화 항 (예: Frobenius norm)
  - Bayesian 접근[4]

#### **2. 하이퍼파라미터 선택**
$$d^* = \arg\min_d \text{Val-Loss}$$

- 작은 $d$ (2-5): 대부분 충분
- 클래스 수보다는 작을 수 있음 (직교 효율성)

#### **3. 계산 복잡도 관리**

대각 제약 시:
$$\text{Complexity} = O(Nv·c^2 + m·d·c)$$
(GCN과 유사, 작은 $d$ 오버헤드)

일반 행렬:
$$\text{Complexity} = O(Nv(c^2 + d^3) + m(cd^2 + d^3))$$
→ $d$는 작아야 함

#### **4. 벤치마크 해석**
- 호모필리 데이터: GCN/GAT와 경쟁력 유지 필수
- 이질 데이터: SheafNN의 우월성 기대

#### **5. 이론적 분석 심화**
- Weisfeiler-Lehman 표현력 특성화
- Convergence rate 분석
- 일반화 경계(generalization bound) 도출

### 9.4 개방형 연구 문제

1. **더 높은 차수 구조**
   - Simplicial complex 및 CW complex로 확장
   - Copresheaf (쌍방향 구조)의 역할[6]

2. **Sheaf 구조의 자동 검색**
   - 데이터로부터 최적 sheaf 클래스 추정
   - NAS(신경 아키텍처 탐색) 통합

3. **이론-실제 간극**
   - 왜 작은 $d$가 실제로 충분한가?
   - Expressivity와 generalization의 균형

4. **적응적 학습**
   - 각 계층에서 sheaf를 학습할지 결정
   - Depth와 stalk dimension의 공동 최적화

5. **대규모 그래프**
   - 표본 기반 근사
   - 분산 학습(distributed learning)

***

## 10. 결론 (Conclusion)

**Sheaf Neural Networks**는 그래프 신경망의 기본 가정을 재검토하여, 대수-위상 수학의 도구를 통해 이질성, 비대칭성, 다차원 관계를 자연스럽게 처리하는 프레임워크를 제시했습니다.

### 핵심 성과
- ✅ **이론적**: Graph Laplacian → Sheaf Laplacian 일반화
- ✅ **실증적**: 이질 그래프에서 SOTA 성능 (최대 20% 향상)
- ✅ **확장성**: Cooperative SNN, 유향 하이퍼그래프 등으로 확장
- ✅ **적용**: 추천시스템, 분자 모델링 등 실무 응용

### 지속적 영향
- 2020-2025: 73 → 276+ 인용 (주요 논문)
- 다수의 후속 연구 분기
- 범주론적 ML 기초 마련

### 실무 적용 시 체크리스트
1. 데이터의 호모필리 수준 평가
2. 스톡 차원 $d$ 실험적 결정 ($d = 2$부터 시작)
3. 직교 제약 사용 권장 (과적합 방지)
4. 정규화와 early stopping 병행
5. 이질 벤치마크에서 성능 검증

***

## 출처
[1] 2012.06333v1.pdf https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/29099e5a-afad-4d80-813f-2e599fa98b1d/2012.06333v1.pdf
[2] Neural Sheaf Diffusion: A Topological Perspective on ... - NIPS https://papers.neurips.cc/paper_files/paper/2022/file/75c45fca2aa416ada062b26cc4fb7641-Paper-Conference.pdf
[3] Sheaves as a Framework for Understanding and Interpreting Model Fit https://arxiv.org/pdf/2105.10414.pdf
[4] Bayesian Sheaf Neural Networks https://arxiv.org/pdf/2410.09590.pdf
[5] Unifying Learning on Directed and Undirected Hypergraphs https://arxiv.org/abs/2510.04727
[6] Copresheaf Topological Neural Networks: A Generalized Deep ... https://arxiv.org/html/2505.21251v3
[7] Algebraic Topological Networks via the Persistent Local Homology Sheaf https://arxiv.org/pdf/2311.10156.pdf
[8] Joint Diffusion Processes as an Inductive Bias in Sheaf Neural Networks http://arxiv.org/pdf/2407.20597.pdf
[9] Sheaf theory: from deep geometry to deep learning https://arxiv.org/pdf/2502.15476.pdf
[10] Cooperative Sheaf Neural Networks https://arxiv.org/pdf/2507.00647.pdf
[11] Sheaf4Rec: Sheaf Neural Networks for Graph-based Recommender Systems https://arxiv.org/pdf/2304.09097.pdf
[12] Sheaf Neural Networks with Connection Laplacians https://arxiv.org/pdf/2206.08702.pdf
[13] Neural Sheaf Diffusion: A Topological Perspective on Heterophily and
  Oversmoothing in GNNs https://arxiv.org/pdf/2202.04579.pdf
[14] Generalization of Graph Neural Network Models for ... https://www.arxiv.org/pdf/2510.03571.pdf
[15] [PDF] Sheaf Neural Networks https://www.semanticscholar.org/paper/Sheaf-Neural-Networks-Hansen-Gebhart/c46425405ddd6b55385bd45f5ac8bbd51e361423
[16] Enhanced Pre-training of Graph Neural Networks for ... https://arxiv.org/html/2510.12401v1
[17] Investigating Out-of-Distribution Generalization of GNNs https://arxiv.org/html/2402.08228v2
[18] Cellular Sheaves on Higher-Dimensional Structures https://arxiv.org/abs/2505.23993
[19] Joint Diffusion Processes as an Inductive Bias in Sheaf ... https://arxiv.org/html/2407.20597v1
[20] Generalization of Graph Neural Network Models for ... https://arxiv.org/html/2510.03571v1
[21] Copresheaf Topological Neural Networks: A Generalized ... https://arxiv.org/html/2505.21251v1
[22] Unifying Learning on Directed and Undirected Hypergraphs https://arxiv.org/html/2510.04727v1
[23] Statistical physics analysis of graph neural networks https://arxiv.org/html/2503.01361v3
[24] [2506.02842] Sheaves Reloaded: A Directional Awakening https://arxiv.org/abs/2506.02842
[25] GRAPH CONVOLUTIONAL NETWORKS FROM THE PER https://proceedings.mlr.press/v196/gebhart22a/gebhart22a.pdf
[26] Graph Research Lab @ ANU - Publications https://graphlabanu.github.io/website/publications/
[27] PDE-inspired sheaf neural networks https://www.logml.ai/logml2022/projects2022/project21/
[28] Simple and Asymmetric Graph Contrastive Learning ... https://papers.neurips.cc/paper_files/paper/2023/file/3430bcc30cdaabd0bf6c5d0c31bda67c-Paper-Conference.pdf
[29] Neural Sheaf Diffusion: A Topological Perspective on ... https://iclr.cc/virtual/2022/9113
[30] Attention-based Sheaf Neural Networks https://www.mlmi.eng.cam.ac.uk/files/2021-2022_dissertations/attention-based-sheaf-neural-networks.pdf
[31] Asymmetric augmented paradigm-based graph neural ... https://www.sciencedirect.com/science/article/abs/pii/S0306457324002565
[32] Topological Deep Learning - Part 2: Sheaf Neural Networks https://www.sci.unich.it/geodeep2022/slides/GDL_SummerSchool_Part2.pdf
[33] Sheaf Neural Networks https://openreview.net/pdf?id=GgcgIJsT8HD
[34] [2012.06333] Sheaf Neural Networks https://arxiv.org/abs/2012.06333
[35] Simple and Asymmetric Graph Contrastive Learning ... https://openreview.net/forum?id=UK8mA3DRnb&noteId=0POENE2MzY
[36] Topological Neural Networks https://www.emergentmind.com/topics/topological-neural-networks
[37] The Role of Artificial Intelligence in Early Disease Detection: Current Applications and Future Prospects https://gjeac.com/index.php/GJEAIC/article/view/1
[38] Model Data-Driven untuk Prediksi Digitalisasi UMKM Menggunakan GMM dan XGBoost https://jurnal.pustakagalerimandiri.co.id/index.php/pustakaai/article/view/984
[39] Abstract 7426: Leveraging deep learning to enable precision medicine via the IMPROVE benchmarking framework https://aacrjournals.org/cancerres/article/85/8_Supplement_1/7426/759414/Abstract-7426-Leveraging-deep-learning-to-enable
[40] Application of Denoising Diffusion Probabilistic Model for Synthesizing Composite Electrode Microstructures for Solid Oxide Cells https://iopscience.iop.org/article/10.1149/MA2025-031244mtgabs
[41] SIGMA: An Efficient Heterophilous Graph Neural Network with Fast Global
  Aggregation https://arxiv.org/abs/2305.09958v1
[42] Are Heterophily-Specific GNNs and Homophily Metrics Really Effective?
  Evaluation Pitfalls and New Benchmarks http://arxiv.org/pdf/2409.05755.pdf
[43] Graph Neural Networks for Graphs with Heterophily: A Survey http://arxiv.org/pdf/2202.07082v1.pdf
[44] Finding Global Homophily in Graph Neural Networks When Meeting
  Heterophily https://arxiv.org/pdf/2205.07308.pdf
[45] Learn from Heterophily: Heterophilous Information-enhanced Graph Neural
  Network http://arxiv.org/pdf/2403.17351.pdf
[46] The Heterophilic Graph Learning Handbook: Benchmarks, Models,
  Theoretical Analysis, Applications and Challenges https://arxiv.org/html/2407.09618v1
[47] GCN-SL: Graph Convolutional Networks with Structure Learning for Graphs
  under Heterophily https://arxiv.org/pdf/2105.13795.pdf
[48] Revisiting the Message Passing in Heterophilous Graph Neural Networks http://arxiv.org/pdf/2405.17768.pdf
[49] Understanding Heterophily for Graph Neural Networks https://arxiv.org/html/2401.09125v2
[50] Solving Oversmoothing in GNNs via Nonlocal Message ... https://arxiv.org/pdf/2512.08475.pdf
[51] The Expressive Power of Graph Neural Networks: A Survey https://arxiv.org/pdf/2308.08235.pdf
[52] Understanding Heterophily for Graph Neural Networks https://arxiv.org/pdf/2401.09125.pdf
[53] ADMP-GNN: Adaptive Depth Message Passing GNN https://arxiv.org/html/2509.01170v1
[54] [2408.05486] Topological Blindspots: Understanding and ... https://arxiv.org/abs/2408.05486
[55] A critical look at the evaluation of GNNs under heterophily https://arxiv.org/html/2302.11640v2
[56] Adaptive Message Passing: A General Framework to ... https://arxiv.org/html/2312.16560v3
[57] The Expressive Power of Graph Neural Networks: A Survey https://arxiv.org/html/2308.08235v2
[58] HeTGB: A Comprehensive Benchmark for Heterophilic Text ... https://arxiv.org/html/2503.04822v1
[59] Understanding Oversmoothing in GNNs as Consensus ... https://arxiv.org/html/2501.19089v1
[60] On the Expressivity of Persistent Homology in Graph ... https://arxiv.org/abs/2302.09826
[61] Re-evaluating the Advancements of Heterophilic Graph ... https://arxiv.org/html/2409.05755v2
[62] Solving Over-Smoothing in GNNs via Nonlocal Message Passing https://arxiv.org/html/2512.08475v1
[63] An Algebraic Platform for Crafting Topology-Aware GNNs https://arxiv.org/html/2412.08835v1
[64] Revisiting Heterophily For Graph Neural Networks https://papers.neurips.cc/paper_files/paper/2022/file/092359ce5cf60a80e882378944bf1be4-Paper-Conference.pdf
[65] Does Depth Really Hurt GNNs? Injective Message Passing ... https://openreview.net/pdf?id=9xto7V7oqi
[66] [R] Graph Neural Networks through the lens of Differential Geometry and Algebraic Topology https://www.reddit.com/r/MachineLearning/comments/qwt9ni/r_graph_neural_networks_through_the_lens_of/
[67] Techniques to Mitigate GNN Oversmoothing https://apxml.com/courses/graph-neural-networks-gnns/chapter-3-gnn-training-complexities/mitigating-oversmoothing
[68] On the Expressive Power of Geometric Graph Neural Networks https://proceedings.mlr.press/v202/joshi23a/joshi23a.pdf
[69] Heterophily and Graph Neural Networks: Past, Present and ... https://par.nsf.gov/servlets/purl/10435541
[70] Demystifying Oversmoothing in Attention-Based Graph ... https://proceedings.neurips.cc/paper_files/paper/2023/file/6e4cdfdd909ea4e34bfc85a12774cba0-Paper-Conference.pdf
[71] Heterophily and Graph Neural Networks: https://www.jiongzhu.net/assets/files/zhu2023heterophily.pdf
[72] Graph Geometric Algebra networks for ... https://www.nature.com/articles/s41598-024-84483-0
[73] Under review as a conference paper at ICLR 2024 https://openreview.net/pdf?id=ctXZJLBbyb
[74] On Over-Squashing in Message Passing Neural Networks https://liner.com/ko/review/on-oversquashing-in-message-passing-neural-networks-impact-width-depth
[75] A Theoretical Study of Neural Network Expressive Power ... https://openreview.net/forum?id=L7gyAKWpiM
