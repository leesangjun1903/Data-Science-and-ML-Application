
# Simplicial 2-Complex Convolutional Neural Networks

**저자:** Eric Bunch, Qian You, Glenn Fung, Vikas Singh  
**발표:** NeurIPS 2020 Workshop — Topological Data Analysis and Beyond  
**arXiv:** 2012.06010 (2020.12.10)

---

## 1. 핵심 주장 및 주요 기여 요약

최근 그래프 또는 하이퍼그래프 구조의 데이터를 다루는 신경망이 개발되었으나, 그래프 구조는 쌍별(pairwise) 관계에 제한되며, 하이퍼그래프는 하이퍼엣지 간 고차 관계를 고려하지 못한다. 이에 대한 중간 지점으로 풍부한 이론적 기반을 가진 **단체 복합체(simplicial complex)** 위에서의 합성곱 신경망 레이어를 개발하였다.

**핵심 기여:**
1. **Simplicial 2-complex 위에서의 CNN 레이어 정의**: Bunch et al. (2020)은 Schaub et al. (2020)의 정규화 방법을 따라 boundary map과 Hodge Laplacian의 정규화된 버전을 사용하여, 단체 복합체의 모든 수준(0-chain, 1-chain, 2-chain)에서 동시에 표현을 유지하는 합성곱 레이어 아키텍처를 제안하였다.
2. **MNIST 데이터셋을 통한 개념 증명(proof of concept)**: MNIST 손글씨 숫자 분류에서 단체 복합체 합성곱 레이어를 사용하였으며, 기존 전통적 합성곱 레이어를 보강(augment)할 수 있음을 절삭 연구(ablation study)를 통해 상세히 보여주었다.

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

지난 10년간 신경망은 그래프 또는 하이퍼그래프 구조의 데이터에 적용되어 왔으나, 그래프 구조는 노드 간 쌍별 연결에 제한되고, 하이퍼그래프는 에지 사이의 고차 관계를 포착하지 못한다. 단체 복합체는 그래프와 하이퍼그래프 사이의 중간 지점으로, 풍부한 이론을 활용할 수 있다.

구체적으로, 기존 GNN이 **삼각형(triangle)** 이나 **클리크(clique)** 같은 고차 구조를 직접적으로 학습에 반영하지 못하는 한계를 극복하고자 했습니다.

### 2.2 제안하는 방법 (수식 포함)

#### (a) 단체 복합체(Simplicial Complex) 정의

단체 복합체 $S$는 유한 집합 $X$의 부분집합들의 모음으로, 부분 집합에 대해 닫혀 있는 구조이다. $S$를 구성하는 유한 집합을 **면(face)**이라 하며, 면 $s$의 차원은 $\dim(s) = |s| - 1$로 정의된다. 0차원 면은 꼭짓점(vertex), $v_i$로 표기한다.

$k$-chain의 모음을 $C_k$로 표기하며, 각 $C_k$는 $\mathbb{R}$ 위의 유한 차원 벡터 공간을 이룬다.

#### (b) 경계 연산자(Boundary Operator) 및 Hodge Laplacian

각 차원 $k$에 대해 경계 연산자(boundary operator) $\partial_k : C_k \to C_{k-1}$이 정의되며, 이를 행렬로 표현하면 $B_k$ ($n_{k-1} \times n_k$ 행렬)입니다:

$$
\partial_k(v_0, v_1, \ldots, v_{k}) = \sum_{i=0}^{k} (-1)^{i} (v_0, \ldots, \hat{v}_i, \ldots, v_{k})
$$

여기서 $\hat{v}_i$는 $v_i$를 제거한다는 의미입니다.

**Hodge $k$-Laplacian** $L_k$는 다음과 같이 정의됩니다:

$$
L_k = B_k^\top B_k + B_{k+1} B_{k+1}^\top = L_k^{\text{down}} + L_k^{\text{up}}
$$

그래프 라플라시안과 마찬가지로, $L_k$는 $S$ 위의 신호를 적절히 전파하는 방법을 기술한다.

논문에서는 Schaub et al. [10]의 방법에 따라 $L_k$를 정규화하여 사용하였으며, 구현의 용이성과 실용성을 이유로 이 방법을 선택했다.

#### (c) 합성곱 레이어 정의

Bunch et al. (2020)의 모델에서, 수평 화살표는 정규화된 Hodge Laplacian의 적용에 해당하고, 대각선 화살표는 정규화된 경계(boundary) 또는 쌍대경계(coboundary) 맵의 적용에 해당한다. 각 노드에서 입력들은 합산된 후 활성화 함수 $\phi$를 통과한다.

Ebli et al. (2020)과 SCoNe와 달리, 단체 복합체의 **모든 수준**에서 동시에 표현(representation)을 유지하는 아키텍처를 제안한다. 정규화 세부사항을 생략하면, 합성곱 레이어는 다음의 형태를 취합니다:

$$
X_0^{(h+1)} = \phi\!\left(\tilde{L}_0 \, X_0^{(h)} W_{00}^{(h)} \;+\; \tilde{B}_1^\top \, X_1^{(h)} W_{10}^{(h)}\right)
$$

$$
X_1^{(h+1)} = \phi\!\left(\tilde{B}_1 \, X_0^{(h)} W_{01}^{(h)} \;+\; \tilde{L}_1 \, X_1^{(h)} W_{11}^{(h)} \;+\; \tilde{B}_2^\top \, X_2^{(h)} W_{21}^{(h)}\right)
$$

$$
X_2^{(h+1)} = \phi\!\left(\tilde{B}_2 \, X_1^{(h)} W_{12}^{(h)} \;+\; \tilde{L}_2 \, X_2^{(h)} W_{22}^{(h)}\right)
$$

여기서:
- $X_k^{(h)}$: $h$번째 은닉층에서의 $k$-chain 특징 행렬
- $\tilde{L}_k$: 정규화된 Hodge $k$-Laplacian
- $\tilde{B}_k$: 정규화된 경계 행렬
- $W_{ij}^{(h)}$: 학습 가능한 가중치 행렬
- $\phi$: 원소별 활성화 함수

$k=0$인 경우, 이 정의는 그래프 합성곱 네트워크(GCN)에서의 합성곱 레이어 정의와 일치한다.

#### (d) 이미지에서 단체 복합체 구성 (MNIST 적용)

$h \times w$ 픽셀 이미지에서 커널 크기 $k$와 스텝 크기 $s$를 선택하여 0-면(zero face)의 격자를 형성하고, 수평·수직·대각선으로 인접한 0-면을 1-면으로 연결한다. 쌍별 연결된 세 0-면의 삼중항이 있으면 2-면을 추가한다.

0-면에 부착된 특징 벡터는 해당 0-면에 대응하는 $k \times k$ 픽셀값을 펼쳐 $k^2 \times 1$ 벡터로 만든 것이다.

### 2.3 모델 구조

모델 구조를 도식화하면:

```
     X₀⁽ʰ⁾ --------[L̃₀]-------→ X₀⁽ʰ⁺¹⁾
       ↕ B̃₁                  ↕ B̃₁ᵀ
     X₁⁽ʰ⁾ --------[L̃₁]-------→ X₁⁽ʰ⁺¹⁾  
       ↕ B̃₂                  ↕ B̃₂ᵀ
     X₂⁽ʰ⁾ --------[L̃₂]-------→ X₂⁽ʰ⁺¹⁾
```

- **수평 경로** ($\tilde{L}_k$): 같은 차원의 단체(simplex) 간 정보 전파 (예: 엣지↔엣지)
- **대각선 경로** ($\tilde{B}_k$, $\tilde{B}_k^\top$): 차원 간 정보 전파 (예: 노드↔엣지, 엣지↔삼각형)

### 2.4 성능 향상

MNIST 데이터셋에서 손글씨 숫자 분류 실험을 수행했으며, 단체 복합체 합성곱 레이어가 전통적 합성곱 레이어를 **보강(augment)**할 수 있음을 절삭 연구를 통해 입증했다.

이는 개념 증명(proof of concept) 수준의 결과이며, 단체 복합체 기반 합성곱이 위상적 정보를 활용하여 기존 CNN에 추가적인 학습 정보를 제공할 수 있음을 보여줍니다.

### 2.5 한계

1. **Orientation Equivariance 미충족**: 원 저자들은 경험적 연구에서 Leaky ReLU 활성화 함수를 사용했는데, 이는 홀수 함수(odd function)가 아니므로 방향 등변성(orientation equivariance)을 만족하지 못한다.

2. **Admissibility 조건**: 이러한 레이어들로 구성된 합성곱 신경망은, 활성화 함수 $\phi$가 홀수(odd)일 때에만 admissible하다.

3. 엣지 분류나 단체 복합체 자체 분류 등의 일반적 태스크, 특히 구조가 서로 다른 단체 복합체 집합을 다룰 때 추가적인 방법이 필요하다.

4. **확장성 문제**: 2-complex 이상의 고차 단체로의 확장, 대규모 데이터셋에 대한 실험이 제한적입니다.

5. 합성곱 접근법에 의한 계산이 복합체의 조합론적 구조에 강하게 결합되어 있어, 이전에 보지 못한 단체 구조에 대한 일반화가 저해될 수 있다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 이론적 기반

Hodge Laplacian은 다음과 같은 **Hodge 분해(Hodge Decomposition)**를 유도합니다:

$$
C_k = \text{im}(B_{k+1}) \oplus \text{im}(B_k^\top) \oplus \ker(L_k)
$$

이는 $k$-chain 공간이 세 가지 직교 부분공간 — **gradient**, **curl**, **harmonic** — 으로 분해됨을 의미합니다. 이 분해 덕분에 모델은 단순 그래프에서는 구분 불가능한 위상적 특징(topological features)을 학습할 수 있어 일반화 성능의 잠재적 향상 근거가 됩니다.

### 3.2 GCN과의 관계 및 확장

$k=0$ 수준에서 $L_0 = B_1 B_1^\top$이고, 이는 그래프 라플라시안과 일치합니다. 따라서:

$$
X_0^{(h+1)} = \phi\!\left(\tilde{L}_0 \, X_0^{(h)} W^{(h)}\right)
$$

는 Kipf & Welling (2017)의 GCN 레이어와 본질적으로 동일합니다. S2C-CNN은 이를 **1-chain과 2-chain 수준으로 확장**하여 고차 상호작용을 포착합니다.

### 3.3 표현력(Expressivity)과 일반화

Bodnar et al. (2021)은 MPSN이 Ebli et al. (2020)과 Bunch et al. (2020)이 정의한 특정 스펙트럼 합성곱 연산자를 일반화함을 보였다.

$p$차원 단체 복합체에서 $n$-단체가 $S_n$개($n = 0, 1, \ldots, p$)인 경우, 각 $M_n$이 가역(invertible)이면, ReLU SCNN 레이어가 표현할 수 있는 함수의 선형 영역(linear regions) 수의 최적 상한을 이론적으로 도출할 수 있다.

MPSN의 선형 영역 수 $R_{\text{MPSN}} \geq R_{\text{SCNN}}$이 성립하여, 메시지 패싱 방식이 스펙트럼 합성곱 방식보다 표현력이 우월함이 증명되었다.

### 3.4 일반화 향상을 위한 방향

| 전략 | 설명 |
|------|------|
| **홀수 활성화 함수 사용** | $\tanh$, odd-ReLU 등으로 orientation equivariance 확보 |
| **어텐션 메커니즘 도입** | 고정된 라플라시안 대신 학습된 가중치로 적응적 메시지 전달 |
| **Hodge 분해 활용** | gradient/curl/harmonic 성분을 독립적으로 필터링 |
| **정규화 및 드롭아웃** | over-fitting, over-smoothing 방지 |
| **Cross-rank 상호작용 강화** | 다양한 차원 간 메시지 패싱 확대 |

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4.1 연구에 미친 영향

이 논문은 **Topological Deep Learning (TDL)** 분야의 초기 핵심 연구 중 하나로서:

1. Bunch et al. (2020)을 포함한 연구들이 이러한 위상적 도메인 위에서 학습하는 머신러닝 모델 설계의 기반을 마련하였다.

2. TDL 아키텍처 서베이에서 SCCONV(Bunch et al., 2020)는 단체 복합체 기반 분류(classification) 모델로 분류된다.

3. 이 연구의 한계를 극복하기 위해 MPSN이 제안되었으며, Simplicial Weisfeiler-Lehman(SWL) 채색 절차를 도입하여 SWL과 MPSN이 WL 테스트보다 엄격히 강력하고 3-WL 테스트보다 약하지 않음을 보였다.

### 4.2 향후 연구 시 고려할 점

1. **확장성(Scalability)**: 대규모 복합체에서 Hodge Laplacian 계산의 시간/공간 복잡도
2. **동적 위상(Dynamic Topology)**: 시간에 따라 변하는 단체 구조 처리
3. **이론적 보장**: 표현력, 수렴성, over-smoothing에 대한 엄격한 이론적 분석
4. gradient vanishing, over-smoothing, over-fitting 문제로 인해 기존 단체 신경망은 일반적으로 매우 얕은 모델에 제한된다.
5. 단체 복합체의 엄격한 기하학적 제약이 데이터에 의해 존중되지 않을 경우 가짜 연결(spurious connection)이 도입될 위험이 있으며, Cellular Complex는 이를 일반화하여 면이 세 노드 이상을 포함할 수 있다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

아래 표는 S2C-CNN 이후 등장한 주요 단체 복합체/위상 기반 신경망 연구들을 비교합니다:

| 연구 | 연도 | 주요 특징 | S2C-CNN과의 차이점 |
|------|------|-----------|-------------------|
| **SNN** (Ebli et al.) | 2020 | 단체 복합체 위의 GNN 일반화; 적절한 합성곱 개념 정의 | 단일 $k$-chain 수준에서만 동작; S2C-CNN은 모든 수준 동시 처리 |
| **SCoNe** (Roddenberry et al.) | 2021 | 대수적 위상에 기반한 궤적 예측 아키텍처; ICML 2021 | 홀수 비선형 활성화 함수 요구를 이론적으로 정당화 |
| **MPSN** (Bodnar et al.) | 2021 | 단체 복합체 위에서 메시지 패싱을 수행하며 SWL 채색 절차를 도입; GNN이 실패하는 강정규 그래프를 구분 가능 | 스펙트럼 접근 → 공간적 메시지 패싱; 이론적 표현력 보장 |
| **CW Networks** (Bodnar et al.) | 2021 | SC의 엄격한 조합론적 구조를 넘어 regular Cell Complex로 확장; SC와 그래프를 유연하게 포괄 | SC의 제약(삼각형만 허용)을 완화 |
| **BScNets** (Chen et al.) | 2022 | 기존 GCN 프레임워크를 체계적으로 일반화하여 여러 차원의 고차 그래프 구조 간 상호작용을 통합; 상당한 마진으로 SOTA 능가 | Block Hodge Laplacian으로 다수준 구조 동시 학습 |
| **SAT** (Goh et al.) | 2022 | 합성곱 접근의 일반화 한계를 극복하기 위해 어텐션 메커니즘 도입; GAT를 단체 복합체로 일반화 | 고정 Laplacian 대신 학습된 어텐션 가중치 사용 |
| **SCNN** (Yang et al.) | 2022 | Hodge 분해를 단체 합성곱에 적용하여 SNN의 성능을 개선 | 상하 이웃(upper/lower neighborhood)을 독립적으로 필터링 |
| **GSAN** (Battiloro et al.) | 2023 | 마스크된 셀프-어텐션 레이어를 활용하여 단체 복합체 데이터를 처리; 순열 등변성 및 단체 인식 속성 증명 | 어텐션 + 이론적 속성 보장의 결합 |
| **E(n)-EMPSN** (Eijkelboom et al.) | 2023 | E(n) 등변 메시지 패싱 단체 네트워크; ICML 2023 | 물리적 대칭(등변성) 보장 |
| **DeepSCNN** (Tang et al.) | 2024 | gradient vanishing, over-smoothing, over-fitting 문제를 해결하기 위한 깊은 단체 합성곱 네트워크; 단체 엣지 샘플링(SES) 기술 도입 | 깊은 네트워크 학습 가능 |
| **Bi-SCNN** (Yan & Kuruoglu) | 2024 | 가중 이진 부호 순전파와 단체 합성곱을 결합; 모델 복잡도 감소 및 실행 시간 단축, over-smoothing 완화 | 효율성 중심; 이진화로 경량화 |

### 연구 발전의 전체적 흐름

```
S2C-CNN / SNN (2020)  ─→  SCoNe (2021) / MPSN (2021)  ─→  CW Networks (2021)
       │                            │                            │
       └───→ BScNets (2022) / SAT (2022) / SCNN (2022)          │
                        │                                         │
                        └───→ GSAN (2023) / E(n)-EMPSN (2023)   │
                                     │                            │
                                     └───→ DeepSCNN / Bi-SCNN / TopoX (2024-2025)
```

TDL에서 다자간 상호작용(multi-way interactions)은 위상적 구조를 딥러닝으로 임베딩할 수 있는 유용한 특징을 구성하며, 고차 관계는 효과적이거나 강건한 메시지 패싱 스키마의 범위를 제공한다. TDL은 단체 및 셀 복합체 등 다양한 위상적 도메인에서 동작하여, 관계형 데이터에 나타나는 다양한 유형의 다자간 상호작용을 모델링하는 프레임워크를 제공한다.

---

## 참고 자료

1. **Bunch, E., You, Q., Fung, G., Singh, V.** (2020). "Simplicial 2-Complex Convolutional Neural Nets." *NeurIPS 2020 Workshop on TDA and Beyond.* arXiv:2012.06010. [OpenReview](https://openreview.net/forum?id=TLbnsKrt6J-), [GitHub](https://github.com/AmFamMLTeam/simplicial-2-complex-cnns)
2. **Ebli, S., Defferrard, M., Spreemann, G.** (2020). "Simplicial Neural Networks." *NeurIPS 2020 Workshop on TDA and Beyond.* arXiv:2010.03633.
3. **Roddenberry, T. M., Glaze, N., Segarra, S.** (2021). "Principled Simplicial Neural Networks for Trajectory Prediction." *ICML 2021.* arXiv:2102.10058.
4. **Bodnar, C., Frasca, F., Wang, Y., et al.** (2021). "Weisfeiler and Lehman Go Topological: Message Passing Simplicial Networks." *ICML 2021.* arXiv:2103.03212.
5. **Bodnar, C., Frasca, F., Otter, N., et al.** (2021). "Weisfeiler and Lehman Go Cellular: CW Networks." *NeurIPS 2021.*
6. **Chen, Y., Gel, Y. R., Poor, H. V.** (2022). "BScNets: Block Simplicial Complex Neural Networks." *AAAI 2022.*
7. **Goh, C. W. J., Bodnar, C., Liò, P.** (2022). "Simplicial Attention Networks." *ICLR 2022 Workshop.*
8. **Yang, M., Isufi, E., Leus, G.** (2022). "Simplicial Convolutional Neural Networks." *ICASSP 2022.*
9. **Battiloro, C., Testa, L., Giusti, L., et al.** (2023). "Generalized Simplicial Attention Neural Networks." arXiv.
10. **Eijkelboom, F., Hesselink, R., Bekkers, E.** (2023). "E(n) Equivariant Message Passing Simplicial Networks." *ICML 2023.*
11. **Tang, C. et al.** (2024). "DeepSCNN: A Simplicial Convolutional Neural Network for Deep Learning." *Applied Intelligence*, Springer.
12. **Yan, Y., Kuruoglu, E. E.** (2024). "Binarized Simplicial Convolutional Neural Networks." *Neural Networks*, Elsevier.
13. **Papillon, M. et al.** (2024). "Architectures of Topological Deep Learning: A Survey of Message-Passing Topological Neural Networks." arXiv:2304.10031.
14. **Hajij, M. et al.** (2024). "Position: Topological Deep Learning is the New Frontier for Relational Learning." *ICML 2024.* PMC 11973457.
15. **Schaub, M. T., Benson, A. R., Horn, P., et al.** (2020). "Random Walks on Simplicial Complexes and the Normalized Hodge 1-Laplacian." *SIAM Review*, 62(2):353-391.

<details>

# Simplicial 2-Complex Convolutional Neural Networks

아래에서는 편의상 이 논문을 **S2CCNN**(Simplicial 2-Complex Convolutional Neural Networks)로 부르겠습니다. 참고로 arXiv 제목은 *“Simplicial 2-Complex Convolutional Neural Nets”*, OpenReview 제목은 *“Simplicial 2-Complex Convolutional Neural Networks”*로 표기가 조금 다릅니다. 또한 이 논문은 **NeurIPS 2020의 TDA and Beyond 워크숍** 채택 논문이며, 분량이 5쪽인 **짧은 워크숍 논문**입니다. 따라서 아래 평가는 원문 주장과 후속 연구를 함께 보며, 과도한 일반화는 피해서 정리하겠습니다. 

## 1. 핵심 주장과 주요 기여 요약

**핵심 주장**은 간단합니다. 그래프 신경망은 노드-엣지의 **쌍대(pairwise) 관계**에는 강하지만, 삼각형(2-simplex)까지 포함되는 **고차 상호작용**을 직접 다루기에는 구조적으로 제한적입니다. S2CCNN은 이를 해결하기 위해 **0-단체(노드), 1-단체(엣지), 2-단체(삼각형)** 위의 특징을 동시에 다루는 **2차원 simplicial complex용 convolution layer**를 제안합니다. 

이 논문의 주요 기여는 다음 네 가지로 요약할 수 있습니다.  
1) **2-complex 전용 합성곱 계층**을 제안해 노드/엣지/삼각형 특징을 함께 업데이트한다.  
2) 경계행렬 \(B_1, B_2\)와 Hodge Laplacian 계열 정규화를 사용해 **같은 차원 내 전파**와 **차원 간 전파**를 모두 설계한다.  
3) 각 simplex 차원별로 다른 특징 차원 \(F_k\)를 허용해 **멀티모달 표현**이 가능하다고 본다.  
4) MNIST를 simplicial 2-complex로 인코딩한 소규모 실험에서, 단순 CNN/그래프 합성곱 대비 **정확도 향상**을 보이며 고차 구조의 잠재력을 시연한다. 

---

## 2. 논문 상세 설명

### 2-1. 해결하고자 하는 문제

논문이 겨냥한 문제는 다음입니다.

- **그래프는 pairwise 관계만 표현**한다.
- **하이퍼그래프는 고차 관계를 표현**할 수 있지만, 하이퍼엣지들 사이의 위상적 관계를 충분히 구조화하지 못한다.
- 따라서 노드-엣지-삼각형이 닫힘성(closed under subsets)을 만족하는 **simplicial complex**가 그래프와 하이퍼그래프의 중간 지점이자, 더 풍부한 수학적 도구를 제공한다고 본다. 

즉, 이 논문은 “**고차 상호작용을 가진 데이터를 어떻게 CNN/GNN처럼 학습할 것인가?**”를 2차 simplicial complex 위에서 푸는 초기 시도입니다. 특히 후속 연구 관점에서 보면, 이 논문은 이후의 simplicial/message-passing/attention 계열 모델들이 발전하는 출발점 중 하나로 작동했습니다. 

---

### 2-2. 제안하는 방법: 수식 중심 설명

#### (a) 기본 위상 연산자

simplicial complex의 \(k\)-chain 공간에서 경계 연산자 \(\partial_k\)의 행렬표현을 \(B_k\)라 두면, 논문은 \(k\)-차 Hodge Laplacian을 다음처럼 둡니다.

$$
L_k = B_k^\top B_k + B_{k+1} B_{k+1}^\top .
$$

이 식은 그래프 Laplacian의 고차 일반화로 볼 수 있으며, \(k\)-simplex 위의 신호가 어떻게 전파되어야 하는지를 정하는 핵심 연산자입니다. 특히 \(k=0\)이면 일반 그래프 Laplacian과 연결됩니다. 

또한 2-complex \(S\)에서 각 차원별 특징행렬을

$$
X_k \in \mathbb{R}^{n_k \times F_k}, \qquad k \in \{0,1,2\}
$$

로 둡니다. 여기서 \(n_k = |S_k|\)는 \(k\)-simplex의 수이고, \(F_k\)는 그 차원의 특징 차원입니다. 논문은 \(F_k\)가 \(k\)마다 달라도 되므로 **차원별 이질적 특징**을 수용할 수 있다고 설명합니다. 

#### (b) 정규화와 adjacency-like 연산자

논문은 normalized Hodge Laplacian 계열 아이디어를 따라 여러 degree 행렬 \(D_i\)와, 상향/하향 인접성을 나타내는 adjacency-like 행렬 \(A_0^u, A_1^u, A_1^d, A_2^d\), 그리고 self-loop를 포함한 정규화 버전 \(\tilde A_0^u, \tilde A_1^u, \tilde A_1^d, \tilde A_2^d\)를 정의합니다. 직관적으로는

- \(\uparrow\): 공상위 simplex(coface)를 공유하는 **상위 인접성**
- \(\downarrow\): 공하위 simplex(face)를 공유하는 **하위 인접성**

을 반영한 연산자입니다. 논문의 핵심은 이 연산자들로 그래프의 normalized adjacency를 고차 구조로 일반화하는 데 있습니다. 

#### (c) 핵심 업데이트 식

논문이 제안한 한 층의 업데이트는 다음과 같습니다.

$$
X_0^{(h+1)}
=
\sigma\!\left(
D_1^{-1} B_1 X_1^{(h)} W_{0,1}^{(h)}
+
\tilde A_{\uparrow,0} X_0^{(h)} W_{0,0}^{(h)}
\right),
$$

$$
X_1^{(h+1)}
=
\sigma\!\left(
B_2 D_3 X_2^{(h)} W_{1,2}^{(h)}
+
(\tilde A_{\downarrow,1}+\tilde A_{\uparrow,1}) X_1^{(h)} W_{1,1}^{(h)}
+
D_2 B_1^\top D_1^{-1} X_0^{(h)} W_{1,0}^{(h)}
\right),
$$

$`X_2^{(h+1)}=
\sigma\!\left(
\tilde A_{\downarrow,2} X_2^{(h)} W_{2,2}^{(h)}
+
D_4 B_2^\top D_5^{-1} X_1^{(h)} W_{2,1}^{(h)}
\right).`$

여기서 \(\sigma\)는 비선형 활성함수이고, \(W_{i,j}^{(h)}\)는 차원 \(j\)의 정보를 차원 \(i\)로 보내기 위한 학습 파라미터입니다. 이 수식의 의미는 명확합니다.  
- **같은 차원 안에서는** \(\tilde A\)를 통해 확산(convolution)하고,  
- **인접 차원 사이에서는** \(B_k\) 또는 \(B_k^\top\)를 통해 incidence 기반 메시지를 주고받습니다.  
즉, 단순한 “노드 이웃 평균”이 아니라, **노드↔엣지↔삼각형**을 오가는 교차-차원 합성곱을 수행합니다. 

---

### 2-3. 모델 구조

이 논문은 이론적으로는 일반 2-complex convolution layer를 제안하지만, 실험은 **MNIST 분류용 proof-of-concept**에 가깝습니다. 이미지 \(h \times w\)에 대해 커널 크기 \(k\), stride \(s\)를 정하고, 각 지역 패치를 **0-face**로 두며, 수평/수직/대각 인접 패치 사이에 **1-face**, 그리고 가능한 경우 삼각 연결에 **2-face**를 추가해 이미지를 2-complex로 바꿉니다. 노드 특징은 해당 \(k \times k\) 패치를 펼친 벡터이고, 엣지/삼각형 특징은 값 1의 스칼라로 둡니다. 

실험 구조는 세 가지 비교입니다.  
- **기본선**: 전통적 convolution layer + 1-hidden-layer FC  
- **그래프 변형**: 앞단에 GCN 계층 추가  
- **simplicial 변형**: 앞단에 S2CCNN 계층 추가  

즉, 이 논문은 “S2CCNN만으로 끝까지 학습”하기보다, **기존 CNN 앞단에 고차 구조 layer를 prepend**하는 방식으로 유용성을 보여줍니다. 

---

### 2-4. 성능 향상

MNIST의 저자 실험에서 평균 정확도는 다음과 같습니다.

- \(\text{CONV1D-FC}: 87.40 \pm 0.70\)
- \(\text{GCONV-CONV1D-FC}: 87.42 \pm 0.59\)
- \(\text{SCCONV-CONV1D-FC}: 91.10 \pm 0.40\)

즉, S2CCNN을 붙인 모델이 기본선 대비 약 **3.70 percentage points**, 그래프 합성곱 대비 약 **3.68 percentage points** 개선되었습니다. 저자도 이 결과를 “state-of-the-art”가 아니라 **further study를 정당화하는 정도의 시사적 결과**로 해석합니다. 

이 결과가 의미 있는 이유는, 동일한 데이터에서 **그래프화만 한 것보다 삼각형까지 포함한 simplicial lifting**이 더 나았다는 점입니다. 즉, 문제에 따라서는 **2-simplex 정보가 일반화에 실제로 기여**할 수 있다는 초기 증거입니다. 다만 실험 규모가 작아서 강한 결론으로 읽으면 안 됩니다. 

---

### 2-5. 한계

이 논문의 한계는 비교적 분명합니다.

1. **2-complex까지만 다룸**: 노드-엣지-삼각형까지만 다루며, 더 높은 차원 simplicial complex로 일반화된 설계는 직접 제시하지 않습니다. 후속 연구는 이 모델을 “order-2에 제한된 특별한 경우”로 해석합니다. 

2. **실험이 매우 제한적**: MNIST 소규모 실험(학습 1000장, 테스트 1000장, 5회 반복) 중심이며, 본격적인 고차 관계 데이터셋이나 여러 downstream task 검증이 없습니다. 저자 스스로도 결과가 SOTA가 아니며 더 많은 조사가 필요하다고 적습니다. 

3. **일반화 이론 부족**: permutation/orientation equivariance, stability, oversmoothing, perturbation robustness 같은 이론적 분석이 거의 없습니다. 이후 연구들이 바로 이 지점을 보강합니다. 

4. **후속 연구 시각에서 보면 표현력이 낮은 축에 속함**: 2025년 Hodge-aware 연구는 S2CCNN을 **one-step shifting만 사용하는 SCCNN의 특수형**으로 해석하며, lower/upper 기여를 분리하지 않고 단순 합산하는 구조라고 설명합니다. 이는 구현은 단순하지만, 더 정교한 주파수 제어·다중 홉·Hodge subspace 분해 능력은 제한될 수 있음을 시사합니다. 

---

## 3. 특히 중요한 쟁점: 모델의 일반화 성능 향상 가능성

### 3-1. 이 논문이 일반화에 기여할 수 있는 이유

S2CCNN의 가장 큰 일반화 잠재력은 **맞는 구조 편향(inductive bias)**에 있습니다. 그래프만 쓰면 사라지는 삼각형/면 정보까지 모델에 직접 주입하므로, 문제의 진짜 생성 메커니즘이 고차 상호작용에 의존할수록 더 적은 데이터로도 더 잘 일반화할 가능성이 있습니다. MNIST 실험에서 그래프 버전보다 simplicial 버전이 더 나았다는 사실은 이 가설과 부합합니다. 

또한 이 모델은 \(X_0, X_1, X_2\)를 분리해 다루므로, 같은 입력이라도 **노드 수준 정보**, **관계(엣지) 정보**, **폐곡면/삼각형 정보**를 나눠 학습할 수 있습니다. 이런 다층 표현은 단순 그래프 신경망보다 더 구조적으로 풍부한 가설 공간을 제공합니다. 

### 3-2. 하지만 원 논문만으로 일반화 향상을 강하게 주장하긴 어렵다

그럼에도 불구하고, **원 논문 자체만 놓고 일반화 성능 향상을 강하게 주장하기는 어렵습니다**. 이유는 세 가지입니다.  
- 데이터셋이 사실상 MNIST 하나뿐이고,  
- simplicial structure가 이미지로부터 인위적으로 구성된 것이며,  
- OOD, topology shift, remeshing, orientation 변화 등에 대한 검증이 없기 때문입니다. 

즉, 이 논문은 “**일반화 향상의 가능성을 보인 초기 시도**”이지, “일반화에 대해 이미 검증된 해법”은 아닙니다. 이 구분이 중요합니다. 

### 3-3. 후속 연구가 보여준 일반화 향상 메커니즘

후속 연구들은 S2CCNN의 일반화 가능성을 **어떤 조건에서 현실화할 수 있는지** 더 구체적으로 보여줍니다.

**(i) 대칭성 보존: orientation/permutation equivariance**  
Roddenberry et al.(ICML 2021)은 simplicial 모델이 permutation equivariance, orientation equivariance, simplicial awareness를 가져야 한다고 주장했고, odd 비선형성(예: \(\tanh\))을 사용할 때 이런 성질을 만족하는 구조가 **보지 못한 trajectory**에 더 잘 일반화한다고 보고했습니다. 같은 논문에서 기존 SCNN(Ebli 2020)과 S2CCNN(Bunch 2020)은 trajectory 예측에서 SCoNe보다 대체로 약한 결과를 보였습니다. 즉, 단순히 simplex를 쓰는 것만으로는 부족하고, **대칭성과 표현 설계**가 일반화에 중요하다는 뜻입니다. 

**(ii) lower/upper 구조 분리와 다중 홉 필터**  
Hodge-aware/SCCNN 계열 연구는 S2CCNN이 사실상 **one-step special case**이며, 더 나은 일반화를 위해서는 lower/upper convolution을 분리하고, multi-hop filtering과 inter-simplicial coupling을 체계적으로 설계해야 한다고 봅니다. 이들은 permutation/orientation equivariance, spectral 해석, perturbation stability, oversmoothing 완화까지 함께 분석합니다. 후속 실험에서는 SCCNN이 Bunch 모델보다 더 좋은 결과를 보이기도 했습니다. 

**(iii) attention 기반 적응형 이웃 가중**  
2022년의 SAN/SAT 계열은 simplicial 이웃들의 중요도를 **학습적으로 재가중**하는 self-attention을 도입했습니다. 이들은 기존의 고정된 convolution보다 **새로운 구조에 적응**하기 쉽고, orientation-equivariant signed attention을 포함해 trajectory/image 분류에서 기존 simplicial convolution 및 GNN보다 좋은 결과를 보고했습니다. 일반화 관점에서는 “어떤 상위/하위 이웃을 더 중요하게 볼지”를 데이터에 맞춰 학습한다는 점이 장점입니다. 

**(iv) 안정성·과도평활화(oversmoothing) 제어**  
2025년 Hodge-aware 연구는 작은 topology perturbation에 대한 **안정성 경계**를 분석했고, lower/upper 분리와 Hodge-aware 설계가 oversmoothing 완화에 도움이 된다고 주장합니다. 같은 해의 COSMOS는 PDE 기반 continuous simplicial network로서 **simplicial perturbation stability**와 oversmoothing 제어를 전면에 내세웠습니다. 이는 앞으로 일반화 연구가 “정확도”뿐 아니라 “구조 변화에 대한 안정성”으로 이동하고 있음을 보여줍니다. 

**(v) 아직 해결되지 않은 핵심 문제: remeshing/generalization across equivalent triangulations**  
가장 중요한 최신 경고는 MANTRA(ICLR 2025)입니다. 이 벤치마크는 여러 graph/simplicial/cellular 모델을 manifold triangulation 과제에서 비교했는데, simplicial complex 기반 모델들이 대체로 **비자명한 topological property 예측에서 graph 모델보다 낫거나 비슷**했지만, barycentric subdivision 같은 **위상은 같고 삼각분할만 바뀐 경우** 성능이 떨어졌다고 보고합니다. 즉, 현재의 고차 모델들은 **진정한 의미의 triangulation-invariant generalization**에는 아직 미치지 못합니다. 

### 3-4. 제 판단: S2CCNN의 일반화 향상 가능성은 “있지만, 조건부”다

정리하면, S2CCNN은 **고차 구조가 본질인 문제**에서는 그래프 모델보다 더 잘 일반화할 여지가 분명히 있습니다. 하지만 그 가능성을 실제 성능으로 바꾸려면 최소한 다음이 필요합니다.

$$
\text{좋은 일반화} \approx
\text{고차 구조 편향}
+
\text{대칭성 보존}
+
\text{안정성 제어}
+
\text{공정한 벤치마크}
$$

원 논문은 이 식에서 첫 항을 제시한 초기 작업이고, 후속 연구들이 나머지 항들을 채워 넣고 있다고 보는 것이 가장 정확합니다. 

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

아래 비교는 **서로 다른 데이터셋/과제/평가지표**를 사용한 논문들을 개념적으로 비교한 것입니다. 따라서 절대적인 “순위표”라기보다, **S2CCNN 이후 어떤 병목이 어떻게 해결되었는가**를 보는 것이 적절합니다. 이 점은 최근 TopoBench와 MANTRA가 왜 표준화된 벤치마크를 강조하는지를 통해서도 확인됩니다. 

### (1) Ebli et al., *Simplicial Neural Networks* (2020)
- **의의**: simplicial convolution 자체를 체계적으로 제시한 초기 대표작입니다.
- **S2CCNN와 차이**: SNN은 주로 **한 simplex 차원 내부**의 convolution에 무게가 있고, 후속 Hodge-aware 분석에 따르면 lower/upper 성분을 분리하지 않아 gradient/curl subspace를 함께 처리합니다.
- **일반화 관점**: S2CCNN은 여기서 더 나아가 **0/1/2-simplex 간 교차 차원 메시지**를 명시적으로 다뤘다는 점이 차별점입니다. 반면 후속 관점에서는 둘 다 아직 초기형으로 분류됩니다. 

### (2) Bodnar et al., *Weisfeiler and Lehman Go Topological: Message Passing Simplicial Networks* (ICML 2021)
- **의의**: simplicial complex용 message passing을 제안하고, **Simplicial WL(SWL)** 기반의 표현력 분석을 제시했습니다.
- **S2CCNN 대비 개선점**: “고차 구조를 쓴다”는 직관을 넘어서, **왜 더 표현력이 강한지**를 이론적으로 설명했습니다.
- **일반화 관점**: 표현력은 올라가지만, 이것이 곧 remeshing-invariant generalization을 뜻하지는 않습니다. 이후 MANTRA는 고차 MP 계열도 subdivision에 취약할 수 있음을 보여줍니다. 

### (3) Roddenberry et al., *Principled Simplicial Neural Networks for Trajectory Prediction* (ICML 2021)
- **의의**: permutation equivariance, orientation equivariance, simplicial awareness라는 **설계 원칙**을 명시했습니다.
- **S2CCNN 대비 개선점**: 일반화가 잘 되려면 단순한 고차 구조 사용만으로는 부족하고, **대칭성을 보존하는 설계**가 필요하다는 점을 보였습니다.
- **실증**: 보지 못한 trajectory에 대한 일반화에서 tanh 기반 admissible 구조가 ReLU/sigmoid 기반보다 유리했고, Ocean/Berlin trajectory에서도 SCoNe가 SCNN/S2CCNN보다 나은 결과를 보였습니다.
- **시사점**: S2CCNN을 개선하려면 활성함수, orientation 처리, simplicial awareness를 재설계해야 합니다. 

### (4) Giusti et al., *Simplicial Attention Neural Networks* / Goh et al., *Simplicial Attention Networks* (2022)
- **의의**: simplicial domain에 **attention**을 도입했습니다.
- **S2CCNN 대비 개선점**: 고정된 convolution 커널 대신, 상위/하위 이웃의 기여를 **데이터 의존적으로 조절**합니다.
- **일반화 관점**: 새로운 구조나 비균질한 복잡도에서 attention이 더 유연한 inductive bias를 줄 수 있으며, trajectory/image/inductive-transductive 과제에서 기존 simplicial convolution보다 우수한 결과가 보고되었습니다. 

### (5) Yang et al., *Hodge-Aware Convolutional Learning on Simplicial Complexes* (TMLR 2025)
- **의의**: 최근 가장 중요한 이론적 정리 중 하나입니다.
- **핵심**: lower/upper adjacency 분리, inter-simplicial coupling, higher-order convolution을 갖춘 일반 구조를 제시하고, **Hodge subspace 보존**, **stability**, **oversmoothing 완화**를 분석합니다.
- **S2CCNN와 관계**: S2CCNN을 **one-step special case**로 재해석합니다.
- **일반화 관점**: 제 개인적 판단으로, “원 논문이 가진 일반화 잠재력”을 실질적인 방법론으로 확장한 가장 직접적인 후속 계열입니다. 

### (6) Ballester et al., *MANTRA: The Manifold Triangulations Assemblage* (ICLR 2025)
- **의의**: 진짜 higher-order/topological 정보를 담은 대규모 벤치마크를 제시했습니다.
- **핵심 결과**: simplicial 모델들은 비자명한 topological property 예측에서 graph 모델보다 전반적으로 강한 면이 있지만, **subdivision/remeshing**에서 성능 저하가 남아 있습니다.
- **일반화 관점**: 앞으로의 핵심은 “고차 구조를 쓰는가”가 아니라, “**같은 위상인데 분할만 달라져도 잘 버티는가**”입니다. 

### (7) Telyatnikov et al., *TopoBench: A Framework for Benchmarking Topological Deep Learning* (JDMLR 2025)
- **의의**: topological deep learning 파이프라인을 표준화하는 벤치마킹 프레임워크입니다.
- **영향**: S2CCNN 같은 모델을 앞으로 연구할 때, 단순히 한두 데이터셋에서 좋아 보이는지보다 **lifting 방식, 전처리, 하이퍼파라미터, 비교군 공정성**까지 통제해야 함을 제도화했습니다.
- **일반화 관점**: 모델 일반화 성능의 신뢰성은 구조뿐 아니라 **평가 프로토콜의 표준화**에도 달려 있음을 보여줍니다. 

---

## 5. 해당 논문이 앞으로의 연구에 미친 영향

이 논문의 가장 큰 영향은, **“simplicial complex 위에서도 GCN처럼 층을 만들 수 있다”**는 매우 직접적인 설계를 보여주었다는 점입니다. 이후 연구들은 이 아이디어를 세 방향으로 확장했습니다.

1. **표현력 분석 방향**: MPSN처럼 “왜 더 강한가?”를 이론화.   
2. **대칭성/일반화 방향**: PSNN처럼 orientation/permutation equivariance를 전면화.   
3. **주파수·안정성 방향**: Hodge-aware/SCCNN처럼 lower-upper 분리, stability, oversmoothing 분석을 정교화. 

즉, S2CCNN은 오늘 기준에서 가장 강한 모델은 아니지만, **후속 계열의 설계 어휘를 열어준 논문**이라고 평가하는 것이 적절합니다. 특히 후속 연구가 이 모델을 “특수한 저차 버전”으로 재해석했다는 사실 자체가, 이 논문이 계보의 출발점 중 하나였음을 보여줍니다. 

---

## 6. 앞으로 연구 시 고려할 점

### (a) remeshing / triangulation invariance를 명시적으로 다뤄야 함
같은 위상 구조라도 triangulation만 달라지면 성능이 흔들리는 문제는 아직 큽니다. 따라서 앞으로는 **subdivision-consistent loss**, **spectral transfer regularization**, **mesh refinement augmentation** 같은 방향이 중요합니다. MANTRA는 이 문제가 실제로 남아 있음을 보여줍니다. 

### (b) orientation equivariance를 기본 설계 원리로 넣어야 함
simplicial data에서 simplex orientation은 임의적인 경우가 많습니다. 따라서 활성함수와 메시지 패싱 규칙이 orientation 변화에 대해 일관되게 반응하도록 설계해야 합니다. PSNN과 SAT 계열이 이 점을 분명히 보여줍니다. 

### (c) lower/upper/Hodge subspace를 분리한 설계가 유리함
원 논문은 좋은 출발이지만, 후속 연구 기준으로는 너무 단순합니다. gradient/curl/harmonic 성분을 구분하고 lower/upper convolution을 분리하는 것이 일반화와 해석 가능성 모두에 더 유리합니다. 

### (d) self-supervised/topology-preserving pretraining이 유망함
레이블이 적은 현실 문제에서는 supervised만으로 일반화 확보가 어렵습니다. 2023년 *TopoSRL*은 topology-preserving self-supervised learning을 통해 higher-order interaction과 long-range dependency를 보존하는 표현을 학습하려 했습니다. 향후 S2CCNN 계열에도 이런 사전학습이 접목될 가능성이 큽니다. 

### (e) 진짜 고차 데이터로 평가해야 함
그래프를 단순 lifting한 데이터만으로는 “정말 simplicial 모델이 필요한가?”에 답하기 어렵습니다. MANTRA가 강조하듯, **본질적으로 higher-order인 데이터셋**에서 평가해야 논문의 기여가 분명해집니다. 

### (f) 계산 효율성과 확장성도 중요함
고차 구조는 표현력은 좋지만 비용이 커질 수 있습니다. 최근에는 binarized simplicial CNN처럼 효율성과 oversmoothing 완화를 동시에 보려는 시도도 나왔고, continuous/PDE 기반 모델도 등장했습니다. 따라서 앞으로는 “더 정확한가?”뿐 아니라 “더 안정적이고 scalable한가?”까지 같이 봐야 합니다. 

---

## 7. 최종 요약

한 문장으로 요약하면, **S2CCNN은 “그래프 합성곱을 simplicial 2-complex로 확장해 노드-엣지-삼각형을 함께 학습하자”는 초기이자 중요한 제안**입니다. 원 논문의 실험 증거는 제한적이지만, 이후 연구들은 이 아이디어를 표현력, 대칭성, 안정성, attention, 벤치마크 표준화 방향으로 확장했습니다. 따라서 오늘 시점에서 이 논문을 읽는 가장 좋은 방법은, **완성형 SOTA 모델**로 보기보다 **현대 simplicial/topological deep learning의 출발점 중 하나**로 읽는 것입니다. 그리고 일반화 성능 향상이라는 관점에서는, 원 논문의 고차 inductive bias가 유망한 씨앗이었고, 후속 연구가 그 씨앗에 **equivariance, Hodge-awareness, stability, benchmarking**을 덧붙이며 실제 연구 프로그램으로 발전시킨 것으로 보는 것이 정확합니다. 

---

## 참고자료 / 참고한 논문 제목

아래는 제가 실제로 참고한 주요 자료들입니다.

1. **Eric Bunch, Qian You, Glenn Fung, Vikas Singh, “Simplicial 2-Complex Convolutional Neural Nets / Networks”** (NeurIPS 2020 TDA and Beyond Workshop; arXiv:2012.06010).   
2. **Stefania Ebli, Michaël Defferrard, Gard Spreemann, “Simplicial Neural Networks”** (arXiv:2010.03633, 2020).   
3. **Cristian Bodnar et al., “Weisfeiler and Lehman Go Topological: Message Passing Simplicial Networks”** (ICML 2021).   
4. **T. Mitchell Roddenberry, Nicholas Glaze, Santiago Segarra, “Principled Simplicial Neural Networks for Trajectory Prediction”** (ICML 2021).   
5. **L. Giusti et al., “Simplicial Attention Neural Networks”** (arXiv:2203.07485, 2022).   
6. **Christopher Wei Jin Goh, Cristian Bodnar, Pietro Liò, “Simplicial Attention Networks”** (arXiv:2204.09455, 2022 Workshop version).   
7. **Maosheng Yang, Geert Leus, Elvin Isufi, “Hodge-Aware Convolutional Learning on Simplicial Complexes”** (Transactions on Machine Learning Research, 2025).   
8. **Rubén Ballester et al., “MANTRA: The Manifold Triangulations Assemblage”** (ICLR 2025).   
9. **Lev Telyatnikov et al., “TopoBench: A Framework for Benchmarking Topological Deep Learning”** (Journal of Data-centric Machine Learning Research, 2025).   
10. **OpenReview, “TopoSRL: Topology preserving self-supervised Simplicial Representation Learning”** (NeurIPS 2023 poster).   
11. **Aref Einizade et al., “COSMOS: Continuous Simplicial Neural Networks”** (arXiv preprint, 2025).   
12. **Yi Yan, Ercan E. Kuruoglu, “Binarized Simplicial Convolutional Neural Networks”** (arXiv preprint, 2024).   

</details>
