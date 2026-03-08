# Principled Simplicial Neural Networks for Trajectory Prediction

# 1. 핵심 주장과 주요 기여 요약

이 논문은 **단체 복합체(simplicial complex)** 위에 정의된 데이터를 처리하기 위한 신경망 아키텍처를 구축하면서, 단체 복합체의 체인 복합체(chain complex) 상 사상(map)을 분석하여 세 가지 바람직한 성질을 정의한다: **순열 등변성(permutation equivariance)**, **방향 등변성(orientation equivariance)**, **단체 인식(simplicial awareness)**.

처음 두 성질은 단체 복합체에서 노드 인덱싱과 심플렉스 방향이 임의적이라는 사실을 반영하며, 마지막 성질은 신경망의 출력이 단체 복합체의 일부 차원이 아닌 **전체 단체 복합체**에 의존해야 함을 인코딩한다.

이 세 가지 성질에 기반하여, 대수적 위상수학(algebraic topology) 도구에 근거한 단순한 합성곱 아키텍처 **SCoNe(Simplicial Complex Network)**를 경로 예측(trajectory prediction) 문제에 제안하고, **홀수(odd), 비선형 활성 함수**를 사용할 때 세 가지 성질을 모두 만족함을 증명한다.

**주요 기여:**
1. 단체 신경망(SNN)이 갖추어야 할 3대 원칙적(principled) 성질의 공리적 정의
2. SCoNe 아키텍처 제안 및 해당 성질 만족에 대한 이론적 증명
3. 합성·실제 데이터에서의 경로 예측 성능 및 일반화 검증

---

# 2. 상세 분석

## 2.1 해결하고자 하는 문제

기존의 그래프 신경망(GNN)은 **노드 간 쌍방 관계(pairwise relationship)** 만을 모델링하며, 현실의 네트워크 구조와 노드 상호작용은 쌍방 연결 외에도 노드 간 **고차 상호작용(higher-order interactions)** 을 포함하는 경우가 빈번하다. 경로 예측 문제에서 에지(edge) 위의 흐름(flow) 데이터는 그래프의 삼각형(triangle) 등 고차 구조에 의해 크게 영향을 받는데, 이전 연구들은 에지에 임의로 부여된 방향(orientation)에 대한 흐름을 다루는 데 전기 회로의 전류 분석과 유사한 접근을 취했지만, 기존 SNN 아키텍처(Ebli et al., 2020; Bunch et al., 2020)들은 방향 등변성 등 핵심 성질을 이론적으로 보장하지 못했다.

특히, 기존 SNN(Ebli et al., 2020)의 저자들은 경험적 연구에서 **Leaky ReLU** 활성 함수를 사용했는데, 이는 홀수 함수가 아니므로 방향 등변성을 만족하지 못한다.

## 2.2 제안하는 방법 — SCoNe 아키텍처

### 2.2.1 단체 복합체 및 Hodge Laplacian 기본 구조

$k$-차 단체 복합체 $\mathcal{X}$에서 **경계 연산자(boundary operator)** $\partial_k : C_k \to C_{k-1}$이 정의된다. 이를 행렬로 표현하면 **접합 행렬(incidence matrix)** $B_k$가 된다. **Hodge Laplacian**은 다음과 같이 정의된다:

$$L_k = B_k^\top B_k + B_{k+1} B_{k+1}^\top = L_{k,\text{down}} + L_{k,\text{up}}$$

여기서:
- $L_{k,\text{down}} = B_k^\top B_k$: 하위 인접성(lower adjacency) 정보
- $L_{k,\text{up}} = B_{k+1} B_{k+1}^\top$: 상위 인접성(upper adjacency) 정보

Hodge Laplacian의 핵(kernel) $\ker(\Delta_1)$로의 사영이 경계 맵의 핵 $\ker(\partial_1)$로의 사영보다 크게 우수한 성능을 보이며, 이는 $\mathcal{X}_2$(삼각형) 정보를 통합하는 것이 중요함을 시사한다.

### 2.2.2 SCoNe 단일 레이어 수식

SCoNe의 한 레이어는 1-체인(1-chain) $c^{(\ell)} \in C_1$을 입력으로 받아 다음과 같이 갱신한다:

$$c^{(\ell+1)} = \phi\Big( W_1^{(\ell)} \cdot B_1^\top B_1 \, c^{(\ell)} + W_2^{(\ell)} \cdot B_2 B_2^\top \, c^{(\ell)} + W_3^{(\ell)} \cdot c^{(\ell)} \Big)$$

여기서:
- $W_1^{(\ell)}, W_2^{(\ell)}, W_3^{(\ell)}$: 학습 가능한 스칼라 가중치
- $B_1$: 노드-에지 접합 행렬 ($\partial_1$에 해당)
- $B_2$: 에지-삼각형 접합 행렬 ($\partial_2$에 해당)
- $\phi$: **홀수(odd), 비선형** 원소별 활성 함수 (예: $\tanh$)

최종 readout은 다음과 같이 경계 연산자를 적용하여 0-체인(노드 확률)으로 변환한다:

$$p = \text{softmax}\Big(B_1 \, c^{(L)}\Big)$$

### 2.2.3 세 가지 핵심 성질

**Property 1 (Permutation Equivariance)**: 단체 복합체 $\mathcal{X}$에서 임의의 순열 행렬 $P_j$에 대해, $\text{SCN}\_{W,\partial}(c_j) = P_\ell \, \text{SCN}_{W,P\partial}(P_j c_j)$.

**Property 2 (Orientation Equivariance)**: 방향 뒤집기 행렬 $D_j$ (대각 원소가 $\pm 1$)에 대해, $\text{SCN}\_{W,\partial}(c_j) = D_\ell \, \text{SCN}_{W,D\partial}(D_j c_j)$.

**Property 3 (Simplicial Awareness)**: 출력 $\text{SCN}\_{W,\partial}: C_j \to C_\ell$이 입출력 차원($j, \ell$)이 아닌 다른 차원 $k$의 구조에도 의존하는 성질. 이 세 성질을 모두 만족하는 아키텍처를 **admissible**이라 정의한다.

**Proposition 1 (핵심 정리):**

SCoNe (Algorithm 1)가 admissible하기 위한 필요충분조건은 $\phi$가 **홀수(odd)이고 비선형(nonlinear)** 함수인 것이다.

$$\phi(-x) = -\phi(x), \quad \forall x \in \mathbb{R} \quad \text{(odd)}$$

이 조건은 에지 방향이 임의로 선택되어도 신경망 출력이 일관됨을 보장한다.

## 2.3 모델 구조

SCoNe의 전체 파이프라인은 다음과 같다:

1. **입력 인코딩**: 궤적(trajectory)의 노드 시퀀스 $[i_0, i_1, \ldots, i_{m-1}]$를 에지(oriented edge)의 1-체인으로 리프팅
2. **SCoNe 레이어** $\times L$: 위 수식에 따라 하위($B_1^\top B_1$), 상위($B_2 B_2^\top$), 항등(identity) 세 연산자를 통해 합성곱 수행
3. **Readout**: $B_1 c^{(L)}$에 softmax를 적용하여 다음 노드 예측 확률 분포 도출

SCoNe는 노드 및 에지 레벨 데이터뿐만 아니라 그래프의 고차 구조(삼각형)를 활용하여 매우 일반화 가능한 경로 예측 모델을 학습한다.

## 2.4 성능 향상

합성(synthetic) 데이터셋에서 무작위 방향 에지를 사용한 경로 예측 과제에서 SCoNe를 Markov chain, RNN, 조화 사영(harmonic projection) 방법, SCNN (Ebli et al., 2020), S2CCNN (Bunch et al., 2020)과 비교하여 테스트 정확도를 평가했다.

합성 데이터셋(수동 방향 에지)에서 다양한 활성 함수별 비교, 그리고 Ocean Drifters 및 Berlin 실제 데이터셋에서도 평가가 수행되었다.

## 2.5 한계

- **스케일러빌리티**: 접합 행렬의 크기가 커질 경우 연산 비용이 증가
- 이산(discrete) SNN은 각 다항식 차수에 대해 정보 전파가 고정되어 적응성이 제한된다.
- 기존 단체 신경망들은 기울기 소실(gradient vanishing), 과평활(over-smoothing), 과적합(over-fitting) 문제로 인해 매우 얕은 모델에 제한되는 경향이 있다.
- SCoNe는 1-체인 상에서만 동작하며, 여러 차수의 체인을 동시에 유지하는 아키텍처(예: Bunch et al.)와 비교하여 표현력에 한계가 있을 수 있음

---

# 3. 모델의 일반화 성능 향상 가능성

이 논문에서 일반화(generalization)는 핵심 주제 중 하나이며, 두 가지 방식으로 검증되었다:

### 3.1 역방향 테스트 (Reversed Test Set)

일반화 성질을 두 가지 방식으로 실증하는데, 첫째, 훈련 세트는 동일하게 유지하면서 테스트 세트의 궤적 방향을 **역전(reverse)** 시킨다. 따라서, 궤적의 특정 방향을 "암기(memorizing)"하는 데 지나치게 의존하는 모델은 이 테스트에서 성능이 저하된다.

SCoNe가 역방향 테스트에서도 높은 정확도를 유지하는 이유는 **방향 등변성** 때문이다. 에지 방향이 뒤집혀도 $\phi(-x) = -\phi(x)$ 조건에 의해 출력이 일관되게 변환되므로, 모델이 흐름의 **구조적 패턴**을 학습한 것이지 특정 방향을 암기한 것이 아님을 입증한다.

### 3.2 전이 학습 (Transfer Learning)

훈련과 테스트에 사용되는 단체 복합체의 구조를 달리하여, 한 복합체에서 학습한 모델이 다른 복합체에서도 작동하는지 평가한다. 이는 단체 인식(simplicial awareness)이 보장하는 성질로, 모델이 전체 위상 구조에 의존하여 학습하므로 새로운 구조에도 전이가 가능하다.

### 3.3 일반화의 이론적 근거

일반화 성능 향상의 핵심은 다음의 수학적 구조에 기인한다:

$$c^{(\ell+1)} = \phi\Big( W_1^{(\ell)} L_{1,\text{down}} \, c^{(\ell)} + W_2^{(\ell)} L_{1,\text{up}} \, c^{(\ell)} + W_3^{(\ell)} c^{(\ell)} \Big)$$

- $L_{1,\text{down}}$: 두 에지가 공유 노드를 통해 인접하는 정보 (그래프 구조)
- $L_{1,\text{up}}$: 두 에지가 공통 삼각형의 면(face)인지에 대한 정보 (고차 위상 구조)

Hodge Laplacian의 핵으로의 사영이 경계 맵의 핵으로의 사영보다 크게 우수한 성능을 보이며, 이는 삼각형 정보($\mathcal{X}_2$)를 통합하는 것이 이 문제에서 중요함을 시사한다.

즉, **Hodge 분해(Hodge decomposition)** 에 의해 에지 위의 흐름은 세 직교 부분공간으로 분해된다:

$$C_1 = \text{im}(B_1^\top) \oplus \text{im}(B_2) \oplus \ker(\Delta_1)$$

- $\text{im}(B_1^\top)$: gradient 흐름 (노드 포텐셜의 기울기)
- $\text{im}(B_2)$: curl 흐름 (삼각형의 회전)
- $\ker(\Delta_1)$: 조화(harmonic) 흐름 (위상적 구멍에 대응)

SCoNe는 이 세 성분을 독립적으로 가중하여 처리함으로써, 위상 구조의 본질적 정보를 포착하고 일반화를 촉진한다.

---

# 4. 향후 연구에 미치는 영향 및 고려 사항

## 4.1 연구에 미치는 영향

1. **이론적 기초 수립**: Roddenberry et al.은 순열, 방향 등변성 및 단체 인식 성질을 논의하고 SCoNe를 경로 예측에 제안함으로써, 후속 단체 신경망 연구의 이론적 기준을 마련했다.

2. **후속 아키텍처 촉발**: 이 논문 이후 다양한 단체 신경망이 등장했으며, 모두 SCoNe의 세 가지 성질을 참조 또는 확장한다:
   - SCNN (Yang, Isufi & Leus, ICASSP 2022): 심플렉스 위에 정의된 데이터를 학습하는 SCNN 아키텍처를 제안하고, 순열 및 방향 등변성, 복잡도, 스펙트럼 분석을 연구했다.
   - SAT (Goh, Bodnar & Liò, ICLR Workshop 2022): 이웃 심플렉스 간 상호작용을 동적으로 가중하고 새로운 구조에 적응할 수 있는 Simplicial Attention Networks를 제안했다.
   - COSIMO (Einizade et al., 2025): 단체 복합체 위의 PDE에서 유도된 연속 SNN 아키텍처를 도입하고, 단체 섭동(perturbation)에 대한 안정성의 이론적·실험적 근거를 제공했다.

3. **Topological Deep Learning 분야 성장**: GNN과 달리 SNN의 이론적 이해는 아직 발전 중이며, SCoNe의 원칙적 접근은 이 분야의 이론화를 가속했다.

## 4.2 향후 연구 시 고려할 점

1. **Over-smoothing 문제**: COSIMO 논문에서는 과평활(over-smoothing) 현상을 조사하여, 이산 SNN보다 더 나은 제어가 가능함을 보여주었다. 깊은 SCoNe 모델 설계 시 이 문제를 반드시 고려해야 한다.

2. **연속 동역학 접근**: 기존 SNN이 이산 필터링에 주로 의존하는 것은 제한적일 수 있으며, 단체 복합체 위의 편미분방정식(PDE)은 연속 동역학을 포착하는 원리적 접근을 제공한다.

3. **쌍곡 기하학으로의 확장**: 기존 단체 신경망은 주로 유클리드 공간에 심플렉스 특징을 임베딩하지만, 유클리드 임베딩은 특히 스케일 프리 성질이나 계층적 구조를 가진 네트워크에서 큰 왜곡을 초래할 수 있다.

4. **대규모 데이터 확장성**: 접합 행렬 연산의 계산 복잡도 개선, 고차 심플렉스의 효율적 구성 방법 연구 필요

5. **다양한 태스크 적용**: 경로 예측 외에도 노드/에지 분류, 링크 예측, 결측값 보간 등으로의 확장 가능성 탐색

---

# 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 핵심 특징 | SCoNe 대비 차별점 |
|------|------|-----------|------------------|
| **SNN** (Ebli et al.) | 2020 | Hodge Laplacian 기반 합성곱, GCN의 단체 확장 | 하위/상위 이웃 분리 처리 없음; Leaky ReLU 사용으로 방향 등변성 미충족 |
| **S2CCNN** (Bunch et al.) | 2020 | 모든 차원의 체인을 동시에 유지 | 정규화된 경계 맵 사용; admissible 조건($\phi$가 odd)은 동일 |
| **MPSN** (Bodnar et al.) | 2021 | 메시지 패싱을 단체 복합체로 확장, SWL 테스트 | MPSN은 이상적 조건에서 vanilla GNN보다 강력함이 증명되어, 표현력 측면에서 우위 |
| **SCoNe** (본 논문) | 2021 | 3대 원칙 공리화, odd 비선형 활성 함수 필요충분 조건 | **이론적 기초** 제공이 핵심 차별점 |
| **SCNN** (Yang et al.) | 2022 | Hodge 분해를 단체 합성곱에 적용하여 SNN의 성능을 개선 | 독립적 하위/상위 필터, 더 유연한 스펙트럼 필터 |
| **SAT/SAN** (Goh et al. / Giusti et al.) | 2022 | 단체 복합체에 정의된 데이터에 마스크된 자기-어텐션 레이어를 활용하는 Simplicial Attention Neural Networks 도입 | 어텐션 기반으로 구조 적응성 강화 |
| **BScNets** (Chen et al.) | 2022 | 그래프 Laplacian을 블록 Hodge Laplacian으로 대체 | 블록 구조로 다차원 상호작용 체계적 통합 |
| **HiGCN** (AAAI 2024) | 2024 | SNN, SCNN 등 Hodge Laplacian 기반 단체 GCN보다 유연하며, 경계 연산자를 통한 정보 교환 제약을 극복 | Flower-Petals Laplacian으로 다차수 심플렉스 간 유연한 상호작용 |
| **Bi-SCNN** (2024) | 2024 | 이진화된 단체 합성곱과 가중 이진-부호 전방 전파를 결합한 새로운 아키텍처 | 모델 복잡도 감소, 실행 시간 단축, over-smoothing 완화 |
| **COSIMO** (Einizade et al.) | 2025 | 단체 복합체 위 PDE에서 유도된 연속 SNN으로, 안정성에 대한 이론적·실험적 검증 제공 | 연속 수용 영역(receptive field)으로 over-smoothing 제어 향상 |
| **HSCN** (2025) | 2025 | 쌍곡 기하학과 단체 신경망을 결합한 최초의 모델 | 계층적 구조 데이터에 대한 왜곡 감소 |

---

# 참고 자료 (출처)

1. **T. M. Roddenberry, N. Glaze, S. Segarra**, "Principled Simplicial Neural Networks for Trajectory Prediction," *ICML 2021* (PMLR 139). [arXiv:2102.10058](https://arxiv.org/abs/2102.10058), [PMLR PDF](http://proceedings.mlr.press/v139/roddenberry21a/roddenberry21a.pdf)
2. **GitHub 구현**: [nglaze00/SCoNe_GCN](https://github.com/nglaze00/SCoNe_GCN)
3. **S. Ebli, M. Defferrard, G. Spreemann**, "Simplicial Neural Networks," *NeurIPS 2020 Workshop on TDA and Beyond*, 2020.
4. **E. Bunch, Q. You, G. Fung, V. Singh**, "Simplicial 2-Complex Convolutional Neural Networks," *NeurIPS 2020 Workshop on TDA and Beyond*, 2020.
5. **C. Bodnar et al.**, "Weisfeiler and Lehman Go Topological: Message Passing Simplicial Networks," *ICML 2021*.
6. **M. Yang, E. Isufi, G. Leus**, "Simplicial Convolutional Neural Networks," *ICASSP 2022*. [TU Delft](https://research.tudelft.nl/en/publications/simplicial-convolutional-neural-networks/)
7. **C. W. J. Goh, C. Bodnar, P. Liò**, "Simplicial Attention Networks," *ICLR 2022 Workshop*. [OpenReview](https://openreview.net/forum?id=ScfRNWkpec)
8. **L. Giusti et al.**, "Simplicial Attention Neural Networks," *arXiv:2203.07485*, 2022.
9. **Y. Chen, Y. R. Gel, H. V. Poor**, "BScNets: Block Simplicial Complex Neural Networks," *AAAI 2022*.
10. **A. Einizade et al.**, "Continuous Simplicial Neural Networks (COSIMO)," *arXiv:2503.12919*, 2025. [OpenReview](https://openreview.net/forum?id=fPPfFMVTZo)
11. **Binarized Simplicial Convolutional Neural Networks (Bi-SCNN)**, *Neural Networks*, 2024. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0893608024008578)
12. **Hyperbolic Simplicial Convolutional Network (HSCN)**, *Expert Systems with Applications*, 2025. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0957417426004811)
13. **DeepSCNN**, *Applied Intelligence*, Springer, 2025. [Springer](https://link.springer.com/article/10.1007/s10489-024-06121-6)
14. **Higher-Order GCN with Flower-Petals Laplacians (HiGCN)**, *arXiv:2309.12971*, 2024. [arXiv](https://arxiv.org/html/2309.12971)
15. **Awesome Topological Deep Learning** 목록: [GitHub](https://github.com/lrnzgiusti/awesome-topological-deep-learning)

---

아래 내용은 사용자가 제공한 논문 본문과 보충자료에 근거해 정리했습니다. 다만, 사용자가 요청한 “2020년 이후 관련 최신 연구 비교 분석”은 현재 대화에 제공된 자료만으로는 충분하지 않아, 그 부분은 **제가 확실히 말할 수 있는 범위**까지만 제한적으로 서술합니다. 확인되지 않은 최신 논문을 임의로 덧붙이지 않겠습니다.

# 1. 핵심 주장과 주요 기여 간결 요약

이 논문의 핵심 주장은 다음입니다.

기존 그래프 신경망을 단순히 고차 구조로 확장하는 것만으로는 충분하지 않으며, simplicial complex 위의 신경망은 반드시 다음 세 성질을 원칙적으로 만족해야 한다는 것입니다.

$$
\text{(i) permutation equivariance},\quad
\text{(ii) orientation equivariance},\quad
\text{(iii) simplicial awareness}
$$

이를 바탕으로 저자들은 궤적 예측(trajectory prediction)을 위한 simplicial neural network인 **SCoNe**를 제안합니다. 이 모델은 경로를 edge-flow, 즉 $1$-chain으로 표현하고, 경계연산자(boundary operator)와 Hodge Laplacian의 상·하위 구조를 함께 활용합니다.

가장 중요한 이론적 기여는, 제안 아키텍처가 위 세 성질을 만족하려면 활성함수 $\phi$가 반드시 **홀함수(odd)** 이면서 **비선형(nonlinear)** 이어야 한다는 점을 보인 것입니다. 실험적으로도 이러한 “원칙적 설계”가 보지 못한 궤적이나 방향이 뒤집힌 테스트셋에 대해 더 좋은 **일반화 성능**으로 이어진다고 주장합니다.

# 2. 자세한 설명

## 2.1 해결하고자 하는 문제

논문이 해결하려는 문제는 크게 두 층위입니다.

첫째, **simplicial complex 위 데이터에 적합한 신경망 설계 원리**를 정립하는 것입니다. 그래프는 노드와 엣지의 쌍대 관계만 표현하지만, 실제 데이터는 삼각형 이상의 고차 상호작용을 포함할 수 있습니다. 예를 들어, 이동 경로나 흐름(flow)은 노드가 아니라 주로 엣지 위에 놓이고, 삼각형 구조는 국소적 순환이나 보존 법칙을 설명하는 데 중요합니다.

둘째, 이러한 원리를 바탕으로 **trajectory prediction** 문제를 풀고자 합니다. 입력이 부분 궤적

$$
[i_0,i_1,\dots,i_{m-1}]
$$

일 때, 다음 노드

$$
i_m
$$

를 예측하는 문제입니다.

저자들은 궤적이 종종 Hodge decomposition 관점에서 harmonic한 성질을 갖는다고 봅니다. 즉, 자연스러운 이동은 대체로 불필요한 국소 순환을 만들지 않고, 보존적 흐름처럼 보이기 쉽다는 직관입니다.

---

## 2.2 배경: simplicial complex, boundary operator, Hodge Laplacian

### Simplicial complex

simplicial complex $X$는 노드, 엣지, 삼각형 등의 simplex 집합입니다. $k$-simplex는 원소 수가 $k+1$인 simplex입니다.

### Boundary operator

논문은 oriented simplex에 대해 경계연산자 $\partial_k$를 다음과 같이 정의합니다.

```math
\partial_k [i_0,i_1,\dots,i_k]
=
\sum_{j=0}^{k} (-1)^j [i_0,\dots,i_{j-1},i_{j+1},\dots,i_k]
```

중요한 성질은

$$
\partial_{k-1}\circ \partial_k = 0
$$

입니다.

### Hodge Laplacian

$k$차 Hodge Laplacian은

$$
\Delta_k = \partial_k^\top \partial_k + \partial_{k+1}\partial_{k+1}^\top
$$

로 정의됩니다.

특히 $1$-chain 공간 $C_1$에 대해

$$
\Delta_1 = \partial_1^\top \partial_1 + \partial_2 \partial_2^\top
$$

이며, 이는 edge flow를 분석하는 핵심 연산자입니다.

### Hodge decomposition

논문은

```math
C_k = \text{im}(\partial_{k+1})
\oplus
\text{im}(\partial_k^\top)
\oplus
\ker(\Delta_k)
```

를 사용합니다.

$k=1$일 때:

```math
C_1
=
\text{im}(\partial_2)
\oplus
\text{im}(\partial_1^\top)
\oplus
\ker(\Delta_1)
```

각 항의 의미는 다음과 같습니다.

$$
\text{im}(\partial_2)
$$

는 triangle 주위의 curl 성분,

$$
\text{im}(\partial_1^\top)
$$

는 node potential의 gradient 성분,

$$
\ker(\Delta_1)
$$

는 harmonic 성분입니다.

이 분해는 trajectory가 왜 단순 노드 신호가 아니라 edge flow로 다뤄져야 하는지 설명합니다.

---

## 2.3 제안하는 방법: admissible simplicial neural architecture

논문의 가장 중요한 이론적 틀은 **admissibility**입니다.

### 1) Permutation equivariance

simplex의 인덱싱은 임의적이므로, 라벨을 바꿔도 결과가 일관돼야 합니다.

논문 정의는 다음과 같습니다.

```math
\mathrm{SCN}_{W,\partial}(c_j)
=
P_\ell\, \mathrm{SCN}_{W,P\partial}(P_j c_j)
```

여기서 $P=\{P_k\}$는 각 차수 simplex에 대한 permutation matrix이고,

$$
[P\partial]_k := P_{k-1}\partial_k P_k^\top
$$

입니다.

### 2) Orientation equivariance

엣지나 삼각형의 방향도 임의적이므로, orientation을 뒤집어도 결과가 그에 맞게 바뀌어야 합니다.

```math
\mathrm{SCN}_{W,\partial}(c_j)
=
D_\ell\, \mathrm{SCN}_{W,D\partial}(D_j c_j)
```

여기서 $D_k$는 대각원소가 $\pm 1$인 diagonal matrix입니다.

### 3) Simplicial awareness

모델 출력이 특정 차수의 simplex만 보고 결정되면 안 되고, 전체 simplicial structure의 영향을 받아야 합니다. 특히 $2$-simplex(삼각형)가 바뀌면 결과도 바뀔 수 있어야 합니다.

### Admissibility

세 조건을 모두 만족하면 admissible architecture입니다.

---

## 2.4 모델 구조: SCoNe

논문은 차원 $2$의 simplicial complex에서 trajectory prediction을 위한 **SCoNe**를 제안합니다.

### 입력 표현

부분 궤적

$$
[i_0,i_1,\dots,i_{m-1}]
$$

를 oriented edge들의 합으로 바꿉니다.

$$
c_1^0 = \sum_{j=0}^{m-2} [i_j,i_{j+1}]
$$

즉, 순차 데이터를 $1$-chain으로 “lift and collapse”합니다.

### 각 층의 업데이트

핵심 layer는 다음입니다.

```math
c_1^{\ell+1}
=
\phi\!\left(
\partial_2\partial_2^\top c_1^\ell W_2^\ell
+
c_1^\ell W_1^\ell
+
\partial_1^\top \partial_1 c_1^\ell W_0^\ell
\right)
```

이 식은 세 항으로 나뉩니다.

첫째,

$$
\partial_2\partial_2^\top c_1^\ell W_2^\ell
$$

는 상위 인접성, 즉 triangle을 통한 정보 전달입니다.

둘째,

$$
c_1^\ell W_1^\ell
$$

는 자기항입니다.

셋째,

$$
\partial_1^\top \partial_1 c_1^\ell W_0^\ell
$$

는 하위 인접성, 즉 node를 매개로 한 정보 전달입니다.

즉, 그래프 GNN의 메시지 패싱을 simplicial complex의 위아래 incidence 구조로 일반화한 형태입니다.

### 출력 단계

마지막에는 $1$-chain을 $0$-chain으로 변환합니다.

$$
c_0^{L+1} = \partial_1 c_1^L W_0^L
$$

그 후 마지막 노드 $i_{m-1}$의 이웃 집합 $N(i_{m-1})$ 위에서 softmax를 취해 다음 노드를 예측합니다.

$$
z = \mathrm{softmax}\big(\{[c_0^{L+1}]_j : j\in N(i_{m-1})\}\big)
$$

예측은

$$
\hat{i}_m = \arg\max_j z_j
$$

입니다.

---

## 2.5 왜 홀수이면서 비선형인 활성함수가 필요한가

이 논문의 가장 중요한 수학적 주장입니다.

### 홀함수여야 하는 이유: orientation equivariance

orientation이 뒤집히면 edge coefficient의 부호가 바뀝니다. 따라서 활성함수도 그 부호 반전을 일관되게 반영해야 합니다.

논문은 연속이고 원소별 적용되는 activation에 대해, orientation equivariance가 성립하려면

$$
\phi(-x) = -\phi(x)
$$

즉, $\phi$가 홀함수여야 함을 보입니다.

대표적으로 $\tanh$는 홀함수이고, sigmoid나 ReLU는 홀함수가 아닙니다.

### 비선형이어야 하는 이유: simplicial awareness

논문은 $\phi$가 선형이면, 최종 출력이 사실상 $\partial_2$의 영향을 잃어버린다고 보입니다. 즉, triangle 구조를 반영하지 못하게 됩니다.

선형 활성함수의 경우 최종 출력이 본질적으로

```math
c_0^{L+1}
=
\partial_1
\left(
I + \partial_1^\top \partial_1
\right)^L w
```

형태가 되어 $\partial_2$에 의존하지 않게 됩니다. 그러면 $2$-simplex 정보가 사라져 simplicial awareness를 잃습니다.

그래서 admissibility를 위해서는

$$
\phi \text{ is odd and nonlinear}
$$

가 필요합니다.

논문은 실용적으로 $\tanh$를 제안합니다.

---

## 2.6 성능 향상과 실험 결과

논문은 synthetic 데이터, Ocean Drifters, Berlin pathfinding 데이터에서 실험합니다.

### Synthetic 데이터

표 1(a)의 핵심 결과는 다음과 같습니다.

표준 테스트셋에서:
- RNN: $0.73$
- Markov: $0.70$
- SCoNe(tanh): $0.69$
- ker $(\Delta_1)$ projection: $0.55$
- SCNN: $0.64$
- S2CCNN: $0.62$

여기서는 RNN이 가장 높지만, 저자들은 이것이 훈련 경로 패턴을 잘 “암기”했기 때문이라고 해석합니다.

### 일반화 테스트 1: reversed trajectories

훈련은 원래 방향, 테스트는 궤적 방향을 반대로 뒤집습니다.

이 경우:
- RNN: $0.01$
- Markov: $0.24$
- ker $(\Delta_1)$ : $0.58$
- SCoNe(tanh): $0.59$
- SCNN: $0.48$
- S2CCNN: $0.47$

여기서 SCoNe와 harmonic projection이 강했고, 특히 RNN은 사실상 붕괴합니다. 이는 SCoNe가 단순 패턴 암기가 아니라 구조적 규칙을 학습했다는 저자 주장과 연결됩니다.

### 일반화 테스트 2: transfer-like setting

훈련을 복합체의 한 지역 궤적에만 제한하고, 테스트는 다른 지역 궤적에 수행합니다.

- ker $(\Delta_1)$ : $0.58$
- SCoNe(tanh): $0.61$
- SCNN: $0.42$
- S2CCNN: $0.57$

지역적으로 보지 못한 구조에 대해서도 SCoNe가 상대적으로 안정적입니다.

### 활성함수 비교

수동 orientation을 주어 훈련 방향과 align되게 만들면 표준 테스트에서는 sigmoid/ReLU도 잘 나옵니다. 하지만 reversed 테스트에서는 급락합니다.

표 1(b):
- tanh: STD $0.65$, REV $0.63$
- ReLU: STD $0.65$, REV $0.24$
- sigmoid: STD $0.66$, REV $0.10$
- identity: STD $0.27$, REV $0.31$

이 결과는 매우 중요합니다. 훈련 분포와 맞는 orientation 편향이 있을 때는 비-admissible 모델도 잘 보이지만, 분포가 바뀌면 무너집니다. 반면 $\tanh$ 기반 SCoNe는 안정적입니다.

### 실제 데이터

표 1(c):
- Ocean Drifters: SCoNe $0.50$, ker $(\Delta_1)$ $0.45$, SCNN $0.18$, S2CCNN $0.38$
- Berlin: SCoNe $0.92$, RNN $0.79$, SCNN $0.85$, S2CCNN $0.88$

실제 데이터에서도 SCoNe가 최고 성능을 보입니다.

---

## 2.7 한계

논문에서 직접 드러나는 한계는 다음과 같습니다.

첫째, 제안 모델은 **trajectory prediction on 2-dimensional simplicial complexes**에 초점이 맞춰져 있습니다. 일반적인 고차 simplicial complex나 더 다양한 태스크로의 확장은 논문에서 충분히 검증되지 않았습니다.

둘째, 입력 궤적을

$$
c_1^0 = \sum [i_j,i_{j+1}]
$$

처럼 단순 합으로 표현하므로, 순서 정보가 일부 압축됩니다. 저자도 “mostly captured”라고 표현했지 완전 보존된다고 하지는 않았습니다. 즉, 서로 다른 시퀀스가 같은 $1$-chain 표현으로 collapse될 수 있습니다.

셋째, 표준 테스트에서는 RNN이 synthetic에서 더 높았습니다. 따라서 본 논문의 장점은 “항상 최고 정확도”라기보다, **분포 이동과 보지 못한 구조에 대한 일반화** 쪽에 더 가깝습니다.

넷째, admissibility가 강한 귀납 편향을 제공하지만, 모든 문제에서 최선이라는 보장은 없습니다. 예를 들어 orientation 자체가 실제 의미를 갖는 물리계에서는 완전한 orientation equivariance가 오히려 제약이 될 수도 있습니다. 이 부분은 논문에서 깊게 다루지 않습니다.

다섯째, 최신 대규모 벤치마크나 더 다양한 transformer 기반 trajectory model과의 비교는 이 논문 범위 밖입니다.

# 3. 일반화 성능 향상 가능성 중심 분석

이 논문의 가장 큰 의미는 정확도 자체보다 **왜 일반화가 좋아질 수 있는가**를 설계 원리 수준에서 제시했다는 점입니다.

## 3.1 일반화의 근거: 대칭성 보존

논문의 핵심 논리는 다음과 같습니다.

사용자가 정하는 것은 node labeling과 simplex orientation인데, 이 둘은 데이터 생성의 본질이 아니라 **표현상의 임의성**입니다. 따라서 모델이 이 임의성에 민감하면, 훈련 데이터의 우연한 표기법을 학습하게 됩니다.

즉, permutation equivariance와 orientation equivariance를 강제하면 모델은 다음 종류의 허위 규칙을 덜 배우게 됩니다.

- 특정 노드 번호에만 의존하는 규칙
- 특정 edge 방향 코딩에만 의존하는 규칙

이런 허위 규칙을 제거하면, 모델은 보다 구조적이고 transferable한 표현을 학습할 가능성이 커집니다.

이를 수식적으로 보면, 모델이

$$
f_{\partial}(c)
$$

일 때, 좋은 일반화는 대체로

```math
f_{\partial}(c)
=
D f_{D\partial}(Dc)
\quad\text{and}\quad
f_{\partial}(c)
=
P f_{P\partial}(Pc)
```

를 만족할 때 강화됩니다. 즉, 입력 표현이 바뀌어도 본질적 계산은 유지됩니다.

## 3.2 일반화의 근거: higher-order structure 활용

단순 그래프 구조만 쓰면 삼각형이 주는 국소 위상 정보를 놓칩니다. SCoNe는

$$
\partial_2\partial_2^\top
$$

항을 통해 triangle 기반 상위 인접성을 직접 사용합니다. 이는 trajectory가 단순히 edge 연결성만으로 설명되지 않고, 국소 순환 가능성이나 hole 구조와 연결될 때 중요합니다.

특히 harmonic flow 관점에서 trajectories는

$$
\ker(\Delta_1)
$$

와 밀접한데,

$$
\Delta_1 = \partial_1^\top\partial_1 + \partial_2\partial_2^\top
$$

이므로 triangle 정보가 빠지면 harmonic structure 자체가 달라집니다. 이는 실험에서 ker $(\partial_1)$ 보다 ker $(\Delta_1)$ 가 훨씬 나은 결과를 내는 것으로도 뒷받침됩니다.

## 3.3 일반화의 근거: 비선형성이 higher-order dependency를 살린다

이 논문은 단순히 “비선형이 중요하다”가 아니라, **비선형이 없으면 triangle 정보를 최종 출력까지 전달할 수 없다**는 점을 보인 것이 중요합니다.

즉, 일반화 성능 향상은 단순한 capacity 증가 때문만이 아니라, 비선형성이 있어야만 higher-order structure가 표현력에 실질적으로 기여하기 때문입니다.

이 점은 매우 중요한데, 선형 모델은 겉으로는 simplicial operator를 쓰더라도 실제 출력은 결국 낮은 차수 구조로 환원될 수 있습니다. 이 논문은 그 함정을 명확히 지적합니다.

## 3.4 실험이 보여준 일반화 메시지

가장 설득력 있는 실험은 reversed trajectories입니다. 훈련 데이터와 테스트 데이터의 “방향 규칙”이 바뀌자, sequence memorization에 가까운 방법은 크게 무너졌고, admissible한 구조 기반 방법은 상대적으로 유지되었습니다.

이는 일반화 성능 향상 가능성을 다음처럼 해석하게 합니다.

$$
\text{Generalization}
\approx
\text{invariance/equivariance to arbitrary representation}
+
\text{use of true higher-order topology}
$$

즉, 이 논문은 “일반화는 더 큰 모델에서 오기보다, 올바른 대칭성과 구조를 반영한 설계에서 온다”는 메시지를 줍니다.

# 4. 앞으로의 연구에 미치는 영향과 고려할 점

## 4.1 미치는 영향

이 논문은 simplicial neural network 연구에 세 가지 중요한 영향을 줍니다.

### 첫째, “무엇이 좋은 simplicial network인가”에 대한 기준 제시

이전 연구들이 주로 새로운 convolution 식을 제안하는 데 집중했다면, 이 논문은 아키텍처의 바람직한 성질을 먼저 정의했습니다. 즉, 방법론보다 먼저 **설계 원칙**을 제시했습니다.

이는 이후 연구에서 단순 성능 비교를 넘어,
- permutation equivariant인가
- orientation equivariant인가
- higher-order structure를 실제로 쓰는가

를 평가 기준으로 삼게 만드는 효과가 있습니다.

### 둘째, 활성함수 선택의 중요성 부각

이 논문은 simplicial domain에서 activation function이 단순 구현 세부사항이 아니라, 모델의 수학적 정당성과 일반화 성능을 좌우하는 핵심 요소임을 보여줍니다.

특히

$$
\phi(-x)=-\phi(x)
$$

가 중요하다는 점은 이후 edge-flow, cochain, gauge-like representation을 다루는 모델들에 직접적 영향을 줄 수 있습니다.

### 셋째, trajectory prediction을 higher-order topology로 푸는 관점 강화

trajectory를 sequence로만 보는 대신, edge-flow와 harmonic structure로 해석하는 관점을 강화했습니다. 이는 교통, 해양 흐름, 이동 데이터, 경로 계획 등에서 topology-aware learning의 가능성을 넓힙니다.

---

## 4.2 앞으로 연구 시 고려할 점

### 1) 더 강한 일반화 검증

이 논문의 가장 강한 주장 중 하나가 일반화이므로, 앞으로는 더 다양한 분포 이동 설정이 필요합니다. 예를 들면:
- 복합체 크기 변화
- hole의 개수 변화
- 훈련과 테스트의 topology 변화
- 노이즈와 누락 edge가 있는 경우

즉, 같은 복합체 내부의 지역 일반화뿐 아니라, **서로 다른 simplicial complex 간 전이**를 더 엄밀히 평가해야 합니다.

### 2) 시퀀스 정보 손실 문제

현재 입력 표현

$$
c_1^0 = \sum_j [i_j,i_{j+1}]
$$

은 순서를 압축합니다. 앞으로는
- positional encoding,
- path-order aware message passing,
- temporal simplicial models

등을 결합해, topology와 sequence를 동시에 보존하는 방식이 필요합니다.

### 3) 더 고차원 구조로의 확장

논문은 주로 $2$-complex에 집중합니다. 그러나 실제 문제는 tetrahedron 이상의 상호작용을 가질 수 있습니다. 따라서 앞으로는

$$
\partial_k,\quad \Delta_k
$$

를 일반 $k$에 대해 사용하면서도 계산 효율과 안정성을 유지하는 모델이 필요합니다.

### 4) orientation equivariance의 적용 범위 점검

모든 문제에서 orientation이 임의적인 것은 아닙니다. 예를 들어 물리적 유량이나 인과 방향성이 실제 의미를 가지면, 완전한 orientation equivariance가 적절하지 않을 수 있습니다. 따라서 앞으로는
- 임의 orientation과
- 의미 있는 orientation
을 구분하는 문제 설정이 중요합니다.

### 5) 표현력과 귀납 편향의 균형

admissibility는 일반화를 도울 수 있지만, 동시에 모델 공간을 제한합니다. 앞으로는 이 제약이 어떤 태스크에서는 이득이고 어떤 태스크에서는 손해인지, 더 정교한 이론과 실험이 필요합니다.

### 6) 최신 아키텍처와의 비교 필요

이 논문은 당시 기준으로 의미 있는 비교를 했지만, 이후에는 attention, transformer, equivariant geometric learning, topological deep learning이 크게 발전했습니다. 따라서 앞으로는 더 현대적인 baseline과 비교하여, simplicial inductive bias의 고유 이점을 재검증할 필요가 있습니다.

# 5. 2020년 이후 관련 최신 연구 비교 분석

이 부분은 제공된 자료 범위 내에서만 제한적으로 정리합니다.

논문 본문과 보충자료에서 직접 비교한 2020년 전후 관련 연구는 다음입니다.

## 5.1 Ebli et al. (2020), Simplicial Neural Networks

이 연구는 simplicial convolution을 제안하지만, 이 논문 저자들은 **orientation 문제를 충분히 다루지 않았다**고 비판합니다. 보충자료에 따르면, 해당 구조도 elementwise activation이 홀함수이면 orientation equivariance를 만족할 수 있지만, 원저자들은 leaky ReLU를 사용해 이 성질을 만족하지 못했다고 지적합니다.

비교 요점:
- 장점: simplicial convolution의 초기 정립
- 한계: orientation equivariance가 설계 원리로 명시되지 않음
- 본 논문의 차별점: admissibility를 원리로 제시하고, activation 조건까지 명확히 규정

## 5.2 Bunch et al. (2020), Simplicial 2-Complex CNN

이 연구는 각 차수의 chain을 모두 유지하는 구조를 사용합니다. 역시 보충자료에서 저자들은, 이 구조도 홀함수 activation을 쓰면 admissibility와 유사한 성질을 만족할 수 있지만, 원 논문은 ReLU 계열을 사용했다고 설명합니다.

비교 요점:
- 장점: 여러 차수의 표현을 동시에 유지
- 한계: orientation-aware 설계 원리가 약함
- 본 논문의 차별점: trajectory prediction에 특화되고, 일반화 관점에서 admissibility의 실험적 이점 제시

## 5.3 이 논문의 상대적 위치

제공 자료만 기준으로 보면, 이 논문의 상대적 강점은 다음입니다.

- 단순히 simplicial convolution을 제안한 것이 아니라, 설계 원칙을 이론화함
- odd/nonlinear activation의 필요성을 명시적으로 증명함
- 일반화 성능, 특히 unseen trajectories에 초점을 맞춤

반면, 제공 자료만으로는 2021년 이후의 더 최신 논문들과의 엄밀한 비교는 할 수 없습니다. 그 부분은 별도 문헌조사가 필요합니다.

# 6. 최종 정리

이 논문은 simplicial neural network 설계에서 가장 중요한 질문을 던집니다. “고차 구조를 쓴다”는 말만으로는 충분하지 않고, 그 구조가 임의적인 표기법에 흔들리지 않으며 실제로 higher-order topology를 반영해야 한다는 것입니다.

SCoNe의 핵심 식은

```math
c_1^{\ell+1}
=
\phi\!\left(
\partial_2\partial_2^\top c_1^\ell W_2^\ell
+
c_1^\ell W_1^\ell
+
\partial_1^\top \partial_1 c_1^\ell W_0^\ell
\right)
```

이고, 이 모델이 좋은 일반화를 가지려면 활성함수는

$$
\phi(-x)=-\phi(x)
$$

를 만족하는 홀함수이면서, 동시에 higher-order dependence를 보존하기 위해 비선형이어야 합니다.

실험적으로도 이 원칙은 훈련 분포와 다른 방향, 보지 못한 지역 구조에 대한 예측에서 강점을 보였습니다. 따라서 이 논문의 가장 큰 공헌은 단순한 정확도 향상이 아니라, **simplicial deep learning의 일반화 가능성을 대칭성과 위상 구조의 관점에서 설명했다는 점**입니다.

# 참고자료 / 출처

1. Roddenberry, T. Mitchell, Nicholas Glaze, Santiago Segarra. “Principled Simplicial Neural Networks for Trajectory Prediction.” ICML 2021.  
2. 논문 보충자료(Supplementary Material for “Principled Simplicial Neural Networks for Trajectory Prediction”).  
3. GitHub 코드 링크(논문 내 기재): https://github.com/nglaze00/SCoNe_GCN  
4. Ebli, S., Defferrard, M., Spreemann, G. “Simplicial Neural Networks.” NeurIPS Workshop on Topological Data Analysis and Beyond, 2020.  
5. Bunch, E., You, Q., Fung, G., Singh, V. “Simplicial 2-Complex Convolutional Neural Networks.” NeurIPS Workshop on Topological Data Analysis and Beyond, 2020.  
6. Schaub, M. T., Benson, A. R., Horn, P., Lippner, G., Jadbabaie, A. “Random Walks on Simplicial Complexes and the Normalized Hodge 1-Laplacian.” SIAM Review, 2020.  
7. Roddenberry, T. M., Segarra, S. “HodgeNet: Graph Neural Networks for Edge Data.” Asilomar Conference on Signals, Systems, and Computers, 2019.  
8. Jia, J., Schaub, M. T., Segarra, S., Benson, A. R. “Graph-based Semi-supervised & Active Learning for Edge Flows.” KDD, 2019.  
9. Wu, H., Chen, Z., Sun, W., Zheng, B., Wang, W. “Modeling Trajectories with Recurrent Neural Networks.” IJCAI, 2017.  
10. Ghosh, A., Rozemberczki, B., Ramamoorthy, S., Sarkar, R. “Topological Signatures for Fast Mobility Analysis.” ACM SIGSPATIAL, 2018.  

> **주의**: 위 분석은 공개된 논문 및 검색 결과를 바탕으로 작성되었습니다. SCoNe의 구체적인 수치적 실험 결과(정확도 테이블의 개별 수치 등)는 원 논문의 Table 1을 직접 참조하시기 바랍니다. 수식의 세부 표기는 논문의 표기 관례에 맞추어 재구성한 것이며, 원본 PDF와 미세한 차이가 있을 수 있습니다.
