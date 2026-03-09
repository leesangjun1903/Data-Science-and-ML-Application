# Computing Persistent Homology

---

## 1. 핵심 주장과 주요 기여 요약

필터링된 $d$-차원 단체 복합체(filtered simplicial complex)의 지속적 호몰로지(persistent homology)가 다항식 환(polynomial ring) 위의 특정 등급 모듈(graded module)의 표준 호몰로지에 해당함을 보여주었다. 이 분석은 임의의 체(field) 위에서 지속적 호몰로지 군의 간단한 기술(simple description)이 존재함을 확립하고, 임의의 체 위에서 임의의 차원에서의 지속적 호몰로지를 계산하는 자연스러운 알고리즘을 유도한다. 이 결과는 이전에 $S^3$의 부분 복합체와 $\mathbb{Z}_2$ 계수에 한정되었던 알고리즘을 일반화하고 확장한다. 비-체(non-field) 위에서는 간단한 분류가 존재하지 않음을 증명하였으나, 임의의 주 아이디얼 정역(PID) 위에서 개별 지속적 호몰로지 군을 계산하는 알고리즘을 제시하였다.

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존 위상적 데이터 분석에서, 데이터의 형태학적(topological) 특징을 다중 스케일에서 추적하는 것은 핵심 과제이다. 지속적 호몰로지(PH)는 위상적 데이터 분석(TDA)에서 여러 스케일에 걸쳐 지속되는 데이터의 정성적 특징을 연구하기 위해 사용되는 방법으로, 입력 데이터의 섭동에 강건하고, 차원과 좌표에 독립적이며, 입력의 정성적 특징에 대한 압축된 표현을 제공한다.

**이전의 한계:**
- Edelsbrunner, Letscher, Zomorodian (2002)의 기존 알고리즘은 $S^3$의 부분 복합체 및 $\mathbb{Z}_2$ 계수에만 적용 가능했음
- 임의 차원, 임의 체(field) 계수에 대한 일반적 계산 알고리즘 부재
- 지속적 호몰로지의 대수적 구조에 대한 체계적인 이해 부족

### 2.2 제안하는 방법 (수식 포함)

#### (a) 필터링된 복합체 (Filtered Complex)

공간 $K$가 단체 복합체일 때, 필터링(filtration)은 다음과 같은 중첩된 부분 복합체의 열이다:

$$\emptyset = K_0 \subseteq K_1 \subseteq K_2 \subseteq \cdots \subseteq K_m = K$$

$K_i = K_m$ for all $i \geq m$이면 $K$를 필터링된 복합체(filtered complex)라 부른다.

#### (b) 호몰로지 군 (Homology Group)

$k$차 사슬 군(chain group) $C_k$에서 경계 연산자 $\partial_k: C_k \to C_{k-1}$가 정의되며:

$$Z_k = \ker(\partial_k), \quad B_k = \text{im}(\partial_{k+1})$$

$k$차 호몰로지 군은 $H_k = Z_k / B_k$이며, 그 원소는 동류(homologous) 순환의 클래스이다.

#### (c) 지속적 호몰로지의 대수적 구조 — Persistence Module

논문의 핵심 기여는 필터링된 복합체의 호몰로지를 **등급 모듈(graded module)**로 통합적으로 기술한 것이다. 지속적 호몰로지 모듈은 다항식 환 $R = F[t]$ ($F$는 체, $t$는 형식 변수) 위의 등급 모듈로 정의된다:

$$\mathcal{M} = \bigoplus_{i \geq 0} H_k(K_i)$$

여기서 $t$의 작용은 포함 사상 $K_i \hookrightarrow K_{i+1}$에 의해 유도된 호몰로지 사상이다.

#### (d) 구조 정리 (Structure Theorem)

PID 위의 등급 모듈에 대한 구조 정리(structure theorem for graded modules over PID)를 적용하여, 필터링의 모든 지속 쌍(persistence pairs)에 대한 지식이 지속적 모듈을 완전히 특성화함을 증명하였다.

체 $F$ 위에서 $F[t]$-모듈의 분해는 다음과 같다:

$$\mathcal{M} \cong \left( \bigoplus_{i=1}^{n} \Sigma^{\alpha_i} F[t] \right) \oplus \left( \bigoplus_{j=1}^{m} \Sigma^{\gamma_j} F[t]/(t^{n_j}) \right)$$

여기서:
- $\Sigma^{\alpha}$는 등급(grading)의 $\alpha$-이동(shift)
- 왼쪽 직합 부분은 **자유 부분(free part)** — "영원히 지속되는" 호몰로지 클래스
- 오른쪽 직합 부분은 **비틀림 부분(torsion part)** — 유한한 지속 시간 $n_j$를 가지는 호몰로지 클래스

구조 정리는 구조를 자유 부분(왼쪽)과 비틀림 부분(오른쪽)으로 분해하며, 후자의 비틀림 원소들이 지속 시간을 특성화한다.

각 비틀림 항 $\Sigma^{\gamma_j} F[t]/(t^{n_j})$는 시간 $\gamma_j$에서 "탄생(birth)"하고 $\gamma_j + n_j$에서 "사멸(death)"하는 호몰로지 특징을 나타낸다. 이를 **지속 다이어그램(persistence diagram)** 또는 **바코드(barcode)**로 시각화한다:

$$\text{Persistence pair: } (\gamma_j, \, \gamma_j + n_j), \quad \text{persistence} = n_j$$

#### (e) 알고리즘

알고리즘의 핵심은 경계 행렬(boundary matrix)에 대한 **열 축소(column reduction)** 과정이다:

1. 경계 연산자를 등급 행렬(graded matrix) $M_k$로 표현:

$$M_k(i,j) = t^{\deg(\sigma_j) - \deg(\tau_i)} \cdot [\tau_i : \sigma_j]$$

여기서 $[\tau_i : \sigma_j]$는 $\sigma_j$의 경계에서 $\tau_i$의 계수이며, $\deg$는 필터링에서의 차수를 나타낸다.

2. 열-사다리꼴 형태(column-echelon form)로 변환하여 체 위에서의 계산 알고리즘을 유도하고, 비-체의 경우에는 개별 지속적 군을 계산하는 알고리즘을 기술한다.

3. 시간 복잡도: 단체(simplex)의 수 $m$에 대해 $O(m^3)$ (worst-case 행렬 축소).

### 2.3 모델 구조

이 논문은 전통적 의미의 "모델"이 아닌, **대수적-위상적 계산 프레임워크**를 제시한다:

```
입력: 필터링된 단체 복합체 K = (K_0 ⊆ K_1 ⊆ ... ⊆ K_m)
          ↓
    경계 행렬 M_k 구성 (등급 다항식 환 위)
          ↓
    열 축소 (Column Reduction)
          ↓
    지속 쌍 (birth, death) 추출
          ↓
출력: 지속 다이어그램 / 바코드
```

### 2.4 성능 향상

- **일반성(Generality):** 이전에 $S^3$의 부분 복합체와 $\mathbb{Z}_2$ 계수로 제한되었던 알고리즘을 임의 차원, 임의 체 위로 확장하였다.
- **이론적 완전성:** 필터링의 모든 지속 쌍이 지속적 모듈을 완전히 특성화하므로, 서로 다른 지속 쌍을 가진 필터링은 반드시 비동형(non-isomorphic) 지속적 모듈을 가진다.
- **PID 위의 계산:** 체가 아닌 PID(예: $\mathbb{Z}$) 위에서도 개별 지속적 호몰로지 군을 계산하는 알고리즘을 제공.

### 2.5 한계

1. **시간 복잡도:** 표준 알고리즘으로 점 구름의 지속적 호몰로지를 계산하는 것은 최악의 경우 단체 수에 대해 3차(cubic) 복잡도를 가진다.
2. **비-체 위의 분류 불가:** 비-체 위에서는 간단한 분류가 존재하지 않는다. 이는 $\mathbb{Z}$ 계수에서의 비틀림 현상 때문이다.
3. **단체 복합체 크기 폭발:** Čech 필터링의 포괄적 특성으로 인해 단체 수가 입력 점의 수에 대해 지수적으로 증가한다.
4. 지속적 호몰로지는 높은 수준의 추상화, 비-위상적 변화에 대한 둔감성, 점 구름 데이터로의 제한 등 여러 한계를 가진다.
5. 지속적 호몰로지는 전역적(global)이어서, 위상적 불변량이 일반적으로 전체 데이터 집합에 대한 것이므로 국소화된 모델이 필요한 경우 한계가 있다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 지속적 호몰로지와 신경망 일반화

최근 연구들은 지속적 호몰로지를 활용하여 딥 러닝 모델의 일반화 성능을 분석하고 향상시키는 방향으로 발전하고 있다:

**① Persistent Homology Dimension (PHD) — Birdal et al. (NeurIPS 2021)**

학습 이론과 TDA를 새롭게 연결하여, 일반화 오류가 "지속적 호몰로지 차원(persistent homology dimension, PHD)"이라는 개념으로 동등하게 상한될 수 있음을 보였으며, 기존 연구 대비 훈련 역학에 대한 추가적인 기하학적 또는 통계적 가정을 요구하지 않는다. TDA 도구를 활용하여 현대 딥 신경망 규모에서 PHD를 효율적으로 추정하는 알고리즘을 개발하였고, 제안된 접근법이 다양한 설정에서 네트워크의 내재적 차원을 효율적으로 계산할 수 있으며, 이것이 일반화 오류를 예측함을 보여주었다.

**② Topological Regularization (2023)**

소규모 표본(small-sample-size) 환경에서 과파라미터화된 딥 신경망의 일반화가 어렵다는 문제에 대해, 특징 추출기에 의해 유도된 푸시-포워드 확률 측도를 연구하고, 지속적 호몰로지를 통해 이 측도의 "분리(separation)" 특성을 처음으로 특성화하였으며, 이 특성을 강제하면 더 나은 일반화로 이어짐을 이론적으로 증명하였다.

**③ PH를 활용한 일반화 갭 추정 (ICONIP 2023)**

대수적 위상학과 관련성 측도를 활용하여 학습 중 신경망 동작을 조사하고, 신경망을 위상 공간 위의 함수적 위상 그래프로 정의하여 위상적 요약을 계산함으로써 일반화 갭을 추정한다. 이를 통해 과적합을 식별하고 적시에 조기 중지(early-stopping) 결정을 내릴 수 있다.

**④ PH를 활용한 검증 세트 없는 일반화 모니터링 (NeurIPS 2021 Workshop)**

신경망의 학습을 지속적 호몰로지(PH)로 연구하며, 단체 복합체 표현을 사용하여 연속적인 신경망 상태 간의 PH 다이어그램 거리 변화가 검증 정확도와 상관됨을 보여, 검증 세트 없이도 일반화 오류를 내재적으로 추정할 수 있음을 시사한다.

### 3.2 위상적 딥 러닝 (Topological Deep Learning)

위상적 딥 러닝(TDL)은 그래프 및 단체 복합체와 같은 복잡한 데이터 구조에서 위상적 특징을 통합하여 신경망의 능력을 향상시키지만, 구조적 섭동이 모델의 안정성과 일반화에 어떤 영향을 미치는지에 대한 중요한 갭이 존재하며, 위상적 노이즈를 처리할 때 과적합과 취약한 예측이 우려된다.

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 이 논문이 향후 연구에 미치는 영향

Zomorodian–Carlsson (2005)은 TDA의 이론적·알고리즘적 토대를 놓은 **기초 논문(foundational paper)**이다:

1. **지속적 호몰로지의 대수적 정립:** "persistence module = graded module over $F[t]$"이라는 핵심 통찰은 이후 모든 TDA 연구의 대수적 기초가 되었다.
2. **다중 파라미터 지속성(Multiparameter Persistence)으로의 확장:** 다중 파라미터 지속적 호몰로지는 Carlsson & Zomorodian (2007)에서 처음 탐색되었으며, 이상치(outlier)나 밀도 변화가 있는 데이터, 실수값 함수가 부여된 데이터, 큰 국소 노이즈가 있는 함수형 데이터 분석에 적합하다. 단, 다중 파라미터 필터링에서는 지속적 호몰로지의 기본 정리가 더 이상 유효하지 않으며 "일반화된 구간"이 존재하지 않는다.
3. **소프트웨어 생태계 발전:** JavaPlex, Perseus, Dipha, Dionysus, jHoles, GUDHI, Rivet, Ripser, PHAT, R-TDA 등 다양한 오픈소스 소프트웨어가 개발되었다.
4. **응용 분야 확장:** 지속적 호몰로지 계산은 이미지 분석, 패턴 비교·인식, 네트워크 분석, 컴퓨터 비전, 계산 생물학, 종양학, 화학 구조 등 다양한 분야에 적용되고 있다.

### 4.2 향후 연구 시 고려할 점

| 고려사항 | 설명 |
|----------|------|
| **계산 효율성** | $O(m^3)$의 기본 복잡도를 극복하기 위한 알고리즘 최적화 필요 |
| **대규모 데이터** | 분산/병렬 컴퓨팅, GPU 가속 등 활용 |
| **다중 파라미터** | 바코드 표현의 일반화 불가 → 새로운 불변량 연구 |
| **비-위상적 정보** | PH만으로는 기하학적/조합론적 정보를 포착하지 못함 |
| **딥러닝 통합** | PH 특징을 미분 가능하게 만들어 end-to-end 학습에 통합 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 계산 효율성 향상

| 연구 | 기여 | 논문 |
|------|------|------|
| **Ripser (Bauer, 2021)** | Vietoris–Rips 지속적 바코드 계산을 위해 공경계 연산자와 필터링 순서의 암시적 표현에 기반하며, apparent pair를 활용하는 방법으로 시간과 메모리 사용 모두에서 이전 소프트웨어 대비 상당한 개선을 달성하였다. | *J. Appl. Comput. Topol.* 5(3), 2021 |
| **Ripser++ (Zhang et al., 2020)** | GPU 가속 소프트웨어 Ripser++를 개발하여, 원래 Ripser 대비 총 실행 시간에서 최대 30배 속도 향상과 CPU 메모리 사용을 최대 2.0배 절감하였다. | SoCG 2020 |
| **분산 PH 계산 (2024)** | 지속적 호몰로지는 전자 현미경이나 차세대 망원경과 같은 고해상도 이미징 장치로 얻은 이미지에서 관련 통찰을 자동으로 추출하는 강력한 수학적 방법이다. 대규모 분산 계산 접근법 제안. | *J. Supercomputing*, 2024 |

### 5.2 PH의 이론적 일반화

| 연구 | 기여 | 비교 |
|------|------|------|
| **Persistent Laplacian (Wang et al., 2020)** | 지속적 호몰로지에 비해 더 깊은 데이터 분석이 가능하며, 조화 스펙트럼(영 고유값)이 PH의 위상적 출력을 완전히 복원하는 한편, 비-조화 스펙트럼(비영 고유값)이 데이터 형태의 추가적인 기하학적/조합론적 정보를 포착하여, PH가 감지하지 못하는 필터링에서의 호모토피적 기하학적 형태 변화를 기술한다. | Zomorodian-Carlsson의 PH가 놓치는 기하학적 정보를 스펙트럼 방법으로 보완 |
| **Persistent Path Homology (Chowdhury & Mémoli, 2018; Dey et al., 2022)** | 비대칭적 방향 정보를 포함하는 방향 네트워크 분석을 위해, TDA의 필터링 방법과 경로 호몰로지 이론을 통합한 지속적 경로 호몰로지를 도입하였다. | 방향 그래프에서의 PH 일반화 |
| **Persistent Hypergraph Homology (Bressan et al., 2019)** | 표준 지속적 단체 호몰로지를 하이퍼그래프로 일반화하였으며, 임베디드 호몰로지 이론을 사용하여 불완전한 정보가 있는 시스템(예: 공동 저자 네트워크에서 3인 협업은 존재하지만 모든 쌍별 협업이 존재하지 않는 경우)을 처리하는 데 적합하다. | 단체 복합체 → 하이퍼그래프로 확장 |

### 5.3 딥 러닝과의 융합

| 연구 | 기여 |
|------|------|
| **PHD와 일반화 (Birdal et al., NeurIPS 2021)** | PH 차원으로 일반화 오류 상한 설정; 검증 세트 불필요한 모니터링 가능 |
| **Topological Regularization (2023)** | PH 기반 분리 조건을 정규화 항으로 도입 → 일반화 향상 이론적 증명 |
| **TDA Beyond PH (Wei et al., 2025)** | TDA는 응용 수학과 데이터 과학에서 빠르게 발전하는 분야로 PH가 주요 도구이며, TDL과 결합하여 과학, 공학, 의학, 산업 분야에서 큰 성공을 거두었지만, 높은 수준의 추상화, 비-위상적 변화에 대한 둔감성 등의 한계로 인해 PH를 넘어서는 포괄적 리뷰를 제시한다. |
| **고차원 PH를 위한 스펙트럴 방법 (2024)** | 유클리드 거리 외에 12가지 거리를 PH 입력으로 조사하였으며, 노이즈와 이상치 하에서 PH를 수행하기 위한 최신 접근법으로 Fermat 거리 등을 활용한다. |

### 5.4 핵심 비교 요약 도표

```
Zomorodian-Carlsson (2005)
  │
  ├─ 이론 확장 ──→ Multiparameter PH (Carlsson-Zomorodian, 2007/2009)
  │                  Zig-zag PH (Carlsson-De Silva, 2010)
  │                  Persistent Path Homology (2018, 2022)
  │
  ├─ 계산 가속 ──→ Ripser (Bauer, 2021)
  │                  Ripser++ GPU (Zhang et al., 2020)
  │                  Distributed PH (2024)
  │
  ├─ 정보 보완 ──→ Persistent Laplacian (2020)
  │                  Persistent Sheaf Laplacian (2025)
  │                  Persistent Hypergraph Homology (2019)
  │
  └─ DL 융합 ───→ PHD & Generalization (NeurIPS 2021)
                     Topological Regularization (2023)
                     TDL Robustness Framework (2025)
```

---

## 참고자료

1. **Zomorodian, A., Carlsson, G.** "Computing Persistent Homology." *Discrete & Computational Geometry* 33, 249–274 (2005). — [Springer](https://link.springer.com/article/10.1007/s00454-004-1146-y)
2. **Otter, N. et al.** "A roadmap for the computation of persistent homology." *EPJ Data Science* 6:17 (2017). — [PMC/Springer](https://pmc.ncbi.nlm.nih.gov/articles/PMC6979512/)
3. **Bauer, U.** "Ripser: efficient computation of Vietoris–Rips persistence barcodes." *J. Appl. Comput. Topol.* 5(3), 391–423 (2021). — [Springer](https://link.springer.com/article/10.1007/s41468-021-00071-5)
4. **Zhang, S., Xiao, M., Wang, H.** "GPU-Accelerated Computation of Vietoris-Rips Persistence Barcodes." *SoCG 2020*. — [Dagstuhl](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.SoCG.2020.70)
5. **Birdal, T. et al.** "Intrinsic Dimension, Persistent Homology and Generalization in Neural Networks." *NeurIPS 2021*. — [arXiv:2111.13171](https://arxiv.org/abs/2111.13171)
6. **Barbara, A., Bennani, Y., Karkazan, J.** "On the Use of Persistent Homology to Control the Generalization Capacity of a Neural Network." *ICONIP 2023*. — [Springer](https://link.springer.com/chapter/10.1007/978-981-99-8132-8_21)
7. **Corneanu, C. et al.** "Persistent Homology Captures the Generalization of Neural Networks Without A Validation Set." *NeurIPS 2021 Workshop*. — [OpenReview](https://openreview.net/forum?id=BM64dm9HvN)
8. **Wei et al.** "Topological data analysis and topological deep learning beyond persistent homology: a review." *Artificial Intelligence Review* (2025). — [Springer](https://link.springer.com/article/10.1007/s10462-025-11462-w)
9. **Topological Regularization for Representation Learning via Persistent Homology.** *Mathematics* 11(4), 1008 (2023). — [MDPI/ResearchGate](https://www.researchgate.net/publication/368583212)
10. **Persistent homology classification algorithm.** *PeerJ Comput. Sci.* 9:e1195 (2023). — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10280283/)
11. **Ichinomiya, T.** "Machine learning of time series data using persistent homology." *Scientific Reports* 15, 20508 (2025). — [Nature](https://www.nature.com/articles/s41598-025-06551-3)
12. **A unified spectral-persistent homology framework for stable and generalizable topological deep learning.** *Discover Computing* (2025). — [Springer](https://link.springer.com/article/10.1007/s10791-025-09783-z)
13. Stanford 원본 PDF — [geometry.stanford.edu](https://geometry.stanford.edu/lgl_2024/papers/zc-cph-04/zc-cph-04.pdf)
14. **A Review of TDA and TDL in Molecular Sciences.** *J. Chem. Inf. Model.* (2025). — [ACS](https://pubs.acs.org/doi/10.1021/acs.jcim.5c02266)
