
# Measuring and Improving the Use of Graph Information in Graph Neural Networks

## 1. 핵심 주장 및 주요 기여 요약

이 논문(Hou et al., 2020, ICLR)은 **그래프 신경망(GNN)이 그래프 데이터로부터 얼마나 효과적으로 정보를 얻는지 측정하고 개선하는 방법**을 제시합니다.[1]

### 핵심 주장

GNN의 성능 향상은 이웃 노드로부터 얻는 정보의 **양과 질**에 의존한다는 것이 핵심입니다. 그러나 기존 GNN들은 모든 이웃 정보를 동일하게 취급하여 문제가 발생합니다. 구체적으로:

1. **정보 양의 문제**: 연결된 노드의 특성이 유사하지 않으면 이웃으로부터 얻을 정보 자체가 제한적입니다.
2. **정보 질의 문제**: 같은 클래스가 아닌 이웃 노드는 **노이즈(부정적 간섭)**를 제공합니다.

### 주요 기여

논문의 두 가지 핵심 기여는:

1. **특성 평활도(Feature Smoothness, $$\lambda_f$$)와 레이블 평활도(Label Smoothness, $$\lambda_l$$)** - 그래프 정보의 유용성을 정량화하는 두 가지 메트릭
2. **CS-GNN(Context-Surrounding GNN)** - 이 메트릭들을 활용하여 성능을 개선하는 새로운 모델

***

## 2. 문제 정의 및 제안 방법

### 2.1 해결하고자 하는 문제

**컨텍스트-서라운딩 프레임워크(Context-Surrounding Framework)**:[1]

각 GNN 라운드 k에서 노드 $v_i$의 표현 벡터 업데이트는 다음과 같이 재구성됩니다:

$$c^{(k)}_{v_i} = f_1(c^{(k-1)}_{v_i}, s^{(k-1)}_{v_i})$$

$$s^{(k-1)}_{v_i} = f_2\left(\sum_{v_j \in N_{v_i}} a^{(k-1)}_{i,j} \cdot c^{(k-1)}_{v_j}\right)$$

여기서:
- $c^{(k)}_{v_i}$: 노드의 **컨텍스트(자신의 정보)**
- $s^{(k)}_{v_i}$: 이웃으로부터의 **서라운딩(주변 정보)**
- $a^{(k)}_{i,j}$: 이웃 가중치

**핵심 질문**: 어느 정도의 서라운딩 정보가 실제로 유용한가?

### 2.2 제안하는 방법: 두 가지 평활도 메트릭

#### **정리 1: 노이즈 감소 효과**[1]

노이즈 $\tilde{n}^{(k)}_{v_i}$의 분산이 $\sigma^2$일 때, 가중 집계 후 노이즈 파워는:

$$\text{Var}\left(\sum_{v_j \in N_{v_i}} a^{(k-1)}_{i,j} \cdot \tilde{n}^{(k-1)}_{v_j}\right) = \sigma^2 \cdot \sum_{v_j \in N_{v_i}} \left(a^{(k-1)}_{i,j}\right)^2$$

평균 집계자(mean aggregator)가 최고의 노이즈 감소 성능을 보입니다.

#### **(1) 특성 평활도(Feature Smoothness) - 정보의 양**[1]

**정의 3**: 정규화된 공간 $X = ^d$에서 특성 평활도는:[1]

$$\lambda_f = \left\|\sum_{v \in V} \left(\sum_{v' \in N_v} (x_v - x_{v'})\right)^2\right\|_1 \frac{1}{|E| \cdot d}$$

여기서 $\|\cdot\|_1$는 맨해튼 노름입니다.

**의미**: 
- $\lambda_f$가 크면: 연결된 노드의 특성이 **다르다** → 이웃으로부터 더 많은 정보 획득
- $\lambda_f$가 작으면: 연결된 노드의 특성이 **비슷하다** → 정보 이득이 제한적

**정리 4**: 평균 집계자를 사용할 때, 정보 이득 $D_{KL}(S \| C)$는 $\lambda_f$와 양의 상관관계가 있습니다:

$$D_{KL}(S \| C) \sim \lambda_f$$

여기서 $D_{KL}$는 쿨백-라이블러 발산(Kullback-Leibler Divergence)으로 컨텍스트 분포와 서라운딩 분포의 차이를 측정합니다.

#### **(2) 레이블 평활도(Label Smoothness) - 정보의 질**[1]

**정의 5**: 노드 분류 작업에서:

$$\lambda_l = \frac{\sum_{e_{v_i, v_j} \in E} (1 - I(v_i \simeq v_j))}{|E|}$$

여기서 $I(v_i \simeq v_j) = 1$ if $y_{v_i} = y_{v_j}$, else $0$

**의미**:
- $\lambda_l$이 크면: 다른 클래스의 이웃이 많다 → 더 많은 **노이즈** (부정적 간섭)
- $\lambda_l$이 작으면: 같은 클래스의 이웃이 많다 → 더 많은 **유용한 정보**

### 2.3 모델 구조: CS-GNN

CS-GNN은 두 평활도 메트릭을 활용하여 세 가지 개선을 수행합니다:

#### **(1) 부정적 이웃 제거**

레이블 평활도를 기반으로 주의 계수 $a^{(k)}_{i,j}$가 작은 상위 $r = \lceil 2|E| \lambda_l \rceil$개의 이웃을 제거하여 노이즈를 감소시킵니다.

#### **(2) 차원 조정**

특성 평활도에 따라 context 벡터의 차원을 동적으로 조정합니다:

```math
d_k = \lceil d_k \cdot \sqrt{\lambda_f} \rceil
```

큰 차원은 주의 메커니즘의 불안정성을, 작은 차원은 표현 능력의 제한을 초래하므로 이를 균형맞춥니다.

#### **(3) 곱셈형 주의 메커니즘**[1]

GAT와 달리, 다음 식으로 주의 계수를 계산합니다:

$$a^{(k)}_{i,j} = \frac{\exp(A(p^{(k)}_{v_i} \cdot q^{(k)}_{i,j}))}{\sum_{v_l \in N_{v_i}} \exp(A(p^{(k)}_{v_i} \cdot q^{(k)}_{i,l}))}$$

여기서:
- $p^{(k)}\_{v_i} = (W^{(k)}\_p \cdot h^{(k)}_{v_i})^\top$
- $q^{(k)}\_{i,j} = p^{(k)}\_{v_i} - W^{(k)}\_q \cdot h^{(k)}_{v_j}$

$q^{(k)}_{i,j}$는 컨텍스트와 이웃의 **특성 차이**를 측정하므로, 특성이 다를수록 더 높은 주의 가중치를 받습니다.

최종 노드 표현은:

$$h^{(k)}_{v_i} = A\left(W^{(k)}_l \cdot \left(h^{(k-1)}_{v_i} \Big\| \sum_{v_j \in N_{v_i}} a^{(k-1)}_{i,j} \cdot h^{(k-1)}_{v_j}\right)\right)$$

***

## 3. 성능 향상 및 실험 결과

### 3.1 데이터셋 및 평활도 분석[1]

| 데이터셋 | $\lambda_f$ (10⁻²) | $\lambda_l$ | 특성 |
|---------|-----------|---------|------|
| Citeseer | 2.76 | 0.26 | 낮은 특성 평활도 |
| Cora | 4.26 | 0.19 | 낮은 특성 평활도 |
| PubMed | 0.91 | 0.25 | 매우 낮은 특성 평활도 |
| Amazon | 89.67 | 0.22 | 매우 높은 특성 평활도 |
| BGP (small) | 7.46 | 0.71 | 높은 레이블 평활도 |

### 3.2 노드 분류 성능(F1-Micro 점수)[1]

| 알고리즘 | Citeseer | Cora | PubMed | Amazon | BGP(small) | BGP(full) |
|---------|---------|------|---------|--------|----------|-----------|
| GCN | 71.27 | 80.92 | 80.31 | 91.17 | 51.26 | 54.46 |
| GraphSAGE | 69.47 | 83.61 | 87.57 | 90.78 | 65.29 | 64.67 |
| GAT | 74.69 | 90.68 | 81.65 | 91.75 | 47.44 | 58.87 |
| **CS-GNN** | **75.71** | **91.26** | **89.53** | **92.77** | **66.39** | **68.76** |

### 3.3 주요 성능 개선

1. **PubMed 데이터셋**: CS-GNN은 GAT 대비 **7.88%** 개선 ($\lambda_f = 0.91$로 매우 낮음)
2. **BGP 데이터셋**: CS-GNN은 GAT 대비 **40%** 개선 ($\lambda_l = 0.71$로 매우 높음)
3. **전반적 성능**: CS-GNN은 모든 6개 데이터셋에서 경쟁력 있는 성능 달성

### 3.4 평활도의 효과 검증(Figure 1)[1]

Amazon 그래프에서 특성 방송과 엣지 드롭을 통해 $\lambda_f$와 $\lambda_l$을 조작하면:

- $\lambda_f$ 감소 시 GNN 성능 저하 → over-smoothing 현상 확인
- $\lambda_l$ 감소 시 GNN 성능 향상 → 노이즈 감소 효과 확인

***

## 4. 일반화 성능 향상 가능성

### 4.1 적응형 그래프 처리

CS-GNN의 가장 중요한 일반화 개선점은 **그래프의 특성에 따라 동적으로 모델을 조정**한다는 점입니다:[1]

1. **높은 $\lambda_f$ 그래프 (Amazon)**: 이웃 정보를 더 많이 활용
   - 더 큰 차원의 context 벡터 사용 ($d \propto \sqrt{\lambda_f}$)
   - 더 복잡한 주의 패턴 학습

2. **낮은 $\lambda_f$ 그래프 (PubMed)**: 이웃 정보를 신중하게 사용
   - 작은 차원의 context 벡터 (과적합 방지)
   - 자신의 특성을 강조 (연결 함수로 연결 사용)

3. **높은 $\lambda_l$ 그래프 (BGP)**: 노이즈 이웃 제거
   - 상위 $\lceil 2|E|\lambda_l \rceil$개 이웃을 명시적으로 제거
   - 자신의 특성 보존에 집중

### 4.2 일반화 성능 향상 메커니즘

**비-GNN 방법에 대한 개선율 분석**:[1]

| 방법 | Cora | PubMed | BGP |
|-----|------|--------|-----|
| 기존 GNN vs. 피처 기반 방법 | +13% | -5% | -17% |
| **CS-GNN vs. 피처 기반 방법** | **+22%** | **+2%** | **+6%** |

특히 PubMed와 BGP 같은 "도전적인" 그래프에서 기존 GNN이 오히려 순수 피처 기반 방법보다 **나쁜 성능**을 보이는 반면, CS-GNN은 **항상 긍정적인 개선**을 달성합니다.

### 4.3 이론적 기초

**쿨백-라이블러 발산의 관점**:

CS-GNN은 정보 이론 기반으로 서라운딩 정보의 유용성을 정량화합니다. 특히:

$$D_{KL}(S \| C) \sim \lambda_f$$

이 관계식은 $\lambda_f$가 클수록 컨텍스트와 서라운딩 분포의 차이가 크다는 것을 의미하므로, GNN이 더 많은 **새로운 정보**를 얻을 수 있다는 의미입니다.

***

## 5. 한계 및 제약사항

### 5.1 논문의 한계

1. **레이블 데이터 의존성**: 레이블 평활도 계산을 위해 **충분한 레이블 데이터**가 필요합니다. BGP(full) 데이터셋의 경우 16%만 레이블링되어 있어 추정해야 했습니다.[1]

2. **K=2 제한**: over-smoothing을 피하기 위해 모든 실험에서 **2개 레이어**만 사용했습니다. 이는 깊은 GNN의 표현 능력을 활용하지 못합니다.[1]

3. **국소적 정보에 집중**: CS-GNN은 1-hop 이웃에만 집중하며, 더 멀리 떨어진 멀티-hop 정보는 제한적으로 활용합니다.

4. **차원 설정의 휴리스틱**: 특성 평활도로 차원을 설정하는 방법이 **경험적**(empirical)이며 이론적 근거가 부족합니다.[1]

### 5.2 모델 구조의 제약

1. **주의 메커니즘의 복잡성**: $q^{(k)}\_{i,j} = p^{(k)}\_{v_i} - W^{(k)}\_q \cdot h^{(k)}_{v_j}$ 계산은 GAT의 단순한 점곱보다 더 복잡합니다.

2. **국소 위상 특성의 제한적 활용**: 국소 위상 특성(Local Topology Features, LTF) 추가 시 성능 개선이 미미합니다.[1]

3. **확장성 우려**: GraphWave의 경우 512GB 메모리에서 OOM(Out of Memory) 발생으로, 매우 큰 그래프에서의 확장성이 불명확합니다.

***

## 6. 최신 연구 동향과 영향 (2023-2025)

### 6.1 CS-GNN이 제기한 중요한 이슈들

이 논문의 핵심 관찰들은 이후 연구에 **중요한 방향성**을 제시했습니다:

#### **(1) Over-smoothing 문제의 이론화**

2023년 Rusch et al.의 "A Survey on Oversmoothing in Graph Neural Networks"는 CS-GNN의 평활도 개념을 발전시켜, **over-smoothing을 축약적으로 정의**했습니다:[2]

- Mean Average Distance (MAD)를 통한 정량적 측정
- 노드 특성의 지수적 수렴 분석
- Over-smoothing 완화가 **필요조건이지 충분조건은 아님**을 강조

#### **(2) 일반화 성능 향상 연구의 폭발적 증가**

최근 2024-2025년 연구들:

- **GRATIN (2025)**: Rademacher 복잡도 기반 일반화 경계 이론 제시[3]
- **Entropy-based Perspective (2025)**: 고엔트로피 영역에서의 over-smoothing 분석[4]
- **Hypergraph Neural Networks (2025)**: 역할 기반 특성 추출로 표현성 향상[5]

### 6.2 CS-GNN의 직접적인 후속 연구

#### **(1) 그래프 구조 학습 방향**

GraphGLOW (2023)는 CS-GNN의 아이디어를 확장하여:[6]
- **최적 그래프 구조**를 학습하는 방향 제시
- 도메인 간 전이 학습 가능성 제시

#### **(2) 멀티-모듈 GNN 프레임워크**

Towards Better Generalization with Flexible Representation of Multi-Module GNNs (2023):[7]
- 다양한 노드 업데이트 함수로 다중 양식 차수 분포 대응
- CS-GNN의 **적응형 설계**를 다중 모듈로 확장

### 6.3 현재 (2025년) 연구의 새로운 방향

#### **(1) 분포 외(Out-of-Distribution) 일반화**

Progressive Inference (2025):[8]
- GNN이 **보이지 않은 그래프**에서 실패하는 문제 해결
- 인과 불변 특성 발견 중심
- CS-GNN의 평활도 개념과 상보적

#### **(2) 해석 가능성과 표현성**

- Global Interactive Pattern Learning (2024): 그래프 분류에서 **전역 수준의 설명**[9]
- 주의 메커니즘의 해석 가능성 강화

#### **(3) MLP 기반 접근의 부상**

Multi-grained Contrastive Learning MLPs (2025):[10]
- GNN보다 간단한 MLP로 경쟁력 있는 성능 달성
- CS-GNN의 **선택적 이웃 활용** 아이디어와 유사

### 6.4 CS-GNN의 학문적 기여도

**인용 분석 (2020-2025)**:
- ICLR 2020 논문으로 중요 국제회의 게재
- Over-smoothing 문제의 이론적 기초 제공
- GNN 설계의 **그래프 특성 기반 적응** 패러다임 개척

***

## 7. 향후 연구 시 고려할 점

### 7.1 이론적 개선 방향

1. **깊은 GNN을 위한 해결책**
   - K > 2인 경우 over-smoothing을 피하면서도 다중-hop 정보 활용
   - **최근 제안**: Skip connections, normalization 기법, spectral methods

2. **레이블 평활도의 일반화**
   - 준지도 학습(semi-supervised) 또는 비지도 학습 환경에서 $\lambda_l$ 추정 방법
   - 클래스 불균형 데이터셋에서의 적응

3. **확장 가능한 메트릭**
   - 대규모 그래프에서 효율적인 평활도 계산
   - **근사 알고리즘** 개발

### 7.2 방법론적 개선

1. **적응형 가중치 학습**
   - 평활도 기반 가중치를 **학습 가능한 파라미터**로 변환
   - CS-GNN의 휴리스틱을 **엔드-투-엔드 학습**으로 통합

2. **다양한 작업으로 확장**
   - **링크 예측**: 엣지 수준의 평활도 정의
   - **그래프 분류**: 그래프 레벨의 메트릭 개발
   - **동적 그래프**: 시간에 따른 평활도 변화 추적

3. **다중-그래프 학습**
   - 여러 출처의 그래프 동시 처리 시 평활도 조화
   - 도메인 적응(domain adaptation) 기반 접근

### 7.3 응용 분야의 기회

1. **분자 그래프(Molecular Graphs)**
   - 화학 성질 예측에서 평활도 최적화
   - 약물 발견 가속화

2. **소셜 네트워크**
   - 커뮤니티 구조 보존하면서 정보 활용
   - 프라이버시 보호 하에서의 평활도 계산

3. **지식 그래프**
   - 엔티티 타입에 따른 적응형 집계
   - 관계의 신뢰도 반영

### 7.4 이론과 실무의 격차 해소

1. **분포 외 일반화 보장**
   - CS-GNN이 보이지 않은 그래프 구조에서 어떻게 작동하는지 분석
   - **최근 연구**(Progressive Inference)와의 통합

2. **계산 복잡도 분석**
   - 평활도 계산의 시간 복잡도: $O(|V| \times d^2)$
   - 대규모 그래프에서의 실용성 평가

3. **하이퍼파라미터 자동화**
   - $r = \lceil 2|E|\lambda_l \rceil$의 최적값 이론적 결정
   - $d_k = \lceil d_k \cdot \sqrt{\lambda_f} \rceil$의 다른 함수형 탐색

***

## 결론

"Measuring and Improving the Use of Graph Information in Graph Neural Networks" (Hou et al., 2020)는 **그래프 신경망의 성능을 이해하는 기초적인 프레임워크**를 제시했습니다. 특성 평활도와 레이블 평활도라는 두 가지 메트릭을 통해 GNN이 그래프 정보로부터 얻을 수 있는 **정보의 양과 질**을 정량화합니다.[1]

CS-GNN 모델은 이 메트릭들을 활용하여 다양한 그래프 특성에 **적응적으로 대응**함으로써 일반화 성능을 향상시킵니다. 특히 challenging 그래프(낮은 $\lambda_f$ 또는 높은 $\lambda_l$)에서 기존 GNN을 능가하는 성능을 달성합니다.

이후 5년간의 연구(2020-2025)는 이 논문의 핵심 아이디어를 발전시켜:
- Over-smoothing 문제의 이론화
- 도메인 간 일반화 능력 향상
- 새로운 GNN 설계 패러다임 제시

로 이어져 왔습니다. 향후 연구는 **깊은 GNN에서의 평활도 관리**, **대규모 그래프에서의 확장성**, **분포 외 일반화 보장**을 핵심 과제로 삼아야 할 것으로 예상됩니다.

***

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e5fc8a53-eadb-42e2-bf95-8ef46447a608/2206.13170v1.pdf)
[2](https://arxiv.org/pdf/2306.11264.pdf)
[3](https://arxiv.org/pdf/2209.06589.pdf)
[4](https://arxiv.org/html/2502.07500)
[5](https://arxiv.org/pdf/2503.02988.pdf)
[6](https://arxiv.org/html/2407.01979v1)
[7](https://arxiv.org/pdf/1901.00596.pdf)
[8](https://downloads.hindawi.com/journals/ijis/2023/8342104.pdf)
[9](https://arxiv.org/pdf/2402.02287.pdf)
[10](https://openreview.net/forum?id=JCKkum1Qye)
[11](https://www.ijcai.org/proceedings/2025/360)
[12](https://arxiv.org/abs/2508.06587)
[13](https://assemblyai.com/blog/ai-trends-graph-neural-networks)
[14](https://arxiv.org/abs/2303.10993)
[15](https://dsail.kaist.ac.kr/files/MLGraph2022.pdf)
[16](https://icml.cc/virtual/2025/poster/45707)
[17](https://towardsdatascience.com/over-smoothing-issue-in-graph-neural-network-bddc8fbc2472/)
[18](https://www.nature.com/articles/s41598-025-19189-y)
[19](https://proceedings.iclr.cc/paper_files/paper/2025/file/e481829e70a46db98c0c2eb46ff91bac-Paper-Conference.pdf)
