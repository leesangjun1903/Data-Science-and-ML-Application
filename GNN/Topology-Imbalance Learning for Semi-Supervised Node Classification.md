# Topology-Imbalance Learning for Semi-Supervised Node Classification

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

본 논문은 그래프 기반 반지도 노드 분류(Semi-Supervised Node Classification)에서 기존 연구들이 **수량 불균형(Quantity Imbalance, QINL)** 만을 다뤄왔다는 한계를 지적하고, 그래프 고유의 새로운 불균형 문제인 **위상 불균형(Topology Imbalance, TINL)** 을 최초로 정식 제안합니다.

> **핵심 주장**: 레이블된 노드의 수(quantity)뿐 아니라 **그래프 내 레이블 노드의 위치(topological position)** 도 모델의 결정 경계(decision boundary)를 왜곡시킨다.

### 주요 기여

| 기여 | 설명 |
|------|------|
| **TINL 문제 정의** | 위상 불균형 문제를 최초로 그래프 특화 불균형 학습 문제로 규정 |
| **Totoro 메트릭** | Influence Conflict 기반의 위상적 위치 측정 지표 고안 |
| **ReNode 방법론** | 레이블 노드 위치에 따른 훈련 가중치 재조정 model-agnostic 프레임워크 제안 |
| **통합 분석 프레임워크** | LP(Label Propagation)를 통해 TINL과 QINL을 통합 분석 |
| **GNN 평가 신관점** | 위상 불균형 민감도를 GNN 아키텍처 평가의 새 기준으로 제시 |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

#### 위상 불균형(Topology Imbalance)의 세 가지 특성

1. **Ubiquity (편재성)**: 그래프의 복잡한 연결 구조로 인해 서로 다른 클래스의 노드들은 자연스럽게 비대칭적인 위상 구조를 가지며, 완전히 대칭적인 레이블 집합 구성이 사실상 불가능

2. **Perniciousness (유해성)**: 레이블 노드의 영향력은 위상적 거리에 따라 감소하므로, 클래스 경계 근처의 레이블 노드는 **영향 충돌(Influence Conflict)** 을, 레이블 노드에서 멀리 떨어진 노드는 **영향 부족(Influence Insufficient)** 문제를 야기

3. **Orthogonality (직교성)**: QINL은 클래스 단위의 총 레이블 수를 다루는 반면, TINL은 각 레이블 노드의 개별 위치를 다루므로 두 문제는 독립적(orthogonal)

#### 핵심 문제 현상

$$\text{결정 경계 이동} \leftarrow \text{노드 영향력 경계} \neq \text{실제 클래스 경계}$$

수량 균형(Quantity-balanced) 상태에서도 위상 불균형은 여전히 존재하며, 이 경우 소수 클래스의 레이블 노드가 경계 근처에 집중되어 영향력 경계가 대형 클래스 방향으로 이동합니다.

---

### 2.2 제안하는 방법 (수식 포함)

#### Step 1: Label Propagation을 통한 통합 분석

레이블 전파 수렴 결과 $\mathbf{Y}$:

$$\mathbf{Y} = \alpha(\mathbf{I} - (1-\alpha)\mathbf{A}')^{-1}\mathbf{Y}^0 \tag{1}$$

- $\mathbf{I}$: 단위 행렬
- $\alpha \in (0,1]$: 랜덤 워크 재시작 확률
- $\mathbf{A}' = \mathbf{D}^{-\frac{1}{2}}\mathbf{A}\mathbf{D}^{-\frac{1}{2}}$: 정규화된 인접 행렬
- $\mathbf{Y}^0$: 초기 레이블 분포 (레이블 노드는 one-hot 벡터)

LP의 예측 결과 $q_i = \arg\max_j Y_{ij}$는 GNN의 결정 경계와 높은 일관성을 보이므로(t-SNE 시각화로 확인), LP를 통해 TINL과 QINL을 통합적으로 분석하는 프레임워크를 제공합니다.

#### Step 2: Totoro 메트릭 — 위상 불균형 측정

Personalized PageRank 행렬 $\mathbf{P}$:

$$\mathbf{P} = \alpha(\mathbf{I} - (1-\alpha)\mathbf{A}')^{-1} \tag{2}$$

노드 $v$의 Totoro 값 (영향 충돌 기댓값):

$$T_v = \mathbb{E}_{x \sim P_{v,:}}\left[\sum_{j \in [1,k], j \neq y_v} \frac{1}{|\mathcal{C}_j|} \sum_{i \in \mathcal{C}_j} P_{i,x}\right] \tag{3}$$

- $y_v$: 노드 $v$의 실제 클래스 레이블
- $P_{v,:}$: 노드 $v$의 PPR 확률 벡터 (랜덤 워크 분포)
- $\mathcal{C}_j$: $j$번째 클래스의 레이블 노드 집합
- $\frac{1}{|\mathcal{C}_j|}$: 클래스 간 영향력 비교를 위한 정규화 항

> **해석**: $T_v$가 클수록 노드 $v$는 클래스 경계에 가까이 위치하며, 작을수록 클래스 중심에 가까이 위치

데이터셋 수준의 전체 위상 불균형 지표:

$$\text{Overall Conflict} = \sum_{b \in \mathcal{L}} T_b$$

실험에서 Overall Conflict와 GNN 성능 사이 Pearson 상관계수 = **-0.618** (p < 0.01)으로 유의한 음의 상관관계 확인

#### Step 3: ReNode — 인스턴스별 노드 가중치 재조정

코사인 어닐링 방식의 훈련 가중치 스케줄:

$$w_v = w_{\min} + \frac{1}{2}(w_{\max} - w_{\min})\left(1 + \cos\left(\frac{\text{Rank}(T_v)}{|\mathcal{L}|}\pi\right)\right), \quad v \in \mathcal{L} \tag{4}$$

- $w_v$: 레이블 노드 $v$의 훈련 가중치
- $w_{\min}, w_{\max}$: 가중치 보정 인수의 하한 및 상한 하이퍼파라미터
- $\text{Rank}(T_v)$: $T_v$를 오름차순으로 정렬했을 때의 순위

**TINL 전용 훈련 손실** $L_T$:

$$L_T = -\frac{1}{|\mathcal{L}|}\sum_{v \in \mathcal{L}} w_v \sum_{c=1}^{k} y_v^{*c} \log g_v^c, \quad \mathbf{g} = \text{softmax}(\mathcal{F}(\mathbf{X}, \mathbf{A}, \theta)) \tag{5}$$

**TINL + QINL 복합 훈련 손실** $L_Q$:

$$L_Q = -\frac{1}{|\mathcal{L}|}\sum_{v \in \mathcal{L}} w_v \frac{\bar{|\mathcal{C}|}}{|\mathcal{C}_j|} \sum_{c=1}^{k} y_v^{*c} \log g_v^c \tag{6}$$

- $\bar{|\mathcal{C}|}$: 클래스별 훈련 크기의 평균
- $\frac{\bar{|\mathcal{C}|}}{|\mathcal{C}_j|}$: 수량 불균형에 대한 클래스 빈도 기반 재가중치 항

**대규모 그래프용** (PPRGo 기반):

$$\mathbf{g}' = \text{softmax}(\hat{\mathbf{P}}\mathcal{F}'(\mathbf{X}, \theta')) \tag{7}$$

---

### 2.3 모델 구조

ReNode는 특정 GNN 아키텍처에 종속되지 않는 **model-agnostic** 훈련 프레임워크입니다:

```
[그래프 G = (V, E, L) 입력]
        ↓
[PPR 행렬 P 계산 (Eq. 2)]
        ↓
[Totoro 값 T_v 계산 (Eq. 3) → 각 레이블 노드의 위상적 위치 측정]
        ↓
[ReNode 가중치 w_v 계산 (Eq. 4) → 코사인 어닐링]
        ↓
[임의의 GNN 인코더 F(X, A, θ) → g = softmax(F)]
        ↓
[재가중된 손실 L_T 또는 L_Q로 훈련 (Eq. 5 or 6)]
        ↓
[노드 분류 예측]
```

**지원 GNN 모델**: GCN, GAT, PPNP, GraphSAGE, ChebGCN, SGC (실험에서 검증)

---

### 2.4 성능 향상

#### TINL 단독 시나리오 (Table 1)

| 모델 | 데이터셋 | W-F (기준) | W-F (ReNode) | 향상 |
|------|---------|-----------|-------------|------|
| GCN | CORA | 79.1±1.1 | 79.8±0.9** | +0.7 |
| GAT | CORA | 76.0±1.7 | 77.7±2.0** | +1.7 |
| PPNP | CORA | 80.5±1.6 | 81.9±0.6** | +1.4 |
| SGC | CORA | 74.9±2.1 | 77.0±1.1** | +2.1 |
| GCN | PubMed | 74.6±2.1 | 76.1±1.5** | +1.5 |

- 6개 GNN 모델, 5개 데이터셋 대부분에서 통계적으로 유의미한 성능 향상 (p < 0.05 또는 p < 0.01)

#### 위상 불균형 수준별 효과 (Table 2)

ReNode는 위상 불균형이 높을(High) 때 가장 큰 향상을 보임:

- CORA-High: 76.5 → 78.7** (+2.2)
- CORA-Low: 79.7 → 80.4** (+0.7)

#### TINL + QINL 복합 시나리오 (Table 3)

불균형 비율 $\rho = 10$에서도 효과적:
- CB + ReNode: CB 단독 대비 Macro-F1 전반적 향상
- RW + ReNode: 그래프 특화 방법(DR-GCN, RA-GCN, G-SMOTE)을 초과

#### 대규모 그래프 (Figure 4)

Reddit, MAG-Scholar 데이터셋에서 PPRGo 대비 일관된 성능 향상, 특히 레이블 크기가 클수록 효과 증가

---

### 2.5 한계점

1. **동질 연결 그래프(Homogeneous Graph) 전용**: 연결된 노드가 유사하다는 가정 기반 → 단백질 네트워크 등 이질 연결 그래프(Heterogeneous Graph)에는 적용 한계

2. **저연결성 그래프 한계**: CiteSeer와 같이 그래프 연결성이 낮은 경우, 충돌 탐지 방법이 노드 위상적 위치를 정확히 반영하지 못해 성능 향상 폭이 감소

3. **Cold Start 문제**: 레이블 비율이 극도로 낮을 때 레이블 노드 간 영향 충돌이 미미하여 위상적 위치 측정이 불충분

4. **계산 복잡도**: PPR 행렬 계산이 대규모 그래프에서 비용이 높으며, 근사 방법(PPRGo)에 의존

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 Model-Agnostic 특성에 의한 일반화

ReNode는 어떤 GNN 인코더 $\mathcal{F}$에도 독립적으로 적용 가능한 **플러그앤플레이(Plug-and-Play)** 방식입니다. 훈련 손실 함수의 가중치만 수정하므로:

$$\text{기존 손실} \rightarrow L_T = -\frac{1}{|\mathcal{L}|}\sum_{v \in \mathcal{L}} w_v \sum_{c=1}^{k} y_v^{*c} \log g_v^c$$

이 특성이 6가지 서로 다른 GNN 아키텍처에서 일관된 성능 향상을 가능하게 합니다.

### 3.2 QINL 방법과의 직교적 결합에 의한 일반화

TINL과 QINL은 독립적 문제이므로, 기존 QINL 방법(Re-weight, Focal Loss, Class Balanced Loss)과 **직교적으로 결합**될 수 있습니다:

$$L_Q = -\frac{1}{|\mathcal{L}|}\sum_{v \in \mathcal{L}} \underbrace{w_v}_{\text{TINL 항}} \cdot \underbrace{\frac{\bar{|\mathcal{C}|}}{|\mathcal{C}_j|}}_{\text{QINL 항}} \sum_{c=1}^{k} y_v^{*c} \log g_v^c$$

실험 결과, 일반적 QINL 방법 + ReNode의 조합이 그래프 특화 QINL 방법(DR-GCN, RA-GCN, G-SMOTE)보다 우수한 성능을 달성 → **복합 불균형 시나리오에서의 일반화** 입증

### 3.3 대규모 그래프로의 일반화

PPRGo와의 통합을 통해 수백만 노드 규모의 MAG-Scholar 데이터셋에서도 동작하며, 귀납적(inductive) 설정에서도 유효성을 입증합니다:

$$\mathbf{g}' = \text{softmax}(\hat{\mathbf{P}}\mathcal{F}'(\mathbf{X}, \theta'))$$

근사 PPR 행렬 $\hat{\mathbf{P}}$을 사용하므로 글로벌 그래프 위상 구조에 대한 의존성을 줄이고 효율적 처리가 가능합니다.

### 3.4 GNN 아키텍처 평가 관점으로의 일반화

위상 불균형 민감도(Topology-Imbalance Sensitivity)를 GNN 평가의 새로운 기준으로 제시:

| GNN | 위상 불균형 민감도 |
|-----|----------------|
| GCN | 높음 (민감) |
| PPNP | 중간 |
| GAT | 낮음 (강건) |

- GCN: 이웃 특성의 단순 평균 집계 → 노이즈 필터링 메커니즘 부재 → 높은 민감도
- GAT: 동적 어텐션 가중치 → 이웃 정보 필터링 → 낮은 민감도
- PPNP: 무한 컨볼루션 메커니즘으로 원거리 노드 정보 집계 → 중간 민감도

이 관점은 **훈련 셋 선택에 따른 GNN 성능 변동성**을 부분적으로 설명합니다.

### 3.5 위상 불균형이 높을수록 일반화 효과 증가

Table 2에서 확인되듯, ReNode는 데이터셋의 위상 불균형 수준이 높을수록 더 큰 성능 향상을 보입니다. 실제 그래프 데이터에서 위상 불균형은 **항상 존재(ubiquitous)** 하므로, 실제 응용 환경에서의 일반화 잠재력이 높습니다.

---

## 4. 앞으로의 연구에 미치는 영향과 고려할 점

### 4.1 연구에 미치는 영향

#### (1) 새로운 연구 패러다임 개척
위상 불균형을 독립적 연구 문제로 정의함으로써, 기존 QINL 중심의 그래프 불균형 학습 연구에 새로운 차원을 추가했습니다. 향후 그래프 학습의 불균형 문제를 논할 때 TINL을 필수적으로 고려해야 합니다.

#### (2) GNN 설계 원칙에 대한 시사점
위상 불균형 민감도 분석은 GNN 아키텍처를 설계할 때 **정보 집계 메커니즘의 노이즈 필터링 능력**이 중요함을 시사합니다. 특히 어텐션 기반 집계 또는 장거리 정보 전파 메커니즘이 위상 불균형에 더 강건할 수 있습니다.

#### (3) 반지도 학습에서 레이블 노드 선택 전략
Totoro 메트릭은 능동 학습(Active Learning) 맥락에서 **어떤 노드를 레이블링할지 선택하는 기준**으로 활용될 수 있습니다. 위상 불균형이 낮은 레이블 집합을 구성하면 모델 성능이 향상될 것으로 기대됩니다.

### 4.2 앞으로 연구 시 고려할 점

#### (1) 이질 연결 그래프(Heterogeneous Graph)로의 확장
현재 ReNode는 동질 연결 그래프(연결된 노드가 유사하다는 가정)에만 적용됩니다. 이질 그래프(지식 그래프, 바이오 네트워크 등)에서 위상 불균형을 측정하고 해결하는 방법 연구가 필요합니다.

#### (2) 동적 그래프(Dynamic Graph)에서의 위상 불균형
시간에 따라 그래프 구조가 변하는 동적 그래프에서는 위상적 위치 자체가 시간에 따라 변합니다. 동적 그래프에서 실시간으로 Totoro를 계산하고 가중치를 조정하는 방법 개발이 필요합니다.

#### (3) 그래프 링크 예측, 그래프 분류로의 확장
본 논문은 노드 분류에 집중하지만, 링크 예측이나 그래프 수준 분류에서도 위상 불균형이 성능에 영향을 미칠 수 있습니다. 이에 대한 이론적 분석과 실증적 검증이 필요합니다.

#### (4) 레이블 비율이 극히 낮은 환경(Few-shot)
Cold Start 문제를 해결하기 위해, 레이블 노드 간 충돌이 거의 없는 상황에서도 위상적 위치를 추정할 수 있는 방법(예: 구조적 특징 활용, 그래프 생성 모델 활용)이 연구되어야 합니다.

#### (5) 이론적 근거 강화
현재 ReNode의 가중치 스케줄(코사인 어닐링)은 경험적 근거에 기반합니다. 왜 이 스케줄이 최적인지에 대한 이론적 분석이 필요합니다. 특히 그래프 스펙트럼 이론 관점에서 위상 불균형과 모델 일반화 오류 사이의 이론적 관계 규명이 중요합니다.

#### (6) 자기지도 학습(Self-supervised Learning)과의 결합
최근 그래프 대조 학습(Graph Contrastive Learning)이 주목받고 있습니다. 레이블 없이도 위상적 위치 정보를 활용하는 자기지도 방식의 위상 불균형 해소 방법 연구가 유망합니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

아래는 본 논문이 직접 인용하거나 동일한 문제를 다루는 관련 연구들과의 비교입니다. 단, **2020년 이후 ReNode를 후속으로 인용한 논문들은 본 PDF에 포함되어 있지 않으므로**, 여기서는 논문 내 언급된 2020~2021년 관련 연구만 정확하게 비교 분석합니다.

### 5.1 양적 불균형 관련 방법과의 비교

| 방법 | 발표 | 접근법 | TINL 처리 | 한계 |
|------|------|--------|----------|------|
| **DR-GCN** [Shi et al., IJCAI 2020] | 2020 | 클래스 조건부 적대 훈련 + 미레이블 노드 분포 제약 | ❌ | 위상 불균형 미고려 |
| **RA-GCN** [Ghorbani et al., arXiv 2021] | 2021 | 적대적 훈련으로 클래스별 샘플 가중치 자동 학습 | ❌ | 위상 불균형 미고려 |
| **GraphSMOTE** [Zhao et al., WSDM 2021] | 2021 | 합성 노드 + 엣지 생성으로 소수 클래스 오버샘플링 | ❌ | 위상 불균형 미고려 |
| **AdaGCN** [Shi et al., arXiv 2021] | 2021 | 부스팅 알고리즘 기반 불균형 처리 | ❌ | 위상 불균형 미고려 |
| **ReNode (본 논문)** | NeurIPS 2021 | PPR 기반 위상 위치 측정 + 인스턴스별 재가중 | ✅ | 동질 그래프 한정, 저연결성 한계 |

### 5.2 노드 위치 인식 관련 방법과의 비교

| 방법 | 접근법 | 차이점 |
|------|--------|--------|
| **P-GNN** [You et al., ICML 2019] | 앵커 노드로부터의 절대적 위치 추정 | 클래스 정보 미구분, 앵커 노드 의존 |
| **GraphReach** [Nishad et al., IJCAI 2020] | 도달 가능성 기반 위치 인식 | 절대 위치, 클래스 경계 대비 상대 위치 미측정 |
| **Totoro (본 논문)** | PPR 기반 영향 충돌로 클래스 경계 대비 **상대적 위치** 측정 | 클래스 구분, 앵커 노드 불필요, 반지도 설정 최적화 |

### 5.3 LP와 GNN 통합 관점

**Wang & Leskovec (arXiv 2020)** "Unifying Graph Convolutional Neural Networks and Label Propagation"은 GNN과 LP의 통합 관점을 제공하며, 본 논문은 이 관점을 활용하여 LP의 노드 영향력 경계가 GNN 결정 경계를 효과적으로 반영함을 실증적으로 보입니다.

---

## 참고 자료

**주요 참고 문헌 (본 PDF에서 직접 인용된 문헌)**

1. **Chen, D., Lin, Y., Zhao, G., Ren, X., Li, P., Zhou, J., & Sun, X. (NeurIPS 2021)**. "Topology-Imbalance Learning for Semi-Supervised Node Classification." *35th Conference on Neural Information Processing Systems (NeurIPS 2021)*. — **본 논문**

2. **Shi, M., Tang, Y., Zhu, X., Wilson, D. A., & Liu, J. (IJCAI 2020)**. "Multi-Class Imbalanced Graph Convolutional Network Learning." *29th IJCAI*, 2879–2885.

3. **Zhao, T., Zhang, X., & Wang, S. (WSDM 2021)**. "GraphSMOTE: Imbalanced Node Classification on Graphs with Graph Neural Networks." *WSDM '21*, 833–841.

4. **Ghorbani, M., et al. (arXiv 2021)**. "RA-GCN: Graph Convolutional Network for Disease Prediction Problems with Imbalanced Data." *arXiv:2103.00221*.

5. **Bojchevski, A., et al. (KDD 2020)**. "Scaling Graph Neural Networks with Approximate PageRank." *KDD 2020*, 2464–2473.

6. **Wang, H., & Leskovec, J. (arXiv 2020)**. "Unifying Graph Convolutional Neural Networks and Label Propagation." *arXiv:2002.06755*.

7. **Kipf, T. N., & Welling, M. (ICLR 2017)**. "Semi-supervised Classification with Graph Convolutional Networks." *ICLR 2017*.

8. **Klicpera, J., Bojchevski, A., & Günnemann, S. (ICLR 2019)**. "Predict then Propagate: Graph Neural Networks meet Personalized PageRank." *ICLR 2019*.

9. **You, J., Ying, R., & Leskovec, J. (ICML 2019)**. "Position-aware Graph Neural Networks." *ICML 2019*, 7134–7143.

10. **Shi, S., et al. (arXiv 2021)**. "AdaGCN: Adaptive Boosting Algorithm for Graph Convolutional Networks on Imbalanced Node Classification." *arXiv:2105.11625*.

> **주의**: 2020년 이후 ReNode를 인용한 후속 연구(2022~2024)에 대한 비교 분석은 본 PDF에 포함된 정보의 범위를 초과하므로, 정확성을 위해 해당 논문들에 대한 추가 분석은 제공하지 않습니다. 관련 최신 연구는 Google Scholar에서 "Topology-Imbalance" 또는 "ReNode" 키워드로 검색하실 것을 권장합니다.
