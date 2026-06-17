# Towards Robust Graph Neural Networks for Noisy Graphs with Sparse Labels (RS-GNN)

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

본 논문은 **노이즈 엣지(noisy edges)**와 **희소 레이블(sparse labels)**이 동시에 존재하는 실제 그래프 환경에서 GNN의 성능이 크게 저하된다는 문제를 지적하고, 이 두 가지를 **동시에** 해결하는 프레임워크 **RS-GNN(Robust Structural GNN)**을 제안한다.

### 주요 기여 (3가지)

| 기여 | 내용 |
|------|------|
| **새로운 문제 정의** | 노이즈 그래프 + 희소 레이블의 복합 문제를 최초로 정식 연구 |
| **RS-GNN 프레임워크** | 링크 예측기로 그래프 노이즈 제거 및 밀도화(densification)를 동시 수행 |
| **광범위한 실험** | 4개 벤치마크, 4종 노이즈 유형에서 SOTA 대비 우월한 성능 입증 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**두 가지 핵심 문제가 GNN의 메시지 패싱(message passing) 메커니즘을 손상시킨다:**

#### 문제 1: 노이즈 엣지
- 적대적 공격(Metattack, Nettack) 또는 고유 노이즈로 인해 서로 다른 레이블/특징을 가진 노드들이 연결됨
- 메시지 패싱 과정에서 노이즈/오류가 전파되어 노드 표현을 오염시킴

#### 문제 2: 희소 레이블
$K$-layer GNN에서, 레이블 노드는 자신의 $K$-hop 이웃까지만 학습에 참여시킨다. 레이블 수가 적을수록 학습에 참여하는 미레이블 노드 비율이 급감한다.

**예비 분석 결과 (Fig. 2):**
- 레이블 비율이 0.1 → 0.01로 감소 시, 미참여 노드 비율이 Citeseer에서 약 0.2 → 0.8로 급증
- 엣지 수를 원본의 3배로 늘리면 미참여 노드 비율이 0.8 → 0.3으로 대폭 감소

---

### 2.2 제안 방법 및 수식

#### 2.2.1 문제 형식 정의

그래프 $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathbf{X})$, 레이블 노드 $\mathcal{V}_L$, 레이블 $\mathcal{Y}$가 주어질 때:

$$\text{동시에 학습: } f_E: (v_i, v_j) \rightarrow \mathbf{S}_{ij}, \quad f_\mathcal{G}: (\mathbf{S}, \mathbf{X}) \rightarrow \hat{\mathcal{Y}}$$

여기서 $\mathbf{S} \in [0,1]^{N \times N}$는 노이즈가 제거되고 밀도화된 인접 행렬.

---

#### 2.2.2 링크 예측기 (MLP 기반)

GNN 대신 **MLP**를 링크 예측기로 사용 (노이즈 그래프에서 GNN의 메시지 패싱이 노이즈를 증폭시키는 것을 방지):

$$\mathbf{z}_i = MLP(\mathbf{x}_i)$$

$$w(i, j) = f(\mathbf{z}_i^T \mathbf{z}_j) \quad (f: \text{ReLU})$$

---

#### 2.2.3 특징 유사도 가중 엣지 재구성 손실 (핵심 수식)

노이즈 엣지의 영향을 줄이기 위해 **특징 유사도**로 샘플을 재가중:

$$\mathcal{L}_E = \sum_{v_i \in \mathcal{V}} \sum_{v_j \in \mathcal{N}(v_i)} \left[ \exp\!\left(-\frac{\|\mathbf{x}_i - \mathbf{x}_j\|^2}{\sigma^2}\right)(w(i,j)-1)^2 + \sum_{n=1}^{Q} \mathbb{E}_{v_n \sim P_n(v_i)} \exp\!\left(\frac{\|\mathbf{x}_i - \mathbf{x}_n\|^2}{\sigma^2}\right)(w(i,n)-0)^2 \right]$$

**직관적 설명:**
- **양성 샘플** ( $v_j \in \mathcal{N}(v_i)$ ): 특징이 유사할수록($\exp$ 큰) $w(i,j) \to 1$로 강하게 학습 → 정상 엣지 보존
- 특징이 다르면($\exp$ 작음) 손실 영향이 작아 → 노이즈 엣지 자동 하향
- **음성 샘플** ($v_n$): 특징이 다를수록 $w(i,n) \to 0$으로 강하게 학습 → 잘못된 엣지 억제

---

#### 2.2.4 그래프 디노이징 및 밀도화

$$\mathbf{S}_{ij} = \begin{cases} w(i,j) & \text{if } w(i,j) > T_l \text{ and } v_j \in \mathcal{N}(v_i) \cup \mathcal{S}(v_i) \\ 0 & \text{else} \end{cases}$$

- $\mathcal{N}(v_i)$: 노이즈 그래프의 기존 이웃
- $\mathcal{S}(v_i)$: 코사인 유사도 기준 상위 $K$개 후보 노드 집합 (계산 효율화)
- $T_l$: 엣지 포함/제외 임계값

---

#### 2.2.5 GNN 분류기 손실

$$\mathcal{L}_{GNN} = \sum_{v_i \in \mathcal{V}_L} l(\hat{\mathbf{y}}_i, \mathbf{y}_i)$$

($l$: cross-entropy)

---

#### 2.2.6 미레이블 노드 레이블 평활화 정규화

$$\mathcal{L}_u = \sum_{v_i \in \mathcal{V}_u} \sum_{v_j \in \mathcal{V}} \mathbf{T}_{ij} \|\hat{\mathbf{y}}_i - \hat{\mathbf{y}}_j\|^2$$

$$\mathbf{T}_{ij} = \begin{cases} \mathbf{S}_{ij} & \text{if } \mathbf{S}_{ij} > T_h \\ 0 & \text{otherwise} \end{cases}$$

엣지 가중치가 높을수록 두 노드의 예측 레이블이 유사해야 함을 강제 → 미레이블 노드를 **직접적으로** 학습에 참여시킴.

---

#### 2.2.7 최종 목적 함수

$$\arg\min_{\theta_E, \theta_\mathcal{G}} \mathcal{L}_{GNN} + \alpha \mathcal{L}_E + \beta \mathcal{L}_u$$

| 하이퍼파라미터 | 역할 |
|---|---|
| $\alpha$ | 링크 예측기의 그래프 재구성 기여도 조절 |
| $\beta$ | 레이블 평활화 정규화 강도 조절 |

---

### 2.3 모델 구조

```
입력: 노이즈 그래프 G = (V, E, X)
         ↓
[MLP 링크 예측기 f_E]
  - 노드 특징 X → 노드 임베딩 Z
  - 특징 유사도 가중 손실 L_E로 학습
  - 출력: 디노이즈 + 밀도화된 인접 행렬 S
         ↓
[GCN 분류기 f_G]
  - 입력: (S, X)
  - 출력: 노드 레이블 예측 Ŷ
         ↓
[레이블 평활화 L_u]
  - 고가중치 예측 엣지로 미레이블 노드 예측 정규화
         ↓
최종 손실: L_GNN + α·L_E + β·L_u (End-to-End 학습)
```

---

### 2.4 성능 향상

**Table 1 주요 결과 (레이블 비율 1%, Cora 기준):**

| 방법 | Raw Graph | Random Noise | Non-Targeted Attack | Targeted Attack |
|------|-----------|--------------|---------------------|-----------------|
| GCN | 65.5±0.5 | 59.2±0.7 | 26.8±2.5 | 45.3±1.2 |
| Pro-GNN | 65.9±1.3 | 56.1±3.0 | 41.7±5.7 | 49.7±0.9 |
| **RS-GNN** | **75.3±0.6** | **71.8±1.5** | **70.8±0.7** | **67.8±1.2** |

**핵심 성능 포인트:**
- Non-targeted attack(Metattack)에서 GCN 대비 **+44%p** (Cora-ML: 13.2 → 73.2%)
- 그래프 밀도화 후 참여 미레이블 노드 수 급증 (Cora: 212 → 1,383개)

---

### 2.5 한계점

논문에서 명시적으로 언급하거나 분석을 통해 도출되는 한계:

1. **속성 노이즈 미처리**: 구조적 노이즈만 다루고, 노드 속성 자체의 노이즈(예: 소셜 네트워크의 허위 프로필)는 미고려
2. **레이블 노이즈 미고려**: 레이블 자체에 노이즈가 있을 경우 처리 불가
3. **동질성(homophily) 가정 의존**: 유사한 특징을 가진 노드가 같은 레이블을 가진다는 가정에 기반 → 이질성(heterophily) 그래프에서 성능 저하 우려
4. **하이퍼파라미터 민감도**: $\alpha, \beta, T_l, T_h, K, \sigma$ 등 다수의 하이퍼파라미터 튜닝 필요
5. **확장성(Scalability)**: 후보 집합 $\mathcal{S}(v_i)$ 전략으로 완화했으나, 초대형 그래프에서의 효율성은 추가 검증 필요
6. **GCN 백본 고정**: 실험에서 GCN만 백본으로 사용, 다른 GNN 백본(GAT, GIN 등)과의 결합 효과는 미래 연구로 남김

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화를 가능하게 하는 핵심 메커니즘

#### (1) 귀납적 그래프 학습(Inductive Graph Learning)
MLP 기반 링크 예측기는 **노드 속성만으로** 엣지 가중치를 예측하므로, 학습 시 보지 못한 노드(새 노드)에도 적용 가능한 귀납적(inductive) 특성을 잠재적으로 가짐.

#### (2) 레이블 평활화를 통한 정규화 효과
$\mathcal{L}_u$는 미레이블 노드의 예측 일관성을 강제함으로써 **과적합 방지 및 일반화** 역할:

$$\mathcal{L}_u = \sum_{v_i \in \mathcal{V}_u} \sum_{v_j \in \mathcal{V}} \mathbf{T}_{ij} \|\hat{\mathbf{y}}_i - \hat{\mathbf{y}}_j\|^2$$

이는 그래프 기반 Laplacian 정규화와 유사한 효과로, 표현의 부드러운 변화(smooth variation)를 유도.

#### (3) 다양한 노이즈 유형에 대한 강건성
- Random noise, Metattack(비표적), Nettack(표적) 모두에서 일관된 성능 유지
- **다양한 수준의 perturbation rate(0~25%)에서도 안정적** (Fig. 4)

#### (4) 그래프 희소도 독립성
Table 2 결과:

| Edge Rate | GCN | RS-GNN | 개선폭 |
|-----------|-----|--------|--------|
| 20% | 54.5 | 63.7 | **+9.2%** |
| 100% | 64.8 | 71.2 | +6.4% |

그래프가 희소할수록 RS-GNN의 상대적 개선이 더 큼 → **다양한 밀도 수준에서 안정적 일반화**

#### (5) 레이블 비율 독립적 효과
Fig. 6 분석: Metattack 공격 그래프에서 레이블 비율이 높아져도 RS-GNN이 계속 우월 → 노이즈 제거 자체가 독립적 기여

### 3.2 일반화의 잠재적 약점

- **동질성 그래프 편향**: 링크 예측이 "유사한 특징 = 연결 가능"에 기반하므로, 이질성 그래프(heterophily graph)에서는 일반화 실패 가능
- **분포 외 노이즈(OOD noise)**: 학습 시 보지 못한 새로운 유형의 공격에 대한 일반화는 미검증

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

### 4.1 비교 프레임워크

```
해결 목표 기준 분류
├── 노이즈 방어
│   ├── Pro-GNN (Jin et al., KDD 2020)
│   ├── GNNGuard (Zhang & Zitnik, NeurIPS 2020)
│   └── RS-GNN (Dai et al., WSDM 2022) ← 본 논문
├── 희소 레이블
│   ├── Self-Training (Li et al., AAAI 2018)
│   ├── SuperGAT (Kim & Oh, ICLR 2021)
│   └── NRGNN (Dai et al., KDD 2021)
└── 그래프 구조 학습
    └── Pro-GNN, RS-GNN
```

### 4.2 세부 비교 분석

| 방법 | 노이즈 방어 | 희소 레이블 | 그래프 재구성 | 계산 효율 | 주요 메커니즘 |
|------|------------|------------|--------------|-----------|-------------|
| **Pro-GNN** (KDD 2020) | ✅ | ❌ | ✅ (직접 학습) | ❌ (고비용) | Low-rank + sparsity 제약 |
| **GNNGuard** (NeurIPS 2020) | ✅ | ❌ | ❌ | ✅ | 이웃 중요도 가중치 |
| **NRGNN** (KDD 2021) | △ | ✅ | △ | ✅ | 레이블 노이즈 + 희소 레이블 |
| **SuperGAT** (ICLR 2021) | △ | ✅ | ❌ | ✅ | 자기지도 엣지 예측 |
| **RS-GNN** (WSDM 2022) | ✅ | ✅ | ✅ (링크 예측) | ✅ | 특징 유사도 가중 재구성 |

### 4.3 RS-GNN의 차별점

**vs Pro-GNN:**
- Pro-GNN: 그래프를 직접 학습($O(N^2)$ 파라미터), 저랭크 제약으로 계산 비용 큼
- RS-GNN: MLP 링크 예측기로 간접 학습, 후보 집합 $\mathcal{S}(v_i)$로 $O(NK)$ 수준으로 계산 절약

**vs GNNGuard:**

GNNGuard는 다음과 같이 이웃 가중치를 조정:
$$h_i^{(l+1)} = \sigma\left(\sum_{j \in \mathcal{N}(i)} \hat{a}_{ij} \cdot W^{(l)} h_j^{(l)}\right)$$
단, 희소 레이블 문제는 미해결. RS-GNN은 그래프 밀도화를 통해 희소 레이블도 동시 해결.

**vs NRGNN (Dai et al., KDD 2021):**
NRGNN은 **레이블 노이즈 + 희소 레이블**에 집중, 구조적 노이즈 방어 능력은 RS-GNN보다 약함.

---

## 5. 향후 연구에 미치는 영향 및 고려할 점

### 5.1 미치는 영향

#### (1) 복합 문제 해결 패러다임 제시
기존 연구들이 **노이즈 방어** 또는 **희소 레이블** 중 하나만 다루던 것을, RS-GNN은 **동시 해결 프레임워크**로 통합 → 이후 연구들이 복합 조건을 고려하는 흐름을 촉진

#### (2) 그래프 구조 학습의 실용적 접근
직접적 그래프 학습(Direct graph learning, Pro-GNN)의 고비용 문제를 **링크 예측기 기반 간접 학습**으로 해결 → 확장 가능한 그래프 구조 학습 방향 제시

#### (3) 레이블 평활화의 재조명
미레이블 노드에 대한 레이블 평활화를 **예측된 엣지 기반으로 적용**하는 아이디어는 반지도 학습(semi-supervised learning) 연구에 새로운 방향 제공

#### (4) 실용적 배경 확장
사기 탐지(fraud detection), 봇 탐지(bot detection) 등 **실제 희소 레이블 + 노이즈 환경** 응용에 직접적 영향

---

### 5.2 향후 연구 시 고려할 점

#### (1) 노드 속성 노이즈 처리
논문 자체가 한계로 인정한 부분: 구조적 노이즈만 다루므로, **속성 노이즈(attribute noise)와 구조 노이즈의 통합 처리** 연구 필요

$$\text{확장 문제: } \mathcal{G} = (\mathcal{V}, \mathcal{E}^{noisy}, \mathbf{X}^{noisy}) + \mathcal{V}_L^{sparse, noisy}$$

#### (2) 이질성 그래프(Heterophily Graph) 적용
동질성 가정을 완화하여, 서로 다른 레이블의 노드들이 연결되는 이질성 그래프(예: 웹 그래프, 일부 지식 그래프)에서의 적용 가능성 연구 필요. 최근 H2GCN, FAGCN 등 이질성 GNN과의 결합 고려 가능.

#### (3) 동적 그래프(Dynamic Graph) 확장
시간에 따라 노이즈와 레이블이 변화하는 동적 그래프 환경에서의 강건성 연구 필요.

#### (4) 그래프 변환기(Graph Transformer)와의 결합
최근 GraphGPS, Graphormer 등 어텐션 기반 그래프 변환기가 발전함에 따라, RS-GNN의 링크 예측 및 밀도화 전략을 **그래프 변환기 백본과 결합**하는 방향 모색.

#### (5) 대규모 그래프 확장성
현재 실험은 최대 19,717개 노드(Pubmed)에 그침. **수백만 노드 규모** 그래프에서의 적용을 위한 샘플링 전략 및 분산 학습 방법 필요.

#### (6) 설명 가능성(Explainability)
어떤 엣지가 노이즈로 제거되고 어떤 엣지가 추가되었는지에 대한 **해석 가능한 그래프 변환** 메커니즘 개발 필요.

#### (7) 레이블 노이즈 통합
레이블 자체가 불완전하거나 오류가 있는 경우를 동시 처리하는 프레임워크 개발 (논문에서 미래 연구로 명시).

#### (8) 더 강력한 적응형 공격 대응
현재 Metattack, Nettack 수준의 공격만 평가. **Adaptive Attack**(방어 방법을 알고 있는 공격자)에 대한 강건성 평가 필요.

---

## 참고자료

**주요 출처:**
- **Enyan Dai, Wei Jin, Hui Liu, Suhang Wang. "Towards Robust Graph Neural Networks for Noisy Graphs with Sparse Labels." WSDM 2022. arXiv:2201.00232v1.**
- Wei Jin, Yao Ma, et al. "Graph structure learning for robust graph neural networks." KDD 2020.
- Xiang Zhang and Marinka Zitnik. "GNNGuard: Defending graph neural networks against adversarial attacks." NeurIPS 2020. arXiv:2006.08149.
- Enyan Dai, Charu Aggarwal, and Suhang Wang. "NRGNN: Learning a Label Noise Resistant Graph Neural Network on Sparsely and Noisily Labeled Graphs." KDD 2021.
- Dongkwan Kim and Alice Oh. "How to find your friendly neighborhood: Graph attention design with self-supervision." ICLR 2021.
- Thomas N Kipf and Max Welling. "Semi-supervised classification with graph convolutional networks." arXiv:1609.02907, 2016.
- Daniel Zügner and Stephan Günnemann. "Adversarial attacks on graph neural networks via meta learning." arXiv:1902.08412, 2019.
- Negin Entezari et al. "All You Need Is Low (Rank) Defending Against Adversarial Attacks on Graphs." WSDM 2020.

> **주의:** 2020년 이후 최신 연구(GNNGuard, NRGNN 등)와의 비교 분석 중 일부는 해당 논문의 내용과 각 논문의 공개 arXiv 자료를 기반으로 작성하였으며, RS-GNN이 직접 실험 비교하지 않은 방법(GNNGuard 등)에 대한 성능 수치 직접 비교는 제시하지 않았습니다.
