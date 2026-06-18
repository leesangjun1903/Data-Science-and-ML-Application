# GraphSMOTE: Imbalanced Node Classification on Graphs with Graph Neural Networks

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

GraphSMOTE는 그래프 구조 데이터에서 발생하는 **불균형 노드 분류(Imbalanced Node Classification)** 문제를 해결하기 위해, 기존 SMOTE 기반 오버샘플링 알고리즘을 GNN에 맞게 확장한 새로운 프레임워크입니다.

기존 방법들의 두 가지 핵심 문제를 지적합니다:
1. **관계 정보 부재**: 생성된 합성 노드에 엣지(연결 관계) 정보가 없음
2. **고차원 원본 공간에서의 보간 문제**: 원본 입력 공간에서 직접 보간 시 도메인 외 샘플이 생성됨

### 주요 기여

| 기여 항목 | 설명 |
|-----------|------|
| 새로운 문제 정의 | 그래프에서의 노드 클래스 불균형 문제를 최초로 체계적으로 다룸 |
| GraphSMOTE 프레임워크 | 임베딩 공간에서 오버샘플링 + 엣지 생성기를 결합한 일반적 프레임워크 |
| 실험적 검증 | 3개 데이터셋에서 모든 기준선 대비 큰 폭으로 성능 향상 |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

**불균형 노드 분류 문제**는 그래프 내 노드들이 소속된 클래스의 크기가 심하게 불균형할 때 발생합니다.

$$\text{Imbalance Ratio} = \frac{\min_i(|\mathcal{C}_i|)}{\max_i(|\mathcal{C}_i|)}$$

예를 들어, 소셜 네트워크의 봇 탐지에서 봇 계정은 전체의 1% 미만이므로, 기존 GNN은 다수 클래스(정상 사용자) 위주로 학습되어 소수 클래스(봇) 탐지 성능이 크게 저하됩니다.

**공식 문제 정의**: 그래프 $\mathcal{G} = \{\mathcal{V}, \mathbf{A}, \mathbf{F}\}$가 주어질 때,

$$f(\mathcal{V}, \mathbf{A}, \mathbf{F}) \rightarrow \mathbf{Y} $$

다수 클래스와 소수 클래스 모두에서 잘 동작하는 노드 분류기 $f$를 학습하는 것이 목표입니다.

---

### 2.2 제안 방법 (수식 포함)

GraphSMOTE는 네 가지 핵심 컴포넌트로 구성됩니다.

#### (1) Feature Extractor (특징 추출기)

GraphSage를 백본으로 사용하며, 메시지 패싱 및 통합 과정은 다음과 같습니다:

$$\mathbf{h}^1_v = \sigma(\mathbf{W}^1 \cdot CONCAT(\mathbf{F}[v, :],\ \mathbf{F} \cdot \mathbf{A}[:, v])) $$

- $\mathbf{F}[v, :]$: 노드 $v$의 원본 속성 벡터
- $\mathbf{A}[:, v]$: 인접 행렬의 $v$번째 열 (이웃 정보)
- $\mathbf{W}^1$: 학습 가능한 가중치 행렬
- $\sigma$: ReLU 등의 활성화 함수

> 원본 고차원 입력 공간 대신 저차원의 표현 공간에서 보간을 수행함으로써 더 자연스러운 샘플을 생성합니다.

---

#### (2) Synthetic Node Generation (합성 노드 생성)

**Step 1 – 최근접 이웃 탐색**: 소수 클래스 내에서 동일 클래스의 최근접 이웃을 탐색합니다.

$$nn(v) = \arg\min_u \|\mathbf{h}^1_u - \mathbf{h}^1_v\|, \quad \text{s.t.}\ \mathbf{Y}_u = \mathbf{Y}_v $$

**Step 2 – 합성 노드 생성**: 보간(interpolation)으로 새로운 임베딩을 생성합니다.

$$\mathbf{h}^1_{v'} = (1 - \delta) \cdot \mathbf{h}^1_v + \delta \cdot \mathbf{h}^1_{nn(v)} $$

- $\delta \sim \text{Uniform}[0, 1]$: 무작위 보간 계수
- $\mathbf{h}^1_v$와 $\mathbf{h}^1_{nn(v)}$가 동일 클래스이므로 $\mathbf{h}^1_{v'}$도 해당 클래스에 속할 가능성이 높음

---

#### (3) Edge Generator (엣지 생성기)

합성 노드에 관계 정보를 부여하기 위해 **가중 내적(weighted inner product)** 기반 엣지 생성기를 도입합니다:

$$\mathbf{E}_{v,u} = softmax(\sigma(\mathbf{h}^1_v \cdot \mathbf{S} \cdot \mathbf{h}^1_u)) $$

- $\mathbf{S}$: 노드 간 상호작용을 캡처하는 파라미터 행렬

엣지 생성기의 손실 함수 (인접 행렬 재구성):

$$\mathcal{L}_{edge} = \|\mathbf{E} - \mathbf{A}\|^2_F $$

**두 가지 엣지 전략**:

- **이진 엣지 (GraphSMOTE $_T$ )**: 임계값 $\eta$를 사용하여 이진화

$$\tilde{\mathbf{A}}[v', u] = \begin{cases} 1, & \text{if } \mathbf{E}_{v',u} > \eta \\ 0, & \text{otherwise} \end{cases} $$

- **소프트 엣지 (GraphSMOTE $_O$ )**: 연속값 사용 (그래디언트 역전파 가능)

$$\tilde{\mathbf{A}}[v', u] = \mathbf{E}_{v',u} $$

---

#### (4) GNN Classifier (GNN 분류기)

확장된 그래프 $\tilde{\mathcal{G}} = \{\tilde{\mathbf{A}}, \tilde{\mathbf{H}}\}$ 위에서 또 다른 GraphSage 블록으로 분류를 수행합니다:

$$\mathbf{h}^2_v = \sigma(\mathbf{W}^2 \cdot CONCAT(\mathbf{h}^1_v,\ \tilde{\mathbf{H}}^1 \cdot \tilde{\mathbf{A}}[:, v])) $$

$$\mathbf{P}_v = softmax(\sigma(\mathbf{W}^c \cdot CONCAT(\mathbf{h}^2_v,\ \mathbf{H}^2 \cdot \tilde{\mathbf{A}}[:, v]))) $$

분류 손실 함수 (Cross-Entropy):

$$\mathcal{L}_{node} = \sum_{u \in \tilde{\mathcal{V}}_L} \sum_c \mathbf{1}(Y_u == c) \cdot \log(\mathbf{P}_v[c]) $$

추론 시 예측 클래스:

$$Y'_v = \arg\max_c \mathbf{P}_{v,c} $$

---

#### (5) 최종 최적화 목표

$$\min_{\theta, \phi, \varphi} \mathcal{L}_{node} + \lambda \cdot \mathcal{L}_{edge} $$

- $\theta$: Feature Extractor 파라미터
- $\phi$: Edge Generator 파라미터
- $\varphi$: Node Classifier 파라미터

---

### 2.3 모델 구조 요약

```
입력 그래프 G = {V, A, F}
        ↓
[Feature Extractor] (GraphSage Block 1)
→ 노드 임베딩 H¹ 생성
        ↓
[Synthetic Node Generator] (SMOTE in Embedding Space)
→ 소수 클래스 합성 노드 임베딩 생성
        ↓
[Edge Generator] (Weighted Inner Product)
→ 합성 노드의 엣지 예측 → 확장된 인접 행렬 Ã 생성
        ↓
[GNN Classifier] (GraphSage Block 2 + Linear Layer)
→ 균형 잡힌 그래프에서 노드 분류
```

---

### 2.4 성능 향상

**Table 1 요약** (3개 데이터셋):

| 방법 | Cora AUC-ROC | BlogCatalog AUC-ROC | Twitter AUC-ROC |
|------|-------------|---------------------|-----------------|
| Origin | 0.914 | 0.586 | 0.577 |
| SMOTE | 0.920 | 0.595 | 0.604 |
| Embed-SMOTE | 0.913 | 0.588 | 0.606 |
| **GraphSMOTE $_{preO}$ ** | **0.934** | **0.641** | **0.636** |

- Cora에서 Over-sampling 대비 AUC-ROC **+0.016** 향상
- BlogCatalog에서 F-Score **0.074 → 0.126** (약 70% 향상)
- 불균형 비율이 극단적일수록(ratio=0.1) Re-weight 대비 **+0.0326** AUC 향상

---

### 2.5 한계점

1. **Transductive 설정 한정**: 훈련과 테스트가 동일 그래프에서 수행되어야 하므로, **귀납적(Inductive) 설정**에서의 적용이 제한적입니다.
2. **엣지 생성기의 단순성**: 가중 내적 기반의 단순한 구조로 복잡한 엣지 분포를 완전히 모델링하기 어렵습니다.
3. **과도한 오버샘플링 문제**: 오버샘플링 스케일이 너무 크면 중복 정보로 인해 성능이 오히려 저하됩니다.
4. **하이퍼파라미터 민감도**: $\lambda$ 값이 너무 크면 성능이 급격히 하락하며, $1e{-6}$ ~ $4e{-6}$ 범위가 권장됩니다.
5. **이진 분류 위주 검증**: Twitter 데이터셋은 이진 분류이고, 다중 클래스(38개)인 BlogCatalog에서 F-Score가 여전히 매우 낮습니다(0.126).
6. **그래프 동질성(homophily) 가정**: 엣지 생성기가 동질적인 그래프 구조에 최적화되어 이질적(heterophilic) 그래프에서의 성능이 불확실합니다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 다양한 불균형 비율에 대한 일반화

Table 2에서 불균형 비율을 $\{0.1, 0.2, 0.4, 0.6\}$으로 변화시킨 실험 결과:

$$\text{Improvement} \propto \frac{1}{\text{Imbalance Ratio}}$$

불균형이 심할수록 GraphSMOTE의 성능 향상이 더 두드러집니다. 특히 ratio=0.1에서 GraphSMOTE $_{preO}$ 가 Re-weight 대비 **+0.0326** AUC를 기록합니다.

### 3.2 다양한 GNN 백본에 대한 일반화

GCN을 백본으로 교체했을 때도 GraphSMOTE의 모든 변형이 최고 성능을 달성합니다(Table 3).

- **GraphSage 기반**: 사전 학습(pre-training)이 성능 향상에 크게 기여
- **GCN 기반**: 사전 학습 없이도 준수한 성능 → GCN의 표현 능력이 상대적으로 단순해 사전 학습 효과가 덜 중요

이는 GraphSMOTE가 **특정 GNN 구조에 종속되지 않는 일반적 프레임워크**임을 보여줍니다.

### 3.3 임베딩 공간의 질과 일반화

사전 학습(Pre-training)의 효과:

$$\mathcal{L}_{pretrain} = \mathcal{L}_{edge}$$

엣지 재구성 손실로 Feature Extractor와 Edge Generator를 먼저 수렴시키면, 임베딩 공간의 의미적 밀도가 향상되어 SMOTE 보간 결과의 신뢰성이 높아집니다. 이는 GraphSMOTE $\_{preO}$ 와 GraphSMOTE $_{preT}$ 가 일관되게 높은 성능을 보이는 이유입니다.

### 3.4 일반화 성능 향상을 위한 추가 가능성

1. **더 강력한 엣지 생성기**: VAE 기반 생성 모델이나 GAN을 활용한 엣지 생성
2. **귀납적 설정으로 확장**: GraphSage의 귀납적 학습 특성을 활용하여 새로운 그래프에도 적용 가능한 방향
3. **다양한 오버샘플링 전략**: ADASYN, Borderline-SMOTE 등을 임베딩 공간에 적용

---

## 4. 향후 연구에 미치는 영향 및 고려 사항

### 4.1 향후 연구에 미치는 영향

#### (1) 그래프 불균형 학습의 기준선(Baseline) 확립
GraphSMOTE는 그래프 불균형 노드 분류 문제를 처음으로 체계적으로 다룬 연구로서, 이후 연구들의 표준 비교 대상이 되었습니다.

#### (2) 임베딩 공간에서의 데이터 증강 패러다임 정착
원본 입력 공간이 아닌 **중간 임베딩 공간에서 데이터 증강**을 수행하는 패러다임을 그래프 도메인에 적용하여, 이후 연구들이 더 정교한 생성 모델(VAE, GAN 등)을 활용하는 방향으로 발전했습니다.

#### (3) 엣지 생성 문제의 독립적 연구 촉진
합성 노드의 관계 정보 생성을 별도의 중요 문제로 정의함으로써, 링크 예측(Link Prediction)과 불균형 학습의 융합 연구를 자극했습니다.

### 4.2 2020년 이후 관련 최신 연구 비교 분석

#### ImGAGN (2021)
- **논문**: "ImGAGN: Imbalanced Network Embedding via Generative Adversarial Graph Networks" (Qu et al., 2021)
- **차이점**: GAN을 활용하여 더 사실적인 소수 클래스 노드를 생성. GraphSMOTE의 선형 보간 한계를 비선형 생성으로 극복
- **비교**: GraphSMOTE보다 생성 품질이 높지만 학습 불안정성이 단점

#### PC-GNN (2021)
- **논문**: "Pick and Choose: A GNN-based Imbalanced Learning Approach for Fraud Detection" (Liu et al., WWW 2021)
- **차이점**: 부정 탐지(Fraud Detection)에 특화하여 엣지 샘플링과 노드 분류를 결합. 소수 클래스 주변 이웃 선택 전략을 도입
- **비교**: 특정 도메인에 최적화되어 있어 일반성은 GraphSMOTE보다 낮음

#### TAM (2023)
- **논문**: "TAM: Topology-Aware Margin Loss for Class-Imbalanced Node Classification" (Song et al., ICML 2023)
- **차이점**: 데이터 증강 대신 **토폴로지를 고려한 마진 손실 함수**를 설계하여 그래프 구조 정보를 손실 함수에 직접 통합
- **비교**: GraphSMOTE의 데이터 증강 방식 대신 알고리즘 레벨 접근으로, 합성 노드 생성의 부작용 없이 소수 클래스를 강조

#### GraphENS (2021)
- **논문**: "GraphENS: Neighbor-Aware Ego Network Synthesis for Class-Imbalanced Node Classification" (Park et al., ICLR 2022)
- **차이점**: Ego Network(자아 네트워크) 합성을 통해 노드와 그 주변 이웃을 함께 생성. GraphSMOTE가 단순 엣지 예측에 의존하는 반면, 실제 이웃 구조를 함께 생성
- **비교**: 위상 구조 보존 측면에서 GraphSMOTE보다 우수

#### 비교 요약표

| 논문 | 방법론 | 그래프 구조 보존 | 귀납적 설정 | 핵심 한계 |
|------|--------|----------------|------------|-----------|
| **GraphSMOTE** | 임베딩 SMOTE + 엣지 예측 | 보통 | 제한적 | 단순 엣지 생성기 |
| ImGAGN | GAN 기반 생성 | 보통 | 제한적 | 학습 불안정성 |
| PC-GNN | 이웃 선택 + 오버샘플링 | 높음 | 가능 | 도메인 특화 |
| TAM | 토폴로지 인식 마진 손실 | 높음 | 가능 | 생성 없음 |
| GraphENS | Ego 네트워크 합성 | 높음 | 제한적 | 복잡한 구조 |

---

### 4.3 앞으로 연구 시 고려할 점

#### (1) 이질적 그래프(Heterophilic Graph)에서의 적용
GraphSMOTE는 동질적(homophilic) 그래프를 가정하지만, 실제 많은 그래프(예: 사기 탐지 네트워크)는 이질적 구조를 가집니다. 이질적 그래프에서의 소수 클래스 임베딩 품질 저하 문제를 해결해야 합니다.

#### (2) 동적 그래프(Dynamic Graph)로의 확장
정적 그래프에 한정된 현재 방법을 시간에 따라 변화하는 동적 그래프에 적용하면, 시간적 클래스 불균형 문제를 다룰 수 있습니다.

#### (3) 더 정교한 생성 모델 도입
단순 선형 보간(SMOTE) 대신 **조건부 VAE(CVAE)** 또는 **Score-based Diffusion Model**을 활용하여 더 다양하고 현실적인 소수 클래스 노드를 생성하는 방향이 유망합니다.

$$\mathbf{h}^1_{v'} \sim p_\theta(\mathbf{h} | y_{minority}, \mathcal{G})$$

#### (4) 노드 분류 외 태스크로 확장
논문 자체에서 언급한 바와 같이, **엣지 분류(Edge Type Prediction)**, **그래프 분류(Graph Classification)** 등 다른 그래프 태스크에서의 불균형 문제로 확장이 필요합니다.

#### (5) 프라이버시 보존 학습과의 결합
합성 데이터 생성 과정에서 **차분 프라이버시(Differential Privacy)**를 적용하면, 민감한 소셜 네트워크 데이터에서의 활용성이 높아집니다.

#### (6) 클래스 불균형의 동적 탐지
오버샘플링 스케일을 고정 하이퍼파라미터로 설정하지 않고, 학습 과정에서 **자동으로 적응적으로 결정**하는 메커니즘이 필요합니다.

---

## 참고 자료

1. **원본 논문**: Tianxiang Zhao, Xiang Zhang, Suhang Wang. "GraphSMOTE: Imbalanced Node Classification on Graphs with Graph Neural Networks." *WSDM 2021*. https://doi.org/10.1145/3437963.3441720

2. **관련 논문**:
   - Hamilton et al. "Inductive Representation Learning on Large Graphs." *NIPS 2017* (GraphSage)
   - Kipf & Welling. "Semi-Supervised Classification with Graph Convolutional Networks." *ICLR 2017* (GCN)
   - Chawla et al. "SMOTE: Synthetic Minority Over-sampling Technique." *JAIR 2002*
   - Ando & Huang. "Deep Over-Sampling Framework for Classifying Imbalanced Data." *ECML-PKDD 2017*
   - Liu et al. "Pick and Choose: A GNN-based Imbalanced Learning Approach for Fraud Detection." *WWW 2021*
   - Park et al. "GraphENS: Neighbor-Aware Ego Network Synthesis for Class-Imbalanced Node Classification." *ICLR 2022*
   - Song et al. "TAM: Topology-Aware Margin Loss for Class-Imbalanced Node Classification." *ICML 2023*

3. **GitHub 코드**: https://github.com/TianxiangZhao/GraphSmote
