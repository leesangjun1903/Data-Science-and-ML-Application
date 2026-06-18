# GraphENS: Neighbor-Aware Ego Network Synthesis for Class-Imbalanced Node Classification

---

## 📌 참고 자료

- **주 논문**: Park, J., Song, J., & Yang, E. (2022). *GraphENS: Neighbor-Aware Ego Network Synthesis for Class-Imbalanced Node Classification*. **ICLR 2022**.
- Zhao et al. (2021). *GraphSMOTE: Imbalanced Node Classification on Graphs with Graph Neural Networks*. WSDM 2021.
- Qu et al. (2021). *ImGAGN: Imbalanced Network Embedding via Generative Adversarial Graph Networks*. KDD 2021.
- Shi et al. (2020). *Multi-class Imbalanced Graph Convolutional Network Learning*. IJCAI 2020.
- Chawla et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique*. JAIR.
- Kang et al. (2020). *Decoupling Representation and Classifier for Long-Tailed Recognition*. ICLR 2020.
- Verma et al. (2019). *Manifold Mixup: Better Representations by Interpolating Hidden States*. ICML 2019.
- Kipf & Welling (2017). *Semi-supervised Classification with Graph Convolutional Networks*. ICLR 2017.
- Hamilton et al. (2017). *Inductive Representation Learning on Large Graphs*. NeurIPS 2017.
- Velickovic et al. (2018). *Graph Attention Networks*. ICLR 2018.

> ⚠️ **정확도 관련 고지**: 본 답변은 제공된 PDF 원문에 직접 근거하며, 논문에 명시되지 않은 내용은 추론임을 별도 표기합니다. 2020년 이후 최신 연구 비교는 논문 내 인용 범위와 공개된 정보 내에서 기술합니다.

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

GraphENS의 핵심 주장은 다음 가설에서 출발합니다:

> **"클래스 불균형 노드 분류에서 GNN의 주요 문제는 소수 클래스 노드 자체에 대한 과적합(node memorization)보다, 메시지 패싱을 통해 접하는 소수 클래스의 이웃 집합(neighbor set)에 대한 과적합(neighbor memorization)이 더 심각하다."**

### 3대 핵심 기여

| 기여 | 내용 |
|------|------|
| **문제 정의** | GNN에서의 "neighbor memorization" 문제를 최초로 실험적으로 규명 |
| **방법론** | ego network 전체(중심 노드 + 1-hop 이웃)를 합성하는 GraphENS 제안 |
| **성능** | 다수 벤치마크 데이터셋에서 기존 방법 대비 일관된 성능 향상 달성 |

---

## 2. 상세 분석

### 2-1. 해결하고자 하는 문제

#### (1) 클래스 불균형 문제의 그래프 도메인 특수성

일반적인 클래스 불균형 처리 방법(re-weighting, oversampling 등)은 이미지 분류 등에서 개발되었으며, **그래프 데이터의 메시지 패싱(message passing) 특성**을 고려하지 않습니다.

그래프에서 노드 $v$의 표현은 다음과 같이 이웃 정보를 집계하여 형성됩니다:

$$x_v^{(l+1)} = h_l\left(x_v^{(l)},\ \phi_l\left(\{m_l(x_v^{(l)}, x_u^{(l)}, e_{v,u}) \mid u \in \mathcal{N}(v)\}\right)\right)$$

예를 들어 GCN의 경우:

$$x_v^{(l+1)} = \Theta \sum_{u \in \mathcal{N}(v) \cup \{v\}} \frac{e_{v,u}}{\sqrt{\hat{d}_u \hat{d}_v}} x_u^{(l)}, \quad \hat{d}_v = 1 + \sum_{u \in \mathcal{N}(v)} e_{v,u}$$

이 구조 때문에, 소수 클래스 노드 $v_{minor}$를 학습할 때 모델은 $v_{minor}$ 자체뿐 아니라 **$\mathcal{N}(v_{minor})$ (이웃 집합)에도 과적합**될 수 있습니다.

#### (2) Neighbor Memorization 문제의 실험적 규명

논문은 두 가지 "교체 실험(replacing experiment)"으로 이를 검증합니다:

- **Node-replacing experiment**: 학습된 이웃을 고정하고 노드 피처만 미학습 노드로 교체 → 정확도 소폭 하락
- **Neighbor-replacing experiment**: 학습된 노드 피처를 고정하고 이웃만 미학습 이웃으로 교체 → 정확도 **급격히 하락**

실험 결과(Figure 1(c), (d))는 기존 방법들(Re-weighting, Oversampling)이 neighbor-replacing 시 성능 저하가 node-replacing보다 훨씬 크다는 것을 보여주며, **neighbor memorization이 더 심각한 문제임을 입증**합니다.

---

### 2-2. 제안 방법 (수식 포함)

GraphENS는 두 가지 핵심 컴포넌트로 구성됩니다.

#### **컴포넌트 1: Neighbor Sampling (이웃 샘플링)**

**목표**: 소수 클래스 노드 $v_{minor}$와 임의로 선택된 타겟 노드 $v_{target}$의 ego network를 결합하여, 다양한 이웃 환경을 경험하게 함으로써 neighbor memorization을 완화합니다.

**Step 1: 인접 노드 분포 정의**

실제 노드 $v$에 대한 인접 노드 분포 $p(u|v)$:

$$p(u|v) = \frac{1}{|\mathcal{N}(v)|} \quad \text{if } u \in V,\ \{u,v\} \in E, \quad \text{else } p(u|v) = 0$$

**Step 2: Ego Network 유사도 계산 (KL-Divergence)**

노드 $v$의 aggregated logit:

$$\hat{o}_v = \frac{1}{|\mathcal{N}(v)|+1} \sum_{u \in (\mathcal{N}(v) \cup v)} o_u$$

두 ego network 간 KL-Divergence:

$$\phi = \text{KL}\left(\sigma(\hat{o}_{minor}) \| \sigma(\hat{o}_{target})\right)$$

여기서 $\sigma$는 softmax 함수. 정규화된 거리:

$$\hat{\phi} = \frac{1}{1 + e^{-\phi}}$$

**Step 3: 합성 노드 $v_{mixed}$의 인접 노드 분포**

$$\boxed{p(u|v_{mixed}) = \hat{\phi}\, p(u|v_{minor}) + (1 - \hat{\phi})\, p(u|v_{target})}$$

- $v_{target}$이 $v_{minor}$와 유사할수록 → $\hat{\phi}$ 감소 → $v_{target}$의 이웃 비중 증가
- $v_{minor}$의 혼합 비율은 항상 최소 0.5 보장 (소수 클래스 의미 보존)

이 분포에서 이웃을 비복원 추출하며, 이웃 수는 원본 그래프의 degree 분포에서 샘플링합니다.

#### **컴포넌트 2: Saliency-based Node Mixing (현저성 기반 노드 혼합)**

**목표**: 타겟 노드의 클래스-특화(class-specific) 특성은 제거하고, 클래스-공통(class-generic) 특성만을 활용하여 새로운 소수 클래스 노드를 생성합니다.

**Feature Saliency 정의**:

분류 손실 $\mathcal{L}(G, \mathbf{y})$에 대한 입력 특성의 그래디언트 행렬:

$$\frac{\partial \mathcal{L}(G, \mathbf{y})}{\partial X} \in \mathbb{R}^{|V| \times d}$$

노드 $v$의 $i$번째 피처에 대한 saliency 값:

$$s_{(v,i)} = \left| \left[ \frac{\partial \mathcal{L}(G, \mathbf{y})}{\partial X} \right]_{(v,i)} \right|$$

**Node Mixup 수식**:

마스킹 비율 $K = k\hat{\phi}$ ($k$: 하이퍼파라미터, $\hat{\phi}$: 정규화된 거리)

바이너리 마스크 $M_K \in \mathbb{R}^d$: $v_{target}$의 $K\%$ 피처를 0으로 마스킹 (saliency 기반 multinomial 분포에서 샘플링)

$\lambda \sim \text{Beta}(\alpha, \alpha)$로 혼합 비율 샘플링:

$$\boxed{v_{mixed} = (1 - \Lambda_K) \odot v_{minor} + \Lambda_K \odot v_{target}, \quad \text{where } \Lambda_K = \lambda \cdot M_K}$$

- $v_{minor}$와 $v_{target}$이 멀수록($\hat{\phi}$ 증가) → $K$ 증가 → 더 많은 클래스-특화 피처 마스킹

---

### 2-3. 모델 구조

GraphENS의 전체 파이프라인 (Algorithm 1 기반):

```
훈련 루프 (t = 1, ..., T):
  ┌── 워밍업 단계 (t ≤ 10):
  │   단순 선형 보간으로 vmixed 생성
  │   vminor의 이웃을 그대로 사용
  │
  └── 본 단계 (t > 10):
      1. KL-Divergence로 φ 계산 (이전 epoch의 logits 사용)
      2. Saliency-based Node Mixing:
         - 마스킹 비율 K = kφ̂ 계산
         - vtarget의 class-specific 피처 마스킹 (MK 생성)
         - vmixed = (1-ΛK)⊙vminor + ΛK⊙vtarget
      3. Neighbor Sampling:
         - p(u|vmixed) = φ̂·p(u|vminor) + (1-φ̂)·p(u|vtarget)
         - 해당 분포에서 이웃 샘플링 (단방향 엣지 설정)
      4. 합성 ego network를 원본 그래프에 추가
      5. 확장된 그래프로 GNN 학습
      6. Confidence 집계 및 saliency 업데이트
```

**타겟 노드 선택 전략**: 모든 클래스에서 타겟 노드를 선택 (동일 소수 클래스 제한 없음)

$$c_{target} \sim \text{Multinomial}(\log N_1, \log N_2, \ldots, \log N_C)$$

여기서 $N_i$는 $i$번째 클래스의 데이터 수. 이는 다수 클래스로의 편향을 방지하면서도 다양성을 확보합니다.

**중요한 구현 세부사항**: 합성된 노드로의 **수신 방향 엣지(incoming edges)에만 메시지 패싱을 허용**하여, 원본 그래프의 다른 노드에 대한 메시지 패싱 영향을 최소화합니다.

---

### 2-4. 성능 향상

#### 지도 학습 결과 (Imbalance Ratio: 100)

**Cora-LT, GCN 기준**:

| Method | Acc. | bAcc. | F1 |
|--------|------|-------|-----|
| Vanilla | 73.66 | 62.72 | 63.70 |
| Re-Weight | 75.20 | 68.79 | 69.27 |
| GraphSMOTE | 76.76 | 69.31 | 70.21 |
| **GraphENS** | **77.76** | **72.94** | **73.13** |

**AmazonComputers (GraphSAGE, Imbalance ratio: 244)**:

| Method | Acc.(bAcc.) | F1 |
|--------|------------|-----|
| Re-Weight | 90.04 | 90.11 |
| GraphSMOTE | 89.31 | 89.39 |
| **GraphENS** | **91.94** | **91.94** |

#### 반지도 학습 결과 (GCN, Cora-Semi):

| Method | Acc. | bAcc. | F1 |
|--------|------|-------|-----|
| Vanilla | 68.38 | 62.04 | 60.92 |
| PC Softmax | 70.96 | 65.34 | 64.63 |
| GraphSMOTE | 69.20 | 63.43 | 62.35 |
| **GraphENS** | **72.68** | **67.67** | **67.94** |

#### Ablation Study (CiteSeer-semi, GraphSAGE):

| Method | Acc | bAcc. | F1 |
|--------|-----|-------|-----|
| GraphENS (w/o SM, NS) | 41.82 | 39.31 | 32.15 |
| GraphENS (w/o SM) | 49.24 | 48.77 | 45.85 |
| GraphENS (w/o PS) | 49.66 | 47.96 | 45.93 |
| **GraphENS** | **51.12** | **48.91** | **46.78** |

→ 각 컴포넌트(Saliency Masking, Neighbor Sampling, Prediction Similarity) 모두 성능에 기여함을 검증.

---

### 2-5. 한계점

논문에 명시되거나 논리적으로 도출 가능한 한계:

1. **계산 비용 증가**: 매 epoch마다 KL-Divergence 계산, saliency 맵 계산, ego network 합성이 필요하여 원본 GNN 학습 대비 추가 연산이 요구됩니다.

2. **워밍업 의존성**: 초기 10 epoch 동안 단순 보간을 사용하며, 모델이 충분히 학습되지 않은 상태의 logit을 활용하는 구조적 제약이 있습니다.

3. **하이퍼파라미터 민감성**: 마스킹 하이퍼파라미터 $k$, 온도 $\tau$, Beta 분포 파라미터 $\alpha$ 등의 튜닝이 필요합니다.

4. **동적 그래프 미적용**: 정적 그래프를 가정하며, 시간에 따라 변하는 동적 그래프에 대한 적용 가능성은 검증되지 않았습니다.

5. **이진 클래스 불균형에 특화**: ImGAGN 등 이진 분류에 특화된 방법과 달리, 다중 클래스를 지원하지만 클래스 수가 매우 많을 경우 타겟 클래스 분포 설계의 최적성이 불분명합니다.

6. **ego network의 1-hop 제한**: 1-hop 이웃만을 합성 대상으로 하며, 더 깊은 이웃 구조는 고려하지 않습니다.

---

## 3. 모델의 일반화 성능 향상 가능성

GraphENS의 일반화 성능 향상은 여러 메커니즘을 통해 이루어집니다.

### 3-1. Neighbor Memorization 완화를 통한 일반화

실험(Table 5)에서 GraphENS는 neighbor-replacing 실험에서 기존 방법 대비 **성능 저하 폭이 현저히 작습니다**. 예를 들어 Cora-LT에서:

| Method | Node(Seen/Unseen) | Neighbor(Seen/Unseen) |
|--------|-------------------|----------------------|
| Re-Weight | 96.80 / 89.74 | 96.80 / 83.94 |
| GraphSMOTE | 96.78 / 89.35 | 96.78 / 83.99 |
| **GraphENS** | **97.94 / 90.48** | **97.94 / 91.06** |

GraphENS는 **미학습 이웃 집합**에 대해서도 높은 정확도를 유지하며, 이는 모델이 특정 이웃 패턴에 과적합되지 않았음을 의미합니다.

### 3-2. Manifold Assumption 활용

논문은 "similar predictions of neural networks indicate the close proximity in the manifold" (Van Engelen & Hoos, 2020)를 핵심 귀납적 편향으로 활용합니다. 전체 클래스에서 타겟 노드를 선택하되, 예측 유사도(KL-Divergence)를 기반으로 ego network를 보간함으로써 **소수 클래스의 결정 경계를 확장하고 평활화(smooth)**합니다.

이는 Verma et al. (2019)의 Manifold Mixup 원리와 유사합니다:

$$v_{mixed} = (1 - \Lambda_K) \odot v_{minor} + \Lambda_K \odot v_{target}$$

**볼록 결합(convex combination)**은 결정 경계의 평활성을 유도합니다 (Figure 6 시각화).

### 3-3. Saliency 기반 Class-Generic 정보 활용

클래스-특화(class-specific) 피처를 마스킹하고 클래스-공통(class-generic) 정보만 결합함으로써, 생성된 소수 클래스 노드는 **클래스 의미론(class semantics)을 보존**하면서도 다양한 특성을 갖게 됩니다.

이는 다음 두 효과를 동시에 달성합니다:
- **표현 다양성 증가** → 일반화 향상
- **클래스 의미 보존** → 노이즈 샘플 생성 방지

### 3-4. 전체 클래스 활용의 이점

동일 소수 클래스 내에서만 타겟을 선택하는 GraphSMOTE와 달리, GraphENS는 **전체 클래스**에서 타겟을 선택합니다(Figure 3에서 실험적 검증). 이는 특히 소수 클래스 샘플이 극히 적을 때 중복 샘플 생성을 방지하고, **합성 ego network의 다양성을 극대화**합니다.

### 3-5. 반지도 학습으로의 확장 가능성

논문은 라벨 없는 노드(unlabeled nodes)를 $(C+1)$번째 클래스로 처리하여 ego network 합성에 활용하는 **GraphENS†** 변형을 제안하며, 이 경우 추가적인 성능 향상이 관찰됩니다(Table 8). 이는 라벨 없는 노드의 풍부한 구조 정보를 활용하여 일반화 성능을 더욱 향상시킬 수 있음을 시사합니다.

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

아래는 논문 내 언급 및 공개된 정보에 근거한 비교입니다.

| 방법 | 발표 | 핵심 아이디어 | 그래프 구조 고려 | 소수 클래스 다양성 | GraphENS 대비 |
|------|------|-------------|-----------------|-----------------|--------------|
| **GraphSMOTE** (Zhao et al., 2021) | WSDM'21 | 두 소수 노드 보간 + 엣지 예측기 | ✅ (엣지 예측) | ❌ (동일 클래스만) | 동일 클래스 제한으로 다양성 부족; neighbor memorization 여전히 존재 |
| **ImGAGN** (Qu et al., 2021) | KDD'21 | GAN으로 소수 노드 생성 | ✅ (임계값 기반 연결) | ❌ (동일 클래스만) | 이진 분류 중심, 다중 클래스 확장 어려움 |
| **DR-GCN** (Shi et al., 2020) | IJCAI'20 | conditional GAN으로 가상 소수 노드 생성 | △ (이웃 유사성 정규화) | △ | 추가 GAN 학습 필요, 불안정성 |
| **cRT** (Kang et al., 2020) | ICLR'20 | 표현/분류기 분리 학습 | ❌ | - | 메시지 패싱에 의한 oversmoothing 시 실패 |
| **PC Softmax** (Hong et al., 2021) | CVPR'21 | logit 사후 보정 | ❌ | - | 표현 학습 자체를 개선하지 않음 |
| **GraphENS** (Park et al., 2022) | ICLR'22 | **전체 ego network 합성** + KL 유사도 기반 이웃 샘플링 + saliency 기반 노드 혼합 | ✅✅ (ego network 전체) | ✅✅ (전체 클래스 활용) | — |

**주요 차별점 요약**:

1. **GraphSMOTE vs. GraphENS**: GraphSMOTE는 동일 소수 클래스 내 두 노드만 활용하여 다양성이 제한됨. GraphENS는 전체 클래스를 활용하고 KL-Divergence 기반 유사도로 품질을 제어함.

2. **ImGAGN vs. GraphENS**: ImGAGN은 독립적인 클래스별 생성기가 필요하여 다중 클래스 확장이 비효율적. GraphENS는 단일 프레임워크로 다중 클래스 대응.

3. **DR-GCN vs. GraphENS**: DR-GCN은 추가 GAN 훈련이 필요하나 GraphENS는 기존 GNN 프레임워크 내에서 동작하며 구조적 변경 불필요.

---

## 5. 앞으로의 연구에 미치는 영향과 고려사항

### 5-1. 연구에 미치는 영향

**① Neighbor Memorization의 새로운 문제 정의**

GraphENS는 그래프 도메인에서 기존에 간과되었던 **"neighbor memorization"** 문제를 최초로 체계적으로 정의하고 실험적으로 입증했습니다. 이는 그래프 기반 클래스 불균형 연구의 새로운 방향을 제시하며, 향후 연구들이 이웃 구조 관련 과적합 문제를 명시적으로 고려하도록 유도합니다.

**② Ego Network 단위 증강의 패러다임 제시**

노드 피처만을 합성하는 기존 방법과 달리, **ego network 전체(중심 노드 + 이웃)**를 하나의 증강 단위로 취급하는 새로운 패러다임을 확립했습니다. 이는 그래프 증강 연구 전반에 영향을 미칠 수 있습니다.

**③ 사전 학습 모델과의 결합 가능성**

GraphENS의 saliency 기반 접근법은 GNN 기반 사전 학습 모델(pre-trained GNN)과 결합하여, 적은 레이블로도 효과적인 소수 클래스 증강이 가능한 방향으로 발전할 수 있습니다.

**④ 이종 그래프 및 지식 그래프로의 확장**

ego network 합성 프레임워크는 이종 그래프(heterogeneous graph)나 지식 그래프(knowledge graph)에서의 클래스 불균형 문제에 적용될 수 있는 잠재력을 가집니다.

### 5-2. 향후 연구 시 고려할 점

**① 더 깊은 이웃 구조 고려**

현재 GraphENS는 1-hop 이웃만을 합성합니다. 2-hop 이상의 이웃 구조를 고려한 **multi-hop ego network 합성**은 더 풍부한 맥락 정보를 제공할 수 있습니다. 다만, 계산 복잡도와의 트레이드오프를 신중히 고려해야 합니다.

**② 동적 그래프 적용**

실세계의 많은 그래프는 시간에 따라 변화합니다. **동적 그래프(temporal/dynamic graph)에서의 neighbor memorization** 문제와 GraphENS의 적용 가능성은 중요한 연구 방향입니다.

**③ KL-Divergence 대안 탐색**

두 ego network 간 유사도를 측정하기 위해 KL-Divergence를 사용하지만, 이는 비대칭적(non-symmetric)이며 수치적 불안정성이 있을 수 있습니다. **Jensen-Shannon Divergence**, **Wasserstein Distance** 등의 대안적 유사도 측정 방법의 비교 연구가 필요합니다.

**④ 설명 가능성(Explainability) 강화**

Saliency 기반 피처 마스킹은 gradient에 의존하는데, gradient saliency의 신뢰성에 대한 논란이 있습니다 (Simonyan et al., 2013 방법의 한계). **SHAP, GNNExplainer** 등 보다 강건한 설명 방법을 활용한 클래스-특화 피처 분리 연구가 필요합니다.

**⑤ 라지 스케일 그래프 효율화**

현재 구현은 전체 그래프의 logit을 매 epoch 계산하여, 수백만 노드를 가진 대규모 그래프에서는 확장성 문제가 발생할 수 있습니다. **미니배치 기반 GraphENS** 또는 **근사 ego network 합성** 방법 개발이 필요합니다.

**⑥ 자기지도 학습(Self-supervised Learning)과의 결합**

GraphENS의 ego network 합성 프레임워크를 **대조 학습(contrastive learning)** 기반 그래프 자기지도 학습(예: GraphCL, SimGRACE 등)과 결합하면, 라벨 없는 데이터를 더욱 효과적으로 활용할 수 있을 것으로 예상됩니다.

**⑦ 클래스 불균형 탐지 자동화**

실세계 데이터에서 불균형 비율은 알려지지 않을 수 있습니다. 불균형 정도를 자동으로 탐지하고 오버샘플링 비율을 적응적으로 조정하는 **메타러닝(meta-learning) 기반 GraphENS** 확장이 유망한 방향입니다.

---

> **⚠️ 중요 고지**: 위 내용 중 "5-2. 향후 연구 시 고려할 점"의 일부(특히 ①~⑦)는 논문의 한계점과 관련 분야 트렌드를 바탕으로 한 **합리적 추론**이며, 논문 원문에 직접 명시된 내용은 아닙니다. 이를 명확히 구분하여 참고하시기 바랍니다.
