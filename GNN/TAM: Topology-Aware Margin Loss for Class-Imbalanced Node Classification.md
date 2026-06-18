# TAM: Topology-Aware Margin Loss for Class-Imbalanced Node Classification 

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

기존의 클래스 불균형 처리 방법들은 소수 클래스 노드를 **집단적으로(as a group)** 보상하는 방식을 사용한다. 이 과정에서 **그래프의 위상(topology) 정보를 무시**하기 때문에, 다수 클래스와 비정상적으로 많이 연결된 소수 노드를 과도하게 보상하여 **다수 클래스의 False Positive(위양성)가 급증**한다는 것이 핵심 문제 제기이다.

TAM은 각 노드의 **지역 위상(local topology)**을 학습 목표에 반영하여, 노드별로 마진을 적응적으로 조절함으로써 이 문제를 해결한다.

### 주요 기여 (3가지)

| 기여 | 설명 |
|------|------|
| **현상 규명** | 소수 노드 보상으로 인한 위양성은 그래프 전체에 균등하지 않으며, 다수 클래스와 비정상적으로 연결된 소수 노드 주변에 집중됨을 실험적으로 증명 |
| **방법론 제안** | 노드의 위상을 클래스 통계와 비교하여 개별적으로 마진을 조절하는 TAM(Topology-Aware Margin) 제안 |
| **플러그인 호환성** | 기존 불균형 처리 방법(GraphSMOTE, ReNode, GraphENS 등)과 독립적으로 결합 가능하며 일관된 성능 향상 제공 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**그래프 기반 노드 분류에서의 클래스 불균형 문제**는 단순한 수량 불균형을 넘어선다. GNN의 **메시지 패싱(message passing)** 특성으로 인해, 다수 클래스 이웃을 많이 가진 소수 클래스 노드를 과도하게 보상하면, 해당 노드가 주변 다수 클래스 노드의 표현 학습을 왜곡시켜 **다수 클래스에 대한 위양성**을 급증시킨다.

기존 방법(Re-Weight, Balanced Softmax, GraphSMOTE 등)의 한계:
- 소수 클래스를 **수량 기준으로 일괄 보상** → 위상 정보 무시
- **ReNode**: 위상 경계 근처 노드 가중치 감소 → 동질 그래프(homophilous graph)에서만 유효, 클래스 쌍별 연결성 미고려

### 2.2 제안 방법 (수식 포함)

#### 핵심 개념 정의

**정의 1: 이웃 레이블 분포 (Neighbor Label Distribution, NLD)**

$$\mathcal{D}_{i,j} = \frac{|\{v \in \mathcal{N}(i) \cup \{i\} \mid y_v = j\}|}{d_i + 1} \tag{1}$$

- $\mathcal{D} \in \mathbb{R}^{|V| \times |\mathcal{Y}|}$: 각 노드 $i$의 이웃 레이블 분포 행렬
- $\mathcal{N}(i)$: 노드 $i$의 인접 노드 집합, $d_i$: 노드 $i$의 차수

**정의 2: 클래스별 연결성 행렬 (Class-wise Connectivity Matrix)**

$$\mathcal{C}_{i,j} = \frac{1}{|\{v \in V \mid y_v = i\}|} \sum_{u \in \{v \in V \mid y_v = i\}} \mathcal{D}_{u,j} \tag{2}$$

- $\mathcal{C} \in \mathbb{R}^{|\mathcal{Y}| \times |\mathcal{Y}|}$: 클래스 $i$에 속하는 노드들의 평균 이웃 레이블 분포

#### 전체 학습 목표 함수

$$\mathcal{L}_{TAM} = \frac{1}{|V^L|} \sum_{v \in V^L} \mathcal{L}\left(l_v + \alpha m_v^{ACM} + \beta m_v^{ADM},\ y_v\right) \tag{5}$$

- $l_v \in \mathbb{R}^{|\mathcal{Y}|}$: 노드 $v$의 로짓 벡터
- $m_v^{ACM} \in \mathbb{R}^{|\mathcal{Y}|}$: 비정상 연결성 인식 마진 (Anomalous Connectivity-aware Margin)
- $m_v^{ADM} \in \mathbb{R}^{|\mathcal{Y}|}$: 비정상 분포 인식 마진 (Anomalous Distribution-aware Margin)
- $\alpha, \beta$: 각 항의 강도를 조절하는 하이퍼파라미터

#### 구성 요소 1: ACM (Anomalous Connectivity-aware Margin)

노드 $v$의 클래스 $t$에 대한 ACM:

$$m_{v,t}^{ACM} = -\max\left(\log\left(\frac{\mathcal{C}_{y_v, y_v}}{\mathcal{D}_{v, y_v}}\right) \cdot \left(\frac{\mathcal{D}_{v,t}}{\mathcal{C}_{y_v, t}}\right),\ 0\right) \tag{6}$$

**직관적 해석:**
- $\frac{\mathcal{C}\_{y_v, y_v}}{\mathcal{D}_{v, y_v}}$ 가 크면 → 노드 $v$가 자신의 클래스 동질성 패턴을 따르지 않음 → 전체 마진 감소
- $\frac{\mathcal{D}\_{v,t}}{\mathcal{C}_{y_v, t}}$ 가 크면 → 노드 $v$가 클래스 $t$와 클래스 평균보다 많이 연결 → 클래스 $t$의 마진 추가 감소
- 자신의 클래스($t = y_v$)에 대한 마진은 0으로 설정됨

#### 구성 요소 2: ADM (Anomalous Distribution-aware Margin)

NLD 공간에서 코사인 법칙을 이용한 상대적 거리 계산:

$$\cos A_{v,t} = \frac{JS(\mathcal{D}_{v,:}, \mathcal{C}_{y_v,:})^2 + JS(\mathcal{C}_{t,:}, \mathcal{C}_{y_v,:})^2 - JS(\mathcal{D}_{v,:}, \mathcal{C}_{t,:})^2}{2 \cdot JS(\mathcal{D}_{v,:}, \mathcal{C}_{y_v,:}) \cdot JS(\mathcal{C}_{t,:}, \mathcal{C}_{y_v,:})} \tag{7}$$

$$m_{v,t}^{ADM} = -\frac{JS(\mathcal{D}_{v,:},\ \mathcal{C}_{y_v,:}) \cdot \cos A_{v,t}}{JS(\mathcal{C}_{t,:},\ \mathcal{C}_{y_v,:})} \tag{8}$$

- $JS(\cdot, \cdot)$: Jensen-Shannon 발산 (Jensen-Shannon Divergence)
- $A_{v,t}$: 자기 클래스 평균 NLD → 노드 NLD의 방향벡터와 자기 클래스 평균 NLD → 목표 클래스 평균 NLD의 방향벡터 사이의 각도
- 노드가 목표 클래스 NLD에 가까울수록 마진을 더 크게 감소시킴 (구별하기 어려운 클래스에 대해 더 강하게 보정)

#### 구성 요소 3: 클래스별 온도 스케일링 (Class-wise Temperature)

레이블이 없는 노드의 예측 정확도를 높이기 위해:

$$\pi_k = \delta \cdot \frac{N_k}{\frac{1}{|\mathcal{Y}|}\sum_{s \in |\mathcal{Y}|} N_s} + (1 - \delta)$$

$$T_k = \frac{1}{\phi\left(\pi_k + 1 - \max_j \pi_j\right)} \tag{9}$$

- $N_k$: 클래스 $k$의 노드 수, $\delta = 0.4$ (고정), $\phi$: 하이퍼파라미터
- 소수 클래스($N_k$ 작음) → 큰 $T_k$ → 로짓을 더 크게 스케일링 → 소수 클래스 예측 편향 완화
- **훈련에는 사용하지 않고**, 이웃 노드 레이블 추정에만 활용

### 2.3 모델 구조 및 파이프라인

```
[입력] 그래프 G(X, V, E, Y)
    ↓
[Step 1] GNN으로 노드 로짓 계산 → 클래스별 온도 적용
    ↓
[Step 2] 이웃 레이블 분포 D 계산
         (레이블 있는 노드: 원핫 벡터 / 레이블 없는 노드: Softmax 예측 활용)
    ↓
[Step 3] 클래스별 연결성 행렬 C 계산
    ↓
[Step 4] ACM 계산: m^ACM (비정상 연결성 기반 마진)
    ↓
[Step 5] ADM 계산: m^ADM (JS 발산 + 코사인 각도 기반)
    ↓
[Step 6] 조정된 로짓으로 손실 계산: L_TAM
    ↓
[출력] 훈련된 GNN 파라미터 θ
```

TAM은 기존 GNN 아키텍처(GCN, GAT, GraphSAGE)의 출력 로짓만 조정하므로, **어떤 GNN 및 불균형 처리 방법과도 결합 가능**하다.

### 2.4 성능 향상

**동질 그래프(Homophilous Graphs)** - bAcc. 기준 (ρ=10):

| 데이터셋 | 베이스라인 (BalancedSoftmax, GCN) | +TAM |
|---------|----------------------------------|------|
| Cora | 68.46 | **69.90** (+1.44) |
| CiteSeer | 53.70 | **55.54** (+1.84) |
| PubMed | 72.97 | **74.13** (+1.16) |

**이질 그래프(Heterophilous Graphs)** - bAcc. 기준 (ρ=5):

| 데이터셋 | 베이스라인 (BalancedSoftmax, GAT) | +TAM |
|---------|-----------------------------------|------|
| Chameleon | 41.47 | **42.56** (+1.09) |
| Wisconsin | 41.20 | **48.44** (+7.24) |

- 모든 9개 설정(3 데이터셋 × 3 아키텍처)에서 최고 성능 달성 (동질 그래프)
- 이질 그래프에서도 대부분의 경우 성능 향상

**Ablation Study 결과** (CiteSeer + GCN, F1 기준):

| 구성 | F1 |
|------|-----|
| 기본 (BalancedSoftmax) | 50.73 |
| + ACM | 53.54 |
| + ADM | 51.95 |
| + ACM + ADM | 54.08 |
| + ACM + ADM + Cls-wise Tk | **55.54** |

### 2.5 한계

1. **저정확도 환경에서의 성능 저하**: Squirrel 데이터셋처럼 기본 정확도가 낮은 경우, NLD와 $\mathcal{C}$ 추정이 부정확해져 개선 효과가 미미하다.
2. **하이퍼파라미터 민감성**: $\alpha$, $\beta$ 값이 극단적으로 크거나 작으면 성능이 저하된다.
3. **계산 비용**: 매 에폭마다 NLD와 $\mathcal{C}$를 재계산해야 하므로, 대규모 그래프에서 계산 복잡도가 증가할 수 있다.
4. **ReNode와의 결합 효과 제한**: 일부 설정(특히 GAT + ReNode + CiteSeer 등)에서 TAM 추가 시 소폭 성능 저하가 관찰되었다.
5. **전이 학습 시나리오 미검증**: 인덕티브(inductive) 설정이나 새로운 그래프에 대한 일반화 검증이 부족하다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화 성능 향상의 이론적 근거

TAM이 일반화에 기여하는 핵심 메커니즘은 **마진 기반 학습의 이론적 보장**에 있다. 논문에서 언급하는 Balanced Softmax는 다음의 일반화 경계(generalization bound)를 최소화한다:

$$\mathcal{L} = \mathcal{L}_{CE}(l_v + m, y_v) = -\log\left(\frac{e^{l_{v,y_v} + \log N_{y_v}}}{\sum_{k \in \mathcal{Y}} e^{l_{v,k} + \log N_k}}\right) \tag{4}$$

TAM은 이를 **노드별 위상 정보로 세밀화**함으로써, 단순 수량 기반 마진보다 더 정밀한 결정 경계를 형성한다.

### 3.2 일반화 성능 향상의 실증적 근거

**① 플러그인 호환성을 통한 일반화:**
- logit 조정(Balanced Softmax), 노드 가중치 조정(ReNode), 오버샘플링(GraphENS) 등 **서로 다른 전략의 베이스라인 모두에서 일관된 개선**을 보임
- 이는 TAM이 특정 방법에 과적합된 것이 아님을 시사

**② 다양한 그래프 구조에서의 일반화:**
- 동질 그래프(Cora, CiteSeer, PubMed)와 이질 그래프(Chameleon, Squirrel, Wisconsin) 모두에서 효과적
- 기존 ReNode가 이질 그래프에서 효과가 없는 것과 대비됨

**③ 다양한 GNN 아키텍처에서의 일반화:**
- GCN, GAT, GraphSAGE 등 서로 다른 메시지 패싱 방식의 아키텍처 모두에서 일관된 성능 향상

**④ 다양한 불균형 비율(ρ)에서의 일반화:**
- $\rho = 5$와 $\rho = 10$ 두 설정 모두에서 일관된 개선 확인

### 3.3 일반화 측면의 한계 및 개선 필요 사항

- **레이블 없는 노드 추정의 품질**: TAM은 미레이블 노드의 클래스를 모델 예측으로 추정하는데, 초기 학습 단계(warmup 5 epoch)에서 이 추정이 부정확할 수 있다. 이를 개선하면 초기 학습의 안정성이 향상되어 일반화에 도움이 될 것이다.
- **분포 변화(Distribution Shift) 대응**: 학습 시 추정한 $\mathcal{C}$ 행렬이 테스트 시의 그래프 구조와 다를 경우 성능이 저하될 수 있다. 이는 특히 동적 그래프나 전이 학습 시나리오에서 문제가 된다.
- **저성능 기반 환경**: Squirrel처럼 기본 GNN 성능이 낮은 경우, 부정확한 예측으로 인한 노이즈가 TAM의 일반화를 방해한다.

---

## 4. 최신 관련 연구 비교 분석 (2020년 이후)

### 4.1 비교 분석 표

| 방법 | 연도 | 주요 접근 | 위상 고려 | 이질 그래프 지원 | 플러그인 호환 |
|------|------|-----------|-----------|----------------|--------------|
| **GraphSMOTE** (Zhao et al.) | 2021 | 소수 노드 보간 합성 | 부분적 | ❌ | ❌ |
| **DR-GCN** (Shi et al.) | 2020 | 조건부 GAN 노드 생성 | ❌ | ❌ | ❌ |
| **ReNode** (Chen et al.) | 2021 | 위상 경계 노드 가중치 조절 | ✅ (경계만) | ❌ | ❌ |
| **GraphENS** (Park et al.) | 2021 | 이웃 분포 수준 노드 합성 | ✅ | ❌ | ❌ |
| **ImGAGN** (Qu et al.) | 2021 | GAN 기반 소수 노드 생성 | ❌ | ❌ | ❌ (이진 분류) |
| **Balanced Softmax** (Ren et al.) | 2020 | 수량 기반 로짓 조정 | ❌ | ⚠️ (제한적) | ✅ |
| **TAM** (Song et al.) | 2022 | 위상 인식 노드별 마진 조정 | ✅ (세밀) | ✅ | ✅ |

### 4.2 TAM 이후 관련 연구 방향

TAM 이후에도 그래프 기반 클래스 불균형 처리 연구는 계속 발전하고 있다. 다만, 본 논문(2022년 ICML)이 발표된 이후의 후속 연구에 대해서는 제공된 PDF 원문에 포함되어 있지 않으므로, **해당 내용은 제 학습 데이터 기반의 일반적 지식으로 제공하며, 구체적 수치나 결과는 확인이 필요하다**는 점을 명시한다.

일반적인 후속 연구 흐름:
- **LTE4G** (Long-Tail Expertise for Graphs): 전문가 모델 앙상블을 그래프에 적용
- **GraphMixup**: 그래프 구조를 고려한 Mixup 기반 데이터 증강
- **Topology-aware Oversampling**: TAM의 위상 인식 개념을 오버샘플링과 결합하는 방향

---

## 5. 향후 연구에 미치는 영향 및 고려 사항

### 5.1 향후 연구에 미치는 영향

**① 위상 인식 손실 함수의 패러다임 확립**

TAM은 그래프 학습에서 손실 함수 설계 시 **노드의 지역 위상을 반드시 고려해야 한다**는 인식을 확립했다. 단순히 클래스 수량만을 고려하는 기존 방식에서 벗어나, **이웃 레이블 분포 통계를 활용한 적응적 마진 조정**이라는 새로운 방향을 제시했다.

**② 이질 그래프에서의 불균형 처리 가능성 입증**

이전 연구들이 주로 동질 그래프에서만 효과를 보인 것과 달리, TAM은 이질 그래프(Chameleon, Wisconsin, Squirrel)에서도 효과적임을 보여주었다. 이는 향후 **다양한 실세계 그래프**(소셜 네트워크, 사기 탐지 네트워크 등)에서의 적용 가능성을 높였다.

**③ 플러그인 방식의 설계 철학**

TAM의 플러그인 설계는 이후 연구에서 **모듈러 방식의 손실 함수 설계**를 촉진하는 영향을 미쳤다. 기존 방법을 대체하는 것이 아니라 보완하는 방식은 실용적 적용 측면에서 중요한 기여이다.

**④ 클래스별 연결성 행렬($\mathcal{C}$)의 활용**

$\mathcal{C}$ 행렬의 개념은 그래프의 전역적 클래스 상호작용 패턴을 정량화하는 유용한 도구로, 향후 **그래프 이질성 분석, 커뮤니티 탐지, 링크 예측** 등 다양한 분야에서 응용될 수 있다.

### 5.2 향후 연구 시 고려해야 할 점

**① 동적 그래프(Dynamic Graph)로의 확장**

현재 TAM은 정적 그래프를 가정한다. 실세계 그래프는 시간에 따라 변화하므로, **시간적으로 변화하는 NLD와 $\mathcal{C}$ 행렬을 효율적으로 업데이트**하는 방법이 필요하다. 이를 위해 온라인 학습(online learning) 프레임워크와의 결합을 고려해야 한다.

**② 대규모 그래프에서의 계산 효율성**

매 에폭마다 모든 노드의 NLD와 $\mathcal{C}$를 재계산하는 것은 수백만 노드를 가진 대규모 그래프에서 계산 병목이 된다. **미니배치 기반 NLD 추정**이나 **근사 알고리즘** 개발이 필요하다.

**③ 레이블 잡음(Label Noise)에 대한 강건성**

현재 TAM은 레이블이 정확하다고 가정하지만, 실세계에서는 레이블 잡음이 흔하다. 잡음이 있는 레이블이 $\mathcal{C}$ 행렬 추정에 미치는 영향을 분석하고, **잡음 강건 버전**을 개발하는 것이 중요하다.

**④ 극심한 불균형(Extreme Imbalance) 시나리오**

현재 실험에서는 최대 $\rho = 10$의 불균형 비율을 다루었다. 실세계의 사기 탐지나 희귀 질병 예측 등의 도메인에서는 $\rho > 100$ 이상의 극심한 불균형이 존재할 수 있다. 이런 극단적 설정에서의 성능과 NLD 추정의 신뢰성을 검증해야 한다.

**⑤ 전이 학습 및 Few-shot 시나리오**

TAM이 학습 과정에서 $\mathcal{C}$ 행렬을 추정하는 방식은 전이 학습 시나리오(새로운 그래프에 적용)에서 검증되지 않았다. **메타러닝(meta-learning)**이나 **few-shot 그래프 학습**과의 결합 가능성을 탐구할 필요가 있다.

**⑥ 이론적 보장의 강화**

현재 일반화 경계는 Balanced Softmax의 결과를 차용하는 수준이다. TAM의 위상 인식 마진 조정이 일반화 오차에 미치는 영향을 **PAC-Bayes 프레임워크** 등을 통해 이론적으로 분석하면 방법론의 신뢰성이 높아질 것이다.

**⑦ 자기지도학습(Self-supervised Learning)과의 결합**

레이블 없는 노드의 정보를 활용하기 위해, TAM의 클래스별 온도 스케일링을 **대조 학습(contrastive learning)** 기반의 표현 학습과 결합하면 소수 클래스의 표현 품질을 더욱 향상시킬 수 있을 것이다.

---

## 참고 자료

**주 논문:**
- Song, J., Park, J., & Yang, E. (2022). **TAM: Topology-Aware Margin Loss for Class-Imbalanced Node Classification**. *Proceedings of the 39th International Conference on Machine Learning (ICML 2022)*, PMLR 162. (제공된 PDF 원문)

**논문 내 인용 주요 참고문헌:**
- Ren, J. et al. (2020). Balanced meta-softmax for long-tailed visual recognition. *NeurIPS 33*.
- Zhao, T. et al. (2021). GraphSMOTE: Imbalanced node classification on graphs with graph neural networks. *WSDM 2021*.
- Park, J. et al. (2021). GraphENS: Neighbor-aware ego network synthesis for class-imbalanced node classification. *ICLR 2021*.
- Chen, D. et al. (2021). Topology-imbalance learning for semi-supervised node classification. *NeurIPS 34*.
- Shi, M. et al. (2020). Multiclass imbalanced graph convolutional network learning. *IJCAI 2020*.
- Cao, K. et al. (2019). Learning imbalanced datasets with label-distribution-aware margin loss. *NeurIPS 32*.
- Menon, A. K. et al. (2020). Long-tail learning via logit adjustment. *ICLR 2020*.
- Welling, M. & Kipf, T. N. (2016). Semi-supervised classification with graph convolutional networks. *ICLR 2017*.
- Veličković, P. et al. (2018). Graph attention networks. *ICLR 2018*.
- Hamilton, W. et al. (2017). Inductive representation learning on large graphs. *NeurIPS 30*.
- Pei, H. et al. (2019). Geom-GCN: Geometric graph convolutional networks. *ICLR 2019*.
- Rozemberczki, B. et al. (2021). Multi-scale attributed node embedding. *Journal of Complex Networks*.
- Sen, P. et al. (2008). Collective classification in network data. *AI Magazine 29(3)*.
