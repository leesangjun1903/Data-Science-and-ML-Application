# Graph Convolutional Networks for Learning with Few Clean and Many Noisy Labels

## 논문 정보
- **저자**: Ahmet Iscen, Giorgos Tolias, Yannis Avrithis, Ondřej Chum, Cordelia Schmid
- **소속**: Google Research, CTU Prague, Inria
- **arXiv**: 1910.00324v3 (2020년 8월 24일)

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장
이 논문은 **소수의 클린 레이블(few clean labels)과 다수의 노이지 레이블(many noisy labels)이 공존하는 환경**에서 효과적으로 분류기를 학습하는 새로운 프레임워크를 제안합니다. GCN을 이진 분류기(binary classifier)로 활용하여 노이지 예제의 클래스 관련성(relevance)을 예측하고, 이를 가중치로 사용하여 최종 분류기를 학습합니다.

### 주요 기여 3가지
1. **소수의 클린 + 대규모 노이지 데이터를 결합한 분류기 학습** 프레임워크 제안
2. **GCN을 이진 분류기로 활용한 노이지 데이터 정제(cleaning)** — 논문에 따르면 이 접근법은 최초 시도
3. Low-Shot ImageNet 및 Low-Shot Places365, Mini-ImageNet 벤치마크에서 **기존 최고 성능(SOTA) 대비 유의미한 정확도 향상** 달성

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능, 한계

### 2.1 해결하고자 하는 문제

**문제 설정**:
- 클래스 집합 $C$ (크기 $|C| = K$)에 대해 클래스당 클린 예제가 $k$개 ($k \in \{1, 2, 5, 10, 20\}$)만 존재
- 클린 예제 집합: $X_C$, 클래스 $c$에 대한 클린 집합: $X_C^c$
- 추가로 노이지 레이블이 붙은 대규모 데이터셋 $X_N$이 존재
- 확장 집합: $X_E^c = X_C^c \cup X_N^c$
- **목표**: 노이지 데이터를 활용하여 $K$-way 분류기의 정확도를 향상

**핵심 도전 과제**:
- 노이지 데이터를 아무 처리 없이 사용하면 성능이 오히려 저하될 수 있음
- 클린 데이터가 극히 적어 노이지 정제를 위한 지도 신호가 부족함
- 전이학습(transfer learning)으로 얻은 특징 추출기(feature extractor)는 고정되어야 하므로 표현 학습과 분리된 정제 방법 필요

---

### 2.2 제안 방법 (수식 포함)

#### Step 1: 그래프 구성

각 클래스 $c$에 대해, 클린 + 노이지 데이터의 특징 행렬 $V = [\mathbf{v}\_1, \ldots, \mathbf{v}\_N] \in \mathbb{R}^{d \times N}$을 구성합니다. ( $\mathbf{v}\_i = g_\theta(x_i)$ )

**친화도 행렬(Affinity Matrix)** $A \in \mathbb{R}^{N \times N}$:

$$a_{ij} = [\mathbf{v}_i^\top \mathbf{v}_j]_+ \quad \text{if } \mathbf{v}_i, \mathbf{v}_j \text{ are reciprocal nearest neighbors, else } 0$$

자기 연결(self-connection)을 추가한 정규화 행렬:

$$\tilde{A} = D^{-1}(A + I), \quad D = \text{diag}((A+I)\mathbf{1})$$

#### Step 2: GCN 구조

각 GCN 레이어는 다음 형태의 함수:

$$f_\Theta(\tilde{A}, Z) = h(\Theta^\top Z \tilde{A}) \tag{1}$$

여기서 $Z \in \mathbb{R}^{l \times N}$은 입력 특징, $\Theta \in \mathbb{R}^{l \times n}$은 학습 파라미터, $h$는 비선형 활성화 함수입니다.

본 논문에서 사용하는 **2-layer GCN** (스칼라 출력):

$$F_\Theta(\tilde{A}, V) = \sigma(\Theta_2^\top [\Theta_1^\top V \tilde{A}]_+ \tilde{A}) \tag{2}$$

- $\Theta_1 \in \mathbb{R}^{d \times m}$, $\Theta_2 \in \mathbb{R}^{m \times 1}$
- $[\cdot]_+$: ReLU 함수
- $\sigma(a) = (1 + e^{-a})^{-1}$: 시그모이드 함수
- 출력: 각 예제 $x_i$에 대한 클래스 관련성 값 $F_\Theta(\tilde{A}, V)_i \in [0, 1]$

#### Step 3: 손실 함수 (이진 크로스 엔트로피)

GCN을 이진 분류기로 학습: 클린 예제 → 타겟 1, 노이지 예제 → 타겟 0

$$\mathcal{L}_G(V, \tilde{A}; \Theta) = -\frac{1}{k}\sum_{i=1}^{k} \log\left(F_\Theta(\tilde{A}, V)_i\right) - \frac{\lambda}{N-k}\sum_{i=k+1}^{N} \log\left(1 - F_\Theta(\tilde{A}, V)_i\right) \tag{3}$$

- 앞의 항: 클린 예제를 양성(positive)으로 분류하는 손실
- 뒤의 항: 노이지 예제를 음성(negative)으로 분류하는 손실
- $\lambda$: 노이지 예제의 중요도 가중치 (하이퍼파라미터)

> **핵심 아이디어**: 노이지 예제를 무조건 음성으로 설정하더라도, 그래프 전파를 통해 클린 예제와 강하게 연결된 노이지 예제는 높은 관련성 점수를 받을 수 있음. 이는 Noise-Contrastive Estimation(NCE)과 유사한 원리.

#### Step 4: 관련성 기반 분류기 학습

**관련성 가중치 정의**:

$$r(x_i) = \begin{cases} F_\Theta(\tilde{A}, V)_i & \text{if } x_i \in X_N^c \\ 1 & \text{if } x_i \in X_C^c \end{cases}$$

**방법 1: 클래스 프로토타입(Class Prototype)**

$$\mathbf{w}_c = \frac{1}{r(X_E^c)} \sum_{x \in X_E^c} r(x) g_\theta(x) \tag{5}$$

**방법 2: 코사인 분류기 학습(Cosine Classifier Learning)**

$$L(C, X_E, \theta; W) = -\sum_{c \in C} \frac{1}{r(X_E^c)} \sum_{x \in X_E^c} r(x) \log(\boldsymbol{\sigma}(s\hat{W}^\top \hat{g}_\theta(x))_c) \tag{6}$$

**방법 3: 딥 네트워크 파인튜닝(Deep Network Fine-tuning)**

- 위 식 (6)에서 특징 추출기 파라미터 $\theta$도 함께 최적화
- 클린 + 노이지 데이터 모두 접근 필요

**분류(예측)**:

$$\pi_{\theta, W}(x) = \arg\max_c \hat{\mathbf{w}}_c^\top \hat{g}_\theta(x) \tag{4}$$

---

### 2.3 모델 구조 요약

```
[입력: 클린 k장 + 노이지 N-k장]
        ↓
[특징 추출기 g_θ (ResNet-10 or ResNet-50, 고정)]
        ↓
[클래스별 친화도 그래프 구성 (Reciprocal k-NN, k=50)]
        ↓
[2-layer GCN → 관련성 점수 r(x_i) ∈ [0,1] 출력]
        ↓ (이진 크로스 엔트로피 손실로 학습)
[관련성 가중치 기반 분류기 학습]
    ├── 클래스 프로토타입 (5)
    ├── 코사인 분류기 (6)
    └── 딥 네트워크 파인튜닝 (8)
        ↓
[K-way 분류기 πθ,W]
```

**주요 구현 세부사항**:
- GCN: Adam optimizer, lr=0.1, 100 iterations, dropout=0.5
- 내부 표현 차원: $m = 16$
- Reciprocal top-50 최근접 이웃으로 친화도 행렬 구성
- 입력 차원: ResNet-10 → $d=512$, ResNet-50 (PCA 후) → $d=256$

---

### 2.4 성능 향상

**Low-Shot ImageNet (Novel Classes, Top-5 Accuracy)**:

| 방법 | k=1 | k=5 | k=20 |
|------|-----|-----|------|
| Class proto. (클린만) [9] | 45.3 | 69.3 | 77.8 |
| Label Propagation | 62.6 | 74.6 | 77.7 |
| MLP | 63.6 | 73.7 | 77.6 |
| **Ours (GCN, proto.)** | **67.8** | **73.9** | **78.2** |
| Ours (cosine classifier) | 73.2 | 75.6 | 80.7 |
| Ours (fine-tune) | 74.1 | 77.7 | 82.6 |
| Diffusion-logistic [5] (ResNet-50) | 64.0 | 79.7 | 86.3 |
| **Ours (fine-tune, ResNet-50)** | **80.2** | **83.3** | **88.3** |

- 클린 데이터만 사용 대비 1-shot에서 **+20% 이상** 개선 (45.3 → 67.8, proto 기준)
- 같은 데이터를 사용한 Diffusion [5] 대비 모든 k에서 우수 (ResNet-50 기준, cosine/fine-tune)

**Mini-ImageNet (5-way classification)**:

| 방법 | k=1 | k=5 |
|------|-----|-----|
| Class proto. [9] | 54.2 | 71.2 |
| Label Propagation | 67.0 | 74.8 |
| **Ours (GCN)** | **68.2** | **74.7** |

---

### 2.5 한계점

1. **고노이즈 클래스 실패**: "muzzle" 클래스처럼 노이즈 비율이 94%에 달하는 경우 관련성 가중치 할당이 부정확 (양성 평균 0.18 < 음성 평균 0.30)
2. **클린 데이터 부재 클래스**: 텍스트 쿼리로 노이지 데이터를 전혀 수집할 수 없는 클래스(예: maillot, missile)에는 적용 불가
3. **특징 추출기 고정 의존성**: 표현 학습과 노이지 정제가 분리되어 있어, 특징 추출기 품질에 크게 의존
4. **클래스명 필요**: 텍스트 기반 크롤링을 위해 클래스명이 사전에 알려져야 함 → 추상적 개념 클래스에 적용 어려움
5. **λ 하이퍼파라미터 민감성**: k가 달라지면 최적 λ가 달라지며, 검증셋 튜닝 필요
6. **그래프 구성 비용**: 클래스별 별도 그래프 구성이 필요하나, 100M 전체 그래프보다는 효율적

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화 향상의 핵심 메커니즘

**그래프 기반 구조 활용으로 인한 일반화**:

GCN이 단순한 MLP에 비해 우수한 이유는 정규화(affinity matrix $\tilde{A}$)를 통한 **정보 전파**에 있습니다. 클린 예제와 연결된 노이지 예제는 그래프 구조를 통해 관련성이 "전파"되므로, k=1의 극단적 제한 상황에서도 주변 샘플의 정보를 활용할 수 있습니다.

$$F_\Theta(\tilde{A}, V) = \sigma(\Theta_2^\top [\Theta_1^\top V \tilde{A}]_+ \tilde{A})$$

위 식에서 $\tilde{A}$가 두 번 적용됨으로써 **2-hop 이웃 정보까지 집계**됩니다. 이는 클린 예제가 1개뿐인 상황에서 주변 노이지 데이터를 통해 클래스 분포를 더 잘 추정할 수 있음을 의미합니다.

### 3.2 일반화를 위한 3단계 분리(Decoupling) 전략

논문은 세 단계를 명시적으로 분리합니다:
1. **표현 학습**: 베이스 클래스로 feature extractor $g_\theta$ 학습 (고정)
2. **노이지 정제**: GCN으로 관련성 점수 추정
3. **분류기 학습**: 관련성 가중치 기반 분류기 최적화

이 분리 전략은 few-shot 설정에서의 **과적합(overfitting) 위험을 근본적으로 낮춥니다**. 클린 예제가 k=1개인 상황에서도 전체 파이프라인이 안정적으로 작동하는 이유입니다.

### 3.3 다양한 도메인으로의 일반화 실험 결과

| 데이터셋 | 아키텍처 | 제안 방법 (fine-tune) k=1 |
|---------|---------|--------------------------|
| Low-Shot ImageNet | ResNet-10 | 74.1 |
| Low-Shot ImageNet | ResNet-50 | 80.2 |
| Low-Shot Places365 | ResNet-10 | 51.8 |
| Mini-ImageNet | ConvNet-128 | 68.2 |

- ImageNet(객체 분류)과 Places365(장면 분류)라는 이질적 도메인 모두에서 일관된 개선 확인
- ImageNet 기반 특징 추출기를 Places365에 그대로 적용해도 효과적 → **도메인 간 일반화 가능성** 입증

### 3.4 λ 파라미터와 일반화의 관계

Figure 4 분석 결과:
- $k$가 작을수록(극단적 few-shot) 최적 $\lambda$가 작아짐 → 노이지 데이터의 기여를 늘림
- $k$가 클수록 최적 $\lambda$가 커짐 → 노이지 예제를 더 강하게 음성으로 처리
- 이는 클린 데이터 수에 따라 **자동으로 조절 가능한 메커니즘**의 필요성을 시사

### 3.5 일반화 향상을 위한 추가 가능성

1. **더 강력한 feature extractor**: Vision Transformer(ViT) 등 더 나은 표현을 사용하면 GCN의 그래프 품질 향상 기대
2. **더 깊은 GCN**: 현재 2-layer GCN을 더 깊게 만들거나, Graph Attention Network(GAT)로 대체 시 관련성 추정 개선 가능
3. **Self-training 루프**: GCN으로 예측한 관련성을 바탕으로 feature extractor를 반복적으로 업데이트하면 일반화 성능 향상 기대

---

## 4. 앞으로의 연구에 미치는 영향 및 연구 시 고려할 점

### 4.1 연구에 미치는 영향

**1) Few-Shot + Noisy Label 결합 연구의 기반 구축**

이 논문은 few-shot learning과 noisy label learning을 결합한 새로운 실험 프로토콜을 정의하고, Low-Shot ImageNet/Places365에 노이지 데이터를 추가하는 확장 벤치마크를 제시했습니다. 이후 연구들이 이 설정을 참조 기준점으로 활용합니다.

**2) GCN의 새로운 활용 방식 제시**

기존 GCN은 주로 semi-supervised 분류(Kipf & Welling, 2017) 또는 few-shot 예측(Garcia & Bruna, 2018)에 사용되었으나, 이 논문은 GCN을 **노이지 데이터 정제기**로 활용하는 새로운 패러다임을 제시했습니다. 이는 데이터 정제 분야에서 그래프 기반 방법론의 확장 가능성을 열었습니다.

**3) Inductive 접근법의 중요성 부각**

Douze et al. [5]의 transductive 방법(추론 시 전체 100M 데이터 필요)과 달리, 이 논문은 **inductive 분류기**를 학습하여 추론 시 소수의 프로토타입만으로 분류가 가능함을 보였습니다. 이는 실용적 배포 관점에서 중요한 기여입니다.

### 4.2 2020년 이후 관련 최신 연구 비교 분석

아래 연구들은 이 논문의 문제 설정 또는 방법론과 직접적으로 관련된 주요 후속 연구들입니다.

> ⚠️ **주의**: 아래 연구들에 대한 세부 내용은 제가 직접 해당 논문 원문을 확인하지 않았으므로, 일반적으로 알려진 내용을 바탕으로 서술하며 부정확할 수 있습니다. 정확한 수치와 방법론은 원문을 참조하시기 바랍니다.

| 연구 | 핵심 방법 | 이 논문과의 관계 |
|------|----------|----------------|
| **TAFSSL** (Lichtenstein et al., ECCV 2020) | 태스크-적응형 특징 부공간 학습 | few-shot에서 특징 표현 개선 |
| **LFOSA** (Tang et al., 2021) | open-set noisy labels + few-shot | 노이지 레이블 범위를 open-set으로 확장 |
| **Meta-Noise** (Fang et al., 2021 방향) | 메타러닝 + 노이지 레이블 | 메타러닝 프레임워크에서 노이지 처리 |
| **Curriculum Learning + Noisy** 계열 | 점진적 노이지 예제 선택 | 정적 관련성 가중치 → 동적 커리큘럼으로 발전 |
| **CLIP/Vision-Language 모델** 계열 | 텍스트-이미지 대응 활용 | 클래스명 기반 노이지 데이터 수집 방식의 발전 |

**주요 발전 방향**:

1. **자기 지도 학습(Self-Supervised Learning) 통합**: DINO, SimCLR 등으로 학습된 강력한 특징 추출기를 활용하면 GCN의 그래프 품질 자체가 향상됩니다.

2. **Vision-Language Model 활용**: CLIP과 같은 모델은 텍스트-이미지 의미 정렬이 강력하여, 클래스명 기반 노이지 데이터 필터링을 더 정밀하게 할 수 있습니다. 이 논문의 텍스트 기반 크롤링 + 시각적 GCN 정제 파이프라인을 CLIP으로 대체/강화할 수 있습니다.

3. **동적 그래프 구조 학습**: 이 논문은 고정된 reciprocal k-NN 그래프를 사용하지만, Graph Attention Network(GAT)나 학습 가능한 그래프 구조를 활용하면 더 유연한 관련성 추정이 가능합니다.

### 4.3 향후 연구 시 고려할 점

**방법론적 고려사항**:

1. **λ 자동 조정 메커니즘**: 현재는 검증셋으로 λ를 수동 튜닝합니다. Meta-learning 방식으로 λ를 자동 적응시키는 방법이 필요합니다.

2. **동적 관련성 업데이트**: 현재는 GCN 학습 후 관련성 점수가 고정되지만, 분류기 학습과 GCN 학습을 **반복적으로(iterative)** 수행하는 self-training 루프가 성능을 높일 수 있습니다.

3. **노이즈 유형 다양화**: 현재는 텍스트 기반 웹 크롤링 노이즈만 고려합니다. 레이블 오류(annotation error), 클래스 혼동(class confusion) 등 다양한 노이즈 유형에 대한 견고성 검증이 필요합니다.

4. **클래스 불균형 처리**: 클래스별 노이지 데이터 수가 0에서 620,142까지 극단적으로 다양합니다. 이에 대한 더 정교한 불균형 처리 전략이 필요합니다.

**실험 설계적 고려사항**:

5. **더 강력한 기준선 비교**: 2020년 이후 등장한 CLIP, DINOv2 기반 few-shot 방법들과의 비교가 필요합니다.

6. **텍스트 이외의 노이지 소스**: 현재는 Flickr(YFCC100M) 텍스트 기반 크롤링이지만, 이미지 검색 엔진이나 소셜 미디어 등 다양한 소스에서의 일반화도 검증되어야 합니다.

7. **계산 효율성**: 클래스별 별도 GCN 학습은 클래스 수가 많아질수록 비용이 증가합니다. 클래스 간 공유 표현을 활용하는 효율적인 구조 탐색이 필요합니다.

---

## 참고 자료

- **본 논문 원문**: Iscen, A., Tolias, G., Avrithis, Y., Chum, O., & Schmid, C. (2020). *Graph convolutional networks for learning with few clean and many noisy labels*. arXiv:1910.00324v3.
- **Kipf & Welling (2017)**: Semi-supervised classification with graph convolutional networks. ICLR 2017.
- **Douze et al. (2018)**: Low-shot learning with large-scale diffusion. CVPR 2018.
- **Gidaris & Komodakis (2018)**: Dynamic few-shot visual learning without forgetting. CVPR 2018.
- **Snell et al. (2017)**: Prototypical networks for few-shot learning. NeurIPS 2017.
- **Hariharan & Girshick (2017)**: Low-shot visual recognition by shrinking and hallucinating features. CVPR 2017.
- **Zhou et al. (2003)**: Learning with local and global consistency (Label Propagation). NeurIPS 2003.
- **Gutmann & Hyvärinen (2010)**: Noise-contrastive estimation. AISTATS 2010.
