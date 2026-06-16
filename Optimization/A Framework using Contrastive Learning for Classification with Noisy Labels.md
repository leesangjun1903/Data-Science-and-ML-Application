# A Framework using Contrastive Learning for Classification with Noisy Labels

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

이 논문의 핵심 주장은 **Contrastive Learning을 사전 학습(pre-training) 단계로 활용하면, 어떤 손실 함수(loss function)를 사용하더라도 노이즈 레이블(noisy label)에 대한 강건성(robustness)을 체계적으로 향상시킬 수 있다**는 것입니다.

### 주요 기여 (Key Contributions)

| 기여 항목 | 설명 |
|---|---|
| **범용 프레임워크** | Contrastive pre-training이 CE, NFL+RCE, ELR 등 어떤 손실 함수에도 적용 가능 |
| **가중치 적용 Supervised Contrastive Loss** | GMM 기반 샘플 가중치($w_i$)를 Supervised Contrastive Loss에 통합 |
| **포괄적 실증 연구** | Pseudo-labeling, GMM sample selection, weighted supervised contrastive learning, mixup with bootstrapping을 결합한 fine-tuning 단계 설계 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

딥러닝 분류 모델은 **노이즈 레이블에 과적합(overfitting)** 되는 문제가 있습니다. 기존의 강건 손실 함수(robust loss functions)들도 노이즈 비율이 높아질수록 급격히 성능이 저하됩니다 (Figure 1 참조: 80% 대칭 노이즈에서 CE는 ~8%, ELR은 ~18%, NFL+RCE는 ~43%의 낮은 정확도 기록). 이를 해결하기 위한 기존 방법들(semi-supervised, noise transition matrix, sample selection 등)은 많은 하이퍼파라미터와 복잡성을 수반합니다.

**핵심 문제 정의:**

$$D = \{(x_i, \overline{y_i})\}_{i=1..n}, \quad x_i \in \mathbb{R}^d, \quad \overline{y_i} \in \{1, \cdots, K\}$$

여기서 $\overline{y_i}$는 노이즈가 포함된 레이블이고, 실제 정답 레이블 $y_i$는 관측 불가능(unobservable)합니다. 목표는 노이즈가 있는 환경에서도 일반화 성능이 높은 DNN $f$의 최적 파라미터 $\theta$를 학습하는 것입니다.

---

### 2.2 제안하는 방법 및 수식

#### Phase (a): Pre-training Phase

**Step a1: Unsupervised Contrastive Learning (SimCLR 기반)**

NCE(Noise Contrastive Estimation) 기반 손실 함수:

$$L_i = -\log \frac{\exp(\mathbf{z}_i^T \mathbf{z}_{j(i)}/\tau)}{\sum_{a \in A(i)} \exp(\mathbf{z}_i^T \mathbf{z}_a/\tau)}, \quad i \in I \tag{7}$$

- $\mathbf{z}$: 특징 벡터(feature vector)
- $\tau$: temperature 파라미터
- $j(i)$: anchor 이미지의 augmented version 인덱스
- $A(i) = I \setminus \{i\}$: 미니배치 내 anchor를 제외한 모든 샘플

**Step a2: Classification (robust loss 적용)**

세 가지 손실 함수를 비교 실험합니다.

**① Cross Entropy (CE) — baseline:**

$$l_{ce} = -\sum_{k=1}^{K} q(k|\mathbf{x}_i) \log(p(k|\mathbf{x}_i)) \tag{1}$$

**② Normalized Focal Loss + Reverse Cross Entropy (NFL+RCE):**

$$l_{nfl} = \frac{-\sum_{k=1}^{K} q(k|\mathbf{x}_i)(1 - p(k|\mathbf{x}_i))^\gamma \log(p(k|\mathbf{x}_i))}{-\sum_{j=1}^{K}\sum_{k=1}^{K} q(y=j|\mathbf{x}_i)(1 - p(k|\mathbf{x}_i))^\gamma \log(p(k|\mathbf{x}_i))} \tag{2}$$

$$l_{rce} = -\sum_{k=1}^{K} p(k|\mathbf{x}_i) \log(q(k|\mathbf{x}_i)) \tag{3}$$

$$l_{nfl+rce} = \alpha \cdot l_{nfl} + \beta \cdot l_{rce} \tag{4}$$

여기서 $\alpha = \beta = 1.0$으로 설정 ($\gamma = 0.5$).

**③ Early Learning Regularization (ELR):**

$$l_{elr} = l_{ce} + \frac{\lambda_{elr}}{N} \log\left(1 - \sum_{k=1}^{K} p(k|\mathbf{x}_i) t(k|\mathbf{x}_i)\right) \tag{5}$$

temporal ensembling을 통한 타겟 업데이트:

$$t(k|\mathbf{x}_i)^{(l)} = \beta \cdot t(k|\mathbf{x}_i)^{(l-1)} + (1-\beta) \cdot p(k|\mathbf{x}_i)^{(l)} \tag{6}$$

---

#### Phase (b): Fine-tuning Phase

**Step b1: GMM 기반 샘플 가중치 계산**

학습 손실 분포에 2성분 GMM을 피팅하여 각 샘플의 정확도 확률을 가중치로 사용:

$$w_i = p(k=0 \mid l_i) \tag{8}$$

- $l_i$: 샘플 $i$의 손실값
- $k=0$: 낮은 손실을 가진 "깨끗한(clean)" 성분

**Step b2: Weighted Supervised Contrastive Learning**

레이블 정보를 활용한 Supervised Contrastive Loss:

$$L_i = -\log \frac{1}{|P(i)|} \sum_{p \in P(i)} \frac{\exp(\mathbf{z}_i^T \mathbf{z}_p / \tau)}{\sum_{a \in A(i)} \exp(\mathbf{z}_i^T \mathbf{z}_a / \tau)} \tag{9}$$

GMM 가중치를 적용한 확장 버전 (본 논문의 핵심 기여):

$$L_i = -\log \frac{1}{|P(i)|} \sum_{p \in P(i)} \frac{\widetilde{w_{p,i}} \exp(\mathbf{z}_i^T \mathbf{z}_p / \tau)}{\sum_{a \in A(i)} \exp(\mathbf{z}_i^T \mathbf{z}_p / \tau)} \tag{10}$$

여기서 $\widetilde{w_{p,i}} = 1$ if $p = j(i)$, else $\widetilde{w_{p,i}} = w_i$. 모든 샘플이 노이즈로 간주되면 식 (10)은 식 (7)의 비지도 contrastive loss로 단순화됩니다.

---

### 2.3 모델 구조

```
[Input: Noisy Dataset (x, ȳ)]
        ↓
┌─────────────────────────────────────┐
│  (a) Pre-training Phase              │
│  ┌──────────────────────────────┐   │
│  │ (a1) SimCLR Unsupervised     │   │ → Contrastive Loss (식 7)
│  │      Contrastive Learning    │   │
│  │  - ResNet18 Encoder          │   │
│  │  - Projection Head (MLP)     │   │
│  │  - Data Augmentation         │   │
│  └──────────────────────────────┘   │
│             ↓                        │
│  ┌──────────────────────────────┐   │
│  │ (a2) Classification          │   │ → CE / NFL+RCE / ELR (식 1~6)
│  │  - Warm-up (frozen encoder)  │   │
│  │  - Full model training       │   │
│  └──────────────────────────────┘   │
│             ↓ Pseudo-labels (ŷ)     │
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│  (b) Fine-tuning Phase               │
│  ┌──────────────────────────────┐   │
│  │ (b1) GMM Sample Selection    │   │ → w_i = p(k=0|l_i) (식 8)
│  └──────────────────────────────┘   │
│             ↓ weights (w)            │
│  ┌──────────────────────────────┐   │
│  │ (b2) Weighted Supervised     │   │ → Weighted SupCon Loss (식 10)
│  │      Contrastive Learning    │   │
│  └──────────────────────────────┘   │
│             ↓                        │
│  ┌──────────────────────────────┐   │
│  │ (b3) Final Classification    │   │ → ŷ_final
│  └──────────────────────────────┘   │
└─────────────────────────────────────┘
```

**인코더**: ResNet18 (ImageNet 사전 학습 없이)
**Projection Head**: MLP (1 hidden layer + ReLU)
**분류기**: MLP (1 hidden layer + Batch Norm + ReLU)
**옵티마이저**: Adam (표현 학습), SGD + cosine annealing (분류)

---

### 2.4 성능 향상

#### 주요 결과 (Table 1 기준)

| Dataset | Noise Type | Noise Rate | Loss | Base | Pre-training |
|---|---|---|---|---|---|
| CIFAR10 | Sym | 80% | ELR | 18.3 | **84.8** |
| CIFAR10 | Sym | 80% | NFL+RCE | 42.8 | **59.9** |
| CIFAR100 | Sym | 80% | ELR | 16.2 | **45.3** |
| CIFAR100 | Sym | 60% | NFL+RCE | 47.0 | **61.8** |
| CIFAR100 | Asym | 40% | ELR | 67.6 | **67.6** |

**Real-world 데이터셋 결과 (Table 3):**

| Dataset | Loss | Base | Pre-t | Fine-tune |
|---|---|---|---|---|
| Webvision | CE | 51.8 | 57.1 | **58.4** |
| Webvision | ELR | 53.0 | 58.1 | **59.0** |
| Clothing1M | CE | 54.8 | 59.1 | **61.5** |

**Fine-tuning 단계**는 CIFAR10의 66%, CIFAR100의 80% 이상의 케이스에서 추가 성능 향상을 보였으며, 비대칭 노이즈에서 더 큰 이득을 나타냈습니다.

---

### 2.5 한계점

1. **계산 비용**: Pre-training 단계만으로도 baseline 대비 **약 2.4배** 연산 시간 증가, 전체 프레임워크는 **3~4.5배** 증가
2. **하이퍼파라미터 민감도**: 특히 80% 대칭 노이즈에서 학습률에 따라 성능이 크게 달라짐 (CE와 NFL+RCE가 반대 경향을 보임)
3. **조기 과적합 감지 어려움**: 클린 검증 세트 없이 memorization phase의 시작점을 감지하기 어려움 — TSP(Training Stop Point)나 CKA 지표가 완벽한 해결책을 제공하지 못함
4. **이미지 해상도 제한**: 계산 자원 제약으로 128×128로 이미지를 축소하여 실험, 기존 논문들과 직접 비교가 어려움
5. **비대칭 노이즈에서의 불안정성**: ELR의 경우 비대칭 노이즈에서 하이퍼파라미터 변화에 민감하게 반응
6. **이론적 이해 부족**: Contrastive pre-training과 noisy label 분류기 간 상호작용의 이론적 분석 미흡

---

## 3. 모델의 일반화 성능 향상 가능성

이 논문에서 일반화 성능 향상과 직접적으로 연관된 메커니즘은 다음과 같습니다.

### 3.1 Contrastive Pre-training의 일반화 효과

Contrastive learning은 **레이블 정보 없이** 입력 데이터의 증강(augmented views) 쌍을 비교하여 표현을 학습합니다. 이 과정에서 학습된 표현은 노이즈 레이블의 영향을 받지 않으므로, 분류기의 초기 파라미터 공간이 **노이즈로부터 분리된(decoupled) 의미론적 구조**를 가지게 됩니다.

CKA(Centered Kernel Alignment) 분석 결과에서 이를 확인할 수 있습니다:

> "It is interesting to note that the first layer of the pre-trained model remains very similar to the same layer computed by contrastive learning. Such behavior was expected in order to improve the robustness against noisy labels."

이는 contrastive pre-training이 인코더의 초기 레이어를 **안정적인 표현 공간**에 고정시켜, 이후 분류 학습 시 노이즈 레이블에 의한 표현 왜곡을 최소화함을 의미합니다.

### 3.2 Pseudo-label을 통한 일반화 향상

Figure 3에서 보여주듯, contrastive pre-training 후 생성된 **pseudo-label의 정확도가 노이즈가 포함된 원래 레이블보다 항상 높습니다**. 특히 노이즈 비율이 높을수록 이 이득이 더 커집니다.

$$\text{Pseudo-label Acc} > \text{Ground Truth Acc (noisy)} \quad \forall \text{ noise ratio}$$

이는 모델이 레이블 노이즈로부터 독립적인 데이터의 **내재적 구조(intrinsic structure)**를 학습했음을 시사하며, 이 구조 기반의 예측이 더 일반화된 레이블 정보를 제공합니다.

### 3.3 GMM 기반 Sample Weighting의 일반화 효과

$$w_i = p(k=0 \mid l_i) \tag{8}$$

손실값 분포에서 낮은 손실을 가진 샘플에 높은 가중치를 부여함으로써, **모델이 신뢰할 수 있는 샘플에 집중적으로 학습**할 수 있습니다. Figure 4에서 GMM으로 선택된 클린 서브셋의 정확도가 0.6에서 0.93까지 향상됨을 보였습니다.

### 3.4 Warm-up 전략의 일반화 기여

분류기 학습 초기에 인코더를 동결(freeze)하고 분류 헤드만 학습하는 warm-up 전략은, contrastive learning으로 학습된 **표현 공간이 초기 과적합에 의해 손상되는 것을 방지**합니다. 이는 특히 높은 노이즈 비율의 대칭 노이즈에서 효과적이었습니다.

### 3.5 일반화 성능 향상의 이론적 배경

Contrastive pre-training을 통한 일반화 향상은 다음 세 가지 관점에서 설명됩니다:

1. **Memorization Effect 억제**: Deep network는 초기에 클린 샘플을 먼저 학습한 후 노이즈 샘플을 암기(memorize)합니다. Contrastive pre-training으로 학습된 표현은 레이블 독립적이므로, 이후 supervised fine-tuning 시 클린 패턴을 더 오래 유지합니다.

2. **표현 공간의 분리성(Separability)**: Contrastive learning은 서로 다른 클래스의 표현을 자연스럽게 분리하는 경향이 있어, 이후 분류기 학습 시 noisy label에 의한 결정 경계 왜곡을 줄입니다.

3. **전이 학습 효과**: 비지도 contrastive learning은 ImageNet 등 대규모 사전 학습 없이도 준수한 표현을 학습할 수 있어, 제한된 계산 자원에서도 일반화 성능을 향상시킵니다.

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

아래는 논문에서 직접 언급된 방법들과 이 논문의 방법론을 비교한 표입니다.

| 방법 | 핵심 전략 | CIFAR10 80%S | CIFAR100 80%S | 복잡도 |
|---|---|---|---|---|
| **Ours (ELR + Pre-t)** | Contrastive Pre-training + ELR | **84.8** | **45.3** | 중간 |
| DivideMix [Li et al., 2020] | Semi-supervised + MixMatch + 2 networks | **92.9** | **59.6** | 매우 높음 |
| ELR [Liu et al., 2020] | Early Learning Regularization | 73.9 | 29.7 | 낮음 |
| SELF [Nguyen et al., 2019] | Self-ensembling filtering | 69.9 | 42.1 | 중간 |
| JoCoR [Wei et al., 2020] | Co-regularization | 25.5 | 12.9 | 중간 |
| Co-teaching+ [Yu et al., 2019] | Disagreement-based selection | 23.5 | 14.0 | 중간 |
| Taks [Song et al., 2020] | No-regret sample selection | 40.2 | 16.0 | 낮음 |

**중요 참고사항**: DivideMix의 경우 Ortego et al. [2020]의 재현 실험에서 CIFAR100 대칭 노이즈 80%에서 59.6% 대신 **49.5%**, 비대칭 노이즈 40%에서 72.1% 대신 **50.9%**로 유의미하게 낮게 보고되었습니다. 이는 DivideMix의 재현 안정성 문제를 시사합니다.

또한 이 논문에서 Webvision 데이터셋에서 ResNet50 + 224×224 해상도를 사용했을 때, CE는 **75.7%**, ELR은 **76.2%**를 달성하였는데, 이는 DivideMix(77.3%)와 ELR+(77.8%)에 근접한 수치로서 더 작은 모델(Inception-ResNet-v2 대비 ResNet50)을 사용했다는 점에서 경쟁력 있는 결과입니다.

---

## 5. 앞으로의 연구에 미치는 영향 및 고려 사항

### 5.1 이 논문이 미치는 영향

**① "Pre-training as a universal robustness booster" 패러다임 확립**

이 논문은 contrastive pre-training이 특정 방법론에 종속되지 않고 **어떤 손실 함수에도 범용적으로 적용 가능한 robustness 향상 도구**임을 실증했습니다. 이는 이후 연구에서 self-supervised pre-training을 noisy label learning의 표준 구성 요소로 채택하는 방향에 영향을 줍니다.

**② Self-supervised + Noisy Label Learning의 교차 연구 활성화**

SimCLR, MoCo 등 self-supervised learning 방법론을 noisy label 설정에 적용한 초기 체계적 연구로, 이후 BYOL, SimSiam, DINO 등 더 발전된 self-supervised 방법론과 noisy label learning의 결합 연구를 촉진했습니다.

**③ 실용적 관점에서의 기여**

단일 GPU(8GB RAM)에서 ResNet18로 수행 가능한 프레임워크를 제시함으로써, **제한된 자원에서도 noisy label 문제에 대응할 수 있는 현실적인 방법론**을 제공했습니다.

---

### 5.2 향후 연구에서 고려할 사항

**① 더 발전된 Self-supervised Learning 방법론 통합**

이 논문은 SimCLR과 MoCo만 비교했지만, 이후 등장한 **BYOL, SimSiam, DINO, MAE** 등의 방법론이 더 나은 표현을 학습할 수 있는지 탐색이 필요합니다. 특히 negative pair가 필요 없는 방법들(BYOL, SimSiam)은 메모리 효율성 측면에서 이점이 있습니다.

**② 조기 과적합 감지(Early Stopping) 문제 해결**

논문에서 명확히 한계로 인정한 이 문제는 매우 중요합니다. 클린 검증 세트 없이 memorization phase를 감지하기 위해 다음을 고려할 수 있습니다:
- **표현 공간의 변화율(CKA 기반)**: CKA가 과적합과 명확한 상관관계를 보이지 못했지만, 더 세밀한 레이어별 분석이 가능합니다
- **예측 안정성 기반 조기 종료**: 연속 에포크 간 클래스 변경 수의 급격한 감소를 신호로 사용
- **손실 분포 변화 감지**: GMM 성분의 분리도(separation)가 memorization phase와 연관될 수 있음

**③ 이론적 분석 강화**

Contrastive pre-training이 왜 noisy label에 대한 robustness를 향상시키는지에 대한 이론적 설명이 부족합니다. 다음 방향의 연구가 필요합니다:
- PAC-learning 또는 정보 이론적 관점에서의 분석
- 표현 공간의 클래스 분리도(class separability)와 노이즈 robustness 간의 이론적 연결

**④ 노이즈 유형 다양화**

이 논문은 대칭(symmetric)과 비대칭(asymmetric) 노이즈만 다루었지만, 실제 환경에서는 **인스턴스 의존적 노이즈(instance-dependent noise)**, **특징 의존적 노이즈(feature-dependent noise)** 등 더 복잡한 노이즈 패턴이 존재합니다.

**⑤ 더 큰 모델 및 고해상도 실험**

논문 자체에서도 인정했듯이, ResNet50 + 224×224 + 전체 Clothing1M 데이터셋으로의 확장 실험이 필요합니다. Vision Transformer(ViT) 기반 인코더와의 결합도 연구할 가치가 있습니다.

**⑥ Multi-modal 노이즈 레이블 학습**

최근 CLIP과 같은 비전-언어 모델이 등장함에 따라, 텍스트 정보를 활용한 contrastive pre-training이 이미지 분류의 noisy label 문제를 더 효과적으로 해결할 수 있는지 탐색이 필요합니다.

**⑦ 하이퍼파라미터 자동화**

논문에서 지적했듯이 클린 검증 세트 없이 최적 하이퍼파라미터를 결정하기 어렵습니다. **AutoML, Bayesian optimization, 또는 meta-learning** 기법을 활용하여 noisy label 환경에서의 하이퍼파라미터 탐색을 자동화하는 연구가 필요합니다.

---

## 참고자료

- **주요 논문 (분석 대상)**:
  - Ciortan, M., Dupuis, R., & Peel, T. (2021). *A Framework using Contrastive Learning for Classification with Noisy Labels*. arXiv:2104.09563v1

- **논문 내 주요 인용 문헌**:
  - Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). *A simple framework for contrastive learning of visual representations*. arXiv:2002.05709 [SimCLR]
  - He, K., Fan, H., Wu, Y., Xie, S., & Girshick, R. (2020). *Momentum contrast for unsupervised visual representation learning*. CVPR [MoCo]
  - Li, J., Socher, R., & Hoi, S. C. H. (2020). *DivideMix: Learning with noisy labels as semi-supervised learning*. ICLR
  - Liu, S., Niles-Weed, J., Razavian, N., & Fernandez-Granda, C. (2020). *Early-learning regularization prevents memorization of noisy labels*. NeurIPS [ELR]
  - Ma, X., Huang, H., Wang, Y., Romano, S., Erfani, S., & Bailey, J. (2020). *Normalized loss functions for deep learning with noisy labels*. ICML [NFL+RCE]
  - Khosla, P., et al. (2020). *Supervised contrastive learning*. NeurIPS
  - Wang, Y., Ma, X., Chen, Z., Luo, Y., Yi, J., & Bailey, J. (2019). *Symmetric cross entropy for robust learning with noisy labels*. ICCV
  - Ortego, D., Arazo, E., Albert, P., O'Connor, N. E., & McGuinness, K. (2020). *Multi-objective interpolation training for robustness to label noise*. arXiv
  - Song, H., Kim, M., Park, D., & Lee, J. G. (2020). *Learning from noisy labels with deep neural networks: A survey*. arXiv:2007.08199
  - Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019). *Similarity of neural network representations revisited*. ICML [CKA]
