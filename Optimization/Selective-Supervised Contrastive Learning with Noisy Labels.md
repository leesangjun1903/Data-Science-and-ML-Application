# Selective-Supervised Contrastive Learning with Noisy Labels (Sel-CL) 

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

기존 Supervised Contrastive Learning (Sup-CL)은 표현 학습에서 강력한 성능을 보이지만, **노이즈 레이블이 존재할 때 pair-wise 방식으로 구성된 noisy pairs가 표현 학습을 직접적으로 왜곡**한다. Sel-CL은 이 직접적 원인을 해결하기 위해, **노이즈 비율(noise rate)을 사전에 알지 못해도** 신뢰할 수 있는 쌍(confident pairs)을 선별하여 Sup-CL을 수행한다.

### 주요 기여 3가지

| 기여 | 내용 |
|------|------|
| ① 새로운 프레임워크 제안 | 노이즈 레이블 환경에서 selective-supervised contrastive learning (Sel-CL) 제안 |
| ② 노이즈 비율 불필요 | 노이즈 비율 추정 없이 confident examples → confident pairs를 점진적으로 선별하는 positive cycle 구성 |
| ③ 실험적 검증 | 합성·실세계 노이즈 데이터셋(CIFAR-10/100, WebVision-50)에서 SOTA 성능 달성 |

---

## 2. 해결하고자 하는 문제, 제안 방법(수식 포함), 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

딥러닝은 대규모 고품질 레이블에 의존하지만, 실제 데이터는 웹 크롤링 등으로 수집된 **노이즈 레이블**을 포함한다. 이로 인해:

- 노이즈 레이블 → 잘못된 학습 신호 → **표현 공간 왜곡(corrupted representations)**
- Sup-CL은 pair-wise 방식으로 작동하므로, noisy pairs가 직접 표현 학습을 손상시킴
- 기존 방법(MOIT+, MoPro 등)은 Sup-CL의 pair-wise 특성을 충분히 고려하지 않아 **sub-optimal**

$$\text{noisy labels} \rightarrow \text{noisy pairs} \rightarrow \text{corrupted representations} \rightarrow \text{poor generalization}$$

### 2.2 제안 방법 (수식 포함)

#### 전체 파이프라인 개요

$$\mathcal{D}_e = \{(x_i, \tilde{y}_i)\}_{i=1}^{n} \xrightarrow{\text{warm-up}} \text{low-dim } z_i \xrightarrow{\text{Step 1}} \mathcal{T} \xrightarrow{\text{Step 2}} \mathcal{G} \xrightarrow{\text{Step 3}} \text{Sup-CL}$$

#### Step 1: Confident Examples 선별

두 저차원 표현 $z_i$, $z_j$ 사이의 **코사인 유사도**:

$$d(z_i, z_j) = \frac{z_i z_j^\top}{\|z_i\| \|z_j\|}  \tag{1}$$

Top- $K$ 이웃의 레이블로 **pseudo-label** $\hat{y}_i$ 생성 후 사후확률 추정:

$$\hat{q}_c(x_i) = \frac{1}{K} \sum_{\substack{k=1 \\ x_k \in \mathcal{N}_i}}^{K} \mathbb{I}[\hat{y}_k = c], \quad c \in [C] \tag{2}$$

Cross-entropy 손실 기반으로 $c$번째 클래스의 confident example set $\mathcal{T}_c$:

$$\mathcal{T}_c = \{(x_i, \tilde{y}_i) \mid \ell(\hat{q}(x_i), \tilde{y}_i) < \gamma_c,\; i \in [n]\}, \quad c \in [C] \tag{3}$$

- $\gamma_c$: 클래스 균형을 위해 $\alpha$ 분위수로 동적 결정
- 최종: $\mathcal{T} = \bigcup_{c=1}^{C} \mathcal{T}_c$

#### Step 2: Confident Pairs 선별

**Step 2a** — $\mathcal{T}$에서 직접 구성하는 confident pairs $\mathcal{G}'$:

$$\mathcal{G}' = \{P_{ij} \mid \tilde{y}_i = \tilde{y}_j,\; (x_i, \tilde{y}_i),\, (x_j, \tilde{y}_j) \in \mathcal{T}\} \tag{4}$$

**Step 2b** — 표현 유사도 기반으로 noisy positive pairs에서 추가 선별 $\mathcal{G}''$:

$$\mathcal{G}'' = \{P_{ij} \mid \tilde{s}_{ij} = 1,\; d(z_i, z_j) > \gamma\} \tag{5}$$

- $\tilde{s}_{ij} = \mathbb{I}[\tilde{y}_i = \tilde{y}_j]$: similarity label
- $\gamma$: $\mathcal{G}'$의 표현 유사도 $\beta$ 분위수로 동적 결정

최종 confident pair set: $\mathcal{G} = \mathcal{G}' \cup \mathcal{G}''$

#### Step 3: 선별된 Pairs로 표현 학습

**Supervised Contrastive Loss** (선별된 confident pairs 기반):

$$\mathcal{L} = \sum_{i \in I} \mathcal{L}_i(z_i) = \sum_{i \in I} \frac{-1}{|\mathcal{G}(i)|} \sum_{g \in \mathcal{G}(i)} \log \frac{\exp(z_i \cdot z_g / \tau)}{\sum_{a \in A(i)} \exp(z_i \cdot z_a / \tau)} \tag{6}$$

- $A(i) = I \setminus \{i\}$: anchor를 제외한 인덱스 집합
- $\mathcal{G}(i)$: $i$에 대한 선별된 confident positive indices
- $\tau$: temperature parameter

**Mixup 정규화** (표현 학습 강건성 강화):

$$x_i = \lambda x_a + (1-\lambda) x_b, \quad \lambda \sim \text{Beta}(\alpha_m, \alpha_m) \tag{}$$

$$\mathcal{L}_i^{\text{MIX}}(z_i) = \lambda \mathcal{L}_a(z_i) + (1-\lambda)\mathcal{L}_b(z_i) \tag{7}$$

**Classification Loss** (confident examples 기반):

$$\mathcal{L}^{\text{CLS}} = \sum_{(x_i, \tilde{y}_i) \in \mathcal{T}} \ell(\hat{p}(x_i), \tilde{y}_i) \tag{8}$$

**Similarity Loss** (유사도 레이블 직접 학습):

$$\mathcal{L}^{\text{SIM}} = \sum_{i \in I} \sum_{j \in A(i)} \ell(\hat{p}(x_i)\hat{p}(x_j),\; \mathbb{I}[P_{i'j'} \in \mathcal{G}]) \tag{9}$$

**총 손실 함수**:

$$\mathcal{L}^{\text{ALL}} = \mathcal{L}^{\text{MIX}} + \lambda_c \mathcal{L}^{\text{CLS}} + \lambda_s \mathcal{L}^{\text{SIM}} \tag{10}$$

- $\lambda_c = 1$, $\lambda_s = 0.01$ (모든 실험에서 고정)

### 2.3 모델 구조

```
입력 (Noisy Labels)
        ↓
[Backbone Encoder f] → 고차원 표현 v_i
        ↓                     ↓
[Projection Head]      [Classifier Head]
저차원 표현 z_i         예측 p̂(x_i)
        ↓
  Confident Examples T 선별 (코사인 유사도 + Cross-Entropy)
        ↓
  Confident Pairs G 선별 (G' ∪ G'')
        ↓
  Supervised Contrastive Learning + Mixup
        ↓
Pre-trained Encoder f (Fine-tuning 단계로 이전)
        ↓
[새로운 Classifier Head 추가]
Fine-tuning (Sel-CL+)
```

**두 단계 학습 방식:**
- **Stage 1 (Pre-training)**: Sel-CL로 robust representation 학습
- **Stage 2 (Fine-tuning)**: Pre-trained encoder $f$ + 새 classifier head → Sel-CL+

### 2.4 성능 향상

#### CIFAR-10/100 (표현 학습 품질, Weighted KNN %)

| 방법 | Clean | Sym 20% | Sym 80% | Asym 10% | Asym 40% |
|------|-------|---------|---------|----------|----------|
| Sup-CL | 72.66 | 58.32 | 41.00 | 71.11 | 68.00 |
| MOIT | 77.48 | 67.42 | 55.58 | 74.86 | 72.60 |
| **Sel-CL** | **77.94** | **75.36** | **62.49** | **76.77** | **72.71** |

- 80% symmetric noise에서 MOIT(55.58%) < Uns-CL(56.23%) < **Sel-CL(62.49%)**

#### CIFAR-10 Test Accuracy (%)

| 방법 | Sym 20% | Sym 90% | Asym 40% |
|------|---------|---------|----------|
| DivideMix | 95.0 | 74.2 | 91.4 |
| MOIT+ | 94.1 | 74.7 | 93.3 |
| **Sel-CL+** | **95.5** | **81.9** | **93.4** |

#### WebVision-50 (실세계 노이즈)

| 방법 | WebVision top-1 | ILSVRC12 top-1 |
|------|-----------------|----------------|
| NGC | 79.16 | 74.44 |
| **Sel-CL+** | **79.96** | **76.84** |

### 2.5 한계점

1. **컴퓨팅 자원**: Contrastive Learning 특성상 **대용량 batch size 또는 memory bank** 필요 → 메모리 부담 증가
2. **KNN 연산 비용**: K-Nearest Neighbor 알고리즘으로 인한 추가 계산 비용 발생 (대규모 데이터셋 적용 시 병목)
3. **비대칭 노이즈 고노이즈 취약**: 90% symmetric noise 환경에서 ProtoMix, NGC 대비 성능 열세 (Sel-CL+ 67.5% vs NGC 80.5%)
4. **하이퍼파라미터 $\alpha$, $\beta$의 데이터셋 의존성**: 분위수 설정이 데이터셋마다 달라 자동화 어려움

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화 향상의 핵심 메커니즘

**① Positive Cycle (긍정적 순환 학습)**

$$\text{Better Confident Pairs} \rightarrow \text{Better Representations} \rightarrow \text{Better Confident Pairs} \rightarrow \cdots$$

각 epoch마다 선별 품질과 표현 품질이 상호 강화되는 구조로, Fig. 3(c)에서 label precision이 학습 초기 ~84%에서 ~97%까지 점진적으로 향상됨을 확인.

**② Similarity Label 활용의 일반화 기여**

$\mathcal{G}''$의 핵심 통찰: 클래스 레이블이 틀렸더라도 **두 샘플이 같은 클래스로 오분류된 경우 similarity label은 올바름**.

$$\mathcal{G}'' = \{P_{ij} \mid \tilde{s}_{ij} = 1,\; d(z_i, z_j) > \gamma\}$$

이는 특히 **비대칭 노이즈(asymmetric noise)**에서 효과적으로, 의미적으로 유사한 클래스 간 혼동이 일어나도 표현 공간에서 같은 군집을 형성하는 쌍을 활용 가능.

**③ Mixup 정규화의 역할**

$$\mathcal{L}_i^{\text{MIX}}(z_i) = \lambda \mathcal{L}_a(z_i) + (1-\lambda)\mathcal{L}_b(z_i)$$

Mixup은 표현 공간의 interpolation을 강제하여 **결정 경계를 부드럽게** 만들고, 노이즈에 의한 날카로운 경계 왜곡을 방지.

**④ 클래스 균형 선별의 일반화 효과**

$\alpha$ 분위수 기반 class-balanced selection으로 특정 클래스로의 편향(bias) 없이 각 클래스별로 균등한 수의 confident examples를 선별 → 균형 잡힌 표현 학습 가능.

**⑤ 2단계 학습의 일반화 이점**

- Stage 1: 노이즈에 강건한 **범용 표현** 학습
- Stage 2: 정제된 표현 위에서 **새로운 분류기** fine-tuning

이 구조는 pre-trained representation의 일반화 능력이 downstream task에 효과적으로 전이되도록 설계됨. Table 8에서 Sel-CL init.이 Uns-CL init.보다 DivideMix, ELR+ 모두에서 성능 향상을 가져옴.

**⑥ Weighted KNN 평가를 통한 일반화 측정**

Linear probe 없이 K=200인 weighted KNN으로 표현 품질 직접 측정 → 학습된 표현이 특정 분류기에 과적합되지 않고 **범용적으로 구조화**되었음을 확인.

### 3.2 일반화 한계 및 고려사항

- **도메인 특화 한계**: CIFAR, WebVision 등 이미지 분류에 집중되어 object detection, NLP 등 타 도메인 일반화는 미검증
- **극단적 노이즈(90%)**: 일반화에 필요한 신뢰 가능한 쌍의 수가 너무 적어져 학습 불안정

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

### 4.1 주요 경쟁 방법 체계적 비교

| 방법 | 연도 | 핵심 아이디어 | 노이즈 처리 전략 | Sel-CL과의 차이 |
|------|------|--------------|----------------|----------------|
| **SimCLR** [Chen et al., ICML 2020] | 2020 | Instance-level unsupervised CL | ✗ 노이즈 무관 | Label 정보 미활용 |
| **Sup-CL** [Khosla et al., NeurIPS 2020] | 2020 | 클래스 기반 supervised CL | ✗ 노이즈에 취약 | 모든 pairs 사용 → noisy pairs 문제 |
| **DivideMix** [Li et al., ICLR 2020] | 2020 | GMM으로 clean/noisy 분리 + 반지도 학습 | GMM 기반 분리 | CL 미활용, 표현 학습 부재 |
| **ELR/ELR+** [Liu et al., NeurIPS 2020] | 2020 | Early-learning regularization | 정규화 기반 | Pair-wise 특성 미고려 |
| **MOIT+** [Ortego et al., CVPR 2021] | 2021 | Sup-CL + Mixup + 반지도 | 정규화 추가 | Point-wise 선별, 모든 noisy pairs에 CL 적용 |
| **MoPro** [Li et al., ICLR 2021] | 2021 | 모멘텀 프로토타입 기반 pseudo-label + CL | Prototype 정제 | Noise rate 추정 필요 경향 |
| **ProtoMix** [Li et al., ICCV 2021] | 2021 | Prototype Mixup + CL | 1-stage, pseudo-label | Asymmetric noise에서 Sel-CL보다 열세 |
| **NGC** [Wu et al., ICCV 2021] | 2021 | 그래프 기반 noisy label 통합 처리 | Open-world noise | High sym. noise에서 강점 |
| **C2D** [Zheltonozhskii et al., WACV 2022] | 2022 | Uns-CL pre-training → 기존 방법 초기화 | 2-stage | Sup-CL 미활용, pair 선별 없음 |
| **Sel-CL** [Li et al., arXiv 2022] | 2022 | **Pair-wise 선별 + Sup-CL** | **Confident pair 선별** | **노이즈 비율 불필요, pair-wise 특성 직접 활용** |

### 4.2 방법론적 관점 비교

```
노이즈 처리 방식 분류:
┌─────────────────────────────────────────────────────────┐
│ 1. 손실 함수 수정: GCE, Forward, ELR                     │
│ 2. 데이터 재가중: MentorNet, L2R                         │
│ 3. 샘플 선별: Co-teaching, DivideMix, Sel-CL            │
│ 4. 표현 학습 + CL: MOIT+, MoPro, NGC, Sel-CL            │
└─────────────────────────────────────────────────────────┘
```

**Sel-CL의 차별점**: 기존 방법들이 example-level 선별에 집중할 때, **pair-level 선별**로 Sup-CL의 pair-wise 특성을 직접 활용.

### 4.3 성능 수치 비교 (CIFAR-100, Asymmetric 40%)

| 방법 | Test Accuracy (%) |
|------|------------------|
| DivideMix | 51.0 |
| ELR | 70.0 |
| MOIT+ | 74.0 |
| ProtoMix | 48.8 |
| NGC | — |
| **Sel-CL+** | **74.2** |

비대칭 노이즈에서 Sel-CL+가 SOTA 달성 ($\mathcal{G}''$의 similarity-based pair 선별 덕분).

---

## 5. 앞으로의 연구에 미치는 영향 및 고려할 점

### 5.1 연구에 미치는 영향

**① Pair-level 노이즈 처리 패러다임 확립**

기존의 example-level 접근에서 **pair-level 접근**으로 관점을 전환. 이는 contrastive learning 기반 노이즈 학습 연구의 새로운 방향성을 제시하며, 향후 triplet, multi-view 등 다양한 pair 구성 방식 연구를 촉진할 것으로 예상.

**② Noise Rate-Free 방법론의 실용화**

노이즈 비율 사전 지식 없이 동적 분위수 기반으로 threshold를 결정하는 방식은 **실세계 적용 가능성**을 크게 높임. 향후 자동화된 threshold 결정 방법론 연구에 기여.

**③ Positive Cycle 학습 프레임워크**

선별과 표현 학습의 상호 강화 구조는 curriculum learning, self-training 등 다양한 분야에서 참고할 수 있는 **일반적 설계 원칙** 제공.

**④ Downstream Task 확장 가능성**

논문에서 미래 연구로 object detection, text matching 언급. Robust pre-trained representation의 transfer learning 관점에서 **foundation model fine-tuning** 연구에도 영향 가능.

### 5.2 앞으로 연구 시 고려할 점

**① 계산 효율성 개선**
- KNN 연산의 $O(n^2)$ 복잡도 → Approximate Nearest Neighbor (FAISS 등) 적극 활용
- Memory bank 크기 최적화 연구 필요
- 대규모 데이터셋(ImageNet-full 등)으로의 확장성 검증 필요

**② 극단적 노이즈 환경 강건성**

$$\text{90\% noise rate} \rightarrow |\mathcal{G}| \approx 0 \Rightarrow \text{학습 불안정}$$

극단적 노이즈에서도 유효한 confident pairs를 확보할 수 있는 **보완 메커니즘** 연구 필요 (예: 증강 기반 다양성 확보, prototype 보완 등).

**③ 비대칭/인스턴스 의존 노이즈 대응**

실세계 노이즈는 종종 **instance-dependent**한 특성을 가짐. Sel-CL의 pair 선별이 이러한 구조적 노이즈에 대응할 수 있는지 이론적 분석 필요.

$$\text{transition matrix: } T_{ij}(x) = P(\tilde{y}=j \mid y=i, x) \neq P(\tilde{y}=j \mid y=i)$$

**④ 멀티모달 및 타 도메인 적용**

- Vision-Language 모델 (CLIP 등) 환경에서 noisy pair 선별 전략 적용
- NLP, 의료영상, 시계열 등 비이미지 도메인으로의 확장
- Self-supervised foundation model의 noisy fine-tuning 시 Sel-CL 원리 적용 가능성

**⑤ 이론적 분석 강화**

현재 논문은 경험적 검증 중심. 향후:
- Confident pair 선별의 **수렴 보장(convergence guarantee)**
- 선별된 pairs의 **노이즈 감소율 이론적 bound**
- Generalization error와 pair 품질 간의 **이론적 관계** 규명

**⑥ 준지도·능동학습과의 결합**

$$\text{Sel-CL} + \text{Semi-supervised Learning} \rightarrow \text{labeled + unlabeled + noisy data 통합 처리}$$

소량의 clean labels + 대량의 noisy labels 시나리오에서 능동학습(active learning)으로 clean annotation을 효율적으로 활용하는 연구 방향.

**⑦ 공정성(Fairness) 관점**

클래스 균형 선별이 소수 클래스에 불리하게 작용할 수 있음 → long-tail 분포 + noisy label 동시 처리 연구 필요.

---

## 참고 자료

**본 답변에서 직접 참고한 문헌:**

1. **Li, S., Xia, X., Ge, S., & Liu, T. (2022).** "Selective-Supervised Contrastive Learning with Noisy Labels." *arXiv:2203.04181v1* [cs.CV]. GitHub: https://github.com/ShikunLi/Sel-CL

2. **Khosla, P., et al. (2020).** "Supervised Contrastive Learning." *NeurIPS 2020.*

3. **Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020).** "A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)." *ICML 2020.*

4. **Li, J., Socher, R., & Hoi, S. C. H. (2020).** "DivideMix: Learning with Noisy Labels as Semi-supervised Learning." *ICLR 2020.*

5. **Liu, S., Niles-Weed, J., Razavian, N., & Fernandez-Granda, C. (2020).** "Early-Learning Regularization Prevents Memorization of Noisy Labels (ELR)." *NeurIPS 2020.*

6. **Ortego, D., et al. (2021).** "Multi-Objective Interpolation Training for Robustness to Label Noise (MOIT)." *CVPR 2021.*

7. **Li, J., Xiong, C., & Hoi, S. (2021).** "Learning from Noisy Data with Robust Representation Learning (ProtoMix)." *ICCV 2021.*

8. **Wu, Z.-F., et al. (2021).** "NGC: A Unified Framework for Learning with Open-World Noisy Data." *ICCV 2021.*

9. **Zheltonozhskii, E., et al. (2022).** "Contrast to Divide: Self-Supervised Pre-Training for Learning with Noisy Labels (C2D)." *WACV 2022.*

10. **He, K., Fan, H., Wu, Y., Xie, S., & Girshick, R. (2020).** "Momentum Contrast for Unsupervised Visual Representation Learning (MoCo)." *CVPR 2020.*

11. **Zhang, H., Cisse, M., Dauphin, Y. N., & Lopez-Paz, D. (2018).** "mixup: Beyond Empirical Risk Minimization." *ICLR 2018.*
