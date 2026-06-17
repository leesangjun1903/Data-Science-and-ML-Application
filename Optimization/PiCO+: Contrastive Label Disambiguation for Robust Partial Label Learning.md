# PiCO+: Contrastive Label Disambiguation for Robust Partial Label Learning 

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

PiCO+는 **Partial Label Learning(PLL)** 에서 두 가지 핵심 난제인 **표현 학습(representation learning)** 과 **레이블 모호성 해소(label disambiguation)** 를 하나의 통합 프레임워크로 해결하며, 기존 지도 학습에 근접하는 성능을 달성할 수 있다고 주장합니다. 특히 실제 환경에서 ground-truth가 후보 집합에 포함되지 않는 **Noisy PLL** 설정까지 확장하여 강건한 학습을 가능하게 합니다.

### 주요 기여

| 기여 분야 | 내용 |
|-----------|------|
| **방법론** | PLL에 Contrastive Learning을 최초로 적용한 PiCO 프레임워크 제안 |
| **실용성** | Noisy PLL을 위한 PiCO+ 확장: 거리 기반 클린 샘플 선택 + 반지도 학습 |
| **실험** | 표준/노이즈 PLL 및 Fine-grained 분류 데이터셋에서 SOTA 달성 |
| **이론** | EM 알고리즘 관점에서의 수학적 정당화 제공 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**Partial Label Learning(PLL)** 의 형식적 정의:
- 입력 공간 $\mathcal{X}$, 출력 레이블 공간 $\mathcal{Y} = \{1, 2, \ldots, C\}$
- 훈련 데이터셋: $\mathcal{D} = \{(\boldsymbol{x}\_i, Y_i)\}_{i=1}^n$
- 기본 가정: ground-truth $y_i \in Y_i$ (표준 PLL), $y_i \notin Y_i$ 가능 (Noisy PLL)

**핵심 딜레마:**
- 좋은 표현 학습 → 효과적인 레이블 모호성 해소에 필요
- 레이블 불확실성 → 표현 학습 품질 저하
- 두 문제가 **상호 의존적**이어서 기존 방법으로는 동시 해결 불가

**추가 문제 (Noisy PLL):**
- 후보 집합에 ground-truth가 없을 경우, 기존 PLL 방법은 오류 레이블에 과적합됨
- 예: CIFAR-10에서 20% 노이즈 시 PRODEN 정확도 $-10.62\%$ 급락

---

### 2.2 제안 방법 및 수식

#### ▶ 기본 분류 손실 (Classification Loss)

각 샘플 $\boldsymbol{x}_i$에 정규화된 의사 타겟 벡터 $\boldsymbol{s}_i \in [0,1]^C$ 를 할당:

$$\mathcal{L}_{\text{cls}}(f; \boldsymbol{x}_i, Y_i) = \sum_{j=1}^{C} -s_{i,j} \log(f^j(\boldsymbol{x}_i))$$

$$\text{s.t.} \quad \sum_{j \in Y_i} s_{i,j} = 1 \quad \text{and} \quad s_{i,j} = 0, \; \forall j \notin Y_i $$

#### ▶ Contrastive Loss (대조 학습 손실)

임베딩 풀 $A = B_q \cup B_k \cup \text{queue}$ 에서 샘플별 대조 손실:

$$\mathcal{L}_{\text{cont}}(g; \boldsymbol{x}, \tau, A) = -\frac{1}{|P(\boldsymbol{x})|} \sum_{\boldsymbol{k}_+ \in P(\boldsymbol{x})} \log \frac{\exp(\boldsymbol{q}^\top \boldsymbol{k}_+ / \tau)}{\sum_{\boldsymbol{k}' \in A(\boldsymbol{x})} \exp(\boldsymbol{q}^\top \boldsymbol{k}' / \tau)} $$

여기서 $\tau$는 온도 파라미터, $A(\boldsymbol{x}) = A \setminus \{\boldsymbol{q}\}$

#### ▶ Positive Set 선택 (핵심 혁신)

분류기의 예측을 활용해 동일 예측 레이블 샘플을 Positive Set으로 구성:

$$\tilde{y} = \arg\max_{j \in Y} f^j(\text{Aug}_q(\boldsymbol{x}))$$

$$P(\boldsymbol{x}) = \{\boldsymbol{k}' \mid \boldsymbol{k}' \in A(\boldsymbol{x}), \; \tilde{y}' = \tilde{y}\} $$

#### ▶ 의사 타겟 업데이트 (Prototype-based Label Disambiguation)

이동 평균(Moving-Average) 방식으로 의사 타겟을 점진적 업데이트:

$$\boldsymbol{s} = \phi \boldsymbol{s} + (1-\phi)\boldsymbol{z}, \quad z_c = \begin{cases} 1 & \text{if } c = \arg\max_{j \in Y} \boldsymbol{q}^\top \boldsymbol{\mu}_j \\ 0 & \text{else} \end{cases} $$

초기화: $s_j = \frac{1}{|Y|} \mathbb{I}(j \in Y)$ (균등 분포), $\phi \in (0,1)$은 이동 평균 계수

#### ▶ 프로토타입 업데이트

클래스 $c$의 프로토타입 임베딩 벡터를 이동 평균으로 갱신:

$$\boldsymbol{\mu}_c = \text{Normalize}(\gamma \boldsymbol{\mu}_c + (1-\gamma)\boldsymbol{q}), \quad \text{if } c = \arg\max_{j \in Y} f^j(\text{Aug}_q(\boldsymbol{x})) $$

#### ▶ PiCO 전체 손실

$$\mathcal{L}_{\text{pico}} = \mathcal{L}_{\text{cls}} + \lambda \mathcal{L}_{\text{cont}} $$

---

#### ▶ PiCO+: Noisy PLL 확장

**① 거리 기반 클린 샘플 선택:**

$$\mathcal{D}_{\text{clean}} = \{(\boldsymbol{x}_i, Y_i) \mid \boldsymbol{q}_i^\top \boldsymbol{\mu}_{\tilde{y}_i} > \kappa_\delta\} $$

$\kappa_\delta$: cosine similarity의 $(100-\delta)$-백분위수 임계값

**② Noisy 샘플의 Positive Set (Neighbor-Augmented):**

Label-driven 방식:

$$P_{\text{noisy}}(\boldsymbol{x}) = \{\boldsymbol{k}' \mid \boldsymbol{k}' \in A(\boldsymbol{x}), \; \hat{y}' = \hat{y}\}, \quad \hat{y} = \begin{cases} \arg\max_{1 \le j \le L} f^j(\cdot) & \boldsymbol{x} \in \mathcal{D}_{\text{noisy}} \\ \arg\max_{j \in Y} f^j(\cdot) & \text{else} \end{cases} $$

kNN 기반 방식:

$$P_{\text{knn}}(\boldsymbol{x}) = \{\boldsymbol{k}' \mid \boldsymbol{k}' \in A(\boldsymbol{x}) \cap \mathcal{N}_k(\boldsymbol{x})\} $$

**③ Prototype-based Label Guessing (Noisy 샘플):**

$$s'_j = \frac{\exp(\boldsymbol{q}^\top \boldsymbol{\mu}_j / \tau)}{\sum_{t=1}^{L} \exp(\boldsymbol{q}^\top \boldsymbol{\mu}_t / \tau)}, \quad \forall 1 \le j \le L $$

**④ Mixup 정규화:**

$$\boldsymbol{x}^m = \sigma \text{Aug}_q(\boldsymbol{x}_i) + (1-\sigma)\text{Aug}_q(\boldsymbol{x}_j), \quad \boldsymbol{s}^m = \sigma\hat{\boldsymbol{s}}_i + (1-\sigma)\hat{\boldsymbol{s}}_j, \quad \sigma \sim \text{Beta}(\varsigma, \varsigma) $$

**⑤ PiCO+ 전체 손실:**

$$\mathcal{L}_{\text{pico+}} = \mathcal{L}_{\text{mix}} + \alpha \mathcal{L}_{\text{clean}} + \beta(\mathcal{L}_{\text{n-cont}} + \mathcal{L}_{\text{knn}} + \mathcal{L}_{\text{n-cls}}) $$

$\alpha = 2, \beta = 0.1$ (클린 샘플 주도적 학습)

---

### 2.3 모델 구조

```
입력 이미지 x
    ↓ Data Augmentation (SimAugment / RandAugment)
┌─────────────────────┐    ┌──────────────────────────┐
│   Query Network g   │    │   Key Network g' (모멘텀) │
│  (ResNet-18 Encoder)│    │  (ResNet-18 Encoder)     │
│   + Prediction Head │    │   + Projection Head      │
│   (2-layer MLP)     │    │   (2-layer MLP)          │
└─────────────────────┘    └──────────────────────────┘
         ↓ q (L2-norm)              ↓ k (L2-norm)
    ┌────┴──────────────────────────┴────┐
    │        Embedding Pool A            │
    │  A = B_q ∪ B_k ∪ Queue(8192)     │
    └──────────────┬─────────────────────┘
                   ↓
    ┌──────────────┴────────────────┐
    │     Positive Set Selection    │
    │  P(x): 동일 예측 레이블 기반   │
    └──────────────┬────────────────┘
         ↓                    ↓
   Contrastive Loss      Class Prototypes μ_c
   (L_cont)               ↓
                    Prototype-based
                    Label Disambiguation
                    (의사 타겟 s 업데이트)
                          ↓
                   Classifier (Softmax)
                   Cross-Entropy Loss (L_cls)
```

**주요 하이퍼파라미터:**
- Backbone: ResNet-18
- Queue 크기: 8,192
- 모멘텀 계수: 0.999 (네트워크), $\gamma = 0.99$ (프로토타입)
- 온도: $\tau = 0.07$
- 손실 가중치: $\lambda = 0.5$
- Batch size: 256, Epochs: 800

---

### 2.4 성능 향상

#### 표준 PLL (Standard PLL)

| 데이터셋 | 방법 | $q=0.1$ | $q=0.3$ | $q=0.5$ |
|---------|------|---------|---------|---------|
| CIFAR-10 | **PiCO+** | **95.99%** | **95.73%** | **95.33%** |
| CIFAR-10 | PiCO | 94.39% | 94.18% | 93.58% |
| CIFAR-10 | LWS (SOTA) | 90.30% | 88.99% | 86.16% |
| CIFAR-10 | Fully Supervised | ~94.91% | - | - |

- CIFAR-10에서 기존 최고 대비 최대 $+5.80\%$ 향상
- CIFAR-100에서 최대 $+16.01\%$ 향상

#### Noisy PLL

| 데이터셋 | 방법 | $q=0.5, \eta=0.2$ |
|---------|------|---------|
| CIFAR-10 | **PiCO+** | **92.59%** |
| CIFAR-10 | PRODEN | 77.15% |
| CIFAR-10 | LWS | 61.96% |

- 노이즈 비율 $\eta=0.2$ 시 기존 최고 대비 최대 $+12.52\%$ 향상

#### Fine-grained 분류 (CUB-200)

- PiCO: PRODEN 대비 $+9.61\%$ (CUB-200, $q=0.05$)

---

### 2.5 한계점

1. **계산 비용**: 대조 학습 + 프로토타입 업데이트로 인한 추가 메모리 및 연산 (약 800 에포크 필요)
2. **하이퍼파라미터 민감성**: $\phi$, $\lambda$, $\alpha$, $\beta$ 등 다수의 하이퍼파라미터가 성능에 민감하게 영향
3. **Noisy PLL 이론 부재**: PiCO+의 noisy 처리 부분에 대한 이론적 보장 미흡
4. **이진 분리의 한계**: 거리 기반 클린/노이즈 분리가 경계선상 샘플에 취약할 수 있음 (확인 편향 문제 존재)
5. **이미지 분류 위주**: 텍스트, 멀티모달 등 타 도메인 검증 부재
6. **수렴 분석 부재**: 이론적 수렴 보장 없이 실험적 관찰에 의존

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 이론적 근거: EM 관점에서의 일반화

논문은 PiCO가 다음의 우도 최대화 문제를 암묵적으로 수행함을 증명합니다:

$$\arg\max_\theta \sum_{i=1}^n \log P(Y_i, \boldsymbol{x}_i | \theta) \ge \arg\max_\theta \sum_{i=1}^n \sum_{y_i \in Y_i} \pi_i^{y_i} \log \frac{P(\boldsymbol{x}_i, y_i|\theta)}{\pi_i^{y_i}} $$

**Theorem 1**: 동일 클래스 데이터가 von Mises-Fisher(vMF) 분포를 따른다고 가정할 때:

$$R_1 = \sum_{S_j \in \mathcal{D}_C} \frac{n_j}{n} \|\boldsymbol{\mu}_j\|^2 \le \sum_{S_j \in \mathcal{D}_C} \frac{n_j}{n} \|\boldsymbol{\mu}_j\| = R_2 $$

대조 손실 최소화(Eq. 15)는 우도의 하한(Eq. 16)을 최대화하는 것과 동치이며, $\|\boldsymbol{\mu}_j\| \to 1$ 일수록 하한이 tight해집니다.

### 3.2 일반화 향상의 핵심 메커니즘

**① 클래스 내 분산 최소화:**

$$\text{(alignment term)} \approx \frac{1}{\tau n} \sum_{S_j \in \mathcal{D}_C} \sum_{\boldsymbol{x} \in S_j} \|g(\boldsymbol{x}) - \boldsymbol{\mu}_j\|^2 $$

이는 클래스 내 분산을 최소화하여 임베딩 공간에서 클래스 간 명확한 경계를 형성합니다. 문헌[54](Saunshi et al., 2019)에 따르면 **클래스 내 편차 최소화**는 다운스트림 태스크의 일반화 오차 상한을 tighter하게 만듭니다.

**② vMF 분포 기반 클러스터링:**
- 대조 학습이 유닛 하이퍼구면 위에서 데이터를 vMF 분포의 혼합으로 매핑
- 클러스터 집중도 파라미터 $\kappa$ 와 평균 벡터 노름 간의 관계 (대 $\|\boldsymbol{\mu}_j\|$):

$$\kappa \approx \frac{d-1}{2(1-\|\boldsymbol{\mu}_j\|)}, \quad \text{(large } \|\boldsymbol{\mu}_j\| \text{에서 유효)} $$

높은 $\|\boldsymbol{\mu}_j\|$ → 높은 집중도 → 더 구별 가능한 표현 → **일반화 향상**

**③ Mixup 정규화 효과:**
- PiCO+에서 Mixup은 완전 지도 학습에서도 $+1.05\%$ 향상
- 노이즈 PLL에서 $+2.92\%$ (CIFAR-10), 더 큰 효과 → **노이즈 강건성**이 일반화에 기여

**④ 균일성(Uniformity) 항의 역할:**
대조 손실의 (b)항은 임베딩 공간에서 표현의 균일한 분포를 장려하여 정보 보존성을 높임:

$$\text{(b)} = \frac{1}{n}\sum_{\boldsymbol{x}\in\mathcal{D}} \log \sum_{\boldsymbol{k}'\in A(\boldsymbol{x})} \exp(\boldsymbol{q}^\top\boldsymbol{k}'/\tau)$$

이는 Wang & Isola (2020)[11]의 분석처럼 표현의 다양성을 보장합니다.

### 3.3 도메인 일반화 가능성

- **Fine-grained 분류**: CUB-200에서 기존 방법 대비 $+9.61\%$ → 어려운 분류 과제에서도 강한 일반화
- **비균일 데이터 생성**: Table 12에서 비균일 레이블 플리핑 행렬 하에서도 안정적 성능
- **이론의 이전 가능성**: EM 해석이 MoCo, SupCon 등 다른 CL 방법에도 적용 가능

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 향후 연구에 미치는 영향

**① PLL-CL 연구 패러다임 전환**
- 이 논문은 PLL에 CL을 최초로 적용하여, 이후 연구들이 표현 학습 품질을 핵심 요소로 다루게 하는 방향을 제시합니다. 후속 연구에서 다양한 CL 변형(예: SimCLR, BYOL 기반)을 PLL에 적용하는 시도가 늘어날 것으로 예상됩니다.

**② EM 프레임워크의 이론 확장**
- EM + CL의 통합 이론은 다른 약지도 학습(Weakly Supervised Learning) 설정에서도 활용 가능합니다. 특히 반지도 학습, 인스턴스 의존적 노이즈 레이블 학습 등에 적용 가능한 이론적 토대를 제공합니다.

**③ Open-world PLL로의 확장**
- PiCO+의 noisy PLL 처리 방식은 실제 환경의 레이블 불완전성 문제를 다루는 연구에 직접적 영향을 미칩니다. 추후 오픈셋(Open-set) 환경, OOD(Out-of-Distribution) 탐지와의 결합 연구가 활성화될 가능성이 있습니다.

**④ 프로토타입 학습의 재조명**
- 클래스 프로토타입 기반의 레이블 모호성 해소는 few-shot learning, zero-shot learning에서도 중요합니다. PiCO의 접근은 메타 학습(meta-learning) 연구와의 교차 적용 가능성을 열어줍니다.

---

### 4.2 향후 연구 시 고려사항

**① 이론적 강화**
- 현재 Theorem 1의 하한과 실제 우도 간 간극이 존재함 ($R_1 \le R_2$). 더 tight한 바운드 개발이 필요합니다.
- PiCO+ (noisy PLL)에 대한 이론적 수렴 보장이 아직 없으므로, 향후 이론 연구가 필요합니다.

**② 스케일 및 효율성**
- 800 에포크 학습은 실제 산업 적용에 부담입니다. 학습 효율화(early stopping 기준, 경량화 프로토타입 등)에 대한 연구가 필요합니다.
- 대규모 데이터셋(ImageNet 수준) 및 Vision Transformer(ViT) 백본 적용 검증이 요구됩니다.

**③ 멀티모달 및 타 도메인 적용**
- 현재 이미지 분류에 집중되어 있으나, 텍스트 분류(NLP), 의료 영상, 음성 인식 등에서의 PLL 문제에 적용 가능성을 검토해야 합니다.
- 특히 **대형 언어 모델(LLM)의 레이블 모호성** 문제(예: instruction tuning에서의 모호한 인간 평가)와의 연결도 중요한 연구 방향입니다.

**④ Noisy PLL의 더 정교한 분리 전략**
- 거리 기반 클린 샘플 선택은 고정 임계값($\kappa_\delta$)에 의존합니다. 적응적(adaptive) 임계값 설정이나 베이지안 접근법으로 개선 가능합니다.
- 확인 편향(Confirmation Bias) 완화를 위한 앙상블 방식의 활용도 고려해야 합니다.

**⑤ 캔디데이트 집합 생성 현실성**
- 실험에서 캔디데이트 집합 생성이 균일 플리핑 행렬 가정에 의존합니다. 실제 어노테이터의 행동 모델을 반영한 더 현실적인 생성 프로세스와의 비교 검증이 필요합니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

아래는 논문 내 인용 및 관련 연구를 바탕으로 구성한 비교 분석입니다. ⚠️ **주의**: 2023년 이후 최신 논문 비교는 본 논문(arXiv 2022.11)에 포함되지 않으므로, 제가 직접 확인한 내용만 서술하며 불확실한 외부 논문 정보는 포함하지 않습니다.

### 논문 내 인용된 2020년 이후 주요 관련 연구

| 연구 | 연도 | 핵심 방법 | PiCO와의 관계 |
|------|------|----------|--------------|
| **PRODEN** (Lv et al., ICML 2020) | 2020 | 자기 훈련 방식의 잠재 레이블 분포 업데이트 | PiCO가 CIFAR-10에서 최대 +5.80% 우위 |
| **CC** (Feng et al., NeurIPS 2020) | 2020 | 집합 수준 균일 데이터 생성 가정의 일관성 분류기 | PiCO 대비 성능 열위 |
| **LWS** (Wen et al., ICML 2021) | 2021 | 후보 레이블에 대한 가중 손실 함수 | PiCO가 CIFAR-100에서 +16% 이상 우위 |
| **SupCon** (Khosla et al., NeurIPS 2020) | 2020 | 지도 대조 학습 | PiCO의 CL 모듈 설계 기반 |
| **MoCo** (He et al., CVPR 2020) | 2020 | 모멘텀 대조 학습 | PiCO 아키텍처의 직접적 기반 |
| **DivideMix** (Li et al., ICLR 2020) | 2020 | 노이즈 레이블을 반지도 학습으로 처리 | PiCO+의 반지도 프레임워크 설계에 영감 |
| **MoPro** (Li et al., ICLR 2021) | 2021 | 모멘텀 프로토타입 기반 웹 지도 학습 | PiCO의 소프트 레이블 추정 방식 참고 |
| **Noisy PLL 이론** (Lv et al., 2021) | 2021 | 평균 기반 손실의 이론적 강건성 분석 | PiCO+의 noisy PLL 설정 동기 제공, 실질적 해결책은 PiCO+가 제시 |

### 방법론적 비교

```
표현 학습 품질:
Uniform Features < PRODEN < PiCO (t-SNE 시각화 근거)

레이블 모호성 해소 능력 (MMC 기준):
MA Soft Probs < One-hot/Soft Prototype < PiCO (Moving-Average 방식 우수)

노이즈 강건성:
LWS ≈ PRODEN < GCE < PiCO < PiCO+
```

---

## 참고자료 (출처)

본 답변은 다음 자료를 기반으로 작성되었습니다:

1. **Wang, H., Xiao, R., Li, Y., Feng, L., Niu, G., Chen, G., & Zhao, J.** (2022). *PiCO+: Contrastive Label Disambiguation for Robust Partial Label Learning*. arXiv:2201.08984v3. (제공된 PDF 원문)

2. **Wang, H., et al.** (2022). *PiCO: Contrastive label disambiguation for partial label learning*. ICLR 2022. (논문 내 [9] 인용)

3. **Khosla, P., et al.** (2020). *Supervised Contrastive Learning*. NeurIPS 2020. (논문 내 [8] 인용)

4. **He, K., et al.** (2020). *Momentum Contrast for Unsupervised Visual Representation Learning*. CVPR 2020. (논문 내 [12] 인용)

5. **Lv, J., et al.** (2020). *Progressive identification of true labels for partial-label learning*. ICML 2020. (논문 내 [18] 인용)

6. **Wen, H., et al.** (2021). *Leveraged weighted loss for partial label learning*. ICML 2021. (논문 내 [19] 인용)

7. **Feng, L., et al.** (2020). *Provably consistent partial-label learning*. NeurIPS 2020. (논문 내 [20] 인용)

8. **Wang, T., & Isola, P.** (2020). *Understanding contrastive representation learning through alignment and uniformity on the hypersphere*. ICML 2020. (논문 내 [11] 인용)

9. **Li, J., Socher, R., & Hoi, S.C.H.** (2020). *DivideMix: Learning with noisy labels as semi-supervised learning*. ICLR 2020. (논문 내 [15] 인용)

10. **Saunshi, N., et al.** (2019). *A theoretical analysis of contrastive unsupervised representation learning*. ICML 2019. (논문 내 [54] 인용)

11. **Banerjee, A., et al.** (2005). *Clustering on the unit hypersphere using von Mises-Fisher distributions*. JMLR. (논문 내 [27] 인용)

---

> ⚠️ **정확도 관련 고지**: PiCO+ 논문 발표(2022년 11월) 이후 출판된 외부 논문(2023년 이후)과의 직접 비교는 제공된 PDF에 포함되지 않으므로 본 답변에서 제외하였습니다. 불확실한 외부 수치나 논문 정보를 임의로 생성하지 않았습니다.
