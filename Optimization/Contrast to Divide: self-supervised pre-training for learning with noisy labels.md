# Contrast to Divide: self-supervised pre-training for learning with noisy labels

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

C2D(Contrast to Divide)의 핵심 주장은 다음과 같습니다:

> **"노이즈 레이블 학습(LNL)에서 warm-up 단계의 한계를 자기지도 사전학습(self-supervised pre-training)으로 대체함으로써, 노이즈 수준에 무관하게 강인하고 정확한 노이즈 샘플 분리 및 반지도 학습이 가능하다."**

기존 DivideMix는 warm-up 단계에서 노이즈 데이터 전체를 활용한 지도학습에 의존했기 때문에:
- 높은 노이즈 레벨에 취약
- 하이퍼파라미터에 민감
- 오버피팅 문제 발생

이를 해결하기 위해 SimCLR 기반 대조 학습으로 레이블 없이 사전학습을 수행합니다.

### 주요 기여

| 기여 항목 | 내용 |
|-----------|------|
| **C2D 프레임워크 제안** | 자기지도 사전학습을 LNL에 통합 |
| **Warm-up 단계 개선** | 기존 대비 2~6배 짧은 warm-up (CIFAR-10: 10→5 epoch, CIFAR-100: 30→5 epoch) |
| **노이즈 감지 정확도 향상** | 높은 초기 ROC-AUC 및 빠른 상승 |
| **SOTA 달성** | CIFAR-10/100 고노이즈(90%) 환경에서 대폭 개선 |
| **외부 데이터 불필요** | ImageNet 사전학습 없이 Clothing-1M 유사 성능 달성 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

DivideMix의 warm-up 단계는 두 가지 목표를 동시에 달성해야 합니다:

1. **분리 가능한 손실 분포(Separable Loss Distribution)**: 클린 샘플과 노이즈 샘플의 손실 값이 구별 가능해야 함
2. **좋은 피처 추출(Feature Extraction)**: 반지도 학습 단계에서 활용될 고품질 표현 학습

그러나 이 두 목표는 **트레이드오프** 관계에 있습니다:
- 오래 훈련 → 더 좋은 피처, 하지만 노이즈에 오버피팅 → 분리 능력 감소
- 일찍 중단 → 분리 능력 유지, 하지만 피처 품질 저하

### 2.2 제안하는 방법 (수식 포함)

#### Step 1: SimCLR 기반 자기지도 사전학습

SimCLR의 대조 손실(NT-Xent Loss)을 사용합니다:

$$\mathcal{L}_{i,j} = -\log \frac{\exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_j)/\tau)}{\sum_{k=1}^{2N} \mathbf{1}_{[k \neq i]} \exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_k)/\tau)}$$

여기서:
- $\mathbf{z}_i, \mathbf{z}_j$: 동일 이미지의 두 augmented view에서 얻은 표현
- $\text{sim}(\mathbf{u}, \mathbf{v}) = \mathbf{u}^\top \mathbf{v} / (\|\mathbf{u}\| \|\mathbf{v}\|)$: 코사인 유사도
- $\tau$: temperature 파라미터
- $N$: 미니배치 크기

이 과정은 **레이블을 전혀 사용하지 않으므로**, 노이즈 레이블의 영향을 완전히 제거합니다.

#### Step 2: 짧은 Warm-up

자기지도 사전학습된 가중치로 초기화 후, 5 epoch의 짧은 지도학습 warm-up을 수행합니다. 이 단계에서 레이블 스무딩(label smoothing) 등 노이즈 대응 기법을 활용합니다.

#### Step 3: GMM 기반 노이즈 분리 (DivideMix 방식 계승)

훈련된 모델의 손실 분포에 **혼합 가우시안 모델(Gaussian Mixture Model)**을 적합합니다:

$$p(l_i) = \sum_{k=1}^{2} \pi_k \cdot \mathcal{N}(l_i \mid \mu_k, \sigma_k^2)$$

각 샘플 $i$에 대한 클린 확률:

$$w_i = p(\text{clean} \mid l_i) = \frac{\pi_{\text{clean}} \cdot \mathcal{N}(l_i \mid \mu_{\text{clean}}, \sigma_{\text{clean}}^2)}{p(l_i)}$$

임계값 $\tau = 0.03$ (DivideMix의 0.5보다 훨씬 낮음)을 기준으로:

$$\text{샘플 } i \in \begin{cases} \hat{X} \text{ (클린 집합)} & \text{if } w_i \geq \tau \\ \hat{U} \text{ (비레이블 집합)} & \text{if } w_i < \tau \end{cases}$$

> C2D에서 $\tau$가 낮아도 되는 이유는, 자기지도 사전학습 덕분에 대부분의 노이즈 샘플이 **매우 높은 손실**을 가져 이미 고확신으로 분리되기 때문입니다.

#### Step 4: MixMatch 기반 반지도 학습

분리된 클린 집합 $\hat{X}$와 비레이블 집합 $\hat{U}$를 이용해 MixMatch를 적용합니다.

MixMatch의 전체 손실:

$$\mathcal{L} = \mathcal{L}_X + \lambda_U \mathcal{L}_U$$

여기서:

$$\mathcal{L}_X = \frac{1}{|\hat{X}'|} \sum_{x, p \in \hat{X}'} H(p, f_\theta(x))$$

$$\mathcal{L}_U = \frac{1}{C|\hat{U}'|} \sum_{u, q \in \hat{U}'} \|q - f_\theta(u)\|_2^2$$

- $H$: 크로스 엔트로피
- $f_\theta$: 모델
- $\lambda_U$: 비레이블 손실 가중치 (튜닝됨, 예: 500 for CIFAR-100 80% noise)
- $\hat{X}', \hat{U}'$: MixUp 후의 혼합 배치

MixUp:

$$\tilde{x} = \lambda x_i + (1-\lambda) x_j, \quad \tilde{y} = \lambda y_i + (1-\lambda) y_j, \quad \lambda \sim \text{Beta}(\alpha, \alpha)$$

### 2.3 모델 구조

```
[입력 데이터 (레이블 포함, 노이즈 있음)]
          │
          ▼
┌─────────────────────────────────┐
│ 1단계: SimCLR 자기지도 사전학습  │
│  - 레이블 무시                   │
│  - 1000 epoch (CIFAR)           │
│  - NT-Xent Loss 사용            │
│  - 도메인 갭 없음                │
└─────────────────────────────────┘
          │ 초기화
          ▼
┌─────────────────────────────────┐
│ 2단계: 짧은 Warm-up (5 epoch)   │
│  - 노이즈 레이블로 지도학습      │
│  - 레이블 스무딩 적용            │
└─────────────────────────────────┘
          │ 손실 분포
          ▼
┌─────────────────────────────────┐
│ 3단계: GMM 기반 노이즈 분리      │
│  - 클린/노이즈 분류              │
│  - τ = 0.03                     │
└─────────────────────────────────┘
          │ 클린셋 / 비레이블셋
          ▼
┌─────────────────────────────────┐
│ 4단계: MixMatch 반지도 학습     │
│  - 클린셋: Cross-Entropy Loss   │
│  - 비레이블셋: MSE Loss         │
│  - MixUp 데이터 증강            │
└─────────────────────────────────┘
          │ 반복 (3↔4)
          ▼
     [최종 분류기]
```

**사용 아키텍처**: PreAct ResNet-18, ResNet-50

**2개 네트워크 병렬 운용**: DivideMix 방식을 계승하여 두 네트워크가 서로의 노이즈 분리 결과를 활용하여 상호 학습(co-training)

### 2.4 성능 향상

#### CIFAR-10 (대칭 노이즈)

| 방법 | 20% | 50% | 80% | 90% |
|------|-----|-----|-----|-----|
| DivideMix | 95.7 | 94.4 | 92.9 | 75.4 |
| **C2D (ours)** | **96.23** | **95.15** | **94.30** | **93.42** |
| **향상폭** | +0.53 | +0.75 | +1.4 | **+18.02** |

#### CIFAR-100 (대칭 노이즈)

| 방법 | 20% | 50% | 80% | 90% |
|------|-----|-----|-----|-----|
| DivideMix | 76.9 | 74.2 | 61.3 | 31.0 |
| **C2D (ResNet-18)** | **78.32** | **76.07** | **67.43** | **58.45** |
| **향상폭** | +1.42 | +1.87 | +6.13 | **+27.45** |

#### Clothing-1M (실제 노이즈, ~38.5%)

| 방법 | 정확도 |
|------|--------|
| DivideMix | 74.76% |
| ELR+ | **74.81%** |
| **C2D** | 74.30% |

> Clothing-1M에서 C2D는 ImageNet 사전학습 없이도 거의 동등한 성능을 달성

### 2.5 한계

1. **높은 사전학습 비용**: SimCLR을 1000 epoch 훈련해야 하며, 4개의 NVIDIA 2080 Ti GPU가 필요합니다. 실용적 적용에서 계산 비용이 상당합니다.

2. **비대칭 노이즈(Asymmetric Noise) 취약성**: 대칭 노이즈에서는 우수하지만, 비대칭 노이즈(40%)에서는 peak-final 정확도 격차가 커집니다 (CIFAR-10: 93.45% → 90.75%).

3. **Task Gap**: 대조 학습은 다운스트림 태스크와 무관하게 사전학습되므로, 태스크 갭(task gap)이 존재합니다.

4. **Clothing-1M 성능 한계**: 낮은 노이즈 비율 + ImageNet과 유사 도메인에서 자기지도 사전학습의 이점이 줄어듭니다.

5. **오픈셋(Open-set) 노이즈 미고려**: 논문은 closed-set 노이즈 설정만을 다루며, 훈련 클래스에 없는 이미지가 포함된 경우는 다루지 않습니다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 자기지도 사전학습이 일반화에 기여하는 메커니즘

#### (1) 노이즈 불변 표현 학습

레이블 없이 학습된 표현은 **레이블 노이즈에 의한 편향(bias)**이 없습니다.

$$\text{(기존)} \quad \theta^* = \arg\min_\theta \mathbb{E}_{(x,\tilde{y}) \sim \tilde{D}} [\ell(f_\theta(x), \tilde{y})]$$

$$\text{(C2D)} \quad \theta_{\text{SSL}}^* = \arg\min_\theta \mathbb{E}_{x \sim D} [\mathcal{L}_{\text{contrastive}}(f_\theta(x))]$$

SimCLR 학습된 표현 $\theta_{\text{SSL}}^*$은 실제 데이터 분포 $D$로부터 학습되므로, 노이즈 레이블 $\tilde{y}$의 영향을 받지 않습니다. 이것이 **일반화의 근본적 이점**입니다.

#### (2) 도메인 갭 제거

기존의 ImageNet 지도학습 사전학습은 source-target 도메인 차이로 인해 일반화 성능이 저하될 수 있습니다. 반면, C2D는 **훈련 데이터 자체**에서 사전학습을 수행하므로:

$$d(\mathcal{D}_{\text{train}}, \mathcal{D}_{\text{pretrain}}) = 0$$

이는 특히 의료 영상, 위성 이미지 등 **전문 도메인**에서 큰 이점을 가집니다.

#### (3) 피처 클러스터링 품질 향상 (UMAP 분석)

논문의 Figure 1(UMAP 시각화)에 따르면:

- **20% 노이즈**: C2D 피처가 DivideMix 대비 더 명확한 클러스터 형성
- **90% 노이즈**: DivideMix 피처는 클래스 구분이 붕괴되지만, C2D 피처는 일정 수준의 구조 유지

이는 C2D의 표현이 **노이즈 수준에 무관하게 일반화 가능한 구조**를 학습했음을 의미합니다.

#### (4) LNL과 반지도 학습의 갭 감소

논문의 Table 3에서 핵심적 발견:

| 방법 | 80% 노이즈 | 90% 노이즈 |
|------|-----------|-----------|
| MixMatch (SimCLR init.) | 71.86 | 66.10 |
| MixMatch (레이블 없음) | 70.46 | 64.60 |
| **C2D (ours)** | **71.65** | **64.30** |

C2D는 **실제 정보량이 더 많은** 반지도 학습 방법(MixMatch)과 거의 동등한 성능을 보입니다. 이는 좋은 표현이 **레이블 정보 부족을 보상**할 수 있음을 시사합니다:

$$\text{Information}(C2D) \approx \text{Information}(\text{Semi-supervised Oracle})$$

#### (5) 높은 노이즈 레벨에서의 안정성 (ROC-AUC 분석)

Figure 2에 따르면:
- C2D는 **초기 ROC-AUC가 더 높음** (사전학습 피처의 품질)
- **더 빠른 상승 속도** (적은 epoch으로 수렴)
- **더 안정적인 감소** (효과적 노이즈율의 안정적 감소)
- Peak-Final 간 격차가 DivideMix보다 작음

#### (6) 감독 사전학습과의 결정적 차이

ImageNet 지도학습 사전학습으로 초기화된 DivideMix는 Figure 3에서 보이듯이, **클린-노이즈 샘플의 손실 분포가 거의 겹쳐** 분리에 실패합니다.

$$\text{ImageNet 초기화 후}: \quad p(l \mid \text{clean}) \approx p(l \mid \text{noisy}) \quad \Rightarrow \text{분리 실패}$$

반면 C2D는:

$$p(l \mid \text{clean}) \ll p(l \mid \text{noisy}) \quad \Rightarrow \text{명확한 분리}$$

이는 지도 사전학습의 빠른 적응 특성이 오히려 노이즈 내성을 약화시킬 수 있음을 보여줍니다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4.1 앞으로의 연구에 미치는 영향

#### (1) 자기지도 학습과 LNL의 통합 패러다임 확립

C2D는 자기지도 학습을 LNL의 전처리 단계로 통합하는 **새로운 패러다임**을 제시합니다. 이후 연구들이 이 방향성을 계승할 것으로 예상됩니다. 실제로 이후 연구(예: SOP, UNICON 등)에서 유사한 접근이 시도되었습니다.

#### (2) 도메인 갭 문제 해결의 새로운 관점

기존 전이학습(Transfer Learning) 패러다임에 도전합니다. **"외부 대규모 데이터셋보다 자체 데이터의 자기지도 학습이 더 효과적일 수 있다"**는 명제는 데이터가 희소하거나 전문적인 도메인에서 중요한 시사점을 가집니다.

#### (3) 반지도 학습의 노이즈 레이블 강인성 연구 촉진

C2D가 거의 반지도 학습(Semi-SL) 오라클 수준에 도달했다는 사실은, **LNL과 Semi-SL의 통합적 이해**를 위한 새로운 연구 방향을 제시합니다.

#### (4) 의료/전문 도메인 적용 가능성

ImageNet 등 대규모 주석 데이터가 없는 분야에서 **노이즈 레이블 + 자기지도 학습** 조합의 실용적 가능성을 보여줍니다.

### 4.2 앞으로 연구 시 고려할 점

#### (1) 계산 비용 최적화

- SimCLR 1000 epoch 훈련은 4개 GPU 필요 → 경량화 연구 필요
- BYOL, MoCo v3, DINO 등 더 효율적인 자기지도 방법 적용 고려
- **고려 방향**: 짧은 사전학습으로도 충분한 성능을 낼 수 있는지 분석 필요

#### (2) 비대칭 노이즈 및 오픈셋 노이즈 대응

- 비대칭 노이즈에서 peak-final 정확도 격차 문제
- 실제 환경의 노이즈는 완전 무작위가 아님 (클래스 의존적 노이즈)
- **고려 방향**: 클래스-의존적 노이즈 전이 행렬 추정과 자기지도 학습의 결합

#### (3) 더 강력한 자기지도 프레임워크 탐색

- SimCLR보다 이후 등장한 방법들 (DINO, MAE, SimSiam 등) 활용
- 특히 Vision Transformer(ViT) 기반 자기지도 학습과의 결합
- **고려 방향**: DINO의 self-distillation 특성이 노이즈 분리에 더 유리할 수 있음

#### (4) 레이블 활용 효율성

- C2D는 클린 레이블을 전혀 활용하지 않는 완전 비지도 사전학습
- 소량의 클린 레이블을 활용하는 **감독 대조 학습(Supervised Contrastive Learning, SupCon)**과의 결합 고려

$$\mathcal{L}_{\text{SupCon}} = \sum_{i \in I} \frac{-1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(\mathbf{z}_i \cdot \mathbf{z}_p / \tau)}{\sum_{a \in A(i)} \exp(\mathbf{z}_i \cdot \mathbf{z}_a / \tau)}$$

#### (5) 노이즈 분리 방법의 고도화

- GMM 기반 분리는 단순하지만, 손실 분포가 단순 가우시안을 따르지 않는 경우 취약
- **고려 방향**: 에너지 기반 모델, 플로우 기반 밀도 추정 등을 활용한 더 유연한 분리 방법

#### (6) 이론적 보장 연구

- C2D의 경험적 성능은 우수하지만, **이론적 보장**이 부재
- 자기지도 표현의 노이즈 분리 능력에 대한 이론적 분석 필요
- **고려 방향**: 정보 이론적 관점에서 $I(\mathbf{z}; y_{\text{clean}})$와 $I(\mathbf{z}; \tilde{y})$의 관계 분석

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 방법 | 특징 | C2D와의 차이점 |
|------|------|------|----------------|
| **DivideMix** (Li et al., 2020) | GMM + MixMatch | LNL→Semi-SL 전환 | warm-up 불안정, 고노이즈 취약 |
| **ELR+** (Liu et al., 2020) | Early Learning Regularization | 조기학습 정규화로 노이즈 메모라이제이션 방지 | 자기지도 사전학습 미사용 |
| **C2D (본 논문)** | SimCLR + DivideMix | 자기지도 사전학습 통합 | 도메인 갭 없음, 고노이즈 강인 |
| **UNICON** (Karim et al., 2022) | Contrastive + Uniform Selection | 균일 샘플링 기반 노이즈 분리 | C2D보다 더 정교한 샘플 선택 |
| **SOP** (Liu et al., 2022) | Self-supervised + Over-parameterized | 과파라미터 네트워크 활용 | 이론적 보장 제공 |
| **NoiseRank** | 그래프 기반 노이즈 탐지 | 샘플 간 관계 활용 | 자기지도 미사용 |

> **주의**: UNICON, SOP 등 2021년 이후 논문들에 대한 세부 수치는 본 논문(ICLR 2021 제출)에 포함되지 않았으므로, 해당 논문들의 원문을 직접 확인하시기 바랍니다.

---

## 참고자료

1. **본 논문 (주요 출처)**:
   - Anonymous authors, "Contrast to Divide: Self-supervised Pre-training for Learning with Noisy Labels," *Under review at ICLR 2021* (제공된 PDF)

2. **핵심 기반 논문**:
   - Li, J., Socher, R., & Hoi, S.C.H. (2020). "DivideMix: Learning with Noisy Labels as Semi-supervised Learning." *ICLR 2020*. https://openreview.net/forum?id=HJgExaVtwr
   - Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020a). "A Simple Framework for Contrastive Learning of Visual Representations." *arXiv:2002.05709*
   - Chen, T., Kornblith, S., Swersky, K., Norouzi, M., & Hinton, G. (2020b). "Big Self-Supervised Models are Strong Semi-Supervised Learners." *arXiv:2006.10029*
   - Berthelot, D., et al. (2019). "MixMatch: A Holistic Approach to Semi-Supervised Learning." *NeurIPS 2019*

3. **관련 비교 연구**:
   - Liu, S., Niles-Weed, J., Razavian, N., & Fernandez-Granda, C. (2020). "Early-Learning Regularization Prevents Memorization of Noisy Labels." *arXiv:2007.00151*
   - Khosla, P., et al. (2020). "Supervised Contrastive Learning." *arXiv:2004.11362*

4. **GitHub 코드**:
   - https://github.com/ContrastToDivide/C2D
   - SimCLR 구현: https://github.com/HobbitLong/SupContrast
