# Early-Learning Regularization Prevents Memorization of Noisy Labels

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

딥러닝 모델은 노이즈 레이블로 학습 시 **"Early Learning(초기 학습)"** 단계에서 먼저 정확한 레이블을 학습한 후, 이후 **"Memorization(암기)"** 단계에서 잘못된 레이블까지 암기하는 현상이 발생한다. 이 논문은 이 현상이 단순 선형 모델에서도 발생하는 **고차원 분류의 근본적 현상**임을 이론적으로 증명하고, 이를 방지하기 위한 **Early-Learning Regularization(ELR)** 프레임워크를 제안한다.

### 주요 기여 3가지

| 기여 | 내용 |
|------|------|
| **이론적 기여** | Early Learning & Memorization이 선형 모델에서도 발생함을 수학적으로 증명 (Theorem 1) |
| **방법론적 기여** | Cross-entropy 손실의 그래디언트를 보정하는 새로운 정규화 항(ELR) 제안 |
| **실험적 기여** | CIFAR-10/100, Clothing1M, WebVision 등 표준 벤치마크에서 SOTA 수준 달성 |

---

## 2. 상세 설명

### 2-1. 해결하고자 하는 문제

노이즈 레이블 환경에서 딥 신경망을 학습할 때 발생하는 두 가지 핵심 문제:

1. **Early Learning 단계** (초기): 올바른 레이블 패턴을 먼저 학습 → 잘못된 레이블 예시도 일시적으로 정확히 예측
2. **Memorization 단계** (후기): 잘못된 레이블까지 완전히 암기 → 일반화 성능 급격히 저하

기존 방법(Sample Selection, Label Correction)은 레이블 자체를 수정하거나 선택하는 방식으로, 이 현상을 근본적으로 방지하지 못한다.

---

### 2-2. 제안 방법 및 수식

#### (A) 이론적 배경: 선형 모델에서의 증명

데이터는 두 Gaussian 혼합에서 추출:

$$\mathbf{x} \sim \mathcal{N}(+\mathbf{v}, \sigma^2 I_{p \times p}) \quad \text{if } \mathbf{y}^* = (1,0)$$

$$\mathbf{x} \sim \mathcal{N}(-\mathbf{v}, \sigma^2 I_{p \times p}) \quad \text{if } \mathbf{y}^* = (0,1)$$

노이즈 레이블 생성:

$$\mathbf{y}^{[i]} = \begin{cases} (\mathbf{y}^*)^{[i]} & \text{with probability } 1 - \Delta \\ \tilde{\mathbf{y}}^{[i]} & \text{with probability } \Delta \end{cases} \tag{1}$$

선형 분류기의 Cross-Entropy 손실:

$$\min_{\Theta \in \mathbb{R}^{2 \times p}} \mathcal{L}_{\text{CE}}(\Theta) := -\frac{1}{n} \sum_{i=1}^{n} \sum_{c=1}^{2} \mathbf{y}_c^{[i]} \log(\mathcal{S}(\Theta \mathbf{x}^{[i]})_c) \tag{2}$$

이 손실의 그래디언트:

$$\nabla \mathcal{L}_{\text{CE}}(\Theta)_c = \frac{1}{n} \sum_{i=1}^{n} \mathbf{x}^{[i]} \left( \mathcal{S}(\Theta \mathbf{x}^{[i]})_c - \mathbf{y}_c^{[i]} \right)$$

**Theorem 1 (비공식)**: $\Delta \in (0,1)$에 대해 적절한 조건 하에서:
- $t < T$: Early Learning 성공 — 그래디언트가 올바른 분리 방향 $\mathbf{v}$와 잘 정렬됨
- $t = 0 \to T$: 올바른 레이블 예시의 그래디언트 계수 감소, 잘못된 레이블 예시의 계수 증가
- $t \to \infty$: 분류기가 모든 노이즈 레이블을 완전히 암기

#### (B) 딥러닝 모델에서의 그래디언트 분석

딥 신경망의 예측 확률:

$$\mathbf{p}^{[i]} := \mathcal{S}\left(\mathcal{N}_{\mathbf{x}^{[i]}}(\Theta)\right) \tag{3}$$

Cross-Entropy 손실:

$$\mathcal{L}_{\text{CE}}(\Theta) := -\frac{1}{n} \sum_{i=1}^{n} \sum_{c=1}^{C} \mathbf{y}_c^{[i]} \log \mathbf{p}_c^{[i]} \tag{4}$$

그래디언트:

$$\nabla \mathcal{L}_{\text{CE}}(\Theta) = \frac{1}{n} \sum_{i=1}^{n} \nabla \mathcal{N}_{\mathbf{x}^{[i]}}(\Theta) \left(\mathbf{p}^{[i]} - \mathbf{y}^{[i]}\right) \tag{5}$$

레이블 노이즈의 영향은 $\mathbf{p}^{[i]} - \mathbf{y}^{[i]}$ 항에 집중됨을 확인.

#### (C) Early-Learning Regularization (ELR) 손실 함수

$$\mathcal{L}_{\text{ELR}}(\Theta) := \mathcal{L}_{\text{CE}}(\Theta) + \frac{\lambda}{n} \sum_{i=1}^{n} \log\left(1 - \langle \mathbf{p}^{[i]}, \mathbf{t}^{[i]} \rangle\right) \tag{6}$$

- $\mathbf{t}^{[i]}$: 과거 모델 출력으로부터 추정된 **target probability vector**
- $\lambda$: 정규화 강도 하이퍼파라미터
- 로그 함수: softmax 내의 지수 함수를 상쇄하는 역할

**ELR 손실의 그래디언트 (Lemma 2)**:

$$\nabla \mathcal{L}_{\text{ELR}}(\Theta) = \frac{1}{n} \sum_{i=1}^{n} \nabla \mathcal{N}_{\mathbf{x}^{[i]}}(\Theta) \left(\mathbf{p}^{[i]} - \mathbf{y}^{[i]} + \lambda \mathbf{g}^{[i]}\right) \tag{7}$$

여기서 $\mathbf{g}^{[i]} \in \mathbb{R}^C$의 각 원소:

$$g_c^{[i]} := \frac{\mathbf{p}_c^{[i]}}{1 - \langle \mathbf{p}^{[i]}, \mathbf{t}^{[i]} \rangle} \sum_{k=1}^{C} \left( t_k^{[i]} - t_c^{[i]} \right) \mathbf{p}_k^{[i]}, \quad 1 \leq c \leq C \tag{8}$$

**정규화 항의 작동 원리**:
- **올바른 레이블 예시**: Early Learning 이후 $\mathbf{p}^{[i]} \approx \mathbf{y}^{[i]}$가 되어 CE 항이 소멸하려 할 때, $\mathbf{g}^{[i]}$가 이를 보완하여 올바른 예시의 그래디언트 크기를 유지
- **잘못된 레이블 예시**: CE 항의 부호를 $\mathbf{g}^{[i]}$가 상쇄하여 암기 방지

#### (D) Target 추정: Temporal Ensembling

$$\mathbf{t}^{[i]}(k) := \beta \mathbf{t}^{[i]}(k-1) + (1-\beta)\mathbf{p}^{[i]}(k) \tag{9}$$

- $\beta$: momentum (0 ≤ β < 1)
- 과거 예측의 지수이동평균으로 안정적인 target 유지

---

### 2-3. 모델 구조

```
ELR (기본)
├── 단일 신경망 (ResNet-34)
├── Temporal Ensembling으로 target 추정
└── ELR 손실로 학습

ELR+ (확장)
├── 두 개의 신경망 (상호 target 추정)
├── Weight Averaging (Mean Teacher 방식)
├── Temporal Ensembling
└── Mixup Data Augmentation
```

---

### 2-4. 성능 향상

#### CIFAR-10 (ResNet-34, Symmetric Noise)

| 방법 | 20% | 40% | 60% | 80% |
|------|-----|-----|-----|-----|
| Cross Entropy | 86.98 | 81.88 | 74.14 | 53.82 |
| SL | 89.83 | 87.13 | 82.81 | 68.12 |
| **ELR** | **91.16** | **89.15** | **86.12** | **73.86** |
| **ELR\*** | **92.12** | **91.43** | **88.87** | **80.69** |

#### Clothing1M (ResNet-50 pretrained)

| CE | DivideMix | ELR | **ELR+** |
|----|-----------|-----|---------|
| 69.10 | 74.76 | 72.87 | **74.81** |

---

### 2-5. 한계

1. **이론적 갭**: 선형 모델에서의 증명과 비선형 딥러닝 모델 간의 간격이 완전히 메워지지 않음
2. **ELR의 DivideMix 대비 성능**: ILSVRC12 top-1 정확도 등에서 DivideMix에 뒤처지는 경우 존재
3. **하이퍼파라미터 의존성**: $\beta=0$ 설정 시 성능이 38%까지 떨어지는 등 temporal ensembling 설정에 민감
4. **정규화 항 동역학 미분석**: ELR 정규화 자체의 이론적 수렴 분석 부재
5. **고노이즈 환경**: 80% 이상의 극단적 노이즈에서 성능 저하 여전히 존재

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 일반화 성능 향상의 핵심 메커니즘

ELR이 일반화 성능을 향상시키는 근본적인 이유는 **그래디언트 수준에서의 노이즈 영향 차단**에 있다.

일반적 Cross-Entropy 학습의 문제:

$$\nabla \mathcal{L}_{\text{CE}}(\Theta) = \frac{1}{n}\sum_{i=1}^{n} \nabla \mathcal{N}_{\mathbf{x}^{[i]}}(\Theta)\underbrace{(\mathbf{p}^{[i]} - \mathbf{y}^{[i]})}_{\text{레이블 노이즈 영향 집중}}$$

Early Learning 이후 올바른 레이블 예시의 $\|\mathbf{p}^{[i]} - \mathbf{y}^{[i]}\| \to 0$이 되면, 잘못된 레이블 예시의 그래디언트가 지배적이 되어 과적합 발생.

ELR의 보정:

$$\nabla \mathcal{L}_{\text{ELR}}(\Theta) = \frac{1}{n}\sum_{i=1}^{n} \nabla \mathcal{N}_{\mathbf{x}^{[i]}}(\Theta)\underbrace{(\mathbf{p}^{[i]} - \mathbf{y}^{[i]} + \lambda \mathbf{g}^{[i]})}_{\text{노이즈 영향 보정된 항}}$$

### 3-2. 일반화 성능에 기여하는 세부 요소

**(1) 암기 방지를 통한 평탄한 손실 지형(Loss Landscape) 유지**
- 잘못된 레이블 암기 = 과도하게 날카로운(sharp) 손실 극소점 형성
- ELR은 모델이 더 완만하고 일반화 가능한 극소점에 수렴하도록 유도

**(2) Semi-supervised 학습 기법과의 시너지**
- Temporal Ensembling이 일종의 **자기 교사(self-teaching)** 역할
- 노이즈 레이블을 넘어 데이터의 진정한 분포 구조를 학습
- Weight Averaging(ELR+)은 예측의 일관성을 강화하여 분산 감소 효과

**(3) Mixup Augmentation의 결합 효과**
- ELR+에서 Mixup을 적용하면 결정 경계 근방의 선형성 강제
- 일반화 성능이 특히 고노이즈(80%) 환경에서 크게 향상

Ablation study (CIFAR-10, Symmetric 80%):

| 구성 | 정확도 |
|------|--------|
| 1 Network, no mixup, no WA | 72.54 |
| 1 Network, mixup, WA | 87.23 |
| 2 Networks, mixup, WA | **88.62** |

**(4) 잘못된 레이블 예시에 대한 올바른 예측 유지**
- Figure 1의 bottom row에서 확인: 잘못된 레이블 예시들도 진짜 클래스로 올바르게 예측 유지
- 이는 모델이 **진짜 데이터 분포**를 학습했음을 의미 → 일반화 성능 보장

**(5) 하이퍼파라미터 강건성**
- $\beta$, $\lambda$ 변화에 대해 비교적 안정적인 성능 유지
- 다양한 노이즈 유형(symmetric, asymmetric)에서 일관된 성능

---

## 4. 앞으로의 연구에 미치는 영향과 고려사항

### 4-1. 연구에 미치는 영향

**(1) 이론적 영향**
- 노이즈 레이블 학습 문제를 **그래디언트 동역학** 관점에서 분석하는 새로운 패러다임 제시
- 선형 모델에서의 수학적 증명은 후속 이론 연구의 기반이 됨
- "Early Learning은 딥러닝에 특수한 현상이 아닌 고차원 분류의 본질적 현상"이라는 통찰

**(2) 방법론적 영향**
- Sample Selection 없이도 노이즈에 강건한 학습 가능함을 증명
- **정규화 기반 접근법**이 레이블 수정/선택보다 효율적일 수 있음을 시사
- Semi-supervised 학습 기법(Temporal Ensembling, Mean Teacher)을 지도학습 노이즈 문제에 응용하는 방향성 제시

**(3) 실용적 영향**
- 의료 영상, 크라우드소싱, 웹 크롤링 데이터 등 **실세계 노이즈 레이블 문제**에 직접 적용 가능
- 비교적 단순한 구조(단일 네트워크 + 정규화)로도 경쟁력 있는 성능 달성

---

### 4-2. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의**: 아래는 제가 학습 데이터를 기반으로 알고 있는 연구들로, 2020년 이후 실제 발표된 논문들에 대한 완전한 최신 정보를 보장하기 어렵습니다. 부정확할 수 있는 세부 수치는 제외하고 논문 제목과 방향성 위주로 서술합니다.

#### ELR 이후의 주요 연구 방향

| 연구 방향 | 대표 연구 | ELR과의 관계 |
|-----------|-----------|--------------|
| **대조 학습 + 노이즈 레이블** | SimCLR, MoCo 기반 접근 | 표현 학습으로 노이즈 내성 강화 |
| **준지도 학습 확장** | DivideMix (Li et al., 2020) | 두 네트워크 + MixMatch 결합 |
| **Transformer 기반** | ViT와 노이즈 레이블 결합 연구 | 대규모 모델의 노이즈 내성 분석 |
| **그래프 기반** | GNN을 이용한 레이블 전파 | 데이터 구조 활용 노이즈 수정 |
| **메타 학습 기반** | MAML 응용 | 소량 클린 데이터 활용 |

#### DivideMix (2020, ICLR)과의 비교

| 항목 | ELR | DivideMix |
|------|-----|-----------|
| 핵심 접근 | 정규화 기반 | 반지도 학습 + 혼합 모델 |
| 복잡도 | 낮음 (단순한 손실 변경) | 높음 (GMM fitting 필요) |
| 학습 시간 | 1.1h | 5.4h (CIFAR-10 기준) |
| CIFAR-10 80% | 93.2 | 93.2 (유사) |
| Clothing1M | 74.81 (ELR+) | 74.76 |
| 이론적 기반 | 명시적 이론 증명 있음 | 상대적으로 경험적 |

---

### 4-3. 앞으로 연구 시 고려할 점

**(1) 이론과 실제의 간극 해소**
- 선형 모델 → 비선형 딥러닝 모델로의 이론적 확장 필요
- 특히 Transformer, 대규모 언어 모델(LLM)에서의 Early Learning 현상 분석
- 정규화 항의 수렴 속도 및 안정성에 대한 이론적 분석

**(2) 노이즈 모델의 다양화**
- 현재 연구는 symmetric/asymmetric 노이즈에 집중
- 현실적 노이즈: **인스턴스 의존적(instance-dependent) 노이즈**, 클래스 불균형 노이즈
- 특히 의료 데이터에서의 **어노테이터 불일치(annotator disagreement)** 모델링

**(3) 대규모 모델과 데이터에서의 적용성**
- ELR은 CIFAR-급 벤치마크에서 검증되었으나 ImageNet 전체 규모에서의 효율성 검증 필요
- Foundation Model (CLIP, DINO 등) fine-tuning 시 노이즈 레이블 문제 대응

**(4) 계산 효율성**
- ELR+는 두 네트워크를 사용하여 메모리와 시간 비용 2배 증가
- 단일 네트워크로도 유사한 성능을 달성하는 경량화 방법 연구 필요

**(5) 자기지도 학습(Self-supervised learning)과의 결합**
- Pre-training을 통한 좋은 표현 학습 후 노이즈 레이블 fine-tuning
- Contrastive learning이 Early Learning 현상에 미치는 영향 분석

**(6) 멀티모달 노이즈 레이블**
- 텍스트-이미지, 음성-텍스트 쌍에서의 노이즈 레이블 문제
- ELR의 정규화 원리를 멀티모달 설정으로 확장

**(7) Early Learning 탐지의 자동화**
- 현재 Early Learning 단계의 종료 시점 $T$는 이론적으로 정의되나 실제 자동 탐지 메커니즘 부재
- 적응적(adaptive) 정규화 강도 조절 방법 연구

---

## 참고 자료

- **Liu, S., Niles-Weed, J., Razavian, N., & Fernandez-Granda, C. (2020).** "Early-Learning Regularization Prevents Memorization of Noisy Labels." *NeurIPS 2020.* arXiv:2007.00151v2

- **Li, J., Socher, R., & Hoi, S.C.H. (2020).** "DivideMix: Learning with Noisy Labels as Semi-supervised Learning." *ICLR 2020.*

- **Arpit, D. et al. (2017).** "A closer look at memorization in deep networks." *ICML 2017.*

- **Laine, S., & Aila, T. (2018).** "Temporal ensembling for semi-supervised learning." *ICLR 2018.*

- **Tarvainen, A., & Valpola, H. (2017).** "Mean teachers are better role models." *NeurIPS 2017.*

- **Zhang, H. et al. (2018).** "mixup: Beyond empirical risk minimization." *ICLR 2018.*

- **Han, B. et al. (2018).** "Co-teaching: Robust training of deep neural networks with extremely noisy labels." *NeurIPS 2018.*
