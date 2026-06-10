# Beyond Synthetic Noise: Deep Learning on Controlled Noisy Labels

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

이 논문의 핵심 주장은 **기존 연구가 합성(synthetic) 노이즈 레이블에만 의존해왔으며, 이는 실제 웹 레이블 노이즈(real-world web label noise)의 특성을 제대로 반영하지 못한다**는 것입니다. 합성 노이즈(blue noise)와 실제 웹 노이즈(red noise)는 근본적으로 다른 특성을 가지므로, 합성 노이즈 기반의 연구 결과가 실제 환경에 그대로 적용되지 않을 수 있습니다.

### 세 가지 주요 기여

| 기여 | 내용 |
|------|------|
| **벤치마크 구축** | 최초의 통제된(controlled) 실제 웹 레이블 노이즈 벤치마크 구축 (212,588 이미지, 약 800,000 어노테이션) |
| **새로운 방법론** | MentorMix: MentorNet + Mixup을 결합한 효과적인 노이즈 극복 방법 |
| **대규모 실증 연구** | 다양한 노이즈 레벨/타입/아키텍처/학습 설정에 걸친 최대 규모의 DNN 연구 |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

**문제 1: 통제된 실제 노이즈 벤치마크의 부재**

기존 데이터셋의 한계:
- **합성 노이즈 데이터셋** (CIFAR 등): 인위적 분포에서 생성되어 실제 노이즈를 반영하지 못함
- **비통제 웹 노이즈 데이터셋** (WebVision, Clothing1M): 노이즈 레벨이 고정되고 알 수 없어 통제 실험 불가

**문제 2: 합성 노이즈 연구 결과의 일반화 불확실성**

Zhang et al. (2017)은 DNN이 noisy label에서 일반화를 잘 못한다고 했으나, Rolnick et al. (2017)은 노이즈 분포를 약간 바꾸면 DNN이 robust하다는 상반된 결과를 제시했습니다. 이러한 모순은 합성 노이즈의 인위성에서 기인합니다.

---

### 2.2 데이터셋 구성

**Blue Noise (합성 노이즈)**: 대칭 레이블 플리핑(symmetric label flipping)

노이즈 레벨 $p$에서 각 학습 예제 $(x_i, y_i)$의 레이블을 다른 클래스로 무작위 변경:

$$P(\tilde{y}_i \neq y_i) = p, \quad \tilde{y}_i \sim \text{Uniform}(\{1,\ldots,m\} \setminus \{y_i\})$$

**Red Noise (웹 노이즈)**: 잘못 레이블된 웹 이미지로 clean 이미지를 대체

- Google Image Search (text-to-image: 82%, image-to-image: 18%)로 수집
- 3-5명의 워커가 바이너리 어노테이션 ("레이블이 이미지에 맞는가?")
- 다수결 투표로 최종 레이블 결정

**데이터셋 규모**:

| 데이터셋 | 클래스 수 | 노이즈 레벨 |
|----------|----------|------------|
| Red/Blue Mini-ImageNet | 100 | 0, 5, 10, 15, 20, 30, 40, 50, 60, 80% |
| Red/Blue Stanford Cars | 196 | 0, 5, 10, 15, 20, 30, 40, 50, 60, 80% |

**Blue vs Red 노이즈의 핵심 차이**:

| 차이점 | Blue Noise | Red Noise |
|--------|-----------|-----------|
| 시각적/의미적 유사도 | 낮음 | 높음 |
| 노이즈 레벨 | 클래스 레벨 | 인스턴스 레벨 |
| 어휘 | 고정된 클래스 어휘 | 개방형 어휘 |

---

### 2.3 제안 방법: MentorMix

#### 배경: MentorNet 목적 함수

MentorNet (Jiang et al., 2018)은 다음 목적함수를 최소화합니다:

$$\boldsymbol{w}^* = \underset{\boldsymbol{w} \in \mathbb{R}^d, \mathbf{v} \in [0,1]^n}{\arg\min} \mathbb{F}(\mathbf{v}, \boldsymbol{w}) = \frac{1}{n}\sum_{i=1}^{n} v_i \ell(g_s(\mathbf{x}_i; \boldsymbol{w}), y_i) + \theta\|\boldsymbol{w}\|_2^2 + G(\mathbf{v}; \gamma) $$

여기서:
- $v_i \in [0,1]$: 각 학습 예제에 대한 잠재적 가중치
- $G(\mathbf{v}; \gamma)$: 커리큘럼/가중치 스킴을 결정하는 정규화 항
- $\theta$: L2 정규화 파라미터

#### 배경: Mixup 목적 함수

Mixup (Zhang et al., 2018)은 경험적 vicinal risk를 최소화합니다:

$$\boldsymbol{w}^* = \underset{\boldsymbol{w}}{\arg\min} \frac{1}{n}\sum_{i=1}^{n}\frac{1}{n}\sum_{j=1}^{n} \mathbb{E}_\lambda[\ell(g_s(\tilde{\mathbf{x}}_{ij}; \boldsymbol{w}), \tilde{\mathbf{y}}_{ij})] $$

혼합 샘플은 다음과 같이 계산됩니다:

$$\tilde{\mathbf{x}}_{ij} = \lambda \mathbf{x}_i + (1-\lambda)\mathbf{x}_j $$

$$\tilde{\mathbf{y}}_{ij} = \lambda \mathbf{y}_i + (1-\lambda)\mathbf{y}_j $$

여기서 $\lambda \sim \text{Beta}(\alpha, \alpha)$.

#### MentorMix 핵심 수식

Self-paced regularizer $G(\mathbf{v}) = -\gamma\|\mathbf{v}\|_1$ (Kumar et al., 2010)을 사용하여, Mixup과 MentorNet을 결합한 목적함수:

$$\mathbb{F}(\tilde{\mathbf{v}}, \boldsymbol{w}) = \frac{1}{n^2}\sum_{i=1}^{n}\sum_{j=1}^{n}\mathbb{E}_\lambda[\tilde{v}_{ij}\ell(\tilde{\mathbf{x}}_{ij}, \tilde{\mathbf{y}}_{ij}) - \gamma\tilde{v}_{ij}] $$

$\boldsymbol{w}$가 고정될 때 최적 가중치의 닫힌 형태(closed-form) 해:

$$\tilde{v}^*_{ij} = \underset{\tilde{\mathbf{v}} \in [0,1]^{n \times n}}{\arg\min} \mathbb{F}_{\boldsymbol{w}}(\tilde{\mathbf{v}}) = \mathbf{1}(\ell(\tilde{\mathbf{x}}_{ij}, \tilde{\mathbf{y}}_{ij}) \leq \gamma) $$

중요도 샘플링을 위한 밀도 함수:

```math
P_{\mathbf{v}}(v_i = 1 | \mathbf{x}_i, y_i) = \frac{\exp(v^*_i / t)}{\sum_{j=1}^{n}\exp(v^*_j / t)}
```

Eq.(6)의 이진값 특성을 활용하여 Eq.(5)의 일부를 재작성하면:

$$\sum_{(\mathbf{x}_i, y_i) \sim \mathcal{D}} \sum_{(\mathbf{x}_j, y_j) \sim P_\mathbf{v}} \mathbb{E}_\lambda[\ell(\tilde{\mathbf{x}}_{ij}, \tilde{\mathbf{y}}_{ij})] - \gamma $$

훈련 시 상수 $\gamma$는 제거되며, **더 낮은 손실을 가진 예제가 mixup에 더 많이 사용**됩니다.

#### Algorithm 1 요약 (MentorMix)

```
Input: 미니배치 D_m, 하이퍼파라미터 γ_p, α
1. 각 (x_i, y_i)에 대해 ℓ(x_i, y_i) 계산
2. γ_p-번째 백분위수 손실 ℓ_p(D_m) 설정
3. γ ← EMA(ℓ_p(D_m))  // 지수이동평균 업데이트
4. v*_i ← MentorNet(ℓ(x_i, y_i), γ)  // 가중치 계산
5. P_v = softmax(v*)  // 샘플링 분포
6. 각 (x_i, y_i)에 대해:
   - (x_j, y_j)를 P_v에서 복원 추출
   - λ ← Beta(α, α)
   - λ ← v*_i·max(λ, 1-λ) + (1-v*_i)·min(λ, 1-λ)  // 안정화
   - x̃_ij ← λx_i + (1-λ)x_j
   - ỹ_ij ← λy_i + (1-λ)y_j
   - ℓ_i = ℓ(x̃_ij, ỹ_ij)
   - (선택) 별도 MentorNet으로 ℓ_i 재가중
7. return (1/|D_m|) Σℓ_i
```

#### MentorMix의 핵심 아이디어

- **MentorNet**: "깨끗한" 레이블을 가진 예제 식별 (낮은 손실 → 높은 가중치)
- **Mixup**: 식별된 깨끗한 예제들을 우선적으로 사용하여 경험적 vicinal risk 최소화
- **두 방법의 상보적 결합**: MentorNet이 노이즈가 적은 예제를 선별하고, Mixup이 그 예제들로 강건한 표현을 학습

---

### 2.4 모델 구조

논문은 새로운 백본 아키텍처를 제안하지 않고, 기존 아키텍처를 활용합니다:

- **제안 데이터셋 실험**: Inception-ResNet-v2 (기본), EfficientNet-B5, MobileNet-V2, ResNet-50/101, Inception-V2/V3
- **CIFAR 실험**: ResNet-32
- **WebVision 실험**: Inception-ResNet-v2

| 아키텍처 | 파라미터 수 | 이미지 크기 | ImageNet Top-1 |
|---------|-----------|------------|----------------|
| EfficientNet-B5 | 28.3M | 456 | 83.3% |
| Inception-ResNet-V2 | 54.2M | 299 | 80.3% |
| ResNet-101 | 42.5M | 224 | 77.5% |
| MobileNet-V2 | 2.2M | 224 | 71.6% |

---

### 2.5 성능 향상

#### 제안 데이터셋에서의 성능 (평균, 10개 노이즈 레벨)

| 방법 | Mini-ImageNet (Fine-tuned Blue) | Mini-ImageNet (Scratch Blue) | Stanford Cars (Fine-tuned Red) |
|------|--------------------------------|------------------------------|-------------------------------|
| Vanilla | 82.3±1.9 | 58.3±10.3 | 82.4±6.9 |
| MentorNet | 82.9±1.7 | 61.8±10.3 | 82.6±6.6 |
| Mixup | 81.7±1.8 | 60.7±9.8 | 85.0±6.2 |
| **MentorMix** | **84.2±0.7** | **70.9±3.4** | **86.9±5.5** |

특히 **Scratch Blue Mini-ImageNet에서 70.9%로 Vanilla 대비 +12.6%p** 개선이 두드러집니다.

#### CIFAR 공개 벤치마크 성능

**CIFAR-100** (노이즈 레벨별 정확도):

| 방법 | 20% | 40% | 60% | 80% |
|------|-----|-----|-----|-----|
| MentorNet (2018) | 73.5 | 68.5 | 61.2 | 35.5 |
| Mixup (2018) | 73.9 | 66.8 | 58.8 | 40.1 |
| **MentorMix** | **78.6** | **71.3** | **64.6** | **41.2** |

**CIFAR-10** (노이즈 레벨별 정확도):

| 방법 | 20% | 40% | 60% | 80% |
|------|-----|-----|-----|-----|
| Mixup (2018) | 94.0 | 91.5 | 86.8 | 76.9 |
| **MentorMix** | **95.6** | **94.2** | **91.3** | **81.0** |

#### WebVision 1.0 성능 (ILSVRC12 validation)

| 방법 | ILSVRC12 Top-1 | WebVision Top-1 |
|------|---------------|----------------|
| Vanilla | 61.7 | 70.9 |
| MentorNet (extra clean labels) | 64.2 | 72.6 |
| **MentorMix (no extra labels)** | **67.5** | **74.3** |

→ 기존 최고 성능 대비 **약 3% top-1 정확도 향상** (extra 클린 레이블 없이)

---

### 2.6 한계

논문에서 명시적으로 언급되거나 유추할 수 있는 한계:

1. **계산 비용**: 수십만 V100 GPU 시간이 소요되는 대규모 실험이 필요하며, 일반 연구자가 재현하기 어려울 수 있습니다.
2. **데이터셋 규모**: Red Mini-ImageNet이 50K로 제한되어 일부 희귀 클래스에서 잘못 레이블된 이미지 수집이 어렵습니다.
3. **도메인 한정성**: 이미지 분류에 집중되어 있어 NLP, 음성 등 다른 도메인으로의 직접 적용 가능성이 불분명합니다.
4. **하이퍼파라미터 민감도**: MentorMix도 $\gamma_p$, $\alpha$ 등의 하이퍼파라미터 튜닝이 필요합니다.
5. **클린 레이블 미사용**: clean label이 전혀 없는 세팅을 가정하나, 일부 클린 데이터 활용 시 추가 개선 가능성이 있습니다.

---

## 3. 모델의 일반화 성능 향상 가능성

이 논문은 일반화 성능과 관련하여 세 가지 중요한 발견을 제시합니다.

### 발견 1: DNN은 Red Noise에서 훨씬 더 잘 일반화

논문은 Blue Noise와 Red Noise에서의 최종 정확도 표준편차를 비교하여 일반화 성능의 차이를 보여줍니다:

| 노이즈 설정 | Mini-ImageNet (Fine-tuned) | Stanford Cars (Scratch) |
|------------|--------------------------|------------------------|
| Blue Noise | 0.205 | 0.347 |
| Red Noise (setting 0) | 0.051 | 0.146 |

Red Noise의 표준편차가 Blue Noise의 **약 1/2 ~ 1/4 수준**으로, DNN이 웹 레이블 노이즈에 더 강건함을 보여줍니다. 저자들은 이를 웹 이미지가 시각적/의미적으로 실제 클래스와 더 관련성이 높기 때문으로 가설화합니다.

이는 다음과 같은 **일반화 성능 향상 가능성**을 시사합니다:
- 실제 웹 크롤링 데이터를 활용할 때, 합성 노이즈를 가정한 방법보다 더 나은 일반화를 기대할 수 있음
- 웹 이미지 기반의 약지도 학습(weakly supervised learning)이 생각보다 효과적일 수 있음

### 발견 2: Red Noise에서 DNN이 패턴을 먼저 학습하지 않을 수 있음

Arpit et al. (2017)은 DNN이 초기 학습 단계에서 일반화 가능한 패턴을 먼저 학습한다고 주장했습니다. 이는 peak 정확도와 final 정확도의 차이(drop)로 측정됩니다:

$$\text{Drop} = \text{Peak Accuracy} - \text{Final Accuracy}$$

실험 결과, Blue Noise에서는 노이즈 레벨이 높아질수록 drop이 크게 증가하지만, **Red Noise에서의 drop은 현저히 작고 Stanford Cars에서는 거의 0에 가깝습니다.**

이 발견은 **일반화 관점에서 중요한 함의**를 가집니다:
- Red Noise에서는 조기 종료(early stopping)가 일반화 성능 향상에 효과적이지 않을 수 있음
- 웹 노이즈는 더 복잡한 패턴 구조를 가져 DNN이 일반화 패턴 학습에 더 많은 시간이 필요함

### 발견 3: ImageNet 사전훈련 아키텍처가 노이즈 레이블에서도 일반화

Kornblith et al. (2019)의 발견을 노이즈 데이터로 확장합니다. 7개 아키텍처에서 ImageNet 정확도와 노이즈 데이터 fine-tuning 정확도 간의 Pearson 상관계수:

| 데이터셋 | Red Noise | Blue Noise |
|---------|----------|-----------|
| Mini-ImageNet | $r = 0.91$ | $r = 0.90$ |
| Stanford Cars | $r = 0.88$ | $r = 0.85$ |

**핵심 시사점**: 더 좋은 사전훈련 모델이 노이즈 레이블에서도 더 나은 일반화를 보입니다. 이는 **전이 학습(transfer learning)이 노이즈 레이블 문제에서 강력한 일반화 향상 도구**임을 의미합니다.

### MentorMix의 일반화 성능 향상 메커니즘

MentorMix가 일반화를 향상시키는 이유:

1. **Vicinal Risk 최소화**: 단순 경험적 위험이 아닌 vicinal risk를 최소화함으로써, 훈련 분포 인근의 공간에서도 낮은 손실을 달성하여 일반화 향상

2. **커리큘럼 학습**: 초기에는 쉬운(손실이 낮은) 예제부터 학습하여 노이즈 레이블에 과적합되기 전에 의미 있는 특징을 학습

3. **Mixed-up 예제의 정규화 효과**: Mixup이 생성하는 보간된 샘플이 암묵적 정규화(implicit regularization) 역할을 수행

---

## 4. 앞으로의 연구에 미치는 영향 및 고려사항

### 4.1 앞으로의 연구에 미치는 영향

**① 벤치마크로서의 영향**

이 논문이 구축한 통제된 웹 레이블 노이즈 벤치마크는 이후 연구들이 합성 노이즈뿐만 아니라 실제 노이즈에서도 방법론을 검증해야 하는 표준을 제시했습니다. 새로운 robust learning 방법들은 이제 red noise 설정에서도 성능을 증명해야 하는 요구사항이 생겼습니다.

**② 방법론적 패러다임 변화**

- 합성 노이즈에서 잘 작동하는 방법이 실제 노이즈에서는 그렇지 않을 수 있다는 경고는, 이후 연구자들이 더 **다양한 노이즈 타입**에서 방법을 검증하도록 촉구합니다.
- MentorMix의 curriculum learning + vicinal risk minimization 결합 아이디어는 이후 다양한 변형 연구의 기반이 됩니다.

**③ 전이 학습의 재조명**

더 좋은 사전훈련 모델이 노이즈에도 강건하다는 발견은, 이후 **대규모 사전훈련 모델(CLIP, ViT 등)을 노이즈 레이블 문제에 적용하는 연구**에 이론적 근거를 제공합니다.

**④ 실용적 권고사항의 영향**

논문의 실용적 권고사항들 (사전훈련 모델 사용, early stopping의 제한적 효과 등)은 이후 실무에서 noisy label 문제를 다루는 엔지니어들에게 가이드라인을 제공합니다.

---

### 4.2 2020년 이후 관련 최신 연구 비교 분석

#### DivideMix (Li et al., ICLR 2020)

이 논문과 동시기 연구로, noisy label을 semi-supervised learning 문제로 접근합니다.

- **접근 방식**: GMM으로 클린/노이즈 샘플을 분리하고, MixMatch를 사용
- **MentorMix와의 차이**: DivideMix는 클린 레이블 없이도 clean/noisy 분리를 수행하나, 두 개의 네트워크(co-training)가 필요하여 복잡도가 높음
- **성능**: WebVision mini에서 DivideMix가 MentorMix보다 높은 성능을 달성

$$\mathcal{L} = \mathcal{L}_X + \lambda_u \mathcal{L}_U$$

#### ELR (Early-Learning Regularization, Liu et al., NeurIPS 2020)

조기 학습 단계의 특성을 활용한 정규화 방법:

$$\ell_{ELR}(\theta) = \frac{1}{n}\sum_i \left[\ell_{CE}(f(x_i;\theta), \tilde{y}_i) - \lambda \log(1 - f(x_i;\theta) \cdot t_i)\right]$$

여기서 $t_i$는 모델 예측의 지수이동평균입니다.

이 논문이 제시한 "DNN이 초기에 패턴을 먼저 학습"한다는 발견을 활용합니다.

#### CORES² (Cheng et al., ICML 2021)

Confidence regularized self-training으로 noisy label을 다루며, 본 논문의 벤치마크에서 평가되는 후속 연구입니다.

#### SOP (Liu et al., ICML 2022)

Sample-dependent과 class-dependent noise를 동시에 처리하는 방법으로, 본 논문의 instance-level noise 특성 발견에서 영감을 받았습니다.

#### Noisy Labels with Foundation Models (2022-2024)

CLIP, DINOv2 등 대규모 사전훈련 모델을 활용한 연구들은 본 논문의 "더 나은 사전훈련 모델이 노이즈에서도 일반화"라는 발견의 자연스러운 확장입니다.

| 연구 | 주요 아이디어 | 본 논문과의 관계 |
|------|-------------|----------------|
| DivideMix (ICLR 2020) | GMM + MixMatch semi-supervised | MentorMix보다 높은 성능, 더 복잡한 파이프라인 |
| ELR (NeurIPS 2020) | 조기 학습 단계 활용 정규화 | 본 논문의 패턴 학습 발견 활용 |
| CORES² (ICML 2021) | 신뢰도 기반 자기훈련 | 본 논문 벤치마크에서 평가 |
| ProMix (2023) | 프롬프트 기반 + foundation model | 본 논문의 사전훈련 발견 확장 |

---

### 4.3 앞으로 연구 시 고려할 점

**① 노이즈 타입의 다양성 고려**

- 합성 노이즈와 웹 노이즈 모두에서 검증하는 것이 필수화되었습니다.
- Instance-dependent noise, asymmetric noise, open-set noise 등 다양한 노이즈 타입 고려 필요
- 특히 Red Noise의 **인스턴스 레벨** 특성을 고려한 방법론 설계 필요

**② 사전훈련 모델의 적극 활용**

- 이 논문이 보인 사전훈련 모델과 노이즈 강건성의 높은 상관관계($r > 0.88$)를 기반으로, CLIP, ViT, DINOv2 등 최신 대규모 사전훈련 모델을 노이즈 레이블 설정에 적용하는 연구가 필요합니다.
- Fine-tuning 전략(full fine-tuning vs. linear probing vs. adapter)에 따른 노이즈 강건성 차이 분석 필요

**③ Early Stopping 전략의 재검토**

- Red Noise에서 peak-final accuracy drop이 작다는 발견은, 실제 웹 크롤링 데이터에서는 **early stopping보다 충분한 훈련이 더 중요**할 수 있음을 시사합니다.
- 노이즈 타입에 따른 최적 학습 스케줄 연구 필요

**④ 평가 지표의 다양화**

- Peak accuracy 외에도 final accuracy, stability across noise levels 등 다양한 지표를 함께 고려해야 합니다.
- 실제 응용에서는 특정 노이즈 레벨에서의 성능보다 **다양한 노이즈 레벨에 걸친 안정적 성능**이 중요할 수 있습니다.

**⑤ 계산 효율성과 실용성**

- 이 논문의 실험은 수십만 V100 GPU 시간이 필요하여 재현이 어렵습니다. 향후 연구는 더 효율적인 방법론 개발이 필요합니다.
- 하이퍼파라미터에 덜 민감한 방법론 개발이 중요합니다.

**⑥ 다양한 태스크로의 확장**

- 현재 이미지 분류에 집중된 연구를 **객체 탐지, 세그멘테이션, NLP, 멀티모달** 등으로 확장할 때, 각 도메인에서의 실제 노이즈 특성 파악이 선행되어야 합니다.

**⑦ 레이블 품질 추정의 통합**

- Confident Learning (Northcutt et al., 2019/2021) 등 레이블 품질 추정 방법과 robust learning의 통합이 유망합니다.

---

## 참고자료

**주 논문**
- Lu Jiang, Di Huang, Mason Liu, Weilong Yang. "Beyond Synthetic Noise: Deep Learning on Controlled Noisy Labels." *ICML 2020*. arXiv:1911.09781v3

**논문 내 인용 문헌**
- Zhang et al. "Understanding deep learning requires rethinking generalization." *ICLR 2017*
- Arpit et al. "A closer look at memorization in deep networks." *ICML 2017*
- Jiang et al. "MentorNet: Learning data-driven curriculum for very deep neural networks on corrupted labels." *ICML 2018*
- Zhang et al. "Mixup: Beyond empirical risk minimization." *ICLR 2018*
- Kornblith et al. "Do better ImageNet models transfer better?" *CVPR 2019*
- Kumar et al. "Self-paced learning for latent variable models." *NeurIPS 2010*
- Li et al. "WebVision database: Visual learning and understanding from web data." *arXiv 2017*
- Rolnick et al. "Deep learning is robust to massive label noise." *arXiv 2017*
- Northcutt et al. "Confident learning: Estimating uncertainty in dataset labels." *arXiv 2019*

**2020년 이후 비교 연구**
- Li et al. "DivideMix: Learning with noisy labels as semi-supervised learning." *ICLR 2020*
- Liu et al. "Early-Learning Regularization Prevents Memorization of Noisy Labels." *NeurIPS 2020*
- Cheng et al. "Learning with Instance-Dependent Label Noise: A Sample Sieve Approach." *ICLR 2021*
