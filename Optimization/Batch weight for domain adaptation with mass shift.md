# Batch Weight for Domain Adaptation with Mass Shift

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

본 논문은 **비지도 도메인 전이(unsupervised domain transfer)** 에서 소스 도메인과 타겟 도메인 간의 **모드 질량 불균형(mode-mass imbalance)** 문제를 정의하고, 이를 해결하는 원칙적 방법론을 제안합니다.

기존 GAN 기반 도메인 전이 모델(CycleGAN, MUNIT 등)은 소스와 타겟 분포의 **모드 빈도가 동일하다고 암묵적으로 가정**합니다. 그러나 현실에서는 두 독립적으로 샘플링된 도메인 간에 클래스 빈도가 다를 수 있으며(예: MNIST의 균일한 숫자 분포 vs. SVHN의 불균일한 숫자 분포), 이 경우 기존 방법은 **의미론적 불일치(semantic mismatch)** 를 야기합니다.

### 주요 기여 (3가지)

| 기여 | 내용 |
|------|------|
| **①** 확률론적 형식화 | 도메인 전이에 대한 엄밀한 확률적 프레임워크 제시 |
| **②** Batch Weight 방법론 | 모드 질량 불균형 보정을 위한 훈련 샘플 재가중치 기법 |
| **③** Joint Discriminator (JD) | 픽셀 수준이 아닌 추상적·고수준의 사이클 일관성을 강제하는 새로운 아키텍처 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**모드 질량 불균형(Mode-Mass Imbalance):**

- 기존 GAN 기반 도메인 전이는 생성 분포 $\mathbb{P}_y^G$가 타겟 분포 $\mathbb{Q}_y$와 **정확히 일치**하도록 학습합니다.
- 그러나 "올바른" 전이 함수는 소스 도메인의 모드 빈도를 유지해야 하며, 이는 타겟 분포와 다를 수 있습니다.
- 예: MNIST(균일 분포) → SVHN(1이 약 20%) 전이 시, 올바른 전이는 생성된 샘플에서 1이 약 10%여야 하지만, GAN은 SVHN의 분포를 맞추려 해서 의미론적 불일치 발생.
- 잠재 변수(latent variable)를 통한 공유 의미론 모델링도 **확률 질량을 보존**하므로, 모드-질량 불균형이 있을 경우 근본적으로 유효하지 않음.

### 2.2 제안하는 방법

#### (1) 단일-도메인 배치 가중치 (One-sided Batch Weight)

Radon-Nikodym 도함수를 이용하여, 타겟 분포 기댓값을 생성 분포 기댓값으로 변환:

$$\mathbb{E}_{Y \sim \mathbb{Q}_y}[D(Y)] = \mathbb{E}_{X \sim \mathbb{P}_x}\left[D(G(X)) \frac{d\mathbb{Q}_y}{d\mathbb{P}_y^G}(G(X))\right] \tag{5}$$

미지의 Radon-Nikodym 도함수 $\frac{d\mathbb{Q}_y}{d\mathbb{P}_y^G}$를 신경망 $W$로 추정:

$$\inf_{W \in \mathcal{W}} \left(\mathbb{E}_{X \sim \mathbb{P}_x}[D(G(X)) \cdot W(X)] - \mathbb{E}_{Y \sim \mathbb{Q}_y}[D(Y)]\right)^2 \tag{6}$$

제약 조건: $\mathcal{W} = \{W : \mathbb{E}_{X \sim \mathbb{P}_x}[W(X)] = 1, W \geq 0\}$ → **softmax 레이어**로 구현

최종 Wasserstein 배치 가중 목적 함수:

$$\inf_{G,W} \sup_D \left(\mathbb{E}_{X \sim \mathbb{P}_x}[D(G(X)) \cdot W(X)] - \mathbb{E}_{Y \sim \mathbb{Q}_y}[D(Y)]\right)^2 \tag{7}$$

#### (2) 양-도메인 배치 가중치 (Two-sided Batch Weight) — JD-BW

두 결합 분포 $\mathbb{P}\_{xy}$, $\mathbb{Q}_{xy}$의 조건부 분포 동일성 가정:

$$\mathbb{P}_{y|x} = \mathbb{Q}_{y|x} \quad \text{and} \quad \mathbb{P}_{x|y} = \mathbb{Q}_{x|y} \tag{8}$$

혼합 분포 $\mathbb{M} = \frac{1}{2}(\mathbb{P}\_{xy} + \mathbb{Q}\_{xy}$ )에 대해 Radon-Nikodym 도함수 $w = \frac{d\mathbb{P}\_{xy}}{d\mathbb{Q}_{xy}}$ 를 이용:

$$w(x,y) = (v(x,y))^{-1}, \quad (x,y) \in \text{supp}\,\mathbb{P}_{xy} \tag{9}$$

$$\mathbb{P}_{xy} \frac{1}{2}(1 + w(X,Y)) = \mathbb{M} = \mathbb{Q}_{xy} \frac{1}{2}(1 + w^{-1}(X,Y)) \tag{10}$$

생성기를 통한 결합 분포 근사:

```math
\mathbb{P}_{xy} \approx \mathbb{P}_{xy}^G := (\text{id} \otimes G_{yx})\#\mathbb{P}_x
```

```math
\mathbb{Q}_{xy} \approx \mathbb{Q}_{xy}^G := (G_{xy} \otimes \text{id})\#\mathbb{Q}_y
```

Wasserstein 프레임워크에서의 완전한 목적 함수:

$$\inf_{G_{xy},G_{yx},W} \sup_D \left( \mathbb{E}_{X \sim \mathbb{P}_x} \frac{1}{2} D(X, G_{yx}(X)) \times (1 + W(X, G_{yx}(X))) \right.$$

$$\left. - \mathbb{E}_{Y \sim \mathbb{Q}_y} \frac{1}{2} D(G_{xy}(Y), Y) \times (1 + W(G_{xy}(Y), Y)^{-1}) \right) \tag{12}$$

가중치 네트워크 (합성 구조, 가장 안정적):

$$W_x : \mathcal{X} \to \mathbb{R}, \quad W_y : \mathcal{Y} \to \mathbb{R}$$

$$w_{\mathbf{x}} = \frac{1}{2}\left(\sigma(W_x(\mathbf{x})) + \sigma(-W_y(G_{xy}(\mathbf{x})))\right)$$

$$w_{\mathbf{y}} = \frac{1}{2}\left(\sigma(-W_x(G_{yx}(\mathbf{y}))) + \sigma(W_y(\mathbf{y}))\right)$$

### 2.3 모델 구조

```
전체 아키텍처: JD-BW (Joint Discriminator - Batch Weighted)
├── 생성기 (Generators)
│   ├── G_xy: X → Y (ResNet 기반, 노이즈 벡터 입력 포함)
│   └── G_yx: Y → X (ResNet 기반, 노이즈 벡터 입력 포함)
├── 결합 판별기 (Joint Discriminator D: X × Y → ℝ)
│   └── 각 레벨에서 개별 이미지 특징 + 연결(concat) 특징 계산
│       스펙트럴 정규화 + 그래디언트 페널티
└── 가중치 네트워크 (Weight Network W)
    ├── W_x: X → ℝ (DCGAN 판별기 구조)
    └── W_y: Y → ℝ (DCGAN 판별기 구조)
```

**핵심 설계 선택:**
- 생성기에 **ResNet** 사용 → 픽셀 수준 동일성(identity) 편향 부여
- **Joint Discriminator**: $\mathcal{X} \times \mathcal{Y}$ 공간에서 결합 분포를 직접 판별 → 픽셀 수준이 아닌 추상적 사이클 일관성 강제
- 소프트맥스 기반 배치 정규화로 가중치 제약 조건 만족

### 2.4 성능 향상

| 실험 | JD-BW (제안) | MUNIT (기준) |
|------|-------------|-------------|
| MNIST → SR-MNIST | ✅ 올바른 digit class 매칭 | ❌ zeros를 다른 digit으로 잘못 매칭 |
| MNIST → SVHN | ✅ 합리적 전이, 모드 붕괴 없음 | ❌ 반대 방향에서 완전한 모드 붕괴 |
| Edges → Shoes&Bags | ✅ 정확한 클래스 전이 | ❌ bag-edges → shoes로 잘못 전이 |
| CelebA → Portraits | ✅ 소스 특징 대부분 보존 | ⚠️ 포즈만 보존, 나머지는 노이즈 의존 |

**정량적 검증:** Figure 3에서 훈련 과정 중 batch weight가 클래스별로 점진적으로 조정되어 두 도메인의 모드 빈도가 일치함을 확인.

### 2.5 한계

1. **이미지 선명도**: Joint Discriminator는 픽셀 수준 재구성 품질을 직접 최적화하지 않으므로, MUNIT보다 생성 이미지가 덜 선명할 수 있음.
2. **훈련 불안정성**: 단일-도메인 가중치(Algorithm 2)는 가중치 네트워크가 배치 내 단일 샘플에 전체 가중치를 할당하는 실패 모드 존재.
3. **계산 비용**: 판별기 1 step당 생성기 업데이트 비율이 달라 (JD: 5:1 vs MUNIT: 1:1) 공정한 비교 어려움.
4. **비고유성(Non-uniqueness)**: 주어진 주변 분포에 대해 조건부를 만족하는 결합 분포가 여러 개 존재하므로, 생성기 아키텍처의 암묵적 편향에 의존.
5. **정량적 지표 부재**: FID 등 정량적 지표 없이 시각적 결과만으로 평가.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화 관련 이론적 기반

본 논문의 핵심 기여는 **분포 불일치에 강건한 도메인 전이**를 가능하게 한다는 점에서 일반화 성능과 직결됩니다.

**이론적 근거:** 기존 방법의 암묵적 가정인 $\mathbb{Q}_y = \mathbb{P}_y^G$ 대신:

$$\text{supp}\,\mathbb{Q}_y \subset \text{supp}\,\mathbb{P}_y \tag{2}$$

라는 **훨씬 약한 가정**만을 사용합니다. 이는 훨씬 넓은 범위의 도메인 쌍에 적용 가능함을 의미합니다.

### 3.2 일반화 성능 향상 메커니즘

**(a) 모드 커버리지 보장**

가중치 네트워크가 타겟 도메인에서 과소 표현된 모드에 높은 가중치를, 과대 표현된 모드에 낮은 가중치를 부여함으로써, 생성기가 **모든 의미론적 모드를 적절히 학습**합니다. 이는 새로운 데이터에 대한 일반화를 향상시킵니다.

**(b) 고수준 사이클 일관성**

```math
\text{Joint Discriminator}: D: \mathcal{X} \times \mathcal{Y} \to \mathbb{R}
```

픽셀 수준의 $L_1$ 사이클 일관성( $\|G_{yx}(G_{xy}(x)) - x\|\_1$ ) 대신, 결합 분포 $\mathbb{P}\_{xy}^G$와 $\mathbb{Q}_{xy}^G$ 사이의 분포 수준 일관성을 강제합니다. 이는 **추상적·의미론적 수준에서의 일관성**을 보장하여, 픽셀 수준 대응이 없는 복잡한 도메인에도 일반화 가능합니다.

**(c) 노이즈 항의 적절한 역할 분리**

- **MUNIT**: 노이즈 항이 클래스/모드 정보를 인코딩하는 실패 모드 발생
- **JD-BW**: 노이즈 항이 **도메인 고유 스타일**만을 인코딩, 소스 이미지의 의미론적 내용은 생성기가 보존

이 분리는 새로운 소스 이미지에 대해 더 예측 가능하고 일관된 전이를 가능하게 합니다.

**(d) 양-도메인 재가중의 안정성**

양-도메인 가중치의 대칭적 구조로 인해:
> "한 도메인에서 어떤 모드의 가중치가 낮아지면, 다른 도메인의 해당 모드는 높은 가중치를 받게 되어 결국 균형이 이루어진다."

이 메커니즘은 훈련 중 어떤 샘플도 완전히 배제되지 않도록 보장하며 ($\geq \frac{1}{2}$ 최소 가중치), **훈련 안정성과 일반화** 모두에 기여합니다.

### 3.3 일반화의 실증적 증거

- Figure 4 (MNIST → SVHN): 동일한 노이즈 벡터를 고정하고 다른 MNIST digit을 입력했을 때, 일관된 스타일의 SVHN 이미지 생성 → **구조적 일반화** 입증
- Figure 8 (CelebA → Portrait): 소스 이미지의 얼굴 특징을 광범위하게 보존 → **콘텐츠 일반화** 입증

---

## 4. 향후 연구에 미치는 영향 및 고려 사항

### 4.1 향후 연구에 미치는 영향

**(a) 분포 불일치 문제의 명시적 처리**

이 논문은 도메인 적응 연구에서 **모드-질량 불균형을 독립적인 연구 문제**로 정의하고 공식화했습니다. 향후 연구에서 도메인 전이 방법론을 설계할 때, 분포 불일치를 기본 가정으로 포함해야 함을 환기시킵니다.

**(b) 확률론적 도메인 전이 프레임워크**

결합 분포 $\mathbb{P}\_{xy}$, $\mathbb{Q}_{xy}$를 통한 수학적 형식화는, 도메인 전이를 **최적 운송(Optimal Transport)** 이론과 연결하는 연구의 기반을 제공합니다.

**(c) 중요도 가중치의 비지도 추정**

레이블 없이 Radon-Nikodym 도함수를 신경망으로 추정하는 접근법은 **공변량 이동(covariate shift)** 및 **레이블 이동(label shift)** 보정 연구에도 영감을 줍니다.

**(d) Joint Discriminator 패러다임**

결합 분포를 직접 판별하는 아이디어는 이후 **대조 학습(contrastive learning)** 기반 표현 학습과의 연결 가능성을 시사합니다.

### 4.2 향후 연구 시 고려할 점

#### (1) 가중치 네트워크의 안정적 훈련

배치 내 가중치 붕괴(한 샘플에 모든 가중치 집중) 문제를 해결하기 위한 더 강건한 정규화 방법 연구가 필요합니다. 예: 엔트로피 정규화 가중치 추정.

#### (2) 정량적 평가 지표 개발

모드-질량 불균형이 있는 도메인 전이를 평가하는 표준 지표가 부재합니다. **FID(Fréchet Inception Distance)** 는 타겟 분포와의 일치를 측정하므로, "올바른 전이"를 측정하는 데 부적합할 수 있습니다. **클래스-조건부 전이 정확도** 등 새로운 지표 개발이 필요합니다.

#### (3) 확장성 문제

고해상도(128×128 이상)에서 Joint Discriminator는 $\mathcal{X} \times \mathcal{Y}$ 공간에서 작동하므로 계산 비용이 급증합니다. **효율적인 크로스-도메인 주의 메커니즘(cross-domain attention)** 등을 통한 확장 방법 연구가 필요합니다.

#### (4) 동적 모드 불균형 처리

본 논문은 정적인 모드 불균형을 가정하지만, 실제 스트리밍 데이터에서는 분포가 시간에 따라 변할 수 있습니다. **온라인 배치 가중치 업데이트** 방법론으로의 확장이 필요합니다.

#### (5) 멀티-도메인 전이로의 확장

본 방법론은 두 도메인 간의 전이를 가정합니다. 세 개 이상의 도메인 간 전이 시 결합 분포의 정의와 가중치 네트워크의 설계를 재고해야 합니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의**: 이하의 내용은 제가 학습한 데이터를 기반으로 한 분석이며, 논문 PDF에 직접 인용된 내용이 아닙니다. 개별 논문의 세부 수치는 원문 확인을 권장합니다.

### 5.1 관련 연구 흐름

| 연구 | 핵심 아이디어 | 본 논문과의 관계 |
|------|-------------|----------------|
| **DRIT++** (Lee et al., 2020, IJCV) | 다양한 도메인 간 이미지 번역, 콘텐츠/스타일 분리 | 모드 불균형 미처리 |
| **StarGAN v2** (Choi et al., 2020, CVPR) | 다중 도메인 스타일 전이 | 분포 불일치 비고려 |
| **CUT** (Park et al., 2020, ECCV) | Contrastive Unpaired Translation, 단방향 전이 | 효율적이나 질량 불균형 미처리 |
| **Diffusion-based I2I** (2022~) | 확산 모델 기반 도메인 전이 | 모드 커버리지 개선 가능성 있으나 불균형 명시 처리 부재 |
| **EGSDE** (Zhao et al., 2022) | 에너지 기반 확산 도메인 전이 | 유연한 분포 표현 |

### 5.2 본 논문의 차별적 지위

```
기존 연구 패러다임:
  P_x → G → P_y^G ≈ Q_y (강한 가정)

본 논문 패러다임:
  P_x → G (weighted) → P_y^G
  such that supp(Q_y) ⊂ supp(P_y^G) (약한 가정)
  
2020년 이후 주류:
  - Diffusion 기반: 분포 표현력 향상 ↑, 불균형 처리 ✗
  - Contrastive 기반: 효율성 ↑, 불균형 처리 ✗
  - 본 논문의 핵심 문제의식은 여전히 미해결 상태
```

### 5.3 향후 연결 방향

본 논문의 **배치 가중치** 아이디어는 다음과 같은 최신 연구 방향과 결합 가능합니다:

1. **확산 모델 + 배치 가중치**: 확산 기반 도메인 전이의 모드 불균형 보정
2. **자기 지도 학습 + 중요도 가중치**: 레이블 없이 도메인 불변 표현 학습
3. **페더레이션 러닝**: 클라이언트 간 분포 불일치 보정에 배치 가중치 적용

---

## 참고 자료

- **주요 논문**: Bińkowski, M., Hjelm, R. D., & Courville, A. (2019). *Batch weight for domain adaptation with mass shift*. arXiv:1905.12760v1.
- **논문 내 인용 참고문헌**:
  - Goodfellow et al. (2014). *Generative Adversarial Nets*. NeurIPS.
  - Zhu et al. (2017). *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks*. arXiv:1703.10593. (CycleGAN)
  - Huang et al. (2018). *Multimodal Unsupervised Image-to-Image Translation*. ECCV. (MUNIT)
  - Arjovsky et al. (2017). *Wasserstein GAN*. arXiv:1701.07875.
  - Gulrajani et al. (2017). *Improved Training of Wasserstein GANs*. arXiv:1704.00028.
  - Cohen et al. (2018). *Distribution Matching Losses Can Hallucinate Features in Medical Image Translation*. MICCAI.
  - Diesendruck et al. (2018). *Importance Weighted Generative Networks*. arXiv:1806.02512.
  - He et al. (2015). *Deep Residual Learning for Image Recognition*. arXiv:1512.03385.
  - Miyato et al. (2018). *Spectral Normalization for Generative Adversarial Networks*. arXiv:1802.05957.
  - Dumoulin et al. (2016). *Adversarially Learned Inference*. arXiv:1606.00704. (ALI)
