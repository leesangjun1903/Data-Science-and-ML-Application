# Learning Domain Adaptive Features with Unlabeled Domain Bridges

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

본 논문의 핵심 주장은 다음과 같습니다: **소스 도메인과 타겟 도메인 간의 격차(domain gap)가 매우 클 때, 기존의 도메인 적응(Domain Adaptation) 방법들은 한계를 보이며, 이를 해결하기 위해 레이블이 없는 중간 브리지 도메인(unlabeled bridge domain)을 활용하는 새로운 패러다임이 필요하다.**

### 주요 기여 (3가지)

| 기여 | 설명 |
|------|------|
| **(1) 새로운 학습 패러다임 제안** | 소스-타겟 도메인 간 격차가 매우 큰 상황에서의 도메인 적응 문제 정의 |
| **(2) CFGAN (CycleFlow GAN)** | 브리지 도메인을 활용한 이미지-이미지 변환 프레임워크 제안 |
| **(3) PADA (Prototypical Adversarial Domain Adaptation)** | 브리지 도메인을 활용한 비지도 도메인 적응 모델 제안 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존 방법론의 한계:

- **CycleGAN** 등의 이미지 변환 모델: 소스-타겟 도메인이 인접해 있다는 가정에 의존
- **UDA (Unsupervised Domain Adaptation)**: MMD, 적대적 훈련 기반 방법들도 도메인 간 거리가 클 때 성능 저하
- **부정적 전이(Negative Transfer)**: 두 도메인이 이질적일수록 클래스와 무관한 특징이 전이되어 성능 저하

**도메인 격차 측정**: 논문은 $\mathcal{H}\Delta\mathcal{H}$ 발산을 사용하여 도메인 간 거리를 정의합니다:

$$\text{dist}(\mathcal{D}_s, \mathcal{D}_t) = d_{\mathcal{H}\Delta\mathcal{H}}(\mathcal{D}_s, \mathcal{D}_t)$$

$$\hat{d}_{\mathcal{A}} = 2(1 - 2\epsilon)$$

여기서 $\epsilon$은 소스/타겟을 구분하는 이진 분류기의 일반화 오류입니다.

브리지 도메인 $\mathcal{D}_b$ 선택 조건:

$$\text{dist}(\mathcal{D}_s, \mathcal{D}_b) < \text{dist}(\mathcal{D}_s, \mathcal{D}_t)$$
$$\text{dist}(\mathcal{D}_b, \mathcal{D}_t) < \text{dist}(\mathcal{D}_s, \mathcal{D}_t)$$

---

### 2.2 제안하는 방법 및 수식

#### **[방법 1] CFGAN (CycleFlow GAN)**

기존 CycleGAN의 손실 함수:

$$\mathcal{L}_{\mathcal{T}_{adv}}(G, D_{\mathcal{T}}, \mathcal{S}, \mathcal{T}) = \mathbb{E}_{z \sim p_{\text{data}}(\mathcal{T})}[\log D_{\mathcal{T}}(z)] + \mathbb{E}_{x \sim p_{\text{data}}(\mathcal{S})}[\log(1 - D_{\mathcal{T}}(G(x)))]$$

$$\mathcal{L}_{\text{cyc}}(G, F) = \mathbb{E}_{x \sim p_{\text{data}}(\mathcal{S})}[\|F(G(x)) - x\|_1] + \mathbb{E}_{z \sim p_{\text{data}}(\mathcal{T})}[\|G(F(z)) - z\|_1]$$

CFGAN의 적대적 손실 함수 (브리지 도메인 $\mathcal{B}$ 추가):

$$\mathcal{L}_{\mathcal{T}_{adv}}(G_{\mathcal{S}\to\mathcal{B}}, D_{\mathcal{B}}, G_{\mathcal{B}\to\mathcal{T}}, D_{\mathcal{T}}, \mathcal{S}, \mathcal{B}, \mathcal{T}) =$$
$$\mathbb{E}_{y \sim p_{\text{data}}(\mathcal{B})}[\log D_{\mathcal{B}}(y)] + \lambda\mathbb{E}_{z \sim p_{\text{data}}(\mathcal{T})}[\log D_{\mathcal{T}}(z)]$$
$$+ \mathbb{E}_{x \sim p_{\text{data}}(\mathcal{S})}[\log(1 - D_{\mathcal{B}}(G_{\mathcal{S}\to\mathcal{B}}(x)))]$$
$$+ \lambda\mathbb{E}_{y \sim p_{\text{data}}(\mathcal{B})}[\log(1 - D_{\mathcal{T}}(G_{\mathcal{B}\to\mathcal{T}}(G_{\mathcal{S}\to\mathcal{B}}(x))))]$$

CFGAN의 사이클 일관성 손실:

$$\mathcal{L}_{\text{cyc}}(G_{\mathcal{S}\to\mathcal{B}}, G_{\mathcal{B}\to\mathcal{T}}, F_{\mathcal{T}\to\mathcal{B}}, F_{\mathcal{B}\to\mathcal{S}}) =$$
$$\mathbb{E}_{y \sim p_{\text{data}}(\mathcal{B})}[\|G_{\mathcal{S}\to\mathcal{B}}(F_{\mathcal{B}\to\mathcal{S}}(y)) - y\|_1]$$
$$+ \mathbb{E}_{x \sim p_{\text{data}}(\mathcal{S})}[\|F_{\mathcal{B}\to\mathcal{S}}(G_{\mathcal{S}\to\mathcal{B}}(x)) - x\|_1]$$
$$+ \lambda\mathbb{E}_{z \sim p_{\text{data}}(\mathcal{T})}[\|G_{\mathcal{B}\to\mathcal{T}}(F_{\mathcal{T}\to\mathcal{B}}(z)) - z\|_1]$$
$$+ \lambda\mathbb{E}_{x \sim p_{\text{data}}(\mathcal{S})}[\|F_{\mathcal{T}\to\mathcal{B}}(G_{\mathcal{B}\to\mathcal{T}}(G_{\mathcal{S}\to\mathcal{B}}(x))) - G_{\mathcal{S}\to\mathcal{B}}(x)\|_1]$$

---

#### **[방법 2] PADA (Prototypical Adversarial Domain Adaptation)**

PADA는 세 가지 구성요소로 이루어집니다:

**(a) 태스크 손실 (Cross-Entropy):**

$$\mathcal{L}_{ce} = -\mathbb{E}_{(x_s, y_s) \sim \hat{\mathcal{D}}_s} \sum_{k=1}^{K} \mathbf{1}[k = y_s] \log(C(f_G))$$

**(b) 적대적 도메인 정렬 손실 (ADA):**

$$\mathcal{L}_{DI} = -\mathbb{E}[l_f \log P(l_f)] + \mathbb{E}(1 - l_f)[\log P(1 - l_f)]$$

**(c) 프로토타입 네트워크 (Prototypical Matching Network):**

각 클래스의 프로토타입 계산:

$$\mathbf{c}_k = \frac{1}{|S_k|} \sum_{(x_i, y_i) \in S_k} f_\phi(x_i)$$

쿼리 포인트에 대한 클래스 분포:

$$p_\phi(y = k \mid \mathbf{x}) = \frac{\exp(-d(f_\phi(\mathbf{x}), \mathbf{c}_k))}{\sum_{k'} \exp(-d(f_\phi(\mathbf{x}), \mathbf{c}_{k'}))}$$

클래스 수준 불일치 손실 (MMD 기반):

$$L_G\left(\{\mu^s_c\}, \{\mu^t_c\}, \{\mu^b_c\}\right) \triangleq \frac{1}{C}\sum_{c=1}^{C}\|\tilde{\mu}^s_c - \tilde{\mu}^t_c\|^2_{\mathcal{H}} + \frac{1}{C}\sum_{c=1}^{C}\|\tilde{\mu}^s_c - \tilde{\mu}^b_c\|^2_{\mathcal{H}} + \frac{1}{C}\sum_{c=1}^{C}\|\tilde{\mu}^t_c - \tilde{\mu}^b_c\|^2_{\mathcal{H}}$$

여기서 $\{\tilde{\mu}^s_c\}$, $\{\tilde{\mu}^t_c\}$, $\{\tilde{\mu}^b_c\}$는 각각 소스, 타겟, 브리지 도메인의 재생 커널 힐베르트 공간(RKHS) $\mathcal{H}$에서의 클래스별 프로토타입입니다.

**(d) 분리 구성요소 (Disentanglement):**

클래스 무관 특징 추출을 위한 엔트로피 손실:

$$\mathcal{L}_{ent} = -\frac{1}{n_s}\sum_{j=1}^{n_s}\log C(f^j_{ci}) - \frac{1}{n_t}\sum_{j=1}^{n_t}\log C(f^j_{ci})$$

**(e) 상호 정보 최소화 (MINE 기반):**

$$I(\mathcal{P}, \mathcal{Q}) = \frac{1}{n}\sum_{i=1}^{n}T(p, q, \theta) - \log\left(\frac{1}{n}\sum_{i=1}^{n}e^{T(p, q', \theta)}\right)$$

---

### 2.3 모델 구조

```
[CFGAN 구조]
S → G_{S→B} → B → G_{B→T} → T
T → F_{T→B} → B → F_{B→S} → S
(Discriminators: D_S, D_B, D_T)

[PADA 구조]
입력 데이터 (S/B/T)
    ↓ Feature Extractor G (공유 가중치)
  특징 벡터
    ├── Domain Identifier (DI) → ADA
    ├── Prototype Net (P) → MMD Loss
    └── Disentangler (D)
            ├── f_di (domain-invariant)
            └── f_ds (domain-specific)
                    → Reconstructor R (L2 Loss)
                    → Mutual Info Minimizer (MINE)
```

**CFGAN 생성기**: 6~9개의 Residual Block (c7s1-64, d128, d256, R256×6 or ×9, u128, u64, c7s1-3)

**판별기**: 70×70 PatchGAN (C64-C128-C256-C512)

---

### 2.4 성능 향상

**DomainNet (이미지-이미지 변환 및 인식):**

| Method | qdr→rel | rel→qdr | Mean |
|--------|---------|---------|------|
| Source Only | 0.31 | 0.13 | 0.22 |
| CycleGAN | 0.35 | 0.15 | 0.25 |
| DAN | 0.50 | 0.18 | 0.34 |
| **CFGAN (Ours)** | **0.52** | **0.27** | **0.39** |

**Digit-Five (도메인 적응):**

| Method | SVHN→MNIST | MNIST→SVHN | Average |
|--------|-----------|-----------|---------|
| DANN | 65.4 | 17.7 | 44.7 |
| DAN | 70.4 | 21.5 | 48.8 |
| ADDA | 69.2 | 19.7 | 47.5 |
| **PADA (Ours)** | **72.1** | **24.3** | **50.1** |

**DomainNet (대규모 도메인 적응):**

| Method | Average |
|--------|---------|
| Source Only | 10.8 |
| DAN | 13.8 |
| MCD | 12.6 |
| **PADA (Ours)** | **14.2** |

---

### 2.5 한계점

1. **브리지 도메인 선택의 어려움**: 적합한 브리지 도메인을 수동으로 선택해야 하며, 자동 선택 메커니즘이 없음
2. **계산 비용 증가**: 두 단계의 GAN 훈련(S→B, B→T)으로 인한 훈련 비용 증가
3. **단일 브리지 도메인 제한**: 도메인 격차가 매우 클 경우 하나의 브리지 도메인으로는 부족할 수 있음
4. **레이블 의존성**: 소스 도메인의 레이블에만 의존하여 타겟 도메인 분류 성능이 제한될 수 있음
5. **부정적 전이의 완전한 제거 불가**: 클래스 분리를 통해 완화하지만 완전히 해소되지 않음

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화 향상의 핵심 메커니즘

**(1) 단계적 도메인 브리징을 통한 일반화:**

도메인 적응 이론(Ben-David et al., 2010)에 따르면, 타겟 위험은 다음과 같이 상한이 존재합니다:

$$\epsilon_T(h) \leq \epsilon_S(h) + \frac{1}{2}d_{\mathcal{H}\Delta\mathcal{H}}(\mathcal{D}_s, \mathcal{D}_t) + \lambda$$

브리지 도메인을 통해 직접적인 $d_{\mathcal{H}\Delta\mathcal{H}}(\mathcal{D}_s, \mathcal{D}_t)$ 대신 다음을 최소화합니다:

$$d_{\mathcal{H}\Delta\mathcal{H}}(\mathcal{D}_s, \mathcal{D}_b) + d_{\mathcal{H}\Delta\mathcal{H}}(\mathcal{D}_b, \mathcal{D}_t) < d_{\mathcal{H}\Delta\mathcal{H}}(\mathcal{D}_s, \mathcal{D}_t)$$

이는 A-Distance 실험으로 검증되었습니다 (Figure 7(a)).

**(2) 프로토타입 기반 클래스 수준 정렬:**

기존 방법들이 주변 분포(marginal distribution)만 정렬하는 것과 달리, PADA는 **클래스 수준의 프로토타입 정렬**을 통해 조건부 분포까지 정렬합니다. 이는 클래스별 특징 공간의 일관성을 보장하여 타겟 도메인에서의 일반화를 크게 향상시킵니다.

**(3) 특징 분리를 통한 부정적 전이 억제:**

도메인 불변 특징 $f_{di}$와 도메인 특화 특징 $f_{ds}$를 분리하고 상호 정보를 최소화:

$$I(f_{di}; f_{ds}) \to 0$$

이를 통해 클래스 무관 정보가 분류에 영향을 미치지 않도록 하여 일반화 성능을 높입니다.

**(4) 공유 가중치 구조:**

세 도메인(S, B, T)의 특징 추출기가 가중치를 공유함으로써, 더 일반적인 도메인 불변 표현을 학습합니다.

### 3.2 일반화 성능의 실험적 검증

- **A-Distance 감소**: PADA 특징의 A-distance가 DANN, ResNet 대비 낮아 소스-타겟 특징 분포가 더 잘 정렬됨
- **훈련 오류 감소**: 브리지 적용 시 Real→Sketch 태스크에서 훈련 오류가 감소하고 정확도 상승
- **t-SNE 시각화**: PADA 특징이 DAN, ADDA 대비 클래스 간 분리가 더 명확함

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4.1 연구에 미치는 영향

**(1) 대규모 도메인 격차 연구의 새로운 방향성:**
본 논문은 기존 연구들이 간과했던 **매우 큰 도메인 격차** 시나리오를 공식화함으로써, 실제 적용 환경에서의 도메인 적응 연구에 새로운 방향을 제시합니다.

**(2) 중간 도메인 활용 패러다임의 확산:**
브리지 도메인의 개념은 이후 다양한 연구에서 채택되었습니다. 예를 들어, 여러 개의 중간 도메인을 자동 생성하는 방향으로 발전합니다 (DLOW, Domain Flow 등).

**(3) 클래스 수준 정렬의 중요성 강조:**
프로토타입 기반 클래스 수준 MMD는 이후 연구들이 단순 주변 분포 정렬을 넘어서도록 영향을 미쳤습니다.

### 4.2 2020년 이후 관련 최신 연구 비교 분석

아래는 본 논문과 유사한 방향의 2020년 이후 주요 연구들입니다:

#### **(a) CDTrans (ICLR 2022)**
- **논문**: "CDTrans: Cross-domain Transformer for Unsupervised Domain Adaptation" (Xu et al., 2022)
- **주요 차이**: Transformer 기반 구조로 Self-attention을 활용한 도메인 정렬. PADA의 프로토타입 매칭보다 더 유연한 특징 정렬 가능
- **한계 극복**: PADA의 CNN 기반 특징 추출기를 Transformer로 대체하여 글로벌 문맥 정보 활용

#### **(b) SSRT (CVPR 2022)**
- **논문**: "Safe Self-Refinement for Transformer-based Domain Adaptation" (Sun et al., 2022)
- **관련성**: 도메인 불변 표현 학습에서 자기 정제(Self-Refinement) 도입
- **차이점**: 브리지 도메인 대신 의사 레이블(pseudo label)을 활용한 단계적 적응

#### **(c) PMTrans (ECCV 2022)**
- **논문**: "PMTrans: Patch Mix Transformer for Unsupervised Domain Adaptation" (Zhu et al., 2022)
- **관련성**: 도메인 간 중간 표현 생성
- **차이점**: 실제 브리지 도메인 대신 패치 혼합(Patch Mixing)으로 가상 중간 도메인 생성

#### **(d) SPA (CVPR 2021)**
- **논문**: "Gradually Vanishing Bridge for Adversarial Domain Adaptation" (Chen et al., 2020)
- **관련성**: 가장 직접적으로 관련된 후속 연구
- **개선**: 점진적으로 소멸하는 브리지를 도입하여 소스에서 타겟으로의 부드러운 전이 구현
- **차이점**: 외부 브리지 도메인 대신 학습 가능한 가상 브리지 도메인 사용

#### **(e) SHOT (ICML 2020)**
- **논문**: "Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation" (Liang et al., 2020)
- **관련성**: 소스 데이터 없이 타겟 도메인 적응
- **차이점**: 브리지 도메인이 아닌 정보 최대화와 의사 레이블 기법 활용

| 방법 | 브리지 도메인 | 클래스 정렬 | 특징 분리 | Transformer |
|------|-------------|------------|---------|------------|
| PADA (본 논문, 2019) | 외부 도메인 수동 선택 | MMD 프로토타입 | ✓ (MINE) | ✗ |
| GVB (2020) | 학습 가능한 가상 브리지 | ✗ | ✗ | ✗ |
| SHOT (2020) | ✗ | 정보 최대화 | ✗ | ✗ |
| CDTrans (2022) | ✗ | Cross-Attention | ✗ | ✓ |
| PMTrans (2022) | 패치 믹싱 가상 브리지 | ✓ | ✗ | ✓ |

### 4.3 앞으로 연구 시 고려할 점

**(1) 브리지 도메인의 자동 선택/생성:**
- 현재 수동으로 브리지 도메인을 선택해야 하는 한계를 극복하기 위해, **메타러닝** 또는 **Diffusion Model** 기반의 자동 브리지 도메인 생성 연구가 필요합니다.
- 여러 개의 브리지 도메인을 계층적으로 구성하는 방향 탐색

**(2) 트랜스포머 기반 아키텍처와의 결합:**
- PADA의 CNN 기반 특징 추출기를 **Vision Transformer (ViT)** 또는 **CLIP** 기반으로 대체하여 더 강력한 도메인 불변 표현 학습 가능성 탐구

**(3) 멀티모달 확장:**
- 텍스트-이미지, 이미지-비디오 등의 크로스 모달 시나리오에서 브리지 도메인 아이디어 적용

**(4) 이론적 보장 강화:**
- 브리지 도메인을 사용했을 때의 타겟 위험 상한에 대한 더 엄밀한 이론적 분석 필요

**(5) Few-Shot 및 Zero-Shot 설정으로의 확장:**
- 소스 도메인의 레이블이 극히 적은 상황에서도 브리지 도메인을 활용한 도메인 적응 가능성 탐구

**(6) 계산 효율성 개선:**
- 두 단계의 GAN 훈련으로 인한 높은 계산 비용을 줄이기 위한 경량화 연구 필요

**(7) 도메인 격차 자동 측정:**
- $\mathcal{H}\Delta\mathcal{H}$ 발산 외에 더 효율적인 도메인 격차 측정 지표 개발 (예: Optimal Transport 기반)

---

## 참고자료

**본 논문:**
- Li, Y., & Peng, X. (2019). "Learning Domain Adaptive Features with Unlabeled Domain Bridges." *arXiv:1912.05004v1*. [https://arxiv.org/abs/1912.05004](https://arxiv.org/abs/1912.05004)

**논문 내 인용 주요 참고문헌:**
- Zhu, J.-Y., Park, T., Isola, P., & Efros, A. A. (2017). "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks." *ICCV 2017.*
- Ben-David, S., et al. (2010). "A theory of learning from different domains." *Machine Learning, 79(1-2):151–175.*
- Long, M., et al. (2015). "Learning Transferable Features with Deep Adaptation Networks." *ICML 2015.*
- Ganin, Y., & Lempitsky, V. (2015). "Unsupervised Domain Adaptation by Backpropagation." *ICML 2015.*
- Tzeng, E., et al. (2017). "Adversarial Discriminative Domain Adaptation." *CVPR 2017.*
- Peng, X., et al. (2018). "Moment Matching for Multi-Source Domain Adaptation." *arXiv:1812.01754.*
- Belghazi, M. I., et al. (2018). "Mutual Information Neural Estimation." *ICML 2018.*
- Pan, Y., et al. (2019). "Transferrable Prototypical Networks for Unsupervised Domain Adaptation." *CVPR 2019.*
- Tan, B., et al. (2017). "Distant Domain Transfer Learning." *AAAI 2017.*

**2020년 이후 비교 연구:**
- Chen, X., et al. (2020). "Gradually Vanishing Bridge for Adversarial Domain Adaptation." *CVPR 2020.*
- Liang, J., et al. (2020). "Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation." *ICML 2020.*
- Xu, T., et al. (2022). "CDTrans: Cross-domain Transformer for Unsupervised Domain Adaptation." *ICLR 2022.*
- Zhu, Y., et al. (2022). "PMTrans: Patch Mix Transformer for Unsupervised Domain Adaptation." *ECCV 2022.*

> **⚠️ 정확도 주의**: 2020년 이후 비교 연구 부분은 논문 PDF에 직접 수록된 내용이 아니며, 공개된 학술 데이터베이스(arXiv, CVPR/ICCV/ECCV/ICLR proceedings)를 기반으로 작성되었습니다. 일부 세부 수치나 방법론적 세부 사항은 원 논문을 직접 확인하시기 바랍니다.
