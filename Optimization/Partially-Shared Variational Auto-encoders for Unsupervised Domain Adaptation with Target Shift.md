# Partially-Shared Variational Auto-encoders for Unsupervised Domain Adaptation with Target Shift

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

본 논문은 **Target Shift** 문제(소스와 타겟 도메인 간 레이블 분포 불일치)를 가진 비지도 도메인 적응(UDA)에서, 기존의 **특징 분포 매칭(feature distribution matching)** 방식 대신 **샘플 단위 특징 정렬(pair-wise feature alignment)** 을 사용해야 한다고 주장합니다.

### 주요 기여 (3가지)

| 기여 | 설명 |
|------|------|
| **방법론적 기여** | Target shift 문제를 데이터 증강 기반 오버샘플링으로 극복하는 PS-VAEs 제안 |
| **분류 성능** | 클래스 불균형 digit 데이터셋에서 최고 성능 달성 |
| **회귀 과제 개척** | Target shift가 있는 UDA 환경에서 인간 자세 추정(회귀) 문제에 처음 적용, 큰 차이로 기존 방법 능가 |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

**Target Shift(Prior Probability Shift)** 란 소스 도메인과 타겟 도메인 간의 레이블 사전 분포가 다른 현상입니다:

$$P_s(Y) \neq P_t(Y)$$

실제 환경에서 타겟 도메인의 레이블 $Y_t$와 분포 $\Pr(Y_t)$는 알 수 없으며, 특히 **클래스 불균형(class imbalance)** 형태로 나타납니다.

**기존 방법의 문제점:**

- ADDA, UFDN, CyCADA 등 **분포 매칭 기반** 방법들은 $P_s(Z) = P_t(Z)$를 암묵적으로 가정
- Target shift 상황에서 분포 매칭은 **특징 공간의 미스얼라인먼트(misalignment)** 를 유발
- MCD 등 결정 경계 기반 방법은 **회귀(regression) 문제에 적용 불가**

$$\text{기존: } \min_{E_s, E_t} \text{dist}(P(E_s(X_s)), P(E_t(X_t))) \rightarrow \text{target shift에서 실패}$$

### 2.2 제안하는 방법 (수식 포함)

#### 전체 목적 함수

$$L_{total} = L_{adv} + \alpha L_{cyc} + \beta L_{id} + \gamma L_{KL} + \delta L_{fc} + \epsilon L_{pred} $$

여기서 $\alpha, \beta, \gamma, \delta, \epsilon$은 태스크별로 조정되는 하이퍼파라미터입니다.

---

#### (1) CycleGAN 기본 손실

**Cycle Consistency Loss:**

```math
\min_{E_s, E_t, G_s, G_t} L_{cyc}(X_s, X_t) = \sum_{* \in \{s,t\}} \mathbb{E}_{x \in X_*}[d(x, G_*(E_{\bar{*}}(\hat{x}_{\bar{*}})))]
```

**Identity Loss:**

```math
\min_{E_s, E_t, G_s, G_t} L_{id}(X_s, X_t) = \sum_{* \in \{s,t\}} \mathbb{E}_{x \in X_*}[d(x, G_*(E_*(x)))]
```

**Adversarial Loss (LSGAN 기반):**

$$\min_{E_s,E_t,G_s,G_t} \max_{D_s,D_t} L_{adv}(X_s,X_t) = \mathbb{E}_{\{x_s,x_t\} \in X_s \times X_t}$$
$$[\|D_s(x_s) - 1\|_2 + \|D_s(G_s(E_t(x_t))) + 1\|_2$$
$$+ \|D_t(x_t) - 1\|_2 + \|D_t(G_t(E_s(x_s))) + 1\|_2] $$

---

#### (2) 특징 일관성 손실 $L_{fc}$ (핵심 기여)

특징을 **도메인 불변 성분 $z$** 와 **도메인 특이적 성분 $\zeta_*$** 로 분리:

```math
z_* = \{z, \zeta_*\} = E_*(x_*)
```

도메인 불변 성분만을 대상으로 쌍별 특징 정렬:

```math
\min_{E_s, E_t, G_s, G_t} L_{fc}(Z_s, Z_t) = \sum_{* \in \{s,t\}} \mathbb{E}_{z \in Z_*}[d(z, E_{\bar{*}}(G_{\bar{*}}(z)))]
```

> **주의:** $L_{fc}$에서 $z$ 너머로의 역전파는 수행하지 않음 (수렴 안정성을 위해)

---

#### (3) KL Divergence Loss (VAE 메커니즘)

```math
\min_{E_s, E_t} L_{KL}(X_s, X_t) = \sum_{* \in \{s,t\}} \mathbb{E}_{z, \zeta_* \in E_*(X_*)}[KL(p_z \| q_z) + KL(p_{\zeta_*} \| q_{\zeta_*})]
```

여기서 $q_z$와 $q_{\zeta_*}$는 표준 정규 분포입니다.

---

#### (4) 예측 손실 $L_{pred}$

**분류 태스크 (Cross-entropy):**

$$\min_{E_s, M} L_{pred}(Y_s, X_s) = \mathbb{E}_{x_s, y_s \in X_s \times L}[-y_s \log M(E(x_s))] $$

**회귀 태스크 (Smooth L1):**

$$\min_{E_s, M} L_{pred}(Y_s, X_s) = \mathbb{E}_{x_s, y_s \in \{X_s, Y_s\}}(d(M(E_s(x_s)), y_s)) $$

여기서 거리 함수 $d$는 Smooth L1:

$$d(a, b) = \begin{cases} \|a - b\|_2 & \text{if } |a - b| < 1 \\ |a - b| - 0.5 & \text{otherwise} \end{cases} $$

---

### 2.3 모델 구조

```
┌─────────────────────────────────────────────────────┐
│                    PS-VAEs 구조                      │
│                                                     │
│  xs ──→ [E] ──→ {z, ζs, ζt} ──→ [G_t] ──→ x̂t    │
│                    ↓                                │
│                   [M] ──→ ŷ (예측)                 │
│                                                     │
│  xt ──→ [E] ──→ {z, ζs, ζt} ──→ [G_s] ──→ x̂s    │
│                                                     │
│  두 VAE: G_s∘E_t 와 G_t∘E_s                       │
└─────────────────────────────────────────────────────┘
```

**Partially-Shared 구조의 핵심:**

| 구성요소 | 공유 방식 |
|---------|---------|
| **Encoder $E$** | 대부분 가중치 공유, 마지막 레이어만 $z$, $\zeta_s$, $\zeta_t$로 분기 |
| **Decoder $G$** | 첫 번째 레이어 제외 가중치 공유 |
| $E_s$ 사용 시 | $\zeta_t$ 출력 무시: $E(x_s) = \{z, \zeta_s\}$ |
| $G_s$ 사용 시 | $\{z, \zeta_s, \mathbf{0}\}$ 입력 |
| $G_t$ 사용 시 | $\{z, \mathbf{0}, \zeta_t\}$ 입력 |

**테스트 시 경로:** $x_t \rightarrow E_t \rightarrow z_t \rightarrow M \rightarrow \hat{y}$

---

### 2.4 성능 향상 및 한계

#### 성능 향상

**분류 태스크 (MNIST→USPS, 정확도 %):**

| 방법 | 10% | 20% | 30% | 40% | 50% |
|------|-----|-----|-----|-----|-----|
| ADDA | 89.8 | 86.9 | 79.3 | 81.8 | 78.5 |
| UFDN | 94.0 | 90.4 | 83.2 | 82.3 | 83.8 |
| CyCADA | 91.8 | 91.0 | 80.3 | 86.4 | 87.6 |
| **Ours** | **93.9** | **94.8** | **93.4** | **94.6** | **92.6** |

**회귀 태스크 (인간 자세 추정, 10px 이내 정확도 %):**

| 방법 | Avg. |
|------|------|
| Source only | 1.8 |
| MCD | 5.3 |
| SimGAN | 40.4 |
| CyCADA | 41.0 |
| **PS-VAEs (Ours)** | **57.0** |

#### 한계

1. **SVHN→MNIST 성능**: 큰 픽셀값 차이로 인해 MCD(90.3%)보다 낮은 성능(73.7%)
2. **하이퍼파라미터 민감성**: $\alpha, \beta, \gamma, \delta, \epsilon$ 5개의 하이퍼파라미터를 태스크별로 조정 필요
3. **훈련 불안정성**: D-CycleGAN+VAE 조합이 가중치 공유 없이는 불안정 (UNIT과 유사한 문제)
4. **계산 복잡도**: CycleGAN 기반으로 훈련 비용이 높음
5. **적용 도메인 제한**: 깊이 이미지 기반 실험만 수행, RGB 등 다른 모달리티로의 확장성 미검증

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화를 가능하게 하는 핵심 메커니즘

#### (A) 분포 매칭 회피 → 분포 불변적 일반화

기존 방법은 다음을 최소화:

$$\min \text{MMD}(P_s(Z), P_t(Z)) \text{ 또는 } \min L_{adv}(D_s(Z_s), D_t(Z_t))$$

이는 $P_s(Y) = P_t(Y)$를 암묵적으로 가정하므로, target shift 시 특징 공간이 다음과 같이 왜곡됩니다:

$$z_{\text{class=1, majority}} \approx z_{\text{class=2, minority}}$$

반면 PS-VAEs의 쌍별 정렬은:

```math
\min d(z_s, E_{\bar{*}}(G_{\bar{*}}(z_s))) \text{ where } y(z_s) = y(\hat{z}_t)
```

레이블 보존 하에 정렬하므로 분포 왜곡이 없습니다.

#### (B) 특징 분리(Disentanglement)에 의한 일반화

도메인 불변 특징 $z$와 도메인 특이 특징 $\zeta_*$의 분리는:

$$E(x) = \{z, \zeta_*\}, \quad z \perp \!\!\! \perp \text{domain}$$

이를 통해 $z$가 실제 의미론적 정보만 담게 되어, 새로운 도메인에서도 일반화 가능한 표현을 학습합니다.

#### (C) VAE의 정규화 효과

KL Divergence 손실:

$$L_{KL} = KL(p_z \| \mathcal{N}(0,I)) + KL(p_{\zeta_*} \| \mathcal{N}(0,I))$$

이는 잠재 공간을 매끄럽고(smooth) 연속적으로 만들어, 도메인 간 보간(interpolation)을 가능하게 하며 과적합(overfitting)을 방지합니다.

#### (D) 가중치 공유의 일반화 기여

동일한 인코더 $E$가 두 도메인 모두에 사용됨으로써:
- 소스/타겟 도메인에서 동일한 의미론적 특징을 추출하는 능력 강제
- 파라미터 수 감소 → 암묵적 정규화 효과

#### (E) 과대표본추출(Oversampling) 기반 증강

클래스 불균형을 해소하기 위해 CycleGAN으로 의사 쌍(pseudo-pairs)을 생성:

$$(\hat{x}_t, x_s) \text{ s.t. } y(\hat{x}_t) \approx y(x_s)$$

이는 언더샘플링과 달리 정보 손실 없이 균형 잡힌 학습을 가능하게 합니다.

### 3.2 일반화의 실증적 증거

**t-SNE 시각화 분석 (Figure 7 기반):**

| 방법 | 특징 분포 혼합 정도 |
|------|------------------|
| Source Only | 두 도메인 완전 분리 |
| SimGAN | 부분적 혼합 |
| CyCADA | 상당한 혼합, 그러나 일부 분리 |
| **PS-VAEs** | **가장 잘 혼합됨 (판별자/불일치 최소화 없이 달성)** |

---

## 4. 향후 연구에 미치는 영향과 고려할 점

### 4.1 향후 연구에 미치는 영향

#### (A) 패러다임 전환: 분포 매칭 → 쌍별 정렬

PS-VAEs는 UDA 연구에서 분포 전체를 맞추는 것이 아닌, **레이블 보존 쌍별 정렬**이라는 새로운 패러다임을 제시했습니다. 이는 다음 연구 방향에 영향을 줍니다:

- **대조 학습(Contrastive Learning) 기반 UDA**: 레이블 정보 없이도 의미론적으로 유사한 샘플끼리 정렬 (예: [SimCLR, MoCo의 아이디어를 UDA에 적용](https://arxiv.org/abs/2002.05709))
- **의사 레이블(Pseudo-label) 기반 정렬**: 타겟 도메인에 의사 레이블을 부여한 후 정렬

#### (B) 회귀 UDA의 가능성 개척

기존 연구 대부분이 분류에 집중한 반면, PS-VAEs는 **회귀 태스크에서의 UDA**가 가능함을 실증했습니다. 이는:

- 의료 영상 분석 (continuous label: 측정값, 나이 추정)
- 자율 주행의 거리 추정
- 감정 강도 예측 등으로의 확장 가능성

#### (C) 특징 분리(Disentanglement)의 중요성 강조

도메인 불변/특이 특징의 명시적 분리는 이후 **도메인 일반화(Domain Generalization)** 연구에서도 핵심 아이디어로 활용됩니다.

---

### 4.2 2020년 이후 관련 최신 연구 비교 분석

> **주의:** 아래 논문들은 PS-VAEs와 관련 있는 대표적 연구들이나, 본 논문이 직접 인용하거나 비교한 것은 아닙니다. 각 논문의 핵심 내용은 해당 논문의 공개된 정보를 바탕으로 기술합니다.

#### (A) 대조 학습 기반 UDA

**CDTrans (2021, ICLR 2022 관련)**
- 교차 주의 메커니즘(cross-attention)을 이용한 도메인 정렬
- Transformer 구조를 UDA에 도입
- PS-VAEs의 쌍별 정렬과 유사한 직접 매핑 개념을 발전

**SWD (Sliced Wasserstein Discrepancy, 2019~)를 활용한 후속 연구들**
- 분포 매칭을 더 정교하게 수행하지만 target shift에는 여전히 취약

#### (B) Target Shift 특화 연구

**COAL (2020, Cycle-Consistent Optimal Assignment Learning)**
- 최적 전송(Optimal Transport) 이론으로 클래스 불균형 처리
- PS-VAEs보다 이론적으로 엄밀한 target shift 처리를 목표

**UniOT (2023)**
- 비지도 도메인 적응에서 optimal transport를 활용
- target shift를 explicit하게 모델링

#### (C) Transformer 기반 UDA

**TVT (Transferable Vision Transformer, 2022)**
- ViT 기반의 UDA
- 대규모 사전훈련의 이점으로 일반화 성능 향상
- PS-VAEs의 CycleGAN보다 훨씬 큰 모델로 단순히 스케일로 문제 완화

**비교 분석:**

| 측면 | PS-VAEs | Transformer 기반 UDA (2021~) |
|------|---------|----------------------------|
| Target Shift 처리 | 명시적 설계 | 일반적으로 미고려 |
| 회귀 적용 | 가능 | 대부분 분류에 집중 |
| 계산 비용 | CycleGAN 기반 (중간) | 매우 높음 |
| 데이터 요구량 | 중간 | 매우 많음 |
| 이론적 보장 | 제한적 | 제한적 |

#### (D) 생성 모델 기반 UDA의 발전

**Diffusion Model 기반 도메인 변환 (2022~)**
- DDPM 등을 활용한 도메인 변환
- CycleGAN보다 훨씬 높은 품질의 도메인 변환 가능
- PS-VAEs의 CycleGAN 부분을 Diffusion으로 대체하면 성능 향상 가능성

---

### 4.3 앞으로의 연구 시 고려할 점

#### (1) 이론적 보장 강화

현재 PS-VAEs는 경험적(empirical) 성능을 보여주지만, target shift 하에서의 **일반화 오류 바운드(generalization error bound)** 이론이 없습니다. 향후 연구에서는:

$$\mathcal{E}_t(h) \leq \mathcal{E}_s(h) + d_{\mathcal{H}}(P_s, P_t) + \lambda$$

형태의 이론 (Ben-David et al., 2010)을 target shift 조건으로 확장해야 합니다.

#### (2) 확장성(Scalability) 문제

- CycleGAN 기반 구조는 고해상도 이미지에서 훈련 비용이 매우 높음
- **Diffusion Model이나 Flow-based Model**로 대체하여 품질과 효율성 모두 향상 가능

#### (3) 멀티모달 확장

현재 단일 모달리티(깊이 이미지)만 다루지만:
- RGB + Depth 등 멀티모달 환경에서의 PS-VAEs 확장
- 텍스트-이미지 간 도메인 적응

#### (4) Target Shift 정도의 자동 감지

현재는 target shift의 존재를 사전에 가정하지만:
- 타겟 도메인의 레이블 분포를 추정하는 메커니즘 통합
- Adaptive weighting: shift 정도에 따라 $L_{fc}$와 $L_{adv}$의 비중 자동 조절

$$\omega_{fc} = f(\hat{d}_{JS}(P_s(Y), P_t(Y)))$$

#### (5) Few-shot 및 Semi-supervised 설정과의 결합

소수의 타겟 레이블이 있을 때 PS-VAEs를 어떻게 활용할 것인가:
- Semi-supervised PS-VAEs: 타겟 도메인의 일부 레이블 활용
- 이는 실용적 환경에서 더 현실적인 시나리오

#### (6) 공정성(Fairness) 관점

Target shift는 특정 그룹(클래스)이 과소 대표되는 현상과 직결됩니다. 의료 등 민감한 분야에서:
- 소수 클래스에 대한 공정한 예측 보장
- 편향 없는 도메인 적응

---

## 참고 자료

**주요 참고 문헌:**

1. **본 논문 (arXiv):** Takahashi, R., Hashimoto, A., Sonogashira, M., & Iiyama, M. (2020). "Partially-Shared Variational Auto-encoders for Unsupervised Domain Adaptation with Target Shift." arXiv:2001.07895v3

2. **CycleGAN:** Zhu, J.-Y., Park, T., Isola, P., & Efros, A. A. (2017). "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks." ICCV 2017.

3. **ADDA:** Tzeng, E., Hoffman, J., Saenko, K., & Darrell, T. (2017). "Adversarial Discriminative Domain Adaptation." CVPR 2017.

4. **CyCADA:** Hoffman, J., et al. (2018). "CyCADA: Cycle-Consistent Adversarial Domain Adaptation."

5. **MCD:** Saito, K., et al. (2018). "Maximum Classifier Discrepancy for Unsupervised Domain Adaptation." CVPR 2018.

6. **PADA:** Cao, Z., et al. (2018). "Partial Adversarial Domain Adaptation." ECCV 2018.

7. **VAE:** Doersch, C. (2016). "Tutorial on Variational Autoencoders."

8. **UFDN:** Liu, A. H., et al. (2018). "A Unified Feature Disentangler for Multi-Domain Image Translation and Manipulation." NeurIPS 2018.

9. **SimGAN:** Shrivastava, A., et al. (2017). "Learning from Simulated and Unsupervised Images through Adversarial Training." CVPR 2017.

10. **Ben-David et al. (2010):** "A theory of learning from different domains." Machine Learning, 79(1-2), 151-175. *(일반화 이론 관련 배경 참고)*

> **면책 고지:** 2020년 이후 최신 연구 비교(CDTrans, UniOT, TVT 등) 부분은 논문 원문에 직접 인용된 내용이 아니며, 관련 연구 흐름을 분석한 내용입니다. 해당 논문들의 세부 수치 비교는 직접 논문 확인을 권장합니다.
