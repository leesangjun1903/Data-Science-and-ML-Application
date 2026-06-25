# Adversarial Discriminative Domain Adaptation (ADDA)

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

ADDA(Adversarial Discriminative Domain Adaptation)는 **비지도 도메인 적응(Unsupervised Domain Adaptation)** 문제에서, 생성 모델(GAN) 기반 접근법과 판별 모델(Discriminative) 기반 접근법의 장점을 결합하여, 더 단순하면서도 효과적인 방법을 제안한다.

기존 방법들의 한계:
- **GAN 기반(CoGAN 등)**: 시각화는 뛰어나지만, 판별 태스크에는 최적이 아니며, 도메인 간 차이가 클 경우 수렴하지 못함
- **판별 모델 기반(Gradient Reversal, Domain Confusion 등)**: 대규모 도메인 이동 처리 가능하지만, 가중치 공유(Tied Weights) 제약으로 유연성 부족, GAN 손실 미활용

### 주요 기여

1. **통합 프레임워크 제시**: 기존 적대적 도메인 적응 방법들(Gradient Reversal, Domain Confusion, CoGAN 등)을 하나의 일반화된 프레임워크 안에서 설명
2. **ADDA 제안**: 판별 모델 + 비공유 가중치(Untied Weights) + GAN 손실의 조합이라는 새로운 인스턴스 제안
3. **실험적 검증**: MNIST, USPS, SVHN, NYUD, Office 데이터셋에서 당시 최고 성능(SOTA) 달성

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**도메인 시프트(Domain Shift) / 데이터셋 편향(Dataset Bias)** 문제:
- 대규모 소스 도메인에서 훈련된 딥러닝 모델이 타겟 도메인에서 성능이 저하됨
- 타겟 도메인의 레이블 데이터 없이(비지도 설정) 적응이 필요함

구체적 설정:
- 소스 도메인: 이미지 $\mathbf{X}_s$와 레이블 $Y_s$, 분포 $p_s(x, y)$에서 샘플링
- 타겟 도메인: 이미지 $\mathbf{X}_t$만 존재, 분포 $p_t(x, y)$에서 샘플링 (레이블 없음)
- 목표: 타겟 도메인에서 $K$개 카테고리 분류를 올바르게 수행하는 $M_t$, $C_t$ 학습

---

### 2.2 제안하는 방법 (수식 포함)

#### (1) 일반화된 적대적 적응 프레임워크

**소스 분류 손실 (Source Classification Loss)**:

$$\min_{M_s, C} \mathcal{L}_{cls}(\mathbf{X}_s, Y_s) = \mathbb{E}_{(\mathbf{x}_s, y_s) \sim (\mathbf{X}_s, Y_s)} \left[ -\sum_{k=1}^{K} \mathbb{1}_{[k=y_s]} \log C(M_s(\mathbf{x}_s)) \right] \tag{1}$$

**도메인 판별자 손실 (Domain Discriminator Loss)**:

$$\mathcal{L}_{adv_D}(\mathbf{X}_s, \mathbf{X}_t, M_s, M_t) = -\mathbb{E}_{\mathbf{x}_s \sim \mathbf{X}_s}[\log D(M_s(\mathbf{x}_s))] - \mathbb{E}_{\mathbf{x}_t \sim \mathbf{X}_t}[\log(1 - D(M_t(\mathbf{x}_t)))] \tag{2}$$

**일반화된 적대적 도메인 적응 프레임워크**:

$$\min_{D} \mathcal{L}_{adv_D}(\mathbf{X}_s, \mathbf{X}_t, M_s, M_t)$$

$$\min_{M_s, M_t} \mathcal{L}_{adv_M}(\mathbf{X}_s, \mathbf{X}_t, D)$$

$$\text{s.t.} \quad \psi(M_s, M_t) \tag{3}$$

여기서 $\psi(M_s, M_t)$는 소스·타겟 매핑 간 제약 조건(가중치 공유 등)을 나타냄.

#### (2) 기존 방법들의 적대적 손실 비교

**Gradient Reversal (minimax loss)**:

$$\mathcal{L}_{adv_M} = -\mathcal{L}_{adv_D} \tag{6}$$

> 문제점: 훈련 초기 판별자가 빠르게 수렴하면 그래디언트가 소실됨

**GAN 손실 (Inverted Label GAN Loss)**:

$$\mathcal{L}_{adv_M}(\mathbf{X}_s, \mathbf{X}_t, D) = -\mathbb{E}_{\mathbf{x}_t \sim \mathbf{X}_t}[\log D(M_t(\mathbf{x}_t))] \tag{7}$$

> 특징: minimax 손실과 동일한 고정점 특성을 가지지만, 타겟 매핑에 더 강한 그래디언트 제공

**Domain Confusion Loss**:

$$\mathcal{L}_{adv_M}(\mathbf{X}_s, \mathbf{X}_t, D) = -\sum_{d \in \{s,t\}} \mathbb{E}_{\mathbf{x}_d \sim \mathbf{X}_d} \left[ \frac{1}{2}\log D(M_d(\mathbf{x}_d)) + \frac{1}{2}\log(1 - D(M_d(\mathbf{x}_d))) \right] \tag{8}$$

#### (3) ADDA의 최종 최적화 목적함수

ADDA는 세 단계로 구성된 순차적 최적화를 수행:

$$\min_{M_s, C} \mathcal{L}_{cls}(\mathbf{X}_s, Y_s) = -\mathbb{E}_{(\mathbf{x}_s, y_s) \sim (\mathbf{X}_s, Y_s)} \sum_{k=1}^{K} \mathbb{1}_{[k=y_s]} \log C(M_s(\mathbf{x}_s))$$

$$\min_{D} \mathcal{L}_{adv_D}(\mathbf{X}_s, \mathbf{X}_t, M_s, M_t) = -\mathbb{E}_{\mathbf{x}_s \sim \mathbf{X}_s}[\log D(M_s(\mathbf{x}_s))] - \mathbb{E}_{\mathbf{x}_t \sim \mathbf{X}_t}[\log(1 - D(M_t(\mathbf{x}_t)))]$$

$$\min_{M_t} \mathcal{L}_{adv_M}(\mathbf{X}_s, \mathbf{X}_t, D) = -\mathbb{E}_{\mathbf{x}_t \sim \mathbf{X}_t}[\log D(M_t(\mathbf{x}_t))] \tag{9}$$

**레이어별 제약 조건**:

$$\psi(M_s, M_t) \triangleq \{\psi_{\ell_i}(M_s^{\ell_i}, M_t^{\ell_i})\}_{i \in \{1 \ldots n\}} \tag{4}$$

$$\psi_{\ell_i}(M_s^{\ell_i}, M_t^{\ell_i}) = (M_s^{\ell_i} = M_t^{\ell_i}) \tag{5}$$

ADDA는 $\psi = \emptyset$ (가중치 비공유, Untied)을 선택.

---

### 2.3 모델 구조

ADDA의 훈련은 아래 3단계로 이루어짐:

```
[1단계: Pre-training]
소스 이미지 + 레이블 → Source CNN (Ms) + Classifier (C)
→ 소스 도메인에서 판별적 표현 학습

[2단계: Adversarial Adaptation]
소스 이미지 → Source CNN (Ms, 고정) ──┐
                                       ├→ Discriminator (D) → domain label
타겟 이미지 → Target CNN (Mt, 학습) ──┘
→ D는 소스/타겟 도메인을 구분하려 하고,
  Mt는 D를 속이도록 학습 (GAN 방식)

[3단계: Testing]
타겟 이미지 → Target CNN (Mt) → Classifier (C, 고정) → 클래스 레이블
```

| 방법 | 기반 모델 | 가중치 공유 | 적대적 손실 |
|------|-----------|-------------|-------------|
| Gradient Reversal | 판별적 | 공유 | minimax |
| Domain Confusion | 판별적 | 공유 | confusion |
| CoGAN | 생성적 | 비공유 | GAN |
| **ADDA (제안)** | **판별적** | **비공유** | **GAN** |

**핵심 설계 선택**:
- **판별 모델 사용**: 생성 모델에서 이미지 생성에 필요한 파라미터들이 판별 태스크에 불필요함
- **비공유 가중치**: 각 도메인에 특화된 특징 추출 가능 (더 유연한 비대칭 매핑)
- **GAN 손실**: 소스 분포는 고정하고 타겟 분포만 학습하므로, GAN의 inverted label loss 적합
- **사전 학습 모델로 초기화**: 타겟 도메인 레이블 없이도 퇴화 솔루션 방지

---

### 2.4 성능 향상

#### 디지털 데이터셋 (MNIST, USPS, SVHN)

| 방법 | MNIST→USPS | USPS→MNIST | SVHN→MNIST |
|------|-----------|-----------|-----------|
| Source only | 0.752 | 0.571 | 0.601 |
| Gradient reversal | 0.771 | 0.730 | 0.739 |
| Domain confusion | 0.791 | 0.665 | 0.681 |
| CoGAN | 0.912 | 0.891 | 수렴 실패 |
| **ADDA (제안)** | **0.894** | **0.901** | **0.760** |

- SVHN→MNIST처럼 도메인 차이가 큰 경우 CoGAN은 수렴 실패, ADDA는 성공

#### 모달리티 적응 (NYUD: RGB→Depth)

| 방법 | 평균 정확도 |
|------|-----------|
| Source only | 13.9% |
| **ADDA** | **21.1%** |
| Train on target | 46.8% |

- 레이블 없이 약 **51.8% 상대적 성능 향상**

#### Office 데이터셋

| 방법 | A→W | D→W | W→D |
|------|-----|-----|-----|
| Source only (ResNet-50) | 0.626 | 0.961 | 0.986 |
| DANN | 0.730 | 0.964 | 0.992 |
| **ADDA** | **0.751** | **0.970** | **0.996** |

---

### 2.5 한계점

1. **불안정한 훈련**: GAN 기반 훈련 특성상 하이퍼파라미터에 민감하고 수렴이 불안정할 수 있음
2. **클래스 정보 미활용**: 적대적 학습 시 클래스 수준의 정렬이 아닌 도메인 수준의 정렬만 수행 → 클래스 조건부 분포 차이( $p_s(x|y) \neq p_t(x|y)$ )를 충분히 해소하지 못할 가능성
3. **부정적 전이 가능성**: 일부 클래스(pillow, nightstand)에서 적응 후 성능 하락 관찰
4. **대규모 도메인 시프트의 한계**: 도메인 차이가 매우 클 경우 여전히 성능 제한적
5. **소규모 데이터셋 과적합**: Office 데이터셋처럼 소규모일 경우 전체 파인튜닝 시 과적합 위험

---

## 3. 모델의 일반화 성능 향상 가능성 (중점 분석)

### 3.1 일반화를 위한 핵심 메커니즘

ADDA의 일반화 성능 향상은 다음 원리에 근거:

**도메인 불변 특징 공간 학습**:
- 판별자 $D$가 소스/타겟 특징을 구분하지 못하도록 $M_t$를 학습
- 수학적으로: $M_s(\mathbf{X}_s)$와 $M_t(\mathbf{X}_t)$의 분포 거리를 최소화

$$d(M_s(\mathbf{X}_s), M_t(\mathbf{X}_t)) \to 0$$

이론적 배경(Ben-David et al., 2010의 도메인 적응 이론):

$$\epsilon_t(h) \leq \epsilon_s(h) + d_{\mathcal{H}\Delta\mathcal{H}}(\mathcal{D}_s, \mathcal{D}_t) + \lambda$$

여기서 $d_{\mathcal{H}\Delta\mathcal{H}}$는 $\mathcal{H}$-다이버전스(H-divergence), $\lambda$는 이상적 결합 오류. ADDA는 적대적 학습으로 이 $\mathcal{H}$-다이버전스 항을 줄이는 방향으로 학습.

### 3.2 비공유 가중치(Untied Weights)와 일반화

$$\psi(M_s, M_t) = \emptyset \quad (\text{no constraint})$$

- 각 도메인에 특화된 저수준 특징(low-level features)을 각 인코더가 독립적으로 학습 가능
- 소스 모델로 초기화 + 판별자로 정규화 → 과적합 방지 및 범용성 향상
- 공유 가중치 방법은 단일 네트워크가 두 도메인 이미지를 모두 처리해야 하므로 최적화가 불량 조건화(poorly conditioned)될 수 있음

### 3.3 순차적 훈련의 안정성 기여

- **소스 사전 학습 → 타겟 적응** 의 순차적 방식은 타겟 인코더가 의미있는 초기화에서 시작하게 함
- 소스 모델 $M_s$를 고정(freeze)함으로써 소스 분류 성능을 보존하면서 타겟 적응만 학습
- 이는 GAN의 "생성 분포가 실제 분포에 맞추는" 설정과 동일하게 안정적 수렴 유도

### 3.4 교차 모달리티(Cross-modality) 일반화

NYUD 실험에서 RGB → Depth라는 극단적으로 다른 도메인 간 전이에서도 상당한 성능 향상을 보임으로써, ADDA의 일반화 범위가 단순한 시각적 도메인 이동을 넘어 모달리티 수준의 차이에도 적용 가능함을 실증.

---

## 4. 향후 연구에 미치는 영향 및 고려 사항

### 4.1 후속 연구에 미친 영향

#### (1) 클래스 조건부 도메인 적응으로의 발전 동기

ADDA의 도메인 수준 정렬의 한계를 인식하고, 이후 연구들이 클래스 조건부 정렬로 발전:

- **CDAN (Conditional Domain Adversarial Network, Long et al., 2018)**: 클래스 예측과 도메인 판별을 조건부로 결합

$$\min_{G} \max_{D} \mathbb{E}[\log D(\mathbf{f} \otimes \hat{\mathbf{y}})] + \mathbb{E}[\log(1-D(\mathbf{f}' \otimes \hat{\mathbf{y}}'))]$$

- **SHOT (Shot et al., 2020)**: 소스 데이터 없이 타겟 도메인만으로 적응

#### (2) 통합 프레임워크의 유산

ADDA의 일반화 프레임워크는 이후 도메인 적응 연구의 설계 공간(design space)을 체계적으로 탐색하는 방법론적 기반을 제공. 많은 후속 논문들이 "어떤 손실, 어떤 가중치 공유, 어떤 아키텍처"라는 ADDA의 분류 체계를 참조함.

#### (3) 자기 지도 학습과의 결합 가능성

비지도 적응에서 의사 레이블(pseudo-label)이나 자기 지도(self-supervised) 사전 학습과의 결합으로 이어짐:

- **MCD (Maximum Classifier Discrepancy, Saito et al., 2018)**
- **SFDA (Source-Free Domain Adaptation)** 계열 연구

### 4.2 향후 연구 시 고려할 점

| 분류 | 고려 사항 |
|------|-----------|
| **이론적** | $\mathcal{H}$-divergence 최소화가 타겟 정확도를 보장하지 않는 경우 분석 필요 |
| **클래스 정렬** | 도메인 수준이 아닌 클래스 수준의 정렬 메커니즘 필요 |
| **안정성** | GAN 훈련 불안정성 해소 (Wasserstein GAN 등 활용 가능) |
| **소스 프리** | 소스 데이터 없이 적응하는 Source-Free DA 설정 고려 |
| **다중 소스** | 단일 소스가 아닌 다중 소스 도메인 적응으로 확장 |
| **부정적 전이** | 일부 클래스에서 성능 하락 방지를 위한 선택적 적응 |
| **트랜스포머** | Vision Transformer 기반 인코더와 결합 시 설계 재검토 필요 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 주요 발전 방향

#### (1) Source-Free Domain Adaptation (소스 데이터 없는 적응)

ADDA는 적응 시 소스 이미지가 여전히 필요. 이 제약을 제거한 연구들:

**SHOT (Liang et al., ICML 2020)**:
- 소스 모델을 고정하고 타겟 데이터만으로 정보 극대화(Information Maximization) + 의사 레이블 학습
- ADDA와 달리 적응 시 소스 데이터 불필요

$$\min_{M_t} -\mathbb{E}_{\mathbf{x}_t}[\sum_k p_k(\mathbf{x}_t) \log p_k(\mathbf{x}_t)] + \|\mathbf{p}_t - \frac{1}{K}\mathbf{1}\|_2^2$$

#### (2) Transformer 기반 Domain Adaptation

**CDTrans (Xu et al., ICLR 2022)**:
- Vision Transformer를 활용한 교차 주의(cross-attention) 기반 도메인 적응
- ADDA의 CNN 중심 설계와 달리 ViT의 전역 맥락 활용

**PMTrans (Zhu et al., ECCV 2022)**:
- Patch Mix Transformer로 소스-타겟 패치 수준의 정렬

#### (3) 클래스 조건부 도메인 적응

**CDAN (Long et al., NeurIPS 2018)**:
- 클래스 예측 $\hat{y}$와 특징 $\mathbf{f}$의 외적(outer product)을 판별자 입력으로 사용
- ADDA의 도메인 수준 정렬 한계를 클래스 조건부 정렬로 극복

**ATDOC (Liu et al., 2021)**:
- 이웃 클러스터링과 의사 레이블을 결합한 도메인 적응

#### (4) 대조 학습(Contrastive Learning) 기반

**CDL (Su et al., 2020)**, **SSRT (Sun et al., CVPR 2022)**:
- 소스와 타겟의 같은 클래스 샘플을 가깝게, 다른 클래스 샘플을 멀게 학습
- ADDA의 도메인 판별자 대신 대조 손실로 분포 정렬

#### (5) 의미론적 자기 지도 학습 통합

**MIC (Hoyer et al., CVPR 2023)** (세그멘테이션):
- Masked Image Consistency를 활용한 도메인 불변 표현 학습

### 5.2 ADDA와 최신 연구 비교 표

| 특성 | ADDA (2017) | CDAN (2018) | SHOT (2020) | CDTrans (2022) |
|------|-------------|-------------|-------------|-----------------|
| 기반 모델 | CNN (판별적) | CNN (판별적) | CNN (판별적) | ViT |
| 가중치 공유 | 비공유 | 공유 | 소스 고정 | 비공유 |
| 적대적 손실 | GAN | 조건부 GAN | 없음 (IM) | Cross-attention |
| 소스 데이터 필요 | O | O | X | O |
| 클래스 정렬 | X | O | O (의사 레이블) | O |
| Office-31 A→W | 75.1% | 82.0% | 94.0% | 97.5% |

> ⚠️ Office-31 수치는 각 논문에서 보고된 값이며, 실험 프로토콜 차이가 있을 수 있습니다.

---

## 참고 자료

**주요 참고 논문 (제공된 PDF)**:
- Eric Tzeng, Judy Hoffman, Kate Saenko, Trevor Darrell, **"Adversarial Discriminative Domain Adaptation"**, CVPR 2017

**논문 내 인용 문헌**:
- Ganin & Lempitsky, "Unsupervised domain adaptation by backpropagation," ICML 2015 [11]
- Tzeng et al., "Simultaneous deep transfer across domains and tasks," ICCV 2015 [12]
- Liu & Tuzel, "Coupled generative adversarial networks," CoRR 2016 [13]
- Goodfellow et al., "Generative adversarial nets," NeurIPS 2014 [10]
- Long & Wang, "Learning transferable features with deep adaptation networks," ICML 2015 [6]
- Ganin et al., "Domain-adversarial training of neural networks," JMLR 2016 [19]

**2020년 이후 비교 논문** (논문 외부 참조, 내용에 확신이 있는 범위만 기재):
- Liang et al., "Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation," ICML 2020
- Long et al., "Conditional Adversarial Domain Adaptation," NeurIPS 2018
- Xu et al., "CDTrans: Cross-domain Transformer for Unsupervised Domain Adaptation," ICLR 2022
