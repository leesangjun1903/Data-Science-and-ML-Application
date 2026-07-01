# Factorized Adversarial Networks (FAN) for Unsupervised Domain Adaptation 

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

FAN(Factorized Adversarial Networks)은 비지도 도메인 적응(Unsupervised Domain Adaptation, UDA) 문제를 해결하기 위해 **잠재 특징 공간(latent feature space)을 두 개의 보완적 서브공간으로 분리(factorize)**하는 것이 핵심 아이디어입니다.

- **도메인 특화 서브공간(Domain-Specific Subspace, DSS)**: 도메인 고유 특성(배경, 화질 등) 저장
- **태스크 특화 서브공간(Task-Specific Subspace, TSS)**: 분류에 필요한 카테고리 정보 저장

그 후, TSS에 대해서만 적대적 학습(adversarial training)을 적용하여 소스-타겟 도메인 간 분포 불일치를 최소화합니다.

### 주요 기여 (3가지)

| 기여 | 내용 |
|------|------|
| ① 새로운 프레임워크 | 특징 공간 분리(factorization)와 적대적 학습을 통합한 FAN 제안 |
| ② 아키텍처 분석 | 4가지 네트워크 구조 비교 및 분리된 서브공간 시각화 |
| ③ 실험적 검증 | 벤치마크 데이터셋(MNIST, USPS, SVHN) 및 대규모 실세계 태깅 데이터셋에서 SOTA 달성 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

딥러닝 모델을 소스 도메인에서 학습한 후 타겟 도메인에 적용할 때 발생하는 **도메인 시프트(domain shift)** 문제를 해결합니다.

- **설정**: 소스 도메인은 레이블이 있는 대량 데이터 보유, 타겟 도메인은 레이블 없음
- **목표**: 타겟 도메인에서 레이블 없이 효과적인 이미지 분류 수행
- **기존 방법의 한계**:
  - 파인튜닝: 타겟 도메인 레이블 필요 → 비용↑
  - 합성 데이터 생성: 실제 데이터 분포와 차이 존재
  - ADDA/DSN: 전체 특징 공간에 적응 → 도메인 무관 분류 정보와 도메인 특화 정보가 혼재

### 2.2 제안하는 방법 (수식 포함)

#### Stage 1: 소스 도메인 특징 분리 학습

인코더 $Enc(\mathbf{x}; \theta_e)$가 입력 $\mathbf{x}$를 잠재 특징 $\mathbf{h}$로 인코딩하고, 이를 두 부분으로 분리합니다:

$$\mathbf{h} = [\mathbf{h}_d, \mathbf{h}_t]$$

여기서 $\mathbf{h}_d$는 도메인 특화 특징, $\mathbf{h}_t$는 태스크 특화 특징입니다.

태스크 특화 특징에서 로짓(logit) 공간으로의 매핑:

$$\mathbf{h}_l = M(\mathbf{h}_t; \theta_m)$$

소스 도메인 전체 목적 함수:

$$\mathcal{L}_{\text{source}} = \alpha \mathcal{L}_c + \beta \mathcal{L}_m + \mathcal{L}_r \tag{1}$$

**(a) 분류 손실 (Cross-Entropy Loss)**:

$$\mathcal{L}_c = -\sum_{i=1}^{N} \mathbf{y}_i \cdot \log \hat{\mathbf{y}}_i \tag{2}$$

여기서 $\hat{\mathbf{y}} = \text{softmax}(M(\mathbf{h}_t; \theta_m))$

**(b) 상호 정보 손실 (Mutual Information Loss)** — 두 서브공간의 직교성 강제:

$$\mathcal{L}_m = \sum_{i=1}^{N} \left\| \mathbf{h}_{ti}^{\mathbf{T}} \mathbf{h}_{di} \right\|^2 \tag{3}$$

**(c) 재구성 손실 (Reconstruction Loss)**:

$$\mathcal{L}_r = \sum_{i=1}^{N} \left\| \mathbf{x}_i - Dec(\mathbf{h}_{di}, \mathbf{h}_{li}; \theta_d) \right\|^2 \tag{4}$$

> **핵심 설계 의도**: $\mathcal{L}_c$는 $\mathbf{h}_t$가 분류 정보를 유지하도록 하고, $\mathcal{L}_r$은 $\mathbf{h}_d$가 도메인 특화 정보를 담도록 유도하며, $\mathcal{L}_m$은 두 공간을 직교하게 분리합니다.

#### Stage 2: 타겟 도메인 적대적 적응

타겟 도메인 목적 함수:

$$\mathcal{L}_{\text{target}} = \mu \mathcal{L}_{\text{adv}_D} + \nu \mathcal{L}_{\text{adv}_M} + \mathcal{L}_r \tag{5}$$

**판별자(Discriminator) $D$ 최적화**:

$$\min_{D} \mathcal{L}_{\text{adv}_D} = -\mathbb{E}_{\mathbf{x}_s \sim \mathcal{S}} \log D(M^s(\mathbf{h}_t^s; \theta_m^s)) - \mathbb{E}_{\mathbf{x}_t \sim \mathcal{T}} \log(1 - D(M^t(\mathbf{h}_t^t; \theta_m^t))) \tag{6}$$

**타겟 도메인 인코더 최적화**:

$$\min_{\Theta} \mathcal{L}_{\text{adv}_M} = -\mathbb{E}_{\mathbf{x}_t \sim \mathcal{T}} \log(D(M^t(\mathbf{h}_t^t; \theta_m^t))) \tag{7}$$

판별자는 소스/타겟 도메인의 로짓(logit) 공간을 구분하도록 학습되고, 타겟 인코더는 판별자를 속이도록 학습됩니다.

### 2.3 모델 구조

```
[소스 도메인 입력]
       ↓
   [Encoder (CNN)]
       ↓
   ┌───┴────┐
   ↓        ↓
  h_d      h_t         ← 특징 분리
   │        │
   │    [Mapping M]
   │        ↓
   │       h_l (logit)
   │        │
   └──┬─────┘
      ↓
   [Decoder]   → 재구성 출력 (L_r)
   
h_l → [Discriminator] ← h_l (타겟)
         ↑↓ 적대적 학습
[타겟 도메인 인코더]
```

- **소스 도메인**: LeNet(digits) / ResNet-50(실세계) 기반 인코더 + 디코더(Deconv)
- **타겟 도메인**: 소스 네트워크 구조와 동일하나, 가중치는 소스에서 초기화 후 독립적으로 학습
- **판별자**: 3개의 FC 레이어 (500-500-1 for digits, 1024-2048-1 for real-world)
- **비대칭 학습**: 소스 네트워크 고정 후 타겟 네트워크 학습

### 2.4 성능 향상

**Digits 데이터셋 결과:**

| Method | MNIST→USPS | USPS→MNIST | SVHN→MNIST |
|--------|-----------|-----------|-----------|
| Baseline | 0.752 | 0.571 | 0.601 |
| ADDA | 0.894 | 0.901 | 0.760 |
| **FAN (Ours)** | **0.921** | **0.910** | **0.925** |
| FAN (full) | **0.963** | **0.971** | — |

특히 SVHN→MNIST에서 ADDA 대비 **+16.5%p** 성능 향상이 두드러집니다.

**Ablation Study (SVHN→MNIST):**

| Network 구조 | 정확도 |
|-------------|--------|
| Joint feature | 0.829 |
| Feature separation | 0.858 |
| Feature concatenation | 0.905 |
| **Full factorization (FAN)** | **0.925** |

**실세계 태깅 데이터셋 (Crawling→Mobile):**

| Method | Top-1 | Top-5 |
|--------|-------|-------|
| No adaptation | 0.3571 | 0.6607 |
| ADDA (full) | 0.4386 | 0.7533 |
| **FAN (full)** | **0.4632** | **0.7838** |

### 2.5 한계점

1. **2단계 학습의 복잡성**: 소스 네트워크 고정 후 타겟 학습이라는 순차적 구조로 인해 end-to-end 최적화가 어려움
2. **하이퍼파라미터 민감성**: $\alpha, \beta, \mu, \nu$ 등 여러 균형 파라미터 수동 조정 필요
3. **직교성 제약의 근사성**: $\mathcal{L}_m$이 내적의 제곱 합으로 직교성을 근사하지만 완전한 분리 보장 불가
4. **이미지 분류 태스크에 한정**: 논문 자체에서 "향후 다른 비전 태스크로 확장하겠다"고 언급
5. **벤치마크 규모**: Digits 데이터셋은 상대적으로 소규모이며 단순한 도메인 차이를 가짐
6. **GAN 학습 불안정성**: 적대적 학습의 내재적 불안정성 문제

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화 성능 향상의 핵심 메커니즘

FAN의 일반화 성능 향상은 다음 세 가지 메커니즘의 상호작용에서 비롯됩니다:

**(1) 도메인 불변 특징 추출을 통한 일반화**

$$\mathbf{h}_t^* = \arg\min_{\mathbf{h}_t} \mathcal{L}_c(\mathbf{h}_t) + \beta \mathcal{L}_m(\mathbf{h}_t, \mathbf{h}_d)$$

태스크 특화 서브공간이 분류 손실만을 최소화하고 도메인 특화 정보를 배제하므로, 도메인 간 분포 차이가 줄어들어 새로운 도메인에 대한 일반화 능력이 향상됩니다.

**(2) 타겟 데이터 규모 증가에 따른 성능 향상**

실세계 데이터셋 실험에서 타겟 훈련 데이터의 비율을 달리했을 때:

$$\text{Top-1: } 10\% \rightarrow 50\% \rightarrow 100\%: \quad 0.3946 \rightarrow 0.4041 \rightarrow 0.4632$$

이는 FAN이 더 많은 비레이블 타겟 데이터를 활용할수록 분포 불일치를 더 효과적으로 감소시켜 일반화 성능이 개선됨을 보여줍니다.

**(3) 재구성 손실의 정규화 효과**

타겟 도메인에서 재구성 손실 $\mathcal{L}_r$을 사용함으로써, 타겟 인코더가 도메인 특화 서브공간에 타겟 도메인의 구조적 정보를 보존하도록 제약합니다. 이는 과적합(overfitting)을 방지하는 정규화 역할을 합니다.

### 3.2 t-SNE 시각화를 통한 일반화 능력 확인

- **적응 전**: 소스(SVHN)와 타겟(MNIST)의 로짓 공간이 명확히 분리됨
- **적응 후**: 타겟 도메인 샘플들이 소스 도메인 클러스터와 일치하는 군집 형성
- **도메인 특화 서브공간**: 적응 후 두 도메인의 특화 공간이 더욱 명확히 분리 → 분리가 잘 되었음을 증명

### 3.3 일반화 한계와 잠재적 개선 방향

- **클래스 불균형 문제**: TSS 적응 시 특정 클래스의 분포 불일치가 더 클 경우 처리 방안 미제시
- **멀티소스 도메인 일반화**: 단일 소스-타겟 쌍만 다루며, 여러 소스를 결합한 일반화는 미탐구
- **오픈셋 도메인 적응**: 소스와 타겟이 동일한 클래스를 공유한다고 가정하여 실제 환경의 새로운 클래스 처리 불가

---

## 4. 앞으로의 연구에 미치는 영향과 고려할 점

### 4.1 연구에 미치는 영향

**(1) 특징 분리(Disentanglement) 기반 도메인 적응의 방향 제시**

FAN은 단순히 전체 특징 공간을 정렬하는 것보다 **목적에 따른 특징 분리 후 부분적 정렬**이 더 효과적임을 실험적으로 입증했습니다. 이는 이후 DIRT-T, MCC, ToAlign 등의 연구에서 클래스 조건부 정렬, 전이 가능한 특징 선택 등의 아이디어로 발전됩니다.

**(2) 재구성 기반 자기지도 학습과의 결합**

타겟 도메인에서 레이블 없이 재구성 손실로 구조적 정보를 학습하는 방식은, 이후 자기지도학습(Self-Supervised Learning) 기반 도메인 적응 연구의 선행 개념으로 볼 수 있습니다.

**(3) 대규모 실세계 데이터셋 벤치마크 제시**

기존 연구가 MNIST, Office 등 소규모 데이터셋에 집중할 때, 15만 장 이상의 실세계 태깅 데이터셋을 구축하고 평가함으로써 실용적 도메인 적응 연구의 필요성을 강조했습니다.

### 4.2 향후 연구 시 고려할 점

**(1) 완전한 End-to-End 학습**

현재 2단계 학습 구조를 단일 최적화 문제로 통합하면 더 효율적이고 안정적인 학습이 가능합니다.

**(2) 의미론적(Semantic) 정렬 강화**

현재 FAN은 분포 수준의 정렬에 초점을 맞추지만, 클래스 조건부(class-conditional) 정렬을 추가하면 negative transfer를 방지할 수 있습니다:

$$\mathcal{L}_{\text{semantic}} = \sum_{k=1}^{K} d\left(P(\mathbf{h}_t | y=k)_{\mathcal{S}},\ P(\mathbf{h}_t | y=k)_{\mathcal{T}}\right)$$

**(3) 사전학습 모델(Pretrained Model) 활용**

ViT(Vision Transformer), CLIP 등의 강력한 사전학습 표현과 FAN의 분리 메커니즘을 결합하면 더 강력한 도메인 일반화 성능을 기대할 수 있습니다.

**(4) 직교성 제약의 개선**

단순 내적 기반 $\mathcal{L}_m$ 대신 정보 이론적으로 더 엄밀한 독립성 측도(예: Total Correlation, HSIC)를 사용하여 분리 품질을 향상시킬 수 있습니다.

**(5) 오픈셋 및 멀티소스 시나리오**

실제 환경에서는 소스-타겟 간 클래스 집합이 다를 수 있으므로, partial/open-set domain adaptation으로의 확장이 필요합니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

아래 비교는 FAN과 직접적으로 관련된 주요 후속 연구들에 대한 분석이며, 일부 내용은 제 학습 데이터 기반의 정보임을 참고하시기 바랍니다.

### 5.1 주요 후속 연구 비교표

| 논문 | 핵심 아이디어 | FAN과의 차이점 | 성능(SVHN→MNIST) |
|------|-------------|--------------|-----------------|
| **SHOT** (Liang et al., ICML 2020) | 소스 없이 타겟만으로 정보 극대화 | 소스 모델 고정, 타겟 엔트로피 최소화 | ~0.982 |
| **MCC** (Jin et al., ECCV 2020) | 클래스 혼동 행렬 기반 정렬 | 클래스 조건부 분포 정렬 강화 | ~0.956 |
| **DANN + Self-Training** (Liu et al., 2021) | 의사 레이블(pseudo-label)과 결합 | 반복적 자기학습으로 일반화 향상 | ~0.970+ |
| **CDTrans** (Xu et al., ICLR 2022) | Transformer 기반 도메인 적응 | Cross-Domain Attention으로 특징 정렬 | N/A (Office-Home 기준) |
| **PMTrans** (Zhu et al., ECCV 2022) | Patch Mix + Transformer | Patch-level 도메인 정렬 | N/A |

### 5.2 핵심 발전 방향과 FAN의 위치

```
FAN (2018)                   SHOT (2020)              CDTrans (2022)
특징 분리 +          →       소스-free 적응    →      Transformer 기반
적대적 정렬                   + 자기지도 학습           의미론적 정렬
     ↓                              ↓                       ↓
도메인 특화/태스크 특화       타겟 중심 최적화         패치 레벨 세밀한 정렬
서브공간 분리의 선구          (소스 접근 불필요)         (대규모 사전학습 활용)
```

### 5.3 분석: FAN의 강점과 한계 (2020년 이후 관점)

**강점 유지 영역:**
- 특징 분리 아이디어는 **Domain-Specific Batch Normalization** (Chang et al., 2019), **MixStyle** (Zhou et al., ICLR 2021) 등으로 계승
- 재구성 손실 기반 자기지도 신호는 **SSL 기반 UDA** 연구의 토대

**2020년 이후 연구가 개선한 부분:**
- **소스-free 도메인 적응**: SHOT(2020)은 소스 데이터 없이도 적응 가능 → FAN의 소스 데이터 의존성 극복
- **클래스 조건부 정렬**: FAN은 전체 분포 정렬 → 이후 연구들은 클래스별 정렬로 negative transfer 방지
- **사전학습 모델 활용**: CLIP, ViT 기반 모델들이 FAN이 LeNet/ResNet-50으로 달성한 성능을 큰 폭으로 초과

---

## 참고 자료

**주요 참고 논문 (PDF 원문 기반):**
- **Jian Ren, Jianchao Yang, Ning Xu, David J. Foran**, "Factorized Adversarial Networks for Unsupervised Domain Adaptation," arXiv:1806.01376v1, 2018.

**논문 내 인용 문헌 (원문 References):**
- Goodfellow et al., "Generative Adversarial Nets," NeurIPS 2014 [9]
- Bousmalis et al., "Domain Separation Networks," NeurIPS 2016 [18]
- Tzeng et al., "Adversarial Discriminative Domain Adaptation (ADDA)," arXiv:1702.05464, 2017 [24]
- Ganin & Lempitsky, "Unsupervised Domain Adaptation by Backpropagation (RevGrad)," ICML 2015 [23]
- He et al., "Deep Residual Learning for Image Recognition (ResNet)," arXiv:1512.03385, 2015 [13]
- Ghifary et al., "Deep Reconstruction-Classification Networks (DRCN)," ECCV 2016 [19]
- Liu & Tuzel, "Coupled Generative Adversarial Networks (CoGAN)," NeurIPS 2016 [25]

**2020년 이후 비교 연구 (학습 데이터 기반, 원문 미확인):**
- Liang et al., "Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation (SHOT)," ICML 2020
- Jin et al., "Minimum Class Confusion for Versatile Domain Adaptation (MCC)," ECCV 2020
- Xu et al., "CDTrans: Cross-domain Transformer for Unsupervised Domain Adaptation," ICLR 2022
- Zhou et al., "Domain Generalization with MixStyle," ICLR 2021

> ⚠️ **정확도 주의**: 2020년 이후 최신 연구 비교 부분의 성능 수치 일부는 제 학습 데이터의 기억에 기반하며, 정확한 수치 확인을 위해서는 각 논문의 원문을 직접 참조하시기 바랍니다.
