# Instance Adaptive Self-Training for Unsupervised Domain Adaptation (IAST)

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장
기존 Self-Training(ST) 기반 UDA 방법들은 **글로벌/클래스 단위의 고정된 임계값**을 사용하여 pseudo-label을 생성하기 때문에, 예측 점수가 낮은 "hard" 이미지의 핵심 정보를 무시하고 데이터 다양성이 부족하다는 문제를 가진다. IAST는 **인스턴스 단위의 적응적 임계값**과 **영역 기반 정규화**를 결합하여 이를 해결한다.

### 주요 기여 3가지

| 기여 | 내용 |
|------|------|
| **Instance Adaptive Selector (IAS)** | 이미지(인스턴스) 단위로 클래스별 임계값을 적응적으로 조정하여 pseudo-label 품질 향상 |
| **Region-Guided Regularization** | pseudo-label 영역(confidence region)은 smoothing, 비pseudo-label 영역(ignored region)은 sharpening |
| **확장성 (Scalability)** | 모델 구조 의존성 없이 다른 UDA 방법(AT 등)에 plug-in 방식으로 적용 가능 |

---

## 2. 상세 분석

### 2-1. 해결하고자 하는 문제

**문제 1: 기존 pseudo-label 생성의 한계**
- **상수 임계값(Constant threshold)**: 모든 클래스, 모든 이미지에 동일한 $\theta = 0.9$ 적용 → "easy" 클래스 편향
- **클래스 균형 임계값(CBST)**: 전체 이미지 집합에서 클래스별 동일 임계값 → 예측 점수가 낮은 "hard" 이미지 무시
- 결과: 정보 중복(information redundancy)과 pseudo-label 다양성 부족

**문제 2: 정규화의 불완전성**
- 기존 CRST[35]는 pseudo-label 영역에만 정규화를 적용 → 비pseudo-label 영역의 학습 신호 부재

---

### 2-2. 제안하는 방법 (수식 포함)

#### 전체 목적 함수

$$\min_{\mathbf{w}} \mathcal{L}_{CE}(\mathbf{w}, \hat{\mathbb{Y}}_T) + \mathcal{L}_R(\mathbf{w}) = \mathcal{L}_{CE}(\mathbf{w}, \hat{\mathbb{Y}}_T) + (\lambda_i \mathcal{R}_i(\mathbf{w}) + \lambda_c \mathcal{R}_c(\mathbf{w})) \tag{3}$$

#### Self-Training 기본 손실 함수

$$\min_{\mathbf{w}} \mathcal{L}_{CE} = -\frac{1}{|\mathbb{X}_S|} \sum_{\mathbf{x}_s \in \mathbb{X}_S} \sum_{c=1}^{C} y_s^{(c)} \log p(c|\mathbf{x}_s, \mathbf{w}) - \frac{1}{|\mathbb{X}_T|} \sum_{\mathbf{x}_t \in \mathbb{X}_T} \sum_{c=1}^{C} \hat{y}_t^{(c)} \log p(c|\mathbf{x}_t, \mathbf{w}) \tag{1}$$

#### Adversarial Training (Warm-up) 손실 함수

$$\min_{\mathbf{w}} \max_{\mathbf{D}} \mathcal{L}_{AT} = -\frac{1}{|\mathbb{X}_S|} \sum_{\mathbf{x}_s \in \mathbb{X}_S} \sum_{c=1}^{C} y_s^{(c)} \log p(c|\mathbf{x}_s, \mathbf{w}) + \frac{\lambda_{adv}}{|\mathbb{X}_T|} \sum_{\mathbf{x}_t \in \mathbb{X}_T} [\mathbf{D}(\mathbf{M}(\mathbf{x}_t, \mathbf{w})) - 1]^2 \tag{2}$$

---

#### (A) Instance Adaptive Selector (IAS)

**Pseudo-label 생성 기준:**

$$\hat{y}_t^{(c)} = \begin{cases} 1, & \text{if } c = \arg\max_c p(c|\mathbf{x}_t, \mathbf{w}) \text{ and } p(c|\mathbf{x}_t, \mathbf{w}) > \theta^{(c)} \\ 0, & \text{otherwise} \end{cases} \tag{5}$$

**EMA(Exponential Moving Average) 임계값 업데이트:**

$$\theta_t^{(c)} = \beta \theta_{t-1}^{(c)} + (1 - \beta)\Psi(\mathbf{x}_t, \theta_{t-1}^{(c)}) \tag{6}$$

- $\beta$: momentum factor (과거 임계값 정보 보존 비율)
- $\beta$가 클수록 임계값이 더 smooth하게 변화

**인스턴스별 로컬 임계값 계산:**

$$\Psi(\mathbf{x}_t, \theta_{t-1}^{(c)}) = \mathbb{P}_{\mathbf{x}_t}^{(c)} \left[ \alpha \theta_{t-1}^{(c)^{\gamma}} |\mathbb{P}_{\mathbf{x}_t}^{(c)}| \right] \tag{7}$$

- $\alpha$: pseudo-label로 선택할 상위 비율
- $\gamma$: "Hard" class weight decay 파라미터
- $\theta_{t-1}^{(c)^{\gamma}}$: "hard" 클래스(낮은 $\theta$)는 더 많이 감쇠, "easy" 클래스(높은 $\theta$)는 약하게 감쇠

> **직관**: 인스턴스 $x_t$에 대해 클래스 $c$의 예측 확률을 내림차순 정렬 후, 상위 $\alpha \times 100\%$ 위치의 확률값을 로컬 임계값으로 사용. 이를 EMA로 글로벌 임계값과 결합.

---

#### (B) Region-Guided Regularization

**Confidence Region (pseudo-label 영역) - KLD 최소화:**

$$\mathcal{R}_c = -\frac{1}{|\mathbb{X}_T|} \sum_{\mathbf{x}_t \in \mathbb{X}_T} \mathbb{I}_{\mathbf{x}_t} \sum_{c=1}^{C} \frac{1}{C} \log p(c|\mathbf{x}_t, \mathbf{w}) \tag{8}$$

- 예측이 균등 분포에 가까워질수록 $\mathcal{R}_c$ 감소
- 모델이 noisy pseudo-label에 과적합되지 않도록 **smoothing** 효과

**Ignored Region (비pseudo-label 영역) - 엔트로피 최소화:**

$$\mathcal{R}_i = -\frac{1}{|\mathbb{X}_T|} \sum_{\mathbf{x}_t \in \mathbb{X}_T} \mathbb{I}_{\mathbf{x}_t}^{\complement} \sum_{c=1}^{C} p(c|\mathbf{x}_t, \mathbf{w}) \log p(c|\mathbf{x}_t, \mathbf{w}) \tag{9}$$

- 감독 신호가 없는 영역에서 모델이 더 **sharp**한 예측을 하도록 유도
- 엔트로피를 최소화하여 저신뢰도 영역에서도 의미있는 특징 학습 촉진

---

### 2-3. 모델 구조 (3단계 학습 파이프라인)

```
┌─────────────────────────────────────────────────────────────┐
│ Phase (a): Warm-up                                          │
│   {X_S, Y_S, X_T} → Adversarial Training → G₀ (초기 모델) │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase (b): Pseudo-Label Generation                          │
│   G → 예측 생성 → IAS (인스턴스별 적응 임계값) → ŷ_t      │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase (c): Self-Training                                    │
│   M 학습: L_CE(ŷ_t) + λ_i R_i + λ_c R_c                   │
│   → M의 파라미터를 G에 복사 → Phase (b)로 반복 (3 rounds)  │
└─────────────────────────────────────────────────────────────┘
```

**네트워크**: DeepLab-v2 + ResNet-101 백본  
**하이퍼파라미터**: $\alpha=0.2$, $\beta=0.9$, $\gamma=8.0$, $\lambda_i=3.0$, $\lambda_c=0.1$

---

### 2-4. 성능 향상

#### GTA5 → Cityscapes (mIoU %)

| 방법 | 유형 | mIoU |
|------|------|------|
| AdaptSegNet [27] | AT | 42.4 |
| AdvEnt [29] | AT | 45.4 |
| CBST [36] | ST | 45.9 |
| MRKLD [35] | ST | 47.1 |
| BLF [19] | AT+ST | 48.5 |
| **IAST (ours)** | **AT+ST** | **51.5** |
| **IAST-MST (ours)** | **AT+ST+Multi-scale** | **52.2** |

#### SYNTHIA → Cityscapes (mIoU* %)

| 방법 | mIoU* (13 classes) |
|------|-------------------|
| AdaptMR [34] | 53.8 |
| **IAST (ours)** | **57.0** |

#### Ablation Study (GTA5 → Cityscapes)

| 구성 | mIoU | 향상 |
|------|------|------|
| Source only | 35.6 | - |
| + Warm-up | 43.8 | +8.2 |
| + Constant ST | 45.1 | +1.3 |
| + IAS | 49.8 | +4.7 |
| + $\mathcal{R}_c$ | 50.7 | +0.9 |
| + $\mathcal{R}_i$ | **51.5** | +0.8 |

---

### 2-5. 한계

1. **벤치마크 한정**: 합성→실제(synthetic-to-real) 시나리오에만 집중. 다른 도메인 쌍(예: 날씨 변화, 의료 영상 등)에서의 검증 부족
2. **세그멘테이션 태스크 한정**: 분류, 객체 검출 등 다른 태스크에 대한 직접적 검증 없음
3. **하이퍼파라미터 민감성**: $\alpha$, $\beta$, $\gamma$, $\lambda_i$, $\lambda_c$ 등 조정해야 할 파라미터가 많음
4. **순차적 처리**: IAS가 인스턴스를 순차적으로 처리하여 병렬화에 제약이 있을 수 있음
5. **Warm-up 단계의 AT 의존성**: 완전한 ST만으로는 성능이 제한될 수 있음 (AT warm-up 필요)

---

## 3. 모델 일반화 성능 향상 가능성

IAST가 일반화 성능을 향상시키는 메커니즘은 세 가지 관점에서 분석할 수 있다.

### 3-1. Pseudo-Label 다양성을 통한 일반화

기존 CBST는 전체 데이터셋 기준으로 상위 20% 픽셀을 선택하므로, "easy" 클래스(도로, 하늘 등)의 픽셀이 pseudo-label을 독점한다. IAS는 **각 인스턴스 내에서** 상위 $\alpha$비율을 선택하므로, 상대적으로 어려운 이미지에서도 pseudo-label이 생성된다.

$$\text{IAS의 pseudo-label 비율} = 36.5\% \quad \text{vs} \quad \text{CBST} = 20.0\%$$

이 다양성 증가가 **도메인 일반화 능력 향상**의 핵심이다.

### 3-2. Hard Class Weight Decay를 통한 균형 학습

$\Psi(\mathbf{x}\_t, \theta_{t-1}^{(c)}) = \mathbb{P}\_{\mathbf{x}_t}^{(c)}\left[\alpha\theta\_{t-1}^{(c)^{\gamma}}|\mathbb{P}\_{\mathbf{x}_t}^{(c)}|\right]$에서 $\gamma=8$로 설정 시, "hard" 클래스의 $\theta^{(c)}$가 작기 때문에 $\theta^{(c)^\gamma}$는 더욱 작아진다. 이는 "hard" 클래스의 pseudo-label 선택 비율을 줄여 **노이즈를 억제**하고, 역설적으로 모델이 더 신뢰할 수 있는 "hard" 클래스 특징을 학습하게 한다.

### 3-3. Region-Guided Regularization을 통한 과적합 방지

| 영역 | 정규화 방식 | 일반화 기여 |
|------|------------|-------------|
| Confidence region | KLD 최소화 ($\mathcal{R}_c$) | Noisy pseudo-label 과적합 방지 → label smoothing 효과 |
| Ignored region | 엔트로피 최소화 ($\mathcal{R}_i$) | 감독 신호 없는 영역에서도 low-entropy 예측 유도 → 미지 영역 일반화 |

이 두 가지 정규화는 **상보적(complementary)**으로 작동하여, 모델이 pseudo-label에 과적합되지 않으면서도 타겟 도메인의 구조적 정보를 최대한 흡수하도록 한다.

### 3-4. 다른 UDA 방법과의 결합을 통한 일반화

IAST를 다른 방법에 적용 시 성능 향상:

| 기본 방법 | GTA5 Base | GTA5 +IAST | $\Delta$ |
|-----------|----------|-----------|---------|
| AdaptSeg [27] | 42.4 | 50.2 | **+7.8** |
| AdvEnt [29] | 45.4 | 49.8 | **+4.4** |
| Source only | 35.6 | 48.8 | **+13.2** |

이는 IAST가 기존 방법들의 일반화 성능을 **플러그인 방식으로 향상**시킬 수 있음을 의미하며, 방법론적 일반성(generality)을 입증한다.

### 3-5. 반지도 학습으로의 확장

| 방법 | 1/8 labeled | 1/4 labeled | 1/2 labeled |
|------|------------|------------|------------|
| AdvSemi [11] | 58.8 | 62.3 | 65.7 |
| **IAST** | **64.6** | **66.7** | **69.8** |

레이블 데이터가 적을수록 IAST의 이점이 더 크게 나타나며, 이는 제한된 감독 상황에서의 강력한 일반화 능력을 시사한다.

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4-1. 향후 연구에 미치는 영향

**① Self-Training의 재조명**

IAST는 AT+ST 조합이 AT 단독(43.7 mIoU)보다 월등히 우수함을 체계적으로 입증했다. 이후 연구들이 self-training을 UDA의 핵심 전략으로 다시 주목하게 하는 계기가 되었으며, pseudo-label 품질 향상이 핵심 연구 방향으로 자리잡게 했다.

**② 인스턴스/이미지 단위 적응의 패러다임 전환**

클래스 수준이나 전역 수준의 일괄 처리에서 벗어나, 각 샘플의 특성을 반영한 **인스턴스 적응형** 처리의 중요성을 강조했다. 이는 이후 연구들에서 샘플별 신뢰도 추정, 동적 임계값 방법론으로 발전했다.

**③ 정규화의 이원적 적용**

Pseudo-label 영역과 비pseudo-label 영역을 **구분하여 다르게 정규화**하는 아이디어는 이후 분할 학습에서의 영역별 처리 전략에 영감을 주었다.

**④ 확장성 중심의 설계 철학**

특정 네트워크 구조에 종속되지 않는 "decorator" 패턴의 프레임워크 설계는 이후 UDA 연구에서 모듈형 설계의 중요성을 부각시켰다.

---

### 4-2. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의**: 아래 비교는 제공된 논문(IAST, 2020.08)과의 연관성을 기반으로 하며, 해당 후속 논문들의 원문을 직접 확인하지 않은 경우 내용의 일부가 불정확할 수 있습니다. 가능한 한 원문 논문을 직접 참조하시기 바랍니다.

| 연구 방향 | 관련 후속 연구 | IAST와의 차이점 |
|-----------|--------------|----------------|
| **Teacher-Student 기반 ST** | DAFormer (CVPR 2022), MIC (CVPR 2023) | Transformer 백본 사용, 더 강력한 data augmentation과 EMA teacher 결합 |
| **신뢰도 기반 pseudo-label 선택** | SIM (ICLR 2021), ProDA (CVPR 2021) | 프로토타입 기반 pseudo-label 정제, 분포 정렬 방식 |
| **Contrastive Learning + UDA** | CPSL (CVPR 2022), CTF (ECCV 2022) | Contrastive loss를 UDA에 도입, 클래스 경계 학습 강화 |
| **Vision Transformer 기반 UDA** | DAFormer, SegFormer-UDA | IAST의 ResNet-101 대비 훨씬 강력한 표현력, GTA5→Cityscapes에서 60%+ mIoU 달성 |

**특히 주목할 비교: IAST vs DAFormer (추정)**
- IAST: 52.2 mIoU (GTA5→Cityscapes, ResNet-101)
- DAFormer: ~68.3 mIoU (GTA5→Cityscapes, Transformer 기반) — 백본의 차이가 성능 격차의 주요 원인

이는 IAST의 한계인 **백본 아키텍처 의존성**을 보여주며, IAST의 방법론을 Transformer 백본에 적용하는 것이 유망한 연구 방향임을 시사한다.

---

### 4-3. 앞으로 연구 시 고려할 점

**① 더 강력한 백본과의 결합**
- IAST는 DeepLab-v2 + ResNet-101에 한정. SegFormer, Swin Transformer 등 최신 아키텍처에 IAS와 region-guided regularization을 적용하면 추가적인 성능 향상 가능

**② 동적 하이퍼파라미터 스케줄링**
- $\alpha$, $\beta$, $\gamma$, $\lambda_i$, $\lambda_c$의 학습 과정에서의 동적 조정 메커니즘 연구 필요 (현재는 고정값 사용)

**③ Noisy pseudo-label에 대한 더 정교한 처리**
- 현재 KLD smoothing만으로는 부분적. **Noise-robust loss functions** (예: GCE loss, SL loss) 또는 **신뢰도 점수 기반 가중치** 적용 검토

**④ 멀티 도메인 및 도메인 연속 적응**
- 현재는 단일 소스→단일 타겟. **Multi-source** 또는 **continual domain adaptation** 시나리오로의 확장

**⑤ 객체 검출, 깊이 추정 등 다른 태스크로의 일반화**
- semantic segmentation에 특화된 현재 구조를 다른 dense prediction 태스크에 적용하는 연구 필요

**⑥ 클래스 불균형 문제의 근본적 해결**
- HWD가 "hard" 클래스의 pseudo-label을 줄이지만, 근본적인 클래스 불균형은 해결되지 않음. **클래스 균형 샘플링**과의 결합 또는 **focal loss** 적용 고려

**⑦ 이론적 보장(Theoretical Guarantee)**
- EMA 임계값이 수렴하는 조건, pseudo-label 품질과 최종 성능의 관계에 대한 이론적 분석 부재. PAC-learning 프레임워크 등을 통한 이론적 근거 마련 필요

---

## 참고자료

- **주 논문**: Ke Mei, Chuang Zhu, Jiaqi Zou, Shanghang Zhang, "Instance Adaptive Self-Training for Unsupervised Domain Adaptation," arXiv:2008.12197v1, 2020. (제공된 PDF)
- **비교 논문들** (논문 내 인용 기준):
  - Zou et al., "Unsupervised Domain Adaptation for Semantic Segmentation via Class-Balanced Self-Training (CBST)," ECCV 2018 [36]
  - Zou et al., "Confidence Regularized Self-Training (CRST)," ICCV 2019 [35]
  - Tsai et al., "Learning to Adapt Structured Output Space for Semantic Segmentation (AdaptSeg)," CVPR 2018 [27]
  - Vu et al., "AdvEnt: Adversarial Entropy Minimization for Domain Adaptation in Semantic Segmentation," CVPR 2019 [29]
  - Li et al., "Bidirectional Learning for Domain Adaptation of Semantic Segmentation (BLF)," CVPR 2019 [19]
  - Lian et al., "Constructing Self-Motivated Pyramid Curriculums for Cross-Domain Semantic Segmentation (PyCDA)," CVPR 2019 [20]
  - Zheng & Yang, "Unsupervised Scene Adaptation with Memory Regularization In Vivo (AdaptMR)," arXiv:1912.11164 [34]

# Instance Adaptive Self-Training for Unsupervised Domain Adaptation

## 1. 핵심 주장과 주요 기여

Instance Adaptive Self-Training for Unsupervised Domain Adaptation (IAST)는 의미론적 분할(semantic segmentation)을 위한 비지도 도메인 적응(UDA) 문제를 해결하는 자기 학습(self-training) 프레임워크입니다. 본 논문의 핵심 주장은 다음과 같습니다.[1]

**성능 우위성**: 자기 학습(ST) 방법이 적대적 학습(AT) 방법보다 우수하며(ST 47.8% > AT 43.7%), 두 방법을 결합한 혼합 방법(AT+ST 49.0%)이 가장 효과적입니다. IAST는 GTA5 to Cityscapes 벤치마크에서 52.2% mIoU를 달성하여 당시 최고 성능(SOTA)을 기록했습니다.[1]

**확장성과 성능의 균형**: 기존 혼합 방법들은 서브모듈 간의 강한 결합으로 인해 확장성과 유연성이 부족합니다. IAST는 모델 구조나 특수한 의존성이 없어 다른 비자기 학습 UDA 방법에 쉽게 적용할 수 있으며, AdaptSeg와 AdvEnt에 적용 시 각각 7.8%와 4.4%의 성능 향상을 보였습니다.[1]

**주요 기여**:
- **인스턴스 적응형 선택기(Instance Adaptive Selector, IAS)**: 이미지 단위로 각 의미 범주에 대한 적응형 가짜 레이블 임계값을 선택하고, "어려운" 클래스의 비율을 동적으로 감소시켜 가짜 레이블의 노이즈를 제거합니다[1]
- **영역 기반 정규화(Region-guided Regularization)**: 신뢰 영역의 예측을 부드럽게 하고 무시된 영역의 예측을 명확하게 하는 이중 정규화 전략을 제안합니다[1]
- **일반화 가능성**: 반지도 의미론적 분할 작업에도 확장 가능하며, Cityscapes 데이터셋에서 기존 방법들을 크게 능가했습니다[1]

## 2. 문제 정의, 제안 방법, 모델 구조, 및 성능

### 2.1 해결하고자 하는 문제

**도메인 이동(Domain Shift)**: 조명, 객체 시점, 이미지 배경의 차이로 인해 학습 데이터(소스 도메인)와 테스트 데이터(타겟 도메인) 간의 분포 불일치가 발생하며, 이는 레이블이 없는 타겟 도메인에서 성능 저하를 초래합니다.[1]

**가짜 레이블의 품질 문제**: 기존 자기 학습 방법의 주요 장애물은 고품질 가짜 레이블 생성입니다.[1]
- **정보 중복성과 노이즈**: 생성기는 높은 신뢰도를 가진 픽셀만 가짜 레이블로 유지하고 낮은 신뢰도의 픽셀은 무시하는 경향이 있습니다[1]
- **클래스 균형 자기 학습(CBST)의 한계**: CBST는 모든 관련 이미지에서 각 클래스에 대한 순위 기반 참조 신뢰도를 사용하여, 대부분의 픽셀이 낮은 예측 점수를 가진 어려운 이미지의 핵심 정보를 무시합니다[1]

### 2.2 제안하는 방법 (수식 포함)

IAST 프레임워크는 3단계로 구성됩니다:[1]

**(a) 워밍업 단계**: 비자기 학습 방법(예: 적대적 학습)을 사용하여 소스 및 타겟 데이터로 초기 분할 모델 $$M_0$$를 학습합니다.

**(b) 가짜 레이블 생성 단계**: 인스턴스 적응형 선택기(IAS)를 통해 가짜 레이블을 생성합니다.

**(c) 자기 학습 단계**: 타겟 데이터를 사용하여 분할 모델 $$M$$을 학습합니다.

**전체 목적 함수**:

$$
\min_w L_{CE}(w, \hat{Y}_T) + L_R(w) = L_{CE}(w, \hat{Y}_T) + (\lambda_i R_i(w) + \lambda_c R_c(w))
$$

여기서 $$L_{CE}$$는 교차 엔트로피 손실, $$\hat{Y}_T$$는 가짜 레이블 집합, $$R_i$$와 $$R_c$$는 각각 무시된 영역과 신뢰 영역의 정규화이며, $$\lambda_i$$, $$\lambda_c$$는 정규화 가중치입니다.[1]

#### **인스턴스 적응형 선택기 (IAS)**

가짜 레이블 생성 전략:

$$
\hat{y}_t^{(c)} = \begin{cases} 
1, & \text{if } c = \arg\max_c p(c|x_t, w) \text{ and } p(c|x_t, w) > \theta^{(c)} \\
0, & \text{otherwise}
\end{cases}
$$

여기서 $$\theta^{(c)}$$는 클래스 $$c$$에 대한 신뢰도 임계값입니다.[1]

**지수 이동 평균(EMA) 임계값**:

$$
\theta_t^{(c)} = \beta \theta_{t-1}^{(c)} + (1-\beta)\Psi(x_t, \theta_{t-1}^{(c)})
$$

$$
\Psi(x_t, \theta_{t-1}^{(c)}) = P_{x_t}^{(c)}\left[\alpha \frac{\theta_{t-1}^{(c)}}{\gamma |P_{x_t}^{(c)}|}\right]
$$

여기서 $$\beta$$는 모멘텀 인자, $$\alpha$$는 비율 파라미터, $$\gamma$$는 가중치 감쇠 파라미터입니다. 각 인스턴스 $$x_t$$에 대해 각 클래스의 신뢰도 확률을 내림차순으로 정렬하고, $$\alpha \times 100\%$$ 신뢰도 확률을 로컬 임계값 $$\theta_{x_t}^{(c)}$$로 사용합니다.[1]

**"어려운" 클래스 가중치 감쇠(HWD)**: $$\gamma$$를 통해 "어려운" 클래스의 가짜 레이블 비율을 감소시킵니다. "어려운" 클래스의 임계값 $$\theta_{t-1}^{(c)}$$가 낮기 때문에 HWD는 더 많은 가짜 레이블을 감소시킵니다.[1]

#### **영역 기반 정규화**

**신뢰 영역 KLD 최소화**:

$$
R_c = -\frac{1}{|X_T|} \sum_{x_t \in X_T} \sum_{I_{x_t}} \sum_{c=1}^C \frac{1}{C} \log p(c|x_t, w)
$$

여기서 $$I_{x_t} = \{1 | \hat{y}_t^{(h,w)} > 0\}$$는 신뢰 영역입니다[1]. 예측 결과를 균일 분포에 가깝게 만들어 모델이 가짜 레이블에 과적합되는 것을 방지합니다[1].

**무시된 영역 엔트로피 최소화**:

$$
R_i = -\frac{1}{|X_T|} \sum_{x_t \in X_T} \sum_{I_{x_t}^{\complement}} \sum_{c=1}^C p(c|x_t, w) \log p(c|x_t, w)
$$

여기서 $$I_{x_t}^{\complement} = \{1 | \hat{y}_t^{(h,w)} = 0\}$$는 무시된 영역입니다[1]. 무시된 영역의 예측을 "명확하게" 만들어 모델이 감독 신호 없이도 유용한 특징을 학습하도록 촉진합니다[1].

### 2.3 모델 구조

IAST는 Deeplab-v2 아키텍처를 기본 네트워크로 사용하며, ResNet-101을 백본 네트워크로 선택했습니다. 모든 배치 정규화 레이어의 가중치는 고정되었으며, Deeplab-v2는 ImageNet에서 사전 학습되었습니다.[1]

**학습 설정**:
- 최적화기: Adam, 학습률 $$2.5 \times 10^{-5}$$, 배치 크기 6, 4 에폭[1]
- 가짜 레이블 파라미터: $$\alpha = 0.2$$, $$\beta = 0.9$$, $$\gamma = 8.0$$[1]
- 정규화 가중치: $$\lambda_i = 3.0$$, $$\lambda_c = 0.1$$[1]
- 이미지 크기: 1024 × 512, 종횡비 2.0[1]

**다단계 자기 학습**: (b) 단계와 (c) 단계를 한 번 수행하는 것을 1라운드로 계산하며, 실험에서는 총 3라운드를 수행했습니다.[1]

### 2.4 성능 향상

**GTA5 to Cityscapes**:
- IAST: 51.5% mIoU (멀티스케일 테스트 시 52.2%)[1]
- 기존 최고 성능 대비: AdaptSegNet(42.4%) 대비 +9.6%, MRKLD(47.1%) 대비 +4.8%, BLF(48.5%) 대비 +3.7%[1]

**SYNTHIA to Cityscapes**:
- IAST: 49.8% mIoU (16클래스), 57.0% mIoU* (13클래스)[1]
- 기존 최고 성능 대비: AdaptMR(46.5%, 53.8%) 대비 상당한 향상[1]

**절제 연구(Ablation Study)**:
- 워밍업(43.8%) → 상수 ST(45.1%, +1.3%) → IAS 추가(49.8%, +4.7%) → 신뢰 영역 정규화 추가(50.7%, +0.9%) → 무시된 영역 정규화 추가(51.5%, +0.8%)[1]

**반지도 학습 확장**:
Cityscapes 데이터셋에서 1/8, 1/4, 1/2 레이블 비율로 테스트한 결과, IAST는 각각 64.6%, 66.7%, 69.8%의 정확도를 달성하여 기존 방법들을 크게 능가했습니다.[1]

### 2.5 한계

논문에서 명시적으로 언급된 한계는 제한적이지만, 다음과 같은 사항을 고려할 수 있습니다:

**가짜 레이블 품질**: 일련의 고신뢰도 가짜 레이블 생성 기법을 사용했음에도 불구하고, 가짜 레이블의 품질은 여전히 실제 레이블만큼 좋지 않으며, 이는 노이즈 레이블이 여전히 존재함을 의미합니다.[1]

**계산 비용**: 다단계 자기 학습(3라운드)과 멀티스케일 테스트는 추가적인 계산 비용을 요구할 수 있습니다.[1]

**특정 도메인 의존성**: GTA5, SYNTHIA와 같은 합성 데이터에서 Cityscapes와 같은 실제 데이터로의 적응에 초점을 맞추었으며, 다른 도메인 조합에 대한 일반화는 추가 검증이 필요할 수 있습니다.[1]

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 인스턴스 적응형 접근의 일반화 기여

**다양성 증가**: IAS는 이미지 단위로 적응형 임계값을 사용하여 가짜 레이블의 다양성을 증가시킵니다. 클래스 균형 방법은 전체 타겟 세트의 20% 픽셀을 신뢰 영역으로 사용하는 반면, IAS는 각 이미지의 20%를 사용하여 "어려운" 클래스(보행자, 트럭 등)의 정보를 더 많이 포함합니다.[1]

**전역 및 지역 정보 결합**: EMA를 통해 과거의 역사적 정보(전역)와 현재 인스턴스의 정보(지역)를 결합하여 각 인스턴스가 적응형 임계값을 얻습니다. 이는 모델이 다양한 도메인 특성에 더 잘 적응할 수 있게 합니다.[1]

**노이즈 감소**: HWD는 "어려운" 클래스의 가짜 레이블 비율을 동적으로 감소시켜 노이즈 레이블의 영향을 완화하고, 가짜 레이블의 품질을 향상시킵니다. 실험에서 $$\gamma = 8$$일 때 가짜 레이블의 mIoU가 68.2%로 증가했습니다.[1]

### 3.2 영역 기반 정규화의 일반화 효과

**과적합 방지**: 신뢰 영역 KLD 최소화는 모델이 노이즈가 있는 가짜 레이블에 맹목적으로 신뢰하는 것을 방지하고, 예측을 균일 분포에 가깝게 부드럽게 만듭니다. 이는 모델이 새로운 도메인에서도 더 안정적인 예측을 할 수 있게 합니다.[1]

**특징 학습 촉진**: 무시된 영역 엔트로피 최소화는 감독 신호 없이도 낮은 엔트로피의 "명확한" 예측을 유도하여, 모델이 무시된 영역에서 유용한 특징을 학습하도록 촉진합니다. 이는 UDA에서 효과적임이 입증되었습니다.[1]

### 3.3 일반화 가능성의 실증적 증거

**다양한 벤치마크 성능**: IAST는 GTA5 to Cityscapes뿐만 아니라 SYNTHIA to Cityscapes에서도 최고 성능을 달성했으며, 후자의 도메인 간격이 훨씬 크다는 점에서 일반화 능력을 입증했습니다.[1]

**다른 UDA 방법에 적용 가능**: AdaptSeg와 AdvEnt에 IAST를 적용한 결과, 각각 7.8%와 4.4%의 성능 향상을 보였으며, 이는 IAST가 다양한 기본 방법에 일반화될 수 있음을 보여줍니다.[1]

**반지도 학습 확장**: IAST를 반지도 의미론적 분할 작업에 적용한 결과, 다양한 레이블 비율에서 기존 방법들을 능가했으며, 이는 다양한 학습 패러다임에 대한 일반화 능력을 시사합니다.[1]

## 4. 앞으로의 연구에 미치는 영향과 향후 연구 방향

### 4.1 IAST의 영향

**자기 학습의 재조명**: IAST는 자기 학습이 UDA 및 반지도 학습 작업에서 가진 잠재력을 재고하도록 촉진했습니다. 논문 발표 후 412회 인용되었으며, 이는 학계에서 상당한 관심을 받았음을 보여줍니다.[2][1]

**가짜 레이블 품질 개선**: IAST는 인스턴스 적응형 접근을 통해 가짜 레이블의 품질과 다양성을 향상시키는 새로운 방향을 제시했습니다. 이후 연구들은 가짜 레이블 품질 향상 메커니즘을 더욱 발전시켰습니다.[3][4][5][1]

**확장 가능한 프레임워크**: 모델 구조나 특수 의존성이 없는 IAST의 설계는 다른 UDA 방법에 쉽게 통합될 수 있는 "데코레이터" 역할을 할 수 있음을 보여주었습니다. 이는 모듈식 UDA 프레임워크 개발에 영향을 미쳤습니다.[1]

### 4.2 최신 연구 동향과 향후 연구 방향

#### **테스트 시간 적응(Test-Time Adaptation, TTA)**

최근 연구들은 소스 데이터 접근 없이 테스트 시간에만 모델을 적응시키는 TTA에 초점을 맞추고 있습니다. TTA는 도메인 이동이 지속적으로 변화하는 실제 환경에서 더 실용적입니다.[6][7][8]

**연구 방향**: 
- IAST의 인스턴스 적응형 접근을 TTA 설정에 적용하여 동적 도메인 이동에 대응[7]
- 배치 정규화(BN) 레이어만 조작하여 도메인 지식을 학습하는 방법 탐구[7]
- 연속적 도메인 이동에서 과거 지식 보존과 새로운 도메인 적응의 균형 유지[8]

#### **비전-언어 기반 모델(Vision-Language Foundation Models)**

CLIP과 같은 대규모 비전-언어 기반 모델은 도메인 적응 및 일반화에서 강력한 성능을 보이고 있습니다.[9][10][11]

**연구 방향**:
- CLIP의 도메인 불변 특성을 활용한 UDA 성능 향상[10][11]
- 프롬프트 학습을 통한 도메인 적응 및 일반화[11][9]
- 소스 데이터 없는(source-free) 설정에서 CLIP 활용[12]
- IAST의 가짜 레이블 생성 전략과 CLIP의 제로샷 예측 결합 가능성 탐구

#### **가짜 레이블 품질 향상**

가짜 레이블의 노이즈는 여전히 자기 학습의 주요 과제입니다.[13][4][3]

**연구 방향**:
- 전체 단계에서 가짜 레이블 품질을 향상시키는 메커니즘 개발[14][3]
- 불확실성 추정을 통한 가짜 레이블 선택 개선[13]
- 이웃 의미 일관성 및 공간 근접성을 활용한 가짜 레이블 정제[15]
- 메타 학습을 통한 가짜 인스턴스 중요도 추정[16]

#### **도메인 일반화(Domain Generalization)**

도메인 적응과 달리, 도메인 일반화는 타겟 도메인 데이터 없이도 보이지 않는 도메인에 일반화하는 것을 목표로 합니다.[17][18][9]

**연구 방향**:
- 다중 소스 도메인에서 도메인 불변 및 도메인 특정 특징 학습[18]
- CLIP 기반 도메인 일반화 및 적응 방법 개발[9]
- 진화하는 도메인에서 동적 잠재 표현 학습[19]
- IAST의 적응형 접근을 도메인 일반화 설정에 적용 가능성 탐구

#### **자기 학습의 한계 극복**

자기 학습은 의미 드리프트(semantic drift) 문제와 신경망의 과신 문제를 겪습니다.[13]

**연구 방향**:
- 가짜 레이블과 수동 레이블 간의 균형 유지[13]
- 불확실성을 고려한 하이브리드 메트릭 개발[13]
- 자기 학습 반복 중 성능 저하 방지 메커니즘[13]
- 부정 학습을 통한 혼란스러운 샘플의 영향 감소[20]

#### **소스 데이터 없는 도메인 적응(Source-Free Domain Adaptation)**

실제 환경에서 소스 데이터는 민감한 정보를 포함하거나 접근이 제한될 수 있습니다.[21][22][23]

**연구 방향**:
- 사전 학습된 소스 모델과 타겟 도메인 데이터만으로 적응[22][21]
- 파라미터 효율적 적응 방법(Low-Rank Adaptation) 개발[21]
- 프로토타입 기반 가짜 레이블 노이즈 제거[23]
- IAST의 인스턴스 적응형 접근을 소스 프리 설정에 확장

### 4.3 향후 연구 시 고려할 점

**계산 효율성**: IAST의 다단계 자기 학습은 계산 비용이 높을 수 있으므로, 효율적인 자기 학습 알고리즘 개발이 필요합니다.[24][25]

**다양한 도메인 조합**: 합성-실제 데이터 외에도 다양한 도메인 조합(예: 날씨 변화, 시간대 변화)에 대한 검증이 필요합니다.[26][27]

**멀티모달 확장**: 시각, 오디오, 물리적 도메인을 포함하는 진정한 멀티모달 자기 학습 시스템 개발이 유망합니다.[24]

**인간 피드백 결합**: 자율 학습과 인간 피드백을 결합한 하이브리드 시스템 탐구가 필요합니다.[24]

**개방 세트 도메인 적응(Open-Set Domain Adaptation)**: 타겟 도메인에 소스 도메인에 없는 미지의 클래스가 존재하는 경우를 다루는 연구가 활발히 진행되고 있습니다.[28][10]

**연속 도메인 적응(Continual Domain Adaptation)**: 도메인이 지속적으로 변화하는 환경에서 과거 지식을 유지하면서 새로운 도메인에 적응하는 방법 개발이 중요합니다.[29][8]

**평가 메트릭 및 벤치마크**: 도메인 적응 및 일반화를 위한 더 포괄적이고 표준화된 벤치마크 개발이 필요합니다.[30][27]

IAST는 자기 학습 기반 UDA의 중요한 이정표를 제시했으며, 가짜 레이블 품질 향상, 인스턴스 적응형 접근, 그리고 확장 가능한 프레임워크 설계를 통해 후속 연구에 지속적인 영향을 미치고 있습니다. 향후 연구는 TTA, 비전-언어 모델, 소스 프리 적응, 그리고 연속 학습과 같은 더 실용적이고 도전적인 설정으로 확장되어야 하며, 계산 효율성과 일반화 능력의 균형을 유지해야 합니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9e56355c-7ad3-44bf-8327-98442484d289/2008.12197v1.pdf)
[2](https://arxiv.org/abs/2008.12197)
[3](https://arxiv.org/abs/2407.08971)
[4](https://openaccess.thecvf.com/content/CVPR2023/papers/Cheng_BoxTeacher_Exploring_High-Quality_Pseudo_Labels_for_Weakly_Supervised_Instance_Segmentation_CVPR_2023_paper.pdf)
[5](https://www.sciencedirect.com/science/article/abs/pii/S0167865525003332)
[6](https://eccv.ecva.net/virtual/2024/poster/1938)
[7](https://arxiv.org/abs/2312.10165)
[8](https://openaccess.thecvf.com/content/CVPR2024/papers/Yang_A_Versatile_Framework_for_Continual_Test-Time_Domain_Adaptation_Balancing_Discriminability_CVPR_2024_paper.pdf)
[9](https://arxiv.org/pdf/2504.14280.pdf)
[10](https://arxiv.org/abs/2307.16204)
[11](https://openaccess.thecvf.com/content/ICCV2023W/OODCV/papers/Singha_AD-CLIP_Adapting_Domains_in_Prompt_Space_Using_CLIP_ICCVW_2023_paper.pdf)
[12](https://github.com/jindongli-Ai/Survey_on_CLIP-Powered_Domain_Generalization_and_Adaptation)
[13](https://arxiv.org/html/2401.00575v1)
[14](https://ieeexplore.ieee.org/document/11023636/)
[15](https://www.sciencedirect.com/science/article/abs/pii/S0925231224011962)
[16](https://aclanthology.org/2023.acl-long.92.pdf)
[17](http://arxiv.org/pdf/1710.03463.pdf)
[18](https://arxiv.org/pdf/2110.09410.pdf)
[19](https://arxiv.org/pdf/2401.08464.pdf)
[20](https://pure.kaist.ac.kr/en/publications/p-pseudolabel-enhanced-pseudo-labeling-framework-with-network-pru)
[21](https://arxiv.org/pdf/2502.21313.pdf)
[22](http://arxiv.org/pdf/2212.09563.pdf)
[23](https://arxiv.org/html/2509.16942v1)
[24](https://www.theaugmentededucator.com/p/when-ai-teaches-itself-the-breakthrough)
[25](https://www.sciencedirect.com/science/article/pii/S0925231224016758)
[26](https://openaccess.thecvf.com/content/WACV2024/papers/Zhao_Unsupervised_Domain_Adaptation_for_Semantic_Segmentation_With_Pseudo_Label_Self-Refinement_WACV_2024_paper.pdf)
[27](https://www.nature.com/articles/s41597-024-03951-4)
[28](https://www.sciencedirect.com/science/article/abs/pii/S1077314224003114)
[29](https://sukzoon1234.tistory.com/75)
[30](https://arxiv.org/pdf/2403.02714.pdf)
[31](https://arxiv.org/pdf/2104.12928.pdf)
[32](https://arxiv.org/abs/2405.16819)
[33](https://arxiv.org/html/2302.06992)
[34](http://arxiv.org/pdf/2106.09890.pdf)
[35](https://arxiv.org/pdf/1908.01342.pdf)
[36](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123710409.pdf)
[37](https://arxiv.org/abs/2005.10876)
[38](https://github.com/Raykoooo/IAST)
[39](https://arxiv.org/html/2505.24656v2)
[40](https://eccv.ecva.net/virtual/2024/poster/2261)
[41](https://proceedings.nips.cc/paper/2021/file/c1fea270c48e8079d8ddf7d06d26ab52-Paper.pdf)
[42](http://papers.neurips.cc/paper/8335-category-anchor-guided-unsupervised-domain-adaptation-for-semantic-segmentation.pdf)
[43](https://dl.acm.org/doi/10.1007/978-3-030-58574-7_25)
[44](https://openreview.net/forum?id=R6JvkqWijY)
[45](https://www.computer.org/csdl/journal/tp/2025/07/10930817/25bqhIyK3TO)
[46](https://pure.kaist.ac.kr/en/publications/semi-supervised-domain-adaptation-via-selective-pseudo-labeling-a/)
[47](https://papers.nips.cc/paper_files/paper/2022/hash/5c882988ce5fac487974ee4f415b96a9-Abstract-Conference.html)
[48](https://www.sciencedirect.com/science/article/pii/S0167865524002836)
[49](https://www.ijcai.org/proceedings/2024/0516.pdf)
[50](https://dl.acm.org/doi/10.1007/978-3-031-20497-5_13)
[51](https://arxiv.org/abs/2302.06992)
[52](https://arxiv.org/html/2502.06272v1)
[53](https://arxiv.org/abs/2301.10418)
[54](https://arxiv.org/pdf/2106.11344.pdf)
[55](https://arxiv.org/html/2403.07798v1)
[56](https://www.i-aida.org/course/domain-adaptation-generalization/)
[57](https://github.com/junha1125/Domain-Adaptation-Generalization-in-ECCV-2024)
[58](https://www.lidsen.com/journals/neurobiology/neurobiology-06-04-141)
[59](https://neurips.cc/virtual/2024/poster/93787)
[60](https://www.nature.com/articles/s41598-025-19121-4)
[61](https://papers.neurips.cc/paper_files/paper/2022/file/1e97fb8a7c9737e9e9f4e0389b25efe8-Paper-Conference.pdf)
[62](https://dl.acm.org/doi/10.1145/3674399.3674462)
[63](https://cvpr.thecvf.com/virtual/2025/workshop/32364)
[64](https://www.sciencedirect.com/science/article/abs/pii/S0888327024008227)
[65](https://ieeexplore.ieee.org/document/10484417/)
[66](https://pmc.ncbi.nlm.nih.gov/articles/PMC10614300/)
[67](https://pmc.ncbi.nlm.nih.gov/articles/PMC7964033/)
[68](https://pmc.ncbi.nlm.nih.gov/articles/PMC2322951/)
[69](https://journals.asm.org/doi/10.1128/aac.00777-24)
[70](https://pmc.ncbi.nlm.nih.gov/articles/PMC10096293/)
[71](https://pmc.ncbi.nlm.nih.gov/articles/PMC8210411/)
[72](https://arxiv.org/pdf/2312.17726.pdf)
[73](https://www.mdpi.com/1420-3049/28/7/3016/pdf?version=1680059217)
[74](https://pmc.ncbi.nlm.nih.gov/articles/PMC5337427/)
[75](https://arxiv.org/abs/2408.00727)
[76](https://www.nature.com/articles/s41467-023-44676-z)
[77](https://www.sciencedirect.com/science/article/pii/S0048733320302225)
[78](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0336387)
[79](https://dmqa.korea.ac.kr/uploads/seminar/%5B250207%5DDMQA_Openseminar_Test_Time_Adaptation.pdf)
[80](https://alinlab.kaist.ac.kr/resource/2025_SPRING_AI602/AI602_Lec4_Vision_Language_Foundation_Models.pdf)
[81](https://pubs.rsna.org/page/radiology/author-instructions)
[82](https://openreview.net/forum?id=x5LvBK43wg)
[83](https://openreview.net/forum?id=FRjflOWx2W&noteId=wiHrklMuy8)
[84](https://aacrjournals.org/clincancerres/article/31/22/4698/767043/Phase-I-Dose-Escalation-Trial-Combining-Olaparib)
[85](https://dl.acm.org/doi/10.1609/aaai.v38i14.29527)
