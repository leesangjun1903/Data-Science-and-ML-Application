# FixBi: Bridging Domain Spaces for Unsupervised Domain Adaptation 

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

FixBi는 **소스 도메인과 타겟 도메인 사이의 큰 분포 격차(large domain discrepancy)** 문제를 해결하기 위해, 두 도메인 사이에 **고정 비율 기반의 중간(intermediate) 도메인들을 생성**하고, 이를 통해 점진적으로 지식을 전달하는 UDA 방법론을 제안합니다.

기존 방법들은 소스→타겟 도메인으로의 **직접적(direct) 적응**에 의존했기 때문에, 두 도메인 간의 격차가 클 경우 성능 저하가 심각했습니다. FixBi는 이 문제를 **중간 도메인 브리징(bridging)**으로 해결합니다.

### 주요 기여 (3가지)

| 기여 | 내용 |
|------|------|
| **Fixed Ratio-based Mixup** | 고정 비율로 소스·타겟 도메인 사이의 중간 도메인 생성 |
| **Confidence-based Learning** | 양방향 매칭(긍정 pseudo-label) + 자기 패널티(부정 pseudo-label) |
| **Consistency Regularization** | 두 모델의 안정적 수렴 보장 |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

**UDA(Unsupervised Domain Adaptation)**에서:
- 소스 도메인: 레이블이 있는 데이터 $\mathcal{X}^s = \{(x_i^s, y_i^s)\}_{i=1}^{N_s}$
- 타겟 도메인: 레이블이 없는 데이터 $\mathcal{X}^t = \{(x_i^t)\}_{i=1}^{N_t}$

$P(\mathcal{X}^s)$와 $P(\mathcal{X}^t)$ 사이의 **큰 주변 분포(marginal distribution) 차이**가 핵심 장벽입니다. 기존 DANN, MMD 기반, GAN 기반 방법들은 이 큰 격차를 직접 극복하려 했으나, 두 도메인의 거리가 클수록 효과가 떨어집니다.

---

### 2.2 제안하는 방법 (수식 포함)

#### ① Fixed Ratio-based Mixup

소스 샘플 $(x_i^s, y_i^s)$와 타겟 샘플 $(x_i^t, \hat{y}_i^t)$를 혼합합니다:

$$\tilde{x}_i^{st} = \lambda x_i^s + (1 - \lambda) x_i^t$$

$$\tilde{y}_i^{st} = \lambda y_i^s + (1 - \lambda) \hat{y}_i^t$$

여기서 $\lambda \in \{\lambda_{sd}, \lambda_{td}\}$ s.t. $\lambda_{sd} + \lambda_{td} = 1$

- **SDM (Source-Dominant Model)**: $\lambda_{sd} = 0.7$ → 소스 도메인에 강한 지도
- **TDM (Target-Dominant Model)**: $\lambda_{td} = 0.3$ → 타겟 도메인과 높은 유사성

Fixed ratio-based mixup의 손실 함수:

$$\mathcal{L}_{fm} = \frac{1}{B} \sum_{i=1}^{B} \hat{y}_i^{st} \log(p(y|\tilde{x}_i^{st}))$$

여기서 $\hat{y}_i^{st} = \text{argmax}\, p(y|\tilde{x}_i^{st})$, $B$는 미니배치 크기

> **기존 랜덤 믹스업과의 차이**: 기존 방법 $\lambda \sim \text{Beta}(\alpha, \alpha)$은 도메인 무작위성이 크고 두 모델에 서로 다른 관점을 보장하지 못합니다. Fixed ratio는 두 모델이 명확히 구분되는 보완적(complementary) 특성을 가지도록 강제합니다.

---

#### ② Confidence-based Learning

**[양방향 매칭 - Bidirectional Matching with Positive Pseudo-labels]**

한 모델의 예측 신뢰도가 임계값 $\tau$를 초과할 때 긍정 pseudo-label로 상대 모델을 지도합니다:

$$\mathcal{L}_{bim} = \frac{1}{B} \sum_{i=1}^{B} \mathbb{1}(\max(p(y|x_i^t) > \tau))\, \hat{y}_i^t \log(q(y|x_i^t))$$

여기서 $\hat{y}_i^t = \text{argmax}\, p(y|x_i^t)$, $p$와 $q$는 두 모델의 확률 분포

- FixMatch는 단방향(one-way)이지만, FixBi는 **양방향(bidirectional)** 매칭 가능

---

**[자기 패널티 - Self-penalization with Negative Pseudo-labels]**

신뢰도가 $\tau$ 미만인 예측(부정 pseudo-label)에 대해 해당 클래스 확률을 0에 가깝게 만들도록 자기 자신을 패널티:

$$\mathcal{L}_{sp} = \frac{1}{B} \sum_{i=1}^{B} \mathbb{1}(\max(p(y|x_i^t) < \tau))\, \hat{y}_i^t \log(1 - p(y|x_i^t))$$

- 기존 연구[12, 34, 44]들이 **낮은 신뢰도 예측을 무시**한 것과 달리, FixBi는 이를 **의미 있는 학습 신호**로 활용

---

**[적응형 임계값 (Adaptive Threshold)]**

고정 임계값 대신 미니배치의 평균과 표준편차로 동적으로 조정:

$$\tau = \text{mean} - 2 \times \text{std}$$

학습 초기에는 낮고, 학습이 진행될수록 높아지는 특성을 반영합니다.

---

#### ③ Consistency Regularization

$\lambda_{cr} = 0.5$로 생성된 중간 도메인에서 두 모델의 출력이 일관되도록 L2 손실:

$$\mathcal{L}_{cr} = \frac{1}{B} \sum_{i=1}^{B} \| p(y|\tilde{x}_i^{st}) - q(y|\tilde{x}_i^{st}) \|_2^2$$

두 모델이 서로 다른 도메인 공간에서 학습되더라도 같은 중간 영역에서 일관된 예측을 유지하도록 합니다.

---

### 2.3 모델 구조

```
[학습 흐름]

소스 도메인 (labeled)  ──┐
                          ├─── Fixed Ratio Mixup (λ_sd=0.7) ──→ SDM (CNN + Classifier)
타겟 도메인 (unlabeled) ──┤                                            ↕ Bidirectional Matching
                          └─── Fixed Ratio Mixup (λ_td=0.3) ──→ TDM (CNN + Classifier)
                          
                          ─── Mixup (λ_cr=0.5) ──→ Consistency Regularization (SDM ↔ TDM)
```

- **백본**: Office-31, Office-Home → ResNet-50 / VisDA-2017 → ResNet-101
- **베이스라인**: DANN (분석용), MSTN (성능 비교용)
- **두 모델 SDM, TDM은 동일한 구조**이지만 서로 다른 mixup 비율로 학습
- **최종 예측**: SDM + TDM의 소프트맥스 출력의 앙상블(합산)

#### 학습 절차 (Algorithm 1 요약)

| 단계 | 내용 |
|------|------|
| Warm-up ( $e \leq k$ ) | SDM·TDM 각각 $\mathcal{L}\_{fm}$ + $\mathcal{L}_{sp}$로 독립 학습 |
| Main ( $e > k$ ) | $\mathcal{L}\_{bim}$ (양방향 매칭) + $\mathcal{L}_{cr}$ (일관성 정규화) 추가 |

---

### 2.4 성능 향상

#### Office-31 (ResNet-50)

| 방법 | A→W | D→A | W→A | **Avg** |
|------|-----|-----|-----|---------|
| DANN | 82.0 | 68.2 | 67.4 | 82.2 |
| MSTN | 91.3 | 72.7 | 65.6 | 86.5 |
| SRDC | 95.7 | 76.7 | 77.1 | 90.8 |
| RSDA-MSTN | 96.1 | 77.4 | 78.9 | 91.1 |
| **FixBi** | **96.1** | **78.7** | **79.4** | **91.4** |

- DANN 대비 평균 **+7.1%** 향상
- W→A 태스크에서 베이스라인 대비 **+13.8%**

#### Office-Home (ResNet-50)

| 방법 | Avg |
|------|-----|
| SRDC | 71.3 |
| RSDA-MSTN | 70.9 |
| **FixBi** | **72.7** |

#### VisDA-2017 (ResNet-101)

| 방법 | Avg |
|------|-----|
| DMRL | 75.5 |
| CAN | 87.2 |
| **FixBi** | **87.2** |

- 베이스라인(MSTN) 대비 **+22.2%**
- 다른 mixup 기반 방법들보다 **약 12% 향상**

---

### 2.5 한계점

논문에서 명시적으로 기술된 한계와 분석을 통해 도출되는 한계는 다음과 같습니다:

| 한계 | 설명 |
|------|------|
| **하이퍼파라미터 민감도** | $\lambda_{sd}=0.7$, $\lambda_{td}=0.3$ 설정이 최적임을 실험으로 확인했으나, 데이터셋마다 최적값이 다를 수 있음 |
| **Pseudo-label 노이즈** | 베이스라인 모델(DANN/MSTN)의 초기 pseudo-label 품질에 성능이 의존 |
| **두 모델의 계산 비용** | SDM과 TDM 두 네트워크를 동시에 학습하므로 단일 모델 대비 메모리·연산량 증가 |
| **VisDA 'truck' 클래스** | VisDA-2017에서 truck 클래스 정확도가 25.7%로 현저히 낮음 (Table 6) |
| **도메인 수 고정** | 중간 도메인이 두 개의 고정 비율로만 생성되어, 더 세밀한 중간 단계 활용 미흡 |

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 중간 도메인을 통한 일반화

FixBi의 핵심 아이디어는 **도메인 사이의 공간을 단계적으로 탐색**하는 것입니다. 이는 모델이 단순히 소스→타겟 매핑을 학습하는 것이 아니라, **연속적인 도메인 스펙트럼 위에서 표현(representation)을 학습**하도록 유도합니다.

$$\text{Source} \xrightarrow{\lambda_{sd}=0.7} \tilde{\mathcal{X}}^{sd} \xrightarrow{\lambda_{cr}=0.5} \tilde{\mathcal{X}}^{cr} \xrightarrow{\lambda_{td}=0.3} \tilde{\mathcal{X}}^{td} \rightarrow \text{Target}$$

이 구조는 모델이 **특정 도메인에 과적합(overfitting)** 되지 않도록 하여 일반화를 돕습니다.

### 3.2 보완적 모델 앙상블에 의한 일반화

SDM과 TDM은 **클래스별 정확도에서 서로 다른 강점과 약점**을 가집니다(Figure 3). 이들의 앙상블은 단일 모델보다 더 강건한 예측을 생성합니다:

- 단일 관점 앙상블 (0.3,0.3): Avg **84.2%**
- 단일 관점 앙상블 (0.7,0.7): Avg **84.7%**
- **FixBi 양방향 관점 (0.7,0.3): Avg 87.0%**

### 3.3 Self-penalization의 일반화 효과

낮은 신뢰도 예측에 대한 패널티는 모델이 **불확실한 샘플에 대해 과신(overconfidence)하지 않도록** 합니다. 이는 calibration 관점에서 일반화 성능을 높이는 효과가 있습니다.

### 3.4 Consistency Regularization의 역할

$$\mathcal{L}_{cr} = \frac{1}{B} \sum_{i=1}^{B} \| p(y|\tilde{x}_i^{st}) - q(y|\tilde{x}_i^{st}) \|_2^2$$

이 손실은 두 모델이 **같은 입력에 대해 일관된 예측**을 내놓도록 강제하며, 이는 representation의 안정성과 일반화에 기여합니다.

### 3.5 t-SNE 시각화에서의 확인

Figure 5에서 DANN(베이스라인)은 타겟 도메인 특징이 소스 클러스터 주변에 분산되어 있으나, **FixBi는 타겟 도메인 특징이 소스 도메인 클러스터와 가까운 위치에 조밀하게 형성**됩니다. 이는 더 높은 도메인 불변(domain-invariant) 표현 학습을 의미합니다.

---

## 4. 앞으로의 연구에 미치는 영향과 고려할 점

### 4.1 연구에 미치는 영향

#### ① 중간 도메인 브리징 패러다임의 확산
FixBi는 **점진적 도메인 전이(progressive domain transfer)**라는 새로운 관점을 제시했습니다. 이 아이디어는 이후 연구에서:
- 더 많은 수의 중간 도메인 생성
- 적응형 비율(adaptive ratio) 학습
- 다중 소스 도메인으로 확장

의 방향으로 발전될 수 있습니다.

#### ② 신뢰도 기반 학습의 정교화
Positive/Negative pseudo-label을 모두 활용하는 아이디어는 **준지도학습과 UDA의 경계를 좁히는** 중요한 기여입니다. 이는 이후 노이즈 레이블 학습, 오픈셋 UDA 등에 응용될 수 있습니다.

#### ③ 플러그인(Plugin) 가능한 구조
FixBi는 DANN, MSTN 등 **기존 UDA 방법에 플러그인으로 적용 가능**한 구조를 가집니다. 이는 향후 연구에서 더 강력한 베이스라인과 결합하는 방향으로 이어질 수 있습니다.

---

### 4.2 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의**: 아래 비교는 제공된 논문(FixBi, CVPR 2021)의 내용과 일반적으로 알려진 연구 동향을 기반으로 합니다. 2021년 이후 논문들의 구체적 수치는 해당 논문을 직접 확인하시기 바랍니다.

| 연구 | 발표 | 핵심 아이디어 | FixBi와의 관계 |
|------|------|--------------|---------------|
| **SRDC** (Tang et al.) | CVPR 2020 | 클러스터링 + 구조적 정규화 | FixBi가 Office-Home에서 능가 |
| **RSDA-MSTN** (Gu et al.) | CVPR 2020 | 구면 공간에서의 pseudo-label | FixBi와 유사한 성능 (Office-31) |
| **DMRL** (Wu et al.) | ECCV 2020 | 이중 랜덤 믹스업 정규화 | FixBi의 고정 비율이 더 효과적임을 입증 |
| **DM-ADA** (Xu et al.) | AAAI 2020 | 적대적 도메인 믹스업 | FixBi가 VisDA에서 +11.6% 능가 |
| **CDTrans** (Xu et al.) | ICCV 2021 | Transformer 기반 cross-domain attention | Transformer 시대의 UDA로 확장 |
| **SPA** | 2021 이후 | Semantic Prototype Alignment | 클래스 수준 정렬 강화 |

**FixBi와 후속 연구의 차별점:**

```
FixBi (2021): Mixup 기반 중간 도메인 + 이중 모델 + 신뢰도 학습
     ↓
CDTrans 등: Self-attention으로 도메인 불변 특징 직접 학습
     ↓
최신 연구: Foundation Model (CLIP 등) 활용 UDA로 진화
```

---

### 4.3 향후 연구 시 고려할 점

#### ① 더 세밀한 중간 도메인 생성
현재 FixBi는 2개의 고정 비율($\lambda_{sd}, \lambda_{td}$)만 사용합니다. 향후에는:

$$\lambda_k = \frac{k}{K}, \quad k = 0, 1, \ldots, K$$

와 같이 **K개의 균등한 중간 도메인**을 생성하거나, 데이터의 분포 거리(Wasserstein distance 등)에 따라 **적응적으로 비율을 결정**하는 방향을 고려할 수 있습니다.

#### ② Transformer/Foundation Model과의 결합
ViT(Vision Transformer), CLIP 등의 강력한 사전학습 모델과 FixBi의 브리징 아이디어를 결합하면, 더 일반화된 도메인 불변 표현을 학습할 수 있습니다.

#### ③ 다중 소스/타겟 도메인으로 확장
현재는 단일 소스→단일 타겟 구조이지만, 실제 환경에서는 여러 소스/타겟 도메인이 존재합니다. **Multi-source FixBi** 확장이 의미있는 연구 방향입니다.

#### ④ Pseudo-label 품질 개선
초기 pseudo-label이 베이스라인 모델(DANN, MSTN)에 의존하므로, **자기지도학습(self-supervised learning)**으로 더 좋은 초기 표현을 확보한 후 FixBi를 적용하는 방향도 고려할 수 있습니다.

#### ⑤ 계산 효율성 개선
두 개의 모델(SDM, TDM)을 동시에 학습하는 구조는 메모리와 연산량이 2배 소모됩니다. **파라미터 공유(parameter sharing)** 또는 **지식 증류(knowledge distillation)**를 통해 단일 모델로 압축하는 연구가 필요합니다.

#### ⑥ 오픈셋(Open-set) UDA로 확장
현재는 소스와 타겟의 클래스가 동일하다고 가정하지만, 실제에서는 타겟에만 존재하는 클래스가 있을 수 있습니다. Self-penalization을 미지 클래스 탐지에 활용하는 방향으로 확장 가능합니다.

---

## 참고 자료

**주요 참고 문헌 (논문 내 인용 포함):**

1. **Na, J., Jung, H., Chang, H. J., & Hwang, W. (2021).** *FixBi: Bridging Domain Spaces for Unsupervised Domain Adaptation.* CVPR 2021. *(제공된 PDF)*

2. **Ganin, Y., & Lempitsky, V. (2015).** *Unsupervised domain adaptation by back propagation.* ICML 2015. [DANN]

3. **Sohn, K., et al. (2020).** *FixMatch: Simplifying semi-supervised learning with consistency and confidence.* arXiv:2001.07685.

4. **Wu, Y., Inkpen, D., & El-Roby, A. (2020).** *Dual mixup regularized learning for adversarial domain adaptation.* ECCV 2020. [DMRL]

5. **Tang, H., Chen, K., & Jia, K. (2020).** *Unsupervised domain adaptation via structurally regularized deep clustering.* CVPR 2020. [SRDC]

6. **Gu, X., Sun, J., & Xu, Z. (2020).** *Spherical space domain adaptation with robust pseudo-label loss.* CVPR 2020. [RSDA]

7. **Zhang, H., Cisse, M., Dauphin, Y. N., & Lopez-Paz, D. (2018).** *Mixup: Beyond empirical risk minimization.* ICLR 2018.

8. **Han, B., et al. (2020).** *Robust training of deep neural networks with extremely noisy labels.* NeurIPS 2020. [Co-teaching]

9. **GitHub Repository:** https://github.com/NaJaeMin92/FixBi
