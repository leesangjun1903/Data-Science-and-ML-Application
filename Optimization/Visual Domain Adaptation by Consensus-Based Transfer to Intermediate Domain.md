# Visual Domain Adaptation by Consensus-Based Transfer to Intermediate Domain

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장
본 논문(AAAI-20, Samsung SDS)은 소스 도메인과 타깃 도메인이 **현저히 상이할 때** 기존 UDA(Unsupervised Domain Adaptation) 방법들이 한계를 보인다는 문제를 지적하며, 두 도메인을 **추상적 중간 도메인(Intermediate Domain)** 으로 동시에 정렬(align)하는 프레임워크를 제안합니다.

### 주요 기여
| 기여 항목 | 설명 |
|---|---|
| 중간 도메인 정렬 | 소스·타깃 모두를 새로운 잠재 도메인으로 쌍방향 변환 |
| 합의 기반 앙상블 분류기 | 다수 분류기의 예측 합의로 모호한 샘플 처리 |
| 이중 구조 아키텍처 | Forward/Inverse Network로 균형 잡힌 도메인 변환 학습 |
| 효율성 | SBADA 대비 약 68배 빠른 학습 속도(1.12s vs 76.73s/epoch) |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

기존 UDA의 두 가지 주류 접근법은 다음과 같은 한계를 지닙니다.

- **도메인 분류기 기반(Domain Classifier-based)**: DANN 등은 소스·타깃의 특징 분포가 **겹친다는 가정** 하에 작동하므로, 두 도메인이 너무 다르면 성능 저하
- **생성 모델 기반(Generative Model-based)**: CycleGAN, SBADA 등은 한 도메인을 다른 도메인으로 변환 시 **분류에 중요한 속성(예: 숫자 형태)이 왜곡**될 수 있음

**핵심 문제**: 소스·타깃 도메인 간 격차가 클 때(예: MNIST→SVHN), 단방향 스타일 변환은 정보 손실을 유발

---

### 2.2 제안 방법 (수식 포함)

#### (1) 크로스-도메인 이미지 어댑터 (Cross-domain Image Adaptors)

두 어댑터 $A_s$(소스용), $A_t$(타깃용)는 각각 **컬러 변환기(Color Transformer)** 와 **공간 변환기(Spatial Transformer)** 로 구성됩니다.

**컬러 변환** ($\mathbf{T}_c \in \mathbb{R}^{3\times9}$):

$$[r'_i, g'_i, b'_i]^T = \mathbf{T}_c [r_i, g_i, b_i, r_i^2, g_i^2, b_i^2, r_ig_i, r_ib_i, g_ib_i]^T \tag{1}$$

**공간 변환** ($\mathbf{T}_s \in \mathbb{R}^{2\times3}$):

$$[x', y']^T = \mathbf{T}_s [x, y, 1]^T \tag{2}$$

픽셀 강도는 서브픽셀 위치 $\mathbf{T}_s^{-1}[x', y']$에서 양선형 보간(bilinear interpolation)으로 추정하며, 모두 end-to-end 학습 가능합니다.

#### (2) 3가지 손실 함수

**① 지도 손실 (Supervised Loss)**:

$$L_s(\mathbf{X}, \mathbf{Y}, \theta) = -\frac{1}{N}\sum_{i=1}^{N} \mathbf{y}_o^{(i)T} \log p(\mathbf{y}|\mathbf{x}^{(i)}, \theta) \tag{3}$$

**② 합의 손실 (Consensus Loss)**:

앙상블 예측:
$$\hat{p}(\mathbf{y}|\mathbf{x}^{(i)}) = \frac{1}{N_c}\sum_{k=1}^{N_c} p(\mathbf{y}|\mathbf{x}^{(i)}, \theta_k) \tag{4}$$

$$\hat{p}^-(\mathbf{y}|\mathbf{x}^{(i)}) = \frac{1}{N_c}\sum_{k=1}^{N_c} p(\mathbf{y}|\mathbf{x}^{(i)}, \theta_k^-)$$

합의 손실:
$$L_c(\mathbf{X}) = -\frac{1}{N}\sum_{i=1}^{N}\sum_{k=1}^{N_c} \hat{p}(\mathbf{y}|\mathbf{x}^{(i)}) \log p(\mathbf{y}|\mathbf{x}^{(i)}, \theta_k) \tag{5}$$

$$L_c^-(\mathbf{X}) = -\frac{1}{N}\sum_{i=1}^{N}\sum_{k=1}^{N_c} \hat{p}^-(\mathbf{y}|\mathbf{x}^{(i)}) \log p(\mathbf{y}|\mathbf{x}^{(i)}, \theta_k^-)$$

**③ 혼동 손실 (Fooling Loss)**:

분류기가 너무 빨리 수렴하는 것을 방지하는 음의 주변 엔트로피:

$$L_f(\mathbf{X}, \theta) = \frac{1}{N}\sum_{i=1}^{N} p(\mathbf{y}|\mathbf{x}^{(i)}, \theta) \log p(\mathbf{y}|\mathbf{x}^{(i)}, \theta) \tag{6}$$

#### (3) 순차적 학습 알고리즘 (7단계)

| 단계 | 업데이트 대상 | 손실 함수 | 목적 |
|---|---|---|---|
| Step 1 | $G, C_1,\ldots,C_{N_c}$ | $L_s$ | 소스 레이블로 순방향 네트워크 학습 |
| Step 2 | $C_1,\ldots,C_{N_c}$ | $L_f$ | 타깃에 대한 분류기 혼동 유도 |
| Step 3 | $A_t, G$ | $L_c$ | 타깃의 특징을 중간 도메인으로 정렬 |
| Step 4 | $G^-, C_1^-,\ldots,C_{N_c}^-$ | $L_s$ | 역방향 네트워크 지도 학습 |
| Step 5 | $C_1^-,\ldots,C_{N_c}^-$ | $L_f$ | 소스에 대한 역분류기 혼동 유도 |
| Step 6 | $G^-$ | $L_c^-$ | 역방향 특징 생성기 업데이트 |
| Step 7 | $A_s$ | $L_s$ | 소스 어댑터 업데이트 |

수식으로 정리하면:

$$\arg\min_{G, C_1,\ldots,C_{N_c}} \sum_{k=1}^{N_c} L_s(\mathbf{X}_s, \mathbf{Y}_s, \theta_k) \tag{7}$$

$$\arg\min_{C_1,\ldots,C_{N_c}} \sum_{k=1}^{N_c} L_f(\mathbf{X}_t, \theta_k) \tag{8}$$

$$\arg\min_{A_t, G} L_c(\mathbf{X}_t) \tag{9}$$

타깃 의사 레이블 추정:
$$\hat{\mathbf{y}}_t^{(i)} = \hat{p}(\mathbf{y}|\mathbf{x}_t^{(i)}) \tag{10}$$

$$\arg\min_{G^-, C_1^-,\ldots,C_{N_c}^-} \sum_{k=1}^{N_c} L_s(\mathbf{X}_t, \mathbf{Y}_t^*, \theta_k^-) \tag{11}$$

$$\arg\min_{C_1^-,\ldots,C_{N_c}^-} \sum_{k=1}^{N_c} L_f(\mathbf{X}_s, \theta_k^-) \tag{12}$$

$$\arg\min_{G^-} L_c^-(\mathbf{X}_s) \tag{13}$$

$$\arg\min_{A_s} \sum_{k=1}^{N_c} L_s(\mathbf{X}_s, \mathbf{Y}_s, \theta_k^-) \tag{14}$$

---

### 2.3 모델 구조

```
소스 도메인 Xs ──→ [As: 컬러변환+공간변환] ──┐
                                         ├──→ [G: 특징생성기] ──→ [C1,...,CNc] ──→ 앙상블 결과
타깃 도메인 Xt ──→ [At: 컬러변환+공간변환] ──┘   (순방향 네트워크)
                                         │
                                         └──→ [G⁻: 역특징생성기] ──→ [C1⁻,...,CNc⁻]
                                                (역방향 네트워크, 학습 후 제거)
```

- $\theta_k = \{A_s, A_t, G, C_k\}$ (순방향)
- $\theta_k^- = \{A_s, A_t, G^-, C_k^-\}$ (역방향)
- 분류기 수: $N_c = 5$ (실험적으로 포화점)

---

### 2.4 성능 향상

**Table 3 주요 결과**:

| 적응 방향 | DANN | SBADA | **IEDA(제안)** |
|---|---|---|---|
| MNIST→SVHN | 35.7% | 61.1% | **78.5%** |
| SVHN→MNIST | 71.1% | 76.1% | **98.9%** |
| USPS→MNIST | 73.0% | 95.0% | **97.5%** |
| STL→CIFAR | - | - | **62.3%** |
| X-ray (폐렴) | - | 67.21% | **72.37%** |

**학습 효율 (MNIST→SVHN)**:
- SBADA: 76.73s/epoch × 500 epochs, 정확도 61.1%
- **IEDA**: 1.12s/epoch × 100 epochs, 정확도 78.5%

---

### 2.5 한계

1. **제한적인 변환기 종류**: 컬러/공간 변환기만 사용하여 복잡한 도메인 차이(예: 텍스처, 조명 등)에 대한 일반화가 제한적
2. **의사 레이블 노이즈**: 타깃 도메인 레이블이 없어 예측 레이블을 사용하므로 초기 학습 단계에서 노이즈 존재 (역방향 네트워크 학습에만 사용하여 영향 최소화)
3. **분류기 수 민감성**: 유사 도메인에서는 $N_c$에 민감하며 도메인마다 최적 분류기 수가 다를 수 있음
4. **픽셀 수준 중간 도메인 해석 어려움**: 중간 도메인 자체가 추상적이어서 시각적으로 해석하기 어려운 경우 존재
5. **하이퍼파라미터 공유**: 데이터셋 간 파라미터를 공유한다고 하나, 학습률 등 세부 설정은 여전히 조정 필요

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 중간 도메인 정렬의 일반화 기여

기존 방법은 소스→타깃 또는 타깃→소스의 **단방향 이동**을 시도합니다. 반면 IEDA는 두 도메인을 **제3의 중간 도메인으로 동시에 이동**시킵니다.

$$\text{기존}: \mathbf{X}_s \xrightarrow{\text{style transfer}} \tilde{\mathbf{X}}_s \approx \mathbf{X}_t$$

$$\text{IEDA}: \mathbf{X}_s \xrightarrow{A_s} \mathbf{X}_{\text{mid}} \xleftarrow{A_t} \mathbf{X}_t$$

이 구조는 다음 측면에서 일반화를 향상합니다:

- **정보 보존**: 어느 한 도메인의 스타일로 완전히 변환하지 않으므로 분류 핵심 특징 유지
- **대칭적 최적화**: $A_s$와 $A_t$가 균형 있게 학습되어 도메인 편향 감소
- **도메인 불변 표현 학습**: 중간 도메인에서의 특징이 소스/타깃 모두에 대해 동등하게 유효

### 3.2 앙상블 합의 기반 일반화

단일 분류기 대비 다수 분류기 합의는 **예측 불확실성을 명시적으로 모델링**합니다.

$$\hat{p}(\mathbf{y}|\mathbf{x}) = \frac{1}{N_c}\sum_{k=1}^{N_c} p(\mathbf{y}|\mathbf{x}, \theta_k)$$

- 모호한 샘플에 대한 **편향 감소 효과** (bias reduction via ensemble)
- 각 분류기가 서로 다른 랜덤 초기화로 시작하여 **다양한 결정 경계** 학습
- 합의 손실은 분류기 간 **일관성 강제**, 즉 과적합 방지

실험적으로: 합의 손실을 단순 교차 엔트로피(의사 레이블 기반)로 대체 시 78.5%→32.7%로 급격한 성능 하락.

### 3.3 혼동 손실(Fooling Loss)의 정규화 효과

$$L_f(\mathbf{X}, \theta) = \frac{1}{N}\sum_{i=1}^{N} p(\mathbf{y}|\mathbf{x}^{(i)}, \theta) \log p(\mathbf{y}|\mathbf{x}^{(i)}, \theta)$$

이는 **음의 엔트로피**로, 분류기가 타깃 샘플에 과도하게 확신하는 것을 방지합니다.

- 특징 생성기와 어댑터가 충분히 수렴하기 전에 분류기가 먼저 수렴하는 **조기 수렴 문제 방지**
- 정규화 손실(Zou et al., 2019) 대비 실험에서 78.5% vs 58.62%로 우수

### 3.4 의료 영상 등 다양한 도메인으로의 확장성

X-ray 폐렴 인식 실험(Chest X-ray→RSNA Dataset)에서 72.37% 달성(MCDDA 67.21% 대비 향상)은 이 프레임워크가 **특정 도메인에 국한되지 않음**을 보여줍니다.

### 3.5 파라미터 공유를 통한 일반화

논문에서 "모든 데이터셋에 대해 동일한 학습 파라미터를 공유"한다고 명시하며, 이는 데이터셋별 하이퍼파라미터 튜닝 없이도 일반화 가능한 구조임을 시사합니다.

---

## 4. 앞으로의 연구에 미치는 영향과 고려할 점

### 4.1 앞으로의 연구에 미치는 영향

#### (1) 중간 도메인 개념의 확장

IEDA의 "제3 도메인으로의 쌍방향 정렬" 개념은 **다중 소스 도메인 적응(Multi-source DA)** 또는 **도메인 일반화(Domain Generalization)** 로 확장할 수 있습니다. 여러 소스 도메인을 공통 잠재 도메인으로 정렬하는 연구 방향이 촉진됩니다.

#### (2) 합의 기반 준지도 학습

앙상블 분류기의 합의를 이용한 의사 레이블링 전략은 **준지도 학습(Semi-supervised Learning)** 및 **지속 학습(Continual Learning)** 에도 적용 가능합니다.

#### (3) 변환기 설계의 모듈화

컬러/공간 변환기의 분리 설계는 **플러그인 방식의 도메인 어댑터** 연구를 자극하며, Vision Transformer 시대의 Adapter-based fine-tuning 연구와 연결됩니다.

#### (4) 의료, 자율주행 등 응용 분야

소스·타깃 간 격차가 큰 실제 응용(의료 이미징, 위성 영상, 자율주행 도메인 변환)에서의 UDA 연구에 직접적인 기여를 합니다.

---

### 4.2 앞으로 연구 시 고려할 점

#### (1) 중간 도메인의 이론적 보장

현재 중간 도메인은 암묵적으로 학습되며, 이것이 실제로 최적 중간점인지에 대한 **이론적 보장이 부족**합니다. Ben-David et al.의 도메인 적응 이론을 중간 도메인 설정으로 확장하는 연구가 필요합니다.

#### (2) 고해상도/복잡한 변환 처리

현재의 컬러·공간 변환기는 단순한 선형/아핀 변환에 국한됩니다. **StyleGAN, Diffusion Model 기반의 더 표현력 있는 어댑터** 도입을 고려해야 합니다.

#### (3) 의사 레이블의 노이즈 강건성

Step 4-6에서 사용하는 의사 레이블($\mathbf{Y}_t^*$)의 노이즈가 역방향 네트워크 학습에 미치는 영향에 대한 심층 분석이 필요합니다. 신뢰도 기반 필터링(confidence-based filtering)과 결합하면 성능이 향상될 수 있습니다.

#### (4) 분류기 수 $N_c$ 자동 결정

현재 $N_c=5$는 경험적으로 설정되었습니다. **베이지안 최적화** 또는 **동적 앙상블 크기 조정** 방법과 결합하여 자동화할 필요가 있습니다.

#### (5) 도메인 거리 측도와의 연계

두 도메인 간 거리(예: Wasserstein distance, $\mathcal{H}$-divergence)를 명시적으로 측정하여 중간 도메인 학습의 **수렴성과 안정성**을 보장하는 연구가 요구됩니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의**: 아래 비교는 제공된 논문 원문과 제 학습 데이터에 기반하며, 2020년 이후 논문의 세부 수치에 대해서는 원문 확인을 권장합니다.

### 5.1 주요 후속/관련 연구

| 연구 | 발표 | 핵심 방법 | IEDA 대비 차이 |
|---|---|---|---|
| **CDTrans** (Xu et al., 2021) | ICCV 2021 | 크로스 어텐션 트랜스포머 기반 DA | 픽셀 변환 없이 어텐션으로 도메인 정렬 |
| **SSRT** (Sun et al., 2022) | CVPR 2022 | Self-supervised ViT 기반 DA | Vision Transformer의 자기지도 표현 활용 |
| **SPA** (Wang et al., 2022) | - | 스타일-패치 어텐션 | 패치 수준 스타일 전이 |
| **PMTrans** (Zhu et al., 2022) | ECCV 2022 | 패치 혼합 트랜스포머 | 패치 수준 중간 도메인 구성 |
| **DomainBed** (Gulrajani & Lopez-Paz, 2021) | ICLR 2021 | UDA 방법론 벤치마크 | 다양한 방법 통합 비교 |

### 5.2 핵심 발전 방향과 IEDA의 위치

```
IEDA (2020) ──→ ViT 기반 DA ──→ Foundation Model 기반 DA
[CNN+픽셀변환]   [어텐션+패치정렬]   [CLIP, DINO 활용]
```

**① Vision Transformer 기반 UDA**:

2021년 이후 ViT(Vision Transformer)를 UDA에 도입하는 연구가 급증했습니다. CDTrans(2021)는 소스-타깃 간 크로스 어텐션으로 도메인 정렬을 수행합니다. IEDA의 픽셀 수준 변환에 비해 **더 유연한 비국소적(non-local) 특징 정렬**이 가능합니다.

$$\text{CrossAttention}(Q_s, K_t, V_t) = \text{softmax}\left(\frac{Q_s K_t^T}{\sqrt{d}}\right)V_t$$

IEDA의 affine 변환보다 훨씬 풍부한 표현이 가능하지만, 계산 비용이 큽니다.

**② 프리트레인 모델(Foundation Model) 활용**:

CLIP(Radford et al., 2021), DINOv2(Oquab et al., 2023) 등 대규모 사전학습 모델의 등장으로 적은 타깃 데이터로도 강력한 도메인 적응이 가능해졌습니다. IEDA가 처음부터 특징을 학습하는 것과 달리, **이미 학습된 도메인 불변 표현을 활용**하는 방향으로 발전했습니다.

**③ Source-free DA로의 확장**:

2021년 이후 소스 데이터 없이 타깃만으로 적응하는 **Source-free DA** 연구(SHOT, Liang et al., 2020)가 주목받고 있습니다. IEDA는 학습 시 소스 데이터가 필요하므로 이 방향으로의 확장 연구가 필요합니다.

**④ 대조 학습(Contrastive Learning) 기반 DA**:

SimCLR, MoCo 등의 대조 학습 원리를 DA에 결합한 연구가 등장했습니다. IEDA의 합의 손실이 **클래스 내 응집성**을 어느 정도 달성하지만, 명시적인 대조 손실은 더 강력한 경계를 형성할 수 있습니다.

### 5.3 종합 비교

| 항목 | IEDA (2020) | 최신 ViT 기반 (2021+) |
|---|---|---|
| 백본 | CNN | ViT/Transformer |
| 도메인 정렬 수준 | 픽셀 + 특징 | 패치/토큰 수준 |
| 중간 도메인 | 명시적 잠재 도메인 | 암묵적 표현 공간 |
| 소스 데이터 필요 | 필요 | Source-free 가능 |
| 계산 효율 | 매우 높음 | 상대적으로 낮음 |
| 의사 레이블 전략 | 앙상블 합의 | 신뢰도 기반 필터링 |

**결론적으로**, IEDA는 2020년 당시 CNN 기반 UDA의 한계를 효과적으로 돌파하는 방법론을 제시했으며, 특히 **도메인 간 격차가 큰 어려운 시나리오**에서의 우수성이 입증되었습니다. 이후 ViT와 대형 언어모델의 등장으로 패러다임이 변화했지만, **중간 도메인 정렬의 개념**, **합의 기반 앙상블**, **이중 구조 아키텍처**의 아이디어는 현재에도 유효하며 계속 발전·응용될 수 있습니다.

---

## 참고 자료

**기본 논문**:
- Choi, J., Choi, Y., Kim, J., Chang, J., Kwon, I., Gwon, Y., & Min, S. (2020). *Visual Domain Adaptation by Consensus-Based Transfer to Intermediate Domain*. AAAI-2020, pp. 10655–10662.

**논문 내 인용 참고문헌**:
- Ganin, Y., & Lempitsky, V. (2015). *Unsupervised domain adaptation by backpropagation*. ICML.
- Russo, P., et al. (2018). *From source to target and back: symmetric bidirectional adaptive GAN*. CVPR. (SBADA)
- Saito, K., et al. (2018). *Maximum classifier discrepancy for unsupervised domain adaptation*. CVPR.
- Jaderberg, M., et al. (2015). *Spatial transformer networks*. NIPS.
- Zou, Y., et al. (2019). *Confidence regularized self-training*. ICCV.
- Gong, R., et al. (2019). *DLOW: Domain flow for adaptation and generalization*. CVPR.

**2020년 이후 관련 연구 (학습 데이터 기반, 원문 확인 권장)**:
- Liang, J., et al. (2020). *Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation*. ICML. (SHOT)
- Xu, T., et al. (2021). *CDTrans: Cross-domain Transformer for Unsupervised Domain Adaptation*. ICCV.
- Gulrajani, I., & Lopez-Paz, D. (2021). *In search of lost domain generalization*. ICLR. (DomainBed)
- Radford, A., et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision*. ICML. (CLIP)
