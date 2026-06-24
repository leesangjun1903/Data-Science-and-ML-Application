# Transferable Attention for Domain Adaptation (TADA)

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장
기존 적대적 도메인 적응(adversarial domain adaptation) 방법들은 이미지 전체를 전역적으로 정렬(align)하는 방식을 사용하지만, **이미지의 모든 영역이 동등하게 전이 가능(transferable)한 것은 아니다.** 배경과 같은 비전이 가능한 영역을 강제로 정렬하면 **부정적 전이(negative transfer)**가 발생할 수 있으며, 도메인 간 유사도가 낮은 이미지 역시 강제 정렬 시 해롭다.

### 주요 기여
| 기여 항목 | 내용 |
|---|---|
| **전이 가능 주의 메커니즘 제안** | 영역 수준(local)과 이미지 수준(global)의 두 가지 보완적 주의 메커니즘 도입 |
| **전이 가능 지역 주의(Local Attention)** | 다중 영역별 도메인 판별기를 통해 전이 가능한 영역을 강조 |
| **전이 가능 전역 주의(Global Attention)** | 단일 이미지 수준 도메인 판별기를 통해 전이 가능한 이미지를 강조 |
| **주의 기반 엔트로피 손실** | 전이 가능성이 높은 이미지의 예측 확실성을 향상시키는 주의 엔트로피 손실 설계 |
| **성능** | Office-31, Office-Home 벤치마크에서 당시 최고 성능(SOTA) 달성 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**문제 1: 영역 수준의 불균등한 전이 가능성**
- 이미지의 배경 영역은 분류 과제와 관련이 없으며, 이를 강제로 정렬하면 부정적 전이가 발생
- 기존 방법들은 전체 이미지의 feature를 하나의 단위로 처리하여 세밀한 구조를 무시

**문제 2: 이미지 수준의 불균등한 전이 가능성**
- 도메인 간 특징 공간에서 매우 이질적인 이미지는 전이 가능성이 낮음
- 이런 이미지에 대해 entropy를 강제로 최소화하면 분류기 혼란 야기

**설정: 비지도 도메인 적응(Unsupervised Domain Adaptation)**

$$\mathcal{D}_s = \{(\boldsymbol{x}_i^s, y_i^s)\}_{i=1}^{n_s}, \quad \mathcal{D}_t = \{\boldsymbol{x}_j^t\}_{j=1}^{n_t}$$

소스 도메인은 레이블이 있고, 타겟 도메인은 레이블이 없으며 두 도메인은 **서로 다른 확률 분포**를 따름.

---

### 2.2 제안하는 방법 (수식 포함)

#### 기반: DANN의 목적 함수

$$C_0(\theta_f, \theta_y, \theta_d) = \frac{1}{n_s} \sum_{\boldsymbol{x}_i \in \mathcal{D}_s} L_y(G_y(G_f(\boldsymbol{x}_i)), y_i) - \frac{\lambda}{n} \sum_{\boldsymbol{x}_i \in \mathcal{D}_s \cup \mathcal{D}_t} L_d(G_d(G_f(\boldsymbol{x}_i)), d_i) \tag{1}$$

여기서 $n = n_s + n_t$, $\lambda$는 분류 손실과 도메인 적응 손실 사이의 균형 하이퍼파라미터.

---

#### (A) 전이 가능 지역 주의(Transferable Local Attention)

**Step 1: 다중 영역별 도메인 판별기 손실**

단일 도메인 판별기 $G_d$를 $K$개의 영역별 판별기 $G_d^k$ ($k=1,2,...,K$)로 분할:

$$L_l = \frac{1}{Kn} \sum_{k=1}^{K} \sum_{\boldsymbol{x}_i \in \mathcal{D}_s \cup \mathcal{D}_t} L_d(G_d^k(\boldsymbol{f}_i^k), d_i) \tag{2}$$

- $\boldsymbol{f}_i^k = G_f(\boldsymbol{x}_i)^k$: 이미지 $i$의 영역 $k$에서의 feature
- ResNet-50 기준 마지막 합성곱 레이어: $7 \times 7 \times 2048$ → $K = 49$

**Step 2: 지역 주의 값 생성 (엔트로피 기반)**

각 영역 $k$의 도메인 판별기 출력 $\hat{d}_i^k = G_d^k(\boldsymbol{f}_i^k)$에 대해:

$$w_i^k = 1 - H(\hat{d}_i^k) \tag{3}$$

- $H(p) = -\sum_j p_j \cdot \log(p_j)$: 엔트로피 함수
- 판별기가 소스/타겟 구분에 **불확실할수록** 엔트로피가 높아 → 해당 영역이 더 전이 가능 → 주의 값 $w_i^k$ 감소
- 반대로 판별기가 **확실하게 구분**할 수 있는 영역(소스에만 있는 배경 등) → 주의 값 작음

> **직관**: 두 도메인에서 비슷하게 보이는 영역일수록 판별기가 혼동 → 엔트로피 높음 → $w_i^k$ 작음. 잠깐, 이 방향을 재확인하면: $w_i^k = 1 - H(\hat{d}_i^k)$이므로, 판별기가 **확신을 가지고** 한 도메인으로 분류 → 엔트로피 낮음 → $w_i^k$ 높음. 즉, **이미 잘 정렬된(전이 가능성이 증명된) 영역**에 더 큰 가중치를 부여.

**Step 3: 잔차 연결(Residual Connection)을 통한 주의 적용**

$$\boldsymbol{h}_i^k = (1 + w_i^k) \cdot \boldsymbol{f}_i^k \tag{4}$$

잔차 연결로 잘못된 주의(wrong attention)의 부정적 영향을 완화.

---

#### (B) 전이 가능 전역 주의(Transferable Global Attention)

**Step 1: 전역 도메인 판별기 손실**

$$L_g = \frac{1}{n} \sum_{\boldsymbol{x}_i \in \mathcal{D}_s \cup \mathcal{D}_t} L_d(G_d(G_b(\boldsymbol{h}_i), d_i)) \tag{5}$$

$G_b$: 병목(bottleneck) 레이어.

**Step 2: 전역 주의 값 생성**

$$m_i = 1 + H(\hat{d}_i) \tag{6}$$

- $\hat{d}_i = G_d(G_b(\boldsymbol{h}_i))$: 전역 판별기 출력
- 이미지가 도메인 간에 **유사할수록** → 판별기가 혼동 → 엔트로피 높음 → $m_i$ 커짐
- 반대로 이질적인 이미지 → 판별기가 확신 → 엔트로피 낮음 → $m_i$ 작음

**Step 3: 주의 엔트로피 손실(Attentive Entropy Loss)**

$$L_h = -\frac{1}{n} \sum_{\boldsymbol{x}_i \in \mathcal{D}_s \cup \mathcal{D}_t} \sum_{j=1}^{c} m_i \cdot p_{i,j} \cdot \log(p_{i,j}) \tag{7}$$

- $c$: 클래스 수, $p_{i,j}$: 이미지 $i$를 클래스 $j$로 예측할 확률
- 전이 가능성이 높은 이미지($m_i$ 큰)에 대해 엔트로피 최소화를 더 강하게 적용 → 예측 확실성 향상

---

#### (C) 분류 손실

$$L_y = \frac{1}{n_s} \sum_{\boldsymbol{x}_i \in \mathcal{D}_s} L_y(G_y(G_b(\boldsymbol{h}_i)), y_i) \tag{8}$$

---

#### (D) 최종 통합 목적 함수 (TADA)

$$C(\theta_f, \theta_b, \theta_y, \theta_d, \theta_d^k|_{k=1}^{K}) = L_y + \gamma L_h - \lambda(L_g + L_l)$$

$$= \frac{1}{n_s} \sum_{\boldsymbol{x}_i \in \mathcal{D}_s} L_y(G_y(G_b(\boldsymbol{h}_i)), y_i)$$

$$- \frac{\gamma}{n} \sum_{\boldsymbol{x}_i \in \mathcal{D}} \sum_{j=1}^{C} m_i \cdot p_{i,j} \cdot \log(p_{i,j})$$

$$- \frac{\lambda}{n} \left[ \sum_{\boldsymbol{x}_i \in \mathcal{D}} L_d(G_d(G_b(\boldsymbol{h}_i), d_i)) + \frac{1}{K} \sum_{k=1}^{K} \sum_{\boldsymbol{x}_i \in \mathcal{D}} L_d(G_d^k((G_f(\boldsymbol{x}_i))^k), d_i) \right] \tag{9}$$

**미니맥스 최적화:**

$$(\hat{\theta}_f, \hat{\theta}_b, \hat{\theta}_y) = \arg\min_{\theta_f, \theta_b, \theta_y} C\left(\theta_f, \theta_b, \theta_y, \theta_d, \theta_d^k|_{k=1}^{K}\right)$$

$$(\hat{\theta}_d, \hat{\theta}_d^1, ..., \hat{\theta}_d^K) = \arg\max_{\theta_d, \theta_d^1, ..., \theta_d^K} C\left(\theta_f, \theta_b, \theta_y, \theta_d, \theta_d^k|_{k=1}^{K}\right) \tag{10}$$

하이퍼파라미터: $\lambda = 1.0$, $\gamma = 0.1$ (전체 실험에서 고정)

---

### 2.3 모델 구조

```
[Source/Target Image]
        ↓
  [Feature Extractor Gf]  ←── ResNet-50 (ImageNet pretrained)
        ↓                 ↘
  [Local Features fᵢᵏ]    [Region-wise Discriminators Gd¹...GdK]
  (K=49 regions)                    ↓
        ↓               [Local Attention wᵢᵏ = 1 - H(d̂ᵢᵏ)]
  [Attended Features hᵢᵏ = (1+wᵢᵏ)·fᵢᵏ]
        ↓
  [Bottleneck Layer Gb]
        ↓              ↘
  [Global Discriminator Gd]
        ↓
  [Global Attention mᵢ = 1 + H(d̂ᵢ)]
        ↓
  [Classifier Gy] → [Prediction] → [Classification Loss + Attentive Entropy Loss]
```

**구성 요소:**
- **백본**: ResNet-50 (ImageNet 사전학습)
- **지역 판별기**: $K=49$개 (7×7 feature map 기반), 각각 독립적
- **병목 레이어**: 전역 feature 추출
- **전역 판별기**: 1개 (이미지 전체 수준)
- **분류기**: 소스 레이블로 학습

---

### 2.4 성능 향상

#### Office-31 결과 (ResNet-50)

| 방법 | A→W | D→W | W→D | A→D | D→A | W→A | **Avg** |
|---|---|---|---|---|---|---|---|
| ResNet-50 | 68.4 | 96.7 | 99.3 | 68.9 | 62.5 | 60.7 | 76.1 |
| DANN | 82.0 | 96.9 | 99.1 | 79.7 | 68.2 | 67.4 | 82.2 |
| MADA | 90.0 | 97.4 | 99.6 | 87.8 | 70.3 | 66.4 | 85.2 |
| GTA | 89.5 | 97.9 | 99.8 | 87.7 | 72.8 | 71.4 | 86.5 |
| **TADA (local+global)** | **94.3** | **98.7** | **99.8** | **91.6** | **72.9** | **73.0** | **88.4** |

#### Office-Home 결과 (ResNet-50)

| 방법 | Avg |
|---|---|
| ResNet-50 | 46.1 |
| JAN | 58.3 |
| TADA (local) | 63.9 |
| TADA (global) | 65.7 |
| **TADA (local+global)** | **67.6** |

**주목할 점**: Pr→Cl과 같은 어려운 전이 태스크에서 JAN 대비 10점 이상 향상.

---

### 2.5 한계점

| 한계 | 설명 |
|---|---|
| **소규모 데이터 취약성** | W(795장), D(498장)처럼 데이터가 적을 때 지역 주의 학습이 불충분 |
| **계산 비용 증가** | K=49개의 독립적 도메인 판별기로 인한 파라미터 및 연산량 증가 |
| **하이퍼파라미터 민감성** | $\lambda$, $\gamma$ 고정값 사용 → 태스크별 최적화 미흡 가능 |
| **이미지 분류에 특화** | 객체 검출, 시맨틱 분할 등 다른 태스크로의 직접 확장성 미검증 |
| **단일 모달리티** | 텍스트, 3D 등 다양한 모달리티에 대한 적용 연구 부재 |
| **Region 분할 방식 고정** | 7×7 grid 기반 고정 분할 → 의미론적 영역(semantic region)과 불일치 가능 |

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화를 향상시키는 메커니즘

**① 세밀한 분포 정렬(Fine-grained Distribution Alignment)**

기존 방법이 이미지 전체 feature의 분포를 정렬하는 것과 달리, TADA는 **영역별로 독립적인 분포 정렬**을 수행:

$$L_l = \frac{1}{Kn} \sum_{k=1}^{K} \sum_{\boldsymbol{x}_i \in \mathcal{D}_s \cup \mathcal{D}_t} L_d(G_d^k(\boldsymbol{f}_i^k), d_i)$$

이를 통해 멀티모달 분포 구조를 더 정밀하게 포착 → **타겟 도메인에서의 클래스 경계 선명화**

**② 부정적 전이 억제**

$w_i^k = 1 - H(\hat{d}_i^k)$를 통해 배경처럼 전이에 방해되는 영역의 가중치를 낮춤:

$$\boldsymbol{h}_i^k = (1 + w_i^k) \cdot \boldsymbol{f}_i^k$$

잔차 연결 구조($1 + w_i^k$)는 최소 원본 feature를 보존하여 극단적인 주의 오류를 방지.

**③ 선택적 예측 신뢰도 향상**

$m_i = 1 + H(\hat{d}_i)$를 통해 도메인 간 유사한 이미지에만 강한 엔트로피 최소화 적용:

$$L_h = -\frac{1}{n} \sum_{\boldsymbol{x}_i} \sum_{j=1}^{c} m_i \cdot p_{i,j} \cdot \log(p_{i,j})$$

→ **타겟 도메인에서 예측 신뢰도 향상** + 이질적 이미지에 대한 과신(overconfidence) 방지

**④ 보완적 주의의 시너지**

Ablation study 결과:
- TADA (local only): Avg 84.5%
- TADA (global only): Avg 86.7%
- TADA (local+global): **88.4%**

두 모듈의 결합이 각각보다 큰 향상 → 지역 주의의 세밀성 + 전역 주의의 강건성이 상호 보완.

**⑤ t-SNE 시각화**
- ResNet → DANN → MADA → TADA 순으로 소스/타겟 도메인 feature가 점점 더 명확하게 31개 클러스터로 분리
- TADA에서 클래스 간 경계가 가장 뚜렷 → **타겟 도메인에서의 분류 일반화 능력 최고**

### 3.2 일반화 측면에서의 이론적 근거

**Ben-David et al.의 이론적 프레임워크**에 따르면, 타겟 도메인 오류의 상한은:

$$\epsilon_t(h) \leq \epsilon_s(h) + d_{\mathcal{H}\Delta\mathcal{H}}(\mathcal{D}_s, \mathcal{D}_t) + \lambda^*$$

- $d_{\mathcal{H}\Delta\mathcal{H}}$: 도메인 간 발산(divergence)
- TADA의 지역/전역 주의는 **전이 가능한 영역/이미지에 집중하여 $d_{\mathcal{H}\Delta\mathcal{H}}$를 효과적으로 감소**

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 연구에 미치는 영향

**① 어텐션 기반 도메인 적응의 패러다임 제시**

TADA는 "모든 영역/이미지를 동등하게 취급하지 말라"는 원칙을 최초로 체계적으로 도입. 이후 연구들의 중요한 설계 원칙으로 자리잡음.

**② 엔트로피를 전이 가능성 측도로 활용**

$w_i^k = 1 - H(\hat{d}_i^k)$와 $m_i = 1 + H(\hat{d}_i)$는 레이블 없이도 전이 가능성을 정량화하는 우아한 방법을 제시. 이후 다양한 무감독 학습 시나리오에 응용.

**③ 다중 판별기 + 주의 메커니즘의 통합**

MADA(다중 클래스별 판별기)를 영역 수준으로 확장하는 아이디어를 제공.

**④ 객체 탐지/분할에서의 도메인 적응 연구 촉진**

영역별 전이 가능성의 아이디어는 이후 객체 탐지의 도메인 적응 연구(region proposal 기반)에 영향.

### 4.2 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의**: 아래 연구들에 대한 일부 세부 내용은 제 학습 데이터의 한계로 인해 완전한 정확성을 보장하기 어렵습니다. 논문 제목과 핵심 방향성을 중심으로 서술하며, 구체적 수치는 원문 확인을 권장합니다.

| 연구 | 발표 | TADA와의 관계 | 핵심 개선점 |
|---|---|---|---|
| **CDAN (Long et al., 2018→NeurIPS)** | NeurIPS 2018 | 조건부 적대적 정렬 | 클래스 조건부 도메인 판별기로 다중모달 구조 포착 |
| **ATDOC (Liu et al., 2021)** | CVPR 2021 | 어텐션 기반 타겟 도메인 최적화 | 타겟 도메인 데이터에 대한 더 세밀한 어텐션 |
| **CDTrans (Xu et al., 2021)** | arXiv 2021 | Transformer 기반 | Cross-domain Transformer로 지역 특징 전이 |
| **TVT (Yang et al., 2021)** | arXiv 2021 | Vision Transformer 도입 | ViT의 self-attention을 도메인 적응에 직접 활용 |
| **PMTrans (Zhu et al., 2022)** | ECCV 2022 | Patch-level 전이 | Patch Mix를 통한 세밀한 도메인 전이 |

**주요 트렌드 비교:**

```
TADA (2019)          →    Transformer 시대 (2021~)
────────────────────────────────────────────────
CNN + 수동 grid 분할  →    ViT self-attention 자동 분할
엔트로피 기반 주의    →    Query-Key-Value 기반 주의
49개 고정 영역        →    가변적 의미론적 패치
ResNet-50 백본        →    ViT, Swin Transformer 백본
```

**TADA의 지속적 관련성:**
- TADA의 **"전이 가능한 영역에 집중"** 원칙은 Transformer 기반 방법에서도 핵심 아이디어로 유지
- Self-attention의 attention map이 TADA의 local attention과 유사한 역할 수행

### 4.3 향후 연구 시 고려할 점

**① 의미론적 영역 분할 도입**
- 현재 7×7 고정 grid → **Superpixel, SAM(Segment Anything Model)** 기반 의미론적 영역 분할로 대체
- 의미 있는 객체 단위로 전이 가능성 측정 가능

**② Vision Transformer와의 통합**
- ViT의 patch-level self-attention과 TADA의 local attention을 결합
- Multi-head attention의 각 head를 다른 영역의 전이 가능성 측정에 활용

$$w_i^k \leftarrow \text{Attention}(Q_k, K_k, V_k) \cdot (1 - H(\hat{d}_i^k))$$

**③ 준지도/소수샷 시나리오 확장**
- 타겟 도메인에 소수의 레이블이 있는 경우(few-shot DA)에서의 TADA 적용
- 레이블 있는 타겟 샘플로 global attention 개선

**④ 다중 도메인 및 도메인 일반화(Domain Generalization)**
- 단일 소스 → 타겟 페어를 넘어 **다중 소스 도메인**에서의 전이 가능 주의 연구
- 특정 타겟 도메인 없이 일반화 가능한 모델 학습

**⑤ 자기 지도 학습(Self-supervised Learning)과의 결합**
- CLIP, MAE 등 대규모 사전학습 모델의 표현을 활용하여 전이 가능성 측정 정확도 향상

**⑥ 이론적 보장 강화**
- 현재 엔트로피 기반 전이 가능성의 이론적 근거 부족
- **PAC-Bayesian** 또는 **정보 이론적 프레임워크**로 bound 도출

**⑦ 적응형 하이퍼파라미터**
- $\lambda=1.0$, $\gamma=0.1$ 고정값의 한계 극복
- **메타 학습(meta-learning)** 기반 자동 하이퍼파라미터 조정

**⑧ 계산 효율성 개선**
- K=49개 판별기의 병렬 처리 및 경량화
- **지식 증류(Knowledge Distillation)**를 통한 다중 판별기 압축

---

## 참고자료

**주요 참고 논문 (논문 내 인용):**
- Wang, X., Li, L., Ye, W., Long, M., Wang, J. (2019). **"Transferable Attention for Domain Adaptation."** AAAI 2019. *(본 논문)*
- Ganin, Y., Lempitsky, V. (2015). **"Unsupervised Domain Adaptation by Backpropagation."** ICML 2015.
- Long, M. et al. (2018). **"Conditional Adversarial Domain Adaptation."** NeurIPS 2018.
- Pei, Z. et al. (2018). **"Multi-Adversarial Domain Adaptation."** AAAI 2018.
- Vaswani, A. et al. (2017). **"Attention Is All You Need."** NeurIPS 2017.
- Grandvalet, Y., Bengio, Y. (2005). **"Semi-supervised Learning by Entropy Minimization."** NeurIPS 2005.
- Wang, F. et al. (2017). **"Residual Attention Network for Image Classification."** CVPR 2017.
- He, K. et al. (2016). **"Deep Residual Learning for Image Recognition."** CVPR 2016.
- Saenko, K. et al. (2010). **"Adapting Visual Category Models to New Domains."** ECCV 2010. *(Office-31)*
- Venkateswara, H. et al. (2017). **"Deep Hashing Network for Unsupervised Domain Adaptation."** arXiv. *(Office-Home)*

**2020년 이후 비교 연구 (참고):**
- Xu, M. et al. (2021). **"CDTrans: Cross-Domain Transformer for Unsupervised Domain Adaptation."** arXiv:2109.06165.
- Yang, J. et al. (2021). **"TVT: Transferable Vision Transformer for Unsupervised Domain Adaptation."** arXiv:2108.05988.

> **면책 사항**: 2020년 이후 최신 연구에 대한 구체적 성능 수치는 원문 논문을 직접 확인하시기를 권장합니다. TADA 논문 자체의 내용은 제공된 PDF를 기반으로 정확하게 서술하였습니다.
