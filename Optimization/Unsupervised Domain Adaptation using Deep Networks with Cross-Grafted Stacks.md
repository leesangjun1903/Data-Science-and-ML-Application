# Unsupervised Domain Adaptation using Deep Networks with Cross-Grafted Stacks

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

이 논문은 인간의 인지 과정(조류·자동차 전문가가 얼굴 인식 영역을 활용해 새로운 객체를 인식하는 현상, Gauthier et al., 2000)에서 영감을 받아, **Cross-Grafted Representation Stacks(CGRS)** 기반의 비지도 도메인 적응(Unsupervised Domain Adaptation, UDA) 모델(UDAR)을 제안합니다.

핵심 주장은 다음과 같습니다:
> 서로 다른 도메인의 디코더에서 얻은 **다층 수용 야(receptive field)**를 교차 접합(cross-graft)하여 중간 연관 공간(association space)을 생성하고, 이를 통해 도메인 갭을 효과적으로 줄일 수 있다.

### 주요 기여

| 기여 | 설명 |
|------|------|
| **CGRS 구조** | 소스/타겟의 VAE 디코더 계층을 교차 접합하여 중간 연관 공간 생성 |
| **디커플링** | CGRS는 자체 도메인 네트워크와 분리되어 전이 및 재활용 가능 |
| **이중 채널** | $\mathbf{X}^{st}$, $\mathbf{X}^{ts}$ 두 채널로 완전한 연관 공간 구성 |
| **일반화 능력** | 학습된 CGRS를 미학습 도메인 시나리오에 전이 가능 |
| **생성적 해석 가능성** | 연관 이미지의 시각적 확인을 통한 적응 과정 이해 가능 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**비지도 도메인 적응(UDA)**에서 소스 도메인 $\mathbf{X}_s = \{x_i^s, y_i^s\}\_{i=1}^{n_s}$(레이블 있음)와 타겟 도메인 $\mathbf{X}_t = \{x_i^t\}\_{i=1}^{n_t}$(레이블 없음) 간의 분포 불일치 문제를 해결합니다.

- 데이터 분포: $x_i^s \sim \mathcal{P}(\mathbf{X}_s, Y_s)$, $x_i^t \sim \mathcal{Q}(\mathbf{X}_t, Y_t)$, $\mathcal{P} \neq \mathcal{Q}$
- 기존 방법들은 도메인 불변 특징 학습에만 집중하여 **중간 연관 공간의 명시적 생성 및 활용**이 부족함
- 도메인 간 직접 변환 시 적대적 학습이 불안정해질 수 있는 문제

### 2.2 제안하는 방법 및 수식

#### Phase 1: 정보 인코딩 (VAE, Module A)

VAE를 활용하여 소스와 타겟의 잠재 인코딩 $z_s$, $z_t$를 획득합니다. 공유 잠재 공간은 사전 분포 $\mathcal{N}(0, I)$를 따릅니다.

**VAE 손실 함수:**

$$L_{VAEs} = L_{like}^{pixel} + L_{prior} \tag{8}$$

$$L_{like}^{pixel} = -\lambda_1 \{ \mathbb{E}_{q_s(z_s|\mathbf{X}_s)}[\log p_s(\mathbf{X}_s|z_s)] + \mathbb{E}_{q_t(z_t|\mathbf{X}_t)}[\log p_t(\mathbf{X}_t|z_t)] \} \tag{9}$$

$$L_{prior} = \lambda_2 \{ D_{KL}(q_s(z_s|x_s) \| p(z)) + D_{KL}(q_t(z_t|x_t) \| p(z)) \} \tag{10}$$

여기서 $D_{KL}$은 Kullback-Leibler 발산, $\lambda_1$, $\lambda_2$는 하이퍼파라미터입니다.

#### Phase 2: 연관 공간 생성 (CGRS, Module B)

CGRS는 잠재 공간 $z = \{z_s, z_t\}$를 새로운 분포 $\mathcal{P}$로 매핑합니다:

$$\mathcal{D}(z) \mapsto \mathbf{X} \in \mathcal{P} \tag{1}$$

**고수준 디코더 계층을 통한 계층적 전이:**

$$\mathcal{P}_i = \{p_i(\mathbf{m}_1 | z), p_i(\mathbf{m}_2 | \mathbf{m}_1), \ldots, p_i(\mathbf{m}_N | \mathbf{m}_{N-1})\} \tag{2}$$

**저수준 디코더 계층을 통한 최종 연관 공간 생성:**

$$\mathcal{P}_{ij} = \{p_j(\mathbf{n}_1 | \mathbf{m}_N), p_j(\mathbf{n}_2 | \mathbf{n}_1), \ldots, p_j(\mathbf{n}_M | \mathbf{n}_{M-1})\} \tag{3}$$

- $i, j \in \{s, t\}$: $i=s, j=t$이면 소스→타겟 전이 공간 $\mathcal{M}_{st}$
- $i=t, j=s$이면 타겟→소스 전이 공간 $\mathcal{M}_{ts}$

**교차 접합 매핑:**

$$z \oplus \Phi_s \oplus \Phi_t \rightarrow \mathbf{X}^{st} \tag{4}$$

$$z \oplus \Phi_t \oplus \Phi_s \rightarrow \mathbf{X}^{ts} \tag{5}$$

- $D^{st} \equiv D_s^h \circ D_t^l$ (소스 고수준 + 타겟 저수준)
- $D^{ts} \equiv D_t^h \circ D_s^l$ (타겟 고수준 + 소스 저수준)

#### Phase 3: 레이블 정렬 (GAN, Module C)

Jensen-Shannon 발산을 최소화하여 연관 분포를 정렬합니다:

$$p(\mathbf{X}_s^{st} | z_s, \theta_E, \theta_D) \Leftarrow p(\mathbf{X}_t^{st} | z_t, \theta_E, \theta_D, \theta_G), \quad \text{w.r.t.} \min JSD(p(\mathbf{X}_s^{st}) \| p(\mathbf{X}_t^{st})) \tag{6}$$

$$p(\mathbf{X}_s^{ts} | z_s, \theta_E, \theta_D) \Leftarrow p(\mathbf{X}_t^{ts} | z_t, \theta_E, \theta_D, \theta_G), \quad \text{w.r.t.} \min JSD(p(\mathbf{X}_s^{ts}) \| p(\mathbf{X}_t^{ts})) \tag{7}$$

**적대적 손실 함수:**

$$L_G^{st}(E_s, D^{st}, D_1) = \lambda_0 \{ \mathbb{E}_{x_s}[\log D_1(D^{st}(z_s))] + \mathbb{E}_{x_s, z_s}[\log(1 - D_1(G_1(D^{st}(z_t))))] \} \tag{11}$$

$$L_G^{ts}(E_t, D^{ts}, D_2) = \lambda_0 \{ \mathbb{E}_{x_t}[\log D_2(D^{ts}(z_s))] + \mathbb{E}_{x_t, z_t}[\log(1 - D_2(G_2(D^{ts}(z_t))))] \} \tag{12}$$

$$L_G = L_G^{st} + L_G^{ts} \tag{13}$$

**콘텐츠 일관성 손실 (Masked PMSE):**

$$L_s^{st} = \mathbb{E}_{\mathbf{X}_s^{st}, z}\left(\frac{1}{k}\|D^{st}(z_s) - G_1(D^{st}(z_t)) \circ \mathbf{m}\|_2^2 - \frac{1}{k^2}((D^{st}(z_s) - G_1(D^{st}(z_t)))^T \mathbf{m})^2\right) \tag{14}$$

$$L_s^{ts} = \mathbb{E}_{\mathbf{X}_s^{ts}, z}\left(\frac{1}{k}\|D^{ts}(z_s) - G_2(D^{ts}(z_t)) \circ \mathbf{m}\|_2^2 - \frac{1}{k^2}((D^{ts}(z_s) - G_2(D^{ts}(z_t)))^T \mathbf{m})^2\right) \tag{15}$$

$$L_s = \lambda_3(L_s^{st} + L_s^{ts}) \tag{16}$$

**분류 손실 (Softmax Cross-Entropy):**

$$L_T = \mathbb{E}[-y_s^T \log T(\mathbf{X}_s^{st}) - y_s^T \log T(\mathbf{X}_s^{ts})] \tag{17}$$

**전체 목적 함수 (Minimax):**

$$\min_{E, D, G} \max_{D_1, D_2} = L_{VAEs} + L_G + L_s + L_T \tag{18}$$

### 2.3 모델 구조

```
[소스 X_s] ──→ [Encoder E_s (공유 고수준)] ──→ z_s ──→ [Decoder D_s] ──→ X̂_s
                                                    ↓
                                              [CGRS Module B]
[타겟 X_t] ──→ [Encoder E_t (공유 고수준)] ──→ z_t ──→ [Decoder D_t] ──→ X̂_t
                                                    ↓
                              ┌─────────────────────┴─────────────────────┐
                         [D^st = D_s^h ∘ D_t^l]            [D^ts = D_t^h ∘ D_s^l]
                              │                                           │
                    X_s^st, X_t^st                              X_s^ts, X_t^ts
                              │                                           │
                    [Generator G1, Discriminator D1]   [Generator G2, Discriminator D2]
                              │                                           │
                         [Domain Confusion L_domain] + [Task Classification L_task]
```

**5개 모듈 구성:**
- **Module A**: 커플 VAE (소스/타겟 인코더-디코더, 고수준 공유)
- **Module B**: CGRS (교차 접합 연관 공간 생성)
- **Module C**: 도메인 정렬 및 전이 (GAN: $G_1, G_2$, $D_1, D_2$)
- **Module D**: 도메인 혼동 메트릭 ($L_{domain}$)
- **Module E**: 태스크 분류기 ($L_{task}$)

### 2.4 성능 향상 및 한계

#### 성능 결과 (Table 1 기반)

| 시나리오 | DANN | PixelDA | UNIT | UDAR($\mathbf{X}^{st}$) | UDAR($\mathbf{X}^{ts}$) | Target Only |
|----------|------|---------|------|----------|----------|------------|
| MNIST→MNIST-M | 0.766 | 0.982 | 0.920 | 0.890 | **0.983** | 0.983 |
| MNIST-M→MNIST | 0.851 | 0.922 | 0.932 | **0.983** | 0.871 | 0.985 |
| MNIST→USPS | 0.774 | 0.959 | 0.960 | **0.961** | 0.943 | 0.980 |
| USPS→MNIST | 0.833 | 0.942 | 0.951 | **0.956** | 0.953 | 0.985 |
| MNIST→M-Digits | 0.864 | 0.734 | 0.903 | **0.916** | 0.883 | 0.982 |
| Fashion→Fashion-M | 0.604 | 0.805 | 0.796 | 0.766 | **0.813** | 0.920 |

**대부분의 시나리오에서 SOTA 달성**, 특히 양방향 균형 성능이 우수합니다.

#### 한계점

1. **Fashion 시나리오 성능 저하**: 복잡한 텍스처 및 강한 노이즈 환경에서 연관 이미지의 정보 손실 발생
2. **채널 간 비대칭성**: MNIST↔MNIST-M에서 두 채널 간 정확도 차이 약 0.1로 비대칭적 표현 학습
3. **CGRS 구조 민감성**: 고/저수준 레이어 비율(H:L)에 따라 성능이 변동되며 최적 비율이 시나리오별로 상이
4. **실험 범위 제한**: MNIST 계열의 단순한 데이터셋 위주로 실험되어 복잡한 자연 이미지(예: VisDA, Office-31)에 대한 검증 부족
5. **계산 복잡도**: VAE + GAN + CGRS의 복합 구조로 인한 학습 비용 증가
6. **하이퍼파라미터 민감성**: $\lambda_0, \lambda_1, \lambda_2, \lambda_3$ 등 다수의 하이퍼파라미터 튜닝 필요

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 CGRS의 전이 가능성 (Table 2 기반)

논문의 핵심 주장 중 하나는 **CGRS의 일반화 능력**입니다. 하나의 시나리오에서 학습된 CGRS를 다른 시나리오에 재활용하는 실험 결과:

| 학습된 CGRS \ 테스트 시나리오 | MNIST→MNIST-M | MNIST→USPS | MNIST→M-Digits | Fashion→Fashion-M |
|------|------|------|------|------|
| MNIST→MNIST-M | 0.850 (0.983) | 0.958 (0.945) | 0.915 (0.883) | 0.809 (0.760) |
| Fashion→Fashion-M | 0.955 (0.881) | 0.932 (0.935) | 0.825 (0.913) | 0.766 (0.813) |

- CGRS를 고정(freeze)하고 적대적 정렬 및 레이블 정렬 부분만 파인튜닝해도 **합리적인 성능 유지**
- 특히 MNIST→MNIST-M 및 Fashion→Fashion-M의 CGRS가 다른 3개 시나리오 모두에서 잘 작동

### 3.2 일반화 향상 메커니즘 분석

**구조적 일반화:**
CGRS가 자체 도메인 네트워크와 디커플링되어 있어, 다양한 도메인 쌍에 유연하게 적용 가능합니다. $D^{st}$와 $D^{ts}$ 두 채널은 서로 보완적 역할을 하여 적응 견고성을 높입니다.

**CGRS 구조 민감도 분석 (Figure 5):**
- 콘텐츠 유사, 배경 상이(MNIST→MNIST-M, Fashion→Fashion-M): **H5L1** (고수준 비율 높을수록 유리)
- 배경 유사, 콘텐츠 상이(MNIST→USPS, MNIST→M-Digits): **H2L4** (저수준 비율 높을수록 유리)

> 이는 도메인 갭의 성격(스타일 vs. 구조적 차이)에 따라 적절한 표현 수준이 다름을 시사하며, 이를 통해 **태스크 특화 일반화 전략**이 가능합니다.

**준지도 학습(Semi-supervised) 시나리오 (Table 3):**

$$\text{타겟 레이블 1000개 추가} \Rightarrow \text{MNIST→MNIST-M: } 0.890 \rightarrow 0.988$$

소량의 타겟 레이블만으로도 성능이 크게 향상되어 **준지도 시나리오로의 확장성**이 높습니다.

**t-SNE 시각화 분석:**
두 채널 모두 소스($\bullet$ 파란색)와 타겟($\bullet$ 빨간색)의 특징 분포가 효과적으로 정렬됨을 확인. 특히 $\mathbf{X}^{ts}$ 채널이 MNIST→MNIST-M 시나리오에서 더 조밀한 정렬을 보임.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4.1 연구에 미치는 영향

**① 중간 연관 공간 패러다임의 확산**
도메인 간 직접 변환이 아닌 **중간 공간을 통한 점진적 적응** 패러다임을 명시적으로 제시했습니다. 이는 다단계 도메인 적응 연구의 이론적 기반을 제공합니다.

**② 신경과학 영감 AI 설계의 구체화**
인간 인지 메커니즘(전문가의 얼굴 인식 영역 활용)을 실제 딥러닝 구조로 구현한 선례를 남겨, 신경과학과 AI의 융합 연구에 방향성을 제시합니다.

**③ 디커플링 및 모듈화 설계의 중요성 강조**
CGRS의 모듈화 설계는 이후 **플러그인(plug-in) 방식 도메인 적응 모듈** 연구로 이어질 수 있습니다.

**④ 생성적 접근법의 해석 가능성**
연관 이미지의 시각화를 통해 도메인 갭 감소 과정을 직관적으로 이해할 수 있게 하여, **설명 가능한 AI(XAI)** 관점에서의 도메인 적응 연구에 기여합니다.

### 4.2 향후 연구 시 고려할 점

**① 복잡한 자연 이미지로의 확장**
MNIST 계열의 단순 데이터셋을 넘어 Office-31, VisDA-2017, DomainNet 등 대규모 복잡 데이터셋에서의 검증이 필요합니다. CGRS 구조가 고해상도 복잡 이미지에서도 효과적인지 확인해야 합니다.

**② 트랜스포머 기반 백본과의 결합**
2020년 이후 ViT(Vision Transformer) 등이 등장하면서, CNN 기반 VAE를 트랜스포머로 교체하거나 결합하는 연구가 필요합니다. 어텐션 메커니즘을 활용한 적응적 CGRS 구성이 유망한 방향입니다.

**③ 다중 소스 도메인 확장**
현재 모델은 단일 소스-단일 타겟 구조입니다. 다중 소스 도메인에서 CGRS를 구성하는 방법과 각 소스의 기여도를 동적으로 조절하는 메커니즘 연구가 필요합니다.

**④ 이론적 수렴 보장**
VAE + GAN의 복합 구조는 학습 불안정성 문제가 있습니다. 도메인 갭 감소에 대한 이론적 수렴 보장 및 오차 한계(error bound) 분석이 뒷받침되어야 합니다.

**⑤ 자동화된 CGRS 구조 탐색**
고/저수준 레이어 비율이 시나리오마다 다르므로, Neural Architecture Search(NAS)를 활용한 자동 최적 CGRS 구조 탐색 방법 연구가 필요합니다.

**⑥ 프라이버시 보존 도메인 적응**
연합 학습(Federated Learning) 환경에서 소스 도메인 데이터 없이 CGRS를 활용하는 방향도 고려할 수 있습니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의**: 아래 비교는 제가 학습한 데이터 기반이며, 각 논문의 세부 수치는 원문 확인을 권장합니다.

### 5.1 주요 후속 연구 동향

#### ① SHOT (Liang et al., ICML 2020)
- **논문**: "Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation"
- **핵심 차이**: 소스 데이터 없이 소스 모델의 가설(hypothesis)만으로 적응
- **CGRS와 비교**: CGRS는 소스 데이터 접근이 필요하지만, SHOT은 소스 데이터 없이도 작동 → **프라이버시 측면에서 SHOT이 유리**
- **한계**: 생성적 해석 가능성 부족

#### ② DAPL (Ge et al., CVPR 2022)
- **논문**: "Domain Adaptation via Prompt Learning"
- **핵심 차이**: CLIP 등 사전학습 모델에 프롬프트를 활용한 도메인 적응
- **CGRS와 비교**: CGRS는 CNN 기반 생성적 접근, DAPL은 대형 사전학습 모델 활용 → **스케일과 적용 범위에서 큰 차이**

#### ③ CDTrans (Xu et al., ICLR 2022)
- **논문**: "CDTrans: Cross-domain Transformer for Unsupervised Domain Adaptation"
- **핵심 차이**: 크로스 어텐션을 통해 소스-타겟 특징 상호작용
- **CGRS와 비교**: CGRS의 교차 접합 아이디어와 유사한 철학(교차 도메인 정보 융합), 트랜스포머로 발전된 형태로 해석 가능
- **성능**: Office-Home에서 73.7% 달성

#### ④ SSRT (Sun et al., CVPR 2022)
- **논문**: "Safe Self-Refinement for Transformer-based Domain Adaptation"
- **핵심 차이**: 트랜스포머 기반 자기 정제(self-refinement) 전략
- **CGRS와 비교**: 생성 모델 없이 특징 정렬에 집중, CGRS의 생성적 중간 공간 개념은 다른 방향

### 5.2 종합 비교표

| 항목 | CGRS(UDAR) | SHOT (2020) | CDTrans (2022) | SSRT (2022) |
|------|------------|-------------|----------------|-------------|
| **백본** | CNN+VAE | ResNet | ViT | ViT |
| **소스 데이터 필요** | ✅ | ❌ | ✅ | ✅ |
| **생성적 접근** | ✅ | ❌ | ❌ | ❌ |
| **해석 가능성** | 높음 | 낮음 | 중간 | 낮음 |
| **복잡 데이터셋 검증** | 제한적 | ✅ | ✅ | ✅ |
| **모듈 전이성** | ✅ | ❌ | 부분적 | ❌ |
| **학습 안정성** | 중간(GAN) | 높음 | 높음 | 높음 |

### 5.3 CGRS 관점에서의 시사점

2020년 이후 연구 트렌드는:
1. **트랜스포머 백본 전환** (CNN → ViT)
2. **소스 프리(source-free) UDA** 증가
3. **프롬프트 학습 및 대형 사전학습 모델 활용**

CGRS의 **교차 도메인 정보 융합 철학**은 CDTrans 등에서 크로스 어텐션 형태로 계승되고 있으나, **생성적 중간 공간**이라는 독특한 접근은 아직 충분히 탐구되지 않은 영역입니다.

---

## 참고 자료

1. **주 논문**: Jinyong Hou, Xuejie Ding, Jeremiah D. Deng, "Unsupervised Domain Adaptation using Deep Networks with Cross-Grafted Stacks," arXiv:1902.06328v1, 2019.

2. **비교 논문들 (논문 내 인용)**:
   - Ganin et al., "Domain-adversarial training of neural networks," JMLR 2016 [7]
   - Bousmalis et al., "Unsupervised pixel-level domain adaptation with generative adversarial networks (PixelDA)," CVPR 2017 [4]
   - Liu et al., "Unsupervised image-to-image translation networks (UNIT)," NIPS 2017 [20]
   - Tzeng et al., "Adversarial discriminative domain adaptation (ADDA)," CVPR 2017 [31]
   - Kingma & Welling, "Auto-encoding variational bayes," arXiv 2013 [16]
   - Goodfellow et al., "Generative adversarial nets," NIPS 2014 [11]
   - Gauthier et al., "Expertise for cars and birds recruits brain areas involved in face recognition," Nature Neuroscience 2000 [8]

3. **2020년 이후 비교 연구** (학습 데이터 기반, 원문 확인 권장):
   - Liang et al., "Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation (SHOT)," ICML 2020
   - Xu et al., "CDTrans: Cross-domain Transformer for Unsupervised Domain Adaptation," ICLR 2022
   - Sun et al., "Safe Self-Refinement for Transformer-based Domain Adaptation (SSRT)," CVPR 2022
