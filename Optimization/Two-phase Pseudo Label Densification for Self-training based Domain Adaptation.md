# Two-phase Pseudo Label Densification for Self-training based Domain Adaptation (TPLD) 

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

기존 self-training 기반 UDA(Unsupervised Domain Adaptation) 방법들은 **신뢰도가 높은 예측값만을 pseudo label로 선택**하기 때문에, 실제 학습 과정에서 **희소한(sparse) pseudo label** 문제가 필연적으로 발생한다. 이 논문은 이를 UDA self-training의 **근본적인 한계**로 규정하고, 이를 단계적으로 해결하는 **Two-phase Pseudo Label Densification (TPLD)** 프레임워크를 제안한다.

### 주요 기여

| # | 기여 내용 |
|---|-----------|
| 1 | Pseudo label densification을 self-training 기반 도메인 적응에서 **최초로 공식 정의 및 탐구** |
| 2 | **1단계: Voting 기반 densification** — 슬라이딩 윈도우를 통한 공간적 상관관계 활용 |
| 3 | **2단계: Easy-Hard 분류 기반 densification** — Easy 샘플은 full pseudo label, Hard 샘플은 adversarial learning 적용 |
| 4 | **Bootstrapping 메커니즘** 도입으로 noisy pseudo label 처리 강화 |
| 5 | CBST, CRST 등 기존 self-training 방법에 **plug-in** 방식으로 적용 가능 |
| 6 | GTA5→Cityscapes, SYNTHIA→Cityscapes 두 벤치마크에서 **state-of-the-art** 달성 |

---

## 2. 세부 분석

### 2.1 해결하고자 하는 문제

**문제의 본질: Sparse Pseudo Label**

기존 CBST/CRST는 아래 최적화 문제를 통해 pseudo label을 생성한다:

```math
\hat{y}_t^{(k)*} = \begin{cases} 1, & \text{if } k = \arg\max_k \left\{ \frac{p(k|\mathbf{x_t};\mathbf{w})}{\lambda_k} \right\} \text{ and } p(k|\mathbf{x_t};\mathbf{w}) > \lambda_k \\ 0, & \text{otherwise} \end{cases}
```

여기서 $\lambda_k$는 클래스 $k$에 대한 신뢰도 임계값이며, 초기 학습 단계에서는 상위 $p=0.2$ 비율만을 선택하기 때문에 pseudo label이 **매우 희소**해진다.

- 임계값을 단순히 낮추면(즉, $p$ 증가) → 초기 학습에서 **noisy, 비신뢰성 예측** 누적
- Sparse pseudo label → **부족한 학습 신호** → 최적 모델에서 이탈

### 2.2 제안하는 방법 및 수식

#### 기반: Self-training Loss (CRST 공식)

$$\min_{\mathbf{w}, \hat{\mathbf{Y}}_T} \mathcal{L}_{st}(\mathbf{w}, \hat{\mathbf{Y}}_T) = -\sum_{s \in S} \sum_{k=1}^{K} y_s^{(k)} \log p(k|\mathbf{x_s};\mathbf{w}) - \sum_{t \in T}\left[\sum_{k=1}^{K} \hat{y}_t^{(k)} \log \frac{p(k|\mathbf{x_t};\mathbf{w})}{\lambda_k} - \alpha r_c(\mathbf{w}, \hat{\mathbf{Y}}_T)\right] $$

- 첫 번째 항: 소스 도메인 학습 (source supervision)
- 두 번째 항: 타겟 도메인 pseudo label 재학습
- 세 번째 항: 과신뢰(overconfident) 방지를 위한 confidence regularizer ($\alpha r_c$)

#### Bootstrapping 메커니즘 (Noisy Label 처리)

$$\sum_{t \in T} \sum_{k=1}^{K} \left[\beta \hat{y}_t^{(k)} + (1-\beta) \frac{p(k|\mathbf{x_t};\mathbf{w})}{\lambda_k}\right] \log \frac{p(k|\mathbf{x_t};\mathbf{w})}{\lambda_k} $$

- $\beta$: pseudo label과 모델 예측의 혼합 비율
- noisy pseudo label의 영향을 완화하면서 모델 예측의 confidence를 동시에 향상

---

#### Phase 1: Voting 기반 Densification

슬라이딩 윈도우를 통해 **인접 픽셀의 공간적 상관관계**를 이용, 신뢰도 높은 예측을 주변 unlabeled 픽셀로 전파한다.

**pseudo label 생성 수식:**

```math
\hat{y}_t^{(k)*} = \begin{cases} 1, & \text{if } k = \arg\max_k \left\{ \frac{p(k|\mathbf{x_t};\mathbf{w})}{\lambda_k} \right\} \text{ and } p(k|\mathbf{x_t};\mathbf{w}) > \lambda_k \\ \mathbf{Voting}\!\left(\frac{p(k|\mathbf{x_t};\mathbf{w})}{\lambda_k}\right), & \text{otherwise} \end{cases}
```

**프로세스:**
1. Unlabeled 픽셀에서 상위 2개 경쟁 클래스 탐색
2. 해당 클래스들의 이웃 confident 값 pooling
3. 원래 예측값과 pooled 값의 weighted sum ($\alpha = 0.7$)
4. 임계값($\frac{p(k|\mathbf{x_t};\mathbf{w})}{\lambda_k} > 1$) 초과 시 pseudo label 할당

**1단계 목적함수:**

```math
\min_{\mathbf{w}, \hat{\mathbf{Y}}_T} \mathcal{L}_{st_1}(\mathbf{w}, \hat{\mathbf{Y}}_T) = -\sum_{s \in S}\sum_{k=1}^{K} y_s^{(k)} \log p(k|\mathbf{x_s};\mathbf{w}) - \sum_{t \in T}\left[\sum_{k=1}^{K}\left\{\beta\hat{y}_t^{(k)} + (1-\beta)\frac{p(k|\mathbf{x_t};\mathbf{w})}{\lambda_k}\right\}\log\frac{p(k|\mathbf{x_t};\mathbf{w})}{\lambda_k} - \alpha r_c(\mathbf{w}, \hat{\mathbf{Y}}_T)\right]
```

---

#### Phase 2: Easy-Hard 분류 기반 Densification

**이미지 수준 신뢰도 점수:**

$$conf_t = \frac{1}{K'}\sum_{k=1}^{K'} \frac{N_t^{k*}}{N_t^k} \cdot \frac{1}{\lambda_k}$$

- $N_t^k$: 클래스 $k$로 예측된 픽셀 수
- $N_t^{k*}$: $N_t^k$ 중 $\lambda_k$보다 높은 예측값을 가진 픽셀 수
- $\frac{1}{\lambda_k}$: 희귀 클래스 과소표현 방지 (rare class upsampling 효과)

상위 $q=30\%$를 **easy**, 나머지를 **hard** 샘플로 분류.

**Easy 샘플: Full pseudo label 생성**

```math
\hat{y}_{t_e}^{(k)*} = \begin{cases} 1, & \text{if } k = \arg\max_k \left\{ \frac{p(k|\mathbf{x_t};\mathbf{w})}{\lambda_k} \right\} \text{ and } p(k|\mathbf{x_t};\mathbf{w}) > \lambda_k \\ \left(\frac{p(k|\mathbf{x_t};\mathbf{w})}{\lambda_k}\right)^{\gamma}, & \text{otherwise} \end{cases}
```

- $\gamma=2$: 낮은 prediction 값을 더 낮게 calibrate하여 noisy label 억제

**Easy 샘플 학습 목적함수:**

```math
\min_{\mathbf{w}, \hat{\mathbf{Y}}_T} \mathcal{L}_{st_2}(\mathbf{w}, \hat{\mathbf{Y}}_T) = -\sum_{t \in T}\left[\sum_{k=1}^{K}\left\{\beta\hat{y}_t^{(k)} + (1-\beta)\frac{p(k|\mathbf{x_t};\mathbf{w})}{\lambda_k}\right\}\log\frac{p(k|\mathbf{x_t};\mathbf{w})}{\lambda_k}\right]
```

**Hard 샘플: Intra-domain Adversarial Learning**

Discriminator $D_{intra}$의 학습 목적:

$$\min_{\theta_{D_{intra}}} \frac{1}{|e|}\sum_e L_{D_{intra}}(I_e, 1) + \frac{1}{|h|}\sum_h L_{D_{intra}}(I_h, 0) $$

Segmentation network의 adversarial 목적:

$$\min_{\theta_{seg}} \frac{1}{|h|}\sum_h L_{D_{intra}}(I_h, 1) $$

- $I_t$: 타겟 도메인의 **weighted self-information map**
- Hard 샘플의 feature를 Easy 샘플과 정렬 (hard-to-easy alignment)

---

### 2.3 모델 구조

```
┌─────────────────────────────────────────────────────────────┐
│                    TPLD Framework                           │
│                                                             │
│  ┌──────────────────┐     ┌──────────────────────────────┐  │
│  │   Phase 1 (6 rounds)   │   Phase 2 (3 rounds)         │  │
│  │                  │     │                              │  │
│  │ Source data ─────┤     │  ┌─────────┐  ┌──────────┐  │  │
│  │ Target data ─────┤     │  │  Easy   │  │   Hard   │  │  │
│  │       ↓          │     │  │ samples │  │ samples  │  │  │
│  │ Sliding Window   │     │  │Full PL  │  │ Adv.Loss │  │  │
│  │    Voting        │     │  │ L_st2   │  │ L_adv    │  │  │
│  │       ↓          │     │  └────┬────┘  └────┬─────┘  │  │
│  │  L_st1 (Eq.5)   │     │       └─────┬───────┘        │  │
│  │ (Bootstrap+Vote) │     │             ↓                │  │
│  └──────────────────┘     │      Combined Loss           │  │
│          ↓ Model Transfer │                              │  │
│          └────────────────┘                              │  │
└─────────────────────────────────────────────────────────────┘
```

**백본 및 세그멘테이션 네트워크:**
- Backbone: VGG-16, ResNet-101 (ImageNet pre-trained)
- Segmentation: DeepLab-v2, DeepLab-v3
- Base framework: CRST-MRKLD

---

### 2.4 성능 향상

#### GTA5 → Cityscapes

| Method | Backbone | mIoU | R-mIoU |
|--------|----------|------|--------|
| CRST(MRKLD) | DeepLab-v2 + ResNet-101 | 47.0 | 30.3 |
| **TPLD** | DeepLab-v2 + ResNet-101 | **51.2 (+4.2)** | **35.1 (+4.8)** |
| CRST(MRKLD) | DeepLab-v3 + ResNet-101 | 41.2 | 23.7 |
| **TPLD** | DeepLab-v3 + ResNet-101 | **44.7 (+3.5)** | **27.3 (+3.6)** |

#### SYNTHIA → Cityscapes

| Method | Backbone | mIoU |
|--------|----------|------|
| CRST(MRKLD) | DeepLab-v3 + ResNet-101 | 48.1 |
| **TPLD** | DeepLab-v3 + ResNet-101 | **55.7 (+7.6)** |

#### 다양한 Self-training 방법에의 범용 적용 (GTA5→Cityscapes)

| Base Method | Base mIoU | +TPLD | 향상 |
|-------------|-----------|-------|------|
| CBST | 45.9 | 47.8 | +1.9 |
| CRST(LRENT) | 45.9 | 47.3 | +1.4 |
| CRST(MRKLD) | 47.0 | 51.2 | **+4.2** |

---

### 2.5 한계점

논문에서 명시적으로 인정하는 한계 및 분석을 통해 도출되는 한계:

1. **하이퍼파라미터 민감도**: $\alpha$, $q$, $\gamma$, voting 횟수/윈도우 크기 등 다수의 하이퍼파라미터가 존재하며, 새로운 도메인에서 재탐색이 필요
2. **Voting의 local 특성**: 슬라이딩 윈도우는 local 정보만 활용하므로, 넓은 영역의 semantic 정보 전파에는 한계 (반복 횟수 증가 시 smoothing 효과 및 noise 증가)
3. **단일 도메인 쌍 적용**: 실험이 합성→실제 시나리오(GTA5/SYNTHIA→Cityscapes)에만 한정되어, 다양한 실제→실제(real-to-real) 시나리오의 검증이 미흡
4. **계산 복잡도**: 9 라운드(1단계 6, 2단계 3)의 반복 학습 + adversarial discriminator 추가로 기존 대비 학습 비용 증가
5. **Semantic segmentation에 국한**: 객체 탐지, 깊이 추정 등 타 태스크로의 일반화는 미검증

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화 관련 핵심 메커니즘

**① 풍부한 학습 신호를 통한 일반화 향상**

Sparse pseudo label은 학습 신호 부족으로 이어지며, 이는 모델이 **sub-optimal**한 feature representation을 학습하게 만든다. TPLD는 pseudo label을 단계적으로 densify함으로써:

$$\text{Dense PL Coverage} \uparrow \Rightarrow \text{Training Signal} \uparrow \Rightarrow \text{Feature Generalization} \uparrow$$

**② Bootstrapping을 통한 Noise Robustness (일반화의 전제 조건)**

$$\mathcal{L}_{bootstrap} = \sum_{t \in T}\sum_{k=1}^{K}\left[\beta\hat{y}_t^{(k)} + (1-\beta)\frac{p(k|\mathbf{x_t};\mathbf{w})}{\lambda_k}\right]\log\frac{p(k|\mathbf{x_t};\mathbf{w})}{\lambda_k}$$

- $\beta$ 조정으로 pseudo label과 모델 예측을 혼합 → noisy label에 대한 **과적합 방지**
- 모델이 ground truth에 가까운 easy 샘플 위주로 학습하면서도 hard 샘플에 대한 **편향 방지**

**③ Easy-Hard 분류와 intra-domain 정렬**

$$conf_t = \frac{1}{K'}\sum_{k=1}^{K'}\frac{N_t^{k*}}{N_t^k} \cdot \frac{1}{\lambda_k}$$

- 이미지 수준의 신뢰도 기반 분류 → 학습 커리큘럼(curriculum learning) 효과
- Hard 샘플에 대한 adversarial alignment:

$$\min_{\theta_{seg}}\frac{1}{|h|}\sum_h L_{D_{intra}}(I_h, 1)$$

이를 통해 hard 샘플(target domain에서 어려운 케이스)의 feature가 easy 샘플과 정렬되어 **intra-domain 분포 일관성** 향상 → 타겟 도메인 전반에서의 일반화 성능 개선

**④ 희귀 클래스 커버리지 향상**

$\frac{1}{\lambda_k}$ 가중치를 통해 희귀 클래스가 포함된 이미지를 easy 샘플로 더 적극적으로 포함:

$$R\text{-}mIoU: 30.3 \rightarrow 35.1 \quad (+4.8\%)$$

이는 **클래스 균형 측면의 일반화** 향상을 의미한다.

**⑤ Plug-in 범용성을 통한 일반화**

TPLD가 CBST, CRST(LRENT), CRST(MRKLD) 등 다양한 baseline에 일관적인 성능 향상을 보임은, 특정 방법에 과적합된 것이 아닌 **일반적인 일반화 메커니즘**임을 시사한다.

### 3.2 일반화 한계

- 실험이 **도시 주행 장면(Cityscapes)**에 특화 → 의료, 위성 등 다른 도메인에서의 검증 부재
- Voting의 공간적 상관관계 가정은 **텍스처가 균일하지 않은 도메인**에서는 성립하지 않을 수 있음
- **도메인 이동의 크기(domain gap)**에 따라 easy-hard 분류 비율 $q$의 최적값이 달라짐

---

## 4. 향후 연구에 미치는 영향 및 고려 사항

### 4.1 향후 연구에 미치는 영향

**① Pseudo Label Quality의 새로운 관점 제시**

기존 연구들이 pseudo label의 **정확도(precision)** 향상에 집중했다면, TPLD는 **커버리지(coverage/density)** 측면의 중요성을 처음으로 명시적으로 제기했다. 이는 이후 연구에서 pseudo label 품질의 다차원적 평가 기준으로 자리잡을 수 있다.

**② Curriculum Learning과 Domain Adaptation의 결합**

Easy-Hard 분류 전략은 curriculum learning의 UDA 응용으로 볼 수 있으며, 이후 연구들이 **동적 curriculum 설계**를 self-training에 통합하는 방향으로 발전하는 데 영향을 미쳤다.

**③ Intra-domain 정렬의 중요성 강조**

기존 UDA가 소스-타겟 **inter-domain** 정렬에 집중했다면, TPLD는 타겟 도메인 내부의 **intra-domain** 분포 일관성 확보의 중요성을 제시했다.

**④ 실용적 plug-in 프레임워크의 가치**

기존 self-training 방법에 minimal 수정으로 적용 가능한 설계는, 이후 **모듈식 UDA 연구**의 방향성을 제시한다.

---

### 4.2 2020년 이후 관련 최신 연구 비교 분석

아래 분석은 논문 원문에 직접 언급된 내용이 아닌, **공개적으로 잘 알려진 해당 분야 후속 연구**들과의 비교이므로 일부 수치에 대한 정확도에 유의하시기 바랍니다.

| 연구 | 핵심 아이디어 | TPLD와의 관계 |
|------|--------------|--------------|
| **DAFormer** (Hoyer et al., CVPR 2022) | Transformer 기반 backbone + Context-aware feature alignment | TPLD는 CNN 기반, DAFormer는 Transformer로 feature representation 자체를 강화. Pseudo label quality 관점에서 상호 보완 가능 |
| **HRDA** (Hoyer et al., ECCV 2022) | High-resolution/Low-resolution 크롭 기반 multi-scale self-training | TPLD의 Voting이 local 공간 정보를 활용하는 반면, HRDA는 multi-scale 정보를 통합 |
| **ProDA** (Zhang et al., CVPR 2021) | Prototype 기반 distribution alignment + pseudo label 정제 | TPLD는 spatial voting, ProDA는 class prototype 정렬로 pseudo label 품질 개선 — 접근 방향 상이 |
| **MIC** (Hoyer et al., CVPR 2023) | Masked Image Consistency를 통한 context 독립적 feature 학습 | Self-training의 consistency regularization 강화로 TPLD의 bootstrapping과 유사한 목적 |
| **UniDA / Open-set DA** (관련 연구들) | 알려지지 않은 클래스를 포함한 open-set 환경 | TPLD는 closed-set UDA에 특화되어 있어 open-set 환경에서는 easy-hard 분류 기준 재설계 필요 |

**주요 트렌드와 TPLD의 위치:**

```
2020년: TPLD — Pseudo Label Densification 제안 (sparse PL 문제 최초 공식화)
   ↓
2021년: ProDA — Prototype 기반 PL 정제 (class-level 분포 정렬)
   ↓  
2022년: DAFormer, HRDA — Transformer + 고해상도 self-training
   ↓
2023년: MIC, UniDA — Consistency + Open-set 확장
```

TPLD는 Transformer 기반 방법들이 지배하는 최신 트렌드에서는 절대 성능이 뒤처지지만, **sparse pseudo label 문제 정의와 단계적 densification 패러다임**은 여전히 유효한 통찰이다.

---

### 4.3 향후 연구 시 고려 사항

**① Transformer 백본과의 통합**

```python
# 개념적 방향
TPLD_Voting + DAFormer(Transformer backbone)
→ Dense PL + 강력한 feature representation
```

Vision Transformer(ViT) 기반 backbone에서의 self-attention map이 TPLD의 voting에서 활용하는 공간적 상관관계를 더 효과적으로 포착할 수 있다.

**② 자동화된 하이퍼파라미터 적응**

$q$, $\gamma$, $\alpha$, voting 횟수/크기를 **도메인 갭의 크기에 따라 자동 조정**하는 meta-learning 또는 AutoML 기반 접근 고려.

**③ Multi-source Domain Adaptation으로의 확장**

다수의 소스 도메인에서 타겟 도메인으로 적응 시, easy-hard 분류 기준이 소스 도메인별로 달라질 수 있으며, 이를 위한 **소스-인식 confidence score** 설계가 필요하다.

**④ Open-set 및 Source-free UDA로의 확장**

- **Source-free UDA**: 소스 데이터 없이 타겟 도메인 데이터만으로 적응 — bootstrapping 메커니즘의 역할이 더욱 중요해짐
- **Open-set UDA**: Unknown 클래스에 대한 easy-hard 분류 기준 재정의 필요

**⑤ 의료 영상, 원격 탐지 등 도메인 확장**

Urban driving scene에 특화된 공간적 상관관계 가정이 다른 도메인에서 성립하는지 검증 필요. 특히 의료 영상에서는 병변 영역이 작고 비균질적이어서 Voting의 window 크기 설계가 중요하다.

**⑥ Diffusion Model 기반 Data Augmentation과의 결합**

최근의 Diffusion Model을 활용한 도메인 간 이미지 변환과 TPLD의 pseudo label densification을 결합하면, 학습 데이터의 다양성과 pseudo label의 밀도를 동시에 향상시킬 수 있다.

**⑦ 계산 효율성 개선**

9 라운드의 반복 학습은 실용적으로 비용이 크다. **온라인 방식(online voting)**이나 **지식 증류(knowledge distillation)**를 통해 단일 패스로 유사한 효과를 달성하는 방향 탐색이 필요하다.

---

## 참고자료

- **주 논문 (원문)**: Inkyu Shin, Sanghyun Woo, Fei Pan, In So Kweon. "Two-phase Pseudo Label Densification for Self-training based Domain Adaptation." arXiv:2012.04828v1, December 2020.
- **CBST**: Zou, Y., Yu, Z., Kumar, B.V., Wang, J. "Unsupervised domain adaptation for semantic segmentation via class-balanced self-training." ECCV 2018.
- **CRST**: Zou, Y., Yu, Z., Liu, X., Kumar, B.V., Wang, J. "Confidence regularized self-training." ICCV 2019.
- **ADVENT**: Vu, T.H., Jain, H., Bucher, M., Cord, M., Pérez, P. "Advent: Adversarial entropy minimization for domain adaptation in semantic segmentation." CVPR 2019.
- **Adapt-SegMap**: Tsai, Y.H. et al. "Learning to adapt structured output space for semantic segmentation." CVPR 2018.
- **CLAN**: Luo, Y. et al. "Taking a closer look at domain shift: Category-level adversaries for semantics consistent domain adaptation." CVPR 2019.
- **Bootstrapping**: Reed, S. et al. "Training deep neural networks on noisy labels with bootstrapping." arXiv, 2014.
- **DAFormer**: Hoyer, L. et al. "DAFormer: Improving Network Architectures and Training Strategies for Domain-Adaptive Semantic Segmentation." CVPR 2022. *(후속 연구 비교용, 논문 원문 미인용)*
- **ProDA**: Zhang, P. et al. "Prototypical Pseudo Label Denoising and Target Structure Learning for Domain Adaptive Semantic Segmentation." CVPR 2021. *(후속 연구 비교용, 논문 원문 미인용)*

> **⚠️ 정확도 주의**: DAFormer, HRDA, ProDA, MIC 등 2020년 이후 최신 연구와의 비교 분석 부분(4.2절)의 수치 및 세부 내용은 논문 원문에 포함되지 않은 내용으로, 공개된 일반 지식에 기반한 분석입니다. 해당 논문들의 원문을 직접 확인하시기를 권장합니다.
