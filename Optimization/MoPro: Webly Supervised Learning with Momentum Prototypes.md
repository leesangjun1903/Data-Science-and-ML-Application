# MoPro: Webly Supervised Learning with Momentum Prototypes 

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

MoPro는 **웹에서 자동 수집된 노이즈가 많은 약한 레이블(weakly-labeled) 데이터**로부터 효율적으로 시각적 표현(representation)을 학습하는 방법론입니다. 핵심 주장은 다음과 같습니다:

> "지도학습의 어노테이션 비확장성(annotation unscalability)과 자기지도학습의 계산 비확장성(computation unscalability)을 동시에 극복할 수 있다."

### 주요 기여 (3가지)

| 기여 | 내용 |
|------|------|
| **온라인 레이블 노이즈 보정** | 훈련 중 실시간으로 잘못된 레이블을 교정 |
| **OOD 샘플 제거** | 분포 밖(out-of-distribution) 샘플을 자동 탐지 및 제거 |
| **표현 학습** | 대조 학습 기반의 강건하고 잘 보정된 임베딩 공간 학습 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

#### 배경 문제

1. **지도학습의 한계**: 수백만 장의 이미지에 수동 어노테이션은 비용이 너무 큼
2. **자기지도학습의 한계**: 대규모 계산 자원 필요, 다운스트림 태스크에서 지도학습 대비 성능 열위
3. **기존 웹 지도학습의 한계**: 노이즈를 무시하고 단순 Cross-Entropy 적용

#### 웹 데이터의 두 가지 노이즈 유형

- **레이블 노이즈**: 이미지에 잘못된 클래스 레이블이 할당됨 (예: 레몬 이미지에 오렌지 레이블)
- **OOD 샘플**: 어떤 클래스에도 속하지 않는 완전히 무관한 이미지

논문은 WebVision 데이터셋에서 약 **34%의 이미지가 아웃라이어**임을 언급합니다.

---

### 2.2 제안하는 방법 (수식 포함)

#### 전체 프레임워크 구성요소

```
입력 이미지 x_i
    ↓ (약한 augmentation)
CNN Encoder f(·) → 2048-D 표현 v_i
    ↓
Classifier h(·) → 클래스 예측 p_i
    ↓
Projection Network g(·) → 128-D 임베딩 z_i (L2 정규화)
    ↓
모멘텀 프로토타입 C와 대조
```

#### (1) 대조 손실 (Contrastive Loss)

**프로토타입 대조 손실** $\mathcal{L}_{\text{pro}}$:

$$\mathcal{L}_{\text{pro}}^{i} = -\log \frac{\exp(\boldsymbol{z}_i \cdot \boldsymbol{c}_{\hat{y}_i} / \tau)}{\sum_{k=1}^{K} \exp(\boldsymbol{z}_i \cdot \boldsymbol{c}_k / \tau)}$$

- $\boldsymbol{z}_i$: 샘플 $i$의 L2 정규화된 128차원 임베딩
- $\boldsymbol{c}_{\hat{y}_i}$: 의사 레이블 $\hat{y}_i$에 해당하는 모멘텀 프로토타입
- $\tau$: 온도 파라미터 (실험에서 $\tau = 0.1$)
- $K$: 전체 클래스 수

**인스턴스 대조 손실** $\mathcal{L}_{\text{ins}}$:

$$\mathcal{L}_{\text{ins}}^{i} = -\log \frac{\exp(\boldsymbol{z}_i \cdot \boldsymbol{z}_i' / \tau)}{\sum_{r=0}^{R} \exp(\boldsymbol{z}_i \cdot \boldsymbol{z}_r' / \tau)}$$

- $\boldsymbol{z}_i'$: 모멘텀 인코더로 생성된 같은 이미지의 모멘텀 임베딩 (강한 augmentation 적용)
- $R$: 큐(queue)에 저장된 네거티브 모멘텀 임베딩 수 (크기: 8192)

#### (2) 분류 손실 (Cross-Entropy Loss)

$$\mathcal{L}_{\text{ce}}^{i} = -\log(p_i^{\hat{y}_i})$$

- $p_i^{\hat{y}_i}$: 분류기가 의사 레이블 $\hat{y}_i$에 대해 출력한 확률

#### (3) 전체 학습 목적함수

$$\mathcal{L} = \sum_{i=1}^{n} \left(\mathcal{L}_{\text{ce}}^{i} + \lambda_{\text{pro}} \mathcal{L}_{\text{pro}}^{i} + \lambda_{\text{ins}} \mathcal{L}_{\text{ins}}^{i}\right)$$

모든 실험에서 $\lambda_{\text{pro}} = \lambda_{\text{ins}} = 1$로 단순화.

#### (4) 노이즈 보정 (Noise Correction)

**소프트 의사 레이블 생성**:

$$\boldsymbol{q}_i = \alpha \boldsymbol{p}_i + (1-\alpha)\boldsymbol{s}_i$$

$$s_i^k = \frac{\exp(\boldsymbol{z}_i \cdot \boldsymbol{c}_k / \tau)}{\sum_{k=1}^{K} \exp(\boldsymbol{z}_i \cdot \boldsymbol{c}_k / \tau)}$$

- $\alpha = 0.5$: 분류기 예측과 프로토타입 유사도의 결합 가중치
- $\boldsymbol{p}_i$: 분류기의 소프트맥스 출력 확률
- $\boldsymbol{s}_i$: 프로토타입과의 코사인 유사도 기반 확률 분포

**하드 의사 레이블 결정 규칙**:

$$\hat{y}_i = \begin{cases} \arg\max_k q_i^k & \text{if } \max_k q_i^k > T \\ y_i & \text{elseif } q_i^{y_i} > 1/K \\ \text{OOD} & \text{otherwise} \end{cases}$$

- $T = 0.8$ (WebVision-V1.0), $T = 0.6$ (WebVision-V2.0)
- $1/K$: 균일 확률 (랜덤 추측 수준)

#### (5) 모멘텀 프로토타입 업데이트

$$\boldsymbol{c}_k \leftarrow \text{Normalize}(m\boldsymbol{c}_k + (1-m)\boldsymbol{z}_i), \quad \forall i \in \{i \mid \hat{y}_i = k\}$$

$$\text{Normalize}(\boldsymbol{c}) = \frac{\boldsymbol{c}}{\|\boldsymbol{c}\|_2}$$

- $m = 0.999$: 모멘텀 계수 (안정적이고 느린 업데이트)
- OOD 샘플은 프로토타입 업데이트에서 **제외**

---

### 2.3 모델 구조

```
┌─────────────────────────────────────────────────────┐
│                    MoPro 구조                        │
├─────────────────────────────────────────────────────┤
│  입력: 약한 augmentation 이미지 x̃_i                │
│    ↓                                                │
│  ResNet-50 Encoder f(·)  → v_i ∈ ℝ^{2048}         │
│    ├── Classifier h(·)   → p_i ∈ ℝ^K (softmax)    │
│    └── Projection MLP g(·) → z_i ∈ ℝ^{128} (L2)  │
│                                                     │
│  모멘텀 브랜치 (파라미터 고정, EMA 업데이트):       │
│  강한 augmentation x̃'_i                           │
│    ↓                                               │
│  Momentum Encoder f'(g'(·)) → z'_i ∈ ℝ^{128}     │
│    └── Queue of past embeddings (size=8192)        │
│                                                     │
│  Momentum Prototypes C ∈ ℝ^{128×K}                │
│    └── Moving average of clean sample embeddings   │
└─────────────────────────────────────────────────────┘
```

**구현 세부사항**:
- 인코더: ResNet-50
- 투영 네트워크: 1개 은닉층을 가진 MLP (SimCLR 방식)
- 모멘텀 인코더: 주 인코더의 EMA (MoCo 방식)
- 배치 크기: 256, 에폭: 90, SGD 옵티마이저
- 워밍업: 처음 10 에폭은 원래 레이블로 전체 샘플 훈련

---

### 2.4 성능 향상

#### 업스트림 태스크 (WebVision-V1.0)

| 방법 | 아키텍처 | WebVision Top-1 | ImageNet Top-1 |
|------|---------|----------------|----------------|
| Cross-Entropy | ResNet-50 | 66.4 | 57.7 |
| SOM (Tu et al., 2020) | ResNet-50 | 72.2 | 65.0 |
| **MoPro (ours)** | **ResNet-50** | **73.9** | **67.8** |

#### 저샷 분류 (VOC07, k=1 기준)

| 방법 | 데이터셋 | mAP (k=1) |
|------|---------|-----------|
| MoCo v2 | ImageNet | 46.3 |
| CE (Sup.) | ImageNet | 54.3 |
| **MoPro** | **WebVision-V1.0** | **59.5** (+10.5 vs ImageNet Sup.) |
| **MoPro** | **WebVision-V2.0** | **64.8** |

#### 저자원 파인튜닝 (ImageNet 1%)

| 방법 | Top-1 (1%) |
|------|-----------|
| SwAV (800 epochs) | 53.9 |
| BYOL (1000 epochs) | 53.2 |
| **MoPro** (WebVision-V1.0, **90 epochs**) | **71.2** (+17.3 vs 최고 자기지도학습) |

#### 객체 탐지 (COCO, 1× schedule)

| 방법 | AP^bb | AP^mk |
|------|-------|-------|
| CE (ImageNet Sup.) | 38.9 | 35.4 |
| MoCo (Instagram-1B) | 38.9 | 35.4 |
| **MoPro** (WebVision-V1.0) | **39.7** | **36.1** |
| **MoPro** (WebVision-V2.0) | **40.7** | **36.8** |

---

### 2.5 한계점

논문에서 직접 언급하거나 실험 결과에서 유추할 수 있는 한계:

1. **대규모 데이터에서 OOD 임계값 조정 필요**: WebVision-V2.0에서는 $T=0.6$으로 조정해야 함 (클래스 수 증가 시 최적 임계값이 달라짐)

2. **클래스 불일치 문제**: WebVision-V2.0 (5k 클래스)으로 사전훈련 시 ImageNet (1k 클래스) 파인튜닝 성능이 V1.0보다 **오히려 낮음** — 태스크 특화 데이터의 중요성 시사

3. **초기 워밍업에 의존**: 노이즈 보정 전 10에폭 워밍업이 필요하며, 워밍업 품질이 이후 프로토타입 품질에 영향

4. **하이퍼파라미터 민감도**: $\tau$, $T$, $\alpha$, $m$ 등 여러 하이퍼파라미터가 있으며 데이터셋 특성에 따른 조정 필요

5. **단일 모달리티**: 현재는 이미지에만 적용 (텍스트, 비디오 등 다른 모달리티 미검토 — 논문은 이를 향후 과제로 명시)

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 분포 이동(Distribution Shift)에 대한 강건성

MoPro가 일반화 성능을 향상시키는 핵심 메커니즘을 실험적으로 검증:

| 방법 | 데이터셋 | ImageNet-R 정확도↑ | ImageNet-R 보정오차↓ | ImageNet-A 정확도↑ | ImageNet-A 보정오차↓ |
|------|---------|-------------------|--------------------|--------------------|-------------------|
| CE (Sup.) | ImageNet | 36.14 | 19.66 | 0.03 | 62.50 |
| CE | WebVision-V1.0 | 49.56 | 10.05 | 10.24 | 37.84 |
| **MoPro** | **WebVision-V1.0** | **54.87** | **5.73** | **11.93** | **35.85** |

**보정 오차(Calibration Error)**: 모델의 confidence와 실제 정확도 간의 불일치를 측정. MoPro는 보정 오차를 크게 줄여 **실세계 적용 시 더 신뢰할 수 있는 예측** 제공.

### 3.2 일반화 성능 향상의 원인 분석

#### (a) 웹 데이터의 다양성 활용
웹 이미지는 ImageNet 큐레이션 데이터와 달리 다양한 배경, 스타일, 조명, 뷰포인트를 포함. 이 **자연적 다양성**이 OOD 시나리오에 대한 강건성 기여.

$$\text{ImageNet-R 정확도}: 36.14\% \rightarrow 54.87\% \quad (+18.73\%p)$$

#### (b) 대조 학습 기반 임베딩 공간의 구조적 특성
프로토타입 대조 손실 $\mathcal{L}_{\text{pro}}$는 다음을 동시에 달성:

$$\text{클래스 내 집중도} \uparrow \quad \text{AND} \quad \text{클래스 간 분리도} \uparrow$$

이로 인해 임베딩 공간이 더 **구조적**이고, 새로운 분포에 대한 선형 분류기(Linear SVM) 적합이 용이해짐.

#### (c) OOD 샘플 처리 전략의 일반화 기여
OOD 샘플을 $\mathcal{L}\_{\text{ce}}$와 $\mathcal{L}\_{\text{pro}}$에서 **제외**하지만 $\mathcal{L}_{\text{ins}}$에는 **포함**:

$$\mathcal{L}_{\text{ins}}^{i(\text{OOD})} \neq 0, \quad \mathcal{L}_{\text{pro}}^{i(\text{OOD})} = \mathcal{L}_{\text{ce}}^{i(\text{OOD})} = 0$$

이 전략은 OOD 샘플을 인-디스트리뷰션 샘플로부터 **공간적으로 분리**시켜, 테스트 시 OOD 탐지 능력을 향상시킴.

#### (d) 저샷 학습에서의 일반화
- **VOC07 (k=1)**: ImageNet 지도학습 대비 **+10.5 mAP** 향상
- **Places205 (k=1)**: ImageNet 지도학습 대비 **+2.0%** 향상

이는 MoPro가 학습한 표현이 **도메인 불변적 특징**을 더 잘 포착함을 의미.

#### (e) 스케일과 노이즈 처리의 시너지 효과

데이터 크기 증가에 따른 성능 변화 (VOC07 k=1):

$$\text{CE}_{V0.5}: 49.8 \rightarrow \text{MoPro}_{V0.5}: 54.3 \rightarrow \text{MoPro}_{V1.0}: 59.5 \rightarrow \text{MoPro}_{V2.0}: 64.8$$

노이즈 보정과 데이터 스케일의 **상보적 효과**가 일반화 성능을 지속적으로 향상.

### 3.3 모델 보정(Model Calibration)과 일반화의 관계

$\ell_2$ 보정 오차(Calibration Error):

$$\text{CE}_{ImageNet}: 19.66\% \rightarrow \text{MoPro}_{WebVision}: 5.73\%$$

잘 보정된 모델은 불확실한 예측에서 낮은 confidence를 출력하므로, **과신(overconfidence)** 문제가 줄어 실세계 배포 시 안전성이 향상됨.

---

## 4. 향후 연구에 미치는 영향 및 고려할 점

### 4.1 연구에 미치는 영향

#### (a) 웹 지도학습 패러다임의 재정의
MoPro는 "노이즈는 데이터 규모로 덮을 수 있다"는 기존 통념에 반론을 제시하며, **노이즈 인식 학습(noise-aware learning)**이 웹 데이터 활용의 핵심임을 보임.

#### (b) 대조 학습과 노이즈 학습의 결합
자기지도학습의 대조 학습 기법(MoCo, SimCLR)을 약한 지도 학습에 통합한 선구적 시도로, 이후 연구들의 방향성을 제시.

#### (c) 프로토타입 기반 온라인 노이즈 보정
오프라인 노이즈 정제(costly 재학습)가 아닌 **온라인(one-pass) 방식**으로 노이즈를 처리하는 접근법을 제시, 실용성을 크게 향상.

#### (d) 전이학습 연구에 대한 기여
같은 데이터·계산 예산에서 **웹 지도학습 ≈ ImageNet 지도학습** 성능을 최초로 달성, 대규모 사전훈련 패러다임 다양화에 기여.

---

### 4.2 향후 연구 시 고려할 점

#### (a) 멀티모달 확장
논문은 이미지 데이터에만 집중하나, 웹에는 텍스트-이미지 쌍, 비디오 등이 풍부함. **CLIP** 등 비전-언어 모델과의 결합이 유망한 방향.

#### (b) 동적 임계값(Adaptive Threshold) 설계
현재 $T$는 고정 하이퍼파라미터이며, 클래스 불균형과 훈련 진행에 따른 **동적 임계값** 조정 메커니즘이 필요.

$$T_{\text{adaptive}}(t, k) = f(\text{training epoch}, \text{class difficulty})$$

#### (c) 장기 훈련(Catastrophic Forgetting) 문제
모멘텀 프로토타입이 점진적으로 업데이트될 때, 초기 노이즈가 장기간 프로토타입에 잔류할 가능성 → **망각 방지 메커니즘** 연구 필요.

#### (d) 더 정교한 OOD 탐지
현재 단순 임계값 기반 OOD 탐지는 **점수 기반 OOD 탐지**(e.g., energy-based, Mahalanobis distance) 방법 대비 한계 가능성 → 더 정교한 OOD 스코어링 함수 도입 고려.

#### (e) 클래스 계층 구조 활용
웹 데이터의 레이블은 종종 계층적 관계(예: "개" → "치와와")를 가짐. 이를 프로토타입 설계에 반영하는 **계층적 프로토타입** 연구.

#### (f) 공정성(Fairness) 및 데이터 편향
웹 크롤링 데이터는 사회적 편향을 내포할 수 있으며, 노이즈 보정 과정에서 특정 그룹의 샘플이 체계적으로 OOD로 분류될 위험 → **공정성 인식 노이즈 보정** 연구 필요.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 자기지도 학습 계열과의 비교

| 방법 | 발표 | 핵심 아이디어 | MoPro와의 관계 | ImageNet 1% Top-1 |
|------|-----|-------------|---------------|------------------|
| **SimCLR** (Chen et al., 2020) | ICML 2020 | 인스턴스 대조 학습 | MoPro의 $\mathcal{L}_{\text{ins}}$ 기반 | 48.3 (1000 epoch) |
| **MoCo v2** (Chen et al., 2020) | arXiv 2020 | 모멘텀 큐 + MLP 헤드 | MoPro의 모멘텀 인코더 기반 | ~55 (200 epoch 추정) |
| **BYOL** (Grill et al., 2020) | NeurIPS 2020 | 부정 샘플 없는 대조 학습 | 네거티브 없는 접근 | 53.2 (1000 epoch) |
| **SwAV** (Caron et al., 2020) | NeurIPS 2020 | 온라인 클러스터링 + 교차 뷰 예측 | 프로토타입 유사하나 지도 없음 | 53.9 (800 epoch) |
| **MoPro** (Li et al., 2021) | ICLR 2021 | 약한 지도 + 모멘텀 프로토타입 | - | **71.2 (90 epoch)** |

**핵심 차이**: MoPro는 약한 레이블 정보를 활용하므로 자기지도학습보다 훨씬 **적은 에폭**으로 훨씬 **높은 성능**을 달성.

### 5.2 노이즈 레이블 학습 계열과의 비교

| 방법 | 핵심 전략 | OOD 처리 | 계산 복잡도 | 대규모 적용성 |
|------|---------|---------|-----------|------------|
| **DivideMix** (Li et al., 2020) | GMM 기반 깨끗/노이즈 분리 + 반지도학습 | ✗ | 높음 (co-training) | 제한적 |
| **MentorNet** (Jiang et al., 2018) | 커리큘럼 학습 | ✗ | 높음 | 제한적 |
| **SOM** (Tu et al., 2020) | 메모리 네트워크 | 부분적 | 중간 | 중간 |
| **MoPro** | 모멘텀 프로토타입 + 대조 학습 | ✅ | **낮음 (온라인)** | **우수** |

### 5.3 MoPro 이후 발전한 관련 연구 방향

> ⚠️ **주의**: 이하 언급되는 2021년 이후 연구들은 제공된 논문 PDF에 포함되지 않은 내용으로, 제가 학습한 지식을 바탕으로 기술하며 **일부 세부 수치는 확인이 필요**합니다.

#### (a) CLIP (Radford et al., 2021, OpenAI)
- **핵심**: 4억 개 웹 크롤링 이미지-텍스트 쌍으로 대조 학습
- **MoPro와의 관계**: MoPro의 웹 데이터 활용 철학을 멀티모달로 확장
- **차이점**: CLIP은 텍스트 감독을 활용하는 반면 MoPro는 클래스 레이블만 사용

#### (b) ALIGN (Jia et al., 2021, Google)
- **핵심**: 18억 개 노이즈 이미지-텍스트 쌍으로 학습, 노이즈를 규모로 극복
- **MoPro와의 관계**: "노이즈 vs 규모" 논쟁에서 규모 우위를 보이나 MoPro의 노이즈 보정 접근과 대비됨

#### (c) DINO (Caron et al., 2021, Facebook)
- **핵심**: 자기 증류 기반 자기지도학습, ViT 아키텍처와 결합
- **MoPro와의 관계**: 프로토타입 없이 teacher-student 구조로 유사한 표현 학습

#### (d) 웹 지도학습 후속 연구 방향
MoPro 이후 연구들은 다음 방향으로 발전:
- **멀티모달 약한 레이블**: 텍스트 쿼리를 함께 활용한 노이즈 보정
- **ViT 백본 적용**: ResNet 대신 Vision Transformer와의 결합
- **더 정교한 프로토타입**: 계층적/혼합 가우시안 프로토타입

---

## 참고 자료

**주 논문**:
- Li, J., Xiong, C., & Hoi, S. C. H. (2021). **MoPro: Webly Supervised Learning with Momentum Prototypes**. *ICLR 2021*. (제공된 PDF)

**논문 내 인용 핵심 참고문헌** (PDF 내 References 섹션 기반):
- He, K., et al. (2019). Momentum Contrast for Unsupervised Visual Representation Learning (MoCo). *arXiv:1911.05722*
- Chen, T., et al. (2020a). A Simple Framework for Contrastive Learning of Visual Representations (SimCLR). *ICML 2020*
- Li, J., et al. (2020b). Prototypical Contrastive Learning of Unsupervised Representations (PCL). *arXiv:2005.04966*
- Grill, J.-B., et al. (2020). Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning (BYOL). *arXiv:2006.07733*
- Caron, M., et al. (2020). Unsupervised Learning of Visual Features by Contrasting Cluster Assignments (SwAV). *arXiv:2006.09882*
- Li, W., et al. (2017). WebVision Database: Visual Learning and Understanding from Web Data. *arXiv:1708.02862*
- Li, J., et al. (2020a). DivideMix: Learning with Noisy Labels as Semi-Supervised Learning. *ICLR 2020*
- Hendrycks, D., et al. (2020). The Many Faces of Robustness. *arXiv:2006.16241*
- Kang, B., et al. (2020). Decoupling Representation and Classifier for Long-Tailed Recognition. *ICLR 2020*
