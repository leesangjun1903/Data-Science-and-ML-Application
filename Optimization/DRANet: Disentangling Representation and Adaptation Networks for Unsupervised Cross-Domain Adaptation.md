# DRANet: Disentangling Representation and Adaptation Networks for Unsupervised Cross-Domain Adaptation

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

DRANet(Disentangling Representation and Adaptation Networks)은 기존의 도메인 적응(Domain Adaptation) 방법들이 **도메인 간 공유 특징(shared features)**에 집중하는 것과 달리, **각 도메인의 고유한 특성(individual domain characteristics)**을 보존하면서 콘텐츠(content)와 스타일(style)로 표현을 분리(disentangle)하여 비지도 방식으로 크로스 도메인 적응을 수행한다는 것을 핵심으로 합니다.

### 주요 기여

| 기여 항목 | 내용 |
|---|---|
| **단일 네트워크 다방향 적응** | 하나의 인코더-디코더 네트워크로 양방향/다방향 도메인 전환 가능 |
| **비선형 표현 분리** | 선형 분리의 한계를 극복한 비선형 매니폴드 기반 콘텐츠-스타일 분리 |
| **CADT 모듈** | 복잡한 장면에서 씬 구조를 유지하면서 스타일을 전환하는 Content-Adaptive Domain Transfer |
| **레이블 불필요** | 어떠한 클래스 레이블도 없이 비지도 학습 방식으로 도메인 적응 수행 |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

기존 비지도 도메인 적응 방법들의 한계는 다음과 같습니다:

- **공유 특징 공간 의존**: 기존 방법들(DANN, CyCADA 등)은 모든 도메인 이미지를 하나의 공유 특징 공간으로 매핑하여 각 도메인의 고유 특성을 손실시킴
- **다중 인코더-디코더 필요**: 기존 분리 학습 방법들은 도메인별로 별개의 인코더와 생성기를 필요로 함
- **레이블 의존성**: 태스크 분류기 학습을 위해 그라운드 트루스 클래스 레이블이 요구됨
- **선형 분리의 한계**: Zhang et al. [42]의 선형 분리 방법은 도메인 간 분포 차이를 효과적으로 처리하지 못함

---

### 2.2 제안하는 방법 (수식 포함)

#### (1) 특징 추출 (Feature Extraction)

가중치를 공유하는 인코더 $E$로 소스와 타겟 이미지의 특징을 추출합니다:

$$\mathcal{F}_X = E(I_X), \quad \mathcal{F}_Y = E(I_Y) \tag{1}$$

#### (2) 비선형 콘텐츠-스타일 분리 (Non-linear Disentanglement)

분리기 $S$를 통해 특징을 콘텐츠 $\mathcal{C}$와 스타일 $\mathcal{S}$로 분리합니다:

$$\mathcal{C}_X = w_X S(\mathcal{F}_X), \quad \mathcal{S}_X = \mathcal{F}_X - w_X S(\mathcal{F}_X) \tag{2}$$

여기서 $w_X$는 콘텐츠 공간의 분포를 정규화하는 학습 가능한 스케일 파라미터이며, 도메인 간 분포 이동을 보상합니다.

- **콘텐츠**: 비선형 함수 $S$와 학습 가능한 $w_X$의 곱으로 정의
- **스타일**: 전체 특징에서 콘텐츠 성분을 빼는 방식으로 정의

#### (3) 도메인 전환 특징 합성 (Domain Transfer Feature Synthesis)

$$\mathcal{F}_{X \to Y} = w_{X \to Y} \mathcal{C}_X + \mathcal{S}_Y, \quad \mathcal{F}_{Y \to X} = w_{Y \to X} \mathcal{C}_Y + \mathcal{S}_X$$

$$\text{where} \quad w_{X \to Y} = \frac{w_Y}{w_X}, \quad w_{Y \to X} = \frac{w_X}{w_Y} \tag{3}$$

소스 도메인의 콘텐츠에 타겟 도메인의 스타일을 합산하여 도메인 전환을 수행합니다. 스케일 파라미터 $w_{X \to Y}$는 도메인 간 분포 차이를 보상합니다.

#### (4) 이미지 생성 (Image Generation)

$$I_{X \to Y} = G(\mathcal{F}_{X \to Y}), \quad I_{Y \to X} = G(\mathcal{F}_{Y \to X})$$

$$I'_X = G(\mathcal{F}_X), \quad I'_Y = G(\mathcal{F}_Y) \tag{4}$$

---

### 2.3 Content-Adaptive Domain Transfer (CADT)

복잡한 장면(예: 자율주행 장면)에서 콘텐츠 유사도를 기반으로 더 적합한 스타일 특징을 선택합니다.

#### 콘텐츠 유사도 행렬 (Content Similarity Matrix)

$$\mathcal{H}_{row} = \sigma_{row}\left(\mathcal{C}_X \cdot \mathcal{C}_Y^\top\right) = \begin{bmatrix} \mathcal{C}_{11} & \cdots & \mathcal{C}_{1b} \\ \vdots & \ddots & \vdots \\ \mathcal{C}_{b1} & \cdots & \mathcal{C}_{bb} \end{bmatrix}, \quad \mathcal{C}_X, \mathcal{C}_Y \in \mathbb{R}^{B \times N} \tag{5}$$

여기서 $\sigma_{row}$는 행(row) 방향의 소프트맥스 연산, $B$는 배치 크기, $N$은 특징 차원입니다.

#### 콘텐츠 적응형 스타일 특징 생성

$$\hat{\mathcal{S}}_Y = \mathcal{H}_{row} \mathcal{S}_Y, \quad \text{where} \quad \mathcal{S}_Y \in \mathbb{R}^{B \times N} \tag{6}$$

#### 반대 방향의 유사도 행렬

$$\mathcal{H}_{col} = \left(\sigma_{col}\left(\mathcal{C}_X \cdot \mathcal{C}_Y^\top\right)\right)^\top \tag{7}$$

---

### 2.4 학습 손실 함수 (Training Loss)

#### 전체 목적 함수

$$\min_{E, S, G} \left( \sum_{d \in \{X, Y\}} \max_{D_d} \mathcal{L}^d \right) \tag{8}$$

$$\mathcal{L}^d = \alpha_1 \mathcal{L}^d_{Rec} + \alpha_2 \mathcal{L}^d_{GAN} + \alpha_3 \mathcal{L}^d_{Con} + \alpha_4 \mathcal{L}^d_{Per} \tag{9}$$

#### (a) 재구성 손실 (Reconstruction Loss)

$$\mathcal{L}^d_{Rec} = \mathcal{L}_1(I_d, I'_d), \quad \text{where} \quad I'_d = G(E(I_d)) \tag{10}$$

입력 이미지와 재구성된 이미지 간의 L1 손실로, 인코더와 생성기가 이미지를 충실히 재구성하도록 학습합니다.

#### (b) 적대적 손실 (Adversarial Loss)

$$\mathcal{L}^Y_{GAN} = \mathbb{E}_{y \sim p_{data}(Y)}[\log D_Y(y)] + \mathbb{E}_{(x,y) \sim p_{data}(X,Y)}[\log(1 - D_Y(I_{X \to Y}(x, y)))] \tag{11}$$

PatchGAN 판별기와 힌지(hinge) 버전의 적대적 손실을 사용하며, 소스와 타겟 양 방향에 모두 적용됩니다.

#### (c) 일관성 손실 (Consistency Loss)

$$\mathcal{L}^X_{Con} = \mathcal{L}_1(\mathcal{C}_X, \mathcal{C}_{X \to Y}) + \mathcal{L}_1(\mathcal{S}_X, \mathcal{S}_{Y \to X})$$

$$\mathcal{L}^Y_{Con} = \mathcal{L}_1(\mathcal{C}_Y, \mathcal{C}_{Y \to X}) + \mathcal{L}_1(\mathcal{S}_Y, \mathcal{S}_{X \to Y}) \tag{12}$$

도메인 전환 후에도 콘텐츠와 스타일 성분이 유지되는지를 검증하는 손실입니다.

#### (d) 지각적 손실 (Perceptual Loss)

$$\mathcal{L}^X_{Per} = \mathcal{L}^X_{Content} + \lambda \mathcal{L}^X_{Style}$$

$$\mathcal{L}^Y_{Content} = \sum_{l \in L_C} \|P_l(I_X) - P_l(I_{X \to Y})\|^2_2 \tag{13}$$

$$\mathcal{L}^Y_{Style} = \sum_{l \in L_S} \|\mathcal{G}(P_l(I_Y)) - \mathcal{G}(P_l(I_{X \to Y}))\|^2_F \tag{14}$$

여기서 $\mathcal{G}$는 Gram Matrix를 구성하는 함수이며, $P_l$은 사전 학습된 지각 네트워크의 $l$번째 레이어 특징입니다. 레이블 없이 콘텐츠와 스타일 분리를 학습할 수 있게 합니다.

---

### 2.5 모델 구조

```
[전체 파이프라인]

I_X, I_Y → [E: 공유 인코더] → F_X, F_Y
                                    ↓
                              [S: 분리기]
                                    ↓
                  C_X, S_X    /          \    C_Y, S_Y
                              ↓          ↓
              F_{X→Y} = w_{X→Y}·C_X + S_Y
              F_{Y→X} = w_{Y→X}·C_Y + S_X
                              ↓
                        [G: 생성기]
                              ↓
          I_{X→Y}, I_{Y→X}, I'_X, I'_Y

[판별기]
D_X: X 도메인 판별
D_Y: Y 도메인 판별

[지각 네트워크]
P: VGG 기반 사전학습 네트워크로 지각 손실 계산
```

---

### 2.6 성능 향상

#### 숫자 분류 태스크 (Digit Classification)

| 방법 | MNIST→USPS | USPS→MNIST | MNIST→MNIST-M | MNIST-M→MNIST |
|---|---|---|---|---|
| Source Only | 80.2 | 44.9 | 62.5 | 97.8 |
| DANN [9] | 85.1 | 73.0 | 77.4 | - |
| DSN [3] | 91.3 | - | 83.2 | - |
| ADDA [37] | 90.1 | 95.2 | - | - |
| CyCADA [15] | 95.6 | 96.5 | - | - |
| LC + CycleGAN [39,44] | 97.1 | **98.3** | - | - |
| **Ours (Bi-dir)** | **98.2** | 97.8 | **98.7** | **99.3** |

#### 의미론적 분할 태스크 (Semantic Segmentation, GTA5→CityScapes)

| 방법 | mIoU | fwIoU | Pixel Acc. |
|---|---|---|---|
| Source Only | 21.7 | 47.4 | 62.5 |
| CyCADA [15] | 39.5 | 72.4 | 82.3 |
| LC [39] | 40.5 | 75.1 | 84.0 |
| **Ours (w/o CADT)** | 40.6 | 75.6 | 84.9 |
| **Ours (with CADT)** | **41.4** | **76.4** | **85.7** |

#### 애블레이션 스터디 결과

| 비선형성 | 정규화 | MNIST→USPS | USPS→MNIST |
|---|---|---|---|
| ✗ | ✗ | 11.2 | 87.1 |
| ✗ | ✓ | 90.7 | 90.2 |
| ✓ | ✗ | 96.6 | 90.9 |
| ✓ | ✓ | **98.2** | **97.8** |

---

### 2.7 한계점

1. **배치 크기 의존성**: CADT는 미니배치 내 콘텐츠 유사도 행렬을 활용하므로, 배치 크기가 작을 경우 유사도 탐색 범위가 제한됩니다.

2. **콘텐츠-스타일 분리의 불완전성**: 학습 초기 단계에서 분리기가 완전하지 않아 콘텐츠 혼합(content-mixed) 이미지가 생성되는 문제가 있으며, 이를 CADT로 완화하지만 근본적으로 해결하지는 못합니다.

3. **도메인 수 증가 시 확장성**: 논문에서 3개 도메인까지의 실험을 보여주지만, 도메인 수가 크게 증가할 경우 $w_{X \to Y}$ 파라미터의 폭발적 증가 및 학습 복잡도 문제가 있습니다.

4. **복잡한 장면 구조에서의 한계**: GTA5→CityScapes의 mIoU 41.4는 Target Only(67.4)와 여전히 큰 격차가 존재합니다.

5. **단일 공유 인코더의 제약**: 도메인 특화(domain-specific) 인코딩이 부재하여, 도메인 간 분포 차이가 매우 클 경우 한계가 있을 수 있습니다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화 성능을 높이는 설계 요소

#### (a) 도메인 특성 보존을 통한 일반화

DRANet은 기존 방법과 달리 각 도메인의 고유 특성을 보존합니다. 이는 **도메인 불변 특징(domain-invariant features)**만을 추출하는 것이 아니라, **도메인 고유 특징을 분리하여 선택적으로 활용**함으로써 다양한 도메인 조합에 유연하게 대응할 수 있습니다.

$$\mathcal{C}_X = w_X S(\mathcal{F}_X) \quad \leftarrow \text{domain-specific content}$$

$$\mathcal{S}_X = \mathcal{F}_X - w_X S(\mathcal{F}_X) \quad \leftarrow \text{domain-specific style}$$

#### (b) 데이터 증강 효과를 통한 일반화

논문에서 흥미로운 발견은 DRANet으로 생성한 스타일 전환 이미지로 분류기를 학습하면, **일부 경우 타겟 도메인 데이터만 사용한 모델보다 높은 성능**을 보인다는 것입니다 (MNIST→MNIST-M: 98.7 vs Target Only: 96.2). 이는 DRANet이 하나의 소스 이미지로 타겟 이미지 수만큼 다양한 학습 샘플을 생성하는 효과적인 데이터 증강 기제로 작동함을 시사합니다.

#### (c) 다방향 적응을 통한 일반화

단일 모델로 세 도메인(MNIST, USPS, MNIST-M) 간 적응이 가능하며, **명시적으로 학습하지 않은 도메인 쌍(MNIST-M ↔ USPS) 간에도 적응**이 이루어지는 일반화 능력을 보여줍니다. 이는 콘텐츠-스타일 분리가 도메인 독립적인 표현을 학습함을 의미합니다.

#### (d) CADT를 통한 안정적 학습과 일반화

콘텐츠 유사도 기반의 스타일 선택은 학습 초기의 불안정성을 완화하고, 구조적으로 유사한 씬에서 스타일 전환이 이루어지도록 유도합니다. 이는 특히 의미론적 분할과 같은 복잡한 태스크에서 일반화 성능을 향상시킵니다:

$$\hat{\mathcal{S}}_Y = \mathcal{H}_{row} \mathcal{S}_Y \quad \leftarrow \text{content-aware style selection}$$

#### (e) 비선형 매니폴드 학습의 역할

비선형 분리기 $S$는 선형 방법으로는 분리하기 어려운 복잡한 도메인 분포 차이를 효과적으로 처리합니다. 애블레이션 연구에서 비선형성 추가 시 MNIST→USPS 정확도가 90.7%에서 98.2%로 대폭 향상된 것이 이를 입증합니다.

### 3.2 일반화 성능의 한계 및 개선 방향

| 한계 | 원인 | 개선 방향 |
|---|---|---|
| Target Only 대비 격차 | 레이블 없는 타겟 적응의 근본적 한계 | Self-training, Pseudo-labeling 결합 |
| 도메인 수 증가 시 성능 저하 | 단일 공유 인코더의 표현력 한계 | 도메인별 어댑터(adapter) 추가 |
| 복잡한 씬에서 구조 왜곡 | 스타일 전환 시 콘텐츠 보존의 어려움 | 더 강력한 콘텐츠 보존 손실 설계 |

---

## 4. 향후 연구에 미치는 영향과 고려 사항

### 4.1 연구에 미치는 영향

#### (a) 단일 네트워크 다방향 도메인 적응의 패러다임 제시

DRANet은 **하나의 네트워크로 여러 도메인 쌍을 동시에 처리**하는 가능성을 보여줌으로써, 실용적인 도메인 적응 시스템 설계에 새로운 방향을 제시합니다. 향후 대규모 멀티 도메인 적응(예: 10개 이상의 도메인) 연구로 확장될 수 있습니다.

#### (b) 완전 비지도 방식의 가능성

레이블 없이도 경쟁력 있는 성능을 달성함으로써, **실세계의 레이블 비용 문제**를 해결하는 연구 방향에 중요한 레퍼런스가 됩니다. 특히 의료 영상, 위성 영상 등 레이블링 비용이 높은 분야에 적용 가능성이 큽니다.

#### (c) 콘텐츠-스타일 분리 기반 데이터 증강

DRANet 기반의 데이터 증강이 분류기 성능을 향상시킨다는 발견은, **합성 데이터 증강(synthetic data augmentation)**과 도메인 적응을 결합하는 연구의 토대가 됩니다.

#### (d) Foundation Model 시대의 시사점

현재의 Vision-Language Model(CLIP, DALL-E 등)과 결합될 경우, 더 풍부한 의미론적 콘텐츠 표현을 활용한 콘텐츠-스타일 분리가 가능할 수 있으며, DRANet의 프레임워크가 이러한 연구의 기초가 될 수 있습니다.

---

### 4.2 앞으로 연구 시 고려할 점

#### (a) Transformer 기반 아키텍처로의 확장

DRANet은 CNN 기반 인코더-디코더를 사용합니다. 2020년 이후 Vision Transformer(ViT)의 등장으로, **Transformer 기반의 콘텐츠-스타일 분리**가 더 효과적일 수 있음을 고려해야 합니다. Self-attention 메커니즘은 전역적(global) 콘텐츠 구조를 포착하는 데 유리합니다.

#### (b) 도메인 일반화(Domain Generalization)로의 확장

DRANet은 테스트 시 타겟 도메인 이미지가 필요한 도메인 적응(Domain Adaptation) 방식입니다. 타겟 도메인 데이터 없이도 일반화 가능한 **도메인 일반화(Domain Generalization)** 설정으로의 확장이 중요한 연구 방향입니다.

#### (c) Source-Free Domain Adaptation과의 통합

최근 소스 데이터 없이 사전 학습된 모델만으로 적응하는 **Source-Free Domain Adaptation** 연구와의 결합 가능성을 고려해야 합니다.

#### (d) 콘텐츠-스타일 분리의 이론적 토대 강화

현재 콘텐츠와 스타일의 분리가 수식 $\mathcal{S}_X = \mathcal{F}_X - w_X S(\mathcal{F}_X)$와 같이 경험적으로 정의됩니다. **정보 이론(Information Theory)** 관점에서 두 성분 간의 독립성을 보장하는 이론적 정당성이 필요합니다 (예: Mutual Information Minimization).

#### (e) 배치 크기 독립적인 CADT 설계

현재 CADT는 미니배치 내 유사도를 계산하므로 배치 크기에 의존적입니다. **메모리 뱅크(Memory Bank)** 방식으로 더 넓은 범위의 콘텐츠 유사도를 활용하는 개선이 필요합니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

아래는 제가 알고 있는 범위 내에서의 비교 분석입니다. 논문 원문에 직접 언급되지 않은 내용은 제 학습 데이터 기반이므로, 세부 수치의 정확성을 위해 원 논문 확인을 권장합니다.

### 5.1 주요 관련 연구 비교

| 연구 | 방법론 | DRANet 대비 차이점 |
|---|---|---|
| **DAFormer** (Hoyer et al., CVPR 2022) | Transformer 기반 도메인 적응, Rare Class Sampling | Transformer 아키텍처 채택, 픽셀 수준 적응에 강점 |
| **HRDA** (Hoyer et al., ECCV 2022) | 고해상도 크롭 기반 의미 분할 적응 | 고해상도 처리로 세밀한 경계 표현 향상 |
| **SPA** (Wang et al., 2022) | Style-Perceptual Augmentation | 스타일 증강에 집중, 분리 학습 없음 |
| **CLIP-DA** 계열 (2022~) | CLIP 등 대규모 사전학습 모델 활용 | 풍부한 의미론적 사전지식 활용 |
| **CDTrans** (Xu et al., ICLR 2022) | Cross-attention Transformer로 도메인 전환 | Attention 기반 도메인 정렬 |

### 5.2 DRANet의 강점과 한계 (최신 연구 대비)

**강점:**
- 단일 네트워크로 다방향 도메인 전환이 가능한 **효율성**
- 레이블 없이 콘텐츠-스타일 분리를 학습하는 **완전 비지도 방식**
- CADT를 통한 **구조 보존 스타일 전환**

**한계 (최신 연구 대비):**
- DAFormer, HRDA 등 Transformer 기반 방법들이 의미론적 분할에서 더 높은 mIoU를 달성
- 대규모 사전학습 모델(CLIP 등)을 활용하지 못하는 구조적 한계
- Self-training 기반 방법들(예: pseudo-label 활용)과 결합하면 성능이 더 향상될 것으로 예상

---

## 참고 자료

### 원문 논문
- **Seunghun Lee, Sunghyun Cho, Sunghoon Im.** "DRANet: Disentangling Representation and Adaptation Networks for Unsupervised Cross-Domain Adaptation." arXiv:2103.13447v2, 2021.

### 논문 내 인용 주요 참고문헌
- [9] Ganin et al., "Domain-adversarial training of neural networks." JMLR, 2016. (DANN)
- [15] Hoffman et al., "CyCADA: Cycle-consistent adversarial domain adaptation." ICML, 2018.
- [2] Bousmalis et al., "Unsupervised pixel-level domain adaptation with GANs." CVPR, 2017. (pixelDA)
- [3] Bousmalis et al., "Domain separation networks." NIPS, 2016. (DSN)
- [37] Tzeng et al., "Adversarial discriminative domain adaptation." CVPR, 2017. (ADDA)
- [39] Ye et al., "Light-weight calibrator." CVPR, 2020. (LC)
- [42] Zhang et al., "Style separation and synthesis via GANs." ACM MM, 2018.
- [18] Johnson et al., "Perceptual losses for real-time style transfer." ECCV, 2016.

### 비교 분석 관련 후속 연구 (학습 데이터 기반)
- Hoyer et al., "DAFormer: Improving Network Architectures and Training Strategies for Domain-Adaptive Semantic Segmentation." CVPR 2022.
- Hoyer et al., "HRDA: Context-Aware High-Resolution Domain-Adaptive Semantic Segmentation." ECCV 2022.
- Xu et al., "CDTrans: Cross-domain Transformer for Unsupervised Domain Adaptation." ICLR 2022.

> **⚠️ 주의사항**: 2020년 이후 최신 연구 비교 분석 부분(Section 5)의 일부 수치 및 세부 내용은 제 학습 데이터를 기반으로 하며, 100% 정확성을 보장하지 않습니다. 정확한 수치 비교를 위해서는 각 원 논문을 직접 확인하시기 바랍니다.
