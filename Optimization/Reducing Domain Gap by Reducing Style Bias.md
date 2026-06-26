# Reducing Domain Gap by Reducing Style Bias

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장
CNN의 **스타일 편향(style bias / texture bias)** 이 도메인 이동(domain shift) 문제의 근본 원인 중 하나이며, 이 편향을 줄이면 도메인 간 격차(domain gap)가 자연적으로 감소한다는 것이 핵심 주장입니다.

> 인간은 객체의 **형상(shape/content)** 기반으로 인식하지만, CNN은 **텍스처(texture/style)** 기반으로 인식하는 경향이 강하며, 스타일은 도메인 간 변화에 민감하므로 도메인 이동 취약성의 원인이 됩니다.

### 주요 기여
| 기여 항목 | 내용 |
|---|---|
| **새로운 관점** | 도메인 격차 문제를 CNN의 내재적 귀납 편향(inductive bias) 관점에서 분석 |
| **SagNet 프레임워크** | Style-Agnostic Networks: 스타일 무관 표현 학습 프레임워크 제안 |
| **도메인 레이블 불필요** | 도메인 정보 없이 단일 소스에서도 적용 가능한 범용성 |
| **직교성(Orthogonality)** | 기존 DA/DG 방법들과 결합하여 추가 성능 향상 가능 |
| **광범위한 검증** | DG, UDA, SSDA 세 가지 시나리오에서 모두 성능 개선 검증 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**도메인 이동(Domain Shift)** 문제를 세 가지 설정에서 동시에 해결합니다:

- **Domain Generalization (DG)**: 미지의 타겟 도메인에 대한 일반화
- **Unsupervised Domain Adaptation (UDA)**: 레이블 없는 타겟 데이터 활용
- **Semi-supervised Domain Adaptation (SSDA)**: 소수의 타겟 레이블 활용

기존 방법들이 도메인 분포를 직접 정렬하는 방식(MMD, adversarial alignment 등)에 의존한 반면, 이 논문은 **CNN의 내재적 스타일 편향 자체를 제거**하는 근본적 접근을 취합니다.

---

### 2.2 제안하는 방법 (수식 포함)

#### 스타일 표현: 채널별 평균 및 표준편차

$$\mu(\mathbf{z}) = \frac{1}{HW} \sum_{h=1}^{H} \sum_{w=1}^{W} \mathbf{z}_{hw} $$

$$\sigma(\mathbf{z}) = \sqrt{\frac{1}{HW} \sum_{h=1}^{H} \sum_{w=1}^{W} (\mathbf{z}_{hw} - \mu(\mathbf{z}))^2 + \epsilon} $$

여기서 $\mathbf{z} \in \mathbb{R}^{D \times H \times W}$는 특징 추출기 $G_f$의 중간 특징 맵입니다.

---

#### 🔷 Module 1: Style Randomization (SR) — Content-Biased Learning

랜덤하게 선택된 이미지 $\mathbf{x}'$의 특징 맵 $\mathbf{z}'$와 입력 $\mathbf{z}$의 스타일을 보간하여 새로운 스타일을 생성합니다:

$$\hat{\mu} = \alpha \cdot \mu(\mathbf{z}) + (1 - \alpha) \cdot \mu(\mathbf{z}') $$

$$\hat{\sigma} = \alpha \cdot \sigma(\mathbf{z}) + (1 - \alpha) \cdot \sigma(\mathbf{z}') $$

$$\text{SR}(\mathbf{z}, \mathbf{z}') = \hat{\sigma} \cdot \left(\frac{\mathbf{z} - \mu(\mathbf{z})}{\sigma(\mathbf{z})}\right) + \hat{\mu} $$

여기서 $\alpha \sim \text{Uniform}(0, 1)$은 랜덤 보간 가중치입니다.

**Content-Biased Loss:**

$$\min_{G_f, G_c} \mathcal{L}_c = -\mathbb{E}_{(\mathbf{x}, \mathbf{y}) \in S} \sum_{k=1}^{K} y_k \log G_c(\text{SR}(G_f(\mathbf{x}), \mathbf{z}'))_k $$

> 스타일을 랜덤화함으로써 네트워크가 콘텐츠(형상)에만 집중하도록 강제합니다.

---

#### 🔶 Module 2: Content Randomization (CR) — Adversarial Style-Biased Learning

반대로, 콘텐츠를 랜덤화하여 스타일 편향 네트워크를 구성합니다:

$$\text{CR}(\mathbf{z}, \mathbf{z}') = \sigma(\mathbf{z}) \cdot \left(\frac{\mathbf{z}' - \mu(\mathbf{z}')}{\sigma(\mathbf{z}')}\right) + \mu(\mathbf{z}) $$

**Style-Biased Loss** ($G_s$ 학습용):

$$\min_{G_s} \mathcal{L}_s = -\mathbb{E}_{(\mathbf{x}, \mathbf{y}) \in S} \sum_{k=1}^{K} y_k \log G_s(\text{CR}(G_f(\mathbf{x}), \mathbf{z}'))_k $$

**Adversarial Loss** ($G_f$ 학습용, 스타일 정보를 클래스 판별에 무력화):

$$\min_{G_f} \mathcal{L}_{adv} = -\lambda_{adv} \mathbb{E}_{(\mathbf{x}, \cdot) \in S} \sum_{k=1}^{K} \frac{1}{K} \log G_s(\text{CR}(G_f(\mathbf{x}), \mathbf{z}'))_k $$

> $\mathcal{L}_{adv}$는 스타일 편향 예측을 균등 분포에 맞추도록 최적화 — 즉, 스타일이 클래스를 판별하지 못하게 합니다.

---

#### 🟢 Module 3: Unsupervised Consistency Loss (UDA/SSDA 확장)

레이블이 없는 데이터에 대한 일관성 학습:

```math
\min_{G_f, G_c} \mathcal{L}_{unl} = \lambda_{unl} \mathbb{E}_{\mathbf{x} \in S_{unl}} \sum_{k=1}^{K} \left\{ G_c(\text{SR}(G_f(\mathbf{x}), \mathbf{z}'))_k - G_c(G_f(\mathbf{x}))_k \right\}^2
```

---

### 2.3 모델 구조

```
Input x ──► [Feature Extractor Gf] ──► z
                    │
        ┌───────────┼───────────────┐
        │           │               │
  [SR Module]  [Adversarial]  [CR Module]
        │         Learning         │
        ▼                          ▼
[Content-Biased   ◄──────  [Style-Biased
  Network Gc]               Network Gs]
(Target Output)           (Auxiliary Only)
```

| 구성 요소 | 역할 | 테스트 시 사용 |
|---|---|---|
| $G_f$ (Feature Extractor) | 초기 몇 개 스테이지 | ✅ |
| $G_c$ (Content-Biased Network) | 나머지 스테이지, 콘텐츠 집중 | ✅ |
| $G_s$ (Style-Biased Network) | $G_c$와 동일 구조, 스타일 집중 | ❌ (학습 시만) |

**핵심 설계 특징:**
- 테스트 시 $G_c \circ G_f$는 원래 CNN과 **동일한 구조** → 추가 파라미터/연산 없음
- 미니배치 내 셔플로 $\mathbf{z}'$ 구성 → 추가 forward pass 불필요
- 어파인 변환 파라미터에 대해 적대적 학습 수행

---

### 2.4 성능 향상

#### Domain Generalization (PACS, ResNet-18)

| 방법 | Art | Cartoon | Sketch | Photo | **Avg.** |
|---|---|---|---|---|---|
| DeepAll | 78.12 | 75.10 | 68.43 | 95.37 | 79.26 |
| JiGen | 79.42 | 75.25 | 71.35 | 96.03 | 80.51 |
| MASF | 80.29 | 77.17 | 71.69 | 94.99 | 81.04 |
| MMLD | 81.28 | 77.16 | 72.29 | **96.09** | 81.83 |
| **SagNet** | **83.58** | **77.66** | **76.30** | 95.47 | **83.25** |

#### UDA (Office-Home, ResNet-50 기반)

DANN + SagNet: 57.7% → 60.6% (+2.9%), CDAN + SagNet: 64.0% → 65.3% (+1.3%)

#### SSDA (DomainNet, ResNet-34, 1-shot)

MME: 66.4% → MME + SagNet: 68.6% (+2.2%)

---

### 2.5 한계

1. **최적 하이퍼파라미터 민감도**: $\lambda_{adv}$가 너무 크면 최적화가 불안정해지고 오히려 성능 하락 → 적절한 trade-off 탐색 필요
2. **랜덤화 스테이지 선택**: 너무 낮은 레이어(저수준 스타일)나 너무 높은 레이어(고수준 의미 정보 손실) 모두 효과 감소 → 수동 설정 필요
3. **스타일 표현의 단순성**: 채널별 평균/분산만을 스타일로 정의 → 더 복잡한 스타일 정보를 포착하지 못할 수 있음
4. **콘텐츠-스타일 이분법의 가정**: 실제로 두 정보는 얽혀 있을 수 있음
5. **대규모·복잡 데이터셋 한계**: DomainNet에서 CDAN과의 조합 효과가 제한적

---

## 3. 모델 일반화 성능 향상 가능성

### 3.1 핵심 메커니즘: 스타일 편향 → 도메인 격차 상관관계 실증

논문은 **shape bias와 A-distance(도메인 격차)** 사이의 명시적 상관관계를 실험으로 증명합니다:

$$d_{\mathcal{A}} = 2(1 - \epsilon)$$

여기서 $\epsilon$은 두 도메인을 구분하도록 학습된 SVM 분류기의 일반화 오차입니다.

Shape bias가 높아질수록 A-distance가 감소 → **콘텐츠 기반 표현이 본질적으로 더 domain-invariant**

### 3.2 일반화 향상의 다섯 가지 경로

```
1. 스타일 랜덤화(SR)
   └─► 무한한 스타일 변형 데이터 증강 효과
   └─► 스타일에 불변한(style-invariant) 표현 학습

2. 적대적 스타일-편향 학습(ASBL)
   └─► 특징 공간에서 스타일 정보의 클래스 판별력 제거
   └─► 더 domain-invariant한 표현 유도

3. 도메인 레이블 불필요
   └─► 단일 소스 도메인에서도 적용 가능
   └─► 도메인 경계가 불명확한 실제 시나리오에 적합

4. 기존 방법과의 직교성
   └─► UDA 방법(DANN, JAN, CDAN 등)과 결합 시 일관된 추가 성능 향상
   └─► 분포 정렬 + 편향 제거의 시너지

5. 일관성 학습(Consistency Learning)
   └─► 레이블 없는 데이터의 SR 적용/미적용 예측 일관성 강제
   └─► 타겟 도메인 데이터의 효과적 활용
```

### 3.3 일반화 가능성의 이론적 근거

CNN의 스타일 편향을 줄이는 것은 Vapnik-Chervonenkis 이론 관점에서 **가설 공간을 콘텐츠 기반으로 제약**하는 효과가 있습니다. 이는 도메인 간 공유되는 인과적 특징(causal features)에 집중하게 하여 이론적으로도 일반화에 유리합니다.

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 연구에 미치는 영향

#### 🔵 패러다임 전환
기존의 "도메인 분포 정렬" 중심 패러다임에서 **"CNN의 내재적 귀납 편향 조정"** 이라는 새로운 관점을 제시합니다. 이후 연구들이 CNN의 귀납 편향 자체를 도메인 일반화 문제의 핵심으로 다루는 흐름을 촉진했습니다.

#### 🔵 데이터 증강 기반 DG 연구 촉진
스타일 랜덤화 아이디어는 이후 더 정교한 특징 공간 증강 연구들에 영향을 주었습니다 (예: MixStyle, DSU 등).

#### 🔵 Plug-and-Play 모듈 설계 트렌드
테스트 시 오버헤드 없이 학습 중에만 작동하는 보조 모듈 설계 방식은 이후 많은 연구에서 채택되었습니다.

---

### 4.2 2020년 이후 관련 최신 연구 비교 분석

| 논문 | 핵심 방법 | SagNet과의 관계 | 주요 차이점 |
|---|---|---|---|
| **MixStyle** (Zhou et al., ICLR 2021) | 인스턴스 정규화 통계를 혼합하여 새로운 스타일 생성 | SagNet의 SR 아이디어와 유사 | 적대적 학습 없이 순수 증강만으로 단순화; 레이어 내 삽입 |
| **DSU** (Li et al., ICLR 2022) | 특징 통계의 불확실성을 모델링하여 스타일 다양화 | SR 개념 확장 | 분포 수준의 불확실성 모델링으로 더 정교한 스타일 공간 탐색 |
| **pAdaIN** (Nuriel et al., CVPR 2021) | 픽셀 단위 AdaIN으로 스타일 변환 | SR의 픽셀 수준 확장 | 더 세밀한 공간적 스타일 제어 |
| **EFDM** (Zhang et al., CVPR 2022) | 특징 분포 매칭을 통한 스타일 전이 | AdaIN 기반 스타일 조작의 대안 | 히스토그램 매칭으로 더 정확한 통계 전이 |
| **DomainBed** (Gulrajani & Lopez-Paz, ICLR 2021) | DG 방법들의 공정한 벤치마크 | SagNet 포함 평가 | 재현성/공정성 이슈 제기; 하이퍼파라미터 튜닝의 중요성 강조 |

#### DomainBed에서의 SagNet 위치
Gulrajani & Lopez-Paz (2021)의 DomainBed 벤치마크는 공정한 하이퍼파라미터 탐색 하에서 많은 정교한 DG 방법들이 단순한 ERM(Empirical Risk Minimization) 베이스라인 대비 일관된 우위를 보이지 못함을 보였습니다. 이는 SagNet을 포함한 기존 방법들의 실제 일반화 효과에 대한 보다 엄격한 재평가를 촉구합니다.

---

### 4.3 앞으로 연구 시 고려할 점

#### 📌 방법론적 고려사항

1. **스타일 표현의 정교화**
   - 채널별 1차·2차 통계를 넘어 고차 통계(higher-order statistics), Gram matrix, 혹은 학습된 스타일 표현 사용 검토
   - Vision Transformer(ViT)에서 스타일 정보가 어떻게 인코딩되는지 분석 필요

2. **동적 $\lambda_{adv}$ 스케줄링**
   - 고정 하이퍼파라미터 대신 학습 진행에 따라 동적으로 조정하는 방법 연구

3. **Vision Transformer로의 확장**
   - SagNet은 CNN의 채널 통계를 기반으로 설계되었으나, ViT에서는 패치 임베딩 기반 스타일 정의가 필요
   - 최근 ViT 기반 DG 연구 (예: TVT, CDTrans 등)와의 통합 검토

4. **인과적 관점 통합**
   - 스타일 = 비인과적 특징, 콘텐츠 = 인과적 특징으로 보는 Invariant Risk Minimization (IRM) 등과의 이론적 연결 탐색

5. **공정한 평가 프로토콜 준수**
   - DomainBed가 지적한 하이퍼파라미터 튜닝 공정성 문제를 고려하여 엄격한 비교 실험 설계

6. **의료 영상 등 도메인 특수 응용**
   - 의료 영상에서의 스캐너 간 도메인 격차(scanner gap)는 스타일 편향과 직결 → SagNet 원리의 적용 가능성 높음

#### 📌 이론적 고려사항

7. **Shape bias ↔ 일반화 성능 상관관계의 보편성 검증**
   - 모든 데이터셋/태스크에서 성립하는지 추가 이론적·실험적 검증 필요

8. **스타일과 콘텐츠의 얽힘(entanglement) 문제**
   - 완전한 분리가 어려운 경우 (예: 텍스처가 의미론적으로 중요한 경우) 처리 방안 연구

---

## 참고 자료

**논문 원문:**
- Nam, H., Lee, H., Park, J., Yoon, W., & Yoo, D. (2021). **Reducing Domain Gap by Reducing Style Bias**. *CVPR 2021*. arXiv:1910.11645v4.

**논문 내 핵심 인용 문헌:**
- Geirhos, R. et al. (2019). **ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness**. *ICLR 2019*.
- Huang, X. & Belongie, S. (2017). **Arbitrary style transfer in real-time with adaptive instance normalization**. *ICCV 2017*.
- Ganin, Y. et al. (2016). **Domain-adversarial training of neural networks**. *JMLR 2016*.
- Saito, K. et al. (2019). **Semi-supervised domain adaptation via minimax entropy**. *ICCV 2019*.

**비교 분석에 사용된 2020년 이후 연구:**
- Zhou, K. et al. (2021). **Domain Generalization with MixStyle**. *ICLR 2021*.
- Gulrajani, I. & Lopez-Paz, D. (2021). **In Search of Lost Domain Generalization**. *ICLR 2021*. (DomainBed)
- Li, Y. et al. (2022). **Uncertainty Modeling for Out-of-Distribution Generalization**. *ICLR 2022*. (DSU)
- Zhang, Y. et al. (2022). **Exact Feature Distribution Matching for Arbitrary Neural Style Transfer and Domain Generalization**. *CVPR 2022*. (EFDM)

> **정확도 주의:** 비교 분석 테이블의 일부 수치 및 DomainBed 관련 내용은 해당 논문들의 주요 내용을 기반으로 요약한 것이며, 세부 수치는 각 원문 확인을 권장합니다.
