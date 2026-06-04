# A Novel Self-Supervised Re-labeling Approach for Training with Noisy Labels

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

본 논문은 **mCT-S2R** (modified Co-Teaching with Self-Supervision and Re-labeling)이라는 프레임워크를 제안합니다. 핵심 주장은 다음과 같습니다:

> "자기지도학습(Self-Supervised Learning)을 통해 레이블 없이 강건한 특징을 학습하고, 이를 기반으로 노이즈 레이블 데이터를 재레이블링(Re-labeling)함으로써, 기존 Co-teaching 방식 대비 현저히 향상된 노이즈 내성 분류 성능을 달성할 수 있다."

### 주요 기여

| 기여 항목 | 설명 |
|---|---|
| 자기지도 사전학습 도입 | RotNet 기반 레이블 독립적 특징 학습으로 노이즈 오버피팅 방지 |
| 대손실 샘플 재레이블링 | 소손실 샘플의 클래스 평균으로 노이즈 샘플에 의사 레이블 부여 |
| 단일 네트워크 최종 학습 | 재레이블링 이후 하나의 네트워크만 사용하여 계산 효율 향상 |
| 벤치마크 성능 우위 | MNIST, CIFAR-10, CIFAR-100에서 SOTA 초과 달성 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

딥러닝은 대규모 정제 데이터에 의존하지만, 현실에서는 웹 크롤링·크라우드소싱 등으로 수집된 **노이즈 레이블**이 불가피합니다. 심층 신경망은 메모리제이션 능력으로 인해 노이즈 레이블에도 과적합되는 문제가 있습니다 (Zhang et al., 2016; Arpit et al., 2017).

기존 Co-teaching [Han et al., 2018]은 두 병렬 네트워크가 서로의 소손실 샘플을 교환하며 학습하지만:
- 훈련이 길어질수록 두 네트워크가 동일한 상태로 수렴
- 대손실(large-loss) 샘플을 단순 무시하여 데이터 활용 비효율
- 두 네트워크를 전체 훈련 내내 유지하는 계산 부담

### 2.2 제안 방법 및 수식

#### Step 1: 자기지도 사전학습 (RotNet 기반)

이미지를 $0°, 90°, 180°, 270°$ 회전시켜 생성한 레이블로 네트워크를 사전학습합니다.

$$\mathcal{L}^{ce} = -\log\left(\frac{e^{u^{[i]}}}{\sum_{k=1}^{K} e^{u^{[k]}}}\right)$$

여기서 $K=4$ (4가지 회전 각도), $i$는 정답 회전 클래스 인덱스입니다.

#### Step 2: 소손실 샘플 계산 (Co-teaching 수정판)

에폭 $T$에서 미니배치 $\mathcal{D}^{mb}$에 대해 소손실 샘플 집합을 선택:

$$\mathcal{D}^j = \arg\min_{x: |x| \geq R(T)|\mathcal{D}^{mb}|} \mathcal{L}^{ce}(j, x)$$

네트워크 가중치 업데이트 (피어 네트워크의 소손실 샘플로 학습):

$$w_j = w_j - \eta \nabla \mathcal{L}^{ce}(j, \mathcal{D}^{\bar{j}})$$

$(j \in \{p, q\},\; \text{if } j=p \text{ then } \bar{j}=q)$

손실 샘플 비율 스케줄링:

```math
R(T) = 1 - \min\left\{\frac{T}{T_k}\varepsilon,\; \varepsilon\right\}
```

여기서 $\varepsilon$은 노이즈율, $T_k$는 $R(T)$가 일정해지는 에폭입니다.

#### Step 3: 대손실 샘플 재레이블링

클래스별 평균 특징벡터 계산 ($N$개의 소손실 샘플 사용):

$$\mu_k = \frac{1}{N}\sum_{i \in S^p_k} f^p_i, \quad k \in \{1, \ldots, K\}$$

대손실 샘플 $i$에 대해 각 클래스 평균까지의 거리 $d$를 계산하고, 소프트맥스로 유사도 변환:

$$\hat{d} = \text{softmax}(-d)$$

의사 레이블 및 신뢰도 할당:

$$y^r = \arg\max_k \hat{d}[k]$$

$$c = \max_k \hat{d}[k]$$

#### Step 4: 증강 데이터로 최종 학습

신뢰도 임계값 $\kappa$를 초과하는 재레이블 샘플 $\mathcal{D}^r$를 선택:

$$\mathcal{D}^{aug} = \mathcal{D}^s \cup \mathcal{D}^r, \quad \text{where } \mathcal{D}^r = \{x : c \geq \kappa\}$$

이후 단일 네트워크 $p$만으로 $\mathcal{D}^{aug}$를 학습합니다.

### 2.3 모델 구조

```
[Image Rotation Module]
        ↓
[Conv 3x3, 128, LReLU] × 2
        ↓
[MaxPool 2x2, stride 2] + [Dropout p=0.25]
        ↓
[Conv 3x3, 256, LReLU] × 3
        ↓
[MaxPool 2x2, stride 2] + [Dropout p=0.25]
        ↓
[Conv 3x3, 512, LReLU]
[Conv 3x3, 256, LReLU]
[Conv 3x3, 128, LReLU]
        ↓
[AvgPool] → [Dense: 128 → K]
```

- **사전학습 시**: $K=4$ (회전 예측), 전체 네트워크 학습
- **노이즈 학습 시**: 처음 3개 Conv 레이어만 사전학습 가중치 로드, 나머지 랜덤 초기화
- **두 병렬 네트워크** $p, q$는 $T_{update}$ 에폭까지만 사용

### 2.4 성능 향상

#### CIFAR-10 결과 (표 3 기준)

| 노이즈 모델 | Co-teaching | mCT-R | mCT-S2R |
|---|---|---|---|
| Pair-45% | 72.62 | 78.09 | **80.58** |
| Symmetry-50% | 74.02 | 77.69 | **81.23** |
| Symmetry-20% | 82.32 | 84.89 | **87.21** |

#### CIFAR-100 결과 (표 4 기준)

| 노이즈 모델 | Co-teaching | mCT-R | mCT-S2R |
|---|---|---|---|
| Pair-45% | 34.81 | 31.80 | **38.67** |
| Symmetry-50% | 41.37 | 41.46 | **52.87** |
| Symmetry-20% | 54.23 | 56.29 | **60.49** |

### 2.5 한계점

1. **노이즈율 $\varepsilon$ 의존성**: $R(T)$ 계산 시 $\varepsilon$을 알아야 하며, 실제 환경에서 이를 정확히 추정하는 것은 어렵습니다 (단, $\varepsilon=0.5$ 고정 시에도 준수한 성능 확인).
2. **단순 거리 기반 재레이블링**: 클래스 평균까지의 유클리드 거리만 사용하여 클래스 내 분산이 큰 데이터셋에서 재레이블링 오류 가능성이 있습니다.
3. **클래스 수 확장성**: CIFAR-100 pairflip-45%에서 mCT-R이 Co-teaching보다 낮은 성능을 보이는 등, 클래스 수가 많을수록 재레이블링 오류가 누적될 수 있습니다.
4. **소규모 아키텍처**: ResNet 등 대형 모델과의 비교 미흡.
5. **실제 노이즈 데이터셋 검증 부족**: WebVision, Clothing1M 등 실세계 노이즈 데이터셋 실험 없음.
6. **RotNet 의존성**: 회전 대칭성이 없는 도메인(의료 영상 등)에서는 pretext task 부적합.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 자기지도 사전학습의 일반화 기여

RotNet 사전학습은 레이블 없이 데이터 분포를 학습하므로, 노이즈 레이블로 인한 편향된 특징 학습을 방지합니다.

$$\text{특징 품질: } \underbrace{\text{mCT-R (랜덤 초기화)}}_{\text{클러스터 미분리}} \ll \underbrace{\text{mCT-S2R (RotNet 초기화)}}_{\text{명확한 클러스터 형성}}$$

t-SNE 시각화(Figure 4, 5)에서 자기지도 사전학습을 사용했을 때 CIFAR-10(10개), CIFAR-100(100개)의 클러스터가 명확하게 분리됨을 확인합니다. 이는 재레이블링 정확도를 높이고 최종 분류기의 결정 경계를 더 일반화 가능하게 만듭니다.

### 3.2 데이터 증강을 통한 일반화

재레이블링된 대손실 샘플을 훈련 세트에 추가($\mathcal{D}^{aug} = \mathcal{D}^s \cup \mathcal{D}^r$)함으로써:

- 유효 훈련 데이터 크기 증가 → 과적합 감소
- 다양한 샘플 분포 학습 → 결정 경계 일반화

### 3.3 하이퍼파라미터 로버스트니스

| 하이퍼파라미터 | 실험 범위 | 일반화 관찰 |
|---|---|---|
| $T_{update}$ | {10, 20, 30} | 성능 일관성 유지 |
| $\kappa$ | {0.80, 0.90, 0.95} | 성능 안정적 |
| $\varepsilon$ | 실제값 vs 0.5 고정 | 0.5 고정 시 오히려 향상되는 경우도 존재 |

특히 $\varepsilon=0.5$로 고정해도 CIFAR-100 Pair-45%에서 실제값 사용($38.67$) 대비 고정값($41.06$)이 더 높게 나타나, 노이즈율을 정확히 모르는 실제 환경에서도 **일반화 성능 유지**가 가능함을 보여줍니다.

### 3.4 초기 레이어 전이를 통한 일반화

RotNet의 초기 3개 Conv 레이어 가중치만 전이하고 나머지는 랜덤 초기화함으로써:
- **일반적 저수준 특징**(엣지, 텍스처)은 사전학습에서 습득
- **태스크 특화 고수준 특징**은 분류 태스크에 맞게 재학습
- 두 네트워크가 동일 모델로 수렴하는 문제를 방지 → 다양한 관점에서의 소손실 샘플 선택 가능

---

## 4. 향후 연구에 미치는 영향 및 고려 사항

### 4.1 향후 연구에 미치는 영향

#### (1) 자기지도학습 + 노이즈 레이블 연구의 융합 촉진
본 논문은 자기지도학습을 노이즈 레이블 처리의 전처리 단계로 체계적으로 적용한 초기 연구 중 하나로, 이후 SimCLR, MoCo, DINO 등 강력한 자기지도학습 방법론과 노이즈 레이블 처리의 결합을 촉진하는 방향을 제시합니다.

#### (2) 데이터 재활용 패러다임 확립
기존의 "대손실 샘플 무시" 전략에서 "대손실 샘플 재레이블링 후 활용"으로의 패러다임 전환을 제안합니다.

#### (3) 계산 효율적 프레임워크 설계 방향
두 네트워크를 $T_{update}$까지만 사용하고 이후 단일 네트워크로 학습하는 방식은 계산 효율을 중시하는 연구 방향에 영향을 줍니다.

### 4.2 향후 연구 시 고려할 점

#### (1) 더 강력한 자기지도학습 방법 적용
RotNet 대신 대조학습 기반(SimCLR, MoCo v2, DINO)의 사전학습 적용 시 특징 품질이 대폭 향상될 수 있습니다.

#### (2) 동적 재레이블링
현재는 $T_{update}$ 시점에 한 번만 재레이블링하지만, 훈련 중 반복적으로 재레이블링을 수행하면 더 정확한 레이블을 얻을 수 있습니다.

#### (3) 혼합(Soft) 레이블 활용
현재 하드 레이블 방식 대신 소프트 레이블(클래스별 확률 분포)을 의사 레이블로 사용하면 불확실성을 더 잘 반영할 수 있습니다.

#### (4) 실세계 노이즈 데이터셋 검증
WebVision, Clothing1M, ANIMAL-10N 등 실제 노이즈가 포함된 대규모 데이터셋에서의 검증이 필요합니다.

#### (5) 노이즈율 자동 추정
$\varepsilon$ 을 사전에 알아야 한다는 가정을 제거하는 자동 노이즈율 추정 방법과의 결합이 중요합니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 주요 후속 연구

| 논문 | 방법 | 특징 | mCT-S2R 대비 |
|---|---|---|---|
| **DivideMix** (Li et al., 2020) | GMM 기반 깨끗/노이즈 데이터 분리 + MixMatch 반지도학습 | CIFAR-10 Sym-50%: **93.49%** | 대폭 향상 |
| **ELR** (Liu et al., 2020) | Early-stopping 정규화 + 지수이동평균 | 단순하지만 효과적 | CIFAR-100에서 유사 수준 |
| **CORES²** (Cheng et al., 2021) | 신뢰도 기반 샘플 선택 | 클래스 수준 신뢰도 모델링 | 개선됨 |
| **UNICON** (Karim et al., 2022) | 대조학습 + GMM 분리 | SimCLR 특징 활용 | 크게 개선 |
| **SOP** (Liu et al., 2022) | 과최적화 방지 정규화 | 이론적 보장 강화 | 개선됨 |

### 5.2 핵심 비교

```
mCT-S2R (2019) → DivideMix (2020) → UNICON (2022)
     ↑                  ↑                  ↑
소손실 기반          GMM 분리          대조학습 융합
재레이블링          반지도학습          강력한 특징
```

**CIFAR-100 Symmetry-50% 정확도 추이:**

$$\underbrace{41.37}_{\text{Co-teaching}} \rightarrow \underbrace{52.87}_{\text{mCT-S2R}} \rightarrow \underbrace{73.9}_{\text{DivideMix}} \rightarrow \underbrace{78.5}_{\text{UNICON}}$$

### 5.3 mCT-S2R의 위치

mCT-S2R은 발표 당시(2019년 말~2020년 초) 기준으로 의미 있는 성능 향상을 달성했으나, 이후 **반지도학습과의 결합**, **대조학습의 강력한 특징 표현**, **GMM 기반 정확한 클린/노이즈 분리** 등을 활용한 방법들이 등장하면서 절대 성능 면에서는 추월당했습니다. 그러나 **계산 효율성**과 **재레이블링 아이디어의 선구성**은 여전히 중요한 기여로 평가됩니다.

---

## 참고자료

**본 논문**
- Mandal, D., Bharadwaj, S., & Biswas, S. (2019). "A Novel Self-Supervised Re-labeling Approach for Training with Noisy Labels." arXiv:1910.05700v3.

**논문 내 주요 인용 문헌**
- Han, B., et al. (2018). "Co-teaching: Robust training of deep neural networks with extremely noisy labels." NeurIPS. (Co-teaching 원논문)
- Gidaris, S., Singh, P., & Komodakis, N. (2018). "Unsupervised representation learning by predicting image rotations." arXiv:1803.07728. (RotNet)
- Malach, E., & Shalev-Shwartz, S. (2017). "Decoupling when to update from how to update." NeurIPS. (Decoupling)
- Jiang, L., et al. (2017). "MentorNet: Learning data-driven curriculum for very deep neural networks on corrupted labels." arXiv:1712.05055.

**2020년 이후 비교 연구**
- Li, J., et al. (2020). "DivideMix: Learning with Noisy Labels as Semi-supervised Learning." ICLR 2020.
- Liu, S., et al. (2020). "Early-Learning Regularization Prevents Memorization of Noisy Labels." NeurIPS 2020.
- Karim, N., et al. (2022). "UniCon: Combating Label Noise Through Uniform Selection and Contrastive Learning." CVPR 2022.
- Liu, S., et al. (2022). "Self-Supervised Learning is More Robust to Dataset Imbalance." ICLR 2022.

> **⚠️ 주의**: DivideMix, UNICON, ELR, CORES², SOP의 구체적 수치는 해당 논문의 원문 기준이며, 실험 설정에 따라 다를 수 있습니다. mCT-S2R과의 직접 비교 수치는 각 논문의 공식 결과를 확인하시기 바랍니다. 본 답변은 제공된 PDF와 해당 분야의 공개된 연구 흐름을 기반으로 작성되었습니다.
