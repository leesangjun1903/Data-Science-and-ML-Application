# Training Region-based Object Detectors with Online Hard Example Mining (OHEM)

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장
Object Detection 학습 시 발생하는 **클래스 불균형 문제**(쉬운 배경 예시 압도적 다수 vs. 어려운 예시 소수)를 해결하기 위해, SGD 기반 온라인 학습 환경에서 적용 가능한 **Online Hard Example Mining(OHEM)** 알고리즘을 제안한다.

### 주요 기여
| 기여 항목 | 설명 |
|---|---|
| 휴리스틱 제거 | bg_lo, fg-bg ratio 등 수동 설정 하이퍼파라미터 불필요 |
| 성능 향상 | PASCAL VOC 2007/2012, MS COCO에서 일관된 mAP 향상 |
| 확장성 | 데이터셋이 크고 어려울수록 효과 증가 (MS COCO) |
| 상보성 | Multi-scale, Iterative bbox regression 등과 독립적으로 결합 가능 |
| 구현 효율성 | Readonly RoI network 구조로 메모리·속도 효율 확보 |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

Object Detection 학습에서 발생하는 두 가지 핵심 문제:

**① 심각한 클래스 불균형 (Class Imbalance)**

$$\text{Imbalance Ratio} \approx 70:1 \sim 100{,}000:1 \quad (\text{background} : \text{foreground})$$

Sliding-window 방식에서는 배경 예시가 최대 $100{,}000 : 1$ 비율로 압도적이며, region proposal 방식에서도 여전히 $70:1$ 수준이다.

**② 기존 Fast R-CNN의 비효율적 학습**

기존 Fast R-CNN(FRCN)은 다음의 휴리스틱에 의존:
- **bg_lo = 0.1**: IoU가 $[0.1, 0.5)$ 인 배경만 사용 (0 이하 무시)
- **fg:bg = 1:3 비율 강제 유지**: 미니배치의 25%를 foreground로 고정

이러한 휴리스틱은 **suboptimal**하며 드문 hard example을 놓치는 문제가 있다.

---

### 2.2 제안하는 방법 (수식 포함)

#### OHEM 알고리즘의 핵심 수식

**Fast R-CNN의 per-RoI 손실 함수:**

$$\mathcal{L}(p, u, t^u, v) = \mathcal{L}_{cls}(p, u) + \lambda [u \geq 1] \mathcal{L}_{loc}(t^u, v)$$

여기서:
- $p = (p_0, \ldots, p_K)$: softmax 클래스 확률 분포
- $u$: ground-truth 클래스 레이블
- $t^u$: 클래스 $u$에 대한 예측 bounding box 좌표
- $v$: ground-truth bounding box 좌표
- $\mathcal{L}_{cls}(p, u) = -\log p_u$: 분류 손실 (Cross-entropy)
- $\mathcal{L}_{loc}$: Smooth-L1 bounding box regression 손실

$$\mathcal{L}_{loc}(t^u, v) = \sum_{i \in \{x,y,w,h\}} \text{smooth}_{L_1}(t^u_i - v_i)$$

$$\text{smooth}_{L_1}(x) = \begin{cases} 0.5x^2 & \text{if } |x| < 1 \\ |x| - 0.5 & \text{otherwise} \end{cases}$$

**OHEM의 Hard Example 선택 기준:**

SGD 이터레이션 $t$에서 이미지의 모든 RoI 집합 $R$에 대해 각 RoI의 손실을 계산:

$$\mathcal{L}_i = \mathcal{L}(p_i, u_i, t_i^{u_i}, v_i), \quad \forall r_i \in R$$

Hard example 선택:

$$R_{\text{hard-sel}} = \underset{|S|=B/N}{\arg\max_{S \subseteq R}} \sum_{r_i \in S} \mathcal{L}_i$$

즉, 손실 기준으로 내림차순 정렬 후 상위 $B/N$개 선택.

**중복 제거 (NMS with relaxed threshold):**

겹치는 RoI 간 손실 중복 계산 방지를 위해 NMS 적용:

$$\text{NMS threshold} = 0.7 \quad (\text{IoU 기준})$$

**Foreground RoI 기준:**

$$\text{IoU}(r_i, g) \geq 0.5 \Rightarrow \text{foreground}$$

**Background RoI 기준 (OHEM에서 개선):**

기존 FRCN:

$$\text{bg lo} \leq \max_g \text{IoU}(r_i, g) < 0.5, \quad \text{bg lo} = 0.1$$

OHEM 적용 시:

$$0 \leq \max_g \text{IoU}(r_i, g) < 0.5 \quad (\text{bg lo} = 0, \text{제한 없음})$$

---

### 2.3 모델 구조

OHEM은 Fast R-CNN을 기반으로 하며, **두 개의 RoI 네트워크**를 사용하는 구조를 채택한다.

```
입력 이미지
    │
    ▼
[Convolutional Network]  ← Conv feature map 생성
    │
    ├──────────────────────────────────┐
    ▼                                  ▼
[Readonly RoI Network (a)]      [Regular RoI Network (b)]
 - 모든 RoI (|R| ≈ 4000)에 대해    - Hard RoI만 (B=128) 선택하여
   Forward pass만 수행              Forward + Backward 수행
 - 각 RoI의 손실 계산
    │
    ▼
[Hard & Diverse RoI Sampler]
 - 손실 기준 정렬 → 상위 B개 선택
 - NMS (IoU=0.7)로 중복 제거
    │
    ▼ R_hard-sel
[Regular RoI Network (b)]
 - Gradient 계산 및 업데이트
    │
    ▼
[Conv Network Backward]
```

**계산 순서:**
1. Conv Network forward
2. Readonly RoI Network: 모든 $r_i \in R$에 대해 forward
3. Hard RoI Sampler: $R_{\text{hard-sel}}$ 선택
4. Regular RoI Network: $R_{\text{hard-sel}}$에 대해 forward + backward
5. Conv Network backward

**구현 세부사항 (논문 설정):**

| 파라미터 | 값 |
|---|---|
| $N$ (images per mini-batch) | 2 |
| $B$ (batch size) | 128 |
| $\|R\|$ (전체 RoI 수) | $\approx 4000$ |
| NMS IoU threshold | 0.7 |
| Learning rate | 0.001 |
| LR decay | 30k iter마다 $\times 0.1$ |
| Total SGD iterations | 80k (VOC07) |

---

### 2.4 성능 향상

#### PASCAL VOC 2007 결과

| 방법 | 학습 데이터 | M | B | mAP (%) |
|---|---|---|---|---|
| Fast R-CNN | 07 | | | 66.9 |
| FRCN* (재현) | 07 | | | 67.2 |
| **Ours (OHEM)** | 07 | | | **69.9** |
| FRCN* | 07 | ✓ | ✓ | 72.4 |
| MR-CNN | 07 | ✓ | ✓ | 74.9 |
| **Ours (OHEM)** | 07 | ✓ | ✓ | **75.1** |
| **Ours (OHEM)** | 07+12 | ✓ | ✓ | **78.9** |

#### PASCAL VOC 2012 결과

| 방법 | 학습 데이터 | mAP (%) |
|---|---|---|
| Fast R-CNN | 12 | 65.7 |
| **Ours (OHEM)** | 12 | **69.8** |
| MR-CNN | 07++12 | 73.9 |
| **Ours (OHEM)** | 07++12 | **76.3** |

#### MS COCO 2015 test-dev 결과

| 방법 | AP@[0.5:0.95] | AP@0.50 |
|---|---|---|
| Fast R-CNN | 19.7 | 35.9 |
| **Ours (OHEM)** | **22.6** | **42.5** |
| Ours + Multi-scale | 24.4 | 44.4 |
| Ours* + Multi-scale (trainval) | 25.5 | 45.9 |

#### 계산 오버헤드 (VGG16 기준)

| | FRCN | OHEM |
|---|---|---|
| 시간 (sec/iter) | 0.57 | 1.00 |
| 최대 메모리 (GB) | 6.4 | 8.7 |

---

### 2.5 한계

1. **추가 메모리 사용**: Readonly RoI network 유지로 추가 메모리 필요 (VGG16 기준 +2.3GB)
2. **학습 속도 증가**: 약 $1.75\times$ 학습 시간 증가
3. **Anchor-free 방법에 직접 적용 어려움**: RoI 기반 구조를 전제로 설계됨
4. **카테고리별 성능 편차**: bottle, chair, tvmonitor 등 특정 클래스에서 더 큰 향상 — 이유가 논문에서 미해명
5. **Convergence 보장 없음**: SVM의 hard negative mining과 달리 수렴 증명 부재
6. **Region proposal 의존성**: Selective Search 기반으로, proposal 품질에 성능이 종속

---

## 3. 모델의 일반화 성능 향상 가능성

OHEM이 일반화 성능에 기여하는 메커니즘은 다음과 같이 분석할 수 있다.

### 3.1 어려운 예시 학습을 통한 일반화

$$\text{Hard Example} = \{r_i \mid \mathcal{L}_i \text{ is large}\}$$

모델이 **현재 취약한 패턴**에 집중적으로 학습함으로써, 단순한 랜덤 샘플링 대비 더 다양하고 어려운 경계 사례(boundary cases)를 학습한다. 이는 과적합보다는 **학습 분포의 다양성 확대**로 이어진다.

### 3.2 bg_lo 제거의 효과

기존 FRCN의 bg_lo = 0.1은 IoU가 0~0.1인 쉬운 배경 영역(하늘, 잔디 등)을 의도적으로 제외했다. OHEM에서 이를 제거함으로써 **다양한 배경 패턴**에 대한 학습이 가능해진다:

$$R_{\text{bg}} = \{r_i \mid 0 \leq \max_g \text{IoU}(r_i, g) < 0.5\}$$

이는 모델이 실제 추론 환경에서 만날 수 있는 다양한 배경 분포에 더 강건하게 만든다.

### 3.3 자동 fg-bg 균형 조절

명시적 fg:bg = 1:3 비율 대신, OHEM은 손실 기반으로 자동 조절:

$$P(\text{선택}_{r_i}) \propto \mathcal{L}_i$$

특정 클래스가 무시되면 그 손실이 증가하여 자연스럽게 샘플링 확률이 높아지는 **자기 조정(self-correcting)** 메커니즘이다. 이는 데이터셋 도메인이 바뀌어도 자동 적응하는 효과를 제공한다.

### 3.4 MS COCO에서의 검증

MS COCO는 PASCAL VOC보다 훨씬 크고 다양한 데이터셋(80 클래스, 작은 객체 다수):

$$\Delta \text{AP}_{[0.5:0.95]} = 22.6 - 19.7 = +2.9$$
$$\Delta \text{AP}_{50} = 42.5 - 35.9 = +6.6$$

특히 중간 크기 객체(medium):

$$\Delta \text{AP}_{\text{med}} = 23.7 - 18.8 = +4.9$$

이 결과는 **데이터셋이 크고 다양할수록 OHEM의 일반화 효과가 커짐**을 실증한다.

### 3.5 강건한 그래디언트 추정

$N=1$ (이미지 1장/미니배치) 실험에서:

| 방법 | N=2 mAP | N=1 mAP | 차이 |
|---|---|---|---|
| FRCN | 67.2 | 66.3 | -0.9 |
| **OHEM** | **69.9** | **69.7** | **-0.2** |

OHEM은 이미지 수가 줄어도 성능 저하가 거의 없어, **배치 구성 변화에 강건**하다.

---

## 4. 앞으로의 연구에 미치는 영향과 고려할 점

### 4.1 후속 연구에 미친 영향

OHEM은 이후 객체 검출 분야에 광범위한 영향을 미쳤다:

**① Focal Loss (RetinaNet, 2017)**

Lin et al.은 OHEM의 아이디어를 발전시켜 **동적 가중치**를 손실 함수에 직접 내장:

$$\mathcal{L}_{\text{focal}}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

이는 OHEM의 이산적 선택(hard selection)을 연속적 가중치로 대체한 것으로, OHEM의 핵심 직관을 계승한다.

**② Anchor-free Detectors에서의 응용**

FCOS, CenterNet 등 anchor-free 방법에서도 hard example 선택 전략이 중요하게 다뤄진다. ATSS(Adaptive Training Sample Selection, 2020)는 OHEM의 선택 전략을 anchor 할당 단계까지 확장한다.

**③ 다른 태스크로의 확장**

OHEM의 원리는 Instance Segmentation (Mask R-CNN), 3D Detection, Video Object Detection, NLP 등 다양한 분야의 hard example 처리 전략에 영향을 주었다.

### 4.2 2020년 이후 관련 최신 연구 비교 분석

#### ① Probabilistic Anchor Assignment (PAA, 2020)

> Kim & Lee, "Probabilistic Anchor Assignment with IoU Prediction for Object Detection," ECCV 2020

- OHEM의 손실 기반 선택을 **확률적 앵커 할당**으로 발전
- IoU 예측을 통해 앵커 품질을 추정하고 GMM으로 fg/bg를 동적 분리
- OHEM보다 원칙적인 확률 이론 기반

$$p(\text{fg} \mid r_i) = \frac{\pi_{\text{fg}} \cdot \mathcal{N}(\text{IoU}_i; \mu_{\text{fg}}, \sigma_{\text{fg}}^2)}{\pi_{\text{fg}} \cdot \mathcal{N}(\text{IoU}_i; \mu_{\text{fg}}, \sigma_{\text{fg}}^2) + \pi_{\text{bg}} \cdot \mathcal{N}(\text{IoU}_i; \mu_{\text{bg}}, \sigma_{\text{bg}}^2)}$$

#### ② Varifocal Loss (VFL, 2021)

> Zhang et al., "VarifocalNet: An IoU-aware Dense Object Detector," CVPR 2021

- Focal Loss를 확장, IoU-aware 점수로 가중치 부여
- OHEM의 이진 선택 대신 연속적 중요도 가중치

$$\mathcal{L}_{\text{VFL}} = \begin{cases} -q(q \log p + (1-q)\log(1-p)) & q > 0 \\ -\alpha p^\gamma \log(1-p) & q = 0 \end{cases}$$

#### ③ ATSS (Adaptive Training Sample Selection, 2020)

> Zhang et al., "Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection," CVPR 2020

- 통계적 기준으로 앵커를 자동 선택 (평균 + 표준편차 기반 IoU threshold)
- OHEM의 loss 기반 선택과 달리 **공간적 통계**를 활용

#### ④ GFL (Generalized Focal Loss, 2020)

> Li et al., "Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection," NeurIPS 2020

- 분류와 localization quality를 통합한 단일 표현으로 hard example 문제 접근
- OHEM에서 제기된 fg/bg 불균형을 손실 함수 설계 차원에서 해결

#### ⑤ DETReg / Sparse DETR (2021~)

- Transformer 기반 Detection에서는 attention mechanism이 암묵적으로 hard example에 집중하는 효과를 가짐
- OHEM의 명시적 선택이 불필요할 수 있으나, 학습 초기 수렴 가속을 위해 여전히 응용 가능

#### 비교 요약표

| 방법 | 연도 | 핵심 아이디어 | OHEM 대비 차이점 |
|---|---|---|---|
| **OHEM** | 2016 | 손실 기반 이산 선택 | 기준 방법 |
| Focal Loss | 2017 | 연속적 가중치 손실 | 선택→가중치, 단일단계 |
| ATSS | 2020 | 통계적 앵커 선택 | 공간적 통계 활용 |
| PAA | 2020 | 확률적 앵커 할당 | GMM 기반 soft assignment |
| VFL | 2021 | IoU-aware 가중치 | Quality-aware 확장 |
| GFL | 2020 | 통합 focal loss | 손실 설계 차원 해결 |

---

### 4.3 앞으로 연구 시 고려할 점

**① Anchor-free / DETR 계열과의 통합**

현재 주류인 DETR, Deformable DETR 등 transformer 기반 방법에서 OHEM의 원리를 어떻게 적용할지 연구 필요. 특히 Hungarian matching 기반 학습과 hard example 선택의 결합 방식이 미해결 과제다.

**② Hard Example의 정의 다양화**

현재 OHEM은 단순히 **손실 크기**를 기준으로 하나, 다음 기준들을 복합적으로 고려하는 방향이 유망하다:

$$\text{difficulty}(r_i) = f(\mathcal{L}_i, \text{IoU}_i, \text{class frequency}_i, \text{scale}_i)$$

**③ Curriculum Learning과의 결합**

초기에는 쉬운 예시, 후기에는 어려운 예시로 점진적 학습:

$$\tau(t) = \mathcal{L}_{\text{threshold}}(t) = \mathcal{L}_{\max} \cdot \left(1 - e^{-\lambda t}\right)$$

이를 OHEM에 접목하여 학습 안정성 향상 가능.

**④ Self-Supervised / Semi-Supervised 환경**

라벨이 부족한 환경에서 OHEM을 어떻게 적용할지 연구 필요. Pseudo-label과 결합 시 noisy hard example 문제 발생 가능.

**⑤ 수렴 이론 정립**

기존 SVM hard negative mining은 수렴 보장이 있으나, OHEM은 SGD 기반 수렴 증명이 없다. 이론적 분석이 향후 연구에서 중요하다.

**⑥ 개별 카테고리 성능 분석**

논문 스스로 "흥미롭고 열린 질문"으로 남긴 카테고리별 성능 차이의 원인 규명:
- Bottle, Chair, TV Monitor에서 더 큰 향상 → 시각적 복잡성, 크기 분산 등과의 연관성 분석 필요

---

## 참고 자료

1. **원 논문**: Shrivastava, A., Gupta, A., & Girshick, R. (2016). "Training Region-based Object Detectors with Online Hard Example Mining." *CVPR 2016*. (제공된 PDF)

2. Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). "Focal Loss for Dense Object Detection." *ICCV 2017*.

3. Zhang, S., Chi, C., Yao, Y., Lei, Z., & Li, S. Z. (2020). "Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection." *CVPR 2020*.

4. Kim, K., & Lee, H. S. (2020). "Probabilistic Anchor Assignment with IoU Prediction for Object Detection." *ECCV 2020*.

5. Zhang, H., Wang, Y., Dayoub, F., & Sunderhauf, N. (2021). "VarifocalNet: An IoU-aware Dense Object Detector." *CVPR 2021*.

6. Li, X., Wang, W., Wu, L., Chen, S., Hu, X., Li, J., ... & Yang, J. (2020). "Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection." *NeurIPS 2020*.

7. Girshick, R. (2015). "Fast R-CNN." *ICCV 2015*.
