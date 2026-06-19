# Gradient Harmonized Single-stage Detector (GHM)

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

GHM 논문의 핵심 주장은 **단일 단계 객체 검출기(one-stage detector)에서 발생하는 두 가지 불균형 문제(positive/negative 예제 간 불균형, easy/hard 예제 간 불균형)를 그래디언트 노름(gradient norm) 분포의 불균형으로 통합하여 설명할 수 있다**는 것입니다.

즉, 클래스 불균형 문제는 결국 **그래디언트 밀도(gradient density)의 불균형**으로 귀결되며, 이를 조화롭게 조절함으로써 훈련 효율과 성능을 동시에 향상시킬 수 있다고 주장합니다.

### 주요 기여 (3가지)

| 기여 | 내용 |
|------|------|
| **이론적 기여** | 예제 불균형 문제를 그래디언트 노름 분포로 재해석하는 새로운 관점 제시 |
| **방법론적 기여** | GHM을 분류 손실(GHM-C)과 회귀 손실(GHM-R)에 각각 임베딩 |
| **실용적 기여** | 별도의 데이터 샘플링 전략 없이 COCO test-dev에서 41.6 mAP 달성 (Focal Loss 대비 +0.8) |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

단일 단계 검출기는 훈련 시 두 가지 근본적인 불균형(disharmony)에 직면합니다:

1. **Positive vs. Negative 불균형**: 배경(negative) 예제의 수가 전경(positive) 예제보다 압도적으로 많음
2. **Easy vs. Hard 예제 불균형**: 쉬운 예제가 압도적으로 많아 어려운 예제의 학습 기여를 무력화

기존 방법들의 한계:
- **OHEM**: 대부분의 예제를 버려 훈련 비효율
- **Focal Loss**: 두 개의 하이퍼파라미터($\alpha$, $\gamma$) 튜닝 필요, 데이터 분포 변화에 정적(static)으로 대응

---

### 2.2 제안하는 방법 (수식 포함)

#### Step 1: 그래디언트 노름 정의

이진 교차 엔트로피 손실:

```math
L_{CE}(p, p^*) = \begin{cases} -\log(p) & \text{if } p^* = 1 \\ -\log(1-p) & \text{if } p^* = 0 \end{cases}
```

모델 출력 $x$에 대한 그래디언트 ( $p = \text{sigmoid}(x)$ ):

$$\frac{\partial L_{CE}}{\partial x} = p - p^* \tag{2}$$

그래디언트 노름 $g$ 정의:

```math
g = |p - p^*| = \begin{cases} 1-p & \text{if } p^* = 1 \\ p & \text{if } p^* = 0 \end{cases}
```

$g$는 예제의 난이도(difficulty)를 나타내며:
- $g \approx 0$: 쉬운 예제 (잘 분류됨)
- $g \approx 1$: 어려운 예제 또는 아웃라이어

#### Step 2: 그래디언트 밀도 함수

$$GD(g) = \frac{1}{l_\epsilon(g)} \sum_{k=1}^{N} \delta_\epsilon(g_k, g) \tag{4}$$

여기서:

$$\delta_\epsilon(x, y) = \begin{cases} 1 & \text{if } y - \frac{\epsilon}{2} \leq x < y + \frac{\epsilon}{2} \\ 0 & \text{otherwise} \end{cases} \tag{5}$$

$$l_\epsilon(g) = \min\left(g + \frac{\epsilon}{2}, 1\right) - \max\left(g - \frac{\epsilon}{2}, 0\right) \tag{6}$$

#### Step 3: 그래디언트 밀도 조화 파라미터

$$\beta_i = \frac{N}{GD(g_i)} \tag{7}$$

- 밀도가 높은 구간의 예제 → $\beta_i$ 작아짐 → **다운 웨이팅**
- 밀도가 낮은 구간의 예제 → $\beta_i$ 커짐 → **업 웨이팅**

#### Step 4: GHM-C Loss (분류)

```math
L_{GHM-C} = \frac{1}{N} \sum_{i=1}^{N} \beta_i L_{CE}(p_i, p_i^*) = \sum_{i=1}^{N} \frac{L_{CE}(p_i, p_i^*)}{GD(g_i)}
```

#### Step 5: 단위 구간 근사 (Unit Region Approximation)

계산 복잡도 $O(N^2)$를 해결하기 위해 히스토그램 근사 도입:

$$\hat{GD}(g) = \frac{R_{ind(g)}}{\epsilon} = R_{ind(g)} \cdot M $$

$$\hat{\beta}_i = \frac{N}{\hat{GD}(g_i)} $$

$$\hat{L}_{GHM-C} = \frac{1}{N} \sum_{i=1}^{N} \hat{\beta}_i L_{CE}(p_i, p_i^*) $$

시간 복잡도: $O(MN)$, 병렬 처리 가능

#### Step 6: EMA (Exponential Moving Average)

미니배치 통계의 노이즈를 줄이기 위한 지수 이동 평균:

$$S_j^{(t)} = \alpha S_j^{(t-1)} + (1-\alpha) R_j^{(t)} $$

$$\hat{GD}(g) = \frac{S_{ind(g)}}{\epsilon} = S_{ind(g)} \cdot M $$

#### Step 7: GHM-R Loss (회귀)

기존 Smooth L1 손실의 한계: $|d| > \delta$인 모든 예제의 그래디언트 노름이 동일하게 1 → 예제 난이도 구별 불가

**Authentic Smooth L1 (ASL1) 손실** 도입:

$$ASL_1(d) = \sqrt{d^2 + \mu^2} - \mu $$

그래디언트:

$$\frac{\partial ASL_1}{\partial d} = \frac{d}{\sqrt{d^2 + \mu^2}} $$

그래디언트 노름 $g_r = \left|\frac{d}{\sqrt{d^2 + \mu^2}}\right| \in [0, 1)$로 범위가 제한됨

**GHM-R Loss:**

$$L_{GHM-R} = \frac{1}{N} \sum_{i=1}^{N} \beta_i ASL_1(d_i) = \sum_{i=1}^{N} \frac{ASL_1(d_i)}{GD(g_{r_i})} $$

---

### 2.3 모델 구조

```
입력 이미지 (800px)
        ↓
  Backbone (ResNet/ResNeXt)
        ↓
  FPN (Feature Pyramid Network)
        ↓
  RetinaNet Head
   ┌────────────────────┐
   │  Classification    │ → GHM-C Loss
   │  Branch            │
   ├────────────────────┤
   │  Box Regression    │ → GHM-R Loss
   │  Branch            │
   └────────────────────┘
```

- **Backbone**: ResNet-50 (ablation), ResNeXt-101-32x8d (최종 결과)
- **Neck**: FPN
- **Head**: RetinaNet 기반 (3 scales × 3 aspect ratios anchors)
- **특이사항**: 전용 bias 초기화 불필요, 별도의 데이터 샘플링 전략 불필요

---

### 2.4 성능 향상

#### Ablation: 단위 구간 수 M의 영향 (GHM-C, COCO minival)

| M | AP | AP₅₀ | AP₇₅ |
|---|-----|-------|-------|
| 5 | 33.4 | 51.7 | 35.6 |
| 10 | 34.6 | 53.9 | 36.5 |
| 20 | 35.2 | 54.4 | 36.9 |
| **30** | **35.8** | **55.5** | **38.1** |
| 40 | 35.4 | 54.8 | 36.3 |

#### 분류 손실 비교 (ResNet-50, COCO minival)

| 방법 | AP | AP₅₀ | AP₇₅ |
|------|-----|-------|-------|
| CE | 28.6 | 43.3 | 30.7 |
| OHEM (ResNet-101) | 31.1 | 47.2 | 33.2 |
| Focal Loss | 35.6 | 55.6 | 38.2 |
| **GHM-C** | **35.8** | **55.5** | **38.1** |

#### 회귀 손실 비교

| 방법 | AP | AP₅₀ | AP₇₅ | AP₈₀ | AP₉₀ |
|------|-----|-------|-------|-------|-------|
| SL₁ | 35.8 | 55.5 | 38.1 | 31.4 | 11.9 |
| ASL₁ | 35.7 | 55.0 | 38.1 | 31.5 | 12.1 |
| **GHM-R** | **36.4** | 54.6 | **38.7** | **32.2** | **13.1** |

> GHM-R은 AP@IoU=0.5를 소폭 낮추지만, 높은 IoU 임계값에서 성능 향상 → 정밀한 위치 추정에 효과적

#### COCO test-dev 최종 성능 (단일 모델)

| 방법 | 네트워크 | AP | AP₅₀ |
|------|---------|-----|-------|
| Faster RCNN | FPN-ResNet-101 | 36.2 | 59.1 |
| Mask RCNN | FPN-ResNeXt-101 | 39.8 | 62.3 |
| Focal Loss | RetinaNet-ResNeXt-101 | 40.8 | 61.1 |
| **GHM-C+GHM-R** | **RetinaNet-ResNeXt-101** | **41.6** | **62.8** |

#### 훈련 속도 비교

| 방법 | AP | 반복당 평균 시간(s) |
|------|-----|-----------------|
| CE | 28.6 | 0.566 |
| GHM-C Standard | 35.9 | 13.675 |
| **GHM-C RU** | **35.8** | **0.824** |

단위 구간 근사(RU)로 **약 16.6배 속도 향상**, CE 대비 약 1.46배만 증가

---

### 2.5 한계점

1. **최적 그래디언트 분포 미정의**: 논문 스스로 "균등 분포가 최적인지 불확실"하다고 인정
2. **회귀에서 easy 예제의 모호성**: 회귀 easy 예제가 항상 비중요하지 않음 (COCO 고IoU 평가에서 중요), GHM-R은 이를 일부 처리하나 완전한 해결책은 아님
3. **미니배치 의존성**: 통계적 특성상 배치 크기에 민감할 수 있음 (EMA로 완화)
4. **$\epsilon$ 하이퍼파라미터**: 단위 구간 크기 설정이 필요 (M=30으로 경험적 결정)
5. **GPU 최적화 미완성**: 당시 손실 계산이 완전히 GPU 구현되지 않아 추가 최적화 여지 존재
6. **two-stage detector의 분류 손실**에 대한 GHM-C 적용 실험 부족

---

## 3. 모델의 일반화 성능 향상 가능성

GHM은 여러 측면에서 일반화 성능 향상 가능성을 내포합니다:

### 3.1 아웃라이어 억제를 통한 일반화

$$\text{아웃라이어 조건: } g \to 1 \Rightarrow GD(g) \text{가 상대적으로 높음} \Rightarrow \beta_i \text{가 작아짐}$$

수렴된 모델에서도 지속적으로 존재하는 아웃라이어 예제들은 노이즈 레이블이거나 극단적 케이스일 가능성이 높습니다. GHM은 이들을 자동으로 **다운 웨이팅**함으로써:
- 노이즈에 과적합되는 현상 방지
- 더 일반적인 패턴 학습 촉진

### 3.2 동적 적응성 (Dynamic Adaptability)

Focal Loss가 **정적(static)** 손실 함수인 반면, GHM-C는 매 반복마다 데이터 분포를 재통계하여 **동적(dynamic)**으로 가중치를 조절합니다:

$$\beta_i^{(t)} = \frac{N}{\hat{GD}^{(t)}(g_i)}$$

훈련 초기에는 hard 예제에, 훈련 후기에는 다른 그룹에 자동 포커스 → **훈련 커리큘럼 자동화** 효과

### 3.3 도메인 이전(Domain Transfer) 가능성

GHM의 동적 특성은 데이터 분포가 변하는 도메인에서도 자동으로 적응합니다:
- 새로운 도메인의 데이터 분포 변화 → 그래디언트 밀도 자동 재계산 → 가중치 자동 조정
- 파인튜닝(fine-tuning) 시나리오에서 효과적

### 3.4 Two-stage Detector로의 일반화 검증

논문에서 GHM-R의 two-stage detector 적용을 실험적으로 검증:

| 방법 | AP | AP₅₀ | AP₇₅ |
|------|-----|-------|-------|
| SL₁ (Faster RCNN) | 36.4 | 58.7 | 38.8 |
| GHM-R (Faster RCNN) | 37.4 | 58.9 | 39.9 |

이는 GHM이 특정 아키텍처에 종속되지 않는 **범용적 메커니즘**임을 보여줍니다.

### 3.5 다른 태스크로의 확장 가능성

GHM의 핵심 아이디어(그래디언트 밀도 기반 재가중치)는 클래스 불균형이 존재하는 모든 태스크에 적용 가능:
- 의료 영상 진단 (정상/비정상 불균형)
- 자율주행 (배경/물체 불균형)
- 자연어 처리의 토큰 분류

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4.1 연구에 미치는 영향

#### (1) 그래디언트 관점의 손실 함수 설계 패러다임 확립
GHM은 손실 함수 설계를 예제 가중치 관점이 아닌 **그래디언트 분포 관점**으로 전환하는 새로운 패러다임을 제시합니다. 이후 많은 연구가 그래디언트 분포 분석을 출발점으로 삼게 됩니다.

#### (2) 동적 손실 함수 연구 촉진
정적 Focal Loss의 한계를 지적하고 동적 손실 함수의 필요성을 제시함으로써, 훈련 상태에 적응하는 손실 함수 연구가 활발해졌습니다.

#### (3) 앵커-프리(Anchor-free) 검출기와의 결합
FCOS, CenterNet 등 이후 등장한 앵커-프리 검출기들도 유사한 클래스 불균형 문제를 겪으며, GHM 아이디어의 적용이 논의됩니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 ATSS (Adaptive Training Sample Selection, CVPR 2020)

**Zhang et al., "Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection," CVPR 2020**

| 비교 항목 | GHM | ATSS |
|---------|-----|------|
| 핵심 접근 | 그래디언트 밀도 기반 재가중치 | IoU 통계 기반 동적 앵커 선택 |
| 문제 | 그래디언트 불균형 | 앵커 선택 기준 모호성 |
| 적응성 | 동적 (매 배치 재계산) | 정적 (규칙 기반) |
| 추가 파라미터 | 최소 ($M$, $\alpha$) | 없음 |

GHM이 **손실 함수 수준**에서 불균형을 해결한다면, ATSS는 **훈련 샘플 선택 수준**에서 해결합니다.

### 5.2 VFL / VariFocalNet (CVPR 2021)

**Zhang et al., "VarifocalNet: An IoU-aware Dense Object Detector," CVPR 2021**

**Varifocal Loss:**

$$VFL(p, q) = \begin{cases} -q(q\log(p) + (1-q)\log(1-p)) & q > 0 \\ -\alpha p^\gamma \log(1-p) & q = 0 \end{cases}$$

| 비교 항목 | GHM-C | Varifocal Loss |
|---------|-------|----------------|
| 타겟 | 이진 레이블 | IoU-aware soft label |
| 불균형 처리 | 그래디언트 밀도 | 비대칭 포커싱 |
| 정확도 | 41.6 (COCO) | 더 높은 mAP 달성 |

VFL은 GHM의 아이디어를 발전시켜 **예측 품질(IoU)을 레이블에 통합**하는 방향으로 진화합니다.

### 5.3 GFL (Generalized Focal Loss, NeurIPS 2020)

**Li et al., "Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection," NeurIPS 2020**

$$\mathcal{L}_{QFL}(\sigma) = -|y - \sigma|^\beta ((1-y)\log(1-\sigma) + y\log(\sigma))$$

GFL은 GHM처럼 **연속적인 레이블**을 다루는 방향으로 발전하며, 품질 점수 예측을 통합합니다. GHM의 그래디언트 조화 개념이 더 정교한 레이블 표현과 결합됩니다.

### 5.4 OTA (Optimal Transport Assignment, CVPR 2021)

**Ge et al., "OTA: Optimal Transport Assignment for Object Detection," CVPR 2021**

최적 수송 이론(Optimal Transport Theory)을 활용해 전역적으로 최적의 앵커-GT 매핑을 찾습니다. GHM이 **손실 수준**에서 처리하는 것을, OTA는 **할당 수준**에서 전역적으로 최적화합니다.

### 5.5 TOOD (Task-aligned One-stage Object Detection, ICCV 2021)

**Feng et al., "TOOD: Task-Aligned One-Stage Object Detection," ICCV 2021**

분류와 회귀의 불일치(misalignment)를 해결하는 연구로, GHM이 각 브랜치를 독립적으로 처리하는 것과 달리, **두 태스크를 통합적으로 정렬**합니다.

### 비교 요약 테이블

| 연구 | 연도 | 핵심 아이디어 | GHM 대비 관계 |
|------|------|-------------|-------------|
| GHM | 2018 | 그래디언트 밀도 기반 재가중치 | 기준 |
| ATSS | 2020 | 통계 기반 동적 샘플 선택 | 샘플 선택으로 확장 |
| GFL | 2020 | 연속 레이블 + 분포 예측 | 레이블 표현으로 확장 |
| VFL | 2021 | IoU-aware 비대칭 손실 | 비대칭 처리로 발전 |
| OTA | 2021 | 전역 최적 할당 | 전역 최적화로 발전 |
| TOOD | 2021 | 태스크 정렬 | 다중 태스크 통합으로 발전 |

---

## 6. 앞으로 연구 시 고려할 점

### 6.1 최적 그래디언트 분포 정의 문제
논문 자체가 인정하듯, **균등 분포가 최적인가**에 대한 이론적 근거가 부족합니다. 태스크와 데이터셋에 따라 다른 목표 분포가 필요할 수 있습니다.

> **연구 방향**: 정보 이론, 최적 제어 이론 등을 활용한 최적 그래디언트 분포의 이론적 도출

### 6.2 미니배치 크기 의존성
GHM의 통계는 미니배치에 의존하므로, 작은 배치 크기에서 통계적 신뢰성이 떨어집니다.

> **연구 방향**: 배치 크기에 무관한 통계 추정 방법 또는 Cross-batch 통계 활용

### 6.3 앵커-프리 검출기와의 통합
현재 GHM은 앵커 기반 검출기를 전제로 설계되었으나, FCOS, CenterNet 등에서의 적용 방법이 명확하지 않습니다.

> **연구 방향**: 앵커-프리 프레임워크에서의 그래디언트 밀도 재정의

### 6.4 Transformer 기반 검출기와의 결합
DETR, Deformable DETR 등 Transformer 기반 검출기는 이분 매칭(bipartite matching)을 사용하여 불균형 문제를 다르게 접근합니다. GHM의 아이디어를 이 맥락에서 재해석할 필요가 있습니다.

> **연구 방향**: Attention 가중치와 그래디언트 밀도의 관계 분석

### 6.5 다른 도메인으로의 일반화 검증
의료 영상, 위성 영상 등 클래스 불균형이 심각한 도메인에서 GHM의 효과를 체계적으로 검증하는 연구가 필요합니다.

### 6.6 Self-supervised / Semi-supervised 학습과의 결합
레이블이 부족한 환경에서 GHM의 적용 가능성 탐색이 필요합니다.

---

## 참고 자료

### 논문 (본문에서 직접 인용 및 분석에 활용)

1. **Buyu Li, Yu Liu, Xiaogang Wang**, "Gradient Harmonized Single-stage Detector," *AAAI 2019*, arXiv:1811.05181v1 [cs.CV], 13 Nov 2018.

2. **Tsung-Yi Lin et al.**, "Focal Loss for Dense Object Detection," *ICCV 2017*.

3. **Shrivastava et al.**, "Training Region-based Object Detectors with Online Hard Example Mining," *CVPR 2016*.

4. **Ren et al.**, "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks," *NeurIPS 2015*.

5. **Lin et al.**, "Feature Pyramid Networks for Object Detection," *CVPR 2017*.

6. **He et al.**, "Mask R-CNN," *ICCV 2017*.

### 2020년 이후 비교 연구 (제목 및 출처)

7. **Zhang et al.**, "Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection (ATSS)," *CVPR 2020*.

8. **Li et al.**, "Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection (GFL)," *NeurIPS 2020*.

9. **Zhang et al.**, "VarifocalNet: An IoU-aware Dense Object Detector (VFL)," *CVPR 2021*.

10. **Ge et al.**, "OTA: Optimal Transport Assignment for Object Detection," *CVPR 2021*.

11. **Feng et al.**, "TOOD: Task-Aligned One-Stage Object Detection," *ICCV 2021*.

> **정확도 참고**: 2020년 이후 연구 비교 분석 부분(섹션 5)은 각 논문의 제목과 개요 수준에서의 비교이며, 세부 수치는 각 원본 논문을 직접 확인하시기를 권장합니다.
