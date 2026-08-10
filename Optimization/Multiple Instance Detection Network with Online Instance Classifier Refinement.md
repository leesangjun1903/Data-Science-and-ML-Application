# Multiple Instance Detection Network with Online Instance Classifier Refinement

> **참고 자료**
> - Tang, P., Wang, X., Bai, X., & Liu, W. (2017). *Multiple Instance Detection Network with Online Instance Classifier Refinement*. arXiv:1704.00138v1.
> - Bilen, H., & Vedaldi, A. (2016). *Weakly Supervised Deep Detection Networks*. CVPR.
> - Kantorov, V., et al. (2016). *ContextLocNet*. ECCV.

---

## 1. Executive Summary (10문장 이내)

이 논문은 이미지 수준의 레이블(image-level label)만을 사용하는 **약지도 객체 탐지(Weakly Supervised Object Detection, WSOD)** 문제를 해결하기 위한 새로운 프레임워크를 제안한다.  
기존 MIL 기반 방법들은 객체의 일부분만 탐지하는 discriminative part 문제를 가지고 있었다.  
저자들은 이를 해결하기 위해 **Multiple Instance Detection Network(MIDN)**와 **Online Instance Classifier Refinement(OICR)** 알고리즘을 결합한 단일 end-to-end 네트워크를 설계하였다.  
OICR은 각 SGD 순전파(forward pass) 이후 상위 점수 proposal과 공간적으로 겹치는 인접 proposal에 레이블을 전파하여 인스턴스 분류기를 온라인으로 정제한다.  
초기 학습 시 노이즈가 많은 레이블 문제를 해결하기 위해 proposal 점수를 가중치로 사용하는 **가중 손실 함수(weighted loss)**를 도입하였다.  
다단계 정제 구조는 여러 스트림(stream)을 공유 표현(shared representation)으로 연결하며, 각 스트림이 다음 스트림의 감독 신호를 생성한다.  
PASCAL VOC 2007에서 **47.0% mAP**, **64.3% CorLoc**을 달성하여 당시 최고 성능을 크게 상회하였다.  
VOC 2012에서도 42.5% mAP, 65.6% CorLoc을 기록하였다. 대안적(alternative) 훈련 전략 대비 OICR은 훈련 시간을 단축하면서도 더 높은 성능을 보였다.  
다만 고양이, 개, 사람 등 비강체(non-rigid) 객체에 대한 성능은 여전히 제한적이며, 이는 향후 과제로 남아 있다.

---

### 1-1. 연구의 목적과 필요성

| 항목 | 내용 |
|------|------|
| **문제 배경** | 완전 지도 학습(fully supervised) 객체 탐지는 Bounding box 수준의 정밀 어노테이션이 필요하나, 이는 매우 비용이 크다 |
| **대안의 매력** | 이미지 레이블(이미지 태그)은 인터넷에서 손쉽게 획득 가능 |
| **기존 방법의 한계** | MIL 기반 WSOD는 객체 전체가 아닌 가장 두드러진 부분(discriminative part)만 탐지하는 경향이 있음 (p.1, Fig.1 left) |
| **연구 목적** | 이미지 레이블만으로 객체 전체를 정확히 탐지할 수 있는 end-to-end 네트워크 설계 |

> 💡 **약지도 학습(Weakly Supervised Learning)**: 완전한 레이블(예: bounding box 좌표) 없이 이미지 수준의 태그(예: "이 이미지에 고양이가 있다")만으로 모델을 학습하는 방법

---

## 2. 핵심 주장과 근거 표

| 핵심 주장 | 근거 | 위치 |
|-----------|------|------|
| MIL 기반 기본 탐지기는 객체 일부분만 탐지 | Fig. 1 left: 상위 점수 proposal A가 객체를 제대로 포함하지 못함 | p.1, Fig.1 |
| 공간적으로 겹치는 proposal은 유사한 레이블을 공유해야 함 | 인접 proposal이 더 큰 객체 영역을 포함할 가능성이 높음 | p.2 |
| OICR이 대안적 훈련보다 성능과 효율 모두 우수 | Fig. 4: OICR vs alternative 비교에서 OICR이 모든 정제 단계에서 높은 성능 | p.6, Fig.4 |
| 가중 손실이 비가중 손실보다 효과적 | Table 1: 가중 손실(37.9 mAP) vs 비가중 손실(32.8 mAP) | p.6, Table 1 |
| 다단계 정제가 성능을 점진적으로 향상 | Fig. 4: 정제 횟수 증가에 따른 mAP 향상 (0→1→2→3회) | p.6-7, Fig.4 |
| 제안 방법이 당시 SOTA 초월 | Table 2: OICR-Ens.+FRCNN 47.0% vs WSDDN-Ens. 39.3% | p.7, Table 2 |

---

### 2-1. 상세 설명

#### 해결하고자 하는 문제

기존 WSOD 방법(특히 WSDDN [4])은 이미지 분류 점수를 proposal 점수의 가중 합으로 계산하므로, **객체의 가장 두드러진 부분**만 높은 점수를 얻는다. 이로 인해 탐지 기준인 IoU > 0.5를 만족하지 못하는 경우가 많다. (p.1-2)

> 💡 **IoU(Intersection over Union)**: 예측 박스와 정답 박스의 교집합 넓이를 합집합 넓이로 나눈 값. 0~1 사이 값으로 1에 가까울수록 정확한 탐지.

#### 제안하는 방법 및 수식

**[Step 1] Multiple Instance Detection Network (MIDN) - 기본 인스턴스 분류기**

두 개의 병렬 FC 스트림을 통해 proposal 행렬 $\mathbf{x}^c, \mathbf{x}^d \in \mathbb{R}^{C \times |R|}$을 생성한다.

$$[\sigma(\mathbf{x}^c)]_{ij} = \frac{e^{x^c_{ij}}}{\sum_{k=1}^{C} e^{x^c_{kj}}}$$

$$[\sigma(\mathbf{x}^d)]_{ij} = \frac{e^{x^d_{ij}}}{\sum_{k=1}^{|R|} e^{x^d_{ik}}}$$

- $[\sigma(\mathbf{x}^c)]_{ij}$: proposal $j$가 클래스 $i$에 속할 확률 (클래스 방향 softmax)
- $[\sigma(\mathbf{x}^d)]_{ij}$: proposal $j$가 클래스 $i$ 이미지 분류에 기여하는 정규화된 가중치 (proposal 방향 softmax)

Proposal 점수 행렬:

$$\mathbf{x}^R = \sigma(\mathbf{x}^c) \odot \sigma(\mathbf{x}^d)$$

- $\odot$: 요소별 곱(element-wise product)

이미지의 클래스 $c$ 점수:

$$\phi_c = \sum_{r=1}^{|R|} x^R_{cr}$$

- $\phi_c$: 클래스 $c$에 대한 이미지 전체 점수
- $|R|$: 전체 proposal 수

기본 인스턴스 분류기 학습 손실 (이진 교차 엔트로피):

$$\mathbf{L}_b = -\sum_{c=1}^{C} \{y_c \log \phi_c + (1 - y_c) \log(1 - \phi_c)\} $$

- $y_c \in \{0, 1\}$: 이미지에 클래스 $c$ 존재 여부
- $C$: 전체 클래스 수

> 💡 **Softmax**: 여러 값을 0~1 사이의 확률로 변환하는 함수. 두 방향으로 softmax를 적용함으로써 클래스 확률과 proposal 기여도를 분리하여 계산.

---

**[Step 2] Online Instance Classifier Refinement (OICR)**

클래스 $c$의 최고 점수 proposal 선택:

$$j_c^{k-1} = \arg\max_{r} x^{R(k-1)}_{cr} $$

- $j_c^{k-1}$: $k-1$번째 단계에서 클래스 $c$의 가장 높은 점수를 받은 proposal 인덱스
- $x^{R(k-1)}_{cr}$: $k-1$번째 정제 단계에서 proposal $r$의 클래스 $c$ 점수

비가중 정제 손실:

$$\mathbf{L}^k_r = -\frac{1}{|R|} \sum_{r=1}^{|R|} \sum_{c=1}^{C+1} y^k_{cr} \log x^{Rk}_{cr} $$

- $y^k_{cr}$: $k$번째 정제에서 proposal $r$의 클래스 $c$ 레이블 (IoU 기반으로 자동 생성)
- $x^{Rk}_{cr}$: $k$번째 정제 분류기의 proposal $r$에 대한 클래스 $c$ 점수
- $C+1$번째 차원: 배경(background) 클래스

가중 정제 손실 (초기 학습 노이즈 완화):

$$\mathbf{L}^k_r = -\frac{1}{|R|} \sum_{r=1}^{|R|} \sum_{c=1}^{C+1} w^k_r y^k_{cr} \log x^{Rk}_{cr} $$

- $w^k_r$: proposal $r$의 손실 가중치 = 선택된 최고 점수 proposal의 점수 $x^{Rk}_{cj^k_c}$ (Algorithm 1의 11번 줄)

> 💡 **가중 손실의 직관**: 초기 학습에서는 분류기가 불안정하므로 최고 점수가 낮게 형성됨 → $w^k_r$ 작음 → 노이즈 레이블의 영향력 감소. 학습이 진행될수록 신뢰도 높은 proposal의 $w^k_r$가 커져 정확한 학습 신호 제공.

전체 네트워크 손실:

$$\mathbf{L} = \mathbf{L}_b + \sum_{k=1}^{K} \mathbf{L}^k_r $$

- $K$: 총 정제 횟수 (논문에서 $K=3$으로 설정)
- $\mathbf{L}_b$: MIDN 기본 손실
- $\mathbf{L}^k_r$: $k$번째 정제 단계 손실

---

#### 모델 구조 (Fig. 3 기준)

```
입력 이미지
    ↓
Conv Layers (VGG M 또는 VGG16)
    ↓
SPP Layer (고정 크기 feature 출력)
    ↓
Two FC Layers (proposal feature vector 생성)
    ↓
┌─────────────────────────────────────────┐
│ Stream 0: MIDN                          │
│ FC → Softmax(class) × Softmax(prop)    │
│ → image score → L_b                    │
├─────────────────────────────────────────┤
│ Stream 1: 1st Instance Classifier      │
│ FC → Softmax(class+1) → L_r^1         │
│ (supervision from Stream 0 output)     │
├─────────────────────────────────────────┤
│ Stream 2: 2nd Instance Classifier      │
│ → L_r^2 (supervision from Stream 1)   │
├─────────────────────────────────────────┤
│ Stream 3: 3rd Instance Classifier      │
│ → L_r^3 (supervision from Stream 2)   │
└─────────────────────────────────────────┘
    ↓
Total Loss = L_b + L_r^1 + L_r^2 + L_r^3
```

> 💡 **SPP(Spatial Pyramid Pooling)**: 다양한 크기의 입력 이미지/영역에서 고정된 크기의 feature를 추출하는 기법. 서로 다른 크기의 proposal을 동일한 FC 레이어에 입력할 수 있게 해줌.

---

#### 성능 향상 및 한계

| 구분 | 내용 |
|------|------|
| **성능 향상** | VOC 2007: 47.0% mAP (이전 SOTA 대비 약 7.7%p 향상) |
| **성능 향상** | VOC 2007 CorLoc: 64.3% (이전 SOTA 대비 약 8%p 향상) |
| **성능 향상** | VOC 2012: 42.5% mAP, 65.6% CorLoc |
| **한계 1** | 비강체 객체(고양이, 개, 사람)에서 부분 탐지 문제 지속 |
| **한계 2** | 인접 유사 객체가 있을 때 과대 박스(overlarge box) 생성 |
| **한계 3** | 공간적 중복만을 기준으로 레이블 전파 (시각적 유사도 미활용) |
| **한계 4** | Selective Search 기반 proposal에 의존 (약 2,000개/이미지) |

---

## 3. 주장별 페이지/Figure/Table 번호

| 주장 | 근거 위치 |
|------|-----------|
| 기존 WSOD의 discriminative part 문제 | p.1-2, **Fig. 1 (left)** |
| 공간 중복 proposal 레이블 전파 아이디어 | p.2, **Fig. 1 (right)** |
| 다단계 정제 시각화 | p.2, **Fig. 2** |
| 전체 네트워크 구조 | p.4, **Fig. 3** |
| 가중 손실의 필요성 | p.6, **Table 1** |
| OICR vs alternative 전략 비교 | p.6, **Fig. 4** |
| IoU threshold 민감도 분석 | p.6, **Fig. 5** |
| VOC 2007 mAP 비교 | p.7, **Table 2** |
| VOC 2007 CorLoc 비교 | p.7, **Table 3** |
| VOC 2012 결과 | p.7, **Table 4** |
| 성공/실패 사례 시각화 | p.8, **Fig. 6** |

---

## 4. 저자 직접 보고 결과 vs. 해석 분리

### 4-1. 저자가 직접 보고한 결과

**연구 주제**: 이미지 레이블만을 이용한 약지도 객체 탐지에서 MIL과 온라인 인스턴스 분류기 정제를 결합한 단일 end-to-end 네트워크 제안 (Abstract)

**방법**: MIDN을 기반으로 $K=3$번의 정제 스트림을 추가하며, 각 스트림은 이전 스트림의 출력을 감독 신호로 사용. IoU threshold $I_t = 0.5$ 사용. (p.4-5, Section 3)

**결과** (저자 직접 보고):
- OICR-Ens.+FRCNN: VOC 2007 **47.0% mAP**, **64.3% CorLoc** (p.3, Table 2, 3)
- OICR-Ens.+FRCNN: VOC 2012 **42.5% mAP**, **65.6% CorLoc** (Table 4)
- 가중 손실 vs 비가중 손실: **37.9% vs 32.8%** mAP (Table 1)
- 단 1회 정제만으로도 mAP **29.5 → 35.6%** 향상 (p.6)

---

### 4-2. 분석자 해석

> ⚠️ **이하 내용은 논문에 직접 명시되지 않은 분석자의 해석입니다.**

1. **가중 손실의 커리큘럼 학습 특성**: $w^k_r$을 proposal 점수로 설정하는 방식은 학습 커리큘럼(curriculum learning)과 유사한 효과를 낸다. 초기에는 쉬운 샘플만 강하게 학습하고, 어려운 샘플은 점진적으로 반영하는 구조이다.

2. **정제 횟수 수렴 현상**: Fig. 4에서 2→3회 정제 시 성능 향상이 미미한 것은 네트워크가 수렴하여 3회의 감독 신호가 2회와 거의 동일해지기 때문으로 해석된다. 이는 정제 횟수에 대한 최적값이 데이터셋과 모델 복잡도에 따라 다를 수 있음을 시사한다.

3. **FRCNN 후처리의 효과**: OICR-Ens.(42.0%) → OICR-Ens.+FRCNN(47.0%)로의 5%p 향상은 의사 레이블(pseudo ground truth) 기반 지도 학습의 효과를 보여주나, 이는 완전 약지도 방식이 아닌 준지도 학습(semi-supervised)에 가깝다는 점에서 공정한 비교가 어렵다.

---

## 5. 통계적 취약점 및 비교 불가능한 수치

| 항목 | 문제점 |
|------|--------|
| **단일 시드 실험** | 랜덤 시드(random seed)에 대한 반복 실험 없음. 표준편차/신뢰구간 미보고. |
| **OICR-Ens. vs WSDDN-Ens.** | WSDDN-Ens.는 F/M/16 3개 모델 앙상블, OICR-Ens.는 M+16 2개 모델 앙상블 → **모델 수와 구성이 달라 직접 비교 불공정** |
| **OICR-Ens.+FRCNN 비교** | FRCNN을 추가 학습한 반지도 방식을 완전 약지도 방법들과 동일 테이블에 비교 → **감독 정보량이 달라 비교 불공정** |
| **"0 time" MIDN 성능 차이** | 저자들은 자신들의 0회 정제 결과(~29.5%)가 WSDDN(30.9%)보다 낮은 이유를 "플랫폼 차이"로만 설명. 구체적 원인 미분석. |
| **VOC 2012 일부 클래스** | "person" 클래스에서 OICR-Ens.+FRCNN이 1.0% mAP로 매우 낮음. 이상값(outlier) 가능성이 있으나 별도 분석 없음. |
| **IoU threshold 실험** | VGG M 모델에서만 진행. VGG16에 대한 검증 없음. |

---

## 6. 문서가 답하지 않는 질문

1. **다른 데이터셋 일반화**: COCO, Open Images 등 더 복잡한 데이터셋에서도 동일한 성능 향상이 나타나는가?

2. **proposal 수의 영향**: Selective Search의 약 2,000개 대신 더 많거나 적은 proposal을 사용할 경우 성능은 어떻게 변하는가?

3. **정제 스트림 간 상관관계**: 각 정제 스트림이 실제로 학습하는 특징(feature)의 차이는 무엇인가? (해석 가능성 부재)

4. **계산 비용**: 훈련 시간 단축을 "많이(a lot)" 표현했으나 구체적인 수치(시간, FLOPs 등)가 없다.

5. **proposal 방법 의존성**: Selective Search 대신 RPN(Region Proposal Network) 등 다른 proposal 방법 사용 시 성능 변화는?

6. **클래스 불균형 처리**: 20개 클래스 중 일부(bottle, chair 등)는 성능이 낮은데 클래스 불균형에 대한 대처 방안이 없다.

7. **IoU 기반 레이블 전파의 한계**: 시각적으로 유사하지 않지만 공간적으로 겹치는 다른 객체에 잘못된 레이블이 전파되는 경우에 대한 정량적 분석이 없다.

8. **$K=3$ 최적값의 이론적 근거**: 왜 3번 정제가 최적인지에 대한 이론적 설명이 없다.

---

## 7. 가장 중요한 그림 5개 해석

### Fig. 1 (p.1) - 정제 전후 비교

| 항목 | 정제 전 (left) | 정제 후 (right) |
|------|---------------|----------------|
| 최고 점수 proposal | A (0.21) - 너무 작음, 객체 일부만 포함 | D (0.71) - 객체 전체 포함 |
| 다른 proposal | B(0.07), C(0.04) 유사하게 낮은 점수 | 올바른 분리(A: 0.11, B: 0.19) |
| 해석 | 기존 MIDN만으로는 discriminative part 문제 발생 | OICR로 레이블이 A→B,C→D로 전파되어 전체 객체 탐지 |

**분석자 해석**: 정제 후 점수 분포가 훨씬 명확하게 분리되어(D: 0.71 vs 나머지) 분류기의 판별력(discriminability)이 크게 향상되었음을 시각적으로 증명한다.

---

### Fig. 2 (p.2) - 다단계 정제 시각화

각 행이 정제 단계(0~3)를 나타내며, 단계가 증가할수록:
- 초기(0단계): IoU가 낮은 빨간 박스(부분 탐지)
- 이후 단계: 점차 IoU가 높은 녹색 박스(전체 객체 포함)

**분석자 해석**: 레이블 전파가 실제로 공간적으로 확장되며 작동함을 직관적으로 보여준다. 그러나 각 행에서 동일 이미지의 다른 단계가 아닌 여러 이미지를 보여주므로, 단일 이미지에서의 점진적 개선 과정을 직접 확인하기 어렵다는 한계가 있다.

---

### Fig. 3 (p.4) - 전체 네트워크 아키텍처

MIDN(기본 분류기)과 $K$개의 정제 스트림이 공유 proposal feature에서 분기하는 구조를 보여준다.

**핵심 설계 결정**:
- **공유 표현(shared representation)**: 모든 스트림이 동일한 proposal feature vector를 입력으로 받아 계산 효율성과 표현 일관성 동시 확보
- **직렬 감독(sequential supervision)**: $k$번째 스트림이 $k-1$번째의 출력을 레이블로 사용 → 오류 전파(error propagation) 가능성 존재

**분석자 해석**: 이 구조는 knowledge distillation과 유사한 메커니즘으로 볼 수 있으며, 각 스트림이 이전 스트림의 '교사' 역할을 한다.

---

### Fig. 4 (p.6) - 정제 횟수 및 훈련 전략 비교

| 관찰 | 의미 |
|------|------|
| OICR > alternative (모든 정제 횟수에서) | 공유 표현의 효과가 실증됨 |
| 0회 → 1회 정제 시 가장 큰 향상 | 첫 번째 정제가 핵심 기여 요소 |
| 2회 → 3회 향상 미미 | 성능 포화(saturation) 현상 |

**분석자 해석**: alternative 방식도 정제 횟수에 따라 개선되지만 OICR에 비해 지속적으로 낮다. 이는 공유 표현을 통한 정보 공유가 단순한 sequential 훈련보다 효과적임을 시사한다.

---

### Fig. 6 (p.8) - 성공/실패 사례

| 성공 (녹색) | 실패 (빨간) |
|------------|------------|
| bicycle, bus, motorbike 등 강체(rigid) 객체 | cat, dog, person 등 비강체(non-rigid) 객체 |
| 다양한 크기와 비율에서 강건함 | 객체의 머리 등 가장 두드러진 부분만 탐지 |

**분석자 해석**: 실패 패턴이 두 가지로 구분된다: (1) 인접한 유사 객체를 함께 포함하는 과대 박스, (2) 비강체 객체의 부분 탐지. 이는 공간적 중복만을 기준으로 하는 레이블 전파의 근본적 한계를 보여준다. 시각적 유사도(visual similarity)를 추가하면 이를 개선할 수 있음을 저자들도 인정한다.

---

## 8. 결론, 시사점 및 후속 연구

### 8-1. 저자가 제시한 시사점 및 후속 연구 계획

**저자 제시 시사점** (p.8, Section 5):
- MIDN + 다단계 인스턴스 분류기를 단일 네트워크로 통합하는 프레임워크의 유효성 입증
- OICR 알고리즘이 다른 약지도 시각 학습 태스크에도 적용 가능함

**저자 제시 후속 연구 계획** (p.8):
- 공간적 중복 이외에 **인스턴스 시각적 유사도(instance visual similarity)** 등 추가 단서(cue) 활용

---

### 8-1. 모델의 일반화 성능 향상 가능성

| 개선 방향 | 상세 내용 |
|-----------|-----------|
| **다양한 backbone 적용** | VGG 외 ResNet, EfficientNet 등 강력한 backbone 사용 시 일반화 성능 향상 기대 |
| **시각적 유사도 기반 레이블 전파** | 공간적 중복만이 아닌 feature 공간에서의 유사도를 추가 기준으로 활용하면 비강체 객체 처리 개선 |
| **더 다양한 proposal 활용** | Selective Search 외 EdgeBoxes, RPN 등 다양한 proposal 방법 앙상블 |
| **도메인 적응** | 의료 영상, 위성 영상 등 다른 도메인으로의 전이 가능성: 이미지 레이블 수집이 쉬운 분야에서 즉시 활용 가능 |
| **다중 레이블 처리** | 현재 단순 이진(존재/부재) 레이블만 사용; 계층적 레이블이나 속성 레이블 활용 가능성 |
| **데이터 증강 강화** | 현재 5개 스케일 + 수평 반전만 사용; 더 적극적인 증강으로 일반화 개선 가능 |

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의**: 이하 2020년 이후 연구 정보는 저자의 학습 데이터 기반 지식이며, 최신 논문의 정확한 수치는 직접 확인이 필요합니다.

| 논문 | 연도 | 핵심 아이디어 | OICR 대비 개선 |
|------|------|--------------|---------------|
| **PCL (Proposal Cluster Learning)** [Tang et al., TPAMI 2020] | 2020 | 클러스터링 기반 proposal 그룹화로 더 정확한 레이블 생성 | 공간적 중복 대신 의미적 클러스터링 사용 |
| **WSOD2** [Zeng et al., ICCV 2019] | 2019 | 위에서 아래로(top-down) 및 아래에서 위로(bottom-up) 정보 결합 | 전역-지역 정보 동시 활용 |
| **C-MIL** [Wan et al., CVPR 2019] | 2019 | 연속화된 MIL 목적함수로 최적화 안정성 향상 | 비볼록(non-convex) 최적화 문제 완화 |
| **SLV** [Chen et al., CVPR 2020] | 2020 | Self-supervised learning과 WSOD 결합 | 레이블 없는 사전학습 활용 |
| **Transformer 기반 WSOD** | 2021+ | Self-attention으로 전역 맥락 정보 캡처 | OICR의 지역적 공간 관계만 이용하는 한계 극복 |

**OICR이 후속 연구에 미친 영향**:

1. **온라인 레이블 생성 패러다임 확립**: OICR의 SGD 내 실시간 레이블 생성 아이디어는 이후 많은 WSOD 연구의 기본 패러다임이 되었다.

2. **다단계 정제의 표준화**: 단일 단계가 아닌 다단계 정제 구조는 후속 연구들이 널리 채택하였다.

3. **의사 레이블(Pseudo Label) 연구의 촉진**: OICR의 온라인 레이블 전파 방식은 준지도 학습 및 자기 훈련(self-training) 연구에도 영향을 주었다.

**앞으로 연구 시 고려할 점**:

| 고려사항 | 세부 내용 |
|----------|-----------|
| **오류 누적(error accumulation)** | 정제 스트림 간 레이블 오류가 누적될 수 있음 → 강건한 레이블 정정 메커니즘 필요 |
| **대규모 데이터셋 검증** | PASCAL VOC(20클래스) 외 COCO(80클래스) 수준의 대규모 검증 필요 |
| **비전 트랜스포머(ViT) 통합** | ViT의 attention map을 레이블 전파에 활용하면 시각적 유사도 기반 정제 가능 |
| **계산 효율성** | Proposal-based 방법의 계산 비용이 크므로 Anchor-free 방법과의 결합 고려 |
| **클래스 불균형 대처** | 희귀 클래스에 대한 별도 가중치 부여 전략 필요 |
| **Foundation Model 활용** | CLIP, SAM 등 대규모 사전학습 모델의 zero-shot 능력과 OICR 아이디어 결합 가능성 |

---

## 용어 요약 정리

> 본문 전체에서 등장한 주요 용어들을 한곳에 모았습니다.

| 용어 | 설명 |
|------|------|
| **WSOD** | Weakly Supervised Object Detection. 이미지 레이블만으로 객체를 탐지하는 방법 |
| **MIL** | Multiple Instance Learning. 여러 후보(instance)가 담긴 bag에서 양성 instance를 찾는 학습 방식 |
| **MIDN** | Multiple Instance Detection Network. MIL을 CNN으로 구현한 기본 탐지 네트워크 |
| **OICR** | Online Instance Classifier Refinement. SGD 내에서 실시간으로 분류기를 정제하는 알고리즘 |
| **mAP** | mean Average Precision. 모든 클래스의 AP 평균. 탐지 성능의 주요 지표 |
| **CorLoc** | Correct Localization. 훈련 세트에서 객체 위치를 올바르게 찾은 비율 |
| **Selective Search** | 이미지에서 객체 후보 영역(proposal)을 생성하는 전통적인 알고리즘 |
| **Pseudo Ground Truth** | 약지도 모델의 예측을 마치 정답 레이블처럼 사용하여 완전 지도 모델을 추가 학습하는 기법 |
| **Dilated Convolution** | 커널 사이에 빈 공간(dilation)을 두어 넓은 수용 영역(receptive field)을 갖는 합성곱 연산 |
| **End-to-End Trainable** | 전처리-특징추출-분류의 전 과정을 하나의 네트워크로 역전파를 통해 동시에 학습하는 방식 |
