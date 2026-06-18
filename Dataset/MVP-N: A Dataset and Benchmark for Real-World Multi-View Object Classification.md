# MVP-N: A Dataset and Benchmark for Real-World Multi-View Object Classification 

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

MVP-N은 기존 멀티뷰 객체 분류 데이터셋의 세 가지 핵심 한계를 극복하기 위해 제안된 **실세계 기반의 정밀 분류(fine-grained) 멀티뷰 데이터셋**이다:

1. 기존 데이터셋은 **합성(synthetic) 객체**와 **거친 분류(coarse-grained)** 중심
2. **검증 세트(validation split)** 부재로 테스트 세트에서 하이퍼파라미터 튜닝
3. **뷰 수준(view-level) 정보량 어노테이션** 부재로 분석의 한계

### 주요 기여

| 기여 항목 | 내용 |
|-----------|------|
| **데이터셋 제안** | 44개 실제 소매 제품, 16k 실제 촬영 뷰, 9k 멀티뷰 세트, HPIQ 어노테이션 포함 |
| **방법론 정리** | 2015~2022년 39개 멀티뷰 feature aggregation 방법을 P1/P2/P3 기준으로 정리 |
| **새로운 평가 지표** | HPIQ 기반의 MCDU(Mean Confidence Difference between predictions and Uninformative views) 제안 |
| **벤치마크 실험** | 4개 feature aggregation 방법 + 12개 soft label 방법 벤치마킹 |

---

## 2. 해결 문제 · 제안 방법 · 모델 구조 · 성능 · 한계

### 2.1 해결하고자 하는 문제

실세계 멀티뷰 분류에서의 세 가지 현실적 요구사항(practicability properties):

- **P1**: 훈련/테스트 단계 모두에서 **임의 개수의 입력 뷰** 허용
- **P2**: 카메라 위치나 상대적 포즈 같은 **공간 관계 정보 불필요**
- **P3**: **임의 시점**에서 획득된 뷰, 순서 무관(permutation-invariant)

또한, 유사 외관을 가진 소매 제품들(예: 동일 브랜드 다른 맛) 사이의 **클래스 간 뷰 유사성(inter-class view similarity)**으로 인해 발생하는 **멀티뷰 레이블 노이즈(multi-view label noise)** 문제를 다룬다.

---

### 2.2 제안하는 방법 및 수식

#### (1) HPIQ (Human-Perceived Information Quantity) 어노테이션

각 뷰에 대해 인간 판단 기반으로 세 등급으로 분류:

- **Sufficiently informative**: 해당 뷰만으로 정확한 분류 가능
- **Less informative**: 부분적 정보 포함, 단독 분류 불보장
- **Uninformative**: 추가 뷰 없이는 분류 불가

품질 관리 과정에서 "Less informative"는 필터링하고, 나머지는 3명 추가 어노테이션으로 합의 도출.

#### (2) MCDU (Mean Confidence Difference between predictions and Uninformative views) — 핵심 신규 지표

$$MCDU = \frac{1}{N} \sum_{i=1}^{N} \left(\max(p_i) - p_i(y_i)\right)$$

- $N$: 검증/테스트 세트의 모든 uninformative 뷰에서 틀린 예측의 수
- $p_i$: 예측된 확률 분포 벡터
- $y_i$: 정답 클래스 레이블
- $\max(p_i)$: 예측 중 가장 높은 확률값

> **해석**: MCDU가 높을수록, 모델이 uninformative 뷰에서 잘못된 예측을 **높은 자신감으로** 수행하여 전체 멀티뷰 성능을 저하시킨다. 따라서 **낮을수록 좋다(↓)**.

#### (3) 멀티뷰 정확도 계산 (Mean Rule)

최종 멀티뷰 예측은 평균 규칙(mean rule)으로 뷰별 예측을 결합:

$$\hat{y} = \arg\max_c \frac{1}{V} \sum_{v=1}^{V} p_v(c)$$

여기서 $V$는 뷰의 수, $p_v(c)$는 $v$번째 뷰에서 클래스 $c$에 대한 예측 확률.

#### (4) HPIQ 소프트 레이블 구성

- **Informative 뷰**: 하드 레이블(one-hot) 그대로 사용
- **Uninformative 뷰**: 동일 그룹 클래스에 대한 **균등 분포(uniform distribution)**를 소프트 레이블로 사용

$$\tilde{y}_i = \begin{cases} \text{one-hot}(y_i) & \text{if informative} \\ \frac{1}{|\mathcal{G}|} \mathbf{1}_{\mathcal{G}} & \text{if uninformative} \end{cases}$$

여기서 $\mathcal{G}$는 해당 객체가 속한 유사 외관 그룹, $|\mathcal{G}|$는 그룹 내 클래스 수.

---

### 2.3 모델 구조

논문에서 벤치마킹한 방법들의 전체 파이프라인:

```
Arbitrary Views
     ↓
Feature Extraction (CNN/ViT: ResNet-18 등)
     ↓
[Two-stage] Feature Aggregation (FA)
  ├── MVCNN-new: element-wise max pooling
  ├── GVCNN: grouping + intra-group pooling
  ├── DAN: self-attention mechanism
  └── CVR: optimal transport + transformer encoder
     ↓
[Three-stage (Hypergraph)] HL/HNN → FC → Prediction
[Three-stage (Part)] RPN → Top-K regions → FA → FC → Prediction
     ↓
[Soft label] View-level predictions → Score Fusion (SF)
```

#### 훈련 세부사항

| 항목 | 설정값 |
|------|--------|
| Backbone | ResNet-18 (ImageNet 사전학습) |
| Optimizer | SGD (momentum=0.9, weight decay= $10^{-3}$ ) |
| Single-view 학습 | 30 epochs, batch=128, lr= $10^{-2}$, 10 epoch마다 절반 감소 |
| Multi-view 학습 | 50 epochs, batch=32, lr= $10^{-3}$, cosine annealing |
| 실험 반복 | 5회 (다양한 random seed) |

---

### 2.4 성능 결과

#### Feature Aggregation 방법 비교 (Table 4)

| Method | MVA (Test, %) ↑ | MCC ↑ | MCW ↓ | Model Size (M) ↓ |
|--------|-----------------|--------|--------|-----------------|
| MVCNN-new | $89.35 \pm 1.21$ | **0.8792** | 0.6552 | **11.20** |
| GVCNN | $85.42 \pm 1.37$ | 0.8267 | **0.6055** | 24.04 |
| DAN | $\mathbf{91.61 \pm 0.94}$ | 0.8602 | 0.6211 | 17.50 |
| CVR | $79.99 \pm 2.52$ | 0.8339 | 0.6457 | 34.38 |

**DAN이 최고 MVA(91.61%)** 달성: self-attention 메커니즘이 informative 뷰의 가중치를 높여 성능 향상.

#### Soft Label 방법 비교 (Table 3, Test 기준)

| Method | SVAI (%) ↑ | MCDU ↓ | MVA (%) ↑ |
|--------|-----------|--------|-----------|
| CE (baseline) | 99.15 | 0.3892 | 83.37 |
| SAT | 99.00 | **0.2145** | $87.37 \pm 1.15$ |
| KD | **99.49** | 0.3737 | $86.77 \pm 1.24$ |
| SEAL | 98.41 | 0.1326 | $86.42 \pm 0.74$ |
| **HPIQ** | 99.68 | **0.1481** | $\mathbf{94.36 \pm 0.56}$ |

**HPIQ가 MVA 94.36%로 최고 성능** — 그러나 SVA(단일 뷰 전체 정확도)는 63.31%로 최저. 이는 uninformative 뷰의 낮은 정확도가 멀티뷰 성능에 직접 영향을 주지 않음을 보여준다.

---

### 2.5 한계점

1. **소규모 데이터셋**: 44개 카테고리, 9k 멀티뷰 세트는 대규모 실세계 응용에 비해 제한적
2. **단일 도메인**: 소매 제품에만 한정 → 다른 도메인(예: 공업 부품, 의료 기기)으로의 직접 확장 불가
3. **Backbone 제한**: ResNet-18만 사용 → 최신 대형 모델(ViT, CLIP 등) 검증 없음
4. **HPIQ 어노테이션 비용**: 이미지당 평균 45초 소요, 확장 비용이 큼
5. **뷰 수 제한**: 최대 6개 뷰로 구성 → 더 많은 뷰 시나리오 미검증
6. **Hypergraph/Part 방법 미벤치마킹**: 세 가지 단계 방법 중 two-stage만 집중 비교

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 HPIQ 기반 일반화 분석

HPIQ 어노테이션을 활용하면 **단순 정확도를 넘어서 인간 인지와의 일치도**를 평가할 수 있다. 실험 결과:

- **SVAI ≈ 99%**: 모든 방법이 informative 뷰에서는 높은 정확도 달성
- **SVA vs SVAI 갭**: 단일 뷰 오류의 대부분은 uninformative 뷰에서 발생

이는 모델이 **informative 뷰를 올바르게 활용하는 능력**이 일반화 성능의 핵심임을 시사한다.

### 3.2 뷰 수에 따른 일반화 (Table 5 분석)

$$\text{MVA}(V) = f(\text{비율}_{\text{uninformative}}, \text{aggregation method})$$

| 관찰 | 의미 |
|------|------|
| uninformative 뷰 증가 → soft label 방법 MVA 급감 | soft label은 uninformative 비율에 민감 |
| 2뷰에서는 soft label 방법 일부가 DAN 초과 | 적은 뷰에서는 soft label이 효과적 |
| 6뷰에서는 DAN이 모든 soft label 방법 초과 | 다뷰 환경에서는 feature aggregation이 우세 |

**일반화 관점**: uninformative 뷰 비율이 낮은 실제 환경에서는 soft label 방법이, 높은 환경에서는 DAN과 같은 attention 기반 feature aggregation이 더 강건한 일반화 성능을 제공.

### 3.3 SAT (Self-Adaptive Training)의 일반화 메커니즘

SAT는 아래와 같이 적응적으로 soft label을 구성:

$$\tilde{y}_i^{(t)} = \alpha \cdot \hat{p}_i^{(t)} + (1-\alpha) \cdot y_i^{\text{hard}}$$

여기서 $\hat{p}_i^{(t)}$는 $t$ 시점의 모델 예측, $\alpha$는 학습 진행에 따라 증가하는 가중치. 이 방식이 **낮은 MCDU(0.2145)**와 **높은 MVA(87.37%)**를 동시에 달성하여 일반화 성능 향상에 기여.

### 3.4 DAN의 Self-Attention 기반 일반화

DAN의 self-attention은 다음과 같이 뷰별 가중치를 동적으로 계산:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

이를 통해 informative 뷰에 더 높은 가중치를 부여함으로써, **uninformative 뷰의 노이즈 영향을 동적으로 억제** → 일반화 성능 향상의 핵심 메커니즘.

### 3.5 Confusion Matrix에서 드러난 일반화 패턴

혼동 행렬 분석(Figure 5)에서:
- 대부분의 오류가 **동일 그룹(유사 외관 객체)** 내에서 발생
- 이는 모델이 **의미적으로 합리적인 오류**를 범함 → 인간의 판단과 유사
- 향후 **그룹 내 세밀한 특징 학습**이 일반화 개선의 핵심 방향

---

## 4. 앞으로의 연구에 미치는 영향과 고려할 점

### 4.1 연구에 미치는 영향

#### (1) 실세계 평가 표준 정립
MVP-N은 기존 합성 데이터셋(ModelNet40 등)의 한계를 극복하는 **실세계 벤치마크 표준**을 제시. 향후 실용적 멀티뷰 방법 개발의 기준점이 될 것.

#### (2) 노이즈 레이블 학습의 새로운 관점
멀티뷰 환경에서의 레이블 노이즈는 기존 단일 이미지 노이즈 학습과 **근본적으로 다른 구조**를 가짐. HPIQ 기반의 뷰-레벨 노이즈 분석은 노이즈 학습 연구의 새 방향 제시.

#### (3) 평가 지표 혁신
MCDU는 단순 정확도로는 포착하기 어려운 **모델 자신감의 적절성**을 측정. 이는 단일 이미지 분류에서도 응용 가능 (informative/uninformative 분리 평가).

#### (4) 실응용 연구 촉진
로봇 그래스핑, 소매 자동 결제, 제조 결함 탐지 등 **엣지 디바이스 기반 다중 카메라 시스템** 연구에 직접적 기여.

---

### 4.2 향후 연구 시 고려할 점

#### (1) 더 강력한 Backbone 활용
논문은 ResNet-18만 사용했으나, 향후에는:
- **Vision Transformer (ViT)**, **CLIP** 등 대형 사전학습 모델 적용
- **DeiT** [Hugo Touvron et al., ICML 2021]의 경량 ViT가 효율성 측면에서 유망

#### (2) 자동 HPIQ 예측 모델
현재 HPIQ 어노테이션은 인간 수작업에 의존 → **자동 정보량 추정 모델** 개발 필요:

$$\hat{q}_v = f_\theta(\mathbf{x}_v) \in [0, 1]$$

이를 feature aggregation의 가중치로 직접 활용 가능.

#### (3) Informative View 자동 선택 (View Selection)
뷰 수가 증가할수록 uninformative 뷰의 비율도 증가 → **적응형 뷰 선택 메커니즘** 필요:

$$V^* = \arg\max_{S \subseteq \mathcal{V}, |S|=k} \text{MVA}(S)$$

#### (4) 도메인 일반화 (Domain Generalization)
훈련(Collection A)과 테스트(Collection A+B) 간 뷰 분포 차이 존재 → **도메인 적응/일반화 기법** 적용 가능성 탐구.

#### (5) 계산 효율성과 성능의 균형
엣지 디바이스 배포를 위해:
- **모델 경량화**: Knowledge Distillation, Pruning
- **추론 지연 시간 최적화**: CVR은 backbone 대비 2배 이상 지연 → 비실용적

#### (6) 그룹 구조 활용한 계층적 학습
19개 유사 외관 그룹 구조를 명시적으로 활용:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{fine}} + \lambda \cdot \mathcal{L}_{\text{group}}$$

계층적 손실함수로 그룹 내 세밀한 구분 능력 강화.

#### (7) 통신 대역폭 고려
논문이 향후 과제로 언급한 **다중 카메라 시스템에서의 통신 대역폭**:
- 각 카메라에서 특징을 추출 후 압축 전송하는 **분산 추론(Distributed Inference)** 연구 필요

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 방법론 | P1 | P2 | P3 | MVP-N과의 관계 |
|------|------|--------|----|----|-----|----------------|
| **View-GCN** [Wei et al., CVPR 2020] | 2020 | View graph + GCN | ✗ | ✗ | ✓ | 공간 관계 필요 → P2 불만족 |
| **DAN** [Nie et al., TIP 2021] | 2021 | Self-attention | ✓ | ✓ | ✓ | MVP-N 최고 성능(91.61%) |
| **CVR** [Wei et al., ICCV 2021] | 2021 | Optimal transport + Transformer | ✓ | ✓ | ✓ | 가장 큰 모델, 최저 성능(79.99%) |
| **MVT** [Chen et al., BMVC 2021] | 2021 | Vision Transformer | ✗ | ✓ | ✓ | 임의 뷰 수 불허 → P1 불만족 |
| **HGNN+** [Gao et al., TPAMI 2022] | 2022 | Hypergraph neural network | ✓ | ✓ | ✓ | 추론 시 다수 샘플 필요 → 단일 테스트 불가 |
| **VFMVAC** [Liu et al., PR 2022] | 2022 | View filtering + aggregating conv | ✗ | ✓ | ✗ | P1, P3 불만족 |
| **SAT** [Huang et al., NeurIPS 2020] | 2020 | Self-adaptive training | - | - | - | MVP-N에서 soft label 중 최고 MVA(87.37%) |
| **SEAL** [Chen et al., AAAI 2021] | 2021 | Instance-dependent noise | - | - | - | 최저 MCDU 달성, MCCI 희생 |
| **OLS** [Zhang et al., TIP 2021] | 2021 | Online label smoothing | - | - | - | 기존 LS 대비 소폭 개선 |

### 핵심 관찰

1. **2020년 이후 Transformer 기반 방법의 증가**: MVT, CVR 등이 Transformer를 도입했으나, MVP-N에서 CVR의 성능(79.99%)은 오히려 최저 → **멀티뷰 환경에서 multi-head attention의 중복 정보 문제** 제기
2. **GCN 기반 방법의 한계**: View-GCN 등은 공간 관계(P2) 또는 고정 뷰 수(P1) 제약으로 실세계 적용 한계
3. **Soft label 방법의 실용성**: 계산 비용이 낮고, uninformative 뷰 비율이 작을 때 feature aggregation 방법과 경쟁 가능

---

## 참고자료

- **주 논문**: Wang, R., et al. "MVP-N: A Dataset and Benchmark for Real-World Multi-View Object Classification." *NeurIPS 2022 Track on Datasets and Benchmarks*. (제공된 PDF 원문)
- Su, H., et al. "Multi-view convolutional neural networks for 3d shape recognition." *ICCV 2015*. [MVCNN]
- Feng, Y., et al. "GVCNN: Group-view convolutional neural networks for 3D shape recognition." *CVPR 2018*.
- Nie, W., et al. "DAN: Deep-attention network for 3D shape recognition." *IEEE TIP 2021*.
- Wei, X., et al. "Learning canonical view representation for 3D shape recognition with arbitrary views." *ICCV 2021*. [CVR]
- Wei, X., et al. "View-GCN: View-based graph convolutional network for 3D shape analysis." *CVPR 2020*.
- Huang, L., et al. "Self-adaptive training: beyond empirical risk minimization." *NeurIPS 2020*. [SAT]
- Chen, P., et al. "Beyond class-conditional assumption: SEAL." *AAAI 2021*.
- Gao, Y., et al. "HGNN+: General hypergraph neural networks." *IEEE TPAMI 2022*.
- Peterson, J.C., et al. "Human uncertainty makes classification more robust." *ICCV 2019*.
- GitHub 저장소: https://github.com/SMNUResearch/MVP-N
