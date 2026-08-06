# Heterogeneous Ranking in Industrial-Scale Recommender Systems: A Case Study

> **⚠️ 중요 고지**: 본 분석은 제공된 PDF 원문에만 기반하며, 원문에서 명확히 확인되지 않는 내용은 명시적으로 표시합니다. 모든 수치와 주장은 원문 인용 근거를 함께 제시합니다.

---

## 1. Executive Summary (10문장 이내)

Google Discover는 웹 기사, 동영상, UGC 등 다양한 콘텐츠를 단일 랭킹 모델로 통합 서비스한다.  
이종(heterogeneous) 콘텐츠 환경에서는 데이터 이질성과 상호작용 이질성이 동시에 발생하여 기존 Shared MLP 기반 모델이 negative transfer 및 minority-type collapse에 취약하다.  
저자들은 명시적 이질성 신호를 게이팅 네트워크와 전문가 표현 양쪽에 주입하는 **HA-MoE** 아키텍처를 제안한다.  
이질성 컨텍스트를 활용한 **HA-Gating**과 선형 변조 레이어 **HDLM**을 통해 전문가 특화를 유도하되 운영 비용 증가를 최소화한다.  
모델의 내부 동작을 해석하고 추적하기 위한 관측 프레임워크 **LENS**도 함께 도입한다.  
평가 지표로는 전역 랭킹 성능과 콘텐츠 유형 간 교차 랭킹 정확성을 결합한 **DL-AUC**를 제안한다.  
오프라인 평가에서 HA-MoE는 pInterest와 pDisinterest 모두에서 기존 모델 대비 DL-AUC 향상을 달성한다.  
특히 Standard MMoE가 겪는 pDisinterest 회귀 문제를 HA-MoE가 해결함으로써 multi-task 최적화 충돌을 완화한다.  
온라인 A/B 테스트에서 DAU +0.22%, Diverse Engagement Rate +0.54% 등 실질적 지표 개선이 확인되었다.  
이 연구는 산업 규모 이종 추천 시스템에서의 이질성 명시 모델링의 중요성을 실증적으로 검증한다.

### 1-1. 연구의 목적과 필요성

**목적**: Google Discover처럼 개방형 웹에서 수집된 다양한 콘텐츠 유형(웹 기사, 동영상, UGC 등)을 단일 통합 랭킹 모델로 효과적으로 순위화하는 방법론 개발

**필요성** (Abstract, Section 1):

| 문제 유형 | 구체적 현상 |
|---|---|
| 데이터 이질성 | 피처 밀도 불균형, 메타데이터 구조 편차, 비대칭 신호 가용성 |
| 상호작용 이질성 | 콘텐츠 유형별 상이한 engagement 패턴(클릭 vs. 시청 시간) |
| 분포 불일치 | 클릭률 최적화 시 클릭베이트 기사 과도 노출 |
| Negative Transfer | 다중 목표 최적화 시 seesaw 현상 발생 |
| Minority-type Collapse | 지배적 콘텐츠 유형이 소수 유형을 압도 |

---

## 2. 핵심 주장과 근거 표

| # | 핵심 주장 | 근거/증거 | 위치 |
|---|---|---|---|
| 1 | 단일 Shared MLP는 이종 콘텐츠 랭킹에 부적합 | Web+ vs. Vid−: 0.971, Vid+ vs. Web−: 0.830 (Δ=0.141) | Table 2, p.7 |
| 2 | HA-MoE가 negative transfer 없이 다중 태스크 성능 향상 | pInterest 0.691, pDisinterest 0.949 (둘 다 최고) | Table 1, p.6 |
| 3 | HA-Gating과 HDLM 각각이 성능에 기여 | Ablation: w/o HA-Gating(0.689/0.949), w/o HDLM(0.690/0.945) vs. Full(0.691/0.949) | Table 1, p.6 |
| 4 | DL-AUC가 표준 글로벌 AUC가 놓치는 교차 유형 편향을 포착 | 글로벌 AUC는 다수 유형 내 비교에 지배되어 교차 편향 마스킹 | Section 4.1.1, p.5 |
| 5 | HA-MoE가 운영 비용 내에서 동작 | 모델 크기 +5% 미만, 서빙 레이턴시 +0.5% 미만 | Section 4.1.2, p.6 |
| 6 | LENS/PIEM이 전문가 특화 구조적 이상 탐지 가능 | Snapshot A의 낮은 PIEM 점수 = 기능 분화 실패 | Figure 5, p.7 |
| 7 | 오프라인 개선이 온라인으로 전환됨 | DAU +0.22%±0.11%, Diverse Engagement Rate +0.54%±0.07% | Table 3, p.8 |

---

## 2-1. 상세 설명

### 해결하고자 하는 문제

Google Discover의 단일 랭킹 모델 기반(Shared MLP + 11개 task head)이 다음 세 가지 문제에 직면:

1. **분포 불일치**: 콘텐츠 유형 간 engagement 분포 차이로 인한 다수 유형 편향
2. **Seesaw 현상**: 클릭 최대화 ↔ dismissal 최소화 간 trade-off
3. **Minority-type Collapse**: 피처가 풍부한 지배적 유형(웹 기사)이 소수 유형(동영상) 억압

---

### 제안 방법 및 수식

#### 전체 예측 수식 (Equation 1, p.4)

$$p_t = T_t\left(\sum_{n=1}^{N} g_t(x, h)_n \cdot \tilde{E}_n(x, h)\right)$$

- $p_t$: 태스크 $t$의 예측 확률
- $g_t(x, h)_n$: 태스크 $t$에 대한 전문가 $n$의 게이팅 확률
- $\tilde{E}_n(x, h)$: 이질성 적응 전문가 출력
- $T_t$: 태스크별 타워 네트워크

#### HA-Gating (Equation 2, p.4)

$$g_t(x, h) = \text{Softmax}(W_t \Phi(x, h) + b_t)$$

- $W_t \in \mathbb{R}^{N \times d_\Phi}$: 게이팅 가중치 행렬
- $b_t \in \mathbb{R}^N$: 편향 벡터
- $\Phi(x, h) \in \mathbb{R}^{d_\Phi}$: 밀집 피처 $x$와 이질성 신호 $h$의 경량 융합 함수
- 초기 배포에서는 $\Phi(x, h) = [x \| h]$ (단순 연결)로 구현

#### HDLM (Equations 3-4, p.4)

스케일 벡터와 시프트 벡터 계산:

$$\gamma_n(h) = W_{\gamma,n}h + b_{\gamma,n}, \quad \beta_n(h) = W_{\beta,n}h + b_{\beta,n}$$

- $W_{\gamma,n}, W_{\beta,n} \in \mathbb{R}^{d_e \times d_h}$: 변조 가중치 행렬

후기 아핀 변환을 통한 적응 표현:

$$\tilde{E}_n(x, h) = \gamma_n(h) \odot E_n(x) + \beta_n(h)$$

- $\odot$: 원소별 곱셈
- 최종 전문가 레이어에만 적용하여 훈련 속도와 안정성 유지

#### 통합 손실 함수 (Equations 5-7, p.4)

$$\mathcal{L} = \alpha \mathcal{L}_p + (1-\alpha)\mathcal{L}_r$$

Pointwise BCE 손실:

$$\mathcal{L}_p = \sum_{i \in \mathcal{B}} \sum_{t \in \mathcal{T}} w_t \mathcal{L}_{\text{BCE}}(y_{t,i}, p_{t,i})$$

Pairwise RankNet 손실:

$$\mathcal{L}_r = \frac{1}{|\mathcal{B}^+| \times |\mathcal{B}^-|} \sum_{i \in \mathcal{B}^+} \sum_{j \in \mathcal{B}^-} \log\left(1 + e^{-(s_i - s_j)}\right)$$

#### DL-AUC (Equations 10-13, p.5-6)

$$\text{DL-AUC} = \lambda \cdot \text{Micro-AUC} + (1-\lambda) \cdot \text{Macro-xAUC}$$

$$\text{Micro-AUC} = \frac{1}{|\mathcal{D}^+||\mathcal{D}^-|} \sum_{i \in \mathcal{D}^+} \sum_{j \in \mathcal{D}^-} \mathbb{I}(p_i > p_j)$$

$$\text{xAUC}(A, B) = \frac{1}{|\mathcal{D}^+_A||\mathcal{D}^-_B|} \sum_{i \in \mathcal{D}^+_A} \sum_{j \in \mathcal{D}^-_B} \mathbb{I}(p_i > p_j)$$

$$\text{Macro-xAUC} = \frac{1}{|C|(|C|-1)} \sum_{A,B \in C, A \neq B} \text{xAUC}(A, B)$$

- $\lambda = 0.8$로 설정 (전역 유틸리티 4배 중시)

#### PIEM (Equations 8-9, p.5)

최적 순열 탐색:

$$\sigma^* = \arg\min_\sigma \sum_{n=1}^{N} \text{JSD}(\mathbf{v}_{A,n} \| \mathbf{v}_{B,\sigma(n)})$$

PIEM 점수:

$$\text{PIEM}(A, B) = \frac{1}{N} \sum_{n=1}^{N} \left(1 - \sqrt{\text{JSD}(\mathbf{v}_{A,n} \| \mathbf{v}_{B,\sigma^*(n)})}\right)$$

---

### 모델 구조 요약

```
입력: x (밀집 피처) + h (이질성 신호)
    ↓
Φ(x, h) = [x ∥ h] (게이팅 융합)
    ↓
HA-Gating: g_t(x,h) → 태스크별 전문가 가중치
    ↓
N=4개 공유 전문가 {E_n}
    ↓
HDLM: γ_n(h) ⊙ E_n(x) + β_n(h) → Ẽ_n(x,h)
    ↓
K=11개 태스크별 타워 → 예측 확률 p_t
```

---

### 성능 향상

| 지표 | Shared MLP | Standard MMoE | HA-MoE |
|---|---|---|---|
| pInterest DL-AUC | 0.679 | 0.686 | **0.691** |
| pDisinterest DL-AUC | 0.939 | 0.934 (⬇️ 회귀) | **0.949** |
| Vid+ vs. Web− xAUC | 0.830 | N/A | **0.900** |
| Δ Cross-type Gap | 0.141 | N/A | **0.060** |

---

### 한계

1. 공개 데이터셋 부재: 사내 독점 데이터만 사용 → 재현 불가 (Section 2, p.2)
2. $\Phi(x,h)$가 단순 연결(concatenation)로 구현 → 더 표현력 있는 융합 미탐색 (Section 4.1.2, p.6)
3. PIEM의 낮은 점수가 모델 성능 저하인지 데이터 드리프트인지 구별 어려움 (Section 3.3, p.5)
4. 전문가 수 $N=4$가 {2,4,8,16} 중 grid search로 결정되었으나 이론적 근거 없음

---

## 3. 주장별 페이지/Figure/Table 번호

| 주장 | 근거 위치 |
|---|---|
| 이종 콘텐츠 통합 랭킹 필요성 | p.1-2, Figure 1 |
| HA-MoE 아키텍처 설계 | p.3-4, Figure 2 |
| HA-Gating 수식 | p.4, Equation 2 |
| HDLM 수식 | p.4, Equations 3-4 |
| 통합 손실 함수 | p.4, Equations 5-7 |
| DL-AUC 정의 | p.5, Equations 10-13 |
| PIEM 정의 | p.5, Equations 8-9 |
| 오프라인 성능 비교 | p.6-7, Table 1 |
| 교차 유형 xAUC 비교 | p.7, Table 2 |
| 전문가 활성화 시각화 | p.7, Figures 3-4 |
| 전문가 특화 추적 | p.7-8, Figure 5 |
| 온라인 A/B 테스트 결과 | p.8, Table 3 |
| 운영 오버헤드 | p.6, Section 4.1.2 |

---

## 4. 저자 보고 결과 vs. 해석 분리

### 저자가 직접 보고한 결과

**오프라인 평가 (Table 1, p.6)**:
- HA-MoE: pInterest DL-AUC = 0.691, pDisinterest DL-AUC = 0.949
- Shared MLP: pInterest = 0.679, pDisinterest = 0.939
- Standard MMoE: pInterest = 0.686, pDisinterest = 0.934 (pDisinterest 회귀)

**교차 유형 xAUC (Table 2, p.7)**:
- Shared MLP: Web+ vs. Vid− = 0.971, Vid+ vs. Web− = 0.830, Δ = 0.141
- HA-MoE: Web+ vs. Vid− = 0.960, Vid+ vs. Web− = 0.900, Δ = 0.060

**온라인 A/B 테스트 (Table 3, p.8)**:
- DAU: +0.22% ± 0.11%
- Viewed Impressions: +0.48% ± 0.34%
- Scroll Depth: +0.34% ± 0.25%
- Diverse Feed Rate: +0.36% ± 0.03%
- Diverse Engagement Rate: +0.54% ± 0.07%

**운영 비용 (Section 4.1.2, p.6)**:
- 모델 크기: +5% 미만
- 서빙 레이턴시 (p50, p95): +0.5% 미만
- 훈련 속도: 측정 노이즈 범위 내 유지

### 분석자 해석

1. **DL-AUC 절대값의 맥락**: 보고된 DL-AUC 수치(0.679~0.691)는 $\lambda=0.8$로 설정된 가중 평균이며, 이는 **운영 정책 파라미터**이므로 다른 시스템과 직접 비교 불가
2. **Ablation 결과의 상보성**: HA-MoE w/o HA-Gating (0.689/0.949)과 w/o HDLM (0.690/0.945)을 비교하면, pDisinterest에서는 HA-Gating이, pInterest에서는 HDLM이 더 중요한 역할을 하는 것으로 해석 가능 (저자는 명시하지 않음)
3. **xAUC 비대칭성**: Web+ vs. Vid− 가 Vid+ vs. Web− 보다 일관되게 높은 것은 웹 기사가 구조적으로 불리한 dismissal 패턴을 가질 수 있음을 시사 (저자 해석과 동일하나 메커니즘은 미확인)
4. **PIEM의 인과성 한계**: Snapshot A의 낮은 PIEM이 낮은 오프라인 지표와 상관되지만, PIEM이 성능을 **직접 예측**한다는 인과 주장은 저자도 명시적으로 피하고 있음

---

## 5. 통계적으로 취약한 부분과 비교 불가능한 수치

| 항목 | 문제점 | 심각도 |
|---|---|---|
| **Viewed Impressions 신뢰구간** | +0.48% ± 0.34% → 95% CI가 [+0.14%, +0.82%]로 넓음, 신호 약함 | 🔴 높음 |
| **Scroll Depth 신뢰구간** | +0.34% ± 0.25% → 95% CI가 [+0.09%, +0.59%]로 경계선상 | 🟠 중간 |
| **DAU 통계적 유의성** | +0.22% ± 0.11%는 통계적으로 유의하나 실용적 유의성은 맥락 의존적 | 🟡 주의 |
| **오프라인 DL-AUC 절대값** | $\lambda=0.8$은 Google Discover 특수 정책값; 타 시스템과 직접 비교 불가 | 🔴 높음 |
| **Ablation 차이 크기** | pInterest: 0.689→0.690→0.691 (0.001~0.002 차이), 통계적 유의성 검증 없음 | 🔴 높음 |
| **Standard MMoE xAUC 부재** | Table 2에 MMoE의 xAUC 미보고 → 비교 불완전 | 🟠 중간 |
| **7일 홀드아웃 대표성** | ~1천만 샘플, 7일 데이터 → 계절성/장기 트렌드 미포착 가능 | 🟡 주의 |
| **전문가 수 N=4 선택** | Grid search 결과만 보고, 통계적 유의성 검증 없음 | 🟡 주의 |
| **PIEM 점수 절대적 기준치 미제시** | "높은/낮은 PIEM"의 경계값 정의 없음 | 🟠 중간 |

---

## 6. 문서가 답하지 않는 질문

1. **재현 가능성**: 독점 데이터셋만 사용하여 공개 벤치마크에서의 성능 검증 없음 → 일반화 가능성 미확인
2. **콘텐츠 유형 수 확장성**: 현재 3가지 주요 유형(웹 기사, 동영상, UGC) 시각화만 제시; 더 많은 유형 추가 시 성능 변화 미보고
3. **$\alpha$ 하이퍼파라미터 민감도**: BCE와 RankNet 손실의 균형 파라미터 $\alpha$ 값과 민감도 분석 미제시
4. **콜드 스타트 성능**: 신규 콘텐츠 유형(예: Generative AI 카드) 초기 도입 시 성능 영향 미검증
5. **장기 성능 안정성**: 7일 A/B 테스트 이후 장기 사용자 행동 변화 미측정
6. **$\Phi(x,h)$ 융합 함수 비교**: 단순 concatenation 외 attention, cross-network 등 대안 비교 없음
7. **PIEM 임계값 설정 방법**: 어떤 PIEM 점수가 실제 배포 중단을 촉발해야 하는지 기준 미제시
8. **태스크 가중치 $w_t$ 최종값**: GradNorm으로 동적 조정된 최종 가중치 미보고
9. **전문가 수 N과 성능의 관계**: N=2,4,8,16 각각의 성능 곡선 미제시

---

## 7. 가장 중요한 그림 5개 해석

### Figure 1 (p.2): Google Discover 개요
Google Discover의 시스템 아키텍처를 보여주는 그림이다. 다양한 콘텐츠 유형(스포츠, 뉴스, 동영상, UGC 등)이 단일 랭킹 모델을 통해 개인화 피드로 통합되는 구조를 시각화한다. 이 그림은 문제의 본질—하나의 모델이 이질적 오픈웹 콘텐츠를 처리해야 하는 도전—을 직관적으로 제시하며, 연구 동기를 정당화하는 핵심 근거로 기능한다.

### Figure 2 (p.3): HA-MoE 아키텍처 개요
HA-MoE의 전체 구조를 도식화한다. 이질성 신호 $h$가 (1) 다중 게이팅 네트워크 $G_1, ..., G_K$와 (2) 각 전문가의 HDLM 레이어 양쪽에 동시에 주입되는 이중 경로를 명확히 보여준다. 랭킹 피처 $x$와 이질성 신호 $h$의 융합 $\Phi(x,h)$가 게이팅 입력으로 사용되고, 각 전문가 출력은 $(\gamma_n(h), \beta_n(h))$로 변조된 후 태스크별 타워로 전달되는 흐름을 제시한다. 이 구조가 파라미터 대폭 증가 없이 이질성 적응을 달성하는 핵심 설계임을 시각화한다.

### Figure 3 (p.7): 범용 태스크의 전문가 활성화 패턴
태스크 A(밀집 범용 액션: 좋아요)와 태스크 B(희소 범용 액션: 설문 피드백)의 콘텐츠 유형별 전문가 활성화 히트맵이다. **태스크 A**: 단일 전문가가 모든 콘텐츠 유형에서 지배적으로 활성화됨 → 범용 행동은 공유 학습이 효과적. **태스크 B**: 두 전문가가 분산 활성화됨 → 희소 레이블은 약간의 전문화가 필요함. 이는 모델이 태스크 특성에 따라 자동으로 전문가 할당 전략을 달리 학습함을 실증한다.

### Figure 4 (p.7): 특화 태스크의 전문가 활성화 패턴
태스크 C(상세 뷰 클릭)와 태스크 D(텍스트 확장)의 전문가 활성화 패턴을 보여준다. Figure 3과 달리 **콘텐츠 유형에 따라 서로 다른 전문가가 활성화되는 분산 패턴**이 관찰된다. 예를 들어 웹 기사와 동영상이 서로 다른 전문가를 주로 사용한다. 이는 HA-MoE가 포맷 의존적 행동에 대해 실질적인 전문가 특화를 학습하고 있음을 직접적으로 증명한다.

### Figure 5 (p.7): PIEM을 통한 전문가 특화 추적
세 모델 스냅샷(A, B, C)의 태스크-전문가 활성화 프로파일 히트맵과 PIEM 점수를 비교한다. **스냅샷 B와 C**: 각 전문가가 서로 다른 태스크를 담당하는 분산된 전문화 패턴 → 높은 PIEM(B,C) 점수. **스냅샷 A**: 대부분 태스크가 단일 전문가에 집중되는 활성화 불균형 → 낮은 PIEM(A,B) 점수. 이 그림은 PIEM이 전문가 기능 분화의 부재(functional collapse)를 정량적으로 탐지할 수 있음을 실증하며, 오프라인 평가 완료 전 조기 경보 신호로서의 실용성을 보여준다.

---

## 8. 결론 및 후속 연구

### 저자가 제시한 시사점 (Section 5, p.8)

1. **이질성 명시 모델링의 가치**: 이질성 컨텍스트를 게이팅과 전문가 표현 양쪽에 주입하면 negative transfer 없이 다중 태스크 성능 향상 가능
2. **운영 해석 가능성의 중요성**: 블랙박스 MoE의 내부 동작을 LENS로 가시화하면 배포 신뢰성 향상
3. **평가 지표 재설계 필요성**: 표준 글로벌 AUC는 이종 환경에서 불충분; DL-AUC처럼 교차 세그먼트 정확성을 포함해야 함

### 저자가 제시한 후속 연구 계획 (Section 5, p.8)

| 방향 | 내용 |
|---|---|
| 게이팅 고도화 | 더 표현력 있는 게이팅 메커니즘 탐색 |
| 희소 라우팅 | 확장 가능한 sparse routing 도입 |
| 전문가 변조 개선 | 전문가 변조 메커니즘 정교화 |
| 사용자 컨텍스트 확장 | 콘텐츠 중심 신호를 넘어 사용자 수준 컨텍스트 통합 |
| 리스트와이즈 최적화 | 아이템 수준에서 피드 수준 오케스트레이션으로 확장 |
| LLM 통합 | LLM 기반 이종 랭킹 접근법 탐색 |

---

### 8-1. 모델의 일반화 성능 향상 가능성

**현재 한계**:

본 논문의 핵심 한계는 독점 데이터에 대한 의존성이다. 저자들은 *"Public datasets lack the content heterogeneity required to study such dynamics"* (p.2)라고 명시하며 일반화 가능성 검증을 포기하였다.

**일반화 성능 향상을 위한 고려 사항**:

1. **도메인 적응 가능성**: HDLM의 조건부 아핀 변환은 FiLM [25]에서 파생되었으며, FiLM은 시각적 추론, NLP 등 다양한 도메인에서 일반화 성능이 검증된 바 있다. 따라서 HA-MoE의 HDLM 구성 요소는 이론적으로 다른 이종 추천 환경(예: 이커머스 멀티카테고리, 뉴스+광고 통합)에도 적용 가능할 것으로 추정된다. 단, 이는 **분석자 추론**이며 논문에서 직접 검증되지 않았다.

2. **$\Phi(x,h)$ 표현력과 일반화의 트레이드오프**: 현재 단순 concatenation으로 구현된 $\Phi(x,h) = [x \| h]$는 훈련 안정성을 위해 단순화된 것이다. 더 복잡한 융합 함수(cross-attention, feature interaction 레이어 등)는 표현력을 높이지만 과적합 위험도 증가시킬 수 있다. 일반화를 위해서는 드롭아웃, 정규화 등의 추가 기법과 함께 탐색이 필요하다.

3. **전문가 수 N의 적응적 선택**: 현재 N=4는 Google Discover의 콘텐츠 유형 구성에 최적화된 값이다. 다른 플랫폼에서는 콘텐츠 다양성에 따라 N을 동적으로 결정하거나, 계층적 전문가 구조(전문가 내 전문가)를 채택해야 할 수 있다.

4. **PIEM의 이식 가능성**: PIEM은 레이블 없이 경량 행동 프로파일만으로 작동하므로, 다른 MoE 기반 추천 시스템에도 이식 가능하다. 이는 논문의 가장 일반화 가능한 기여 중 하나다.

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

> **⚠️ 주의**: 아래 비교는 논문 내 인용 문헌 및 일반적으로 알려진 연구 동향에 기반합니다. 각 논문의 구체적 수치는 원 논문 직접 확인을 권장합니다.

#### 관련 최신 연구와의 비교

| 연구 | 연도 | 핵심 기여 | HA-MoE와의 차이점 |
|---|---|---|---|
| **PLE** [29] (Tang et al.) | 2020 | 태스크별 전용 전문가 + 공유 전문가 분리 | HA-MoE는 이질성 신호로 게이팅을 **조건화**하는 점이 추가됨 |
| **STAR** [27] (Sheng et al.) | 2021 | 다중 도메인 CTR 예측을 위한 스타 토폴로지 | STAR는 도메인 분리가 전제; HA-MoE는 단일 통합 모델 유지 |
| **PEPNet** [6] (Chang et al.) | 2023 | 개인화 사전 정보 주입 EPNet | 사용자 맞춤화 중심; HA-MoE는 콘텐츠 유형 이질성 중심 |
| **M³oE** [35] (Zhang et al.) | 2024 | 다중 도메인 다중 태스크 MoE | 유사 방향이나 Google Discover의 오픈웹 이질성 특수성 미고려 |
| **GradNorm** [8] (Chen et al.) | 2018 | 그래디언트 정규화 기반 손실 균형 | HA-MoE에 통합 사용됨 (태스크 가중치 동적 조정) |
| **xAUC** [16] (Kallus & Zhou) | 2019 | 이분 랭킹의 공정성 측정 | DL-AUC의 Macro-xAUC 구성 요소의 이론적 기반 |

#### 본 논문이 앞으로의 연구에 미치는 영향

1. **산업 실증 연구의 벤치마크**: 이종 추천에서의 MoE 적용을 대규모 실제 배포 사례로 검증한 드문 연구로, 향후 유사 연구의 참조 기준이 될 수 있다.

2. **DL-AUC의 확산 가능성**: 이종 환경 평가를 위한 DL-AUC는 단순하고 해석 가능하여 다른 이종 추천 시스템 연구에서 채택될 가능성이 높다. 단, $\lambda$ 파라미터의 표준화 논의가 필요하다.

3. **LENS/PIEM의 방법론적 기여**: MoE 모델의 블랙박스 문제에 대한 레이블 없는 경량 진단 방법으로, 추천 시스템 모니터링 연구에 기여할 수 있다.

#### 앞으로 연구 시 고려할 점

| 고려 사항 | 내용 |
|---|---|
| **공개 데이터 부재 해결** | 이종성을 갖는 합성 벤치마크 데이터셋 구축 또는 반공개 데이터 활용 필요 |
| **희소 라우팅과의 결합** | Switch Transformer, GShard 등 희소 MoE와의 결합으로 확장성 향상 탐색 |
| **LLM 임베딩과의 통합** | 콘텐츠 의미 정보를 이질성 신호 $h$로 활용 (예: 텍스트 임베딩) |
| **공정성과 이질성의 관계** | 소수 콘텐츠 유형 보호가 공급자 공정성과 어떻게 연결되는지 이론적 분석 필요 |
| **연속 학습 안정성** | Warm-start 재훈련에서 PIEM이 낮아지는 패턴의 체계적 연구 |
| **리스트와이즈 이질성 모델링** | 현재 아이템 수준 예측에서 피드 전체 최적화로 확장 시 이질성 처리 방법 |
| **$\lambda$ 파라미터 학습** | DL-AUC의 $\lambda$를 고정하지 않고 비즈니스 목표에 따라 동적 조정하는 방법 |

---

## 참고 자료 (논문 내 인용 문헌 기반)

본 분석은 다음 원문 및 논문 내 인용 문헌을 기반으로 작성되었습니다:

**원문**:
- Bai, D., Liu, J., Tang, Z., Wu, P., Al-Thawr, N., & Wang, L. (2026). *Heterogeneous Ranking in Industrial-Scale Recommender Systems: A Case Study*. RecSys '26. arXiv:2607.27577v1

**논문 내 주요 인용 문헌**:
- [4] Burges et al. (2005). *Learning to rank using gradient descent*. ICML.
- [6] Chang et al. (2023). *PEPNet*. KDD 2023.
- [8] Chen et al. (2018). *GradNorm*. ICML.
- [16] Kallus & Zhou (2019). *xAUC*. NeurIPS 32.
- [17] Kuhn (1955). *The Hungarian method*. Naval Research Logistics Quarterly.
- [23] Ma et al. (2018). *MMoE*. KDD 2018.
- [25] Perez et al. (2018). *FiLM*. AAAI.
- [27] Sheng et al. (2021). *STAR*. CIKM 2021.
- [29] Tang et al. (2020). *PLE*. RecSys 2020.
- [35] Zhang et al. (2024). *M³oE*. SIGIR 2024.
- [37] Zhao et al. (2024). *Retrievable Domain-Sensitive Feature Memory*. arXiv:2405.12892.
