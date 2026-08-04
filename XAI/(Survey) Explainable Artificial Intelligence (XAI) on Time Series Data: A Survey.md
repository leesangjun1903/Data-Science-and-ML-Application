# Explainable Artificial Intelligence (XAI) on Time Series Data: A Survey

> **⚠️ 정확도 주의사항**: 본 답변은 제공된 PDF 원문에 기반합니다. PDF에 명시되지 않은 내용(특히 8-2의 최신 연구 비교)은 일반적 AI 연구 지식을 바탕으로 작성하되, 해당 부분을 명확히 표시합니다. 수치나 실험 결과는 논문 원문에서 직접 인용된 것만 제시합니다.

---

## 1. Executive Summary (10문장 이내)

이 논문은 시계열 데이터에 적용된 XAI(설명 가능한 인공지능) 방법론을 체계적으로 조사한 서베이 논문이다.  
딥러닝 모델이 시계열 분석의 최신 기술로 자리잡았으나, 높은 복잡도로 인해 해석이 어렵다는 근본적 한계가 존재한다.  
의료, 자율주행 등 고위험 분야에서 모델의 해석 가능성은 신뢰성 확보의 전제 조건이다.  
저자들은 CNN 기반 역전파 방법, 섭동 기반 방법, RNN 어텐션 메커니즘, 데이터 마이닝 기반 방법(SAX, Fuzzy Logic), 대표 예시 기반 방법(Shapelets) 등을 분류·정리한다.  
XAI 방법은 설명의 범위(로컬/글로벌)와 대상 청중(개발자/사용자/의사결정자)에 따라 구분된다.  
대부분의 방법은 개발자를 대상으로 기술적 설명에 집중하며, 최종 사용자와의 상호작용은 간과되는 경향이 있다.  
설명 평가 방법으로는 질적 평가(전문가 검토)와 양적 평가(섭동 분석, 무작위화 검사)가 존재하나, 시계열의 비직관적 특성으로 인해 질적 평가의 한계가 있다.  
시계열 특화 XAI 방법의 부재와 설명의 강건성·안정성 보장 메트릭의 부족이 주요 한계로 지적된다.  
저자들은 XAI가 단순 신뢰성 제공을 넘어 모델의 강건성과 신뢰도를 높이는 새로운 훈련 방식으로 발전할 잠재력이 있다고 주장한다.  
궁극적으로 사람 중심의 엔드-투-엔드 XAI 시스템 구축이 향후 과제로 제시된다.

### 1-1. 연구 목적과 필요성

| 구분 | 내용 |
|------|------|
| **핵심 문제** | 딥러닝의 블랙박스 특성으로 인한 시계열 모델 해석 불가능성 |
| **실용적 필요성** | 의료 진단, 자율주행 등 고위험 분야에서 모델 결정에 대한 신뢰 확보 필수 |
| **학문적 공백** | CV·NLP 분야 대비 시계열 XAI 연구의 상대적 부재 |
| **시계열 고유 과제** | 인간이 시계열을 직관적으로 이해하지 못함 → 설명 평가 자체가 어려움 (Introduction, p.1) |
| **정확도의 한계** | 높은 정확도(accuracy)만으로는 모델의 안정성·강건성을 보장할 수 없음 (Section II-A, p.3) |

---

## 2. 핵심 주장과 근거 표

| 핵심 주장 | 근거 | 위치 |
|-----------|------|------|
| 딥러닝 기반 시계열 모델은 해석 불가 | CNN, RNN은 높은 정확도 달성하나 내부 작동 불투명 | Abstract, p.1 |
| 시계열 XAI는 CV/NLP 대비 연구 부족 | 시계열의 비직관적 특성으로 질적 평가가 어려움 | Intro, p.1 |
| 대부분 XAI 방법은 개발자 중심 | 31개 방법 중 대다수가 개발자를 대상 청중으로 설정 | Table I, p.9 |
| 정확도만으로 시스템 안전성 불충분 | 소규모 입력 노이즈에 모델 출력이 급변 가능 | Section II-A, p.3 |
| XAI가 강건성 향상에 기여 가능 | Hartl et al.의 feature sensitivity, RATIO의 counterfactual | Section V-B, p.10 |
| 표준화된 설명 평가 메트릭 부재 | 각 방법마다 평가 방식이 상이, 범용 메트릭 없음 | Section VI, p.10 |
| 사용자 상호작용 고려 부족 | 대부분의 방법이 사용자 피드백 루프를 설계하지 않음 | Section VII, p.11 |

---

## 2-1. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 향상 및 한계

### 🔴 해결하고자 하는 문제

1. **해석 불가능성**: 딥러닝 모델(CNN, RNN, Transformer)의 내부 결정 과정 불투명
2. **시계열 특화 XAI 방법 부재**: 대부분의 방법이 CV 분야에서 전용(轉用)
3. **설명 평가 기준 부재**: 설명의 질을 객관적으로 측정할 통합 메트릭 없음
4. **강건성·안정성 보장 불가**: 기존 XAI 방법은 adversarial attack에 취약한지 판단 불가

---

### 🔵 주요 방법론과 수식

#### (1) Class Activation Mapping (CAM)
*(Wang et al. [26], Fawaz et al. [27], Oviedo et al. [28]; Section III-A1, p.4)*

$N$개의 채널에서 클래스 $c$에 대한 활성화 맵:

$$
M_c(t) = \sum_{k=1}^{N} w_k^c \cdot f_k(t)
$$

여기서:
- $f_k(t)$: 필터 $k$의 시점 $t$에서의 활성화 값
- $w_k^c$: 클래스 $c$에 대한 필터 $k$의 가중치 (소프트맥스 레이어 이전 선형 레이어에서 학습)
- Global Average Pooling (GAP) 필요 → 아키텍처 제약 존재

#### (2) Gradient×Input
*(Strodthoff et al. [32], Siddiqui et al. [1], Cho et al. [33]; Section III-A1, p.5)*

$$
R(x_i) = \frac{\partial \hat{y}}{\partial x_i} \cdot x_i
$$

여기서:
- $x_i$: 입력 시계열의 $i$번째 값
- $\hat{y}$: 모델 예측 출력
- $R(x_i)$: 입력 $x_i$의 예측에 대한 기여도(relevance)

#### (3) Perturbation-based (Occlusion Sensitivity)
*(ConvTimeNet [40]; Section III-A2, p.5)*

$$
\Delta y_c = y_c(x) - y_c(x_{\text{occluded}})
$$

여기서:
- $y_c(x)$: 원본 입력에서 클래스 $c$의 예측 확률
- $y_c(x_{\text{occluded}})$: 특정 시간 구간을 마스킹한 입력에서의 예측 확률
- $\Delta y_c$가 클수록 해당 구간의 기여도가 높음

#### (4) Symbolic Aggregate Approximation (SAX)
*(Senin & Malinchik [56], Le Nguyen et al. [57]; Section III-C, p.6)*

**Step 1 - PAA (Piecewise Aggregate Approximation)**:

$$
\bar{x}_j = \frac{N}{w} \sum_{i=\frac{N}{w}(j-1)+1}^{\frac{N}{w} \cdot j} x_i
$$

여기서:
- $N$: 시계열 길이
- $w$: 세그먼트 수
- $\bar{x}_j$: $j$번째 세그먼트의 평균값

**Step 2 - 심볼 할당**: 가우시안 분포 가정 하에 등확률 구간으로 심볼 할당

#### (5) Shapelets 기반 정보 이득
*(Ye & Keogh [77]; Section III-D, p.7)*

$$
\text{InfoGain}(S, D) = H(D) - H(D | d(S, \cdot))
$$

$$
H(D) = -\sum_{c} p_c \log p_c
$$

여기서:
- $S$: 후보 shapelet 서열
- $D$: 전체 데이터셋
- $d(S, T)$: shapelet $S$와 시계열 $T$ 사이의 최소 거리
- $p_c$: 클래스 $c$의 비율

#### (6) Attention Mechanism (LSTM 기반)
*(Choi et al. [45], Schockaert et al. [44]; Section III-B, p.6)*

$$
\alpha_t = \text{softmax}(W_a \cdot h_t + b_a)
$$

$$
\text{context} = \sum_t \alpha_t \cdot h_t
$$

여기서:
- $h_t$: 시점 $t$에서 LSTM의 은닉 상태
- $\alpha_t$: 시점 $t$의 어텐션 가중치
- $W_a, b_a$: 학습 가능한 파라미터

---

### 🟠 모델 구조 요약

```
XAI for Time Series
├── Post-hoc Methods (CNN 설명)
│   ├── Backpropagation-based: CAM, Gradient×Input, LRP
│   └── Perturbation-based: Occlusion Sensitivity, Counterfactuals
├── Ante-hoc Methods (RNN 내재적 설명)
│   └── Attention Mechanism: LSTM-FCN, Transformer (TFT)
├── Data Mining Methods (시계열 특화)
│   ├── SAX 기반: SAX-VSM, SAX-SFA-SEQL
│   └── Fuzzy Logic: FRBS, Fuzzy Cognitive Map
└── Example-based Methods
    └── Shapelets: BSPCOVER, GST, AI-PR-CNN
```

---

### 🟡 성능 향상 및 한계

| 방법론 | 성능 향상 | 한계 |
|--------|----------|------|
| CAM | 예측에 기여한 서브시퀀스 시각화 가능 | GAP 레이어 필수 → 아키텍처 제약 |
| Gradient×Input | 분류·회귀 모두 적용 가능 | 그래디언트 포화(saturation) 문제 |
| Occlusion Sensitivity | 모델 무관(agnostic) 적용 가능 | 계산 비용 높음 |
| Attention Mechanism | Ante-hoc → 추가 계산 불필요 | 어텐션 ≠ 설명이라는 비판 존재 |
| Shapelets | 가변 길이 시계열 처리 가능, 해석 직관적 | 계산 시간 과다 (최적화 연구 진행 중) |
| SAX | 반복 패턴 검출에 효과적, 노이즈 강건 | 고차원 특성 표현에 한계 |

---

## 3. 각 주장의 위치 (페이지/Figure/Table 번호)

| 주장 | 위치 |
|------|------|
| 시계열 XAI의 비직관성 문제 | Introduction, p.1 |
| XAI 핵심 용어 정의 (Explainability, Trustworthiness 등) | Section II, p.2, **Fig. 2** |
| 안정성·강건성·신뢰도의 연결 구조 | Section II-A, p.3, **Fig. 3** |
| CAM 방법론 설명 | Section III-A1, p.4 |
| Gradient×Input 시각화 | Section III-A1, p.5, **Fig. 4** |
| Occlusion Sensitivity 시각화 | Section III-A2, p.5, **Fig. 5** |
| Attention Mechanism 유형 분류 | Section III-B, p.6, **Fig. 6** |
| Shapelets 예시 | Section III-D, p.7, **Fig. 7** |
| 전체 XAI 방법 요약 | **Table I**, p.9 |
| 필터 클러스터링을 통한 글로벌 설명 | Section IV-B, p.8, **Fig. 8** |
| 섭동 분석 비교 결과 | Section VI-B, p.10, **Fig. 9** |
| 설명 평가 방법 요약 | **Table II**, p.12 |
| 시계열 특화 CNN-XAI 방법 부재 지적 | Section VII, p.11 |

---

## 4. 저자 보고 결과 vs. 독자 해석 분리

### 저자가 직접 보고한 내용 (원문 인용)

| 구분 | 내용 | 출처 |
|------|------|------|
| **연구 주제** | "explainability of models applied on time series has not gather much attention compared to the computer vision or the natural language processing fields" | Abstract, p.1 |
| **방법** | CAM은 GAP 레이어 필요, Gradient×Input은 단일 순전파·역전파로 설명 생성 | Section III-A1, p.4-5 |
| **결과** | "most salient features are the same than the features with highest potential to cause missclassification" (Hartl et al.) | Section V-B, p.10 |
| **한계** | "there is a lack of explainable methods applied on CNNs specifically designed for time series tasks" | Section VII, p.11 |
| **평가 한계** | "qualitative evaluations might have a limited potential in the time series field" | Section VI-A, p.10 |
| **미래 방향** | "XAI field has more potential than just facilitating trustworthiness... potential to lead to new metrics and training practices" | Section VII, p.11 |

### 본 분석자의 해석

| 구분 | 해석 |
|------|------|
| **방법론 편향** | Table I를 분석하면, 31개 방법 중 대다수(약 70%)가 개발자를 대상 청중으로 설정 → 현장 적용 가능성에 대한 구조적 편향 존재 |
| **평가 순환 문제** | 설명의 질을 섭동 기반으로 평가하는 방식은, 설명 방법과 평가 방법이 동일한 가정을 공유할 경우 순환 논리에 빠질 위험 |
| **Ante-hoc 과대평가 가능성** | 어텐션 가중치를 "설명"으로 간주하는 방식은 Jain & Wallace (2019) 등에서 비판받은 바 있으나, 이 논문에서는 해당 비판이 충분히 다루어지지 않음 |
| **서베이 범위 한계** | 논문이 2021년 4월 제출(arXiv v1) 기준이므로, 이후 등장한 주요 방법(TimeSHAP, LIME for TS 등)은 포함되지 않음 |

---

## 5. 통계적으로 취약한 부분 및 비교 불가능한 수치

> ⚠️ 아래 항목들은 신중하게 해석해야 합니다.

| 취약점 유형 | 구체적 내용 | 위치 |
|------------|-----------|------|
| **벤치마크 부재** | 대부분의 XAI 방법이 서로 다른 데이터셋(UCR, MIMIC, JIGSAWS 등)에서 평가 → 직접 성능 비교 **불가능** | Table I, p.9 |
| **정량적 평가 부재** | Table I의 31개 방법 중 "No" 평가가 대다수 → 설명 품질의 객관적 비교 근거 없음 | Table I, p.9 |
| **표준화 메트릭 없음** | Table II에서 각 방법이 상이한 평가 접근법 사용(Expert assessment vs. Perturbation approach) → 비교 불가 | Table II, p.12 |
| **데이터셋 다양성** | Wang et al. [26]은 80개 이상 UCR 데이터셋 사용 vs. 일부 연구는 단일 도메인 데이터셋만 사용 → 일반화 가능성 불균형 | Section III-A1, p.5 |
| **"Best of our knowledge" 한계** | 저자들이 여러 주장에 "to the best of our knowledge"를 사용 → 포괄성 보장 불가 | Section IV-B, p.8; Section V-B, p.10 |
| **인과 vs. 상관 혼동 가능성** | 어텐션 가중치를 "기여도"로 해석하는 것이 실제 인과관계를 반영하는지 검증 미흡 | Section III-B, p.5-6 |

---

## 6. 논문이 답하지 않는 질문

| 미답 질문 | 중요도 |
|-----------|--------|
| 각 XAI 방법 간의 직접적 설명 품질 비교 결과는? | ★★★★★ |
| 시계열 XAI를 위한 표준화된 벤치마크 데이터셋은 무엇이어야 하는가? | ★★★★★ |
| 어텐션 가중치가 실제로 모델의 결정 근거를 반영하는가(Attention ≠ Explanation 문제)? | ★★★★★ |
| 다변량(multivariate) 시계열에 특화된 XAI 방법은 무엇인가? | ★★★★☆ |
| 설명의 품질(faithfulness)과 사용자 이해도(comprehensibility) 간의 trade-off를 어떻게 정량화하는가? | ★★★★☆ |
| 비정기적(irregular) 시계열에 대한 XAI 방법론은? | ★★★★☆ |
| 실시간(real-time) 시계열 예측 시스템에서의 XAI 적용 가능성은? | ★★★☆☆ |
| 사용자 신뢰도와 설명 방식 간의 인과관계를 어떻게 측정하는가? | ★★★☆☆ |
| XAI 방법 자체의 계산 비용이 실용화에 미치는 영향은? | ★★★☆☆ |
| 설명의 안정성(동일 입력에 대한 동일 설명 보장)을 어떻게 측정하는가? | ★★★☆☆ |

---

## 7. 가장 중요한 그림 5개 해석

### Fig. 1 (p.2) — 서베이 구조 개요도
**해석**: 논문 전체의 질문 구조("Why? What for? How? How well?")를 섹션과 연결하는 지도. XAI 방법을 "목적 → 범위 → 방법론 → 평가"의 순서로 체계화한 프레임워크를 제시한다. 이 그림은 단순 목차 이상의 의미로, 저자들이 XAI를 어떤 철학적 관점에서 분류하는지 보여준다.

### Fig. 2 (p.2) — XAI 목적 지식 그래프
**해석**: Explainability, Interpretability, Trustworthiness, Interactivity, Stability, Robustness, Reproducibility, Confidence 8가지 개념의 상호 관계를 방향 그래프로 표현. **핵심 통찰**: Trustworthiness가 중심 노드로, 나머지 개념들이 이를 향해 또는 이로부터 파생됨 → XAI의 궁극적 목표가 신뢰성 확보임을 시사. Explainability가 Trustworthiness를 "validates"하고, Interpretability가 "contributes to"하는 관계가 명확히 구분된다.

### Fig. 6 (p.6) — 어텐션 메커니즘 유형 4가지
**해석**: (a) Global Temporal, (b) Global Spatio-temporal, (c) Local Temporal, (d) Local Spatio-temporal 어텐션의 시각화. 색상의 농도가 어텐션 가중치의 크기를 반영한다. **주목할 점**: Global 어텐션은 전체 시계열에서 중요 시점을 균등하게 분산하여 파악하는 반면, Local 어텐션은 특정 시간 구간에 집중한다. 이는 예측 작업의 성격(장기/단기 의존성)에 따라 적합한 어텐션 유형이 다름을 시사한다. (Fig. 6, Schockaert et al. [44] 재현)

### Fig. 7 (p.7) — Shapelets 예시
**해석**: 두 클래스(Class 1, Class 2)에서 각 2개의 인스턴스에 동일한 shapelet1이 서로 다른 위치에서 발생함을 보여준다. **핵심**: Shapelet은 클래스 구분에 가장 결정적인 서브시퀀스로, 그 형태(shape)가 클래스 귀속의 설명 근거가 된다. 이는 모델이 "왜 이 시계열을 이 클래스로 분류했는가"에 대해 인간이 이해할 수 있는 직접적 근거를 제공하는 방식이다. (Fig. 7, Li et al. [83] 재현)

### Fig. 9 (p.11) — 섭동 분석 비교 (CHAP vs. LRP vs. Random)
**해석**: X축은 섭동된 영역의 비율, Y축은 채널 55의 활성화 변화량 합계. 가우시안, 역전파(Inverse), 제로(Zero) 세 종류의 섭동을 적용했을 때 CHAP(Cho et al.)와 LRP, Random 방법 간 차이를 비교한다. **통계적 취약점 주의**: 단일 채널(Channel 55)만을 대상으로 한 결과로, 다른 채널에서도 동일한 패턴이 나타나는지 불명확하다. 그러나 CHAP가 대부분의 섭동 조건에서 LRP보다 활성화 변화가 더 두드러진다는 점은 CHAP의 설명이 더 "faithful"할 가능성을 시사한다. (Fig. 9, Cho et al. [33] 재현)

---

## 8. 결론 — 연구자 제시 시사점, 후속 연구 계획 및 추가 방향

### 연구자들이 제시한 시사점 (Section VII, p.11)

| 시사점 | 내용 |
|--------|------|
| 시계열 특화 CNN-XAI 방법 필요 | 대부분의 방법이 CV에서 전용, 시계열 고유 특성을 살린 방법 개발 필요 |
| 사용자 상호작용 고려 부족 | 기술적 설명에만 집중, 사용자·개발자와의 상호작용 시스템 구축 필요 |
| XAI의 강건성 보장 잠재력 | Explainability insights → 새로운 훈련 방식·메트릭으로 발전 가능 |
| 엔드-투-엔드 XAI 시스템 부재 | 인터랙티브 피드백 시스템이 신뢰 구축의 핵심 경로가 될 것 |

### 연구자들이 제시한 후속 연구 방향

- 시계열 고유 특성에 최적화된 CNN 설명 방법 개발
- 인식론적 불확실성(epistemic uncertainty) 정량화를 위한 XAI 기반 메트릭 개발
- 최종 사용자(의사, 외과의, 운전자 등)와 AI 시스템 간 신뢰 구축 인터페이스 설계
- XAI를 통한 adversarial robustness 향상 방법론 탐구

---

### 8-1. 모델의 일반화 성능 향상 가능성

논문은 일반화 성능에 대해 직접적인 실험 결과를 제시하지 않으나, 다음의 메커니즘을 통한 일반화 향상 가능성을 시사한다:

**① Diversity Penalty를 통한 일반화 (Section V-B, p.10)**

Gee et al. [94]의 PDL 방법은 다양성 페널티(diversity penalty)를 도입하여 더 다양한 프로토타입을 학습한다:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{class}} + \lambda \cdot \mathcal{L}_{\text{diversity}}
$$

$$
\mathcal{L}_{\text{diversity}} = -\sum_{i \neq j} d(p_i, p_j)
$$

여기서 $p_i, p_j$는 학습된 프로토타입, $d(\cdot)$는 거리 함수. 이를 통해 잠재 공간에서 클래스 경계가 어려운 영역에 집중하여 **분포 외(out-of-distribution) 샘플에 대한 안정성** 향상.

**② Adversarial Training과 ACET의 결합 (Section V-B, p.10)**

Ates et al. [97]의 RATIO:

$$
\mathcal{L}_{\text{RATIO}} = \alpha \cdot \mathcal{L}_{\text{AT}} + (1-\alpha) \cdot \mathcal{L}_{\text{ACET}}
$$

- $\mathcal{L}_{\text{AT}}$: Adversarial Training loss → 적대적 공격에 대한 강건성
- $\mathcal{L}_{\text{ACET}}$: Adversarial Confidence Enhanced Training loss → OOD 샘플 처리
- **한계**: 두 목적 함수 간의 최적 $\alpha$ 결정 기준이 명확하지 않음

**③ Regularization을 통한 해석 가능한 Shapelet 학습 (Section III-D, p.7)**

Wang et al. [81]과 Kidger et al. [80]의 정규화 항:

$$
\mathcal{L}_{\text{shapelet}} = \mathcal{L}_{\text{class}} + \lambda_1 \mathcal{L}_{\text{interpretability}} + \lambda_2 \mathcal{L}_{\text{orthogonality}}
$$

해석 가능한 shapelet을 강제함으로써 모델이 과적합 특성(spurious features)에 의존하지 않도록 유도 → 일반화 성능 향상 가능성 존재.

**④ 일반화와 관련된 논문의 한계점**

> ⚠️ 논문은 각 방법의 일반화 성능을 체계적으로 비교하는 실험을 수행하지 않음. 대부분의 방법이 서로 다른 데이터셋에서 평가되어 일반화 성능 비교가 **구조적으로 불가능**한 상태.

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ 아래 내용은 AI 연구 분야의 일반적 지식을 바탕으로 작성하며, 본 논문(2021년 4월 arXiv 제출)에 포함되지 않은 연구들입니다. 개별 논문의 구체적 수치는 확인 불가하므로 방향성 중심으로 서술합니다.

#### 본 논문 이후 등장한 주요 연구 방향

| 연구 방향 | 대표 연구 (참고 수준) | 본 논문과의 관계 |
|-----------|---------------------|----------------|
| **TimeSHAP** (2021) | Bento et al., "TimeSHAP: Explaining Recurrent Models through Sequence Perturbations" (KDD 2021) | 본 논문이 SHAP의 시계열 적용 가능성을 언급([51])했으나, 시계열 특화 구현이 부재했던 공백을 채움 |
| **Transformer 기반 XAI** (2021~) | Temporal Fusion Transformer의 attention 해석성 심화 연구들 | 본 논문의 TFT[50] 언급을 확장 |
| **Counterfactual TS XAI** (2021~) | Delaney et al., "Instance-Based Counterfactual Explanations for Time Series Classification" | 본 논문의 RATIO[93] 이후 발전된 counterfactual 접근법 |
| **Foundation Models for TS** (2023~) | TimeGPT, MOIRAI 등 대규모 시계열 사전학습 모델 | 본 논문에서 논의되지 않은 새로운 블랙박스 문제 생성 |
| **Concept-based XAI for TS** (2022~) | 시계열을 위한 TCAV(Testing with Concept Activation Vectors) 확장 | 본 논문의 글로벌 설명 방법 부재를 보완 |

#### 본 논문이 앞으로의 연구에 미치는 영향

1. **분류 체계 기여**: Post-hoc/Ante-hoc, Model-specific/Model-agnostic, Local/Global, 대상 청중의 4차원 분류 체계는 이후 XAI 연구의 표준 분류 프레임워크로 활용 가능

2. **연구 공백 명시**: 시계열 특화 CNN-XAI 방법의 부재, 표준화된 평가 메트릭의 필요성을 명확히 제시 → 후속 연구의 문제 설정 가이드

3. **강건성-설명 가능성 연결**: XAI를 단순 해석 도구가 아닌 강건성 향상 수단으로 바라보는 관점 제시

#### 앞으로의 연구 시 고려할 점

| 고려사항 | 구체적 방향 |
|----------|------------|
| **표준 벤치마크 구축** | 다양한 XAI 방법을 동일한 조건에서 비교할 수 있는 시계열 XAI 전용 벤치마크 필요 |
| **다변량 시계열 특화** | 변수 간 상호작용을 설명하는 방법론 (본 논문은 단변량 중심) |
| **실시간 XAI** | 스트리밍 시계열에 적용 가능한 온라인 XAI 방법 개발 |
| **Foundation Model XAI** | GPT 스타일의 대형 시계열 모델에 대한 XAI 방법론 |
| **인간-AI 협업 설계** | 본 논문이 강조한 사용자 상호작용 시스템의 실험적 구현 및 평가 |
| **인과적 XAI** | 상관관계 기반 설명에서 인과관계 기반 설명으로 전환 |
| **Privacy-aware XAI** | 의료 시계열 등에서 프라이버시를 보호하면서 설명을 제공하는 방법 |
| **어텐션 ≠ 설명 문제 해결** | 어텐션 가중치의 faithfulness를 보장하는 이론적 기반 마련 |

---

## 참고 자료

**주요 참고 논문 (원문 내 인용)**

1. Rojat, T., Puget, R., Filliat, D., Del Ser, J., Gelin, R., & Díaz-Rodríguez, N. (2021). "Explainable Artificial Intelligence (XAI) on Time Series Data: A Survey." arXiv:2104.00950v1

2. Arrieta, A. B., et al. (2020). "Explainable artificial intelligence (XAI): Concepts, taxonomies, opportunities and challenges toward responsible AI." *Information Fusion*, 58, 82–115. [논문 내 참고문헌 5]

3. Lundberg, S. M., & Lee, S.-I. (2017). "A unified approach to interpreting model predictions." *NeurIPS*. [논문 내 참고문헌 52]

4. Zhou, B., et al. (2016). "Learning deep features for discriminative localization." *CVPR*. [논문 내 참고문헌 29]

5. Arnout, H., et al. (2019). "Towards a rigorous evaluation of XAI methods on time series." *ICCVW*. [논문 내 참고문헌 101]

6. Lim, B., et al. (2019). "Temporal fusion transformers for interpretable multi-horizon time series forecasting." arXiv:1912.09363. [논문 내 참고문헌 50]

7. Ye, L., & Keogh, E. (2009). "Time series shapelets: a new primitive for data mining." *KDD*. [논문 내 참고문헌 76]

8. Vaswani, A., et al. (2017). "Attention is all you need." *NeurIPS*. [논문 내 참고문헌 49]
