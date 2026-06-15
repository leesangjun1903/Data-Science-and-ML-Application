# Noise-resistant Deep Metric Learning with Ranking-based Instance Selection

---

## 📌 참고 자료

- **주 논문**: Chang Liu et al., "Noise-resistant Deep Metric Learning with Ranking-based Instance Selection," arXiv:2103.16047v2, 2021.
- **GitHub**: https://github.com/alibaba-edu/Ranking-based-Instance-Selection
- 논문 내 인용 문헌들 (Co-teaching [Han et al., NeurIPS 2018], MCL [Wang et al., CVPR 2020], SoftTriple [Qian et al., ICCV 2019] 등)

> ⚠️ **주의**: 2020년 이후 관련 최신 연구 비교 분석 섹션은, 본 논문의 참고문헌 범위 및 제 학습 데이터 내에서 확인 가능한 연구들만을 기반으로 서술하였습니다. 2021년 이후 발표된 일부 논문은 직접 검색·열람이 불가능하므로, 해당 부분에서 불확실한 내용은 억지로 기술하지 않고 명시적으로 표시합니다.

---

## 1. 핵심 주장 및 주요 기여 요약

### 🎯 핵심 주장

실세계 데이터의 **노이즈 레이블(noisy labels)** 문제는 Deep Metric Learning(DML)의 성능을 심각하게 저하시키며, 기존 분류(classification) 태스크 위주의 노이즈 대응 방법들은 DML에 직접 적용하기 어렵다. 본 논문은 DML에 특화된 노이즈 저항 학습 방법인 **PRISM(Probabilistic Ranking-based Instance Selection with Memory)** 을 제안한다.

### 🏆 주요 기여

| 기여 항목 | 내용 |
|-----------|------|
| **PRISM 알고리즘 제안** | Memory Bank를 활용한 확률적 노이즈 식별 및 필터링 |
| **sTRM 기법** | 슬라이딩 윈도우 기반 임계값 안정화 방법 |
| **가속화 기법** | 클래스 센터 벡터로 개별 샘플 대체하여 시간복잡도 $O(PKN) \to O(PK(C))$ 감소 |
| **Small Cluster 노이즈 모델** | 실세계 open-set 노이즈를 모방한 새로운 노이즈 합성 방법 제안 |
| **CARS-98N 데이터셋 구축** | Pinterest 크롤링 기반 실세계 노이즈 데이터셋 신규 구축 |
| **광범위한 실험** | 12개 기존 방법 대비 최대 **Precision@1 +6.06%** 향상 |

---

## 2. 상세 분석

### 2-1. 해결하고자 하는 문제

**Deep Metric Learning (DML)** 은 유사한 데이터 쌍을 특징 공간에서 가깝게, 비유사 쌍은 멀리 배치하는 거리 척도를 학습하는 방법이다.

$$S(f(x_i), f(x_j)) = \frac{f(x_i)^T f(x_j)}{\|f(x_i)\| \|f(x_j)\|} \tag{1}$$

그런데 실세계 데이터에는 다음과 같은 노이즈 레이블 문제가 존재한다:

- **인간 어노테이션 오류**: 레이블 부착 작업자의 실수
- **자동화 수집 오류**: 크롤링 등 자동 수집 과정에서의 잘못된 레이블
- **Open-set noise**: 데이터셋에 정의된 클래스에 속하지 않는 데이터 유입

기존 노이즈 저항 방법들은 **분류(classification)** 태스크에 집중되어 있으며, DML에서는:
1. 좋은 유사도 메트릭이 있어야 노이즈를 탐지할 수 있고
2. 노이즈 없는 깨끗한 데이터가 있어야 좋은 메트릭을 학습할 수 있는 **닭-달걀 문제(chicken-and-egg problem)** 가 발생한다.

---

### 2-2. 제안 방법 (수식 포함)

#### 🔷 Step 1: 클린 데이터 확률 계산 — $P_{\text{clean}}(i)$

Memory Bank $\mathcal{M} = \{(v_0, y_0), (v_1, y_1), \ldots, (v_M, y_M)\}$ 에 저장된 과거 피처들을 이용하여, 샘플 $(x_i, y_i)$의 레이블이 클린할 확률을 정의한다:

$$P_{\text{clean}}(i) = \frac{\exp(T(x_i, y_i))}{\sum_{k \in C} \exp(T(x_i, k))} \tag{2}$$

$$T(x_i, k) = \frac{1}{M_k} \sum_{(v_j, y_j) \in \mathcal{M},\, y_j = k} S(f(x_i), v_j) \tag{3}$$

여기서:
- $M_k$: Memory Bank에서 클래스 $k$에 속하는 샘플 수
- $T(x_i, k)$: $x_i$와 클래스 $k$의 메모리 피처들 간 평균 유사도
- Eq.(2)는 Bayesian 관점에서 $P(Y=k \mid X=x_i)$에 해당 (균일 사전 확률 가정)

#### 🔷 Step 2: 임계값 결정 — TRM & sTRM

**TRM (Top-R Method)**: 각 미니배치 내에서 $P_{\text{clean}}(i)$가 하위 $R\%$인 샘플을 노이즈로 처리

**sTRM (Smooth Top-R Method)**: 최근 $\tau$개 배치의 $R$번째 백분위수 평균으로 임계값을 안정화

$$m = \frac{1}{\tau} \sum_{j=t-\tau}^{t} Q_j \tag{4}$$

여기서 $Q_j$는 $j$번째 미니배치에서의 $R$번째 백분위수 $P_{\text{clean}}(i)$ 값이다.

#### 🔷 Step 3: 가속화 — 클래스 센터 벡터 도입

개별 샘플 대신 클래스 평균 피처 벡터 $w_k$를 사용:

$$\frac{\sum_{(v_j, y_j) \in \mathcal{M},\, y_j=k} S(f(x_i), v_j)}{M_k} = w_k \frac{f(x_i)}{\|f(x_i)\|} \tag{5}$$

$$w_k = \frac{1}{M_k} \sum_{(v_j, y_j) \in \mathcal{M},\, y_j=k} \frac{v_j}{\|v_j\|} \tag{6}$$

이를 통해 $P_{\text{clean}}(i)$의 가속화 버전:

$$P_{\text{clean}}(i) = \exp\!\left(w_{y_i} \frac{f(x_i)}{\|f(x_i)\|}\right) \Bigg/ \sum_{k \in C} \exp\!\left(w_k \frac{f(x_i)}{\|f(x_i)\|}\right) \tag{7}$$

- 시간복잡도: $O(PKN) \to O(PK|C|)$, 가속 비율 $= \frac{N}{|C|}$
- SOP 데이터셋에서 **6.9배** 속도 향상 달성

#### 🔷 Step 4: 손실 함수

**Batch 기반 Contrastive Loss**:

$$L_{\text{batch}}(\mathcal{B}) = \sum_{\substack{(x_i,y_i),(x_j,y_j)\in\mathcal{B} \\ y_i \neq y_j}} \max(S(f(x_i), f(x_j)) - \lambda, 0) - \sum_{\substack{(x_i,y_i),(x_j,y_j)\in\mathcal{B} \\ y_i = y_j}} S(f(x_i), f(x_j)) \tag{8}$$

**Memory Bank 기반 Contrastive Loss**:

$$L_{\text{bank}}(\mathcal{M}, \mathcal{B}) = \sum_{\substack{(x_i,y_i)\in\mathcal{B},\,(v_j,y_j)\in\mathcal{M} \\ y_i \neq y_j}} \max(S(f(x_i), v_j) - \lambda, 0) - \sum_{\substack{(x_i,y_i)\in\mathcal{B},\,(v_j,y_j)\in\mathcal{M} \\ y_i = y_j}} S(f(x_i), v_j) \tag{9}$$

**Proxy 기반 SoftTriple Loss**:

$$L_{\text{SoftTriple}} = -\log \frac{\exp(\lambda(S'_{i,y_i} - \delta))}{\exp(\lambda(S'_{i,y_i} - \delta)) + \exp(\lambda S'_{i,j})} \tag{10}$$

$$S'_{i,j} = \frac{\sum_{h=1}^{H} \exp\!\left(\gamma f(x_i)^\top p_j^h\right) f(x_i)^\top p_j^h}{\sum_{h=1}^{H} \exp\!\left(\gamma f(x_i)^\top p_j^h\right)} \tag{11}$$

---

### 2-3. 모델 구조

```
입력 미니배치 B
        ↓
  CNN Backbone (BN-Inception / ResNet-50)
        ↓
  피처 추출 f(x_i)
        ↓
  ┌─────────────────────────┐
  │  Memory Bank M          │ ← 과거 클린 샘플 피처 저장 (FIFO)
  │  클래스 센터 벡터 {w_k} │
  └─────────────────────────┘
        ↓
  P_clean(i) 계산 (Eq. 7)
        ↓
  sTRM/TRM 임계값 m 계산
        ↓
  B_clean 분리 (P_clean(i) > m)
        ↓
  손실 계산 L(B_clean)
  [L_batch + L_bank 또는 L_SoftTriple]
        ↓
  역전파 및 파라미터 업데이트
        ↓
  w_k 업데이트 (클린 샘플만)
```

**백본 네트워크**:
- CARS, CUB, CARS-98N: **BN-Inception** (512차원 출력)
- SOP, Food-101N: **ResNet-50** (128차원 출력)

---

### 2-4. 성능 향상

#### 대칭 노이즈(Symmetric Noise) - Table 1

| 방법 | CARS 50% | SOP 50% | CUB 50% |
|------|----------|---------|---------|
| MCL (기준) | 46.88 | 67.21 | 31.18 |
| **MCL + PRISM** | **72.93** | **72.85** | **56.03** |
| 향상폭 | +26.05% | +5.64% | +24.85% |

#### Small Cluster 노이즈 - Table 2

| 방법 | CARS 50% | SOP 50% | CUB 50% |
|------|----------|---------|---------|
| MCL (기준) | 36.43 | 68.71 | 41.58 |
| **MCL + PRISM** | **68.26** | **73.84** | **53.46** |

#### 실세계 노이즈 - Table 3

| 방법 | CARS-98N P@1 | Food-101N P@1 |
|------|-------------|--------------|
| MCL (기준) | 38.73 | 52.58 |
| **MCL + PRISM** | **57.95 (+49%)** | 52.47 (소폭 하락) |
| **Soft Triple + PRISM** | **64.81** | **64.46** |

---

### 2-5. 한계점

1. **노이즈율 R 사전 지정 필요**: 실세계에서 정확한 노이즈율을 알기 어려움. R 값에 따라 성능 편차 존재 (Table 6 참조).

2. **Food-101N에서 MCL+PRISM 미개선**: 특정 클래스에 고유한 open-set 노이즈가 존재하는 경우(예: apple-pie 클래스에만 오트밀 이미지), PRISM의 Memory-bank 기반 필터링이 효과적이지 않음. 이런 경우 **다중 센터(multi-center) proxy 기반 손실**이 더 적합.

3. **초기 학습 불안정**: 첫 번째 이터레이션에서 클래스 센터 벡터가 $\vec{0}$이므로, 초기에는 노이즈 필터링이 비활성화됨.

4. **Memory Bank 크기 의존성**: Memory Bank 크기를 훈련 데이터셋 전체 크기로 설정하여 메모리 요구량이 큼.

5. **클린 데이터셋 없이 동작하지만**, 노이즈율 추정을 위한 하이퍼파라미터 $R$ 튜닝에 일정한 사전 지식이 필요.

6. **균일 사전 확률 가정**: Eq.(2)에서 $P(Y=k)$를 균일 분포로 가정하나, 실세계 클래스 불균형 데이터에서는 성능 저하 가능.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 핵심 메커니즘과 일반화의 관계

PRISM의 일반화 성능 향상은 다음 세 가지 측면에서 분석할 수 있다:

#### (1) 메모리 기반 과거 피처 활용 (Time-averaged Feature Representation)

$$T(x_i, k) = \frac{1}{M_k} \sum_{(v_j, y_j) \in \mathcal{M},\, y_j=k} S(f(x_i), v_j)$$

이 메커니즘은 **과거 여러 이터레이션의 피처를 평균화**하여 노이즈를 탐지함으로써, 단일 배치의 편향(bias)에 과적합(overfit)되지 않도록 한다. 이는 앙상블 효과(ensemble effect)와 유사하여 **더 안정적이고 일반화된 클래스 표현**을 가능하게 한다.

Figure 2에서도 확인되듯, 노이즈 필터링 없이 학습하면 초기에 성능이 좋다가 이후 급격히 하락하는 **과적합 패턴**을 보이는 반면, PRISM은 안정적으로 수렴한다.

#### (2) Pclean(i)의 소프트맥스 구조 — 클래스 간 상대적 비교

단순히 같은 클래스 내 유사도만 보는 것이 아니라, **모든 클래스 대비 상대적 유사도**를 통해 클린 확률을 계산한다:

$$P_{\text{clean}}(i) = \frac{\exp(T(x_i, y_i))}{\sum_{k \in C} \exp(T(x_i, k))}$$

이는 **판별력(discriminability)** 을 직접 측정하는 것으로, unseen 클래스에 대한 일반화와 밀접히 연관된다. 클린 샘플만을 학습에 사용함으로써 **클래스 간 경계(decision boundary)** 가 더 명확하게 형성된다.

#### (3) sTRM의 안정화 효과

$$m = \frac{1}{\tau} \sum_{j=t-\tau}^{t} Q_j$$

단일 배치에서 임계값을 결정하는 TRM 대비, sTRM은 **슬라이딩 윈도우 평균**으로 임계값을 안정화한다. 이는 배치 샘플링의 편향에 의한 오필터링(over/under-filtering)을 줄여 **훈련 안정성과 일반화 성능을 동시에 향상**시킨다. Figure 3에서 모든 $\tau$ 값에서 sTRM이 TRM보다 우수한 성능을 보임이 확인된다.

#### (4) 클린 데이터셋에서의 성능 분석 (Supplementary Table 4)

| 데이터셋 | MCL (R=0) | R=2% | R=5% | R=10% |
|---------|----------|------|------|-------|
| CUB | 60.8 | 60.4 | 60.0 | 60.1 |
| CARS | 82.1 | 81.3 | 80.2 | 79.3 |
| SOP | 81.0 | **81.2** | **81.1** | 80.8 |

- **SOP에서 낮은 R값(2~5%)에서 클린 데이터 대비 성능 향상**: 원본 SOP에도 일정한 노이즈가 존재한다는 간접 증거
- CARS/CUB에서는 필터링이 과도해지면 성능 저하 → **R값의 신중한 설정이 일반화 성능의 핵심**

#### (5) 랜드마크 인식(Landmark Recognition) 실험

Supplementary의 Table 2~3에 따르면, Oxford/Babenko's Landmark 데이터셋에서 훈련하고 RParis에서 평가했을 때:

| 방법 | mAP Easy | mAP Medium | mAP Hard |
|------|----------|------------|----------|
| MCL | 60.8 | 47.9 | 24.8 |
| MCL+PRISM | **61.7** | **48.8** | **25.7** |

이는 PRISM이 **도메인(landmark recognition)과 데이터셋 규모(대규모)** 를 달리하는 상황에서도 일반화 성능을 향상시킴을 보여준다.

### 3-2. 일반화 성능의 이론적 해석

DML에서 일반화는 **훈련 클래스 → 미확인 테스트 클래스**에 대한 전이 능력을 의미한다. PRISM이 일반화를 향상시키는 이론적 근거:

1. **노이즈 레이블은 거짓 양성(false positive)/거짓 음성(false negative) 쌍을 생성** → 임베딩 공간 내 클래스 경계를 불명확하게 만듦
2. PRISM의 클린 샘플 필터링은 **올바른 pair 관계만을 학습에 사용** → 임베딩 공간의 구조적 정합성(structural coherence) 향상
3. Memory Bank를 통한 **시간적 앙상블**은 단일 이터레이션의 노이즈 그래디언트를 완화

---

## 4. 향후 연구에 미치는 영향 및 고려 사항

### 4-1. 연구에 미치는 영향

#### (A) DML + 노이즈 저항성의 교차 연구 영역 개척

본 논문은 DML에서 노이즈 레이블 문제를 **체계적으로 다룬 선구적 연구** 중 하나로, 이 교차 영역(noise-robust DML)에서의 후속 연구를 촉진한다.

#### (B) Small Cluster 노이즈 모델의 기여

기존 symmetric noise, pairwise noise 모델에 더해, **클러스터 기반 open-set 노이즈 모델**을 제안함으로써 보다 현실적인 노이즈 시뮬레이션 환경을 제공한다. 이는 벤치마크 표준으로 활용될 가능성이 있다.

#### (C) Memory Bank의 다목적 활용 방향 제시

기존에 더 많은 학습 쌍(pairs)을 확보하기 위해 사용되던 Memory Bank를 **노이즈 탐지**에도 활용하는 이중적 용도를 보여줌으로써, 유사한 아이디어의 확장 연구를 자극한다.

#### (D) Self-supervised Learning과의 접목 가능성

PRISM의 Memory Bank 메커니즘은 MoCo(Momentum Contrast)와 구조적으로 유사하다. 향후 **자기지도학습(Self-supervised Learning)** 에서 pseudo-label의 노이즈를 처리하는 데 PRISM의 아이디어가 접목될 수 있다.

---

### 4-2. 향후 연구 시 고려할 점

#### 🔴 방법론적 개선 방향

1. **적응적 노이즈율 추정 (Adaptive Noise Rate Estimation)**
   - 현재 $R$은 사전 정의된 하이퍼파라미터
   - 훈련 과정에서 $R$을 동적으로 추정하는 방법 필요 (예: GMM 기반 모델링, DIVIDE-MIX 방식 참고)

2. **클래스 불균형(Class Imbalance) 처리**
   - 현재 균일 사전확률 가정 → 불균형 데이터에서 편향 가능
   - 클래스별 가중치를 부여한 $P_{\text{clean}}(i)$ 설계 필요

3. **Semi-supervised 학습과의 통합**
   - 노이즈 샘플을 단순히 버리는 대신, **약지도(weakly-supervised)** 또는 **반지도(semi-supervised)** 방식으로 활용
   - 예: 노이즈 샘플에 소프트 레이블 부여 후 정규화 항으로 사용

4. **동적 Memory Bank 설계**
   - 고정 크기의 FIFO 방식은 클래스 불균형 상황에서 일부 클래스의 과소표현 문제 유발
   - 클래스별 균형을 맞춘 **class-balanced memory bank** 설계 필요

5. **다중 모달(Multi-modal) 확장**
   - 텍스트-이미지 쌍(예: CLIP 스타일)에서 노이즈 레이블 처리
   - $T(x_i, k)$의 유사도 함수를 다중 모달 맥락에 맞게 확장

#### 🟡 평가 및 벤치마크 관련

6. **더 다양한 노이즈 유형 실험**
   - 현실에서는 인스턴스 의존적(instance-dependent) 노이즈가 흔하나 본 논문에서는 다루지 않음
   - Instance-dependent noise에서의 PRISM 성능 검증 필요

7. **새로운 평가 메트릭 도입**
   - Precision@1, MAP@R 외에도 **Recall@K, NDCG** 등 다양한 검색 평가 지표에서의 검증 필요

#### 🟢 응용 분야 확장

8. **Few-shot Learning에서의 적용**
   - 클래스당 샘플 수가 매우 적은 환경(SOP의 평균 5.26장)에서의 노이즈 저항성 강화

9. **연속 학습(Continual Learning)에서의 노이즈 관리**
   - 새로운 클래스가 추가되는 환경에서 Memory Bank의 노이즈 누적 문제 해결

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ 아래 비교는 본 논문의 인용 목록과 제 학습 데이터 범위 내 확인 가능한 연구를 기반으로 작성합니다. 직접 열람하지 못한 논문의 세부 수치는 기재하지 않습니다.

### 5-1. 본 논문이 인용하거나 비교한 2020년 이후 연구

| 논문 | 발표연도/venue | 내용 | PRISM과의 관계 |
|------|----------------|------|----------------|
| **MCL** (Wang et al.) | CVPR 2020 | Cross-Batch Memory for Embedding Learning | PRISM의 베이스 방법. PRISM은 MCL에 노이즈 필터링 추가 |
| **Circle Loss** (Sun et al.) | CVPR 2020 | 페어 유사도 최적화 통합 관점 | 고노이즈 환경에서 성능 급락 (Table 1, 15.24%) |
| **Proxy Anchor Loss** (Kim et al.) | CVPR 2020 | 효율적 proxy 기반 DML | PRISM의 proxy 기반 손실 확장 가능성 |
| **ProxyNCA++** (Teh et al.) | ECCV 2020 | ProxyNCA 개선 | 비교 대상 포함 |
| **Group Loss** (Elezi et al.) | ECCV 2020 | 그룹 기반 DML | 비교 대상 포함 |
| **Smooth-AP** (Brown et al.) | ECCV 2020 | 리스트 기반 AP 최적화 | 대규모 검색 성능 향상 방향 |
| **Metric Learning Reality Check** (Musgrave et al.) | ECCV 2020 | DML 평가 재검토 | MAP@R 평가 기준 제공 |
| **Beyond Synthetic Noise** (Jiang et al.) | ICML 2020 | 제어된 노이즈 데이터셋 | 노이즈 모델 설계에 영향 |
| **Fewer Proxies** (Zhu et al.) | NeurIPS 2020 | 그래프 기반 proxy DML | 비교 관련 |

### 5-2. PRISM의 연구 포지셔닝

```
노이즈 저항성
    ↑
    │  PRISM ★ (DML + 노이즈 저항)
    │
    │  Co-teaching/F-correction (분류 + 노이즈 저항)
    │
    └─────────────────────────────────────────→ DML 성능
              MCL, Circle Loss, Proxy Anchor
```

- **노이즈 분류 방법들** (Co-teaching, F-correction): DML 태스크에 직접 적용하면 성능 부족
- **순수 DML 방법들** (MCL, Circle Loss): 노이즈에 취약, 특히 50% 노이즈에서 성능 급락
- **PRISM**: 두 강점을 결합하여 고노이즈 환경에서 안정적인 DML 달성

### 5-3. 향후 관련 연구 방향 (2021년 이후 예상 트렌드)

> ⚠️ 아래는 논문의 연구 방향성과 일반적인 ML 트렌드를 기반으로 한 예측으로, 구체적 논문 수치는 직접 검증하지 못하였습니다.

1. **Vision-Language 모델(CLIP 등)과 DML의 결합**: 대규모 웹 데이터의 노이즈를 처리하는 DML 연구 증가 예상
2. **자기지도학습과 노이즈 저항성의 통합**: pseudo-label 기반 자기지도 DML에서 PRISM 류의 필터링 적용
3. **그래프 신경망(GNN) 기반 노이즈 탐지**: 샘플 간 관계를 그래프로 모델링하여 더 정교한 노이즈 탐지

---

## 📝 종합 요약

**PRISM**은 DML에서 노이즈 레이블 문제를 해결하기 위해 Memory Bank와 확률적 랭킹 기반 필터링을 결합한 독창적 방법이다. 핵심 강점은:

1. **닭-달걀 문제 해결**: 과거 피처를 활용한 온라인 필터링으로 점진적 해결
2. **효율성**: 클래스 센터 벡터 도입으로 추가 계산 비용 최소화 (~6% 추가)
3. **범용성**: 다양한 DML 손실함수(MCL, SoftTriple)와 호환
4. **일반화 성능**: 메모리 기반 시간적 앙상블과 판별적 확률 계산으로 unseen 클래스에 대한 강인한 임베딩 학습

주요 한계인 노이즈율 $R$ 사전 지정과 클래스 불균형 처리는 향후 연구의 핵심 과제이며, 이 연구는 **노이즈 저항 DML이라는 중요한 연구 방향을 선도**하였다는 점에서 학술적 기여가 크다.
