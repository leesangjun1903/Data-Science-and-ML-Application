# MOST: Multi-Source Domain Adaptation via Optimal Transport for Student-Teacher Learning 

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

MOST는 **다중 소스 도메인 적응(Multi-Source Domain Adaptation, MSDA)** 문제를 **최적 수송(Optimal Transport, OT) 이론**과 **모방 학습(Imitation Learning)** 을 결합하여 해결하는 새로운 프레임워크를 제안합니다.

핵심 아이디어는 다음과 같습니다:

> 여러 소스 도메인의 전문가(domain experts)들을 결합한 **교사 분류기(Teacher Classifier)** 를 구성하고, 타겟 도메인에서 동작하는 **학생 분류기(Student Classifier)** 가 OT 기반으로 교사를 모방(imitate)함으로써 도메인 간 지식 이전을 달성한다.

### 주요 기여

| 기여 항목 | 내용 |
|-----------|------|
| **이론적 기여** | OT 기반 모방 학습 이론을 도메인 적응에 엄밀하게 적용 (Proposition 2, Theorem 3, 4, 6) |
| **방법론적 기여** | Teacher-Student 구조의 MSDA 모델 제안 (데이터 이동 + 레이블 이동 동시 완화) |
| **실험적 기여** | Digits-five, Office-Caltech10, Office-31에서 당시 SOTA 달성 |
| **확장성** | 제안 패러다임은 강화학습 등 다른 학습 문제에도 적용 가능 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**Multi-Source Domain Adaptation (MSDA)** 에는 두 가지 핵심 난제가 존재합니다:

1. **데이터 이동(Data Shift)**: $\mathcal{X}^S$와 $\mathcal{X}^T$ 사이의 입력 분포 불일치
2. **레이블 이동(Label Shift)**: $p^S(y|x) \neq p^T(y|x)$ 로 인한 조건부 분포 불일치

기존 단일 소스 DA 방법들은 이 두 문제를 동시에 효과적으로 다루지 못했으며, 다중 소스를 어떻게 통합할지에 대한 이론적 근거가 부족했습니다.

---

### 2.2 제안하는 방법 (수식 포함)

#### 2.2.1 최적 수송 기본 이론

**Kantorovich Problem (KP)**:

$$\mathcal{K}_d(\mathbb{Q}, \mathbb{P}) := \min_{\gamma \in \Gamma(\mathbb{Q}, \mathbb{P})} \mathbb{E}_{(\mathbf{x}, \mathbf{y}) \sim \gamma} [d(\mathbf{x}, \mathbf{y})]$$

**엔트로픽 정규화 (Entropic Regularized OT)**:

```math
\mathcal{W}_d^\epsilon(\mathbb{Q}, \mathbb{P}) := \min_{\gamma \in \Gamma(\mathbb{Q}, \mathbb{P})} \left\{ \mathbb{E}_{(\mathbf{x},\mathbf{y})\sim\gamma} [d(\mathbf{x}, \mathbf{y})] + \epsilon D_{KL}(\gamma \| \mathbb{Q} \otimes \mathbb{P}) \right\}
```

**쌍대 형식 (Dual Form)**:

```math
\mathcal{W}_d^\epsilon(\mathbb{Q}, \mathbb{P}) = \max_{\phi} \left\{ \mathbb{E}_\mathbb{Q}[\phi_\epsilon^c(\mathbf{x})] + \mathbb{E}_\mathbb{P}[\phi(\mathbf{y})] \right\}
```

여기서 

```math
\phi_\epsilon^c(\mathbf{x}) := -\epsilon \log \mathbb{E}_\mathbb{P}\left[\exp\left\{\frac{-d(\mathbf{x},\mathbf{y})+\phi(\mathbf{y})}{\epsilon}\right\}\right]
```

**도메인 간 비용 함수**:

```math
d(\mathbf{z}_1, \mathbf{z}_2) := \lambda d_\mathcal{X}(\mathbf{x}_1, \mathbf{x}_2) + d_\mathcal{Y}(y_1^\triangle, y_2^\triangle)
```

이 비용 함수는 **특징 공간의 거리**와 **예측 레이블 간의 차이** 를 동시에 고려합니다.

---

#### 2.2.2 OT 기반 모방 학습 (Proposition 2)

도메인 $\mathcal{X}^A$ 의 교사 $h^A$를 모방하도록 $\mathcal{X}^B$ 의 학생 $h^B$ 를 학습시키기 위해, 다음 등가 관계를 활용합니다:

```math
\mathcal{W}_d(\mathbb{P}_{A,h^A}, \mathbb{P}_{B,h^B}) = \min_{L: L_\#\mathbb{P}^A = \mathbb{P}^B} \mathbb{E}_{\mathbf{x} \sim \mathbb{P}^A}\left[\lambda d_\mathcal{X}(\mathbf{x}, L(\mathbf{x})) + d_\mathcal{Y}(h^A(\mathbf{x}), h^B(L(\mathbf{x})))\right]
```

```math
= \min_{H: H_\#\mathbb{P}^B = \mathbb{P}^A} \mathbb{E}_{\mathbf{x} \sim \mathbb{P}^B}\left[\lambda d_\mathcal{X}(\mathbf{x}, H(\mathbf{x})) + d_\mathcal{Y}(h^B(\mathbf{x}), h^A(H(\mathbf{x})))\right]
```

이는 최적 수송 맵 $H^*$가 $\mathbb{P}^B$ 의 각 샘플을 $\mathbb{P}^A$ 공간에서 가장 가까운 대응점으로 이동시켜 $h^B$ 가 $h^A$ 의 예측을 모방하게 함을 의미합니다.

---

#### 2.2.3 다중 소스 전문가 교사 구성 (Section 5.2)

혼합 분포 $\mathbb{P}\_\pi^S := \sum_{k=1}^K \pi_k \mathbb{P}_k^S$ 위에서 동작하는 교사를 다음과 같이 구성합니다:

$$h^S(\mathbf{x}, y) = \sum_{k=1}^K \frac{\pi_k p_k^S(\mathbf{x}, y)}{\sum_{j=1}^K \pi_j p_j^S(\mathbf{x}, y)} h_k^S(\mathbf{x}, y)$$

**Theorem 4** 에 의해 각 전문가가 $\epsilon$-적격 분류기이면 교사도 $\epsilon$-적격 분류기:

$$\mathcal{L}(h^S, f^S, \mathbb{P}_\pi^S) \leq \max_{1 \leq k \leq K} \mathcal{L}(h_k^S, f_k^S, \mathbb{P}_k^S)$$

실제 구현에서는 소스 도메인 판별기 $\mathcal{C}$ 를 통해 가중치를 추정:

$$h^S(\mathbf{x}, y) = \sum_{k=1}^K \mathcal{C}(\mathbf{x}, y, k) h_k^S(\mathbf{x}, y)$$

---

#### 2.2.4 학생 학습 이론 (Section 5.3)

학생 분류기 $h^T$ 를 학습하기 위한 상한 분해:

$$\mathcal{W}_d(\mathbb{P}_{T,h^T}, \mathbb{P}_{T,f^T}) \leq \underbrace{\mathcal{W}_d(\mathbb{P}_{T,h^T}, \mathbb{P}_{S,h^S}^\pi)}_{\text{모방 항}} + \underbrace{\mathcal{L}(h^S, f^S, \mathbb{P}_\pi^S)}_{\text{교사 품질}} + \underbrace{\mathcal{W}_d(\mathbb{P}_{S,f^S}^\pi, \mathbb{P}_{T,f^T})}_{\text{자연 도메인 이동 (상수)}}$$

데이터 이동을 줄이기 위해 생성기 $G^S$, $G^T$ 를 사용하여 공통 잠재 공간으로 매핑:

```math
\min_{h^T, G^T} \left\{ \mathcal{L}(h^S \circ G^S, f^S, \mathbb{P}_\pi^S) + \mathcal{W}_d(\mathbb{Q}_{T,h^T}, \mathbb{Q}_{S,h^S}^\pi) \right\}
```

---

#### 2.2.5 최종 학습 목적 함수

$$\mathcal{L} = \sum_{k=1}^K \mathcal{L}_k^{de} + \mathcal{L}^\mathcal{C} + \alpha \mathcal{L}^{WS} + \beta \mathcal{L}^{pl} + \gamma \mathcal{L}^{clus} $$

각 항의 의미:

| 손실 항 | 수식 | 역할 |
|---------|------|------|
| $\mathcal{L}_k^{de}$ | $\mathbb{E}_{(\mathbf{x},y)\sim\mathcal{D}_k^S}[CE(h_k^S(G^S(\mathbf{x})), y)]$ | 소스 도메인 전문가 학습 |
| $\mathcal{L}^\mathcal{C}$ | $\mathbb{E}_{(\mathbf{x},y,t)\sim\mathcal{D}}[CE(\mathcal{C}(\mathbf{x},y),t)]$ | 소스 도메인 판별기 학습 |
| $\mathcal{L}^{WS}$ | $\mathcal{W}\_d^\epsilon(\mathbb{Q}_{T,h^T}, \mathbb{Q}\_{S,h^S}^\pi)$ | OT 기반 모방 학습 |
| $\mathcal{L}^{pl}$ | $\mathbb{E}\_{\mathbf{x}\sim\mathbb{P}_\pi^S, \mathbb{P}^T}[CE(h^S(G^S(\mathbf{x})), h^T(G^T(\mathbf{x})))]$ | 의사 레이블 모방 |
| $\mathcal{L}^{clus}$ | $\mathcal{L}^{ent} + \mathcal{L}^{vat}$ | 클러스터링 가정 강화 |

**WS 항의 구체적 형식**:

```math
\min_{h^T, G^T} \mathcal{L}^{WS} = \min_{h^T, G^T} \max_\phi \left\{ \mathbb{E}_{\mathbb{P}^T}\!\left[-\epsilon\log\mathbb{E}_{\mathbb{P}_\pi^S}\!\left[\exp\!\left\{\frac{1}{\epsilon}\gamma(\mathbf{x}^S, \mathbf{x}^T)\right\}\right]\right] + \mathbb{E}_{\mathbb{P}_\pi^S}\!\left[\phi(G^S(\mathbf{x}^S))\right] \right\}
```

여기서:

$$\gamma(\mathbf{x}^S, \mathbf{x}^T) = \phi(G^S(\mathbf{x}^S)) - d(G^S(\mathbf{x}^S), G^T(\mathbf{x}^T))$$

$$d(G^S(\mathbf{x}^S), G^T(\mathbf{x}^T)) = d_\mathcal{Y}(h^T(G^T(\mathbf{x}^T)), h^S(G^S(\mathbf{x}^S))) + \lambda\|G^T(\mathbf{x}^T) - G^S(\mathbf{x}^S)\|$$

---

### 2.3 모델 구조

```
┌─────────────────────────────────────────────────────────┐
│                     MOST 구조                            │
│                                                          │
│  Source Domain 1 ──→ h₁ˢ ─┐                            │
│  Source Domain 2 ──→ h₂ˢ ─┤──→ Σ (가중합) ──→ Teacher hˢ│
│  Source Domain K ──→ hₖˢ ─┘     ↑                      │
│         ↑                    C (판별기)                  │
│         Gˢ (공유 생성기)                                 │
│                                                          │
│  Target Domain ──→ Gᵀ ──→ hᵀ (Student)                 │
│                            ↕ (WS Distance 최소화)       │
│                         Teacher hˢ                       │
└─────────────────────────────────────────────────────────┘
```

**주요 구성 요소**:
- **$G^S$, $G^T$**: 특징 추출 생성기 (공유 가중치 옵션)
- **$h_k^S$ ($k=1,...,K$)**: 소스 전문가 분류기
- **$\mathcal{C}$**: 소스 도메인 판별기 (가중치 계산용)
- **$h^T$**: 타겟 도메인 학생 분류기
- **$\phi$ (Kantorovich Potential Network)**: WS 거리 계산용 신경망

---

### 2.4 성능 향상

#### Digits-Five 결과 (Table 1)

| 방법 | →mm | →mt | →up | →sv | →sy | **Avg** |
|------|-----|-----|-----|-----|-----|---------|
| M³SDA [Peng et al., 2019] | 72.8 | 98.4 | 96.1 | 81.3 | 89.6 | 87.7 |
| LtC-MSDA [Wang et al., 2020] | 85.6 | 99.0 | 98.3 | 83.2 | 93.0 | 91.8 |
| **MOST (ours)** | **91.5** | **99.6** | **98.4** | **90.9** | **96.4** | **95.4** |

→ 이전 SOTA 대비 **+3.6%p** 향상 (Avg 기준)

#### Office-Caltech10 (Table 2)

| 방법 | →W | →D | →C | →A | **Avg** |
|------|----|----|----|----|---------|
| M³SDA [Peng et al., 2019] | 99.5 | 99.2 | 92.2 | 94.5 | 96.4 |
| **MOST (ours)** | **100** | **100** | **96.0** | **96.4** | **98.1** |

→ **+1.7%p** 향상

#### Office-31 (Table 3)

| 방법 | →D | →W | →A | **Avg** |
|------|----|----|----|---------|
| LtC-MSDA [Wang et al., 2020] | 99.6 | 97.2 | 56.9 | 84.6 |
| **MOST (ours)** | **100** | **98.7** | **60.6** | **86.4** |

→ **+1.8%p** 향상, 어려운 →A 작업에서 **+3.7%p**

---

### 2.5 한계점

논문에서 명시적으로 언급된 한계와 분석을 통해 도출된 한계를 구분하여 제시합니다.

**논문 내 확인 가능한 한계**:

1. **계산 복잡도**: Kantorovich Potential Network $\phi$ 를 각 미니배치마다 여러 번 업데이트해야 하므로 GAN과 유사한 학습 불안정성이 존재할 수 있음
2. **하이퍼파라미터 민감성**: $\alpha, \beta, \gamma, \epsilon, \lambda, \theta$ 등 다수의 트레이드오프 파라미터 존재
3. **벤치마크 한정**: 주로 이미지 분류 벤치마크에서만 평가됨 (NLP, 시계열 등 미평가)
4. **소스 도메인 수 확장성**: 소스 도메인 수 $K$가 매우 클 경우의 확장성 분석 미흡

**분석을 통해 도출된 한계**:

5. **레이블된 타겟 없음 가정**: 완전 비지도 설정만 다루며, 부분 지도 설정 미지원
6. **소스 도메인 품질 의존성**: 개별 전문가의 품질에 강하게 의존 ($\max_k \mathcal{L}(h_k^S)$ 상한)

---

## 3. 일반화 성능 향상 가능성

### 3.1 이론적 일반화 보장 (Theorem 6)

MOST의 가장 강력한 일반화 근거는 **Theorem 6** 의 상한 분해입니다:

```math
\min_{h^T, G^T} \mathcal{W}_d\!\left(\mathbb{P}_{T,h^T}^{G^T}, \mathbb{P}_{T,f^T}^{G^T}\right) \leq \mathcal{L}(h^*_S \circ G^*_S, f^S, \mathbb{P}_\pi^S) + \mathcal{W}_d\!\left(\mathbb{P}_{S,f^S_*}^{G^S_*}, \mathbb{P}_{T,f^T_*}^{G^T_*}\right)
```

이 부등식의 의미:

| 항 | 의미 | 최소화 방법 |
|----|------|------------|
| $\mathcal{L}(h^\*\_S \circ G^*\_S, f^S, \mathbb{P}_\pi^S)$ | 교사의 소스 도메인 오류 | 양질의 도메인 전문가 학습 |
| $\mathcal{W}\_d(\mathbb{P}\_{S,f^S_*}^{G^S_\*}, \mathbb{P}\_{T,f^T_\*}^{G^T_\*})$ | 공통 공간에서의 자연 도메인 이동 | 생성기 $G^S, G^T$ 의 품질 향상 |

즉, **교사의 품질이 보장되고 도메인 이동이 최소화되면**, 타겟 도메인에서의 일반화 오류가 이론적으로 줄어듦을 보장합니다.

### 3.2 데이터 이동과 레이블 이동의 동시 완화

**기존 방법의 한계**: 대부분의 DA 방법은 데이터 이동만 다루거나, 레이블 이동을 별도 처리

**MOST의 접근**: OT 비용 함수 $d(\mathbf{z}\_1, \mathbf{z}\_2) = \lambda d_\mathcal{X}(\mathbf{x}\_1, \mathbf{x}\_2) + d_\mathcal{Y}(y_1^\triangle, y_2^\triangle)$ 가 두 이동을 **단일 프레임워크에서 동시 처리**

클러스터링 관점에서: 최적 수송 맵에 의해 $G^T(\mathbf{x}^T)$ 는 **같은 레이블을 가진** $G^S(\mathbf{x}^S)$ 의 클러스터로 이동 → 레이블 이동 자동 완화

### 3.3 클러스터링 가정을 통한 추가 일반화

$$\mathcal{L}^{clus} = \mathcal{L}^{ent} + \mathcal{L}^{vat}$$

- **$\mathcal{L}^{ent}$**: 타겟 예측의 엔트로피 최소화 → 결정 경계가 저밀도 영역에 위치하도록 유도
- **$\mathcal{L}^{vat}$**: 가상 적대적 훈련으로 지역 Lipschitz 조건 강화

Ablation study에서 $\mathcal{L}^{pl} + \mathcal{L}^{WS}$ 만으로도 SOTA 달성, $\mathcal{L}^{clus}$ 추가 시 추가 향상 확인 (Digits-five: 94.2% → 96.0%)

### 3.4 혼합 분포 전략

균일 혼합 $\pi = [\frac{1}{K}, ..., \frac{1}{K}]$ 과 데이터 수 비례 혼합이 유사한 성능을 보임 → 혼합 전략에 대한 **강건성(robustness)** 이 높음을 의미

---

## 4. 미래 연구에 대한 영향 및 고려사항

### 4.1 앞으로의 연구에 미치는 영향

#### 4.1.1 OT + 지식 증류의 통합 패러다임

MOST는 **최적 수송 이론**과 **Teacher-Student 학습**을 결합하는 새로운 연구 방향을 제시합니다. 이후 연구인 **TIDOT** (Nguyen et al., IJCAI 2021)에서 이미 이 방향이 계속 발전되고 있습니다. 향후 다음 분야로 확장될 가능성이 높습니다:

- **연속 도메인 적응** (Continual Domain Adaptation)
- **테스트 시간 적응** (Test-Time Adaptation)
- **페더레이티드 학습** (Federated Learning with heterogeneous data)

#### 4.1.2 레이블 이동을 포함한 이론적 프레임워크

기존 DA 이론 (Ben-David et al., 2010)이 주로 데이터 이동에 집중한 반면, MOST는 **데이터 + 레이블 이동을 동시에 다루는 이론적 틀**을 제공합니다. 이는 보다 현실적인 도메인 이동 시나리오 연구의 토대가 됩니다.

#### 4.1.3 일반화 가능한 모방 학습 프레임워크

논문에서 "우리의 일반 패러다임은 강화학습을 포함한 다양한 학습 문제에 적용 가능"하다고 명시하고 있으며, 이는 다음 연구 방향을 시사합니다:

- **로봇 학습**: 시뮬레이션(소스)→실제 환경(타겟)의 Sim-to-Real 전이
- **NLP 도메인 적응**: 텍스트 도메인 간 OT 기반 정렬

---

### 4.2 앞으로 연구 시 고려할 점

#### 4.2.1 계산 효율성 개선

**현재 문제**: WS 거리 계산을 위한 Kantorovich Potential Network의 반복 업데이트는 계산 비용이 높음

**연구 방향**:
- **Sliced Wasserstein Distance** 활용으로 계산 복잡도 $O(n \log n)$ 으로 감소
- **Mini-batch OT** 기법 (Fatras et al., 2021)과의 통합
- **Sinkhorn 알고리즘** 기반 효율적 OT 계산

#### 4.2.2 소스 도메인 품질 불균일성 처리

**현재 문제**: Theorem 4의 상한이 $\max_k \mathcal{L}(h_k^S)$ 이므로, **단 하나의 저품질 소스 도메인**이 전체 성능을 저하시킬 수 있음

**연구 방향**:
- **부정적 전이(Negative Transfer) 탐지** 및 자동 소스 선택 메커니즘
- **신뢰도 기반 가중치 조정** (Reliable OT, Xu et al., 2020c와 유사한 접근)

#### 4.2.3 확장 가능한 소스 도메인 처리

**현재 문제**: $K$ 개의 소스 도메인이 증가할수록 도메인 판별기의 복잡도가 $O(K)$ 로 증가

**연구 방향**:
- **계층적 도메인 집계** (Hierarchical Domain Aggregation)
- **메타 학습(Meta-Learning)** 기반 소스 도메인 통합

#### 4.2.4 Partial 및 Open-Set DA로의 확장

**현재 문제**: 소스-타겟 간 레이블 공간이 완전히 일치한다고 가정

**연구 방향**:
- **Partial DA**: 타겟에 소스의 일부 클래스만 존재
- **Open-Set DA**: 타겟에 미지의 클래스 존재
- OT 기반의 **이상치 탐지(Outlier Detection)** 통합

#### 4.2.5 사전 학습 모델과의 통합

**현재 문제**: MOST는 ResNet, AlexNet 등 중규모 백본 기반으로 실험

**연구 방향**:
- **Vision-Language Model (CLIP, ViT)** 과의 통합
- Few-shot / Zero-shot 설정에서 OT 기반 DA 적용

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의**: 아래 비교는 논문 내 인용 문헌 및 제가 보유한 학습 데이터(2023년 초까지)를 기반으로 합니다. 2021년 이후 발표된 일부 논문에 대해 정확도를 100% 확신하기 어려운 경우, 해당 내용을 명시합니다.

### 5.1 논문 내 인용된 2020년 이후 연구

| 논문 | 핵심 방법 | MOST와의 차이 |
|------|-----------|--------------|
| **LtC-MSDA** (Wang et al., ECCV 2020) | Graph Convolutional Network으로 카테고리 수준 도메인 정렬 | 그래프 기반 vs OT 기반 |
| **MDDA** (Zhao et al., AAAI 2020) | 생성기와 분류기를 별도로 파인튜닝, 도메인 가중치로 예측 집계 | 분리 학습 vs 동시 학습 |
| **TIDOT** (Nguyen et al., IJCAI 2021) | MOST의 직접적 후속 연구: Teacher Imitation with OT | MOST의 개선 버전 |
| **LAMDA** (Le et al., ICML 2021) | 레이블 매칭 기반 깊은 도메인 적응 | 단일 소스 DA에 집중 |

### 5.2 MOST 이후 주목할 만한 연구 방향 (학습 데이터 기반, 정확도 보통 수준)

**⚠️ 아래 내용은 제 학습 데이터에서 파악된 일반적 연구 트렌드이며, 특정 논문 제목/저자/수치의 정확성에 대한 완전한 확신이 어렵습니다. 직접 ArXiv나 Google Scholar에서 확인을 권장합니다.**

#### 5.2.1 Vision Foundation Model 기반 DA (2022-2023)

- **CLIP** (Radford et al., 2021)을 활용한 제로샷 도메인 적응 연구들이 등장
- 대규모 사전학습 표현이 도메인 이동에 강건함을 실험적으로 확인
- **MOST 관련성**: 교사 모델로 Foundation Model 사용 가능성 시사

#### 5.2.2 테스트 시간 적응 (Test-Time Adaptation, TTA)

- **TENT** (Wang et al., ICLR 2021): 테스트 시 배치 정규화 통계 조정
- OT 기반 TTA 연구로 발전 가능
- **MOST와의 차이**: MOST는 학습 시 타겟 데이터 필요, TTA는 배포 후 적응

#### 5.2.3 Source-Free Domain Adaptation

- 소스 데이터 없이 적응하는 프라이버시 보존 설정
- **MOST의 한계**: 학습 중 소스 데이터 접근이 필수적

### 5.3 종합 비교 표

| 방법 | 이론적 보장 | 레이블 이동 처리 | 계산 효율성 | 다중 소스 | 소스 도메인 불필요 |
|------|------------|----------------|------------|---------|----------------|
| DANN (2015) | 부분적 | ✗ | 높음 | ✗ | ✗ |
| DeepJDOT (2018) | OT 기반 | ✗ | 중간 | ✗ | ✗ |
| M³SDA (2019) | 통계 모멘트 | 부분적 | 높음 | ✓ | ✗ |
| LtC-MSDA (2020) | 그래프 기반 | ✗ | 중간 | ✓ | ✗ |
| **MOST (2021)** | **OT 엄밀** | **✓** | **중간** | **✓** | **✗** |
| TENT (2021) | ✗ | ✗ | 매우 높음 | ✗ | **✓** |

---

## 참고 자료

### 직접 참고한 자료
- **원문 논문**: Nguyen, T., Le, T., Zhao, H., Tran, Q. H., Nguyen, T., & Phung, D. (2021). "MOST: Multi-Source Domain Adaptation via Optimal Transport for Student-Teacher Learning." *Proceedings of the 37th Conference on Uncertainty in Artificial Intelligence (UAI 2021)*, PMLR 161:225–235.

### 논문 내 인용 참고문헌 (주요)
- Villani, C. (2008). *Optimal Transport: Old and New*. Springer.
- Santambrogio, F. (2015). *Optimal Transport for Applied Mathematicians*. Birkhäuser.
- Genevay, A., Cuturi, M., Peyré, G., & Bach, F. (2016). Stochastic optimization for large-scale optimal transport. *NIPS*.
- Courty, N., Flamary, R., Tuia, D., & Rakotomamonjy, A. (2017). Optimal transport for domain adaptation. *IEEE TPAMI*.
- Damodaran, B. B., et al. (2018). DeepJDOT. *ECCV*.
- Peng, X., et al. (2019). Moment matching for multi-source domain adaptation. *ICCV*.
- Wang, H., et al. (2020). Learning to combine: Knowledge aggregation for multi-source domain adaptation. *ECCV*.
- Zhao, S., et al. (2020). Multi-source distilling domain adaptation. *AAAI*.
- Mansour, Y., Mohri, M., & Rostamizadeh, A. (2009). Domain adaptation with multiple sources. *NIPS*.
- Hoffman, J., Mohri, M., & Zhang, N. (2018). Algorithms and theory for multiple-source adaptation. *NeurIPS*.
- Nguyen, T., et al. (2021). TIDOT: A teacher imitation learning approach for domain adaptation with optimal transport. *IJCAI*.
- Le, T., et al. (2021). LAMDA: Label matching deep domain adaptation. *ICML*.
- Miyato, T., et al. (2019). Virtual adversarial training. *IEEE TPAMI*.
- Redko, I., et al. (2019). Optimal transport for multi-source domain adaptation under target shift. *AISTATS*.
