# Reducing the Covariate Shift by Mirror Samples in Cross Domain Alignment

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

이 논문은 **비지도 도메인 적응(Unsupervised Domain Adaptation, UDA)** 에서 기존의 공변량 이동(Covariate Shift) 감소 방법들이 가진 근본적인 딜레마를 지적합니다.

> **핵심 딜레마**: 샘플링 편향(Sampling Bias)이 존재하는 데이터셋에서 공변량 이동을 줄이기 위해 도메인 정렬을 수행하면, 오히려 공변량 이동의 가정 조건인 $p^s(y|x) = p^t(y|x)$를 위반하게 된다.

### 주요 기여

| 기여 항목 | 설명 |
|-----------|------|
| **딜레마 발굴** | 샘플 기반 공변량 이동 감소의 내재적 모순 이론적 규명 |
| **Mirror Sample 개념 제안** | 상대 도메인에서의 동등 샘플(equivalent sample) 개념 도입 |
| **LNA + ER 방법론** | Local Neighbor Approximation과 Equivalence Regularization을 결합한 Mirror Sample 생성 방법 |
| **Mirror Loss 설계** | Mirror Pair 간 정렬을 위한 KL-Divergence 기반 손실함수 |
| **이론적 보증** | Proposition 1, 2를 통한 점근적 수렴 성질 증명 |
| **SOTA 성능** | Office-31, Office-Home, ImageCLEF, VisDA2017에서 최고 성능 달성 |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

#### 공변량 이동의 정의

소스 도메인 $p^s(x, y)$와 타겟 도메인 $p^t(x, y)$가 주어질 때, 공변량 이동(Covariate Shift)은 다음을 의미합니다:

$$p^s(x) \neq p^t(x), \quad \text{but} \quad p^s(y|x) = p^t(y|x)$$

#### 샘플링 편향과 딜레마

실제 데이터셋은 내재 분포(underlying distribution)로부터 **편향된 샘플링** 결과입니다. 예를 들어 Office-Home의 "Bed" 카테고리에서:
- Art 도메인: "Bunk Bed" 비율 ≈ 0%
- Clipart 도메인: "Bunk Bed" 비율 ≈ 7.1%

이러한 편향된 샘플에 대해 도메인 정렬(moment matching, prototype alignment 등)을 수행하면:

$$p^s(y|\tilde{x}) \neq p^t(y|\tilde{x})$$

가 성립하게 되어, 정렬된 공간 $\tilde{x}$에서 조건부 분포 동일성 가정이 **위반**됩니다.

---

### 2.2 제안하는 방법 (수식 포함)

#### (A) Mirror Sample의 정의

**Mirror Sample**은 상대 도메인에서의 동등한 샘플을 의미합니다. Optimal Transport 관점에서:

```math
\mathbb{T}^s_\# p^s(x^s) = \mathbb{T}^t_\# p^t(x^t)
```

를 만족하는 경우 $x^s \in D^S$와 $x^t \in D^T$는 서로의 Mirror가 됩니다.

#### (B) Local Neighbor Approximation (LNA)

타겟 샘플 $x^t_j$의 Mirror Sample을 생성하기 위해, 소스 도메인에서 가장 가까운 $k$개의 이웃을 찾습니다:

$$\tilde{X}^S(x^t_j) = \arg \top^k_{x \in X^S} d(x, x^t_j) \tag{1}$$

그 후, Mirror Sample은 다음과 같이 가중 합산으로 추정합니다:

$$\tilde{x}^s(x^t_j) = \sum_{x \in \tilde{X}^S(x^t_j)} \omega(x, x^t_j) x \tag{2}$$

여기서 $\omega(x, x^t_j)$는 $1/k$ 또는 거리에 반비례하는 가중치입니다.

#### (C) Equivalence Regularization (ER)

클래스 $c$에 대한 중심(anchor) $\mu^t_c$를 기준으로, 타겟 샘플의 상대적 위치 벡터를 정의합니다:

$$q^t_c(x^t_j) = \frac{\exp\{-d(x^t_j, \mu^t_c)\}}{\sum_{c=1}^{M} \exp\{-d(x^t_j, \mu^t_c)\}} \tag{3}$$

$$q^t(x^t_j) = \left[q^t_1(x^t_j), q^t_2(x^t_j), \cdots, q^t_M(x^t_j)\right] \tag{4}$$

LNA로 생성된 Mirror Sample $\tilde{x}^s(x^t_j)$의 소스 도메인 앵커 대비 상대 위치:

$$q^s_c(\tilde{x}^s(x^t_j)) = \frac{\exp\{-d(\tilde{x}^s(x^t_j), \mu^s_c)\}}{\sum_{c=1}^{M} \exp\{-d(\tilde{x}^s(x^t_j), \mu^s_c)\}} \tag{5}$$

$$q^s(\tilde{x}^s(x^t_j)) = \left[q^s_1(\tilde{x}^s(x^t_j)), q^s_2(\tilde{x}^s(x^t_j)), \cdots, q^s_M(\tilde{x}^s(x^t_j))\right] \tag{6}$$

#### (D) Mirror Loss

Mirror Pair 간의 KL-Divergence를 최소화하는 Mirror Loss:

$$\mathcal{L}_{mr,x} = \frac{1}{n^t}\sum_{j=1}^{n^t} \text{KL}\left(q^t(\tilde{x}^s(x^t_j)) \| q^t(x^t_j)\right) + \frac{1}{n^s}\sum_{i=1}^{n^s} \text{KL}\left(q^s(\tilde{x}^t(x^s_i)) \| q^s(x^s_i)\right) \tag{7}$$

- **첫 번째 항**: 타겟 샘플 $x^t_j$와 그 Mirror $\tilde{x}^s(x^t_j)$의 상대 위치 정렬
- **두 번째 항**: 소스 샘플 $x^s_i$와 그 Mirror $\tilde{x}^t(x^s_i)$의 상대 위치 정렬

#### (E) 전체 손실 함수

$$\mathcal{L} = \mathcal{L}^s + \mathcal{L}^t + \gamma(\mathcal{L}_{mr,f} + \mathcal{L}_{mr,g}) \tag{8}$$

- $\mathcal{L}^s$: 소스 도메인 Cross-Entropy Loss

$$\mathcal{L}^s = -\frac{1}{n^s}\sum_{i=1}^{n^s}\sum_{c=1}^{M} \mathbb{I}(y^s_i = c)\log p^s_{i,c}$$

- $\mathcal{L}^t$: 타겟 도메인 비지도 클러스터링 Loss (보조 분포 $z^t_i$를 soft pseudo label로 활용)

$$\mathcal{L}^t = -\frac{1}{n^t}\sum_{i=1}^{n^t} z^t_i \cdot \log p^t_i, \quad z^t_{i,c} \propto \frac{p^t_{i,c}}{\left(\sum_{i=1}^{n^t} p^t_{i,c}\right)^{1/2}}$$

- $\gamma$: Mirror Loss의 가중치 하이퍼파라미터

---

### 2.3 모델 구조

```
[Source Data] ─────────────────────────────────────────┐
                     ┌─────────────┐                   │
[Target Data] ──────►│  Backbone   │──► f ──► FC ──► g ──► Classifier
                     │  (ResNet50/ │         │         │
                     │  ResNet101) │         │         │
                     └─────────────┘         │         │
                                             │         │
                                    ┌────────▼─────────▼────────┐
                                    │  LNA (Local Neighbor       │
                                    │  Approximation)            │
                                    │  → Mirror Sample 생성       │
                                    └────────────┬───────────────┘
                                                 │
                                    ┌────────────▼───────────────┐
                                    │  ER (Equivalence           │
                                    │  Regularization)           │
                                    │  → Mirror Loss 계산         │
                                    │  L_mr,f + L_mr,g           │
                                    └────────────────────────────┘
```

- **특징 레이어 $f$**: Backbone 마지막 pooling 레이어 출력 ( $f \in \mathbb{R}^{d_f}$ )
- **특징 레이어 $g$**: FC 레이어 이후 출력 ( $g \in \mathbb{R}^{d_g}$ )
- **LNA와 ER**: $f$와 $g$ 양쪽 모두에 적용하여 $\mathcal{L}\_{mr,f}$와 $\mathcal{L}_{mr,g}$ 계산
- **클러스터**: 에폭마다 소스/타겟 클래스 중심을 업데이트: $\mu_{f,c} = 0.5\mu^s_{f,c} + 0.5\mu^t_{f,c}$

---

### 2.4 성능 향상

**Table 1 기준 주요 성능 비교 (평균 정확도 %):**

| Method | Office-31 | Office-Home | ImageCLEF | VisDA2017 |
|--------|-----------|-------------|-----------|-----------|
| FixBi (CVPR 2021) | 91.4 | 72.7 | 86.0* | 87.2† |
| SRDC (CVPR 2020) | 90.8 | 71.3 | **90.9** | — |
| BSP-TSA | 90.6 | 71.2 | 88.9* | 82.0 |
| **Ours (Mirror)** | **91.7** | **73.4** | **91.6** | **87.9†** |

**Ablation Study (Office-Home / Office-31):**

| Baseline | FC Mirror | BK Mirror | Office-Home | Office-31 |
|----------|-----------|-----------|-------------|-----------|
| ✓ | | | 66.5 | 85.5 |
| ✓ | ✓ | | 71.7 | 89.7 |
| ✓ | | ✓ | 71.8 | 90.0 |
| ✓ | ✓ | ✓ | **73.4** | **91.7** |

→ Mirror Loss 추가로 Office-31에서 **최소 +5.2%p**, Office-Home에서 **최소 +4.2%p** 향상

---

### 2.5 한계점

논문이 직접 언급한 한계:

1. **극단적 편향/희소 데이터 취약**: 데이터셋이 극도로 편향되거나 희소할 경우 Mirror Sample 구성이 불안정
2. **시각 분류 태스크 한정**: 현재 방법론은 비주얼 분류 태스크에 국한
3. **이산적 이웃 탐색의 한계**: LNA는 연속 분포를 이산적 이웃으로 근사하므로 이상적인 Mirror에 대한 오차 존재
4. **계산 비용**: 에폭마다 클러스터 중심을 갱신하고 Top-k 이웃 탐색 필요 ($O(n^s)$, $O(n^t)$ 추가 비용)
5. **하이퍼파라미터 민감도**: $k$ 값에 따라 성능 변동 존재 ($k=3$이 최적, $k=9$에서 성능 하락)

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 이론적 근거: Asymptotic Properties

논문은 두 가지 Proposition으로 일반화 성능 향상을 이론적으로 보증합니다.

**Proposition 1** (분포 정렬과 도메인 불일치):

도메인 $S$와 $T$의 밀도 함수 $\Phi_S(x)$, $\Phi_T(x)$가 given일 때:

$$\Phi_S(x) \overset{a.s.}{=} \Phi_T(x) \implies d_{\mathcal{H}\Delta\mathcal{H}}(S, T) \to 0$$

여기서 $d_{\mathcal{H}\Delta\mathcal{H}}$는 다음과 같이 정의됩니다:

$$d_{\mathcal{H}\Delta\mathcal{H}}(S, T) = 2\sup_{h, h' \in \mathcal{H}} \left|\Pr_{x \sim D^S}[h(x) \neq h'(x)] - \Pr_{x \sim D^T}[h(x) \neq h'(x)]\right|$$

**의미**: $\mathcal{L}_{mr,x}$를 최소화하면 Mirror Pair의 상대적 위치 벡터가 일치하게 되고:
1. 소스/타겟 클래스 중심(anchor)이 동일해짐
2. Mirror Pair들이 공통 중심에 대해 동일한 상대 위치를 가짐

이를 통해 경험적 밀도 함수 $\hat{\Phi}_S(x) = \hat{\Phi}_T(x)$가 달성되고, Glivenko-Cantelli 정리에 의해:

$$\Phi_S(x) \overset{a.s.}{=} \hat{\Phi}_S(x) = \hat{\Phi}_T(x) \overset{a.s.}{=} \Phi_T(x) \tag{9}$$

**Proposition 2** (타겟 오류 상한 감소):

Ben-David et al. (2010)의 이론적 프레임워크를 따라 $\lambda = \min_{h \in \mathcal{H}}\{\mathcal{R}_S(h, h_S) + \mathcal{R}_T(h, h_T)\}$로 정의할 때:

$$\lambda_m + \frac{1}{2}d^m_{\mathcal{H}\Delta\mathcal{H}} \leq \lambda + \frac{1}{2}d_{\mathcal{H}\Delta\mathcal{H}} \tag{10}$$

**핵심 통찰**: Mirror Loss 최소화로 내재 분포 불일치가 0에 수렴하면, 가설 클래스 $\mathcal{H}$의 유효 가설이 더 완화되어 $\lambda$ 값이 낮아지고, 결과적으로 타겟 오류 상한이 감소합니다.

### 3.2 실험적 근거

**시각 패턴 편향 보정 효과 (Table 4, Ar→Cl):**

| 시각 패턴 | Source/Target 비율 | W/O Mirror 오류율 | Mirror 오류율 |
|-----------|-------------------|------------------|--------------|
| Bunk Bed | 0.0% / 7.1% | 8.5% | **2.0%** |
| With People (Bed) | 2.5% / 20.4% | 28.8% | **14.9%** |
| W/O Digit (Calendar) | 15.0% / 42.6% | 31.0% | **24.0%** |
| With Brush | 0.0% / 37.0% | 42.0% | **25.8%** |

→ **도메인 간 패턴 분포 차이가 클수록 Mirror Loss의 보정 효과가 큼**

**t-SNE 클러스터 시각화**에서도 Mirror 적용 시 훈련 종료(epoch 200) 단계에서 소스/타겟 클러스터 형태 유사성이 현저히 향상됨.

### 3.3 일반화 성능 향상 메커니즘 정리

```
샘플링 편향 존재
       │
       ▼
기존 정렬: 편향된 위치로 정렬 → 조건부 분포 왜곡 → 일반화 실패
       │
Mirror 접근:
       │
       ▼
LNA: 상대 도메인에서 동등 샘플 근사
       │
       ▼
ER: Mirror Pair의 앵커 대비 상대 위치 동일화
       │
       ▼
내재 분포 구조 보존 + 도메인 정렬 동시 달성
       │
       ▼
d_{H△H} ↓ + λ ↓ → 타겟 오류 상한 ↓ → 일반화 성능 ↑
```

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4.1 향후 연구에 미치는 영향

#### (A) 이론적 측면

- **샘플링 편향 인식의 중요성**: 도메인 적응 연구에서 "데이터셋이 내재 분포의 편향된 샘플"이라는 현실적 관점을 명시적으로 다루는 연구 방향을 촉진
- **공변량 이동 가정 재검토**: 기존 공변량 이동 이론의 암묵적 한계를 드러내, 보다 현실적인 도메인 이동 이론 수립을 자극
- **Optimal Transport와 Mirror의 연결**: OT 이론 기반의 도메인 정렬 연구에 새로운 관점 제공

#### (B) 방법론적 측면

- **가상 샘플 생성 패러다임**: GAN 없이 특징 공간에서 직접 동등 샘플을 생성하는 경량화 방법론의 가능성 제시
- **플러그인(plug-in) 가능성**: 기존 UDA 프레임워크에 Mirror Loss를 추가 모듈로 통합 가능한 범용적 설계
- **Multi-source/Semi-supervised DA 확장**: Mirror 개념의 다중 소스 또는 반지도 학습으로의 확장 연구 유도

#### (C) 응용 측면

- 의료 이미징, 자율주행 등 도메인 간 샘플링 편향이 심각한 분야에서의 적용 가능성
- 오픈셋 도메인 적응에서 알려지지 않은 클래스의 Mirror 처리 방법 연구 필요

---

### 4.2 향후 연구 시 고려할 점

#### (A) Mirror Sample 품질 개선

현재 LNA는 유클리드 거리 기반의 단순 이웃 탐색입니다. 다음을 고려할 수 있습니다:

- **Attention 기반 Mirror 생성**: 클래스 내 세분화된 시각 패턴을 고려한 가중치 학습
- **Graph Neural Network 활용**: 샘플 간 구조적 관계를 활용한 더 정교한 Mirror 추정
- **동적 $k$ 선택**: 데이터 분포에 따라 $k$를 적응적으로 결정하는 메커니즘

#### (B) 극단적 편향 데이터 대응

- Mirror Sample이 편향된 클래스를 대표하기 위한 **오버샘플링 전략** 결합
- **클래스 불균형**과 **도메인 편향**을 동시에 처리하는 통합 손실 함수 설계

#### (C) 다양한 태스크로의 확장

- 객체 검출(Object Detection), 세그멘테이션(Segmentation) 등으로 Mirror 개념 확장
- **텍스트/NLP 도메인 적응**: 언어 특징 공간에서의 Mirror Sample 정의 방법 연구
- **시계열 데이터**: 시간 의존성을 고려한 Mirror 구성

#### (D) 이론적 엄밀성 강화

- Proposition 2의 $\lambda_m \leq \lambda$ 조건이 성립하기 위한 더 엄격한 조건 분석
- 유한 샘플에서의 수렴 속도(Convergence Rate) 분석

#### (E) 대규모 모델(Foundation Model)과의 결합

- **Vision-Language 모델(CLIP 등)** 기반 특징 공간에서의 Mirror Sample 활용
- **Prompt Tuning** 등 파라미터 효율적 방법과의 결합 시 Mirror Loss의 역할 재정의

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

아래는 본 논문(NeurIPS 2021)과 관련된 2020년 이후 주요 연구들의 비교입니다.

| 연구 | 핵심 아이디어 | 샘플링 편향 고려 | 가상 샘플 생성 | 이론적 보증 |
|------|---------------|-----------------|---------------|-------------|
| **본 논문 (NeurIPS 2021)** | Mirror Sample + Mirror Loss | ✅ 명시적 | ✅ (LNA+ER) | ✅ (Prop. 1, 2) |
| **SHOT (ICML 2020)** [32] | Source-free DA, 가설 전이 | ❌ | ❌ | 부분적 |
| **FixBi (CVPR 2021)** [36] | 고정 비율 믹스업으로 도메인 브리징 | ❌ | 중간 도메인 생성 | ❌ |
| **SRDC (CVPR 2020)** [49] | 구조적 정규화 딥 클러스터링 | ❌ | ❌ | ❌ |
| **RSDA-MSTN (CVPR 2020)** [22] | 구형 공간 도메인 적응 | ❌ | ❌ | 부분적 |
| **TransSDA (CVPR 2021)** [31] | 전이 가능 의미론적 증강 | 부분적 | ✅ | ❌ |

### 비교 분석 요점

1. **SHOT vs. Mirror**: SHOT은 소스 데이터 없이 작동하는 Source-free 설정에 집중하는 반면, Mirror는 소스 데이터를 활용하되 샘플링 편향을 명시적으로 처리합니다. 두 접근은 상호 보완적입니다.

2. **FixBi vs. Mirror**: FixBi는 고정 비율 믹스업으로 중간 도메인을 생성하지만, 도메인 내 분포 구조 보존에 대한 고려가 부족합니다. Mirror는 분포 구조를 보존하면서 정렬합니다.

3. **TransSDA vs. Mirror**: TransSDA도 의미론적 증강을 통한 가상 샘플 생성을 시도하지만, Mirror처럼 상대 도메인에서의 동등 샘플을 명시적으로 정의하지는 않습니다.

4. **OT 기반 연구와의 관계**: DeepJDOT [13]과 같은 OT 기반 방법들은 샘플 수준 정렬을 수행하지만, 샘플링 편향으로 인한 조건부 분포 왜곡 문제를 직접 해결하지는 않습니다. Mirror는 OT 이론과의 연결을 이론적으로 제공하면서도 계산 효율성을 유지합니다.

---

## 참고 자료

1. **본 논문**: Yin Zhao, Minquan Wang, Longjun Cai. "Reducing the Covariate Shift by Mirror Samples in Cross Domain Alignment." *NeurIPS 2021*. (제공된 PDF)

2. **이론적 프레임워크 기반**: Shai Ben-David et al. "A theory of learning from different domains." *Machine Learning*, 79(1-2):151–175, 2010. (논문 내 참고문헌 [2])

3. **SHOT**: Jian Liang, Dapeng Hu, and Jiashi Feng. "Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation." *ICML 2020*. (논문 내 참고문헌 [32])

4. **FixBi**: Jaemin Na et al. "FixBi: Bridging Domain Spaces for Unsupervised Domain Adaptation." *CVPR 2021*. (논문 내 참고문헌 [36])

5. **SRDC**: Hui Tang, Ke Chen, and Kui Jia. "Unsupervised Domain Adaptation via Structurally Regularized Deep Clustering." *CVPR 2020*. (논문 내 참고문헌 [49])

6. **Glivenko-Cantelli 정리**: Rachev et al. "Glivenko–Cantelli Theorem and Bernstein–Kantorovich Invariance Principle." (논문 내 참고문헌 [41])

7. **Optimal Transport**: "Computational Optimal Transport." *Foundations and Trends in Machine Learning*, 11(5-6):355–607, 2019. (논문 내 참고문헌 [1])

> **주의**: 2020년 이후 최신 연구(예: CLIP 기반 DA, Prompt Tuning 기반 DA 등)와의 상세 비교는 해당 논문이 2021년 발표되어 이후 연구를 다루지 않으며, 본 답변에서의 비교는 논문 내 인용 문헌 및 일반적 도메인 적응 연구 동향에 근거합니다. 논문 출판 이후의 직접적 비교 실험 데이터는 논문 원본에 포함되어 있지 않음을 명시합니다.
