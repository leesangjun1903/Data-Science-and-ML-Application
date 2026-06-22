# A Theory of Label Propagation for Subpopulation Shift

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

이 논문(Cai et al., 2021, arXiv:2102.11203)은 **서브포퓰레이션 이동(Subpopulation Shift)** 환경에서의 도메인 적응 문제를 다룬다. 기존의 분포 매칭(Distribution Matching) 기반 방법들이 서브포퓰레이션 이동 상황에서 실패할 수 있음을 지적하고, **레이블 전파(Label Propagation)** 기반의 일관성 정규화(Consistency Regularization)가 이론적으로 보장된 해결책임을 증명한다.

### 주요 기여 (3가지)

| 기여 | 내용 |
|------|------|
| **이론적 프레임워크** | 서브포퓰레이션 이동 하에서 레이블 전파를 통한 학습의 이론적 틀 제시 |
| **정확도 보장** | 확장(Expansion) 가정 하에 타겟 도메인 오류에 대한 유한 샘플 보장 제공 |
| **일반화 프레임워크** | 반지도학습, 도메인 일반화 등 다양한 설정을 포괄하는 일반화된 레이블 전파 프레임워크 제시 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**서브포퓰레이션 이동(Subpopulation Shift)** 은 소스와 타겟 도메인의 클래스 내부 구성 비율이 다를 때 발생한다. 예시:
- 소스: 차량 클래스 = 50% 자동차 + 50% 오토바이
- 타겟: 차량 클래스 = 10% 자동차 + 90% 오토바이

기존 분포 매칭 방법(DANN 등)은 이론적 결함이 있다(Zhao et al., 2019). 또한 기존 이론(Wei et al., 2021)은 타겟 도메인에서의 pseudo-labeler가 어느 정도 정확해야 한다는 강한 가정을 필요로 한다.

**핵심 질문:** 레이블 없는 데이터만을 사용하여 서브포퓰레이션 이동 하에서 소스에서 타겟 도메인으로 증명 가능한(provably) 방식으로 전이할 수 있는가?

---

### 2.2 제안하는 방법 (수식 포함)

#### 설정 (Assumption 1)

소스 분포 $S$와 타겟 분포 $T$는 다음과 같은 구조를 가진다:

$$\text{supp}(S) = \bigcup_{i=1}^{m} S_i, \quad \text{supp}(T) = \bigcup_{i=1}^{m} T_i$$

여기서 $S_i \cap S_j = T_i \cap T_j = S_i \cap T_j = \emptyset$ ($i \neq j$), 그리고 $x \in S_i \cup T_i$에 대해 ground truth 클래스 $g^*(x) = y_i$로 일정하다.

**추가 가정:**

1. **교사 분류기의 마진 조건 ($\gamma > 0$):**

$$\mathbb{P}_{x \sim S_i}[g_{tc}(x) = y_i] \geq \mathbb{P}_{x \sim S_i}[g_{tc}(x) = k] + \gamma, \quad \forall k \in \{1,\cdots,K\} \setminus \{y_i\}$$

2. **도메인 이동 비율 상한 ($r$):**

$$\frac{\mathbb{P}_T[T_i]}{\mathbb{P}_S[S_i]} \leq r, \quad \forall i \in \{1, \cdots, m\}$$

#### 일관성 정규화 항 (Consistency Regularizer)

혼합 분포 $\frac{1}{2}(S+T)$ 위에서의 정규화 항:

$$R_{\mathcal{B}}(g) := \mathbb{P}_{x \sim \frac{1}{2}(S+T)}\left[\exists x' \in \mathcal{B}(x), \text{ s.t. } g(x) \neq g(x')\right]$$

여기서 $\mathcal{B}(x)$는 입력 $x$의 이웃 집합(데이터 증강 또는 거리 기반)이다. 낮은 $R_{\mathcal{B}}(g)$ 값은 $\mathcal{B}(x)$ 내에서 예측이 일관적임을 의미한다.

#### 확장 속성 (Definition 2.1)

**1. 곱셈적 확장 (Multiplicative Expansion):**

$\frac{1}{2}(S+T)$가 $(a,c)$-곱셈적 확장을 만족한다는 것은, 임의의 $i$와 $\mathbb{P}_{\frac{1}{2}(S_i+T_i)}[A] \leq a$인 $A \subset S_i \cup T_i$에 대해:

$$\mathbb{P}_{\frac{1}{2}(S_i+T_i)}[\mathcal{N}(A)] \geq \min\left(c\,\mathbb{P}_{\frac{1}{2}(S_i+T_i)}[A],\; 1\right)$$

**2. 상수적 확장 (Constant Expansion):**

$\frac{1}{2}(S+T)$가 $(q, \xi)$ -상수적 확장을 만족한다는 것은, $\mathbb{P}\_{\frac{1}{2}(S+T)}[A] \geq q$이고 $\mathbb{P}_{\frac{1}{2}(S_i+T_i)}[A] \leq \frac{1}{2}$인 모든 $A$에 대해:

$$\mathbb{P}_{\frac{1}{2}(S+T)}[\mathcal{N}(A)] \geq \min\left(\xi,\, \mathbb{P}_{\frac{1}{2}(S+T)}[A]\right) + \mathbb{P}_{\frac{1}{2}(S+T)}[A]$$

#### 알고리즘 (Algorithm)

분류기 $g$를 다음의 최적화 문제의 해로 정의한다:

```math
g = \underset{g:\mathcal{X} \to \mathcal{Y},\, g \in G}{\text{argmin}}\; L^S_{01}(g, g_{tc})
```

$$\text{s.t.} \quad R_{\mathcal{B}}(g) \leq \mu \tag{1}$$

여기서 $L^S\_{01}(g, g_{tc}) := \mathbb{P}\_{x \sim S}[g(x) \neq g_{tc}(x)]$는 소스 도메인에서 교사 분류기와의 0-1 손실이다.

---

### 2.3 주요 정리 (Main Theorems)

#### Theorem 2.1: 곱셈적 확장 하의 타겟 오류 상한

Assumption 1이 성립하고 $\frac{1}{2}(S+T)$가 $(\frac{1}{2}, c)$-곱셈적 확장을 만족할 때, 알고리즘 (1)이 반환하는 분류기 $g$의 타겟 오류 $\epsilon_T(g) := \mathbb{P}_{x \sim T}[g(x) \neq g^*(x)]$는:

$$\epsilon_T(g) \leq \max\!\left(\frac{c+1}{c-1},\; 3\right) \frac{8r\mu}{\gamma}$$

#### Theorem 2.2: 상수적 확장 하의 타겟 오류 상한

Assumption 1이 성립하고 $\frac{1}{2}(S+T)$가 $(q, \mu)$-상수적 확장을 만족할 때:

$$\epsilon_T(g) \leq \left(2\max(q,\mu) + \mu\right)\frac{8r\mu}{\gamma}$$

**핵심 인사이트:** $\mu \to 0$ (ground truth의 일관성 오류가 작아질수록), 교사 분류기의 정확도에 무관하게 $\epsilon_T(g) \to 0$.

---

### 2.4 딥 신경망을 위한 유한 샘플 보장 (Theorem 2.3)

각 도메인에서 $n$개의 i.i.d. 데이터가 있을 때, All-layer margin $m(f,x,y)$ (Wei and Ma, 2019)을 이용하여 다음의 마진 기반 알고리즘을 사용한다:

$$g = \underset{g:\mathcal{X}\to\mathcal{Y},g\in G}{\text{argmin}} \; \mathbb{P}_{x \sim \hat{S}}[m(f,x,g_{tc}(x)) \leq t]$$

$$\text{s.t.} \quad \mathbb{P}_{x \sim \frac{1}{2}(\hat{S}+\hat{T})}[m_{\mathcal{B}}(f,x) \leq t] \leq \mu \tag{2}$$

$(\frac{1}{2},c)$-곱셈적 확장 하에서 확률 $1-\delta$로:

$$\epsilon_T(g) \leq \frac{8r}{\gamma}\left(\max\!\left(\frac{c+1}{c-1},3\right)\hat{\mu} + \Delta\right)$$

여기서

$$\Delta = \widetilde{O}\!\left(\left(\mathbb{P}_{x\sim\hat{S}}[m(f^*,x,g_{tc}(x))\leq t] - L^{\hat{S}}_{01}(g^*,g_{tc})\right) + \frac{\sum_i \sqrt{q}\|W_i\|_F}{t\sqrt{n}} + \sqrt{\frac{\log(1/\delta)+p\log n}{n}}\right)$$

$$\hat{\mu} = \mu + \widetilde{O}\!\left(\frac{\sum_i\sqrt{q}\|W_i\|_F}{t\sqrt{n}} + \sqrt{\frac{\log(1/\delta)+p\log n}{n}}\right)$$

$n \to \infty$일 때 $\Delta \to 0$, $\hat{\mu} \to \mu$가 되어 Theorem 2.1, 2.2의 결과로 수렴하며, **샘플 복잡도가 입력 차원에 대해 지수적으로 증가하지 않는다**.

---

### 2.5 일반화된 프레임워크 (Section 3)

제3의 커버링 분포 $U$ (Assumption 2)를 도입하여 더 일반적인 설정을 다룬다:

$$R_{\mathcal{B}}(g) := \mathbb{P}_{x \sim U}[\exists x' \in \mathcal{B}(x), \text{ s.t. } g(x) \neq g(x')]$$

$U$가 $(\frac{1}{2},c)$-곱셈적 확장을 만족할 때:

$$\epsilon_T(g) \leq \max\!\left(\frac{c+1}{c-1},3\right)\frac{4\kappa r\mu}{\gamma}$$

이 일반 프레임워크는 다음 5가지 설정을 포괄한다:

```
(a) 비지도 도메인 적응: U = ½(S+T), κ=2
(b) 반지도 학습:        S = T = U
(c) 도메인 확장:        T = U
(d) 도메인 외삽:        S∪T를 U가 연결
(e) 다중소스 도메인 적응/일반화
```

---

### 2.6 모델 구조

별도의 새로운 신경망 아키텍처를 제안하는 것이 아니라, 다음의 **2단계 파이프라인**을 사용한다:

1. **Step 1 (교사 학습):** SwAV (자기지도학습) 표현으로 초기화된 ResNet-50을 소스 도메인에서 파인튜닝 → 교사 분류기 $g_{tc}$ 획득
2. **Step 2 (레이블 전파):** FixMatch의 일관성 정규화 목적 함수로 추가 파인튜닝
   - 약한 증강(weak augmentation) 샘플에 대한 지도 손실
   - 강한 증강(strong augmentation)의 예측이 약한 증강의 예측과 일치하도록 유도

---

### 2.7 성능 향상

**ENTITY-30 (BREEDS, 서브포퓰레이션 이동 시뮬레이션):**

| 방법 | 소스 정확도 | 타겟 정확도 |
|------|-----------|-----------|
| Train on Source | 91.91±0.23 | 56.73±0.32 |
| DANN | 92.81±0.50 | 61.03±4.63 |
| MDD | 92.67±0.54 | 63.95±0.28 |
| **FixMatch (제안)** | **90.87±0.15** | **72.60±0.51** |

→ 기준선 대비 **+15.87%p**, 분포 매칭 방법 대비 **+8.65%p** 향상

**Office-31 (스타일 이동):**

| 방법 | 평균 정확도 |
|------|-----------|
| MDD | 89.16 |
| MDD+FixMatch | **89.84** |

**Office-Home:**

| 방법 | 평균 정확도 |
|------|-----------|
| MDD | 68.0 |
| MDD+FixMatch | **69.6** |

---

### 2.8 한계점

1. **최적화 과정 미분석:** 알고리즘 (1)의 해를 찾았다고 가정하고 분석하지만, 실제 최적화 과정(경사하강법 등)에 대한 이론적 보장은 없다.
2. **확장 속성 검증의 어려움:** 실제 데이터에서 확장 속성이 성립하는지 사전에 확인하기 어렵다.
3. **이진화된 서브포퓰레이션 구조 가정:** 각 서브포퓰레이션 내 ground truth 레이블이 일정하다는 가정은 실제 데이터에서 완벽히 성립하지 않을 수 있다.
4. **마진 $\gamma$에 대한 의존성:** 교사 분류기의 마진이 매우 작을 경우 보장이 약해진다.
5. **계산 비용:** SwAV 사전학습이 필요하며, 이는 상당한 계산 자원을 요구한다.
6. **실험 범위 제한:** 이미지 도메인에 집중되어 있어 다른 모달리티에 대한 검증이 부족하다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 이론적 일반화 보장의 핵심 메커니즘

논문의 가장 핵심적인 일반화 인사이트는 **교사 분류기의 성능에 무관하게 타겟 도메인 성능이 수렴한다**는 점이다. Theorem 2.1에서:

$$\epsilon_T(g) \leq \max\!\left(\frac{c+1}{c-1},3\right)\frac{8r\mu}{\gamma}$$

이 상한에서 **교사 분류기의 타겟 도메인 오류가 직접 등장하지 않는다**. 이는 Wei et al. (2021)의 결과인 $O\!\left(\frac{1}{c}\text{error}(g_{tc}) + \mu\right)$와 결정적으로 다른 점으로, 타겟 도메인에서의 pseudo-labeler가 완전히 틀리더라도 일관성 정규화만 충분히 작으면 좋은 성능을 보장한다.

### 3.2 일반화 성능 향상 요인 분석

**요인 1: 소스에서도 성능 개선 (Teacher Improvement)**
- 알고리즘 (1)은 $R_{\mathcal{B}}(g) \leq \mu$ 제약 하에 소스 손실을 최소화
- 이 과정이 소스 도메인 $S_i$의 예측도 정제하여, 교사보다 더 낮은 소스 오류를 달성할 수 있음
- 즉, label propagation이 **소스→타겟** 방향뿐 아니라 **소스 내부**도 개선

**요인 2: 확장 속성과 지역-전역 일관성 변환**

확장 속성은 지역적 일관성(local consistency)을 전역적 일관성(global consistency)으로 변환시켜 준다. 구체적으로, minority set $\widetilde{M}$의 상한을 통해:

$$\mathbb{P}_{\frac{1}{2}(S+T)}[\widetilde{M}] \leq \max\!\left(\frac{c+1}{c-1}, 3\right)\mu \quad \text{(곱셈적 확장 하)}$$

이것이 타겟 오류 상한으로 전환된다.

**요인 3: 표현 공간에서의 적용 가능성**

논문은 확장 속성과 일관성 정규화가 입력 공간뿐 아니라 **표현 공간(representation space)**에서도 적용될 수 있음을 명시한다:

$$d(x, x') = \|h(x) - h(x')\|$$

이는 자기지도학습(SwAV)으로 학습된 표현에서 서브포퓰레이션이 클러스터링되는 성질을 이론적으로 활용하는 근거가 된다.

**요인 4: 샘플 복잡도의 차원 무관성**

Theorem 2.3에서, 필요 샘플 복잡도가 입력 차원 $d$에 대해 지수적으로 의존하지 않는다. 이는 고차원 데이터(이미지 등)에서도 실용적임을 보장한다.

### 3.3 일반화 가능 설정의 확장

Section 3의 일반화 프레임워크는 다음과 같은 다양한 학습 시나리오에서 일반화 성능 향상을 이론적으로 보장한다:

- **도메인 외삽(Domain Extrapolation):** 소스와 타겟이 직접 연결되지 않아도, 제3의 커버링 데이터 $U$를 통해 레이블 정보를 전파 가능
- **다중소스 도메인 적응:** 여러 소스 도메인의 합집합 $U$가 타겟을 커버하면 보장 성립
- **반지도학습:** $S=T=U$인 특수 케이스로 Wei et al. (2021) 대비 더 강한 결과 제공

---

## 4. 앞으로의 연구에 미치는 영향 및 고려 사항

### 4.1 연구에 미치는 영향

**① 서브포퓰레이션 이동 이론의 토대 구축**

분포 매칭 패러다임의 이론적 한계(Zhao et al., 2019)에 대한 대안적 이론적 틀을 제공한다. 이 논문은 "왜 일관성 정규화가 서브포퓰레이션 이동에서 잘 작동하는가"에 대한 첫 번째 형식적 답변을 제시한다.

**② 이론-실험 간 브릿지 역할**

FixMatch(Sohn et al., 2020)와 같은 반지도학습 방법을 도메인 적응에 이론적 정당성과 함께 적용할 수 있는 근거를 제공, 향후 이론 기반 알고리즘 설계를 촉진한다.

**③ 자기지도학습과 도메인 적응의 연결**

SwAV 표현이 서브포퓰레이션의 확장 속성을 feature space에서 만족시킬 수 있다는 실험적 검증은, 자기지도학습 표현이 도메인 적응에 유리한 이유를 설명하는 이론적 틀을 열었다.

**④ 일반화 프레임워크의 범용성**

도메인 일반화, 다중소스 도메인 적응, 도메인 외삽을 하나의 통합 프레임워크로 분석하여 서로 다른 연구 커뮤니티 간 이론적 교류를 촉진한다.

### 4.2 앞으로의 연구 시 고려 사항

**① 최적화 과정의 이론적 분석 필요**

현재 논문은 알고리즘 (1)의 전역 최솟값이 주어진다고 가정한다. 실제로는 비볼록 최적화이므로:
- 경사하강법이 이 해를 찾을 수 있는 조건은 무엇인가?
- 반복 학습(iterative self-training)과의 수렴 분석이 요구된다

**② 확장 속성의 검증 가능성**

실제 데이터에서 확장 상수 $c$나 $\xi$를 추정하는 방법론 개발이 필요하다. 이는 이론적 보장의 실용적 적용 가능성을 위해 필수적이다.

**③ 마진 $\gamma$가 작은 경우의 강건성**

교사 분류기의 마진이 작은 경우 보장이 크게 약화된다. 마진이 데이터 의존적으로 추정될 때의 적응적 보장(adaptive guarantee) 개발이 필요하다.

**④ 레이블 노이즈와의 결합**

교사 분류기가 소스 도메인에서도 잡음이 있는 레이블로 학습된 경우(weakly supervised 설정), 기존 Assumption 1(a)의 마진 가정이 완화되어야 한다.

**⑤ 비전 외 도메인 적용**

실험이 이미지 분류에 집중되어 있으므로 NLP(도메인 이동이 빈번한 감성 분석, 개체명 인식 등), 의료 데이터, 시계열에서의 검증이 필요하다.

**⑥ 라지 랭귀지 모델(LLM)과의 연계**

사전학습 LLM의 fine-tuning은 본질적으로 서브포퓰레이션 이동 문제이다. 이 프레임워크를 prompt-tuning이나 RLHF 환경에 적용하는 연구 방향이 유망하다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 이론적 관련 연구

| 논문 | 방법론 | 이 논문과의 관계 |
|------|--------|----------------|
| **Wei et al. (2021)** "Theoretical Analysis of Self-Training with Deep Networks on Unlabeled Data" (ICLR 2021) | 확장 가정 기반 반지도학습 이론 | 본 논문의 직접적 기반. 본 논문은 도메인 적응으로 확장하고, 교사의 타겟 성능 의존성을 제거 |
| **Santurkar et al. (2021)** "BREEDS: Benchmarks for Subpopulation Shift" (ICLR 2021) | 서브포퓰레이션 이동 벤치마크 | 본 논문의 실험 플랫폼(ENTITY-30) 제공 |
| **Kumar et al. (2020)** "Understanding Self-Training for Gradual Domain Adaptation" | 점진적 도메인 적응의 자기학습 분석 | 점진적 구조를 가정하지만 서브포퓰레이션 이동 특화 이론 없음 |
| **Zhao et al. (2019a)** "On Learning Invariant Representations for Domain Adaptation" | 불변 표현의 이론적 한계 | 분포 매칭 실패 사례를 이론적으로 증명, 본 논문의 동기 부여 |
| **Chen et al. (2020c)** "Self-Training Avoids Using Spurious Features under Domain Shift" | 자기학습과 강건한 특징 학습 분석 | 확률적 가정 하에서만 성립, 서브포퓰레이션 구조 미분석 |

### 5.2 알고리즘적 관련 연구

| 논문 | 방법 | 비교 |
|------|------|------|
| **Sohn et al. (2020)** "FixMatch" (NeurIPS 2020) | 일관성 정규화 + 자기학습의 반지도학습 | 본 논문이 이를 도메인 적응에 이론적 정당성과 함께 적용 |
| **Xie et al. (2020)** "UDA: Unsupervised Data Augmentation" (NeurIPS 2020) | 데이터 증강 기반 일관성 학습 | 본 논문의 $\mathcal{B}(\cdot)$ 개념과 연결; 이론 보장 없음 |
| **Ganin et al. (2016)** "DANN" (JMLR 2016) | 도메인 적대적 학습 | 서브포퓰레이션 이동에서 본 논문 대비 8% 이상 낮은 성능 |
| **Zhang et al. (2019)** "MDD" (ICML 2019) | 마진 기반 도메인 불일치 최소화 | 스타일 이동에서는 효과적이지만 서브포퓰레이션 이동에서 열위 |
| **Caron et al. (2020)** "SwAV" (NeurIPS 2020) | 클러스터 일관성 기반 자기지도학습 | 본 논문의 표현 학습 기반으로 사용; 확장 속성을 feature space에서 만족시킴 |

### 5.3 비교 분석 요약

```
이론적 보장        강도: 본 논문 > Wei et al.(2021) > Kumar et al.(2020) > Chen et al.(2020c)
                   (타겟 의존성 제거 측면에서)

서브포퓰레이션     특화도: 본 논문 ≈ Santurkar et al.(2021) >> DANN, MDD
이동 특화

알고리즘           실용성: FixMatch(단독) ≈ 본 논문 > MDD > DANN
실용성             (서브포퓰레이션 이동 기준)

일반화 범위        본 논문의 프레임워크가 반지도학습, 다중소스, 도메인 일반화를 모두 포괄
```

**결론적으로**, 이 논문은 서브포퓰레이션 이동 하의 도메인 적응에 대한 최초의 포괄적 이론적 프레임워크를 제공하며, 실용적 알고리즘(FixMatch + SwAV)과 이론을 효과적으로 연결한 선구적 연구이다. 다만 최적화 분석, 확장 속성 검증, 비이미지 도메인 적용 등은 향후 연구의 중요한 방향으로 남아 있다.

---

## 참고자료

- **주 논문:** Tianle Cai, Ruiqi Gao, Jason D. Lee, Qi Lei. "A Theory of Label Propagation for Subpopulation Shift." arXiv:2102.11203v3 [cs.LG], July 21, 2021.
- Wei, C., Shen, K., Chen, Y., and Ma, T. (2021). "Theoretical analysis of self-training with deep networks on unlabeled data." *International Conference on Learning Representations (ICLR)*.
- Wei, C. and Ma, T. (2019). "Improved sample complexities for deep networks and robust classification via an all-layer margin." arXiv:1910.04284.
- Santurkar, S., Tsipras, D., and Madry, A. (2021). "BREEDS: Benchmarks for subpopulation shift." *ICLR*.
- Sohn, K. et al. (2020). "FixMatch: Simplifying semi-supervised learning with consistency and confidence." *NeurIPS 33*.
- Caron, M. et al. (2020). "Unsupervised learning of visual features by contrasting cluster assignments." *NeurIPS* (SwAV).
- Xie, Q. et al. (2020). "Unsupervised data augmentation for consistency training." *NeurIPS 33*.
- Ganin, Y. et al. (2016). "Domain-adversarial training of neural networks." *JMLR*, 17(1):2096–2030.
- Zhang, Y. et al. (2019). "Bridging theory and algorithm for domain adaptation." *ICML*, pages 7404–7413.
- Zhao, H. et al. (2019a). "On learning invariant representation for domain adaptation." arXiv:1901.09453.
- Li, B. et al. (2020). "Rethinking distributional matching based domain adaptation." arXiv:2006.13352.
- Kumar, A., Ma, T., and Liang, P. (2020). "Understanding self-training for gradual domain adaptation." arXiv:2002.11361.
- Chen, Y., Wei, C., Kumar, A., and Ma, T. (2020c). "Self-training avoids using spurious features under domain shift." arXiv:2006.10032.
- Ben-David, S. et al. (2010). "A theory of learning from different domains." *Machine Learning*, 79(1-2):151–175.
