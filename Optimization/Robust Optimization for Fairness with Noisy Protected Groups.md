# Robust Optimization for Fairness with Noisy Protected Groups

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

이 논문은 머신러닝의 그룹 기반 공정성(group-based fairness) 기준을 적용할 때, **보호 집단(protected group)의 레이블이 노이즈를 포함하는 현실적인 상황**에서 기존의 순진한(naïve) 접근법이 실패할 수 있음을 보이고, 이를 해결하기 위한 두 가지 강인한(robust) 최적화 방법을 제안합니다.

### 세 가지 주요 기여

1. **이론적 분석 (Naive Approach의 한계):** 노이즈 그룹 $\hat{G}$에서 공정성 기준이 만족될 때, 실제 그룹 $G$에 대한 공정성 위반의 상한(upper bound)을 TV 거리로 제공합니다.

2. **두 가지 로버스트 최적화 방법 제안:**
   - **방법 1:** 분포적 강건 최적화(Distributionally Robust Optimization, DRO) 기반 접근법
   - **방법 2:** 소프트 그룹 할당(Soft Group Assignments) 기반 접근법

3. **실증적 검증:** UCI 데이터셋(Adult, Credit) 두 가지 사례 연구를 통해 두 로버스트 접근법이 모든 노이즈 수준에서 실제 그룹에 대한 공정성 기준을 만족시킴을 보입니다.

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

많은 그룹 기반 공정성 기준(demographic parity, equality of opportunity, equalized odds 등)은 실제 보호 집단 레이블 $G$를 알고 있다고 가정합니다. 그러나 현실에서는:

- **설문 응답 편향:** 응답자가 종교, 정치적 성향, 성적 지향에 대해 사회적으로 바람직한 답변을 선택함
- **추정에 의한 노이즈:** 인종/성별 분류기로 집단 레이블을 추정할 때 발생하는 오류
- **시간적 편차:** 10년 전 수집된 센서스 데이터가 현재를 반영하지 못함
- **대리 변수 사용:** 우편번호로 사회경제적 집단을 추정하는 경우

즉, 실무자는 실제 그룹 $G \in \{1, ..., m\}$ 대신 노이즈 그룹 $\hat{G} \in \{1, ..., \hat{m}\}$만 접근 가능합니다.

**핵심 질문:**
- $\hat{G}$에 공정성 제약을 만족시키면 $G$에 대해서는 어떤 보장이 되는가?
- 노이즈 모델 정보를 활용하여 $G$에 대한 공정성을 어떻게 보장할 수 있는가?

---

### 2.2 기본 최적화 설정

진짜 그룹 레이블 $G$를 알 때의 공정성 제약 학습 문제:

$$\min_{\theta} f(\theta) \quad \text{s.t.} \quad g_j(\theta) \leq 0, \; \forall j \in \mathcal{G} \tag{1}$$

여기서:
- $f(\theta) = \mathbb{E}[l(\theta, X, Y)]$: 학습 손실 (예: hinge loss)
- $g_j(\theta) = \mathbb{E}[h(\theta, X, Y) | G = j]$: 그룹별 공정성 제약
- $h(\theta, X, Y)$: 공정성 메트릭 함수 (예: demographic parity의 경우 $h = \mathbb{1}[\phi(X;\theta)>0] - \mathbb{E}[\mathbb{1}[\phi(X;\theta)>0]]$)

**Naïve Approach** (노이즈 그룹만 사용):

$$\min_{\theta} f(\theta) \quad \text{s.t.} \quad \hat{g}_j(\theta) \leq 0, \; \forall j \in \mathcal{G} \tag{2}$$

여기서 $\hat{g}_j(\theta) = \mathbb{E}[h(\theta, X, Y) | \hat{G} = j]$.

---

### 2.3 Naïve Approach의 이론적 한계 (Section 4)

#### TV 거리 기반 상한 (Theorem 1)

$X,Y|G=j \sim p_j$, $X,Y|\hat{G}=j \sim \hat{p}_j$ 로 정의할 때:

**Theorem 1:** 모델 $\theta$가 $\hat{g}_j(\theta) \leq 0 \; \forall j \in \mathcal{G}$를 만족하고, $|h(\theta, x_1, y_1) - h(\theta, x_2, y_2)| \leq 1$이며, $TV(p_j, \hat{p}_j) \leq \gamma_j$이면:

$$g_j(\theta) \leq \gamma_j, \quad \forall j \in \mathcal{G}$$

**증명 핵심:** Kantorovich-Rubinstein 정리를 활용하여:

$$|g_j(\theta) - \hat{g}_j(\theta)| = |\mathbb{E}_{p_j}[h] - \mathbb{E}_{\hat{p}_j}[h]| \leq TV(p_j, \hat{p}_j)$$

이 성립하고, $\hat{g}_j(\theta) \leq 0$이므로 $g_j(\theta) \leq TV(p_j, \hat{p}_j) \leq \gamma_j$.

#### TV 거리 추정 (Lemma 1)

$P(G=j) = P(\hat{G}=j)$를 가정하면:

$$TV(p_j, \hat{p}_j) \leq P(G \neq \hat{G} | G = j)$$

이를 통해 TV 거리를 실제로 추정 가능한 오분류율로 대체할 수 있습니다.

---

### 2.4 방법 1: 분포적 강건 최적화 (DRO, Section 5)

#### 아이디어

$TV(p_j, \hat{p}_j) \leq \gamma_j$를 알면, $p_j$를 포함하는 분포 집합 위에서 최악의 경우(worst-case)를 최적화함으로써 실제 그룹 $G$에 대한 공정성을 보장할 수 있습니다.

#### DRO 제약 최적화 문제

$$\min_{\theta \in \Theta} f(\theta) \quad \text{s.t.} \quad \max_{\substack{\tilde{p}_j: TV(\tilde{p}_j, \hat{p}_j) \leq \gamma_j \\ \tilde{p}_j \ll p}} \tilde{g}_j(\theta) \leq 0, \quad \forall j \in \mathcal{G} \tag{3}$$

여기서 $\tilde{g}\_j(\theta) = \mathbb{E}_{X,Y \sim \tilde{p}_j}[h(\theta, X, Y)]$.

#### 일반 DRO 형식

$$\min_{\theta \in \Theta} \max_{q: D(q,p) \leq \gamma} \mathbb{E}_{X,Y \sim q}[l(\theta, X, Y)] \tag{4}$$

#### 경험적 Lagrangian 공식화

$n$개의 샘플로 경험적 형태:

$$\min_{\theta} \frac{1}{n}\sum_{i=1}^{n} l(\theta, X_i, Y_i) \quad \text{s.t.} \quad \max_{\tilde{p}_j \in \mathbb{B}_{\gamma_j}(\hat{p}_j)} \sum_{i=1}^{n} \tilde{p}_j^i h(\theta, X_i, Y_i) \leq 0, \quad \forall j \in \mathcal{G} \tag{9}$$

여기서 $\mathbb{B}\_{\gamma_j}(\hat{p}\_j) = \{\tilde{p}\_j \in \mathbb{R}^n : \frac{1}{2}\sum\_{i=1}^{n}|\tilde{p}\_j^i - \hat{p}\_j^i| \leq \gamma\_j, \sum_{i=1}^{n}\tilde{p}_j^i = 1, \tilde{p}\_j^i \geq 0\}$

#### 해결 방법 (Algorithm 2: Projected GDA)

$$\min_{\theta} \max_{\lambda \geq 0} \max_{\substack{\tilde{p}_j \in \mathbb{R}^n, \tilde{p}_j \geq 0, \\ j=1,...,m}} f(\theta) + \sum_{j=1}^{m} \lambda_j f_j(\theta, \tilde{p}_j) \quad \text{s.t.} \quad \|\tilde{p}_j - \hat{p}_j\|_1 \leq 2\gamma_j, \|\tilde{p}_j\|_1 = 1 $$

$\ell_1$-ball 위에의 효율적인 투영을 이용하여 Projected GDA 알고리즘으로 풀 수 있습니다.

---

### 2.5 방법 2: 소프트 그룹 할당 (Soft Group Assignments, Section 6)

#### 아이디어

Kallus et al. [37]의 감사(auditing) 기준을 학습에 적용하여, 노이즈 모델 $P(G=j|\hat{G}=k)$를 보조 데이터셋에서 추정하고, 이를 활용한 로버스트 공정성 기준을 최적화합니다.

#### 소프트 가중치 집합

$w(j|\hat{y}, y, k)$가 $P(G=j|\hat{Y}(\theta)=\hat{y}, Y=y, \hat{G}=k)$를 추정하는 함수라 할 때, 가능한 함수 집합:

```math
\mathcal{W}(\theta) = \left\{ w : \sum_{\hat{y},y \in \{0,1\}} w(j|\hat{y},y,k) P(\hat{Y}(\theta)=\hat{y}, Y=y|\hat{G}=k) = P(G=j|\hat{G}=k), \; \sum_{j=1}^{m} w(j|\hat{y},y,k)=1, \; w \geq 0 \right\}
```

#### 로버스트 공정성 기준

$$\max_{w \in \mathcal{W}(\theta)} g_j(\theta, w) \leq 0, \quad \forall j \in \mathcal{G} \tag{6}$$

여기서 $g_j(\theta, w) = \frac{\mathbb{E}[h(\theta,X,Y)w(j|\hat{Y}(\theta),Y,\hat{G})]}{P(G=j)}$

#### 로버스트 최적화 문제

$$\min_{\theta \in \Theta} f(\theta) \quad \text{s.t.} \quad \max_{w \in \mathcal{W}(\theta)} g_j(\theta, w) \leq 0, \quad \forall j \in \mathcal{G} \tag{7}$$

Lagrangian으로 변환:

$$\min_{\theta \in \Theta} \max_{\lambda \in \Lambda} \mathcal{L}(\theta, \lambda) \tag{8}$$

$$\mathcal{L}(\theta, \lambda) = f(\theta) + \sum_{j=1}^{m} \lambda_j \max_{w \in \mathcal{W}(\theta)} g_j(\theta, w), \quad \Lambda \subseteq \mathbb{R}^m_+$$

---

### 2.6 모델 구조 및 알고리즘

#### Ideal Algorithm (Algorithm 1) - 이론적 수렴 보장

$\theta$-player와 $\lambda$ -player 간의 제로섬 게임으로 해석:

1. **Best response on $\theta$:** Oracle 기반 Algorithm 3으로 $\hat{\theta}^{(t)}$ 산출
2. **Gradient estimation:** $\delta_j^{(t)} \leq \mathbb{E}\_{\theta \sim \hat{\theta}^{(t)}}[\max_{w \in \mathcal{W}(\theta)} g_j(\theta,w)] \leq \delta_j^{(t)} + \rho'$
3. **Ascent on $\lambda$:** $\tilde{\lambda}\_j^{(t+1)} \leftarrow \lambda_j^{(t)} + \eta_\lambda \delta_j^{(t)}$, $\lambda^{(t+1)} \leftarrow \Pi_\Lambda(\tilde{\lambda}^{(t+1)})$
4. **Return:** $\bar{\theta} = \frac{1}{T}\sum_{t=1}^{T} \hat{\theta}^{(t)}$

**Theorem 4 (수렴 보장):** $R = T^{1/4}$, $\eta_\lambda = \frac{R}{B_\lambda \sqrt{T}}$로 설정하면:

$$\mathbb{E}_{\theta \sim \bar{\theta}}[f(\theta)] \leq f(\theta^*) + \mathcal{O}\left(\frac{1}{T^{1/4}}\right) + \rho$$

$$\mathbb{E}_{\theta \sim \bar{\theta}}\left[\max_{w \in \mathcal{W}(\theta)} g_j(\theta, w)\right] \leq \mathcal{O}\left(\frac{1}{T^{1/4}}\right) + \rho'$$

#### Practical Algorithm (Algorithm 4) - 계산 효율적

1. **w 최적화 (LP):** $w^{(t)} \leftarrow \max_{w \in \mathcal{W}(\theta^{(t)})} \sum_{j=1}^{m} \lambda_j^{(t)} g_j(\theta^{(t)}, w)$
2. **$\theta$ gradient step:** $\theta^{(t+1)} \leftarrow \theta^{(t)} - \eta_\theta \nabla_\theta \left[f_0(\theta^{(t)}) + \sum_{j=1}^{m} \lambda_j^{(t)} g_j(\theta^{(t)}, w^{(t+1)})\right]$
3. **$\lambda$ ascent:** $\tilde{\lambda}\_j^{(t+1)} \leftarrow \lambda_j^{(t)} + \eta_\lambda g_j(\theta^{(t+1)}, w^{(t+1)})$, $\lambda^{(t+1)} \leftarrow \Pi_\Lambda(\tilde{\lambda}^{(t+1)})$

---

### 2.7 성능 향상 및 한계

#### 성능 향상

| 방법 | 공정성 보장 | 오차율 | 보수성 |
|------|------------|--------|--------|
| Naïve | 노이즈 증가 시 실패 | 낮음 | - |
| DRO | 모든 노이즈 수준에서 평균적으로 만족 | 높음 | 높음 |
| Soft Assignments | 모든 노이즈 수준에서 평균적으로 만족 | 중간 | 낮음 |

- **Adult dataset (γ=0.3):** Naïve 접근법은 진짜 그룹 제약 위반, DRO/SA는 평균적으로 만족
- **SA > DRO:** SA가 DRO보다 낮은 오차율 달성 (더 정밀한 노이즈 모델 사용)

#### 한계

1. **DRO의 과보수성:** Lemma 1의 경계가 느슨하여 DRO가 불필요하게 보수적인 모델 생성
2. **보조 데이터셋 필요:** SA 방법은 $P(G=j|\hat{G}=k)$ 추정을 위한 별도 데이터셋 필요
3. **Ideal Algorithm의 계산 비용:** 비볼록(non-convex) 최소화 오라클 필요, 다단계 중첩 문제
4. **이진 분류기에 국한:** 현재 이진 분류기에만 적용 (다중 클래스로의 확장 필요)
5. **교차성(Intersectionality) 미고려:** 그룹 교차점에서의 불공정성은 다루지 않음
6. **보조 데이터셋과 주 데이터셋 간의 분포 불일치 문제**

---

## 3. 일반화 성능 향상 가능성

### 3.1 DRO와 일반화의 연결

DRO는 본질적으로 **최악의 경우 분포(worst-case distribution)** 위에서 최적화하는 방식으로, 이는 분포 이동(distribution shift)에 대한 강건성을 제공합니다. 구체적으로:

**일반화 성능 향상 메커니즘:**

$$\min_{\theta \in \Theta} \max_{\tilde{p}_j: TV(\tilde{p}_j, \hat{p}_j) \leq \gamma_j} \tilde{g}_j(\theta) \leq 0$$

이 공식은 학습 데이터와 테스트 데이터 분포 간의 차이가 $\gamma_j$ 내에 있을 경우, **테스트셋에서도 공정성 제약이 유지됨을 보장**합니다. 이는 일종의 PAC-Bayes 스타일의 일반화 경계와 유사한 역할을 합니다.

### 3.2 소프트 그룹 할당과 일반화

소프트 그룹 할당 방법은:
- 더 정밀한 노이즈 모델 $P(G=j|\hat{G}=k)$를 활용하여 **불필요한 과보수성 감소**
- 과보수성 감소 → **오차율 감소** → 실제 적용에서의 유용성 증가
- DRO 대비 더 낮은 오차율로도 공정성 제약 만족 가능

### 3.3 일반화 관련 한계

논문 자체에서 언급한 일반화 관련 고려사항 (Appendix F.1 footnote 3):

> "만약 $P(G=k|\hat{G}=j)$가 테스트와 다른 분포를 가진 보조 데이터셋에서 추정된다면, 테스트셋에서 진짜 그룹 제약을 만족시키는 데 일반화 문제가 생길 수 있다."

**이는 향후 연구에서 다음을 고려해야 함을 시사합니다:**
- 보조 데이터셋과 주 데이터셋의 공변량 이동(covariate shift) 보정
- 유한 샘플에서의 노이즈 모델 추정 오류의 영향 분석

### 3.4 유한 샘플 수렴과 일반화

Duchi and Namkoong [19]의 DRO 이론을 통해 경험적 DRO 문제의 유한 샘플 수렴률이 보장되며, 이는 충분한 훈련 데이터가 있을 때 학습된 제약이 테스트셋으로 일반화될 수 있음을 암시합니다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4.1 연구 영향

**직접적 후속 연구:**
- **Narasimhan et al. (NeurIPS 2020) [51]:** 본 논문의 직접적 후속 작업으로, 대규모 제약 조건을 동시에 적용하는 일반화 방법 제안. 소프트 그룹 할당의 확장판으로 볼 수 있음

**공정성 연구 방향 전환:**
- 이전 연구들이 "깨끗한" 그룹 레이블을 전제했다면, 이 논문 이후 **불완전한 그룹 정보 하에서의 공정성**이 중요한 연구 방향으로 부상
- 실용적 공정성(Practical Fairness) 연구 활성화

**DRO와 공정성의 연결:**
- DRO를 공정성 보장에 체계적으로 적용하는 프레임워크 제시
- 이후 연구들이 다양한 발산 메트릭(Wasserstein, f-divergence 등)을 공정성에 적용하는 데 기반 제공

### 4.2 향후 연구 시 고려할 점

#### (1) 노이즈 모델 추정의 불확실성
현재 방법은 $P(G=j|\hat{G}=k)$가 정확히 추정된다고 가정합니다. 실제로는 소수의 보조 데이터로 추정하므로:
- **통계적 불확실성**을 반영한 신뢰구간 기반 접근법 필요
- Bootstrap 또는 Bayesian 방법으로 노이즈 모델 불확실성 전파

#### (2) DRO 이웃 크기 보정
Lemma 1의 $TV(p_j, \hat{p}_j) \leq P(G \neq \hat{G}|G=j)$는 상한이므로 느슨할 수 있습니다:
- **더 타이트한 상한** 개발 필요
- 데이터 적응형(adaptive) $\gamma_j$ 설정 방법 연구

#### (3) 교차성(Intersectionality) 처리
현재 방법은 단일 보호 속성을 가정합니다:
- 인종 × 성별과 같은 **교차 그룹**에서의 공정성 보장
- 교차 그룹의 노이즈가 더 심할 수 있음 (소수 집단의 데이터 부족)

#### (4) 연속적/부분적 그룹 멤버십
현재 이진/카테고리 그룹을 가정합니다:
- **퍼지(fuzzy) 그룹 멤버십** 처리 방법 연구
- 소프트 할당 방법의 자연스러운 확장 가능

#### (5) 확장성
현재 실험은 선형 모델에 국한:
- 심층 신경망(Deep Neural Networks)에 적용 시 계산 복잡도 증가
- 미니배치 기반 확률적 알고리즘의 수렴 보장 필요

#### (6) 보조 데이터셋 분포 불일치
논문 자체에서 미래 연구로 언급:
- 보조 데이터셋과 주 데이터셋 간 분포 차이가 있을 때의 영향 분석
- 도메인 적응(domain adaptation) 기법과의 결합

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

아래는 논문 내에서 직접 인용된 2020년 이후 연구 및 제가 확인할 수 있는 범위 내의 관련 연구입니다. **2020년 이후 직접 인용되지 않은 논문에 대한 구체적 내용은 확실성을 위해 제한적으로 서술합니다.**

### 5.1 논문에서 직접 인용된 2020년 관련 연구

| 논문 | 접근법 | 차이점 |
|------|--------|--------|
| **Lahoti et al. (NeurIPS 2020) [42]** "Fairness without Demographics through Adversarially Reweighted Learning" | 적대적 재가중치로 Rawlsian Max-Min 공정성 강제 | 그룹이 완전 미관측, 동등성(parity) 기반 제약 미사용 |
| **Kallus et al. (FAccT 2020) [37]** "Assessing Algorithmic Fairness with Unobserved Protected Class using Data Combination" | 소프트 그룹 할당으로 감사(auditing) | 학습(training)이 아닌 감사에 초점; 본 논문이 학습으로 확장 |
| **Narasimhan et al. (NeurIPS 2020) [51]** "Approximate Heavily-Constrained Learning with Lagrange Multiplier Models" | 대규모 제약 동시 처리 | 본 논문의 소프트 할당 방법을 일반화 |
| **Awasthi et al. (AISTATS 2020) [4]** "Equalized Odds Post-processing under Imperfect Group Information" | 사후 처리(post-processing), 조건부 독립 가정 | 사후 처리에 국한, 조건부 독립 필요 |
| **Mozannar et al. (2020) [47]** "Fair Learning with Private Demographic Data" | 예측기가 보호 속성과 독립일 때 결과 | 제한적 가정 하에서만 성립 |

### 5.2 비교 분석 요약

```
노이즈/미관측 그룹에서의 공정성 연구 위치:

[보수적 ←→ 정밀]
Hashimoto et al. (Max-Min, 그룹 완전 미지)
       ↓
Wang et al. (DRO, TV 거리 상한 사용)      ← 본 논문
       ↓
Wang et al. (Soft Assignments, 노이즈 모델 활용) ← 본 논문
       ↓
Kallus et al. (감사만, 최적 노이즈 모델 활용)
```

**본 논문의 독특한 위치:**
- Hashimoto et al.보다 더 많은 정보(노이즈 모델)를 활용하여 덜 보수적
- Kallus et al.의 감사 프레임워크를 **학습(training)**으로 확장
- 두 방법의 이론-실용성 트레이드오프를 명시적으로 분석

---

## 참고 자료 (출처)

본 답변은 다음 논문의 전문(full paper)을 기반으로 작성되었습니다:

**주요 참고 논문:**
- **Wang, S., Guo, W., Narasimhan, H., Cotter, A., Gupta, M., & Jordan, M. I. (2020).** "Robust Optimization for Fairness with Noisy Protected Groups." *NeurIPS 2020.* arXiv:2002.09343v3

**논문 내 인용된 주요 참고 문헌:**
- Kallus, N., Mao, X., & Zhou, A. (2020). "Assessing algorithmic fairness with unobserved protected class using data combination." FAccT 2020. [37]
- Duchi, J., & Namkoong, H. (2018). "Learning models with uniform performance via distributionally robust optimization." arXiv:1810.08750. [19]
- Namkoong, H., & Duchi, J. (2016). "Stochastic gradient methods for distributionally robust optimization with f-divergences." NeurIPS 2016. [48]
- Hardt, M., Price, E., & Srebro, N. (2016). "Equality of opportunity in supervised learning." NeurIPS 2016. [32]
- Narasimhan, H., Cotter, A., Zhou, Y., Wang, S., & Guo, W. (2020). "Approximate heavily-constrained learning with lagrange multiplier models." NeurIPS 2020. [51]
- Lahoti, P., et al. (2020). "Fairness without demographics through adversarially reweighted learning." arXiv:2006.13114. [42]
- Cotter, A., et al. (2019). "Optimization with non-differentiable constraints with applications to fairness, recall, churn, and other goals." JMLR, 20(172):1-59. [14]
- Cotter, A., Jiang, H., & Sridharan, K. (2019). "Two-player games for efficient non-convex constrained optimization." ALT 2019. [13]
- Hashimoto, T. B., et al. (2018). "Fairness without demographics in repeated loss minimization." ICML 2018. [33]
- Lamy, A., et al. (2019). "Noise-tolerant fair classification." NeurIPS 2019. [43]
