# Understanding the Limits of Unsupervised Domain Adaptation via Data Poisoning

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

이 논문(Mehra et al., NeurIPS 2021)의 핵심 주장은 다음 두 가지입니다:

1. **이론적 주장**: 소스 도메인 오류와 주변 분포(marginal distribution) 불일치를 최소화하는 것만으로는 타겟 도메인 오류의 감소를 **보장할 수 없다**. 이는 표현(representation)에 의해 유도된 레이블링 함수 불일치(labeling function mismatch)가 증가할 수 있기 때문이다.

2. **실증적 주장**: 소스 도메인에 단 10%의 독성 데이터(poisoned data)만 추가해도 현재의 주요 UDA 방법들이 타겟 도메인 정확도를 거의 0%까지 떨어뜨릴 수 있다.

### 주요 기여

| 기여 | 내용 |
|------|------|
| 이론적 기여 | 타겟 도메인 오류에 대한 **하한(lower bound)** 증명 |
| 분석적 기여 | UDA가 성공/실패/불확정되는 3가지 데이터 분포 케이스 제시 |
| 실증적 기여 | 새로운 데이터 독화 공격(mislabeled, watermarking, clean-label) 제안 및 평가 |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

**비지도 도메인 적응(UDA)** 은 레이블이 있는 소스 도메인 $\mathcal{D}_S$와 레이블이 없는 타겟 도메인 $\mathcal{D}_T$ 간의 분포 차이($P_S \neq P_T$)를 극복하여 타겟에서도 좋은 성능을 내는 것을 목표로 합니다.

기존 UDA 알고리즘(DANN, CDAN 등)은 다음 **상한(upper bound)** [Ben-David et al., 2010]을 최소화합니다:

$$e_T(h) \leq e_S(h) + D_1(\tilde{p}_S, \tilde{p}_T) + \min\{e_T(\tilde{f}_S, \tilde{f}_T), e_S(\tilde{f}_S, \tilde{f}_T)\}$$

그러나 이 논문은 **"상한이 작다고 해서 타겟 오류가 작다는 보장이 없다"** 는 문제를 제기합니다. 구체적으로, 다음과 같은 문제를 해결하려 합니다:

- 기존 UDA의 실패 사례("negative transfer")를 이론적으로 설명
- UDA 방법들의 적대적 환경에서의 취약성 정량화

---

### 2.2 제안하는 방법 (수식 포함)

#### (A) 타겟 도메인 오류 하한 (Theorem 1)

**표기법 정의:**
- $g: \mathcal{X} \rightarrow \mathcal{Z}$: 입력을 표현 공간으로 매핑하는 표현 함수
- $h: \mathcal{Z} \rightarrow [0,1]$: 분류 가설
- $\tilde{f}(z) := \mathbb{E}_{\mathcal{D}}[f(x)|g(x) = z]$: 표현에 의해 유도된 레이블링 함수
- $D_1(\tilde{p}, \tilde{p}') = \int_{\mathcal{Z}} |\tilde{p}(z) - \tilde{p}'(z)| dz$: 전변동 거리(Total Variation Distance)

**Theorem 1 (하한):**

$$\boxed{e_T(h) \geq \max\{e_S(\tilde{f}_S, \tilde{f}_T),\ e_T(\tilde{f}_S, \tilde{f}_T)\} - e_S(h) - D_1(\tilde{p}_S, \tilde{p}_T)}$$

**Corollary 1.1 (KL 발산 버전):**

Pinsker 부등식 $D_1(p, p') \leq \sqrt{0.5 D_{KL}(p \| p')}$에 의해:

$$e_T(h) \geq \max\{e_S(\tilde{f}_S, \tilde{f}_T),\ e_T(\tilde{f}_S, \tilde{f}_T)\} - e_S(h) - \sqrt{0.5 D_{KL}(\tilde{p}_S \| \tilde{p}_T)}$$

**Corollary 1.2 (상한과 결합):**

$$|e_T(h) - e_S(\tilde{f}_S, \tilde{f}_T)| \leq e_S(h) + D_1(\tilde{p}_S, \tilde{p}_T)$$

$$|e_T(h) - e_T(\tilde{f}_S, \tilde{f}_T)| \leq e_S(h) + D_1(\tilde{p}_S, \tilde{p}_T)$$

**핵심 해석:**

기존 UDA는 다음을 최소화합니다:

$$\min_{g, h}\ e_S(h) + D_1(\tilde{p}_S, \tilde{p}_T) \tag{UDA Objective}$$

과파라미터화된 모델에서 $e_S(h) \approx 0$, $D_1(\tilde{p}_S, \tilde{p}_T) \approx 0$이 달성되면:

$$e_T(h) \geq \max\{e_S(\tilde{f}_S, \tilde{f}_T),\ e_T(\tilde{f}_S, \tilde{f}_T)\}$$

즉, **표현 공간에서 소스와 타겟의 유도 레이블링 함수가 불일치할 경우($e(\tilde{f}_S, \tilde{f}_T)$가 크면)**, UDA는 provably 실패합니다.

---

#### (B) 데이터 독화 공격

**공격 방법 1: 오레이블 독화 (Mislabeled Poisoning)**

- **Wrong-label correct-domain**: 소스 도메인 데이터에 잘못된 레이블 부여
- **Wrong-label incorrect-domain**: 타겟 도메인 데이터에 잘못된 레이블 부여 후 소스에 추가

**공격 방법 2: 워터마킹 독화 (Watermarking Attack)**

타겟 이미지 $t$와 소스 이미지 $s$를 볼록 결합(convex combination):

$$p = \alpha t + (1 - \alpha) s, \quad \alpha \in [0, 1]$$

$\alpha$는 타겟 이미지가 독화 이미지에서 시각적으로 보이지 않을 정도로 설정.

**공격 방법 3: 클린 레이블 독화 (Clean-label Poisoning)**

이중 수준 최적화(bilevel optimization)로 표현됩니다:

$$\min_{u} \sum_{i=1}^{N_{\text{poison}}} \| g(x_{\text{test}}^{\text{target}}; \theta) - g(u_i; \theta) \|_2^2$$

$$\text{s.t.} \quad \|x_i^{\text{base}} - u_i\| \leq \epsilon, \quad i = 1, \ldots, N_{\text{poison}}$$

$$\min_{\theta}\ \mathcal{L}_{\text{UDA}}\left(\hat{\mathcal{D}}^{\text{source}} \cup \hat{\mathcal{D}}^{\text{poison}},\ \hat{\mathcal{D}}^{\text{target}}; \theta\right) \tag{4}$$

- 첫 번째 문제: 독화 데이터를 타겟 테스트 포인트의 표현 근방에 배치
- 두 번째 문제: UDA 방법으로 표현 함수 $\theta$ 최적화
- 제약조건: 독화 데이터가 베이스 데이터에서 너무 멀어지지 않도록 보장 ($\epsilon$-ball 내)

---

### 2.3 모델 구조

논문은 새로운 모델 아키텍처를 제안하는 것이 아니라, 기존 UDA 방법들의 취약성을 분석합니다. 평가 대상 모델:

| 방법 | 핵심 메커니즘 |
|------|-------------|
| **DANN** | 도메인 판별기(discriminator)로 주변 분포 정렬 |
| **CDAN** | 분류기 출력 + 표현을 활용한 조건부 정렬 |
| **MCD** | 두 분류기의 불일치(discrepancy) 최대화/최소화 |
| **SSL** | 자기지도 보조 태스크(예: 회전각 예측)로 도메인 정렬 |
| **IW-DAN** | DANN + 중요도 가중(importance weighting) |
| **IW-CDAN** | CDAN + 중요도 가중 |

---

### 2.4 성능 향상 및 한계

#### 실험 결과 (성능 저하 관점)

**Digits 벤치마크 (Table 1 기반)**:

| 방법 | 독화 전 (Clean) | 독화 후 (Poison_target) |
|------|----------------|------------------------|
| DANN (MNIST→USPS) | 92.17% | **0.97%** |
| CDAN (MNIST→MNIST_M) | 73.88% | **0.59%** |
| MCD (MNIST→MNIST_M) | 93.95% | **0.37%** |
| SSL (SVHN→MNIST) | 66.85% | **0.31%** |

**Office-31 벤치마크 (Table 2 기반, D→A 태스크)**:

| 방법 | Clean | Poison_target |
|------|-------|---------------|
| DANN | 64.67% | **17.58%** |
| CDAN | 71.25% | **11.19%** |

Office-31의 경우 ImageNet 사전 학습 표현이 독화 효과를 완화하지만 완전히 제거하지는 못합니다.

#### 논문이 명시한 한계

1. **하한의 타이트니스**: 제안된 하한은 총 변동 거리(TV distance)에 의존하며, 이는 추정이 어려운 엄격한 거리 척도입니다. 다른 발산 메트릭으로의 확장은 미래 연구과제로 남겨두었습니다.

2. **클린 레이블 공격의 계산 복잡도**: 이중 수준 최적화 문제(Eq. 4)의 고계산 복잡도로 인해, 전체 타겟 도메인이 아닌 **단일 테스트 포인트**의 오분류만을 시연합니다.

3. **이진 분류 가정**: 이론적 분석은 이진 분류($f: \mathcal{X} \rightarrow [0,1]$)를 기반으로 하여, 다중 클래스로의 직접적인 확장이 필요합니다.

4. **방어 메커니즘 부재**: 논문은 공격을 제안하지만, 이에 대한 방어 방법은 제시하지 않습니다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 하한이 시사하는 일반화 조건

논문은 UDA의 성공을 위한 **필요조건**을 하한을 통해 명확히 합니다:

$$e_T(h) \geq \max\{e_S(\tilde{f}_S, \tilde{f}_T),\ e_T(\tilde{f}_S, \tilde{f}_T)\} - e_S(h) - D_1(\tilde{p}_S, \tilde{p}_T)$$

타겟 오류가 낮으려면($e_T(h) \approx 0$), 다음이 **동시에** 만족되어야 합니다:

$$e(\tilde{f}_S, \tilde{f}_T) \approx 0 \quad \text{AND} \quad e_S(h) \approx 0 \quad \text{AND} \quad D_1(\tilde{p}_S, \tilde{p}_T) \approx 0$$

### 3.2 세 가지 케이스 분석과 일반화 함의

**Case 1 (유리한 경우 - Favorable Case):**

소스와 타겟의 레이블링 함수가 일치 ($\tilde{f}_S = \tilde{f}_T$):

$$e(\tilde{f}_S, \tilde{f}_T) = 0 \Rightarrow e_T(h) \approx 0 \quad \text{(UDA 성공)}$$

**Case 2 (불리한 경우 - Unfavorable Case):**

동일한 UDA 목적함수 최소화가 레이블링 함수 불일치를 최대화:

$$e(\tilde{f}_S, \tilde{f}_T) = 1, \quad e_S(h) \approx 0, \quad D_1(\tilde{p}_S, \tilde{p}_T) \approx 0 \Rightarrow e_T(h) \approx 1$$

**Case 3 (모호한 경우 - Ambiguous Case):**

UDA 목적함수에 두 개의 동등한 전역 최솟값이 존재:

- 해 1: $e(\tilde{f}_S, \tilde{f}_T)$ 소 → UDA 성공
- 해 2: $e(\tilde{f}_S, \tilde{f}_T)$ 대 → UDA 실패
- **레이블 정보 없이는 두 해를 구별 불가** → 독화에 취약

### 3.3 일반화 성능 향상을 위한 시사점

논문은 다음과 같은 방향을 제시합니다:

1. **조건부 분포 정렬의 중요성**: 주변 분포 정렬($D_1(\tilde{p}_S, \tilde{p}_T) \approx 0$)만으로 불충분하며, **클래스 조건부 분포**( $p(z|y)$ )를 정렬해야 타겟 일반화가 보장됩니다.

2. **레이블링 함수 불일치 모니터링**: $e(\tilde{f}_S, \tilde{f}_T)$를 추정하거나 최소화하는 알고리즘 설계가 필요합니다.

3. **적대적 평가의 필요성**: 벤치마크 데이터셋에서의 성능만으로는 UDA 방법의 실제 일반화 능력을 측정할 수 없으며, 독화 공격과 같은 적대적 평가가 필요합니다.

4. **pseudo-label 활용의 양면성**: CDAN의 경우 pseudo-label이 정확할 때 클래스 정렬에 도움이 되지만(독화 저항성), 부정확해지면 오히려 역효과가 납니다.

---

## 4. 향후 연구에 미치는 영향 및 고려 사항

### 4.1 향후 연구에 미치는 영향

#### (A) UDA 이론 연구

- 기존 상한 중심의 분석에서 **하한-상한 결합 분석**으로 패러다임 전환을 촉구합니다.
- $e(\tilde{f}_S, \tilde{f}_T)$를 직접 최소화하는 새로운 목적함수 설계의 필요성을 제시합니다.
- TV 거리 이외의 다양한 발산 메트릭(Wasserstein, MMD 등)으로 하한을 확장하는 연구가 필요합니다.

#### (B) 강건한 UDA 알고리즘 연구

- 독화 데이터에 강건한 UDA 알고리즘 설계 (robust UDA)에 대한 연구 동기를 제공합니다.
- 독화 탐지(poisoning detection) + UDA의 결합 연구를 촉진합니다.

#### (C) 평가 방법론 연구

- 벤치마크 평가를 넘어 **적대적 설정에서의 평가**를 UDA 방법론의 표준 평가 프로토콜로 확립하는 데 기여합니다.

### 4.2 2020년 이후 관련 최신 연구 비교 분석

아래는 본 논문과 관련된 2020년 이후 연구들의 비교입니다. **단, 이하 논문들은 본 PDF에 직접 인용된 것과 제 학습 지식을 기반으로 제시하며, 세부 수치는 해당 논문 원문 확인을 권장합니다.**

#### 관련 연구 1: Zhao et al. (ICML 2019, 논문 내 인용 [36])
**"On Learning Invariant Representations for Domain Adaptation"**

$$e_T(h) \leq e_S(h) + d_{\mathcal{H}\Delta\mathcal{H}}(\tilde{p}_S, \tilde{p}_T) + \lambda^*$$

여기서 $\lambda^*$는 이상적인 결합 가설의 오류. 이 논문은 **레이블 분포 차이가 있을 때 도메인 불변 표현이 실패함**을 보였으나, 본 논문은 이를 표현 공간의 레이블링 함수 불일치로 더 직접적으로 설명합니다.

| 비교 항목 | Zhao et al. (2019) | Mehra et al. (2021) |
|-----------|-------------------|---------------------|
| 분석 방향 | 상한(upper bound) | 하한(lower bound) |
| 실패 설명 | 레이블 분포 차이 | 유도 레이블링 함수 불일치 |
| 적대적 평가 | 없음 | 데이터 독화 공격 |

#### 관련 연구 2: des Combes et al. (2020, 논문 내 인용 [6])
**"Domain Adaptation with Conditional Distribution Matching and Generalized Label Shift"**

IW-DAN, IW-CDAN 제안. 레이블 분포 불일치를 중요도 가중으로 보정. 그러나 본 논문 결과에 따르면 IW-DAN, IW-CDAN도 독화에 취약합니다(Table 2).

#### 관련 연구 3: 조건부 분포 정렬 연구 동향 (2020 이후)

본 논문의 핵심 통찰—**주변 분포가 아닌 조건부 분포 정렬이 중요**—은 이후 다음과 같은 연구 방향을 촉진했습니다:

- **ATDOC** (Liu et al., 2021): 타겟 도메인의 지역 구조를 활용한 클래스 조건부 정렬
- **CDTrans** (Xu et al., 2021): Transformer 기반 크로스 도메인 조건부 정렬
- **ToAlign** (Wei et al., 2021): 태스크 지향적 정렬

이들 연구는 모두 $p_S(z|y) \approx p_T(z|y)$를 달성하는 방향으로 발전하였으며, 이는 본 논문의 하한이 시사하는 $e(\tilde{f}_S, \tilde{f}_T) \approx 0$ 조건과 일맥상통합니다.

### 4.3 앞으로 연구 시 고려할 점

#### (A) 이론적 측면

1. **다중 클래스 및 회귀로의 확장**: 현재 하한은 이진 분류($f: \mathcal{X} \rightarrow [0,1]$)에 한정되어 있어, 다중 클래스($K > 2$) 및 회귀 문제로의 확장이 필요합니다.

2. **더 타이트한 하한 개발**: TV 거리 대신 추정 가능한 발산 메트릭(MMD, Wasserstein 등)을 사용한 하한 개발:

$$e_T(h) \geq f\left(e(\tilde{f}_S, \tilde{f}_T),\ e_S(h),\ \text{MMD}(\tilde{p}_S, \tilde{p}_T)\right)$$

3. **PAC-Bayes 틀과의 통합**: 확률적 모델에 대한 일반화 보장으로 확장.

#### (B) 알고리즘 측면

1. **$e(\tilde{f}_S, \tilde{f}_T)$ 직접 최소화**: 유도 레이블링 함수 불일치를 명시적으로 최소화하는 목적함수:

$$\min_{g, h}\ e_S(h) + D_1(\tilde{p}_S, \tilde{p}_T) + \lambda \cdot e(\tilde{f}_S, \tilde{f}_T)$$

단, $\tilde{f}_T$는 관찰 불가능하므로 pseudo-label이나 엔트로피 최소화로 근사가 필요합니다.

2. **강건한 UDA (Robust UDA)**: 독화 데이터에 강건한 UDA 알고리즘 개발. 예를 들어, 소스 데이터에 대한 이상치 탐지(outlier detection)와 UDA를 결합.

3. **데이터 독화 방어**: 독화 데이터 필터링, 데이터 증강(augmentation)을 통한 강건성 향상.

#### (C) 평가 측면

1. **표준화된 적대적 UDA 벤치마크 구축**: 독화 공격 하에서의 UDA 성능을 표준 평가 지표로 도입.

2. **현실적인 공격 시나리오**: 공격자가 타겟 도메인 레이블 정보에 부분적으로만 접근하는 **회색 박스(gray-box)** 시나리오 연구.

3. **클린 레이블 공격의 효율화**: 이중 수준 최적화의 계산 효율을 높여 전체 타겟 도메인에 대한 클린 레이블 공격을 가능하게 하는 연구.

---

## 참고자료

**주요 참고 논문 (논문 내 인용):**

1. **Mehra, A., Kailkhura, B., Chen, P.-Y., & Hamm, J. (2021).** "Understanding the Limits of Unsupervised Domain Adaptation via Data Poisoning." *NeurIPS 2021.*

2. **Ben-David, S., Blitzer, J., Crammer, K., Kulesza, A., Pereira, F., & Vaughan, J. W. (2010).** "A theory of learning from different domains." *Machine Learning, 79(1):151–175.*

3. **Ben-David, S., Blitzer, J., Crammer, K., Pereira, F. (2007).** "Analysis of representations for domain adaptation." *NIPS 2007.*

4. **Zhao, H., Des Combes, R. T., Zhang, K., & Gordon, G. (2019).** "On learning invariant representations for domain adaptation." *ICML 2019.*

5. **Ganin, Y., et al. (2016).** "Domain-adversarial training of neural networks." *JMLR, 17(1):2096–2030.*

6. **Long, M., Cao, Z., Wang, J., & Jordan, M. I. (2017).** "Conditional adversarial domain adaptation." *arXiv:1705.10667.*

7. **Saito, K., Watanabe, K., Ushiku, Y., & Harada, T. (2018).** "Maximum classifier discrepancy for unsupervised domain adaptation." *CVPR 2018.*

8. **des Combes, R. T., Zhao, H., Wang, Y.-X., & Gordon, G. (2020).** "Domain adaptation with conditional distribution matching and generalized label shift." *arXiv:2003.04475.*

9. **Johansson, F. D., Sontag, D., & Ranganath, R. (2019).** "Support and invertibility in domain-invariant representations." *AISTATS 2019.*

10. **Wang, Z., Dai, Z., Póczos, B., & Carbonell, J. (2019).** "Characterizing and avoiding negative transfer." *CVPR 2019.*

11. **Xu, J., Xiao, L., & López, A. M. (2019).** "Self-supervised domain adaptation for computer vision tasks." *IEEE Access, 7:156694–156706.*

12. **Shafahi, A., et al. (2018).** "Poison frogs! Targeted clean-label poisoning attacks on neural networks." *arXiv:1804.00792.*

> **⚠️ 주의**: 4.2절의 ATDOC, CDTrans, ToAlign 등 2020년 이후 연구 비교는 제 학습 데이터 기반의 일반적 지식에 근거하며, 해당 논문 원문 직접 확인을 권장합니다. 세부 수치나 주장은 원문 PDF에서 검증된 내용만 포함하였습니다.
