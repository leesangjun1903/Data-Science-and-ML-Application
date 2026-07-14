# Domain Adaptation with Asymmetrically-Relaxed Distribution Alignment

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

기존 도메인 적응(Domain Adaptation)의 표준 적대적(adversarial) 방법은 소스와 타겟 도메인의 잠재 공간(latent space) 분포를 **정확히 일치(exact matching)**시키려 한다. 그러나 이 논문은 이러한 방식이 레이블 분포 불일치(label distribution shift) 상황에서 **이론적으로 실패할 수밖에 없음**을 증명하고, 이를 극복하기 위해 **비대칭적으로 완화된 분포 정렬(Asymmetrically-Relaxed Distribution Alignment)**을 제안한다.

### 주요 기여 (논문 직접 인용 기반)

| 기여 | 내용 |
|------|------|
| **이론적 문제 제기** | 표준 DANN이 레이블 분포 이동 시 타겟 오류에 양의 하한을 강제함을 증명 (Proposition 3.1) |
| **새로운 정렬 목표** | 정확한 분포 일치 대신 밀도 비율(density ratio) 상한 조건으로 완화 |
| **이론적 보장** | 명확한 가정 하에서 타겟 도메인 성능 보장 제공 (Theorem 4.3) |
| **실용적 거리 함수** | f-발산, Wasserstein 거리, Reweighting 거리의 β-admissible 버전 도출 |
| **실증적 검증** | 합성 및 실제 데이터셋에서 성능 향상 입증 |

---

## 2. 세부 설명

### 2-1. 해결하고자 하는 문제

#### 표준 도메인 적응의 한계

표준 도메인 적대적 방법(DANN 등)은 다음 목표를 최소화한다:

$$\min_{\phi, h} \mathcal{E}_S(\phi, h) + \lambda D(p_S^\phi, p_T^\phi) + \Omega(\phi, h) $$

여기서 $D$는 분포 간 거리(JS-발산, Wasserstein 거리 등)이며, $D(p_S^\phi, p_T^\phi) = 0 \iff p_S^\phi \equiv p_T^\phi$이다.

타겟 도메인 에러는 다음과 같이 분해된다:

$$\mathcal{E}_T(\phi, h) = \mathcal{E}_S(\phi, h) + \underbrace{\int dz\, p_T^\phi(z)(r_T(z;\phi,h) - r_S(z;\phi,h))}_{\text{Term 2: 레이블 일관성}} + \underbrace{\int dz\left(p_T^\phi(z) - p_S^\phi(z)\right)r_S(z;\phi,h)}_{\text{Term 3: 분포 불일치}} $$

**핵심 문제**: 표준 방법은 Term 1과 Term 3을 최소화하지만, 이 최소화가 **Term 2를 임의로 증가**시킬 수 있다.

#### Proposition 3.1 (실패 증명)

$\rho_U = \int dx\, p_U(x)f(x)$ (양성 레이블 비율)로 정의할 때:

> **Proposition 3.1**: $D(p_S^\phi, p_T^\phi) = 0 \iff p_S^\phi \equiv p_T^\phi$이면, $\mathcal{E}_S(\phi, h) = D(p_S^\phi, p_T^\phi) = 0$은 다음을 함의한다:
> $$\mathcal{E}_T(\phi, h) \geq |\rho_S - \rho_T|$$

**직관적 예시**: 소스 도메인에 개 50%, 고양이 50%, 타겟 도메인에 개 25%, 고양이 75%인 경우, 정확한 분포 일치를 달성하면 타겟 정확도는 최대 75%에 불과하다.

---

### 2-2. 제안하는 방법 (수식 포함)

#### 핵심 아이디어: 밀도 비율 상한 조건

Term 3에 대해 다음 부등식이 성립한다:

$$\int dz\left(p_T^\phi(z) - p_S^\phi(z)\right)r_S(z;\phi,h) \leq \left(\sup_{z \in \mathcal{Z}} \frac{p_T^\phi(z)}{p_S^\phi(z)} - 1\right)\mathcal{E}_S(\phi, h) $$

따라서 $\mathcal{E}_S(\phi, h) = 0$이면, 밀도 비율이 상한으로 bounded된 경우 Term 3도 0이 된다.

**제안 조건**: 정확한 분포 일치($p_T^\phi \equiv p_S^\phi$) 대신:

$$\sup_{z \in \mathcal{Z}} \frac{p_T^\phi(z)}{p_S^\phi(z)} \leq 1 + \beta, \quad \beta \geq 0$$

이는 타겟 표현(representation) 지지(support)가 소스 지지에 포함됨을 의미한다.

#### β-admissible 거리 정의

> **Definition 3.3**: 분포 공간에서 거리 $D_\beta$가 **β-admissible**이라 함은:
> - $D_\beta(p, q) = 0$ when $\sup_{z \in \mathcal{Z}} p(z)/q(z) \leq 1 + \beta$
> - $D_\beta(p, q) > 0$ otherwise

#### 새로운 학습 목표

$$\min_{\phi, h} \mathcal{E}_S(\phi, h) + \lambda D_\beta(p_T^\phi, p_S^\phi) + \Omega(\phi, h)$$

**비대칭성의 의미**: $p_T^\phi / p_S^\phi$만 상한을 두고 ($p_S^\phi / p_T^\phi$는 제한 없음), 타겟이 소스 커버리지 안에 포함되도록 유도한다.

---

### 2-3. 구체적 거리 함수들

#### (i) f-발산 기반 β-admissible 거리

표준 f-발산 $D_f(p, q) = \int dz\, p(z) f\left(\frac{q(z)}{p(z)}\right)$에서, $f$를 부분 선형화(partially linearize):

$$\bar{f}_\beta(u) = \begin{cases} f(u) + C_{f,\beta} & \text{if } u \leq \frac{1}{1+\beta} \\ f'\!\left(\frac{1}{1+\beta}\right)u - f'\!\left(\frac{1}{1+\beta}\right) & \text{if } u > \frac{1}{1+\beta} \end{cases}$$

Adversarial training을 위한 dual form:

```math
D_{\bar{f}_\beta}(p, q) = \sup_{T:\mathcal{Z} \mapsto \text{dom}(\bar{f}^*_\beta)} \mathbb{E}_{z \sim q}[T(z)] - \mathbb{E}_{z \sim p}[f^*(T(z))]
```

JS-발산의 경우 (fDANN):

$$D_{\bar{f}_\beta}(p, q) = \sup_{g:\mathcal{Z} \mapsto (0,1]} \mathbb{E}_{z \sim q}\!\left[\log \frac{g(z)}{2+\beta}\right] + \mathbb{E}_{z \sim p}\!\left[\log\!\left(1 - \frac{g(z)}{2+\beta}\right)\right] $$

#### (ii) 완화된 Wasserstein 거리

$$W_\beta(p, q) = \inf_{\gamma \in \Pi_\beta(p,q)} \mathbb{E}_{(z_1, z_2) \sim \gamma}[\|z_1 - z_2\|]$$

여기서 $\Pi_\beta(p,q)$는 $\forall z_1: \int dz\, \gamma(z_1, z) = p(z_1)$, $\forall z_2: \int dz\, \gamma(z, z_2) \leq (1+\beta)q(z_2)$를 만족하는 결합분포 집합.

Dual form (WDANN):

$$W_\beta(p, q) = \sup_g \mathbb{E}_{z \sim p}[g(z)] - (1+\beta)\mathbb{E}_{z \sim q}[g(z)] $$

$s.t. \; \forall z \in \mathcal{Z},\; g(z) \geq 0;\quad \forall z_1, z_2 \in \mathcal{Z},\; g(z_1) - g(z_2) \leq \|z_1 - z_2\|$

#### (iii) Reweighting 거리 (sDANN)

임의의 거리 $D$에 대해 reweighting 함수 $w: \mathcal{Z} \mapsto [0,1]$을 이용:

$$D_\beta(p, q) = \min_{w \in \mathcal{W}_{\beta,q}} D(p, q_w) $$

여기서 

```math
\mathcal{W}_{\beta,q} = \left\{w: \mathcal{Z} \mapsto [0,1],\; \int dz\, q(z)w(z) = \frac{1}{1+\beta}\right\}
```

**Implicit-reweighting-by-sorting**: 미니배치에서 $f_2(g(z))$가 큰 순서로 $\frac{1}{1+\beta}$ 비율의 샘플에 $w(z)=1$ 할당.

---

### 2-4. 모델 구조

논문의 모델 구조는 표준 DANN의 확장이다:

```
입력 x
    ↓
[인코더 φ: X → Z] (L-Lipschitz 조건)
    ↓
┌───────────────────┐
│ 잠재 표현 z        │
└───────┬───────────┘
        ├──→ [레이블 분류기 h] → 소스 레이블 예측 (ε_S 최소화)
        └──→ [도메인 분류기 g] → β-admissible 거리 D_β 최소화/최대화
```

- **인코더**: 3-레이어 FC (합성) 또는 LeNet (이미지)
- **레이블 분류기**: 잠재 공간에서 레이블 예측
- **도메인 분류기**: β-admissible 거리 최적화 (JS, Wasserstein, Reweighting)

---

### 2-5. 이론적 보장 (Theorem 4.3)

**Theorem 4.3**: L-Lipschitz 매핑 $\phi$와 이진 분류기 $h$에 대해, Construction 4.1의 조건( $(L, \beta, \Delta, \delta_1, \delta_2)$ )과 Assumption 4.2 (소스-타겟 연결성)가 만족되면:

$$\mathcal{E}_T(\phi, h) \leq (1+\beta)\mathcal{E}_S(\phi, h) + 3\delta_1 + 2(1+\beta)\delta_2 + \delta_3$$

**구성 조건 요약**:
1. $\phi$가 L-Lipschitz: $d_\mathcal{Z}(\phi(x_1), \phi(x_2)) \leq L\, d_\mathcal{X}(x_1, x_2)$
2. 비대칭 완화 정렬: $\exists B \subset \mathcal{Z}$ s.t. $p_T^\phi(z)/p_S^\phi(z) \leq 1+\beta$ for all $z \in B$, and $p_T^\phi(B) \geq 1 - \delta_1$
3. 소스 도메인의 잠재 공간 분리: $C_0 \cap C_1 = \emptyset$, $p_S(C_0 \cup C_1) \geq 1-\delta_2$, $\inf_{z_0 \in \phi(C_0), z_1 \in \phi(C_1)} d_\mathcal{Z}(z_0, z_1) \geq \Delta > 0$

**Corollary 4.5**: Assumption 4.4 (타겟 지지가 레이블 일관 연결 클러스터로 구성, 각 클러스터가 소스와 겹침)가 만족되고 연속 매핑 $\phi$가 조건을 충족하면:

$$\mathcal{E}_S(\phi, h) = 0 \implies \mathcal{E}_T(\phi, h) = 0$$

---

### 2-6. 성능 향상

#### 합성 데이터셋 결과 (레이블 분포 이동: 소스 50:50, 타겟 10:90)

| 방법 | 정확도 (%) |
|------|-----------|
| Source-only | 89.4 ± 1.1 |
| DANN (표준) | 59.1 ± 5.1 |
| WDANN | 50.8 ± 32.1 |
| **sDANN-2** | **99.9 ± 0.0** |
| **fDANN-2** | **99.9 ± 0.0** |
| **WDANN2-2** | **99.7 ± 0.2** |

#### MNIST→USPS (레이블 이동 있음, 타겟 [0-4])

| 방법 | 정확도 (%) |
|------|-----------|
| Source-only | 74.3 ± 1.0 |
| DANN | 50.0 ± 1.9 |
| **sDANN-4** | **81.0 ± 1.6** |
| fDANN-4 | 75.9 ± 1.6 |

---

### 2-7. 한계

논문 자체에서 명시한 한계:

1. **연결성 가정의 한계**: 두 도메인이 완전히 disjoint인 경우 연결성 가정(Assumption 4.2)이 적용되지 않을 수 있다.
2. **β 선택의 어려움**: 최적 β는 타겟 레이블 분포에 의존하며, 이는 일반적으로 미지수이다. β가 너무 작으면 레이블 이동을 수용 못하고, 너무 크면 Term 3의 상한이 커진다.
3. **Lipschitz 연속성 강제의 어려움**: 신경망의 Lipschitz 상수를 정확히 제어하는 것은 여전히 미해결 문제이다.
4. **이론과 실제 간극**: 목표를 완벽히 최적화해도 원하는 정렬이 학습됨을 보장하기 어렵다 (Figure 2(a)의 실패 케이스).
5. **이진 분류 한정**: 이론적 분석이 이진 분류에 초점을 맞춤.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 일반화를 위한 이론적 메커니즘

#### 밀도 비율 조건과 일반화의 연결

수식 (5)가 핵심이다:

$$\int dz\left(p_T^\phi(z) - p_S^\phi(z)\right)r_S(z;\phi,h) \leq \left(\sup_{z \in \mathcal{Z}} \frac{p_T^\phi(z)}{p_S^\phi(z)} - 1\right)\mathcal{E}_S(\phi, h)$$

이 부등식은 **소스 오류가 낮고 밀도 비율이 bounded되면**, 타겟 도메인의 분포 불일치 항이 자동으로 제어됨을 보여준다. 정확한 분포 일치는 불필요하며, **타겟 support가 소스 support에 포함**되는 것만으로 충분하다.

#### 연결성 가정에 의한 레이블 일관성 보장

Theorem 4.3의 증명 핵심은: Lipschitz 연속 매핑이 연결된 영역을 마진이 있는 두 영역으로 분리할 수 없다는 성질이다. 타겟 데이터가 소스와 연결 경로를 가지면, **출처 도메인에서 배운 레이블 구조가 타겟으로 전파**된다.

구체적으로, Corollary 4.5는 다음 조건이 만족되면 완벽한 타겟 분류기를 보장한다:
- 타겟 support가 레이블 일관 연결 클러스터로 구성
- 각 클러스터가 소스 분포와 겹침
- 연속 매핑 $\phi$가 relaxed alignment 조건 만족

### 3-2. 표준 DANN 대비 일반화 이점

**표준 DANN의 일반화 실패 시나리오**:

$$\text{Proposition 3.1: } \mathcal{E}_S = 0, \; p_S^\phi \equiv p_T^\phi \implies \mathcal{E}_T \geq |\rho_S - \rho_T|$$

레이블 비율 차이가 클수록 타겟 오류 하한이 높아져 일반화가 원천적으로 불가능하다.

**제안 방법의 일반화 보장**:

$$\mathcal{E}_T(\phi, h) \leq (1+\beta)\mathcal{E}_S(\phi, h) + 3\delta_1 + 2(1+\beta)\delta_2 + \delta_3$$

- 소스 오류 $\mathcal{E}_S$를 낮추면 타겟 오류 상한도 낮아짐
- $\delta_1, \delta_2, \delta_3$는 정렬 품질과 데이터 분포의 관측 가능한 특성으로 결정
- 가정이 만족될 때 **소스에서 타겟으로의 지식 전이가 이론적으로 보장**

### 3-3. 비대칭성이 일반화에 기여하는 이유

정확한 분포 일치는 타겟과 소스의 레이블 비율을 동일하게 강제하여 다음과 같은 왜곡을 유발한다:

- 타겟의 양성 클래스 일부가 소스의 음성 클래스 영역으로 매핑됨
- 이는 레이블 경계를 흐트러뜨려 타겟 도메인 전반에 걸친 일반화를 저해

**비대칭 완화**는 타겟이 소스 안에 "포함(covered)"되면 충분하므로:
- 타겟의 클래스별 구조를 보존하면서 도메인 이동에 적응 가능
- 소스에서 학습된 결정 경계가 타겟에서도 유효하게 유지

---

## 4. 앞으로의 연구에 미치는 영향과 고려할 점

### 4-1. 연구에 미치는 영향

#### 이론적 영향

1. **도메인 적응 이론의 재정립**: 기존 Ben-David et al. (2010a)의 H-발산 기반 경계가 도메인 적대적 방법을 정당화하기에 불충분함을 명시적으로 보여줌. 이는 도메인 적응 이론 전반에 걸쳐 새로운 분석 프레임워크가 필요함을 시사한다.

2. **비대칭 분포 정렬의 일반화**: 단순히 두 분포를 일치시키는 것이 아닌, **한쪽 방향의 포함 관계(containment)**를 목표로 하는 새로운 패러다임을 제시. 이는 분포 이동 연구 전반(robust ML, out-of-distribution generalization 등)에 영향을 미칠 수 있다.

3. **모델 독립적 가정의 중요성**: 이론적 경계에서 φ-의존적(model-dependent) 가정을 피하고 데이터 분포에 관한 모델 독립적 가정을 사용하는 방법론 제시.

#### 알고리즘적 영향

1. **부분 도메인 적응(Partial Domain Adaptation)의 이론적 기반 강화**: 타겟이 소스의 일부 클래스만 포함하는 시나리오에 자연스럽게 연결.

2. **레이블 불균형(Class Imbalance) 문제와의 연결**: 레이블 분포 이동 처리 방식이 클래스 불균형 학습에 새로운 시각 제공.

3. **GAN 기반 생성 모델**: β-admissible 거리의 아이디어가 조건부 생성 모델에서의 분포 정렬에 응용 가능.

### 4-2. 2020년 이후 관련 최신 연구 비교 분석

> **주의**: 이하 2020년 이후 연구는 본 논문(arXiv:1903.01689v2)의 PDF에는 포함되어 있지 않습니다. 제가 가진 학습 데이터 기반으로 관련 연구를 제시하며, 일부 세부 사항은 부정확할 수 있습니다. 독자께서 직접 원문을 확인하시길 권장합니다.

#### 관련 연구 동향

**1. ILVR / Partial Domain Adaptation 계열**
- Cao et al. (2018)의 Selective Adversarial Networks에서 발전하여, 타겟 support가 소스의 부분집합인 상황에 대한 연구가 활발화. 본 논문의 비대칭 정렬 아이디어와 직접 연관.

**2. Label Distribution Shift 처리**
- Lipton et al. (2018) "Detecting and Correcting for Label Shift"와의 연결 선상에서, 레이블 이동을 명시적으로 추정하여 보정하는 방향의 연구 (예: BBSE, RLLS 등)가 발전. 본 논문은 추정 없이도 β 조절로 이를 암묵적으로 처리한다는 차이가 있음.

**3. 이론적 도메인 적응 경계 연구**
- Ben-David et al. 경계의 한계를 극복하려는 시도들 (예: Acuna et al. 2021 "Towards Optimal Strategies for Training Self-Supervised Video Representation Models", Zhao et al. 2019 등)이 유사한 문제의식을 공유.

**4. 조건부 도메인 적응**
- CDAN (Long et al., 2018) 등 클래스 조건부 정렬 방법들은 레이블 구조를 명시적으로 활용하여 분포 이동 처리. 본 논문과 상보적 접근.

#### 비교 표 (개념적)

| 특성 | 본 논문 (Wu et al., 2019) | 표준 DANN | 부분 DA 방법들 |
|------|--------------------------|-----------|--------------|
| 레이블 이동 처리 | ✅ (β 조절로 암묵적 처리) | ❌ | 부분적 |
| 이론적 보장 | ✅ (명확한 가정 하) | 불충분 | 제한적 |
| 타겟 레이블 불필요 | ✅ | ✅ | ✅ |
| 비대칭 정렬 | ✅ | ❌ | 일부 |

### 4-3. 앞으로 연구 시 고려할 점

#### 방법론적 확장

1. **β 자동 추정**: 현재 β는 수동 설정이 필요하다. 타겟 레이블 분포를 적응적으로 추정하는 방법 (예: label shift detection 방법과 결합)을 통해 자동화 필요.

2. **다중 클래스 확장**: 이론이 이진 분류에 집중됨. 다중 클래스 설정에서의 밀도 비율 조건과 이론적 경계 확장이 필요하다. 특히 클래스 수가 많은 경우 per-class density ratio 조건이 더 복잡해진다.

3. **Lipschitz 연속성 보장**: 신경망의 Lipschitz 상수 제어는 이론적 보장에 핵심이나 실용적으로 어렵다. Spectral Normalization (Miyato et al., 2018), Lipschitz regularization 등과의 통합 연구가 필요하다.

#### 이론적 확장

4. **완전히 disjoint한 도메인 처리**: 논문 자체에서 인정한 한계로, 두 도메인 support가 겹치지 않는 경우의 이론 개발이 필요하다.

5. **유한 샘플 보장**: 현재 이론은 모집단 수준에서의 결과이며 유한 샘플 근사 분석이 빠져 있다. PAC-learning 프레임워크와의 결합 연구 필요.

6. **다중 소스 도메인 확장**: 여러 소스 도메인에서 타겟으로의 비대칭 정렬 이론 개발.

#### 실용적 고려사항

7. **하이퍼파라미터 민감도**: β 선택이 성능에 크게 영향을 미친다. 실용적 가이드라인 (예: 검증 데이터 기반 선택 방법) 필요.

8. **대규모 데이터셋 확장성**: MNIST/USPS 수준을 넘어 ImageNet 규모의 데이터셋에서의 검증 필요.

9. **연결성 가정의 실용성 평가**: 실제 데이터에서 Assumption 4.4 (연결된 클러스터가 소스와 겹침)가 얼마나 자주 만족되는지 체계적 분석 필요.

---

## 참고자료

**주요 참고 논문 (본 논문 PDF에 포함된 참고문헌)**:

1. **Wu, Y., Winston, E., Kaushik, D., & Lipton, Z. (2019)**. "Domain Adaptation with Asymmetrically-Relaxed Distribution Alignment." arXiv:1903.01689v2

2. **Ganin, Y., et al. (2016)**. "Domain-adversarial training of neural networks." *The Journal of Machine Learning Research*, 17(1):2096–2030.

3. **Ben-David, S., et al. (2010a)**. "A theory of learning from different domains." *Machine learning*, 79(1-2):151–175.

4. **Ben-David, S., et al. (2010b)**. "Impossibility theorems for domain adaptation." *AISTATS*, pp. 129–136.

5. **Nowozin, S., Cseke, B., & Tomioka, R. (2016)**. "f-GAN: Training generative neural samplers using variational divergence minimization." *NIPS*, pp. 271–279.

6. **Arjovsky, M., Chintala, S., & Bottou, L. (2017)**. "Wasserstein GAN." arXiv:1701.07875.

7. **Gulrajani, I., et al. (2017)**. "Improved training of Wasserstein GANs." *NIPS*, pp. 5767–5777.

8. **Lipton, Z. C., Wang, Y.-X., & Smola, A. (2018)**. "Detecting and correcting for label shift with black box predictors." arXiv:1802.03916.

9. **Cao, Z., et al. (2018a)**. "Partial transfer learning with selective adversarial networks." *CVPR*, pp. 2724–2732.

10. **Cao, Z., et al. (2018b)**. "Partial adversarial domain adaptation." *ECCV*, pp. 139–155.

11. **Shu, R., et al. (2018)**. "A DIRT-T approach to unsupervised domain adaptation." arXiv:1802.08735.

12. **Shimodaira, H. (2000)**. "Improving predictive inference under covariate shift by weighting the log-likelihood function." *Journal of statistical planning and inference*, 90(2):227–244.
