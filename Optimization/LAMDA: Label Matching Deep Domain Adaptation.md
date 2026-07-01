# LAMDA: Label Matching Deep Domain Adaptation 

---

## 1. 핵심 주장과 주요 기여 요약

### 1.1 핵심 주장

LAMDA 논문의 핵심 주장은 다음 두 가지로 압축됩니다:

1. **기존 도메인 불변 표현(domain invariant representation) 학습의 한계**: 소스와 타겟 도메인의 주변 레이블 분포(marginal label distribution)가 크게 다를 경우, 도메인 불변 표현을 강제하면 **레이블 시프트(label shift)가 증가**하여 타겟 도메인 성능이 저하된다.

2. **Wasserstein 거리 기반의 명시적 레이블 시프트 정량화**: 변환 $T$를 이용해 타겟 예시를 소스 예시와 연결함으로써 레이블 시프트를 **명시적으로 정의하고 측정**할 수 있는 새로운 이론적 프레임워크를 제안한다.

### 1.2 주요 기여

| 기여 항목 | 내용 |
|---|---|
| **새로운 이론적 프레임워크** | 변환 $T$를 통해 소스·타겟 가설을 결합하여 레이블 시프트를 명시적으로 정의 |
| **Wasserstein 기반 상한 도출** | 데이터 시프트와 레이블 시프트를 WS 거리로 상한 제시 (Theorem 1, 3) |
| **잠재 공간에서의 이론 분석** | 데이터 시프트 최소화 → 잠재 공간 분포 정렬 + 재구성 손실 최소화 (Theorem 4) |
| **도메인 불변 표현의 한계 이론화** | 강제적 도메인 불변 표현이 레이블 시프트를 증가시킬 수 있음을 수식으로 증명 (Theorem 8) |
| **LAMDA 알고리즘 제안** | 멀티클래스 판별기 + 최적 수송 비용을 활용한 레이블 매칭 DA 방법 |
| **실험적 우수성** | 다양한 실세계 데이터셋에서 SOTA 대비 성능 향상 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존 Deep Domain Adaptation(DDA)의 핵심 접근법은 소스와 타겟 도메인 간의 **데이터 시프트(data shift)**를 줄이기 위해 도메인 불변 잠재 표현을 학습하는 것이다. 그러나 이 접근법에는 두 가지 근본적 문제가 존재한다:

**문제 1: 레이블 시프트의 정의 부재**

기존 연구들은 레이블 시프트를 다음 두 가지 방식으로 해석했으나, 모두 한계가 있었다:
- $p^s(y|\mathbf{x}) \neq p^t(y|\mathbf{x})$: 소스·타겟 예시를 연결하는 메커니즘 부재
- $p^s(y) \neq p^t(y)$: 조건부 분포를 무시하는 단순한 접근

**문제 2: 도메인 불변 표현의 역효과**

Zhao et al. (2019)에서 지적되었듯이, 소스와 타겟의 주변 레이블 분포가 크게 다를 때 도메인 불변 표현을 강제하면 타겟 도메인의 전반적 손실이 증가한다. 기존 연구는 이진 분류, 결정론적 레이블링, 절대 손실 함수라는 제한적 설정에서만 이를 분석했다.

---

### 2.2 제안하는 방법 (수식 포함)

#### 2.2.1 이론적 설정

소스 가설 클래스 $\mathcal{H}^s := \{h^s: \mathcal{X}^s \to \Delta_C\}$와 변환 $T: \mathcal{X}^t \to \mathcal{X}^s$를 도입한다. 타겟 가설 클래스는 다음과 같이 유도된다:

$$\mathcal{H}^t := \{h^t: \mathcal{X}^t \to \Delta_C \mid h^t(\cdot) = h^s(T(\cdot)) \text{ for some } h^s \in \mathcal{H}^s\}$$

Push-forward 분포 

```math
\mathbb{P}^{\#} := T_{\#}\mathbb{P}^t
```

 (타겟을 $T$로 수송한 분포)와 일반 기대 손실:

$$R^{a,b}(h) := \int \ell(y, h(\mathbf{x})) p^b(y|\mathbf{x}) p^a(\mathbf{x}) \, dy \, d\mathbf{x}$$

성능 격차:

$$\Delta R(h^s, h^t) := \left| R^t(h^t) - R^s(h^s) \right|$$

#### 2.2.2 Theorem 1: 성능 격차 상한

> **가정 (A.1)**: $M := \sup_{h^s \in \mathcal{H}^s, \mathbf{x} \in \mathcal{X}^s, y \in \mathcal{Y}} |\ell(y, h^s(\mathbf{x}))| < \infty$

```math
\Delta R(h^s, h^t) \leq M \left( W_{c_{0/1}}\left(\mathbb{P}^s, \mathbb{P}^{\#}\right) + \mathbb{E}_{\mathbb{P}^t}\left[\|\Delta p(\cdot|\mathbf{x})\|_1\right] \right)
```

여기서:
$$\Delta p(\cdot|\mathbf{x}) := \left\| \left[ p^t(y=i|\mathbf{x}) - p^s(y=i|T(\mathbf{x})) \right]_{i=1}^{C} \right\|_1$$

- **첫 번째 항**

```math
W_{c_{0/1}}(\mathbb{P}^s, \mathbb{P}^{\#})
```
: **데이터 시프트** (소스 분포와 타겟의 push-forward 분포 간 Wasserstein 거리)

- **두 번째 항**

```math
\mathbb{E}_{\mathbb{P}^t}[\|\Delta p(\cdot|\mathbf{x})\|_1]
```

: **레이블 시프트** ( $p^t(y|\mathbf{x})$와 $p^s(y|T(\mathbf{x}))$ 간 발산의 기댓값)

#### 2.2.3 Theorem 3: 강화된 상한

> **가정 (A.2)**: $\ell$이 $\Delta_C$ 위의 노름 $\|\cdot\|$에 대해 $k$-Lipschitz

$h^s$가 최적 결합 

```math
\gamma^* \in \Gamma(\mathbb{P}^s, \mathbb{P}^{\#})
```

에 대해 $\phi$ -Lipschitz 전이 가능하면:

```math
\Delta R(h^s, h^t) \leq M\left(\mathbb{E}_{\mathbb{P}^t}[\|\Delta p(\cdot|\mathbf{x})\|_1] + 2\phi(\lambda)\right) + kC\lambda W_{c,p}\left(\mathbb{P}^s, \mathbb{P}^{\#}\right)
```

모든 $\lambda > 0$에 대해 성립한다.

#### 2.2.4 Theorem 4: 잠재 공간에서의 최적화 등치

$T = T^2 \circ T^1$ (여기서 $T^1: \mathcal{X}^t \to \mathcal{Z}$, $T^2: \mathcal{Z} \to \mathcal{X}^s$)일 때:

```math
\min_{T^1, T^2} W_{c,p}\left(\left(T^2 \circ T^1\right)_{\#} \mathbb{P}^t, \mathbb{P}^s\right) = \min_{T^1, T^2} \min_{\substack{G^1: T^1_{\#}\mathbb{P}^t = G^1_{\#}\mathbb{P}^s}} \mathbb{E}_{\mathbf{x} \sim \mathbb{P}^s}\left[c\left(\mathbf{x}, T^2\left(G^1(\mathbf{x})\right)\right)^p\right]^{1/p}
```

이를 완화(relaxation)하면:

```math
\min_{T^1, T^2, G^1} \left( \mathbb{E}_{\mathbf{x} \sim \mathbb{P}^s}\left[c\left(\mathbf{x}, T^2\left(G^1(\mathbf{x})\right)\right)^p\right]^{1/p} + \alpha D\left(G^1_{\#}\mathbb{P}^s, T^1_{\#}\mathbb{P}^t\right) \right)
```

#### 2.2.5 최종 최적화 목적함수

```math
\min_{T^1, T^2, G^1} \left( \beta \mathbb{E}_{\mathbf{x} \sim \mathbb{P}^s}\left[c\left(\mathbf{x}, T^2(G^1(\mathbf{x}))\right)^p\right]^{1/p} + \alpha D\left(G^1_{\#}\mathbb{P}^s, T^1_{\#}\mathbb{P}^t\right) + \mathbb{E}_{(\mathbf{x},y) \sim \mathcal{D}^s}\left[\ell\left(y, \mathcal{A}\left(G^1(\mathbf{x})\right)\right)\right] \right) \quad (5)
```

#### 2.2.6 Theorem 8: 도메인 불변 표현의 한계

$\tilde{c}(\mathbf{z}\_1, \mathbf{z}\_2) = \sup_{\mathcal{A} \in \mathcal{H}^a} \|\mathcal{A}(\mathbf{z}_1) - \mathcal{A}(\mathbf{z}_2)\|_1$을 정의하면, 주변 레이블 분포 차이에 대한 상한:

```math
\left\| \left[p^s(y=i) - p^t(y=i)\right]_{i=1}^C \right\|_1 \leq R^s_1(h^s) + R^t_1(h^t) + W_{\tilde{c},p}\left(G^1_{\#}\mathbb{P}^s, T^1_{\#}\mathbb{P}^t\right)
```

> **함의**:

```math
W_{\tilde{c},p}(G^1_{\#}\mathbb{P}^s, T^1_{\#}\mathbb{P}^t)
```

를 강제로 줄이면 $R^s_1(h^s) + R^t_1(h^t)$가 증가하여 타겟 분류 성능이 저하된다.

---

### 2.3 모델 구조

```
[Target Domain Xt]           [Source Domain Xs]
        |                           |
       T1 (Feature Extractor)      G1 (Feature Extractor)
        |                           |
        └─────────[Latent Space Z]──┘
                        |
             ┌──────────┴──────────┐
             |                     |
     Multi-class Discriminator d   Classifier A
     (C+1 출력)                    (Source supervised)
             |
        T2 (Decoder, Reconstruction)
```

**구성 요소:**
- **$G^1 = T^1 = G$** (실제 구현에서는 공유): 소스/타겟 도메인 → 잠재 공간
- **$T^2$**: 잠재 공간 → 소스 도메인 (재구성, ablation에서 $\beta=0$ 설정)
- **멀티클래스 판별기 $d$** ($C+1$ 출력): 소스 클래스 영역 식별 + 도메인 구분
- **분류기 $\mathcal{A}$**: 소스 레이블 감독 학습 ($\mathcal{A}$와 $S$ 공유)
- **수송 확률 네트워크 $S(\mathbf{x})$**: 타겟 샘플을 소스 클래스 영역으로 수송

**LAMDA 학습 목적함수:**

판별기 $d$ 업데이트:

$$\max_d \mathcal{L}_d := \sum_{i=1}^C \mathbb{E}_{(\mathbf{x},y) \sim \mathcal{D}^s \wedge y=i}\left[\log d_i\left(G^1(\mathbf{x})\right)\right] + \mathbb{E}_{\mathbf{x} \sim \mathbb{P}^t}\left[\log d_{C+1}\left(T^1(\mathbf{x})\right)\right] + \mathbb{E}_{\mathbf{x} \sim \mathbb{P}^s}\left[\log\left(1 - d_{C+1}\left(G^1(\mathbf{x})\right)\right)\right]$$

생성기 $G^1, T^1$ 업데이트:

$$\mathcal{L}_g := I(G^1) + J(T^1) + \alpha \cdot TC(T^1) + \beta R(T^2, G^1) + \mathcal{L}_\mathcal{A}$$

총 수송 비용(Total Transport Cost):

$$TC(T^1) := \mathbb{E}_{\mathbf{x} \sim \mathbb{P}^t}\left[-\sum_{i=1}^C S_i(\mathbf{x}) \log d_i\left(T^1(\mathbf{x})\right)\right]$$

**Min-Max 형태의 전체 목적함수:**

$$\max_d \min_{G^1, T^1, T^2, \mathcal{A}} \left( I(G^1) + J(T^1) + K(d, T^1) + \beta R(T^2, G^1) + \mathcal{L}_\mathcal{A} \right)$$

---

### 2.4 성능 향상

#### Office-Home (ResNet-50)

| Method | Avg Accuracy |
|---|---|
| ResNet-50 baseline | 46.1% |
| DANN | 57.6% |
| CDAN | 65.8% |
| SHOT | 71.8% |
| **LAMDA** | **72.0%** |

#### Office-31 (ResNet-50)

| Method | Avg Accuracy |
|---|---|
| RWOT | 90.8% |
| **LAMDA** | **93.0%** |

#### Digits (MNIST→SVHN, 특히 어려운 태스크)

| Method | Accuracy |
|---|---|
| 2위 방법 | 71.4% |
| **LAMDA** | **82.1%** (+10.7%) |

---

### 2.5 한계

논문에서 인정하거나 분석에서 도출되는 한계:

1. **비감독 DA에서 레이블 시프트 완전 해소 불가**: 타겟 레이블 부재 시 $T^1$이 타겟 클래스 영역을 잘못된 소스 클래스 영역으로 매핑할 가능성 존재 (Figure 2 참조)
2. **멀티클래스 판별기의 확장성**: 클래스 수 $C$가 매우 크면 $C+1$개 출력의 판별기 학습이 불안정해질 수 있음
3. **이론의 다양한 DA 설정 확장 미완성**: 멀티소스, 부분 DA, 오픈셋 DA 등으로의 확장은 미래 연구로 남김
4. **하이퍼파라미터 민감도**: $\alpha \in [0.1, 0.5]$ 범위 탐색 필요
5. **$\beta=0$ 설정**: 재구성 항의 실용적 효과가 제한적임을 경험적으로 확인했으나, 이론적 근거는 약함

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 이론적 일반화 보장

LAMDA의 일반화 성능은 Theorem 1, 3, 4, 8을 통해 다층적으로 보장된다.

**Theorem 3의 일반화 상한 해석:**

```math
\Delta R(h^s, h^t) \leq \underbrace{M \cdot \mathbb{E}_{\mathbb{P}^t}[\|\Delta p(\cdot|\mathbf{x})\|_1]}_{\text{레이블 시프트 항}} + \underbrace{2M\phi(\lambda)}_{\text{가설의 전이 가능성}} + \underbrace{kC\lambda W_{c,p}(\mathbb{P}^s, \mathbb{P}^{\#})}_{\text{데이터 시프트 항}}
```

- $\lambda$를 조정해 **데이터 시프트와 가설 전이 가능성 간의 트레이드오프** 제어
- $\phi(\lambda) \to 0$이면 데이터 시프트 최소화가 지배적
- 이는 LAMDA가 단순히 훈련 데이터에 과적합하지 않고 **이론적으로 타겟 일반화 손실의 상한이 감소**함을 보장

**Theorem 4의 일반화 의미:**

```math
\min_{T^1, T^2} W_{c,p}\left((T^2 \circ T^1)_{\#}\mathbb{P}^t, \mathbb{P}^s\right) = \min_{T^1,T^2,G^1} \left[\text{분포 정렬} + \text{재구성 손실}\right]
```

데이터 시프트 최소화가 자동으로:
1. 잠재 공간에서 소스·타겟 분포 정렬

( 

```math
D(G^1_{\#}\mathbb{P}^s, T^1_{\#}\mathbb{P}^t) \to 0
```

)

2. 재구성 손실 최소화 (모드 붕괴 방지)

를 동시에 달성하므로, **표현의 품질을 유지하면서 일반화**가 가능하다.

### 3.2 레이블 매칭을 통한 일반화 향상

**핵심 메커니즘**: 멀티클래스 판별기 $d$가 소스 클래스 영역을 명시적으로 정의하고, 수송 비용 $TC(T^1)$이 타겟 샘플을 **올바른 소스 클래스 영역으로 유도**:

$$TC(T^1) = \mathbb{E}_{\mathbf{x} \sim \mathbb{P}^t}\left[-\sum_{i=1}^C S_i(\mathbf{x}) \log d_i(T^1(\mathbf{x}))\right]$$

이는 최적 수송 이론에 기반한 비용으로, 타겟 샘플이 의미적으로 유사한 소스 클래스에 매핑되도록 강제하여 **의미론적 일관성(semantic consistency)**을 유지한다.

**이진 vs 멀티클래스 판별기의 일반화 차이:**

| 설정 | Office-Home Avg |
|---|---|
| 이진 판별기 | 68.7% |
| 멀티클래스 판별기 (LAMDA) | 72.0% |

멀티클래스 판별기가 클래스별 구조 정보를 보존하므로 타겟 도메인의 **클래스 경계 일반화**가 개선된다.

### 3.3 Corollary 6의 일반화 함의

$$\left\|\left[p^s(y=i) - p^t(y=i)\right]_{i=1}^C\right\|_1 \leq \text{레이블 시프트 항의 하한}$$

이 하한은 **주변 레이블 분포 차이가 클수록 도메인 불변 표현만으로는 일반화 한계**가 있음을 명시적으로 보여준다. LAMDA는 이를 해소하기 위해 레이블 매칭을 직접 최적화한다.

### 3.4 t-SNE 시각화를 통한 일반화 근거

논문 Figure 4에서:
- ResNet-50 단독: 소스(A)에서는 잘 분류되나 타겟(D)에서 혼재
- LAMDA: 31개 클래스가 명확히 분리된 클러스터 형성 → **소스·타겟 모두에서의 일반화**

---

## 4. 앞으로의 연구에 미치는 영향과 고려사항

### 4.1 앞으로의 연구에 미치는 영향

#### (1) 레이블 시프트 정의의 패러다임 전환

LAMDA는 변환 $T$를 통해 소스·타겟을 결합함으로써 레이블 시프트를 **처음으로 명시적이고 수학적으로 엄밀하게** 정의했다. 이는 이후 연구들이 레이블 시프트를 단순 주변 분포 차이가 아닌 **조건부 분포 차이**로 정량화하는 기반이 된다.

#### (2) Wasserstein 거리의 DA 이론에서의 표준화

WS 거리가 JS 거리보다 수치적 안정성과 연속성 측면에서 우월함을 이론적·실험적으로 보이며, **OT 기반 DA 이론의 표준 도구**로 자리매김하는 데 기여한다.

#### (3) 멀티클래스 판별기 설계의 새로운 방향

기존 이진 판별기 대신 $C+1$ 출력 멀티클래스 판별기를 도입함으로써, **클래스 인식(class-aware) 도메인 정렬**이라는 새로운 설계 원칙을 제시한다.

#### (4) 이론과 알고리즘의 긴밀한 연결

이론에서 도출된 최적화 목적함수(Eq. 5)가 직접 알고리즘(Algorithm 1)으로 구현되는 **이론 주도 알고리즘 설계**의 모범 사례를 제시한다.

---

### 4.2 앞으로 연구 시 고려할 점

#### (1) 다양한 DA 설정으로의 확장

논문 자체에서 미래 연구로 남긴 부분:

- **부분 DA(Partial DA)**: 타겟 클래스가 소스 클래스의 부분집합
- **오픈셋 DA(Open-set DA)**: 타겟 도메인에 소스에 없는 새로운 클래스 존재
- **멀티소스 DA**: 여러 소스 도메인 → 단일 타겟 (MOST, Nguyen et al., 2021a가 후속 연구)
- **도메인 일반화(Domain Generalization)**: 특정 타겟 도메인 없이 여러 도메인에 걸친 일반화

#### (2) 레이블 시프트 추정의 계산 비용

실제 구현에서 $\mathbb{E}_{\mathbb{P}^t}[\|\Delta p(\cdot|\mathbf{x})\|_1]$는 타겟 레이블이 없어 직접 계산 불가능하다. 이를 **간접적으로 추정하는 효율적인 방법** 개발이 필요하다.

#### (3) 대규모 클래스 수에서의 확장성

멀티클래스 판별기가 $C+1$ 출력을 가지므로, **클래스 수가 수백~수천 개인 fine-grained 분류** 문제에서는 판별기 학습 불안정 문제가 발생할 수 있다. 계층적 판별기나 프로토타입 기반 접근이 대안이 될 수 있다.

#### (4) 소스 없는 DA(Source-free DA)와의 통합

SHOT(Liang et al., 2020) 등 소스 데이터 없이 타겟만으로 적응하는 연구가 활발해지고 있다. LAMDA의 이론적 프레임워크를 **소스 가설만 사용 가능한 설정**으로 확장하는 연구가 필요하다.

#### (5) 전이 가능성 측정 자동화

$\phi$-Lipschitz 전이 가능성 조건이 실제 네트워크에서 만족되는지 **사전에 검증하는 방법**이 부재하다. 전이 가능성을 예측하는 메트릭(예: Transferability Estimation) 연구와의 결합이 중요하다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

논문에서 직접 비교한 2020년 이후 방법 및 이후 발전된 연구를 비교한다.

### 5.1 논문 내 포함된 2020년 이후 방법과의 비교

| 방법 | 핵심 아이디어 | Office-Home Avg | Office-31 Avg |
|---|---|---|---|
| **SHOT** (Liang et al., 2020, ICML) | 소스 없는 DA, 정보 극대화 + 의사 레이블 | 71.8% | 88.6% |
| **ETD** (Li et al., 2020, CVPR) | Enhanced Transport Distance | 67.3% | 86.2% |
| **RWOT** (Xu et al., 2020, CVPR) | Reliable Weighted OT | 67.6% | 90.8% |
| **MDD+Implicit Alignment** (Jiang et al., 2020) | 클래스 조건부 도메인 정렬 | 69.5% | 88.8% |
| **LAMDA** (Le et al., 2021) | 레이블 매칭 + 멀티클래스 판별기 + OT | **72.0%** | **93.0%** |

### 5.2 LAMDA 이후 관련 연구 방향 (논문에서 직접 인용되지 않은 연구는 제한적으로 서술)

논문에서 저자들의 후속 연구로 언급된 내용:

- **MOST** (Nguyen et al., 2021a, UAI): 멀티소스 DA에 OT + 학생-교사 학습 적용
- **TIDOT** (Nguyen et al., 2021b, IJCAI): 교사 모방 학습과 OT를 결합한 DA

### 5.3 LAMDA 대비 각 방법의 차별점

| 비교 항목 | SHOT | RWOT | LAMDA |
|---|---|---|---|
| 소스 데이터 필요 여부 | ✗ (Source-free) | ✓ | ✓ |
| 레이블 시프트 명시적 처리 | ✗ | 부분적 | ✓ |
| 이론적 상한 제공 | 부분적 | ✓ | ✓ |
| 멀티클래스 인식 판별기 | ✗ | ✗ | ✓ |
| OT 기반 비용 | ✗ | ✓ (Reliable WOT) | ✓ (TC) |
| 레이블 시프트 이론화 수준 | 낮음 | 중간 | **높음** |

---

## 참고자료

**주 논문:**
- Le, T., Nguyen, T., Ho, N., Bui, H., & Phung, D. (2021). **LAMDA: Label Matching Deep Domain Adaptation**. *Proceedings of the 38th International Conference on Machine Learning (ICML 2021)*, PMLR 139.

**논문 내 주요 인용 문헌:**
- Zhao, H., Des Combes, R. T., Zhang, K., & Gordon, G. (2019). On learning invariant representations for domain adaptation. *ICML 2019*.
- Courty, N., Flamary, R., Habrard, A., & Rakotomamonjy, A. (2017). Joint distribution optimal transportation for domain adaptation. *NeurIPS 2017*.
- Damodaran, B. B., et al. (2018). DeepJDOT: Deep joint distribution optimal transport for unsupervised domain adaptation. *ECCV 2018*.
- Liang, J., Hu, D., & Feng, J. (2020). Do we really need to access the source data? Source hypothesis transfer for unsupervised domain adaptation (SHOT). *ICML 2020*.
- Xu, R., et al. (2020). Reliable weighted optimal transport for unsupervised domain adaptation (RWOT). *CVPR 2020*.
- Li, M., et al. (2020). Enhanced transport distance for unsupervised domain adaptation (ETD). *CVPR 2020*.
- Ganin, Y., & Lempitsky, V. (2015). Unsupervised domain adaptation by backpropagation (DANN). *ICML 2015*.
- Long, M., et al. (2018). Conditional adversarial domain adaptation (CDAN). *NeurIPS 2018*.
- Villani, C. (2008). *Optimal Transport: Old and New*. Springer.
- Ben-David, S., et al. (2010). A theory of learning from different domains. *Machine Learning*.
- Redko, I., et al. (2017). Theoretical analysis of domain adaptation with optimal transport. *ECML-PKDD 2017*.
- Johansson, F. D., Sontag, D., & Ranganath, R. (2019). Support and invertibility in domain-invariant representations. *AISTATS 2019*.
- Tolstikhin, I. O., et al. (2018). Wasserstein auto-encoders. *ICLR 2018*.
- Arjovsky, M., Chintala, S., & Bottou, L. (2017). Wasserstein GAN. *ICML 2017*.

> **주의**: 2020년 이후 LAMDA를 후속 연구에서 비교한 구체적 논문들(예: CDTrans, PMTrans 등 Vision Transformer 기반 DA 연구)에 대한 세부 수치는 해당 논문 원문을 직접 확인하지 못했으므로 본 답변에서 수치 비교를 생략하였습니다. 확실히 확인된 정보만을 제공하였습니다.
