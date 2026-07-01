# Global-Local Regularization Via Distributional Robustness (GLOT-DR)

> **참고 자료:**
> - Phan, H., Le, T., Phung, T., Bui, A., Ho, N., & Phung, D. (2023). "Global-Local Regularization Via Distributional Robustness." *Proceedings of the 26th International Conference on Artificial Intelligence and Statistics (AISTATS 2023)*. PMLR: Volume 206. arXiv:2203.00553v3.
> - 논문 내 인용 문헌 (Blanchet & Murthy, 2019; Sinha et al., 2018; Liu & Wang, 2016; Zhang et al., 2019; Zhao et al., 2020 등)

---

## 1. 핵심 주장 및 주요 기여 (요약)

### 핵심 주장
기존의 Wasserstein 기반 분포 강건성(Distributional Robustness, DR) 프레임워크는 **지역(local) 정규화만** 수행하고, 원본 분포와 가장 어려운(challenging) 분포를 **분리(decouple)**하여 모델링 능력이 제한된다. 이 논문은 이를 극복하기 위해 **전역(global) + 지역(local) 정규화**를 동시에 수행하는 새로운 OT(Optimal Transport) 기반 DR 프레임워크 **GLOT-DR (Global-Local Optimal Transport based Distributional Robustness)**를 제안한다.

### 주요 기여 요약

| 기여 | 내용 |
|---|---|
| **이론적 기여** | 원본 분포와 가장 어려운 분포를 결합(couple)하는 특수 결합 분포(joint distribution) 및 Wasserstein 불확실성 설계 |
| **방법론적 기여** | 쌍대형(dual form) 없이 **폐쇄형(closed-form) 해** 유도 |
| **실용적 기여** | DG, DA, SSL, AML 등 4가지 실제 학습 문제에 통합 적용 가능한 범용 프레임워크 제공 |
| **성능적 기여** | 각 도메인에서 최신 기법 대비 일관된 성능 향상 입증 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존 WDR(Wasserstein Distributional Robustness)의 표준 공식은 다음과 같다:

$$\sup_{\tilde{\mathbb{P}}: \mathcal{W}_c(\mathbb{P}, \tilde{\mathbb{P}}) < \epsilon} \mathbb{E}_{\tilde{Z} \sim \tilde{\mathbb{P}}} \left[ r\left(\tilde{Z}\right) \right] \tag{1}$$

이의 쌍대형(dual form)은:

```math
\inf_{\lambda \geq 0} \left\{ \lambda\epsilon + \mathbb{E}_{Z \sim \mathbb{P}} \left[ \sup_{\tilde{Z}} \left\{ r\left(\tilde{Z}\right) - \lambda c\left(Z, \tilde{Z}\right) \right\} \right] \right\}
```

**문제점 1:** 위험 함수 $r$이 오직 $\tilde{Z} \sim \tilde{\mathbb{P}}$만 관여하여, 원본 샘플 $Z$와 섭동 샘플 $\tilde{Z}$를 **동시에 포함하는 위험 함수**(예: TRADES의 $KL(p(\tilde{Z}) \| p(Z))$ )를 표현하기 어렵다.

**문제점 2:** 배치(batch) 수준의 **전역 정규화 항**(예: 도메인 간 분포 정렬)을 삽입하는 것이 구조적으로 불가능하다.

**문제점 3:** 쌍대형의 $\lambda$에 대한 최소화가 계산적으로 다루기 어렵다(computationally intractable).

---

### 2.2 제안하는 방법 (수식 포함)

#### 2.2.1 앵커 샘플과 섭동 샘플의 구조적 정의

원본 확률변수 $Z$와 섭동 확률변수 $\tilde{Z}$를 다음과 같이 정의한다:

$$Z := \left[ \left[ \left[ X^S_{kij}, Y^S_{kij} \right]^K_{k=1} \right]^{B^S_k}_{i=1} \right]^{n^S}_{j=0}, \quad \left[ \left[ X^T_{ij} \right]^{B^T}_{i=1} \right]^{n^T}_{j=0} \tag{3}$$

$$\tilde{Z} := \left[ \left[ \left[ \tilde{X}^S_{kij}, \tilde{Y}^S_{kij} \right]^K_{k=1} \right]^{B^S_k}_{i=1} \right]^{n^S}_{j=0}, \quad \left[ \left[ \tilde{X}^T_{ij} \right]^{B^T}_{i=1} \right]^{n^T}_{j=0} \tag{4}$$

- $j=0$: **앵커 샘플** (섭동 없음, $\tilde{X}^S_{ki0} = X^S_{ki}$)
- $j \geq 1$: **섭동 샘플** ($\epsilon$-ball 내 섭동 허용)

#### 2.2.2 Wasserstein-ball 제약 하의 비용 메트릭 $\rho$

$$\rho\left(Z, \tilde{Z}\right) := \underbrace{\infty \sum_{k=1}^{K}\sum_{i=1}^{B^S_k} \left\| X^S_{ki0} - \tilde{X}^S_{ki0} \right\|^q_p + \infty \sum_{i=1}^{B^T} \left\| X^T_{i0} - \tilde{X}^T_{i0} \right\|^q_p}_{\text{앵커 유지 강제 (}j=0\text{)}} + \underbrace{\sum_{k=1}^{K}\sum_{i=1}^{B^S_k}\sum_{j=1}^{n^S} \left\| X^S_{kij} - \tilde{X}^S_{kij} \right\|^q_p + \sum_{i=1}^{B^T}\sum_{j=1}^{n^T} \left\| X^T_{ij} - \tilde{X}^T_{ij} \right\|^q_p}_{\text{섭동 허용}} + \underbrace{\infty \sum_{k=1}^{K}\sum_{i=1}^{B^S_k}\sum_{j=0}^{n^S} \rho_l\left(Y^S_{kij}, \tilde{Y}^S_{kij}\right)}_{\text{레이블 섭동 금지}}$$

이 비용 메트릭은 거의 확실히(almost surely): **(i)** $j=0$ 샘플을 앵커로 고정, **(ii)** $j \geq 1$에 대해 입력 섭동 허용, **(iii)** 레이블 섭동 금지를 보장한다.

#### 2.2.3 핵심 최적화 문제 (DR)

$$\min_{\theta, \phi} \max_{\tilde{\mathbb{P}}: \mathcal{W}_\rho(\mathbb{P}, \tilde{\mathbb{P}}) \leq \epsilon} \mathbb{E}_{\tilde{Z} \sim \tilde{\mathbb{P}}} \left[ r\left(\tilde{Z}; \phi, \theta\right) \right] \tag{5}$$

비용 함수:

$$r\left(\tilde{Z}; \phi, \theta\right) := \alpha r^l\left(\tilde{Z}; \phi, \theta\right) + \beta r^g\left(\tilde{Z}; \phi, \theta\right) + \mathcal{L}\left(\tilde{Z}; \phi, \theta\right)$$

- $r^l$: **지역 정규화** (앵커-섭동 샘플 간 차이 최소화, 국소 평활성 강제)
- $r^g$: **전역 정규화** (배치 수준의 도메인 간 분포 정렬)
- $\mathcal{L}$: **분류 손실** (교차 엔트로피)

#### 2.2.4 엔트로픽 정규화 및 동치 문제

Eq. (5)를 결합 분포 $\gamma$에 대한 탐색으로 변환 (Lemma 3.1):

$$\min_{\theta, \phi} \max_{\gamma \in \Gamma_\epsilon} \mathbb{E}_{(Z, \tilde{Z}) \sim \gamma} \left[ r\left(\tilde{Z}; \phi, \theta\right) \right] \tag{6}$$

엔트로픽 정규화 추가:

```math
\min_{\theta, \phi} \max_{\gamma \in \Gamma_\epsilon} \left\{ \mathbb{E}_{(Z, \tilde{Z}) \sim \gamma} \left[ r\left(\tilde{Z}; \phi, \theta\right) \right] + \frac{1}{\lambda} \mathbb{H}(\gamma) \right\}
```

#### 2.2.5 폐쇄형 해 (Theorem 3.2)

$q = \infty$ 조건에서 내부 최대화 문제의 최적 결합 분포 $\gamma^*$:

$$\gamma^*\left(Z, \tilde{Z}\right) = \prod_{k=1}^{K}\prod_{i=1}^{B^S_k}\prod_{j=0}^{n^S_k} p^S_k\left(X^S_{ki}, Y^S_{ki}\right) \prod_{i=1}^{B^T}\prod_{j=0}^{n^T} p^T\left(X^T_i\right) \cdot \prod_{k=1}^{K}\prod_{i=1}^{B^S_k}\prod_{j=0}^{n^S_k} q^S_{ki}\left(\tilde{X}^S_{kij} \mid X^S_{ki}, Y^S_{ki}; \psi\right) \prod_{i=1}^{B^T}\prod_{j=1}^{n^T} q^T_i\left(\tilde{X}^T_{ij} \mid X^T_i; \psi\right) \tag{8}$$

여기서 **지역 분포**:

```math
q^S_{ki}\left(\tilde{X}^S_{kij} \mid X^S_{ki}, Y^S_{ki}; \psi\right) \propto \exp\left\{\lambda\left[\alpha s\left(X^S_{ki}, \tilde{X}^S_{kij}; \psi\right) + \ell\left(\tilde{X}^S_{kij}, Y^S_{ki}; \psi\right)\right]\right\}
```

```math
q^T_i\left(\tilde{X}^T_{ij} \mid X^T_i; \psi\right) \propto \exp\left\{\lambda \alpha s\left(X^T_i, \tilde{X}^T_{ij}; \psi\right)\right\}
```

#### 2.2.6 최종 최적화 목적함수 (Eq. 9, 10)

$$\min_{\psi} \mathbb{E}_{\forall k: (X^S_{ki}, Y^S_{ki})^{B^S_k}_{i=1} \overset{iid}{\sim} \mathbb{P}^S_k,\ X^T_{1:B^T} \overset{iid}{\sim} \mathbb{P}^T} \left[ r\left(\tilde{Z}; \psi\right) \right] \tag{9}$$

$$r\left(\tilde{Z}; \psi\right) = \underbrace{\mathbb{E}_{[\tilde{X}^S_{kij}]_j \sim q^S_{ki}} \left[\alpha s(X^S_{ki}, \tilde{X}^S_{kij}; \psi) + \ell(\tilde{X}^S_{kij}, Y^S_{ki}; \psi)\right]}_{\text{지역 정규화 + 분류 손실}} + \underbrace{\mathbb{E}_{[\tilde{X}^T_{ij}]_j \sim q^T_i} \left[\alpha s\left(X^T_i, \tilde{X}^T_{ij}; \psi\right)\right]}_{\text{타겟 지역 정규화}} + \underbrace{\beta r^g\left(\left[X^S_{ki}\right]_{k,i}, \left[X^T_i\right]_i; \psi\right)}_{\text{전역 정규화}} \tag{10}$$

**지역 정규화 함수 $s$:** 대칭 KL 발산 사용

$$s\left(X, \tilde{X}; \psi\right) = \frac{1}{2} KL\left(f_\psi(X) \| f_\psi\left(\tilde{X}\right)\right) + \frac{1}{2} KL\left(f_\psi\left(\tilde{X}\right) \| f_\psi(X)\right)$$

---

### 2.3 태스크별 전역 정규화 함수 정의

#### Domain Adaptation (DA) / Semi-Supervised Learning (SSL)

$$r^g = \mathcal{W}_d\left(\frac{1}{B^S}\sum_{i=1}^{B^S}\delta_{U^S_i},\ \frac{1}{B^T}\sum_{j=1}^{B^T}\delta_{U^T_j}\right) \tag{11}$$

$$d\left(U^S_i, U^T_j\right) := \rho_d\left(g_\phi(X^S_i), g_\phi(X^T_j)\right) + \gamma \rho_l\left(h_\theta(g_\phi(X^S_i)), h_\theta(g_\phi(X^T_j))\right)$$

→ 소스-타겟 도메인 간 특징 및 예측 분포의 불일치 감소

#### Domain Generalization (DG)

$$r^g = \sum_{m=1}^{M} \sum_{k=1}^{K} \frac{1}{K} \mathcal{W}_d\left(\tilde{\mathbb{P}}_{km}, \tilde{\mathbb{P}}_m\right) \tag{13}$$

→ 클래스별로 도메인 간 특징 분포를 정렬하여 도메인 불변 표현 학습

#### Adversarial Machine Learning (AML)

$$r^g = \mathcal{W}_d\left(\frac{1}{B^S_1}\sum_{i=1}^{B^S_1}\delta_{U^S_i},\ \frac{1}{B^S_1 n^S}\sum_{i=1}^{B^S_1}\sum_{j=1}^{n^S}\delta_{U^S_{ij}}\right) \tag{14}$$

$$d\left(U^S_i, U^S_{\bar{i}j}\right) = \mathbb{I}_{Y^S_{1i} = Y^S_{1\bar{i}}} \left[\rho_d\left(g_\phi(X^S_{1i}), g_\phi(X^S_{1\bar{i}j})\right) + \gamma \rho_l\left(h_\theta(g_\phi(X^S_{1i})), h_\theta(g_\phi(X^S_{1\bar{i}j}))\right)\right]$$

→ 적대적 예제를 같은 레이블의 정상 예제 군집으로 이동시켜 강건성 향상

---

### 2.4 모델 구조

```
입력 데이터
    ↓
특징 추출기 (Feature Extractor) g_φ
    ↓
잠재 공간 (Latent Space)
    ├── 지역 분포 q_ki^S, q_i^T (SVGD로 섭동 샘플 생성)
    │       ↓
    │   섭동 샘플 X̃ (ε-ball 내)
    │       ↓
    │   지역 정규화: s(X, X̃; ψ) [대칭 KL]
    │
    └── 전역 정규화: W_d(·, ·) [엔트로픽 정규화 OT]
            ├── Kantorovich 포텐셜 네트워크 φ (FC → ReLU → FC)
            └── 도메인 간 / 클래스 간 분포 정렬

분류기 (Classifier) h_θ
    ↓
예측 출력
```

**섭동 샘플 생성:** Projected SVGD (Algorithm 1) 사용

$$X^{l+1}_i = \Pi_{B_\epsilon(X)}\left[X^l_i + \eta_l \hat{\phi}^*(X^l_i)\right]$$

```math
\hat{\phi}^*(X) = \frac{1}{n}\sum_{j=1}^{n}\left[k(X^l_j, X)\nabla_{X^l_j}\log\tilde{p}(X^l_j) + \nabla_{X^l_j}k(X^l_j, X)\right]
```

RBF 커널: 

```math
k(X, \tilde{X}) = \exp\left\{-\frac{\|X - \tilde{X}\|^2_2}{2\sigma^2}\right\}
```

---

### 2.5 성능 향상 결과

#### Domain Generalization (DG) - Single Source (CIFAR-C)

| 데이터셋 | ME-ADA (2위) | **GLOT-DR** | 향상 |
|---|---|---|---|
| CIFAR-10-C (평균) | 80.5% | **83.7%** | **+3.2%** |
| CIFAR-100-C (평균) | 52.3% | **55.7%** | **+3.4%** |

#### Domain Generalization (DG) - Multi-Source (PACS)

| Sketch 도메인 | Epi-FCR (2위) | **GLOT-DR** | 향상 |
|---|---|---|---|
| 정확도 | 65.0% | **65.4%** | +0.4% |
| 평균 | 72.6% | **73.5%** | **+0.9%** |

#### Domain Adaptation (DA) - Office-31

| 방법 | 평균 정확도 |
|---|---|
| ETD (OT 기반 2위) | 86.2% |
| **GLOT-DR** | **87.8%** |
| A→W 향상 | **+4.1%** |

#### SSL (CIFAR-10, n=4, 4000 labels)

| 방법 | 정확도 |
|---|---|
| VAT | 86.6% |
| LOT-DR | 88.1% |
| **GLOT-DR** | **89.2%** |

#### Adversarial Machine Learning (AML) - CIFAR-10

| 방법 | NAT | PGD200 | AA |
|---|---|---|---|
| TRADES | 81.64 | 53.11 | 49.77 |
| PGD-AT | 83.36 | 52.21 | 49.00 |
| **GLOT-DR** | **84.13** | **53.18** | **49.94** |

---

### 2.6 한계점

1. **계산 비용 증가:** GLOT-DR은 VAT/LOT-DR 대비 약 **25% 추가 실행 시간** 소요 (SVGD 샘플링 및 Kantorovich 네트워크 추가로 인한 오버헤드)

2. **하이퍼파라미터 민감성:** $\alpha$ (지역 가중치), $\beta$ (전역 가중치), $\lambda$ (엔트로픽 정규화), $n^S$, $n^T$ (섭동 샘플 수), SVGD 반복 횟수 $L$ 등 다수의 하이퍼파라미터 조정 필요

3. **이론적 일반화 바운드 부재:** 경험적 성능 향상은 입증하였으나, 명시적 일반화 오차 바운드에 대한 이론적 분석은 제한적

4. **대규모 데이터셋 적용 검증 미흡:** ImageNet 규모의 대규모 데이터셋에서의 확장성 검증이 부족

5. **연속 레이블(회귀) 적용:** 현재 분류 문제에 특화되어 있어 회귀 등 다른 태스크로의 직접 확장이 제한적

---

## 3. 모델의 일반화 성능 향상 가능성 (중점 분석)

### 3.1 일반화 성능 향상의 핵심 메커니즘

GLOT-DR이 일반화 성능을 향상시키는 원리는 다음 세 가지 축으로 분석된다:

#### (A) 전역 정규화를 통한 도메인 불변 표현 학습

전역 정규화 항 $r^g$는 최적 운송(OT) 거리를 통해 **소스-타겟 도메인 간 분포 불일치를 직접 최소화**한다. 특히 DA/DG 설정에서:

$$r^g = \mathcal{W}_d\left(\frac{1}{B^S}\sum_{i=1}^{B^S}\delta_{U^S_i},\ \frac{1}{B^T}\sum_{j=1}^{B^T}\delta_{U^T_j}\right)$$

이 항은 **특징 공간 $g_\phi(\cdot)$과 예측 공간 $h_\theta(\cdot)$ 모두에서 정렬**을 수행하여, 단순 특징 정렬(예: DANN)보다 더 세밀한 도메인 적응을 가능케 한다. 이는 DG 실험에서 가장 어려운 **Sketch 도메인**(색채 없는 스케치)에서 약 2.4% 향상이라는 주목할 만한 결과로 이어진다.

#### (B) 지역 정규화를 통한 결정 경계 평활화

지역 분포 $q^S_{ki}$는 앵커 샘플 $X^S_{ki}$ 주변 $\epsilon$-ball 내에서 고 위험(high-risk) 섭동 샘플을 생성한다:

```math
q^S_{ki} \propto \exp\left\{\lambda\left[\alpha s\left(X^S_{ki}, \tilde{X}^S_{kij}; \psi\right) + \ell\left(\tilde{X}^S_{kij}, Y^S_{ki}; \psi\right)\right]\right\}
```

이 공식은 **최대 불일치(maximum divergence) 영역을 집중적으로 탐색**하면서 해당 영역에서의 손실을 최소화함으로써, 결정 경계 근방의 평활성을 향상시키고 테스트 시 분포 이동(distribution shift)에 강건한 모델을 학습한다.

#### (C) 원본-섭동 분포의 결합(coupling)을 통한 표현력 향상

기존 DRO에서 $r(\tilde{Z})$는 오직 섭동 샘플만을 고려하지만, GLOT-DR의 $r(\tilde{Z}; \phi, \theta)$는 앵커 샘플과 섭동 샘플 **모두를 동시에 고려**한다. 이를 통해:
- 앵커와 섭동 샘플 간의 **예측 일관성(prediction consistency)** 강제
- TRADES 스타일의 KL 정규화를 자연스럽게 포함 가능
- 배치 수준의 전역 효과와 샘플 수준의 지역 효과를 **유기적으로 결합**

#### (D) Ablation Study를 통한 각 항의 기여도 확인

Office-31 DA 실험의 ablation 결과:

$$\underbrace{\text{VAT (input)}}_{\text{80.6\%}} < \underbrace{\text{VAT (latent)}}_{\text{83.0\%}} < \underbrace{\text{GOT-DR}}_{\text{84.3\%}} < \underbrace{\text{LOT-DR}}_{\text{85.4\%}} < \underbrace{\text{GLOT-DR}}_{\textbf{87.8\%}}$$

- 지역 정규화만(LOT-DR): +1.1% vs GOT-DR
- 전역 + 지역 정규화(GLOT-DR): **+2.4% vs LOT-DR** → 전역 정규화의 강력한 시너지 효과

#### (E) SVGD 기반 다중 섭동 샘플의 다양성 증가

SVGD는 단순 PGD와 달리 **입자들 간의 반발력(repulsion)**을 통해 다양한 섭동 방향을 탐색한다:

$$\hat{\phi}^*(X) = \frac{1}{n}\sum_{j=1}^{n}\left[\underbrace{k(X^l_j, X)\nabla_{X^l_j}\log\tilde{p}(X^l_j)}_{\text{고확률 영역 탐색}} + \underbrace{\nabla_{X^l_j}k(X^l_j, X)}_{\text{입자 다양성 유지}}\right]$$

이로 인해 섭동 샘플 수 $n$이 증가할수록 일반화 성능이 지속적으로 향상됨을 실험적으로 확인하였다 (DA: $n=2$일 때 GOT-DR 능가, SSL: $n=1 \to 4$로 증가 시 지속 향상).

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4.1 연구에 미치는 영향

#### (1) 분포 강건성 프레임워크의 통합화
GLOT-DR은 DG, DA, SSL, AML을 **단일 프레임워크로 통합**하는 선례를 제시했다. 향후 연구들이 이 통합 프레임워크를 기반으로 더 다양한 태스크(예: Few-Shot Learning, Continual Learning)로 확장하는 방향을 촉진할 것이다.

#### (2) 전역-지역 정규화 패러다임의 확산
지역 평활성과 전역 분포 정렬을 **동시에 추구하는 이중 정규화 패러다임**은 향후 정규화 기법 설계의 새로운 원칙으로 자리잡을 가능성이 있다.

#### (3) OT + SVGD 결합의 가능성 탐색
Optimal Transport와 Stein Variational Gradient Descent를 결합하는 방식은 **베이지안 딥러닝, 생성 모델, 메타 학습** 등에서 새로운 연구 방향을 열 수 있다.

#### (4) 적대적 학습에서의 분포 매칭 관점 도입
AML에서의 전역 정규화 항(Eq. 14)은 적대적 예제의 분포를 정상 예제의 분포로 이동시키는 **분포 매칭 기반 방어 전략**으로, AutoAttack 등 강력한 공격에 대한 새로운 방어 패러다임을 제시한다.

---

### 4.2 앞으로의 연구 시 고려할 점

#### (1) 계산 효율성 개선
GLOT-DR의 가장 큰 실용적 한계는 **SVGD 샘플링과 Kantorovich 네트워크**로 인한 추가 계산 비용이다. 향후 연구에서는:
- **경량화된 SVGD 대안** (예: 단일 스텝 근사, 결정론적 입자 최적화)
- **Sinkhorn 알고리즘의 GPU 병렬화**를 통한 OT 계산 가속화
- **지식 증류(Knowledge Distillation)**를 통한 경량 모델로의 전이를 고려해야 한다.

#### (2) 이론적 일반화 바운드 수립
현재 논문은 경험적 성능 향상에 집중되어 있으나, 향후 연구에서는:
$$\text{Generalization Gap} \leq f(\epsilon, n, \text{model complexity}, \mathcal{W}_\rho(\mathbb{P}, \tilde{\mathbb{P}}))$$
형태의 **PAC-Bayes 또는 Rademacher 복잡도 기반의 이론적 바운드** 수립이 필요하다.

#### (3) 대규모 데이터셋 및 Foundation Model과의 연계
ImageNet-1K, 또는 **CLIP, ViT 등 대형 사전학습 모델**에서 GLOT-DR의 적용 가능성을 검토해야 한다. 특히 사전학습 모델의 잠재 공간에서 전역-지역 정규화를 수행하는 **파인튜닝 프레임워크**로의 확장이 유망하다.

#### (4) 하이퍼파라미터 자동화
$\alpha$, $\beta$, $\lambda$, $n^S$, $n^T$의 **자동 조정(AutoML, NAS)** 메커니즘 개발이 실용적 적용을 위해 필수적이다. 적응적 가중치 스케줄링 전략이 필요하다.

#### (5) 레이블 효율성 및 Few-Shot 설정으로의 확장
현재 DA/SSL 실험은 충분한 레이블 데이터를 가정하지만, **극소 레이블(1-shot, 5-shot)** 설정에서의 GLOT-DR 성능 및 전역 정규화 항의 적절성 검토가 필요하다.

#### (6) 공정성(Fairness) 및 인과적 표현 학습
전역 정규화를 통한 분포 정렬 메커니즘을 **알고리즘 공정성(Group Fairness)**이나 **인과적 도메인 일반화(Causal DG)**에 적용하는 연구 방향이 유망하다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

> **⚠️ 주의:** 아래 비교 분석은 논문 내 인용 문헌과 제가 학습 데이터 기반으로 알고 있는 공개 연구 정보를 종합한 것입니다. 2023년 이후 최신 연구의 경우 정보가 불완전할 수 있으므로, 구체적 수치는 원문 논문을 반드시 직접 확인하시기 바랍니다.

### 5.1 분류별 비교

| 연구 | 발표 | 접근법 | GLOT-DR 대비 차이점 |
|---|---|---|---|
| **ME-ADA** (Zhao et al., NeurIPS 2020) | 2020 | 최대 엔트로피 적대적 데이터 증강 | 전역 정규화 없음; CIFAR-10-C에서 GLOT-DR 대비 -3.2% |
| **ADT** (Dong et al., NeurIPS 2020) | 2020 | 정규 분포 가정 기반 적대적 분포 학습 | 분포 가정이 강하며 유연성 부족; PGD200에서 -7% |
| **Wang et al.** (2021) | 2021 | DA에서 분포 강건 학습 | DRO를 DA에 적용하나 지역-전역 결합 없음 |
| **Nguyen-Duc et al., AISTATS 2022** | 2022 | 입자 기반 적대적 지역 분포 정규화 | 지역 정규화만 수행; SSL에서 GLOT-DR 대비 -1% 이상 |
| **Bui et al.** (arXiv 2022) | 2022 | 통합 Wasserstein DR 적대적 학습 | AML에 특화; DA/DG/SSL 통합 프레임워크 부재 |
| **TRADES** (Zhang et al., ICML 2019) | 2019 | 정확도-강건성 트레이드오프 | 지역 KL 정규화만; 전역 효과 없음 |
| **DeepJDOT** (Damodaran et al., ECCV 2018) | 2018 | OT 기반 DA | 전역 정규화만; 지역 정규화 없음 |

### 5.2 핵심 비교 관점

```
                    지역 정규화    전역 정규화    다중 태스크    폐쇄형 해
PGD-AT              ✓             ✗             ✗             ✓
TRADES              ✓             ✗             ✗             ✓
ME-ADA              ✓             ✗             ✗ (DG only)   ✗
ADT                 ✗             ✓ (제한적)    ✗             ✗
DeepJDOT            ✗             ✓             ✗             ✗
Nguyen-Duc et al.   ✓             ✗             ✗             부분
GLOT-DR             ✓             ✓             ✓             ✓
```

### 5.3 GLOT-DR의 차별성 정리

GLOT-DR이 기존 2020년 이후 연구들과 차별화되는 핵심 포인트:

1. **유일한 전역-지역 이중 정규화 통합**: 동시에 두 효과를 결합한 프레임워크
2. **폐쇄형 해의 존재**: 쌍대형 최소화 없이 직접 최적 분포 도출
3. **범용성**: 단일 프레임워크가 DG/DA/SSL/AML 모두 포괄
4. **SVGD 기반 다양성 있는 섭동 샘플링**: 기존 PGD 기반 단일 최악 샘플 생성 방식보다 풍부한 분포 탐색

---

## 종합 결론

GLOT-DR은 분포 강건성 최적화 연구에서 **이론적 엄밀성과 실용적 범용성을 동시에 달성**한 중요한 기여를 한 논문이다. 전역-지역 이중 정규화라는 설계 원칙은 향후 일반화 성능 향상 연구의 핵심 참조점이 될 것이며, OT + SVGD의 결합은 분포 강건 머신러닝의 새로운 계산 패러다임을 제시한다. 다만 계산 비용, 이론적 바운드, 대규모 모델 적용 가능성의 측면에서 추가 연구가 필요하다.
