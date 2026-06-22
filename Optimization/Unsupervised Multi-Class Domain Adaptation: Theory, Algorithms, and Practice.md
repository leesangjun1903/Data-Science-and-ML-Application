# Unsupervised Multi-Class Domain Adaptation: Theory, Algorithms, and Practice

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

기존 비지도 도메인 적응(UDA) 이론은 **이진 분류(binary classification)** 설정에서 유도되었으나, 실제 알고리즘은 **다중 클래스(multi-class)** 환경에서 작동한다는 이론-알고리즘 간 괴리가 존재한다. 이 논문은 이 괴리를 메우기 위해 다음을 제안한다:

1. **Multi-Class Scoring Disagreement (MCSD)** 발산 개념을 새롭게 제안
2. MCSD 기반 **도메인 적응 상한(bound)** 유도
3. 이론에서 자연스럽게 도출되는 **알고리즘 프레임워크** 설계

### 주요 기여

| 기여 | 내용 |
|------|------|
| 이론적 기여 | MCSD 발산 정의, Rademacher complexity 기반 PAC 바운드 |
| 알고리즘 기여 | McDalNets 프레임워크, SymmNets-V2 알고리즘 |
| 실용적 기여 | Closed/Partial/Open Set UDA에 대한 통합 프레임워크 |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 향상 및 한계

### 2.1 해결하고자 하는 문제

**기존 이론의 한계:**

- Ben-David et al. [10]의 $\mathcal{H}\Delta\mathcal{H}$-발산: 이진 분류 전용, 레이블 불일치만 측정
- Zhang et al. [18]의 MDD (Margin Disparity Discrepancy): 다중 클래스를 scalar-valued 상대 마진 함수로 측정 → **모든 클래스 간 관계를 완전히 포착하지 못함**

MDD는 다음과 같이 정의된다:

$$d^{(\rho)}_{MD}(P_x, Q_x) := \sup_{f' \in \mathcal{F}} \left[\mathbb{E}_{Q_x} \Phi_\rho(\rho_{f'}(\cdot, h_f)) - \mathbb{E}_{P_x} \Phi_\rho(\rho_{f'}(\cdot, h_f))\right] \tag{4}$$

여기서 $\rho_f(x, y) = \frac{1}{2}(f_y(x) - \max_{y' \neq y} f_{y'}(x))$ 는 **상대 마진(relative margin)** 함수이며, $\Phi_\rho$는 ramp loss:

$$\Phi_\rho(x) := \begin{cases} 0, & \rho \leq x \\ 1 - x/\rho, & 0 < x < \rho \\ 1, & x \leq 0 \end{cases} \tag{5}$$

**문제점:** MDD의 상대 마진은 $f$의 최대값 성분(즉, $h_f$)에만 의존하므로, 다중 클래스 scoring function $f$와 $f'$ 간의 불일치를 **완전히 특성화하지 못한다.**

---

### 2.2 제안 방법 (수식 포함)

#### 2.2.1 절대 마진 함수 (Absolute Margin Function)

$$\mu_k(f(x), y) = \begin{cases} +f_k(x), & k = y \\ -f_k(x), & k \in \mathcal{Y} \setminus \{y\} \end{cases} \tag{6}$$

sum-to-zero 제약 $\sum_{k=1}^K f_k(x) = 0$ 하에서 정의되며, $\mu_k \geq 0$ 이면 올바른 분류를 의미한다.

#### 2.2.2 Multi-Class Scoring Disagreement (MCSD)

$$\text{MCSD}^{(\rho)}_D(f', f'') := \frac{1}{K} \mathbb{E}_{x \sim D} \| M^{(\rho)}(f'(x)) - M^{(\rho)}(f''(x)) \|_1 \tag{7}$$

여기서 $M^{(\rho)}(f(x)) \in [0,1]^{K \times K}$ 는 절대 마진 위반의 행렬:

$$M^{(\rho)}_{i,j}(f(x)) = \Phi_\rho(\mu_i(f(x), j)) \tag{8}$$

MCSD는 행렬 형태로 **모든 클래스 쌍의 element-wise 불일치를 집계**한다.

#### 2.2.3 MCSD 발산 (MCSD Divergence)

$$d^{(\rho)}_{MCSD}(P_x, Q_x) := \sup_{f', f'' \in \mathcal{F}} \left[\text{MCSD}^{(\rho)}_{Q_x}(f', f'') - \text{MCSD}^{(\rho)}_{P_x}(f', f'')\right] \tag{9}$$

#### 2.2.4 도메인 적응 상한 (Theorem 1)

$$\mathcal{E}_Q(h_f) \leq \mathcal{E}^{(\rho)}_P(f) + d^{(\rho)}_{MCSD}(P_x, Q_x) + \lambda \tag{10}$$

여기서:
- $\mathcal{E}\_Q(h_f) := \mathbb{E}_{(x,y) \sim Q} \mathbb{I}[h_f(x) \neq y]$: 타겟 도메인 기대 오류
- $\mathcal{E}^{(\rho)}\_P(f) := \mathbb{E}\_{(x,y) \sim P} \sum_{k=1}^K \Phi_\rho(\mu_k(f(x), y))$: 소스 도메인 마진 손실
- $\lambda = \mathcal{E}^{(\rho)}_P(f^\*) + \mathcal{E}^{(\rho)}_Q(f^\*)$: 최적 함수 $f^*$에 의한 결합 오류

#### 2.2.5 PAC 바운드 (Theorem 2)

Rademacher 복잡도 기반으로 유한 표본에서의 상한:

$$\mathcal{E}_Q(h_f) \leq \mathcal{E}^{(\rho)}_{\hat{P}}(f) + d^{(\rho)}_{MCSD}(\hat{P}_x, \hat{Q}_x) + \left(\frac{2K^2}{\rho} + \frac{4K}{\rho}\right)\hat{\mathfrak{R}}_S(\Pi_1(\mathcal{F})) + \frac{4K}{\rho}\hat{\mathfrak{R}}_T(\Pi_1(\mathcal{F})) + 6K\sqrt{\frac{\log\frac{4}{\delta}}{2n_s}} + 3K\sqrt{\frac{\log\frac{4}{\delta}}{2n_t}} + \lambda \tag{20}$$

여기서 $\hat{\mathfrak{R}}_S(\Pi_1(\mathcal{F}))$는 경험적 Rademacher 복잡도이다.

---

### 2.3 알고리즘 프레임워크

#### 2.3.1 McDalNets (Multi-class Domain-adversarial learning Networks)

이론적 상한 (10)을 최소화하기 위한 minimax 최적화:

$$\min_{f, \psi} \mathcal{E}^{(\rho)}_{\hat{P}^\psi}(f) + \left[\text{MCSD}^{(\rho)}_{\hat{Q}^\psi_x}(f', f'') - \text{MCSD}^{(\rho)}_{\hat{P}^\psi_x}(f', f'')\right]$$
$$\max_{f', f''} \left[\text{MCSD}^{(\rho)}_{\hat{Q}^\psi_x}(f', f'') - \text{MCSD}^{(\rho)}_{\hat{P}^\psi_x}(f', f'')\right] \tag{22}$$

대리 목적함수(surrogate objectives):

**L1/MCD [16]:**

$$\text{SurMCSD}_{D^\psi}(f', f'') = \mathbb{E}_{x \sim D} \frac{1}{K} \|\phi(f'(\psi(x))) - \phi(f''(\psi(x)))\|_1 \tag{24}$$

**KL:**
$$\mathbb{E}_{x \sim D} \frac{1}{2}\left[\text{KL}(\phi(f'(\psi(x))), \phi(f''(\psi(x)))) + \text{KL}(\phi(f''(\psi(x))), \phi(f'(\psi(x))))\right] \tag{25}$$

**CE:**
$$\mathbb{E}_{x \sim D} \frac{1}{2}\left[\text{CE}(\phi(f'(\psi(x))), \phi(f''(\psi(x)))) + \text{CE}(\phi(f''(\psi(x))), \phi(f'(\psi(x))))\right] \tag{26}$$

#### 2.3.2 SymmNets (Domain-Symmetric Networks)

$f^s$, $f^t$, $f^{st} = [f^s; f^t]$로 구성되는 대칭 네트워크:

**소스/타겟 태스크 분류기 학습:**

$$\min_{f^s} \mathcal{L}^s_{\hat{P}^\psi}(f^s) = -\frac{1}{n_s} \sum_{i=1}^{n_s} \omega_{y^s_i} \log(p^s_{y^s_i}(x^s_i)) \tag{33}$$

$$\min_{f^t} \mathcal{L}^t_{\hat{P}^\psi}(f^t) = -\frac{1}{n_s} \sum_{i=1}^{n_s} \omega_{y^s_i} \log(p^t_{y^s_i}(x^s_i)) \tag{34}$$

**도메인 혼동(Confusion) 목적함수 (소스):**

$$\min_\psi \text{ConFUSE}^{st}_{\hat{P}^\psi}(f^s, f^t) = -\frac{1}{2n_s}\sum_{i=1}^{n_s} \omega_{y^s_i} \log(p^{st}_{y^s_i}(x^s_i)) - \frac{1}{2n_s}\sum_{i=1}^{n_s} \omega_{y^s_i} \log(p^{st}_{y^s_i+K}(x^s_i)) \tag{35}$$

**도메인 혼동 목적함수 (타겟):**

$$\min_\psi \text{ConFUSE}^{st}_{\hat{Q}^\psi_x}(f^s, f^t) = -\frac{1}{2n_t}\sum_{j=1}^{n_t}\sum_{k=1}^K p^{st}_{k+K}(x^t_j)\log(p^{st}_k(x^t_j)) - \frac{1}{2n_t}\sum_{j=1}^{n_t}\sum_{k=1}^K p^{st}_k(x^t_j)\log(p^{st}_{k+K}(x^t_j)) \tag{36}$$

**도메인 판별(Discrimination) 목적함수:**

$$\min_{f^s, f^t} \text{DisCRIM}^{st}_{\hat{P}^\psi, \hat{Q}^\psi_x}(f^s, f^t) = -\frac{1}{n_s}\sum_{i=1}^{n_s} \omega_{y^s_i} \log(p^{st}_{y^s_i}(x^s_i)) - \frac{1}{n_t}\sum_{j=1}^{n_t} \log\left(\sum_{k=1}^K p^{st}_{k+K}(x^t_j)\right) \tag{37}$$

**전체 학습 목적함수:**

$$\min_\psi \text{ConFUSE}^{st}_{\hat{P}^\psi}(f^s, f^t) + \lambda \text{ConFUSE}^{st}_{\hat{Q}^\psi_x}(f^s, f^t)$$
$$\min_{f^s, f^t} \mathcal{L}^s_{\hat{P}^\psi}(f^s) + \mathcal{L}^t_{\hat{P}^\psi}(f^t) + \text{DisCRIM}^{st}_{\hat{P}^\psi, \hat{Q}^\psi_x}(f^s, f^t) \tag{38}$$

---

### 2.4 모델 구조

```
[소스 도메인 P]  [타겟 도메인 Qx]
       ↓                ↓
   Feature Extractor ψ (ResNet 등)
       ↓
   ┌───────────────────────────────┐
   │  f^s (소스 분류기, K neurons) │
   │  f^t (타겟 분류기, K neurons) │
   │  f^st = [f^s; f^t] (2K neurons) │
   └───────────────────────────────┘
       ↓
  ConFUSE (혼동 손실) + DisCRIM (판별 손실)
```

- **ψ**: 학습 가능한 특징 추출기 (ResNet-50, ResNet-101, AlexNet, LeNet 등)
- **$f^s$, $f^t$**: 각각 소스와 타겟을 담당하는 분류기 (파라미터 공유)
- **$f^{st}$**: $f^s$와 $f^t$를 연결한 2K 출력의 연결 분류기

---

### 2.5 성능 향상

| 데이터셋 | Source Only | DANN | MDD | MCD (L1) | SymmNets-V2 |
|---------|------------|------|-----|----------|-------------|
| Office-31 | 81.8% | 82.8% | 84.5% | 84.7% | **89.1%** |
| ImageCLEF | 82.7% | 84.2% | 86.7% | 87.0% | **89.7%** |
| Office-Home | 58.9% | 60.0% | 61.1% | 62.0% | **68.1%** |
| Digits | 70.5% | 72.5% | N/C | 90.6% | **96.0%** |
| VisDA-2017 | 41.8% | 58.4% | N/C | 70.4% | **71.3%** |

*N/C: 수렴 실패*

---

### 2.6 한계점

1. **Ramp loss의 vanishing gradient 문제**: MCSD를 직접 최적화할 수 없어 대리 함수(surrogate)를 사용해야 함
2. **비대칭 발산**: MCSD 발산은 $P_x$와 $Q_x$에 대해 대칭이 아님
3. **$\lambda$ 추정 불가**: 이상적 함수 $f^*$에 의존하는 상수 $\lambda$는 직접 계산이 어려움
4. **폐쇄 집합 가정 이론**: 이론적 분석의 대부분이 closed set UDA에 한정
5. **Source-free UDA 미지원**: 소스 데이터가 반드시 필요한 구조

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 이론적 일반화 보장

PAC 바운드 (Theorem 2)에 따르면 타겟 오류는:

$$\mathcal{E}_Q(h_f) \leq \underbrace{\mathcal{E}^{(\rho)}_{\hat{P}}(f)}_{\text{소스 경험 오류}} + \underbrace{d^{(\rho)}_{MCSD}(\hat{P}_x, \hat{Q}_x)}_{\text{도메인 거리}} + \underbrace{\mathcal{O}\left(\frac{K^2}{\rho}\hat{\mathfrak{R}}(\Pi_1(\mathcal{F}))\right)}_{\text{복잡도 항}} + \underbrace{\lambda}_{\text{이상적 결합 오류}}$$

**일반화 향상을 위한 세 가지 조건:**

1. **소스 경험 오류 최소화**: $\mathcal{E}^{(\rho)}_{\hat{P}}(f)$ → 충분한 소스 레이블 데이터와 정규화
2. **도메인 거리 최소화**: $d^{(\rho)}_{MCSD}(\hat{P}_x, \hat{Q}_x)$ → 조건부 특징 분포 정렬
3. **가설 공간 복잡도 제어**: $\hat{\mathfrak{R}}(\Pi_1(\mathcal{F}))$ → 정규화, 드롭아웃 등

### 3.2 MCSD의 세밀한 특성화가 일반화에 미치는 영향

MCSD가 기존 발산 대비 우월한 이유:

| 발산 | 특성화 수준 | 불일치 측정 |
|------|------------|------------|
| $\mathcal{H}\Delta\mathcal{H}$ [10] | 레이블 불일치만 | 0-1 손실 |
| MDD [18] | Scalar 상대 마진 | 최대 성분만 고려 |
| **MCSD (본 논문)** | **행렬 형태 절대 마진** | **모든 K개 클래스 쌍 고려** |

MCSD는 Proposition 3에 의해 K개 entry 쌍의 $\varphi$-거리 합으로 분해 가능:

$$\|M^{(\rho)}(f'(x)) - M^{(\rho)}(f''(x))\|_1 = \sum_{k=1}^K \varphi(f'_k(x), f''_k(x))$$

이는 **조건부 특징 분포(conditional feature distributions)**를 더 세밀하게 정렬하므로, 일반화 성능 향상에 직접적으로 기여한다.

### 3.3 Cross-Domain Training의 일반화 기여

Proposition 4에 의해 $\mathcal{L}^t_{\hat{P}^\psi}(f^t)$를 최소화하는 $f^{t*}$는 동시에 $\mathcal{E}^{(\rho)}_{\hat{P}^\psi}(f^t)$도 최소화한다. 이는 타겟 분류기가 소스 데이터로부터 **뉴런 단위 대응(neuron-wise correspondence)**을 학습하게 하여, 도메인 혼동 목적함수가 더 효과적으로 동작하게 한다.

### 3.4 Partial/Open Set으로의 확장 가능성

- **Partial UDA**: 소스 클래스 가중치 $\omega_k = \frac{1}{n_t}\sum_{j=1}^{n_t} p^t_k(x^t_j)$를 통해 불필요한 소스 클래스 억제
- **Open Set UDA**: 추가 뉴런으로 미지 클래스 처리 ($\nu = 6$의 슈퍼클래스 샘플링)

이러한 확장은 **실세계 분포 변화에 더 강인한 일반화**를 가능하게 한다.

---

## 4. 앞으로의 연구에 미치는 영향과 연구 시 고려할 점

### 4.1 미래 연구에 미치는 영향

1. **이론-알고리즘 연결 강화**: MCSD 기반 바운드는 향후 UDA 알고리즘 설계의 이론적 토대를 제공한다. 특히 multi-class 환경에서 알고리즘의 효과를 이론적으로 정당화하는 틀을 확립했다.

2. **일반화된 발산 개념**: MCSD 발산은 다양한 도메인 정렬 방법론 (예: OT 기반, 프로토타입 기반 등)에 대한 이론적 분석 도구로 활용될 수 있다.

3. **유니파이드 프레임워크**: McDalNets 프레임워크는 DANN, MCD, MDD 등 기존 방법을 통합 설명하며, 새로운 surrogate 함수 개발의 방향을 제시한다.

4. **다양한 UDA 설정 통합**: Closed/Partial/Open Set UDA를 단일 프레임워크로 처리하는 방식은 이후 **Universal Domain Adaptation** 연구의 선행 연구로 기능한다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

아래는 본 논문과 관련된 2020년 이후의 주요 연구 흐름을 정리한 것이다. 단, 이하의 연구들은 해당 논문 내에 직접 인용되지 않은 것들이며, 필자의 지식에 기반하므로 개별 세부 수치는 원 논문을 통해 확인을 권장한다.

### 5.1 Source-Free Domain Adaptation

본 논문은 소스 데이터에 접근이 필요하다는 한계가 있으나, 이후 연구들은 소스 없이 적응하는 방향으로 발전했다:

- **SHOT (Liang et al., ICML 2020)**: 소스 모델만을 사용하는 Source-Free UDA. 정보 최대화와 자기-지도 학습을 결합하여 소스 데이터 없이 타겟 도메인에 적응.
- **NRC (Yang et al., NeurIPS 2021)**: 이웃 관계 클러스터링을 통해 Source-Free UDA 수행.

**비교 관점**: MCSD 이론은 소스-타겟 분포 거리를 명시적으로 측정하지만, Source-Free 설정에서는 $d^{(\rho)}_{MCSD}(\hat{P}_x, \hat{Q}_x)$ 계산이 불가능하다. 이론 확장이 필요한 영역이다.

### 5.2 Vision-Language 모델 기반 도메인 적응

- **CLIP 기반 UDA (2022~)**: CLIP (Radford et al., 2021)과 같은 대규모 사전학습 모델을 활용한 도메인 적응 연구들이 등장. 이 경우 도메인 발산 자체가 크게 줄어들어 이론적 바운드의 $\lambda$ 항이 작아질 가능성이 있다.

### 5.3 Test-Time Adaptation (TTA)

- **Tent (Wang et al., ICLR 2021)**: 테스트 시간에 엔트로피를 최소화하여 적응. SymmNets의 도메인 혼동 목적함수와 유사한 엔트로피 기반 접근이지만, 레이블된 소스 없이 온라인으로 적응.

### 5.4 Optimal Transport 기반 이론

- **DeepJDOT (Damodaran et al., ECCV 2018)**: OT를 활용한 깊은 도메인 적응. 본 논문의 MCSD 발산과 OT 발산을 통합하는 이론적 프레임워크는 향후 연구 과제이다.

### 5.5 비교 요약표

| 방법 | 이론 근거 | 다중 클래스 | Source-Free | 설정 다양성 |
|------|----------|------------|-------------|------------|
| DANN [13] | $\mathcal{H}\Delta\mathcal{H}$-발산 | 부분적 | ✗ | Closed |
| MDD [18] | Scalar 마진 발산 | 부분적 | ✗ | Closed |
| **본 논문 (MCSD)** | **행렬 MCSD 발산** | **완전** | ✗ | Closed/Partial/Open |
| SHOT (2020) | 정보 최대화 | 완전 | ✓ | Closed/Partial |
| NRC (2021) | 이웃 클러스터링 | 완전 | ✓ | Closed |

---

### 연구 시 고려할 점

1. **Source-Free 환경에서의 이론 확장**: MCSD 발산을 소스 모델(분포)만으로 추정하는 방법 개발이 필요하다.

2. **대규모 사전학습 모델과의 통합**: ViT, CLIP 등의 backbone에서 MCSD 기반 정렬 전략의 효과 및 Rademacher 복잡도 추정 방법을 재검토해야 한다.

3. **$\lambda$ 항의 실질적 추정**: 결합 오류 $\lambda$를 실제 학습에서 추정하거나 상한을 제공하는 방법이 연구되어야 한다. 이 항이 크면 도메인 정렬 효과가 제한될 수 있다.

4. **다중 소스 도메인 확장**: 현재 이론은 단일 소스 도메인을 가정하지만, 실제 응용에서는 다중 소스가 일반적이다.

5. **Surrogate 선택의 이론적 정당화**: 어떤 surrogate 함수가 MCSD를 가장 잘 근사하는지에 대한 이론적 보장이 추가로 필요하다.

6. **연속적 도메인 변화(Gradual Domain Shift)**: 정적 두 도메인 가정에서 벗어나 시간에 따라 변화하는 도메인에 MCSD를 적용하는 연구가 필요하다.

---

## 참고 자료

**주요 참고 논문 (본 논문 내 인용)**:
- Zhang, Y., Deng, B., Tang, H., Zhang, L., & Jia, K. (2020). *Unsupervised Multi-Class Domain Adaptation: Theory, Algorithms, and Practice*. arXiv:2002.08681v2
- Ben-David, S. et al. (2010). *A theory of learning from different domains*. Machine Learning, 79(1-2), 151-175.
- Mansour, Y., Mohri, M., & Rostamizadeh, A. (2009). *Domain adaptation: Learning bounds and algorithms*. COLT 2009.
- Zhang, Y., Liu, T., Long, M., & Jordan, M. (2019). *Bridging theory and algorithm for domain adaptation*. ICML 2019.
- Saito, K. et al. (2018). *Maximum classifier discrepancy for unsupervised domain adaptation*. CVPR 2018.
- Ganin, Y. et al. (2017). *Domain-adversarial training of neural networks*. JMLR, 17(1).
- Dogan, U., Glasmachers, T., & Igel, C. (2016). *A unified view on multi-class support vector classification*. JMLR, 17(45).
- Mohri, M., Rostamizadeh, A., & Talwalkar, A. (2012). *Foundations of Machine Learning*. MIT Press.

**2020년 이후 관련 연구 (비교 분석 참고)**:
- Liang, J. et al. (2020). *Do we really need to access the source data? Source hypothesis transfer for unsupervised domain adaptation*. ICML 2020.
- Wang, D. et al. (2021). *Tent: Fully test-time adaptation by entropy minimization*. ICLR 2021.
- Yang, S. et al. (2021). *Exploiting the intrinsic neighborhood structure for source-free domain adaptation*. NeurIPS 2021.
