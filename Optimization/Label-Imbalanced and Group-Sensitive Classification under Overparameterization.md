# Label-Imbalanced and Group-Sensitive Classification under Overparameterization

**[Kini et al., NeurIPS 2021]**

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

이 논문의 핵심 주장은 다음과 같습니다:

> **과매개변수화(overparameterization)된 딥러닝 모델을 훈련 손실이 0이 될 때까지(Terminal Phase of Training, TPT) 학습할 때, 클래스 불균형 문제를 해결하기 위해서는 로짓(logit)에 대한 가산적(additive) 조정이 아닌 승산적(multiplicative) 조정이 필수적이다.**

또한 두 종류의 조정은 훈련의 서로 다른 단계에서 상이한 역할을 수행하므로, 이를 통합한 새로운 손실 함수인 **Vector-Scaling (VS) Loss**를 제안합니다.

### 주요 기여

| 기여 | 내용 |
|------|------|
| 이론적 분석 | 승산적 vs 가산적 조정의 역할 구분 (Theorem 1) |
| 새로운 손실 함수 | VS-loss 제안 (기존 LA-loss, CDT-loss를 특수 케이스로 포괄) |
| 그룹 민감 확장 | Group-VS-loss 도입 (레이블/그룹 불균형 통합 처리) |
| 일반화 분석 | 가우시안 혼합 모델에서 공정성 트레이드오프 이론 제시 (Theorem 2) |

---

## 2. 문제, 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

**배경:** 딥러닝 모델은 훈련 오류가 0이 된 이후에도 계속 학습이 진행되는 TPT(Terminal Phase of Training) 상태에 도달합니다. 이 상태에서:

- **Weighted Cross-Entropy (wCE)**: 클래스 불균형 문제 해결에 실패
- **LA-loss (가산적 조정)**: TPT에서 결정 경계를 적절히 조정하지 못함
- **CDT-loss (승산적 조정)**: TPT에서는 효과적이나, 훈련 초기에는 소수 클래스에 해로움

**핵심 질문:**
1. 가산적/승산적 하이퍼파라미터가 각각 어떤 마진 조정 메커니즘을 갖는가?
2. 훈련의 각 단계에서 어떤 조정이 효과적인가?

### 2.2 제안하는 방법 (수식 포함)

#### 이진 VS-loss

$$\ell_{\text{VS}}(y, f_{\mathbf{w}}(\mathbf{x})) = \omega_y \cdot \log\left(1 + e^{\iota_y} \cdot e^{-\Delta_y y f_{\mathbf{w}}(\mathbf{x})}\right) \tag{1}$$

#### 다중 클래스 VS-loss

$$\ell_{\text{VS}}(y, \mathbf{f}_{\mathbf{w}}(\mathbf{x})) = -\omega_y \log\left(\frac{e^{\Delta_y f_y(\mathbf{x}) + \iota_y}}{\sum_{c \in [C]} e^{\Delta_c f_c(\mathbf{x}) + \iota_c}}\right) \tag{2}$$

여기서:
- $\omega_y > 0$: 클래스별 가중치 파라미터
- $\iota_y \in \mathbb{R}$: 가산적(additive) 로짓 파라미터
- $\Delta_y > 0$: 승산적(multiplicative) 로짓 파라미터

#### 기존 기법과의 관계

$$\begin{cases}
\Delta_y = 1,\; \iota_y = 0,\; \omega_y = \pi_y^{-1} & \Rightarrow \text{wCE (Weighted Cross-Entropy)} \\
\Delta_y = 1 & \Rightarrow \text{LA-loss} \\
\iota_y = 0 & \Rightarrow \text{CDT-loss}
\end{cases}$$

#### 그룹 민감 VS-loss

$$\ell_{\text{Group-VS}}(y, g, f_{\mathbf{w}}(\mathbf{x})) = \omega_{y,g} \cdot \log\left(1 + e^{\iota_{y,g}} \cdot e^{-\Delta_{y,g} y f_{\mathbf{w}}(\mathbf{x})}\right) \tag{3}$$

#### Cost-Sensitive SVM (CS-SVM)

$$\min_{\mathbf{w}} \|\mathbf{w}\|_2 \quad \text{s.t.} \quad \begin{cases} \langle \mathbf{w}, h(\mathbf{x}_i) \rangle \geq \delta, & y_i = +1 \\ \langle \mathbf{w}, h(\mathbf{x}_i) \rangle \leq -1, & y_i = -1 \end{cases}, \quad i \in [n] \tag{4}$$

$\delta > 1$이면 소수 클래스에 더 큰 마진을 부여합니다.

### 2.3 핵심 이론 (Theorem 1: VS-loss = CS-SVM)

**Theorem 1.** 이진 훈련 데이터 $\{\mathbf{x}\_i, y_i\}\_{i=1}^n$이 선형 분리 가능하다고 가정하자. 즉, $\exists \mathbf{w}: y\_i \mathbf{w}^T h(\mathbf{x}\_i) \geq 1, \forall i \in [n]$. 노름 제약 최적화 해 $\mathbf{w}\_R = \arg\min_{\|\mathbf{w}\|_2 \leq R} L_n(\mathbf{w})$에 대해:

$$\lim_{R \to \infty} \frac{\mathbf{w}_R}{\|\mathbf{w}_R\|_2} = \frac{\hat{\mathbf{w}}_\delta}{\|\hat{\mathbf{w}}_\delta\|_2}, \quad \delta = \frac{\Delta_-}{\Delta_+}$$

**핵심 함의:**
- TPT에서 $\omega_\pm$와 $\iota_\pm$는 **효과 없음** → 모두 동일한 SVM 해로 수렴
- $\Delta_\pm$만이 결정 경계를 조정함: $\Delta_- > \Delta_+ \Leftrightarrow \delta > 1$ → 소수 클래스에 유리

### 2.4 Observation 1: 훈련 초기의 역설

$f_{\mathbf{w}}(\mathbf{x}) = 0$으로 초기화 시 VS-loss의 그래디언트:

$$\nabla_{\mathbf{w}} \ell_{\text{VS}}(y, f_{\mathbf{w}}(\mathbf{x})) = -\omega_y \Delta_y \sigma(-\Delta_y y f_{\mathbf{w}}(\mathbf{x}) + \iota_y) \cdot y h(\mathbf{x})$$

초기($f_{\mathbf{w}}(\mathbf{x}) = 0$)에서 $\Delta_y$는 $\omega_y$와 동일한 역할을 합니다. CDT-loss에서 $\Delta_+ < \Delta_-$가 필요하므로 초기에는 소수 클래스의 그래디언트가 더 작아져 **훈련이 잘못된 방향으로 진행**됩니다. 이를 가산적 조정 $\iota_+ > \iota_-$로 보완할 수 있습니다.

### 2.5 모델 구조

```
입력 x
    ↓
특징 추출기 h(·): x → R^p  (고정된 특징 또는 딥넷 인코더)
    ↓
선형 분류기 f_w(x) = ⟨w, h(x)⟩  (로짓 벡터)
    ↓
VS-loss 적용: Δ_y (승산적) + ι_y (가산적) + ω_y (가중치)
    ↓
Softmax → 예측 클래스
```

실험에서는 **ResNet50** (ImageNet pretrain) 및 **ResNet32** (CIFAR)를 백본으로 사용합니다.

### 2.6 성능 향상

**CIFAR-10/100 레이블 불균형 실험 (Top-1 Accuracy %):**

| 방법 | CIFAR-10 LT-100 | CIFAR-10 STEP-100 | CIFAR-100 LT-100 | CIFAR-100 STEP-100 |
|------|:-:|:-:|:-:|:-:|
| CE | 71.94 | 62.69 | 38.82 | 39.49 |
| wCE | 72.6 | 67.3 | 40.5 | 40.1 |
| LDAM-DRW | 77.03 | 76.92 | 42.04 | 45.36 |
| LA-loss | 80.81 | 78.23 | 42.87 | 45.69 |
| CDT-loss | 79.55 | 73.26 | 42.57 | 44.12 |
| **VS-loss** | **80.82** | **79.10** | **43.52** | **46.53** |

**Waterbirds 그룹 불균형 실험:**

| 방법 | Symm. DEO | Bal. acc. | Worst acc. |
|------|:-:|:-:|:-:|
| CE | 25.3 | 84.9 | 68.1 |
| Group VS | **18.1** | **88.1** | **76.7** |
| CE + DRO | 16.3 | 88.7 | 75.2 |
| Group VS + DRO | **11.8** | **90.2** | **78.9** |

### 2.7 한계

1. **고정 특징 가정**: 이론 분석은 고정된 특징 표현을 가정하며, 딥넷에서 특징이 동시에 학습되는 실제 환경과 괴리 존재
2. **하이퍼파라미터 튜닝 복잡성**: $(\tau, \gamma)$ 조합 탐색을 위해 validation set 필요 (Bi-level optimization은 미래 과제로 남김)
3. **이론의 이진 분류 중심**: 다중 클래스 확장은 보충 자료에 한정
4. **선형 분리 가능성 가정**: Theorem 1은 데이터가 선형 분리 가능할 때만 성립
5. **그룹 레이블 필요**: 훈련 시 그룹 소속이 알려져야 하며, 테스트 시에는 불필요하지만 실제 데이터에서 그룹 레이블 획득이 어려울 수 있음

---

## 3. 일반화 성능 향상 가능성

### 3.1 Theorem 2: CS-SVM의 균형 오류 분석

**가우시안 혼합 모델(GMM)** 하에서의 이론적 일반화 분석:

$$\overline{R}_+ := Q\left(\mathbf{e}_1^T \mathbf{VS}\boldsymbol{\rho}_\delta + b_\delta / q_\delta\right), \quad \overline{R}_- := Q\left(-\mathbf{e}_2^T \mathbf{VS}\boldsymbol{\rho}_\delta - b_\delta / q_\delta\right)$$

$$\mathcal{R}_{\text{bal}} \xrightarrow{P} \overline{\mathcal{R}}_{\text{bal}} := \frac{\overline{R}_+ + \overline{R}_-}{2}$$

여기서 핵심 함수 $\eta_\delta$는:

$$\eta_\delta(q, \boldsymbol{\rho}, b) := \mathbb{E}\left[\left(G + E_Y^T \mathbf{VS}\boldsymbol{\rho} + \frac{bY - \Delta_Y}{q}\right)_-^2\right] - (1 - \|\boldsymbol{\rho}\|_2^2)\gamma$$

그리고 $(q_\delta, \boldsymbol{\rho}\_\delta, b_\delta)$는 $\eta_\delta(q_\delta, \boldsymbol{\rho}\_\delta, b_\delta) = 0$을 만족하는 유일한 삼중항입니다.

### 3.2 공정성 트레이드오프 분석

논문은 다음 세 가지 중요한 트레이드오프를 이론적으로 규명합니다:

**관찰 1: 균형 오류와 표준 오류의 동시 개선**

$\delta$를 적절히 조정하면 CS-SVM이 표준 오류 $\mathcal{R}$도 개선함을 보입니다. 즉, VS-loss는 공정성 메트릭과 표준 정확도를 **동시에** 개선할 수 있습니다.

**관찰 2: 최적 $\delta^\star$의 성질**

균형 오류 $\mathcal{R}_{\text{bal}}$을 최소화하는 최적값 $\delta^\star$에서:

$$R_+ = R_- = Q\left(\frac{\ell_- + \ell_+}{2}\right)$$

이는 두 클래스의 조건부 오류가 **완벽하게 균등화**됨을 의미합니다.

**관찰 3: Equal Opportunity 달성**

그룹 민감 설정에서 GS-SVM은 특정 $\delta_0(\gamma)$에서 명시적 제약 없이 DEO = 0을 달성합니다:

$$\mathcal{R}_{\text{deo}} = \mathcal{R}_{+,1} - \mathcal{R}_{+,2} = 0$$

### 3.3 과매개변수화 비율 $\gamma = d/n$의 영향

- $\gamma > \gamma^\star$: 선형 분리 가능 구역 → VS-loss가 CS-SVM 해로 수렴하여 최적 균형 오류 달성
- $\gamma$가 증가할수록 Equal Opportunity를 달성하는 $\delta_0$값도 증가
- 이중 하강(double descent) 현상이 균형 오류에서도 관찰됨

### 3.4 일반화 성능 향상의 메커니즘 요약

```
훈련 초기          훈련 후기 (TPT)
─────────────────  ──────────────────────────────────
iota_y (가산적)   → 빠른 수렴, 훈련 안정성 확보
                  
Delta_y (승산적)  → CS-SVM 해로 수렴 (결정 경계 최적화)
                     소수 클래스에 큰 마진 부여
                     균형 오류와 Equal Opportunity 동시 최적화
```

---

## 4. 앞으로의 연구에 미치는 영향 및 고려사항

### 4.1 연구에 미치는 영향

**① 이론적 영향**

- 암묵적 편향(implicit bias) 이론을 불균형 학습 문제로 확장한 선구적 연구
- SVM ↔ Cross-Entropy 손실의 연결고리를 VS-loss/CS-SVM으로 일반화
- CGMT(Convex Gaussian Minimax Theorem) 프레임워크를 공정성 분석에 최초 적용

**② 알고리즘적 영향**

- 기존 LA-loss, CDT-loss를 특수 케이스로 통합하는 통합 프레임워크 제시
- 레이블 불균형과 그룹 불균형을 단일 프레임워크에서 처리하는 선례 마련
- DRO(Distributionally Robust Optimization)와의 조합 가능성 시연

**③ 실용적 영향**

- NLP, 컴퓨터 비전에서의 공정성 학습에 직접 적용 가능한 손실 함수 제공
- 의료 진단, 신용 평가 등 레이블 불균형이 심각한 도메인에 응용 가능

### 4.2 앞으로 연구 시 고려할 점

**① 특징 학습과의 통합**

현재 이론은 고정된 특징을 가정합니다. 딥러닝에서는 특징이 동시에 학습되므로, **Neural Collapse** 현상(Papyan et al., 2020)과의 연계 분석이 필요합니다.

**② 하이퍼파라미터 자동화**

현재 $(\tau, \gamma)$ 그리드 탐색이 필요하며, 이를 자동화하는 **bi-level optimization** 또는 **meta-learning** 기반 접근이 유망합니다.

**③ 다양한 공정성 지표로의 확장**

Equal Opportunity 외에도 **Equalized Odds**, **Demographic Parity** 등 다른 공정성 지표에 대한 이론 확장이 필요합니다.

**④ 그룹 레이블 없는 설정**

훈련 데이터에서 그룹 레이블 획득이 어려운 현실적 상황을 고려한 **레이블 없는 공정성 학습** 방법론 개발이 필요합니다.

**⑤ 비선형 분류기로의 확장**

Theorem 1은 선형 분리 가능성을 가정하므로, 비선형 딥러닝 모델에 대한 이론적 확장이 필요합니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 레이블 불균형 학습 관련 연구

| 논문 | 방법 | VS-loss와의 차이점 |
|------|------|-------------------|
| **LA-loss** (Menon et al., 2020, arXiv:2007.07314) | 가산적 로짓 조정 $\iota_y = \tau\log(\pi_y)$ | Fisher 일관성 기반, TPT에서 비효과적 |
| **CDT-loss** (Ye et al., 2020) | 승산적 로짓 조정 $\Delta_y = (N_y/N_{\max})^\gamma$ | 훈련 초기 소수 클래스에 해로움 |
| **LDAM** (Cao et al., NeurIPS 2019) | 레이블 분포 인식 마진 손실 | 가산적 조정만 사용, VS-loss의 특수 케이스 |
| **BBN** (Zhou et al., 2020) | 이중 브랜치 네트워크 | 데이터 수준 방법, 알고리즘 수준과 직교 |
| **Decoupling** (Kang et al., 2020) | 표현/분류기 분리 학습 | VS-loss와 결합 가능 |

### 5.2 그룹 민감 학습 관련 연구

| 논문 | 방법 | VS-loss와의 차이점 |
|------|------|-------------------|
| **DRO** (Sagawa et al., arXiv:1911.08731) | 최악 서브그룹 오류 최소화 | VS+DRO 조합 시 상호 보완적 |
| **Sagawa et al., ICML 2020** (arXiv 기반) | 과매개변수화가 허위 상관 악화시킴을 분석 | VS-loss의 동기 제공 |

### 5.3 암묵적 편향 이론 관련 연구

| 논문 | 주요 결과 | VS-loss와의 연관성 |
|------|----------|-------------------|
| **Soudry et al., JMLR 2018** | CE 손실의 GD는 SVM 해로 수렴 | Theorem 1의 기반 이론 |
| **Ji & Telgarsky, 2018** (arXiv:1803.07300) | 로지스틱 회귀의 암묵적 편향 | VS-loss 이론의 확장 기반 |
| **Rosset et al., NIPS 2003** | 마진 최대화 손실 함수 이론 | Theorem 1과 직접 연관 |

### 5.4 비교 분석 종합

VS-loss의 핵심 차별점은 다음 수식으로 명확히 표현됩니다:

$$\underbrace{\iota_y}_{\text{가산적: 초기 수렴}} + \underbrace{\Delta_y}_{\text{승산적: TPT 마진}} \xrightarrow{\text{VS-loss}} \text{균형 오류} \downarrow + \text{Equal Opportunity} \uparrow$$

기존 연구들이 가산적 또는 승산적 조정 중 하나만 사용한 반면, VS-loss는 두 조정의 **상보적 역할**을 이론적으로 규명하고 통합함으로써 TPT 전체에 걸쳐 일관된 성능 향상을 달성합니다.

---

## 참고 자료

1. **본 논문**: Kini, G. R., Paraskevas, O., Oymak, S., & Thrampoulidis, C. (2021). *Label-Imbalanced and Group-Sensitive Classification under Overparameterization*. NeurIPS 2021.
2. **코드**: https://github.com/orparask/VS-Loss
3. **LA-loss**: Menon, A. K., et al. (2020). *Long-tail learning via logit adjustment*. arXiv:2007.07314.
4. **CDT-loss**: Ye, H.-J., et al. (2020). *Identifying and compensating for feature deviation in imbalanced deep learning*.
5. **LDAM**: Cao, K., et al. (2019). *Learning imbalanced datasets with label-distribution-aware margin loss*. NeurIPS 2019.
6. **DRO**: Sagawa, S., et al. (2019). *Distributionally robust neural networks for group shifts*. arXiv:1911.08731.
7. **Implicit Bias**: Soudry, D., et al. (2018). *The implicit bias of gradient descent on separable data*. JMLR, 19(1).
8. **Neural Collapse**: Papyan, V., Han, X. Y., & Donoho, D. L. (2020). *Prevalence of neural collapse during the terminal phase of deep learning training*. PNAS.
9. **CGMT**: Thrampoulidis, C., Oymak, S., & Hassibi, B. (2015). *Regularized linear regression: A precise analysis of the estimation error*. COLT 2015.
