# To Grok Grokking: Provable Grokking in Ridge Regression

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

이 논문은 **그로킹(Grokking)** — 훈련 데이터에 과적합(overfitting)된 이후 한참 뒤에야 일반화(generalization)가 시작되는 현상 — 을 **고전적인 릿지 회귀(Ridge Regression)** 설정에서 수학적으로 엄밀하게 증명한 최초의 연구입니다.

핵심 주장은 다음 세 단계가 순차적으로 발생한다는 것입니다:

1. **(i) 조기 과적합**: 훈련 초반에 모델이 훈련 데이터를 과적합함
2. **(ii) 지속적 불량 일반화**: 과적합 이후에도 오랫동안 일반화 성능이 낮게 유지됨
3. **(iii) 최종 일반화 달성**: GD가 결국 좋은 일반화를 보장하는 전역 최솟값(global minimum)에 도달함

### 주요 기여

| 기여 | 내용 |
|------|------|
| **최초의 엔드-투-엔드 증명** | 그로킹의 세 단계 모두를 엄밀히 증명 |
| **정량적 그로킹 시간 경계** | 하이퍼파라미터의 함수로서 그로킹 시간 $t_2 - t_1$에 대한 정량적 하한 도출 |
| **선형 설정에서의 그로킹** | 비선형 구조 없이도 그로킹이 발생함을 증명 |
| **비선형 신경망으로의 확장** | 이론적 경계가 비선형 ReLU 네트워크에서도 질적으로 일치함을 실험적으로 확인 |
| **하이퍼파라미터 제어** | 그로킹을 원리적으로 증폭/제거할 수 있음을 이론·실험적으로 제시 |

---

## 2. 해결하고자 하는 문제, 제안하는 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

기존 그로킹 연구들은 다음과 같은 한계를 가졌습니다:

- **Lyu et al. (2024)**: KKT 점으로의 수렴만 보장하며, 전역 최적성과 그로킹을 직접 증명하지 못함
- **Mohamadi et al. (2024)**: GD가 좋은 일반화를 갖는 해로 수렴함을 증명하지 못함
- **Xu et al. (2024)**: 1회 반복 이후로 일반화 지연이 지속됨을 보이지 못함
- **Boursier et al. (2025)**: 비정규화 해가 일반화하지 못하고 정규화 해가 일반화함을 증명하지 못함

이 논문은 **엔드-투-엔드 수학적 보장**이 결여된 공백을 채우는 것을 목표로 합니다.

### 2.2 제안하는 방법 (수식 포함)

#### 문제 설정

- **교사 함수(Teacher)**: $N^\*(\boldsymbol{x}) = \langle \boldsymbol{\theta}^\*, \boldsymbol{\phi}(\boldsymbol{x}) \rangle$, $\boldsymbol{\theta}^* \in \mathbb{R}^m$
- **학생 모델(Student)**: $N(\boldsymbol{x}; \boldsymbol{\theta}) = \langle \boldsymbol{\theta}, \boldsymbol{\phi}(\boldsymbol{x}) \rangle$, $\boldsymbol{\theta} \in \mathbb{R}^m$
- 특징 맵: $\boldsymbol{\phi}(\boldsymbol{x}): \mathbb{R}^d \mapsto \mathbb{R}^m$ (고정)
- 데이터: $\{(\boldsymbol{x}\_i, y_i)\}_{i=1}^n$, $y_i = N^*(\boldsymbol{x}_i)$ (실현 가능 설정)

#### 훈련 목적 함수 (릿지 회귀)

$$L_n(\boldsymbol{\theta}; \lambda) = \underbrace{\frac{1}{2n}\sum_{i=1}^{n}(N(\boldsymbol{x}_i;\boldsymbol{\theta}) - N^*(\boldsymbol{x}_i))^2}_{L_n(\boldsymbol{\theta})\text{ : 비정규화 손실}} + \underbrace{\frac{\lambda}{2}\|\boldsymbol{\theta}\|_2^2}_{L_\lambda(\boldsymbol{\theta})\text{ : } \ell_2 \text{ 패널티}} $$

여기서 $\lambda > 0$은 정규화 파라미터입니다.

#### 경사 하강법 (GD) 업데이트

$$\boldsymbol{\theta}^{(t+1)} = \boldsymbol{\theta}^{(t)} - \eta \nabla_{\boldsymbol{\theta}} L_n(\boldsymbol{\theta}^{(t)}; \lambda) $$

구체적으로 전개하면:

$$\boldsymbol{\theta}^{(t+1)} = \boldsymbol{\theta}^{(t)} - \frac{\eta}{n}\sum_{i=1}^{n}\left\langle \boldsymbol{\theta}^{(t)} - \boldsymbol{\theta}^*, \boldsymbol{\phi}(\boldsymbol{x}_i)\right\rangle \boldsymbol{\phi}(\boldsymbol{x}_i) - \eta\lambda\boldsymbol{\theta}^{(t)}$$

이는 행렬 형식으로:

$$\boldsymbol{\theta}^{(t)} = \left(\boldsymbol{I}_m - \eta\left(\frac{1}{n}\boldsymbol{\Phi}^\top\boldsymbol{\Phi} + \lambda\boldsymbol{I}_m\right)\right)\boldsymbol{\theta}^{(t-1)}$$

#### 직교 분해를 통한 핵심 분석

$\boldsymbol{\theta}^{(t)} = \boldsymbol{\theta}\_\parallel^{(t)} + \boldsymbol{\theta}_\perp^{(t)}$로 분해합니다:
- $\boldsymbol{\theta}_\parallel^{(t)}$: $\boldsymbol{\Phi}$의 행 공간 성분 (데이터 스패닝 부분공간)
- $\boldsymbol{\theta}_\perp^{(t)}$: $\boldsymbol{\Phi}$의 영 공간 성분 (보완 부분공간)

GD 업데이트는 각각:

$$\boldsymbol{\theta}_\parallel^{(t+1)} = \boldsymbol{\theta}_\parallel^{(t)} - \frac{\eta}{n}\boldsymbol{\Phi}^\top\boldsymbol{\Phi}\boldsymbol{\theta}_\parallel^{(t)} - \eta\lambda\boldsymbol{\theta}_\parallel^{(t)}$$

$$\boldsymbol{\theta}_\perp^{(t+1)} = (1-\eta\lambda)\boldsymbol{\theta}_\perp^{(t)}$$

이 분해가 그로킹의 핵심입니다. $\boldsymbol{\theta}\_\perp^{(t)} = (1-\eta\lambda)^t \boldsymbol{\theta}_\perp^{(0)}$이므로, $\lambda$가 작을 때 보완 부분공간 성분은 매우 느리게 감쇠합니다.

### 2.3 주요 정리들

#### 정리 4.1 (영 교사에 대한 엔드-투-엔드 그로킹)

$N^*(\boldsymbol{x}) = 0$일 때, $\eta \leq 2/(L + 2\lambda)$이면 임의의 $t \in \mathbb{N}$에 대해:

**(i) 훈련 손실 수렴 (빠름)**:

$$L_n(\boldsymbol{\theta}^{(t)}) \leq \frac{L}{2} \cdot \left(1 - \frac{1}{n}\eta\lambda_{\min}^+(\boldsymbol{\Phi}^\top\boldsymbol{\Phi}) - \eta\lambda\right)^{2t} \cdot \|\boldsymbol{\theta}^{(0)}\|_2^2$$

**(ii) 일반화 오류 하한 (느림)**, 확률 $1 - 2e^{-(m-n)/32}$ 이상:

$$L(\boldsymbol{\theta}^{(t)}) \geq \lambda_{\min}(\boldsymbol{\Sigma}) \cdot (1-\eta\lambda)^{2t} \cdot \frac{(m-n)\nu^2}{2}$$

**(iii) 파라미터 노름 감쇠**:

$$\|\boldsymbol{\theta}^{(t)}\|_2^2 \leq (1-\eta\lambda)^{2t} \cdot \|\boldsymbol{\theta}^{(0)}\|_2^2$$

> **핵심 통찰**: 훈련 손실은 $(1 - \frac{1}{n}\eta\lambda_{\min}^+(\boldsymbol{\Phi}^\top\boldsymbol{\Phi}) - \eta\lambda)^{2t}$ 속도로 수렴하고, 일반화 손실은 $(1-\eta\lambda)^{2t}$로 수렴합니다. $\lambda$가 충분히 작으면 전자가 후자보다 **훨씬 빠르게** 수렴하여 그로킹이 발생합니다.

#### 정리 4.2 (실현 가능 릿지 회귀에 대한 엔드-투-엔드 그로킹)

다음을 정의합니다:
- $t_1 := \max\{t \in \mathbb{N} : L_n(\boldsymbol{\theta}^{(t)}) \geq \epsilon\}$: 과적합 완료 시점
- $t_2 := \min\{t \in \mathbb{N} : L(\boldsymbol{\theta}^{(t)}) \leq c\}$: 일반화 달성 시점

**충분 조건**:

충분히 큰 표본 크기:
$$n = \Omega\left(\frac{b^4\|\boldsymbol{\theta}^*\|_2^4}{\epsilon^2}\log\left(\frac{1}{\delta}\right)\right) $$

충분히 큰 특징 공간 차원:

```math
m = n + \Omega\left(\max\left\{\log\left(\frac{1}{\delta}\right),\ \frac{\|\boldsymbol{\theta}^*\|_2^2}{\nu^2},\ \frac{c^2}{\lambda_{\min}^2(\boldsymbol{\Sigma})\nu^2}\right\}\right)
```

충분히 작은 가중치 감쇠:

```math
\lambda = O\left(\min\left\{\frac{\epsilon}{\|\boldsymbol{\theta}^*\|_2^2},\ \frac{b\sqrt{\epsilon}}{\|\boldsymbol{\theta}^*\|_2}\right\}\right)
```

**결론**: 확률 $1-\delta$ 이상으로:

$$t_1 \leq \frac{n\ln\left(\frac{6b^2\|\boldsymbol{\theta}^{(0)}\|_2^2}{\epsilon}\right)}{2\eta\lambda_{\min}^+(\boldsymbol{\Phi}^\top\boldsymbol{\Phi})} $$

$$t_2 \geq \frac{\ln\left(\frac{(m-n)\nu^2}{2}\left(\sqrt{\frac{c}{\lambda_{\min}(\boldsymbol{\Sigma})}} + \|\boldsymbol{\theta}^*\|_2\right)^{-2}\right)}{4\eta\lambda} $$

그리고 충분히 큰 $t$에 대해 $L(\boldsymbol{\theta}^{(t)}) \leq \epsilon$.

#### 가우시안 분포 가정 하의 명시적 경계 (실험용)

$\boldsymbol{\phi}(\boldsymbol{x}_i) \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0, \frac{1}{m}\boldsymbol{I}_m)$이고 $\eta\lambda \leq 0.01$이면:

$$t_2 \geq \frac{\ln\left(\frac{(m-n)\nu^2}{8m\epsilon}\right)}{2.02\eta\lambda} \quad \text{and} \quad t_1 \leq \frac{n\ln\left(\frac{14m\nu^2}{\epsilon}\right)}{2\eta\lambda_{\min}^+(\boldsymbol{\Phi}^\top\boldsymbol{\Phi})} $$

### 2.4 모델 구조

이 논문의 학생 모델은 **선형 회귀 모델**입니다:

$$N(\boldsymbol{x}; \boldsymbol{\theta}) = \langle \boldsymbol{\theta}, \boldsymbol{\phi}(\boldsymbol{x})\rangle, \quad \boldsymbol{\theta} \in \mathbb{R}^m$$

특징 맵 $\boldsymbol{\phi}(\boldsymbol{x})$는 임의의 고정된 맵일 수 있으며, 실험에서는 다음을 포함합니다:
- 항등 맵 ( $\boldsymbol{\phi}(\boldsymbol{x}) = \boldsymbol{x}$ )
- ReLU 랜덤 특징 ( $\boldsymbol{\phi}_j(\boldsymbol{x}) = \sigma(\langle \boldsymbol{w}_j, \boldsymbol{x}\rangle)$ )
- 랜덤 푸리에 특징 ( $\boldsymbol{\phi}(\boldsymbol{x}) = \sqrt{2/m}\cos(\boldsymbol{W}\boldsymbol{x} + \boldsymbol{b})$ )

초기화: $\boldsymbol{\theta}^{(0)} \sim \mathcal{N}(\boldsymbol{0}, \nu^2\boldsymbol{I}_m)$

### 2.5 하이퍼파라미터가 그로킹에 미치는 영향

| 하이퍼파라미터 | $t_1$에 대한 영향 | $t_2$에 대한 영향 | 그로킹 시간 $(t_2-t_1)$에 대한 영향 |
|---|---|---|---|
| $\lambda \downarrow$ (작게) | 거의 영향 없음 | $t_2 \propto 1/\lambda$로 증가 | **크게 증폭** |
| $n \downarrow$ (작게) | $t_1$ 감소 (빠른 과적합) | 상대적으로 작은 영향 | 증폭 (주로 $t_1$ 감소) |
| $m \uparrow$ (크게) | $t_1$, $t_2$에 직접 영향 작음 | 상대적으로 작은 영향 | 직접 영향 작음 |
| $\nu^2 \uparrow$ (크게) | $t_1 \propto \ln(\nu^2)$으로 증가 | $t_2 \propto \ln(\nu^2)$으로 증가 (더 빠르게) | 증폭 |

### 2.6 성능 향상 및 한계

**성능 향상**:
- 이론 경계가 실험 결과와 정량적으로 일치 (Figure 2, 3, 4)
- 비선형 신경망에서도 동일한 하이퍼파라미터 의존성 패턴 확인
- 노이즈가 있는 레이블, 랜덤 푸리에 특징 등 다양한 설정에서도 그로킹 관찰

**한계**:
- **실현 불가능(non-realizable) 설정**: 레이블 노이즈가 있는 경우 엔드-투-엔드 증명이 부재
- **비선형 신경망**: 두 레이어 ReLU 네트워크에서 과적합 발생 증명 자체가 어려움
- **Lazy-to-Rich 전환**: 기존 연구들이 주장하는 lazy→rich 체제 전환을 통한 그로킹 증명 미달성
- **$\lambda_{\min}^+(\boldsymbol{\Phi}^\top\boldsymbol{\Phi})$의 의존성**: $n$과 $m$에 대한 정량적 경계 부재 (Marcenko-Pastur 법칙은 점근적 결과만 제공)
- **분류 문제**: 회귀 설정에 한정됨

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화를 방해하는 구조적 원인

이 논문은 그로킹에서 일반화 지연의 구조적 원인을 명확히 밝힙니다.

**과도하게 매개변수화된 상황** ($m \gg n$)에서:

$$\boldsymbol{\theta}_\perp^{(t)} = (1-\eta\lambda)^t \boldsymbol{\theta}_\perp^{(0)}$$

보완 부분공간 성분 $\boldsymbol{\theta}_\perp$는 순수히 가중치 감쇠에 의해서만 감쇠되며, 데이터 정보는 전달되지 않습니다. 일반화 오류의 하한은:

$$L(\boldsymbol{\theta}^{(t)}) \geq \lambda_{\min}(\boldsymbol{\Sigma})\|\boldsymbol{\theta}^{(t)}\|_2^2 \geq \lambda_{\min}(\boldsymbol{\Sigma})(1-\eta\lambda)^{2t}\|\boldsymbol{\theta}_\perp^{(0)}\|_2^2$$

$\lambda$가 작을 때 $(1-\eta\lambda)^{2t}$가 매우 느리게 감소하여 일반화가 오래 지연됩니다.

### 3.2 일반화 보장 메커니즘 (Theorem 4.6)

일반화 성능은 **라데마허 복잡도(Rademacher Complexity)** 기반의 균일 수렴(uniform convergence) 논증에 의해 보장됩니다.

GD는 정규화 목적함수의 전역 최솟값 $\boldsymbol{\theta}\_\lambda^* = \arg\min_{\boldsymbol{\theta}} L_n(\boldsymbol{\theta}; \lambda)$로 수렴합니다. 이때:

```math
\|\boldsymbol{\theta}_\lambda^*\|_2 \leq \|\boldsymbol{\theta}^*\|_2
```

함수 클래스를 $\mathcal{H}_{\boldsymbol{\theta}}(B) = \{\boldsymbol{x} \mapsto \langle\boldsymbol{\theta}, \boldsymbol{\phi}(\boldsymbol{x})\rangle : \|\boldsymbol{\theta}\|_2 \leq B\}$로 정의하면:

$$\text{Rad}_{S_n}(\mathcal{H}_{\boldsymbol{\theta}}(B)) \leq B\sqrt{\frac{L}{n}}$$

이를 통해 (Theorem 4.6):

```math
n = \Omega\left(\frac{b^4\|\boldsymbol{\theta}^*\|_2^4}{\epsilon^2}\log\left(\frac{1}{\delta}\right)\right) \Rightarrow L(\boldsymbol{\theta}_\lambda^*) \leq 2L_n(\boldsymbol{\theta}_\lambda^*) + \epsilon
```

### 3.3 일반화 향상을 위한 실질적 시사점

이 논문은 일반화 성능 향상을 위한 다음과 같은 구체적 지침을 제공합니다:

**1. 가중치 감쇠 $\lambda$의 역할**

$\lambda$는 두 가지 상충적 역할을 합니다:
- 너무 작으면: $t_2 \propto 1/\lambda$로 일반화가 매우 늦게 달성됨
- 너무 크면: 모델 클래스 복잡성을 줄여 균일 수렴을 보장하지만, 편향(bias)을 증가시킬 수 있음

최적 $\lambda$: 

```math
\lambda = O\left(\min\left\{\frac{\epsilon}{\|\boldsymbol{\theta}^*\|_2^2}, \frac{b\sqrt{\epsilon}}{\|\boldsymbol{\theta}^*\|_2}\right\}\right)
```

**2. 표본 크기 $n$의 역할**

$n$이 클수록 일반화가 더 이른 시점에, 더 확실하게 달성됩니다. 단, $n$을 늘리면 $\lambda_{\min}^+(\boldsymbol{\Phi}^\top\boldsymbol{\Phi})$가 커져 $t_1$이 감소하고 $t_2 - t_1$이 줄어들어 그로킹이 약화될 수 있습니다.

**3. 특징 공간 차원 $m$의 역할**

$m \gg n$일수록 그로킹이 더 강하게 나타납니다. 이는 보완 부분공간의 차원이 커지기 때문입니다. 그러나 $m$을 늘린다고 최종 일반화 성능이 나빠지지는 않습니다.

**4. 임계값 기반 정확도 지표**

연속 손실 대신 임계값 기반 지표 $\mathbb{P}_{\boldsymbol{x}}\left(\left(N(\boldsymbol{x};\boldsymbol{\theta}^{(t)}) - N^*(\boldsymbol{x})\right)^2 \leq \epsilon\right)$를 사용하면 선형 회귀에서도 분류에서와 같은 plateau 현상을 재현할 수 있습니다.

**5. 그로킹은 피할 수 있다**

이 논문의 핵심 메시지 중 하나는 그로킹이 딥러닝의 **내재적 실패 모드가 아니라 특정 하이퍼파라미터 조건의 결과**라는 것입니다. $\lambda$를 적절히 키우면 일반화 지연을 제거할 수 있습니다.

---

## 4. 앞으로의 연구에 미치는 영향과 고려할 점

### 4.1 앞으로의 연구에 미치는 영향

**이론적 측면**:

1. **비실현 가능 설정으로의 확장**: 레이블 노이즈가 있거나 교사가 모델 클래스 밖에 있는 경우의 증명이 필요합니다. 논문은 이를 중요한 미해결 문제로 남겨두고 있습니다.

2. **비선형 신경망에서의 엄밀한 증명**: Lazy-to-Rich 전환을 통한 그로킹 증명이 여전히 미완성입니다. NTK 분석으로는 $m$개의 뉴런의 집합적 기여로 인해 과적합이 지속됨을 보이기 어렵습니다.

3. **분류 문제로의 확장**: 회귀와 달리 분류에서는 로지스틱 손실 등 다른 손실 함수를 사용하며, Beck et al. (2025)처럼 점근적 가정 하에서만 그로킹 증거가 존재합니다.

4. **Marchenko-Pastur 법칙의 유한 샘플 버전**: $\lambda_{\min}^+(\boldsymbol{\Phi}^\top\boldsymbol{\Phi})$의 $n$, $m$에 대한 유한 샘플 정량적 경계는 현재 알려져 있지 않습니다.

5. **Riemannian 노름 최소화와의 연결**: Boursier et al. (2025)의 "ridgeless→ridge" 전환 프레임워크와 이 논문의 결과를 통합하는 이론이 필요합니다.

**실용적 측면**:

1. **하이퍼파라미터 튜닝 지침**: 그로킹 시간에 대한 정량적 경계는 실제 모델 훈련에서 언제 일반화가 달성될지 예측하고 $\lambda$, $\eta$, $\nu^2$를 설정하는 데 활용 가능합니다.

2. **조기 종료 전략 재검토**: 그로킹 현상은 조기 종료가 좋은 일반화를 놓칠 수 있음을 시사합니다. 언제까지 훈련해야 하는지에 대한 이론적 근거를 제공합니다.

3. **LLM에서의 그로킹**: Zhu et al. (2024), Wang et al. (2024)에서 관찰된 LLM의 그로킹 현상을 이해하는 기초 이론으로 활용 가능합니다.

### 4.2 앞으로 연구 시 고려할 점

**수학적 도구 관련**:

- 현재의 균일 수렴 기반 일반화 경계는 $n = \Omega(b^4\|\boldsymbol{\theta}^*\|_2^4/\epsilon^2)$라는 다소 큰 표본 크기를 요구합니다. PAC-Bayes 경계나 안정성 기반 방법론이 더 tight한 경계를 줄 수 있습니다.

- 연속 시간 극한(gradient flow)과의 일관성 분석이 필요합니다.

**설정 관련**:

- 회귀에서 연속적으로 감소하는 손실과 분류에서의 plateau가 어떻게 통합적으로 이해될 수 있는지 추가 분석이 필요합니다.

- 미실현 가능(agnostic) 학습에서 가중치 감쇠를 너무 작게 하면 일반화를 완전히 방해할 수 있다는 논문의 관찰은 중요한 경계 조건입니다.

**비선형 확장 관련**:

- 두 레이어 ReLU 네트워크에서 과적합이 지속됨을 증명하는 것이 주요 장벽입니다. 이를 위해 NTK 범위를 벗어나는 새로운 분석 도구가 필요합니다.

- Lauditi et al. (2025)의 관찰 — 가중치 감쇠가 초기 NTK를 데이터 의존 NTK로 전환시킨다 — 이 그로킹으로 이어질 수 있는지 분석이 필요합니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 논문 | 설정 | 방법론 | 그로킹 증명 수준 | 본 논문과의 차이 |
|------|------|--------|-----------------|----------------|
| **Power et al. (2022)** | 모듈러 산술, 알고리즘 데이터 | 실험적 관찰 | 관찰만 | 최초 명명, 이론 부재 |
| **Blanc et al. (2020)** | 행렬 감지, 얕은 신경망 | 묵시적 정규화 | 간접적 관찰 | 엄밀한 증명 없음 |
| **Liu et al. (2022, NeurIPS)** | 알고리즘 데이터 | 표현 학습 렌즈 | 설명적 | 엄밀한 경계 없음 |
| **Nanda et al. (2023, ICLR)** | 모듈러 산술 | 메커니즘적 해석 | 설명적 | 정량적 경계 없음 |
| **Kumar et al. (2024, ICLR)** | 일반 신경망 | Lazy→Rich 전환 | 부분적 | 전역 수렴 및 일반화 증명 부재 |
| **Lyu et al. (2024, ICLR)** | 동차 신경망 | 커널→리치 체제 전환 | KKT 점 수렴만 | 전역 최적성 및 일반화 증명 불완전 |
| **Mohamadi et al. (2024, ICML)** | 모듈러 덧셈 | 2층 이차 네트워크, $\ell_\infty$ 정규화 | 부분적 | GD 수렴 및 일반화 증명 불완전 |
| **Xu et al. (2024, ICLR)** | XOR 군집 데이터, 이진 분류 | ReLU 네트워크 | 부분적 (1회 이후 지연 미증명) | 일반화 지연의 지속성 미증명 |
| **Boursier et al. (2025, NeurIPS)** | Smooth loss, GF + weight decay | Ridgeless→Ridge 전환 | 부분적 | 비정규화 해의 일반화 실패 증명 부재 |
| **Levi et al. (2024, ICLR)** | 선형 회귀 | 랜덤 행렬 이론 | 비엄밀 (non-rigorous) | 공식 보장 부재, 가우시안 데이터만 |
| **Beck et al. (2025, ICML)** | 로지스틱 회귀 | 점근적 가정 | 증거 제시 | 엄밀한 증명 부재 |
| **Mallinar et al. (2025, ICML)** | 모듈러 산술 | Recursive Feature Machines | 실험적 | GD 기반 아님 |
| **Žunkovič & Ilievski (2024, JMLR)** | 선형 분류 | 정량적 경계 제시 | 비엄밀 | 엄밀한 증명 부재 |
| **본 논문 (Xu et al., 2026, ICML)** | 릿지 회귀 (선형) | GD + $\ell_2$ 정규화, 직교 분해 | **완전한 엔드-투-엔드 증명** | 최초의 완전 엄밀 증명 |

**요약**: 기존 연구들은 그로킹을 관찰하거나 부분적으로 설명했지만, 세 단계를 모두 수학적으로 엄밀하게 증명한 연구는 본 논문이 최초입니다. 특히 정량적 그로킹 시간 경계를 하이퍼파라미터의 함수로 제시한 것은 이전에 없었습니다.

---

## 참고 자료

**주요 논문 (PDF 원문 기반)**:
- Xu, M., Vardi, G., & Safran, I. (2026). "To Grok Grokking: Provable Grokking in Ridge Regression." *Proceedings of the 43rd ICML*. arXiv:2601.19791v3

**논문 내 인용 문헌**:
- Power, A., et al. (2022). "Grokking: Generalization beyond overfitting on small algorithmic datasets." arXiv:2201.02177
- Lyu, K., et al. (2024). "Dichotomy of early and late phase implicit biases can provably induce grokking." *ICLR 2024*
- Mohamadi, M. A., et al. (2024). "Why do you grok? A theoretical analysis on grokking modular addition." *ICML 2024*
- Kumar, T., et al. (2024). "Grokking as the transition from lazy to rich training dynamics." *ICLR 2024*
- Xu, Z., et al. (2024). "Benign overfitting and grokking in ReLU networks for XOR cluster data." *ICLR 2024*
- Boursier, E., et al. (2025). "A theoretical framework for grokking: Interpolation followed by Riemannian norm minimisation." *NeurIPS 2025*
- Levi, N., et al. (2024). "Grokking in linear estimators–a solvable model that groks without understanding." *ICLR 2024*
- Beck, A., et al. (2025). "Grokking at the edge of linear separability." *ICML 2025*
- Blanc, G., et al. (2020). "Implicit regularization for deep neural networks driven by an Ornstein-Uhlenbeck like process." *COLT 2020*
- Nanda, N., et al. (2023). "Progress measures for grokking via mechanistic interpretability." *ICLR 2023*
- Mallinar, N. R., et al. (2025). "Emergence in non-neural models: grokking modular arithmetic via AGOP." *ICML 2025*
- Žunkovič, B., & Ilievski, E. (2024). "Grokking phase transitions in learning local rules with gradient descent." *JMLR*, 25(199):1–52
- Bartlett, P. L., & Mendelson, S. (2002). "Rademacher and Gaussian complexities." *JMLR*, 3:463–482
- Shalev-Shwartz, S., & Ben-David, S. (2014). *Understanding Machine Learning: From Theory to Algorithms*. Cambridge University Press
- Marcenko, V. A., & Pastur, L. A. (1967). "Distribution of eigenvalues for some sets of random matrices." *Mathematics of the USSR-Sbornik*, 1(4):457
