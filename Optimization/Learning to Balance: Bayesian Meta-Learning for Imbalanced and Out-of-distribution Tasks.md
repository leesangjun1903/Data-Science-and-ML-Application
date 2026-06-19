# Learning to Balance: Bayesian Meta-Learning for Imbalanced and Out-of-distribution Tasks 

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

기존 메타러닝 방법들은 **모든 태스크에 대해 동일한 방식으로 메타 지식(meta-knowledge)을 활용**하도록 학습되어 있습니다. 그러나 현실에서는:

- **태스크 불균형(Task Imbalance)**: 태스크마다 학습 인스턴스 수가 다름
- **클래스 불균형(Class Imbalance)**: 태스크 내 클래스별 인스턴스 수가 다름
- **분포 외 태스크(Out-of-Distribution, OOD)**: 메타테스트 태스크가 메타트레이닝과 다른 분포에서 옴

이러한 현실적 문제를 해결하기 위해, **Bayesian Task-Adaptive Meta-Learning (Bayesian TAML)**을 제안합니다. 이 모델은 **태스크 및 클래스별로 메타러닝과 태스크 특화 학습의 비중을 동적으로 조절**하는 세 가지 균형 변수를 학습합니다.

### 주요 기여 (3가지)

| 기여 | 내용 |
|------|------|
| **문제 정의** | 현실적인 태스크 분포(불균형 + OOD)에서의 메타러닝 문제를 공식적으로 정의 |
| **방법론 제안** | 세 가지 균형 변수를 베이지안 추론으로 학습하는 Bayesian TAML 프레임워크 제안 |
| **실험 검증** | 다양한 현실적 불균형 few-shot 분류 태스크에서 기존 방법 대비 유의미한 성능 향상 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

MAML(Finn et al., 2017)의 세 가지 한계를 극복합니다:

**한계 1: 클래스 불균형**
- 헤드(head) 클래스가 테일(tail) 클래스의 그래디언트를 지배 → 테일 클래스 성능 저하

**한계 2: 태스크 불균형**
- 고정된 스텝사이즈 $\alpha$와 내부 그래디언트 스텝 수로 태스크별 적응 불가

**한계 3: OOD 태스크**
- 공유 초기화 파라미터 $\boldsymbol{\theta}$가 분포 외 태스크에 비효율적

---

### 2.2 제안하는 방법 (수식 포함)

#### ■ MAML 기본 목적함수 (비교 기준)

$$\min_{\boldsymbol{\theta}} \sum_{\tau \sim p(\tau)} \mathcal{L}\!\left(\boldsymbol{\theta} - \alpha \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}; \mathcal{D}^{\tau}); \tilde{\mathcal{D}}^{\tau}\right) $$

#### ■ 세 가지 균형 변수 (Balancing Variables)

| 변수 | 역할 | 처리 대상 |
|------|------|-----------|
| $\boldsymbol{\omega}^{\tau} = (\omega_1^{\tau}, \ldots, \omega_C^{\tau}) \in [0,1]^C$ | 클래스별 그래디언트 스케일링 | 클래스 불균형 |
| $\boldsymbol{\gamma}^{\tau} = (\gamma_1^{\tau}, \ldots, \gamma_L^{\tau}) \in [0,\infty)^L$ | 레이어별 학습률 배율 | 태스크 불균형 |
| $\mathbf{z}^{\tau}$ | 초기 파라미터 변조 | OOD 태스크 |

#### ■ Task-Adaptive Meta-Learning (TAML) 업데이트 규칙

**Step 1: 초기 파라미터 변조 (OOD 대응)**

$$\boldsymbol{\theta}_0 = \boldsymbol{\theta} * \mathbf{z}^{\tau} $$

여기서 채널 가중치는 $\boldsymbol{\theta}_0 \leftarrow \boldsymbol{\theta} \circ \mathbf{z}^{\tau}$, 편향은 $\boldsymbol{\theta}_0 \leftarrow \boldsymbol{\theta} + \mathbf{z}^{\tau}$로 정의됩니다.

**Step 2: 태스크 및 클래스 적응적 내부 업데이트**

$$\boldsymbol{\theta}_k = \boldsymbol{\theta}_{k-1} - \boldsymbol{\gamma}^{\tau} \circ \boldsymbol{\alpha} \circ \sum_{c=1}^{C} \omega_c^{\tau} \nabla_{\boldsymbol{\theta}_{k-1}} \mathcal{L}(\boldsymbol{\theta}_{k-1}; \mathcal{D}_c^{\tau}), \quad k = 1, \ldots, K $$

- $\boldsymbol{\alpha}$: Meta-SGD 방식의 학습 가능한 전역 학습률 벡터
- $\omega_c^{\tau}$: 테일 클래스일수록 크게 → 클래스 불균형 보정
- $\gamma_l^{\tau}$: 많은 샘플의 태스크일수록 크게 → 태스크별 학습량 조절

#### ■ 생성 모델 (Generative Process)

$$p(\mathbf{Y}^{\tau}, \tilde{\mathbf{Y}}^{\tau}, \boldsymbol{\phi}^{\tau} | \mathbf{X}^{\tau}, \tilde{\mathbf{X}}^{\tau}; \boldsymbol{\theta}) = p(\boldsymbol{\phi}^{\tau}) \prod_{n=1}^{N_{\tau}} p(y_n^{\tau} | x_n^{\tau}, \boldsymbol{\phi}^{\tau}; \boldsymbol{\theta}) \prod_{m=1}^{M_{\tau}} p(\tilde{y}_m^{\tau} | \tilde{x}_m^{\tau}, \boldsymbol{\phi}^{\tau}; \boldsymbol{\theta}) $$

여기서 $\boldsymbol{\phi}^{\tau} = \{\tilde{\boldsymbol{\omega}}^{\tau}, \tilde{\boldsymbol{\gamma}}^{\tau}, \tilde{\mathbf{z}}^{\tau}\}$는 세 균형 변수의 입력을 통칭합니다.

#### ■ 변분 추론 (Variational Inference)

진짜 사후분포 $p(\boldsymbol{\phi}^{\tau} | \mathcal{D}^{\tau}, \tilde{\mathcal{D}}^{\tau})$가 다루기 어렵기 때문에, 근사 사후분포를 사용합니다:

$$q(\boldsymbol{\phi}^{\tau} | \mathcal{D}^{\tau}; \psi)$$

ELBO (Evidence Lower Bound):

$$\mathcal{L}^{\tau}_{\boldsymbol{\theta}, \psi} = \frac{N_{\tau} + M_{\tau}}{M_{\tau}} \sum_{m=1}^{M_{\tau}} \mathbb{E}_{q(\boldsymbol{\phi}^{\tau} | \mathcal{D}^{\tau}; \psi)}\!\left[\log p(\tilde{y}_m^{\tau} | \tilde{x}_m^{\tau}, \boldsymbol{\phi}^{\tau}; \boldsymbol{\theta})\right] - \mathrm{KL}\!\left[q(\boldsymbol{\phi}^{\tau} | \mathcal{D}^{\tau}; \psi) \| p(\boldsymbol{\phi}^{\tau})\right] $$

완전 인수분해 가정:

$$q(\boldsymbol{\phi}^{\tau} | \mathcal{D}^{\tau}; \psi) = \prod_c q(\tilde{\omega}_c^{\tau} | \mathcal{D}^{\tau}; \psi) \prod_l q(\tilde{\gamma}_l^{\tau} | \mathcal{D}^{\tau}; \psi) \prod_i q(\tilde{z}_i^{\tau} | \mathcal{D}^{\tau}; \psi) $$

각 차원이 학습 가능한 평균과 분산을 가진 **단변량 가우시안**을 따른다고 가정하며, 사전분포는 $p(\boldsymbol{\phi}^{\tau}) = \mathcal{N}(0, 1)$로 설정하여 KL 발산의 닫힌 형태를 보장합니다.

#### ■ 최종 메타-트레이닝 목적함수 (Monte Carlo 근사)

$$\min_{\boldsymbol{\theta}, \psi} \frac{1}{M_{\tau}} \sum_{m=1}^{M_{\tau}} \frac{1}{S} \sum_{s=1}^{S} -\log p(\tilde{y}_m^{\tau} | \tilde{x}_m^{\tau}, \boldsymbol{\phi}_s^{\tau}; \boldsymbol{\theta}) + \frac{1}{N_{\tau} + M_{\tau}} \mathrm{KL}\!\left[q(\boldsymbol{\phi}^{\tau} | \mathcal{D}^{\tau}; \psi) \| p(\boldsymbol{\phi}^{\tau})\right] $$

#### ■ 메타테스트 예측 (MC 근사)

```math
p(\tilde{y}_*^{\tau} | \tilde{x}_*^{\tau}; \boldsymbol{\theta}) = \mathbb{E}_q\!\left[p(\tilde{y}_*^{\tau} | \tilde{x}_*^{\tau}, \boldsymbol{\phi}^{\tau}; \boldsymbol{\theta})\right] \approx \frac{1}{S} \sum_{s=1}^{S} p(\tilde{y}_*^{\tau} | \tilde{x}_*^{\tau}, \boldsymbol{\phi}_s^{\tau}; \boldsymbol{\theta}), \quad \boldsymbol{\phi}_s^{\tau} \sim q(\boldsymbol{\phi}^{\tau} | \mathcal{D}^{\tau}; \psi)
```

---

### 2.3 모델 구조

#### ■ 추론 네트워크 (Inference Network): 계층적 집합 인코더

```
입력: 분류 데이터셋 D^τ (클래스별 인스턴스 집합)
     ↓
[NN₁] 인스턴스 레벨 인코딩 (3×3 Conv × 2, MaxPool × 2, FC)
     ↓
StatisticsPooling (클래스별): mean + variance + cardinality
     → s_c (클래스 표현)
     ↓
[NN₂] 클래스 레벨 인코딩 (FC layers)
     ↓
StatisticsPooling (태스크 레벨): mean + variance + cardinality
     → v^τ (태스크 표현)
     ↓
┌─────────────────┬──────────────────┬─────────────────┐
│ω^τ 생성 (s_c)   │γ^τ 생성 (v^τ)   │z^τ 생성 (v^τ)  │
│(μ_ω, σ_ω) 출력  │(μ_γ, σ_γ) 출력  │(μ_z, σ_z) 출력  │
└─────────────────┴──────────────────┴─────────────────┘
```

**StatisticsPooling**의 중요성:
- **평균(Mean)**: 집합의 대표 통계
- **분산(Variance)**: 데이터 다양성 포착 (복제 인스턴스 탐지)
- **기수(Cardinality, N)**: 불균형 정도 직접 감지 → 단순 평균 풀링의 한계 극복

집합-of-집합 구조의 수학적 정당성 (DeepSets 정리 적용):

```math
F\!\left(\left\{\{\mathbf{x}_{1,1}, \ldots\}, \ldots, \{\mathbf{x}_{C,1}, \ldots\}\right\}\right) = \mathrm{NN}_3\!\left(\sum_{c=1}^{C} \mathrm{NN}_2\!\left(\sum_{i=1}^{N} \mathrm{NN}_1(\mathbf{x}_{c,i})\right)\right)
```

---

### 2.4 성능 향상

#### ■ Any-shot 분류 (Table 1)

| 모델 | CIFAR-FS→CIFAR-FS | CIFAR-FS→SVHN (OOD) | miniImageNet→CUB (OOD) |
|------|:-----------------:|:-------------------:|:----------------------:|
| MAML | 71.55 | 45.17 | 65.77 |
| Meta-SGD | 72.71 | 46.45 | 65.94 |
| MT-net | 72.30 | 49.17 | 66.09 |
| ABML | 67.24 | 36.52 | 57.88 |
| Proto. Networks | 73.24 | 42.91 | 60.80 |
| **Bayesian TAML** | **75.15** | **51.87** | **71.71** |

특히 OOD 태스크(SVHN, CUB)에서 **5%p 이상의 큰 성능 향상**이 관찰됩니다.

#### ■ Multi-dataset Any-shot 분류 (Table 2)

OOD 데이터셋인 Traffic Signs에서 Bayesian TAML이 **64.81%**로 MAML(**51.96%**)보다 약 13%p 높은 성능을 보입니다.

#### ■ Ablation Study 결과

**베이지안 모델링 효과 (Table 5):**

| 모델 | CIFAR-FS | SVHN(OOD) |
|------|:--------:|:---------:|
| MAML | 70.19 | 41.81 |
| Deterministic TAML | 73.82 | 46.78 |
| Bayesian TAML (Naive) | 73.52 | 47.15 |
| **Bayesian TAML (MC)** | **75.15** | **51.87** |

→ Bayesian 모델링이 특히 OOD 태스크에서 결정론적 버전 대비 **5%p** 향상

**데이터셋 인코딩 효과 (Table 6):**

$$\text{Mean} + \text{Var.} + N \; (75.15\%) > \text{Mean} + N \; (74.88\%) > \text{Mean} \; (73.69\%)$$

---

### 2.5 한계

1. **계산 비용**: 세 균형 변수 추론 및 MC 샘플링으로 MAML 대비 추론 비용 증가 (메타테스트 시 $S=10$ 샘플 필요)
2. **백본 의존성**: 4-block Conv를 기준으로 검증되어, 더 깊은 네트워크(ResNet, ViT 등)로의 확장 검증 미흡
3. **태스크 분포 가정**: 균일분포 $N_c \sim \text{Unif}(1, 50)$에서 샘플링하는 방식이 모든 현실 시나리오를 대표하지 않을 수 있음
4. **완전 인수분해(Fully Factorized) 가정**: 세 균형 변수 간 상관관계를 무시 → 더 복잡한 사후분포 표현 어려움
5. **회귀/강화학습으로의 확장 미검증**: 이미지 분류에 집중되어 타 도메인 적용 가능성 미확인

---

## 3. 일반화 성능 향상 가능성 (중점 분석)

### 3.1 메타 지식과 태스크 특화 학습의 동적 균형

Bayesian TAML의 핵심 일반화 메커니즘은 **"얼마나 메타 지식에 의존할지"를 태스크별로 자동 결정**하는 것입니다.

$$\boldsymbol{\theta}_k = \underbrace{\boldsymbol{\theta} * \mathbf{z}^{\tau}}_{\text{적응적 초기화}} - \underbrace{\boldsymbol{\gamma}^{\tau} \circ \boldsymbol{\alpha}}_{\text{적응적 학습률}} \circ \sum_{c=1}^C \underbrace{\omega_c^{\tau}}_{\text{클래스 가중치}} \nabla_{\boldsymbol{\theta}_{k-1}} \mathcal{L}(\boldsymbol{\theta}_{k-1}; \mathcal{D}_c^{\tau})$$

- **적은 샘플 태스크**: $\gamma^{\tau} \to 0$, $\mathbf{z}^{\tau} \approx \mathbf{1}$ → 메타 지식에 가깝게 유지
- **많은 샘플 태스크**: $\gamma^{\tau} \gg 0$ → 태스크 특화 학습에 더 의존
- **OOD 태스크**: $\mathbf{z}^{\tau}$가 큰 분산을 학습하여 초기화를 재배치

### 3.2 베이지안 불확실성의 일반화 기여

베이지안 프레임워크가 일반화에 기여하는 세 가지 경로:

**① 앙상블 효과**: MC 샘플링으로 다양한 태스크 특화 예측기를 앙상블

```math
p(\tilde{y}_*^{\tau} | \tilde{x}_*^{\tau}) \approx \frac{1}{S} \sum_{s=1}^{S} p(\tilde{y}_*^{\tau} | \tilde{x}_*^{\tau}, \boldsymbol{\phi}_s^{\tau}; \boldsymbol{\theta})
```

**② 불확실성 기반 OOD 적응**: $\mathbf{z}^{\tau}$의 큰 분산이 OOD 태스크에서 효과적 학습률 증가 역할

T-SNE 시각화(Figure 4)에서 OOD 태스크($\text{SVHN}$, $\text{CUB}$)의 $\boldsymbol{\theta} * \mathbb{E}[\mathbf{z}^{\tau}]$가 초기 $\boldsymbol{\theta}$에서 더 멀리 이동함을 확인

**③ KL 정규화의 과적합 방지**:

$$\frac{1}{N_{\tau} + M_{\tau}} \mathrm{KL}\!\left[q(\boldsymbol{\phi}^{\tau} | \mathcal{D}^{\tau}; \psi) \| p(\boldsymbol{\phi}^{\tau})\right]$$

KL 항이 균형 변수의 지나친 특화를 방지하여 일반화 성능 유지

### 3.3 계층적 집합 인코딩의 일반화 기여

**왜 계층적 인코딩이 일반화에 유리한가:**

단순 평균 풀링은 클래스 라벨 정보와 샘플 수를 무시하는 반면, StatisticsPooling은:

- **기수(N)**: 불균형 정도를 직접 감지 → 새로운 불균형 패턴에 즉각 적응
- **분산(Var)**: 데이터 품질 추정 (복제 샘플 탐지) → 정보량 기반 균형 조절
- **계층 구조**: 클래스 내 + 클래스 간 두 레벨 통계 포착 → 임의 $C$-way 태스크에 일반화

$$\mathbf{v}^{\tau} = \text{StatisticsPooling}\!\left(\{\text{NN}_2(\mathbf{s}_c)\}_{c=1}^C\right), \quad \mathbf{s}_c = \text{StatisticsPooling}\!\left(\{\text{NN}_1(\mathbf{x})\}_{\mathbf{x} \in \mathbf{X}_c^{\tau}}\right)$$

### 3.4 Ablation에서 확인된 일반화 효과

Figure 5(a): 태스크 크기가 5~2000까지 변화할 때, Bayesian z-TAML과 γ-TAML은 **학습 범위를 벗어난 2000-shot 태스크(Extrapolation 영역)에서도** Meta-SGD, MAML 대비 더 높은 성능 유지 → **분포 외 태스크 크기에 대한 일반화** 확인

---

## 4. 앞으로의 연구 영향 및 고려할 점

### 4.1 앞으로의 연구에 미치는 영향

#### ① 현실적 메타러닝 벤치마크 확립
이 논문은 동일 샷(N-way K-shot) 고정 설정에서 벗어나 **any-shot 분류**라는 더 현실적인 평가 패러다임을 제시했습니다. 이후 연구들이 보다 현실적인 설정에서 평가하는 것을 촉진했습니다.

#### ② 메타 지식의 선택적 활용 패러다임
"언제, 얼마나 메타 지식을 사용할 것인가"라는 질문이 이후 연구들의 핵심 과제로 부각되었습니다. 특히 continual learning, domain adaptation과의 교차점에서 이 관점이 재조명됩니다.

#### ③ 베이지안 메타러닝 활성화
균형 변수에 베이지안 불확실성을 도입한 접근법은 이후 메타러닝에서 불확실성 추정을 더 정교하게 다루는 연구들의 기반이 되었습니다.

#### ④ 집합 인코딩의 중요성 재확인
계층적 StatisticsPooling이 단순 평균 풀링보다 유의미하게 우수함을 보여 이후 **태스크 컨텍스트 인코딩** 연구에 영향을 미쳤습니다.

---

### 4.2 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의**: 아래 언급되는 2020년 이후 논문들에 대해서는, 제가 학습한 데이터를 기반으로 기술하지만 세부 수치나 방법론의 일부가 불완전할 수 있습니다. 제공된 PDF 외 출처는 기억 기반이므로 참고용으로만 활용하시길 권장합니다.

#### ■ Meta-Dataset (Triantafillou et al., ICLR 2020)
- 본 논문과 동시기에 발표된 대규모 다중 데이터셋 메타러닝 벤치마크
- Bayesian TAML과의 관계: Bayesian TAML의 다중 데이터셋 실험(Table 2)은 Meta-Dataset의 서브셋을 활용하여 검증
- **차이점**: Meta-Dataset은 벤치마크 제안에 집중, Bayesian TAML은 불균형 대응 메커니즘 제안

#### ■ CNAPS / Simple CNAPS (Requeima et al. NeurIPS 2019; Bateni et al. CVPR 2020)
- 조건부 신경 어댑티브 프로세스를 통한 태스크 적응
- **Bayesian TAML과 유사점**: 태스크별 파라미터 생성
- **차이점**: 클래스/태스크 불균형을 명시적으로 다루지 않음

#### ■ BOHB / Hyperparameter Optimization in Meta-Learning
- 베이지안 하이퍼파라미터 최적화와 메타러닝의 결합 연구들이 Bayesian TAML의 베이지안 균형 변수 학습 아이디어를 확장

#### ■ 대조 비교표 (2020년 이후)

| 논문/방법 | 불균형 처리 | OOD 처리 | 베이지안 | 주요 접근 |
|-----------|:-----------:|:--------:|:--------:|-----------|
| **Bayesian TAML** | ✅ (명시적) | ✅ | ✅ | 균형 변수 학습 |
| ANIL (Raghu et al., ICLR 2020) | ❌ | ❌ | ❌ | 피처 재사용 |
| BOIL (Oh et al., ICLR 2021) | ❌ | △ | ❌ | 바디 온리 내부 루프 |
| UniSiam (Lu et al., CVPR 2022) | △ | △ | ❌ | 대조 학습 결합 |

> 위 표는 일부 정보가 불완전할 수 있으며, 각 논문을 직접 확인하시길 강력히 권장합니다.

---

### 4.3 앞으로 연구 시 고려할 점

#### ① 더 강력한 백본과의 결합 검증
Bayesian TAML은 4-block Conv에서만 검증되었습니다. **Vision Transformer(ViT)**, **ResNet-12** 등 최신 백본에서의 성능 및 계산 비용 분석이 필요합니다.

#### ② 균형 변수 간 상관관계 모델링
현재 완전 인수분해 가정을 사용합니다:
$$q(\boldsymbol{\phi}^{\tau} | \mathcal{D}^{\tau}; \psi) = \prod_c q(\tilde{\omega}_c^{\tau}) \prod_l q(\tilde{\gamma}_l^{\tau}) \prod_i q(\tilde{z}_i^{\tau})$$

**$\boldsymbol{\omega}^{\tau}$, $\boldsymbol{\gamma}^{\tau}$, $\mathbf{z}^{\tau}$** 간 실제 상관관계를 포착하는 **정규화 흐름(Normalizing Flow)** 또는 **full-covariance 가우시안** 적용이 성능 향상 가능성이 있습니다.

#### ③ 연속 학습(Continual Learning)으로의 확장
태스크가 순차적으로 도달하는 continual learning 설정에서 Bayesian TAML의 균형 메커니즘을 활용하면 **망각(catastrophic forgetting)** 방지에 도움이 될 수 있습니다.

#### ④ 대형 언어 모델(LLM)에서의 적용 가능성
Few-shot prompting 상황에서 태스크 불균형(각기 다른 few-shot 예제 수)이 발생합니다. LLM의 in-context learning에 Bayesian TAML의 균형 개념을 적용하는 연구 가능성이 있습니다.

#### ⑤ 공정성(Fairness)과의 연계
클래스 불균형 처리 메커니즘($\boldsymbol{\omega}^{\tau}$)은 알고리즘 공정성 연구에서의 소수 집단 보호와 개념적으로 연결됩니다. 의료 데이터, 금융 데이터 등 민감한 도메인에서의 적용 연구가 의미 있습니다.

#### ⑥ 계산 효율화
메타테스트 시 $S=10$ MC 샘플이 필요한 점은 실시간 응용의 병목입니다. **Amortized Inference** 또는 **결정론적 근사** 방법으로 계산 비용을 줄이면서 MC 근사의 이점을 보존하는 연구가 필요합니다.

---

## 참고 자료

**주요 참고 논문 (PDF에서 직접 인용됨):**

1. **Lee et al. (2020)** - "Learning to Balance: Bayesian Meta-Learning for Imbalanced and Out-of-distribution Tasks", ICLR 2020. arXiv:1905.12917v3
2. **Finn et al. (2017)** - "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks", ICML 2017
3. **Li et al. (2017)** - "Meta-SGD: Learning to Learn Quickly for Few-Shot Learning", arXiv:1707.09835
4. **Snell et al. (2017)** - "Prototypical Networks for Few-Shot Learning", NeurIPS 2017
5. **Finn et al. (2018)** - "Probabilistic Model-Agnostic Meta-Learning", NeurIPS 2018
6. **Yoon et al. (2018)** - "Bayesian Model-Agnostic Meta-Learning", NeurIPS 2018
7. **Gordon et al. (2018)** - "Meta-Learning Probabilistic Inference for Prediction", ICLR 2018
8. **Triantafillou et al. (2019)** - "Meta-Dataset: A Dataset of Datasets for Learning to Learn from Few Examples", ICLR 2020
9. **Kingma & Welling (2013)** - "Auto-Encoding Variational Bayes", arXiv:1312.6114
10. **Zaheer et al. (2017)** - "Deep Sets", NeurIPS 2017
11. **Oreshkin et al. (2018)** - "TADAM: Task Dependent Adaptive Metric for Improved Few-Shot Learning", NeurIPS 2018

**코드 저장소**: https://github.com/haebeom-lee/l2b
