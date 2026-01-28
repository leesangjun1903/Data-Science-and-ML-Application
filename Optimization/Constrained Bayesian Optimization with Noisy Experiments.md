
# Constrained Bayesian Optimization with Noisy Experiments

## 1. 논문의 핵심 주장과 주요 기여

### 1.1 핵심 주장

Letham et al. (2018)의 "Constrained Bayesian Optimization with Noisy Experiments"는 **고노이즈 환경에서의 베이지안 최적화(BO) 성능 저하 문제**를 해결합니다. 논문의 핵심 주장은 다음과 같습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)

기존 베이지안 최적화 방법은 A/B 테스트와 같은 실제 실험 환경에서 **세 가지 중요한 한계**를 가집니다: (1) 측정 노이즈가 매우 높을 때 성능이 급격히 저하되고, (2) 제약 조건(제약 함수의 노이즈)을 효과적으로 처리하지 못하며, (3) 배치 최적화에서 노이즈를 제대로 다루지 못합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)

### 1.2 주요 기여

논문의 주요 기여는 **Noisy Expected Improvement (NEI) 획득 함수**와 **준몬테카를로(QMC) 근사**입니다.

| 기여 항목 | 설명 |
|----------|------|
| **NEI (Noisy Expected Improvement)** | 고노이즈 관찰 및 노이즈가 있는 제약 조건을 정확히 통합하는 새로운 획득 함수 |
| **QMC 적분 기법** | NEI의 고차원 적분을 효율적으로 계산하기 위한 준몬테카를로 방법 |
| **배치 최적화 확장** | 순차적 및 비동기 배치 평가를 지원하는 통합 프레임워크 |
| **실제 적용 사례** | Facebook의 순위 시스템 및 HHVM 컴파일러 최적화 문제에 성공적으로 적용 |

***

## 2. 해결하고자 하는 문제, 제안하는 방법 및 모델 구조

### 2.1 문제 정의

베이지안 최적화 문제는 다음과 같이 정의됩니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)

$$\min_{\mathbf{x}} f(\mathbf{x}) \quad \text{subject to} \quad c_j(\mathbf{x}) \leq 0, \quad j = 1, \ldots, J$$

여기서 $f(\mathbf{x})$는 목표 함수, $c_j(\mathbf{x})$는 제약 함수이며, 모두 블랙박스 함수이고 노이즈가 있는 관찰만 가능합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)

$$y_i = f(\mathbf{x}_i) + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \tau_i^2)$$

### 2.2 기존 방법의 한계

#### 2.2.1 노이즈가 있는 기대개선(EI) 처리

기존 접근 방식은 **플러그인(plug-in) 휴리스틱**을 사용하여 비관찰 가능한 최적값 $f^\*$을 기울기 평균 $\hat{g}^* = \min_\mathbf{x} \mu_f(\mathbf{x})$로 대체합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)

$`\alpha_{EI}(\mathbf{x}|f^*) = \sigma_f(\mathbf{x})z\Phi(z) + \sigma_f(\mathbf{x})\phi(z), \quad z = \frac{f^* - \mu_f(\mathbf{x})}{\sigma_f(\mathbf{x})}`$

그러나 이 방법은 높은 노이즈 환경에서 수렴이 느리고 후보점 간의 군집화(clustering)를 유발합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)

#### 2.2.2 제약 조건 처리의 문제

Schonlau et al. (1998)의 제약 기대개선(CEI): [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)

$`\alpha_{EIC}(\mathbf{x}|f_c^*) = \alpha_{EI}(\mathbf{x}|f_c^*)\prod_{j=1}^J P(c_j(\mathbf{x}) \leq 0)`$

이 방법은 노이즈가 있는 제약을 다루기 어렵고, 실행 가능한 최적값 $f_c^*$을 설정하기 위한 명확한 원칙이 없습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)

### 2.3 제안하는 방법: 노이즈 기대개선 (NEI)

#### 2.3.1 유틸리티 함수 정의

논문은 먼저 명확한 **유틸리티 함수**를 정의합니다. 실행 가능한 관찰 집합을 $S = \{i : c_j(\mathbf{x}_i) \leq 0 \forall j\}$라 하면: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)

$$u(n) = \begin{cases} -\min_{i \in S} f(\mathbf{x}_i) & \text{if } |S| > 0 \\ -M & \text{otherwise} \end{cases}$$

여기서 $M$은 실행 가능한 해가 없을 때의 페널티입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)

#### 2.3.2 노이즈 없는 설정의 개선도

개선도 함수는: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)

$$I(\mathbf{x}) = \begin{cases} 0 & \text{if } \mathbf{x} \text{ is infeasible} \\ M - f(\mathbf{x}) & \text{if } \mathbf{x} \text{ is feasible and } |S_n| = 0 \\ \max(0, f_c^* - f(\mathbf{x})) & \text{if } \mathbf{x} \text{ is feasible and } |S_n| > 0 \end{cases}$$

이를 통해 제약을 고려한 기대개선은: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)

$$\alpha_{EI_\mathbf{x}}(\mathbf{x}|f^n, c^n) = \begin{cases} \alpha_{EI}(\mathbf{x}|f_c^*)\prod_{j=1}^J \Phi\left(-\frac{\mu_{c_j}(\mathbf{x})}{\sigma_{c_j}(\mathbf{x})}\right) & \text{if } |S_n| > 0 \\ (M - \mu_f(\mathbf{x}))\prod_{j=1}^J \Phi\left(-\frac{\mu_{c_j}(\mathbf{x})}{\sigma_{c_j}(\mathbf{x})}\right) & \text{otherwise} \end{cases}$$

#### 2.3.3 노이즈가 있는 설정의 NEI

핵심 혁신은 **노이즈가 있는 관찰의 실제 함수값에 대한 사후 분포**를 명시적으로 통합하는 것입니다. $\mathcal{D}\_f$와 $\mathcal{D}_{c_j}$가 노이즈가 있는 관찰이면, 진정한 함수값의 사후 분포는: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)

$$f^n|\mathcal{D}_f \sim \mathcal{N}(\mu_f, \Sigma_f)$$

$$c_j^n|\mathcal{D}\_{c_j} \sim \mathcal{N}(\mu_{c_j}, \Sigma_{c_j})$$

**노이즈 기대개선(NEI)**은 이 불확실성에 대한 기대값입니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)

$$\alpha_{NEI}(\mathbf{x}|\mathcal{D}) = \int_{\mathbb{R}^n} \int_{\mathbb{R}^{Jn}} \alpha_{EI_\mathbf{x}}(\mathbf{x}|f^n, c^n) p(f^n|\mathcal{D}_f) \prod_{j=1}^J p(c_j^n|\mathcal{D}_{c_j}) \, dc^n \, df^n$$

이 공식은 **heuristics 없이** 노이즈를 정확히 다룹니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)

### 2.4 준몬테카를로 (QMC) 근사

#### 2.4.1 QMC 방법의 원리

NEI 적분은 해석적 형태가 없으므로 수치 적분이 필요합니다. 표준 몬테카를로(MC)는 $O(1/\sqrt{N})$ 수렴률을 가지지만, **준몬테카를로(QMC) 방법**은 저이산 수열(low-discrepancy sequence)을 사용하여 $O((\log N)^d/N)$ 수렴률을 달성합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)

| 방법 | 수렴률 | 특징 |
|------|------|------|
| 몬테카를로 (MC) | $O(1/\sqrt{N})$ | 무작위 샘플링, 느린 수렴 |
| 준몬테카를로 (QMC) | $O((\log N)^d/N)$ | 공간 채우기 수열, 빠른 수렴 |

#### 2.4.2 정규 분포에 대한 변환

Proposition 1 (Dick et al., 2013)에 따르면, $A$를 $\Sigma = AA^\top$를 만족하는 행렬(예: 촐레스키 분해)이라 하면: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)

$$\int_{\mathbb{R}^d} f(\mathbf{y})p(\mathbf{y}|\mu, \Sigma)d\mathbf{y} = \int_{ [)^d} f(A\Phi^{-1}(\mathbf{u}) + \mu)d\mathbf{u}$$

#### 2.4.3 NEI 적분의 QMC 근사

대각 분산 $\Sigma = \text{diag}(\Sigma_f, \Sigma_{c_1}, \ldots, \Sigma_{c_J})$와 $\mu = [\mu_f, \mu_{c_1}, \ldots, \mu_{c_J}]$에 대해: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)

$$\alpha_{NEI}(\mathbf{x}|\mathcal{D}) = \int_{ )^{n(J+1)}} \alpha_{EI_\mathbf{x}}(\mathbf{x}|\tilde{f}^n(\mathbf{u}), \tilde{c}^n(\mathbf{u})) d\mathbf{u}$$

여기서 $[\tilde{f}^n(\mathbf{u}), \tilde{c}^n(\mathbf{u})]^\top = A\Phi^{-1}(\mathbf{u}) + \mu$

QMC 근사는: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)

$$\alpha_{NEI}(\mathbf{x}|\mathcal{D}) \approx \frac{1}{N}\sum_{k=1}^N \alpha_{EI_\mathbf{x}}(\mathbf{x}|\tilde{f}^n(\mathbf{t}_k), \tilde{c}^n(\mathbf{t}_k))$$

여기서 $\{\mathbf{t}_1, \ldots, \mathbf{t}_N\}$은 **Sobol 수열** 같은 저이산 수열입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)

### 2.5 알고리즘: QMC 통합을 사용한 노이즈 EI

**알고리즘 1**의 주요 단계: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)

| 단계 | 작업 |
|------|------|
| 1 | 목표 및 제약 함수에 대한 GP 커널 초매개변수 추론 |
| 2 | 관찰 지점에서 목표/제약의 사후 분포 계산: $f^n\|\mathcal{D}_f \sim \mathcal{N}(\mu_f, \Sigma_f)$ |
| 3 | 대각 분산 행렬과 평균 벡터 구성 |
| 4 | Cholesky 분해 $\Sigma = AA^\top$ 계산 |
| 5 | Sobol 수열로부터 QMC 샘플 생성: $\mathbf{t}_1, \ldots, \mathbf{t}_N$ |
| 6-8 | 각 샘플에 대해 노이즈 없는 GP 모델 구성 |
| 9-13 | NEI 값 계산: 각 샘플의 EI를 평균화 |

**핵심 특징**: 1-8단계는 후보점 $\mathbf{x}$와 독립적이므로 최적화 시작 전에 캐시되어 효율성을 높입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)

### 2.6 배치 및 비동기 최적화 확장

NEI는 **배치 최적화**로 자연스럽게 확장됩니다. 대기 중인 $m$개 점의 미관측 결과를 $\mathbf{f}^b = [f(\mathbf{x}_1^b), \ldots, f(\mathbf{x}_m^b)]$이라 하면: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)

$`\alpha_{EIB}(\mathbf{x}|f^*) = \int_{\mathbb{R}^m} \alpha_{EI}(\mathbf{x}|\min(f^*, \mathbf{f}^b))p(\mathbf{f}^b|\mathcal{D}_f)d\mathbf{f}^b`$

NEI는 이를 $[\mathbf{f}^n, \mathbf{f}^b]$에 대한 적분으로 확장하며, QMC 근사가 효율적으로 작동합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)

***

## 3. 모델 성능 향상 및 일반화 가능성

### 3.1 합성 함수 실험 결과

논문은 **4가지 표준 벤치마크 문제**에서 광범위한 실험을 수행합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)

| 문제 | 차원 | 제약 수 | 특징 |
|------|------|--------|------|
| Gramacy | 2 | 2 | 비선형 제약 |
| Hartmann 6 | 6 | 1 | 고차원 |
| Branin | 2 | 1 | 표준 테스트 함수 |
| Gardner | 2 | 1 | 강하게 상관된 제약 |

#### 3.1.1 QMC 효율성

Figure 2의 결과는 QMC가 동일한 적분 오차를 달성하기 위해 **MC보다 절반의 샘플만 필요**함을 보여줍니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)
- MC: 50개 샘플 필요 → 20% 거리 오차
- QMC: 25개 샘플로 동일 성능

최적화 성능 측면에서 **QMC는 16개 샘플로 MC의 50개 샘플과 동등**한 결과를 달성합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)

#### 3.1.2 NEI vs 다른 방법 비교

**Figure 3**은 4가지 방법의 비교 결과입니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)

| 방법 | Gramacy | Hartmann6 | Branin | Gardner |
|------|---------|-----------|--------|---------|
| NEI | ✓ 최고 성능 | ✓ 최고 성능 | ✓ 최고 성능 | ≈ PESC와 동등 |
| EI+휴리스틱 | 차선 | 차선 | 차선 | 차선 |
| Spearmint EI | 저조 | 저조 | 저조 | 저조 |
| PESC | 저조 | 저조 | 저조 | ≈ NEI |

**주요 결과**: NEI는 노이즈가 있는 제약 조건이 있는 모든 문제에서 **기존 방법을 일관되게 능가**합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)

### 3.2 모델 식별 성능

Figure 4는 최종 식별된 최적점의 질을 비교합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)

$$\text{식별된 최적성} = (B - \mu_f(\mathbf{x}))\prod_{j=1}^J P(c_j(\mathbf{x}) \leq 0)$$

| 측정 항목 | NEI | EI+휴리스틱 |
|----------|-----|-----------|
| 목표 함수값 | -2.52 | -1.15 |
| 실행 가능 확률 | 0.95+ | 0.88 |
| 최종 반복에서의 개선 | 우월 | 비교 대상 |

NEI는 **더 나은 목표 함수값을 가진 실행 가능한 점을 식별**합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)

### 3.3 일반화 성능 향상

#### 3.3.1 높은 노이즈에서의 강건성

Figure S7 (보충 자료)은 EI+휴리스틱이 고노이즈 환경에서 **점들의 군집화(clustering)**를 보이는 반면, NEI는 **최적값 주변의 적절한 탐색**을 유지함을 보여줍니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)

#### 3.3.2 제약 조건 불확실성 처리

표준 편차가 신호 크기와 비슷한 환경에서:
- Gramacy: 노이즈 표준편차 = 0.5, 신호 범위 = [0.6, 1.0]
- Hartmann6: 노이즈 표준편차 = 0.2, 신호 범위 = [-3.0, -0.5]

NEI는 노이즈가 있는 제약을 **정확히 모델링**하여 실행 가능성 추정을 개선합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)

#### 3.3.3 배치 최적화 효율성

**배치 크기 = 5인 경우**:
- 초기화: 50개 점의 준무작위 수열
- 최적화: 9개 배치 × 5 = 45개 NEI 제안

NEI는 **배치당 더 나은 점**을 제안하여 더 빠른 수렴을 달성합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)

***

## 4. 실제 응용: Facebook 사례 연구

### 4.1 순위 시스템 최적화

#### 4.1.1 문제 설정

Facebook의 컨텐츠 순위 시스템을 최적화합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)
- **차원**: 6개의 인덱서 파라미터
- **목표**: 사용자 참여도 최대화
- **제약**: 컨텐츠 품질 메트릭 최소값 유지

#### 4.1.2 결과

$$\text{상관계수}(\text{목표}, \text{제약}) = -0.78$$

**Figure 5** 결과: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)
- 초기화: 31개 준무작위 점
- NEI 배치: 3개 점만으로도 기저선과 초기화 배치 **모두를 능가**
- 실행 가능한 구성: 기저선 대비 **유의미한 개선 달성**

### 4.2 HHVM 컴파일러 최적화

#### 4.2.1 문제 설정

Facebook의 PHP/Hack 런타임 성능 개선: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)
- **차원**: 7개 컴파일러 플래그 (일부 정수값)
  - 함수 인라인 임계값
  - 코드 레이아웃 핫-콜드 임계값
  - 기타 JIT 최적화 파라미터
- **목표**: CPU 시간 최소화
- **제약**: 피크 메모리 사용량 증가 제한

#### 4.2.2 결과

$$\text{상관계수}(\text{CPU 시간}, \text{메모리}) = 0.21$$

**Figure 6** 결과: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)

| 단계 | CPU 시간 (상대값) | 실행 가능 확률 |
|------|------------------|--------------|
| 준무작위 초기화 (30개) | baseline | 0.77 |
| NEI 배치 (70개 추가) | -2.5% ~ -5% | **0.89** |
| 최적 구성 (실험 83) | **최대 감소** | 제약 만족 |

**주요 성과**: NEI 제안점의 **거의 모든 점이 기저선 대비 CPU 시간을 개선**하고, 실행 가능성 확률이 0.77에서 0.89로 향상됩니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)

***

## 5. 한계와 앞으로의 연구 방향

### 5.1 논문의 한계

#### 5.1.1 레플리케이션 불가능

NEI 획득 함수는 이미 관찰된 점에서 0의 값을 가지므로 **레플리케이션(점 반복 평가)을 권장하지 않습니다**. 이는: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)
- 연속 공간에서는 인접한 점을 샘플링하여 보완 가능
- 하지만 이산 문제에서는 제한이 될 수 있음 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)

Binois et al. (2017)은 특정 상황에서 레플리케이션이 예측 분산을 감소시킬 수 있음을 보였습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)

#### 5.1.2 비실행 가능 지역의 정보 가치

NEI는 **근시적(myopic) 유틸리티 함수**를 기반하므로 비실행 가능 지역의 정보 가치를 인식하지 못합니다. 제약 경계 근처의 탐색을 제한할 수 있습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)

#### 5.1.3 독립성 가정

모델은 **목표와 제약의 독립성**, 그리고 **제약 간 독립성**을 가정합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)

수정 방안:
- 제약 간 상관성: 멀티태스크 GP 사용 가능
- 목표-제약 상관성: 해석적 EI 형태를 잃을 수 있음 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)

#### 5.1.4 높은 차원에서의 확장성

NEI 적분의 차원은 $n(J+1)$ (관찰점 수 × (1 + 제약 수))입니다. 많은 관찰과 제약이 있을 때 저이산성 가정(effective dimensionality)이 깨질 수 있습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)

### 5.2 향후 연구 방향

#### 5.2.1 대안적 적분 방법

Chevalier & Ginsbourger (2013), Marmin et al. (2016)의 **절단 다변량 정규 분포** 공식이 MC/QMC 대안이 될 수 있습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)

#### 5.2.2 레플리케이션 전략

Jalali et al. (2017)의 **통합 기대 조건부 개선(Integrated Expected Conditional Improvement)** 같은 비근시적 방법 통합. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)

#### 5.2.3 다중 목표 확장

다중 목표 문제에 대한 NEI 일반화:
- 기대 하이퍼볼륨 개선
- 파레토 프론트 탐색 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)

#### 5.2.4 하이브리드 모델

- 신경망 대리 모델 결합
- 물리 기반 제약 통합
- 전이 학습 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)

***

## 6. 2020년 이후 관련 최신 연구 비교 분석

### 6.1 제약 기반 지식 기울기 방법 (2021)

**Chen et al. (2021)**: "A New Knowledge Gradient-based Method for Constrained Bayesian Optimization" [semanticscholar](https://www.semanticscholar.org/paper/17c0feceac3252b7c88b0de257a7cc1ec764fde1)

| 항목 | NEI (2018) | c-KG (2021) |
|------|-----------|-----------|
| 획득 함수 | 기대개선 기반 | 지식 기울기 기반 |
| 배치 최적화 | 지원 | 지원 |
| 노이즈 처리 | 통합 근거 | 휴리스틱 |
| 계산 복잡도 | 낮음 | 중간 |
| 이론적 보장 | 수렴 | 수렴 보장 증명 |

**주요 차이**: c-KG는 한 단계 베이즈 최적성을 보장하지만 계산이 더 복잡합니다. [semanticscholar](https://www.semanticscholar.org/paper/17c0feceac3252b7c88b0de257a7cc1ec764fde1)

### 6.2 제약 기대개선의 수렴률 분석 (2025)

**Wang et al. (2025)**: "Convergence Rates of Constrained Expected Improvement" [arxiv](https://arxiv.org/abs/2505.11323)

논문은 **제약 기대개선(CEI)의 첫 번째 수렴 속도 분석**을 제공합니다: [arxiv](https://arxiv.org/abs/2505.11323)

$$\text{간단한 리그릿} = \begin{cases} \mathcal{O}(t^{-\frac{1}{2}}\log^{\frac{d+1}{2}}(t)) & \text{SE 커널} \\ \mathcal{O}(t^{-\frac{\nu}{2\nu+d}}\log^{\frac{\nu}{2\nu+d}}(t)) & \text{Matérn 커널} \end{cases}$$

**시사점**: NEI는 경험적 우월성이 있으나 이론적 수렴 속도 분석이 부족합니다. [arxiv](https://arxiv.org/abs/2505.11323)

### 6.3 두 단계 선행 제약 최적화 (2024)

**Zhang et al. (2024)**: "Two-step Lookahead Constrained Bayesian Optimization" [ecommons.cornell](https://ecommons.cornell.edu/items/18b8f45c-c754-4d34-bec6-cde0e1c682de)

| 항목 | NEI (2018) | 2-OPT-C (2024) |
|------|-----------|----------------|
| 최적성 | 1단계 근시적 | 2단계 선행 |
| 계산 비용 | 낮음 | 중간 |
| 제약 경계 탐색 | 제한적 | 우월 |
| 쿼리 효율성 | baseline | **2-10배 개선** |

**혁신**: 가능-불가능 경계 근처 탐색을 **2단계 전략**으로 개선합니다. [ecommons.cornell](https://ecommons.cornell.edu/items/18b8f45c-c754-4d34-bec6-cde0e1c682de)

### 6.4 다목적 제약 최적화 (2024)

**Low et al. (2024)**: "Evolution-guided Bayesian optimization for constrained..." [nature](https://www.nature.com/articles/s41524-024-01274-x)

다목적 제약 문제(cMOOP)에서 진화 알고리즘과 BO 결합: [nature](https://www.nature.com/articles/s41524-024-01274-x)

$$\text{qNEHVI} + \text{NSGA-III} \rightarrow \text{EGBO}$$

**결과**: 제약 위반 28% 감소, 파레토 프론트 집중도 개선. [nature](https://www.nature.com/articles/s41524-024-01274-x)

### 6.5 정보 이론 기반 방법 (2025)

**여러 연구**: 정보 이론 기반 획득 함수 재조명 [arxiv](https://arxiv.org/pdf/2502.06789.pdf)

| 방법 | 특징 | 노이즈 성능 |
|------|------|-----------|
| **Entropy Search (ES)** | KL 발산 최소화 | 중간 |
| **Predictive Entropy Search (PES)** | 최적위 불확실성 감소 | 약함 |
| **Max-value Entropy Search (MES)** | 저계산 비용 | 우월 |
| **Alpha Entropy Search (AES)** | α-발산 유연성 | 우월 (2025) |

Fernández-Sánchez et al. (2025)의 **Alpha Entropy Search**는 α-발산을 사용하여 KL 기반 방법보다 더 유연합니다. [sciencedirect](https://www.sciencedirect.com/science/article/pii/S0950705125006586)

### 6.6 Thompson Sampling 기반 방법 (2020-2025)

**Kandasamy et al. (2018), Zerefa et al. (2024)**: [arxiv](https://arxiv.org/abs/1705.09236)

| 항목 | EI 기반 (NEI) | Thompson Sampling |
|------|--------------|------------------|
| 병렬화 용이성 | 명시적 처리 | 자연스러움 |
| 높은 노이즈 환경 | 우월 | 비교 가능 |
| 계산 효율 | 높음 | 중간-높음 |
| 이론적 분석 | 경험적 | 강한 이론적 보장 |

Thompson Sampling은 **강한 이론적 보장**을 제공하며, 고노이즈 환경에서도 경쟁력 있는 성능을 보입니다. [arxiv](https://arxiv.org/abs/1705.09236)

### 6.7 하이브리드 모델 기반 제약 최적화 (2023)

**Lu & Paulson (2023)**: "No-Regret Constrained BO with Hybrid Models" [arxiv](https://www.arxiv.org/abs/2305.03824)

**제약 상한 정량 경계(CUQB)** 방법:
- 백박스 + 블랙박스 함수의 복합 구조 활용
- 미분 가능 샘플 평균 근사로 효율적 최적화
- 누적 리그릿 경계: 부분선형 의존성 [arxiv](https://www.arxiv.org/abs/2305.03824)

### 6.8 비동기 및 분산 제약 최적화 (2024-2025)

**Zerefa et al. (2024)**, **Huynh et al. (2025)**: [arxiv](https://arxiv.org/abs/2410.15543)

**pc-BO-TS** (Process-Constrained BO via Thompson Sampling):
- 다중 반응기 시스템 최적화
- 계층적 제약 처리
- 비동기 배치 평가 [arxiv](https://arxiv.org/pdf/2408.02551.pdf)

**qNEHVI-BO** (다목적 병렬 제약 BO):
- 진화 알고리즘과 결합
- 제약 위반 28% 감소 [nature](https://www.nature.com/articles/s41524-024-01274-x)

### 6.9 준몬테카를로의 최신 발전 (2024-2025)

**Liu et al. (2021, 2024), Bartuska et al. (2024)**: [jmlr](https://www.jmlr.org/papers/volume22/21-0498/21-0498.pdf)

QMC 방법의 개선:
- **Quasi-Newton + QMC**: 가변 베이즈 추론에서 수렴 가속 [jmlr](https://www.jmlr.org/papers/volume22/21-0498/21-0498.pdf)
- **QMC for PDE**: 고차원 매개변수에서 차원-무결성 수렴률 [arxiv](https://arxiv.org/html/2405.03529v2)
- **이중 루프 QMC**: 베이지안 역설계 문제에 최적화 [arxiv](https://arxiv.org/html/2405.03529v2)

**주요 발견**: QMC의 $O((\log N)^d/N)$ 비율이 정규성 조건 하에서 유지되며, NEI의 "유효 차원성" 가정이 실제로 더 많은 문제에서 유효합니다. [arxiv](https://arxiv.org/html/2405.03529v2)

***

## 7. 논문의 미래 영향과 앞으로의 고려사항

### 7.1 학술적 영향

#### 7.1.1 인용도 및 확산

논문은 **469회 인용**(Google Scholar 기준)으로 이 분야의 **핵심 기여**로 확립되었습니다. 특히: [arxiv](https://arxiv.org/abs/1706.07094)
- **산업 응용**: Facebook(Meta)의 실제 시스템에 통합 [ai.meta](https://ai.meta.com/research/publications/constrained-bayesian-optimization-with-noisy-experiments/)
- **오픈소스**: BoTorch 등 주요 라이브러리에 구현 [arxiv](https://arxiv.org/pdf/1910.06403.pdf)
- **교육**: 베이지안 최적화 튜토리얼의 표준 사례 [arxiv](https://arxiv.org/pdf/1807.02811.pdf)

#### 7.1.2 이론적 기반

논문은 **의사결정 이론과 유틸리티 최대화**를 NEI 설계의 명시적 원리로 제시하여, 이후 방법들의 이론적 정당화 근거를 제공합니다. 특히: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)
- 유틸리티 기반 프레임워크로 휴리스틱 제거
- QMC의 효율성을 베이지안 설정에 처음 적용
- 노이즈와 제약의 통합 처리 원칙 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)

### 7.2 산업적 영향

#### 7.2.1 A/B 테스팅 패러다임 전환

NEI의 성공은 **대규모 기술 기업들에서 고노이즈 최적화 사용**을 정당화했습니다: [ai.meta](https://ai.meta.com/research/publications/constrained-bayesian-optimization-with-noisy-experiments/)
- Facebook: 수십 개의 실제 A/B 테스트 운영
- 동적 가격 책정, 추천 시스템, 컴파일러 튜닝 등

#### 7.2.2 의사결정 지원 도구

네이티브 제약 처리로 인해 실무자들이 **다중 목표 trade-off**를 명시적으로 모델링 가능: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)
- 성능 지표 vs 리소스 사용
- 사용자 경험 vs 시스템 안정성

### 7.3 앞으로의 핵심 연구 방향

#### 7.3.1 확장성 및 고차원

**현재 병목**: NEI 적분의 차원 = $n(J+1)$ (관찰 수 × 제약 수) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)

**미래 방향**:
1. **차원 감소**: 활성 부공간(active subspace) 학습
2. **구조 활용**: 제약 간 의존성 명시적 모델링
3. **스파스 그리드**: 고차원 적분을 위한 차원 수정(dimension-folding) QMC

#### 7.3.2 이론과 실무의 격차 해소

**현재**: NEI는 경험적으로 우수하나 수렴 속도 증명 부족 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)

**필요한 분석**:
- RKHS(재현 커널 힐버트 공간) 설정에서 수렴률
- 노이즈 수준에 따른 샘플 복잡도
- 정규화 조건 하에서의 최적성 갭 [arxiv](https://arxiv.org/abs/2505.11323)

#### 7.3.3 적응형 알고리즘

**근시적 한계** 극복: [ecommons.cornell](https://ecommons.cornell.edu/items/18b8f45c-c754-4d34-bec6-cde0e1c682de)
1. **다단계 선행**: 2-OPT-C의 장점 활용
2. **정보 가치 학습**: 비실행 가능 지역의 가치 동적 추정
3. **제약 활성화 학습**: 바인딩 제약에 집중

#### 7.3.4 멀티모달성 및 구조

**복잡한 실문제 특성** 대응: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)
- **다봉우리(multimodal)** 목적 함수
- **불연속 제약 경계**
- **계층적/네스팅 제약**

해결책:
- 혼합 가우스 프로세스
- 분류 기반 부분 공간 분할
- 국소 최적화와 전역 탐색 균형

#### 7.3.5 강화 학습 통합

정책 최적화 문제의 연속 확장: [semanticscholar](https://www.semanticscholar.org/paper/9f5d0f77e9384062c5ce21c9b862649551b12944)
- 맥락 기반 최적화 (contextual optimization)
- 온라인 학습 설정
- 전이/메타 학습

### 7.4 실무 적용 시 고려사항

#### 7.4.1 매개변수 민감도

논문의 주요 하이퍼파라미터: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)
- **QMC 샘플 수 N**: 적분 오차와 계산 비용의 트레이드오프
  - 권장: $N = 16 \sim 50$ (대부분 문제에서)
  - 높은 차원: $N = 100+$
- **페널티 M**: 실행 불가능 비용
  - 기본값: 설계 공간의 최대 예측값보다 큼

#### 7.4.2 GP 모델 선택

논문은 **Matérn 5/2 커널**을 사용하지만, 실무에서는: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)
- **매끄러움**: 문제 도메인의 물리적 이해 필요
- **초매개변수**: NUTS 샘플러로 사후 추론 (계산 비용 ↑)
- **대안**: 최대우도 추정 (속도 ↑, 불확실성 ↓)

#### 7.4.3 노이즈 추정

**핵심 가정**: 진정한 노이즈 분산 $\tau_i^2$ 알려짐 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)
- 실제로는 신뢰 구간에서 추정
- **영향**: 과소/과대 추정은 성능 저하
- **해결**: Empirical Bayes, 음이항 로그우도 최적화

#### 7.4.4 제약 독립성

기존 가정: 목표와 제약, 제약 간 독립성 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)

**실무 개선**:
- 상관성이 있으면 연합 GP 사용
- 공통 커널 구조 공유로 샘플 효율 향상

***

## 결론

"Constrained Bayesian Optimization with Noisy Experiments"는 **의사결정 이론 기반의 유틸리티 최대화 프레임워크**와 **QMC 통합**을 통해 고노이즈 제약 최적화 문제를 해결한 획기적인 논문입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)

**핵심 기여**:
1. 휴리스틱 없이 **노이즈를 정확히 다루는 NEI 공식** 도출 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)
2. QMC를 통한 **효율적인 고차원 적분 근사** (MC 대비 2배 샘플 효율) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)
3. **배치/비동기 최적화로의 자연스러운 확장** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)
4. **Facebook의 실제 시스템에 대한 성공적 적용** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c8e4d9a-0fa4-41ac-aac8-a6e97af072d9/1706.07094v2.pdf)

최근 연구(2020-2025)는 이를 발전시켜 **선행 전략**, **정보 이론 방법**, **다목적 확장**, **강한 이론적 보장** 등을 추가했습니다. 그러나 NEI의 **단순성과 실무 적용성**은 여전히 최고의 강점이며, 고노이즈 환경의 제약 최적화에서 표준 기준으로 자리 잡았습니다. [ai.meta](https://ai.meta.com/research/publications/constrained-bayesian-optimization-with-noisy-experiments/)

***

## 참고문헌

<span style="display:none">[^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58][^1_59][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_70][^1_71][^1_72][^1_73][^1_74][^1_75][^1_76][^1_77][^1_78][^1_79][^1_80][^1_81][^1_82][^1_83][^1_84][^1_85][^1_86][^1_87][^1_88][^1_89][^1_90][^1_91][^1_92][^1_93][^1_94][^1_95][^1_96][^1_97][^1_98][^1_99]</span>

<div align="center">⁂</div>

[^1_1]: 1706.07094v2.pdf

[^1_2]: https://www.semanticscholar.org/paper/17c0feceac3252b7c88b0de257a7cc1ec764fde1

[^1_3]: https://arxiv.org/abs/2505.11323

[^1_4]: https://ecommons.cornell.edu/items/18b8f45c-c754-4d34-bec6-cde0e1c682de

[^1_5]: https://www.nature.com/articles/s41524-024-01274-x

[^1_6]: https://arxiv.org/pdf/2502.06789.pdf

[^1_7]: https://www.sciencedirect.com/science/article/pii/S0950705125006586

[^1_8]: https://arxiv.org/abs/1705.09236

[^1_9]: https://arxiv.org/abs/2410.15543

[^1_10]: https://arxiv.org/html/2410.15543v1

[^1_11]: https://www.arxiv.org/abs/2305.03824

[^1_12]: https://www.sciencedirect.com/science/article/pii/S004578252500338X

[^1_13]: https://arxiv.org/pdf/2408.02551.pdf

[^1_14]: https://www.jmlr.org/papers/volume22/21-0498/21-0498.pdf

[^1_15]: https://arxiv.org/html/2405.03529v2

[^1_16]: https://proceedings.mlr.press/v80/buchholz18a/buchholz18a.pdf

[^1_17]: https://arxiv.org/abs/1706.07094

[^1_18]: https://ai.meta.com/research/publications/constrained-bayesian-optimization-with-noisy-experiments/

[^1_19]: https://arxiv.org/pdf/1910.06403.pdf

[^1_20]: https://arxiv.org/pdf/1807.02811.pdf

[^1_21]: https://www.semanticscholar.org/paper/9f5d0f77e9384062c5ce21c9b862649551b12944

[^1_22]: https://www.semanticscholar.org/paper/0e5ceae89bdd57a1f7fbfa94fcc97f93535f583a

[^1_23]: https://arxiv.org/abs/2105.13245

[^1_24]: https://www.semanticscholar.org/paper/252e762236617ae5e431b9dc079240b4ec03644c

[^1_25]: https://www.semanticscholar.org/paper/e8bdbd64ad40a80b8c4ad75cedd36c062aa7ffc2

[^1_26]: https://ieeexplore.ieee.org/document/9340602/

[^1_27]: https://link.springer.com/10.1007/s00034-020-01529-0

[^1_28]: https://www.semanticscholar.org/paper/8d54ad2add5eeb77ea7ebec4b6c1c73b33c65179

[^1_29]: https://www.cambridge.org/core/product/identifier/S2632673620000106/type/journal_article

[^1_30]: https://www.tandfonline.com/doi/full/10.1080/17517575.2020.1717002

[^1_31]: https://ieeexplore.ieee.org/document/9206396/

[^1_32]: http://arxiv.org/pdf/2403.03816.pdf

[^1_33]: https://arxiv.org/pdf/1706.07094.pdf

[^1_34]: https://arxiv.org/pdf/2202.07549.pdf

[^1_35]: http://arxiv.org/pdf/1803.08661.pdf

[^1_36]: https://arxiv.org/pdf/1506.01349.pdf

[^1_37]: https://projecteuclid.org/journals/bayesian-analysis/volume-14/issue-2/Constrained-Bayesian-Optimization-with-Noisy-Experiments/10.1214/18-BA1110.pdf

[^1_38]: https://documentserver.uhasselt.be/bitstream/1942/45255/1/Constrained Bayesian Optimization_%20A%20Review.pdf

[^1_39]: https://dl.acm.org/doi/10.5555/3540261.3540429

[^1_40]: https://repository.tilburguniversity.edu/bitstreams/85a67fa4-5336-4996-996c-093bd948a1d3/download

[^1_41]: https://d-nb.info/1182864236/34

[^1_42]: https://botorch.org/docs/next/constraints/

[^1_43]: https://www.themoonlight.io/ko/review/quasi-monte-carlo-for-bayesian-design-of-experiment-problems-governed-by-parametric-pdes

[^1_44]: https://arxiv.org/pdf/2106.10600.pdf

[^1_45]: https://arxiv.org/pdf/2308.13222.pdf

[^1_46]: https://arxiv.org/html/2411.03641v1

[^1_47]: https://ar5iv.labs.arxiv.org/html/1807.01604

[^1_48]: https://arxiv.org/list/math/new

[^1_49]: https://arxiv.org/html/2505.11323v1

[^1_50]: https://arxiv.org/pdf/2403.11374.pdf

[^1_51]: https://arxiv.org/list/physics/new

[^1_52]: https://arxiv.org/html/2512.17569v1

[^1_53]: https://arxiv.org/pdf/2104.02865.pdf

[^1_54]: https://arxiv.org/pdf/2506.04769.pdf

[^1_55]: https://arxiv.org/abs/2112.02833

[^1_56]: https://arxiv.org/html/2502.03644v1

[^1_57]: https://www.auai.org/uai2014/proceedings/individuals/107.pdf

[^1_58]: https://link.springer.com/10.1007/s42824-025-00198-1

[^1_59]: https://www.ewadirect.com/proceedings/ace/article/view/27247

[^1_60]: http://medrxiv.org/lookup/doi/10.1101/2025.10.25.25338798

[^1_61]: https://www.taylorfrancis.com/books/9781003226277

[^1_62]: https://onepetro.org/JPT/article/77/09/1/789468/Mobile-Platform-Enables-AI-Powered-Underwater

[^1_63]: https://ieeexplore.ieee.org/document/11202603/

[^1_64]: https://www.worldscientific.com/doi/10.1142/S1752890926500030

[^1_65]: https://www.semanticscholar.org/paper/3a225a64335a75f6ac5c15cc058a35a521dc105d

[^1_66]: http://ieeexplore.ieee.org/document/4761074/

[^1_67]: https://arxiv.org/pdf/2101.08743.pdf

[^1_68]: http://arxiv.org/pdf/2504.07742.pdf

[^1_69]: https://arxiv.org/pdf/2205.11568.pdf

[^1_70]: https://arxiv.org/abs/2209.15367

[^1_71]: http://arxiv.org/pdf/2310.03912.pdf

[^1_72]: http://arxiv.org/pdf/1606.04624.pdf

[^1_73]: https://par.nsf.gov/servlets/purl/10582429

[^1_74]: https://proceedings.mlr.press/v161/nguyen21d/nguyen21d.pdf

[^1_75]: http://www.arxiv.org/abs/2101.08743

[^1_76]: https://www.arxiv.org/pdf/2411.17071.pdf

[^1_77]: https://proceedings.neurips.cc/paper_files/paper/2015/file/57c0531e13f40b91b3b0f1a30b529a1d-Paper.pdf

[^1_78]: https://www.semanticscholar.org/paper/A-New-Knowledge-Gradient-based-Method-for-Bayesian-Chen-Liu/17c0feceac3252b7c88b0de257a7cc1ec764fde1

[^1_79]: https://proceedings.neurips.cc/paper/2020/file/6dfe08eda761bd321f8a9b239f6f4ec3-Paper.pdf

[^1_80]: http://arxiv.org/pdf/2002.02820.pdf

[^1_81]: https://people.orie.cornell.edu/jdai/thesis/JianWuThesis.pdf

[^1_82]: https://num.pyro.ai/en/0.15.3/examples/thompson_sampling.html

[^1_83]: http://proceedings.mlr.press/v108/frohlich20a/frohlich20a.pdf

[^1_84]: https://warwick.ac.uk/fac/sci/mathsys/people/students/mathsysii/tokaeva/ma932_report_1.pdf

[^1_85]: https://gdmarmerola.github.io/ts-for-bayesian-optim/

[^1_86]: https://www.themoonlight.io/en/review/alpha-entropy-search-for-new-information-based-bayesian-optimization

[^1_87]: https://arxiv.org/pdf/2508.02426.pdf

[^1_88]: https://arxiv.org/pdf/2411.17071.pdf

[^1_89]: https://arxiv.org/pdf/2504.07952.pdf

[^1_90]: https://arxiv.org/pdf/2510.10984.pdf

[^1_91]: https://arxiv.org/pdf/2506.21900.pdf

[^1_92]: https://arxiv.org/pdf/2511.06790.pdf

[^1_93]: https://arxiv.org/pdf/2508.10949.pdf

[^1_94]: https://arxiv.org/pdf/2406.07413.pdf

[^1_95]: https://arxiv.org/pdf/2509.00496.pdf

[^1_96]: https://arxiv.org/pdf/2601.10583.pdf

[^1_97]: https://arxiv.org/pdf/2508.02621.pdf

[^1_98]: http://proceedings.mlr.press/v84/kandasamy18a/kandasamy18a.pdf

[^1_99]: https://www.research-collection.ethz.ch/entities/publication/cbc79b4d-2f77-4cbc-89ef-f0c009c1466a

 Low et al. (2024). "Evolution-guided Bayesian optimization for constrained..." Nature 2024 [nature](https://www.nature.com/articles/s41524-024-01274-x)

 Fernández-Sánchez et al. (2025). "Alpha entropy search for new information-based Bayesian optimization." Neurocomputing [sciencedirect](https://www.sciencedirect.com/science/article/pii/S0950705125006586)

 Huynh et al. (2025). "Parallel constrained Bayesian optimization via batched Thompson sampling..." Computers & Operations Research [sciencedirect](https://www.sciencedirect.com/science/article/pii/S004578252500338X)
