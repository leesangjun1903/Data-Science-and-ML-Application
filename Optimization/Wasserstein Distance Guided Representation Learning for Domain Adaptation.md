# Wasserstein Distance Guided Representation Learning for Domain Adaptation (WDGRL)

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장
WDGRL은 도메인 적응(Domain Adaptation)에서 소스 도메인과 타겟 도메인 간의 분포 불일치(domain discrepancy)를 줄이기 위해, 기존의 GAN 기반 도메인 분류기(cross-entropy 손실) 대신 **Wasserstein Distance**를 활용하면 더 안정적인 그래디언트와 더 나은 일반화 성능을 달성할 수 있다고 주장합니다.

### 주요 기여

| 기여 항목 | 설명 |
|-----------|------|
| **새로운 방법론 제안** | Wasserstein Distance를 도메인 불변 표현 학습에 적용한 WDGRL 제안 |
| **이론적 증명** | K-Lipschitz 가정 하에 Wasserstein distance 기반 일반화 경계(Generalization Bound) 증명 |
| **그래디언트 우월성 증명** | 기존 adversarial 방법(DANN)의 그래디언트 소실 문제 해결 |
| **실증적 우수성** | 감정 분류, 이미지 분류 등 다수 벤치마크에서 SOTA 달성 |
| **모듈화 가능성** | 기존 도메인 적응 프레임워크(DDC, DAN, DSN 등)에 플러그인 형태로 통합 가능 |

---

## 2. 상세 분석

### 2-1. 해결하고자 하는 문제

**비지도 도메인 적응(Unsupervised Domain Adaptation)** 환경에서:
- 소스 도메인: 레이블이 있는 데이터 $X^s = \{(x_i^s, y_i^s)\}_{i=1}^{n_s}$
- 타겟 도메인: 레이블이 없는 데이터 $X^t = \{x_j^t\}_{j=1}^{n_t}$
- 두 도메인은 동일한 특징 공간을 공유하지만, 서로 다른 주변 분포 $P_{x^s} \neq P_{x^t}$를 가짐

**기존 방법의 문제점:**
1. **MMD**: 분포 간 평균 임베딩 거리만 측정 → 고차 통계 무시
2. **DANN (Gradient Reversal Layer)**: 도메인 분류기가 두 분포를 완벽히 구분할 때 **그래디언트 소실(gradient vanishing)** 발생

특히 그래디언트 소실 문제는 최적 도메인 분류기가 다음과 같을 때:

$$\sigma(f_d^*(h)) = \frac{p(h)}{p(h) + q(h)}$$

소스 데이터가 분포 밀도가 높은 region A에만 존재하고 타겟 데이터가 없는 경우, 그래디언트는 다음과 같이 거의 0이 됩니다:

$$\frac{\partial \mathcal{L}_D}{\partial f_d} = y - \sigma(f_d(f_g)) \approx 0$$

---

### 2-2. 제안하는 방법 (수식 포함)

#### (1) Wasserstein Distance 정의

$p$-번째 Wasserstein Distance는 다음과 같이 정의됩니다:

$$W_p(\mathbb{P}, \mathbb{Q}) = \left(\inf_{\mu \in \Gamma(\mathbb{P}, \mathbb{Q})} \int \rho(x, y)^p \, d\mu(x, y)\right)^{1/p}$$

Kantorovich-Rubinstein 쌍대 표현(1차 Wasserstein, Earth-Mover Distance):

$$W_1(\mathbb{P}, \mathbb{Q}) = \sup_{\|f\|_L \leq 1} \mathbb{E}_{x \sim \mathbb{P}}[f(x)] - \mathbb{E}_{x \sim \mathbb{Q}}[f(x)]$$

여기서 $\|f\|_L = \sup |f(x) - f(y)| / \rho(x, y)$ 는 Lipschitz semi-norm입니다.

#### (2) Domain Critic을 통한 Wasserstein Distance 추정

특징 추출기 $f_g: \mathbb{R}^m \rightarrow \mathbb{R}^d$ (파라미터 $\theta_g$)와 도메인 크리틱 $f_w: \mathbb{R}^d \rightarrow \mathbb{R}$ (파라미터 $\theta_w$)를 이용하여:

$$W_1(\mathbb{P}_{h^s}, \mathbb{P}_{h^t}) = \sup_{\|f_w\|_L \leq 1} \mathbb{E}_{\mathbb{P}_{x^s}}[f_w(f_g(x))] - \mathbb{E}_{\mathbb{P}_{x^t}}[f_w(f_g(x))]$$

경험적 Wasserstein Distance 손실:

$$\mathcal{L}_{wd}(x^s, x^t) = \frac{1}{n^s}\sum_{x^s \in X^s} f_w(f_g(x^s)) - \frac{1}{n^t}\sum_{x^t \in X^t} f_w(f_g(x^t))$$

#### (3) Lipschitz 제약: Gradient Penalty

가중치 클리핑 대신 Gradient Penalty (Gulrajani et al., 2017의 WGAN-GP 방식 채택):

$$\mathcal{L}_{grad}(\hat{h}) = \left(\|\nabla_{\hat{h}} f_w(\hat{h})\|_2 - 1\right)^2$$

여기서 $\hat{h}$는 소스와 타겟 표현 쌍 사이의 직선 위 임의의 점들을 포함합니다.

#### (4) 도메인 크리틱 최적화

$$\max_{\theta_w} \{\mathcal{L}_{wd} - \gamma \mathcal{L}_{grad}\}$$

#### (5) 특징 추출기 최적화 (Minimax)

$$\min_{\theta_g} \max_{\theta_w} \{\mathcal{L}_{wd} - \gamma \mathcal{L}_{grad}\}$$

#### (6) 분류기(Discriminator)와 결합한 최종 목적함수

분류 손실(크로스 엔트로피):

$$\mathcal{L}_c(x^s, y^s) = -\frac{1}{n_s}\sum_{i=1}^{n_s}\sum_{k=1}^{l} \mathbf{1}(y_i^s = k) \cdot \log f_c(f_g(x_i^s))_k$$

최종 목적함수:


```math
\min_{\theta_g, \theta_c} \left\{ \mathcal{L}_c + \lambda \max_{\theta_w} \left[\mathcal{L}_{wd} - \gamma \mathcal{L}_{grad}\right] \right\}
```

- $\lambda$: 판별성과 전달성 간 균형 계수
- $\gamma$: gradient penalty 균형 계수 (최소화 단계에서는 0으로 설정)

---

### 2-3. 모델 구조

```
[소스 데이터 / 타겟 데이터]
         ↓
  [Feature Extractor: f_g(θ_g)]
    ↙              ↘
[Domain Critic: f_w(θ_w)]  [Discriminator: f_c(θ_c)]
[Wasserstein Distance 추정] [Classification Loss]
    ↓                         ↓
[adversarial 방식으로 θ_g 업데이트]
```

**학습 알고리즘 (Algorithm 1):**
1. 미니배치 샘플링 (소스 + 타겟, 각 32개)
2. **내부 루프 (n=5회):** 도메인 크리틱을 Gradient Ascent로 최적화
   - $\theta_w \leftarrow \theta_w + \alpha_1 \nabla_{\theta_w}[\mathcal{L}\_{wd}(x^s, x^t) - \gamma \mathcal{L}_{grad}(\hat{h})]$
3. **외부 업데이트:** 분류기와 특징 추출기를 Gradient Descent로 업데이트
   - $\theta_c \leftarrow \theta_c - \alpha_2 \nabla_{\theta_c} \mathcal{L}_c(x^s, y^s)$
   - $\theta_g \leftarrow \theta_g - \alpha_2 \nabla_{\theta_g}[\mathcal{L}\_c(x^s, y^s) + \mathcal{L}_{wd}(x^s, x^t)]$

---

### 2-4. 성능 향상

#### Amazon Review 데이터셋 (12개 적응 태스크, 정확도 %)

| 방법 | AVG |
|------|-----|
| S-only | 77.84 |
| MMD | 81.22 |
| DANN | 80.74 |
| CORAL | 82.16 |
| **WDGRL** | **82.43** |

- 12개 태스크 중 10개에서 1위

#### Office-Caltech 데이터셋 (Decaf features, 12개 적응 태스크, 정확도 %)

| 방법 | AVG |
|------|-----|
| S-only | 85.44 |
| MMD | 92.03 |
| DANN | 87.67 |
| CORAL | 90.76 |
| **WDGRL** | **92.74** |

#### Email Spam 데이터셋 평균: **89.90%** (DANN 86.98%, MMD 87.00% 대비 우세)

#### 20 Newsgroup 데이터셋 평균: **95.77%** (DANN 95.47%, MMD 94.11% 대비 우세)

---

### 2-5. 한계점

1. **계산 비용**: 도메인 크리틱을 반복적으로 최적화해야 하므로 MMD나 CORAL보다 계산 비용이 높음
2. **하이퍼파라미터 민감성**: $\lambda$, $\gamma$, critic training step $n$, 학습률 등 여러 하이퍼파라미터 튜닝 필요
3. **단일 레이어 적용**: 논문에서는 마지막 은닉층 하나에만 WDGRL 적용 (DAN처럼 다층 적용 가능하지만 실험에서는 생략)
4. **이미지 데이터 한계**: 이미지에 직접 적용하는 정교한 아키텍처(CNN 등) 실험 미포함
5. **레이블 불균형 고려 미흡**: 클래스 조건부 분포 정렬(class-conditional alignment)이 없어 클래스별 정렬이 부족할 수 있음

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 이론적 일반화 경계

**Lemma 1.** $\mu_s, \mu_t \in \mathcal{P}(\mathcal{X})$가 두 확률 측도이고, 가설 클래스 $\mathcal{H}$의 모든 가설 $h$가 $K$-Lipschitz 연속이라 가정할 때:

$$\epsilon_t(h, h') \leq \epsilon_s(h, h') + 2K W_1(\mu_s, \mu_t)$$

**Theorem 1.** (타겟 오류 상한) 모든 $h \in \mathcal{H}$에 대해:

$$\epsilon_t(h) \leq \epsilon_s(h) + 2K W_1(\mu_s, \mu_t) + \lambda$$

여기서 $\lambda = \min_{h \in \mathcal{H}}[\epsilon_s(h) + \epsilon_t(h)]$는 이상적 가설의 결합 오류입니다.

**증명 핵심:**

```math
\epsilon_t(h) \leq \epsilon_t(h^*) + \epsilon_t(h^*, h) \leq \epsilon_t(h^*) + \epsilon_s(h, h^*) + 2KW_1(\mu_s, \mu_t) \leq \epsilon_s(h) + 2KW_1(\mu_s, \mu_t) + \lambda
```

**Theorem 3.** (경험적 측도에 대한 일반화 경계) 확률 $1-\delta$ 이상으로:

$$\epsilon_t(h) \leq \epsilon_s(h) + 2K W_1(\hat{\mu}_s, \hat{\mu}_t) + \lambda + 2K\sqrt{\frac{2\log(1/\delta)}{\lambda'}} \left(\sqrt{\frac{1}{N_s}} + \sqrt{\frac{1}{N_t}}\right)$$

### 3-2. 일반화 성능 향상에 기여하는 요소

#### (A) 그래디언트 안정성 → 더 나은 특징 학습
Wasserstein loss의 그래디언트:

$$\frac{\partial \mathcal{L}_W}{\partial \theta_g}: \quad \frac{\partial \mathcal{L}_W}{\partial f_w} = \begin{cases} +1 & x \sim \mathbb{P}_{x^s} \\ -1 & x \sim \mathbb{P}_{x^t} \end{cases}$$

분포가 어디에 있든 항상 안정적인 그래디언트 제공 → **더 완전한 분포 정렬** → 일반화 향상

#### (B) 경험적 Wasserstein Distance 최소화
타겟 오류 상한의 핵심 항 $2K W_1(\hat{\mu}_s, \hat{\mu}_t)$를 직접 최소화함으로써, 이론적으로 보장된 방식으로 타겟 도메인 오류를 줄임

#### (C) Lipschitz 연속성 가정의 현실성
신경망에서 선형 변환과 sigmoid, ReLU 등 활성화 함수는 모두 Lipschitz 연속이며, 정규화를 통해 $K$를 제한 가능 → 가정이 실제로 충족 가능

---

## 4. 앞으로의 연구에 미치는 영향과 고려 사항

### 4-1. 연구에 미치는 영향

#### (A) 최적 수송 이론의 도메인 적응 적용 촉진
WDGRL은 Wasserstein/Optimal Transport 이론을 딥러닝 기반 도메인 적응에 체계적으로 도입한 선구적 연구로서, 이후 수많은 연구의 기반이 됨

#### (B) 이론-실증 일치의 모범 사례
단순 경험적 성능 비교를 넘어 일반화 경계를 수학적으로 증명하여, **이론적으로 뒷받침된 방법론 설계**의 중요성을 강조

#### (C) 모듈형 설계 패러다임
WDGRL이 기존 프레임워크(DDC, DAN, DSN 등)에 플러그인 형태로 통합 가능하다는 설계 철학은 이후 **컴포넌트 기반 도메인 적응 연구**에 영향을 줌

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의**: 아래 비교는 제가 학습한 지식 범위 내에서 서술하며, 2020년 이후 논문의 세부 수치는 직접 검증이 어려울 수 있습니다. 확실한 정보만 기술합니다.

### 5-1. WDGRL과 후속 연구 비교

#### (1) Conditional Domain Adversarial Networks (CDAN, Long et al., 2018, NeurIPS)

- **핵심 아이디어**: 도메인 판별기의 입력으로 특징과 분류기 예측의 **multilinear conditioning**을 사용
- **WDGRL과의 차이**: WDGRL은 주변 분포(marginal distribution)만 정렬하는 반면, CDAN은 조건부 분포(conditional distribution, 클래스별)까지 정렬
- **개선 방향**: WDGRL에 클래스 조건부 Wasserstein Distance를 결합하면 더 강력한 정렬 가능

#### (2) Sliced Wasserstein Distance for Domain Adaptation (2019-2020년대 다수 연구)

- **핵심 아이디어**: 고차원에서 Wasserstein Distance 계산 비용을 줄이기 위해 1차원 투영 후 평균하는 **Sliced Wasserstein Distance** 활용

```math
SW_p(\mathbb{P}, \mathbb{Q}) = \int_{\mathbb{S}^{d-1}} W_p(\theta_\# \mathbb{P}, \theta_\# \mathbb{Q}) \, d\sigma(\theta)
```

- **WDGRL 대비 장점**: 계산 효율성 향상

#### (3) Optimal Transport for Domain Adaptation with Label Information (2020 이후)

- **핵심 아이디어**: 레이블 정보를 활용한 클래스 조건부 최적 수송

$$W_c(\mathbb{P}_s, \mathbb{P}_t) = \sum_k \pi_k W_1(\mathbb{P}_s^{(k)}, \mathbb{P}_t^{(k)})$$

- **WDGRL 대비 개선**: 클래스별 분포 정렬로 더 세밀한 도메인 정렬

#### (4) Domain Adaptation with Pre-trained Transformers (2021-2023, BERT/ViT 기반)

- **핵심 변화**: WDGRL이 MLP/CNN 기반 특징 추출기를 사용한 반면, 최신 연구들은 **사전학습된 트랜스포머 모델**을 특징 추출기로 사용
- WDGRL의 도메인 크리틱 메커니즘을 트랜스포머 기반으로 확장하는 연구들이 등장

### 5-2. 비교 요약 테이블

| 특성 | WDGRL (2018) | CDAN (2018) | Sliced-W (2020+) | Transformer DA (2021+) |
|------|-------------|-------------|-----------------|----------------------|
| 거리 척도 | $W_1$ | Cross-entropy + conditioning | Sliced $W_p$ | 다양 |
| 분포 정렬 | 주변 분포 | 조건부 분포 | 주변 분포 | 주변/조건부 |
| 계산 비용 | 중간 | 중간 | 낮음 | 높음 (사전학습 포함) |
| 이론적 보장 | ✅ (일반화 경계 증명) | 부분적 | ✅ | 제한적 |
| 클래스 정보 활용 | 제한적 | ✅ | 제한적 | ✅ |

---

## 6. 앞으로 연구 시 고려할 점

### 6-1. 방법론적 개선 방향

#### (A) 클래스 조건부 Wasserstein Distance 도입
현재 WDGRL은 주변 분포만 정렬. 클래스별 분포 정렬:

$$\mathcal{L}_{wd}^{class} = \sum_{k=1}^{K} W_1(\mathbb{P}_{h^s|y=k}, \mathbb{P}_{h^t|y=k})$$

를 추가하면 **클래스 구분력 유지**와 **도메인 불변성**을 동시에 달성 가능

#### (B) 다중 소스 도메인 확장
현재는 단일 소스-타겟 쌍만 고려. 다중 소스 도메인:

$$\mathcal{L}_{total} = \mathcal{L}_c + \lambda \sum_{i=1}^{M} W_1(\mathbb{P}_{h^{s_i}}, \mathbb{P}_{h^t})$$

#### (C) 타겟 도메인 레이블 활용 (Semi-supervised 설정)
소량의 타겟 레이블이 있는 semi-supervised 환경에서의 WDGRL 확장

#### (D) 계산 효율화
- Sliced Wasserstein Distance 활용으로 critic 학습 반복 횟수 줄이기
- Neural OT(Optimal Transport) 기법과 결합

### 6-2. 이론적 발전 방향

#### (A) 더 타이트한 일반화 경계
현재 경계: $\epsilon_t(h) \leq \epsilon_s(h) + 2K W_1(\mu_s, \mu_t) + \lambda$

- $\lambda$ 항(이상적 가설의 결합 오류)을 제어하는 방법 연구
- 고차 Wasserstein distance ($W_2$)를 활용한 경계 개선

#### (B) 레이블 시프트(Label Shift) 처리
현재 WDGRL은 공변량 시프트(covariate shift)만 가정. 레이블 분포가 다른 경우:
$$P_s(y) \neq P_t(y)$$
에 대한 이론적 확장 필요

### 6-3. 응용 도메인 확장 시 고려사항

| 응용 분야 | 고려할 점 |
|----------|----------|
| **자연어처리 (NLP)** | 이산적 텍스트 특징에서 Wasserstein 계산의 적절성, 사전학습 LM과의 통합 |
| **의료 이미징** | 소규모 데이터셋에서의 도메인 크리틱 과적합 방지 |
| **자율주행** | 실시간 추론을 위한 계산 비용 최적화 |
| **시계열 데이터** | 시간적 의존성을 고려한 Wasserstein 거리 확장 (e.g., Temporal WD) |

---

## 참고 자료

**주요 참고 논문 (논문 내 인용 기준):**

1. **Shen, J., Qu, Y., Zhang, W., & Yu, Y. (2018).** "Wasserstein Distance Guided Representation Learning for Domain Adaptation." *AAAI 2018.* arXiv:1707.01217v4

2. **Arjovsky, M., Chintala, S., & Bottou, L. (2017).** "Wasserstein GAN." arXiv:1701.07875

3. **Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., & Courville, A. (2017).** "Improved Training of Wasserstein GANs." arXiv:1704.00028

4. **Ganin, Y., et al. (2016).** "Domain-Adversarial Training of Neural Networks." *JMLR.*

5. **Redko, I., Habrard, A., & Sebban, M. (2016).** "Theoretical Analysis of Domain Adaptation with Optimal Transport." arXiv:1610.04420

6. **Ben-David, S., et al. (2007).** "Analysis of Representations for Domain Adaptation." *NIPS.*

7. **Bolley, F., Guillin, A., & Villani, C. (2007).** "Quantitative Concentration Inequalities for Empirical Measures on Non-compact Spaces." *Probability Theory and Related Fields.*

8. **Villani, C. (2008).** "Optimal Transport: Old and New."

9. **Long, M., et al. (2015).** "Learning Transferable Features with Deep Adaptation Networks." *ICML.* (DAN)

10. **Tzeng, E., et al. (2017).** "Adversarial Discriminative Domain Adaptation." arXiv:1702.05464
